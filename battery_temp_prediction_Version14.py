"""
============================================================
  锂电池内部温度预测 - PINN-LSTM 完整项目 (v14)

  基于 v13 的修改内容 (v14)：
  1. 热参数初始值设为目标值：Cc=87.7, Cs=4.15, Rc=2.51, Rs=8.47
  2. 添加热参数正则化损失，将参数拉向目标值（归一化 L2）
  3. 物理参数学习率从 1e-2 降低至 5e-3
  4. 热参数单独设置更小的学习率 THERMAL_LR=1e-3
  5. Phase2 总损失加入热参数正则项
  6. 每 epoch 结束后对热参数进行 clamp（软约束）

  v13 原有修复内容：
  1. Phase1期间禁用早停，确保物理约束能生效
  2. 加入L2正则化防止Phase1过拟合
  3. 缩短Phase1(10epoch)，让物理约束更早介入
  4. 增大总epoch和早停耐心，给Phase2充分时间
  5. Phase2开始时重置早停计数器和学习率调度器

  物理模型：Bernardi 产热 + 二节点集总热模型
  可训练参数：Cc, Cs, Rc, Rs, OCV多项式, dOCV/dT多项式
  对比模型：纯LSTM / PINN-LSTM / GRU
============================================================
"""

import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================
#  1. 全局配置
# ============================================================

DATA_DIR = r'E:\code\zhuyaoshuju'
FILE_A = 'Battery_A_1-300_with_SOH_SOC_base5max_raw_clip.csv'
FILE_B = 'Battery_B_1-300_with_SOH_SOC_base5max_raw_clip.csv'
PATH_A = os.path.join(DATA_DIR, FILE_A)
PATH_B = os.path.join(DATA_DIR, FILE_B)

OUTPUT_DIR = r'E:\code\battery_temp_prediction\output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEMPORAL_FEATURES = ['voltage_v', 'soc', 'temp_surface_c', 'current_a',
                     'dv_dt', 'dts_dt', 'temp_rise']
STATIC_FEATURES = ['soh']
TARGET = 'temp_core_c'
CYCLE_COL = 'cycle_global'

DISCHARGE_LABEL = None
DISCHARGE_CURRENT_THRESHOLD = -0.5

T_AMBIENT = 25.0

TRAIN_CYCLES = 30
VAL_CYCLES = 8
TEST_CYCLES = 20
INTRA_CYCLE_DOWNSAMPLE = 1

WINDOW_SIZE = 60
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
SOH_EMBED_DIM = 32
OCV_POLY_ORDER = 6
DOCV_DT_POLY_ORDER = 4

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
PHYSICS_LR = 5e-3             # ★ v14: 从 1e-2 降低到 5e-3
WEIGHT_DECAY = 1e-4           # ★ 修复: L2正则化防止过拟合
EPOCHS = 200                  # ★ 修复: 从150增加到200
EARLY_STOP_PATIENCE = 30      # ★ 修复: 从20增加到30
LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.5
PHASE1_EPOCHS = 10            # ★ 修复: 从30缩短到10，更早引入物理约束
SEED = 42

# 目标热参数 (来自文献/实验标定)
TARGET_Cc = 87.7              # J/K  核心热容
TARGET_Cs = 4.15              # J/K  表面热容
TARGET_Rc = 2.51              # K/W  核心→表面热阻
TARGET_Rs = 8.47              # K/W  表面→环境热阻
THERMAL_LR = 1e-3             # ★ v14: 热参数专用学习率
THERMAL_REG_LAMBDA = 0.001    # ★ v14: 热参数正则化系数

MODEL_CN_NAMES = {
    'base_lstm':  '基线LSTM',
    'gru':        'GRU模型',
    'pinn_lstm':  'PINN-LSTM',
}


def get_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f"[设备] GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device('cpu')
        print("[设备] CPU")
    return dev


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
#  2. OCV 和 dOCV/dT 参考数据 (Samsung INR18650-30Q)
# ============================================================

OCV_SOC_REF = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
OCV_V_REF   = np.array([3.20, 3.45, 3.55, 3.62, 3.67, 3.72, 3.76, 3.80, 3.84, 3.92, 4.15])

DOCV_SOC_REF = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
DOCV_DT_REF  = np.array([0.0001, 0.00005, 0.0, -0.00005, -0.0001, -0.00015,
                          -0.00018, -0.0002, -0.00022, -0.00025, -0.00025])


def fit_ocv_polynomial():
    coeffs = np.polyfit(OCV_SOC_REF, OCV_V_REF, OCV_POLY_ORDER)
    return coeffs[::-1].astype(np.float32)


def fit_docv_dt_polynomial():
    coeffs = np.polyfit(DOCV_SOC_REF, DOCV_DT_REF, DOCV_DT_POLY_ORDER)
    return coeffs[::-1].astype(np.float32)


# ============================================================
#  3. 数据加载与采样
# ============================================================

def load_raw_data():
    print("\n[1/7] 加载数据 ...")
    if not os.path.exists(PATH_A):
        raise FileNotFoundError(f"找不到文件: {PATH_A}")
    if not os.path.exists(PATH_B):
        raise FileNotFoundError(f"找不到文件: {PATH_B}")
    df_a = pd.read_csv(PATH_A)
    df_b = pd.read_csv(PATH_B)
    print(f"  Battery A: {df_a.shape[0]:,} 行, {df_a.shape[1]} 列")
    print(f"  Battery B: {df_b.shape[0]:,} 行, {df_b.shape[1]} 列")
    return df_a, df_b


def filter_discharge(df, name=''):
    n0 = len(df)
    if DISCHARGE_LABEL is not None and 'step_type' in df.columns:
        df = df[df['step_type'] == DISCHARGE_LABEL].copy()
    elif 'current_a' in df.columns:
        df = df[df['current_a'] < DISCHARGE_CURRENT_THRESHOLD].copy()
    else:
        print(f"  [警告] {name}: 无法筛选放电数据")
        return df
    print(f"  {name} 放电筛选: {n0:,} → {len(df):,} 行")
    return df


def inspect_columns(df, name=''):
    print(f"\n  --- {name} 数据概览 ---")
    print(f"  列名: {list(df.columns)}")
    for col in ['voltage_v', 'current_a', 'soc', 'soh',
                'temp_surface_c', 'temp_core_c', CYCLE_COL, 'time_s']:
        if col in df.columns:
            print(f"    {col}: [{df[col].min():.4f} ~ {df[col].max():.4f}], "
                  f"缺失={df[col].isnull().sum()}")
        else:
            print(f"    {col}: *** 不存在 *** ← 请检查列名")


def stratified_cycle_sampling(df, n_cycles, seed=SEED):
    rng = np.random.RandomState(seed)
    cycle_soh = df.groupby(CYCLE_COL)['soh'].mean().reset_index()
    cycle_soh.columns = [CYCLE_COL, 'mean_soh']
    bins = [0.0, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 1.01]
    cycle_soh['soh_bin'] = pd.cut(cycle_soh['mean_soh'], bins=bins)
    selected = []
    for _, group in cycle_soh.groupby('soh_bin', observed=True):
        if len(group) == 0:
            continue
        n_from = max(1, round(n_cycles * len(group) / len(cycle_soh)))
        n_from = min(n_from, len(group))
        chosen = group.sample(n=n_from, random_state=rng)
        selected.extend(chosen[CYCLE_COL].tolist())
    remaining = list(set(cycle_soh[CYCLE_COL]) - set(selected))
    rng.shuffle(remaining)
    while len(selected) < n_cycles and remaining:
        selected.append(remaining.pop())
    if len(selected) > n_cycles:
        selected = rng.choice(selected, size=n_cycles, replace=False).tolist()
    return sorted(selected)


def downsample_within_cycle(df, factor=INTRA_CYCLE_DOWNSAMPLE):
    if factor <= 1:
        return df
    result = []
    for _, grp in df.groupby(CYCLE_COL):
        grp = grp.sort_values('time_s').reset_index(drop=True)
        result.append(grp.iloc[::factor])
    df_out = pd.concat(result, ignore_index=True)
    print(f"  循环内{factor}倍降采样: {len(df):,} → {len(df_out):,} 行")
    return df_out


def sample_and_split(df_a, df_b):
    print("\n[2/7] 分层采样 ...")
    total_a = TRAIN_CYCLES + VAL_CYCLES
    cycles_a = stratified_cycle_sampling(df_a, total_a, seed=SEED)
    rng = np.random.RandomState(SEED + 1)
    rng.shuffle(cycles_a)
    train_cycles = sorted(cycles_a[:TRAIN_CYCLES])
    val_cycles = sorted(cycles_a[TRAIN_CYCLES:])
    test_cycles = stratified_cycle_sampling(df_b, TEST_CYCLES, seed=SEED + 2)

    print(f"  训练循环 ({len(train_cycles)}个): {train_cycles[:5]} ... {train_cycles[-3:]}")
    print(f"  验证循环 ({len(val_cycles)}个):  {val_cycles}")
    print(f"  测试循环 ({len(test_cycles)}个): {test_cycles[:5]} ... {test_cycles[-3:]}")

    df_train = df_a[df_a[CYCLE_COL].isin(train_cycles)].copy()
    df_val   = df_a[df_a[CYCLE_COL].isin(val_cycles)].copy()
    df_test  = df_b[df_b[CYCLE_COL].isin(test_cycles)].copy()

    df_train = downsample_within_cycle(df_train)
    df_val   = downsample_within_cycle(df_val)
    df_test  = downsample_within_cycle(df_test)

    for tag, d in [('训练', df_train), ('验证', df_val), ('测试', df_test)]:
        print(f"  {tag}: {len(d):,} 行, SOH [{d['soh'].min():.4f} ~ {d['soh'].max():.4f}]")
    return df_train, df_val, df_test


# ============================================================
#  4. 特征工程
# ============================================================

def add_derived_features(df):
    df = df.sort_values([CYCLE_COL, 'time_s']).reset_index(drop=True)
    dt = max(1, INTRA_CYCLE_DOWNSAMPLE)
    df['dv_dt']  = df.groupby(CYCLE_COL)['voltage_v'].diff().fillna(0) / dt
    df['dts_dt'] = df.groupby(CYCLE_COL)['temp_surface_c'].diff().fillna(0) / dt
    df['temp_rise'] = df.groupby(CYCLE_COL)['temp_surface_c'].transform(
        lambda x: x - x.iloc[0])
    df['dtc_dt_real'] = df.groupby(CYCLE_COL)['temp_core_c'].diff().fillna(0) / dt
    df['dts_dt_real'] = df.groupby(CYCLE_COL)['temp_surface_c'].diff().fillna(0) / dt
    return df


def clean_data(df, name=''):
    all_cols = TEMPORAL_FEATURES + STATIC_FEATURES + [TARGET]
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} 缺少以下列: {missing}\n"
            f"你的数据列名为: {list(df.columns)}\n"
            f"请修改脚本顶部配置区的列名。")
    n0 = len(df)
    df = df.dropna(subset=all_cols + ['dtc_dt_real', 'dts_dt_real'])
    for col in ['temp_surface_c', 'temp_core_c']:
        if col in df.columns:
            df = df[(df[col] > -10) & (df[col] < 80)]
    if n0 != len(df):
        print(f"  {name} 清洗: {n0:,} → {len(df):,} 行")
    return df.reset_index(drop=True)


class FeatureScaler:
    def __init__(self):
        self.temporal_scaler = MinMaxScaler()
        self.static_scaler  = MinMaxScaler()
        self.target_scaler  = MinMaxScaler()

    def fit_transform(self, df):
        df = df.copy()
        for col in ['voltage_v', 'current_a', 'soc', 'soh', 'temp_surface_c', 'temp_core_c']:
            df[col + '_raw'] = df[col].values
        df[TEMPORAL_FEATURES] = self.temporal_scaler.fit_transform(df[TEMPORAL_FEATURES])
        df[STATIC_FEATURES]   = self.static_scaler.fit_transform(df[STATIC_FEATURES])
        df[[TARGET]]          = self.target_scaler.fit_transform(df[[TARGET]])
        return df

    def transform(self, df):
        df = df.copy()
        for col in ['voltage_v', 'current_a', 'soc', 'soh', 'temp_surface_c', 'temp_core_c']:
            df[col + '_raw'] = df[col].values
        df[TEMPORAL_FEATURES] = self.temporal_scaler.transform(df[TEMPORAL_FEATURES])
        df[STATIC_FEATURES]   = self.static_scaler.transform(df[STATIC_FEATURES])
        df[[TARGET]]          = self.target_scaler.transform(df[[TARGET]])
        return df

    def inverse_target(self, values):
        return self.target_scaler.inverse_transform(values.reshape(-1, 1)).flatten()


def prepare_features(df_train, df_val, df_test):
    print("\n[3/7] 特征工程 ...")
    df_train = add_derived_features(df_train)
    df_val   = add_derived_features(df_val)
    df_test  = add_derived_features(df_test)
    df_train = clean_data(df_train, '训练集')
    df_val   = clean_data(df_val, '验证集')
    df_test  = clean_data(df_test, '测试集')
    scaler = FeatureScaler()
    df_train = scaler.fit_transform(df_train)
    df_val   = scaler.transform(df_val)
    df_test  = scaler.transform(df_test)
    print("  完成")
    return df_train, df_val, df_test, scaler


# ============================================================
#  5. 数据集
# ============================================================

PHYSICS_COLS = ['voltage_v_raw', 'current_a_raw', 'soc_raw', 'soh_raw',
                'temp_surface_c_raw', 'temp_core_c_raw',
                'dtc_dt_real', 'dts_dt_real']


class BatteryDataset(Dataset):
    def __init__(self, X_temporal, X_static, y, physics_data):
        self.X_temporal = torch.FloatTensor(X_temporal)
        self.X_static   = torch.FloatTensor(X_static)
        self.y          = torch.FloatTensor(y)
        self.physics    = torch.FloatTensor(physics_data)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_temporal[idx], self.X_static[idx], self.y[idx], self.physics[idx]


def create_sequences(df, window_size=WINDOW_SIZE):
    X_t, X_s, Y, P, C = [], [], [], [], []
    for cycle, grp in df.groupby(CYCLE_COL):
        grp = grp.sort_values('time_s').reset_index(drop=True)
        if len(grp) <= window_size:
            continue
        tv = grp[TEMPORAL_FEATURES].values
        sv = grp[STATIC_FEATURES].values
        yv = grp[TARGET].values
        pv = grp[PHYSICS_COLS].values
        for i in range(window_size, len(grp)):
            X_t.append(tv[i - window_size:i])
            X_s.append(sv[i])
            Y.append(yv[i])
            P.append(pv[i])
            C.append(cycle)
    X_t = np.array(X_t, dtype=np.float32)
    X_s = np.array(X_s, dtype=np.float32)
    Y   = np.array(Y, dtype=np.float32)
    P   = np.array(P, dtype=np.float32)
    C   = np.array(C)
    print(f"    {len(Y):,} 个样本 (窗口={window_size}, 循环数={len(np.unique(C))})")
    return X_t, X_s, Y, P, C


def build_dataloaders(df_train, df_val, df_test):
    print("\n[4/7] 构造序列 ...")
    print("  训练集:")
    Xt_tr, Xs_tr, y_tr, p_tr, c_tr = create_sequences(df_train)
    print("  验证集:")
    Xt_va, Xs_va, y_va, p_va, c_va = create_sequences(df_val)
    print("  测试集:")
    Xt_te, Xs_te, y_te, p_te, c_te = create_sequences(df_test)

    train_ds = BatteryDataset(Xt_tr, Xs_tr, y_tr, p_tr)
    val_ds   = BatteryDataset(Xt_va, Xs_va, y_va, p_va)
    test_ds  = BatteryDataset(Xt_te, Xs_te, y_te, p_te)

    print(f"\n  训练: {len(train_ds):,}  验证: {len(val_ds):,}  测试: {len(test_ds):,}")

    kw = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **kw)
    return train_ld, val_ld, test_ld, c_te


# ============================================================
#  6. 模型定义
# ============================================================

class TrainablePhysics(nn.Module):
    def __init__(self):
        super().__init__()
        # ★ v14: 初始值直接设为目标热参数，利用 softplus 对大值近似恒等特性
        self.raw_Cc = nn.Parameter(torch.tensor(TARGET_Cc))
        self.raw_Cs = nn.Parameter(torch.tensor(TARGET_Cs))
        self.raw_Rc = nn.Parameter(torch.tensor(TARGET_Rc))
        self.raw_Rs = nn.Parameter(torch.tensor(TARGET_Rs))
        self.ocv_coeffs     = nn.Parameter(torch.tensor(fit_ocv_polynomial()))
        self.docv_dt_coeffs = nn.Parameter(torch.tensor(fit_docv_dt_polynomial()))
        self.log_sigma_data = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_core = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_surf = nn.Parameter(torch.tensor(0.0))

    @property
    def Cc(self):
        return F.softplus(self.raw_Cc)

    @property
    def Cs(self):
        return F.softplus(self.raw_Cs)

    @property
    def Rc(self):
        return F.softplus(self.raw_Rc)

    @property
    def Rs(self):
        return F.softplus(self.raw_Rs)

    def compute_ocv(self, soc):
        result = torch.zeros_like(soc)
        for i, c in enumerate(self.ocv_coeffs):
            result = result + c * soc.pow(i)
        return result

    def compute_docv_dt(self, soc):
        result = torch.zeros_like(soc)
        for i, c in enumerate(self.docv_dt_coeffs):
            result = result + c * soc.pow(i)
        return result

    def compute_qgen(self, voltage, current, soc, tc, ts):
        ocv = self.compute_ocv(soc)
        docv_dt = self.compute_docv_dt(soc)
        t_avg = (tc + ts) / 2.0
        q_irrev = current * (voltage - ocv)
        q_rev   = -current * t_avg * docv_dt
        return q_irrev + q_rev

    def physics_loss(self, voltage, current, soc, tc, ts, dtc_dt, dts_dt):
        q_gen = self.compute_qgen(voltage, current, soc, tc, ts)
        q_cs  = (tc - ts) / self.Rc
        q_sa  = (ts - T_AMBIENT) / self.Rs
        res_core = self.Cc * dtc_dt - q_gen + q_cs
        res_surf = self.Cs * dts_dt - q_cs + q_sa
        return torch.mean(res_core ** 2), torch.mean(res_surf ** 2)

    def adaptive_loss(self, l_data, l_core, l_surf):
        s_d = torch.exp(self.log_sigma_data)
        s_c = torch.exp(self.log_sigma_core)
        s_s = torch.exp(self.log_sigma_surf)
        return (l_data / (2*s_d**2) + self.log_sigma_data +
                l_core / (2*s_c**2) + self.log_sigma_core +
                l_surf / (2*s_s**2) + self.log_sigma_surf)

    def thermal_regularization(self, lambda_reg=THERMAL_REG_LAMBDA):
        """★ v14: 归一化 L2 正则项，将热参数拉向目标值，避免大参数主导"""
        reg = (
            (self.Cc - TARGET_Cc) ** 2 / (TARGET_Cc ** 2) +
            (self.Cs - TARGET_Cs) ** 2 / (TARGET_Cs ** 2) +
            (self.Rc - TARGET_Rc) ** 2 / (TARGET_Rc ** 2) +
            (self.Rs - TARGET_Rs) ** 2 / (TARGET_Rs ** 2)
        )
        return lambda_reg * reg


class BaseLSTM(nn.Module):
    def __init__(self, t_dim, s_dim):
        super().__init__()
        self.lstm = nn.LSTM(t_dim + s_dim, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, dropout=DROPOUT)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(64, 1))

    def forward(self, x_t, x_s):
        x_s_e = x_s.unsqueeze(1).expand(-1, x_t.size(1), -1)
        out, _ = self.lstm(torch.cat([x_t, x_s_e], dim=-1))
        return self.fc(out[:, -1, :]).squeeze(-1)


class SOHAwareLSTM(nn.Module):
    def __init__(self, t_dim, s_dim):
        super().__init__()
        self.lstm = nn.LSTM(t_dim, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, dropout=DROPOUT)
        self.soh_br = nn.Sequential(
            nn.Linear(s_dim, SOH_EMBED_DIM), nn.ReLU(),
            nn.Linear(SOH_EMBED_DIM, SOH_EMBED_DIM), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + SOH_EMBED_DIM, 64), nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x_t, x_s):
        out, _ = self.lstm(x_t)
        h = out[:, -1, :]
        s = self.soh_br(x_s)
        return self.fc(torch.cat([h, s], dim=1)).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, t_dim, s_dim):
        super().__init__()
        self.gru = nn.GRU(t_dim, HIDDEN_SIZE, NUM_LAYERS,
                          batch_first=True, dropout=DROPOUT)
        self.soh_br = nn.Sequential(
            nn.Linear(s_dim, SOH_EMBED_DIM), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + SOH_EMBED_DIM, 64), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(64, 1))

    def forward(self, x_t, x_s):
        out, _ = self.gru(x_t)
        h = out[:, -1, :]
        s = self.soh_br(x_s)
        return self.fc(torch.cat([h, s], dim=1)).squeeze(-1)


class PINNModel(nn.Module):
    def __init__(self, t_dim, s_dim):
        super().__init__()
        self.predictor = SOHAwareLSTM(t_dim, s_dim)
        self.physics   = TrainablePhysics()

    def forward(self, x_t, x_s):
        return self.predictor(x_t, x_s)


MODEL_REGISTRY = {
    'base_lstm': BaseLSTM,
    'gru':       GRUModel,
    'pinn_lstm': PINNModel,
}


def build_model(name, t_dim, s_dim):
    model = MODEL_REGISTRY[name](t_dim, s_dim)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  模型: {MODEL_CN_NAMES.get(name, name)}  参数量: {n_p:,}")
    return model


# ============================================================
#  7. 训练 (★ 核心修复区)
# ============================================================

class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE):
        self.patience = patience
        self.counter  = 0
        self.best     = None
        self.stop     = False

    def step(self, val_loss):
        if self.best is None or val_loss < self.best - 1e-6:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

    def reset(self):
        """★ 修复: Phase2开始时重置早停"""
        self.counter = 0
        self.best = None
        self.stop = False


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, is_pinn):
    model.train()
    sum_loss, sum_data, sum_phys, n_total = 0., 0., 0., 0
    for x_t, x_s, y, phys in loader:
        x_t, x_s, y, phys = x_t.to(device), x_s.to(device), y.to(device), phys.to(device)
        optimizer.zero_grad()
        pred = model(x_t, x_s)
        l_data = criterion(pred, y)
        if is_pinn and epoch >= PHASE1_EPOCHS:
            V, I, soc, tc, ts = phys[:,0], phys[:,1], phys[:,2], phys[:,5], phys[:,4]
            dtc, dts = phys[:,6], phys[:,7]
            l_core, l_surf = model.physics.physics_loss(V, I, soc, tc, ts, dtc, dts)
            adaptive_loss = model.physics.adaptive_loss(l_data, l_core, l_surf)
            thermal_reg = model.physics.thermal_regularization()
            loss = adaptive_loss + thermal_reg
            sum_phys += (l_core.item() + l_surf.item()) * len(y)
        else:
            loss = l_data
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = len(y)
        sum_loss += loss.item() * bs
        sum_data += l_data.item() * bs
        n_total  += bs
    return sum_loss/n_total, sum_data/n_total, sum_phys/n_total


def validate(model, loader, criterion, device):
    model.eval()
    s, n = 0., 0
    with torch.no_grad():
        for x_t, x_s, y, _ in loader:
            x_t, x_s, y = x_t.to(device), x_s.to(device), y.to(device)
            l = criterion(model(x_t, x_s), y)
            s += l.item() * len(y)
            n += len(y)
    return s / n


def train_model(model, train_ld, val_ld, device, model_name='model'):
    cn = MODEL_CN_NAMES.get(model_name, model_name)
    print(f"\n[5/7] 训练: {cn}")
    model = model.to(device)
    is_pinn = isinstance(model, PINNModel)
    criterion = nn.MSELoss()

    # ★ 修复: 加入 weight_decay 正则化
    if is_pinn:
        thermal_params = [model.physics.raw_Cc, model.physics.raw_Cs,
                          model.physics.raw_Rc, model.physics.raw_Rs]
        thermal_ids = {id(p) for p in thermal_params}
        other_physics_params = [p for p in model.physics.parameters()
                                 if id(p) not in thermal_ids]
        optimizer = torch.optim.Adam([
            {'params': list(model.predictor.parameters()), 'lr': LEARNING_RATE,
             'weight_decay': WEIGHT_DECAY},
            {'params': other_physics_params, 'lr': PHYSICS_LR},
            {'params': thermal_params, 'lr': THERMAL_LR},  # ★ v14: 热参数单独小学习率
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                     weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR)
    es = EarlyStopping()

    best_val = float('inf')
    save_path = os.path.join(OUTPUT_DIR, f'{model_name}_best.pth')

    hist = {'train': [], 'val': [], 'data_loss': [], 'phys_loss': []}
    if is_pinn:
        for k in ['Cc', 'Cs', 'Rc', 'Rs']:
            hist[k] = []
        hist['ocv_coeffs']     = []
        hist['docv_dt_coeffs'] = []
        hist['sigma_data']     = []
        hist['sigma_core']     = []
        hist['sigma_surf']     = []

    phase2_started = False
    t0 = time.time()

    for ep in range(EPOCHS):
        # ★ 修复: Phase2开始时重置早停和调度器
        if is_pinn and ep == PHASE1_EPOCHS and not phase2_started:
            phase2_started = True
            es.reset()
            # 重置学习率调度器
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR)
            print(f"\n  ★ Phase 2 开始 (Epoch {ep+1}): 物理约束已启用，早停已重置")

        try:
            tl, dl, pl = train_one_epoch(model, train_ld, optimizer, criterion, device, ep, is_pinn)
        except Exception as e:
            print(f"  [训练异常] Epoch {ep+1}: {e}")
            traceback.print_exc()
            break

        vl = validate(model, val_ld, criterion, device)

        hist['train'].append(tl)
        hist['val'].append(vl)
        hist['data_loss'].append(dl)
        hist['phys_loss'].append(pl)

        if is_pinn:
            p = model.physics
            hist['Cc'].append(p.Cc.item())
            hist['Cs'].append(p.Cs.item())
            hist['Rc'].append(p.Rc.item())
            hist['Rs'].append(p.Rs.item())
            hist['ocv_coeffs'].append(p.ocv_coeffs.detach().cpu().numpy().copy())
            hist['docv_dt_coeffs'].append(p.docv_dt_coeffs.detach().cpu().numpy().copy())
            hist['sigma_data'].append(torch.exp(p.log_sigma_data).item())
            hist['sigma_core'].append(torch.exp(p.log_sigma_core).item())
            hist['sigma_surf'].append(torch.exp(p.log_sigma_surf).item())

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), save_path)

        scheduler.step(vl)

        # ★ v14: 每 epoch 后对热参数进行 clamp（软约束，限制在物理合理范围内）
        if is_pinn:
            with torch.no_grad():
                model.physics.raw_Cc.clamp_(min=20.0, max=200.0)
                model.physics.raw_Cs.clamp_(min=1.0, max=20.0)
                model.physics.raw_Rc.clamp_(min=0.5, max=10.0)
                model.physics.raw_Rs.clamp_(min=2.0, max=30.0)

        # ★ 修复: Phase1期间不启用早停
        if is_pinn and ep < PHASE1_EPOCHS:
            pass  # Phase1不早停
        else:
            if es.step(vl):
                print(f"  早停 @ Epoch {ep+1}")
                break

        if (ep+1) % 10 == 0 or ep == 0 or (is_pinn and ep == PHASE1_EPOCHS):
            phase = ("阶段1(纯数据)" if ep < PHASE1_EPOCHS else "阶段2(PINN)") if is_pinn else "—"
            msg = (f"  Epoch {ep+1:>3d}/{EPOCHS}  train={tl:.6f}  val={vl:.6f}  "
                   f"{phase}  [{time.time()-t0:.0f}s]")
            if is_pinn and ep >= PHASE1_EPOCHS:
                msg += (f"\n    Cc={p.Cc.item():.4f}  Cs={p.Cs.item():.4f}  "
                        f"Rc={p.Rc.item():.4f}  Rs={p.Rs.item():.4f}  "
                        f"σd={torch.exp(p.log_sigma_data).item():.4f}  "
                        f"σc={torch.exp(p.log_sigma_core).item():.4f}  "
                        f"σs={torch.exp(p.log_sigma_surf).item():.4f}")
            print(msg)

    # 加载最优权重
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))
    total_epochs = len(hist['train'])
    print(f"  训练完成  总epoch={total_epochs}  最佳验证损失={best_val:.6f}  耗时 {time.time()-t0:.1f}s")

    if is_pinn:
        p = model.physics
        print(f"\n  ★ 辨识的热参数:")
        print(f"    Cc = {p.Cc.item():.4f} J/K  (核心热容)")
        print(f"    Cs = {p.Cs.item():.4f} J/K  (表面热容)")
        print(f"    Rc = {p.Rc.item():.4f} K/W  (核心→表面热阻)")
        print(f"    Rs = {p.Rs.item():.4f} K/W  (表面→环境热阻)")

    return model, hist


# ============================================================
#  8. 评估指标
# ============================================================

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def predict_all(model, loader, device):
    model.eval()
    preds, trues, phys_list = [], [], []
    with torch.no_grad():
        for x_t, x_s, y, ph in loader:
            x_t, x_s = x_t.to(device), x_s.to(device)
            preds.append(model(x_t, x_s).cpu().numpy())
            trues.append(y.numpy())
            phys_list.append(ph.numpy())
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(phys_list)


def compute_metrics(y_true, y_pred):
    err = y_pred - y_true
    ae  = np.abs(err)
    mask = np.abs(y_true) > 0.1
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'median_ae': np.median(ae),
        'max_ae': np.max(ae),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs(err[mask] / y_true[mask])) * 100 if mask.sum() > 0 else float('nan'),
        'bias': np.mean(err),
        'std_error': np.std(err),
        'p95': np.percentile(ae, 95),
        'p99': np.percentile(ae, 99),
        'within_05': np.mean(ae <= 0.5) * 100,
        'within_10': np.mean(ae <= 1.0) * 100,
    }


def print_metrics(y_true, y_pred, label=''):
    m = compute_metrics(y_true, y_pred)
    print(f"\n  {'='*50}")
    print(f"  评估结果: {label}")
    print(f"  {'='*50}")
    print(f"  MSE:          {m['mse']:.6f}")
    print(f"  RMSE:         {m['rmse']:.4f} ℃")
    print(f"  MAE:          {m['mae']:.4f} ℃")
    print(f"  中位数绝对误差: {m['median_ae']:.4f} ℃")
    print(f"  最大绝对误差:   {m['max_ae']:.4f} ℃")
    print(f"  R²:           {m['r2']:.6f}")
    print(f"  MAPE:         {m['mape']:.2f} %")
    print(f"  误差偏差:      {m['bias']:+.4f} ℃")
    print(f"  误差标准差:    {m['std_error']:.4f} ℃")
    print(f"  95分位误差:    {m['p95']:.4f} ℃")
    print(f"  99分位误差:    {m['p99']:.4f} ℃")
    print(f"  ±0.5℃内比例:  {m['within_05']:.2f} %")
    print(f"  ±1.0℃内比例:  {m['within_10']:.2f} %")
    print(f"  {'='*50}")
    return m


# ============================================================
#  9. 绘图 + CSV 输出函数
# ============================================================

def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [图] {filename}")


def save_csv(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  [CSV] {filename}")


def plot_训练曲线(hist, cn_name):
    df = pd.DataFrame({
        'epoch': range(1, len(hist['train'])+1),
        '训练损失_总': hist['train'],
        '验证损失': hist['val'],
        '数据损失': hist['data_loss'],
        '物理损失': hist['phys_loss'],
    })
    save_csv(df, f'{cn_name}_训练历史.csv')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist['train'], label='训练损失(总)', lw=1.5)
    ax.plot(hist['val'],   label='验证损失', lw=1.5)
    if max(hist['phys_loss']) > 0:
        ax.plot(hist['phys_loss'], label='物理损失', lw=1, ls='--', alpha=0.7)
        ax.axvline(x=PHASE1_EPOCHS, color='gray', ls=':', alpha=0.5, label='阶段2开始')
    ax.set_xlabel('Epoch'); ax.set_ylabel('损失')
    ax.set_title(f'{cn_name} 训练曲线'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_训练曲线.png')


def plot_温度预测对比(y_true, y_pred, cn_name, cycle_ids, phys_data):
    unique_cycles = np.unique(cycle_ids)
    soh_per_cycle = {}
    for cyc in unique_cycles:
        m = cycle_ids == cyc
        soh_per_cycle[cyc] = np.mean(phys_data[m, 3])

    sorted_cycles = sorted(unique_cycles, key=lambda c: soh_per_cycle[c], reverse=True)
    n_total = len(sorted_cycles)
    if n_total >= 3:
        selected = [sorted_cycles[0], sorted_cycles[n_total // 2], sorted_cycles[-1]]
    elif n_total == 2:
        selected = [sorted_cycles[0], sorted_cycles[-1]]
    else:
        selected = sorted_cycles

    # CSV: 代表性循环
    csv_rows = []
    for cyc in selected:
        m = cycle_ids == cyc
        yt = y_true[m]; yp = y_pred[m]; soh_val = soh_per_cycle[cyc]
        for t_idx in range(len(yt)):
            csv_rows.append({
                '循环': cyc, 'SOH': soh_val, '时间步': t_idx,
                '真实温度': yt[t_idx], '预测温度': yp[t_idx],
                '误差': yp[t_idx] - yt[t_idx], '绝对误差': abs(yp[t_idx] - yt[t_idx])
            })
    save_csv(pd.DataFrame(csv_rows), f'{cn_name}_温度预测_分循环数据.csv')

    # CSV: 逐循环统计
    cycle_stats = []
    for cyc in sorted_cycles:
        m = cycle_ids == cyc
        yt = y_true[m]; yp = y_pred[m]
        cycle_stats.append({
            '循环': cyc, 'SOH': soh_per_cycle[cyc], '样本数': m.sum(),
            'RMSE': np.sqrt(mean_squared_error(yt, yp)),
            'MAE': mean_absolute_error(yt, yp),
            '最大误差': np.max(np.abs(yp - yt)),
            '平均偏差': np.mean(yp - yt),
        })
    save_csv(pd.DataFrame(cycle_stats), f'{cn_name}_逐循环统计.csv')

    # 图1: 代表性循环
    n_sel = len(selected)
    fig, axes = plt.subplots(n_sel, 1, figsize=(14, 4 * n_sel))
    if n_sel == 1:
        axes = [axes]
    for ax, cyc in zip(axes, selected):
        m = cycle_ids == cyc
        yt = y_true[m]; yp = y_pred[m]; soh_val = soh_per_cycle[cyc]
        t = np.arange(len(yt))
        ax.plot(t, yt, label='真实内部温度', lw=1.2, color='#2C3E50')
        ax.plot(t, yp, label='预测内部温度', lw=1.2, color='#E74C3C', ls='--')
        ax.fill_between(t, yt, yp, alpha=0.15, color='#E74C3C')
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('温度 (℃)')
        ax.set_title(f'循环 {cyc} (SOH={soh_val:.4f})')
        ax.legend(loc='upper left'); ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(t, np.abs(yt - yp), lw=0.8, color='#27AE60', alpha=0.6, label='|误差|')
        ax2.set_ylabel('绝对误差 (℃)', color='#27AE60')
        ax2.tick_params(axis='y', labelcolor='#27AE60')
        ax2.legend(loc='upper right')
    fig.suptitle(f'{cn_name} 代表性循环温度预测对比', fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_温度预测对比_分循环.png')

    # 图2: 全部循环
    fig, ax = plt.subplots(figsize=(16, 5))
    offset = 0; tick_positions, tick_labels_list = [], []
    for cyc in sorted_cycles:
        m = cycle_ids == cyc
        yt = y_true[m]; yp = y_pred[m]; t = np.arange(len(yt)) + offset
        ax.plot(t, yt, lw=0.8, color='#2C3E50', alpha=0.7)
        ax.plot(t, yp, lw=0.8, color='#E74C3C', alpha=0.7)
        ax.axvline(x=offset, color='gray', ls=':', alpha=0.3, lw=0.5)
        tick_positions.append(offset + len(yt) // 2)
        tick_labels_list.append(f'{cyc}\n({soh_per_cycle[cyc]:.3f})')
        offset += len(yt) + 50
    ax.plot([], [], lw=1.2, color='#2C3E50', label='真实内部温度')
    ax.plot([], [], lw=1.2, color='#E74C3C', label='预测内部温度')
    ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels_list, fontsize=7)
    ax.set_xlabel('循环编号 (SOH)'); ax.set_ylabel('温度 (℃)')
    ax.set_title(f'{cn_name} 全部测试循环温度预测汇总')
    ax.legend(loc='upper left'); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_温度预测对比_全部循环.png')

    # 图3: 散点图
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.1, s=5, c='steelblue')
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='理想线 (y=x)')
    ax.set_xlabel('真实温度 (℃)'); ax.set_ylabel('预测温度 (℃)')
    ax.set_title(f'{cn_name} 预测值 vs 真实值')
    ax.legend(); ax.grid(alpha=0.3); ax.set_aspect('equal')
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_预测散点图.png')


def plot_误差分布(y_true, y_pred, cn_name):
    err = y_pred - y_true; ae = np.abs(err)
    save_csv(pd.DataFrame({
        '误差均值': [np.mean(err)], '误差标准差': [np.std(err)],
        '误差最小值': [np.min(err)], '误差最大值': [np.max(err)],
        '绝对误差均值': [np.mean(ae)], '绝对误差中位数': [np.median(ae)],
        'P95绝对误差': [np.percentile(ae, 95)], 'P99绝对误差': [np.percentile(ae, 99)],
    }), f'{cn_name}_误差汇总统计.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(err, bins=60, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(0, color='red', ls='--', lw=1.5)
    axes[0].set_xlabel('预测误差 (℃)'); axes[0].set_ylabel('频次')
    axes[0].set_title(f'{cn_name} 误差分布直方图'); axes[0].grid(alpha=0.3)
    axes[1].plot(ae, alpha=0.3, lw=0.5, color='steelblue')
    axes[1].axhline(y=np.mean(ae), color='red', ls='--', lw=1,
                    label=f'平均|误差|={np.mean(ae):.4f}℃')
    axes[1].set_xlabel('样本点'); axes[1].set_ylabel('绝对误差 (℃)')
    axes[1].set_title(f'{cn_name} 绝对误差变化'); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_误差分布.png')


def plot_误差箱线图_按SOH(y_true, y_pred, soh, cn_name):
    err = np.abs(y_pred - y_true)
    bins = [(1.0, 0.95), (0.95, 0.90), (0.90, 0.85), (0.85, 0.80)]
    data, labels, stats_rows = [], [], []
    for hi, lo in bins:
        m = (soh <= hi) & (soh > lo)
        if m.sum() > 0:
            e = err[m]; data.append(e); labels.append(f'({lo:.2f},{hi:.2f}]')
            stats_rows.append({
                'SOH区间': f'({lo:.2f},{hi:.2f}]', '样本数': int(m.sum()),
                'RMSE': np.sqrt(np.mean((y_pred[m]-y_true[m])**2)),
                'MAE': np.mean(e), '中位数AE': np.median(e), '最大AE': np.max(e),
                'P95_AE': np.percentile(e, 95), '偏差': np.mean(y_pred[m]-y_true[m]),
            })
    if not data:
        return
    save_csv(pd.DataFrame(stats_rows), f'{cn_name}_SOH区间误差统计.csv')
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    colors = ['#AED6F1', '#A9DFBF', '#F9E79F', '#F5B7B1']
    for patch, c in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(c)
    ax.set_xlabel('SOH 区间'); ax.set_ylabel('绝对误差 (℃)')
    ax.set_title(f'{cn_name} 不同SOH区间误差箱线图'); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_SOH区间误差箱线图.png')


def plot_逐循环RMSE(y_true, y_pred, cycle_ids, phys_data, cn_name):
    unique_cycles = np.unique(cycle_ids)
    rows = []
    for cyc in unique_cycles:
        m = cycle_ids == cyc
        if m.sum() > 10:
            rows.append({
                '循环': cyc, 'SOH': np.mean(phys_data[m, 3]),
                'RMSE': np.sqrt(mean_squared_error(y_true[m], y_pred[m])),
                'MAE': mean_absolute_error(y_true[m], y_pred[m]),
                '样本数': int(m.sum()),
            })
    if len(rows) < 2:
        return
    df = pd.DataFrame(rows).sort_values('SOH', ascending=False)
    save_csv(df, f'{cn_name}_逐循环RMSE.csv')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['SOH'].values, df['RMSE'].values, 'o-', color='steelblue', lw=1.5, ms=6)
    ax.set_xlabel('SOH'); ax.set_ylabel('RMSE (℃)')
    ax.set_title(f'{cn_name} 不同SOH下的RMSE变化')
    ax.invert_xaxis(); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_SOH-RMSE变化趋势.png')


# ---------- PINN 专属 ----------

def plot_热参数收敛(hist, cn_name):
    if 'Cc' not in hist or len(hist['Cc']) == 0:
        return
    df = pd.DataFrame({
        'epoch': range(1, len(hist['Cc'])+1),
        'Cc_JK': hist['Cc'], 'Cs_JK': hist['Cs'],
        'Rc_KW': hist['Rc'], 'Rs_KW': hist['Rs'],
    })
    save_csv(df, f'{cn_name}_热参数历史.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    params = [('Cc', 'J/K', '核心热容'), ('Cs', 'J/K', '表面热容'),
              ('Rc', 'K/W', '核心→表面热阻'), ('Rs', 'K/W', '表面→环境热阻')]
    for ax, (key, unit, title) in zip(axes.flatten(), params):
        vals = hist[key]
        ax.plot(vals, lw=1.5, color='steelblue')
        ax.axhline(y=vals[-1], color='red', ls='--', alpha=0.6,
                   label=f'最终值: {vals[-1]:.4f} {unit}')
        ax.axvline(x=PHASE1_EPOCHS, color='gray', ls=':', alpha=0.5, label='阶段2开始')
        ax.set_xlabel('Epoch'); ax.set_ylabel(f'{key} ({unit})')
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle(f'{cn_name} 热参数收敛曲线', fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_热参数收敛曲线.png')


def plot_OCV辨识对比(model, cn_name):
    if not isinstance(model, PINNModel):
        return
    soc_range = torch.linspace(0, 1, 200)
    with torch.no_grad():
        ocv_learned  = model.physics.compute_ocv(soc_range).numpy()
        docv_learned = model.physics.compute_docv_dt(soc_range).numpy()
    save_csv(pd.DataFrame({
        'SOC': soc_range.numpy(), 'OCV_辨识_V': ocv_learned, 'dOCV_dT_辨识_VK': docv_learned,
    }), f'{cn_name}_OCV辨识数据.csv')
    save_csv(pd.DataFrame({
        'SOC_参考': OCV_SOC_REF, 'OCV_参考_V': OCV_V_REF, 'dOCV_dT_参考_VK': DOCV_DT_REF,
    }), f'{cn_name}_OCV参考数据.csv')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(soc_range.numpy(), ocv_learned, 'b-', lw=2, label='模型辨识值')
    axes[0].plot(OCV_SOC_REF, OCV_V_REF, 'ro', ms=8, label='文献参考值')
    axes[0].set_xlabel('SOC'); axes[0].set_ylabel('OCV (V)')
    axes[0].set_title('OCV-SOC 曲线对比'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(soc_range.numpy(), docv_learned * 1000, 'b-', lw=2, label='模型辨识值')
    axes[1].plot(DOCV_SOC_REF, np.array(DOCV_DT_REF) * 1000, 'ro', ms=8, label='文献参考值')
    axes[1].set_xlabel('SOC'); axes[1].set_ylabel('dOCV/dT (mV/K)')
    axes[1].set_title('熵热系数对比'); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle(f'{cn_name} OCV与熵热系数辨识结果', fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_OCV辨识对比.png')


def plot_内阻与SOH关系(phys_data, model, cn_name):
    if not isinstance(model, PINNModel):
        return
    V, I, soc, soh = phys_data[:,0], phys_data[:,1], phys_data[:,2], phys_data[:,3]
    with torch.no_grad():
        ocv = model.physics.compute_ocv(torch.tensor(soc, dtype=torch.float32)).numpy()
    valid = np.abs(I) > 0.5
    r_eff = np.full_like(V, np.nan)
    r_eff[valid] = (V[valid] - ocv[valid]) / I[valid]
    soh_bins = np.arange(0.79, 1.005, 0.01)
    rows = []
    for i in range(len(soh_bins) - 1):
        m = valid & (soh >= soh_bins[i]) & (soh < soh_bins[i+1])
        if m.sum() > 10:
            rows.append({
                'SOH_center': (soh_bins[i] + soh_bins[i+1]) / 2,
                'R_eff_mean_ohm': np.nanmean(r_eff[m]),
                'R_eff_std_ohm': np.nanstd(r_eff[m]),
                'R_eff_mean_mohm': np.nanmean(r_eff[m]) * 1000,
                '样本数': int(m.sum()),
            })
    if len(rows) < 2:
        return
    df = pd.DataFrame(rows)
    save_csv(df, f'{cn_name}_内阻SOH数据.csv')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(df['SOH_center'], df['R_eff_mean_mohm'],
                yerr=df['R_eff_std_ohm']*1000, fmt='o-', color='steelblue',
                capsize=3, lw=1.5, label='等效内阻 (均值±标准差)')
    ax.set_xlabel('SOH'); ax.set_ylabel('等效内阻 (mΩ)')
    ax.set_title(f'{cn_name} 等效内阻随SOH变化')
    ax.invert_xaxis(); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_内阻与SOH关系.png')


def plot_损失分解(hist, cn_name):
    if max(hist['phys_loss']) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(hist['data_loss'], label='数据损失', lw=1.5)
    axes[0].plot(hist['phys_loss'], label='物理损失', lw=1.5)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('损失值')
    axes[0].set_title('数据损失 vs 物理损失'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].axvline(x=PHASE1_EPOCHS, color='gray', ls=':', alpha=0.5)
    dl = np.array(hist['data_loss']); pl = np.array(hist['phys_loss'])
    ratio = pl / (dl + pl + 1e-10) * 100
    axes[1].plot(ratio, lw=1.5, color='steelblue')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('物理损失占比 (%)')
    axes[1].set_title('物理损失占总损失比例'); axes[1].grid(alpha=0.3)
    axes[1].axvline(x=PHASE1_EPOCHS, color='gray', ls=':', alpha=0.5)
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_损失分解.png')


def plot_OCV系数演变(hist, cn_name):
    if 'ocv_coeffs' not in hist or len(hist['ocv_coeffs']) == 0:
        return
    coeffs_arr = np.array(hist['ocv_coeffs'])
    df = pd.DataFrame(coeffs_arr, columns=[f'a{i}' for i in range(coeffs_arr.shape[1])])
    df.insert(0, 'epoch', range(1, len(df)+1))
    save_csv(df, f'{cn_name}_OCV系数历史.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i in range(coeffs_arr.shape[1]):
        axes[0].plot(coeffs_arr[:, i], lw=1.2, label=f'a{i}')
    axes[0].axvline(x=PHASE1_EPOCHS, color='gray', ls=':', alpha=0.5, label='阶段2开始')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('系数值')
    axes[0].set_title('OCV 多项式系数演变'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    soc = np.linspace(0, 1, 100)
    snapshots = sorted(set([min(s, len(coeffs_arr)-1)
                            for s in [0, PHASE1_EPOCHS, len(coeffs_arr)//2, len(coeffs_arr)-1]]))
    colors_snap = ['#AED6F1', '#85C1E9', '#3498DB', '#1A5276']
    for idx, ep in enumerate(snapshots):
        c = coeffs_arr[ep]
        ocv_val = sum(c[i] * soc**i for i in range(len(c)))
        axes[1].plot(soc, ocv_val, lw=1.5, color=colors_snap[idx % len(colors_snap)],
                     label=f'Epoch {ep+1}')
    axes[1].plot(OCV_SOC_REF, OCV_V_REF, 'ro', ms=6, zorder=10, label='文献参考值')
    axes[1].set_xlabel('SOC'); axes[1].set_ylabel('OCV (V)')
    axes[1].set_title('OCV 曲线随训练的演变'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle(f'{cn_name} OCV多项式系数演变', fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_OCV系数演变.png')


def plot_熵热系数演变(hist, cn_name):
    if 'docv_dt_coeffs' not in hist or len(hist['docv_dt_coeffs']) == 0:
        return
    coeffs_arr = np.array(hist['docv_dt_coeffs'])
    df = pd.DataFrame(coeffs_arr, columns=[f'b{i}' for i in range(coeffs_arr.shape[1])])
    df.insert(0, 'epoch', range(1, len(df)+1))
    save_csv(df, f'{cn_name}_熵热系数历史.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i in range(coeffs_arr.shape[1]):
        axes[0].plot(coeffs_arr[:, i], lw=1.2, label=f'b{i}')
    axes[0].axvline(x=PHASE1_EPOCHS, color='gray', ls=':', alpha=0.5, label='阶段2开始')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('系数值')
    axes[0].set_title('dOCV/dT 多项式系数演变'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    soc = np.linspace(0, 1, 100)
    snapshots = sorted(set([min(s, len(coeffs_arr)-1)
                            for s in [0, PHASE1_EPOCHS, len(coeffs_arr)//2, len(coeffs_arr)-1]]))
    colors_snap = ['#ABEBC6', '#58D68D', '#27AE60', '#1E8449']
    for idx, ep in enumerate(snapshots):
        c = coeffs_arr[ep]
        val = sum(c[i] * soc**i for i in range(len(c)))
        axes[1].plot(soc, val * 1000, lw=1.5, color=colors_snap[idx % len(colors_snap)],
                     label=f'Epoch {ep+1}')
    axes[1].plot(DOCV_SOC_REF, np.array(DOCV_DT_REF)*1000, 'ro', ms=6, zorder=10, label='文献参考值')
    axes[1].set_xlabel('SOC'); axes[1].set_ylabel('dOCV/dT (mV/K)')
    axes[1].set_title('熵热系数曲线随训练的演变'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle(f'{cn_name} 熵热系数演变', fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_熵热系数演变.png')


def plot_自适应权重演变(hist, cn_name):
    if 'sigma_data' not in hist or len(hist['sigma_data']) == 0:
        return
    df = pd.DataFrame({
        'epoch': range(1, len(hist['sigma_data'])+1),
        'sigma_data': hist['sigma_data'], 'sigma_core': hist['sigma_core'],
        'sigma_surf': hist['sigma_surf'],
        'weight_data': [1.0/(2*s**2+1e-8) for s in hist['sigma_data']],
        'weight_core': [1.0/(2*s**2+1e-8) for s in hist['sigma_core']],
        'weight_surf': [1.0/(2*s**2+1e-8) for s in hist['sigma_surf']],
    })
    save_csv(df, f'{cn_name}_自适应权重历史.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(hist['sigma_data'], lw=1.5, label='σ_data (数据)')
    axes[0].plot(hist['sigma_core'], lw=1.5, label='σ_core (核心热模型)')
    axes[0].plot(hist['sigma_surf'], lw=1.5, label='σ_surface (表面热模型)')
    axes[0].axvline(x=PHASE1_EPOCHS, color='gray', ls=':', alpha=0.5, label='阶段2开始')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('σ 值')
    axes[0].set_title('自适应权重 σ 演变'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(df['weight_data'], lw=1.5, label='权重_数据')
    axes[1].plot(df['weight_core'], lw=1.5, label='权重_核心热模型')
    axes[1].plot(df['weight_surf'], lw=1.5, label='权重_表面热模型')
    axes[1].axvline(x=PHASE1_EPOCHS, color='gray', ls=':', alpha=0.5, label='阶段2开始')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('等效权重 1/(2σ²)')
    axes[1].set_title('各损失项等效权重演变'); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle(f'{cn_name} 自适应损失权重演变', fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_自适应权重演变.png')


def plot_参数演变全景图(hist, cn_name):
    if 'Cc' not in hist or len(hist['Cc']) == 0:
        return
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    params_top = [
        ('Cc', 'J/K', '核心热容', '#3498DB'), ('Cs', 'J/K', '表面热容', '#2ECC71'),
        ('Rc', 'K/W', '核心→表面热阻', '#E74C3C'), ('Rs', 'K/W', '表面→环境热阻', '#F39C12'),
    ]
    for ax, (key, unit, title, color) in zip(axes[0], params_top):
        vals = hist[key]
        ax.plot(vals, lw=1.5, color=color)
        ax.axhline(y=vals[-1], color='black', ls='--', alpha=0.4, label=f'最终: {vals[-1]:.4f}')
        ax.axvline(x=PHASE1_EPOCHS, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel(f'{key} ({unit})')
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    if 'sigma_data' in hist and len(hist['sigma_data']) > 0:
        axes[1][0].plot(hist['sigma_data'], lw=1.5, color='#3498DB', label='σ_data')
        axes[1][0].plot(hist['sigma_core'], lw=1.5, color='#E74C3C', label='σ_core')
        axes[1][0].plot(hist['sigma_surf'], lw=1.5, color='#F39C12', label='σ_surf')
        axes[1][0].axvline(x=PHASE1_EPOCHS, color='gray', ls=':', alpha=0.5)
        axes[1][0].set_xlabel('Epoch'); axes[1][0].set_ylabel('σ')
        axes[1][0].set_title('自适应权重 σ'); axes[1][0].legend(fontsize=8); axes[1][0].grid(alpha=0.3)
        w_d = [1.0/(2*s**2+1e-8) for s in hist['sigma_data']]
        w_c = [1.0/(2*s**2+1e-8) for s in hist['sigma_core']]
        w_s = [1.0/(2*s**2+1e-8) for s in hist['sigma_surf']]
        axes[1][1].plot(w_d, lw=1.5, color='#3498DB', label='w_data')
        axes[1][1].plot(w_c, lw=1.5, color='#E74C3C', label='w_core')
        axes[1][1].plot(w_s, lw=1.5, color='#F39C12', label='w_surf')
        axes[1][1].axvline(x=PHASE1_EPOCHS, color='gray', ls=':', alpha=0.5)
        axes[1][1].set_xlabel('Epoch'); axes[1][1].set_ylabel('1/(2σ²)')
        axes[1][1].set_title('等效权重'); axes[1][1].legend(fontsize=8); axes[1][1].grid(alpha=0.3)
    else:
        axes[1][0].set_visible(False); axes[1][1].set_visible(False)
    axes[1][2].plot(hist['data_loss'], lw=1.5, label='数据损失', color='#3498DB')
    axes[1][2].plot(hist['phys_loss'], lw=1.5, label='物理损失', color='#E74C3C')
    axes[1][2].axvline(x=PHASE1_EPOCHS, color='gray', ls=':', alpha=0.5)
    axes[1][2].set_xlabel('Epoch'); axes[1][2].set_ylabel('Loss')
    axes[1][2].set_title('损失分解'); axes[1][2].legend(fontsize=8); axes[1][2].grid(alpha=0.3)
    axes[1][3].plot(hist['train'], lw=1.5, label='训练(总)', color='#3498DB')
    axes[1][3].plot(hist['val'], lw=1.5, label='验证', color='#2ECC71')
    axes[1][3].axvline(x=PHASE1_EPOCHS, color='gray', ls=':', alpha=0.5)
    axes[1][3].set_xlabel('Epoch'); axes[1][3].set_ylabel('Loss')
    axes[1][3].set_title('训练曲线'); axes[1][3].legend(fontsize=8); axes[1][3].grid(alpha=0.3)
    fig.suptitle(f'{cn_name} 全部参数演变全景图', fontsize=15, y=1.01)
    fig.tight_layout()
    save_fig(fig, f'{cn_name}_参数演变全景图.png')


# ============================================================
#  10. 完整评估流程
# ============================================================

def full_evaluate(model, test_ld, scaler, device, hist, model_name, cycle_ids):
    cn = MODEL_CN_NAMES.get(model_name, model_name)
    print(f"\n[6/7] 评估: {cn}")

    y_pred_n, y_true_n, phys = predict_all(model, test_ld, device)
    y_pred = scaler.inverse_target(y_pred_n)
    y_true = scaler.inverse_target(y_true_n)
    soh = phys[:, 3]

    metrics = print_metrics(y_true, y_pred, cn)

    bins = [(1.0, 0.95), (0.95, 0.90), (0.90, 0.85), (0.85, 0.80)]
    print(f"\n  {'SOH区间':<16} {'RMSE(℃)':>8}  {'MAE(℃)':>8}  {'样本数':>8}")
    print(f"  {'-'*46}")
    for hi, lo in bins:
        m = (soh <= hi) & (soh > lo)
        n = m.sum()
        if n > 0:
            r = np.sqrt(mean_squared_error(y_true[m], y_pred[m]))
            a = mean_absolute_error(y_true[m], y_pred[m])
            print(f"  ({lo:.2f}, {hi:.2f}]    {r:>8.4f}  {a:>8.4f}  {n:>8,}")

    print(f"\n  生成图表与CSV ...")
    for func, name in [
        (lambda: plot_训练曲线(hist, cn), '训练曲线'),
        (lambda: plot_温度预测对比(y_true, y_pred, cn, cycle_ids, phys), '温度预测对比'),
        (lambda: plot_误差分布(y_true, y_pred, cn), '误差分布'),
        (lambda: plot_误差箱线图_按SOH(y_true, y_pred, soh, cn), 'SOH误差箱线图'),
        (lambda: plot_逐循环RMSE(y_true, y_pred, cycle_ids, phys, cn), '逐循环RMSE'),
    ]:
        try:
            func()
        except Exception as e:
            print(f"  [绘图异常] {name}: {e}")

    if isinstance(model, PINNModel):
        for func, name in [
            (lambda: plot_热参数收敛(hist, cn), '热参数收敛'),
            (lambda: plot_OCV辨识对比(model, cn), 'OCV辨识'),
            (lambda: plot_内阻与SOH关系(phys, model, cn), '内阻SOH'),
            (lambda: plot_损失分解(hist, cn), '损失分解'),
            (lambda: plot_OCV系数演变(hist, cn), 'OCV系数演变'),
            (lambda: plot_熵热系数演变(hist, cn), '熵热系数演变'),
            (lambda: plot_自适应权重演变(hist, cn), '自适应权重'),
            (lambda: plot_参数演变全景图(hist, cn), '全景图'),
        ]:
            try:
                func()
            except Exception as e:
                print(f"  [绘图异常] {name}: {e}")

    return metrics


# ============================================================
#  11. 模型对比
# ============================================================

def plot_模型对比(all_results):
    if len(all_results) < 2:
        return
    names = [MODEL_CN_NAMES.get(k, k) for k in all_results.keys()]
    rmse = [v['rmse'] for v in all_results.values()]
    mae  = [v['mae'] for v in all_results.values()]
    r2   = [v['r2'] for v in all_results.values()]
    w10  = [v['within_10'] for v in all_results.values()]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12'][:len(names)]
    for ax, vals, ylabel, title in [
        (axes[0,0], rmse, 'RMSE (℃)', 'RMSE 对比'),
        (axes[0,1], mae, 'MAE (℃)', 'MAE 对比'),
        (axes[1,0], r2, 'R²', 'R² 对比'),
        (axes[1,1], w10, '比例 (%)', '±1.0℃内预测比例'),
    ]:
        bars = ax.bar(names, vals, color=colors, edgecolor='black', alpha=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v*1.001, f'{v:.4f}', ha='center', fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.3, axis='y')
    fig.suptitle('模型对比总结', fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, '模型对比总结.png')


# ============================================================
#  12. 主流程
# ============================================================

def run_experiment(model_name='pinn_lstm'):
    set_seed()
    device = get_device()

    df_a, df_b = load_raw_data()
    inspect_columns(df_a, 'Battery A')
    inspect_columns(df_b, 'Battery B')

    df_a = filter_discharge(df_a, 'Battery A')
    df_b = filter_discharge(df_b, 'Battery B')

    df_train, df_val, df_test = sample_and_split(df_a, df_b)
    df_train, df_val, df_test, scaler = prepare_features(df_train, df_val, df_test)
    train_ld, val_ld, test_ld, cycle_ids_test = build_dataloaders(df_train, df_val, df_test)

    t_dim = len(TEMPORAL_FEATURES)
    s_dim = len(STATIC_FEATURES)
    model = build_model(model_name, t_dim, s_dim)
    model, hist = train_model(model, train_ld, val_ld, device, model_name)

    model = model.to(device)
    metrics = full_evaluate(model, test_ld, scaler, device, hist, model_name, cycle_ids_test)
    return metrics


def run_all():
    all_results = {}
    for name in ['base_lstm', 'gru', 'pinn_lstm']:
        cn = MODEL_CN_NAMES.get(name, name)
        print(f"\n{'#'*60}")
        print(f"#  实验: {cn} ({name})")
        print(f"{'#'*60}")
        try:
            all_results[name] = run_experiment(name)
        except Exception as e:
            print(f"  [错误] {cn}: {e}")
            traceback.print_exc()
            err_path = os.path.join(OUTPUT_DIR, f'{cn}_错误日志.txt')
            with open(err_path, 'w', encoding='utf-8') as f:
                f.write(f"模型: {cn} ({name})\n错误: {e}\n\n{traceback.format_exc()}")
            print(f"  错误日志已保存: {err_path}")

    if all_results:
        print(f"\n\n{'='*80}")
        print(f"  [7/7] 模型对比结果")
        print(f"{'='*80}")
        h = (f"  {'模型':<12} {'RMSE':>7} {'MAE':>7} {'R²':>8} "
             f"{'MaxAE':>7} {'MAPE%':>7} {'±1℃%':>7}")
        print(h); print(f"  {'-'*62}")
        for n, m in all_results.items():
            cn = MODEL_CN_NAMES.get(n, n)
            print(f"  {cn:<12} {m['rmse']:>7.4f} {m['mae']:>7.4f} {m['r2']:>8.4f} "
                  f"{m['max_ae']:>7.4f} {m['mape']:>7.2f} {m['within_10']:>7.2f}")
        plot_模型对比(all_results)
        df_r = pd.DataFrame(all_results).T
        df_r.index.name = '模型'
        csv_path = os.path.join(OUTPUT_DIR, '模型对比结果.csv')
        df_r.to_csv(csv_path, encoding='utf-8-sig')
        print(f"\n  对比结果已保存: {csv_path}")
    print(f"\n  所有输出文件位于: {OUTPUT_DIR}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_experiment(sys.argv[1])
    else:
        run_all()