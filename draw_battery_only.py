"""
Cylindrical Li-ion Battery Physical Thermal Model (Left Panel Only)
===================================================================
Draws a single publication-quality figure of an 18650 battery with
internal two-node thermal network overlay.

Usage:
    pip install matplotlib numpy
    python draw_battery_only.py

Output:
    battery_physical_thermal_model.png (300 dpi)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Arc
from matplotlib.path import Path
import matplotlib.patches as mpatches

# ============================================================
# CONFIGURATION
# ============================================================
DPI = 300
FIG_W = 7.0    # inches (single column width)
FIG_H = 9.0    # inches

# Battery geometry (in data coordinates)
BAT_CX = 0.0
BAT_CY = 0.0
BAT_W = 1.6       # cylinder width
BAT_H = 3.8       # cylinder height
TILT = 0.15       # perspective ratio for top/bottom ellipses

# Colors
C_SHELL = (1.0, 0.68, 0.78)       # pink shell
C_SHELL_EDGE = (0.60, 0.28, 0.38)
C_CORE = (1.0, 0.88, 0.92)
C_METAL = (0.73, 0.73, 0.77)
C_METAL_BRIGHT = (0.94, 0.94, 0.97)
C_METAL_EDGE = (0.35, 0.35, 0.38)
C_INSUL = (0.12, 0.12, 0.12)
C_TC = (0.50, 0.0, 0.50)          # purple
C_TS = (0.85, 0.10, 0.10)         # red
C_RC = (1.0, 0.78, 0.85)          # light pink
C_RS = (1.0, 0.55, 0.15)          # orange
C_WIRE = "black"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
})


# ============================================================
# BATTERY DRAWING
# ============================================================

def draw_cylinder(ax):
    """Draw the 3D cylindrical battery body."""
    cx, cy = BAT_CX, BAT_CY
    a = BAT_W / 2          # ellipse semi-major (x)
    b = a * TILT           # ellipse semi-minor (y, perspective)
    top_y = cy + BAT_H / 2
    bot_y = cy - BAT_H / 2

    # --- Side wall with gradient shading ---
    n = 60
    for i in range(n):
        frac = (i + 0.5) / n
        xi = cx - a + frac * BAT_W
        w = BAT_W / n
        d = abs(frac - 0.5) / 0.5
        # color: brighter in center, darker at edges
        r = 1.0
        g = 0.68 - 0.15 * d
        bl = 0.78 - 0.10 * d
        alpha = 0.30 + 0.30 * (1.0 - d ** 2)
        ax.add_patch(plt.Rectangle((xi, bot_y), w, BAT_H,
                                   fc=(r, g, bl, alpha), ec="none", zorder=1))

    # --- Core region (lighter inner area) ---
    ca = a * 0.52
    for i in range(30):
        frac = (i + 0.5) / 30
        xi = cx - ca + frac * 2 * ca
        w = 2 * ca / 30
        d = abs(frac - 0.5) / 0.5
        alpha = 0.10 + 0.15 * (1.0 - d ** 2)
        ax.add_patch(plt.Rectangle((xi, bot_y + BAT_H * 0.07), w, BAT_H * 0.86,
                                   fc=(*C_CORE, alpha), ec="none", zorder=1.5))

    # --- Side outlines ---
    ax.plot([cx - a, cx - a], [bot_y, top_y], color=C_SHELL_EDGE, lw=1.6, zorder=3)
    ax.plot([cx + a, cx + a], [bot_y, top_y], color=C_SHELL_EDGE, lw=1.6, zorder=3)

    # --- Bottom ellipse ---
    ax.add_patch(Ellipse((cx, bot_y), BAT_W, 2 * b,
                         fc=(*C_SHELL, 0.55), ec=C_SHELL_EDGE, lw=1.4, zorder=2))

    # --- Top ellipse ---
    ax.add_patch(Ellipse((cx, top_y), BAT_W, 2 * b,
                         fc=(*C_CORE, 0.88), ec=C_SHELL_EDGE, lw=1.4, zorder=4))

    # --- Shadow under battery (subtle) ---
    ax.add_patch(Ellipse((cx, bot_y - 0.08), BAT_W * 1.05, b * 1.2,
                         fc=(0.0, 0.0, 0.0, 0.06), ec="none", zorder=0.5))

    return a, b, top_y, bot_y


def draw_positive_cap(ax, a, b, top_y):
    """Draw the metallic positive terminal cap on top of the battery."""
    cx = BAT_CX
    cap_h = 0.16
    cap_a = a * 0.90
    cap_b = b * 0.88

    base_y = top_y
    face_y = base_y + cap_h

    # --- Side wall gradient (metallic cylinder wall) ---
    n = 40
    for i in range(n):
        frac = (i + 0.5) / n
        xi = cx - cap_a + frac * 2 * cap_a
        w = 2 * cap_a / n
        d = abs(frac - 0.5) / 0.5
        v = 0.82 - 0.30 * d ** 1.4
        ax.add_patch(plt.Rectangle((xi, base_y), w, cap_h,
                                   fc=(v, v, v + 0.02, 0.96), ec="none", zorder=4.5))

    # Side edges
    ax.plot([cx - cap_a, cx - cap_a], [base_y, face_y], color=C_METAL_EDGE, lw=1.4, zorder=4.6)
    ax.plot([cx + cap_a, cx + cap_a], [base_y, face_y], color=C_METAL_EDGE, lw=1.4, zorder=4.6)

    # --- Top face: outer metal ring ---
    ax.add_patch(Ellipse((cx, face_y), 2 * cap_a, 2 * cap_b,
                         fc=C_METAL, ec=C_METAL_EDGE, lw=1.3, zorder=5))

    # Specular highlight (upper-left area)
    ax.add_patch(Ellipse((cx - cap_a * 0.25, face_y + cap_b * 0.20),
                         cap_a * 0.75, cap_b * 0.50,
                         fc=C_METAL_BRIGHT, ec="none", alpha=0.55, zorder=5.05))

    # --- Black insulation ring ---
    ins_a = cap_a * 0.70
    ins_b = cap_b * 0.70
    ax.add_patch(Ellipse((cx, face_y), 2 * ins_a, 2 * ins_b,
                         fc=C_INSUL, ec=(0.0, 0.0, 0.0), lw=0.9, zorder=5.1))

    # --- Central bump (raised positive terminal) ---
    bp_a = cap_a * 0.36
    bp_b = cap_b * 0.50

    # Bump shadow ring
    ax.add_patch(Ellipse((cx, face_y + 0.015), 2 * bp_a * 1.12, 2 * bp_b * 1.12,
                         fc=(0.55, 0.55, 0.58), ec=(0.38, 0.38, 0.40), lw=0.6, zorder=5.2))
    # Bump surface
    ax.add_patch(Ellipse((cx, face_y + 0.03), 2 * bp_a, 2 * bp_b,
                         fc=(0.80, 0.80, 0.84), ec=(0.50, 0.50, 0.52), lw=0.8, zorder=5.3))
    # Bump specular
    ax.add_patch(Ellipse((cx - bp_a * 0.20, face_y + 0.04),
                         bp_a * 0.55, bp_b * 0.42,
                         fc=(0.97, 0.97, 1.0), ec="none", alpha=0.72, zorder=5.4))

    # "+" symbol
    ax.text(cx, face_y + cap_b + 0.10, "+", fontsize=13, fontweight="bold",
            ha="center", va="center", zorder=6)


# ============================================================
# CIRCUIT ELEMENT PRIMITIVES
# ============================================================

def draw_heat_source(ax, x, y, r=0.13):
    """Draw heat source symbol: circle with sine wave."""
    ax.add_patch(plt.Circle((x, y), r, fc="white", ec="black", lw=1.7, zorder=10))
    t = np.linspace(-0.65 * r, 0.65 * r, 50)
    wave = np.sin(t / r * 2.5 * np.pi) * r * 0.35
    ax.plot(t + x, wave + y, "k-", lw=1.3, zorder=11)


def draw_resistor_h(ax, x1, x2, y, color, lw_wire=1.8):
    """Horizontal resistor block between x1 and x2 at height y."""
    mx = (x1 + x2) / 2
    rl = (x2 - x1) * 0.50
    rh = 0.13
    # wires
    ax.plot([x1, mx - rl / 2], [y, y], color=C_WIRE, lw=lw_wire, zorder=9)
    ax.plot([mx + rl / 2, x2], [y, y], color=C_WIRE, lw=lw_wire, zorder=9)
    # resistor box
    ax.add_patch(FancyBboxPatch((mx - rl / 2, y - rh / 2), rl, rh,
                                boxstyle="round,pad=0.015", fc=color, ec="black",
                                lw=1.4, zorder=10))
    return mx, y


def draw_resistor_v(ax, x, y1, y2, color, lw_wire=1.8):
    """Vertical resistor block between y1 (top) and y2 (bottom) at x."""
    my = (y1 + y2) / 2
    rl = abs(y1 - y2) * 0.50
    rw = 0.13
    # wires
    ax.plot([x, x], [y1, my + rl / 2], color=C_WIRE, lw=lw_wire, zorder=9)
    ax.plot([x, x], [my - rl / 2, y2], color=C_WIRE, lw=lw_wire, zorder=9)
    # resistor box
    ax.add_patch(FancyBboxPatch((x - rw / 2, my - rl / 2), rw, rl,
                                boxstyle="round,pad=0.015", fc=color, ec="black",
                                lw=1.4, zorder=10))
    return x, my


def draw_capacitor_v(ax, x, y_top, y_bot, lw_wire=1.8):
    """Vertical capacitor between y_top and y_bot at x."""
    my = (y_top + y_bot) / 2
    gap = 0.08
    pw = 0.18  # plate half-width
    # wires
    ax.plot([x, x], [y_top, my + gap], color=C_WIRE, lw=lw_wire, zorder=9)
    ax.plot([x, x], [my - gap, y_bot], color=C_WIRE, lw=lw_wire, zorder=9)
    # plates
    ax.plot([x - pw, x + pw], [my + gap, my + gap], color="black", lw=2.8, zorder=10)
    ax.plot([x - pw, x + pw], [my - gap, my - gap], color="black", lw=2.8, zorder=10)
    return my


def draw_ground(ax, x, y, size=0.12):
    """Standard ground symbol."""
    s = size
    ax.plot([x, x], [y, y - s * 0.45], color=C_WIRE, lw=1.8, zorder=9)
    ax.plot([x - s, x + s], [y - s * 0.45, y - s * 0.45], color=C_WIRE, lw=1.8, zorder=9)
    ax.plot([x - s * 0.62, x + s * 0.62], [y - s * 0.70, y - s * 0.70], color=C_WIRE, lw=1.8, zorder=9)
    ax.plot([x - s * 0.28, x + s * 0.28], [y - s * 0.95, y - s * 0.95], color=C_WIRE, lw=1.8, zorder=9)


def draw_dot(ax, x, y, color, r=0.055):
    """Node dot."""
    ax.add_patch(plt.Circle((x, y), r, fc=color, ec="black", lw=0.9, zorder=12))


# ============================================================
# INTERNAL THERMAL NETWORK
# ============================================================

def draw_thermal_network(ax):
    """Draw the two-node thermal model inside the battery."""
    cx = BAT_CX

    # --- Key positions ---
    qe_x, qe_y = cx - 0.28, BAT_CY - 0.85    # heat source
    tc_x, tc_y = cx - 0.28, BAT_CY + 0.45     # Tc node
    ts_x, ts_y = cx + 0.42, BAT_CY + 0.45     # Ts node
    gnd_y = BAT_CY - 1.55                       # ground rail y

    # --- Heat source Qe ---
    qe_r = 0.13
    draw_heat_source(ax, qe_x, qe_y, r=qe_r)
    ax.text(qe_x - 0.32, qe_y, r"$Q_e$", fontsize=12, ha="center", va="center", zorder=11)

    # Wire: Qe top → Tc
    ax.plot([qe_x, qe_x], [qe_y + qe_r, tc_y], color=C_WIRE, lw=1.7, zorder=9)
    # Wire: Qe bottom → ground
    ax.plot([qe_x, qe_x], [qe_y - qe_r, gnd_y], color=C_WIRE, lw=1.7, zorder=9)

    # --- Tc node ---
    draw_dot(ax, tc_x, tc_y, C_TC)
    ax.text(tc_x, tc_y + 0.18, r"$T_c$", fontsize=12, ha="center", va="center", zorder=12)

    # --- Ts node ---
    draw_dot(ax, ts_x, ts_y, C_TS)
    ax.text(ts_x, ts_y + 0.18, r"$T_s$", fontsize=12, ha="center", va="center", zorder=12)

    # --- Rc: horizontal between Tc and Ts ---
    rc_mx, _ = draw_resistor_h(ax, tc_x, ts_x, tc_y, C_RC)
    ax.text(rc_mx, tc_y + 0.16, r"$R_c$", fontsize=11, ha="center", va="center", zorder=11)

    # --- Cc: from Tc downward to ground ---
    cc_top = tc_y - 0.12
    cc_bot = gnd_y + 0.08
    cc_my = draw_capacitor_v(ax, tc_x, cc_top, cc_bot)
    ax.text(tc_x - 0.25, cc_my, r"$C_c$", fontsize=11, ha="center", va="center", zorder=11)

    # --- Rs: from Ts downward ---
    rs_top = ts_y - 0.12
    rs_bot = gnd_y + 0.55
    _, rs_my = draw_resistor_v(ax, ts_x, rs_top, rs_bot, C_RS)
    ax.text(ts_x + 0.20, rs_my, r"$R_s$", fontsize=11, ha="left", va="center", zorder=11)
    # wire from Rs bottom to ground
    ax.plot([ts_x, ts_x], [rs_bot, gnd_y], color=C_WIRE, lw=1.5, zorder=9)

    # --- Cs: parallel path from Ts node ---
    cs_x = ts_x + 0.38
    ax.plot([ts_x, cs_x], [ts_y, ts_y], color=C_WIRE, lw=1.5, zorder=9)  # horizontal branch
    cs_top = ts_y - 0.12
    cs_bot = gnd_y + 0.08
    cs_my = draw_capacitor_v(ax, cs_x, cs_top, cs_bot)
    ax.text(cs_x + 0.25, cs_my, r"$C_s$", fontsize=11, ha="left", va="center", zorder=11)
    # wire from Cs bottom to ground rail
    ax.plot([cs_x, cs_x], [cs_bot, gnd_y], color=C_WIRE, lw=1.5, zorder=9)

    # --- Ground rail ---
    ax.plot([qe_x - 0.12, cs_x + 0.12], [gnd_y, gnd_y], color=C_WIRE, lw=1.8, zorder=9)

    # Ground symbol
    gnd_x = (tc_x + ts_x) / 2
    draw_ground(ax, gnd_x, gnd_y, size=0.13)
    ax.text(gnd_x, gnd_y - 0.28, r"$T_a$", fontsize=12, ha="center", va="top", zorder=11)

    return tc_x, tc_y, ts_x, ts_y


# ============================================================
# ANNOTATIONS
# ============================================================

def draw_annotations(ax, tc_x, tc_y, ts_x, ts_y):
    """Draw dashed leader lines and temperature labels."""
    ann_x = BAT_CX + BAT_W / 2 + 0.55  # annotation text x position

    # --- Internal Temperature (Tc) ---
    ax.plot([tc_x + 0.08, ann_x - 0.05], [tc_y, tc_y],
            ls="--", color=C_TC, lw=1.2, zorder=8)
    ax.text(ann_x, tc_y + 0.10, "Internal Temperature",
            fontsize=10, fontweight="bold", color=C_TC, va="center", zorder=12)
    ax.text(ann_x, tc_y - 0.12, r"$T_c$",
            fontsize=11, color="black", fontstyle="italic", va="center", zorder=12)

    # --- Surface Temperature (Ts) ---
    ts_ann_y = ts_y - 0.50
    ax.plot([ts_x + 0.08, ts_x + 0.08], [ts_y, ts_ann_y],
            ls="--", color=C_TS, lw=1.2, zorder=8)
    ax.plot([ts_x + 0.08, ann_x - 0.05], [ts_ann_y, ts_ann_y],
            ls="--", color=C_TS, lw=1.2, zorder=8)
    ax.text(ann_x, ts_ann_y + 0.10, "Surface Temperature",
            fontsize=10, fontweight="bold", color=C_TS, va="center", zorder=12)
    ax.text(ann_x, ts_ann_y - 0.12, r"$T_s$",
            fontsize=11, color="black", fontstyle="italic", va="center", zorder=12)


# ============================================================
# MAIN
# ============================================================

def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(-2.0, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # 1. Draw battery body
    a, b, top_y, bot_y = draw_cylinder(ax)

    # 2. Draw positive terminal cap
    draw_positive_cap(ax, a, b, top_y)

    # 3. Draw internal thermal network
    tc_x, tc_y, ts_x, ts_y = draw_thermal_network(ax)

    # 4. Draw annotations
    draw_annotations(ax, tc_x, tc_y, ts_x, ts_y)

    # 5. Title below figure
    ax.text(BAT_CX + 0.3, -2.95,
            "(a) Cylindrical Li-ion Battery\nPhysical Thermal Model",
            fontsize=12, fontweight="bold", ha="center", va="top", color="black")

    # Save
    out = "battery_physical_thermal_model.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"✓ Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
