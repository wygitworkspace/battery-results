"""
18650 Cylindrical Li-ion Battery Two-Node Thermal Model & Thermal Equivalent Circuit
=====================================================================================
Uses Microsoft Visio (via COM automation) to draw the figure.

Requirements:
    - Windows OS
    - Microsoft Visio installed
    - pip install pywin32

Run:
    python draw_visio_thermal.py

Output:
    cylindrical_battery_two_node_thermal_model.vsdx
    cylindrical_battery_two_node_thermal_model.png
"""

import win32com.client
import os
import sys
import math

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_VSDX = "cylindrical_battery_two_node_thermal_model.vsdx"
OUTPUT_PNG = "cylindrical_battery_two_node_thermal_model.png"

# Page size (inches)
PAGE_W = 11.0
PAGE_H = 7.0

# Colors (RGB as hex integer for Visio: 0xBBGGRR format)
def rgb(r, g, b):
    """Convert RGB (0-255) to Visio color integer (BGR)."""
    return b * 65536 + g * 256 + r

COLOR_PINK_SHELL = rgb(255, 180, 200)
COLOR_PINK_LIGHT = rgb(255, 220, 230)
COLOR_METAL = rgb(190, 190, 195)
COLOR_METAL_DARK = rgb(130, 130, 135)
COLOR_INSUL_BLACK = rgb(30, 30, 30)
COLOR_TC = rgb(128, 0, 128)       # purple
COLOR_TS = rgb(220, 25, 25)       # red
COLOR_RC = rgb(255, 195, 210)     # light pink
COLOR_RS = rgb(255, 145, 40)      # orange
COLOR_WIRE = rgb(0, 0, 0)
COLOR_ARROW_BLUE = rgb(90, 155, 235)
COLOR_WHITE = rgb(255, 255, 255)
COLOR_BLACK = rgb(0, 0, 0)

# Visio constants
visLayerMember = 0
visSectionObject = 1
visRowFill = 3
visRowLine = 2


# ============================================================
# VISIO HELPER FUNCTIONS
# ============================================================

class VisioDrawer:
    """Wrapper for Visio COM automation."""

    def __init__(self):
        print("Starting Microsoft Visio...")
        self.app = win32com.client.Dispatch("Visio.Application")
        self.app.Visible = True
        self.doc = self.app.Documents.Add("")
        self.page = self.doc.Pages(1)
        self.page.PageSheet.CellsU("PageWidth").FormulaU = f"{PAGE_W} in"
        self.page.PageSheet.CellsU("PageHeight").FormulaU = f"{PAGE_H} in"
        print(f"Page created: {PAGE_W}\" x {PAGE_H}\"")

    def draw_rect(self, x1, y1, x2, y2, fill_color=None, line_color=COLOR_BLACK,
                  line_weight=0.01, opacity=1.0, rounding=0):
        """Draw a rectangle. Coords in inches from bottom-left."""
        shp = self.page.DrawRectangle(x1, y1, x2, y2)
        if fill_color is not None:
            shp.CellsU("FillForegnd").FormulaU = f"RGB({fill_color & 0xFF},{(fill_color >> 8) & 0xFF},{(fill_color >> 16) & 0xFF})"
        else:
            shp.CellsU("FillPattern").FormulaU = "0"
        shp.CellsU("LineColor").FormulaU = f"RGB({line_color & 0xFF},{(line_color >> 8) & 0xFF},{(line_color >> 16) & 0xFF})"
        shp.CellsU("LineWeight").FormulaU = f"{line_weight} in"
        if opacity < 1.0:
            shp.CellsU("FillForegndTrans").FormulaU = f"{(1 - opacity) * 100}%"
        if rounding > 0:
            shp.CellsU("Rounding").FormulaU = f"{rounding} in"
        return shp

    def draw_ellipse(self, cx, cy, rx, ry, fill_color=None, line_color=COLOR_BLACK,
                     line_weight=0.008, opacity=1.0):
        """Draw an ellipse centered at (cx, cy)."""
        shp = self.page.DrawOval(cx - rx, cy - ry, cx + rx, cy + ry)
        if fill_color is not None:
            shp.CellsU("FillForegnd").FormulaU = f"RGB({fill_color & 0xFF},{(fill_color >> 8) & 0xFF},{(fill_color >> 16) & 0xFF})"
        else:
            shp.CellsU("FillPattern").FormulaU = "0"
        shp.CellsU("LineColor").FormulaU = f"RGB({line_color & 0xFF},{(line_color >> 8) & 0xFF},{(line_color >> 16) & 0xFF})"
        shp.CellsU("LineWeight").FormulaU = f"{line_weight} in"
        if opacity < 1.0:
            shp.CellsU("FillForegndTrans").FormulaU = f"{(1 - opacity) * 100}%"
        return shp

    def draw_line(self, x1, y1, x2, y2, color=COLOR_BLACK, weight=0.015, dash=0):
        """Draw a line segment."""
        shp = self.page.DrawLine(x1, y1, x2, y2)
        shp.CellsU("LineColor").FormulaU = f"RGB({color & 0xFF},{(color >> 8) & 0xFF},{(color >> 16) & 0xFF})"
        shp.CellsU("LineWeight").FormulaU = f"{weight} in"
        if dash:
            shp.CellsU("LinePattern").FormulaU = str(dash)
        return shp

    def draw_text(self, cx, cy, text, size=10, bold=False, italic=False,
                  color=COLOR_BLACK, font="Times New Roman"):
        """Place a text box centered at (cx, cy)."""
        # Create a small rectangle as text container (no fill, no line)
        w, h = len(text) * size * 0.009 + 0.3, size * 0.022 + 0.1
        shp = self.page.DrawRectangle(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        shp.CellsU("FillPattern").FormulaU = "0"
        shp.CellsU("LinePattern").FormulaU = "0"
        shp.Text = text
        shp.CellsU("Char.Size").FormulaU = f"{size} pt"
        shp.CellsU("Char.Color").FormulaU = f"RGB({color & 0xFF},{(color >> 8) & 0xFF},{(color >> 16) & 0xFF})"
        if bold:
            shp.CellsU("Char.Style").FormulaU = "1"
        if italic:
            shp.CellsU("Char.Style").FormulaU = "2"
        if bold and italic:
            shp.CellsU("Char.Style").FormulaU = "3"
        return shp

    def draw_arrow(self, x1, y1, x2, y2, color=COLOR_BLACK, weight=0.03):
        """Draw a line with arrow end."""
        shp = self.draw_line(x1, y1, x2, y2, color=color, weight=weight)
        shp.CellsU("EndArrow").FormulaU = "4"
        shp.CellsU("EndArrowSize").FormulaU = "4"
        return shp

    def save_and_export(self):
        """Save as .vsdx and export as PNG."""
        vsdx_path = os.path.abspath(OUTPUT_VSDX)
        png_path = os.path.abspath(OUTPUT_PNG)

        self.doc.SaveAs(vsdx_path)
        print(f"✓ Saved: {vsdx_path}")

        # Export to PNG
        self.page.Export(png_path)
        print(f"✓ Exported: {png_path}")

    def close(self):
        """Close Visio (optional - leave open for manual editing)."""
        # Don't close - leave Visio open so user can edit
        print("\nVisio is left open for manual editing.")
        print("You can modify shapes, colors, positions directly in Visio.")


# ============================================================
# DRAWING THE BATTERY (LEFT PANEL)
# ============================================================

def draw_battery_body(v, ox, oy):
    """
    Draw the cylindrical battery at offset (ox, oy).
    Returns key coordinates for connecting the thermal network.
    """
    # Battery dimensions
    bw = 1.2   # width
    bh = 2.8   # height
    ellipse_ry = 0.12  # perspective ellipse height

    # Main body (pink rectangle with transparency)
    v.draw_rect(ox - bw / 2, oy - bh / 2, ox + bw / 2, oy + bh / 2,
                fill_color=COLOR_PINK_SHELL, line_color=COLOR_PINK_SHELL,
                line_weight=0.012, opacity=0.5)

    # Left and right outlines
    v.draw_line(ox - bw / 2, oy - bh / 2, ox - bw / 2, oy + bh / 2,
                color=rgb(160, 80, 100), weight=0.015)
    v.draw_line(ox + bw / 2, oy - bh / 2, ox + bw / 2, oy + bh / 2,
                color=rgb(160, 80, 100), weight=0.015)

    # Bottom ellipse
    v.draw_ellipse(ox, oy - bh / 2, bw / 2, ellipse_ry,
                   fill_color=COLOR_PINK_SHELL, line_color=rgb(160, 80, 100),
                   line_weight=0.012, opacity=0.6)

    # Top ellipse
    v.draw_ellipse(ox, oy + bh / 2, bw / 2, ellipse_ry,
                   fill_color=COLOR_PINK_LIGHT, line_color=rgb(160, 80, 100),
                   line_weight=0.012, opacity=0.85)

    # Core region (lighter inner rectangle)
    cw = bw * 0.55
    ch = bh * 0.85
    v.draw_rect(ox - cw / 2, oy - ch / 2, ox + cw / 2, oy + ch / 2,
                fill_color=COLOR_PINK_LIGHT, line_color=COLOR_PINK_LIGHT,
                line_weight=0, opacity=0.3)

    # --- Positive terminal cap ---
    cap_h = 0.14
    cap_w = bw * 0.88
    cap_base_y = oy + bh / 2 + ellipse_ry * 0.3

    # Cap side wall
    v.draw_rect(ox - cap_w / 2, cap_base_y, ox + cap_w / 2, cap_base_y + cap_h,
                fill_color=COLOR_METAL, line_color=COLOR_METAL_DARK, line_weight=0.012)

    # Cap top ellipse (outer ring)
    v.draw_ellipse(ox, cap_base_y + cap_h, cap_w / 2, ellipse_ry * 0.85,
                   fill_color=COLOR_METAL, line_color=COLOR_METAL_DARK, line_weight=0.012)

    # Insulation ring
    ins_w = cap_w * 0.68
    v.draw_ellipse(ox, cap_base_y + cap_h, ins_w / 2, ellipse_ry * 0.60,
                   fill_color=COLOR_INSUL_BLACK, line_color=COLOR_BLACK, line_weight=0.008)

    # Central bump
    bump_w = cap_w * 0.35
    v.draw_ellipse(ox, cap_base_y + cap_h + 0.02, bump_w / 2, ellipse_ry * 0.40,
                   fill_color=COLOR_METAL, line_color=COLOR_METAL_DARK, line_weight=0.008)

    # "+" text
    v.draw_text(ox, cap_base_y + cap_h + ellipse_ry + 0.12, "+", size=11, bold=True)

    return {
        "cx": ox, "cy": oy,
        "top": oy + bh / 2,
        "bot": oy - bh / 2,
        "left": ox - bw / 2,
        "right": ox + bw / 2,
    }


def draw_internal_thermal_network(v, bat):
    """Draw the two-node thermal model inside the battery."""
    ox = bat["cx"]
    oy = bat["cy"]

    # Key positions
    qx, qy = ox - 0.2, oy - 0.7
    tcx, tcy = ox - 0.2, oy + 0.3
    tsx, tsy = ox + 0.35, oy + 0.3
    gy = oy - 1.2  # ground

    # --- Heat source Qe ---
    r = 0.12
    v.draw_ellipse(qx, qy, r, r, fill_color=COLOR_WHITE, line_color=COLOR_BLACK, line_weight=0.015)
    # sine wave approximation (just a "~" text)
    v.draw_text(qx, qy, "~", size=12, bold=True)
    v.draw_text(qx - 0.30, qy, "Qe", size=9, italic=True)

    # Wire Qe top to Tc
    v.draw_line(qx, qy + r, qx, tcy, color=COLOR_WIRE, weight=0.015)
    # Wire Qe bottom to ground
    v.draw_line(qx, qy - r, qx, gy, color=COLOR_WIRE, weight=0.015)

    # --- Tc node ---
    v.draw_ellipse(tcx, tcy, 0.05, 0.05, fill_color=COLOR_TC, line_color=COLOR_BLACK, line_weight=0.008)
    v.draw_text(tcx, tcy + 0.16, "Tc", size=9, italic=True, color=COLOR_TC)

    # --- Ts node ---
    v.draw_ellipse(tsx, tsy, 0.05, 0.05, fill_color=COLOR_TS, line_color=COLOR_BLACK, line_weight=0.008)
    v.draw_text(tsx, tsy + 0.16, "Ts", size=9, italic=True, color=COLOR_TS)

    # --- Rc between Tc and Ts ---
    rc_len = 0.25
    rcx = (tcx + tsx) / 2
    v.draw_line(tcx, tcy, rcx - rc_len / 2, tcy, color=COLOR_WIRE, weight=0.015)
    v.draw_rect(rcx - rc_len / 2, tcy - 0.05, rcx + rc_len / 2, tcy + 0.05,
                fill_color=COLOR_RC, line_color=COLOR_BLACK, line_weight=0.012, rounding=0.02)
    v.draw_line(rcx + rc_len / 2, tcy, tsx, tsy, color=COLOR_WIRE, weight=0.015)
    v.draw_text(rcx, tcy + 0.13, "Rc", size=9, italic=True)

    # --- Cc below Tc ---
    cc_top = tcy - 0.15
    cc_bot = gy + 0.10
    cc_mid = (cc_top + cc_bot) / 2
    v.draw_line(tcx, tcy, tcx, cc_top, color=COLOR_WIRE, weight=0.015)
    # capacitor plates
    pw = 0.14
    v.draw_line(tcx - pw, cc_mid + 0.04, tcx + pw, cc_mid + 0.04, color=COLOR_BLACK, weight=0.025)
    v.draw_line(tcx - pw, cc_mid - 0.04, tcx + pw, cc_mid - 0.04, color=COLOR_BLACK, weight=0.025)
    v.draw_line(tcx, cc_mid + 0.04, tcx, cc_top, color=COLOR_WIRE, weight=0.015)
    v.draw_line(tcx, cc_mid - 0.04, tcx, gy, color=COLOR_WIRE, weight=0.015)
    v.draw_text(tcx - 0.20, cc_mid, "Cc", size=9, italic=True)

    # --- Rs below Ts ---
    rs_top = tsy - 0.15
    rs_bot = gy + 0.40
    rs_mid = (rs_top + rs_bot) / 2
    rs_len = 0.25
    v.draw_line(tsx, tsy, tsx, rs_mid + rs_len / 2, color=COLOR_WIRE, weight=0.015)
    v.draw_rect(tsx - 0.05, rs_mid - rs_len / 2, tsx + 0.05, rs_mid + rs_len / 2,
                fill_color=COLOR_RS, line_color=COLOR_BLACK, line_weight=0.012, rounding=0.02)
    v.draw_line(tsx, rs_mid - rs_len / 2, tsx, gy, color=COLOR_WIRE, weight=0.015)
    v.draw_text(tsx + 0.18, rs_mid, "Rs", size=9, italic=True)

    # --- Cs parallel to Rs ---
    csx = tsx + 0.35
    v.draw_line(tsx, tsy, csx, tsy, color=COLOR_WIRE, weight=0.012)
    cs_mid = (tsy - 0.15 + gy + 0.10) / 2
    v.draw_line(csx, tsy, csx, cs_mid + 0.04, color=COLOR_WIRE, weight=0.015)
    v.draw_line(csx - pw, cs_mid + 0.04, csx + pw, cs_mid + 0.04, color=COLOR_BLACK, weight=0.025)
    v.draw_line(csx - pw, cs_mid - 0.04, csx + pw, cs_mid - 0.04, color=COLOR_BLACK, weight=0.025)
    v.draw_line(csx, cs_mid - 0.04, csx, gy, color=COLOR_WIRE, weight=0.015)
    v.draw_text(csx + 0.20, cs_mid, "Cs", size=9, italic=True)

    # --- Ground rail ---
    v.draw_line(qx - 0.08, gy, csx + 0.08, gy, color=COLOR_WIRE, weight=0.018)

    # Ground symbol
    gx = (tcx + tsx) / 2
    gs = 0.10
    v.draw_line(gx, gy, gx, gy - gs * 0.4, color=COLOR_WIRE, weight=0.015)
    v.draw_line(gx - gs, gy - gs * 0.4, gx + gs, gy - gs * 0.4, color=COLOR_WIRE, weight=0.015)
    v.draw_line(gx - gs * 0.6, gy - gs * 0.7, gx + gs * 0.6, gy - gs * 0.7, color=COLOR_WIRE, weight=0.015)
    v.draw_line(gx - gs * 0.25, gy - gs, gx + gs * 0.25, gy - gs, color=COLOR_WIRE, weight=0.015)
    v.draw_text(gx, gy - gs - 0.15, "Ta", size=9, italic=True)

    # --- Annotations ---
    ann_x = bat["right"] + 0.55
    # Tc annotation
    v.draw_line(tcx + 0.08, tcy, ann_x - 0.05, tcy, color=COLOR_TC, weight=0.008, dash=2)
    v.draw_text(ann_x + 0.45, tcy + 0.08, "Internal Temperature", size=8, bold=True, color=COLOR_TC)
    v.draw_text(ann_x + 0.15, tcy - 0.10, "Tc", size=9, italic=True, color=COLOR_BLACK)

    # Ts annotation
    ann_ts_y = tsy - 0.40
    v.draw_line(tsx + 0.08, tsy, tsx + 0.08, ann_ts_y, color=COLOR_TS, weight=0.008, dash=2)
    v.draw_line(tsx + 0.08, ann_ts_y, ann_x - 0.05, ann_ts_y, color=COLOR_TS, weight=0.008, dash=2)
    v.draw_text(ann_x + 0.45, ann_ts_y + 0.08, "Surface Temperature", size=8, bold=True, color=COLOR_TS)
    v.draw_text(ann_x + 0.15, ann_ts_y - 0.10, "Ts", size=9, italic=True, color=COLOR_BLACK)


# ============================================================
# RIGHT PANEL: Equivalent circuit
# ============================================================

def draw_equivalent_circuit(v, ox, oy):
    """Draw the complete two-node thermal equivalent circuit."""
    # Layout
    lx = ox - 1.0       # left rail
    rx = ox + 1.8       # right rail
    ty = oy + 1.0       # top rail
    by = oy - 0.8       # bottom rail
    tc_x = ox - 0.3
    ts_x = ox + 0.8

    # --- Rails ---
    v.draw_line(lx, ty, rx, ty, color=COLOR_WIRE, weight=0.02)
    v.draw_line(lx, by, rx, by, color=COLOR_WIRE, weight=0.02)

    # --- Qe heat source (left) ---
    qm_y = (ty + by) / 2
    r = 0.15
    v.draw_line(lx, ty, lx, qm_y + r, color=COLOR_WIRE, weight=0.018)
    v.draw_line(lx, qm_y - r, lx, by, color=COLOR_WIRE, weight=0.018)
    v.draw_ellipse(lx, qm_y, r, r, fill_color=COLOR_WHITE, line_color=COLOR_BLACK, line_weight=0.018)
    v.draw_text(lx, qm_y, "~", size=14, bold=True)
    v.draw_text(lx - 0.30, qm_y, "Qe", size=10, italic=True)

    # --- Tc node ---
    v.draw_ellipse(tc_x, ty, 0.06, 0.06, fill_color=COLOR_TC, line_color=COLOR_BLACK, line_weight=0.01)
    v.draw_text(tc_x, ty + 0.18, "Tc", size=10, italic=True, color=COLOR_TC)

    # --- Rc between Tc and Ts ---
    rc_len = 0.40
    rcx = (tc_x + ts_x) / 2
    v.draw_line(tc_x, ty, rcx - rc_len / 2, ty, color=COLOR_WIRE, weight=0.018)
    v.draw_rect(rcx - rc_len / 2, ty - 0.06, rcx + rc_len / 2, ty + 0.06,
                fill_color=COLOR_RC, line_color=COLOR_BLACK, line_weight=0.013, rounding=0.02)
    v.draw_line(rcx + rc_len / 2, ty, ts_x, ty, color=COLOR_WIRE, weight=0.018)
    v.draw_text(rcx, ty + 0.15, "Rc", size=10, italic=True)

    # --- Ts node ---
    v.draw_ellipse(ts_x, ty, 0.06, 0.06, fill_color=COLOR_TS, line_color=COLOR_BLACK, line_weight=0.01)
    v.draw_text(ts_x, ty + 0.18, "Ts", size=10, italic=True, color=COLOR_TS)

    # --- Rs between Ts and right ---
    rs_len = 0.40
    rsx = (ts_x + rx) / 2
    v.draw_line(ts_x, ty, rsx - rs_len / 2, ty, color=COLOR_WIRE, weight=0.018)
    v.draw_rect(rsx - rs_len / 2, ty - 0.06, rsx + rs_len / 2, ty + 0.06,
                fill_color=COLOR_RS, line_color=COLOR_BLACK, line_weight=0.013, rounding=0.02)
    v.draw_line(rsx + rs_len / 2, ty, rx, ty, color=COLOR_WIRE, weight=0.018)
    v.draw_text(rsx, ty + 0.15, "Rs", size=10, italic=True)

    # --- Right vertical wire ---
    v.draw_line(rx, ty, rx, by, color=COLOR_WIRE, weight=0.018)

    # --- Cc below Tc ---
    pw = 0.16
    cc_mid = (ty + by) / 2
    v.draw_line(tc_x, ty, tc_x, cc_mid + 0.05, color=COLOR_WIRE, weight=0.018)
    v.draw_line(tc_x - pw, cc_mid + 0.05, tc_x + pw, cc_mid + 0.05, color=COLOR_BLACK, weight=0.028)
    v.draw_line(tc_x - pw, cc_mid - 0.05, tc_x + pw, cc_mid - 0.05, color=COLOR_BLACK, weight=0.028)
    v.draw_line(tc_x, cc_mid - 0.05, tc_x, by, color=COLOR_WIRE, weight=0.018)
    v.draw_text(tc_x + 0.22, cc_mid, "Cc", size=10, italic=True)

    # --- Cs below Ts ---
    cs_mid = (ty + by) / 2
    v.draw_line(ts_x, ty, ts_x, cs_mid + 0.05, color=COLOR_WIRE, weight=0.018)
    v.draw_line(ts_x - pw, cs_mid + 0.05, ts_x + pw, cs_mid + 0.05, color=COLOR_BLACK, weight=0.028)
    v.draw_line(ts_x - pw, cs_mid - 0.05, ts_x + pw, cs_mid - 0.05, color=COLOR_BLACK, weight=0.028)
    v.draw_line(ts_x, cs_mid - 0.05, ts_x, by, color=COLOR_WIRE, weight=0.018)
    v.draw_text(ts_x + 0.22, cs_mid, "Cs", size=10, italic=True)

    # --- Ground symbol ---
    gx = (tc_x + ts_x) / 2
    gs = 0.12
    v.draw_line(gx, by, gx, by - gs * 0.4, color=COLOR_WIRE, weight=0.018)
    v.draw_line(gx - gs, by - gs * 0.4, gx + gs, by - gs * 0.4, color=COLOR_WIRE, weight=0.018)
    v.draw_line(gx - gs * 0.6, by - gs * 0.7, gx + gs * 0.6, by - gs * 0.7, color=COLOR_WIRE, weight=0.018)
    v.draw_line(gx - gs * 0.25, by - gs, gx + gs * 0.25, by - gs, color=COLOR_WIRE, weight=0.018)
    v.draw_text(gx, by - gs - 0.18, "Ta", size=10, italic=True)


# ============================================================
# MAIN
# ============================================================

def main():
    v = VisioDrawer()

    # === LEFT PANEL: Battery + internal thermal network ===
    bat = draw_battery_body(v, ox=2.8, oy=3.8)
    draw_internal_thermal_network(v, bat)

    # Subtitle (a)
    v.draw_text(2.8, 1.0,
                "(a) Cylindrical Li-ion Battery Physical Thermal Model",
                size=10, bold=True)

    # === Transition arrow ===
    v.draw_arrow(5.2, 3.8, 5.9, 3.8, color=COLOR_ARROW_BLUE, weight=0.04)

    # === RIGHT PANEL: Equivalent circuit ===
    draw_equivalent_circuit(v, ox=7.8, oy=3.8)

    # Subtitle (b)
    v.draw_text(7.8, 1.0,
                "(b) Complete Two-Node Thermal Equivalent Circuit",
                size=10, bold=True)

    # === Save and export ===
    v.save_and_export()
    v.close()

    print("\n" + "=" * 60)
    print("Done! You can now edit the diagram directly in Visio.")
    print(f"File: {os.path.abspath(OUTPUT_VSDX)}")
    print("=" * 60)


if __name__ == "__main__":
    if sys.platform != "win32":
        print("Error: This script requires Windows with Microsoft Visio installed.")
        sys.exit(1)
    main()
