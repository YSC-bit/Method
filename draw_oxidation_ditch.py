"""
draw_oxidation_ditch.py
=======================
绘制污水处理厂双氧化沟（Dual Oxidation Ditch）工艺流程图，用于 SCI 论文发表。

运行方式：
    python draw_oxidation_ditch.py

输出文件：
    oxidation_ditch_process.tiff  —— 300 DPI, LZW 压缩（投稿用）
    oxidation_ditch_process.png   —— 300 DPI PNG（预览/草稿用）
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ─────────────────────── CONFIG ─────────────────────────────────
FIG_W = 20      # inches  (landscape)
FIG_H = 11      # inches
DPI   = 300
FS    = 9.0     # base font size (pt)
# ────────────────────────────────────────────────────────────────

# Register Noto CJK font for Chinese character support
import matplotlib.font_manager as _fm
_cjk_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
import os as _os
if _os.path.exists(_cjk_path):
    _fm.fontManager.addfont(_cjk_path)
    _cjk_font = "Noto Sans CJK JP"
else:
    _cjk_font = "DejaVu Sans"

matplotlib.rcParams.update({
    "font.family": _cjk_font,
    "font.size":   FS,
})

# ── colour palette ───────────────────────────────────────────────
C = dict(
    inflow   = "#C8E6FA",   # light blue
    screen   = "#D0ECE7",   # green-teal
    settle   = "#FFF9C4",   # light yellow
    bio      = "#FFE0B2",   # orange-buff  (biological tanks)
    sludge   = "#E8D5C4",   # tan
    outflow  = "#B2DFDB",   # teal
    return_  = "#F3E5F5",   # lavender
    excess   = "#FCE4EC",   # pink
    border   = "#455A64",
    arrow    = "#263238",
    label    = "#1A237E",
)
ALPHA = 0.92
LW    = 1.0

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

# ─── helpers ────────────────────────────────────────────────────

def box(cx, cy, w, h, label, sublabel="",
        color=C["screen"], bold=False,
        fs=None, zorder=4):
    fs = fs or FS
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.008",
        linewidth=LW, edgecolor=C["border"],
        facecolor=color, alpha=ALPHA, zorder=zorder
    )
    ax.add_patch(rect)
    fw = "bold" if bold else "normal"
    ty = cy + (h * 0.13 if sublabel else 0)
    ax.text(cx, ty, label, ha="center", va="center",
            fontsize=fs, fontweight=fw,
            color=C["label"] if bold else "#1a1a1a", zorder=zorder + 1)
    if sublabel:
        ax.text(cx, cy - h * 0.22, sublabel,
                ha="center", va="center",
                fontsize=fs - 1.5, color="#555555",
                style="italic", zorder=zorder + 1)


def oval(cx, cy, rx, ry, label, color=C["screen"], bold=False, fs=None, zorder=4):
    fs = fs or FS
    ell = mpatches.Ellipse((cx, cy), 2 * rx, 2 * ry,
                            linewidth=LW, edgecolor=C["border"],
                            facecolor=color, alpha=ALPHA, zorder=zorder)
    ax.add_patch(ell)
    fw = "bold" if bold else "normal"
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fs, fontweight=fw,
            color=C["label"] if bold else "#1a1a1a", zorder=zorder + 1)


def diamond(cx, cy, w, h, label, color=C["settle"], fs=None, zorder=4):
    fs = fs or FS
    dx, dy = w / 2, h / 2
    pts = np.array([
        [cx,      cy + dy],
        [cx + dx, cy     ],
        [cx,      cy - dy],
        [cx - dx, cy     ],
    ])
    poly = plt.Polygon(pts, closed=True,
                       linewidth=LW, edgecolor=C["border"],
                       facecolor=color, alpha=ALPHA, zorder=zorder)
    ax.add_patch(poly)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fs - 0.5, color="#1a1a1a", zorder=zorder + 1)


def arrow(x0, y0, x1, y1, color=C["arrow"], lw=1.3,
          rad=0.0, style="-|>", zorder=3, label="", label_side="top"):
    ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=style,
                    color=color, lw=lw,
                    connectionstyle=f"arc3,rad={rad}",
                    mutation_scale=9,
                ),
                zorder=zorder)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dy = 0.013 if label_side == "top" else -0.016
        ax.text(mx, my + dy, label,
                ha="center", va="center",
                fontsize=FS - 2, color="#555555",
                style="italic", zorder=zorder + 1)


def dashed_arrow(x0, y0, x1, y1, color="#888888", lw=1.0,
                 rad=0.0, label="", label_side="top", zorder=3):
    ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color, lw=lw,
                    linestyle="dashed",
                    connectionstyle=f"arc3,rad={rad}",
                    mutation_scale=8,
                ),
                zorder=zorder)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dy = 0.013 if label_side == "top" else -0.016
        ax.text(mx, my + dy, label,
                ha="center", va="center",
                fontsize=FS - 2, color="#777777",
                style="italic", zorder=zorder + 1)


def brace_label(x, y, text, color="#333333", fs=None):
    fs = fs or FS - 1
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, color=color,
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", edgecolor="none", alpha=0.8),
            zorder=10)


# ════════════════════════════════════════════════════════════════
#  LAYOUT  (all coordinates in [0, 1] normalised figure space)
# ════════════════════════════════════════════════════════════════
#
#  Row-Y positions (top → bottom)
#  y_top  = 0.87   (title band)
#  y1     = 0.73   main process row 1  (pretreatment)
#  y2     = 0.47   main process row 2  (biological)
#  y3     = 0.22   sludge / return row
#
#  Columns (left → right): 0.06, 0.18, 0.30, 0.42, 0.54, 0.66, 0.78, 0.90

y1, y2, y3 = 0.73, 0.47, 0.21

BW, BH = 0.105, 0.085   # standard box width / height
SW, SH = 0.090, 0.075   # small box

# ── title ───────────────────────────────────────────────────────
ax.text(0.50, 0.95,
        "Dual Oxidation Ditch Wastewater Treatment Process",
        ha="center", va="center",
        fontsize=FS + 3, fontweight="bold", color="#0D47A1", zorder=5)
ax.text(0.50, 0.90,
        "双氧化沟污水处理工艺流程图",
        ha="center", va="center",
        fontsize=FS + 1, color="#1565C0", zorder=5)

# ────────────────────────────────────────────────────────────────
#  PRETREATMENT  ROW  (y1)
# ────────────────────────────────────────────────────────────────
#  Inflow → Coarse Screen → Grit Chamber → Fine Screen → Regulating Tank
#               → Primary Settling Tank   (optional, shown as dashed)

x_inflow = 0.05
x_cs     = 0.155
x_gc     = 0.275
x_fs     = 0.395
x_reg    = 0.515
x_pst    = 0.640

oval(x_inflow, y1, 0.042, 0.042,
     "进水\nInfluent", color=C["inflow"], bold=True)

box(x_cs,  y1, BW, BH,
    "粗格栅\nCoarse Screen", color=C["screen"])

box(x_gc,  y1, BW, BH,
    "沉砂池\nGrit Chamber", color=C["screen"])

box(x_fs,  y1, BW, BH,
    "细格栅\nFine Screen", color=C["screen"])

box(x_reg, y1, BW, BH,
    "调节池\nEqualization\nTank", color=C["screen"])

box(x_pst, y1, BW, BH,
    "初沉池\nPrimary\nSettling Tank",
    color=C["settle"])

# arrows — pretreatment row
arrow(x_inflow + 0.043, y1,  x_cs  - BW/2, y1)
arrow(x_cs  + BW/2,     y1,  x_gc  - BW/2, y1)
arrow(x_gc  + BW/2,     y1,  x_fs  - BW/2, y1)
arrow(x_fs  + BW/2,     y1,  x_reg - BW/2, y1)
arrow(x_reg + BW/2,     y1,  x_pst - BW/2, y1)

# ── pumping station between reg tank and PST (small sub-box) ───
x_pump = (x_reg + BW/2 + x_pst - BW/2) / 2
box(x_pump, y1 - 0.02, 0.060, 0.045,
    "提升泵站\nLift Pump", color=C["screen"], fs=FS - 1)

# ────────────────────────────────────────────────────────────────
#  BIOLOGICAL  ROW  (y2)
# ────────────────────────────────────────────────────────────────
#  Primary Settling → Anoxic Zone → Oxidation Ditch 1 (Aerobic) →
#  Oxidation Ditch 2 (Anaerobic) → Secondary Settling Tank → Disinfection → Outflow

x_anox = 0.155
x_od1  = 0.310
x_od2  = 0.490
x_sst  = 0.660
x_dis  = 0.790
x_out  = 0.920

# from PST down to anoxic zone
arrow(x_pst, y1 - BH/2, x_pst, (y1 + y2) / 2,  label="")
arrow(x_pst, (y1 + y2) / 2, x_anox - BW/2 - 0.01, (y1 + y2) / 2)
arrow(x_anox - BW/2 - 0.01, (y1 + y2) / 2, x_anox, y2 + BH/2)

box(x_anox, y2, BW + 0.01, BH + 0.01,
    "缺氧区\nAnoxic Zone", color=C["bio"], bold=True)

# Oxidation Ditch 1
box(x_od1, y2, BW + 0.03, BH + 0.02,
    "氧化沟 1\nOxidation Ditch 1",
    sublabel="（好氧 Aerobic）",
    color=C["bio"], bold=True, fs=FS)

# Oxidation Ditch 2
box(x_od2, y2, BW + 0.03, BH + 0.02,
    "氧化沟 2\nOxidation Ditch 2",
    sublabel="（兼氧 Anoxic/Aerobic）",
    color=C["bio"], bold=True, fs=FS)

box(x_sst, y2, BW, BH,
    "二沉池\nSecondary\nSettling Tank",
    color=C["settle"])

box(x_dis, y2, BW, BH,
    "消毒池\nDisinfection\nTank",
    color=C["outflow"])

oval(x_out, y2, 0.042, 0.042,
     "出水\nEffluent", color=C["outflow"], bold=True)

# arrows — biological row
arrow(x_anox + (BW + 0.01) / 2, y2, x_od1 - (BW + 0.03) / 2, y2,
      label="内回流\nInternal Recycle", label_side="top")
arrow(x_od1  + (BW + 0.03) / 2, y2, x_od2 - (BW + 0.03) / 2, y2)
arrow(x_od2  + (BW + 0.03) / 2, y2, x_sst - BW/2, y2)
arrow(x_sst  + BW/2,            y2, x_dis - BW/2, y2)
arrow(x_dis  + BW/2,            y2, x_out - 0.043, y2)

# ── internal recycle arrow (OD1 → Anoxic) curved ──────────────
arrow(x_od1, y2 + (BH + 0.02) / 2 + 0.01,
      x_anox, y2 + (BH + 0.01) / 2 + 0.01,
      rad=-0.30, color="#1565C0",
      label="内回流\nNitrate Recycle", label_side="top")

# ────────────────────────────────────────────────────────────────
#  SLUDGE  ROW  (y3)
# ────────────────────────────────────────────────────────────────
x_st   = 0.310   # sludge thickener
x_dw   = 0.490   # dewatering
x_disp = 0.660   # final disposal

box(x_st,   y3, BW, BH,
    "污泥浓缩池\nSludge\nThickener", color=C["sludge"])

box(x_dw,   y3, BW, BH,
    "污泥脱水\nSludge\nDewatering", color=C["sludge"])

box(x_disp, y3, BW, BH,
    "污泥处置\nSludge\nDisposal", color=C["sludge"])

# sludge flow from SST down to thickener
arrow(x_sst, y2 - BH/2, x_sst, (y2 + y3) / 2,
      label="剩余污泥\nExcess Sludge", label_side="top")
arrow(x_sst, (y2 + y3) / 2, x_st + BW/2, (y2 + y3) / 2)
arrow(x_st + BW/2, (y2 + y3) / 2, x_st, y3 + BH/2)
arrow(x_st  + BW/2, y3, x_dw  - BW/2, y3)
arrow(x_dw  + BW/2, y3, x_disp - BW/2, y3)

# return sludge from SST back to Anoxic zone
dashed_arrow(x_sst, y2 - BH/2 - 0.005,
             x_anox, y2 - (BH + 0.01) / 2 - 0.005,
             rad=0.25,
             label="回流污泥  Return Sludge",
             label_side="top", color="#1B5E20")

# ── aeration symbol (rotary brush or disc aerator) ──────────────
def aeration_symbol(cx, cy, color="#1565C0"):
    """Draw a small stylised aerator symbol."""
    circ = plt.Circle((cx, cy), 0.018,
                      linewidth=0.8, edgecolor=color,
                      facecolor="none", alpha=0.7, zorder=6)
    ax.add_patch(circ)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        dx = 0.022 * np.cos(angle)
        dy = 0.022 * np.sin(angle) * (FIG_W / FIG_H)   # aspect correction
        ax.annotate("",
                    xy=(cx + dx, cy + dy),
                    xytext=(cx, cy),
                    arrowprops=dict(
                        arrowstyle="-|>", color=color, lw=0.6,
                        mutation_scale=5),
                    zorder=6)
    ax.text(cx, cy - 0.045, "曝气\nAerator",
            ha="center", va="top",
            fontsize=FS - 2.5, color=color, zorder=7)

aeration_symbol(x_od1, y2 - 0.005)
aeration_symbol(x_od2, y2 - 0.005)

# ── legend ──────────────────────────────────────────────────────
legend_items = [
    (C["inflow"],  "进/出水  In/Effluent"),
    (C["screen"],  "预处理单元  Pretreatment"),
    (C["settle"],  "沉淀单元  Settling"),
    (C["bio"],     "生物处理单元  Biological Treatment"),
    (C["sludge"],  "污泥处理单元  Sludge Treatment"),
    (C["outflow"], "消毒/出水  Disinfection/Effluent"),
]
lx, ly = 0.01, 0.38
for color, lbl in legend_items:
    rect = FancyBboxPatch((lx, ly - 0.014), 0.025, 0.025,
                          boxstyle="round,pad=0.003",
                          linewidth=0.7, edgecolor=C["border"],
                          facecolor=color, alpha=ALPHA, zorder=8)
    ax.add_patch(rect)
    ax.text(lx + 0.030, ly, lbl,
            ha="left", va="center",
            fontsize=FS - 1.5, color="#222222", zorder=9)
    ly -= 0.040

# solid/dashed line legend
ax.annotate("",
            xy=(lx + 0.025, 0.12), xytext=(lx, 0.12),
            arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                            lw=1.2, mutation_scale=8), zorder=8)
ax.text(lx + 0.030, 0.12, "水流方向  Water Flow",
        ha="left", va="center", fontsize=FS - 1.5, zorder=9)

ax.annotate("",
            xy=(lx + 0.025, 0.09), xytext=(lx, 0.09),
            arrowprops=dict(arrowstyle="-|>", color="#1B5E20",
                            lw=1.0, linestyle="dashed",
                            mutation_scale=8), zorder=8)
ax.text(lx + 0.030, 0.09, "回流污泥  Return Sludge",
        ha="left", va="center", fontsize=FS - 1.5, zorder=9)

# ── border frame ─────────────────────────────────────────────────
frame = FancyBboxPatch((0.005, 0.01), 0.990, 0.980,
                       boxstyle="round,pad=0.005",
                       linewidth=1.5, edgecolor="#455A64",
                       facecolor="none", zorder=1)
ax.add_patch(frame)

# ── footer ───────────────────────────────────────────────────────
ax.text(0.99, 0.015,
        "Fig. Dual Oxidation Ditch Wastewater Treatment Process Flow Diagram",
        ha="right", va="bottom",
        fontsize=FS - 2, color="#888888", style="italic", zorder=5)

# ════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)

out_tiff = "oxidation_ditch_process.tiff"
out_png  = "oxidation_ditch_process.png"

fig.savefig(out_tiff, dpi=DPI, format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches="tight")
fig.savefig(out_png,  dpi=DPI, format="png",
            bbox_inches="tight")

print(f"Saved: {out_tiff}")
print(f"Saved: {out_png}")
plt.close(fig)
