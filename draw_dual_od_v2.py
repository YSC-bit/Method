"""
draw_dual_od_v2.py
==================
仿照 A²O-MBR 参考图风格，绘制双氧化沟污水处理工艺流程图。

工艺拓扑（与用户提供的原始流程图完全一致）：
  进水 → 预处理 → 厌氧区 → 缺氧区 → 氧化沟1 ↔(交替运行) → 氧化沟2
       → 二沉池 → 消毒池 → 出水
  二沉池 → 污泥回流 → 厌氧区（底部虚线）
  二沉池 → 剩余污泥 → 污泥处理（向下支路）

风格元素（参照 A²O-MBR 参考图）：
  - 浅绿色整体背景面板
  - 3-D 透视箱体（正面 + 顶面 + 右侧面阴影）
  - 绿色主流向实线箭头
  - 橙红色虚线污泥回流箭头（底部路由）
  - 青色虚线"交替运行"标注
  - 红色虚框"Operation condition"区 + Input / HRT 标注
  - 左侧城市图标（进水）/ 右侧自然图标（出水）
  - 曝气轮刷符号（OD1 & OD2）
  - 右侧工艺名称标牌

输出：
  dual_od_process.png   300 DPI
  dual_od_process.tiff  300 DPI, LZW
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import os

# ── CJK font ──────────────────────────────────────────────────────
import matplotlib.font_manager as _fm
_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_cjk):
    _fm.fontManager.addfont(_cjk)
    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 8.0

# ── canvas ────────────────────────────────────────────────────────
FIG_W, FIG_H = 26, 10
DPI = 300
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── overall background panel ──────────────────────────────────────
bg = FancyBboxPatch((0.25, 0.40), FIG_W - 0.5, FIG_H - 0.95,
                    boxstyle="round,pad=0.12",
                    linewidth=2, edgecolor="#81C784",
                    facecolor="#E8F5E9", alpha=0.97, zorder=1)
ax.add_patch(bg)

# ═════════════════════════════════════════════════════════════════
#  COLOUR HELPERS
# ═════════════════════════════════════════════════════════════════

def _h2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _lighten(c, f=0.35):
    r, g, b = _h2rgb(c) if isinstance(c, str) else c
    return (min(1, r + (1-r)*f), min(1, g + (1-g)*f), min(1, b + (1-b)*f))


def _darken(c, f=0.22):
    r, g, b = _h2rgb(c) if isinstance(c, str) else c
    return (r*(1-f), g*(1-f), b*(1-f))


# ═════════════════════════════════════════════════════════════════
#  3-D BOX
# ═════════════════════════════════════════════════════════════════

def box3d(cx, cy, w, h, dx=0.28, dy=0.20,
          face_color="#5B8FD4", lw=0.9,
          label="", sublabel="", fs=9.0, zorder=4):
    top_c  = _lighten(face_color, 0.40)
    side_c = _darken(face_color,  0.26)

    # front face
    ax.add_patch(MplPolygon([
        [cx-w/2, cy-h/2], [cx+w/2, cy-h/2],
        [cx+w/2, cy+h/2], [cx-w/2, cy+h/2],
    ], closed=True, facecolor=face_color, edgecolor="white",
       linewidth=lw, zorder=zorder, alpha=0.96))

    # top face
    ax.add_patch(MplPolygon([
        [cx-w/2,    cy+h/2],
        [cx+w/2,    cy+h/2],
        [cx+w/2+dx, cy+h/2+dy],
        [cx-w/2+dx, cy+h/2+dy],
    ], closed=True, facecolor=top_c, edgecolor="white",
       linewidth=lw, zorder=zorder, alpha=0.96))

    # right side face
    ax.add_patch(MplPolygon([
        [cx+w/2,    cy-h/2],
        [cx+w/2+dx, cy-h/2+dy],
        [cx+w/2+dx, cy+h/2+dy],
        [cx+w/2,    cy+h/2],
    ], closed=True, facecolor=side_c, edgecolor="white",
       linewidth=lw, zorder=zorder, alpha=0.96))

    ty = cy + (h * 0.12 if sublabel else 0)
    ax.text(cx, ty, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="white",
            zorder=zorder+2)
    if sublabel:
        ax.text(cx, cy - h * 0.22, sublabel,
                ha="center", va="center",
                fontsize=fs - 1.5, color="#F5F5F5",
                style="italic", zorder=zorder+2)


# ═════════════════════════════════════════════════════════════════
#  ICONS
# ═════════════════════════════════════════════════════════════════

def city_icon(cx, cy, s=1.0, z=5):
    """City / domestic wastewater icon."""
    ax.add_patch(mpatches.Rectangle(
        (cx-1.1*s, cy-0.55*s), 2.2*s, 0.18*s,
        facecolor="#A5D6A7", edgecolor="none", zorder=z))

    buildings = [(-0.80, 0.00, 0.38, 1.00),
                 (-0.30, 0.00, 0.32, 1.30),
                 ( 0.10, 0.00, 0.42, 0.80),
                 ( 0.62, 0.00, 0.30, 1.05)]
    cols = ["#78909C", "#90A4AE", "#546E7A", "#607D8B"]
    for (bx, by, bw, bh), c in zip(buildings, cols):
        ax.add_patch(mpatches.Rectangle(
            (cx+bx*s, cy+by*s-0.37*s), bw*s, bh*s,
            facecolor=c, edgecolor="white", linewidth=0.4, zorder=z))
        for wy in np.linspace(0.12, bh-0.15, 3):
            for wx in [0.07, bw-0.14]:
                ax.add_patch(mpatches.Rectangle(
                    (cx+(bx+wx)*s, cy+(by+wy)*s-0.37*s),
                    0.08*s, 0.12*s,
                    facecolor="#FFF9C4", edgecolor="none", zorder=z+1))

    ax.text(cx, cy-0.78*s, "Domestic\nwastewater",
            ha="center", va="top", fontsize=8, color="#37474F", zorder=z+1)


def nature_icon(cx, cy, s=1.0, z=5):
    """River / nature icon (effluent)."""
    ax.add_patch(Ellipse(
        (cx, cy-0.32*s), 2.0*s, 0.62*s,
        facecolor="#81D4FA", edgecolor="#29B6F6",
        linewidth=0.7, zorder=z, alpha=0.88))
    for pts, c in [
        ([[-0.90, 0.05], [-0.20, 0.92], [0.40, 0.05]], "#66BB6A"),
        ([[-0.05, 0.10], [ 0.60, 0.98], [1.10, 0.10]], "#4CAF50"),
    ]:
        ax.add_patch(MplPolygon([[cx+p[0]*s, cy+p[1]*s] for p in pts],
                                closed=True, facecolor=c, edgecolor="white",
                                linewidth=0.5, zorder=z))
    ax.text(cx, cy-0.76*s, "Effluent / 出水",
            ha="center", va="top", fontsize=8,
            color="#1B5E20", fontweight="bold", zorder=z+1)


# ── aerator symbol ────────────────────────────────────────────────

def aerator(cx, cy, r=0.28, color="#29B6F6", z=7):
    ax.add_patch(plt.Circle((cx, cy), r,
                            facecolor="#E3F2FD", edgecolor=color,
                            linewidth=0.8, zorder=z, alpha=0.90))
    for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
        ax.plot([cx, cx + r*0.70*np.cos(ang)],
                [cy, cy + r*0.70*np.sin(ang)],
                color=color, lw=0.9, zorder=z+1)
    ax.text(cx, cy - r - 0.18, "Aeration",
            ha="center", va="top", fontsize=6.0, color=color, zorder=z+1)


# ═════════════════════════════════════════════════════════════════
#  ARROW HELPERS
# ═════════════════════════════════════════════════════════════════

def arr_h(x0, x1, y, color="#2E7D32", lw=2.2,
          label="", lside="top", z=6):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=13), zorder=z)
    if label:
        dy = 0.18 if lside == "top" else -0.22
        ax.text((x0+x1)/2, y+dy, label,
                ha="center", va="bottom" if lside == "top" else "top",
                fontsize=7.5, color=color, fontweight="bold", zorder=z+1)


def routed_arrow(xs, ys, color="#E64A19", lw=1.8,
                 linestyle="--", label="", label_idx=None, z=6):
    """Multi-segment path with arrowhead at the last point."""
    ax.plot(xs, ys, color=color, lw=lw, linestyle=linestyle,
            solid_capstyle="round", zorder=z)
    ax.annotate("",
                xy=(xs[-1], ys[-1]),
                xytext=(xs[-2] + (xs[-1]-xs[-2])*0.55,
                        ys[-2] + (ys[-1]-ys[-2])*0.55),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=11), zorder=z)
    if label and label_idx is not None:
        i = label_idx
        mx = (xs[i] + xs[i+1]) / 2
        my = (ys[i] + ys[i+1]) / 2
        ax.text(mx, my, label,
                ha="center", va="center", fontsize=8,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", edgecolor="none", alpha=0.85),
                zorder=z+2)


# ═════════════════════════════════════════════════════════════════
#  LAYOUT PARAMETERS
# ═════════════════════════════════════════════════════════════════

y_mid  = 5.20      # vertical centre of tanks
bh     = 2.10      # tank height
bw_s   = 1.70      # small/medium tank width
bw_od  = 2.05      # oxidation ditch width
PX     = 0.28      # 3-D x-offset
PY     = 0.20      # 3-D y-offset
GAP    = 0.16      # gap between box right-edge (incl. 3D offset) and arrow start

# X centres  (spaced to fit all 8 process units + icons)
Xc = {
    "city"   :  1.40,
    "pre"    :  3.30,
    "anaero" :  5.30,   # Anaerobic zone
    "anoxic" :  7.25,   # Anoxic zone
    "od1"    :  9.50,   # Oxidation Ditch 1
    "od2"    : 11.80,   # Oxidation Ditch 2
    "sst"    : 14.00,   # Secondary Clarifier
    "dis"    : 16.10,   # Disinfection
    "nature" : 18.10,   # effluent icon
    "sludge" : 14.00,   # Sludge treatment (below SST)
}

# Badge centre (right side, clear of effluent icon)
badge_cx = FIG_W - 1.90
badge_cy = y_mid + 0.30

# ═════════════════════════════════════════════════════════════════
#  ICONS
# ═════════════════════════════════════════════════════════════════
city_icon(Xc["city"],   y_mid, s=0.78)
nature_icon(Xc["nature"], y_mid, s=0.78)

# ═════════════════════════════════════════════════════════════════
#  PROCESS TANKS  (3-D boxes)
# ═════════════════════════════════════════════════════════════════
box3d(Xc["pre"],    y_mid, bw_s, bh, PX, PY,
      face_color="#78909C",
      label="预处理",      sublabel="Pretreatment\n(格栅/沉砂池)")

box3d(Xc["anaero"], y_mid, bw_s, bh, PX, PY,
      face_color="#8D6E63",
      label="厌氧区",      sublabel="Anaerobic\nZone")

box3d(Xc["anoxic"], y_mid, bw_s, bh, PX, PY,
      face_color="#7986CB",
      label="缺氧区",      sublabel="Anoxic\nZone")

box3d(Xc["od1"],   y_mid, bw_od, bh, PX, PY,
      face_color="#42A5F5",
      label="氧化沟 1",   sublabel="Oxidation\nDitch 1")

box3d(Xc["od2"],   y_mid, bw_od, bh, PX, PY,
      face_color="#26C6DA",
      label="氧化沟 2",   sublabel="Oxidation\nDitch 2")

box3d(Xc["sst"],   y_mid, bw_s, bh, PX, PY,
      face_color="#5C6BC0",
      label="二沉池",      sublabel="Secondary\nClarifier")

box3d(Xc["dis"],   y_mid, bw_s, bh, PX, PY,
      face_color="#26A69A",
      label="消毒池",      sublabel="Disinfection")

# ── aerators on both oxidation ditches ───────────────────────────
aerator(Xc["od1"], y_mid - bh * 0.28, r=0.28)
aerator(Xc["od2"], y_mid - bh * 0.28, r=0.28)

# ═════════════════════════════════════════════════════════════════
#  MAIN FLOW ARROWS  (dark-green solid)
# ═════════════════════════════════════════════════════════════════
c_flow = "#2E7D32"

arr_h(Xc["city"]   + 0.92,
      Xc["pre"]    - bw_s/2  - GAP,           y_mid, c_flow,
      label="进水 Influent")
arr_h(Xc["pre"]    + bw_s/2  + PX + GAP,
      Xc["anaero"] - bw_s/2  - GAP,           y_mid, c_flow)
arr_h(Xc["anaero"] + bw_s/2  + PX + GAP,
      Xc["anoxic"] - bw_s/2  - GAP,           y_mid, c_flow)
arr_h(Xc["anoxic"] + bw_s/2  + PX + GAP,
      Xc["od1"]   - bw_od/2 - GAP,            y_mid, c_flow)
arr_h(Xc["od1"]   + bw_od/2 + PX + GAP,
      Xc["od2"]   - bw_od/2 - GAP,            y_mid, c_flow)
arr_h(Xc["od2"]   + bw_od/2 + PX + GAP,
      Xc["sst"]   - bw_s/2  - GAP,            y_mid, c_flow)
arr_h(Xc["sst"]   + bw_s/2  + PX + GAP,
      Xc["dis"]   - bw_s/2  - GAP,            y_mid, c_flow)
arr_h(Xc["dis"]   + bw_s/2  + PX + GAP,
      Xc["nature"] - 0.92,                    y_mid, c_flow,
      label="出水 Effluent")

# ═════════════════════════════════════════════════════════════════
#  ALTERNATE OPERATION  (teal dashed, above OD1–OD2)
#  Shows the two ditches alternate between aerobic / anoxic roles
# ═════════════════════════════════════════════════════════════════
c_alt  = "#00838F"
y_alt  = y_mid + bh/2 + PY + 0.55

routed_arrow(
    [Xc["od1"],  Xc["od1"],  Xc["od2"],  Xc["od2"]],
    [y_mid + bh/2 + PY, y_alt, y_alt, y_mid + bh/2 + PY],
    color=c_alt, lw=1.8, linestyle="--",
    label="Alternate operation\n交替运行",
    label_idx=1
)

# ═════════════════════════════════════════════════════════════════
#  SLUDGE RETURN  (orange-red dashed, routed below all tanks)
#  Secondary Clarifier → Anaerobic Zone  (matching image 2)
# ═════════════════════════════════════════════════════════════════
c_sludge = "#E64A19"
y_below  = y_mid - bh/2 - 0.80

routed_arrow(
    [Xc["sst"],   Xc["sst"],
     Xc["anaero"], Xc["anaero"]],
    [y_mid - bh/2, y_below,
     y_below,       y_mid - bh/2],
    color=c_sludge, lw=1.9, linestyle="--",
    label="Sludge Return / 污泥回流",
    label_idx=1
)

# ═════════════════════════════════════════════════════════════════
#  EXCESS SLUDGE  (brown arrow downward from SST → Sludge Treatment box)
# ═════════════════════════════════════════════════════════════════
y_sludge_box = y_mid - bh/2 - 2.10
c_mud = "#6D4C41"

ax.annotate("",
            xy=(Xc["sst"] + 0.50, y_sludge_box + 0.48 + PY),
            xytext=(Xc["sst"] + 0.50, y_mid - bh/2),
            arrowprops=dict(arrowstyle="-|>", color=c_mud, lw=1.7,
                            mutation_scale=10), zorder=6)
ax.text(Xc["sst"] + 0.86, (y_sludge_box + 0.48 + y_mid - bh/2) / 2,
        "Excess Sludge\n剩余污泥",
        ha="left", va="center", fontsize=7.5, color=c_mud, zorder=6)

box3d(Xc["sst"] + 0.50, y_sludge_box, 1.70, 0.96, 0.18, 0.14,
      face_color="#8D6E63",
      label="污泥处理", sublabel="Sludge Treatment", fs=8.0)

# ═════════════════════════════════════════════════════════════════
#  OPERATION CONDITION  dashed red frame
#  Encloses Pretreatment → OD2  (same scope as reference image)
# ═════════════════════════════════════════════════════════════════
oc_x0 = Xc["pre"]  - bw_s/2  - 0.32
oc_x1 = Xc["od2"]  + bw_od/2 + PX + 0.32
oc_y0 = y_below - 0.45
oc_y1 = y_alt   + 0.52

oc_rect = FancyBboxPatch((oc_x0, oc_y0), oc_x1-oc_x0, oc_y1-oc_y0,
                          boxstyle="round,pad=0.08",
                          linewidth=1.8, edgecolor="#C62828",
                          facecolor="none", linestyle="dashed", zorder=3)
ax.add_patch(oc_rect)

ax.text(oc_x0 + 0.22, oc_y1 - 0.10,
        "Operation condition",
        ha="left", va="top",
        fontsize=10.5, fontweight="bold", color="#C62828", zorder=7)

ax.text(oc_x0 + 0.22, oc_y1 - 0.54,
        "Input",
        ha="left", va="top",
        fontsize=11.5, fontweight="bold", color="#C62828", zorder=7)

ax.text(oc_x0 + 1.15, oc_y1 - 0.54,
        "HRT",
        ha="left", va="top",
        fontsize=10.0, color="#C62828", fontweight="bold", zorder=7)

# ═════════════════════════════════════════════════════════════════
#  PROCESS TITLE BADGE  (top-right corner)
# ═════════════════════════════════════════════════════════════════
ax.add_patch(FancyBboxPatch(
    (badge_cx - 1.50, badge_cy - 1.20), 3.00, 2.40,
    boxstyle="round,pad=0.15",
    linewidth=2, edgecolor="#3949AB",
    facecolor="#C5CAE9", alpha=0.92, zorder=7))

ax.text(badge_cx, badge_cy + 0.55,
        "双氧化沟",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color="#1A237E", zorder=8)
ax.text(badge_cx, badge_cy - 0.30,
        "Dual\nOxidation Ditch",
        ha="center", va="center",
        fontsize=11, fontweight="bold", color="#283593", zorder=8)

# ═════════════════════════════════════════════════════════════════
#  TOP TITLE
# ═════════════════════════════════════════════════════════════════
ax.text(FIG_W / 2, FIG_H - 0.38,
        "双氧化沟污水处理工艺流程图  "
        "Dual Oxidation Ditch Wastewater Treatment Process",
        ha="center", va="top",
        fontsize=14, fontweight="bold", color="#0D47A1", zorder=9)

# ═════════════════════════════════════════════════════════════════
#  SAVE
# ═════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.2)

out_tiff = "dual_od_process.tiff"
out_png  = "dual_od_process.png"

fig.savefig(out_tiff, dpi=DPI, format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches="tight")
fig.savefig(out_png,  dpi=DPI, format="png",
            bbox_inches="tight")

print(f"Saved: {out_tiff}")
print(f"Saved: {out_png}")
plt.close(fig)
