"""
draw_model_arch.py
==================
绘制 DeformTimeModel 模型结构图，用于 SCI 论文发表。

运行方式：
    python draw_model_arch.py

输出文件：
    model_architecture.tiff  —— 300 DPI, LZW 压缩（投稿用）
    model_architecture.png   —— 300 DPI PNG（预览用）
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ─────────────────────── CONFIG ─────────────────────────────────
FIG_W   = 13       # inches
FIG_H   = 20       # inches
DPI     = 300
FS      = 8.5      # base font size (pt)
# ────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    FS,
})

# Colour palette
C = dict(
    io       = "#D0E8FF",
    norm     = "#E8F5E9",
    embed    = "#FFF9C4",
    enc_bg   = "#F3E5F5",
    attn1d   = "#BBDEFB",
    attn2d   = "#FFE0B2",
    ca       = "#F8BBD0",
    fuse     = "#E1F5FE",
    gru      = "#DCEDC8",
    fc       = "#FFF3E0",
    arrow    = "#455A64",
    border   = "#455A64",
    enc_bord = "#8E24AA",
    lnorm    = "#C8E6C9",
    drop     = "#F5F5F5",
)
ALPHA = 0.92
LW    = 0.9

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


# ─── helpers ─────────────────────────────────────────────────────

def draw_box(x, y, w, h, label, sublabel="",
             color=C["io"], bold=False, alpha=ALPHA,
             border_color=C["border"], lw=LW, fontsize=None, zorder=3):
    fs = fontsize or FS
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.005",
        linewidth=lw, edgecolor=border_color,
        facecolor=color, alpha=alpha, zorder=zorder
    )
    ax.add_patch(rect)
    fw = "bold" if bold else "normal"
    ty = y + (h * 0.12 if sublabel else 0)
    ax.text(x, ty, label,
            ha="center", va="center", fontsize=fs, fontweight=fw,
            color="#1A237E" if bold else "#1a1a1a", zorder=zorder + 1)
    if sublabel:
        ax.text(x, y - h * 0.22, sublabel,
                ha="center", va="center", fontsize=fs - 1.5,
                color="#555555", style="italic", zorder=zorder + 1)


def draw_arrow(x0, y0, x1, y1, color=C["arrow"], lw=1.2,
               rad=0.0, zorder=2):
    ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=lw,
                    connectionstyle=f"arc3,rad={rad}",
                    mutation_scale=8
                ),
                zorder=zorder)


def shape_label(x, y, text):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=FS - 2.0, color="#5D4037", zorder=5,
            bbox=dict(boxstyle="round,pad=0.12",
                      fc="white", ec="none", alpha=0.75))


# ════════════════════════════════════════════════════════════════
# Vertical layout  (y: 1 = top, 0 = bottom)
# ════════════════════════════════════════════════════════════════
CX  = 0.50      # main spine centre-x
BW  = 0.28      # standard box width
BH  = 0.033     # standard box height
GAP = 0.020     # gap between boxes

y = 0.978       # running y pointer (always points to box centre)

def next_y(h=BH):
    global y
    y -= h / 2
    out = y
    y -= h / 2 + GAP
    return out

# ── main spine ──────────────────────────────────────────────────
Y_INPUT     = next_y()
Y_INSTNORM  = next_y()
Y_EMBED     = next_y()
Y_PRENORM   = next_y()

ENC_TOP_Y   = y + GAP * 0.4   # encoder background top

Y_ENCLNORM  = next_y()

# branch diverge point
Y_BRANCH_DIV = y             # y right after enc_lnorm's gap

BH2 = 0.031                  # branch-box height
BGAP = 0.018                 # gap inside branch

P1X = 0.245   # 1D branch centre-x
P2X = 0.755   # 2D branch centre-x
PBW = 0.195   # branch box width

def branch_y(n):
    """nth branch box y (0-indexed, top first)."""
    return Y_BRANCH_DIV - n * (BH2 + BGAP) - BH2 / 2

Y_B1_RESHAPE  = branch_y(0)
Y_B1_DA1D     = branch_y(1)
Y_B1_DROP1    = branch_y(2)
Y_B1_MLP      = branch_y(3)
Y_B1_DROP2    = branch_y(4)
Y_B1_BOTTOM   = branch_y(5)

# 2D branch same y-levels
Y_B2_UNFOLD   = Y_B1_RESHAPE
Y_B2_DA2D     = Y_B1_DA1D
Y_B2_DROP1    = Y_B1_DROP1
Y_B2_MLP      = Y_B1_MLP
Y_B2_DROP2    = Y_B1_DROP2
Y_B2_WRITEOUT = Y_B1_BOTTOM

# resume main spine below branches
y = Y_B1_BOTTOM - BH2 / 2 - GAP

Y_CONCAT    = next_y()
Y_CA1       = next_y()
Y_FCFUSE    = next_y()

ENC_BOT_Y   = y + GAP * 0.4   # encoder background bottom

Y_POSTLNORM = next_y()
Y_CA2       = next_y()
Y_GRU       = next_y()
Y_FCSEQ     = next_y()
Y_PROJ      = next_y()
Y_DENORM    = next_y()
Y_OUTPUT    = next_y()


# ════════════════════════════════════════════════════════════════
# Draw encoder background
# ════════════════════════════════════════════════════════════════
enc_bg = FancyBboxPatch(
    (0.055, ENC_BOT_Y), 0.89, ENC_TOP_Y - ENC_BOT_Y,
    boxstyle="round,pad=0.010",
    linewidth=1.8, edgecolor=C["enc_bord"],
    facecolor=C["enc_bg"], alpha=0.30, zorder=1
)
ax.add_patch(enc_bg)
ax.text(0.065, (ENC_TOP_Y + ENC_BOT_Y) / 2,
        "Encoder Block\n(x e_layers)",
        ha="left", va="center", fontsize=FS - 1,
        color=C["enc_bord"], fontweight="bold",
        rotation=90, zorder=2)


# ════════════════════════════════════════════════════════════════
# Main spine boxes
# ════════════════════════════════════════════════════════════════
draw_box(CX, Y_INPUT,    BW, BH, "Input",
         sublabel="[B, L, F]", color=C["io"], bold=True)

draw_box(CX, Y_INSTNORM, BW, BH, "Instance Normalization",
         sublabel="mean & std detach", color=C["norm"])

draw_box(CX, Y_EMBED,    BW, BH,
         "Value Embedding",
         sublabel="Local_Temporal / Deform_Temporal", color=C["embed"])

draw_box(CX, Y_PRENORM,  BW, BH, "Pre-LayerNorm",
         sublabel="nn.LayerNorm(d_model)", color=C["lnorm"])

# ── inside encoder ───────────────────────────────────────────────
draw_box(CX, Y_ENCLNORM, BW, BH, "LayerNorm",
         sublabel="(per encoder layer)", color=C["lnorm"], zorder=4)

# ── 1D deformable attention branch ──────────────────────────────
draw_box(P1X, Y_B1_RESHAPE, PBW, BH2,
         "Reshape", sublabel="(b*n_days) L/n d",
         color=C["attn1d"], zorder=4)

draw_box(P1X, Y_B1_DA1D, PBW, BH2,
         "DeformAtten1D",
         sublabel="n_heads x head_dim",
         color=C["attn1d"], bold=True, zorder=4)

draw_box(P1X, Y_B1_DROP1, PBW, BH2,
         "Residual + DropPath",
         color=C["drop"], zorder=4)

draw_box(P1X, Y_B1_MLP, PBW, BH2,
         "MLP (FFN)",
         sublabel="d -> d, ReLU",
         color=C["attn1d"], zorder=4)

draw_box(P1X, Y_B1_DROP2, PBW, BH2,
         "Residual + DropPath",
         color=C["drop"], zorder=4)

draw_box(P1X, Y_B1_BOTTOM, PBW, BH2,
         "Reshape back",
         sublabel="b L d  (1D out)",
         color=C["attn1d"], zorder=4)

# ── 2D patch deformable attention branch ────────────────────────
draw_box(P2X, Y_B2_UNFOLD, PBW, BH2,
         "Patch Unfold",
         sublabel="[B, N_p, patch_len, d]",
         color=C["attn2d"], zorder=4)

draw_box(P2X, Y_B2_DA2D, PBW, BH2,
         "DeformAtten2D",
         sublabel="Conv2d QKV / n_heads=1",
         color=C["attn2d"], bold=True, zorder=4)

draw_box(P2X, Y_B2_DROP1, PBW, BH2,
         "Residual + DropPath",
         color=C["drop"], zorder=4)

draw_box(P2X, Y_B2_MLP, PBW, BH2,
         "MLP (FFN)",
         sublabel="LayerNorm then d -> d",
         color=C["attn2d"], zorder=4)

draw_box(P2X, Y_B2_DROP2, PBW, BH2,
         "Residual + DropPath",
         color=C["drop"], zorder=4)

draw_box(P2X, Y_B2_WRITEOUT, PBW, BH2,
         "Write-Out Linear",
         sublabel="N_p*patch_len -> L  (2D out)",
         color=C["attn2d"], zorder=4)

# ── fusion ───────────────────────────────────────────────────────
draw_box(CX, Y_CONCAT, BW, BH, "Concat",
         sublabel="[B, L, 2*d_model]",
         color=C["fuse"], zorder=4)

draw_box(CX, Y_CA1, BW, BH,
         "Channel Attention  (CA1 / ChannelGate)",
         sublabel="mean+max pool -> MLP -> sigmoid gate",
         color=C["ca"], bold=True, zorder=4)

draw_box(CX, Y_FCFUSE, BW, BH,
         "Linear  2d -> d",
         sublabel="[B, L, d_model]",
         color=C["fuse"], zorder=4)

# ── post-encoder ─────────────────────────────────────────────────
draw_box(CX, Y_POSTLNORM, BW, BH,
         "Layernorm  (post-encoder)",
         sublabel="subtract mean over L", color=C["lnorm"])

draw_box(CX, Y_CA2, BW, BH,
         "Channel Attention  (CA2 / ChannelGate)",
         sublabel="[B, L, d_model]",
         color=C["ca"], bold=True)

draw_box(CX, Y_GRU, BW, BH,
         "GRU",
         sublabel="d_layers stacked, batch_first",
         color=C["gru"], bold=True)

draw_box(CX, Y_FCSEQ, BW, BH,
         "FC   seq_len -> d_model -> pred_len",
         sublabel="Linear-LeakyReLU-Linear (on L dim)",
         color=C["fc"])

draw_box(CX, Y_PROJ, BW, BH,
         "Projection  d_model -> c_out",
         sublabel="nn.Linear", color=C["fc"])

draw_box(CX, Y_DENORM, BW, BH,
         "De-normalization",
         sublabel="out = out * std + mean", color=C["norm"])

draw_box(CX, Y_OUTPUT, BW, BH,
         "Output",
         sublabel="[B, pred_len, c_out]",
         color=C["io"], bold=True)


# ════════════════════════════════════════════════════════════════
# Arrows — main spine
# ════════════════════════════════════════════════════════════════
spine_pairs = [
    (Y_INPUT,    Y_INSTNORM),
    (Y_INSTNORM, Y_EMBED),
    (Y_EMBED,    Y_PRENORM),
    (Y_PRENORM,  Y_ENCLNORM - BH / 2),
    # skip (encoder top handled by branches)
    (Y_FCFUSE,   Y_POSTLNORM),
    (Y_POSTLNORM,Y_CA2),
    (Y_CA2,      Y_GRU),
    (Y_GRU,      Y_FCSEQ),
    (Y_FCSEQ,    Y_PROJ),
    (Y_PROJ,     Y_DENORM),
    (Y_DENORM,   Y_OUTPUT),
]

for (ya, yb) in spine_pairs:
    draw_arrow(CX, ya - BH / 2 - 0.002, CX, yb + BH / 2 + 0.002)

# ── encoder LayerNorm -> branches diverge ───────────────────────
draw_arrow(CX, Y_ENCLNORM - BH / 2 - 0.002,
           P1X, Y_B1_RESHAPE + BH2 / 2 + 0.002, rad=0.25)
draw_arrow(CX, Y_ENCLNORM - BH / 2 - 0.002,
           P2X, Y_B2_UNFOLD + BH2 / 2 + 0.002, rad=-0.25)

# ── 1D branch internal ───────────────────────────────────────────
for (ya, yb) in [
    (Y_B1_RESHAPE, Y_B1_DA1D),
    (Y_B1_DA1D,    Y_B1_DROP1),
    (Y_B1_DROP1,   Y_B1_MLP),
    (Y_B1_MLP,     Y_B1_DROP2),
    (Y_B1_DROP2,   Y_B1_BOTTOM),
]:
    draw_arrow(P1X, ya - BH2 / 2 - 0.001,
               P1X, yb + BH2 / 2 + 0.001)

# ── 2D branch internal ───────────────────────────────────────────
for (ya, yb) in [
    (Y_B2_UNFOLD,  Y_B2_DA2D),
    (Y_B2_DA2D,    Y_B2_DROP1),
    (Y_B2_DROP1,   Y_B2_MLP),
    (Y_B2_MLP,     Y_B2_DROP2),
    (Y_B2_DROP2,   Y_B2_WRITEOUT),
]:
    draw_arrow(P2X, ya - BH2 / 2 - 0.001,
               P2X, yb + BH2 / 2 + 0.001)

# ── branches converge to Concat ──────────────────────────────────
draw_arrow(P1X, Y_B1_BOTTOM - BH2 / 2 - 0.002,
           CX, Y_CONCAT + BH / 2 + 0.002, rad=-0.25)
draw_arrow(P2X, Y_B2_WRITEOUT - BH2 / 2 - 0.002,
           CX, Y_CONCAT + BH / 2 + 0.002, rad=0.25)

# ── Concat -> CA1 -> FC ──────────────────────────────────────────
draw_arrow(CX, Y_CONCAT - BH / 2 - 0.002,
           CX, Y_CA1   + BH / 2 + 0.002)
draw_arrow(CX, Y_CA1   - BH / 2 - 0.002,
           CX, Y_FCFUSE+ BH / 2 + 0.002)


# ════════════════════════════════════════════════════════════════
# Residual skip arrows (curved, from input of a sub-block to output)
# ════════════════════════════════════════════════════════════════
def skip_arrow(x_branch, y_top, y_bot, side="left", rad_val=0.5):
    xoff = -PBW / 2 - 0.02 if side == "left" else PBW / 2 + 0.02
    xs = x_branch + xoff
    draw_arrow(xs, y_top - BH2 / 2 - 0.001,
               xs, y_bot + BH2 / 2 + 0.001,
               rad=0.0, lw=0.8, color="#90A4AE")
    ax.annotate("",
                xy=(x_branch - PBW / 2 + 0.002, y_bot),
                xytext=(xs, y_bot + BH2 / 2 + 0.001),
                arrowprops=dict(arrowstyle="-|>", color="#90A4AE",
                                lw=0.8, mutation_scale=6),
                zorder=2)
    ax.annotate("",
                xy=(xs, y_top - BH2 / 2 - 0.001),
                xytext=(x_branch - PBW / 2 + 0.002, y_top),
                arrowprops=dict(arrowstyle="-", color="#90A4AE",
                                lw=0.8),
                zorder=2)

# 1D: residual around DA1D
skip_arrow(P1X, Y_B1_RESHAPE, Y_B1_DROP1, side="left")
# 1D: residual around MLP
skip_arrow(P1X, Y_B1_DROP1,   Y_B1_DROP2, side="left")
# 2D: residual around DA2D
skip_arrow(P2X, Y_B2_UNFOLD,  Y_B2_DROP1, side="right")
# 2D: residual around MLP
skip_arrow(P2X, Y_B2_DROP1,   Y_B2_DROP2, side="right")


# ════════════════════════════════════════════════════════════════
# Branch headers
# ════════════════════════════════════════════════════════════════
ax.text(P1X, Y_B1_RESHAPE + BH2 / 2 + 0.018,
        "1D Deformable\nAttention Path",
        ha="center", va="bottom", fontsize=FS - 0.5,
        fontweight="bold", color="#1565C0", zorder=6)

ax.text(P2X, Y_B2_UNFOLD + BH2 / 2 + 0.018,
        "2D Patch Deformable\nAttention Path",
        ha="center", va="bottom", fontsize=FS - 0.5,
        fontweight="bold", color="#E65100", zorder=6)


# ════════════════════════════════════════════════════════════════
# Shape annotations on the right side of main spine
# ════════════════════════════════════════════════════════════════
shape_label(CX + BW / 2 + 0.01, Y_INPUT,    "[B, L, F]")
shape_label(CX + BW / 2 + 0.01, Y_EMBED,    "[B, L, d]")
shape_label(CX + BW / 2 + 0.01, Y_CONCAT,   "[B, L, 2d]")
shape_label(CX + BW / 2 + 0.01, Y_CA1,      "[B, L, 2d]")
shape_label(CX + BW / 2 + 0.01, Y_FCFUSE,   "[B, L, d]")
shape_label(CX + BW / 2 + 0.01, Y_GRU,      "[B, L, d]")
shape_label(CX + BW / 2 + 0.01, Y_OUTPUT,   "[B, T, c]")


# ════════════════════════════════════════════════════════════════
# Legend
# ════════════════════════════════════════════════════════════════
legend_items = [
    (C["io"],     "Input / Output"),
    (C["norm"],   "Normalization"),
    (C["embed"],  "Embedding"),
    (C["attn1d"], "1D Deformable Attn"),
    (C["attn2d"], "2D Deformable Attn"),
    (C["ca"],     "Channel Attention (CA)"),
    (C["fuse"],   "Fusion / FC"),
    (C["gru"],    "GRU"),
    (C["drop"],   "Residual + DropPath"),
]
handles = [mpatches.Patch(facecolor=col, edgecolor="#455A64",
                          linewidth=0.6, label=lbl)
           for col, lbl in legend_items]
ax.legend(handles=handles, loc="lower right",
          bbox_to_anchor=(0.995, 0.002),
          fontsize=FS - 1.5, ncol=2,
          framealpha=0.85, edgecolor="#BDBDBD",
          title="Component Legend", title_fontsize=FS - 1)


# ════════════════════════════════════════════════════════════════
# Title
# ════════════════════════════════════════════════════════════════
ax.text(0.50, 0.9985,
        "DeformTimeModel Architecture",
        ha="center", va="top", fontsize=FS + 3,
        fontweight="bold", color="#1A237E")

ax.text(0.50, 0.9935,
        "Deformable Temporal Attention with Dual-Path Encoder, "
        "Channel Attention Gates and GRU Decoder",
        ha="center", va="top", fontsize=FS - 0.5, color="#37474F")


# ════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════
fig.tight_layout(pad=0.3)

png_path  = "model_architecture.png"
tiff_path = "model_architecture.tiff"

fig.savefig(png_path,  dpi=DPI, bbox_inches="tight",
            facecolor="white")
fig.savefig(tiff_path, dpi=DPI, bbox_inches="tight",
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"})

print(f"Saved: {png_path}")
print(f"Saved: {tiff_path}  (300 DPI, LZW)")
