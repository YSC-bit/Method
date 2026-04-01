"""
combine_figures.py
------------------
将四张结果对比图拼接为 2×2 的 SCI 论文标准格式图。

使用方法：
    1. 将四张图片（img_a.png, img_b.png, img_c.png, img_d.png）
       放到与本脚本相同的目录下，或在下方 IMAGE_PATHS 中修改路径。
    2. 运行：python combine_figures.py
    3. 输出文件为 combined_figure.tiff（300 DPI，符合 SCI 投稿要求）
       同时生成 combined_figure.png 便于预览。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# ── 配置区：按实际文件名修改 ────────────────────────────────────────────────
IMAGE_PATHS = [
    "img_a.png",   # 子图 (a)
    "img_b.png",   # 子图 (b)
    "img_c.png",   # 子图 (c)
    "img_d.png",   # 子图 (d)
]
OUTPUT_TIFF = "combined_figure.tiff"
OUTPUT_PNG  = "combined_figure.png"

# SCI 期刊常用规格：双栏宽度约 17 cm，单栏约 8.5 cm
# 此处使用 17 cm × 14 cm（宽×高），可根据目标期刊要求调整
FIG_WIDTH_CM  = 17    # 单位：厘米
FIG_HEIGHT_CM = 14
DPI           = 300   # 300 DPI 满足绝大多数 SCI 期刊要求

# 子图标签字体大小（pt）
LABEL_FONTSIZE = 14
LABEL_FONTWEIGHT = 'bold'
LABEL_X = -0.04   # 相对于子图左边界的水平偏移（可微调）
LABEL_Y =  1.02   # 相对于子图上边界的垂直偏移（可微调）
# ────────────────────────────────────────────────────────────────────────────


def cm2inch(cm):
    return cm / 2.54


def main():
    missing = [p for p in IMAGE_PATHS if not Path(p).exists()]
    if missing:
        print("❌ 以下图片文件不存在，请检查路径：")
        for m in missing:
            print(f"   {m}")
        return

    labels = ['(a)', '(b)', '(c)', '(d)']
    images = [mpimg.imread(p) for p in IMAGE_PATHS]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(cm2inch(FIG_WIDTH_CM), cm2inch(FIG_HEIGHT_CM)),
        dpi=DPI
    )

    for ax, img, label in zip(axes.flat, images, labels):
        ax.imshow(img)
        ax.axis('off')
        ax.text(
            LABEL_X, LABEL_Y, label,
            transform=ax.transAxes,
            fontsize=LABEL_FONTSIZE,
            fontweight=LABEL_FONTWEIGHT,
            va='bottom', ha='left'
        )

    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5)

    fig.savefig(OUTPUT_TIFF, dpi=DPI, format='tiff',
                bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(OUTPUT_PNG, dpi=DPI, format='png', bbox_inches='tight')
    plt.close(fig)

    print(f"✅ 已生成：{OUTPUT_TIFF}（300 DPI，TIFF/LZW 压缩，投稿用）")
    print(f"✅ 已生成：{OUTPUT_PNG}（300 DPI，PNG，预览用）")


if __name__ == '__main__':
    main()
