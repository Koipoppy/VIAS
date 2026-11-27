# -*- coding: utf-8 -*-
"""Figures for the fusion-strategy critique report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "fusion_critique"
METRICS = ROOT / "outputs" / "pan18" / "fusion_metrics.json"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["SimSun", "Times New Roman"],
    "font.size": 9, "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
INK = "#1A1A1A"; MUTED = "#4D4D4D"; BORDER = "#2F3E4E"
BLUE = "#5B6F8A"; GRAY = "#D9D9D9"; MID = "#8FA1B3"


def box(ax, x, y, w, h, text, sub=None, fc="white"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=BORDER, linewidth=.8))
    ax.text(x + w / 2, y + h * .70, text, ha="center", va="center",
            fontsize=10.2, color=INK, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h * .22, sub, ha="center", va="center",
                fontsize=8.2, color=MUTED)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                 mutation_scale=8, color=BORDER, linewidth=.8))


def schematic():
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.5), dpi=300,
                             gridspec_kw={"hspace": .30})
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 2.65); ax.axis("off")
    # Late fusion
    ax = axes[0]
    ax.text(.1, 2.52, "A. 证据级 / late fusion（当前实现）", fontsize=10.4,
            color=INK, fontweight="bold")
    box(ax, .25, .30, 1.5, .8, "文本", "100 条推文")
    box(ax, .25, 1.52, 1.5, .8, "图像", "首图")
    box(ax, 2.35, .30, 1.6, .8, "文本模型", "P(Y|text)")
    box(ax, 2.35, 1.52, 1.6, .8, "图像模型", "P(Y|image)")
    box(ax, 4.65, .62, 1.6, .9, "加权融合", "log pool / 权重")
    box(ax, 6.95, .62, 1.3, .9, "后验", "max P(Y)")
    arrow(ax, 1.75, .70, 2.35, .70); arrow(ax, 1.75, 1.92, 2.35, 1.92)
    arrow(ax, 3.95, .70, 4.65, .92); arrow(ax, 3.95, 1.92, 4.65, 1.22)
    arrow(ax, 6.25, 1.07, 6.95, 1.07)
    ax.text(4.35, .08, "只交换概率，不交换特征；没有跨模态交互", fontsize=8.4, color=MUTED)

    # Joint fusion
    ax = axes[1]
    ax.text(.1, 2.52, "B. 特征级 / joint fusion（应该做）", fontsize=10.4,
            color=INK, fontweight="bold")
    box(ax, .25, .30, 1.5, .8, "文本", "100 条推文")
    box(ax, .25, 1.52, 1.5, .8, "图像", "多图")
    box(ax, 2.15, .30, 1.6, .8, "文本编码器", "h_text")
    box(ax, 2.15, 1.52, 1.6, .8, "图像编码器", "h_image")
    box(ax, 4.35, .62, 1.95, .9, "跨模态交互", "cross-attn")
    box(ax, 6.85, .62, 1.35, .9, "联合分类", "P(Y|T,I)")
    arrow(ax, 1.75, .70, 2.15, .70); arrow(ax, 1.75, 1.92, 2.15, 1.92)
    arrow(ax, 3.75, .70, 4.35, .88); arrow(ax, 3.75, 1.92, 4.35, 1.26)
    arrow(ax, 6.30, 1.07, 6.85, 1.07)
    ax.text(4.20, .08, "融合发生在特征/表示层；模型能学习何时信文本、何时信图像", fontsize=8.4, color=MUTED)

    fig.tight_layout(pad=.3)
    p = OUT / "fusion_schematic.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=.02, facecolor="white")
    plt.close(fig)
    return p


def bars():
    m = json.loads(METRICS.read_text(encoding="utf-8"))["results"]["all"]
    methods = [("text_only", "文本"), ("image_only", "图像"),
               ("equal_prob", "等权"), ("credibility", "可信度"),
               ("calibrated_late", "校准"), ("stacked_lr", "堆叠"),
               ("conflict_handled", "冲突")]
    fig, ax = plt.subplots(figsize=(7.0, 2.3), dpi=300)
    vals = [m[k]["accuracy"] for k, _ in methods]
    colors = [BLUE, GRAY, MID, "#C9CED6", "#8FA1B3", BLUE, GRAY]
    bars_ = ax.bar(range(len(vals)), vals, .6, color=colors, edgecolor=INK, linewidth=.5)
    for b, v in zip(bars_, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([x[1] for x in methods], fontsize=8.2)
    ax.set_ylim(.5, .85)
    ax.set_ylabel("Accuracy", fontsize=8.4)
    ax.grid(axis="y", linestyle=":", linewidth=.5, color="#B0B0B0", alpha=.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=.5)
    p = OUT / "fusion_accuracy.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=.02, facecolor="white")
    plt.close(fig)
    return p


def main():
    print(schematic())
    print(bars())


if __name__ == "__main__":
    main()
