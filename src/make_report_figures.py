# -*- coding: utf-8 -*-
"""Academic-style report figures: pipeline and sample composition."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["SimSun", "Times New Roman"],
    "font.size": 9,
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

INK = "#1A1A1A"
BORDER = "#2F3E4E"
MUTED = "#4D4D4D"
GRID = "#B0B0B0"
MALE = "#5B6F8A"
FEMALE = "#D9D9D9"
AGE = "#C9CED6"


def pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 1.05), dpi=300)
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 1.05)
    ax.axis("off")

    labels = ["多模态输入", "证据单元", "六维可信度", "加权贝叶斯", "冲突/可解释"]
    subs = ["文本 / 人脸 / 自述", "22 字段", "六维权重", "属性后验", "D-S + LOO"]
    w, gap, h, y, x0 = 1.32, 0.18, 0.85, 0.10, 0.15
    for i, (lab, sub) in enumerate(zip(labels, subs)):
        x = x0 + i * (w + gap)
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white",
                               edgecolor=BORDER, linewidth=0.8))
        ax.text(x + 0.06, y + h - 0.08, str(i + 1), fontsize=8,
                color=MUTED, ha="left", va="top")
        ax.text(x + w / 2, y + 0.58, lab, fontsize=11, color=INK,
                ha="center", va="center", fontweight="bold")
        ax.text(x + w / 2, y + 0.30, sub, fontsize=8.4, color=MUTED,
                ha="center", va="center")
        if i < len(labels) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w + 0.015, y + h / 2), (x + w + gap - 0.015, y + h / 2),
                arrowstyle="->", mutation_scale=8, color=BORDER, linewidth=0.8,
            ))
    fig.tight_layout(pad=0.02)
    path = os.path.join(OUT, "pipeline_diagram.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    return path


def samples():
    ages = ["22", "24", "28", "32", "36"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2), dpi=300, sharey=True)
    for ax, title, n in [(axes[0], "Pilot（n=50）", 5),
                         (axes[1], "30 例案件包（n=30）", 3)]:
        male = [n] * 5
        female = [n] * 5
        ax.bar(ages, male, width=0.55, color=MALE, edgecolor=INK,
               linewidth=0.5, label="男")
        ax.bar(ages, female, bottom=male, width=0.55, color=FEMALE,
               edgecolor=INK, linewidth=0.5, label="女")
        ax.set_title(title, fontsize=9.2, color=INK)
        ax.set_ylim(0, 12)
        ax.set_xlabel("年龄", fontsize=8.3)
        ax.set_ylabel("样本数", fontsize=8.3)
        ax.tick_params(labelsize=7.8)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, color=GRID, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BORDER)
        ax.spines["bottom"].set_color(BORDER)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.5)
    path = os.path.join(OUT, "sample_composition.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    return path


def method_comparison():
    m = json.load(open(os.path.join(ROOT, "outputs", "metrics.json"), encoding="utf-8"))
    methods = ["text_only", "face_only", "fused_equal", "fused_cred", "fused_conflict"]
    labels = ["文本", "人脸", "等权", "可信度", "可信度+冲突"]
    clean_g = [m["clean"][k]["gender"]["accuracy"] for k in methods]
    clean_a = [m["clean"][k]["age"]["accuracy"] for k in methods]
    inj_g = [m["injected"][k]["injected_subset"]["gender"]["accuracy"] for k in methods]
    inj_a = [m["injected"][k]["injected_subset"]["age"]["accuracy"] for k in methods]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), dpi=300, sharey=True)
    x = np.arange(len(methods))
    width = 0.36
    for ax, g, a, title in [
        (axes[0], clean_g, clean_a, "Clean（n=50）"),
        (axes[1], inj_g, inj_a, "Injected（n=10）"),
    ]:
        b1 = ax.bar(x - width / 2, g, width, color=MALE, edgecolor=INK,
                    linewidth=0.5, label="性别 Acc")
        b2 = ax.bar(x + width / 2, a, width, color=AGE, edgecolor=INK,
                    linewidth=0.5, label="年龄 Acc")
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=6.2, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.4)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=9.0, color=INK)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, color=GRID, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BORDER)
        ax.spines["bottom"].set_color(BORDER)
        ax.tick_params(labelsize=7.4)
    axes[0].set_ylabel("Accuracy", fontsize=8.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False,
               fontsize=7.8, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.5)
    path = os.path.join(OUT, "method_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    return path


def _cm(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def confusions():
    df = pd.read_csv(os.path.join(ROOT, "outputs", "method_predictions_clean.csv"))
    d = df[df["method"] == "fused_cred"]
    genders = ["male", "female"]
    gender_labels = ["男", "女"]
    ages = ["18-24", "25-29", "30-34", "35-40"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), dpi=300)
    for ax, cm, labels, title in [
        (axes[0], _cm(d["true_gender"], d["pred_gender"], genders),
         gender_labels, "性别（融合可信度）"),
        (axes[1], _cm(d["true_age_bucket"], d["pred_age_bucket"], ages), ages,
         "年龄段（融合可信度）"),
    ]:
        ax.imshow(cm, cmap="Greys", vmin=0, vmax=max(1, cm.max()))
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7.4)
        ax.set_yticklabels(labels, fontsize=7.4)
        ax.set_xlabel("预测", fontsize=8.0)
        ax.set_ylabel("真实", fontsize=8.0)
        ax.set_title(title, fontsize=9.0, color=INK)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=7.4,
                        color="white" if cm[i, j] > cm.max() / 2 else INK)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
            spine.set_linewidth(0.6)
        ax.tick_params(length=0)
    fig.tight_layout(pad=0.5)
    path = os.path.join(OUT, "confusions.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    print(pipeline())
    print(samples())
    print(method_comparison())
    print(confusions())


if __name__ == "__main__":
    main()
