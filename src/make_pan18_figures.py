# -*- coding: utf-8 -*-
"""Figures for the PAN18 full multimodal evaluation report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "pan18_figures"
METRICS = ROOT / "outputs" / "pan18" / "fusion_metrics.json"
PRED = ROOT / "outputs" / "pan18" / "fusion_predictions.csv"
LANGS = ["en", "es", "ar"]
LANG_LABEL = {"en": "英语", "es": "西班牙语", "ar": "阿拉伯语"}
METHOD_LABEL = {
    "text_only": "文本", "image_only": "图像", "equal_prob": "等权",
    "credibility": "可信度", "calibrated_late": "校准融合",
    "stacked_lr": "堆叠融合", "conflict_handled": "冲突处理",
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["SimSun", "Times New Roman"],
    "font.size": 9, "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
INK = "#1A1A1A"; MUTED = "#4D4D4D"; GRID = "#B0B0B0"


def composition():
    fig, ax = plt.subplots(figsize=(6.6, 2.15), dpi=300)
    x = np.arange(len(LANGS)); w = 0.36
    train = [3000, 3000, 1500]; test = [1900, 2200, 1000]
    ax.bar(x - w / 2, train, w, color="#5B6F8A", edgecolor=INK, linewidth=.5, label="训练")
    ax.bar(x + w / 2, test, w, color="#D9D9D9", edgecolor=INK, linewidth=.5, label="测试")
    for bars in (ax.containers[0], ax.containers[1]):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 40,
                    f"{int(b.get_height())}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([LANG_LABEL[z] for z in LANGS])
    ax.set_ylabel("用户数", fontsize=8.5); ax.set_ylim(0, 3500)
    ax.grid(axis="y", linestyle=":", linewidth=.5, color=GRID, alpha=.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    fig.tight_layout(pad=.5)
    p = OUT / "pan18_composition.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=.02, facecolor="white")
    plt.close(fig); return p


def method_compare():
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    methods = ["text_only", "image_only", "calibrated_late", "stacked_lr"]
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.1), dpi=300, sharey=True)
    x = np.arange(len(methods)); colors = ["#5B6F8A", "#D9D9D9", "#8FA1B3", "#C9CED6"]
    for ax, lang in zip(axes, LANGS):
        vals = [data["results"][lang][m]["accuracy"] for m in methods]
        bars = ax.bar(x, vals, .62, color=colors, edgecolor=INK, linewidth=.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6.8)
        ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in methods],
                                             rotation=20, ha="right", fontsize=7)
        ax.set_title(LANG_LABEL[lang], fontsize=9)
        ax.set_ylim(.5, 1.04)
        ax.grid(axis="y", linestyle=":", linewidth=.5, color=GRID, alpha=.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("性别准确率", fontsize=8.2)
    fig.tight_layout(pad=.5)
    p = OUT / "pan18_methods.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=.02, facecolor="white")
    plt.close(fig); return p


def _cm(y, p):
    labels = ["female", "male"]
    idx = {v: i for i, v in enumerate(labels)}
    m = np.zeros((2, 2), dtype=int)
    for a, b in zip(y, p):
        m[idx[a], idx[b]] += 1
    return m


def confusions(method="stacked_lr"):
    df = pd.read_csv(PRED)
    d = df[df["method"] == method]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.55), dpi=300)
    for ax, lang in zip(axes, ["all", "en"]):
        q = d if lang == "all" else d[d["lang"] == lang]
        m = _cm(q["gender"].tolist(), q["pred"].tolist())
        ax.imshow(m, cmap="Greys", vmin=0, vmax=max(1, m.max()))
        labels = ["女", "男"]
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("预测"); ax.set_ylabel("真实")
        ax.set_title("全量" if lang == "all" else "英语", fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > m.max() / 2 else INK)
        for s in ax.spines.values():
            s.set_color("#2F3E4E"); s.set_linewidth(.6)
        ax.tick_params(length=0)
    fig.tight_layout(pad=.5)
    p = OUT / "pan18_confusions.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=.02, facecolor="white")
    plt.close(fig); return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(composition())
    print(method_compare())
    print(confusions())


if __name__ == "__main__":
    main()
