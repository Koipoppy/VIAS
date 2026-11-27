# -*- coding: utf-8 -*-
"""Figures for the PAN18 embedding-concat fusion report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ORDER = ["image_vit", "image", "text_tfidf", "text", "late_logpool", "late_stack",
         "concat", "ensemble", "concat_shuffled"]
LABELS = {
    "image_vit": "Image only (ViT 10-img)",
    "image": "Image only (linear)",
    "text_tfidf": "Text only (TF-IDF)",
    "text": "Text only (XLM-R)",
    "late_logpool": "Late log-pool",
    "late_stack": "Late stack",
    "concat": "Concat (ours)",
    "ensemble": "Concat + late",
    "concat_shuffled": "Concat, shuffled image",
}
COLORS = {
    "image_vit": "#CED4DA",
    "image": "#9AA5B1",
    "text_tfidf": "#748FFC",
    "text": "#4C6EF5",
    "late_logpool": "#F59F00",
    "late_stack": "#F76707",
    "concat": "#0CA678",
    "ensemble": "#12B886",
    "concat_shuffled": "#ADB5BD",
}


def load(out_dir: Path) -> dict:
    return json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))


def method_table(metrics: dict, split: str = "test") -> pd.DataFrame:
    rows = {}
    for m in ORDER:
        if m in metrics["results"]:
            r = metrics["results"][m][split]
            rows[LABELS[m]] = {"accuracy": r["accuracy"], "auc": r["auc"],
                               "macro_f1": r["macro_f1"], "ece": r["ece"],
                               "brier": r["brier"]}
    return pd.DataFrame(rows).T


def plot_methods(metrics: dict, out: Path) -> None:
    tbl = method_table(metrics)
    method_keys = [m for m in ORDER if m in metrics["results"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(tbl))
    for ax, metric, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "auc", "AUC")]:
        vals = tbl[metric].to_numpy()
        colors = [COLORS[m] for m in method_keys]
        bars = ax.barh(y, vals, color=colors, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(tbl.index)
        ax.invert_yaxis()
        ax.set_xlim(0.5 if metric == "accuracy" else 0.5, 1.0 if metric == "auc" else 0.9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="x", color="#E9ECEF", linewidth=0.8)
        ax.set_axisbelow(True)
        for b, v in zip(bars, vals):
            ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                    va="center", fontsize=8.5)
        if "Text only (XLM-R)" in tbl.index:
            base = tbl.loc["Text only (XLM-R)", metric]
            ax.axvline(base, color="#4C6EF5", linestyle="--", linewidth=0.9, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out / "pan18_joint_methods.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_language(metrics: dict, out: Path) -> None:
    methods = [m for m in ["image", "text", "late_stack", "concat", "ensemble",
                           "concat_shuffled"] if m in metrics["results"]]
    langs = ["en", "es", "ar"]
    x = np.arange(len(langs))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for i, m in enumerate(methods):
        vals = [metrics["results"][m]["per_lang"][l]["accuracy"] for l in langs]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width,
               label=LABELS[m], color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(["English", "Spanish", "Arabic"])
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "pan18_joint_per_language.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _box(ax, x, y, w, h, text, color="#F2F3F5", edge="#495057"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=1.0, edgecolor=edge, facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5,
            color="#1A1A1A", wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.0, color="#495057"))


def plot_schematic(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.02, 0.62, 0.15, 0.22, "100 tweets\n(XML)", color="#E7F5FF")
    _box(ax, 0.23, 0.62, 0.16, 0.22, "XLM-R encoder\n(fine-tuned)", color="#D0EBFF")
    _box(ax, 0.45, 0.62, 0.13, 0.22, "h_text\n768-d", color="#E7F5FF")
    _box(ax, 0.02, 0.16, 0.15, 0.22, "10 profile photos", color="#FFF4E6")
    _box(ax, 0.23, 0.16, 0.16, 0.22, "ViT encoder\n(frozen)", color="#FFE8CC")
    _box(ax, 0.45, 0.16, 0.13, 0.22, "mean h_image\n768-d", color="#FFF4E6")
    _box(ax, 0.64, 0.39, 0.10, 0.24, "proj\n256-d", color="#D3F9D8")
    _box(ax, 0.77, 0.39, 0.09, 0.24, "concat\n512-d", color="#B2F2BB")
    _box(ax, 0.88, 0.39, 0.10, 0.24, "MLP\nP(female)", color="#8CE99A")

    _arrow(ax, 0.17, 0.73, 0.23, 0.73)
    _arrow(ax, 0.39, 0.73, 0.45, 0.73)
    _arrow(ax, 0.17, 0.27, 0.23, 0.27)
    _arrow(ax, 0.39, 0.27, 0.45, 0.27)
    _arrow(ax, 0.58, 0.73, 0.64, 0.60)
    _arrow(ax, 0.58, 0.27, 0.64, 0.42)
    _arrow(ax, 0.74, 0.51, 0.77, 0.51)
    _arrow(ax, 0.86, 0.51, 0.88, 0.51)

    ax.text(0.5, 0.94, "Embedding-concat fusion (P1a)", ha="center", fontsize=12,
            fontweight="bold", color="#1A1A1A")
    ax.text(0.5, 0.03, "Text branch is trained end-to-end; the image branch is a frozen "
                       "gender ViT, mean-pooled over up to 10 photos.",
            ha="center", fontsize=8, color="#5B6770")
    fig.savefig(out / "pan18_joint_schematic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/pan18_joint")
    args = ap.parse_args()
    out = Path(args.out)
    metrics = load(out)
    plot_methods(metrics, out)
    plot_per_language(metrics, out)
    plot_schematic(out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
