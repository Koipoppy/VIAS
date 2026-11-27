# -*- coding: utf-8 -*-
"""Figures for the PAN18 four-way fusion comparison report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "报告归档"
OUT = ARCHIVE / "07_2026-11-27_四种融合对比"
BASE = ROOT / "outputs" / "pan18_joint" / "metrics.json"
CROSS = ROOT / "outputs" / "pan18_joint" / "crossattn_2026-11-27" / "metrics_crossattn.json"
ATTR = OUT / "four_fusion_attribution.json"

METHODS = ["text", "image", "late_stack", "concat", "crossattn_ens", "ensemble"]
LABELS = {
    "text": "Text only",
    "image": "Image only",
    "late_stack": "① Late fusion",
    "concat": "② Embedding concat",
    "crossattn_ens": "③ Cross-attention",
    "ensemble": "④ Concat + late",
}
COLORS = {
    "text": "#4C6EF5",
    "image": "#9AA5B1",
    "late_stack": "#F59F00",
    "concat": "#0CA678",
    "crossattn_ens": "#7048E8",
    "ensemble": "#12B886",
}


def load():
    base = json.loads(BASE.read_text(encoding="utf-8"))["results"]
    cross = json.loads(CROSS.read_text(encoding="utf-8"))
    rows = {}
    for key in METHODS:
        src = cross["results"] if key == "crossattn_ens" else base
        if key not in src:
            continue
        rows[key] = {
            "acc": src[key]["test"]["accuracy"],
            "auc": src[key]["test"]["auc"],
            "ece": src[key]["test"]["ece"],
            "brier": src[key]["test"]["brier"],
            "log_loss": src[key]["test"]["log_loss"],
            "per_lang": {l: src[key]["per_lang"][l]["accuracy"] for l in ["en", "es", "ar"]},
        }
    return rows, cross


def plot_overall(rows, out: Path):
    keys = [k for k in METHODS if k in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    y = np.arange(len(keys))
    specs = [("acc", "Accuracy", (0.75, 0.90)), ("auc", "AUC", (0.82, 0.95)),
             ("ece", "ECE (lower is better)", (0.0, 0.03))]
    for ax, (metric, title, xlim) in zip(axes, specs):
        vals = [rows[k][metric] for k in keys]
        bars = ax.barh(y, vals, color=[COLORS[k] for k in keys], height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[k] for k in keys])
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="x", color="#E9ECEF", linewidth=0.8)
        ax.set_axisbelow(True)
        for b, v in zip(bars, vals):
            ax.text(v + (xlim[1] - xlim[0]) * 0.012, b.get_y() + b.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=8)
        if metric == "acc":
            ax.axvline(rows["text"]["acc"], color=COLORS["text"], linestyle="--",
                       linewidth=0.9, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out / "four_fusion_overall.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_language(rows, out: Path):
    keys = [k for k in ["text", "late_stack", "concat", "crossattn_ens", "ensemble"] if k in rows]
    langs = ["en", "es", "ar"]
    x = np.arange(len(langs))
    width = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(8.8, 4.1))
    for i, k in enumerate(keys):
        vals = [rows[k]["per_lang"][l] for l in langs]
        ax.bar(x + (i - (len(keys) - 1) / 2) * width, vals, width,
               label=LABELS[k], color=COLORS[k])
    ax.set_xticks(x)
    ax.set_xticklabels(["English", "Spanish", "Arabic"])
    ax.set_ylim(0.75, 0.90)
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "four_fusion_per_language.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_seeds(cross, out: Path):
    seeds = [("20261127", 0.8539), ("20261128", 0.8598), ("20261129", 0.8412)]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    names = [s[0][-2:] for s in seeds]
    vals = [s[1] for s in seeds]
    bars = ax.bar(names, vals, color="#7048E8", width=0.5)
    ens = cross["results"]["crossattn_ens"]["test"]["accuracy"]
    mean = cross["seed_summary"]["accuracy_mean"]
    std = cross["seed_summary"]["accuracy_std"]
    ax.axhline(ens, color="#0CA678", linewidth=1.4, label=f"3-seed ensemble {ens:.4f}")
    ax.axhline(mean, color="#495057", linewidth=0.9, linestyle="--",
               label=f"seed mean {mean:.4f} ± {std:.4f}")
    ax.fill_between([-0.5, 2.5], mean - std, mean + std, color="#495057", alpha=0.08)
    ax.set_ylim(0.82, 0.88)
    ax.set_xlabel("Seed (202611xx)")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.8)
    ax.set_axisbelow(True)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "four_fusion_crossattn_seeds.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_attribution(out: Path):
    data = json.loads(ATTR.read_text(encoding="utf-8"))
    keys = ["old_text_old_image", "new_text_old_image", "old_text_new_image", "new_text_new_image"]
    labels = ["Old text + old image", "New text + old image",
              "Old text + new image", "New text + new image"]
    vals = [data["results"][k]["accuracy"] for k in keys]
    base = vals[0]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    colors = ["#ADB5BD", "#4C6EF5", "#F59F00", "#0CA678"]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.axhline(base, color="#495057", linestyle="--", linewidth=0.9,
               label=f"old + old = {base:.4f}")
    ax.set_ylim(0.75, 0.89)
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for b, v in zip(bars, vals):
        delta = (v - base) * 100
        text = f"{v:.4f}" + (f"\n({delta:+.2f} pp)" if b is not bars[0] else "")
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, text,
                ha="center", va="bottom", fontsize=8.5)
    plt.setp(ax.get_xticklabels(), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "four_fusion_attribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cross = load()
    plot_overall(rows, OUT)
    plot_per_language(rows, OUT)
    plot_seeds(cross, OUT)
    if ATTR.exists():
        plot_attribution(OUT)
    pd.DataFrame(rows).T.to_csv(OUT / "four_fusion_metrics.csv", encoding="utf-8-sig")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
