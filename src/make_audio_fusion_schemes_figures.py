# -*- coding: utf-8 -*-
"""Figures for the audio fusion-scheme comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "报告归档"
OUT = ARCHIVE / "09_2026-11-28_音频融合方案对比"
METRICS = ROOT / "data" / "processed" / "audio" / "audio_fusion_schemes_metrics.json"

AGE_ORDER = ["single_text", "single_face", "single_audio", "late_equal", "late_ridge",
             "late_text_face", "concat_ridge", "concat_text_face", "gated", "attention", "oracle"]
AGE_LABELS = ["Text", "Face", "Audio", "Late equal", "Late ridge", "Late T+F",
              "Concat 3", "Concat T+F", "Gated", "Attention", "Oracle"]
COLORS = ["#4C6EF5", "#F06595", "#F59F00", "#ADB5BD", "#868E96", "#495057",
          "#0CA678", "#12B886", "#7048E8", "#5F3DC4", "#CED4DA"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    age = m["age"]
    mae = [age[k]["test"]["mae"] for k in AGE_ORDER]
    err = None
    lows, highs = [], []
    for k in AGE_ORDER:
        ci = age[k].get("test_ci")
        if ci:
            lows.append(mae[AGE_ORDER.index(k)] - ci["mae_ci"][0])
            highs.append(ci["mae_ci"][1] - mae[AGE_ORDER.index(k)])
        else:
            lows.append(0); highs.append(0)
    yerr = np.array([lows, highs])
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    bars = ax.bar(range(len(AGE_ORDER)), mae, color=COLORS, width=0.65)
    ax.errorbar(range(len(AGE_ORDER)), mae, yerr=yerr, fmt="none", ecolor="#343A40",
                elinewidth=0.8, capsize=2)
    ax.set_xticks(range(len(AGE_ORDER))); ax.set_xticklabels(AGE_LABELS, rotation=25, ha="right", fontsize=7.8)
    ax.set_ylabel("MAE (years)"); ax.set_ylim(0, 6.6)
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.8); ax.set_axisbelow(True)
    for b, v in zip(bars, mae):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_title("Age MAE with actor-level 95% CI", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "audio_fusion_age_mae.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    bacc = [age[k]["test"]["bucket_accuracy"] for k in AGE_ORDER]
    lows, highs = [], []
    for k in AGE_ORDER:
        ci = age[k].get("test_ci")
        if ci:
            v = age[k]["test"]["bucket_accuracy"]
            lows.append(v - ci["bucket_ci"][0]); highs.append(ci["bucket_ci"][1] - v)
        else:
            lows.append(0); highs.append(0)
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    bars = ax.bar(range(len(AGE_ORDER)), bacc, color=COLORS, width=0.65)
    ax.errorbar(range(len(AGE_ORDER)), bacc, yerr=np.array([lows, highs]), fmt="none",
                ecolor="#343A40", elinewidth=0.8, capsize=2)
    ax.set_xticks(range(len(AGE_ORDER))); ax.set_xticklabels(AGE_LABELS, rotation=25, ha="right", fontsize=7.8)
    ax.set_ylabel("bucket accuracy"); ax.set_ylim(0, 0.85)
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.8); ax.set_axisbelow(True)
    for b, v in zip(bars, bacc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_title("Age bucket accuracy with actor-level 95% CI", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "audio_fusion_age_bucket.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    g = m["gender"]
    order = ["text", "face", "audio", "late_text_face", "late_3", "concat_logreg"]
    labels = ["Text", "Face", "Audio", "Late T+F", "Late 3", "Concat"]
    acc = [g[k]["test"]["accuracy"] for k in order]
    auc = [g[k]["test"]["auc"] for k in order]
    x = np.arange(len(order)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 3.7))
    b1 = ax.bar(x - w / 2, acc, w, color="#0CA678", label="Accuracy")
    b2 = ax.bar(x + w / 2, auc, w, color="#4C6EF5", label="AUC")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0.5, 1.03); ax.set_ylabel("score")
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                    f"{b.get_height():.3f}", ha="center", fontsize=7.5)
    ax.set_title("Gender: fusion schemes", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "audio_fusion_gender.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
