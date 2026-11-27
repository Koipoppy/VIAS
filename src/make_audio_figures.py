# -*- coding: utf-8 -*-
"""Figures for the audio multimodal dataset/report."""
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
OUT = ARCHIVE / "08_2026-11-28_音频多模态测试"
CASES = ROOT / "data" / "processed" / "audio" / "audio_multimodal_cases.csv"
METRICS = ROOT / "data" / "processed" / "audio" / "audio_final_metrics.json"

BUCKETS = ["18-24", "25-29", "30-34", "35-40"]


def plot_composition(cases: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(BUCKETS))
    for ax, split in zip(axes, ["fit", "test"]):
        sub = cases[cases.split == split]
        male = [int(((sub.gender == "male") & (sub.age_bucket == b)).sum()) for b in BUCKETS]
        female = [int(((sub.gender == "female") & (sub.age_bucket == b)).sum()) for b in BUCKETS]
        ax.bar(x, male, color="#4C6EF5", label="male")
        ax.bar(x, female, bottom=male, color="#F06595", label="female")
        ax.set_xticks(x)
        ax.set_xticklabels(BUCKETS)
        ax.set_title(f"{split} (n={len(sub)})", fontsize=10)
        ax.set_ylabel("clips")
        ax.grid(axis="y", color="#E9ECEF", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("CREMA-D subset composition", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "audio_dataset_composition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gender(m: dict, out: Path):
    methods = ["text_cal", "face_cal", "audio_cal", "fusion_text_face", "fusion_3"]
    labels = ["Text", "Face", "Audio", "Text+Face", "Text+Face+Audio"]
    acc = [m["gender"][k]["test"]["accuracy"] for k in methods]
    auc = [m["gender"][k]["test"]["auc"] for k in methods]
    x = np.arange(len(methods)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 3.8))
    b1 = ax.bar(x - w / 2, acc, w, color="#0CA678", label="Accuracy")
    b2 = ax.bar(x + w / 2, auc, w, color="#4C6EF5", label="AUC")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("score")
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                    f"{b.get_height():.3f}", ha="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "audio_gender_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_age(m: dict, out: Path):
    methods = ["text", "face_zero", "audio_zero", "audio_head", "face_head",
               "fusion_text_face", "fusion_3"]
    labels = ["Text", "Face 0-shot", "Audio 0-shot", "Audio head", "Face head",
              "Text+Face", "Text+Face+Audio"]
    mae = [m["age"][k]["test"]["mae"] for k in methods]
    bacc = [m["age"][k]["test"]["bucket_accuracy"] for k in methods]
    x = np.arange(len(methods)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    bars = ax.bar(x, mae, color="#F59F00", width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel("MAE (years)"); ax.set_ylim(0, 5.5)
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.8); ax.set_axisbelow(True)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08,
                f"{b.get_height():.2f}", ha="center", fontsize=7.5)
    ax = axes[1]
    bars = ax.bar(x, bacc, color="#7048E8", width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel("bucket accuracy"); ax.set_ylim(0, 0.6)
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.8); ax.set_axisbelow(True)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "audio_age_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(CASES)
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    plot_composition(cases, OUT)
    plot_gender(m, OUT)
    plot_age(m, OUT)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
