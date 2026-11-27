# -*- coding: utf-8 -*-
"""Figures for the PAN18 concat vs cross-attention report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "pan18_joint" / "crossattn_2026-11-27"

COLORS = {
    "text": "#8C8C8C",
    "image": "#B0B0B0",
    "late_stack": "#4C72B0",
    "concat": "#DD8452",
    "crossattn_ens": "#55A868",
}


def load() -> dict:
    return json.loads((OUT / "metrics_crossattn.json").read_text(encoding="utf-8"))


def fig_methods(m: dict) -> None:
    methods = [("text", "Text"), ("image", "Image"), ("late_stack", "Late stack"),
               ("concat", "Concat"), ("crossattn_ens", "Cross-attn")]
    values = [m["results"][k]["test"]["accuracy"] for k, _ in methods]
    seed_std = m["seed_summary"]["accuracy_std"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    bars = ax.bar([label for _, label in methods], values,
                  color=[COLORS[k] for k, _ in methods], width=0.62)
    bars[-1].set_hatch("//")
    ax.errorbar(len(methods) - 1, values[-1], yerr=seed_std, fmt="none",
                ecolor="#222222", capsize=5, lw=1.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006,
                f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.70, 0.90)
    ax.set_ylabel("PAN18 test accuracy (5,100 users)")
    ax.set_title("Fusion comparison: concat vs cross-attention")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_methods.png", bbox_inches="tight")
    plt.close(fig)


def fig_per_language(m: dict) -> None:
    langs = ["en", "es", "ar"]
    concat = [m["results"]["concat"]["per_lang"][l]["accuracy"] for l in langs]
    cross = [m["results"]["crossattn_ens"]["per_lang"][l]["accuracy"] for l in langs]
    x = np.arange(len(langs))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    ax.bar(x - width / 2, concat, width, label="Concat (local)", color=COLORS["concat"])
    ax.bar(x + width / 2, cross, width, label="Cross-attn (3-seed)", color=COLORS["crossattn_ens"])
    for i, (a, b) in enumerate(zip(concat, cross)):
        ax.text(i - width / 2, a + 0.004, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([{"en": "English", "es": "Spanish", "ar": "Arabic"}[l] for l in langs])
    ax.set_ylim(0.70, 0.92)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-language test accuracy")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_language.png", bbox_inches="tight")
    plt.close(fig)


def fig_seeds(m: dict) -> None:
    seeds = m["seeds"]
    names = list(seeds)
    accs = [seeds[s]["test"]["accuracy"] for s in names]
    lls = [seeds[s]["test"]["log_loss"] for s in names]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), dpi=200)
    axes[0].plot(range(len(names)), accs, "o-", color=COLORS["crossattn_ens"])
    axes[0].axhline(m["results"]["crossattn_ens"]["test"]["accuracy"], ls="--",
                    color="#333333", lw=1, label="3-seed ensemble")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([s.replace("seed", "") for s in names])
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Seed variance: accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(range(len(names)), lls, "o-", color="#C44E52")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([s.replace("seed", "") for s in names])
    axes[1].set_ylabel("Test log loss")
    axes[1].set_title("Seed variance: log loss")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_seeds.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = load()
    fig_methods(m)
    fig_per_language(m)
    fig_seeds(m)
    print("wrote", OUT / "fig_methods.png")
    print("wrote", OUT / "fig_per_language.png")
    print("wrote", OUT / "fig_seeds.png")


if __name__ == "__main__":
    main()
