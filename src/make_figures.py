# -*- coding: utf-8 -*-
"""Generate pilot figures from outputs/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PRIMARY = "#1B2A4A"
GREEN = "#1B7D46"
AMBER = "#D4820A"
RED = "#C0392B"
GRAY = "#8C8A84"
LIGHT = "#D6E4F0"

OUT = "outputs/figures"
os.makedirs(OUT, exist_ok=True)
METHOD_LABELS = {
    "text_only": "Text-only",
    "face_only": "Face-only",
    "fused_equal": "Fused (equal)",
    "fused_cred": "Fused (credibility)",
    "fused_conflict": "Fused + conflict",
}


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def plot_confusion(ax, cm, labels, title):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(title, fontsize=10, color=PRIMARY)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else PRIMARY, fontsize=8)
    return im


def fig_method_comparison():
    m = json.load(open("outputs/metrics.json", encoding="utf-8"))
    methods = list(METHOD_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(methods))
    w = 0.36
    for ax, subset, title in [
        (axes[0], "clean", "Clean 50 cases"),
        (axes[1], "injected", "Injected 10 cases (stress test)"),
    ]:
        if subset == "clean":
            g = [m["clean"][k]["gender"]["accuracy"] for k in methods]
            a = [m["clean"][k]["age"]["accuracy"] for k in methods]
        else:
            g = [m["injected"][k]["injected_subset"]["gender"]["accuracy"] for k in methods]
            a = [m["injected"][k]["injected_subset"]["age"]["accuracy"] for k in methods]
        ax.bar(x - w / 2, g, w, label="Gender acc", color=PRIMARY)
        ax.bar(x + w / 2, a, w, label="Age-bucket acc", color=AMBER)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[k] for k in methods], rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10, color=PRIMARY)
        ax.grid(axis="y", alpha=0.25)
        for i, (gv, av) in enumerate(zip(g, a)):
            ax.text(i - w / 2, gv + 0.02, f"{gv:.2f}", ha="center", fontsize=6)
            ax.text(i + w / 2, av + 0.02, f"{av:.2f}", ha="center", fontsize=6)
    axes[0].set_ylabel("Accuracy", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Pilot method comparison", fontsize=12, color=PRIMARY)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "method_comparison.png"), dpi=180)
    plt.close(fig)


def fig_confusions():
    df = pd.read_csv("outputs/method_predictions_clean.csv")
    d = df[df["method"] == "fused_cred"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    plot_confusion(axes[0], confusion(d["true_gender"], d["pred_gender"], ["male", "female"]),
                   ["male", "female"], "Gender confusion (fused credibility)")
    plot_confusion(axes[1], confusion(d["true_age_bucket"], d["pred_age_bucket"],
                                     ["18-24", "25-29", "30-34", "35-40"]),
                   ["18-24", "25-29", "30-34", "35-40"], "Age-bucket confusion (fused credibility)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confusions.png"), dpi=180)
    plt.close(fig)


def fig_posterior_examples():
    clean = {r["case_id"]: r for r in load_jsonl("outputs/case_results_clean.jsonl")}
    inj = {r["case_id"]: r for r in load_jsonl("outputs/case_results_injected.jsonl")}
    cases = [("C001", clean["C001"], "Clean case C001"),
             ("C005", inj["C005"], "Injected case C005")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2))
    genders = ["male", "female"]
    ages = ["18-24", "25-29", "30-34", "35-40"]
    for row, (cid, r, title) in enumerate(cases):
        ax = axes[row, 0]
        vals = [r["gender_posterior"].get(g, 0) for g in genders]
        ax.bar(genders, vals, color=[PRIMARY, AMBER])
        ax.set_ylim(0, 1)
        ax.set_title(f"{title} — gender", fontsize=9, color=PRIMARY)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax = axes[row, 1]
        vals = [r["age_posterior"].get(a, 0) for a in ages]
        ax.bar(ages, vals, color=PRIMARY)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title} — age bucket", fontsize=9, color=PRIMARY)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Posterior examples", fontsize=12, color=PRIMARY)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "posterior_examples.png"), dpi=180)
    plt.close(fig)


def fig_contributions():
    inj = {r["case_id"]: r for r in load_jsonl("outputs/case_results_injected.jsonl")}
    r = inj["C005"]
    items = r["top_evidence_gender"][:3] + r["top_evidence_age"][:3]
    labels = [f"{x['evidence_id']}\n({x['source_type']})" for x in items]
    vals = [x["contribution_pred"] for x in items]
    colors = [GREEN if v >= 0 else RED for v in vals]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("Leave-one-out contribution to predicted label", fontsize=9)
    ax.set_title("C005 evidence contributions", fontsize=10, color=PRIMARY)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "evidence_contributions.png"), dpi=180)
    plt.close(fig)


def fig_conflict_distribution():
    clean = load_jsonl("outputs/case_results_clean.jsonl")
    inj = load_jsonl("outputs/case_results_injected.jsonl")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    c = [r["max_conflict"] for r in clean]
    i = [r["max_conflict"] for r in inj if r["injected"]]
    ax.scatter(np.zeros(len(c)) + np.random.RandomState(0).normal(0, 0.03, len(c)), c,
               color=PRIMARY, alpha=0.65, s=22, label="Clean cases")
    ax.scatter(np.ones(len(i)) + np.random.RandomState(1).normal(0, 0.03, len(i)), i,
               color=RED, alpha=0.85, s=30, label="Injected cases")
    ax.axhline(0.50, color=AMBER, linestyle="--", linewidth=1, label="τ = 0.50")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Clean", "Injected"])
    ax.set_ylabel("Max conflict", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Conflict scores: clean vs injected", fontsize=10, color=PRIMARY)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "conflict_distribution.png"), dpi=180)
    plt.close(fig)


def fig_graph(cid):
    g = json.load(open(f"outputs/evidence_graphs/{cid}.json", encoding="utf-8"))
    G = nx.DiGraph()
    for n in g["nodes"]:
        attrs = {k: v for k, v in n.items() if k != "id"}
        G.add_node(n["id"], **attrs)
    for e in g["edges"]:
        if e["source"] in G and e["target"] in G:
            G.add_edge(e["source"], e["target"], weight=e.get("weight", 0.0), kind=e.get("kind", ""))
    # manual three-column layout
    ev = [n for n in G.nodes if G.nodes[n]["type"] == "evidence"]
    hy = [n for n in G.nodes if G.nodes[n]["type"] == "hypothesis"]
    tg = [n for n in G.nodes if G.nodes[n]["type"] == "target"]
    pos = {}
    for i, n in enumerate(ev):
        pos[n] = (0, -i)
    for i, n in enumerate(hy):
        pos[n] = (1, -i * 0.8)
    for i, n in enumerate(tg):
        pos[n] = (2, -len(ev) / 2 + 0.5)
    colors = []
    for n in G.nodes:
        t = G.nodes[n]["type"]
        colors.append({"evidence": LIGHT, "hypothesis": AMBER, "target": GREEN}[t])
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(ev))))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=1800, edgecolors=PRIMARY)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6,
                            labels={n: n.replace("C" + cid[1:] + "-", "") for n in G.nodes})
    widths = [max(0.3, 2.5 * G[u][v].get("weight", 0.1)) for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, width=widths,
                           edge_color=GRAY, alpha=0.65, arrowsize=10)
    ax.set_title(f"{cid} evidence graph", fontsize=11, color=PRIMARY)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"evidence_graph_{cid}.png"), dpi=180)
    plt.close(fig)


def fig_age_bucket_accuracy():
    df = pd.read_csv("outputs/method_predictions_clean.csv")
    buckets = ["18-24", "25-29", "30-34", "35-40"]
    methods = ["text_only", "face_only", "fused_cred"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(buckets))
    w = 0.26
    for i, m in enumerate(methods):
        d = df[df["method"] == m]
        vals = [float((d[d["true_age_bucket"] == b]["pred_age_bucket"] == b).mean()) for b in buckets]
        ax.bar(x + (i - 1) * w, vals, w, label=METHOD_LABELS[m],
               color=[PRIMARY, AMBER, GREEN][i])
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_title("Age-bucket accuracy by true bucket", fontsize=10, color=PRIMARY)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "age_bucket_accuracy.png"), dpi=180)
    plt.close(fig)


def main():
    fig_method_comparison()
    fig_confusions()
    fig_posterior_examples()
    fig_contributions()
    fig_conflict_distribution()
    fig_graph("C005")
    fig_graph("C030")
    fig_age_bucket_accuracy()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
