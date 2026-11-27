# -*- coding: utf-8 -*-
"""Attribute the late-fusion improvement to text vs image evidence.

Uses the old (2026-11-26) component posteriors and the new (2026-11-27)
component posteriors, fits one consistent stacker on the old calib split,
and evaluates all four combinations on the same official test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "报告归档"
OUT_DIR = ARCHIVE / "07_2026-11-27_四种融合对比"


def _key(df: pd.DataFrame):
    return list(zip(df["author_id"], df["lang"]))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _stacker(cal: pd.DataFrame, test: pd.DataFrame, y_cal, y_test, tcol: str, icol: str):
    langs = sorted(cal["lang"].unique())
    def feats(df):
        lt, li = _logit(df[tcol].to_numpy()), _logit(df[icol].to_numpy())
        return np.column_stack([lt, li, lt * li] +
                               [df["lang"].eq(l).astype(float).to_numpy() for l in langs])
    clf = LogisticRegression(max_iter=3000, C=0.5).fit(feats(cal), y_cal)
    p = clf.predict_proba(feats(test))[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, p >= 0.5)),
        "log_loss": float(log_loss(y_test, np.clip(p, 1e-9, 1 - 1e-9))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="data/processed/pan18")
    ap.add_argument("--new", default="outputs/pan18_joint")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    old, new, out = Path(args.old), Path(args.new), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    otr = pd.read_csv(old / "text_predictions_calib.csv")
    ote = pd.read_csv(old / "text_predictions_test.csv")
    itr = pd.read_csv(old / "image_predictions_calib.csv")
    ite = pd.read_csv(old / "image_predictions_test.csv")
    nt = pd.read_csv(new / "text_probs.csv")
    ni = pd.read_csv(new / "image_probs.csv")
    ntmap = dict(zip(_key(nt), nt["p_female"]))
    nimap = dict(zip(_key(ni), ni["p_female"]))

    cal = otr.merge(itr[["author_id", "lang", "image_prob_female"]],
                    on=["author_id", "lang"], how="inner")
    cal["new_text"] = [ntmap.get(k, np.nan) for k in _key(cal)]
    cal["new_image"] = [nimap.get(k, np.nan) for k in _key(cal)]
    cal = cal.dropna(subset=["new_text", "new_image"]).reset_index(drop=True)

    test = ote.merge(ite[["author_id", "lang", "image_prob_female"]],
                     on=["author_id", "lang"], how="left")
    test["image_prob_female"] = test["image_prob_female"].fillna(0.5)
    test["new_text"] = [ntmap.get(k, 0.5) for k in _key(test)]
    test["new_image"] = [nimap.get(k, 0.5) for k in _key(test)]
    y_cal = (cal["gender"] == "female").astype(int).to_numpy()
    y_test = (test["gender"] == "female").astype(int).to_numpy()

    combos = [
        ("old_text_old_image", "旧文本 TF-IDF + 旧首图", "text_prob_female", "image_prob_female"),
        ("new_text_old_image", "新文本 XLM-R + 旧首图", "new_text", "image_prob_female"),
        ("old_text_new_image", "旧文本 TF-IDF + 新十图", "text_prob_female", "new_image"),
        ("new_text_new_image", "新文本 XLM-R + 新十图", "new_text", "new_image"),
    ]
    results = {}
    base = None
    for key, label, tcol, icol in combos:
        r = _stacker(cal, test, y_cal, y_test, tcol, icol)
        if base is None:
            base = r["accuracy"]
        r["label"] = label
        r["delta_vs_old"] = r["accuracy"] - base
        results[key] = r
    results["interaction_pp"] = (
        (results["new_text_new_image"]["accuracy"] - base)
        - (results["new_text_old_image"]["accuracy"] - base)
        - (results["old_text_new_image"]["accuracy"] - base)
    )
    payload = {
        "n": {"calib": int(len(cal)), "test": int(len(test))},
        "baseline": "旧文本（TF-IDF 全文）+ 旧图像（首图，ViT 分类头）",
        "stacker": "LR(C=0.5): logit(text), logit(image), interaction, language one-hot",
        "results": results,
    }
    (out / "four_fusion_attribution.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(results).T.to_csv(out / "four_fusion_attribution.csv", encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
