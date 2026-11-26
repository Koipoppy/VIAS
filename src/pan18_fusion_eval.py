# -*- coding: utf-8 -*-
"""Multimodal fusion evaluation on PAN18 (gender, text + image).

Methods:
  text_only, image_only, equal_prob, credibility (weighted log pool),
  calibrated_late (weights tuned on calibration split), conflict_handled.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vias_system" / "src"))
from vias.core.credibility import score_evidence  # noqa: E402
from vias.core.fusion import conflict_handle, weighted_fuse  # noqa: E402
from vias.core.schema import EvidenceUnit  # noqa: E402

LANGS = ["en", "es", "ar"]


def load_pair(text_path: Path, image_path: Path) -> pd.DataFrame:
    t = pd.read_csv(text_path)
    i = pd.read_csv(image_path)
    d = t.merge(i, on=["author_id", "lang", "gender"], how="left")
    d["text_prob_female"] = d["text_prob_female"].clip(1e-6, 1 - 1e-6)
    d["image_prob_female"] = d["image_prob_female"].fillna(0.5).clip(1e-6, 1 - 1e-6)
    return d


def evidence(row, modality: str, p_female: float) -> EvidenceUnit:
    p = {"female": float(p_female), "male": 1.0 - float(p_female)}
    if modality == "text":
        c = score_evidence("text_model", n_chars=int(getattr(row, "n_chars", 0) or 0))
        return EvidenceUnit(
            schema_version="1.1", evidence_id=f"{row.author_id}-T", case_id=row.author_id,
            modality="text", source_type="text_model", source_name="pan18_tfidf",
            content={}, gender_probs=p, credibility=c,
        )
    c = score_evidence("face_model", face_quality=0.8)
    return EvidenceUnit(
        schema_version="1.1", evidence_id=f"{row.author_id}-I", case_id=row.author_id,
        modality="image", source_type="face_model", source_name="vit_gender",
        content={}, gender_probs=p, credibility=c,
    )


def pred_from_prob(f: float) -> str:
    return "female" if f >= 0.5 else "male"


def metric(y, pred, prob_f):
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "auc": round(float(roc_auc_score([1 if x == "female" else 0 for x in y], prob_f)), 4),
    }


def run_method(df: pd.DataFrame, method: str, text_w: float = 0.5):
    probs = []
    for row in df.itertuples(index=False):
        tf, im = float(row.text_prob_female), float(row.image_prob_female)
        if method == "text_only":
            f = tf
        elif method == "image_only":
            f = im
        elif method == "equal_prob":
            f = 0.5 * tf + 0.5 * im
        elif method == "credibility":
            post, _ = weighted_fuse(
                {"female": 0.5, "male": 0.5},
                [evidence(row, "text", tf), evidence(row, "image", im)],
                "gender", use_credibility=True, weight_power=1.5,
            )
            f = post["female"]
        elif method == "calibrated_late":
            logf = text_w * math.log(tf) + (1 - text_w) * math.log(im)
            logm = text_w * math.log(1 - tf) + (1 - text_w) * math.log(1 - im)
            m = max(logf, logm)
            f = math.exp(logf - m) / (math.exp(logf - m) + math.exp(logm - m))
        elif method == "conflict_handled":
            post, log, ev, mx, pair, used = conflict_handle(
                [evidence(row, "text", tf), evidence(row, "image", im)],
                "gender", tau=0.65, downweight=0.25, max_rounds=3,
                weight_power=1.5,
            )
            f = post["female"]
        else:
            raise ValueError(method)
        probs.append(f)
    y = df["gender"].tolist()
    return metric(y, [pred_from_prob(x) for x in probs], probs)


def _features(df: pd.DataFrame):
    import numpy as np
    tf = df["text_prob_female"].clip(1e-5, 1 - 1e-5).to_numpy()
    im = df["image_prob_female"].clip(1e-5, 1 - 1e-5).to_numpy()
    lt = np.log(tf / (1 - tf))
    li = np.log(im / (1 - im))
    return np.column_stack([lt, li, tf, im, tf * im, np.abs(tf - im)])


def fit_stacked(calib: pd.DataFrame, lang: str | None = None):
    d = calib if lang is None else calib[calib["lang"] == lang]
    model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    model.fit(_features(d), (d["gender"] == "female").astype(int))
    return model


def predict_stacked(model, df: pd.DataFrame):
    return model.predict_proba(_features(df))[:, 1].tolist()


def best_text_weight(calib: pd.DataFrame):
    best = (0.5, -1.0)
    for w in [x / 20 for x in range(21)]:
        m = run_method(calib, "calibrated_late", w)
        if m["accuracy"] > best[1]:
            best = (w, m["accuracy"])
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.processed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    calib = []
    test = []
    for lang in LANGS:
        calib.append(load_pair(root / f"text_predictions_calib_{lang}.csv",
                               root / f"image_predictions_calib_{lang}.csv"))
        test.append(load_pair(root / f"text_predictions_test_{lang}.csv",
                              root / f"image_predictions_test_{lang}.csv"))
    calib = pd.concat(calib, ignore_index=True)
    test = pd.concat(test, ignore_index=True)

    methods = ["text_only", "image_only", "equal_prob", "credibility",
               "calibrated_late", "stacked_lr", "conflict_handled"]
    weights = {lang: best_text_weight(calib[calib["lang"] == lang]) for lang in LANGS}
    stacked = {lang: fit_stacked(calib, lang) for lang in LANGS}
    stacked_all = fit_stacked(calib)
    results = {}
    for lang in LANGS + ["all"]:
        d = test if lang == "all" else test[test["lang"] == lang]
        results[lang] = {}
        for method in methods:
            if method == "stacked_lr":
                probs = predict_stacked(stacked[lang], d) if lang != "all" else predict_stacked(stacked_all, d)
                results[lang][method] = metric(d["gender"], [pred_from_prob(x) for x in probs], probs)
            elif method == "calibrated_late":
                # Per-language weight; for the pooled table use the language-specific weight.
                probs = []
                for row in d.itertuples(index=False):
                    w = weights[row.lang]
                    tf, im = float(row.text_prob_female), float(row.image_prob_female)
                    lf = w * math.log(tf) + (1 - w) * math.log(im)
                    lm = w * math.log(1 - tf) + (1 - w) * math.log(1 - im)
                    m = max(lf, lm)
                    probs.append(math.exp(lf - m) / (math.exp(lf - m) + math.exp(lm - m)))
                results[lang][method] = metric(d["gender"], [pred_from_prob(x) for x in probs], probs)
            else:
                results[lang][method] = run_method(d, method)
    # Persist test probabilities/predictions for confusion matrices/reporting.
    pred_rows = []
    for method in methods:
        for row in test.itertuples(index=False):
            tf, im = float(row.text_prob_female), float(row.image_prob_female)
            if method == "text_only":
                f = tf
            elif method == "image_only":
                f = im
            elif method == "equal_prob":
                f = 0.5 * tf + 0.5 * im
            elif method == "credibility":
                post, _ = weighted_fuse(
                    {"female": 0.5, "male": 0.5},
                    [evidence(row, "text", tf), evidence(row, "image", im)],
                    "gender", use_credibility=True, weight_power=1.5)
                f = post["female"]
            elif method == "calibrated_late":
                w = weights[row.lang]
                lf = w * math.log(tf) + (1 - w) * math.log(im)
                lm = w * math.log(1 - tf) + (1 - w) * math.log(1 - im)
                m = max(lf, lm)
                f = math.exp(lf - m) / (math.exp(lf - m) + math.exp(lm - m))
            elif method == "stacked_lr":
                f = float(predict_stacked(stacked[row.lang], pd.DataFrame([row._asdict()]))[0])
            else:
                post, *_ = conflict_handle(
                    [evidence(row, "text", tf), evidence(row, "image", im)],
                    "gender", tau=0.65, downweight=0.25, max_rounds=3,
                    weight_power=1.5)
                f = post["female"]
            pred_rows.append({"author_id": row.author_id, "lang": row.lang,
                              "gender": row.gender, "method": method,
                              "prob_female": f, "pred": pred_from_prob(f)})
    pd.DataFrame(pred_rows).to_csv(out / "fusion_predictions.csv", index=False)
    payload = {"weights": weights, "results": results,
               "n": {k: int((test["lang"] == k).sum()) for k in LANGS} | {"all": int(len(test))}}
    (out / "fusion_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("saved", out / "fusion_metrics.json")


if __name__ == "__main__":
    main()
