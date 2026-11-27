# -*- coding: utf-8 -*-
"""Train lightweight age-bucket heads on frozen audio/face embeddings.

The pretrained audio age head is poorly calibrated on CREMA-D (MAE ~14.7y
before calibration). We keep the frozen encoders and fit a 4-class age head on
the fit actors, selecting regularisation on calib; the test actors stay unseen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "audio"
BUCKETS = ["18-24", "25-29", "30-34", "35-40"]
CENTERS = np.array([21.0, 27.0, 32.0, 37.5])


def load_emb(path: Path, cases: pd.DataFrame) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    mapping = {str(c): i for i, c in enumerate(z["case_id"])}
    return np.stack([z["emb"][mapping[c]] for c in cases["case_id"]]).astype(np.float32)


def fit_head(x_fit, y_fit, x_cal, y_cal, y_age_cal, x_all):
    best = None
    for c in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        clf = LogisticRegression(max_iter=3000, C=c, class_weight="balanced")
        clf.fit(x_fit, y_fit)
        p_cal = clf.predict_proba(x_cal)
        order = [list(clf.classes_).index(b) for b in BUCKETS]
        p_cal = p_cal[:, order]
        age_cal = p_cal @ CENTERS
        mae = float(np.mean(np.abs(age_cal - y_age_cal)))
        acc = float(accuracy_score(y_cal, [BUCKETS[i] for i in p_cal.argmax(1)]))
        score = (mae, -acc)
        if best is None or score < best[0]:
            best = (score, c, clf)
    _, c, clf = best
    order = [list(clf.classes_).index(b) for b in BUCKETS]
    p = clf.predict_proba(x_all)[:, order]
    return p, p @ CENTERS, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(OUT / "audio_multimodal_cases.csv"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    cases = pd.read_csv(args.cases)
    split = cases["split"].to_numpy()
    y_bucket = cases["age_bucket"].to_numpy()
    y_age = cases["age"].to_numpy(float)
    fit = split == "fit"
    cal = split == "calib"
    test = split == "test"

    scaler = StandardScaler().fit(load_emb(OUT / "audio_embeddings.npz", cases)[fit])
    audio_x = scaler.transform(load_emb(OUT / "audio_embeddings.npz", cases))
    face_scaler = StandardScaler().fit(load_emb(OUT / "face_embeddings.npz", cases)[fit])
    face_x = face_scaler.transform(load_emb(OUT / "face_embeddings.npz", cases))

    out = {"n": {"fit": int(fit.sum()), "calib": int(cal.sum()), "test": int(test.sum())}}
    preds = cases[["case_id", "split", "gender", "age", "age_bucket"]].copy()
    for name, x in [("audio_head", audio_x), ("face_head", face_x)]:
        p, age, c = fit_head(x[fit], y_bucket[fit], x[cal], y_bucket[cal], y_age[cal], x)
        pred_bucket = [BUCKETS[i] for i in p.argmax(1)]
        preds[f"{name}_age"] = age
        preds[f"{name}_bucket"] = pred_bucket
        for b, i in zip(BUCKETS, range(4)):
            preds[f"{name}_p_{b}"] = p[:, i]
        out[name] = {
            "C": c,
            "calib_mae": float(np.mean(np.abs(age[cal] - y_age[cal]))),
            "calib_bucket_accuracy": float(accuracy_score(y_bucket[cal], np.array(pred_bucket)[cal])),
            "test_mae": float(np.mean(np.abs(age[test] - y_age[test]))),
            "test_bucket_accuracy": float(accuracy_score(y_bucket[test], np.array(pred_bucket)[test])),
            "test_pm1": float(np.mean([abs(BUCKETS.index(a) - BUCKETS.index(b)) <= 1
                                       for a, b in zip(y_bucket[test], np.array(pred_bucket)[test])])),
        }

    # Age fusion with text expected age: ridge on calib; test once.
    feats = np.column_stack([cases["text_age_expected"], preds["audio_head_age"], preds["face_head_age"]])
    ridge = Ridge(alpha=1.0).fit(feats[cal], y_age[cal])
    fused = ridge.predict(feats)
    fused_bucket = [BUCKETS[int(np.argmin(np.abs(CENTERS - v)))] for v in fused]
    out["fusion_text_audio_face"] = {
        "test_mae": float(np.mean(np.abs(fused[test] - y_age[test]))),
        "test_bucket_accuracy": float(accuracy_score(y_bucket[test], np.array(fused_bucket)[test])),
        "test_pm1": float(np.mean([abs(BUCKETS.index(a) - BUCKETS.index(b)) <= 1
                                   for a, b in zip(y_bucket[test], np.array(fused_bucket)[test])])),
        "coef": ridge.coef_.tolist(), "intercept": float(ridge.intercept_),
    }
    preds["age_fusion"] = fused
    preds["age_fusion_bucket"] = fused_bucket
    preds.to_csv(Path(args.out) / "audio_age_head_predictions.csv", index=False, encoding="utf-8-sig")
    (Path(args.out) / "audio_age_head_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
