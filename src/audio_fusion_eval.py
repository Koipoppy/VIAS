# -*- coding: utf-8 -*-
"""Evaluate audio/face/text evidence and late fusion on the CREMA-D set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "audio"
BUCKETS = ["18-24", "25-29", "30-34", "35-40"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(y, p, bins=10):
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    acc = (pred == y).astype(float)
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def gmetrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "log_loss": float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p),
    }


def bucket_of(age):
    if age <= 24.5:
        return "18-24"
    if age <= 29.5:
        return "25-29"
    if age <= 34.5:
        return "30-34"
    return "35-40"


def ametrics(y_age, y_bucket, pred_age, pred_bucket):
    return {
        "n": int(len(y_age)),
        "mae": float(np.mean(np.abs(pred_age - y_age))),
        "bias": float(np.mean(pred_age - y_age)),
        "bucket_accuracy": float(accuracy_score(y_bucket, pred_bucket)),
        "pm1_bucket": float(np.mean([abs(BUCKETS.index(a) - BUCKETS.index(b)) <= 1
                                     for a, b in zip(y_bucket, pred_bucket)])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(OUT / "audio_multimodal_cases.csv"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    d = pd.read_csv(args.cases)
    split = d["split"].to_numpy()
    y = (d["gender"] == "female").astype(int).to_numpy()
    y_age = d["age"].to_numpy(float)
    y_bucket = d["age_bucket"].to_numpy()

    text_raw = d["text_p_female"].to_numpy(float)
    face_raw = d["face_p_female"].to_numpy(float)
    audio_raw = d["audio_p_female"].to_numpy(float)

    # Platt calibration for text/face; audio is already near-perfect.
    def calib(p):
        m = LogisticRegression(max_iter=1000).fit(logit(p[split == "calib"]).reshape(-1, 1),
                                                  y[split == "calib"])
        return m.predict_proba(logit(p).reshape(-1, 1))[:, 1]

    text_c, face_c, audio_c = calib(text_raw), calib(face_raw), calib(audio_raw)
    feat3 = np.column_stack([logit(text_c), logit(face_c), logit(audio_c)])
    feat2 = np.column_stack([logit(text_c), logit(face_c)])
    f3 = LogisticRegression(max_iter=2000, C=0.5).fit(feat3[split == "calib"], y[split == "calib"])
    f2 = LogisticRegression(max_iter=2000, C=0.5).fit(feat2[split == "calib"], y[split == "calib"])
    p3 = f3.predict_proba(feat3)[:, 1]
    p2 = f2.predict_proba(feat2)[:, 1]

    methods = {
        "text": text_raw, "text_cal": text_c,
        "face": face_raw, "face_cal": face_c,
        "audio": audio_raw, "audio_cal": audio_c,
        "fusion_text_face": p2, "fusion_3": p3,
    }
    gender = {k: {"test": gmetrics(y[split == "test"], v[split == "test"]),
                  "calib": gmetrics(y[split == "calib"], v[split == "calib"])}
              for k, v in methods.items()}

    # Age: linear calibration/fusion on calib.
    age_features = {
        "text": d[["text_age_expected"]].to_numpy(float),
        "face": d[["face_age"]].to_numpy(float),
        "audio": d[["audio_age"]].to_numpy(float),
    }
    age_methods, age_preds = {}, {}
    for name, x in age_features.items():
        m = LinearRegression().fit(x[split == "calib"], y_age[split == "calib"])
        pred = m.predict(x)
        age_methods[name] = {"test": ametrics(y_age[split == "test"], y_bucket[split == "test"],
                                              pred[split == "test"],
                                              [bucket_of(v) for v in pred[split == "test"]]),
                             "calib": ametrics(y_age[split == "calib"], y_bucket[split == "calib"],
                                               pred[split == "calib"],
                                               [bucket_of(v) for v in pred[split == "calib"]])}
        age_preds[name] = pred
    for name, cols in [("fusion_face_audio", ["face_age", "audio_age"]),
                       ("fusion_text_face_audio", ["text_age_expected", "face_age", "audio_age"])]:
        x = d[cols].to_numpy(float)
        m = LinearRegression().fit(x[split == "calib"], y_age[split == "calib"])
        pred = m.predict(x)
        age_methods[name] = {"test": ametrics(y_age[split == "test"], y_bucket[split == "test"],
                                              pred[split == "test"],
                                              [bucket_of(v) for v in pred[split == "test"]]),
                             "calib": ametrics(y_age[split == "calib"], y_bucket[split == "calib"],
                                               pred[split == "calib"],
                                               [bucket_of(v) for v in pred[split == "calib"]])}
        age_preds[name] = pred

    out = {
        "n": {"fit": int((split == "fit").sum()), "calib": int((split == "calib").sum()),
              "test": int((split == "test").sum())},
        "gender": gender, "age": age_methods,
        "notes": [
            "audio and face are the same CREMA-D actor; text is a synthetic SMP pilot pairing",
            "text language zh, audio language en",
            "calib includes 60 clips from held-out actors; test 60 clips from disjoint actors",
        ],
    }
    (Path(args.out) / "audio_fusion_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pred = d[["case_id", "split", "gender", "age", "age_bucket"]].copy()
    for k, v in methods.items():
        pred[f"p_{k}"] = v
    for k, v in age_preds.items():
        pred[f"age_{k}"] = v
    pred.to_csv(Path(args.out) / "audio_fusion_predictions.csv", index=False, encoding="utf-8-sig")

    print("GENDER (test)")
    print(pd.DataFrame({k: v["test"] for k, v in gender.items()}).T[
        ["n", "accuracy", "macro_f1", "auc", "ece", "brier"]].to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nAGE (test)")
    print(pd.DataFrame({k: v["test"] for k, v in age_methods.items()}).T[
        ["n", "mae", "bias", "bucket_accuracy", "pm1_bucket"]].to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
