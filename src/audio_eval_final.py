# -*- coding: utf-8 -*-
"""Final evaluation table for the CREMA-D audio+face+text test set."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "audio"
BUCKETS = ["18-24", "25-29", "30-34", "35-40"]
CENTERS = np.array([21.0, 27.0, 32.0, 37.5])


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(y, p, bins=10):
    conf = np.maximum(p, 1 - p); pred = (p >= .5).astype(int); acc = (pred == y).astype(float)
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def gm(y, p):
    pred = (p >= .5).astype(int)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1])),
            "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p)}


def bucket_of(age):
    return BUCKETS[int(np.argmin(np.abs(CENTERS - age)))]


def am(y_age, y_bucket, age):
    pred_b = [bucket_of(v) for v in age]
    return {"n": int(len(y_age)), "mae": float(np.mean(np.abs(age - y_age))),
            "bias": float(np.mean(age - y_age)),
            "bucket_accuracy": float(accuracy_score(y_bucket, pred_b)),
            "pm1": float(np.mean([abs(BUCKETS.index(a) - BUCKETS.index(b)) <= 1
                                  for a, b in zip(y_bucket, pred_b)]))}


def main():
    d = pd.read_csv(OUT / "audio_multimodal_cases.csv")
    h = pd.read_csv(OUT / "audio_age_head_predictions.csv", encoding="utf-8-sig")
    h = h[["case_id", "audio_head_age", "face_head_age",
           *[f"audio_head_p_{b}" for b in BUCKETS], *[f"face_head_p_{b}" for b in BUCKETS]]]
    d = d.merge(h, on="case_id", how="left")
    split = d["split"].to_numpy(); cal = split == "calib"; test = split == "test"
    y = (d["gender"] == "female").astype(int).to_numpy()
    y_age = d["age"].to_numpy(float); y_b = d["age_bucket"].to_numpy()

    text_raw = d.text_p_female.to_numpy(float); face_raw = d.face_p_female.to_numpy(float)
    audio_raw = d.audio_p_female.to_numpy(float)
    def platt(p):
        m = LogisticRegression(max_iter=1000).fit(logit(p[cal]).reshape(-1, 1), y[cal])
        return m.predict_proba(logit(p).reshape(-1, 1))[:, 1]
    text_c, face_c, audio_c = platt(text_raw), platt(face_raw), platt(audio_raw)
    f2 = LogisticRegression(max_iter=2000, C=0.5).fit(
        np.column_stack([logit(text_c), logit(face_c)])[cal], y[cal])
    f3 = LogisticRegression(max_iter=2000, C=0.5).fit(
        np.column_stack([logit(text_c), logit(face_c), logit(audio_c)])[cal], y[cal])
    p2 = f2.predict_proba(np.column_stack([logit(text_c), logit(face_c)]))[:, 1]
    p3 = f3.predict_proba(np.column_stack([logit(text_c), logit(face_c), logit(audio_c)]))[:, 1]
    gender = {k: {"test": gm(y[test], v[test]), "calib": gm(y[cal], v[cal])} for k, v in {
        "text": text_raw, "text_cal": text_c, "face": face_raw, "face_cal": face_c,
        "audio": audio_raw, "audio_cal": audio_c,
        "fusion_text_face": p2, "fusion_3": p3}.items()}

    # Age calibrations: linear per-modality on calib.
    age = {}
    for name, x in [("text", d.text_age_expected.to_numpy(float)),
                    ("audio_zero", d.audio_age.to_numpy(float)),
                    ("face_zero", d.face_age.to_numpy(float)),
                    ("audio_head", d.audio_head_age.to_numpy(float)),
                    ("face_head", d.face_head_age.to_numpy(float))]:
        m = LinearRegression().fit(x[cal].reshape(-1, 1), y_age[cal])
        age[name] = m.predict(x.reshape(-1, 1))
    # Fusion on calibrated ages: text + face (audio optional).
    X2 = np.column_stack([age["text"], age["face_head"]])
    X3 = np.column_stack([age["text"], age["face_head"], age["audio_head"]])
    age["fusion_text_face"] = Ridge(alpha=1.0).fit(X2[cal], y_age[cal]).predict(X2)
    age["fusion_3"] = Ridge(alpha=1.0).fit(X3[cal], y_age[cal]).predict(X3)
    # Probability ensemble: equal mean of bucket probabilities.
    tprob = d[[f"text_age_{b}" for b in BUCKETS]].to_numpy(float)
    aprob = d[[f"audio_head_p_{b}" for b in BUCKETS]].to_numpy(float)
    fprob = d[[f"face_head_p_{b}" for b in BUCKETS]].to_numpy(float)
    avg = (tprob + aprob + fprob) / 3
    age["prob_avg3"] = avg @ CENTERS
    age_metrics = {k: {"test": am(y_age[test], y_b[test], v[test]),
                       "calib": am(y_age[cal], y_b[cal], v[cal])} for k, v in age.items()}

    summary = {
        "n": {"fit": int((split == "fit").sum()), "calib": int(cal.sum()), "test": int(test.sum())},
        "actors": int(d.actor_id.nunique()),
        "gender": gender, "age": age_metrics,
        "dataset": {
            "audio_source": "CREMA-D (audio+video, same actor)",
            "text_source": "SMP 2016 pilot (zh, synthetic pairing)",
            "face_source": "CREMA-D video frames (same actor)",
            "audio_model": "audeering/wav2vec2-large-robust-6-ft-age-gender",
            "face_gender_model": "rizvandwiki/gender-classification",
            "face_age_model": "nateraw/vit-age-classifier",
            "synthetic_text_pairing": True,
        },
    }
    (OUT / "audio_final_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = d[["case_id", "actor_id", "split", "gender", "age", "age_bucket"]].copy()
    for k, v in gender.items():
        rows[f"p_{k}"] = v
    for k, v in age.items():
        rows[f"age_{k}"] = v
    rows.to_csv(OUT / "audio_final_predictions.csv", index=False, encoding="utf-8-sig")
    print("GENDER")
    print(pd.DataFrame({k: v["test"] for k, v in gender.items()}).T[
        ["accuracy", "macro_f1", "auc", "ece", "brier"]].to_string(float_format=lambda x: f"{x:.4f}"))
    print("AGE")
    print(pd.DataFrame({k: v["test"] for k, v in age_metrics.items()}).T[
        ["mae", "bias", "bucket_accuracy", "pm1"]].to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
