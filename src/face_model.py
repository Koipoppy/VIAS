# -*- coding: utf-8 -*-
"""Run pretrained face age/gender models on the 50 pilot images.

Models:
  nateraw/vit-age-classifier          9 age buckets
  rizvandwiki/gender-classification   female / male

Outputs:
  data/processed/face_predictions.csv
  reports/face_model_metrics.json
"""
import argparse
import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

AGE_BUCKETS = ["18-24", "25-29", "30-34", "35-40"]
AGE_SOURCE = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

# Column-stochastic mapping: source bucket -> target bucket.
# 20-29 is split evenly between 18-24 and 25-29; 30-39 between 30-34 and 35-40.
AGE_MAP = {
    "0-2": {"18-24": 1.0},
    "3-9": {"18-24": 1.0},
    "10-19": {"18-24": 1.0},
    "20-29": {"18-24": 0.5, "25-29": 0.5},
    "30-39": {"30-34": 0.5, "35-40": 0.5},
    "40-49": {"35-40": 1.0},
    "50-59": {"35-40": 1.0},
    "60-69": {"35-40": 1.0},
    "70+": {"35-40": 1.0},
}


def map_age(probs: dict) -> dict:
    out = {b: 0.0 for b in AGE_BUCKETS}
    for src, p in probs.items():
        for tgt, w in AGE_MAP.get(src, {}).items():
            out[tgt] += p * w
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/processed/pilot_cases.csv")
    ap.add_argument("--out", default="data/processed/face_predictions.csv")
    ap.add_argument("--report", default="reports/face_model_metrics.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from transformers import pipeline

    cases = pd.read_csv(args.cases)
    age_pipe = pipeline("image-classification", model="nateraw/vit-age-classifier", device=0 if args.device == "cuda" else -1)
    gender_pipe = pipeline("image-classification", model="rizvandwiki/gender-classification", device=0 if args.device == "cuda" else -1)

    rows = []
    for i, row in cases.iterrows():
        img = row["image_path"]
        age_raw = {d["label"]: float(d["score"]) for d in age_pipe(img, top_k=9)}
        gender_raw = {d["label"]: float(d["score"]) for d in gender_pipe(img, top_k=2)}
        # normalize / fill
        age_probs = {k: age_raw.get(k, 0.0) for k in AGE_SOURCE}
        tot = sum(age_probs.values())
        if tot > 0:
            age_probs = {k: v / tot for k, v in age_probs.items()}
        male = float(gender_raw.get("male", 0.0))
        female = float(gender_raw.get("female", 0.0))
        s = male + female
        if s > 0:
            male, female = male / s, female / s
        age4 = map_age(age_probs)
        rec = {
            "case_id": row["case_id"],
            "uid": row["uid"],
            "true_gender": row["gender"],
            "true_age": int(row["age"]),
            "true_age_bucket": row["age_bucket"],
            "face_gender_male": male,
            "face_gender_female": female,
            "face_pred_gender": "male" if male >= female else "female",
            "face_age_pred_bucket": max(age4, key=age4.get),
            "face_age_raw_top": max(age_probs, key=age_probs.get),
        }
        for b in AGE_BUCKETS:
            rec[f"face_age_{b}"] = float(age4[b])
        for k in AGE_SOURCE:
            rec[f"face_age_raw_{k}"] = float(age_probs[k])
        rows.append(rec)
        print(f"{i+1}/{len(cases)} {row['case_id']} age={age_probs} gender={male:.3f}/{female:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    metrics = {
        "n": int(len(df)),
        "gender_accuracy": float(accuracy_score(df["true_gender"], df["face_pred_gender"])),
        "gender_f1_macro": float(f1_score(df["true_gender"], df["face_pred_gender"], average="macro")),
        "age_accuracy": float(accuracy_score(df["true_age_bucket"], df["face_age_pred_bucket"])),
        "age_f1_macro": float(f1_score(df["true_age_bucket"], df["face_age_pred_bucket"], average="macro")),
        "gender_report": classification_report(df["true_gender"], df["face_pred_gender"], output_dict=True),
        "age_report": classification_report(df["true_age_bucket"], df["face_age_pred_bucket"], output_dict=True),
        "age_raw_top_distribution": df["face_age_raw_top"].value_counts().to_dict(),
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if not k.endswith("_report")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
