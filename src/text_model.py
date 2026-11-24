# -*- coding: utf-8 -*-
"""Train text-only gender and age-bucket models on SMP 2016.

The pilot users are excluded from training to avoid leakage.
Outputs:
  models/text_vectorizer.joblib
  models/text_gender.joblib
  models/text_age.joblib
  data/processed/text_predictions.csv
  data/processed/self_reports.csv
  reports/text_model_metrics.json
"""
import argparse
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

AGE_BUCKETS = ["18-24", "25-29", "30-34", "35-40"]
GENDERS = ["male", "female"]

GENDER_PATTERNS = [
    (r"(?:我|本人|俺)(?:是|一个|一名)?(?:男|女)(?:生|人|孩|的)?", 0.85),
    (r"性别[:：]?(男|女)", 0.9),
    (r"(?:^|[，。！？\s])(男|女)生(?:[，。！？\s]|$)", 0.65),
    (r"(?:我|本人)(?:是|为)?(男|女)(?:的)?(?:[，。！？\s]|$)", 0.8),
]

AGE_PATTERNS = [
    (r"(?:今年|今年已经|已经)?(\d{2})岁", 0.8),
    (r"(?:19|20)(\d{2})年(?:出生|生人|生)", 0.75),
    (r"(?:我|本人)(?:是|为)?(\d{2})后", 0.6),
    (r"(\d{2})后(?:[，。！？\s]|$)", 0.5),
]


def extract_self_report(text: str):
    """Return (gender, gender_conf, age, age_conf, snippets)."""
    t = text.replace(" ", "")
    g_votes = {}
    a_votes = {}
    snippets = []
    for pat, conf in GENDER_PATTERNS:
        for m in re.finditer(pat, t):
            s = m.group(0)
            if "男" in s:
                g = "male"
            elif "女" in s:
                g = "female"
            else:
                continue
            g_votes[g] = g_votes.get(g, 0) + conf
            snippets.append(s[:40])
    for pat, conf in AGE_PATTERNS:
        for m in re.finditer(pat, t):
            try:
                if "年" in m.group(0) and "出生" in m.group(0):
                    year = int(m.group(1))
                    year = 1900 + year if year > 30 else 2000 + year
                    age = 2016 - year
                else:
                    age = int(m.group(1))
            except Exception:
                continue
            if 10 <= age <= 80:
                a_votes[age] = a_votes.get(age, 0) + conf
                snippets.append(m.group(0)[:40])
    g = max(g_votes, key=g_votes.get) if g_votes else None
    g_conf = min(0.95, g_votes[g]) if g else 0.0
    a = max(a_votes, key=a_votes.get) if a_votes else None
    a_conf = min(0.95, a_votes[a]) if a else 0.0
    return g, g_conf, a, a_conf, snippets[:5]


def bucket_of(age: int) -> str:
    if age <= 24:
        return "18-24"
    if age <= 29:
        return "25-29"
    if age <= 34:
        return "30-34"
    return "35-40"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/smp_train.csv")
    ap.add_argument("--pilot", default="data/processed/smp_pilot.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    train = pd.read_csv(args.train)
    pilot = pd.read_csv(args.pilot)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        max_features=100000,
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vec.fit_transform(train["text_clean"].fillna(""))
    X_pilot = vec.transform(pilot["text_clean"].fillna(""))

    gender_clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    gender_clf.fit(X_train, train["gender"])

    age_clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    age_clf.fit(X_train, train["age_bucket"])

    g_pred = gender_clf.predict(X_pilot)
    g_prob = gender_clf.predict_proba(X_pilot)
    a_pred = age_clf.predict(X_pilot)
    a_prob = age_clf.predict_proba(X_pilot)

    g_classes = list(gender_clf.classes_)
    a_classes = list(age_clf.classes_)

    rows = []
    for i, row in pilot.reset_index(drop=True).iterrows():
        rec = {
            "uid": row["uid"],
            "true_gender": row["gender"],
            "true_age": int(row["age"]),
            "true_age_bucket": row["age_bucket"],
            "pred_gender": g_pred[i],
            "pred_age_bucket": a_pred[i],
            "text_gender_male": float(g_prob[i, g_classes.index("male")]),
            "text_gender_female": float(g_prob[i, g_classes.index("female")]),
        }
        for b in AGE_BUCKETS:
            rec[f"text_age_{b}"] = float(a_prob[i, a_classes.index(b)]) if b in a_classes else 0.0
        rows.append(rec)
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(os.path.join(args.out_dir, "text_predictions.csv"), index=False)

    # self-report evidence
    sr_rows = []
    for _, row in pilot.iterrows():
        g, gc, a, ac, snips = extract_self_report(row["text_clean"])
        sr_rows.append(
            {
                "uid": row["uid"],
                "self_gender": g,
                "self_gender_conf": gc,
                "self_age": a,
                "self_age_conf": ac,
                "self_age_bucket": bucket_of(a) if a is not None else None,
                "snippets": " | ".join(snips),
            }
        )
    sr_df = pd.DataFrame(sr_rows)
    sr_df.to_csv(os.path.join(args.out_dir, "self_reports.csv"), index=False)

    metrics = {
        "n_train": int(len(train)),
        "n_pilot": int(len(pilot)),
        "gender_accuracy": float(accuracy_score(pilot["gender"], g_pred)),
        "gender_f1_macro": float(f1_score(pilot["gender"], g_pred, average="macro")),
        "age_accuracy": float(accuracy_score(pilot["age_bucket"], a_pred)),
        "age_f1_macro": float(f1_score(pilot["age_bucket"], a_pred, average="macro")),
        "gender_report": classification_report(pilot["gender"], g_pred, output_dict=True),
        "age_report": classification_report(pilot["age_bucket"], a_pred, output_dict=True),
        "self_report_gender_coverage": float(sr_df["self_gender"].notna().mean()),
        "self_report_age_coverage": float(sr_df["self_age"].notna().mean()),
    }
    with open(os.path.join(args.report_dir, "text_model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    joblib.dump(vec, os.path.join(args.model_dir, "text_vectorizer.joblib"))
    joblib.dump(gender_clf, os.path.join(args.model_dir, "text_gender.joblib"))
    joblib.dump(age_clf, os.path.join(args.model_dir, "text_age.joblib"))

    print(json.dumps({k: v for k, v in metrics.items() if not k.endswith("_report")}, ensure_ascii=False, indent=2))
    print("self-report gender coverage", metrics["self_report_gender_coverage"])
    print("self-report age coverage", metrics["self_report_age_coverage"])
    print(sr_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
