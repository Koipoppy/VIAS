# -*- coding: utf-8 -*-
"""Predict PAN18 test text with the saved per-language text models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from pan18_text_model import read_tweets


def load_bundle(path):
    """Load a TF-IDF bundle across sklearn minor versions.

    Models trained with sklearn>=1.7 no longer carry ``multi_class``; sklearn
    1.6 still reads it in predict_proba. For the binary gender task the value
    does not change the decision rule, so restore the old default.
    """
    bundle = joblib.load(path)
    model = bundle["model"]
    if not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="en,es,ar")
    args = ap.parse_args()
    test = pd.read_csv(args.test_manifest)
    test["text"] = [read_tweets(p) for p in test["text_path"]]
    test = test[test["text"].str.len() > 0].reset_index(drop=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    metrics = {}
    frames = []
    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        d = test[test["lang"] == lang].reset_index(drop=True)
        model_path = Path(args.model_dir) / f"{lang}_text.joblib"
        if len(d) == 0 or not model_path.exists():
            continue
        bundle = load_bundle(model_path)
        proba = bundle["model"].predict_proba(bundle["vectorizer"].transform(d["text"]))
        classes = list(bundle["model"].classes_)
        female = proba[:, classes.index("female")]
        male = proba[:, classes.index("male")]
        pred = ["female" if f >= m else "male" for f, m in zip(female, male)]
        rec = d[["author_id", "gender", "lang", "n_chars"]].copy()
        rec["split"] = "test"
        rec["text_prob_female"] = female
        rec["text_prob_male"] = male
        rec["text_pred"] = pred
        frames.append(rec)
        metrics[lang] = {
            "n": int(len(d)),
            "accuracy": float(accuracy_score(d["gender"], pred)),
            "macro_f1": float(f1_score(d["gender"], pred, average="macro")),
        }
        rec.to_csv(out / f"text_predictions_test_{lang}.csv", index=False)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / "text_predictions_test.csv", index=False)
    (out / "text_test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("saved", out / "text_predictions_test.csv")


if __name__ == "__main__":
    main()
