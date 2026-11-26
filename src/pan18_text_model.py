# -*- coding: utf-8 -*-
"""Train/evaluate per-language PAN18 text gender models.

Outputs:
  pan18/processed/text_train.csv
  pan18/processed/text_predictions_{calib,test}.csv
  pan18/processed/text_metrics.json
  pan18/processed/models/{lang}_text.joblib
"""
from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def read_tweets(path: str) -> str:
    try:
        root = ET.fromstring(Path(path).read_text(encoding="utf-8", errors="ignore"))
        docs = [d.text or "" for d in root.findall(".//document")]
        return " ".join(x.strip() for x in docs if x.strip())
    except Exception:
        return ""


def stratified_split(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    fit_idx, cal_idx = [], []
    for _, group in df.groupby(["lang", "gender"]):
        idx = list(group.index)
        rng.shuffle(idx)
        n_cal = max(1, int(len(idx) * frac))
        cal_idx.extend(idx[:n_cal])
        fit_idx.extend(idx[n_cal:])
    return df.loc[fit_idx].copy(), df.loc[cal_idx].copy()


def fit_predict(fit: pd.DataFrame, targets: list[pd.DataFrame], seed: int):
    model = LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0,
                               random_state=seed)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                          max_features=250000, min_df=2, sublinear_tf=True,
                          lowercase=True, strip_accents="unicode")
    x = vec.fit_transform(fit["text"])
    y = fit["gender"]
    model.fit(x, y)
    classes = list(model.classes_)
    out = []
    for target in targets:
        proba = model.predict_proba(vec.transform(target["text"]))
        female = proba[:, classes.index("female")]
        male = proba[:, classes.index("male")]
        pred = ["female" if f >= m else "male" for f, m in zip(female, male)]
        out.append((female, male, pred))
    return (vec, model), out


def metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--test-manifest", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--langs", default="en,es,ar")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)

    train = pd.read_csv(args.train_manifest)
    train["text"] = [read_tweets(p) for p in train["text_path"]]
    train = train[train["text"].str.len() > 0].reset_index(drop=True)
    train.to_csv(out / "text_train.csv", index=False)
    if args.test_manifest:
        test = pd.read_csv(args.test_manifest)
        test["text"] = [read_tweets(p) for p in test["text_path"]]
        test = test[test["text"].str.len() > 0].reset_index(drop=True)
        test.to_csv(out / "text_test.csv", index=False)
    else:
        test = pd.DataFrame(columns=train.columns)

    fit, calib = stratified_split(train, args.calib_frac, args.seed)
    all_metrics = {}
    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        f = fit[fit["lang"] == lang].reset_index(drop=True)
        c = calib[calib["lang"] == lang].reset_index(drop=True)
        t = test[test["lang"] == lang].reset_index(drop=True)
        if len(f) == 0:
            continue
        (vec, model), (cal_out,) = fit_predict(f, [c], args.seed)
        targets = [t] if len(t) else []
        (vec_full, model_full), outs = fit_predict(
            train[train["lang"] == lang].reset_index(drop=True), targets, args.seed)
        joblib.dump({"vectorizer": vec_full, "model": model_full},
                    out / "models" / f"{lang}_text.joblib")
        pairs = [("calib", c, cal_out)]
        if len(t):
            pairs.append(("test", t, outs[0]))
        for name, frame, probs in pairs:
            female, male, pred = probs
            rec = frame[["author_id", "gender", "lang", "n_chars"]].copy()
            rec["split"] = name
            rec["text_prob_female"] = female
            rec["text_prob_male"] = male
            rec["text_pred"] = pred
            rec.to_csv(out / f"text_predictions_{name}_{lang}.csv", index=False)
            all_metrics.setdefault(lang, {})[name] = metrics(frame["gender"], pred)
        print(lang, all_metrics[lang], flush=True)
    (out / "text_metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved", out / "text_metrics.json")


if __name__ == "__main__":
    main()
