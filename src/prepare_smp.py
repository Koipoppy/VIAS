# -*- coding: utf-8 -*-
"""Prepare SMP 2016 user profiles and tweets.

Outputs:
  data/processed/smp_all.csv    all labelled users with text
  data/processed/smp_train.csv  training split (pilot users excluded)
  data/processed/smp_pilot.csv  50 pilot users (5 male + 5 female for ages 22/24/28/32/36)
"""
import argparse
import os
import re
from collections import defaultdict

import pandas as pd


CJK = r"\u4e00-\u9fff"


def detokenize(s: str) -> str:
    """Remove spaces between CJK characters/punctuation, keep Latin word spaces."""
    s = re.sub(rf"(?<=[{CJK}])\s+(?=[{CJK}])", "", s)
    s = re.sub(rf"(?<=[{CJK}])\s+(?=[，。！？、；：])", "", s)
    s = re.sub(rf"(?<=[，。！？、；：])\s+(?=[{CJK}])", "", s)
    return s.strip()


def load_labels(path: str) -> dict:
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("||")
            if len(p) < 4:
                continue
            uid, gender, birth, loc = p[0], p[1], p[2], p[3]
            try:
                birth = int(birth)
            except ValueError:
                continue
            labels[uid] = {
                "uid": uid,
                "gender": "male" if gender == "m" else "female",
                "birth_year": birth,
                "age": 2016 - birth,
                "location": loc.strip(),
            }
    return labels


def load_status(paths: list) -> dict:
    texts = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                p = line.rstrip("\n").split(",", 5)
                if len(p) != 6:
                    continue
                uid, text = p[0], p[5].strip()
                if text:
                    texts[uid].append(text)
    return texts


def age_bucket(age: int) -> str:
    if age <= 24:
        return "18-24"
    if age <= 29:
        return "25-29"
    if age <= 34:
        return "30-34"
    return "35-40"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smp-dir", default="data/raw/smp")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--ages", default="22,24,28,32,36")
    ap.add_argument("--per-cell", type=int, default=5)
    ap.add_argument("--max-chars", type=int, default=5000)
    args = ap.parse_args()

    labels = load_labels(os.path.join(args.smp_dir, "train_train_labels.txt"))
    texts = load_status(
        [
            os.path.join(args.smp_dir, "train_train_status.txt"),
            os.path.join(args.smp_dir, "valid_valid_status.txt"),
        ]
    )

    rows = []
    for uid, d in labels.items():
        tweets = texts.get(uid, [])
        if not tweets:
            continue
        raw = "\n".join(tweets)
        clean = detokenize(raw)
        rows.append(
            {
                **d,
                "age_bucket": age_bucket(d["age"]),
                "n_tweets": len(tweets),
                "n_chars": len(clean),
                "text_raw": raw,
                "text_clean": clean,
            }
        )
    all_df = pd.DataFrame(rows)
    # keep only plausible adult ages for this pilot
    all_df = all_df[(all_df["age"] >= 18) & (all_df["age"] <= 40)].copy()
    all_df = all_df.sort_values(["age", "gender", "n_tweets"], ascending=[True, True, False])

    pilot_parts = []
    for age in [int(x) for x in args.ages.split(",")]:
        for gender in ["male", "female"]:
            cand = all_df[(all_df["age"] == age) & (all_df["gender"] == gender)]
            cand = cand.sort_values(["n_tweets", "n_chars"], ascending=False)
            pilot_parts.append(cand.head(args.per_cell))
    pilot = pd.concat(pilot_parts).drop_duplicates("uid").reset_index(drop=True)

    train = all_df[~all_df["uid"].isin(pilot["uid"])].copy().reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)
    all_df.to_csv(os.path.join(args.out_dir, "smp_all.csv"), index=False)
    train.to_csv(os.path.join(args.out_dir, "smp_train.csv"), index=False)
    pilot.to_csv(os.path.join(args.out_dir, "smp_pilot.csv"), index=False)

    print("all users:", len(all_df))
    print("pilot users:", len(pilot))
    print(pilot.groupby(["age", "gender"]).size().to_string())
    print("train users:", len(train))
    print("pilot age buckets:", pilot["age_bucket"].value_counts().to_dict())
    print("pilot text chars: min/median/max",
          int(pilot["n_chars"].min()), int(pilot["n_chars"].median()), int(pilot["n_chars"].max()))


if __name__ == "__main__":
    main()
