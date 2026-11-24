# -*- coding: utf-8 -*-
"""Pair each SMP pilot user with an AFAD face of the same age and gender."""
import argparse
import os
from collections import defaultdict

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="data/processed/smp_pilot.csv")
    ap.add_argument("--afad-dir", default="data/raw/afad/sampled")
    ap.add_argument("--out", default="data/processed/pilot_cases.csv")
    args = ap.parse_args()

    pilot = pd.read_csv(args.pilot)
    images = defaultdict(list)
    for age in sorted(pilot["age"].unique()):
        for gender in ["male", "female"]:
            d = os.path.join(args.afad_dir, f"{age}_{gender}")
            if not os.path.isdir(d):
                raise FileNotFoundError(d)
            files = sorted(
                os.path.join(d, f)
                for f in os.listdir(d)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            images[(age, gender)] = files

    rows = []
    case_no = 0
    for (age, gender), group in pilot.groupby(["age", "gender"], sort=True):
        group = group.sort_values("uid").reset_index(drop=True)
        imgs = images[(age, gender)]
        if len(group) > len(imgs):
            raise RuntimeError(f"Not enough AFAD images for {age}/{gender}: {len(group)} > {len(imgs)}")
        for i, row in group.iterrows():
            case_no += 1
            rows.append(
                {
                    "case_id": f"C{case_no:03d}",
                    "uid": row["uid"],
                    "gender": row["gender"],
                    "age": int(row["age"]),
                    "age_bucket": row["age_bucket"],
                    "location": row["location"],
                    "n_tweets": int(row["n_tweets"]),
                    "n_chars": int(row["n_chars"]),
                    "text_clean": row["text_clean"],
                    "image_path": imgs[i],
                    "afad_age": int(age),
                    "afad_gender": gender,
                }
            )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(["age", "gender"]).size().to_string())
    print("cases:", len(df))
    print("all images exist:", all(os.path.exists(p) for p in df["image_path"]))
    print(df[["case_id", "uid", "gender", "age", "age_bucket", "image_path"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
