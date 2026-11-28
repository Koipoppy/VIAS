# -*- coding: utf-8 -*-
"""Build 128-d TF-IDF SVD text embeddings for the SMP pilot text pool.

The SVD is fit on the SMP training split (not on the pilot or audio cases) and
then applied to the 50 pilot texts used for synthetic pairing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "audio"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectorizer", default=str(ROOT.parent / "vias_system" / "models" / "text_vectorizer.joblib"))
    ap.add_argument("--train", default=str(ROOT / "data" / "processed" / "smp_train.csv"))
    ap.add_argument("--pilot", default=str(ROOT / "data" / "processed" / "smp_pilot.csv"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--components", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20261128)
    args = ap.parse_args()
    vec = joblib.load(args.vectorizer)
    train = pd.read_csv(args.train)
    pilot = pd.read_csv(args.pilot)
    xt = vec.transform(train["text_clean"].fillna(""))
    xp = vec.transform(pilot["text_clean"].fillna(""))
    svd = TruncatedSVD(n_components=args.components, random_state=args.seed).fit(xt)
    emb = svd.transform(xp).astype(np.float32)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "text_embeddings.npz", uid=pilot["uid"].to_numpy(), emb=emb)
    print("text embeddings", emb.shape, "explained", float(svd.explained_variance_ratio_.sum()))


if __name__ == "__main__":
    main()
