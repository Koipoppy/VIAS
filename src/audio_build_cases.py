# -*- coding: utf-8 -*-
"""Assemble the audio+face+text multimodal test cases.

Audio and face come from the same CREMA-D actor (20-40). Text is a synthetic
pairing: an SMP 2016 pilot user with the same gender and age bucket, whose text
model was trained without the pilot users. The pairing is explicitly marked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "audio"
SYSTEM_MODELS = ROOT.parent / "vias_system" / "models"
AGE_CENTERS = {"18-24": 21.0, "25-29": 27.0, "30-34": 32.0, "35-40": 37.5}
BUCKETS = list(AGE_CENTERS)


def text_features(pilot: pd.DataFrame) -> pd.DataFrame:
    vec = joblib.load(SYSTEM_MODELS / "text_vectorizer.joblib")
    g = joblib.load(SYSTEM_MODELS / "text_gender.joblib")
    a = joblib.load(SYSTEM_MODELS / "text_age.joblib")
    x = vec.transform(pilot["text_clean"].fillna(""))
    gp = g.predict_proba(x)
    ap = a.predict_proba(x)
    gcols = list(g.classes_)
    acols = list(a.classes_)
    out = pilot[["uid", "gender", "age_bucket", "text_clean"]].copy()
    out["text_p_female"] = gp[:, gcols.index("female")] if "female" in gcols else 0.5
    for b in BUCKETS:
        out[f"text_age_{b}"] = ap[:, acols.index(b)] if b in acols else 0.0
    out["text_age_expected"] = sum(out[f"text_age_{b}"] * AGE_CENTERS[b] for b in BUCKETS)
    out["text_age_pred_bucket"] = out[[f"text_age_{b}" for b in BUCKETS]].idxmax(axis=1).str.replace("text_age_", "")
    out = out.rename(columns={"text_clean": "text"})
    return out


def assign_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    """Speaker-disjoint split, stratified by gender x age bucket."""
    actors = manifest[["actor_id", "gender", "age_bucket"]].drop_duplicates()
    split = {}
    for _, group in actors.groupby(["gender", "age_bucket"]):
        ids = sorted(group["actor_id"].tolist())
        n = len(ids)
        n_cal = max(1, int(round(n * 0.25)))
        n_test = max(1, int(round(n * 0.25)))
        for i, actor in enumerate(ids):
            split[actor] = "test" if i < n_test else "calib" if i < n_test + n_cal else "fit"
    return manifest.assign(split=manifest["actor_id"].map(split))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    manifest = pd.read_csv(out / "crema_manifest.csv")
    audio = pd.read_csv(out / "audio_features.csv")
    face = pd.read_csv(out / "face_features.csv")
    cases = manifest.merge(audio.drop(columns=["clip"]), on="case_id", how="left")
    cases = cases.merge(face.drop(columns=["clip"]), on="case_id", how="left")
    pilot = pd.read_csv(ROOT / "data" / "processed" / "smp_pilot.csv")
    text = text_features(pilot)

    # Round-robin text candidates per (gender, age_bucket), then same gender fallback.
    pools = {k: g.sample(frac=1, random_state=20261128).reset_index(drop=True)
             for k, g in text.groupby(["gender", "age_bucket"])}
    fallback = {k: g.sample(frac=1, random_state=20261129).reset_index(drop=True)
                for k, g in text.groupby("gender")}
    counters = {}
    rows = []
    for r in cases.to_dict("records"):
        key = (r["gender"], r["age_bucket"])
        pool = pools.get(key)
        if pool is None or len(pool) == 0:
            pool = fallback[r["gender"]]
        idx = counters.get(key, 0) % len(pool)
        counters[key] = counters.get(key, 0) + 1
        t = pool.iloc[idx]
        rows.append({
            "case_id": r["case_id"], "actor_id": r["actor_id"], "clip": r["clip"],
            "gender": r["gender"], "age": r["age"], "age_bucket": r["age_bucket"],
            "sentence_code": r["sentence_code"], "emotion": r["emotion"],
            "audio_wav": r["wav_path"], "face_frames": int(r["n_frames"]),
            "audio_p_female": float(r["audio_p_female"]),
            "audio_age": float(r["audio_age"]),
            "face_p_female": float(r["face_p_female"]),
            "face_age": float(r["face_age"]),
            **{f"face_age_{b}": float(r[f"face_age_{b}"]) for b in BUCKETS},
            "text_uid": t.uid, "text_lang": "zh", "text": str(t.text)[:500],
            "text_p_female": float(t.text_p_female),
            **{f"text_age_{b}": float(t[f"text_age_{b}"]) for b in BUCKETS},
            "text_age_expected": float(t.text_age_expected),
            "text_age_pred_bucket": t.text_age_pred_bucket,
            "text_gender_label": t.gender, "text_age_bucket_label": t.age_bucket,
            "synthetic_text_pairing": True,
            "same_actor_audio_face": True,
        })
    df = pd.DataFrame(rows)
    df = assign_splits(df)
    df.to_csv(out / "audio_multimodal_cases.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n": int(len(df)),
        "actors": int(df.actor_id.nunique()),
        "splits": df.groupby("split").size().to_dict(),
        "gender": df.groupby(["split", "gender"]).size().unstack(fill_value=0).to_dict(),
        "age_bucket": df.groupby(["split", "age_bucket"]).size().unstack(fill_value=0).to_dict(),
        "synthetic_text_pairing": True,
        "same_actor_audio_face": True,
        "audio_language": "en",
        "text_language": "zh",
    }
    (out / "audio_multimodal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
