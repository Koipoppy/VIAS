# -*- coding: utf-8 -*-
"""Face gender/age evidence from the CREMA-D same-actor frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "audio"
GENDER_MODEL = "rizvandwiki/gender-classification"
AGE_MODEL = "nateraw/vit-age-classifier"
TARGET_BUCKETS = ["18-24", "25-29", "30-34", "35-40"]
SOURCE_ORDER = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
SOURCE_MIDPOINT = {"0-2": 1, "3-9": 6, "10-19": 14.5, "20-29": 24.5, "30-39": 34.5,
                   "40-49": 44.5, "50-59": 54.5, "60-69": 64.5, "70+": 75.0}
AGE_MAP = {
    "0-2": {"18-24": 1.0}, "3-9": {"18-24": 1.0}, "10-19": {"18-24": 1.0},
    "20-29": {"18-24": 0.5, "25-29": 0.5},
    "30-39": {"30-34": 0.5, "35-40": 0.5},
    "40-49": {"35-40": 1.0}, "50-59": {"35-40": 1.0},
    "60-69": {"35-40": 1.0}, "70+": {"35-40": 1.0},
}


def map_age_probs(probs: dict) -> tuple[dict, float]:
    out = {b: 0.0 for b in TARGET_BUCKETS}
    expected = 0.0
    for src, p in probs.items():
        for tgt, w in AGE_MAP.get(src, {}).items():
            out[tgt] += p * w
        expected += p * SOURCE_MIDPOINT.get(src, 40.0)
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}, expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(OUT / "crema_manifest.csv"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    manifest = pd.read_csv(args.manifest)
    out = Path(args.out)
    device = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    g_processor = AutoImageProcessor.from_pretrained(GENDER_MODEL, use_fast=True)
    g_model = AutoModelForImageClassification.from_pretrained(GENDER_MODEL).to(device).eval()
    g_labels = {i: str(v).lower() for i, v in g_model.config.id2label.items()}
    g_female = next(i for i, v in g_labels.items() if v == "female")
    g_male = next(i for i, v in g_labels.items() if v == "male")
    a_processor = AutoImageProcessor.from_pretrained(AGE_MODEL, use_fast=True)
    a_model = AutoModelForImageClassification.from_pretrained(AGE_MODEL).to(device).eval()
    a_labels = {i: str(v) for i, v in a_model.config.id2label.items()}

    rows, embs = [], []
    for n, r in enumerate(manifest.itertuples(index=False), 1):
        frames = sorted((out / "faces").glob(f"{r.clip}_f*.jpg"))
        if not frames:
            rows.append({"case_id": r.case_id, "clip": r.clip, "face_p_female": 0.5,
                         "face_age": np.nan, "face_n_frames": 0,
                         **{f"face_age_{b}": 1 / len(TARGET_BUCKETS) for b in TARGET_BUCKETS}})
            embs.append(np.zeros(768, dtype=np.float32))
            continue
        imgs = [Image.open(p).convert("RGB") for p in frames]
        with torch.no_grad():
            g_in = g_processor(imgs, return_tensors="pt").to(device)
            g_out = g_model(**g_in, output_hidden_states=True)
            g_p = torch.softmax(g_out.logits, dim=-1).cpu().numpy()
            g_emb = g_out.hidden_states[-1][:, 0].cpu().numpy()
            a_in = a_processor(imgs, return_tensors="pt").to(device)
            a_p = torch.softmax(a_model(**a_in).logits, dim=-1).cpu().numpy()
        pf = float(np.mean(g_p[:, g_female] / np.maximum(g_p[:, g_female] + g_p[:, g_male], 1e-9)))
        age_probs = {a_labels[i]: float(a_p[:, i].mean()) for i in range(a_p.shape[1])}
        target_age, expected = map_age_probs(age_probs)
        rows.append({
            "case_id": r.case_id, "clip": r.clip,
            "face_p_female": pf, "face_age": expected, "face_n_frames": len(frames),
            **{f"face_age_{b}": target_age[b] for b in TARGET_BUCKETS},
        })
        embs.append(g_emb.mean(axis=0).astype(np.float32))
        if n % 25 == 0:
            print(f"{n}/{len(manifest)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "face_features.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(out / "face_embeddings.npz", emb=np.stack(embs),
                        case_id=manifest["case_id"].to_numpy())
    print(json.dumps({"n": len(df), "face_n_frames_mean": float(df.face_n_frames.mean()),
                      "face_p_female_mean": float(df.face_p_female.mean())}, indent=2))


if __name__ == "__main__":
    main()
