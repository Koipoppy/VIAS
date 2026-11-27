# -*- coding: utf-8 -*-
"""Run the audeering wav2vec2 age/gender model on the CREMA-D subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import Wav2Vec2FeatureExtractor

from audio_age_gender_model import AgeGenderModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "audeering/wav2vec2-large-robust-6-ft-age-gender"
OUT = ROOT / "data" / "processed" / "audio"


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        g = np.gcd(int(sr), int(target_sr))
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(OUT / "crema_manifest.csv"))
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    manifest = pd.read_csv(args.manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    model = AgeGenderModel.from_pretrained(args.model_dir).to(device).eval()

    rows, embs = [], []
    with torch.no_grad():
        for i, r in enumerate(manifest.itertuples(index=False), 1):
            wav = load_audio(ROOT / r.wav_path)
            inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
            values = inputs["input_values"].to(device)
            emb, age, gender = model(values)
            age = float(age.squeeze().cpu())
            g = gender.squeeze().cpu().numpy()
            p_f = float(g[0] / max(g[0] + g[1], 1e-9))
            rows.append({
                "case_id": r.case_id,
                "clip": r.clip,
                "audio_p_female": p_f,
                "audio_p_male": 1.0 - p_f,
                "audio_p_child": float(g[2]),
                "audio_age_0_1": age,
                "audio_age": max(0.0, min(100.0, age * 100.0)),
                "audio_emb_dim": int(emb.shape[1]),
            })
            embs.append(emb.squeeze(0).cpu().numpy().astype(np.float32))
            if i % 25 == 0:
                print(f"{i}/{len(manifest)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "audio_features.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(out / "audio_embeddings.npz", emb=np.stack(embs),
                        case_id=manifest["case_id"].to_numpy())
    summary = {
        "n": int(len(df)),
        "audio_age_min": float(df.audio_age.min()),
        "audio_age_max": float(df.audio_age.max()),
        "audio_p_female_mean": float(df.audio_p_female.mean()),
        "embedding_dim": int(df.audio_emb_dim.iloc[0]),
    }
    (out / "audio_features_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
