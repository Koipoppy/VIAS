# -*- coding: utf-8 -*-
"""Build the joint text+image dataset for PAN18 embedding-concat fusion.

Outputs (under --out):
  manifest_joint.csv     author_id, lang, gender, split (fit/calib/test), paths
  texts.csv              author_id, lang, split, text
  tokens_{maxlen}.npz    input_ids / attention_mask aligned to manifest_joint.csv
  image_features.npz     per-author mean image embedding + probability
  image_features.csv     per-author image statistics (n_images, prob_female)

The official PAN18 train split is divided into fit/calib only; the official
test split is never used for training, early stopping or calibration.
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261127


def read_tweets(path: str) -> str:
    try:
        root = ET.fromstring(Path(path).read_text(encoding="utf-8", errors="ignore"))
        docs = [d.text or "" for d in root.findall(".//document")]
        return " ".join(x.strip() for x in docs if x.strip())
    except Exception:
        return ""


def stratified_split(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    fit_idx, cal_idx = [], []
    for _, group in df.groupby(["lang", "gender"]):
        idx = np.array(group.index)
        rng.shuffle(idx)
        n_cal = max(1, int(round(len(idx) * frac)))
        cal_idx.extend(idx[:n_cal].tolist())
        fit_idx.extend(idx[n_cal:].tolist())
    return df.loc[fit_idx].copy(), df.loc[cal_idx].copy()


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_manifest(processed: Path, out: Path, calib_frac: float, seed: int) -> pd.DataFrame:
    train = pd.read_csv(processed / "train" / "manifest_train.csv")
    test = pd.read_csv(processed / "test" / "manifest_test.csv")
    fit, calib = stratified_split(train, calib_frac, seed)
    fit["split"], calib["split"], test["split"] = "fit", "calib", "test"
    joint = pd.concat([fit, calib, test], ignore_index=True)
    joint = joint[["author_id", "lang", "gender", "split", "text_path", "photo_dir", "n_photos"]]
    joint.to_csv(out / "manifest_joint.csv", index=False)
    print(joint.groupby(["split", "lang", "gender"]).size().to_string(), flush=True)
    return joint


def build_texts(joint: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for r in joint.itertuples(index=False):
        rows.append({
            "author_id": r.author_id, "lang": r.lang, "split": r.split,
            "gender": r.gender, "text": clean_text(read_tweets(r.text_path)),
        })
    texts = pd.DataFrame(rows)
    texts = texts[texts["text"].str.len() > 0].reset_index(drop=True)
    texts.to_csv(out / "texts.csv", index=False)
    print("texts", texts.shape, flush=True)
    return texts


def build_tokens(texts: pd.DataFrame, out: Path, model_name: str, max_len: int,
                 chunks: int = 1) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Keep row alignment with manifest_joint.csv: missing text becomes an empty string
    # so downstream code masks it instead of silently shifting indices.
    manifest = pd.read_csv(out / "manifest_joint.csv")
    by_key = {(r.author_id, r.lang): r.text for r in texts.itertuples(index=False)}
    all_texts = [by_key.get((a, l), "") for a, l in zip(manifest["author_id"], manifest["lang"])]
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    step = max_len - 2
    n = len(all_texts)
    input_ids = np.full((n, chunks, max_len), pad, dtype=np.int32)
    attention_mask = np.zeros((n, chunks, max_len), dtype=np.int8)
    for i, text in enumerate(all_texts):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        for k in range(chunks):
            piece = ids[k * step:(k + 1) * step]
            if not piece:
                continue
            seq = [bos] + piece + [eos]
            input_ids[i, k, :len(seq)] = seq
            attention_mask[i, k, :len(seq)] = 1
    np.savez_compressed(out / f"tokens_{max_len}_c{chunks}.npz", input_ids=input_ids,
                        attention_mask=attention_mask)
    covered = attention_mask.sum(axis=(1, 2))
    print("tokens", input_ids.shape, "empty_text", int(sum(len(t) == 0 for t in all_texts)),
          "tokens_covered_median", int(np.median(covered)), flush=True)


def _load_image(path: str):
    try:
        from PIL import Image

        img = Image.open(path)
        img.draft("RGB", (256, 256))
        return img.convert("RGB")
    except Exception:
        return None


def extract_image_features(joint: pd.DataFrame, out: Path, model_name: str,
                           device: str, batch_size: int, max_images: int,
                           workers: int) -> pd.DataFrame:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device).eval()
    id2label = {i: str(v).lower() for i, v in model.config.id2label.items()}
    female_idx = next((i for i, v in id2label.items() if v == "female"), None)
    male_idx = next((i for i, v in id2label.items() if v == "male"), None)
    if female_idx is None or male_idx is None:
        raise SystemExit(f"gender labels not found: {id2label}")

    records: list[tuple[int, str]] = []
    for idx, r in enumerate(joint.itertuples(index=False)):
        paths = sorted(Path(r.photo_dir).glob("*.jpeg"))[:max_images]
        records.extend((idx, str(p)) for p in paths)
    print("images to encode", len(records), flush=True)

    dim = int(model.config.hidden_size)
    emb_sum = np.zeros((len(joint), dim), dtype=np.float64)
    prob_sum = np.zeros(len(joint), dtype=np.float64)
    n_images = np.zeros(len(joint), dtype=np.int32)
    valid = np.zeros(len(joint), dtype=bool)

    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                imgs = list(ex.map(_load_image, [p for _, p in chunk]))
            keep = [(idx, img) for (idx, _), img in zip(chunk, imgs) if img is not None]
            if not keep:
                continue
            inputs = processor([img for _, img in keep], return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, 0, :].float().cpu().numpy()
            probs = torch.softmax(outputs.logits, dim=-1).float().cpu().numpy()
            female = probs[:, female_idx]
            male = probs[:, male_idx]
            denom = female + male
            female = np.divide(female, denom, out=np.full_like(female, 0.5), where=denom > 0)
            for (idx, _), h, p in zip(keep, hidden, female):
                emb_sum[idx] += h
                prob_sum[idx] += float(p)
                n_images[idx] += 1
                valid[idx] = True
            if (start // batch_size) % 100 == 0:
                print(f"{start + len(chunk)}/{len(records)}", flush=True)

    dim = emb_sum.shape[1]
    emb = np.zeros((len(joint), dim), dtype=np.float32)
    prob = np.full(len(joint), 0.5, dtype=np.float32)
    for i in range(len(joint)):
        if n_images[i] > 0:
            emb[i] = (emb_sum[i] / n_images[i]).astype(np.float32)
            prob[i] = float(prob_sum[i] / n_images[i])
    np.savez_compressed(out / "image_features.npz", emb=emb, prob_female=prob,
                        n_images=n_images, valid=valid)
    stats = joint[["author_id", "lang", "gender", "split"]].copy()
    stats["n_images"] = n_images
    stats["image_prob_female"] = prob
    stats["image_pred"] = np.where(prob >= 0.5, "female", "male")
    stats.to_csv(out / "image_features.csv", index=False)
    print("image features", emb.shape, "missing", int((n_images == 0).sum()), flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed/pan18")
    ap.add_argument("--out", default="data/processed/pan18_joint")
    ap.add_argument("--text-model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--image-model", default="rizvandwiki/gender-classification")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=1,
                    help="number of 512-token chunks per user (1 = first chunk only)")
    ap.add_argument("--calib-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-tokens", action="store_true")
    args = ap.parse_args()

    processed, out = Path(args.processed), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joint = build_manifest(processed, out, args.calib_frac, args.seed)
    texts = build_texts(joint, out)
    if not args.skip_tokens:
        build_tokens(texts, out, args.text_model, args.max_len, args.chunks)
    if not args.skip_images:
        extract_image_features(joint, out, args.image_model, args.device,
                               args.batch_size, args.max_images, args.workers)


if __name__ == "__main__":
    main()
