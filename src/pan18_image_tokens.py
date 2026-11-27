# -*- coding: utf-8 -*-
"""Extract per-image ViT CLS embeddings for cross-attention fusion.

Reads the existing joint manifest and writes `image_tokens.npz` next to it:

    emb_images  [N, max_images, hidden]  float16
    image_mask  [N, max_images]          bool
    prob_female [N]                      float32
    n_images    [N]                      int32
    valid       [N]                      bool

The per-author mean in `image_features.npz` is left untouched so the concat
baseline stays exactly reproducible.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def _load_image(path: str):
    try:
        from PIL import Image

        img = Image.open(path)
        img.draft("RGB", (256, 256))
        return img.convert("RGB")
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/pan18_joint")
    ap.add_argument("--out", default=None, help="default: <data>/image_tokens.npz")
    ap.add_argument("--image-model", default="rizvandwiki/gender-classification")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-authors", type=int, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    data_dir = Path(args.data)
    out_path = Path(args.out) if args.out else data_dir / "image_tokens.npz"
    manifest = pd.read_csv(data_dir / "manifest_joint.csv")
    if args.limit_authors:
        manifest = manifest.iloc[: args.limit_authors].copy()

    processor = AutoImageProcessor.from_pretrained(args.image_model, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(args.image_model).to(args.device).eval()
    id2label = {i: str(v).lower() for i, v in model.config.id2label.items()}
    female_idx = next((i for i, v in id2label.items() if v == "female"), None)
    male_idx = next((i for i, v in id2label.items() if v == "male"), None)
    if female_idx is None or male_idx is None:
        raise SystemExit(f"gender labels not found: {id2label}")

    records: list[tuple[int, int, str]] = []
    for idx, r in enumerate(manifest.itertuples(index=False)):
        paths = sorted(Path(r.photo_dir).glob("*.jpeg"))[: args.max_images]
        records.extend((idx, slot, str(p)) for slot, p in enumerate(paths))
    print(f"authors={len(manifest)} images={len(records)}", flush=True)

    dim = int(model.config.hidden_size)
    n = len(manifest)
    emb = np.zeros((n, args.max_images, dim), dtype=np.float16)
    mask = np.zeros((n, args.max_images), dtype=bool)
    prob_sum = np.zeros(n, dtype=np.float64)
    n_images = np.zeros(n, dtype=np.int32)

    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            chunk = records[start:start + args.batch_size]
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                imgs = list(ex.map(_load_image, [p for _, _, p in chunk]))
            keep = [(idx, slot, img) for (idx, slot, _), img in zip(chunk, imgs)
                    if img is not None]
            if not keep:
                continue
            inputs = processor([img for _, _, img in keep], return_tensors="pt").to(args.device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, 0, :].float().cpu().numpy()
            probs = torch.softmax(outputs.logits, dim=-1).float().cpu().numpy()
            female = probs[:, female_idx]
            male = probs[:, male_idx]
            denom = female + male
            female = np.divide(female, denom, out=np.full_like(female, 0.5), where=denom > 0)
            for (idx, slot, _), h, p in zip(keep, hidden, female):
                emb[idx, slot] = h.astype(np.float16)
                mask[idx, slot] = True
                prob_sum[idx] += float(p)
                n_images[idx] += 1
            if (start // args.batch_size) % 100 == 0:
                print(f"{start + len(chunk)}/{len(records)}", flush=True)

    prob = np.full(n, 0.5, dtype=np.float32)
    valid = n_images > 0
    prob[valid] = (prob_sum[valid] / n_images[valid]).astype(np.float32)
    np.savez_compressed(out_path, emb_images=emb, image_mask=mask, prob_female=prob,
                        n_images=n_images, valid=valid)
    print(f"wrote {out_path} shape={emb.shape} missing={int((~valid).sum())}", flush=True)


if __name__ == "__main__":
    main()
