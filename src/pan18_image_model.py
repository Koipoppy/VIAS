# -*- coding: utf-8 -*-
"""Fast batch image gender inference for PAN18 users.

Uses the model processor directly (fast image processor) and a single forward
pass per image batch instead of one transformers.pipeline call per user.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL = "rizvandwiki/gender-classification"


def _load_one(p):
    try:
        img = Image.open(p)
        img.draft("RGB", (256, 256))  # decode JPEG at reduced size; model resizes to 224
        return img.convert("RGB")
    except Exception:
        return None


def load_images(paths):
    with ThreadPoolExecutor(max_workers=16) as ex:
        return list(ex.map(_load_one, paths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--authors-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=["calib", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--langs", default="en,es,ar")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-images", type=int, default=1,
                    help="images per user; 1 is the fast full-user profile setting")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    authors = pd.read_csv(args.authors_csv)
    keys = set(zip(authors["author_id"], authors["lang"]))
    manifest = manifest[[(a, l) in keys for a, l in zip(manifest["author_id"], manifest["lang"])]]
    records = []
    for row in manifest.itertuples(index=False):
        for p in sorted(Path(row.photo_dir).glob("*.jpeg"))[:args.max_images]:
            records.append((row.author_id, row.lang, row.gender, str(p)))

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(MODEL).to(device).eval()
    id2label = {i: str(v).lower() for i, v in model.config.id2label.items()}
    sums = defaultdict(lambda: [0.0, 0.0, 0])
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            chunk = records[start:start + args.batch_size]
            imgs = load_images([x[3] for x in chunk])
            valid = [(meta, img) for meta, img in zip(chunk, imgs) if img is not None]
            if not valid:
                continue
            inputs = processor([img for _, img in valid], return_tensors="pt").to(device)
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
            for (meta, _), row in zip(valid, probs):
                author, lang, gender, _ = meta
                d = {id2label[i]: float(row[i]) for i in range(len(row))}
                f = d.get("female", 0.0)
                m = d.get("male", 0.0)
                s = f + m
                if s <= 0:
                    continue
                acc = sums[(author, lang, gender)]
                acc[0] += f / s
                acc[1] += m / s
                acc[2] += 1
            if True:
                print(f"{min(start+args.batch_size, len(records))}/{len(records)} images", flush=True)

    rows = []
    for (author, lang, gender), (f, m, n) in sums.items():
        if n == 0:
            continue
        rows.append({
            "author_id": author, "lang": lang, "gender": gender, "n_images": n,
            "image_prob_female": f / n, "image_prob_male": m / n,
            "image_pred": "female" if f >= m else "male",
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    out = Path(args.out)
    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        d = df[df["lang"] == lang]
        if len(d):
            d.to_csv(out.parent / f"image_predictions_{args.split}_{lang}.csv", index=False)
    metrics = {}
    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        d = df[df["lang"] == lang]
        if len(d):
            metrics[lang] = {
                "n": int(len(d)),
                "accuracy": float(accuracy_score(d["gender"], d["image_pred"])),
                "macro_f1": float(f1_score(d["gender"], d["image_pred"], average="macro")),
                "mean_images": float(d["n_images"].mean()),
            }
    out.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
