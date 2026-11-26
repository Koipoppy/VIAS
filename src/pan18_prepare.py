# -*- coding: utf-8 -*-
"""Prepare PAN 2018 Author Profiling data for the multimodal evaluation.

The official archives contain, per language:
  <split>/<lang>/text/<author>.xml    100 tweets
  <split>/<lang>/photo/<author>/*.jpeg 10 images
  labels: training <root>/<lang>/<lang>.txt; test <root>/<lang>.txt

This script extracts the archive and writes a per-author manifest.
"""
from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path


def read_labels(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":::" not in line:
            continue
        author, gender = line.rsplit(":::", 1)
        gender = gender.strip().lower()
        if gender not in {"male", "female"}:
            # PAN files occasionally have truncated trailing lines; skip them.
            continue
        out[author.strip()] = gender
    return out


def find_label_file(root: Path, lang: str) -> Path:
    candidates = [root / lang / f"{lang}.txt", root / f"{lang}.txt"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"label file for {lang} under {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--langs", default="en,es,ar")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zpath = Path(args.zip)
    if not args.no_extract:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
        print("extracted", zpath, "->", out, flush=True)
    # Locate the single top-level directory.
    roots = [p for p in out.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise SystemExit(f"expected one top-level directory, got {roots}")
    root = roots[0]
    rows = []
    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        labels = read_labels(find_label_file(root, lang))
        for author, gender in labels.items():
            text = root / lang / "text" / f"{author}.xml"
            photos = root / lang / "photo" / author
            n_photos = len(list(photos.glob("*.jpeg"))) if photos.exists() else 0
            rows.append({
                "split": args.split, "lang": lang, "author_id": author,
                "gender": gender, "text_path": text.as_posix(),
                "photo_dir": photos.as_posix(), "n_photos": n_photos,
                "n_chars": text.stat().st_size if text.exists() else 0,
            })
    manifest = out / f"manifest_{args.split}.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("manifest", manifest, "rows", len(rows), flush=True)
    for lang in sorted({r["lang"] for r in rows}):
        sub = [r for r in rows if r["lang"] == lang]
        print(lang, len(sub), "male", sum(r["gender"] == "male" for r in sub),
              "female", sum(r["gender"] == "female" for r in sub), flush=True)


if __name__ == "__main__":
    main()
