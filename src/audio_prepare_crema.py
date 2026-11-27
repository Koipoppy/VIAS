# -*- coding: utf-8 -*-
"""Download a compact CREMA-D subset and extract same-actor face frames.

The public CREMA-D mirror hosts the real WAV/FLV files (not LFS pointers).
We take actors aged 20-40 so their age buckets line up with the existing
SMP/AFAD pilot buckets, then pick up to four clips per actor from different
sentences. For each clip we keep the WAV, the FLV, and three face frames.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master"
RAW = ROOT / "data" / "raw" / "audio" / "crema"
META = ROOT / "data" / "raw" / "audio" / "crema_meta"
OUT = ROOT / "data" / "processed" / "audio"
BUCKETS = [(18, 24, "18-24"), (25, 29, "25-29"), (30, 34, "30-34"), (35, 40, "35-40")]
SKIP_ACTORS = {1064, 1076}  # known broken/missing audio in the official notes


def bucket(age: int) -> str | None:
    for lo, hi, name in BUCKETS:
        if lo <= age <= hi:
            return name
    return None


def fetch(url: str, path: Path, retries: int = 3) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vias-audio-pilot/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) < 100:
                raise ValueError(f"too small: {len(data)}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return True
        except Exception as exc:
            if attempt == retries - 1:
                print(f"FAIL {url}: {exc}", flush=True)
    return False


def get_metadata() -> pd.DataFrame:
    META.mkdir(parents=True, exist_ok=True)
    dem = META / "VideoDemographics.csv"
    sent = META / "SentenceFilenames.csv"
    fetch("https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/VideoDemographics.csv", dem)
    fetch("https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/SentenceFilenames.csv", sent)
    d = pd.read_csv(dem)
    d.columns = [c.strip().lower() for c in d.columns]
    d = d.rename(columns={"actorid": "actor_id", "age": "age", "sex": "sex"})
    d = d[d["age"].between(18, 40) & ~d["actor_id"].isin(SKIP_ACTORS)]
    return d


def select_clips(dem: pd.DataFrame, sent_path: Path, per_actor: int) -> pd.DataFrame:
    s = pd.read_csv(sent_path)
    s = s.rename(columns={"Filename": "clip"})
    p = s["clip"].str.split("_", expand=True)
    s["actor_id"] = p[0].astype(int)
    s["sentence"] = p[1]
    s["emotion"] = p[2]
    s["level"] = p[3]
    s = s[s["actor_id"].isin(set(dem["actor_id"]))]
    rows = []
    for actor, group in s.groupby("actor_id"):
        # Prefer XX-level neutral clips, then one clip per sentence.
        group = group.assign(_lvl=group["level"].map({"XX": 0, "LO": 1, "MD": 2, "HI": 3}).fillna(9),
                             _emo=group["emotion"].map({"NEU": 0, "ANG": 1, "HAP": 2, "SAD": 3, "FEA": 4, "DIS": 5}).fillna(9))
        group = group.sort_values(["_lvl", "_emo", "sentence"])
        chosen = []
        seen = set()
        for _, r in group.iterrows():
            if r["sentence"] in seen:
                continue
            chosen.append(r)
            seen.add(r["sentence"])
            if len(chosen) >= per_actor:
                break
        rows.extend(chosen)
    return pd.DataFrame(rows)


def extract_frames(flv: Path, out_dir: Path, clip: str, n_frames: int = 3) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(flv))
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    saved = 0
    if total > 0:
        positions = [int(total * f) for f in (0.3, 0.5, 0.7)][:n_frames]
    else:
        positions = list(range(n_frames))
    for i, pos in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # Keep the central portrait crop; CREMA-D is a close-up face video.
        h, w = frame.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = frame[y0:y0 + side, x0:x0 + side]
        path = out_dir / f"{clip}_f{i}.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if path.exists():
            saved += 1
    cap.release()
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-actor", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()
    dem = get_metadata()
    clips = select_clips(dem, META / "SentenceFilenames.csv", args.per_actor)
    clips = clips.merge(dem[["actor_id", "age", "sex"]], on="actor_id", how="left")
    clips["age_bucket"] = clips["age"].map(bucket)
    clips = clips[clips["age_bucket"].notna()].reset_index(drop=True)
    print("actors", dem.actor_id.nunique(), "clips", len(clips), flush=True)
    print(clips.groupby(["sex", "age_bucket"]).size().to_string(), flush=True)

    jobs = []
    for r in clips.itertuples(index=False):
        jobs.append(("wav", r.clip, RAW / "wav" / f"{r.clip}.wav", f"{BASE}/AudioWAV/{r.clip}.wav"))
        jobs.append(("flv", r.clip, RAW / "flv" / f"{r.clip}.flv", f"{BASE}/VideoFlash/{r.clip}.flv"))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch, url, path): (kind, clip) for kind, clip, path, url in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            kind, clip = futures[fut]
            fut.result()
            if i % 50 == 0:
                print(f"downloaded {i}/{len(jobs)}", flush=True)

    rows = []
    for r in clips.itertuples(index=False):
        wav = RAW / "wav" / f"{r.clip}.wav"
        flv = RAW / "flv" / f"{r.clip}.flv"
        if not wav.exists() or wav.stat().st_size < 100:
            continue
        n_frames = 0
        if not args.skip_video and flv.exists():
            n_frames = extract_frames(flv, OUT / "faces", r.clip)
        rows.append({
            "case_id": f"A-{r.clip}",
            "actor_id": r.actor_id,
            "clip": r.clip,
            "gender": "female" if str(r.sex).lower().startswith("f") else "male",
            "age": int(r.age),
            "age_bucket": r.age_bucket,
            "sentence_code": r.sentence,
            "emotion": r.emotion,
            "level": r.level,
            "wav_path": wav.relative_to(ROOT).as_posix(),
            "flv_path": flv.relative_to(ROOT).as_posix() if flv.exists() else "",
            "n_frames": n_frames,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "crema_manifest.csv", index=False, encoding="utf-8-sig")
    print("manifest", out.shape, "with_frames", int((out.n_frames > 0).sum()), flush=True)
    print(out.groupby(["gender", "age_bucket"]).size().to_string(), flush=True)


if __name__ == "__main__":
    main()
