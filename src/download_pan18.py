# -*- coding: utf-8 -*-
"""Resumable chunked downloader for large Zenodo files.

Zenodo throttles a single connection; this downloads fixed-size ranges in
parallel, verifies HTTP 206, and preallocates the output file.  Re-running with
the same output skips an already-complete file (size check), but does not
resume partial chunks automatically.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

CHUNK = 1024 * 1024


def content_length(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def fetch_chunk(url: str, path: Path, idx: int, size: int, chunk: int) -> int:
    start = idx * chunk
    end = min(size - 1, start + chunk - 1)
    expected = end - start + 1
    last = None
    for attempt in range(20):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                             timeout=120)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "10"))
                time.sleep(min(60, max(2, wait)))
                raise RuntimeError("HTTP 429")
            if r.status_code != 206:
                raise RuntimeError(f"HTTP {r.status_code}")
            data = r.content
            if len(data) != expected:
                raise RuntimeError(f"short chunk {len(data)} != {expected}")
            with open(path, "r+b") as f:
                f.seek(start)
                f.write(data)
            return len(data)
        except Exception as exc:  # retry transient resets
            last = exc
            time.sleep(min(15, 2 + attempt))
    raise RuntimeError(f"chunk {idx} failed: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--chunk-kb", type=int, default=1024)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--size", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    size = args.size or content_length(args.url)
    print(f"size={size} ({size/1024**3:.2f} GiB) -> {out}", flush=True)
    chunk = args.chunk_kb * 1024
    n = (size + chunk - 1) // chunk
    if out.exists() and out.stat().st_size == size and not args.resume:
        print("already complete (use --resume to repair a partial/preallocated file)", flush=True)
        return
    if not out.exists() or out.stat().st_size != size:
        with open(out, "wb") as f:
            f.truncate(size)
    missing = list(range(n))
    if args.resume:
        # A preallocated chunk that is all zero is treated as missing.
        missing = []
        with open(out, "rb") as f:
            for i in range(n):
                f.seek(i * chunk)
                data = f.read(min(chunk, size - i * chunk))
                if not data or data.count(0) == len(data):
                    missing.append(i)
        print(f"resume: {len(missing)}/{n} chunks missing", flush=True)
    done = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_chunk, args.url, out, i, size, chunk): i
                   for i in missing}
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 20 == 0 or done == n:
                elapsed = max(0.001, time.time() - started)
                print(f"{done}/{n} chunks, {done*chunk/1024**2:.0f} MiB, "
                      f"{done*chunk/1024**2/elapsed:.2f} MiB/s", flush=True)
    print("complete", out, out.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
