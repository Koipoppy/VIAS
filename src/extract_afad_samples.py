# -*- coding: utf-8 -*-
"""Extract a small stratified sample from the AFAD-Lite tar.xz archive.

AFAD-Lite directory layout: AFAD-Lite/<age>/<111|112>/<file>.jpg
111 = male, 112 = female (verified visually on 33/111 and 33/112).
"""
import argparse
import os
import tarfile
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ages", default="20,24,28,32,36")
    ap.add_argument("--per-cell", type=int, default=5)
    args = ap.parse_args()

    ages = [int(x) for x in args.ages.split(",")]
    targets = {(a, sub) for a in ages for sub in ("111", "112")}
    counts = Counter()
    os.makedirs(args.out, exist_ok=True)

    with tarfile.open(args.archive, "r|xz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            if len(parts) < 4 or parts[0] != "AFAD-Lite":
                continue
            age_s, sub = parts[1], parts[2]
            if not age_s.isdigit():
                continue
            key = (int(age_s), sub)
            if key not in targets or counts[key] >= args.per_cell:
                continue
            if not m.name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            gender = "male" if sub == "111" else "female"
            outdir = os.path.join(args.out, f"{age_s}_{gender}")
            os.makedirs(outdir, exist_ok=True)
            f = tf.extractfile(m)
            if f is None:
                continue
            outpath = os.path.join(outdir, os.path.basename(m.name))
            with open(outpath, "wb") as out:
                out.write(f.read())
            counts[key] += 1
            print(f"{key} {outpath}", flush=True)
            if all(counts[t] >= args.per_cell for t in targets):
                break

    missing = [t for t in targets if counts[t] < args.per_cell]
    print("counts:", dict(counts))
    print("missing:", missing)


if __name__ == "__main__":
    main()
