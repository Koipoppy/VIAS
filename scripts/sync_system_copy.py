# -*- coding: utf-8 -*-
"""Keep the runtime copy and the git-tracked copy of vias_system in sync.

There are two checkouts of the service:
  vias_pilot/vias_system   git-tracked source of truth
  ../vias_system           workspace runtime copy (models/, var/ live here)

Run from the repo root:
  python scripts/sync_system_copy.py --check   # exit 1 if code differs
  python scripts/sync_system_copy.py           # repo -> runtime copy

models/, var/, __pycache__ and *.pyc are runtime-only and never copied.
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "vias_system"
DST = REPO_ROOT.parent / "vias_system"

EXCLUDE_DIRS = {"models", "var", "__pycache__", ".pytest_cache", ".ruff_cache", ".git"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_FILES = {".DS_Store"}


def _walk(root: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if path.suffix in EXCLUDE_SUFFIXES or path.name in EXCLUDE_FILES:
            continue
        yield rel, path


def compare() -> list[str]:
    problems = []
    src_files = {rel for rel, p in _walk(SRC) if p.is_file()}
    dst_files = {rel for rel, p in _walk(DST) if p.is_file()}
    for rel in sorted(src_files - dst_files):
        problems.append(f"missing in runtime copy: {rel}")
    for rel in sorted(dst_files - src_files):
        problems.append(f"extra in runtime copy: {rel}")
    for rel in sorted(src_files & dst_files):
        if not filecmp.cmp(SRC / rel, DST / rel, shallow=False):
            problems.append(f"differs: {rel}")
    return problems


def sync() -> tuple[int, int]:
    copied = removed = 0
    for rel, path in _walk(SRC):
        target = DST / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or not filecmp.cmp(path, target, shallow=False):
            shutil.copy2(path, target)
            copied += 1
    src_files = {rel for rel, p in _walk(SRC) if p.is_file()}
    for rel, path in _walk(DST):
        if path.is_file() and rel not in src_files:
            path.unlink()
            removed += 1
    return copied, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not SRC.exists() or not DST.exists():
        raise SystemExit(f"missing checkout: {SRC} or {DST}")
    if args.check:
        problems = compare()
        if problems:
            print("\n".join(problems))
            raise SystemExit(1)
        print("vias_system copies are in sync")
        return
    copied, removed = sync()
    print(f"synced {SRC} -> {DST}: {copied} copied, {removed} removed")


if __name__ == "__main__":
    main()
