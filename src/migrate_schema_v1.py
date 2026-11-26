# -*- coding: utf-8 -*-
"""Migrate v0.1 evidence JSONL files to Schema v1.0.

- set schema_version to 1.0
- drop the legacy persisted `effective_weight` (now derived at runtime)
- validate every record with the v1.0 Pydantic model
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema import EvidenceUnit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def paths():
    fixed = [
        ROOT / "outputs/evidence_units_clean.jsonl",
        ROOT / "outputs/evidence_units_injected.jsonl",
    ]
    cases = sorted((ROOT / "cases_synthetic").glob("SYN-*/evidence_units.jsonl"))
    return [p for p in fixed + cases if p.exists()]


def migrate(path: Path):
    source = path.read_text(encoding="utf-8").splitlines()
    out = []
    bad = []
    for lineno, line in enumerate(source, 1):
        if not line.strip():
            continue
        record = json.loads(line)
        record.pop("effective_weight", None)
        record["schema_version"] = "1.0"
        try:
            EvidenceUnit.model_validate(record)
        except Exception as exc:  # noqa: BLE001
            bad.append((lineno, str(exc).splitlines()[0]))
            continue
        out.append(json.dumps(record, ensure_ascii=False))
    if bad:
        raise SystemExit(f"{path}: {len(bad)} invalid records; first={bad[0]}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(path)
    return len(out)


def main():
    total = 0
    for path in paths():
        n = migrate(path)
        total += n
        print(f"{path.relative_to(ROOT)}: {n}")
    print("total:", total)


if __name__ == "__main__":
    main()
