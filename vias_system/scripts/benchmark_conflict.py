# -*- coding: utf-8 -*-
"""Benchmark the conflict/credibility defaults on the pilot artifacts.

This is a development script, not a runtime dependency.  It reads the pilot's
evidence JSONL and case labels, then compares the baseline TV configuration
with the improved defaults (weight contrast + attribute-specific tau).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vias.core.fusion import conflict_handle  # noqa: E402
from vias.core.schema import EvidenceUnit  # noqa: E402

AGE_MID = {"18-24": 21.0, "25-29": 27.0, "30-34": 32.0, "35-40": 37.5}


def load_units(path: Path) -> dict[str, list[EvidenceUnit]]:
    out: dict[str, list[EvidenceUnit]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            unit = EvidenceUnit.model_validate(json.loads(line))
            out[unit.case_id].append(unit)
    return out


def load_labels(path: Path) -> dict[str, dict]:
    return {
        item["case_id"]: item
        for item in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def predict(units, attr, tau, power):
    posterior, log, adjusted, max_conflict, pair, used = conflict_handle(
        units, attr, tau=tau, downweight=0.25, max_rounds=3, weight_power=power
    )
    return max(posterior, key=posterior.get), max_conflict


def evaluate(pilot_dir: Path, power: float, tau_gender: float, tau_age: float):
    clean = load_units(pilot_dir / "outputs/evidence_units_clean.jsonl")
    injected = load_units(pilot_dir / "outputs/evidence_units_injected.jsonl")
    clean_labels = load_labels(pilot_dir / "outputs/case_results_clean.jsonl")
    injected_labels = load_labels(pilot_dir / "outputs/case_results_injected.jsonl")
    injected_ids = {cid for cid, row in injected_labels.items() if row.get("injected")}

    cg = ca = fp = 0
    mae = 0.0
    for cid, units in clean.items():
        g, mg = predict(units, "gender", tau_gender, power)
        a, ma = predict(units, "age", tau_age, power)
        cg += g == clean_labels[cid]["true_gender"]
        ca += a == clean_labels[cid]["true_age_bucket"]
        mae += abs(AGE_MID[a] - clean_labels[cid]["true_age"])
        fp += (mg > tau_gender or ma > tau_age)

    ig = ia = det = 0
    for cid in injected_ids:
        g, mg = predict(injected[cid], "gender", tau_gender, power)
        a, ma = predict(injected[cid], "age", tau_age, power)
        ig += g == injected_labels[cid]["true_gender"]
        ia += a == injected_labels[cid]["true_age_bucket"]
        det += (mg > tau_gender or ma > tau_age)

    n_clean = len(clean)
    n_inj = len(injected_ids)
    return {
        "clean_gender_accuracy": round(cg / n_clean, 4),
        "clean_age_accuracy": round(ca / n_clean, 4),
        "clean_age_mae": round(mae / n_clean, 4),
        "clean_any_conflict_rate": round(fp / n_clean, 4),
        "injected_gender_accuracy": round(ig / n_inj, 4),
        "injected_age_accuracy": round(ia / n_inj, 4),
        "injected_detection_rate": round(det / n_inj, 4),
        "n_clean": n_clean,
        "n_injected": n_inj,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", default=str(ROOT.parent / "vias_pilot"))
    parser.add_argument("--out", default=str(ROOT / "docs/conflict_benchmark.json"))
    args = parser.parse_args()
    pilot = Path(args.pilot_dir)
    if not (pilot / "outputs/evidence_units_clean.jsonl").exists():
        raise SystemExit(f"pilot artifacts not found: {pilot}")

    baseline = evaluate(pilot, power=1.0, tau_gender=0.50, tau_age=0.50)
    improved = evaluate(pilot, power=1.5, tau_gender=0.65, tau_age=0.50)
    payload = {
        "note": "Pilot 50 clean + 10 injected; small sample, not a significance test.",
        "baseline": baseline,
        "improved": improved,
        "delta": {
            k: round(improved[k] - baseline[k], 4)
            for k in baseline if isinstance(baseline[k], float)
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
