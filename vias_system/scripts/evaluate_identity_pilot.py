# -*- coding: utf-8 -*-
"""Synthetic candidate-layer evaluation on the pilot cases.

This is a *pilot* target-layer test, not a real identity benchmark.  For each
case we build a candidate set containing the true synthetic subject plus four
distractors, and rank candidates with the attribute-compatible scorer.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vias.core.schema import EvidenceUnit  # noqa: E402
from vias.schemas import CandidateIdentity  # noqa: E402
from vias.services.identity_service import score_units  # noqa: E402


def load_units(path: Path) -> dict[str, list[EvidenceUnit]]:
    out = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            unit = EvidenceUnit.model_validate(json.loads(line))
            out[unit.case_id].append(unit)
    return out


def load_labels(path: Path) -> dict[str, dict]:
    return {x["case_id"]: x for x in (
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", default=str(ROOT.parent / "vias_pilot"))
    ap.add_argument("--out", default=str(ROOT / "docs/identity_benchmark.json"))
    ap.add_argument("--distractors", type=int, default=4)
    args = ap.parse_args()
    pilot = Path(args.pilot_dir)
    labels = load_labels(pilot / "outputs/case_results_clean.jsonl")
    units = load_units(pilot / "outputs/evidence_units_clean.jsonl")
    ids = sorted(labels)
    rng = random.Random(20261125)

    top1 = top3 = 0
    details = []
    for cid in ids:
        row = labels[cid]
        candidates = [CandidateIdentity(
            candidate_id=f"cand:{cid}:true",
            display_name=f"true:{cid}",
            gender=row["true_gender"],
            age_bucket=row["true_age_bucket"],
        )]
        pool = [x for x in ids if x != cid and (
            labels[x]["true_gender"] != row["true_gender"]
            or labels[x]["true_age_bucket"] != row["true_age_bucket"]
        )]
        for j, other in enumerate(rng.sample(pool, min(args.distractors, len(pool)))):
            candidates.append(CandidateIdentity(
                candidate_id=f"cand:{cid}:d{j}",
                display_name=f"distractor:{other}",
                gender=labels[other]["true_gender"],
                age_bucket=labels[other]["true_age_bucket"],
            ))
        result = score_units(cid, candidates, units[cid], weight_power=1.5)
        true_id = candidates[0].candidate_id
        ranking = [x["candidate_id"] for x in result["ranking"]]
        top1 += ranking[0] == true_id
        top3 += true_id in ranking[:3]
        details.append({"case_id": cid, "n": len(candidates),
                        "true_rank": ranking.index(true_id) + 1,
                        "prediction": ranking[0]})
    payload = {
        "note": "Synthetic candidate sets from pilot subjects; attribute-compatible ranking, not real identity evidence.",
        "n": len(ids),
        "candidates_per_case": args.distractors + 1,
        "accuracy_at_1": round(top1 / len(ids), 4),
        "recall_at_3": round(top3 / len(ids), 4),
        "details": details,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "details"}, ensure_ascii=False, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
