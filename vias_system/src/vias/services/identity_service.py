# -*- coding: utf-8 -*-
"""Target-identity / candidate-layer inference.

Two modes are supported:

* ``identity`` evidence: evidence carries ``identity_probs`` over candidate IDs;
* ``attributes`` evidence: candidates have gender/age attributes and the
  evidence's gender/age posteriors are used as a compatibility likelihood.

The latter is the pilot fallback when no true candidate-specific evidence
(face match, name/relation match, etc.) is available.  The API contract is the
same; replacing the scorer with a real candidate matcher requires no endpoint
change.
"""
from __future__ import annotations

import math
from typing import Iterable

from ..core.credibility import recompute_score
from ..core.fusion import conflict_pair
from ..core.schema import EvidenceUnit
from ..errors import ApiError
from ..repository import Repository
from ..schemas import CandidateIdentity

EPS = 1e-9


def normalize_candidates(candidates: Iterable[CandidateIdentity]) -> tuple[list[CandidateIdentity], dict[str, float]]:
    items = list(candidates)
    if not items:
        raise ApiError(400, "INVALID_CANDIDATES", "target=identity 时必须提供 candidates")
    ids = [c.candidate_id for c in items]
    if len(set(ids)) != len(ids):
        raise ApiError(400, "INVALID_CANDIDATES", "candidate_id 不能重复")
    provided = [c.prior for c in items if c.prior is not None]
    priors: dict[str, float]
    if not provided:
        priors = {c.candidate_id: 1.0 / len(items) for c in items}
    else:
        total = sum(float(p) for p in provided)
        if total > 1.0 + 1e-9:
            raise ApiError(400, "INVALID_CANDIDATES", "候选先验之和不能超过 1")
        missing = len(items) - len(provided)
        rest = max(0.0, 1.0 - total) / missing if missing else 0.0
        priors = {c.candidate_id: (float(c.prior) if c.prior is not None else rest) for c in items}
        s = sum(priors.values())
        if s <= 0:
            priors = {c.candidate_id: 1.0 / len(items) for c in items}
        else:
            priors = {k: v / s for k, v in priors.items()}
    return items, priors


def _candidate_likelihood(unit: EvidenceUnit, candidate: CandidateIdentity) -> tuple[float, str | None]:
    """Return (likelihood, mode) for one evidence/candidate pair.

    None means the evidence cannot score this candidate (neutral).
    """
    if unit.identity_probs:
        return max(EPS, float(unit.identity_probs.get(candidate.candidate_id, 0.0))), "identity"
    likelihood = 1.0
    matched = False
    if candidate.gender and unit.gender_probs:
        likelihood *= max(EPS, float(unit.gender_probs.get(candidate.gender, 0.0)))
        matched = True
    if candidate.age_bucket and unit.age_probs:
        likelihood *= max(EPS, float(unit.age_probs.get(candidate.age_bucket, 0.0)))
        matched = True
    if not matched:
        return 1.0, None
    return max(EPS, likelihood), "attributes"


def score_units(case_id: str, candidates: list[CandidateIdentity],
                units: list[EvidenceUnit], weight_power: float = 1.5) -> dict:
    items, priors = normalize_candidates(candidates)
    logp = {c.candidate_id: math.log(max(priors[c.candidate_id], EPS)) for c in items}
    used: dict[str, dict] = {}
    modes = set()
    for unit in units:
        w = max(1e-6, min(1.0, unit.effective_weight() ** float(weight_power)))
        for cand in items:
            likelihood, mode = _candidate_likelihood(unit, cand)
            if mode is None:
                continue
            logp[cand.candidate_id] += w * math.log(likelihood)
            modes.add(mode)
            used[unit.evidence_id] = {
                "evidence_id": unit.evidence_id,
                "source_type": unit.source_type,
                "weight": round(w, 4),
                "mode": mode,
            }
    if not used:
        raise ApiError(422, "NO_IDENTITY_EVIDENCE",
                       "没有可用于候选身份推断的证据：需要 identity_probs 或候选属性兼容证据",
                       {"case_id": case_id, "candidates": [c.candidate_id for c in items]})
    m = max(logp.values())
    exps = {k: math.exp(v - m) for k, v in logp.items()}
    total = sum(exps.values())
    posterior = {k: v / total for k, v in exps.items()}
    ranking = sorted(
        [
            {
                "candidate_id": c.candidate_id,
                "display_name": c.display_name,
                "prior": round(priors[c.candidate_id], 6),
                "probability": round(posterior[c.candidate_id], 6),
            }
            for c in items
        ],
        key=lambda x: x["probability"],
        reverse=True,
    )
    pred = ranking[0]["candidate_id"]
    mode = "identity" if modes == {"identity"} else "attributes" if modes == {"attributes"} else "mixed"
    return {
        "posterior": {k: round(v, 6) for k, v in posterior.items()},
        "ranking": ranking,
        "prediction": pred,
        "confidence": ranking[0]["probability"],
        "n_candidates": len(items),
        "n_evidence": len(used),
        "mode": mode,
        "weight_power": weight_power,
        "prior": {k: round(v, 6) for k, v in priors.items()},
        "used_evidence": list(used.values()),
    }


def identity_fuse(prior: dict[str, float], units: list[EvidenceUnit],
                  weight_power: float = 1.5) -> tuple[dict[str, float], list[dict]]:
    logp = {cid: math.log(max(float(p), EPS)) for cid, p in prior.items()}
    used = []
    for unit in units:
        if not unit.identity_probs:
            continue
        p = unit.identity_probs
        total = sum(max(0.0, float(x)) for x in p.values())
        if total <= 0:
            continue
        p = {k: max(0.0, float(v)) / total for k, v in p.items()}
        w = max(1e-6, min(1.0, unit.effective_weight() ** float(weight_power)))
        for cid in logp:
            logp[cid] += w * math.log(max(p.get(cid, 0.0), EPS))
        used.append({"evidence_id": unit.evidence_id, "weight": round(w, 4)})
    m = max(logp.values())
    exps = {k: math.exp(v - m) for k, v in logp.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}, used


def identity_conflict(units: list[EvidenceUnit], candidates: list[CandidateIdentity],
                      tau: float = 0.50, downweight: float = 0.25,
                      max_rounds: int = 3, weight_power: float = 1.5):
    items, prior = normalize_candidates(candidates)
    ev = [u.model_copy(deep=True) for u in units if u.identity_probs]
    if not ev:
        raise ApiError(422, "NO_IDENTITY_EVIDENCE",
                       "没有可用于身份冲突处理的 identity_probs 证据")
    log = []
    handled = set()
    initial_max = 0.0
    initial_pair = None
    for _ in range(max_rounds + 1):
        best = 0.0
        best_pair = None
        for i in range(len(ev)):
            for j in range(i + 1, len(ev)):
                pi, pj = ev[i].identity_probs, ev[j].identity_probs
                if not pi or not pj:
                    continue
                key = tuple(sorted([ev[i].evidence_id, ev[j].evidence_id]))
                if key in handled:
                    continue
                c = conflict_pair(pi, pj)
                if c["conflict"] > best:
                    best = c["conflict"]
                    best_pair = {"a": ev[i].evidence_id, "b": ev[j].evidence_id,
                                 "tv": round(c["tv"], 4), "ds": round(c["ds"], 4)}
        if initial_pair is None:
            initial_max, initial_pair = best, best_pair
        if best <= tau or best_pair is None or len(log) >= max_rounds:
            break
        by_id = {e.evidence_id: e for e in ev}
        a, b = by_id[best_pair["a"]], by_id[best_pair["b"]]
        handled.add(tuple(sorted([a.evidence_id, b.evidence_id])))
        if a.effective_weight() < b.effective_weight():
            target, other = a, b
        elif b.effective_weight() < a.effective_weight():
            target, other = b, a
        else:
            target, other = (a, b) if a.credibility.source_reliability <= b.credibility.source_reliability else (b, a)
        target.weight_multiplier *= downweight
        target.conflict_flag = True
        target.conflict_degree = max(target.conflict_degree, best)
        target.credibility.consistency = max(0.20, min(0.95, 1.0 - best))
        target.credibility = recompute_score(target.credibility)
        target.note = (target.note + f" identity downweighted: conflict={best:.2f} with {other.evidence_id}").strip()
        log.append({"round": len(log) + 1, "evidence_a": a.evidence_id,
                    "evidence_b": b.evidence_id, "conflict": round(best, 4),
                    "tv": best_pair["tv"], "ds": best_pair["ds"],
                    "downweighted": target.evidence_id,
                    "weight_multiplier": round(target.weight_multiplier, 4)})
    posterior, used = identity_fuse(prior, ev, weight_power)
    ranking = sorted(
        [{"candidate_id": c.candidate_id, "display_name": c.display_name,
          "prior": round(prior[c.candidate_id], 6),
          "probability": round(posterior[c.candidate_id], 6)} for c in items],
        key=lambda x: x["probability"], reverse=True,
    )
    return {
        "detected": bool(initial_max > tau),
        "max_conflict": round(float(initial_max), 6),
        "initial_pair": initial_pair,
        "rounds": log,
        "posterior": {k: round(v, 6) for k, v in posterior.items()},
        "ranking": ranking,
        "prediction": ranking[0]["candidate_id"],
        "confidence": ranking[0]["probability"],
        "n_evidence": len(used),
        "used_evidence": used,
        "adjusted": ev,
    }


class IdentityService:
    def __init__(self, repo: Repository):
        self.repo = repo

    def score(self, case_id: str, candidates: list[CandidateIdentity],
              evidence_ids: list[str] | None = None,
              weight_power: float = 1.5) -> dict:
        if not self.repo.get_case(case_id):
            raise ApiError(404, "CASE_NOT_FOUND", f"案件不存在：{case_id}")
        if not candidates:
            raise ApiError(400, "INVALID_CANDIDATES", "target=identity 时必须提供 candidates")
        units = self.repo.list_evidence(case_id, evidence_ids)
        if not units:
            raise ApiError(404, "EVIDENCE_NOT_FOUND", "案件下没有证据",
                           {"case_id": case_id})
        found = {u.evidence_id for u in units}
        if evidence_ids and set(evidence_ids) - found:
            raise ApiError(404, "EVIDENCE_NOT_FOUND", "部分证据不存在",
                           {"missing": sorted(set(evidence_ids) - found)})
        return score_units(case_id, candidates, units, weight_power)

    def conflict(self, case_id: str, candidates: list[CandidateIdentity],
                 evidence_ids: list[str] | None = None, tau: float = 0.50,
                 downweight: float = 0.25, max_rounds: int = 3,
                 weight_power: float = 1.5) -> dict:
        if not self.repo.get_case(case_id):
            raise ApiError(404, "CASE_NOT_FOUND", f"案件不存在：{case_id}")
        if not candidates:
            raise ApiError(400, "INVALID_CANDIDATES", "attribute=identity 时必须提供 candidates")
        units = self.repo.list_evidence(case_id, evidence_ids)
        if not units:
            raise ApiError(404, "EVIDENCE_NOT_FOUND", "案件下没有证据",
                           {"case_id": case_id})
        found = {u.evidence_id for u in units}
        if evidence_ids and set(evidence_ids) - found:
            raise ApiError(404, "EVIDENCE_NOT_FOUND", "部分证据不存在",
                           {"missing": sorted(set(evidence_ids) - found)})
        return identity_conflict(units, candidates, tau=tau, downweight=downweight,
                                 max_rounds=max_rounds, weight_power=weight_power)
