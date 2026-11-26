# -*- coding: utf-8 -*-
"""Weighted evidence fusion + TV-based conflict detection / weight iteration.

This is the pilot's simplified implementation of the project's
"hierarchical Bayesian + conflict handling" stages:

  * evidence layer: each evidence unit supplies P(H | E) for the attribute;
  * attribute layer: gender / age-bucket hypotheses;
  * target layer: the predicted gender and age bucket.

Fusion is a weighted logarithmic opinion pool:
    log P(H | E_1..E_n) ∝ log prior(H) + Σ_i ω_i log P_i(H)
where ω_i is the evidence's effective credibility weight.  This is the
weighted-Bayes formula in the implementation plan, with the model
posteriors used as likelihood estimates (normalised per attribute).
"""
import math

from .credibility import recompute_score
from .schema import EvidenceUnit

GENDERS = ["male", "female"]
AGE_BUCKETS = ["18-24", "25-29", "30-34", "35-40"]
PRIORS = {h: 0.5 for h in GENDERS}
PRIORS.update({b: 0.25 for b in AGE_BUCKETS})


def _probs(e: EvidenceUnit, attr: str):
    return e.gender_probs if attr == "gender" else e.age_probs


def norm_probs(p: dict) -> dict:
    keys = p.keys()
    out = {k: max(0.0, float(p.get(k, 0.0))) for k in keys}
    s = sum(out.values())
    if s <= 0:
        n = len(out)
        return {k: 1.0 / n for k in out}
    return {k: v / s for k, v in out.items()}


def conflict_pair(p: dict, q: dict) -> dict:
    """Confidence-weighted total-variation conflict between distributions.

    v1.0 previously used ``1 - Σ p q`` as a "D-S mass".  For probability
    distributions on the same frame that quantity is nonzero even when
    ``p == q`` (e.g. two identical 0.8/0.2 distributions produce 0.32), so
    identical evidence was treated as conflicting.  The conflict measure is
    now total variation, which is 0 for identical distributions and 1 for
    disjoint ones, multiplied by the confidence of the two top labels.
    """
    keys = set(p) | set(q)
    pn, qn = norm_probs(p), norm_probs(q)
    tv = 0.5 * sum(abs(pn.get(k, 0.0) - qn.get(k, 0.0)) for k in keys)
    # Kept for audit/backward-compatible reporting; not used as conflict.
    ds = 1.0 - sum(pn.get(k, 0.0) * qn.get(k, 0.0) for k in keys)
    conf = min(max(pn.values()), max(qn.values()))
    conflict = tv * conf
    return {
        "tv": max(0.0, min(1.0, tv)),
        "ds": max(0.0, min(1.0, ds)),
        "tv_adj": max(0.0, min(1.0, conflict)),
        "conflict": max(0.0, min(1.0, conflict)),
    }


def weighted_fuse(prior: dict, evidence: list, attr: str,
                  use_credibility: bool = True,
                  weight_power: float = 1.0) -> tuple[dict, list]:
    logp = {h: math.log(max(prior.get(h, 1e-6), 1e-6)) for h in prior}
    used = []
    for e in evidence:
        p = _probs(e, attr)
        if not p:
            continue
        p = norm_probs(p)
        w = e.effective_weight() if use_credibility else 1.0
        if use_credibility and weight_power != 1.0:
            # Contrast: >1 sharpens the influence of high-credibility evidence.
            w = max(1e-6, min(1.0, w ** float(weight_power)))
        for h in logp:
            logp[h] += w * math.log(max(p.get(h, 1e-9), 1e-9))
        used.append({"evidence_id": e.evidence_id, "weight": round(w, 4)})
    m = max(logp.values())
    exps = {h: math.exp(v - m) for h, v in logp.items()}
    s = sum(exps.values())
    return {h: v / s for h, v in exps.items()}, used


def max_pairwise_conflict(evidence: list, attr: str, exclude: set | None = None) -> tuple[float, dict | None]:
    exclude = exclude or set()
    best = 0.0
    best_pair = None
    for i in range(len(evidence)):
        for j in range(i + 1, len(evidence)):
            pi, pj = _probs(evidence[i], attr), _probs(evidence[j], attr)
            if not pi or not pj:
                continue
            key = tuple(sorted([evidence[i].evidence_id, evidence[j].evidence_id]))
            if key in exclude:
                continue
            c = conflict_pair(pi, pj)
            if c["conflict"] > best:
                best = c["conflict"]
                best_pair = {
                    "a": evidence[i].evidence_id,
                    "b": evidence[j].evidence_id,
                    "tv": c["tv"],
                    "ds": c["ds"],
                }
    return best, best_pair


def update_consistency(evidence: list, attr: str):
    """Consistency = 1 - max conflict with any other evidence; then rescore."""
    for e in evidence:
        p = _probs(e, attr)
        if not p:
            continue
        mx = 0.0
        for f in evidence:
            if f is e:
                continue
            q = _probs(f, attr)
            if not q:
                continue
            mx = max(mx, conflict_pair(p, q)["conflict"])
        e.credibility.consistency = max(0.20, min(0.95, 1.0 - mx))
        e.credibility = recompute_score(e.credibility)
        e.note = (e.note + f" consistency={e.credibility.consistency:.2f}").strip()


def conflict_handle(evidence: list, attr: str, tau: float = 0.50,
                    max_rounds: int = 3, downweight: float = 0.25,
                    weight_power: float = 1.0):
    """Detect the strongest conflict, downweight the weaker source, iterate.

    Each conflicting pair is handled at most once: after the weaker source
    is downweighted, the winner is not penalised again.  This prevents the
    posterior from collapsing to the prior when two sources disagree.
    """
    ev = [e.model_copy(deep=True) for e in evidence]
    log = []
    handled = set()
    initial_max, initial_pair = max_pairwise_conflict(ev, attr)
    for rnd in range(max_rounds):
        mx, pair = max_pairwise_conflict(ev, attr, exclude=handled)
        if mx <= tau or pair is None:
            break
        by_id = {e.evidence_id: e for e in ev}
        a, b = by_id[pair["a"]], by_id[pair["b"]]
        key = tuple(sorted([pair["a"], pair["b"]]))
        handled.add(key)
        if a.effective_weight() < b.effective_weight():
            target, other = a, b
        elif b.effective_weight() < a.effective_weight():
            target, other = b, a
        else:
            if a.credibility.source_reliability <= b.credibility.source_reliability:
                target, other = a, b
            else:
                target, other = b, a
        target.weight_multiplier *= downweight
        target.conflict_flag = True
        target.conflict_degree = max(target.conflict_degree, mx)
        target.note = (
            target.note
            + f" downweighted round {rnd + 1}: conflict={mx:.2f} with {other.evidence_id}"
        ).strip()
        # only the losing source loses consistency; the winner is not penalised
        target.credibility.consistency = max(0.20, min(0.95, 1.0 - mx))
        target.credibility = recompute_score(target.credibility)
        log.append({
            "round": rnd + 1,
            "evidence_a": pair["a"],
            "evidence_b": pair["b"],
            "tv": round(pair["tv"], 4),
            "ds": round(pair["ds"], 4),
            "conflict": round(mx, 4),
            "downweighted": target.evidence_id,
            "weight_multiplier": round(target.weight_multiplier, 4),
        })
    prior = {h: 0.5 for h in GENDERS} if attr == "gender" else {b: 0.25 for b in AGE_BUCKETS}
    posterior, used = weighted_fuse(
        prior, ev, attr, use_credibility=True, weight_power=weight_power
    )
    return posterior, log, ev, initial_max, initial_pair, used
