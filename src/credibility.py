# -*- coding: utf-8 -*-
"""Rule-based six-dimension credibility scoring for the pilot.

Dimensions follow the implementation plan:
  source_reliability (来源), authority (权威性), objectivity (客观性),
  timeliness (时效性), completeness (完整性), consistency (一致性).

The score is a weighted mean.  consistency is updated after pairwise
conflict detection, which is the pilot's simplified "weight iteration".
"""
from schema import Credibility

DIM_WEIGHTS = {
    "source_reliability": 0.20,
    "authority": 0.15,
    "objectivity": 0.20,
    "timeliness": 0.15,
    "completeness": 0.15,
    "consistency": 0.15,
}

BASE_PROFILES = {
    "text_model": {
        "source_reliability": 0.65,
        "authority": 0.60,
        "objectivity": 0.75,
        "timeliness": 0.85,
        "completeness": 0.70,
        "consistency": 0.70,
    },
    "self_report": {
        "source_reliability": 0.75,
        "authority": 0.80,
        "objectivity": 0.55,
        "timeliness": 0.85,
        "completeness": 0.60,
        "consistency": 0.70,
    },
    "face_model": {
        "source_reliability": 0.70,
        "authority": 0.65,
        "objectivity": 0.85,
        "timeliness": 0.80,
        "completeness": 0.80,
        "consistency": 0.70,
    },
    "synthetic_conflict": {
        "source_reliability": 0.68,
        "authority": 0.72,
        "objectivity": 0.55,
        "timeliness": 0.80,
        "completeness": 0.60,
        "consistency": 0.50,
    },
    # Reserved interface profile: evidence is present but has no probabilities yet.
    "stub": {
        "source_reliability": 0.30,
        "authority": 0.30,
        "objectivity": 0.50,
        "timeliness": 0.50,
        "completeness": 0.10,
        "consistency": 0.50,
    },
}


def weighted_score(dims: dict) -> float:
    return sum(DIM_WEIGHTS[k] * float(dims[k]) for k in DIM_WEIGHTS)


def score_evidence(source_type: str, n_chars: int = 0, self_conf: float = 0.0,
                   face_quality: float = 0.80) -> Credibility:
    dims = dict(BASE_PROFILES[source_type])
    if source_type == "text_model":
        # longer text -> more complete evidence
        dims["completeness"] = max(0.40, min(1.0, n_chars / 5000.0))
    elif source_type == "self_report":
        dims["completeness"] = max(0.50, min(1.0, 0.50 + 0.40 * self_conf))
        dims["objectivity"] = max(0.45, min(0.70, 0.45 + 0.20 * (1.0 - self_conf)))
    elif source_type == "face_model":
        dims["completeness"] = face_quality
    s = weighted_score(dims)
    return Credibility(**dims, score=s, weight=s)


def recompute_score(c: Credibility) -> Credibility:
    dims = {
        "source_reliability": c.source_reliability,
        "authority": c.authority,
        "objectivity": c.objectivity,
        "timeliness": c.timeliness,
        "completeness": c.completeness,
        "consistency": c.consistency,
    }
    s = weighted_score(dims)
    return Credibility(**dims, score=s, weight=s)
