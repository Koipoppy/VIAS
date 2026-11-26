# -*- coding: utf-8 -*-
"""Hierarchical weighted-Bayes inference service."""
from __future__ import annotations

from ..core.fusion import AGE_BUCKETS, GENDERS, weighted_fuse
from ..core.schema import EvidenceUnit
from ..errors import ApiError
from ..repository import Repository, utc_now
from .identity_service import IdentityService

DEFAULT_PRIORS = {
    "gender": {g: 0.5 for g in GENDERS},
    "age": {b: 0.25 for b in AGE_BUCKETS},
}


def _validate_prior(attr: str, prior: dict | None) -> dict:
    default = dict(DEFAULT_PRIORS[attr])
    if not prior:
        return default
    if attr == "gender":
        keys = set(GENDERS)
    else:
        keys = set(AGE_BUCKETS)
    if set(prior) != keys:
        raise ApiError(400, "INVALID_PRIOR",
                       f"{attr} 先验键必须为 {sorted(keys)}",
                       {"got": sorted(prior)})
    vals = {k: float(v) for k, v in prior.items()}
    if any(v < 0 for v in vals.values()):
        raise ApiError(400, "INVALID_PRIOR", "先验值不能为负")
    total = sum(vals.values())
    if total <= 0:
        raise ApiError(400, "INVALID_PRIOR", "先验和必须大于 0")
    return {k: v / total for k, v in vals.items()}


def _probs(unit: EvidenceUnit, attr: str):
    return unit.gender_probs if attr == "gender" else unit.age_probs


class InferenceService:
    def __init__(self, repo: Repository, identity: IdentityService | None = None):
        self.repo = repo
        self.identity = identity or IdentityService(repo)

    def run(self, case_id: str, attribute: str = "both",
            evidence_ids: list[str] | None = None,
            prior: dict | None = None,
            weight_power: float = 1.5,
            target: str = "attribute", candidates=None,
            context: dict | None = None) -> dict:
        if not self.repo.get_case(case_id):
            raise ApiError(404, "CASE_NOT_FOUND", f"案件不存在：{case_id}")
        if target == "identity":
            if not candidates:
                raise ApiError(400, "INVALID_CANDIDATES",
                               "target=identity 时必须提供 candidates")
            result = self.identity.score(case_id, candidates, evidence_ids, weight_power)
            payload = {
                "inference_id": None,
                "case_id": case_id,
                "target": "identity",
                "attribute": "identity",
                "weight_power": weight_power,
                "candidates": [c.model_dump() for c in candidates],
                "context": context or {},
                "results": {"identity": result},
                "used_evidence": result.get("used_evidence", []),
                "created_at": utc_now(),
            }
            inference_id = self.repo.save_inference(case_id, payload)
            payload["inference_id"] = inference_id
            self.repo.audit("inference.identity", case_id, None,
                            {"inference_id": inference_id,
                             "prediction": result["prediction"]})
            return payload
        if target != "attribute":
            raise ApiError(400, "INVALID_TARGET", "target 必须为 attribute 或 identity")
        units = self.repo.list_evidence(case_id, evidence_ids)
        if not units:
            raise ApiError(404, "EVIDENCE_NOT_FOUND", "案件下没有证据",
                           {"case_id": case_id})
        found = {u.evidence_id for u in units}
        if evidence_ids and set(evidence_ids) - found:
            raise ApiError(404, "EVIDENCE_NOT_FOUND", "部分证据不存在",
                           {"missing": sorted(set(evidence_ids) - found)})

        attrs = ["gender", "age"] if attribute == "both" else [attribute]
        if attribute == "both" and prior and set(prior) != {"gender", "age"}:
            raise ApiError(400, "INVALID_PRIOR",
                           "attribute=both 时 prior 必须为 {gender: {...}, age: {...}}",
                           {"got": sorted(prior)})
        results = {}
        used_all = []
        any_usable = False
        for attr in attrs:
            usable = [u for u in units if _probs(u, attr)]
            if not usable:
                results[attr] = {
                    "error": "no_probability_evidence",
                    "message": f"没有可用于 {attr} 的概率证据",
                    "n_evidence": len(units),
                }
                continue
            any_usable = True
            attr_prior = None
            if prior:
                attr_prior = prior.get(attr) if attribute == "both" else prior
            p = _validate_prior(attr, attr_prior)
            posterior, used = weighted_fuse(
                p, usable, attr, use_credibility=True, weight_power=weight_power
            )
            pred = max(posterior, key=posterior.get)
            results[attr] = {
                "posterior": {k: round(v, 6) for k, v in posterior.items()},
                "prediction": pred,
                "confidence": round(float(posterior[pred]), 6),
                "n_evidence": len(usable),
                "prior": p,
                "weight_power": weight_power,
            }
            for item in used:
                item["attribute"] = attr
                used_all.append(item)

        if not any_usable:
            raise ApiError(422, "NO_PROBABILITY_EVIDENCE",
                           "所选证据没有可用于推断的概率",
                           {"case_id": case_id, "attributes": attrs})

        payload = {
            "inference_id": None,
            "case_id": case_id,
            "target": "attribute",
            "attribute": attribute,
            "weight_power": weight_power,
            "context": context or {},
            "results": results,
            "used_evidence": used_all,
            "created_at": utc_now(),
        }
        inference_id = self.repo.save_inference(case_id, payload)
        payload["inference_id"] = inference_id
        self.repo.audit("inference.bayes", case_id, None,
                        {"inference_id": inference_id, "attribute": attribute})
        return payload

    def get(self, inference_id: str) -> dict:
        payload = self.repo.get_inference(inference_id)
        if not payload:
            raise ApiError(404, "INFERENCE_NOT_FOUND",
                           f"推断记录不存在：{inference_id}")
        return payload
