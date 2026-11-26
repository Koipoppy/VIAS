# -*- coding: utf-8 -*-
"""Evidence conflict detection and confidence-weighted downweighting."""
from __future__ import annotations

from ..core.fusion import conflict_handle
from ..core.schema import EvidenceUnit
from ..errors import ApiError
from ..repository import Repository, utc_now
from .identity_service import IdentityService


def _probs(unit: EvidenceUnit, attr: str):
    return unit.gender_probs if attr == "gender" else unit.age_probs


class ConflictService:
    def __init__(self, repo: Repository, identity: IdentityService | None = None):
        self.repo = repo
        self.identity = identity or IdentityService(repo)

    def run(self, case_id: str, attribute: str = "both",
            evidence_ids: list[str] | None = None, tau: float | None = None,
            tau_gender: float = 0.65, tau_age: float = 0.50,
            tau_identity: float = 0.50, candidates=None,
            downweight: float = 0.25, max_rounds: int = 3,
            weight_power: float = 1.5, persist: bool = True) -> dict:
        if not self.repo.get_case(case_id):
            raise ApiError(404, "CASE_NOT_FOUND", f"案件不存在：{case_id}")
        units = self.repo.list_evidence(case_id, evidence_ids)
        if not units:
            raise ApiError(404, "EVIDENCE_NOT_FOUND", "案件下没有证据",
                           {"case_id": case_id})
        found = {u.evidence_id for u in units}
        if evidence_ids and set(evidence_ids) - found:
            raise ApiError(404, "EVIDENCE_NOT_FOUND", "部分证据不存在",
                           {"missing": sorted(set(evidence_ids) - found)})

        if attribute == "identity":
            if not candidates:
                raise ApiError(400, "INVALID_CANDIDATES",
                               "attribute=identity 时必须提供 candidates")
            identity_result = self.identity.conflict(
                case_id, candidates, evidence_ids=evidence_ids,
                tau=tau if tau is not None else tau_identity,
                downweight=downweight, max_rounds=max_rounds,
                weight_power=weight_power,
            )
            adjusted = identity_result.pop("adjusted", [])
            if persist:
                for unit in adjusted:
                    self.repo.update_evidence(unit)
            payload = {
                "conflict_id": None,
                "case_id": case_id,
                "tau": tau,
                "taus": {"identity": tau if tau is not None else tau_identity},
                "weight_power": weight_power,
                "attributes": {"identity": identity_result},
                "downweighted": [x["downweighted"] for x in identity_result["rounds"]],
                "persisted": bool(persist),
                "created_at": utc_now(),
            }
            conflict_id = self.repo.save_conflict(case_id, payload)
            payload["conflict_id"] = conflict_id
            self.repo.audit("evidence.conflict.identity", case_id, None,
                            {"conflict_id": conflict_id,
                             "downweighted": payload["downweighted"]})
            return payload

        taus = {
            "gender": float(tau if tau is not None else tau_gender),
            "age": float(tau if tau is not None else tau_age),
        }
        attrs = ["gender", "age"] if attribute == "both" else [attribute]
        attributes = {}
        downweighted: list[str] = []
        working = {u.evidence_id: u.model_copy(deep=True) for u in units}
        any_usable = False
        for attr in attrs:
            usable = [u for u in working.values() if _probs(u, attr)]
            if not usable:
                attributes[attr] = {
                    "detected": False,
                    "error": "no_probability_evidence",
                    "max_conflict": 0.0,
                    "rounds": [],
                }
                continue
            any_usable = True
            posterior, log, adjusted, initial_max, initial_pair, used = conflict_handle(
                usable, attr, tau=taus[attr], max_rounds=max_rounds,
                downweight=downweight, weight_power=weight_power
            )
            pred = max(posterior, key=posterior.get)
            attributes[attr] = {
                "detected": bool(initial_max > taus[attr]),
                "tau": taus[attr],
                "max_conflict": round(float(initial_max), 6),
                "initial_pair": initial_pair,
                "rounds": log,
                "posterior": {k: round(v, 6) for k, v in posterior.items()},
                "prediction": pred,
                "confidence": round(float(posterior[pred]), 6),
                "n_evidence": len(usable),
            }
            for e in adjusted:
                working[e.evidence_id] = e
            for item in log:
                eid = item.get("downweighted")
                if eid and eid not in downweighted:
                    downweighted.append(eid)

        if not any_usable:
            raise ApiError(422, "NO_PROBABILITY_EVIDENCE",
                           "所选证据没有可用于冲突处理的概率",
                           {"case_id": case_id, "attributes": attrs})

        if persist:
            for unit in working.values():
                self.repo.update_evidence(unit)

        payload = {
            "conflict_id": None,
            "case_id": case_id,
            "tau": tau,
            "taus": taus,
            "weight_power": weight_power,
            "attributes": attributes,
            "downweighted": downweighted,
            "persisted": bool(persist),
            "created_at": utc_now(),
        }
        conflict_id = self.repo.save_conflict(case_id, payload)
        payload["conflict_id"] = conflict_id
        self.repo.audit("evidence.conflict", case_id, None,
                        {"conflict_id": conflict_id, "downweighted": downweighted})
        return payload
