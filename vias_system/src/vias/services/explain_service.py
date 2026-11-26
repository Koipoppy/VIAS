# -*- coding: utf-8 -*-
"""Explainability: leave-one-out contributions, evidence chain and narrative."""
from __future__ import annotations

from ..core.fusion import weighted_fuse
from ..core.schema import EvidenceUnit
from ..errors import ApiError
from ..repository import Repository, utc_now
from ..schemas import CandidateIdentity
from .identity_service import score_units
from .inference_service import InferenceService


def _probs(unit: EvidenceUnit, attr: str):
    return unit.gender_probs if attr == "gender" else unit.age_probs


def _summary(unit: EvidenceUnit) -> str:
    content = unit.content or {}
    if content.get("snippet"):
        return str(content["snippet"])[:120]
    claims = content.get("self_report") or {}
    if claims.get("snippets"):
        return " / ".join(str(x) for x in claims["snippets"][:2])[:120]
    return f"{unit.source_type} evidence"


class ExplainService:
    def __init__(self, repo: Repository, inference: InferenceService):
        self.repo = repo
        self.inference = inference

    def run(self, inference_id: str | None = None, case_id: str | None = None,
            attribute: str = "both", top_k: int = 3) -> dict:
        if inference_id:
            run = self.repo.get_inference(inference_id)
            if not run:
                raise ApiError(404, "INFERENCE_NOT_FOUND",
                               f"推断记录不存在：{inference_id}")
            if case_id and case_id != run["case_id"]:
                raise ApiError(400, "CASE_MISMATCH",
                               "case_id 与 inference_id 不匹配",
                               {"inference_case_id": run["case_id"]})
            case_id = run["case_id"]
        else:
            if not case_id:
                raise ApiError(400, "INVALID_REQUEST",
                               "必须提供 inference_id 或 case_id")
            run = self.inference.run(case_id, attribute=attribute)
            inference_id = run["inference_id"]

        all_units = self.repo.list_evidence(case_id)
        used_ids = {item["evidence_id"] for item in run.get("used_evidence", [])}
        results = {}

        if run.get("target") == "identity" or "identity" in (run.get("results") or {}):
            candidates = [CandidateIdentity.model_validate(x) for x in run.get("candidates", [])]
            if not candidates:
                raise ApiError(422, "NO_EXPLANATION", "身份推断缺少候选集")
            result = run["results"]["identity"]
            weight_power = float(run.get("weight_power", 1.5))
            identity_units = [u for u in all_units if not used_ids or u.evidence_id in used_ids]
            full = score_units(case_id, candidates, identity_units, weight_power)
            pred = full["prediction"]
            contributions = []
            for unit in identity_units:
                remaining = [u for u in identity_units if u.evidence_id != unit.evidence_id]
                if remaining:
                    try:
                        without = score_units(case_id, candidates, remaining, weight_power)
                        contrib = float(full["posterior"][pred]) - float(without["posterior"][pred])
                    except ApiError:
                        contrib = 0.0
                else:
                    contrib = float(full["posterior"][pred])
                contributions.append({
                    "evidence_id": unit.evidence_id,
                    "source_type": unit.source_type,
                    "source_name": unit.source_name,
                    "weight": round(unit.effective_weight(), 4),
                    "contribution": round(contrib, 6),
                    "direction": "support" if contrib > 0 else "contradict" if contrib < 0 else "neutral",
                    "summary": _summary(unit),
                })
            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            top = contributions[:top_k]
            nodes = [{"id": f"target:{case_id}", "type": "target",
                      "label": f"{case_id} · identity = {pred}"},
                     {"id": "hypothesis:identity", "type": "hypothesis",
                      "label": "H_identity"}]
            edges = [{"source": "hypothesis:identity", "target": f"target:{case_id}",
                      "weight": round(float(full["confidence"]), 4), "kind": "posterior"}]
            for item in top:
                eid = item["evidence_id"]
                nodes.append({"id": f"evidence:{eid}", "type": "evidence",
                              "label": f"{item['source_type']} · {eid}"})
                edges.append({"source": f"evidence:{eid}", "target": "hypothesis:identity",
                              "weight": item["contribution"], "kind": "leave_one_out"})
            supports = [x for x in top if x["contribution"] > 0]
            contradicts = [x for x in top if x["contribution"] < 0]
            parts = [f"案件 {case_id} 的目标身份排序第一为 {pred}（置信度 {float(full['confidence']):.3f}）。"]
            if supports:
                parts.append("主要支持证据：" + "、".join(
                    f"{x['evidence_id']}（+{x['contribution']:.3f}）" for x in supports) + "。")
            if contradicts:
                parts.append("主要反对证据：" + "、".join(
                    f"{x['evidence_id']}（{x['contribution']:.3f}）" for x in contradicts) + "。")
            if not supports and not contradicts:
                parts.append("没有可计算的证据贡献。")
            results["identity"] = {
                "prediction": pred,
                "confidence": full["confidence"],
                "posterior": full["posterior"],
                "ranking": full["ranking"],
                "top_evidence": top,
                "chain": {"nodes": nodes, "edges": edges},
                "narrative": "".join(parts),
            }
            attribute = "identity"
        for attr, result in (run.get("results") or {}).items():
            if attr == "identity":
                continue
            if "error" in result:
                results[attr] = result
                continue
            if attribute != "both" and attr != attribute:
                continue
            used_for_attr = [
                item["evidence_id"] for item in run.get("used_evidence", [])
                if item.get("attribute") == attr
            ]
            subset_ids = set(used_for_attr) or used_ids
            units = [u for u in all_units if u.evidence_id in subset_ids and _probs(u, attr)]
            if not units:
                results[attr] = {
                    "error": "no_probability_evidence",
                    "message": f"没有可用于 {attr} 解释的概率证据",
                }
                continue
            weight_power = float(run.get("weight_power", 1.5))
            prior = result.get("prior") or (
                {"male": 0.5, "female": 0.5} if attr == "gender"
                else {"18-24": 0.25, "25-29": 0.25, "30-34": 0.25, "35-40": 0.25}
            )
            posterior = result["posterior"]
            pred = result["prediction"]
            contributions = []
            for unit in units:
                remaining = [u for u in units if u.evidence_id != unit.evidence_id]
                if remaining:
                    without, _ = weighted_fuse(
                        prior, remaining, attr, use_credibility=True,
                        weight_power=weight_power
                    )
                    contrib = float(posterior[pred]) - float(without[pred])
                else:
                    contrib = float(posterior[pred])
                contributions.append({
                    "evidence_id": unit.evidence_id,
                    "source_type": unit.source_type,
                    "source_name": unit.source_name,
                    "weight": round(unit.effective_weight(), 4),
                    "contribution": round(contrib, 6),
                    "direction": "support" if contrib > 0 else "contradict" if contrib < 0 else "neutral",
                    "summary": _summary(unit),
                })
            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            top = contributions[:top_k]

            nodes = [{"id": f"target:{case_id}", "type": "target",
                      "label": f"{case_id} · {attr} = {pred}"}]
            nodes.append({"id": f"hypothesis:{attr}", "type": "hypothesis",
                          "label": f"H_{attr}"})
            edges = [{"source": f"hypothesis:{attr}", "target": f"target:{case_id}",
                      "weight": round(float(result.get("confidence", 0.0)), 4),
                      "kind": "posterior"}]
            for item in top:
                eid = item["evidence_id"]
                nodes.append({"id": f"evidence:{eid}", "type": "evidence",
                              "label": f"{item['source_type']} · {eid}"})
                edges.append({"source": f"evidence:{eid}",
                              "target": f"hypothesis:{attr}",
                              "weight": item["contribution"],
                              "kind": "leave_one_out"})

            supports = [x for x in top if x["contribution"] > 0]
            contradicts = [x for x in top if x["contribution"] < 0]
            parts = [
                f"案件 {case_id} 的 {attr} 融合结果为 {pred}（置信度 {float(result.get('confidence', 0.0)):.3f}）。"
            ]
            if supports:
                parts.append("主要支持证据：" + "、".join(
                    f"{x['evidence_id']}（+{x['contribution']:.3f}）" for x in supports
                ) + "。")
            if contradicts:
                parts.append("主要反对证据：" + "、".join(
                    f"{x['evidence_id']}（{x['contribution']:.3f}）" for x in contradicts
                ) + "。")
            if not supports and not contradicts:
                parts.append("没有可计算的证据贡献。")
            results[attr] = {
                "prediction": pred,
                "confidence": result.get("confidence"),
                "posterior": posterior,
                "top_evidence": top,
                "chain": {"nodes": nodes, "edges": edges},
                "narrative": "".join(parts),
            }

        if not results:
            raise ApiError(422, "NO_EXPLANATION",
                           "没有可用于解释的推断结果",
                           {"inference_id": inference_id})
        payload = {
            "explanation_id": None,
            "inference_id": inference_id,
            "case_id": case_id,
            "attribute": attribute,
            "results": results,
            "created_at": utc_now(),
        }
        explanation_id = self.repo.save_explanation(case_id, inference_id, payload)
        payload["explanation_id"] = explanation_id
        self.repo.audit("inference.explain", case_id, None,
                        {"explanation_id": explanation_id,
                         "inference_id": inference_id})
        return payload
