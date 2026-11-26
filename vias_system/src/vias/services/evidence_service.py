# -*- coding: utf-8 -*-
"""Evidence ingestion and six-dimension credibility evaluation."""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from ..config import Settings
from ..core.credibility import DIM_WEIGHTS, recompute_score, score_evidence
from ..core.face_parser import FaceParser
from ..core.schema import AGE_BUCKETS, Credibility, EvidenceUnit
from ..core.text_parser import TextParser
from ..errors import ApiError
from ..repository import Repository, utc_now
from ..schemas import ALLOWED_SOURCE_MODALITY, EvidenceUploadJson
from ..storage import Storage

PII_PATTERNS = [
    (re.compile(r"https?://\S+"), "[URL]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"@[\w.-]+"), "[MENTION]"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[PHONE]"),
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "[ID]"),
    (re.compile(r"(?<!\d)\d{6,}(?!\d)"), "[NUM]"),
]


def sanitize_text(text: str) -> tuple[str, bool]:
    changed = False
    out = text
    for pattern, repl in PII_PATTERNS:
        new = pattern.sub(repl, out)
        if new != out:
            changed = True
            out = new
    return out, changed


def _norm(value: dict | None) -> dict | None:
    if value is None:
        return None
    out = {k: max(0.0, float(v)) for k, v in value.items()}
    total = sum(out.values())
    if total <= 0:
        return out
    return {k: v / total for k, v in out.items()}


def _client_probs(value: dict | None, keys: set[str], label: str) -> dict | None:
    if value is None:
        return None
    if set(value) != keys:
        raise ApiError(400, "INVALID_PROBABILITIES",
                       f"{label} 的键必须为 {sorted(keys)}",
                       {"got": sorted(value)})
    out = {k: float(v) for k, v in value.items()}
    if any(v < 0 or v > 1 for v in out.values()):
        raise ApiError(400, "INVALID_PROBABILITIES",
                       f"{label} 的值必须在 [0,1]")
    if abs(sum(out.values()) - 1.0) > 1e-6:
        raise ApiError(400, "INVALID_PROBABILITIES",
                       f"{label} 的概率和必须为 1",
                       {"sum": sum(out.values())})
    return out


class EvidenceService:
    def __init__(self, settings: Settings, repo: Repository,
                 storage: Storage, text_parser: TextParser,
                 face_parser: FaceParser):
        self.settings = settings
        self.repo = repo
        self.storage = storage
        self.text_parser = text_parser
        self.face_parser = face_parser

    def _validate_combo(self, payload: EvidenceUploadJson):
        allowed = ALLOWED_SOURCE_MODALITY.get(payload.source_type, set())
        if payload.modality not in allowed:
            raise ApiError(
                400, "SOURCE_MODALITY_MISMATCH",
                f"source_type={payload.source_type} 不支持 modality={payload.modality}",
                {"allowed": sorted(allowed)},
            )

    def _credibility(self, source_type: str, content: dict,
                     overrides: dict[str, float] | None = None) -> Credibility:
        claims = content.get("self_report") or {}
        base = score_evidence(
            source_type,
            n_chars=int(content.get("n_chars", 0) or 0),
            self_conf=float(claims.get("age_conf", claims.get("gender_conf", 0.0)) or 0.0),
            face_quality=float(content.get("face_quality", 0.80) or 0.80),
        )
        if not overrides:
            return base
        data = base.model_dump()
        for key, value in overrides.items():
            if key not in DIM_WEIGHTS:
                raise ApiError(400, "INVALID_OVERRIDE",
                               f"可信度维度不存在：{key}",
                               {"allowed": sorted(DIM_WEIGHTS)})
            if not 0.0 <= float(value) <= 1.0:
                raise ApiError(400, "INVALID_OVERRIDE",
                               f"{key} 必须在 [0,1]")
            data[key] = float(value)
        data["score"] = float(sum(DIM_WEIGHTS[k] * data[k] for k in DIM_WEIGHTS))
        data["weight"] = data["score"]
        return Credibility(**data)

    def create(self, payload: EvidenceUploadJson, raw: bytes | None = None,
               filename: str | None = None) -> dict:
        self._validate_combo(payload)
        self.repo.ensure_case(payload.case_id)

        evidence_id = payload.evidence_id or f"ev_{uuid.uuid4().hex[:16]}"
        if self.repo.get_evidence(evidence_id):
            raise ApiError(409, "EVIDENCE_EXISTS", f"证据已存在：{evidence_id}",
                           {"evidence_id": evidence_id})

        filename = filename or payload.filename or "blob.bin"
        raw_info = None
        if raw is None and payload.file_base64:
            import base64
            try:
                raw = base64.b64decode(payload.file_base64, validate=True)
            except Exception as exc:
                raise ApiError(400, "INVALID_FILE", "file_base64 不是合法 base64") from exc
        if raw is not None:
            raw_info = self.storage.save(payload.case_id, evidence_id, filename, raw)

        content = dict(payload.content or {})
        text = payload.text
        if text is None and payload.modality == "audio":
            text = content.get("asr_text")
        if text is None and raw is not None and payload.modality == "text":
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ApiError(400, "TEXT_DECODE_ERROR",
                               "文本文件必须是 UTF-8") from exc
        if text is not None:
            text, changed = sanitize_text(text)
            content["text"] = text
            content["n_chars"] = len(text)
            content["snippet"] = text[:160]
            payload.pii_redacted = payload.pii_redacted or changed

        if raw_info:
            content.update({
                "filename": raw_info["filename"],
                "bytes": raw_info["bytes"],
                "sha256": raw_info["sha256"],
            })

        gp = _client_probs(payload.gender_probs, {"male", "female"}, "gender_probs")
        ap = _client_probs(payload.age_probs, set(AGE_BUCKETS), "age_probs")
        ip = _client_probs(
            payload.identity_probs,
            set(payload.identity_probs or {}),
            "identity_probs",
        )
        if ip is not None and not ip:
            raise ApiError(400, "INVALID_PROBABILITIES", "identity_probs 不能为空")
        if gp is None and ap is None and payload.modality == "text" and text:
            gp, ap, parsed = self.text_parser.parse(text, payload.source_type)
            content.update(parsed)
        elif gp is None and ap is None and payload.modality == "audio" and text:
            # ASR transcript sample chain: the caller provides ASR text; the
            # text model turns it into attribute evidence.
            gp, ap, parsed = self.text_parser.parse(text, "text_model")
            content.update(parsed)
            content["asr_text"] = text
            content["status"] = "asr_ready"
        elif gp is not None or ap is not None:
            content.setdefault("model", "client_supplied")

        if payload.modality == "behavior" and isinstance(content.get("events"), list):
            events = [e for e in content["events"] if isinstance(e, dict)]
            content["n_events"] = len(events)
            content["event_types"] = sorted({str(e.get("action")) for e in events if e.get("action")})
            content["n_ips"] = len({str(e.get("ip")) for e in events if e.get("ip")})
            content["n_devices"] = len({str(e.get("device")) for e in events if e.get("device")})
            content["status"] = "parsed"

        if payload.modality == "image" and gp is None and ap is None:
            path = raw_info["path"] if raw_info else None
            if path:
                gp, ap, parsed = self.face_parser.parse(path)
                content.update(parsed)
            else:
                content.setdefault("status", "stored")
        elif payload.modality in {"audio", "behavior"}:
            if gp is not None or ap is not None or ip is not None:
                content.setdefault("status", "asr_ready" if payload.modality == "audio" else "parsed")
            elif content.get("status") not in {"asr_ready", "parsed"}:
                content["status"] = "reserved"
            content.setdefault("note", "audio/behavior model reserved; ASR transcript/behavior features only")
        elif payload.modality == "metadata":
            content.setdefault("status", "metadata")

        credibility = self._credibility(payload.source_type, content)
        unit = EvidenceUnit(
            schema_version="1.0",
            evidence_id=evidence_id,
            case_id=payload.case_id,
            modality=payload.modality,
            source_type=payload.source_type,
            source_name=payload.source_name,
            content=content,
            raw_ref=raw_info["raw_ref"] if raw_info else None,
            time_range=payload.time_range,
            language=payload.language,
            gender_probs=gp,
            age_probs=ap,
            identity_probs=ip,
            credibility=credibility,
            injected=payload.injected,
            synthetic=payload.synthetic,
            pii_redacted=payload.pii_redacted,
        )
        self.repo.insert_evidence(unit)
        self.repo.audit("evidence.upload", payload.case_id, payload.request_id,
                        {"evidence_id": evidence_id, "modality": payload.modality})
        return self._response(unit, raw_info)

    def _response(self, unit: EvidenceUnit, raw_info: dict | None = None,
                  created_at: str | None = None) -> dict:
        if unit.modality in {"audio", "behavior"} and unit.content.get("status") == "reserved":
            status = "reserved"
        elif unit.gender_probs or unit.age_probs or unit.identity_probs:
            status = "ready"
        else:
            status = "stored_no_probabilities"
        return {
            "evidence_id": unit.evidence_id,
            "case_id": unit.case_id,
            "status": status,
            "modality": unit.modality,
            "source_type": unit.source_type,
            "source_name": unit.source_name,
            "schema_version": unit.schema_version,
            "raw_ref": unit.raw_ref,
            "content": {
                k: v for k, v in unit.content.items()
                if k not in {"text"}  # do not echo full text
            },
            "content_snippet": unit.content.get("snippet"),
            "probabilities": {
                "gender": unit.gender_probs,
                "age": unit.age_probs,
                "identity": unit.identity_probs,
            },
            "credibility": unit.credibility.model_dump(),
            "created_at": created_at or utc_now(),
            "storage": raw_info and {
                "bytes": raw_info["bytes"], "sha256": raw_info["sha256"],
            },
        }

    def evaluate(self, case_id: str, evidence_ids: list[str] | None = None,
                 overrides: dict[str, dict[str, float]] | None = None) -> dict:
        if not self.repo.get_case(case_id):
            raise ApiError(404, "CASE_NOT_FOUND", f"案件不存在：{case_id}")
        units = self.repo.list_evidence(case_id, evidence_ids)
        if not units:
            raise ApiError(404, "EVIDENCE_NOT_FOUND", "案件下没有证据",
                           {"case_id": case_id})
        found = {u.evidence_id for u in units}
        if evidence_ids and set(evidence_ids) - found:
            raise ApiError(404, "EVIDENCE_NOT_FOUND",
                           "部分证据不存在",
                           {"missing": sorted(set(evidence_ids) - found)})
        results = []
        for unit in units:
            override = (overrides or {}).get(unit.evidence_id)
            unit.credibility = self._credibility(unit.source_type, unit.content, override)
            self.repo.update_evidence(unit)
            results.append({
                "evidence_id": unit.evidence_id,
                "source_type": unit.source_type,
                "credibility": unit.credibility.model_dump(),
                "effective_weight": round(unit.effective_weight(), 4),
            })
        self.repo.audit("evidence.evaluate", case_id, None,
                        {"evidence_ids": [u.evidence_id for u in units]})
        return {"case_id": case_id, "evaluated": results}

    def get(self, evidence_id: str) -> dict:
        unit = self.repo.get_evidence(evidence_id)
        if not unit:
            raise ApiError(404, "EVIDENCE_NOT_FOUND", f"证据不存在：{evidence_id}")
        return self._response(unit, created_at=self.repo.evidence_created_at(evidence_id))

    def list(self, case_id: str) -> dict:
        if not self.repo.get_case(case_id):
            raise ApiError(404, "CASE_NOT_FOUND", f"案件不存在：{case_id}")
        units = self.repo.list_evidence(case_id)
        return {"case_id": case_id, "count": len(units),
                "evidence": [
                    self._response(u, created_at=self.repo.evidence_created_at(u.evidence_id))
                    for u in units
                ]}
