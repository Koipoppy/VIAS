# -*- coding: utf-8 -*-
"""Evidence ingestion, evaluation and conflict endpoints."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from ..config import Settings
from ..errors import ApiError
from ..schemas import (ConflictRequest, ConflictResponse, EvaluateRequest,
                       EvaluateResponse, EvidenceUploadJson, UploadResponse)
from ..services.conflict_service import ConflictService
from ..services.evidence_service import EvidenceService
from .deps import get_conflict_service, get_evidence_service, get_settings

router = APIRouter(prefix="/api/evidence", tags=["evidence"])

_UPLOAD_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["case_id", "modality", "source_type", "source_name"],
                    "properties": {
                        "case_id": {"type": "string"},
                        "modality": {"type": "string", "enum": ["text", "image", "audio", "behavior", "metadata"]},
                        "source_type": {"type": "string", "enum": ["text_model", "face_model", "self_report", "audio_stub", "behavior_stub", "synthetic_conflict", "metadata"]},
                        "source_name": {"type": "string"},
                        "text": {"type": "string"},
                        "file": {"type": "string", "format": "binary"},
                        "content": {"type": "string", "description": "JSON object string"},
                        "metadata": {"type": "string", "description": "JSON object string"},
                        "gender_probs": {"type": "string", "description": "JSON object string"},
                        "age_probs": {"type": "string", "description": "JSON object string"},
                        "identity_probs": {"type": "string", "description": "candidate_id -> probability JSON object string"},
                        "time_range": {"type": "string", "description": "JSON object string"},
                        "language": {"type": "string"},
                        "synthetic": {"type": "boolean"},
                        "pii_redacted": {"type": "boolean"},
                        "injected": {"type": "boolean"},
                        "request_id": {"type": "string"},
                    },
                }
            },
            "application/json": {"schema": EvidenceUploadJson.model_json_schema()},
        },
    }
}


def _json_field(form, name: str, default=None):
    raw = form.get(name)
    if raw is None or raw == "":
        return default
    if isinstance(raw, UploadFile):
        raise ApiError(400, "INVALID_REQUEST", f"{name} 必须是 JSON 字段，不是文件")
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ApiError(400, "INVALID_REQUEST", f"{name} 不是合法 JSON") from exc


def _bool_field(form, name: str, default: bool = False) -> bool:
    raw = form.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


async def _read_upload(request: Request, settings: Settings):
    content_type = (request.headers.get("content-type") or "").lower()
    raw: bytes | None = None
    filename: str | None = None

    if content_type.startswith("application/json"):
        try:
            data = await request.json()
        except Exception as exc:
            raise ApiError(400, "INVALID_REQUEST", "请求体不是合法 JSON") from exc
        if not isinstance(data, dict):
            raise ApiError(400, "INVALID_REQUEST", "JSON 请求体必须是对象")
        payload = EvidenceUploadJson.model_validate(data)
    elif content_type.startswith("multipart/form-data"):
        form = await request.form()
        file = form.get("file")
        if isinstance(file, UploadFile):
            raw = await file.read()
            filename = file.filename
            await file.close()
            if len(raw) > settings.max_upload_bytes:
                raise ApiError(413, "PAYLOAD_TOO_LARGE",
                               f"文件超过 {settings.max_upload_bytes // (1024 * 1024)} MB 限制")
        content = _json_field(form, "content", {})
        metadata = _json_field(form, "metadata", {})
        if not isinstance(content, dict) or not isinstance(metadata, dict):
            raise ApiError(400, "INVALID_REQUEST", "content/metadata 必须是 JSON 对象")
        content = {**metadata, **content}
        data = {
            "case_id": form.get("case_id"),
            "modality": form.get("modality"),
            "source_type": form.get("source_type"),
            "source_name": form.get("source_name"),
            "evidence_id": form.get("evidence_id") or None,
            "text": form.get("text") or None,
            "content": content,
            "filename": filename,
            "gender_probs": _json_field(form, "gender_probs"),
            "age_probs": _json_field(form, "age_probs"),
            "identity_probs": _json_field(form, "identity_probs"),
            "time_range": _json_field(form, "time_range"),
            "language": form.get("language") or None,
            "synthetic": _bool_field(form, "synthetic"),
            "pii_redacted": _bool_field(form, "pii_redacted"),
            "injected": _bool_field(form, "injected"),
            "request_id": form.get("request_id") or request.headers.get("X-Request-ID"),
        }
        payload = EvidenceUploadJson.model_validate(data)
    else:
        raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE",
                       "请使用 multipart/form-data 或 application/json")

    if not payload.request_id:
        payload.request_id = request.headers.get("X-Request-ID")
    return payload, raw, filename


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED,
             summary="多模态证据接入",
             description="上传文本/图像/音频/行为/元数据证据；multipart 或 JSON。",
             openapi_extra=_UPLOAD_OPENAPI)
async def upload_evidence(
        request: Request,
        service: EvidenceService = Depends(get_evidence_service),
        settings: Settings = Depends(get_settings)):

    try:
        payload, raw, filename = await _read_upload(request, settings)
        return service.create(payload, raw=raw, filename=filename)
    except ValidationError as exc:
        details = [
            {"loc": [str(x) for x in item.get("loc", [])],
             "msg": item.get("msg", "invalid value"),
             "type": item.get("type", "value_error")}
            for item in exc.errors()
        ]
        raise ApiError(400, "INVALID_REQUEST", "证据字段校验失败", details) from exc
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(400, "INVALID_EVIDENCE", "证据无法解析",
                       {"reason": str(exc)}) from exc


@router.post("/evaluate", response_model=EvaluateResponse, summary="证据可信度评估")
def evaluate_evidence(
        body: EvaluateRequest,
        service: EvidenceService = Depends(get_evidence_service)):
    return service.evaluate(body.case_id, body.evidence_ids, body.overrides)


@router.post("/conflict", response_model=ConflictResponse, summary="证据冲突处理")
def handle_conflict(
        body: ConflictRequest,
        service: ConflictService = Depends(get_conflict_service)):
    return service.run(
        body.case_id, attribute=body.attribute, evidence_ids=body.evidence_ids,
        tau=body.tau, tau_gender=body.tau_gender, tau_age=body.tau_age,
        tau_identity=body.tau_identity, candidates=body.candidates,
        downweight=body.downweight, max_rounds=body.max_rounds,
        weight_power=body.weight_power, persist=body.persist,
    )


@router.get("/{evidence_id}", summary="查询单条证据")
def get_evidence(
        evidence_id: str,
        service: EvidenceService = Depends(get_evidence_service)):
    return service.get(evidence_id)
