# -*- coding: utf-8 -*-
"""Health, case and record query endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..config import Settings
from ..errors import ApiError
from ..repository import Repository
from ..services.evidence_service import EvidenceService
from .deps import get_evidence_service, get_repo, get_settings

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health(settings: Settings = Depends(get_settings),
           repo: Repository = Depends(get_repo)):
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
        "schema_version": "1.0",
        "storage": "local",
        "face_model_enabled": settings.enable_face_model,
    }


@router.get("/cases/{case_id}")
def get_case(case_id: str, repo: Repository = Depends(get_repo)):
    summary = repo.case_summary(case_id)
    if not summary:
        raise ApiError(404, "CASE_NOT_FOUND", f"案件不存在：{case_id}")
    return summary


@router.get("/cases/{case_id}/evidence")
def list_case_evidence(case_id: str,
                       service: EvidenceService = Depends(get_evidence_service)):
    return service.list(case_id)


@router.get("/inferences/{inference_id}")
def get_inference(inference_id: str, repo: Repository = Depends(get_repo)):
    payload = repo.get_inference(inference_id)
    if not payload:
        raise ApiError(404, "INFERENCE_NOT_FOUND",
                       f"推断记录不存在：{inference_id}")
    return payload


@router.get("/explanations/{explanation_id}")
def get_explanation(explanation_id: str, repo: Repository = Depends(get_repo)):
    payload = repo.get_explanation(explanation_id)
    if not payload:
        raise ApiError(404, "EXPLANATION_NOT_FOUND",
                       f"解释记录不存在：{explanation_id}")
    return payload
