# -*- coding: utf-8 -*-
"""Bayesian inference and explanation endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..schemas import (BayesRequest, ConflictResponse, ExplainRequest,
                       ExplainResponse, InferenceResponse)
from ..services.explain_service import ExplainService
from ..services.inference_service import InferenceService
from .deps import get_explain_service, get_inference_service

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.post("/bayes", response_model=InferenceResponse, summary="层次贝叶斯身份推理")
def bayes(body: BayesRequest,
          service: InferenceService = Depends(get_inference_service)):
    return service.run(body.case_id, attribute=body.attribute,
                       evidence_ids=body.evidence_ids, prior=body.prior,
                       weight_power=body.weight_power,
                       target=body.target, candidates=body.candidates,
                       context={k: v for k, v in {
                           "account_a": body.account_a,
                           "account_b": body.account_b,
                           "platform": body.platform,
                       }.items() if v})


@router.post("/explain", response_model=ExplainResponse, summary="可解释性溯源查询")
def explain(body: ExplainRequest,
            service: ExplainService = Depends(get_explain_service)):
    return service.run(inference_id=body.inference_id, case_id=body.case_id,
                       attribute=body.attribute, top_k=body.top_k)
