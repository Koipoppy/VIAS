# -*- coding: utf-8 -*-
"""FastAPI dependencies."""
from fastapi import Request

from ..config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_repo(request: Request):
    return request.app.state.repo


def get_evidence_service(request: Request):
    return request.app.state.evidence_service


def get_inference_service(request: Request):
    return request.app.state.inference_service


def get_identity_service(request: Request):
    return request.app.state.identity_service


def get_conflict_service(request: Request):
    return request.app.state.conflict_service


def get_explain_service(request: Request):
    return request.app.state.explain_service
