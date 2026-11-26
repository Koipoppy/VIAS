# -*- coding: utf-8 -*-
"""VIAS full-system FastAPI application."""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import evidence, health, inference
from .config import Settings
from .db import Database
from .errors import error_body, install_error_handlers
from .repository import Repository
from .services.conflict_service import ConflictService
from .services.evidence_service import EvidenceService
from .services.explain_service import ExplainService
from .services.identity_service import IdentityService
from .services.inference_service import InferenceService
from .storage import Storage
from .core.face_parser import FaceParser
from .core.text_parser import TextParser

logger = logging.getLogger("vias")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))

    db = Database(settings.db_path)
    db.init()
    repo = Repository(db)
    storage = Storage(settings.storage_dir, settings.max_upload_bytes)
    text_parser = TextParser(settings.text_model_dir)
    face_parser = FaceParser(settings.enable_face_model, settings.face_device)

    evidence_service = EvidenceService(settings, repo, storage, text_parser, face_parser)
    identity_service = IdentityService(repo)
    inference_service = InferenceService(repo, identity_service)
    conflict_service = ConflictService(repo, identity_service)
    explain_service = ExplainService(repo, inference_service)

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "VIAS 虚实融合身份推断完全体 API：证据接入、可信度评估、"
            "层次贝叶斯推断、冲突处理、可解释溯源。"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    app.state.db = db
    app.state.repo = repo
    app.state.storage = storage
    app.state.text_parser = text_parser
    app.state.face_parser = face_parser
    app.state.evidence_service = evidence_service
    app.state.identity_service = identity_service
    app.state.inference_service = inference_service
    app.state.conflict_service = conflict_service
    app.state.explain_service = explain_service

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:16]}"
        request.state.request_id = request_id
        started = time.perf_counter()

        # Optional API key: health/docs stay public.
        public = {"/", "/docs", "/redoc", "/openapi.json", "/api/health"}
        if settings.api_key and request.url.path.startswith("/api") and request.url.path not in public:
            if request.headers.get("X-API-Key") != settings.api_key:
                return JSONResponse(
                    status_code=401,
                    content=error_body("AUTH_INVALID", "缺少或错误的 X-API-Key",
                                       None, request_id),
                )

        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_upload_bytes + 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content=error_body("PAYLOAD_TOO_LARGE", "请求体超过限制",
                                   {"max_bytes": settings.max_upload_bytes}, request_id),
            )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        return response

    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(evidence.router)
    app.include_router(inference.router)

    @app.get("/", include_in_schema=False)
    def root():
        return {
            "service": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/api/health",
            "endpoints": [
                "POST /api/evidence/upload",
                "POST /api/evidence/evaluate",
                "POST /api/inference/bayes",
                "POST /api/evidence/conflict",
                "POST /api/inference/explain",
            ],
        }

    return app


app = create_app()
