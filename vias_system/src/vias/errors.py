# -*- coding: utf-8 -*-
"""Structured API errors and exception handlers."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("vias.api")


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str,
                 details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def error_body(code: str, message: str, details: Any = None,
               request_id: str | None = None) -> dict:
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    if request_id:
        body["error"]["request_id"] = request_id
    return body


def install_error_handlers(app: FastAPI):
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, exc.details,
                               getattr(request.state, "request_id", None)),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        details = []
        for item in exc.errors():
            details.append({
                "loc": [str(x) for x in item.get("loc", [])],
                "msg": item.get("msg", "invalid value"),
                "type": item.get("type", "value_error"),
            })
        return JSONResponse(
            status_code=400,
            content=error_body("INVALID_REQUEST", "请求参数校验失败", details,
                               getattr(request.state, "request_id", None)),
        )

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            body = error_body(detail.get("code", "HTTP_ERROR"),
                              detail.get("message", str(detail)),
                              detail.get("details"),
                              getattr(request.state, "request_id", None))
        else:
            body = error_body("HTTP_ERROR", str(detail),
                              None, getattr(request.state, "request_id", None))
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error request_id=%s", getattr(request.state, "request_id", "-"))
        return JSONResponse(
            status_code=500,
            content=error_body("INTERNAL_ERROR", "服务内部错误",
                               None, getattr(request.state, "request_id", None)),
        )
