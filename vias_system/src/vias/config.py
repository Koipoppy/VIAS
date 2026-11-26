# -*- coding: utf-8 -*-
"""Runtime configuration for the VIAS full system."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    app_name: str = "VIAS 虚实融合身份推断 API"
    version: str = "1.0.0"
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    db_path: Path | None = None
    storage_dir: Path | None = None
    text_model_dir: Path | None = None
    enable_face_model: bool = False
    face_device: str = "cpu"
    max_upload_bytes: int | None = None
    api_key: str = ""
    cors_origins: list[str] = field(default_factory=list)
    log_level: str = "INFO"

    def __post_init__(self):
        if self.db_path is None:
            self.db_path = Path(os.getenv("VIAS_DB_PATH", self.project_root / "var" / "vias.db"))
        if self.storage_dir is None:
            self.storage_dir = Path(os.getenv("VIAS_STORAGE_DIR", self.project_root / "var" / "uploads"))
        if self.text_model_dir is None:
            self.text_model_dir = Path(os.getenv("VIAS_TEXT_MODEL_DIR", self.project_root / "models"))
        self.db_path = Path(self.db_path)
        self.storage_dir = Path(self.storage_dir)
        self.text_model_dir = Path(self.text_model_dir)
        if not self.api_key:
            self.api_key = os.getenv("VIAS_API_KEY", "")
        if not self.cors_origins:
            raw = os.getenv("VIAS_CORS_ORIGINS", "")
            self.cors_origins = [x.strip() for x in raw.split(",") if x.strip()]
        self.enable_face_model = _env_bool("VIAS_ENABLE_FACE_MODEL", self.enable_face_model)
        self.face_device = os.getenv("VIAS_FACE_DEVICE", self.face_device)
        if self.max_upload_bytes is None:
            self.max_upload_bytes = int(os.getenv("VIAS_MAX_UPLOAD_MB", "25")) * 1024 * 1024
        self.max_upload_bytes = int(self.max_upload_bytes)
        self.log_level = os.getenv("VIAS_LOG_LEVEL", self.log_level).upper()

    def ensure_dirs(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
