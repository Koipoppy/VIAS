# -*- coding: utf-8 -*-
"""Local object storage for uploaded evidence.

The default backend is the filesystem.  `raw_ref` is a stable `local://`
URI so the same API contract can later be backed by MinIO/S3.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .errors import ApiError


class Storage:
    def __init__(self, root: str | Path, max_bytes: int):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(name: str, default: str) -> str:
        name = Path(name or default).name
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        return name or default

    def save(self, case_id: str, evidence_id: str, filename: str,
             data: bytes) -> dict:
        if len(data) > self.max_bytes:
            raise ApiError(413, "PAYLOAD_TOO_LARGE",
                           f"文件超过 {self.max_bytes // (1024 * 1024)} MB 限制",
                           {"bytes": len(data)})
        case_dir = self._safe(case_id, "case")
        evidence_dir = self._safe(evidence_id, "evidence")
        safe_name = self._safe(filename, "blob.bin")
        directory = self.root / case_dir / evidence_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / safe_name
        path.write_bytes(data)
        rel = path.relative_to(self.root).as_posix()
        return {
            "raw_ref": f"local://{rel}",
            "path": str(path),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "filename": safe_name,
        }
