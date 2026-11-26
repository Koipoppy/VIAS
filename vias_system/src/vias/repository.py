# -*- coding: utf-8 -*-
"""Repository layer: JSON payloads with indexed case/evidence identifiers."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Iterable

from .core.schema import EvidenceUnit
from .db import Database
from .errors import ApiError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str):
    return json.loads(value)


class Repository:
    def __init__(self, db: Database):
        self.db = db

    # ---------------------------------------------------------------- cases
    def ensure_case(self, case_id: str, metadata: dict | None = None) -> dict:
        now = utc_now()
        with self.db.connect() as conn:
            row = conn.execute("SELECT payload FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if row:
                return _loads(row["payload"])
            payload = {"case_id": case_id, "created_at": now,
                       "metadata": metadata or {}}
            conn.execute(
                "INSERT INTO cases(case_id, created_at, payload) VALUES (?,?,?)",
                (case_id, now, _dumps(payload)),
            )
            return payload

    def get_case(self, case_id: str) -> dict | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT payload FROM cases WHERE case_id=?", (case_id,)).fetchone()
            return _loads(row["payload"]) if row else None

    def case_summary(self, case_id: str) -> dict | None:
        case = self.get_case(case_id)
        if not case:
            return None
        with self.db.connect() as conn:
            ev = conn.execute(
                "SELECT COUNT(*) AS n FROM evidence WHERE case_id=?", (case_id,)
            ).fetchone()["n"]
            inf = conn.execute(
                "SELECT inference_id, created_at, payload FROM inferences "
                "WHERE case_id=? ORDER BY created_at DESC LIMIT 1", (case_id,)
            ).fetchone()
            con = conn.execute(
                "SELECT conflict_id, created_at FROM conflicts "
                "WHERE case_id=? ORDER BY created_at DESC LIMIT 1", (case_id,)
            ).fetchone()
            exp = conn.execute(
                "SELECT explanation_id, created_at FROM explanations "
                "WHERE case_id=? ORDER BY created_at DESC LIMIT 1", (case_id,)
            ).fetchone()
        out = dict(case)
        out.update({
            "evidence_count": ev,
            "latest_inference_id": inf["inference_id"] if inf else None,
            "latest_conflict_id": con["conflict_id"] if con else None,
            "latest_explanation_id": exp["explanation_id"] if exp else None,
        })
        return out

    # ------------------------------------------------------------- evidence
    def insert_evidence(self, unit: EvidenceUnit) -> None:
        now = utc_now()
        payload = json.loads(unit.model_dump_json())
        try:
            with self.db.connect() as conn:
                conn.execute(
                    "INSERT INTO evidence(evidence_id, case_id, modality, source_type, "
                    "schema_version, created_at, payload) VALUES (?,?,?,?,?,?,?)",
                    (unit.evidence_id, unit.case_id, unit.modality, unit.source_type,
                     unit.schema_version, now, _dumps(payload)),
                )
        except sqlite3.IntegrityError as exc:
            raise ApiError(409, "EVIDENCE_EXISTS",
                           f"证据已存在：{unit.evidence_id}",
                           {"evidence_id": unit.evidence_id}) from exc

    def get_evidence(self, evidence_id: str) -> EvidenceUnit | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM evidence WHERE evidence_id=?", (evidence_id,)
            ).fetchone()
        if not row:
            return None
        return EvidenceUnit.model_validate(_loads(row["payload"]))

    def evidence_created_at(self, evidence_id: str) -> str | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT created_at FROM evidence WHERE evidence_id=?", (evidence_id,)
            ).fetchone()
        return row["created_at"] if row else None

    def list_evidence(self, case_id: str, evidence_ids: Iterable[str] | None = None) -> list[EvidenceUnit]:
        query = "SELECT payload FROM evidence WHERE case_id=?"
        params: list = [case_id]
        ids = list(evidence_ids) if evidence_ids is not None else []
        if ids:
            query += " AND evidence_id IN (%s)" % ",".join("?" for _ in ids)
            params.extend(ids)
        query += " ORDER BY created_at ASC, evidence_id ASC"
        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [EvidenceUnit.model_validate(_loads(r["payload"])) for r in rows]

    def update_evidence(self, unit: EvidenceUnit) -> None:
        payload = json.loads(unit.model_dump_json())
        with self.db.connect() as conn:
            cur = conn.execute(
                "UPDATE evidence SET payload=? WHERE evidence_id=?",
                (_dumps(payload), unit.evidence_id),
            )
            if cur.rowcount == 0:
                raise ApiError(404, "EVIDENCE_NOT_FOUND",
                               f"证据不存在：{unit.evidence_id}")

    def delete_evidence(self, evidence_id: str) -> bool:
        with self.db.connect() as conn:
            cur = conn.execute("DELETE FROM evidence WHERE evidence_id=?", (evidence_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------ inferences
    def save_inference(self, case_id: str, payload: dict) -> str:
        inference_id = payload.get("inference_id") or f"inf_{uuid.uuid4().hex[:16]}"
        payload["inference_id"] = inference_id
        now = utc_now()
        payload.setdefault("created_at", now)
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO inferences(inference_id, case_id, created_at, payload) "
                "VALUES (?,?,?,?)",
                (inference_id, case_id, now, _dumps(payload)),
            )
        return inference_id

    def get_inference(self, inference_id: str) -> dict | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM inferences WHERE inference_id=?", (inference_id,)
            ).fetchone()
        return _loads(row["payload"]) if row else None

    def list_inferences(self, case_id: str) -> list[dict]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM inferences WHERE case_id=? ORDER BY created_at DESC",
                (case_id,),
            ).fetchall()
        return [_loads(r["payload"]) for r in rows]

    # ------------------------------------------------------------- conflicts
    def save_conflict(self, case_id: str, payload: dict) -> str:
        conflict_id = payload.get("conflict_id") or f"con_{uuid.uuid4().hex[:16]}"
        payload["conflict_id"] = conflict_id
        now = utc_now()
        payload.setdefault("created_at", now)
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO conflicts(conflict_id, case_id, created_at, payload) "
                "VALUES (?,?,?,?)",
                (conflict_id, case_id, now, _dumps(payload)),
            )
        return conflict_id

    # ---------------------------------------------------------- explanations
    def save_explanation(self, case_id: str, inference_id: str | None, payload: dict) -> str:
        explanation_id = payload.get("explanation_id") or f"exp_{uuid.uuid4().hex[:16]}"
        payload["explanation_id"] = explanation_id
        now = utc_now()
        payload.setdefault("created_at", now)
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO explanations(explanation_id, case_id, inference_id, "
                "created_at, payload) VALUES (?,?,?,?,?)",
                (explanation_id, case_id, inference_id, now, _dumps(payload)),
            )
        return explanation_id

    def get_explanation(self, explanation_id: str) -> dict | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM explanations WHERE explanation_id=?",
                (explanation_id,),
            ).fetchone()
        return _loads(row["payload"]) if row else None

    # ----------------------------------------------------------------- audit
    def audit(self, action: str, case_id: str | None = None,
              request_id: str | None = None, payload: dict | None = None):
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO audits(request_id, action, case_id, created_at, payload) "
                "VALUES (?,?,?,?,?)",
                (request_id, action, case_id, utc_now(), _dumps(payload or {})),
            )
