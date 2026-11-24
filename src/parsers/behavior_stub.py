# -*- coding: utf-8 -*-
"""Reserved behavior-log interface (2026-11).

Expected input
--------------
- log_uri: CSV/JSON file or MinIO URI
- events: list of dicts, each with:
    timestamp, action (login/post/interact/transfer), object, ip, device
- platform / subject_id: optional metadata

Expected output
---------------
An EvidenceUnit with modality="behavior".  In the full route the event
sequence feeds a sequence model; in the pilot this is intentionally a
placeholder with no gender_probs / age_probs.

This module never raises: content.status == "reserved".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credibility import score_evidence
from schema import EvidenceUnit


def parse_behavior(
    *,
    case_id: str,
    evidence_id: str | None = None,
    log_uri: str | None = None,
    events: list | None = None,
    platform: str | None = None,
    subject_id: str | None = None,
    time_range: dict | None = None,
) -> EvidenceUnit:
    eid = evidence_id or f"{case_id}-B-STUB"
    return EvidenceUnit(
        schema_version="0.1",
        evidence_id=eid,
        case_id=case_id,
        modality="behavior",
        source_type="behavior_stub",
        source_name="reserved behavior-log parser",
        content={
            "status": "reserved",
            "log_uri": log_uri,
            "events": events or [],
            "platform": platform,
            "subject_id": subject_id,
            "note": "Login/post/interact/transfer event model reserved for 2026-11; not used in pilot.",
        },
        raw_ref=log_uri,
        time_range=time_range,
        language=None,
        gender_probs=None,
        age_probs=None,
        credibility=score_evidence("stub"),
        note="reserved interface; no model output",
    )
