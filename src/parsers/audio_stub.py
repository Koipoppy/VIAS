# -*- coding: utf-8 -*-
"""Reserved audio interface (2026-11).

Expected input
--------------
- audio_uri: path to WAV/MP3, or a MinIO URI
- asr_text: transcript produced by an ASR model (Whisper/AISHELL)
- speaker_id / duration_sec / language: optional metadata

Expected output
---------------
An EvidenceUnit with modality="audio".  In the full route, the ASR text
feeds the text evidence parser and a voice model can supply
gender_probs / age_probs.  In the pilot these are intentionally None:
the interface is reserved, not implemented.

This module never raises: it returns a placeholder EvidenceUnit whose
content.status == "reserved", so an upstream pipeline can call it safely.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credibility import score_evidence
from schema import EvidenceUnit


def parse_audio(
    *,
    case_id: str,
    evidence_id: str | None = None,
    audio_uri: str | None = None,
    asr_text: str | None = None,
    speaker_id: str | None = None,
    duration_sec: float | None = None,
    language: str | None = None,
    time_range: dict | None = None,
) -> EvidenceUnit:
    eid = evidence_id or f"{case_id}-A-STUB"
    return EvidenceUnit(
        schema_version="0.1",
        evidence_id=eid,
        case_id=case_id,
        modality="audio",
        source_type="audio_stub",
        source_name="reserved audio parser",
        content={
            "status": "reserved",
            "audio_uri": audio_uri,
            "asr_text": asr_text,
            "speaker_id": speaker_id,
            "duration_sec": duration_sec,
            "note": "ASR + voice gender/age model reserved for 2026-11; not used in pilot.",
        },
        raw_ref=audio_uri,
        time_range=time_range,
        language=language,
        gender_probs=None,
        age_probs=None,
        credibility=score_evidence("stub"),
        note="reserved interface; no model output",
    )
