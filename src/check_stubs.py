# -*- coding: utf-8 -*-
"""Self-check for the reserved audio/behavior interfaces."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parsers import parse_audio, parse_behavior


def main():
    a = parse_audio(
        case_id="SYN-001",
        audio_uri="minio://cases/SYN-001/call.wav",
        asr_text="你好，我今年二十四岁。",
        speaker_id="SPK-1",
        duration_sec=12.5,
        language="zh",
        time_range={"start": "2026-11-20T10:00:00", "end": "2026-11-20T10:00:12"},
    )
    b = parse_behavior(
        case_id="SYN-001",
        log_uri="minio://cases/SYN-001/behavior.json",
        events=[
            {"ts": "2026-11-19T21:00:00", "action": "login", "ip": "203.0.113.10", "device": "iPhone"},
            {"ts": "2026-11-19T21:10:00", "action": "post", "object": "note-1", "ip": "203.0.113.10", "device": "iPhone"},
        ],
        platform="平台A",
        subject_id="SYN-001",
    )

    assert a.modality == "audio" and a.content["status"] == "reserved"
    assert a.gender_probs is None and a.age_probs is None
    assert b.modality == "behavior" and b.content["status"] == "reserved"
    assert b.gender_probs is None and b.age_probs is None
    assert a.credibility.score > 0 and b.credibility.score > 0

    print("audio stub:", a.evidence_id, a.modality, a.content["status"], a.credibility.score)
    print("behavior stub:", b.evidence_id, b.modality, b.content["status"], b.credibility.score)
    print("stubs OK")


if __name__ == "__main__":
    main()
