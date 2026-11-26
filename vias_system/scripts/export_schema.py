# -*- coding: utf-8 -*-
"""Export the service's Pydantic EvidenceUnit schema to JSON Schema."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vias.core.schema import EvidenceUnit  # noqa: E402


def main():
    schema = EvidenceUnit.model_json_schema()
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/Koipoppy/VIAS/vias_system/schema/evidence_unit.schema.json",
        "title": "VIAS Evidence Unit Schema v1.1",
        "description": "统一证据单元 v1.1：属性层 + 候选身份层（identity_probs）。",
        **schema,
    }
    out = ROOT / "schema" / "evidence_unit.schema.json"
    out.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved:", out)
    print("properties:", len(schema["properties"]),
          "additionalProperties:", schema.get("additionalProperties"))


if __name__ == "__main__":
    main()
