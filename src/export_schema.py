# -*- coding: utf-8 -*-
"""Export the Pydantic EvidenceUnit model to JSON Schema v1.0."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema import EvidenceUnit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "schema", "evidence_unit.schema.json")


def main():
    schema = EvidenceUnit.model_json_schema()
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/Koipoppy/VIAS/schema/evidence_unit.schema.json",
        "title": "VIAS Evidence Unit Schema v1.0",
        "description": "统一证据单元：多模态解析、可信度、融合、冲突与可解释模块之间的数据契约。",
        **schema,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("saved:", OUT)
    print("properties:", len(schema["properties"]),
          "additionalProperties:", schema.get("additionalProperties"))


if __name__ == "__main__":
    main()
