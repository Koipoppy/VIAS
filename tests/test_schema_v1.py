# -*- coding: utf-8 -*-
"""Schema v1.0 contract tests (stdlib unittest; pytest-compatible)."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schema import EvidenceUnit  # noqa: E402


def load_example():
    return json.loads((ROOT / "schema/evidence_unit.example.json").read_text(encoding="utf-8"))


class SchemaV1Tests(unittest.TestCase):
    def test_example_valid_and_version_default(self):
        e = EvidenceUnit.model_validate(load_example())
        self.assertEqual(e.schema_version, "1.0")
        self.assertAlmostEqual(
            e.effective_weight(), e.credibility.weight * e.weight_multiplier
        )

    def test_legacy_version_accepted(self):
        ex = load_example()
        ex["schema_version"] = "0.1"
        self.assertEqual(EvidenceUnit.model_validate(ex).schema_version, "0.1")

    def test_gender_probs_must_sum_to_one(self):
        ex = load_example()
        ex["gender_probs"] = {"male": 0.8, "female": 0.3}
        with self.assertRaises(ValueError):
            EvidenceUnit.model_validate(ex)

    def test_gender_probs_keys_exact(self):
        ex = load_example()
        ex["gender_probs"] = {"male": 0.5, "female": 0.5, "other": 0.0}
        with self.assertRaises(ValueError):
            EvidenceUnit.model_validate(ex)

    def test_age_probs_keys_exact(self):
        ex = load_example()
        ex["age_probs"] = {"18-24": 0.5, "25-29": 0.5}
        with self.assertRaises(ValueError):
            EvidenceUnit.model_validate(ex)

    def test_credibility_range(self):
        ex = load_example()
        ex["credibility"]["score"] = 1.2
        with self.assertRaises(ValueError):
            EvidenceUnit.model_validate(ex)

    def test_extra_field_forbidden(self):
        ex = load_example()
        ex["effective_weight"] = 0.751
        with self.assertRaises(ValueError):
            EvidenceUnit.model_validate(ex)

    def test_source_type_enum(self):
        ex = load_example()
        ex["source_type"] = "unknown_source"
        with self.assertRaises(ValueError):
            EvidenceUnit.model_validate(ex)

    def test_time_range_extra_forbidden(self):
        ex = load_example()
        ex["time_range"]["tz"] = "Asia/Shanghai"
        with self.assertRaises(ValueError):
            EvidenceUnit.model_validate(ex)

    def test_json_schema_contract(self):
        schema = json.loads(
            (ROOT / "schema/evidence_unit.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(schema["properties"]), 22)
        self.assertIs(schema.get("additionalProperties"), False)
        self.assertEqual(
            set(schema["properties"]["source_type"]["enum"]),
            {
                "text_model", "face_model", "self_report", "audio_stub",
                "behavior_stub", "synthetic_conflict", "metadata",
            },
        )
        self.assertIn("1.0", schema["properties"]["schema_version"]["enum"])
        self.assertIs(
            schema["$defs"]["Credibility"].get("additionalProperties"), False
        )

    def test_all_evidence_valid(self):
        paths = [
            ROOT / "outputs/evidence_units_clean.jsonl",
            ROOT / "outputs/evidence_units_injected.jsonl",
        ]
        paths += sorted((ROOT / "cases_synthetic").glob("SYN-*/evidence_units.jsonl"))
        total = 0
        for path in paths:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                EvidenceUnit.model_validate(json.loads(line))
                total += 1
        self.assertGreaterEqual(total, 442)


if __name__ == "__main__":
    unittest.main(verbosity=2)
