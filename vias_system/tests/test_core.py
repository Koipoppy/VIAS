# -*- coding: utf-8 -*-
"""Core contract and algorithm tests (stdlib unittest)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vias.core.credibility import score_evidence  # noqa: E402
from vias.core.fusion import conflict_handle, conflict_pair, weighted_fuse  # noqa: E402
from vias.core.schema import AGE_BUCKETS, Credibility, EvidenceUnit  # noqa: E402
from vias.config import Settings  # noqa: E402
from vias.core.text_parser import TextParser  # noqa: E402
from vias.services.evidence_service import sanitize_text  # noqa: E402


def cred(weight=0.7):
    return Credibility(
        source_reliability=weight, authority=weight, objectivity=weight,
        timeliness=weight, completeness=weight, consistency=weight,
        score=weight, weight=weight,
    )


def unit(eid, *, gender=None, age=None, weight=0.7, source_type="text_model"):
    return EvidenceUnit(
        schema_version="1.0", evidence_id=eid, case_id="CASE-CORE",
        modality="text", source_type=source_type, source_name="test",
        content={"snippet": "x"}, gender_probs=gender, age_probs=age,
        credibility=cred(weight),
    )


class SchemaTests(unittest.TestCase):
    def test_json_schema_contract(self):
        schema = json.loads((ROOT / "schema/evidence_unit.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(len(schema["properties"]), 23)
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(
            set(schema["properties"]["source_type"]["enum"]),
            {"text_model", "face_model", "self_report", "audio_stub",
             "behavior_stub", "synthetic_conflict", "metadata"},
        )

    def test_example_valid(self):
        ex = json.loads((ROOT / "schema/evidence_unit.example.json").read_text(encoding="utf-8"))
        self.assertEqual(EvidenceUnit.model_validate(ex).schema_version, "1.1")

    def test_probability_sum_rejected(self):
        ex = json.loads((ROOT / "schema/evidence_unit.example.json").read_text(encoding="utf-8"))
        ex["age_probs"] = {b: 0.1 for b in AGE_BUCKETS}
        with self.assertRaises(ValueError):
            EvidenceUnit.model_validate(ex)

    def test_extra_field_rejected(self):
        ex = json.loads((ROOT / "schema/evidence_unit.example.json").read_text(encoding="utf-8"))
        ex["effective_weight"] = 0.5
        with self.assertRaises(ValueError):
            EvidenceUnit.model_validate(ex)


class CredibilityTests(unittest.TestCase):
    def test_ranges(self):
        c = score_evidence("text_model", n_chars=5000)
        self.assertGreaterEqual(c.score, 0.0)
        self.assertLessEqual(c.score, 1.0)
        self.assertAlmostEqual(c.score, c.weight)

    def test_stub_profile(self):
        c = score_evidence("audio_stub")
        self.assertLess(c.score, 0.5)

    def test_pii_redaction(self):
        text, changed = sanitize_text(
            "联系我 test@example.com 或 13800138000，见 https://x.test/a"
        )
        self.assertTrue(changed)
        for marker in ("[EMAIL]", "[PHONE]", "[URL]"):
            self.assertIn(marker, text)


class FusionTests(unittest.TestCase):
    def test_posterior_normalized(self):
        ev = [
            unit("e1", gender={"male": 0.9, "female": 0.1}),
            unit("e2", gender={"male": 0.1, "female": 0.9}),
        ]
        post, used = weighted_fuse({"male": 0.5, "female": 0.5}, ev, "gender")
        self.assertAlmostEqual(sum(post.values()), 1.0, places=9)
        self.assertEqual(len(used), 2)

    def test_weight_power_increases_high_weight_influence(self):
        high = unit("high", gender={"male": 0.9, "female": 0.1}, weight=0.9)
        low = unit("low", gender={"male": 0.1, "female": 0.9}, weight=0.2)
        p1, _ = weighted_fuse({"male": 0.5, "female": 0.5}, [high, low], "gender",
                              weight_power=1.0)
        p2, _ = weighted_fuse({"male": 0.5, "female": 0.5}, [high, low], "gender",
                              weight_power=2.0)
        self.assertGreater(p2["male"], p1["male"])

    def test_conflict_pair_identical_zero(self):
        p = {"male": 0.8, "female": 0.2}
        self.assertAlmostEqual(conflict_pair(p, p)["conflict"], 0.0, places=9)

    def test_conflict_handler_downweights(self):
        ev = [
            unit("a", gender={"male": 0.9, "female": 0.1}, weight=0.8),
            unit("b", gender={"male": 0.1, "female": 0.9}, weight=0.4),
        ]
        post, log, adjusted, mx, pair, used = conflict_handle(
            ev, "gender", tau=0.50, weight_power=1.5
        )
        self.assertGreater(mx, 0.5)
        self.assertTrue(log)
        self.assertEqual(len(adjusted), 2)
        self.assertTrue(any(e.conflict_flag for e in adjusted))
        self.assertAlmostEqual(sum(post.values()), 1.0, places=9)


class ConfigTests(unittest.TestCase):
    def test_default_paths(self):
        s = Settings()
        self.assertEqual(s.project_root, ROOT)
        self.assertEqual(s.text_model_dir, ROOT / "models")
        self.assertEqual(s.db_path, ROOT / "var" / "vias.db")


class TextParserTests(unittest.TestCase):
    def test_model_probabilities(self):
        parser = TextParser(ROOT / "models")
        gp, ap, content = parser.parse("我今年28岁，是一名女生。", "text_model")
        self.assertIsNotNone(gp)
        self.assertIsNotNone(ap)
        self.assertAlmostEqual(sum(gp.values()), 1.0, places=6)
        self.assertAlmostEqual(sum(ap.values()), 1.0, places=6)
        self.assertIn("model", content)

    def test_self_report(self):
        parser = TextParser(ROOT / "models")
        gp, ap, content = parser.parse("我今年24岁，是男生。", "self_report")
        self.assertIsNotNone(gp)
        self.assertEqual(max(gp, key=gp.get), "male")
        self.assertEqual(max(ap, key=ap.get), "18-24")


if __name__ == "__main__":
    unittest.main(verbosity=2)
