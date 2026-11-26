# -*- coding: utf-8 -*-
"""End-to-end API tests (stdlib unittest; runnable without pytest)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vias.config import Settings  # noqa: E402
from vias.main import create_app  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models"


def make_client(tmp: str, **kwargs) -> TestClient:
    settings = Settings(
        db_path=Path(tmp) / "vias.db",
        storage_dir=Path(tmp) / "uploads",
        text_model_dir=MODELS,
        **kwargs,
    )
    return TestClient(create_app(settings), raise_server_exceptions=False)


def text_payload(case_id: str, male: float, female: float, age_bucket: str = "25-29"):
    age = {b: 0.05 for b in ["18-24", "25-29", "30-34", "35-40"]}
    age[age_bucket] = 0.85
    age = {k: v / sum(age.values()) for k, v in age.items()}
    return {
        "case_id": case_id,
        "modality": "text",
        "source_type": "text_model",
        "source_name": "unit-test-model",
        "text": f"这是 {case_id} 的测试文本。",
        "gender_probs": {"male": male, "female": female},
        "age_probs": age,
    }


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.client = make_client(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health_and_openapi(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["schema_version"], "1.0")
        r = self.client.get("/openapi.json")
        self.assertEqual(r.status_code, 200)
        paths = r.json()["paths"]
        for path in [
            "/api/evidence/upload", "/api/evidence/evaluate",
            "/api/inference/bayes", "/api/evidence/conflict",
            "/api/inference/explain",
        ]:
            self.assertIn(path, paths)

    def test_full_flow(self):
        cid = "CASE-FLOW"
        a = text_payload(cid, 0.9, 0.1, "18-24")
        b = text_payload(cid, 0.1, 0.9, "30-34")
        for payload in (a, b):
            r = self.client.post("/api/evidence/upload", json=payload)
            self.assertEqual(r.status_code, 201, r.text)
            self.assertEqual(r.json()["status"], "ready")

        r = self.client.post("/api/evidence/evaluate", json={"case_id": cid})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()["evaluated"]), 2)

        r = self.client.post("/api/inference/bayes", json={"case_id": cid, "attribute": "both"})
        self.assertEqual(r.status_code, 200, r.text)
        bayes = r.json()
        self.assertEqual(bayes["weight_power"], 1.5)
        self.assertIn("gender", bayes["results"])
        self.assertIn("age", bayes["results"])
        inference_id = bayes["inference_id"]

        r = self.client.post("/api/evidence/conflict", json={"case_id": cid, "attribute": "both"})
        self.assertEqual(r.status_code, 200, r.text)
        conflict = r.json()
        self.assertEqual(conflict["taus"], {"gender": 0.65, "age": 0.50})
        self.assertEqual(conflict["weight_power"], 1.5)
        self.assertTrue(conflict["attributes"]["gender"]["detected"])
        self.assertTrue(conflict["downweighted"])

        r = self.client.post("/api/inference/explain", json={"inference_id": inference_id, "top_k": 2})
        self.assertEqual(r.status_code, 200, r.text)
        explain = r.json()
        self.assertIn("gender", explain["results"])
        self.assertTrue(explain["results"]["gender"]["narrative"])
        self.assertTrue(explain["results"]["gender"]["top_evidence"])

    def test_multipart_upload_auto_text_model(self):
        cid = "CASE-UPLOAD"
        r = self.client.post(
            "/api/evidence/upload",
            data={"case_id": cid, "modality": "text", "source_type": "text_model",
                  "source_name": "upload", "text": "我今年28岁，是一名女生。"},
            files={"file": ("note.txt", b"raw content", "text/plain")},
        )
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual(body["status"], "ready")
        self.assertIsNotNone(body["probabilities"]["gender"])
        self.assertTrue(body["raw_ref"].startswith("local://"))

    def test_invalid_probabilities(self):
        payload = text_payload("CASE-BAD", 0.8, 0.3)
        r = self.client.post("/api/evidence/upload", json=payload)
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(r.json()["error"]["code"], "INVALID_PROBABILITIES")

    def test_source_modality_mismatch(self):
        payload = text_payload("CASE-MISMATCH", 0.5, 0.5)
        payload["modality"] = "image"
        r = self.client.post("/api/evidence/upload", json=payload)
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(r.json()["error"]["code"], "SOURCE_MODALITY_MISMATCH")

    def test_no_probability_evidence(self):
        cid = "CASE-IMG"
        r = self.client.post(
            "/api/evidence/upload",
            data={"case_id": cid, "modality": "image", "source_type": "face_model",
                  "source_name": "face"},
            files={"file": ("face.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        )
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json()["status"], "stored_no_probabilities")
        r = self.client.post("/api/inference/bayes", json={"case_id": cid})
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(r.json()["error"]["code"], "NO_PROBABILITY_EVIDENCE")

    def test_api_key(self):
        with tempfile.TemporaryDirectory() as td:
            client = make_client(td, api_key="secret")
            self.assertEqual(client.get("/api/health").status_code, 200)
            r = client.post("/api/evidence/upload", json=text_payload("CASE-AUTH", 0.5, 0.5))
            self.assertEqual(r.status_code, 401)
            r = client.post("/api/evidence/upload", json=text_payload("CASE-AUTH", 0.5, 0.5),
                            headers={"X-API-Key": "secret"})
            self.assertEqual(r.status_code, 201, r.text)


    def test_unknown_field_rejected(self):
        payload = text_payload("CASE-UNKNOWN", 0.5, 0.5)
        payload["bogus_field"] = "x"
        r = self.client.post("/api/evidence/upload", json=payload)
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(r.json()["error"]["code"], "INVALID_REQUEST")

    def test_query_records(self):
        cid = "CASE-QUERY"
        r = self.client.post("/api/evidence/upload", json=text_payload(cid, 0.7, 0.3))
        self.assertEqual(r.status_code, 201, r.text)
        eid = r.json()["evidence_id"]
        self.assertEqual(self.client.get(f"/api/cases/{cid}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/cases/{cid}/evidence").status_code, 200)
        self.assertEqual(self.client.get(f"/api/evidence/{eid}").status_code, 200)
        r = self.client.post("/api/inference/bayes", json={"case_id": cid})
        self.assertEqual(r.status_code, 200, r.text)
        iid = r.json()["inference_id"]
        self.assertEqual(self.client.get(f"/api/inferences/{iid}").status_code, 200)
        r = self.client.post("/api/inference/explain", json={"inference_id": iid})
        self.assertEqual(r.status_code, 200, r.text)
        xid = r.json()["explanation_id"]
        self.assertEqual(self.client.get(f"/api/explanations/{xid}").status_code, 200)


    def test_duplicate_evidence(self):
        cid = "CASE-DUP"
        payload = text_payload(cid, 0.7, 0.3)
        payload["evidence_id"] = "ev_dup_fixed"
        self.assertEqual(self.client.post("/api/evidence/upload", json=payload).status_code, 201)
        r = self.client.post("/api/evidence/upload", json=payload)
        self.assertEqual(r.status_code, 409, r.text)
        self.assertEqual(r.json()["error"]["code"], "EVIDENCE_EXISTS")

    def test_evaluate_overrides(self):
        cid = "CASE-OVERRIDE"
        r = self.client.post("/api/evidence/upload", json=text_payload(cid, 0.6, 0.4))
        self.assertEqual(r.status_code, 201, r.text)
        eid = r.json()["evidence_id"]
        r = self.client.post("/api/evidence/evaluate", json={
            "case_id": cid, "evidence_ids": [eid],
            "overrides": {eid: {"authority": 0.9}},
        })
        self.assertEqual(r.status_code, 200, r.text)
        self.assertAlmostEqual(r.json()["evaluated"][0]["credibility"]["authority"], 0.9)
        r = self.client.post("/api/evidence/evaluate", json={
            "case_id": cid, "overrides": {eid: {"bogus": 0.9}},
        })
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(r.json()["error"]["code"], "INVALID_OVERRIDE")

    def test_invalid_prior(self):
        cid = "CASE-PRIOR"
        self.assertEqual(self.client.post("/api/evidence/upload", json=text_payload(cid, 0.6, 0.4)).status_code, 201)
        r = self.client.post("/api/inference/bayes", json={
            "case_id": cid, "attribute": "gender", "prior": {"male": 0.7},
        })
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(r.json()["error"]["code"], "INVALID_PRIOR")

    def test_conflict_persist_false(self):
        cid = "CASE-NOPERSIST"
        ids = []
        for male, female in [(0.9, 0.1), (0.1, 0.9)]:
            r = self.client.post("/api/evidence/upload", json=text_payload(cid, male, female))
            self.assertEqual(r.status_code, 201, r.text)
            ids.append(r.json()["evidence_id"])
        before = [self.client.app.state.repo.get_evidence(i).weight_multiplier for i in ids]
        r = self.client.post("/api/evidence/conflict", json={
            "case_id": cid, "attribute": "gender", "persist": False,
        })
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse(r.json()["persisted"])
        after = [self.client.app.state.repo.get_evidence(i).weight_multiplier for i in ids]
        self.assertEqual(before, after)

    def test_json_base64_and_pii(self):
        cid = "CASE-BASE64"
        import base64
        payload = text_payload(cid, 0.7, 0.3)
        payload["text"] = "联系 test@example.com 或见 https://x.test/a"
        payload["filename"] = "note.txt"
        payload["file_base64"] = base64.b64encode(b"raw").decode()
        r = self.client.post("/api/evidence/upload", json=payload)
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertTrue(body["raw_ref"].startswith("local://"))
        self.assertIn("[EMAIL]", body["content_snippet"])
        self.assertIn("[URL]", body["content_snippet"])

    def test_request_id_and_404(self):
        r = self.client.get("/api/evidence/does-not-exist")
        self.assertEqual(r.status_code, 404)
        self.assertTrue(r.headers.get("X-Request-ID"))
        self.assertEqual(r.json()["error"]["code"], "EVIDENCE_NOT_FOUND")


    def test_invalid_time_range(self):
        cid = "CASE-TIME"
        payload = text_payload(cid, 0.6, 0.4)
        payload["time_range"] = {"start": "2026-01-01", "tz": "UTC"}
        r = self.client.post("/api/evidence/upload", json=payload)
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(r.json()["error"]["code"], "INVALID_REQUEST")

    def test_nested_prior_for_both(self):
        cid = "CASE-NESTED-PRIOR"
        self.assertEqual(self.client.post("/api/evidence/upload", json=text_payload(cid, 0.6, 0.4)).status_code, 201)
        r = self.client.post("/api/inference/bayes", json={
            "case_id": cid, "attribute": "both",
            "prior": {
                "gender": {"male": 0.7, "female": 0.3},
                "age": {"18-24": 0.4, "25-29": 0.3, "30-34": 0.2, "35-40": 0.1},
            },
        })
        self.assertEqual(r.status_code, 200, r.text)
        self.assertAlmostEqual(r.json()["results"]["gender"]["prior"]["male"], 0.7)


    def test_identity_target_and_explanation(self):
        cid = "CASE-IDENTITY"
        r = self.client.post("/api/evidence/upload", json=text_payload(cid, 0.1, 0.9, "25-29"))
        self.assertEqual(r.status_code, 201, r.text)
        candidates = [
            {"candidate_id": "cand_true", "display_name": "候选A",
             "gender": "female", "age_bucket": "25-29"},
            {"candidate_id": "cand_male", "display_name": "候选B",
             "gender": "male", "age_bucket": "30-34"},
            {"candidate_id": "cand_female2", "display_name": "候选C",
             "gender": "female", "age_bucket": "35-40"},
        ]
        r = self.client.post("/api/inference/bayes", json={
            "case_id": cid, "target": "identity", "candidates": candidates,
        })
        self.assertEqual(r.status_code, 200, r.text)
        bayes = r.json()
        ident = bayes["results"]["identity"]
        self.assertEqual(ident["prediction"], "cand_true")
        self.assertEqual(ident["ranking"][0]["candidate_id"], "cand_true")
        iid = bayes["inference_id"]

        r = self.client.post("/api/inference/explain", json={"inference_id": iid})
        self.assertEqual(r.status_code, 200, r.text)
        ex = r.json()["results"]["identity"]
        self.assertEqual(ex["prediction"], "cand_true")
        self.assertTrue(ex["narrative"])
        self.assertTrue(ex["top_evidence"])

    def test_identity_requires_candidates(self):
        cid = "CASE-IDENTITY-NOCAND"
        self.assertEqual(self.client.post("/api/evidence/upload", json=text_payload(cid, 0.5, 0.5)).status_code, 201)
        r = self.client.post("/api/inference/bayes", json={"case_id": cid, "target": "identity"})
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(r.json()["error"]["code"], "INVALID_CANDIDATES")


    def test_identity_conflict_and_explain(self):
        cid = "CASE-ID-CONFLICT"
        for probs in ({"cand_a": 0.9, "cand_b": 0.1}, {"cand_a": 0.1, "cand_b": 0.9}):
            payload = text_payload(cid, 0.5, 0.5)
            payload["identity_probs"] = probs
            r = self.client.post("/api/evidence/upload", json=payload)
            self.assertEqual(r.status_code, 201, r.text)
        candidates = [{"candidate_id": "cand_a"}, {"candidate_id": "cand_b"}]
        r = self.client.post("/api/inference/bayes", json={
            "case_id": cid, "target": "identity", "candidates": candidates,
        })
        self.assertEqual(r.status_code, 200, r.text)
        iid = r.json()["inference_id"]
        r = self.client.post("/api/evidence/conflict", json={
            "case_id": cid, "attribute": "identity", "candidates": candidates,
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["attributes"]["identity"]["detected"])
        self.assertTrue(body["downweighted"])
        r = self.client.post("/api/inference/explain", json={"inference_id": iid})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["results"]["identity"]["prediction"] in {"cand_a", "cand_b"}, True)


    def test_audio_asr_and_behavior_sample_chains(self):
        cid = "CASE-MODAL"
        audio = {
            "case_id": cid, "modality": "audio", "source_type": "audio_stub",
            "source_name": "call-asr", "text": "我今年28岁，是一名女生。",
        }
        r = self.client.post("/api/evidence/upload", json=audio)
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json()["status"], "ready")
        self.assertIsNotNone(r.json()["probabilities"]["gender"])

        behavior = {
            "case_id": cid, "modality": "behavior", "source_type": "behavior_stub",
            "source_name": "platform-log",
            "content": {"events": [
                {"action": "login", "ip": "203.0.113.10", "device": "iPhone"},
                {"action": "post", "ip": "203.0.113.10", "device": "iPhone"},
            ]},
        }
        r = self.client.post("/api/evidence/upload", json=behavior)
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual(body["status"], "stored_no_probabilities")
        self.assertEqual(body["content"]["n_events"], 2)
        self.assertEqual(body["content"]["n_ips"], 1)

    def test_payload_too_large(self):
        with tempfile.TemporaryDirectory() as td:
            client = make_client(td, max_upload_bytes=10)
            r = client.post(
                "/api/evidence/upload",
                data={"case_id": "CASE-BIG", "modality": "text",
                      "source_type": "text_model", "source_name": "big"},
                files={"file": ("big.txt", b"x" * 100, "text/plain")},
            )
            self.assertEqual(r.status_code, 413, r.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
