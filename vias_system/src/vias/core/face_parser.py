# -*- coding: utf-8 -*-
"""Optional face parser (ViT age/gender models).

The face model is intentionally opt-in: in API deployments images may arrive
with upstream model probabilities, and loading two ViTs on startup is expensive.
Set VIAS_ENABLE_FACE_MODEL=1 to enable local inference.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .schema import AGE_BUCKETS

logger = logging.getLogger("vias.face_parser")

AGE_SOURCE = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
AGE_MAP = {
    "0-2": {"18-24": 1.0},
    "3-9": {"18-24": 1.0},
    "10-19": {"18-24": 1.0},
    "20-29": {"18-24": 0.5, "25-29": 0.5},
    "30-39": {"30-34": 0.5, "35-40": 0.5},
    "40-49": {"35-40": 1.0},
    "50-59": {"35-40": 1.0},
    "60-69": {"35-40": 1.0},
    "70+": {"35-40": 1.0},
}


def map_age(probs: dict) -> dict:
    out = {b: 0.0 for b in AGE_BUCKETS}
    for src, p in probs.items():
        for tgt, w in AGE_MAP.get(src, {}).items():
            out[tgt] += p * w
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else {b: 0.25 for b in AGE_BUCKETS}


class FaceParser:
    def __init__(self, enabled: bool, device: str = "cpu"):
        self.enabled = enabled
        self.device = device
        self._age = None
        self._gender = None

    def _load(self):
        if not self.enabled:
            return False
        if self._age is not None and self._gender is not None:
            return True
        try:
            from transformers import pipeline
            dev = 0 if self.device.startswith("cuda") else -1
            self._age = pipeline("image-classification", model="nateraw/vit-age-classifier", device=dev)
            self._gender = pipeline("image-classification", model="rizvandwiki/gender-classification", device=dev)
            return True
        except Exception:
            logger.exception("failed to load face models; image evidence stored without probabilities")
            self._age = self._gender = None
            return False

    def parse(self, path: str | Path) -> tuple[dict | None, dict | None, dict]:
        if not self.enabled:
            return None, None, {"status": "stored", "model": "face_model_disabled"}
        if not self._load():
            return None, None, {"status": "face_model_error", "model": "face_model_unavailable"}
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            age_raw = {d["label"]: float(d["score"]) for d in self._age(img, top_k=9)}
            gender_raw = {d["label"]: float(d["score"]) for d in self._gender(img, top_k=2)}
            age_src = {k: age_raw.get(k, 0.0) for k in AGE_SOURCE}
            total = sum(age_src.values())
            age_src = {k: v / total for k, v in age_src.items()} if total > 0 else age_src
            male = float(gender_raw.get("male", 0.0))
            female = float(gender_raw.get("female", 0.0))
            s = male + female
            if s > 0:
                male, female = male / s, female / s
            return ({"male": male, "female": female},
                    map_age(age_src),
                    {"status": "parsed", "model": "vit_age_gender",
                     "raw_age_top": max(age_src, key=age_src.get)})
        except Exception:
            logger.exception("face parsing failed for %s", path)
            return None, None, {"status": "face_model_error", "model": "face_model_error"}
