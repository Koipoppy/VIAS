# -*- coding: utf-8 -*-
"""Text evidence parser.

If the pilot TF-IDF models are present, they produce gender/age probabilities.
If not, the parser falls back to explicit self-report claims; it never fabricates
probabilities for arbitrary text.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import joblib
import numpy as np

from .schema import AGE_BUCKETS, GENDERS

logger = logging.getLogger("vias.text_parser")

GENDER_PATTERNS = [
    (r"(?:我|本人|俺)(?:是|一个|一名)?(?:男|女)(?:生|人|孩|的)?", 0.85),
    (r"(?:我|本人)[^。！？\n]{0,16}?(?:是|为)?(?:一个|一名)?(?:男|女)(?:生|人|孩|的)?", 0.75),
    (r"性别[:：]?(男|女)", 0.9),
    (r"(?:^|[，。！？\s])(男|女)生(?:[，。！？\s]|$)", 0.65),
    (r"(?:我|本人)(?:是|为)?(男|女)(?:的)?(?:[，。！？\s]|$)", 0.8),
]
AGE_PATTERNS = [
    (r"(?:今年|今年已经|已经)?(\d{2})岁", 0.8),
    (r"(?:19|20)(\d{2})年(?:出生|生人|生)", 0.75),
    (r"(?:我|本人)(?:是|为)?(\d{2})后", 0.6),
    (r"(\d{2})后(?:[，。！？\s]|$)", 0.5),
]


def bucket_of(age: int) -> str:
    if age <= 24:
        return "18-24"
    if age <= 29:
        return "25-29"
    if age <= 34:
        return "30-34"
    return "35-40"


def extract_self_report(text: str) -> dict:
    t = text.replace(" ", "")
    g_votes: dict[str, float] = {}
    a_votes: dict[int, float] = {}
    snippets: list[str] = []
    for pat, conf in GENDER_PATTERNS:
        for m in re.finditer(pat, t):
            s = m.group(0)
            g = "male" if "男" in s else "female" if "女" in s else None
            if g:
                g_votes[g] = g_votes.get(g, 0.0) + conf
                snippets.append(s[:40])
    for pat, conf in AGE_PATTERNS:
        for m in re.finditer(pat, t):
            try:
                if "年" in m.group(0) and "出生" in m.group(0):
                    year = 1900 + int(m.group(1)) if int(m.group(1)) > 30 else 2000 + int(m.group(1))
                    age = 2016 - year
                else:
                    age = int(m.group(1))
            except Exception:
                continue
            if 10 <= age <= 80:
                a_votes[age] = a_votes.get(age, 0.0) + conf
                snippets.append(m.group(0)[:40])
    g = max(g_votes, key=g_votes.get) if g_votes else None
    age = max(a_votes, key=a_votes.get) if a_votes else None
    return {
        "gender": g,
        "gender_conf": min(0.95, g_votes[g]) if g else 0.0,
        "age": age,
        "age_conf": min(0.95, a_votes[age]) if age else 0.0,
        "age_bucket": bucket_of(age) if age is not None else None,
        "snippets": snippets[:5],
    }


def _norm(value: dict) -> dict:
    out = {k: max(0.0, float(v)) for k, v in value.items()}
    total = sum(out.values())
    if total <= 0:
        n = len(out)
        return {k: 1.0 / n for k in out}
    return {k: v / total for k, v in out.items()}


class TextParser:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self._vectorizer = None
        self._gender = None
        self._age = None
        self._load_attempted = False

    def _load(self) -> bool:
        if self._load_attempted:
            return self._vectorizer is not None
        self._load_attempted = True
        files = [self.model_dir / "text_vectorizer.joblib",
                 self.model_dir / "text_gender.joblib",
                 self.model_dir / "text_age.joblib"]
        if not all(p.exists() for p in files):
            logger.warning("text models not found in %s; self-report fallback only", self.model_dir)
            return False
        try:
            self._vectorizer = joblib.load(files[0])
            self._gender = joblib.load(files[1])
            self._age = joblib.load(files[2])
            logger.info("loaded text models from %s", self.model_dir)
            return True
        except Exception:
            logger.exception("failed to load text models")
            self._vectorizer = self._gender = self._age = None
            return False

    def _model_probs(self, text: str) -> tuple[dict | None, dict | None, dict]:
        if not self._load():
            return None, None, {}
        x = self._vectorizer.transform([text])
        gp = self._gender.predict_proba(x)[0]
        ap = self._age.predict_proba(x)[0]
        g_classes = [str(x) for x in self._gender.classes_]
        a_classes = [str(x) for x in self._age.classes_]
        gender = {k: float(gp[g_classes.index(k)]) if k in g_classes else 0.0 for k in GENDERS}
        age = {k: float(ap[a_classes.index(k)]) if k in a_classes else 0.0 for k in AGE_BUCKETS}
        return _norm(gender), _norm(age), {
            "model": "tfidf_char_1_3_lr",
            "gender_model": "tfidf_char_1_3_lr",
            "age_model": "tfidf_char_1_3_lr",
        }

    def parse(self, text: str, source_type: str) -> tuple[dict | None, dict | None, dict]:
        text = (text or "").strip()
        if not text:
            return None, None, {"n_chars": 0, "snippet": ""}
        claims = extract_self_report(text)
        content = {
            "n_chars": len(text),
            "snippet": text[:160],
            "self_report": claims,
        }

        if source_type == "self_report":
            gp = ap = None
            if claims["gender"]:
                gp = {g: 0.15 for g in GENDERS}
                gp[claims["gender"]] = 0.85
                gp = _norm(gp)
            if claims["age_bucket"]:
                ap = {b: 0.05 for b in AGE_BUCKETS}
                ap[claims["age_bucket"]] = 0.85
                ap = _norm(ap)
            content["model"] = "regex_self_report"
            return gp, ap, content

        gp, ap, meta = self._model_probs(text)
        if gp is not None:
            content.update(meta)
            return gp, ap, content

        # No model: only explicit self-report claims are usable.
        gp = ap = None
        if claims["gender"]:
            gp = {g: 0.15 for g in GENDERS}
            gp[claims["gender"]] = 0.85
            gp = _norm(gp)
        if claims["age_bucket"]:
            ap = {b: 0.05 for b in AGE_BUCKETS}
            ap[claims["age_bucket"]] = 0.85
            ap = _norm(ap)
        content["model"] = "regex_self_report_fallback"
        return gp, ap, content
