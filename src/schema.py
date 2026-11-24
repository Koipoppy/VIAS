# -*- coding: utf-8 -*-
"""Unified evidence-unit schema for the VIAS pilot."""
from typing import Literal

from pydantic import BaseModel, Field


class Credibility(BaseModel):
    """Six-dimension credibility used by the project's technical route."""

    source_reliability: float
    authority: float
    objectivity: float
    timeliness: float
    completeness: float
    consistency: float
    score: float
    weight: float


class EvidenceUnit(BaseModel):
    schema_version: str = "0.1"
    evidence_id: str
    case_id: str
    modality: Literal["text", "image", "audio", "behavior", "metadata"]
    source_type: str
    source_name: str
    content: dict = Field(default_factory=dict)
    raw_ref: str | None = None
    time_range: dict | None = None
    language: str | None = None
    gender_probs: dict[str, float] | None = None
    age_probs: dict[str, float] | None = None
    credibility: Credibility
    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    conflict_flag: bool = False
    conflict_degree: float = 0.0
    injected: bool = False
    synthetic: bool = False
    pii_redacted: bool = False
    weight_multiplier: float = 1.0
    note: str = ""

    def effective_weight(self) -> float:
        return max(1e-6, self.credibility.weight * self.weight_multiplier)


class CaseResult(BaseModel):
    case_id: str
    uid: str
    true_gender: str
    true_age: int
    true_age_bucket: str
    predicted_gender: str
    predicted_age_bucket: str
    gender_posterior: dict[str, float]
    age_posterior: dict[str, float]
    conflict_detected: bool
    max_conflict: float
    conflict_rounds: int
    evidence_ids: list[str]
    top_evidence_gender: list[dict]
    top_evidence_age: list[dict]
    injected: bool
