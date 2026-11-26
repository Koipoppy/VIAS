# -*- coding: utf-8 -*-
"""Pydantic request/response contracts for the public API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Modality = Literal["text", "image", "audio", "behavior", "metadata"]
SourceType = Literal[
    "text_model", "face_model", "self_report", "audio_stub",
    "behavior_stub", "synthetic_conflict", "metadata",
]
Attribute = Literal["gender", "age", "both", "identity"]

ALLOWED_SOURCE_MODALITY = {
    "text_model": {"text"},
    "self_report": {"text"},
    "synthetic_conflict": {"text"},
    "face_model": {"image"},
    "audio_stub": {"audio"},
    "behavior_stub": {"behavior"},
    "metadata": {"metadata"},
}


class EvidenceUploadJson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$", min_length=1)
    modality: Modality
    source_type: SourceType
    source_name: str = Field(min_length=1)
    evidence_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.:-]+$")
    text: str | None = None
    content: dict[str, Any] = Field(default_factory=dict)
    filename: str | None = None
    file_base64: str | None = None
    gender_probs: dict[str, float] | None = None
    age_probs: dict[str, float] | None = None
    identity_probs: dict[str, float] | None = None
    time_range: dict[str, Any] | None = None
    language: str | None = Field(default=None, max_length=16)
    synthetic: bool = False
    pii_redacted: bool = False
    injected: bool = False
    request_id: str | None = Field(default=None, max_length=64)


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    evidence_ids: list[str] | None = None
    overrides: dict[str, dict[str, float]] | None = None


class CandidateIdentity(BaseModel):
    """A candidate in the target-identity layer.

    ``gender``/``age_bucket`` are optional attribute constraints used by the
    pilot's attribute-compatible candidate scorer.  Production can instead
    supply candidate-specific ``identity_probs`` on evidence.
    """

    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$", min_length=1)
    display_name: str | None = None
    prior: float | None = Field(default=None, ge=0.0, le=1.0)
    gender: Literal["male", "female"] | None = None
    age_bucket: Literal["18-24", "25-29", "30-34", "35-40"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BayesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    target: Literal["attribute", "identity"] = "attribute"
    attribute: Attribute = "both"
    evidence_ids: list[str] | None = None
    prior: dict[str, Any] | None = None
    candidates: list[CandidateIdentity] | None = None
    account_a: str | None = None
    account_b: str | None = None
    platform: str | None = None
    weight_power: float = Field(default=1.5, ge=0.5, le=3.0)

    @model_validator(mode="after")
    def _check_target(self):
        if self.target == "attribute" and self.attribute == "identity":
            raise ValueError("target=attribute 时 attribute 不能为 identity；请用 target=identity")
        return self


class ConflictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    attribute: Attribute = "both"
    evidence_ids: list[str] | None = None
    tau: float | None = Field(default=None, ge=0.0, le=1.0)
    tau_gender: float = Field(default=0.65, ge=0.0, le=1.0)
    tau_age: float = Field(default=0.50, ge=0.0, le=1.0)
    tau_identity: float = Field(default=0.50, ge=0.0, le=1.0)
    candidates: list[CandidateIdentity] | None = None
    downweight: float = Field(default=0.25, gt=0.0, le=1.0)
    max_rounds: int = Field(default=3, ge=1, le=10)
    weight_power: float = Field(default=1.5, ge=0.5, le=3.0)
    persist: bool = True


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inference_id: str | None = None
    case_id: str | None = None
    attribute: Attribute = "both"
    top_k: int = Field(default=3, ge=1, le=10)


class UploadResponse(BaseModel):
    evidence_id: str
    case_id: str
    status: str
    modality: str
    source_type: str
    source_name: str
    schema_version: str
    raw_ref: str | None
    content: dict[str, Any]
    content_snippet: str | None = None
    probabilities: dict[str, Any]
    credibility: dict[str, Any]
    created_at: str
    storage: dict[str, Any] | None = None


class EvaluateResponse(BaseModel):
    case_id: str
    evaluated: list[dict[str, Any]]


class InferenceResponse(BaseModel):
    inference_id: str
    case_id: str
    target: str
    attribute: str
    weight_power: float
    candidates: list[dict[str, Any]] | None = None
    context: dict[str, Any] | None = None
    results: dict[str, Any]
    used_evidence: list[dict[str, Any]]
    created_at: str


class ConflictResponse(BaseModel):
    conflict_id: str
    case_id: str
    tau: float | None
    taus: dict[str, float]
    weight_power: float
    attributes: dict[str, Any]
    downweighted: list[str]
    persisted: bool
    created_at: str


class ExplainResponse(BaseModel):
    explanation_id: str
    inference_id: str
    case_id: str
    results: dict[str, Any]
    created_at: str


class ErrorResponse(BaseModel):
    error: dict[str, Any]
