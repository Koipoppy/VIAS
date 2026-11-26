# -*- coding: utf-8 -*-
"""Unified evidence-unit schema for the VIAS pilot.

Schema v1.0 (project-internal freeze, 2026-11-25).  The Pydantic model is the
source of truth; ``schema/evidence_unit.schema.json`` is generated from it by
``src/export_schema.py``.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AGE_BUCKETS = ("18-24", "25-29", "30-34", "35-40")
GENDERS = ("male", "female")
SOURCE_TYPES = (
    "text_model",
    "face_model",
    "self_report",
    "audio_stub",
    "behavior_stub",
    "synthetic_conflict",
    "metadata",
)


class Credibility(BaseModel):
    """Six-dimension credibility used by the project's technical route."""

    model_config = ConfigDict(extra="forbid")

    source_reliability: float = Field(ge=0, le=1, description="来源可靠度，0–1")
    authority: float = Field(ge=0, le=1, description="权威性，0–1")
    objectivity: float = Field(ge=0, le=1, description="客观性，0–1")
    timeliness: float = Field(ge=0, le=1, description="时效性，0–1")
    completeness: float = Field(ge=0, le=1, description="完整性，0–1")
    consistency: float = Field(ge=0, le=1, description="一致性，0–1；冲突后更新")
    score: float = Field(ge=0, le=1, description="六维加权综合分，0–1")
    weight: float = Field(ge=0, le=1, description="有效权重基数，默认等于 score")


class TimeRange(BaseModel):
    """Optional event-time interval; ISO-8601 strings."""

    model_config = ConfigDict(extra="forbid")

    start: str | None = Field(default=None, description="ISO-8601 起始时间")
    end: str | None = Field(default=None, description="ISO-8601 结束时间")


class EvidenceUnit(BaseModel):
    """One modality-specific evidence card, shared by all pipeline stages."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1", "1.0"] = Field(
        default="1.0", description="Schema 版本；新输出固定 1.0，0.1 仅兼容旧数据"
    )
    evidence_id: str = Field(
        min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$",
        description="案件内唯一 ID，建议 <case_id>-<MOD>-<ATTR>",
    )
    case_id: str = Field(
        min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$",
        description="案件 ID；合成案件以 SYN- 开头",
    )
    modality: Literal["text", "image", "audio", "behavior", "metadata"] = Field(
        description="证据模态"
    )
    source_type: Literal[
        "text_model", "face_model", "self_report", "audio_stub",
        "behavior_stub", "synthetic_conflict", "metadata",
    ] = Field(description="来源类型；audio/behavior 当前为预留 stub")
    source_name: str = Field(min_length=1, description="具体模型、规则或设备名称")
    content: dict = Field(
        default_factory=dict,
        description="模态相关内容；v1.0 按 modality 约定必填键，见数据规范",
    )
    raw_ref: str | None = Field(default=None, description="原始文件引用（MinIO/本地路径）")
    time_range: TimeRange | None = Field(default=None, description="事件时间范围")
    language: str | None = Field(default=None, max_length=16, description="BCP-47 语言标签，如 zh")
    gender_probs: dict[str, float] | None = Field(
        default=None, description="性别概率；键必须为 male/female，和为 1"
    )
    age_probs: dict[str, float] | None = Field(
        default=None, description="年龄段概率；键必须为四个目标段，和为 1"
    )
    credibility: Credibility = Field(description="六维可信度")
    supports: list[str] = Field(default_factory=list, description="支持的假设 ID")
    contradicts: list[str] = Field(default_factory=list, description="反对的假设 ID")
    conflict_flag: bool = Field(default=False, description="是否被冲突处理标记")
    conflict_degree: float = Field(default=0.0, ge=0, le=1, description="最大冲突度，0–1")
    injected: bool = Field(default=False, description="是否人工注入的矛盾证据")
    synthetic: bool = Field(default=False, description="是否合成案件/合成配对")
    pii_redacted: bool = Field(default=False, description="文本是否已脱敏")
    weight_multiplier: float = Field(
        default=1.0, gt=0, le=1, description="冲突降权系数，0–1"
    )
    note: str = Field(default="", description="处理说明")

    @field_validator("gender_probs")
    @classmethod
    def _validate_gender_probs(cls, value: dict[str, float] | None):
        if value is None:
            return value
        if set(value) != set(GENDERS):
            raise ValueError("gender_probs keys must be exactly male/female")
        if any(not (0.0 <= float(p) <= 1.0) for p in value.values()):
            raise ValueError("gender_probs values must be in [0, 1]")
        if abs(sum(float(p) for p in value.values()) - 1.0) > 1e-6:
            raise ValueError("gender_probs must sum to 1")
        return value

    @field_validator("age_probs")
    @classmethod
    def _validate_age_probs(cls, value: dict[str, float] | None):
        if value is None:
            return value
        if set(value) != set(AGE_BUCKETS):
            raise ValueError(f"age_probs keys must be exactly {list(AGE_BUCKETS)}")
        if any(not (0.0 <= float(p) <= 1.0) for p in value.values()):
            raise ValueError("age_probs values must be in [0, 1]")
        if abs(sum(float(p) for p in value.values()) - 1.0) > 1e-6:
            raise ValueError("age_probs must sum to 1")
        return value

    def effective_weight(self) -> float:
        """Computed at runtime; not a persisted v1.0 field."""
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
