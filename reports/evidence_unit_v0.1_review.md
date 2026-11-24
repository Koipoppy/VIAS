# 统一证据单元 Schema v0.1 — 评审稿

**日期**：2026-11-24  
**状态**：待评审（pilot 已按此实现）  
**配套文件**：`src/schema.py`、`schema/evidence_unit.schema.json`、`schema/evidence_unit.example.json`

---

## 1. 目的

把文本、图像、音频、行为日志统一成同一张“证据卡”，让解析、可信度、融合、冲突、可解释各模块读写同一个结构。

## 2. 模态与来源

| 字段 | 取值 |
|---|---|
| `modality` | `text` / `image` / `audio` / `behavior` / `metadata` |
| `source_type` | `text_model` / `face_model` / `self_report` / `audio_stub` / `behavior_stub` / `synthetic_conflict` |

> 音频/行为当前是**预留接口**：`audio_stub` / `behavior_stub` 返回占位证据，`content.status="reserved"`，不参与融合。

## 3. 字段表

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `schema_version` | str | 否 | 默认 `0.1` |
| `evidence_id` | str | 是 | 案件内唯一；建议 `<case_id>-<MOD>-<ATTR>` |
| `case_id` | str | 是 | 案件 ID；合成案件用 `SYN-xxx` |
| `modality` | enum | 是 | 见上表 |
| `source_type` | str | 是 | 见上表 |
| `source_name` | str | 是 | 具体模型/规则/设备 |
| `content` | dict | 否 | 模态相关字段（见 §5） |
| `raw_ref` | str | 否 | MinIO/本地路径，指回原始文件 |
| `time_range` | dict | 否 | `{start, end}`，事件时间范围 |
| `language` | str | 否 | `zh` / `en` / … |
| `gender_probs` | dict | 否 | `{"male":p,"female":p}`，和为 1 |
| `age_probs` | dict | 否 | `{"18-24":p,...,"35-40":p}`，和为 1 |
| `credibility` | Credibility | 是 | 六维 + score + weight |
| `supports` | list[str] | 否 | 支持的假设 ID |
| `contradicts` | list[str] | 否 | 反对的假设 ID |
| `conflict_flag` | bool | 否 | 是否被冲突处理标记 |
| `conflict_degree` | float | 否 | 最大冲突度 |
| `injected` | bool | 否 | 是否人工注入的矛盾证据 |
| `synthetic` | bool | 否 | 是否合成案件/合成配对 |
| `pii_redacted` | bool | 否 | 文本是否已脱敏 |
| `weight_multiplier` | float | 否 | 冲突降权系数，默认 1.0 |
| `note` | str | 否 | 处理说明 |

## 4. Credibility

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `source_reliability` | float | 是 | 来源可靠度 0–1 |
| `authority` | float | 是 | 权威性 0–1 |
| `objectivity` | float | 是 | 客观性 0–1 |
| `timeliness` | float | 是 | 时效性 0–1 |
| `completeness` | float | 是 | 完整性 0–1 |
| `consistency` | float | 是 | 一致性 0–1（冲突后更新） |
| `score` | float | 是 | 六维加权综合分 |
| `weight` | float | 是 | 有效权重（默认 = score） |

## 5. 模态相关 `content`

| 模态 | 关键字段 |
|---|---|
| text | `text` / `snippet`、`model`、`n_chars` |
| image | `image`、`face_quality`、`detected_objects` |
| audio（预留） | `status="reserved"`、`audio_uri`、`asr_text`、`speaker_id`、`duration_sec` |
| behavior（预留） | `status="reserved"`、`log_uri`、`events`（`timestamp/action/object/ip/device`）、`platform`、`subject_id` |
| metadata | `source`、`collected_at`、`operator` |

## 6. 示例

完整示例见 `schema/evidence_unit.example.json`；核心片段：

```json
{
  "schema_version": "0.1",
  "evidence_id": "SYN-001-T-A",
  "case_id": "SYN-001",
  "modality": "text",
  "source_type": "text_model",
  "source_name": "TF-IDF char 1-3 gram + LogisticRegression",
  "content": {"model": "text_age", "n_chars": 8512, "snippet": "我今年24，在杭州读研"},
  "age_probs": {"18-24": 0.589, "25-29": 0.212, "30-34": 0.140, "35-40": 0.059},
  "credibility": {"source_reliability": 0.65, "authority": 0.60, "objectivity": 0.75,
                  "timeliness": 0.85, "completeness": 1.0, "consistency": 0.70,
                  "score": 0.751, "weight": 0.751},
  "synthetic": true,
  "pii_redacted": true
}
```

## 7. 待评审问题

1. `source_type` 枚举是否够用？是否要加 `document`（PDF/扫描件）？
2. `age_probs` 固定 4 段，是否要支持精确年龄/自定义分段？
3. 是否要加 `candidate_identity` 字段（目标层候选集）？
4. `credibility` 六维权重是否项目统一，还是按来源类型不同？
5. `content` 是自由 dict 还是按 modality 定义子 Schema？
6. `synthetic` / `pii_redacted` 是否保留为正式字段？
7. 证据 ID / 案件 ID 命名规范是否确认？

## 8. 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| 0.1 | 2026-11-24 | 初版：text/image/audio/behavior/metadata；六维可信度；冲突/可解释字段；合成/脱敏标记 |
