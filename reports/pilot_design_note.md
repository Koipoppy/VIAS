# Pilot Design Note — 证据单元与推断 v0.1（历史设计说明）

> **状态更新（2026-11-25）**：Schema 已升级并冻结为 v1.0；字段约束、枚举、数据规范与版本规则以 [`technical_scheme_and_data_spec_v1.0.md`](technical_scheme_and_data_spec_v1.0.md) 为准。本文保留 pilot 设计背景与公式。

**日期**：2026-11-24  
**配套代码**：`src/schema.py`、`src/credibility.py`、`src/fusion.py`、`src/pipeline.py`

---

## 1. 标签体系

| 属性 | 标签 | 说明 |
|---|---|---|
| 性别 | `male` / `female` | 二分类；暂不处理 unknown/other |
| 年龄 | 精确年龄 | SMP：`2016 − birth_year`；AFAD：图片目录年龄 |
| 年龄段 | `18-24` / `25-29` / `30-34` / `35-40` | 贝叶斯目标层使用 |

**映射**：AFAD 目录 `111=male`、`112=female`；年龄取目录第一段（22/24/28/32/36）。人脸模型原始 9 段（0-2…70+）映射到 4 个目标段，映射矩阵见 `src/face_model.py`。

---

## 2. 证据类型

| 证据 | source_type | modality | 概率字段 | 说明 |
|---|---|---|---|---|
| 文本性别模型 | `text_model` | text | `gender_probs` | TF-IDF 字符 1–3 gram + LogisticRegression |
| 文本年龄模型 | `text_model` | text | `age_probs` | 同上，4 段多分类 |
| 人脸性别模型 | `face_model` | image | `gender_probs` | ViT `rizvandwiki/gender-classification` |
| 人脸年龄模型 | `face_model` | image | `age_probs` | ViT `nateraw/vit-age-classifier`，9→4 段映射 |
| 音频（预留） | `audio_stub` | audio | — | ASR/声纹模型预留；`content.status="reserved"`，不参与融合 |
| 行为（预留） | `behavior_stub` | behavior | — | 登录/发帖/互动/转账序列预留；`content.status="reserved"`，不参与融合 |
| 自述性别 | `self_report` | text | `gender_probs` | 正则，pilot 覆盖 0% |
| 自述年龄 | `self_report` | text | `age_probs` | 正则，pilot 覆盖 22% |
| 注入矛盾 | `synthetic_conflict` | text | 对应属性 | 仅压力测试用，`injected=true` |

**预留接口**：`src/parsers/audio_stub.py` / `behavior_stub.py` 返回 `modality="audio"` / `"behavior"` 的占位 `EvidenceUnit`，`content.status="reserved"`，不带概率、不参与融合；自检脚本 `src/check_stubs.py`。这样后续接 ASR/声纹/行为序列模型时，不需要改 Schema 和融合层。

---

## 3. 证据单元字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `schema_version` | str | 否 | 默认 `0.1` |
| `evidence_id` | str | 是 | 案件内唯一 |
| `case_id` | str | 是 | 案件 ID |
| `modality` | text/image/audio/behavior/metadata | 是 | 模态 |
| `source_type` | str | 是 | 来源类型 |
| `source_name` | str | 是 | 具体模型/规则 |
| `content` | dict | 是 | 片段、图片路径、注入标记 |
| `raw_ref` | str | 否 | MinIO/本地路径 |
| `time_range` | dict | 否 | `{start, end}` |
| `language` | str | 否 | `zh` / `en` |
| `gender_probs` | dict[str,float] | 否 | 性别概率分布 |
| `age_probs` | dict[str,float] | 否 | 年龄段概率分布 |
| `credibility` | Credibility | 是 | 六维 + 综合分 + 权重 |
| `conflict_flag` | bool | 是 | 是否被冲突处理标记 |
| `conflict_degree` | float | 是 | 最大冲突度 |
| `injected` | bool | 是 | 是否人工注入 |
| `synthetic` | bool | 否 | 是否合成案件/配对 |
| `pii_redacted` | bool | 否 | 文本是否已脱敏 |
| `weight_multiplier` | float | 是 | 冲突降权系数 |
| `note` | str | 否 | 处理说明 |

---

## 4. 六维可信度

| 维度 | 权重 | 来源：文本模型 | 自述 | 人脸模型 | 注入矛盾 |
|---|---|---|---|---|---|
| 来源可靠度 | 0.20 | 0.65 | 0.75 | 0.70 | 0.68 |
| 权威性 | 0.15 | 0.60 | 0.80 | 0.65 | 0.72 |
| 客观性 | 0.20 | 0.75 | 0.45–0.70 | 0.85 | 0.55 |
| 时效性 | 0.15 | 0.85 | 0.85 | 0.80 | 0.80 |
| 完整性 | 0.15 | 0.40–1.00（文本长度） | 0.50–1.00（置信度） | 0.70–0.80 | 0.60 |
| 一致性 | 0.15 | 0.70 | 0.70 | 0.70 | 0.50 |

`score = Σ_d w_d × dim_d`；`ω_i = score × weight_multiplier`。

---

## 5. 层次结构

```
证据层：E_text_gender, E_text_age, E_face_gender, E_face_age, E_self_*, E_injected_*
        ↓
属性/假设层：H_gender(male/female), H_age(18-24/25-29/30-34/35-40)
        ↓
目标层：T = (gender, age_bucket)
```

Pilot 只做两层属性假设，不做“候选真实身份”层。

---

## 6. 融合

对每个属性分别融合：

```
log P(H | E_1..E_n) ∝ log P(H) + Σ_i ω_i × log P_i(H)
```

- 性别先验：0.5 / 0.5；
- 年龄段先验：0.25 × 4；
- `P_i(H)` 为模型/规则输出的概率分布；
- 归一化后取 argmax。

这是项目“加权贝叶斯”的简化实现；模型后验被当作似然的近似。

---

## 7. 冲突

```
K = 1 − Σ_h p(h) q(h)                  # D-S 冲突质量
C = K × min(max(p), max(q))            # 置信度加权
```

- 阈值 `τ = 0.50`；
- 若 `C > τ`：对有效权重较低的一方 `weight_multiplier ×= 0.25`；
- 被降权证据的 `consistency = 1 − C`，重算综合分；
- 每对证据只处理一次，最多 3 轮。

---

## 8. 可解释

- **贡献度**：leave-one-out，`Δ = P(预测标签 | 全部) − P(预测标签 | 去掉该证据)`；
- **Top-k**：按 `|Δ|` 排序，报告 Top-3；
- **证据链图**：证据 → 假设 → 目标，边权为贡献/后验；
- **逐案结果**：`outputs/case_results_*.jsonl`。

---

## 9. 指标

| 指标 | 说明 |
|---|---|
| Gender Acc / Macro-F1 | 性别二分类 |
| Age Acc / Macro-F1 | 4 段分类 |
| Age MAE | 以段中点（21/27/32/37.5）计算 |
| Conflict Detection | 注入案例中 `max_conflict > τ` 的比例 |
| Clean FPR | 干净案例被标记为冲突的比例 |
| Recovery | 注入案例处理后预测正确的比例 |

---

## 10. 假设与限制

- 文本用户与人脸是**同年龄、同性别配对**，不是同一人；不验证“同一人”假设。
- 30 例合成案件包在 `cases_synthetic/`：20 例干净 + 10 例注入冲突；它是 10 月“30 例样例案件”的**结构占位**，不是真实案件。
- 人脸模型为公开预训练模型，未在 AFAD 上微调；年龄段粗。
- 可信度为规则分，未学习/校准。
- Pilot 规模 50 例，指标只作基线，不做显著性结论。
