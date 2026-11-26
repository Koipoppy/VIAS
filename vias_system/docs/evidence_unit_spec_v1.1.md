# VIAS 技术方案与数据规范 v1.1（服务版）

> 本文继承 pilot 的 Schema/数据规范；运行时唯一实现为 `src/vias/core/schema.py`，pilot 目录不参与运行时。

**版本**：1.0（项目内部冻结）  
**日期**：2026-11-25  
**状态**：技术方案与数据规范定稿；统一证据单元 Schema v1.1 已冻结；团队评审待确认  
**适用范围**：课题四子任务三“新型网络社交场景虚实融合身份推断”pilot  
**配套文件**：`src/vias/core/schema.py`、`schema/evidence_unit.schema.json`、`schema/evidence_unit.example.json`、`docs/ARCHITECTURE.md`

> 本文是 pilot 的技术方案与数据规范，不替代项目实施方案、示范单位数据协议或伦理审批文件。涉及真实案件时，以示范单位授权和单位数据制度为准。

---

## 1. 目标与边界

### 1.1 目标

把文本、人脸、自述等多模态证据统一为**证据单元**，经六维可信度加权、贝叶斯融合、冲突处理，输出目标主体的性别与年龄段后验，并给出 Top-3 证据和证据链。

### 1.2 本版覆盖

- 模态：中文文本、亚洲人脸、自述；音频支持 ASR 文本样例链路，行为支持事件特征；内置 ASR/声纹/行为模型未接入。
- 属性：性别（二分类）、年龄段（4 段）。
- 数据：SMP CUP 2016 文本、AFAD-Lite 人脸、30 例合成案件包。
- 评估：50 例 pilot 的 clean / conflict 两个场景。

### 1.3 本版不覆盖

- 真实身份定位、候选身份排序；
- 音频 ASR/声纹、行为序列模型；
- 文档/扫描件解析、跨语言；
- 真实案件数据、示范单位生产环境。

---

## 2. 术语与标签体系

| 属性 | 标签 | 说明 |
|---|---|---|
| 性别 | `male` / `female` | 二分类；v1.0 不处理 unknown/other |
| 精确年龄 | 整数 | 只作数据标注/评估辅助，不作为 Schema 必填输出 |
| 年龄段 | `18-24` / `25-29` / `30-34` / `35-40` | 贝叶斯目标层；四类互斥 |

**映射规则**：

- AFAD 目录 `111=male`、`112=female`；年龄取目录第一段（22/24/28/32/36）。
- 人脸年龄模型原始 9 段映射到 4 个目标段；映射矩阵在 `src/face_model.py`。
- SMP 精确年龄由 `2016 − birth_year` 得到，再映射到 4 段；仅保留 18–40 岁用户。

---

## 3. 技术方案

### 3.1 五段式流程

```
多模态解析 → 统一证据单元 → 六维可信度 → 加权贝叶斯融合 → 冲突处理/可解释
```

每一段是一个可运行模块；模块之间只通过 `EvidenceUnit` 和 `CaseResult` 交换数据。

### 3.2 证据类型

| 证据 | `source_type` | `modality` | 概率字段 | 说明 |
|---|---|---|---|---|
| 文本性别模型 | `text_model` | text | `gender_probs` | TF-IDF 字符 1–3 gram + LogisticRegression |
| 文本年龄模型 | `text_model` | text | `age_probs` | 同上，4 段多分类 |
| 人脸性别模型 | `face_model` | image | `gender_probs` | ViT `rizvandwiki/gender-classification` |
| 人脸年龄模型 | `face_model` | image | `age_probs` | ViT `nateraw/vit-age-classifier`，9→4 段映射 |
| 自述性别 | `self_report` | text | `gender_probs` | 正则；pilot 覆盖 0% |
| 自述年龄 | `self_report` | text | `age_probs` | 正则；pilot 覆盖 22% |
| 注入矛盾 | `synthetic_conflict` | text | 对应属性 | 仅压力测试；`injected=true` |
| 音频（预留） | `audio_stub` | audio | — | `content.status="reserved"`，不参与融合 |
| 行为（预留） | `behavior_stub` | behavior | — | `content.status="reserved"`，不参与融合 |
| 平台元数据 | `metadata` | metadata | — | v1.0 预留，未接入 |

### 3.3 层次结构

```
证据层 E：文本/人脸/自述/注入证据
        ↓
假设层 H：H_gender(male/female)、H_age(18-24/25-29/30-34/35-40)
        ↓
目标层 T：(gender, age_bucket)
```

Pilot 只做属性假设，不做“候选真实身份”层。

### 3.4 六维可信度

| 维度 | 权重 | 取值 | 说明 |
|---|---:|---|---|
| 来源可靠度 `source_reliability` | 0.20 | 0–1 | 来源/采集方式 |
| 权威性 `authority` | 0.15 | 0–1 | 模型/规则/自述的权威程度 |
| 客观性 `objectivity` | 0.20 | 0–1 | 是否受主观表达影响 |
| 时效性 `timeliness` | 0.15 | 0–1 | 证据时间与案件时间接近程度 |
| 完整性 `completeness` | 0.15 | 0–1 | 文本长度/图像质量/字段齐全度 |
| 一致性 `consistency` | 0.15 | 0–1 | 与其他证据的一致程度；冲突后更新 |

```
score = Σ_d w_d × dim_d
effective_weight = credibility.weight × weight_multiplier
```

`credibility.weight` 默认等于 `score`；`weight_multiplier` 初始为 1.0，冲突降权时乘以 0.25。

### 3.5 加权贝叶斯融合

对每个属性分别融合：

```
log P(H | E_1..E_n) ∝ log P(H) + Σ_i ω_i × log P_i(H)
```

- 性别先验：`male=0.5, female=0.5`；
- 年龄段先验：四段各 0.25；
- `P_i(H)` 为模型/规则输出分布；归一化后取 argmax；
- 这是项目“加权贝叶斯”的简化实现，模型后验被当作似然的近似。

### 3.6 冲突处理

```
TV = 0.5 × Σ_h |p(h) − q(h)|          # 全变差；相同分布为 0
C = TV × min(max(p), max(q))          # 置信度加权
```

- 默认阈值：gender `τ = 0.65`，age `τ = 0.50`；
- 若 `C > τ`：对有效权重较低的一方 `weight_multiplier ×= 0.25`；
- 被降权证据 `consistency = 1 − C`，重算综合分；
- 每对证据只处理一次，最多 3 轮；`conflict_degree` 记录最大冲突度。

### 3.7 可解释

- **贡献度**：leave-one-out，`Δ = P(预测标签 | 全部) − P(预测标签 | 去掉该证据)`；
- **Top-k**：按 `|Δ|` 排序，报告 Top-3；
- **证据链图**：证据 → 假设 → 目标，边权为贡献/后验；
- **逐案结果**：`outputs/case_results_*.jsonl`。

### 3.8 评估指标

| 指标 | 定义 |
|---|---|
| 性别 Acc / Macro-F1 | 二分类 |
| 年龄 Acc / Macro-F1 | 4 段分类 |
| 年龄 MAE | 以段中点（21/27/32/37.5）计算 |
| 冲突检出 | 注入案例中 `max_conflict > τ` 的比例 |
| 干净误报 | 干净案例被标记冲突的比例 |
| 恢复率 | 注入案例处理后预测正确的比例 |

---

## 4. 数据规范

### 4.1 数据源与许可

| 数据 | 来源 | 用途 | 许可/限制 |
|---|---|---|---|
| 中文文本 | SMP CUP 2016 微博用户画像 | 合成案件文本侧 | 公开研究数据；仅研究；不重新分发原始数据 |
| 人脸图像 | AFAD-Lite | 合成案件人脸侧 | 仅学术研究；不发布/不再分发 |
| 合成案件包 | 上述两源按同年龄/同性别配对 | pilot 与 30 例结构验证 | 不是真实案件；不对外演示 |

### 4.2 样本构造

- **Pilot**：50 例，按 5 个年龄 × 2 性别分层；每格 5 例。
- **30 例案件包**：每格 3 例；20 例干净 + 10 例注入冲突。
- **配对规则**：文本用户与人脸图像同年龄、同性别；**不是同一个人**，不验证“同一人”假设。
- **标签来源**：SMP 的 birth_year、AFAD 的目录年龄/性别；不使用模型预测值作 ground truth。
- **划分**：文本模型用 SMP 2,824 个用户训练，50 个 pilot 用户不参与训练；人脸模型为公开预训练模型，未在 AFAD 上微调。

### 4.3 个人信息与脱敏

文本进入案件包前做基础正则脱敏：

| 类型 | 替换 |
|---|---|
| URL | `[URL]` |
| 邮箱 | `[EMAIL]` |
| @提及 | `[MENTION]` |
| 手机号 | `[PHONE]` |
| 18 位证件号 | `[ID]` |
| 6 位以上长数字 | `[NUM]` |

- 文本截断到 5,000 字符；
- **正则脱敏不保证完全匿名**；仍可能残留可识别信息；
- 人脸为公开研究数据集图像，按 AFAD 学术研究条款使用；
- 任何对外演示须替换为合成人脸/虚构文本，或取得示范单位书面授权。

### 4.4 数据质量与验收

| 检查 | 标准 |
|---|---|
| Schema 校验 | 每条证据通过 `EvidenceUnit` v1.0 + JSON Schema |
| 概率和 | `gender_probs` / `age_probs` 和为 1（±1e-6） |
| 标签覆盖 | 50 例 pilot 的性别/年龄段均有 ground truth |
| 文本长度 | ≤5,000 字符；无 URL/邮箱/手机/证件号 |
| 图像 | 可解码、单张人脸、质量分 ≥0.6 |
| 案件包 | 30 个目录，每案 6 个文件；20 干净 + 10 注入 |
| 证据数量 | 442 条输出证据 + 147 条案件证据 = 589 条，0 校验失败 |

### 4.5 存储、访问与留存

- 原始数据/案件文件：本地受控目录或 MinIO；**不进入公共 GitHub 仓库**。
- 结构化结果：PostgreSQL 表 `cases`、`evidence_units`、`case_results`、`inference_jobs`（见 `deploy/`）。
- 访问：项目内部最小权限；真实案件需示范单位授权与审批。
- 留存：按项目/单位数据制度执行；pilot 合成数据在项目结束后按约定清理。
- 公开仓库只保留代码、Schema、文档、图表和脱敏统计，不保留原始文本/人脸/证据 JSONL。

### 4.6 禁止事项

- 不得把 SMP/AFAD 原始数据或案件包中的文本/人脸重新分发；
- 不得把合成配对结果表述为“同一人”或“真实身份”；
- 不得用 pilot 指标对外宣称生产级精度；
- 不得绕过示范单位的数据/伦理审批接入真实案件。

---

## 5. 统一证据单元 Schema v1.1

### 5.1 设计原则

1. **一个证据卡**：所有模态共用一套字段，模态差异放在 `content`。
2. **概率与可信度分离**：模型输出概率与证据可信度分开，融合时再相乘。
3. **可追溯**：`raw_ref` 指回原始文件，`evidence_id` 案件内唯一。
4. **可审计**：`injected`、`synthetic`、`pii_redacted` 明确标记数据性质。
5. **可扩展**：音频/行为在无模型输出时以 `reserved`/`parsed` 占位，不改变主结构。
6. **强校验**：v1.0 禁止额外字段；枚举、范围、概率和由 Pydantic + JSON Schema 双重约束。

### 5.2 字段表（23 个）

| # | 字段 | 类型 | 必填 | 约束/说明 |
|---:|---|---|---|---|
| 1 | `schema_version` | string | 否 | `0.1` / `1.0` / `1.1`；新输出默认 `1.1` |
| 2 | `evidence_id` | string | 是 | 案件内唯一；`^[A-Za-z0-9_.:-]+$`；建议 `<case_id>-<MOD>-<ATTR>` |
| 3 | `case_id` | string | 是 | 案件 ID；`^[A-Za-z0-9_.:-]+$`；合成案件 `SYN-xxx` |
| 4 | `modality` | enum | 是 | `text` / `image` / `audio` / `behavior` / `metadata` |
| 5 | `source_type` | enum | 是 | 见 §5.3；当前 7 类 |
| 6 | `source_name` | string | 是 | 具体模型/规则/设备；非空 |
| 7 | `content` | object | 否 | 模态相关内容；按 §5.4 约定必填键 |
| 8 | `raw_ref` | string/null | 否 | MinIO/本地路径；指回原始文件 |
| 9 | `time_range` | object/null | 否 | `{start, end}`；ISO-8601；禁止额外键 |
| 10 | `language` | string/null | 否 | BCP-47，如 `zh`；最长 16 |
| 11 | `gender_probs` | object/null | 否 | 键必须为 `male`/`female`；值 0–1；和为 1 |
| 12 | `age_probs` | object/null | 否 | 键必须为四段；值 0–1；和为 1 |
| 13 | `identity_probs` | object/null | 否 | 候选身份概率；键为 candidate_id；值 0–1；和为 1 |
| 14 | `credibility` | object | 是 | 六维 + `score` + `weight`；8 个子字段均 0–1 |
| 15 | `supports` | string[] | 否 | 支持的假设 ID；默认空 |
| 16 | `contradicts` | string[] | 否 | 反对的假设 ID；默认空 |
| 17 | `conflict_flag` | bool | 否 | 是否被冲突处理标记；默认 false |
| 18 | `conflict_degree` | number | 否 | 最大冲突度；0–1；默认 0 |
| 19 | `injected` | bool | 否 | 是否人工注入；默认 false |
| 20 | `synthetic` | bool | 否 | 是否合成案件/配对；默认 false |
| 21 | `pii_redacted` | bool | 否 | 文本是否已脱敏；默认 false |
| 22 | `weight_multiplier` | number | 否 | 冲突降权系数；`0 < x ≤ 1`；默认 1.0 |
| 23 | `note` | string | 否 | 处理说明；默认空 |

> `effective_weight` 是运行时派生值（`credibility.weight × weight_multiplier`），**不是 v1.1 的持久化字段**；v0.1 输出中的该字段在迁移时已移除。

### 5.3 枚举

| 枚举 | 取值 |
|---|---|
| `modality` | `text`、`image`、`audio`、`behavior`、`metadata` |
| `source_type` | `text_model`、`face_model`、`self_report`、`audio_stub`、`behavior_stub`、`synthetic_conflict`、`metadata` |
| `schema_version` | `0.1`（旧）、`1.0`（兼容）、`1.1`（当前） |

### 5.4 `content` 约定

`content` 在 JSON Schema 中是开放对象，但按模态约定以下必填键：

| 模态 | 必填/关键键 |
|---|---|
| text（模型） | `model`、`n_chars`、`snippet` |
| text（自述） | `snippet`、`claimed` / `claimed_age` / `claimed_bucket` |
| image | `image`、`face_quality`、`afad_age`、`afad_gender`（或 `raw_top_bucket`） |
| audio（预留） | `status="reserved"`、`audio_uri`、`asr_text`、`speaker_id`、`duration_sec` |
| behavior（预留） | `status="reserved"`、`log_uri`、`events`、`platform`、`subject_id` |
| metadata | `source`、`collected_at`、`operator` |

v1.1 不引入按模态的 discriminated union；原因是要保持现有解析器兼容。若后续需要强约束，在 v1.2 增加子 Schema。

### 5.5 不变量

1. `evidence_id` 在案件内唯一；`case_id` 与目录/案件元数据一致。
2. `gender_probs` 与 `age_probs` 若存在，键集合必须精确匹配，和为 1（±1e-6）。
3. `credibility` 六维、`score`、`weight` 均在 [0,1]。
4. `conflict_degree` 在 [0,1]；`weight_multiplier` 在 (0,1]。
5. `audio_stub` / `behavior_stub` 的 `content.status="reserved"`，不带概率，不参与融合。
6. `injected=true` 只用于压力测试；`synthetic=true` 的案件不得表述为真实身份。
7. v1.0 禁止额外字段（`additionalProperties: false`）；新增字段必须升级版本。

### 5.6 示例

完整示例见 `schema/evidence_unit.example.json`。核心片段：

```json
{
  "schema_version": "1.0",
  "evidence_id": "SYN-001-T-A",
  "case_id": "SYN-001",
  "modality": "text",
  "source_type": "text_model",
  "source_name": "TF-IDF char 1-3 gram + LogisticRegression",
  "content": {"model": "text_age", "n_chars": 8512, "snippet": "我今年24，在杭州读研"},
  "age_probs": {"18-24": 0.589, "25-29": 0.212, "30-34": 0.14, "35-40": 0.059},
  "credibility": {
    "source_reliability": 0.65, "authority": 0.6, "objectivity": 0.75,
    "timeliness": 0.85, "completeness": 1.0, "consistency": 0.7,
    "score": 0.751, "weight": 0.751
  },
  "synthetic": true,
  "pii_redacted": true
}
```

### 5.7 版本与变更

| 版本 | 日期 | 变更 |
|---|---|---|
| 0.1 | 2026-11-24 | 初版：22 字段；text/image/audio/behavior/metadata；六维可信度；冲突/可解释字段 |
| **1.0** | **2026-11-25** | **冻结**：枚举化 `source_type`；增加范围/概率和校验；`extra=forbid`；`time_range` 结构化；`schema_version` 默认 1.0；移除持久化 `effective_weight` |
| **1.1** | **2026-11-25** | **目标身份层**：新增可选 `identity_probs`（candidate_id → 概率）；新输出默认 1.1；v1.0 数据仍可读 |

**变更规则**：

- v1.0/v1.1 冻结后，字段名、枚举、约束不做破坏性修改；
- 兼容新增走 v1.1（新增可选字段/子 Schema）；破坏性修改走 v2.0；
- 每次变更同步更新 `src/vias/core/schema.py`、JSON Schema、示例、本文档和测试。

---

## 6. 模块与存储映射

| 层 | 实现 | 输入/输出 |
|---|---|---|
| 解析 | `src/text_model.py`、`src/face_model.py`、`src/parsers/*` | 原始文本/图像 → 概率 |
| 证据单元 | `src/vias/core/schema.py` | 概率 + 可信度 → `EvidenceUnit` |
| 可信度 | `src/credibility.py` | 来源/质量 → 六维 + score |
| 融合 | `src/fusion.py` | 多证据 → 属性后验 |
| 冲突 | `src/fusion.py` | 证据对 → `conflict_degree` / 降权 |
| 可解释 | `src/pipeline.py` | 后验 → LOO / Top-3 / 证据链 |
| 存储 | `deploy/initdb/01_schema.sql` | `cases` / `evidence_units` / `case_results` / `inference_jobs` |

---

## 7. 验收与复现

```bash
python src/export_schema.py                 # 重新生成 JSON Schema
python src/migrate_schema_v1.py             # v0.1 → v1.0 迁移（已执行）
python tests/test_schema_v1.py              # Schema 约束测试
python src/check_stubs.py                   # 音频/行为预留接口
python -I src/pipeline.py --tau 0.50 --downweight 0.25
```

**当前验收结果**：

- JSON Schema：23 个 properties；`additionalProperties=false`；枚举/范围/概率和约束齐全；
- 示例：通过 Pydantic v1.0 校验；
- 证据：442 条输出证据 + 147 条案件证据 = 589 条，迁移后 0 失败；
- 音频/行为：无 ASR/模型输出时 reserved；ASR 文本可走文本模型，行为只做事件特征。
- 团队评审：**待确认**（本版为项目内部冻结，不冒充团队已批准）。

---

## 8. 未决与后续版本

| 事项 | v1.0 处理 | 后续 |
|---|---|---|
| unknown/other 性别 | 不支持 | v1.1 评估是否加入 `unknown` |
| 精确年龄输出 | 只作标注/评估 | v1.1 增加可选 `age_years` |
| 候选身份层 | 不覆盖 | v2.0 与目标层一起设计 |
| `content` 子 Schema | 规范约定，不强制 | v1.1 按 modality 做 discriminated union |
| 文档/扫描件 | 不支持 | 视需求增加 `document` 模态 |
| 团队评审 | 待确认 | 评审通过后发布 v1.0.1 或 v1.1 |

---

## 附录 A：文件清单

| 文件 | 作用 |
|---|---|
| `src/vias/core/schema.py` | Pydantic 模型（v1.0 唯一真源） |
| `schema/evidence_unit.schema.json` | 机器可读 Schema v1.1 |
| `schema/evidence_unit.example.json` | 示例证据 |
| `docs/ARCHITECTURE.md` | 分层、ADR、扩展点、技术债 |
| `docs/evidence_unit_spec_v1.1.md` | 本文 |
