# Pilot 交付说明

**日期**：2026-11-24  
**状态**：可运行、可复现的 pilot v0.1  
**范围**：文本 + 人脸，性别 / 年龄段代理目标，50 个合成多模态样本

---

## 先看这四份

1. **一页纸**：[`reports/executive_summary.md`](reports/executive_summary.md) — 给项目负责人的 1 页总结
2. **完整报告**：[`reports/pilot_report.md`](reports/pilot_report.md) — 数据、方法、结果、案例、局限、下一步
3. **设计说明**：[`reports/pilot_design_note.md`](reports/pilot_design_note.md) — 证据单元 Schema、六维可信度、融合/冲突公式
4. **Schema 评审稿**：[`reports/evidence_unit_v0.1_review.md`](reports/evidence_unit_v0.1_review.md) — 字段表、必填/枚举、待评审问题

---

## GitHub 仓库范围

本仓库只提交代码、Schema、报告、测试、图表、环境配置；以下内容因隐私、许可和体积原因**不推送**：

- `data/raw/`、`data/processed/`（SMP/AFAD 原始与训练数据）
- `cases_synthetic/SYN-*/` 下的 `text.txt`、`face.jpg`、`evidence_units.jsonl`、`result.json`（案件数据）
- `models/`（文本模型）
- `outputs/evidence_units_*.jsonl`、`outputs/case_results_*.jsonl`、`outputs/method_predictions_*.csv`（含文本片段/逐案结果）
- `*.zip`、`papers_pilot/`、本地项目资料 PDF/

仓库保留：`cases_synthetic/README.md`、`DATA_NOTICE.md`、`manifest.csv`，以及代码、报告、图表、测试和环境配置。

---

## 关键结果

| 方法 | 性别 Acc | 年龄 Acc | 年龄 MAE |
|---|---|---|---|
| 文本单模态 | 0.780 | 0.360 | 4.62 |
| 人脸单模态 | 0.960 | 0.460 | 6.36 |
| **融合（可信度）** | **0.960** | **0.500** | **3.94** |
| 融合 + 冲突处理 | 0.920 | 0.500 | 3.98 |

- 注入 10 例矛盾证据：冲突检出 **9/10（90%）**；干净集误报 **7/50（14%）**。
- 冲突处理把注入子集年龄从 **0.30 → 0.40**，但干净集性别 **0.96 → 0.92**。
- 结论：链路跑通；性别可用；年龄是瓶颈；可信度/冲突需校准。

---

## 10 月计划进度

| 计划项 | 状态 | 说明 |
|---|---|---|
| 需求调研 / 公开数据可得性 | ✅ 公开数据 | 27 个数据集清单+可达性；示范单位脱敏样本待确认 |
| 原型边界（文本、图像先行） | ✅ | text+image 已实现；audio/behavior 预留 stub |
| 技术方案 / 数据规范 | ✅ pilot 版 | `pilot_design_note.md` + `evidence_unit_v0.1_review.md` |
| 统一证据单元 Schema | ✅ v0.1 | `src/schema.py` + `schema/evidence_unit.schema.json` + 示例 |
| 30 例样例案件 | ⚠️ 合成案件包 | `cases_synthetic/` 30 例（20 干净 + 10 注入），**不是真实案件** |
| 开发环境 | ✅ pilot + compose | Python 环境已跑通；`deploy/` 提供 PG14+MinIO+Redis |

---

## 交付物清单

| 路径 | 内容 |
|---|---|
| `src/schema.py` | 证据单元 Pydantic 模型 |
| `src/credibility.py` | 六维可信度规则 |
| `src/fusion.py` | 加权融合 + 置信度加权 D-S 冲突 |
| `src/pipeline.py` | 端到端流水线（证据→融合→冲突→可解释→指标） |
| `src/prepare_smp.py` | SMP 2016 解析与 train/pilot 划分 |
| `src/build_cases.py` | SMP 用户与 AFAD 人脸配对 |
| `src/extract_afad_samples.py` | AFAD 分层采样 |
| `src/text_model.py` | 文本性别/年龄模型 |
| `src/face_model.py` | 人脸性别/年龄模型 |
| `src/make_figures.py` | 图表生成 |
| `src/parsers/` | audio/behavior 预留接口（stub） |
| `src/check_stubs.py` | 预留接口自检 |
| `src/build_case_packages.py` | 30 例合成案件打包 |
| `schema/evidence_unit.schema.json` | JSON Schema v0.1 |
| `schema/evidence_unit.example.json` | 证据单元示例 |
| `cases_synthetic/` | 30 例合成案件包（20 干净 + 10 注入） |
| `deploy/` | PostgreSQL 14 + MinIO + Redis compose |
| `data/processed/` | 训练/ pilot CSV、文本/人脸预测、自述抽取 |
| `models/` | 文本模型（joblib） |
| `outputs/evidence_units_*.jsonl` | 证据单元（干净/注入） |
| `outputs/case_results_*.jsonl` | 逐案结果、后验、冲突日志、Top 证据 |
| `outputs/evidence_graphs/*.json` | 证据链图数据 |
| `outputs/metrics.json` | 全部指标 |
| `outputs/figures/*.png` | 8 张图表 |
| `reports/text_model_metrics.json` | 文本模型指标 |
| `reports/face_model_metrics.json` | 人脸模型指标 |
| `reports/evidence_unit_v0.1_review.md` | Schema v0.1 一页评审稿 |

---

## 复现

```bash
cd vias_pilot
pip install -r requirements.txt

python src/prepare_smp.py
python src/build_cases.py
python src/text_model.py
python src/face_model.py          # 首次会下载 ViT 权重
python src/pipeline.py --tau 0.50 --downweight 0.25
python src/make_figures.py
python src/check_stubs.py         # 音频/行为预留接口自检
python src/build_case_packages.py # 生成 30 例合成案件包

# 可选：项目级环境（需要 Docker）
cd deploy && cp .env.example .env && docker compose up -d
```

**数据**：SMP 2016 原始文件在 `data/raw/smp/`（约 79 MB）；AFAD-Lite 原始归档约 865 MB，已只保留 60 张采样图在 `data/raw/afad/sampled/`。重采样见 `data/raw/afad/README.txt` 和 `src/extract_afad_samples.py`。

---

## 边界与注意

- **不是真实身份 pilot**：文本用户与人脸是同年龄、同性别配对，不是同一个真实人；目标变量是性别/年龄段。
- **30 例案件是合成的**：`cases_synthetic/` 是 10 月“30 例样例案件”的结构占位（20 干净 + 10 注入），不是真实案件；真实案件仍需示范单位脱敏样本。
- **样本小**：50 例 pilot + 30 例合成案件包，指标只作基线，不做显著性结论。
- **音频/行为是预留接口**：`audio_stub` / `behavior_stub` 返回占位证据，不参与融合，不是已实现模型。
- **人脸模型非中文专用**：公开 ViT 模型，未在 AFAD 微调；年龄输出粗（9 段）。
- **文本年龄弱**：TF-IDF + LR，年龄 0.36；自述年龄覆盖仅 22%，自述性别覆盖 0%。
- **可信度是规则**：六维分数未学习/校准；可信度加权与等权结果相同，说明区分度不足。
- **冲突处理有副作用**：干净集误报 14%，会误伤正确证据；下一版需属性-specific τ、置信度门控、连续降权。
- **未覆盖**：LLM 解析、候选身份集合、前端、30 例**真实**案件；音频/行为只有预留接口，未接模型。

---

## 建议下一步

1. **年龄模型**：AFAD/MegaAge 微调 + 回归/更细分段 + 概率校准。
2. **可信度**：用验证集学习来源可靠度（truth discovery/EM）。
3. **冲突**：属性-specific τ、双方置信度门控、连续降权。
4. **数据**：30 例真实/脱敏案件 + 候选身份集合，验证“账号→真实身份”。
5. **工程**：LLM 解析、服务化、最小前端、30 例演示录屏。
