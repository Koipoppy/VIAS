# VIAS — 虚实融合身份推断

> **仓库内容**：`vias_pilot/` 是 pilot 归档；`vias_system/` 是完全体 FastAPI 服务（证据接入、可信度、贝叶斯、冲突、解释、候选身份层）。

小规模、端到端的可行性验证：把 SMP 2016 中文文本与 AFAD-Lite 亚洲人脸配对成 50 个多模态样本，按项目技术路线跑通：

```
多模态解析 → 统一证据单元 → 六维可信度 → 加权贝叶斯融合 → 冲突处理 → 可解释
```

> **这是 pilot，不是生产系统。** 文本用户与人脸是同年龄、同性别的**配对**，不是同一个真实人；目标变量是性别/年龄段，不是真实身份。

> **仓库范围**：只包含代码、Schema、测试、图表、环境配置；所有报告统一归档在工作区根部的 `../报告归档/`。原始数据、30 例案件包中的 `text`/`face`/`evidence`/`result` 文件、模型和证据 JSONL 因隐私、许可和体积原因**不推送**；完整本地包见 [DELIVERY.md](../报告归档/03_2026-11-25_Pilot定稿与验收/DELIVERY.md)。

## PAN 2018 全量多模态评估（2026-11-26）

在 PAN 2018 Author Profiling 官方训练/测试集全量（EN/ES/AR，7,500 训练 / 5,100 测试用户）上评估文本、图像、融合与冲突处理：

- 报告：[`../报告归档/05_2026-11-26_PAN2018全量评估/PAN2018_Multimodal_Gender_Report_2026-11-26.pdf`](../报告归档/05_2026-11-26_PAN2018全量评估/PAN2018_Multimodal_Gender_Report_2026-11-26.pdf)
- 复现脚本：`src/pan18_prepare.py`、`src/pan18_text_model.py`、`src/pan18_text_predict.py`、`src/pan18_image_model.py`、`src/pan18_fusion_eval.py`
- 数据不进入公共仓库；原始 zip/处理中间件在 `data/` 下。
- 关键结果：文本 0.792、图像 0.573、堆叠融合 0.795；图像是弱信号，等权/可信度融合反而降到 0.710。

## PAN 2018 Embedding Concat 融合（2026-11-27）

按 [`Fusion_Strategy_Critique_2026-11-27.pdf`](../报告归档/06_2026-11-27_融合诊断与Concat/Fusion_Strategy_Critique_2026-11-27.pdf) 的 P1a 路线，把融合从决策层前移到表示层：XLM-R 对 100 条推文做 4×512 token 分块编码，ViT 对最多 10 张资料图取均值；两侧各投影到 256 维后拼接为 512 维，再接 MLP 端到端训练。官方测试集只在最终评估时打开。

- 报告：[`../报告归档/06_2026-11-27_融合诊断与Concat/PAN2018_Embedding_Concat_Fusion_2026-11-27.pdf`](../报告归档/06_2026-11-27_融合诊断与Concat/PAN2018_Embedding_Concat_Fusion_2026-11-27.pdf)（[Markdown 版](../报告归档/06_2026-11-27_融合诊断与Concat/PAN2018_Embedding_Concat_Fusion_2026-11-27.md)）
- 四种融合方式对比：[`../报告归档/07_2026-11-27_四种融合对比/PAN2018_Four_Fusion_Comparison_2026-11-27.pdf`](../报告归档/07_2026-11-27_四种融合对比/PAN2018_Four_Fusion_Comparison_2026-11-27.pdf)（[Markdown 版](../报告归档/07_2026-11-27_四种融合对比/PAN2018_Four_Fusion_Comparison_2026-11-27.md)）
- 脚本：`src/pan18_joint_data.py`、`src/pan18_joint_fusion.py`、`src/make_joint_fusion_figures.py`、`src/build_joint_fusion_report.py`、`src/make_four_fusion_figures.py`、`src/build_four_fusion_report.py`、`src/eval_fusion_attribution.py`
- 划分：fit 6,000 / calib 1,500 / test 5,100；fit 与 calib 来自官方训练集，测试集全程隔离。
- 关键结果：文本 XLM-R 0.8296、十图图像 0.7565、TF-IDF 0.7904、最优 late fusion 0.8600；embedding concat（从零训练 + 逐用户门控）0.8549，相对文本 +2.53 个百分点（McNemar p<0.001），相对 late fusion −0.51 个百分点（p=0.272，统计上持平）；两者 logit 集成 0.8698（相对 late +0.99，p=0.002）。打乱图像对照 0.8153，完整消融见报告。

## 音频多模态测试（2026-11-28）

用 CREMA-D 构建音频+人脸同人的 232 例测试集（58 位演员，20–40 岁），并用 SMP 2016 pilot 文本做同性别/同年龄段的合成配对；按演员隔离 fit 112 / calib 60 / test 60。

- 数据集卡：[`../报告归档/08_2026-11-28_音频多模态测试/AUDIO_MULTIMODAL_DATASET_2026-11-28.md`](../报告归档/08_2026-11-28_音频多模态测试/AUDIO_MULTIMODAL_DATASET_2026-11-28.md)
- 报告：[PDF](../报告归档/08_2026-11-28_音频多模态测试/AUDIO_MULTIMODAL_FUSION_REPORT_2026-11-28.pdf)（[Markdown](../报告归档/08_2026-11-28_音频多模态测试/AUDIO_MULTIMODAL_FUSION_REPORT_2026-11-28.md)）
- 脚本：`src/audio_prepare_crema.py`、`src/audio_extract_audio.py`、`src/audio_extract_face.py`、`src/audio_build_cases.py`、`src/audio_train_heads.py`、`src/audio_eval_final.py`
- 关键结果：音频性别校准后 1.000（表演语音易任务）；音频年龄 zero-shot 原始 MAE≈14.7 岁→线性校准 4.02；文本年龄 3.63；文本+人脸融合年龄桶 0.500；音频缺失时文本+人脸性别 0.883。
- 融合方案对比：[PDF](../报告归档/09_2026-11-28_音频融合方案对比/AUDIO_FUSION_SCHEMES_REPORT_2026-11-28.pdf)（[Markdown](../报告归档/09_2026-11-28_音频融合方案对比/AUDIO_FUSION_SCHEMES_REPORT_2026-11-28.md)）

## 结果速览

| 方法 | 性别 Acc | 年龄 Acc | 年龄 MAE |
|---|---|---|---|
| 文本单模态 | 0.780 | 0.360 | 4.62 |
| 人脸单模态 | 0.960 | 0.460 | 6.36 |
| 融合（可信度） | **0.960** | **0.500** | **3.94** |
| 融合 + 冲突处理 | 0.920 | 0.500 | 3.98 |

- 注入 10 例矛盾证据：冲突检出 **9/10（90%）**，干净集误报 **7/50（14%）**。
- 冲突处理把注入子集年龄准确率从 0.30 提到 **0.40**，但干净集性别从 0.96 降到 0.92。
- 详细报告：[`../报告归档/02_2026-11-24_Pilot设计/pilot_report.md`](../报告归档/02_2026-11-24_Pilot设计/pilot_report.md)
- 设计说明：[`../报告归档/02_2026-11-24_Pilot设计/pilot_design_note.md`](../报告归档/02_2026-11-24_Pilot设计/pilot_design_note.md)

## 10 月计划补齐情况

| 计划项 | 状态 |
|---|---|
| 需求调研 / 公开数据可得性 | ✅ 公开数据集已摸底/核验；示范单位脱敏样本待确认 |
| 原型边界（文本、图像先行） | ✅ text+image 已实现；audio/behavior 预留 stub |
| 技术方案 / 数据规范 | ✅ v1.0 定稿 | `../报告归档/03_2026-11-25_Pilot定稿与验收/technical_scheme_and_data_spec_v1.0.md` |
| 统一证据单元 Schema | ✅ v1.0 冻结（项目内） | `src/schema.py` + JSON Schema + 示例 |
| 30 例样例案件 | ⚠️ **30 个合成案件包**（20 干净 + 10 注入），不是真实案件 |
| 开发环境 | ✅ pilot Python 环境；`deploy/` 提供项目级 compose |

新增交付物：

- [`cases_synthetic/`](cases_synthetic/) — 30 个合成案件包（`case.json`、`text.txt`、`face.jpg`、`evidence_units.jsonl`、`result.json`、`case_card.md`），含 20 干净 + 10 注入冲突；**明确标注 synthetic，不是真实案件**。
- [`src/parsers/`](src/parsers/) — audio/behavior 预留接口（`audio_stub.py`、`behavior_stub.py`），自检 `src/check_stubs.py`。
- [`schema/`](schema/) — `evidence_unit.schema.json`、`evidence_unit.example.json`（Schema v1.0）。
- [`../报告归档/03_2026-11-25_Pilot定稿与验收/technical_scheme_and_data_spec_v1.0.md`](../报告归档/03_2026-11-25_Pilot定稿与验收/technical_scheme_and_data_spec_v1.0.md) — 技术方案与数据规范 v1.0（含 22 字段表、枚举、约束、版本规则）。
- [`../报告归档/02_2026-11-24_Pilot设计/evidence_unit_v0.1_review.md`](../报告归档/02_2026-11-24_Pilot设计/evidence_unit_v0.1_review.md) — v0.1 历史评审稿。
- [`deploy/`](deploy/) — PostgreSQL 14 + MinIO + Redis 的 Docker Compose、初始化 SQL、环境说明。

## 快速开始

```bash
cd vias_pilot

# 0. 依赖
pip install -r requirements.txt

# 1. SMP 2016：解析标签 + 微博，划分 train/pilot
python src/prepare_smp.py

# 2. 构建 50 个多模态样本（需要 AFAD-Lite 采样图）
python src/build_cases.py

# 3. 文本模型（TF-IDF + LogisticRegression）
python src/text_model.py

# 4. 人脸模型（首次会下载 ViT 权重；需要网络/代理）
python src/face_model.py

# 5. 证据链 + 冲突 + 可解释
python src/pipeline.py --tau 0.50 --downweight 0.25

# 6. 图表
python src/make_figures.py

# 7. 音频/行为预留接口自检 + 30 例合成案件包
python src/check_stubs.py
python src/build_case_packages.py

# 8. 可选：项目级环境（需要 Docker）
cd deploy && cp .env.example .env && docker compose up -d
```

## 数据

| 数据 | 位置 | 说明 |
|---|---|---|
| SMP 2016 | `data/raw/smp/` | 3,200 标签用户；2,874 个 18–40 岁且有文本；pilot 50 |
| AFAD-Lite 采样图 | `data/raw/afad/sampled/` | 22/24/28/32/36 × 男/女 × 5 张 |
| 处理后 CSV | `data/processed/` | `smp_train.csv`、`smp_pilot.csv`、`pilot_cases.csv` |
| 30 例合成案件 | `cases_synthetic/` | 20 干净 + 10 注入；**synthetic，不是真实案件** |

数据调研清单：[`../报告归档/01_2026-11-16_数据调研/性别年龄推断数据集清单_2026-11-16.xlsx`](../报告归档/01_2026-11-16_数据调研/性别年龄推断数据集清单_2026-11-16.xlsx)（27 个公开数据集、链接、可达性、大小）。

AFAD-Lite 原始归档约 865 MB（27 个分卷），本仓库保留 **60 张采样图**（pilot 用其中 50 张）；重采样说明见 `data/raw/afad/README.txt` 与 `src/extract_afad_samples.py`。

## 输出

| 文件 | 内容 |
|---|---|
| `outputs/evidence_units_clean.jsonl` | 干净场景证据单元 |
| `outputs/evidence_units_injected.jsonl` | 注入场景证据单元 |
| `outputs/case_results_clean.jsonl` | 干净场景逐案结果、后验、Top 证据 |
| `outputs/case_results_injected.jsonl` | 注入场景逐案结果、冲突日志 |
| `outputs/evidence_graphs/*.json` | 证据链图数据 |
| `outputs/metrics.json` | 全部指标 |
| `outputs/figures/*.png` | 图表 |
| `reports/text_model_metrics.json` | 文本模型指标 |
| `reports/face_model_metrics.json` | 人脸模型指标 |
| `outputs/pan18_joint/metrics.json` | embedding concat 全量指标与检验 |
| `outputs/pan18_joint/pan18_joint_*.png` | concat 方法对比、分语言、结构图 |

## 目录结构

```
vias_pilot/
├── src/
│   ├── schema.py                 # 证据单元 Pydantic 模型
│   ├── credibility.py            # 六维可信度
│   ├── fusion.py                 # 加权融合 + 冲突处理
│   ├── pipeline.py               # 端到端流水线
│   ├── prepare_smp.py            # SMP 解析/划分
│   ├── build_cases.py            # 多模态配对
│   ├── build_case_packages.py    # 30 例合成案件包
│   ├── extract_afad_samples.py   # AFAD 采样
│   ├── text_model.py             # 文本模型
│   ├── face_model.py             # 人脸模型
│   ├── make_figures.py           # 图表
│   ├── check_stubs.py            # audio/behavior 接口自检
│   └── parsers/
│       ├── audio_stub.py         # 音频预留接口
│       └── behavior_stub.py      # 行为日志预留接口
├── schema/                       # JSON Schema + 示例
├── cases_synthetic/              # 30 例合成案件包（20 干净 + 10 注入）
├── deploy/                       # PG14 + MinIO + Redis compose
├── data/
│   ├── raw/                      # SMP 原始文件、AFAD 采样图
│   └── processed/                # 训练/ pilot CSV
├── models/                       # 文本模型（joblib）
├── outputs/                      # 证据单元、结果、指标、图
└── reports/                      # 指标 JSON/运行日志（报告已移至 ../报告归档）
```

## 关键限制

1. 50 例 pilot + 30 例合成案件包，都是合成配对，不是真实案件/真实身份；指标只作基线。
2. 人脸模型为公开预训练 ViT，未在 AFAD 上微调；年龄输出粗。
3. 文本年龄模型弱（0.36），自述年龄覆盖仅 22%，自述性别覆盖 0%。
4. 可信度为规则分，未学习/校准；冲突处理在干净集有 14% 误报。
5. 未覆盖音频、行为日志、LLM 解析、候选身份集合和前端。

## 下一步

- 年龄：AFAD/MegaAge 微调、年龄回归、概率校准。
- 文本：中文 BERT、改进自述抽取。
- 可信度：truth discovery / 来源可靠度学习。
- 冲突：属性-specific τ、置信度门控、连续降权。
- 数据：30 例真实/脱敏案件，加入候选身份，验证“账号→真实身份”。
- 工程：LLM 解析、服务化、最小前端、30 例演示。
