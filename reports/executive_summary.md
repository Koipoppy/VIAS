# Pilot 一页纸（给项目负责人）

**日期**：2026-11-24  
**范围**：性别 / 年龄段代理目标；文本 + 人脸两种模态；50 个合成多模态样本  
**代码与结果**：`vias_pilot/`

## 做了什么

用 SMP 2016 中文微博（2,874 个 18–40 岁带标签用户）训练文本性别/年龄模型，用 AFAD-Lite 亚洲人脸跑预训练人脸性别/年龄模型；按“同年龄、同性别”配对出 50 个多模态样本，跑通：

**证据解析 → 统一证据单元 → 六维可信度 → 加权贝叶斯融合 → 冲突注入/处理 → Top-k 证据 + 证据链图**

## 关键结果

| 方法 | 性别 Acc | 年龄 Acc | 年龄 MAE |
|---|---|---|---|
| 文本单模态 | 0.780 | 0.360 | 4.62 |
| 人脸单模态 | 0.960 | 0.460 | 6.36 |
| **融合（可信度）** | **0.960** | **0.500** | **3.94** |
| 融合 + 冲突处理 | 0.920 | 0.500 | 3.98 |

- 注入 10 例矛盾证据：冲突检出 **9/10（90%）**；干净集误报 **7/50（14%）**。
- 冲突处理把注入子集年龄准确率从 **0.30 → 0.40**，但干净集性别从 **0.96 → 0.92**。
- 年龄是主瓶颈：人脸年龄模型粗（多数输出 20–29），文本年龄弱（0.36），自述年龄覆盖仅 22%。

## 结论

1. 五段链路能端到端跑通；性别已达到可用基线（0.96）。
2. 年龄只有 0.50，不能作为最终结论；下一版优先换/微调年龄模型、做概率校准。
3. 规则式六维可信度区分度不足（可信度加权与等权结果完全一致），需要学习式来源可靠度。
4. 冲突处理能检出 90% 注入冲突，但会误伤干净样本；需要属性-specific 阈值、置信度门控和连续降权。
5. 当前是“属性代理”pilot，不是“账号→真实身份”；真实身份需要 30 例脱敏样本和候选身份集合。

## 10 月计划补齐

| 计划项 | 状态 |
|---|---|
| 需求调研 / 公开数据可得性 | ✅ 公开数据；示范单位脱敏样本待确认 |
| 原型边界（文本、图像先行） | ✅ text+image；audio/behavior 预留 stub |
| 技术方案 / 数据规范 | ✅ v1.0 定稿（`technical_scheme_and_data_spec_v1.0.md`） |
| 统一证据单元 Schema | ✅ v1.0 冻结（项目内）；22 字段 + JSON Schema + 示例 |
| 30 例样例案件 | ⚠️ 30 个**合成**案件包（20 干净 + 10 注入），不是真实案件 |
| 开发环境 | ✅ pilot Python 环境；`deploy/` 提供项目级 compose |

## 交付物

- 报告：`reports/pilot_report.md`
- 设计说明：`reports/pilot_design_note.md`
- 技术方案与数据规范：`reports/technical_scheme_and_data_spec_v1.0.md`
- v0.1 历史评审稿：`reports/evidence_unit_v0.1_review.md`
- Schema v1.0：`schema/evidence_unit.schema.json`、`schema/evidence_unit.example.json`
- 指标：`outputs/metrics.json`
- 逐案结果：`outputs/case_results_clean.jsonl`、`outputs/case_results_injected.jsonl`
- 证据单元：`outputs/evidence_units_clean.jsonl`、`outputs/evidence_units_injected.jsonl`
- 图表：`outputs/figures/`
- 30 例合成案件包：`cases_synthetic/`（20 干净 + 10 注入）
- Schema：`schema/evidence_unit.schema.json`、`schema/evidence_unit.example.json`（v1.0）
- 预留接口：`src/parsers/audio_stub.py`、`src/parsers/behavior_stub.py`
- 项目级环境：`deploy/docker-compose.yml` + `deploy/README.md`
- 代码：`src/`（schema、credibility、fusion、pipeline、text_model、face_model、make_figures、build_case_packages）

## 建议下一步（按优先级）

1. **年龄模型**：AFAD/MegaAge 微调 + 年龄回归/更细分段 + 概率校准。
2. **可信度**：用验证集学习每个来源的可靠度（truth discovery/EM），替代拍脑袋规则。
3. **冲突**：属性-specific τ、双方置信度门控、连续降权。
4. **数据**：30 例真实/脱敏案件 + 候选身份集合，验证“账号→真实身份”。
5. **工程**：LLM 解析、服务化、最小前端、30 例演示录屏。
