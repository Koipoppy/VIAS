# 10 月计划验收报告

**日期**：2026-11-24  
**标准**：需求调研与数据可得性摸底；原型边界（文本、图像先行，音频/行为预留）；技术方案与数据规范；统一证据单元 Schema；30 例样例案件；开发环境；技术路线。

## 总判定：**PASS_WITH_PARTIALS**

- 按 pilot 口径通过；有 4 项 PARTIAL（外部依赖/待评审）
- 检查项：{'PASS': 63, 'PARTIAL': 4}

## 测试矩阵

| 要求 | 检查 | 状态 | 证据/说明 |
|---|---|---|---|
| 1. 数据可得性 | 中文文本 行数 = 6 | PASS | actual=6 |
| 1. 数据可得性 | 人脸图像 行数 = 15 | PASS | actual=15 |
| 1. 数据可得性 | 多模态英文 行数 = 6 | PASS | actual=6 |
| 1. 数据可得性 | 数据集总数 >= 27 | PASS | actual=27 |
| 1. 数据可得性 | 每行有可达性 + 大小 | PASS | missing=0 |
| 1. 数据可得性 | 存在可点击链接 | PASS | hyperlinks=45 |
| 1. 数据可得性 | 示范单位脱敏样本 | PARTIAL | 清单已覆盖公开数据集；示范单位样本可得性待确认（外部依赖） |
| 2. 原型边界 | 文本预测 50 条 | PASS | rows=50 |
| 2. 原型边界 | 人脸预测 50 条 | PASS | rows=50 |
| 2. 原型边界 | 证据包含 text + image | PASS | modalities=['image', 'text'] |
| 2. 原型边界 | 音频/行为预留接口 | PASS | audio/behavior stubs return reserved EvidenceUnit |
| 3. 技术方案/数据规范 | 设计说明包含「标签体系」 | PASS |  |
| 3. 技术方案/数据规范 | 设计说明包含「六维可信度」 | PASS |  |
| 3. 技术方案/数据规范 | 设计说明包含「层次结构」 | PASS |  |
| 3. 技术方案/数据规范 | 设计说明包含「融合」 | PASS |  |
| 3. 技术方案/数据规范 | 设计说明包含「冲突」 | PASS |  |
| 3. 技术方案/数据规范 | 设计说明包含「可解释」 | PASS |  |
| 3. 技术方案/数据规范 | Schema 评审稿存在 | PASS | D:\Products\L1918实习工作\VIAS\vias_pilot\reports\evidence_unit_v0.1_review.md |
| 3. 技术方案/数据规范 | 评审稿含字段表 | PASS |  |
| 3. 技术方案/数据规范 | 评审稿含待评审问题 | PASS |  |
| 3. 技术方案/数据规范 | README 含 10 月计划补齐 | PASS |  |
| 4. 统一证据单元 Schema | 字段 schema_version | PASS |  |
| 4. 统一证据单元 Schema | 字段 raw_ref | PASS |  |
| 4. 统一证据单元 Schema | 字段 time_range | PASS |  |
| 4. 统一证据单元 Schema | 字段 language | PASS |  |
| 4. 统一证据单元 Schema | 字段 synthetic | PASS |  |
| 4. 统一证据单元 Schema | 字段 pii_redacted | PASS |  |
| 4. 统一证据单元 Schema | 必填字段 | PASS | [] |
| 4. 统一证据单元 Schema | modality 含 audio/behavior | PASS | typing.Literal['text', 'image', 'audio', 'behavior', 'metadata'] |
| 4. 统一证据单元 Schema | JSON Schema 可解析 | PASS | properties=22 |
| 4. 统一证据单元 Schema | 示例通过 Pydantic 校验 | PASS |  |
| 4. 统一证据单元 Schema | 输出证据全部可校验 | PASS | validated=442, bad=0 |
| 4. 统一证据单元 Schema | 定型/冻结 | PARTIAL | v0.1 已可运行；尚未团队评审/字段冻结 |
| 6. 开发环境 | requirements.txt | PASS | D:\Products\L1918实习工作\VIAS\vias_pilot\requirements.txt |
| 6. 开发环境 | 依赖 numpy | PASS |  |
| 6. 开发环境 | 依赖 pandas | PASS |  |
| 6. 开发环境 | 依赖 sklearn | PASS |  |
| 6. 开发环境 | 依赖 joblib | PASS |  |
| 6. 开发环境 | 依赖 pydantic | PASS |  |
| 6. 开发环境 | 依赖 matplotlib | PASS |  |
| 6. 开发环境 | 依赖 networkx | PASS |  |
| 6. 开发环境 | 依赖 torch | PASS |  |
| 6. 开发环境 | 依赖 transformers | PASS |  |
| 6. 开发环境 | 依赖 PIL | PASS |  |
| 6. 开发环境 | 文本模型文件 | PASS |  |
| 端到端 | 端到端 pipeline 运行 | PASS | exit=0；5"
    ],
    "injected_recovery_rate": 0.4
  },
  "coverage": {
    "n_cases": 50,
    "self_report_gender": 0.0,
    "self_report_age": 0.22,
    "text_gender_evidence": 1.0,
    "text_age_evidence": 1.0,
    "face_gender_evidence": 1.0,
    "face_age_evidence": 1.0
  }
}
saved outputs to outputs
 |
| 6. 开发环境 | pipeline 指标可用 | PASS | gender=0.96, age=0.5 |
| 端到端 | audio/behavior 接口自检 | PASS | exit=0；audio stub: SYN-001-A-STUB audio reserved 0.37
behavior stub: SYN-001-B-STUB behavior reserved 0.37
stubs OK
 |
| 端到端 | 30 例案件包重建 | PASS | exit=0；cases: 30 injected: 10
gender correct: 26 / 30
age correct: 16 / 30
output: cases_synthetic
 |
| 6. 开发环境 | docker-compose 三服务 | PASS | services=['postgres', 'minio', 'redis'] |
| 6. 开发环境 | initdb SQL | PASS |  |
| 6. 开发环境 | 项目级容器实际启动 | PARTIAL | 本机未安装 Docker；compose 已做 YAML 校验，未实际启动容器 |
| 5. 30 例样例案件 | 30 行 | PASS | rows=30 |
| 5. 30 例样例案件 | 20 干净 + 10 注入 | PASS | injected=10 |
| 5. 30 例样例案件 | 30 个案件目录 | PASS | dirs=30 |
| 5. 30 例样例案件 | 每案 6 个文件齐全 | PASS |  |
| 5. 30 例样例案件 | 每案证据 >=4 且可校验 | PASS | bad=0 |
| 5. 30 例样例案件 | 文本脱敏无 URL/邮箱/手机/证件 | PASS | pii_files=0 |
| 5. 30 例样例案件 | 文本长度 <= 5100 | PASS | too_long=0 |
| 5. 30 例样例案件 | 30 例均标记 synthetic | PASS | synthetic=30 |
| 5. 30 例样例案件 | 真实/脱敏案件 | PARTIAL | 当前 30 例为公开数据构造的合成样例；真实案件需示范单位脱敏样本（外部依赖） |
| 5. 30 例样例案件 | 案件包 README + 数据声明 | PASS |  |
| 7. 技术路线 | 融合/冲突/可解释函数存在 | PASS | weighted_fuse, conflict_handle, leave_one_out |
| 7. 技术路线 | 干净集性别 0.96 | PASS |  |
| 7. 技术路线 | 干净集年龄 0.50 | PASS |  |
| 7. 技术路线 | 冲突检出 0.90 | PASS |  |
| 7. 技术路线 | 干净误报 0.14 | PASS |  |

## 关键指标

| 指标 | 值 |
|---|---|
| 性别 Acc（融合可信度） | 0.960 |
| 年龄 Acc（融合可信度） | 0.500 |
| 年龄 MAE | 3.94 岁 |
| 冲突检出 | 90%（9/10） |
| 干净误报 | 14%（7/50） |

## 30 例案件包

- `cases_synthetic/`：30 个案件目录，20 干净 + 10 注入。
- 这 30 例是**合成案件**（公开数据 + 同年龄/同性别配对），不是真实案件。
- 30 例子集：性别 26/30，年龄段 16/30。

## 待补 / 外部依赖

1. **30 例真实/脱敏案件**：需示范单位提供样本并走审批；当前是合成占位。
2. **项目级容器实际启动**：本机未安装 Docker；compose 文件已做 YAML 校验。
3. **Schema 定型/冻结**：v0.1 可运行，但未团队评审。
4. **音频/行为模型**：只有预留接口，没有 ASR/声纹/行为序列模型。

## 复现

```bash
python tests/test_october_acceptance.py
```