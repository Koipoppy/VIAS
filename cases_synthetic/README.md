# 30 例合成样例案件包（pilot）

> ⚠️ **这是 30 例“合成”案件，不是 30 例真实案件。**  
> 文本用户与人脸按“同年龄、同性别”配对，但不是同一个真实人；目标是属性（性别/年龄段），不是真实身份。  
> 真实案件仍需示范单位脱敏样本，本包只能作为 10 月计划“30 例样例案件”的**结构与流程占位**。

> **仓库说明**：GitHub 仓库只保留本 README、`manifest.csv`、`DATA_NOTICE.md`；`SYN-*/` 下的 `text.txt`、`face.jpg`、`evidence_units.jsonl`、`result.json` 在本地，不推送。

## 选样

- 从 50 个 pilot 样本中，按年龄（22/24/28/32/36）× 性别（男/女）各取 3 例。
- 每个格子取第 1、3、5 例；第 5 例正好是 10 个注入冲突案例之一。
- 结果：**30 例 = 20 例干净 + 10 例注入冲突**。

## 目录结构

```
cases_synthetic/
├── README.md                 # 本文件
├── DATA_NOTICE.md            # 数据来源与使用限制
├── manifest.csv              # 30 例清单与指标
├── SYN-001/
│   ├── case.json             # 案件元数据、ground truth、合成标记
│   ├── case_card.md          # 一页案件卡：证据、后验、冲突、Top 证据
│   ├── text.txt              # 脱敏后的中文文本（≤5,000 字符）
│   ├── face.jpg              # 配对的 AFAD-Lite 人脸
│   ├── evidence_units.jsonl  # 统一证据单元（含可信度/冲突标记）
│   └── result.json           # 模型结果：预测、后验、冲突日志、Top 证据
└── ...
```

## 这个 30 例子集上的结果

| 指标 | 值 |
|---|---|
| 性别正确 | **26 / 30（86.7%）** |
| 年龄段正确 | **16 / 30（53.3%）** |
| 注入冲突案例 | 10 |
| 冲突检出 | 见 `manifest.csv` 的 `conflict_detected` |

> 与 50 例全量结果（性别 0.92、年龄 0.50）略有差异，因为这里只取了 30 例，且包含 10 例注入冲突。指标只作 pilot 基线。

## 怎么用

1. 先看 `manifest.csv`，按 `injected_conflict`、`gender_correct`、`age_correct` 筛案例。
2. 打开某个 `SYN-xxx/case_card.md`，看证据、后验、冲突和 Top 证据。
3. `evidence_units.jsonl` 可直接喂给 `src/pipeline.py` 的融合/可解释模块。
4. `result.json` 是 `fused_conflict` 方法的逐案输出。

## 使用限制

- 仅用于内部 pilot 验证；**不得**当作真实案件、不得对外分发。
- 文本已做基础脱敏（URL/@/邮箱/电话/证件号/长数字），但不保证完全匿名。
- 人脸来自 AFAD-Lite（学术研究用途）；不要发布或再分发。
- 详见 [`DATA_NOTICE.md`](DATA_NOTICE.md)。
