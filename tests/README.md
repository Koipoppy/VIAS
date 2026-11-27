# 验收测试

`test_october_acceptance.py` 是“10 月计划”的结构化验收脚本，检查 67 项：数据清单、原型边界、技术方案、Schema、30 例案件包、开发环境、技术路线。

## 运行

```bash
cd vias_pilot
python tests/test_october_acceptance.py
```

## 前置数据

脚本需要完整的**本地**数据包：

- `data/processed/pilot_cases.csv` 等处理后 CSV
- `cases_synthetic/SYN-*/` 下的案件文件
- `outputs/metrics.json`、`outputs/evidence_units_*.jsonl`

公开 GitHub 仓库**不包含**这些数据（隐私/许可/体积原因）；在全新 clone 上运行会在 preflight 停止，并提示缺少本地数据包。完整本地包见 `../报告归档/03_2026-11-25_Pilot定稿与验收/DELIVERY.md`。
