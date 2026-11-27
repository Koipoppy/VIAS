# PAN 2018 Embedding Concat 融合（2026-11-27）

按 [`Fusion_Strategy_Critique_2026-11-27.pdf`](Fusion_Strategy_Critique_2026-11-27.pdf) 的 P1a 路线，把融合从决策层前移到表示层：XLM-R 对 100 条推文做 4×512 token 分块编码，ViT 对最多 10 张资料图取均值；两侧各投影到 256 维后拼接为 512 维，再经逐用户门控与 MLP。官方测试集只在最终评估时打开。

## 结论

- **concat 相对文本单模态显著提升**：0.8543 vs 0.8296（+2.47 个百分点，McNemar p<0.001，bootstrap 95% CI [+1.55, +3.37]）。
- **concat 与最优 late fusion 统计上持平**：0.8600 vs 0.8543（−0.57 个百分点，McNemar p=0.204，CI [−1.43, +0.29]）。
- **打乱图像对照回落到文本水平**：0.8255，说明增益来自图像信息，而不是更多参数或初始化。
- P1a 验收通过；下一步可进入 P1b（cross-attention）做受控对比。

## 数据与划分

PAN 2018 Author Profiling 官方训练/测试集（EN/ES/AR，7,500 / 5,100 用户）。

| 划分 | 人数 | 用途 |
|---|---:|---|
| fit | 6,000 | 训练文本编码器、图像头、融合头 |
| calib | 1,500 | 早停、温度缩放、late fusion 堆叠 |
| test | 5,100 | 只做最终评估 |

## 结果（官方测试集 n=5,100）

| 方法 | Accuracy | Macro-F1 | AUC | ECE | Brier | Δ vs text (pp) | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 图像单模态（ViT 十图均值） | 0.6594 | 0.6355 | 0.7635 | 0.0214 | 0.2157 | −17.02 | <0.001 |
| 图像单模态（线性头） | 0.7565 | 0.7565 | 0.8362 | 0.0227 | 0.1653 | −7.31 | <0.001 |
| 文本单模态（XLM-R） | 0.8296 | 0.8296 | 0.9057 | 0.0101 | 0.1224 | — | — |
| 文本单模态（TF-IDF，参考） | 0.7904 | 0.7904 | 0.8652 | 0.0241 | 0.1488 | −3.92 | <0.001 |
| late fusion（校准对数意见池） | 0.8588 | 0.8588 | 0.9286 | 0.0126 | 0.1041 | +2.92 | <0.001 |
| late fusion（堆叠 LR） | 0.8600 | 0.8600 | 0.9283 | 0.0137 | 0.1038 | +3.04 | <0.001 |
| **embedding concat（门控）** | **0.8543** | **0.8543** | **0.9219** | 0.0162 | 0.1082 | **+2.47** | **<0.001** |
| concat + 打乱图像（对照） | 0.8255 | 0.8254 | 0.8848 | 0.0320 | 0.1339 | −0.41 | 0.261 |

概率先在校准集上做温度缩放；Δ 与检验使用校准后的硬预测。AUC 只反映排序，不受温度影响。

## concat 消融

| 变体 | Accuracy | AUC | ECE | Δ vs text (pp) |
|---|---:|---:|---:|---:|
| 从零训练（无文本初始化） | 0.8194 | 0.8970 | 0.0108 | −1.02 |
| 文本初始化，无门控 | 0.8486 | 0.9202 | 0.0173 | +1.90 |
| 文本初始化 + 图像后验特征 | 0.8390 | 0.9148 | 0.0538 | +0.94 |
| 文本初始化 + 门控（本方案） | 0.8543 | 0.9219 | 0.0162 | +2.47 |
| 门控 + 打乱图像 | 0.8255 | 0.8848 | 0.0320 | −0.41 |

两个关键设计：

1. **文本初始化**：concat 的文本编码器从文本单模态 checkpoint 初始化，再联合微调；从零训练会因图像分支的早期捷径而弱化文本编码器（0.8194）。
2. **逐用户门控**：`g = Sigmoid(W_g [z_t; z_i])`，`z_i' = g ⊙ z_i`，允许模型对弱图像样本压低图像权重；相对无门控提升 0.57 个百分点。

## 解释与局限

- 十图聚合把图像单模态从首图 0.573 提升到 0.7565，这是 late fusion 在本数据上变强的直接原因；11-27 诊断报告中的“late fusion 无法提升”结论基于首图 0.573 的旧设置，需要按新结果修正。
- 图像编码器仍是冻结的公开性别 ViT；未做人脸检测/质量门控，十图均值会把低质量图与合影噪声平均进来。
- 文本按 4×512 token 分块，覆盖中位用户但长尾仍被截断；更长上下文可能继续提升文本基线。
- 单一种子；bootstrap 只覆盖测试集抽样不确定性，不覆盖训练随机性。多 seed 均值与方差待补。
- PAN 2018 是账号级性别标签，不是真实身份；本结果不涉及候选身份层，不能外推到 80% 真实身份指标。

## 复现

```bash
python src/pan18_joint_data.py                 # manifest + 4×512 tokens + 十图特征
python src/pan18_joint_fusion.py image         # 图像单模态
python src/pan18_joint_fusion.py tfidf         # TF-IDF 参考
python src/pan18_joint_fusion.py text          # XLM-R 文本单模态
python src/pan18_joint_fusion.py concat --init-from models/pan18_joint/text.pt --gate
python src/pan18_joint_fusion.py concat --init-from models/pan18_joint/text.pt --gate --shuffle-images
python src/pan18_joint_fusion.py late
python src/pan18_joint_fusion.py evaluate
python src/make_joint_fusion_figures.py
python src/build_joint_fusion_report.py
```

数据与模型权重不进入公共仓库；脚本默认从 `data/processed/pan18` 与 `models/pan18_joint` 读取。
