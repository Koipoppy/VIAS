# 项目书要求与本系统实现对照

**依据**：项目实施方案（新型网络社交场景虚实融合身份推断，子任务4）  
**实现**：`vias_system/`  
**口径**：pilot 级完整链路；真实身份数据/模型尚未接入的部分明确标注。

---

## 1. 项目书要求的技术流程

项目书把任务分成五段：

1. **多模态数据接入与解析**：文本/图像/音频，格式归一化、跨模态语义对齐、特征融合、结构化证据。
2. **证据链可靠性评估**：来源、权威性、客观性、时效性、完整性、一致性，单证据可信度与权重。
3. **层次贝叶斯推理**：证据层 → 事实/假设层 → 目标身份/结论层；加权后验。
4. **冲突处理与权重迭代**：冲突度量、矛盾校正、动态权重、异常证据剔除。
5. **可解释性**：线索影响分析、全局可信度分支、推理链路追溯。

**最终目标**：不是性别/年龄，而是“目标身份判定 + 候选身份后验排序”；性别/年龄只是中间属性层。

---

## 2. 实现对照

| 项目书要求 | 当前实现 | 状态 |
|---|---|---|
| 文本接入/解析 | `/api/evidence/upload`，UTF-8 文本、正则自述、TF-IDF+LR 性别/年龄模型 | ✅ pilot 基线 |
| 图像接入/解析 | 文件落盘；可选 ViT 人脸性别/年龄；也可由调用方传概率 | ✅ 可选 |
| 音频接入 | `audio_stub`；有 `asr_text` 时走文本模型形成属性证据，无 ASR 时 reserved | ⚠️ 样例链路 |
| 行为日志接入 | `behavior_stub`；解析事件数/类型/IP/设备，暂不产出身份概率 | ⚠️ 样例链路 |
| 格式归一化/统一 Token | 统一为 `EvidenceUnit`，而非 BPE/ViT/Whisper Token 序列 | ⚠️ 结构化替代 |
| 跨模态语义对齐 | 通过候选身份层/属性兼容做证据融合；未接 CLIP/Q-Former | ⚠️ 未接大模型 |
| 证据结构化 | 23 字段 EvidenceUnit，JSON Schema v1.1 | ✅ |
| 单证据可信度 | 六维：来源、权威、客观、时效、完整、一致 | ✅ |
| 可信度加权 | `ω=(weight×multiplier)^weight_power`，默认 1.5 | ✅ |
| 证据链综合可信度 | 逐证据 score/weight；可查询 | ✅ |
| 层次贝叶斯 | 证据层 → 性别/年龄假设层 → 候选身份层 | ✅ pilot |
| 目标身份后验 | `/api/inference/bayes` `target=identity` + `candidates` | ✅ pilot |
| 候选身份排序 | `results.identity.ranking`，按后验降序 | ✅ |
| 证据冲突度量 | TV 冲突：`0.5Σ|p−q|`，置信度加权 | ✅ |
| 权重迭代 | 低权证据 ×0.25，最多 3 轮，写回 | ✅ |
| 异常证据剔除 | 冲突降权；未做硬删除 | ⚠️ 软处理 |
| 线索影响分析 | leave-one-out 贡献、Top-k | ✅ |
| 全局可信度分支 | 证据链节点/边、置信度 | ✅ |
| 推理链路追溯 | `explanation_id`、`inference_id`、证据链 | ✅ |
| 账号对/候选集 | `target=identity` 内联 candidates；`account_a/account_b/platform` 作为上下文写入 inference | ⚠️ 部分 |
| 真实身份定位准确率 | 无真实账号对/真值；pilot 合成候选集 Accuracy@1=0.64、Recall@3=0.90 | ⚠️ 不能宣称 80% |

---

## 3. 目标层实现说明

### 3.1 两种候选打分模式

1. **identity 模式**：证据带 `identity_probs`（candidate_id → 概率），直接做加权对数意见池。
2. **attributes 模式**：候选带 `gender`/`age_bucket`，用证据的性别/年龄后验做兼容似然。  
   这是没有真实身份证据时的 pilot 回退，不代表真实人脸/姓名比对。

### 3.2 候选集

`POST /api/inference/bayes`：

```json
{
  "case_id": "CASE-001",
  "target": "identity",
  "candidates": [
    {"candidate_id": "cand_a", "gender": "female", "age_bucket": "25-29"},
    {"candidate_id": "cand_b", "gender": "male", "age_bucket": "30-34"}
  ],
  "weight_power": 1.5
}
```

输出：

```json
{
  "results": {
    "identity": {
      "prediction": "cand_a",
      "confidence": 0.82,
      "ranking": [
        {"candidate_id": "cand_a", "probability": 0.82},
        {"candidate_id": "cand_b", "probability": 0.18}
      ]
    }
  }
}
```

### 3.3 合成候选集评测

`scripts/evaluate_identity_pilot.py`：每个 pilot 案件构造 1 个真候选 + 4 个干扰候选（共 5 个），用属性兼容打分。

- Accuracy@1 = **0.64**
- Recall@3 = **0.90**
- 这是**合成候选集**，不是真实身份识别；不能作为 80% 指标证据。

---

## 4. 明确未完成/不能声称的部分

1. **真实身份数据**：没有账号对真值、姓名/证件/人脸同一性标注；不能计算真实身份准确率。
2. **LLM/CLIP/Whisper**：项目书描述的 BPE/ViT/Whisper/CLIP/Q-Former 未接入；当前是 TF-IDF/ViT 基线 + 结构化证据。
3. **音频/行为模型**：音频支持调用方提供 ASR 文本走属性模型；没有内置 ASR/声纹；行为只做事件特征，不产出身份概率。
4. **账号对建模**：候选集内联在请求中；`account_a/account_b` 只作为上下文，不参与同一性判定。
5. **异常证据剔除**：只做降权，没有硬删除/隔离。
6. **目标层评测**：只有合成候选集；需要真实案件/候选集才能评估项目书指标。

---

## 5. 结论

当前系统已经具备项目书五段链路的**服务化骨架**，并补上了候选身份层；但它是“**pilot 级完整链路**”，不是“真实身份定位生产系统”。  
要对外声称项目书指标，必须补真实账号对/候选集数据、真实身份证据模型和身份级评测。
