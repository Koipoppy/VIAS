# VIAS 完全体服务质量报告

**日期**：2026-11-25  
**范围**：`vias_system/` 服务层、API、Schema、可信度、融合、冲突、解释  
**测试**：36 个 unittest 用例（14 core + 22 API）  
**结论**：服务链路可运行；修复了 8 个实现问题；补齐候选身份层；冲突/可信度默认参数在 pilot 上有可复现的提升；年龄模型仍是主要瓶颈。

---

## 1. 测试矩阵

```bash
cd vias_system
python -m unittest discover -s tests -v
```

| 类别 | 覆盖 |
|---|---|
| Schema | 23 字段、示例、概率和、额外字段、JSON Schema 契约 |
| 可信度 | 范围、stub profile、PII 脱敏 |
| 融合 | 后验归一、weight_power 影响、相同分布冲突为 0、冲突降权 |
| 文本解析 | 内置模型概率、自述正则 |
| API | 五接口端到端、multipart、JSON+base64、查询、409/400/401/404/413/422、API Key、未知字段、request id |
| 目标身份层 | target=identity、候选排序、身份冲突、身份解释、account pair 上下文 |
| 持久化 | 证据创建时间、冲突 persist=false 不写回 |

当前结果：**36/36 通过**。

另外做过一次真实 uvicorn 冒烟：

```text
upload ×2 → evaluate → bayes → conflict → bayes → explain
```

全部 200；冲突降权与解释叙述正常。

---

## 2. 发现并修复的问题

| # | 问题 | 影响 | 修复 |
|---:|---|---|---|
| 1 | `K=1−Σpq` 对相同分布也非零 | 一致证据被误判冲突，clean FPR 偏高 | 改为 `TV=0.5Σ|p−q|`，相同分布为 0 |
| 2 | 两个属性冲突时后写覆盖先写 | gender/age 同时处理会丢失一侧的降权/一致性 | 用 working copy 顺序处理，最后统一写回 |
| 3 | `max_upload_bytes` 显式值被环境默认覆盖 | 测试/部署无法收紧上传上限 | Settings 只在未显式设置时读环境 |
| 4 | Pydantic 请求模型未知字段被忽略 | 参数白名单名存实亡 | `ConfigDict(extra="forbid")` |
| 5 | 自述正则漏掉“我今年24岁，是男生” | 自述覆盖偏低 | 增加跨年龄短句模式 |
| 6 | 文本模型 content 缺少统一 `model` 键 | 下游难以识别解析器 | 补 `model` 字段 |
| 7 | GET 证据的 `created_at` 是响应时间 | 时间不可追溯 | 从 evidence 表读取创建时间 |
| 8 | `src/vias` 布局后 `project_root` 多取了一层 | 默认 DB/模型路径错误 | 改为 `parents[2]`，加 `ConfigTests` |

---

## 3. 效果改进（pilot 50 clean + 10 injected）

**改进配置**：

- `weight_power = 1.5`（可信度对比度）
- `tau_gender = 0.65`，`tau_age = 0.50`
- TV 冲突 + 置信度加权

| 指标 | 旧 D-S（τ=.50, power=1） | TV 基线（τ=.50, power=1） | 新默认（TV, τg=.65/τa=.50, power=1.5） |
|---|---:|---:|---:|
| clean 性别 Acc | 0.92 | 0.92 | **0.94** |
| clean 年龄 Acc | 0.50 | 0.50 | 0.50 |
| clean 年龄 MAE | 3.98 | 3.94 | **3.94** |
| clean 任何冲突率 | 0.14 | 0.10 | **0.06** |
| injected 性别 Acc | 0.80 | 0.80 | 0.80 |
| injected 年龄 Acc | 0.40 | 0.30 | **0.40** |
| injected 检出率 | 0.90 | 0.90 | 0.90 |

**解释**：

- 旧 `1−Σpq` 把一致证据也判为冲突，是 clean 误报高的根因之一；TV 修正把 clean 冲突率从 14% 降到 10%，阈值/对比度再降到 6%。
- `weight_power=1.5` 让高可信来源在融合中更有话语权；在无冲突场景把注入子集年龄 0.30→0.40，抵消 TV 修正带来的年龄回退。
- 属性分开阈值：gender 更严格（0.65）降低误报，age 保留 0.50 避免把真实年龄冲突全部压掉。
- 为区分中间步骤：TV 修正但未调参（power=1, τ=.50）的 clean 冲突率为 10%；旧 D-S 为 14%；最终新默认为 6%。
- 样本只有 50+10，以上是**工程验证**，不是统计显著性结论。

复现：

```bash
cd vias_system
python scripts/benchmark_conflict.py
```

输出：`docs/conflict_benchmark.json`。

### 3.1 目标身份层 pilot

项目书最终目标是候选身份排序，不是性别/年龄。新增 `target=identity` 后，用 50 个 pilot 案件构造“1 真候选 + 4 干扰候选”的合成候选集：

- Accuracy@1 = **0.64**
- Recall@3 = **0.90**
- 这是属性兼容打分（性别/年龄）的 pilot 结果，不是真实人脸/姓名/关系证据；
- 说明目标层链路可用，但不能据此声称项目书 80% 指标。

### 3.2 未采纳的尝试：换文本年龄模型

尝试把文本年龄模型从 LogisticRegression 换成 SGD/class-weight（pilot 原始准确率 0.36→0.42），但融合后的 clean 年龄 0.50→0.46、注入年龄 0.30→0.30，说明单模型准确率提升没有转化为校准良好的证据。暂不替换；下一步应先做概率校准/温度缩放，再比较融合收益。

---

## 4. 架构与仓库整理

已完成：

- `app/` → 标准 `src/vias/` 包；
- 增加 `pyproject.toml`，支持 `pip install -e .`；
- 服务与 pilot 边界明确：pilot 冻结为历史基线，服务为 canonical；
- `docs/ARCHITECTURE.md` 记录分层、依赖规则、ADR、扩展点、技术债；
- `docs/VIAS_Full_System_API_Solution.md` 同步 API、默认阈值、weight_power；
- 测试从 10 个扩到 36 个；新增 `tests/test_core.py`；
- 增加 `scripts/benchmark_conflict.py`，让改进可复现；
- **依赖锁定（2026-11-27）**：`scikit-learn` 固定为 `>=1.6,<1.7`，并用 1.6.1 重训 `models/text_*.joblib`。此前模型由 sklearn 1.9 生成、被 1.6 加载，`multi_class` 属性缺失导致 3 个用例失败；重训后 36/36 通过。
- **双副本同步（2026-11-27）**：`vias_pilot/vias_system` 为 git 源码，工作区根部 `vias_system/` 为运行时副本；`scripts/sync_system_copy.py --check` 校验一致，`models/`、`var/` 不参与同步。

---

## 5. 仍暴露的问题与下一步突破点

按优先级：

1. **年龄模型（最高）**：clean 年龄 0.50、MAE 3.94；注入子集 0.40。下一步用 AFAD/MegaAge 微调 + 年龄回归/序数分类 + 概率校准；这是比调阈值更大的增量。
2. **人脸模型接入**：默认关闭；生产应内置权重、缓存、GPU/CPU 批处理，并做质量门控。
3. **可信度学习**：当前是规则分；用验证集做 source reliability/EM 或温度校准，让 `weight_power` 不再是手工超参。
4. **冲突策略**：注入年龄检出仅 50%；需要区分“自然分歧”和“恶意注入”，引入来源先验/时间一致性，而不是只看分布距离。
5. **异步与规模**：同步 API、SQLite 单写者；批量案件应加 jobs + PostgreSQL + MinIO。
6. **内容子 Schema**：`content` 仍是开放对象；下一步按 modality 做 discriminated union，减少脏字段。
7. **安全/运维**：API Key 只是最低限度；生产需网关鉴权、限流、审计、Prometheus 指标。
8. **前端/运营**：当前只有 Swagger；案件工作台、证据链可视化、人工复核流是下一阶段。

---

## 6. 结论

服务侧已经具备“可运行、可持久化、可审计、可测试”的完全体形态；本次整理解决了目录/打包/测试/参数校验和冲突语义问题，并用 pilot 数据验证了默认参数改进。剩余最大瓶颈不在 API，而在年龄模型精度与真实数据/业务闭环。
