# VIAS 系统架构说明

**版本**：1.0（2026-11-25）  
**适用**：`vias_system/` 完全体服务  
**配套**：`docs/VIAS_Full_System_API_Solution.md`、`docs/QUALITY_REPORT.md`

---

## 1. 仓库边界

```text
VIAS/
├── vias_pilot/            # git 仓库；pilot 归档 + vias_system 源码
│   └── vias_system/       # git 跟踪的服务源码（source of truth）
├── vias_system/           # 工作区运行时副本：models/、var/ 只在这里
├── vias_pilot_handoff_2026-11-24.zip
├── vias_system_handoff_2026-11-25.zip
└── 性别年龄推断数据集清单_2026-11-16.xlsx
```

- **`vias_pilot` 是冻结的历史基线**：保留 50 例结果、报告和实验脚本，便于复核；不再作为运行时依赖。
- **`vias_pilot/vias_system` 是服务源码的唯一 git 来源**；工作区根部的 `vias_system/` 是运行时副本，内置模型与 SQLite 只放在这里。
- 两边代码必须一致：修改后运行 `python scripts/sync_system_copy.py`（`--check` 只校验）。`models/`、`var/`、`__pycache__` 不参与同步。
- pilot 与服务不共享导入路径，避免“改 pilot 影响服务”或“改服务破坏历史实验”。代价是 core 有少量重复；这是有意的边界，不是隐性耦合。

---

## 2. 服务分层

```text
HTTP / FastAPI
   ↓
API 路由（参数解析、鉴权、错误映射）
   ↓
Service（evidence / inference / conflict / explain）
   ↓
Core（schema / credibility / fusion / text_parser / face_parser）
   ↓
Repository（SQLite JSON payload） + Storage（本地文件）
```

### 2.1 依赖规则

| 层 | 可以依赖 | 不可以依赖 |
|---|---|---|
| `api` | service、schema、errors | 直接 SQL、直接文件 IO |
| `services` | core、repository、storage | FastAPI Request/Response |
| `core` | 标准库、Pydantic | repository、API、文件系统（parser 除外） |
| `repository` | db、core schema | API、service |
| `storage` | 标准库 | API、service、DB |

### 2.2 目录

```text
vias_system/
├── src/vias/
│   ├── main.py             # create_app / 中间件
│   ├── config.py           # 环境配置
│   ├── errors.py           # ApiError + 统一错误体
│   ├── schemas.py          # API 请求/响应模型
│   ├── db.py               # SQLite schema/连接
│   ├── repository.py       # 案件/证据/推断/冲突/解释
│   ├── storage.py          # 原始文件、sha256、raw_ref
│   ├── api/                # health / evidence / inference
│   ├── services/           # 四个业务服务
│   └── core/               # schema / credibility / fusion / parsers
├── models/                 # 内置文本模型
├── schema/                 # JSON Schema v1.1 + 示例
├── tests/                  # 36 个单元/集成测试
├── docs/                   # 方案说明、架构、质量报告
├── scripts/                # 文档生成、pilot 基准复现
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

---

## 3. 数据生命周期

```text
upload
  ├─ 原始文件 → storage（local://... + sha256）
  ├─ 文本脱敏 → content
  ├─ 文本/人脸解析 → gender_probs/age_probs（可客户端传入）
  └─ EvidenceUnit v1.0 → evidence 表

evaluate
  └─ 六维可信度 → credibility JSON → evidence 表

bayes
  ├─ 读取案件证据
  ├─ ω = (weight × multiplier)^weight_power
  └─ 后验 + inference_id → inferences 表

conflict
  ├─ TV 冲突 + 置信度加权
  ├─ 低权证据降权
  └─ 冲突日志 → conflicts 表；persist=true 时回写 evidence

explain
  ├─ 读取 inference_id
  ├─ 逐条 leave-one-out
  └─ Top-k + 证据链 + 叙述 → explanations 表
```

**ID 规则**：`ev_*`、`inf_*`、`con_*`、`exp_*`；案件 ID 由调用方给出，`SYN-*` 表示合成案件。

---

## 4. 关键设计决策（ADR）

### ADR-1：同步 API，而不是任务队列

五个接口在本地模型/规则下耗时可控，直接返回结果更利于调试和集成。若接入大模型或批量解析，再加 `jobs` 表和 worker；API 保持路径不变，增加 `202 + job_id` 模式即可。

### ADR-2：SQLite + 本地文件作为默认后端

课题组内部/演示不需要外部依赖；Repository/Storage 已隔离，替换 PostgreSQL/MinIO 时 API 不变。

### ADR-3：Schema v1.1 冻结，pilot 不参与运行时

23 字段、`extra=forbid`、概率和/范围校验；pilot 的 JSONL 只作为历史数据。避免“pilot 改字段、服务悄悄坏掉”。

### ADR-4：人脸模型可选

ViT 模型体积大、启动慢；默认关闭，允许上游传概率。开启 `VIAS_ENABLE_FACE_MODEL=1` 后本地推理；这是部署选择，不是 API 语义差异。

### ADR-5：TV 冲突，而不是 `1−Σpq`

旧实现的 `K=1−Σpq` 对相同分布也非零（0.8/0.2 对自身得到 0.32），会把一致证据误判为冲突。现改为：
`TV=0.5Σ|p−q|`，`C=TV×min(max p,max q)`；相同分布为 0，冲突语义正确。

### ADR-6：可信度对比度

`weight_power=1.5` 对高可信证据做幂次增强；pilot 上在不降干净集指标的情况下，把注入子集年龄准确率 0.30→0.40。默认值可被请求覆盖。

### ADR-7：候选身份层不塞进属性层

性别/年龄是中间假设，不是最终目标。`identity_probs` 作为证据的可选字段，候选集通过 `target=identity` 请求传入；目标层打分支持“候选身份概率”与“属性兼容”两种模式。没有真实候选证据时只做 pilot 合成评测，不声称真实身份准确率。

### ADR-8：属性分开的冲突阈值

gender `τ=0.65`、age `τ=0.50`；pilot 上干净集任何冲突率 14%→6%，注入检出保持 90%。`tau` 显式传入时统一覆盖两属性，保留兼容性。

---

## 5. 扩展点

| 需求 | 改动点 | API 是否变化 |
|---|---|---|
| PostgreSQL | 新 `Repository` 实现 | 不变 |
| MinIO/S3 | `Storage.save` | `raw_ref` 改 `s3://` |
| 新人脸/年龄模型 | `core/face_parser.py` 或上游传概率 | 不变 |
| 异步批处理 | `jobs` 表 + worker + 202 | 增加可选模式 |
| 新模态 | `modality`/`source_type` 枚举 + parser | Schema 小版本升级 |
| 认证/限流 | 网关或中间件 | 不变 |
| 观测 | 中间件 + Prometheus | 不变 |

---

## 6. 已知技术债

1. `vias_pilot` 与 `vias_system` 的 core 有重复；pilot 已冻结，暂不合并。
2. SQLite 单写者；高并发需 PostgreSQL。
3. 同步 API；批量/大模型需任务队列。
4. 人脸模型未内置权重，默认关闭。
5. 年龄模型仍是 pilot 基线；这是当前精度瓶颈。
6. `content` 仍是开放对象，按 modality 的必填键写在文档里，JSON Schema 未做 discriminated union。
7. 没有速率限制/配额；生产需网关层加。
8. 没有前端；当前以 Swagger/API 为主。
