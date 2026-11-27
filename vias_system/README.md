# VIAS 虚实融合身份推断系统（完全体 API）

从 `vias_pilot` 的脚本/JSONL 升级为可部署服务：证据接入、可信度评估、层次贝叶斯推断、冲突处理、可解释溯源五个正式接口。

> **不是真实身份系统**：输出性别/年龄段后验；pilot 的合成数据不进入默认库。真实案件需示范单位授权与审批。

## 快速启动

```bash
cd vias_system
pip install -e .
uvicorn vias.main:app --host 0.0.0.0 --port 8000
```

- Swagger: http://127.0.0.1:8000/docs
- OpenAPI: http://127.0.0.1:8000/openapi.json
- 健康检查: http://127.0.0.1:8000/api/health

默认 DB `var/vias.db`、上传目录 `var/uploads/`；首次启动自动建表。

## 五个接口

| 方法 | 路径 | 作用 |
|---|---|---|
| POST | `/api/evidence/upload` | 多模态证据接入（multipart 或 JSON） |
| POST | `/api/evidence/evaluate` | 六维可信度评估 |
| POST | `/api/inference/bayes` | 属性后验或候选身份后验（`target=identity`） |
| POST | `/api/evidence/conflict` | TV 冲突检测/降权 |
| POST | `/api/inference/explain` | 属性/身份推断的 LOO 贡献、Top-k 证据链、中文解释 |

辅助查询：

- `GET /api/cases/{case_id}`
- `GET /api/cases/{case_id}/evidence`
- `GET /api/evidence/{evidence_id}`
- `GET /api/inferences/{inference_id}`
- `GET /api/explanations/{explanation_id}`

## 最小调用示例

```bash
# 1. 文本证据（显式概率；也可不传，由内置模型推断）
curl -X POST http://127.0.0.1:8000/api/evidence/upload \
  -F "case_id=CASE-001" \
  -F "modality=text" \
  -F "source_type=text_model" \
  -F "source_name=tfidf_lr" \
  -F "text=我今年28岁，是一名女生。" \
  -F "gender_probs={\"male\":0.1,\"female\":0.9}" \
  -F "age_probs={\"18-24\":0.05,\"25-29\":0.8,\"30-34\":0.1,\"35-40\":0.05}"

# 2. 可信度
curl -X POST http://127.0.0.1:8000/api/evidence/evaluate \
  -H "Content-Type: application/json" -d '{"case_id":"CASE-001"}'

# 3. 贝叶斯
curl -X POST http://127.0.0.1:8000/api/inference/bayes \
  -H "Content-Type: application/json" -d '{"case_id":"CASE-001","attribute":"both"}'

# 4. 冲突
curl -X POST http://127.0.0.1:8000/api/evidence/conflict \
  -H "Content-Type: application/json" -d '{"case_id":"CASE-001","attribute":"both"}'

# 5. 解释（用第 3 步返回的 inference_id）
curl -X POST http://127.0.0.1:8000/api/inference/explain \
  -H "Content-Type: application/json" -d '{"inference_id":"inf_...","top_k":3}'
```

## 配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `VIAS_DB_PATH` | `var/vias.db` | SQLite 路径 |
| `VIAS_STORAGE_DIR` | `var/uploads` | 原始文件目录 |
| `VIAS_TEXT_MODEL_DIR` | `models` | 文本模型 |
| `VIAS_ENABLE_FACE_MODEL` | `0` | 本地跑 ViT 人脸模型 |
| `VIAS_FACE_DEVICE` | `cpu` | `cpu` / `cuda` |
| `VIAS_MAX_UPLOAD_MB` | `25` | 上传上限 |
| `VIAS_API_KEY` | 空 | 设置后 `/api/*` 需 `X-API-Key` |
| `VIAS_CORS_ORIGINS` | 空 | 逗号分隔 |

人脸模型默认关闭：图片可由调用方传入 `gender_probs/age_probs`；开启本地模型需安装 `torch/transformers`。

## Docker

```bash
cd vias_system
docker compose up --build
```

## 测试

```bash
cd vias_system
python tests/test_api.py
```

当前 36/36 通过：五接口端到端、target=identity 候选排序/身份冲突/身份解释、multipart/JSON+base64、内置文本模型、非法概率/先验、模态不匹配、无概率证据、API Key、上传超限、未知字段、查询与 404、冲突 persist=false。

## 文档

- [完整方案说明 Markdown](../../报告归档/04_2026-11-25_系统方案/VIAS_Full_System_API_Solution.md)
- [完整方案说明 DOCX](../../报告归档/04_2026-11-25_系统方案/VIAS_Full_System_API_Solution.docx)
- [项目书要求对照](../../报告归档/04_2026-11-25_系统方案/PROJECT_REQUIREMENTS.md)
- [架构说明](../../报告归档/04_2026-11-25_系统方案/ARCHITECTURE.md)
- [质量报告](../../报告归档/04_2026-11-25_系统方案/QUALITY_REPORT.md)
- [统一证据单元规范](../../报告归档/04_2026-11-25_系统方案/evidence_unit_spec_v1.1.md)
- [JSON Schema](schema/evidence_unit.schema.json)

## 目录

```
vias_system/
├── src/vias/
│   ├── main.py
│   ├── config.py / db.py / repository.py / storage.py / schemas.py / errors.py
│   ├── api/          # 五个接口 + 查询
│   ├── services/     # evidence / inference / conflict / explain
│   └── core/         # schema / credibility / fusion / text_parser / face_parser
├── models/           # 内置文本模型
├── schema/           # JSON Schema v1.1 + 示例
├── tests/             # 36 个单元/集成测试
├── docs/              # benchmark JSON（方案/架构/质量报告已移至 ../../报告归档）
├── pyproject.toml     # 打包与依赖
├── Makefile
├── Dockerfile
└── docker-compose.yml
```
