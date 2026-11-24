# 项目级开发环境（Docker Compose）

对齐实施方案的部署要求：**PostgreSQL 14（证据元数据/评分/推理结果）+ MinIO（原始证据文件）+ Redis（任务队列）+ Docker**。

> 这是**项目级环境**；pilot 本身是本地 Python 进程，不依赖这套容器。  
> 本机当前未安装 Docker，未执行 `docker compose config`；`docker-compose.yml` 已用 PyYAML 6.0.2 做语法解析校验。

## 1. 前置条件

- Docker 24+
- Docker Compose v2（`docker compose` 子命令）

Ubuntu 22.04 示例：

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER   # 重新登录后生效
```

## 2. 启动

```bash
cd deploy
cp .env.example .env
# 修改 .env 里的密码，不要用默认值
docker compose up -d
docker compose ps
docker compose logs -f postgres
```

停止 / 清理：

```bash
docker compose down          # 保留数据卷
docker compose down -v       # 删除数据卷（谨慎）
```

## 3. 服务与端口

| 服务 | 镜像 | 端口 | 用途 |
|---|---|---|---|
| `postgres` | `postgres:14` | `5432` | 案件、证据单元、推理结果、任务 |
| `minio` | `minio/minio` | `9000` / `9001` | 原始证据文件；9001 是控制台 |
| `redis` | `redis:7-alpine` | `6379` | 任务队列/缓存 |

连接串（默认 `.env.example`）：

```
PostgreSQL: postgresql://vias:vias_dev_password@localhost:5432/vias
MinIO:      http://localhost:9000   (console: http://localhost:9001)
Redis:      redis://localhost:6379/0
```

## 4. 初始化

`initdb/01_schema.sql` 会在 PostgreSQL 首次启动时自动执行，创建：

- `cases`：案件元数据、合成/非真实身份标记、ground truth
- `evidence_units`：统一证据单元（JSONB payload + 关键字段索引）
- `case_results`：逐案推理结果
- `inference_jobs`：任务队列状态

MinIO 首次使用需建桶：

```bash
# 安装 mc 后
mc alias set vias http://localhost:9000 vias vias_dev_password
mc mb vias/vias-evidence
```

或直接打开 `http://localhost:9001` 控制台建桶。

## 5. 与实施方案的对应

| 实施方案要求 | 本 compose |
|---|---|
| Ubuntu 22.04 LTS | 宿主机/目标机；容器内为官方 Linux 镜像 |
| PostgreSQL 14 | `postgres:14` |
| MinIO | `minio/minio` |
| Redis | `redis:7-alpine` |
| Docker 24 | Docker Compose v2 |
| 内网离线部署 | 可把镜像 `docker save/load` 到内网；或使用内网 registry |

**未包含**：应用服务（FastAPI/Vue）、GPU 推理容器、Kubernetes。Pilot 阶段应用在本地 Python 跑；容器化留到 12 月 V0.1 联调时再做。

## 6. 安全注意

- `.env.example` 里是开发密码，**不要**用于生产/内网正式环境。
- 默认把端口映射到 `localhost`；如需局域网访问，改 `docker-compose.yml` 的 `ports` 绑定地址，并加防火墙。
- MinIO 控制台不要暴露到公网。
- 真实/脱敏案件数据入 MinIO 前，先走单位的数据安全/伦理审批。
