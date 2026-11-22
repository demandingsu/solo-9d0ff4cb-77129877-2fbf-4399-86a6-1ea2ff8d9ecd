# Ad Attribution Skew-Join Planner

广告归因作业按（设备， 广告活动）键配对曝光与点击。热点键会把单个分片打爆，
本服务为其生成**加盐分桶（salting）规划**：值班员提交同批各键的曝光行数、
点击行数与原分片，并设定各片总行数 / 配对数上限；规划器只在**所有分片都
达标**时才发布分桶规则与各片估算负载，否则保留 SQLite 中的上一版。

## 配对模型

- 同一键内，曝光与点击各自按原表顺序从 1 编号。
- 键的配对输出为笛卡尔积：`pairs = impressions × clicks`，每对记录恰好配对一次。
- 键的行数负载：`rows = impressions + clicks`。

## 热点判定与分桶规则

键满足任一条件即为热点（相对其**原分片**上限）：

```
impressions + clicks  > max_rows(原片)   或
impressions × clicks  > max_pairs(原片)
```

热点键的处理：

1. **行数较少的一侧（m 行）全量复制到每个盐桶**；行数较多的一侧（M 行）按行号
   **连续切分**为 B 段，每桶一段（`split_sizes` 尽量均分，余数分给靠前的桶）。
   两侧行数相等时复制点击侧（确定性约定）。
2. 最小桶数按原片上限求闭式解：

   ```
   B = max(ceil(M / (max_rows − m)), ceil(M × m / max_pairs), 1)
   ```

   前提：`m ≤ max_pairs` 且 `m < max_rows`，否则该键在此上限下无解（409）。
3. **落点**：桶按 [原片， 其余片按 id 升序] 的顺序首次适配（first-fit）放入
   剩余容量足够的分片；若放不下，把 B 提升到"能装进某片剩余容量"的最小值
   重试；无处可放则判定无解。
4. 大侧行号区间划分 1..M 且不重叠，小侧全量复制 ⇒ 每对 （曝光， 点击） 恰好
   由一个桶输出，`Σ 桶 pairs = M × m` 恒成立。

其余键**留在原片**，行数与配对数照常计入负载。规划全程确定性：相同输入必得
相同规则，且规则仅凭行号即可复现（每个桶都带显式 `row_start`/`row_end`）。

## 发布语义

- 仅当所有分片 `rows ≤ max_rows` 且 `pairs ≤ max_pairs` 时才落库为新版本
  （单事务 `INSERT`，版本号单调递增）。
- 无解 → `409`，发布失败 → `500`；两种情况都不写库，**上一版照常可查**。
- 历史版本全部保留：`GET /plans/{version}` 随时可取。

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build        # API 在 http://localhost:8000，文档在 /docs
docker compose --profile test run --rm tests   # 跑测试套件
```

SQLite 数据落在命名卷 `plan-data`（容器内 `/data/plans.db`），重启不丢版本。

### 本地运行

```bash
pip install -r requirements-dev.txt
PLAN_DB=plans.db uvicorn app.main:app --port 8000
pytest tests/ -q
```

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/plans` | 提交批次；全部达标则发布新版本（201），无解 409，发布失败 500 |
| GET | `/plans` | 版本列表（新→旧） |
| GET | `/plans/current` | 当前生效版本（无则 404） |
| GET | `/plans/{version}` | 任一历史版本 |
| GET | `/health` | 存活探针 |

## 在 /docs 自测

打开 `http://localhost:8000/docs`，用 `POST /plans` 的 **Try it out** 依次提交
`examples/` 中的报文：

### 1. 热点分片 — `examples/hot-shard.json`

`dev-42|camp-7`：10000 曝光 × 40 点击 = 400 000 对 > 300 000 上限 ⇒ 热点。
期望 201，规则为：

- 点击侧（40 行）复制到 2 个桶；曝光侧切为 `[1,5000]`、`[5001,10000]` 两段；
- 桶 0 落 `s1`，桶 1 因 `s1` 配对余额不足**溢出到 `s2`**；
- `Σ pairs = 200 000 + 200 000 = 400 000`，每对恰好一次；
- `shard_loads` 三片全部 `within_limits: true`。

### 2. 失败重试 — `examples/infeasible.json` → `examples/retry-fixed.json`

1. 提交 `infeasible.json`：两个冷键合计 21 000 行 > `max_rows 20 000`，
   加盐无法挽救 ⇒ **409**，响应带 `current_version`（上一版号）。
2. `GET /plans/current`：仍是上一版，未受影响。
3. 把 `max_rows` 提到 25 000 重试（`retry-fixed.json`）⇒ **201**，版本号 +1。
4. `GET /plans/1`：历史版本仍可查；`GET /plans` 列出全部版本。

## 实测结果（2026-11-22，Python 3.11 / FastAPI 0.165）

- `pytest tests/ -q`：**19 passed**（规划器 12 项 + API 7 项）。
- 本地 uvicorn 实测上述流程：
  - 热点批次 → 201 v1，分桶与负载与预期完全一致（s1: 5 160 行 / 202 000 对，
    s2: 5 350 行 / 203 000 对，s3: 0）；
  - 无解批次 → 409，`current` 保持 v1；修正后重试 → 201 v2；v1 仍可查；
  - 小侧 300 001 行 > `max_pairs` 的键 → 409「cannot be salted」，`current` 不变；
  - 重启进程后 `GET /plans/current` 仍返回 v2（SQLite 持久化生效）；
  - `/docs` 返回 200，OpenAPI 含全部 4 条业务路径。
- 本开发环境无 Docker，`docker-compose.yml` 未实际构建，已做 YAML 结构校验；
  镜像内命令与本地实测命令一致（`uvicorn app.main:app` / `pytest`）。

## 目录结构

```
app/
  planner.py   # 热点判定、桶数闭式解、落点放置、负载核算（纯函数，确定性）
  models.py    # 请求/响应 Pydantic 模型
  store.py     # SQLite 版本化存储（单事务发布，历史可查）
  main.py      # FastAPI 路由与错误映射
tests/         # 规划器单测 + API 集成测试（19 项）
examples/      # /docs 自测报文：热点分片、无解、修正重试
Dockerfile / docker-compose.yml / requirements*.txt
```

## 限制与未完成项

- **放置启发式**：桶放置为确定性的首次适配（原片优先），非全局最优装箱；
  极端碎片场景下可能判定无解，而理论上存在更优放置。
- **仅单侧复制**：未实现双侧加盐（如两阶段 skew join）；当较小侧本身接近
  上限时直接判无解。
- **不做跨键迁移**：冷键永不搬片，冷键累积超上限即判无解（符合需求约定）。
- **无鉴权/限流/审计**，多实例并发写依赖 SQLite 单写者；生产化需换外部
  数据库并加幂等键。
- 大 M 时桶数提升循环按"最小可行 B"步进，最坏迭代次数与分片数同阶，
  未做性能压测。
