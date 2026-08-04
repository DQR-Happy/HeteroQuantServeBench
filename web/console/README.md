# HQSB Console

浏览器交互推理、部署管理、真实遥测与实验追溯入口。React + TypeScript + Vite + Ant Design + ECharts；API 位于 `hqsb/console`，模型在独立工作进程中运行。

详细逐页操作、指标口径、API、配置、故障处理和文件职责见 [完整前端使用说明书](../../docs/manual/前端使用说明书.md)。实际验证见 [v0.2 验收报告](../../docs/audit/HQSB_Console_v0.2_验收报告_20260921.md)。`docs/` 当前被仓库忽略，公开检出可能不含这些私有文档；本文件保留可独立使用的启动说明。

## 当前能力

- Jetson / CUDA 上 Qwen3 FP16 greedy 真实逐 token 生成、SSE 续读、取消和超时。
- 持久任务与事件、幂等提交、有界串行设备队列、部署 epoch、独立进程清理。
- 运行详情、JSON/Markdown 导出、基础可比性检查；质量未评估会明确显示。
- `/proc/meminfo` 与 `tegrastats` 实际采样；历史 verdict、报告、算子源码只读查看和 SHA-256 下载校验。
- 管理员可配置 OpenAI-compatible 外部服务；不代表 ROCm / MUSA / Ascend 硬件已认证。
- v0.2：提交前选择关闭详录、基础观测或算子诊断；运行详情显示阶段瀑布、逐 token 成本、生命周期内存、算子耗时和输入形状。
- v0.2：有界 profiler 窗口、Chrome trace 下载、原始事件服务端分页检索，以及带验证方法的候选优化方案导出。
- v0.2：深度分析页连接 E02-07 的真实阶段/kernel/计数器证据与 E05-02 的量化存储、执行路径和质量结果；历史采集明确标注。
- v0.2：量化页可从已就绪 PyTorch 模型生成 RTN W4/W8 存储候选，转换任务可追溯、可取消；制品不自动部署，也不表示质量通过。
- v0.2.1：打开 `/memory-flow` 或侧栏「内存与数据流」，查看真实共享内存结构、去重权重面积图、KV 分层图与 CUDA copy 方向；图块可点击查看证据。默认选最近算子详录，`?run=<id>` 可固定某次请求，历史回放无需加载模型。

W8/W4 原生低比特模型回接、任意算子热切换、GPU 频率写入、批量多节点调度、RKNN Provider 和多用户权限尚未开放。

## v0.2 观测工作流

1. 在“设备与部署”加载模型；“内存归因与生命周期”同时显示整机 MemAvailable、进程 RSS/PSS，以及带时间戳的 worker allocator 快照。模型加载后的可用内存已经包含权重占用影响。
2. 在“推理工作台”选择观测模式。基础观测记录主机阶段、token 和内存；算子诊断最多捕获 prefill 与前 8 个 decode step，首 decoder 层带模块 range。所有观测都有成本，不能将插桩数据当作未插桩性能基准。
3. 推理结束后打开“查看完整运行 → 全链路观测”。依次查看阶段与逐 token、算子耗时、原始事件查询、优化假设和请求内存快照。CPU self、CUDA self、累计 kernel 工作时间与阶段墙钟时间不混算。
4. 下载 Chrome trace 供兼容工具离线查看；下载优化方案记录证据、候选操作、验证方法和限制。方案中的假设不是已经完成的优化。
5. 打开“深度分析”钻取已有阶段实验。所有历史数据保留来源与 SHA-256；不会将一次历史 NCU 重放归为当前请求的观测。
6. 在“量化与质量”创建 RTN 候选，进入运行记录追踪任务，再下载 manifest。完成转换后仍需独立质量与低比特执行验证；当前 FP16 部署保持自身执行契约。

旧运行没有详细观测时显示缺失原因，不对历史请求补造阶段或 kernel 数据。外部 OpenAI-compatible 端点只暴露协议层能力，不自动获得远端框架或设备内部时间线。

## 在 Jetson 安装与启动

以下命令从 Mac 项目根目录输入，实际安装、构建、执行均由代理在 Jetson 完成。Mac 不运行 Node 构建或 Python 服务。前提是远端已有 Node 22、Python 3.10、NVIDIA torch/transformers 环境和模型 `~/models/hqsb/Qwen3-1.7B`；不要安装 PyPI torch 覆盖 Jetson 的厂商 wheel。

```bash
./scripts/sync_to_jetson.sh
./scripts/remote_run.sh python3 -m venv --system-site-packages .venv-console
./scripts/remote_run.sh .venv-console/bin/python -m pip install -e '.[console]'
./scripts/remote_run.sh bash -lc 'cd web/console && npm ci && npm run build'
./scripts/remote_run.sh bash scripts/console/serve.sh
```

保持服务终端打开。另一终端建立仅做连接转发的 SSH 隧道：

```bash
ssh -N -L 8765:127.0.0.1:8765 jetson@192.168.10.7
```

在第三个终端读取远端令牌，复制到登录页，不要提交到 Git：

```bash
./scripts/remote_run.sh cat .console/access-token
```

浏览器打开 <http://127.0.0.1:8765>，登录 → 设备与部署 → 加载部署 → 推理工作台。第一次加载会较慢，加载本身有独立任务记录。停止服务可在服务终端 Ctrl+C；页面卸载先排空请求。

本轮验收使用独立远端目录 `/home/jetson/work/HQSB-console-dev`，避免覆盖原工作区。要使用该目录，将命令前加 `HQSB_REMOTE_DIR=/home/jetson/work/HQSB-console-dev`。服务读取历史证据可加 `env HQSB_EVIDENCE_ROOT=/home/jetson/work/HeteroQuantServeBench`；默认从自身仓库读取。

## 配置与数据

默认 `configs/console/default.yaml`。`--config` 可选择其他服务端配置，`--host` 默认回环地址，`--port` 默认 8765。`HQSB_CONSOLE_DATA` 覆盖 `.console/`；`HQSB_EVIDENCE_ROOT` 指向有历史证据的仓库根目录。

Jetson/CUPTI 权限受限时，管理员可显式加 `--privileged-worker`（或 YAML `privileged_worker: true`）；默认关闭。需要账户已有对应 `sudo -n` 权限，只提权固定私有管道设备 worker，API 保持普通用户，不修改 sudoers/全局驱动权限。必须信任本机仓库和依赖，不能作为恶意本地代码沙箱。没有 CUDA 活动时页面明确显示 CPU-only/partial，开关本身不等于 GPU 验收通过。

`.console/` 包含私有令牌、SQLite WAL 运行记录和进程锁。运行记录默认不保存输入原文（仅 SHA-256），但保存输出、指标和实际部署快照；开启“保存输入”后才保存提示词。不要将私有数据或下载报告公开发布。备份先停止服务，再复制整个数据目录。没有自动清理历史的后台任务。

默认单操作者，不支持多租户隔离。需要公网部署时使用 HTTPS 反向代理和 `secure_cookie: true`，禁用 SSE 代理缓冲，限制入口访问；当前验收仅覆盖 SSH 隧道与回环访问。

## 开发与检查

```bash
./scripts/remote_run.sh .venv-console/bin/python -m pip install pytest==8.4.2 ruff==0.11.13
./scripts/remote_run.sh .venv-console/bin/python -m pytest tests/unit/console -q
./scripts/remote_run.sh .venv-console/bin/python scripts/audit/import_dependency_gate.py
./scripts/remote_run.sh bash -lc 'cd web/console && npm test && npm run build'
./scripts/remote_run.sh .venv-console/bin/python scripts/console/export_schema.py
./scripts/remote_run.sh bash -lc 'cd web/console && npm run api:generate'
```

显式硬件验收会加载模型并提交测试提示词，服务需已运行：

```bash
./scripts/remote_run.sh .venv-console/bin/python scripts/console/validate_live.py
./scripts/remote_run.sh bash -lc 'cd web/console && npx playwright install chromium && npm run test:e2e'
```

Playwright 在 Jetson 上启动 Chromium。硬件验收脚本先完成并使部署 ready，再运行浏览器测试。结果位于远端 `reports/console/`；该目录不跟踪私有运行数据。`npm run dev` 也必须通过远端代理执行，Vite 默认只绑定 `127.0.0.1:5173`，代理到同机 8765 API；需要另建 5173 SSH 隧道。

## 维护入口

| 路径 | 职责 |
|---|---|
| `src/App.tsx`、`src/main.tsx` | 认证入口、导航路由、主题与 QueryClient |
| `src/pages/` | 总览、真实推理、部署、运行、证据、对比和讲述页面 |
| `src/streaming.ts`、`src/hooks.ts` | 增量 SSE 解析、重放游标、快照恢复 |
| `src/api/` | HTTP 客户端、生成 API 类型、页面读模型 |
| `src/features/observability/` | 请求观测模式、阶段与 token 图、算子表、trace 查询、优化假设、能力矩阵；只渲染服务端证据 |
| `src/features/memory/` | 整机、进程、allocator 与生命周期快照的分范围展示 |
| `src/features/research/` | 只读历史阶段、内核、计数器与量化证据；独立量化候选表单 |
| `src/components.tsx`、`src/styles.css` | 通用组件、图表与响应式布局 |
| `src/streaming.test.ts`、`tests/console.spec.ts` | 分片/乱序防护单测与真实浏览器验收 |
| `../../hqsb/console/` | 顶层应用组合层，不允许下层反向 import |
| `../../hqsb/backends/interactive.py` | 独立交互 Provider，不修改 C4 基准语义 |
| `../../contracts/console/openapi.json` | 从真实服务代码导出的 v1 契约 |

禁止将静态演示数据混入正式页面。新增平台先接入实际 Provider 与能力边界；新增指标先说明采样对象、时钟、单位和缺失状态；新增写操作必须具备队列、状态、鉴权、取消与证据闭环。
