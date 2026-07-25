# HQSB Console

浏览器交互推理、部署管理、真实遥测与实验追溯入口。React + TypeScript + Vite + Ant Design + ECharts；API 位于 `hqsb/console`，模型在独立工作进程中运行。

详细逐页操作、指标口径、API、配置、故障处理和文件职责见 [完整前端使用说明书](../../docs/manual/前端使用说明书.md)。实际验证见 [验收报告](../../docs/audit/HQSB_Console_验收报告_20260920.md)。`docs/` 当前被仓库忽略，公开检出可能不含这些私有文档；本文件保留可独立使用的启动说明。

## 当前能力

- Jetson / CUDA 上 Qwen3 FP16 greedy 真实逐 token 生成、SSE 续读、取消和超时。
- 持久任务与事件、幂等提交、有界串行设备队列、部署 epoch、独立进程清理。
- 运行详情、JSON/Markdown 导出、基础可比性检查；质量未评估会明确显示。
- `/proc/meminfo` 与 `tegrastats` 实际采样；历史 verdict、报告、算子源码只读查看和 SHA-256 下载校验。
- 管理员可配置 OpenAI-compatible 外部服务；不代表 ROCm / MUSA / Ascend 硬件已认证。

W8/W4 原生低比特模型回接、任意算子热切换、GPU 频率写入、批量多节点调度、RKNN Provider 和多用户权限尚未开放。

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
| `src/components.tsx`、`src/styles.css` | 通用组件、图表与响应式布局 |
| `src/streaming.test.ts`、`tests/console.spec.ts` | 分片/乱序防护单测与真实浏览器验收 |
| `../../hqsb/console/` | 顶层应用组合层，不允许下层反向 import |
| `../../hqsb/backends/interactive.py` | 独立交互 Provider，不修改 C4 基准语义 |
| `../../contracts/console/openapi.json` | 从真实服务代码导出的 v1 契约 |

禁止将静态演示数据混入正式页面。新增平台先接入实际 Provider 与能力边界；新增指标先说明采样对象、时钟、单位和缺失状态；新增写操作必须具备队列、状态、鉴权、取消与证据闭环。
