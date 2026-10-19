# HQSB Console

浏览器交互推理、部署管理、真实遥测与实验追溯入口。React + TypeScript + Vite + Ant Design + ECharts；API 位于 `hqsb/console`，模型在独立工作进程中运行。

详细逐页操作、指标口径、API、配置、故障处理和文件职责见内部完整前端使用说明书；实际验证见内部 v0.2 验收报告。`docs/` 整体被仓库忽略，公开检出不含这些私有文档；本文件保留可独立使用的启动说明。

## UI v0.4：项目工作台

### L5 模型结构与压缩

`/model-structure/build` 检查实际加载的模块，创建幅值剪枝、单遍 Wanda 评分剪枝、门控 MLP 通道裁剪或 SVD 低秩制品；`/model-structure/artifacts` 查看真实参数及存储变化，显式卸载并加载制品或恢复原始浮点模型；`/model-structure/evaluation` 运行同协议 PPL 和固定输出长度推理，对照原始基线。压缩在隔离 CPU 子进程执行，原始权重不改写，完成后不会自动加载。非结构化和 2:4 剪枝仍走稠密计算，不能据此声称 GPU 加速；SVD 和通道裁剪则改变实际执行模块与参数尺寸。每次运行保存原始来源、制品哈希、实际调用凭据和质量结果。边界与接口见 [L5 说明](../../infra/README.l5-structure.md)。

### L7 解码与生成算法

`/decoding/run` 使用固定典型输入或原始 prompt，选择 greedy、温度/Top-k/Top-p 采样、提示词查找推测、独立草稿推测以及有限候选 Mask/Gather。`/decoding/draft` 显示独立草稿的驻留身份和字节，并显式卸载；`/decoding/compare` 核对同负载封存记录的 Token 一致性、Forward/接受/回滚及性能。草稿首次加载与持续显存成本均记录；不是所有推测都会更快。原 `/playground` 默认行为保留并提供新入口，运行详情有单独 L7 页签。详细接口、语言/模块、KV 不变量与未接能力见 [L7 说明](../../infra/README.l7-decoding.md)。

### L9 多卡与分布式执行

`/parallel/run|topology|peers|compare` 分别负责方案执行、设备与通信、跨节点请求 DP 和结果对照。使用已验证的原始模型制品，先卸载驻留模型；私有进程退出后统一封存。CPU 逻辑、多 GPU 与异构双节点证据分开，通信诊断默认关闭。详见 [L9 说明](../../infra/README.l9-parallel.md)。

### 内置量化质量回归

`/precision` 的“质量评测数据集”默认提供 WikiText-2 Mini：来自 raw 测试集的固定 8 段英文，每段最多 64 Token，逐 Token 经过当前权重和 KV 路径计分。先用浮点基线评测，再加载量化方案运行同一预设，将双方运行 ID 填入“基线与候选质量对照”，查看 PPL、平均 NLL 增量和 PPL 倍率。可在双方运行前登记相同质量阈值；未登记时只报告测量结果。旧的自定义文本 / logits / 标准答案模式仍保留。数据随项目保存，日常使用不依赖联网；该子集用于英文语料的快速回归，其他能力可通过相应任务集评测。来源、协议、模块和 API 见 [L4 表示精度与量化说明](../../infra/README.l4-precision.md#内置-wikitext-2-mini-与-ppl)。

质量结果现会自动关联每个模型的已封存浮点基线，显示 PPL 增幅、绝对变化和 NLL 增量；运行列表也提供“PPL 相对基线”。首个有效基线固定保存，后续重测不自动替换，可在合格浮点记录显式设为新基线。未量化的 FP16 / BF16 / FP32 标注实际精度；模型制品、分词器、源码、依赖版本、设备或评分协议不同会分开建立基线。未测过浮点版本时会提示先建立，不会虚构下降比例。手动多运行对照继续用于进一步分析及预登记阈值裁决。

### 通用模型架构

本页新增“计算图与 L3 执行”：按请求选择 Eager、Dynamo 捕获或 Inductor 编译，调整形状与 Fullgraph，运行固定样例或独立 PPL。FX 节点通过已有模块路径关联权重；支持历史图、源码、实际图调用/新图计数及严格身份对照。原 ATen 架构采集继续保留。首批仅开放原生浮点/默认 KV 的编译组合，CUDA Graph 暂不启用，详见 [L3 说明](../../infra/README.l3-runtime.md)。

`/model-architecture` 先展示不依赖模型族白名单的权重路径树；模型加载后可查看实际 nn.Module 类与参数。点击“采集运行架构”显式提交固定小样本，查看实际 ATen 算子、Tensor 依赖、Prefill / Decode 和来源指纹，支持筛选、邻域图和完整 JSON 导出。模型族标签及已核对的静态示意图作为补充，未识别家族不会阻止通用解析。采集有额外开销，仅用于诊断。模块、接口、持久化及能力边界见[模型架构观测说明](../../infra/README.model-architecture.md)。

### 推理输出格式

推理页按 Markdown 实时渲染累计输出，支持标题、列表、引用、行内代码、代码块、链接，以及 GFM 表格、任务列表和删除线。`$...$` 行内公式与 `$$` 围栏内的独立公式通过本地打包的 KaTeX 渲染，支持分数、上下标和 `\boxed{}`；不请求外部公式服务或字体 CDN。长代码、宽表格和独立公式在内容区内横向滚动；复制输出保留模型原始 Markdown/LaTeX 文本。未闭合代码围栏和公式在生成过程中仍可显示，后续快照会继续更新；不合法公式保留为可读原文。输出不执行原始 HTML；本次仅调整浏览器呈现，不修改推理请求、SSE 协议或原始运行记录。

### 内嵌使用指南

`/guide` 提供与当前实现对应的十一章操作手册，覆盖机器连接、模型下载、固定输入评测、推理参数、算子版本替换、预注册实验、内存/KV/CPU 页与 profiler、原件保存及十层能力边界。顶部“使用指南”按当前页面定位章节；评测标题区和“设置与帮助”另有入口，环境导航与快速跳转可直接找到整份手册。手册只做导航与说明，不提交任务或改变部署。新增字段、profile 或能力时同步维护 `src/pages/UserGuide.tsx`，样式归 `src/features/help/styles.css`。

### ModelScope 权重下载

`/models`（环境 → 模型下载与接入）从登记发布方实时索引公开仓库，按当前 GPU/主机或容器/下载磁盘筛选，选择分支或标签后检查配置与文件哈希，显式下载。进度使用节点实际传输字节和已校验文件数，支持取消/续传、重启后保留任务，以及封存后动态登记到推理和评测列表。下载不自动加载 GPU 模型。平台镜像、资源预估与实测运行分开标注；当前不接量化、多模态、远程 Python 和未适配架构。模块职责、数据与 API 契约见 [模型下载说明](../../infra/README.model-hub.md)。

重设计覆盖全部既有路由，使用项目任务导航、运行筛选目录、证据检查器、推理编辑器和实验计划工作区。API v1、请求字段、认证、SSE 和下载协议保持原样；没有新增运行依赖。

- `src/shell/`：项目导航、快速跳转（按钮或 Cmd/Ctrl+K）、小屏抽屉与登录。
- `src/styles/`：共享语义 tokens、外壳、页面组合；旧版单体样式已替换。
- `src/pages/RunDirectory.tsx`：30 条/页，当前页筛选；`q/state/kind/page` 写入 URL。
- `/observations`：观测目录 → 固定运行对象 → 左侧证据视图；窄屏自动改顶部 tabs。
- `/optimization`：已冻结计划目录与实验工作区；质量、性能和机制保留后端的独立判定。
- `/benchmarks`：性能测试主工作面，预选固定典型输入和多个版本化方案；按用例比较真实输出、参考评分与性能。原推理工作台继续作为交互案例。

## 固定输入的快速对照

“评测 → 性能与质量对照”默认包含算术、结构化提取、上下文检索和开放解释，输入原文与 SHA-256 可展开核对。选择基线、最多三个候选、统一输出预算和重复次数后开始；后台负责预热、方案重载、测量与原部署恢复。刷新通过 URL `?benchmark=<id>` 恢复查看，不重复执行；任务可取消，历史可重新打开。测试环境保持免登录。

推理工作台在本地模型就绪后，通过 `POST /requests/token-budget` 使用同一 worker 的常驻 tokenizer 计算真实输入长度，包含官方聊天模板或原始文本的特殊 Token。输入检查防抖、旧响应按部署 epoch 和输入隔离；不生成 Token、不新增推理记录、不在 API 进程加载 tokenizer。“最大值”使用 `min(部署输出上限, 总窗口－实际输入长度, 4096)`，超额手动预算会明确提示并阻止提交。

运行时仍独立校验窗口。输入超窗产生 `CONTEXT_LIMIT` 失败记录、零输出及部分诊断制品；清理确认成功后保留常驻模型与 epoch，修改预算即可继续提交。设备执行或清理异常仍会关闭 worker、使部署失效。已失效/卸载的模型在输入区有明确的重新加载入口，当前输入与设置在同一页面操作中保留。

结果按用例排列方案，分别展示任务正确率、Token 一致率、首 Token、生成 E2E、Decode 吞吐、输入/输出数量、峰值分配与实际输出。开放解释不自动评分，Token 一致率与准确率分别统计；输出数量不同的结果不计算 E2E 相对比例。每项可跳转同 run 的实际算子凭据与原件。可选机制诊断另做一轮，与性能样本分开保存。版本化用例用于快速比较功能；正式性能与质量评测采用独立进程和对应任务数据集。

模块：`features/benchmarks/Results.tsx` 只呈现服务端冻结报告，`pages/Benchmarks.tsx` 管理选择、显式提交和轮询；评分与统计属于 Python `hqsb.experiments.benchmark_suite`，任务和 journal 所有权属于 `hqsb.console.benchmark_campaigns`。输入不允许浏览器任意改写。验收命令仍须通过项目远端代理执行：

```bash
./scripts/remote_run.sh python3 -m pytest tests/unit/console/test_benchmarks.py -q
./scripts/remote_run.sh bash -lc 'cd web/console && HQSB_CONSOLE_URL=http://127.0.0.1:8768 npx playwright test tests/benchmarks.spec.ts'
# 真实模型额外显式设置 HQSB_BENCHMARK_UI_LIVE=1；会占用 GPU 并重载模型。
```

真实功能通过同源 HTTP 服务访问。**不要直接双击 `index.html` 或 `dist/index.html`**，应用入口通过 HTTP 服务加载资源。`public/preview.html` 是明确标注的离线截图预览，可双击且不调用后端；生成命令仍须在 Jetson 执行：

```bash
./scripts/remote_run.sh python3 scripts/console/build_preview.py --screenshots reports/console/v04
```

生成器需要已保存的十张验收截图。私有截图、生成 HTML 和重设计前的 `.backups/` 不进入 Git；预览不替代交互工作台。安装、构建和测试仍遵守项目的远端执行规则。

## 在推理界面切换算子

进入 `/playground` 的“执行配置 → 算子实现与版本”，按算子类别选择带版本的固定方案，点击“加载并应用算子”或“应用并重载模型”。加载成功并获得新的部署 epoch 后，“开始推理”才可使用所选方案；选择与当前加载值不一致时会提示待应用并禁用推理。RMSNorm scalar v1 / half2 v2 等版本分别可选，组合方案不会改变未列出的算子。

模型输出下方的“本次运行凭据与观测”展示实际方案、逻辑调用计数、封存原件下载及阶段/token/内存摘要，并链接同次运行的 KV、OS 页和 GPU profiler 详情。普通推理与实验任务使用同一凭据展示。制品缺失/身份不符、加载失败和缺少观测都明确显示；目录中的候选不表示质量或性能验收通过。

当前 `8878` 测试入口对应 `3090-inf` 的 `/root/HeteroQuantServeBench`，配置为 `configs/console/3090-inf.yaml`，保留免登录；机器选择与隧道说明见 [机器连接说明](../../infra/README.machine-connections.md)。下面 kernel-ui 命令是原 Jetson 隔离目录的历史验收入口，需要显式指定目标。

显式真实模型验收通过 `HQSB_KERNEL_UI_LIVE=1` 启用，默认不会执行新用例中的模型任务。该验收会串行重载当前服务，需独占设备；最终恢复框架原生模型就绪状态：

```bash
HQSB_REMOTE_HOST=jetson HQSB_REMOTE_DIR=/home/jetson/work/HQSB-kernel-ui-20261016 ./scripts/remote_run.sh \
  env HQSB_CONSOLE_URL=http://127.0.0.1:8768 HQSB_KERNEL_UI_LIVE=1 \
  bash -lc 'cd web/console && npx playwright test tests/kernel-ui.spec.ts'
```

结果位于 `reports/console/kernel-ui/`，包括 `acceptance.json`、逐方案运行 ID、下载哈希、CUDA 活动证明和界面截图；记录包含功能与证据联通验收，独立质量和性能评测另行执行。

## 当前能力

- Jetson / CUDA 上 Qwen3 FP16 greedy 真实逐 token 生成、SSE 续读、取消和超时。
- 持久任务与事件、幂等提交、有界串行设备队列、部署 epoch、独立进程清理。
- 运行详情、JSON/Markdown 导出、基础可比性检查；质量未评估会明确显示。
- `/proc/meminfo` 与 `tegrastats` 实际采样；历史 verdict、报告、算子源码只读查看和 SHA-256 下载校验。
- 管理员可配置 OpenAI-compatible 外部服务；各硬件后端按实际节点配置验证。
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
4. 下载 Chrome trace 供兼容工具离线查看；下载优化方案记录证据、候选操作、验证方法和限制。方案包含待执行的候选措施与验证步骤。
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
ssh -N -L 8765:127.0.0.1:8765 <jetson-user>@<jetson-lan-ip>
```

浏览器打开 <http://127.0.0.1:8765> 即可进入工作台，测试阶段默认免登录，无需 API Key 或访问令牌。进入“设备与部署”加载部署，再打开“推理工作台”。第一次加载会较慢，加载本身有独立任务记录。停止服务可在服务终端 Ctrl+C；页面卸载先排空请求。

需要恢复令牌认证时，启动服务添加 `--require-auth`，或在服务配置中设置 `auth_required: true`。此时才会生成并读取 `.console/access-token`，浏览器显示登录页；免登录模式会忽略已有令牌及过期会话。

本轮验收使用独立远端目录 `/home/jetson/work/HQSB-console-dev`，避免覆盖原工作区。要使用该目录，将命令前加 `HQSB_REMOTE_DIR=/home/jetson/work/HQSB-console-dev`。服务读取历史证据可加 `env HQSB_EVIDENCE_ROOT=/home/jetson/work/HeteroQuantServeBench`；默认从自身仓库读取。

## 配置与数据

默认 `configs/console/default.yaml`。`--config` 可选择其他服务端配置，`--host` 默认回环地址，`--port` 默认 8765。`HQSB_CONSOLE_DATA` 覆盖 `.console/`；`HQSB_EVIDENCE_ROOT` 指向有历史证据的仓库根目录。

Jetson/CUPTI 权限受限时，管理员可显式加 `--privileged-worker`（或 YAML `privileged_worker: true`）；默认关闭。需要账户已有对应 `sudo -n` 权限，只提权固定私有管道设备 worker，API 保持普通用户，不修改 sudoers/全局驱动权限。必须信任本机仓库和依赖，不能作为恶意本地代码沙箱。没有 CUDA 活动时页面明确显示 CPU-only/partial，GPU 采集状态见实际 CUDA 活动记录。

`.console/` 包含 SQLite WAL 运行记录和进程锁；启用认证后还包含私有令牌。运行记录默认不保存输入原文（仅 SHA-256），但保存输出、指标和实际部署快照；开启“保存输入”后才保存提示词。不要将私有数据或下载报告公开发布。备份先停止服务，再复制整个数据目录。没有自动清理历史的后台任务。

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

L6 入口为 `/memory-cache/configure` 与 `/memory-cache/records`，分别负责缓存方案和同输入运行对照；共享 deployment/run URL 身份。支持动态、静态、卸载预取、重计算、有界前缀复用，显示实际张量/allocator/搬运与进程缺页。L8 调度页可选择 GPU 块池；仍需稠密 Attention 拼装。默认单请求行为和免登录保留；细节、接口与边界见 [L6 说明](../../infra/README.l6-memory.md)。
