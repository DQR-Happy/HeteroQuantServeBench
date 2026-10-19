# L9 多卡与分布式执行

入口为“优化 → 多卡与分布式执行”。`/parallel/run` 配置并行执行与固定负载，`/parallel/topology` 查看设备/后端能力，`/parallel/peers` 执行已登记双节点 DP，`/parallel/compare` 比较封存结果。模型、运行身份通过 URL 保留，旧的单请求、L4–L8 路径不变。

## 能做什么

| 策略 | 实际执行 | 资源与限制 |
| --- | --- | --- |
| single | 单个私有进程加载原始浮点模型，逐请求 greedy | GPU 与 CPU 都可；不依赖 torch.distributed，兼容未编译分布式组件的 Jetson wheel |
| dp | 每设备一个进程和完整模型副本；请求 i 交给 i % world_size 的 rank | 推理副本 DP，不是训练 DDP；每 rank 拥有独立请求 KV，至少一条请求/rank |
| tp | Transformers 5.8 原生 `tp_plan=auto`，真实参数分片及 collective | 需要官方模型 TP plan、可整除形状/官方分片规则与可用进程组；记录局部/全局形状，没有实际分片会失败，不加载完整副本冒充 TP |
| layer_shard | 明确 device_map，将连续层放到多卡；embedding、norm、LM head 在首卡 | Llama/Qwen2/Qwen3/Gemma3-text 已定义放置适配器；这是顺序模型并行，不是多微批 PP 流水或通信计算重叠 |
| prefill_decode | rank 0 完成 prefill；将实际逐层 K/V 和已预测首 Token 传给 rank 1，再继续 decode | 两个完整模型副本，默认动态 KV；当前只支持 dense full-attention Llama/Qwen2/Qwen3；不支持滑窗、混合状态、量化/压缩 KV |
| dp_hosts | 当前节点及一个已登记 HTTP/SSH 节点，各运行一个 single 子任务，按奇偶请求索引分片 | 可异构 CPU 架构和 PyTorch wheel；只传请求/结果，不传权重、KV 或 collective；两端必须部署本版 L9 API，分别校验同一源 manifest/Tokenizer |

CUDA TP/DP 使用 NCCL，P/D 使用 Gloo CPU Tensor send/recv，CPU 逻辑诊断使用 FP32/Gloo。进程组能力未编译或设备不可见时明确拒绝，不能将多进程挤在同一 GPU 上冒充多卡。CPU 进程编号只代表诊断 rank。

尚未接入：多微批 PP 流水、CP/SP、MoE EP、专家平衡、TP×PP×DP 混合并行、跨主机 NCCL/TP、跨主机 KV 传输、不同版本 KV 转换。L3 编译、L4 量化、L5 压缩、L6 特殊缓存、L7 推测与 L8 连续合批的组合没有在 L9 中开放。原有 S10 `hqsb.distributed` 的拓扑、账本、故障、边界语义测试保留；那些纯语义模块不被计算成这里已执行的 GPU 能力。

## 使用顺序

1. 在模型页面下载并验证源权重。每个目标节点独立准备本地制品，网页不能提交任意文件路径或启动任意远程命令。
2. 在设备与部署页卸载当前节点的驻留模型。L9 需要拥有模型设备；不会自动卸载其他人的任务，也不会把量化驻留模型混入原始权重测试。
3. 在“执行方案”选择模型、single、CUDA、精度、固定原始文本与预算，运行基线。每行一条请求，最多 8 条，每条最多 512 输入 Token、128 输出 Token；总上下文继续受部署和模型上限约束。
4. 保持相同输入/预算/精度/固定长度/通信观测口径，选择候选策略。设备选择表示当前节点可见 GPU 索引，子进程内部会映射为本地编号，并保留请求计划。
5. 查看每 rank 权重字节、分片形状、TP plan、设备、allocator、KV、请求输出与通信事件。任务完成后全部私有进程退出，权重与 KV 不驻留；需要交互推理时再到原设备页加载模型。
6. 在“结果对照”选择两个不同的已完成任务。保存的比较同时绑定原始封存 manifest、权重、Tokenizer、源码、运行版本、真实输入 Token 和输出 Token，不以生成时间单独宣布优化成功。

双节点 DP：先在“机器连接”登记 SSH 或 HTTP 节点，在该节点准备相同模型并卸载驻留权重，再在“跨节点 DP”选择连接和该节点真实部署 ID。计划需要 single、至少两条请求；本地负责偶数请求，远端负责奇数请求。每个子任务沿用普通 L9 的设备租约、预算、幂等键和取消机制。

按钮禁用时沿用 ActionButton，悬停/键盘聚焦说明缺失条件并链接设备、模型或节点页面。禁用提示不执行任何卸载/切换动作。

## 模块、接口、开发语言

| 模块 | 职责/所有权 | 输入 → 输出 |
| --- | --- | --- |
| `core/parallel_execution.py` | Python/Pydantic；纯配置与能力契约，无 torch | 策略、设备、负载与预算 → 严格计划/拒绝原因/能力目录 |
| `backends/parallel_probe.py` | Python；一次隔离只读探测 | 目标运行环境 → 可见 GPU、空闲量、UUID、P2P 可访问性、分布式/NCCL/Gloo 编译能力 |
| `backends/parallel_launcher.py` | Python/Linux；私有 torchrun agent 生命周期 | 已解析参数 → rank 子进程，父进程死亡信号防止遗留 |
| `backends/parallel_rank.py` | Python/PyTorch；每 rank 权重、请求 KV 与进程组 | 私有 input.json → 逐 rank 状态、真实输出、分片/内存/通信 JSON |
| `backends/parallel_model.py` | Python/Transformers；源验证、native 模型与放置 | 管理员登记路径、计划 → 已验证浮点模型/Tokenizer、greedy 输出、参数/KV inventory |
| `backends/parallel_handoff.py` | Python/torch.distributed；P/D KV 协议 | 完整动态 KV、首 Token → CPU staging/send/recv、接收侧 DynamicCache、字节和阻塞阶段时间 |
| `console/parallel_jobs.py` | Python；单节点独占租约、预算、进程组监督、发布 | 严格计划/登记部署/幂等键 → 持久任务、有限时取消、进程退出 ACK、封存凭据 |
| `console/parallel_peers.py` | Python/httpx；双节点子任务 ID、幂等、取消与结果合并 | 已登记连接/部署与计划 → 独立子任务、来源核对、合并证据；不拥有模型 Tensor |
| `console/parallel_routes.py` | Python/FastAPI；HTTP 契约与严格对照 | HTTP → 目录/探测/任务/封存记录/比较结果 |
| `features/parallel`、`pages/Parallel.tsx` | TypeScript/React；页面与查询/URL 身份 | 实际 API → 执行、前置操作、资源、结果与比较展示 |

Python 负责配置、生命周期、库调用与证据；模型计算/TP/collective 复用 PyTorch、Transformers、CUDA/NCCL/Gloo 的实际实现。没有新增自定义 C++/CUDA Kernel。下层 backends/core 不依赖原有上层 `hqsb.distributed` 语义包，保留既有依赖方向和 CPU-minimal import 边界。

## 资源与故障契约

- 默认任务从未量化 safetensors 原件加载，`local_files_only`、`trust_remote_code=False`、`use_safetensors=True`；加载前校验 manifest。按 dtype 与张量头估计参数字节，包含磁盘重复的 tied tensor 时估计偏大。
- CPU 诊断保守预留 2×每 rank 参数量加 512 MiB 主机空间；普通 CUDA 副本先加载普通可换出的 CPU Tensor，再将完整模型移到 GPU，保留每 rank 参数量加 512 MiB 主机预算。该显式路径避免直接 GPU checkpoint dispatch 在 Jetson 上的 pinned host memory 峰值；TP 保留原生分片加载、按层放置保留明确 device_map。加载路径写入凭据，不发生隐式 OOM 重试或 CPU 部分卸载。Jetson 的共享内存不被伪装成独立 CPU/GPU 两份容量。预算不是任意模型都不 OOM 的保证。
- 每任务至少保留 512 MiB 磁盘余量，检查制品配额；KV handoff 上限 256 MiB。最大 4 设备/进程，默认超时 120 秒、上限 300 秒，进程组 collective 超时最多 60 秒。逐参数细目最多 256 条并标记截断，完整参数字节仍累积。
- 通信 CPU profiler 默认关闭；显式开启时最多 64 个总输出 Token，额外开销可能很大。性能对照需要保持采样开关一致，诊断模式与不采样结果不能混为同口径。
- 当前节点所有 PyTorch 驻留部署必须卸载。控制服务持有 L9 独占 owner/pending，使用与原单请求路径同源的 GPU 文件锁；按设备索引排序取锁。失败不隐式退回 single、CPU 或不同精度。
- single/layer_shard 不启动不必要的 torchrun；多 rank 启动私有进程组、不自动重启失败 rank。取消先终止本任务的进程组/后代，超时后强杀并确认退出，未确认时隔离后续 L9 执行。agent/rank 设置 Linux 父进程死亡信号。
- 只有进程退出后监督进程才封存完整结果。失败/取消保存 partial；清理 unknown/failed 不能被标为成功。服务重启沿用 store interrupted 语义，不自动重放模型任务。
- 双节点 coordinator 只取消本次拥有的子任务。远端提交使用固定幂等键，网络结果不确定时重用同一键，不创建替代任务；无法确认远端清理时保留 unknown，子任务有自己的最长执行期限。远端 ID/提交键保留供追踪，重启不会自动重放。

## API 与保存

```text
GET  /api/console/v1/parallel/catalog
GET  /api/console/v1/parallel/hardware
GET  /api/console/v1/parallel/jobs
POST /api/console/v1/parallel/jobs
GET  /api/console/v1/parallel/peers
POST /api/console/v1/parallel/federated-jobs
GET  /api/console/v1/runs/{id}/parallel
POST /api/console/v1/parallel/comparisons
POST /api/console/v1/requests/{id}/cancel
```

普通任务请求：`{deployment_id, plan}`；plan 包含 strategy、device_type、devices、precision、prompts、max_output_tokens、fixed_output_length、trace_collectives、timeout_s。双节点请求额外提供 peer_machine_id、peer_deployment_id，客户端不能提供任意 URL、权重路径、NCCL 环境变量或远程命令。POST 沿用 Idempotency-Key 与免登录测试节点的现有调用约束。

运行保存为统一 store 的 `kind=distributed`；配置、状态、进度、取消、结果均可从普通运行记录追踪。`.console/captures/<run_id>/` 保存私有输入、worker 日志、逐 rank JSON、parallel-receipt.json、execution-receipt.json 与 supervisor 的 experiment-manifest.json。逐 rank 内容汇入被封存的总凭据；比较读取封存哈希而非信任可修改的工作文件。比较另保存 `.console/l9-comparisons/<hash>.json`。

## 测量与结论边界

直接记录：实际请求分片、Token IDs、输入 Token hash、模型/Tokenizer 身份、库版本、设备、局部/全局参数形状与字节、TP plan、层放置、KV 形状/字节、allocator allocated/reserved/peak、加载/生成/任务总时间、Token/s、P/D KV payload 和阻塞传输时间、通信 profiler CPU 事件名/次数/CPU span。

单节点 workload 取最慢 rank 的同口径本地持续时间，含工作结束 barrier，排除加载/launcher；job_total 含启动、加载、执行、退出与监督开销。没有额外 warmup，首请求可能包含首次库初始化。P/D 的每请求生成 Token/s 只计 Decode；首 Token 时间包含该 Decode rank 等待 KV 的本地时段，**不是服务入口 TTFT**。整组 workload/任务时间才包括所有阶段。

双节点 aggregate Token/s 使用 coordinator 自己的单调时钟，包含两端冷加载、HTTP 提交/轮询、执行与清理 ACK；不能将它与单节点排除加载的热循环 Token/s 混为同口径加速比。两端局部 workload/加载时间分别保留在 participant_receipts；不相减不同主机时钟来编造链路耗时。

通信 CPU 事件不是 GPU 已完成 collective 的设备计时；P/D 字节是 payload，不是包含协议头的线上字节。P2P 可访问不代表已测 NVLink 带宽。GPU Cache 命中、物理页迁移、互连利用率、跨节点时钟校准与正式扩展效率目前没有直接测量。

贪心逐 Token 一致只支持当前输入和模型结论。浮点分片或异构硬件可能改变近似并列 logits 的 argmax，比较保留分歧并拒绝合格对照，不以放宽阈值掩盖。PPL 不执行这里的请求分片/路由，不能冒充并行执行正确性；正式质量还应使用既有独立评测。一次结果没有正式 speedup、质量不变、完全兼容或分布式生产稳定性主张。

## 验证

通过项目代理在目标端执行，不在 Mac 运行代码：

```bash
./scripts/remote_run.sh pytest -q tests/unit/console/test_parallel_execution.py tests/unit/distributed
./scripts/remote_run.sh python -m scripts.console.validate_l9
./scripts/remote_run.sh bash -c 'cd web/console && HQSB_CONSOLE_URL=http://127.0.0.1:8768 npx playwright test parallel.spec.ts'
```

验证脚本需要空闲服务与 `jetson-l9` 转发地址，保存原模型并在 finally 恢复，不下载或改写权重。实际环境差异、失败原因、通过/跳过项目及原始运行见 `reports/console/l9-parallel/report.md`。

语义参考：[Transformers TP inference](https://huggingface.co/docs/transformers/v4.57.0/perf_infer_gpu_multi)、[PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html)。本实现以实际安装的 PyTorch 2.8 / Transformers 5.8 源码与 Jetson wheel 探测为准，不能把上游全部能力计为本项目已经接通。
