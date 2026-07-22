# HeteroQuantServeBench 项目完整剖析、最终架构、全阶段路线图与秋招证据手册

> 历史架构与设计蓝图，正文基于下述旧提交。2026-09-20 的实现状态和操作入口以 [现状总览](../project_status.md) 与 [使用说明书](../manual/使用说明书.md) 为准；此文中的“最终架构”不表示已经完成全部实验验收。

> 文档版本：2026-08-17  
> 审计对象：`DQR-Happy/HeteroQuantServeBench` 的 `origin/main`  
> 审计基线：`4bc59e76be3262f788792410fb590dd8af5a4054`  
> 基线提交时间：2026-05-10 10:52:14 +08:00  
> 基线提交说明：`chore: update README, evidence ledger and project status`  
> 受控文件总数：210  
> 审计方法：只读取 Git 树、源码、配置和文档；本次没有安装依赖、构建、运行测试、运行模型或执行 benchmark。  
> 文档定位：项目总说明书、最终技术蓝图、Coding Agent 执行总纲、实验设计手册、证据与求职表达手册。

---

## 0. 如何阅读与使用本文档

本文档不是对 README 的扩写，而是整个 HQSB 项目的长期“控制平面”。它同时回答六类问题：

1. **项目到底是什么**：它解决的工程问题、最终边界、与普通“写几个 CUDA kernel”项目的区别。
2. **现在究竟完成到哪里**：把“有代码”“有测试”“在真实硬件运行”“接入完整模型”“产生服务收益”严格区分。
3. **最终应该长成什么样**：模块、依赖方向、稳定契约、数据流、证据流和扩展边界。
4. **每个文件负责什么**：覆盖当前 Git 树中的全部 210 个文件，并说明未来应如何演化。
5. **每个阶段如何交付**：S00–S15 以及必须插入的 S04.5，每阶段给出输入、任务、产物、实验、门禁、验收标准和能力映射。
6. **怎样把项目转化为秋招竞争力**：所有性能结论必须落到可复查证据、简历 bullet、面试讲解和可演示资产。

本文使用以下事实标签。任何 Coding Agent 更新文档时都必须保留这种区分：

| 标签 | 含义 | 可以怎样表述 |
|---|---|---|
| `PLANNED` | 只有规划，没有实现 | “规划了”“将实现”，不能说“支持” |
| `SOURCE` | 源码或配置已存在，本次仅静态读取 | “已实现源码”，不能自动声称“正确/可运行” |
| `TEST` | 有对应自动化测试，并有当前基线的可复核测试结果 | “测试验证通过” |
| `RUNTIME` | 在目标硬件/软件栈真实运行，有原始数据和环境指纹 | “在指定环境实测” |
| `MODEL` | 已接入真实模型，完成端到端语义与性能验证 | “模型级有效” |
| `SERVICE` | 在并发、调度、缓存和 SLO 条件下验证 | “服务级有效” |
| `PORTABLE` | 至少两个硬件/后端按统一协议复现 | “可迁移/跨硬件” |

> 当前仓库文档中存在证据元数据漂移：最新 Git 基线是 `4bc59e7`，而 `docs/project_status.md` 和 `docs/evidence_ledger.md` 的正文仍写 `4dda6f8`。因此，本文把仓库中“runtime-verified”条目视为**仓库记录的历史运行声明**；在没有把原始证据纳入可访问证据包并重新绑定 commit/config/hash 前，不把它提升为本次独立实测结论。

---

## 1. 项目的根本使命

### 1.1 一句话定义

HQSB 应最终成为一个以**版本化契约和证据链**为骨架、贯通**模型语义—工作负载—性能剖析—高性能算子—量化—编译—推理 Runtime—在线 Serving—分布式—异构硬件**的 LLM 推理优化实验与工程平台。

它不应只是：

- 一个只有单算子 microbenchmark 的 CUDA demo；
- 一个把 vLLM、TensorRT-LLM 或 SGLang 包一层脚本的 benchmark；
- 一个只给出“快了 X%”但没有环境、正确性、原始数据和原因分析的作品集；
- 一个同时堆很多技术名词、却没有稳定模块边界和端到端闭环的“大杂烩”。

它要证明的是：你能够从一个真实模型和真实硬件上的瓶颈出发，做出可复现、可解释、可接入、可回退、能产生端到端收益的优化，并把它工程化为可维护的平台。

### 1.2 对目标岗位的价值

目标岗位是“推理优化 + 算子开发融合”的 AI Infra 工程师。招聘方实际寻找的不是会背 API 的人，而是能够跨层定位和解决问题的人：

| 岗位核心要求 | HQSB 必须给出的证据 |
|---|---|
| Linux、C++、Python 工程能力 | CMake/CUDA/C++ 算子、Python 编排、CI、测试、错误与日志规范 |
| PyTorch 与模型结构理解 | Qwen3 block、GQA、RoPE、KV Cache、RMSNorm、SwiGLU、prefill/decode 的调用与 shape 证据 |
| CUDA/SIMT 与 GPU 架构 | warp/block、访存合并、向量化、reduction、occupancy、register/shared memory、stream 语义 |
| 高性能算子 | 多版本 kernel、正确性门禁、benchmark、Nsight 解释、dispatcher、fallback |
| Triton/CUTLASS/TensorRT | 同一算子或 GEMM 的多后端实现、适用边界和可解释选型 |
| 低比特量化 | 权重量化、激活/KV 量化、校准、误差传播、量化算子、模型质量与吞吐权衡 |
| vLLM/SGLang/推理引擎 | paged KV cache、continuous batching、prefix cache、speculative decoding、CUDA Graph |
| AI Compiler | FX/Inductor/Triton/MLIR/TVM 等图捕获、pattern fusion、lowering、autotune 与代码生成 |
| 分布式与通信 | TP/PP/EP、NCCL、通信计算重叠、拓扑感知、RDMA 概念与测量 |
| 异构芯片 | CUDA 与 Ascend C/CANN 的等价契约、能力检测、统一 workload 和跨硬件报告 |
| Serving 与性能治理 | OpenAI 协议、调度、排队、SLO、P99、goodput、背压、取消、故障注入 |
| 科学实验与表达 | 假设—变量—原始数据—统计—结论—局限—复现命令—简历/面试证据链 |

### 1.3 项目最终要形成的纵向闭环

```text
真实模型与权重
  ↓ 固定模型、token、版本、hash
WorkloadSpec（prefill/decode/并发/序列长度）
  ↓
Reference Backend + 正确性基线
  ↓ profile before optimize
Profiler / Roofline / Amdahl / shape census
  ↓ 选择真正热点
CUDA / Triton / CUTLASS / Ascend 算子
  ↓ correctness before performance
torch.library / module replacement / FX pattern integration
  ↓ micro → block → model
量化 / 图编译 / Runtime / KV Cache / 调度
  ↓ model → service
Serving / 分布式 / 多硬件 / SLO
  ↓ evidence before claim
原始证据 → 标准结果 → 报告 → 回归门禁 → 简历与面试材料
```

### 1.4 本项目最有辨识度的四个“护城河”

1. **跨层归因**：不止告诉别人 kernel 快了，而是用 Amdahl 定律和模型级测量解释为什么端到端快或不快。
2. **真实语义接入**：不对模型做“看起来像”的替换；严格保持权重、数学、dtype、stream、autograd/inference mode 和生成语义。
3. **统一契约与证据**：同一套 ModelArtifact、WorkloadSpec、OperatorSpec、Backend、QuantArtifact、BenchmarkResult、TraceEvent 贯穿所有后端。
4. **异构可迁移性**：把硬件特性留在 backend/kernel 内，把模型、工作负载、指标和报告留在公共层。

---

## 2. 完成度与证据成熟度模型

### 2.1 功能成熟度等级

每个功能都必须标注等级，禁止仅用“完成/未完成”二元描述：

| 等级 | 名称 | 必须满足 |
|---|---|---|
| M0 | 规划 | 有目标和边界，没有源码 |
| M1 | 源码实现 | API/代码存在，静态结构合理 |
| M2 | 自动验证 | CPU 或设备测试覆盖正确性、异常路径和契约 |
| M3 | 硬件实测 | 目标硬件真实运行，记录环境、commit、配置、原始数据 |
| M4 | 模型集成 | 接入真实模型/真实 shape，算子级与模型级正确性通过 |
| M5 | 端到端收益 | 模型吞吐、TTFT、TPOT、内存或能耗有统计显著、可解释收益 |
| M6 | 服务/生产 | 并发、SLO、P99、故障、可观测、回退与部署验证 |
| M7 | 跨硬件复现 | 至少两种硬件/后端复现同一协议，结论可比较 |

### 2.2 性能结论的七道门

任何性能数字进入 README、简历或面试材料前，必须通过：

1. **身份门**：commit、配置 hash、模型 hash、OperatorSpec、环境指纹齐全。
2. **语义门**：reference 和 candidate 数学定义、输入域、dtype、epsilon、布局一致。
3. **正确性门**：max/mean/RMSE/cosine/L2 relative error，加首错位定位；模型输出也要验证。
4. **测量门**：warmup、同步、重复次数、percentile、冷/热启动边界明确。
5. **资源门**：显存、温度、功耗、频率、OOM、throttle、其他进程可解释。
6. **归因门**：通过 profiler、roofline、Amdahl 和消融实验说明“为什么”。
7. **外推门**：明确适用 shape、模型、硬件和不适用条件，禁止把单点 microbench 外推为全模型收益。

### 2.3 当前阶段判定

基于本次**静态阅读**，项目的合理判定如下：

| 阶段 | 仓库自述 | 本文判定 | 当前成熟度 | 说明 |
|---|---|---|---|---|
| S00 | 已完成 | 主体完成 | M1–M3 混合 | 基线、模型 manifest、基础测试与历史运行记录存在 |
| S01 | 已完成 | 主体完成 | M1–M2，部分历史 M3 | 契约、schema、registry、CI、日志、测试已成形，但存在可选依赖边界问题 |
| S02 | 已完成 | 主体完成 | M1–M3 | 基线、profiling、roofline、memory、工作负载存在；若要对外使用须重绑原始证据 |
| S03 | 已完成 | 主体完成 | M1–M3 | CUDA RMSNorm/fused 算子、多版本、bench/test 存在；尚未达到模型集成 M4 |
| S04 | 已完成/验收通过 | **主体实现完成，严格验收待补齐** | M1–M3，未到 M4 | 多后端代码已存在，但 dispatcher、tail safety、stream、CUTLASS 选择和证据可追溯性仍有关键缺口 |
| S04.5 | 仓库中没有 | **必须插入，下一阶段** | M0 | 模型理解、算子替换和端到端闭环；完成后再进入量化更有价值 |
| S05–S15 | 未开始/规划 | 规划 | M0 | 目录占位或文档方案，不能声称功能已支持 |

当前项目已能从源码层面展示的功能：

- 版本化契约、配置、schema 迁移、registry、日志、错误和 benchmark 结果骨架；
- Qwen3-1.7B 本地模型加载、manifest/hash、model-core 基准、工作负载与 profiling 辅助；
- CUDA RMSNorm V0/V1/V2、fused residual+RMSNorm、测试和 microbenchmark；
- Triton RMSNorm/GEMM、CUTLASS GEMM 对照、CUDA ctypes bridge、能力检测和 dispatcher；
- Jetson 环境/资源采集、功耗能量、Roofline/Amdahl、正确性与结果汇总；
- S00–S15 的路线规划和多类工程文档模板。

当前不能严谨声称已经具备的能力：

- 自定义算子已经无缝替换 Qwen3 模块或计算图；
- microbenchmark 加速已经转化为 TTFT/TPOT/tokens/s 收益；
- 量化、Runtime、Serving、Ascend、分布式、编译器、生产部署已经实现；
- CUTLASS、Triton、CUDA 的所有 shape 都安全、最优或自动正确选择；
- 所有历史性能数字能由新 clone 直接复现。

---

## 3. 最终顶层架构

### 3.1 分层架构

```text
┌──────────────────────────────────────────────────────────────────────┐
│  Experience / Evidence Plane                                        │
│  CLI · Experiment Registry · Reports · Dashboard · Resume Evidence  │
├──────────────────────────────────────────────────────────────────────┤
│  Serving Plane                                                      │
│  OpenAI API · Router · Scheduler · Admission · Streaming · SLO       │
├──────────────────────────────────────────────────────────────────────┤
│  Runtime Plane                                                      │
│  KV Manager · Continuous Batching · Prefix Cache · Spec Decode       │
│  CUDA Graph · Memory Planner · vLLM/SGLang/TRT-LLM adapters          │
├──────────────────────────────────────────────────────────────────────┤
│  Optimization Plane                                                 │
│  QuantLab · KernelLab · Fusion · Graph Compiler · Autotune           │
├──────────────────────────────────────────────────────────────────────┤
│  Framework Integration Plane                                        │
│  torch.library · Module Swap · FX/Inductor · Meta/FakeTensor         │
├──────────────────────────────────────────────────────────────────────┤
│  Backend & Hardware Plane                                            │
│  PyTorch · CUDA · Triton · CUTLASS · Ascend C/CANN · CPU/Edge       │
├──────────────────────────────────────────────────────────────────────┤
│  Stable Contract Plane                                              │
│  C1 Model · C2 Workload · C3 Operator · C4 Backend · C5 Quant       │
│  C6 Result · C7 Trace · Config · Registry · Schema · Error/Logging   │
└──────────────────────────────────────────────────────────────────────┘

横切能力：Correctness · Benchmark · Profiling · Observability · CI
          Reproducibility · Security · Documentation · Artifact Store
```

### 3.2 目标目录结构

目标架构尽量延续现有目录，不做无意义大迁移。新增内容优先落入已有边界：

```text
HeteroQuantServeBench/
├── hqsb/
│   ├── core/                 # 纯契约和基础设施，不依赖具体硬件/模型
│   ├── models/               # 模型制品、结构解析、shape census、适配器
│   ├── benchmark/            # 统一执行、正确性、指标、profile、报告数据模型
│   ├── backends/             # PyTorch/vLLM/SGLang/TRT-LLM/llama.cpp/Ascend 适配
│   ├── integration/          # [新增] torch.library、module swap、FX pattern
│   ├── quant/                # PTQ/QAT metadata、calibration、packing、KV quant
│   ├── runtime/              # [新增] KV、batching、cache、spec decode、graph
│   ├── serving/              # gateway、scheduler、router、SLO、observability
│   ├── distributed/          # [新增] TP/PP/EP、collective、topology
│   ├── compiler/             # [新增] FX/Inductor/MLIR/TVM lowering/autotune
│   └── hardware/             # Jetson/数据中心 GPU/Ascend 能力与实验协议
├── ops/
│   ├── cuda/                 # CUDA C++ 算子、稳定 C++/C ABI、bench/test
│   ├── triton/               # Triton kernel 与 autotune config
│   ├── cutlass/              # [可由现 cutlass_gemm 演进] CUTLASS/CuTe 算子
│   ├── ascend/               # Ascend C/TBE/CANN 算子
│   ├── common/               # [新增] 跨后端 spec、reference、测试向量
│   ├── capability.py         # 能力探测
│   ├── dispatcher.py         # 只做选择，不混入测量/下载/全局状态
│   └── bindings/             # [从 bridge 演进] torch extension/C ABI/bindings
├── configs/                  # 模型/算子/量化/backend/workload/环境配置
├── benchmarks/
│   ├── workloads/            # 版本化 workload suite
│   ├── schemas/              # raw/normalized/report schema
│   ├── scripts/              # 稳定用户入口
│   ├── raw/                  # 本机临时证据，不直接作为唯一归档
│   └── normalized/           # 可比较的标准化结果
├── experiments/              # [新增] 每项实验 manifest，不放临时脚本碎片
├── reports/                  # raw runtime artifact；由索引绑定对象存储/Release
├── docs/
│   ├── architecture/         # 架构、ADR、ownership、契约
│   ├── stages/               # 阶段执行计划
│   ├── reports/              # 结论性报告，禁止伪造未跑数字
│   ├── benchmark/            # 方法与指标口径
│   ├── interview/            # [新增] 项目叙事、FAQ、演示路径
│   └── templates/            # 实验/ADR/优化/交接/验收模板
├── scripts/                  # 环境、模型、profile、迁移、发布入口
├── tests/
│   ├── unit/                 # CPU/轻量逻辑
│   ├── correctness/          # 多后端数值与模型语义
│   ├── integration/          # torch.compile/module/backend/runtime
│   ├── performance/          # [新增] 非硬编码速度、受控 GPU 回归
│   └── test_vectors/         # 版本化输入/输出/误差阈值
├── cpp/                      # Runtime/edge worker/公共 C++ 基础设施
├── third_party/              # 固定版本、license、获取方法，不提交大仓库
└── .github/workflows/        # CPU CI、GPU nightly、文档、release、security
```

### 3.3 模块职责和依赖规则

| 模块 | 应负责 | 不应负责 |
|---|---|---|
| `hqsb/core` | 数据契约、schema、registry、错误、ID、日志、配置 | import torch/CUDA、加载具体模型、调用某个 backend |
| `hqsb/models` | 模型制品、hash、模型结构/shape、标准模型适配 | 调度策略、服务网络、硬件专属 kernel |
| `hqsb/benchmark` | 测量协议、指标、统计、正确性、profile 数据归一化 | 偷偷选择优化实现、修改模型语义 |
| `hqsb/backends` | 把统一契约适配到具体执行引擎 | 定义新的公共契约、把引擎特殊字段泄漏给上层 |
| `hqsb/integration` | torch.library、module/FX 替换、Meta/FakeTensor、编译集成 | 实现硬件 kernel 算法本体 |
| `ops/*` | 算子实现、能力、dispatcher、后端内 benchmark/test | 模型下载、HTTP serving、全局实验编排 |
| `hqsb/quant` | 校准、scale/zero point、packing、量化 artifact 和策略 | 与具体服务框架强绑定 |
| `hqsb/runtime` | KV cache、batching、memory、decode loop、执行计划 | HTTP API、简历报告生成 |
| `hqsb/serving` | 协议、路由、排队、SLO、流式、背压、服务观测 | kernel 内部实现 |
| `hqsb/distributed` | parallel plan、collective、拓扑、通信观测 | 模型数值 reference |
| `hqsb/compiler` | graph capture、pattern、lowering、codegen、autotune | 隐藏正确性/fallback 逻辑 |
| `hqsb/hardware` | 设备探测、电源/温度协议、硬件规范和计数器 | 模型业务语义 |

强制依赖方向：

```text
core ← models/benchmark ← backends/integration/quant
     ← ops adapters ← runtime ← serving

hardware 可被 benchmark/backends/ops/runtime 使用，不能反向 import 业务层。
docs/configs/scripts 可以编排上述模块，但不得成为唯一实现所在地。
```

### 3.4 稳定契约 C1–C7

| 契约 | 语义 | 最终必须具备的字段/能力 |
|---|---|---|
| C1 `ModelArtifact` | “跑的是哪个模型” | family、revision、weight hash、tokenizer hash、config、dtype、format、local/remote provenance、license |
| C2 `WorkloadSpec` | “跑了什么请求” | ISL/OSL、batch/concurrency、prompt token source、seed、warmup/repeat、arrival process、streaming、stop rules |
| C3 `OperatorSpec` | “算子数学与实现边界” | equation、shape/layout/stride、dtype、epsilon、accumulation、tolerance、alignment、stream、workspace、fallback、backend/version |
| C4 `Backend/Capability` | “谁能在何处执行” | probe、supported combinations、load/run/unload、health、reasoned fallback、actual selected implementation |
| C5 `QuantArtifact` | “量化后制品是什么” | scheme、group size、granularity、calibration set/hash、scale dtype、packing、kernel compatibility、quality delta |
| C6 `BenchmarkResult` | “结果怎样复核” | identity、raw sample、percentiles、memory/power/energy、correctness、selected backend、failure、artifact URIs |
| C7 `TraceEvent` | “一次请求发生了什么” | run/trace/span、stage、timestamps、queue/prefill/decode、kernel/backend、batch/cache、error、hardware state |

契约升级必须满足：schema version 增量、旧版本迁移、未来版本拒绝、未知字段策略、文档与测试同步。不能让某个 backend 的临时字段直接污染公共 schema。

### 3.5 一次实验的数据流

1. 配置加载器合并 defaults/file/env/CLI，生成稳定 `config_hash`。
2. `ModelArtifact` 验证权重、tokenizer、revision 和 hash。
3. `WorkloadSpec` 固定 token 输入、prefill/decode 边界、重复与并发。
4. Capability 探测产生可解释能力矩阵；dispatcher 返回“选择结果 + 原因”，不能只返回函数。
5. Backend 执行 reference/candidate；TraceEvent 贯穿 operator、block、model、service。
6. Correctness gate 先决定是否允许性能测量。
7. Benchmark engine 保存逐次 raw samples，再计算 summary；summary 不能替代 raw。
8. Profiler/硬件计数器与结果通过 run_id 对齐。
9. Normalizer 生成跨硬件可比较结果，同时保留原始单位与上下文。
10. Report generator 输出假设、方法、结果、原因、局限、复现路径和 claim 等级。

### 3.6 最终质量标准

- **高内聚**：算子、模型、Runtime、Serving 各自闭环；实验入口不复制核心逻辑。
- **低耦合**：公共层不依赖具体 backend；更换 Qwen3/硬件/Runtime 不改公共 schema。
- **可回退**：任何优化都可按 capability、shape、dtype、错误或数值风险回退 reference。
- **可诊断**：失败必须有稳定错误类型、上下文、选择原因和 trace，不允许静默降级。
- **可复现**：commit + config hash + model hash + environment + raw samples 足以重放。
- **可测试**：unit/correctness/integration/performance/service 分层，GPU 不可用时明确 skip 原因。
- **可观测**：micro/model/service 的指标能通过 run_id/trace_id 关联。
- **可维护**：每个优化有 owner、ADR、OperatorSpec、测试、benchmark、报告和失效条件。

---

## 4. 当前架构与最终架构差距

| 领域 | 当前仓库 | 最终目标 | 关键缺口 |
|---|---|---|---|
| 契约与工程地基 | C1–C7、schema、config、registry、logging 已存在 | 贯穿所有后端和实验 | optional dependency 边界、schema 表达力、artifact 可追溯性 |
| 模型 | Qwen3 loader/manifest/config/smoke | 架构解析、shape census、多模型 adapter | 没有模块调用台账、真实算子替换、block/model correctness |
| Benchmark | model-core、metrics、memory、power、profile | micro→model→service 统一实验 | 口径一致性、字段实际生效、raw artifact 归档 |
| CUDA 算子 | RMSNorm、fused residual+RMSNorm 多版本 | 稳定 torch op、真实 stream、全面 shape/dtype | C ABI stream=0、validation、模型接入 |
| Triton | RMSNorm/GEMM | tail-safe、可调优、持久 cache、编译集成 | GEMM tail safety、autotune cache/失效、异常路径 |
| CUTLASS | 单个默认 GEMM bench | 版本固定、策略/shape 矩阵、dispatcher 选择 | 未接 dispatcher、状态检查与数值门禁不足 |
| 量化 | 目录占位 | 权重/激活/KV 多方案与 kernel | S05 全部 |
| 编译/框架 | 尚无稳定 integration 层 | torch.library、Meta、module、FX/Inductor | S04.5/S06 |
| Runtime | 尚无 | KV、batching、cache、spec decode、CUDA Graph | S07 |
| Serving | 占位 | OpenAI API、SLO、routing、observability | S08 |
| Ascend | 占位 | Ascend C/CANN 等价后端 | S09 |
| 分布式 | 尚无 | TP/PP/EP、NCCL、拓扑、通信重叠 | S10 |
| 编译器 | 尚无 | FX/Inductor/MLIR/TVM 路线 | S11 |
| 跨硬件 | Jetson 协议 | 统一硬件矩阵和归一化 benchmark | S12 |
| 生产化 | CPU CI 雏形 | GPU nightly、release、container、K8s、SRE | S13/S15 |
| 证据 | 文档 ledger + 被忽略的本地 raw | 可下载、可复核、绑定版本的 evidence bundle | status/commit 漂移，raw 未随仓库分发 |

---

## 5. 当前目录与全部文件用途

### 5.1 根目录与工程入口

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `.github/workflows/ci.yml` | CPU CI，覆盖 Python 版本和基础测试/文档流程 | 恢复并强制 lint、format、mypy；拆分 CPU PR gate、GPU nightly、release、security；不可用能力要显式 skip |
| `.gitignore` | 排除模型权重、构建、缓存、本地 reports 等 | raw 不能仅因被忽略而失去证据；增加 evidence index/对象存储，不提交密钥和大模型 |
| `CMakeLists.txt` | 顶层 CMake，组织 CUDA 子项目 | 统一 options、架构列表、依赖版本、install/export、测试开关和 sanitizer 配置 |
| `README.md` | 项目入口、当前功能、快速开始、阶段状态 | 只保留已证实能力；性能 claim 链接 evidence；避免与 `project_status.md` 漂移 |
| `pyproject.toml` | Python 包、依赖组、pytest/工具配置 | 核心包保持 CPU-minimal；benchmark/torch/triton/serving/ascend 分组；建立静态检查门禁 |
| `third_party/README.md` | 外部依赖获取说明，尤其 CUTLASS | 必须固定 commit/tag、license、校验和、兼容矩阵；不能依赖外部 main 分支 |

### 5.2 `benchmarks/`：可复现实验资产

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `benchmarks/normalized/.gitkeep` | 保留标准化结果目录 | 后续存小型、脱敏、可提交的跨硬件汇总；schema/version 必须明确 |
| `benchmarks/raw/.gitkeep` | 保留原始结果目录 | 本机 raw 可暂存，但正式证据应由 manifest 指向 Release/对象存储 |
| `benchmarks/schemas/golden_reference_schema.json` | golden reference JSON Schema | 与 C6、模型 hash、tokenizer hash、tolerance 和迁移版本对齐 |
| `benchmarks/scripts/generate_golden.py` | 生成模型 golden 输出 | 固定 seed/token IDs/model hash；避免只存文本；输出首 token/logit/hidden 可选证据 |
| `benchmarks/scripts/run_jetson_baseline.py` | 编排 Jetson 多 workload 基线 | 通过统一 engine/contract，不复制指标口径；记录电源、温度和 throttle |
| `benchmarks/scripts/run_model_core.py` | 单 workload 的 model-core runner | 明确 tokenizer/HTTP 不在口径内；所有 WorkloadSpec 字段必须实际生效 |
| `benchmarks/scripts/summarize_baseline.py` | 汇总 raw 结果到 CSV | 保留样本数、误差/置信区间、失败，不只输出均值 |
| `benchmarks/workloads/golden/isl32_osl32.json` | ISL=32、OSL=32 历史 golden | 补齐当前 schema/model hash并进入 correctness gate |
| `benchmarks/workloads/golden/isl128_osl32.json` | ISL=128、OSL=32 历史 golden | 同上；代表短上下文常规 decode |
| `benchmarks/workloads/golden/isl512_osl32.json` | ISL=512、OSL=32 历史 golden | 同上；代表中等 prefill |
| `benchmarks/workloads/golden/isl2048_osl32.json` | ISL=2048、OSL=32 历史 golden | 同上；代表长 prefill；需与模型上下文配置绑定 |

### 5.3 `configs/`：单一事实源

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `configs/backends/.gitkeep` | backend 配置占位 | 增加 PyTorch/vLLM/SGLang/TRT-LLM/Ascend 配置和 capability override |
| `configs/benchmarks/.gitkeep` | benchmark 配置目录占位 | 目录已有实际 YAML；可移除占位但不是必要动作 |
| `configs/benchmarks/jetson_qwen3_fp16.yaml` | Jetson Qwen3 FP16 工作负载与测量设置 | 成为唯一 workload source；覆盖 batch/concurrency/seed/repeat/streaming 并验证字段生效 |
| `configs/environment/jetson_python_lock.txt` | Jetson Python 环境快照 | 当前较宽；最终拆 runtime-minimal 与 full-dev lock，记录来源和架构 |
| `configs/environment/jetson_runtime.txt` | CUDA/PyTorch/系统关键版本 | 自动采集并绑定 run_id，补 driver、firmware、power mode、clock、container digest |
| `configs/models/.gitkeep` | 模型配置目录占位 | 目录已有 Qwen3 配置；未来保留多模型 adapter 的配置索引 |
| `configs/models/qwen3_1_7b.yaml` | Qwen3-1.7B 模型身份、路径和运行配置 | 加 revision/hash/tokenizer/architecture facts；禁止把机器路径当可移植配置 |
| `configs/operators/fused_residual_rmsnorm.json` | fused residual+RMSNorm 的 C3 OperatorSpec | 补数学方程、output/residual 语义、累加 dtype、alias、stream、layout、shape 域 |
| `configs/operators/rmsnorm_v0.json` | CUDA RMSNorm V0 spec | 标记 reference/baseline、限制和数值阈值 |
| `configs/operators/rmsnorm_v1.json` | CUDA RMSNorm V1 spec | 记录 warp reduction、适用 shape/occupancy、fallback |
| `configs/operators/rmsnorm_v2.json` | CUDA RMSNorm V2 spec | 记录 vectorized alignment、tail、float4/half2 和不适用边界 |
| `configs/quantization/.gitkeep` | 量化配置占位 | S05 增加 W8A16/W4A16/W8A8/KV8/KV4、校准和 packing 配置 |

### 5.4 `cpp/`：C++ Runtime 与边缘执行占位

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `cpp/common/.gitkeep` | 公共 C++ 基础设施占位 | 放 status/result、logging、allocator、trace、contract parser；避免复制 CUDA helper |
| `cpp/edge_worker/.gitkeep` | 边缘 worker 占位 | S07/S13 实现轻量执行进程、健康检查、模型生命周期与 IPC |

### 5.5 `docs/architecture/`：架构事实

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `docs/architecture/module_ownership.md` | 模块 ownership 与依赖方向 | 增加 integration/runtime/compiler/distributed；把禁止依赖变成自动静态检查 |
| `docs/architecture/portable_kernel_backends.md` | CUDA/Triton/CUTLASS/TileLang/HIP 等后端路线说明 | 修正 CUTLASS 与 ROCm 可移植性表述；每条能力链接官方版本与本仓库实证 |
| `docs/architecture/顶层架构.md` | 当前顶层方案和 C1–C7 定义 | 以本文架构为后续修订依据；加入 S04.5、证据平面和分层 maturity |

### 5.6 `docs/benchmark/`：测量口径

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `docs/benchmark/methodology.md` | benchmark 方法、边界和复现规范 | 统一 micro/block/model/service；增加统计、thermal、冷/热、并发、失败处理 |
| `docs/benchmark/metric_definitions.md` | TTFT、TPOT、吞吐、功耗等指标定义 | 与代码公式逐项一致；当前需重点核对 decode_tokens_per_s 口径 |
| `docs/benchmark/model_sha256_manifest.txt` | 模型文件 SHA256 清单 | 不自引用；绑定具体 revision；生成脚本与验证脚本互为门禁 |
| `docs/benchmark/qwen3_model_manifest.json` | Qwen3 模型 manifest | 加 tokenizer、config、总字节、文件计数、来源、license 和 schema version |

### 5.7 `docs/` 根状态文件

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `docs/hardware/jetson_environment.md` | Jetson 硬件、系统、CUDA/PyTorch 与运行协议说明 | 与自动环境采集结果绑定；补 power mode、clock、temperature、throttle、firmware、容器和采集时间 |
| `docs/evidence_ledger.md` | claim 到代码/测试/运行证据的台账 | 当前基线 commit 已漂移；改为机器可校验条目，加入 evidence URI、hash、状态过期规则 |
| `docs/project_status.md` | 仓库当前状态 Source of Truth | 当前正文基线 `4dda6f8` 落后于审计基线 `4bc59e7`；应由脚本从 Git/ledger 生成关键元数据 |

### 5.8 `docs/reports/`：现有报告

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `docs/reports/S00_阶段验收报告.md` | S00 基线恢复与验收记录 | 保留历史事实；新增验收必须绑定当前 commit 和证据包 |
| `docs/reports/S01_开发报告.md` | S01 契约/工程实现说明 | 将设计决策沉淀为 ADR，避免只罗列文件 |
| `docs/reports/S01_阶段验收报告.md` | S01 门禁与结论 | 区分测试结果和静态声明，记录命令/环境 |
| `docs/reports/S02_开发报告.md` | S02 模型基线和 profiling 开发说明 | 补全真实 shape 与 profiler artifact 索引 |
| `docs/reports/S02_阶段验收报告.md` | S02 验收记录 | 将 raw profiler、model hash、config hash 绑定为证据包 |
| `docs/reports/S03_benchmark_report.md` | CUDA 算子性能报告 | 增加置信区间、Nsight 指标和模型形状覆盖率 |
| `docs/reports/S03_开发报告.md` | S03 算子开发过程 | 保留失败尝试、版本差异和 design rationale |
| `docs/reports/S03_阶段验收报告.md` | S03 correctness/bench 验收 | 在模型集成完成前只声称 M3，不声称 M4/M5 |
| `docs/reports/S04_comparison_report.md` | CUDA/Triton/cuBLAS/CUTLASS 对照 | 修正工作负载 shape，补 tail/stream/dispatcher 和可复现 raw |
| `docs/reports/S04_开发报告.md` | S04 多 DSL 开发说明 | 明确“代码路径存在”与“dispatcher 真正选择”的差异 |
| `docs/reports/S04_阶段验收报告.md` | S04 自验收 | 当前结论需在 P0/P1 缺口修复后重验；保留原文作为历史版本 |
| `docs/reports/baseline_report.md` | 六 workload 模型基线与内存画像 | 口径与原始数据建立双向索引；区分 model-core 与 service |
| `docs/reports/fused_residual_optimization_log.md` | fused residual+RMSNorm 优化日志 | 补语义、读写流量模型、RAW 依赖和未获益原因 |
| `docs/reports/ncu_report.md` | Nsight Compute kernel 指标分析 | 每条结论绑定 `.ncu-rep`/CSV、kernel 名称和 shape |
| `docs/reports/nsys_report.md` | Nsight Systems 时间线分析 | 对齐 prefill/decode/CPU launch/同步和 trace_id |
| `docs/reports/pytorch_profile_report.md` | PyTorch Profiler 热点分析 | 补调用栈、shape、调用次数、self/total time 与 selection rationale |
| `docs/reports/rmsnorm_optimization_log.md` | RMSNorm V0→V2 优化日志 | 作为面试核心材料，补真实模型调用、stream 和端到端结果 |

### 5.9 `docs/stages/`：阶段计划文件

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `docs/stages/S00_现状审计与基线恢复.md` | 仓库审计、恢复、事实边界 | 保留；所有阶段开始前复用“基线冻结”流程 |
| `docs/stages/S01_核心契约与工程质量体系.md` | 契约、schema、registry、CI 地基 | 补 optional dependency 和 artifact store 质量项 |
| `docs/stages/S02_模型基线与全栈Profiling.md` | 模型 workload、profiling、roofline | 增加完整 architecture/shape census，为 S04.5 提前产出输入 |
| `docs/stages/S03_CUDA算子性能工程.md` | CUDA 多版本算子闭环 | 增加 torch op binding、current stream 和模型级准备条件 |
| `docs/stages/S04_Triton_CUTLASS与KernelDSL.md` | 多 DSL、多后端对照 | 增加 tail-safe、CUTLASS dispatch、persistent autotune、真实 shape 门禁 |
| `docs/stages/S05_量化与低精度推理.md` | 量化路线 | 在其前新增 S04.5；S05 只接已验证的集成接口 |
| `docs/stages/S06_框架集成与图优化.md` | 框架适配、图优化/融合 | 将基础 torch.library/module/FX 接入前移至 S04.5，S06 专注规模化 graph/compiler |
| `docs/stages/S07_推理Runtime内核.md` | KV、batching、runtime adapter | 与 vLLM/SGLang/TRT-LLM/llama.cpp 统一 C4 backend |
| `docs/stages/S08_ServeFabric与性能治理.md` | Serving、路由、SLO、观测 | 必须用真实并发和故障场景，不只启动 HTTP server |
| `docs/stages/S09_AscendC_CANN异构后端.md` | Ascend C/CANN 后端 | 保持等价 C3/C4/C6 契约，记录 CUDA↔Ascend mapping |
| `docs/stages/S10_分布式推理与通信.md` | TP/PP/EP、NCCL、通信优化 | 增加网络拓扑、集群环境、通信计算 overlap 证据 |
| `docs/stages/S11_AI编译器与自动优化.md` | graph/lowering/autotune/compiler | 以 S04.5 的 FX/Meta 基础为输入，形成可观察 lowering pipeline |
| `docs/stages/S12_跨硬件评估与统一Benchmark.md` | 跨硬件比较 | 禁止直接用绝对延迟排名；加入成本、能效、容量、精度和 capability |
| `docs/stages/S13_生产化云原生与可靠性.md` | 容器、K8s、SRE、安全 | 加 canary、rollback、容量规划、混沌、供应链和 SBOM |
| `docs/stages/S14_训推协同与前沿扩展.md` | 训推协同和前沿方法 | 只选能形成闭环的方向，避免为了名词堆砌扩张 |
| `docs/stages/S15_发布开源与求职证据.md` | 发布、开源、演示和求职 | 把所有 claim 固化成 release/evidence/演示/面试材料 |

### 5.10 `docs/templates/`：过程规范

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `docs/templates/adr.md` | Architecture Decision Record 模板 | 每个重大接口/依赖/算法选择都写 context/options/decision/consequence |
| `docs/templates/experiment.md` | 实验计划与报告模板 | 增加 hypothesis、controlled variables、raw URI、statistical test、claim level |
| `docs/templates/handoff.md` | Agent/开发者交接模板 | 加基线 commit、已改文件、未完成、风险、复现命令、禁止事项 |
| `docs/templates/optimization_log.md` | 优化迭代日志模板 | 每次迭代记录瓶颈证据、改动、正确性、性能、反例、回退条件 |

### 5.11 `hqsb/` 包入口与 backend

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `hqsb/__init__.py` | Python 包版本和顶层元信息 | 不要在包导入时加载 torch/Triton/CUDA；版本与 release tag 对齐 |
| `hqsb/backends/.gitkeep` | backend 目录历史占位 | 目录已有实现；可以保留但不承担功能 |
| `hqsb/backends/__init__.py` | backend 导出入口 | 必须 lazy import 可选依赖，CPU minimal 安装不能因 torch 缺失失败 |
| `hqsb/backends/dummy.py` | 确定性的 C4 Backend 参考实现 | 长期作为 contract conformance、engine 和故障测试基准 |
| `hqsb/backends/pytorch.py` | PyTorch Qwen3 FP16 reference backend | 让 WorkloadSpec 全字段生效；输出实际执行配置、选中 attention 和 fallback；模型 identity 需包含 revision/hash |

### 5.12 `hqsb/benchmark/`：统一测量内核

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `hqsb/benchmark/.gitkeep` | benchmark 目录历史占位 | 无运行语义 |
| `hqsb/benchmark/__init__.py` | benchmark 公共 API 导出 | 使用 lazy/optional import，避免核心 import 被 torch 等可选依赖污染 |
| `hqsb/benchmark/cli.py` | argparse 正整数/非负数校验 | 扩充配置错误到统一错误码，不在多个脚本复制 validator |
| `hqsb/benchmark/correctness.py` | golden、确定性、首错位与数值比较 | 增加 operator/block/model/service 分层 oracle，支持 logits/top-k/KL/生成一致性 |
| `hqsb/benchmark/engine.py` | 通过 Backend 契约执行 benchmark 并生成 C6 | 统一 warmup/sample/sync；修正 decode throughput 口径；记录实际 backend/implementation |
| `hqsb/benchmark/memory.py` | 权重、KV cache、RSS/swap 等核算 | 加 allocator reserved/allocated/peak、fragmentation、workspace、paged KV 利用率 |
| `hqsb/benchmark/metrics.py` | percentile、latency summary、数值差异 | 加 bootstrap CI、outlier policy、goodput、TPOT inter-token distribution |
| `hqsb/benchmark/model_core.py` | prefill/first-token/decode 三阶段模型基准 | 成为所有 backend 的模型级公共 runner；与 engine/文档公式唯一化 |
| `hqsb/benchmark/profiling.py` | PyTorch Profiler operator 表抽取 | 输出 shape/call count/stack/module/phase；能和 Nsight kernel 通过 trace 对齐 |
| `hqsb/benchmark/resource_monitor.py` | 后台 tegrastats 采集 | 抽象为 Monitor contract；支持 nvidia-smi/DCGM/Ascend 指标，记录采样丢失 |
| `hqsb/benchmark/roofline.py` | arithmetic intensity、roofline、Amdahl 分析 | 使用实测带宽/算力，不只理论峰值；加入 fusion 前后 bytes/FLOPs 模型 |
| `hqsb/benchmark/tegrastats_parser.py` | Jetson tegrastats 解析与能量积分 | 对固件格式变化容错；保存原始行、时间戳和解析错误比例 |
| `hqsb/benchmark/workload.py` | 固定长度 token workload 构造 | 支持真实/合成 token、长度分布、共享前缀、arrival process，并保证可重放 |
| `hqsb/benchmark/workload_config.py` | 从 YAML 加载标准 workload suite | schema 校验、默认值展开、config hash，禁止未知字段静默忽略 |

### 5.13 `hqsb/core/`：稳定控制面

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `hqsb/core/.gitkeep` | core 目录历史占位 | 无运行语义 |
| `hqsb/core/__init__.py` | core API 出口 | 只能导出纯 Python 稳定对象，避免反向导入具体实现 |
| `hqsb/core/config/__init__.py` | 配置模块导出 | 保持小而稳定 |
| `hqsb/core/config/loader.py` | defaults < file < env < CLI 合并与 hash | 对敏感字段脱敏；记录展开后配置；列表/嵌套 merge 语义写入契约 |
| `hqsb/core/contracts/__init__.py` | C1–C7 导出 | 解决可选依赖导入链；避免 `import hqsb.core` 触发模型/backend |
| `hqsb/core/contracts/backend.py` | C4 capability/backend 抽象 | 增加 setup/teardown/health/cancel/actual selection，区分静态与运行能力 |
| `hqsb/core/contracts/base.py` | 版本化 Pydantic 契约基类 | 统一额外字段、serialization、schema version、hash 和迁移策略 |
| `hqsb/core/contracts/model.py` | C1 ModelArtifact | 加 provenance、tokenizer/config hash、quant/adapter、完整 identity |
| `hqsb/core/contracts/operator.py` | C3 OperatorSpec | 当前 dtype/tolerance 表达偏简；扩展输入级 dtype、layout、stride、alias、accumulation、shape constraints |
| `hqsb/core/contracts/quant.py` | C5 QuantArtifact | S05 加 calibration、scale、packing、kernel/runtime compatibility 和质量结果 |
| `hqsb/core/contracts/result.py` | C6 BenchmarkResult | 加 raw sample URI/hash、actual implementation、correctness status、phase metrics、failure samples |
| `hqsb/core/contracts/trace.py` | C7 TraceEvent | 贯通 operator→model→runtime→service；与 OpenTelemetry 映射 |
| `hqsb/core/contracts/workload.py` | C2 WorkloadSpec | 扩并发/arrival/stream/cancel/shared-prefix/speculative 等字段，并确保 backend 真正执行 |
| `hqsb/core/errors.py` | 错误分类与稳定退出码 | 增加 error cause chain、retryability、backend/device context，保持兼容 |
| `hqsb/core/ids.py` | run/trace/span ID 生成 | 支持确定性实验 ID 与全局唯一 trace ID 的不同需求 |
| `hqsb/core/logging.py` | JSONL 结构化日志与 trace context | 统一字段、脱敏、采样，最终可导出 OpenTelemetry |
| `hqsb/core/registry/__init__.py` | registry 导出 | 保持纯 Python |
| `hqsb/core/registry/registry.py` | backend/operator/quantizer/monitor/reporter 注册 | 注册条目需携带 capability、版本、优先级、来源；冲突与卸载可诊断 |
| `hqsb/core/schema/__init__.py` | schema API 导出 | 保持稳定 |
| `hqsb/core/schema/migrate.py` | legacy golden/result/operator 迁移 | 所有迁移无损或明确 loss report；增加 dry-run 和批量 artifact index |
| `hqsb/core/schema/versioning.py` | schema version 和迁移链 | 未来版本拒绝、循环/缺失迁移检测、兼容策略和测试 |

### 5.14 `hqsb/hardware/`、`hqsb/models/` 与未来域

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `hqsb/hardware/__init__.py` | 硬件能力模块导出 | 只导出轻量接口，设备库 lazy import |
| `hqsb/hardware/jetson.py` | Jetson 温度、电源模式、冷却和实验协议 | 加命令结果、权限失败、throttle、clock lock、恢复原状态和 run metadata |
| `hqsb/models/__init__.py` | 模型 API 导出 | lazy import transformers/torch，导出 architecture adapter |
| `hqsb/models/loader.py` | local-only Qwen3 加载、dtype/attention、OOM fallback、释放 | identity 不只 model_id；fallback 必须可见；加入 device map 与量化 artifact 兼容检查 |
| `hqsb/models/manifest.py` | manifest 解析和文件 hash 验证 | 修复空 manifest、路径穿越、重复项；canonical path 必须位于模型根目录 |
| `hqsb/quant/.gitkeep` | QuantLab 占位 | S05 加 calibration、quantizer、packing、evaluation、kernel compatibility |
| `hqsb/serving/.gitkeep` | ServeFabric 占位 | S08 加 API/gateway/scheduler/router/SLO/metrics；避免所有逻辑堆一个 server.py |

### 5.15 `ops/` Python 控制与 Triton

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `ops/__init__.py` | 算子包入口 | 不应导入不可用 Triton/CUDA；能力缺失时核心/reference 仍可用 |
| `ops/_tilelang_probe.py` | TileLang 最小 kernel 能力实测探针 | 固定 probe 语义、超时、缓存 key 和错误原因；不要在 import 时编译 |
| `ops/ascend/.gitkeep` | Ascend 算子占位 | S09 实现与 C3 等价的 RMSNorm/GEMM/量化算子 |
| `ops/capability.py` | Triton/CUTLASS/TileLang/cuBLAS/CUDA 库能力检测 | 能力缓存须包含设备、版本、进程环境；返回结构化 reason，不吞掉所有错误 |
| `ops/cuda_bridge.py` | ctypes 绑定 CUDA RMSNorm shared library | 当前需补 device/dtype/shape/contiguous/alignment/stream 校验；最终优先 torch.library/C++ extension |
| `ops/dispatcher.py` | capability→arch→shape→fallback 的选择器 | 解除 torch/cuBLAS fallback 对 Triton import 的耦合；真正选择 CUTLASS；返回选择原因；GEMM 考虑 M/N/K/dtype/layout |
| `ops/triton/.gitkeep` | Triton 目录历史占位 | 无运行语义 |
| `ops/triton/__init__.py` | Triton 算子导出 | Triton 缺失时可安全 import 包，调用时再报告 capability |
| `ops/triton/gemm.py` | Triton GEMM reference/autotune | P0：K/N/M tail mask 正确；真实 Qwen shape；persistent cache；配置失效；M=1 专门策略；Meta/compile 集成 |
| `ops/triton/rmsnorm.py` | Triton RMSNorm reference/autotune | 支持真实行数/hidden/dtype/epsilon、循环覆盖、稳定 autotune、数值门禁和 current stream |

### 5.16 `ops/cuda/common` 与设备探针

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `ops/cuda/common/cuda_check.cuh` | CUDA API 错误检查宏 | 转换为可传播 status/exception，避免库代码直接终止进程；带 device/stream context |
| `ops/cuda/common/test_metrics.h` | C++ 五项数值误差指标 | 与 Python correctness 共享阈值语义；处理 NaN/Inf/零范数 |
| `ops/cuda/common/test_util.h` | 轻量 C++ 测试辅助 | 长期可接 Catch2/GoogleTest，但无需为框架而重构；必须有清晰失败上下文 |
| `ops/cuda/device_query/CMakeLists.txt` | 构建设备查询程序 | 纳入 install/test，可配置 CUDA arch |
| `ops/cuda/device_query/device_query.cu` | 输出 CUDA 设备与关键属性 | 增加 memory bandwidth 计算所需参数、SM/register/shared、compute capability、driver/runtime |

### 5.17 `ops/cuda/rmsnorm/`：核心 CUDA 算子实验室

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `ops/cuda/rmsnorm/CMakeLists.txt` | 构建 RMSNorm shared lib、test、bench | 固定 arch/options；支持 sanitizers、debug line info、install/export 和 torch binding |
| `ops/cuda/rmsnorm/bench/bench_rmsnorm.cu` | 多版本 RMSNorm microbenchmark | 使用真实 shape ledger，输出 JSON raw、选择版本、bytes/FLOPs、温度；统计而非单值 |
| `ops/cuda/rmsnorm/include/hqsb/rmsnorm.h` | 公共 C++/C API 契约 | 接受显式 `cudaStream_t`；明确 ownership、workspace、alignment、dtype、error code、ABI version |
| `ops/cuda/rmsnorm/src/rmsnorm_c_api.cu` | 给 ctypes 的稳定 C ABI | 当前固定 default stream 是关键缺口；加入 current stream 参数、参数校验和错误返回 |
| `ops/cuda/rmsnorm/src/rmsnorm_dispatcher.cu` | 在 V0/V1/V2 中选择实现 | 以 shape/dtype/alignment/arch 为 key；返回选择原因；支持强制版本用于消融 |
| `ops/cuda/rmsnorm/src/rmsnorm_launchers.h` | 内部 launcher 声明 | 保持不暴露内部模板；统一 stream 和错误语义 |
| `ops/cuda/rmsnorm/src/rmsnorm_reference.cu` | CUDA/host reference 路径 | reference 以正确性优先；明确 FP32 accumulation 和 epsilon |
| `ops/cuda/rmsnorm/src/rmsnorm_v0.cu` | 基础 shared-memory/reduction 版本 | 用作教学和性能基线，不删除；报告瓶颈和资源使用 |
| `ops/cuda/rmsnorm/src/rmsnorm_v1.cu` | warp shuffle 优化版本 | 说明 warp reduction、跨 warp 汇总、适用 hidden/block |
| `ops/cuda/rmsnorm/src/rmsnorm_v2.cu` | vectorized float4/half2 与 tail fallback | 严格 alignment/tail；解释 FP16 事务宽度、occupancy 与小 hidden 退化 |
| `ops/cuda/rmsnorm/tests/test_rmsnorm.cu` | dtype/shape/version/dispatcher/异常 correctness | 增加真实 Qwen shape、non-contiguous/stream、多 batch rows、NaN/Inf、随机/极值、sanitizer |

### 5.18 `ops/cuda/fused_residual_rmsnorm/`

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `ops/cuda/fused_residual_rmsnorm/CMakeLists.txt` | 构建 fused 算子/test/bench | 与 RMSNorm 公共基础设施统一，输出 shared/torch binding |
| `ops/cuda/fused_residual_rmsnorm/bench/bench_fused_residual_rmsnorm.cu` | 融合前后 microbenchmark | 比较 unfused 两 kernel 与 fused；精确计算减少的 HBM bytes 和中间张量 |
| `ops/cuda/fused_residual_rmsnorm/include/hqsb/fused_residual_rmsnorm.h` | fused 算子公共接口 | 明确输出是 residual、normalized 或二者；alias/in-place、stream、dtype、epsilon |
| `ops/cuda/fused_residual_rmsnorm/src/fused_residual_rmsnorm.cu` | V0/V1 fused kernel | 当前特别注意 FP16 residual 写回舍入改变后续语义；模型接入需逐层误差验证 |
| `ops/cuda/fused_residual_rmsnorm/tests/test_fused_residual_rmsnorm.cu` | fused correctness/版本对照 | 加 block/model reference、长生成误差、in-place/alias/stream 和随机 shape |

### 5.19 `ops/cuda/cutlass_gemm/`

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `ops/cuda/cutlass_gemm/CMakeLists.txt` | 构建 CUTLASS FP16 GEMM bench | 固定 CUTLASS commit，显式 include/arch，支持多个 tile/epilogue 配置 |
| `ops/cuda/cutlass_gemm/bench_cutlass_gemm.cu` | 默认 tensor-op GEMM 对照 | 加 `can_implement`、initialize/run status、CUDA error、阈值 gate；输出 tile/stage/alignment；接入 dispatcher |

### 5.20 `scripts/bench/`：性能与分析入口

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `scripts/bench/analyze_hotspots.py` | 从 profile 生成 Roofline/Amdahl 热点决策 | 输入/输出 schema 化；报告 hotspot coverage 和选择理由 |
| `scripts/bench/bench_s04.py` | S04 CUDA/Triton/cuBLAS/CUTLASS 对照 | 修正 Qwen3 intermediate 为 6144 等真实 shape；全 tail matrix；原始样本；CUTLASS 统一调用；结果绑定 commit |
| `scripts/bench/dump_triton_ir.py` | 导出 Triton IR/metadata | 固定 kernel/config/shape，归档 TTIR/TTGIR/LLVM/PTX/metadata 并与性能 run 对齐 |
| `scripts/bench/ncu_profile.sh` | Nsight Compute 采集 | 参数化 kernel/shape/section；输出工具版本、命令和 raw artifact hash |
| `scripts/bench/nsys_profile.sh` | Nsight Systems 采集 | 加 NVTX phase，输出 model/block/kernel 对齐 timeline |
| `scripts/bench/profile_model.py` | PyTorch Profiler 模型采集 | 按 prefill/decode 分段；记录 shape/module stack/call count/export trace |
| `scripts/bench/run_jetson_baseline.sh` | Jetson 基线一键入口 | 失败安全、环境恢复、显式输出目录和 run manifest；不改写系统状态后遗留 |
| `scripts/bench/run_s02_baseline.py` | 契约化 S02 baseline 编排 | 统一走 BenchmarkEngine，禁止与旧 runner 产生两套指标公式 |

### 5.21 `scripts/` 其他入口

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `scripts/check_docs.py` | 检查相对链接完整性 | 加 status/commit/file inventory/schema 示例一致性，成为 CI 门禁 |
| `scripts/env/collect_jetson_env.sh` | 采集 Jetson 硬件软件环境 | 脱敏 hostname/path；记录功耗模式、clock、temperature、driver/runtime、容器 digest |
| `scripts/migrate_legacy.py` | 旧 golden/result/spec 迁移 CLI | dry-run、目录批量、loss report、幂等、备份与明确退出码 |
| `scripts/models/download_qwen3_modelscope.py` | 从 ModelScope 下载 Qwen3 | 下载与运行分离；固定 revision、临时目录、hash 校验、失败清理、license 提示 |
| `scripts/models/dump_model_manifest.py` | 生成模型 manifest | canonical relative path、排序、拒绝 symlink 越界、自引用和重复 |
| `scripts/models/smoke_qwen3.py` | 加载并生成最小 smoke | 只证明加载/生成；输出身份和环境，不能代替 benchmark/correctness |
| `scripts/models/verify_qwen3.py` | 检查 Qwen3 配置和架构 | S04.5 扩为 architecture facts/层数/hidden/GQA/RoPE/MLP/模块类型检查 |
| `scripts/models/verify_qwen3_hashes.py` | 验证模型文件 SHA256，使用稳定退出码 | 防路径穿越/空清单/重复；输出机器可读 summary |

### 5.22 `reports/`

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `reports/jetson/.gitkeep` | 本地 Jetson 证据目录占位 | raw 可不进 Git，但每次正式实验必须生成 evidence manifest、hash 和永久 URI |

### 5.23 `tests/`：当前全部测试文件

| 文件 | 当前用途 | 最终演进/注意事项 |
|---|---|---|
| `tests/conftest.py` | 测试路径/fixture 基础 | 避免只靠 `sys.path` 掩盖打包问题；增加 capability/seed/temp artifact fixtures |
| `tests/correctness/.gitkeep` | 多后端正确性目录占位 | S04.5 建立 operator/block/model golden matrix |
| `tests/integration/.gitkeep` | 集成测试占位 | torch.library/module/FX/compile/runtime/service 合同测试 |
| `tests/property/test_percentile_property.py` | percentile 数学不变量属性测试 | 扩展 summary、energy、throughput 和 schema roundtrip 性质 |
| `tests/test_vectors/.gitkeep` | 版本化测试向量占位 | 保存小型合法输入、edge shape、golden hash，不提交权重 |
| `tests/unit/.gitkeep` | unit 目录历史占位 | 无运行语义 |
| `tests/unit/core/test_config.py` | 配置 precedence/hash | 加嵌套/list/secret/unknown/env parsing |
| `tests/unit/core/test_contracts.py` | C1–C7 字段和严格校验 | 随 schema 扩展；加 roundtrip/hash/forward-version |
| `tests/unit/core/test_correctness.py` | golden/determinism/首错位 | 加 NaN/Inf、logits/top-k/KL、phase-localization |
| `tests/unit/core/test_dependency.py` | core 依赖方向静态检查 | 当前存在 `assert ... or True` 弱化风险；必须删除绕过并覆盖 package `__init__` 的间接导入 |
| `tests/unit/core/test_dummy_backend.py` | DummyBackend 合同 | 加生命周期、故障、capability、cancel、actual selection |
| `tests/unit/core/test_errors.py` | 错误类别和退出码 | 加 cause/retryability/serialization |
| `tests/unit/core/test_ids.py` | run/trace/span ID | 加并发唯一性、格式、确定性 run key |
| `tests/unit/core/test_jetson.py` | Jetson 协议防御逻辑 | mock 权限/命令失败/温度超限/恢复状态 |
| `tests/unit/core/test_logging.py` | JSONL/trace 日志 | 加字段 schema、脱敏、并发上下文 |
| `tests/unit/core/test_memory.py` | 权重/KV 字节核算 | 加 GQA、paged KV、quant KV、allocator overhead/fragmentation |
| `tests/unit/core/test_migration.py` | legacy schema 迁移 | 加幂等、loss report、未知/未来版本、批量失败 |
| `tests/unit/core/test_operator_spec_example.py` | OperatorSpec 示例合法性 | 覆盖所有 operator config；验证 shape/dtype/tolerance/stream |
| `tests/unit/core/test_profiling.py` | profiler 字段兼容和表抽取 | 加 module/shape/call/phase/NVTX 关联 |
| `tests/unit/core/test_pytorch_backend.py` | PyTorchBackend C4 合规 | 确保每个 WorkloadSpec 字段实际影响执行；artifact identity 完整 |
| `tests/unit/core/test_registry.py` | registry 注册冲突/卸载 | 加 capability/version/priority/lazy factory/thread safety |
| `tests/unit/core/test_roofline.py` | Roofline/Amdahl 数学 | 加测量峰值、fusion bytes、边界/单位错误 |
| `tests/unit/core/test_schema_versioning.py` | 版本迁移链 | 加缺口、循环、未来版本、multi-step provenance |
| `tests/unit/core/test_workload_config.py` | YAML workload 加载 | 加未知字段、缺失、长度分布、hash 和展开结果 |
| `tests/unit/ops/test_capability.py` | capability 无异常与缓存 | 缓存 key 纳入版本/device；区分 unavailable 与 probe failed |
| `tests/unit/ops/test_cuda_bridge.py` | ctypes bridge 与跨后端一致性 | 加非法 device/dtype/layout/shape、current stream、库符号/ABI mismatch |
| `tests/unit/ops/test_dispatcher.py` | dispatcher 路由/fallback | 加 CUTLASS 可选、真实 GEMM shape/dtype、reason code、强制 backend、Triton 缺失 |
| `tests/unit/ops/test_triton_gemm.py` | Triton GEMM correctness/autotune | P0 扩 K/N/M 非整除、M=1、odd sizes、FP16/BF16/FP32、non-contiguous 或显式拒绝 |
| `tests/unit/ops/test_triton_rmsnorm.py` | Triton RMSNorm correctness | 扩多行/真实 hidden/odd/极值/stream/autocast |
| `tests/unit/test_cli.py` | CLI 非法参数 | 加 config/路径/退出码/机器可读错误 |
| `tests/unit/test_loader.py` | 模型路径和关键文件负向测试 | 加 identity/fallback/OOM cleanup/mock revision |
| `tests/unit/test_manifest.py` | manifest 解析与完整性 | 必须新增空清单、重复、绝对路径、`../`、symlink 越界、自引用 |
| `tests/unit/test_metrics.py` | percentile/latency/error summary | 加空/单值/NaN/Inf/单位/置信区间 |
| `tests/unit/test_tegrastats_parser.py` | tegrastats 解析和能量积分 | 加格式漂移、缺采样、乱序时间、单位变化 |
| `tests/unit/test_workload.py` | 固定 token 长度生成 | 加 seed、边界、tokenizer 特殊 token、共享前缀 |

### 5.24 当前文件清单的架构结论

当前 210 个文件形成了一个相当完整的**前四阶段实验平台骨架**，但代码密度集中在 `core`、`benchmark`、Qwen3 baseline 和少数 CUDA/Triton 算子。`quant`、`serving`、`ascend`、C++ Runtime、集成测试等目录仍主要是占位。项目下一步不宜直接扩更多名词和空模块，应优先让已有模型、算子、dispatcher、benchmark 真正闭环，这正是 S04.5 的目的。

---

## 6. 当前 S04 严格静态验收结论与进入下一阶段前的修复门

### 6.1 总结

S04 已完成的主体工作足以证明你接触并实现了 CUDA、Triton、CUTLASS、TileLang capability、dispatcher、bridge 和多后端 benchmark，但还不足以严谨声称“多后端算子系统已完成并可接入模型”。合理结论是：

> **S04 主体实现完成；严格验收暂缓。先完成安全性、调度真实性、真实 shape 和证据闭环，再进入 S04.5。**

### 6.2 P0：必须先修复，否则不应接模型

| 问题 | 风险 | 期望修复 | 验收证据 |
|---|---|---|---|
| Triton GEMM K/N tail mask 不完整，存在 modulo/repeated store 语义 | 非整除 shape 可能越界、重复写或错误 | 对 M/N/K 每个边界完整 mask；禁止用 modulo 掩盖无效列 | odd/non-multiple shape matrix；sanitizer；与 torch/cuBLAS 数值对照 |
| C ABI 固定 `stream=0` | 与 PyTorch current stream 不一致，可能竞态或隐式同步 | API 显式接受 stream；bridge 传 `torch.cuda.current_stream().cuda_stream` | 多 stream 依赖测试、无额外全局同步、timeline 证据 |
| bridge 缺 tensor 校验 | CPU tensor、错误 dtype、非 contiguous、错误 shape 可崩溃/错算 | Python + C++ 双层校验，稳定错误码/异常 | 完整负向测试和 ABI mismatch 测试 |
| fallback 路径导入 Triton | Triton 缺失时理论 fallback 仍可能 import 失败 | lazy import；reference/cuBLAS 不依赖 Triton 包可用 | 无 Triton 环境的 import/dispatch 单测（mock 即可） |

### 6.3 P1：S04 严格完成条件

| 问题 | 期望结果 |
|---|---|
| dispatcher 从未选 CUTLASS，GEMM 选择忽略 shape/dtype | CUTLASS 成为实际候选；key 至少含 backend capability、arch、M/N/K、dtype、layout/alignment；输出 reason code |
| `rmsnorm_config_hash` 没真正约束 autotune | cache key 与 kernel/version/device/shape/dtype/config 集绑定；持久化且能检测失效 |
| CUTLASS 仅默认 Gemm | 固定 CUTLASS 版本；检查 `can_implement` 和所有 status；至少探索适合 decode M=1 和 prefill 的两个配置族 |
| S04 benchmark shape 与 Qwen3 manifest 不符 | 从 shape census 自动生成 workload；Qwen3 hidden=2048、intermediate=6144 等不得手写错 |
| raw reports/IR 被忽略且无永久索引 | 每次正式实验生成 evidence manifest/hash，并发布可下载 artifact 或 release asset |
| optional dependency 边界与弱断言 | CPU-minimal import 真正受门禁；删除 `assert ... or True` 类型绕过 |
| `BenchmarkEngine` 与文档 decode 公式漂移 | 唯一公式实现 + 单元测试 + schema 字段定义；明确是否含 first token |
| PyTorchBackend 忽略 WorkloadSpec 字段 | 所有字段生效或明确 capability 拒绝，结果记录 actual values |
| manifest 空/穿越/重复路径 | 解析时拒绝并有负向测试 |
| fused FP16 舍入语义 | 定义 exact semantics；对 block/model 长链误差和生成质量验证，必要时保留 FP32 residual 或回退 |

### 6.4 S04 严格验收包

应新增或更新：

- `docs/reports/S04_strict_acceptance_report.md`
- `docs/reports/S04_tail_stream_safety_report.md`
- `benchmarks/workloads/operators/qwen3_s04_shapes.yaml`
- `benchmarks/normalized/s04_backend_matrix.json`
- `tests/correctness/test_s04_backend_matrix.py`
- `tests/integration/test_torch_current_stream.py`
- `reports/<run_id>/manifest.json` 及 raw samples/profiler artifact（通过永久 URI 索引）

严格验收必须满足：所有 P0 关闭；真实 Qwen shape 和边界 shape correctness 通过；dispatcher 的每个宣称后端至少有一条实际被选路径；环境/commit/config/model/operator hash 完整；报告不把 microbenchmark 等同于模型收益。

---

## 7. 全阶段路线图总览

### 7.1 阶段之间的因果关系

| 阶段 | 根本目的 | 对后续提供什么 | 最终能力层级 |
|---|---|---|---|
| S00 | 恢复可信事实 | 可工作的基线、身份和安全边界 | 工程审计、复现意识 |
| S01 | 建立稳定契约 | 模块间共同语言和质量门禁 | 平台架构、工程质量 |
| S02 | 找到真实瓶颈 | baseline、profile、shape、优化优先级 | 模型理解、性能分析 |
| S03 | 掌握 CUDA 优化闭环 | 多版本 kernel、正确性和 profiler 解释 | CUDA/SIMT/算子开发 |
| S04 | 比较多种 Kernel DSL | CUDA/Triton/CUTLASS 适用边界和 dispatch | 多后端、DSL、autotune |
| **S04.5** | **把算子真正放回模型** | **torch op、module/FX 替换、micro→model 归因** | **框架集成、端到端优化** |
| S05 | 降低精度与内存/算力成本 | QuantArtifact、量化 kernel、质量/性能曲线 | 量化、低比特、误差分析 |
| S06 | 让图和框架系统化采用优化 | fusion、compile、lowering、graph guard | PyTorch/编译器集成 |
| S07 | 控制 decode 执行与内存 | KV、batching、cache、spec decode、runtime adapters | 推理引擎内核 |
| S08 | 在真实请求下治理性能 | API、调度、SLO、观测、故障 | Serving/性能工程 |
| S09 | 把方法迁移到 Ascend | 等价算子/backend/工具链 | NPU/异构开发 |
| S10 | 扩展到多卡多机 | parallelism、collective、overlap | 分布式推理 |
| S11 | 自动化图到 kernel 的优化 | compiler pipeline、codegen、autotune | AI Compiler |
| S12 | 科学比较不同硬件 | 统一 benchmark、成本/能效/capability | 异构评测、选型 |
| S13 | 进入生产系统 | 容器、K8s、SRE、安全、回滚 | 平台/生产化 |
| S14 | 连接训练与前沿能力 | 训推 artifact、稀疏/MoE/长上下文等 | 前沿扩展与研究转化 |
| S15 | 把成果变为可信影响力 | release、文档、demo、简历、面试证据 | 开源协作与技术表达 |

### 7.2 阶段执行通用规则

每阶段都按以下顺序执行，Coding Agent 不得跳过：

1. **Freeze**：记录输入 commit、环境、模型/配置 hash、前置阶段证据。
2. **Question**：写出要回答的工程问题和可证伪假设。
3. **Contract**：先定义输入、输出、错误、fallback、schema 和目录归属。
4. **Reference**：建立正确但不一定快的 reference。
5. **Implement**：逐个最小可比较版本实现；一次只改变一个主要变量。
6. **Correctness**：先 unit，再 operator/block/model，失败即停止性能 claim。
7. **Measure**：保留 raw samples、环境、命令、tool artifacts。
8. **Explain**：profile、roofline、Amdahl、消融、反例和适用边界。
9. **Gate**：自动化测试、文档、静态检查、性能回归和人工 review。
10. **Publish**：开发报告、验收报告、evidence ledger、演示和求职表述同步。

---

## 8. S00：现状审计与基线恢复

**根本目的**：先知道仓库真正有什么、什么能复现、什么只是历史或计划，建立不会污染后续结论的可信起点。

**当前状态**：主体完成；当前成熟度 M1–M3 混合。需要把 status/evidence 的 commit 漂移纳入持续门禁。

### 8.1 输入

- Git 历史、tracked/untracked/ignored 清单；
- 模型目录和已有 manifest；
- README、旧报告、脚本和运行产物；
- 目标硬件/软件环境。

### 8.2 工作包

1. 生成仓库 inventory：文件、大小、大文件、秘密、权重、临时构建物、绝对路径。
2. 冻结模型 identity：revision、config、tokenizer、权重列表与 hash。
3. 修复最小加载/CLI/workload/metrics/manifest 负向路径。
4. 对历史声明分级：source/test/runtime/historical/planned。
5. 建立 README→status→ledger→evidence 的链接关系。
6. 记录工作树和本机专用脚本，不把历史改写/强推工具当项目功能。

### 8.3 产出

- 仓库审计清单、model manifest、environment snapshot；
- `S00_阶段验收报告.md`、`project_status.md`、`evidence_ledger.md`；
- 最小 CPU tests、hash/CLI 验证入口；
- secrets/weights/build/reports 的 ignore 策略。

### 8.4 验收门

- 新 clone 能在无模型、无 GPU 时完成 CPU-minimal import/test；
- 缺文件、坏 hash、非法 CLI、模型路径错误均返回可诊断非零状态；
- 不存在无法解释的“已完成”声明；
- status 的 commit 与当前 release 自动一致；
- 本地 raw 未提交时有永久证据索引方案。

**体现能力**：Linux/Git 审计、可复现工程、artifact integrity、错误设计、基础测试、事实边界意识。

---

## 9. S01：核心契约与工程质量体系

**根本目的**：在实现更多优化前，先让模型、工作负载、算子、backend、量化、结果和 trace 拥有稳定共同语言，防止后期模块互相绑死。

**当前状态**：主体完成，M1–M2；需修 optional dependency、弱断言、契约表达力和证据 artifact。

### 9.1 输入

- S00 的真实 inventory 和模型/结果格式；
- 未来 CUDA、Ascend、Runtime、Serving 需要交换的数据；
- legacy golden、operator metadata 和 benchmark JSON。

### 9.2 工作包

1. 定义并版本化 C1–C7，拒绝未知/缺失/未来版本。
2. 建立 deterministic config merge/hash 和敏感字段脱敏。
3. RegistryHub 管理 backend/operator/quantizer/monitor/reporter。
4. 稳定错误分类、exit code、run/trace/span ID、JSONL logging。
5. 实现 DummyBackend 和 contract-native BenchmarkEngine。
6. 实现 schema migration 和 loss report。
7. 配置包依赖组、CPU CI、文档链接检查、模块 ownership。
8. 给核心数学添加 property tests，给依赖方向添加 AST/import tests。

### 9.3 产出

- `hqsb/core/*`、Dummy/PyTorch backend 基础；
- `pyproject.toml`、CPU CI、pytest markers；
- ADR/experiment/optimization/handoff 模板；
- migration CLI 和 contract examples；
- `S01_开发报告.md`、`S01_阶段验收报告.md`。

### 9.4 验收门

- 核心包在未安装 torch/Triton/CUDA 时可 import；
- 任意 backend 仅依赖 C4 即可注册/执行/产出 C6；
- config hash 稳定，schema migration 可追踪；
- CI 强制测试、文档、format/lint/type（不能注释掉当“可选”）；
- dependency test 不含恒真逃生条件。

**体现能力**：高内聚低耦合、接口抽象、Python 工程、schema 演进、插件系统、CI/CD、可观测基础。

---

## 10. S02：模型基线与全栈 Profiling

**根本目的**：回答“真实模型在真实 workload 上慢在哪里、为什么慢、先优化什么”，确保优化选择由证据驱动。

**当前状态**：主体完成，M1–M3；必须补强 shape census、口径统一与 raw evidence 可访问性。

### 10.1 输入

- 冻结的 Qwen3-1.7B ModelArtifact；
- Jetson FP16 workload suite；
- PyTorch reference backend；
- tegrastats、PyTorch Profiler、Nsight Systems/Compute。

### 10.2 工作包

1. 建立六类 workload：tiny、short、balanced、long-prefill、decode-heavy、long-balanced。
2. 严格拆分 prefill、first token、decode；定义 TTFT、TPOT、E2E、tokens/s。
3. 收集权重、KV cache、allocator peak、RSS/swap、功率、能量、温度。
4. PyTorch Profiler 找 operator/module/shape/call count。
5. Nsight Systems 找 CPU launch、同步、kernel timeline、空洞。
6. Nsight Compute 找 DRAM/L2、warp、occupancy、register、instruction 和 roofline。
7. 通过 Amdahl 估算候选热点最大端到端收益，形成 Hotspot Decision。
8. **新增必要产出**：Qwen3 block 解剖和真实 shape census，区分 prefill/decode。

### 10.3 必做实验

| 实验 | 自变量 | 指标 | 目的 |
|---|---|---|---|
| 长度扫描 | ISL/OSL | TTFT/TPOT/E2E/energy | 区分 prefill 与 decode scaling |
| batch 扫描 | batch 1/2/4… | throughput/latency/memory | 找容量和并行度边界 |
| 冷热启动 | cold/warm | load/compile/first request | 防止编译/缓存污染 steady-state |
| profiler phase | prefill/decode | op time/calls/shapes | 锁定热点和真实形状 |
| memory model | context/batch/dtype | predicted vs measured | 验证 KV/权重/allocator 核算 |
| power protocol | clock/power mode/temp | tok/J、energy/request | 形成 edge AI 差异化证据 |

### 10.4 产出与门禁

- `baseline_report.md`、`pytorch_profile_report.md`、`nsys_report.md`、`ncu_report.md`；
- `qwen3_architecture_map.md`、`qwen3_shape_census.json/csv`；
- raw profiler/trace/evidence manifest；
- Hotspot Decision 必须覆盖模型主要耗时，并解释为何先做 RMSNorm/fusion/GEMM；
- 文档指标公式与代码单一实现完全一致；
- 模型输出 deterministic/golden gate 通过。

**体现能力**：Transformer/Qwen 结构、prefill/decode、KV cache、性能分析、GPU 工具、Roofline/Amdahl、科学测量。

---

## 11. S03：CUDA 算子性能工程

**根本目的**：完整展示从正确 reference 到 CUDA 多版本优化、硬件计数器解释和稳健 dispatcher 的算子开发方法论。

**当前状态**：主体完成，M1–M3；未接真实模型，current stream/torch binding 在 S04/S04.5 补齐。

### 11.1 输入

- S02 Hotspot Decision 与 shape census；
- RMSNorm、residual+RMSNorm 数学定义；
- Jetson sm_87 设备属性和实测带宽；
- C3 OperatorSpec、统一 correctness metrics。

### 11.2 工作包

1. V0：可读 baseline，建立数值 reference 和资源模型。
2. V1：warp shuffle/reduction，减少同步/shared memory。
3. V2：float4/half2 向量化、对齐检查、scalar tail、shape-aware block。
4. fused residual+RMSNorm：合并 HBM 往返，明确 residual/output 语义。
5. dispatcher：dtype/shape/alignment/arch 路由和强制版本开关。
6. C++ correctness：随机、极值、真实 shape、非对齐、非法参数、NaN/Inf。
7. benchmark：warmup、CUDA event、重复、GB/s、占理论/实测带宽比例。
8. profiler：解释 load transaction、occupancy、register、warp stall、block size。
9. sanitizer：越界、race、leak；明确库 API 不隐藏分配。

### 11.3 核心实验

- hidden size：100/128/512/2048/6144/8192/odd；
- rows：1、decode batch、prefill tokens 展平；
- dtype：FP32/FP16/BF16（支持或明确拒绝）；
- block size/warps、V0/V1/V2、对齐/非对齐；
- fused vs unfused 的实际 bytes 和 latency；
- stream、in-place/alias、并发 stream；
- bandwidth roofline 与性能不升反降案例。

### 11.4 验收门

- 所有实现共享同一 OperatorSpec/reference/tolerance；
- 不支持组合可解释回退，无 silent wrong result；
- 性能结论至少三次独立 run，包含 raw samples；
- Nsight 指标能够解释 V1/V2 成败；
- API 显式 stream、错误和 workspace；
- 产出 `S03_benchmark_report.md`、两份 optimization log、开发/验收报告。

**体现能力**：C++/CUDA、SIMT、reduction、vectorization、访存、occupancy、数值稳定、microbenchmark、Nsight、API 设计。

---

## 12. S04：Triton、CUTLASS/CuTe 与 Kernel DSL

**根本目的**：学会在手写 CUDA、Triton、CUTLASS 和 vendor library 之间做基于 shape/硬件/维护成本的工程选择，而不是追求某个框架“永远最快”。

**当前状态**：主体源码完成，M1–M3；严格验收暂缓，详见第 6 节。

### 12.1 输入

- S03 CUDA RMSNorm/fused 算子与 benchmark；
- Qwen3 真实 GEMM/RMSNorm shape；
- Triton/CUTLASS/TileLang 版本和 capability；
- cuBLAS/PyTorch reference。

### 12.2 工作包

1. capability probe：区别“未安装”“架构不支持”“编译失败”“运行错误”。
2. Triton RMSNorm：reference、覆盖任意 hidden、autotune 和 IR 分析。
3. Triton GEMM：完整 M/N/K tail、安全 accumulation、M=1 与大 M 策略。
4. CUTLASS GEMM：固定版本，探索 tile/stage/alignment/epilogue，检查所有 status。
5. CUDA bridge/torch binding：device/dtype/layout/shape/stream/ABI 安全。
6. dispatcher：capability→arch→shape→dtype/layout→cost/benchmark cache→fallback。
7. autotune：cache key、持久化、版本/设备失效、首次编译成本和 steady state 分开。
8. IR/SASS 解释：register、spill、shared memory、occupancy、load/store 宽度。

### 12.3 必做对照实验

- RMSNorm：PyTorch/CUDA V0/V1/V2/Triton ref/Triton tuned；
- GEMM：PyTorch/cuBLAS/Triton/CUTLASS，覆盖 QKV、O、gate/up/down 和 LM head shape；
- decode M=1、small batch 与 prefill large-M 分开；
- 规则 shape 与 odd/tail shape；
- 首次 compile/autotune latency 与 steady state；
- dispatcher 强制选择消融，验证每条路径确实执行；
- IR/资源用量与性能关联。

### 12.4 验收门

- 第 6 节 P0/P1 全部关闭；
- 所有实际支持 shape 无越界/错算；
- dispatcher 选择结果与 reason 写入 C6；
- CUTLASS/Triton 版本、config 和 IR 可追踪；
- 报告给出“何时选 CUDA/Triton/CUTLASS/cuBLAS”的决策表；
- 仍不得声称模型级收益，直到 S04.5。

**体现能力**：Triton、CUTLASS/CuTe、kernel DSL、autotune、代码生成、capability/fallback、性能可移植性。

---

## 13. S04.5：模型架构理解与算子集成闭环（下一阶段，强制插入）

**根本目的**：把“我写了一个快 kernel”升级为“我理解真实 Qwen3 的数学和执行路径，并能安全替换模型中的实现，证明或否定端到端收益”。这是整个项目从算子作品到 AI Infra 工程项目的分水岭。

**为什么现在做比直接进入量化更好**：量化会同时改变权重格式、数学误差、算子和 Runtime。如果在 FP16 reference 下尚未掌握模型结构、替换边界、正确性和 micro→model 归因，进入 S05 会把问题混在一起，最终既讲不清量化，也讲不清算子。

### 13.1 精确界定“模型架构替换”

本阶段不更改 Qwen3 的模型定义和权重含义。所谓替换是：

- 保持相同权重、tensor shape、RMSNorm epsilon、RoPE/GQA/KV 语义和 generation 行为；
- 用自定义 CUDA/Triton/CUTLASS 实现替换等价 operator/module/fused subgraph；
- 通过 capability 和开关随时回到 PyTorch reference；
- 任何因 FP16 舍入、alias、layout、stream 或融合顺序产生的语义变化都必须显式定义、测量和限制。

### 13.2 输入

- 通过严格验收的 S04 算子和 dispatcher；
- Qwen3-1.7B 固定 ModelArtifact；
- S02 workload 与 profiler；
- Transformers/PyTorch 实际模型模块树和 forward 路径；
- operator/block/model 多级 reference。

### 13.3 WP1：Qwen3 架构白盒化

必须输出逐层架构图和调用台账，至少覆盖：

1. Embedding、decoder layers、final RMSNorm、LM head；
2. 每层 input RMSNorm、Q/K/V/O projection；
3. GQA：query heads、KV heads、head_dim、repeat/共享方式；
4. RoPE：position IDs、cos/sin、cache 和 dtype；
5. attention mask、causal 语义、prefill 与 single-token decode 的差异；
6. KV cache 的 layer/head/sequence/head_dim 布局、增长和读写；
7. post-attention RMSNorm、gate/up/down projection、SwiGLU；
8. 两条 residual 路径和可融合边界；
9. 权重 tying、logits 和 generation loop；
10. eager/SDPA/其他 attention implementation 的实际选择。

对六个 workload 分别收集：模块名、调用次数、输入输出 shape、dtype、stride/layout、contiguous、device、phase、耗时占比。所有 shape 由运行时 hook/profiler 生成，不能手抄猜测。

**产出**：

- `docs/reports/S04.5_qwen3_architecture_report.md`
- `benchmarks/normalized/qwen3_shape_census.json`
- `benchmarks/normalized/qwen3_operator_call_matrix.csv`
- 一张 prefill/decode 数据流图和一张 block 结构图。

### 13.4 WP2：建立正式自定义算子边界

每个接入算子至少有：

- `torch.library` schema；
- PyTorch/CPU composite reference；
- CUDA 与 Triton 实现注册；CUTLASS 适用时注册；
- Meta/FakeTensor implementation，支持 FX/export/torch.compile shape propagation；
- capability check 和 reasoned fallback；
- current CUDA stream；
- autocast/inference_mode 语义；
- device/dtype/shape/layout/stride/alignment/epsilon 校验；
- 明确是否支持 autograd；本项目推理路径可声明无 backward，但必须明确错误；
- stable namespace/version，避免与第三方 op 冲突。

第一批只做两个代表性算子：

1. `hqsb::rms_norm(input, weight, eps) -> output`
2. `hqsb::fused_add_rms_norm(input, residual, weight, eps) -> (normalized, updated_residual)`，具体输出按模型语义确定。

### 13.5 WP3：三层替换路径

| 层级 | 实现 | 目的 | 必须验证 |
|---|---|---|---|
| L1 直接 op | 独立调用 `torch.ops.hqsb.*` | 验证 binding、dispatch、stream、correctness | operator shapes/dtypes/edges |
| L2 module swap | 把模型 RMSNorm 模块替换为兼容模块 | 验证权重共享、state_dict、device/dtype、调用次数 | 单 block 与全模型 |
| L3 graph pattern | FX/Inductor 捕获 residual-add→RMSNorm pattern | 验证真实融合、guard、graph break、fallback | eager/compile、不同 shape |

替换系统要提供显式开关：`reference`、`cuda`、`triton`、`auto`；结果必须记录请求实现与实际实现。禁止 monkey patch 后无法恢复原模型。

### 13.6 WP4：四级正确性

| 级别 | 比较对象 | 核心指标 | 通过标准 |
|---|---|---|---|
| Operator | 单个 RMSNorm/fused op | max/mean/RMSE/cosine/L2rel、NaN/Inf | 每 dtype/shape 使用预注册 tolerance，所有 edge case 通过 |
| Block | 一个 decoder layer | hidden state、KV 输出、logits proxy | 多 token/position/cache 状态下误差受控 |
| Model | 完整 prefill/decode | logits、top-k、KL、token sequence | golden workloads 通过；差异可定位至 layer/op |
| Long generation | 多步 decode | token divergence、累积误差、生成质量 | 明确 exact-match 或 quality threshold；不能只看首 token |

额外检查：多 stream、non-default stream、autocast、`torch.compile`、CUDA Graph capture（若宣称兼容）、不同 batch/ISL、非 contiguous（支持或明确拒绝）、OOM/fallback。

### 13.7 WP5：三层性能与 Amdahl 闭环

1. **Micro**：同一真实 shape 下 PyTorch/CUDA/Triton/auto 的 latency、GB/s、资源利用。
2. **Block**：单层或 N 层重复，测 kernel launch、intermediate allocation、融合节省。
3. **Model**：六 workload 的 TTFT、TPOT、E2E、prefill/decode tokens/s、peak memory、energy。
4. **归因**：用原始热点占比和 micro speedup 预测上限，再与真实 model speedup 比较。
5. **解释偏差**：调用次数、launch overhead、Python/框架开销、图 break、同步、内存布局、算子占比变化。

即使最终模型收益很小，也是一份高价值结果：只要能够证明 kernel 快、模型收益为何被 Amdahl/launch/非热点限制，并指出下一步真正该优化什么。

### 13.8 S04.5 六个正式实验工作负载

| Case | ISL | OSL | 主要观察 |
|---|---:|---:|---|
| tiny | 32 | 16/32 | launch overhead、短生成、冷启动敏感 |
| short | 128 | 32 | 常规交互请求 |
| balanced | 512 | 128 | prefill/decode 综合 |
| long-prefill | 2048 | 32 | 大 M、prefill 融合、内存带宽 |
| decode-heavy | 128 | 256 | 单 token decode 累积收益与长链误差 |
| long-balanced | 2048 | 128 | 长上下文 + 长输出、KV/能量综合 |

每个 case 对照 `reference`、`cuda`、`triton`、`auto`；若某实现不支持，必须记录 capability reason，不能丢弃样本。

### 13.9 S04.5 交付物

- 源码：`hqsb/integration/torch_ops.py`、`module_swap.py`、`fx_patterns.py`、`ops/bindings/*` 等合理边界；
- 配置：operator integration、backend preference、六 workload；
- 测试：operator/block/model/stream/compile/fallback；
- 数据：shape census、call matrix、raw micro/block/model、profiler/trace；
- 报告：
  - `S04.5_qwen3_architecture_report.md`
  - `S04.5_operator_integration_report.md`
  - `S04.5_correctness_report.md`
  - `S04.5_micro_to_model_report.md`
  - `S04.5_阶段验收报告.md`
- 演示：一个命令切换 reference/custom/auto；一个 notebook 或 CLI 展示选择原因、正确性、模型收益。

### 13.10 S04.5 严格验收标准

- Qwen3 架构和真实 shape/call census 完整；不存在 8192/6144 等手写混淆；
- 至少 RMSNorm 和一个真实 fused pattern 可开关替换并可无损恢复；
- CPU/reference、CUDA、Triton、Meta/FakeTensor 边界清晰；
- operator/block/model/长 decode correctness 全通过；
- current stream、autocast、fallback 和 unsupported shape 行为通过测试；
- 六 workload 有 micro/block/model raw evidence；
- 报告能量化 Amdahl 预测与实际模型收益的差异；
- 达到至少 M4；若端到端有稳定收益则达到 M5；
- 完成后才进入 S05。

**体现能力**：Qwen/Transformer 白盒理解、PyTorch internals、torch.library、FX/compile、算子接入、stream/内存语义、多级 correctness、端到端性能归因。这是最应在简历和面试中重点讲的一阶段。

---

## 14. S05：量化与低精度推理

**根本目的**：把量化从“调用一个库压缩模型”提升为可解释的制品、误差、kernel、内存和模型/服务收益体系。

**前置条件**：S04.5 M4 通过；否则无法分辨量化误差来自哪里，也无法可靠接入低比特算子。

### 14.1 工作包

1. 定义 C5 QuantArtifact：scheme、granularity、group size、scale/zero、calibration、packing、kernel compatibility。
2. 建立 FP16 reference 与 W8A16、W4A16、W8A8（硬件允许时）、KV8/KV4 路线。
3. 校准数据集：来源、license、token 分布、hash、样本覆盖、敏感层分析。
4. per-tensor/per-channel/per-group、symmetric/asymmetric、static/dynamic 消融。
5. weight packing/dequant、quantized GEMM 或 W4A16 kernel；融合 dequant+GEMM/epilogue。
6. outlier/敏感层策略、mixed precision、RMSNorm/attention/LM head 保留精度策略。
7. KV cache 量化：layout、scale 粒度、读写开销、长上下文误差。
8. 模型质量：perplexity/任务集/生成一致性；性能：模型大小、peak memory、TTFT/TPOT/tokens/s/energy。
9. 与主流方法/实现（AWQ/GPTQ/SmoothQuant/torchao等概念或可用基线）做公平比较，明确自研部分。

### 14.2 核心实验矩阵

- bit-width × group size × calibration size；
- layer sensitivity heatmap；
- dequant overhead vs GEMM saving；
- prefill large-M vs decode M=1；
- context length × KV bit-width；
- quality–latency–memory–energy Pareto frontier；
- generic library vs HQSB kernel；
- weight-only 与 activation/KV quant 的收益来源分解。

### 14.3 产出与验收

- `hqsb/quant/` 完整策略、artifact、calibration、packing/eval；
- `configs/quantization/*.yaml`；量化 test vectors；低比特算子/dispatcher；
- `S05_quantization_methodology.md`、`S05_quality_report.md`、`S05_performance_report.md`、验收报告；
- 任一量化 claim 同时包含模型质量、内存、性能和适用硬件；
- QuantArtifact 可重载、hash 稳定、与模型/算子/runtime capability 可校验；
- 达到 M4，至少一个方案稳定端到端有效则 M5。

**体现能力**：PTQ/低比特理论、校准、误差传播、packing、量化 GEMM、KV cache、质量性能权衡、实验设计。

---

## 15. S06：框架集成与图优化

**根本目的**：让优化不依赖手工改一处模型源码，而是通过规范 operator、graph capture、pattern fusion 和编译后端系统化接入。

### 15.1 与 S04.5 的边界

- S04.5 负责“证明两个算子能安全接入一个真实模型”；
- S06 负责“把这种接入扩展为可维护的图优化/编译机制”，覆盖更多 pattern、动态 shape、guard 和 backend。

### 15.2 工作包

1. 完善 torch.library 自定义 op：Meta/FakeTensor、decomposition、autograd policy、functionalization/alias。
2. FX/export 图捕获：识别 Qwen block 的 RMSNorm、residual、SwiGLU、QKV/rope 等 pattern。
3. pattern rewrite/fusion pass，记录命中/拒绝原因和 graph diff。
4. torch.compile/Inductor 集成：graph break、guard、dynamic shape、cache、recompile 计数。
5. CUDA Graph：capture 条件、静态 buffer、memory pool、fallback。
6. 将 CUDA/Triton/CUTLASS/量化 kernel 作为 lowering target。
7. 处理 state_dict、device movement、serialization、model export 和版本兼容。
8. 建立 eager/reference/compiled/fused 的 correctness/performance matrix。

### 15.3 产出与验收

- `hqsb/compiler` 或 `hqsb/integration` 中的 pass/registry/lowering；
- graph fixtures、before/after graph、pattern coverage 报告；
- `S06_graph_integration_report.md`、`S06_compile_report.md`、验收报告；
- 六 workload 无不可解释 graph break；dynamic shape guard/recompile 可观测；
- fusion 命中和 fallback 都有测试；
- 编译耗时与 steady-state 收益分开；达到 M4/M5。

**体现能力**：PyTorch internals、FX/Export/Inductor、图融合、动态 shape、编译缓存、CUDA Graph、框架级工程。

---

## 16. S07：推理 Runtime 内核

**根本目的**：掌握 LLM decode 的真正执行系统：内存、KV、批处理、调度、缓存和模型执行，而不仅是调用 `generate()`。

### 16.1 工作包

1. Backend adapter matrix：PyTorch、vLLM、SGLang、TensorRT-LLM、llama.cpp（按环境选择实际可用子集）。
2. 模型 artifact translation：HF/quantized/engine 格式、版本和 capability。
3. KV cache manager：page/block、free list、allocation、eviction、fragmentation、OOM。
4. continuous batching：prefill/decode iteration、动态加入/退出、fairness。
5. prefix cache：hash、共享、命中、污染、eviction。
6. speculative decoding：draft/target、acceptance、rollback、收益模型。
7. CUDA Graph 和 allocator/memory pool；减少 launch/分配开销。
8. runtime scheduler：chunked prefill、decode priority、budget、preemption。
9. cancellation、timeout、错误恢复和资源释放。
10. runtime trace：每步 batch、KV blocks、selected kernels、queue/compute/memory。

### 16.2 实验

- batch/concurrency/sequence distribution 的吞吐—延迟曲线；
- KV page size、fragmentation、capacity、eviction；
- continuous vs static batching；
- prefix cache hit rate/workload locality；
- speculative draft length/acceptance/target overhead；
- CUDA Graph on/off；
- PyTorch/vLLM/SGLang/TRT-LLM 同 workload 公平对照；
- quantization × runtime interaction。

### 16.3 产出与验收

- Runtime C++/Python 核心或深入 adapter，不接受只有 shell wrapper；
- KV/scheduler/cache correctness 和压力测试；
- `S07_runtime_architecture.md`、`S07_kv_cache_report.md`、`S07_scheduler_report.md`、backend comparison、验收报告；
- 请求取消/OOM/异常不会泄漏 blocks 或留下坏状态；
- 所有 runtime 指标与 C2/C6/C7 对齐；达到 M5。

**体现能力**：vLLM/SGLang 原理、PagedAttention/KV、continuous batching、speculative decoding、allocator、CUDA Graph、推理引擎设计。

---

## 17. S08：ServeFabric 与性能治理

**根本目的**：把模型 Runtime 变成在真实并发、网络、排队和故障下可治理的推理服务，并用 SLO 而非单请求峰值评价系统。

### 17.1 工作包

1. OpenAI-compatible API：chat/completions、streaming、usage、错误协议。
2. Gateway：认证、配额、请求校验、模型路由、trace propagation。
3. Admission/backpressure：队列上限、超载拒绝、timeout、cancel。
4. Scheduler/router：长度/优先级/tenant/capability/SLO-aware。
5. 多 backend worker 生命周期、health、warmup、drain、rolling restart。
6. streaming token path、客户端断开和资源回收。
7. observability：queue、TTFT、TPOT、E2E、P50/P95/P99、goodput、cache、GPU、error。
8. load generator：Poisson/burst/trace replay、长度分布、共享前缀、多租户。
9. 故障注入：worker crash、OOM、慢请求、网络中断、模型加载失败、设备降频。
10. capacity/SLO controller：最大并发、队列策略、降级/回退。

### 17.2 核心实验

- QPS 扫描到饱和，画 latency-throughput-goodput 曲线；
- arrival 分布：恒定、Poisson、burst；
- streaming/non-streaming；
- short/long 混部与 head-of-line blocking；
- scheduler 策略、priority/fairness、多租户隔离；
- cache on/off 与命中率；
- 故障恢复时间、失败率、资源泄漏；
- 单 backend vs capability/SLO 路由。

### 17.3 产出与验收

- 服务 API、loadgen、dashboard/metrics、deployment config；
- `S08_serving_architecture.md`、`S08_slo_capacity_report.md`、`S08_fault_report.md`、验收报告；
- 在目标 SLO 下报告 goodput，不以吞吐最大值掩盖 P99 崩溃；
- cancel/timeout/backpressure/failure 有集成测试；
- trace 能从 HTTP 请求定位到 Runtime batch 和 kernel；达到 M6。

**体现能力**：在线 Serving、调度、尾延迟、排队论直觉、OpenAI 协议、可观测性、可靠性、容量规划。

---

## 18. S09：Ascend C/CANN 异构后端

**根本目的**：证明你能把同一个优化问题从 CUDA 迁移到 NPU 工具链，并理解“统一语义、不同硬件映射”，这对异构推理岗位极具辨识度。

### 18.1 工作包

1. 环境与 capability：芯片型号、CANN/驱动/固件、算子编译和 profiler。
2. C4 Ascend backend 和 C3 operator mapping；保持公共 contract 不变。
3. 实现 RMSNorm、fused residual+RMSNorm，以及一个量化/GEMM 代表路径。
4. 数据布局、tiling、UB/L1/GM、vector/cube unit、流水和 double buffer 分析。
5. ACL/ATB/torch_npu 或适合框架接入；current stream/queue、workspace 和 error。
6. msprof 等工具采集，建立 Ascend roofline/瓶颈解释。
7. 模型级集成、fallback、同一 workload 和 correctness。
8. 形成 CUDA warp/block/shared/DRAM 到 Ascend core/UB/GM/tiling 的概念映射表。

### 18.2 实验与验收

- 同 OperatorSpec 在 CUDA/Triton/Ascend 上的 correctness；
- tiling、block_dim、UB 占用、double buffer 消融；
- model prefill/decode、memory、energy（可采则采）；
- capability 缺失/版本不兼容/shape 不支持的错误与 fallback；
- 输出 `S09_ascend_backend_report.md`、`S09_cuda_ascend_mapping.md`、operator logs、验收报告；
- 达到 M4/M5，跨硬件同协议后为 M7。

**体现能力**：Ascend C/CANN、NPU 架构、tiling、内存层次、异构抽象、工具链迁移、跨硬件调试。

---

## 19. S10：分布式推理与通信

**根本目的**：理解模型无法单卡容纳或需要多卡吞吐时，计算、内存、通信和调度如何共同决定性能。

### 19.1 工作包

1. 建模并实现/适配 tensor parallel、pipeline parallel、expert parallel；明确适用模型。
2. topology discovery：PCIe/NVLink/NVSwitch/网络/RDMA 能力和带宽延迟基线。
3. collective microbench：all-reduce/all-gather/reduce-scatter/all-to-all，消息尺寸扫描。
4. attention/MLP sharding、权重/KV 分布和通信量模型。
5. 通信计算 overlap、stream/event、bucket/chunk、async collective。
6. rank lifecycle、timeout、错误传播、部分失败和 cleanup。
7. 分布式 trace：rank、collective、compute、wait、straggler 对齐。
8. 多节点部署时记录网络拓扑、NCCL 环境和容器/驱动兼容。

### 19.2 实验与验收

- 卡数 scaling：1/2/4/8，strong/weak scaling；
- message size 与算法/协议；
- TP degree × batch/sequence；
- overlap on/off、不同 chunk/bucket；
- all-to-all 和 MoE load imbalance（若进入 MoE）；
- straggler/链路降速/单 rank 失败；
- 输出 `S10_parallelism_design.md`、`S10_collective_report.md`、`S10_scaling_report.md`、验收报告；
- 报告 compute/communication/idle 三分解和 scaling efficiency；达到 M5/M6。

**体现能力**：NCCL、TP/PP/EP、通信模型、拓扑、RDMA 概念、overlap、多进程调试、分布式可靠性。

---

## 20. S11：AI 编译器与自动优化

**根本目的**：把人工发现的 pattern 和 kernel 经验转化为可重复的 graph→IR→lowering→codegen→autotune 流程。

### 20.1 工作包

1. 选定主线：PyTorch FX/Export + Inductor/Triton 是最贴近现有项目的首选；MLIR/TVM 作为扩展。
2. 建立 front-end graph capture、shape/dtype propagation、guards 和 graph break diagnostics。
3. 定义 HQSB IR 或明确映射到已有 IR；保留 source op/module/trace metadata。
4. pattern fusion：RMSNorm/residual、SwiGLU、QKV/RoPE、dequant+GEMM 等。
5. lowering registry 将 pattern 映射 CUDA/Triton/CUTLASS/Ascend。
6. schedule/autotune search space、cost model、cache、invalid config filtering。
7. codegen artifact 版本、编译缓存、recompile、binary compatibility。
8. differential testing：eager vs compiled、random graph/shape、fallback。
9. compile time、cache hit、steady-state、模型收益分别测量。

### 20.2 产出与验收

- 可视化 before/after graph/IR；pass 与 lowering registry；autotune database；
- 至少两个真实 Qwen pattern 自动识别并选择 HQSB kernel；
- dynamic shape/guard/fallback 可诊断；
- `S11_compiler_architecture.md`、`S11_fusion_report.md`、`S11_autotune_report.md`、验收报告；
- correctness、compile overhead、cache、model speedup 全闭环；达到 M4/M5。

**体现能力**：FX/Inductor、MLIR/TVM 概念、IR、pattern rewrite、lowering、codegen、cost model、autotune、编译缓存。

---

## 21. S12：跨硬件评估与统一 Benchmark

**根本目的**：建立一种不偏袒单一芯片、能回答“这个 workload 应部署在哪里”的科学评测体系。

### 21.1 工作包

1. 统一 capability taxonomy：dtype、memory、kernel、runtime、service、distributed。
2. 同一 ModelArtifact 语义、WorkloadSpec、correctness、C6/C7，后端只处理差异。
3. 统一环境采集：驱动、runtime、firmware、power mode、clock、container、价格/功耗假设。
4. raw→normalized pipeline，保留原始单位，避免不透明评分。
5. 指标维度：TTFT、TPOT、goodput、capacity、memory、quality、power/energy、cost、availability。
6. 设备公平性：精度、batch、并发、编译/warmup、功耗模式、模型版本一致或明确不同。
7. Pareto frontier 与 deployment recommendation，不给单一“总分”掩盖权衡。
8. 证据包可由第三方下载、校验和复现。

### 21.2 产出与验收

- hardware/backend capability matrix；统一 suite 和 normalized schema；
- Jetson + 至少一种数据中心 GPU/Ascend 的同协议结果；
- `S12_cross_hardware_report.md`、`S12_cost_energy_report.md`、`S12_deployment_guide.md`；
- 结论明确 workload、SLO、精度和成本假设；达到 M7。

**体现能力**：异构评测、硬件选型、性能/能效/成本建模、benchmark 治理、技术决策。

---

## 22. S13：生产化、云原生与可靠性

**根本目的**：证明优化系统不仅能在开发机跑一次，还能安全部署、升级、观测、扩缩、回滚和应对故障。

### 22.1 工作包

1. reproducible container：multi-stage、最小镜像、GPU/Ascend runtime compatibility、non-root。
2. artifact supply chain：模型/engine/kernel 签名或 hash、SBOM、license、漏洞扫描。
3. Kubernetes/设备插件/资源请求、node selector、topology、hugepage/shared memory（按需）。
4. readiness/liveness/startup、graceful drain、rolling/canary、rollback。
5. autoscaling：queue、goodput、TTFT/TPOT、GPU utilization 的多指标策略。
6. SRE：SLO/SLI/error budget、告警、runbook、capacity、postmortem。
7. fault injection：process/device/network/storage/model artifact/corruption/thermal。
8. 多租户配额、隔离、认证、日志脱敏、secret 管理。
9. GPU nightly/performance regression、release gate、artifact retention。

### 22.2 产出与验收

- container/compose/K8s manifests、dashboard/alerts/runbooks、SBOM；
- canary/rollback、故障演练、容量与成本报告；
- `S13_production_architecture.md`、`S13_reliability_report.md`、`S13_security_supply_chain.md`、验收报告；
- 在故障中维持或明确违反 SLO，资源可恢复、证据可追踪；达到 M6。

**体现能力**：Docker/Kubernetes、GPU 调度、SRE、供应链、安全、发布治理、容量与成本。

---

## 23. S14：训推协同与前沿扩展

**根本目的**：在完整主链条稳定后，选择能强化岗位匹配的前沿问题，展示把论文/新模型能力转化为工程系统的能力。

### 23.1 推荐优先方向

按与目标岗位的相关性排序：

1. **MoE 推理**：expert routing、all-to-all、load balance、expert cache、量化。
2. **长上下文**：KV 压缩/量化/淘汰、chunked prefill、attention backend、内存容量。
3. **Speculative decoding**：draft/target、acceptance、tree/spec variants 和 service scheduling。
4. **稀疏/结构化算子**：硬件支持、真实 speedup 与质量。
5. **训推 artifact 一致性**：checkpoint→quant→engine，adapter/LoRA merge、版本与回滚。
6. **Diffusion/DiT 推理**：招聘要求覆盖时，可复用 GEMM/fusion/quant/compiler/serving 方法，但不应破坏 LLM 主线。

### 23.2 执行约束与验收

- 一次只选择 1–2 个方向；每个方向必须进入统一 contract/evidence/runtime/service；
- 必须有 baseline、论文/方法假设、正确性/质量、kernel/runtime/service 性能；
- 输出独立 design/experiment/report/limitations；
- 不因“支持名词”而添加只有空目录或 wrapper 的功能；
- 达到 M4–M6，形成一项能深入讲 20 分钟的代表成果。

**体现能力**：研究阅读、方法复现、跨层工程、前沿模型、从论文到系统的转化能力。

---

## 24. S15：发布、开源与求职证据

**根本目的**：把全部技术工作转化为第三方可理解、可运行、可核验、可讨论的公开成果；这一步决定项目能否真正为秋招服务。

### 24.1 工作包

1. 形成 release：版本、changelog、兼容矩阵、安装/quickstart、known limitations。
2. 发布 evidence bundle：raw/normalized/report/tool artifacts 的 manifest、hash、永久链接。
3. 文档信息架构：项目概览、架构、教程、实验、API、贡献、FAQ、故障排查。
4. 一键 demo：reference→profile→custom op→model speedup→service benchmark。
5. 可视化：架构、模型 block、kernel 版本、roofline、Pareto、trace、SLO 曲线。
6. 开源卫生：license、third-party notice、security policy、issue/PR template、codeowners。
7. 复现挑战：在新机器/容器按文档运行关键路径并记录所有缺口。
8. 求职证据：resume bullets、STAR stories、3/10/30 分钟讲解、常见追问、失败案例。
9. 保持 claim ledger：README/简历/报告中所有数字可点到证据。

### 24.2 验收门

- 新读者 5 分钟理解价值，30 分钟能跑 CPU/demo，目标硬件有明确复现路径；
- 至少一个 end-to-end hero story 和两个深入子故事；
- 所有数字带硬件、模型、workload、baseline、correctness、重复和 artifact；
- 简历不出现无法证明的“支持/提升”；
- 发布资产、源码 commit 和报告互相绑定。

**体现能力**：开源协作、release engineering、技术写作、演示、证据管理、面试表达、工程领导力。

---

## 25. 项目应完成的完整实验目录

本节是实验 backlog。每项实验都必须用 `docs/templates/experiment.md` 创建预注册计划，至少包含：假设、输入、控制变量、自变量、因变量、样本、正确性、环境、停止条件、raw 输出、分析方法、局限、claim 等级。

### 25.1 E00：复现与身份实验

| ID | 实验 | 核心问题 | 主要产出 |
|---|---|---|---|
| E00-01 | clean clone CPU-minimal | 无 GPU/torch 时核心包是否独立 | wheel/import/test log |
| E00-02 | model manifest integrity | 缺失/篡改/空/重复/越界能否阻止加载 | negative test report |
| E00-03 | environment fingerprint | 一次运行能否唯一绑定环境 | machine-readable env JSON |
| E00-04 | result reproducibility | 相同 identity/config 是否可重放 | run comparison/report |
| E00-05 | status drift | README/status/ledger/commit 是否一致 | CI consistency report |

### 25.2 E02：模型与全栈 Profiling

| ID | 实验 | 自变量/对照 | 指标/结论 |
|---|---|---|---|
| E02-01 | Qwen architecture census | phase、workload | module/shape/dtype/layout/calls |
| E02-02 | ISL/OSL scaling | 六 workload | TTFT/TPOT/E2E/tokens/s |
| E02-03 | memory scaling | batch/context/dtype | weight/KV/allocator/peak |
| E02-04 | power/thermal | power mode/clock/temp | tok/J、J/request、throttle |
| E02-05 | profiler triangulation | PyTorch/NSys/NCU | operator→kernel→hardware stall |
| E02-06 | Amdahl prediction | candidate hotspot/speedup | predicted model upper bound |

### 25.3 E03：CUDA RMSNorm 与融合

| ID | 实验 | 自变量 | 指标/目标 |
|---|---|---|---|
| E03-01 | kernel version | V0/V1/V2 | latency/GB/s/resources |
| E03-02 | shape sweep | rows/hidden/odd/tail | correctness/selection boundary |
| E03-03 | dtype/alignment | FP32/FP16/BF16/alignment | vectorization 收益与 fallback |
| E03-04 | launch config | threads/warps/block | occupancy/register/stall |
| E03-05 | fusion | unfused/fused | bytes、launch、latency、semantic error |
| E03-06 | stream safety | default/non-default/concurrent | race/sync/timeline |
| E03-07 | sanitizer | edge/invalid | OOB/race/leak 归零 |

### 25.4 E04：Triton/CUTLASS/多后端

| ID | 实验 | 对照 | 关键结论 |
|---|---|---|---|
| E04-01 | RMSNorm backend | torch/CUDA/Triton | shape/dtype 适用区间 |
| E04-02 | GEMM backend | cuBLAS/Triton/CUTLASS | Qwen projection shape 决策表 |
| E04-03 | tail correctness | regular/odd/non-multiple | 安全边界 |
| E04-04 | decode vs prefill | M=1/small M/large M | 不同 backend 优势来源 |
| E04-05 | autotune | fixed/tuned/cache-hit/miss | 搜索收益、首次成本、稳定性 |
| E04-06 | IR/resource | configs | register/spill/shared/occupancy 与性能 |
| E04-07 | dispatcher replay | auto/forced | 选择正确性、reason、fallback |
| E04-08 | capability failure | missing/version/compile/run fail | 降级是否稳健可诊断 |

### 25.5 E04.5：模型接入

| ID | 实验 | 对照 | 必须回答 |
|---|---|---|---|
| E045-01 | operator correctness | torch/CUDA/Triton | 数值阈值和边界 |
| E045-02 | block correctness | original/swapped/fused | 误差从哪层产生 |
| E045-03 | model correctness | 六 workload、多步生成 | logits/token divergence |
| E045-04 | module swap | reference/custom/restore | state_dict、dtype/device、调用次数 |
| E045-05 | FX pattern | eager/rewrite/compile | 命中率、graph break、guard |
| E045-06 | micro→block→model | all backends | Amdahl 预测与真实收益 |
| E045-07 | framework semantics | stream/autocast/compile/graph | 兼容与 fallback |
| E045-08 | long decode | OSL 256+ | 累积误差和稳定收益 |

### 25.6 E05：量化

| ID | 实验 | 自变量 | 指标 |
|---|---|---|---|
| E05-01 | weight-only bits | W8/W4、group size | size/quality/latency |
| E05-02 | activation quant | static/dynamic、granularity | quality/kernel/overhead |
| E05-03 | calibration | dataset/size/coverage | error/robustness |
| E05-04 | layer sensitivity | per-layer bit/skip | quality recovery/cost |
| E05-05 | quant GEMM | library/HQSB、fused dequant | prefill/decode speed |
| E05-06 | KV quant | bits/context/scale | memory/quality/TPOT |
| E05-07 | mixed precision | layer/operator policy | Pareto frontier |
| E05-08 | quant artifact | save/load/version/mismatch | reproducibility/capability |

### 25.7 E06：图优化与编译

| ID | 实验 | 对照 | 指标 |
|---|---|---|---|
| E06-01 | pattern coverage | workload/model variant | hits/reject reasons |
| E06-02 | graph break | eager/compile | graph count/break/recompile |
| E06-03 | fusion | unfused/fused | memory/launch/model latency |
| E06-04 | dynamic shape | ISL/batch变化 | guards/cache/correctness |
| E06-05 | CUDA Graph | on/off | CPU overhead/latency/memory |
| E06-06 | compile cache | cold/warm/version change | compile time/cache hit/invalidation |

### 25.8 E07–E08：Runtime 与 Serving

| ID | 实验 | 自变量 | 关键指标 |
|---|---|---|---|
| E07-01 | KV page | page/block size | capacity/fragmentation/latency |
| E07-02 | continuous batching | scheduler/budget | throughput/TTFT/TPOT |
| E07-03 | prefix cache | locality/hit/eviction | saved prefill/cache memory |
| E07-04 | speculative decode | draft/steps/acceptance | accepted tokens/target call/speed |
| E07-05 | runtime adapter | PyTorch/vLLM/SGLang/TRT | 公平 model-core 对比 |
| E08-01 | load sweep | QPS/concurrency | goodput/P50/P99/error |
| E08-02 | arrival | constant/Poisson/burst | queue/tail/SLO |
| E08-03 | mixed lengths | short/long mix | HOL blocking/fairness |
| E08-04 | streaming/cancel | client行为 | time-to-first-chunk/resource cleanup |
| E08-05 | overload | admission/backpressure | rejection/SLO/recovery |
| E08-06 | worker failure | crash/OOM/slow | MTTR/failure/leak |

### 25.9 E09–E12：异构、分布式与跨硬件

| ID | 实验 | 核心变量 | 产出 |
|---|---|---|---|
| E09-01 | CUDA↔Ascend operator | same C3/shapes | correctness/perf/mapping |
| E09-02 | Ascend tiling | UB/block/double buffer | profiler/tile decision |
| E10-01 | collective sweep | op/message/topology | bandwidth/latency curve |
| E10-02 | parallel scaling | TP/PP/cards | scaling efficiency |
| E10-03 | overlap | stream/chunk | compute/comm/idle breakdown |
| E10-04 | straggler/failure | delay/rank failure | resilience report |
| E11-01 | compiler lowering | patterns/backends | compile/correctness/perf |
| E11-02 | autotune/cost model | search/budget/cache | best config/search cost |
| E12-01 | unified suite | device/backend | normalized result matrix |
| E12-02 | cost-energy | price/power/SLO | Pareto/deployment guide |

### 25.10 E13–E15：生产与发布

| ID | 实验 | 核心问题 | 产出 |
|---|---|---|---|
| E13-01 | clean container deploy | 是否环境可复现 | image/SBOM/deploy log |
| E13-02 | canary/rollback | 错误版本如何恢复 | rollout report |
| E13-03 | autoscaling | SLO 驱动还是 GPU 利用率 | scaling/cost report |
| E13-04 | chaos | process/device/network/artifact | reliability/postmortem |
| E15-01 | third-party reproduction | 新用户能否复现关键故事 | reproduction report |
| E15-02 | demo rehearsal | 3/10/30 分钟版本 | video/script/FAQ |
| E15-03 | claim audit | 每个数字能否点击证据 | final evidence ledger |

---

## 26. 最终报告与制品目录

### 26.1 每阶段固定六类制品

1. **Design**：目标、边界、接口、选项、ADR。
2. **Implementation**：模块、关键算法、数据流、错误/fallback。
3. **Correctness**：oracle、矩阵、阈值、失败定位、局限。
4. **Performance**：方法、raw、统计、profiler、解释、反例。
5. **Acceptance**：门禁逐条结果、未完成、风险、claim 等级。
6. **Evidence Manifest**：commit/config/model/operator/environment/raw/report 的 URI 和 SHA256。

### 26.2 建议新增报告清单

| 阶段 | 建议报告 |
|---|---|
| S04 strict | `S04_strict_acceptance_report.md`、`S04_tail_stream_safety_report.md`、`S04_backend_selection_report.md` |
| S04.5 | `S04.5_qwen3_architecture_report.md`、`S04.5_operator_integration_report.md`、`S04.5_correctness_report.md`、`S04.5_micro_to_model_report.md`、验收报告 |
| S05 | quant methodology、calibration、quality、kernel、KV quant、Pareto、验收报告 |
| S06 | graph architecture、pattern coverage、compile/guard/cache、fusion performance、验收报告 |
| S07 | runtime architecture、KV/cache、scheduler、spec decode、backend comparison、验收报告 |
| S08 | serving architecture、SLO/capacity、routing/scheduler、fault/observability、验收报告 |
| S09 | Ascend environment、operator、model integration、CUDA mapping、验收报告 |
| S10 | parallelism design、collective、overlap、scaling/failure、验收报告 |
| S11 | compiler architecture、IR/lowering、fusion、autotune/cache、验收报告 |
| S12 | methodology、capability matrix、cross-hardware、cost/energy、deployment guide |
| S13 | production architecture、deployment、SRE、security/SBOM、chaos、验收报告 |
| S14 | 每个选中前沿方向独立 design/correctness/performance/limitations |
| S15 | release notes、reproduction guide、demo guide、final claim ledger、interview FAQ |

### 26.3 Evidence Manifest 建议字段

```yaml
schema_version: "1.0.0"
run_id: "..."
stage: "S04.5"
claim_ids: ["..."]
git:
  commit: "..."
  dirty: false
model_artifact:
  id: "..."
  manifest_sha256: "..."
config:
  uri: "..."
  sha256: "..."
operator_artifacts:
  - name: "hqsb::rms_norm"
    spec_sha256: "..."
    binary_sha256: "..."
environment:
  uri: "..."
  sha256: "..."
commands: ["..."]
raw_artifacts:
  - uri: "..."
    sha256: "..."
normalized_artifacts: []
reports: []
correctness_status: "pass|fail|not_run"
claim_level: "SOURCE|TEST|RUNTIME|MODEL|SERVICE|PORTABLE"
limitations: []
```

---

## 27. AI Infra 能力—项目证据矩阵

| 能力 | 在项目中如何体现 | 最强证据阶段 | 面试应能回答 |
|---|---|---|---|
| C++/CUDA | kernel、ABI、stream、dispatcher、CMake、sanitizer | S03/S04/S04.5 | reduction、访存、向量化、occupancy、错误/stream |
| Python 工程 | contracts、registry、backend、benchmark、CLI、测试 | S01–S08 | 依赖边界、schema、typing、error、可测试性 |
| Transformer/Qwen | block/GQA/RoPE/KV/SwiGLU/prefill/decode | S02/S04.5 | 每个 tensor shape 和 phase 差异 |
| PyTorch internals | torch.library、Meta、module swap、FX/compile | S04.5/S06 | dispatch key、FakeTensor、graph break、guard |
| Triton | RMSNorm/GEMM、autotune、IR、tail | S04 | program model、mask、num_warps、register/spill |
| CUTLASS/CuTe | GEMM 配置、tile/stage/epilogue、status | S04 | 为什么某 shape 赢/输，M=1 与 prefill 差异 |
| 性能分析 | Profiler/NSys/NCU/Roofline/Amdahl | S02–S04.5 | 从热点到优化、从 micro 到 model 的归因 |
| 数值正确性 | 多指标、oracle、block/model/长生成 | S03–S06 | tolerance、FP16 accumulation、误差传播 |
| 量化 | PTQ、calibration、packing、quant GEMM/KV | S05 | scheme、group、outlier、质量—性能权衡 |
| 推理 Runtime | paged KV、batching、cache、spec decode | S07 | 内存块、调度迭代、碎片、acceptance |
| Serving | API、SLO、tail、routing、backpressure | S08 | goodput、HOL、过载、cancel/stream |
| Ascend | Ascend C/CANN、tiling、NPU profiler | S09 | CUDA↔NPU 概念映射、UB/GM/cube/vector |
| 分布式 | TP/PP/EP、NCCL、overlap、topology | S10 | 通信量、collective、scaling、straggler |
| AI Compiler | FX/IR/pattern/lowering/codegen/autotune | S06/S11 | graph→kernel、dynamic shape、cache |
| 异构评测 | capability、统一 workload、成本/能效 | S12 | 怎样公平比较、何时选什么硬件 |
| 生产工程 | CI/GPU nightly、container/K8s/SRE/SBOM | S13/S15 | 部署、回滚、SLO、供应链、性能回归 |
| 技术领导/表达 | ADR、ownership、报告、开源、demo | 全阶段/S15 | 失败案例、tradeoff、如何让第三方复现 |

### 27.1 与“微电子本科 + 计算机硕士”背景的独特结合

这个项目最适合把你的背景讲成一条连续能力链：

- 微电子训练让你自然强调硬件数据通路、存储层次、带宽、并行和能耗；
- 计算机硕士训练让你能把硬件能力组织成契约、编译、Runtime、Serving 和分布式系统；
- HQSB 则是两者的交汇证据：从 register/shared/DRAM 到 PyTorch graph，再到 KV scheduler 和 P99 SLO。

面试时不要把两个学历背景讲成“转专业”；应讲成“从芯片与体系结构视角理解计算代价，再用软件系统把优化落到模型和服务”。

### 27.2 目标岗位知识与技能优先级

这里的优先级不是按学习耗时排序，而是按“对推理优化/算子开发融合岗位的录用影响”和“必须在 HQSB 中留下工程证据的急切程度”排序。

#### P0：必须掌握并在项目中形成强证据

| 知识/技能 | 理论掌握标准 | 实践掌握标准 | HQSB 证据 |
|---|---|---|---|
| Linux、C++、Python | 进程/线程、虚存、动态链接、编译、异常/RAII、并发和性能基础 | 能独立调试构建、ABI、内存、脚本、包、测试、profile | S01、S03、S04、S13 |
| 计算机体系结构/GPU | cache/带宽/延迟、SIMT、warp、occupancy、register/shared/DRAM、roofline | 能根据硬件计数器解释 kernel 快慢，做资源/访存设计 | S02–S04 |
| CUDA 算子开发 | execution/memory model、同步、原子、stream/event、数值精度 | 写多版本算子、dispatcher、binding、sanitizer、benchmark、NCU | S03/S04.5 |
| Transformer/LLM 推理 | attention/GQA/MQA、RoPE、RMSNorm、SwiGLU、KV、prefill/decode、sampling | 能逐层画 Qwen3，给出真实 shape/call/内存/热点 | S02/S04.5 |
| PyTorch 内部机制 | dispatcher、operator registration、module/state_dict、autocast、FakeTensor、FX/compile | torch.library + Meta + module/FX 替换，处理 stream/fallback | S04.5/S06 |
| 性能工程方法 | benchmark 统计、Roofline、Amdahl、正确性、实验控制 | micro/block/model/service 分层证据和原因分析 | 全项目，重点 S02–S04.5 |
| Triton/CUTLASS/vendor library | program/tile模型、GEMM hierarchy、autotune、适用边界 | 同 shape 多后端正确性/性能/IR 与 shape-aware dispatch | S04 |
| 低精度与量化 | FP16/BF16/TF32/INT8/INT4、scale、granularity、calibration、误差 | W4/W8/KV quant、packing、quant GEMM、质量性能 Pareto | S05 |

#### P1：主线完成后必须具备，决定岗位上限

| 知识/技能 | 理论掌握标准 | 实践掌握标准 | HQSB 证据 |
|---|---|---|---|
| 推理 Runtime | paged KV、continuous batching、prefix cache、chunked prefill、spec decode、CUDA Graph | 不只会调 vLLM；能实现/修改关键策略并做压力实验 | S07 |
| Serving 与性能治理 | 排队、tail latency、goodput、backpressure、SLO、流式/取消 | OpenAI API、loadgen、scheduler/router、故障与可观测 | S08 |
| AI Compiler | graph/IR、pattern、lowering、codegen、dynamic shape、cost model | FX/Inductor 主线，至少两个真实 pattern 自动 lower | S06/S11 |
| 分布式推理 | TP/PP/EP、collective、通信模型、拓扑、overlap、straggler | NCCL microbench、多卡 scaling、trace/overlap/故障 | S10 |
| Ascend C/CANN | NPU 内存/计算层次、tiling、CANN 运行与工具 | 等价算子/backend、模型接入、msprof、CUDA 映射 | S09 |
| vLLM/SGLang/TRT-LLM/llama.cpp | 各自执行模型、KV、scheduler、kernel/engine 特点 | 同 C4/C2/C6 的公平 adapter 和深入对比 | S07 |

#### P2：形成差异化与高级工程可信度

| 知识/技能 | 掌握/体现标准 | HQSB 证据 |
|---|---|---|
| 云原生与 SRE | 容器/K8s/GPU 调度、autoscaling、canary/rollback、SLO、SBOM、安全 | S13 |
| 跨硬件评测与选型 | 性能、质量、能耗、成本、容量和软件成熟度的 Pareto 决策 | S12 |
| MoE/长上下文/稀疏 | 选择一个方向做 kernel→runtime→service 闭环 | S14 |
| Diffusion/Stable Diffusion/DiT | 理解与 LLM 不同的 workload、attention/GEMM/conv/quant/serving；有岗位需求再扩 | S14 可选 |
| 训推协同 | checkpoint/adapter/quant/engine artifact 生命周期和一致性 | S14 |
| Vibe coding/LLM 辅助开发 | 能给 agent 无歧义输入输出/门禁，并亲自审计正确性、性能 claim 和架构一致性 | 本文、阶段 handoff、review 记录 |
| 技术写作与协作 | ADR、optimization log、review、release、第三方复现、英文文档 | 全阶段，重点 S15 |

### 27.3 每项技能应达到的深度层次

- **会解释**：能从第一性原理说明，不只背定义。
- **会实现**：能写最小正确版本，理解接口和错误路径。
- **会测量**：知道基线、变量、指标、统计和工具陷阱。
- **会优化**：能提出假设、读 profiler、做消融并解释反例。
- **会集成**：能处理框架、模型、Runtime 和服务边界。
- **会治理**：能通过契约、测试、证据、fallback、监控和发布让别人安全使用。

P0 技能至少达到“会集成”，其中 CUDA、模型、性能工程应达到“会治理”；P1 技能至少完成一个模块的“会优化/会集成”；P2 不求全部铺开，但选中的方向必须有真实闭环。

---

## 28. 简历、项目介绍与面试证据

### 28.1 项目一句话

> 设计并实现 HeteroQuantServeBench：一个以版本化契约和可复现证据链为核心、贯通 Qwen 模型 profiling、CUDA/Triton/CUTLASS 算子、框架集成、量化、Runtime/Serving 与异构硬件评测的 LLM 推理优化平台。

在 S04.5/后续尚未完成前，应缩小为：

> 构建面向 Jetson Qwen3-1.7B 的推理优化实验平台，完成契约化 benchmark、模型全栈 profiling，以及 CUDA/Triton/CUTLASS RMSNorm/GEMM 多后端算子工程；正在补齐真实模型算子替换与端到端收益闭环。

### 28.2 简历 bullet 模板

只能把 `{...}` 替换为经过 evidence gate 的真实数字：

- 基于 PyTorch Profiler、Nsight Systems/Compute、Roofline 与 Amdahl 对 Qwen3-1.7B 的 prefill/decode 热点、真实 tensor shape、KV/显存和能耗进行分层归因，形成 `{N}` 类 workload 的可复现实验协议与优化优先级。
- 使用 CUDA C++ 实现 RMSNorm V0/V1/V2 及 residual+RMSNorm 融合算子，通过 warp shuffle、向量化、alignment/tail fallback 优化，在 `{hardware/shape/dtype}` 上相对 `{baseline}` 达到 `{speedup}`，并用 `{NCU metrics}` 解释收益与退化边界。
- 实现 Triton/CUTLASS/cuBLAS/CUDA 统一 capability 与 shape-aware dispatcher，覆盖 `{shape matrix}` 和数值/stream/fallback 门禁，形成不同 prefill/decode GEMM 场景的 backend 决策表。
- 通过 `torch.library`、Meta/FakeTensor、module swap 与 FX pattern 将自定义算子接入 Qwen3，完成 operator/block/model 多级正确性和六 workload micro→model 归因，使 `{metric}` 改善 `{value}`；若没有显著收益，则写“验证端到端收益受 `{Amdahl/launch}` 限制并定位下一热点”。
- 构建版本化 C1–C7 契约、schema migration、registry、结构化 trace 和 evidence ledger，使模型、算子、量化、backend 和 benchmark 结果按 commit/config/model/operator/environment hash 可追踪。
- 后续完成量化/Runtime/Serving 后再添加：W4A16/KV quant 质量—内存—吞吐 Pareto；continuous batching/paged KV/spec decode；SLO goodput/P99；Ascend/多卡/跨硬件结果。

### 28.3 3 分钟项目讲法

1. **问题**：单看模型或单看 kernel 都不能保证推理优化有效，真正难点是跨层归因和可复现落地。
2. **架构**：C1–C7 契约统一模型、workload、operator、backend、quant、result、trace；优化从 profiling 进入 kernel，再接回模型和服务。
3. **代表工作**：Qwen3 prefill/decode profiling；CUDA RMSNorm 多版本；Triton/CUTLASS 对照与 dispatcher；S04.5 模型替换闭环。
4. **最深洞察**：没有万能最快后端；micro speedup 受真实 shape、调用占比、launch 和内存语义限制，必须用 Amdahl 与模型实测闭环。
5. **结果与下一步**：只说证据支持的数字；下一步从 S04.5 进入量化和 Runtime。

### 28.4 10 分钟技术讲法

- 1 分钟：问题与目标岗位关联；
- 1 分钟：架构和契约；
- 2 分钟：Qwen3 prefill/decode 与热点证据；
- 3 分钟：RMSNorm V0/V1/V2、融合、Triton/CUTLASS tradeoff；
- 2 分钟：torch.library/module/FX、正确性和 micro→model；
- 1 分钟：失败/反例、局限、量化/Runtime 路线。

### 28.5 30 分钟深入面试路径

1. 从 RMSNorm 数学推导 FLOPs/bytes/arithmetic intensity；
2. 画 V0/V1/V2 block/warp reduction 和访存；
3. 解释 float4/half2、alignment、tail、occupancy 与 FP16 退化；
4. 比较 Triton program 与 CUDA block、CUTLASS GEMM hierarchy；
5. 展示真实 Qwen block、shape census 和替换点；
6. 解释 current stream、autocast、Meta/FakeTensor、FX graph/fallback；
7. 展示 operator/block/model 正确性和 Amdahl；
8. 扩展到 quant kernel、KV cache、continuous batching 和 serving SLO；
9. 讨论 Ascend/分布式迁移；
10. 以失败优化和证据治理收尾。

### 28.6 必须准备的高频追问

- 为什么选 RMSNorm，而不是直接优化 GEMM/attention？profile 证据与 Amdahl 上限是什么？
- V1/V2 为什么快？为什么某些 hidden/FP16 下反而慢？
- `float4`/`half2` 对齐和 tail 如何保证？非 contiguous 怎么处理？
- default stream 与 PyTorch current stream 有什么风险？
- Triton GEMM mask 写错会发生什么？怎样测试没有越界？
- CUTLASS 默认配置为什么可能在 M=1 赢或输？tile/stage 如何选？
- dispatcher 如何避免基于一次噪声测量过拟合？cache 如何失效？
- 自定义 op 怎样被 `torch.compile` 看见？Meta/FakeTensor 做什么？
- fused residual+RMSNorm 的 FP16 舍入为何可能改变模型结果？
- kernel 快 `{X}` 为什么模型只快 `{Y}`？
- prefill 与 decode 的 GEMM shape 和优化目标如何不同？
- GQA 如何改变 KV cache 大小和 attention 读流量？
- W4A16 的瓶颈是 dequant 还是 GEMM？M=1 与 large-M 有何差别？
- paged KV cache 如何分配、回收、避免碎片？continuous batching 的迭代是什么？
- 为什么看 goodput/P99 而不是只看 tokens/s？
- 如何公平比较 Jetson、数据中心 GPU 和 Ascend？

### 28.7 最有价值的失败案例

失败结果不要删除，它们往往比“快了”更显水平：

- FP16 half2 因事务/实现细节没有带来预期收益；
- 小 hidden 下向量化负载不均而退化；
- 过大 block 导致 occupancy 崩塌；
- fused kernel 因 RAW 依赖/舍入语义收益有限；
- Triton autotune 在 edge GPU 选择非最优配置；
- microbenchmark 快，但算子占比低或 launch/框架开销使模型收益小；
- CUTLASS/cuBLAS/Triton 在不同 M/N/K 上没有统一赢家。

每个失败案例按“假设—证据—根因—修复/边界—后续决策”讲述。

---

## 29. 全局 Definition of Done

一个模块或阶段只有同时满足以下条目才可标“完成”：

### 29.1 设计与接口

- 职责、输入、输出、错误、stream/workspace/ownership、fallback 清晰；
- 与 C1–C7 对齐，schema/version/compatibility 有定义；
- 有 ADR 记录关键取舍，无循环依赖或跨层泄漏。

### 29.2 正确性

- reference 与数学定义存在；
- 正常、边界、随机、异常、NaN/Inf、shape/dtype/layout/stream 覆盖；
- 适当层级的 operator/block/model/service correctness 通过；
- tolerance 基于 dtype/算法/累加定义，不是一刀切。

### 29.3 性能

- 方法、warmup、sync、samples、统计、环境、温度/功耗明确；
- raw samples 与 profiler artifact 可访问；
- baseline 公平，首次编译与 steady state 分开；
- 原因通过 profiler/模型解释；负结果和适用边界保留；
- micro claim 不冒充 model/service claim。

### 29.4 工程质量

- unit/correctness/integration/performance 测试分层；
- CPU CI 和目标硬件 CI/nightly 合理拆分；
- lint/type/doc/schema/security 门禁生效；
- 可选依赖缺失可诊断，不破坏无关功能；
- 无 secrets、权重、机器绝对路径、未固定外部 main 依赖。

### 29.5 证据与发布

- commit/config/model/operator/environment/raw/report hash 完整；
- status/evidence/README 同步；
- 新用户有复现路径和 known limitations；
- 简历/面试 claim 与 evidence ledger 一一对应。

---

## 30. 主要风险与反模式

| 反模式 | 为什么危险 | 应对 |
|---|---|---|
| 阶段文档很多、功能目录很多，但没有真实闭环 | 面试追问两层即暴露 | 每阶段至少一个 hero path 达到 M4/M5 |
| 只跑 happy path shape | kernel tail/布局最容易 silent wrong | 真实 shape + odd/tail/property/sanitizer |
| 把 source 存在写成“已支持” | 证据不可信 | 强制 maturity/claim 标签 |
| raw 被 gitignore 后没有归档 | 性能数字无法第三方审计 | evidence manifest + Release/对象存储 |
| 过早进入量化/Serving | 基础集成问题和新变量混在一起 | 先完成 S04 strict + S04.5 |
| 只优化 microbenchmark | 端到端可能无收益 | micro→block→model→service 与 Amdahl |
| fallback 静默 | 看似运行其实没用自定义实现 | 结果记录 requested/actual/reason |
| 自动调优不可复现 | 配置随进程/机器漂移 | 固定 search space、cache key、失效和 raw |
| 依赖外部 main | 未来构建与结果漂移 | pin commit/tag/hash/license |
| 对所有硬件用同一绝对性能结论 | 忽略功耗、成本、容量、软件成熟度 | 多维 Pareto + workload/SLO 前提 |
| 简历堆技术栈 | 没有深入故事 | 一个纵向主故事 + 两个深挖子故事 |
| 让 Coding Agent 自行宣称通过 | 可能把计划/模拟数字写成结果 | 验收必须引用实际 evidence；未运行即标未验证 |

---

## 31. 从当前状态开始的推荐执行顺序

### 31.1 第一批：关闭 S04 安全与事实缺口

1. 修 Triton GEMM 全维 tail mask 和 M=1 行为。
2. CUDA C ABI/bridge 改 current stream，补 tensor/ABI validation。
3. 解除 fallback 对 Triton import 的依赖。
4. manifest 路径/空/重复安全；dependency test 去掉恒真绕过。
5. 统一 benchmark metric 公式和 WorkloadSpec 实际生效语义。

**退出条件**：P0 全部有测试，任何 unsupported 输入只会正确回退或明确失败。

### 31.2 第二批：让 S04 多后端声明真实

1. CUTLASS 固定版本、status/correctness、多个配置。
2. dispatcher 真正选择 CUTLASS；shape/dtype/layout-aware 并输出 reason。
3. autotune persistent cache、key 与 invalidation。
4. 从 Qwen shape census 生成 benchmark matrix。
5. evidence manifest 和 raw artifact 发布路径。

**退出条件**：每个后端至少有被自动选择的真实路径，报告可以重放。

### 31.3 第三批：完成 S04.5

严格按第 13 节 WP1→WP5：先白盒模型，再定义 op，再 module，再 FX，再 correctness，最后 performance。不得先写 speedup 报告再补正确性。

**退出条件**：至少 M4，六 workload 完整；能明确回答 micro→model。

### 31.4 第四批：S05→S06

- 先以 S04.5 的稳定 integration 接 W8A16/W4A16；
- 再把自定义/量化算子纳入图 pass/lowering；
- 输出质量—内存—性能 Pareto，而非单一压缩率。

### 31.5 第五批：S07→S08

- Runtime 先完成 KV/continuous batching/adapter；
- Serving 再做 HTTP/SLO/route/failure；
- 避免把一个现成引擎 server 启动成功当成 Runtime 开发完成。

### 31.6 第六批：按岗位选择增强

- 算子/异构岗位优先 S09 + S11；
- 推理引擎/Serving 岗位优先 S10 + S13；
- 顶尖综合 AI Infra 目标最终完成 S09–S13 的代表性闭环，不必每个子功能同等深度。

### 31.7 最后：S14 精选、S15 发布

只选择能形成可证据闭环的前沿扩展；随后冻结 release、做第三方复现、录制 demo、整理简历与面试 FAQ。

---

## 32. 项目完成后的最终能力画像

当 S00–S15 按本文标准完成时，这个项目能够证明你不是只会某一层 API，而是具备以下完整能力：

1. 能读懂真实 Transformer 模型、生成过程和 tensor shape；
2. 能用 profiler 和硬件计数器从模型找到根因，而非猜热点；
3. 能用 CUDA/Triton/CUTLASS/Ascend 写正确、可回退、可解释的高性能算子；
4. 能处理 FP16/BF16/INT8/INT4、量化 artifact、误差传播和低比特 kernel；
5. 能把自定义算子接入 PyTorch、FX/Inductor 和真实模型；
6. 能理解 KV cache、continuous batching、prefix cache、speculative decoding 和 CUDA Graph；
7. 能在 Serving 并发与 SLO 下做调度、路由、性能治理和故障恢复；
8. 能做多卡通信、异构 backend、跨硬件评测和成本/能效选择；
9. 能用契约、schema、CI、测试、trace、artifact 和文档把优化工程化；
10. 能诚实地解释成功、失败、适用边界，并让第三方复现。

这就是项目的最终意义：它不是保证某个岗位录用的“技术栈清单”，而是构建一个可审计的事实，证明你能沿着 AI 推理系统最关键的纵向链条持续做出正确工程判断。

---

## 33. 最终结论

当前 HQSB 已经拥有扎实的 S00–S04 骨架和一组可深入讲解的 CUDA/Triton/CUTLASS 工作，但真正决定项目层次的下一步不是继续横向添加模块，而是：

> **先严格补齐 S04 的安全、调度与证据缺口，然后完成 S04.5，把 Qwen3 架构、真实 shape、自定义算子、PyTorch/FX 接入、正确性和 micro→model 收益全部跑通。**

完成 S04.5 后，S05 量化、S06 编译、S07 Runtime 和 S08 Serving 都会拥有稳定支点；否则它们容易变成互不相连的 demo。对“推理优化/算子开发融合”岗位而言，S02→S03→S04→S04.5 是最重要的主故事，S05/S07/S09/S11 根据岗位形成第二层差异化，S12/S13/S15 则把技术深度转化为工业可信度和秋招证据。

本文应作为后续 Coding Agent 的总纲。每次阶段完成后只做三类更新：

1. 把对应条目从 `PLANNED/SOURCE` 提升到有证据支持的成熟度；
2. 写入实际文件、实验、raw artifact、报告和 claim ID；
3. 保留未完成项、失败结论和适用边界，绝不以计划替代事实。
