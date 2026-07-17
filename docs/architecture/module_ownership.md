# 模块 Ownership 与依赖规则

> 阶段：S01（核心契约与工程质量体系）；S06 追加 `integration` 区域与规则 R2；
> S07 追加 `runtime` 区域与规则 R3/R4；S08 追加 `serving` 区域与规则 R5/R6；
> S10 追加 `distributed` 区域与规则 R7/R8；S11 追加 `compiler` 区域与规则 R9/R10；
> S12 追加 `evaluation` 区域与规则 R11/R12；
> S13 追加 `infra` 区域与规则 R13/R14；
> S14 追加 `experimental` 区域与规则 R15/R16；
> S15 追加 `release` 区域与规则 R17/R18
> 版本：1.9.0

本文定义 HQSB 各 Python 模块的职责边界、所有权和依赖方向，是后续所有阶段
开发与 Code Review 的约束依据。任何违反依赖方向的导入都会在
`tests/unit/core/test_dependency.py`、`tests/unit/integration/test_import_boundaries.py`、
`tests/unit/runtime/test_runtime_import_boundaries.py`、
`tests/unit/serving/test_serving_import_boundaries.py`、
`tests/unit/distributed/test_distributed_import_boundaries.py`、
`tests/unit/compiler/test_compiler_import_boundaries.py`
与 `tests/unit/experimental/test_experimental_import_boundaries.py`、
`tests/unit/release/test_release_import_boundaries.py`
以及 `scripts/audit/import_dependency_gate.py`（规则 R1–R18）中被 CI 拦截。

## 1. 依赖图（Dependency Graph）

```mermaid
graph TD
    subgraph core["hqsb.core — 稳定地基（不依赖任何具体实现）"]
        errors["errors"]
        ids["ids"]
        logging["logging"]
        contracts["contracts (C1-C7)"]
        schema["schema (versioning/migrate)"]
        config["config (loader/hash)"]
        registry["registry"]
    end

    subgraph backends["hqsb.backends"]
        dummy["dummy"]
    end

    subgraph benchmark["hqsb.benchmark"]
        engine["engine"]
        metrics["metrics"]
        workload["workload"]
        model_core["model_core"]
        monitor["resource_monitor"]
    end

    subgraph models["hqsb.models"]
        loader["loader"]
        manifest["manifest"]
    end

    subgraph integration["hqsb.integration — 框架集成 / 图优化（S06）"]
        specs["specs / dispatch"]
        graph["graph / patterns / guards / cache / lowering"]
        telemetry["telemetry / differential / experiment"]
    end

    subgraph runtime["hqsb.runtime — 请求状态 / KV / 调度 / Runtime 契约（S07）"]
        request["request / adapter / parity"]
        schedule["scheduler / kv / prefix_cache"]
        runtime_exp["graph_route / spec_decode / failure / comparison / policy_ab"]
    end

    subgraph serving["hqsb.serving — 服务平面 / 网关 / 策略 / 证据（S08）"]
        protocol["protocol / sse / gateway"]
        policy["slo / arrival / admission / router / circuit / cache"]
        evidence["observability / service_ab / experiment / specs"]
    end

    subgraph distributed["hqsb.distributed — 拓扑 / parallel plan / collective / 证据（S10）"]
        topo["topology / placement / ranks / probes"]
        comm["backend / collectives / sequence / faults"]
        plan["parallel_plan / ledger / scaling / overlap / boundary / moe / traces"]
        devev["telemetry / specs / experiment / interface_map"]
    end

    subgraph compiler["hqsb.compiler — 图捕获 / IR / lowering / autotune / cache（S11）"]
        front["identity / ir / records / capture / guards"]
        mid["pattern_library / rewrite / targets / lowering / backend / codegen"]
        search["autotune / costmodel / cache / portable / aigate"]
        cev["telemetry / specs / experiment / interface_map"]
    end

    subgraph evaluation["hqsb.evaluation — 可比性 / capability / 四层重放 / 能效成本 / Pareto / lineage（S12）"]
        evf["identity / records / contracts / layers / campaign"]
        evc["candidates / comparability / platform / capability / benchmark / repeatability"]
        evm["roofline / energy / cost / pareto / maturity / lineage"]
        eve["telemetry / specs / experiment / interface_map"]
    end

    subgraph infra["hqsb.infra — 生产化 / 云原生 / 可靠性（S13）"]
        inf["identity / records / contracts / campaign"]
        ins["supply_chain / deployment / scheduling / artifacts / lifecycle"]
        inc["capacity / autoscaling / observability / faults / canary / security"]
        inv["telemetry / specs / experiment / interface_map"]
    end

    core --> backends
    core --> benchmark
    core --> models
    benchmark --> backends
    benchmark --> models
    benchmark --> integration
    integration --> runtime
    runtime --> serving
    serving --> distributed
    models --> compiler
    benchmark --> compiler
    models --> evaluation
    benchmark --> evaluation
    subgraph experimental["hqsb.experimental — 训推协同 / 前沿扩展 / 证据治理（S14）"]
        exf["identity / records / contracts / campaign"]
        exp1["dependencies / training / parity / posttraining / frontier"]
        exp2["speculative / moe / long_context / sparsity"]
        exp3["multimodal / agent / edge"]
        exv["telemetry / specs / experiment / interface_map"]
    end

    subgraph release["hqsb.release — 发布 / 开源 / 求职证据（S15）"]
        rlf["identity / records / contracts / campaign"]
        rl1["claims / quickstart / docs_gate"]
        rl2["hero_replay / figures / supply_chain"]
        rl3["demo / narrative / clean_room / upstream / first_impression"]
        rlv["telemetry / specs / experiment / interface_map"]
    end

    core --> infra
    core --> experimental
    core --> release

    classDef concrete fill:#f9e8e8,stroke:#c44;
    class backends,models,integration,runtime,serving,distributed,compiler,evaluation,infra,experimental,release concrete;
```

**规则：箭头只能从具体层指向 `core`，`core` 永不指向具体层。**
`hqsb.integration` 位于 `benchmark` 之上（可复用 `benchmark.metrics` 的数值口径），
但 `core`/`models`/`benchmark` 都不得反向依赖它（规则 R2）。
`hqsb.runtime` 位于 `integration` 之上（请求/KV/调度层消费算子能力与编译契约），
`core`/`models`/`benchmark`/`backends`/`hardware`/`quant`/`integration` 都不得反向
依赖它（规则 R3），且 runtime 自身不 import `ops`（规则 R4）。
`hqsb.serving` 位于服务平面（网关/策略/证据层），所有下层区域都不得反向
依赖它（规则 R6），serving 自身不 import `ops`（规则 R5），且不得在模块级 import
`torch`/`triton`/`numpy`（保证 CPU-minimal 可导入）。
`hqsb.distributed` 位于最顶端（多设备通信/并行/证据层，S10），所有下层区域都不得
反向依赖它（规则 R7），distributed 自身不 import `ops`（规则 R8），同样不得在模块级
import `torch`/`triton`/`numpy`。
`hqsb.evaluation` 位于跨硬件评估层（S12），所有下层区域（含 compiler）都不得反向依赖它
（规则 R11），evaluation 自身不 import `ops`（规则 R12），且不得在模块级 import
`torch`/`triton`/`numpy`。
`hqsb.infra` 位于生产化/云原生/可靠性层（S13）：它消费服务（S08）与评估（S12）的证据，
所有其他区域（含 evaluation）都不得反向依赖它（规则 R13），infra 自身不 import `ops`
（规则 R14），不得在模块级 import `torch`/`triton`/`numpy`，且**只依赖 `hqsb.core`**——
集群/镜像/故障/canary/多租户契约必须能在 CPU-minimal 环境独立导入与测试。

## 2. 模块边界与 Ownership

| 模块 | 职责 | 允许依赖 | 禁止依赖 |
|---|---|---|---|
| `hqsb.core.errors` | 错误分类 + exit code | stdlib | 任何项目模块 |
| `hqsb.core.ids` | run/trace/span ID | stdlib | — |
| `hqsb.core.logging` | 结构化日志 + trace context | stdlib | — |
| `hqsb.core.contracts` | C1–C7 版本化 schema + Backend ABC | `core.errors`、`pydantic` | 具体 backend/operator/model/serving |
| `hqsb.core.schema` | 版本化 + 迁移 | `core.errors` | — |
| `hqsb.core.config` | 配置加载 + hash | `core.errors`、`pydantic`、`yaml` | — |
| `hqsb.core.registry` | 插件注册 | `core.errors` | — |
| `hqsb.backends.*` | 具体 backend 实现 | `core` | 不得被 `core` 反向依赖 |
| `hqsb.benchmark.engine` | backend 接口编排 → BenchmarkResult | `core`、`benchmark.metrics` | 具体 backend/model loader |
| `hqsb.benchmark.model_core` | PyTorch reference 内部实现（S02 归入 backend） | `core`、`models.loader` | — |
| `hqsb.models.loader` | 模型加载（local-only） | `core.contracts`、`core.errors` | — |
| `hqsb.integration.*` | 框架集成：算子 schema/dispatcher、Meta/FakeTensor、图 IR、pattern 重写、guard/cache/lowering、CUDA Graph 契约、错误/ABI、生命周期、adapter、differential、C6/C7 投影 | `core`、`benchmark.metrics` | `ops`（kernel 由 lowering target 描述）、模块级 `torch`/`triton`（必须函数内惰性导入） |
| `hqsb.runtime.*` | 推理 Runtime：canonical request 与 capability 协商、adapter 七类操作、语义 parity oracle、请求状态机与 C7 span、iteration 账本、paged KV 几何/生命周期/容量、static/continuous/chunked 调度、prefix cache、graph 桶与 attention 能力、speculative/MTP 契约、失败矩阵、公平比较、策略 A/B、C6/C7 投影、实验脚手架 | `core`、`benchmark.metrics`（数值口径）、`integration.cuda_graph`（函数内惰性复用 S06 契约） | `ops`（kernel 以 capability/provider 名称描述）、`serving`、模块级 `torch`/`triton`（必须函数内惰性导入） |
| `hqsb.serving.*` | 服务平面：协议/SSE/错误目录、网关与请求状态机、传输（stdlib HTTP + 模拟）、时间边界与投递账本、SLO/到达/loadgen、公平性与队列策略、准入/熔断、多后端路由与 cache-aware 路由、故障注入、可观测性、服务级 A/B、S08 实验脚手架与接口对照表 | `core`、`benchmark.metrics`（数值口径）、`runtime` 公共契约对象（metrics/policy_ab/request/prefix_cache/comparison/telemetry/experiment） | `ops`（kernel 以 capability/provider 名称描述）、模块级 `torch`/`triton`/`numpy`（必须函数内惰性导入）、任何下游区域反向依赖 |
| `hqsb.distributed.*` | 多设备通信与并行层：拓扑身份/链路探测/placement、rank 与 group 身份、collective 语义与 CPU oracle、communicator 状态机与故障 oracle、ParallelPlan/通信账本、scaling 协议、overlap 区间代数、PP/CP/SP 门禁、MoE dispatch/combine、多 rank trace 与归因、C6/C7 投影、S10 实验脚手架与 300 步接口对照表 | `core`（错误/契约）、`runtime.experiment`（统一实验记录字段） | `ops`（collective/kernel 以 capability/provider 名称描述）、模块级 `torch`/`triton`/`numpy`（必须函数内惰性导入）、任何下游区域反向依赖 |
| `hqsb.compiler.*` | AI 编译器层：多级 artifact 身份与 lineage、HQSB canonical/targeted IR 与 verifier、capture/break/guard/symbolic domain census、语义 pattern 重写（near-miss 拒绝、幂等、原子性）、target capability 与 lowering registry、backend 契约与 CompileRun、IR→codegen→binary→counter 归因、autotune 搜索/预算/holdout、cost model regret 与低置信回退、编译制品 cache（key/事务/失效/损坏）、TVM/MLIR 可迁移 lowering 接口、AI 候选零信任门链、C6/C7 投影、S11 实验脚手架与 320 步接口对照表 | `core`（错误/契约/版本门）；其他区域一律不 import（`hqsb.compiler` 只依赖 `core`） | `ops`（kernel 以 capability/provider 名称 + artifact locator/hash 描述）、模块级 `torch`/`triton`/`numpy`（必须函数内惰性导入，FX importer 等重依赖仅函数内探测）、任何下游区域反向依赖 |
| `hqsb.infra.*` | 生产化/云原生/可靠性层（S13）：ReleaseBundle 与 OCI digest DAG 身份、§23 十四条交叉约束、campaign 目录与执行安全策略、供应链 gate（SBOM/漏洞/秘密/许可证/provenance/非 root）、clean 环境 bootstrap 与部署状态机、accelerator placement/capability label/NUMA 拓扑与隔离、模型制品 cache/原子激活/回滚/GC、probe/drain/rolling 生命周期、内存账本与 token/KV admission、metric-age 感知 autoscaling、request→hardware 可观测与 RCA、故障合同/降级/恢复、canary 状态机与自动回滚、多租户身份/配额/脱敏/审计、C6/C7 投影与 418 步接口对照表 | `core`（错误/契约/版本门）；其他区域一律不 import（`hqsb.infra` 只依赖 `core`） | `ops`（容器/模型/kernel 一律以 digest/identity 记录描述）、模块级 `torch`/`triton`/`numpy`（必须函数内惰性导入）、任何下游区域反向依赖、任何真实的集群/注册表凭据 |
| `hqsb.experimental.*` | 训推协同与前沿扩展层（S14）：canonical 身份与 seed bundle、S14 词汇表与四个状态机、§7 七个统一证据对象（TrainingRun/Checkpoint/ServingModel/PolicySnapshot/Trajectory/FrontierStudyContract/AdoptionDecision）、§22 运行目录布局与执行安全策略、依赖与 feature flag 边界（E14-01）、分布式训练状态与 checkpoint（E14-02）、转换 DAG 与七层训推一致性门（E14-03）、SFT/DPO/GRPO 手算 oracle 与 policy staleness（E14-04）、前沿 ADR 与预注册（E14-05）、四个条件 P0 分支（E14-F1…F4）、三个可选迁移（E14-06/07/08）、C6/C7 投影、冻结词汇表审计、三重门实验脚手架与 **480 步接口对照表** | `core`（错误/契约/版本门）；其他区域一律不 import（`hqsb.experimental` 只依赖 `core`） | `ops`（trainer/engine/kernel/device 一律以 capability/provider 名称 + artifact identity 描述）、模块级 `torch`/`triton`/`numpy`/`transformers`/`ray`/`vllm`（必须函数内惰性探测）、任何下游区域反向依赖 |
| `hqsb.release.*` | 发布/开源/求职证据层（S15）：canonical 身份与三个冻结对象（ReleaseCandidateSnapshot/PublicEvidenceBundle/FinalAcceptanceDecision）、ClaimRecord 与十门裁决、ContributionRecord、六态实验状态与 11 个实验 record、§22 统一运行数据包与执行安全策略、全局 Claim Ledger/声明扫描/证据门（E15-01）、clean CPU quickstart 契约（E15-02）、GPU/NPU hero 重放与 Amdahl 预测（E15-03）、可执行文档/双语一致性/能力矩阵（E15-04）、release 供应链/provenance/SBOM/许可证（E15-05）、图表→raw lineage 与重生成（E15-06）、demo 故障注入与诚实降级（E15-07）、3/10/30 分钟讲述与对抗问答（E15-08）、第三方 clean-room 复现（E15-09）、真实上游贡献（E15-10）、目标读者首屏研究（E15-11）、四重门实验脚手架与 **495 步接口对照表** | `core`（错误/契约/版本门）；其他区域一律不 import（`hqsb.release` 只依赖 `core`） | `ops`（包/模型/kernel/容器/上游仓库一律以 digest/identity/manifest 记录描述）、模块级 `torch`/`triton`/`numpy`/`requests`/`httpx`（必须函数内惰性探测）、任何下游区域反向依赖、任何真实发布/上游提交副作用 |
| `hqsb.evaluation.*` | 跨硬件评估与统一 benchmark 层（S12）：字节/canonical/aggregate 身份、Comparison Contract 与字段分类、四态可比性裁决与非法 join 防护、candidate 身份与上游证据五态、capability 四级证据与失效规则、四层统一重放（Observation/NormalizedResult/11 条交叉校验）、重复性与排除账本、分层 Roofline/Amdahl 预测与残差、能量窗口/积分/能效、云/自建 TCO 与成本敏感性、业务画像 Pareto 与决策回归、软件成熟度 rubric、端到端 lineage 与重生成、C6/C7 投影、S12 实验脚手架与 360 步接口对照表 | `core`（错误/契约/版本门）；其他区域一律不 import（`hqsb.evaluation` 只依赖 `core`） | `ops`（kernel 以 capability/provider 名称 + artifact locator 描述）、模块级 `torch`/`triton`/`numpy`（必须函数内惰性导入）、任何下游区域反向依赖、任何内置价格/电价/汇率常量（campaign 输入） |

## 3. 依赖方向约束（强制）

1. **`hqsb.core` 不得 import** `hqsb.backends`、`hqsb.models`、`hqsb.quant`、
   `hqsb.serving`、`hqsb.benchmark`、`hqsb.integration`、`hqsb.runtime`、`ops`。
2. **`hqsb.benchmark.engine` 不得 import 具体 backend** —— backend 一律通过
   构造函数注入或从 `Registry` 解析。
3. **`hqsb.core` 仅允许 import** `hqsb.core.*`、`pydantic`、`yaml` 与 stdlib。
4. **`hqsb.integration` 不得 import `ops`**，且**不得在模块级 import
   `torch`/`triton`/`numpy`**（重依赖必须函数内惰性导入，缺失时给结构化原因；
   S06 交付的 CPU-minimal 可导入性由此保证）。
5. **`hqsb.models` / `hqsb.benchmark` / `hqsb.core` 不得 import
   `hqsb.integration`**（gate 规则 R2）—— 集成层在它们之上，不是它们的依赖。
6. **`hqsb.runtime` 不得被 `core`/`models`/`benchmark`/`backends`/`hardware`/
   `quant`/`integration` import**（gate 规则 R3）—— 请求/KV/调度层在它们之上；
   反向依赖会把“可测量”与“测量器”绑死。
7. **`hqsb.runtime` 不得 import `ops`**（gate 规则 R4），且**不得在模块级 import
   `torch`/`triton`/`numpy`**：Runtime 适配器只声明 capability/provider 名称与
   源码身份，重引擎一律函数内惰性探测（缺失时给结构化 reason，而不是 ImportError）。
8. **`hqsb.serving` 不得被任何下层区域 import**（gate 规则 R6）：服务平面不可被
   “被服务”的实现反向绑死。
9. **`hqsb.serving` 不得 import `ops`**（gate 规则 R5），且**不得在模块级 import
   `torch`/`triton`/`numpy`**：kernel 以 capability/provider 名称描述，重引擎一律
   函数内惰性探测。
10. **`hqsb.distributed` 不得被任何下层区域 import**（gate 规则 R7）：多设备通信层在
    最顶端，反向依赖会把“并行/通信策略”与被并行的实现绑死。
11. **`hqsb.distributed` 不得 import `ops`**（gate 规则 R8），且**不得在模块级 import
    `torch`/`triton`/`numpy`**：collective 后端以 capability/provider 名称描述，
    重引擎（NCCL/HCCL/torch.distributed）一律函数内惰性探测并给出结构化 reason。
12. **`hqsb.compiler` 不得被任何下层区域 import**（gate 规则 R9）：
    `core`/`models`/`benchmark`/`backends`/`hardware`/`integration`/`quant`/`runtime`/
    `serving`/`distributed` 都不得反向依赖编译器——否则“可测量”的实现会被优化器绑死。
13. **`hqsb.compiler` 不得 import `ops`**（gate 规则 R10），且**不得在模块级 import
    `torch`/`triton`/`numpy`**：kernel 以 capability/provider 名称加 artifact locator/hash
    描述；lowering registry 在 import 时不得编译、创建 device context 或下载依赖；
    FX/Export 等 torch 依赖只在函数内惰性导入并给出结构化 reason（`NOT_INSTALLED`）。
14. **`hqsb.evaluation` 不得被任何下层区域 import**（gate 规则 R11）：
    `core`/`models`/`benchmark`/`backends`/`hardware`/`integration`/`quant`/`runtime`/
    `serving`/`distributed`/`compiler` 都不得反向依赖评估层——否则"可测量"的实现会被
    测量器绑死。
15. **`hqsb.infra` 不得被任何下层区域 import**（gate 规则 R13）：
    生产化/部署/故障/发布治理层消费服务与评估证据，反向依赖会把"被部署/被治理"的实现
    绑到部署器上；因此 `core`/`models`/`benchmark`/`backends`/`hardware`/`quant`/
    `integration`/`runtime`/`serving`/`distributed`/`compiler`/`evaluation` 都不得 import `hqsb.infra`。
16. **`hqsb.infra` 不得 import `ops`**（gate 规则 R14），且**不得在模块级 import
    `torch`/`triton`/`numpy`**：镜像/模型/kernel 一律以 digest/identity 记录描述，
    该层只依赖 `hqsb.core`，在 CPU-minimal 环境可独立导入与测试；集群凭据、
    注册表凭据与真实故障目标一律作为 campaign 输入，不内置。
17. **`hqsb.evaluation` 不得 import `ops`**（gate 规则 R12），且**不得在模块级 import
    `torch`/`triton`/`numpy`**：kernel 以 capability/provider 名称 + artifact locator 描述；
    该层只依赖 `hqsb.core`，在 CPU-minimal 环境可独立导入与测试；价格/电价/汇率不内置，
    一律作为带日期的 campaign 输入 artifact。
18. **`hqsb.experimental` 不得被任何下层区域 import**（gate 规则 R15）：
    train-serve 协调与前沿扩展层消费 S07/S08/S10/S13 的证据，反向依赖会把
    "被优化的推理主线"绑到 feature-gated 的研究路径上；
    因此 `core`/`models`/`benchmark`/`backends`/`hardware`/`quant`/`integration`/
    `runtime`/`serving`/`distributed`/`compiler`/`evaluation`/`infra` 都不得 import `hqsb.experimental`。
19. **`hqsb.experimental` 不得 import `ops`**（gate 规则 R16），且**不得在模块级 import
    `torch`/`triton`/`numpy`/`transformers`/`ray`/`vllm`**：trainer、engine、kernel 与设备
    一律以 capability/provider 名称 + artifact identity 描述；该层只依赖 `hqsb.core`，
    在 CPU-minimal 环境可独立导入与测试（子进程探针实测拉入重框架数 = 0）。
    这是 E14-01 想证明的边界"在源码结构上就成立"的前提。
20. **`hqsb.release` 不得被任何下层区域 import**（gate 规则 R17）：
    发布/开源/求职证据层消费 S00–S14 的冻结证据（acceptance、claim ledger、evidence
    bundle），反向依赖会把"被发布/被审计"的实现绑到发布器上；
    因此 `core`/`models`/`benchmark`/`backends`/`hardware`/`quant`/`integration`/
    `runtime`/`serving`/`distributed`/`compiler`/`evaluation`/`infra`/`experimental`
    都不得 import `hqsb.release`。
21. **`hqsb.release` 不得 import `ops`**（gate 规则 R18），且**不得在模块级 import
    `torch`/`triton`/`numpy`/`requests`/`httpx`**：包、模型、kernel、容器与上游仓库
    一律以 digest/identity/manifest 记录描述；该层只依赖 `hqsb.core`，在 CPU-minimal
    环境可独立导入与测试（审计工具本身不得要求 GPU 或联网）。

## 4. 扩展点（Extension Points）

第三方开发者（或 Coding Agent）只需：

1. 实现 `hqsb.core.contracts.Backend` 抽象接口；
2. 在 `RegistryHub.backends` 注册其工厂；
3. 通过 `BenchmarkEngine` 运行并产出 `BenchmarkResult`。

参考实现：`hqsb/backends/dummy.py`。

## 5. 验证方式

```bash
pytest tests/unit/core/test_dependency.py -q                        # hqsb/core 的 import 边界
pytest tests/unit/integration/test_import_boundaries.py -q          # integration 边界 + 无模块级重依赖
pytest tests/unit/runtime/test_runtime_import_boundaries.py -q      # runtime 边界 + R3/R4 + 无模块级重依赖
pytest tests/unit/serving/test_serving_import_boundaries.py -q      # serving 边界 + R5/R6 + 无模块级重依赖
pytest tests/unit/distributed/test_distributed_import_boundaries.py -q  # distributed 边界 + R7/R8 + 无模块级重依赖
pytest tests/unit/compiler/test_compiler_import_boundaries.py -q        # compiler 边界 + R9/R10 + 无模块级重依赖
pytest tests/unit/evaluation/test_evaluation_import_boundaries.py -q    # evaluation 边界 + R11/R12 + 无模块级重依赖
pytest tests/unit/infra/test_infra_import_boundaries.py -q              # infra 边界 + R13/R14 + 无模块级重依赖
pytest tests/unit/experimental/test_experimental_import_boundaries.py -q  # experimental 边界 + R15/R16 + 无模块级重依赖
pytest tests/unit/release/test_release_import_boundaries.py -q            # release 边界 + R17/R18 + 无模块级重依赖
python3 scripts/audit/import_dependency_gate.py                     # 规则 R1–R18 + 环检测（0 violations）
```

这些检查用 AST 静态扫描 import（不走运行时），任何违反依赖方向的导入都会失败；
`hqsb.integration`/`hqsb.runtime`/`hqsb.serving`/`hqsb.distributed`/`hqsb.compiler`/
`hqsb.evaluation`/`hqsb.experimental`
的 CPU-minimal 可导入性另由子进程探针测试固定（不依赖本进程是否已 import torch）。

> 注意（S10 落地约束）：`hqsb/distributed/__init__.py` 只提供 PEP 562 惰性
> `_LAZY` 映射，**不在模块级（包括 `TYPE_CHECKING` 块）导入子模块**——包级导入边
> 会形成 `hqsb.distributed → experiment → specs → hqsb.distributed` 的环，被 R1–R8
> 的环检测拒绝。

> 注意（S13 落地约束）：`hqsb/infra/__init__.py` 同样只提供 PEP 562 惰性 `_LAZY`
> 映射（同样的环风险：`hqsb.infra → experiment → specs → hqsb.infra`）；
> `hqsb.infra` **只依赖 `hqsb.core`**，`configs/infra/*.yaml` 为冻结词汇表（无测量值）。

> 注意（S11 落地约束）：`hqsb/compiler/__init__.py` 同样只提供 PEP 562 惰性 `_LAZY`
> 映射（同样的环风险：`hqsb.compiler → experiment → specs → hqsb.compiler`）；
> `hqsb.compiler` **只依赖 `hqsb.core`**（不依赖 integration/runtime/quant 等任何其他
> 区域），以保证编译层可以在 CPU-minimal 环境独立导入与测试。

> 注意（S14 落地约束）：`hqsb/experimental/__init__.py` 同样只提供 PEP 562 惰性 `_LAZY`
> 映射（同样的环风险：`hqsb.experimental → experiment → specs → hqsb.experimental`）；
> `hqsb.experimental` **只依赖 `hqsb.core`**，`configs/experimental/*.yaml` 为冻结词汇表
> （由 `scripts/experimental/gen_experimental_specs.py` 生成，**不含任何测量值**）。

> 注意（S15 落地约束）：`hqsb/release/__init__.py` 同样只提供 PEP 562 惰性 `_LAZY`
> 映射（同样的环风险：`hqsb.release → experiment → specs → hqsb.release`）；
> `hqsb.release` **只依赖 `hqsb.core`**，`configs/release/*.yaml` 为冻结词汇表
> （由 `scripts/release/gen_release_specs.py` 生成，**不含任何测量值**）。
