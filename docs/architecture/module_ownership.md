# 模块 Ownership 与依赖规则

> 阶段：S01（核心契约与工程质量体系）；S06 追加 `integration` 区域与规则 R2；
> S07 追加 `runtime` 区域与规则 R3/R4；S08 追加 `serving` 区域与规则 R5/R6
> 版本：1.3.0

本文定义 HQSB 各 Python 模块的职责边界、所有权和依赖方向，是后续所有阶段
开发与 Code Review 的约束依据。任何违反依赖方向的导入都会在
`tests/unit/core/test_dependency.py`、`tests/unit/integration/test_import_boundaries.py`、
`tests/unit/runtime/test_runtime_import_boundaries.py`
与 `scripts/audit/import_dependency_gate.py`（规则 R1–R4）中被 CI 拦截。

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

    core --> backends
    core --> benchmark
    core --> models
    benchmark --> backends
    benchmark --> models
    benchmark --> integration
    integration --> runtime
    runtime --> serving

    classDef concrete fill:#f9e8e8,stroke:#c44;
    class backends,models,integration,runtime,serving concrete;
```

**规则：箭头只能从具体层指向 `core`，`core` 永不指向具体层。**
`hqsb.integration` 位于 `benchmark` 之上（可复用 `benchmark.metrics` 的数值口径），
但 `core`/`models`/`benchmark` 都不得反向依赖它（规则 R2）。
`hqsb.runtime` 位于 `integration` 之上（请求/KV/调度层消费算子能力与编译契约），
`core`/`models`/`benchmark`/`backends`/`hardware`/`quant`/`integration` 都不得反向
依赖它（规则 R3），且 runtime 自身不 import `ops`（规则 R4）。
`hqsb.serving` 位于最顶端（服务平面/网关/策略/证据层），所有下层区域都不得反向
依赖它（规则 R6），serving 自身不 import `ops`（规则 R5），且不得在模块级 import
`torch`/`triton`/`numpy`（保证 CPU-minimal 可导入）。

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
8. **`hqsb.serving` 不得被任何下层区域 import**（gate 规则 R6）：服务平面在最顶端，
   反向依赖会把“服务”与“被服务”的实现绑死。
9. **`hqsb.serving` 不得 import `ops`**（gate 规则 R5），且**不得在模块级 import
   `torch`/`triton`/`numpy`**：kernel 以 capability/provider 名称描述，重引擎一律
   函数内惰性探测。

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
python3 scripts/audit/import_dependency_gate.py                     # 规则 R1–R6 + 环检测（0 violations）
```

这些检查用 AST 静态扫描 import（不走运行时），任何违反依赖方向的导入都会失败；
`hqsb.integration` 与 `hqsb.runtime` 的 CPU-minimal 可导入性另由子进程探针测试
固定（不依赖本进程是否已 import torch）。
