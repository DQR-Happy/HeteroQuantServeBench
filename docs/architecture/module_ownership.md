# 模块 Ownership 与依赖规则

> 阶段：S01（核心契约与工程质量体系）
> 版本：1.0.0

本文定义 HQSB 各 Python 模块的职责边界、所有权和依赖方向，是后续所有阶段
开发与 Code Review 的约束依据。任何违反依赖方向的导入都会在
`tests/unit/core/test_dependency.py` 中被 CI 拦截。

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

    core --> backends
    core --> benchmark
    core --> models
    benchmark --> backends
    benchmark --> models

    classDef concrete fill:#f9e8e8,stroke:#c44;
    class backends,models concrete;
```

**规则：箭头只能从具体层指向 `core`，`core` 永不指向具体层。**

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

## 3. 依赖方向约束（强制）

1. **`hqsb.core` 不得 import** `hqsb.backends`、`hqsb.models`、`hqsb.quant`、
   `hqsb.serving`、`hqsb.benchmark`、`ops`。
2. **`hqsb.benchmark.engine` 不得 import 具体 backend** —— backend 一律通过
   构造函数注入或从 `Registry` 解析。
3. **`hqsb.core` 仅允许 import** `hqsb.core.*`、`pydantic`、`yaml` 与 stdlib。

## 4. 扩展点（Extension Points）

第三方开发者（或 Coding Agent）只需：

1. 实现 `hqsb.core.contracts.Backend` 抽象接口；
2. 在 `RegistryHub.backends` 注册其工厂；
3. 通过 `BenchmarkEngine` 运行并产出 `BenchmarkResult`。

参考实现：`hqsb/backends/dummy.py`。

## 5. 验证方式

```bash
pytest tests/unit/core/test_dependency.py -q
```

该测试用 AST 静态扫描 `hqsb/core` 的所有 import，违规则失败。
