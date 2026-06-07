# S06 图集成与编译链设计（Design 制品）

> 本文件是 S06 六类固定制品中的 **Design** 制品：目标、边界、接口、选项与被
> 采纳/否决的决策（ADR）。**不包含任何实验结果或数字**。
> 依据：`docs/stages/S06_框架集成与图优化.md`、
> `docs/stage_experiments/details/S06/README.md` §5–§17、
> `docs/architecture/HQSB_项目完整剖析.md` §3.2/§15/§26。

## 1. 要解决的问题

S04.5 证明「两个算子能安全接进真实 Qwen」；如果 S06 只是再做一遍手工 module
swap，它就没有存在价值。S06 必须让接入变成**机制**：同一个算子能力能穿过
schema → dispatcher → fake → 图捕获 → pattern 重写 → lowering → 编译缓存 →
真实 kernel → 模型，并且在动态 shape、版本变化、编译失败、反复启停和跨模型
复用时仍然可解释、可回退、可审计。

## 2. 分层与职责（code map）

```text
eager Python/module 层   hqsb/integration/adapter.py（ModelAdapter/BackendAdapter、
                         census、硬编码扫描、改动预算、dummy backend）
        ↓
Dispatcher 层            hqsb/integration/specs.py + dispatch.py
                         （schema = 编译契约；注册矩阵；redispatch；capability；fallback）
        ↓
Meta/FakeTensor 层       hqsb/integration/meta.py
                         （符号维度、元数据契约、real-vs-fake oracle、无分配证据）
        ↓
Dynamo/Export/FX 层      hqsb/integration/graph.py + guards.py
                         （图 IR + capture mode/IR level；guard/break/recompile 记账）
        ↓
Rewrite/Lowering 层      hqsb/integration/patterns.py + lowering.py
                         （结构候选→语义谓词→副本重写；能力驱动选型与拒绝原因）
        ↓
Compile artifact/cache   hqsb/integration/cache.py
                         （graph/compile identity；相位计时；entry 校验；失效矩阵）
        ↓
Runtime/CUDA Graph 层    hqsb/integration/lifecycle.py + cuda_graph.py
                         （生命周期状态机、资源核算、capture/replay 契约与 claim 门）
        ↓
证据与契约投影           hqsb/integration/telemetry.py（C6/C7）
                         hqsb/integration/taxonomy.py + abi.py（错误/ABI）
                         hqsb/integration/differential.py（四级正确性地板）
```

**为什么是 `hqsb/integration`**：控制平面 §3.2 明确把 `torch.library`、
module swap、`FX pattern`、Meta/FakeTensor 划给 `hqsb/integration`；
`hqsb/compiler` 是 S11 的家（IR/cost model/schedule search/autotune/codegen）。
S06 只交付**有限 pattern 与 lowering registry**，不提前吞掉 S11 的范围。

## 3. 依赖方向（与 `module_ownership.md` 一致）

```text
hqsb.core ← hqsb.models / hqsb.benchmark ← hqsb.backends / hqsb.integration / hqsb.quant
          ← ops adapters ← runtime ← serving
```

- `hqsb.integration` 只依赖 `hqsb.core`（契约/错误）与 `hqsb.benchmark.metrics`
  （复用既有数值口径）；
- **不**依赖 `ops/*`（kernel 由 lowering target 描述，实际执行属执行层）；
- `hqsb.core`/`hqsb.models`/`hqsb.benchmark` **不得**依赖 `hqsb.integration`
  （`import_dependency_gate.py` 规则 R1/R2 + 单测固定）；
- torch/triton 一律**函数内惰性 import**，模块级 import 由测试禁止。

## 4. ADR 摘要

### ADR-S06-01：C6/C7 用投影而不是改 schema

- **Context**：E06 details §14 要求 C6 至少携带 14 个图/编译字段、C7 至少记录
  18 类事件；C1–C7 是 S01 冻结契约，带版本化与迁移测试。
- **Options**：(a) 给 C6/C7 加字段/枚举成员；(b) 用既有扩展点投影。
- **Decision**：(b)。字段落在 `BenchmarkResult.summary`（命名空间 `s06`）、
  `artifact_links`、`CorrectnessReport.details`；事件落在 `TraceEvent.attributes`
  （`hqsb_kind` + `hqsb_run_id`），`event_type` 只作粗分类。
- **Consequence**：零迁移风险；代价是 C6/C7 的"S06 字段"不在 pydantic 静态字段里，
  因此必须用 `c6_field_coverage()`/`c7_kind_coverage()` 把可寻址性变成可执行断言
  （`test_lowering_telemetry.py` 固定）。

### ADR-S06-02：`hqsb::fused_add_rms_norm` 采用 functional 契约

- **Context**：E06-01 §7 要求 functional / mutable 二选一并与 S04.5 语义一致。
- **Options**：(a) mutable（原地写 residual，省一次写）；(b) functional（返回新张量）。
- **Decision**：(b)。理由：alias/functionalization/CUDA Graph 写风险最小；
  schema 与实际 kernel 的写行为天然一致，不可能"撒谎"。
- **Consequence**：多出的拷贝/分配必须在 E06-07 的 allocation 记账中显式出现
  （`AllocationAccount`）；若将来需要 mutable，必须是独立私有 op。

### ADR-S06-03：lowering 规则不得以模型/模块名为键

- **Context**：E06-07 §5「若选择规则只是硬编码模块名，应在 E06-11 失败」。
- **Decision**：`PriorityRule` 只允许能力标签（op/dtype/model 无关）；
  `LoweringRegistry.selection_audit()` 静态扫描规则文本中的模型/模块名 token；
  `adapter.hardcode_scan()` 对 core 代码做同样检查，例外必须带
  `# hqsb-hardcode-allow` 标记并被计数上报。
- **Consequence**：第二目标只能通过 adapter + capability 接入；core 内的
  模型特例会被测试拦截。

### ADR-S06-04：CUDA Graph 是 P1，默认 `NOT_CLAIMED`

- **Context**：E06-08 只在项目声称该能力时才必做；未做时不允许用 `NOT_RUN` 规避。
- **Decision**：交付完整契约（捕获前置条件、静态缓冲、输出契约、失败矩阵、
  pool 记账、break-even）与 `claim_status()` 门；`graph_spec.yaml` 显式
  `claim_cuda_graph: false`。
- **Consequence**：报告只能写 `NOT_CLAIMED`，任何"CUDA Graph 降低了 CPU launch"
  之类表述在本阶段都不可用。

### ADR-S06-05：S04.5 前置证据改为标记文件（更严）

- **Context**：S05 的检查把 `hqsb/integration/` 目录存在当作 S04.5 M4 证据，
  而 S06 的代码就落在该路径。
- **Decision**：改为要求 `hqsb/integration/s04_5_evidence.json`（S04.5 执行产物）。
- **Consequence**：S06 脚手架不会伪造上游证据；代价是 S04.5 必须落一个约定文件
  （契约见 `S06_开发报告.md` §10.1）。

### ADR-S06-06：失败与降级一律带 reason，且 strict 模式必须失败

- **Decision**：`RouteDecision`（requested/actual/reason/fallback_used）与
  `FailureRecord` 是唯一合法的降级表达；`FallbackPolicy(strict=True)` 与
  `FallbackRegistry(strict=True)` 不返回替代路径；`S06ResultFields` 在
  requested≠actual 且无 `fallback_reason` 时**构造即拒绝**。
- **Consequence**：静默降级在数据结构层面不可表达。

## 5. 证据与数据布局

运行目录（`hqsb.integration.experiment.RUN_LAYOUT`，由 `RunDirectory.create()` 物化）：

```text
experiment_results/S06/E06-xx/<run_id>/
  preregistration.json      # 预注册（含 claim_boundary，未填即拒绝）
  prerequisites.json        # 前置门逐项证据/原因
  environment_fingerprint.json
  status.json | verdict.json  # 前者为非结论状态；后者默认拒绝写结论
  report.md                 # write_report_skeleton()：NOT_RUN 占位，无数字
  commands/ stdout/ stderr/
  environment/ operator/
  graph/{before,after,guards,breaks}/
  compile/{phases,generated,cache}/
  dispatch/ correctness/ performance/ memory/ profiler/ traces/ errors/
```

`EvidenceManifest` 在控制平面 §26.3 字段之外增加 S06 扩展块（compile mode、
graph/compile identity、graph/break 计数、cache 层与命中、编译相位耗时、
requested/actual lowering、observed kernel、fallback reason），与 C6 投影字段一一对应。

## 6. 与相邻阶段的边界

| 阶段 | S06 消费 | S06 交付 |
|---|---|---|
| S04.5 | 可逆替换语义、逐层数值基线、s04_5 证据标记 | 面向图系统的 stable op target 与 fallback 契约 |
| S05 | QuantArtifact scheme/group/layout/tail policy（以 `QuantDescriptor` 鸭子类型消费，不 import） | pattern 量化边界与 `lowering` 的 group/tail 能力校验 |
| S07 | — | compiled callable 契约、shape bounds、actual lowering、phase 基线、C6/C7 |
| S11 | — | 有限 pattern/lowering registry、跨模型 pattern 漂移、避免硬编码的边界 |

## 7. 未决问题（open questions，交给执行者/后续阶段）

1. **pattern 覆盖率与动态时间**：需要真实 C7 计数才能把静态 hit 映射到 time coverage；
   当前 `coverage_report()` 接受 `RuntimeUse` 但无真实数据。
2. **Inductor 版本专用内部 API**：`graph.from_fx_graph_module` 只使用公开图接口；
   若执行时需要 Inductor 内部 IR/debug 信息，必须隔离进版本化 adapter
   （details README §5.4 的要求）。
3. **多架构 fatbin**：`abi.CompatibilityMatrix` 目前按"精确相等"的保守规则判定
   （与 `ops/dispatcher.py` 一致），多目标 fatbin 的支持留待后续（S04 验收报告 §6 已登记）。
4. **第二真实模型**：见 `S06_开发报告.md` §10.3。
