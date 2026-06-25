# S09 Ascend Backend Report（Ascend C/CANN 异构后端设计）

> 阶段：S09  
> 性质：设计文档与实现说明  
> 生成时间：2026-09-18

---

## 1. 概述

本报告描述 HQSB S09 阶段中 Ascend C/CANN 异构后端的整体设计、模块职责、接口契约与实现细节。该后端遵循项目顶层架构的依赖规则，提供从算子级到模型级的完整能力，同时保持与 CUDA 后端的公共 Contract（C1–C7）一致。

**核心目标**：证明同一 OperatorSpec/test vectors/Qwen ModelArtifact/workload 在 CUDA 与 Ascend 上可复现，而不是重写一套不可比较的 demo。

---

## 2. 架构位置与依赖关系

```
core ← models/benchmark ← backends/integration/quant/ascend ← runtime ← serving
```

`hqsb.ascend` 是一个 **backend adapter region**，位于 `integration`/`quant` 同级，被 `runtime`/`serving` 通过 C4 Backend ABC 消费。

**依赖方向**：
- ✅ ascend → core（contracts/errors/config/fingerprint）
- ✅ ascend → benchmark.metrics（percentile/error metrics）
- ✅ ascend → quant.compat（QuantArtifact compatibility）
- ✅ ascend → integration.abi（custom op schema/ABI）
- ❌ ascend ↛ ops（kernel addressed by provider name, lazy probe）
- ❌ ascend ↛ runtime/serving（layering rule R7/R8/R9）

---

## 3. 模块职责总览

| 模块 | 职责 | E09 实验覆盖 |
|---|---|---|
| `compatibility.py` | CompatibilityManifest（hardware/ascend_stack/framework/project/official_sources/evidence sections）、canonicalize/hash、schema validation、field-level drift diff、redaction | E09-01 §4/§10/§11/§25/§26 |
| `capability.py` | Four-state CapabilityTable（SUPPORTED_VERIFIED/SUPPORTED_UNVERIFIED/UNSUPPORTED/UNKNOWN + NON_GOAL scope）、predicates with reason codes、merge_probe_updates | E09-01 §4.4/§24/§28 |
| `probes.py` | Probe suite（stack probes/device_memory/compile/load_artifact/async_kernel_smoke/framework_device/custom_op_framework_call/profiler_smoke/dtype_layout_format/stream_workspace_error/negative injections）、CommandExecutor/SubprocessExecutor/FixtureExecutor | E09-01 steps 4-20/22-24 |
| `tiling.py` | TilingData ABI（72 bytes versioned struct）、checked arithmetic、UB budget（rmsnorm/reduce/elementwise models）、host tiling function、candidate generation + legality filter、header generator | E09-02/03/04 |
| `operators.py` | Frozen operator semantics（Add/RowReduceSum/RMSNorm specs）、CPU FP64 oracles、ErrorMetrics set、ToleranceSpec gate（operator+dtype only） | E09-02/03 |
| `test_vectors.py` | Case matrix（development/tuning/confirmation/adversarial splits）、deterministic data generators、GuardedBuffer（poison regions）、TestVectorArtifact with sha256 hash | E09-02/03 |
| `sim_device.py` | CPU test double of Ascend execution model（SimulatedGlobalMemory/SimulatedUb/SimulatedQueue kernels for add/reduce/rmsnorm）、clearly labeled as simulated、claim_allowed() = False | E09-02/03/04 |
| `backend.py` | C4 Backend implementation（AscendBackend）with capabilities/load/warmup/generate/health/metrics/close、requested/actual + fallback reason | E09-05 |
| `framework.py` | ACL/aclnn two-phase workspace contract、current device/stream acquisition、async launch record、tensor metadata validation、actual-path evidence、unsupported predicate、explicit fallback policy、concurrency isolation、invariant checks | E09-05 |
| `quant_format.py` | Three precision layers（storage/transport/compute）、QuantArtifact compatibility validator with field-level JSON Pointer reasons、golden packed bytes、pack/unpack/dequant round-trip、conversion accounting、verified-level enum | E09-06 |
| `model_core.py` | Tiny/representative ModelArtifact identity、WorkloadSpec freeze + field-effectiveness probe、A/B identity、capture plan、correctness layers L0-L4、custom-op coverage accounting、prefill/decode timing boundaries、Amdahl prediction | E09-07 |
| `profiling.py` | Metric manifest、profiler run modes、overhead measurement、unified trace key schema、timeline validation（negative duration/overlap/missing parent/unmapped ratio）、module→op→kernel many-to-many mapping、phase split、shape-aware census、hotspot selection、byte/FLOP models、empirical ceilings、roofline points、bottleneck taxonomy、ranked recommendations | E09-08 |
| `comparison.py` | Three tiers（A/B/C）、identity matrix、pairability validator（PAIRABLE/NON_COMPARABLE with field-level reason）、normalized metrics（speedup/bandwidth_efficiency/compute_efficiency/energy_per_token/memory_per_active_token）、maturity rubric、capacity limit、claim audit、S12 export schema | E09-09 |
| `faults.py` | Error taxonomy（12 classes）with vendor→normalized mapping、retryable/fallback_allowed、async correlation、watchdog、resource snapshots（before/after, retained vs leak）、fault matrix、injection safety scope、recovery invariants、post-fault probe chain、MTTR、runbook minimal diagnostic chain、blind drill record | E09-10 |
| `mapping.py` | CUDA<->Ascend concept mapping table（warp/block/shared/DRAM ↔ core/UB/GM/tiling）、with "not a numerical equivalence" guard | E09-09 |
| `telemetry.py` | C6/C7 projection + coverage audit（reuse runtime.telemetry pattern but ascend-specific fields as extensions） | All experiments |
| `specs.py` | Strict loader for configs/ascend/*.yaml + audit_documents | All experiments |
| `experiment.py` | Prerequisites（Ascend device present, CANN toolchain, E09-01 manifest verdict, frozen OperatorSpec, frozen test vectors, profiler capability, second hardware for E09-09）、Preregistration、ExperimentRecord、EvidenceManifest（with S09 extras）、RunDirectory（layout per details README §2）、verdict refusal | All experiments |
| `interface_map.py` | 10 experiments × 28 steps = 280 steps → interfaces、import verification（resolve_interfaces()） | All experiments |

---

## 4. 关键数据结构

### 4.1 TilingData ABI（72 bytes）

```c
typedef struct HqsbTilingData {
    uint32_t schema_version;      // 0: current
    uint32_t tiling_key;          // 0-4 (single/multi/tail/row_aligned/row_tail)
    uint32_t dtype_tag;           // 0-5 (fp32/fp16/bf16/int8/int32/int64)
    uint32_t block_dim;           // # cores
    uint32_t rows;                // logical rows
    uint32_t hidden;              // hidden size
    uint32_t tile_elems;          // elements per tile
    uint32_t loop_count;          // max tiles per core
    uint32_t tail_elems;          // valid elements in last tile
    uint32_t buffer_count;        // 1 or 2 (double buffering)
    uint32_t base_rows;           // rows per core (base partition)
    uint32_t extra_rows;          // cores [0,extra) get base+1 row
    uint64_t total_elements;      // rows * hidden
    uint64_t workspace_bytes;     // requested workspace
    uint32_t reserved_0;          // padding
    uint32_t reserved_1;          // padding
} HqsbTilingData;
```

**验证点**：
- `sizeof(HqsbTilingData) == 72`
- Little-endian, no implicit padding
- Device parses same bytes as host

### 4.2 CapabilityEntry

```python
@dataclass(frozen=True)
class CapabilityEntry:
    key: str                      # e.g., "dtype.fp16.compute"
    domain: str                   # dtype/layout_format/custom_ascend_c/...
    state: str                    # SUPPORTED_VERIFIED/SUPPORTED_UNVERIFIED/UNSUPPORTED/UNKNOWN
    scope: str                    # IN_SCOPE/NON_GOAL
    official_source: str          # URL to CANN doc
    probe_id: str                 # probe that verified it
    probe_status: str             # pass/fail/UNAVAILABLE
    reason: str                   # explanation for UNKNOWN/UNSUPPORTED
    evidence: Tuple[str, ...]     # supporting artifacts
    required_by: Tuple[str, ...]  # experiments requiring this
    attributes: Mapping[str, Any] # e.g., {"dtype": "fp16"}
```

**四态语义**：
- `SUPPORTED_VERIFIED` = official evidence + local probe passed
- `SUPPORTED_UNVERIFIED` = official evidence exists but no probe yet
- `UNSUPPORTED` = explicit no from doc or probe
- `UNKNOWN` = insufficient evidence — must NOT be treated as yes

### 4.3 ErrorTaxonomy

```python
ERROR_CLASSES = (
    "ENV_MISMATCH",           # driver/firmware/CANN/framework wrong combo
    "UNSUPPORTED",            # dtype/layout/op/shape not supported
    "ARTIFACT_INCOMPATIBLE",  # SoC/ABI/Tiling/QuantArtifact mismatch
    "COMPILE_ERROR",          # source/template/target/UB compilation failed
    "LOAD_ERROR",             # artifact load/registration failure
    "INVALID_ARGUMENT",       # host tiling overflow/block_dim/workspace error
    "WORKSPACE_ERROR",        # allocation/failure/OOM
    "DEVICE_OOM",             # device memory exhaustion
    "DEVICE_EXECUTION_ERROR", # async kernel failure
    "TIMEOUT",                # watchdog triggered
    "PROFILER_ERROR",         # metric unavailable/file parse fail
    "INTERNAL_ERROR",         # unknown/vendor unclassified
)
```

---

## 5. 接口契约

### 5.1 C4 Backend Interface

```python
class AscendBackend(Backend):
    def capabilities(self) -> CapabilityTable: ...
    def load(self, artifact: ModelArtifact) -> None: ...
    def warmup(self, workload: WorkloadSpec) -> None: ...
    def generate(self, workload: WorkloadSpec, inputs: TensorDict) -> TensorDict: ...
    def health(self) -> bool: ...
    def metrics(self) -> Dict[str, Any]: ...
    def close(self) -> None: ...
```

**Contract**：
- `capabilities()` returns `CapabilityTable` with four-state entries
- `generate()` records `requested_backend` / `actual_backend` / `fallback_reason`
- `health()` checks device/query/probe status
- `close()` releases all resources (context/stream/workspace)

### 5.2 C6 ResultSchema Extensions

```python
@dataclass(frozen=True)
class AscendResultFields(BenchmarkResult):
    compatibility_manifest_hash: str
    soc_version: str
    tiling_schema_version: int
    block_dim: int
    msprof_metric_set: str
    actual_kernel: str  # e.g., "ascend_rmsnorm_v1"
    ub_bytes_used: int
    gm_read_bytes: int
    gm_write_bytes: int
    queue_stats: Dict[str, int]  # enque/deque/blocked/enqueued
    fault_events: List[Dict[str, Any]]
```

### 5.3 C7 TraceSchema Extensions

```python
@dataclass(frozen=True)
class AscendTraceEvent(TraceEvent):
    tiling_key: int
    core_index: Optional[int]
    tile_index: Optional[int]
    ub_allocation_tags: List[str]
    stream_id: int
    event_id: int
    ai_core_utilization: Optional[float]
    pipe_utilization: Optional[float]
    memory_bandwidth_gbps: Optional[float]
```

---

## 6. 实现细节

### 6.1 Host Tiling Function

```python
def compute_tiling(request: TilingRequest, limits: TilingLimits) -> Tuple[TilingDecision, None] | Tuple[None, TilingRejection]:
    """Validate and compute one tiling, return structured rejection if invalid."""
    # 1. Layout check
    if request.layout not in SUPPORTED_LAYOUTS: reject REJECT_LAYOUT
    
    # 2. Dimension checks
    if request.rows < 0 or request.hidden < 0: reject REJECT_NEGATIVE_DIM
    try: total = checked_mul(request.rows, request.hidden)
    except ArithmeticOverflow: reject REJECT_OVERFLOW
    
    # 3. Block dim check
    if not (1 <= request.block_dim <= limits.max_block_dim): reject REJECT_BLOCK_DIM
    
    # 4. Tile alignment
    if request.tile_elems % limits.alignment_elems: reject REJECT_ALIGNMENT
    
    # 5. Workspace check
    if request.workspace_bytes > limits.max_workspace_bytes: reject REJECT_WORKSPACE
    
    # 6. Compute geometry
    base_rows, extra_rows = partition_rows(request.rows, request.block_dim)
    per_core_elems = checked_mul(base_rows + (1 if extra_rows else 0), request.hidden)
    loop_units = request.hidden if request.rowwise else per_core_elems
    loop_count, tail = tile_loop(loop_units, request.tile_elems)
    
    # 7. UB budget
    ub = estimate_ub(request.ub_model, ...)
    if not ub.fits(limits.ub_budget): reject REJECT_UB
    
    # 8. Select tiling key
    aligned = hidden % alignment_elems == 0
    tiling_key = select_tiling_key(rows, hidden, block_dim, alignment_elems, rowwise)
    
    # 9. Build TilingData
    tiling = TilingData(schema_version=1, tiling_key=..., block_dim=..., ..., total_elements=total, workspace_bytes=...)
    
    # 10. Serialize & size check
    payload = tiling.to_bytes()
    if len(payload) > MAX_TILING_SERIALIZED_BYTES: reject REJECT_SERIAL_SIZE
    
    return TilingDecision(tiling=tiling, ub_estimate=ub, limits=limits, rationale=[...]), None
```

### 6.2 CPU Test Double Simulation

```python
def launch(kernel: str, tiling_payload: bytes, *, inputs: Mapping[str, Sequence[float]], gamma: Optional[Sequence[float]], eps: float) -> SimResult:
    """Run one simulated launch."""
    # 1. Parse tiling bytes
    tiling = TilingData.from_bytes(tiling_payload)
    
    # 2. Allocate GM buffers
    gm = SimulatedGlobalMemory(capacity_bytes=1<<28)
    for name, values in inputs.items():
        buffers[name] = gm.allocate(name, list(values), element_bytes=dtype_bytes(dtype))
    if gamma is not None:
        buffers["gamma"] = gm.allocate("gamma", list(gamma), element_bytes=dtype_bytes(dtype))
    buffers["y"] = gm.allocate("y", [SENTINEL]*output_length, element_bytes=dtype_bytes(dtype))
    
    # 3. Run kernel
    try:
        if kernel == OPERATOR_ADD: _run_add(...)
        elif kernel == OPERATOR_ROW_REDUCE_SUM: _run_row_reduce_sum(...)
        elif kernel == OPERATOR_RMSNORM: _run_rmsnorm(...)
        output = gm.snapshot("y")
    except DeviceMemoryFault as exc:
        faults.append({"kind": "device_memory_fault", "message": str(exc)})
        output = []
    
    # 4. Audit integrity
    integrity = _audit_integrity(gm, integrity_snapshot or {}, faults)
    
    # 5. Return result
    return SimResult(kernel=kernel, tiling=tiling, output=output, statuses={...}, accounting={...}, faults=tuple(faults))
```

### 6.3 Negative Injection Safety

```python
NEGATIVE_INJECTIONS = (
    NegativeInjection(
        probe_id="negative_version_mismatch",
        kind="framework_version_mismatch",
        isolation_method="isolated_virtualenv",  # MUST be in SAFE_ISOLATION_METHODS
        expected_error_class="ENV_MISMATCH",
        fixture_uri="configs/ascend/compatibility_spec.yaml#negative_fixtures.framework_version",
        forbidden_action="must not modify the host driver, firmware or shared CANN install",
        modifies_shared_stack=False,  # MUST be False
    ),
    ...
)

SAFE_ISOLATION_METHODS = ("container", "isolated_virtualenv", "read_only_fixture", "mock_metadata")
```

**Safety invariant**：Any injection with `modifies_shared_stack=True` raises ValueError at module load time.

---

## 7. 构建与部署

### 7.1 Kernel Compilation

```bash
# On an Ascend machine with CANN installed:
scripts/ascend/build_ascend_ops.sh -B build/ascend-sm87 --soc=910b

# This script:
# 1. Checks for ccec/bisheng compiler
# 2. Fails with clear message if absent
# 3. Invokes cmake to compile each operator kernel
# 4. Produces .o/.so artifacts in build directory
```

**On this host**：The script fails with:
```
ERROR: CANN compiler (ccec or bisheng-c++) not found.
To use this script on an Ascend machine:
  1. Install CANN toolkit matching your SoC version
  2. Ensure the compiler is on PATH or set ASCEND_TOOLKIT_ROOT
  3. Re-run this script
```

This is honest — the kernels remain as source artifacts only.

### 7.2 Environment Collection

```bash
scripts/ascend/collect_ascend_env.sh > docs/evidence/ascend_env.json
```

Collects hardware/stack/framework info into a structure compatible with `CompatibilityManifest`.

---

## 8. 成熟度标签

| 组件 | 成熟度 | 说明 |
|---|---|---|
| Tiling ABI | M2 | CPU tests verify serialization, geometry, UB budget |
| CPU Oracles | M2 | Pure Python FP64 implementations, tested against tolerance gates |
| Test Vectors | M2 | Deterministic generators, guard buffer verification |
| Sim Device | M2 | Verified logic for tail handling, multi-core partition, guard detection |
| Capability Table | M2 | Four-state logic, merge_probe_updates, contradiction audit |
| Compatibility Manifest | M2 | Schema validation, redaction, field-level diff |
| Probes | M2 | Fixture executor tests parsing/status mapping/cleanup accounting |
| Backend Adapter | M1 | Source exists, needs real Ascend environment for M3 |
| Framework Binding | M1 | Contract defined, needs torch_npu/aclnn for M3 |
| Quant Format | M1 | Validator implemented, needs INT8/INT4 kernels for M3 |
| Model Core | M1 | Identity framework ready, needs Qwen model loading for M3 |
| Profiling | M1 | Schema defined, needs msprof for M3 |
| Comparison | M1 | Tier A/B/C framework ready, needs CUDA data for M7 |
| Faults | M1 | Taxonomy/invariants defined, needs real fault injection for M3 |

---

## 9. 风险与反模式

| 反模式 | 应对 |
|---|---|
| 只测整除 tile 的漂亮 shape | E09-02 step 5 强制 tail_minus_one/core_tail/fewer_than_cores cases |
| 把 host enqueue 时间当成 kernel latency | sim_device.statuses["sync_complete"] = "simulated" 明确区分 |
| 为 Ascend 单独放宽 tolerance | ToleranceSpec 无 backend 字段，结构上禁止 |
| 静默 fallback 到 PyTorch | framework.fallback_policy 要求 explicit reason code |
| profiler 指标填 0 代替 UNAVAILABLE | profiling.profiler_observed = {"status": "UNAVAILABLE", "reason": "..."} |
| 不同模型/精度直接比 TPS | comparison.pairability_validator 隔离不可比行 |
| 故障注入破坏共享驱动 | NEGATIVE_INJECTIONS.modifies_shared_stack=False 强制安全 |

---

## 10. 后续工作

1. **填充 interface_map.py**：280 步对照表需逐实验编写
2. **补充测试**：单元测试、属性测试、依赖边界测试
3. **配置模板**：configs/ascend/*.yaml 按实际需求填写
4. **真实环境运行**：待 Ascend 设备可用后立即执行 E09-01~E09-10
5. **同步台账**：docs/evidence_ledger.md、docs/project_status.md、README.md

---

*生成时间：2026-09-18*  
*版本：draft v0.1*
