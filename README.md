# HeteroQuantServeBench

Heterogeneous quantization, kernel optimization, serving, and benchmarking
platform for NVIDIA CUDA and Huawei Ascend C/CANN backends.

HQSB 统一管理模型制品、工作负载、算子、量化、运行时、服务、通信、编译与跨硬件
benchmark，使每一次优化都能从 Kernel 追踪到模型、服务和硬件收益。

> **Profile before optimize；correctness before performance；evidence before claim。**

## Current stage

**S08（ServeFabric 与性能治理）—— 接口/代码层就位；实验层 BLOCKED。**

S08 已交付 E08-01~E08-11 全部 **264 个实验步骤**的能力接口（`hqsb/serving/`
26 模块：协议兼容子集与错误目录、SSE 线缆级一致性、网关与请求状态机、传输与
背压、SLO 预注册与计数漏斗、可重放到达过程与 loadgen、公平性与队列策略、
准入/熔断/路由/cache-aware 路由、故障注入、端到端可观测性、服务级严格 A/B、
S08 实验脚手架）+ `scripts/serving/run_e08.py` 驱动 + `configs/serving/` 12 份
冻结配置，由 131 个新增测试（全量 1894 passed）、依赖边界 gate（规则 R1–R6）、
接口解析（264 步 / 418 接口 / 791 引用）、契约审计与 smoke 自检证明。
**未执行任何正式实验、未产出任何容量/延迟/吞吐/goodput/命中率/能耗结论**。

实验层为 `BLOCKED`：S08 的 8 条前置中 7 条未满足（S07 P0 verdict、双后端注册、
冻结请求夹具、冻结 SLO、拓扑记录、loadgen 校准、协议树内的环境指纹）。驱动入口
默认拒绝产结论：

```bash
python3 scripts/serving/run_e08.py --list                    # 11 实验 / 264 步
python3 scripts/serving/run_e08.py --experiment E08-01 --prerequisites  # 前置门 → BLOCKED
python3 scripts/serving/run_e08.py --interface-map           # 264 步 / 418 接口对照表
python3 scripts/serving/run_e08.py --smoke                   # 无模型夹具 smoke（标注 smoke）
python3 scripts/serving/run_e08.py --experiment E08-01 --execute          # 仍拒绝（前置缺失）
```

详见 `docs/reports/S08_开发报告.md`（含「实验步骤 → 代码接口」对照表）、
`docs/reports/S08_阶段验收报告.md`（区分代码层验收与实验层 BLOCKED）与
`docs/reports/S08_serving_architecture.md`（Design 制品）。

S07（推理 Runtime 内核，200 步 / 1763 测试）与 S06（框架集成与图优化）、
S05（量化）、S04（Triton/CUTLASS/Kernel DSL）已完成代码/接口层交付；S04 跨架构
补验（2026-09-18，RTX 3090 / sm_86）见 `docs/reports/S04_阶段验收报告.md` §8。
下一阶段顺序：**S07 实验执行（P0）→ S04.5（真实模型算子回接）→ S05/S06 实验执行 →
S08 实验执行**。

阶段路线图见 [`docs/architecture/顶层架构.md`](docs/architecture/顶层架构.md) 与
[`docs/stages/`](docs/stages/)。模块边界与依赖规则见
[`docs/architecture/module_ownership.md`](docs/architecture/module_ownership.md)。

## Quick start

目标环境：NVIDIA Jetson Orin Nano Super 8GB（sm_87）、JetPack 6 / L4T R36.4.3、
CUDA 12.6、Python 3.10、PyTorch 2.5（NVIDIA 构建）。

### 0. 安装（可选依赖组）

```bash
pip install -e .              # 仅核心：pydantic + PyYAML（CPU 可跑全部测试）
pip install -e ".[dev]"       # 附加 pytest/ruff/mypy
pip install -e ".[benchmark]" # 附加 torch/transformers/modelscope（S02）
```

### 1. CPU 单元测试（无需 GPU / 模型权重）

```bash
python3 -m pytest -q                 # 全量
python3 -m pytest -m unit -q         # 纯单元测试
python3 -m pytest -m property -q     # 属性/不变量测试
# CI/开发机口径（排除硬件/E2E/性能用例）：
python3 -m pytest -m "not hardware and not e2e and not performance" -q   # 1894 passed（S08 口径）
```

### 2. CUDA 算子库（RMSNorm 多版本 + fused + correctness + benchmark）

```bash
cmake -S . -B build/jetson-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build/jetson-release --parallel

# correctness（CTest 集成）
ctest --test-dir build/jetson-release --output-on-failure

# benchmark：RMSNorm V0/V1/V2 扫参
build/jetson-release/bin/hqsb_rmsnorm_bench \
  --rows 512 --hidden 2048 --dtype fp32 --variant all

# benchmark：fused residual+rmsnorm
build/jetson-release/bin/hqsb_fused_residual_rmsnorm_bench \
  --rows 512 --hidden 2048 --dtype fp32 --variant all
```

算子库公共 API 见 `ops/cuda/rmsnorm/include/hqsb/rmsnorm.h`，V0/V1/V2 性能数据
与退化分析见 `docs/reports/S03_benchmark_report.md`。

### 3. Qwen3-1.7B model-core smoke

```bash
# 下载模型（与 benchmark 严格分离）
python3 scripts/models/download_qwen3_modelscope.py

# 校验本地快照 SHA256（artifact integrity gate）
python3 scripts/models/verify_qwen3_hashes.py

# 最小加载 + 生成 smoke
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python3 scripts/models/smoke_qwen3.py
```

完整六 workload model-core baseline：

```bash
python3 benchmarks/scripts/run_jetson_baseline.py
```

## Reproducibility

所有 benchmark 结果绑定：Git commit、模型与 tokenizer 的 SHA256 manifest、配置、
环境（nvpmodel/clock/温度）、PyTorch/CUDA/Transformers/ModelScope 版本与 seed。
原始 sample 保存于 `reports/` 并以 artifact 索引引用，summary 不可反推 raw。

## Status matrix

图例：**Implemented** = 源码存在且被测试或运行验证；**Verified** = 有运行证据；
**Experimental** = 存在但未闭环；**Planned** = 仅有规划。

| 能力 | 状态 | 位置 | 证据 |
|---|---|---|---|
| CUDA device query | Verified | `ops/cuda/device_query/` | `reports/jetson/20260808_010256/device_query.txt` |
| CUDA RMSNorm V0（shared reduction） | Verified | `ops/cuda/rmsnorm/` | `reports/jetson/20260808_010256/rmsnorm_runs.txt` |
| Qwen3-1.7B model loader（local-only） | Verified | `hqsb/models/loader.py` | `reports/dev/llm/` 各 run |
| 模型 SHA256 manifest 校验 | Implemented | `hqsb/models/manifest.py`、`scripts/models/verify_qwen3_hashes.py` | `tests/unit/test_manifest.py` |
| model-core benchmark 引擎 | Verified | `hqsb/benchmark/model_core.py` | `reports/dev/llm/20260812_094129/` |
| tegrastats monitor/parser | Implemented | `hqsb/benchmark/resource_monitor.py`、`tegrastats_parser.py` | `tests/unit/test_tegrastats_parser.py` |
| 六 workload baseline orchestrator | Verified | `benchmarks/scripts/run_jetson_baseline.py` | `reports/dev/llm/20260812_094129/summary.csv` |
| golden 数值回归基线（4 份） | Implemented | `benchmarks/workloads/golden/` | 已迁移到 C6 schema（S01） |
| C1–C7 版本化 Contract | Implemented | `hqsb/core/contracts/` | `tests/unit/core/test_contracts.py` |
| 统一配置加载 + hash | Implemented | `hqsb/core/config/loader.py` | `tests/unit/core/test_config.py` |
| 插件注册表 Registry | Implemented | `hqsb/core/registry/registry.py` | `tests/unit/core/test_registry.py` |
| 错误分类 + exit code | Implemented | `hqsb/core/errors.py` | `tests/unit/core/test_errors.py` |
| Schema 版本化 + legacy 迁移 | Implemented | `hqsb/core/schema/` | `tests/unit/core/test_migration.py` |
| Dummy backend（C4 参考实现） | Implemented | `hqsb/backends/dummy.py` | `tests/unit/core/test_dummy_backend.py` |
| backend-interface benchmark engine | Implemented | `hqsb/benchmark/engine.py` | `tests/unit/core/test_dummy_backend.py` |
| PyTorchBackend（FP16 Reference Runtime） | Verified | `hqsb/backends/pytorch.py` | `reports/dev/llm/s02_smoke_tiny.json`（decode 9.28 tok/s） |
| Roofline/Amdahl 分析 | Implemented | `hqsb/benchmark/roofline.py` | `tests/unit/core/test_roofline.py` |
| golden/determinism/首错位对比 | Implemented | `hqsb/benchmark/correctness.py` | `tests/unit/core/test_correctness.py` |
| YAML workload 单一事实源 | Implemented | `hqsb/benchmark/workload_config.py` | `tests/unit/core/test_workload_config.py` |
| KV cache/内存核算 | Implemented | `hqsb/benchmark/memory.py` | `tests/unit/core/test_memory.py` |
| PyTorch Profiler 采集 | Verified | `hqsb/benchmark/profiling.py` | `reports/dev/profiler/s02/hotspot_summary.json` |
| Jetson 实验协议 | Implemented | `hqsb/hardware/jetson.py` | `tests/unit/core/test_jetson.py` |
| RMSNorm 算子库（V0/V1/V2 + dispatcher） | Verified | `ops/cuda/rmsnorm/` | `ctest`（33 checks）；V2 +56%~142% |
| Fused residual+rmsnorm 算子 | Verified | `ops/cuda/fused_residual_rmsnorm/` | `ctest`（15 checks） |
| CUDA 测试框架 + 数值指标 | Implemented | `ops/cuda/common/{test_util,test_metrics}.h` | CTest 集成 |
| 后端能力检测（Triton 实测编译 probe） | Implemented | `ops/capability.py` | `tests/unit/ops/test_capability.py` |
| CUDA shared lib ctypes 绑定 | Implemented | `ops/cuda_bridge.py` | `tests/unit/ops/test_cuda_bridge.py` |
| CUDA 库自报编译架构（跨架构 dispatch 依据） | Verified | `ops/cuda/rmsnorm/src/rmsnorm_c_api.cu` | sm_86 返回 `(8,6)`、sm_87 返回 `(8,7)`；`docs/reports/S04_阶段验收报告.md` §8 |
| 统一 dispatcher（CUDA/Triton/cuBLAS/CUTLASS fallback） | Verified | `ops/dispatcher.py` | `tests/unit/ops/test_dispatcher.py`（含双向跨架构用例） |
| CUTLASS GEMM（按 arch 自适应 tile + 正确性门禁） | Verified | `ops/cuda/cutlass_gemm/` | CTest `cutlass_gemm_correctness`；E04-01 |
| Triton RMSNorm（reference + autotune） | Verified | `ops/triton/rmsnorm.py` | sm_86/sm_87 均通过正确性门禁 |
| Triton GEMM（reference + autotune） | Verified | `ops/triton/gemm.py` | autotune 随 shape 选不同 tile（E04-01） |
| S04 跨架构 tile/autotune 迁移实验 | Verified | `scripts/audit/run_e04_01_cross_arch_tile_transfer.py` | `docs/stage_experiments/S04/E04-01/raw/` |
| 模型制品门禁（客户端缓存元数据排除） | Verified | `hqsb/models/manifest.py` | `verify_qwen3_hashes.py` → 13/14 PASS |
| CPU 单元测试 | Implemented | `tests/` | `pytest -m "not hardware and not e2e and not performance" -q`（1894 passed，2026-09-18，S08 口径） |
| QuantLab 量化语义/RTN/golden/packing/制品 | Implemented | `hqsb/quant/{spec,rounding,rtn,golden,packing,artifact}.py` | `tests/unit/quant/`（936 passed 子集） |
| QuantLab 兼容/故障注入/校准/统计 | Implemented | `hqsb/quant/{compat,faults,calibration,stats,fixtures}.py` | `tests/unit/quant/test_artifact_compat.py`、`test_calibration_stats.py` |
| QuantLab 模型级评估（coverage/apply/quality/execution/oracle/model_eval） | Implemented | `hqsb/quant/` | `test_coverage_apply_quality.py`、`test_execution_model_eval.py` |
| QuantLab 工业方法 adapter（GPTQ/AWQ/SmoothQuant） | Implemented | `hqsb/quant/adapters/` | `test_adapters_units_policy.py` |
| QuantLab 敏感性与混合精度 policy | Implemented | `hqsb/quant/{units,sensitivity,policy}.py` | `test_adapters_units_policy.py` |
| 激活量化 / KV 缓存量化（P1） | Implemented | `hqsb/quant/{activation,kv}.py` | `test_activation_kv.py` |
| Pareto 与部署决策层（五道门/场景/推荐） | Implemented | `hqsb/quant/decision.py` | `test_decision.py` |
| 低比特 fused-dequant GEMM kernel（Triton W4/W8） | Verified（`development`/`smoke`，sm_86） | `ops/quant/w4a16_triton.py`、`executors.py` | `tests/unit/ops/test_quant_kernels.py`（对 oracle 分层容差） |
| 实验驱动入口（默认不产结论） | Implemented | `scripts/quant/run_e05.py`、`hqsb/quant/experiment.py` | `test_experiment_interface_map.py`；`--mode execute` 退出码 7 |
| 实验步骤→接口对照表（190 步 / 219 接口） | Implemented | `hqsb/quant/interface_map.py` | `test_experiment_interface_map.py::TestInterfaceMap` |
| QuantLab 实验执行 | **BLOCKED**（S04.5 M4 前置缺失） | `docs/stage_experiments/details/S05/` | `run_e05.py --mode status` |
| 算子 schema / dispatcher 边界（schema 契约、注册冲突、redispatch、capability、fallback） | Implemented | `hqsb/integration/{specs,dispatch}.py` | `tests/unit/integration/test_specs_dispatch.py` |
| Meta/FakeTensor 元数据契约与 symbolic shape | Implemented | `hqsb/integration/meta.py` | `tests/unit/integration/test_meta_graph.py` |
| 图 IR（capture mode/IR level、结构哈希、graph diff） | Implemented | `hqsb/integration/graph.py` | `test_meta_graph.py::TestGraphIR` |
| pattern 重写（语义谓词、字段级拒绝、副本重写、四种覆盖率） | Implemented | `hqsb/integration/patterns.py` | `tests/unit/integration/test_patterns.py` |
| guard / graph break / recompile / fallback 五类分离与 storm 阈值 | Implemented | `hqsb/integration/guards.py` | `test_guards_cache.py::TestCompileLedger`、`TestStorm` |
| compile identity / cache 校验 / 失效矩阵 / 损坏注入 / break-even | Implemented | `hqsb/integration/cache.py` | `test_guards_cache.py` |
| lowering registry 与收益归因（allocation / Amdahl / 消融） | Implemented | `hqsb/integration/lowering.py` | `test_lowering_telemetry.py` |
| CUDA Graph 契约与 claim 门（P1） | Implemented（`NOT_CLAIMED`） | `hqsb/integration/cuda_graph.py`、`configs/integration/graph_spec.yaml` | `test_cuda_graph_lifecycle.py::TestClaimGate` |
| 错误分类 / 事务化执行 / ABI load 前判定 | Implemented | `hqsb/integration/{taxonomy,abi}.py` | `test_adapter_taxonomy.py` |
| 生命周期 / 资源斜率 / 弱引用存活 / stream 审计 | Implemented | `hqsb/integration/lifecycle.py` | `test_cuda_graph_lifecycle.py::TestLeakStatistics` |
| 跨模型/后端复用与反硬编码扫描 | Implemented | `hqsb/integration/adapter.py` | `test_adapter_taxonomy.py::TestHardcodeScanner` |
| 四级 differential（路径矩阵、预注册容差、首次发散定位） | Implemented | `hqsb/integration/differential.py` | `tests/unit/integration/test_differential.py` |
| C6/C7 投影 + 字段/事件覆盖审计 | Implemented | `hqsb/integration/telemetry.py` | `test_lowering_telemetry.py::TestC6Projection` |
| 实验驱动入口（默认不产结论） | Implemented | `scripts/integration/run_e06.py`、`hqsb/integration/experiment.py` | `--mode execute` 退出码 7；`test_s06_experiment_scaffolding.py` |
| 实验步骤→接口对照表（218 步 / 381 接口） | Implemented | `hqsb/integration/interface_map.py` | `test_s06_experiment_scaffolding.py::TestInterfaceMap` |
| S06 实验执行 | **BLOCKED**（S04.5 M4 / S05 P0 前置缺失） | `docs/stage_experiments/details/S06/` | `run_e06.py --mode status` |
| Runtime 请求语义与 capability 协商（六态、silent 降级拒绝） | Implemented | `hqsb/runtime/request.py` | `tests/unit/runtime/test_request_capability.py` |
| Runtime adapter 契约（七类操作/状态机/诚实探测/注册表） | Implemented | `hqsb/runtime/adapter.py` | `test_adapter_parity.py::TestDummyAdapter`、`TestProbe` |
| 语义 parity oracle（greedy/多 seed 分布/streaming/参数生效矩阵） | Implemented | `hqsb/runtime/parity.py` | `test_adapter_parity.py::TestParityOracles` |
| 请求状态机 / C7 span / iteration 账本 / token 守恒 | Implemented | `hqsb/runtime/trace.py` | `tests/unit/runtime/test_trace_ledger.py` |
| Paged KV 几何/生命周期/碎片分类/容量/有界 OOM | Implemented | `hqsb/runtime/kv.py` | `tests/unit/runtime/test_kv_blocks.py` |
| Static/continuous/chunked 调度（确定性模拟 + 账本） | Implemented | `hqsb/runtime/scheduler.py` | `tests/unit/runtime/test_scheduler_batching.py` |
| Prefix cache（key 绑定/碰撞政策/refcount/净收益模型） | Implemented | `hqsb/runtime/prefix_cache.py` | `tests/unit/runtime/test_prefix_cache.py` |
| Graph 桶与 attention 能力矩阵（含 2×2 factorial 与 claim 门） | Implemented（`NOT_RUN`） | `hqsb/runtime/graph_route.py` | `tests/unit/runtime/test_graph_attention.py` |
| Speculative/MTP 契约（精确 acceptance/residual、rollback、P1 门） | Implemented（`NOT_RUN`） | `hqsb/runtime/spec_decode.py` | `tests/unit/runtime/test_spec_decode.py` |
| 失败矩阵与资源恢复（28 例 + 有界 OOM + 长稳判据） | Implemented | `hqsb/runtime/failure.py` | `tests/unit/runtime/test_failure_matrix.py` |
| 公平比较（tier/两表分离/统一重算/token 分母审计/Pareto） | Implemented | `hqsb/runtime/comparison.py` | `test_comparison_policyab.py::TestMetricRecompute` |
| 策略 A/B（选题门禁/唯一变量/ABBA/因果链/裁决） | Implemented | `hqsb/runtime/policy_ab.py` | `test_comparison_policyab.py::TestCausalChainAndDecision` |
| Runtime 时间与 token 口径（三分母 + 配对 CI） | Implemented | `hqsb/runtime/metrics.py` | `tests/unit/runtime/test_metrics_tokens.py` |
| S07 C6/C7 投影 + 字段/事件覆盖审计 | Implemented | `hqsb/runtime/telemetry.py` | `test_s07_experiment_scaffolding.py::TestTelemetryProjection` |
| Runtime 冻结配置（8 份严格加载 + 契约审计） | Implemented | `configs/runtime/*.yaml`、`hqsb/runtime/specs.py` | `test_s07_experiment_scaffolding.py::TestConfigs` |
| 手册 §4 统一实验记录（结论必须带 raw evidence） | Implemented | `hqsb/runtime/experiment.py`（`ExperimentRecord`） | `test_s07_experiment_scaffolding.py::TestExperimentRecord` |
| C2/C6/C7 对齐审计（runtime 指标 ↔ workload 契约） | Implemented | `hqsb/runtime/telemetry.py`（`c2_alignment`） | `test_s07_experiment_scaffolding.py::TestTelemetryProjection` |
| 实验驱动入口（默认不产结论） | Implemented | `scripts/runtime/run_e07.py`、`hqsb/runtime/experiment.py` | `--mode execute` 退出码 7；`test_s07_experiment_scaffolding.py` |
| 实验步骤→接口对照表（200 步 / 308 接口） | Implemented | `hqsb/runtime/interface_map.py` | `test_s07_experiment_scaffolding.py::TestInterfaceMap` |
| S07 实验执行 | **BLOCKED**（S04.5 M4 / S05 P0 / S06 P0 前置缺失） | `docs/stage_experiments/details/S07/` | `run_e07.py --mode status` |
| ServeFabric 协议平面（ProtocolProfile/ErrorCatalog/SSE 一致性） | Implemented | `hqsb/serving/{protocol,sse}.py` | `tests/unit/serving/test_protocol_sse.py` |
| ServeFabric 网关平面（请求状态机/SSE 交付/取消/drain） | Implemented | `hqsb/serving/{gateway,transport,transport_http,timing,pipeline}.py` | `test_gateway_lifecycle.py` |
| ServeFabric 策略平面（SLO/到达/loadgen/公平/准入/路由/熔断/cache） | Implemented | `hqsb/serving/{slo,arrival,loadgen,fairness,policies,admission,router,circuit,cache_routing,clients}.py` | `test_policy_planes.py`、`test_backend_evidence_planes.py` |
| ServeFabric 证据平面（观测/服务 A/B/实验脚手架/接口对照） | Implemented | `hqsb/serving/{observability,telemetry,service_ab,experiment,specs,interface_map,faults}.py` | `test_s08_experiment_scaffolding.py`、`test_serving_import_boundaries.py` |
| ServeFabric 冻结配置（12 份严格加载 + 契约审计） | Implemented | `configs/serving/*.yaml`、`hqsb/serving/specs.py` | `test_s08_experiment_scaffolding.py::TestSpecs` |
| ServeFabric 实验驱动入口（默认不产结论） | Implemented | `scripts/serving/run_e08.py` | `run_e08.py --smoke`；`--execute` 拒绝 |
| ServeFabric 实验步骤→接口对照表（264 步 / 418 接口） | Implemented | `hqsb/serving/interface_map.py` | `test_s08_experiment_scaffolding.py::TestInterfaceMap` |
| S08 实验执行 | **BLOCKED**（S07 P0 等 7 条前置缺失） | `docs/stage_experiments/details/S08/` | `run_e08.py --experiment E08-01 --prerequisites` |
| KernelLab（CUTLASS/Ascend C） | Planned | `ops/ascend/`（CUTLASS 待网络恢复） | S05/S09 |
| 真实 Runtime 适配器（vLLM/SGLang/TensorRT-LLM/llama.cpp） | Planned（本机均未安装，探测实测 `NOT_INSTALLED`） | `hqsb/runtime/adapter.py`（契约/注册表/探测已就位） | S07 实验执行 |
| BenchLab（跨硬件统一 benchmark） | Planned | `benchmarks/` | — |

## Planned components

- **QuantLab**：RTN、GPTQ、AWQ、SmoothQuant 与论文方法复现
- **KernelLab**：CUDA、Triton 与 Ascend C 算子
- **Runtime adapters**：TensorRT、llama.cpp、vLLM 与 Ascend 运行时
- **ServeFabric**：OpenAI-compatible 异构推理网关（S08 接口/代码层已就位，实验层 BLOCKED）
- **BenchLab**：latency、throughput、memory、accuracy、energy 报告

## Hardware

- **NVIDIA Jetson Orin Nano Super 8GB**（sm_87）—— 边缘设备**正式实验**机：
  统一内存行为、`tegrastats`、`nvpmodel`/`jetson_clocks`、功耗/温度/热降频、
  OOM 与容量边界、Jetson 正式性能报告
- **NVIDIA RTX 3090**（sm_86, x86_64）—— **开发机**：CUDA 构建、算子开发、
  单元/集成测试、探索性 profiling 与 benchmark。结果标注为
  `development` / `smoke` / `cross-architecture validation` / `exploratory benchmark`，
  不写入 `reports/jetson/**`，不用于计算跨硬件 speedup
- Orange Pi AI Pro 20T
- 按需 NVIDIA datacenter/desktop GPU 实例

机器角色与授权边界见 `AGENTS.md`（本机文件）。通用 CUDA 开发机验收入口：

```bash
./scripts/bench/run_cuda_dev_baseline.sh --platform rtx3090 --cuda-arch 86
```

## Repository integrity

- 模型权重、`*.safetensors`/`*.gguf`/`*.onnx`/`*.engine` 一律被 `.gitignore` 排除；
- SSH 私钥（`id_rsa*`、`id_ed25519*`）、`.env`、`credentials*`、`secrets/` 禁止入库；
- 默认配置不含机器专属绝对路径（模型路径经 `~` 展开）。
