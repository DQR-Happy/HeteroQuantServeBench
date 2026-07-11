# HeteroQuantServeBench

Heterogeneous quantization, kernel optimization, serving, and benchmarking
platform for NVIDIA CUDA and Huawei Ascend C/CANN backends.

HQSB 统一管理模型制品、工作负载、算子、量化、运行时、服务、通信、编译与跨硬件
benchmark，使每一次优化都能从 Kernel 追踪到模型、服务和硬件收益。

> **Profile before optimize；correctness before performance；evidence before claim。**

## Current stage

**S14（训推协同与前沿扩展）—— 接口/代码层就位；实验层 BLOCKED。**

S14 已交付 E14-01~E14-05、E14-F1~F4、E14-06~08 共 **12 项实验、480 个实验步骤**的能力接口
（`hqsb/experimental/` **21 模块 / 17,328 行**：canonical 身份与 seed bundle、S14 词汇表与四个状态机、
§7 七个统一证据对象、§22 运行目录布局与执行安全策略、依赖与 feature flag 边界（E14-01）、
分布式训练状态与 checkpoint（E14-02）、转换 DAG 与**七层训推一致性门**（E14-03）、
SFT/DPO/GRPO **手算 oracle** 与 policy staleness（E14-04）、前沿 ADR 与预注册（E14-05）、
四个条件 P0 分支（F1 speculative/MTP、F2 MoE、F3 long-context、F4 sparsity）、
三个可选迁移（E14-06 多模态、E14-07 Agent、E14-08 端侧）、C6/C7 投影、
冻结词汇表审计、三重门脚手架与 480 步接口对照表）
+ `scripts/experimental/` 驱动与 2 个生成器 + `configs/experimental/` 8 份冻结词汇表（**无测量值**），
由 103 个 S14 专用测试（单元 89 + 属性 14，含 12 实验参数化；全量 **3221 passed, 4 deselected**）、
依赖边界 gate（规则 R1–**R16**；`violations=0 cycles=0`）、
接口解析（**480/480 步 / 589 接口 / 993 引用 / 0 失败**）、8/8 配置逐字段审计与 16 模块 CPU smoke 自检证明。
**未执行任何正式实验、未产出任何训练/转换/rollout/前沿/多模态/Agent/端侧结论数字。**

实验层为 `BLOCKED`：必需前置 `upstream_verdicts`、`experimental_environment`、`holdout_isolation`
未满足，且本机未解析到分布式 launcher / 第二设备 / profiler。驱动入口默认拒绝产结论：

```bash
python3 scripts/experimental/run_e14.py --list                    # 12 实验 / 480 步
python3 scripts/experimental/run_e14.py --prerequisites --probe    # 前置门 → satisfied=false
python3 scripts/experimental/run_e14.py --interface-map            # 480 步 / 589 接口对照表
python3 scripts/experimental/run_e14.py --spec-audit               # 8/8 配置逐字段审计
python3 scripts/experimental/run_e14.py --smoke                    # CPU 自检 smoke（claim_allowed=false）
python3 scripts/experimental/run_e14.py --experiment E14-02        # 仍拒绝（status=BLOCKED, conclusion=false）
```

详见 `docs/reports/S14_开发报告.md`（含「实验步骤 → 代码接口」对照表）、
`docs/reports/S14_阶段验收报告.md`（区分代码层验收与实验层 BLOCKED）；
480 步逐条对照见 `docs/reports/S14_interface_map_generated.md`。

S13（生产化/云原生/可靠性，418 步）同样为代码/接口层就位、实验层 `BLOCKED`
（`hqsb/infra/` 19 模块 + `scripts/infra/run_e13.py` + `configs/infra/` 13 份配置；
见 `docs/reports/S13_开发报告.md`、`docs/reports/S13_production_architecture.md`）。

S12（跨硬件评估与统一 Benchmark，360 步）同样为代码/接口层就位、实验层 `BLOCKED`
（`hqsb/evaluation/` 21 模块 + `scripts/evaluation/run_e12.py` + `configs/evaluation/` 12 份配置；
见 `docs/reports/S12_开发报告.md`、`docs/reports/S12_cross_hardware_design.md`）。
S11（AI 编译器，320 步）、S10（分布式，300 步）、S08（ServeFabric，264 步）、
S07（Runtime，200 步）、S06（图优化）、S05（量化）、S04（Triton/CUTLASS/Kernel DSL）
已完成代码/接口层交付；
S09（Ascend C/CANN）为**部分交付**（缺 `experiment/interface_map/driver/configs/tests`，
见 `docs/reports/S10_开发报告.md` §2.2）；S04 跨架构补验（2026-09-18，RTX 3090 / sm_86）见
`docs/reports/S04_阶段验收报告.md` §8。
下一阶段顺序：**S08 服务契约 + S12 容量/质量基线补齐 → 隔离集群/注册表/扫描器/遥测后端 →
S13 实验执行（E13-01…E13-11，含故障授权与 runbook 演练）**。

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
| CPU 单元测试 | Implemented | `tests/` | `pytest -m "not hardware and not e2e and not performance" -q`（3118 passed，2026-09-19，S13 口径；S12 口径 2922） |
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
| 分布式拓扑身份/漂移/降级边政策（含 preflight 负向 fixture） | Implemented | `hqsb/distributed/{topology,ranks,placement,probes}.py` | `tests/unit/distributed/test_topology_placement.py` |
| collective 语义/oracle/带宽公式登记表/α–β 拟合/拐点 | Implemented | `hqsb/distributed/collectives.py` | `test_collectives.py`、`tests/property/test_distributed_invariants.py::TestBandwidthAlgebra` |
| communicator 状态机/序列/预检/超时/watchdog/清理 | Implemented | `hqsb/distributed/sequence.py` | `test_sequence_faults.py::TestStateMachine`、`::TestSequenceAndPreflight`、`::TestTimeoutsAndWatchdog` |
| 分布式故障 oracle/恢复级别/资源闭环/19 类故障矩阵 | Implemented | `hqsb/distributed/faults.py` | `test_sequence_faults.py::TestFaultOracle`、`::TestSafetyAndRecovery`、`::TestResourceClosure` |
| TP 计划/census/能力矩阵/shard round-trip/direct-load 审计 | Implemented | `hqsb/distributed/parallel_plan.py` | `test_parallel_ledger.py::TestCensus`、`::TestTpCapability`、`::TestShardsAndPlan`、`::TestDirectLoadAndCases` |
| 通信账本（expected↔observed + 原因码）与内存分桶对账 | Implemented | `hqsb/distributed/ledger.py` | `test_parallel_ledger.py::TestLedger` |
| strong/weak/capacity scaling（无 T1 拒绝强扩展）、时间分解、pairability | Implemented | `hqsb/distributed/scaling.py` | `test_scaling_overlap.py::TestScalingPrereg`、`::TestResourceMatrixAndBaseline`、`::TestDecompositionAndPairability` |
| overlap 区间集合/依赖 DAG/三组 schedule/chunk/ABBA/因果裁决 | Implemented | `hqsb/distributed/overlap.py` | `test_scaling_overlap.py::TestIntervalAlgebra`、`::TestOverlapSchedule` |
| PP/CP/SP P1 门禁、rubric、stage balance、bubble、adopt/reject | Implemented（`NOT_RUN_NOT_CLAIMED`） | `hqsb/distributed/boundary.py` | `test_boundary_moe.py::TestBoundaryActivation`、`::TestBoundarySelection`、`::TestBoundaryPlans`、`::TestBoundaryVerdict` |
| MoE RouteArtifact/dispatch+combine oracle/skew/imbalance/placement holdout | Implemented | `hqsb/distributed/moe.py` | `test_boundary_moe.py::TestMoeRouting`、`::TestMoeOracle`、`::TestMoePlacementAndVerdict` |
| 多 rank trace/时钟校准/事件配对/根因分类/放大指标/runbook | Implemented | `hqsb/distributed/traces.py` | `test_traces_telemetry.py::TestTraceEventsAndClocks`、`::TestPairingAndSkew`、`::TestAttribution`、`::TestInjectionPlans` |
| S10 C6/C7 投影 + 统一数据表 schema + 覆盖审计 | Implemented | `hqsb/distributed/telemetry.py` | `test_traces_telemetry.py::TestTelemetryProjection` |
| S10 冻结配置（12 份严格加载 + 逐字段契约审计） | Implemented | `configs/distributed/*.yaml`、`hqsb/distributed/specs.py` | `test_s10_experiment_scaffolding.py::TestSpecs` |
| S10 实验驱动入口（默认不产结论） | Implemented | `scripts/distributed/run_e10.py` | `--smoke`、`--execute` 拒绝（退出提示 BLOCKED） |
| S10 实验步骤→接口对照表（300 步 / 381 接口 / 548 引用） | Implemented | `hqsb/distributed/interface_map.py` | `test_s10_experiment_scaffolding.py::TestInterfaceMap` |
| 分布式依赖边界（R7/R8 + 无模块级重依赖 + 子进程探针） | Implemented | `hqsb/distributed/__init__.py`、`tests/unit/distributed/test_distributed_import_boundaries.py` | `pytest tests/unit/distributed/test_distributed_import_boundaries.py -q`（6 passed） |
| S09 Ascend 交付完整性 | **部分交付**（缺 `experiment/interface_map/driver/configs/tests`） | `hqsb/ascend/`（7 模块） | `docs/reports/S10_开发报告.md` §2.2 |
| S10 实验执行 | **BLOCKED**（8/9 硬前置缺失：S07 P0、S08 trace、双加速器、封存 manifest、backend 身份、单卡 reference、model/workload、协议树指纹） | `docs/stage_experiments/details/S10/` | `run_e10.py --experiment E10-01 --prerequisites` |
| S11 多级 artifact 身份 + lineage + 三合一身份 | Implemented | `hqsb/compiler/identity.py` | `tests/unit/compiler/test_ir_identity.py::TestArtifactIdentity`、`::TestLineage` |
| S11 HQSB canonical/targeted IR + 14 项 verifier + round-trip | Implemented | `hqsb/compiler/ir.py` | `test_ir_identity.py::TestVerifier`、`::TestSerializationAndDiff` |
| S11 统一数据契约（状态机/失败分类/成本键/§21 记录） | Implemented | `hqsb/compiler/records.py` | `tests/property/test_compiler_invariants.py`、`test_s11_experiment_scaffolding.py::TestTelemetryProjection` |
| S11 capture/break/metadata/coverage census（E11-01） | Implemented | `hqsb/compiler/capture.py` | `tests/unit/compiler/test_capture_guards.py::TestBreakAudit`、`::TestCoverage` |
| S11 guard/variant/重编译安全（E11-05） | Implemented | `hqsb/compiler/guards.py` | `test_capture_guards.py::TestVariantRegistry`、`::TestCompileAccounting` |
| S11 语义 pattern 重写 + near-miss 拒绝 + 幂等（E11-02） | Implemented（FP=0 语料） | `hqsb/compiler/{pattern_library,rewrite}.py` | `tests/unit/compiler/test_rewrite.py::TestCorpusAndPipeline` |
| S11 lowering registry + 选择链 + dispatch 证据（E11-03） | Implemented | `hqsb/compiler/{targets,lowering,backend}.py` | `tests/unit/compiler/test_lowering_dispatch.py::TestSelection`、`::TestDispatch` |
| S11 IR→codegen→binary→counter 归因（E11-04） | Implemented | `hqsb/compiler/codegen.py` | `tests/unit/compiler/test_backend_codegen.py::TestMechanisms` |
| S11 autotune 空间/预算/holdout（E11-06） | Implemented | `hqsb/compiler/autotune.py` | `tests/unit/compiler/test_autotune.py` |
| S11 cost model regret + 安全回退（E11-07） | Implemented | `hqsb/compiler/costmodel.py` | `tests/unit/compiler/test_costmodel.py` |
| S11 编译制品 cache（key/事务/失效/损坏）（E11-08） | Implemented | `hqsb/compiler/cache.py` | `tests/unit/compiler/test_cache.py` |
| S11 TVM/MLIR 可迁移 lowering 接口（E11-09） | Implemented（接口层） | `hqsb/compiler/portable.py` | `tests/unit/compiler/test_portable_aigate.py::TestLegalization` |
| S11 AI 候选零信任门链 G0–G10（E11-10，P1） | Implemented（未激活） | `hqsb/compiler/aigate.py` | `test_portable_aigate.py::TestGateChain` |
| S11 冻结配置（12 份严格加载 + 逐字段契约审计） | Implemented | `configs/compiler/*.yaml`、`hqsb/compiler/specs.py` | `test_s11_experiment_scaffolding.py::TestInterfaceMapAndSpecs::test_specs_load_and_audit` |
| S11 实验驱动入口（默认不产结论） | Implemented | `scripts/compiler/run_e11.py` | `--smoke`、`--prerequisites`、`--execute` 拒绝 |
| S11 实验步骤→接口对照表（320 步 / 417 接口 / 693 引用） | Implemented | `hqsb/compiler/interface_map.py` | `test_s11_experiment_scaffolding.py::TestInterfaceMapAndSpecs::test_interface_map_resolves_every_reference` |
| S11 实验执行 | **BLOCKED**（6 条必需前置中 3 条缺失：S03/S04 硬件证据、S06 模型级 pattern correctness、冻结编译器环境指纹；另 2 项 advisory） | `docs/stage_experiments/details/S11/` | `run_e11.py --prerequisites` |
| S14 训推身份与词汇表（canonical digest / seed bundle / 4 状态机 / 40 表 schema） | Implemented | `hqsb/experimental/{identity,records}.py` | `tests/unit/experimental/test_experimental_foundations.py` |
| S14 §7 七个统一证据对象 + 跨实验不变量 | Implemented | `hqsb/experimental/contracts.py` | `test_experimental_foundations.py::test_new_contracts_validate_and_reject_a_conclusion` |
| S14 §22 运行目录布局 + 执行安全策略（协议树只读） | Implemented | `hqsb/experimental/campaign.py` | `tests/property/test_experimental_invariants.py::test_run_layout_is_always_under_artifacts_and_never_in_the_protocol_tree` |
| S14 E14-01~E14-08 + E14-F1~F4 实验接口（12 项 × 40 步） | Implemented（`SOURCE_INTEGRATED`） | `hqsb/experimental/{dependencies,training,parity,posttraining,frontier,speculative,moe,long_context,sparsity,multimodal,agent,edge}.py` | `tests/unit/experimental/test_e14_interfaces.py`（12 参数化 × 4 类断言） |
| S14 三重门脚手架（默认拒绝产结论）+ EvidenceManifest | Implemented | `hqsb/experimental/experiment.py` | `test_experimental_foundations.py::test_triple_gate_refuses_a_conclusion` |
| S14 冻结配置（8 份生成式词汇表 + 逐字段审计） | Implemented | `configs/experimental/*.yaml`、`hqsb/experimental/specs.py` | `scripts/experimental/gen_experimental_specs.py --check`、`run_e14.py --spec-audit` |
| S14 实验驱动入口（默认不产结论） | Implemented | `scripts/experimental/run_e14.py` | `--smoke`、`--experiment E14-02` → `status=BLOCKED, conclusion=false` |
| S14 实验步骤→接口对照表（480 步 / 589 接口 / 993 引用） | Implemented | `hqsb/experimental/interface_map.py` | `tests/unit/experimental/test_e14_interfaces.py::test_interface_map_reports_full_coverage` |
| S14 依赖边界 R15/R16（experimental 只依赖 core、不 import ops、无模块级重依赖） | Implemented | `scripts/audit/import_dependency_gate.py`（rules=1.8.0） | `tests/unit/experimental/test_experimental_import_boundaries.py`（含子进程导入纯度探针） |
| S14 实验执行 | **BLOCKED**（必需前置 `upstream_verdicts`/`experimental_environment`/`holdout_isolation` 未满足；本机无分布式 launcher / 第二设备 / profiler；12 项实验无 run、无 raw、无结论数字） | `docs/stage_experiments/details/S14/` | `run_e14.py --prerequisites --json`；`docs/reports/S14_阶段验收报告.md` §6.2 |
| S13 ReleaseBundle/OCI digest DAG 身份与可重建四级（E13-01） | Implemented | `hqsb/infra/identity.py` | `tests/unit/infra/test_infra_foundations.py::TestIdentity` |
| S13 三套状态机 + §24 状态传播 + 93 张表 schema | Implemented | `hqsb/infra/records.py` | `test_infra_foundations.py::TestRecords`、`tests/property/test_infra_invariants.py` |
| S13 §23 十四条交叉约束 V01–V14（含脱敏） | Implemented | `hqsb/infra/contracts.py` | `test_infra_foundations.py::TestContracts` |
| S13 §21 制品目录（52）+ campaign manifest + §20.2 安全策略与 target 授权 | Implemented | `hqsb/infra/campaign.py` | `test_infra_foundations.py::TestCampaign` |
| S13 供应链 gate：分层镜像/SBOM 完整性/漏洞例外/秘密与模型命中/许可证/provenance/非 root（E13-01） | Implemented | `hqsb/infra/supply_chain.py` | `tests/unit/infra/test_e13_infra_experiments_a.py::TestSupplyChain` |
| S13 clean bootstrap、部署状态机、readiness 九条件、cold/warm、失败清理（E13-02） | Implemented | `hqsb/infra/deployment.py` | `test_e13_infra_experiments_a.py::TestDeployment` |
| S13 placement/label 信任/拓扑/隔离与负向 placement（E13-03） | Implemented | `hqsb/infra/scheduling.py` | `test_e13_infra_experiments_a.py::TestScheduling` |
| S13 制品 cache/原子提交/lease/GC/切换/回滚（E13-04） | Implemented | `hqsb/infra/artifacts.py` | `test_e13_infra_experiments_a.py::TestArtifacts` |
| S13 三类 probe/终止预算/drain/rolling/forced kill（E13-05） | Implemented | `hqsb/infra/lifecycle.py` | `test_e13_infra_experiments_a.py::TestLifecycle` |
| S13 内存账本/KV 预算/admission/residual/margin（E13-06） | Implemented | `hqsb/infra/capacity.py` | `tests/unit/infra/test_e13_infra_experiments_b.py::TestCapacity` |
| S13 autoscaling 控制环路/指标时效/稳定性/成本（E13-07） | Implemented | `hqsb/infra/autoscaling.py` | `test_e13_infra_experiments_b.py::TestAutoscaling` |
| S13 语义约定/基数/采样/告警/RCA/遥测缺失与脱敏（E13-08） | Implemented | `hqsb/infra/observability.py` | `test_e13_infra_experiments_b.py::TestObservability` |
| S13 故障合同/ground truth/MTTD-MTTM-MTTR/降级/恢复不变量（E13-09） | Implemented | `hqsb/infra/faults.py` | `test_e13_infra_experiments_b.py::TestFaults` |
| S13 canary 状态机/G0–G8 硬门/序贯决策/回滚闭环/override 审计（E13-10） | Implemented | `hqsb/infra/canary.py` | `test_e13_infra_experiments_b.py::TestCanary` |
| S13 多租户威胁模型/RBAC/配额竞态/滥用/噪声邻居/审计/12 不变量（E13-11，P1） | Implemented | `hqsb/infra/security.py` | `test_e13_infra_experiments_b.py::TestSecurity` |
| S13 §22 七投影 + 表覆盖审计 + 证据等级 | Implemented | `hqsb/infra/telemetry.py` | `test_infra_foundations.py::TestTelemetry` |
| S13 冻结配置（13 份严格加载 + 逐字段契约审计 + 生成器一致） | Implemented | `configs/infra/*.yaml`、`hqsb/infra/specs.py`、`scripts/infra/gen_infra_specs.py` | `test_e13_scaffolding_assets.py::TestSpecs` |
| S13 实验驱动入口（三重门默认拒绝产结论） | Implemented | `scripts/infra/run_e13.py`、`hqsb/infra/experiment.py` | `test_e13_scaffolding_assets.py::TestExperimentGates` |
| S13 实验步骤→接口对照表（418 步 / 343 接口 / 638 引用） | Implemented | `hqsb/infra/interface_map.py`、`scripts/infra/gen_interface_map.py` | `test_e13_scaffolding_assets.py::TestInterfaceMap` |
| S13 部署与可观测资产（容器/Helm/告警/面板/CI/runbook 模板） | Implemented（模板未验证） | `infra/**` | `test_e13_scaffolding_assets.py::TestInfraAssets` |
| S13 依赖边界 R13/R14 + 只依赖 core | Implemented | `scripts/audit/import_dependency_gate.py`（rules 1.7.0） | `tests/unit/infra/test_infra_import_boundaries.py` |
| S13 实验执行 | **BLOCKED**（必需前置 `s08_service_contract`、`s12_capacity_and_quality_baseline` 缺失；本机无集群/注册表/扫描器/遥测后端） | `docs/stage_experiments/details/S13/` | `scripts/infra/run_e13.py --prerequisites` |
| S12 三种身份 hash + 逻辑 URI + 版本化 canonicalization | Implemented | `hqsb/evaluation/identity.py` | `tests/unit/evaluation/test_lineage.py` |
| S12 Comparison Contract + 四态裁决 + 非法 join 防护（E12-01） | Implemented | `hqsb/evaluation/{contracts,comparability,candidates}.py` | `tests/unit/evaluation/test_comparability.py` |
| S12 capability 四级证据 + 失效规则（E12-02） | Implemented | `hqsb/evaluation/{capability,platform}.py` | `tests/unit/evaluation/test_eval_capability.py` |
| S12 四层统一重放 + 11 条交叉校验（E12-03） | Implemented | `hqsb/evaluation/benchmark.py` | `tests/unit/evaluation/test_benchmark.py` |
| S12 重复性/漂移/排除账本（E12-04） | Implemented | `hqsb/evaluation/repeatability.py` | `tests/unit/evaluation/test_repeatability.py` |
| S12 Roofline/Amdahl 预测与残差（E12-05） | Implemented | `hqsb/evaluation/roofline.py` | `tests/unit/evaluation/test_eval_roofline.py` |
| S12 能量窗口/积分/能效（E12-06） | Implemented | `hqsb/evaluation/energy.py` | `tests/unit/evaluation/test_energy.py` |
| S12 云/自建 TCO 与成本敏感性（E12-07，无内置价格） | Implemented | `hqsb/evaluation/cost.py` | `tests/unit/evaluation/test_cost.py` |
| S12 业务画像 Pareto + 决策回归（E12-08） | Implemented | `hqsb/evaluation/pareto.py` | `tests/unit/evaluation/test_pareto.py` |
| S12 软件成熟度 rubric（E12-09） | Implemented | `hqsb/evaluation/maturity.py` | `tests/unit/evaluation/test_maturity.py` |
| S12 端到端 lineage + 重生成（E12-10） | Implemented | `hqsb/evaluation/lineage.py` | `tests/unit/evaluation/test_lineage.py` |
| S12 冻结配置（12 份）与驱动入口（默认不产结论） | Implemented | `configs/evaluation/`、`scripts/evaluation/run_e12.py` | `tests/unit/evaluation/test_e12_scaffolding.py` |
| S12 实验步骤→接口对照表（360 步 / 415 接口 / 727 引用） | Implemented | `hqsb/evaluation/interface_map.py` | `test_e12_scaffolding.py::TestInterfaceMapAndSpecs` |
| S12 实验执行 | **BLOCKED**（4 条必需前置中 3 条缺失：S03–S11 上游证据链、多硬件覆盖、冻结评估环境指纹；另 3 项 advisory） | `docs/stage_experiments/details/S12/` | `run_e12.py --prerequisites` |
| 真实多卡 collective / TP / scaling / overlap / 故障注入运行 | **未执行**（本机仅 1×RTX 3090；CPU loopback `claim_allowed()=False`） | `hqsb/distributed/collectives.py::LoopbackCollectiveExecutor` | `run_e10.py --smoke`（标注 smoke） |
| KernelLab（CUTLASS/Ascend C） | Planned | `ops/ascend/`（CUTLASS 待网络恢复） | S05/S09 |
| 真实 Runtime 适配器（vLLM/SGLang/TensorRT-LLM/llama.cpp） | Planned（本机均未安装，探测实测 `NOT_INSTALLED`） | `hqsb/runtime/adapter.py`（契约/注册表/探测已就位） | S07 实验执行 |
| BenchLab（跨硬件统一 benchmark） | **接口/代码层就位（S12）** | `hqsb/evaluation/` | 见 S12 各行 |

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
