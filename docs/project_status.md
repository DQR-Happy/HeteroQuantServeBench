# HQSB 项目现状报告（Project Status）

> 生成时间：2026-08-17（S13 章节与头部阶段声明同步于 2026-09-19）
> 跨架构补验：2026-09-18（RTX 3090 / sm_86）
> 当前工作树基线：`b83ebde`（`docs: add S09 CUDA-Ascend mapping, backend report and troubleshooting runbook`，S10 开始前工作树干净）
> 历史基线 Commit：`9b403aa`（`chore: stop tracking the stage-experiment tree`，2026-06-02）；`3c2453e`（S06 追加）
> 当前阶段：S13（生产化、云原生与可靠性）—— **接口/代码层就位；实验层 BLOCKED**（2 条必需前置缺失，见 §19；S12 章节见 §18）
> 下一阶段：补齐 S08 服务契约与 S12 容量/质量基线前置 → 在隔离集群/注册表/扫描器/遥测后端上执行 E13-01…E13-11（含故障授权与 runbook 演练）
> **最新追加**：S14（训推协同与前沿扩展，§20）与 **S15（发布、开源与求职证据，§21）** 均已交付**接口/代码层**，实验层 `BLOCKED`（2026-09-19）
> （历史路径：S07 实验执行（P0）→ S04.5（真实模型算子回接）→ S05/S06/S08 实验执行 → S10 实验执行（≥2 加速器））
>
> 说明：本报告正文生成于 `4dda6f8`，其后仓库推进到 `9b403aa` 并在其上追加
> S05/S06/S07/S08 四轮交付；S09 为部分交付（见 §16.4 漂移登记）；历史条目保留不改，
> 新增章节见 §12/§13/§14/§15/§16。

本报告是仓库当前事实的 Source of Truth。任何“已完成 / 已测量”的声明都必须能
定位到代码、测试或运行证据；无法定位的声明一律降级为 historical/planned。
详见 [`evidence_ledger.md`](evidence_ledger.md)。

---

## 1. 项目是什么

HeteroQuantServeBench（HQSB）是一个面向 **GPU（NVIDIA CUDA）/ NPU（Huawei Ascend
C/CANN）/ 边缘与云端** 环境的 LLM 推理优化实验与工程平台。它统一管理模型制品、
工作负载、算子、量化、运行时、服务、通信、编译与跨硬件 benchmark，使每一次优化
都能从 Kernel 追踪到模型、服务和硬件收益。

核心原则：**Profile before optimize；correctness before performance；evidence
before claim。**

顶层架构见 [`architecture/顶层架构.md`](architecture/顶层架构.md)。

---

## 2. 当前阶段判定

| 判定项 | 结论 |
|---|---|
| 当前所处阶段 | **S08（ServeFabric 与性能治理）—— 接口/代码层就位；实验层 BLOCKED** |
| 判定依据 | `hqsb/serving/` 26 模块 + `scripts/serving/run_e08.py` + `configs/serving/` 12 份冻结配置；E08-01~E08-11 全部 264 步的接口解析通过（418 接口）、131 个新增测试、依赖边界 gate R1–R6 0 违规；`run_e08.py --experiment E08-01 --prerequisites` 实测 7/8 前置 MISS → 全部实验 `BLOCKED`（§15） |
| 前序阶段 | S00 → S01 → S02 → S03 → S04 已完成验收；S04.5/S05/S06/S07 为「接口层就位、实验层 BLOCKED」 |
| 待解除阻塞 | S07 P0 verdict、双后端注册、冻结请求夹具、冻结 SLO、拓扑记录、loadgen 校准、协议树内环境指纹（§15.4） |

S04 实测环境能力（**Triton 3.7.1 / CUTLASS 4.7.0 / TileLang 0.1.13 三个 DSL 在
sm_87 全部可用**、cuBLAS 可用），实现 Triton RMSNorm/GEMM、CUTLASS GEMM 对照、
CUDA shared lib 的 ctypes 绑定、统一 dispatcher/capability，并用统一 benchmark
证明 CUDA/Triton/cuBLAS/CUTLASS 的性能权衡。详见 §8（S04 补齐内容）。

---

## 3. 完整 Inventory

### 3.1 `hqsb/`（Python 包）

| 路径 | 内容 | 状态 |
|---|---|---|
| `__init__.py` | 包元信息（v0.1.0） | Implemented |
| `models/loader.py` | `load_qwen3`：local-only 加载、dtype/attention、OOM fallback、资源释放 | Verified |
| `models/manifest.py` | SHA256 manifest 解析 + `verify_model_files` | **本次新增**，Implemented |
| `models/__init__.py` | 导出 `load_qwen3` | Implemented |
| `benchmark/metrics.py` | percentile / latency_summary / numerical_diff_summary | Implemented |
| `benchmark/model_core.py` | model-core 三阶段 benchmark（prefill/first-token/decode） | Verified |
| `benchmark/workload.py` | 固定 token 长度 workload 生成 | Implemented |
| `benchmark/resource_monitor.py` | `TegrastatsMonitor` 后台采集 | Implemented |
| `benchmark/tegrastats_parser.py` | tegrastats 行解析 + 功率/能量积分 | Implemented |
| `benchmark/cli.py` | `positive_int` / `non_negative_int` argparse 校验 | Implemented |
| `benchmark/engine.py` | backend-interface benchmark engine → BenchmarkResult | **S01 新增**，Implemented |
| `benchmark/roofline.py` | Roofline/Amdahl + 热点分类 | **S02 新增**，Implemented |
| `benchmark/correctness.py` | golden/determinism/首错位定位 | **S02 新增**，Implemented |
| `benchmark/workload_config.py` | YAML workload 单一事实源 | **S02 新增**，Implemented |
| `benchmark/memory.py` | KV cache/权重/RSS/swap 核算 | **S02 新增**，Implemented |
| `benchmark/profiling.py` | PyTorch Profiler operator 表提取 | **S02 新增**，Implemented |
| `benchmark/__init__.py` | 导出 benchmark API | Implemented |
| `core/errors.py` | 错误分类 + exit code（1–9） | **S01 新增**，Implemented |
| `core/ids.py` | run/trace/span ID 生成 | **S01 新增**，Implemented |
| `core/logging.py` | JSON lines 结构化日志 + trace context | **S01 新增**，Implemented |
| `core/contracts/` | C1–C7 版本化 schema + Backend ABC | **S01 新增**，Implemented |
| `core/schema/` | SchemaVersion + 迁移框架 + legacy 迁移 | **S01 新增**，Implemented |
| `core/config/` | 分层配置加载 + hash | **S01 新增**，Implemented |
| `core/registry/` | 插件注册表 + RegistryHub | **S01 新增**，Implemented |
| `backends/dummy.py` | DummyBackend 参考实现（C4） | **S01 新增**，Implemented |
| `backends/pytorch.py` | PyTorchBackend（C4，FP16 Qwen3 reference） | **S02 新增**，Implemented |
| `hardware/jetson.py` | Jetson 实验协议（温度/冷却/电源模式） | **S02 新增**，Implemented |
| `quant/` | 量化语义/RTN/golden/packing/artifact/compat/faults/stats/calibration/coverage/apply/quality/execution/oracle/model_eval/adapters/units/sensitivity/policy/activation/kv/decision/experiment/interface_map/config_io + fixtures（28 模块） | **S05 新增**，Implemented（test-verified，见 `evidence_ledger.md` §11） |
| `integration/` | 框架集成与图优化：specs / dispatch / meta / graph / patterns / guards / cache / lowering / cuda_graph / taxonomy / abi / lifecycle / adapter / differential / telemetry / policies / experiment / interface_map（18 模块） | **S06 新增**，Implemented（test-verified，见 `evidence_ledger.md` §12） |
| `runtime/` | 推理 Runtime：request / adapter / parity / trace / kv / scheduler / prefix_cache / graph_route / spec_decode / failure / comparison / policy_ab / metrics / telemetry / specs / experiment / interface_map（18 模块） | **S07 新增**，Implemented（test-verified，见 `evidence_ledger.md` §13） |
| `serving/` | 空目录（S08 规划占位） | Planned |

### 3.2 `ops/`（算子）

| 路径 | 内容 | 状态 |
|---|---|---|
| `cuda/common/cuda_check.cuh` | 公共 CUDA 错误检查宏 | Implemented |
| `cuda/common/test_util.h` | 轻量断言测试框架 | **S03 新增**，Implemented |
| `cuda/common/test_metrics.h` | 5 项数值对比指标 | **S03 新增**，Implemented |
| `cuda/device_query/` | 设备信息查询（.cu + CMakeLists） | Verified |
| `cuda/rmsnorm/` | **RMSNorm 算子库**：V0 shared / V1 warp shuffle / V2 vectorized + dispatcher + reference + test + bench | **S03 重构**，Verified（CTest 通过） |
| `cuda/fused_residual_rmsnorm/` | **第二热点算子**：fused residual+rmsnorm（V0/V1）+ test + bench | **S03 新增**，Verified（CTest 通过） |
| `cuda/rmsnorm/src/rmsnorm_c_api.cu` | `extern "C"` 稳定 C ABI + shared lib（ctypes 绑定） | **S04 新增**，Verified |
| `cuda/cutlass_gemm/` | CUTLASS FP16 GEMM 对照（默认 tensor-op 配置） | **S04 追加**，Verified |
| `capability.py` | 统一能力检测（Triton/CUTLASS/TileLang 实测 probe + cuBLAS + CUDA lib） | **S04 新增**，Implemented |
| `_tilelang_probe.py` | TileLang 最小 kernel 复现（无 future annotations） | **S04 追加**，Verified |
| `cuda_bridge.py` | ctypes 绑定 CUDA shared lib | **S04 新增**，Implemented |
| `dispatcher.py` | 统一 dispatcher（capability→arch→shape→fallback） | **S04 新增**，Implemented |
| `triton/rmsnorm.py` | Triton RMSNorm（reference + autotune） | **S04 新增**，Verified |
| `triton/gemm.py` | Triton GEMM（reference + autotune，`input_precision=ieee`） | **S04 新增**，Verified |
| `ascend/` | 空目录（.gitkeep） | Planned |
| `third_party/` | 外部依赖目录（cutlass gitignore，README 记录获取方式） | **S04 追加** |

### 3.3 `benchmarks/`

| 路径 | 内容 | 状态 |
|---|---|---|
| `schemas/golden_reference_schema.json` | golden 数值回归 JSON Schema | Implemented |
| `scripts/run_model_core.py` | 单 workload model-core runner | Verified |
| `scripts/run_jetson_baseline.py` | 六 workload 编排 | Verified |
| `scripts/generate_golden.py` | golden 生成器 | Implemented |
| `scripts/summarize_baseline.py` | CSV 汇总 | Implemented |
| `workloads/golden/` | 4 份 golden（isl32/128/512/2048 × osl32） | legacy（未进回归门禁） |
| `raw/` `normalized/` | 空（.gitkeep） | Planned |

### 3.4 `configs/`

| 路径 | 内容 | 状态 |
|---|---|---|
| `models/qwen3_1_7b.yaml` | Qwen3-1.7B 模型配置 | Implemented |
| `benchmarks/jetson_qwen3_fp16.yaml` | Jetson FP16 benchmark 配置 | Implemented |
| `environment/jetson_python_lock.txt` | Python 全量锁（含 ROS/Jupyter 等非必要包） | Implemented |
| `environment/jetson_runtime.txt` | 关键运行版本清单 | Implemented |
| `operators/rmsnorm_v0.json` | RMSNorm C3 OperatorSpec（V0） | **S01 新增**，Implemented |
| `operators/rmsnorm_v1.json` | RMSNorm C3 OperatorSpec（V1） | **S03 新增**，Implemented |
| `operators/rmsnorm_v2.json` | RMSNorm C3 OperatorSpec（V2） | **S03 新增**，Implemented |
| `operators/fused_residual_rmsnorm.json` | fused residual+rmsnorm C3 OperatorSpec | **S03 新增**，Implemented |
| `backends/` | 空（.gitkeep） | Planned |
| `quantization/` | RTN W8/W4/非对称、scheme 矩阵、校准、kernel、激活、KV、决策 spec + policy 示例（10 份，均经 `config_io.load_any` 严格校验） | **S05 新增**，Implemented |
| `integration/` | 算子契约、pattern 声明、编译策略、cache 规格、graph 规格、资源规格、复用规格（7 份，均经 `policies.load_directory` 严格校验 + 与代码逐字段审计） | **S06 新增**，Implemented |
| `runtime/` | request / kv / scheduler / prefix / graph+attention / spec-decode / failure / comparison 冻结配置（8 份，均经 `specs.RuntimeSpecs.load` 严格校验 + 契约漂移审计） | **S07 新增**，Implemented |

### 3.5 `scripts/`

| 路径 | 内容 | 状态 |
|---|---|---|
| `models/download_qwen3_modelscope.py` | 模型下载 | Implemented |
| `models/verify_qwen3.py` | 模型架构/配置校验 | Implemented |
| `models/verify_qwen3_hashes.py` | SHA256 快照校验 CLI（退出码 0/1/2） | **本次新增**，Implemented |
| `models/smoke_qwen3.py` | Qwen 加载+生成 smoke | Verified |
| `models/dump_model_manifest.py` | 生成模型 manifest JSON | Implemented |
| `migrate_legacy.py` | legacy golden/result → 新 schema 迁移 CLI | **S01 新增**，Implemented |
| `check_docs.py` | 文档相对链接完整性检查 | **S01 新增**，Implemented |
| `bench/run_s02_baseline.py` | S02 contract-native baseline orchestrator | **S02 新增**，Implemented |
| `bench/profile_model.py` | PyTorch Profiler 采集 runner | **S02 新增**，Implemented |
| `bench/analyze_hotspots.py` | Roofline/Amdahl hotspot 决策分析 | **S02 新增**，Implemented |
| `bench/nsys_profile.sh` | Nsight Systems 采集 runner | **S02 新增**，Implemented |
| `bench/ncu_profile.sh` | Nsight Compute 采集 runner | **S02 新增**，Implemented |
| `bench/run_jetson_baseline.sh` | Jetson CUDA baseline 一键脚本 | Implemented |
| `integration/run_e06.py` | S06 实验驱动（status/preregister/interface-map/self-check/execute，默认不产结论） | **S06 新增**，Implemented |
| `runtime/run_e07.py` | S07 实验驱动（status/preregister/interface-map/self-check/execute，默认不产结论） | **S07 新增**，Implemented |
| `env/collect_jetson_env.sh` | 环境采集脚本 | Implemented |
| `common/git_commit.sh` | 本地 git 历史改写辅助（**已 gitignore，勿入库**） | 本地工具，见 §8 |

### 3.6 `tests/`

| 路径 | 内容 | 状态 |
|---|---|---|
| `conftest.py` | sys.path 注入，保证 `import hqsb` 可用 | **本次新增** |
| `unit/test_metrics.py` | percentile/summary/数值误差 | **本次新增** |
| `unit/test_tegrastats_parser.py` | 行解析 + 能量积分 | **本次新增** |
| `unit/test_manifest.py` | manifest 解析 + 完整性校验 | **本次新增** |
| `unit/test_workload.py` | 固定长度 workload | **本次新增** |
| `unit/test_loader.py` | 路径/目录校验负向路径 | **S00 新增** |
| `unit/test_cli.py` | CLI 参数校验 | **S00 新增** |
| `unit/core/` | errors/ids/logging/contracts/schema/config/registry/dummy/engine/migration/dependency 共 12 模块 | **S01 新增** |
| `unit/core/test_roofline.py` | Roofline/Amdahl 数学 | **S02 新增** |
| `unit/core/test_correctness.py` | golden/determinism/首错位 | **S02 新增** |
| `unit/core/test_workload_config.py` | YAML workload 单一事实源 | **S02 新增** |
| `unit/core/test_memory.py` | KV cache/权重字节核算 | **S02 新增** |
| `unit/core/test_profiling.py` | operator 表提取（新旧字段名） | **S02 新增** |
| `unit/core/test_pytorch_backend.py` | PyTorchBackend 契约合规 | **S02 新增** |
| `unit/core/test_jetson.py` | Jetson 协议防御式表面 | **S02 新增** |
| `property/test_percentile_property.py` | 百分位不变量属性测试 | **S01 新增** |
| `unit/quant/`、`unit/ops/test_quant_kernels.py`、`property/test_quant_properties.py` | 量化单元/属性/kernel 测试 | **S05 新增**，Implemented |
| `unit/integration/`（10 文件）、`property/test_integration_invariants.py` | S06 接口层测试 371 项（schema/dispatch/meta/graph/pattern/guard/cache/lowering/cuda_graph/lifecycle/adapter/taxonomy/ABI/differential/telemetry/脚手架/依赖边界） | **S06 新增**，Implemented |
| `unit/runtime/`（13 文件）、`property/test_runtime_invariants.py` | S07 接口层测试 460 项（request/capability、adapter/probe、parity、trace/账本、KV、调度、prefix、graph+attention、spec decode、失败矩阵、比较/A-B、脚手架/配置/接口表、依赖边界、属性不变量） | **S07 新增**，Implemented |
| `correctness/` `integration/` `test_vectors/` | 空（.gitkeep） | Planned（S04.5 起填充真实模型正确性矩阵） |

### 3.7 `docs/` 与 `reports/`

| 路径 | 内容 | 状态 |
|---|---|---|
| `architecture/顶层架构.md` | 顶层架构 + C1–C7 Contract | Implemented |
| `stages/S00–S15.md` | 16 个阶段路线图 | Implemented |
| `benchmark/` | methodology / metric_definitions / manifest | Implemented |
| `hardware/jetson_environment.md` | Jetson 环境清单 | Implemented |
| `architecture/module_ownership.md` | 依赖图 + 模块 ownership | **S01 新增** |
| `templates/` | ADR / 实验 / 优化日志 / handoff 模板 | **S01 新增** |
| `project_status.md` / `evidence_ledger.md` | 本报告 + 证据台账 | Implemented |
| `reports/S01_开发报告.md` / `S01_阶段验收报告.md` | S01 交付与验收 | **S01 新增** |
| `reports/S02_开发报告.md` / `S02_阶段验收报告.md` | S02 交付与验收 | **S02 新增** |
| `reports/baseline_report.md` | S02 baseline（六 workload + KV cache 画像） | **S02 新增** |
| `reports/pytorch_profile_report.md` | PyTorch Profiler hotspot 证据 | **S02 新增** |
| `reports/nsys_report.md` / `ncu_report.md` | Nsight Systems / Compute 分析 | **S02 新增** |
| `reports/S06_开发报告.md` / `S06_阶段验收报告.md` | S06 交付与验收（接口层；实验层 BLOCKED） | **S06 新增** |
| `reports/S06_graph_integration_design.md` | S06 Design 制品（编译链分层、ADR、数据布局、边界） | **S06 新增** |
| `reports/` | raw 运行证据（**gitignored，仅本机保留**） | runtime-verified |

---

## 4. 阶段完成度映射

| 阶段 | 名称 | 状态 |
|---|---|---|
| S00 | 现状审计与基线恢复 | **已完成（验收通过）** |
| S01 | 核心契约与工程质量 | **已完成（验收通过）** |
| S02 | 模型基线与全栈 Profiling | **已完成（验收通过）** |
| S03 | CUDA 算子性能工程 | **已完成（验收通过）** |
| S04 | Triton / CUTLASS / Kernel DSL | **已完成（验收通过）；「多架构未验证」例外已于 2026-09-18 关闭**（E04-01，sm_86 + sm_87） |
| S04.5 | 真实模型算子回接 | **已定义，未开始**（`docs/stages/S04.5_真实模型算子回接.md`，本轮新增） |
| S05 | 量化与低精度推理 | **接口/代码层就位（E05-01~E05-10 共 190 步能力接口）；实验层 BLOCKED**（S04.5 M4 前置缺失） |
| S06 | 框架集成与图优化 | **接口/代码层就位（E06-01~E06-11 共 218 步能力接口 + 371 新增测试）；实验层 BLOCKED**（S04.5 M4 与 S05 P0 前置缺失）。`hqsb/integration/` 18 模块 + `scripts/integration/run_e06.py` + `configs/integration/` 7 份；CUDA Graph 记 `NOT_CLAIMED` |
| S07 | 推理 Runtime 内核 | **接口/代码层就位（E07-01~E07-10 共 200 步能力接口 + 460 新增测试）；实验层 BLOCKED**（S04.5 M4 / S05 P0 / S06 P0 前置缺失）。`hqsb/runtime/` 18 模块 + `scripts/runtime/run_e07.py` + `configs/runtime/` 8 份；CUDA Graph 与 speculative/MTP 均记 `NOT_RUN` |
| S08 | ServeFabric 与性能治理 | **接口/代码层就位（E08-01~E08-11 共 264 步 + 131 新增测试）；实验层 BLOCKED**（S07 P0 等 7 条前置缺失，见 §15） |
| S09 | Ascend C/CANN 异构后端 | **部分交付**（7 模块 + 3 份映射/报告文档；`experiment/interface_map/driver/configs/tests` 缺失，见 §16.4） |
| S10 | 分布式推理与通信 | **接口/代码层就位（E10-01~E10-10 共 300 步 + 240 新增测试）；实验层 BLOCKED**（8/9 硬前置缺失，见 §16） |
| S11–S15 | 编译器 / 跨硬件 / 云原生 / 训推 / 发布 | 空目录或纯规划 |

**S05 准入判定（2026-09-18）**：**放行**，附加两条约束——
S04.5 的「未量化 HQSB 算子路径」数值基线必须先建（否则无法区分量化误差与算子
回接误差）；S05 的低比特 GEMM 不得假设 dispatcher 已接入 CUTLASS（至今没有该分支）。
完整判定见 `docs/reports/S04_阶段验收报告.md` §8。建议顺序：
`S04 补齐（已完成）→ S04.5 → S05`。

> 结论：S04 实测环境能力（Triton 3.7.1 / CUTLASS 4.7.0 / TileLang 0.1.13 三个
> DSL 在 sm_87 全部可用），实现 Triton RMSNorm/GEMM + CUTLASS GEMM 对照 + 统一
> dispatcher/capability。CUDA vs Triton 的 FP32 差异已用访存事务宽度解释
> （float4 领先，稳定）；GEMM 四方对照证明"没有万能最快的后端"（CUTLASS 默认
> 配置已具竞争力，decode 窄矩阵是 cuBLAS 薄弱区）。

---

## 5. S00 补齐内容（已完成）

1. **哈希校验能力**（负向：hash 不符 → 可诊断非零退出）
   - `hqsb/models/manifest.py`：manifest 解析 + `verify_model_files`
   - `scripts/models/verify_qwen3_hashes.py`：退出码 0（通过）/1（操作错误）/2（校验失败）
   - `hqsb/models/loader.py` 新增 `verify_manifest` 可选参数，加载前做 artifact integrity gate
2. **CLI 非法参数校验**：`hqsb/benchmark/cli.py` 提供 `positive_int`，
   `run_model_core.py` / `generate_golden.py` 改用 `type=positive_int` 并抽取 `build_parser()`
3. **CPU 最小单元测试**：`tests/unit/` 7 个模块 + `conftest.py`
4. **`.gitignore` 宽泛匹配修复**：`models/`、`reports/` 锚定为 `/models/`、`/reports/`
5. **README 状态矩阵与三条 smoke 复现路径**
6. **项目现状报告 + 证据台账 + S00 验收报告**

---

## 6. S01 补齐内容（已完成）

1. **`hqsb/core` 稳定地基**
   - `errors.py`：统一错误分类 + exit code（Usage=2/Config=3/Schema=4/Registry=5/Backend=6/Capability=7/Artifact=8/Benchmark=9）
   - `ids.py` / `logging.py`：run/trace/span ID + JSON lines 结构化日志
   - `contracts/`：C1 ModelArtifact / C2 WorkloadSpec / C3 OperatorSpec / C4 Backend+Capability / C5 QuantArtifact / C6 BenchmarkResult / C7 TraceEvent（pydantic 版本化 schema，`extra="forbid"`）
   - `schema/`：SchemaVersion + 显式迁移框架 + legacy 迁移（`migrate_any`）
   - `config/`：分层配置加载（defaults < file < env < CLI）+ 确定性 SHA256 hash
   - `registry/`：`Registry` + `RegistryHub`（backends/operators/quantizers/monitors/reporters）
2. **Backend 参考实现与 engine**：`hqsb/backends/dummy.py`（C4 参考实现，确定性输出）
   + `hqsb/benchmark/engine.py`（backend 接口编排 → BenchmarkResult）
3. **legacy 迁移**：`scripts/migrate_legacy.py` + `configs/operators/rmsnorm_v0.json`
   （model-core result → C6、golden → C6、RMSNorm → C3）
4. **工程化**：`pyproject.toml`（打包 + 可选依赖组 benchmark/serving/ascend/dev）
   + `.github/workflows/ci.yml`（CPU CI 3.10–3.12）+ pytest markers
5. **文档模板与检查**：`docs/templates/`（ADR/实验/优化日志/handoff）
   + `scripts/check_docs.py` + `docs/architecture/module_ownership.md`
6. **测试**：12 个 core 测试模块 + property 测试；全量 **166 passed**

---

## 7. S02 补齐内容（已完成）

1. **FP16 Reference Runtime**：`hqsb/backends/pytorch.py`（`PyTorchBackend`，C4 契约）
   - 幂等 `load` / `warmup` / `generate`（repetitions 次 model-core pass）/ `health`/`metrics`/`close`
   - `generate` 产出 `GenerationOutput`，`backend_metrics` 含 KV cache/权重/RSS/swap/CUDA 内存
2. **分析模块**：
   - `roofline.py`：Roofline 模型 + Amdahl 定律 + 热点分类/排序 + Orin FP16 预设
   - `correctness.py`：token hash / 序列对比（首错位定位）/ logits 容差 / determinism / golden 对比
   - `workload_config.py`：YAML 六 workload 单一事实源 → `WorkloadSpec`
   - `memory.py`：KV cache 字节核算 / 权重字节 / RSS/swap / CUDA 快照
   - `profiling.py`：`profile_model_core` + operator 表提取（兼容新旧 PyTorch 字段名）
3. **Jetson 实验协议**：`hqsb/hardware/jetson.py`（温度/冷却/电源模式/平台探测）
4. **计时与内存修正**：`model_core.py` 保存 raw ITL + KV cache + 权重/RSS；`GenerationOutput` 扩展 `backend_metrics`；`engine.run` 增加 `load_artifact`；`loader.py` Jetson 内存决策
5. **脚本**：`run_s02_baseline.py`、`profile_model.py`、`analyze_hotspots.py`、`nsys_profile.sh`、`ncu_profile.sh`
6. **测试**：7 个新测试模块；全量 **238 passed**
7. **真实硬件证据**：端到端 smoke（decode 9.28 tok/s、KV cache 4.59MB、权重 3.44GB）+ 真实 CUDA Profiler（decode GEMM ~78%、prefill GEMM ~48%）→ S03 Hotspot Decision

---

## 8. S03 补齐内容（已完成）

1. **RMSNorm 算子库重构**（`ops/cuda/rmsnorm/`，include/src/tests/bench）
   - 公共 API `ops/cuda/rmsnorm/include/hqsb/rmsnorm.h`：stream-aware、无隐藏分配、无 Python 依赖
   - V0 shared reduction（S00 baseline 提取）→ V1 warp shuffle → V2 float4/half2 vectorized
   - dispatcher（dtype/shape 选择 + 不支持组合明确 fallback）
   - CPU FP64 reference（独立于 GPU kernel）
2. **第二热点算子**（`ops/cuda/fused_residual_rmsnorm/`）：fused residual+rmsnorm V0/V1
3. **测试框架**：`common/test_util.h`（轻量断言）+ `common/test_metrics.h`（5 项数值对比）
4. **correctness 测试**：rmsnorm 33 checks + fused 15 checks，CTest 2/2 passed
5. **benchmark**：device event + host submit+sync 双口径，block 扫参，occupancy 查询
6. **性能结论**：V2 +56%（block=256）~+142%（block=512）；三个真实退化（FP16 half2 / 小 hidden / block=1024）；fused V1 无收益（RAW 依赖）
7. **C3 OperatorSpec**：`rmsnorm_v1/v2.json`、`fused_residual_rmsnorm.json`
8. **文档**：Optimization Logs、`S03_benchmark_report.md`、开发报告、验收报告

---

## 9. S04 补齐内容（已完成）

1. **环境能力实测**（推翻"Jetson 不支持 Triton"默认假设）
   - `ops/capability.py`：Triton/CUTLASS/TileLang 实测编译 probe（非仅 import）、
     cuBLAS、CUDA shared lib 定位，永不抛异常 + `lru_cache`
   - 结论：三个 DSL 在 sm_87 **全部可用**——Triton 3.7.1（最小 kernel
     max_err=0.0）、CUTLASS 4.7.0（FP16 GEMM max_err~0.03）、TileLang 0.1.13
     （elementwise add max_err=0.0）。CUTLASS 初因 GitHub 443 超时无法获取，
     网络恢复后已补齐（见第 8 点）
2. **CUDA shared lib + ctypes 绑定**：`rmsnorm_c_api.cu`（`extern "C"`）+ 
   `ops/cuda_bridge.py`（`rmsnorm_forward`），让 Python 统一 dispatcher 能调 S03 CUDA kernel
3. **Triton 实现**：`ops/triton/rmsnorm.py`（reference + autotune）、
   `ops/triton/gemm.py`（tiled GEMM reference + autotune，`input_precision="ieee"`）
4. **统一 dispatcher**：`ops/dispatcher.py`（capability→arch→shape/dtype→fallback 四层策略）
5. **脚本**：`bench_s04.py`（四方对照）、`dump_triton_ir.py`（TTGIR/LLIR/PTX + 寄存器元数据）
6. **测试**：`tests/unit/ops/` 5 文件，全量 **270 passed**（新增 32；S04 结束口径，
   E00-06 @ e4a031c 复跑当前 HEAD 全量为 **340 passed**，见 §9 末）
7. **性能结论**（详见 `S04_comparison_report.md`）
   - FP32 RMSNorm：CUDA V2 领先（float4，稳定）
   - FP16 RMSNorm：短 kernel 测量波动，不具可复现性（诚实记录）
   - GEMM 四方对照：CUTLASS 默认配置已具竞争力，窄矩阵（M=1）是 cuBLAS 薄弱区
   - autotune 在 edge 设备非最优 → 证明 auto-tuning 非全局常量
8. **网络恢复后补全（本阶段追加）**：
   - **CUTLASS 4.7.0**：`third_party/cutlass`（gitignore）+ `hqsb_cutlass_gemm_bench`
     FP16 GEMM 对照（`OpClassTensorOp + Sm80`，正确性 max_err ~0.03）
   - **TileLang 0.1.13**：`ops/_tilelang_probe.py`（elementwise add 实测 max_err=0.0）
   - **capability 扩展**：三个 DSL（Triton/CUTLASS/TileLang）全部实测可用
   - **HIP/ROCm/OpenCL L2 技术说明**：`architecture/portable_kernel_backends.md`
9. **S00 E00-06 审计复验（2026-09-04, HEAD `e4a031c`）**：
   - 全量 `pytest -q` 复跑 **340 passed**（19.22 s，exit 0），log
     `docs/stage_experiments/S00/E00-06/raw/rerun/s2_14_s4_13_pytest.stdout`
   - capability 复验：CUDA(8,7) / Triton 3.7.1 / CUTLASS 4.7.0 / TileLang 0.1.13 /
     cuBLAS 全部可用，notes 为空，`.../rerun/s4_capability.json`
   - compute-sanitizer host 泄漏 0 bytes（PASS 35 checks），
     `.../rerun/s3_11_compute_sanitizer.stdout`
   - CPU wheel `pip wheel . --no-deps` 构建成功，`.../rerun/s1_10_pip_wheel.stdout`
   - 审计修正：README/ledger 中过时的“238/270 passed”已按当前 HEAD 口径更新或降级；
     `hqsb/rmsnorm.h` 修正为 `ops/cuda/rmsnorm/include/hqsb/rmsnorm.h`；S03 历史绝对
     数字以 tracked `docs/reports/S0x_*_report.md` 为准（电源/频率不同时复跑不可比）。

---

## 10. 风险与注意事项

- **`scripts/common/git_commit.sh`**：包含 `git rebase --root` 与
  `git push origin main --force`，会改写历史并强推。已被 `.gitignore` 排除，属
  本机工具；**不应提交**，若需团队协作应改为无强推的安全流程。
- **顶层 `/reports/` 被 `.gitignore` 忽略**（已锚定，仅忽略 raw 数据）：raw 证据
  仅存于本机，未纳入版本控制。跨机器审计需依赖 artifact 索引或另行归档。
- **`configs/environment/jetson_python_lock.txt`** 是全量系统 pip 冻结（含 ROS、
  Jupyter 等无关包），非项目最小依赖集；已在 `pyproject.toml` 建立按依赖组划分的
  可选依赖，但 Jetson 专用 torch wheel 仍需按硬件 pin。
- **golden 数据仅 4 份**（osl 均为 32），与六 workload 基线不对齐；已迁移到 C6
  schema。`correctness.py` 已就绪，算子级 golden 门禁待 S05/S06 接入。
- **profiling 需 root**：CUPTI 采集需 `sudo` + `kernel.perf_event_paranoid=0`
  （已文档化在 `pytorch_profile_report.md` / `nsys_report.md`）。
- **Nsight Systems / Compute 未在本次会话实际采集**：脚本已就绪，Jetson L4T 下
  ncu 部分 metric 受限；已用 Roofline + benchmark 双口径 + occupancy 作为替代证据。
- **compute-sanitizer GPU debug 受 L4T 限制**：host 泄漏检查通过（0 bytes leaked），
  GPU debug 需 datacenter GPU。
- **ruff / mypy** 本机未安装（CI 会安装），本机可 `pip install -e ".[dev]"` 补装。

---

## 11. 下一步（S04.5 → S05 输入）

**2026-09-18 跨架构补验后的更新**：

已由本轮补验关闭/修正：

- ~~多架构验证待云端环境补~~ → **已完成**（RTX 3090 sm_86 + Jetson sm_87，
  E04-01：`docs/stage_experiments/S04_实验清单.md` §2）；
- **CUTLASS 默认配置不可移植**：`128×256×64×3`（144 KiB）在 sm_86 无法启动，
  已改为按设备共享内存预算选型（`large`/`compact`）；
- **CUTLASS 对照原为伪造数据**（内核未启动仍输出时间/正确性），已改为失败即
  非零退出并纳入 CTest；
- **dispatcher 架构硬编码**已消除，改由共享库自报编译架构；
- **制品门禁恢复可复现**（`.msc` 客户端缓存索引排除后 13/14 PASS），
  E00-05 由 FAIL 转 PASS。

仍需在后续阶段处理：

- **S04.5（真实模型算子回接）**：建立「未量化 HQSB 算子路径」的逐层数值基线与
  全模型 token/logits 对齐；**S05 的精度验收依赖此基线**；
- **S05 量化**：RTN/GPTQ/AWQ/SmoothQuant（`hqsb/quant/` + 低比特 kernel）；
- **dispatcher 接入 CUTLASS/TileLang**：目前只有能力标记，没有 dispatch 分支；
- **CUTLASS tile/stage 性能扫参**（本轮只做了架构可行性选型）；
- **多架构 fatbin 支持**与**按 CUDA 语义的兼容判定**（当前为精确相等，偏保守）。

S04 handoff 与验收见 [`reports/S04_阶段验收报告.md`](reports/S04_阶段验收报告.md)；
跨架构补验流水线见
`reports/dev/rtx3090/20260918_021649/RTX3090_ACCEPTANCE_REPORT.md`。

---

## 12. S05 补齐内容（接口/代码层，实验层 BLOCKED）

按任务约束「提供接口、不执行实验」，S05 交付了 E05-01~E05-10 全部 190 个实验
步骤的能力接口与测试，**未执行任何正式实验、未产出任何质量/性能/内存/能耗结论**。

1. **量化数学与制品**：`hqsb/quant/spec.py`（语义冻结 + factorial 矩阵）、
   `rounding.py`、`rtn.py`（自实现 RTN，axis 通用）、`golden.py`（精确有理数
   双实现交叉校验）、`packing.py`（canonical/kernel layout）、`artifact.py`
   （版本化 QuantArtifact，原子保存/身份哈希）、`compat.py`（兼容判定/repack/
   迁移）、`faults.py`（35 例故障注入）、`fixtures.py`（合成自检制品）。
2. **统计与校准**：`stats.py`（bootstrap/非劣效）、`calibration.py`（四段隔离/
   泄漏审计/最小充分预算）。
3. **模型级评估**：`coverage.py`、`apply.py`（可逆权重替换）、`quality.py`、
   `execution.py`（执行标签/claim 审计）、`oracle.py`（量化 vs kernel oracle）、
   `model_eval.py`（内存阶梯/相位计时/统一结果表/五道门）。
4. **工业方法**：`adapters/`（GPTQ/AWQ/SmoothQuant 五层 + 字段映射审计）。
5. **敏感性与决策**：`units.py`、`sensitivity.py`、`policy.py`、`activation.py`
   （P1）、`kv.py`（P1）、`decision.py`（注册/门/Pareto/场景/推荐）。
6. **执行层**：`ops/quant/`（capability/executors/w4a16_triton/microbench/safety），
   Triton W4/W8 fused-dequant GEMM 对 oracle 分层容差通过（本机 smoke）。
7. **驱动与配置**：`scripts/quant/run_e05.py`（status/preregister/interface-map/
   self-check/execute，默认拒绝结论）、`configs/quantization/*.yaml`（10 份）、
   `hqsb/quant/interface_map.py`（190 步→219 接口，全部解析）。
8. **测试**：全量 **936 passed**（新增约 280）；依赖边界 0 violations；
   故障注入 35/35 捕获；golden 11 向量 0 失配。

**阻塞**：S04.5 M4「真实模型算子回接」前置未满足。`run_e05.py --mode
execute --confirm-execute` 如实拒绝（退出码 7），不产结论。

S05 报告：`docs/reports/S05_开发报告.md`、`docs/reports/S05_阶段验收报告.md`。

> 注：S05 原前置检查把「`hqsb/integration/` 目录存在」当作 S04.5 M4 证据；
> S06 交付落在同一路径后，该检查已被**收紧**为要求执行证据标记
> `hqsb/integration/s04_5_evidence.json`（`hqsb/quant/experiment.py`，
> 见 §13 与 `docs/reports/S06_开发报告.md` §2.3）。

---

## 13. S06 补齐内容（接口/代码层，实验层 BLOCKED）

按任务约束「提供接口、不执行实验」，S06 交付了 E06-01~E06-11 全部 **218 个
实验步骤**的能力接口与测试，**未执行任何正式实验、未产出任何图/编译/性能/
内存/命中率结论数字**。

1. **算子与 dispatcher 边界**：`specs.py`（schema = 编译契约、单一 schema owner
   审计）、`dispatch.py`（注册矩阵冲突拒绝、dispatch table 快照与 diff、
   redispatch 递归/耗尽守卫、capability 驱动选择、确定性 fallback、opcheck 计划）。
2. **Meta/Fake 与图 IR**：`meta.py`（符号维度、元数据契约、real-vs-fake 逐字段
   oracle、无分配证据 spy、guard 集合）、`graph.py`（capture mode/IR level 标注、
   结构哈希、graph diff、FX/声明式适配）。
3. **重写与 lowering**：`patterns.py`（结构候选 → 语义谓词 → 副本重写、字段级
   拒绝原因、四种覆盖率、负向变异测试）、`lowering.py`（能力驱动选型、拒绝原因、
   分配对账、Amdahl 归因、消融矩阵）。
4. **guard/编译/cache**：`guards.py`（五类事件分离计数、动态策略、有序 shape
   trace、storm 阈值、guard 最小性审计）、`cache.py`（graph/compile identity、
   15 个编译相位、entry 执行前校验、scratch 损坏注入、失效矩阵、并发锁、
   break-even）。
5. **失败与生命周期**：`taxonomy.py`（14 层 stage + 稳定 reason code + 事务化执行
   + 确定性 fallback + 消息脱敏）、`abi.py`（load 前 ABI/arch 判定）、
   `lifecycle.py`（状态机、分段斜率/平台期/泄漏判定、弱引用探针、stream 审计、
   teardown）、`cuda_graph.py`（P1 契约 + claim 门，当前 `NOT_CLAIMED`）。
6. **跨目标复用与证据投影**：`adapter.py`（model/backend 协议、硬编码扫描、
   改动分类、dummy backend、identity 碰撞）、`differential.py`（8 条路径矩阵、
   预注册容差、冻结融合语义、首次发散定位、正确性矩阵）、`telemetry.py`
   （C6/C7 投影 + 字段/事件覆盖审计）、`policies.py`（7 类冻结 spec + 漂移审计）。
7. **驱动与配置**：`scripts/integration/run_e06.py`
   （status/preregister/interface-map/self-check/execute，默认拒绝产结论）、
   `configs/integration/*.yaml`（7 份）。
8. **测试**：全量 **1303 passed**（本阶段新增 371）；依赖边界 gate 0 违规 0 环
   （新增 region `integration` 与规则 R2）；接口解析 218 步 / 381 接口全部通过；
   ruff 本阶段文件全绿（仓库其余 76 处为既有）。

**阻塞**：S06 的 7 条硬前提中 6 条未满足（S04.5 执行证据标记、S04.5 证据目录、
S04.5 验收报告、六 workload FP16 基线、S05 P0 verdict、环境指纹）。
`run_e06.py --mode execute --confirm-execute` 如实拒绝（退出码 7），不产结论。

**既有文档债务（未修）**：`docs/stage_experiments/details/*/README.md` 43 处相对
链接断裂（S04/S04.5 清单文件随 `9b403aa` 移出树；控制平面文档实际位于
`docs/architecture/`）。协议目录冻结。

S06 报告：`docs/reports/S06_开发报告.md`、`docs/reports/S06_阶段验收报告.md`、
`docs/reports/S06_graph_integration_design.md`。

---

## 14. S07 补齐内容（接口/代码层，实验层 BLOCKED）

按任务约束「提供接口、不执行实验」，S07 交付了 E07-01~E07-10 全部 **200 个
实验步骤**的能力接口与测试，**未执行任何正式实验、未产出任何 TTFT/TPOT/TPS/
容量/碎片/命中率/显存/能耗结论数字**。

1. **请求语义与能力协商**：`request.py`（canonical `RequestSpec`/`SamplingSpec`/
   `StopSpec`/`ModelIdentity`/`BackendSpec`、capability 六态、requested→resolved
   与 silent-degradation 拒绝、唯一主 runtime 校验）。
2. **Adapter 契约与探测**：`adapter.py`（七类操作、生命周期状态机、stream 校验、
   cancel 三时刻、dummy/reference adapter、注册表、`find_spec` 诚实探测——本机
   vllm/sglang/tensorrt_llm/llama_cpp 实测 `NOT_INSTALLED`，主 runtime 选择 `ok=False`）。
3. **语义 oracle**：`parity.py`（greedy 逐 step + 首差异、多 seed 分布比较并拒绝
   单 seed 断言、streaming 拼接、边界 case、参数生效矩阵、unsupported 负向矩阵）。
4. **Trace 与账本**：`trace.py`（请求状态机、C7 span（显式 request 上下文 +
   脱敏）、iteration 账本与 token 守恒、时钟校准、插桩开销、hot path、
   由 raw 生成的状态机/调用链）。
5. **KV**：`kv.py`（几何、block 生命周期状态机与不变量、refcount/共享、
   碎片**命名分类**与预测—实测对账、容量二分、**每个 kind 一条有限 OOM 阶梯**、
   上下文最早层拒绝、长稳斜率）。
6. **调度**：`scheduler.py`（static/continuous/chunked 确定性调度、token/sequence
   budget、按完整 ISL 预留的 admission、preemption/recompute、chunk 覆盖审计、
   Jain 公平、拥塞点、策略曲线；所有 payload 带 `simulated=True`）。
7. **Prefix cache**：`prefix_cache.py`（15+2 字段 key 绑定、digest 敏感性、
   两种碰撞政策与伪造记录 fixture、多 KV group 交集、refcount/eviction 保护、
   净收益模型）。
8. **Graph/Attention**：`graph_route.py`（桶与越界 fallback、replay 区分度、
   capture break-even、attention 逐字段能力判定、2×2 factorial 与交互项、
   phase 分离、claim 门、惰性复用 S06 CUDA Graph 契约）。
9. **Speculative/MTP（P1）**：`spec_decode.py`（算法具名、精确有理数 acceptance/
   residual、greedy exactness、多 seed 分布门、KV commit/rollback 审计、
   cycle 成本与收益模型、MTP 自带契约、`claim_status → NOT_RUN`）。
10. **失败与恢复**：`failure.py`（28 例冻结矩阵、10 条公共不变量、cancel 时间线、
    有界 OOM 序列、上下文滥用检查、并发释放/load-close 场景、分段斜率与
    「稳态仍增长即阻塞 PASS」、run 分离）。
11. **公平比较**：`comparison.py`（tier A–D 自动派生、common/best-valid 两表分离、
    从 raw 统一重算、token 三分母审计、冷热相位、per hardware×workload Pareto、
    limitations、S08 稳定接口面）。
12. **策略 A/B**：`policy_ab.py`（选题门禁、ADR、唯一变量 identity、ABBA/随机区组、
    配对效应与 guard band、因果链四条件、回归包线、消融、按预注册裁决、pilot 隔离）。
13. **口径与投影**：`metrics.py`（时间口径、三分母守恒、run 级分布与配对 bootstrap）、
    `telemetry.py`（C6 20 字段 + C7 span 链投影与覆盖审计，不改冻结 schema）。
14. **驱动与配置**：`scripts/runtime/run_e07.py`
    （status/preregister/interface-map/self-check/execute，默认拒绝产结论）、
    `configs/runtime/*.yaml`（8 份严格校验 + 契约审计）、
    `hqsb/runtime/interface_map.py`（200 步 / 308 接口 / 429 引用全部解析）。
15. **测试**：全量 **1763 passed**（本阶段新增 460：415 单元 + 45 属性）；
    依赖边界 gate 规则 R1–R4 0 违规 0 环（`RULES_VERSION` 1.1.0）；
    ruff 本阶段文件全绿；wheel 142 entries 含 18 个 runtime 模块。

### 14.1 阻塞与交接

**阻塞**：8 条硬前提中 7 条未满足——S04.5 M4 执行标记、S05 quality/kernel verdict、
S06 稳定 capability verdict、冻结请求夹具、runtime capability probe、主 runtime
选择、协议树内环境指纹。`run_e07.py --mode execute --confirm-execute` 如实拒绝
（退出码 7），不产结论。

**前置门收紧（防自我解锁）**：指纹与 probe/selection 只接受 `docs/stage_experiments/**`
（协议证据树，本阶段代码从不写入）；脚手架写在 `experiment_results/` 的指纹**不计入**
（回归测试 `test_scaffolding_written_fingerprint_does_not_satisfy_the_gate`）。

**既有问题（未修，已登记）**：`scripts/audit/run_e01_06_*.py` 的 `overall=FAIL`
（`logs_joinable`、`schema_missing_required_field`、`schema_unknown_field`）在干净
HEAD 工作树上同样 FAIL，属既有 `schema_field_gap` / `run_trace_linkage_gap`；
S06 报告中的「PASS（22/22）」指 case 数，不是 overall。

**既有文档债务（未修）**：`docs/stage_experiments/details/*/README.md` 43 处相对链接
断裂（S04/S04.5 清单随 `9b403aa` 移出树；控制平面文档实际位于 `docs/architecture/`）。
协议目录冻结。

**未覆盖边界**：真实 runtime 适配器未实现（引擎均未安装）；prefix cache 仅支持
offset 0 的 block chain；调度器是确定性模拟器；CUDA Graph 与 speculative/MTP 均
`NOT_RUN`。

S07 报告：`docs/reports/S07_开发报告.md`、`docs/reports/S07_阶段验收报告.md`、
`docs/reports/S07_runtime_architecture.md`。

---

## 15. S08（ServeFabric 与性能治理）—— 接口/代码层就位，实验层 BLOCKED

> 追加时间：2026-09-18。S08 按任务约束「提供接口、不执行实验」交付
> E08-01~E08-11 全部 **264 个实验步骤**的能力接口与测试，**未执行任何正式实验、
> 未产出任何容量/延迟/吞吐/goodput/命中率/显存/能耗结论数字**。实验层 `BLOCKED`。

### 15.1 新增模块（`hqsb/serving/`，26 个）

1. **协议平面**：`protocol.py`（`ProtocolProfile`/`ErrorCatalog`/canonical 请求路径/
   响应 oracle/负向语料/一致性矩阵）、`sse.py`（编解码/增量解析/framing oracle/
   流-非流配对）。
2. **网关平面**：`gateway.py`（请求状态机/SSE 交付/取消/drain）、`transport.py` +
   `transport_http.py`（抽象 + stdlib HTTP/1.1+SSE）、`timing.py`（五种 TTFT/时钟域/
   时间守恒）、`pipeline.py`（投递账本/五层缓冲/取消线性化）。
3. **策略平面**：`slo.py`（SLO 预注册/goodput/计数漏斗/G*）、`arrival.py`（可重放
   到达/保真门/突发恢复）、`loadgen.py`（开闭环/客户端有效性/no-op 校准）、
   `clients.py`（可重放客户端行为脚本）、`fairness.py`（成本模型/Jain/饥饿/HOL）、
   `policies.py`（FIFO/严格优先级/加权公平统一接口）。
4. **后端平面**：`router.py`（注册表/硬过滤/评分/route-vs-actual）、`circuit.py`
   （熔断/隔离/恢复）、`cache_routing.py`（前缀身份/四策略净收益/倾斜）、
   `faults.py`（故障矩阵/注入证据/尝试血缘/爆炸半径）、`admission.py`（压力状态机/
   有界队列/重试预算/过载波形）。
5. **证据平面**：`observability.py`（trace context/span/metrics/logs/根因）、
   `telemetry.py`（C6/C7 投影）、`service_ab.py`（瓶颈证据表/guardrail/四态裁决）、
   `experiment.py`（前置门/预注册/run 布局/verdict 拒绝）、`specs.py`（12 份配置
   严格加载 + 契约审计）、`interface_map.py`（264 步对照 + 导入校验）、
   `dummy_backend.py`（无模型协议夹具）。

### 15.2 配置与驱动

- `configs/serving/*.yaml` 12 份冻结配置（严格键校验 + 逐文档契约审计全绿）。
- `scripts/serving/run_e08.py`：`--list/--prerequisites/--interface-map/--smoke/
  --experiment`，默认拒绝产结论。

### 15.3 测试与门禁

- 全量 **1894 passed**（本阶段新增 131 = 125 单元 + 6 属性）。
- 依赖边界 gate 规则升至 **R1–R6**（`RULES_VERSION` 1.2.0）0 违规 0 环；
  `module_ownership.md` 1.3.0 新增 `serving` 区域。
- 接口解析：264 步 / 418 唯一接口 / 791 引用全部解析。
- ruff：`hqsb/serving`、`tests/unit/serving`、`tests/property/test_serving_invariants.py`、
  `scripts/serving` 全绿；仓库其余 74 处为既有问题，未新增。

### 15.4 阻塞与交接

**阻塞**：7 条硬前置未满足——S07 P0 verdict、双后端注册、冻结请求夹具、冻结 SLO、
拓扑记录、loadgen 校准、协议树内环境指纹。`run_e08.py --experiment E08-01
--prerequisites` 如实报告 `satisfied=false`；`--execute` 仍拒绝，不产结论。

**前置门收紧（防自我解锁）**：前置证据只接受 `docs/stage_experiments/S08/**`（协议
树）；脚手架写在 `experiment_results/` 的指纹不计入（回归测试
`test_prerequisites_do_not_self_unlock`）。

**未覆盖边界**：真实 Backend/引擎未接入（`dummy_backend.claim_allowed()` 恒 False）；
HTTP 绑定为 stdlib 参考实现；真实并发/竞态需真实后端实验。

S08 报告：`docs/reports/S08_开发报告.md`、`docs/reports/S08_阶段验收报告.md`、
`docs/reports/S08_serving_architecture.md`。

---

## 16. S10（分布式推理与通信）—— 接口/代码层就位，实验层 BLOCKED

> 追加时间：2026-09-19。S10 按任务约束「提供接口、不执行实验」交付
> E10-01~E10-10 全部 **300 个实验步骤**的能力接口与测试，**未执行任何正式实验、未产出任何
> collective/TP/scaling/overlap/MoE/故障结论数字**。实验层 `BLOCKED`。

### 16.1 新增模块（`hqsb/distributed/`，20 个）

1. **拓扑与身份平面**：`topology.py`（ObservationScope/HostIdentity/NumaTopology/AcceleratorRecord/
   VisibleDeviceAudit/PcieLink/FabricLink/AffinityRecord/NicRecord/RdmaStackRecord/BackendRuntimeConfig/
   TopologyManifest+canonical hash/六档漂移分级/DegradedLinkPolicy/硬不变量/E10-02 门禁）、
   `ranks.py`（RunIdentity/RankIdentity/GroupMembership）、`placement.py`（PlacementPlan→rank table/
   launcher、planned-actual 比对、替代 placement）、`probes.py`（只读探针 + P2P/copy/NUMA/RDMA/
   data-path 证据 + preflight 负向 fixture）。
2. **通信平面**：`backend.py`（backend/harness 身份、能力三态、计时语义、requested/actual、错误归一）、
   `collectives.py`（6 类 op 语义、CPU oracle、rank-coded 生成、guard、size/rank grid、带宽公式登记表、
   α–β 分段与拐点、官方工具交叉验证、停止规则、loopback 测试双）、`faults.py`（错误目录、恢复级别、
   故障 oracle、时间指标、安全边界、控制面/watchdog、abort 协调器、资源快照、19 类故障矩阵、
   盲判演练、有界性汇总）、`sequence.py`（communicator 状态机、序列命名空间、call record、
   metadata preflight、timeout、延迟阶梯、错误传播、清理顺序、TP 安全门）。
3. **模型与性能平面**：`parallel_plan.py`（Qwen census/TP 能力/ParallelPlan/shard round-trip/
   direct-load 审计/KV ownership/8 级 correctness case/不整除策略）、`ledger.py`（expected↔observed
   通信账本 + 原因码 + 内存分桶对账）、`scaling.py`（strong/weak/capacity work unit、资源矩阵、
   T1 诚实性、时间分解、pairability、scaling 拟合、异常标记、confirmation、五态裁决）、
   `overlap.py`（依赖 DAG、合法窗口、三组 schedule、chunk、stream policy、区间集合 overlap、
   竞争/开销、因果矩阵、ABBA、phase policy）、`boundary.py`（P1 激活、候选定义、rubric、成本模型、
   PP/CP/SP plan、stage balance/bubble、adopt/reject）、`moe.py`（claim level、RouteArtifact、
   dispatch/combine oracle、count matrix、skew profile、imbalance、placement A/B、holdout、分级裁决）、
   `traces.py`（统一 trace 事件、时钟校准、事件配对、arrival/completion skew、phase breakdown、
   通信矩阵、baseline 变异度、根因分类器、放大指标、注入计划、runbook、公共 summary）。
4. **证据与脚手架**：`telemetry.py`（C6/C7 投影 + 覆盖审计 + 统一数据表 schema）、
   `specs.py`（12 份配置严格加载 + 逐字段审计）、`experiment.py`（9 条前置门/预注册/统一记录/
   Evidence Manifest/RunDirectory/verdict 拒绝/环境指纹）、`interface_map.py`（300 步对照 + 导入校验）。

### 16.2 配置与驱动

- `configs/distributed/*.yaml` 12 份冻结配置（严格键校验 + 与代码逐字段审计全绿；其中
  `timeout/scaling/fault/tolerance` 为 `template`，正式 run 前须冻结）。
- `scripts/distributed/run_e10.py`：`--list/--prerequisites/--interface-map/--smoke/--execute`，
  默认拒绝产结论。

### 16.3 测试与门禁

- 全量 **2134 passed, 4 deselected**（本阶段新增 **240** = 225 单元 + 15 属性）。
- 依赖边界 gate 规则升至 **R1–R8**（`RULES_VERSION` 1.3.0）0 违规 0 环；
  `module_ownership.md` 1.4.0 新增 `distributed` 区域。
- 接口解析：**300 步 / 381 唯一接口 / 548 引用**全部解析。
- ruff：`hqsb/distributed`、`tests/unit/distributed`、`tests/property/test_distributed_invariants.py`、
  `scripts/distributed` 全绿；仓库其余 83 处为既有问题，未新增。

### 16.4 阻塞与交接

**阻塞**：8/9 条硬前置未满足——S07 P0 verdict、S08 trace、双加速器（本机仅 1×RTX 3090）、
封存 topology manifest、冻结 backend 身份、单卡 reference、冻结 model/workload、协议树指纹。
`run_e10.py --experiment E10-01 --prerequisites` 如实报告 `satisfied=false`；`--execute` 仍拒绝。

**前置门收紧（防自我解锁）**：前置证据只接受 `docs/stage_experiments/S10/**`（协议树）；
脚手架写在 `experiment_results/` 的指纹不计入（回归测试
`test_prerequisites_do_not_self_unlock`）。

**漂移登记（文档 vs 代码）**：`docs/reports/S09_*.md` 声称交付的
`hqsb/ascend/{experiment,interface_map,backend,framework,quant_format,model_core,profiling,
comparison,faults,mapping,telemetry,specs}.py`、`scripts/ascend/run_e09.py`、`configs/ascend/`、
`tests/unit/ascend/` 在**当前工作树全部不存在**（实测）；S09 应降级为"部分交付"。
本阶段未改写上游报告，仅登记并给出最小回退建议（S10 开发报告 §2.2）。

**未覆盖边界**：无第二加速器/多节点/RDMA/Ascend 设备；collective 只经 CPU loopback 自检
（`claim_allowed()` 恒 False）；E10-07 为 `NOT_RUN_NOT_CLAIMED`（未声称 PP/CP/SP）；
依赖门禁尚未覆盖 `hqsb.ascend` 区域。

S10 报告：`docs/reports/S10_开发报告.md`、`docs/reports/S10_阶段验收报告.md`、
`docs/reports/S10_parallelism_design.md`。

---

## 17. S11（AI 编译器与自动优化）—— 接口/代码层就位，实验层 BLOCKED

> 追加时间：2026-09-19。S11 按任务约束「提供接口、不执行实验」交付 E11-01~E11-10
> 全部 **320 个实验步骤**的能力接口与测试，**未执行任何正式实验、未产出任何
> capture/rewrite/lowering/autotune/cache/性能结论数字**。实验层 `BLOCKED`。

### 17.1 新增模块（`hqsb/compiler/`，20 个）

1. **基础层（L0）**：`identity.py`（ArtifactIdentity/双 hash/canonicaliser/三合一身份/lineage DAG/版本冻结）、
   `ir.py`（符号维度与约束、effect 三态、IRValue/IROp/IRGraph、14 项 IR verifier、序列化 round-trip、
   FX importer 惰性 torch、IR diff）、`records.py`（case 状态机、失败分类目录、编译成本键与 break-even、
   §21.1–21.5 五类记录 schema）。
2. **前端层（L1）**：`capture.py`（capture 矩阵、break 目录与定位、source/shape/effect metadata、
   四维完整度、五类 coverage 与 claim guard、debug backend、有序 shape trace、原生 trace 计划与 join、
   repeatability、hero manifest、负向 fixture）、`guards.py`（9 类 guard 与失败动作、variant 与 lookup、
   域覆盖/重叠分析、重编译可解释性、wrong-reuse 审计、策略总成本对照、variant 预算、并发、动态策略文档）。
3. **重写层（L2）**：`pattern_library.py`（两个冻结 pattern 契约与 signature、8 条 predicate 目录、
   near-miss 变异轴、语料计划、CPU composed/fused 参考 oracle）、`rewrite.py`（结构匹配、语义判定、
   proof record、原子重写与回滚、pass pipeline/固定点/幂等/确定性、语料评估 FP=0、provenance 审计、
   metadata diff、失败注入）。
4. **后端层（L3）**：`targets.py`（target 快照与冻结版本门、capability 判定与结构化原因、只读探针、
   toolchain 导出能力）、`lowering.py`（registry 与候选字段、完整候选表选择链、reference/forced/auto 策略、
   materialize 与 fallback 可达性、dispatch telemetry 与双重证据、pre-launch 检查、失败注入、
   正确性/性能顺序、compile breakdown）、`backend.py`（九步 backend 契约、CompileRun 组装、
   debug-backend 拒绝、artifact manifest）、`codegen.py`（生成源制品、导出计划、resource 解析、
   memory/launch ledger、机制假设表与 ablation、Amdahl 归因、profiler 计划、重建检查、跨层 diff）。
5. **搜索层（L4）**：`autotune.py`（search space/约束/候选 identity、静态过滤与 false-reject 审计、
   B0–B3 预算与公平性、组切分与 holdout、trial sandbox 与 reset、测量顺序、oracle/winner/confirmation、
   holdout 评估、搜索成本与 break-even、tuning DB、策略文档）、`costmodel.py`（特征 schema 与泄漏审计、
   label/tie、group split 与 final-test 一次性、基线、常数/线性/pairwise 模型、regret/ranking/overhead、
   confidence 与 OOD/risk-coverage、安全回退链、非法候选注入、模型损坏门、methodology/deployment 双裁决）。
6. **制品层（L5/L6/L7）**：`cache.py`（22 字段 key 规范与 test vectors、事务发布、9 步安全读取、
   11 例损坏注入与隔离、14 行失效矩阵、并发/键遗漏/淘汰、C0–C4 与指标）、`portable.py`（栈选择与 scope、
   语义映射表、legalisation 四态、loop IR 校验、schedule/pass trace 与重放、bridge 契约与开销、
   开发成本 rubric、角色对照、采用裁决）、`aigate.py`（task/provenance、沙箱与 harness 锁、
   hidden/metamorphic 设计、G0–G10 门链、wrong corpus、fast_p/Pareto、admission schema、
   迭代血缘、对抗检测矩阵、claim 边界）。
7. **证据与脚手架（L8）**：`telemetry.py`（C6/C7 投影 + 12 张 raw 表 schema + 覆盖审计）、
   `specs.py`（12 份配置严格加载 + 逐字段审计）、`experiment.py`（9 条前置门/预注册/统一记录/
   Evidence Manifest（含 S11 身份字段）/RunDirectory/verdict 拒绝/环境指纹）、
   `interface_map.py`（320 步对照 + 导入校验）。

### 17.2 配置与驱动

- `configs/compiler/*.yaml` 12 份冻结配置（capture/guard/pattern/target/lowering/autotune/costmodel/
  cache/portable/aigate/tolerance/experiment），严格键校验 + 与代码逐字段审计全绿。
- `scripts/compiler/run_e11.py`：`--list/--prerequisites/--interface-map/--spec-audit/--smoke/--execute`，
  默认拒绝产结论；`--prerequisites` 实测 6 条必需中 3 条未满足。

### 17.3 测试与门禁

- 全量 **2549 passed, 4 deselected**（本阶段新增 **415** = 396 单元 + 19 属性）。
- 依赖边界 gate 规则升至 **R1–R10**（`RULES_VERSION` 1.4.0，files=214，edges=481）0 违规 0 环；
  `module_ownership.md` 1.5.0 新增 `compiler` 区域与 R9/R10。
- 接口解析：**320 步 / 417 唯一接口 / 693 引用**全部解析。
- 配置审计：12/12 文档加载 + 逐字段审计通过（含 6 个 kind 的集合一致性比对）。

### 17.4 阻塞与交接

**阻塞（3/6 必需前置 + 2 项 advisory）**：

| 前置 | 状态 | 证据 |
|---|---|---|
| S01 契约与身份 | 满足 | `docs/stage_experiments/S01/E01-01/raw/verdict.json` |
| S02 Qwen reference 可重放 | 满足 | `docs/stage_experiments/S02/E02-01/raw/verdict.json` + `configs/models/qwen3_1_7b.yaml` |
| S03/S04 kernel 硬件证据 | **缺失** | 协议树无 `S03/S04/**/verdict.json`，`reports/dev/**/gate*/s04_backend_baseline.json` 亦无 |
| S06 pattern 模型级 correctness | **缺失** | 协议树无 `S06/**/verdict.json` |
| 冻结编译器环境指纹 | **缺失** | 无含 torch/triton 版本的 `environment_fingerprint.json` |
| 独立证据目录 / 稳定 CLI | 满足 | 本层 `hqsb/compiler/experiment.py`、`scripts/compiler/run_e11.py` |
| S05 QuantArtifact（advisory） | 未满足 | 量化分支记 `NOT_APPLICABLE_CAPABILITY` |
| IR/binary 导出工具（advisory） | 未满足 | nvcc/cuobjdump/nvdisasm/nsys/ncu 均 `NOT_RUN_TOOL_UNAVAILABLE` |

**未覆盖边界**：TVM/MLIR 与真实 Qwen/Inductor 运行均未执行（仅接口与 CPU oracle）；
E11-10（P1）未激活；跨机器/跨架构结论未做（本机 1×RTX 3090，`development` 角色）。

S11 报告：`docs/reports/S11_开发报告.md`、`docs/reports/S11_阶段验收报告.md`、
`docs/reports/S11_compiler_architecture.md`。

---

## 18. S12（跨硬件评估与统一 Benchmark）—— 接口/代码层就位，实验层 BLOCKED

> 追加时间：2026-09-19。S12 按任务约束「提供接口、不执行实验」交付 E12-01~E12-10
> 全部 **360 个实验步骤**的能力接口与测试，**未执行任何正式实验、未产出任何
> 可比性/capability/性能/能耗/成本/成熟度/lineage 结论数字**。实验层 `BLOCKED`。

### 18.1 新增模块（`hqsb/evaluation/`，21 个）

1. **基础层（L0）**：`identity.py`（byte/canonical/aggregate 三 hash、版本化 canonicalization、
   逻辑 URI、EntityRef）、`records.py`（协议状态、四态裁决、10 类缺失语义、182 张表 schema、
   §3.2 状态传播为数据、R0–R4 重生成等级）、`contracts.py`（Comparison Contract 六段、
   49 字段分类、16 审计维度、reason codes）、`layers.py`（四层语义/estimand/边界/token accounting/
   SLO-goodput/Amdahl）、`campaign.py`（campaign manifest、上游证据五态、状态矩阵、acceptance、目录布局）。
2. **可比性与能力（E12-01/E12-02）**：`candidates.py`、`comparability.py`（四态裁决 + 非法 join 防护 +
   单位/边界/质量依赖/actual-backend 审计 + suite manifest）、`platform.py`（平台身份交叉核对 + telemetry
   字段 canonical 化）、`capability.py`（69 feature + 四级证据 + 升级/失效 + coverage join + 负向探测）。
3. **重放与稳定性（E12-03/E12-04）**：`benchmark.py`（Observation/NormalizedResult/11 条交叉校验/
   run 计划/冷启动/预热/thermal/health）、`repeatability.py`（复制层级/平衡 schedule/预注册阈值/异常规则/
   排除账本/纯 Python 统计/bootstrap/variance/frontier membership）。
4. **解释与成本（E12-05/E12-06/E12-07）**：`roofline.py`（四类 roof + 流量分离 + 预测/残差/消融 +
   calibration/validation 不泄漏）、`energy.py`（四边界 + accumulator/integral + total/incremental +
   compliant 分母 + 10 项质量门）、`cost.py`（无内置价格；价格快照/部署单元/10 项双重计数/敏感性/独立复算）。
5. **决策与成熟度（E12-08/E12-09）**：`pareto.py`（六画像模板 + 硬约束 + NO_FEASIBLE + 决策回归 10 用例 +
   成熟度三合法路径）、`maturity.py`（8 维锚定 rubric + 四类工时 + silent fallback 高风险 + 分歧不平均）。
6. **lineage 与脚手架（E12-10）**：`lineage.py`（entity 追加式/DAG 校验/reverse trace/fault 定位/per-transform diff）、
   `telemetry.py`（C6/C7 投影 + 表校验 + 覆盖审计）、`specs.py`（12 份配置 + 逐字段审计）、
   `experiment.py`（前置门/预注册/Evidence Manifest/RunDirectory/三重门拒绝）、`interface_map.py`（360 步对照）。

### 18.2 配置与驱动

- `configs/evaluation/*.yaml` 12 份冻结配置（comparability/capability/benchmark/repeatability/roofline/
  energy/cost/pareto/maturity/lineage/campaign/experiment），严格键校验 + 与代码逐字段审计全绿；
  只冻结**词汇与结构**，价格/电价/汇率一律是 campaign 输入，不内置。
- `scripts/evaluation/run_e12.py`：`--list/--prerequisites/--interface-map/--spec-audit/--smoke/--execute`，
  默认拒绝产结论；`--prerequisites` 实测 4 条必需中 3 条未满足。

### 18.3 测试与门禁

- 全量 **2922 passed, 4 deselected**（本阶段新增 **373** = 319 单元 + 38 属性 + 16 边界/脚手架）。
- 依赖边界 gate 规则升至 **R1–R12**（`RULES_VERSION` 1.6.0，files=236，edges=550）0 违规 0 环；
  `module_ownership.md` 1.6.0 新增 `evaluation` 区域与 R11/R12。
- 接口解析：**360 步 / 415 唯一接口 / 727 引用**全部解析。
- 配置审计：12/12 文档加载 + 逐字段审计通过。
- 驱动 smoke：十模块 CPU 自检全绿并标注 `claim_allowed=false`（非实验）。

### 18.4 阻塞与交接

**阻塞（3/4 必需前置 + 3 项 advisory）**：

| 前置 | 状态 | 证据 |
|---|---|---|
| S01 契约与身份 | 满足 | `docs/stage_experiments/S01/E01-01/raw/verdict.json` |
| S02 ModelArtifact/WorkloadSpec/质量门 | 满足 | `docs/stage_experiments/S02/E02-01/raw/verdict.json` + `configs/models/qwen3_1_7b.yaml` |
| S03–S11 上游证据链（≥2 阶段 verdict） | **缺失** | 协议树仅 S01/S02 有 verdict |
| 多硬件覆盖（≥3 硬件，或 2 硬件+2 架构） | **缺失** | 无 ≥3 平台实例/架构证据 |
| 冻结评估环境指纹 | **缺失** | 无含 torch/arch 的 `environment_fingerprint.json` |
| 独立证据目录 / 稳定 CLI | 满足 | 本层 `hqsb/evaluation/campaign.py`、`scripts/evaluation/run_e12.py` |
| profiler/导出工具（advisory） | 未满足 | nvcc/nsys/ncu 等 `NOT_RUN_TOOL_UNAVAILABLE` |
| 功率计能力证据（advisory） | 未满足 | energy 只能到接口层 |
| 价格快照（advisory） | 未满足 | cost 只能到接口层 |

**未覆盖边界**：无真实多硬件/多架构实验；energy/cost/成熟度均停在接口层；`artifacts/S12/` 目录约定
落到 `experiment_results/S12/`（仓库既有约定，见开发报告 §9 漂移登记）。

S12 报告：`docs/reports/S12_开发报告.md`、`docs/reports/S12_阶段验收报告.md`、
`docs/reports/S12_cross_hardware_design.md`。

---

## 19. S13（生产化、云原生与可靠性）—— 接口/代码层就位，实验层 BLOCKED

**范围**：`hqsb/infra/`（19 个模块）+ `configs/infra/`（13 份冻结词汇表）+
`scripts/infra/`（驱动 + 2 个生成器）+ `infra/`（容器/Helm/可观测/CI/runbook 模板）。
11 项实验（E13-01…E13-11）共 **418 个步骤**全部有接口落点。

**已交付能力**（每项都有测试证明，详见 `docs/evidence_ledger.md` §18）：

| 层 | 模块 | 能力摘要 |
|---|---|---|
| 身份与 gate | `identity` / `records` / `contracts` / `campaign` | ReleaseBundle 与 OCI digest DAG 身份、可重建四级、三套状态机、§23 十四条交叉约束（V01–V14）、§21 目录与 §20.2 安全策略与 target 授权 |
| E13-01 | `supply_chain` | 分层镜像闭包、SBOM 完整性、漏洞/严重度/可达性/例外、秘密与模型命中即 FAIL、许可证、provenance/attestation subject 闭合、非 root/只读/capability、构建上下文负例 |
| E13-02 | `deployment` | clean level（L1/L2/L3）、污染检测、部署状态机、readiness 九条件语义、pre-ready 流量排除、首请求正确性、cold/warm 分离、失败清理、自动化判定 |
| E13-03 | `scheduling` | PlacementPlan 硬/软约束、设备清单与 capability label 信任（TTL/owner/保护）、filter/score/bind 链路、NUMA/拓扑/链路/P2P、共享三模式与隔离层、越权与跨租户失败 |
| E13-04 | `artifacts` | content-addressed cache key、staging→verify→原子提交（marker 之后于 durable）、兼容性 gate、lease/pin 与 GC 安全、激活/切换/回滚、混版本与 KV 隔离、故障注入 |
| E13-05 | `lifecycle` | 三类 probe 语义（startup/readiness/liveness 分离且不与负载耦合）、终止预算、drain 时间线与 post-drain 拒绝、请求/令牌完整性、retry/duplicate 账、rolling/PDB、forced kill 记账、资源释放 |
| E13-06 | `capacity` | 逐组件/逐 rank 内存账本、KV 估算（block rounding/prefix/Metadata）、token_work、六类 admission 策略与 reason code、reserve/commit 原子性、residual/false decision/margin holdout、S12 回流 |
| E13-07 | `autoscaling` | 控制环路时延、desired replicas 与 clamp、metric age fail-safe、stabilization/rate limit/cooldown、控制指标（过冲/波动/settling）、成本、降载保护、失败矩阵 |
| E13-08 | `observability` | 语义约定（版本化/单位/三层边界）、基数与禁用标签、采样与保留、SLI/diagnostic 分离、直方图 bucket、metric↔raw 对账、告警（impact/query/owner/runbook/阈值来源）、RCA（层级/范围/置信度/备选）、遥测缺失与脱敏 |
| E13-09 | `faults` | 故障合同（hypothesis/targets/blast radius/预期/abort/kill switch/repetitions/claim 边界）、ground truth、MTTD/MTTM/MTTR、降级分类、retry/fallback 有界、恢复三轴不变量、残留观察、postmortem 与裁决 |
| E13-10 | `canary` | 状态机（禁跳阶段）、G0–G8 gate（硬门优先）、session 粘性分配、序贯决策与信息量、流量/预算分阶段、case-mix 可比性、rollback 闭环、false rate 与 holdout、override 审计 |
| E13-11 | `security` | 威胁模型（能力/非目标/fail policy）、租户映射、RBAC 权限图与间接提权、配额账本与并发竞态、滥用矩阵、噪声邻居、遥测脱敏、审计覆盖、12 条不变量裁决、结论措辞边界 |
| 证据与脚手架 | `telemetry` / `specs` / `experiment` / `interface_map` | §22 七投影 + 93 表 schema 校验、13 份配置严格加载与逐字段审计、三重门（execute/前置/raw）拒绝产结论、418 步接口解析 |

**测试与门禁（本机实测）**：

- 全量：**3118 passed, 4 deselected**（S12 基线 2922；S13 专用新增测试 **195** 个 = 153 单元 + 42 属性）；
- 依赖边界门：`rules=1.7.0 files=256 modules=256 edges=599 violations=0 cycles=0 status=PASS`
  （新增 R13「下层不得 import infra」、R14「infra 不得 import ops」）；
- 接口解析：`{steps: 418, interfaces: 343, references: 638, ok: true}`；
- 配置审计：13/13 文档加载 + 逐字段一致；生成物与代码同步（`--check` 通过）；
- 前置门：`run_e13.py --prerequisites` → `satisfied=false`（缺 S08/S12 两条必需）；
- lint（新增范围）：`All checks passed`；文档链接：43 broken（**既有债务**，未新增）。

**未覆盖边界（BLOCKED 与限制）**：

1. 必需前置 `s08_service_contract`、`s12_capacity_and_quality_baseline` 未满足 →
   实验层 `BLOCKED`，E13-01…E13-11 无 run / 无 raw / 无结论数字；
2. 本机无 docker/kubectl/helm/cosign/syft/grype/promtool，无测试集群、注册表与遥测后端；
3. `infra/**` 资产为模板（`IMPLEMENTED_UNVERIFIED` / `DESIGN_ONLY`），未构建/未渲染/未部署/未演练；
4. `configs/infra/**` 只冻结词汇与结构，**不含测量值**；阈值/副本/流量比例必须由 campaign 冻结；
5. §21 逻辑目录（`artifacts/S13/<campaign>/`）落到仓库既有物理根 `experiment_results/S13/`；
6. E13-11 的 11/12 条租户不变量当前为 `NOT_RUN`（无多身份环境）；
7. mypy 未覆盖 `hqsb/infra`（CI 中 mypy 亦为注释状态，仅计划 `hqsb/core`）。

**报告**：`docs/reports/S13_开发报告.md`（含「实验步骤 → 代码接口」对照表）、
`docs/reports/S13_阶段验收报告.md`（代码层验收 vs 实验层 BLOCKED）、
`docs/reports/S13_production_architecture.md`（Design 制品：5 平面、13 项决策、回退方式）、
`docs/reports/S13_interface_map_generated.md`（418 步逐条对照，生成物）。

## 20. S14（训推协同与前沿扩展）—— 接口/代码层就位，实验层 BLOCKED

S14 已交付 E14-01～E14-05、E14-F1～F4、E14-06～08 共 **12 项实验、480 个实验步骤**的能力接口
（`hqsb/experimental/` **21 个模块，17,328 行**）。每个实验都有可调用的驱动入口
（`scripts/experimental/run_e14.py --experiment <id>`），且**默认不产出结论**。

**职责**：把"训练 → artifact → 推理/服务"的**证据链**做成统一契约与执行脚手架——
canonical 身份与 seed bundle、S14 词汇表与四个状态机、§7 七个统一证据对象、
§22 运行目录布局与执行安全策略、依赖与 feature flag 边界（E14-01）、
分布式训练状态与 checkpoint（E14-02）、转换 DAG 与**七层训推一致性门**（E14-03）、
SFT/DPO/GRPO **手算 oracle** 与 policy staleness（E14-04）、前沿 ADR 与预注册（E14-05）、
四个条件 P0 分支（E14-F1 speculative/MTP、F2 MoE、F3 long-context、F4 sparsity）、
三个可选迁移（E14-06 多模态、E14-07 Agent、E14-08 端侧）、C6/C7 投影、
冻结词汇表审计、三重门脚手架与 **480 步接口对照表**。

**依赖方向**：只依赖 `hqsb.core`。gate **R15** 禁止 13 个下层区域 import `hqsb.experimental`；
gate **R16** 禁止 `hqsb.experimental` import `ops`，且禁止模块级 `torch`/`triton`/`numpy`/
`transformers`/`ray`/`vllm`（必须函数内惰性探测）——因子进程探针实测：`import hqsb.experimental`
拉入重框架数 = **0**。

### 20.1 新增模块（`hqsb/experimental/`，21 个）

| 模块 | 行数 | 职责 |
|---|---|---|
| `__init__.py` | 135 | PEP 562 惰性 `_LAZY`；`STAGE`/`EXPERIMENTS`/`PRIMARY_CHAIN` 等顶层常量 |
| `identity.py` | 448 | canonical digest（非有限数拒绝）、seed bundle（7 角色）、rank 身份、lineage DAG（环检测）、文件 inventory |
| `records.py` | 881 | 状态/成熟度/adoption/capability 词汇表 + **4 个状态机** + **40 张表 schema** |
| `contracts.py` | 871 | §7 七个统一证据对象（TrainingRun/Checkpoint/ServingModel/PolicySnapshot/Trajectory/FrontierStudyContract/AdoptionDecision）+ 跨实验不变量 |
| `campaign.py` | 346 | §22 运行目录布局、执行安全策略（30 条禁止项、11 条隔离要求）、预算校验、可提交文件白名单 |
| `dependencies.py` | 1454 | **E14-01**：extra↔flag↔capability↔CLI 映射、wheel 元数据/内容审计、导入纯度、flag 解析与冲突、10 类负例、SBOM 边界 |
| `training.py` | 959 | **E14-02**：global batch 等价、loss 重归一化、逐层对账、collective/显存账本、checkpoint 完整性、resume 连续性、故障注入 |
| `parity.py` | 790 | **E14-03**：转换 DAG、显式 mapping、**七层语义门**、adapter merge、6 类错误制品、原子发布与 cache |
| `posttraining.py` | 1007 | **E14-04**：SFT/DPO/GRPO oracle、mask/长度归一化、logprob 一致性、policy lineage、有界异步队列、staleness 扫描 |
| `frontier.py` | 1254 | **E14-05**：文献注册表、论文条件矩阵、estimand、硬前置门、单一 primary、AdoptionDecision 预注册、协议 hash 冻结 |
| `speculative.py` | 909 | **E14-F1**：accept/residual 手算、逐位置接受率、cycle 成本重建、break-even、低 acceptance 保护 |
| `moe.py` | 1126 | **E14-F2**：router top-k、CV/Gini/max-mean、A2A 与 GEMM 账本、per-rank 分解、total vs active 显存 |
| `long_context.py` | 911 | **E14-F3**：KV payload/metadata 分离、无截断证明、chunk 边界与 position/mask、混部公平性 |
| `sparsity.py` | 1087 | **E14-F4**：N:M compliance、算法误差 vs 实现误差、actual dispatch、metadata 成本、Amdahl 残差 |
| `multimodal.py` | 941 | **E14-06**：阶段状态机、子模型 identity、任务原生指标、预处理进 E2E、三态计时 |
| `agent.py` | 1157 | **E14-07**：13 状态工作流、tool 校验先于执行、critical path、9 类故障注入 |
| `edge.py` | 1216 | **E14-08**：路线 A/B、delegate partition、cold/warm/sustained、MAP_ONLY 约束、技术地图 |
| `telemetry.py` | 459 | C6/C7 扩展字段 + 10 个投影器 + 覆盖度报告；模态指标拒绝 LLM 单位 |
| `specs.py` | 397 | `configs/experimental` 审计（禁测量值、禁绝对路径、跨文档一致性） |
| `experiment.py` | 625 | 三重门脚手架、`EvidenceManifest`、前置检查（仓库 + 可选机器探测） |
| `interface_map.py` | 355 | **480 步 → 993 条引用 → 589 个唯一接口** 的解析与校验 |

### 20.2 配置与驱动

| 制品 | 位置 | 说明 |
|---|---|---|
| 实验驱动 | `scripts/experimental/run_e14.py`（357 行） | `--list` / `--prerequisites` / `--interface-map` / `--objects` / `--spec-audit` / `--spec-check` / `--smoke` / `--experiment`；**默认拒绝产出结论** |
| 词汇表生成器 | `scripts/experimental/gen_experimental_specs.py`（547 行） | `--write` / `--check`；8 份 YAML 的单一事实源 |
| 接口图生成器 | `scripts/experimental/gen_interface_map.py`（121 行） | 生成并校验 `S14_interface_map_generated.md`；不完整映射拒绝落盘 |
| 冻结词汇表 | `configs/experimental/*.yaml`（8 份） | `dependency-policy` / `extras` / `failure-semantics` / `environment-matrix` / `frontier-branches` / `adoption-rules` / `statistics-plan` / `profiling-contract`。**不含任何测量值、不含本机绝对路径** |
| 宿主加固（本机，不入库） | `scripts/env/patch_vscode_server_heap.sh` | 扩展宿主堆上限注入；`--check` / `--revert` / `VSCODE_HEAP_MB`；见 `AGENTS.md` §9 |

### 20.3 测试与门禁

| 项 | 结果 |
|---|---|
| 全量 CPU 测试 | **3221 passed, 4 deselected** |
| S14 专项测试 | **103 passed**（`tests/unit/experimental/` 3 个文件 89 项 + `tests/property/test_experimental_invariants.py` 14 项） |
| 依赖边界门 | `rules=1.8.0 files=277 modules=277 edges=667 violations=0 cycles=0 status=PASS`（新增区域 `experimental` + 规则 **R15/R16**） |
| 480 步接口解析 | `experiments=12 steps=480/480 interfaces=589 references=993 ok=True`；`steps_without_interfaces=[]` |
| 词汇表审计 | `kinds=8 missing_kinds=[] ok=True cross_document_problems=0` |
| lint | `All checks passed!`（本阶段文件） |
| 文档链接 | 本阶段新增文件 **0 断链**；报告 43 处断链全部落在既有文件 |
| 三重门 | `write_verdict` 在无 `--execute`／前置未满足／无 raw samples 时**拒绝**写结论（有测试证明） |

### 20.4 阻塞与交接

**实验层全部 `BLOCKED`**：必需前置 `upstream_verdicts`、`experimental_environment`、
`holdout_isolation` 未满足；`distributed_launcher`/`second_device`/profiler 未解析；
E14-03/04/05/F*/06/07/08 的上游链未闭环。**12 项实验无任何 run、无 raw、无结论数字。**

交接给 S15：
- 可消费：480 步接口对照、依赖/CI 边界证据、七个证据对象与 manifest 字段、冻结词汇表、前置阻塞的精确原因与复跑命令；
- **不可**消费：任何"已支持/已加速/已验证"的 S14 声明；论文/设计文档中的预期收益；
  `N/A_BY_ADR` 形式的范围缩减（该 ADR 本身未执行）。

未覆盖边界：① 模块成熟度上限 `SOURCE_INTEGRATED`；② 未实现真实 trainer/engine/device 调用
（R16 有意禁止 import `ops`）；③ E14-06/07/08 未做 ADR 决策，状态为**未执行**而非 `N/A_BY_ADR`；
④ 未选中任何前沿分支；⑤ 多模态/Agent/端侧能力**未被声称**；⑥ 不修改 `docs/stage_experiments/**`。

**报告**：`docs/reports/S14_开发报告.md`（含「实验步骤 → 代码接口」对照表 §5、
模块调用示例 §4、未完成项 §8）
`docs/reports/S14_阶段验收报告.md`（代码层验收通过 vs 实验层 BLOCKED，测试标准逐条核验）
`docs/reports/S14_interface_map_generated.md`（480 步逐条对照，生成物）。

---

## 21. S15（发布、开源与求职证据）—— 接口/代码层就位，实验层 BLOCKED

S15 已交付 E15-01～E15-11 共 **11 项实验、495 个实验步骤**的能力接口
（`hqsb/release/` **19 个模块，14,562 行**）。每个实验都有可调用的驱动入口
（`scripts/release/run_e15.py --experiment <id>`），且**默认不产出结论**。

**职责**：把全部技术工作转化为可核验的公开证据——三个冻结对象
（`ReleaseCandidateSnapshot`/`PublicEvidenceBundle`/`FinalAcceptanceDecision`）、
`ClaimRecord` 十门裁决与稳定 ID、`ContributionRecord` 人/Agent/第三方边界、
全局 Claim Ledger/声明扫描/证据门（E15-01）、clean CPU quickstart 契约（E15-02）、
GPU/NPU hero 重放与 Amdahl 预测（E15-03）、可执行文档/双语一致性/能力矩阵（E15-04）、
release 供应链/provenance/SBOM/许可证（E15-05）、图表→raw lineage 与重生成（E15-06）、
demo 故障注入与诚实降级（E15-07）、3/10/30 分钟讲述与对抗问答（E15-08）、
第三方 clean-room 复现（E15-09）、真实上游贡献（E15-10）、目标读者首屏研究（E15-11）、
**四重门**实验脚手架与 **495 步接口对照表**。

**依赖方向**：只依赖 `hqsb.core`。gate **R17** 禁止 14 个下层区域 import `hqsb.release`；
gate **R18** 禁止 `hqsb.release` import `ops`，且禁止模块级 `torch`/`triton`/`numpy`/
`requests`/`httpx`（必须函数内惰性探测）——因子进程探针实测：`import hqsb.release`
拉入重框架数 = **0**。

### 21.1 新增模块（`hqsb/release/`，19 个）

| 模块 | 行数 | 职责 |
|---|---|---|
| `__init__.py` | 136 | PEP 562 惰性 `_LAZY`；`STAGE`/`EXPERIMENTS`/`P0_EXPERIMENTS`/`STAGE_GATES` |
| `identity.py` | 559 | canonical digest（NaN 拒绝）、稳定 claim id（事实签名）、证据等级偏序、本地 URI 拒绝、文件 inventory |
| `records.py` | 1476 | 六态状态 + claim/finding/help/复现/disposition/demo 词汇表 + **11 个实验 record schema** + §22 数据包 |
| `contracts.py` | 932 | 三个冻结对象 + `ClaimRecord` 十门 + `ContributionRecord` + 可信度乘积规则 |
| `campaign.py` | 376 | `artifacts/S15` 布局、写保护（协议树只读）、执行安全（20 条禁止项/8 条隔离） |
| `experiment.py` | 758 | **四重门**脚手架、`EvidenceManifest`、前置检查（仓库 + 机器探测） |
| `telemetry.py` | 415 | ActionLog、time-to-event（小样本不外推）、EvidenceLookupTrial、DetectorMetrics、FindingTracker |
| `specs.py` | 334 | `configs/release` 审计（禁测量值/禁绝对路径/未知键/跨文档一致性） |
| `claims.py` | 1436 | **E15-01**：公开面 inventory、四类检测器、归一化、绑定、EvidenceGraph、URI/digest/重建、orphan/stale/冲突/单位/外推、负对照、渠道渲染、冻结 |
| `quickstart.py` | 880 | **E15-02**：契约、支持矩阵、session、八段时钟、干预分级、环境证据、canonical 比较、隐私扫描、负例、逐门裁决 |
| `hero_replay.py` | 977 | **E15-03**：环境指纹/可比性、设备健康、profiler、身份、capability、正确性矩阵、actual path、confirmatory 计划、Amdahl、blocked 估计、一致性裁决、stop 规则 |
| `docs_gate.py` | 1163 | **E15-04**：五 inventory、链接/anchor/命令/Schema/CLI/矩阵/双语/版本/隐私/可访问性逐门、缺陷归因 |
| `supply_chain.py` | 1091 | **E15-05**：版本语义、build definition、manifest、provenance、SBOM、漏洞 disposition、许可证/权利、秘密/PII、重复构建、消费验证、撤回 |
| `figures.py` | 813 | **E15-06**：FigureSpec、抽样（冻结 seed）、lineage 六层、聚合/实验单位/误差线、D0–D5 差异等级、Pareto、ACCESS_FRICTION |
| `demo.py` | 731 | **E15-07**：目标、段预算、状态机、测量状态、故障矩阵、演练、录屏 manifest |
| `narrative.py` | 562 | **E15-08**：契约、事实卡、岗位矩阵、三种时长内容门、题库、六维 rubric、FAQ/简历 |
| `clean_room.py` | 619 | **E15-09**：契约、独立性声明、received materials、HelpLog（L0–L4）、复现层级矩阵、修复复测 |
| `upstream.py` | 522 | **E15-10**：边界归因、最小复现、IssueDraft、PatchDesign、ReviewRound、十一维质量门、disposition |
| `first_impression.py` | 480 | **E15-11**：契约、参与者画像、任务集、答案 key、编码、误解、角色 block 分析、复测 |
| `interface_map.py` | 302 | **495 步 → 796 引用** 解析与校验（dataclass 字段引用可解析） |

### 21.2 配置与驱动

| 制品 | 位置 | 说明 |
|---|---|---|
| 实验驱动 | `scripts/release/run_e15.py`（341 行） | `--list`/`--prerequisites`/`--interface-map`/`--objects`/`--spec-audit`/`--spec-check`/`--smoke`/`--experiment`；**默认拒绝产出结论** |
| 词汇表生成器 | `scripts/release/gen_release_specs.py`（346 行） | `--check` 漂移校验；8 份 YAML 单一事实源 |
| 接口图生成器 | `scripts/release/gen_interface_map.py`（83 行） | 生成并校验 `S15_interface_map_generated.md`；不完整映射拒绝落盘 |
| 冻结词汇表 | `configs/release/*.yaml`（8 份） | `claim-taxonomy`/`detector-patterns`/`evidence-package`/`documentation-policy`/`release-policy`/`figure-audit`/`demo-narrative`/`reproduction-upstream`。**不含任何测量值、不含本机绝对路径** |

### 21.3 测试与门禁

| 项 | 结果 |
|---|---|
| 全量 CPU 测试 | **3293 passed, 4 deselected** |
| S15 专项测试 | **72 passed**（`tests/unit/release/` 3 文件 40 项 + `tests/property/test_release_invariants.py` 28 项 + 边界 4 项） |
| 依赖边界门 | `rules=1.9.0 files=297 modules=297 edges=722 violations=0 cycles=0 status=PASS`（新增区域 `release` + 规则 **R17/R18**） |
| 495 步接口解析 | `steps=495/495 references=796 ok=True steps_without_interfaces=0` |
| 词汇表审计 | `kinds=8 missing_kinds=[] duplicate_kinds=[] ok=True cross_document_problems=0` |
| 驱动 smoke | `status=smoke claim_allowed=False`（11 实验 + 8 基础设施模块全通过） |
| lint | `All checks passed!`（本阶段文件） |
| 文档链接 | 本阶段新增文件 **0 断链**；43 处断链全部落在既有文件 |
| 四重门 | `write_verdict` 在无 `--execute`／前置未满足／无 raw samples／无 candidate+ledger 时**拒绝**写结论 |

### 21.4 阻塞与交接

**实验层全部 `BLOCKED`**：`release_candidate`/`claim_ledger`/`public_artifact`（G0 未建立）、
`reviewer_resource`/`participant_resource`/`upstream_authorization`/`release_authorization`（外部资源/授权缺失）、
`accelerator_device`/`documentation_builder`/`clean_environment_builder`/`scanner_tooling`/`recording_tooling`/
`network_access`（未探测）。**11 项实验无任何 run、无 raw、无结论数字。**

未覆盖边界：① 模块成熟度上限 `SOURCE`+`TEST`；② 不真正构建/发布/联系上游/跑加速器（R18 有意禁止 import `ops`）；
③ E15-10 上游贡献、E15-09 外部复现、E15-11 参与者研究均未执行（不是 `N/A_BY_ADR`，是 `BLOCKED`）；
④ 未声称任何 `RUNTIME` 及以上结论；⑤ 不修改 `docs/stage_experiments/**`。

**报告**：`docs/reports/S15_开发报告.md`（含「实验步骤 → 代码接口」对照表 §5、
模块调用示例 §4、未完成项 §8）
`docs/reports/S15_阶段验收报告.md`（代码层验收 vs 实验层 BLOCKED，测试/验收标准逐条核验）
`docs/reports/S15_interface_map_generated.md`（495 步逐条对照，生成物）。
