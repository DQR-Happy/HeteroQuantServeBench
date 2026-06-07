# HQSB 项目现状报告（Project Status）

> 生成时间：2026-08-17
> 跨架构补验：2026-09-18（RTX 3090 / sm_86）
> 基线 Commit：`9b403aa`（`chore: stop tracking the stage-experiment tree`，2026-06-02）
> 当前阶段：S06（框架集成与图优化）—— **接口/代码层就位；实验层 BLOCKED**（S04.5 M4 与 S05 P0 前置缺失，见 §4、§13）
> 下一阶段：S04.5（真实模型算子回接）→ S05 实验执行 → S06 实验执行
>
> 说明：本报告正文生成于 `4dda6f8`，其后仓库推进到 `9b403aa`（HEAD，tracked 343
> 文件），S05/S06 两轮交付在其上追加；历史条目保留不改，新增章节见 §12/§13。

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
| 当前所处阶段 | **S04（Triton、CUTLASS/CuTe 与 Kernel DSL）** |
| 判定依据 | S03 已完成验收；`ops` 增加 Triton/cuBLAS 对照 + 统一 dispatcher/capability 并通过测试 |
| 前序阶段 | S00 → S01 → S02 → S03 —— 均已完成 |

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
| S07–S15 | Runtime / Serving / Ascend / 分布式 / 编译 / 跨硬件 / 云原生 / 训推 / 发布 | 空目录或纯规划 |

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
