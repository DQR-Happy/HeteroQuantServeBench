# HQSB 项目现状报告（Project Status）

> 生成时间：2026-08-16
> 基线 Commit：`4dda6f8`（`refactor docs into staged roadmap and architecture spec`）
> 当前阶段：S03（CUDA 算子性能工程）—— 已完成

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
| 当前所处阶段 | **S03（CUDA 算子性能工程）** |
| 判定依据 | S02 已完成验收；`ops/cuda` 重构为算子库（RMSNorm V0/V1/V2 + fused + dispatcher）并通过测试 |
| 前序阶段 | S00（现状审计）→ S01（核心契约）→ S02（模型基线/Profiling）—— 均已完成 |

S03 将 S00 的单文件 RMSNorm baseline 重构为工业级 CUDA Operator Lab：多版本、
stream-aware 无隐藏分配 C/C++ API、dispatcher、CPU reference、correctness 测试
与 benchmark，并对第二热点（Residual+RMSNorm fusion）做同深度处理。详见 §7
（S03 补齐内容）。

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
| `quant/` `serving/` | 空目录（S04–S08 规划占位） | Planned |

### 3.2 `ops/`（算子）

| 路径 | 内容 | 状态 |
|---|---|---|
| `cuda/common/cuda_check.cuh` | 公共 CUDA 错误检查宏 | Implemented |
| `cuda/common/test_util.h` | 轻量断言测试框架 | **S03 新增**，Implemented |
| `cuda/common/test_metrics.h` | 5 项数值对比指标 | **S03 新增**，Implemented |
| `cuda/device_query/` | 设备信息查询（.cu + CMakeLists） | Verified |
| `cuda/rmsnorm/` | **RMSNorm 算子库**：V0 shared / V1 warp shuffle / V2 vectorized + dispatcher + reference + test + bench | **S03 重构**，Verified（CTest 通过） |
| `cuda/fused_residual_rmsnorm/` | **第二热点算子**：fused residual+rmsnorm（V0/V1）+ test + bench | **S03 新增**，Verified（CTest 通过） |
| `triton/` `ascend/` | 空目录（.gitkeep） | Planned |

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
| `backends/` `quantization/` | 空（.gitkeep） | Planned |

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
| `correctness/` `integration/` `test_vectors/` | 空（.gitkeep） | Planned |

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
| `reports/` | raw 运行证据（**gitignored，仅本机保留**） | runtime-verified |

---

## 4. 阶段完成度映射

| 阶段 | 名称 | 状态 |
|---|---|---|
| S00 | 现状审计与基线恢复 | **已完成（验收通过）** |
| S01 | 核心契约与工程质量 | **已完成（验收通过）** |
| S02 | 模型基线与全栈 Profiling | **已完成（验收通过）** |
| S03 | CUDA 算子性能工程 | **已完成（验收通过）** |
| S04 | Triton / CUTLASS / Kernel DSL | 未开始（RMSNorm 教学闭环可作为 Triton 对照） |
| S05–S15 | 量化 / 框架集成 / Runtime / Serving / Ascend / 分布式 / 编译 / 跨硬件 / 云原生 / 训推 / 发布 | 空目录或纯规划 |

> 结论：S03 已完成 RMSNorm V0→V1→V2 多版本 + dispatcher + reference + 测试 +
> benchmark，显著加速（V2 +56%~142%）与三个真实退化均可用硬件指标解释。第二
> 热点（fused residual+rmsnorm）完成多版本与 RAW 依赖退化分析。GEMM 主热点按
> "不替代 cuBLAS/CUTLASS" 约定留待 S04（Triton/CUTLASS DSL）与 S05（低比特量化）。

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
   - 公共 API `hqsb/rmsnorm.h`：stream-aware、无隐藏分配、无 Python 依赖
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

## 9. 风险与注意事项

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

## 10. 下一步（S04 输入）

S03 完成后进入 S04（Triton / CUTLASS / Kernel DSL）：
- RMSNorm 已有 CUDA 多版本 baseline + dispatcher，可作为 Triton 对照的
  correctness/performance 参考
- GEMM 主热点（decode ~78%）按约定不自研通用 GEMM，改用 CUTLASS/Triton 建立
  对照并优化 layout/epilogue/shape selection
- FP16 half8（float4 装 8 个 half）与 fused 寄存器驻留优化作为 S04 可选的
  Kernel DSL 教学案例

S03 handoff 与验收见 [`reports/S03_阶段验收报告.md`](reports/S03_阶段验收报告.md)。
