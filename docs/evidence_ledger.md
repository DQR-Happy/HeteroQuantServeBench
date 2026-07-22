# HQSB Evidence Ledger（证据台账 / Claim Ledger）

> 本文是分阶段追加的历史台账，以下“当前阶段”仅表示当时记录。2026-09-20 的统一状态见 [现状总览](project_status.md) 和 [全仓库审查报告](audit/全项目审查报告_20260920.md)；具体 raw 的 FAIL/BLOCKED 不被旧汇总中的完成措辞覆盖。

> 生成时间：2026-08-17
> 基线 Commit：`4dda6f8`（历史正文）；当前工作树基线 `b83ebde`（S10 起）／`3c2453e`（历史追加章节）
> 当前阶段：S10（接口/代码层就位，实验层 BLOCKED；§15）
> 追加章节：§10（RTX 3090 / sm_86 跨架构补验）、§11（S05）、§12（S06）、§13（S07）、§14（S08）、§15（S10）

每个声明（claim）按证据强度分级：

- **source-only**：源码存在，但未经测试或运行验证；
- **test-verified**：有 CPU/单元测试覆盖；
- **runtime-verified**：在本机 Jetson 有运行证据（`reports/`，本地保留）；
- **historical-unreproduced**：历史运行过，但当前树无法复现或无保留证据；
- **planned**：仅有规划，未实现。

任何“已完成 / 性能数字”若无法归入前三类，一律降级为 `historical-unreproduced` 或
`planned`。

---

## 1. 声明台账

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| C1 | CUDA device query 可运行，返回非零设备数 | runtime-verified | `ops/cuda/device_query/device_query.cu`；`reports/jetson/20260808_010256/device_query.txt` |
| C2 | RMSNorm V0 correctness PASS（`max_abs_error<=5e-4`） | runtime-verified | `ops/cuda/rmsnorm/rmsnorm_baseline.cu`；`reports/jetson/20260808_010256/rmsnorm_runs.txt` |
| C3 | Qwen3-1.7B local-only 加载成功 | runtime-verified | `hqsb/models/loader.py`；`reports/dev/llm/*/`（load_time_s 字段） |
| C4 | 六 workload model-core baseline 完成且 deterministic | runtime-verified | `reports/dev/llm/20260812_094129/*.json` + `summary.csv` |
| C5 | 模型 SHA256 manifest 校验正确 | test-verified | `hqsb/models/manifest.py`；`tests/unit/test_manifest.py` |
| C6 | hash 不符返回可诊断非零退出 | test-verified + runtime-verified | `scripts/models/verify_qwen3_hashes.py`（退出码 0/1/2）；`tests/unit/test_manifest.py` |
| C7 | 非法 CLI 参数返回非零退出 | test-verified | `hqsb/benchmark/cli.py`；`tests/unit/test_cli.py` |
| C8 | tegrastats 解析与能量积分正确 | test-verified | `hqsb/benchmark/tegrastats_parser.py`；`tests/unit/test_tegrastats_parser.py` |
| C9 | 固定长度 workload 生成严格等于指定 token 数 | test-verified | `hqsb/benchmark/workload.py`；`tests/unit/test_workload.py` |
| C10 | 模型目录缺失 / 关键文件缺失返回可诊断错误 | test-verified | `hqsb/models/loader.py`；`tests/unit/test_loader.py` |
| C11 | golden 数值回归基线（4 份）可作为回归基准 | historical-unreproduced | `benchmarks/workloads/golden/*.json`（缺 `sha256_manifest` 字段、仅 4/6 workload，未进回归门禁） |
| C12 | PyTorch Profiler / Nsight / roofline 热点分析已完成 | planned | 无（属 S02） |
| C13 | CUDA 算子库（多版本/dispatcher/profiler 闭环）已完成 | planned | 无（属 S03） |
| C14 | QuantLab / Runtime / Serving / Ascend / 分布式 / 编译 / 跨硬件已实现 | planned | 空目录（属 S04–S12） |

---

## 2. 证据强度说明

### 2.1 runtime-verified 证据的局限

`reports/` 目录被 `.gitignore` 整体排除，raw JSON/CSV 证据**仅存于本机**，未随
仓库分发。因此：

- 这些声明对“本机”是 runtime-verified；
- 对“新 clone 的另一台机器”是 historical（无法从 Git tree 复现，除非另行归档
  artifact）。

跨机器审计前，需将 raw artifact 以版本化索引或对象存储归档（属 S01/S12）。

### 2.2 性能数字来源

`reports/dev/llm/20260812_094129/summary.csv` 中的关键数字（示例，完整见 CSV）：

| case | ISL | OSL | TTFT(ms) | E2E(ms) | Decode(t/s) | deterministic |
|---|---|---|---|---|---|---|
| tiny | 32 | 16 | ~121 | ~1762 | ~9.1 | True |
| short | 128 | 32 | ~141 | ~3430 | ~9.4 | True |
| balanced | 512 | 128 | ~372 | ~12883 | ~10.2 | True |
| long_prefill | 2048 | 32 | ~2531 | ~5646 | ~9.9 | True |
| decode_heavy | 128 | 256 | ~121 | ~25499 | ~10.0 | True |
| long_balanced | 2048 | 128 | ~2546 | ~16561 | ~9.1 | True |

这些数字的环境/版本绑定见各 JSON 的 `hardware`/`software` 字段（Orin, sm_87,
PyTorch 2.5.0a0+nv24.08, CUDA 12.6, Transformers 5.8.0, ModelScope 1.29.0）。
数字均为 **model-core** 口径，不含 tokenizer/HTTP/queue/network（见
`benchmark/methodology.md` 与 `metric_definitions.md`）。

---

## 3. S01 声明台账

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| S1-1 | C1–C7 契约定义完整且未知字段/缺字段拒绝 | test-verified | `hqsb/core/contracts/`；`tests/unit/core/test_contracts.py` |
| S1-2 | Schema 版本化：迁移链 + 未来版本拒绝 | test-verified | `hqsb/core/schema/versioning.py`；`test_schema_versioning.py` |
| S1-3 | 配置分层 precedence 与 hash 稳定性 | test-verified | `hqsb/core/config/loader.py`；`test_config.py` |
| S1-4 | Registry 注册/冲突/卸载 | test-verified | `hqsb/core/registry/registry.py`；`test_registry.py` |
| S1-5 | 错误分类 + 稳定 exit code | test-verified | `hqsb/core/errors.py`；`test_errors.py` |
| S1-6 | Dummy backend 仅凭 Contract 注册/运行/写结果 | test-verified | `hqsb/backends/dummy.py`；`test_dummy_backend.py` |
| S1-7 | `core` 不依赖任何具体 backend/ops/model | test-verified | `tests/unit/core/test_dependency.py`（AST 静态扫描） |
| S1-8 | legacy golden/result → C6 迁移 | test-verified + runtime-verified | `hqsb/core/schema/migrate.py`；`test_migration.py`；`scripts/migrate_legacy.py` 实测退出码 0 |
| S1-9 | RMSNorm metadata → C3 OperatorSpec | test-verified | `configs/operators/rmsnorm_v0.json`；`test_operator_spec_example.py` |
| S1-10 | CPU 最小包打包（无 torch/CUDA 依赖） | runtime-verified | `pip wheel . --no-deps` 成功（S01 时 wheel 38 文件）；E00-06 @ e4a031c 复跑成功：hqsb-0.1.0-py3-none-any.whl（58 entries，sha256 b1ddaaa2…），log `docs/stage_experiments/S00/E00-06/raw/rerun/s1_10_pip_wheel.stdout` |
| S1-11 | 文档相对链接无断裂 | runtime-verified | `scripts/check_docs.py` 退出码 0 |

---

## 4. S02 声明台账

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| S2-1 | Roofline/Amdahl 数学正确 | test-verified | `hqsb/benchmark/roofline.py`；`tests/unit/core/test_roofline.py` |
| S2-2 | golden/determinism/首错位定位正确 | test-verified | `hqsb/benchmark/correctness.py`；`test_correctness.py` |
| S2-3 | YAML 六 workload 单一事实源加载 | test-verified | `hqsb/benchmark/workload_config.py`；`test_workload_config.py` |
| S2-4 | KV cache/权重字节核算正确 | test-verified | `hqsb/benchmark/memory.py`；`test_memory.py` |
| S2-5 | operator 表提取（新旧 PyTorch 字段名） | test-verified | `hqsb/benchmark/profiling.py`；`test_profiling.py` |
| S2-6 | PyTorchBackend 契约合规（无权重） | test-verified | `hqsb/backends/pytorch.py`；`test_pytorch_backend.py` |
| S2-7 | Jetson 协议防御式表面 | test-verified | `hqsb/hardware/jetson.py`；`test_jetson.py` |
| S2-8 | FP16 Reference Runtime 端到端可用 | runtime-verified | `reports/dev/llm/s02_smoke_tiny.json`（decode 9.28 tok/s, correctness=true） |
| S2-9 | KV cache 画像（28 layers, 8 kv_heads, 114688 B/token） | runtime-verified | `s02_smoke_tiny.json` + `baseline_report.md` §4 |
| S2-10 | 权重 3.44GB 为内存绝对大头 | runtime-verified | `s02_smoke_tiny.json`（model_weight_bytes=3,441,149,952） |
| S2-11 | Decode GEMM 主导（~78%） | runtime-verified | `reports/dev/profiler/s02/hotspot_summary.json`（真实 CUDA trace） |
| S2-12 | Prefill GEMM ~48%、elementwise ~35–40% | runtime-verified | `reports/dev/profiler/s02/hotspot_summary.json` |
| S2-13 | Hotspot Decision 来自 Profile（非拍脑袋） | runtime-verified | `hotspot_analysis.json` + `pytorch_profile_report.md` §4 |
| S2-14 | 全量测试 238 passed（S02 结束口径） | historical-unreproduced | S02 结束时的 `pytest -q` raw 未归档；当前 HEAD（e4a031c）全量复跑为 **340 passed**（见 S4-13 修正与 §8 E06-1） |

---

## 5. S03 声明台账

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| S3-1 | RMSNorm V0/V1/V2 correctness（33 checks） | runtime-verified | `ops/cuda/rmsnorm/tests/test_rmsnorm.cu`；`ctest` 1/2 passed |
| S3-2 | Fused residual+rmsnorm correctness（15 checks） | runtime-verified | `ops/cuda/fused_residual_rmsnorm/tests/test_fused_residual_rmsnorm.cu`；`ctest` 2/2 passed |
| S3-3 | 5 项数值对比指标（max/mean/RMSE/cosine/L2rel） | test-verified | `ops/cuda/common/test_metrics.h` |
| S3-4 | dispatcher dtype/shape 路由 + 不支持组合 fallback | test-verified | `rmsnorm_dispatcher.cu`；`test_dispatcher_selection` + `test_invalid_arguments` |
| S3-5 | V2 float4 显著加速（+56%~142% vs V0） | runtime-verified | `hqsb_rmsnorm_bench` 输出（`S03_benchmark_report.md` §1） |
| S3-6 | FP16 half2 退化（32-bit 事务） | runtime-verified | S03 `hqsb_rmsnorm_bench`（本机 build/jetson-release/bin）FP16 33.73 GB/s vs FP32 87 GB/s；对应行见 `docs/reports/S03_benchmark_report.md`（数值依赖 nvpmodel/频率，见 E00-06 报告） |
| S3-7 | 小 hidden V2 退化（负载不均） | runtime-verified | S03 小 hidden 扫参 hidden=100：V1 21.67 vs V2 19.10 GB/s；对应行见 `docs/reports/S03_benchmark_report.md`（数值依赖运行状态） |
| S3-8 | block=1024 occupancy 崩塌 | runtime-verified | S03 block sweep：V2 67.51 GB/s @ occupancy=1；对应行见 `docs/reports/S03_benchmark_report.md`（数值依赖运行状态） |
| S3-9 | fused V1 无收益（RAW 依赖） | runtime-verified | `hqsb_fused_residual_rmsnorm_bench`（本机 build/jetson-release/bin）V0~V1≈65 GB/s；对应行见 `docs/reports/fused_residual_optimization_log.md` / `docs/reports/S03_benchmark_report.md` |
| S3-10 | 无隐藏分配/stream-aware API | source-only + test-verified | `ops/cuda/rmsnorm/include/hqsb/rmsnorm.h` 契约 + correctness 通过（原记录路径 `hqsb/rmsnorm.h` 不存在的漂移由 E00-06 修正） |
| S3-11 | memcheck host 泄漏 0 bytes | runtime-verified | E00-06 @ e4a031c 复跑 `compute-sanitizer --tool memcheck --leak-check full`：`PASS test_rmsnorm (35 checks)`，LEAK SUMMARY 0 bytes leaked in 0 allocations（GPU debug 因 L4T 禁用，与 S0x 文档一致）；log `docs/stage_experiments/S00/E00-06/raw/rerun/s3_11_compute_sanitizer.stdout` |
| S3-12 | 对齐陷阱修复（FP16 奇数 hidden / FP32 非 4 倍数） | test-verified | `rmsnorm_v2.cu` scalar-tail 回退 + `test_fp16_non_aligned_fallback` |

---

## 6. S04 声明台账

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| S4-1 | Triton 3.7.1 在 sm_87 可用（实测编译运行） | runtime-verified | S04 探针 `/tmp/triton_probe.py`（最小 kernel max_err=0.0，原 raw 未归档）；E00-06 @ e4a031c 经 `detect_capabilities()` 复验 triton_available=true / triton_version=3.7.1，见 `docs/stage_experiments/S00/E00-06/raw/rerun/s4_capability.json` |
| S4-2 | CUTLASS 4.7.0 在 sm_87 可用（第三方 headers + FP16 GEMM） | runtime-verified | `third_party/cutlass`；`hqsb_cutlass_gemm_bench` 正确性 max_err ~0.03 |
| S4-3 | 能力检测永不抛异常 + 缓存 | test-verified | `tests/unit/ops/test_capability.py` |
| S4-4 | dispatcher 四层策略（capability/arch/shape/fallback） | test-verified | `tests/unit/ops/test_dispatcher.py`（15 用例） |
| S4-5 | CUDA vs Triton RMSNorm 正确性一致（同阈值） | test-verified | `test_triton_rmsnorm.py` + `test_cuda_bridge.py::test_cross_backend_consistency` |
| S4-6 | Triton RMSNorm 部分列丢失 bug 修复（循环覆盖整行） | test-verified | `rmsnorm.py` loop；hidden=2048 BLOCK=1024 正确 |
| S4-7 | Triton `tl.dot` TF32 降精度修复（input_precision=ieee） | test-verified | `gemm.py`；FP32 GEMM 精确对比通过 |
| S4-8 | FP16 RMSNorm 短 kernel 测量波动（两次 run 相反，不具可复现性） | runtime-verified | `bench_s04.py` 两次完整 run 结果相反（诚实修正） |
| S4-9 | FP32 RMSNorm CUDA V2 领先（稳定） | runtime-verified | `bench_s04.py`：0.19–0.22 vs 0.23–0.28 ms（两次一致） |
| S4-10 | GEMM 四方对照（CUTLASS 1×2048×2048 反超 cuBLAS ~2×） | runtime-verified | `bench_s04.py`：cutlass 0.12 vs cublas 0.24 |
| S4-11 | autotune 非全局常量（edge 设备选 BLOCK 非最优） | runtime-verified | `bench_s04.py`：triton_optimized 慢于 triton_reference |
| S4-12 | Triton IR 元数据（RMSNorm 34 regs / GEMM 128 regs，0 spills） | runtime-verified | `dump_triton_ir.py` + `reports/dev/s04/ir/metadata.json` |
| S4-13 | 全量测试 270 passed（S04 结束时新增 32；当前 HEAD 口径见 §8 E06-1） | historical-unreproduced | S04 结束时的 `pytest -q` raw 未归档；E00-06 @ e4a031c 全量复跑 **340 passed**，log `docs/stage_experiments/S00/E00-06/raw/rerun/s2_14_s4_13_pytest.stdout` |
| S4-14 | TileLang 0.1.13 在 sm_87 可用（elementwise add 实测） | runtime-verified | `ops/_tilelang_probe.py`：max_err=0.0 |
| S4-15 | 三个 DSL（Triton/CUTLASS/TileLang）能力检测全通过 | runtime-verified | E00-06 @ e4a031c 复验 `detect_capabilities()`：cuda(8,7)/triton/cutlass/tilelang/cublas 全 True，notes 为空；log `docs/stage_experiments/S00/E00-06/raw/rerun/s4_capability.json` |
| S4-16 | CUTLASS GEMM 正确性（对照 host FP32 参考） | runtime-verified | `bench_cutlass_gemm.cu`：max_err ~0.03（FP16 精度内） |
| S4-17 | HIP/ROCm/OpenCL 迁移路径已记录 | source-only | `docs/architecture/portable_kernel_backends.md` |

---

## 7. 审计检查结果

- **manifest 自引用修复**：原 `model_sha256_manifest.txt` 含自引用行
  （`./model_sha256_manifest.txt`），其哈希无法自洽；已移除该行，实测模型快照
  `14/14 verified`（见 `reports/S00_阶段验收报告.md` §4.1）。
- **`.gitignore` 宽泛匹配修复**：`models/`、`reports/` 曾误忽略
  `hqsb/models/`、`scripts/models/`、`configs/models/`、`docs/reports/`，
  是“模型链路不可复现”的根因；已锚定为 `/models/`、`/reports/`
  （见 `reports/S00_阶段验收报告.md` §4.2）。
- **秘密检查**：仓库工作树无 `.env`/`*.pem`/`*.key`/`id_rsa*` 等秘密文件；
  `.gitignore` 已显式排除。
- **大模型权重**：`*.safetensors`/`*.gguf`/`*.onnx`/`*.engine`/`*.pt`/`*.pth`/`*.bin`
  均被排除，工作树无权重。
- **绝对路径**：默认配置使用 `~/models/hqsb/Qwen3-1.7B`（经 `~` 展开），未硬编码
  机器专属绝对路径；历史 result JSON 内的 `local_path` 为运行时记录，不属默认配置。
- **高风险本地脚本**：`scripts/common/git_commit.sh` 含 `--force` 强推，已被
  gitignore，见 `project_status.md` §8。

---

## 8. S00 E00-06 复验与审计修正（2026-09-04）

> 背景：S00 实验 E00-06（README / Project Status / Evidence Ledger ↔ Git tree
> 漂移审计）在 HEAD `e4a031c` 运行。对下列运行类 claim 在本机复跑取证，并修正
> 缺少 raw artifact 的历史数字口径（S2-14/S4-13 降级为 historical-unreproduced，
> 当前口径见下表）。raw 目录：
> `docs/stage_experiments/S00/E00-06/raw/`（复跑 log 见其 `rerun/` 子目录）。

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| E06-1 | 当前 HEAD（e4a031c）全量 `pytest -q` 复跑 **340 passed**（19.22 s，exit 0） | runtime-verified | `docs/stage_experiments/S00/E00-06/raw/rerun/s2_14_s4_13_pytest.stdout` |
| E06-2 | capability 复验：CUDA(8,7) / Triton 3.7.1 / CUTLASS / TileLang 0.1.13 / cuBLAS 全部可用，notes 为空 | runtime-verified | `docs/stage_experiments/S00/E00-06/raw/rerun/s4_capability.json` |
| E06-3 | compute-sanitizer host 泄漏 0 bytes（PASS 35 checks；GPU debug 受 L4T 限制） | runtime-verified | `docs/stage_experiments/S00/E00-06/raw/rerun/s3_11_compute_sanitizer.stdout` |

修正摘要：`hqsb/rmsnorm.h` → `ops/cuda/rmsnorm/include/hqsb/rmsnorm.h`（S3-10）；
S3-6/7/8/9 补充 tracked 报告指针；S1-10/S3-11/S4-1/S4-15 补充 E00-06 复跑 raw；
S2-14/S4-13 按“缺 raw artifact”降级为 historical-unreproduced。

---

## 9. S00 E00-07 安全边界审计（2026-09-04）

> 背景：S00 实验 E00-07（tracked 文件 secret / 个人绝对路径 / 大权重 / 构建物 /
> raw report 扫描与 `.gitignore` 边界审计）在 HEAD `e4a031c` 运行。§7“秘密检查/
> 绝对路径”的旧结论被收紧：默认配置（YAML `~/` 形式）干净，但发现 **3 处机器专属
> 绝对路径被跟踪**，已就地脱敏（处置见下表）。raw 目录：
> `docs/stage_experiments/S00/E00-07/raw/`。

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| E07-1 | 安全边界扫描工具可重复运行，fixture 自检 5/5（私钥/机器路径/60 MiB 权重/构建物检出，良性源码不误报），可作 CI/发布前门禁 | runtime-verified | `scripts/audit/run_e00_07_repo_security_scan.py`；`docs/stage_experiments/S00/E00-07/raw/self_test.json` |
| E07-2 | 处置后 tracked 无真实 secret / 权重 / 构建物 / ≥50 MiB 文件 / 机器专属绝对路径（post 复扫 high=medium=low=0） | runtime-verified | `.../raw/hits_pre.jsonl`（修复前 3 命中）、`hits_post.jsonl`（0）、`verdict.json`（overall=PASS） |
| E07-3 | 机器专属路径就地脱敏：`jetson_python_lock.txt` 两条本地构建源 → `onnxruntime-gpu==1.20.0` / `torchaudio==2.5.1a0+1661daf` + 行内 sha256/来源注释；`qwen3_model_manifest.json` local_path → `~/models/hqsb/Qwen3-1.7B`，生成器 `dump_model_manifest.py` 同步修复 | runtime-verified | `.../raw/dispositions.json`；工作树变更文件（HEAD 可回退原文） |
| E07-4 | ignore 边界：`check-ignore` 对已 tracked 文件恒为空（Git 语义）；被忽略目录根下 tracked 存量 = `/docs` 47（历史文档）+ `/reports/` 1（`.gitkeep`）均 keep-exempt；tracked 无 raw/log/时间戳 run 数据文件 | runtime-verified | `.../raw/ignore_audit.json`、`gitignore_rules.json` |

与 §7 的关系：§7 对“绝对路径”的审计口径是“默认配置”（YAML/CLI），仍成立；E00-07
把口径扩大到**全部 tracked 内容**（含环境锁与归档 manifest），并就地修正了发现的
3 处机器专属路径。

---

## 10. S04 跨架构补验（RTX 3090 / sm_86，2026-09-18）

> 背景：S04 首次验收（2026-08-17）登记了「仅在 sm_87 运行、多架构未验证」的例外
> （`S04_阶段验收报告.md` §3.3）。本节在第二架构（RTX 3090, sm_86, 82 SM）上补验
> 并关闭该例外。
>
> **证据性质声明**：本节证据来自 **RTX 3090 开发机**，标注为 `development` /
> `smoke` / `cross-architecture validation` / `exploratory benchmark`。
> **不是 Jetson 正式实验结论**，未写入 `reports/jetson/**`，未计算跨硬件 speedup。
> 完整流水线：`reports/dev/rtx3090/20260918_021649/RTX3090_ACCEPTANCE_REPORT.md`；
> 归档 run：`reports/dev/rtx3090/20260917_193004/verdict.json`。
>
> 与 §6 的关系：§6 的 S4-* 条目全部为 sm_87 口径，**原文保持不变**；本节条目为
> sm_86 口径与跨架构结论，两者不可混用。

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| S4-18 | 共享库自报编译架构：新增 C ABI `hqsb_rmsnorm_query_build_arch`，CMake 从 `CMAKE_CUDA_ARCHITECTURES` 解析后经编译宏注入；sm_86 构建返回 `(8,6)`、sm_87 构建返回 `(8,7)` | runtime-verified | `ops/cuda/rmsnorm/CMakeLists.txt`、`ops/cuda/rmsnorm/src/rmsnorm_c_api.cu`；`nm -D` 确认符号导出；SM87 独立构建实测返回 `(8,7)`（日志 `.../gate4_capability/sm87_archcheck.log`） |
| S4-19 | dispatcher 不再硬编码架构：`_CUDA_LIB_ARCH=(8,7)` 已删除，改为「设备实测能力 vs 库自报架构」比较，任一侧未知即不宣称可用 | test-verified + runtime-verified | `ops/dispatcher.py`；`tests/unit/ops/test_dispatcher.py` 新增 5 个双向跨架构用例；sm_86 实跑选出 `cuda/v2_vectorized`（max_abs_err 7.15e-07） |
| S4-20 | **不存在跨架构可移植的全局 CUTLASS tile 配置**：默认 `128×256×64×3` 需 147456 B 动态 smem，sm_87 可用、sm_86（opt-in 101376 B）**无法启动**（`cudaFuncSetAttribute` 失败 → `kErrorInternal`）；`compact` `128×128×64×3`（98304 B）两架构均可行 | runtime-verified | E04-01：`docs/stage_experiments/S04/E04-01/raw/cutlass_feasibility.json`、`cross_arch_comparison.json`；CUTLASS 侧机制 `third_party/cutlass/include/cutlass/gemm/device/gemm_universal_adapter.h:273-282` |
| S4-21 | CUTLASS GEMM 在 sm_86 上按设备共享内存预算**自动选型**并纳入 CTest 正确性门禁 | runtime-verified | `ops/cuda/cutlass_gemm/bench_cutlass_gemm.cu`；`ctest` 3/3 passed（`cutlass_gemm_correctness`）；`--config large` 正确返回退出码 4 |
| S4-22 | 原 CUTLASS 对照实现在 sm_86 上**输出过伪造数据**：内核从未启动仍打印 `median_ms=0.0025` 与毒化内存算出的 `max_err`。已改为失败即非零退出（0/2/3/4） | runtime-verified | 修复前后对照见 `reports/dev/rtx3090/20260918_021649/gate5_backends/cutlass_run.log`；诊断 `/tmp/hqsb_diag`（`status=7(Error Internal)`, `cudaGetLastError=1(invalid argument)`） |
| S4-23 | autotune 非全局常量（sm_86 实测）：同一设备上 GEMM 在 M=1（`64×128×32`）与 M=512（`128×64×32`）选中**不同** tile；RMSNorm 在 fp32/fp16 间切换 `BLOCK`（512 / 1024） | runtime-verified | E04-01：`.../raw/triton_autotune.json`、`cross_arch_comparison.json` |
| S4-24 | FP16 GEMM 容差模型修正：原 `atol + rtol*|expected|` 对长度 K 的归约在输出过零处不成立（sm_86 上 cuBLAS 与 Triton 相对 FP64 的 `max_abs_error` **完全相同 0.0556**，却因单个近零元素违约）。改为对照 FP64 参考的尺度无关门禁（relative-L2 ≤ 1e-2、cosine ≥ 0.9999），并新增负向对照 | test-verified | `tests/unit/ops/test_triton_gemm.py`（含 `test_fp16_gate_rejects_wrong_output`）；`pytest tests/unit/ops/ -q` → 全绿 |
| S4-25 | `bench_s04.py` 每条计时附带 correctness 判定，并区分 `cold_call_ms`（含 Triton JIT）与稳态中位数；CUTLASS 跳过时记录原因而非静默丢弃 | runtime-verified | `scripts/bench/bench_s04.py`；`reports/dev/rtx3090/s04_backend_baseline.json`（`all_correctness_passed: true`） |
| S4-26 | 模型制品 manifest 记录了**不可复现的客户端缓存索引** `.msc`：相同 model id / `allow_patterns` / 客户端版本下连续三次下载得到两个不同摘要（`9bb0066c…`×2、`869ed3e2…`）；`.mv` 确定性可复现。记录中的 manifest 是从既有快照生成，故其 `.msc` 条目仅对该目录自洽 | runtime-verified | 三次独立下载对照（本轮 E04-04）；处置见 S4-27 |
| S4-27 | 制品门禁恢复可复现：新增 `CLIENT_CACHE_METADATA=(".msc",)`，客户端缓存元数据排除出摘要比对但**显式上报**（`ignored_client_metadata`）；下载脚本改为由 manifest 驱动 `allow_patterns` 并自校验 | runtime-verified | `hqsb/models/manifest.py`、`scripts/models/download_qwen3_modelscope.py`；`verify_qwen3_hashes.py` → `13/14 verified, 0 missing, 0 mismatched, 0 extra, 1 client-cache metadata excluded`（exit 0） |
| S4-28 | E00-05 真实模型 smoke 由 FAIL 转为 **PASS**（制品门禁修复后） | runtime-verified | `reports/dev/rtx3090/e00_05_smoke/`；`identity_ok=True`、`artifact_hash=e7af1c75…`（与 S00 记录一致）、3/3 子进程 exit 0、跨进程 token 哈希一致、`overall: PASS` |
| S4-29 | 跨进程一致性误报已修复：原实现在**全部子进程失败**时报 `ok=True`（`None` 哈希集合大小为 1） | runtime-verified | `scripts/audit/run_e00_05_qwen_tiny_smoke.py`（新增 `n_successful_runs` 前置条件） |
| S4-30 | 当前 HEAD 全量单元测试口径：`pytest -m "not hardware and not e2e and not performance" -q` → **613 passed, 0 failed** | runtime-verified | 本轮 `gate1_python`（归档于 `reports/dev/rtx3090/20260917_193004/gate1_python/`） |
| S4-31 | SM87（Jetson）构建未被跨架构改动破坏 | runtime-verified | `-DCMAKE_CUDA_ARCHITECTURES=87` 独立构建：`build arch = 8.7`、`*.sm_87.cubin`、C ABI 返回 `(8, 7)` |
| S4-32 | S04.5 子阶段正式定义（真实模型算子回接），并补齐 `docs/stage_experiments/S04_实验清单.md` / `S04.5_实验清单.md` | source-only | `docs/stages/S04.5_真实模型算子回接.md`、`docs/stage_experiments/S04_实验清单.md`、`docs/stage_experiments/S04.5_实验清单.md` |

### 10.1 本节引入的降级与修正

- **S4-2 / S4-16（原 CUTLASS sm_87 正确性）**：结论本身不变（sm_87 上默认配置可
  运行且 `max_err ~0.03`），但原实现**未检查内核是否启动**，因此在其他架构上
  可能输出伪造数据。该缺口的适用范围已由 S4-22 明确限定为「非 sm_87 架构」。
- **S2-14 / S4-13（历史 passed 计数）**：维持 `historical-unreproduced`；
  当前口径以 **S4-30（613 passed）** 为准。
- §6 的 S4-1 … S4-17 **全部为 sm_87 口径**，未因本轮改动而失效。

### 10.2 未验证项（如实登记）

1. sm_90 / sm_100 / sm_120 等更新架构未实测（本轮只有 sm_86 与 sm_87）；
2. TileLang 在 sm_86 上未验证（未安装，标记 optional unavailable）；
3. 多架构 fatbin（`-DCMAKE_CUDA_ARCHITECTURES="86;90"`）不受支持：架构自报只取
   第一项（已知限制，见 `S04_阶段验收报告.md` §6）；
4. 跨架构兼容判定为**精确相等**，比 CUDA 真实二进制兼容规则保守；
5. 本节的时延数据为单轮探索性测量，未固定 DVFS/时钟，**不构成稳定性能结论**。

---

## 11. S05 声明台账（接口/代码层，2026-09-18）

> 背景：S05（量化与低精度推理）按任务约束「提供接口、不执行实验」交付了
> E05-01~E05-10 全部 190 个实验步骤的能力接口与测试，**未执行任何正式实验、
> 未产出任何质量/性能/内存/能耗结论数字**。本节所有条目为 `test-verified`
> （接口正确性）或带 `development`/`smoke` 标注的运行证据；**实验层判定为
> `BLOCKED`**（S04.5 M4 前置缺失，见 S5-BLOCK）。配套报告：
> `docs/reports/S05_开发报告.md`、`docs/reports/S05_阶段验收报告.md`。

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| S5-1 | 量化语义冻结 + factorial 矩阵：非法组合（symmetric+twos_complement、per-channel 声明 group_size、per-group 缺 group_size）构造时拒绝 | test-verified | `hqsb/quant/spec.py`；`tests/unit/quant/test_spec_and_rounding.py` |
| S5-2 | 自实现 RTN（per-tensor/channel/group、对称/非对称、axis 通用、zero/constant group、NaN/Inf、尾块）+ 精确有理数 golden 双实现交叉校验 | test-verified | `hqsb/quant/rtn.py`、`golden.py`；`tests/unit/quant/test_rtn_and_golden.py`（golden 11 向量 0 失配） |
| S5-3 | canonical nibble/byte + kernel layout（nibble order、尾块、alignment、parent hash）往返与非法输入（非整数 code、非正 scale、未知 layout、payload 长度）拒绝 | test-verified | `hqsb/quant/packing.py`；`tests/unit/quant/test_packing.py`、`tests/property/test_quant_properties.py` |
| S5-4 | 版本化 QuantArtifact：原子保存、身份/canonical hash 稳定、未知 scheme 字段拒绝、symmetric 含 zeros 拒绝、payload bit-flip 拒绝 | test-verified | `hqsb/quant/artifact.py`；`tests/unit/quant/test_artifact_compat.py` |
| S5-5 | 兼容判定（DIRECT/REPACK/REQUANTIZE/FALLBACK/REJECT）+ lossless repack 不变量 + 迁移计划（未来版本拒绝） | test-verified | `hqsb/quant/compat.py`；`tests/unit/quant/test_artifact_compat.py` |
| S5-6 | 故障注入矩阵 35 例（schema/model/quant/pack/capability）全部 pre-launch 拒绝且 golden 原件不被破坏 | test-verified | `hqsb/quant/faults.py`；`test_artifact_compat.py::TestFaultMatrix` |
| S5-7 | 校准四段数据隔离、样本身份哈希、泄漏审计（id/文本哈希/父文档/模板）、因子抽样确定性与最小充分预算选择规则（零阈值拒绝选择） | test-verified | `hqsb/quant/calibration.py`；`tests/unit/quant/test_calibration_stats.py` |
| S5-8 | 分位数/bootstrap/配对/聚类 bootstrap/效应量/非劣效判定（PASS/FAIL/INCONCLUSIVE 三分） | test-verified | `hqsb/quant/stats.py`；`test_calibration_stats.py::TestStatistics` |
| S5-9 | 模型级评估：覆盖枚举、可逆权重替换（fake_dequant/storage_only 标签）、logit/token 质量指标、执行标签与 claim 审计、内存阶梯、相位计时、统一结果表、五道门顺序 | test-verified | `hqsb/quant/{coverage,apply,quality,execution,model_eval}.py`；`tests/unit/quant/test_coverage_apply_quality.py`、`test_execution_model_eval.py` |
| S5-10 | 工业方法 adapter 五层（GPTQ/AWQ/SmoothQuant）：字段映射审计（无 silent drop）、合成源记录转换、等价性检查 | test-verified | `hqsb/quant/adapters/`；`tests/unit/quant/test_adapters_units_policy.py` |
| S5-11 | 干预单元（fused/tied 合并）与混合精度 policy（YAML 往返、约束校验、贪心搜索轨迹、反事实） | test-verified | `hqsb/quant/{units,sensitivity,policy}.py`；`test_adapters_units_policy.py` |
| S5-12 | 激活量化（静态/动态、per-token、饱和率、SmoothQuant 等价、w8a8 int32 精确累加）与 KV 缓存（容量模型、注意力 oracle、最大上下文搜索） | test-verified | `hqsb/quant/{activation,kv}.py`；`tests/unit/quant/test_activation_kv.py` |
| S5-13 | 决策层：候选注册、五道门、Pareto（点/不确定性）、部署场景、推荐矩阵、发布包、回归阈值 | test-verified | `hqsb/quant/decision.py`；`tests/unit/quant/test_decision.py` |
| S5-14 | Triton W4/W8 fused-dequant GEMM 对 kernel oracle 分层容差通过（W4 对称/非对称、W8、尾块 N=7/K=96 G=48） | runtime-verified（`development`/`smoke`，RTX 3090 sm_86） | `ops/quant/w4a16_triton.py`；`tests/unit/ops/test_quant_kernels.py::TestFusedDequantKernel` |
| S5-15 | 实验驱动入口默认不产结论：`--mode status` 报告 BLOCKED、`--mode execute --confirm-execute` 拒绝（退出码 7） | test-verified + runtime-verified | `scripts/quant/run_e05.py`；`tests/unit/quant/test_experiment_interface_map.py` + 实测退出码 |
| S5-16 | 实验步骤→接口对照表：190 步、219 接口全部 import 解析成功 | test-verified | `hqsb/quant/interface_map.py`；`tests/unit/quant/test_experiment_interface_map.py::TestInterfaceMap` |
| S5-17 | 全量测试 **936 passed, 0 failed**（含既有回归 + 本阶段新增约 280 测试） | runtime-verified | `pytest -q`（本机）；依赖边界 gate 0 violations / 0 cycles |
| S5-BLOCK | S05 **实验层 BLOCKED**：S04.5 M4「真实模型算子回接」前置未满足（`hqsb/integration`、S04.5 实验证据、S04.5 验收报告、六 workload FP16 基线四项缺失） | source-only（缺失事实，由 `check_prerequisites` 实测登记） | `hqsb/quant/experiment.py::check_prerequisites`；`scripts/quant/run_e05.py --mode status` |

> 本阶段不新增 `runtime-verified` 之外的性能/质量结论；S5-14 的 kernel 正确性
> 为接口层 smoke 证据，**不是** E05-06 实验结论（E05-06 实验层 BLOCKED）。

---

## 12. S06 声明台账（接口/代码层，2026-09-18）

> 背景：S06（框架集成与图优化）按任务约束「提供接口、不执行实验」交付了
> E06-01~E06-11 全部 **218 个实验步骤**的能力接口与测试，**未执行任何正式实验、
> 未产出任何 graph/guard/recompile/cache/lowering/kernel/性能/内存结论数字**。
> 本节条目为 `test-verified`（接口正确性）；**实验层判定为 `BLOCKED`**
> （S04.5 M4 / S05 P0 前置缺失，见 S6-BLOCK）。配套报告：
> `docs/reports/S06_开发报告.md`、`docs/reports/S06_阶段验收报告.md`、
> `docs/reports/S06_graph_integration_design.md`。

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| S6-1 | 算子 schema 作为编译契约：mutation/alias/autograd/autocast/inference-only 显式声明，缺失或自相矛盾即拒绝；schema hash 稳定 | test-verified | `hqsb/integration/specs.py`；`tests/unit/integration/test_specs_dispatch.py::TestOperatorSchema` |
| S6-2 | 单一 schema owner 审计：Python `torch.library.define` 与 C++ `TORCH_LIBRARY/m.def` 定义点计数，两名 owner 即冲突 | test-verified | `hqsb/integration/specs.py::audit_schema_owners`；`TestSchemaOwnerAudit` |
| S6-3 | 注册矩阵冲突拒绝：重复 schema/同 key 不同实现/第三方 namespace 占用/实现与 schema hash 不符，全部结构化拒绝，禁止 last-load-wins | test-verified | `hqsb/integration/dispatch.py::RegistrationMatrix`；`TestRegistrationMatrix` |
| S6-4 | dispatch 真实性：capability 驱动选择返回 requested/selected/internal route/reason；不满足时回退 composite 并记录原因；Triton/quant 作为内部 route 而非独立 dispatch key | test-verified | `hqsb/integration/dispatch.py::select_implementation`；`TestSelectionAndRedispatch` |
| S6-5 | redispatch 守卫：按 (op, key set) 检测递归、key set 耗尽与深度上限，不依赖全局递归计数器 | test-verified | `hqsb/integration/dispatch.py::RedispatchGuard`；`TestSelectionAndRedispatch` |
| S6-6 | fallback 策略确定性：固定优先级链、strict 模式拒绝隐式回退、禁用能力必须带 reason、无可用实现显式报错 | test-verified | `hqsb/integration/dispatch.py::FallbackPolicy`；`TestFallbackPolicy`、`tests/property/test_integration_invariants.py` |
| S6-7 | Meta/Fake 元数据契约：符号维度不被具体值替换、stride/alias 逐字段与 real 比对、非法 rank/dtype/shape 在分配前拒绝、fake 路径可证明"无真实分配/无 kernel/无 payload 读取" | test-verified | `hqsb/integration/meta.py`；`tests/unit/integration/test_meta_graph.py` |
| S6-8 | 图 IR 与结构身份：capture mode / IR level 必须标注、结构哈希对节点名不变、拓扑变化必然改变哈希、graph diff 记录增删/死代码/impure 节点 | test-verified | `hqsb/integration/graph.py`；`TestGraphIR`、`tests/property/test_integration_invariants.py::TestHashes` |
| S6-9 | pattern 语义安全：结构只是候选，谓词（eps/alpha/axis/extra user/mutation/impure/weight/capability/version）全通过才可重写；拒绝带字段级 reason；重写在副本上进行、原图不被污染 | test-verified | `hqsb/integration/patterns.py`；`tests/unit/integration/test_patterns.py` |
| S6-10 | false-positive 变异测试：eps/alpha/in-place/未知版本 4 类近失全部被拒（`ok=True`） | test-verified | `patterns.mutation_report`；`TestMutationTesting`、`run_e06.py --mode self-check` |
| S6-11 | 覆盖率口径分离：node/call/time/model coverage 与 precision/recall/false-positive 分别计算，缺数据时比值落在 [0,1] | test-verified | `patterns.coverage_report`；`TestCoverage`、`test_integration_invariants.py::TestCoverageInvariants` |
| S6-12 | guard/break/recompile/assert/fallback 五类事件分离计数，各自带 request/graph/reason；storm 阈值预注册且逐条报告越界原因 | test-verified | `hqsb/integration/guards.py`；`TestCompileLedger`、`TestStorm`、`TestGuardMinimality` |
| S6-13 | compile identity 敏感性：graph/op schema/tensor metadata/model/quant policy/rewrite spec 与 torch/triton/ABI/arch/guard 任一变化都改变 digest | test-verified | `hqsb/integration/cache.py`；`TestCompileIdentity`、`test_integration_invariants.py::TestHashes` |
| S6-14 | cache entry 在执行前校验：路径穿越、缺失、截断、位翻转、外部架构、陈旧 schema/ABI、不完整写入、不可信制品全部检出（`pre_execution=True`） | test-verified | `cache.validate_entry`；`TestCacheEntryValidation` |
| S6-15 | 损坏注入只作用于 scratch 副本：`CorruptionFixture` 拒绝修改 scratch 根之外的目标，golden 不可达 | test-verified | `cache.CorruptionFixture`；`TestCorruptionFixture` |
| S6-16 | 失效矩阵逐因子 expected/actual 对比，含 `irrelevant_metadata_control` 控制项；未观测因子记 `<NOT_OBSERVED>` 而非默认通过 | test-verified | `cache.evaluate_invalidation`；`TestInvalidationMatrix` |
| S6-17 | break-even 数学：`N = extra_compile_cost / (eager - steady)` 向上取整；分母 ≤0 时返回无摊销点并给出 reason | test-verified | `cache.compute_break_even`；`TestBreakEven`、`test_integration_invariants.py` |
| S6-18 | 并发/原子性：锁文件陈旧检测 + 原子写（tmp+fsync+rename），第二个 writer 被拒绝；manifest 往返与淘汰确定性 | test-verified | `cache.CacheLock`、`cache.atomic_write_text`、`cache.evict`；`TestCacheManifestAndLock` |
| S6-19 | lowering 能力驱动：dtype/layout/rank/M/arch/group/workspace 逐项校验，拒绝原因逐个记录；选型规则不得以模型/模块名为键（静态审计） | test-verified | `hqsb/integration/lowering.py`；`TestLoweringRegistry` |
| S6-20 | 分配与收益归因：理论中间量/workspace/拷贝/anchor 与实际节省对账，残差非零即 `explained=False`；Amdahl 预测与实测差必须逐因子解释 | test-verified | `lowering.AllocationAccount.reconcile`、`AttributionReport`；`TestAllocationAccounting`、`TestAmdahlAndAttribution` |
| S6-21 | CUDA Graph 契约与 claim 门：未观测的前置条件视为失败；in-flight 缓冲区拒绝释放；越界 shape 走非 graph 路径；证据不完整时 `NOT_CLAIMED`，未执行时 `NOT_RUN` | test-verified | `hqsb/integration/cuda_graph.py`；`tests/unit/integration/test_cuda_graph_lifecycle.py` |
| S6-22 | 错误分类：14 层 stage 全覆盖、reason code 带 severity/retryable/fallback/user message、fatal 类禁止 fallback、消息脱敏（路径/地址/权重文本） | test-verified | `hqsb/integration/taxonomy.py`；`TestErrorTaxonomy`、`TestFailureRecord` |
| S6-23 | 事务化执行：validate→prepare→execute→validate→commit；不变量破坏时 abort 并丢弃 shadow 输出（无半 token/半 residual）；状态机拒绝乱序调用 | test-verified | `taxonomy.TransactionalPlan`；`TestTransactionalPlan` |
| S6-24 | ABI 兼容判定在 load 前：架构/ABI/toolchain/unknown 字段逐项拒绝，二进制 hash 与符号缺失可检出 | test-verified | `hqsb/integration/abi.py`；`TestABI` |
| S6-25 | 生命周期：状态机拒绝非法迁移；泄漏判定区分 bounded cache 与持续增长（分段斜率 + CI + plateau）；弱引用探针不持有对象；stream 审计检出隐式全局同步 | test-verified | `hqsb/integration/lifecycle.py`；`TestLifecycleMachine`、`TestLeakStatistics`、`TestLivenessAndStreams` |
| S6-26 | 反硬编码：core 代码扫描 10 条规则（模型类名/模块路径/固定层数/后端 if 链/节点名/形状常量/环境开关等），例外必须显式标记并计数上报 | test-verified | `hqsb/integration/adapter.py::hardcode_scan`；`TestHardcodeScanner`（含植入 4 类违规的负向 fixture） |
| S6-27 | 跨目标复用隔离：identity 碰撞检查（`*_id/_hash/_key/_identity`）、`changes_global_default` 必须为假、dummy backend 恒不允许性能 claim | test-verified | `adapter.identity_collision_check`、`AdapterRegistration`、`DummyBackendAdapter`；`TestReuse`、`TestAdapterRegistration`、`TestDummyBackendAdapter` |
| S6-28 | 四级 differential 门禁：容差按 level×dtype 预注册（缺失即错误、不给全局默认）、NaN/Inf 独立硬门、首次不匹配索引保留、路径矩阵缺失单元显式化 | test-verified | `hqsb/integration/differential.py`；`tests/unit/integration/test_differential.py` |
| S6-29 | 冻结融合语义：functional（不原地写 residual）、FP32 累加、舍入后读取；参考实现为纯 Python FP64 组合而不是目标 kernel | test-verified | `differential.FROZEN_ADD_RMSNORM_SEMANTICS`、`fused_add_rms_norm_reference`；`TestFusionSemantics` |
| S6-30 | C6/C7 投影完整且不破坏冻结 schema：14 个 C6 字段全部可寻址（`missing=[]`）、18 类 S06 事件都映射到合法 C7 事件（`invalid=[]`）、run/span/parent/单调时间可关联 | test-verified | `hqsb/integration/telemetry.py`；`TestC6Projection`、`TestC7Projection` |
| S6-31 | 配置即冻结：7 类 YAML 严格加载（未知键拒绝）、算子契约与代码 schema 逐字段审计、pattern 声明与代码无漂移 | test-verified | `hqsb/integration/policies.py`；`tests/unit/integration/test_s06_experiment_scaffolding.py::TestConfigs` |
| S6-32 | 运行脚手架默认不产结论：verdict 在未执行/无 raw/前置未满足时一律落 `BLOCKED`；报告骨架标注 `NOT_RUN` 且不含数字；run layout 覆盖 graph/compile/dispatch/correctness/performance 等 19 个子目录 | test-verified | `hqsb/integration/experiment.py`；`TestRunDirectory` |
| S6-33 | 接口对照表 218 步 / 381 接口 / 629 引用全部 import 解析成功（对照表不会腐烂成文档） | test-verified | `hqsb/integration/interface_map.py`；`resolve_interfaces()` → `ok=True`；`TestInterfaceMap` |
| S6-34 | 依赖方向：`hqsb.integration` 模块级不 import torch/triton、不 import `ops`；`core`/`models`/`benchmark` 不 import integration；gate 规则 R1+R2 0 违规 0 环 | test-verified | `tests/unit/integration/test_import_boundaries.py`；`scripts/audit/import_dependency_gate.py`（PASS） |
| S6-35 | S05 前置门收紧（更严而非更松）：S04.5 M4 证据从「目录存在」改为「存在执行证据标记 `s04_5_evidence.json`」，避免 S06 脚手架伪造上游证据 | test-verified | `hqsb/quant/experiment.py`；`test_s06_experiment_scaffolding.py::TestPrerequisites::test_s06_scaffolding_is_not_accepted_as_s04_5_evidence` |
| S6-36 | 全量测试 **1303 passed, 0 failed**（4 deselected；本阶段新增 371） | runtime-verified（本机） | `python3 -m pytest tests/ -q -m "not hardware and not e2e and not performance"`（本机 RTX 3090 / CPU 测试口径） |
| S6-37 | 实验驱动默认拒绝：`--mode status` 6/7 前置 MISS → BLOCKED；`--mode execute --confirm-execute` 退出码 7；`--mode preregister` 只写预注册与 `NOT_RUN` 骨架 | runtime-verified（本机） | `scripts/integration/run_e06.py`；`docs/reports/S06_阶段验收报告.md` §7 |
| S6-38 | CPU-minimal 打包包含 S06 交付：wheel（124 entries）含 19 个 `hqsb/integration/` 模块；CI 入口两步（pytest 口径 + dependency gate）全通过 | runtime-verified（本机，`development`） | `pip wheel . --no-deps --no-build-isolation`（离线环境需关构建隔离）；`.github/workflows/ci.yml` |
| S6-BLOCK | S06 **实验层 BLOCKED**：7 条硬前提中 6 条未满足（S04.5 M4 证据标记/S04.5 证据目录/S04.5 验收报告/六 workload FP16 基线/S05 P0 verdict/环境指纹） | source-only（缺失事实，由 `check_prerequisites` 实测登记） | `hqsb/integration/experiment.py::check_prerequisites`；`run_e06.py --mode status` |
| S6-CLAIM | CUDA Graph（P1）**未声称**：`claim_cuda_graph: false`，`claim_status(executed=False)` → `NOT_RUN` | source-only | `configs/integration/graph_spec.yaml`；`cuda_graph.claim_status` |
| S6-DEBT | 既有文档债务：`docs/stage_experiments/details/*/README.md` 43 处相对链接断裂（S04/S04.5 清单随 `9b403aa` 移出树；控制平面文档实际在 `docs/architecture/`）；协议目录冻结，登记不修 | source-only | `scripts/check_docs.py`（43 broken，全部既有） |

> 本节不含任何实验结果数字。S6-36/S6-37 是本机测试与驱动行为证据（`development`
> 级别），**不是** E06 实验结论；E06-01~E06-11 实验层全部 BLOCKED。

---

## 13. S07 声明台账（接口/代码层，2026-09-18）

> 背景：S07（推理 Runtime 内核）按任务约束「提供接口、不执行实验」交付了
> E07-01~E07-10 全部 **200 个实验步骤**的能力接口与测试，**未执行任何正式实验、
> 未产出任何 TTFT/TPOT/TPS/容量/碎片/命中率/显存/能耗结论数字**。本节条目为
> `test-verified`（接口正确性）或本机 `development` 级别的驱动/测试行为证据；
> **实验层判定为 `BLOCKED`**（见 S7-BLOCK）。配套报告：
> `docs/reports/S07_开发报告.md`、`docs/reports/S07_阶段验收报告.md`、
> `docs/reports/S07_runtime_architecture.md`。

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| S7-1 | capability 协商可审计：六态词表、UNKNOWN 默认拒绝 claim、EMULATED 必须给测量成本、requested≠actual 必须带 reason（silent 降级在构造期被拒绝） | test-verified | `hqsb/runtime/request.py`；`tests/unit/runtime/test_request_capability.py` |
| S7-2 | 请求语义冻结在 token ID 层：`RequestSpec.request_hash` 对 model/input/sampling/stop 敏感且稳定；greedy 与 sampling 不可混用 | test-verified | `hqsb/runtime/request.py`；`test_request_capability.py::TestModelIdentityAndRequest`、`TestSamplingAndStop` |
| S7-3 | Adapter 七类操作 + 生命周期状态机：非法迁移与 close 后 generate 被拒；close 幂等；stream 序号/重复/丢失/final/文本被校验；cancel 需声明 in-flight 政策 | test-verified | `hqsb/runtime/adapter.py`；`test_adapter_parity.py::TestDummyAdapter`、`TestStreamAndCancelContracts` |
| S7-4 | 引擎能力来自**探测**而非文档：`probe_environment()` 用 `find_spec`，只有 `AVAILABLE` 才可用；`select_primary_engine` 在无唯一可用引擎时 `ok=False` | test-verified + runtime-verified（本机） | `hqsb/runtime/adapter.py`；本机实测 vllm/sglang/tensorrt_llm/llama_cpp 全部 `NOT_INSTALLED`，`run_e07.py --mode self-check` 输出 `primary_selection_ok=false` |
| S7-5 | 语义 parity：greedy 逐 step 比较并定位首分歧；logit 容差独立判定；sampling 分布比较要求 ≥2 seed，单 seed 与 INCONCLUSIVE 均拒绝声称等价 | test-verified | `hqsb/runtime/parity.py`；`test_adapter_parity.py::TestParityOracles` |
| S7-6 | 参数逐项生效门禁：每个 claimable 字段必须有「输出变化」或 telemetry 证据；unsupported 字段必须以 REJECT/FALLBACK 且带 reason，`SILENT_IGNORE` 判失败 | test-verified | `parity.parameter_effect_matrix`、`parity.unsupported_negative_matrix`；`TestParameterAndUnsupportedMatrices` |
| S7-7 | 请求状态机与终态约束：非法迁移被拒；取消/超时后不得重新 admit；非终态收尾被拒；取消后 emitted token 可检出 | test-verified | `hqsb/runtime/trace.py`；`tests/unit/runtime/test_trace_ledger.py::TestRequestStateMachine` |
| S7-8 | C7 span 关联：跨请求 parent 与缺 source symbol 被检出；敏感属性脱敏；iteration 账本 token 守恒（`previous + scheduled - rollback == new` 非零即拒绝） | test-verified | `trace.SpanCollector.join_audit`、`trace.conservation_audit`；`test_trace_ledger.py::TestSpans`、`TestIterationLedger` |
| S7-9 | 插桩开销分级：off/minimal/full/profiler 分开测量，full/profiler 永不可用于计时 | test-verified | `trace.instrumentation_overhead`；`test_trace_ledger.py::TestClocksAndOverhead` |
| S7-10 | KV 几何与记账：`bytes_per_token` 公式与量化侧数据分离；内部碎片定义为 slot 浪费；未命名内存类别被拒；残差超容差即 `explained=False`（不允许叫「碎片」） | test-verified | `hqsb/runtime/kv.py`；`tests/unit/runtime/test_kv_blocks.py::TestGeometry`、`TestMemoryReconciliation` |
| S7-11 | block 生命周期不变量：free 无 owner/reader、refcount 与 readers 一致、无瞬时状态残留、active/shared 不可 evict、free 后访问被拒（双 free 被吸收、refcount 错误被不变量检出） | test-verified | `kv.BlockPool.invariant_report` 及三个 fault fixture；`TestBlockPool` |
| S7-12 | 容量与 OOM 有界：容量二分对脏下界/无上界拒绝；**每个 OOM kind 一条有限阶梯且必然以 `reject`/`fail` 结束**；超长 context 按最早可知层拒绝（先占后拒 = 缺陷） | test-verified | `kv.find_capacity_boundary`、`kv.OOM_LADDERS`、`kv.oom_action`、`failure.ContextAbuseCheck`；`TestCapacityAndOOM`、`test_failure_matrix.py::TestOomPolicy`、`test_runtime_invariants.py::TestKvAccounting` |
| S7-13 | 调度守恒与预留：逐轮 token 守恒（随机 trace 属性测试）；chunk 区间精确 tile prompt；`admission_reserve_full_isl` 按 `prompt + max_new` 预留 block，防止过度接纳与 thrash；公平/拥塞指标有界 | test-verified | `hqsb/runtime/scheduler.py`；`tests/unit/runtime/test_scheduler_batching.py`、`test_runtime_invariants.py::TestSchedulerConservation` |
| S7-14 | 调度输出标注为结构而非测量：所有 payload 带 `simulated=True`，时间字段为 0 或来自真实 run | test-verified | `scheduler.SimulationResult.as_dict`；`test_scheduler_batching.py::TestSimulationInvariants::test_simulation_payload_is_labelled_simulated` |
| S7-15 | prefix cache key 绑定全部 KV 语义：任一 key 字段（15 字段 + parent chain + block token hash）变化必然改变 digest；负向 fixture 在 key 上变异而非在 identity dict 上 | test-verified | `hqsb/runtime/prefix_cache.py`；`tests/unit/runtime/test_prefix_cache.py::TestPrefixKey`、`TestNegativeFixtures`、`test_runtime_invariants.py::TestPrefixKeySensitivity` |
| S7-16 | 碰撞政策有真实差异：`digest_plus_token_equality` 拒绝伪造记录并计数，`strong_digest` 会被同一伪造记录欺骗（对照组）；多 KV group 命中取交集，单组命中不得虚报 | test-verified | `prefix_cache.collision_is_detected`、`PrefixCache.lookup`；`TestCollisionPolicy`、`TestLookupSemantics::test_block_groups_take_the_intersection` |
| S7-17 | refcount/eviction 保护：refcount>0 的 entry 拒绝丢弃；eviction 遵守字节上限；release 可选择保留；不变量报告检出 refcount 漂移 | test-verified | `PrefixCache._drop`、`PrefixCache.evict`；`TestLifecycleAndEviction` |
| S7-18 | graph 路由可证：bucket 命中/越界带 fallback 与 reason 与 padding；bucket 数 > `max_graphs` 拒绝；同输入多 replay 被检出；break-even 与无摊销点分开；claim 门在未执行/证据不全/spec 不声称时分别为 `NOT_RUN`/`NOT_CLAIMED` | test-verified | `hqsb/runtime/graph_route.py`；`tests/unit/runtime/test_graph_attention.py::TestGraphSpec`、`TestReplayAccounting`、`TestPhaseAndClaim` |
| S7-19 | attention 能力逐字段判定：dtype/kv dtype/head_dim/GQA/phase/paged/context/alignment/graph 可捕获性逐项给 reason 与 fallback；`API 配置 ≠ 实际 kernel` | test-verified | `graph_route.check_attention_support`、`attention_matrix`；`TestAttentionCapability` |
| S7-20 | 2×2 factorial 与符号确定性：graph 主效应按 `cuda_graph − eager`、attention 主效应按 `candidate − default` 计算，因子顺序由调用方声明（不随命名/字母序翻转符号）；不兼容 cell 记 `UNSUPPORTED` 且禁止外推；缺 metric 不得凭空补 | test-verified | `graph_route.two_factor_analysis`；`TestFactorial` |
| S7-21 | speculative 数学精确：acceptance `min(1, p/q)` 与 residual `normalize(max(0, p-q))` 用 `Fraction` 精确校验（含零概率与退化情形拒绝）；golden case 覆盖 all-accept/first-reject/零概率 | test-verified | `hqsb/runtime/spec_decode.py`；`tests/unit/runtime/test_spec_decode.py::TestAcceptanceMathematics` |
| S7-22 | speculative 守恒与收益：`advanced = accepted + correction`（混淆即拒绝）；`rollback = proposed - accepted` 且无 stale token；收益用 cycle 成本公式给出 effective TPOT 与 target calls/token；无 advanced 时拒绝除法 | test-verified | `CycleRecord`、`kv_commit_rollback_audit`、`BenefitModel`；`TestCyclesAndRollback`、`TestBenefitModel` |
| S7-23 | MTP 不得借用严格采样保证（除非自证目标分布保持）；P1 claim 默认 `NOT_RUN`，证据不全 `NOT_CLAIMED`；`configs/runtime/spec_decode_spec.yaml: claim: false` | test-verified | `spec_decode.assert_mtp_does_not_borrow`、`claim_status`；`TestMtpAndClaim` |
| S7-24 | 失败矩阵完整且冻结：28 例覆盖 request_control/resource/runtime，每例声明 expected action 与 extra-output 政策并共享 10 条公共不变量；表中缺失/重复/失败都被报出 | test-verified | `hqsb/runtime/failure.py`；`tests/unit/runtime/test_failure_matrix.py::TestFailureMatrix`、`TestOutcomeTable` |
| S7-25 | cancel 时间语义：五时刻单调校验；`no_extra_token` 契约下不得输出额外 token；observation/block-release/cleanup 延迟可计算 | test-verified | `failure.CancelTimeline`；`TestCancelTimeline` |
| S7-26 | 长稳判据：warmup/steady 分段拟合，稳态仍增长即 `GROWING` 并阻塞 PASS；`allocator reserve` 不自动等于泄漏；无稳态段直接拒绝 | test-verified | `failure.resource_slope_report`、`leak_blocks_pass`；`TestRecoveryAndLongRun` |
| S7-27 | 公平比较分级：tier A–D 由受控变量自动派生；common-denominator 与 best-valid 两表分离（后者需预注册）；不可比行必须带 reason；质量门失败的行不得进入对照表 | test-verified | `hqsb/runtime/comparison.py`；`tests/unit/runtime/test_comparison_policyab.py::TestTiers`、`TestRowsAndModes` |
| S7-28 | 统一口径重算：所有速率从 raw 重算；自报 TPS 与重算不一致即审计失败；token 三分母（logical/useful/model-computed）分离，取消/超时/recompute 不从分母删除 | test-verified | `comparison.recompute_metrics`、`comparison.token_denominator_audit`、`metrics.TokenLedger`；`TestMetricRecompute`、`tests/unit/runtime/test_metrics_tokens.py::TestTokenLedger` |
| S7-29 | 冷热分离与 Pareto：install/build 只报告不得混入稳态；Pareto 按 hardware × workload 分别计算，被支配点保留可见；no-winner 当 CI 跨零 | test-verified | `comparison.cold_warm_report`、`pareto_front`、`compare_runs`；`TestColdWarmAndPareto` |
| S7-30 | A/B 选题门禁：七项条件不全或没有证据引用即拒绝；ADR 必须声明 metrics/workloads/invariants/rollback；A/B identity 只允许 patch/build 不同（其余差异被点名）；ABBA/随机区组平衡且拒绝长连续段；pilot 数据不得进入 final | test-verified | `hqsb/runtime/policy_ab.py`；`TestSelectionGateAndAdr`、`TestAbIdentityAndSchedule` |
| S7-31 | 因果链与裁决：E2E 变化而近因未变 → 不可归因；近因变而 E2E 未变 → Amdahl 限制（合法负结果）；correctness/safety 失败或 guardrail 回归 → `ROLLBACK`；attributable 且无显著收益 → `PASS_NEGATIVE`；其余 `INCONCLUSIVE`；裁决必须使用预注册 primary，换指标即拒绝 | test-verified | `policy_ab.causal_chain_check`、`decide`；`TestCausalChainAndDecision` |
| S7-32 | C6/C7 投影完整且不破坏冻结 schema：C6 20 个字段全部可寻址（`missing=[]`）、C7 span 链（11 类）全部映射到合法事件（`invalid=[]`）、run/request/parent/单调可关联 | test-verified | `hqsb/runtime/telemetry.py`；`test_s07_experiment_scaffolding.py::TestTelemetryProjection` |
| S7-33 | 配置即冻结：8 类 YAML 严格加载（未知键/重复 kind 拒绝）、与代码词表逐字段审计（fragment/oom/eviction/tier/phase/objective 等） | test-verified | `hqsb/runtime/specs.py`、`configs/runtime/*.yaml`；`TestConfigs` |
| S7-34 | 运行脚手架默认不产结论：verdict 在未执行/无 raw/前置未满足时一律落 `BLOCKED`；报告骨架标注 `NOT_RUN` 且不含数字；run layout 覆盖 §18 全部子目录 | test-verified | `hqsb/runtime/experiment.py`；`TestRunDirectory` |
| S7-35 | 前置门不可被本阶段脚手架自我解锁：指纹与 probe/selection 只接受 `docs/stage_experiments/**`；脚手架写在 `experiment_results/` 的指纹不计入 | test-verified | `experiment.check_prerequisites`；`TestPrerequisites::test_scaffolding_written_fingerprint_does_not_satisfy_the_gate` |
| S7-36 | 接口对照表 200 步 / 308 接口 / 429 引用全部 import 解析成功（含 dataclass 字段与类注解形式的接口） | test-verified | `hqsb/runtime/interface_map.py`；`resolve_interfaces()` → `ok=True`；`TestInterfaceMap` |
| S7-37 | 依赖方向：`hqsb.runtime` 模块级不 import torch/triton/numpy、不 import `ops`；`core`/`models`/`benchmark`/`backends`/`hardware`/`quant`/`integration` 不 import runtime；gate 规则 R1–R4 0 违规 0 环（rules 1.1.0） | test-verified + runtime-verified（本机） | `tests/unit/runtime/test_runtime_import_boundaries.py`；`scripts/audit/import_dependency_gate.py`（PASS，138 files / 313 edges） |
| S7-38 | 全量测试 **1763 passed, 0 failed**（4 deselected；本阶段新增 460：415 单元 + 45 属性） | runtime-verified（本机） | `.venv/bin/python -m pytest tests/ -q -m "not hardware and not e2e and not performance"` |
| S7-39 | 实验驱动默认拒绝：`--mode status` 7/8 前置 MISS（+C6/C7 OK）→ 全部 `BLOCKED`；`--mode execute --confirm-execute` 退出码 7；`--mode preregister` 只写预注册与 `NOT_RUN` 骨架；`--mode interface-map` 输出 200 步对照表 | runtime-verified（本机） | `scripts/runtime/run_e07.py`；`docs/reports/S07_阶段验收报告.md` §7 |
| S7-40 | CPU-minimal 打包包含 S07 交付：wheel（142 entries）含 18 个 `hqsb/runtime/` 模块 | runtime-verified（本机，`development`） | `.venv/bin/python -m pip wheel . --no-deps --no-build-isolation` |
| S7-41 | e01_06 的 `overall=FAIL`（`logs_joinable`、`schema_missing_required_field`、`schema_unknown_field`）为**既有**问题 | runtime-verified（本机，对照工作树） | 在 `git worktree add /tmp/hqsb-head HEAD` 的干净树上实测同样 FAIL；脚本自述为 `schema_field_gap` / `run_trace_linkage_gap` |
| S7-42 | 手册 §4 统一记录可执行化：`ExperimentRecord` 字段与顺序对齐手册 §4（另含 §5.7 要求的 `fallback_reason`）；结论状态必须带 `decision` 与 raw evidence URI（`performance_samples_uri`/`profile_artifacts_uri`），模板不可带结论状态 | test-verified | `hqsb/runtime/experiment.py`；`test_s07_experiment_scaffolding.py::TestExperimentRecord`；`run_e07.py --mode self-check`（`conclusion_without_evidence_refused=true`） |
| S7-43 | C2 对齐审计：WorkloadSpec 的 14 个字段逐一映射到 runtime 载体（token IDs/sampling/stop/budget/warmup-steady/run 级重复/确定性 arrival offset…），并计入前置门 `c6_c7_schema_available`（C2+C6+C7 同时可用才放行） | test-verified | `telemetry.c2_alignment`、`experiment.check_prerequisites`；`test_s07_experiment_scaffolding.py::TestTelemetryProjection::test_c2_alignment_covers_every_workload_field` |
| S7-BLOCK | S07 **实验层 BLOCKED**：8 条硬前提中 7 条未满足（S04.5 M4 标记、S05 verdict、S06 verdict、冻结请求夹具、capability probe、主 runtime 选择、协议树内环境指纹） | source-only（缺失事实，由 `check_prerequisites` 实测登记） | `hqsb/runtime/experiment.py::check_prerequisites`；`run_e07.py --mode status` |
| S7-CLAIM | CUDA Graph 与 speculative/MTP 均**未声称**：`claim_cuda_graph: false`、`claim: false` → `claim_status(executed=False)` = `NOT_RUN` | source-only | `configs/runtime/graph_attention_spec.yaml`、`configs/runtime/spec_decode_spec.yaml`；`graph_route.claim_status`、`spec_decode.claim_status` |
| S7-LIMIT | 未覆盖边界（如实登记）：真实 runtime 适配器未实现（引擎均未安装）；prefix cache 仅支持 offset 0 的 block chain；调度器是确定性模拟器（时间数字必须来自真实 run） | source-only | `hqsb/runtime/adapter.py::CANDIDATE_ENGINES`；`prefix_cache.py` 模块 docstring；`scheduler.SimulationResult.simulated` |

> 本节不含任何实验结果数字。S7-38/S7-39/S7-40 是本机测试与驱动行为证据
> （`development` 级别），**不是** E07 实验结论；E07-01~E07-10 实验层全部 BLOCKED。

---

## 14. S08 声明台账（接口/代码层，2026-09-18）

> 背景：S08（ServeFabric 与性能治理）按任务约束「提供接口、不执行实验」交付了
> E08-01~E08-11 全部 **264 个实验步骤**的能力接口与测试，**未执行任何正式实验、
> 未产出任何容量/延迟/吞吐/goodput/命中率/显存/能耗结论数字**。本节条目为
> `test-verified`（接口正确性）或本机 `development` 级别的驱动/测试行为证据；
> **实验层判定为 `BLOCKED`**（见 S8-BLOCK）。配套报告：
> `docs/reports/S08_开发报告.md`、`docs/reports/S08_阶段验收报告.md`、
> `docs/reports/S08_serving_architecture.md`。

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| S8-1 | 协议子集冻结并可审计：`ProtocolProfile`（wire/schema/语义/生命周期四类兼容声明）、未知字段拒绝而非丢弃、默认值展开进入 `normalized` | test-verified | `hqsb/serving/protocol.py`；`tests/unit/serving/test_protocol_sse.py::TestProtocolProfile`、`TestValidation` |
| S8-2 | 错误目录自洽：状态/码/重试性/阶段一致；永久错误不得伪装 500、过载不得伪装入参错误；未登记码分类时显式报错 | test-verified | `hqsb/serving/protocol.py::ErrorCatalog.validate`、`slo.classify_failure`；`test_protocol_sse.py::TestErrorCatalog`、`test_policy_planes.py::TestSloFunnel` |
| S8-3 | SSE 线缆级：增量解析器容忍任意 TCP 边界（含 UTF-8 码点内切分）；framing oracle（序列单调/终帧唯一且最后）；流-非流配对；token 静默丢失被拒 | test-verified | `hqsb/serving/sse.py`；`test_protocol_sse.py::TestSseCodec`、`tests/property/test_serving_invariants.py::TestSseReassemblerComposition` |
| S8-4 | 请求状态机：非法迁移被拒；传输层拒绝也记录完整轨迹；断连/超时后直接传播取消 | test-verified | `hqsb/serving/gateway.py::RequestStateMachine`；`test_gateway_lifecycle.py::TestRequestStateMachine` |
| S8-5 | 网关端到端（模拟传输）：非流/流/拒绝/取消/断连/故障/drain 全路径；投递账本审计通过 | test-verified | `hqsb/serving/gateway.py`、`transport.py`、`dummy_backend.py`；`test_gateway_lifecycle.py`（24 项） |
| S8-6 | 错误模型/重复 token 门禁：身份不匹配映射 502 且不返回错误模型内容；外部提交后透明重试被禁；尝试血缘检出跨尝试 token | test-verified | `test_gateway_lifecycle.py::test_identity_mismatch_never_returns_a_success`、`test_backend_evidence_planes.py::TestFaults` |
| S8-7 | SLO 预注册：模板未冻结拒绝评估 goodput/`G*`；good 需协议成功+身份+TTFT+TPOT+E2E 四条件；计数漏斗单调且逐差归因 | test-verified | `hqsb/serving/slo.py`；`test_policy_planes.py::TestSloFunnel`、`test_serving_invariants.py::TestFunnelConservation` |
| S8-8 | 到达过程可重放：五种分布（恒定/泊松/on-off/批量/复合）长程均值对齐；trace 内容 hash 稳定；保真门标注 `LOADGEN_INVALID` | test-verified | `hqsb/serving/arrival.py`、`loadgen.py`；`test_policy_planes.py::TestArrival`、`test_serving_invariants.py::TestArrivalMeanRate` |
| S8-9 | 公平性：Jain 只含 backlogged 租户；成本模型版本化；slowdown 需隔离基线；HOL trace 六构型 | test-verified | `hqsb/serving/fairness.py`；`test_policy_planes.py::TestFairness` |
| S8-10 | 队列策略统一接口：FIFO/严格优先级/加权公平同一决策记录；work-conserving 审计；退款只退未执行部分 | test-verified | `hqsb/serving/policies.py`；`test_policy_planes.py::TestPolicies` |
| S8-11 | 准入：压力状态机需最小 dwell；有界队列硬上限拒绝；无界队列有 kill guard；重试预算尊重提交边界 | test-verified | `hqsb/serving/admission.py`；`test_policy_planes.py::TestAdmission` |
| S8-12 | 路由：注册表拒绝重复 id、代际原子替换；硬过滤先于评分；缺失遥测不为 0；route-vs-actual 守恒；no-feasible 稳定拒绝 | test-verified | `hqsb/serving/router.py`；`test_backend_evidence_planes.py::TestRouter` |
| S8-13 | 熔断：排除类不计数、未知类报错、隔离非退避、半开探测恢复、迁移可复算 | test-verified | `hqsb/serving/circuit.py`；`test_backend_evidence_planes.py::TestCircuit` |
| S8-14 | cache-aware：身份负向夹具全部改变 digest；版本失效；匹配器与 oracle 一致；联合策略净收益；倾斜上界 | test-verified | `hqsb/serving/cache_routing.py`；`test_backend_evidence_planes.py::TestCacheRouting`、`TestPrefixMatcherBounds` |
| S8-15 | 可观测性：traceparent 校验与信任边界；prompt 文本脱敏；直方图分位从 raw 重算；根因需反事实 | test-verified | `hqsb/serving/observability.py`；`test_backend_evidence_planes.py::TestObservability` |
| S8-16 | C6/C7 投影不改冻结 schema，字段/span 覆盖审计通过 | test-verified | `hqsb/serving/telemetry.py`；`test_s08_experiment_scaffolding.py::TestSpecs` |
| S8-17 | 12 份冻结配置严格键校验 + 逐文档契约审计全绿；未知键拒绝、缺失文档拒绝 | test-verified | `hqsb/serving/specs.py`；`test_s08_experiment_scaffolding.py::TestSpecs` |
| S8-18 | 实验脚手架：前置门不自我解锁（只读协议树）；模板不可带结论状态；无 raw samples 拒绝结论 | test-verified | `hqsb/serving/experiment.py`；`test_s08_experiment_scaffolding.py::TestPrerequisites`、`TestVerdictRefusal` |
| S8-19 | 接口对照：264 步 / 418 唯一接口 / 791 引用全部解析；每步映射到真实可导入符号 | test-verified | `hqsb/serving/interface_map.py`；`test_s08_experiment_scaffolding.py::TestInterfaceMap` |
| S8-20 | 依赖边界：serving 无模块级 torch/triton/numpy、不 import ops、下游不反向依赖；gate R1–R6 0 违规 0 环 | test-verified | `tests/unit/serving/test_serving_import_boundaries.py`；`scripts/audit/import_dependency_gate.py`（rules=1.2.0） |
| S8-21 | 全量测试 **1894 passed, 0 failed**（4 deselected；本阶段新增 131 = 125 单元 + 6 属性） | runtime-verified（本机） | `.venv/bin/python -m pytest -m "not hardware and not e2e and not performance" -q` |
| S8-22 | 驱动默认拒绝：`--prerequisites` 实测 7/8 前置 MISS → 全部 `BLOCKED`；`--smoke` 无模型夹具经网关 200、SSE/账本校验通过（标注 smoke）；`--interface-map` 输出 264 步对照 | runtime-verified（本机） | `scripts/serving/run_e08.py`；`docs/reports/S08_阶段验收报告.md` §5 |
| S8-BLOCK | S08 **实验层 BLOCKED**：7 条硬前置未满足（S07 P0 verdict、双后端注册、冻结请求夹具、冻结 SLO、拓扑、loadgen 校准、协议树指纹） | source-only（缺失事实，由 `check_prerequisites` 实测登记） | `hqsb/serving/experiment.py::check_prerequisites`；`run_e08.py --experiment E08-01 --prerequisites` |
| S8-LIMIT | 未覆盖边界（如实登记）：真实 Backend/引擎未接入（`dummy_backend.claim_allowed()` 恒 False）；HTTP 绑定为 stdlib 参考实现；真实并发/竞态需真实后端实验 | source-only | `hqsb/serving/dummy_backend.py`、`transport_http.py` |

> 本节不含任何实验结果数字。S8-21/S8-22 是本机测试与驱动行为证据
> （`development` 级别），**不是** E08 实验结论；E08-01~E08-11 实验层全部 BLOCKED。

---

## 15. S10 声明台账（接口/代码层，2026-09-19）

> 基线：`b83ebde`（工作树在本阶段开始前干净）。本节**不含任何实验结果数字**；
> S10-20 及以后是本机测试与驱动行为证据（`development` 级别），**不是** E10 实验结论；
> E10-01~E10-06、E10-08~E10-10 实验层全部 BLOCKED，E10-07 为 NOT_RUN_NOT_CLAIMED。

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| S10-01 | 拓扑身份/节点边图谱/漂移分级/降级边政策/硬不变量：schema 校验拒绝悬空节点、重复不一致边、无理由 UNAVAILABLE；六档漂移含 downstream action | test-verified | `hqsb/distributed/topology.py`；`tests/unit/distributed/test_topology_placement.py::TestManifestSchema`、`::TestDrift`、`::TestDegradedPolicy` |
| S10-02 | PlacementPlan → rank table/launcher 映射由同一对象生成；planned/actual 比对含 device/local_rank/CPU affinity；替代 placement 结构上不得作为性能声明 | test-verified | `hqsb/distributed/placement.py`；`test_topology_placement.py::TestPlacement` |
| S10-03 | rank/run 身份与 group membership：global/local/node/group rank 不混用；run identity 跨 rank 一致性校验；模糊 backend 版本被拒绝 | test-verified | `hqsb/distributed/ranks.py`；`test_topology_placement.py::TestRankInvariants` |
| S10-04 | Preflight 负向 fixture：重复设备、未知 UUID、rank gap、组顺序不一致 **在 communicator 初始化之前**被拒绝 | test-verified | `hqsb/distributed/probes.py`；`test_topology_placement.py::TestPreflightFixtures` |
| S10-05 | collective 语义冻结：6 类 op 的 count 语义、dtype 宽度、rank-coded 生成器与 CPU oracle（含 AllGather 组序、ReduceScatter 分段、AllToAllV 守恒） | test-verified | `hqsb/distributed/collectives.py`；`tests/unit/distributed/test_collectives.py::TestOracles` |
| S10-06 | 带宽公式 ID 化：algbw/NCCL-tests correction/α–β/派生列；vendor 原生列与派生列结构分离（不冒充物理链路流量） | test-verified | `collectives.bandwidth_row`/`FORMULA_REGISTRY`；`TestBandwidthFormulas`；`tests/property/test_distributed_invariants.py::TestBandwidthAlgebra` |
| S10-07 | 计时语义：host API 返回时间不得当 collective latency；异步 work handle 必须 wait 后测量 | test-verified | `hqsb/distributed/backend.py::TimingCalibration`；`test_collectives.py::TestStopRulesAndLoopback` |
| S10-08 | 能力矩阵三态：UNKNOWN 永不授权执行；requested/actual 不同且无 reason 时构造失败 | test-verified | `backend.CapabilityDecision`/`RequestedActual`；`test_collectives.py::TestGrids`、`::TestStopRulesAndLoopback` |
| S10-09 | communicator 状态机与错误后复用拒绝；sequence 与 (group, epoch) 绑定，重建不得复用旧 seq | test-verified | `hqsb/distributed/sequence.py`；`tests/unit/distributed/test_sequence_faults.py::TestStateMachine`、`::TestSequenceAndPreflight` |
| S10-10 | metadata preflight 给出首个分叉 group/seq/rank/field；dtype 同字节不同语义可识别；handshake 明确非 barrier、仅 debug/fault 模式 | test-verified | `sequence.preflight_check`/`byte_counts_match_but_semantics_differ`；`test_sequence_faults.py::TestSequenceAndPreflight` |
| S10-11 | timeout 由健康 P99×倍率并 clamp 到 [min, max]；fault 后修改需显式拒绝；延迟阶梯覆盖阈值两侧并区分误杀 | test-verified | `sequence.TimeoutSpec`/`delayed_rank_ladder`/`evaluate_delayed_case`；`test_sequence_faults.py::TestTimeoutsAndWatchdog` |
| S10-12 | 故障 oracle 与"正确失败"定义：8 类作弊式通过（弱恢复级别、部分输出被消费、communicator 复用、旧 epoch、外部强杀无记录、资源增长超限、终态不完整、检测超界）逐一拒绝 | test-verified | `hqsb/distributed/faults.py::evaluate_fault`；`test_sequence_faults.py::TestFaultOracle` |
| S10-13 | 安全边界与恢复级别：blast radius/专用 job/非实验 PID 保护/共享网络不动；网络注入需隔离或显式 PARTIALLY_VALIDATED；恢复级别有序且按故障表预注册 | test-verified | `faults.SafetyScope`/`FaultInjection`/`RECOVERY_LEVELS`；`test_sequence_faults.py::TestSafetyAndRecovery` |
| S10-14 | 资源闭环：allocator cache 与 leak 区分；MTTD/MTTR 时间线单调性校验；按 fault 汇总边界（应用可恢复/需重启/需隔离/未真实验证） | test-verified | `faults.ResourceSnapshot`/`snapshot_delta`/`TimeMetrics`/`boundedness_summary`；`test_sequence_faults.py::TestResourceClosure` |
| S10-15 | Qwen census → 参数形状表 → TP 能力矩阵（整除/KV 策略/设备数）与 ParallelPlan（参数覆盖、shard 范围无重叠缺口、fallback 必须 reject、unsupported predicate 必填） | test-verified | `hqsb/distributed/parallel_plan.py`；`tests/unit/distributed/test_parallel_ledger.py::TestCensus`、`::TestTpCapability`、`::TestShardsAndPlan` |
| S10-16 | shard round-trip（row/column）+ 直接分片加载审计（full_then_split 判不通过）+ 不整除只允许 PAD/UNEVEN/REJECT 且覆盖全域 | test-verified | `parallel_plan.verify_shard_ranges`/`audit_direct_load`/`assert_no_truncation`；`test_parallel_ledger.py::TestDirectLoadAndCases`；`TestShardRoundTrip`（属性） |
| S10-17 | 通信账本：expected 从 plan 枚举、observed 独立采集、差异必须带原因码；未解释差异或未配对事件使账本保持开启 | test-verified | `hqsb/distributed/ledger.py`；`test_parallel_ledger.py::TestLedger` |
| S10-18 | 内存对账：分桶（weights/kv/activation/workspace/communicator/graph/allocator/replicated）逐项分解；容量取最大 rank；KV 字节与复制成本入账 | test-verified | `ledger.reconcile_memory`/`rank_skew`/`kv_bytes`；`test_parallel_ledger.py::TestLedger`；`TestLedgerAndMemoryConservation`（属性） |
| S10-19 | scaling 诚实性：无真实 T1 只给 `speedup_from_p0`（不给 efficiency）；缺失资源格为 MISSING+None+reason；weak 定义必须具名；pairability 用 17 字段判定并保留不可配对行 | test-verified | `hqsb/distributed/scaling.py`；`tests/unit/distributed/test_scaling_overlap.py::TestScalingPrereg`、`::TestResourceMatrixAndBaseline`、`::TestDecompositionAndPairability` |
| S10-20 | 时间分解恒等式：compute+exposed_comm+wait+host_gap+sync+unexplained = total，overlap 只减一次，未解释 idle 显式报告 | test-verified | `scaling.TimeDecomposition`；`TestDecompositionAndPairability` |
| S10-21 | overlap 由区间集合计算（merge + 交集测度，不重复计时）；无独立 compute 时拒绝伪重叠；时钟不确定性覆盖差值时结论降级；三组 schedule 只允许一个变量；work handle 生命周期校验 | test-verified | `hqsb/distributed/overlap.py`；`test_scaling_overlap.py::TestIntervalAlgebra`、`::TestOverlapSchedule`；`TestIntervalAlgebra`/`TestOverlapMetricsBounds`（属性） |
| S10-22 | 重叠因果裁决矩阵（§11）与 guardrail 列表落地为数据 + `overlap_verdict` 四态判定；phase/shape policy 带安全回退 | test-verified | `overlap.CAUSAL_MATRIX`/`overlap_verdict`/`phase_policy`；`test_scaling_overlap.py::TestOverlapSchedule::test_overlap_verdict_chain`、`::test_phase_policy_separates_prefill_and_decode` |
| S10-23 | PP/CP/SP P1 门禁：未声称时为 NOT_RUN_NOT_CLAIMED（拒绝标成 PASS）；未命名算法的候选被删除；rubric 打分 + 只选一个主方案；adopt/reject 标准与重新评估触发条件 | test-verified | `hqsb/distributed/boundary.py`；`tests/unit/distributed/test_boundary_moe.py::TestBoundaryActivation`、`::TestBoundarySelection`、`::TestBoundaryVerdict` |
| S10-24 | PP/CP/SP 结构与成本：分区守恒、最慢 stage 决定吞吐、max stage 决定容量、bubble 理想式与实测差、CP/SP 按具名算法算 steps/bytes | test-verified | `boundary.verify_partition`/`stage_balance`/`bubble_metrics`/`cp_sp_metrics`；`test_boundary_moe.py::TestBoundaryPlans` |
| S10-25 | MoE 完成层级与声明门（L0–L4）；无真实 artifact 时不得声明模型质量；dispatch/combine oracle 处理零 token expert/tail/重复 top-k；CountMatrix 守恒与 checked offsets | test-verified | `hqsb/distributed/moe.py`；`test_boundary_moe.py::TestMoeRouting`、`::TestMoeOracle`；`TestMoeConservation`（属性） |
| S10-26 | MoE 负载不均：max/mean、CV、Gini、熵、热点持续性、communication cut、critical-rank load；skew profile 覆盖 uniform→severe→热点对→时变；capacity/drop 明确标注质量语义改变 | test-verified | `moe.imbalance_metrics`/`SKEW_PROFILES`/`capacity_policy_rows`；`test_boundary_moe.py::TestMoeOracle`、`::TestMoePlacementAndVerdict` |
| S10-27 | MoE placement 因果：baseline/treatment 只改 expert→rank；holdout 漂移阈值触发回退；memory 上限阻止"复制缓解"假设 | test-verified | `moe.placement_ab_rows`/`holdout_check`/`topology_aware_placement`；`test_boundary_moe.py::TestMoePlacementAndVerdict` |
| S10-28 | 多 rank trace：事件字段冻结、group+epoch+seq 配对（缺 rank 进 unmapped）、arrival/completion skew、per-rank 分解平衡、通信矩阵、未映射比例阈值 | test-verified | `hqsb/distributed/traces.py`；`tests/unit/distributed/test_traces_telemetry.py::TestTraceEventsAndClocks`、`::TestPairingAndSkew` |
| S10-29 | 归因纪律：从最早超基线事件出发；证据矩阵必须齐备否则 UNKNOWN；分类器打分含 unknown rate；放大指标区分 victim_wait 与用户延迟 | test-verified | `traces.RootCauseClassifier`/`EVIDENCE_MATRIX`/`classifier_scores`/`amplification_metrics`；`test_traces_telemetry.py::TestAttribution` |
| S10-30 | 注入与链路整形：注入计划带 start/end marker 与 cleanup；无隔离网络/无审批的链路整形返回 NOT_RUN（不得伪造真实网络故障） | test-verified | `traces.injection_plan`/`link_shaping_plan`；`test_traces_telemetry.py::TestInjectionPlans` |
| S10-31 | C6/C7 投影不改冻结契约：`summary["s10"]` 命名空间 + 事件 attributes；未知 span kind 拒绝；统一数据表 schema（topology edge/collective sample/parallel event/scaling row）逐行校验 | test-verified | `hqsb/distributed/telemetry.py`；`test_traces_telemetry.py::TestTelemetryProjection` |
| S10-32 | 12 份冻结配置严格加载（未知键/未知 kind/缺失文档均拒绝）并与代码逐字段审计（含 fault matrix 覆盖 19 类故障、tolerance 覆盖 5 类必需 collective） | test-verified | `hqsb/distributed/specs.py`、`configs/distributed/*.yaml`；`tests/unit/distributed/test_s10_experiment_scaffolding.py::TestSpecs` |
| S10-33 | 实验步骤→代码接口对照：300 步 / 381 接口 / 548 引用全部解析（`resolve_interfaces().ok=true`），映射不可腐烂 | test-verified | `hqsb/distributed/interface_map.py`；`test_s10_experiment_scaffolding.py::TestInterfaceMap` |
| S10-34 | 前置门与结论拒绝：`check_prerequisites` 只读协议树（不自解锁）；verdict 需 (allow_execute ∧ 前置 ∧ raw samples>0)；模板不可带结论状态；run 布局 34 子目录 | test-verified | `hqsb/distributed/experiment.py`；`test_s10_experiment_scaffolding.py::TestPrerequisites`、`::TestVerdictRefusal`、`::TestRunDirectory` |
| S10-35 | 依赖边界：无模块级 torch/triton/numpy、无 torch 时整层可导入（子进程探针）、下游 10 个区域不反向依赖、distributed 不 import ops；gate R1–R8 0 违规 0 环 | test-verified | `tests/unit/distributed/test_distributed_import_boundaries.py`；`scripts/audit/import_dependency_gate.py`（rules=1.3.0，files=193，edges=430） |
| S10-36 | 全量测试 **2134 passed, 4 deselected**（S10 新增 240 = 225 单元 + 15 属性） | runtime-verified（本机） | `.venv/bin/python -m pytest -m "not hardware and not e2e and not performance" -q` |
| S10-37 | 驱动默认拒绝：`--prerequisites` 实测 9 条中 8 条 MISS → `BLOCKED`；`--smoke` CPU 自检通过并标注 smoke/`claim_allowed=false`；`--interface-map` 输出 300 步对照 | runtime-verified（本机） | `scripts/distributed/run_e10.py`；`docs/reports/S10_阶段验收报告.md` §5 |
| S10-BLOCK | S10 **实验层 BLOCKED**：8/9 硬前置未满足（S07 P0 verdict、S08 trace、双加速器、封存 topology manifest、冻结 backend 身份、单卡 reference、冻结 model/workload、协议树指纹）；E10-07 为 NOT_RUN_NOT_CLAIMED | source-only（缺失事实，由 `check_prerequisites` 实测登记） | `hqsb/distributed/experiment.py::check_prerequisites`；`run_e10.py --experiment E10-01 --prerequisites` |
| S10-LIMIT | 未覆盖边界（如实登记）：无第二加速器/多节点/RDMA/Ascend 设备；collective 只经 CPU loopback 自检（`claim_allowed()=False`）；4 份配置为 template 待冻结；依赖门禁尚未覆盖 `hqsb.ascend` 区域（S09 交付缺口） | source-only | `hqsb/distributed/collectives.py::LoopbackCollectiveExecutor`；`configs/distributed/{timeout,scaling,fault,tolerance}_spec.yaml`；`docs/reports/S10_开发报告.md` §2.2/§8 |

---

## 16. S11 声明台账（接口/代码层，2026-09-19）

> 基线：`b83ebde`（+ S10 未提交交付）。本节**不含任何 capture/lowering/autotune/cache 结论数字**；
> S11-24 及以后是本机测试与驱动行为证据（`development` 级别），**不是** E11 实验结论；
> E11-01~E11-10 实验层全部 BLOCKED（3/6 必需前置未满足，见 S11-BLOCK）。

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| S11-01 | 多级 artifact 身份：`ArtifactIdentity` 要求 9 个必需字段、双 hash（raw/canonical）、volatile IR 层强制 canonical hash；`identity_hash()` 排除 created_at/command/raw_hash | test-verified | `hqsb/compiler/identity.py`；`tests/unit/compiler/test_ir_identity.py`（TestArtifactIdentity） |
| S11-02 | canonicaliser 只剥离登记在 `NOISE_KEYS` 的噪声键并**记录被剥离路径**；地址/临时符号归一化；语义键永不在噪声表内 | test-verified | `identity.canonicalize_payload`/`check_noise_keys_do_not_touch_semantics`；`tests/unit/compiler/test_ir_identity.py`（TestCanonicalisation）；`tests/property/test_compiler_invariants.py`（TestCanonicalHashAlgebra） |
| S11-03 | 版本冻结：`latest/main/HEAD/unknown/stable` 与空值一律拒绝（`version_is_frozen`），driver 与 target 快照共用同一判定 | test-verified | `identity.version_is_frozen`/`require_frozen_versions`；`tests/unit/compiler/test_ir_identity.py`（TestVersionFreeze）；`tests/unit/compiler/test_lowering_dispatch.py`（TestTargetSnapshot） |
| S11-04 | lineage DAG：重复 id、未知父节点、自环与环检测；按实验给出必需 IR 链（`IDENTITY_CHAINS`）并报告缺失层 | test-verified | `identity.LineageGraph`/`identity_chain_status`；`tests/unit/compiler/test_ir_identity.py`（TestLineage） |
| S11-05 | 三合一身份（graph/semantic/compile）显式标注为**审计键**（不冒充框架内部 cache key），digest 稳定且逐项敏感 | test-verified | `identity.graph_identity`/`semantic_identity`/`compile_identity`；`tests/unit/compiler/test_ir_identity.py`（TestIdentities） |
| S11-06 | HQSB canonical/targeted IR：符号维度 origin/范围校验、effect 三态（PROVED_SAFE/PROVED_UNSAFE/UNKNOWN_UNSAFE）、UNKNOWN 必须带 reason、targeted 层强制候选/选择/fallback | test-verified | `hqsb/compiler/ir.py`；`tests/unit/compiler/test_ir_identity.py`（TestIRTypes）、`::TestVerifier` |
| S11-07 | IR verifier 14 项检查：def-use、重复 id、悬空值、未知 producer/user、环、约束缺失、目标层缺候选/选择/fallback 均结构化报错（带 code/op_id/severity） | test-verified | `ir.IRVerifier`/`verify_graph`；`tests/unit/compiler/test_ir_identity.py`（TestVerifier）；`TestIRVerifierInvariants`（属性） |
| S11-08 | 序列化 round-trip 保持 canonical hash；未知字段拒绝（严格 schema）；human-readable/text 形式与机器形式同源 | test-verified | `ir.IRGraph.to_json/from_dict/to_text/round_trip`；`tests/unit/compiler/test_ir_identity.py`（TestSerializationAndDiff） |
| S11-09 | capture 矩阵覆盖 region×mode×workload 与负向 layout 格；operator-only 格被显式拒绝为 P0；六 workload 与有序 shape trace 冻结 | test-verified | `hqsb/compiler/capture.py`；`tests/unit/compiler/test_capture_guards.py`（TestCaptureMatrix）、`::TestTraceAndRepeatability` |
| S11-10 | graph break 可定位：reason 目录 9 类、必须带 file/line/native reason/fallback action；`UNKNOWN_BREAK` 不允许出现在报告行 | test-verified | `capture.GraphBreak`/`BreakAudit`；`tests/unit/compiler/test_capture_guards.py`（TestBreakAudit） |
| S11-11 | metadata 采集：source/shape/layout/effect 四维完整度单列；fake↔runtime 比对给出字段级差异；缺失必须带 `missing_reason` | test-verified | `capture.SourceLineage`/`TensorMetadataRecord`/`compare_fake_runtime`/`MetadataCompleteness`；`tests/unit/compiler/test_capture_guards.py`（TestMetadata） |
| S11-12 | coverage 口径：op/time/hotspot/phase/pattern 五类分别计算；UNAVAILABLE 不代填；`coverage_claim_guard` 拒绝用 op count 支撑时间/hotspot 声明 | test-verified | `capture.CoverageReport`/`coverage_claim_guard`；`tests/unit/compiler/test_capture_guards.py`（TestCoverage）；`TestCaptureCoverageAlgebra`（属性） |
| S11-13 | 无优化 debug backend：只记录并返回 `gm.forward`，结果结构上标记 `is_debug_backend=True`（不得被表述为已编译） | test-verified | `capture.DebugBackend`/`DebugBackendResult`；`tests/unit/compiler/test_capture_guards.py`（TestCensusAndBackend）；`tests/unit/compiler/test_backend_codegen.py`（TestBackendPlan::test_capture_only_plan_is_a_debug_backend） |
| S11-14 | 原生日志与 HQSB 事件 join：未映射行比例与明细可查；repeatability 检查 canonical 稳定、break/guard 集稳定，并声明同进程复验的局限 | test-verified | `capture.NativeTracePlan`/`join_native_and_hqsb`/`repeatability_check`；`tests/unit/compiler/test_capture_guards.py`（TestTraceAndRepeatability） |
| S11-15 | hero graph handoff manifest：必须带 pattern site、符号约束、编译器版本与**允许对外声称的 capture scope** | test-verified | `capture.HeroGraphManifest`；`tests/unit/compiler/test_capture_guards.py`（TestTraceAndRepeatability::test_hero_manifest_requires_sites_constraints_and_versions） |
| S11-16 | guard 分类 9 类 + 每类失败动作；performance guard 不得声明为语义必需；语义 guard 不得以"仅性能"为由删除（删除需声明替代 variant 域） | test-verified | `hqsb/compiler/guards.py`；`tests/unit/compiler/test_capture_guards.py`（TestGuardTaxonomy）、``（TestCompileAccounting::test_guard_removal_requires_a_replacement_domain） |
| S11-17 | variant 语义：`hit` 要求全部 guard 为真且 artifact 兼容；guard false + reuse 被 `GuardEventRecord.validate` 直接拒绝（wrong reuse 自动 FAIL）；graph hash 相同不构成 hit | test-verified | `guards.VariantRegistry.lookup`；`tests/unit/compiler/test_capture_guards.py`（TestVariantRegistry）；`TestGuardDomainAlgebra`（属性） |
| S11-18 | 域分析：重叠必须有确定优先级、洞必须有 fallback；覆盖率为样本级证据，不超 1 | test-verified | `guards.domain_coverage`；`tests/unit/compiler/test_capture_guards.py`（TestVariantRegistry::test_domain_coverage_reports_holes_and_priority_conflicts）；`TestGuardDomainAlgebra` |
| S11-19 | 重编译可解释：每个 compile 必须映射 guard/path/domain；`unexplained_compile_rate` 非零即不通过；策略对照按部署 horizon 总成本排序（≥3 策略） | test-verified | `guards.compile_explainability`/`compare_strategies`；`tests/unit/compiler/test_capture_guards.py`（TestCompileAccounting） |
| S11-20 | variant 预算与并发：超限明确 fallback/error（禁止静默 eager）；并发只允许完整发布，半成品读取计数为 0 | test-verified | `guards.VariantBudget`/`evaluate_concurrency`；`tests/unit/compiler/test_capture_guards.py`（TestCompileAccounting） |
| S11-21 | pattern 合同：输入/输出/数学/effect/supported/**unsupported** 齐备；signature 与 Python 类名无关；contract digest 稳定 | test-verified | `hqsb/compiler/pattern_library.py`；`tests/unit/compiler/test_rewrite.py`（TestContractsAndSignature） |
| S11-22 | 结构匹配与语义判定分离：matcher 只找候选（含部分命中的"最接近层级"），in-place/alias 变体仍进入候选但由 predicate 拒绝 | test-verified | `rewrite.find_candidates`；`tests/unit/compiler/test_rewrite.py`（TestMatcher） |
| S11-23 | 8 条 predicate 按冻结顺序执行，输出 expected/actual/outcome/proof_source/reject_code；epsilon 允许集、reduction/keepdim、cast 目标、用户/liveness、alias/effect、layout、return contract 逐项可解释 | test-verified | `rewrite.evaluate_predicates`/`Decision.reject_reason`；`tests/unit/compiler/test_rewrite.py`（TestPredicates） |
| S11-24 | 语料零误判：12 类语料（positive/decomposed/near-miss/hard negative/already-fused）FP=0、FN=0，每个拒绝都带 predicate 级理由 | test-verified | `rewrite.build_corpus_rows`/`evaluate_corpus`；`tests/unit/compiler/test_rewrite.py`（TestCorpusAndPipeline::test_corpus_has_no_false_positives）；`TestRewriteIdempotence`（属性） |
| S11-25 | 重写原子性：注入点 matcher/replacement/verifier/commit 四处失败均不改变原图、不留下半改写图；提交后才返回新图（原对象不可变） | test-verified | `rewrite.apply_rewrite`/`atomicity_report`；`tests/unit/compiler/test_rewrite.py`（TestRewrite）；`TestRewriteIdempotence`（test_atomicity_holds_for_every_injection_point） |
| S11-26 | pass 纪律：确定性、固定点收敛、二阶 rewrite=0、元数据不增殖、pass identity 进入 compile identity | test-verified | `rewrite.run_pipeline`/`idempotence_report`/`determinism_report`/`pass_identity`；`tests/unit/compiler/test_rewrite.py`（TestCorpusAndPipeline） |
| S11-27 | provenance 保留：重写后 fused op 保留 source lineage、pattern id、effect 证据与 fallback 子图引用；verifier 失败即回滚 | test-verified | `rewrite.provenance_audit`/`metadata_diff`；`tests/unit/compiler/test_rewrite.py`（TestRewrite::test_rewrite_commits_and_preserves_provenance）、`::test_verifier_blocks_a_broken_replacement` |
| S11-28 | CPU 参考 oracle：fp16/bf16 真实舍入、composed vs fused 的 reduction 输入差异（不依赖 allclose 偶然一致） | test-verified | `hqsb/compiler/pattern_library.py`（composed/fused reference、compare_reference_paths）；`tests/unit/compiler/test_rewrite.py`（TestReferenceOracles） |
| S11-29 | target 快照与 capability：arch/dtype/feature/ABI/资源逐项判定，拒绝原因结构化为 10 类；toolchain 缺失为 `NOT_RUN_TOOL_UNAVAILABLE` 而非估计 | test-verified | `hqsb/compiler/targets.py`；`tests/unit/compiler/test_lowering_dispatch.py`（TestTargetSnapshot）、`::TestCapability` |
| S11-30 | lowering registry：候选必须带 capability/guard/artifact/evidence/fallback；编译候选必须有 build id + artifact locator/hash；注册无副作用（不编译/不建 context） | test-verified | `hqsb/compiler/lowering.py`；`tests/unit/compiler/test_lowering_dispatch.py`（TestRegistry） |
| S11-31 | 选择链输出**完整候选表**：semantic → capability → artifact → guard → evidence → policy，逐候选 reject reason；forced unsupported 明确失败/回退，不静默改 actual | test-verified | `lowering.evaluate_candidates`/`parse_policy`；`tests/unit/compiler/test_lowering_dispatch.py`（TestSelection） |
| S11-32 | dispatch 纪律：actual≠selected 必须带 fallback reason（禁止静默回退）；selected 必须在 eligible 内；实际 kernel 需 ≥2 类独立证据 | test-verified | `lowering.DispatchRow`/`DispatchTelemetry`/`ActualDispatchEvidence`；`tests/unit/compiler/test_lowering_dispatch.py`（TestDispatch） |
| S11-33 | 副作用安全：check→launch→commit，禁止"先跑半个 kernel 再回退"；可能已写状态的 runtime error 默认 FAIL_REQUEST（除非有事务性证明） | test-verified | `lowering.SideEffectGate`/`preflight_checks`；`tests/unit/compiler/test_lowering_dispatch.py`（TestSideEffectsAndFailures） |
| S11-34 | 失败注入：wrong_arch/missing_symbol/schema/ABI/guard/compile/runtime 七类均**不 load、不 launch**，只产生结构化 reason | test-verified | `lowering.inject_lowering_failure`；`tests/unit/compiler/test_lowering_dispatch.py`（TestSideEffectsAndFailures::test_injected_failures_never_load_or_launch） |
| S11-35 | 正确性/性能顺序与成本口径：correctness 7 门前缀语义、性能 6 步、compile breakdown 键集合固定且**不含** steady state（`warm_steady_runtime` 单独测） | test-verified | `lowering.CORRECTNESS_ORDER`/`gate_sequence_ok`/`performance_allowed`/`compile_breakdown`、`records.COMPILER_COST_KEYS`/`validate_timing_breakdown`；`tests/unit/compiler/test_lowering_dispatch.py`（TestOrdersAndCost） |
| S11-36 | backend 契约：九步全部记录 status/timing/output；缺 materialize 时拒绝"返回 gm.forward 当作已编译"；步失败后其后续步不得标 ok | test-verified | `hqsb/compiler/backend.py`；`tests/unit/compiler/test_backend_codegen.py`（TestBackendPlan） |
| S11-37 | CompileRun 投影：步骤计时聚合到统一成本键（`merge_step_timings`）；成功编译必须产出 artifact manifest id；manifest 标记 lineage 完整性 | test-verified | `records.merge_step_timings`/`CompileRun.validate`、`backend.emit_artifact_manifest`；`tests/unit/compiler/test_backend_codegen.py`（TestArtifactManifest） |
| S11-38 | codegen 制品：生成源必须带 entry symbol + parent IR + raw/canonical 双 hash；导出计划逐层标注可用性（缺工具即 NOT_RUN）；SASS 必须绑定正式 binary hash | test-verified | `hqsb/compiler/codegen.py`；`tests/unit/compiler/test_backend_codegen.py`（TestGeneratedSource）、`::TestToolchainAndResources` |
| S11-39 | 跨层归因：resource 解析对缺字段保留 unknown（不补零）；memory/launch ledger 计算节省与差异；机制表要求 IR→binary→counter→runtime + ablation + 替代解释；无证据为 INCONCLUSIVE、反驳须有 counter observation | test-verified | `codegen.parse_resource_usage`/`memory_plan_diff`/`launch_ledger_diff`/`MechanismHypothesis`；`tests/unit/compiler/test_backend_codegen.py`（TestMechanisms）、`::TestToolchainAndResources` |
| S11-40 | Amdahl 归因与 profiler 计划：kernel saving→phase 预测含残差与 fallback 率；profiler 计时不得作为主性能值；有 mutation 的 kernel 禁止多 pass replay | test-verified | `codegen.amdahl_attribution`/`ProfilePlan`；`tests/unit/compiler/test_backend_codegen.py`（TestMechanisms::test_amdahl_residual_is_reported）、``（TestProfilingAndReconstruction::test_mutating_kernel_profile_drops_replay_commands） |
| S11-41 | 重建与差异：semantic rebuild 与 bitwise identical 分开报告；跨层 diff 行覆盖 ir/generated_source/binary/runtime | test-verified | `codegen.reconstruction_check`/`cross_layer_diff_rows`；`tests/unit/compiler/test_backend_codegen.py`（TestProfilingAndReconstruction）、`::TestIRToCodegenChain` |
| S11-42 | autotune 空间：knob 必须带语义；约束必须带类别/理由/可判定 predicate；候选 identity 覆盖 config+kernel build+target；生成超预算即拒绝 | test-verified | `hqsb/compiler/autotune.py`；`tests/unit/compiler/test_autotune.py`（TestSearchSpace）、`::TestCandidateGeneration` |
| S11-43 | 预算与公平：B0–B3 单调、必须有 early stop；不同方法预算不一致被判不公平；split 按语义组切分且无泄漏 | test-verified | `autotune.budget_ladder`/`check_budget_fairness`/`build_split_manifest`/`split_leak_check`；`tests/unit/compiler/test_autotune.py`（TestBudgets）、`::TestSplits` |
| S11-44 | trial 纪律：invalid 候选不得被计时（latency=∞ 不是观测）；correctness 必须先于 benchmark；winner 必须独立 confirmation；测量顺序随机/轮转可复现 | test-verified | `records.AutotuneTrialRecord.validate`/`autotune.correctness_before_benchmark`/`measurement_schedule`；`tests/unit/compiler/test_autotune.py`（TestTrials）、`::TestOracleAndWinners` |
| S11-45 | holdout 与成本：interpolation/boundary/layout-dtype 三类 holdout 有预注册门与 fallback；搜索成本分阶段；无收益时 break-even 标 NEVER | test-verified | `autotune.holdout_evaluate`/`search_cost_breakdown`/`amortization`；`tests/unit/compiler/test_autotune.py`（TestHoldoutAndCost） |
| S11-46 | tuning DB 与策略：key 11 字段齐全（缺一即拒绝）；策略文档禁止在线调优 stateful kernel | test-verified | `autotune.tuning_database_record`/`autotune_policy_document`；`tests/unit/compiler/test_autotune.py`（TestDatabaseAndPolicy） |
| S11-47 | cost model 特征：schema 版本化、泄漏字段（latency/winner/oracle/counter/test id）禁止、关键特征缺失强制 fallback | test-verified | `hqsb/compiler/costmodel.py`；`tests/unit/compiler/test_costmodel.py`（TestFeatureSchema） |
| S11-48 | 数据与切分：CI 重叠标 tie、split 必须分组且锁定、final test 只能消费一次、random-row 只能作诊断 | test-verified | `costmodel.build_labels`/`SplitManifest`/`build_split`；`tests/unit/compiler/test_costmodel.py`（TestLabelsAndSplits） |
| S11-49 | 基线与模型：default/heuristic/random/global-best/nearest/oracle 齐备；模型族含常数/线性/pairwise，fit/predict 在未训练时拒绝 | test-verified | `costmodel.evaluate_baselines`/`model_families`；`tests/unit/compiler/test_costmodel.py`（TestBaselinesAndModels） |
| S11-50 | 主指标：holdout chosen-vs-oracle regret（mean/p95/max/catastrophic/加权），空数据为 UNAVAILABLE 而非 0；排序指标 tie-aware；selection overhead 单列 | test-verified | `costmodel.RegretReport`/`ranking_metrics`/`selection_overhead`；`tests/unit/compiler/test_costmodel.py`（TestMetrics）、`::TestConfidenceAndPolicy` |
| S11-51 | 安全回退链：无合法候选→语义 fallback；OOD/缺特征→heuristic；低置信→benchmark top-k 或 fallback；模型只能在合法集中排序（非法候选注入被阻断） | test-verified | `costmodel.select_with_policy`/`illegal_candidate_injection`/`ModelArtifactGuard`；`tests/unit/compiler/test_costmodel.py`（TestConfidenceAndPolicy）；`TestCostModelInvariants`（属性） |
| S11-52 | 裁决分离：methodology PASS 与 deployment PASS 分开；deployment 门含 regret/tail/catastrophic/overhead/confidence 单调性 | test-verified | `costmodel.DeploymentCriteria`/`methodology_verdict`；`tests/unit/compiler/test_costmodel.py`（TestConfidenceAndPolicy::test_deployment_criteria_separate_pass_from_fail） |
| S11-53 | cache key 规范：22 字段逐项含 inclusion reason；非语义字段（timestamp/pid/output root…）禁止入 key；缺字段即报错（不静默空串）；key test vectors 与 noise 稳定性可复跑 | test-verified | `hqsb/compiler/cache.py`；`tests/unit/compiler/test_cache.py`（TestKeySpec）；`TestCacheKeyAlgebra`（属性） |
| S11-54 | 事务发布：TEMP→VALIDATING→PUBLISHED→COMMITTED（marker 最后）；writer 在任一阶段被杀均不留可读半成品；同 key 二次发布被拒绝；temp 残留可清理 | test-verified | `cache.EntryStore.publish`/`cleanup_temp`；`tests/unit/compiler/test_cache.py`（TestTransactions） |
| S11-55 | 安全读取：parse→schema→key→compat→guard→state→metadata hash→payload hash→load；任一失败 `load_calls=0`；marker 与 manifest 状态不一致即拒绝；metadata 被编辑由自身 hash 检出 | test-verified | `cache.EntryStore.read`/`metadata_fingerprint`；`tests/unit/compiler/test_cache.py`（TestSafeReader） |
| S11-56 | 损坏矩阵 11 例（截断/篡改/翻转/交换/未知 schema/缺 marker/temp 残留/权限/符号链接/writer 被杀）全部在执行前拒绝并隔离 | test-verified | `cache.inject_corruption`/`CORRUPTION_CASES`/`quarantine`；`tests/unit/compiler/test_cache.py`（TestSafeReader::test_corrupted_entries_are_rejected_before_load）、`::test_quarantine_moves_the_entry` |
| S11-57 | 失效矩阵 14 行（噪声必须 HIT、语义/编译/arch/ABI 必须 MISS、guard false 必须 MISS/FALLBACK）；观测与矩阵不符判不通过 | test-verified | `cache.INVALIDATION_MATRIX`/`evaluate_invalidation`/`matrix_table`；`tests/unit/compiler/test_cache.py`（TestInvalidation） |
| S11-58 | 并发/键遗漏/淘汰：半成品读取使评估失败；hidden config 必须在 key 内否则判碰撞风险；淘汰不删 in-use 且可重建 | test-verified | `cache.evaluate_concurrency`/`key_omission_detector`/`EvictionPolicy`；`tests/unit/compiler/test_cache.py`（TestConcurrencyAndOmission）、`::TestEvictionAndPolicy` |
| S11-59 | C0–C4 计时边界与 cache 指标：reset scope 显式（不用模糊 clear_all）、OS page cache 是否清理如实记录；false hit/corrupt execution 必须为 0（安全优先于命中率） | test-verified | `cache.timing_boundaries`/`cache_metrics`/`cache_policy_document`；`tests/unit/compiler/test_cache.py`（TestTelemetryAndMetrics）、`::TestEvictionAndPolicy` |
| S11-60 | 第二编译栈选择与作用域：主栈二选一、版本冻结、scope ceiling 明确；语义映射逐行必须声明 must-preserve；未映射语义默认 block 或显式 external call | test-verified | `hqsb/compiler/portable.py`；`tests/unit/compiler/test_portable_aigate.py`（TestStackSelection）、`::TestSemanticMapping` |
| S11-61 | legalisation 与 loop IR：legal/dynamic-legal/illegal/external 四态、analysis-only 先于 codegen；loop IR 校验 bounds/读写域/reduction init-update/tail/type | test-verified | `portable.analysis_only_legality`/`LoopIRSpec`/`verify_loop_ir`；`tests/unit/compiler/test_portable_aigate.py`（TestLegalization）、`::TestLoopIRAndSchedule` |
| S11-62 | schedule 与 pass trace：每步必须带硬件理由且 IR hash 变化；trace 可重放（无执行器时明确 STRUCTURE_ONLY）；pass 链 hash 连续 | test-verified | `portable.ScheduleStep`/`ScheduleTrace`/`replay_schedule`/`pass_pipeline_trace`；`tests/unit/compiler/test_portable_aigate.py`（TestLoopIRAndSchedule） |
| S11-63 | bridge 契约：ownership/stride/device/stream/sync/error 六项必填；零拷贝声明与 copy-to-contiguous 矛盾即拒绝；bridge overhead 无 timeline 时为 UNAVAILABLE | test-verified | `portable.BridgeContract`/`bridge_audit_plan`/`bridge_overhead`；`tests/unit/compiler/test_portable_aigate.py`（TestBridge） |
| S11-64 | 开发成本与采用裁决：成本 rubric 与 latency 分离；角色对照 8 维齐全；采用需 correctness（+性能证据，若声称生产后端）；允许的公开结论限定 4 类 | test-verified | `portable.development_cost_rubric`/`role_comparison`/`adoption_decision`；`tests/unit/compiler/test_portable_aigate.py`（TestCostAndAdoption） |
| S11-65 | AI 候选信任边界：task package 冻结 reference/harness hash、allowed APIs、forbidden actions、资源上限；provenance 字段完整方可进入 G0 | test-verified | `hqsb/compiler/aigate.py`；`tests/unit/compiler/test_portable_aigate.py`（TestTaskAndProvenance） |
| S11-66 | 沙箱与 harness：无网络/无 secrets/只读 reference/正式 registry/cache 不可写/配额齐全；候选 API 面不得包含 timer/test paths/reference | test-verified | `aigate.SandboxPolicy`/`HarnessLock`；`tests/unit/compiler/test_portable_aigate.py`（TestSandboxAndHarness） |
| S11-67 | 门链 G0–G10 累积：11 道门全部实现 accept/reject；后门 accept 无法覆盖前门 reject（`later_accepts_after_reject` 显式记录）；wrong corpus C0–C11 预期门与实现一致 | test-verified | `aigate.GATE_ORDER`/`run_gate_chain`/`gate_g0..gate_g10`；`tests/unit/compiler/test_portable_aigate.py`（TestGateChain） |
| S11-68 | 作弊与安全门：hardcode/partial output、无同步虚假计时、timer/input/reference 篡改、OOB、race、state 变异各自被对应门拦截；sanitizer 不可用时降级为 canary 证据并降低声明 | test-verified | `aigate.gate_g1..g8`；`tests/unit/compiler/test_portable_aigate.py`（TestGateChain::test_wrong_candidate_classes_are_rejected_at_the_expected_gate）、`::test_sanitizer_unavailable_lowers_the_claim_but_accepts_with_canary` |
| S11-69 | 指标与 admission：`fast_p` 分母含 correctness 失败（不静默剔除）；Pareto 保留 memory/compile 权衡；admission schema 缺任一必需项即 SCHEMA_FAIL；`claim_boundary` 限制对外措辞 | test-verified | `aigate.fast_p`/`resource_pareto`/`validate_admission`/`claim_boundary`；`tests/unit/compiler/test_portable_aigate.py`（TestAdmissionAndMetrics） |
| S11-70 | C6/C7 投影与表 schema：C6 字段含 CompileRun/selection/dispatch/cache/artifact/guard/autotune 映射；C7 事件 kind 映射 + span 链完整性；12 张 raw 表逐行校验 | test-verified | `hqsb/compiler/telemetry.py`；`tests/unit/compiler/test_s11_experiment_scaffolding.py`（TestTelemetryProjection） |
| S11-71 | 12 份冻结配置严格加载（未知键/未知 kind/缺失文档均拒绝）并与代码逐字段审计（capture 轴、guard 分类、pattern 契约、capability 原因、lowering 政策、预算梯度、特征组、cache key/失效/损坏矩阵、legalisation/bridge、候选类别/门链、tolerance、实验目录） | test-verified | `hqsb/compiler/specs.py`、`configs/compiler/*.yaml`；`tests/unit/compiler/test_s11_experiment_scaffolding.py`（TestInterfaceMapAndSpecs） |
| S11-72 | 实验步骤→代码接口对照：**320 步 / 417 唯一接口 / 693 引用**全部解析（`resolve_interfaces().ok=true`），映射不可腐烂 | test-verified | `hqsb/compiler/interface_map.py`；`tests/unit/compiler/test_s11_experiment_scaffolding.py`（TestInterfaceMapAndSpecs::test_interface_map_resolves_every_reference） |
| S11-73 | 前置门与结论拒绝：`check_prerequisites` 只读仓库证据（9 条，含 2 条 advisory）；verdict 需 (allow_execute ∧ 前置 ∧ raw samples>0)；模板不可带结论；run 只写 `experiment_results/S11/` | test-verified | `hqsb/compiler/experiment.py`；`tests/unit/compiler/test_s11_experiment_scaffolding.py`（TestPrerequisites）、`::TestVerdictRefusal`、`::TestRunDirectory` |
| S11-74 | 依赖边界：无模块级 torch/triton/numpy、无 torch 时整层可导入（子进程探针）、10 个下游区域不反向依赖、compiler 不 import ops、只依赖 core；gate R1–**R10** 0 违规 0 环 | test-verified | `tests/unit/compiler/test_compiler_import_boundaries.py`；`scripts/audit/import_dependency_gate.py`（rules=1.4.0，files=214，edges=481） |
| S11-75 | 全量测试 **2549 passed, 4 deselected**（S11 新增 **415** = 396 单元 + 19 属性） | runtime-verified（本机） | `tests/unit/compiler/`、`tests/property/test_compiler_invariants.py`；命令 `.venv/bin/python -m pytest -m "not hardware and not e2e and not performance" -q` |
| S11-76 | 驱动默认拒绝：`--prerequisites` 实测 6 条必需中 3 条未满足 → `satisfied=false`；`--smoke` CPU 自检通过并标注 smoke/`claim_allowed=false`；`--interface-map` 输出 320 步对照；`--spec-audit` 12/12 通过 | runtime-verified（本机） | `scripts/compiler/run_e11.py`；`docs/reports/S11_阶段验收报告.md` §5 |
| S11-BLOCK | S11 **实验层 BLOCKED**：3/6 条必需前置未满足（`s03_s04_kernel_hardware_evidence`、`s06_pattern_model_level_correctness`、`frozen_compiler_environment`）；E11-01~E11-10 无任何 run、无 raw、无结论 | source-only（缺失事实，由 `check_prerequisites` 实测登记） | `hqsb/compiler/experiment.py`（check_prerequisites）；`run_e11.py --prerequisites` |
| S11-LIMIT | 未覆盖边界（如实登记）：无 S03/S04/S06 协议树 verdict、无冻结编译器环境指纹、无 nvcc/cuobjdump/nvdisasm/nsys/ncu 导出工具、无 Ascend/多卡资源；TVM/MLIR 与真实 Qwen/Inductor 运行均未执行（仅接口与 CPU oracle）；E11-10 为 P1 未激活 | source-only | `docs/reports/S11_开发报告.md` §2/§8；`run_e11.py --prerequisites` |

---

## 17. S12 声明台账（接口/代码层，2026-09-19）

> 基线：`b83ebde`（+ S10/S11 未提交交付）。本节**不含任何可比性/capability/性能/能耗/成本/成熟度结论数字**；
> S12-18 及以后是本机测试与驱动行为证据（`development` 级别），**不是** E12 实验结论；
> E12-01~E12-10 实验层全部 `NOT_RUN`（4 条必需前置未满足，见 S12-BLOCK）。

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| S12-01 | 三种身份（byte/canonical/aggregate root）+ 版本化 canonicalization（volatile 字段剥离、NaN/Inf 拒绝）+ 逻辑 URI（`hqsb://S12/<campaign>/<type>/<id>`，非绝对路径） | test-verified | `hqsb/evaluation/identity.py`；`tests/unit/evaluation/test_lineage.py` |
| S12-02 | 缺失语义（10 类结构化状态）与"状态≠0"强制（`numeric_value` 对 missing 状态抛错）；status 传播（§3.2）为数据；182 张表的必填字段清单 | test-verified | `hqsb/evaluation/records.py`；`tests/unit/evaluation/test_comparability.py`、`test_benchmark.py` |
| S12-03 | Comparison Contract（语义/质量/workload/timing/statistics/normalization 六段，schema 校验）+ 49 字段分类 + 16 审计维度 + reason codes | test-verified | `hqsb/evaluation/contracts.py`；`tests/unit/evaluation/test_comparability.py`（TestPolicyAndVerdicts） |
| S12-04 | 四态裁决引擎（evidence→quality→forbidden/invariant diff→conditional→comparable 顺序不可改）；CONDITIONAL 无公式降级 NOT_COMPARABLE | test-verified | `hqsb/evaluation/comparability.py`；`tests/unit/evaluation/test_comparability.py` |
| S12-05 | 非法 join 防护：11 类稳定错误码；跨 contract/跨 layer/quality fail/missing/actual-backend 未知/CONDITIONAL 未 opt-in 均拒绝 | test-verified | `comparability.illegal_join_guard`；`tests/unit/evaluation/test_comparability.py`（TestCellAudit） |
| S12-06 | candidate 身份来自 canonical 字段（显示名不作 join key）+ 上游证据五态（VERIFIED/…/INCOMPATIBLE）；`require_usable` 对缺失 fail-closed | test-verified | `hqsb/evaluation/candidates.py`；`tests/unit/evaluation/test_comparability.py`（TestCandidateIdentity/TestUpstreamEvidence） |
| S12-07 | capability 四级证据（DECLARED/DISCOVERED/VERIFIED/BENCHMARKED）+ 69 个细粒度 feature + 升级/失效规则；probe 失败分类 13 类，失败不当 unsupported；silent fallback 不得验证 requested feature | test-verified | `hqsb/evaluation/capability.py`；`tests/unit/evaluation/test_eval_capability.py` |
| S12-08 | 平台身份多来源交叉核对（冲突整组 blocked）+ telemetry 字段 canonical 化（缺 boundary/unit/semantics → MEASUREMENT_UNAVAILABLE，不伪造 canonical 值） | test-verified | `hqsb/evaluation/platform.py`；`tests/unit/evaluation/test_eval_capability.py`（TestPlatformLayer） |
| S12-09 | 四层统一数据契约：Observation（append-only + actual backend 必填 + 非 OK 不带数值）、NormalizedResult（source ids/formula/unit）、11 条交叉校验约束逐条实现 | test-verified | `hqsb/evaluation/benchmark.py`；`tests/unit/evaluation/test_benchmark.py` |
| S12-10 | 冷启动/预热/thermal/health 分段：cold 不混入 steady、warmup 收敛或报失败、thermal 越界 WAIT/MARK、health 只标记不删除 | test-verified | `hqsb/evaluation/benchmark.py`；`tests/unit/evaluation/test_benchmark.py`（TestStages） |
| S12-11 | 重复性：层级（process/block/day 为正式层级）、平衡 schedule（Latin square）、预注册阈值、异常规则只盯系统指标、排除账本强制 performance-blind basis | test-verified | `hqsb/evaluation/repeatability.py`；`tests/unit/evaluation/test_repeatability.py` |
| S12-12 | 统计（纯 Python、确定性）：bootstrap 区间（run-level）、分组方差、漂移、配对 log ratio、frontier membership；best-of-N 被结构化拒绝 | test-verified | `hqsb/evaluation/repeatability.py`；`tests/unit/evaluation/test_repeatability.py`、`tests/property/test_evaluation_invariants.py` |
| S12-13 | Roofline：四类 roof 分层、theoretical 不得作主预测、logical/expected/measured traffic 分离、calibration/validation 不泄漏且 validation 只消费一次、残差无证据为 INCONCLUSIVE | test-verified | `hqsb/evaluation/roofline.py`；`tests/unit/evaluation/test_eval_roofline.py` |
| S12-14 | 能量：四类 boundary（不同边界不同排序）、accumulator 与积分双路交叉校验、total/incremental 同报、分母只用 compliant work、不确定性不得小于分辨率 | test-verified | `hqsb/evaluation/energy.py`；`tests/unit/evaluation/test_energy.py` |
| S12-15 | 成本：**无内置价格/电价/汇率**（一律 dated 输入）、peak 不得当分母、SKU 映射校验、10 项双重计数审计、独立复算、break-even/单源敏感性 | test-verified | `hqsb/evaluation/cost.py`；`tests/unit/evaluation/test_cost.py`（含 `test_no_hardcoded_prices_in_the_module`） |
| S12-16 | Pareto：六画像机读模板、quality/SLO 硬约束、NO_FEASIBLE_CANDIDATE 一等结果、成熟度三合法路径、10 用例决策回归（标注 algorithm_regression，非实验） | test-verified | `hqsb/evaluation/pareto.py`；`tests/unit/evaluation/test_pareto.py` |
| S12-17 | 成熟度：8 维锚定 rubric（0–4，NOT_EVALUATED≠0）、四类工时分离、silent fallback 高风险、分歧报告不平均、维护工时区间 | test-verified | `hqsb/evaluation/maturity.py`；`tests/unit/evaluation/test_maturity.py` |
| S12-18 | lineage：entity 追加式（同 id 异 hash 拒绝）、DAG cycle/dangling/duplicate/orphan 定位、reverse trace 零人工、fault injection 必须定位到实体与下游 claim、per-transform diff 容差 | test-verified | `hqsb/evaluation/lineage.py`；`tests/unit/evaluation/test_lineage.py` |
| S12-19 | 12 份冻结配置（词汇表，无数字）严格加载 + 与代码逐字段审计；`comparability/capability/benchmark/repeatability/roofline/energy/cost/pareto/maturity/lineage/campaign/experiment` 12 kind | test-verified | `hqsb/evaluation/specs.py`、`configs/evaluation/*.yaml`；`tests/unit/evaluation/test_e12_scaffolding.py`（TestInterfaceMapAndSpecs） |
| S12-20 | 实验步骤→代码接口对照：**360 步 / 415 唯一接口 / 727 引用**全部解析（`resolve_interfaces().ok=true`），映射不可腐烂 | test-verified | `hqsb/evaluation/interface_map.py`；`tests/unit/evaluation/test_e12_scaffolding.py` |
| S12-21 | 前置门与结论拒绝：`check_prerequisites` 只读仓库证据（4 必需 + 3 advisory）；verdict 需 (allow_execute ∧ 前置 ∧ raw samples>0)；run 只写 `experiment_results/S12/` | test-verified | `hqsb/evaluation/experiment.py`；`tests/unit/evaluation/test_e12_scaffolding.py`（TestPrerequisites/TestRunDirectory） |
| S12-22 | 依赖边界：无模块级 torch/triton/numpy、无 torch 时整层可导入（子进程探针）、11 个下游区域不反向依赖、evaluation 不 import ops、只依赖 core；gate R1–**R12** 0 违规 0 环 | test-verified | `tests/unit/evaluation/test_evaluation_import_boundaries.py`；`scripts/audit/import_dependency_gate.py`（rules=1.6.0，files=236，edges=550） |
| S12-23 | 全量测试 **2922 passed, 4 deselected**（S12 新增 **373** = 319 单元 + 38 属性 + 16 边界/脚手架） | runtime-verified（本机） | `tests/unit/evaluation/`、`tests/property/test_evaluation_invariants.py`；命令 `.venv/bin/python -m pytest -m "not hardware and not e2e and not performance" -q` |
| S12-24 | 驱动默认拒绝：`--prerequisites` 实测 4 条必需中 3 条未满足（+ 1 条本层提供）→ `satisfied=false`；`--smoke` 十模块 CPU 自检通过并标注 smoke/`claim_allowed=false`；`--interface-map` 输出 360 步；`--spec-audit` 12/12 | runtime-verified（本机） | `scripts/evaluation/run_e12.py`；`docs/reports/S12_阶段验收报告.md` §5 |
| S12-BLOCK | S12 **实验层 BLOCKED**：必需前置 `s03_s11_upstream_evidence_chain`（实测仅 S01/S02 有 verdict）、`multi_hardware_coverage`、`frozen_evaluation_environment` 未满足；E12-01~E12-10 无任何 run、无 raw、无结论 | source-only（缺失事实，由 `check_prerequisites` 实测登记） | `hqsb/evaluation/experiment.py`（check_prerequisites）；`run_e12.py --prerequisites` |
| S12-LIMIT | 未覆盖边界（如实登记）：无 ≥3 硬件（或 2 硬件+2 架构）实例、无 S03–S11 协议树 verdict、无冻结评估环境指纹、无 profiler/功率计/价格快照；energy/cost 只能到接口层；`artifacts/S12/` 目录约定落到 `experiment_results/S12/`（仓库既有约定） | source-only | `docs/reports/S12_开发报告.md` §2/§8/§9 |


---

## 18. S13 声明台账（生产化 / 云原生 / 可靠性）

> 本节**不含任何实验结果数字**。S13 实验层为 `BLOCKED`（见 S13-BLOCK），下列声明全部是
> 代码层/测试层事实：`test-verified` 表示有自动化测试证明，`runtime-verified` 表示本机命令实测。

| ID | 声明 | 等级 | 证据 |
|---|---|---|---|
| S13-01 | ReleaseBundle 身份不完整不得 ready/部署；tag 不得作为部署身份；模型权重不得进入通用镜像 | test-verified | `hqsb/infra/identity.py`；`tests/unit/infra/test_infra_foundations.py::TestIdentity` |
| S13-02 | OCI index→manifest→config→layer/DiffID 闭包可比对；可重建四级（build repeatability / functional equivalence / bit reproducible / not reproducible）不互相冒充，未归因不得升级 | test-verified | `identity.oci_digest_dag`/`compare_oci_dag`/`reproducibility_verdict`；同上 |
| S13-03 | rollback 链必须能走到 known-good；canary control/candidate 只允许预注册差异 | test-verified | `identity.validate_rollback_chain`/`validate_release_change_scope`；同上 |
| S13-04 | 三套状态机（pod/artifact/request）合法转换受约束，非法跳转被拒 | test-verified | `hqsb/infra/records.py`；`tests/property/test_infra_invariants.py` |
| S13-05 | 缺失是状态不是 0（`numeric_value` 拒绝 missing 码）；§24 状态传播阻断依赖声明 | test-verified | `records.numeric_value`/`propagate_statuses`；属性测试 |
| S13-06 | §23 十四条交叉约束（V01–V14）实现为可引用 reason code 的校验器 | test-verified | `hqsb/infra/contracts.py`；`tests/unit/infra/test_infra_foundations.py::TestContracts` |
| S13-07 | §21 制品目录 52 个子目录 + campaign manifest + §20.2 执行安全策略与 target 授权（wildcard/越界拒绝） | test-verified | `hqsb/infra/campaign.py`；同上 `TestCampaign` |
| S13-08 | 供应链 gate：secret/模型命中即 FAIL；scanner 不可用 ≠ 零漏洞；SBOM completeness 找孤儿；异常必须带 owner/期限；attestation subject 必须等于部署 digest | test-verified | `hqsb/infra/supply_chain.py`；`test_e13_infra_experiments_a.py::TestSupplyChain` |
| S13-09 | clean 污染显式报告；部署状态机路径校验；readiness 需语义条件（healthz 不算）；pre-ready 流量排除；首请求必须与 reference 一致；cold/warm 不得混报；残留即判失败；临场命令即 deviation | test-verified | `hqsb/infra/deployment.py`；同文件 `TestDeployment` |
| S13-10 | capability label 的类/来源/TTL/保护受约束；硬 capability 不得退化为软偏好；容器可见设备集必须是分配子集；rank 映射不符即失败；隔离只对已验证层声明 | test-verified | `hqsb/infra/scheduling.py`；同文件 `TestScheduling` |
| S13-11 | cache key 必须含 tokenizer/config/quant/engine/ABI；下载不得写 active 路径；marker 必须在数据 durable 之后；GC 不得删 active/pinned/rollback 目标；请求不得混版本 | test-verified | `hqsb/infra/artifacts.py`；同文件 `TestArtifacts` |
| S13-12 | 三类 probe 端点必须不同；终止预算覆盖生成+flush；drain 后不得接新执行；截断流不得记完成；forced kill 记为明确违约；资源释放逐项核对 | test-verified | `hqsb/infra/lifecycle.py`；同文件 `TestLifecycle` |
| S13-13 | 内存账本逐组件/逐 rank（最大 rank 约束不被总量掩盖）；KV 估算含 block rounding；admission 决策带 policy/reason/budget；reserve/commit 原子；cancel/error 归还预算 | test-verified | `hqsb/infra/capacity.py`；`test_e13_infra_experiments_b.py::TestCapacity` |
| S13-14 | 残差必须带解释才可扩大 margin；false reject 必须有反事实方法；margin 不得在验证数据上调参；S12 原预测不得被覆盖 | test-verified | `capacity.prediction_residual`/`false_decisions`/`margin_holdout_validation`/`s12_feedback`；同上 |
| S13-15 | desired replicas 由 capacity 模型导出并受 min/max/tolerance 约束；metric 无 timestamp/age 不得驱动决策、stale 一律 fail-safe；降载需持续信号；过冲/波动/settling 有判据；成本（replica/device-minutes）与 SLO 同报 | test-verified | `hqsb/infra/autoscaling.py`；同文件 `TestAutoscaling` |
| S13-16 | 语义约定必须版本化、带单位、含 client/server/core 边界；高基数标识必须显式声明 forbidden；直方图 bucket 必须覆盖 SLO 范围；metric 必须可与 raw 对账 | test-verified | `hqsb/infra/observability.py`；同文件 `TestObservability` |
| S13-17 | 告警必须有 user impact/query/owner/runbook，阈值来源必须标注（模板阈值 = `POLICY_DEFAULT_UNVERIFIED`）；telemetry 缺失判为异常而非 0；RCA 必须带备选解释与置信度、盲化时不得看 ground truth | test-verified | `observability.load_alert_rules`/`rca_verdict`/`telemetry_failure_detection`；同上 |
| S13-18 | 故障合同必须冻结 hypothesis/layer/mechanism/resolved targets/blast radius/预期检测-降级-恢复/abort/kill switch/repetitions/claim 边界；wildcard 与未解析 target 拒绝；语义改变的降级必须重过质量 | test-verified | `hqsb/infra/faults.py`；`test_e13_infra_experiments_b.py::TestFaults` |
| S13-19 | ground truth 生效时间必录（命令返回 ≠ 生效）；MTTD/MTTM/MTTR 以用户稳态为准并受预注册阈值约束；恢复需服务/资源/状态三轴；postmortem 需 owner+验证方式；自动恢复声明只对 `RECOVERED_AUTOMATIC` 成立 | test-verified | `faults.ground_truth`/`reliability_metrics`/`validate_state_recovery`/`postmortem`/`fault_verdicts`；同上 |
| S13-20 | canary 状态机禁止跳阶段；G0–G4 硬门失败即 STOP（性能不可补偿）；流量分配按 session 粘性且可复现；信息量不足不得 promote；`FORCE_PROMOTE` 被拒绝且 override 不得伪装自动；rollback 需请求/身份/质量/资源闭环 | test-verified | `hqsb/infra/canary.py`；同文件 `TestCanary` |
| S13-21 | 多租户威胁模型必须声明 attacker capability/non-goals/failure policy；RBAC 权限图检测间接提权；deny 用例被拒为 PASS、成功攻击为 FAIL；配额并发竞态报最大超卖；滥用必须在分配资源前有界拒绝；噪声干扰以受害者指标计量 | test-verified | `hqsb/infra/security.py`；同文件 `TestSecurity` |
| S13-22 | 12 条租户不变量无证据时为 `NOT_RUN`（不得当 PASS）；结论措辞限定在样本与威胁模型内 | test-verified | `security.invariant_verdicts`/`NEGATIVE_RESULT_WORDING`；同上 |
| S13-23 | §22 七个核心 schema 投影区分必填与可选缺失；93 张表 schema 逐字段校验 | test-verified | `hqsb/infra/telemetry.py`；`tests/unit/infra/test_infra_foundations.py::TestTelemetry` |
| S13-24 | 418 步「实验步骤 → 代码接口」对照由 `PROTOCOL_STEPS` + `resolve_interfaces()` 导入校验；生成物与代码同步（`--check`） | test-verified | `hqsb/infra/interface_map.py`；`scripts/infra/gen_interface_map.py`；`docs/reports/S13_interface_map_generated.md` |
| S13-25 | 驱动默认拒绝：无 `--execute`、前置未满足或无 raw 样本时 `PASS/FAIL/PASS_NEGATIVE` 一律降级 `BLOCKED`；预注册缺阈值/non-claims 直接报错 | test-verified + runtime-verified | `hqsb/infra/experiment.py`；`tests/unit/infra/test_e13_scaffolding_assets.py`；`run_e13.py --experiment E13-01 --json` |
| S13-26 | 依赖边界：`hqsb.infra` 无模块级 torch/triton/numpy/kubernetes、无 torch 时可导入、12 个区域不反向依赖、只依赖 core、不 import ops；gate R1–**R14** 0 违规 0 环；全量 **3118 passed, 4 deselected**（S13 专用新增 195 个） | runtime-verified（本机） | `tests/unit/infra/test_infra_import_boundaries.py`；`scripts/audit/import_dependency_gate.py`（rules=1.7.0，files=256，edges=599） |
| S13-BLOCK | S13 **实验层 BLOCKED**：必需前置 `s08_service_contract`、`s12_capacity_and_quality_baseline` 未满足；本机无 docker/kubectl/helm/cosign/syft/grype/promtool；E13-01~E13-11 无任何 run、无 raw、无结论数字 | source-only（实测登记） | `run_e13.py --prerequisites`；`docs/reports/S13_阶段验收报告.md` §3 |
| S13-LIMIT | 未覆盖边界（如实登记）：`infra/**` 资产为模板（未构建/未渲染/未部署/未演练）；`configs/infra/**` 不含测量值，阈值需 campaign 冻结；§21 逻辑目录落到 `experiment_results/S13/`；E13-11 当前 11/12 不变量为 `NOT_RUN`；mypy 未覆盖 `hqsb/infra` | source-only | `docs/reports/S13_开发报告.md` §8/§9；`docs/reports/S13_production_architecture.md` §5 |

## 19. S14 声明台账（训推协同 / 前沿扩展 / 证据治理）

> 阶段：S14 ｜ 机器：RTX 3090 开发机（x86_64 Linux），`development` ｜ 日期：2026-09-19
> **本阶段未执行任何正式实验**（任务第五节）。因此除"接口/代码层"声明外，
> 所有实验层声明一律 `BLOCKED`，且**不存在任何实验数字**（性能/内存/能耗/命中率/质量均为零）。

| ID | 声明 | 等级 | 证据 |
|---|---|---|---|
| S14-01 | **480 步接口齐备**：12 项实验 × 40 步，每步都映射到可导入符号，无空缺 | runtime-verified（本机，CPU） | `hqsb/experimental/interface_map.py`；`run_e14.py --interface-map` → `experiments=12 steps=480 expected_steps=480 interfaces=589 references=993 ok=True`；`tests/unit/experimental/test_e14_interfaces.py::test_step_table_is_complete`（12 参数化） |
| S14-02 | 接口引用**实际导入解析**：改名/删除即失败，报告不会静默指向空 | runtime-verified | `interface_map.resolve_interfaces()` 对每个符号执行 `importlib` + 属性解析；`gen_interface_map.py --check` 检测生成物漂移 |
| S14-03 | **依赖边界成立**：`hqsb.experimental` 只依赖 `hqsb.core`；13 个下层区域不得反向依赖它（规则 R15） | runtime-verified | `scripts/audit/import_dependency_gate.py`（rules=1.8.0, files=277, edges=667, violations=0, cycles=0, PASS）；`test_experimental_import_boundaries.py::test_lower_regions_do_not_import_experimental` |
| S14-04 | `hqsb.experimental` **不 import `ops`**，且无模块级重依赖（R16）；CPU-minimal 下 `import` 拉入重框架数 = 0 | runtime-verified | 同上 gate；AST 扫描 + **子进程探针** `test_cpu_minimal_import_in_a_subprocess`（`heavy == []`）；`test_no_module_level_heavy_imports` |
| S14-05 | 包入口只提供 PEP 562 惰性映射，模块级不 import 子模块（避免 `experimental → experiment → specs → experimental` 环） | test-verified | `test_package_init_is_lazy_only`；`__init__.py` 的 `_LAZY` |
| S14-06 | **E14-01 依赖/flag 隔离接口**（40 步）：extra↔flag↔capability↔CLI 映射、wheel 产物元数据审计、导入纯度、未知/冲突 flag 拒绝、10 类负例 | test-verified | `hqsb/experimental/dependencies.py`；`test_e14_interfaces.py`；`run_e14.py --smoke` |
| S14-07 | **E14-02 分布式训练状态接口**（40 步）：global batch 等价、token-mean 重归一化、逐层对账、collective/显存账本、checkpoint 完整性、resume 连续性、缺 shard/rank failure fail-closed | test-verified | `training.py`；属性测试 `test_token_mean_loss_differs_from_the_naive_rank_mean`、`test_reconciliation_names_the_first_divergent_tensor` |
| S14-08 | **E14-03 训推一致性接口**（40 步）：转换 DAG、显式 mapping（同名覆盖被拒）、**七层语义门**、adapter merge 语义、6 类错误制品负例、原子发布与 cache | test-verified | `parity.py`；`test_e14_interfaces.py` |
| S14-09 | **E14-04 后训练接口**（40 步）：SFT/DPO/GRPO **手算 oracle**、token mask/长度归一化、policy lineage、mixed-version 拒绝、有界异步队列、staleness 扫描、reward-hacking 判定 | test-verified | `posttraining.py`；属性测试 `test_staleness_sweep_reports_every_missing_point` |
| S14-10 | **E14-05 前沿预注册接口**（40 步）：文献注册表、estimand（"性能更好"被拒）、硬前置门、单一 primary、AdoptionDecision 预注册、未选分支锁 `N/A_BY_ADR`、协议 hash 冻结 | test-verified | `frontier.py`；`contracts.FrontierStudyContract` |
| S14-11 | **E14-F1/F2/F3/F4 条件 P0 接口**（各 40 步）：accept/residual 手算与 break-even；router 多指标不均衡 + dispatch 无丢重乱序 + total/active 显存分离；KV payload/metadata 分离 + 无截断证明 + chunk 语义不变；N:M compliance + 算法误差与实现误差分离 + actual dispatch | test-verified | `speculative.py` / `moe.py` / `long_context.py` / `sparsity.py`；属性测试 `test_pattern_compliance_requires_every_group`、`test_greedy_prefix_match_never_commits_more_than_target_produced`、`test_imbalance_metrics_separate_balanced_from_skewed` |
| S14-12 | **E14-06/07/08 可选迁移接口**（各 40 步）：任务原生指标（**拒绝用 token/s 表达图像音频**）、子模型纳入 artifact 身份、阶段状态机；13 状态工作流 + tool 校验先于执行 + critical path（≠ span 求和）+ 9 类故障注入；路线 A/B + delegate partition + cold/warm/sustained 三态 + `MAP_ONLY` 不得携带设备测量 | test-verified | `multimodal.py` / `agent.py` / `edge.py`；`test_experiment_has_a_negative_control`（12 参数化） |
| S14-13 | **无静默降级**：任何 fallback 必须记录 requested/actual/reason，且 reason 必须来自冻结词汇表 | test-verified | `contracts.check_no_silent_degradation()`；`test_experimental_foundations.py::test_silent_degradation_is_detected` |
| S14-14 | **质量先于性能**：correctness/quality 未过时 `performance_eligible=True` 被拒 | test-verified | `contracts.check_quality_before_performance()`；`test_quality_must_precede_performance` |
| S14-15 | **结论默认不可能产出**（三重门）：写 `PASS/FAIL/PASS_NEGATIVE/FAIL_PERFORMANCE_HYPOTHESIS` 必须同时满足 `--execute` **且** 前置满足 **且** 有 raw samples；驱动根文件受白名单约束 | runtime-verified | `experiment.py::RunDirectory.write_verdict` / `_resolve`；`test_triple_gate_refuses_a_conclusion`、`test_run_directory_refuses_an_unlisted_artefact`；`run_e14.py --json` → `status: BLOCKED` |
| S14-16 | **协议树只读**：运行产物只落 `artifacts/S14/<实验>/<run>/`；写 `docs/stage_experiments/**` 被 `assert_writable` 拒绝 | test-verified | `campaign.assert_writable` / `run_layout`；属性测试 `test_run_layout_is_always_under_artifacts_and_never_in_the_protocol_tree` |
| S14-17 | **冻结词汇表 8 份**（`configs/experimental/*.yaml`）由代码生成、**不含任何测量值、不含本机绝对路径**，跨文档一致性检查通过 | runtime-verified | `gen_experimental_specs.py --check` → `ok: True`；`run_e14.py --spec-audit` → `kinds=8 ok=True cross_document_problems=0` |
| S14-18 | 四个状态机（conversion node / post-training sync / agent workflow / speculation cycle）与 40 张表 schema 结构自洽 | test-verified | `records.validate_state_machines()` == `[]`；`test_state_machines_are_structurally_valid` |
| S14-19 | **全量测试通过**：3221 passed, 4 deselected（S14 专项 103 项：单元 89 + 属性 14，含 12 实验参数化） | runtime-verified（本机） | `.venv/bin/python -m pytest -m "not hardware and not e2e and not performance" -q` |
| S14-20 | lint 无新增问题（本阶段文件 `All checks passed!`；过程中修复 25 个 F401、1 个真 bug `F821`、1 个死变量 `F841`） | runtime-verified | `ruff check hqsb/experimental scripts/experimental tests/unit/experimental tests/property/test_experimental_invariants.py` |
| S14-21 | 本阶段新增文件**零断链**；`check_docs.py` 报告的 43 处断链全部落在既有文件（未修改） | runtime-verified | `python scripts/check_docs.py`；用 `grep -E "S14|experimental"` 过滤为空 |
| S14-22 | 宿主堆上限修复：`NODE_OPTIONS` 被 `server-main.js` 的 `Hd()` 定向剥离，改以 **node CLI 实参**（`execArgv` 默认继承 `process.execArgv`）注入；宿主 cmdline 实测含 `--max-old-space-size=16384` | runtime-verified（本机） | `scripts/env/patch_vscode_server_heap.sh`；`ps -eo cmd \| grep type=extensionHost`；`v8.getHeapStatistics().heap_size_limit` 实测：默认 4.19 GB → 8192 时 8.19 GB → 16384 时 16.19 GB |
| S14-BLOCK | **S14 实验层 BLOCKED**：必需前置 `upstream_verdicts`、`experimental_environment`、`holdout_isolation` 未满足；`distributed_launcher`/`second_device`/profiler 未解析；E14-03/04/05/F*/06/07/08 的上游链未闭环。**12 项实验无任何 run、无 raw、无结论数字** | source-only（实测登记） | `run_e14.py --prerequisites --json`；`--experiment E14-02 --json` → `status: BLOCKED`, `prerequisites_satisfied: false`, `verdict.json: "conclusion": false`；`docs/reports/S14_阶段验收报告.md` §6.2 |
| S14-LIMIT | 未覆盖边界（如实登记）：① 模块成熟度上限 `SOURCE_INTEGRATED`（无实验执行）；② 未实现真实 trainer/engine/device 调用（R16 有意禁止 import `ops`）；③ E14-06/07/08 **未做** ADR 决策，故状态为未执行而**非** `N/A_BY_ADR`；④ 未选中任何前沿分支，未产生分支能力声明；⑤ 多模态/Agent/端侧**能力未被声称**；⑥ 本阶段不修改 `docs/stage_experiments/**`，协议留白只登记不改写 | source-only | `docs/reports/S14_开发报告.md` §8/§9；`docs/reports/S14_阶段验收报告.md` §9 |

## 20. 环境故障复盘：扩展宿主 OOM（S14 期间，非阶段制品）

> 机器：RTX 3090 开发机（x86_64 Linux）｜ 日期：2026-09-19 ｜ 证据等级：`development`（本机实测）
> 完整叙述：`docs/reports/扩展宿主OOM故障复盘.md`；机器侧原始记录：`AGENTS.md` §8/§9。

| ID | 声明 | 等级 | 证据 |
|---|---|---|---|
| ENV-OOM-01 | **根因**：扩展宿主 V8 堆撞 4 GB 默认上限 → `node::OOMErrorHandler` → `abort()` → `SIGABRT` → 宿主被自动重建。**不是**系统内存不足、不是 OOM-killer、不是网络 | runtime-verified（本机） | `FATAL ERROR: Reached heap limit` + `signal: SIGABRT`（`AGENTS.md` §8.2 摘录）；`oom_kill 0`；7 个候选原因逐一证伪（复盘 §3） |
| ENV-OOM-02 | V8 默认 `heap_size_limit` 是**固定值，不随物理内存缩放**：本机 `totalmem=503.5 GB` 时默认 **4.19 GB**（比值 0.83%） | runtime-verified（本机，node v24.18.1） | `v8.getHeapStatistics().heap_size_limit` 实测；对照：`8192→8.19 GB`、`16384→16.19 GB`（复盘 §5.1） |
| ENV-OOM-03 | **注入点是 node CLI 实参，不是环境变量**：`server-main.js` 的 `Hd()` 从宿主 env 删除 `NODE_OPTIONS` 等 5 个变量；而 `execArgv` 默认继承 `process.execArgv` 并被转发 | runtime-verified（本机） | 宿主 `cmdline` 含 `--max-old-space-size=16384`；`/proc/<pid>/environ` **不含** `NODE_OPTIONS`（正是该判据的假阴性来源） |
| ENV-OOM-04 | 官方 hook `~/.vscode-server/server-env-setup` 在当前 CLI 布局下**从未被 source**（机制性失效，非配置错误） | runtime-verified（带对照组） | 探针文件 `/tmp/hqsb-env-setup-ran.log` 始终未生成；对照 `grep -a -c serve-web`=4 vs `grep -a -c env-setup`=0；`server-main.js` 中 `server-env-setup`=0 |
| ENV-OOM-05 | `Developer: Reload Window` **不重启 server**；关闭窗口后 server **继续存活**。二者都不读启动脚本 | runtime-verified | 重载/关窗后 server pid 与启动时间不变；`Kill VS Code Server on Host` 后才变为新 pid |
| ENV-OOM-06 | 加固脚本的 3 处**报告与状态不一致**已修：① 复验提示查 `environ`（假阴性）② `--check` 报参数值而非实测值 ③ **无参重跑静默降级天花板**（实测 16384→8192） | runtime-verified | `scripts/env/patch_vscode_server_heap.sh`；复验：显式 16384 → 磁盘 16384；无参重跑 → `already patched ×3 / patched 0`，天花板保持 16384 |
| ENV-OOM-07 | **载体降级风险**：6 次 SIGABRT 的原始日志已被 16:27 的 server 重启清理（现存日志最早 `20260919T162322`）；仅存的 16 行摘录原在一个 **gitignored** 的本机文件中 | runtime-verified | `ls -1 ~/.vscode-server/data/logs/ \| head -1` → `20260919T162322`；`git check-ignore -v AGENTS.md` → `.gitignore:94`。**本次已随 `docs/reports/` 进入版本控制**（复盘 附录 B.1） |
| ENV-OOM-LIMIT | 未归因/未复现（如实登记）：① **是哪个扩展在涨堆未定位**（需 heap snapshot，该操作本身加压，本轮不执行）；② cgroup `memory.max` 当时记 90 GB、本次实测 **62.0 GB**，未复现；③ `affinity` 宿主隔离**未稳定生效**（本次宿主数=1），未声称"已隔离" | source-only | 复盘 §3.1 / §6.4 / 附录 E |

---

## S15 交付声明（接口/代码层就位；实验层 BLOCKED，2026-09-19）

> 本阶段与 §1 的 claim 分级口径一致：以下声明**只到 `test-verified` 为止**，
> 任何 `runtime-verified` 及以上（真实 release/复现/上游贡献/参与者结论）**均未声称**。

| # | 声明 | 分级 | 证据路径 |
|---|---|---|---|
| S15-C1 | `hqsb/release/` 19 模块 14,562 行，覆盖 E15-01～E15-11 的 495 个实验步骤、796 条接口引用，逐条 import 可解析 | test-verified | `hqsb/release/interface_map.py::resolve_interfaces`；`tests/unit/release/test_e15_interfaces.py::test_all_steps_resolve`（`steps=495/495 ok=True`） |
| S15-C2 | 三个冻结对象 + ClaimRecord 十门裁决 + ContributionRecord 人/Agent/第三方边界可校验 | test-verified | `hqsb/release/contracts.py`；`tests/unit/release/test_release_foundations.py::TestFrozenObjects/TestClaimRecord/TestContributionRecord` |
| S15-C3 | 负向路径在执行前被拒绝：越级 claim、脏树候选、本地 URI、无负对照扫描器、跨族单位、L3 帮助、复用候选 id | test-verified | `tests/unit/release/` + `tests/property/test_release_invariants.py`（28 项） |
| S15-C4 | 依赖方向 R17/R18 成立，`hqsb.release` 在 CPU-minimal 环境可独立导入（子进程拉入重框架 = 0） | test-verified | `scripts/audit/import_dependency_gate.py`（`rules=1.9.0 violations=0`）；`tests/unit/release/test_release_import_boundaries.py` |
| S15-C5 | 四重门：无 `--execute`／前置未满足／无 raw samples／无 candidate+ledger 时拒绝写 `PASS`/`FAIL` | test-verified | `hqsb/release/experiment.py::RunDirectory.write_verdict`；`scripts/release/run_e15.py --experiment E15-01` → `status=BLOCKED conclusion=false` |
| S15-C6 | 冻结词汇表 8 份无测量值、无绝对路径，漂移可检查 | test-verified | `configs/release/*.yaml`；`hqsb/release/specs.py`；`scripts/release/gen_release_specs.py --check`（`ok=True drifted=0`） |
| S15-C7 | 实验层 11 项全部 `BLOCKED`：无 release candidate/ledger/制品、无 reviewer/参与者/上游授权、加速器/构建器/扫描器未探测 | **blocked** | `scripts/release/run_e15.py --prerequisites --probe`（前置清单逐项列出） |
| S15-LIMIT | 未声称：任何实验结论数字；"quickstart 30 分钟可完成""hero story 已复现""已通过 SBOM/许可证/秘密检查""已完成外部复现/上游贡献""招聘者 5 分钟理解项目"（协议 §3 明列为不能声明） | planned | `docs/stage_experiments/details/S15/README.md` §3 |
