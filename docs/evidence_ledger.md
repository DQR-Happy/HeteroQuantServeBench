# HQSB Evidence Ledger（证据台账 / Claim Ledger）

> 生成时间：2026-08-17
> 基线 Commit：`4dda6f8`（历史正文）；当前工作树基线 `3c2453e`
> 当前阶段：S08（接口/代码层就位，实验层 BLOCKED；§14）
> 追加章节：§10（RTX 3090 / sm_86 跨架构补验）、§11（S05）、§12（S06）、§13（S07）、§14（S08）

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

