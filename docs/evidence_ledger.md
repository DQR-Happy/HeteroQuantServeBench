# HQSB Evidence Ledger（证据台账 / Claim Ledger）

> 生成时间：2026-08-17
> 基线 Commit：`4dda6f8`
> 当前阶段：S04（已完成）

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
