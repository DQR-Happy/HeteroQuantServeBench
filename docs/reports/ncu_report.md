# S02 Nsight Compute 报告（Kernel 微架构分析）

> 阶段：S02
> 日期：2026-08-16
> 工具：Nsight Compute CLI（`ncu`，CUDA 12.6）

## 1. 目的

Nsight Compute 对单个 CUDA kernel 做微架构级剖析，记录：
- memory throughput（DRAM / L2 / shared）；
- SM utilization、achieved occupancy；
- stall 原因（long scoreboard / barrier / memory throttle）；
- register / shared memory 用量；
- 理论 vs 实际 FLOPS 与带宽。

用于给 S03 的 kernel 优化提供"为什么慢"的硬件证据（是带宽受限还是计算受限、
occupancy 是否不足）。

## 2. 采集脚本

```bash
./scripts/bench/ncu_profile.sh --kernel '.*gemm.*' --isl 128 --osl 16 \
    --out-dir reports/dev/ncu/<ts>
```

脚本内部：`ncu --kernel-name <regex> --launch-skip 1 --launch-count 4 --set basic`。

`--kernel` 用正则限定目标 kernel（如 `gemm`、`rms_norm`），避免采集所有 kernel
导致运行过慢。

## 3. 权限与平台限制

- 需 root + `kernel.perf_event_paranoid=0`（与 PyTorch Profiler 相同）。
- **Jetson/L4T 上 Nsight Compute 部分 metric 不可用**（统一内存 + L4T 驱动），
  `--set basic` 为保守选择；若 metric 缺失，降级使用 PyTorch Profiler 的
  kernel 级时间 + Roofline 模型推导带宽/FLOPS 边界。

## 4. 结果读取

```bash
ncu --import reports/dev/ncu/<ts>/ncu_report.ncu-rep --page details
```

关注：`Memory Throughput`、`Compute (SM) Throughput`、`Achieved Occupancy`、
`Warp Cycles Per Issued Instruction`、`Stall Reasons`。

## 5. 状态

- **脚本已就绪**（`scripts/bench/ncu_profile.sh`）。
- 本阶段已由 PyTorch Profiler 确认 GEMM kernel（`ampere_fp16_s16816gemm`、
  `cutlass_80_tensorop_f16_s16816gemm`）主导 decode（~78%）。
- Nsight Compute 的 kernel 微架构证据作为 S03 GEMM 优化的前置输入，建议在
  夜间/无交互时段以 root 运行；Jetson L4T 下如 metric 受限，以 Roofline
  推导结果作为替代证据（见 `pytorch_profile_report.md` §3）。

## 6. 与 S03 的关系

若 `ampere_fp16_s16816gemm` 显示 memory throughput 高而 SM 低 → 带宽受限，
S03 优先做 data layout / 低比特以减少访存；若 SM throughput 高 → 计算受限，
S03 优先做 tensor-core 利用率 / 更优 tile 配置（CUTLASS）。
