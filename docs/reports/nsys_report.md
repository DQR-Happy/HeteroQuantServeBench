# S02 Nsight Systems 报告（Runtime Overhead 分析）

> 阶段：S02
> 日期：2026-08-16
> 工具：Nsight Systems CLI 2024.5.4（`nsys`）

## 1. 目的

Nsight Systems 用于刻画 CPU/GPU timeline，识别：
- kernel launch gap（launch 开销）；
- CPU/GPU 同步点（`cudaStreamSynchronize` / host-device 等待）；
- CPU/GPU overlap 程度；
- kernel timeline 与内存拷贝；
- GPU idle（launch/sync 导致的空闲）。

与 PyTorch Profiler 的 kernel 级视角互补：PyTorch Profiler 回答"哪个 kernel 慢"，
Nsight Systems 回答"kernel 之间的空隙和同步浪费在哪"。

## 2. 采集脚本

```bash
./scripts/bench/nsys_profile.sh --isl 128 --osl 32 --out-dir reports/dev/nsys/<ts>
```

脚本内部：
- `nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --stats=true`
- 用一个专用 driver 加载一次模型并跑单次 model-core pass，避免把模型加载
  混入 timeline。

## 3. 权限要求

与 PyTorch Profiler 一致，CUDA trace 采集需要：
1. `sudo sysctl -w kernel.perf_event_paranoid=0`（临时）；
2. 以 root 运行（`sudo env PYTHONPATH=... bash scripts/bench/nsys_profile.sh`）。

## 4. 结果读取

```bash
nsys stats --report cuda_gpu_trace reports/dev/nsys/<ts>/model_core.nsys-rep
nsys stats --report cuda_gpu_kern_sum reports/dev/nsys/<ts>/model_core.nsys-rep
```

关注指标：`CUDA API` 调用次数、`kernel` 数量、launch 间隔、`cudaMemset/拷贝`
大小、GPU idle 占比。

## 5. 状态

- **脚本已就绪**（`scripts/bench/nsys_profile.sh`）。
- 本阶段 PyTorch Profiler 已给出 kernel 级 hotspot（GEMM 主导），Nsight
  Systems 的 launch/sync 层分析作为 S03 开始前的补充证据，建议在夜间/无交互
  时段以 root 运行（单次 ~1–2 分钟）。
- 采集结果将归档到 `reports/dev/nsys/<ts>/`（被 `.gitignore` 忽略，本机保留）。

## 6. 与 S03 的关系

若 Nsight Systems 显示 decode 阶段 GPU idle 占比高（launch/sync 主导），S03
应优先做 **kernel 融合 + CUDA Graph** 减少 launch/sync；若 GPU 基本满载，则
确认瓶颈在 GEMM kernel 本身（与 PyTorch Profiler 结论一致），S03 聚焦
CUTLASS/低比特 GEMM。
