# 可移植 Kernel 后端：HIP / ROCm / OpenCL 技术说明（S04 L2）

> 阶段：S04（Triton、CUTLASS/CuTe 与 Kernel DSL）
> 定位：L2 技术说明（迁移路径、差异点），非本阶段实现目标（无 AMD/开放硬件）

## 1. 目的

S04 的核心是"同一 OperatorSpec 下多实现对照"。CUDA 生态已覆盖（手写 CUDA、
Triton、CUTLASS/CuTe、cuBLAS）。本文记录向 **HIP/ROCm 与 OpenCL** 迁移的路径与
关键差异点，作为未来多硬件扩展的决策依据，避免届时重复调研。

## 2. HIP / ROCm

### 2.1 定位

HIP 是 AMD 的 CUDA 移植层：提供 `hip*` API（`hipMalloc`/`hipLaunchKernel`/...）
与 `hipcc` 编译器，源语法与 CUDA 高度同构。ROCm 是 AMD 的对应 CUDA 运行时/数学
库栈（`hipBLAS` ≈ cuBLAS、`hipBLASLt` ≈ cuBLASLt、`rocBLAS`）。

### 2.2 迁移路径（对本项目）

| 组件 | CUDA 现状 | HIP 迁移 |
|---|---|---|
| RMSNorm V0/V1/V2（手写） | `.cu` + `<<<>>>` | 近乎机械替换：`cudaError_t`→`hipError_t`，`__global__`/`__shfl_down_sync` 语法一致；`float4`/`half2` 对应 `float4`/`__half2` |
| C ABI（ctypes 绑定） | `rmsnorm_c_api.cu` | 改编译为 `hipcc`，符号名不变，Python ctypes 层零改动 |
| Triton | `ops/triton/*.py` | **零改动**：Triton 后端自动生成 HIP/ROCm 代码（`triton.compile` 的 target 参数） |
| CUTLASS/CuTe | `third_party/cutlass` | **零改动**：CUTLASS 原生支持 `ArchTag::Sm80` 的 HIP 编译路径（`cutlass::arch::Sm80` 在 ROCm 映射到 CDNA/RDNA） |
| cuBLAS | `torch.matmul` | `torch` 在 ROCm 构建下自动走 `hipBLAS`，Python 层零改动 |

**结论**：本项目的手写 CUDA 是"薄封装"，HIP 迁移成本主要在**编译工具链**（hipcc）
而非源码；Triton/CUTLASS/Python 层则几乎零成本。

### 2.3 关键差异点

1. **Warp size**：CUDA 固定 32，ROCm（CDNA）warp 为 64。S03 的 V1/V2 用了
   `__shfl_down_sync(0xffffffff, ...)` 硬编码 32-lane shuffle 与 `blockDim >> 5`
   的 warp 数计算，**这些假设在 warp=64 下失效**，需参数化为 `warpSize`。
2. **向量化宽度**：`float4`（128-bit）在 CDNA 上对应 `float4`，但 RDNA 的
   wavefront 模型不同，对齐/占用率需重新评估。
3. **Tensor core 命名**：CUTLASS `Sm80` MMA 指令在 ROCm 映射到 `mfma`（CDNA）
   而非 `mma.sync`，CUTLASS 内部已抽象，但手写 MMA（若 S05 低比特自研）需分叉。

## 3. OpenCL

### 3.1 定位

OpenCL 是开放标准的异构计算 API（`clCreateKernel`/`clEnqueueNDRangeKernel`），
跨 NVIDIA/AMD/Intel/FGPA。与本项目的 CUDA 专有特性（warp shuffle、tensor core、
`float4` 内存事务）相比，OpenCL 抽象层更高、峰值控制更弱。

### 3.2 迁移路径

- 手写 CUDA kernel → OpenCL C（`__kernel`/`get_global_id`）：**重写**，因为
  `__shfl_down_sync`、`__syncthreads` 的对应物是 `sub_group` 扩展与 `barrier`，
  语义差异大。
- Triton/CUTLASS → OpenCL：**不支持**（两者目标 CUDA/HIP/ROCm，非 OpenCL）。

### 3.3 关键差异点

1. **无 warp shuffle 原语**：V1/V2 的 warp 归约依赖 `__shfl_down_sync`，OpenCL
   需用 `sub_group_reduce_add`（cl_khr_subgroups）或 shared memory 退化回 V0。
2. **内存事务宽度不可控**：OpenCL 不保证 `float4` 合并为 128-bit 事务（S03 的
   核心优化手段失效），需靠 `vload4`/`vstore4` 尽力而为。
3. **无 tensor core**：GEMM 峰值无法达到 CUDA tensor core 水平。

**结论**：OpenCL 的定位是"最大可移植性"而非"峰值性能"，与本项目的"峰值控制 +
多后端对照"目标部分冲突。仅当需要覆盖 FPGA/非 NVIDIA 非 AMD 设备时才引入，
且需接受 RMSNorm 退化为 V0（shared reduction）、GEMM 退化为标量。

## 4. 决策记录

| 决策 | 结论 |
|---|---|
| HIP/ROCm 是否纳入 S04 实现 | 否（无 AMD 硬件，仅 L2 技术说明） |
| OpenCL 是否纳入 | 否（峰值控制受限，与本项目目标冲突，仅在明确需求时引入） |
| 未来多硬件入口 | Triton（零改动跨 CUDA/HIP）+ CUTLASS（零改动跨 CUDA/HIP）为**首选路径**；手写 CUDA 仅在需要峰值控制时按 arch 分叉 |

## 5. 与 dispatcher 的关系

`ops/capability.py` 与 `ops/dispatcher.py` 已按"能力检测 → 明确 fallback"设计。
未来加入 HIP/ROCm 时，只需：
1. `capability.py` 增加 `rocm_available`/`hip_available` 探测；
2. `dispatcher.py` 的 `_cuda_arch_matches` 泛化为 `_device_backend_matches`；
3. Triton/CUTLASS 的 Python 调用零改动。
