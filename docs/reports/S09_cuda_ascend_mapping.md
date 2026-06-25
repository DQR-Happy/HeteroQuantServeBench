# S09 CUDA ↔ Ascend Concept Mapping（异构后端概念映射表）

> 阶段：S09  
> 性质：概念映射与差异说明  
> 生成时间：2026-09-18

---

## 1. 重要声明

**本映射表是概念对照，不是数值等价！**

```text
CUDA warp   ≠  Ascend core      (不同执行模型)
CUDA SM     ≠  Ascend AICore    (不同计算单元)
CUDA shared memory ≠ Ascend UB  (不同片上存储层次)
CUDA DRAM   ≠  Ascend GM        (不同全局内存语义)
```

每个硬件有自己的峰值算力、带宽、容量和软件成熟度。比较时必须：
1. 使用各自设备的实测可持续上界（而非营销峰值）
2. 公开算法遍数、字节模型、tile/core 划分
3. 区分 kernel-only 与 conversion-inclusive 口径
4. 承认软件栈成熟度差异（CUDA 更成熟，Ascend 正在追赶）

---

## 2. 执行模型对照

| CUDA | Ascend | 关键差异 |
|---|---|---|
| **Warp** (32 threads, SIMT) | **Core** (vector+cube units) | Warp shuffle (`__shfl_down_sync`) 无直接对应；需显式归约或共享内存 |
| **SM** (Streaming Multiprocessor) | **AICore** (AI Core) | SM 有 tensor cores (MMA)；AICore 有 Cube/Vector 单元 (MFMA) |
| **Block** (grid of warps) | **Core group** (block_dim cores) | Block dim 在 CUDA 由 grid 决定；在 Ascend 需显式指定 `block_dim` |
| **Shared Memory** (L1 cache-like) | **UB** (On-chip buffer) | Shared memory 可随机访问；UB 有严格预算且需 host 预分配 |
| **Registers** | **Local Tensor** | Register file 对开发者透明；Local Tensor 需显式管理生命周期 |
| **Global Memory** (HBM/GDDR) | **GM** (Global Memory) | 语义类似，但 GM 访问延迟可能更高 |
| **Stream** (async execution queue) | **Stream** (async execution queue) | 语义相似，但 CUDA stream 更成熟，Ascend 工具链较新 |
| **Event** (synchronization point) | **Event** (synchronization point) | 语义相同，可用于跨 stream 依赖 |
| **Warp-level primitives** (`__shfl`, `__syncwarp`) | **No direct equivalent** | 需通过 TPipe/TQue 或显式归约实现 |

---

## 3. 内存层次对照

| CUDA | Ascend | 注释 |
|---|---|---|
| **Registers** | **Local Tensor** | 寄存器文件 vs 核心局部缓存 |
| **Shared Memory** (up to 192KB per SM) | **UB** (up to ~512KB per core group) | Shared memory 灵活；UB 有固定预算且需 host 计算 |
| **L1 Cache / Texture Cache** | **L1 / L2 Cache** | Ascend 有 L1/L2 但不可控；CUDA 可通过配置调整 |
| **Global Memory** (HBM2e/GDDR6X) | **GM** (HBM2) | 带宽相似，但可达性不同 |
| **Constant Cache** | **No direct equivalent** | Ascend 无专用常量缓存 |
| **Texture Cache** | **No direct equivalent** | Ascend 无纹理缓存 |

**关键差异**：
- CUDA 的 shared memory 可由 block 内所有 warp 共享，大小可调（最多 48KB per SM in compute capability 7.x+）
- Ascend 的 UB 是 per-core-group 的固定预算，host 必须精确计算每个 kernel 的 UB 需求并拒绝超预算候选
- CUDA 的 L1/cache 行为部分可控；Ascend 的 L1/L2 对用户透明，无法调优

---

## 4. 编程模型对照

### 4.1 Kernel Launch

| CUDA | Ascend |
|---|---|
| `kernel<<<gridDim, blockDim, sharedMem, stream>>>(args)` | `aclrtLaunchKernel(kernel, args, stream, tilingData)` |
| Grid/block 由 launch 参数决定 | TilingData 由 host tiling 函数预先计算并序列化 |
| `sharedMem` 动态分配 | UB 静态预算，host 计算后检查 |

### 4.2 Tiling

| CUDA | Ascend |
|---|---|
| Implicit: gridDim×blockDim covers total work | Explicit: TilingData encodes rows, hidden, tile_elems, loop_count, tail_elems, base_rows, extra_rows |
| Tail handling: thread index checks | Tail handling: `tail_elems` field + valid mask |
| Multi-block reduction: atomicAdd or warp shuffle | Multi-core reduction: partial workspace + final merge |

**示例：RMSNorm row partition**

```python
# CUDA: each block processes one row (if R <= num_blocks)
# Or multiple rows per block if R < num_blocks

# Ascend: explicit partition
base_rows = rows // block_dim
extra_rows = rows % block_dim
rows_per_core[i] = base_rows + (1 if i < extra_rows else 0)
start_row[i] = i * base_rows + min(i, extra_rows)
```

### 4.3 Data Movement

| CUDA | Ascend |
|---|---|
| `cudaMemcpyAsync(dst, src, size, kind, stream)` | `DataCopy(src, dst, size, stream)` |
| Pinned host memory for async | Host memory must be accessible by device |
| Zero-copy via pinned pages | No zero-copy; explicit copy required |

### 4.4 Synchronization

| CUDA | Ascend |
|---|---|
| `cudaStreamSynchronize(stream)` | `aclrtSynchronizeStream(stream)` |
| `cudaEventRecord(event, stream)` | `aclrtSetEvent(event, stream)` |
| `cudaEventSynchronize(event)` | `aclrtSyncEvent(event)` |
| `__syncthreads()` within block | No direct equivalent; use TPipe/TQue barriers |

---

## 5. 计算单元对照

### 5.1 Vector vs Cube

| CUDA | Ascend |
|---|---|
| **CUDA Cores** (scalar FP32/FP16) | **Vector Unit** (SIMD FP32/FP16/BF16) |
| **Tensor Cores** (MMA FP16/FP32/INT8) | **Cube Unit** (MFMA INT8/INT4/FP16) |
| Fused multiply-add in one instruction | Matrix multiply in one instruction |
| Supported on Volta/Ampere/Hopper | Supported on Da Vinci v1/v2/v3 |

**RMSNorm 映射**：
- CUDA：逐元素平方用 CUDA cores，归约用 warp shuffle
- Ascend：逐元素平方用 Vector unit，归约用 local reduce + multi-core partial merge

### 5.2 Precision Support

| Operation | CUDA | Ascend |
|---|---|---|
| **FP32** | ✅ All SMs | ✅ All AICores |
| **FP16** | ✅ SM80+ (tensor cores) | ✅ All AICores |
| **BF16** | ✅ Ampere+ | ✅ Da Vinci v3+ |
| **INT8** | ✅ Tensor cores | ✅ Cube units |
| **INT4** | ✅ Sparse tensor cores | ✅ Cube units (quantized matmul) |

---

## 6. 性能模型差异

### 6.1 Roofline Model

| Metric | CUDA | Ascend |
|---|---|---|
| **Peak Compute** | 300+ TFLOPS (H100) | 256+ TOPS (910B) |
| **Sustainable Bandwidth** | ~1.5 TB/s (HBM3) | ~1.0 TB/s (HBM2) |
| **Latency** | Low (microseconds) | Higher (milliseconds) |
| **Throughput** | Very high | High |

**注意**：Roofline 点的位置取决于实际可持续上界，而非规格峰值。Ascend 的 Cube 单元在矩阵运算上有优势，但向量操作可能不如 CUDA 高效。

### 6.2 流水线机制

| CUDA | Ascend |
|---|---|
| **Multi-stream concurrency** | **TPipe/TQue** (producer-consumer queues) |
| GPU scheduler handles streams | Explicit queue depth management |
| Overlap via separate streams | Overlap via double buffering (buffer_count=2) |

**Double Buffering Trade-off**：
- CUDA：额外流自动重叠，无需显式管理
- Ascend：需手动分配 2 倍 UB，明确权衡 UB 成本 vs 重叠收益

---

## 7. 错误处理对照

| 场景 | CUDA | Ascend |
|---|---|---|
| **OOM** | `cudaErrorMemoryAllocation` | `ACL_ERROR_MEM_ALLOC_FAILED` |
| **Invalid Argument** | `cudaErrorInvalidValue` | `ACL_ERROR_INVALID_PARAM` |
| **Launch Failure** | `cudaErrorLaunchFailure` | `ACL_ERROR_LAUNCH_FAILED` |
| **Async Error** | `cudaGetLastError()` after sync | `aclrtGetLastErrorCode()` after sync |
| **Device Reset** | `cudaDeviceReset()` | `aclrtResetDevice(deviceId)` |

**关键差异**：
- CUDA 的错误码更丰富，文档更完善
- Ascend 的工具链较新，部分错误码可能未完全稳定

---

## 8. Profiler 指标对照

| CUDA Nsight | Ascend msprof | 注释 |
|---|---|---|
| **GPU Utilization** | **AICore Utilization** | 不同计算单元利用率 |
| **SM Occupancy** | **Core Occupancy** | occupancy 计算方式不同 |
| **Memory Throughput** | **Memory Bandwidth** | 语义相同，计数器名称不同 |
| **Tensor Core Usage** | **Cube Utilization** | 矩阵单元利用率 |
| **Vector Instruction Count** | **Vector Instruction Count** | 可直接对比 |
| **Kernel Duration** | **Kernel Duration** | 相同 |
| **Host-Device Copy Time** | **GM Copy Time** | 相同 |
| **Warps Active** | **Active Cores** | 粒度不同 |

**Profiler 可用性**：
- CUDA：Nsight Systems/Compute 成熟，指标丰富
- Ascend：msprof 较新，部分指标可能 unavailable

---

## 9. 最佳实践对照

### 9.1 Tiling Strategy

| CUDA | Ascend |
|---|---|
| Tune block size (128/256/512 threads) | Tune block_dim (cores) and tile_elems |
| Balance register/shared memory usage | Balance UB budget across kernels |
| Minimize global memory accesses | Minimize GM copies, maximize UB reuse |
| Use shared memory for reduction | Use partial workspace + multi-core merge |

### 9.2 Memory Access Pattern

| CUDA | Ascend |
|---|---|
| Coalesced global memory access | Aligned GM access (alignment_elems) |
| Shared memory bank conflict avoidance | UB alignment constraints |
| Constant cache for read-only data | No constant cache; keep on chip if small |

### 9.3 Asynchronous Execution

| CUDA | Ascend |
|---|---|
| Use multiple streams for overlap | Use TPipe/TQue with explicit queue depth |
| Record events for dependency tracking | Record events for synchronization |
| Profile with Nsight to find stalls | Profile with msprof to find pipeline holes |

---

## 10. 公平比较框架

### 10.1 Tier A：Operator Same-Semantics

**Fixed**：OperatorSpec、shape、logical dtype、layout、eps、test vector、tolerance、计时统计和 byte/FLOP 公式。

**Allowed**：CUDA block/grid/shared-memory/Triton tile；Ascend block_dim/GM/UB/Tiling/TPipe。

**Output**：correctness、kernel/end-to-end latency、GB/s/算术效率、workspace、conversion 和 profile 瓶颈。

### 10.2 Tier B：Model Same-Task

**Fixed**：ModelArtifact、tokenizer/template、输入 token、batch、context、请求输出数、sampling、precision/质量、KV 语义和 measurement boundary。

**Allowed**：各平台 native runtime、graph/kernel 实现；但 actual op、fallback、格式转换必须报告。

**Output**：TTFT、TPOT、E2E、actual tokens、memory、energy、correctness/quality、coverage。

### 10.3 Tier C：Best-Valid Deployment

**Fixed**：模型任务、最低质量、SLO、最大内存/功耗或成本约束。

**Allowed**：各平台最优合法 precision、batch/runtime/kernel。结果说明工程选择，不用于声称单一硬件因果。

---

## 11. 结论

**CUDA 与 Ascend 的比较必须在明确约束下进行**：

1. 同一 OperatorSpec → Tier A 公平比较
2. 同一 ModelArtifact + WorkloadSpec → Tier B 公平比较
3. 同质量/SLO/容量约束 → Tier C 工程选择比较

**禁止**：
- 不同模型/精度/上下文直接比 TPS
- 一侧 kernel-only、一侧 framework E2E
- 一侧实测带宽、一侧营销峰值
- 把 profiler 单个百分比脱离 metric set 解释
- 忽略 format、fallback、workspace 和图编译开销

**允许**：
- 公开算法遍数、bytes、tile/core、tail、actual kernel、经验上界和 runtime gap
- 分解为架构、kernel、runtime、capacity、软件成熟度及 residual
- 明确适用边界（shape、model、precision、version）

---

*生成时间：2026-09-18*  
*版本：draft v0.1*
