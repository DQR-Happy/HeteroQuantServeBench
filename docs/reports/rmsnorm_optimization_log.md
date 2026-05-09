# Optimization Log：RMSNorm V0 → V1 → V2

- 日期：2026-08-16
- 算子：RMSNorm（`ops/cuda/rmsnorm/`）
- 状态：V0/V1/V2 全部 Retained（各有用武之地）

## 1. 版本假设与结果

### V0 — shared-memory tree reduction（S00 baseline）

- **假设**：简单、可推理的正确性优先实现；作为所有后续版本的追踪 baseline。
- **实现**：每 row 一个 block，线程 strided 累加平方和 → shared memory 树形约简。
- **结果**（FP32, hidden=2048, rows=512, block=256）：55.94 GB/s。
- **决策**：**Retained**（baseline + 教学对照）。

### V1 — warp-shuffle reduction

- **假设**：V0 的 shared 树形约简含 `log2(blockDim)` 次 `__syncthreads` 与大量
  shared 读写；warp 内约简可全部用 `__shfl_down_sync` 完成，把 shared 流量从
  "每线程一个值"降到"每 warp 一个值"。
- **实现**：warp 内 shuffle 约简 → 每 warp 写 1 个值到 shared → 第 0 个 warp 跨
  warp 约简。
- **结果**：69.16 GB/s（**+24% vs V0**）。
- **决策**：**Retained**（shared 流量假设成立）。

### V2 — vectorized load + warp shuffle

- **假设**：(a) elementwise 算子 memory-bound，`float4`（128-bit）事务比标量
  事务少 4 倍，提升带宽利用率；(b) FP16 `half2` 让字节数减半。
- **实现**：`float4`（FP32）/ `half2`（FP16）向量化 load + V1 的 warp shuffle
  约简；非对齐 shape 走标量 tail。
- **结果**：
  - FP32 block=256：87.16 GB/s（**+56% vs V0**，+26% vs V1）
  - FP32 block=512：135.18 GB/s（**+142% vs V0**，每线程恰好 1 个 float4）
- **决策**：**Retained**（向量化假设 (a) 强成立；假设 (b) 不成立，见 §2）。

## 2. 退化 / 失败假设分析

### 退化 1：FP16 half2 反而更慢（假设 (b) 不成立）

| shape | FP32 V2 | FP16 V2 |
|---|---|---|
| hidden=2048, rows=512 | 87.16 GB/s (0.145 ms) | 33.73 GB/s (0.187 ms) |

**根因**：`half2` 是 32-bit 事务，与标量 `float` 相同，未利用 128-bit 内存事务；
`float4` 才是满带宽路径。FP16 的字节减半优势被 32-bit 事务宽度 + `half2float`/
`float2half` 转换指令抵消。

**结论**：FP16 RMSNorm 应在下一步用 `float4` 加载 8 个 `half`（128-bit 事务）重做，
才可能兑现 traffic 减半的收益。本版本按约定**不为了版本数量追加无假设变体**，
half8 优化登记为 S03 后续项。

### 退化 2：小 hidden（100）V2 慢于 V1

| hidden | V0 | V1 | V2 |
|---|---|---|---|
| 100 | 14.66 | 21.67 | 19.10 |

**根因**：hidden=100 时 vec_count=25，block=256 下仅 25 个线程有 float4 工作，
其余 231 线程空闲（负载严重不均）；向量化收益被负载不均抵消。

**结论**：小 hidden 是 V2 的退化 shape；dispatcher 对 hidden 极小（< 256）可
选 V1 或更小 block。

### 退化 3：block=1024 时 occupancy 崩塌

| block | V2 (GB/s) | occupancy (blocks/SM) |
|---|---|---|
| 256 | 87.16 | 6 |
| 512 | 135.18 | 3 |
| 1024 | 67.51 | 1 |

**根因**：block=1024 时 shared 内存 + 寄存器使每 SM 仅 1 个 block，occupancy 下降，
延迟无法被隐藏。

**结论**：block size 有甜点（512，每线程 1 个 float4）；dispatcher 默认 256 是
保守选择，benchmark 扫参证明 512 更优。

## 3. 逐版本决策汇总

| 版本 | 加速 vs V0 | 保留 | 理由 |
|---|---|---|---|
| V0 | 1.00x | ✅ | baseline/教学 |
| V1 | 1.24x | ✅ | warp shuffle 减少 shared 流量 |
| V2 (FP32) | 1.56–2.42x | ✅ | float4 128-bit 事务满带宽 |
| V2 (FP16 half2) | 0.39x | ⚠️ 退化 | half2 是 32-bit 事务，见退化 1 |

## 4. 与 S03 验收的对应

- **显著加速**：V2 float4（+56% block=256，+142% block=512），由 128-bit 内存
  事务减少解释。
- **无收益/退化**：FP16 half2（退化 1）、小 hidden V2（退化 2）、block=1024
  occupancy 崩塌（退化 3）。
