# Optimization Log：Fused Residual + RMSNorm

- 日期：2026-08-16
- 算子：Fused Residual + RMSNorm（`ops/cuda/fused_residual_rmsnorm/`）
- 状态：V0 Retained；V1 Retained（作为 FP16 唯一实现）

## 1. 背景与假设

S02 Profiler 显示 Qwen3 decoder 的 `aten::add`（residual）与 `aten::copy_`/
`aten::mul`（RMSNorm 归一化）是 elementwise 热点的组成。将
`hidden = residual + hidden` 与 `hidden = rmsnorm(hidden) * weight` 融合为一个
kernel，假设能省去中间 `x + residual` tensor 的 global 往返。

traffic 模型（每元素字节）：

- 独立 add + rmsnorm：`6n + hidden`（add 读 in+res 写 temp = 3n；rmsnorm 读 temp
  两遍 + 写 out + 读 weight = 3n + hidden）
- fused：`5n + hidden`（第一遍读 in+res 写 out 临时 = 3n；第二遍读 out 写 out
  读 weight = 2n + hidden）

理论上限：`6/5 = 1.2x`（不夸大）。

## 2. 版本结果

### V0 — fused + shared reduction（FP32）

- 结果：~65 GB/s（hidden=2048, rows=512, block=256）。

### V1 — fused + warp shuffle + vectorized

- **假设**：与 RMSNorm V2 相同的向量化假设（float4/half2 减少事务）。
- **结果**：~65 GB/s（与 V0 持平，3 次运行在 52–66 GB/s 间波动）。

## 3. 退化分析：V1 无收益（假设不成立）

**根因**：fused 的瓶颈不是 reduction，也不是 load 事务数，而是 **add 中间结果的
write-then-read（RAW）依赖** —— 第一遍把 `x + residual` 写回 `output`，第二遍必须
把它读回再做归一化。这 3 次 global 访存（写临时、读临时、写结果）才是主流量，
向量化没有减少这一依赖链，因此 V1 的 float4 收益被 RAW 依赖掩盖。

对比 RMSNorm（单算子）：其第二遍读的是**原始 input**（可命中 L2），没有
write-then-read 依赖，向量化因此显著有效。

**结论**：这是"同一优化在两种 memory 模式下效果相反"的教材案例，满足 S03 验收
"解释一次无收益"。

## 4. 决策与后续

- **V0 Retained**：FP32 默认实现（更简单，性能与 V1 持平）。
- **V1 Retained**：FP16 唯一实现（V0 是 FP32-only），但 FP16 同样受 RAW 依赖 +
  half2 32-bit 事务双重退化（~40 GB/s）。
- **后续项**：用 **寄存器驻留**（每线程处理少量元素，把 `x + residual` 暂存于
  寄存器而非写回 global）可消除 RAW 依赖，是 fused 融合收益（1.2x）兑现的关键，
  登记为 S03 后续优化项。

## 5. 融合收益核对

独立 add + rmsnorm（未实现独立 kernel，按 traffic 模型）：fused 理论省 1/6
traffic（6n → 5n）。当前 V0/V1 尚未兑现完整收益（因 RAW 依赖），但融合本身
消除了一个 kernel 的 launch/sync 开销（S02 Nsight 层待证），方向上正确。
