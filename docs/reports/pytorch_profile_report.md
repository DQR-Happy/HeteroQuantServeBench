# S02 PyTorch Profiler 报告（Hotspot 证据）

> 阶段：S02
> 日期：2026-08-16
> 工具：`torch.profiler`（CPU + CUDA activities）
> 采集方式：`sudo env PYTHONPATH=... python3 scripts/bench/profile_model.py`
> （CUPTI 需 root；`kernel.perf_event_paranoid=0` 已临时设置）

## 1. 采集协议

- 两个代表性 case：
  - `prefill_heavy`：ISL=1024, OSL=2（prefill 主导）
  - `decode_heavy`：ISL=128, OSL=16（decode 主导）
- `record_shapes=True`、`profile_memory=True`；
- prefill 与 decode 在同一 profiler 上下文，复用 KV cache；
- 结果：`reports/dev/profiler/s02/{prefill_heavy,decode_heavy}_operators.json`、
  `hotspot_summary.json`、`hotspot_analysis.json`。

## 2. 结果

### 2.1 Prefill-heavy（total self CUDA ≈ 2.18 s）

| share | 分类 | 算子 |
|---|---|---|
| 24.9% | GEMM | `aten::mm` |
| 12.5% | GEMM | `cutlass_80_tensorop_f16_s16816gemm_relu` |
| 6.7% | elementwise | `aten::copy_` |
| 5.5% | elementwise | `aten::mul` |
| 4.9% | GEMM | `cutlass_80_tensorop_f16_s16816gemm_relu`（第二变体） |
| 3.9% | elementwise | `unrolled_elementwise_kernel` |

**GEMM 合计 ≈ 42–48%**，elementwise（copy/mul/add）合计 ≈ 35–40%。

### 2.2 Decode-heavy（total self CUDA ≈ 4.17 s）

| share | 分类 | 算子 |
|---|---|---|
| 39.9% | GEMM | `aten::mm` |
| 28.4% | GEMM | `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8` |
| 8.1% | GEMM | `ampere_fp16_s16816gemm_fp16_128x64_ldg8` |
| 2.9% | elementwise | `aten::copy_` |
| 2.1% | elementwise | `aten::mul` |
| 1.4% | elementwise | `aten::cat` |
| 1.3% | GEMM | `cutlass_80_tensorop_f16_s16816gemm_relu` |

**GEMM 合计 ≈ 78%**，elementwise ≈ 10%。

## 3. Roofline / Amdahl 分析

Roofline 模型（Orin Nano Super FP16 名义值）：

| 参数 | 值 |
|---|---|
| peak_flops | 67e12 FLOPS |
| peak_bandwidth | 68e9 B/s |
| ridge point | 985.3 FLOP/byte |

Amdahl 上限（对 decode 主导的 `aten::mm`，share 39.9%）：

```
amdahl_max = 1 / (1 - 0.399) = 1.664
```

即：即使把 `aten::mm` 加速到无穷，decode 整体最多加速 1.66×；若连同
ampere gemm 变体（合计 ~78%）一起加速，理论上限约 4.5×。

## 4. Hotspot Decision（S03 输入）

结论（`scripts/bench/analyze_hotspots.py` 生成）：

> GEMM dominates decode (78%) and is the largest prefill class (48%).
> Do NOT write GEMM from scratch; target CUTLASS/低比特 (int4/int8) GEMM +
> epilogue fusion. Keep RMSNorm as the elementwise/reduction teaching loop,
> and fuse aten::copy_/mul/cat to cut launch/sync overhead.

三个 S03 候选：

1. **GEMM（CUTLASS / 低比特 int4/int8 + epilogue fusion）** —— 最大 Amdahl 上限。
2. **RMSNorm（教学闭环）** —— elementwise/reduction 路径的固定教学案例。
3. **KV-cache / elementwise 融合** —— 融合 `copy_/mul/cat` 减少 launch/sync 与
   内存流量。

## 5. 排除项

- 本报告数字为 **相对 hotspot 排名 + 硬件证据**，非官方 latency baseline；
  profiling 引入额外开销并改变调度。
- model-core TTFT 不包含 tokenizer / HTTP / queue / network。
