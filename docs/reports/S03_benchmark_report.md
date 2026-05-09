# S03 Benchmark 报告（RMSNorm + Fused Residual）

- 日期：2026-08-16
- 硬件：NVIDIA Jetson Orin Nano Super（sm_87, 8 SM）
- 工具：`hqsb_rmsnorm_bench` / `hqsb_fused_residual_rmsnorm_bench`
- 计时：device event（kernel-only）+ host submit+sync 双口径

## 1. RMSNorm 性能矩阵

FP32, block=256（rows=512）：

| hidden | V0 | V1 | V2 | V2 vs V0 |
|---|---|---|---|---|
| 100 | 14.66 | 21.67 | 19.10 | 1.30x（V2 退化） |
| 500 | 44.63 | 48.94 | 69.06 | 1.55x |
| 1024 | 40.03 | 64.12 | 87.53 | 2.19x |
| 2048 | 55.94 | 69.16 | 87.16 | 1.56x |
| 4096 | 54.74 | 70.34 | 92.14 | 1.68x |

block size 扫参（FP32, hidden=2048, rows=512, V2）：

| block | device_ms | bandwidth (GB/s) | occupancy (blocks/SM) |
|---|---|---|---|
| 128 | 0.2226 | 56.56 | 12 |
| 256 | 0.1445 | 87.16 | 6 |
| 512 | 0.0931 | 135.18 | 3 |
| 1024 | 0.1865 | 67.51 | 1 |

> block=512 是甜点：每线程恰好 1 个 `float4`（hidden/512 = 4 元素），128-bit
> 事务满带宽；block=1024 时 occupancy 崩塌至 1。

FP16（hidden=2048, rows=512）：

| variant | device_ms | bandwidth (GB/s) |
|---|---|---|
| v2 (half2) | 0.1867 | 33.73 |

> FP16 half2 退化（见 optimization log 退化 1）。

## 2. Roofline 对照

Orin Nano Super 名义带宽 ~68 GB/s（LPDDR5）。V2 @ block=512 达到 135 GB/s
（effective），超过名义 DRAM 带宽，说明小 tensor 的 RMSNorm 大量命中 **L2 cache**
（`traffic` 含被 L2 吸收的重复读）。对 DRAM-bound 的大 tensor，V2 的有效带宽将
收敛到 ~68 GB/s 上限。

- **V2（FP32）**：87–135 GB/s，已接近/超过 LPDDR5 带宽，属 bandwidth-bound。
- **FP16 half2**：33.73 GB/s，远低于带宽上限，属 **事务宽度受限**（32-bit），
  不是带宽受限 —— 这是退化根因的证据。

## 3. Fused Residual + RMSNorm

FP32, hidden=2048, rows=512, block=256：

| variant | bandwidth (GB/s) |
|---|---|
| v0 (shared) | ~65 |
| v1 (vectorized) | ~65 |

> V0/V1 持平（RAW 依赖主导，见 fused optimization log）。

## 4. 结论

1. RMSNorm V2（float4 + warp shuffle）是旗舰收益版本：+56% (block=256) 到
   +142% (block=512) vs V0，根因是 128-bit 内存事务减少。
2. 三个真实退化均已定位并解释：FP16 half2（32-bit 事务）、小 hidden（负载
   不均）、block=1024（occupancy=1）。
3. Fused V1 无收益（write-then-read 依赖），与 RMSNorm V2 形成鲜明对照，
   说明向量化收益取决于 memory 访问模式。

原始 CSV 数据可通过 benchmark 二进制复现：

```bash
./build/jetson-release/bin/hqsb_rmsnorm_bench --rows 512 --hidden 2048 --dtype fp32 --variant all
./build/jetson-release/bin/hqsb_fused_residual_rmsnorm_bench --rows 512 --hidden 2048 --dtype fp32 --variant all
```
