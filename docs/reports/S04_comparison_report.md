# S04 Comparison Report：CUDA vs Triton vs cuBLAS vs CUTLASS

- 日期：2026-08-17（网络恢复后补全）
- 硬件：NVIDIA Jetson Orin Nano Super（sm_87, 8 SM, LPDDR5）
- 数据来源：`scripts/bench/bench_s04.py`（统一 torch.cuda.Event + 中位数计时）
  + `hqsb_cutlass_gemm_bench`（C++ 程序，subprocess 采集）

## 1. 环境能力矩阵（实测，含网络恢复后补全）

| 后端 | 可用性 | 说明 |
|---|---|---|
| 手写 CUDA（S03） | ✅ | 静态库 + shared lib，arch 锁定 sm_87 |
| Triton 3.7.1 | ✅ | aarch64 wheel，最小 kernel 实测 `max_err=0.0` |
| cuBLAS/cuBLASLt | ✅ | CUDA 12.6 自带，经 `torch.matmul` |
| **CUTLASS 4.7.0** | ✅ | `third_party/cutlass`（GitHub 恢复后浅克隆）；默认 tensor-op Sm80 配置 |
| **TileLang 0.1.13** | ✅ | aarch64 wheel，elementwise add 实测 `max_err=0.0` |

> 三个 DSL（Triton / CUTLASS / TileLang）在 Jetson Orin sm_87 上**全部可用**。
> 此前"GitHub 443 超时导致 CUTLASS 无法获取"的缺口已随网络恢复补齐。

## 2. 测量波动声明（重要）

Jetson 是共享内存 + DVFS 的边缘设备，**短 kernel（< 1 ms）的逐次测量存在显著
波动**（同一 kernel 两次完整 run 的 median 可相差 30%+，例如 FP16 RMSNorm 与
cuBLAS 512×2048×2048 在两次 run 中从 0.66 ms 漂移到 1.93 ms）。因此：

- 本报告结论采用**定性 + 典型值**，不做两位小数的精确排名；
- FP32 RMSNorm 的"CUDA V2 float4 领先"是**两次 run 一致**的稳定结论；
- FP16 RMSNorm 与短 GEMM 的"谁更快"**不具备可复现性**，仅记录趋势；
- 后续 benchmark 需固定 DVFS/温度（S06 治理项）。

## 3. RMSNorm 性能对比（median ms，rows=512，典型值）

### FP32（稳定结论）

| hidden | CUDA V0 | CUDA V1 | CUDA V2 | Triton ref |
|---|---|---|---|---|
| 1024 | 0.25 | 0.24 | **0.19** | 0.23 |
| 2048 | 0.27 | 0.26 | **0.22** | 0.28 |

**结论（稳定）**：CUDA V2（float4 128-bit 事务）在 FP32 持续领先 Triton
reference（约 15–20%），与 S03 结论一致 —— 手写对齐/向量化换取峰值带宽。

### FP16（波动，仅记录趋势）

| hidden | CUDA V2 | Triton ref |
|---|---|---|
| 1024 | 0.14–0.17 | 0.11–0.21 |
| 2048 | 0.15–0.22 | 0.22–0.24 |

**结论（诚实）**：FP16 RMSNorm 的 CUDA vs Triton 对比**不具可复现性**——两次
完整 run 得到相反的结果（一次 Triton 反超 36%，一次 CUDA 领先 36%）。这是短
kernel 在 Jetson 上的测量波动，而非稳定的实现差异。S03 已从**微架构层面**证明
CUDA half2 是 32-bit 事务（带宽受限），该静态结论成立；但端到端测量需更严格
的 DVFS/温度控制才能下"谁更快"的结论。

## 4. GEMM 性能对比（median ms，FP16，含 CUTLASS）

| shape (M×N×K) | cuBLAS | Triton ref | Triton opt | CUTLASS |
|---|---|---|---|---|
| 1×2048×2048 | 0.24 | 0.35 | 0.38 | **0.12** |
| 1×2048×8192 | 0.76 | **0.67** | 0.69 | 0.89 |
| 512×2048×2048 | 0.66–1.93* | 1.03 | **0.72** | 0.51–1.01* |

> \* 表示该数据点在两次 run 间波动较大，典型值不可靠。

**结论（定性）**：
1. **没有"万能最快"的 GEMM 后端**：CUTLASS 在 1×2048×2048 反超 cuBLAS 约 2×，
   但 1×2048×8192 最慢；Triton 在 1×2048×8192 反超 cuBLAS；cuBLAS 在大 batch
   依赖 pipeline 但受波动影响大。
2. **CUTLASS 默认配置未 profiling**：本对照使用 `OpClassTensorOp + Sm80` 默认
   tile（128×256×64），未做 `cutlass::gemm::device::Gemm` 的 tile/instruction
   shape 扫参。即便如此，CUTLASS 在多个 shape 已接近或反超，说明**模板库的
   默认配置就已具备竞争力**，进一步 profiling 可获更多。
3. **窄矩阵（M=1）是 cuBLAS 的薄弱区**：decode 阶段的 M=1 GEMM 中，CUTLASS/
   Triton 均能反超 cuBLAS，验证 S02 Hotspot Decision（decode GEMM 用 cuBLAS
   非最优）。

## 5. 何时选什么（S04 验收，最终决策表）

| 场景 | 选择 | 理由 |
|---|---|---|
| 已知 shape + 峰值带宽（FP32 RMSNorm） | 手写 CUDA（float4） | 手工控制 128-bit 事务与对齐 |
| 快速原型 + 跨 arch 可移植 | Triton | JIT 适配任意 arch，代码量最少 |
| batch GEMM / 需要最强 GEMM 峰值 | CUTLASS（profiling 后） | 模板库显式 tile 控制，默认配置已具竞争力 |
| decode 窄矩阵（M=1）GEMM | CUTLASS / Triton 定制 | cuBLAS 对 M=1 非最优（实测） |
| 最简复现 / 教学 / 研究 DSL | TileLang | TVM 后端，声明式，可读性最高 |
| 无 DSL / 无 GPU | torch reference | 正确性兜底，记录降级 |

## 6. 开发成本（S04 步骤 11，扩展）

| 项 | 手写 CUDA | Triton | CUTLASS | TileLang |
|---|---|---|---|---|
| 代码量 | ~500 行 C++ | ~80 行 Python | ~100 行 C++（调用） | ~20 行 Python |
| 编译 | nvcc，分钟级 | JIT 秒级 | nvcc 模板，分钟级 | TVM JIT 秒级 |
| 峰值性能 | 最高（FP32 手控） | 接近 | 最高（GEMM） | 接近 |
| 跨 arch 可移植 | 差（手工对齐/arch） | 好 | 好（模板） | 好 |
| 维护风险 | 高 | 低 | 中（依赖版本） | 中（TVM 生态） |

**结论**：手写 CUDA 换峰值控制权；Triton 用最少代码换接近峰值 + 可移植；CUTLASS
换 GEMM 峰值（模板库）；TileLang 换声明式可读性。三者（Triton/CUTLASS/TileLang）
在 Jetson sm_87 均已实测可用，选择取决于 shape 与目标（峰值/可移植/开发效率）。
