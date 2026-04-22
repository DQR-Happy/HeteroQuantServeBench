# S03：CUDA 算子性能工程

## 阶段目标

建立工业级 CUDA Operator Lab：以 RMSNorm 为旗舰教学算子，并对 S02 选出的真实热点完成多版本实现、硬件级 Profiling、Dispatcher、Framework 回接前准备和完整退化分析。

## 体现的知识与技能

CUDA C++、GPU 架构、并行算法、内存层次、数值稳定性、Nsight、CMake、C++ 测试和性能建模。

## 输入

- S01 OperatorSpec、registry 和测试框架。
- S02 shape 分布、热点、Amdahl 上限与 golden/test vectors。
- 现有 RMSNorm V0 和 CUDA device query。

## 执行步骤

1. 重构 `ops/cuda/rmsnorm` 为 include/src/tests/bench，保留旧 V0 行为作为可追踪 baseline。
2. 定义 stream-aware、无隐藏分配的 Operator API；CPU/PyTorch 高精度 reference 独立于 benchmark。
3. 实现 V0 shared reduction、V1 warp shuffle；验证非 2 次幂 hidden、不同 rows、对齐和极值。
4. 根据 Profile 决定 V2：vectorized load、FP16/half2、persistent 或 Residual+RMSNorm fusion；每个版本必须有假设。
5. 建立 dtype/rows/hidden/block/alignment/variant 的 benchmark matrix；同时测 device event 与 host submit+sync。
6. 计算每个 case 的 bytes/FLOPs/有效带宽、理论上限和 occupancy 约束。
7. 对 V0/V1/后续版本使用 nsys/ncu，记录同步、bank conflict、DRAM/L2、warp stall、register 和 occupancy。
8. 实现 dispatcher：device/dtype/shape/alignment/capability 选择；未知或不利 case 回退 reference。
9. 为 S02 选出的第二算子重复 reference→variants→profile→dispatcher；深度可低于 RMSNorm，但必须覆盖真实模型 shape。
10. 若热点是 GEMM/Grouped GEMM，不自研通用 GEMM；建立 cuBLAS/CUTLASS 对照并优化 layout/epilogue/shape selection。
11. 为错误 CUDA API、非法 shape、workspace、stream 和异步 lifetime 建立测试。
12. 输出每版本 Optimization Log，记录失败假设、退化 shape 和保留/回滚决策。

## 输出与产出

- 可复用 CUDA operator library、headers、dispatcher 和 fallback。
- RMSNorm ≥3 个有意义版本；第二热点 ≥2 个实现或库策略。
- correctness/property/performance tests 与统一 test vectors。
- nsys/ncu 原始 artifact 索引、Roofline/shape 性能图和 Optimization Logs。
- 供 S06 使用的稳定 C/C++ Operator API。

## 测试标准

- 与 FP64/FP32 reference 比较 max/mean/RMSE/cosine/relative L2；按 dtype/shape 定阈值。
- 覆盖零/极小/极大值、非对齐、非 2 次幂、最小/最大 hidden、多个 stream。
- CUDA memcheck/sanitizer 无越界、竞态和泄漏。
- 每 case warmup/repetition/raw sample 完整；报告全部退化，不只最佳值。
- Dispatcher 对所有支持组合选择正确实现，不支持组合明确 fallback。

## 验收通过标准

- 能用 profiler 指标解释至少一次显著加速和一次无收益/退化。
- 旗舰算子在代表 shape 有稳定优化，且无 correctness regression。
- API 不依赖 Python/Serving/配置文件，可由 S04/S06/S09 复用。
- 对模型端到端预期收益给出 Amdahl 上限，不夸大 micro speedup。

## 明确不做

不为了版本数量实现无假设变体；不从零替代 cuBLAS/CUTLASS 的成熟通用 GEMM。

