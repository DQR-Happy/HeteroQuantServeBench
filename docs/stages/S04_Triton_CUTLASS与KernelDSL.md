# S04：Triton、CUTLASS/CuTe 与 Kernel DSL

## 阶段目标

在相同 OperatorSpec、test vector 和 benchmark 下实现 CUDA、Triton 与高性能库版本，掌握“手写控制、DSL 开发效率、模板库峰值性能”之间的工程权衡，并扩展到 TileLang/HIP 的技术地图。

## 体现的知识与技能

Triton program model、autotune、CUTLASS/CuTe layout/tile/epilogue、GEMM/Grouped GEMM、Kernel DSL、编译链和跨架构可移植性。

## 输入

- S03 的 OperatorSpec、CUDA variants、shape 矩阵和 profiler 结论。
- 云端 NVIDIA GPU 环境；Jetson 对 Triton 支持需能力检测，不能默认。
- S02 真实模型热点 shape。

## 执行步骤

1. 选择 RMSNorm 和第二热点作为对照；保持 semantic、dtype、shape、tolerance 完全一致。
2. 实现 Triton reference 和 optimized variant；明确 program ID、block、mask、reduction 和 memory layout。
3. 建立 Triton autotune search space、key 和缓存策略；训练/测试 shape 分离，防止只对测试集调参。
4. 保存 Triton IR/编译元数据，分析 generated kernel 与 CUDA 的访存、寄存器和 launch 差异。
5. 对 GEMM/Grouped GEMM/融合 epilogue 建立 cuBLAS、CUTLASS/CuTe 和 Triton 三方对照。
6. 使用 CUTLASS profiler 或自建 harness 选择 tile、stage、warp、layout 和 epilogue；记录硬件限制。
7. 实现统一 dispatcher/capability：按 GPU arch、dtype、shape 和依赖可用性选择 CUDA/Triton/CUTLASS。
8. 在至少两种 NVIDIA 架构上运行相同矩阵，分析 autotune 参数是否迁移。
9. 复现一个 TileLang kernel，比较表达力、编译时间和性能；定位为实验 backend。
10. 建立 HIP/ROCm/OpenCL 的 L2 技术说明，可选在 AMD 环境移植一个简单 operator。
11. 报告开发成本：代码量、编译、调试、autotune 成本、性能和维护风险。

## 输出与产出

- `ops/triton` 实现、autotune 配置和缓存元数据。
- CUTLASS/CuTe GEMM/epilogue experiment。
- CUDA/Triton/CUTLASS/TileLang comparison report。
- 多架构性能与参数迁移表、dispatcher capability。

## 测试标准

- 共用 S03 correctness/property vectors；不同实现不得有独立弱化阈值。
- Autotune cache 与 config/hash/GPU arch 绑定，错误 cache 能失效。
- 未安装/不支持的 DSL 走明确 fallback，不影响 CPU/CUDA 核心包。
- 编译失败、OOM、动态 shape 和非对齐输入均有负向测试。

## 验收通过标准

- 至少一个算子能解释 CUDA 与 Triton 的性能差异；至少一个 GEMM/epilogue 完成库级调优。
- 能说明何时选手写 CUDA、Triton、CUTLASS/CuTe 或厂商库。
- 多架构数据证明 auto-tuning 不是全局常量。

