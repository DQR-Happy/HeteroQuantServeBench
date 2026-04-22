# S11：AI 编译器与自动优化

## 阶段目标

建立从 PyTorch 图捕获、Pattern Rewrite、IR Lowering 到 CUDA/Triton/Ascend Backend 的最小 AI Compiler 闭环，并探索成本模型、自动调优和 AI-assisted kernel workflow。

## 体现的知识与技能

编译原理、IR/SSA、dataflow、FX/Dynamo/Inductor、MLIR/LLVM、TVM Relax/TIR、XLA/HLO、ONNX/TensorRT、fusion 和 autotuning。

## 输入

- S06 Custom Op/FX pattern、S03/S04/S09 kernels。
- S02/S07 真实模型图、dynamic shape 和热点。
- S01 Operator/Backend/Result Contract。

## 执行步骤

1. 写编译链技术地图：PyTorch eager→Dynamo/FX→AOTAutograd→Inductor→Triton/C++；MLIR/LLVM、TVM、XLA、ONNX/TensorRT 对照。
2. 定义 compiler experiment boundary：输入 graph/module、shape constraints、target capability、output artifact 和 fallback。
3. 捕获 Qwen 子图，记录 graph break、guards、dynamic shape 和 unsupported op。
4. 实现一个 FX pattern rewrite（如 residual+RMSNorm、RMSNorm+Quantize、RoPE fusion）。
5. 通过 Custom Op 或生成 Kernel lower 到 S03/S04；保存前后 graph/IR。
6. 用 Inductor config/trace 分析 fusion、layout、memory planning 和 generated kernel；做一项源码/配置修改。
7. 在 TVM Relax/TIR 或 MLIR 中复现一个算子/子图 lowering 和 schedule，输出目标代码与 benchmark。
8. 建立 target capability 和 cost model：shape/dtype/layout/arch/compile/runtime；错误预测必须 fallback。
9. 实现 autotune 数据集、search space、cache 和 holdout shape 评估。
10. 探索 AI-assisted kernel generation/optimization：Agent 只能提出候选，必须经过 schema、correctness、sanitizer、benchmark 和 review gate。
11. 研究 LLVM/PTX/SASS 层输出，解释 compiler decision 与硬件行为。
12. 输出 compiler design、IR snapshots、correctness、compile cost 和 runtime speedup。

## 输出与产出

- `hqsb/compiler` target、pass、cost model、artifact cache。
- FX/Inductor rewrite 和一条 TVM/MLIR 实验链。
- Autotune/AI-assisted kernel gate。
- 编译时间、缓存、正确性、性能与 fallback 报告。

## 测试标准

- Graph rewrite semantic equivalence、dynamic/unsupported shape。
- IR verifier、pass idempotence、cache key 和版本失效。
- Generated code 通过 operator/model correctness、sanitizer 和性能回归。
- Compile-time 与 run-time 分开；首次编译不混入 steady-state。

## 验收通过标准

- 至少一个真实 Qwen pattern 从捕获到自定义 kernel 完整 lower。
- 能解释 FX/Inductor、MLIR/LLVM、TVM/XLA 的角色和取舍。
- 自动生成候选不能绕过 correctness/performance gate。

