# S12：跨硬件评估与统一 Benchmark

## 阶段目标

将已有 Kernel、量化、Runtime、Serving、Ascend 和分布式结果统一成可审计的硬件评估体系，回答不同 GPU/NPU/边缘/云平台的收益、成本、能效、软件成熟度和迁移风险。

## 体现的知识与技能

Benchmark 方法、统计、性能建模、Roofline、HW/SW Co-design、芯片评估、TCO/能效、实验设计和数据分析。

## 输入

- S02–S11 的版本化 results/profiles/artifacts。
- Jetson、云端 NVIDIA、Ascend；可选 AMD/CPU。
- 统一 ModelArtifact、WorkloadSpec 和 BenchmarkResult。

## 执行步骤

1. 定义评估问题和业务画像：交互式、吞吐型、长上下文、decode-heavy、MoE、edge power-limited。
2. 冻结模型/精度/quality/workload；不能比较的组合显式标注，而非补零。
3. 建立四层矩阵：operator、model-core、runtime/serving、distributed。
4. 统一正确性和质量门槛；不同 precision 先过质量 gate 再比较性能。
5. 统一指标：latency distribution、throughput/goodput、memory、power/energy、utilization、stability、cost。
6. 运行环境校验、warmup、重复、cooldown 和 artifact 上传；自动检测 thermal/swap/OOM/clock 异常。
7. 建立 performance model：峰值 FLOPS/带宽、算术强度、容量、互联和 Amdahl。
8. 比较实测与模型误差，解释软件、layout、kernel、runtime 或测量原因。
9. 计算能效和成本：J/token、tok/s/W、cost/token、容量密度和租赁/开发成本。
10. 评价软件成熟度：安装、编译、调试、Profiler、框架兼容、稳定性和可维护性。
11. 建立收益预测模板：目标业务分布→硬件/precision/runtime 建议、风险与置信度。
12. 生成 dashboard/静态图、executive summary、methodology、raw artifact index 和 limitations。

## 输出与产出

- 统一 benchmark CLI、矩阵和 schema validator。
- Operator/model/serving/distributed cross-platform results。
- Roofline、Pareto、tail latency、energy/cost 图。
- Hardware evaluation report 和部署建议。

## 测试标准

- 配置/结果完整性、单位、统计和不可比条件自动检查。
- 重复性、异常值、置信区间和 regression。
- Dashboard 数字可回溯 raw sample，不允许手工复制失联。
- 不同模型/精度/环境的比较必须有显式 normalization/quality gate。

## 验收通过标准

- 至少三类硬件或两类硬件+两种架构形成有效对比。
- 每个结论同时包含性能、精度、内存/能耗、软件成本和限制。
- 能从业务 workload 反推平台选择，而非只排名峰值。

