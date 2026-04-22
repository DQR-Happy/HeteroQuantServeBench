# S09：Ascend C/CANN 异构后端

## 阶段目标

把 CUDA 已建立的 Operator、Model、Workload、Correctness 和 Benchmark Contract 迁移到 Ascend/CANN，在 Orange Pi 或可用 Ascend 设备上完成模型适配、算子开发、精度对齐和性能分析。

## 体现的知识与技能

Ascend C、CANN、TorchNPU/MindSpore/MindIE、Da Vinci 架构、Cube/Vector、Tiling、msprof、模型迁移和跨后端精度性能调优。

## 输入

- S01 稳定 Contract 与测试向量。
- S02 Qwen reference、S03 RMSNorm/第二热点语义、S05 QuantArtifact。
- Orange Pi/Ascend 设备、匹配的 firmware/driver/CANN/toolkit。

## 执行步骤

1. 生成硬件/固件/驱动/CANN/TorchNPU/MindIE manifest，校验官方兼容矩阵；禁止混装版本。
2. 实现 Ascend backend capability 和最小 device/model smoke；记录不支持功能。
3. 使用同一模型、tokenizer、workload 和 sampling 运行 reference baseline，先完成输出/精度对齐。
4. 建立 CUDA OperatorSpec 到 Ascend C 的映射；明确 GM/UB、Cube/Vector、Tiling 和多核划分。
5. 先实现 Add/Reduction 教学样例验证工具链，再实现 RMSNorm reference/optimized。
6. 为第二热点或融合算子设计 Ascend C Tiling；若硬件不适合，保留性能模型和不实现决策。
7. 接入 PyTorch/TorchNPU 或 MindIE 调用路径；不支持 shape 有 fallback。
8. 使用 msprof/Ascend Profiler 分析 AICore、Vector/Cube、带宽、流水、host/device 和 runtime gap。
9. 对 FP16/INT8/INT4 能力、layout/format conversion 和量化制品兼容做实验。
10. 运行 model-core/serving benchmark，保存与 CUDA 同 schema 的 raw/summary。
11. 对跨 backend 差异做归因：硬件、precision、kernel、runtime、memory capacity、软件成熟度。
12. 输出迁移 runbook、常见错误、版本和限制。

## 输出与产出

- `ops/ascend` operator library、test/bench、Tiling 文档。
- Ascend Backend adapter、capability 和模型运行记录。
- CUDA↔Ascend correctness/profile/performance report。
- 环境兼容 manifest 和故障排查 runbook。

## 测试标准

- 共用 test vectors、golden 和数值指标；format conversion 单独验证。
- Tiling 覆盖边界 shape、尾块、不同 core 数与 dtype。
- 模型 token/logit/质量对齐；不同精度必须明确阈值。
- msprof 数据与 run ID/commit 绑定。

## 验收通过标准

- 同一算子和同一模型 workload 在 CUDA/Ascend 上均可复现。
- 至少一个 Ascend C 算子有 reference→optimized→profile 闭环。
- 报告避免不同模型/精度直接比 tokens/s，并能解释架构差异。

