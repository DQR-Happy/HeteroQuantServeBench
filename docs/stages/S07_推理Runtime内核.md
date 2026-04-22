# S07：推理 Runtime 内核

## 阶段目标

以源码级方式理解和评估 PyTorch Reference、llama.cpp/TensorRT-LLM 边缘路径与 vLLM/SGLang 云端路径，重点研究 KV Cache、PagedAttention、连续批处理、CUDA Graph 和调度。

## 体现的知识与技能

LLM Runtime、KV Cache、PagedAttention、scheduler、continuous batching、chunked prefill、CUDA Graph、speculative decoding、模型并行和源码调试。

## 输入

- S01 Backend Contract/Capability。
- S02/S05 模型与量化 artifact、统一 workload/result。
- S06 可选 custom/fused operators。
- Jetson 与云端 NVIDIA GPU 环境。

## 执行步骤

1. 实现 PyTorch、edge、cloud backend adapter；明确 load/warmup/generate/stream/cancel/metrics/close。
2. 选择一个云端主 runtime 深入源码（vLLM 或 SGLang）；另一个作为对照，不同时做同等深度修改。
3. 追踪 request→scheduler→batch→model runner→attention/KV→output 的完整调用链并绘图。
4. 分析 KV Cache block/page、allocator、fragmentation、eviction、prefix cache 和容量模型。
5. 研究 continuous batching、chunked prefill、max tokens、queue policy 对吞吐和尾延迟的影响。
6. 评估 eager/CUDA Graph、kernel fusion 和 attention backend；记录 capture/replay 限制。
7. 比较不同 ISL/OSL、batch/concurrency、precision/quant 的 TTFT/TPOT/TPS/memory。
8. 修改一个有明确假设的 Runtime 策略：KV block、batch/token budget、prefix cache 或 CUDA Graph，做源码级 A/B。
9. 评估 speculative decoding/MTP 的最小实验：draft cost、acceptance、latency 和 workload dependence。
10. 研究 PD/EPD/AF 分离和 disaggregated serving 的架构，先单机模拟事件/队列，不伪造集群收益。
11. 将 S06 custom operator 接入可行 runtime；不能接入则解释 ABI/graph/fusion 边界。
12. 输出 Runtime architecture、hot path、memory model、改动和 benchmark。

## 输出与产出

- 标准 Backend adapters 与 capability matrix。
- 主 runtime 源码调用链、KV/scheduler 设计文档。
- 一项源码级策略优化及 A/B 报告。
- edge/cloud/reference 的 workload 对比和 limitations。

## 测试标准

- 同 sampling/tokenizer 下输出合法且长度一致。
- Streaming、cancel、timeout、OOM、超长 context、并发和重复 load/close。
- KV Cache 容量、block 生命周期、prefix hit/miss 和 fragmentation 测试。
- Runtime 修改不破坏 correctness/稳定性；性能保存 raw/tail distribution。

## 验收通过标准

- 能完整解释主 runtime 的请求调度、KV 和模型执行链。
- 至少一个 Runtime 参数/源码改动产生可解释收益或被数据否定。
- Benchmark Contract 可以无条件替换 reference/edge/cloud backend。

