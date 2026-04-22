# S08：ServeFabric 与性能治理

## 阶段目标

在 Backend Contract 之上建立生产风格的 OpenAI-compatible Serving 控制面，覆盖调度、流式、取消、SLO、路由、缓存、故障和可观测性，同时保持服务层不依赖具体 Kernel。

## 体现的知识与技能

系统设计、HTTP/gRPC、异步并发、排队/调度、SLO、动态批处理、KV/Prefix Cache 路由、容错、可观测性和性能压测。

## 输入

- S07 Backend adapters、capability 和 Runtime 指标。
- S01 TraceEvent/BenchmarkResult/error contract。
- 标准 OpenAI request/streaming schema 与 workload distributions。

## 执行步骤

1. 定义 data plane 与 control plane；API/gateway 不直接导入 runtime SDK。
2. 实现 OpenAI-compatible completion/chat 最小协议、SSE streaming、request/trace ID。
3. 实现 admission、queue、priority/fairness、token budget、backpressure 和 overload reject。
4. 实现 timeout、client cancel、retry policy、idempotency boundary、circuit breaker 和 graceful shutdown。
5. 实现 capability-aware routing：model/dtype/context/backend health/SLO/cost；记录选择原因。
6. 实现 health/readiness、模型 load/unload、warmup 和版本切换；避免请求命中未就绪模型。
7. 设计 KV/Prefix Cache affinity、cache key/version、eviction 和命中 telemetry；可接 LMCache/Mooncake 类外部池化实验。
8. 建立 metrics：queue、TTFT、TPOT、E2E、tokens/s、request/s、goodput、P50/P95/P99、cache、OOM、cancel、retry、error。
9. 建立 traces：gateway→queue→backend→prefill/decode→stream；与 Kernel/collective event 关联。
10. 建立负载生成器：固定、Poisson/突发 arrival，ISL/OSL 分布，并发、连接中断和慢客户端。
11. 进行故障演练：backend crash/hang/OOM、模型加载失败、cache 不可用、超时和网络中断。
12. 比较调度/批处理/cache/routing 策略的吞吐—尾延迟—公平性。

## 输出与产出

- `hqsb/serving` gateway/scheduler/router/cache policy。
- OpenAI API、SDK/example、load generator。
- Metrics/trace dashboard、SLO policy 与 runbook。
- Serving benchmark 和故障演练报告。

## 测试标准

- 协议、stream framing、cancel、timeout、invalid request、backpressure。
- 并发/竞态、请求泄漏、graceful shutdown、模型版本和 cache invalidation。
- Backend failure/fallback 不造成错误模型或重复 token。
- 压测报告 tail latency、goodput 和失败率，不只平均 tokens/s。

## 验收通过标准

- 多 Backend 可同时注册并基于 capability/SLO 路由。
- 在过载和单 backend 故障下服务行为符合策略且可观测。
- 至少一项 scheduler/cache 优化有可解释的 service-level 收益。

