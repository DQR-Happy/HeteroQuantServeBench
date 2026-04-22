# S02：模型基线与全栈 Profiling

## 阶段目标

建立可信的 Qwen3-1.7B FP16 Reference Runtime，分别刻画 Prefill 与 Decode 的算子、Runtime、内存和功耗瓶颈，并用数据决定 S03 的重点优化对象。

## 体现的知识与技能

Transformer/LLM、PyTorch、KV Cache、模型加载、推理指标、PyTorch Profiler、Nsight Systems/Compute、Amdahl/Roofline 和科学实验。

## 输入

- S01 Contract、Backend interface 和 BenchmarkResult。
- Qwen3-1.7B 模型/tokenizer manifest 与现有四份 golden。
- Jetson 固定环境和六组 ISL/OSL workload。
- tegrastats monitor/parser。

## 执行步骤

1. 实现 PyTorch reference backend：local-only load、eval/inference mode、dtype/device/attention、cache 和显存统计。
2. 明确 reference semantic：greedy decoding、固定 token IDs、batch=1、无 HTTP/queue/tokenizer 计时。
3. 修正 model-core 计时边界，保存每个 decode step 原始 ITL；区分 prefill forward、first-token selection、decode、E2E。
4. 增加 KV Cache shape/bytes、模型权重、allocator allocated/reserved、RSS、swap 和 load time。
5. 让 YAML 成为 workload 唯一事实源；覆盖 tiny、short、balanced、long-prefill、decode-heavy、long-balanced。
6. 建立 Jetson 实验协议：nvpmodel、clock、温度、cooldown、warmup、重复、后台任务和采样窗口。
7. 运行 correctness/determinism：token hash、first-token logits、golden schema 和多次重复。
8. 运行完整 baseline，保存 raw/normalized/summary、环境和 Commit。
9. 使用 PyTorch Profiler 分别采集代表性 prefill/decode；导出 operator、shape、调用次数、CPU/CUDA time、memory。
10. 使用 Nsight Systems 分析 launch gap、同步、CPU/GPU overlap、kernel timeline、内存拷贝和 GPU idle。
11. 对 Top 候选 Kernel 使用 Nsight Compute；限制采集区域，记录 memory throughput、SM utilization、occupancy、stall、register/shared。
12. 构建 Roofline/Amdahl 分析，将热点分类为 GEMM/Attention、可定制 elementwise/reduction、Runtime launch/sync、Memory/KV/swap。
13. 评估候选对象的总时间占比、理论上限、实现难度、框架可回接性和招聘价值。
14. 输出 Hotspot Decision：RMSNorm 固定做教学闭环，另选一个真实热点；若 GEMM 主导，选择 CUTLASS/低比特/epilogue 而非从零写 GEMM。

## 输出与产出

- PyTorch reference backend 和版本化 Qwen artifact。
- 六 workload 的 raw JSON、normalized CSV 和 summary。
- `baseline_report.md`、`pytorch_profile_report.md`、`nsys_report.md`、`ncu_report.md`。
- Prefill/Decode hotspot ranking、Amdahl 上限和 S03 实验假设。

## 测试标准

- Workload token 数严格等于配置；非法 ISL/OSL 拒绝。
- Golden/determinism 在冻结环境中通过；差异报告定位 first mismatch。
- 计时测试验证同步位置，禁止把异步 launch 当 kernel 完成。
- 短 case 的 tegrastats 样本数达标；时间戳单调且能量积分测试通过。
- 相同环境重复运行波动在预设阈值内；超阈值必须标记不稳定而非取最好值。

## 验收通过标准

- 任意正式结果可追溯到 raw sample、配置、模型 hash、环境和 Commit。
- Prefill 与 Decode 分别有 Top hotspot 和硬件证据。
- S03 的算子选择来自 Profile，包含明确 correctness/performance/stop criteria。
- 报告明确排除 tokenizer、queue、network 等未测开销。

## 明确不做

不在本阶段优化 Kernel，不把 model-core TTFT 冒充线上 TTFT。

