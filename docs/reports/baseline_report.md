# S02 Baseline 报告（Baseline Report）

> 阶段：S02（模型基线与全栈 Profiling）
> 日期：2026-08-16
> 目标硬件：NVIDIA Jetson Orin Nano Super 8GB（sm_87, Ampere）
> 模型：Qwen/Qwen3-1.7B FP16（ModelScope, 1720574976 params ≈ 3.20 GB FP16）

## 1. Reference Runtime 语义

| 项 | 值 |
|---|---|
| 后端 | `PyTorchBackend`（`hqsb/backends/pytorch.py`，C4 Backend 契约） |
| 解码策略 | greedy（argmax） |
| 输入 token | 固定 seed 文本 + 截断到精确 ISL（`make_fixed_token_input`） |
| batch | 1 |
| 计时口径 | model-core 仅（**排除** tokenizer / HTTP / queue / network） |
| attention | eager（`attn_implementation="eager"`） |
| dtype | float16 |

> TTFT 为 model-core TTFT（prefill forward + first-token argmax），**不冒充线上 TTFT**。

## 2. 完整六 workload baseline（S00 历史，model-core 口径）

来源：`reports/dev/llm/20260812_094129/summary.csv`（S00 已在冻结环境运行，
deterministic=true，decode 约 9.1–10.2 tok/s）。

| case | ISL | OSL | TTFT(ms) | E2E(ms) | Decode(t/s) |
|---|---|---|---|---|---|
| tiny | 32 | 16 | ~121 | ~1762 | ~9.1 |
| short | 128 | 32 | ~141 | ~3430 | ~9.4 |
| balanced | 512 | 128 | ~372 | ~12883 | ~10.2 |
| long_prefill | 2048 | 32 | ~2531 | ~5646 | ~9.9 |
| decode_heavy | 128 | 256 | ~121 | ~25499 | ~10.0 |
| long_balanced | 2048 | 128 | ~2546 | ~16561 | ~9.1 |

环境绑定见各 JSON 的 `hardware`/`software` 字段（Orin, sm_87, PyTorch
2.5.0a0+nv24.08, CUDA 12.6, Transformers 5.8.0, ModelScope 1.29.0）。

## 3. 本次 S02 端到端验证（新 contract 引擎）

`PyTorchBackend` + `BenchmarkEngine`（contract-native）在本次会话重跑 tiny
workload（ISL=32, OSL=8, repetitions=1），产出版本化 `BenchmarkResult`
（`reports/dev/llm/s02_smoke_tiny.json`）：

| 指标 | 值 |
|---|---|
| correctness | True（determinism） |
| model_core_ttft_ms | 112.74 |
| decode_tokens_per_s | 9.28 |
| raw ITL（前 7 步，ms） | 109.5, 108.5, 108.3, 107.5, 104.8, 105.7, 104.9 |
| 模型权重字节 | 3,441,149,952（3.44 GB） |
| CUDA allocated | 3289.9 MB |
| 进程 RSS | 876 MB |

> 说明：本次 decode 9.28 tok/s 与 S00 tiny（~9.1）一致，验证新 contract 引擎
> 的计时与 S00 历史基线可对齐，且新引擎额外保存了 raw ITL / KV cache / 权重 /
> RSS 等 S02 要求的内存与逐 token 信息。

## 4. KV Cache 与内存画像

Qwen3-1.7B 结构（从 model config 读取）：

| 参数 | 值 |
|---|---|
| num_layers | 28 |
| num_key_value_heads | 8（GQA） |
| head_dim | 128 |
| element_bytes | 2（FP16） |

KV cache 计算（`hqsb/benchmark/memory.py`）：

```
per_token_bytes = 2 * layers * kv_heads * head_dim * element_bytes
                = 2 * 28 * 8 * 128 * 2 = 114,688 B/token
```

| context | KV cache |
|---|---|
| 40 tokens（ISL 32 + OSL 8） | 4,587,520 B（4.59 MB） |
| 160 tokens（ISL 32 + OSL 128） | 18,350,080 B（18.35 MB） |
| 2176 tokens（ISL 2048 + OSL 128） | 249,561,088 B（249.56 MB） |

结论：KV cache 在长序列下可占 ~250 MB，是可观测但非主要的内存项（模型权重
3.44 GB 占绝对大头）。

## 5. 结论

1. 模型加载 3.20 GB FP16，权重是内存绝对大头（3.44 GB），KV cache 次之。
2. decode 吞吐 ~9–10 tok/s，受限于单 batch 的 GEMM 密集计算（见
   `pytorch_profile_report.md`）。
3. 计时边界已修正：每个 decode step 保存原始 ITL，prefill / first-token /
   decode / E2E 严格分离，全部以 `torch.cuda.synchronize()` 界定。
