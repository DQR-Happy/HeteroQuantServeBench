# HQSB LLM Benchmark Methodology

## Overview

This document defines the methodology for the HeteroQuantServeBench
(HQSB) LLM inference benchmark suite. The benchmark measures
**model-core** performance — raw computational throughput of the
model forward pass without serving infrastructure overhead.

## Reference Model

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-1.7B |
| Source | ModelScope (`Qwen/Qwen3-1.7B`) |
| Architecture | Qwen3ForCausalLM (dense transformer) |
| Precision | FP16 |
| Attention | Eager (native PyTorch) |
| Loading | `local_files_only=True` |

## Reference Backend

ModelScope local model loading with native PyTorch execution.
No custom CUDA kernels, no quantization, no optimized attention
implementations.

## Reference Precision

All benchmarks use FP16 (IEEE 754 half-precision) for both
model weights and activations. This is the standard precision
for Jetson Orin GPU inference.

## Attention Implementation

Eager attention — the default PyTorch SDPA (scaled dot-product
attention) without FlashAttention or other fused kernels. This
provides the most comparable baseline across backends.

## Batch Size

All workloads use batch size = 1. This reflects the typical
edge inference scenario on Jetson-class devices.

## Performance Workloads

The benchmark uses **fixed input token length (ISL)** and **fixed
output token length (OSL)** workloads. Token sequences are
synthetically generated from a deterministic seed text to
ensure reproducibility.

### Workload Cases

| Name | ISL | OSL | Characteristic |
|------|-----|-----|----------------|
| tiny | 32 | 16 | Minimal overhead, quick validation |
| short | 128 | 32 | Light prefill, short decode |
| balanced | 512 | 128 | Balanced prefill/decode |
| long_prefill | 2048 | 32 | Prefill-bound (long context) |
| decode_heavy | 128 | 256 | Decode-bound (long generation) |
| long_balanced | 2048 | 128 | Heavy prefill + medium decode |

## Performance Layers

The HQSB benchmark is organized into layers:

1. **Model Core** — Raw model forward pass (current stage)
2. **Operator** — Individual kernel micro-benchmarks
3. **Online Serving** — HTTP API with request queueing
4. **Numerical Regression** — Model-level correctness comparison
5. **Quality Evaluation** — Downstream task metrics

## Current Stage (Phase 2)

The current benchmark measures model-core inference **without**:
- HTTP server overhead
- Request queueing and batching
- Network transport
- Serving scheduler
- Tokenizer streaming

## Metrics

### Latency
- **prefill_forward_ms**: Time for the initial full-sequence forward pass
- **model_core_ttft_ms**: Prefill + first-token selection (TTFT)
- **decode_total_ms**: Sum of all decode step latencies
- **ITL (Inter-Token Latency)**: Per-step decode latency distribution
- **model_core_e2e_ms**: Total end-to-end latency

### Throughput
- **prefill_tokens_per_s**: Input tokens processed per second during prefill
- **decode_tokens_per_s**: Output tokens generated per second during decode
- **model_core_output_tokens_per_s**: Overall output throughput

### Memory
- **peak_cuda_allocated_mb**: Peak allocated CUDA memory
- **peak_cuda_reserved_mb**: Peak reserved CUDA memory
- **process_rss_mb**: Process RSS memory
- **system_memory_used_mb**: System-wide RAM usage

### Jetson-Specific
- **GPU utilization** (%)
- **GPU temperature** (°C)
- **Power consumption** (W)
- **Energy per token** (J/token)
- **System RAM pressure**

## Reproducibility

All benchmarks record:
- ModelScope model ID and local SHA256 manifest
- PyTorch, CUDA, Transformers, ModelScope versions
- Jetson power mode (`nvpmodel`) and clock status
- GPU model and compute capability
- Workload parameters (ISL, OSL, batch size)

## Next Phase

Phase 3 will introduce:
- CUDA custom RMSNorm kernel
- Golden numerical regression baseline
- Operator-level micro-benchmarks
- End-to-end model replacement validation
