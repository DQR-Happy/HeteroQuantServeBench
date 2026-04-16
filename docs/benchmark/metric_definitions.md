# Metric Definitions

This document defines every metric produced by the HQSB model-core
benchmark. Understanding these definitions is critical for correct
interpretation of benchmark results.

---

## Latency Metrics

### prefill_forward_ms

**Definition**: Wall-clock time for the initial model forward pass
that processes the entire input sequence.

**Measurement**: `time.perf_counter()` bracketing the `model()`
call with the full input, synchronized with `torch.cuda.synchronize()`.

**Excludes**: Tokenizer encoding, data transfer to GPU, first-token
selection.

**Unit**: milliseconds

---

### first_token_selection_ms

**Definition**: Time to select the first output token via argmax
on the prefill output logits.

**Measurement**: `time.perf_counter()` from prefill output to
first `next_token` assignment.

**Unit**: milliseconds

---

### model_core_ttft_ms

**Definition**: Time-To-First-Token at the model-core level.

**Formula**: `prefill_forward_ms + first_token_selection_ms`

**Excludes**: HTTP round-trip, request queueing, network transport,
client-side processing. This is the **minimum achievable TTFT**
for the given hardware and model.

**Unit**: milliseconds

---

### ITL (Inter-Token Latency)

**Definition**: Per-step latency for each token generated during
the decode phase (excluding the first token).

**Measurement**: `time.perf_counter()` per decode iteration,
synchronized with `torch.cuda.synchronize()`.

**Statistics reported**:
- `count`: Number of decode steps
- `mean_ms`: Arithmetic mean
- `median_ms`: 50th percentile
- `p50_ms`: 50th percentile (same as median, via linear interpolation)
- `p95_ms`: 95th percentile
- `p99_ms`: 99th percentile
- `stddev_ms`: Population standard deviation
- `min_ms`, `max_ms`: Range

**Unit**: milliseconds

---

### decode_total_ms

**Definition**: Sum of all ITL values for a single benchmark pass.

**Formula**: `sum(itl_ms)`

**Unit**: milliseconds

---

### model_core_e2e_ms

**Definition**: Total end-to-end model-core latency.

**Formula**: `model_core_ttft_ms + decode_total_ms`

**Excludes**: Model loading time, tokenizer overhead, data preparation.

**Unit**: milliseconds

---

## Throughput Metrics

### prefill_tokens_per_s

**Definition**: Input token processing rate during prefill.

**Formula**: `input_tokens / (prefill_forward_ms / 1000.0)`

**Unit**: tokens/second

---

### decode_tokens_per_s

**Definition**: Output token generation rate during decode.

**Formula**: `(output_tokens - 1) / (decode_total_ms / 1000.0)`

**Note**: The first token is excluded because it is generated during
the prefill phase, not the decode phase.

**Unit**: tokens/second

---

### model_core_output_tokens_per_s

**Definition**: Overall output throughput including both prefill
and decode phases.

**Formula**: `output_tokens / (model_core_e2e_ms / 1000.0)`

**Unit**: tokens/second

---

## Memory Metrics

### peak_cuda_allocated_mb

**Definition**: Peak memory allocated by the PyTorch CUDA allocator
during the benchmark pass.

**Measurement**: `torch.cuda.max_memory_allocated()` called after
`torch.cuda.reset_peak_memory_stats()` at the start of each pass.

**Unit**: MiB (1024² bytes)

---

### peak_cuda_reserved_mb

**Definition**: Peak memory reserved (cached) by the PyTorch CUDA
allocator. This is typically larger than allocated memory because
the allocator caches freed blocks.

**Measurement**: `torch.cuda.max_memory_reserved()`.

**Unit**: MiB

---

### process_rss_mb

**Definition**: Resident Set Size of the Python process, as reported
by the operating system.

**Measurement**: `psutil.Process().memory_info().rss`

**Unit**: MiB

---

### system_memory_used_mb

**Definition**: System-wide RAM usage, as reported by the OS.

**Measurement**: `psutil.virtual_memory().used`

**Important**: On Jetson platforms, CPU and GPU share system memory
(DDR). This metric reflects total memory pressure.

**Unit**: MiB

---

## Jetson Resource Metrics

### avg_power_w / peak_power_w

**Definition**: Average and peak VDD_IN power consumption measured
by tegrastats during the benchmark.

**Measurement**: Tegrastats `VDD_IN` rail, sampled at 100 ms intervals.

**Unit**: Watts

---

### energy_j

**Definition**: Total energy consumed during the benchmark, computed
via trapezoidal integration of power over time.

**Formula**: `sum((P[i] + P[i+1]) / 2 * dt)`

**Unit**: Joules

---

### energy_j_per_output_token

**Definition**: Energy efficiency metric.

**Formula**: `energy_j / output_tokens`

**Unit**: Joules/token

---

### gpu_util_pct

**Definition**: GPU utilization percentage as reported by tegrastats
(`GR3D_FREQ` field).

**Statistics**: Average and peak across the monitoring window.

**Unit**: percent (0-100)

---

### gpu_temp_c / cpu_temp_c

**Definition**: GPU and CPU die temperatures.

**Measurement**: Tegrastats thermal sensors.

**Unit**: Celsius
