"""Model-core benchmark engine for LLM inference.

Measures the raw computational performance of a model without
serving overhead (no HTTP, no queueing, no network transport).

The benchmark decomposes inference into three phases:
    1. **Prefill**: Initial full-sequence forward pass.
    2. **First Token**: Argmax selection of the first output token.
    3. **Decode**: Autoregressive token-by-token generation.

Metrics include latency breakdowns (TTFT, ITL, E2E), throughput
(tokens/s for each phase), and GPU memory usage (allocated + reserved).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import torch

from hqsb.benchmark.memory import (
    compute_kv_cache_info,
    model_kv_cache_config,
    model_weight_bytes,
    process_rss_bytes,
    process_swap_bytes,
)
from hqsb.benchmark.metrics import latency_summary

logger = logging.getLogger(__name__)


def _model_element_size(model: Any) -> int:
    """Return the element size (bytes) of the model's weights.

    Falls back to 2 (FP16) when the dtype cannot be determined.
    """
    try:
        parameter = next(model.parameters())
        return parameter.element_size()
    except (StopIteration, AttributeError):
        return 2


def _kv_cache_accounting(
    model: Any,
    context_length: int,
    *,
    dtype_bytes: int,
) -> Dict[str, Any]:
    """Compute KV-cache shape and byte accounting for a given context length.

    Returns an empty dict if the model config does not expose the required
    structural parameters.
    """
    config = model_kv_cache_config(model)
    required = {"num_layers", "num_kv_heads", "head_dim"}
    if not required.issubset(config):
        return {}

    try:
        info = compute_kv_cache_info(
            num_layers=config["num_layers"],
            num_kv_heads=config["num_kv_heads"],
            head_dim=config["head_dim"],
            context_length=context_length,
            dtype_bytes=dtype_bytes,
        )
    except ValueError:
        return {}

    return {
        "num_layers": info.num_layers,
        "num_kv_heads": info.num_kv_heads,
        "head_dim": info.head_dim,
        "context_length": info.context_length,
        "element_bytes": info.element_bytes,
        "per_token_bytes": info.per_token_bytes(),
        "total_bytes": info.total_bytes(),
    }


@torch.inference_mode()
def benchmark_model_core(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    output_tokens: int,
) -> Dict[str, Any]:
    """Run a single model-core benchmark pass.

    Measures prefill latency, TTFT (Time-To-First-Token), per-token
    decode latency (ITL), end-to-end latency, throughput for each
    phase, and peak CUDA memory usage.

    Args:
        model: A HuggingFace-compatible causal LM model instance.
            Must be in eval mode and on the target device.
        inputs: Dictionary with ``input_ids`` and ``attention_mask``
            tensors, each of shape ``(1, input_length)``.
        output_tokens: Number of tokens to generate. Must be >= 1.

    Returns:
        Dictionary with keys:
            - ``input_tokens``: Input sequence length.
            - ``output_tokens``: Number of generated tokens.
            - ``prefill_forward_ms``: Prefill forward pass time (ms).
            - ``first_token_selection_ms``: First token argmax time (ms).
            - ``model_core_ttft_ms``: Total TTFT (ms).
            - ``decode_total_ms``: Sum of all decode step latencies (ms).
            - ``model_core_e2e_ms``: End-to-end latency (ms).
            - ``prefill_tokens_per_s``: Prefill throughput.
            - ``decode_tokens_per_s``: Decode throughput.
            - ``model_core_output_tokens_per_s``: Overall output throughput.
            - ``itl``: ITL statistics dict (from ``latency_summary``).
            - ``raw_itl_ms``: Raw per-decode-step ITL list (ms).
            - ``peak_cuda_allocated_mb``: Peak allocated CUDA memory (MiB).
            - ``peak_cuda_reserved_mb``: Peak reserved CUDA memory (MiB).
            - ``generated_token_ids``: List of generated token IDs.
            - ``kv_cache``: KV-cache accounting dict (shape/bytes).
            - ``model_weight_bytes``: Total model weight bytes.
            - ``process_rss_bytes``: Process RSS in bytes.
            - ``process_swap_bytes``: Process swap in bytes.

    Raises:
        ValueError: If ``output_tokens`` < 1.
    """
    if output_tokens < 1:
        raise ValueError(f"output_tokens must be >= 1, got {output_tokens}")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    input_token_count: int = input_ids.shape[1]
    device = input_ids.device

    # Reset CUDA memory statistics for accurate peak measurement
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ── Phase 1: Prefill ──────────────────────────────────────────

    if device.type == "cuda":
        torch.cuda.synchronize()

    prefill_start = time.perf_counter()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    prefill_forward_ms = (time.perf_counter() - prefill_start) * 1000.0

    # ── Phase 2: First Token Selection ────────────────────────────

    first_token_start = time.perf_counter()

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values

    if device.type == "cuda":
        torch.cuda.synchronize()

    first_token_selection_ms = (
        time.perf_counter() - first_token_start
    ) * 1000.0

    model_core_ttft_ms = prefill_forward_ms + first_token_selection_ms

    generated_tokens: List[int] = [int(next_token.item())]

    # ── Phase 3: Decode ───────────────────────────────────────────

    itl_ms: List[float] = []
    current_length = input_token_count

    for _step in range(1, output_tokens):
        current_length += 1

        # Build causal attention mask for the growing sequence
        decode_attention_mask = torch.ones(
            (1, current_length),
            dtype=torch.long,
            device=device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        decode_start = time.perf_counter()

        outputs = model(
            input_ids=next_token,
            attention_mask=decode_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - decode_start) * 1000.0
        itl_ms.append(elapsed_ms)
        generated_tokens.append(int(next_token.item()))

    # ── Aggregate Metrics ─────────────────────────────────────────

    decode_total_ms = sum(itl_ms)
    model_core_e2e_ms = model_core_ttft_ms + decode_total_ms

    decode_tokens_count = max(output_tokens - 1, 0)

    # Throughput calculations
    prefill_tps = (
        input_token_count / (prefill_forward_ms / 1000.0)
        if prefill_forward_ms > 0
        else 0.0
    )

    decode_tps = (
        decode_tokens_count / (decode_total_ms / 1000.0)
        if decode_total_ms > 0 and decode_tokens_count > 0
        else 0.0
    )

    output_tps = (
        output_tokens / (model_core_e2e_ms / 1000.0)
        if model_core_e2e_ms > 0
        else 0.0
    )

    # CUDA memory
    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)

    # KV-cache and weight accounting (constant across repetitions).
    kv_cache = _kv_cache_accounting(
        model,
        input_token_count + output_tokens,
        dtype_bytes=_model_element_size(model),
    )

    result: Dict[str, Any] = {
        "input_tokens": input_token_count,
        "output_tokens": output_tokens,
        "prefill_forward_ms": prefill_forward_ms,
        "first_token_selection_ms": first_token_selection_ms,
        "model_core_ttft_ms": model_core_ttft_ms,
        "decode_total_ms": decode_total_ms,
        "model_core_e2e_ms": model_core_e2e_ms,
        "prefill_tokens_per_s": prefill_tps,
        "decode_tokens_per_s": decode_tps,
        "model_core_output_tokens_per_s": output_tps,
        "itl": latency_summary(itl_ms),
        "raw_itl_ms": itl_ms,
        "peak_cuda_allocated_mb": peak_allocated_mb,
        "peak_cuda_reserved_mb": peak_reserved_mb,
        "generated_token_ids": generated_tokens,
        "kv_cache": kv_cache,
        "model_weight_bytes": model_weight_bytes(model),
        "process_rss_bytes": process_rss_bytes(),
        "process_swap_bytes": process_swap_bytes(),
    }

    logger.debug(
        "Benchmark complete: ISL=%d OSL=%d TTFT=%.2fms E2E=%.2fms "
        "Prefill=%.1ftok/s Decode=%.1ftok/s",
        input_token_count,
        output_tokens,
        model_core_ttft_ms,
        model_core_e2e_ms,
        prefill_tps,
        decode_tps,
    )

    return result
