"""Batch-aware model-core benchmark engine for E02-04.

E02-04 answers: *how does static batch size B change throughput, per-request
latency and memory, and where is the safe capacity boundary before OOM?*

``benchmark_model_core`` (see :mod:`hqsb.benchmark.model_core`) is the B=1
reference. This module adds a batch counterpart that feeds a real ``[B, I]``
token tensor (never a broadcast view) through one prefill + ``G-1``
autoregressive decode steps with a shared ``[B, Hkv, T, 128]`` KV cache, and
reports the *batch* throughput metrics defined in E02-04 §6 together with
batch-aware KV accounting and per-row generation ledgers.

Key invariants (see ``docs/stage_experiments/details/S02/E02-04``):

* Static batch only — no request-arrival model, no queueing, no continuous
  batching, no prefix sharing. ``B`` is the width of one forward pass.
* The three token budgets are reported separately (E02-04 §4):

  * input compute tokens    = ``B * I``
  * reserved token capacity = ``B * (I + G)``
  * final KV tokens         = ``B * (I + G - 1)``

* Batch throughput divides by the *batch* work, not by amortized per-request
  wall time; ``Tbatch / B`` is resource amortization, never user latency.
* Every row's generated sequence is kept so the runner can check for
  cross-row contamination, dropped rows, or padding counted as generation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch

from hqsb.benchmark.memory import (
    compute_kv_cache_info,
    model_kv_cache_config,
    model_weight_bytes,
    process_rss_bytes,
    process_swap_bytes,
)
from hqsb.benchmark.metrics import latency_summary, model_core_timings

logger = logging.getLogger(__name__)


class BatchBenchmarkOOM(RuntimeError):
    """Raised when a batched forward fails with an out-of-memory condition.

    Carries the phase (``prefill`` / ``decode``) and the decode step index so
    the runner can record *where* capacity was exhausted instead of guessing
    (E02-04 §7: OOM in prefill vs late decode point at different memory
    items).
    """

    def __init__(self, stage: str, step: int, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.step = step


def _is_oom(exc: BaseException) -> bool:
    """Best-effort detection of a CUDA / allocator out-of-memory condition."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _model_element_size(model: Any) -> int:
    """Element size (bytes) of the model weights; defaults to FP16's 2."""
    try:
        return next(model.parameters()).element_size()
    except (StopIteration, AttributeError):
        return 2


def batch_token_budgets(batch_size: int, isl: int, osl: int) -> Dict[str, int]:
    """The three token budgets E02-04 §4 requires (kept separate, not summed).

    Args:
        batch_size: Static batch width ``B``.
        isl: Input sequence length ``I`` per row.
        osl: Output sequence length ``G`` per row.

    Returns:
        ``input_compute_tokens``, ``reserved_token_capacity`` and
        ``final_kv_tokens``.
    """
    if min(batch_size, isl, osl) < 1:
        raise ValueError(
            f"batch_size/isl/osl must be >= 1, got {batch_size}/{isl}/{osl}"
        )
    return {
        "input_compute_tokens": batch_size * isl,
        "reserved_token_capacity": batch_size * (isl + osl),
        "final_kv_tokens": batch_size * (isl + osl - 1),
    }


def compute_batch_kv_cache_info(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int,
    context_length: int,
    dtype_bytes: int,
) -> Dict[str, Any]:
    """Batch-aware KV-cache byte accounting.

    ``compute_kv_cache_info`` gives the *per-sequence* numbers; the batch
    total is that scaled by ``batch_size`` (weights are shared, KV is not).
    """
    single = compute_kv_cache_info(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        context_length=context_length,
        dtype_bytes=dtype_bytes,
    )
    return {
        "num_layers": single.num_layers,
        "num_kv_heads": single.num_kv_heads,
        "head_dim": single.head_dim,
        "context_length": context_length,
        "batch_size": batch_size,
        "element_bytes": single.element_bytes,
        "per_token_bytes": single.per_token_bytes(),
        "per_sequence_bytes": single.total_bytes(),
        "total_bytes": batch_size * single.total_bytes(),
    }


def _kv_cache_accounting_batch(
    model: Any,
    batch_size: int,
    context_length: int,
    *,
    dtype_bytes: int,
) -> Dict[str, Any]:
    """Read KV structural params from ``model`` and account for a batch."""
    config = model_kv_cache_config(model)
    required = {"num_layers", "num_kv_heads", "head_dim"}
    if not required.issubset(config) or batch_size < 1:
        return {}
    try:
        return compute_batch_kv_cache_info(
            num_layers=config["num_layers"],
            num_kv_heads=config["num_kv_heads"],
            head_dim=config["head_dim"],
            batch_size=batch_size,
            context_length=context_length,
            dtype_bytes=dtype_bytes,
        )
    except ValueError:
        return {}


def batch_throughput(
    batch_size: int,
    isl: int,
    osl: int,
    prefill_ms: float,
    decode_total_ms: float,
    e2e_ms: float,
) -> Dict[str, float]:
    """Batch-level throughput metrics (E02-04 §6).

    All three divide *batch* work by the corresponding batch phase time; none
    of them is a per-request latency:
    """
    prefill_s = prefill_ms / 1000.0
    decode_s = decode_total_ms / 1000.0
    e2e_s = e2e_ms / 1000.0
    return {
        "batch_prefill_tokens_per_s": (
            batch_size * isl / prefill_s if prefill_s > 0 else 0.0
        ),
        "batch_decode_tokens_per_s": (
            batch_size * (osl - 1) / decode_s
            if decode_s > 0 and osl > 1
            else 0.0
        ),
        "batch_output_tokens_per_s": (
            batch_size * osl / e2e_s if e2e_s > 0 else 0.0
        ),
    }


@torch.inference_mode()
def benchmark_model_core_batch(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_tokens: int,
    *,
    capture_logits: bool = False,
) -> Dict[str, Any]:
    """Run one *batched* model-core benchmark pass.

    Mirrors :func:`hqsb.benchmark.model_core.benchmark_model_core` for a
    static batch: one prefill over ``[B, I]``, one first-token selection over
    ``[B, vocab]``, then ``G-1`` autoregressive decode steps with a shared
    ``[B, Hkv, T, 128]`` KV cache. Every timed interval is wrapped in
    ``torch.cuda.synchronize()`` so asynchronous launches are not misrecorded
    as completion (same clock convention as E02-01).

    Args:
        model: HF causal LM in eval mode on the target device.
        input_ids: ``[B, I]`` long tensor (real copies, not a broadcast view).
        attention_mask: ``[B, I]`` long tensor.
        output_tokens: ``G``, generated tokens per row (>= 1).
        capture_logits: When True, capture the *row-0* first-token logits and
            top-K (reference only; full-vocab per-row capture is not needed
            for E02-04 correctness and would be ``B x 151936`` large).

    Returns:
        Dict with batch size, per-row generation ledger, batch TTFT / TPOT /
        E2E / throughput, peak CUDA allocated/reserved, RSS/swap, and
        batch-aware KV accounting. ``generated_token_ids`` is a list of ``B``
        rows so cross-row contamination can be checked by the runner.

    Raises:
        ValueError: Bad ``output_tokens`` or a non-2D ``input_ids``.
        BatchBenchmarkOOM: An OOM during ``prefill`` or ``decode`` (step
            recorded), wrapped so the runner can record the failure stage.
    """
    if output_tokens < 1:
        raise ValueError(f"output_tokens must be >= 1, got {output_tokens}")
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D (B, I), got {input_ids.dim()}D")
    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must be 2D (B, I), got {attention_mask.dim()}D"
        )

    batch_size = input_ids.shape[0]
    input_len = input_ids.shape[1]
    device = input_ids.device

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ── Phase 1: Prefill ──────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    except Exception as exc:  # noqa: BLE001 — re-wrap OOM with phase
        if _is_oom(exc):
            raise BatchBenchmarkOOM("prefill", 0, str(exc)) from exc
        raise
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_forward_ms = (time.perf_counter() - prefill_start) * 1000.0

    # ── Phase 2: First-token selection ────────────────────────────
    first_token_start = time.perf_counter()
    last_logits = outputs.logits[:, -1, :]  # (B, vocab)
    next_token = last_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
    past_key_values = outputs.past_key_values
    if device.type == "cuda":
        torch.cuda.synchronize()
    first_token_selection_ms = (
        time.perf_counter() - first_token_start
    ) * 1000.0

    first_tokens = next_token.flatten().tolist()
    generated: List[List[int]] = [[int(t)] for t in first_tokens]

    first_token_logits: Optional[List[float]] = None
    first_token_topk: Optional[List[Dict[str, Any]]] = None
    if capture_logits:
        first_token_logits = [float(x) for x in last_logits[0].tolist()]
        k = min(10, last_logits.shape[-1])
        topk_values, topk_indices = torch.topk(last_logits[0], k=k)
        first_token_topk = [
            {"token_id": int(idx), "logit": float(val)}
            for val, idx in zip(topk_values.tolist(), topk_indices.tolist())
        ]

    # ── Phase 3: Decode (G-1 autoregressive steps) ────────────────
    itl_ms: List[float] = []
    current_length = input_len
    try:
        for step in range(1, output_tokens):
            current_length += 1
            decode_mask = torch.ones(
                (batch_size, current_length), dtype=torch.long, device=device
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            decode_start = time.perf_counter()
            outputs = model(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - decode_start) * 1000.0
            itl_ms.append(elapsed_ms)
            for row_idx, tok in enumerate(next_token.flatten().tolist()):
                generated[row_idx].append(int(tok))
    except Exception as exc:  # noqa: BLE001 — re-wrap OOM with decode step
        if _is_oom(exc):
            raise BatchBenchmarkOOM("decode", step, str(exc)) from exc
        raise

    # ── Aggregate metrics ─────────────────────────────────────────
    phase = model_core_timings(prefill_forward_ms, first_token_selection_ms, itl_ms)
    decode_total_ms = phase["decode_total_ms"]
    model_core_ttft_ms = phase["model_core_ttft_ms"]
    model_core_e2e_ms = phase["model_core_e2e_ms"]

    throughput = batch_throughput(
        batch_size,
        input_len,
        output_tokens,
        prefill_forward_ms,
        decode_total_ms,
        model_core_e2e_ms,
    )

    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)

    kv_cache = _kv_cache_accounting_batch(
        model,
        batch_size,
        input_len + output_tokens - 1,
        dtype_bytes=_model_element_size(model),
    )

    result: Dict[str, Any] = {
        "batch_size": batch_size,
        "input_tokens": input_len,
        "output_tokens": output_tokens,
        "prefill_forward_ms": prefill_forward_ms,
        "first_token_selection_ms": first_token_selection_ms,
        "model_core_ttft_ms": model_core_ttft_ms,
        "decode_total_ms": decode_total_ms,
        "model_core_e2e_ms": model_core_e2e_ms,
        "batch_prefill_tokens_per_s": throughput["batch_prefill_tokens_per_s"],
        "batch_decode_tokens_per_s": throughput["batch_decode_tokens_per_s"],
        "batch_output_tokens_per_s": throughput["batch_output_tokens_per_s"],
        # E02-04 §6: Tbatch / B is resource amortization, NOT user latency.
        "amortized_per_request_ms": model_core_e2e_ms / batch_size,
        "itl": latency_summary(itl_ms),
        "raw_itl_ms": itl_ms,
        "peak_cuda_allocated_mb": peak_allocated_mb,
        "peak_cuda_reserved_mb": peak_reserved_mb,
        "generated_token_ids": generated,
        "kv_cache": kv_cache,
        "model_weight_bytes": model_weight_bytes(model),
        "process_rss_bytes": process_rss_bytes(),
        "process_swap_bytes": process_swap_bytes(),
    }

    if capture_logits:
        result["first_token_logits"] = first_token_logits
        result["first_token_topk"] = first_token_topk

    logger.debug(
        "Batch benchmark complete: B=%d ISL=%d OSL=%d TTFT=%.2fms E2E=%.2fms "
        "batch_prefill=%.1ftok/s batch_decode=%.1ftok/s batch_output=%.1ftok/s",
        batch_size,
        input_len,
        output_tokens,
        model_core_ttft_ms,
        model_core_e2e_ms,
        throughput["batch_prefill_tokens_per_s"],
        throughput["batch_decode_tokens_per_s"],
        throughput["batch_output_tokens_per_s"],
    )

    return result


__all__ = [
    "BatchBenchmarkOOM",
    "batch_throughput",
    "batch_token_budgets",
    "benchmark_model_core_batch",
    "compute_batch_kv_cache_info",
]
