"""Torch-dependent orchestration for E02-06 (cold / first / steady / reset).

The pure decision logic lives in :mod:`hqsb.benchmark.cold_warm`; this module
performs the measurements that feed it. It is deliberately thin and explicit
about *what* is measured before *what*, because E02-06 is an experiment about
ordering and initialization state (see
``docs/stage_experiments/details/S02/E02-06_cold_warm_and_cache_states.md``):

* :func:`probe_request_with_kv` runs one request while **keeping the live KV
  cache** so its per-layer filled length can be read off the object (protocol
  §8 step 7) — the cold/warm experiment, unlike E02-05, needs the cache object
  itself to prove reset works.
* :func:`measure_request` is the frozen single-request clock (E02-01) plus an
  optional external device-memory sampler.
* :func:`negative_control_no_reset` builds the bounded counter-example: append
  request B to request A's live cache and confirm the state check can see it.
* :func:`run_steady` applies the pre-registered warmup / rolling-window /
  maximum-wait rule before any sample is called "steady".
* :func:`allocator_reuse_probe` is the optional allocator diagnostic (case E).
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from hqsb.benchmark.cold_warm import stability_reached, summarize_series
from hqsb.benchmark.memory import process_rss_bytes, process_swap_bytes
from hqsb.benchmark.memory_model import (
    DeviceMemorySampler,
    kv_cache_metadata,
    memory_snapshot,
)
from hqsb.benchmark.metrics import model_core_timings
from hqsb.benchmark.model_core import benchmark_model_core

__all__ = [
    "PROTOCOL",
    "allocator_reuse_probe",
    "dynamo_counters",
    "measure_request",
    "negative_control_no_reset",
    "probe_request_with_kv",
    "run_steady",
    "snapshot",
]

#: Pre-registered protocol constants (E02-06 §7). They are frozen before the
#: formal runs so the steady rule, the warmup floor and the observation cap are
#: method parameters rather than post-hoc choices.
PROTOCOL: Dict[str, Any] = {
    "warmup_min": 3,
    "steady_max_wait": 5,
    "steady_window": 3,
    "steady_rel_tol": 0.10,
    "sampler_interval_s": 0.01,
    "reset_mechanism": "del_live_cache_and_gc_collect",
    "allocator_empty_cache_in_reset": False,
    "clock_domain": "host_monotonic_ns + torch.cuda.synchronize",
    "attention_backend": "eager",
    "dtype": "float16",
}


def snapshot(*, label: str, with_cuda: bool = True) -> Dict[str, Any]:
    """Thin alias over :func:`hqsb.benchmark.memory_model.memory_snapshot`."""
    result = memory_snapshot(label=label, with_cuda=with_cuda)
    result["label"] = label
    return result


def dynamo_counters() -> Optional[Dict[str, Any]]:
    """Return torch ``_dynamo`` counters, or ``None`` when unavailable.

    Used only as *evidence that nothing compiled implicitly*; the frozen run
    never calls ``torch.compile`` (protocol §8 step 9).
    """
    try:  # pragma: no cover - depends on torch build
        from torch._dynamo.utils import counters  # noqa: PLC0415

        return {
            str(key): int(sum(value.values()) if isinstance(value, dict) else value)
            for key, value in dict(counters).items()
        }
    except Exception:  # noqa: BLE001
        return None


@torch.inference_mode()
def probe_request_with_kv(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    output_tokens: int,
) -> Tuple[Dict[str, Any], Any]:
    """Run one request and return ``(metrics, live_cache)``.

    Unlike :func:`benchmark_model_core` this **does not release** the KV cache:
    the caller must hold the returned cache to inspect it, then drop it to
    perform the reset. Peak-memory stats are intentionally *not* reset here so
    the caller keeps control of when a peak window begins.

    The prefill / first-token / decode decomposition and the derived TTFT /
    E2E values use the single E02-01 clock convention
    (:func:`hqsb.benchmark.metrics.model_core_timings`).
    """
    if output_tokens < 1:
        raise ValueError(f"output_tokens must be >= 1, got {output_tokens}")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_token_count = int(input_ids.shape[1])
    device = input_ids.device

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

    kv_prefill = kv_cache_metadata(outputs.past_key_values)

    first_token_start = time.perf_counter()
    last_logits = outputs.logits[:, -1, :]
    next_token = last_logits.argmax(dim=-1, keepdim=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    first_token_selection_ms = (time.perf_counter() - first_token_start) * 1000.0

    cache = outputs.past_key_values
    generated: List[int] = [int(next_token.item())]
    itl_ms: List[float] = []
    current_length = input_token_count

    for _step in range(1, output_tokens):
        current_length += 1
        decode_mask = torch.ones((1, current_length), dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(
            input_ids=next_token,
            attention_mask=decode_mask,
            past_key_values=cache,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache = outputs.past_key_values
        if device.type == "cuda":
            torch.cuda.synchronize()
        itl_ms.append((time.perf_counter() - start) * 1000.0)
        generated.append(int(next_token.item()))

    kv_final = kv_cache_metadata(cache)
    phase = model_core_timings(prefill_forward_ms, first_token_selection_ms, itl_ms)
    metrics: Dict[str, Any] = {
        "input_tokens": input_token_count,
        "output_tokens": output_tokens,
        "prefill_forward_ms": prefill_forward_ms,
        "first_token_selection_ms": first_token_selection_ms,
        "model_core_ttft_ms": phase["model_core_ttft_ms"],
        "decode_total_ms": phase["decode_total_ms"],
        "model_core_e2e_ms": phase["model_core_e2e_ms"],
        "raw_itl_ms": itl_ms,
        "generated_token_ids": generated,
        "kv_after_prefill": kv_prefill,
        "kv_final": kv_final,
    }
    return metrics, cache


def measure_request(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    output_tokens: int,
    *,
    sampler_interval_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Measure one request with the frozen clock and per-request peak memory.

    ``benchmark_model_core`` resets the allocator high-water marks at the start
    of each call, so the returned peaks belong to this request alone; the
    optional external sampler adds a resolution-independent device-used peak.
    """
    sampler: Optional[DeviceMemorySampler] = None
    if sampler_interval_s:
        sampler = DeviceMemorySampler(interval_s=sampler_interval_s)
        sampler.start()
    try:
        result = benchmark_model_core(model, inputs, output_tokens)
    finally:
        if sampler is not None:
            sampler.stop()
    external_peak = sampler.summary() if sampler is not None else None
    return {"metrics": result, "external_device_peak": external_peak}


def negative_control_no_reset(
    model: Any,
    inputs: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Bounded counter-example: append B to A's live cache ("no reset").

    Prefill once to create a live cache of length ``ISL``, then feed the *same*
    input again **with that cache attached** — exactly what happens if request B
    is served as a continuation instead of an independent request. The cache's
    filled length must grow to ``ISL + ISL``; if it did not, the state check
    would be blind to a failed reset.

    The check is bounded to a single extra prefill (no decode), so it cannot
    grow memory or turn into a long generation.
    """
    input_ids = inputs["input_ids"]
    device = input_ids.device
    input_len = int(input_ids.shape[1])

    with torch.inference_mode():
        first = model(input_ids=input_ids, attention_mask=inputs["attention_mask"],
                      use_cache=True)
        cache = first.past_key_values
        before = kv_cache_metadata(cache)

        combined_mask = torch.ones(
            (1, 2 * input_len), dtype=torch.long, device=device
        )
        second = model(
            input_ids=input_ids,
            attention_mask=combined_mask,
            past_key_values=cache,
            use_cache=True,
        )
        after = kv_cache_metadata(second.past_key_values)
        continuation_token = int(
            second.logits[:, -1, :].argmax(dim=-1).item()
        )

    filled_before = before.get("context_filled")
    filled_after = after.get("context_filled")
    del cache, first, second
    gc.collect()
    return {
        "description": (
            "no-reset continuation: request B's input is appended to request A's "
            "live KV cache"
        ),
        "input_len": input_len,
        "kv_filled_before": filled_before,
        "kv_filled_after": filled_after,
        "expected_filled_after": 2 * input_len,
        "continuation_first_token": continuation_token,
        "detected": filled_after == 2 * input_len and filled_after != input_len,
    }


def run_steady(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    output_tokens: int,
    *,
    warmup_min: int = PROTOCOL["warmup_min"],
    max_wait: int = PROTOCOL["steady_max_wait"],
    window: int = PROTOCOL["steady_window"],
    rel_tol: float = PROTOCOL["steady_rel_tol"],
    sampler_interval_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Warm up, then collect steady samples under the pre-registered rule.

    Every warmup and steady call runs on a **fresh KV cache** (the engine builds
    it per call), which is the "warm model, empty per-request KV" state the
    protocol defines as steady (protocol §3.4). Warmup latencies are recorded
    but are never part of the steady distribution.

    Sampling stops at the first sample count ``>= max(warmup_min, window)``
    whose rolling window is within ``rel_tol``, or at ``max_wait`` samples; if
    the criterion is never met the result is returned with ``reached=False``
    rather than relabelled.
    """
    warmup_latencies: List[float] = []
    for _index in range(warmup_min):
        result = measure_request(model, inputs, output_tokens)
        warmup_latencies.append(result["metrics"]["model_core_e2e_ms"])
    steady_phase_begin = time.monotonic_ns()

    samples: List[Dict[str, Any]] = []
    while len(samples) < max_wait:
        start_ns = time.monotonic_ns()
        measured = measure_request(
            model, inputs, output_tokens, sampler_interval_s=sampler_interval_s
        )
        end_ns = time.monotonic_ns()
        metrics = measured["metrics"]
        samples.append(
            {
                "index": len(samples),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "model_core_e2e_ms": metrics["model_core_e2e_ms"],
                "model_core_ttft_ms": metrics["model_core_ttft_ms"],
                "prefill_forward_ms": metrics["prefill_forward_ms"],
                "first_token_selection_ms": metrics["first_token_selection_ms"],
                "decode_total_ms": metrics["decode_total_ms"],
                "raw_itl_ms": metrics["raw_itl_ms"],
                "generated_token_ids": metrics["generated_token_ids"],
                "peak_cuda_allocated_mb": metrics["peak_cuda_allocated_mb"],
                "peak_cuda_reserved_mb": metrics["peak_cuda_reserved_mb"],
                "process_rss_bytes": process_rss_bytes(),
                "process_swap_bytes": process_swap_bytes(),
                "external_device_peak": measured["external_device_peak"],
            }
        )
        stability = stability_reached(
            [s["model_core_e2e_ms"] for s in samples],
            window=window,
            rel_tol=rel_tol,
            min_samples=max(warmup_min, window),
        )
        if stability["reached"]:
            break

    steady_phase_end = time.monotonic_ns()
    latencies = [s["model_core_e2e_ms"] for s in samples]
    stability = stability_reached(
        latencies, window=window, rel_tol=rel_tol, min_samples=max(warmup_min, window)
    )
    return {
        "warmup_count": len(warmup_latencies),
        "warmup_latencies_ms": warmup_latencies,
        "warmup_summary": summarize_series(warmup_latencies),
        "samples": samples,
        "sample_count": len(samples),
        "latency_summary_ms": summarize_series(latencies),
        "stability": stability,
        "reached": bool(stability["reached"]),
        "steady_phase_window_ns": [steady_phase_begin, steady_phase_end],
        "rule": {
            "warmup_min": warmup_min,
            "max_wait": max_wait,
            "window": window,
            "rel_tol": rel_tol,
        },
    }


def allocator_reuse_probe(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    output_tokens: int,
) -> Dict[str, Any]:
    """Optional case-E diagnostic: does allocator reuse change the next request?

    Compares one request with the allocator cache retained against one after
    ``empty_cache()``. This is a *diagnostic*, never applied before ordinary
    steady samples — injecting ``empty_cache`` into every request would create a
    third running state that does not represent normal repeated serving
    (protocol §8 step 8).
    """
    retained = measure_request(model, inputs, output_tokens)
    before_retained_reserved = None
    if torch.cuda.is_available():
        before_retained_reserved = torch.cuda.memory_reserved() / (1024**2)
    torch.cuda.empty_cache()
    after_empty_reserved = (
        torch.cuda.memory_reserved() / (1024**2)
        if torch.cuda.is_available()
        else None
    )
    freed = measure_request(model, inputs, output_tokens)
    return {
        "enabled": True,
        "retained": {
            "latency_ms": retained["metrics"]["model_core_e2e_ms"],
            "peak_reserved_mb": retained["metrics"]["peak_cuda_reserved_mb"],
            "reserved_before_empty_mb": before_retained_reserved,
        },
        "after_empty_cache": {
            "latency_ms": freed["metrics"]["model_core_e2e_ms"],
            "peak_reserved_mb": freed["metrics"]["peak_cuda_reserved_mb"],
            "reserved_after_empty_mb": after_empty_reserved,
        },
        "latency_delta_ms": (
            freed["metrics"]["model_core_e2e_ms"]
            - retained["metrics"]["model_core_e2e_ms"]
        ),
        "reserved_released_mb": (
            before_retained_reserved - after_empty_reserved
            if before_retained_reserved is not None and after_empty_reserved is not None
            else None
        ),
        "interpretation": (
            "diagnostic only: a small latency delta shows allocator cache reuse "
            "is not a material steady-state cost; reserved release shown separately"
        ),
    }
