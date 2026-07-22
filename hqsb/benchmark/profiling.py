"""PyTorch Profiler integration for prefill/decode hotspot analysis.

Produces a structured operator table (name, count, CPU time, CUDA time,
device memory, input shapes) for a representative prefill and decode pass,
which feeds :mod:`hqsb.benchmark.roofline` for classification and ranking.

Timing caution: profiling adds significant overhead and changes scheduling,
so profile numbers are used for *relative* hotspot ranking and hardware
evidence, never as the official latency baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from hqsb.core.errors import BenchmarkError


def _norm_shapes(shapes: Any, limit: int = 8) -> List[str]:
    """Normalize a raw ``input_shapes`` value into a bounded list of strings.

    ``input_shapes`` may be a list of tuples (one per call) or None. We keep
    only the first ``limit`` unique shapes to bound output size.
    """
    if shapes is None:
        return []
    normalized: List[str] = []
    seen = set()
    try:
        for shape in shapes:
            text = str(tuple(shape)) if isinstance(shape, (list, tuple)) else str(shape)
            if text not in seen:
                seen.add(text)
                normalized.append(text)
            if len(normalized) >= limit:
                break
    except TypeError:
        return []
    return normalized


def _self_device_time_us(event: Any) -> float:
    """Return self device (CUDA) time in µs, across PyTorch versions.

    PyTorch >= 2.0 renamed ``self_cuda_time_total`` to
    ``self_device_time_total``; support both so the table is version-robust.
    """
    value = getattr(event, "self_device_time_total", None)
    if value is None:
        value = getattr(event, "self_cuda_time_total", 0.0)
    return float(value or 0.0)


def _collect_input_shapes(prof: Any, limit: int = 8) -> Dict[str, List[str]]:
    """Collect unique input shapes per operator from raw profiler events.

    ``key_averages()`` loses per-call input shapes, so we scan the raw
    event list when available. A single event's ``input_shapes`` is a tuple
    of per-input shapes (e.g. ``((1, 128), (128, 1024))``); we render it as
    a canonical string. Returns a mapping ``op_key -> [shape strings]``.
    """
    shapes_by_key: Dict[str, List[str]] = {}
    try:
        events = prof.events()
    except Exception:
        return shapes_by_key

    for event in events:
        key = getattr(event, "key", None)
        if key is None:
            continue
        shapes = getattr(event, "input_shapes", None)
        if not shapes:
            continue
        text = str(tuple(shapes)) if isinstance(shapes, (list, tuple)) else str(shapes)
        bucket = shapes_by_key.setdefault(key, [])
        if text not in bucket:
            bucket.append(text)
        if len(bucket) >= limit:
            continue
    return shapes_by_key


def extract_operator_table(prof: Any) -> List[Dict[str, Any]]:
    """Convert a PyTorch profiler result into a structured operator table.

    Each row contains the aggregated ``key`` (operator name), call count,
    self CPU time (µs), self device/CUDA time (µs), self device memory
    (bytes), up to 8 unique input shapes, and the row ``scope``.

    Scope (important for time attribution)
    --------------------------------------
    Kineto reports the *same* GPU work once under the host-side operator that
    launched it (e.g. ``aten::mm``, whose ``self_device_time_total`` is the
    device time of its correlated kernels) and once as the device kernel
    itself (e.g. ``ampere_fp16_s16816gemm...``). Summing ``cuda_time_us``
    across both scopes therefore double-counts every kernel exactly twice.

    Device kernels have no host self time, so ``self_cpu_time_total == 0``
    identifies them; every host-side row (ATen ops and CUDA runtime API calls
    such as ``cudaLaunchKernel``) has a positive self CPU time. The ``scope``
    field is ``"kernel"`` for the former and ``"cpu"`` for the latter; a time
    share must be taken within one scope (see :func:`total_device_time_us`),
    never across both.

    Args:
        prof: A ``torch.profiler.profile`` result (after ``__exit__``).

    Returns:
        Rows sorted by descending self device time.
    """
    try:
        key_averages = prof.key_averages()
    except Exception as exc:
        raise BenchmarkError(f"failed to read profiler key averages: {exc}") from exc

    shapes_by_key = _collect_input_shapes(prof)

    rows: List[Dict[str, Any]] = []
    for event in key_averages:
        cpu_time_us = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
        rows.append(
            {
                "name": event.key,
                "count": int(event.count),
                "cpu_time_us": cpu_time_us,
                "cuda_time_us": _self_device_time_us(event),
                "device_memory_bytes": int(
                    getattr(event, "self_device_memory_usage", 0) or 0
                ),
                "input_shapes": shapes_by_key.get(event.key, [])[:8],
                "scope": "kernel" if cpu_time_us == 0.0 else "cpu",
            }
        )

    return sorted(rows, key=lambda r: r["cuda_time_us"], reverse=True)


def cumulative_kernel_time_us(rows: List[Dict[str, Any]]) -> float:
    """Return the *cumulative GPU kernel work time* of a table, in µs.

    Sums ``cuda_time_us`` over the ``"kernel"`` scope only. This is **not** the
    phase wall-clock time and must not be called the "total GPU time": when
    kernels run on multiple streams and overlap, the sum of kernel durations
    can exceed the wall-clock span of the phase. It is a conservative upper
    bound on device work, used only to normalise per-scope time shares.

    Falling back to summing every row when no kernel-scoped row exists keeps
    the helper usable for hand-built tables (e.g. unit-test fixtures) and for
    CPU-only runs where device times are all zero anyway.
    """
    kernels = [r for r in rows if r.get("scope") == "kernel"]
    basis = kernels if kernels else rows
    return float(sum(float(r.get("cuda_time_us", 0.0) or 0.0) for r in basis))


def scope_totals_us(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return ``{scope: cumulative cuda_time_us}`` for each scope in ``rows``.

    ``"kernel"`` is the cumulative GPU kernel work; ``"cpu"`` is the same
    kernels attributed to the host-side operators that launched them. They are
    two views of the same device work and are *not* additive.
    """
    totals: Dict[str, float] = {}
    for row in rows:
        scope = str(row.get("scope", "unknown"))
        totals[scope] = totals.get(scope, 0.0) + float(
            row.get("cuda_time_us", 0.0) or 0.0
        )
    return totals


def split_by_scope(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a table into ``(host_op_rows, device_kernel_rows)``."""
    cpu_rows = [r for r in rows if r.get("scope") == "cpu"]
    kernel_rows = [r for r in rows if r.get("scope") == "kernel"]
    return cpu_rows, kernel_rows


def attach_time_share(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add ``time_share`` to every row, normalised **within each scope**.

    Each scope is normalised by its own cumulative device time, so the shares
    within one scope sum to 1.0. Shares from different scopes must never be
    added together.
    """
    totals = scope_totals_us(rows)
    result: List[Dict[str, Any]] = []
    for row in rows:
        scope = str(row.get("scope", "unknown"))
        denom = totals.get(scope, 0.0) or 1.0
        augmented = dict(row)
        augmented["time_share"] = float(row.get("cuda_time_us", 0.0) or 0.0) / denom
        result.append(augmented)
    return result


def export_chrome_trace(prof: Any, path: str) -> bool:
    """Export a Chrome trace for later audit; return ``False`` on failure.

    Failure is non-fatal: the aggregate tables remain valid, but the raw trace
    that the E02-02 audit depends on would be missing, so callers should record
    the boolean.
    """
    try:
        prof.export_chrome_trace(path)
        return True
    except Exception:
        return False


def _run_profiled(
    *,
    profiler: Any,
    model: Any,
    prefill_fn: Any,
    decode_fn: Any,
) -> None:
    """Run prefill then decode under a single profiler context."""
    # Prefill
    outputs = prefill_fn()
    # Decode (reuse KV cache)
    decode_fn(outputs)


@torch.inference_mode()
def profile_model_core(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    output_tokens: int,
    *,
    record_shapes: bool = True,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Profile one prefill + ``output_tokens`` decode steps.

    Args:
        model: HF causal LM in eval mode on the target device.
        inputs: ``input_ids``/``attention_mask`` tensors (1, ISL).
        output_tokens: Number of decode steps to profile (>= 2 to capture
            at least one decode iteration; the first output token is the
            prefill logits argmax).

    Returns:
        A ``(profiler, operator_table)`` tuple. ``profiler`` is the raw
        ``torch.profiler.profile`` object; ``operator_table`` is the
        structured table from :func:`extract_operator_table`.

    Raises:
        ValueError: If ``output_tokens`` < 1.
    """
    if output_tokens < 1:
        raise ValueError(f"output_tokens must be >= 1, got {output_tokens}")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    device = input_ids.device
    input_len = input_ids.shape[1]

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=record_shapes,
        profile_memory=True,
        with_stack=False,
    )

    profiler.start()

    # Prefill
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    current_length = input_len

    # Decode
    for _ in range(1, output_tokens):
        current_length += 1
        decode_mask = torch.ones(
            (1, current_length), dtype=torch.long, device=device
        )
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
    profiler.stop()

    table = extract_operator_table(profiler)
    return profiler, table


__all__ = [
    "attach_time_share",
    "cumulative_kernel_time_us",
    "export_chrome_trace",
    "extract_operator_table",
    "profile_model_core",
    "scope_totals_us",
    "split_by_scope",
]
