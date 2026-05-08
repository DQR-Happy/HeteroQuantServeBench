"""PyTorch Profiler integration for prefill/decode hotspot analysis.

Produces a structured operator table (name, count, CPU time, CUDA time,
device memory, input shapes) for a representative prefill and decode pass,
which feeds :mod:`hqsb.benchmark.roofline` for classification and ranking.

Timing caution: profiling adds significant overhead and changes scheduling,
so profile numbers are used for *relative* hotspot ranking and hardware
evidence, never as the official latency baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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
    (bytes), and up to 8 unique input shapes.

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
        rows.append(
            {
                "name": event.key,
                "count": int(event.count),
                "cpu_time_us": float(
                    getattr(event, "self_cpu_time_total", 0.0) or 0.0
                ),
                "cuda_time_us": _self_device_time_us(event),
                "device_memory_bytes": int(
                    getattr(event, "self_device_memory_usage", 0) or 0
                ),
                "input_shapes": shapes_by_key.get(event.key, [])[:8],
            }
        )

    return sorted(rows, key=lambda r: r["cuda_time_us"], reverse=True)


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
    "extract_operator_table",
    "profile_model_core",
]
