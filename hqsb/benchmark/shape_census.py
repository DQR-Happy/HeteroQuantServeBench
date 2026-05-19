"""Module/operator shape census for E02-02.

E02-02 answers one question: *what are the real shapes and call counts of
the kernels that S03 will optimize, and do we actually cover the full
decoder block, the KV cache, and the LM head?*

Two complementary collectors produce that census without any hand-copied
shapes — every value is read off live tensors during a real forward pass:

* :class:`ShapeCensusCollector` — runtime ``forward`` hooks record, per
  ``(module_path, phase)``: call count, input/output shapes, dtypes,
  strides, memory layouts, and contiguity. This is the "runtime hook"
  half and covers the *module* level (``model.layers.0.self_attn.q_proj``,
  etc.).
* :func:`collect_shape_census` — runs one prefill pass and a pre-registered
  set of *representative* decode steps (early / middle / late) under a fresh
  :class:`torch.profiler.profile`, then reuses
  :func:`hqsb.benchmark.profiling.extract_operator_table` to emit, per phase,
  **two independent tables**: the ATen-op view and the device-kernel view.
  Each table is normalised by its own scope, because Kineto reports the same
  GPU work once per host op and once per device kernel.

Timing terminology (E02-02 rework): the per-scope sum of ``cuda_time_us`` is
*cumulative GPU kernel work time*, **not** the phase wall-clock time — with
multiple streams it can exceed the wall-clock span. It is used only to
normalise shares within one scope.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from hqsb.benchmark.correctness import hash_token_sequence
from hqsb.benchmark.profiling import (
    attach_time_share,
    export_chrome_trace,
    extract_operator_table,
    split_by_scope,
)

# Cap on the number of distinct shapes/strides/layouts/dtypes recorded per
# (module, phase, direction) so a pathological module cannot bloat the raw
# JSON. It must be large enough to hold the full KV/mask growth of the longest
# workload: decode_heavy grows the KV seq length through 255 distinct values
# (ISL..ISL+G-1) plus 255 distinct attention-mask lengths, so 1024 keeps the
# whole step ledger instead of silently truncating the tail.
_MAX_UNIQUE = 1024


def _dtype_name(dtype: torch.dtype) -> str:
    """Short dtype name (``float16`` instead of ``torch.float16``)."""
    return str(dtype).removeprefix("torch.")


def _tensor_meta(tensor: torch.Tensor) -> Dict[str, Any]:
    """Describe a single tensor: shape, dtype, stride, layout, contiguity, device.

    ``layout`` is a compact classification: ``contiguous``, ``channels_last``,
    or ``non_contiguous`` (sparse/other layouts are rendered by name). The
    raw stride is kept alongside so the exact memory pattern is preserved.
    """
    if tensor.layout != torch.strided:
        layout = str(tensor.layout)
    elif tensor.is_contiguous():
        layout = "contiguous"
    elif tensor.is_contiguous(memory_format=torch.channels_last):
        layout = "channels_last"
    else:
        layout = "non_contiguous"
    return {
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "layout": layout,
        "contiguous": bool(tensor.is_contiguous()),
    }


def _shape_str(shape: List[int]) -> str:
    return "[" + ", ".join(str(s) for s in shape) + "]"


def _stride_str(stride: List[int]) -> str:
    return "(" + ", ".join(str(s) for s in stride) + ")"


def _flatten_tensors(obj: Any, out: List[torch.Tensor]) -> None:
    """Recursively collect every tensor reachable from ``obj``.

    Handles tensors, tuples/lists, dicts, dataclasses, the transformers
    ``DynamicCache`` (``key_cache``/``value_cache``), and generic
    ``ModelOutput`` objects (via ``__dict__``). Non-tensor leaves (ints,
    bools, strings, ``None``) are skipped.
    """
    if isinstance(obj, torch.Tensor):
        out.append(obj)
        return
    if isinstance(obj, (tuple, list)):
        for item in obj:
            _flatten_tensors(item, out)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _flatten_tensors(value, out)
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            _flatten_tensors(getattr(obj, field.name), out)
        return
    if hasattr(obj, "key_cache") and hasattr(obj, "value_cache"):
        # transformers DynamicCache: the KV cache we must census.
        _flatten_tensors(obj.key_cache, out)
        _flatten_tensors(obj.value_cache, out)
        return
    if hasattr(obj, "__dict__"):
        for value in vars(obj).values():
            _flatten_tensors(value, out)
        return


class ShapeCensusCollector:
    """Runtime forward-hook collector for per-module, per-phase shape census.

    Registers a pre-hook (inputs) and a forward hook (outputs) on every
    submodule. The current phase (``prefill``/``decode``) is a mutable
    attribute set by the caller between passes, so one collector can serve
    both phases without re-registering hooks.
    """

    def __init__(self, max_unique: int = _MAX_UNIQUE) -> None:
        self._max_unique = max_unique
        self._phase = "prefill"
        self._active = False
        self._handles: List[Tuple[Any, Any]] = []
        self._records: Dict[Tuple[str, str], Dict[str, Any]] = {}

    @property
    def phase(self) -> str:
        return self._phase

    def set_phase(self, phase: str) -> None:
        """Switch the phase recorded by subsequent hook invocations."""
        self._phase = phase

    def attach(self, model: torch.nn.Module) -> None:
        """Register hooks on every submodule (root labeled ``root``)."""
        for path, module in model.named_modules():
            label = path or "root"
            pre = module.register_forward_pre_hook(
                self._make_pre_hook(label), with_kwargs=True
            )
            post = module.register_forward_hook(self._make_post_hook(label))
            self._handles.append((pre, post))
        self._active = True

    def detach(self) -> None:
        """Remove all registered hooks and freeze recording."""
        for pre, post in self._handles:
            pre.remove()
            post.remove()
        self._handles.clear()
        self._active = False

    def records(self) -> List[Dict[str, Any]]:
        """Return records sorted by ``(module, phase)`` for stable output."""
        ordered = sorted(self._records.values(), key=lambda r: (r["module"], r["phase"]))
        return [dict(r) for r in ordered]

    def _make_pre_hook(self, label: str):
        def pre_hook(module, args, kwargs):
            if not self._active:
                return
            tensors: List[torch.Tensor] = []
            _flatten_tensors(list(args), tensors)
            _flatten_tensors(kwargs or {}, tensors)
            record = self._get_or_create(label, module)
            record["call_count"] += 1
            self._accumulate(record, "input", tensors)

        return pre_hook

    def _make_post_hook(self, label: str):
        def post_hook(module, args, output):
            if not self._active:
                return
            tensors: List[torch.Tensor] = []
            _flatten_tensors(output, tensors)
            record = self._get_or_create(label, module)
            self._accumulate(record, "output", tensors)

        return post_hook

    def _get_or_create(
        self, label: str, module: torch.nn.Module
    ) -> Dict[str, Any]:
        key = (label, self._phase)
        record = self._records.get(key)
        if record is None:
            record = {
                "module": label,
                "module_type": type(module).__name__,
                "phase": self._phase,
                "call_count": 0,
                "input_shapes": [],
                "input_dtypes": [],
                "input_devices": [],
                "input_strides": [],
                "input_layouts": [],
                "input_contiguous": True,
                "output_shapes": [],
                "output_dtypes": [],
                "output_devices": [],
                "output_strides": [],
                "output_layouts": [],
                "output_contiguous": True,
            }
            self._records[key] = record
        return record

    def _accumulate(
        self, record: Dict[str, Any], direction: str, tensors: List[torch.Tensor]
    ) -> None:
        shape_field = f"{direction}_shapes"
        dtype_field = f"{direction}_dtypes"
        device_field = f"{direction}_devices"
        stride_field = f"{direction}_strides"
        layout_field = f"{direction}_layouts"
        contiguous_field = f"{direction}_contiguous"

        for tensor in tensors:
            meta = _tensor_meta(tensor)
            self._append_unique(record[shape_field], _shape_str(meta["shape"]))
            self._append_unique(record[dtype_field], meta["dtype"])
            self._append_unique(record[device_field], meta["device"])
            self._append_unique(record[stride_field], _stride_str(meta["stride"]))
            self._append_unique(record[layout_field], meta["layout"])
            if not meta["contiguous"]:
                record[contiguous_field] = False

    def _append_unique(self, bucket: List[str], value: str) -> None:
        if value not in bucket and len(bucket) < self._max_unique:
            bucket.append(value)


def _new_profiler() -> torch.profiler.profile:
    """Create the profiler used for census probes (shapes + CPU/CUDA time).

    ``profile_memory`` stays off: memory accounting is E02-05's concern and
    inflates profiler memory on the 8 GiB unified-memory device.
    """
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    )


def _default_probe_steps(total_steps: int) -> List[int]:
    """Pre-registered representative decode steps: early, middle, late."""
    if total_steps <= 0:
        return []
    steps = {1, total_steps}
    if total_steps >= 3:
        steps.add((total_steps + 1) // 2)
    return sorted(s for s in steps if 1 <= s <= total_steps)


def _aggregate_operator_tables(
    tables: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Sum per-step tables by ``(scope, name)`` and re-normalise by scope."""
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for table in tables:
        for row in table:
            key = (str(row.get("scope")), str(row["name"]))
            current = merged.get(key)
            if current is None:
                current = dict(row)
                current["input_shapes"] = list(row.get("input_shapes", []))
                merged[key] = current
            else:
                current["count"] += int(row["count"])
                current["cuda_time_us"] += float(row["cuda_time_us"])
                current["cpu_time_us"] += float(row["cpu_time_us"])
                for shape in row.get("input_shapes", []):
                    if shape not in current["input_shapes"] and len(
                        current["input_shapes"]
                    ) < 16:
                        current["input_shapes"].append(shape)
    return attach_time_share(list(merged.values()))


@torch.inference_mode()
def collect_shape_census(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    output_tokens: int,
    *,
    decode_probe_steps: Optional[List[int]] = None,
    trace_dir: Optional[str] = None,
    prefill_trace_max_isl: Optional[int] = None,
) -> Dict[str, Any]:
    """Collect the module + operator shape census for one workload.

    Runs one full prefill forward (hook + profiler) and the *full* decode of
    ``output_tokens - 1`` steps (hook only, so module ``call_count`` is the
    true runtime count: prefill 1, decode G-1). The profiler covers only the
    pre-registered representative decode steps (early / middle / late), never
    the whole generation, so profiling overhead does not distort the call
    counts.

    Per phase it returns **two independent tables**: ``aten_ops`` (host-side
    ``scope="cpu"``) and ``kernels`` (device ``scope="kernel"``); the two are
    views of the same GPU work and must not be summed. Raw Chrome traces are
    written under ``trace_dir`` when given, for the external audit.

    Args:
        model: HF causal LM in eval mode on the target device.
        inputs: ``input_ids``/``attention_mask`` tensors of shape ``(1, ISL)``.
        output_tokens: Configured OSL (G). Must be >= 1.
        decode_probe_steps: Explicit 1-based decode step indices to profile.
            Defaults to early / middle / late (see :func:`_default_probe_steps`).
        trace_dir: When set, export one Chrome trace per profiled region.

    Returns:
        A dict with ``modules`` (hook census), ``prefill``/``decode`` tables,
        ``decode_probe_steps``, ``decode_steps`` (full = G-1), and the
        instrumented ``generated_token_ids`` / ``sequence_sha256`` (for the
        caller to compare against the un-instrumented reference).
    """
    if output_tokens < 1:
        raise ValueError(f"output_tokens must be >= 1, got {output_tokens}")

    full_decode_steps = output_tokens - 1
    if decode_probe_steps is None:
        probe_steps = _default_probe_steps(full_decode_steps)
    else:
        probe_steps = sorted({int(s) for s in decode_probe_steps})
    probe_steps = [s for s in probe_steps if 1 <= s <= full_decode_steps]
    probe_set = set(probe_steps)

    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    device = input_ids.device
    input_len = input_ids.shape[1]

    collector = ShapeCensusCollector()
    collector.attach(model)

    # ── Prefill: full forward, hook + profiler, one trace ──────────────
    collector.set_phase("prefill")
    profiler = _new_profiler()
    profiler.start()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    if device.type == "cuda":
        torch.cuda.synchronize()
    profiler.stop()
    prefill_rows = attach_time_share(extract_operator_table(profiler))
    prefill_ops, prefill_kernels = split_by_scope(prefill_rows)
    prefill_trace: Optional[str] = None
    if trace_dir and (
        prefill_trace_max_isl is None or input_len <= prefill_trace_max_isl
    ):
        candidate = os.path.join(trace_dir, "prefill_trace.json")
        if export_chrome_trace(profiler, candidate):
            prefill_trace = candidate
    del profiler

    # ── Decode: all G-1 steps under the hook; profiler at probe steps ──
    collector.set_phase("decode")
    generated_tokens: List[int] = [int(next_token.item())]
    per_step: Dict[str, Any] = {}
    current_length = input_len
    for step in range(1, output_tokens):
        is_probe = step in probe_set
        profiler = _new_profiler() if is_probe else None
        if profiler is not None:
            profiler.start()

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
        generated_tokens.append(int(next_token.item()))

        if profiler is not None:
            if device.type == "cuda":
                torch.cuda.synchronize()
            profiler.stop()
            step_rows = attach_time_share(extract_operator_table(profiler))
            step_ops, step_kernels = split_by_scope(step_rows)
            entry: Dict[str, Any] = {
                "step": step,
                "context_len": current_length,
                "aten_ops": step_ops,
                "kernels": step_kernels,
                "trace": None,
            }
            if trace_dir:
                candidate = os.path.join(trace_dir, f"decode_step{step}_trace.json")
                if export_chrome_trace(profiler, candidate):
                    entry["trace"] = candidate
            per_step[str(step)] = entry
            del profiler

    collector.detach()

    decode_ops_agg = _aggregate_operator_tables(
        [per_step[str(s)]["aten_ops"] for s in probe_steps if str(s) in per_step]
    )
    decode_kernels_agg = _aggregate_operator_tables(
        [per_step[str(s)]["kernels"] for s in probe_steps if str(s) in per_step]
    )

    return {
        "input_len": input_len,
        "output_tokens": output_tokens,
        "decode_steps": full_decode_steps,
        "decode_probe_steps": probe_steps,
        "generated_token_ids": generated_tokens,
        "sequence_sha256": hash_token_sequence(generated_tokens),
        "modules": collector.records(),
        "prefill": {
            "probe_scope": "full prefill forward",
            "aten_ops": prefill_ops,
            "kernels": prefill_kernels,
            "trace": prefill_trace,
        },
        "decode": {
            "probe_scope": f"representative decode steps {probe_steps}",
            "probe_steps": probe_steps,
            "per_step": per_step,
            "aten_ops_cumulative_over_probe_steps": decode_ops_agg,
            "kernels_cumulative_over_probe_steps": decode_kernels_agg,
        },
    }


__all__ = [
    "ShapeCensusCollector",
    "collect_shape_census",
]
