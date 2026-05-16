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
* :func:`collect_shape_census` — runs one prefill pass and ``decode_steps``
  decode passes under a fresh :class:`torch.profiler.profile` each, then
  reuses :func:`hqsb.benchmark.profiling.extract_operator_table` to emit
  per-phase *operator* tables. The operator self-device-time share is the
  "耗时占比" (time share) column.

The two halves are kept separate on purpose: hooks give exact module-level
shapes/dtypes/strides/contiguity (which the profiler's aggregated
``key_averages`` loses per call), while the profiler gives the operator
timing (which naive hook timing on nested modules would double count).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import torch

from hqsb.benchmark.profiling import extract_operator_table

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


def _with_time_share(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach ``time_share`` (self CUDA time / phase total) to operator rows."""
    total = sum(float(r["cuda_time_us"]) for r in rows) or 1.0
    result: List[Dict[str, Any]] = []
    for row in rows:
        augmented = dict(row)
        augmented["time_share"] = float(row["cuda_time_us"]) / total
        result.append(augmented)
    return result


@torch.inference_mode()
def collect_shape_census(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    output_tokens: int,
    *,
    decode_profiled_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Collect the module + operator shape census for one workload.

    Runs one full prefill forward (hook + profiler) and the *full* decode of
    ``output_tokens - 1`` steps (hook only), so the module ``call_count`` is
    the true runtime count (prefill 1, decode G-1). The PyTorch profiler only
    covers the first ``decode_profiled_steps`` decode steps, which is enough
    to obtain the per-step operator timing share without replaying the whole
    generation under profiling overhead.

    Args:
        model: HF causal LM in eval mode on the target device.
        inputs: ``input_ids``/``attention_mask`` tensors of shape ``(1, ISL)``.
        output_tokens: Configured OSL (G). Must be >= 1.
        decode_profiled_steps: Number of decode steps covered by the profiler
            (defaults to ``min(output_tokens - 1, 4)``, >= 1). The remaining
            decode steps still run under the module hook for a full call-count
            census.

    Returns:
        A dict with ``modules`` (module census), ``prefill_operators`` /
        ``decode_operators`` (operator tables with ``time_share``), and the
        measured ``decode_steps`` (full = G-1) / ``decode_steps_profiled`` /
        ``input_len`` / ``output_tokens``.
    """
    if output_tokens < 1:
        raise ValueError(f"output_tokens must be >= 1, got {output_tokens}")

    full_decode_steps = output_tokens - 1
    if decode_profiled_steps is None:
        decode_profiled_steps = min(full_decode_steps, 4)
    decode_profiled_steps = max(decode_profiled_steps, 1)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    device = input_ids.device
    input_len = input_ids.shape[1]

    collector = ShapeCensusCollector()
    collector.attach(model)

    # ── Prefill phase (hook + profiler) ────────────────────────────
    collector.set_phase("prefill")
    profiler_prefill = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        # Memory tracking is E02-05's concern and inflates profiler memory on
        # the 8 GiB device; the census only needs shapes + device time.
        profile_memory=False,
        with_stack=False,
    )
    profiler_prefill.start()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    if device.type == "cuda":
        torch.cuda.synchronize()
    profiler_prefill.stop()
    prefill_operators = _with_time_share(extract_operator_table(profiler_prefill))
    # Release the profiler's CUDA buffers before the (long) decode loop so the
    # prefill profiling footprint does not linger across G-1 decode steps.
    del profiler_prefill

    # ── Decode phase (hook for all G-1 steps; profiler for the first few) ──
    collector.set_phase("decode")
    profiler_decode = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    )
    profiler_decode.start()
    decode_operators: Optional[List[Dict[str, Any]]] = None
    current_length = input_len
    for step in range(1, output_tokens):
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

        if step == decode_profiled_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            profiler_decode.stop()
            decode_operators = _with_time_share(extract_operator_table(profiler_decode))
            del profiler_decode

    # If the profiler was never stopped inside the loop (decode shorter than
    # the profiled-step window), stop it now.
    if decode_operators is None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        profiler_decode.stop()
        decode_operators = _with_time_share(extract_operator_table(profiler_decode))

    collector.detach()

    return {
        "input_len": input_len,
        "output_tokens": output_tokens,
        "decode_steps": full_decode_steps,
        "decode_steps_profiled": min(decode_profiled_steps, full_decode_steps),
        "modules": collector.records(),
        "prefill_operators": prefill_operators,
        "decode_operators": decode_operators,
    }


__all__ = [
    "ShapeCensusCollector",
    "collect_shape_census",
]
