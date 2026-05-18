"""Predictive, unit-explicit memory model for E02-05.

E02-05 answers: *where does Qwen3-1.7B FP16 inference actually spend memory,
how does that grow with context and batch, and why do analytic values differ
from what the device reports?* (see
``docs/stage_experiments/details/S02/E02-05_memory_model_validation.md``.)

This module keeps the *predictor* side pure and unit-testable:

* :class:`KvLedger` / :func:`kv_ledger` — the KV formula using **KV heads**
  (``Hkv``), never query heads (``Hq``), plus explicit ``filled`` vs
  ``capacity`` context lengths (protocol §4/§4.1).
* :func:`eager_attention_workspace_bytes` / :func:`activation_lower_bound_bytes`
  — conditional peak terms, documented as a *model*, not a measurement.
* :func:`tensor_inventory` / :func:`storage_groups` / :func:`storage_dedup_summary`
  — parameter/buffer ledger with **storage-level de-duplication**, so tied
  weights (``model.embed_tokens.weight`` vs ``lm_head.weight``) are never
  counted twice and a shared-storage alias is reported as an alias group
  (protocol §3.1).
* :func:`kv_cache_metadata` — per-layer KV metadata read directly from the live
  cache (protocol §8 step 5), so KV bytes come from the *object*, not from a
  slope guess over total device memory.
* :func:`memory_snapshot` / :class:`DeviceMemorySampler` — staged snapshots and
  an external high-water-mark sampler (protocol §8 steps 3/4: allocator
  high-water and external sampling are different resolutions).
* :func:`decompose` — prediction/observation split with an explicit
  ``unresolved_bytes`` bucket. The protocol forbids closing the books by
  inventing a "system overhead" residual.

Unit conventions used throughout (protocol §2): ``B`` = 1 byte, ``KiB``/``MiB``/
``GiB`` = powers of 1024, ``KB``/``MB``/``GB`` = powers of 1000. Every size is
stored in **bytes**; only presentation converts, and the unit is always named.
"""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from hqsb.benchmark.memory import (
    cuda_memory_snapshot,
    device_memory_snapshot,
    host_memory_snapshot,
    model_kv_cache_config,
    process_rss_bytes,
    process_swap_bytes,
)
from hqsb.benchmark.metrics import percentile

KIB = 1024
MIB = 1024**2
GIB = 1024**3
KB = 1000
MB = 1000**2
GB = 1000**3

__all__ = [
    "GIB",
    "KB",
    "KIB",
    "MB",
    "MIB",
    "GB",
    "DeviceMemorySampler",
    "KvLedger",
    "activation_lower_bound_bytes",
    "bytes_to_decimal_mb",
    "bytes_to_gib",
    "bytes_to_mib",
    "decompose",
    "eager_attention_workspace_bytes",
    "format_bytes",
    "kv_cache_metadata",
    "kv_ledger",
    "kv_ledger_from_model",
    "memory_snapshot",
    "model_memory_inventory",
    "storage_dedup_summary",
    "storage_groups",
    "summarize_samples",
    "tensor_inventory",
]


# ───────────────────────────── units ─────────────────────────────


def bytes_to_mib(nbytes: float) -> float:
    """Convert bytes to **MiB** (binary, 2^20)."""
    return float(nbytes) / MIB


def bytes_to_decimal_mb(nbytes: float) -> float:
    """Convert bytes to **MB** (decimal, 10^6).

    Kept separate from :func:`bytes_to_mib` on purpose: protocol §8 step 8
    makes "MiB treated as MB" a detectable anti-example, so the two
    conversions must never collapse into one helper.
    """
    return float(nbytes) / MB


def bytes_to_gib(nbytes: float) -> float:
    """Convert bytes to **GiB** (binary, 2^30)."""
    return float(nbytes) / GIB


def format_bytes(nbytes: float) -> str:
    """Render a size with both binary and decimal units named explicitly."""
    n = float(nbytes)
    return f"{n:.0f} B ({n / MIB:.3f} MiB / {n / MB:.3f} MB)"


# ───────────────────────────── KV ledger ─────────────────────────────


@dataclass(frozen=True)
class KvLedger:
    """KV-cache byte accounting for a dense decoder with GQA.

    ``context_length`` is always an explicit choice of ``filled``
    (``I + G - 1`` after generation) or ``capacity`` (``I + G`` / static
    preallocation). The two must never be conflated (protocol §4.1).

    ``element_bytes`` is the *cache* element size, not the weight dtype's.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    element_bytes: int
    batch_size: int
    context_length: int

    @property
    def per_token_per_layer_bytes(self) -> int:
        """``2 (K+V) * Hkv * Dh * bytes`` — one token, one layer."""
        return 2 * self.num_kv_heads * self.head_dim * self.element_bytes

    @property
    def per_token_all_layers_bytes(self) -> int:
        """Bytes added by one more token across **all** layers."""
        return self.num_layers * self.per_token_per_layer_bytes

    @property
    def per_sequence_bytes(self) -> int:
        return self.per_token_all_layers_bytes * self.context_length

    @property
    def total_bytes(self) -> int:
        """Logical persistent KV bytes for ``batch_size`` sequences."""
        return self.batch_size * self.per_sequence_bytes

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "element_bytes": self.element_bytes,
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "per_token_per_layer_bytes": self.per_token_per_layer_bytes,
            "per_token_all_layers_bytes": self.per_token_all_layers_bytes,
            "per_sequence_bytes": self.per_sequence_bytes,
            "total_bytes": self.total_bytes,
            "total_mib": bytes_to_mib(self.total_bytes),
            "total_mb": bytes_to_decimal_mb(self.total_bytes),
        }


def kv_ledger(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    element_bytes: int,
    batch_size: int,
    context_length: int,
) -> KvLedger:
    """Build a :class:`KvLedger`, rejecting non-positive dimensions.

    Raises:
        ValueError: If any dimension is < 1 (a zero KV head count or a zero
            context would otherwise silently produce a plausible 0-byte
            prediction).
    """
    if min(
        num_layers, num_kv_heads, head_dim, element_bytes, batch_size, context_length
    ) < 1:
        raise ValueError(
            "kv ledger dimensions must be positive: "
            f"L={num_layers} Hkv={num_kv_heads} Dh={head_dim} "
            f"bytes={element_bytes} B={batch_size} T={context_length}"
        )
    return KvLedger(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        element_bytes=element_bytes,
        batch_size=batch_size,
        context_length=context_length,
    )


def kv_ledger_from_model(
    model: Any,
    *,
    batch_size: int,
    context_length: int,
    element_bytes: int = 2,
) -> KvLedger:
    """Read the structural KV params off ``model.config`` and build a ledger."""
    config = model_kv_cache_config(model)
    required = {"num_layers", "num_kv_heads", "head_dim"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"model config missing KV params: {sorted(missing)}")
    return kv_ledger(
        num_layers=config["num_layers"],
        num_kv_heads=config["num_kv_heads"],
        head_dim=config["head_dim"],
        element_bytes=element_bytes,
        batch_size=batch_size,
        context_length=context_length,
    )


# ─────────────────── activation / workspace (model, not measurement) ───────────


def eager_attention_workspace_bytes(
    *,
    batch_size: int,
    num_query_heads: int,
    seq_len: int,
    score_bytes: int = 2,
    softmax_bytes: int = 4,
) -> Dict[str, int]:
    """Conditional eager-attention peak terms for ``[B, Hq, S, S]`` matrices.

    The reference backend uses eager attention, which materializes the
    attention score matrix *and* an fp32 softmax buffer of the same
    ``[B, Hq, S, S]`` shape. A fused attention (sdpa/flash) does not, so this
    is a *model of the current frozen implementation*, never a general
    activation law (protocol §5).

    Note the deliberate use of ``num_query_heads`` (``Hq``): unlike the
    persistent KV cache, the score matrix genuinely has one row per query
    head. Mixing this up with ``Hkv`` is the exact confusion protocol §4 warns
    about in the opposite direction.
    """
    if min(batch_size, num_query_heads, seq_len, score_bytes, softmax_bytes) < 1:
        raise ValueError("eager workspace dims must be positive")
    cells = batch_size * num_query_heads * seq_len * seq_len
    score = cells * score_bytes
    softmax = cells * softmax_bytes
    return {
        "score_cells": cells,
        "score_matrix_bytes": score,
        "softmax_matrix_bytes": softmax,
        "total_bytes": score + softmax,
    }


def activation_lower_bound_bytes(
    *,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    element_bytes: int = 2,
) -> int:
    """Lower bound on one layer's live activations: ``B*S*(hidden + FFN)``.

    This is deliberately a bound, not a sum over all layers: layers are not
    simultaneously live, so adding ``L`` copies would over-predict
    (``Mpeak = max_t[...]``, protocol §5).
    """
    if min(batch_size, seq_len, hidden_size, intermediate_size, element_bytes) < 1:
        raise ValueError("activation dims must be positive")
    return batch_size * seq_len * (hidden_size + intermediate_size) * element_bytes


# ─────────────────── parameter / storage inventory ───────────────────


def _tensor_record(name: str, tensor: torch.Tensor, kind: str) -> Dict[str, Any]:
    """Describe one parameter/buffer including its backing storage identity.

    ``storage_ptr`` is recorded as *auxiliary* evidence only: a raw address is
    not a stable cross-run identity (protocol §3.1). Stable identity comes from
    the tensor name plus the alias group it lands in
    (:func:`storage_groups`), so alias relationships survive a re-run that
    happens to place the weights elsewhere.
    """
    try:
        storage = tensor.untyped_storage()
        storage_ptr = int(storage.data_ptr())
        storage_nbytes = int(storage.nbytes())
    except (RuntimeError, AttributeError):  # pragma: no cover - exotic tensors
        storage_ptr, storage_nbytes = 0, 0

    numel = int(tensor.numel())
    element_bytes = int(tensor.element_size())
    logical = numel * element_bytes
    return {
        "name": name,
        "kind": kind,
        "shape": [int(s) for s in tensor.shape],
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "numel": numel,
        "element_bytes": element_bytes,
        "logical_bytes": logical,
        "storage_ptr": storage_ptr,
        "storage_nbytes": storage_nbytes,
        "storage_offset": int(tensor.storage_offset()),
        # >1.0 means this tensor is a view into a larger allocation (i.e. the
        # backing storage is preallocated beyond the logical element count).
        "storage_over_logical": (
            storage_nbytes / logical if logical > 0 else 0.0
        ),
    }


def tensor_inventory(model: Any) -> Dict[str, Any]:
    """Enumerate parameters and buffers **including tied duplicates**.

    ``named_parameters()`` defaults to ``remove_duplicate=True``, which hides
    the tied ``lm_head.weight`` behind ``model.embed_tokens.weight`` and makes
    a "no duplicates" conclusion unfalsifiable. This function asks for
    ``remove_duplicate=False`` so the tie is *visible* and then removes it by
    storage de-duplication instead of by omission.
    """
    parameters = [
        _tensor_record(name, tensor, "parameter")
        for name, tensor in model.named_parameters(remove_duplicate=False)
    ]
    buffers = [
        _tensor_record(name, tensor, "buffer")
        for name, tensor in model.named_buffers(remove_duplicate=False)
    ]
    return {
        "parameters": parameters,
        "buffers": buffers,
        "parameter_summary": storage_dedup_summary(parameters),
        "buffer_summary": storage_dedup_summary(buffers),
    }


def storage_groups(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group tensor records by backing storage, labelling groups stably.

    Group ids (``storage_0`` …) are assigned in first-seen order and the member
    names are sorted, so the *alias relationship* — not a memory address — is
    the reproducible identity. A group with more than one member is an alias
    group (the tied-embedding case).
    """
    groups: "Dict[Tuple[Any, ...], Dict[str, Any]]" = {}
    for record in records:
        if record["storage_nbytes"] > 0:
            key: Tuple[Any, ...] = (record["storage_ptr"], record["storage_nbytes"])
        else:
            # Zero-size/exotic storage: fall back to the tensor's own name so
            # unrelated empty storages are not merged into one alias group.
            key = ("unique", record["name"])
        group = groups.get(key)
        if group is None:
            group = {
                "member_names": [],
                "storage_nbytes": record["storage_nbytes"],
                "storage_ptr": record["storage_ptr"],
                "storage_offset": record["storage_offset"],
            }
            groups[key] = group
        group["member_names"].append(record["name"])

    result: List[Dict[str, Any]] = []
    for index, group in enumerate(groups.values()):
        members = sorted(group["member_names"])
        result.append(
            {
                "group_id": f"storage_{index}",
                "member_names": members,
                "num_members": len(members),
                "storage_nbytes": group["storage_nbytes"],
                "storage_mib": bytes_to_mib(group["storage_nbytes"]),
                "storage_ptr": group["storage_ptr"],
                "storage_offset": group["storage_offset"],
                "is_alias_group": len(members) > 1,
            }
        )
    return result


def storage_dedup_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Logical vs de-duplicated byte totals plus the alias groups.

    ``logical_total_bytes`` double-counts a tied tensor; ``dedup_total_bytes``
    counts each backing storage once. The difference is exactly the
    duplicate-counting error a naive ``numel * element_size`` sum would make.
    """
    groups = storage_groups(records)
    logical_total = sum(int(r["logical_bytes"]) for r in records)
    dedup_total = sum(int(g["storage_nbytes"]) for g in groups)
    alias_groups = [g for g in groups if g["is_alias_group"]]
    dtypes: Dict[str, int] = {}
    devices: Dict[str, int] = {}
    for record in records:
        dtypes[record["dtype"]] = dtypes.get(record["dtype"], 0) + 1
        devices[record["device"]] = devices.get(record["device"], 0) + 1
    return {
        "num_tensors": len(records),
        "num_unique_storages": len(groups),
        "logical_total_bytes": logical_total,
        "logical_total_mib": bytes_to_mib(logical_total),
        "dedup_total_bytes": dedup_total,
        "dedup_total_mib": bytes_to_mib(dedup_total),
        "duplicate_counted_bytes": logical_total - dedup_total,
        "alias_group_count": len(alias_groups),
        "alias_groups": alias_groups,
        "dtype_histogram": dtypes,
        "device_histogram": devices,
        "total_numel": sum(int(r["numel"]) for r in records),
    }


def model_memory_inventory(model: Any) -> Dict[str, Any]:
    """Inventory parameters + buffers and the resident (de-duplicated) total.

    ``resident_bytes`` is the analytically expected *weight-side* device
    footprint of the frozen FP16 model: unique parameter storages plus unique
    buffer storages. Non-FP16 buffers (e.g. an fp32 RoPE ``inv_freq``) are
    counted by their real dtype, not assumed to be FP16.
    """
    inventory = tensor_inventory(model)
    param_summary = inventory["parameter_summary"]
    buffer_summary = inventory["buffer_summary"]
    resident = param_summary["dedup_total_bytes"] + buffer_summary["dedup_total_bytes"]
    config = model_kv_cache_config(model)
    return {
        **inventory,
        "config_kv": config,
        "resident_bytes": resident,
        "resident_mib": bytes_to_mib(resident),
        "resident_gib": bytes_to_gib(resident),
        "weight_approx_bytes_1p7b_fp16": int(1.7e9) * 2,
    }


# ─────────────────────────── KV cache metadata ───────────────────────────


def _iter_kv_pairs(cache: Any):
    """Yield ``(layer_index, key_tensor, value_tensor)`` across HF cache APIs.

    Three shapes are supported because the transformers KV-cache API has
    changed repeatedly: the legacy ``DynamicCache`` (``key_cache`` /
    ``value_cache`` lists), the ``Cache`` with ``layers[i].keys/values``, and
    the ancient tuple-of-tuples. The chosen path is reported by the caller so
    the evidence never silently reads an empty cache.
    """
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if isinstance(key_cache, (list, tuple)) and isinstance(value_cache, (list, tuple)):
        if key_cache:
            for index, (key, value) in enumerate(zip(key_cache, value_cache)):
                if key is not None and value is not None:
                    yield index, key, value
            return

    layers = getattr(cache, "layers", None)
    if isinstance(layers, (list, tuple)) and layers:
        for index, layer in enumerate(layers):
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None:
                key = getattr(layer, "key_cache", None)
            if value is None:
                value = getattr(layer, "value_cache", None)
            if key is not None and value is not None:
                yield index, key, value
        return

    if isinstance(cache, (list, tuple)):
        for index, item in enumerate(cache):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                yield index, item[0], item[1]


def kv_cache_metadata(cache: Any) -> Dict[str, Any]:
    """Per-layer KV metadata read from the **live** cache object.

    This is the protocol §8 step 5 requirement: KV bytes come from the cache's
    own shapes/storages rather than being inferred from the slope of total
    device memory. ``storage_over_logical`` exposes a view into a larger
    preallocated buffer, i.e. ``Tcapacity`` larger than ``Tfilled``.
    """
    layers: List[Dict[str, Any]] = []
    unique_storages: Dict[int, int] = {}
    source = "none"

    key_attr = getattr(cache, "key_cache", None)
    if isinstance(key_attr, (list, tuple)) and key_attr:
        source = "key_cache_value_cache_lists"
    elif getattr(cache, "layers", None):
        source = "cache_layers_keys_values"
    elif isinstance(cache, (list, tuple)) and cache:
        source = "tuple_of_tuples"

    for index, key, value in _iter_kv_pairs(cache):
        key_numel = int(key.numel())
        value_numel = int(value.numel())
        key_bytes = key_numel * int(key.element_size())
        value_bytes = value_numel * int(value.element_size())
        logical = key_bytes + value_bytes

        storages: Dict[int, int] = {}
        for tensor in (key, value):
            try:
                storage = tensor.untyped_storage()
                storages[int(storage.data_ptr())] = int(storage.nbytes())
            except (RuntimeError, AttributeError):  # pragma: no cover
                continue
        unique_storages.update(storages)
        storage_bytes = sum(storages.values())

        layers.append(
            {
                "layer_index": index,
                "key_shape": [int(s) for s in key.shape],
                "value_shape": [int(s) for s in value.shape],
                "key_dtype": str(key.dtype).removeprefix("torch."),
                "value_dtype": str(value.dtype).removeprefix("torch."),
                "element_bytes": int(key.element_size()),
                "key_contiguous": bool(key.is_contiguous()),
                "value_contiguous": bool(value.is_contiguous()),
                "key_stride": [int(s) for s in key.stride()],
                "value_stride": [int(s) for s in value.stride()],
                "logical_bytes": logical,
                "storage_bytes": storage_bytes,
                "storage_over_logical": (
                    storage_bytes / logical if logical > 0 else 0.0
                ),
                "context_length": (
                    int(key.shape[-2]) if key.dim() >= 2 else None
                ),
            }
        )

    total_logical = sum(int(layer["logical_bytes"]) for layer in layers)
    context_lengths = sorted(
        {
            int(layer["context_length"])
            for layer in layers
            if layer["context_length"] is not None
        }
    )
    head_dims = sorted({int(layer["key_shape"][-1]) for layer in layers if layer["key_shape"]})
    kv_heads = sorted(
        {int(layer["key_shape"][-3]) for layer in layers if len(layer["key_shape"]) >= 3}
    )
    element_bytes = sorted({int(layer["element_bytes"]) for layer in layers})
    per_token_per_layer = None
    if len(layers) == 1 or (kv_heads and head_dims and len(element_bytes) == 1):
        # Only a structural read: 2 (K+V) * Hkv * Dh * dtype bytes per token.
        if layers:
            per_token_per_layer = (
                2 * int(layers[0]["key_shape"][-3]) * int(layers[0]["key_shape"][-1])
                * int(layers[0]["element_bytes"])
            )

    return {
        "source_path": source,
        "num_layers": len(layers),
        "layers": layers,
        "total_logical_bytes": total_logical,
        "total_logical_mib": bytes_to_mib(total_logical),
        "total_unique_storage_bytes": sum(unique_storages.values()),
        "num_unique_storages": len(unique_storages),
        "kv_heads_observed": kv_heads,
        "head_dims_observed": head_dims,
        "element_bytes_observed": element_bytes,
        "per_token_per_layer_bytes_observed": per_token_per_layer,
        "context_lengths_observed": context_lengths,
        "context_filled": max(context_lengths) if context_lengths else None,
        "appears_preallocated": any(
            float(layer["storage_over_logical"]) > 1.001 for layer in layers
        ),
    }


# ───────────────────────── memory snapshots ─────────────────────────


def memory_snapshot(*, label: str = "", with_cuda: bool = True) -> Dict[str, Any]:
    """One staged memory snapshot (protocol §8 step 3).

    ``with_cuda=False`` is for the *pre-context* M0 point: calling any CUDA API
    initializes the context, which would destroy the very baseline M0 exists to
    measure. Allocator stats are recorded alongside device free/total because
    on a unified-memory device under
    ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` the allocator counters may be all
    zero — that has to be visible in the artifact rather than silently used as
    "0 bytes of weights".
    """
    if with_cuda:
        cuda = cuda_memory_snapshot()
        device = device_memory_snapshot()
    else:
        cuda = {
            "allocated_mb": None,
            "reserved_mb": None,
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
        }
        device = {"total_mb": None, "free_mb": None, "is_unified": None}

    used_mb = None
    if device.get("free_mb") is not None and device.get("total_mb") is not None:
        used_mb = device["total_mb"] - device["free_mb"]

    return {
        "label": label,
        "cuda": cuda,
        "device": device,
        "device_used_mb": used_mb,
        "host": host_memory_snapshot(),
        "process_rss_bytes": process_rss_bytes(),
        "process_swap_bytes": process_swap_bytes(),
        "monotonic_ns": time.monotonic_ns(),
    }


def snapshot_used_bytes(snapshot: Dict[str, Any]) -> Optional[int]:
    """Device-used bytes of a snapshot, or ``None`` when unavailable."""
    used_mb = snapshot.get("device_used_mb")
    return None if used_mb is None else int(round(used_mb * MIB))


def summarize_samples(values: Sequence[float]) -> Dict[str, float]:
    """Summarize an external sampler series (protocol §8 step 4).

    ``max`` is the high-water mark; ``p95`` exists because a single 10 ms
    polling spike may miss a shorter peak, so the report must state which
    resolution produced the number. Empty input yields zeros with
    ``count=0`` (never a fabricated peak).
    """
    clean = [float(v) for v in values]
    if not clean:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 0.50),
        "p95": percentile(clean, 0.95),
    }


class DeviceMemorySampler:
    """Poll device used-memory from a background thread.

    An allocator high-water mark and an external sampler are different
    resolutions (protocol §8 step 4); this is the external one, used to catch a
    prefill peak that a single before/after snapshot would miss.
    """

    def __init__(self, interval_s: float = 0.01) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be > 0, got {interval_s}")
        self.interval_s = float(interval_s)
        self._samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                self._samples.append((total_bytes - free_bytes) / MIB)
            except Exception:  # noqa: BLE001 - never kill the measured phase
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def reset(self) -> None:
        self._samples = []

    @property
    def samples(self) -> List[float]:
        return list(self._samples)

    def summary(self) -> Dict[str, float]:
        return summarize_samples(self._samples)


def collect_garbage() -> None:
    """Force a Python GC pass before a snapshot (tensor frees are refcounted,
    but cycles and cached objects still need a collect)."""
    gc.collect()


# ─────────────────────── prediction vs observation ───────────────────────


def decompose(
    *,
    predicted: Dict[str, int],
    observed_bytes: Optional[int],
    explained: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Split ``observed - predicted`` into named, evidence-backed buckets.

    ``explained`` holds *measured* or structurally-derived quantities (attention
    workspace, context/library overhead). Whatever is left is reported as
    ``unresolved_bytes`` — the protocol forbids forcing the books to close by
    relabelling it as generic "allocator overhead".
    """
    predicted_total = int(sum(int(v) for v in predicted.values()))
    explained = explained or {}
    explained_total = int(sum(int(v) for v in explained.values()))

    result: Dict[str, Any] = {
        "predicted_components": {k: int(v) for k, v in predicted.items()},
        "predicted_total_bytes": predicted_total,
        "predicted_total_mib": bytes_to_mib(predicted_total),
        "explained_components": {k: int(v) for k, v in explained.items()},
        "explained_total_bytes": explained_total,
        "explained_total_mib": bytes_to_mib(explained_total),
        "observed_bytes": observed_bytes,
        "observed_mib": (
            bytes_to_mib(observed_bytes) if observed_bytes is not None else None
        ),
    }

    if observed_bytes is None:
        result.update(
            {
                "residual_bytes": None,
                "residual_mib": None,
                "residual_ratio": None,
                "unresolved_bytes": None,
                "unresolved_mib": None,
                "unresolved_ratio": None,
            }
        )
        return result

    residual = int(observed_bytes) - predicted_total
    unresolved = residual - explained_total
    result.update(
        {
            "residual_bytes": residual,
            "residual_mib": bytes_to_mib(residual),
            "residual_ratio": (
                residual / observed_bytes if observed_bytes else None
            ),
            "unresolved_bytes": unresolved,
            "unresolved_mib": bytes_to_mib(unresolved),
            "unresolved_ratio": (
                unresolved / observed_bytes if observed_bytes else None
            ),
        }
    )
    return result
