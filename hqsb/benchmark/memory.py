"""Memory and KV-cache accounting for benchmark results.

Provides pure accounting functions (KV-cache size estimation, model weight
byte counting) that are unit-testable, plus thin helpers that read process
memory (RSS/swap) from ``/proc`` so they work on Jetson without ``psutil``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class KvCacheInfo:
    """Shape and byte accounting for a transformer KV cache.

    Attributes:
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV heads (GQA: fewer than Q heads).
        head_dim: Per-head dimension.
        context_length: Current sequence length in tokens.
        element_bytes: Bytes per element (dtype-dependent).
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    context_length: int
    element_bytes: int

    def per_token_bytes(self) -> int:
        """Bytes per token (K + V, all layers)."""
        # K and V each: layers * kv_heads * head_dim * bytes.
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.element_bytes

    def total_bytes(self) -> int:
        """Total KV cache bytes for the current context length."""
        return self.per_token_bytes() * self.context_length


def compute_kv_cache_info(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_length: int,
    dtype_bytes: int,
) -> KvCacheInfo:
    """Construct a :class:`KvCacheInfo` with validation.

    Raises:
        ValueError: If any structural dimension is non-positive.
    """
    if min(num_layers, num_kv_heads, head_dim, context_length, dtype_bytes) < 1:
        raise ValueError(
            "kv-cache dimensions must be positive: "
            f"layers={num_layers} kv_heads={num_kv_heads} "
            f"head_dim={head_dim} ctx={context_length} bytes={dtype_bytes}"
        )
    return KvCacheInfo(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        context_length=context_length,
        element_bytes=dtype_bytes,
    )


def model_weight_bytes(model: Any) -> int:
    """Return the total byte size of a model's parameters.

    Args:
        model: A PyTorch module (or any object with ``parameters()``).
    """
    total = 0
    for parameter in model.parameters():
        total += parameter.numel() * parameter.element_size()
    return total


def _read_proc_field(field: str) -> Optional[int]:
    """Read a numeric field from ``/proc/self/status`` (bytes when kB-based).

    Returns None if the field is absent or unparsable.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    # "VmRSS:\t  123456 kB" -> 123456 (kB)
                    value = line.split(":", 1)[1].strip().split()[0]
                    return int(value)
    except (OSError, ValueError, IndexError):
        return None
    return None


def process_rss_bytes() -> int:
    """Return the process resident set size in bytes (0 if unavailable).

    ``VmRSS`` in ``/proc/self/status`` is in kB.
    """
    kb = _read_proc_field("VmRSS")
    return (kb or 0) * 1024


def process_swap_bytes() -> int:
    """Return the process swap usage in bytes (0 if unavailable).

    ``VmSwap`` in ``/proc/self/status`` is in kB.
    """
    kb = _read_proc_field("VmSwap")
    return (kb or 0) * 1024


def cuda_memory_snapshot() -> Dict[str, float]:
    """Return current/peak CUDA allocated & reserved memory in MiB.

    Returns zeros when CUDA is unavailable (CPU-only environment).
    """
    import torch

    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
        }
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
    }


def model_kv_cache_config(model: Any) -> Dict[str, int]:
    """Extract KV-cache structural parameters from a HF causal LM config.

    Handles both direct config fields and the nested ``text_config`` layout
    used by some composite (e.g. vision-language) models. Returns an empty
    dict when the parameters cannot be determined.
    """
    config = getattr(model, "config", None)
    if config is None:
        return {}

    def _field(names, default=None):
        for n in names:
            if hasattr(config, n):
                return getattr(config, n)
        return default

    num_layers = _field(("num_hidden_layers", "num_layers", "n_layer"))
    num_heads = _field(("num_key_value_heads", "n_kv_heads", "num_attention_heads", "n_head"))
    head_dim = _field(("head_dim",))
    hidden = _field(("hidden_size", "n_embd"))
    num_attn_heads = _field(("num_attention_heads", "n_head"))

    # head_dim may be derived from hidden_size / num_attention_heads.
    if head_dim is None and hidden and num_attn_heads:
        head_dim = hidden // num_attn_heads

    result: Dict[str, int] = {}
    if num_layers is not None:
        result["num_layers"] = int(num_layers)
    if num_heads is not None:
        result["num_kv_heads"] = int(num_heads)
    if head_dim is not None:
        result["head_dim"] = int(head_dim)
    return result


__all__ = [
    "KvCacheInfo",
    "compute_kv_cache_info",
    "cuda_memory_snapshot",
    "model_kv_cache_config",
    "model_weight_bytes",
    "process_rss_bytes",
    "process_swap_bytes",
]
