"""Memory and KV-cache accounting for benchmark results.

Provides pure accounting functions (KV-cache size estimation, model weight
byte counting) that are unit-testable, plus thin helpers that read process
memory (RSS/swap) from ``/proc`` so they work on Jetson without ``psutil``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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


def _read_meminfo_field(field: str) -> Optional[int]:
    """Read a kB-valued field from ``/proc/meminfo`` and return it in bytes.

    Returns None when the field is absent or unparsable.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    # "MemAvailable:    4462184 kB"
                    return int(line.split(":", 1)[1].strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


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


def host_memory_snapshot() -> Dict[str, float]:
    """Return host DRAM total/available in MiB from ``/proc/meminfo``.

    Returns zeros when ``/proc/meminfo`` is unavailable (non-Linux).
    """
    total = _read_meminfo_field("MemTotal") or 0
    available = _read_meminfo_field("MemAvailable") or 0
    return {
        "total_mb": total / (1024**2),
        "available_mb": available / (1024**2),
    }


def device_memory_snapshot() -> Dict[str, float]:
    """Return CUDA device memory total/free in MiB plus a unified-memory flag.

    ``is_unified`` is True when the GPU has no dedicated VRAM and ``total`` is
    the *whole system DRAM* shared with the CPU. Budget decisions on such
    devices must be derived from measured free memory, never from a hardcoded
    constant.

    Detection prefers ``cudaDeviceProp::integrated``, but many PyTorch builds
    (including the Jetson 2.5 wheel) do **not** expose that attribute, so we
    fall back to comparing the device total against host ``MemTotal``: they
    match on unified-memory parts and differ by orders of magnitude on
    discrete GPUs.

    Returns zeros when CUDA is unavailable.
    """
    import torch

    if not torch.cuda.is_available():
        return {"total_mb": 0.0, "free_mb": 0.0, "is_unified": False}

    free_bytes, total_bytes = torch.cuda.mem_get_info()

    integrated: Optional[bool] = None
    try:
        integrated = bool(torch.cuda.get_device_properties(0).integrated)
    except AttributeError:
        integrated = None  # not exposed by this PyTorch build

    if integrated is None:
        host_total = _read_meminfo_field("MemTotal")
        if host_total:
            # Within 2% of system DRAM => the GPU is sharing it.
            integrated = abs(total_bytes - host_total) <= 0.02 * host_total
        else:
            integrated = False

    return {
        "total_mb": total_bytes / (1024**2),
        "free_mb": free_bytes / (1024**2),
        "is_unified": bool(integrated),
    }


# Fraction of free memory always kept back for activations, KV cache, CUDA
# context and the allocator's own fragmentation.
DEFAULT_RESERVE_RATIO = 0.15
# Absolute floor for that reserve, so tiny devices keep a sane margin.
DEFAULT_RESERVE_MB = 512.0


def memory_budget_bytes(
    free_bytes: int,
    *,
    reserve_ratio: float = DEFAULT_RESERVE_RATIO,
    reserve_mb: float = DEFAULT_RESERVE_MB,
) -> int:
    """Derive a safe placement budget (bytes) from measured free memory.

    The budget is the smaller of two guards, so both the proportional and the
    absolute reserve are honored::

        budget = min(free - reserve_mb, free * (1 - reserve_ratio))

    clamped to ``[0, free]``. This replaces hardcoded budgets (e.g. ``6GB``)
    which overshoot real headroom on unified-memory devices where ``free`` is
    only a fraction of ``total``.

    Raises:
        ValueError: If ``free_bytes`` is negative or ``reserve_ratio`` is
            outside ``[0, 1)``.
    """
    if free_bytes < 0:
        raise ValueError(f"free_bytes must be non-negative, got {free_bytes}")
    if not (0.0 <= reserve_ratio < 1.0):
        raise ValueError(
            f"reserve_ratio must be in [0, 1), got {reserve_ratio}"
        )

    reserve_bytes = max(reserve_mb * (1024**2), free_bytes * reserve_ratio)
    budget = min(free_bytes - reserve_mb * (1024**2), free_bytes - reserve_bytes)
    return int(max(0, min(free_bytes, budget)))


def gpu_memory_budget_bytes(
    *,
    reserve_ratio: float = DEFAULT_RESERVE_RATIO,
    reserve_mb: float = DEFAULT_RESERVE_MB,
) -> int:
    """Return a safe GPU weight-placement budget in bytes (0 without CUDA)."""
    import torch

    if not torch.cuda.is_available():
        return 0
    free_bytes, _total = torch.cuda.mem_get_info()
    return memory_budget_bytes(
        free_bytes, reserve_ratio=reserve_ratio, reserve_mb=reserve_mb
    )


def host_memory_budget_bytes(
    *,
    reserve_ratio: float = DEFAULT_RESERVE_RATIO,
    reserve_mb: float = DEFAULT_RESERVE_MB,
) -> int:
    """Return a safe host (CPU offload) budget in bytes (0 if unknown)."""
    available = _read_meminfo_field("MemAvailable") or 0
    return memory_budget_bytes(
        available, reserve_ratio=reserve_ratio, reserve_mb=reserve_mb
    )


def top_memory_processes(limit: int = 10) -> List[Dict[str, Any]]:
    """Return the top processes by RSS, read from ``/proc`` (no psutil).

    Args:
        limit: Maximum number of processes to return.

    Returns:
        List of ``{"pid", "name", "rss_mb", "cmdline"}`` sorted by RSS desc.
        Unreadable/vanished processes are skipped.
    """
    entries: List[Dict[str, Any]] = []
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return entries

    for pid in pids:
        try:
            with open(f"/proc/{pid}/statm", encoding="utf-8") as fh:
                rss_pages = int(fh.read().split()[1])
            with open(f"/proc/{pid}/comm", encoding="utf-8") as fh:
                name = fh.read().strip()
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmdline = fh.read().decode("utf-8", "replace").replace("\x00", " ").strip()
        except (OSError, ValueError, IndexError):
            continue
        entries.append(
            {
                "pid": pid,
                "name": name,
                "rss_mb": rss_pages * os.sysconf("SC_PAGE_SIZE") / (1024**2),
                "cmdline": cmdline[:200],
            }
        )

    entries.sort(key=lambda item: item["rss_mb"], reverse=True)
    return entries[:limit]


def memory_diagnostics() -> Dict[str, Any]:
    """Return a reproducible breakdown of where device memory went.

    This is the evidence artifact for "why is only half the VRAM free?".
    On unified-memory devices the CUDA ``total`` *is* the system DRAM, so the
    answer is always some combination of process RSS, kernel overhead and
    page cache — never a CUDA-side quota.

    Returns:
        Dict with ``host`` / ``device`` / ``kernel_bytes`` / ``top_processes``
        / ``gpu_budget_bytes`` keys. All sizes are in MiB unless noted.
    """
    host = host_memory_snapshot()
    device = device_memory_snapshot()

    kernel_fields = {
        "slab_mb": ("Slab", "SReclaimable"),
        "cma_mb": ("CmaTotal",),
        "page_tables_mb": ("PageTables",),
        "kernel_stack_mb": ("KernelStack",),
    }
    kernel_bytes: Dict[str, float] = {}
    for key, fields in kernel_fields.items():
        total = 0
        for field in fields:
            total += _read_meminfo_field(field) or 0
        kernel_bytes[key] = total / (1024**2)

    budget = gpu_memory_budget_bytes()
    return {
        "host": host,
        "device": device,
        "kernel_bytes": kernel_bytes,
        "top_processes": top_memory_processes(),
        "gpu_budget_bytes": budget,
        "gpu_budget_mb": budget / (1024**2),
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
    "DEFAULT_RESERVE_MB",
    "DEFAULT_RESERVE_RATIO",
    "KvCacheInfo",
    "compute_kv_cache_info",
    "cuda_memory_snapshot",
    "device_memory_snapshot",
    "gpu_memory_budget_bytes",
    "host_memory_budget_bytes",
    "host_memory_snapshot",
    "memory_budget_bytes",
    "memory_diagnostics",
    "model_kv_cache_config",
    "model_weight_bytes",
    "process_rss_bytes",
    "process_swap_bytes",
    "top_memory_processes",
]


def _format_diagnostics(report: Dict[str, Any]) -> str:
    """Render :func:`memory_diagnostics` as a human-readable table."""
    device = report["device"]
    host = report["host"]
    kernel = report["kernel_bytes"]

    lines = ["HQSB memory diagnostics", "=" * 60]
    lines.append(
        f"device : total {device['total_mb']:.1f} MiB | "
        f"free {device['free_mb']:.1f} MiB | unified={device['is_unified']}"
    )
    lines.append(
        f"host   : total {host['total_mb']:.1f} MiB | "
        f"available {host['available_mb']:.1f} MiB"
    )
    lines.append(f"GPU weight budget: {report['gpu_budget_mb']:.1f} MiB")
    if device["is_unified"]:
        lines.append(
            "NOTE: unified memory -- device 'total' is the shared system DRAM, "
            "so a low 'free' is host-side pressure, not a CUDA quota."
        )
    lines.append("")
    lines.append("Kernel overhead (MiB):")
    for key, value in kernel.items():
        lines.append(f"  {key:<18s} {value:8.1f}")
    lines.append("")
    lines.append("Top processes by RSS:")
    for entry in report["top_processes"]:
        lines.append(
            f"  {entry['rss_mb']:8.1f} MiB  pid={entry['pid']:<7d} {entry['name']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(_format_diagnostics(memory_diagnostics()))
