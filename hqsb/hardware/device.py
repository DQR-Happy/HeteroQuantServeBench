"""Device-agnostic accelerator access for the CUDA and Ascend NPU backends.

Why this module exists
----------------------
HQSB runs the same Stage-2 experiments on NVIDIA (CUDA) and Ascend (NPU)
hardware.  Both expose a PyTorch-style accelerator API, but under different
namespaces (``torch.cuda`` vs ``torch.npu``) and with genuinely different
metric semantics.  Letting ``if device.type == "cuda"`` spread across twenty
modules made the non-CUDA path both untestable and invisible: a machine with no
CUDA silently took the CPU branch or crashed on ``torch.cuda.mem_get_info``.

What this module is **not**
---------------------------
It does not pretend the two devices are the same.  Where a metric has no
equivalent -- CUDA's ``reserved`` pool, for instance -- the accessor returns
``None`` rather than a substituted number, and the caller must handle that
explicitly.  That keeps "the backend does not expose this" distinguishable from
"the value is zero", which is exactly the difference a report must not blur.

CPU remains the fallback: it is a real, labelled backend, never a silent
downgrade of an accelerator run.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Optional, Tuple

from hqsb.core.errors import ConfigError

CUDA = "cuda"
NPU = "npu"
CPU = "cpu"

#: Accelerator kinds this module knows how to drive.
ACCELERATOR_KINDS: Tuple[str, ...] = (CUDA, NPU)

#: Pins the accelerator explicitly (``cuda`` / ``npu`` / ``cpu``).  Unset means
#: "whatever is available, CUDA first", which is the historical behaviour and
#: keeps existing Jetson runs unchanged.
DEVICE_ENV = "HQSB_DEVICE"


def _torch():
    """Import torch lazily; contracts stay independent of device SDKs."""
    import torch

    return torch


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def cuda_available() -> bool:
    """Whether a usable CUDA device is present."""
    if not _module_available("torch"):
        return False
    try:
        return bool(_torch().cuda.is_available())
    except Exception:
        return False


def npu_available() -> bool:
    """Whether torch_npu is importable *and* reports a usable NPU."""
    if not _module_available("torch_npu"):
        return False
    try:
        import torch_npu  # noqa: F401  (registers the torch.npu namespace)

        return bool(_torch().npu.is_available())
    except Exception:
        return False


def available_accelerators() -> Tuple[str, ...]:
    """Every accelerator actually usable here, in preference order."""
    found = []
    if cuda_available():
        found.append(CUDA)
    if npu_available():
        found.append(NPU)
    return tuple(found)


def accelerator_kind(device: Any) -> str:
    """Classify a device (or its string form) as ``cuda``/``npu``/``cpu``."""
    text = str(device)
    if text.startswith(CUDA):
        return CUDA
    if text.startswith(NPU):
        return NPU
    return CPU


def is_accelerator(device: Any) -> bool:
    return accelerator_kind(device) in ACCELERATOR_KINDS


def default_accelerator() -> str:
    """The accelerator to run on, honouring ``HQSB_DEVICE`` when it is set.

    An explicit request is honoured even when the device looks unavailable, so
    the failure surfaces at the allocation site with a real error instead of
    being silently redirected to CPU.
    """
    requested = os.environ.get(DEVICE_ENV, "").strip().lower()
    if requested:
        if requested not in (CUDA, NPU, CPU):
            raise ConfigError(
                f"{DEVICE_ENV}={requested!r} is not a known device; expected one of "
                f"{[CUDA, NPU, CPU]}",
                details={DEVICE_ENV: requested},
            )
        return requested
    found = available_accelerators()
    return found[0] if found else CPU


def resolve_device(prefer: str = "") -> Any:
    """Return the ``torch.device`` to run on.

    Args:
        prefer: Explicit kind (``"cuda"``/``"npu"``/``"cpu"``); empty means use
            :func:`default_accelerator`.
    """
    torch = _torch()
    kind = prefer.strip().lower() if prefer else default_accelerator()
    return torch.device(kind)


def accelerator_index(device: Any) -> int:
    """Device index, defaulting to 0 for a bare ``cuda``/``npu`` spec."""
    index = getattr(device, "index", None)
    return 0 if index is None else int(index)


def _backend_namespace(device: Any):
    """The torch namespace owning this device's accelerator API."""
    kind = accelerator_kind(device)
    if kind == CUDA:
        return _torch().cuda
    if kind == NPU:
        import torch_npu  # noqa: F401  (registers torch.npu)

        return _torch().npu
    return None


def synchronize(device: Any) -> None:
    """Block until queued work on ``device`` completes.

    This is the timing boundary every Stage-2 measurement depends on: without
    it an asynchronous launch would be timed as if it were the computation.
    CPU has no queue, so this is a no-op there.
    """
    backend = _backend_namespace(device)
    if backend is not None:
        backend.synchronize()


def empty_cache(device: Any) -> None:
    """Release cached (unused) device memory back to the allocator."""
    backend = _backend_namespace(device)
    empty_cache_fn = getattr(backend, "empty_cache", None) if backend is not None else None
    if callable(empty_cache_fn):
        empty_cache_fn()


def memory_allocated(device: Any) -> Optional[int]:
    """Bytes currently held by the device allocator, or ``None`` if unknown.

    ``None`` -- not ``0`` -- when the backend does not expose the metric, so an
    absent measurement can never be read as an idle device.
    """
    backend = _backend_namespace(device)
    allocated = getattr(backend, "memory_allocated", None) if backend is not None else None
    return int(allocated()) if callable(allocated) else None


def max_memory_allocated(device: Any) -> Optional[int]:
    """Peak allocator bytes since the last reset, or ``None`` if unknown."""
    backend = _backend_namespace(device)
    peak = getattr(backend, "max_memory_allocated", None) if backend is not None else None
    return int(peak()) if callable(peak) else None


def max_memory_reserved(device: Any) -> Optional[int]:
    """Peak *reserved* pool bytes.

    CUDA-only: the caching allocator keeps a reserved arena, while Ascend has no
    equivalent split between allocated and reserved.  Reporting the CUDA number
    under an NPU label -- or the reverse -- would compare two different
    quantities, so this returns ``None`` where the concept does not exist.
    """
    if accelerator_kind(device) != CUDA:
        return None
    reserved = getattr(_backend_namespace(device), "max_memory_reserved", None)
    return int(reserved()) if callable(reserved) else None


def reset_peak_memory_stats(device: Any) -> None:
    """Reset the peak-memory high-water mark before a measurement window."""
    backend = _backend_namespace(device)
    reset = getattr(backend, "reset_peak_memory_stats", None) if backend is not None else None
    if callable(reset):
        reset()


def mem_get_info(device: Any) -> Optional[Tuple[int, int]]:
    """``(free_bytes, total_bytes)`` for the device, or ``None`` if unsupported.

    On unified-memory parts (Jetson, 310B) ``total`` is the whole system DRAM,
    so callers must budget from ``free`` rather than from ``total``.
    """
    backend = _backend_namespace(device)
    if backend is None:
        return None
    info = getattr(backend, "mem_get_info", None)
    if not callable(info):
        return None
    try:
        free_bytes, total_bytes = info()
    except Exception:
        return None
    return int(free_bytes), int(total_bytes)


def out_of_memory_error(device: Any) -> type:
    """The exception class a device OOM raises on this backend."""
    torch = _torch()
    kind = accelerator_kind(device)
    if kind == CUDA:
        return torch.cuda.OutOfMemoryError
    if kind == NPU:
        # torch_npu surfaces allocation failures as torch.OutOfMemoryError.
        return getattr(torch, "OutOfMemoryError", RuntimeError)
    return RuntimeError


def describe(device: Any) -> dict:
    """A small, reportable description of the device actually in use.

    Records the *actual* backend rather than the requested one, so a run that
    fell back to CPU cannot be filed under an accelerator label.
    """
    torch = _torch()
    kind = accelerator_kind(device)
    name = ""
    if kind == CUDA and torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(accelerator_index(device))
        except Exception:
            name = ""
    elif kind == NPU and npu_available():
        try:
            import torch_npu  # noqa: F401

            name = torch.npu.get_device_name(accelerator_index(device))
        except Exception:
            name = ""
    return {
        "device": str(device),
        "kind": kind,
        "name": name,
        "is_accelerator": kind in ACCELERATOR_KINDS,
    }


__all__ = [
    "ACCELERATOR_KINDS",
    "CPU",
    "CUDA",
    "DEVICE_ENV",
    "NPU",
    "accelerator_index",
    "accelerator_kind",
    "available_accelerators",
    "cuda_available",
    "default_accelerator",
    "describe",
    "empty_cache",
    "is_accelerator",
    "max_memory_allocated",
    "max_memory_reserved",
    "mem_get_info",
    "memory_allocated",
    "npu_available",
    "out_of_memory_error",
    "reset_peak_memory_stats",
    "resolve_device",
    "synchronize",
]
