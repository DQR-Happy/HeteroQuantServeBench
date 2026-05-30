"""Low-bit execution capability probe (E05-06 §13 step 3).

The probe answers, for *this* environment: which bit widths, layouts, dtypes
and shapes can be executed by a real kernel, and with which reason codes when
they cannot. It never raises, never caches across processes, and never
assumes that a documented library feature exists (E05-06 §6: "文档版本不等于
本机能力").

Results feed :func:`hqsb.quant.compat.check_compatibility` as a
``KernelCapability``; a mismatch produces a pre-launch refusal or an explicit
repack requirement rather than a silent FP16 fallback.
"""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hqsb.quant.packing import (
    LAYOUT_W4A16_ROWMAJOR_NK_V1,
    LAYOUT_W8A16_ROWMAJOR_NK_V1,
)

PROBE_SCHEMA_VERSION = "1.0.0"


@dataclass
class LowBitCapability:
    """Machine-readable capability report for the low-bit executors."""

    torch_available: bool = False
    torch_version: str = ""
    cuda_available: bool = False
    device_capability: Optional[Tuple[int, int]] = None
    device_name: str = ""
    triton_available: bool = False
    triton_version: str = ""
    triton_compiles: bool = False
    supported_bits: Tuple[int, ...] = ()
    supported_layouts: Tuple[str, ...] = ()
    supported_dtype: str = "float16"
    supported_group_sizes: Tuple[Optional[int], ...] = ()
    max_shared_memory_bytes: int = 0
    notes: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)
    host: str = ""

    @property
    def fused_dequant_available(self) -> bool:
        return (
            self.torch_available
            and self.cuda_available
            and self.triton_available
            and self.triton_compiles
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": PROBE_SCHEMA_VERSION,
            "host": self.host or platform.node(),
            "python": platform.python_version(),
            "torch_available": self.torch_available,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "device_capability": list(self.device_capability) if self.device_capability else None,
            "device_name": self.device_name,
            "triton_available": self.triton_available,
            "triton_version": self.triton_version,
            "triton_compiles": self.triton_compiles,
            "fused_dequant_available": self.fused_dequant_available,
            "supported_bits": list(self.supported_bits),
            "supported_layouts": list(self.supported_layouts),
            "supported_dtype": self.supported_dtype,
            "supported_group_sizes": list(self.supported_group_sizes),
            "max_shared_memory_bytes": self.max_shared_memory_bytes,
            "notes": list(self.notes),
            "reasons": dict(self.reasons),
        }


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def probe_low_bit_capability(
    *, compile_probe: bool = True, device_index: int = 0
) -> LowBitCapability:
    """Probe torch/CUDA/Triton and compile a minimal low-bit kernel.

    ``compile_probe=False`` skips the actual Triton compilation (fast path for
    import-time checks); the experiment driver must use the default ``True``
    because an importable Triton is not a compiling Triton.
    """
    capability = LowBitCapability()
    capability.host = platform.node()

    if not _module_available("torch"):
        capability.reasons["torch"] = "torch is not installed"
        return capability
    import torch

    capability.torch_available = True
    capability.torch_version = str(torch.__version__)
    if not torch.cuda.is_available():
        capability.reasons["cuda"] = "no CUDA device visible to torch"
        return capability
    capability.cuda_available = True
    properties = torch.cuda.get_device_properties(device_index)
    capability.device_capability = (int(properties.major), int(properties.minor))
    capability.device_name = str(properties.name)
    capability.max_shared_memory_bytes = int(
        getattr(properties, "shared_memory_per_block_optin", 0) or 0
    )

    if not _module_available("triton"):
        capability.reasons["triton"] = "triton is not installed"
        return capability
    import triton

    capability.triton_available = True
    capability.triton_version = str(getattr(triton, "__version__", ""))

    if compile_probe:
        try:
            from ops.quant.w4a16_triton import compile_probe as triton_compile_probe

            triton_compile_probe()
            capability.triton_compiles = True
        except Exception as exc:  # noqa: BLE001 - a failed compile is a capability state
            capability.reasons["triton_compile"] = f"{type(exc).__name__}: {exc}"
            capability.triton_compiles = False

    if capability.triton_compiles:
        capability.supported_bits = (8, 4)
        capability.supported_layouts = (
            LAYOUT_W4A16_ROWMAJOR_NK_V1,
            LAYOUT_W8A16_ROWMAJOR_NK_V1,
        )
        capability.supported_group_sizes = (None, 32, 64, 128)
    return capability


def kernel_capability_from_probe(capability: LowBitCapability):
    """Convert a probe into a :class:`hqsb.quant.compat.KernelCapability`."""
    from hqsb.quant import compat

    arch = (
        f"sm_{capability.device_capability[0]}{capability.device_capability[1]}"
        if capability.device_capability
        else "any"
    )
    available = capability.fused_dequant_available
    reason = ""
    if not available:
        reason = "; ".join(
            f"{key}: {value}" for key, value in sorted(capability.reasons.items())
        ) or "low-bit executor unavailable"
    return compat.KernelCapability(
        kernel_id="hqsb.w4a16.triton" if 4 in capability.supported_bits else "hqsb.lowbit.none",
        provider="triton",
        layouts=tuple(capability.supported_layouts),
        bits=tuple(capability.supported_bits),
        group_sizes=tuple(capability.supported_group_sizes),
        dtype=capability.supported_dtype,
        target_arch=arch,
        abi_version="1",
        available=available,
        unavailable_reason=reason,
        observed_symbol="",
    )


__all__ = [
    "LowBitCapability",
    "PROBE_SCHEMA_VERSION",
    "kernel_capability_from_probe",
    "probe_low_bit_capability",
]
