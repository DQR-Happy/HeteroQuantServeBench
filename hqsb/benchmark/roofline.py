"""Roofline and Amdahl analysis primitives.

These pure functions model the hardware performance envelope so hotspot
candidates can be classified and ranked with *numbers*, not guesses:

* :class:`RooflineModel` — the classic Roofline bound: achievable
  performance is ``min(peak_flops, peak_bandwidth * arithmetic_intensity)``.
* :func:`amdahl_speedup` / :func:`amdahl_max_speedup` — the theoretical
  ceiling of optimizing a fraction ``p`` of total time.
* :func:`classify_hotspot` — maps a measured kernel to an optimization
  category (GEMM/attention, elementwise/reduction, runtime/sync, memory).

All functions are pure and unit-testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Mapping, Sequence

from hqsb.core.errors import ConfigError


class HotspotClass(str, Enum):
    """Optimization category for a measured hotspot."""

    GEMM_ATTENTION = "gemm_attention"
    ELEMENTWISE_REDUCTION = "elementwise_reduction"
    RUNTIME_SYNC = "runtime_sync"
    MEMORY_KV = "memory_kv"
    OTHER = "other"


@dataclass(frozen=True)
class RooflineModel:
    """A hardware Roofline model (FLOPS/bandwidth envelope).

    Attributes:
        name: Model/device name for reporting.
        peak_flops: Peak arithmetic throughput in FLOP/s.
        peak_bandwidth: Peak DRAM bandwidth in bytes/s.
    """

    name: str
    peak_flops: float
    peak_bandwidth: float

    def ridge_point(self) -> float:
        """Return the arithmetic intensity (FLOP/byte) at the ridge point.

        Below this intensity the kernel is bandwidth-bound; above it the
        kernel is compute-bound.
        """
        if self.peak_bandwidth <= 0:
            return 0.0
        return self.peak_flops / self.peak_bandwidth

    def achievable_flops(self, arithmetic_intensity: float) -> float:
        """Return the Roofline upper bound for a given arithmetic intensity."""
        if arithmetic_intensity < 0:
            raise ConfigError(
                f"arithmetic_intensity must be >= 0, got {arithmetic_intensity}"
            )
        bandwidth_bound = self.peak_bandwidth * arithmetic_intensity
        return min(self.peak_flops, bandwidth_bound)

    def is_compute_bound(self, arithmetic_intensity: float) -> bool:
        """Return True if ``arithmetic_intensity`` exceeds the ridge point."""
        return arithmetic_intensity >= self.ridge_point()

    def efficiency(self, achieved_flops: float, arithmetic_intensity: float) -> float:
        """Return achieved performance as a fraction of the Roofline bound.

        Returns 0.0 if the bound is 0 to avoid division by zero.
        """
        bound = self.achievable_flops(arithmetic_intensity)
        if bound <= 0:
            return 0.0
        return achieved_flops / bound


def amdahl_speedup(parallel_fraction: float, speedup_factor: float) -> float:
    """Amdahl's law: overall speedup from accelerating a fraction of time.

    Args:
        parallel_fraction: Fraction of total runtime (0..1) that is sped up.
        speedup_factor: Speedup applied to that fraction (>= 1).

    Returns:
        Overall application speedup (>= 1).

    Raises:
        ConfigError: If ``parallel_fraction`` is outside [0, 1] or
            ``speedup_factor`` < 1.
    """
    if not (0.0 <= parallel_fraction <= 1.0):
        raise ConfigError(
            f"parallel_fraction must be in [0, 1], got {parallel_fraction}"
        )
    if speedup_factor < 1.0:
        raise ConfigError(
            f"speedup_factor must be >= 1, got {speedup_factor}"
        )
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / speedup_factor)


def amdahl_max_speedup(parallel_fraction: float) -> float:
    """Theoretical maximum speedup if the fraction were accelerated infinitely.

    Equivalently ``1 / (1 - parallel_fraction)``.
    """
    if not (0.0 <= parallel_fraction < 1.0):
        raise ConfigError(
            f"parallel_fraction must be in [0, 1), got {parallel_fraction}"
        )
    return 1.0 / (1.0 - parallel_fraction)


@dataclass(frozen=True)
class HotspotCandidate:
    """A ranked optimization candidate derived from profile data.

    Attributes:
        name: Kernel/op name.
        time_share: Fraction of total measured time (0..1).
        classification: :class:`HotspotClass`.
        amdahl_max: Theoretical max speedup if fully eliminated.
        notes: Free-form rationale.
    """

    name: str
    time_share: float
    classification: HotspotClass
    amdahl_max: float
    notes: str = ""


def classify_hotspot(name: str, *, is_gemm: bool = False) -> HotspotClass:
    """Classify a kernel name into an optimization category.

    Uses well-known naming conventions from PyTorch/LLM kernels. Falls back
    to :class:`HotspotClass.OTHER` when no convention matches.
    """
    lower = name.lower()

    if is_gemm or any(
        k in lower
        for k in ("matmul", "gemm", "linear", "conv", "addmm", "bmm", "::mm")
    ):
        return HotspotClass.GEMM_ATTENTION
    if any(k in lower for k in ("attention", "softmax", "flash_attn", "scaled_dot")):
        return HotspotClass.GEMM_ATTENTION
    if any(k in lower for k in ("rms_norm", "layer_norm", "layernorm", "norm", "silu", "gelu", "elementwise", "add", "mul", "copy", "cat", "relu")):
        return HotspotClass.ELEMENTWISE_REDUCTION
    if any(k in lower for k in ("cuda_graph", "synchronize", "memcpy", "allocator", "caching_allocator", "launch")):
        return HotspotClass.RUNTIME_SYNC
    if any(k in lower for k in ("kv_cache", "cache", "repeat_interleave", "index")):
        return HotspotClass.MEMORY_KV
    return HotspotClass.OTHER


def rank_hotspots(
    candidates: Sequence[Mapping],
) -> List[HotspotCandidate]:
    """Build and sort :class:`HotspotCandidate` objects by time share.

    Args:
        candidates: Sequence of dicts with keys ``name`` and ``time_share``
            (required) and optional ``is_gemm`` / ``notes``.

    Returns:
        Candidates sorted by descending ``time_share``, each with a computed
        ``amdahl_max`` and ``classification``.
    """
    ranked: List[HotspotCandidate] = []
    for c in candidates:
        share = float(c["time_share"])
        name = str(c["name"])
        ranked.append(
            HotspotCandidate(
                name=name,
                time_share=share,
                classification=classify_hotspot(
                    name, is_gemm=bool(c.get("is_gemm", False))
                ),
                amdahl_max=amdahl_max_speedup(min(share, 1.0)),
                notes=str(c.get("notes", "")),
            )
        )
    return sorted(ranked, key=lambda x: x.time_share, reverse=True)


# ── Device Roofline presets ─────────────────────────────────────────

# Jetson Orin Nano Super 8GB (Ampere, sm_87). Peak values are nominal:
# the device datasheet advertises ~67 FP16 TFLOPS (with sparsity ~134) and
# ~68 GB/s DRAM. These are *model* values for the Roofline envelope, not
# measured silicon numbers; measured bandwidth/FLOPS replace them at
# runtime where available.
ORIN_NANO_SUPER_FP16 = RooflineModel(
    name="Jetson Orin Nano Super (FP16)",
    peak_flops=67e12,
    peak_bandwidth=68e9,
)


__all__ = [
    "HotspotCandidate",
    "HotspotClass",
    "RooflineModel",
    "ORIN_NANO_SUPER_FP16",
    "amdahl_max_speedup",
    "amdahl_speedup",
    "classify_hotspot",
    "rank_hotspots",
]
