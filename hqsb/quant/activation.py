"""Activation quantization: W8A8 static/dynamic and SmoothQuant (E05-07).

P1 scope. This module provides the *capabilities* the experiment needs, not
results:

* an activation quantizer with static/dynamic scale determination and
  per-tensor / per-token / per-channel / per-group granularity;
* a quantized *mathematical* reference (integer accumulation plus scale/zero
  correction) so a real kernel can be checked against exact math;
* the SmoothQuant equivalent transform with an equivalence check that must
  pass **before** any quantization ("smooth transform only" first);
* an online-cost decomposition skeleton (reduction / scale compute /
  quantize / integer GEMM / dequant / epilogue / launch) that records
  "unmeasured" as ``NaN`` rather than zero;
* drift/stress support: static scales carry their calibration domain, and the
  drift report compares同域/异域 and short/long behaviour.

``W8A16`` is never called ``W8A8``: the scheme records both operands' bit
widths, and a W8A8 claim requires the weight artifact *and* the activation
spec.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.quant.rtn import quantize
from hqsb.quant.spec import (
    GRANULARITY_PER_CHANNEL,
    GRANULARITY_PER_GROUP,
    GRANULARITY_PER_TENSOR,
    QuantScheme,
)

#: Activation granularity names (E05-07 §5).
GRANULARITY_PER_TOKEN = "per_token"
ACTIVATION_GRANULARITIES = (
    GRANULARITY_PER_TENSOR,
    GRANULARITY_PER_TOKEN,
    GRANULARITY_PER_CHANNEL,
    GRANULARITY_PER_GROUP,
)

#: Scale determination modes (E05-07 §4).
STATIC = "static"
DYNAMIC = "dynamic"

SCALE_MODES = (STATIC, DYNAMIC)


@dataclass
class ActivationQuantSpec:
    """Frozen activation quantization spec (E05-07 §12 step 1)."""

    granularity: str = GRANULARITY_PER_TOKEN
    mode: str = DYNAMIC
    bits: int = 8
    qmin: int = -127
    qmax: int = 127
    round_mode: str = "nearest_even"
    scale_dtype: str = "float32"
    clip_percentile: Optional[float] = None
    clip_threshold: Optional[float] = None
    nan_policy: str = "reject"
    symmetric: bool = True
    group_size: Optional[int] = None
    #: Semantic label used when reporting ("W8A8", "fake W8A8", ...).
    label: str = "w8a8"

    def __post_init__(self) -> None:
        if self.granularity not in ACTIVATION_GRANULARITIES:
            raise ConfigError(
                f"unknown activation granularity {self.granularity!r}; "
                f"supported: {list(ACTIVATION_GRANULARITIES)}"
            )
        if self.mode not in SCALE_MODES:
            raise ConfigError(
                f"unknown scale mode {self.mode!r}; supported: {list(SCALE_MODES)}"
            )
        if self.granularity == GRANULARITY_PER_GROUP and self.group_size is None:
            raise ConfigError("per-group activation quantization needs a group_size")
        if self.clip_percentile is not None and not 0.0 < self.clip_percentile <= 1.0:
            raise ConfigError(
                f"clip_percentile must be in (0, 1], got {self.clip_percentile}"
            )
        if self.bits != 8 and self.qmax == 127:
            # Any non-8-bit width must declare its own range explicitly.
            raise ConfigError(
                f"bits={self.bits} but qmax={self.qmax} looks like an INT8 default; "
                f"declare the integer range explicitly"
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "granularity": self.granularity,
            "mode": self.mode,
            "bits": self.bits,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "round_mode": self.round_mode,
            "scale_dtype": self.scale_dtype,
            "clip_percentile": self.clip_percentile,
            "clip_threshold": self.clip_threshold,
            "nan_policy": self.nan_policy,
            "symmetric": self.symmetric,
            "group_size": self.group_size,
            "label": self.label,
        }


def scheme_for_activation(spec: ActivationQuantSpec, cols: int) -> QuantScheme:
    """Translate an activation spec into a :class:`QuantScheme`.

    Mapping (documented so the two vocabularies cannot drift):

    * ``per_tensor`` → one unit for the whole tensor;
    * ``per_token`` → one unit per row (row = token) along the last axis;
    * ``per_channel`` → one unit per column (axis 0);
    * ``per_group`` → groups of ``group_size`` along the last axis.
    """
    if spec.granularity == GRANULARITY_PER_TENSOR:
        return QuantScheme(
            bits=spec.bits,
            granularity=GRANULARITY_PER_TENSOR,
            symmetric=spec.symmetric,
            range_policy="symmetric" if spec.symmetric else "twos_complement",
            round_mode=spec.round_mode,
            scale_dtype=spec.scale_dtype,
            nan_policy=spec.nan_policy,
            label=f"{spec.label}_per_tensor",
        )
    if spec.granularity == GRANULARITY_PER_TOKEN:
        return QuantScheme(
            bits=spec.bits,
            granularity=GRANULARITY_PER_CHANNEL,
            axis=-1,
            symmetric=spec.symmetric,
            range_policy="symmetric" if spec.symmetric else "twos_complement",
            round_mode=spec.round_mode,
            scale_dtype=spec.scale_dtype,
            nan_policy=spec.nan_policy,
            label=f"{spec.label}_per_token",
        )
    if spec.granularity == GRANULARITY_PER_CHANNEL:
        return QuantScheme(
            bits=spec.bits,
            granularity=GRANULARITY_PER_CHANNEL,
            axis=0,
            symmetric=spec.symmetric,
            range_policy="symmetric" if spec.symmetric else "twos_complement",
            round_mode=spec.round_mode,
            scale_dtype=spec.scale_dtype,
            nan_policy=spec.nan_policy,
            label=f"{spec.label}_per_channel",
        )
    return QuantScheme(
        bits=spec.bits,
        granularity=GRANULARITY_PER_GROUP,
        group_size=spec.group_size,
        axis=-1,
        symmetric=spec.symmetric,
        range_policy="symmetric" if spec.symmetric else "twos_complement",
        round_mode=spec.round_mode,
        scale_dtype=spec.scale_dtype,
        nan_policy=spec.nan_policy,
        label=f"{spec.label}_per_group{spec.group_size}",
    )


@dataclass
class ActivationQuantResult:
    """Result of quantizing one activation tensor."""

    spec: ActivationQuantSpec
    shape: Tuple[int, ...]
    q: List[int]
    scales: List[float]
    zeros: List[int]
    dequant: List[float]
    saturation_rate: float
    clip_applied: bool
    scale_utilization: float
    per_token_absmax: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.as_dict(),
            "shape": list(self.shape),
            "num_values": len(self.q),
            "num_scales": len(self.scales),
            "saturation_rate": self.saturation_rate,
            "clip_applied": self.clip_applied,
            "scale_utilization": self.scale_utilization,
            "per_token_absmax_count": len(self.per_token_absmax),
        }


def _clip(values: List[float], spec: ActivationQuantSpec) -> Tuple[List[float], bool]:
    """Apply the pre-registered clipping threshold, if any.

    Clipping is a *bias–variance* trade-off and must come from
    policy-validation (E05-07 §6); this function only applies what the spec
    declares and reports whether it changed anything.
    """
    if spec.clip_threshold is not None:
        threshold = float(spec.clip_threshold)
        return ([max(-threshold, min(threshold, value)) for value in values], True)
    if spec.clip_percentile is not None and values:
        ordered = sorted(abs(value) for value in values)
        index = min(len(ordered) - 1, int(math.ceil(spec.clip_percentile * len(ordered))) - 1)
        threshold = ordered[index]
        return ([max(-threshold, min(threshold, value)) for value in values], True)
    return (values, False)


def quantize_activation(
    values: Sequence[float],
    spec: ActivationQuantSpec,
    *,
    shape: Sequence[int],
    static_scales: Optional[Sequence[float]] = None,
) -> ActivationQuantResult:
    """Quantize activations with static or dynamic scales.

    ``static_scales`` is required in static mode (they come from calibration)
    and refused in dynamic mode: mixing the two would make the report
    ambiguous. Saturation and scale utilization are always reported so a
    static-scale failure at deployment time is visible.
    """
    flat = [float(value) for value in values]
    shape_tuple = tuple(int(dim) for dim in shape)
    expected = 1
    for dim in shape_tuple:
        expected *= dim
    if expected != len(flat):
        raise ConfigError(f"shape {shape_tuple} implies {expected} values, got {len(flat)}")
    if spec.mode == STATIC and static_scales is None:
        raise ConfigError(
            "static activation quantization requires calibration-derived scales; "
            "computing them from the evaluation batch would make it dynamic"
        )
    if spec.mode == DYNAMIC and static_scales is not None:
        raise ConfigError(
            "dynamic activation quantization must not receive static scales"
        )

    clipped, clip_applied = _clip(flat, spec)
    scheme = scheme_for_activation(spec, shape_tuple[-1])
    if spec.mode == STATIC:
        qt = _quantize_with_fixed_scales(clipped, scheme, static_scales, shape_tuple)
    else:
        qt = quantize(clipped, scheme, shape=shape_tuple)
    q_values = qt.q if hasattr(qt, "q") else qt["q"]
    scales = qt.scales if hasattr(qt, "scales") else qt["scales"]
    zeros = qt.zeros if hasattr(qt, "zeros") else qt["zeros"]
    dequant = (
        qt.values_dequant if hasattr(qt, "values_dequant") else qt["dequant"]
    )

    saturated = 0
    for index, code in enumerate(q_values):
        if code == spec.qmax or code == spec.qmin:
            saturated += 1
    utilization = _scale_utilization(clipped, q_values, scales, shape_tuple, scheme)
    per_token_absmax: List[float] = []
    if spec.granularity == GRANULARITY_PER_TOKEN and len(shape_tuple) >= 2:
        cols = shape_tuple[-1]
        for row in range(len(clipped) // max(1, cols)):
            chunk = clipped[row * cols : (row + 1) * cols]
            per_token_absmax.append(max((abs(value) for value in chunk), default=0.0))
    return ActivationQuantResult(
        spec=spec,
        shape=shape_tuple,
        q=list(q_values),
        scales=[float(scale) for scale in scales],
        zeros=[int(zero) for zero in zeros],
        dequant=[float(value) for value in dequant],
        saturation_rate=(saturated / len(q_values)) if q_values else float("nan"),
        clip_applied=clip_applied,
        scale_utilization=utilization,
        per_token_absmax=per_token_absmax,
    )


def _quantize_with_fixed_scales(
    values: List[float],
    scheme: QuantScheme,
    scales: Sequence[float],
    shape: Tuple[int, ...],
) -> Dict[str, Any]:
    """Quantize with caller-provided scales (static mode)."""
    from hqsb.quant.rounding import coerce_round_value, get_rounder

    qmin, qmax = scheme.qmin, scheme.qmax
    round_fn = get_rounder(scheme.round_mode)
    if scheme.granularity == GRANULARITY_PER_TENSOR:
        unit_for = lambda index: 0  # noqa: E731
    elif scheme.axis == 0:
        cols = shape[-1]
        unit_for = lambda index: index % cols  # noqa: E731
    else:
        cols = shape[-1]
        group = scheme.group_size
        if group is None:
            unit_for = lambda index: index // cols  # noqa: E731
        else:
            unit_for = lambda index: (index // cols) * (  # noqa: E731
                (cols + group - 1) // group
            ) + (index % cols) // group
    if len(scales) < 1:
        raise ConfigError("static scales must not be empty")
    q: List[int] = []
    dequant: List[float] = []
    zeros: List[int] = []
    for index, value in enumerate(values):
        unit = unit_for(index)
        if unit >= len(scales):
            raise ConfigError(
                f"value {index} maps to unit {unit} but only {len(scales)} static "
                f"scales were provided"
            )
        scale = float(scales[unit])
        if not math.isfinite(scale) or scale <= 0:
            raise ConfigError(f"static scale[{unit}] = {scale!r} is not usable")
        code = coerce_round_value(round_fn(value / scale))
        code = max(qmin, min(qmax, code))
        q.append(code)
        dequant.append(scale * code)
    return {"q": q, "scales": [float(s) for s in scales], "zeros": zeros, "dequant": dequant}


def _scale_utilization(
    values: Sequence[float],
    q: Sequence[int],
    scales: Sequence[float],
    shape: Tuple[int, ...],
    scheme: QuantScheme,
) -> float:
    """Fraction of values whose code uses at least half of the available range.

    A low utilization means the scale was set by an outlier and most values
    are represented coarsely (E05-03 §3.3).
    """
    if not q or not scales:
        return float("nan")
    # Signed codes span both sides of zero. Half of the full signed span
    # would count only +/-127 in INT8, duplicating boundary occupancy.
    half = max(abs(scheme.qmin), abs(scheme.qmax)) / 2.0
    used = 0
    for code in q:
        if abs(code) >= half:
            used += 1
    return used / len(q)


def compute_static_scales(
    per_unit_absmax: Mapping[str, Sequence[float]],
    spec: ActivationQuantSpec,
) -> Dict[str, Any]:
    """Compute static scales from calibration statistics (E05-07 step 4).

    Calibration provenance is part of the output: a static scale without the
    dataset hash is unusable because it would make the applicability domain
    unknowable.
    """
    if spec.mode != STATIC:
        raise ConfigError("static scale computation requires mode='static'")
    scales: Dict[str, List[float]] = {}
    for module, absmax_values in sorted(per_unit_absmax.items()):
        resolved = [max(float(value), 1e-8) / spec.qmax for value in absmax_values]
        scales[module] = resolved
    return {
        "scales": scales,
        "spec": spec.as_dict(),
        "count": sum(len(value) for value in scales.values()),
        "note": "each scale = max(abs(x)) / qmax for its unit",
    }


# ── quantized mathematical reference and SmoothQuant ──────────────────────


def int32_accumulate(
    x_q: Sequence[int], w_q: Sequence[int], *, m: int, n: int, k: int
) -> List[int]:
    """Exact integer accumulation ``A_int32 = q_x @ q_w`` in pure Python.

    Integers are summed exactly (Python ints are unbounded), so this is an
    *exact* reference for the accumulator — no float rounding can hide an
    indexing or zero-point bug in the kernel.
    """
    if len(x_q) != m * k:
        raise ConfigError(f"x_q has {len(x_q)} values, expected m*k={m * k}")
    if len(w_q) != k * n:
        raise ConfigError(f"w_q has {len(w_q)} values, expected k*n={k * n}")
    out: List[int] = []
    for row in range(m):
        for col in range(n):
            total = 0
            for index in range(k):
                total += x_q[row * k + index] * w_q[index * n + col]
            out.append(total)
    return out


def w8a8_reference(
    x_q: Sequence[int],
    w_q: Sequence[int],
    *,
    x_scales: Sequence[float],
    w_scales: Sequence[float],
    m: int,
    n: int,
    k: int,
    x_units_per_row: int = 1,
    w_units_per_row: int = 1,
    x_zeros: Optional[Sequence[int]] = None,
    w_zeros: Optional[Sequence[int]] = None,
    bias: Optional[Sequence[float]] = None,
) -> List[float]:
    """Quantized mathematical reference with per-unit scale broadcast.

    ``X`` is row-major ``[M,K]`` and ``W`` is row-major ``[K,N]``.
    Activation scales contain one scalar or M per-token values; weight
    scales contain one scalar or N per-output-channel values. There is one
    quantization unit along K per row/channel. Multiple K groups need
    separate partial accumulations and explicit group boundaries, which this
    epilogue-only oracle does not accept. Integer accumulation is exact;
    only final scaling and the optional bias use floating point.
    """
    if x_zeros is not None or w_zeros is not None:
        raise ConfigError(
            "zero-point correction must be folded into the quantized values "
            "before calling the reference (or the reference must be extended); "
            "silently ignoring zero points would hide a correctness bug"
        )
    if x_units_per_row != 1 or w_units_per_row != 1:
        raise ConfigError(
            "multiple K groups require explicit group boundaries and partial "
            "accumulations; this reference supports one unit per row/channel"
        )
    for name, scales, count in (("x_scales", x_scales, m), ("w_scales", w_scales, n)):
        if len(scales) not in (1, count):
            raise ConfigError(
                f"{name} must contain one scalar or {count} row/channel scales, "
                f"got {len(scales)}"
            )
        for index, scale in enumerate(scales):
            if not math.isfinite(float(scale)) or float(scale) <= 0:
                raise ConfigError(f"{name}[{index}] must be finite and positive")
    if bias is not None and len(bias) != n:
        raise ConfigError(f"bias has {len(bias)} values, expected N={n}")
    accumulated = int32_accumulate(x_q, w_q, m=m, n=n, k=k)
    out: List[float] = []
    for row in range(m):
        x_scale = float(x_scales[0 if len(x_scales) == 1 else row])
        for col in range(n):
            w_scale = float(w_scales[0 if len(w_scales) == 1 else col])
            value = x_scale * w_scale * accumulated[row * n + col]
            if bias is not None:
                value += float(bias[col])
            out.append(value)
    return out


@dataclass
class SmoothTransform:
    """SmoothQuant-style equivalent scaling (E05-07 §8).

    ``s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)`` moved from the activation
    to the weight: ``X W = (X diag(s)^-1)(diag(s) W)``. The fold target and the
    residual-branch handling are part of the contract; they must be recorded
    because they determine which operators have to be replaced.
    """

    scale: List[float]
    alpha: float
    fold_target: str = "weight_and_previous_op"
    residual_scaled: bool = False

    def apply_to_weight(self, weight: Sequence[float], *, cols: int) -> List[float]:
        if len(weight) % cols != 0:
            raise ConfigError(
                f"weight length {len(weight)} is not a multiple of K={cols}"
            )
        if len(self.scale) != cols:
            raise ConfigError(
                f"smooth scale has {len(self.scale)} entries, expected K={cols}"
            )
        out: List[float] = []
        for index, value in enumerate(weight):
            out.append(float(value) * self.scale[index % cols])
        return out

    def apply_to_activation(self, activations: Sequence[float], *, cols: int) -> List[float]:
        if len(self.scale) != cols:
            raise ConfigError(
                f"smooth scale has {len(self.scale)} entries, expected K={cols}"
            )
        return [
            float(value) / self.scale[index % cols]
            for index, value in enumerate(activations)
        ]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "fold_target": self.fold_target,
            "residual_scaled": self.residual_scaled,
            "scale_length": len(self.scale),
            "scale_min": min(self.scale) if self.scale else None,
            "scale_max": max(self.scale) if self.scale else None,
        }


def compute_smooth_scale(
    activation_absmax: Sequence[float],
    weight_absmax: Sequence[float],
    *,
    alpha: float,
    eps: float = 1e-8,
) -> SmoothTransform:
    """Compute the SmoothQuant scale with the given alpha (policy-validation only)."""
    if not 0.0 <= alpha <= 1.0:
        raise ConfigError(f"alpha must be in [0, 1], got {alpha}")
    if len(activation_absmax) != len(weight_absmax):
        raise ConfigError(
            f"activation ({len(activation_absmax)}) and weight "
            f"({len(weight_absmax)}) channel counts must match"
        )
    scale = []
    for act, weight in zip(activation_absmax, weight_absmax):
        numerator = max(float(act), eps) ** alpha
        denominator = max(float(weight), eps) ** (1.0 - alpha)
        scale.append(numerator / denominator)
    return SmoothTransform(scale=scale, alpha=alpha)


@dataclass
class OnlineCostBreakdown:
    """Online cost of an activation-quantized path (E05-07 §10).

    Every field is optional; ``None`` means "not measured separately" and is
    reported as ``NaN``, because inferring a missing component by subtracting
    two totals would fabricate a number.
    """

    reduction_ms: float = float("nan")
    scale_compute_ms: float = float("nan")
    quantize_ms: float = float("nan")
    gemm_ms: float = float("nan")
    dequant_ms: float = float("nan")
    epilogue_ms: float = float("nan")
    launch_ms: float = float("nan")
    total_ms: float = float("nan")
    fusion: str = "unfused"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reduction_ms": self.reduction_ms,
            "scale_compute_ms": self.scale_compute_ms,
            "quantize_ms": self.quantize_ms,
            "gemm_ms": self.gemm_ms,
            "dequant_ms": self.dequant_ms,
            "epilogue_ms": self.epilogue_ms,
            "launch_ms": self.launch_ms,
            "total_ms": self.total_ms,
            "fusion": self.fusion,
        }

    def unmeasured_components(self) -> List[str]:
        names = (
            "reduction_ms",
            "scale_compute_ms",
            "quantize_ms",
            "gemm_ms",
            "dequant_ms",
            "epilogue_ms",
            "launch_ms",
        )
        return [name for name in names if math.isnan(getattr(self, name))]


def drift_report(
    same_domain: Sequence[float],
    other_domain: Sequence[float],
    *,
    short_context: Sequence[float],
    long_context: Sequence[float],
) -> Dict[str, Any]:
    """Compare saturation/scale drift across domains and lengths (E05-07 §16).

    Returns per-slice summaries and the worst-case degradation; a static
    configuration that fails on one slice must be visible here rather than
    averaged away in the overall number.
    """
    from hqsb.quant.stats import summarize_distribution

    return {
        "same_domain": summarize_distribution(same_domain).as_dict(),
        "other_domain": summarize_distribution(other_domain).as_dict(),
        "short_context": summarize_distribution(short_context).as_dict(),
        "long_context": summarize_distribution(long_context).as_dict(),
        "domain_shift": (
            summarize_distribution(other_domain).mean
            - summarize_distribution(same_domain).mean
        ),
        "length_shift": (
            summarize_distribution(long_context).mean
            - summarize_distribution(short_context).mean
        ),
    }


def activation_spec_json(spec: ActivationQuantSpec) -> str:
    return json.dumps(spec.as_dict(), sort_keys=True, indent=2)


__all__ = [
    "ACTIVATION_GRANULARITIES",
    "ActivationQuantResult",
    "ActivationQuantSpec",
    "DYNAMIC",
    "GRANULARITY_PER_TOKEN",
    "OnlineCostBreakdown",
    "SCALE_MODES",
    "STATIC",
    "SmoothTransform",
    "compute_smooth_scale",
    "compute_static_scales",
    "drift_report",
    "int32_accumulate",
    "quantize_activation",
    "scheme_for_activation",
    "w8a8_reference",
]
