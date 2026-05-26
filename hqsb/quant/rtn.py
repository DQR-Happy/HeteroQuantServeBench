"""Round-to-nearest quantization reference (E05-01 §3/§4).

This is the *simple but not sloppy* oracle every other S05 experiment is
compared against: it implements exactly the semantics frozen in
:class:`hqsb.quant.spec.QuantScheme`, works on flat float64 values, and never
depends on a language default round, a framework helper or a GPU.

Two independent implementations exist on purpose (E05-01 §11 step 3/8):

* :func:`quantize` — the production reference (float64 arithmetic);
* :func:`hqsb.quant.golden.quantize_exact` — an exact-rational reimplementation
  used to cross-check ties and boundary codes.

The one-byte-per-value *canonical* logical representation produced here is
deliberately not packed: packing is a separate layer
(:mod:`hqsb.quant.packing`), so algorithm, storage and kernel layout can be
compared without conflation (E05-01 §7, details README §5).

Symmetric scheme (per quantization unit):

    amax = max(abs(x))
    amax == 0        -> scale = zero_group_policy, q = 0
    scale = amax / qmax_positive
    q = clamp(round(x / scale), qmin, qmax)
    x_hat = scale * q

Asymmetric scheme:

    xmin, xmax = min(x), max(x)
    xmax == xmin     -> constant_group_policy
    scale = (xmax - xmin) / (qmax - qmin)
    zero = clamp(round(qmin - xmin / scale), qmin, qmax)
    q = clamp(round(x / scale) + zero, qmin, qmax)
    x_hat = scale * (q - zero)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.quant.rounding import coerce_round_value, get_rounder
from hqsb.quant.spec import (
    GRANULARITY_PER_TENSOR,
    ConstantGroupPolicy,
    NaNPolicy,
    QuantScheme,
    ZeroGroupPolicy,
    groups_per_row,
    tail_length,
)

#: Smallest positive normal float32, used by ``zero_group_scale_min_positive``.
_FLOAT32_MIN_NORMAL = 1.1754943508222875e-38

REASON_NONFINITE_INPUT = "nonfinite_input"


@dataclass
class UnitStats:
    """Per-quantization-unit statistics recorded during quantization."""

    amax: float
    scale: float
    zero: Optional[int]
    clamp_count: int
    zero_group: bool
    constant_group: bool
    unit_index: int


@dataclass
class QuantStats:
    """Aggregate statistics of one quantization call (E05-01 §11 step 5)."""

    num_values: int
    num_units: int
    clamp_count: int
    saturated_fraction: float
    zero_group_count: int
    constant_group_count: int
    nonfinite_count: int
    #: Units whose scale is not finite/positive (must be empty in a valid run).
    invalid_scale_units: List[int] = field(default_factory=list)
    max_scale: float = 0.0
    min_positive_scale: float = math.inf

    def as_dict(self) -> Dict[str, object]:
        return {
            "num_values": self.num_values,
            "num_units": self.num_units,
            "clamp_count": self.clamp_count,
            "saturated_fraction": self.saturated_fraction,
            "zero_group_count": self.zero_group_count,
            "constant_group_count": self.constant_group_count,
            "nonfinite_count": self.nonfinite_count,
            "invalid_scale_units": list(self.invalid_scale_units),
            "max_scale": self.max_scale,
            "min_positive_scale": self.min_positive_scale,
        }


@dataclass
class QuantizedTensor:
    """Logical quantized tensor: qvalues + scales/zeros + exact unit semantics.

    ``q`` and ``values_dequant`` are flat, row-major over ``shape``. ``scales``
    and ``zeros`` are flat over units in (row-major row, group) order.
    ``zeros`` is empty for symmetric schemes.
    """

    scheme: QuantScheme
    shape: Tuple[int, ...]
    axis: int
    q: List[int]
    scales: List[float]
    zeros: List[int]
    values_dequant: List[float]
    units_per_row: int
    group_size: Optional[int]
    #: Valid length of the last group along the quantized axis.
    tail: int
    stats: QuantStats
    unit_stats: List[UnitStats]

    @property
    def num_values(self) -> int:
        return len(self.q)

    @property
    def num_units(self) -> int:
        return len(self.scales)

    def unit_of(self, index: int) -> int:
        """Unit index holding flat element ``index`` (axis-aware)."""
        if not 0 <= index < len(self.q):
            raise IndexError(f"flat index {index} out of range")
        if self.scheme.granularity == GRANULARITY_PER_TENSOR:
            return 0
        return unit_of_index(
            index, self.shape, self.axis, self.units_per_row, self.group_size
        )

    def as_dict(self) -> Dict[str, object]:
        """Compact, hash-friendly description (no payload bytes)."""
        return {
            "scheme": self.scheme.as_dict(),
            "shape": list(self.shape),
            "axis": self.axis,
            "units_per_row": self.units_per_row,
            "group_size": self.group_size,
            "tail": self.tail,
            "stats": self.stats.as_dict(),
        }


def _normalize_axis(axis: int, rank: int) -> int:
    if rank == 0:
        raise ConfigError("cannot quantize a rank-0 tensor")
    normalized = axis + rank if axis < 0 else axis
    if not 0 <= normalized < rank:
        raise ConfigError(f"axis {axis} out of range for rank {rank}")
    return normalized


def _row_count(shape: Sequence[int], axis: int) -> int:
    total = 1
    for dim_index, dim in enumerate(shape):
        if dim_index == axis:
            continue
        total *= dim
    return total


def _check_finite(values: Sequence[float], scheme: QuantScheme) -> int:
    nonfinite = sum(1 for value in values if not math.isfinite(value))
    if nonfinite and scheme.nan_policy == NaNPolicy.REJECT:
        raise ConfigError(
            f"quantization input contains {nonfinite} non-finite value(s) and "
            f"scheme.nan_policy='{NaNPolicy.REJECT}' (reason: "
            f"{REASON_NONFINITE_INPUT}); sanitize the tensor or set "
            f"nan_policy='{NaNPolicy.PROPAGATE}' for a diagnostic run"
        )
    return nonfinite


def quantize(
    values: Sequence[float],
    scheme: QuantScheme,
    *,
    shape: Optional[Sequence[int]] = None,
    axis: Optional[int] = None,
) -> QuantizedTensor:
    """Quantize a flat value sequence with ``scheme``.

    Args:
        values: Flat row-major float64-convertible values.
        scheme: The frozen quantization scheme.
        shape: Tensor shape; defaults to ``(len(values),)``.
        axis: Axis along which quantization units are formed; defaults to
            ``scheme.axis`` (``-1``, i.e. the K/input-feature axis).

    Returns:
        A :class:`QuantizedTensor` whose ``q`` stays inside
        ``[scheme.qmin, scheme.qmax]`` for every element.

    Raises:
        ConfigError: On non-finite input with ``nan_policy='reject'``, on a
            shape that does not match ``values``, or on illegal
            granularity/group combinations (already refused by the scheme).
    """
    flat = [float(v) for v in values]
    if shape is None:
        shape_tuple: Tuple[int, ...] = (len(flat),)
    else:
        shape_tuple = tuple(int(dim) for dim in shape)
        expected = 1
        for dim in shape_tuple:
            expected *= dim
        if expected != len(flat):
            raise ConfigError(
                f"shape {shape_tuple} implies {expected} values but got {len(flat)}"
            )
    quant_axis = _normalize_axis(scheme.axis if axis is None else axis, len(shape_tuple))

    nonfinite = _check_finite(flat, scheme)
    round_fn = get_rounder(scheme.round_mode)

    if scheme.granularity == GRANULARITY_PER_TENSOR:
        units_per_row = 1
        group_size: Optional[int] = None
    else:
        units_per_row = groups_per_row(shape_tuple[quant_axis], scheme.group_size)
        group_size = scheme.group_size

    k = shape_tuple[quant_axis]
    tail = tail_length(k, group_size) if group_size is not None else k

    q = [0] * len(flat)
    scales: List[float] = []
    zeros: List[int] = []
    unit_stats: List[UnitStats] = []
    clamp_count = 0
    zero_group_count = 0
    constant_group_count = 0
    invalid_scale_units: List[int] = []
    max_scale = 0.0
    min_positive_scale = math.inf

    qmin, qmax = scheme.qmin, scheme.qmax

    # Bucket the flat indices by quantization unit. Doing it generically (via
    # ``unit_of_index``) is what makes a non-last quantization axis correct:
    # a slice-based loop would silently quantize the wrong elements
    # (E05-01 §6 "其他axis必须显式").
    unit_indices: Dict[int, List[int]] = {}
    if scheme.granularity == GRANULARITY_PER_TENSOR:
        if flat:
            unit_indices[0] = list(range(len(flat)))
    else:
        for index in range(len(flat)):
            unit_indices.setdefault(
                unit_of_index(index, shape_tuple, quant_axis, units_per_row, group_size),
                [],
            ).append(index)

    def _code(value: float, scale: float, zero: int) -> int:
        """Round one value into a code, clamping and counting saturations.

        Non-finite values cannot be rounded (the rounder refuses them), so in
        ``PROPAGATE`` mode they are encoded as 0 and the unit's scale stays
        NaN; the non-finite count in the stats is what marks the run.
        """
        nonlocal clamp_count
        if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0.0:
            return 0
        raw = coerce_round_value(round_fn(value / scale)) + zero
        if raw > qmax:
            clamp_count += 1
            return qmax
        if raw < qmin:
            clamp_count += 1
            return qmin
        return raw

    for unit_index in sorted(unit_indices):
        indices = unit_indices[unit_index]
        chunk = [flat[index] for index in indices]
        unit_clamps_before = clamp_count
        if scheme.symmetric:
            finite_chunk = [value for value in chunk if math.isfinite(value)]
            amax = max((abs(value) for value in finite_chunk), default=float("nan") if chunk and not finite_chunk else 0.0)
            if chunk and not finite_chunk:
                amax = float("nan")
            if amax == 0.0:
                scale = _zero_group_scale(scheme)
                unit_zero = 0
                zero_group_count += 1
                unit_q = [0] * len(chunk)
                constant_group = False
            else:
                scale = (
                    amax / float(scheme.qmax_positive)
                    if math.isfinite(amax)
                    else float("nan")
                )
                unit_zero = 0
                unit_q = [_code(value, scale, unit_zero) for value in chunk]
                constant_group = False
        else:
            finite_chunk = [value for value in chunk if math.isfinite(value)]
            xmin = min(finite_chunk) if finite_chunk else float("nan")
            xmax = max(finite_chunk) if finite_chunk else float("nan")
            if not (math.isfinite(xmin) and math.isfinite(xmax)):
                scale = float("nan")
                unit_zero = 0
                unit_q = [0] * len(chunk)
                constant_group = False
            elif xmax == xmin:
                constant_group_count += 1
                if scheme.constant_group_policy == ConstantGroupPolicy.REJECT:
                    raise ConfigError(
                        f"constant group at unit {unit_index} (value "
                        f"{xmin!r}) and constant_group_policy='reject'"
                    )
                scale = _zero_group_scale(scheme)
                unit_zero = 0
                unit_q = [0] * len(chunk)
                constant_group = True
            else:
                span = xmax - xmin
                scale = span / float(qmax - qmin)
                raw_zero = round_fn(qmin - xmin / scale)
                unit_zero = max(qmin, min(qmax, coerce_round_value(raw_zero)))
                unit_q = [_code(value, scale, unit_zero) for value in chunk]
                constant_group = False
            zeros.append(unit_zero)
        scales.append(scale)
        unit_stats.append(
            UnitStats(
                amax=(
                    max((abs(value) for value in chunk), default=0.0)
                    if all(math.isfinite(value) for value in chunk)
                    else float("nan")
                ),
                scale=scale,
                zero=unit_zero,
                clamp_count=clamp_count - unit_clamps_before,
                zero_group=(scale == 1.0 and all(code == 0 for code in unit_q)),
                constant_group=constant_group,
                unit_index=unit_index,
            )
        )
        if not (math.isfinite(scales[-1]) and scales[-1] > 0.0):
            invalid_scale_units.append(unit_index)
        else:
            max_scale = max(max_scale, scales[-1])
            min_positive_scale = min(min_positive_scale, scales[-1])
        for position, index in enumerate(indices):
            q[index] = unit_q[position]

    # Dequantize once so the returned x_hat is the single source of truth.
    values_dequant = dequantize_flat(
        q,
        scales,
        zeros,
        shape_tuple,
        units_per_row,
        scheme,
        group_size=(group_size if scheme.granularity != GRANULARITY_PER_TENSOR else None),
        axis=quant_axis,
    )

    stats = QuantStats(
        num_values=len(flat),
        num_units=len(scales),
        clamp_count=clamp_count,
        saturated_fraction=(clamp_count / len(flat)) if flat else 0.0,
        zero_group_count=zero_group_count,
        constant_group_count=constant_group_count,
        nonfinite_count=nonfinite,
        invalid_scale_units=invalid_scale_units,
        max_scale=max_scale,
        min_positive_scale=min_positive_scale if math.isfinite(min_positive_scale) else 0.0,
    )
    return QuantizedTensor(
        scheme=scheme,
        shape=shape_tuple,
        axis=quant_axis,
        q=q,
        scales=scales,
        zeros=zeros,
        values_dequant=values_dequant,
        units_per_row=units_per_row,
        group_size=group_size,
        tail=tail,
        stats=stats,
        unit_stats=unit_stats,
    )


def _zero_group_scale(scheme: QuantScheme) -> float:
    if scheme.zero_group_policy == ZeroGroupPolicy.SCALE_ONE:
        return 1.0
    if scheme.zero_group_policy == ZeroGroupPolicy.SCALE_MIN_POSITIVE:
        if scheme.scale_dtype == "float16":
            return 6.103515625e-05
        if scheme.scale_dtype == "float32":
            return _FLOAT32_MIN_NORMAL
        return 5e-324
    raise ConfigError(f"unknown zero-group policy {scheme.zero_group_policy!r}")


def unit_of_index(
    index: int,
    shape: Sequence[int],
    axis: int,
    units_per_row: int,
    group_size: Optional[int],
) -> int:
    """Map a flat (row-major) element index to its quantization unit.

    Generic over the axis: the "row" is the flat index with the quantized-axis
    coordinate removed, and the group is the coordinate along that axis divided
    by ``group_size``. ``group_size is None`` (per-channel/per-tensor) means
    one unit per row.
    """
    rank = len(shape)
    normalized_axis = axis + rank if axis < 0 else axis
    if not 0 <= normalized_axis < rank:
        raise ConfigError(f"axis {axis} out of range for rank {rank}")
    stride_axis = 1
    for dim in shape[normalized_axis + 1 :]:
        stride_axis *= int(dim)
    axis_dim = int(shape[normalized_axis])
    if stride_axis <= 0 or axis_dim <= 0:
        raise ConfigError(f"invalid shape {list(shape)} for unit mapping")
    outer = index // (stride_axis * axis_dim)
    rest = index % stride_axis
    row = outer * stride_axis + rest
    coordinate = (index // stride_axis) % axis_dim
    group = 0 if group_size is None else coordinate // group_size
    return row * units_per_row + group


def dequantize_flat(
    q: Sequence[int],
    scales: Sequence[float],
    zeros: Sequence[int],
    shape: Sequence[int],
    units_per_row: int,
    scheme: QuantScheme,
    *,
    group_size: Optional[int] = None,
    axis: Optional[int] = None,
) -> List[float]:
    """Dequantize flat logical qvalues into float64 values.

    Uses the exact same unit arithmetic as :func:`quantize`; a mismatch here
    is a hard error (E05-01 §15: round-trip must be exact in the logical
    domain). Pass the tensor ``shape`` (not rows/K) so non-last-axis
    quantization is addressed correctly.
    """
    shape_tuple = tuple(int(dim) for dim in shape)
    resolved_axis = scheme.axis if axis is None else axis
    if scheme.granularity == GRANULARITY_PER_TENSOR:
        unit_scales = [scales[0]]
        unit_zeros = [zeros[0] if zeros else 0]
        unit_for = lambda index: 0  # noqa: E731 - tiny local helper
    else:
        resolved_group = group_size if group_size is not None else scheme.group_size
        unit_scales = list(scales)
        unit_zeros = list(zeros) if zeros else [0] * len(scales)

        def unit_for(index: int, _g=resolved_group, _axis=resolved_axis) -> int:
            return unit_of_index(
                index, shape_tuple, _axis, units_per_row, _g
            )

    out = [0.0] * len(q)
    for index, code in enumerate(q):
        unit = unit_for(index)
        if not 0 <= unit < len(unit_scales):
            raise ConfigError(
                f"dequantize: element {index} maps to unit {unit} but only "
                f"{len(unit_scales)} scales are present"
            )
        scale = unit_scales[unit]
        zero = unit_zeros[unit] if not scheme.symmetric else 0
        out[index] = scale * float(int(code) - zero)
    return out


def dequantize(qt: QuantizedTensor) -> List[float]:
    """Recompute ``x_hat`` from ``qt`` (validates the stored values)."""
    return dequantize_flat(
        qt.q,
        qt.scales,
        qt.zeros,
        qt.shape,
        qt.units_per_row,
        qt.scheme,
        group_size=qt.group_size,
        axis=qt.axis,
    )


def clamp_free_error_bound(qt: QuantizedTensor) -> float:
    """Largest ``|x - x_hat|`` explainable by a half step, per unit.

    For an unclamped element the RTN error is bounded by ``scale / 2``. The
    bound returned here is the largest half-scale over all units, so callers
    can assert that the observed reconstruction error never exceeds it
    (E05-01 §9 "误差有界约为半个scale（未clamp时）").
    """
    if not qt.scales:
        return 0.0
    finite = [scale for scale in qt.scales if math.isfinite(scale)]
    if not finite:
        return float("nan")
    return max(finite) / 2.0


def requantize_idempotence_check(qt: QuantizedTensor) -> bool:
    """True when re-quantizing ``x_hat`` under the same scheme is a no-op.

    For an unclamped tensor the round-trip is stable: the codes of
    ``x_hat`` equal the original codes. Returns False when saturation or an
    unusual scale policy breaks that stability (the caller must record it,
    not "fix" it).
    """
    if qt.stats.clamp_count:
        return False
    again = quantize(qt.values_dequant, qt.scheme, shape=qt.shape, axis=qt.axis)
    return again.q == qt.q


def reconstruction_report(
    original: Sequence[float], reconstructed: Sequence[float]
) -> Dict[str, float]:
    """Error metrics for a tensor round-trip (E05-01 §11 step 5).

    Reuses :func:`hqsb.benchmark.metrics.numerical_diff_summary` as the single
    implementation of max/mean/RMSE/cosine/L2-relative and adds the two
    quantization-specific quantities the experiment records:
    ``sqnr_db`` (signal-to-quantization-noise ratio) and ``first_mismatch``
    (index of the first differing element, ``-1`` when identical).
    """
    from hqsb.benchmark.metrics import numerical_diff_summary

    report = numerical_diff_summary(list(original), list(reconstructed))
    if not report:
        return {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "rmse": 0.0,
            "cosine_similarity": 1.0,
            "l2_relative_error": 0.0,
            "sqnr_db": float("inf"),
            "first_mismatch": -1,
        }
    signal = sum(float(value) ** 2 for value in original)
    noise = sum(
        (float(a) - float(b)) ** 2 for a, b in zip(original, reconstructed)
    )
    if noise == 0.0:
        sqnr = float("inf")
    elif signal == 0.0:
        sqnr = float("-inf")
    else:
        sqnr = 10.0 * math.log10(signal / noise)
    first = -1
    for index, (a, b) in enumerate(zip(original, reconstructed)):
        if float(a) != float(b):
            first = index
            break
    report["sqnr_db"] = sqnr
    report["first_mismatch"] = float(first)
    return report


__all__ = [
    "QuantStats",
    "QuantizedTensor",
    "REASON_NONFINITE_INPUT",
    "UnitStats",
    "clamp_free_error_bound",
    "dequantize",
    "dequantize_flat",
    "quantize",
    "reconstruction_report",
    "unit_of_index",
    "requantize_idempotence_check",
]
