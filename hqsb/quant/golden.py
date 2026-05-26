"""Golden quantization vectors and an exact-rational cross implementation.

E05-01 §11 step 2/3/8 require more than one oracle:

1. hand-checkable tiny tensors that exercise every integer code, every tie,
   clamping, the zero/constant group policies and tail groups;
2. an independent implementation that must not share code with the production
   reference (:mod:`hqsb.quant.rtn`), so a single helper bug cannot
   co-validate itself;
3. a serialisable "golden test vector package" reusable by E05-04 (adapters),
   E05-06 (kernels) and E05-09 (compatibility).

This module provides all three. The arithmetic here is **exact**: inputs are
:class:`fractions.Fraction` values, scales/zeros/quantized codes are computed
with rational arithmetic, and only the final JSON fixture uses strings. That
makes every tie and every boundary code reproducible without depending on
binary floating-point behaviour.

Nothing in this module runs a model, a kernel or an experiment; it produces
*fixtures* used by unit tests and by the experiment drivers as ground truth.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.quant.rounding import ROUND_HALF_AWAY_FROM_ZERO, ROUND_NEAREST_EVEN
from hqsb.quant.spec import (
    GRANULARITY_PER_GROUP,
    GRANULARITY_PER_TENSOR,
    ConstantGroupPolicy,
    QuantScheme,
    RangePolicy,
    ZeroGroupPolicy,
    groups_per_row,
)


def _round_exact(value: Fraction, mode: str) -> int:
    """Round an exact rational to an integer with the frozen tie policy."""
    floor_value = value.numerator // value.denominator
    remainder = value - floor_value  # in [0, 1)
    half = Fraction(1, 2)
    if mode == ROUND_NEAREST_EVEN:
        if remainder > half:
            return floor_value + 1
        if remainder < half:
            return floor_value
        return floor_value if floor_value % 2 == 0 else floor_value + 1
    if mode == ROUND_HALF_AWAY_FROM_ZERO:
        if value >= 0:
            return floor_value + 1 if remainder >= half else floor_value
        # negative: floor is further from zero
        magnitude = abs(value)
        floor_mag = magnitude.numerator // magnitude.denominator
        rest = magnitude - floor_mag
        rounded = floor_mag + 1 if rest >= half else floor_mag
        return -rounded
    raise ConfigError(f"exact rounding does not support mode {mode!r}")


def _zero_group_scale_exact(scheme: QuantScheme) -> Fraction:
    if scheme.zero_group_policy == ZeroGroupPolicy.SCALE_ONE:
        return Fraction(1)
    if scheme.zero_group_policy == ZeroGroupPolicy.SCALE_MIN_POSITIVE:
        return Fraction(1, 1 << 20)  # exact placeholder, policy recorded
    raise ConfigError(f"unknown zero-group policy {scheme.zero_group_policy!r}")


@dataclass
class ExactQuantResult:
    """Exact-rational quantization result for one (tiny) tensor."""

    q: List[int]
    scales: List[Fraction]
    zeros: List[Optional[int]]
    values_hat: List[Fraction]
    shape: Tuple[int, ...]
    units_per_row: int

    def to_json(self) -> Dict[str, object]:
        return {
            "q": list(self.q),
            "scales": [str(s) for s in self.scales],
            "zeros": [None if z is None else int(z) for z in self.zeros],
            "values_hat": [str(v) for v in self.values_hat],
            "shape": list(self.shape),
            "units_per_row": self.units_per_row,
        }


def quantize_exact(
    values: Sequence[Fraction],
    scheme: QuantScheme,
    *,
    shape: Optional[Sequence[int]] = None,
    axis: int = -1,
) -> ExactQuantResult:
    """Exact-rational reimplementation of the RTN scheme.

    Independent of :mod:`hqsb.quant.rtn` by construction: no shared helper,
    rational arithmetic throughout. It intentionally supports only the
    per-tensor/per-channel/per-group axis semantics needed by the golden
    fixtures (the production reference is the one exercised on real tensors).
    """
    flat = [Fraction(value) for value in values]
    shape_tuple: Tuple[int, ...] = (
        tuple(shape) if shape is not None else (len(flat),)
    )
    expected = 1
    for dim in shape_tuple:
        expected *= dim
    if expected != len(flat):
        raise ConfigError(f"shape {shape_tuple} does not match {len(flat)} values")
    rank = len(shape_tuple)
    quant_axis = axis + rank if axis < 0 else axis
    if not 0 <= quant_axis < rank:
        raise ConfigError(f"axis {axis} out of range for rank {rank}")

    k = shape_tuple[quant_axis]
    rows = 1
    for dim_index, dim in enumerate(shape_tuple):
        if dim_index != quant_axis:
            rows *= dim

    qmin, qmax = scheme.qmin, scheme.qmax
    if scheme.granularity == GRANULARITY_PER_TENSOR:
        units_per_row = 1
        group_size: Optional[int] = None
    else:
        units_per_row = groups_per_row(k, scheme.group_size)
        group_size = scheme.group_size

    q: List[int] = [0] * len(flat)
    scales: List[Fraction] = []
    zeros: List[Optional[int]] = []
    values_hat: List[Fraction] = [Fraction(0)] * len(flat)

    for row in range(rows):
        base = row * k
        for unit in range(units_per_row):
            if scheme.granularity == GRANULARITY_PER_TENSOR:
                start, length = 0, len(flat)
            else:
                start = base + unit * (group_size or k)
                length = min(group_size or k, k - unit * (group_size or k))
            chunk = flat[start : start + length]

            if scheme.symmetric:
                amax = max((abs(value) for value in chunk), default=Fraction(0))
                if amax == 0:
                    scale = _zero_group_scale_exact(scheme)
                    zero: Optional[int] = 0
                    codes = [0] * length
                else:
                    scale = amax / scheme.qmax_positive
                    zero = 0
                    codes = [
                        max(qmin, min(qmax, _round_exact(value / scale, scheme.round_mode)))
                        for value in chunk
                    ]
            else:
                xmin = min(chunk)
                xmax = max(chunk)
                if xmax == xmin:
                    if scheme.constant_group_policy == ConstantGroupPolicy.REJECT:
                        raise ConfigError("exact: constant group rejected by policy")
                    scale = _zero_group_scale_exact(scheme)
                    zero = 0
                    codes = [0] * length
                else:
                    scale = (xmax - xmin) / (qmax - qmin)
                    raw_zero = _round_exact(
                        Fraction(qmin) - xmin / scale, scheme.round_mode
                    )
                    zero = max(qmin, min(qmax, raw_zero))
                    codes = []
                    for value in chunk:
                        code = (
                            _round_exact(value / scale, scheme.round_mode) + zero
                        )
                        codes.append(max(qmin, min(qmax, code)))

            scales.append(scale)
            zeros.append(zero if not scheme.symmetric else None)
            for offset, code in enumerate(codes):
                index = start + offset
                q[index] = code
                if scheme.symmetric:
                    values_hat[index] = scale * code
                else:
                    values_hat[index] = scale * (code - (zero or 0))
    return ExactQuantResult(
        q=q,
        scales=scales,
        zeros=zeros,
        values_hat=values_hat,
        shape=shape_tuple,
        units_per_row=units_per_row,
    )


@dataclass
class GoldenVector:
    """One golden fixture: input, expected codes, expected bytes, hash."""

    name: str
    scheme: QuantScheme
    shape: Tuple[int, ...]
    axis: int
    inputs: Sequence[Fraction]
    expected: ExactQuantResult
    packed_bytes_hex: Optional[str] = None
    notes: str = ""
    tags: Sequence[str] = field(default_factory=tuple)

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "scheme": self.scheme.as_dict(),
            "scheme_hash": self.scheme.scheme_hash(),
            "shape": list(self.shape),
            "axis": self.axis,
            "inputs": [str(value) for value in self.inputs],
            "expected": self.expected.to_json(),
            "packed_bytes_hex": self.packed_bytes_hex,
            "notes": self.notes,
            "tags": list(self.tags),
        }


def _make(
    name: str,
    scheme: QuantScheme,
    values: Sequence[object],
    *,
    shape: Optional[Sequence[int]] = None,
    axis: int = -1,
    notes: str = "",
    tags: Sequence[str] = (),
) -> GoldenVector:
    fractions = [Fraction(str(value)) for value in values]
    expected = quantize_exact(fractions, scheme, shape=shape, axis=axis)
    return GoldenVector(
        name=name,
        scheme=scheme,
        shape=tuple(shape) if shape is not None else (len(fractions),),
        axis=axis,
        inputs=fractions,
        expected=expected,
        notes=notes,
        tags=tuple(tags),
    )


_SYM4 = QuantScheme(bits=4, granularity=GRANULARITY_PER_GROUP, group_size=4)
_SYM8 = QuantScheme(bits=8, granularity=GRANULARITY_PER_TENSOR)
_ASYM8 = QuantScheme(
    bits=8,
    symmetric=False,
    range_policy=RangePolicy.TWOS_COMPLEMENT,
    granularity=GRANULARITY_PER_TENSOR,
    group_size=None,
    label="rtn_asym8",
)


def build_golden_vectors() -> List[GoldenVector]:
    """Return the golden vector package (E05-01 §11 step 12).

    Coverage:

    * every INT4 code ``-8..7`` reachable through symmetric per-group codes;
    * exact halfway ties ``n + 0.5`` / ``-(n + 0.5)`` under both RTN tie
      policies;
    * clamping at ``qmin``/``qmax``;
    * zero group (all zeros) and the ``scale = 1`` policy;
    * asymmetric constant group;
    * tail group where ``K % G != 0`` and ``G > K``;
    * INT8 extremes at ``[-127, 127]``.
    """
    vectors: List[GoldenVector] = []

    # 1. Every symmetric INT4 code -8..7 (G = 8 covers all codes).
    scheme_all_codes = QuantScheme(
        bits=4, granularity=GRANULARITY_PER_GROUP, group_size=8, label="golden_int4_all_codes"
    )
    vectors.append(
        _make(
            "int4_all_codes_symmetric",
            scheme_all_codes,
            [code for code in range(-8, 8)],
            notes="qmax=7: code -8 is unreachable by design (qmin=-7); the "
            "clamped value -8 documents the saturation boundary",
            tags=("packing", "codes", "symmetric"),
        )
    )

    # 2. Halfway ties, nearest-even. Per-tensor with amax = 7 so that
    #    scale == 1 and the scaled values *are* the halfway numbers (a
    #    smaller amax would rescale them away from the tie).
    tie_inputs = [7, 6.5, 3.5, 2.5, 0.5, -0.5, -2.5, -3.5, -6.5, -7]
    tie_scheme_even = QuantScheme(
        bits=4,
        granularity=GRANULARITY_PER_TENSOR,
        round_mode=ROUND_NEAREST_EVEN,
        label="golden_ties_even",
    )
    tie_scheme_away = QuantScheme(
        bits=4,
        granularity=GRANULARITY_PER_TENSOR,
        round_mode=ROUND_HALF_AWAY_FROM_ZERO,
        label="golden_ties_away",
    )
    vectors.append(
        _make(
            "ties_nearest_even",
            tie_scheme_even,
            tie_inputs,
            notes="scale=1: 2.5->2, 3.5->4, -2.5->-2, -3.5->-4 (ties to even)",
            tags=("round", "tie"),
        )
    )

    # 3. Halfway ties, half away from zero.
    vectors.append(
        _make(
            "ties_half_away_from_zero",
            tie_scheme_away,
            tie_inputs,
            notes="scale=1: 2.5->3, 3.5->4, -2.5->-3, -3.5->-4",
            tags=("round", "tie"),
        )
    )

    # 4. Zero group.
    vectors.append(
        _make(
            "zero_group_scale_one",
            _SYM4,
            [0, 0, 0, 0],
            notes="amax == 0 -> scale = 1, q = 0, no NaN",
            tags=("zero-group",),
        )
    )

    # 5. Tail group: K = 6 with G = 4 -> groups [4, 2].
    vectors.append(
        _make(
            "tail_group_k6_g4",
            _SYM4,
            [1, 2, 3, 4, 5, 6],
            shape=(1, 6),
            notes="last group has valid length 2; no padding value is invented",
            tags=("tail",),
        )
    )

    # 6. G > K: one partial group of length K.
    vectors.append(
        _make(
            "group_larger_than_k",
            _SYM4.with_overrides(group_size=16),
            [0.5, -0.5, 1.5],
            shape=(1, 3),
            notes="G=16 > K=3 -> one group, valid length 3",
            tags=("tail", "g_gt_k"),
        )
    )

    # 7. Asymmetric constant group.
    vectors.append(
        _make(
            "asymmetric_constant_group",
            _ASYM8,
            [7, 7, 7, 7],
            notes="xmax == xmin -> constant group policy encodes q = 0",
            tags=("constant-group", "asymmetric"),
        )
    )

    # 8. Asymmetric range mapping including zero.
    vectors.append(
        _make(
            "asymmetric_range",
            _ASYM8,
            [-1, -0.5, 0, 0.5, 1],
            notes="full two's-complement range with an explicit zero point",
            tags=("asymmetric",),
        )
    )

    # 9. INT8 extremes (symmetric [-127, 127]).
    vectors.append(
        _make(
            "int8_extremes",
            _SYM8,
            [-127.0, -1.0, 0.0, 1.0, 127.0],
            notes="scale = 1 -> codes equal values; boundary codes exact",
            tags=("int8", "boundary"),
        )
    )

    # 10. Boundary codes: with scale = amax / qmax_positive the extreme
    #     values land exactly on qmin/qmax, so the clamp counter must stay 0
    #     (clamping is a numerical safety net, not a normal path). The first
    #     row is all-zero and exercises the zero-group policy again, this
    #     time inside a multi-row tensor.
    vectors.append(
        _make(
            "boundary_codes_and_zero_row",
            _SYM4.with_overrides(group_size=4),
            [0.0, 0.0, 0.0, 0.0, 4.0, -4.0, 2.0, -2.0],
            shape=(2, 4),
            notes="row 0 -> zero group (scale 1); row 1 spans [-4, 4] and hits "
            "both boundary codes exactly, clamp_count == 0",
            tags=("boundary", "zero-group"),
        )
    )

    # 11. Constant non-zero symmetric group.
    vectors.append(
        _make(
            "symmetric_constant_group",
            _SYM4,
            [3, 3, 3, 3],
            notes="amax = 3, scale = 3/7, all codes 7",
            tags=("constant-group", "symmetric"),
        )
    )
    return vectors


def golden_package_json() -> str:
    """Deterministic JSON serialization of the whole golden package."""
    vectors = build_golden_vectors()
    payload = {
        "schema_version": "1.0.0",
        "kind": "hqsb.quant.golden",
        "count": len(vectors),
        "vectors": [vector.to_json() for vector in vectors],
    }
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


def golden_package_hash() -> str:
    """SHA256 of the deterministic golden package (fixture identity)."""
    return hashlib.sha256(golden_package_json().encode("utf-8")).hexdigest()


def cross_check_with_reference(fail_fast: bool = True) -> Dict[str, object]:
    """Compare the exact oracle against :mod:`hqsb.quant.rtn`.

    Returns a report ``{"checked": n, "mismatches": [...]}``. The experiment
    driver uses this as a *capability self-check* (E05-01 step 3); it is not
    an experimental result. A mismatch always indicates a spec
    misunderstanding, never something to "tolerate": the two implementations
    share no code.
    """
    from hqsb.quant import rtn

    mismatches: List[Dict[str, object]] = []
    checked = 0
    for vector in build_golden_vectors():
        checked += 1
        reference = rtn.quantize(
            [float(value) for value in vector.inputs],
            vector.scheme,
            shape=vector.shape,
            axis=vector.axis,
        )
        if reference.q != vector.expected.q:
            mismatches.append(
                {
                    "vector": vector.name,
                    "field": "q",
                    "exact": list(vector.expected.q),
                    "reference": list(reference.q),
                }
            )
            continue
        exact_scales = [float(scale) for scale in vector.expected.scales]
        reference_scales = list(reference.scales)
        for index, (exact, got) in enumerate(zip(exact_scales, reference_scales)):
            if not math.isclose(exact, got, rel_tol=1e-12, abs_tol=0.0):
                mismatches.append(
                    {
                        "vector": vector.name,
                        "field": f"scale[{index}]",
                        "exact": exact,
                        "reference": got,
                    }
                )
                break
    if fail_fast and mismatches:
        raise ConfigError(
            f"golden cross-check failed for {len(mismatches)} vector(s): "
            f"{mismatches[:3]}"
        )
    return {"checked": checked, "mismatches": mismatches}


__all__ = [
    "ExactQuantResult",
    "GoldenVector",
    "build_golden_vectors",
    "cross_check_with_reference",
    "golden_package_hash",
    "golden_package_json",
    "quantize_exact",
]
