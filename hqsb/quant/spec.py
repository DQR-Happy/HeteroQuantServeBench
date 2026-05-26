"""The frozen quantization scheme (E05-01 §1/§3/§4/§6/§7).

"INT4" alone fixes nothing: integer range, round mode, group axis, tail
handling, zero-group policy, scale dtype and packing order are all part of
the artifact identity. This module is the **single place** where those
choices are declared, validated and hashed; every other module (reference
math, packing, artifact, adapters, kernels) consumes a
:class:`QuantScheme` instead of re-deriving the policy.

Frozen choices (E05-01 §1 "range/round/constant 政策唯一"):

* ``RangePolicy.SYMMETRIC`` — ``qmin = -(2^(b-1) - 1)``, ``qmax = 2^(b-1) - 1``
  (drops the most negative code so positive and negative magnitudes are
  symmetric).
* ``RangePolicy.TWOS_COMPLEMENT`` — full two's-complement
  ``qmin = -2^(b-1)``, ``qmax = 2^(b-1) - 1``.
* round mode — from :mod:`hqsb.quant.rounding`; ``nearest_even`` by default.
* zero/constant group policy — explicit, versioned: a zero (all-``0``) group
  gets ``scale = 1.0`` and ``q = 0``; a constant non-zero group is encoded by
  the asymmetric path with ``scale`` from ``(xmax - xmin)`` when that is
  positive, otherwise the constant group policy applies. Both policies are
  recorded in the artifact, never inferred at load time.
* NaN/Inf — rejected by default (:data:`NaNPolicy.REJECT`); ``PROPAGATE`` is
  available for diagnostics only and labels the resulting tensor as
  non-finite.
* packing — logical qvalues are *not* packed in the canonical layer; the
  canonical serialization stores one byte per logical value (see
  :mod:`hqsb.quant.packing`). ``pack_version`` names the packed variant
  family requested for execution.

The scheme is a value object: two schemes are equal iff every frozen field
is equal, and :meth:`QuantScheme.scheme_hash` is stable across processes and
independent of volatile fields (no timestamps, no paths).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from hqsb.core.errors import ConfigError
from hqsb.quant.rounding import ROUND_NEAREST_EVEN, RTN_MODES, is_rtn

#: Granularity names (E05-01 §6).
GRANULARITY_PER_TENSOR = "per_tensor"
GRANULARITY_PER_CHANNEL = "per_channel"
GRANULARITY_PER_GROUP = "per_group"
GRANULARITIES = (
    GRANULARITY_PER_TENSOR,
    GRANULARITY_PER_CHANNEL,
    GRANULARITY_PER_GROUP,
)


class RangePolicy:
    """Signed integer range policies (E05-01 §3, options S1/S2)."""

    SYMMETRIC = "symmetric"
    TWOS_COMPLEMENT = "twos_complement"

    ALL = (SYMMETRIC, TWOS_COMPLEMENT)


class ZeroGroupPolicy:
    """Policy for a group whose ``amax`` (or ``xmax - xmin``) is zero."""

    #: scale := 1.0, q := 0 (stable, avoids 0-division). HQSB default.
    SCALE_ONE = "zero_group_scale_one"
    #: scale := smallest normal positive value of the scale dtype.
    SCALE_MIN_POSITIVE = "zero_group_scale_min_positive"

    ALL = (SCALE_ONE, SCALE_MIN_POSITIVE)


class ConstantGroupPolicy:
    """Policy for an asymmetric group with ``xmax == xmin`` (all-equal)."""

    #: Encode with ``scale`` from the chosen fallback and ``q == 0``.
    ENCODE_ZERO_CODE = "constant_group_zero_code"
    #: Refuse the group (caller must handle the constant tensor explicitly).
    REJECT = "constant_group_reject"

    ALL = (ENCODE_ZERO_CODE, REJECT)


class NaNPolicy:
    """NaN/Inf policy for quantization *inputs*."""

    REJECT = "reject"
    PROPAGATE = "propagate"

    ALL = (REJECT, PROPAGATE)


#: Scale/zero storage dtypes (E05-01 §8 "scale/zp dtype").
SCALE_DTYPES = ("float16", "float32", "float64")
ZERO_DTYPES = ("int8", "int16", "int32")

#: The default compute dtype of the reference implementation.
REFERENCE_COMPUTE_DTYPE = "float64"


def integer_range(bits: int, policy: str) -> Tuple[int, int]:
    """Return ``(qmin, qmax)`` for ``bits`` and a range policy.

    Raises:
        ConfigError: If ``bits`` is out of ``[1, 8]`` or ``policy`` is not one
            of :data:`RangePolicy.ALL`.
    """
    if not isinstance(bits, int) or isinstance(bits, bool):
        raise ConfigError(f"bits must be an int, got {type(bits).__name__}")
    if not 1 <= bits <= 8:
        raise ConfigError(f"bits must be in [1, 8], got {bits}")
    if policy == RangePolicy.SYMMETRIC:
        qmax = (1 << (bits - 1)) - 1
        return (-qmax, qmax)
    if policy == RangePolicy.TWOS_COMPLEMENT:
        return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    raise ConfigError(
        f"unknown range policy {policy!r}; supported: {list(RangePolicy.ALL)}"
    )


def groups_per_row(k: int, group_size: Optional[int]) -> int:
    """Number of quantization units along a ``K`` axis (E05-01 §6).

    ``group_size is None`` means "one unit per row" (per-channel). ``G > K``
    yields exactly one (partially filled) group — the tail is *not* dropped
    and is *not* padded with data from another row.
    """
    if k < 0:
        raise ConfigError(f"K must be non-negative, got {k}")
    if group_size is None:
        return 1 if k > 0 else 0
    if group_size <= 0:
        raise ConfigError(f"group_size must be positive, got {group_size}")
    if k == 0:
        return 0
    return (k + group_size - 1) // group_size


def tail_length(k: int, group_size: Optional[int]) -> int:
    """Valid length of the last group along ``K`` (``0`` when ``K == 0``).

    ``valid = K - floor(K/G)*G``; when ``K % G == 0`` the tail is a full
    ``G`` (i.e. the last group is full). For ``G > K`` the tail is ``K``.
    """
    if k < 0:
        raise ConfigError(f"K must be non-negative, got {k}")
    if group_size is None:
        return k
    if group_size <= 0:
        raise ConfigError(f"group_size must be positive, got {group_size}")
    if k == 0:
        return 0
    remainder = k % group_size
    return remainder if remainder != 0 else min(k, group_size)


@dataclass(frozen=True)
class QuantScheme:
    """The frozen, versioned quantization scheme.

    The default instance is HQSB's canonical reference scheme: 8-bit,
    symmetric, per-channel, ``nearest_even`` round, ``float32`` scales,
    reject-NaN.
    """

    bits: int = 8
    range_policy: str = RangePolicy.SYMMETRIC
    symmetric: bool = True
    granularity: str = GRANULARITY_PER_CHANNEL
    #: Axis along which groups are formed. ``-1`` means the last axis (the
    #: ``K``/input-feature axis of a ``[O, K]`` weight); E05-01 §6 requires
    #: the axis to be explicit in the artifact.
    axis: int = -1
    group_size: Optional[int] = None
    round_mode: str = ROUND_NEAREST_EVEN
    scale_dtype: str = "float32"
    zero_dtype: str = "int8"
    zero_group_policy: str = ZeroGroupPolicy.SCALE_ONE
    constant_group_policy: str = ConstantGroupPolicy.ENCODE_ZERO_CODE
    nan_policy: str = NaNPolicy.REJECT
    compute_dtype: str = REFERENCE_COMPUTE_DTYPE
    #: Name of the packed-variant family requested for execution. The
    #: canonical layer is always one byte per logical value; this field only
    #: names the kernel layout that must exist for a low-bit *execution*
    #: claim (E05-01 §7).
    pack_version: str = "hqsb.nibble.v1"
    #: Human-facing label; part of the identity (two runs with different
    #: labels are different artifacts) but never a substitute for the fields.
    label: str = "rtn"

    def __post_init__(self) -> None:
        self.validate()

    # ── validation ─────────────────────────────────────────────────────

    def validate(self) -> None:
        """Validate every frozen field; raise :class:`ConfigError` on any
        illegal or ambiguous combination.

        This is deliberately strict: E05-01 §4 (step 4) requires illegal
        combinations to be *pre-registered as expected rejects*, so the spec
        must refuse them before any tensor is touched.
        """
        integer_range(self.bits, self.range_policy)

        if self.granularity not in GRANULARITIES:
            raise ConfigError(
                f"unknown granularity {self.granularity!r}; "
                f"supported: {list(GRANULARITIES)}"
            )

        if self.round_mode not in RTN_MODES and not is_rtn(self.round_mode):
            raise ConfigError(
                f"round mode {self.round_mode!r} is unknown; supported: "
                f"{list(RTN_MODES)}"
            )

        if self.scale_dtype not in SCALE_DTYPES:
            raise ConfigError(
                f"unknown scale dtype {self.scale_dtype!r}; supported: "
                f"{list(SCALE_DTYPES)}"
            )
        if self.zero_dtype not in ZERO_DTYPES:
            raise ConfigError(
                f"unknown zero dtype {self.zero_dtype!r}; supported: "
                f"{list(ZERO_DTYPES)}"
            )
        if self.zero_group_policy not in ZeroGroupPolicy.ALL:
            raise ConfigError(
                f"unknown zero-group policy {self.zero_group_policy!r}; "
                f"supported: {list(ZeroGroupPolicy.ALL)}"
            )
        if self.constant_group_policy not in ConstantGroupPolicy.ALL:
            raise ConfigError(
                f"unknown constant-group policy {self.constant_group_policy!r}; "
                f"supported: {list(ConstantGroupPolicy.ALL)}"
            )
        if self.nan_policy not in NaNPolicy.ALL:
            raise ConfigError(
                f"unknown NaN policy {self.nan_policy!r}; "
                f"supported: {list(NaNPolicy.ALL)}"
            )

        if self.granularity == GRANULARITY_PER_TENSOR:
            if self.group_size is not None:
                raise ConfigError(
                    "granularity 'per_tensor' must not declare a group_size; "
                    "an unused group_size is an ambiguous artifact"
                )
        elif self.granularity == GRANULARITY_PER_CHANNEL:
            if self.group_size is not None:
                raise ConfigError(
                    "granularity 'per_channel' must not declare a group_size; "
                    "use 'per_group' with group_size == K for a full row unit"
                )
        else:  # per_group
            if self.group_size is None:
                raise ConfigError(
                    "granularity 'per_group' requires an explicit group_size"
                )
            if self.group_size <= 0:
                raise ConfigError(
                    f"group_size must be positive, got {self.group_size}"
                )

        if self.symmetric:
            if self.range_policy != RangePolicy.SYMMETRIC:
                # A symmetric scheme with a two's-complement range would put
                # different magnitudes on +qmax and -qmin; E05-01 §3 allows
                # both, but the combination must be declared intentionally.
                raise ConfigError(
                    "symmetric=True with range_policy='twos_complement' is "
                    "ambiguous (magnitudes differ per sign); either use "
                    "range_policy='symmetric' or set symmetric=False"
                )
        else:
            if self.range_policy != RangePolicy.TWOS_COMPLEMENT:
                raise ConfigError(
                    "asymmetric quantization uses the full integer range; set "
                    "range_policy='twos_complement'"
                )

        if not self.compute_dtype.startswith("float"):
            raise ConfigError(
                f"compute_dtype must be a floating dtype, got {self.compute_dtype!r}"
            )
        if not isinstance(self.label, str) or not self.label:
            raise ConfigError("label must be a non-empty string")

    # ── derived quantities ─────────────────────────────────────────────

    @property
    def qmin(self) -> int:
        return integer_range(self.bits, self.range_policy)[0]

    @property
    def qmax(self) -> int:
        return integer_range(self.bits, self.range_policy)[1]

    @property
    def qmax_positive(self) -> int:
        """Positive full-scale code used for symmetric scale computation."""
        return (1 << (self.bits - 1)) - 1

    @property
    def levels(self) -> int:
        return self.qmax - self.qmin + 1

    @property
    def stores_zero(self) -> bool:
        """True when the scheme persists zero points (asymmetric only)."""
        return not self.symmetric

    def unit_count(self, k: int) -> int:
        """Number of quantization units along ``K``."""
        return groups_per_row(k, self.group_size)

    def with_overrides(self, **kwargs: Any) -> "QuantScheme":
        """Return a copy with ``kwargs`` replaced (re-validated)."""
        return replace(self, **kwargs)

    def as_dict(self) -> Dict[str, Any]:
        """Deterministic field mapping (sorted keys on serialization)."""
        return {
            "bits": self.bits,
            "range_policy": self.range_policy,
            "symmetric": self.symmetric,
            "granularity": self.granularity,
            "axis": self.axis,
            "group_size": self.group_size,
            "round_mode": self.round_mode,
            "scale_dtype": self.scale_dtype,
            "zero_dtype": self.zero_dtype,
            "zero_group_policy": self.zero_group_policy,
            "constant_group_policy": self.constant_group_policy,
            "nan_policy": self.nan_policy,
            "compute_dtype": self.compute_dtype,
            "pack_version": self.pack_version,
            "label": self.label,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "QuantScheme":
        """Build a scheme from a mapping; unknown keys are rejected."""
        known = set(cls().as_dict())
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(
                f"unknown QuantScheme field(s): {sorted(unknown)}; "
                f"known: {sorted(known)}"
            )
        return cls(**dict(payload))

    def scheme_hash(self) -> str:
        """Stable SHA256 over the frozen scheme fields (identity only)."""
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


#: HQSB's main W8 weight-only scheme (E05-02 primary W8 configuration).
MAIN_W8 = QuantScheme(bits=8, label="rtn_w8_per_channel")
#: HQSB's main W4 weight-only scheme (E05-02 primary W4 configuration).
MAIN_W4 = QuantScheme(bits=4, label="rtn_w4_per_group")
MAIN_W4_GROUP = QuantScheme(
    bits=4, granularity=GRANULARITY_PER_GROUP, group_size=128, label="rtn_w4_g128"
)


@dataclass(frozen=True)
class FactorialCase:
    """One cell of the E05-01 §11-step-4 factorial matrix.

    Attributes:
        payload: The ``QuantScheme`` constructor arguments.
        expected_reject: ``None`` for a legal cell, otherwise a reason code
            that *pre-registers* the cell as illegal. The driver must assert
            that :meth:`resolve` raises :class:`ConfigError` for such a cell —
            an illegal cell that silently succeeds is a spec bug
            (E05-01 §11 step 4: "非法组合预先标 expected reject").
        group_size_expression: ``None``, ``"K"`` or ``"K+1"``; shape-relative
            group sizes need a concrete ``K`` and are resolved by
            :meth:`resolve`.
    """

    payload: Dict[str, Any]
    expected_reject: Optional[str] = None
    group_size_expression: Optional[str] = None

    def resolve(self, k: Optional[int] = None) -> QuantScheme:
        """Materialize the scheme, substituting a concrete ``K`` when needed.

        Raises:
            ConfigError: For an illegal cell, or when a shape-relative group
                size is resolved without ``k``.
        """
        payload = dict(self.payload)
        if self.group_size_expression is not None:
            if k is None:
                raise ConfigError(
                    f"group size expression {self.group_size_expression!r} "
                    f"requires a concrete K"
                )
            if self.group_size_expression == "K":
                payload["group_size"] = k
            elif self.group_size_expression == "K+1":
                payload["group_size"] = k + 1
            else:  # pragma: no cover - defensive
                raise ConfigError(
                    f"unknown group size expression {self.group_size_expression!r}"
                )
        return QuantScheme(**payload)


def _range_for(symmetric: bool) -> str:
    return (
        RangePolicy.SYMMETRIC if symmetric else RangePolicy.TWOS_COMPLEMENT
    )


def factorial_matrix(
    *,
    bits: Tuple[int, ...] = (8, 4),
    granularities: Tuple[str, ...] = (
        GRANULARITY_PER_TENSOR,
        GRANULARITY_PER_CHANNEL,
        GRANULARITY_PER_GROUP,
    ),
    symmetric_values: Tuple[bool, ...] = (True, False),
    group_sizes: Tuple[Optional[int], ...] = (1, 16, 32, 64, 128),
    shape_relative_group_sizes: Tuple[str, ...] = ("K", "K+1"),
) -> "list[FactorialCase]":
    """Enumerate the E05-01 §11-step-4 factorial matrix.

    Covers ``bit x granularity x symmetric x G`` with ``G`` in
    ``{1,16,32,64,128,K,K+1}`` plus the pre-registered illegal cells
    (per-group without a size; symmetric range combined with
    two's-complement). Shape-relative group sizes carry
    ``group_size_expression`` and are materialized by
    :meth:`FactorialCase.resolve`.
    """
    out: "list[FactorialCase]" = []
    for bit in bits:
        for symmetric in symmetric_values:
            for granularity in granularities:
                if granularity != GRANULARITY_PER_GROUP:
                    out.append(
                        FactorialCase(
                            payload={
                                "bits": bit,
                                "granularity": granularity,
                                "symmetric": symmetric,
                                "range_policy": _range_for(symmetric),
                                "label": f"b{bit}_{granularity}_sym{symmetric}",
                            }
                        )
                    )
                    continue
                for group in group_sizes:
                    out.append(
                        FactorialCase(
                            payload={
                                "bits": bit,
                                "granularity": GRANULARITY_PER_GROUP,
                                "symmetric": symmetric,
                                "range_policy": _range_for(symmetric),
                                "group_size": group,
                                "label": f"b{bit}_g{group}_sym{symmetric}",
                            }
                        )
                    )
                for expression in shape_relative_group_sizes:
                    out.append(
                        FactorialCase(
                            payload={
                                "bits": bit,
                                "granularity": GRANULARITY_PER_GROUP,
                                "symmetric": symmetric,
                                "range_policy": _range_for(symmetric),
                                "label": f"b{bit}_{expression}_sym{symmetric}",
                            },
                            group_size_expression=expression,
                        )
                    )
    # Pre-registered illegal cells (must be refused by QuantScheme).
    out.append(
        FactorialCase(
            payload={
                "bits": 4,
                "granularity": GRANULARITY_PER_GROUP,
                "group_size": None,
                "label": "illegal_per_group_without_size",
            },
            expected_reject="per_group_requires_group_size",
        )
    )
    out.append(
        FactorialCase(
            payload={
                "bits": 4,
                "granularity": GRANULARITY_PER_CHANNEL,
                "symmetric": True,
                "range_policy": RangePolicy.TWOS_COMPLEMENT,
                "label": "illegal_symmetric_twos_complement",
            },
            expected_reject="symmetric_range_ambiguous",
        )
    )
    out.append(
        FactorialCase(
            payload={
                "bits": 8,
                "granularity": GRANULARITY_PER_CHANNEL,
                "group_size": 128,
                "label": "illegal_per_channel_with_group_size",
            },
            expected_reject="per_channel_must_not_declare_group_size",
        )
    )
    return out


__all__ = [
    "ConstantGroupPolicy",
    "FactorialCase",
    "GRANULARITIES",
    "GRANULARITY_PER_CHANNEL",
    "GRANULARITY_PER_GROUP",
    "GRANULARITY_PER_TENSOR",
    "MAIN_W4",
    "MAIN_W4_GROUP",
    "MAIN_W8",
    "NaNPolicy",
    "QuantScheme",
    "REFERENCE_COMPUTE_DTYPE",
    "RangePolicy",
    "SCALE_DTYPES",
    "ZERO_DTYPES",
    "ZeroGroupPolicy",
    "factorial_matrix",
    "groups_per_row",
    "integer_range",
    "tail_length",
]
