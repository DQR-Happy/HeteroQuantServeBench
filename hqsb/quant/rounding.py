"""Round modes for HQSB quantization (E05-01 §5).

The round mode is part of the frozen quantization spec: "INT4" alone does not
determine how a tie (``n + 0.5``) or a negative value is rounded, and Python,
NumPy, PyTorch and CUDA do not agree by default. Every HQSB quantization path
must therefore call *these* functions, never a language default.

Frozen policies (``hqsb.quant.spec.ROUND_MODES``):

``nearest_even``
    Round half to even (IEEE 754 default, ``round()`` semantics on exact
    ties). This is the HQSB default.

``half_away_from_zero``
    Round half away from zero (``2.5 -> 3``, ``-2.5 -> -3``).

``trunc`` / ``floor``
    Available only for explicit *diagnostics*: they are **not** round-to-
    nearest and must never be labelled RTN. :func:`get_rounder` refuses to
    hand them out unless the caller passes ``allow_non_rtn=True`` and the
    caller records the deviation (E05-01 §5: "trunc/floor（不应叫 RTN）").

Exact-tie behaviour is defined on the *real number* ``x / scale``. This
module implements the modes on Python floats with exact half detection
(``math.fmod(scaled, 1.0)`` is exact for the values a binary float can
represent), and :mod:`hqsb.quant.golden` cross-checks the tie cases with an
exact-rational implementation so a floating-point surprise cannot silently
redefine the policy.
"""

from __future__ import annotations

import math
from typing import Callable, Dict

from hqsb.core.errors import ConfigError

ROUND_NEAREST_EVEN = "nearest_even"
ROUND_HALF_AWAY_FROM_ZERO = "half_away_from_zero"
ROUND_TRUNC = "trunc"
ROUND_FLOOR = "floor"

#: Modes that are legitimate round-to-nearest implementations.
RTN_MODES = (ROUND_NEAREST_EVEN, ROUND_HALF_AWAY_FROM_ZERO)

#: Modes accepted only as explicitly-labelled diagnostics (never "RTN").
DIAGNOSTIC_MODES = (ROUND_TRUNC, ROUND_FLOOR)

ROUND_MODES = RTN_MODES + DIAGNOSTIC_MODES


def round_nearest_even(value: float) -> float:
    """Round half to even (banker's rounding) on the real value ``value``.

    ``value`` is the already-scaled real number ``x / scale``. Returns a
    float holding an integral value so callers can cast without a second
    rounding decision.
    """
    if not math.isfinite(value):
        raise ConfigError(f"round_nearest_even: non-finite input {value!r}")
    lower = math.floor(value)
    rest = value - lower
    if rest > 0.5:
        return float(lower + 1)
    if rest < 0.5:
        return float(lower)
    # Exact tie: choose the even neighbour. For a binary float, ``value``
    # being exactly ``n + 0.5`` means ``value - lower == 0.5`` exactly.
    return float(lower if lower % 2 == 0 else lower + 1)


def round_half_away_from_zero(value: float) -> float:
    """Round half away from zero: ``2.5 -> 3``, ``-2.5 -> -3``."""
    if not math.isfinite(value):
        raise ConfigError(f"round_half_away_from_zero: non-finite input {value!r}")
    magnitude = abs(value)
    lower = math.floor(magnitude)
    rest = magnitude - lower
    if rest > 0.5:
        rounded = lower + 1
    elif rest < 0.5:
        rounded = lower
    else:
        rounded = lower + 1
    return float(rounded if value >= 0 else -rounded)


def round_trunc(value: float) -> float:
    """Truncate towards zero (diagnostic only; NOT RTN)."""
    if not math.isfinite(value):
        raise ConfigError(f"round_trunc: non-finite input {value!r}")
    return float(math.trunc(value))


def round_floor(value: float) -> float:
    """Floor towards negative infinity (diagnostic only; NOT RTN)."""
    if not math.isfinite(value):
        raise ConfigError(f"round_floor: non-finite input {value!r}")
    return float(math.floor(value))


_ROUNDERS: Dict[str, Callable[[float], float]] = {
    ROUND_NEAREST_EVEN: round_nearest_even,
    ROUND_HALF_AWAY_FROM_ZERO: round_half_away_from_zero,
    ROUND_TRUNC: round_trunc,
    ROUND_FLOOR: round_floor,
}


def coerce_round_value(value) -> int:
    """Return the exact integer index of a rounded value.

    The rounding helpers return floats holding integral values; this helper
    performs the final int conversion with an explicit integrality check so a
    bug in a rounder cannot silently truncate.
    """
    as_float = float(value)
    if not math.isfinite(as_float) or as_float != math.floor(as_float):
        raise ConfigError(f"round result {value!r} is not an exact integer")
    return int(as_float)


def get_rounder(mode: str, *, allow_non_rtn: bool = False) -> Callable[[float], float]:
    """Return the frozen rounder for ``mode``.

    Raises:
        ConfigError: If ``mode`` is unknown, or if a diagnostic (non-RTN) mode
            is requested without ``allow_non_rtn=True``. Refusing the
            diagnostic modes by default prevents a trunc/floor experiment from
            being reported as "RTN".
    """
    if mode not in _ROUNDERS:
        raise ConfigError(
            f"unknown round mode {mode!r}; supported: {sorted(_ROUNDERS)}"
        )
    if mode in DIAGNOSTIC_MODES and not allow_non_rtn:
        raise ConfigError(
            f"round mode {mode!r} is not a round-to-nearest mode and must not "
            f"be labelled RTN; pass allow_non_rtn=True and record the "
            f"deviation explicitly if it is a diagnostic baseline"
        )
    return _ROUNDERS[mode]


def is_rtn(mode: str) -> bool:
    """True when ``mode`` is a legitimate round-to-nearest policy."""
    return mode in RTN_MODES


__all__ = [
    "DIAGNOSTIC_MODES",
    "ROUND_FLOOR",
    "ROUND_HALF_AWAY_FROM_ZERO",
    "ROUND_MODES",
    "ROUND_NEAREST_EVEN",
    "ROUND_TRUNC",
    "RTN_MODES",
    "coerce_round_value",
    "get_rounder",
    "is_rtn",
    "round_floor",
    "round_half_away_from_zero",
    "round_nearest_even",
    "round_trunc",
]
