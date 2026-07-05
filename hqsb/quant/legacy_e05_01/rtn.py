"""A small, explicit Round-To-Nearest weight quantization reference.

This module intentionally uses only the Python standard library.  It is a
semantic oracle and artifact producer, not a fast model execution path.
Every rounding, range, grouping, tail and packing decision is represented in
``RtnSpec`` instead of relying on a framework default.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Iterable, Sequence


SUPPORTED_GRANULARITIES = ("per-tensor", "per-channel", "per-group")


def _float32(value: float) -> float:
    """Round a finite Python float to IEEE-754 little-endian binary32."""

    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as exc:
        raise ValueError("scale overflows float32") from exc
    if not math.isfinite(result):
        raise ValueError("scale must remain finite after float32 conversion")
    return result


def round_nearest_even(value: float) -> int:
    """Return the nearest integer, resolving exact half ties to even.

    The implementation is explicit so the RTN contract does not inherit the
    spelling or conversion behaviour of Python, NumPy, PyTorch or CUDA.
    """

    if not math.isfinite(value):
        raise ValueError("round input must be finite")
    lower = math.floor(value)
    fraction = value - lower
    if fraction < 0.5:
        return int(lower)
    if fraction > 0.5:
        return int(lower + 1)
    return int(lower if lower % 2 == 0 else lower + 1)


@dataclass(frozen=True)
class RtnSpec:
    """Resolved RTN semantics for a logical two-dimensional weight tensor."""

    bits: int
    symmetric: bool
    granularity: str
    group_size: int | None = None
    axis: int = 1
    range_policy: str | None = None
    round_mode: str = "nearest-even"
    zero_group_policy: str = "scale-one-zero-code"
    constant_group_policy: str = "unit-code-exact"
    scale_dtype: str = "float32-le"
    zero_point_dtype: str = "int32-le"
    nan_inf_policy: str = "reject"

    def __post_init__(self) -> None:
        if self.bits not in (4, 8):
            raise ValueError("bits must be exactly 4 or 8")
        if self.granularity not in SUPPORTED_GRANULARITIES:
            raise ValueError(f"unsupported granularity: {self.granularity!r}")
        if self.axis != 1:
            raise ValueError("weight quantization axis must be K/axis=1")
        if self.granularity == "per-group":
            if not isinstance(self.group_size, int) or self.group_size <= 0:
                raise ValueError("per-group quantization needs positive group_size")
        elif self.group_size is not None:
            raise ValueError("group_size is only legal for per-group quantization")
        expected_range = (
            "signed-symmetric-narrow" if self.symmetric else "signed-full"
        )
        if self.range_policy is None:
            object.__setattr__(self, "range_policy", expected_range)
        elif self.range_policy != expected_range:
            raise ValueError(
                f"range_policy must be {expected_range!r} for this scheme"
            )
        if self.round_mode != "nearest-even":
            raise ValueError("only explicit round-to-nearest-even is supported")
        if self.zero_group_policy != "scale-one-zero-code":
            raise ValueError("unsupported zero-group policy")
        if self.constant_group_policy != "unit-code-exact":
            raise ValueError("unsupported constant-group policy")
        if self.scale_dtype != "float32-le":
            raise ValueError("scale dtype must be float32-le")
        if self.zero_point_dtype != "int32-le":
            raise ValueError("zero-point dtype must be int32-le")
        if self.nan_inf_policy != "reject":
            raise ValueError("NaN/Inf policy must be reject")

    @property
    def qmin(self) -> int:
        if self.symmetric:
            return -(2 ** (self.bits - 1) - 1)
        return -(2 ** (self.bits - 1))

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm": "rtn",
            "bits": self.bits,
            "symmetric": self.symmetric,
            "granularity": self.granularity,
            "group_size": self.group_size,
            "axis": self.axis,
            "range_policy": self.range_policy,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "round_mode": self.round_mode,
            "zero_group_policy": self.zero_group_policy,
            "constant_group_policy": self.constant_group_policy,
            "scale_dtype": self.scale_dtype,
            "zero_point_dtype": self.zero_point_dtype,
            "nan_inf_policy": self.nan_inf_policy,
        }


@dataclass(frozen=True)
class QuantizedTensor:
    """Canonical logical qvalues and their exact dequantization metadata."""

    spec: RtnSpec
    original_shape: tuple[int, int]
    qvalues: tuple[int, ...]
    scales: tuple[float, ...]
    zero_points: tuple[int, ...]
    parameter_shape: tuple[int, int]
    group_valid_sizes: tuple[int, ...]
    clamped: tuple[bool, ...]

    def __post_init__(self) -> None:
        rows, cols = self.original_shape
        if rows <= 0 or cols <= 0:
            raise ValueError("logical tensor shape must be non-empty")
        if len(self.qvalues) != rows * cols:
            raise ValueError("qvalue count does not match logical shape")
        if len(self.clamped) != len(self.qvalues):
            raise ValueError("clamp mask does not match logical shape")
        parameter_count = self.parameter_shape[0] * self.parameter_shape[1]
        if len(self.scales) != parameter_count:
            raise ValueError("scale count does not match parameter shape")
        if len(self.zero_points) != parameter_count:
            raise ValueError("zero-point count does not match parameter shape")
        if len(self.group_valid_sizes) != parameter_count:
            raise ValueError("group tail metadata does not match parameter shape")
        if any(q < self.spec.qmin or q > self.spec.qmax for q in self.qvalues):
            raise ValueError("qvalue lies outside the resolved integer range")
        if any(not math.isfinite(scale) or scale <= 0 for scale in self.scales):
            raise ValueError("all scales must be positive and finite")
        if any(
            zero < self.spec.qmin or zero > self.spec.qmax
            for zero in self.zero_points
        ):
            raise ValueError("zero-point lies outside the resolved integer range")
        if self.spec.symmetric and any(self.zero_points):
            raise ValueError("symmetric quantization requires zero-point 0")

    @property
    def clamp_count(self) -> int:
        return sum(self.clamped)


def _normalize_matrix(values: Sequence[Sequence[float]]) -> list[list[float]]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("input must be a non-empty list/tuple of rows")
    rows: list[list[float]] = []
    width: int | None = None
    for source_row in values:
        if not isinstance(source_row, (list, tuple)) or not source_row:
            raise ValueError("each input row must be a non-empty list/tuple")
        row = [float(value) for value in source_row]
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("input matrix must be rectangular")
        if any(not math.isfinite(value) for value in row):
            raise ValueError("NaN and Inf inputs are rejected")
        rows.append(row)
    return rows


def _groups(
    rows: int, cols: int, spec: RtnSpec
) -> tuple[list[list[int]], tuple[int, int]]:
    if spec.granularity == "per-tensor":
        return [list(range(rows * cols))], (1, 1)
    if spec.granularity == "per-channel":
        return [list(range(row * cols, (row + 1) * cols)) for row in range(rows)], (
            rows,
            1,
        )
    assert spec.group_size is not None
    groups_per_row = (cols + spec.group_size - 1) // spec.group_size
    result: list[list[int]] = []
    for row in range(rows):
        for group in range(groups_per_row):
            start = row * cols + group * spec.group_size
            stop = row * cols + min(cols, (group + 1) * spec.group_size)
            result.append(list(range(start, stop)))
    return result, (rows, groups_per_row)


def _quantize_group(
    values: list[float], spec: RtnSpec
) -> tuple[list[int], float, int, list[bool]]:
    qmin, qmax = spec.qmin, spec.qmax
    if spec.symmetric:
        amax = max(abs(value) for value in values)
        if amax == 0.0:
            return [0] * len(values), 1.0, 0, [False] * len(values)
        scale = _float32(amax / qmax)
        if scale <= 0.0:
            raise ValueError("non-zero group scale underflows float32")
        zero = 0
        raw_codes = [round_nearest_even(value / scale) for value in values]
    else:
        xmin, xmax = min(values), max(values)
        if xmax == xmin:
            if xmin == 0.0:
                return [0] * len(values), 1.0, 0, [False] * len(values)
            scale = _float32(abs(xmin))
            if scale <= 0.0:
                raise ValueError("constant group scale underflows float32")
            code = 1 if xmin > 0 else -1
            return [code] * len(values), scale, 0, [False] * len(values)
        scale = _float32((xmax - xmin) / (qmax - qmin))
        if scale <= 0.0:
            raise ValueError("non-constant group scale underflows float32")
        zero_raw = round_nearest_even(qmin - xmin / scale)
        zero = max(qmin, min(qmax, zero_raw))
        raw_codes = [round_nearest_even(value / scale) + zero for value in values]
    clamped = [code < qmin or code > qmax for code in raw_codes]
    qvalues = [max(qmin, min(qmax, code)) for code in raw_codes]
    return qvalues, scale, zero, clamped


def quantize(
    values: Sequence[Sequence[float]], spec: RtnSpec
) -> QuantizedTensor:
    """Quantize a row-major logical ``W[O,K]`` matrix without mutating it."""

    matrix = _normalize_matrix(values)
    rows, cols = len(matrix), len(matrix[0])
    flat = [value for row in matrix for value in row]
    groups, parameter_shape = _groups(rows, cols, spec)
    qvalues = [0] * len(flat)
    clamped = [False] * len(flat)
    scales: list[float] = []
    zeros: list[int] = []
    valid_sizes: list[int] = []
    for indices in groups:
        qgroup, scale, zero, group_clamped = _quantize_group(
            [flat[index] for index in indices], spec
        )
        scales.append(scale)
        zeros.append(zero)
        valid_sizes.append(len(indices))
        for index, code, was_clamped in zip(indices, qgroup, group_clamped):
            qvalues[index] = code
            clamped[index] = was_clamped
    return QuantizedTensor(
        spec=spec,
        original_shape=(rows, cols),
        qvalues=tuple(qvalues),
        scales=tuple(scales),
        zero_points=tuple(zeros),
        parameter_shape=parameter_shape,
        group_valid_sizes=tuple(valid_sizes),
        clamped=tuple(clamped),
    )


def dequantize(tensor: QuantizedTensor) -> list[list[float]]:
    """Reconstruct a logical row-major matrix from canonical qvalues."""

    rows, cols = tensor.original_shape
    groups, parameter_shape = _groups(rows, cols, tensor.spec)
    if parameter_shape != tensor.parameter_shape:
        raise ValueError("artifact parameter shape is inconsistent with its spec")
    flat = [0.0] * (rows * cols)
    for group_index, indices in enumerate(groups):
        scale = tensor.scales[group_index]
        zero = tensor.zero_points[group_index]
        for index in indices:
            flat[index] = scale * (tensor.qvalues[index] - zero)
    return [flat[row * cols : (row + 1) * cols] for row in range(rows)]


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or not all(isinstance(value, int) and value > 0 for value in shape)
    ):
        raise ValueError("shape must be a positive (rows, cols) tuple")
    return shape


def pack_int4(qvalues: Iterable[int], shape: tuple[int, int]) -> bytes:
    """Pack signed INT4 row-major values, byte-aligning every logical row.

    Even K indices occupy the low nibble, odd indices the high nibble.  Signed
    values use two's-complement.  An odd row ends with a zero high nibble.
    """

    rows, cols = _validate_shape(shape)
    values = list(qvalues)
    if len(values) != rows * cols:
        raise ValueError("INT4 value count does not match shape")
    if any(not isinstance(value, int) or value < -8 or value > 7 for value in values):
        raise ValueError("canonical INT4 values must lie in [-8, 7]")
    result = bytearray()
    for row in range(rows):
        offset = row * cols
        for column in range(0, cols, 2):
            low = values[offset + column] & 0xF
            high = values[offset + column + 1] & 0xF if column + 1 < cols else 0
            result.append(low | (high << 4))
    return bytes(result)


def unpack_int4(
    packed: bytes, shape: tuple[int, int], *, reject_nonzero_padding: bool = True
) -> tuple[int, ...]:
    """Unpack canonical signed INT4 and validate odd-row padding."""

    rows, cols = _validate_shape(shape)
    expected = rows * ((cols + 1) // 2)
    if len(packed) != expected:
        raise ValueError(
            f"packed INT4 byte length {len(packed)} does not match {expected}"
        )
    values: list[int] = []
    byte_index = 0
    for _row in range(rows):
        for column in range(0, cols, 2):
            byte = packed[byte_index]
            byte_index += 1
            low = byte & 0xF
            high = (byte >> 4) & 0xF
            values.append(low - 16 if low >= 8 else low)
            if column + 1 < cols:
                values.append(high - 16 if high >= 8 else high)
            elif reject_nonzero_padding and high != 0:
                raise ValueError("odd INT4 row has non-zero padding nibble")
    return tuple(values)


def pack_int8(qvalues: Iterable[int], shape: tuple[int, int]) -> bytes:
    rows, cols = _validate_shape(shape)
    values = list(qvalues)
    if len(values) != rows * cols:
        raise ValueError("INT8 value count does not match shape")
    if any(not isinstance(value, int) or value < -128 or value > 127 for value in values):
        raise ValueError("canonical INT8 values must lie in [-128, 127]")
    return struct.pack(f"<{len(values)}b", *values)


def unpack_int8(packed: bytes, shape: tuple[int, int]) -> tuple[int, ...]:
    rows, cols = _validate_shape(shape)
    expected = rows * cols
    if len(packed) != expected:
        raise ValueError(
            f"packed INT8 byte length {len(packed)} does not match {expected}"
        )
    return tuple(struct.unpack(f"<{expected}b", packed))


__all__ = [
    "QuantizedTensor",
    "RtnSpec",
    "dequantize",
    "pack_int4",
    "pack_int8",
    "quantize",
    "round_nearest_even",
    "unpack_int4",
    "unpack_int8",
]
