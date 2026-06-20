"""Host/device Tiling ABI, checked arithmetic, UB budget and tiling cost model.

``TilingData`` is the ABI between the host tiling function and the device kernel
(details README §6.1).  If the field order, width, alignment or version differs
on either side, the kernel silently computes on garbage — so this module owns
*one* definition and generates the C header from it:

* :data:`TILING_FIELDS` is the single source of truth;
* :func:`tiling_header_source` renders the C header that ``ops/ascend/common``
  must contain verbatim, and ``tests/unit/ascend/test_tiling_abi.py`` fails if
  the committed header drifts from the generated one.  That closes the E09-02
  §11 trap "let host and device each keep an unversioned struct".

Everything numeric goes through :func:`checked_mul`/:func:`checked_add`: a
``block_dim`` larger than the work must produce a rejection, not an unsigned
underflow (E09-02 §11, E09-10 step 15).

The UB budget is deliberately **not** the chip's nominal capacity.  Details
README §6.3 requires framework, queue, alignment and implementation reservations
to be subtracted *and the method recorded*, so :class:`UbBudget` carries a
``method`` string that lands in the run artifacts.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError, UsageError
from hqsb.core.fingerprint import canonical_json, sha256_hex

# ── dtype tags ───────────────────────────────────────────────────────────────

DTYPE_BYTES: Mapping[str, int] = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int32": 4,
    "int64": 8,
}

#: Numeric tag written into the ABI so the device can refuse an unknown dtype
#: instead of reinterpreting bytes.
DTYPE_TAGS: Mapping[str, int] = {
    "fp32": 0,
    "fp16": 1,
    "bf16": 2,
    "int8": 3,
    "int32": 4,
    "int64": 5,
}

#: ``int4`` is storage-only in this ABI: sub-byte packing changes the byte model
#: and belongs to E09-06, not to the element-wise tiling path.
SUB_BYTE_DTYPES: Tuple[str, ...] = ("int4",)

#: Tiling keys.  The aligned and tail paths share one OperatorSpec but are
#: separate device code paths; an unknown key must fail (E09-04 step 13).
TILING_KEY_SINGLE_TILE_ALIGNED = 0
TILING_KEY_MULTI_TILE_ALIGNED = 1
TILING_KEY_TAIL_MASKED = 2
TILING_KEY_ROW_ALIGNED = 3
TILING_KEY_ROW_TAIL = 4

TILING_KEYS: Mapping[int, str] = {
    TILING_KEY_SINGLE_TILE_ALIGNED: "single_tile_aligned",
    TILING_KEY_MULTI_TILE_ALIGNED: "multi_tile_aligned",
    TILING_KEY_TAIL_MASKED: "tail_masked",
    TILING_KEY_ROW_ALIGNED: "row_aligned",
    TILING_KEY_ROW_TAIL: "row_tail",
}

TILING_KEY_NAMES: Mapping[str, int] = {name: key for key, name in TILING_KEYS.items()}

#: Index width the device kernel uses for element offsets.  ``total_elements``
#: beyond this must switch to the 64-bit path or be rejected.
INDEX_WIDTH_BITS = 32
MAX_INDEX32 = (1 << 32) - 1

#: Hard ceiling on the serialized TilingData so a corrupt length cannot make the
#: host allocate unbounded memory before the device ever runs.
MAX_TILING_SERIALIZED_BYTES = 4096


# ── checked arithmetic ───────────────────────────────────────────────────────


class ArithmeticOverflow(UsageError):
    """A tiling computation would overflow the device's index/size width."""


def checked_mul(a: int, b: int, *, what: str = "product") -> int:
    """Multiply with an explicit overflow check instead of silent wrap-around."""
    for value, name in ((a, "a"), (b, "b")):
        if not isinstance(value, int):
            raise UsageError(f"{what}: {name} must be an int, got {type(value).__name__}")
        if value < 0:
            raise UsageError(f"{what}: {name} must be non-negative, got {value}")
    result = a * b
    if result > MAX_INDEX32:
        raise ArithmeticOverflow(
            f"{what}: {a} * {b} = {result} exceeds the {INDEX_WIDTH_BITS}-bit device index "
            f"limit {MAX_INDEX32}; use the 64-bit path or reject the shape",
            details={"what": what, "a": a, "b": b, "result": result},
        )
    return result


def checked_add(a: int, b: int, *, what: str = "sum") -> int:
    for value, name in ((a, "a"), (b, "b")):
        if not isinstance(value, int):
            raise UsageError(f"{what}: {name} must be an int, got {type(value).__name__}")
        if value < 0:
            raise UsageError(f"{what}: {name} must be non-negative, got {value}")
    result = a + b
    if result > MAX_INDEX32:
        raise ArithmeticOverflow(
            f"{what}: {a} + {b} = {result} exceeds {MAX_INDEX32}",
            details={"what": what, "a": a, "b": b, "result": result},
        )
    return result


def dtype_bytes(dtype: str) -> int:
    if dtype in SUB_BYTE_DTYPES:
        raise UsageError(
            f"dtype {dtype!r} is sub-byte; the element-wise tiling ABI addresses whole "
            "bytes. Sub-byte packing is an E09-06 concern and must not be smuggled in here",
            details={"dtype": dtype},
        )
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise UsageError(
            f"unknown dtype {dtype!r}; allowed {sorted(DTYPE_BYTES)} (sub-byte {list(SUB_BYTE_DTYPES)} "
            "is rejected on purpose)",
            details={"dtype": dtype},
        ) from None


# ── the ABI ──────────────────────────────────────────────────────────────────

#: ``(c_field_name, c_type, struct_code, python_attr)`` in ABI order.
TILING_FIELDS: Tuple[Tuple[str, str, str, str], ...] = (
    ("schema_version", "uint32_t", "I", "schema_version"),
    ("tiling_key", "uint32_t", "I", "tiling_key"),
    ("dtype_tag", "uint32_t", "I", "dtype_tag"),
    ("block_dim", "uint32_t", "I", "block_dim"),
    ("rows", "uint32_t", "I", "rows"),
    ("hidden", "uint32_t", "I", "hidden"),
    ("tile_elems", "uint32_t", "I", "tile_elems"),
    ("loop_count", "uint32_t", "I", "loop_count"),
    ("tail_elems", "uint32_t", "I", "tail_elems"),
    ("buffer_count", "uint32_t", "I", "buffer_count"),
    ("base_rows", "uint32_t", "I", "base_rows"),
    ("extra_rows", "uint32_t", "I", "extra_rows"),
    ("total_elements", "uint64_t", "Q", "total_elements"),
    ("workspace_bytes", "uint64_t", "Q", "workspace_bytes"),
    ("reserved_0", "uint32_t", "I", "reserved_0"),
    ("reserved_1", "uint32_t", "I", "reserved_1"),
)

#: Little-endian, no implicit padding — the same convention the generated header
#: documents, so ``sizeof`` on the device equals ``calcsize`` here.
TILING_STRUCT_FORMAT = "<" + "".join(code for _c, _t, code, _a in TILING_FIELDS)
TILING_STRUCT_SIZE = struct.calcsize(TILING_STRUCT_FORMAT)

#: Bump when a field is added, removed, reordered or re-typed.  The device kernel
#: refuses an unknown version immediately rather than reading a shifted struct.
TILING_SCHEMA_VERSION = 1

TILING_HEADER_PATH = "ops/ascend/common/hqsb_tiling_data.h"


def tiling_header_source() -> str:
    """Render the C header the device side must include.

    Generated, not hand-maintained: the committed file under
    ``ops/ascend/common`` is compared against this string in the test suite, so
    host and device cannot drift.
    """
    lines = [
        "/* Generated by hqsb/ascend/tiling.py — do not edit by hand.",
        " *",
        " * Host/device TilingData ABI for the HQSB Ascend C operators (S09).",
        " * The byte layout below is the single source of truth shared with",
        " * hqsb.ascend.tiling.TILING_STRUCT_FORMAT; a test fails if this file",
        " * drifts from the generator.",
        " */",
        "#ifndef HQSB_ASCEND_TILING_DATA_H",
        "#define HQSB_ASCEND_TILING_DATA_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define HQSB_TILING_SCHEMA_VERSION {TILING_SCHEMA_VERSION}u",
        f"#define HQSB_TILING_STRUCT_SIZE {TILING_STRUCT_SIZE}u",
        "",
        "/* Little-endian, naturally ordered so that no implicit padding exists:",
        " * the twelve uint32_t fields occupy 48 bytes, which keeps the two",
        " * uint64_t fields 8-byte aligned.  sizeof(HqsbTilingData) must equal",
        " * HQSB_TILING_STRUCT_SIZE. */",
        "typedef struct HqsbTilingData {",
    ]
    for c_name, c_type, _code, _attr in TILING_FIELDS:
        lines.append(f"    {c_type} {c_name};")
    lines += [
        "} HqsbTilingData;",
        "",
        "/* TilingKey values (hqsb.ascend.tiling).  An unknown key must fail the",
        " * kernel launch; it must never fall through to a default path. */",
    ]
    for key in sorted(TILING_KEYS):
        lines.append(
            f"#define HQSB_TILING_KEY_{TILING_KEYS[key].upper()} {key}u"
        )
    lines += [
        "",
        "#endif /* HQSB_ASCEND_TILING_DATA_H */",
        "",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class TilingData:
    """The serialized host→device tiling record.

    ``rows``/``hidden`` describe the logical tensor; ``tile_elems``/``loop_count``/
    ``tail_elems`` describe the per-core loop; ``base_rows``/``extra_rows``
    describe the row partition across cores (details README §6.3)::

        base_rows  = rows // block_dim
        extra_rows = rows %  block_dim
        rows(core_i)   = base_rows + (1 if i < extra_rows else 0)
        start_row(i)   = i * base_rows + min(i, extra_rows)
    """

    schema_version: int = TILING_SCHEMA_VERSION
    tiling_key: int = TILING_KEY_SINGLE_TILE_ALIGNED
    dtype_tag: int = 0
    block_dim: int = 1
    rows: int = 1
    hidden: int = 0
    tile_elems: int = 0
    loop_count: int = 0
    tail_elems: int = 0
    buffer_count: int = 1
    base_rows: int = 0
    extra_rows: int = 0
    total_elements: int = 0
    workspace_bytes: int = 0
    reserved_0: int = 0
    reserved_1: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != TILING_SCHEMA_VERSION:
            raise ConfigError(
                f"unsupported tiling schema version {self.schema_version}; this build speaks "
                f"version {TILING_SCHEMA_VERSION} and refuses to reinterpret another layout",
                details={"schema_version": self.schema_version},
            )
        if self.tiling_key not in TILING_KEYS:
            raise ConfigError(
                f"unknown tiling key {self.tiling_key}; allowed {sorted(TILING_KEYS)} — an "
                "unknown key must fail rather than fall through to a default path",
                details={"tiling_key": self.tiling_key},
            )
        if self.block_dim < 1:
            raise ConfigError(
                f"block_dim must be >= 1, got {self.block_dim}; zero cores would make every "
                "per-core length a division by zero",
                details={"block_dim": self.block_dim},
            )
        if self.buffer_count not in (1, 2):
            raise ConfigError(
                f"buffer_count must be 1 (single) or 2 (double), got {self.buffer_count}",
                details={"buffer_count": self.buffer_count},
            )
        for name in (
            "rows",
            "hidden",
            "tile_elems",
            "loop_count",
            "tail_elems",
            "base_rows",
            "extra_rows",
            "total_elements",
            "workspace_bytes",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ConfigError(f"{name} must be non-negative, got {value}", details={"field": name})
        if self.extra_rows >= self.block_dim and self.rows:
            raise ConfigError(
                f"extra_rows={self.extra_rows} must be < block_dim={self.block_dim} (it is a remainder)",
                details={"extra_rows": self.extra_rows, "block_dim": self.block_dim},
            )
        if self.tail_elems > self.tile_elems:
            raise ConfigError(
                f"tail_elems={self.tail_elems} exceeds tile_elems={self.tile_elems}",
                details={"tail_elems": self.tail_elems, "tile_elems": self.tile_elems},
            )

    # ── serialization ───────────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        values = tuple(getattr(self, attr) for _c, _t, _code, attr in TILING_FIELDS)
        return struct.pack(TILING_STRUCT_FORMAT, *values)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "TilingData":
        if len(payload) != TILING_STRUCT_SIZE:
            raise ConfigError(
                f"tiling payload is {len(payload)} bytes but the ABI is {TILING_STRUCT_SIZE}; "
                "the host and device disagree about the struct",
                details={"got": len(payload), "expected": TILING_STRUCT_SIZE},
            )
        values = struct.unpack(TILING_STRUCT_FORMAT, payload)
        return cls(**{attr: value for (_c, _t, _code, attr), value in zip(TILING_FIELDS, values)})

    @property
    def sha256(self) -> str:
        return sha256_hex(self.to_bytes().hex())

    def as_dict(self) -> Dict[str, Any]:
        payload = {attr: getattr(self, attr) for _c, _t, _code, attr in TILING_FIELDS}
        payload["tiling_key_name"] = TILING_KEYS[self.tiling_key]
        payload["struct_size"] = TILING_STRUCT_SIZE
        payload["sha256"] = self.sha256
        return payload

    # ── derived geometry ────────────────────────────────────────────────────

    def rows_for_core(self, core_index: int) -> int:
        if not 0 <= core_index < self.block_dim:
            raise UsageError(
                f"core index {core_index} outside [0, {self.block_dim})", details={"core": core_index}
            )
        return self.base_rows + (1 if core_index < self.extra_rows else 0)

    def start_row(self, core_index: int) -> int:
        if not 0 <= core_index < self.block_dim:
            raise UsageError(f"core index {core_index} outside [0, {self.block_dim})")
        return core_index * self.base_rows + min(core_index, self.extra_rows)

    def elements_for_core(self, core_index: int) -> int:
        return checked_mul(self.rows_for_core(core_index), self.hidden, what="elements_for_core")

    def core_histogram(self) -> Dict[str, Any]:
        """Per-core work histogram — the E09-04 §3.4 load-balance evidence."""
        counts = [self.rows_for_core(index) for index in range(self.block_dim)]
        active = [count for count in counts if count > 0]
        return {
            "block_dim": self.block_dim,
            "rows_per_core": counts,
            "active_cores": len(active),
            "idle_cores": self.block_dim - len(active),
            "max_rows": max(counts) if counts else 0,
            "min_rows": min(counts) if counts else 0,
            "imbalance_ratio": (
                round(max(counts) / min(active), 6) if active and min(active) else float("inf")
            ),
        }

    def row_coverage_audit(self) -> Dict[str, Any]:
        """Prove every row is owned by exactly one core (E09-02 step 12/15)."""
        seen: List[int] = []
        for core in range(self.block_dim):
            start = self.start_row(core)
            seen.extend(range(start, start + self.rows_for_core(core)))
        duplicates = len(seen) - len(set(seen))
        return {
            "rows_covered": len(set(seen)),
            "rows_expected": self.rows,
            "duplicates": duplicates,
            "gaps": sorted(set(range(self.rows)) - set(seen)),
            "ok": duplicates == 0 and set(seen) == set(range(self.rows)),
        }


# ── UB budget ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UbBudget:
    """Usable on-chip budget, with the reservation method recorded.

    Details README §6.3 forbids handing the nominal capacity to user tensors:
    framework, queue, alignment and implementation reservations come off first,
    and *how* they were derived must travel with the number.
    """

    nominal_bytes: int
    framework_reserved_bytes: int = 0
    queue_reserved_bytes: int = 0
    alignment_reserved_bytes: int = 0
    implementation_reserved_bytes: int = 0
    method: str = ""

    def __post_init__(self) -> None:
        if self.nominal_bytes <= 0:
            raise ConfigError(
                "UB nominal capacity must come from a device probe, not a guessed constant",
                details={"nominal_bytes": self.nominal_bytes},
            )
        if not self.method:
            raise ConfigError(
                "a UB budget without a recorded method is not auditable (details README §6.3)",
                details={"field": "method"},
            )
        for name in (
            "framework_reserved_bytes",
            "queue_reserved_bytes",
            "alignment_reserved_bytes",
            "implementation_reserved_bytes",
        ):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be non-negative", details={"field": name})

    @property
    def reserved_bytes(self) -> int:
        return (
            self.framework_reserved_bytes
            + self.queue_reserved_bytes
            + self.alignment_reserved_bytes
            + self.implementation_reserved_bytes
        )

    @property
    def usable_bytes(self) -> int:
        usable = self.nominal_bytes - self.reserved_bytes
        if usable <= 0:
            raise ConfigError(
                f"reservations ({self.reserved_bytes}) exceed the nominal UB capacity "
                f"({self.nominal_bytes}); no tile can fit",
                details={"reserved": self.reserved_bytes, "nominal": self.nominal_bytes},
            )
        return usable

    def as_dict(self) -> Dict[str, Any]:
        return {
            "nominal_bytes": self.nominal_bytes,
            "framework_reserved_bytes": self.framework_reserved_bytes,
            "queue_reserved_bytes": self.queue_reserved_bytes,
            "alignment_reserved_bytes": self.alignment_reserved_bytes,
            "implementation_reserved_bytes": self.implementation_reserved_bytes,
            "reserved_bytes": self.reserved_bytes,
            "usable_bytes": self.usable_bytes if self.reserved_bytes < self.nominal_bytes else 0,
            "method": self.method,
        }


@dataclass(frozen=True)
class UbEstimate:
    """Static UB requirement of one candidate (E09-04 §3.2)."""

    x_bytes: int
    gamma_bytes: int
    y_bytes: int
    square_or_cast_temp_bytes: int
    reduce_temp_bytes: int
    scalar_intermediate_bytes: int
    alignment_and_queue_overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.x_bytes
            + self.gamma_bytes
            + self.y_bytes
            + self.square_or_cast_temp_bytes
            + self.reduce_temp_bytes
            + self.scalar_intermediate_bytes
            + self.alignment_and_queue_overhead_bytes
        )

    def fits(self, budget: UbBudget) -> bool:
        return self.total_bytes <= budget.usable_bytes

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "x_bytes": self.x_bytes,
            "gamma_bytes": self.gamma_bytes,
            "y_bytes": self.y_bytes,
            "square_or_cast_temp_bytes": self.square_or_cast_temp_bytes,
            "reduce_temp_bytes": self.reduce_temp_bytes,
            "scalar_intermediate_bytes": self.scalar_intermediate_bytes,
            "alignment_and_queue_overhead_bytes": self.alignment_and_queue_overhead_bytes,
            "total_bytes": self.total_bytes,
        }
        return payload


def estimate_rmsnorm_ub(
    *,
    tile_elems: int,
    dtype: str,
    accum_dtype: str = "fp32",
    buffer_count: int = 1,
    keep_gamma_on_chip: bool = True,
    scalar_bytes: int = 64,
    alignment_bytes: int = 512,
    queue_overhead_bytes: int = 0,
) -> UbEstimate:
    """The RMSNorm budget of E09-04 §3.2, term by term.

    ``buffer_count=2`` doubles every streamed tensor: that is precisely why
    double buffering can *lose* (smaller tile, more loops) and why the estimate
    is computed before any measurement.
    """
    item = dtype_bytes(dtype)
    accum = dtype_bytes(accum_dtype)
    x_bytes = checked_mul(checked_mul(tile_elems, item, what="x_tile"), buffer_count, what="x_buffers")
    y_bytes = checked_mul(checked_mul(tile_elems, item, what="y_tile"), buffer_count, what="y_buffers")
    gamma_bytes = (
        checked_mul(checked_mul(tile_elems, item, what="gamma_tile"), buffer_count, what="gamma_buffers")
        if keep_gamma_on_chip
        else 0
    )
    square_bytes = checked_mul(tile_elems, accum, what="square_temp")
    reduce_bytes = checked_mul(max(1, tile_elems // 8), accum, what="reduce_temp")
    return UbEstimate(
        x_bytes=x_bytes,
        gamma_bytes=gamma_bytes,
        y_bytes=y_bytes,
        square_or_cast_temp_bytes=square_bytes,
        reduce_temp_bytes=reduce_bytes,
        scalar_intermediate_bytes=scalar_bytes,
        alignment_and_queue_overhead_bytes=alignment_bytes + queue_overhead_bytes,
    )


def estimate_elementwise_ub(
    *, tile_elems: int, dtype: str, input_count: int, buffer_count: int = 1, alignment_bytes: int = 512
) -> UbEstimate:
    """Add-style budget: ``input_count`` streamed inputs plus one output."""
    item = dtype_bytes(dtype)
    per_buffer = checked_mul(tile_elems, item, what="elementwise_tile")
    return UbEstimate(
        x_bytes=checked_mul(per_buffer * input_count, buffer_count, what="elementwise_inputs"),
        gamma_bytes=0,
        y_bytes=checked_mul(per_buffer, buffer_count, what="elementwise_output"),
        square_or_cast_temp_bytes=0,
        reduce_temp_bytes=0,
        scalar_intermediate_bytes=64,
        alignment_and_queue_overhead_bytes=alignment_bytes,
    )


# ── host tiling ──────────────────────────────────────────────────────────────

#: Rejection reason codes.  Stable strings so a report can group refusals.
REJECT_NEGATIVE_DIM = "negative_or_zero_dimension"
REJECT_DTYPE = "unsupported_dtype"
REJECT_LAYOUT = "unsupported_layout"
REJECT_BLOCK_DIM = "block_dim_out_of_range"
REJECT_UB = "ub_budget_exceeded"
REJECT_WORKSPACE = "workspace_limit_exceeded"
REJECT_OVERFLOW = "checked_arithmetic_overflow"
REJECT_INDEX_WIDTH = "index_width_exceeded"
REJECT_ALIGNMENT = "tile_alignment_violation"
REJECT_SERIAL_SIZE = "serialized_tiling_too_large"


@dataclass(frozen=True)
class TilingLimits:
    """Device-verified limits.  Every value must come from a probe or a spec."""

    max_block_dim: int
    max_workspace_bytes: int
    alignment_elems: int
    ub_budget: UbBudget
    source: str = ""

    def __post_init__(self) -> None:
        if self.max_block_dim < 1:
            raise ConfigError(
                "max_block_dim must come from a verified capability probe, not a guess",
                details={"max_block_dim": self.max_block_dim},
            )
        if self.alignment_elems < 1:
            raise ConfigError("alignment_elems must be >= 1", details={"alignment_elems": self.alignment_elems})
        if not self.source:
            raise ConfigError(
                "tiling limits without a recorded source cannot be audited against the device",
                details={"field": "source"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_block_dim": self.max_block_dim,
            "max_workspace_bytes": self.max_workspace_bytes,
            "alignment_elems": self.alignment_elems,
            "ub_budget": self.ub_budget.as_dict(),
            "source": self.source,
        }


@dataclass(frozen=True)
class TilingRejection:
    """A refusal with its reason — never a silent clamp."""

    reason_code: str
    detail: str
    requested: Mapping[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": False,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "requested": dict(self.requested),
        }


@dataclass(frozen=True)
class TilingDecision:
    """Host tiling output: the ABI record plus everything needed to audit it."""

    tiling: TilingData
    ub_estimate: UbEstimate
    limits: TilingLimits
    dtype: str
    accum_dtype: str
    layout: str
    rationale: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        payload = self.tiling.as_dict()
        payload.update(
            {
                "ok": True,
                "dtype": self.dtype,
                "accum_dtype": self.accum_dtype,
                "layout": self.layout,
                "ub_estimate": self.ub_estimate.as_dict(),
                "ub_usable_bytes": self.limits.ub_budget.usable_bytes,
                "ub_fits": self.ub_estimate.fits(self.limits.ub_budget),
                "limits": self.limits.as_dict(),
                "rationale": list(self.rationale),
                "core_histogram": self.tiling.core_histogram(),
                "row_coverage": self.tiling.row_coverage_audit(),
            }
        )
        return payload


#: Layouts the v0 tiling path accepts.  Anything else must be converted (and the
#: conversion billed) or refused — never read as if it were contiguous.
SUPPORTED_LAYOUTS: Tuple[str, ...] = ("contiguous_nd",)

#: UB cost models the host tiling path knows how to charge.
UB_MODELS: Tuple[str, ...] = ("rmsnorm", "reduce", "elementwise")


def estimate_ub(
    model: str,
    *,
    tile_elems: int,
    dtype: str,
    accum_dtype: str = "fp32",
    buffer_count: int = 1,
    keep_gamma_on_chip: bool = True,
    input_count: int = 2,
) -> UbEstimate:
    """Charge the declared UB model (see :data:`UB_MODELS`)."""
    if model == "rmsnorm":
        return estimate_rmsnorm_ub(
            tile_elems=tile_elems,
            dtype=dtype,
            accum_dtype=accum_dtype,
            buffer_count=buffer_count,
            keep_gamma_on_chip=keep_gamma_on_chip,
        )
    if model == "reduce":
        item = dtype_bytes(dtype)
        accum = dtype_bytes(accum_dtype)
        streamed = checked_mul(
            checked_mul(tile_elems, item, what="reduce_tile"), buffer_count, what="reduce_buffers"
        )
        return UbEstimate(
            x_bytes=streamed,
            gamma_bytes=0,
            y_bytes=0,
            square_or_cast_temp_bytes=0,
            reduce_temp_bytes=accum,
            scalar_intermediate_bytes=64,
            alignment_and_queue_overhead_bytes=512,
        )
    if model == "elementwise":
        return estimate_elementwise_ub(
            tile_elems=tile_elems, dtype=dtype, input_count=input_count, buffer_count=buffer_count
        )
    raise ConfigError(f"unknown ub_model {model!r}; allowed {list(UB_MODELS)}", details={"ub_model": model})


def partition_rows(rows: int, block_dim: int) -> Tuple[int, int]:
    """``(base_rows, extra_rows)`` of details README §6.3."""
    if rows < 0:
        raise UsageError(f"rows must be non-negative, got {rows}", details={"rows": rows})
    if block_dim < 1:
        raise UsageError(f"block_dim must be >= 1, got {block_dim}", details={"block_dim": block_dim})
    return rows // block_dim, rows % block_dim


def partition_elements(total: int, block_dim: int) -> List[Tuple[int, int]]:
    """``[(start, length), ...]`` per core for a flat element-wise operator."""
    if total < 0:
        raise UsageError(f"total must be non-negative, got {total}")
    if block_dim < 1:
        raise UsageError(f"block_dim must be >= 1, got {block_dim}")
    base, extra = total // block_dim, total % block_dim
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for index in range(block_dim):
        length = base + (1 if index < extra else 0)
        spans.append((cursor, length))
        cursor += length
    return spans


def tile_loop(total_elems: int, tile_elems: int) -> Tuple[int, int]:
    """``(loop_count, tail_elems)`` for one core's tile loop."""
    if tile_elems < 1:
        raise UsageError(f"tile_elems must be >= 1, got {tile_elems}", details={"tile_elems": tile_elems})
    if total_elems < 0:
        raise UsageError(f"total_elems must be non-negative, got {total_elems}")
    loop_count = total_elems // tile_elems
    tail = total_elems % tile_elems
    if tail:
        loop_count += 1
    return loop_count, tail


def select_tiling_key(*, rows: int, hidden: int, block_dim: int, alignment_elems: int, rowwise: bool) -> int:
    """Pick the device code path.  Aligned and tail are separate keys (E09-04 step 13)."""
    aligned = hidden % alignment_elems == 0
    if rowwise:
        return TILING_KEY_ROW_ALIGNED if aligned else TILING_KEY_ROW_TAIL
    if hidden <= alignment_elems and aligned:
        return TILING_KEY_SINGLE_TILE_ALIGNED
    return TILING_KEY_MULTI_TILE_ALIGNED if aligned else TILING_KEY_TAIL_MASKED


@dataclass(frozen=True)
class TilingRequest:
    """Everything the host tiling function is allowed to look at."""

    rows: int
    hidden: int
    dtype: str
    block_dim: int
    tile_elems: int
    layout: str = "contiguous_nd"
    accum_dtype: str = "fp32"
    buffer_count: int = 1
    rowwise: bool = True
    keep_gamma_on_chip: bool = True
    workspace_bytes: int = 0
    #: Which UB model to charge: ``rmsnorm`` (x + gamma + y + square/reduce temps),
    #: ``reduce`` (x + accumulator) or ``elementwise`` (inputs + y).  Declaring it
    #: keeps a reduction candidate from being budgeted as if it streamed gamma.
    ub_model: str = "rmsnorm"

    def __post_init__(self) -> None:
        if self.ub_model not in UB_MODELS:
            raise ConfigError(
                f"unknown ub_model {self.ub_model!r}; allowed {sorted(UB_MODELS)}",
                details={"ub_model": self.ub_model},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "hidden": self.hidden,
            "dtype": self.dtype,
            "block_dim": self.block_dim,
            "tile_elems": self.tile_elems,
            "layout": self.layout,
            "accum_dtype": self.accum_dtype,
            "buffer_count": self.buffer_count,
            "rowwise": self.rowwise,
            "keep_gamma_on_chip": self.keep_gamma_on_chip,
            "workspace_bytes": self.workspace_bytes,
            "ub_model": self.ub_model,
        }


def compute_tiling(request: TilingRequest, limits: TilingLimits) -> Tuple[Optional[TilingDecision], Optional[TilingRejection]]:
    """Validate and compute one tiling, or return a structured refusal.

    Returns a ``(decision, None)`` / ``(None, rejection)`` pair rather than
    raising, because a refusal is a *result* the experiment records (E09-02
    step 7: "each case saves its tiling input, output and decision reason").
    """
    asked = request.as_dict()

    if request.layout not in SUPPORTED_LAYOUTS:
        return None, TilingRejection(
            REJECT_LAYOUT,
            f"layout {request.layout!r} is not supported by the v0 tiling path; convert explicitly "
            "(and bill the conversion) or refuse — do not read it as contiguous",
            asked,
        )
    if request.rows < 0 or request.hidden < 0:
        return None, TilingRejection(REJECT_NEGATIVE_DIM, f"negative dimension in {asked}", asked)
    if request.rows == 0 or request.hidden == 0:
        # An empty tensor is a *policy* decision (E09-02 §4.2: "choose the
        # mathematical zero or refuse; it must not be undefined"), so it is
        # refused here and the refusal is the recorded answer.
        return None, TilingRejection(
            REJECT_NEGATIVE_DIM,
            "an empty dimension is refused by this tiling path; the OperatorSpec must state "
            "whether the mathematical zero is returned instead",
            asked,
        )
    try:
        item_bytes = dtype_bytes(request.dtype)
        dtype_bytes(request.accum_dtype)
    except UsageError as exc:
        return None, TilingRejection(REJECT_DTYPE, str(exc), asked)
    if request.block_dim < 1 or request.block_dim > limits.max_block_dim:
        return None, TilingRejection(
            REJECT_BLOCK_DIM,
            f"block_dim={request.block_dim} outside [1, {limits.max_block_dim}]; the bound comes "
            f"from {limits.source}",
            asked,
        )
    try:
        total = checked_mul(request.rows, request.hidden, what="total_elements")
    except ArithmeticOverflow as exc:
        return None, TilingRejection(REJECT_OVERFLOW, str(exc), asked)
    if total > MAX_INDEX32:
        return None, TilingRejection(
            REJECT_INDEX_WIDTH,
            f"total_elements={total} exceeds the {INDEX_WIDTH_BITS}-bit index width the kernel uses",
            asked,
        )
    if request.tile_elems < 1:
        return None, TilingRejection(REJECT_ALIGNMENT, f"tile_elems must be >= 1, got {request.tile_elems}", asked)
    if request.tile_elems % limits.alignment_elems:
        return None, TilingRejection(
            REJECT_ALIGNMENT,
            f"tile_elems={request.tile_elems} is not a multiple of the verified alignment "
            f"{limits.alignment_elems}",
            asked,
        )
    if request.workspace_bytes > limits.max_workspace_bytes:
        return None, TilingRejection(
            REJECT_WORKSPACE,
            f"workspace_bytes={request.workspace_bytes} exceeds the verified limit "
            f"{limits.max_workspace_bytes}",
            asked,
        )

    base_rows, extra_rows = partition_rows(request.rows, request.block_dim)
    per_core_elems = checked_mul(base_rows + (1 if extra_rows else 0), request.hidden, what="per_core_elems")
    # A row-wise kernel tiles *within one row* (every core runs the same number of
    # tile iterations); a flat element-wise kernel tiles over its whole per-core
    # range, so ``loop_count`` is the worst-case core's iteration count.
    loop_units = request.hidden if request.rowwise else per_core_elems
    loop_count, tail = tile_loop(loop_units, request.tile_elems)

    ub = estimate_ub(
        request.ub_model,
        tile_elems=request.tile_elems,
        dtype=request.dtype,
        accum_dtype=request.accum_dtype,
        buffer_count=request.buffer_count,
        keep_gamma_on_chip=request.keep_gamma_on_chip,
        input_count=2 if request.ub_model == "elementwise" else 1,
    )
    if not ub.fits(limits.ub_budget):
        return None, TilingRejection(
            REJECT_UB,
            f"UB requirement {ub.total_bytes} B exceeds the usable budget "
            f"{limits.ub_budget.usable_bytes} B (nominal {limits.ub_budget.nominal_bytes} B, "
            f"reserved {limits.ub_budget.reserved_bytes} B, method: {limits.ub_budget.method})",
            asked,
        )

    tiling_key = select_tiling_key(
        rows=request.rows,
        hidden=request.hidden,
        block_dim=request.block_dim,
        alignment_elems=limits.alignment_elems,
        rowwise=request.rowwise,
    )
    try:
        tiling = TilingData(
            schema_version=TILING_SCHEMA_VERSION,
            tiling_key=tiling_key,
            dtype_tag=DTYPE_TAGS[request.dtype],
            block_dim=request.block_dim,
            rows=request.rows,
            hidden=request.hidden,
            tile_elems=request.tile_elems,
            loop_count=loop_count,
            tail_elems=tail,
            buffer_count=request.buffer_count,
            base_rows=base_rows,
            extra_rows=extra_rows,
            total_elements=total,
            workspace_bytes=request.workspace_bytes,
        )
    except ConfigError as exc:
        return None, TilingRejection(REJECT_OVERFLOW, str(exc), asked)

    payload = tiling.to_bytes()
    if len(payload) > MAX_TILING_SERIALIZED_BYTES:
        return None, TilingRejection(
            REJECT_SERIAL_SIZE, f"serialized tiling is {len(payload)} B", asked
        )

    rationale = [
        f"rows partitioned as base_rows={base_rows} + extra_rows={extra_rows} over block_dim={request.block_dim}",
        f"loop unit={'hidden' if request.rowwise else 'per_core_elements'}="
        f"{loop_units} → loop_count={loop_count}, tail_elems={tail} (per-core elements={per_core_elems})",
        f"tiling_key={TILING_KEYS[tiling_key]} (hidden={request.hidden}, alignment={limits.alignment_elems})",
        f"UB model={request.ub_model}: {ub.total_bytes} B ≤ usable {limits.ub_budget.usable_bytes} B ({limits.ub_budget.method})",
        f"buffer_count={request.buffer_count}: double buffering trades UB for overlap and is not assumed to win",
    ]
    return (
        TilingDecision(
            tiling=tiling,
            ub_estimate=ub,
            limits=limits,
            dtype=request.dtype,
            accum_dtype=request.accum_dtype,
            layout=request.layout,
            rationale=tuple(rationale),
        ),
        None,
    )


# ── candidate search space (E09-04 steps 6/7) ────────────────────────────────


@dataclass(frozen=True)
class TilingCandidate:
    """One point of the search space, legal or not."""

    block_dim: int
    tile_elems: int
    buffer_count: int
    tiling_key_hint: str = ""
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "block_dim": self.block_dim,
            "tile_elems": self.tile_elems,
            "buffer_count": self.buffer_count,
            "tiling_key_hint": self.tiling_key_hint,
            "note": self.note,
        }


def candidate_space(
    *,
    block_dims: Sequence[int],
    tile_elems: Sequence[int],
    buffer_counts: Sequence[int] = (1, 2),
) -> List[TilingCandidate]:
    return [
        TilingCandidate(block_dim=block, tile_elems=tile, buffer_count=buffers)
        for block in block_dims
        for tile in tile_elems
        for buffers in buffer_counts
    ]


def filter_candidates(
    candidates: Iterable[TilingCandidate], request_template: TilingRequest, limits: TilingLimits
) -> Dict[str, Any]:
    """Legality filter of E09-04 §5 — and it keeps the rejects.

    "Only save the winner" is an explicit prohibition (E09-04 §10), so every
    filtered-out candidate is returned with its reason code.
    """
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for candidate in candidates:
        request = TilingRequest(
            rows=request_template.rows,
            hidden=request_template.hidden,
            dtype=request_template.dtype,
            block_dim=candidate.block_dim,
            tile_elems=candidate.tile_elems,
            layout=request_template.layout,
            accum_dtype=request_template.accum_dtype,
            buffer_count=candidate.buffer_count,
            rowwise=request_template.rowwise,
            keep_gamma_on_chip=request_template.keep_gamma_on_chip,
            workspace_bytes=request_template.workspace_bytes,
        )
        decision, refusal = compute_tiling(request, limits)
        row = candidate.as_dict()
        if decision is not None:
            row["tiling_key"] = decision.tiling.tiling_key
            row["tiling_key_name"] = TILING_KEYS[decision.tiling.tiling_key]
            row["loop_count"] = decision.tiling.loop_count
            row["tail_elems"] = decision.tiling.tail_elems
            row["ub_bytes"] = decision.ub_estimate.total_bytes
            row["active_cores"] = decision.tiling.core_histogram()["active_cores"]
            row["imbalance_ratio"] = decision.tiling.core_histogram()["imbalance_ratio"]
            row["tiling_sha256"] = decision.tiling.sha256
            accepted.append(row)
        else:
            assert refusal is not None
            row.update(refusal.as_dict())
            rejected.append(row)
    # E09-04 §5 also requires "expected work per active core > 0".
    zero_work = [row for row in accepted if row["active_cores"] == 0]
    for row in zero_work:
        accepted.remove(row)
        row["reason_code"] = "no_active_core_work"
        row["detail"] = "every core would receive zero rows; launching it wastes task resources"
        rejected.append(row)
    return {
        "accepted": accepted,
        "rejected": rejected,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "reject_reasons": sorted({row["reason_code"] for row in rejected}),
    }


def tiling_plan_hash(request: TilingRequest, limits: TilingLimits) -> str:
    """Stable hash of (request, limits) — the autotune cache key (control plane §30)."""
    return sha256_hex(canonical_json({"request": request.as_dict(), "limits": limits.as_dict()}))


__all__ = [
    "ArithmeticOverflow",
    "DTYPE_BYTES",
    "DTYPE_TAGS",
    "INDEX_WIDTH_BITS",
    "MAX_INDEX32",
    "MAX_TILING_SERIALIZED_BYTES",
    "REJECT_ALIGNMENT",
    "REJECT_BLOCK_DIM",
    "REJECT_DTYPE",
    "REJECT_INDEX_WIDTH",
    "REJECT_LAYOUT",
    "REJECT_NEGATIVE_DIM",
    "REJECT_OVERFLOW",
    "REJECT_SERIAL_SIZE",
    "REJECT_UB",
    "REJECT_WORKSPACE",
    "SUB_BYTE_DTYPES",
    "SUPPORTED_LAYOUTS",
    "TILING_FIELDS",
    "TILING_HEADER_PATH",
    "TILING_KEYS",
    "TILING_KEY_MULTI_TILE_ALIGNED",
    "TILING_KEY_NAMES",
    "TILING_KEY_ROW_ALIGNED",
    "TILING_KEY_ROW_TAIL",
    "TILING_KEY_SINGLE_TILE_ALIGNED",
    "TILING_KEY_TAIL_MASKED",
    "TILING_SCHEMA_VERSION",
    "TILING_STRUCT_FORMAT",
    "TILING_STRUCT_SIZE",
    "TilingCandidate",
    "TilingData",
    "TilingDecision",
    "TilingLimits",
    "TilingRejection",
    "TilingRequest",
    "UbBudget",
    "UbEstimate",
    "candidate_space",
    "checked_add",
    "checked_mul",
    "compute_tiling",
    "dtype_bytes",
    "estimate_elementwise_ub",
    "estimate_rmsnorm_ub",
    "filter_candidates",
    "partition_elements",
    "partition_rows",
    "select_tiling_key",
    "tile_loop",
    "tiling_header_source",
    "tiling_plan_hash",
]
