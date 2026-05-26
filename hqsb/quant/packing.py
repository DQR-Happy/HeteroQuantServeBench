"""Packing layouts: canonical byte form and kernel-specific variants.

Three layers are kept strictly separate (E05-01 §7, details README §5):

``logical``
    ``q`` values in ``[qmin, qmax]``, one conceptual value per element. This
    is what :mod:`hqsb.quant.rtn` produces and what all quality comparisons
    use.

``canonical``
    A minimal, fully-specified byte serialization of the logical values
    (``hqsb.nibble.v1``: two signed 4-bit codes per byte, *low* nibble first;
    ``hqsb.byte.v1``: one signed 8-bit code per byte). It never carries kernel
    tiling, swizzle or alignment; an algorithm can therefore be compared
    across backends without a layout in the way.

``kernel variant``
    A derived byte layout for one concrete kernel
    (``hqsb.w4a16.rowmajor.nk.v1``), with its own ``layout_id``, version,
    alignment, strides, scale/zero ordering and ``parent_canonical_hash``.
    A kernel variant never redefines the logical values; it only moves them.

Frozen canonical encoding:

* 4-bit: ``byte = (code_even & 0xF) | ((code_odd & 0xF) << 4)`` with
  ``code = q & 0xF`` (two's complement); an odd element count leaves the high
  nibble of the last byte at zero and the element count lives in the
  manifest — a decoder *must* be told the count, so "0 as a valid code" can
  never be confused with padding;
* 8-bit: ``byte = q & 0xFF`` (signed two's complement, row-major);
* byte order inside the payload: row-major, then the quantized axis, i.e. the
  same order as ``q``.

Invalid padding, out-of-range codes, truncated or over-long payloads and
unknown layout ids are refused with stable reason codes
(:class:`~hqsb.core.errors.ArtifactError`), never silently reinterpreted.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from hqsb.core.errors import ArtifactError
from hqsb.quant.spec import QuantScheme

#: Canonical packing family ids (recorded in the artifact).
PACK_NIBBLE_V1 = "hqsb.nibble.v1"
PACK_BYTE_V1 = "hqsb.byte.v1"

#: Kernel layout ids.
LAYOUT_W4A16_ROWMAJOR_NK_V1 = "hqsb.w4a16.rowmajor.nk.v1"
LAYOUT_W4A16_HIFIRST_NK_V1 = "hqsb.w4a16.hifirst.nk.v1"
LAYOUT_W8A16_ROWMAJOR_NK_V1 = "hqsb.w8a16.rowmajor.nk.v1"

#: Nibble order (E05-01 §7 requires it to be frozen per layout).
NIBBLE_LOW_FIRST = "low_first"
NIBBLE_HIGH_FIRST = "high_first"

#: layout id -> (bits, nibble order). Every known layout is explicitly listed:
#: an unknown id is a refusal, never a guess.
KNOWN_LAYOUTS = {
    LAYOUT_W4A16_ROWMAJOR_NK_V1: (4, NIBBLE_LOW_FIRST),
    LAYOUT_W4A16_HIFIRST_NK_V1: (4, NIBBLE_HIGH_FIRST),
    LAYOUT_W8A16_ROWMAJOR_NK_V1: (8, NIBBLE_LOW_FIRST),
}

#: Reason codes (stable machine-readable strings).
REASON_UNKNOWN_LAYOUT = "unknown_layout_id"
REASON_UNKNOWN_PACK_VERSION = "unknown_pack_version"
REASON_BAD_LENGTH = "payload_length_mismatch"
REASON_BAD_PADDING = "nonzero_padding_nibble"
REASON_CODE_OUT_OF_RANGE = "code_out_of_range"
REASON_ODD_ROW = "row_width_not_multiple_of_2"
REASON_ALIGNMENT = "alignment_not_satisfied"
REASON_PARENT_HASH = "parent_canonical_hash_mismatch"


def canonical_pack_version(bits: int) -> str:
    """Canonical pack family for ``bits`` (4 -> nibble, 8 -> byte)."""
    if bits == 4:
        return PACK_NIBBLE_V1
    if bits == 8:
        return PACK_BYTE_V1
    raise ArtifactError(
        f"no canonical packing defined for {bits}-bit (supported: 4, 8)",
        details={"reason_code": REASON_UNKNOWN_PACK_VERSION, "bits": bits},
    )


def packed_row_bytes(bits: int, cols: int) -> int:
    """Bytes needed for ``cols`` logical values in the canonical packing."""
    if bits == 4:
        return (cols + 1) // 2
    if bits == 8:
        return cols
    raise ArtifactError(
        f"no canonical packing defined for {bits}-bit",
        details={"reason_code": REASON_UNKNOWN_PACK_VERSION, "bits": bits},
    )


def pack_canonical(
    q: Sequence[int], bits: int, *, nibble_order: str = NIBBLE_LOW_FIRST
) -> bytes:
    """Pack logical codes into the canonical byte form.

    ``nibble_order`` is part of the *layout identity*, not of the algorithm:
    the canonical (``hqsb.nibble.v1``) form is fixed to low-first, while a
    kernel variant may declare high-first (see :data:`KNOWN_LAYOUTS`).
    """
    codes: List[int] = []
    for code in q:
        as_int = int(code)
        if float(code) != float(as_int):
            raise ArtifactError(
                f"logical code {code!r} is not an integer; packing would "
                f"silently truncate it",
                details={"reason_code": REASON_CODE_OUT_OF_RANGE, "code": code},
            )
        codes.append(as_int)
    if nibble_order not in (NIBBLE_LOW_FIRST, NIBBLE_HIGH_FIRST):
        raise ArtifactError(
            f"unknown nibble order {nibble_order!r}",
            details={"reason_code": REASON_UNKNOWN_PACK_VERSION},
        )
    if bits == 4:
        out = bytearray((len(codes) + 1) // 2)
        for index, code in enumerate(codes):
            nibble = code & 0xF
            even_position = index % 2 == 0
            if nibble_order == NIBBLE_LOW_FIRST:
                if even_position:
                    out[index // 2] |= nibble
                else:
                    out[index // 2] |= nibble << 4
            else:  # high-first: first value of the pair goes to the high nibble
                if even_position:
                    out[index // 2] |= nibble << 4
                else:
                    out[index // 2] |= nibble
        return bytes(out)
    if bits == 8:
        return bytes((code & 0xFF) for code in codes)
    raise ArtifactError(
        f"no canonical packing defined for {bits}-bit",
        details={"reason_code": REASON_UNKNOWN_PACK_VERSION, "bits": bits},
    )


def unpack_canonical(
    payload: bytes,
    count: int,
    bits: int,
    *,
    scheme: Optional[QuantScheme] = None,
    nibble_order: str = NIBBLE_LOW_FIRST,
) -> List[int]:
    """Unpack the canonical byte form back into ``count`` logical codes.

    When ``count`` is odd for a 4-bit payload the high nibble of the last byte
    is required to be zero (:data:`REASON_BAD_PADDING`); the caller cannot
    infer the count from the payload, so an inconsistent ``count`` is a hard
    error rather than a silently wrong tensor.
    """
    if count < 0:
        raise ArtifactError(f"count must be non-negative, got {count}")
    expected = packed_row_bytes(bits, count)
    if len(payload) != expected:
        raise ArtifactError(
            f"canonical payload length {len(payload)} != expected {expected} "
            f"for {count} values at {bits} bit",
            details={
                "reason_code": REASON_BAD_LENGTH,
                "count": count,
                "bits": bits,
                "actual_bytes": len(payload),
                "expected_bytes": expected,
            },
        )
    if nibble_order not in (NIBBLE_LOW_FIRST, NIBBLE_HIGH_FIRST):
        raise ArtifactError(
            f"unknown nibble order {nibble_order!r}",
            details={"reason_code": REASON_UNKNOWN_PACK_VERSION},
        )
    if bits == 4:
        codes = []
        for index in range(count):
            byte = payload[index // 2]
            if nibble_order == NIBBLE_LOW_FIRST:
                nibble = (byte & 0xF) if index % 2 == 0 else ((byte >> 4) & 0xF)
            else:
                nibble = ((byte >> 4) & 0xF) if index % 2 == 0 else (byte & 0xF)
            codes.append(_signed_from_nibble(nibble))
        padding_nibble = (payload[-1] & 0xF) if nibble_order == NIBBLE_HIGH_FIRST else ((payload[-1] >> 4) & 0xF)
        if count % 2 == 1 and padding_nibble:
            raise ArtifactError(
                "trailing padding nibble must be zero",
                details={"reason_code": REASON_BAD_PADDING, "count": count},
            )
        if scheme is not None:
            validate_codes(codes, scheme)
        return codes
    if bits == 8:
        codes = [_signed_from_byte(byte) for byte in payload]
        if scheme is not None:
            validate_codes(codes, scheme)
        return codes
    raise ArtifactError(
        f"no canonical packing defined for {bits}-bit",
        details={"reason_code": REASON_UNKNOWN_PACK_VERSION, "bits": bits},
    )


def unpack_independent(
    payload: bytes,
    count: int,
    bits: int,
    *,
    nibble_order: str = NIBBLE_LOW_FIRST,
) -> List[int]:
    """A second, independent decoder (E05-01 §11 step 8 / E05-06 §9).

    Deliberately implemented with a different control flow than
    :func:`unpack_canonical` — it consumes the payload two values at a time
    from the front and reads the high nibble before the low one — so that a
    single indexing bug cannot validate itself. Kernel-side packers must be
    checked against this function, never only against their own inverse.
    """
    if count < 0:
        raise ArtifactError(
            f"count must be non-negative, got {count}",
            details={"reason_code": REASON_BAD_LENGTH},
        )
    expected = packed_row_bytes(bits, count)
    if len(payload) != expected:
        raise ArtifactError(
            f"payload length {len(payload)} != expected {expected}",
            details={
                "reason_code": REASON_BAD_LENGTH,
                "actual_bytes": len(payload),
                "expected_bytes": expected,
            },
        )
    codes: List[int] = []
    if bits == 4:
        for byte in payload:
            if len(codes) >= count:
                break
            low = _signed_from_nibble(byte & 0xF)
            high = _signed_from_nibble((byte >> 4) & 0xF)
            if nibble_order == NIBBLE_LOW_FIRST:
                codes.append(low)
                if len(codes) >= count:
                    break
                codes.append(high)
            else:
                codes.append(high)
                if len(codes) >= count:
                    break
                codes.append(low)
        if len(codes) != count:
            raise ArtifactError(
                "independent unpacker produced the wrong element count",
                details={"reason_code": REASON_BAD_LENGTH, "count": count},
            )
        return codes
    if bits == 8:
        return [int.from_bytes(bytes([byte]), "little", signed=True) for byte in payload]
    raise ArtifactError(
        f"no canonical packing defined for {bits}-bit",
        details={"reason_code": REASON_UNKNOWN_PACK_VERSION, "bits": bits},
    )


def _signed_from_nibble(nibble: int) -> int:
    """Sign-extend a 4-bit two's-complement code."""
    nibble &= 0xF
    return nibble - 16 if nibble >= 8 else nibble


def _signed_from_byte(byte: int) -> int:
    byte &= 0xFF
    return byte - 256 if byte >= 128 else byte


def validate_codes(codes: Sequence[int], scheme: QuantScheme) -> None:
    """Reject any logical code outside ``[scheme.qmin, scheme.qmax]``.

    Two's-complement bit patterns are all decodable, but a *symmetric S1*
    scheme has no code ``-2^(b-1)``: a payload containing it was produced by a
    different scheme (or corrupted) and must be refused instead of being
    interpreted as a legal value.
    """
    qmin, qmax = scheme.qmin, scheme.qmax
    for index, code in enumerate(codes):
        if not qmin <= code <= qmax:
            raise ArtifactError(
                f"code {code} at index {index} outside [{qmin}, {qmax}] for "
                f"scheme {scheme.label!r}",
                details={
                    "reason_code": REASON_CODE_OUT_OF_RANGE,
                    "index": index,
                    "code": code,
                    "qmin": qmin,
                    "qmax": qmax,
                },
            )


@dataclass(frozen=True)
class KernelLayout:
    """A concrete kernel byte layout (derived from canonical values).

    The layout describes *how* one weight matrix ``W[rows, cols]``
    (``rows`` = output channels, ``cols`` = K/input features) is stored for a
    kernel that reads packed weights directly:

    * nibbles/bytes along ``cols`` within one row are contiguous;
    * rows are padded to ``row_stride_bytes`` (``alignment``-byte aligned);
    * scales and zeros are stored as separate planes in ``[rows, groups]``
      order (row-major over output channels), which is what a per-output-row
      dequant broadcast needs;
    * trailing ``cols`` padding uses the zero code and is *never* consumed:
      the kernel masks it with the recorded ``valid_cols``.
    """

    layout_id: str
    layout_version: str
    bits: int
    rows: int
    cols: int
    padded_cols: int
    valid_cols: int
    row_stride_bytes: int
    alignment: int
    group_size: Optional[int]
    groups_per_row: int
    nibble_order: str
    scale_dtype: str
    zero_dtype: str
    scale_count: int
    zero_count: int
    #: Bytes of the packed payload (rows * row_stride_bytes).
    payload_bytes: int
    #: Bytes of the scale plane and zero plane in their declared dtypes.
    scale_bytes: int
    zero_bytes: int
    parent_canonical_hash: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "layout_id": self.layout_id,
            "layout_version": self.layout_version,
            "bits": self.bits,
            "rows": self.rows,
            "cols": self.cols,
            "padded_cols": self.padded_cols,
            "valid_cols": self.valid_cols,
            "row_stride_bytes": self.row_stride_bytes,
            "alignment": self.alignment,
            "group_size": self.group_size,
            "groups_per_row": self.groups_per_row,
            "nibble_order": self.nibble_order,
            "scale_dtype": self.scale_dtype,
            "zero_dtype": self.zero_dtype,
            "scale_count": self.scale_count,
            "zero_count": self.zero_count,
            "payload_bytes": self.payload_bytes,
            "scale_bytes": self.scale_bytes,
            "zero_bytes": self.zero_bytes,
            "parent_canonical_hash": self.parent_canonical_hash,
        }

    @property
    def total_bytes(self) -> int:
        return self.payload_bytes + self.scale_bytes + self.zero_bytes


def _dtype_bytes(dtype: str) -> int:
    return {
        "float16": 2,
        "float32": 4,
        "float64": 8,
        "int8": 1,
        "int16": 2,
        "int32": 4,
    }[dtype]


def plan_kernel_layout(
    scheme: QuantScheme,
    rows: int,
    cols: int,
    *,
    layout_id: Optional[str] = None,
    alignment: int = 16,
    parent_canonical_hash: Optional[str] = None,
) -> KernelLayout:
    """Plan the kernel layout for a ``[rows, cols]`` weight.

    Pads ``cols`` to the layout's granularity (``2`` values per byte for
    4-bit, ``0`` extra otherwise) and aligns each row to ``alignment`` bytes.
    Raises :class:`ArtifactError` for an unsupported layout id.
    """
    if rows < 0 or cols < 0:
        raise ArtifactError(
            f"rows/cols must be non-negative, got rows={rows} cols={cols}",
            details={"reason_code": REASON_BAD_LENGTH},
        )
    resolved_layout = layout_id or (
        LAYOUT_W4A16_ROWMAJOR_NK_V1 if scheme.bits == 4 else LAYOUT_W8A16_ROWMAJOR_NK_V1
    )
    if resolved_layout not in KNOWN_LAYOUTS:
        raise ArtifactError(
            f"unknown kernel layout id {resolved_layout!r}",
            details={"reason_code": REASON_UNKNOWN_LAYOUT, "layout_id": resolved_layout},
        )
    layout_bits, nibble_order = KNOWN_LAYOUTS[resolved_layout]
    if layout_bits != scheme.bits:
        raise ArtifactError(
            f"layout {resolved_layout!r} stores {layout_bits}-bit values but the "
            f"artifact is {scheme.bits}-bit",
            details={
                "reason_code": REASON_UNKNOWN_LAYOUT,
                "layout_id": resolved_layout,
                "layout_bits": layout_bits,
                "artifact_bits": scheme.bits,
            },
        )
    if alignment <= 0 or alignment % 2 != 0:
        raise ArtifactError(
            f"alignment must be a positive even number of bytes, got {alignment}",
            details={"reason_code": REASON_ALIGNMENT, "alignment": alignment},
        )

    padding = 1 if (scheme.bits == 4 and cols % 2) else 0
    padded_cols = cols + padding
    raw_row_bytes = packed_row_bytes(scheme.bits, padded_cols)
    row_stride = ((raw_row_bytes + alignment - 1) // alignment) * alignment
    groups = scheme.unit_count(cols)
    scale_bytes = _dtype_bytes(scheme.scale_dtype) * (rows * groups)
    zero_count = 0 if scheme.symmetric else rows * groups
    zero_bytes = 0 if scheme.symmetric else _dtype_bytes(scheme.zero_dtype) * zero_count
    return KernelLayout(
        layout_id=resolved_layout,
        layout_version="1",
        bits=scheme.bits,
        rows=rows,
        cols=cols,
        padded_cols=padded_cols,
        valid_cols=cols,
        row_stride_bytes=row_stride,
        alignment=alignment,
        group_size=scheme.group_size,
        groups_per_row=groups,
        nibble_order=nibble_order,
        scale_dtype=scheme.scale_dtype,
        zero_dtype=scheme.zero_dtype,
        scale_count=rows * groups,
        zero_count=zero_count,
        payload_bytes=rows * row_stride,
        scale_bytes=scale_bytes,
        zero_bytes=zero_bytes,
        parent_canonical_hash=parent_canonical_hash,
    )


@dataclass
class PackedTensor:
    """A kernel-layout payload plus its scale/zero planes and integrity hash."""

    layout: KernelLayout
    payload: bytes
    scales: List[float]
    zeros: List[int]
    layout_extra: Dict[str, object] = field(default_factory=dict)

    def payload_sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def variant_hash(self) -> str:
        """Identity of the packed variant: layout + payload + planes."""
        digest = hashlib.sha256()
        digest.update(
            __import__("json")
            .dumps(self.layout.as_dict(), sort_keys=True, separators=(",", ":"))
            .encode("utf-8")
        )
        digest.update(self.payload)
        digest.update(struct.pack(f"<{len(self.scales)}f", *self.scales))
        if self.zeros:
            digest.update(struct.pack(f"<{len(self.zeros)}i", *self.zeros))
        return digest.hexdigest()

    def as_dict(self) -> Dict[str, object]:
        return {
            "layout": self.layout.as_dict(),
            "payload_bytes": len(self.payload),
            "payload_sha256": self.payload_sha256(),
            "variant_hash": self.variant_hash(),
            "layout_extra": dict(self.layout_extra),
        }


def pack_kernel_variant(
    q: Sequence[int],
    scales: Sequence[float],
    zeros: Sequence[int],
    scheme: QuantScheme,
    rows: int,
    cols: int,
    *,
    layout_id: Optional[str] = None,
    alignment: int = 16,
    parent_canonical_hash: Optional[str] = None,
) -> PackedTensor:
    """Build a :class:`PackedTensor` in the requested kernel layout.

    ``q`` is row-major over ``[rows, cols]``. The layout padding is filled with
    the zero code; those values are recorded as padding in the layout
    descriptor and must be masked by the kernel (never multiplied into a
    result).
    """
    layout = plan_kernel_layout(
        scheme,
        rows,
        cols,
        layout_id=layout_id,
        alignment=alignment,
        parent_canonical_hash=parent_canonical_hash,
    )
    if len(q) != rows * cols:
        raise ArtifactError(
            f"q has {len(q)} values but rows*cols = {rows * cols}",
            details={
                "reason_code": REASON_BAD_LENGTH,
                "values": len(q),
                "rows": rows,
                "cols": cols,
            },
        )
    validate_codes(q, scheme)
    for index, scale in enumerate(scales):
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise ArtifactError(
                f"scale[{index}] = {scale!r} is not finite and positive; packing "
                f"a non-positive scale would make the variant undecodable",
                details={
                    "reason_code": REASON_BAD_LENGTH,
                    "index": index,
                    "scale": scale,
                },
            )
    if len(scales) != layout.scale_count:
        raise ArtifactError(
            f"expected {layout.scale_count} scales, got {len(scales)}",
            details={
                "reason_code": REASON_BAD_LENGTH,
                "expected": layout.scale_count,
                "actual": len(scales),
            },
        )
    if not scheme.symmetric and len(zeros) != layout.zero_count:
        raise ArtifactError(
            f"expected {layout.zero_count} zeros, got {len(zeros)}",
            details={
                "reason_code": REASON_BAD_LENGTH,
                "expected": layout.zero_count,
                "actual": len(zeros),
            },
        )

    out = bytearray(layout.payload_bytes)
    for row in range(rows):
        row_bytes = bytearray(layout.row_stride_bytes)
        row_values = [int(code) for code in q[row * cols : (row + 1) * cols]]
        # Padding values are the zero code (harmless only because the kernel
        # masks them; they are not part of the logical tensor).
        if layout.padded_cols > cols:
            row_values.append(0)
        packed = bytearray(
            pack_canonical(row_values, scheme.bits, nibble_order=layout.nibble_order)
        )
        row_bytes[: len(packed)] = packed
        start = row * layout.row_stride_bytes
        out[start : start + layout.row_stride_bytes] = row_bytes
    if len(out) != layout.payload_bytes:
        raise ArtifactError(
            "internal packing size mismatch",
            details={
                "reason_code": REASON_BAD_LENGTH,
                "actual": len(out),
                "expected": layout.payload_bytes,
            },
        )
    return PackedTensor(
        layout=layout,
        payload=bytes(out),
        scales=[float(scale) for scale in scales],
        zeros=[int(zero) for zero in zeros] if not scheme.symmetric else [],
    )


def unpack_kernel_variant(
    packed: PackedTensor,
    scheme: QuantScheme,
    *,
    independent: bool = False,
) -> List[int]:
    """Recover the logical codes from a kernel-layout payload.

    ``independent=True`` routes the nibble decode through
    :func:`unpack_independent`, which is how kernel-side packers are
    cross-validated (E05-06 §9 forbids pack and kernel sharing one index
    implementation).
    """
    layout = packed.layout
    if len(packed.payload) != layout.payload_bytes:
        raise ArtifactError(
            f"payload is {len(packed.payload)} bytes, layout declares "
            f"{layout.payload_bytes}",
            details={
                "reason_code": REASON_BAD_LENGTH,
                "actual": len(packed.payload),
                "expected": layout.payload_bytes,
            },
        )
    if len(packed.payload) % layout.alignment != 0:
        raise ArtifactError(
            f"payload length {len(packed.payload)} is not a multiple of the "
            f"declared alignment {layout.alignment}",
            details={
                "reason_code": REASON_ALIGNMENT,
                "alignment": layout.alignment,
            },
        )
    out: List[int] = []
    row_values_needed = layout.padded_cols
    for row in range(layout.rows):
        start = row * layout.row_stride_bytes
        row_bytes = packed.payload[start : start + layout.row_stride_bytes]
        packed_len = packed_row_bytes(scheme.bits, row_values_needed)
        if independent:
            row_values = unpack_independent(
                row_bytes[:packed_len],
                row_values_needed,
                scheme.bits,
                nibble_order=layout.nibble_order,
            )
        else:
            row_values = unpack_canonical(
                row_bytes[:packed_len],
                row_values_needed,
                scheme.bits,
                nibble_order=layout.nibble_order,
            )
        out.extend(row_values[: layout.cols])
    if len(out) != layout.rows * layout.cols:
        raise ArtifactError(
            "kernel-layout unpack produced the wrong number of values",
            details={"reason_code": REASON_BAD_LENGTH, "values": len(out)},
        )
    validate_codes(out, scheme)
    return out


def variants_summary(variants: Iterable[PackedTensor]) -> List[Dict[str, object]]:
    """Compact, deterministic summary for artifact manifests."""
    return [
        {
            "layout_id": variant.layout.layout_id,
            "layout_version": variant.layout.layout_version,
            "bytes": len(variant.payload),
            "payload_sha256": variant.payload_sha256(),
            "variant_hash": variant.variant_hash(),
            "parent_canonical_hash": variant.layout.parent_canonical_hash,
        }
        for variant in variants
    ]


__all__ = [
    "KNOWN_LAYOUTS",
    "LAYOUT_W4A16_HIFIRST_NK_V1",
    "LAYOUT_W4A16_ROWMAJOR_NK_V1",
    "LAYOUT_W8A16_ROWMAJOR_NK_V1",
    "NIBBLE_HIGH_FIRST",
    "NIBBLE_LOW_FIRST",
    "PACK_BYTE_V1",
    "PACK_NIBBLE_V1",
    "PackedTensor",
    "KernelLayout",
    "REASON_ALIGNMENT",
    "REASON_BAD_LENGTH",
    "REASON_BAD_PADDING",
    "REASON_CODE_OUT_OF_RANGE",
    "REASON_ODD_ROW",
    "REASON_PARENT_HASH",
    "REASON_UNKNOWN_LAYOUT",
    "REASON_UNKNOWN_PACK_VERSION",
    "canonical_pack_version",
    "pack_canonical",
    "pack_kernel_variant",
    "packed_row_bytes",
    "plan_kernel_layout",
    "unpack_canonical",
    "unpack_independent",
    "unpack_kernel_variant",
    "validate_codes",
    "variants_summary",
]
