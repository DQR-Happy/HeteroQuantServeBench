"""``QuantArtifact`` documents: canonical values, packed variants, integrity.

Layering (details README §5, E05-09 §2):

    QuantArtifact
      ├─ identity/provenance          model id + revision + per-tensor hashes
      ├─ canonical_quant              logical qvalues + scales/zeros + semantics
      ├─ packed_variants[]            backend/kernel/arch-specific bytes
      ├─ compatibility                kernel/arch/shape/workspace requirements
      ├─ calibration/method metadata  RTN(NONE/weight-stat) or industrial method
      └─ integrity                    per-file + manifest + identity hashes

Storing only an opaque packed blob is explicitly forbidden
(E05-01 §14): it would make algorithm, packing and kernel effects
indistinguishable. The canonical layer here is *not* an execution format — it
is the reproducible source from which any compatible kernel layout is
re-packed.

On-disk container (a directory, written atomically):

    <artifact_dir>/
      manifest.json          identity + semantics + compatibility + hashes
      canonical.bin          one int8 byte per logical value (auditable form)
      scales.bin             float32 scales, one per quantization unit
      zeros.bin              int32 zero points (absent for symmetric schemes)
      variants/<layout>.bin  packed kernel variants

Volatile fields (``created_at``, absolute paths, run ids, host names) are kept
in the manifest but excluded from :meth:`QuantArtifactDocument.identity_hash`,
so the same artifact written by two processes has the same identity
(E05-01 §11 step 7, §13 "两进程hash不同但数值相同").

Nothing here allocates a device buffer or launches a kernel; device-side
decisions belong to :mod:`hqsb.quant.compat` and the execution layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ArtifactError, ConfigError
from hqsb.quant import packing
from hqsb.quant.rtn import QuantizedTensor
from hqsb.quant.spec import QuantScheme

#: Document schema version of the S05 ``QuantArtifact`` container itself.
ARTIFACT_SCHEMA_VERSION = "1.0.0"
ARTIFACT_KIND = "hqsb.quant.artifact"

#: Fields excluded from the identity hash (volatile provenance only).
VOLATILE_FIELDS = ("created_at", "artifact_dir", "host", "run_id", "commands")

MANIFEST_FILENAME = "manifest.json"
CANONICAL_FILENAME = "canonical.bin"
SCALES_FILENAME = "scales.bin"
ZEROS_FILENAME = "zeros.bin"
VARIANTS_DIRNAME = "variants"

#: Reason codes for artifact-level refusals.
REASON_MANIFEST_MISSING = "manifest_missing"
REASON_MANIFEST_PARSE = "manifest_parse_error"
REASON_SCHEMA_VERSION = "schema_version_unsupported"
REASON_KIND = "artifact_kind_mismatch"
REASON_FILE_MISSING = "payload_file_missing"
REASON_FILE_HASH = "payload_hash_mismatch"
REASON_MANIFEST_HASH = "manifest_hash_mismatch"
REASON_SHAPE = "canonical_shape_mismatch"
REASON_COUNT = "canonical_count_mismatch"
REASON_PLANE_LENGTH = "scale_or_zero_length_mismatch"
REASON_MODEL_IDENTITY = "model_identity_mismatch"
REASON_TENSOR_IDENTITY = "tensor_identity_mismatch"
REASON_PARENT_HASH = "packed_variant_parent_hash_mismatch"
REASON_QUANT_SEMANTICS = "quant_semantics_mismatch"
REASON_INCOMPLETE = "artifact_incomplete"
REASON_MANIFEST_FIELD = "manifest_required_field_invalid"
REASON_PACKED_VALUES = "packed_bytes_value_mismatch"
REASON_PACKED_VALUES = "packed_bytes_value_mismatch"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _strip_volatile(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in VOLATILE_FIELDS}


@dataclass
class ModelIdentity:
    """The model identity an artifact is bound to (E05-09 §3.1).

    A shape check alone cannot notice "same shapes, different weights"; the
    per-tensor source hash and the model root are what make a wrong-weight
    artifact detectable.
    """

    model_id: str
    revision: str
    architecture: str = ""
    config_hash: str = ""
    tokenizer_hash: str = ""
    source_weight_sha256: str = ""
    model_root_sha256: str = ""
    tied_embeddings: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "architecture": self.architecture,
            "config_hash": self.config_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "source_weight_sha256": self.source_weight_sha256,
            "model_root_sha256": self.model_root_sha256,
            "tied_embeddings": self.tied_embeddings,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModelIdentity":
        return cls(**dict(payload))


@dataclass
class TensorIdentity:
    """The tensor (weight) identity: name, shape, dtype, layout, source hash."""

    name: str
    shape: Tuple[int, ...]
    dtype: str
    layout: str = "row_major"
    source_sha256: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
            "source_sha256": self.source_sha256,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TensorIdentity":
        data = dict(payload)
        data["shape"] = tuple(int(dim) for dim in data["shape"])
        return cls(**data)


@dataclass
class CalibrationProvenance:
    """Calibration provenance: NONE/weight-stat for RTN, dataset hash otherwise."""

    kind: str = "NONE"
    dataset_manifest_sha256: str = ""
    sample_count: int = 0
    valid_tokens: int = 0
    subset_seed: Optional[int] = None
    statistics_kind: str = ""
    domain: str = ""
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "sample_count": self.sample_count,
            "valid_tokens": self.valid_tokens,
            "subset_seed": self.subset_seed,
            "statistics_kind": self.statistics_kind,
            "domain": self.domain,
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CalibrationProvenance":
        return cls(**dict(payload))


@dataclass
class CompatibilityRecord:
    """Execution requirements and the compatibility of a packed variant.

    ``status`` is one of the enum values of
    :class:`hqsb.quant.compat.CompatibilityStatus`; the record keeps the
    *declared* requirements so a loader can refuse before allocating device
    memory (E05-09 §4).
    """

    kernel_id: str
    provider: str
    layout_id: str
    target_arch: str = "any"
    abi_version: str = "1"
    supported_bits: Tuple[int, ...] = ()
    supported_groups: Tuple[int, ...] = ()
    dtype: str = "float16"
    m_range: Tuple[int, int] = (1, 1 << 30)
    n_range: Tuple[int, int] = (1, 1 << 30)
    k_range: Tuple[int, int] = (1, 1 << 30)
    workspace_bytes: int = 0
    fallback_policy: str = "fail_closed"
    required_extension: str = ""
    status: str = "unverified"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "provider": self.provider,
            "layout_id": self.layout_id,
            "target_arch": self.target_arch,
            "abi_version": self.abi_version,
            "supported_bits": list(self.supported_bits),
            "supported_groups": list(self.supported_groups),
            "dtype": self.dtype,
            "m_range": list(self.m_range),
            "n_range": list(self.n_range),
            "k_range": list(self.k_range),
            "workspace_bytes": self.workspace_bytes,
            "fallback_policy": self.fallback_policy,
            "required_extension": self.required_extension,
            "status": self.status,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CompatibilityRecord":
        data = dict(payload)
        for key in ("supported_bits", "supported_groups", "m_range", "n_range", "k_range"):
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])
        return cls(**data)


@dataclass
class PackedVariantRecord:
    """A packed kernel variant as recorded in the manifest."""

    layout_id: str
    layout_version: str
    filename: str
    payload_bytes: int
    payload_sha256: str
    variant_hash: str
    parent_canonical_hash: str
    builder: str = "hqsb.quant.packing"
    builder_version: str = "1.0.0"
    target_arch: str = "any"
    alignment: int = 16
    layout_extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layout_id": self.layout_id,
            "layout_version": self.layout_version,
            "filename": self.filename,
            "payload_bytes": self.payload_bytes,
            "payload_sha256": self.payload_sha256,
            "variant_hash": self.variant_hash,
            "parent_canonical_hash": self.parent_canonical_hash,
            "builder": self.builder,
            "builder_version": self.builder_version,
            "target_arch": self.target_arch,
            "alignment": self.alignment,
            "layout_extra": dict(self.layout_extra),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PackedVariantRecord":
        return cls(**dict(payload))


@dataclass
class SizeBreakdown:
    """Itemized artifact byte accounting (E05-01 §10).

    ``total`` is the sum of the itemized parts; ``reconcile`` compares it with
    a measured number and explains the residual instead of hiding it.
    """

    canonical_q_bytes: int
    scale_bytes: int
    zero_bytes: int
    packed_variant_bytes: int
    metadata_bytes: int
    alignment_padding_bytes: int
    checksum_bytes: int

    @property
    def total(self) -> int:
        return (
            self.canonical_q_bytes
            + self.scale_bytes
            + self.zero_bytes
            + self.packed_variant_bytes
            + self.metadata_bytes
            + self.alignment_padding_bytes
            + self.checksum_bytes
        )

    @property
    def canonical_total(self) -> int:
        return self.canonical_q_bytes + self.scale_bytes + self.zero_bytes

    def as_dict(self) -> Dict[str, int]:
        return {
            "canonical_q_bytes": self.canonical_q_bytes,
            "scale_bytes": self.scale_bytes,
            "zero_bytes": self.zero_bytes,
            "packed_variant_bytes": self.packed_variant_bytes,
            "metadata_bytes": self.metadata_bytes,
            "alignment_padding_bytes": self.alignment_padding_bytes,
            "checksum_bytes": self.checksum_bytes,
            "canonical_total_bytes": self.canonical_total,
            "total_bytes": self.total,
        }

    def reconcile(self, measured_bytes: int) -> Dict[str, Any]:
        """Compare the predicted total with a measured size.

        Returns a machine-readable explanation; the caller records it in the
        experiment raw data. A residual is *reported*, never normalized away.
        """
        residual = int(measured_bytes) - self.total
        return {
            "predicted_total_bytes": self.total,
            "measured_total_bytes": int(measured_bytes),
            "residual_bytes": residual,
            "parts": self.as_dict(),
        }


@dataclass
class ValidationIssue:
    """One structured validation failure (stage + stable reason code)."""

    stage: str
    reason_code: str
    message: str
    field_path: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "reason_code": self.reason_code,
            "message": self.message,
            "field_path": self.field_path,
            "details": dict(self.details),
        }


@dataclass
class QuantArtifactDocument:
    """A complete, versioned quantization artifact."""

    tensor: TensorIdentity
    scheme: QuantScheme
    values: List[int]                      # canonical logical qvalues (flat)
    scales: List[float]
    zeros: List[int]
    units_per_row: int
    tail: int
    padded_shape: Tuple[int, ...]
    expected_dequant_dtype: str
    method: str = "rtn"
    method_config_hash: str = ""
    tool_version: str = "hqsb.quant/0.1.0"
    model: ModelIdentity = field(
        default_factory=lambda: ModelIdentity(model_id="", revision="")
    )
    calibration: CalibrationProvenance = field(default_factory=CalibrationProvenance)
    compatibility: List[CompatibilityRecord] = field(default_factory=list)
    variants: List[PackedVariantRecord] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)
    quality_references: Dict[str, Any] = field(default_factory=dict)
    license: str = "Apache-2.0"
    created_at: str = ""
    canonical_pack_version: str = ""
    variant_payloads: Dict[str, bytes] = field(default_factory=dict, repr=False)
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    kind: str = ARTIFACT_KIND
    artifact_dir: str = ""
    host: str = ""
    run_id: str = ""
    commands: List[str] = field(default_factory=list)
    manifest_sha256: str = ""

    # ── construction ───────────────────────────────────────────────────

    @classmethod
    def from_quantized(
        cls,
        qt: QuantizedTensor,
        *,
        tensor_name: str,
        source_dtype: str = "float16",
        source_sha256: str = "",
        model: Optional[ModelIdentity] = None,
        method: str = "rtn",
        method_config_hash: str = "",
        calibration: Optional[CalibrationProvenance] = None,
        compatibility: Optional[Sequence[CompatibilityRecord]] = None,
        known_limitations: Optional[Sequence[str]] = None,
        quality_references: Optional[Mapping[str, Any]] = None,
        source_layout: str = "row_major",
        padded_shape: Optional[Sequence[int]] = None,
    ) -> "QuantArtifactDocument":
        """Build a document from a :class:`QuantizedTensor` (no payloads yet)."""
        if not qt.q:
            raise ArtifactError(
                "cannot build an artifact from an empty tensor",
                details={"reason_code": REASON_SHAPE},
            )
        shape = qt.shape
        if padded_shape is None:
            padded = list(shape)
            if qt.group_size is not None and qt.axis == len(shape) - 1:
                padded[-1] = qt.units_per_row * qt.group_size
        else:
            padded = [int(dim) for dim in padded_shape]
        document = cls(
            tensor=TensorIdentity(
                name=tensor_name,
                shape=shape,
                dtype=source_dtype,
                layout=source_layout,
                source_sha256=source_sha256,
            ),
            scheme=qt.scheme,
            values=list(qt.q),
            scales=list(qt.scales),
            zeros=list(qt.zeros),
            units_per_row=qt.units_per_row,
            tail=qt.tail,
            padded_shape=tuple(padded),
            expected_dequant_dtype="float16",
            method=method,
            method_config_hash=method_config_hash,
            model=model or ModelIdentity(model_id="", revision=""),
            calibration=calibration or CalibrationProvenance(),
            compatibility=list(compatibility or ()),
            known_limitations=list(known_limitations or ()),
            quality_references=dict(quality_references or {}),
            canonical_pack_version=packing.canonical_pack_version(qt.scheme.bits),
            created_at=_utc_now(),
        )
        return document

    def add_variant(self, packed: packing.PackedTensor, *, target_arch: str = "any") -> str:
        """Attach a packed kernel variant; returns its manifest filename."""
        filename = f"{packed.layout.layout_id}.v{packed.layout.layout_version}.bin"
        payload = packed.payload
        record = PackedVariantRecord(
            layout_id=packed.layout.layout_id,
            layout_version=packed.layout.layout_version,
            filename=filename,
            payload_bytes=len(payload),
            payload_sha256=_sha256_bytes(payload),
            variant_hash=packed.variant_hash(),
            parent_canonical_hash=self.canonical_hash(),
            target_arch=target_arch,
            alignment=packed.layout.alignment,
            layout_extra={"layout": packed.layout.as_dict(), **dict(packed.layout_extra)},
        )
        self.variants = [v for v in self.variants if v.layout_id != record.layout_id]
        self.variants.append(record)
        self.variant_payloads[filename] = payload
        return filename

    # ── hashing ────────────────────────────────────────────────────────

    def canonical_bytes(self) -> bytes:
        """Canonical payload: one int8 byte per logical value."""
        return struct.pack(f"<{len(self.values)}b", *self.values)

    def scales_bytes(self) -> bytes:
        return struct.pack(f"<{len(self.scales)}f", *[float(s) for s in self.scales])

    def zeros_bytes(self) -> bytes:
        if self.scheme.symmetric:
            return b""
        return struct.pack(f"<{len(self.zeros)}i", *[int(z) for z in self.zeros])

    def canonical_hash(self) -> str:
        """Hash over canonical payload + *quant semantics* only.

        Deliberately excludes model identity, calibration provenance, license,
        timestamps and paths: this is the hash a packed variant declares as its
        parent (E05-01 §8 "identity与provenance字段分离"), and what a repack
        must preserve. A mutated ``model.revision`` must not look like a
        mutated weight.
        """
        digest = hashlib.sha256()
        digest.update(_canonical_json(self._canonical_semantics_payload()))
        digest.update(self.canonical_bytes())
        digest.update(self.scales_bytes())
        digest.update(self.zeros_bytes())
        return digest.hexdigest()

    def _canonical_semantics_payload(self) -> Dict[str, Any]:
        """Fields that define the quantized values and their exact semantics."""
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "tensor_shape": list(self.tensor.shape),
            "tensor_dtype": self.tensor.dtype,
            "scheme": self.scheme.as_dict(),
            "scheme_hash": self.scheme.scheme_hash(),
            "axis": self.scheme.axis,
            "units_per_row": self.units_per_row,
            "tail": self.tail,
            "padded_shape": list(self.padded_shape),
            "expected_dequant_dtype": self.expected_dequant_dtype,
            "canonical_pack_version": self.canonical_pack_version,
            "canonical_count": len(self.values),
            "scale_count": len(self.scales),
            "zero_count": len(self.zeros),
            "method": self.method,
            "method_config_hash": self.method_config_hash,
        }

    def _canonical_manifest_payload(self) -> Dict[str, Any]:
        return _strip_volatile(
            {
                "schema_version": self.schema_version,
                "kind": self.kind,
                "tensor": self.tensor.as_dict(),
                "scheme": self.scheme.as_dict(),
                "scheme_hash": self.scheme.scheme_hash(),
                "axis": self.scheme.axis,
                "units_per_row": self.units_per_row,
                "tail": self.tail,
                "padded_shape": list(self.padded_shape),
                "expected_dequant_dtype": self.expected_dequant_dtype,
                "canonical_pack_version": self.canonical_pack_version,
                "canonical_count": len(self.values),
                "scale_count": len(self.scales),
                "zero_count": len(self.zeros),
                "method": self.method,
                "method_config_hash": self.method_config_hash,
                "tool_version": self.tool_version,
                "model": self.model.as_dict(),
                "calibration": self.calibration.as_dict(),
                "compatibility": [record.as_dict() for record in self.compatibility],
                "known_limitations": list(self.known_limitations),
                "quality_references": dict(self.quality_references),
                "license": self.license,
                "created_at": self.created_at,
                "artifact_dir": self.artifact_dir,
                "host": self.host,
                "run_id": self.run_id,
                "commands": list(self.commands),
            }
        )

    def manifest_payload(self) -> Dict[str, Any]:
        """The full manifest document that is written to ``manifest.json``."""
        payload = self._canonical_manifest_payload()
        payload.update(
            {
                "canonical_file": {
                    "filename": CANONICAL_FILENAME,
                    "bytes": len(self.canonical_bytes()),
                    "sha256": _sha256_bytes(self.canonical_bytes()),
                },
                "scales_file": {
                    "filename": SCALES_FILENAME,
                    "bytes": len(self.scales_bytes()),
                    "sha256": _sha256_bytes(self.scales_bytes()),
                },
                "zeros_file": (
                    {
                        "filename": ZEROS_FILENAME,
                        "bytes": len(self.zeros_bytes()),
                        "sha256": _sha256_bytes(self.zeros_bytes()),
                    }
                    if not self.scheme.symmetric
                    else None
                ),
                "packed_variants": [record.as_dict() for record in self.variants],
                "canonical_hash": self.canonical_hash(),
            }
        )
        return payload

    def identity_hash(self) -> str:
        """Stable artifact identity (content only; excludes volatile fields)."""
        return _sha256_bytes(_canonical_json(_strip_volatile(self.manifest_payload())))

    def artifact_id(self) -> str:
        """Short, human-facing id derived from the identity hash."""
        return f"qa-{self.identity_hash()[:16]}"

    # ── size accounting ────────────────────────────────────────────────

    def size_breakdown(self, *, metadata_bytes: Optional[int] = None) -> SizeBreakdown:
        """Itemized byte accounting for the artifact as it would be written.

        ``alignment_padding_bytes`` counts the padding a kernel layout adds
        (rows padded to ``row_stride_bytes``) so the difference between
        "logical bytes" and "packed bytes" is visible rather than attributed
        to the compression ratio.
        """
        padding = 0
        for record in self.variants:
            layout_extra = record.layout_extra or {}
            layout = layout_extra.get("layout", {})
            row_stride = int(layout.get("row_stride_bytes", 0) or 0)
            rows = int(layout.get("rows", 0) or 0)
            cols = int(layout.get("cols", 0) or 0)
            padded_cols = int(layout.get("padded_cols", cols) or cols)
            if row_stride and rows:
                layout_bytes = packing.packed_row_bytes(self.scheme.bits, padded_cols)
                padding += rows * max(0, row_stride - layout_bytes)
        if metadata_bytes is None:
            metadata_bytes = len(_canonical_json(self.manifest_payload()))
        return SizeBreakdown(
            canonical_q_bytes=len(self.canonical_bytes()),
            scale_bytes=len(self.scales_bytes()),
            zero_bytes=len(self.zeros_bytes()),
            packed_variant_bytes=sum(len(payload) for payload in self.variant_payloads.values()),
            metadata_bytes=int(metadata_bytes),
            alignment_padding_bytes=padding,
            # One sha256 (32 B) per payload file + one for the manifest.
            checksum_bytes=32 * (3 + len(self.variants)),
        )

    def fp16_equivalent_bytes(self) -> int:
        """Bytes the same tensor would occupy as FP16 (compression baseline)."""
        num_values = 1
        for dim in self.tensor.shape:
            num_values *= dim
        return num_values * 2

    def compression_ratio(self, *, runtime_bytes: Optional[int] = None) -> float:
        """``FP16_bytes / total_runtime_bytes`` (E05-01 §10).

        The default uses the canonical + packed + metadata accounting; callers
        that have a *measured* device number must pass it explicitly, because
        the theoretical ratio is not a runtime memory claim.
        """
        total = runtime_bytes if runtime_bytes is not None else self.size_breakdown().total
        if total <= 0:
            return float("inf")
        return self.fp16_equivalent_bytes() / float(total)

    # ── C5 interoperability ────────────────────────────────────────────

    def to_c5(self):
        """Project onto the stable C5 ``QuantArtifact`` contract.

        The rich S05 document is a superset of C5; other layers keep consuming
        the stable contract (C5 stays version 1.0.0 — the project does not
        silently change a frozen contract).
        """
        from hqsb.core.contracts.quant import QuantArtifact as C5QuantArtifact

        return C5QuantArtifact(
            algorithm=self.method,
            bits=self.scheme.bits,
            granularity=self.scheme.granularity,
            symmetric=self.scheme.symmetric,
            group_size=self.scheme.group_size,
            scale=f"sha256:{_sha256_bytes(self.scales_bytes())}",
            zero_point=(
                f"sha256:{_sha256_bytes(self.zeros_bytes())}"
                if not self.scheme.symmetric
                else None
            ),
            calibration=(
                self.calibration.dataset_manifest_sha256 or None
                if self.calibration.kind != "NONE"
                else None
            ),
            packing=self.variants[0].layout_id if self.variants else self.canonical_pack_version,
            kernel_compatibility=(
                self.compatibility[0].kernel_id if self.compatibility else "unverified"
            ),
            accuracy=dict(self.quality_references) or None,
        )

    # ── persistence ────────────────────────────────────────────────────

    def save(self, artifact_dir: str, *, host: Optional[str] = None, run_id: str = "") -> str:
        """Write the artifact atomically; returns the artifact directory.

        Atomicity (E05-09 §7): the container is written into a temporary
        sibling directory, fsynced, then renamed onto ``artifact_dir``. A
        reader can therefore only ever see a complete artifact; an interrupted
        write leaves no partially-visible manifest.
        """
        parent = os.path.dirname(os.path.abspath(artifact_dir)) or "."
        os.makedirs(parent, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix=".hqsb-quant-", dir=parent)
        try:
            self.artifact_dir = ""
            self.host = host or self.host
            self.run_id = run_id or self.run_id
            if not self.created_at:
                self.created_at = _utc_now()

            canonical = self.canonical_bytes()
            scales = self.scales_bytes()
            zeros = self.zeros_bytes()
            _write_file(os.path.join(tmp_dir, CANONICAL_FILENAME), canonical)
            _write_file(os.path.join(tmp_dir, SCALES_FILENAME), scales)
            if zeros:
                _write_file(os.path.join(tmp_dir, ZEROS_FILENAME), zeros)
            variants_dir = os.path.join(tmp_dir, VARIANTS_DIRNAME)
            os.makedirs(variants_dir, exist_ok=True)
            for filename, payload in self.variant_payloads.items():
                _write_file(os.path.join(variants_dir, filename), payload)

            manifest = self.manifest_payload()
            manifest_bytes = json.dumps(
                manifest, sort_keys=True, indent=2, ensure_ascii=False
            ).encode("utf-8")
            manifest_path = os.path.join(tmp_dir, MANIFEST_FILENAME)
            _write_file(manifest_path, manifest_bytes)
            self.manifest_sha256 = _sha256_bytes(manifest_bytes)

            # Persist the directory metadata before the atomic rename.
            dir_fd = os.open(tmp_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

            if os.path.exists(artifact_dir):
                raise ArtifactError(
                    f"refusing to overwrite existing artifact directory "
                    f"{artifact_dir!r}; remove it explicitly",
                    details={"reason_code": REASON_INCOMPLETE, "path": artifact_dir},
                )
            os.rename(tmp_dir, artifact_dir)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        self.artifact_dir = os.path.abspath(artifact_dir)
        return self.artifact_dir


def _write_file(path: str, payload: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── loading and validation ────────────────────────────────────────────────


def load_manifest(artifact_dir: str) -> Dict[str, Any]:
    """Read and parse ``manifest.json`` (structure stage only)."""
    manifest_path = os.path.join(artifact_dir, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        raise ArtifactError(
            f"no manifest at {manifest_path!r}",
            details={
                "reason_code": REASON_MANIFEST_MISSING,
                "artifact_dir": artifact_dir,
            },
        )
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"manifest at {manifest_path!r} is not readable JSON: {exc}",
            details={"reason_code": REASON_MANIFEST_PARSE, "artifact_dir": artifact_dir},
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactError(
            "manifest root must be a JSON object",
            details={"reason_code": REASON_MANIFEST_PARSE},
        )
    return payload


def _check_schema(manifest: Mapping[str, Any]) -> None:
    from hqsb.core.schema.versioning import SchemaVersion

    kind = manifest.get("kind")
    if kind != ARTIFACT_KIND:
        raise ArtifactError(
            f"manifest kind {kind!r} is not {ARTIFACT_KIND!r}",
            details={"reason_code": REASON_KIND, "kind": kind},
        )
    raw_version = manifest.get("schema_version")
    if not isinstance(raw_version, str):
        raise ArtifactError(
            "manifest is missing a string 'schema_version'",
            details={"reason_code": REASON_SCHEMA_VERSION},
        )
    received = SchemaVersion.parse(raw_version)
    current = SchemaVersion.parse(ARTIFACT_SCHEMA_VERSION)
    if received > current:
        raise ArtifactError(
            f"artifact schema {received} is newer than supported {current}; "
            f"refusing to guess a future layout",
            details={
                "reason_code": REASON_SCHEMA_VERSION,
                "received": str(received),
                "supported": str(current),
            },
        )
    if received < current:
        raise ArtifactError(
            f"artifact schema {received} is older than {current} and requires "
            f"an explicit migration",
            details={
                "reason_code": REASON_SCHEMA_VERSION,
                "received": str(received),
                "supported": str(current),
            },
        )


def _verify_payload(artifact_dir: str, filename: str, expected_sha: str, stage: str) -> bytes:
    path = os.path.join(artifact_dir, filename)
    if not os.path.isfile(path):
        raise ArtifactError(
            f"payload file {filename!r} is missing",
            details={
                "reason_code": REASON_FILE_MISSING,
                "stage": stage,
                "filename": filename,
            },
        )
    payload = open(path, "rb").read()
    actual = _sha256_bytes(payload)
    if actual != expected_sha:
        raise ArtifactError(
            f"payload file {filename!r} hash mismatch",
            details={
                "reason_code": REASON_FILE_HASH,
                "stage": stage,
                "filename": filename,
                "expected_sha256": expected_sha,
                "actual_sha256": actual,
            },
        )
    return payload


def load_document(
    artifact_dir: str,
    *,
    verify_payloads: bool = True,
    verify_variants: bool = True,
    enforce_canonical_hash: bool = True,
) -> QuantArtifactDocument:
    """Load a ``QuantArtifact`` from disk, verifying structure and integrity.

    Order (E05-09 §4): manifest presence → schema version → payload presence
    and hashes → canonical semantics. Capability/model checks happen in
    :mod:`hqsb.quant.compat`, before any device allocation.
    """
    manifest = load_manifest(artifact_dir)
    _check_schema(manifest)
    _require_fields(
        manifest,
        {
            "tensor": dict,
            "scheme": dict,
            "canonical_file": dict,
            "scales_file": dict,
            "canonical_count": int,
        },
    )

    tensor = TensorIdentity.from_mapping(manifest["tensor"])
    # Unknown scheme keys are refused rather than dropped: every field in the
    # scheme changes the math, so a reader that cannot interpret one must
    # fail closed (E05-09 §15 "unknown critical 字段被忽略" is a forbidden
    # pattern).
    try:
        scheme = QuantScheme.from_mapping(dict(manifest["scheme"]))
    except (ConfigError, TypeError, KeyError) as exc:
        raise ArtifactError(
            f"artifact scheme is invalid: {exc}",
            details={"reason_code": REASON_QUANT_SEMANTICS, "stage": "quant"},
        ) from exc
    if manifest.get("scheme_hash") and manifest["scheme_hash"] != scheme.scheme_hash():
        raise ArtifactError(
            "declared scheme_hash does not match the scheme fields",
            details={
                "reason_code": REASON_QUANT_SEMANTICS,
                "stage": "quant",
                "declared": manifest["scheme_hash"],
                "computed": scheme.scheme_hash(),
            },
        )

    canonical_bytes = b""
    scales_bytes = b""
    zeros_bytes = b""
    if verify_payloads:
        try:
            canonical_info = manifest["canonical_file"]
            canonical_bytes = _verify_payload(
                artifact_dir,
                canonical_info["filename"],
                canonical_info["sha256"],
                "schema",
            )
            scales_info = manifest["scales_file"]
            scales_bytes = _verify_payload(
                artifact_dir, scales_info["filename"], scales_info["sha256"], "schema"
            )
            if manifest.get("zeros_file"):
                zeros_info = manifest["zeros_file"]
                zeros_bytes = _verify_payload(
                    artifact_dir, zeros_info["filename"], zeros_info["sha256"], "schema"
                )
            for record in manifest.get("packed_variants", []):
                _verify_payload(
                    artifact_dir,
                    os.path.join(VARIANTS_DIRNAME, record["filename"]),
                    record["payload_sha256"],
                    "pack",
                )
        except (KeyError, TypeError) as exc:
            raise ArtifactError(
                f"manifest payload section is malformed: {exc}",
                details={
                    "reason_code": REASON_MANIFEST_FIELD,
                    "problem": "malformed_payload_section",
                },
            ) from exc

    count = int(manifest.get("canonical_count", -1))
    if count < 0 or len(canonical_bytes) != count:
        raise ArtifactError(
            f"canonical payload has {len(canonical_bytes)} bytes but the "
            f"manifest declares {count} values",
            details={
                "reason_code": REASON_COUNT,
                "stage": "model",
                "expected": count,
                "actual_bytes": len(canonical_bytes),
            },
        )
    num_values = 1
    for dim in tensor.shape:
        num_values *= dim
    if num_values != count:
        raise ArtifactError(
            f"tensor shape {list(tensor.shape)} implies {num_values} values but "
            f"the canonical payload has {count}",
            details={
                "reason_code": REASON_SHAPE,
                "stage": "model",
                "shape": list(tensor.shape),
                "count": count,
            },
        )

    values = list(struct.unpack(f"<{count}b", canonical_bytes)) if count else []
    try:
        packing.validate_codes(values, scheme)
    except ArtifactError as exc:
        exc.details.setdefault("stage", "quant")
        raise

    scale_count = int(manifest.get("scale_count", -1))
    if scale_count < 0 or len(scales_bytes) != 4 * scale_count:
        raise ArtifactError(
            f"scales payload has {len(scales_bytes)} bytes but the manifest "
            f"declares {scale_count} float32 scales",
            details={
                "reason_code": REASON_PLANE_LENGTH,
                "stage": "quant",
                "expected_count": scale_count,
                "actual_bytes": len(scales_bytes),
            },
        )
    scales = list(struct.unpack(f"<{scale_count}f", scales_bytes)) if scale_count else []
    zero_count = int(manifest.get("zero_count", 0))
    if scheme.symmetric:
        if zero_count:
            raise ArtifactError(
                "symmetric artifact must not declare zero points",
                details={
                    "reason_code": REASON_QUANT_SEMANTICS,
                    "stage": "quant",
                    "zero_count": zero_count,
                },
            )
        zeros: List[int] = []
    else:
        if len(zeros_bytes) != 4 * zero_count:
            raise ArtifactError(
                f"zeros payload has {len(zeros_bytes)} bytes but the manifest "
                f"declares {zero_count} int32 values",
                details={
                    "reason_code": REASON_PLANE_LENGTH,
                    "stage": "quant",
                    "expected_count": zero_count,
                    "actual_bytes": len(zeros_bytes),
                },
            )
        zeros = list(struct.unpack(f"<{zero_count}i", zeros_bytes)) if zero_count else []

    units_per_row = int(manifest.get("units_per_row", 1))
    axis = int(manifest.get("axis", scheme.axis))
    if scheme.granularity != "per_tensor" and tensor.shape:
        normalized_axis = axis + len(tensor.shape) if axis < 0 else axis
        if not 0 <= normalized_axis < len(tensor.shape):
            raise ArtifactError(
                f"quantization axis {axis} is out of range for shape "
                f"{list(tensor.shape)}",
                details={
                    "reason_code": REASON_SHAPE,
                    "stage": "model",
                    "axis": axis,
                },
            )
        rows = num_values // max(1, tensor.shape[normalized_axis])
        expected_units = rows * units_per_row
        if scale_count != expected_units:
            raise ArtifactError(
                f"scale count {scale_count} does not match the shape/scheme "
                f"derived unit count {expected_units}",
                details={
                    "reason_code": REASON_PLANE_LENGTH,
                    "stage": "quant",
                    "scale_count": scale_count,
                    "expected": expected_units,
                },
            )
    for index, scale in enumerate(scales):
        import math as _math

        if not (_math.isfinite(scale) and scale > 0.0):
            raise ArtifactError(
                f"scale[{index}] = {scale!r} is not finite and positive",
                details={
                    "reason_code": REASON_QUANT_SEMANTICS,
                    "stage": "quant",
                    "index": index,
                    "scale": scale,
                },
            )

    document = QuantArtifactDocument(
        tensor=tensor,
        scheme=scheme,
        values=values,
        scales=scales,
        zeros=zeros,
        units_per_row=int(manifest.get("units_per_row", 1)),
        tail=int(manifest.get("tail", 0)),
        padded_shape=tuple(int(dim) for dim in manifest.get("padded_shape", tensor.shape)),
        expected_dequant_dtype=str(manifest.get("expected_dequant_dtype", "float16")),
        method=str(manifest.get("method", "unknown")),
        method_config_hash=str(manifest.get("method_config_hash", "")),
        tool_version=str(manifest.get("tool_version", "")),
        model=ModelIdentity.from_mapping(manifest.get("model", {"model_id": "", "revision": ""})),
        calibration=CalibrationProvenance.from_mapping(
            manifest.get("calibration", {})
        ),
        compatibility=[
            CompatibilityRecord.from_mapping(item)
            for item in manifest.get("compatibility", [])
        ],
        variants=[
            PackedVariantRecord.from_mapping(item)
            for item in manifest.get("packed_variants", [])
        ],
        known_limitations=list(manifest.get("known_limitations", [])),
        quality_references=dict(manifest.get("quality_references", {})),
        license=str(manifest.get("license", "")),
        created_at=str(manifest.get("created_at", "")),
        canonical_pack_version=str(manifest.get("canonical_pack_version", "")),
        artifact_dir=os.path.abspath(artifact_dir),
        schema_version=ARTIFACT_SCHEMA_VERSION,
    )
    # ``enforce_canonical_hash=False`` exists for *artifact tooling* that needs
    # to recompute the hash of a deliberately mutated fixture (E05-09 fault
    # injection). The production loader always verifies it.
    if (
        enforce_canonical_hash
        and manifest.get("canonical_hash")
        and manifest["canonical_hash"] != document.canonical_hash()
    ):
        raise ArtifactError(
            "recomputed canonical_hash does not match the manifest",
            details={
                "reason_code": REASON_QUANT_SEMANTICS,
                "stage": "quant",
                "declared": manifest["canonical_hash"],
                "computed": document.canonical_hash(),
            },
        )
    if verify_payloads and verify_variants:
        # A checksum proves the bytes are the writer's; decoding them proves
        # they still mean the same values (E05-09 §8.4/§14).
        for record in document.variants:
            verify_variant_against_canonical(document, record)
    manifest_path = os.path.join(artifact_dir, MANIFEST_FILENAME)
    document.manifest_sha256 = _sha256_file(manifest_path)
    return document


def _require_fields(manifest: Mapping[str, Any], spec: Mapping[str, type]) -> None:
    """Fail closed when a required manifest field is missing or has the wrong type.

    A missing field must never be default-filled (E05-09 §15): a reader that
    silently invents a default would interpret different quantization
    semantics than the writer intended.
    """
    for field_name, expected_type in spec.items():
        if field_name not in manifest:
            raise ArtifactError(
                f"manifest is missing required field {field_name!r}",
                details={
                    "reason_code": REASON_MANIFEST_FIELD,
                    "stage": "schema",
                    "field_path": field_name,
                    "problem": "missing",
                },
            )
        value = manifest[field_name]
        if expected_type is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        else:
            ok = isinstance(value, expected_type)
        if not ok:
            raise ArtifactError(
                f"manifest field {field_name!r} has type "
                f"{type(value).__name__}, expected {expected_type.__name__}",
                details={
                    "reason_code": REASON_MANIFEST_FIELD,
                    "stage": "schema",
                    "field_path": field_name,
                    "problem": "wrong_type",
                    "actual_type": type(value).__name__,
                    "expected_type": expected_type.__name__,
                },
            )


def load_variant_payloads(document: QuantArtifactDocument) -> Dict[str, bytes]:
    """Read every packed variant payload from the artifact directory."""
    payloads: Dict[str, bytes] = {}
    for record in document.variants:
        path = os.path.join(document.artifact_dir, VARIANTS_DIRNAME, record.filename)
        payload = open(path, "rb").read()
        payloads[record.filename] = payload
    return payloads


def verify_variant_against_canonical(
    document: QuantArtifactDocument,
    record: PackedVariantRecord,
    *,
    independent_decoder: bool = True,
) -> Dict[str, Any]:
    """Unpack a packed variant and compare it with the canonical values.

    A file checksum only proves the bytes are the ones the writer wrote; it
    cannot prove the bytes *decode to the same values* (wrong layout, swapped
    planes, another tensor's payload with a refreshed hash). E05-09 §14
    ("同一量化值换layout") and §8.4 require the loader to compare semantics,
    not just hashes. The independent decoder is used by default so pack and
    load cannot share one indexing bug.
    """
    rows, cols = _weight_matrix_shape(document.tensor.shape)
    layout_extra = record.layout_extra or {}
    layout_payload = layout_extra.get("layout", {})
    payload = open(
        os.path.join(document.artifact_dir, VARIANTS_DIRNAME, record.filename), "rb"
    ).read()
    if int(layout_payload.get("rows", rows)) != rows or int(
        layout_payload.get("cols", cols)
    ) != cols:
        raise ArtifactError(
            "packed variant layout dimensions do not match the tensor shape",
            details={
                "reason_code": REASON_SHAPE,
                "stage": "pack",
                "layout_rows": layout_payload.get("rows"),
                "layout_cols": layout_payload.get("cols"),
                "tensor_rows": rows,
                "tensor_cols": cols,
            },
        )
    packed = packing.PackedTensor(
        layout=packing.plan_kernel_layout(
            document.scheme,
            rows,
            cols,
            layout_id=record.layout_id,
            alignment=record.alignment,
            parent_canonical_hash=record.parent_canonical_hash,
        ),
        payload=payload,
        scales=list(document.scales),
        zeros=list(document.zeros),
        layout_extra=dict(layout_extra),
    )
    recovered = packing.unpack_kernel_variant(
        packed, document.scheme, independent=independent_decoder
    )
    if recovered != list(document.values):
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(recovered, document.values))
                if left != right
            ),
            -1,
        )
        raise ArtifactError(
            "packed variant decodes to values that differ from the canonical "
            "layer; refusing to execute mis-encoded bytes",
            details={
                "reason_code": REASON_PACKED_VALUES,
                "stage": "pack",
                "first_mismatch": mismatch,
                "layout_id": record.layout_id,
            },
        )
    return {
        "layout_id": record.layout_id,
        "values_checked": len(recovered),
        "independent_decoder": independent_decoder,
        "parent_canonical_hash": record.parent_canonical_hash,
    }


def _weight_matrix_shape(shape: Sequence[int]) -> Tuple[int, int]:
    if len(shape) == 0:
        raise ArtifactError(
            "rank-0 tensors cannot carry a weight layout",
            details={"reason_code": REASON_SHAPE, "stage": "model"},
        )
    if len(shape) == 1:
        return 1, int(shape[0])
    rows = 1
    for dim in shape[:-1]:
        rows *= int(dim)
    return rows, int(shape[-1])


def load_variant_planes(document: QuantArtifactDocument, record: PackedVariantRecord) -> Dict[str, Any]:
    """Return the scale/zero planes and layout metadata for one variant.

    The planes are shared by every variant of one tensor (they are properties
    of the quantization scheme, not of the byte layout); the layout metadata
    tells the kernel how to address them.
    """
    return {
        "scales": list(document.scales),
        "zeros": list(document.zeros),
        "layout_id": record.layout_id,
        "layout_version": record.layout_version,
        "layout_extra": dict(record.layout_extra),
        "parent_canonical_hash": record.parent_canonical_hash,
    }


def validate_artifact_dir(artifact_dir: str, *, verify_payloads: bool = True) -> Dict[str, Any]:
    """Run the structural/integrity stages and return a machine-readable report.

    Used by the experiment drivers and by E05-09 fault injection: the same
    function must both accept golden artifacts and refuse every mutated one
    *before* any device allocation.
    """
    stages: List[Dict[str, Any]] = []
    try:
        document = load_document(
            artifact_dir,
            verify_payloads=verify_payloads,
            verify_variants=verify_payloads,
        )
    except ArtifactError as exc:
        details = exc.details or {}
        return {
            "accepted": False,
            "identity_hash": "",
            "artifact_id": "",
            "error": exc.message,
            "reason_code": str(details.get("reason_code", "")),
            "stage": str(details.get("stage", "")) or "schema",
            "field_path": str(details.get("field_path", "")),
            "details": dict(details),
            "stages": stages,
        }
    stages.append({"stage": "schema", "status": "pass"})
    stages.append({"stage": "model", "status": "pass"})
    stages.append({"stage": "quant", "status": "pass"})
    if document.variants:
        stages.append({"stage": "pack", "status": "pass"})
    return {
        "accepted": True,
        "identity_hash": document.identity_hash(),
        "artifact_id": document.artifact_id(),
        "canonical_hash": document.canonical_hash(),
        "error": None,
        "reason_code": "",
        "stage": "ok",
        "field_path": "",
        "details": {},
        "stages": stages,
    }


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_SCHEMA_VERSION",
    "CANONICAL_FILENAME",
    "CalibrationProvenance",
    "CompatibilityRecord",
    "MANIFEST_FILENAME",
    "ModelIdentity",
    "PackedVariantRecord",
    "QuantArtifactDocument",
    "REASON_COUNT",
    "REASON_FILE_HASH",
    "REASON_FILE_MISSING",
    "REASON_INCOMPLETE",
    "REASON_KIND",
    "REASON_MANIFEST_FIELD",
    "REASON_MANIFEST_MISSING",
    "REASON_MANIFEST_PARSE",
    "REASON_PACKED_VALUES",
    "REASON_MODEL_IDENTITY",
    "REASON_PARENT_HASH",
    "REASON_PLANE_LENGTH",
    "REASON_QUANT_SEMANTICS",
    "REASON_SCHEMA_VERSION",
    "REASON_SHAPE",
    "REASON_TENSOR_IDENTITY",
    "SCALES_FILENAME",
    "SizeBreakdown",
    "TensorIdentity",
    "VARIANTS_DIRNAME",
    "ValidationIssue",
    "VOLATILE_FIELDS",
    "ZEROS_FILENAME",
    "load_document",
    "load_manifest",
    "load_variant_payloads",
    "load_variant_planes",
    "validate_artifact_dir",
]
