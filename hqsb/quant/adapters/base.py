"""Adapter primitives shared by every industrial-method adapter (E05-04 §4).

Nothing here imports a third-party quantization library. A real run passes a
:class:`SourceTensorRecord` that the library produced (or that the library's
saved artifact contains); the adapter converts it into HQSB canonical values
and audits the mapping. That structure is what makes "algorithm vs packing vs
kernel" separable, and it lets the whole adapter layer be unit-tested without
the library installed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.quant import packing
from hqsb.quant.rtn import dequantize_flat
from hqsb.quant.spec import QuantScheme

#: Fair-comparison levels (E05-04 §5).
A_ALGORITHM = "A"
B_COMMON_PACKING = "B"
C_NATIVE_RUNTIME = "C"
D_OFFLINE_COST = "D"
LEVELS = (A_ALGORITHM, B_COMMON_PACKING, C_NATIVE_RUNTIME, D_OFFLINE_COST)

#: Field-mapping statuses (E05-04 §7.1) — no silent drops.
MAPPED = "mapped"
PRESERVED = "preserved-as-extension"
DERIVED = "derived"
UNSUPPORTED = "unsupported"

MAPPING_STATUSES = (MAPPED, PRESERVED, DERIVED, UNSUPPORTED)


class AdapterError(ConfigError):
    """An adapter refused to convert (with a machine-readable reason)."""


@dataclass
class FieldMapping:
    """One source field's fate during conversion."""

    source_field: str
    target_field: str
    status: str
    loss_reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in MAPPING_STATUSES:
            raise AdapterError(
                f"unknown mapping status {self.status!r}; supported: "
                f"{list(MAPPING_STATUSES)}"
            )
        if self.status == UNSUPPORTED and not self.loss_reason:
            raise AdapterError(
                f"field {self.source_field!r} marked unsupported without a "
                f"loss_reason; an unexplained drop is forbidden"
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_field": self.source_field,
            "target_field": self.target_field,
            "status": self.status,
            "loss_reason": self.loss_reason,
        }


def audit_field_mapping(
    source_fields: Sequence[str], mappings: Sequence[FieldMapping]
) -> Dict[str, Any]:
    """Audit a mapping table against the *complete* source field list.

    Every source field must appear exactly once; a field that is present in
    the source record but absent from the table is reported as
    ``unmapped_fields`` and fails the audit (E05-04 §7.1 "禁止silent drop").
    """
    covered = {mapping.source_field for mapping in mappings}
    duplicates = [
        field_name
        for field_name in covered
        if sum(1 for mapping in mappings if mapping.source_field == field_name) > 1
    ]
    unmapped = [name for name in source_fields if name not in covered]
    unknown = [name for name in covered if name not in set(source_fields)]
    unsupported = [
        mapping.as_dict()
        for mapping in mappings
        if mapping.status == UNSUPPORTED
    ]
    return {
        "audit_passed": not unmapped and not duplicates and not unknown,
        "mapped": sorted(
            mapping.source_field for mapping in mappings if mapping.status == MAPPED
        ),
        "preserved": sorted(
            mapping.source_field
            for mapping in mappings
            if mapping.status == PRESERVED
        ),
        "derived": sorted(
            mapping.source_field
            for mapping in mappings
            if mapping.status == DERIVED
        ),
        "unsupported": unsupported,
        "unmapped_fields": sorted(unmapped),
        "unknown_fields": sorted(unknown),
        "duplicate_fields": sorted(set(duplicates)),
        "coverage": (
            (len(source_fields) - len(unmapped)) / len(source_fields)
            if source_fields
            else 1.0
        ),
    }


@dataclass
class SourceMethodConfig:
    """Layer 1: the third-party method configuration (E05-04 §4.1).

    ``public_config`` is what the user wrote, ``resolved_config`` is the
    library's defaults expanded, and ``config_hash`` covers both plus the
    version — so "GPTQ defaults" can never silently change meaning.
    """

    method: str
    package: str
    version: str
    public_config: Dict[str, Any] = field(default_factory=dict)
    resolved_config: Dict[str, Any] = field(default_factory=dict)
    commit: str = ""
    build_flags: Sequence[str] = ()
    model_mapping: Dict[str, str] = field(default_factory=dict)
    skipped_modules: Sequence[str] = ()
    unsupported_modules: Sequence[str] = ()
    warnings: Sequence[str] = ()
    calibration_manifest_sha256: str = ""
    license: str = ""
    package_available: bool = True
    unavailable_reason: str = ""

    @property
    def config_hash(self) -> str:
        payload = {
            "method": self.method,
            "package": self.package,
            "version": self.version,
            "commit": self.commit,
            "build_flags": list(self.build_flags),
            "public_config": self.public_config,
            "resolved_config": self.resolved_config,
            "model_mapping": dict(sorted(self.model_mapping.items())),
            "skipped_modules": sorted(self.skipped_modules),
            "unsupported_modules": sorted(self.unsupported_modules),
            "calibration_manifest_sha256": self.calibration_manifest_sha256,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "package": self.package,
            "version": self.version,
            "commit": self.commit,
            "build_flags": list(self.build_flags),
            "public_config": dict(self.public_config),
            "resolved_config": dict(self.resolved_config),
            "model_mapping": dict(sorted(self.model_mapping.items())),
            "skipped_modules": list(self.skipped_modules),
            "unsupported_modules": list(self.unsupported_modules),
            "warnings": list(self.warnings),
            "calibration_manifest_sha256": self.calibration_manifest_sha256,
            "license": self.license,
            "package_available": self.package_available,
            "unavailable_reason": self.unavailable_reason,
            "config_hash": self.config_hash,
        }


@dataclass
class SourceTensorRecord:
    """Layer 2 input: one quantized tensor as the source method produced it.

    The record is *self-describing*: it carries the bit width, symmetry,
    group axis/size, zero-point presence and — critically — the byte layout of
    its packed payload (``source_nibble_order``, ``source_container_bits``).
    A converter therefore never has to guess a library's internal order, and a
    wrong declaration is caught by the equivalence test instead of being
    interpreted as valid data.
    """

    name: str
    shape: Tuple[int, ...]
    bits: int
    symmetric: bool
    group_size: Optional[int]
    packed_payload: bytes
    scales: Sequence[float]
    zeros: Sequence[int] = ()
    #: raw source metadata, retained verbatim under a namespaced key
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    source_nibble_order: str = packing.NIBBLE_LOW_FIRST
    source_axis: int = -1
    container: str = "nibble"
    pre_scale: Optional[Sequence[float]] = None
    post_scale: Optional[Sequence[float]] = None

    def source_fields(self) -> List[str]:
        fields = {
            "name",
            "shape",
            "bits",
            "symmetric",
            "group_size",
            "packed_payload",
            "scales",
            "zeros",
            "source_nibble_order",
            "source_axis",
            "container",
            *self.source_metadata.keys(),
        }
        # Transform planes are part of the source artifact whenever present.
        # Omitting them here would let an AWQ/SmoothQuant adapter pass the
        # no-silent-drop audit while silently losing the very transform that
        # distinguishes the method from plain RTN.
        if self.pre_scale is not None:
            fields.add("pre_scale")
        if self.post_scale is not None:
            fields.add("post_scale")
        return sorted(fields)

    def payload_sha256(self) -> str:
        return hashlib.sha256(self.packed_payload).hexdigest()


@dataclass
class CanonicalQuantConverter:
    """Layer 3: source record → HQSB canonical values (E05-04 §4.2).

    The converter *unpacks* the source payload with the record's declared
    layout and returns canonical ``(q, scales, zeros, scheme)``. It never
    re-packs and never normalizes away a difference: an unsupported field is
    reported by :func:`audit_field_mapping`, not fixed here.
    """

    method: str
    scheme_template: QuantScheme

    def scheme_for(self, record: SourceTensorRecord) -> QuantScheme:
        return self.scheme_template.with_overrides(
            bits=record.bits,
            symmetric=record.symmetric,
            group_size=record.group_size,
            label=f"{self.method}_b{record.bits}",
        )

    def convert(
        self, record: SourceTensorRecord
    ) -> Dict[str, Any]:
        scheme = self.scheme_for(record)
        count = 1
        for dim in record.shape:
            count *= dim
        if record.container == "nibble" and record.bits == 4:
            qvalues = packing.unpack_independent(
                record.packed_payload,
                count,
                record.bits,
                nibble_order=record.source_nibble_order,
            )
        elif record.container in ("byte", "int8") or record.bits == 8:
            qvalues = packing.unpack_independent(record.packed_payload, count, 8)
        else:
            raise AdapterError(
                f"unsupported source container {record.container!r} for "
                f"{record.bits}-bit; add an explicit mapping instead of guessing"
            )
        packing.validate_codes(qvalues, scheme)
        rows = 1
        for dim in record.shape[:-1]:
            rows *= dim
        units_per_row = scheme.unit_count(record.shape[-1]) if record.shape else 0
        total_units = rows * units_per_row if scheme.granularity != "per_tensor" else 1
        if len(record.scales) != total_units:
            raise AdapterError(
                f"source scales ({len(record.scales)}) do not match the "
                f"shape/group derived unit count ({total_units}) for "
                f"{record.name!r}"
            )
        if not record.symmetric and len(record.zeros) != len(record.scales):
            raise AdapterError(
                f"asymmetric source tensor {record.name!r} must carry one zero "
                f"point per unit"
            )
        return {
            "name": record.name,
            "scheme": scheme,
            "qvalues": qvalues,
            "scales": [float(scale) for scale in record.scales],
            "zeros": [int(zero) for zero in record.zeros],
            "source_payload_sha256": record.payload_sha256(),
            "source_metadata_namespaced": {
                f"source.{key}": value
                for key, value in sorted(record.source_metadata.items())
            },
            "pre_scale": (
                [float(value) for value in record.pre_scale]
                if record.pre_scale is not None
                else None
            ),
            "post_scale": (
                [float(value) for value in record.post_scale]
                if record.post_scale is not None
                else None
            ),
        }

    def field_mappings(self, record: SourceTensorRecord) -> List[FieldMapping]:
        mappings = [
            FieldMapping("name", "tensor.name", MAPPED),
            FieldMapping("shape", "tensor.shape", MAPPED),
            FieldMapping("bits", "scheme.bits", MAPPED),
            FieldMapping("symmetric", "scheme.symmetric", MAPPED),
            FieldMapping("group_size", "scheme.group_size", MAPPED),
            FieldMapping("scales", "canonical.scales", MAPPED),
            FieldMapping("zeros", "canonical.zeros", MAPPED),
            FieldMapping("packed_payload", "canonical.qvalues", DERIVED),
            FieldMapping("source_nibble_order", "packing.nibble_order", DERIVED),
            FieldMapping("source_axis", "scheme.axis", DERIVED),
            FieldMapping("container", "canonical_pack_version", DERIVED),
        ]
        for key in sorted(record.source_metadata):
            mappings.append(
                FieldMapping(key, f"provenance.source.{key}", PRESERVED)
            )
        if record.pre_scale is not None:
            mappings.append(FieldMapping("pre_scale", "transform.pre_scale", MAPPED))
        if record.post_scale is not None:
            mappings.append(FieldMapping("post_scale", "transform.post_scale", MAPPED))
        return mappings


@dataclass
class PackedVariantBuilder:
    """Layer 4: canonical values → HQSB kernel layouts (E05-04 §4.3)."""

    builder: str = "hqsb.quant.adapters"
    builder_version: str = "1.0.0"

    def build(
        self,
        converted: Mapping[str, Any],
        rows: int,
        cols: int,
        *,
        layout_id: Optional[str] = None,
        alignment: int = 16,
        parent_canonical_hash: Optional[str] = None,
    ) -> packing.PackedTensor:
        scheme = converted["scheme"]
        return packing.pack_kernel_variant(
            converted["qvalues"],
            converted["scales"],
            converted["zeros"],
            scheme,
            rows,
            cols,
            layout_id=layout_id,
            alignment=alignment,
            parent_canonical_hash=parent_canonical_hash,
        )


@dataclass
class RuntimeCapabilityBinding:
    """Layer 5: what the target runtime can actually execute (E05-04 §4.4)."""

    backend: str
    kernel_id: str
    supported_bits: Tuple[int, ...]
    supported_scheme: str
    supported_groups: Tuple[Optional[int], ...]
    dtype: str = "float16"
    m_range: Tuple[int, int] = (1, 1 << 30)
    n_range: Tuple[int, int] = (1, 1 << 30)
    k_range: Tuple[int, int] = (1, 1 << 30)
    workspace_bytes: int = 0
    abi_version: str = "1"
    available: bool = True
    unavailable_reason: str = ""
    fallback_policy: str = "fail_closed"
    observed_kernel_verification: str = "profiler_symbol"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "kernel_id": self.kernel_id,
            "supported_bits": list(self.supported_bits),
            "supported_scheme": self.supported_scheme,
            "supported_groups": list(self.supported_groups),
            "dtype": self.dtype,
            "m_range": list(self.m_range),
            "n_range": list(self.n_range),
            "k_range": list(self.k_range),
            "workspace_bytes": self.workspace_bytes,
            "abi_version": self.abi_version,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "fallback_policy": self.fallback_policy,
            "observed_kernel_verification": self.observed_kernel_verification,
        }


@dataclass
class AdapterResult:
    """The outcome of one adapter run, with a per-tensor audit trail."""

    method: str
    method_config: SourceMethodConfig
    level: str = A_ALGORITHM
    converted: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    field_audits: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    equivalence: Dict[str, Any] = field(default_factory=dict)
    canonical_hashes: Dict[str, str] = field(default_factory=dict)
    offline_cost_s: float = 0.0
    failures: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise AdapterError(f"unknown comparison level {self.level!r}")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "level": self.level,
            "method_config": self.method_config.as_dict(),
            "converted_tensors": sorted(self.converted),
            "field_audits": self.field_audits,
            "equivalence": self.equivalence,
            "canonical_hashes": self.canonical_hashes,
            "offline_cost_s": self.offline_cost_s,
            "failures": list(self.failures),
        }


def capability_probe(package: str) -> Dict[str, Any]:
    """Check whether a third-party package is importable, with a reason."""
    spec = importlib.util.find_spec(package)
    if spec is None:
        return {
            "package": package,
            "available": False,
            "reason": "package not installed (capability, not an error)",
        }
    version = ""
    try:
        module = importlib.import_module(package)
        version = str(getattr(module, "__version__", ""))
    except Exception as exc:  # noqa: BLE001 - a broken install is a capability state
        return {
            "package": package,
            "available": False,
            "reason": f"installed but not importable: {type(exc).__name__}: {exc}",
        }
    return {"package": package, "available": True, "version": version}


def transform_equivalence_check(
    original_fn: Callable[[Any], Any],
    transformed_fn: Callable[[Any], Any],
    inputs: Sequence[Any],
    *,
    tolerance: Mapping[str, float],
    output_extractor: Optional[Callable[[Any], Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Verify a pre/post transform in the *unquantized* model (E05-04 §7.3).

    AWQ/SmoothQuant fold a scaling into neighbours; if the fold is wrong the
    model is already broken before quantization. The check runs both callables
    on identical inputs and compares the outputs with the pre-registered
    tolerance. A failure here means "stop and fix the rewrite", never "the
    quantization is lossy".
    """
    extractor = output_extractor or _default_extractor
    from hqsb.benchmark.metrics import numerical_diff_summary

    rows: List[Dict[str, Any]] = []
    passed = True
    for index, sample in enumerate(inputs):
        original_values = extractor(original_fn(sample))
        transformed_values = extractor(transformed_fn(sample))
        summary = numerical_diff_summary(list(original_values), list(transformed_values))
        ok = (
            summary["max_abs_error"] <= tolerance.get("max_abs", float("inf"))
            and summary["l2_relative_error"] <= tolerance.get("relative_l2", float("inf"))
            and summary["cosine_similarity"] >= tolerance.get("cosine_min", 0.0)
        )
        passed = passed and ok
        rows.append({"sample": index, "passed": ok, "metrics": summary})
    return {
        "passed": passed,
        "samples": len(rows),
        "tolerance": dict(tolerance),
        "records": rows,
        "note": (
            "an unquantized transform that is not equivalent invalidates any "
            "later quantization-quality comparison"
        ),
    }


def _default_extractor(output: Any) -> Sequence[float]:
    if hasattr(output, "detach"):
        return [float(value) for value in output.detach().reshape(-1).tolist()]
    if isinstance(output, (list, tuple)):
        flattened: List[float] = []
        for item in output:
            flattened.extend(_default_extractor(item))
        return flattened
    return [float(output)]


@dataclass
class MethodMatrix:
    """Frozen method matrix (E05-04 §10 step 1)."""

    methods: Sequence[str]
    library_versions: Mapping[str, str]
    module_scope: str
    scheme_policy: str
    calibration_budget: Mapping[str, Any]
    quality_gate: Mapping[str, Any]
    comparison_levels: Sequence[str] = LEVELS
    tunable_parameters: Mapping[str, Sequence[Any]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "methods": list(self.methods),
                "library_versions": dict(self.library_versions),
                "module_scope": self.module_scope,
                "scheme_policy": self.scheme_policy,
                "calibration_budget": dict(self.calibration_budget),
                "quality_gate": dict(self.quality_gate),
                "comparison_levels": list(self.comparison_levels),
                "tunable_parameters": {
                    key: list(value)
                    for key, value in sorted(self.tunable_parameters.items())
                },
            },
            sort_keys=True,
            indent=2,
        )

    @property
    def matrix_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def dequantized_reference(converted: Mapping[str, Any], rows: int, cols: int) -> List[float]:
    """Dequantize a converted record with HQSB's own math (equivalence oracle)."""
    return dequantize_flat(
        converted["qvalues"],
        converted["scales"],
        converted["zeros"],
        (rows, cols),
        converted["scheme"].unit_count(cols),
        converted["scheme"],
        group_size=converted["scheme"].group_size,
        axis=converted["scheme"].axis,
    )


__all__ = [
    "A_ALGORITHM",
    "AdapterError",
    "AdapterResult",
    "B_COMMON_PACKING",
    "C_NATIVE_RUNTIME",
    "CanonicalQuantConverter",
    "D_OFFLINE_COST",
    "DERIVED",
    "FieldMapping",
    "LEVELS",
    "MAPPED",
    "MAPPING_STATUSES",
    "MethodMatrix",
    "PackedVariantBuilder",
    "PRESERVED",
    "RuntimeCapabilityBinding",
    "SourceMethodConfig",
    "SourceTensorRecord",
    "UNSUPPORTED",
    "audit_field_mapping",
    "capability_probe",
    "dequantized_reference",
    "transform_equivalence_check",
]
