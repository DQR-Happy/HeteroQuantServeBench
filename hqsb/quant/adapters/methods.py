"""Concrete adapter declarations for GPTQ, AWQ and SmoothQuant (E05-04 §3).

Each adapter is *declarative*: it names the library, the parameters that must
be resolved, the fields the source artifact provides, the pre/post transform
(if any) and the runtime capability it needs. The numeric conversion is done
by :class:`~hqsb.quant.adapters.base.CanonicalQuantConverter`, which works from
the self-describing source record — so a real run can attach a library's
output and get a canonical artifact plus a mapping audit, while a unit test
can attach a synthetic record with no library installed.

Nothing here claims a library behaviour. ``default_parameters`` lists the
defaults that MUST be expanded from the installed version at run time
(``resolve_defaults``), and ``known_differences`` lists the semantic
differences the literature/implementation documents — each of which is a
"structural variable" that has to be reported rather than normalized away
(E05-04 §3.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.quant.adapters.base import (
    A_ALGORITHM,
    CanonicalQuantConverter,
    FieldMapping,
    RuntimeCapabilityBinding,
    SourceMethodConfig,
    SourceTensorRecord,
    audit_field_mapping,
    capability_probe,
)
from hqsb.quant.rtn import dequantize_flat
from hqsb.quant.spec import GRANULARITY_PER_GROUP, QuantScheme


@dataclass(frozen=True)
class MethodAdapter:
    """One industrial method's declarative adapter description."""

    method: str
    package: str
    import_name: str
    parameters_public: Tuple[str, ...]
    parameters_resolved: Tuple[str, ...]
    source_fields: Tuple[str, ...]
    default_level: str
    transform: str = "none"
    known_differences: Tuple[str, ...] = ()
    extra_extension_fields: Tuple[str, ...] = ()
    required_capability: Mapping[str, Any] = field(default_factory=dict)

    def probe(self) -> Dict[str, Any]:
        return capability_probe(self.import_name if self.import_name else self.package)

    def source_config(
        self,
        *,
        version: str = "",
        public_config: Optional[Mapping[str, Any]] = None,
        resolved_config: Optional[Mapping[str, Any]] = None,
        model_mapping: Optional[Mapping[str, str]] = None,
        skipped_modules: Sequence[str] = (),
        unsupported_modules: Sequence[str] = (),
        calibration_manifest_sha256: str = "",
        unavailable_reason: str = "",
    ) -> SourceMethodConfig:
        probe = self.probe()
        if not probe["available"] and not unavailable_reason:
            unavailable_reason = str(probe["reason"])
        return SourceMethodConfig(
            method=self.method,
            package=self.package,
            version=version or str(probe.get("version", "")),
            public_config=dict(public_config or {}),
            resolved_config=dict(resolved_config or {}),
            model_mapping=dict(model_mapping or {}),
            skipped_modules=tuple(skipped_modules),
            unsupported_modules=tuple(unsupported_modules),
            calibration_manifest_sha256=calibration_manifest_sha256,
            package_available=bool(probe["available"]),
            unavailable_reason=unavailable_reason,
        )

    def field_mappings(self) -> List[FieldMapping]:
        """Declared mapping table for the method's source fields."""
        table = {
            "qweight": FieldMapping("qweight", "canonical.qvalues", "derived"),
            "qzeros": FieldMapping("qzeros", "canonical.zeros", "derived"),
            "scales": FieldMapping("scales", "canonical.scales", "mapped"),
            "g_idx": FieldMapping(
                "g_idx", "provenance.source.g_idx", "preserved-as-extension"
            ),
            "group_size": FieldMapping("group_size", "scheme.group_size", "mapped"),
            "bits": FieldMapping("bits", "scheme.bits", "mapped"),
            "sym": FieldMapping("sym", "scheme.symmetric", "mapped"),
            "version": FieldMapping(
                "version", "provenance.source.version", "preserved-as-extension"
            ),
            # AutoAWQ uses different public/source names for the same
            # canonical semantics.  They are explicit aliases, not
            # unsupported fields.
            "w_bit": FieldMapping("w_bit", "scheme.bits", "mapped"),
            "q_group_size": FieldMapping(
                "q_group_size", "scheme.group_size", "mapped"
            ),
            "zero_point": FieldMapping(
                "zero_point", "scheme.symmetric+canonical.zeros", "derived"
            ),
        }
        if self.transform != "none":
            table["smooth_scale"] = FieldMapping(
                "smooth_scale", "transform.scale", "derived"
            )
            table["alpha"] = FieldMapping(
                "alpha", "provenance.source.alpha", "preserved-as-extension"
            )
        for extra in self.extra_extension_fields:
            table[extra] = FieldMapping(
                extra, f"provenance.source.{extra}", "preserved-as-extension"
            )
        # Any field the method declares but the table does not know is
        # "unsupported" *by declaration* rather than silently ignored.
        for field_name in self.source_fields:
            table.setdefault(
                field_name,
                FieldMapping(
                    field_name,
                    "",
                    "unsupported",
                    loss_reason="no canonical target declared for this field",
                ),
            )
        # A per-method mapping table contains exactly the fields declared by
        # that source format.  Returning mappings for another library's
        # fields would make ``unknown_fields`` fail even when this adapter is
        # complete, and would blur which source schema was actually audited.
        return [table[field_name] for field_name in self.source_fields]

    def converter(self) -> CanonicalQuantConverter:
        scheme = QuantScheme(
            bits=4,
            granularity=GRANULARITY_PER_GROUP,
            group_size=128,
            label=f"{self.method}_template",
        )
        return CanonicalQuantConverter(method=self.method, scheme_template=scheme)

    def convert_record(
        self, record: SourceTensorRecord, *, rows: int, cols: int
    ) -> Dict[str, Any]:
        """Convert one source tensor and audit the mapping (no silent drops)."""
        converter = self.converter()
        converted = converter.convert(record)
        mappings = converter.field_mappings(record)
        audit = audit_field_mapping(record.source_fields(), mappings)
        converted["mapping_audit"] = audit
        return converted

    def capability_binding(self, **overrides: Any) -> RuntimeCapabilityBinding:
        payload = dict(self.required_capability)
        payload.update(overrides)
        return RuntimeCapabilityBinding(**payload)

    def equivariance_check(
        self,
        record: SourceTensorRecord,
        source_dequant: Sequence[float],
        *,
        tolerance: Mapping[str, float],
        rows: Optional[int] = None,
        cols: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compare the source library's dequant with HQSB canonical dequant.

        E05-04 §7.2: if the bitwise qvalues differ after unpacking, the
        adapter fails *before* any model evaluation. The check also returns
        the per-tensor error so a scale-dtype explanation can be attached
        instead of an unexplained mismatch.
        """
        from hqsb.benchmark.metrics import numerical_diff_summary

        resolved_rows = rows or (
            record.shape[0] if len(record.shape) >= 2 else 1
        )
        resolved_cols = cols or record.shape[-1]
        converted = self.convert_record(record, rows=resolved_rows, cols=resolved_cols)
        scheme = converted["scheme"]
        canonical = dequantize_flat(
            converted["qvalues"],
            converted["scales"],
            converted["zeros"],
            (resolved_rows, resolved_cols),
            scheme.unit_count(resolved_cols),
            scheme,
            group_size=scheme.group_size,
            axis=scheme.axis,
        )
        if len(source_dequant) != len(canonical):
            raise ValueError(
                f"source dequant has {len(source_dequant)} values, canonical has "
                f"{len(canonical)}"
            )
        summary = numerical_diff_summary(
            [float(value) for value in source_dequant], canonical
        )
        passed = (
            summary["max_abs_error"] <= tolerance.get("max_abs", float("inf"))
            and summary["l2_relative_error"] <= tolerance.get("relative_l2", float("inf"))
        )
        return {
            "passed": passed,
            "tensor": record.name,
            "metrics": summary,
            "tolerance": dict(tolerance),
            "mapping_audit": converted["mapping_audit"],
        }


GPTQ_ADAPTER = MethodAdapter(
    method="gptq",
    package="gptqmodel",
    import_name="gptqmodel",
    parameters_public=("bits", "group_size", "damp_percent", "desc_act", "sym"),
    parameters_resolved=(
        "damp_percent",
        "desc_act",
        "static_groups",
        "true_sequential",
        "act_order",
        "blocksize",
    ),
    source_fields=("qweight", "qzeros", "scales", "g_idx", "group_size", "bits", "sym"),
    default_level=A_ALGORITHM,
    known_differences=(
        "group axis is the input-feature axis but qweight packing order is "
        "library-specific and must be declared by the source record",
        "act-order/desc_act changes the column order: the canonical converter "
        "must reorder back before any quality comparison",
        "asymmetric qzeros may be stored as -0 offset; the sign convention is "
        "a structural variable",
    ),
    required_capability={
        "backend": "triton",
        "kernel_id": "hqsb.w4a16.triton",
        "supported_bits": (4,),
        "supported_scheme": "symmetric_or_asymmetric_group",
        "supported_groups": (None, 32, 64, 128),
    },
)

AWQ_ADAPTER = MethodAdapter(
    method="awq",
    package="autoawq",
    import_name="awq",
    parameters_public=("w_bit", "q_group_size", "zero_point", "version"),
    parameters_resolved=("zero_point", "version", "duo_scaling", "apply_clip"),
    source_fields=("qweight", "qzeros", "scales", "w_bit", "q_group_size", "zero_point"),
    default_level=A_ALGORITHM,
    transform="awq_scale",
    known_differences=(
        "AWQ folds an activation-aware scale into the weight and the inverse "
        "into the previous operator; the fold point is a graph property that "
        "must be verified with the unquantized transform equivalence check",
        "AWQ's nibble order differs from GPTQ's; the source record must declare "
        "it and the converter must not assume low-first",
    ),
    required_capability={
        "backend": "triton",
        "kernel_id": "hqsb.w4a16.triton",
        "supported_bits": (4,),
        "supported_scheme": "symmetric_or_asymmetric_group",
        "supported_groups": (None, 64, 128),
    },
)

SMOOTHQUANT_ADAPTER = MethodAdapter(
    method="smoothquant",
    package="smoothquant",
    import_name="smoothquant",
    parameters_public=("alpha", "quantize_weights", "smooth_scale", "calib_dataset"),
    parameters_resolved=("alpha", "act_quant_granularity", "fold_position"),
    source_fields=("smooth_scale", "alpha", "scales", "bits", "sym"),
    default_level=A_ALGORITHM,
    transform="smoothquant_scale",
    known_differences=(
        "SmoothQuant's main target is W8A8, so comparing it inside a W4A16 "
        "table is a structural mismatch rather than a ranking",
        "the scale can be folded into the preceding norm instead of the weight; "
        "the fold position changes which operator must be replaced",
        "alpha selection must be made on policy-validation only",
    ),
    required_capability={
        "backend": "triton",
        "kernel_id": "hqsb.w8a8.triton",
        "supported_bits": (8,),
        "supported_scheme": "asymmetric_per_token_static",
        "supported_groups": (None,),
    },
)

METHOD_ADAPTERS: Dict[str, MethodAdapter] = {
    GPTQ_ADAPTER.method: GPTQ_ADAPTER,
    AWQ_ADAPTER.method: AWQ_ADAPTER,
    SMOOTHQUANT_ADAPTER.method: SMOOTHQUANT_ADAPTER,
}


def get_adapter(method: str) -> MethodAdapter:
    if method not in METHOD_ADAPTERS:
        raise KeyError(
            f"unknown method {method!r}; known: {sorted(METHOD_ADAPTERS)}"
        )
    return METHOD_ADAPTERS[method]


def adapter_availability_report() -> Dict[str, Any]:
    """Probe every declared method and report availability *with reasons*.

    Used by the E05-04 driver: a missing library makes that method's run
    ``BLOCKED`` with a reason, never silently replaced by another method.
    """
    report: Dict[str, Any] = {}
    for method, adapter in sorted(METHOD_ADAPTERS.items()):
        probe = adapter.probe()
        report[method] = {
            **probe,
            "declared_level": adapter.default_level,
            "transform": adapter.transform,
            "source_fields": list(adapter.source_fields),
        }
    return report


__all__ = [
    "AWQ_ADAPTER",
    "GPTQ_ADAPTER",
    "METHOD_ADAPTERS",
    "MethodAdapter",
    "SMOOTHQUANT_ADAPTER",
    "adapter_availability_report",
    "get_adapter",
]
