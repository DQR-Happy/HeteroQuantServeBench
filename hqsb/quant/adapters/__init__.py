"""Industrial-method adapters (E05-04): convert *without* binding to a runtime.

The five layers every adapter must implement (E05-04 §4):

    SourceMethodConfig  ->  StatisticsCollector  ->  CanonicalQuantConverter
        ->  PackedVariantBuilder  ->  RuntimeCapabilityBinding

Design rules enforced here:

* the original library config is captured *and* the resolved defaults are
  expanded, so a library upgrade cannot change the meaning of a method name;
* nothing is dropped silently: every source field is ``mapped``,
  ``preserved`` (namespaced extension), ``derived`` or explicitly
  ``unsupported`` with a reason;
* the converter produces HQSB canonical values, never a foreign packed blob,
  so algorithm / packing / kernel effects stay separable;
* a missing library is a *capability* outcome (with a reason), not an
  exception and not a silent substitution.
"""

from __future__ import annotations

from hqsb.quant.adapters.base import (
    LEVELS,
    AdapterError,
    AdapterResult,
    CanonicalQuantConverter,
    FieldMapping,
    MethodMatrix,
    PackedVariantBuilder,
    RuntimeCapabilityBinding,
    SourceMethodConfig,
    SourceTensorRecord,
    audit_field_mapping,
    capability_probe,
    transform_equivalence_check,
)
from hqsb.quant.adapters.methods import (
    AWQ_ADAPTER,
    GPTQ_ADAPTER,
    METHOD_ADAPTERS,
    SMOOTHQUANT_ADAPTER,
    MethodAdapter,
    adapter_availability_report,
    get_adapter,
)

__all__ = [
    "AWQ_ADAPTER",
    "AdapterError",
    "AdapterResult",
    "CanonicalQuantConverter",
    "FieldMapping",
    "GPTQ_ADAPTER",
    "LEVELS",
    "METHOD_ADAPTERS",
    "MethodAdapter",
    "MethodMatrix",
    "PackedVariantBuilder",
    "RuntimeCapabilityBinding",
    "SMOOTHQUANT_ADAPTER",
    "SourceMethodConfig",
    "SourceTensorRecord",
    "adapter_availability_report",
    "audit_field_mapping",
    "capability_probe",
    "get_adapter",
    "transform_equivalence_check",
]
