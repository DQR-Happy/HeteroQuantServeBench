"""Versioned HQSB contracts (C1–C7).

Stable data contracts and the abstract Backend interface shared by every
module in the project. Importing this package must never import a concrete
backend, operator, model loader, or serving implementation.
"""

from hqsb.core.contracts.base import VersionedModel
from hqsb.core.contracts.model import ModelArtifact
from hqsb.core.contracts.workload import WorkloadSpec
from hqsb.core.contracts.operator import OperatorSpec, TensorSpec
from hqsb.core.contracts.backend import (
    Backend,
    BackendCapability,
    GenerationOutput,
    GenerationSample,
)
from hqsb.core.contracts.quant import QuantArtifact
from hqsb.core.contracts.result import (
    BenchmarkResult,
    CorrectnessReport,
    EnvironmentInfo,
    ResourceUsage,
)
from hqsb.core.contracts.trace import TraceEvent, TraceEventType
from hqsb.core.contracts.registry import (
    ContractEntry,
    load_contract_registry,
    registry_checks,
)

__all__ = [
    "Backend",
    "BackendCapability",
    "BenchmarkResult",
    "ContractEntry",
    "CorrectnessReport",
    "EnvironmentInfo",
    "GenerationOutput",
    "GenerationSample",
    "ModelArtifact",
    "OperatorSpec",
    "QuantArtifact",
    "ResourceUsage",
    "TensorSpec",
    "TraceEvent",
    "TraceEventType",
    "VersionedModel",
    "WorkloadSpec",
    "load_contract_registry",
    "registry_checks",
]
