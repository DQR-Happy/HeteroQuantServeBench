"""HQSB core: stable contracts, config, registry, errors, and identifiers.

The ``core`` package is the dependency-free foundation of the project. It
must never import a concrete backend, operator implementation, model loader,
or serving module — those depend on ``core``, not the reverse.
"""

from hqsb.core.errors import (
    ArtifactError,
    BackendError,
    BenchmarkError,
    CapabilityError,
    ConfigError,
    DuplicateRegistrationError,
    ExitCode,
    HqsbError,
    RegistryError,
    RegistryLookupError,
    SchemaError,
    SchemaVersionError,
    UsageError,
    exit_code_for,
)
from hqsb.core.ids import new_run_id, new_span_id, new_trace_id
from hqsb.core.logging import (
    JsonLineFormatter,
    configure_logging,
    get_span_id,
    get_trace_id,
    set_trace_context,
)
from hqsb.core.contracts import (
    Backend,
    BackendCapability,
    BenchmarkResult,
    CorrectnessReport,
    EnvironmentInfo,
    GenerationOutput,
    GenerationSample,
    ModelArtifact,
    OperatorSpec,
    QuantArtifact,
    ResourceUsage,
    TensorSpec,
    TraceEvent,
    TraceEventType,
    VersionedModel,
    WorkloadSpec,
)
from hqsb.core.schema import SchemaVersion, migrate_document
from hqsb.core.config import ConfigLoader, config_hash, deep_merge
from hqsb.core.registry import Registry, RegistryHub
from hqsb.core.fingerprint import (
    CommitSection,
    ConfigSection,
    DeviceSection,
    FingerprintSections,
    ModelSection,
    OsSection,
    PowerSection,
    PythonSection,
    RunFingerprint,
    VolatileObservations,
    compute_run_fingerprint,
    diff_sections,
)

__all__ = [
    # errors
    "ArtifactError",
    "BackendError",
    "BenchmarkError",
    "CapabilityError",
    "ConfigError",
    "DuplicateRegistrationError",
    "ExitCode",
    "HqsbError",
    "RegistryError",
    "RegistryLookupError",
    "SchemaError",
    "SchemaVersionError",
    "UsageError",
    "exit_code_for",
    # ids
    "new_run_id",
    "new_span_id",
    "new_trace_id",
    # logging
    "JsonLineFormatter",
    "configure_logging",
    "get_span_id",
    "get_trace_id",
    "set_trace_context",
    # contracts
    "Backend",
    "BackendCapability",
    "BenchmarkResult",
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
    # schema
    "SchemaVersion",
    "migrate_document",
    # config
    "ConfigLoader",
    "config_hash",
    "deep_merge",
    # registry
    "Registry",
    "RegistryHub",
    # fingerprint
    "CommitSection",
    "ConfigSection",
    "DeviceSection",
    "FingerprintSections",
    "ModelSection",
    "OsSection",
    "PowerSection",
    "PythonSection",
    "RunFingerprint",
    "VolatileObservations",
    "compute_run_fingerprint",
    "diff_sections",
]
