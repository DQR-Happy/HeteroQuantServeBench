"""C1–C7 versioned-contract schema registry (E01-01 step 1).

The registry is the single, machine-readable statement of *which* versioned
schema documents HQSB ships and *what their contract boundaries are*:

* contract name and schema version,
* the implementing Pydantic model (parser == ``model_validate``,
  serializer == ``model_dump_json``, structural validation is inherited),
* a canonical schema digest (SHA256 of the JSON schema) so a claim about a
  schema can be bound to exact bytes,
* the frozen unknown-field policy (``forbid``),
* the frozen legacy policy (explicit migration or refusal — never silent).

Loading the registry and asserting its completeness is the first gate of
E01-01: if one of C1–C7 is missing, unversioned, or has no parser, the rest
of the validation experiment must not be reported as a pass.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Type

from hqsb.core.contracts.backend import BackendCapability
from hqsb.core.contracts.base import VersionedModel
from hqsb.core.contracts.model import ModelArtifact
from hqsb.core.contracts.operator import OperatorSpec
from hqsb.core.contracts.quant import QuantArtifact
from hqsb.core.contracts.result import BenchmarkResult
from hqsb.core.contracts.trace import TraceEvent
from hqsb.core.contracts.workload import WorkloadSpec

#: Unknown-field policy applied by every VersionedModel.
UNKNOWN_FIELD_POLICY = "forbid"
#: Legacy policy: older payloads must be explicitly migrated or refused.
LEGACY_POLICY = "requires_migration"

#: C6 nests a full ``WorkloadSpec``; its JSON schema is therefore regenerated
#: only after the class has been fully rebuilt (see result.py).
BenchmarkResult.model_rebuild()

_CONTRACTS: Tuple[Tuple[str, str, Type[VersionedModel], str], ...] = (
    ("C1", "ModelArtifact", ModelArtifact, "hqsb.core.contracts.model"),
    ("C2", "WorkloadSpec", WorkloadSpec, "hqsb.core.contracts.workload"),
    ("C3", "OperatorSpec", OperatorSpec, "hqsb.core.contracts.operator"),
    (
        "C4",
        "BackendCapability",
        BackendCapability,
        "hqsb.core.contracts.backend",
    ),
    ("C5", "QuantArtifact", QuantArtifact, "hqsb.core.contracts.quant"),
    ("C6", "BenchmarkResult", BenchmarkResult, "hqsb.core.contracts.result"),
    ("C7", "TraceEvent", TraceEvent, "hqsb.core.contracts.trace"),
)


@dataclass(frozen=True)
class ContractEntry:
    """Registry metadata for one versioned contract document."""

    contract: str            # e.g. "C1"
    name: str                # e.g. "ModelArtifact"
    schema_version: str      # e.g. "1.0.0"
    module_path: str         # canonical module that defines the model
    schema_sha256: str       # SHA256 over the canonical JSON schema
    unknown_field_policy: str
    legacy_policy: str
    model_cls: Type[VersionedModel]

    @property
    def parser(self) -> str:
        """Parser identifier (Pydantic model validation on a dict)."""
        return f"{self.module_path}.{self.name}.model_validate"

    @property
    def serializer(self) -> str:
        """Serializer identifier (Pydantic JSON dump)."""
        return f"{self.module_path}.{self.name}.model_dump_json"


def _canonical_schema_sha256(model_cls: Type[VersionedModel]) -> str:
    """Return a stable SHA256 over the model's canonical JSON schema."""
    schema = model_cls.model_json_schema()
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_contract_registry() -> List[ContractEntry]:
    """Build the C1–C7 contract registry snapshot.

    Returns:
        One :class:`ContractEntry` per contract, ordered C1…C7.
    """
    entries: List[ContractEntry] = []
    for contract, name, model_cls, module_path in _CONTRACTS:
        entries.append(
            ContractEntry(
                contract=contract,
                name=name,
                schema_version=model_cls.SCHEMA_VERSION,
                module_path=module_path,
                schema_sha256=_canonical_schema_sha256(model_cls),
                unknown_field_policy=UNKNOWN_FIELD_POLICY,
                legacy_policy=LEGACY_POLICY,
                model_cls=model_cls,
            )
        )
    return entries


def registry_checks(
    entries: Iterable[ContractEntry],
) -> Dict[str, bool]:
    """Verify registry integrity (E01-01 step 1)."""
    listed = list(entries)
    names = [entry.contract for entry in listed]
    result: Dict[str, bool] = {
        "all_seven_registered": {e.contract for e in listed} == {
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
        },
        "no_duplicate_names": len(set(names)) == len(names),
        "every_contract_has_version": all(
            bool(entry.schema_version) for entry in listed
        ),
        "every_contract_has_parser_serializer": all(
            bool(entry.parser) and bool(entry.serializer) for entry in listed
        ),
        "every_contract_has_canonical_fixture": all(
            entry.schema_sha256 and entry.unknown_field_policy
            for entry in listed
        ),
    }
    return result


__all__ = [
    "ContractEntry",
    "LEGACY_POLICY",
    "UNKNOWN_FIELD_POLICY",
    "load_contract_registry",
    "registry_checks",
]
