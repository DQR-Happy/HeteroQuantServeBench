"""Backend/harness identity, capability probing and completion semantics (E10-02).

Details E10-02 is emphatic about two things this module enforces structurally:

* ``enqueue`` is not ``completion`` — :class:`TimingCalibration` refuses to
  derive a collective latency from a host API return time;
* an unprobed capability is ``UNKNOWN`` and **never** authorises execution
  (details E10-02 step 9); the matrix is data, and the decision carries a reason
  code.

The module also holds the requested/actual record required by handbook §5.7, so
a forced algorithm that silently fell back to ``auto`` cannot be reported as if
the forced one ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Supported collective backends (details README §16).
BACKENDS: Tuple[str, ...] = ("nccl", "hccl", "gloo", "other")

#: Capability states: UNKNOWN never authorises execution.
CAPABILITY_STATES: Tuple[str, ...] = ("SUPPORTED", "UNSUPPORTED", "UNKNOWN")

#: Algorithm/protocol override modes (details E10-02 step 10).
ALGORITHM_MODES: Tuple[str, ...] = ("auto", "forced")

#: Algorithm families a forced group may name (backend support decides).
ALGORITHM_FAMILIES: Tuple[str, ...] = ("ring", "tree", "hierarchical", "halving_doubling", "nvls")

#: Protocol names the backends expose.
PROTOCOLS: Tuple[str, ...] = ("LL", "LL128", "Simple", "auto")


@dataclass(frozen=True)
class BackendIdentity:
    """Freeze the backend binary/toolchain identity (details E10-02 step 2)."""

    kind: str
    version: str
    framework: str = ""
    framework_version: str = ""
    source_commit: str = ""
    build_flags: str = ""
    launcher: str = ""
    binary_sha256: str = ""
    official_tool: str = ""
    official_tool_version: str = ""

    def __post_init__(self) -> None:
        if self.kind not in BACKENDS:
            raise ConfigError(
                f"backend kind must be one of {BACKENDS}", details={"field": "kind"}
            )
        if not self.version or self.version.lower() in ("latest", "unknown", ""):
            raise ConfigError(
                "the backend version must be exact (no 'latest'/blank)",
                details={"field": "version"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "source_commit": self.source_commit,
            "build_flags": self.build_flags,
            "launcher": self.launcher,
            "binary_sha256": self.binary_sha256,
            "official_tool": self.official_tool,
            "official_tool_version": self.official_tool_version,
        }


@dataclass(frozen=True)
class HarnessIdentity:
    """Freeze the harness/launcher identity (details E10-02 step 2)."""

    harness_version: str
    harness_commit: str
    python_version: str
    launcher_command: str
    generator_version: str
    oracle_version: str
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "harness_version": self.harness_version,
            "harness_commit": self.harness_commit,
            "python_version": self.python_version,
            "launcher_command": self.launcher_command,
            "generator_version": self.generator_version,
            "oracle_version": self.oracle_version,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CapabilityKey:
    """One (op, dtype, backend) capability cell."""

    op: str
    dtype: str
    backend: str

    def __post_init__(self) -> None:
        if not self.op or not self.dtype:
            raise ConfigError("a capability cell needs an op and a dtype")
        if self.backend not in BACKENDS:
            raise ConfigError(
                f"backend must be one of {BACKENDS}", details={"field": "backend"}
            )

    def as_dict(self) -> Dict[str, str]:
        return {"op": self.op, "dtype": self.dtype, "backend": self.backend}


@dataclass(frozen=True)
class CapabilityDecision:
    state: str
    reason_code: str
    probe: str = ""
    allows_execution: bool = False

    def __post_init__(self) -> None:
        if self.state not in CAPABILITY_STATES:
            raise ConfigError(
                f"state must be one of {CAPABILITY_STATES}", details={"field": "state"}
            )
        if self.state == "UNKNOWN" and self.allows_execution:
            raise ConfigError(
                "an UNKNOWN capability must never authorise execution (E10-02 step 9)",
                details={"field": "allows_execution"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "reason_code": self.reason_code,
            "probe": self.probe,
            "allows_execution": self.allows_execution,
        }


@dataclass
class CapabilityMatrix:
    """A probe-result matrix with explicit unknown cells."""

    entries: Dict[Tuple[str, str, str], CapabilityDecision] = field(default_factory=dict)

    def record(
        self,
        op: str,
        dtype: str,
        backend: str,
        *,
        state: str,
        reason_code: str,
        probe: str = "",
    ) -> CapabilityDecision:
        decision = CapabilityDecision(
            state=state,
            reason_code=reason_code,
            probe=probe,
            allows_execution=state == "SUPPORTED",
        )
        self.entries[(op, dtype, backend)] = decision
        return decision

    def decide(self, key: CapabilityKey) -> CapabilityDecision:
        return self.entries.get(
            (key.op, key.dtype, key.backend),
            CapabilityDecision(
                state="UNKNOWN",
                reason_code="capability.not_probed",
                probe="",
                allows_execution=False,
            ),
        )

    def as_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for (op, dtype, backend), decision in sorted(self.entries.items()):
            row = {"op": op, "dtype": dtype, "backend": backend}
            row.update(decision.as_dict())
            rows.append(row)
        return rows

    def unknown_cells(self) -> List[Tuple[str, str, str]]:
        return sorted(
            key
            for key, decision in self.entries.items()
            if decision.state == "UNKNOWN"
        )


@dataclass(frozen=True)
class AlgoProtocolGroup:
    """An auto or forced algorithm/protocol group (details E10-02 step 10)."""

    mode: str
    algorithm: str = "auto"
    protocol: str = "auto"
    op: str = ""
    requires_explicit_support: bool = True
    actual_evidence_required: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ALGORITHM_MODES:
            raise ConfigError(
                f"mode must be one of {ALGORITHM_MODES}", details={"field": "mode"}
            )
        if self.mode == "forced" and self.algorithm == "auto" and self.protocol == "auto":
            raise ConfigError(
                "a forced group must name an algorithm or protocol",
                details={"field": "algorithm"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "algorithm": self.algorithm,
            "protocol": self.protocol,
            "op": self.op,
            "requires_explicit_support": self.requires_explicit_support,
            "actual_evidence_required": self.actual_evidence_required,
            "notes": self.notes,
        }


def algorithm_groups(ops: Sequence[str]) -> List[AlgoProtocolGroup]:
    """The default auto group plus a small forced set (never a full sweep by default)."""
    groups = [AlgoProtocolGroup(mode="auto")]
    for op in ops:
        for algorithm in ("ring", "tree"):
            groups.append(
                AlgoProtocolGroup(
                    mode="forced",
                    algorithm=algorithm,
                    op=op,
                    notes="only when the locked backend documents support; actual must be read back",
                )
            )
    return groups


# ── requested vs actual (handbook §5.7) ────────────────────────────────────


@dataclass(frozen=True)
class RequestedActual:
    """requested / actual / reason triple; a difference without a reason is refused."""

    requested: str
    actual: str
    reason_code: str = ""
    reason: str = ""
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.actual != self.requested and not self.reason:
            raise ConfigError(
                "actual differs from requested without a fallback reason (handbook §5.7)",
                details={"field": "reason"},
            )

    @property
    def degraded(self) -> bool:
        return self.actual != self.requested

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "actual": self.actual,
            "degraded": self.degraded,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "evidence": self.evidence,
        }


# ── timing semantics ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class TimingCalibration:
    """Separate enqueue from completion (details E10-02 step 11)."""

    enqueue_ns: int
    completion_ns: int
    method: str
    async_handle_waited: bool = True
    host_api_return_used: bool = False

    def validate(self) -> None:
        if self.host_api_return_used:
            raise ConfigError(
                "a host API return time is not a collective latency (E10-02 §9)",
                details={"field": "host_api_return_used"},
            )
        if not self.async_handle_waited:
            raise ConfigError(
                "the async work handle must be waited/completed before measuring",
                details={"field": "async_handle_waited"},
            )
        if self.completion_ns < self.enqueue_ns:
            raise ConfigError("completion precedes enqueue", details={"field": "completion_ns"})

    @property
    def completion_latency_ns(self) -> int:
        self.validate()
        return self.completion_ns - self.enqueue_ns

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enqueue_ns": self.enqueue_ns,
            "completion_ns": self.completion_ns,
            "completion_latency_ns": self.completion_latency_ns,
            "method": self.method,
            "async_handle_waited": self.async_handle_waited,
        }


@dataclass(frozen=True)
class ConnectionCostRecord:
    """Init/first-call/steady cost, never mixed into the small-message curve (§步12)."""

    communicator_init_ms: float
    first_call_ms: float
    steady_ms: float
    method: str = ""
    warmup_calls: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "communicator_init_ms": self.communicator_init_ms,
            "first_call_ms": self.first_call_ms,
            "steady_ms": self.steady_ms,
            "method": self.method,
            "warmup_calls": self.warmup_calls,
            "note": "init/first-call are reported separately from the steady-state curve",
        }


# ── error normalisation ────────────────────────────────────────────────────


#: Vendor/backend error code fragments → normalized distributed error classes.
ERROR_CODE_MAP: Mapping[str, str] = {
    "ncclunhandledcudaerror": "DEVICE_ERROR",
    "ncclsystemerror": "NETWORK",
    "ncclinternalerror": "COMM_ASYNC",
    "ncclinvalidargument": "MISMATCH",
    "ncclinvalidusage": "MISMATCH",
    "ncclremoteerror": "NETWORK",
    "ncclinprogress": "TIMEOUT",
    "nccltimeout": "TIMEOUT",
    "hccl_e_invalid_argument": "MISMATCH",
    "hccl_e_timeout": "TIMEOUT",
    "hccl_e_internal": "COMM_ASYNC",
    "hccl_e_network": "NETWORK",
    "timeout": "TIMEOUT",
    "connection reset": "NETWORK",
    "oom": "DEVICE_OOM",
}


def normalize_backend_error(message: str, vendor_code: str = "") -> Dict[str, Any]:
    """Map a vendor error to a normalized class; an unmapped message stays UNKNOWN."""
    haystack = f"{vendor_code} {message}".lower()
    for fragment, normalized in ERROR_CODE_MAP.items():
        if fragment in haystack:
            return {
                "normalized": normalized,
                "vendor_code": vendor_code,
                "matched_fragment": fragment,
                "raw_message": message,
            }
    return {
        "normalized": "UNKNOWN",
        "vendor_code": vendor_code,
        "matched_fragment": "",
        "raw_message": message,
        "reason": "no mapping rule matched; classify explicitly rather than guessing",
    }


__all__ = [
    "ALGORITHM_FAMILIES",
    "ALGORITHM_MODES",
    "AlgoProtocolGroup",
    "BACKENDS",
    "BackendIdentity",
    "CAPABILITY_STATES",
    "CapabilityDecision",
    "CapabilityKey",
    "CapabilityMatrix",
    "ConnectionCostRecord",
    "ERROR_CODE_MAP",
    "HarnessIdentity",
    "PROTOCOLS",
    "RequestedActual",
    "TimingCalibration",
    "algorithm_groups",
    "normalize_backend_error",
]
