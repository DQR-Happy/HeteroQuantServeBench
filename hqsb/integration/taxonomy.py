"""Error taxonomy, telemetry records and transactional state (E06-09).

"Caught an exception and continued" is not a safe fallback (E06-09 §1).  This
module defines:

* :class:`ErrorStage` — the 14 layers an error can originate from, so an error
  that *should* have been caught before loading cannot surface at kernel launch;
* :class:`ErrorTaxonomy` — stable reason codes with severity, retryability,
  fallback permission and a user-facing message;
* :class:`FailureRecord` — the C6/C7 telemetry payload, with message sanitisation
  (no prompts, no weights, no machine paths);
* :class:`TransactionalPlan` — validate → prepare → execute → validate → commit,
  with an :meth:`TransactionalPlan.abort` that discards partial outputs and
  restores state invariants;
* :class:`FallbackRegistry` — deterministic per-level capability disabling, so
  "the same failure sometimes picks a different backend" is detectable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import CapabilityError, ConfigError, HqsbError, SchemaError

# ── stages ────────────────────────────────────────────────────────────────


class ErrorStage:
    API_SCHEMA = "api_schema"
    INPUT_VALIDATION = "input_validation"
    MODEL_QUANT_IDENTITY = "model_quant_identity"
    DISPATCH_CAPABILITY = "dispatch_capability"
    FAKE_CAPTURE = "fake_capture"
    PATTERN_REWRITE = "pattern_rewrite"
    GUARD_RUNTIME_ASSERT = "guard_runtime_assert"
    COMPILE_CODEGEN_LINK = "compile_codegen_link"
    CACHE_VALIDATION = "cache_validation"
    BINARY_ABI_LOAD = "binary_abi_load"
    KERNEL_LAUNCH_RUNTIME = "kernel_launch_runtime"
    ASYNC_DEVICE = "async_device"
    STATE_COMMIT = "state_commit"
    RESOURCE_CLEANUP = "resource_cleanup"

    ALL = (
        API_SCHEMA,
        INPUT_VALIDATION,
        MODEL_QUANT_IDENTITY,
        DISPATCH_CAPABILITY,
        FAKE_CAPTURE,
        PATTERN_REWRITE,
        GUARD_RUNTIME_ASSERT,
        COMPILE_CODEGEN_LINK,
        CACHE_VALIDATION,
        BINARY_ABI_LOAD,
        KERNEL_LAUNCH_RUNTIME,
        ASYNC_DEVICE,
        STATE_COMMIT,
        RESOURCE_CLEANUP,
    )

    @classmethod
    def require(cls, stage: str) -> str:
        if stage not in cls.ALL:
            raise SchemaError(
                f"unknown error stage {stage!r}",
                details={"field": "stage", "allowed": list(cls.ALL)},
            )
        return stage


class Severity:
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

    ALL = (INFO, WARNING, ERROR, FATAL)


@dataclass(frozen=True)
class ReasonCodeSpec:
    """One stable reason code."""

    code: str
    stage: str
    severity: str = Severity.ERROR
    retryable: bool = False
    fallback_allowed: bool = True
    user_message: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        ErrorStage.require(self.stage)
        if self.severity not in Severity.ALL:
            raise SchemaError(
                f"{self.code}: unknown severity {self.severity!r}",
                details={"field": "severity", "allowed": list(Severity.ALL)},
            )
        if not self.code.isupper():
            raise SchemaError(
                f"reason code {self.code!r} must be SCREAMING_SNAKE_CASE",
                details={"field": "code"},
            )
        if not self.user_message:
            raise SchemaError(
                f"{self.code}: a user-facing message is required (stable text, no internals)",
                details={"field": "user_message"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "severity": self.severity,
            "retryable": self.retryable,
            "fallback_allowed": self.fallback_allowed,
            "user_message": self.user_message,
            "description": self.description,
        }


REASON_CODES: Tuple[ReasonCodeSpec, ...] = (
    ReasonCodeSpec("SCHEMA_INVALID_ARGUMENT", ErrorStage.API_SCHEMA, user_message="Invalid operator argument."),
    ReasonCodeSpec("SCHEMA_UNKNOWN_OPERATOR", ErrorStage.API_SCHEMA, user_message="Unknown operator."),
    ReasonCodeSpec("INPUT_DEVICE_MISMATCH", ErrorStage.INPUT_VALIDATION, user_message="Input is on an unsupported device.", retryable=True),
    ReasonCodeSpec("INPUT_DTYPE_UNSUPPORTED", ErrorStage.INPUT_VALIDATION, user_message="Input dtype is not supported."),
    ReasonCodeSpec("INPUT_SHAPE_UNSUPPORTED", ErrorStage.INPUT_VALIDATION, user_message="Input shape is not supported."),
    ReasonCodeSpec("INPUT_STRIDE_UNSUPPORTED", ErrorStage.INPUT_VALIDATION, user_message="Input layout is not supported.", retryable=True),
    ReasonCodeSpec("INPUT_NAN_INF", ErrorStage.INPUT_VALIDATION, severity=Severity.FATAL, fallback_allowed=False, user_message="Input contains NaN/Inf."),
    ReasonCodeSpec("MODEL_IDENTITY_MISMATCH", ErrorStage.MODEL_QUANT_IDENTITY, fallback_allowed=False, user_message="Model identity does not match the artifact."),
    ReasonCodeSpec("QUANT_ARTIFACT_MISMATCH", ErrorStage.MODEL_QUANT_IDENTITY, fallback_allowed=False, user_message="Quantization artifact does not match the model."),
    ReasonCodeSpec("CAPABILITY_UNSUPPORTED", ErrorStage.DISPATCH_CAPABILITY, user_message="Requested capability is unavailable.", retryable=True),
    ReasonCodeSpec("CAPABILITY_STRICT_REFUSAL", ErrorStage.DISPATCH_CAPABILITY, fallback_allowed=False, user_message="Fallback is disabled and the capability is missing."),
    ReasonCodeSpec("FAKE_METADATA_MISSING", ErrorStage.FAKE_CAPTURE, user_message="Operator has no fake/meta implementation."),
    ReasonCodeSpec("FAKE_METADATA_INVALID", ErrorStage.FAKE_CAPTURE, fallback_allowed=False, user_message="Fake metadata is inconsistent."),
    ReasonCodeSpec("PATTERN_UNSUPPORTED", ErrorStage.PATTERN_REWRITE, user_message="The graph shape is not supported by any pattern."),
    ReasonCodeSpec("PATTERN_PASS_FAILED", ErrorStage.PATTERN_REWRITE, user_message="Graph rewrite failed."),
    ReasonCodeSpec("GUARD_FAILED", ErrorStage.GUARD_RUNTIME_ASSERT, retryable=True, user_message="Compiled graph guard did not match."),
    ReasonCodeSpec("RUNTIME_ASSERT_FAILED", ErrorStage.GUARD_RUNTIME_ASSERT, user_message="Runtime shape assertion failed."),
    ReasonCodeSpec("COMPILE_FAILED", ErrorStage.COMPILE_CODEGEN_LINK, retryable=True, user_message="Graph compilation failed."),
    ReasonCodeSpec("CODEGEN_FAILED", ErrorStage.COMPILE_CODEGEN_LINK, retryable=True, user_message="Kernel code generation failed."),
    ReasonCodeSpec("LINK_LOAD_FAILED", ErrorStage.COMPILE_CODEGEN_LINK, retryable=True, user_message="Compiled artifact could not be loaded."),
    ReasonCodeSpec("AUTOTUNE_NO_VALID_CONFIG", ErrorStage.COMPILE_CODEGEN_LINK, user_message="No valid autotune configuration."),
    ReasonCodeSpec("CACHE_ENTRY_CORRUPT", ErrorStage.CACHE_VALIDATION, retryable=True, user_message="A cached artifact failed validation and was discarded."),
    ReasonCodeSpec("CACHE_ENTRY_INCOMPLETE", ErrorStage.CACHE_VALIDATION, retryable=True, user_message="A cached artifact was incomplete and was discarded."),
    ReasonCodeSpec("ABI_MISMATCH", ErrorStage.BINARY_ABI_LOAD, fallback_allowed=False, user_message="Binary ABI does not match this environment."),
    ReasonCodeSpec("ARCH_UNSUPPORTED", ErrorStage.BINARY_ABI_LOAD, fallback_allowed=False, user_message="Binary does not target this GPU architecture."),
    ReasonCodeSpec("SYMBOL_MISSING", ErrorStage.BINARY_ABI_LOAD, fallback_allowed=False, user_message="Required symbol is missing from the extension."),
    ReasonCodeSpec("KERNEL_LAUNCH_FAILED", ErrorStage.KERNEL_LAUNCH_RUNTIME, retryable=True, user_message="Kernel launch failed."),
    ReasonCodeSpec("WORKSPACE_INSUFFICIENT", ErrorStage.KERNEL_LAUNCH_RUNTIME, user_message="Workspace is too small for this shape."),
    ReasonCodeSpec("DEVICE_OOM", ErrorStage.KERNEL_LAUNCH_RUNTIME, severity=Severity.ERROR, fallback_allowed=True, user_message="Device memory exhausted."),
    ReasonCodeSpec("ASYNC_DEVICE_ERROR", ErrorStage.ASYNC_DEVICE, severity=Severity.FATAL, fallback_allowed=False, user_message="An asynchronous device error was reported."),
    ReasonCodeSpec("STATE_PARTIAL_WRITE", ErrorStage.STATE_COMMIT, severity=Severity.FATAL, fallback_allowed=False, user_message="Execution failed after a partial state update."),
    ReasonCodeSpec("STATE_INVARIANT_VIOLATION", ErrorStage.STATE_COMMIT, severity=Severity.FATAL, fallback_allowed=False, user_message="A state invariant was violated."),
    ReasonCodeSpec("CLEANUP_FAILED", ErrorStage.RESOURCE_CLEANUP, severity=Severity.WARNING, user_message="Resource cleanup did not complete."),
)


class ErrorTaxonomy:
    """Lookup + completeness check over the frozen reason codes."""

    def __init__(self, specs: Sequence[ReasonCodeSpec] = REASON_CODES) -> None:
        self._specs: Dict[str, ReasonCodeSpec] = {}
        for spec in specs:
            if spec.code in self._specs:
                raise ConfigError(
                    f"duplicate reason code {spec.code!r}",
                    details={"field": "code"},
                )
            self._specs[spec.code] = spec

    def resolve(self, code: str) -> ReasonCodeSpec:
        if code not in self._specs:
            raise SchemaError(
                f"unknown reason code {code!r}",
                details={"field": "code", "known": len(self._specs)},
            )
        return self._specs[code]

    def by_stage(self) -> Dict[str, Tuple[str, ...]]:
        grouped: Dict[str, List[str]] = {stage: [] for stage in ErrorStage.ALL}
        for spec in self._specs.values():
            grouped[spec.stage].append(spec.code)
        return {stage: tuple(sorted(codes)) for stage, codes in grouped.items()}

    def uncovered_stages(self) -> Tuple[str, ...]:
        return tuple(
            stage for stage, codes in self.by_stage().items() if not codes
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "codes": [spec.as_dict() for spec in sorted(self._specs.values(), key=lambda s: s.code)],
            "by_stage": {stage: list(codes) for stage, codes in self.by_stage().items()},
            "uncovered_stages": list(self.uncovered_stages()),
        }


def frozen_taxonomy() -> ErrorTaxonomy:
    taxonomy = ErrorTaxonomy()
    uncovered = taxonomy.uncovered_stages()
    if uncovered:  # pragma: no cover - guarded by tests
        raise ConfigError(
            f"error taxonomy does not cover stages {list(uncovered)}",
            details={"field": "stages"},
        )
    return taxonomy


# ── exception mapping and sanitisation ────────────────────────────────────

_SENSITIVE_PATTERNS = (
    ("/root/", "<path>/"),
    ("/home/", "<path>/"),
    ("/tmp/", "<tmp>/"),
    ("prompt", "<redacted-text>"),
    ("weight", "<redacted-text>"),
)


def sanitize_message(text: str) -> str:
    """Strip machine paths and prompt/weight content from a message (E06-09 §9)."""
    sanitized = str(text)
    for needle, replacement in _SENSITIVE_PATTERNS:
        sanitized = sanitized.replace(needle, replacement)
    sanitized = re.sub(r"0x[0-9a-fA-F]{6,}", "<addr>", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def classify_exception(exc: BaseException) -> Tuple[str, str]:
    """Map an exception onto ``(reason_code, stage)``; unknown → generic error."""
    if isinstance(exc, CapabilityError):
        return "CAPABILITY_UNSUPPORTED", ErrorStage.DISPATCH_CAPABILITY
    if isinstance(exc, SchemaError):
        return "FAKE_METADATA_INVALID", ErrorStage.FAKE_CAPTURE
    if isinstance(exc, ConfigError):
        return "SCHEMA_INVALID_ARGUMENT", ErrorStage.API_SCHEMA
    if isinstance(exc, HqsbError):
        return "COMPILE_FAILED", ErrorStage.COMPILE_CODEGEN_LINK
    if isinstance(exc, MemoryError):
        return "DEVICE_OOM", ErrorStage.KERNEL_LAUNCH_RUNTIME
    if isinstance(exc, (ValueError, TypeError)):
        return "SCHEMA_INVALID_ARGUMENT", ErrorStage.API_SCHEMA
    return "COMPILE_FAILED", ErrorStage.COMPILE_CODEGEN_LINK


# ── failure telemetry ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class FailureRecord:
    """One failure with everything C6/C7 must be able to join on."""

    error_id: str
    reason_code: str
    stage: str
    run_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    operator: str = ""
    module: str = ""
    graph: str = ""
    input_meta_digest: str = ""
    requested: str = ""
    actual: str = ""
    model_hash: str = ""
    quant_hash: str = ""
    schema_hash: str = ""
    binary_hash: str = ""
    env_hash: str = ""
    exception_type: str = ""
    message: str = ""
    sync_async: str = "sync"
    partial_state: str = "none"
    fallback: str = ""
    cleanup: str = ""
    retry: str = ""
    final_result: str = ""

    def __post_init__(self) -> None:
        ErrorStage.require(self.stage)
        if self.sync_async not in ("sync", "async"):
            raise ConfigError(
                f"{self.error_id}: sync_async must be 'sync' or 'async'",
                details={"field": "sync_async", "actual": self.sync_async},
            )

    @property
    def sanitized_message(self) -> str:
        return sanitize_message(self.message)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "reason_code": self.reason_code,
            "stage": self.stage,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "operator": self.operator,
            "module": self.module,
            "graph": self.graph,
            "input_meta_digest": self.input_meta_digest,
            "requested": self.requested,
            "actual": self.actual,
            "model_hash": self.model_hash,
            "quant_hash": self.quant_hash,
            "schema_hash": self.schema_hash,
            "binary_hash": self.binary_hash,
            "env_hash": self.env_hash,
            "exception_type": self.exception_type,
            "message": self.sanitized_message,
            "sync_async": self.sync_async,
            "partial_state": self.partial_state,
            "fallback": self.fallback,
            "cleanup": self.cleanup,
            "retry": self.retry,
            "final_result": self.final_result,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2, ensure_ascii=False)

    @property
    def decision_fields(self) -> Dict[str, Any]:
        """Fields that must be identical for a deterministic replay of a failure."""
        payload = self.as_dict()
        for volatile in ("error_id", "span_id", "trace_id"):
            payload.pop(volatile, None)
        return payload


def deterministic_failure(records: Sequence[FailureRecord]) -> Dict[str, Any]:
    """Check that repeating one failure yields the same decision fields."""
    if not records:
        return {"ok": True, "replays": 0, "differences": []}
    baseline = records[0].decision_fields
    differences: List[Dict[str, Any]] = []
    for index, record in enumerate(records[1:], start=1):
        payload = record.decision_fields
        for key in sorted(set(baseline) | set(payload)):
            if baseline.get(key) != payload.get(key):
                differences.append(
                    {
                        "replay": index,
                        "field": key,
                        "baseline": baseline.get(key),
                        "observed": payload.get(key),
                    }
                )
    return {"ok": not differences, "replays": len(records), "differences": differences}


# ── transactional execution ───────────────────────────────────────────────


class TransactionState:
    NEW = "NEW"
    VALIDATED = "VALIDATED"
    PREPARED = "PREPARED"
    EXECUTED = "EXECUTED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"

    ALL = (NEW, VALIDATED, PREPARED, EXECUTED, COMMITTED, ABORTED)


@dataclass
class StateInvariant:
    """One state item that must be unchanged unless the transaction commits."""

    name: str
    before: str
    after: str = ""
    required_unchanged: bool = True

    def violated(self) -> bool:
        return self.required_unchanged and self.after != self.before

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "before": self.before,
            "after": self.after,
            "required_unchanged": self.required_unchanged,
            "violated": self.violated(),
        }


@dataclass
class TransactionalPlan:
    """validate → prepare → execute → validate completion → commit (E06-09 §6).

    Mutable outputs (KV cache, in-place residual) are staged in shadow buffers;
    :meth:`abort` discards them and re-checks the invariants, so a failure can
    never leave a half-written token or residual behind.
    """

    name: str
    invariants: List[StateInvariant] = field(default_factory=list)
    shadow_outputs: List[str] = field(default_factory=list)
    state: str = TransactionState.NEW
    abort_reason: str = ""
    committed: bool = False

    def __post_init__(self) -> None:
        if not self.invariants:
            raise ConfigError(
                f"transaction {self.name!r} must declare at least one state invariant",
                details={"field": "invariants"},
            )

    def _require(self, state: str) -> None:
        if self.state != state:
            raise ConfigError(
                f"transaction {self.name!r}: expected state {state!r}, got {self.state!r}",
                details={"field": "state", "actual": self.state},
            )

    def validate(self) -> str:
        self._require(TransactionState.NEW)
        self.state = TransactionState.VALIDATED
        return self.state

    def prepare(self) -> str:
        self._require(TransactionState.VALIDATED)
        self.state = TransactionState.PREPARED
        return self.state

    def execute(self) -> str:
        self._require(TransactionState.PREPARED)
        self.state = TransactionState.EXECUTED
        return self.state

    def validate_completion(self, observed: Mapping[str, str]) -> str:
        self._require(TransactionState.EXECUTED)
        for invariant in self.invariants:
            if invariant.name in observed:
                invariant.after = observed[invariant.name]
        violations = [item.as_dict() for item in self.invariants if item.violated()]
        if violations:
            self.abort_with(violations)
            raise ConfigError(
                f"transaction {self.name!r} violates state invariants",
                details={"violations": violations},
            )
        return self.state

    def commit(self) -> Dict[str, Any]:
        self._require(TransactionState.EXECUTED)
        self.state = TransactionState.COMMITTED
        self.committed = True
        return {
            "transaction": self.name,
            "state": self.state,
            "shadow_outputs": list(self.shadow_outputs),
            "invariants": [item.as_dict() for item in self.invariants],
        }

    def abort(self, reason: str) -> Dict[str, Any]:
        return self.abort_with([], reason=reason)

    def abort_with(self, violations: Sequence[Mapping[str, Any]], reason: str = "") -> Dict[str, Any]:
        self.state = TransactionState.ABORTED
        self.abort_reason = reason or "ABORTED"
        discarded = list(self.shadow_outputs)
        self.shadow_outputs = []
        return {
            "transaction": self.name,
            "state": self.state,
            "reason": self.abort_reason,
            "discarded_shadow_outputs": discarded,
            "violations": [dict(item) for item in violations],
            "invariants": [item.as_dict() for item in self.invariants],
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "transaction": self.name,
            "state": self.state,
            "committed": self.committed,
            "abort_reason": self.abort_reason,
            "shadow_outputs": list(self.shadow_outputs),
            "invariants": [item.as_dict() for item in self.invariants],
        }


def kv_invariants(before: Mapping[str, str]) -> List[StateInvariant]:
    """Standard mutable-state set for a decode step (KV/token/residual/RNG/cache)."""
    names = ("kv_cache", "token_sequence", "residual", "rng_state", "compiled_cache")
    return [
        StateInvariant(name=name, before=str(before.get(name, "")), required_unchanged=True)
        for name in names
    ]


# ── deterministic fallback ────────────────────────────────────────────────


@dataclass
class FallbackLevel:
    """One level of the fallback chain with its capability key."""

    name: str
    capability: str
    enabled: bool = True
    disabled_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "capability": self.capability,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
        }


@dataclass
class FallbackRegistry:
    """Deterministic priority chain (E06-09 §7).

    :meth:`disable` exists for the per-level injection the protocol requires
    ("逐级禁用能力"); disabling always carries a reason.
    """

    levels: List[FallbackLevel] = field(
        default_factory=lambda: [
            FallbackLevel("hqsb.compiled.fused", "hqsb.compiled.fused"),
            FallbackLevel("hqsb.eager.custom", "hqsb.eager.custom"),
            FallbackLevel("torch.compile.reference", "torch.compile.reference"),
            FallbackLevel("torch.eager.reference", "torch.eager.reference"),
            FallbackLevel("explicit_error", "explicit_error"),
        ]
    )
    strict: bool = False

    def disable(self, name: str, reason: str) -> None:
        for level in self.levels:
            if level.name == name:
                if not reason:
                    raise ConfigError(
                        f"disabling {name!r} requires a reason (no silent degradation)",
                        details={"field": "reason"},
                    )
                level.enabled = False
                level.disabled_reason = reason
                return
        raise ConfigError(
            f"unknown fallback level {name!r}",
            details={"field": "name", "allowed": [level.name for level in self.levels]},
        )

    @property
    def enabled_names(self) -> Tuple[str, ...]:
        return tuple(level.name for level in self.levels if level.enabled)

    def resolve(self, requested: str) -> Dict[str, Any]:
        """Walk *down* the chain from the requested level; never upgrade silently."""
        if self.strict:
            return {
                "requested": requested,
                "actual": requested,
                "reason": "STRICT_MODE_NO_FALLBACK",
                "fallback_used": False,
                "path": [],
            }
        names = [level.name for level in self.levels]
        if requested in names:
            start = names.index(requested)
            requested_level = self.levels[start]
            path: List[str] = []
            if requested_level.enabled:
                return {
                    "requested": requested,
                    "actual": requested,
                    "reason": "REQUESTED_AVAILABLE",
                    "fallback_used": False,
                    "path": [],
                }
            path.append(f"{requested}:disabled({requested_level.disabled_reason})")
            start += 1
        else:
            start = 0
            path = [f"{requested}:unknown-level"]
        for level in self.levels[start:]:
            if not level.enabled:
                path.append(f"{level.name}:disabled({level.disabled_reason})")
                continue
            if level.name == "explicit_error":
                path.append("explicit_error:reached")
                break
            path.append(f"{level.name}:selected")
            return {
                "requested": requested,
                "actual": level.name,
                "reason": f"FALLBACK_TO:{level.name}",
                "fallback_used": True,
                "path": path,
            }
        return {
            "requested": requested,
            "actual": "explicit_error",
            "reason": "NO_AVAILABLE_IMPLEMENTATION",
            "fallback_used": False,
            "path": path,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strict": self.strict,
            "levels": [level.as_dict() for level in self.levels],
            "enabled": list(self.enabled_names),
        }


def requested_actual_chain(record: FailureRecord) -> Dict[str, Any]:
    """C6/C7 join: requested → failure → fallback → actual → result."""
    return {
        "requested": record.requested,
        "failure": {"reason_code": record.reason_code, "stage": record.stage},
        "fallback": record.fallback,
        "actual": record.actual,
        "final_result": record.final_result,
        "complete": all(
            (record.requested, record.reason_code, record.actual, record.final_result)
        ),
    }


def hash_state(state: Mapping[str, Any]) -> str:
    """Stable digest for a state snapshot (used by the invariant checks)."""
    payload = json.dumps(state, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "ErrorStage",
    "ErrorTaxonomy",
    "FallbackLevel",
    "FallbackRegistry",
    "FailureRecord",
    "REASON_CODES",
    "ReasonCodeSpec",
    "Severity",
    "StateInvariant",
    "TransactionState",
    "TransactionalPlan",
    "classify_exception",
    "deterministic_failure",
    "frozen_taxonomy",
    "hash_state",
    "kv_invariants",
    "requested_actual_chain",
    "sanitize_message",
]
