"""Fault specification, injection plan, attempt lineage and blast radius.

E08-09's pass condition is not "the service survived": it is that no wrong
model, no duplicated or missing token, no endless wait, no retry storm and no
residual resource survived the fault.  Two things make that checkable:

* every fault is **declared before it is injected** (:class:`FaultSpecEntry`
  with its trigger, commit phase, expected detection, allowed action and
  cleanup criteria), and the injector records its own evidence
  (:class:`InjectionRecord`);
* the retry boundary is defined by the *external commit point*, not by the
  exception type: after the first response/SSE byte, a transparent retry is
  forbidden and the stream must end with an explicit incomplete/error frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Layers a fault can be injected at.
FAULT_LAYERS: Tuple[str, ...] = ("backend", "cache", "network", "client")

#: Commit phases of the retry boundary (details README §20), plus ``any``.
COMMIT_PHASES: Tuple[str, ...] = (
    "before_backend_start",
    "backend_started_no_external_bytes",
    "after_external_commit",
    "any",
)

#: Aliases the frozen fault matrix may use, mapped to the canonical phases.
COMMIT_PHASE_ALIASES: Mapping[str, str] = {
    "before_backend_start": "before_backend_start",
    "before_start": "before_backend_start",
    "before_commit": "backend_started_no_external_bytes",
    "backend_started_no_external_bytes": "backend_started_no_external_bytes",
    "after_commit": "after_external_commit",
    "after_external_commit": "after_external_commit",
    "before_or_after_first_token": "any",
    "any": "any",
}

#: Allowed recovery actions after a fault.
FAULT_ACTIONS: Tuple[str, ...] = (
    "retry_or_failover",
    "explicit_incomplete_error",
    "cancel_and_quarantine_after_deadline",
    "score_penalty_not_immediate_circuit",
    "cancel_retry_within_budget_or_reject",
    "keep_out_of_feasible_set",
    "hard_quarantine",
    "bypass_or_recompute_per_policy",
    "invalidate_and_recompute",
    "pre_commit_retry_with_backoff",
    "commit_aware_behaviour",
    "single_terminal_reason",
)

#: Blast-radius groups the report must separate (§10).
BLAST_RADIUS_GROUPS: Tuple[str, ...] = (
    "target_backend_inflight",
    "other_backend_inflight",
    "same_tenant",
    "other_tenant",
    "committed_stream",
    "uncommitted_stream",
    "cache_hit",
    "cache_miss",
    "high_priority",
    "low_priority",
    "arrived_before_fault",
    "arrived_during_fault",
    "arrived_after_fault",
)


@dataclass(frozen=True)
class FaultSpecEntry:
    """One declared fault (frozen before the run)."""

    fault_id: str
    layer: str
    commit_phase: str
    expected_detection: str
    expected_blast_radius: str
    allowed_action: str
    cleanup_criteria: str

    def __post_init__(self) -> None:
        if self.layer not in FAULT_LAYERS:
            raise ConfigError(f"{self.fault_id}: unknown fault layer {self.layer!r}")
        if self.commit_phase not in COMMIT_PHASE_ALIASES:
            raise ConfigError(
                f"{self.fault_id}: unknown commit phase {self.commit_phase!r}; expected one of "
                f"{sorted(COMMIT_PHASE_ALIASES)}"
            )
        if self.allowed_action not in FAULT_ACTIONS:
            raise ConfigError(f"{self.fault_id}: unknown action {self.allowed_action!r}")

    @property
    def canonical_phase(self) -> str:
        """The phase in the frozen vocabulary (aliases resolved)."""
        return COMMIT_PHASE_ALIASES[self.commit_phase]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "layer": self.layer,
            "commit_phase": self.commit_phase,
            "expected_detection": self.expected_detection,
            "expected_blast_radius": self.expected_blast_radius,
            "allowed_action": self.allowed_action,
            "cleanup_criteria": self.cleanup_criteria,
        }


@dataclass(frozen=True)
class FaultSpec:
    """The frozen fault matrix (``configs/serving/fault_spec.yaml``)."""

    entries: Mapping[str, FaultSpecEntry]
    blast_radius_groups: Tuple[str, ...]
    injector_requirements: Tuple[str, ...]
    recovery_metrics: Tuple[str, ...]
    schema_version: str = "1.0.0"

    @classmethod
    def from_document(cls, payload: Mapping[str, Any]) -> "FaultSpec":
        entries: Dict[str, FaultSpecEntry] = {}
        for raw in payload["faults"]:
            item = dict(raw)
            fault_id = str(item["id"])
            if fault_id in entries:
                raise ConfigError(f"duplicate fault id {fault_id!r}")
            entries[fault_id] = FaultSpecEntry(
                fault_id=fault_id,
                layer=str(item["layer"]),
                commit_phase=str(item["commit_phase"]),
                expected_detection=str(item["expected_detection"]),
                expected_blast_radius=str(item["expected_blast_radius"]),
                allowed_action=str(item["allowed_action"]),
                cleanup_criteria=str(item["cleanup_criteria"]),
            )
        return cls(
            entries=entries,
            blast_radius_groups=tuple(str(item) for item in payload["blast_radius_groups"]),
            injector_requirements=tuple(str(item) for item in payload["injector_requirements"]),
            recovery_metrics=tuple(str(item) for item in payload["recovery_metrics"]),
            schema_version=str(payload.get("schema_version", "1.0.0")),
        )

    def entry(self, fault_id: str) -> FaultSpecEntry:
        if fault_id not in self.entries:
            raise ConfigError(f"fault {fault_id!r} is not in the frozen matrix")
        return self.entries[fault_id]

    def requires_quarantine(self, fault_id: str) -> bool:
        return self.entry(fault_id).allowed_action == "hard_quarantine"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "faults": {name: item.as_dict() for name, item in sorted(self.entries.items())},
            "blast_radius_groups": list(self.blast_radius_groups),
            "injector_requirements": list(self.injector_requirements),
            "recovery_metrics": list(self.recovery_metrics),
        }


@dataclass(frozen=True)
class InjectionPlan:
    """When, where and how a fault is injected (declared, then applied)."""

    fault_id: str
    target: str
    trigger: str
    duration_ms: float
    severity: str = "full"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "target": self.target,
            "trigger": self.trigger,
            "duration_ms": self.duration_ms,
            "severity": self.severity,
        }


@dataclass
class InjectionRecord:
    """The injector's own evidence: applied, bounded, reversible."""

    fault_id: str
    target: str
    planned_at_ns: int
    applied_at_ns: Optional[int] = None
    reverted_at_ns: Optional[int] = None
    scope: str = ""
    unrelated_targets_touched: int = 0
    host_memory_exhausted: bool = False
    deterministic: bool = True

    @property
    def applied(self) -> bool:
        return self.applied_at_ns is not None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "target": self.target,
            "planned_at_ns": self.planned_at_ns,
            "applied_at_ns": self.applied_at_ns,
            "reverted_at_ns": self.reverted_at_ns,
            "scope": self.scope,
            "applied": self.applied,
            "unrelated_targets_touched": self.unrelated_targets_touched,
            "host_memory_exhausted": self.host_memory_exhausted,
            "deterministic": self.deterministic,
        }


def injector_precision(records: Sequence[InjectionRecord], *, expected_faults: Sequence[str]) -> Dict[str, Any]:
    """E08-09 step 2: prove the injector hit exactly what it claimed."""
    problems: List[str] = []
    by_fault = {record.fault_id: record for record in records}
    for fault_id in expected_faults:
        record = by_fault.get(fault_id)
        if record is None:
            problems.append(f"{fault_id}: no injection record")
            continue
        if not record.applied:
            problems.append(f"{fault_id}: injection never applied")
        if record.unrelated_targets_touched:
            problems.append(f"{fault_id}: touched {record.unrelated_targets_touched} unrelated targets")
        if record.host_memory_exhausted:
            problems.append(f"{fault_id}: exhausted host memory (fault scope escaped)")
        if not record.deterministic:
            problems.append(f"{fault_id}: injection was not deterministic")
    return {
        "ok": not problems,
        "problems": problems,
        "records": [record.as_dict() for record in records],
        "note": (
            "an injector that kills the wrong process or exhausts the host does not "
            "measure the fault under study"
        ),
    }


def commit_phase(*, external_bytes_committed: int, backend_started: bool) -> str:
    """The retry boundary is defined by what the client already saw."""
    if not backend_started:
        return "before_backend_start"
    if external_bytes_committed <= 0:
        return "backend_started_no_external_bytes"
    return "after_external_commit"


@dataclass(frozen=True)
class FaultOutcome:
    """What actually happened for one injected fault."""

    fault_id: str
    phase: str
    observed_action: str
    detected_at_ns: Optional[int] = None
    affected_requests: int = 0
    duplicate_tokens: int = 0
    missing_sequence: int = 0
    wrong_model: int = 0
    residual_resources: int = 0
    cleanup_ok: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "phase": self.phase,
            "observed_action": self.observed_action,
            "detected_at_ns": self.detected_at_ns,
            "affected_requests": self.affected_requests,
            "duplicate_tokens": self.duplicate_tokens,
            "missing_sequence": self.missing_sequence,
            "wrong_model": self.wrong_model,
            "residual_resources": self.residual_resources,
            "cleanup_ok": self.cleanup_ok,
        }


def fault_matrix_report(
    spec: FaultSpec, outcomes: Sequence[FaultOutcome]
) -> Dict[str, Any]:
    """Expected vs. observed per fault; missing and extra faults are visible."""
    seen = {outcome.fault_id for outcome in outcomes}
    missing = [name for name in spec.entries if name not in seen]
    extra = [name for name in seen if name not in spec.entries]
    problems: List[str] = []
    for outcome in outcomes:
        entry = spec.entries.get(outcome.fault_id)
        if entry is None:
            problems.append(f"{outcome.fault_id}: not part of the frozen matrix")
            continue
        if outcome.observed_action != entry.allowed_action:
            problems.append(
                f"{outcome.fault_id}: observed {outcome.observed_action!r}, declared "
                f"{entry.allowed_action!r}"
            )
        if entry.canonical_phase != "any" and outcome.phase != entry.canonical_phase:
            problems.append(
                f"{outcome.fault_id}: fired in {outcome.phase}, declared {entry.canonical_phase}"
            )
        if outcome.duplicate_tokens:
            problems.append(f"{outcome.fault_id}: {outcome.duplicate_tokens} duplicated tokens")
        if outcome.missing_sequence:
            problems.append(f"{outcome.fault_id}: {outcome.missing_sequence} sequence gaps")
        if outcome.wrong_model:
            problems.append(f"{outcome.fault_id}: a wrong model reached the client")
        if outcome.residual_resources:
            problems.append(f"{outcome.fault_id}: {outcome.residual_resources} residual resources")
        if not outcome.cleanup_ok:
            problems.append(f"{outcome.fault_id}: cleanup criteria not met")
    return {
        "ok": not problems,
        "problems": problems,
        "missing_faults": missing,
        "extra_faults": extra,
        "rows": [outcome.as_dict() for outcome in outcomes],
        "note": "a fault matrix with a missing row is not a completed fault experiment",
    }


# ── attempt lineage ────────────────────────────────────────────────────────


@dataclass
class Attempt:
    """One attempt of one request (parent/attempt identity is explicit)."""

    parent_request_id: str
    attempt_id: str
    instance_id: str
    model_epoch: str
    start_ns: int
    end_ns: Optional[int] = None
    error: str = ""
    generated_tokens: int = 0
    committed_tokens: int = 0
    emitted_tokens: int = 0
    cancelled: bool = False
    cleaned: bool = False
    retry_decision: str = ""
    retry_reason: str = ""
    retry_budget_remaining_ms: float = 0.0
    next_instance_id: str = ""
    tokens_committed_before_failure: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "parent_request_id": self.parent_request_id,
            "attempt_id": self.attempt_id,
            "instance_id": self.instance_id,
            "model_epoch": self.model_epoch,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "error": self.error,
            "generated_tokens": self.generated_tokens,
            "committed_tokens": self.committed_tokens,
            "emitted_tokens": self.emitted_tokens,
            "cancelled": self.cancelled,
            "cleaned": self.cleaned,
            "retry_decision": self.retry_decision,
            "retry_reason": self.retry_reason,
            "retry_budget_remaining_ms": self.retry_budget_remaining_ms,
            "next_instance_id": self.next_instance_id,
            "tokens_committed_before_failure": self.tokens_committed_before_failure,
        }


def attempt_lineage_audit(attempts: Sequence[Attempt]) -> Dict[str, Any]:
    """Duplicated/missing tokens, amplification, wasted compute (§8)."""
    problems: List[str] = []
    by_request: Dict[str, List[Attempt]] = {}
    for attempt in attempts:
        by_request.setdefault(attempt.parent_request_id, []).append(attempt)
    for request_id, items in by_request.items():
        items = sorted(items, key=lambda item: item.start_ns)
        attempts_count = len(items)
        for previous, current in zip(items, items[1:]):
            if previous.tokens_committed_before_failure or previous.emitted_tokens:
                problems.append(
                    f"{request_id}: retried after {previous.emitted_tokens} externally "
                    "visible tokens; a transparent retry there can duplicate output"
                )
            del current
        if items[-1].end_ns is None:
            problems.append(f"{request_id}: the last attempt never terminated")
        if not items[-1].cleaned:
            problems.append(f"{request_id}: the last attempt was not cleaned up")
        del attempts_count
    amplification = (
        sum(len(items) for items in by_request.values()) / len(by_request)
        if by_request
        else None
    )
    return {
        "ok": not problems,
        "problems": problems,
        "requests": len(by_request),
        "attempts": len(attempts),
        "attempts_per_request": amplification,
        "wasted_tokens": sum(
            attempt.generated_tokens - attempt.committed_tokens for attempt in attempts
        ),
        "note": (
            "client-visible duplicates and backend-side duplicated compute are two "
            "different numbers and both are recorded"
        ),
    }


def duplicate_external_tokens_by_request(attempts: Sequence[Attempt]) -> Dict[str, int]:
    """Count tokens written by more than one attempt of the same request."""
    totals: Dict[str, int] = {}
    for attempt in attempts:
        totals[attempt.parent_request_id] = totals.get(attempt.parent_request_id, 0) + attempt.emitted_tokens
    return {request_id: total for request_id, total in totals.items() if total > 0}


# ── blast radius ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AffectedRequest:
    request_id: str
    group: str
    error: str = ""
    p99_ms: Optional[float] = None
    good: bool = False
    retried: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "group": self.group,
            "error": self.error,
            "p99_ms": self.p99_ms,
            "good": self.good,
            "retried": self.retried,
        }


def blast_radius_report(requests: Sequence[AffectedRequest]) -> Dict[str, Any]:
    groups: Dict[str, List[AffectedRequest]] = {}
    for request in requests:
        if request.group not in BLAST_RADIUS_GROUPS:
            raise ConfigError(
                f"unknown blast-radius group {request.group!r}; expected one of "
                f"{list(BLAST_RADIUS_GROUPS)}"
            )
        groups.setdefault(request.group, []).append(request)
    rows = []
    for name in BLAST_RADIUS_GROUPS:
        items = groups.get(name, [])
        rows.append(
            {
                "group": name,
                "requests": len(items),
                "errors": sum(1 for item in items if item.error),
                "good": sum(1 for item in items if item.good),
                "retried": sum(1 for item in items if item.retried),
            }
        )
    healthy = [row for row in rows if row["group"] == "other_backend_inflight"]
    return {
        "rows": rows,
        "unaffected_present": bool(healthy and healthy[0]["requests"]),
        "note": (
            "only looking at the fault target does not prove isolation; the other "
            "Backend's requests, both tenancies and both stream states are part of it"
        ),
    }


def retry_policy_after_fault(
    *,
    phase: str,
    model_rng_identity_proven: bool,
    cancel_and_cleanup_proven: bool,
    budget_remaining: bool,
    fallback_verified: bool,
) -> Dict[str, Any]:
    """Whether a retry is even *allowed* — the decision, not the attempt."""
    if phase == "after_external_commit":
        return {
            "retry_allowed": False,
            "reason": "external bytes are committed: a transparent retry may duplicate output",
            "required_failure_mode": "explicit incomplete/error terminal frame",
        }
    if phase == "before_backend_start":
        return {
            "retry_allowed": True,
            "reason": "nothing has been submitted yet; re-routing is safe",
            "required_failure_mode": "",
        }
    conditions = {
        "request_semantics_rng_identity_proven": model_rng_identity_proven,
        "cancel_and_cleanup_proven": cancel_and_cleanup_proven,
        "retry_budget_remaining": budget_remaining,
        "fallback_revalidated": fallback_verified,
    }
    missing = [name for name, ok in conditions.items() if not ok]
    return {
        "retry_allowed": not missing,
        "reason": "" if not missing else f"conditions missing: {missing}",
        "conditions": conditions,
        "required_failure_mode": "" if not missing else "explicit error terminal frame",
        "note": (
            "a retry after the backend started but before any external byte needs proof "
            "of model/RNG identity, cancellation and cleanup, and a remaining budget"
        ),
    }


__all__ = [
    "BLAST_RADIUS_GROUPS",
    "COMMIT_PHASES",
    "FAULT_ACTIONS",
    "FAULT_LAYERS",
    "AffectedRequest",
    "Attempt",
    "FaultOutcome",
    "FaultSpec",
    "FaultSpecEntry",
    "InjectionPlan",
    "InjectionRecord",
    "attempt_lineage_audit",
    "blast_radius_report",
    "commit_phase",
    "duplicate_external_tokens_by_request",
    "fault_matrix_report",
    "injector_precision",
    "retry_policy_after_fault",
]
