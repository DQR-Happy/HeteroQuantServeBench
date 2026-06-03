"""C6/C7 projection for graph/compile facts (E06 details README §14).

E06 requires C6 to carry compile mode, graph/compile identity, graph and break
counts, cache layer/hit, compile phase times, requested/actual lowering, the
observed kernel, the fallback reason, correctness status and raw artifact URIs;
and C7 to carry capture, graph break, guard evaluation/failure, recompile,
pattern candidate/hit/reject, lowering selection, cache lookup/hit/miss/
invalidate, kernel spans, fallback, resource create/destroy and the error stage,
with monotonic timestamps so one request can be followed end to end.

Rather than mutate the frozen C1–C7 schemas (S01 contracts, referenced by
migration tests), this module projects the S06 payload onto their documented
extension points — ``BenchmarkResult.summary`` / ``artifact_links`` /
``CorrectnessReport.details`` and ``TraceEvent.attributes`` — and publishes a
coverage audit (:func:`c6_field_coverage`, :func:`c7_kind_coverage`) proving
every required field lands somewhere addressable.  :func:`project_c6` validates
through the real pydantic model, so a projection that no longer fits the frozen
C6 fails immediately.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.contracts.result import (
    BenchmarkResult,
    CorrectnessReport,
    EnvironmentInfo,
)
from hqsb.core.contracts.trace import TraceEvent, TraceEventType
from hqsb.core.errors import ConfigError

C6_NAMESPACE = "s06"

#: Required S06 C6 fields → the extension path they are stored at.
S06_C6_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("compile_mode", f"summary.{C6_NAMESPACE}.compile_mode"),
    ("graph_identity", f"summary.{C6_NAMESPACE}.graph_identity"),
    ("compile_identity", f"summary.{C6_NAMESPACE}.compile_identity"),
    ("graph_count", f"summary.{C6_NAMESPACE}.graph_count"),
    ("break_count", f"summary.{C6_NAMESPACE}.break_count"),
    ("cache_layer", f"summary.{C6_NAMESPACE}.cache.layer"),
    ("cache_hit", f"summary.{C6_NAMESPACE}.cache.hit"),
    ("compile_phase_times", f"summary.{C6_NAMESPACE}.compile_phase_ms"),
    ("requested_lowering", f"summary.{C6_NAMESPACE}.requested_lowering"),
    ("actual_lowering", f"summary.{C6_NAMESPACE}.actual_lowering"),
    ("observed_kernel", f"summary.{C6_NAMESPACE}.observed_kernel"),
    ("fallback_reason", f"summary.{C6_NAMESPACE}.fallback_reason"),
    ("correctness_status", "correctness.details.compile_correctness_status"),
    ("raw_artifacts", "artifact_links"),
)

#: C7 attributes carrying the S06 event kind on top of the frozen event type.
C7_KIND_ATTRIBUTE = "hqsb_kind"
C7_RUN_ATTRIBUTE = "hqsb_run_id"

#: S06 event kind → frozen ``TraceEventType`` (coarse category), with the
#: rationale recorded so the mapping is auditable rather than arbitrary.
KIND_TO_EVENT_TYPE: Mapping[str, Tuple[str, str]] = {
    "capture": (TraceEventType.MODEL_LOAD.value, "graph capture is part of engine setup"),
    "graph_break": (TraceEventType.PREFILL.value, "occurs inside a model forward"),
    "guard_eval": (TraceEventType.KERNEL.value, "host-side dispatch-time work"),
    "guard_fail": (TraceEventType.KERNEL.value, "host-side dispatch-time work"),
    "recompile": (TraceEventType.MODEL_LOAD.value, "rebuilds a compiled artifact"),
    "runtime_assert": (TraceEventType.KERNEL.value, "compiled-graph internal check"),
    "pattern_candidate": (TraceEventType.MODEL_LOAD.value, "graph rewrite stage"),
    "pattern_hit": (TraceEventType.MODEL_LOAD.value, "graph rewrite stage"),
    "pattern_reject": (TraceEventType.MODEL_LOAD.value, "graph rewrite stage"),
    "lowering_select": (TraceEventType.MODEL_LOAD.value, "lowering stage"),
    "cache_lookup": (TraceEventType.CACHE.value, "cache layer event"),
    "cache_hit": (TraceEventType.CACHE.value, "cache layer event"),
    "cache_miss": (TraceEventType.CACHE.value, "cache layer event"),
    "cache_invalidate": (TraceEventType.CACHE.value, "cache layer event"),
    "kernel": (TraceEventType.KERNEL.value, "device kernel span"),
    "fallback": (TraceEventType.KERNEL.value, "execution path decision at dispatch time"),
    "resource_create": (TraceEventType.MODEL_LOAD.value, "resource lifecycle"),
    "resource_destroy": (TraceEventType.MODEL_LOAD.value, "resource lifecycle"),
    "error": (TraceEventType.OUTPUT.value, "failure surfaced to the request"),
}


def _walk(payload: Mapping[str, Any], pointer: str) -> Tuple[bool, Any]:
    current: Any = payload
    for part in pointer.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        return False, None
    return True, current


@dataclass(frozen=True)
class S06ResultFields:
    """The S06 payload of one run, before it is projected onto C6."""

    compile_mode: str = "eager"
    graph_identity: str = ""
    compile_identity: str = ""
    graph_count: int = 0
    break_count: int = 0
    cache_layer: str = ""
    cache_hit: bool = False
    compile_phase_ms: Mapping[str, float] = field(default_factory=dict)
    requested_lowering: str = ""
    actual_lowering: str = ""
    observed_kernel: str = ""
    fallback_reason: str = ""
    correctness_status: str = "not_run"
    raw_artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.correctness_status not in ("not_run", "pass", "fail", "partial"):
            raise ConfigError(
                f"unknown correctness_status {self.correctness_status!r}",
                details={"field": "correctness_status"},
            )
        if self.graph_count < 0 or self.break_count < 0:
            raise ConfigError(
                "graph_count/break_count must be non-negative",
                details={"field": "graph_count"},
            )
        # Fallback without a reason is exactly the silent degradation the
        # project forbids (AGENTS.md / control plane §30).
        if self.actual_lowering and self.requested_lowering and self.actual_lowering != self.requested_lowering:
            if not self.fallback_reason:
                raise ConfigError(
                    "actual lowering differs from the requested one without a "
                    "fallback_reason; silent degradation is not allowed",
                    details={"field": "fallback_reason"},
                )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "compile_mode": self.compile_mode,
            "graph_identity": self.graph_identity,
            "compile_identity": self.compile_identity,
            "graph_count": self.graph_count,
            "break_count": self.break_count,
            "cache_layer": self.cache_layer,
            "cache_hit": self.cache_hit,
            "compile_phase_ms": dict(self.compile_phase_ms),
            "requested_lowering": self.requested_lowering,
            "actual_lowering": self.actual_lowering,
            "observed_kernel": self.observed_kernel,
            "fallback_reason": self.fallback_reason,
            "correctness_status": self.correctness_status,
            "raw_artifacts": dict(self.raw_artifacts),
        }


def project_c6(
    run_id: str,
    fields: S06ResultFields,
    *,
    environment: Optional[EnvironmentInfo] = None,
    git_commit: Optional[str] = None,
    git_dirty: Optional[bool] = None,
    model_artifact_hash: Optional[str] = None,
    config_hash: Optional[str] = None,
    raw_samples: Sequence[Mapping[str, Any]] = (),
    summary: Optional[Mapping[str, Any]] = None,
) -> BenchmarkResult:
    """Project an S06 payload onto the frozen C6 model and validate it.

    ``BenchmarkResult`` is a ``VersionedModel`` with ``extra="forbid"``, so this
    call is a real schema check: if the frozen contract ever stops accepting the
    projection, construction fails here rather than at report time.
    """
    merged_summary: Dict[str, Any] = dict(summary or {})
    merged_summary[C6_NAMESPACE] = {
        "compile_mode": fields.compile_mode,
        "graph_identity": fields.graph_identity,
        "compile_identity": fields.compile_identity,
        "graph_count": fields.graph_count,
        "break_count": fields.break_count,
        "cache": {"layer": fields.cache_layer, "hit": bool(fields.cache_hit)},
        "compile_phase_ms": dict(fields.compile_phase_ms),
        "requested_lowering": fields.requested_lowering,
        "actual_lowering": fields.actual_lowering,
        "observed_kernel": fields.observed_kernel,
        "fallback_reason": fields.fallback_reason,
    }
    correctness = CorrectnessReport(
        passed=fields.correctness_status in ("pass", "partial"),
        method="hqsb.integration.differential.CorrectnessMatrix",
        details={"compile_correctness_status": fields.correctness_status},
    )
    return BenchmarkResult(
        run_id=run_id,
        timestamp=time.time(),
        environment=environment or EnvironmentInfo(),
        git_commit=git_commit,
        git_dirty=git_dirty,
        model_artifact_hash=model_artifact_hash,
        config_hash=config_hash,
        raw_samples=[dict(sample) for sample in raw_samples],
        summary=merged_summary,
        correctness=correctness,
        artifact_links=dict(fields.raw_artifacts),
    )


def c6_field_coverage(fields: Optional[S06ResultFields] = None) -> Dict[str, Any]:
    """Prove every required S06 C6 field is addressable in the projection."""
    probe = fields or S06ResultFields(
        graph_identity="probe-graph",
        compile_identity="probe-compile",
        cache_layer="probe-layer",
        compile_phase_ms={"cache_lookup": 0.0},
        requested_lowering="probe-requested",
        actual_lowering="probe-actual",
        fallback_reason="probe-reason",
        raw_artifacts={"graph": "artifact://graph"},
    )
    result = project_c6("coverage-probe", probe)
    payload = result.model_dump()
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for name, pointer in S06_C6_FIELDS:
        present, value = _walk(payload, pointer)
        rows.append({"field": name, "pointer": pointer, "present": present})
        if not present:
            missing.append(name)
    return {
        "ok": not missing,
        "fields": rows,
        "missing": missing,
        "schema_version": BenchmarkResult.SCHEMA_VERSION,
    }


# ── C7 projection ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TraceRecord:
    """One S06 event, ready to be projected onto C7."""

    kind: str
    span_id: str
    request_id: str = ""
    parent_span_id: str = ""
    timestamp_ns: int = 0
    name: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KIND_TO_EVENT_TYPE:
            raise ConfigError(
                f"unknown S06 trace kind {self.kind!r}",
                details={"field": "kind", "allowed": sorted(KIND_TO_EVENT_TYPE)},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "span_id": self.span_id,
            "request_id": self.request_id,
            "parent_span_id": self.parent_span_id,
            "timestamp_ns": self.timestamp_ns,
            "name": self.name,
            "attributes": dict(self.attributes),
        }


@dataclass
class TraceCollector:
    """Monotonic, request-scoped collector (no wall-clock ordering hazards)."""

    run_id: str
    records: List[TraceRecord] = field(default_factory=list)
    _clock: int = 0

    def emit(
        self,
        kind: str,
        *,
        span_id: str = "",
        request_id: str = "",
        parent_span_id: str = "",
        name: str = "",
        **attributes: Any,
    ) -> TraceRecord:
        self._clock += 1
        record = TraceRecord(
            kind=kind,
            span_id=span_id or f"{kind}-{self._clock}",
            request_id=request_id,
            parent_span_id=parent_span_id,
            timestamp_ns=self._clock,
            name=name or kind,
            attributes=attributes,
        )
        self.records.append(record)
        return record

    def monotonic(self) -> bool:
        return all(
            left.timestamp_ns <= right.timestamp_ns
            for left, right in zip(self.records, self.records[1:])
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "events": [record.as_dict() for record in self.records],
            "monotonic": self.monotonic(),
        }


def to_trace_events(
    records: Sequence[TraceRecord], *, run_id: str, trace_id: str
) -> Tuple[TraceEvent, ...]:
    """Project S06 records onto the frozen C7 event model.

    The frozen ``event_type`` is a coarse category; ``attributes['hqsb_kind']``
    carries the exact S06 kind.  Extending ``TraceEventType`` was rejected to
    avoid touching a frozen contract used by migration tests (see the S06
    development report §decisions).
    """
    events: List[TraceEvent] = []
    for record in records:
        event_type, _rationale = KIND_TO_EVENT_TYPE[record.kind]
        events.append(
            TraceEvent(
                event_type=TraceEventType(event_type),
                timestamp_ns=record.timestamp_ns,
                trace_id=trace_id,
                span_id=record.span_id,
                parent_span_id=record.parent_span_id or None,
                name=record.name,
                attributes={
                    C7_KIND_ATTRIBUTE: record.kind,
                    C7_RUN_ATTRIBUTE: run_id,
                    "request_id": record.request_id,
                    **dict(record.attributes),
                },
            )
        )
    return tuple(events)


def c7_kind_coverage() -> Dict[str, Any]:
    """Prove every S06 event kind maps onto a valid frozen C7 event."""
    rows: List[Dict[str, Any]] = []
    invalid: List[str] = []
    for kind in sorted(KIND_TO_EVENT_TYPE):
        event_type, rationale = KIND_TO_EVENT_TYPE[kind]
        try:
            TraceEvent(
                event_type=TraceEventType(event_type),
                timestamp_ns=0,
                trace_id="coverage",
                span_id=f"probe-{kind}",
                name=kind,
                attributes={C7_KIND_ATTRIBUTE: kind},
            )
            ok = True
        except Exception:  # noqa: BLE001 - the coverage audit must report, not raise
            ok = False
        rows.append(
            {
                "kind": kind,
                "event_type": event_type,
                "rationale": rationale,
                "valid": ok,
            }
        )
        if not ok:
            invalid.append(kind)
    return {"ok": not invalid, "kinds": rows, "invalid": invalid}


def trace_join_check(events: Sequence[TraceEvent]) -> Dict[str, Any]:
    """Verify run/span/parent/monotonic requirements can be satisfied."""
    by_span = {event.span_id: event for event in events}
    dangling = [
        event.span_id
        for event in events
        if event.parent_span_id and event.parent_span_id not in by_span
    ]
    ordered = list(events)
    non_monotonic = [
        (left.span_id, right.span_id)
        for left, right in zip(ordered, ordered[1:])
        if right.timestamp_ns < left.timestamp_ns
    ]
    missing_run = [
        event.span_id
        for event in events
        if C7_RUN_ATTRIBUTE not in event.attributes
    ]
    return {
        "ok": not dangling and not non_monotonic and not missing_run,
        "dangling_parent": dangling,
        "non_monotonic_pairs": non_monotonic,
        "missing_run_id": missing_run,
        "events": len(events),
    }


def c6_c7_summary() -> Dict[str, Any]:
    """Machine-readable summary used by the driver and the reports."""
    return {
        "c6": c6_field_coverage(),
        "c7": c7_kind_coverage(),
        "c6_schema_version": BenchmarkResult.SCHEMA_VERSION,
        "c7_schema_version": TraceEvent.SCHEMA_VERSION,
    }


def dump_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


__all__ = [
    "C6_NAMESPACE",
    "C7_KIND_ATTRIBUTE",
    "C7_RUN_ATTRIBUTE",
    "KIND_TO_EVENT_TYPE",
    "S06_C6_FIELDS",
    "S06ResultFields",
    "TraceCollector",
    "TraceRecord",
    "c6_c7_summary",
    "c6_field_coverage",
    "c7_kind_coverage",
    "dump_json",
    "project_c6",
    "to_trace_events",
    "trace_join_check",
]
