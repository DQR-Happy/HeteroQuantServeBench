"""C6/C7 projection for the service layer (frozen schemas, no new fields).

The service produces counters, timings and spans; the repository's frozen
contracts are :class:`~hqsb.core.contracts.result.BenchmarkResult` (C6) and
:class:`~hqsb.core.contracts.trace.TraceEvent` (C7).  This module projects the
service evidence onto them **without changing them**: service-specific payloads
go into the ``summary["s08"]`` namespace and into event attributes, exactly as
S06/S07 did for their layers.

It also audits coverage: a projection that silently drops a service field would
make the "one request ID rebuilds the whole chain" claim unfalsifiable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.contracts.result import BenchmarkResult, CorrectnessReport, EnvironmentInfo
from hqsb.core.contracts.trace import TraceEvent, TraceEventType

#: Namespace used inside the frozen C6 ``summary`` map.
C6_NAMESPACE = "s08"

#: Service-level fields the projection must carry (details README §22).
S08_C6_FIELDS: Tuple[str, ...] = (
    "service_id",
    "service_version",
    "service_commit",
    "protocol_profile_hash",
    "slo_spec_hash",
    "backend_registry_uri",
    "backend_instances",
    "selected_routes",
    "policy_versions",
    "payload_trace_hash",
    "arrival_trace_hash",
    "topology_uri",
    "loadgen_calibration_uri",
    "funnel",
    "goodput",
    "latency_summary",
    "queue_summary",
    "cache_summary",
    "admission_summary",
    "fault_summary",
    "resource_summary",
)

#: The service span chain, in causal order (details README §22).
SPAN_CHAIN: Tuple[str, ...] = (
    "HTTP_SERVER",
    "parse_validate",
    "tokenize_template",
    "admission",
    "queue_wait",
    "route",
    "backend_attempt",
    "prefill",
    "decode_iteration",
    "stream_frame",
    "terminal_or_error",
    "cleanup",
)

#: Which C7 event type each service span maps to.
KIND_TO_EVENT_TYPE: Mapping[str, TraceEventType] = {
    "HTTP_SERVER": TraceEventType.NETWORK,
    "parse_validate": TraceEventType.QUEUE,
    "tokenize_template": TraceEventType.QUEUE,
    "admission": TraceEventType.QUEUE,
    "queue_wait": TraceEventType.QUEUE,
    "route": TraceEventType.QUEUE,
    "capability_filter": TraceEventType.QUEUE,
    "health_filter": TraceEventType.QUEUE,
    "score": TraceEventType.QUEUE,
    "backend_attempt": TraceEventType.QUEUE,
    "runtime_wait": TraceEventType.QUEUE,
    "prefill": TraceEventType.PREFILL,
    "decode_iteration": TraceEventType.DECODE,
    "kv_lookup_allocate": TraceEventType.CACHE,
    "attention": TraceEventType.KERNEL,
    "kernel": TraceEventType.KERNEL,
    "sampling": TraceEventType.DECODE,
    "stream_frame": TraceEventType.OUTPUT,
    "serialize": TraceEventType.OUTPUT,
    "socket_write": TraceEventType.NETWORK,
    "terminal_or_error": TraceEventType.OUTPUT,
    "cleanup": TraceEventType.QUEUE,
}


@dataclass
class S08ResultFields:
    """The service payload the projection carries (all optional by design)."""

    service_id: str = ""
    service_version: str = ""
    service_commit: str = ""
    protocol_profile_hash: str = ""
    slo_spec_hash: str = ""
    backend_registry_uri: str = ""
    backend_instances: Sequence[Mapping[str, Any]] = ()
    selected_routes: Sequence[Mapping[str, Any]] = ()
    policy_versions: Mapping[str, str] = field(default_factory=dict)
    payload_trace_hash: str = ""
    arrival_trace_hash: str = ""
    topology_uri: str = ""
    loadgen_calibration_uri: str = ""
    funnel: Mapping[str, Any] = field(default_factory=dict)
    goodput: Mapping[str, Any] = field(default_factory=dict)
    latency_summary: Mapping[str, Any] = field(default_factory=dict)
    queue_summary: Mapping[str, Any] = field(default_factory=dict)
    cache_summary: Mapping[str, Any] = field(default_factory=dict)
    admission_summary: Mapping[str, Any] = field(default_factory=dict)
    fault_summary: Mapping[str, Any] = field(default_factory=dict)
    resource_summary: Mapping[str, Any] = field(default_factory=dict)
    quality_status: str = "not_run"
    raw_artifacts: Mapping[str, str] = field(default_factory=dict)

    def as_summary(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id,
            "service_version": self.service_version,
            "service_commit": self.service_commit,
            "protocol_profile_hash": self.protocol_profile_hash,
            "slo_spec_hash": self.slo_spec_hash,
            "backend_registry_uri": self.backend_registry_uri,
            "backend_instances": [dict(item) for item in self.backend_instances],
            "selected_routes": [dict(item) for item in self.selected_routes],
            "policy_versions": dict(self.policy_versions),
            "payload_trace_hash": self.payload_trace_hash,
            "arrival_trace_hash": self.arrival_trace_hash,
            "topology_uri": self.topology_uri,
            "loadgen_calibration_uri": self.loadgen_calibration_uri,
            "funnel": dict(self.funnel),
            "goodput": dict(self.goodput),
            "latency_summary": dict(self.latency_summary),
            "queue_summary": dict(self.queue_summary),
            "cache_summary": dict(self.cache_summary),
            "admission_summary": dict(self.admission_summary),
            "fault_summary": dict(self.fault_summary),
            "resource_summary": dict(self.resource_summary),
            "quality_status": self.quality_status,
        }


def project_c6(
    run_id: str,
    fields: S08ResultFields,
    *,
    environment: Optional[EnvironmentInfo] = None,
    git_commit: Optional[str] = None,
    git_dirty: Optional[bool] = None,
    model_artifact_hash: Optional[str] = None,
    config_hash: Optional[str] = None,
    raw_samples: Sequence[Mapping[str, Any]] = (),
    summary: Optional[Mapping[str, Any]] = None,
    correctness: Optional[CorrectnessReport] = None,
) -> BenchmarkResult:
    """Project the service payload onto the frozen C6 model and validate it."""
    merged_summary: Dict[str, Any] = dict(summary or {})
    merged_summary[C6_NAMESPACE] = fields.as_summary()
    merged_summary["claim_level"] = "SOURCE"
    return BenchmarkResult(
        run_id=run_id,
        timestamp=time.time(),
        environment=environment or EnvironmentInfo(platform="service-run"),
        git_commit=git_commit,
        git_dirty=git_dirty,
        model_artifact_hash=model_artifact_hash,
        config_hash=config_hash,
        raw_samples=[dict(item) for item in raw_samples],
        summary=merged_summary,
        correctness=correctness,
        artifact_links=dict(fields.raw_artifacts),
    )


def to_trace_events(
    records: Sequence[Mapping[str, Any]], *, run_id: str, trace_id: str
) -> List[TraceEvent]:
    """Project service span records onto C7 events (kind → event type)."""
    events: List[TraceEvent] = []
    for index, record in enumerate(records):
        kind = str(record.get("kind", ""))
        if kind not in KIND_TO_EVENT_TYPE:
            continue
        events.append(
            TraceEvent(
                event_type=KIND_TO_EVENT_TYPE[kind],
                timestamp_ns=int(record.get("start_ns") or 0),
                trace_id=str(record.get("trace_id", trace_id)),
                span_id=str(record.get("span_id", f"{run_id}-{index}")),
                parent_span_id=record.get("parent_span_id") or None,
                name=kind,
                attributes={
                    "request_id": record.get("request_id", ""),
                    "clock": record.get("clock", ""),
                    "duration_ns": record.get("duration_ns"),
                    "iteration": record.get("iteration"),
                    "s08_kind": kind,
                    "error": record.get("error", ""),
                    **{
                        key: value
                        for key, value in dict(record.get("attributes", {})).items()
                        if isinstance(value, (str, int, float, bool))
                    },
                },
            )
        )
    return events


def chain_coverage(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Does the recorded chain contain every link the protocol asks for?"""
    kinds = {str(record.get("kind", "")) for record in records}
    missing = [kind for kind in SPAN_CHAIN if kind not in kinds]
    extra = sorted(kinds - set(SPAN_CHAIN)) if kinds else []
    return {
        "ok": not missing,
        "missing": missing,
        "extra": extra,
        "note": (
            "a missing link is an observability gap; it must be reported, not filled in "
            "by inference"
        ),
    }


def trace_join_check(events: Sequence[TraceEvent]) -> Dict[str, Any]:
    """Parent chains resolve, timestamps are monotonic per trace."""
    by_span = {event.span_id: event for event in events}
    problems: List[str] = []
    for event in events:
        if event.parent_span_id and event.parent_span_id not in by_span:
            problems.append(f"{event.span_id}: parent {event.parent_span_id} is missing")
        if event.parent_span_id:
            parent = by_span[event.parent_span_id]
            if parent.timestamp_ns > event.timestamp_ns:
                problems.append(f"{event.span_id}: starts before its parent")
    return {"ok": not problems, "problems": problems, "events": len(events)}


def c6_field_coverage() -> Dict[str, Any]:
    """Every declared service field must be carried by :class:`S08ResultFields`."""
    carrier = S08ResultFields()
    present = set(carrier.as_summary())
    missing = [name for name in S08_C6_FIELDS if name not in present]
    return {
        "ok": not missing,
        "missing": missing,
        "fields": list(S08_C6_FIELDS),
        "note": "the projection may not drop a service field silently",
    }


def c7_kind_coverage() -> Dict[str, Any]:
    """Every service span kind must have a C7 event-type mapping."""
    from hqsb.serving.observability import SPAN_KINDS

    missing = [kind for kind in SPAN_KINDS if kind not in KIND_TO_EVENT_TYPE]
    invalid = []
    for kind in SPAN_CHAIN:
        if kind not in KIND_TO_EVENT_TYPE:
            invalid.append(kind)
    return {
        "ok": not missing and not invalid,
        "missing": missing,
        "missing_from_chain": invalid,
        "note": "a span without an event type would disappear from the trace",
    }


def coverage_summary() -> Dict[str, Any]:
    """Machine-readable summary used by the driver, the gate and the reports."""
    return {
        "c6": c6_field_coverage(),
        "c7": c7_kind_coverage(),
        "chain": list(SPAN_CHAIN),
        "c6_namespace": C6_NAMESPACE,
        "c6_schema_version": BenchmarkResult.SCHEMA_VERSION,
        "c7_schema_version": TraceEvent.SCHEMA_VERSION,
    }


def dump_json(payload: Any) -> str:
    import json

    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


__all__ = [
    "C6_NAMESPACE",
    "KIND_TO_EVENT_TYPE",
    "S08_C6_FIELDS",
    "S08ResultFields",
    "SPAN_CHAIN",
    "c6_field_coverage",
    "c7_kind_coverage",
    "chain_coverage",
    "coverage_summary",
    "dump_json",
    "project_c6",
    "to_trace_events",
    "trace_join_check",
]
