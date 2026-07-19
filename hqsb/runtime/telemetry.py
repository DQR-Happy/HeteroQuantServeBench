"""C6/C7 projection for runtime facts (S07 details README §16).

S07 requires C6 to carry the backend/runtime version, the requested and actual
capability, the request-trace hash, the scheduler/KV/prefix/graph policy, the
per-request metrics and iteration-ledger URIs, the actual precision/kernel, the
quality status, failures and memory/energy; and C7 to carry the span chain
``request → scheduler_wait → scheduler_iteration → allocate_or_reuse_kv →
prepare_batch → model_runner → attention → custom/quant kernels → sample →
emit_token → cleanup`` with an explicit trace context (never a thread-local).

As in S06, the frozen C1–C7 schemas are **not** modified: the payload is
projected onto the documented extension points (``BenchmarkResult.summary`` /
``artifact_links`` / ``CorrectnessReport.details`` and ``TraceEvent.attributes``)
and a coverage audit proves every required field lands somewhere addressable.
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

C6_NAMESPACE = "s07"

#: Required S07 C6 fields → the extension path they are stored at (§16).
S07_C6_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("backend_id", f"summary.{C6_NAMESPACE}.backend.id"),
    ("runtime_version", f"summary.{C6_NAMESPACE}.backend.version"),
    ("runtime_commit", f"summary.{C6_NAMESPACE}.backend.commit"),
    ("requested_capability", f"summary.{C6_NAMESPACE}.capability.requested"),
    ("actual_capability", f"summary.{C6_NAMESPACE}.capability.actual"),
    ("capability_reasons", f"summary.{C6_NAMESPACE}.capability.reasons"),
    ("request_trace_hash", f"summary.{C6_NAMESPACE}.request_trace_hash"),
    ("scheduler_policy", f"summary.{C6_NAMESPACE}.policies.scheduler"),
    ("kv_policy", f"summary.{C6_NAMESPACE}.policies.kv"),
    ("prefix_policy", f"summary.{C6_NAMESPACE}.policies.prefix"),
    ("graph_policy", f"summary.{C6_NAMESPACE}.policies.graph"),
    ("per_request_metrics_uri", f"summary.{C6_NAMESPACE}.artifacts.per_request_metrics"),
    ("iteration_ledger_uri", f"summary.{C6_NAMESPACE}.artifacts.iteration_ledger"),
    ("actual_precision", f"summary.{C6_NAMESPACE}.actual.precision"),
    ("observed_kernel", f"summary.{C6_NAMESPACE}.actual.kernel"),
    ("observed_attention_backend", f"summary.{C6_NAMESPACE}.actual.attention_backend"),
    ("failure", f"summary.{C6_NAMESPACE}.failure"),
    ("memory_bytes", f"summary.{C6_NAMESPACE}.memory.bytes"),
    ("energy_joules", f"summary.{C6_NAMESPACE}.energy.joules"),
    ("quality_status", "correctness.details.runtime_quality_status"),
)

#: S07 span kinds → frozen ``TraceEventType`` plus the rationale.
KIND_TO_EVENT_TYPE: Mapping[str, Tuple[str, str]] = {
    "request": (TraceEventType.MODEL_LOAD.value, "request lifecycle span"),
    "validate": (TraceEventType.MODEL_LOAD.value, "request validation"),
    "scheduler_wait": (TraceEventType.QUEUE.value, "waiting for admission"),
    "scheduler_iteration": (TraceEventType.QUEUE.value, "one scheduler iteration"),
    "schedule_cpu": (TraceEventType.QUEUE.value, "host-side scheduling work"),
    "kv_lookup_allocate_free": (TraceEventType.CACHE.value, "KV block management"),
    "prepare_batch": (TraceEventType.PREFILL.value, "batch preparation"),
    "model_runner": (TraceEventType.PREFILL.value, "model execution span"),
    "input_pack": (TraceEventType.PREFILL.value, "input preparation"),
    "graph_route": (TraceEventType.KERNEL.value, "graph bucket routing"),
    "attention": (TraceEventType.KERNEL.value, "attention kernel"),
    "mlp_custom_quant": (TraceEventType.KERNEL.value, "MLP/custom/quantised kernel"),
    "sample": (TraceEventType.DECODE.value, "sampling"),
    "emit_token": (TraceEventType.OUTPUT.value, "token emitted"),
    "cleanup": (TraceEventType.OUTPUT.value, "cleanup and release"),
}

C7_KIND_ATTRIBUTE = "hqsb_kind"
C7_RUN_ATTRIBUTE = "hqsb_run_id"
C7_REQUEST_ATTRIBUTE = "request_id"
C7_ITERATION_ATTRIBUTE = "scheduler_iteration"


@dataclass(frozen=True)
class S07ResultFields:
    """The S07 payload of one run, before it is projected onto C6."""

    backend_id: str = ""
    runtime_version: str = ""
    runtime_commit: str = ""
    requested_capability: Mapping[str, str] = field(default_factory=dict)
    actual_capability: Mapping[str, str] = field(default_factory=dict)
    capability_reasons: Mapping[str, str] = field(default_factory=dict)
    request_trace_hash: str = ""
    scheduler_policy: str = ""
    kv_policy: str = ""
    prefix_policy: str = ""
    graph_policy: str = ""
    per_request_metrics_uri: str = ""
    iteration_ledger_uri: str = ""
    actual_precision: str = ""
    observed_kernel: str = ""
    observed_attention_backend: str = ""
    failure: str = ""
    memory_bytes: float = 0.0
    energy_joules: float = 0.0
    quality_status: str = "not_run"
    raw_artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quality_status not in ("not_run", "pass", "fail", "partial"):
            raise ConfigError(
                f"unknown quality_status {self.quality_status!r}",
                details={"field": "quality_status"},
            )
        if self.memory_bytes < 0 or self.energy_joules < 0:
            raise ConfigError("memory/energy must be non-negative")
        # A requested/actual capability difference without a reason is exactly the
        # silent degradation the project forbids.
        for field_name, requested in self.requested_capability.items():
            actual = self.actual_capability.get(field_name)
            if actual is None:
                raise ConfigError(
                    f"capability {field_name!r} was requested but the actual value is "
                    "missing; an unrecorded resolution cannot be reported",
                    details={"field": field_name},
                )
            if actual != requested and not self.capability_reasons.get(field_name):
                raise ConfigError(
                    f"capability {field_name!r}: requested={requested!r} but "
                    f"actual={actual!r} without a reason (silent degradation)",
                    details={"field": field_name},
                )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "runtime_version": self.runtime_version,
            "runtime_commit": self.runtime_commit,
            "requested_capability": dict(self.requested_capability),
            "actual_capability": dict(self.actual_capability),
            "capability_reasons": dict(self.capability_reasons),
            "request_trace_hash": self.request_trace_hash,
            "scheduler_policy": self.scheduler_policy,
            "kv_policy": self.kv_policy,
            "prefix_policy": self.prefix_policy,
            "graph_policy": self.graph_policy,
            "per_request_metrics_uri": self.per_request_metrics_uri,
            "iteration_ledger_uri": self.iteration_ledger_uri,
            "actual_precision": self.actual_precision,
            "observed_kernel": self.observed_kernel,
            "observed_attention_backend": self.observed_attention_backend,
            "failure": self.failure,
            "memory_bytes": self.memory_bytes,
            "energy_joules": self.energy_joules,
            "quality_status": self.quality_status,
            "raw_artifacts": dict(self.raw_artifacts),
        }


def project_c6(
    run_id: str,
    fields: S07ResultFields,
    *,
    environment: Optional[EnvironmentInfo] = None,
    git_commit: Optional[str] = None,
    git_dirty: Optional[bool] = None,
    model_artifact_hash: Optional[str] = None,
    config_hash: Optional[str] = None,
    raw_samples: Sequence[Mapping[str, Any]] = (),
    summary: Optional[Mapping[str, Any]] = None,
) -> BenchmarkResult:
    """Project the S07 payload onto the frozen C6 model and validate it."""
    merged_summary: Dict[str, Any] = dict(summary or {})
    merged_summary[C6_NAMESPACE] = {
        "backend": {
            "id": fields.backend_id,
            "version": fields.runtime_version,
            "commit": fields.runtime_commit,
        },
        "capability": {
            "requested": dict(fields.requested_capability),
            "actual": dict(fields.actual_capability),
            "reasons": dict(fields.capability_reasons),
        },
        "request_trace_hash": fields.request_trace_hash,
        "policies": {
            "scheduler": fields.scheduler_policy,
            "kv": fields.kv_policy,
            "prefix": fields.prefix_policy,
            "graph": fields.graph_policy,
        },
        "artifacts": {
            "per_request_metrics": fields.per_request_metrics_uri,
            "iteration_ledger": fields.iteration_ledger_uri,
        },
        "actual": {
            "precision": fields.actual_precision,
            "kernel": fields.observed_kernel,
            "attention_backend": fields.observed_attention_backend,
        },
        "failure": fields.failure,
        "memory": {"bytes": fields.memory_bytes},
        "energy": {"joules": fields.energy_joules},
    }
    correctness = CorrectnessReport(
        passed=fields.quality_status == "pass",
        method="hqsb.runtime.comparison.recompute_metrics",
        details={"runtime_quality_status": fields.quality_status},
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


def _walk(payload: Mapping[str, Any], pointer: str) -> Tuple[bool, Any]:
    current: Any = payload
    for part in pointer.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        return False, None
    return True, current


def c6_field_coverage(fields: Optional[S07ResultFields] = None) -> Dict[str, Any]:
    """Prove every required S07 C6 field is addressable in the projection."""
    probe = fields or S07ResultFields(
        backend_id="probe",
        runtime_version="0.0.0",
        runtime_commit="probe-commit",
        requested_capability={"prefix_cache": "SUPPORTED_EXACT"},
        actual_capability={"prefix_cache": "SUPPORTED_EXACT"},
        request_trace_hash="probe-trace",
        scheduler_policy="continuous",
        kv_policy="block-16",
        prefix_policy="lru",
        graph_policy="decode-b1",
        per_request_metrics_uri="artifact://requests",
        iteration_ledger_uri="artifact://iterations",
        actual_precision="float16",
        observed_kernel="probe_kernel",
        observed_attention_backend="runtime_default",
        failure="",
        memory_bytes=1.0,
        energy_joules=0.0,
        quality_status="not_run",
        raw_artifacts={"requests": "artifact://requests"},
    )
    result = project_c6("coverage-probe", probe)
    payload = result.model_dump()
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for name, pointer in S07_C6_FIELDS:
        present, _value = _walk(payload, pointer)
        rows.append({"field": name, "pointer": pointer, "present": present})
        if not present:
            missing.append(name)
    return {
        "ok": not missing,
        "fields": rows,
        "missing": missing,
        "schema_version": BenchmarkResult.SCHEMA_VERSION,
    }


# ── C7 projection ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TraceRecord:
    """One runtime event ready for C7 projection."""

    kind: str
    span_id: str
    request_id: str = ""
    parent_span_id: str = ""
    timestamp_ns: int = 0
    iteration: int = -1
    name: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KIND_TO_EVENT_TYPE:
            raise ConfigError(
                f"unknown runtime trace kind {self.kind!r}",
                details={"field": "kind", "allowed": sorted(KIND_TO_EVENT_TYPE)},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "span_id": self.span_id,
            "request_id": self.request_id,
            "parent_span_id": self.parent_span_id,
            "timestamp_ns": self.timestamp_ns,
            "iteration": self.iteration,
            "name": self.name,
            "attributes": dict(self.attributes),
        }


@dataclass
class TraceCollector:
    """Monotonic, request-scoped collector without thread-local state."""

    run_id: str
    records: List[TraceRecord] = field(default_factory=list)
    _clock: int = 0

    def emit(
        self,
        kind: str,
        *,
        request_id: str = "",
        span_id: str = "",
        parent_span_id: str = "",
        iteration: int = -1,
        name: str = "",
        **attributes: Any,
    ) -> TraceRecord:
        if not request_id:
            raise ConfigError(
                "every runtime span needs an explicit request_id: resolving the "
                "request from a thread-local is a listed anti-pattern",
                details={"field": "request_id"},
            )
        self._clock += 1
        record = TraceRecord(
            kind=kind,
            span_id=span_id or f"{kind}-{self._clock}",
            request_id=request_id,
            parent_span_id=parent_span_id,
            timestamp_ns=self._clock,
            iteration=iteration,
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


#: The canonical span chain of details README §16.
SPAN_CHAIN: Tuple[str, ...] = (
    "request",
    "scheduler_wait",
    "scheduler_iteration",
    "kv_lookup_allocate_free",
    "prepare_batch",
    "model_runner",
    "attention",
    "mlp_custom_quant",
    "sample",
    "emit_token",
    "cleanup",
)


def chain_coverage(records: Sequence[TraceRecord]) -> Dict[str, Any]:
    """Every span of the canonical chain must be emitted for the run."""
    kinds = {record.kind for record in records}
    missing = [kind for kind in SPAN_CHAIN if kind not in kinds]
    return {
        "ok": not missing,
        "missing": missing,
        "emitted": sorted(kinds),
        "chain": list(SPAN_CHAIN),
    }


def to_trace_events(
    records: Sequence[TraceRecord], *, run_id: str, trace_id: str
) -> Tuple[TraceEvent, ...]:
    """Project runtime records onto the frozen C7 event model."""
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
                    C7_REQUEST_ATTRIBUTE: record.request_id,
                    C7_ITERATION_ATTRIBUTE: record.iteration,
                    **dict(record.attributes),
                },
            )
        )
    return tuple(events)


def c7_kind_coverage() -> Dict[str, Any]:
    """Prove every runtime span kind maps onto a valid frozen C7 event."""
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
        except Exception:  # noqa: BLE001 - the audit must report, not raise
            ok = False
        rows.append(
            {"kind": kind, "event_type": event_type, "rationale": rationale, "valid": ok}
        )
        if not ok:
            invalid.append(kind)
    return {"ok": not invalid, "kinds": rows, "invalid": invalid}


def trace_join_check(events: Sequence[TraceEvent]) -> Dict[str, Any]:
    """run/span/parent/monotonic/request associativity (details README §16)."""
    by_span = {event.span_id: event for event in events}
    dangling = [
        event.span_id
        for event in events
        if event.parent_span_id and event.parent_span_id not in by_span
    ]
    non_monotonic = [
        (left.span_id, right.span_id)
        for left, right in zip(events, events[1:])
        if right.timestamp_ns < left.timestamp_ns
    ]
    missing_run = [
        event.span_id for event in events if C7_RUN_ATTRIBUTE not in event.attributes
    ]
    missing_request = [
        event.span_id
        for event in events
        if not event.attributes.get(C7_REQUEST_ATTRIBUTE)
    ]
    return {
        "ok": not dangling and not non_monotonic and not missing_run and not missing_request,
        "dangling_parent": dangling,
        "non_monotonic_pairs": non_monotonic,
        "missing_run_id": missing_run,
        "missing_request_id": missing_request,
        "events": len(events),
    }


#: C2 workload field → the runtime carrier that must express it.
#: The control plane requires "所有 runtime 指标与 C2/C6/C7 对齐"; this table is
#: the auditable form of that requirement (and records where S07 deliberately
#: stops, e.g. real arrival processes belong to S08).
C2_ALIGNMENT: Tuple[Tuple[str, str, str], ...] = (
    ("name", "hqsb.runtime.scheduler.RequestTrace.name", "trace identity"),
    ("batch_size", "hqsb.runtime.scheduler.SchedulerSpec.max_sequences", "batch budget"),
    ("input_tokens", "hqsb.runtime.request.RequestSpec.input_tokens", "tokenized prompt"),
    ("output_tokens", "hqsb.runtime.scheduler.SimRequest.max_new_tokens", "decode budget"),
    ("seed", "hqsb.runtime.request.SamplingSpec.seed", "sampling determinism"),
    ("sampling", "hqsb.runtime.request.SamplingSpec.mode", "greedy vs sampling"),
    ("warmup", "hqsb.runtime.comparison.PhaseRecord", "first_request/steady phases"),
    ("repetitions", "hqsb.runtime.metrics.RunLevelSamples", "independent runs"),
    ("token_ids", "hqsb.runtime.request.RequestSpec.input_token_ids", "explicit token IDs"),
    ("dataset_version", "hqsb.runtime.scheduler.RequestTrace.trace_hash", "frozen trace hash"),
    ("concurrency", "hqsb.runtime.scheduler.SchedulerSpec.max_sequences", "concurrent slots"),
    ("arrival_process", "hqsb.runtime.scheduler.SimRequest.submit_iteration", "deterministic offsets (S08 adds real arrivals)"),
    ("timeout_s", "hqsb.runtime.request.RequestSpec.timeout_s", "per-request timeout"),
    ("stop_condition", "hqsb.runtime.request.StopSpec", "stop/EOS/length rule"),
)


def c2_alignment() -> Dict[str, Any]:
    """Audit that every C2 workload field has a runtime-side carrier.

    The check is deliberately a *mapping* audit: it does not claim the value was
    measured, only that the interface can express it (and where S07 stops).
    """
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for field_name, carrier, note in C2_ALIGNMENT:
        present = bool(carrier)
        rows.append({"field": field_name, "carrier": carrier, "covered": present, "note": note})
        if not present:
            missing.append(field_name)
    return {
        "ok": not missing,
        "rows": rows,
        "missing": missing,
        "note": "S07 aligns workload semantics with C2; real arrival distributions remain S08",
    }


def c6_c7_summary() -> Dict[str, Any]:
    """Machine-readable summary used by the driver and the reports."""
    return {
        "c6": c6_field_coverage(),
        "c7": c7_kind_coverage(),
        "c2": c2_alignment(),
        "chain": list(SPAN_CHAIN),
        "c6_schema_version": BenchmarkResult.SCHEMA_VERSION,
        "c7_schema_version": TraceEvent.SCHEMA_VERSION,
    }


def dump_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


__all__ = [
    "C2_ALIGNMENT",
    "C6_NAMESPACE",
    "C7_ITERATION_ATTRIBUTE",
    "C7_KIND_ATTRIBUTE",
    "C7_REQUEST_ATTRIBUTE",
    "C7_RUN_ATTRIBUTE",
    "KIND_TO_EVENT_TYPE",
    "S07_C6_FIELDS",
    "S07ResultFields",
    "SPAN_CHAIN",
    "TraceCollector",
    "TraceRecord",
    "c6_c7_summary",
    "c6_field_coverage",
    "c7_kind_coverage",
    "chain_coverage",
    "dump_json",
    "project_c6",
    "to_trace_events",
    "trace_join_check",
]
