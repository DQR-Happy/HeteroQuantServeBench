"""End-to-end observability: trace context, spans, metrics, logs, root cause.

The requirement is not "a nice dashboard" but that **metrics, logs, traces and
profiles agree on the same request** and that one slow request can be explained
from a single ID.  Four properties are enforced here:

* ids are separate (``request_id`` is the business key, ``trace_id``/
  ``span_id`` follow W3C trace context, and an untrusted inbound header is
  validated against a trust boundary rather than believed);
* a clock domain is attached to every timestamp, so cross-process numbers are
  joined through a measured offset — and GPU time never comes from a host
  enqueue duration;
* every counter can be recomputed from raw terminal events, and histograms keep
  their buckets (an aggregate quantile is never reported as a percentile);
* prompt/token text never enters telemetry; only hashes and lengths do.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.benchmark.metrics import percentile

#: The id relationships of details README §4.
ID_RELATIONSHIPS: Tuple[str, ...] = (
    "request_id",
    "trace_id",
    "span_id",
    "parent_span_id",
    "attempt_id",
    "backend_request_id",
    "batch_id",
    "iteration_id",
    "kernel_correlation_id",
)

#: The service span model (E08-10 §5).
SPAN_KINDS: Tuple[str, ...] = (
    "HTTP_SERVER",
    "parse_validate",
    "tokenize_template",
    "admission",
    "queue_wait",
    "route",
    "capability_filter",
    "health_filter",
    "score",
    "backend_attempt",
    "runtime_wait",
    "prefill",
    "decode_iteration",
    "kv_lookup_allocate",
    "attention",
    "kernel",
    "sampling",
    "stream_frame",
    "serialize",
    "socket_write",
    "terminal_or_error",
    "cleanup",
)

#: The metric definition table (name → kind).
METRIC_KINDS: Mapping[str, str] = {
    "offered": "counter",
    "received": "counter",
    "valid": "counter",
    "admitted": "counter",
    "completed": "counter",
    "good": "counter",
    "rejected": "counter",
    "error": "counter",
    "cancelled": "counter",
    "retried": "counter",
    "fallback": "counter",
    "tokens_emitted": "counter",
    "bytes_emitted": "counter",
    "cache_hit": "counter",
    "cache_eviction": "counter",
    "circuit_transition": "counter",
    "client_ttft_ms": "histogram",
    "server_ttft_ms": "histogram",
    "runtime_ttft_ms": "histogram",
    "queue_wait_ms": "histogram",
    "validation_ms": "histogram",
    "route_ms": "histogram",
    "backend_ms": "histogram",
    "stream_ms": "histogram",
    "e2e_ms": "histogram",
    "itl_ms": "histogram",
    "cleanup_ms": "histogram",
    "recovery_ms": "histogram",
    "queue_depth": "gauge",
    "inflight_requests": "gauge",
    "active_tasks": "gauge",
    "connections": "gauge",
    "kv_bytes": "gauge",
    "cache_bytes": "gauge",
    "rss_bytes": "gauge",
    "gpu_memory_bytes": "gauge",
    "buffered_bytes": "gauge",
    "backend_health": "gauge",
    "circuit_state": "gauge",
}

#: Labels that would explode cardinality if they were allowed.
FORBIDDEN_LABELS: Tuple[str, ...] = ("request_id", "prompt", "model_path", "trace_id", "token_text")

#: Sampling modes.
SAMPLING_MODES: Tuple[str, ...] = (
    "metrics_always_on",
    "minimal_trace",
    "head_sampling",
    "tail_and_error_sampling",
    "full_trace_and_profile",
)

#: Root-cause classes (E08-10 §11).
ROOT_CAUSE_CLASSES: Tuple[str, ...] = (
    "client_send_or_loadgen_lag",
    "ingress_or_network",
    "gateway_parse_or_tokenization",
    "admission_or_queue_hol",
    "route_or_telemetry",
    "backend_scheduler_or_preemption",
    "prefix_miss_or_eviction",
    "prefill_decode_or_kernel",
    "cpu_launch_or_sync",
    "stream_serialize_socket_or_slow_reader",
    "retry_fault_or_recovery",
    "unknown_or_unaccounted",
)

#: The clock domains a service trace may carry.
CLOCK_DOMAINS: Tuple[str, ...] = (
    "loadgen_monotonic",
    "gateway_monotonic",
    "backend_monotonic",
    "wall_clock_utc",
    "gpu_event",
    "profiler",
)

_TRACEPARENT_RE = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace>[0-9a-f]{32})-(?P<span>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$"
)


@dataclass(frozen=True)
class TraceContext:
    """W3C trace context with an explicit trust boundary."""

    trace_id: str
    span_id: str
    parent_span_id: str = ""
    sampled: bool = True
    tracestate: str = ""
    trusted: bool = True
    version: str = "00"

    def header(self) -> str:
        return f"{self.version}-{self.trace_id}-{self.span_id}-{'01' if self.sampled else '00'}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "sampled": self.sampled,
            "tracestate": self.tracestate,
            "trusted": self.trusted,
            "version": self.version,
            "traceparent": self.header(),
        }


def parse_traceparent(value: str, *, trusted_source: bool) -> Dict[str, Any]:
    """Validate an inbound ``traceparent``; a bad one is *refused*, not trusted."""
    match = _TRACEPARENT_RE.match(value.strip().lower())
    if not match:
        return {
            "valid": False,
            "reason": "malformed traceparent (expected 00-<32 hex>-<16 hex>-<2 hex>)",
            "trusted": False,
        }
    version = match.group("version")
    if version == "ff":
        return {"valid": False, "reason": "version ff is forbidden by the spec", "trusted": False}
    trace_id = match.group("trace")
    span_id = match.group("span")
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return {"valid": False, "reason": "all-zero trace/span id is invalid", "trusted": False}
    return {
        "valid": True,
        "reason": "" if trusted_source else "accepted from an untrusted source (validated, not believed)",
        "trusted": bool(trusted_source),
        "context": TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            sampled=bool(int(match.group("flags"), 16) & 0x01),
            trusted=bool(trusted_source),
            version=version,
        ),
    }


def local_trace_context(*, trace_id: str, span_id: str, parent_span_id: str = "") -> TraceContext:
    for name, value, length in (("trace_id", trace_id, 32), ("span_id", span_id, 16)):
        if len(value) != length or any(char not in "0123456789abcdef" for char in value):
            raise ConfigError(f"{name} must be {length} lowercase hex characters")
    return TraceContext(trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id)


@dataclass
class Span:
    """One span, with its clock domain and its parent."""

    kind: str
    span_id: str
    request_id: str
    trace_id: str
    parent_span_id: str = ""
    clock: str = "gateway_monotonic"
    start_ns: Optional[int] = None
    end_ns: Optional[int] = None
    iteration: Optional[int] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""

    def __post_init__(self) -> None:
        if self.kind not in SPAN_KINDS:
            raise ConfigError(f"unknown span kind {self.kind!r}; expected one of {list(SPAN_KINDS)}")
        if self.clock not in CLOCK_DOMAINS:
            raise ConfigError(f"unknown clock domain {self.clock!r}")

    @property
    def duration_ns(self) -> Optional[int]:
        if self.start_ns is None or self.end_ns is None:
            return None
        return self.end_ns - self.start_ns

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "span_id": self.span_id,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "clock": self.clock,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
            "iteration": self.iteration,
            "attributes": dict(self.attributes),
            "error": self.error,
        }


class SpanRecorder:
    """Collects spans and checks the chain (dangling/cross-request/monotonic)."""

    def __init__(self, *, run_id: str) -> None:
        self.run_id = run_id
        self.spans: List[Span] = []

    def emit(
        self,
        kind: str,
        *,
        request_id: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str = "",
        clock: str = "gateway_monotonic",
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        **attributes: Any,
    ) -> Span:
        span = Span(
            kind=kind,
            span_id=span_id,
            request_id=request_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            clock=clock,
            start_ns=start_ns,
            end_ns=end_ns,
            attributes=dict(attributes),
        )
        self.spans.append(span)
        return span

    def join_audit(self) -> Dict[str, Any]:
        by_span = {span.span_id: span for span in self.spans}
        problems: List[str] = []
        for span in self.spans:
            if span.parent_span_id:
                parent = by_span.get(span.parent_span_id)
                if parent is None:
                    problems.append(f"{span.span_id}: parent {span.parent_span_id} is missing")
                elif parent.request_id != span.request_id:
                    problems.append(
                        f"{span.span_id}: parent belongs to request {parent.request_id}"
                    )
                elif (
                    parent.start_ns is not None
                    and span.start_ns is not None
                    and span.start_ns < parent.start_ns
                ):
                    problems.append(f"{span.span_id}: starts before its parent")
            if span.end_ns is not None and span.start_ns is not None and span.end_ns < span.start_ns:
                problems.append(f"{span.span_id}: ends before it starts")
        return {"ok": not problems, "problems": problems, "spans": len(self.spans)}

    def chain_for(self, request_id: str) -> List[Span]:
        return [span for span in self.spans if span.request_id == request_id]

    def critical_path(self, request_id: str) -> Dict[str, Any]:
        """Overlaps are reported; a naive sum is not a critical path."""
        spans = [span for span in self.chain_for(request_id) if span.duration_ns is not None]
        total = sum(span.duration_ns or 0 for span in spans)
        longest = max((span.duration_ns or 0 for span in spans), default=0)
        return {
            "spans": len(spans),
            "sum_of_spans_ns": total,
            "longest_span_ns": longest,
            "note": (
                "the sum of spans can exceed the request; overlaps are expected and must "
                "not be 'fixed' by rescaling"
            ),
        }


# ── metrics ────────────────────────────────────────────────────────────────


@dataclass
class Histogram:
    """Buckets plus raw samples: a percentile is always recomputed from raw."""

    name: str
    unit: str
    buckets: Tuple[float, ...]
    slo_ms: Optional[float] = None
    timeout_ms: Optional[float] = None
    samples: List[float] = field(default_factory=list)
    exemplars: List[Dict[str, Any]] = field(default_factory=list)
    overflow_count: int = 0

    def __post_init__(self) -> None:
        if not self.buckets or list(self.buckets) != sorted(self.buckets):
            raise ConfigError(f"{self.name}: histogram buckets must be sorted and non-empty")
        if self.slo_ms is not None and self.timeout_ms is not None:
            if self.timeout_ms > self.buckets[-1] or self.slo_ms > self.buckets[-1]:
                raise ConfigError(
                    f"{self.name}: the highest bucket ({self.buckets[-1]}) does not cover the "
                    "SLO/timeout; truncation at the top bucket is forbidden (E08-10 §8)"
                )

    def observe(self, value: float, *, exemplar: Optional[Mapping[str, Any]] = None) -> None:
        self.samples.append(float(value))
        if value > self.buckets[-1]:
            self.overflow_count += 1
        if exemplar is not None:
            self.exemplars.append(dict(exemplar))

    def count(self) -> Dict[str, int]:
        counts = {f"le_{bucket}": 0 for bucket in self.buckets}
        for value in self.samples:
            for bucket in self.buckets:
                if value <= bucket:
                    counts[f"le_{bucket}"] += 1
                    break
        return counts

    def quantile(self, q: float) -> Optional[float]:
        if not self.samples:
            return None
        return percentile(sorted(self.samples), q)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "buckets": list(self.buckets),
            "buckets_cumulative": self.count(),
            "count": len(self.samples),
            "sum": sum(self.samples),
            "overflow": self.overflow_count,
            "p50": self.quantile(0.50),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
            "exemplars": list(self.exemplars[-3:]),
        }


class MetricsRegistry:
    """Counters, histograms and gauges with a label whitelist and raw recompute."""

    def __init__(self, *, allowed_labels: Sequence[str] = ("service", "instance", "class", "tenant", "status")) -> None:
        self.allowed_labels = tuple(allowed_labels)
        self.counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.gauges: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        self.observations: List[Dict[str, Any]] = []

    def _labels(self, labels: Mapping[str, str]) -> Tuple[Tuple[str, str], ...]:
        forbidden = [name for name in labels if name in FORBIDDEN_LABELS]
        if forbidden:
            raise ConfigError(
                f"metric label(s) {forbidden} would explode cardinality; use a histogram "
                "exemplar instead (E08-10 §8)"
            )
        unknown = [name for name in labels if name not in self.allowed_labels]
        if unknown:
            raise ConfigError(f"metric label(s) {unknown} are not in the whitelist")
        return tuple(sorted((str(name), str(value)) for name, value in labels.items()))

    def counter(self, name: str, *, value: float = 1.0, **labels: str) -> None:
        if METRIC_KINDS.get(name) != "counter":
            raise ConfigError(f"{name!r} is not declared as a counter")
        key = (name, self._labels(labels))
        self.counters[key] = self.counters.get(key, 0.0) + value
        self.observations.append({"kind": "counter", "name": name, "value": value, "labels": dict(labels)})

    def histogram(
        self,
        name: str,
        *,
        unit: str,
        buckets: Sequence[float],
        slo_ms: Optional[float] = None,
        timeout_ms: Optional[float] = None,
    ) -> Histogram:
        if METRIC_KINDS.get(name) != "histogram":
            raise ConfigError(f"{name!r} is not declared as a histogram")
        if name in self.histograms:
            return self.histograms[name]
        histogram = Histogram(
            name=name,
            unit=unit,
            buckets=tuple(float(item) for item in buckets),
            slo_ms=slo_ms,
            timeout_ms=timeout_ms,
        )
        self.histograms[name] = histogram
        return histogram

    def gauge(self, name: str, value: float, **labels: str) -> None:
        if METRIC_KINDS.get(name) != "gauge":
            raise ConfigError(f"{name!r} is not declared as a gauge")
        self.gauges[(name, self._labels(labels))] = float(value)

    def terminal_events(self, events: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
        """Recompute the counters from raw terminal events (never from self-reports)."""
        counts: Dict[str, int] = {
            "received": 0,
            "valid": 0,
            "admitted": 0,
            "completed": 0,
            "good": 0,
            "rejected": 0,
            "error": 0,
            "cancelled": 0,
        }
        for event in events:
            outcome = str(event.get("outcome", ""))
            if event.get("valid"):
                counts["valid"] += 1
            if event.get("admitted"):
                counts["admitted"] += 1
            if outcome == "completed":
                counts["completed"] += 1
            elif outcome == "rejected":
                counts["rejected"] += 1
            elif outcome == "cancelled":
                counts["cancelled"] += 1
            elif outcome in ("error", "timeout", "backend_failure"):
                counts["error"] += 1
            if event.get("good"):
                counts["good"] += 1
        counts["received"] = len(events)
        return counts

    def consistency_check(self, raw_events: Sequence[Mapping[str, Any]], *, tolerance: float = 0.0) -> Dict[str, Any]:
        """Counters must equal what the raw terminal events imply."""
        recomputed = self.terminal_events(raw_events)
        reported = {
            name: sum(value for (key, _labels), value in self.counters.items() if key == name)
            for name in recomputed
        }
        problems = [
            f"{name}: reported {reported[name]} vs raw {value}"
            for name, value in recomputed.items()
            if abs(reported[name] - value) > tolerance
        ]
        return {
            "ok": not problems,
            "problems": problems,
            "recomputed": recomputed,
            "reported": reported,
        }

    def exposition(self) -> str:
        lines: List[str] = []
        for (name, labels), value in sorted(self.counters.items()):
            label_text = ",".join(f'{key}="{item}"' for key, item in labels)
            lines.append(f"hqsb_{name}{{{label_text}}} {value}")
        for (name, labels), value in sorted(self.gauges.items()):
            label_text = ",".join(f'{key}="{item}"' for key, item in labels)
            lines.append(f"hqsb_{name}{{{label_text}}} {value}")
        for name, histogram in sorted(self.histograms.items()):
            for bucket, count in histogram.count().items():
                lines.append(f"hqsb_{name}_bucket{{le=\"{bucket[3:]}\"}} {count}")
            lines.append(f"hqsb_{name}_count {len(histogram.samples)}")
            lines.append(f"hqsb_{name}_sum {sum(histogram.samples)}")
        return "\n".join(lines) + "\n"


# ── logs ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LogRecord:
    """The structured log schema (E08-10 §9)."""

    timestamp_ns: int
    clock: str
    service_version: str
    instance_id: str
    event: str
    request_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    attempt_id: str = ""
    tenant: str = ""
    request_class: str = ""
    model_epoch: str = ""
    policy_epoch: str = ""
    action: str = ""
    reason: str = ""
    requested: str = ""
    actual: str = ""
    error_cause: str = ""
    retryability: str = ""
    schema_version: str = "1.0.0"
    resource_counters: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "clock": self.clock,
            "service_version": self.service_version,
            "instance_id": self.instance_id,
            "event": self.event,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "attempt_id": self.attempt_id,
            "tenant": self.tenant,
            "request_class": self.request_class,
            "model_epoch": self.model_epoch,
            "policy_epoch": self.policy_epoch,
            "action": self.action,
            "reason": self.reason,
            "requested": self.requested,
            "actual": self.actual,
            "error_cause": self.error_cause,
            "retryability": self.retryability,
            "resource_counters": dict(self.resource_counters),
            "schema_version": self.schema_version,
        }


def redact_text(value: str) -> Dict[str, Any]:
    """Prompt/token text never enters telemetry; a hash and a length do."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return {"sha256": digest, "length": len(value)}


def redaction_audit(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    problems: List[str] = []
    for index, record in enumerate(records):
        for key, value in record.items():
            if key in ("prompt", "token_text", "messages", "content"):
                problems.append(f"record {index}: field {key!r} must not be logged raw")
            if key == "token_ids" and isinstance(value, (list, tuple)) and len(value) > 8:
                problems.append(f"record {index}: token_ids logged verbatim")
    return {
        "ok": not problems,
        "problems": problems,
        "note": "only hashes and lengths are allowed; fixtures are controlled inputs",
    }


# ── sampling and overhead ──────────────────────────────────────────────────


@dataclass(frozen=True)
class SamplingPolicy:
    """Head sampling must not remove the tail/errors from the evidence."""

    mode: str
    head_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.mode not in SAMPLING_MODES:
            raise ConfigError(f"unknown sampling mode {self.mode!r}")

    def keep(
        self,
        *,
        trace_id: str,
        is_error: bool = False,
        is_slow: bool = False,
        manually_selected: bool = False,
    ) -> Dict[str, Any]:
        if self.mode == "full_trace_and_profile":
            return {"keep": True, "reason": "full tracing"}
        if self.mode == "minimal_trace":
            return {"keep": True, "reason": "minimal tracing keeps every request meta-event"}
        if is_error or is_slow or manually_selected:
            return {"keep": True, "reason": "tail/error targeting: never sampled out"}
        if self.mode == "head_sampling":
            bucket = int(trace_id[-2:], 16) / 255.0 if trace_id else 0.0
            return {
                "keep": bucket < self.head_ratio,
                "reason": f"head sample ratio {self.head_ratio}",
            }
        return {"keep": True, "reason": "tail/error sampling keeps a baseline stream"}


@dataclass(frozen=True)
class OverheadObservation:
    mode: str
    cpu_ms: float
    latency_p99_ms: float
    goodput: float
    log_bytes: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "cpu_ms": self.cpu_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "goodput": self.goodput,
            "log_bytes": self.log_bytes,
        }


def overhead_report(
    observations: Sequence[OverheadObservation], *, max_p99_inflation: float = 0.05
) -> Dict[str, Any]:
    """Observer effect must be quantified; a full-trace run is not a timing run."""
    if not observations:
        return {"ok": False, "problems": ["no overhead observation"], "rows": []}
    baseline = next((item for item in observations if item.mode in ("off", "metrics_always_on")), observations[0])
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for item in observations:
        inflation = (
            (item.latency_p99_ms - baseline.latency_p99_ms) / baseline.latency_p99_ms
            if baseline.latency_p99_ms
            else 0.0
        )
        usable_for_timing = inflation <= max_p99_inflation
        if item.mode in ("full_trace_and_profile",) and usable_for_timing:
            problems.append(
                "full tracing shows no measurable inflation: check whether it is really on"
            )
        rows.append(
            {
                **item.as_dict(),
                "p99_inflation": inflation,
                "usable_for_timing": usable_for_timing,
            }
        )
    return {
        "ok": not problems,
        "problems": problems,
        "rows": rows,
        "note": (
            "the absolute performance run uses the low-overhead mode; the full trace is "
            "used for mechanism, with the inflation reported"
        ),
    }


# ── root cause ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RootCauseClaim:
    """A root cause needs a counterfactual; a long span alone is not a cause."""

    request_id: str
    root_cause: str
    evidence_span: str
    counterfactual: str = ""
    counterfactual_supported: bool = False
    unaccounted_ns: int = 0
    selection_rule: str = ""

    def __post_init__(self) -> None:
        if self.root_cause not in ROOT_CAUSE_CLASSES:
            raise ConfigError(
                f"unknown root-cause class {self.root_cause!r}; expected one of "
                f"{list(ROOT_CAUSE_CLASSES)}"
            )

    def verdict(self) -> Dict[str, Any]:
        if self.root_cause == "unknown_or_unaccounted":
            return {
                "status": "UNKNOWN",
                "reason": "the request was not explained; this is an observability gap",
                "claim_allowed": False,
            }
        if not self.counterfactual_supported:
            return {
                "status": "NOT_PROVEN",
                "reason": "no counterfactual supports the claim",
                "claim_allowed": False,
            }
        return {"status": "ROOT_CAUSE", "reason": self.counterfactual, "claim_allowed": True}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "root_cause": self.root_cause,
            "evidence_span": self.evidence_span,
            "counterfactual": self.counterfactual,
            "counterfactual_supported": self.counterfactual_supported,
            "unaccounted_ns": self.unaccounted_ns,
            "selection_rule": self.selection_rule,
            "verdict": self.verdict(),
        }


def slow_request_selection_rule(
    *, candidates: Sequence[Mapping[str, Any]], rule: str = "p99_band_random"
) -> Dict[str, Any]:
    """Pick the slow request by rule, not by which one is easiest to explain."""
    if not candidates:
        return {"ok": False, "reason": "no candidate"}
    if rule == "p99_band_random":
        ranked = sorted(candidates, key=lambda item: float(item.get("e2e_ms", 0.0)), reverse=True)
        band = ranked[: max(1, len(ranked) // 100 or 1)]
        return {
            "ok": True,
            "rule": rule,
            "candidates": [dict(item) for item in band],
            "selection": dict(band[0]),
            "note": "the confirmed case uses the pre-registered rule, not a convenient request",
        }
    raise ConfigError(f"unknown selection rule {rule!r}")


def classify_time_breakdown(spans: Sequence[Span], *, total_ns: int) -> Dict[str, Any]:
    """Attribute time to stages and report what is left unexplained."""
    buckets: Dict[str, int] = {}
    for span in spans:
        if span.duration_ns is None:
            continue
        buckets[span.kind] = buckets.get(span.kind, 0) + span.duration_ns
    accounted = sum(buckets.values())
    return {
        "buckets_ns": buckets,
        "accounted_ns": accounted,
        "total_ns": total_ns,
        "unaccounted_ns": total_ns - accounted,
        "note": (
            "a negative unaccounted time means overlap; a large positive one means the "
            "telemetry is missing a stage"
        ),
    }


def evidence_consistency(
    *,
    metric_value: Optional[float],
    log_value: Optional[float],
    trace_value: Optional[float],
    tolerance_rel: float = 1e-6,
) -> Dict[str, Any]:
    """Metrics, logs and traces must tell the same story for one request."""
    present = [value for value in (metric_value, log_value, trace_value) if value is not None]
    if len(present) < 2:
        return {"ok": False, "reason": "fewer than two signals carry this number"}
    reference = present[0]
    problems = [
        f"signal value {value} differs from {reference}"
        for value in present[1:]
        if reference and abs(value - reference) / abs(reference) > tolerance_rel
    ]
    return {"ok": not problems, "problems": problems, "reference": reference}


__all__ = [
    "CLOCK_DOMAINS",
    "FORBIDDEN_LABELS",
    "Histogram",
    "ID_RELATIONSHIPS",
    "LogRecord",
    "METRIC_KINDS",
    "MetricsRegistry",
    "OverheadObservation",
    "ROOT_CAUSE_CLASSES",
    "RootCauseClaim",
    "SAMPLING_MODES",
    "SPAN_KINDS",
    "SamplingPolicy",
    "Span",
    "SpanRecorder",
    "TraceContext",
    "classify_time_breakdown",
    "evidence_consistency",
    "local_trace_context",
    "overhead_report",
    "parse_traceparent",
    "redact_text",
    "redaction_audit",
    "slow_request_selection_rule",
]
