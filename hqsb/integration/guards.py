"""Guards, graph breaks, recompiles, runtime assertions and fallbacks (E06-05).

Five different events are routinely collapsed into "it recompiled"
(E06-05 §2):

* **guard failure** — a soundness mechanism fired; not a bug by itself;
* **recompile** — no compiled variant matched, so a new one was built;
* **graph break** — Dynamo stopped a graph and continued in eager;
* **runtime assertion** — an export/dynamic-graph check inside the graph;
* **fallback** — policy selected the eager/reference path *before* execution.

This module keeps them in separate counters with per-event reasons, defines the
dynamic-shape policy matrix, the ordered shape trace from the protocol, the
pre-registered storm thresholds and a guard-minimality audit.  It computes
metrics from samples it is given; it never runs an experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError, SchemaError

# ── guards ────────────────────────────────────────────────────────────────


class GuardKind:
    SHAPE = "shape"
    RANK = "rank"
    STRIDE = "stride"
    STORAGE_OFFSET = "storage_offset"
    DTYPE = "dtype"
    DEVICE = "device"
    LAYOUT = "layout"
    REQUIRES_GRAD = "requires_grad"
    AUTOCAST = "autocast"
    INFERENCE_MODE = "inference_mode"
    MODULE_IDENTITY = "module_identity"
    PARAM_IDENTITY = "param_identity"
    PYTHON_SCALAR = "python_scalar"
    CONFIG = "config"
    OP_SCHEMA = "op_schema"
    LIBRARY_VERSION = "library_version"
    QUANT_ARTIFACT = "quant_artifact"
    QUANT_POLICY = "quant_policy"
    BACKEND_CAPABILITY = "backend_capability"
    GLOBAL_STATE = "global_state"
    RNG = "rng"
    CONTROL_FLOW = "control_flow"
    DATA_DEPENDENT = "data_dependent"

    ALL = (
        SHAPE,
        RANK,
        STRIDE,
        STORAGE_OFFSET,
        DTYPE,
        DEVICE,
        LAYOUT,
        REQUIRES_GRAD,
        AUTOCAST,
        INFERENCE_MODE,
        MODULE_IDENTITY,
        PARAM_IDENTITY,
        PYTHON_SCALAR,
        CONFIG,
        OP_SCHEMA,
        LIBRARY_VERSION,
        QUANT_ARTIFACT,
        QUANT_POLICY,
        BACKEND_CAPABILITY,
        GLOBAL_STATE,
        RNG,
        CONTROL_FLOW,
        DATA_DEPENDENT,
    )

    @classmethod
    def require(cls, kind: str) -> str:
        if kind not in cls.ALL:
            raise SchemaError(
                f"unknown guard kind {kind!r}",
                details={"field": "kind", "allowed": list(cls.ALL)},
            )
        return kind


@dataclass(frozen=True)
class GuardRecord:
    """One guard with its source, first input and consequence (E06-05 §4)."""

    kind: str
    expression: str
    source: str
    first_input: str = ""
    failing_input: str = ""
    consequence: str = ""
    necessary: bool = True
    performance_only: bool = False
    introduced_by: str = ""

    def __post_init__(self) -> None:
        GuardKind.require(self.kind)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "expression": self.expression,
            "source": self.source,
            "first_input": self.first_input,
            "failing_input": self.failing_input,
            "consequence": self.consequence,
            "necessary": self.necessary,
            "performance_only": self.performance_only,
            "introduced_by": self.introduced_by,
        }


# ── compile ledger (five-way separation) ──────────────────────────────────


class EventKind:
    CAPTURE = "capture"
    GUARD_EVAL = "guard_eval"
    GUARD_FAIL = "guard_fail"
    RECOMPILE = "recompile"
    GRAPH_BREAK = "graph_break"
    FALLBACK = "fallback"
    RUNTIME_ASSERT = "runtime_assert"
    CACHE_LOOKUP = "cache_lookup"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_INVALIDATE = "cache_invalidate"
    PATTERN_CANDIDATE = "pattern_candidate"
    PATTERN_HIT = "pattern_hit"
    PATTERN_REJECT = "pattern_reject"
    LOWERING_SELECT = "lowering_select"
    KERNEL = "kernel"
    RESOURCE_CREATE = "resource_create"
    RESOURCE_DESTROY = "resource_destroy"
    ERROR = "error"

    ALL = (
        CAPTURE,
        GUARD_EVAL,
        GUARD_FAIL,
        RECOMPILE,
        GRAPH_BREAK,
        FALLBACK,
        RUNTIME_ASSERT,
        CACHE_LOOKUP,
        CACHE_HIT,
        CACHE_MISS,
        CACHE_INVALIDATE,
        PATTERN_CANDIDATE,
        PATTERN_HIT,
        PATTERN_REJECT,
        LOWERING_SELECT,
        KERNEL,
        RESOURCE_CREATE,
        RESOURCE_DESTROY,
        ERROR,
    )

    #: The five events that must never be merged into one number.
    FIVE_WAY = (GUARD_FAIL, RECOMPILE, GRAPH_BREAK, RUNTIME_ASSERT, FALLBACK)


@dataclass(frozen=True)
class CompileEvent:
    """One event with its request/graph context and reason."""

    kind: str
    request_index: int
    graph_variant: str = ""
    reason: str = ""
    timestamp_ns: int = 0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EventKind.ALL:
            raise SchemaError(
                f"unknown compile event kind {self.kind!r}",
                details={"field": "kind", "allowed": list(EventKind.ALL)},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "request_index": self.request_index,
            "graph_variant": self.graph_variant,
            "reason": self.reason,
            "timestamp_ns": self.timestamp_ns,
            "attributes": dict(self.attributes),
        }


@dataclass
class CompileLedger:
    """Per-request event stream with separated counters."""

    events: List[CompileEvent] = field(default_factory=list)
    _last_timestamp: int = 0

    def record(
        self,
        kind: str,
        request_index: int,
        *,
        graph_variant: str = "",
        reason: str = "",
        timestamp_ns: Optional[int] = None,
        **attributes: Any,
    ) -> CompileEvent:
        stamp = timestamp_ns if timestamp_ns is not None else self._last_timestamp + 1
        self._last_timestamp = max(self._last_timestamp, stamp)
        event = CompileEvent(
            kind=kind,
            request_index=request_index,
            graph_variant=graph_variant,
            reason=reason,
            timestamp_ns=stamp,
            attributes=attributes,
        )
        self.events.append(event)
        return event

    def counts(self) -> Dict[str, int]:
        counts = {kind: 0 for kind in EventKind.ALL}
        for event in self.events:
            counts[event.kind] += 1
        return counts

    def five_way(self) -> "FiveWayCounts":
        counts = self.counts()
        return FiveWayCounts(
            guard_failures=counts[EventKind.GUARD_FAIL],
            recompiles=counts[EventKind.RECOMPILE],
            graph_breaks=counts[EventKind.GRAPH_BREAK],
            runtime_asserts=counts[EventKind.RUNTIME_ASSERT],
            fallbacks=counts[EventKind.FALLBACK],
            requests=len({event.request_index for event in self.events}),
        )

    def reasons_for(self, kind: str) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            (event.graph_variant, event.reason)
            for event in self.events
            if event.kind == kind and event.reason
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "counts": self.counts(),
            "five_way": self.five_way().as_dict(),
            "events": [event.as_dict() for event in self.events],
        }

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(event.as_dict(), sort_keys=True) for event in self.events)


@dataclass(frozen=True)
class FiveWayCounts:
    """The five counts, never summed into a single "recompile" figure."""

    guard_failures: int
    recompiles: int
    graph_breaks: int
    runtime_asserts: int
    fallbacks: int
    requests: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "guard_failures": self.guard_failures,
            "recompiles": self.recompiles,
            "graph_breaks": self.graph_breaks,
            "runtime_asserts": self.runtime_asserts,
            "fallbacks": self.fallbacks,
            "requests": self.requests,
        }

    def recompiles_per_request(self) -> float:
        if self.requests <= 0:
            return 0.0
        return self.recompiles / self.requests


# ── dynamic shape policy ──────────────────────────────────────────────────


class DynamicPolicy:
    STATIC = "static"  # dynamic=False: full specialisation
    AUTO = "auto"  # the locked version's default/resolved behaviour
    DYNAMIC = "dynamic"  # dynamic=True: as dynamic as possible
    EXPLICIT_BOUNDS = "explicit_bounds"  # mark_dynamic / dim hints
    BUCKETED = "bucketed"  # finite B/S buckets with padding
    PHASE_SPLIT = "phase_split"  # prefill and decode compiled separately
    EAGER_FALLBACK = "eager_fallback"  # unsupported shapes stay eager

    ALL = (STATIC, AUTO, DYNAMIC, EXPLICIT_BOUNDS, BUCKETED, PHASE_SPLIT, EAGER_FALLBACK)


@dataclass(frozen=True)
class DynamicDimensionSpec:
    """One dynamic dimension with its frozen bounds and bucket policy."""

    name: str
    symbol: str
    bounds: Tuple[int, int]
    buckets: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        low, high = self.bounds
        if not 0 < low <= high:
            raise ConfigError(
                f"dimension {self.name!r}: bounds must satisfy 0 < low <= high, got {self.bounds}",
                details={"field": "bounds", "actual": list(self.bounds)},
            )
        for bucket in self.buckets:
            if not low <= bucket <= high:
                raise ConfigError(
                    f"dimension {self.name!r}: bucket {bucket} outside bounds {self.bounds}",
                    details={"field": "buckets"},
                )

    def bucket_for(self, value: int) -> int:
        """Smallest bucket >= value; out-of-range values are refused, not padded."""
        if value < self.bounds[0] or value > self.bounds[1]:
            raise ConfigError(
                f"dimension {self.name!r}: value {value} outside frozen bounds "
                f"{self.bounds}; route to fallback instead of padding silently",
                details={"field": "value", "actual": value},
            )
        for bucket in self.buckets:
            if value <= bucket:
                return bucket
        return value

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "bounds": list(self.bounds),
            "buckets": list(self.buckets),
        }


@dataclass(frozen=True)
class CompilerConfig:
    """Resolved compiler configuration (recorded, never assumed)."""

    policy: str
    dimensions: Tuple[DynamicDimensionSpec, ...] = ()
    dynamic_flag: Optional[bool] = None
    recompile_limit: Optional[int] = None
    mode: str = "default"
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.policy not in DynamicPolicy.ALL:
            raise ConfigError(
                f"unknown dynamic policy {self.policy!r}",
                details={"field": "policy", "allowed": list(DynamicPolicy.ALL)},
            )

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "policy": self.policy,
            "dynamic_flag": self.dynamic_flag,
            "recompile_limit": self.recompile_limit,
            "mode": self.mode,
            "dimensions": [dim.as_dict() for dim in self.dimensions],
            "notes": list(self.notes),
        }
        return payload


def resolve_compiler_config(
    policy: str,
    dimensions: Sequence[DynamicDimensionSpec] = (),
    *,
    recompile_limit: Optional[int] = None,
    mode: str = "default",
) -> CompilerConfig:
    """Map a policy onto an explicit flag set (the default is never assumed)."""
    dynamic_flag = {
        DynamicPolicy.STATIC: False,
        DynamicPolicy.DYNAMIC: True,
        DynamicPolicy.EXPLICIT_BOUNDS: None,  # expressed via mark_dynamic on the symbols
        DynamicPolicy.AUTO: None,  # locked-version default; recorded as unresolved
    }.get(policy, None)
    notes: List[str] = []
    if policy == DynamicPolicy.AUTO:
        notes.append("AUTO keeps the locked version's resolved default; record it from the run")
    if policy == DynamicPolicy.BUCKETED and not any(dim.buckets for dim in dimensions):
        raise ConfigError(
            "BUCKETED policy needs at least one dimension with buckets",
            details={"field": "buckets"},
        )
    return CompilerConfig(
        policy=policy,
        dimensions=tuple(dimensions),
        dynamic_flag=dynamic_flag,
        recompile_limit=recompile_limit,
        mode=mode,
        notes=tuple(notes),
    )


# ── ordered shape trace ───────────────────────────────────────────────────

#: The protocol's ordered trace (E06-05 §6).  Repeating an old shape inside the
#: sequence is what proves variant reuse; independent per-shape processes cannot.
TRACE_STEPS: Tuple[Tuple[str, int, int, str], ...] = (
    ("B1/S32", 1, 32, "tiny"),
    ("B1/S128", 1, 128, "short"),
    ("B1/S32", 1, 32, "tiny-repeat"),
    ("B2/S128", 2, 128, "batched"),
    ("B1/S512", 1, 512, "balanced"),
    ("decode M1 repeat", 1, 1, "decode"),
    ("noncontiguous", 1, 128, "stride-probe"),
    ("unsupported dtype", 1, 128, "dtype-probe"),
    ("B1/S128 restore", 1, 128, "restore"),
    ("long-prefill", 1, 2048, "long-prefill"),
)


@dataclass(frozen=True)
class ShapeTracePoint:
    """One request in the ordered trace."""

    index: int
    label: str
    batch: int
    sequence: int
    case: str
    stride_class: str = "contiguous"
    dtype: str = "float16"
    device: str = "cuda"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "batch": self.batch,
            "sequence": self.sequence,
            "case": self.case,
            "stride_class": self.stride_class,
            "dtype": self.dtype,
            "device": self.device,
        }


def ordered_shape_trace() -> Tuple[ShapeTracePoint, ...]:
    points = []
    for index, (label, batch, sequence, case) in enumerate(TRACE_STEPS):
        stride_class = "transposed" if case == "stride-probe" else "contiguous"
        dtype = "float64" if case == "dtype-probe" else "float16"
        points.append(
            ShapeTracePoint(
                index=index,
                label=label,
                batch=batch,
                sequence=sequence,
                case=case,
                stride_class=stride_class,
                dtype=dtype,
            )
        )
    return tuple(points)


@dataclass(frozen=True)
class TraceRequestRecord:
    """Per-request observation (the field set E06-05 §12 requires)."""

    point: ShapeTracePoint
    graph_variant: str = ""
    guard_hit: bool = False
    guard_failed: bool = False
    recompiled: bool = False
    graph_break: bool = False
    cache_hit: bool = False
    actual_path: str = ""
    latency_ms: float = 0.0
    correctness_ok: Optional[bool] = None
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request": self.point.as_dict(),
            "graph_variant": self.graph_variant,
            "guard_hit": self.guard_hit,
            "guard_failed": self.guard_failed,
            "recompiled": self.recompiled,
            "graph_break": self.graph_break,
            "cache_hit": self.cache_hit,
            "actual_path": self.actual_path,
            "latency_ms": self.latency_ms,
            "correctness_ok": self.correctness_ok,
            "reason": self.reason,
        }


def reuse_evidence(records: Sequence[TraceRequestRecord]) -> Tuple[Dict[str, Any], ...]:
    """Show, per repeated (B,S,stride,dtype) key, whether a variant was reused."""
    seen: Dict[Tuple[int, int, str, str], str] = {}
    evidence: List[Dict[str, Any]] = []
    for record in records:
        key = (
            record.point.batch,
            record.point.sequence,
            record.point.stride_class,
            record.point.dtype,
        )
        if key in seen:
            evidence.append(
                {
                    "key": list(key),
                    "first_variant": seen[key],
                    "this_variant": record.graph_variant,
                    "reused": seen[key] == record.graph_variant and not record.recompiled,
                    "request_index": record.point.index,
                }
            )
        else:
            seen[key] = record.graph_variant
    return tuple(evidence)


# ── storm thresholds ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class StormThresholds:
    """Pre-registered storm limits (E06-05 §7) — never tuned after the fact."""

    window_requests: int = 1000
    max_recompiles_per_window: int = 8
    max_unique_shape_graph_ratio: float = 0.02
    max_compile_wall_fraction: float = 0.05
    max_graph_variants: int = 32
    max_p95_amplification: float = 1.25
    max_cache_growth_entries: int = 64

    def __post_init__(self) -> None:
        if self.window_requests <= 0:
            raise ConfigError("window_requests must be positive", details={"field": "window_requests"})
        for name in (
            "max_unique_shape_graph_ratio",
            "max_compile_wall_fraction",
            "max_p95_amplification",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ConfigError(
                    f"{name} must be positive; a zero threshold cannot be met",
                    details={"field": name, "actual": value},
                )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "window_requests": self.window_requests,
            "max_recompiles_per_window": self.max_recompiles_per_window,
            "max_unique_shape_graph_ratio": self.max_unique_shape_graph_ratio,
            "max_compile_wall_fraction": self.max_compile_wall_fraction,
            "max_graph_variants": self.max_graph_variants,
            "max_p95_amplification": self.max_p95_amplification,
            "max_cache_growth_entries": self.max_cache_growth_entries,
        }


@dataclass(frozen=True)
class StormObservation:
    """The inputs a storm evaluation needs (all from recorded raw data)."""

    requests: int
    recompiles: int
    compile_wall_ms: float
    total_wall_ms: float
    unique_shapes: int
    graph_variants: int
    fallbacks: int
    baseline_p95_ms: float
    observed_p95_ms: float
    cache_entries_start: int
    cache_entries_end: int
    recompile_limit_reached: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requests": self.requests,
            "recompiles": self.recompiles,
            "compile_wall_ms": self.compile_wall_ms,
            "total_wall_ms": self.total_wall_ms,
            "unique_shapes": self.unique_shapes,
            "graph_variants": self.graph_variants,
            "fallbacks": self.fallbacks,
            "baseline_p95_ms": self.baseline_p95_ms,
            "observed_p95_ms": self.observed_p95_ms,
            "cache_entries_start": self.cache_entries_start,
            "cache_entries_end": self.cache_entries_end,
            "recompile_limit_reached": self.recompile_limit_reached,
        }


@dataclass(frozen=True)
class StormReport:
    """Measured values against the pre-registered thresholds (no verdict alone)."""

    observation: StormObservation
    thresholds: StormThresholds
    breaches: Tuple[Dict[str, Any], ...]

    @property
    def within_thresholds(self) -> bool:
        return not self.breaches

    @property
    def unique_shape_graph_ratio(self) -> float:
        if self.observation.requests <= 0:
            return 0.0
        return self.observation.unique_shapes / self.observation.requests

    @property
    def compile_wall_fraction(self) -> float:
        if self.observation.total_wall_ms <= 0:
            return 0.0
        return self.observation.compile_wall_ms / self.observation.total_wall_ms

    @property
    def p95_amplification(self) -> float:
        if self.observation.baseline_p95_ms <= 0:
            return 0.0
        return self.observation.observed_p95_ms / self.observation.baseline_p95_ms

    @property
    def cache_growth(self) -> int:
        return self.observation.cache_entries_end - self.observation.cache_entries_start

    def as_dict(self) -> Dict[str, Any]:
        return {
            "within_thresholds": self.within_thresholds,
            "breaches": [dict(item) for item in self.breaches],
            "unique_shape_graph_ratio": self.unique_shape_graph_ratio,
            "compile_wall_fraction": self.compile_wall_fraction,
            "p95_amplification": self.p95_amplification,
            "cache_growth": self.cache_growth,
            "observation": self.observation.as_dict(),
            "thresholds": self.thresholds.as_dict(),
        }


def evaluate_storm(thresholds: StormThresholds, observation: StormObservation) -> StormReport:
    """Compare measurement against frozen thresholds (a computation, not a verdict)."""
    breaches: List[Dict[str, Any]] = []
    window = thresholds.window_requests
    expected_recompiles = thresholds.max_recompiles_per_window
    if window and observation.requests:
        expected_recompiles = max(
            1, round(thresholds.max_recompiles_per_window * observation.requests / window)
        )
    if observation.recompiles > expected_recompiles:
        breaches.append(
            {
                "metric": "recompiles",
                "observed": observation.recompiles,
                "limit": expected_recompiles,
                "reason": "RECOMPILE_RATE_ABOVE_PREREGISTERED",
            }
        )
    if observation.requests > 0:
        ratio = observation.unique_shapes / observation.requests
        if ratio > thresholds.max_unique_shape_graph_ratio:
            breaches.append(
                {
                    "metric": "unique_shape_ratio",
                    "observed": ratio,
                    "limit": thresholds.max_unique_shape_graph_ratio,
                    "reason": "SHAPE_DIVERSITY_ABOVE_PREREGISTERED",
                }
            )
    if observation.total_wall_ms > 0:
        fraction = observation.compile_wall_ms / observation.total_wall_ms
        if fraction > thresholds.max_compile_wall_fraction:
            breaches.append(
                {
                    "metric": "compile_wall_fraction",
                    "observed": fraction,
                    "limit": thresholds.max_compile_wall_fraction,
                    "reason": "COMPILE_TIME_SHARE_ABOVE_PREREGISTERED",
                }
            )
    if observation.graph_variants > thresholds.max_graph_variants:
        breaches.append(
            {
                "metric": "graph_variants",
                "observed": observation.graph_variants,
                "limit": thresholds.max_graph_variants,
                "reason": "GRAPH_VARIANT_CAP_EXCEEDED",
            }
        )
    if (
        observation.baseline_p95_ms > 0
        and observation.observed_p95_ms / observation.baseline_p95_ms
        > thresholds.max_p95_amplification
    ):
        breaches.append(
            {
                "metric": "p95_amplification",
                "observed": observation.observed_p95_ms / observation.baseline_p95_ms,
                "limit": thresholds.max_p95_amplification,
                "reason": "TAIL_AMPLIFICATION_ABOVE_PREREGISTERED",
            }
        )
    growth = observation.cache_entries_end - observation.cache_entries_start
    if growth > thresholds.max_cache_growth_entries:
        breaches.append(
            {
                "metric": "cache_growth",
                "observed": growth,
                "limit": thresholds.max_cache_growth_entries,
                "reason": "CACHE_GROWTH_ABOVE_PREREGISTERED",
            }
        )
    if observation.recompile_limit_reached:
        breaches.append(
            {
                "metric": "recompile_limit",
                "observed": True,
                "limit": False,
                "reason": "FRAMEWORK_RECOMPILE_LIMIT_REACHED",
            }
        )
    return StormReport(
        observation=observation, thresholds=thresholds, breaches=tuple(breaches)
    )


# ── guard minimality / soundness ──────────────────────────────────────────


@dataclass(frozen=True)
class GuardReview:
    """Per-guard minimality review (E06-05 §8)."""

    guard: GuardRecord
    semantically_necessary: bool
    performance_only: bool = False
    from_unrelated_python_object: bool = False
    replaceable_by_symbolic: bool = False
    relaxed_and_verified: bool = False
    fallback_on_unsupported: bool = True
    reviewer: str = ""
    evidence: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "guard": self.guard.as_dict(),
            "semantically_necessary": self.semantically_necessary,
            "performance_only": self.performance_only,
            "from_unrelated_python_object": self.from_unrelated_python_object,
            "replaceable_by_symbolic": self.replaceable_by_symbolic,
            "relaxed_and_verified": self.relaxed_and_verified,
            "fallback_on_unsupported": self.fallback_on_unsupported,
            "reviewer": self.reviewer,
            "evidence": self.evidence,
        }


def audit_guard_minimality(reviews: Sequence[GuardReview]) -> Dict[str, Any]:
    """Refuse unsafe relaxation: dropping a semantic guard needs verification.

    A guard may only be reported as relaxed when the caller recorded boundary +
    fuzz verification *and* an unsupported-input fallback.  The audit never
    rewrites guards; it reports what is not justified yet.
    """
    problems: List[Dict[str, Any]] = []
    for review in reviews:
        guard = review.guard
        if review.relaxed_and_verified and guard.necessary and not review.evidence:
            problems.append(
                {
                    "guard": guard.expression,
                    "kind": guard.kind,
                    "reason": "RELAXED_WITHOUT_EVIDENCE",
                }
            )
        if review.relaxed_and_verified and guard.necessary and not review.fallback_on_unsupported:
            problems.append(
                {
                    "guard": guard.expression,
                    "kind": guard.kind,
                    "reason": "RELAXED_WITHOUT_FALLBACK",
                }
            )
        if not review.semantically_necessary and not review.performance_only:
            problems.append(
                {
                    "guard": guard.expression,
                    "kind": guard.kind,
                    "reason": "UNCLASSIFIED_GUARD",
                }
            )
    return {
        "ok": not problems,
        "guards": len(reviews),
        "problems": problems,
        "summary": {
            "necessary": sum(1 for review in reviews if review.semantically_necessary),
            "performance_only": sum(1 for review in reviews if review.performance_only),
            "unrelated_python": sum(
                1 for review in reviews if review.from_unrelated_python_object
            ),
        },
    }


def dynamic_correctness_points(bounds: Tuple[int, int], example: int) -> Tuple[int, ...]:
    """Lower/example/middle/upper/out-of-bounds points for one variant."""
    low, high = bounds
    middle = (low + high) // 2
    return (low, example, middle, high, high + 1)


__all__ = [
    "CompilerConfig",
    "CompileEvent",
    "CompileLedger",
    "DynamicDimensionSpec",
    "DynamicPolicy",
    "EventKind",
    "FiveWayCounts",
    "GuardKind",
    "GuardRecord",
    "GuardReview",
    "ShapeTracePoint",
    "StormObservation",
    "StormReport",
    "StormThresholds",
    "TRACE_STEPS",
    "TraceRequestRecord",
    "audit_guard_minimality",
    "dynamic_correctness_points",
    "evaluate_storm",
    "ordered_shape_trace",
    "resolve_compiler_config",
    "reuse_evidence",
]
