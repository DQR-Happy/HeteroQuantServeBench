"""Lifecycle state machine, resource accounting and leak statistics (E06-10).

Short benchmarks hide the failures this module is built to expose
(E06-10 §1): unbounded graph/cache growth, leaked workspaces, registration
objects collected while implementations still point at them, dangling CUDA Graph
buffers, wrapper↔model reference cycles, unreleased streams/events, resources
left behind on exception paths, and an old compiled graph still executing after
``disable``.

It also encodes the honest reading of "memory went up" (E06-10 §5): a bounded
cache plateau and a genuine leak are different verdicts, and a reserved-pool
high-water mark is not a leak.  Statistics are computed from recorded series;
nothing here runs a long-run experiment.
"""

from __future__ import annotations

import gc
import math
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

# ── lifecycle state machine ───────────────────────────────────────────────


class LifecycleState:
    UNINITIALIZED = "UNINITIALIZED"
    REGISTERED = "REGISTERED"
    MODEL_ATTACHED = "MODEL_ATTACHED"
    COMPILED = "COMPILED"
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    CLOSED = "CLOSED"
    COLLECTED = "COLLECTED"
    ABORTED = "ABORTED"

    ALL = (
        UNINITIALIZED,
        REGISTERED,
        MODEL_ATTACHED,
        COMPILED,
        ACTIVE,
        DISABLED,
        CLOSED,
        COLLECTED,
        ABORTED,
    )


#: Legal transitions.  Anything else is refused: "disable did not really
#: disable" (E06-10 §8) is a state-machine violation, not a warning.
LEGAL_TRANSITIONS: Dict[str, Tuple[str, ...]] = {
    LifecycleState.UNINITIALIZED: (LifecycleState.REGISTERED, LifecycleState.ABORTED),
    LifecycleState.REGISTERED: (LifecycleState.MODEL_ATTACHED, LifecycleState.ABORTED),
    LifecycleState.MODEL_ATTACHED: (
        LifecycleState.COMPILED,
        LifecycleState.ACTIVE,
        LifecycleState.ABORTED,
    ),
    LifecycleState.COMPILED: (
        LifecycleState.ACTIVE,
        LifecycleState.DISABLED,
        LifecycleState.CLOSED,
        LifecycleState.ABORTED,
    ),
    LifecycleState.ACTIVE: (
        LifecycleState.DISABLED,
        LifecycleState.CLOSED,
        LifecycleState.ABORTED,
    ),
    LifecycleState.DISABLED: (
        LifecycleState.ACTIVE,
        LifecycleState.MODEL_ATTACHED,
        LifecycleState.CLOSED,
        LifecycleState.ABORTED,
    ),
    LifecycleState.CLOSED: (LifecycleState.COLLECTED, LifecycleState.ABORTED),
    LifecycleState.COLLECTED: (),
    LifecycleState.ABORTED: (LifecycleState.CLOSED,),
}


@dataclass(frozen=True)
class Transition:
    """One state transition with its reason and (optional) span timing."""

    index: int
    source: str
    target: str
    reason: str = ""
    timestamp_ns: int = 0
    create_span: str = ""
    destroy_span: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "source": self.source,
            "target": self.target,
            "reason": self.reason,
            "timestamp_ns": self.timestamp_ns,
            "create_span": self.create_span,
            "destroy_span": self.destroy_span,
        }


@dataclass
class LifecycleMachine:
    """Tracks one object's lifecycle; illegal transitions are hard errors."""

    name: str
    state: str = LifecycleState.UNINITIALIZED
    transitions: List[Transition] = field(default_factory=list)
    live_handles: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state not in LifecycleState.ALL:
            raise ConfigError(
                f"{self.name}: unknown state {self.state!r}",
                details={"field": "state", "allowed": list(LifecycleState.ALL)},
            )

    def can(self, target: str) -> bool:
        return target in LEGAL_TRANSITIONS.get(self.state, ())

    def transition(
        self,
        target: str,
        *,
        reason: str = "",
        timestamp_ns: int = 0,
        create_span: str = "",
        destroy_span: str = "",
    ) -> Transition:
        if target not in LifecycleState.ALL:
            raise ConfigError(
                f"{self.name}: unknown target state {target!r}",
                details={"field": "state", "allowed": list(LifecycleState.ALL)},
            )
        if not self.can(target):
            raise ConfigError(
                f"{self.name}: illegal transition {self.state} -> {target}; "
                "the lifecycle is frozen and a skipped state is a defect",
                details={
                    "field": "state",
                    "from": self.state,
                    "to": target,
                    "allowed": list(LEGAL_TRANSITIONS.get(self.state, ())),
                },
            )
        transition = Transition(
            index=len(self.transitions),
            source=self.state,
            target=target,
            reason=reason,
            timestamp_ns=timestamp_ns,
            create_span=create_span,
            destroy_span=destroy_span,
        )
        self.transitions.append(transition)
        self.state = target
        return transition

    @property
    def path(self) -> Tuple[str, ...]:
        if not self.transitions:
            return (self.state,)
        return (self.transitions[0].source,) + tuple(item.target for item in self.transitions)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "path": list(self.path),
            "live_handles": list(self.live_handles),
            "transitions": [item.as_dict() for item in self.transitions],
        }


# ── resources ─────────────────────────────────────────────────────────────


class ResourceClass:
    PYTHON = "python"
    HOST_NATIVE = "host_native"
    DEVICE = "device"

    ALL = (PYTHON, HOST_NATIVE, DEVICE)


@dataclass(frozen=True)
class ResourceSpec:
    """One tracked resource with its owner, lifecycle and expected cap."""

    name: str
    resource_class: str
    owner: str
    create_span: str
    destroy_span: str
    cache_cap_entries: Optional[int] = None
    reclaim: str = "released"
    growth_threshold_per_cycle: float = 0.0

    def __post_init__(self) -> None:
        if self.resource_class not in ResourceClass.ALL:
            raise ConfigError(
                f"{self.name}: unknown resource class {self.resource_class!r}",
                details={"field": "resource_class", "allowed": list(ResourceClass.ALL)},
            )
        if not self.owner:
            raise ConfigError(
                f"{self.name}: a resource without an owner cannot be reclaimed",
                details={"field": "owner"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "resource_class": self.resource_class,
            "owner": self.owner,
            "create_span": self.create_span,
            "destroy_span": self.destroy_span,
            "cache_cap_entries": self.cache_cap_entries,
            "reclaim": self.reclaim,
            "growth_threshold_per_cycle": self.growth_threshold_per_cycle,
        }


@dataclass(frozen=True)
class ResourceSnapshot:
    """One measurement point of the long-run trace."""

    cycle: int
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    rss_bytes: int = 0
    pinned_bytes: int = 0
    fd_count: int = 0
    thread_count: int = 0
    python_objects: int = 0
    cache_entries: int = 0
    graph_variants: int = 0
    events: int = 0
    workspace_bytes: int = 0
    kv_bytes: int = 0
    mode: str = ""
    output_hash: str = ""
    stream: str = "default"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "rss_bytes": self.rss_bytes,
            "pinned_bytes": self.pinned_bytes,
            "fd_count": self.fd_count,
            "thread_count": self.thread_count,
            "python_objects": self.python_objects,
            "cache_entries": self.cache_entries,
            "graph_variants": self.graph_variants,
            "events": self.events,
            "workspace_bytes": self.workspace_bytes,
            "kv_bytes": self.kv_bytes,
            "mode": self.mode,
            "output_hash": self.output_hash,
            "stream": self.stream,
        }

    def metric(self, name: str) -> float:
        if name not in self.as_dict():
            raise ConfigError(
                f"unknown snapshot metric {name!r}",
                details={"field": "metric"},
            )
        return float(getattr(self, name))


def segment_snapshots(
    snapshots: Sequence[ResourceSnapshot], warmup_cycles: int, steady_cycles: int
) -> Dict[str, Tuple[ResourceSnapshot, ...]]:
    """Split the trace into warmup / steady / teardown (never one global fit)."""
    if warmup_cycles < 0 or steady_cycles <= 0:
        raise ConfigError(
            "warmup_cycles must be >= 0 and steady_cycles > 0",
            details={"field": "segments"},
        )
    ordered = sorted(snapshots, key=lambda item: item.cycle)
    warmup = tuple(ordered[:warmup_cycles])
    steady = tuple(ordered[warmup_cycles : warmup_cycles + steady_cycles])
    teardown = tuple(ordered[warmup_cycles + steady_cycles :])
    return {"warmup": warmup, "steady": steady, "teardown": teardown}


@dataclass(frozen=True)
class SlopeEstimate:
    """Robust slope with a confidence interval (no "looks flat" verdicts)."""

    metric: str
    slope_per_cycle: float
    ci_low: float
    ci_high: float
    cycles: int

    @property
    def crosses_zero(self) -> bool:
        return self.ci_low <= 0.0 <= self.ci_high

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "slope_per_cycle": self.slope_per_cycle,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "cycles": self.cycles,
            "crosses_zero": self.crosses_zero,
        }


def robust_slope(
    snapshots: Sequence[ResourceSnapshot], metric: str, *, z: float = 1.96
) -> SlopeEstimate:
    """Theil–Sen slope with a normal-approximation CI."""
    values = [(float(item.cycle), item.metric(metric)) for item in snapshots]
    if len(values) < 2:
        return SlopeEstimate(metric=metric, slope_per_cycle=0.0, ci_low=0.0, ci_high=0.0, cycles=len(values))
    slopes: List[float] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            dx = values[j][0] - values[i][0]
            if dx:
                slopes.append((values[j][1] - values[i][1]) / dx)
    slopes.sort()
    mid = len(slopes) // 2
    median = (
        slopes[mid]
        if len(slopes) % 2
        else (slopes[mid - 1] + slopes[mid]) / 2.0
    )
    spread = slopes[-1] - slopes[0]
    half_width = z * spread / max(math.sqrt(max(len(slopes), 1)), 1.0) if spread else 0.0
    return SlopeEstimate(
        metric=metric,
        slope_per_cycle=median,
        ci_low=median - half_width,
        ci_high=median + half_width,
        cycles=len(values),
    )


@dataclass(frozen=True)
class ChangePoint:
    cycle: int
    metric: str
    before: float
    after: float
    kind: str  # "plateau" | "jump" | "growth"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "metric": self.metric,
            "before": self.before,
            "after": self.after,
            "kind": self.kind,
        }


def detect_plateau(
    snapshots: Sequence[ResourceSnapshot], metric: str, *, tolerance: float = 0.0
) -> Optional[ChangePoint]:
    """Find the cycle after which the series stops changing beyond ``tolerance``.

    The candidate tail must contain at least two points: a single last sample is
    trivially "constant" and would mark *every* monotonically growing series as a
    plateau.
    """
    ordered = sorted(snapshots, key=lambda item: item.cycle)
    for index in range(1, len(ordered) - 1):
        tail = [item.metric(metric) for item in ordered[index:]]
        if len(tail) < 2:
            return None
        value = tail[0]
        if all(abs(item - value) <= tolerance for item in tail):
            return ChangePoint(
                cycle=ordered[index].cycle,
                metric=metric,
                before=ordered[index - 1].metric(metric),
                after=value,
                kind="plateau",
            )
    return None


class LeakVerdict:
    BOUNDED_CACHE = "BOUNDED_CACHE"
    STEADY = "STEADY"
    GROWING = "GROWING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

    ALL = (BOUNDED_CACHE, STEADY, GROWING, INSUFFICIENT_DATA)


@dataclass(frozen=True)
class LeakReport:
    """Verdict for one resource series, split by segment."""

    resource: str
    metric: str
    verdict: str
    slope: SlopeEstimate
    plateau: Optional[ChangePoint] = None
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "resource": self.resource,
            "metric": self.metric,
            "verdict": self.verdict,
            "slope": self.slope.as_dict(),
            "plateau": self.plateau.as_dict() if self.plateau else None,
            "detail": self.detail,
        }


def evaluate_leak(
    spec: ResourceSpec,
    snapshots: Sequence[ResourceSnapshot],
    *,
    metric: str = "allocated_bytes",
    warmup_cycles: int = 1,
    steady_cycles: int = 0,
) -> LeakReport:
    """Classify a series as bounded cache / steady / growing.

    A bounded cache reaching its declared cap is *not* a leak; a series whose
    steady slope CI excludes zero is.  Insufficient data is reported as such
    instead of guessed (E06-10 §5).
    """
    segments = segment_snapshots(snapshots, warmup_cycles, steady_cycles or max(len(snapshots) - 1, 1))
    steady = segments["steady"]
    if len(steady) < 3:
        return LeakReport(
            resource=spec.name,
            metric=metric,
            verdict=LeakVerdict.INSUFFICIENT_DATA,
            slope=robust_slope(steady, metric),
            detail="fewer than three steady-window points",
        )
    slope = robust_slope(steady, metric)
    plateau = detect_plateau(steady, metric, tolerance=spec.growth_threshold_per_cycle)
    if plateau is not None and spec.cache_cap_entries is not None:
        return LeakReport(
            resource=spec.name,
            metric=metric,
            verdict=LeakVerdict.BOUNDED_CACHE,
            slope=slope,
            plateau=plateau,
            detail="series plateaus within the declared cache cap",
        )
    if plateau is not None:
        return LeakReport(
            resource=spec.name,
            metric=metric,
            verdict=LeakVerdict.STEADY,
            slope=slope,
            plateau=plateau,
            detail="series plateaus",
        )
    if not slope.crosses_zero and slope.slope_per_cycle > spec.growth_threshold_per_cycle:
        return LeakReport(
            resource=spec.name,
            metric=metric,
            verdict=LeakVerdict.GROWING,
            slope=slope,
            detail="steady-window slope CI excludes zero and exceeds the threshold",
        )
    return LeakReport(
        resource=spec.name,
        metric=metric,
        verdict=LeakVerdict.STEADY,
        slope=slope,
        detail="steady-window slope is indistinguishable from zero",
    )


def frozen_resource_specs() -> Tuple[ResourceSpec, ...]:
    """The resource inventory S06 tracks (owners + reclaim semantics)."""
    return (
        ResourceSpec("model_module", ResourceClass.PYTHON, "model_adapter", "import", "close"),
        ResourceSpec(
            "compiled_callable",
            ResourceClass.PYTHON,
            "compile_engine",
            "compile",
            "close",
            cache_cap_entries=32,
            reclaim="cache_evictable",
        ),
        ResourceSpec(
            "fx_graph_cache",
            ResourceClass.PYTHON,
            "compile_engine",
            "capture",
            "close",
            cache_cap_entries=64,
            reclaim="bounded_cache",
        ),
        ResourceSpec("library_registration", ResourceClass.PYTHON, "operator_registry", "register", "close"),
        ResourceSpec(
            "extension_library",
            ResourceClass.HOST_NATIVE,
            "operator_registry",
            "load",
            "close",
            reclaim="process_lifetime",
        ),
        ResourceSpec("compiler_subprocess", ResourceClass.HOST_NATIVE, "compile_engine", "compile", "compile_end"),
        ResourceSpec(
            "generated_module",
            ResourceClass.DEVICE,
            "compile_engine",
            "codegen",
            "close",
            reclaim="driver_module_cache",
        ),
        ResourceSpec("workspace_buffer", ResourceClass.DEVICE, "lowering_runtime", "execute", "execute_end"),
        ResourceSpec(
            "cuda_graph_pool",
            ResourceClass.DEVICE,
            "cuda_graph_manager",
            "capture",
            "destroy",
            reclaim="graph_private_pool",
        ),
        ResourceSpec("static_buffer", ResourceClass.DEVICE, "cuda_graph_manager", "capture", "destroy"),
        ResourceSpec("event", ResourceClass.DEVICE, "stream_manager", "create", "destroy"),
        ResourceSpec("kv_cache", ResourceClass.DEVICE, "runtime", "allocate", "close"),
    )


# ── object liveness ───────────────────────────────────────────────────────


@dataclass
class LivenessProbe:
    """Weak references + finalizers; the probe never holds the observed object."""

    name: str
    _refs: Dict[str, weakref.ref] = field(default_factory=dict)
    _finalized: List[str] = field(default_factory=list)
    _gc_collected: List[str] = field(default_factory=list)

    def track(self, key: str, obj: Any) -> None:
        def _on_finalize(name: str = key) -> None:
            self._finalized.append(name)

        self._refs[key] = weakref.ref(obj, lambda _ref, name=key: _on_finalize(name))

    def alive(self) -> Tuple[str, ...]:
        return tuple(sorted(name for name, ref in self._refs.items() if ref() is not None))

    def dead(self) -> Tuple[str, ...]:
        return tuple(sorted(name for name, ref in self._refs.items() if ref() is None))

    def collect(self) -> Dict[str, Any]:
        gc.collect()
        return {
            "name": self.name,
            "alive": list(self.alive()),
            "dead": list(self.dead()),
            "finalized": sorted(set(self._finalized)),
            "collected_by_gc": len(self._gc_collected),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tracked": sorted(self._refs),
            "alive": list(self.alive()),
            "dead": list(self.dead()),
            "finalized": sorted(set(self._finalized)),
        }


# ── stream / synchronisation audit ────────────────────────────────────────


@dataclass(frozen=True)
class StreamEvent:
    """One stream-related observation (from a timeline or API trace)."""

    name: str
    stream: str = "default"
    kind: str = "event"  # "event" | "sync" | "wait" | "record"
    timestamp_ns: int = 0
    created: bool = False
    destroyed: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stream": self.stream,
            "kind": self.kind,
            "timestamp_ns": self.timestamp_ns,
            "created": self.created,
            "destroyed": self.destroyed,
        }


def audit_streams(events: Sequence[StreamEvent], expected_calls: int = 0) -> Dict[str, Any]:
    """Detect implicit global synchronisation and event accumulation."""
    syncs = [event for event in events if event.kind == "sync"]
    created = [event for event in events if event.created]
    destroyed = [event for event in events if event.destroyed]
    per_call_sync = len(syncs) > 1 or (expected_calls > 0 and len(syncs) >= expected_calls)
    unbalanced = len(created) - len(destroyed)
    return {
        "syncs": len(syncs),
        "events_created": len(created),
        "events_destroyed": len(destroyed),
        "unbalanced_events": unbalanced,
        "implicit_global_sync": per_call_sync,
        "streams": sorted({event.stream for event in events}),
        "ok": not per_call_sync and unbalanced <= 0,
    }


# ── enable/disable audit ──────────────────────────────────────────────────


@dataclass(frozen=True)
class EnableDisableAudit:
    """Per-switch check that disable really disabled (E06-10 §8)."""

    request_index: int
    mode: str
    actual_backend: str
    graph_id: str = ""
    state_dict_hash: str = ""
    parameter_ids: Tuple[str, ...] = ()
    hook_count: int = 0
    module_tree_digest: str = ""
    output_hash: str = ""
    restore_ok: bool = False
    old_callable_reachable: bool = False
    reusable_after_reenable: bool = False

    @property
    def ok(self) -> bool:
        if self.mode == "disabled":
            return (
                self.actual_backend in ("torch.eager.reference", "reference")
                and not self.old_callable_reachable
            )
        return self.restore_ok

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_index": self.request_index,
            "mode": self.mode,
            "actual_backend": self.actual_backend,
            "graph_id": self.graph_id,
            "state_dict_hash": self.state_dict_hash,
            "parameter_ids": list(self.parameter_ids),
            "hook_count": self.hook_count,
            "module_tree_digest": self.module_tree_digest,
            "output_hash": self.output_hash,
            "restore_ok": self.restore_ok,
            "old_callable_reachable": self.old_callable_reachable,
            "reusable_after_reenable": self.reusable_after_reenable,
            "ok": self.ok,
        }


def audit_enable_disable(records: Sequence[EnableDisableAudit]) -> Dict[str, Any]:
    failures = [item.as_dict() for item in records if not item.ok]
    return {
        "ok": not failures,
        "switches": len(records),
        "failures": failures,
        "hook_counts": sorted({item.hook_count for item in records}),
    }


# ── teardown ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TeardownPlan:
    """Ordered teardown; releasing in-flight resources is refused (E06-10 §17)."""

    stop_new_requests: bool = True
    wait_in_flight: bool = True
    close_backends: bool = True
    drop_references: bool = True
    run_gc: bool = True
    optional_trim: bool = False

    def validate(self, in_flight: int) -> Dict[str, Any]:
        problems: List[str] = []
        if in_flight and not self.wait_in_flight:
            problems.append("RELEASED_WHILE_IN_FLIGHT")
        if not self.stop_new_requests:
            problems.append("NEW_REQUESTS_STILL_ACCEPTED")
        return {"ok": not problems, "problems": problems, "in_flight": in_flight}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stop_new_requests": self.stop_new_requests,
            "wait_in_flight": self.wait_in_flight,
            "close_backends": self.close_backends,
            "drop_references": self.drop_references,
            "run_gc": self.run_gc,
            "optional_trim": self.optional_trim,
        }


@dataclass(frozen=True)
class PostCloseExpectation:
    """A closed callable must fail per contract, never use-after-free."""

    name: str
    expected_error: str
    observed_error: str = ""
    invoked: bool = False

    @property
    def ok(self) -> bool:
        return self.invoked and self.observed_error == self.expected_error

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expected_error": self.expected_error,
            "observed_error": self.observed_error,
            "invoked": self.invoked,
            "ok": self.ok,
        }


def liveness_factory(factory: Callable[[], Any]) -> LivenessProbe:
    """Helper: build an object, track it, drop the strong reference."""
    probe = LivenessProbe(name="artifact")
    obj = factory()
    probe.track("object", obj)
    del obj
    return probe


__all__ = [
    "ChangePoint",
    "EnableDisableAudit",
    "LEGAL_TRANSITIONS",
    "LeakReport",
    "LeakVerdict",
    "LifecycleMachine",
    "LifecycleState",
    "LivenessProbe",
    "PostCloseExpectation",
    "ResourceClass",
    "ResourceSnapshot",
    "ResourceSpec",
    "SlopeEstimate",
    "StreamEvent",
    "TeardownPlan",
    "Transition",
    "audit_enable_disable",
    "audit_streams",
    "detect_plateau",
    "evaluate_leak",
    "frozen_resource_specs",
    "liveness_factory",
    "robust_slope",
    "segment_snapshots",
]
