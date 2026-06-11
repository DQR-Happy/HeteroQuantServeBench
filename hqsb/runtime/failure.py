"""Cancel / timeout / OOM / long-context failure matrix and recovery (E07-09).

E07-09 §2 states the point precisely: a failure is **not an error code**, it is a
transaction that may have executed halfway.  So this module models failures as
(state, injection point, expected action, expected invariants) tuples and checks
the *lifecycle* consequences:

* :func:`frozen_matrix` enumerates request-control, resource and runtime failure
  points; each case declares the expected action and what must not happen;
* :class:`CancelTimeline` records the five instants of §3 and computes the
  observation latency, the extra tokens (with the in-flight policy), the block
  release lag and the cleanup time;
* OOM handling reuses :func:`hqsb.runtime.kv.oom_action`: bounded, ordered and
  deterministic — an unbounded retry loop is not expressible;
* over-long context must be refused at the earliest knowable layer; a run that
  first allocates most of the KV pool and only then rejects is a failure
  (:func:`context_abuse_check`);
* :func:`resource_slope_report` separates warmup/steady and refuses to call an
  allocator reserve a leak, while :func:`leak_blocks_pass` blocks a PASS when a
  resource is still growing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.runtime import kv as kv_mod

# ── failure taxonomy (E07-09 §2) ───────────────────────────────────────────

REQUEST_CONTROL_FAILURES: Tuple[str, ...] = (
    "cancel_waiting",
    "cancel_admitted_before_prefill",
    "cancel_prefill_before",
    "cancel_prefill_mid",
    "cancel_chunk_boundary",
    "cancel_first_token_race",
    "cancel_during_decode",
    "cancel_final_token_race",
    "cancel_repeated",
    "timeout_with_cancel",
    "client_stops_consuming",
)

RESOURCE_FAILURES: Tuple[str, ...] = (
    "admission_capacity_shortfall",
    "kv_allocation_oom",
    "workspace_oom",
    "graph_pool_oom",
    "prefix_cache_pressure",
    "preemption_recompute",
    "context_beyond_model_max",
    "context_beyond_runtime_max",
    "context_beyond_kv_capacity",
    "max_sequences_exceeded",
    "max_tokens_exceeded",
    "host_memory_or_thread_failure",
)

RUNTIME_FAILURES: Tuple[str, ...] = (
    "compile_or_capture_failure",
    "attention_or_kernel_error",
    "invalid_artifact",
    "sampling_or_output_error",
    "worker_process_failure_isolated",
)

ALL_FAILURE_CASES: Tuple[str, ...] = (
    REQUEST_CONTROL_FAILURES + RESOURCE_FAILURES + RUNTIME_FAILURES
)

#: Written before the run; the report compares expected with actual.
EXPECTED_ACTIONS: Tuple[str, ...] = (
    "cancel_and_release",
    "reject_before_allocation",
    "evict_eligible_cache",
    "preempt_and_recompute",
    "reduce_batch",
    "deterministic_fallback",
    "fail_request",
    "complete_then_discard_output",
)

EXTRA_OUTPUT_POLICIES: Tuple[str, ...] = (
    "no_extra_token",
    "in_flight_allowed_discarded",
    "in_flight_allowed_emitted",
)


@dataclass(frozen=True)
class FailureCase:
    """One injection point with its expected action and invariants."""

    case_id: str
    category: str
    injection_point: str
    expected_action: str
    extra_output_policy: str
    must_be_bounded: bool = True
    deadline_ms: float = 0.0
    invariants: Tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.case_id not in ALL_FAILURE_CASES:
            raise ConfigError(
                f"unknown failure case {self.case_id!r}",
                details={"field": "case_id"},
            )
        if self.category not in ("request_control", "resource", "runtime"):
            raise ConfigError(f"unknown failure category {self.category!r}")
        if self.expected_action not in EXPECTED_ACTIONS:
            raise ConfigError(
                f"unknown expected action {self.expected_action!r}",
                details={"allowed": list(EXPECTED_ACTIONS)},
            )
        if self.extra_output_policy not in EXTRA_OUTPUT_POLICIES:
            raise ConfigError(
                f"unknown extra output policy {self.extra_output_policy!r}",
                details={"allowed": list(EXTRA_OUTPUT_POLICIES)},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "injection_point": self.injection_point,
            "expected_action": self.expected_action,
            "extra_output_policy": self.extra_output_policy,
            "must_be_bounded": self.must_be_bounded,
            "deadline_ms": self.deadline_ms,
            "invariants": list(self.invariants),
            "notes": self.notes,
        }


#: Invariants that hold for *every* failure case (E07-09 §4).
COMMON_INVARIANTS: Tuple[str, ...] = (
    "cancelled_or_timed_out_request_is_never_re_admitted",
    "no_token_emitted_beyond_the_contract",
    "block_owner_and_refcount_consistent_after_cleanup",
    "prefix_cache_retains_only_under_an_explicit_policy",
    "batch_token_budget_is_conserved",
    "other_requests_tokens_and_kv_unchanged",
    "request_state_and_rng_do_not_leak_across_requests",
    "failed_graph_or_buffer_is_not_reused",
    "runtime_serves_the_next_healthy_request",
    "close_finally_releases_everything",
)


def frozen_matrix() -> Tuple[FailureCase, ...]:
    """The complete case list, generated so no case can be silently dropped."""
    cases: List[FailureCase] = []

    def add(
        case_id: str,
        category: str,
        injection_point: str,
        expected_action: str,
        extra_output_policy: str = "no_extra_token",
        notes: str = "",
    ) -> None:
        cases.append(
            FailureCase(
                case_id=case_id,
                category=category,
                injection_point=injection_point,
                expected_action=expected_action,
                extra_output_policy=extra_output_policy,
                invariants=COMMON_INVARIANTS,
                notes=notes,
            )
        )

    add("cancel_waiting", "request_control", "before admission", "cancel_and_release")
    add(
        "cancel_admitted_before_prefill",
        "request_control",
        "after admission, before prefill",
        "cancel_and_release",
    )
    add("cancel_prefill_before", "request_control", "prefill start", "cancel_and_release")
    add(
        "cancel_prefill_mid",
        "request_control",
        "mid prefill",
        "cancel_and_release",
        extra_output_policy="in_flight_allowed_discarded",
    )
    add(
        "cancel_chunk_boundary",
        "request_control",
        "chunk boundary",
        "cancel_and_release",
        extra_output_policy="in_flight_allowed_discarded",
        notes="prompt tokens must not be duplicated or half-written",
    )
    add(
        "cancel_first_token_race",
        "request_control",
        "first token emission race",
        "cancel_and_release",
        extra_output_policy="in_flight_allowed_discarded",
    )
    add(
        "cancel_during_decode",
        "request_control",
        "decode iteration",
        "cancel_and_release",
        extra_output_policy="in_flight_allowed_discarded",
    )
    add(
        "cancel_final_token_race",
        "request_control",
        "final token race",
        "complete_then_discard_output",
        extra_output_policy="in_flight_allowed_discarded",
    )
    add("cancel_repeated", "request_control", "second cancel", "cancel_and_release")
    add(
        "timeout_with_cancel",
        "request_control",
        "timeout and cancel together",
        "cancel_and_release",
    )
    add(
        "client_stops_consuming",
        "request_control",
        "model-core stream consumption stops",
        "cancel_and_release",
    )
    add(
        "admission_capacity_shortfall",
        "resource",
        "before allocation",
        "reject_before_allocation",
    )
    add("kv_allocation_oom", "resource", "block allocation", "reduce_batch")
    add("workspace_oom", "resource", "model runner workspace", "reduce_batch")
    add("graph_pool_oom", "resource", "graph pool", "deterministic_fallback")
    add(
        "prefix_cache_pressure",
        "resource",
        "cache retention",
        "evict_eligible_cache",
    )
    add("preemption_recompute", "resource", "KV pressure", "preempt_and_recompute")
    add(
        "context_beyond_model_max",
        "resource",
        "request validation",
        "reject_before_allocation",
    )
    add(
        "context_beyond_runtime_max",
        "resource",
        "runtime admission",
        "reject_before_allocation",
    )
    add(
        "context_beyond_kv_capacity",
        "resource",
        "admission with KV accounting",
        "reject_before_allocation",
    )
    add("max_sequences_exceeded", "resource", "admission", "reject_before_allocation")
    add("max_tokens_exceeded", "resource", "scheduler", "reject_before_allocation")
    add(
        "host_memory_or_thread_failure",
        "resource",
        "host resource",
        "fail_request",
    )
    add(
        "compile_or_capture_failure",
        "runtime",
        "compile/capture",
        "deterministic_fallback",
    )
    add("attention_or_kernel_error", "runtime", "kernel execution", "fail_request")
    add("invalid_artifact", "runtime", "load", "reject_before_allocation")
    add("sampling_or_output_error", "runtime", "sampling", "fail_request")
    add(
        "worker_process_failure_isolated",
        "runtime",
        "isolated worker process",
        "fail_request",
        notes="only inside a safe isolation fixture; never in the timing run",
    )
    covered = {case.case_id for case in cases}
    missing = set(ALL_FAILURE_CASES) - covered
    if missing:
        raise ConfigError(
            "the frozen failure matrix is incomplete: " + ", ".join(sorted(missing)),
            details={"fields": sorted(missing)},
        )
    return tuple(cases)


# ── cancel timeline ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CancelTimeline:
    """The five instants of E07-09 §3 plus the derived lag metrics."""

    request_id: str
    t_cancel_requested_ns: int
    t_cancel_seen_scheduler_ns: int
    t_last_kernel_for_request_ns: int
    t_last_token_ns: int
    t_blocks_released_or_cached_ns: int
    t_cleanup_done_ns: int
    extra_tokens_emitted: int = 0
    extra_output_policy: str = "no_extra_token"
    blocks_released: int = 0
    blocks_retained_as_cache: int = 0

    def __post_init__(self) -> None:
        stamps = [
            self.t_cancel_requested_ns,
            self.t_cancel_seen_scheduler_ns,
            self.t_last_kernel_for_request_ns,
            self.t_last_token_ns,
            self.t_blocks_released_or_cached_ns,
            self.t_cleanup_done_ns,
        ]
        if stamps != sorted(stamps):
            raise ConfigError(
                "cancel instants are not monotonic; cleanup cannot precede the "
                "cancel request",
                details={"field": "timeline"},
            )
        if self.extra_output_policy not in EXTRA_OUTPUT_POLICIES:
            raise ConfigError(f"unknown extra output policy {self.extra_output_policy!r}")
        if (
            self.extra_output_policy == "no_extra_token"
            and self.extra_tokens_emitted != 0
        ):
            raise ConfigError(
                "the contract says no extra token, but tokens were emitted after the "
                "cancel: this is a contract violation, not a measurement",
                details={"field": "extra_tokens_emitted"},
            )

    @property
    def observation_latency_ms(self) -> float:
        return (self.t_cancel_seen_scheduler_ns - self.t_cancel_requested_ns) / 1e6

    @property
    def block_release_lag_ms(self) -> float:
        return (self.t_blocks_released_or_cached_ns - self.t_cancel_seen_scheduler_ns) / 1e6

    @property
    def cleanup_time_ms(self) -> float:
        return (self.t_cleanup_done_ns - self.t_cancel_requested_ns) / 1e6

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "observation_latency_ms": self.observation_latency_ms,
            "block_release_lag_ms": self.block_release_lag_ms,
            "cleanup_time_ms": self.cleanup_time_ms,
            "extra_tokens_emitted": self.extra_tokens_emitted,
            "extra_output_policy": self.extra_output_policy,
            "blocks_released": self.blocks_released,
            "blocks_retained_as_cache": self.blocks_retained_as_cache,
            "allowed_extra_tokens": (
                0 if self.extra_output_policy == "no_extra_token" else None
            ),
        }


# ── OOM / context policy ───────────────────────────────────────────────────


#: Actions that end the OOM transaction (no further attempt is possible).
TERMINAL_OOM_ACTIONS: Tuple[str, ...] = ("reject", "fail")


def oom_sequence(kind: str, *, max_attempts: int = kv_mod.MAX_OOM_ATTEMPTS) -> List[str]:
    """The bounded action sequence an OOM handler is allowed to take.

    The sequence always ends in a terminal action (``reject`` or ``fail``): a
    policy that can run out of attempts without deciding is exactly the unbounded
    retry the protocol forbids.
    """
    if max_attempts > kv_mod.MAX_OOM_ATTEMPTS:
        raise ConfigError(
            f"max_attempts may not exceed {kv_mod.MAX_OOM_ATTEMPTS}; a larger budget "
            "reintroduces unbounded retry",
            details={"field": "max_attempts"},
        )
    actions: List[str] = []
    for attempt in range(max_attempts):
        action = kv_mod.oom_action(kind, attempt)
        actions.append(action)
        if action in TERMINAL_OOM_ACTIONS:
            break
    if actions and actions[-1] not in TERMINAL_OOM_ACTIONS:
        raise ConfigError(
            f"the OOM policy for {kind!r} ran out of attempts without a terminal "
            "action; the failure must be bounded and decided",
            details={"field": "kind"},
        )
    return actions


@dataclass(frozen=True)
class ContextAbuseCheck:
    """A long request must be refused early, not after allocating the pool."""

    requested_tokens: int
    rejected_at_layer: str
    tokens_allocated_before_reject: int
    kv_capacity_tokens: int
    allowed_allocation_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.allowed_allocation_fraction <= 0:
            raise ConfigError("allowed_allocation_fraction must be positive")

    @property
    def ok(self) -> bool:
        if not self.rejected_at_layer:
            return False
        budget = self.kv_capacity_tokens * self.allowed_allocation_fraction
        return self.tokens_allocated_before_reject <= budget

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested_tokens": self.requested_tokens,
            "rejected_at_layer": self.rejected_at_layer,
            "tokens_allocated_before_reject": self.tokens_allocated_before_reject,
            "kv_capacity_tokens": self.kv_capacity_tokens,
            "allowed_allocation_fraction": self.allowed_allocation_fraction,
            "ok": self.ok,
            "note": (
                "a request that allocates most of the KV pool before discovering it "
                "can never fit is an admission-policy defect (E07-09 §6)"
            ),
        }


def context_limit_layers(
    *, requested_tokens: int, model_max: int, runtime_max: int, kv_capacity_tokens: int
) -> Dict[str, Any]:
    check = kv_mod.ContextLimitCheck(
        requested_tokens=requested_tokens,
        model_max=model_max,
        runtime_max=runtime_max,
        kv_capacity_tokens=kv_capacity_tokens,
    )
    return check.as_dict()


# ── concurrency / lifecycle scenarios ──────────────────────────────────────


def concurrency_release_scenarios() -> Tuple[Dict[str, str], ...]:
    """Race cases that must not deadlock or leave a stale reference (§7)."""
    return (
        {
            "case": "one_request_cancelled_in_batch",
            "check": "other requests keep their tokens and KV",
        },
        {
            "case": "shared_prefix_one_cancelled",
            "check": "refcount drops by one; the other reader keeps valid KV",
        },
        {
            "case": "reader_active_during_eviction",
            "check": "eviction refuses a block with refcount > 0",
        },
        {
            "case": "close_with_in_flight_request",
            "check": "in-flight work finishes or is cancelled before release",
        },
        {
            "case": "batch_timeout_storm",
            "check": "no lock convoy; cleanup completes within the deadline",
        },
        {
            "case": "oom_triggers_preemption",
            "check": "preempted tokens are accounted and recomputed exactly once",
        },
    )


def load_close_scenarios() -> Tuple[Dict[str, str], ...]:
    """Repeated load/close with success, failure and double close (§15)."""
    return (
        {
            "case": "load_success_close",
            "expectation": "resources released; handle unusable afterwards",
        },
        {
            "case": "load_failure_close",
            "expectation": "failure is diagnosed; close is still safe",
        },
        {
            "case": "double_close",
            "expectation": "second close is idempotent",
        },
        {
            "case": "load_close_loop",
            "expectation": "no growth in threads/handles/device memory",
        },
    )


# ── resource long-run statistics ───────────────────────────────────────────


@dataclass(frozen=True)
class SegmentSlopes:
    """Warmup and steady separated; a growing steady segment blocks a PASS."""

    warmup: Tuple[kv_mod.ResourceSlope, ...]
    steady: Tuple[kv_mod.ResourceSlope, ...]

    @property
    def growing(self) -> Tuple[str, ...]:
        return tuple(
            slope.name for slope in self.steady if slope.verdict == "GROWING"
        )

    @property
    def ok(self) -> bool:
        return not self.growing

    def as_dict(self) -> Dict[str, Any]:
        return {
            "warmup": [slope.as_dict() for slope in self.warmup],
            "steady": [slope.as_dict() for slope in self.steady],
            "growing": list(self.growing),
            "ok": self.ok,
        }


def resource_slope_report(
    series: Mapping[str, Sequence[float]],
    *,
    warmup_cycles: int,
    tolerance: float,
) -> SegmentSlopes:
    """Fit the warmup and steady segments separately (E07-09 §18)."""
    if warmup_cycles < 1:
        raise ConfigError("warmup_cycles must be at least 1")
    warmup: List[kv_mod.ResourceSlope] = []
    steady: List[kv_mod.ResourceSlope] = []
    for name, values in sorted(series.items()):
        if len(values) <= warmup_cycles:
            raise ConfigError(
                f"resource series {name!r} has no steady segment ({len(values)} "
                f"samples, warmup={warmup_cycles})",
                details={"field": name},
            )
        if len(values[:warmup_cycles]) >= 3:
            warmup.append(
                kv_mod.resource_slope(f"{name}@warmup", values[:warmup_cycles], tolerance=tolerance)
            )
        steady.append(
            kv_mod.resource_slope(f"{name}@steady", values[warmup_cycles:], tolerance=tolerance)
        )
    return SegmentSlopes(warmup=tuple(warmup), steady=tuple(steady))


def leak_blocks_pass(slopes: SegmentSlopes) -> Dict[str, Any]:
    """A PASS is blocked while any steady resource is still growing."""
    return {
        "pass_allowed": slopes.ok,
        "growing": list(slopes.growing),
        "reason": (
            ""
            if slopes.ok
            else "steady resources still grow: " + ", ".join(slopes.growing)
        ),
    }


# ── run separation and outcome table ───────────────────────────────────────


def run_separation_plan() -> Dict[str, Any]:
    """Ordinary timing, profiler and sanitizer runs are never mixed (§19)."""
    return {
        "ordinary": "timing and resource series only; no profiler attached",
        "profiler": "representative cases only; timings from these runs are discarded",
        "sanitizer": "fault fixtures only; never used for latency or capacity claims",
        "rules": [
            "a profiler run may not contribute a latency number",
            "a sanitizer run may not contribute a capacity number",
            "each run records which mode it was in",
        ],
    }


@dataclass(frozen=True)
class FailureOutcome:
    """Observed behaviour of one injected failure case."""

    case_id: str
    expected_action: str
    observed_action: str
    bounded: bool
    extra_tokens_after_request: int
    cleanup_ms: float
    other_request_impact_ms: float
    resources_released: bool
    recovered: bool
    error: str = ""

    @property
    def action_matches(self) -> bool:
        return self.observed_action == self.expected_action

    @property
    def ok(self) -> bool:
        return (
            self.action_matches
            and self.bounded
            and self.extra_tokens_after_request == 0
            and self.resources_released
            and self.recovered
            and not self.error
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "expected_action": self.expected_action,
            "observed_action": self.observed_action,
            "action_matches": self.action_matches,
            "bounded": self.bounded,
            "extra_tokens_after_request": self.extra_tokens_after_request,
            "cleanup_ms": self.cleanup_ms,
            "other_request_impact_ms": self.other_request_impact_ms,
            "resources_released": self.resources_released,
            "recovered": self.recovered,
            "error": self.error,
            "ok": self.ok,
        }


def failure_matrix_table(outcomes: Sequence[FailureOutcome]) -> Dict[str, Any]:
    """Expected/actual table; every frozen case must appear exactly once."""
    expected = {case.case_id for case in frozen_matrix()}
    seen = [outcome.case_id for outcome in outcomes]
    duplicates = sorted({case for case in seen if seen.count(case) > 1})
    missing = sorted(expected - set(seen))
    failures = [outcome.case_id for outcome in outcomes if not outcome.ok]
    return {
        "ok": not missing and not duplicates and not failures,
        "cases": len(outcomes),
        "missing": missing,
        "duplicates": duplicates,
        "failures": failures,
        "rows": [outcome.as_dict() for outcome in outcomes],
    }


def healthy_request_probe(
    probe: Callable[[], bool], *, rounds: int = 3
) -> Dict[str, Any]:
    """After a failure, healthy requests must succeed for several rounds."""
    if rounds < 1:
        raise ConfigError("rounds must be at least 1")
    results: List[bool] = []
    for _ in range(rounds):
        try:
            results.append(bool(probe()))
        except Exception:  # noqa: BLE001 - a raising probe counts as a failure
            results.append(False)
    return {
        "ok": all(results),
        "rounds": rounds,
        "results": results,
        "note": "a single successful request is not recovery evidence",
    }


__all__ = [
    "ALL_FAILURE_CASES",
    "COMMON_INVARIANTS",
    "CancelTimeline",
    "ContextAbuseCheck",
    "EXPECTED_ACTIONS",
    "EXTRA_OUTPUT_POLICIES",
    "FailureCase",
    "FailureOutcome",
    "REQUEST_CONTROL_FAILURES",
    "RESOURCE_FAILURES",
    "RUNTIME_FAILURES",
    "SegmentSlopes",
    "concurrency_release_scenarios",
    "context_limit_layers",
    "failure_matrix_table",
    "frozen_matrix",
    "healthy_request_probe",
    "leak_blocks_pass",
    "load_close_scenarios",
    "oom_sequence",
    "resource_slope_report",
    "run_separation_plan",
]
