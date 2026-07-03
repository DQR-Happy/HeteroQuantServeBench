"""E13-05: startup/readiness/liveness, graceful drain, rolling restart.

Implements ``details/S13/E13-05_*.md`` as data:

* the three probe kinds separated in *semantics* (§2) — a single "process is
  alive" endpoint for all three is a FAIL condition, not a simplification;
* the termination time budget of §3 (the grace period starts before ``preStop``,
  so ``preStop`` + drain + exit share one budget);
* the declared request-completion semantics of §4 (streaming is not exactly-once,
  so the project must state its idempotency/resume policy and measure it);
* the drain timeline and the "no new work after the drain point" invariant;
* per-request token integrity, duplicate/retry accounting and resource release;
* rolling/PDB availability and the forced-kill path (a forced kill is a recorded
  SLO violation, never hidden behind client retries).

Nothing here starts or terminates a process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.infra import records as rec

EXPERIMENT_ID = "E13-05"
TITLE = "Startup/Readiness/Liveness、Graceful Drain 与滚动重启"
CLAIM_BOUNDARY = (
    "本实验通过证明生命周期符合声明；不证明容量模型、autoscaler、故障矩阵或 canary 判定本身正确"
    "（由 E13-06…E13-10 验证）。"
)

SCHEMA_VERSION = "1.0.0"

#: Probe semantics (§2): what each probe is allowed to decide.
PROBE_SEMANTICS: Mapping[str, str] = {
    "startup": "该进程是否仍在合法初始化，而不是卡死？",
    "readiness": "现在接一个新请求是否安全且有能力满足合同？",
    "liveness": "进程是否不可恢复，重启比继续运行更安全？",
}

#: Reasons readiness may be false *without* liveness being false (§2).
READINESS_ONLY_REASONS: Tuple[str, ...] = (
    "model_not_active",
    "quality_not_passed",
    "warmup_incomplete",
    "capacity_guard",
    "dependency_unavailable",
    "draining",
    "release_not_verified",
)

#: Termination budget stages (§3), in the order they happen.
TERMINATION_STAGES: Tuple[str, ...] = (
    "api_delete",
    "endpoint_not_ready_propagation",
    "prestop_trigger",
    "sigterm",
    "inflight_completion_or_cancellation",
    "telemetry_flush",
    "resource_release",
    "process_exit",
    "grace_expiry_sigkill",
)

#: Request completion semantics (from ``records``) that the run must declare.
DECLARED_COMPLETION_SEMANTICS: Tuple[str, ...] = rec.REQUEST_COMPLETION_SEMANTICS

#: Independent variables of the experiment (§7): the paths the run must cover.
COVERAGE_PATHS: Tuple[str, ...] = (
    "normal_startup",
    "slow_startup",
    "stuck_startup",
    "readiness_semantics",
    "liveness_semantics",
    "readiness_flapping",
    "short_inflight",
    "long_streaming_within_grace",
    "long_streaming_exceeding_grace",
    "slow_client_backpressure",
    "client_cancel",
    "prestop_failure_repeat",
    "sigterm_handler",
    "forced_sigkill",
    "rolling_same_version",
    "rolling_version_upgrade",
    "pdb_max_unavailable",
    "capacity_during_rollout",
    "failed_candidate_start",
    "progress_deadline",
)

#: Negative cases of steps 10–14 and 22–24.
NEGATIVE_CASES: Tuple[str, ...] = (
    "SLOW_STARTUP",
    "STUCK_STARTUP",
    "READINESS_FALSE_MODEL_INACTIVE",
    "READINESS_FALSE_QUALITY",
    "READINESS_FALSE_CAPACITY",
    "READINESS_FALSE_DEPENDENCY",
    "LIVENESS_DEADLOCK",
    "READINESS_FLAPPING",
    "PRESTOP_TIMEOUT",
    "PRESTOP_REPEATED",
    "FORCED_SIGKILL",
    "ROLLOUT_PROGRESS_DEADLINE",
    "CANDIDATE_START_FAILURE",
    "CONCURRENT_ROLLOUT",
)

#: Residual resources the post-termination check must account for (step 25).
RELEASE_TARGETS: Tuple[str, ...] = (
    "device_process",
    "device_context",
    "device_memory",
    "kv_cache",
    "cache_lease",
    "socket_connection",
    "pvc_lock",
    "endpoint",
    "queue_slot",
)


# ── probes ────────────────────────────────────────────────────────────────


@dataclass
class ProbeConfig:
    """Steps 1–2: per-kind semantics, thresholds and reason codes."""

    probe_config_id: str
    endpoints: Mapping[str, str] = field(default_factory=dict)
    period_s: Mapping[str, float] = field(default_factory=dict)
    timeout_s: Mapping[str, float] = field(default_factory=dict)
    success_threshold: Mapping[str, int] = field(default_factory=dict)
    failure_threshold: Mapping[str, int] = field(default_factory=dict)
    reason_codes: Tuple[str, ...] = ()
    liveness_signals: Tuple[str, ...] = ()
    startup_covers_stages: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.probe_config_id:
            problems.append("probe config needs an id")
        for kind in rec.PROBE_KINDS:
            if kind not in self.endpoints:
                problems.append(f"{kind} probe endpoint is not declared")
            if kind not in self.period_s or self.period_s.get(kind, 0) <= 0:
                problems.append(f"{kind} probe needs a positive period")
            if kind not in self.failure_threshold or self.failure_threshold.get(kind, 0) <= 0:
                problems.append(f"{kind} probe needs a failure threshold")
        endpoints = {value for value in self.endpoints.values() if value}
        if len(endpoints) == 1 and self.endpoints:
            problems.append(
                "startup/readiness/liveness must not all point at the same endpoint "
                "(§2: probe semantics are different questions)"
            )
        if not self.reason_codes:
            problems.append("readiness/liveness false reasons must be enumerable")
        for signal in self.liveness_signals:
            lowered = signal.lower()
            if any(token in lowered for token in ("queue", "load", "utilization", "saturation")):
                problems.append(
                    f"liveness signal {signal!r} is load-coupled: a busy service would be restarted "
                    "and the load pushed onto the remaining replicas"
                )
        missing_stages = [stage for stage in ("model_load", "warmup") if stage not in self.startup_covers_stages]
        if missing_stages:
            problems.append(
                f"the startup window must cover {missing_stages}: a large model load is normal, not a hang"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "probe_config_id": self.probe_config_id,
            "endpoints": dict(sorted(self.endpoints.items())),
            "period_s": dict(sorted(self.period_s.items())),
            "timeout_s": dict(sorted(self.timeout_s.items())),
            "success_threshold": dict(sorted(self.success_threshold.items())),
            "failure_threshold": dict(sorted(self.failure_threshold.items())),
            "reason_codes": list(self.reason_codes),
            "liveness_signals": list(self.liveness_signals),
            "startup_covers_stages": list(self.startup_covers_stages),
        }


@dataclass
class ProbeResult:
    """Step 8: one probe outcome (and its overhead, which is measured separately)."""

    episode_id: str
    probe_kind: str
    endpoint: str = ""
    result: str = "NOT_RUN"
    reason: str = ""
    latency_ms: float = 0.0
    at: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.probe_kind not in rec.PROBE_KINDS:
            problems.append(f"unknown probe kind {self.probe_kind!r}")
        if self.result not in ("success", "failure", "unknown", "NOT_RUN"):
            problems.append(f"unknown probe result {self.result!r}")
        if self.result == "failure" and not self.reason:
            problems.append("a failing probe must carry a reason code")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "probe_kind": self.probe_kind,
            "endpoint": self.endpoint,
            "result": self.result,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
        }


def probe_overhead(baseline: Mapping[str, float], with_probes: Mapping[str, float], metric: str) -> Dict[str, Any]:
    """Step 8: probes must not block the event loop or create an exec storm."""
    base = float(baseline.get(metric, 0.0))
    probed = float(with_probes.get(metric, 0.0))
    ratio = (probed / base) if base else float("inf")
    return {
        "metric": metric,
        "baseline": base,
        "with_probes": probed,
        "overhead_ratio": ratio,
        "exec_process_count": int(with_probes.get("probe_exec_processes", 0)),
        "note": "an HTTP probe is cheap; shell-backed probes create processes and must be counted",
    }


# ── termination budget and drain ─────────────────────────────────────────


@dataclass
class TerminationPolicy:
    """Steps 3–4: stop accepting, drain inflight, flush, release — inside one budget."""

    policy_id: str
    grace_period_s: float = 0.0
    prestop_s: float = 0.0
    stop_accepting_s: float = 0.0
    max_generation_s: float = 0.0
    slow_client_policy: str = ""
    telemetry_flush_s: float = 0.0
    artifact_lease_release: str = ""
    inflight_policy: str = ""
    retry_idempotency: str = ""
    completion_semantics: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("policy_id", "slow_client_policy", "artifact_lease_release", "inflight_policy",
                     "retry_idempotency"):
            if not getattr(self, name):
                problems.append(f"termination policy requires {name!r}")
        if self.grace_period_s <= 0:
            problems.append("termination grace period must be positive")
        if self.prestop_s >= self.grace_period_s:
            problems.append(
                "preStop consumes the grace budget: preStop + drain must fit inside grace_period_s"
            )
        if self.max_generation_s <= 0:
            problems.append("the maximum generation length inside grace must be declared")
        if self.max_generation_s + self.telemetry_flush_s > self.grace_period_s:
            problems.append(
                "max generation + telemetry flush exceed the grace budget: the pod would be SIGKILLed mid-flight"
            )
        if not self.completion_semantics:
            problems.append(
                "the project must declare its request completion semantics (streaming is not exactly-once)"
            )
        unknown = sorted(set(self.completion_semantics) - set(DECLARED_COMPLETION_SEMANTICS))
        if unknown:
            problems.append(f"unknown completion semantics: {unknown}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "grace_period_s": self.grace_period_s,
            "prestop_s": self.prestop_s,
            "stop_accepting_s": self.stop_accepting_s,
            "max_generation_s": self.max_generation_s,
            "slow_client_policy": self.slow_client_policy,
            "telemetry_flush_s": self.telemetry_flush_s,
            "artifact_lease_release": self.artifact_lease_release,
            "inflight_policy": self.inflight_policy,
            "retry_idempotency": self.retry_idempotency,
            "completion_semantics": list(self.completion_semantics),
        }


@dataclass
class DrainTimeline:
    """Steps 16–25: the ordered evidence of one termination episode."""

    episode_id: str
    delete_ts: str = ""
    endpoint_not_ready_ts: str = ""
    prestop_ts: str = ""
    sigterm_ts: str = ""
    not_ready_ts: str = ""
    stop_accepting_ts: str = ""
    inflight_done_ts: str = ""
    flush_ts: str = ""
    release_ts: str = ""
    exit_ts: str = ""
    forced_kill: bool = False
    policy_id: str = ""
    inflight_at_start: int = 0
    accepted_after_drain: int = 0
    completed: int = 0
    cancelled: int = 0
    timed_out: int = 0
    failed: int = 0
    partial: int = 0
    duplicate: int = 0

    def validate(self, policy: Optional[TerminationPolicy] = None) -> List[str]:
        problems: List[str] = []
        if not self.episode_id:
            problems.append("a drain timeline needs an episode id")
        ordered = [
            ("delete_ts", self.delete_ts),
            ("prestop_ts", self.prestop_ts),
            ("sigterm_ts", self.sigterm_ts),
            ("exit_ts", self.exit_ts),
        ]
        present = [(name, value) for name, value in ordered if value]
        for index in range(1, len(present)):
            if present[index][1] < present[index - 1][1]:
                problems.append(
                    f"termination order violated: {present[index][0]} precedes {present[index - 1][0]}"
                )
        for name in ("delete_ts", "sigterm_ts", "exit_ts"):
            if not getattr(self, name):
                problems.append(f"drain timeline requires {name!r}")
        if self.accepted_after_drain > 0:
            problems.append(
                f"{self.accepted_after_drain} requests started executing after the drain point"
            )
        if self.stop_accepting_ts and self.accepted_after_drain > 0:
            problems.append("stop-accepting and post-drain acceptance contradict each other")
        if not self.not_ready_ts:
            problems.append("the not-ready transition must be recorded (endpoint removal is not instant)")
        if policy is not None:
            problems.extend(policy.validate())
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "delete_ts": self.delete_ts,
            "prestop_ts": self.prestop_ts,
            "sigterm_ts": self.sigterm_ts,
            "not_ready_ts": self.not_ready_ts,
            "exit_ts": self.exit_ts,
            "forced_kill": self.forced_kill,
        }


# ── per-request integrity ────────────────────────────────────────────────


def transition_request(*, state_from: str, state_to: str) -> None:
    """The request state machine is enforced, not documented (§9.3)."""
    rec.REQUEST_STATE_MACHINE.assert_transition(state_from, state_to)


@dataclass
class TokenIntegrity:
    """Step 33: token prefix/full-sequence/version audit for one request."""

    request_id: str
    expected_tokens: Tuple[int, ...] = ()
    observed_tokens: Tuple[int, ...] = ()
    prefix_ok: bool = False
    status: str = ""
    model_version: str = ""
    final_statuses: Tuple[str, ...] = ()
    release_id: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.request_id:
            problems.append("token integrity needs a request id")
        if len(self.final_statuses) > 1 and len(set(self.final_statuses)) > 1:
            problems.append(
                f"request {self.request_id} has multiple different final statuses {list(self.final_statuses)}"
            )
        if self.status == "COMPLETED" and self.expected_tokens and self.observed_tokens != self.expected_tokens:
            problems.append(
                f"request {self.request_id}: completed tokens differ from the expected sequence"
            )
        if self.status == "COMPLETED" and not self.observed_tokens:
            problems.append(f"request {self.request_id}: a completed request must carry its token sequence")
        if self.status in ("CANCELLED", "TIMED_OUT", "FAILED") and not self.final_statuses:
            problems.append(f"request {self.request_id}: a non-completed request still needs a final status")
        if not self.model_version:
            problems.append(f"request {self.request_id}: the model version actually used must be recorded")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "expected_tokens": list(self.expected_tokens),
            "observed_tokens": list(self.observed_tokens),
            "prefix_ok": self.prefix_ok,
            "status": self.status,
            "model_version": self.model_version,
        }


def request_integrity_report(rows: Sequence[TokenIntegrity]) -> Dict[str, Any]:
    """Step 33: one final status per request; partial streams may not be called complete."""
    problems: List[str] = []
    for row in rows:
        problems.extend(row.validate())
    completed = [row for row in rows if row.status == "COMPLETED"]
    partial_completed = [row for row in rows if row.status == "COMPLETED" and not row.prefix_ok]
    if partial_completed:
        problems.append(
            f"{len(partial_completed)} completed requests have a broken token prefix "
            "(a truncated stream must not be reported as complete)"
        )
    return {
        "requests": len(rows),
        "completed": len(completed),
        "partial_completed": len(partial_completed),
        "problems": problems,
        "ok": not problems,
    }


def retry_duplicate_accounting(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 33: retries/duplicates are counted, not laundered into "no errors"."""
    problems: List[str] = []
    duplicates: List[Dict[str, Any]] = []
    post_token_retries: List[Dict[str, Any]] = []
    entries: List[Dict[str, Any]] = []
    for row in rows:
        entry = {
            "request_id": row.get("request_id", ""),
            "attempt_id": row.get("attempt_id", ""),
            "reason": row.get("reason", ""),
            "duplicate": bool(row.get("duplicate", False)),
            "retryable": bool(row.get("retryable", False)),
            "client_visible": bool(row.get("client_visible", False)),
            "before_first_token": bool(row.get("before_first_token", False)),
        }
        entries.append(entry)
        if entry["duplicate"]:
            duplicates.append(entry)
        if not entry["before_first_token"] and entry["retryable"]:
            post_token_retries.append(entry)
            problems.append(
                f"request {entry['request_id']}: retry declared after the first token "
                "(users may see duplicated tokens)"
            )
    return {
        "rows": [
            {
                "request_id": entry["request_id"],
                "attempt_id": entry["attempt_id"],
                "reason": entry["reason"],
                "duplicate": entry["duplicate"],
                "retryable": entry["retryable"],
                "client_visible": entry["client_visible"],
            }
            for entry in entries
        ],
        "duplicates": len(duplicates),
        "post_first_token_retries": len(post_token_retries),
        "problems": problems,
        "ok": not problems,
    }


# ── rolling restart / PDB / forced kill ─────────────────────────────────


@dataclass
class RollingPolicy:
    """Step 5: replicas, surge/unavailable, PDB, deadlines and rollback conditions."""

    policy_id: str
    replicas: int = 0
    max_surge: int = 0
    max_unavailable: int = 0
    pod_disruption_budget: int = 0
    min_ready_seconds: float = 0.0
    progress_deadline_s: float = 0.0
    termination_grace_s: float = 0.0
    rollback_conditions: Tuple[str, ...] = ()
    serialize_with_autoscaler: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.policy_id:
            problems.append("rolling policy needs an id")
        if self.replicas <= 0:
            problems.append("replicas must be positive")
        if self.max_unavailable >= self.replicas:
            problems.append("maxUnavailable must be smaller than the replica count (capacity would go to zero)")
        if self.pod_disruption_budget <= 0:
            problems.append("a PodDisruptionBudget is required for voluntary disruptions")
        if self.progress_deadline_s <= 0:
            problems.append("a progress deadline is required (a stuck rollout must stop, not hang)")
        if not self.rollback_conditions:
            problems.append("rollback conditions must be pre-registered")
        if not self.serialize_with_autoscaler:
            problems.append(
                "rollout and autoscaler must be serialized: two controllers deleting replicas at once "
                "breaks the availability budget"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "replicas": self.replicas,
            "max_surge": self.max_surge,
            "max_unavailable": self.max_unavailable,
            "pod_disruption_budget": self.pod_disruption_budget,
            "min_ready_seconds": self.min_ready_seconds,
            "progress_deadline_s": self.progress_deadline_s,
            "termination_grace_s": self.termination_grace_s,
            "rollback_conditions": list(self.rollback_conditions),
            "serialize_with_autoscaler": self.serialize_with_autoscaler,
        }


def rolling_availability(
    series: Sequence[Mapping[str, Any]], *, policy: RollingPolicy, slo_required: bool = True
) -> Dict[str, Any]:
    """Steps 26–29: available capacity and SLO during the rollout (not just at the end)."""
    problems = list(policy.validate())
    rows: List[Dict[str, Any]] = []
    below_budget: List[Dict[str, Any]] = []
    for sample in series:
        available = int(sample.get("available_replicas", 0))
        entry = {
            "episode_id": sample.get("episode_id", ""),
            "available_replicas": available,
            "ready_replicas": int(sample.get("ready_replicas", 0)),
            "surge": int(sample.get("surge", 0)),
            "unavailable": int(sample.get("unavailable", 0)),
            "slo_ok": bool(sample.get("slo_ok", True)),
        }
        rows.append(entry)
        if available < policy.pod_disruption_budget:
            below_budget.append(entry)
        if slo_required and not entry["slo_ok"]:
            problems.append(
                f"episode {entry['episode_id']}: the rollout exceeded the SLO budget "
                "(control-plane completion is not production acceptance)"
            )
    if below_budget:
        problems.append(
            f"available replicas fell below the PDB threshold in {len(below_budget)} sample(s)"
        )
    return {
        "rows": rows,
        "samples": len(rows),
        "below_budget": below_budget,
        "problems": problems,
        "ok": not problems,
    }


def validate_progress_deadline(
    *, rollout_started: str, stuck_candidate: bool, deadline_s: float, action_taken: str
) -> Dict[str, Any]:
    """Step 31: a stuck candidate must be stopped/rolled back, not left half-deployed."""
    problems: List[str] = []
    if stuck_candidate and action_taken not in ("STOPPED", "ROLLED_BACK"):
        problems.append(
            "a candidate that exceeds the progress deadline must be stopped or rolled back, "
            f"got {action_taken or 'nothing'}"
        )
    if not stuck_candidate and action_taken == "CONTINUED":
        problems.append("'CONTINUED' is only valid when the candidate is not stuck")
    return {
        "rollout_started": rollout_started,
        "deadline_s": deadline_s,
        "action_taken": action_taken,
        "ok": not problems,
        "problems": problems,
    }


def forced_kill_accounting(
    *, timeline: DrainTimeline, lost_requests: int, partial_streams: int, retryable: int, slo_violated: bool
) -> Dict[str, Any]:
    """Step 24 / §14: a forced path has explicit violations — it is not a graceful pass."""
    problems: List[str] = []
    if not timeline.forced_kill:
        problems.append("forced-kill accounting requires a timeline marked forced_kill")
    if not slo_violated and (lost_requests or partial_streams):
        problems.append(
            "lost/partial requests must be reported as an SLO violation, not absorbed by client retries"
        )
    return {
        "episode_id": timeline.episode_id,
        "lost_requests": lost_requests,
        "partial_streams": partial_streams,
        "retryable": retryable,
        "slo_violated": slo_violated,
        "problems": problems,
        "ok": not problems,
        "verdict": "GRACEFUL" if not timeline.forced_kill else "FORCED_WITH_DECLARED_VIOLATION",
        "note": "a forced kill may not be presented as a graceful-termination PASS",
    }


def resource_release_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 25: device/KV/lease/socket/PVC/endpoint must be gone after termination."""
    problems: List[str] = []
    observations: List[Dict[str, Any]] = []
    for row in rows:
        resource = str(row.get("resource", ""))
        if resource not in RELEASE_TARGETS:
            raise ConfigError(f"unknown release target {resource!r}")
        released = bool(row.get("released", False))
        observations.append(
            {
                "episode_id": row.get("episode_id", ""),
                "resource": resource,
                "released": released,
                "residual": bool(row.get("residual", not released)),
                "reason": row.get("reason", ""),
            }
        )
        if not released:
            problems.append(f"{resource} was not released after termination")
    missing = sorted(set(RELEASE_TARGETS) - {row["resource"] for row in observations})
    if missing:
        problems.append(f"no release observation for {missing}")
    return {"rows": observations, "problems": problems, "ok": not problems}


def run_negative_lifecycle_cases(
    cases: Sequence[Mapping[str, Any]], *, policy: TerminationPolicy
) -> Dict[str, Any]:
    """Steps 10–14/22–24/30–32: every injected lifecycle state must behave as declared."""
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        kind = str(case.get("kind", ""))
        if kind not in NEGATIVE_CASES:
            raise ConfigError(f"unknown lifecycle negative case {kind!r}")
        rows.append(
            {
                "case_id": str(case.get("case_id", f"life-neg-{index:03d}")),
                "kind": kind,
                "expected": str(case.get("expected", "DECLARED_BEHAVIOUR")),
                "observed": str(case.get("observed", "")),
                "restart_cascade": bool(case.get("restart_cascade", False)),
                "traffic_affected": bool(case.get("traffic_affected", False)),
                "ok": str(case.get("expected", "DECLARED_BEHAVIOUR")) == str(case.get("observed", ""))
                and not case.get("restart_cascade", False),
            }
        )
    failures = [row for row in rows if not row["ok"]]
    return {
        "rows": rows,
        "cases": len(rows),
        "failures": failures,
        "ok": not failures,
        "policy_id": policy.policy_id,
        "reason": (
            ""
            if not failures
            else "lifecycle negative cases deviated from the declared behaviour: "
            + ", ".join(row["kind"] for row in failures)
        ),
    }


def lifecycle_verdict(
    *,
    probes: Sequence[ProbeResult],
    drain: Mapping[str, Any],
    integrity: Mapping[str, Any],
    rolling: Mapping[str, Any],
    release: Mapping[str, Any],
    negatives: Mapping[str, Any],
) -> Dict[str, Any]:
    """Step 38: which lifecycle guarantees the stage may claim."""
    problems: List[str] = []
    for probe in probes:
        problems.extend(probe.validate())
    if not any(probe.probe_kind == "readiness" for probe in probes):
        problems.append("no readiness probe evidence")
    if not any(probe.probe_kind == "liveness" for probe in probes):
        problems.append("no liveness probe evidence")
    for axis in (drain, integrity, rolling, release, negatives):
        if not axis.get("ok"):
            problems.append("a lifecycle axis failed: " + ", ".join(axis.get("problems", []) or ["(no detail)"]))
    return {
        "problems": problems,
        "verdict": "PASSABLE_AT_CODE_LEVEL" if not problems else "BLOCKED",
        "note": "normal and forced paths are reported separately; forced kills are declared violations",
    }


# ── protocol steps and smoke self-check ──────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 lifecycle state machine", ("records:POD_STATE_MACHINE", "records:REQUEST_STATE_MACHINE")),
    (2, "冻结 probe semantics", ("lifecycle:ProbeConfig", "lifecycle:PROBE_SEMANTICS")),
    (3, "冻结 termination/drain policy", ("lifecycle:TerminationPolicy", "lifecycle:TERMINATION_STAGES")),
    (4, "冻结 retry/idempotency policy", ("lifecycle:DECLARED_COMPLETION_SEMANTICS", "lifecycle:TerminationPolicy")),
    (5, "冻结 rolling policy", ("lifecycle:RollingPolicy",)),
    (6, "部署基线 ReleaseBundle", ("deployment:render_manifests", "deployment:readiness_verdict")),
    (7, "部署端到端事件采集", ("observability:SemanticConvention", "telemetry:project_request_lifecycle_event")),
    (8, "测 probe 自身开销", ("lifecycle:probe_overhead", "lifecycle:ProbeResult")),
    (9, "验证正常 startup", ("lifecycle:ProbeResult", "deployment:DeploymentStageEvent")),
    (10, "注入慢启动", ("lifecycle:run_negative_lifecycle_cases",)),
    (11, "注入启动卡死", ("lifecycle:run_negative_lifecycle_cases",)),
    (12, "验证 readiness 语义", ("lifecycle:READINESS_ONLY_REASONS", "deployment:readiness_verdict")),
    (13, "验证 liveness 语义", ("lifecycle:run_negative_lifecycle_cases", "contracts:validate_liveness_policy")),
    (14, "测试 readiness flapping", ("lifecycle:readiness_flapping",)),
    (15, "建立稳定负载", ("serving:slo.SLOSpec", "lifecycle:TokenIntegrity")),
    (16, "触发单 Pod SIGTERM/delete", ("lifecycle:DrainTimeline",)),
    (17, "验证先停止接新请求", ("contracts:validate_drain_semantics", "lifecycle:DrainTimeline.validate")),
    (18, "验证短 inflight 完成", ("lifecycle:TokenIntegrity",)),
    (19, "验证长 streaming 请求", ("lifecycle:DrainTimeline", "lifecycle:TerminationPolicy.validate")),
    (20, "验证慢客户端/backpressure", ("lifecycle:slow_client_policy_check",)),
    (21, "验证客户端取消/断连", ("lifecycle:client_cancel_accounting",)),
    (22, "验证 preStop 失败/重复", ("lifecycle:run_negative_lifecycle_cases",)),
    (23, "验证 SIGTERM handler", ("lifecycle:DrainTimeline.validate",)),
    (24, "验证 forced SIGKILL", ("lifecycle:forced_kill_accounting",)),
    (25, "检查终止后资源", ("lifecycle:resource_release_report", "lifecycle:RELEASE_TARGETS")),
    (26, "运行同版本 rolling restart", ("lifecycle:rolling_availability",)),
    (27, "运行版本 rolling upgrade", ("lifecycle:rolling_availability", "artifacts:activate_version")),
    (28, "验证 PDB/maxUnavailable", ("lifecycle:RollingPolicy.validate", "lifecycle:rolling_availability")),
    (29, "验证 capacity during rollout", ("lifecycle:rolling_availability", "capacity:AdmissionDecision")),
    (30, "注入新 Pod 启动失败", ("lifecycle:run_negative_lifecycle_cases",)),
    (31, "测试 rollout progress deadline", ("lifecycle:validate_progress_deadline",)),
    (32, "测试连续滚动/并发 drain 保护", ("lifecycle:run_negative_lifecycle_cases", "lifecycle:RollingPolicy")),
    (33, "计算请求完整性", ("lifecycle:request_integrity_report", "lifecycle:retry_duplicate_accounting")),
    (34, "计算生命周期指标", ("lifecycle:lifecycle_metrics",)),
    (35, "执行跨时段重复", ("lifecycle:lifecycle_metrics", "lifecycle:DIFF")),
    (36, "执行 probe/timeout 敏感性", ("lifecycle:probe_sensitivity",)),
    (37, "验证 runbook 自动性", ("deployment:ManualIntervention", "observability:AlertRule")),
    (38, "形成 lifecycle verdict", ("lifecycle:lifecycle_verdict",)),
)


def readiness_flapping(
    failures: Sequence[Mapping[str, Any]], *, failure_threshold: int, hysteresis_s: float
) -> Dict[str, Any]:
    """Step 14: brief dependency/capacity jitter must not churn the endpoint set."""
    problems: List[str] = []
    transitions = 0
    previous: Optional[str] = None
    for sample in failures:
        state = "ready" if sample.get("ready") else "not_ready"
        if previous is not None and state != previous:
            transitions += 1
        previous = state
    if transitions > 1 and hysteresis_s <= 0:
        problems.append(
            f"{transitions} readiness transitions without hysteresis: endpoint churn amplifies tail latency"
        )
    if failure_threshold <= 0:
        problems.append("a failure threshold must be positive")
    return {"transitions": transitions, "failure_threshold": failure_threshold, "hysteresis_s": hysteresis_s,
            "ok": not problems, "problems": problems}


def slow_client_policy_check(*, policy: str, buffer_bytes: int, drain_s: float, budget_s: float) -> Dict[str, Any]:
    """Step 20: one slow consumer must not block the pod forever."""
    problems: List[str] = []
    if not policy:
        problems.append("no slow-client policy declared")
    if buffer_bytes <= 0:
        problems.append("the stream buffer size must be bounded and declared")
    if drain_s > budget_s:
        problems.append(
            f"the slow client held the pod for {drain_s}s beyond the drain budget {budget_s}s"
        )
    return {"policy": policy, "buffer_bytes": buffer_bytes, "drain_s": drain_s, "budget_s": budget_s,
            "ok": not problems, "problems": problems}


def client_cancel_accounting(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 21: cancelling at any stage must stop work and release KV/slot."""
    problems: List[str] = []
    for row in rows:
        stage = str(row.get("stage", ""))
        if stage not in ("queued", "prefill", "decode", "stream"):
            problems.append(f"unknown cancel stage {stage!r}")
        work_stopped = bool(row.get("work_stopped", False))
        bounded_completion = bool(row.get("bounded_completion", False))
        released = bool(row.get("kv_released", False)) and bool(row.get("slot_released", False))
        if not (work_stopped or bounded_completion):
            problems.append(f"cancel at {stage}: device work continued unbounded")
        if not released:
            problems.append(f"cancel at {stage}: KV/slot accounting not released")
    return {"rows": list(rows), "ok": not problems, "problems": problems}


def lifecycle_metrics(drain: Sequence[DrainTimeline]) -> Dict[str, Any]:
    """Step 34: drain/termination distribution and the forced-kill rate."""
    forced = [episode for episode in drain if episode.forced_kill]
    return {
        "episodes": len(drain),
        "forced_kill": len(forced),
        "forced_kill_rate": (len(forced) / len(drain)) if drain else 0.0,
        "completed": sum(episode.completed for episode in drain),
        "cancelled": sum(episode.cancelled for episode in drain),
        "partial": sum(episode.partial for episode in drain),
        "duplicate": sum(episode.duplicate for episode in drain),
        "note": "report distributions over episodes; a single lucky request is not a lifecycle result",
    }


def probe_sensitivity(rows: Sequence[Mapping[str, Any]], *, registered: Sequence[float]) -> Dict[str, Any]:
    """Step 36: the threshold/grace sweep is pre-registered, not chosen after the fact."""
    if not registered:
        raise ConfigError("the probe/grace sweep must be pre-registered")
    values = sorted(float(row.get("grace_s", 0.0)) for row in rows)
    return {
        "registered_values": sorted(float(value) for value in registered),
        "observed_values": values,
        "unregistered": sorted(set(values) - {float(value) for value in registered}),
        "note": "only pre-registered settings may be used to pick the final recommendation",
    }


# ``lifecycle:DIFF`` and ``artifacts:activate_version`` are named here as the
# documented cross-module entry points of steps 27/35.
DIFF = "repeatability-by-episode"


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the lifecycle contracts (smoke, not an experiment)."""
    checks: Dict[str, Any] = {}
    config = ProbeConfig(
        probe_config_id="p1",
        endpoints={"startup": "/startupz", "readiness": "/readyz", "liveness": "/livez"},
        period_s={"startup": 5.0, "readiness": 2.0, "liveness": 5.0},
        timeout_s={"startup": 1.0, "readiness": 1.0, "liveness": 1.0},
        success_threshold={"startup": 1, "readiness": 1, "liveness": 1},
        failure_threshold={"startup": 60, "readiness": 3, "liveness": 3},
        reason_codes=READINESS_ONLY_REASONS,
        liveness_signals=("event_loop_progress",),
        startup_covers_stages=("model_load", "warmup"),
    )
    checks["probe_config_valid"] = config.validate() == []

    shared = ProbeConfig(
        probe_config_id="p2",
        endpoints={"startup": "/healthz", "readiness": "/healthz", "liveness": "/healthz"},
        period_s={"startup": 1.0, "readiness": 1.0, "liveness": 1.0},
        failure_threshold={"startup": 1, "readiness": 1, "liveness": 1},
        reason_codes=("x",),
        liveness_signals=("queue_depth",),
        startup_covers_stages=(),
    )
    checks["shared_endpoint_and_load_coupling_rejected"] = len(shared.validate()) >= 3

    policy = TerminationPolicy(
        policy_id="t1", grace_period_s=120.0, prestop_s=10.0, stop_accepting_s=1.0, max_generation_s=60.0,
        slow_client_policy="bounded-buffer+timeout", telemetry_flush_s=5.0,
        artifact_lease_release="after-inflight", inflight_policy="finish-or-cancel",
        retry_idempotency="before-first-token-only",
        completion_semantics=("COMPLETED_EXACT_VERSION", "CANCELLED_EXPLICITLY"),
    )
    checks["termination_policy_valid"] = policy.validate() == []

    over_budget = TerminationPolicy(
        policy_id="t2", grace_period_s=30.0, prestop_s=30.0, max_generation_s=60.0,
        slow_client_policy="x", telemetry_flush_s=5.0, artifact_lease_release="y",
        inflight_policy="z", retry_idempotency="w", completion_semantics=("NOPE",),
    )
    checks["over_budget_termination_rejected"] = len(over_budget.validate()) >= 3

    timeline = DrainTimeline(
        episode_id="ep1", delete_ts="2026-09-19T00:00:00Z", prestop_ts="2026-09-19T00:00:01Z",
        sigterm_ts="2026-09-19T00:00:11Z", not_ready_ts="2026-09-19T00:00:02Z",
        stop_accepting_ts="2026-09-19T00:00:02Z", exit_ts="2026-09-19T00:00:40Z", policy_id="t1",
    )
    checks["drain_timeline_valid"] = timeline.validate(policy) == []

    leaking = DrainTimeline(
        episode_id="ep2", delete_ts="2026-09-19T00:00:00Z", sigterm_ts="2026-09-19T00:00:05Z",
        exit_ts="2026-09-19T00:00:09Z", accepted_after_drain=2, policy_id="t1",
    )
    checks["post_drain_acceptance_rejected"] = any(
        "after the drain point" in problem for problem in leaking.validate(policy)
    )

    integrity = request_integrity_report(
        [TokenIntegrity(request_id="r1", expected_tokens=(1, 2, 3), observed_tokens=(1, 2),
                        prefix_ok=True, status="COMPLETED", model_version="v1",
                        final_statuses=("COMPLETED",))]
    )
    checks["truncated_stream_detected"] = integrity["ok"] is False
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "note": "接口自检；未启动/未终止任何进程",
    }
