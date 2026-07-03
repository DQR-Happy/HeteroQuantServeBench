"""E13-07: SLO/queue/token multi-signal autoscaling and control stability.

Implements ``details/S13/E13-07_*.md`` as data:

* the control loop of §2 and its total response time
  (``T_metric + T_decision + T_schedule + T_pull + T_model + T_compile + T_warmup + T_ready``)
  — if a burst is shorter than ``T_response``, reactive scaling cannot protect it;
* the desired-replica model of §3 including ready-vs-Running capacity, mixed
  workload and metric age;
* the seven compared policies of §4 (``CPU_ONLY`` is a *negative* baseline, not a
  straw man: every policy gets the same tuning budget and the same traces);
* the stability metrics of §5 and the pre-registered definition of "no sustained
  oscillation";
* desired/current/ready replica truth: ``Pod Running`` is never capacity;
* fail-safe behaviour for missing/stale metrics, controller restarts,
  unschedulable device pools and slow artifacts;
* per-episode cost accounting (replica/device minutes) so a better SLO is never
  reported without its price.

Nothing here talks to HPA or a cluster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.infra import records as rec

EXPERIMENT_ID = "E13-07"
TITLE = "SLO/Queue/Token 多指标 Autoscaling、阶跃/突发/降载与控制稳定性"
CLAIM_BOUNDARY = (
    "本实验通过证明弹性控制在声明 workload 内稳定；不证明底层故障都能恢复，也不证明新 release 一定"
    "安全（由 E13-09/E13-10 验证）。"
)

SCHEMA_VERSION = "1.0.0"

#: Policies compared in §4 (``STATIC_S12_CAPACITY`` is the SLO/cost baseline).
POLICIES: Tuple[str, ...] = rec.AUTOSCALING_POLICIES

#: Metric signals a policy may use, with their units (Prometheus base units).
METRIC_UNITS: Mapping[str, str] = {
    "cpu_utilization": "ratio",
    "accelerator_utilization": "ratio",
    "queue_requests": "requests",
    "queue_tokens": "tokens",
    "oldest_queue_age_seconds": "seconds",
    "ttft_burn_rate": "ratio",
    "tpot_burn_rate": "ratio",
    "slo_goodput": "requests/second",
    "predicted_outstanding_work": "token-seconds",
    "ready_replicas": "replicas",
}

#: Signals that are a *leading* indicator for LLM serving (vs CPU saturation).
LEADING_SIGNALS: Tuple[str, ...] = (
    "queue_tokens",
    "queue_requests",
    "oldest_queue_age_seconds",
    "predicted_outstanding_work",
    "ttft_burn_rate",
)

#: Failure injections of steps 28–32.
FAILURE_CASES: Tuple[str, ...] = (
    "METRIC_MISSING",
    "METRIC_STALE",
    "ADAPTER_UNAVAILABLE",
    "CONTROLLER_RESTART",
    "LEADER_CHANGE",
    "UNSCHEDULABLE_DEVICE_SHORTAGE",
    "SLOW_ARTIFACT_STARTUP",
    "SLOW_MODEL_LOAD",
    "ROLLOUT_CONCURRENCY",
    "MAX_REPLICAS_REACHED",
)

#: Scale-down safety: pods that must not be chosen first.
SCALE_DOWN_PROTECTIONS: Tuple[str, ...] = (
    "longest_inflight_stream",
    "unique_cache_holder",
    "critical_topology_rank",
    "last_ready_replica",
    "candidate_canary_replica",
)

#: A sustained oscillation is defined *before* the run (never after).
OSCILLATION_DEFINITION = (
    "观察窗口内不出现持续极限环：动作频率低于阈值、幅度低于阈值、降载后能在 settling 窗口内收敛"
)

DEFAULT_MAX_ACTIONS_PER_HOUR = 12.0
DEFAULT_MAX_AMPLITUDE = 2.0
DEFAULT_SETTLING_S = 600.0


@dataclass
class MetricSample:
    """§11: a metric value with its timeline (event → export → scrape → query)."""

    sample_id: str
    metric_name: str
    value: float = 0.0
    event_ts: float = 0.0
    export_ts: float = 0.0
    scrape_ts: float = 0.0
    query_ts: float = 0.0
    age_s: float = 0.0
    unit: str = ""
    labels: Mapping[str, str] = field(default_factory=dict)
    used_as_current: bool = True
    reset: bool = False
    double_counted: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.metric_name not in METRIC_UNITS:
            problems.append(f"unknown metric {self.metric_name!r}")
        if not self.unit:
            problems.append(f"{self.metric_name}: unit must be recorded (seconds/bytes/ratio)")
        elif self.unit != METRIC_UNITS.get(self.metric_name) and self.metric_name in METRIC_UNITS:
            problems.append(
                f"{self.metric_name}: unit {self.unit!r} != declared {METRIC_UNITS[self.metric_name]!r}"
            )
        if self.event_ts <= 0:
            problems.append(f"{self.metric_name}: the event timestamp is required")
        for name in ("export_ts", "scrape_ts"):
            value = getattr(self, name)
            if value and value < self.event_ts:
                problems.append(f"{self.metric_name}: {name} precedes the event timestamp")
        if self.age_s < 0:
            problems.append("metric age must not be negative")
        for label in self.labels:
            if label in rec.FORBIDDEN_METRIC_LABELS:
                problems.append(
                    f"{self.metric_name}: label {label!r} is unbounded and belongs in traces/logs"
                )
        if self.reset and self.used_as_current:
            problems.append(f"{self.metric_name}: a counter reset was used as a current value")
        if self.double_counted:
            problems.append(f"{self.metric_name}: the sample is double counted")
        return problems

    def age(self) -> float:
        if self.age_s:
            return self.age_s
        reference = self.query_ts or self.scrape_ts or self.export_ts or self.event_ts
        return max(reference - self.event_ts, 0.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "event_ts": self.event_ts,
            "export_ts": self.export_ts,
            "scrape_ts": self.scrape_ts,
            "age_s": self.age(),
        }


@dataclass
class StaleMetricPolicy:
    """Steps 7/28: staleness thresholds and the fail-safe action."""

    policy_id: str
    max_age_s: Mapping[str, float] = field(default_factory=dict)
    on_stale: str = ""
    on_missing: str = ""
    alert_id: str = ""
    hold_last_known_good: bool = True

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.policy_id:
            problems.append("stale policy needs an id")
        if not self.max_age_s:
            problems.append("per-metric max ages must be frozen")
        for metric in self.max_age_s:
            if metric not in METRIC_UNITS:
                problems.append(f"max age declared for unknown metric {metric!r}")
        if self.on_stale not in ("HOLD", "FAILSAFE_STATIC", "FAILSAFE_MIN"):
            problems.append(
                f"on_stale={self.on_stale!r} must be a fail-safe action (never 'scale down to zero')"
            )
        if self.on_missing not in ("HOLD", "FAILSAFE_STATIC", "FAILSAFE_MIN"):
            problems.append(f"on_missing={self.on_missing!r} must be a fail-safe action")
        if not self.alert_id:
            problems.append("a stale/missing metric must raise an alert")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "max_age_s": dict(sorted(self.max_age_s.items())),
            "on_stale": self.on_stale,
            "on_missing": self.on_missing,
            "alert_id": self.alert_id,
        }


def classify_metric(sample: MetricSample, policy: StaleMetricPolicy) -> Dict[str, Any]:
    """Steps 7/28: fresh/stale/missing decides whether a sample may drive a decision."""
    problems = sample.validate()
    limit = policy.max_age_s.get(sample.metric_name, 0.0)
    age = sample.age()
    if limit and age > limit:
        return {
            "metric_name": sample.metric_name,
            "age_s": age,
            "status": "STALE",
            "action": policy.on_stale,
            "may_scale": False,
            "problems": problems + [f"age {age}s > policy {limit}s"],
        }
    return {
        "metric_name": sample.metric_name,
        "age_s": age,
        "status": "FRESH",
        "action": "USE",
        "may_scale": not problems,
        "problems": problems,
    }


# ── desired replicas ─────────────────────────────────────────────────────


@dataclass
class CapacityModel:
    """The per-replica capacity that E13-06 exports (never a guess)."""

    safe_capacity_per_replica: float = 0.0
    workload_mix: str = ""
    capacity_policy_version: str = ""
    cold_start_ready_s: float = 0.0
    headroom_factor: float = 1.0

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.safe_capacity_per_replica <= 0:
            problems.append("safe capacity per replica must come from E13-06 and be positive")
        if not self.capacity_policy_version:
            problems.append("the capacity policy version must bind the numbers to a policy")
        if not self.workload_mix:
            problems.append("capacity is workload-mix dependent: the mix must be named")
        if self.headroom_factor < 1.0:
            problems.append("headroom factor below 1 would plan for overload")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "safe_capacity_per_replica": self.safe_capacity_per_replica,
            "workload_mix": self.workload_mix,
            "capacity_policy_version": self.capacity_policy_version,
            "cold_start_ready_s": self.cold_start_ready_s,
            "headroom_factor": self.headroom_factor,
        }


def desired_replicas(
    *,
    demand_estimate: float,
    capacity: CapacityModel,
    current_replicas: int,
    min_replicas: int,
    max_replicas: int,
    tolerance: float = 0.1,
) -> Dict[str, Any]:
    """§3: ``ceil(demand / safe_per_replica_capacity × headroom)`` with bounds.

    Deliberately pure: the caller supplies the demand estimate (from the frozen
    policy) and the *ready* replica count, and gets back both the raw and the
    bounded value plus the reason.
    """
    problems = capacity.validate()
    if demand_estimate < 0:
        problems.append("demand estimate must not be negative")
    if min_replicas <= 0:
        problems.append("min replicas must be positive for an LLM service (scale-to-zero needs its own run)")
    if max_replicas < min_replicas:
        problems.append("max replicas must be >= min replicas")
    if capacity.safe_capacity_per_replica <= 0:
        raw = float(current_replicas)
    else:
        raw = math.ceil(
            demand_estimate / capacity.safe_capacity_per_replica * capacity.headroom_factor
        )
    bounded = max(min_replicas, min(raw, max_replicas))
    change_ratio = abs(bounded - current_replicas) / current_replicas if current_replicas else float("inf")
    within_tolerance = change_ratio <= tolerance
    return {
        "desired_raw": raw,
        "desired_bounded": bounded,
        "current_replicas": current_replicas,
        "within_tolerance": within_tolerance,
        "clamped_by": ("min" if raw < min_replicas else "max" if raw > max_replicas else ""),
        "problems": problems,
        "ok": not problems,
        "unit_of_capacity": "qualified_requests_per_second",
    }


@dataclass
class RatePolicy:
    """§4/§15: tolerance, stabilization windows and per-direction rate limits."""

    policy_id: str
    tolerance: float = 0.1
    scale_up_stabilization_s: float = 0.0
    scale_down_stabilization_s: float = 0.0
    scale_up_rate_limit: float = 0.0
    scale_down_rate_limit: float = 0.0
    cooldown_s: float = 0.0
    max_actions_per_hour: float = DEFAULT_MAX_ACTIONS_PER_HOUR

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.policy_id:
            problems.append("rate policy needs an id")
        if not 0 <= self.tolerance < 1:
            problems.append("tolerance must be inside [0, 1)")
        if self.scale_up_rate_limit <= 0:
            problems.append("a scale-up rate limit is required (unbounded scale-up is a cost incident)")
        if self.scale_down_rate_limit <= 0:
            problems.append("a scale-down rate limit is required")
        if self.scale_down_stabilization_s <= 0:
            problems.append(
                "a scale-down stabilization window is required: instant scale-down causes thrashing "
                "and cache loss"
            )
        if self.max_actions_per_hour <= 0:
            problems.append("a maximum action rate must be pre-registered")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "tolerance": self.tolerance,
            "scale_up_stabilization_s": self.scale_up_stabilization_s,
            "scale_down_stabilization_s": self.scale_down_stabilization_s,
            "scale_up_rate_limit": self.scale_up_rate_limit,
            "scale_down_rate_limit": self.scale_down_rate_limit,
            "cooldown_s": self.cooldown_s,
            "max_actions_per_hour": self.max_actions_per_hour,
        }


def stabilize(
    *, desired_raw: int, history: Sequence[Mapping[str, Any]], policy: RatePolicy, now: float,
    sustaining_metric: bool = False,
) -> Dict[str, Any]:
    """Stabilization windows: a scale-down needs a *sustained* low demand.

    A single low sample may not reduce capacity — that is the difference between a
    controller and a random number generator.
    """
    problems = policy.validate()
    window = (
        policy.scale_up_stabilization_s if desired_raw > 0 and _is_scale_up(history, desired_raw)
        else policy.scale_down_stabilization_s
    )
    recent = [row for row in history if now - float(row.get("ts", 0.0)) <= window]
    values = [int(row.get("desired_raw", desired_raw)) for row in recent] or [desired_raw]
    if desired_raw >= max(values):
        desired = max(values)
        direction = "UP"
    else:
        desired = min(values) if sustaining_metric else max(values)
        direction = "DOWN" if sustaining_metric else "HOLD"
    return {
        "desired_raw": desired_raw,
        "desired_stabilized": desired,
        "direction": direction,
        "window_s": window,
        "samples_in_window": len(values),
        "problems": problems,
        "ok": not problems,
    }


def _is_scale_up(history: Sequence[Mapping[str, Any]], desired_raw: int) -> bool:
    if not history:
        return True
    return desired_raw >= max(int(row.get("desired_raw", 0)) for row in history)


def enforce_rate_limit(
    *, current: int, desired: int, policy: RatePolicy, last_action_ts: float, now: float
) -> Dict[str, Any]:
    """Apply the rate limits/cooldown and report what was actually allowed."""
    problems = policy.validate()
    if desired > current:
        allowed = min(desired, math.ceil(current * policy.scale_up_rate_limit))
    elif desired < current:
        allowed = max(desired, math.floor(current * policy.scale_down_rate_limit))
    else:
        allowed = current
    cooling = bool(last_action_ts) and (now - last_action_ts) < policy.cooldown_s
    if cooling and allowed != current:
        allowed = current
    return {
        "requested": desired,
        "allowed": allowed,
        "in_cooldown": cooling,
        "rate_limited": allowed != desired,
        "problems": problems,
        "ok": not problems,
    }


# ── decision record ──────────────────────────────────────────────────────


@dataclass
class AutoscalingDecision:
    """§11 ``AutoscalingDecision`` (validated by §23 V08 as well)."""

    decision_id: str
    episode_id: str
    timestamp: float = 0.0
    policy_version: str = ""
    release_id: str = ""
    model_artifact_id: str = ""
    metric_name_values: Mapping[str, float] = field(default_factory=dict)
    metric_ages: Mapping[str, float] = field(default_factory=dict)
    current_replicas: int = 0
    ready_replicas: int = 0
    available_replicas: int = 0
    desired_raw: int = 0
    desired_stabilized: int = 0
    safe_capacity_per_replica: float = 0.0
    demand_estimate: float = 0.0
    headroom: float = 1.0
    tolerance: float = 0.0
    window_s: float = 0.0
    rate_limit: float = 0.0
    cooldown_s: float = 0.0
    action: str = ""
    reason: str = ""
    controller_generation: str = ""
    scheduled_at: float = 0.0
    ready_at: float = 0.0
    failure_reason: str = ""
    slo_consequence: str = ""
    evidence_refs: Tuple[str, ...] = ()

    def validate(self, stale_policy: Optional[StaleMetricPolicy] = None) -> List[str]:
        problems: List[str] = []
        if self.action not in rec.AUTOSCALING_ACTIONS:
            problems.append(f"unknown autoscaling action {self.action!r}")
        if not self.policy_version:
            problems.append("a decision must name its policy version")
        if not self.metric_name_values:
            problems.append("a decision must record the metric values it used")
        for metric in self.metric_name_values:
            if metric not in METRIC_UNITS:
                problems.append(f"unknown metric {metric!r} in a decision")
        if len(self.metric_ages) != len(self.metric_name_values):
            problems.append("every metric used must carry its age (stale metrics may not look fresh)")
        if self.current_replicas and self.ready_replicas > self.current_replicas:
            problems.append("ready replicas exceed current replicas")
        if self.desired_stabilized < 1 and self.desired_stabilized != 0:
            problems.append("stabilized desired replicas must not be negative")
        if self.action == "SCALE_DOWN" and self.ready_replicas <= 1:
            problems.append("scale-down leaving zero ready capacity would be a self-inflicted outage")
        if stale_policy is not None:
            for metric, age in self.metric_ages.items():
                limit = stale_policy.max_age_s.get(metric, 0.0)
                if limit and age > limit:
                    problems.append(
                        f"{metric}: the decision used a stale sample (age {age}s > {limit}s)"
                    )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "policy_version": self.policy_version,
            "metric_name_values": dict(sorted(self.metric_name_values.items())),
            "metric_ages": dict(sorted(self.metric_ages.items())),
            "current_replicas": self.current_replicas,
            "ready_replicas": self.ready_replicas,
            "desired_raw": self.desired_raw,
            "desired_stabilized": self.desired_stabilized,
            "action": self.action,
            "reason": self.reason,
            "controller_generation": self.controller_generation,
        }


def decide(
    *,
    decision: AutoscalingDecision,
    capacity: CapacityModel,
    rate_policy: RatePolicy,
    stale_policy: StaleMetricPolicy,
    demand: float,
    min_replicas: int,
    max_replicas: int,
    metrics: Sequence[MetricSample],
    sustaining: bool = False,
    history: Sequence[Mapping[str, Any]] = (),
    last_action_ts: float = 0.0,
) -> AutoscalingDecision:
    """Run one full decision tick and explain every stage of it.

    The order matters: classify metric freshness → compute raw demand → stabilize →
    rate-limit.  A stale sample short-circuits to the fail-safe action instead of
    participating in the estimate.
    """
    classifications = [classify_metric(sample, stale_policy) for sample in metrics]
    decision.metric_name_values = {sample.metric_name: sample.value for sample in metrics}
    decision.metric_ages = {sample.metric_name: sample.age() for sample in metrics}
    unusable = [row for row in classifications if not row["may_scale"]]
    if unusable:
        decision.action = "HOLD"
        decision.reason = f"fail-safe: {stale_policy.on_stale} for {[row['metric_name'] for row in unusable]}"
        decision.controller_generation = decision.controller_generation or "gen-unknown"
        return decision

    computed = desired_replicas(
        demand_estimate=demand,
        capacity=capacity,
        current_replicas=decision.current_replicas,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        tolerance=rate_policy.tolerance,
    )
    decision.desired_raw = int(computed["desired_raw"])
    stabilized = stabilize(
        desired_raw=decision.desired_raw, history=history, policy=rate_policy,
        now=decision.timestamp, sustaining_metric=sustaining,
    )
    decision.desired_stabilized = int(stabilized["desired_stabilized"])
    decision.window_s = float(stabilized["window_s"])
    limited = enforce_rate_limit(
        current=decision.current_replicas, desired=decision.desired_stabilized,
        policy=rate_policy, last_action_ts=last_action_ts, now=decision.timestamp,
    )
    final = int(limited["allowed"])
    decision.rate_limit = float(rate_policy.scale_up_rate_limit if final > decision.current_replicas
                                else rate_policy.scale_down_rate_limit)
    decision.cooldown_s = rate_policy.cooldown_s
    if final > decision.current_replicas:
        decision.action, decision.reason = "SCALE_UP", "demand above ready capacity"
    elif final < decision.current_replicas:
        decision.action, decision.reason = "SCALE_DOWN", "sustained demand below ready capacity"
    else:
        decision.action, decision.reason = "HOLD", (
            "within tolerance" if computed["within_tolerance"] else "held by stabilization/rate limit"
        )
    decision.controller_generation = decision.controller_generation or f"gen-{decision.decision_id}"
    return decision


# ── stability, cost and episodes ─────────────────────────────────────────


def control_metrics(
    *,
    episode_id: str,
    readiness_series: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    max_actions_per_hour: float = DEFAULT_MAX_ACTIONS_PER_HOUR,
    max_amplitude: float = DEFAULT_MAX_AMPLITUDE,
    settling_s: float = DEFAULT_SETTLING_S,
    recovery_within_s: float = 0.0,
) -> Dict[str, Any]:
    """§5/§7: detection→ready latency, overshoot, oscillation and settling."""
    problems: List[str] = []
    ready = [int(row.get("ready_replicas", 0)) for row in readiness_series]
    peak = max(ready) if ready else 0
    needed = int(readiness_series[-1].get("needed_replicas", peak) if readiness_series else 0)
    overshoot = max(peak - needed, 0)
    detection_delays = [float(row.get("detection_delay_s", 0.0)) for row in readiness_series if row.get("detection_delay_s")]
    decision_to_ready = [
        float(row["ready_at"]) - float(row["decision_ts"])
        for row in readiness_series
        if row.get("ready_at") and row.get("decision_ts")
    ]
    amplitude = 0
    previous: Optional[int] = None
    for value in ready:
        if previous is not None:
            amplitude = max(amplitude, abs(value - previous))
        previous = value
    actions_per_hour = (len(actions) / (recovery_within_s / 3600.0)) if recovery_within_s else float(len(actions))
    oscillating = actions_per_hour > max_actions_per_hour or amplitude > max_amplitude
    if oscillating:
        problems.append(
            f"sustained oscillation: {actions_per_hour:.2f} actions/hour (limit {max_actions_per_hour}) "
            f"or amplitude {amplitude} (limit {max_amplitude})"
        )
    if recovery_within_s and recovery_within_s > settling_s:
        problems.append(
            f"the episode did not settle within {settling_s}s (took {recovery_within_s}s)"
        )
    return {
        "episode_id": episode_id,
        "detection_delay_s": (sum(detection_delays) / len(detection_delays)) if detection_delays else 0.0,
        "decision_to_ready_s": (sum(decision_to_ready) / len(decision_to_ready)) if decision_to_ready else 0.0,
        "overshoot": overshoot,
        "amplitude": amplitude,
        "actions": len(actions),
        "actions_per_hour": actions_per_hour,
        "oscillating": oscillating,
        "problems": problems,
        "ok": not problems,
        "oscillation_definition": OSCILLATION_DEFINITION,
    }


def cost_accounting(
    *, episode_id: str, policy_id: str, replica_seconds: float, device_seconds: float,
    cold_starts: int, cache_hit_ratio: float, duration_s: float,
    reference_replica_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """§5/§15: an SLO improvement is reported together with its resource cost."""
    problems: List[str] = []
    if duration_s <= 0:
        problems.append("episode duration must be positive")
    if replica_seconds < 0 or device_seconds < 0:
        problems.append("replica/device seconds must not be negative")
    if reference_replica_seconds is not None and reference_replica_seconds > 0:
        delta = (replica_seconds - reference_replica_seconds) / reference_replica_seconds
    else:
        delta = None
    return {
        "episode_id": episode_id,
        "policy_id": policy_id,
        "replica_minutes": replica_seconds / 60.0 if replica_seconds else 0.0,
        "device_minutes": device_seconds / 60.0 if device_seconds else 0.0,
        "cold_starts": cold_starts,
        "cache_hit_ratio": cache_hit_ratio,
        "replica_minutes_delta_vs_baseline": delta,
        "problems": problems,
        "ok": not problems,
    }


@dataclass
class EpisodePlan:
    """Steps 1–3: the frozen episodes, traces and stopping rules."""

    plan_id: str
    episodes: Tuple[str, ...] = rec.AUTOSCALING_EPISODES
    steady_low_duration_s: float = 0.0
    steady_high_duration_s: float = 0.0
    step_up_target: float = 0.0
    burst_duration_s: float = 0.0
    ramp_duration_s: float = 0.0
    periodic_period_s: float = 0.0
    step_down_fraction: float = 0.0
    holdout_episode_ids: Tuple[str, ...] = ()
    stopping_rule: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.plan_id:
            problems.append("episode plan needs an id")
        missing = sorted(set(rec.AUTOSCALING_EPISODES) - set(self.episodes))
        if missing:
            problems.append(f"episodes not planned: {missing}")
        if self.burst_duration_s <= 0:
            problems.append("the short burst duration must be planned (it is compared against T_response)")
        if not (0 < self.step_down_fraction <= 1):
            problems.append("step-down fraction must be inside (0, 1]")
        if not self.holdout_episode_ids:
            problems.append("holdout episodes must be reserved before tuning (no tuning on holdout)")
        if not self.stopping_rule:
            problems.append("a stopping rule is required for overload episodes")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "episodes": list(self.episodes),
            "burst_duration_s": self.burst_duration_s,
            "ramp_duration_s": self.ramp_duration_s,
            "periodic_period_s": self.periodic_period_s,
            "step_down_fraction": self.step_down_fraction,
            "holdout_episode_ids": list(self.holdout_episode_ids),
            "stopping_rule": self.stopping_rule,
        }


def control_loop_delay(
    *, t_metric: float, t_decision: float, t_schedule: float, t_pull: float, t_model: float,
    t_compile: float, t_warmup: float, t_ready: float, burst_duration_s: float
) -> Dict[str, Any]:
    """§2: total response time and the honest conclusion for short bursts."""
    total = t_metric + t_decision + t_schedule + t_pull + t_model + t_compile + t_warmup + t_ready
    covers = burst_duration_s >= total
    return {
        "t_response_s": total,
        "components": {
            "t_metric": t_metric,
            "t_decision": t_decision,
            "t_schedule": t_schedule,
            "t_pull": t_pull,
            "t_model": t_model,
            "t_compile": t_compile,
            "t_warmup": t_warmup,
            "t_ready": t_ready,
        },
        "burst_duration_s": burst_duration_s,
        "reactive_scaling_sufficient": covers,
        "note": (
            ""
            if covers
            else "the burst is shorter than T_response: protect it with headroom/admission/pre-warm, "
            "and do not attribute the failure to the scaling algorithm"
        ),
    }


def run_failure_cases(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Steps 28–32: fail-safe behaviour under metric/controller/scheduler failures."""
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        kind = str(case.get("kind", ""))
        if kind not in FAILURE_CASES:
            raise ConfigError(f"unknown autoscaling failure case {kind!r}")
        dangerous = bool(case.get("scaled_to_min_or_zero", False))
        alerted = bool(case.get("alerted", False))
        rows.append(
            {
                "case_id": str(case.get("case_id", f"as-failure-{index:03d}")),
                "kind": kind,
                "expected": str(case.get("expected", "FAIL_SAFE")),
                "observed": str(case.get("observed", "")),
                "scaled_to_min_or_zero": dangerous,
                "alerted": alerted,
                "admission_protected": bool(case.get("admission_protected", False)),
                "ok": (not dangerous) and alerted,
            }
        )
    failures = [row for row in rows if not row["ok"]]
    return {
        "rows": rows,
        "cases": len(rows),
        "failures": failures,
        "ok": not failures,
        "reason": (
            ""
            if not failures
            else "unsafe autoscaling failure handling: "
            + ", ".join(f"{row['kind']}({'no-alert' if not row['alerted'] else 'unsafe-scale-down'})"
                        for row in failures)
        ),
    }


def scale_down_candidate_selection(
    *, candidates: Sequence[Mapping[str, Any]], protections: Sequence[str] = SCALE_DOWN_PROTECTIONS
) -> Dict[str, Any]:
    """Step 24: do not kill the pod holding the longest stream / the only cache."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        holds = [item for item in candidate.get("holds", ()) or () if item in protections]
        protected = bool(holds) and not candidate.get("policy_accepts_cost", False)
        rows.append(
            {
                "pod_uid": candidate.get("pod_uid", ""),
                "holds": sorted(holds),
                "selected": bool(candidate.get("selected", False)),
                "protected": protected,
                "reason": candidate.get("reason", ""),
            }
        )
        if protected and candidate.get("selected"):
            problems.append(
                f"pod {candidate.get('pod_uid')} was selected for scale-down while holding {sorted(holds)}"
            )
    return {"rows": rows, "problems": problems, "ok": not problems}


def coordination_checks(
    *, admission: Mapping[str, Any], drain: Mapping[str, Any], cache: Mapping[str, Any], rollout: Mapping[str, Any]
) -> Dict[str, Any]:
    """Steps 25–27/32: the four controllers must not amplify each other."""
    problems: List[str] = []
    axes = {"admission": admission, "drain": drain, "cache": cache, "rollout": rollout}
    for name, axis in axes.items():
        if not axis:
            problems.append(f"{name} coordination was not verified")
        elif not axis.get("ok"):
            problems.append(f"{name}: " + "; ".join(axis.get("problems", []) or ["unspecified"]))
    if not rollout.get("serialized_with_autoscaler", True):
        problems.append("rollout and autoscaler are not serialized")
    return {"axes": sorted(axes), "problems": problems, "ok": not problems}


def evaluate_holdout(*, rows: Sequence[Mapping[str, Any]], tuning_episode_ids: Sequence[str]) -> Dict[str, Any]:
    """Step 36: the frozen policy is evaluated on episodes it never saw."""
    problems: List[str] = []
    overlap = sorted(set(row.get("episode_id", "") for row in rows) & set(tuning_episode_ids))
    if overlap:
        problems.append(f"holdout evaluation reused tuning episodes: {overlap}")
    for row in rows:
        if row.get("policy_tuned_after"):
            problems.append(
                f"episode {row.get('episode_id')}: the policy was tuned after the holdout run"
            )
    return {"rows": list(rows), "problems": problems, "ok": not problems}


def autoscaling_verdict(
    *,
    metrics: Mapping[str, Any],
    cost: Mapping[str, Any],
    failures: Mapping[str, Any],
    coordination: Mapping[str, Any],
    holdout: Mapping[str, Any],
    scale_down: Mapping[str, Any],
) -> Dict[str, Any]:
    """Step 38: applicability, stability, cost and the forbidden claims."""
    problems: List[str] = []
    for name, axis in (
        ("control_metrics", metrics),
        ("failures", failures),
        ("coordination", coordination),
        ("holdout", holdout),
        ("scale_down", scale_down),
    ):
        if not axis.get("ok"):
            problems.append(f"{name}: " + "; ".join(axis.get("problems", []) or ["(no detail)"]))
    return {
        "problems": problems,
        "verdict": "PASSABLE_AT_CODE_LEVEL" if not problems else "BLOCKED",
        "forbidden_claims": [
            "elastic capacity without a holdout comparison",
            "scale-to-zero without an interactive cold-start budget",
            "SLO protection without the admission interaction",
        ],
        "cost_reported": bool(cost),
        "note": "reactive autoscaling cannot cover a burst shorter than T_response (see control_loop_delay)",
    }


# ── protocol steps and smoke self-check ──────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 SLO/容量/成本目标", ("autoscaling:CapacityModel", "serving:slo.SLOSpec")),
    (2, "冻结策略与调参预算", ("autoscaling:POLICIES", "autoscaling:EpisodePlan")),
    (3, "冻结负载 episode", ("autoscaling:EpisodePlan", "records:AUTOSCALING_EPISODES")),
    (4, "冻结集群/节点条件", ("autoscaling:ClusterConditions",)),
    (5, "部署 autoscaling telemetry", ("observability:SemanticConvention", "autoscaling:MetricSample")),
    (6, "验证 metric 语义/单位", ("autoscaling:METRIC_UNITS", "autoscaling:MetricSample.validate")),
    (7, "测 metric age 与延迟", ("autoscaling:classify_metric", "autoscaling:StaleMetricPolicy")),
    (8, "验证 desired replica 公式", ("autoscaling:desired_replicas",)),
    (9, "部署 STATIC baseline", ("autoscaling:cost_accounting",)),
    (10, "部署 CPU_ONLY baseline", ("autoscaling:POLICIES",)),
    (11, "部署 ACCELERATOR_UTIL 策略", ("autoscaling:POLICIES",)),
    (12, "部署 QUEUE_REQUESTS 策略", ("autoscaling:POLICIES",)),
    (13, "部署 QUEUE_TOKEN_WORK 策略", ("autoscaling:LEADING_SIGNALS",)),
    (14, "部署 MULTI_SIGNAL 策略", ("autoscaling:decide",)),
    (15, "校准 scale-to-ready 分布", ("deployment:cold_warm_stage_times", "autoscaling:control_loop_delay")),
    (16, "运行 steady low episode", ("autoscaling:decide", "autoscaling:stabilize")),
    (17, "运行 steady high episode", ("autoscaling:decide",)),
    (18, "运行 step-up episode", ("autoscaling:control_metrics",)),
    (19, "运行 short-burst episode", ("autoscaling:control_loop_delay",)),
    (20, "运行 ramp-up episode", ("autoscaling:enforce_rate_limit",)),
    (21, "运行 periodic/repeated burst", ("autoscaling:control_metrics",)),
    (22, "运行 mixed-length episode", ("autoscaling:desired_replicas",)),
    (23, "运行 step-down episode", ("autoscaling:stabilize", "lifecycle:DrainTimeline")),
    (24, "验证 scale-down 候选选择", ("autoscaling:scale_down_candidate_selection",)),
    (25, "验证 E13-06 admission 协同", ("autoscaling:coordination_checks", "capacity:admit")),
    (26, "验证 E13-05 lifecycle 协同", ("autoscaling:coordination_checks", "lifecycle:rolling_availability")),
    (27, "验证 E13-04 cache 协同", ("autoscaling:coordination_checks", "artifacts:single_flight_outcome")),
    (28, "测试 metric missing/stale", ("autoscaling:run_failure_cases", "autoscaling:classify_metric")),
    (29, "测试 controller restart/leader change", ("autoscaling:run_failure_cases",)),
    (30, "测试 unschedulable/device shortage", ("autoscaling:run_failure_cases", "scheduling:filter_nodes")),
    (31, "测试慢 artifact/model startup", ("autoscaling:run_failure_cases", "autoscaling:control_loop_delay")),
    (32, "测试 rollout 与 autoscaling 并发", ("autoscaling:coordination_checks", "lifecycle:RollingPolicy")),
    (33, "计算控制性能", ("autoscaling:control_metrics",)),
    (34, "计算服务与成本", ("autoscaling:cost_accounting",)),
    (35, "做参数敏感性", ("autoscaling:sensitivity_sweep",)),
    (36, "在 holdout episodes 确认", ("autoscaling:evaluate_holdout",)),
    (37, "验证 runbook/alerts", ("observability:build_alert_rule", "autoscaling:run_failure_cases")),
    (38, "形成 autoscaling verdict", ("autoscaling:autoscaling_verdict",)),
)


@dataclass
class ClusterConditions:
    """Step 4: the cluster/node conditions the episodes run under."""

    conditions_id: str
    min_nodes: int = 0
    max_nodes: int = 0
    device_pool: Tuple[str, ...] = ()
    cluster_autoscaler_enabled: bool = False
    image_cache_state: str = ""
    model_cache_state: str = ""
    placement_plan_id: str = ""
    external_capacity_limit: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("conditions_id", "image_cache_state", "model_cache_state", "placement_plan_id"):
            if not getattr(self, name):
                problems.append(f"cluster conditions require {name!r}")
        if self.min_nodes <= 0 or self.max_nodes < self.min_nodes:
            problems.append("node bounds must be recorded and ordered")
        if not self.device_pool:
            problems.append("the schedulable device pool must be recorded (it bounds the replica count)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "conditions_id": self.conditions_id,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "device_pool": list(self.device_pool),
            "cluster_autoscaler_enabled": self.cluster_autoscaler_enabled,
            "image_cache_state": self.image_cache_state,
            "model_cache_state": self.model_cache_state,
            "placement_plan_id": self.placement_plan_id,
            "external_capacity_limit": self.external_capacity_limit,
        }


def sensitivity_sweep(
    rows: Sequence[Mapping[str, Any]], *, parameters: Sequence[str], registered_values: Mapping[str, Sequence[Any]]
) -> Dict[str, Any]:
    """Step 35: the parameter sweep is pre-registered and reported as a Pareto set."""
    problems: List[str] = []
    unregistered: List[Dict[str, Any]] = []
    for row in rows:
        for parameter in parameters:
            if parameter not in registered_values:
                problems.append(f"parameter {parameter!r} has no registered value set")
                continue
            value = row.get(parameter)
            if value is not None and value not in registered_values[parameter]:
                unregistered.append({"row": row.get("row_id", ""), "parameter": parameter, "value": value})
    if unregistered:
        problems.append(f"{len(unregistered)} sweep rows used unregistered parameter values")
    return {
        "rows": list(rows),
        "parameters": list(parameters),
        "unregistered": unregistered,
        "problems": problems,
        "ok": not problems,
        "note": "report the SLO/cost/stability Pareto set, not a single magic parameter combination",
    }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the autoscaling contracts (smoke, not an experiment)."""
    checks: Dict[str, Any] = {}
    capacity = CapacityModel(
        safe_capacity_per_replica=10.0, workload_mix="mixed", capacity_policy_version="cap-v1",
        cold_start_ready_s=210.0, headroom_factor=1.2,
    )
    checks["capacity_valid"] = capacity.validate() == []

    computed = desired_replicas(demand_estimate=45.0, capacity=capacity, current_replicas=4,
                               min_replicas=2, max_replicas=8)
    checks["desired_replicas_ceil"] = computed["desired_raw"] == 6 and computed["desired_bounded"] == 6

    clamped = desired_replicas(demand_estimate=1000.0, capacity=capacity, current_replicas=4,
                              min_replicas=2, max_replicas=8)
    checks["max_replicas_clamped"] = clamped["desired_bounded"] == 8 and clamped["clamped_by"] == "max"

    fresh = MetricSample(sample_id="s1", metric_name="queue_tokens", value=120.0, unit="tokens",
                         event_ts=100.0, export_ts=101.0, scrape_ts=102.0, query_ts=103.0)
    stale_policy = StaleMetricPolicy(
        policy_id="stale-1",
        max_age_s={"queue_tokens": 30.0, "cpu_utilization": 30.0, "ready_replicas": 30.0},
        on_stale="FAILSAFE_STATIC", on_missing="FAILSAFE_STATIC", alert_id="alert-stale-metric",
    )
    checks["stale_policy_valid"] = stale_policy.validate() == []
    checks["fresh_metric_usable"] = classify_metric(fresh, stale_policy)["may_scale"] is True

    old = MetricSample(sample_id="s2", metric_name="queue_tokens", value=120.0, unit="tokens",
                       event_ts=0.0, export_ts=1.0, scrape_ts=2.0, query_ts=1000.0)
    checks["stale_metric_blocked"] = classify_metric(old, stale_policy)["may_scale"] is False

    forbidden_label = MetricSample(sample_id="s3", metric_name="queue_tokens", value=1.0, unit="tokens",
                                   event_ts=1.0, labels={"request_id": "r1"})
    checks["high_cardinality_label_rejected"] = any(
        "unbounded" in problem for problem in forbidden_label.validate()
    )

    rate = RatePolicy(policy_id="rate-1", tolerance=0.1, scale_up_stabilization_s=0.0,
                      scale_down_stabilization_s=300.0, scale_up_rate_limit=2.0,
                      scale_down_rate_limit=0.5, cooldown_s=60.0)
    checks["rate_policy_valid"] = rate.validate() == []

    decision = AutoscalingDecision(
        decision_id="d1", episode_id="ep1", timestamp=1000.0, policy_version="as-v1",
        release_id="rel1", model_artifact_id="m1", current_replicas=2, ready_replicas=2,
        controller_generation="gen-1",
    )
    decide(decision=decision, capacity=capacity, rate_policy=rate, stale_policy=stale_policy,
           demand=45.0, min_replicas=1, max_replicas=8,
           metrics=[MetricSample(sample_id="s4", metric_name="queue_tokens", value=45.0,
                                 unit="tokens", event_ts=999.0, query_ts=1000.0)])
    checks["scale_up_decided"] = decision.action == "SCALE_UP" and decision.desired_raw == 6

    unsafe = AutoscalingDecision(
        decision_id="d2", episode_id="ep1", timestamp=1000.0, policy_version="as-v1",
        current_replicas=1, ready_replicas=1, action="SCALE_DOWN", desired_stabilized=1,
        metric_name_values={"queue_tokens": 1.0}, metric_ages={"queue_tokens": 600.0},
        release_id="rel1", model_artifact_id="m1",
    )
    unsafe_problems = unsafe.validate(stale_policy=stale_policy)
    checks["unsafe_scale_down_flagged"] = len(unsafe_problems) >= 2
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "note": "接口自检；未连接任何集群，未创建/删除任何副本",
    }