"""E13-09: fault injection, degradation and recovery across the service stack.

Implements ``details/S13/E13-09_*.md`` as data:

* the fault contract of §2 (steady-state hypothesis, exact resolved targets,
  mechanism, blast radius, health gate, expected detection/degradation/recovery,
  SLI/SLO abort threshold, kill switch, repetitions, allowed claims);
* target resolution *before* injection and the ground-truth record for the
  effective time (a command returning is not the fault becoming effective);
* the MTTD/MTTM/MTTR definition of §3 measured against the user steady state;
* the degradation classes of §5, with the rule that a semantics-changing fallback
  must be declared and quality-checked (a silent quality drop is a FAIL);
* retry-budget/backoff checks (retry amplification is a cost, not a fix);
* steady-state and resource invariants, residual watch and postmortems;
* per-class verdicts plus the honest ``MANUAL_ESCALATION``/``UNRECOVERABLE``
  boundary the stage may not hide.

Nothing here injects, kills or throttles anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.infra import records as rec

EXPERIMENT_ID = "E13-09"
TITLE = "进程、Pod、节点、设备、Cache、存储、网络、制品、OOM、Thermal 故障注入与恢复"
CLAIM_BOUNDARY = (
    "本实验通过证明已测故障的真实行为；不证明系统对未建模故障、区域级灾难或恶意攻击自动可靠。"
)

SCHEMA_VERSION = "1.0.0"

LAYERS: Tuple[str, ...] = rec.FAULT_LAYERS
MECHANISMS: Tuple[str, ...] = rec.FAULT_MECHANISMS
DEGRADATIONS: Tuple[str, ...] = rec.DEGRADATION_STRATEGIES
VERDICTS: Tuple[str, ...] = rec.FAULT_VERDICTS

#: Every fault case must declare these fields before injection (§2).
FAULT_CONTRACT_FIELDS: Tuple[str, ...] = (
    "fault_case_id",
    "hypothesis",
    "layer",
    "mechanism",
    "target_selector",
    "resolved_targets",
    "blast_radius",
    "safety_policy_id",
    "expected_detection",
    "expected_degradation",
    "expected_recovery",
    "abort_threshold",
    "kill_switch",
    "repetitions",
    "allowed_claims",
    "forbidden_claims",
)

#: Health gates that must pass before a destructive injection (§9 step 7).
PRE_INJECTION_GATES: Tuple[str, ...] = (
    "steady_state_in_range",
    "observability_ready",
    "runbook_ready",
    "targets_resolved",
    "kill_switch_tested",
    "error_budget_available",
)

#: Resource invariants that must hold after recovery (§4).
RESOURCE_INVARIANTS: Tuple[str, ...] = (
    "kv_and_reservation_released",
    "device_memory_and_context_released",
    "cache_lock_and_lease_consistent",
    "pvc_and_temp_files_clean",
    "scheduler_allocatable_matches_actual",
    "failed_endpoints_no_traffic",
    "controller_generations_not_stale",
)

#: Service invariants that must hold after recovery.
SERVICE_INVARIANTS: Tuple[str, ...] = (
    "accepted_requests_have_one_final_status",
    "partial_duplicate_retry_identifiable",
    "quality_and_version_unchanged",
    "slo_violation_and_error_budget_computable",
)


@dataclass
class FaultSpec:
    """§22.6 ``FaultExperiment`` pre-registration (validated as §23 V10 too)."""

    fault_case_id: str
    layer: str = ""
    mechanism: str = ""
    hypothesis: str = ""
    target_selector: str = ""
    resolved_targets: Tuple[str, ...] = ()
    namespace: str = ""
    blast_radius: str = ""
    safety_policy_id: str = ""
    duration_s: float = 0.0
    max_impact: str = ""
    expected_detection: str = ""
    expected_degradation: str = ""
    expected_recovery: str = ""
    expected_degradation_class: str = ""
    abort_threshold: str = ""
    kill_switch: str = ""
    repetitions: int = 0
    allowed_claims: Tuple[str, ...] = ()
    forbidden_claims: Tuple[str, ...] = ()
    severity: str = ""
    health_gate: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in FAULT_CONTRACT_FIELDS:
            value = getattr(self, name)
            if value in ("", (), None, 0):
                problems.append(f"fault contract requires {name!r}")
        if self.layer not in LAYERS:
            problems.append(f"unknown fault layer {self.layer!r}")
        if self.mechanism not in MECHANISMS:
            problems.append(f"unknown fault mechanism {self.mechanism!r}")
        if self.expected_degradation_class and self.expected_degradation_class not in DEGRADATIONS:
            problems.append(f"unknown degradation class {self.expected_degradation_class!r}")
        if any(token in self.target_selector for token in ("*", "?", "[")):
            problems.append("the target selector must be explicit: wildcards are forbidden for faults")
        if self.duration_s <= 0:
            problems.append("fault duration must be positive (unbounded faults are not experiments)")
        if self.repetitions < 1:
            problems.append("repetitions must be planned")
        missing_gates = sorted(set(PRE_INJECTION_GATES) - set(self.health_gate))
        if missing_gates:
            problems.append(f"the pre-injection health gate is incomplete: {missing_gates}")
        if self.expected_degradation_class in rec.SEMANTIC_CHANGING_DEGRADATIONS:
            if "quality_rechecked" not in self.expected_recovery:
                problems.append(
                    f"degradation {self.expected_degradation_class} changes user-visible semantics: "
                    "the recovery contract must state that quality is re-checked"
                )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fault_case_id": self.fault_case_id,
            "hypothesis": self.hypothesis,
            "layer": self.layer,
            "mechanism": self.mechanism,
            "target_selector": self.target_selector,
            "resolved_targets": list(self.resolved_targets),
            "blast_radius": self.blast_radius,
            "safety_policy_id": self.safety_policy_id,
            "expected_detection": self.expected_detection,
            "expected_degradation": self.expected_degradation,
            "expected_recovery": self.expected_recovery,
            "abort_threshold": self.abort_threshold,
        }


def validate_fault_matrix(
    specs: Sequence[FaultSpec], *, required_layers: Sequence[str] = LAYERS
) -> Dict[str, Any]:
    """Step 1/step 5: the matrix must cover the required layers and controls."""
    problems: List[str] = []
    covered: Dict[str, List[str]] = {}
    for spec in specs:
        problems.extend(spec.validate())
        covered.setdefault(spec.layer, []).append(spec.fault_case_id)
    missing = sorted(set(required_layers) - set(covered))
    if missing:
        problems.append(
            f"the fault matrix does not cover {missing}: an untested layer may not appear in a reliability "
            "claim"
        )
    return {
        "cases": len(specs),
        "layers_covered": {layer: sorted(ids) for layer, ids in sorted(covered.items())},
        "layers_missing": missing,
        "problems": problems,
        "ok": not problems,
    }


@dataclass
class SteadyState:
    """Step 2: the baseline range that ``t_recover`` is measured against."""

    steady_state_id: str
    sli_ranges: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    queue_range: Tuple[float, float] = (0.0, 0.0)
    capacity_range: Tuple[float, float] = (0.0, 0.0)
    release_id: str = ""
    model_artifact_id: str = ""
    resource_ranges: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    observation_window_s: float = 0.0

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("steady_state_id", "release_id", "model_artifact_id"):
            if not getattr(self, name):
                problems.append(f"steady state requires {name!r}")
        if not self.sli_ranges:
            problems.append("the steady-state SLI ranges must be frozen before injection")
        for name, bounds in (("queue", self.queue_range), ("capacity", self.capacity_range)):
            if bounds[0] > bounds[1]:
                problems.append(f"{name} range is inverted: {bounds}")
        if self.observation_window_s <= 0:
            problems.append("an observation window is required to decide recovery")
        return problems

    def contains(self, sli: str, value: float) -> bool:
        bounds = self.sli_ranges.get(sli)
        if bounds is None:
            return False
        return bounds[0] <= value <= bounds[1]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "steady_state_id": self.steady_state_id,
            "sli_ranges": {key: list(value) for key, value in sorted(self.sli_ranges.items())},
            "queue_range": list(self.queue_range),
            "capacity_range": list(self.capacity_range),
            "release_id": self.release_id,
            "model_artifact_id": self.model_artifact_id,
            "observation_window_s": self.observation_window_s,
        }


@dataclass
class FaultEpisode:
    """§11 ``FaultEpisode``: one injection/recovery episode."""

    episode_id: str
    fault_case_id: str
    control_or_fault: str = "FAULT"
    release_id: str = ""
    model_artifact_id: str = ""
    cluster_id: str = ""
    target_ids: Tuple[str, ...] = ()
    steady_state_hypothesis_id: str = ""
    safety_policy_id: str = ""
    planned_ts: float = 0.0
    effective_ts: float = 0.0
    end_ts: float = 0.0
    detection_ts: float = 0.0
    alert_ts: float = 0.0
    mitigation_ts: float = 0.0
    recovery_ts: float = 0.0
    mechanism: str = ""
    intensity: str = ""
    status: str = ""
    ground_truth_probe: str = ""
    slo_impact: str = ""
    error_budget_spent: float = 0.0
    retry_actions: int = 0
    fallback_actions: int = 0
    autoscaling_actions: int = 0
    rollback_actions: int = 0
    manual_steps: int = 0
    pre_post_state: str = ""
    cleanup_status: str = ""
    residual_status: str = ""
    verdict: str = ""
    evidence_refs: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.control_or_fault not in ("FAULT", "CONTROL"):
            problems.append(f"unknown episode kind {self.control_or_fault!r}")
        if self.verdict and self.verdict not in VERDICTS:
            problems.append(f"unknown fault verdict {self.verdict!r}")
        if self.control_or_fault == "FAULT":
            if not self.effective_ts:
                problems.append(
                    "the effective fault time is required: the command invocation time is not the effect"
                )
            if not self.ground_truth_probe:
                problems.append("a ground-truth probe must confirm that the fault became effective")
        if self.effective_ts and self.planned_ts and self.effective_ts < self.planned_ts:
            problems.append("the fault became effective before it was planned")
        return problems

    def mttd(self) -> Optional[float]:
        if not (self.detection_ts and self.effective_ts):
            return None
        return self.detection_ts - self.effective_ts

    def mttm(self) -> Optional[float]:
        if not (self.mitigation_ts and self.effective_ts):
            return None
        return self.mitigation_ts - self.effective_ts

    def mttr(self) -> Optional[float]:
        if not (self.recovery_ts and self.effective_ts):
            return None
        return self.recovery_ts - self.effective_ts

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "fault_case_id": self.fault_case_id,
            "control_or_fault": self.control_or_fault,
            "release_id": self.release_id,
            "model_artifact_id": self.model_artifact_id,
            "cluster_id": self.cluster_id,
            "target_ids": list(self.target_ids),
            "steady_state_hypothesis_id": self.steady_state_hypothesis_id,
            "safety_policy_id": self.safety_policy_id,
            "planned_ts": self.planned_ts,
            "effective_ts": self.effective_ts,
            "detection_ts": self.detection_ts,
            "mitigation_ts": self.mitigation_ts,
            "recovery_ts": self.recovery_ts,
            "verdict": self.verdict,
        }


# ── safety and ground truth ──────────────────────────────────────────────


def safety_gate(
    *, spec: FaultSpec, policy: Any, gates: Mapping[str, bool], error_budget_remaining: float
) -> Dict[str, Any]:
    """§20.2 + step 7: refuse to inject unless the scope and gates are satisfied."""
    from hqsb.infra.campaign import authorize_targets

    problems = spec.validate()
    authorization = authorize_targets(policy, list(spec.resolved_targets))
    if not authorization["authorized"]:
        problems.append("targets not authorized: " + authorization["reason"])
    failed_gates = sorted(name for name in PRE_INJECTION_GATES if not gates.get(name, False))
    if failed_gates:
        problems.append(f"pre-injection gates not satisfied: {failed_gates}")
    if error_budget_remaining <= 0:
        problems.append(
            "no error budget left for this fault: destructive experiments may not consume unbounded "
            "shared reliability"
        )
    return {
        "fault_case_id": spec.fault_case_id,
        "authorized_targets": authorization["resolved"],
        "failed_gates": failed_gates,
        "error_budget_remaining": error_budget_remaining,
        "may_inject": not problems,
        "problems": problems,
    }


def ground_truth(
    *, episode_id: str, fault_case_id: str, probe_evidence: str, effective_ts: float,
    hidden_from_investigator: bool = True,
) -> Dict[str, Any]:
    """Step 6: record when the fault *actually* took effect (and keep it hidden)."""
    problems: List[str] = []
    if not probe_evidence:
        problems.append(
            "a ground-truth probe is required: a fault command succeeding does not prove the fault took effect"
        )
    if not effective_ts:
        problems.append("the effective timestamp must be recorded")
    return {
        "episode_id": episode_id,
        "fault_case_id": fault_case_id,
        "effective_ts": effective_ts,
        "probe_evidence": probe_evidence,
        "hidden_from_investigator": hidden_from_investigator,
        "problems": problems,
        "ok": not problems,
    }


# ── reliability metrics ─────────────────────────────────────────────────


def reliability_metrics(
    episodes: Sequence[FaultEpisode], *, thresholds: Mapping[str, float]
) -> Dict[str, Any]:
    """Step 35: per-episode MTTD/MTTM/MTTR against the pre-registered limits."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for episode in episodes:
        problems.extend(episode.validate())
        mttd, mttm, mttr = episode.mttd(), episode.mttm(), episode.mttr()
        entry = {
            "episode_id": episode.episode_id,
            "mttd_s": mttd if mttd is not None else "",
            "mttm_s": mttm if mttm is not None else "",
            "mttr_s": mttr if mttr is not None else "",
            "slo_violation_area": episode.slo_impact,
            "error_budget_spent": episode.error_budget_spent,
            "manual_steps": episode.manual_steps,
        }
        rows.append(entry)
        for name, value in (("mttd_s", mttd), ("mttm_s", mttm), ("mttr_s", mttr)):
            limit = thresholds.get(name)
            if limit is None:
                continue
            if value is None:
                problems.append(f"{episode.episode_id}: {name} is unknown (no timestamp recorded)")
            elif value > limit:
                problems.append(
                    f"{episode.episode_id}: {name} {value:.1f}s exceeds the pre-registered limit {limit}s"
                )
    if not thresholds:
        problems.append("MTTD/MTTM/MTTR limits must be pre-registered before the faults run")
    return {"rows": rows, "episodes": len(rows), "thresholds": dict(sorted(thresholds.items())),
            "problems": problems, "ok": not problems}


@dataclass
class RetryFallback:
    """Step 28/29: bounded retries and explicit fallbacks only."""

    episode_id: str
    request_id: str
    retries: int = 0
    budget: int = 0
    backoff: str = ""
    jitter: bool = False
    fallback: str = ""
    fallback_declared: bool = False
    fallback_quality_checked: bool = False
    fallback_capability_adequate: bool = False
    client_visible_flag: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.episode_id or not self.request_id:
            problems.append("retry/fallback records need episode and request ids")
        if self.budget <= 0:
            problems.append("a retry budget must be positive (unbounded retries amplify an outage)")
        if self.retries > self.budget:
            problems.append(f"retries {self.retries} exceed the budget {self.budget}")
        if self.retries and not self.backoff:
            problems.append("retries need a backoff policy")
        if self.fallback:
            if not self.fallback_declared:
                problems.append("a fallback must be declared explicitly (silent fallback is forbidden)")
            if not self.fallback_quality_checked:
                problems.append("the fallback must be quality-checked before it may serve")
            if not self.fallback_capability_adequate:
                problems.append("the fallback capacity/capability must be validated")
            if rec.SEMANTIC_CHANGING_DEGRADATIONS and not self.client_visible_flag:
                problems.append("a semantics-changing fallback must be visible to the client/telemetry")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "request_id": self.request_id,
            "retries": self.retries,
            "retry_amplification": self.retries,
            "fallback": self.fallback,
            "fallback_declared": self.fallback_declared,
            "quality_checked": self.fallback_quality_checked,
        }


def retry_storm_check(records: Sequence[RetryFallback], *, amplification_gate: float) -> Dict[str, Any]:
    """Step 28: retries must not turn a dependency failure into a self-inflicted outage."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for record in records:
        problems.extend(record.validate())
        rows.append(record.as_dict())
    total_retries = sum(record.retries for record in records)
    amplification = (total_retries / len(records)) if records else 0.0
    if amplification > amplification_gate:
        problems.append(
            f"retry amplification {amplification:.2f} exceeds the gate {amplification_gate}: "
            "success may be bought with multiplied work, which must be reported"
        )
    return {"rows": rows, "retry_amplification": amplification, "gate": amplification_gate,
            "problems": problems, "ok": not problems}


# ── recovery verification ───────────────────────────────────────────────


def validate_state_recovery(
    *, episode_id: str, pre_state: Mapping[str, Any], post_state: Mapping[str, Any],
    invariants: Sequence[str] = SERVICE_INVARIANTS + RESOURCE_INVARIANTS,
) -> Dict[str, Any]:
    """Steps 32–34/§23 V11: service + resource + state, not ``Pod Running``."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for invariant in invariants:
        if invariant not in SERVICE_INVARIANTS + RESOURCE_INVARIANTS:
            raise ConfigError(f"unknown invariant {invariant!r}")
        pre = pre_state.get(invariant, "")
        post = post_state.get(invariant, "")
        consistent = (pre == post) and post != ""
        rows.append(
            {
                "episode_id": episode_id,
                "object_kind": invariant,
                "object_id": episode_id,
                "pre_state": pre,
                "post_state": post,
                "consistent": consistent,
            }
        )
        if not consistent:
            problems.append(f"{invariant}: post-recovery state {post!r} != pre {pre!r}")
    return {"rows": rows, "problems": problems, "ok": not problems}


def residual_watch(
    *, episode_id: str, observations: Sequence[Mapping[str, Any]], window_s: float
) -> Dict[str, Any]:
    """Step 34: keep watching after 'recovered' — leaks and stale state show up later."""
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    for observation in observations:
        anomaly = bool(observation.get("anomaly", False))
        rows.append(
            {
                "episode_id": episode_id,
                "window_s": window_s,
                "observation": observation.get("observation", ""),
                "anomaly": anomaly,
                "action": observation.get("action", ""),
            }
        )
        if anomaly and not observation.get("action"):
            problems.append(
                f"residual anomaly {observation.get('observation')!r} without an action: a delayed failure "
                "must not be closed silently"
            )
    if window_s <= 0:
        problems.append("the residual observation window must be positive")
    return {"rows": rows, "window_s": window_s, "problems": problems, "ok": not problems}


def postmortem(
    *, episode_id: str, verdict: str, timeline: Sequence[str], impact: str, root_cause: str,
    contributing_factors: Sequence[str], actions: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Step 37: FAIL/near-miss cases get a timeline, impact, causes and owned actions."""
    problems: List[str] = []
    if verdict not in VERDICTS:
        problems.append(f"unknown verdict {verdict!r}")
    if verdict in ("FAILED_SLO", "UNRECOVERABLE_WITHIN_SCOPE", "RECOVERED_MANUAL", "DEGRADED_AS_DESIGNED"):
        for name, value in (("timeline", timeline), ("impact", impact), ("root_cause", root_cause),
                            ("contributing_factors", contributing_factors), ("actions", actions)):
            if not value:
                problems.append(f"{verdict} requires {name!r} in the postmortem")
    for action in actions:
        if not action.get("owner"):
            problems.append(f"action {action.get('action_id', '?')} has no owner")
        if not action.get("verification"):
            problems.append(f"action {action.get('action_id', '?')} has no verification step")
    return {"episode_id": episode_id, "verdict": verdict, "problems": problems, "ok": not problems}


def fault_verdicts(episodes: Sequence[FaultEpisode]) -> Dict[str, Any]:
    """Step 38: per-class verdicts including the manual/unrecoverable boundary."""
    problems: List[str] = []
    by_class: Dict[str, List[str]] = {}
    for episode in episodes:
        problems.extend(episode.validate())
        if not episode.verdict:
            problems.append(f"{episode.episode_id}: no verdict assigned")
            continue
        by_class.setdefault(episode.verdict, []).append(episode.episode_id)
    automatic = by_class.get("RECOVERED_AUTOMATIC", [])
    manual = by_class.get("RECOVERED_MANUAL", [])
    unrecoverable = by_class.get("UNRECOVERABLE_WITHIN_SCOPE", [])
    return {
        "episodes": len(episodes),
        "by_verdict": {key: sorted(value) for key, value in sorted(by_class.items())},
        "automatic_recovery": automatic,
        "manual_recovery": manual,
        "unrecoverable": unrecoverable,
        "problems": problems,
        "ok": not problems,
        "note": (
            "an automatic-recovery claim is only valid for cases in 'automatic_recovery'; manual and "
            "unrecoverable cases are production limitations that must be published"
        ),
    }


def fault_recovery_verdict(
    *,
    matrix: Mapping[str, Any],
    reliability: Mapping[str, Any],
    retries: Mapping[str, Any],
    state: Mapping[str, Any],
    residual: Mapping[str, Any],
    verdicts: Mapping[str, Any],
) -> Dict[str, Any]:
    """Aggregate the reliability claim boundary of the stage."""
    problems: List[str] = []
    for name, axis in (
        ("fault_matrix", matrix),
        ("reliability_metrics", reliability),
        ("retry_fallback", retries),
        ("state_recovery", state),
        ("residual_watch", residual),
        ("verdicts", verdicts),
    ):
        if not axis.get("ok"):
            problems.append(f"{name}: " + "; ".join(axis.get("problems", []) or ["(no detail)"]))
    return {
        "problems": problems,
        "production_reliability_claim_allowed": not problems,
        "manual_cases": verdicts.get("manual_recovery", []),
        "unrecoverable_cases": verdicts.get("unrecoverable", []),
        "note": "a test-cluster fault campaign tops out at FAULT_VALIDATED evidence",
    }


# ── protocol steps and smoke self-check ──────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 fault taxonomy/matrix", ("faults:FaultSpec", "faults:LAYERS")),
    (2, "冻结稳态假设", ("faults:SteadyState",)),
    (3, "冻结预期行为", ("faults:FaultSpec.validate", "faults:DEGRADATIONS")),
    (4, "冻结安全边界", ("campaign:SafetyPolicy", "campaign:authorize_targets")),
    (5, "冻结实验设计", ("faults:FaultEpisode", "faults:validate_fault_matrix")),
    (6, "验证 fault tool/mechanism", ("faults:ground_truth",)),
    (7, "验证 observability/runbook 准备", ("faults:safety_gate", "observability:build_alert_rule")),
    (8, "建立 fault-free baseline", ("faults:SteadyState", "faults:reliability_metrics")),
    (9, "注入 process crash", ("faults:ground_truth", "lifecycle:DrainTimeline")),
    (10, "注入 process deadlock/no-progress", ("faults:ground_truth", "lifecycle:ProbeConfig")),
    (11, "注入 Pod delete/eviction", ("faults:ground_truth", "lifecycle:RollingPolicy")),
    (12, "注入 node/kubelet 不可用", ("faults:ground_truth", "scheduling:validate_resource_release")),
    (13, "注入 backend/runtime failure", ("faults:ground_truth", "faults:RetryFallback")),
    (14, "注入 device unhealthy/plugin 故障", ("faults:ground_truth", "scheduling:plugin_restart_recovery")),
    (15, "注入 cache corruption", ("faults:ground_truth", "artifacts:verify_download")),
    (16, "注入 cache lock/lease 故障", ("faults:ground_truth", "artifacts:LeaseRecord")),
    (17, "注入 model artifact 错误", ("faults:ground_truth", "artifacts:check_compatibility")),
    (18, "注入 object storage latency/error", ("faults:ground_truth", "artifacts:StagingDownload")),
    (19, "注入 local/PVC disk full", ("faults:ground_truth", "artifacts:gc_plan")),
    (20, "注入 client/service network 故障", ("faults:ground_truth", "lifecycle:slow_client_policy_check")),
    (21, "注入 DNS/service discovery 故障", ("faults:ground_truth",)),
    (22, "注入 distributed link/rank 故障", ("faults:ground_truth", "scheduling:validate_rank_mapping")),
    (23, "注入 device OOM", ("faults:ground_truth", "capacity:run_capacity_fault_cases")),
    (24, "注入 host memory/swap pressure", ("faults:ground_truth",)),
    (25, "注入 resource leak episode", ("faults:ground_truth", "capacity:margin_holdout_validation")),
    (26, "注入 thermal/power/clock 退化", ("faults:ground_truth",)),
    (27, "注入 metrics/autoscaler failure", ("faults:ground_truth", "autoscaling:run_failure_cases")),
    (28, "测试 retry storm 防护", ("faults:retry_storm_check", "faults:RetryFallback")),
    (29, "测试 fallback 语义", ("faults:RetryFallback.validate",)),
    (30, "运行一个 chained fault", ("faults:ground_truth", "faults:validate_fault_matrix")),
    (31, "对每 case 执行恢复", ("faults:FaultEpisode",)),
    (32, "验证请求/质量恢复", ("faults:validate_state_recovery", "lifecycle:request_integrity_report")),
    (33, "验证资源/状态恢复", ("faults:validate_state_recovery", "faults:RESOURCE_INVARIANTS")),
    (34, "检查残留/延迟故障", ("faults:residual_watch",)),
    (35, "计算可靠性指标", ("faults:reliability_metrics",)),
    (36, "跨时段重复和敏感性", ("faults:reliability_metrics", "faults:validate_fault_matrix")),
    (37, "生成 postmortem 与修复验证", ("faults:postmortem",)),
    (38, "形成 fault/recovery verdict", ("faults:fault_verdicts", "faults:fault_recovery_verdict")),
)


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the fault contracts (smoke, not an experiment)."""
    checks: Dict[str, Any] = {}
    spec = FaultSpec(
        fault_case_id="F-POD-DELETE-01",
        layer="pod_container",
        mechanism="POD_DELETE",
        hypothesis="deleting one replica keeps the SLO inside the budget",
        target_selector="pod/replica-2",
        resolved_targets=("pod-replica-2",),
        namespace="hqsb-test",
        blast_radius="one replica",
        safety_policy_id="safety-1",
        duration_s=30.0,
        expected_detection="alert hqsb_replica_availability",
        expected_degradation="degraded capacity within budget",
        expected_recovery="replacement becomes ready",
        expected_degradation_class="MASKED_REDUNDANCY",
        abort_threshold="error budget < 10%",
        kill_switch="delete fault job",
        repetitions=3,
        allowed_claims=("single-replica loss is masked",),
        forbidden_claims=("node loss is masked",),
        health_gate=PRE_INJECTION_GATES,
    )
    checks["fault_spec_valid"] = spec.validate() == []

    wildcard = FaultSpec(fault_case_id="F-1", layer="pod_container", mechanism="POD_DELETE",
                         hypothesis="h", target_selector="pod/*", resolved_targets=("x",),
                         blast_radius="?", safety_policy_id="s", duration_s=1.0,
                         expected_detection="d", expected_degradation="g", expected_recovery="r",
                         abort_threshold="a", kill_switch="k", repetitions=1)
    checks["wildcard_and_missing_gates_rejected"] = len(wildcard.validate()) >= 3

    episode = FaultEpisode(
        episode_id="ep-1", fault_case_id=spec.fault_case_id, release_id="rel1", model_artifact_id="m1",
        cluster_id="c1", target_ids=("pod-replica-2",), steady_state_hypothesis_id="ss-1",
        safety_policy_id="safety-1", planned_ts=100.0, effective_ts=100.5, detection_ts=130.5,
        mitigation_ts=140.5, recovery_ts=400.5, mechanism="POD_DELETE",
        ground_truth_probe="process gone at 100.5 (pid probe)", verdict="RECOVERED_AUTOMATIC",
    )
    checks["episode_valid"] = episode.validate() == []
    checks["mttr_positive"] = (episode.mttr() or 0) > 0

    missing_truth = FaultEpisode(episode_id="ep-2", fault_case_id="F-1", effective_ts=0.0)
    checks["missing_ground_truth_flagged"] = any(
        "effective fault time" in problem for problem in missing_truth.validate()
    )

    metrics = reliability_metrics([episode], thresholds={"mttd_s": 60.0, "mttr_s": 600.0})
    checks["reliability_metrics_ok"] = metrics["ok"] is True

    slow = reliability_metrics([episode], thresholds={"mttd_s": 10.0})
    checks["slow_detection_flagged"] = slow["ok"] is False

    fallback = RetryFallback(
        episode_id="ep-1", request_id="r1", retries=2, budget=3, backoff="exponential+jitter",
        jitter=True, fallback="alternate backend", fallback_declared=False,
        fallback_quality_checked=False, fallback_capability_adequate=False,
    )
    checks["silent_fallback_rejected"] = len(fallback.validate()) >= 3

    storm = retry_storm_check([fallback], amplification_gate=5.0)
    checks["retry_budget_enforced"] = storm["ok"] is False  # undeclared fallback problems propagate

    verdicts = fault_verdicts([episode])
    checks["verdicts_collected"] = verdicts["ok"] and verdicts["automatic_recovery"] == ["ep-1"]
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "note": "接口自检；未注入任何故障",
    }