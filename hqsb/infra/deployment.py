"""E13-02: clean-environment bootstrap, model verification, cold→ready→first request.

Implements ``details/S13/E13-02_*.md`` §2–§11 as data:

* the three clean levels (L1 clean namespace / L2 clean workload cluster /
  L3 infrastructure bootstrap) and the precondition checklist that decides whether
  the run may call itself "clean";
* the deployment state machine of §3 with explicit ``FAILED_*``/``QUARANTINED``
  outcomes (no skipping verification to become ready);
* the readiness conditions of §4 (a 200 from ``/healthz`` is not readiness);
* the ``DeploymentStageEvent`` record of §10 and the cold/warm stage-time
  aggregates the autoscaling experiment consumes;
* pre-ready traffic exclusion, first-request correctness, negative injections
  (corrupt model, interrupted download, incompatible runtime) and failure cleanup.

Nothing here deploys anything: every function is a contract or a validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.infra import contracts as ct
from hqsb.infra import identity as idn
from hqsb.infra import records as rec

EXPERIMENT_ID = "E13-02"
TITLE = "空集群 Bootstrap、模型校验与 Cold→Ready→首个正确请求"
CLAIM_BOUNDARY = (
    "本实验通过证明自动部署与服务准入正确；不证明 placement 最优、容量安全、生命周期无丢请求、"
    "自动扩缩稳定或故障可恢复（由 E13-03…E13-09 验证）。"
)

SCHEMA_VERSION = "1.0.0"

#: §2 clean levels; the report must name the level it actually achieved.
CLEAN_LEVELS: Tuple[str, ...] = ("L1_CLEAN_NAMESPACE", "L2_CLEAN_WORKLOAD_CLUSTER", "L3_INFRASTRUCTURE_BOOTSTRAP")

#: §3 deployment state machine (a failed step has an explicit FAILED_* state).
DEPLOYMENT_STATES: Tuple[str, ...] = (
    "BASELINE_VERIFIED",
    "INFRA_COMPONENTS_READY",
    "NAMESPACE_POLICY_READY",
    "IMAGE_VERIFIED_PULLED",
    "MODEL_DOWNLOADING",
    "MODEL_VERIFIED",
    "SERVICE_STARTING",
    "MODEL_LOADING_COMPILING",
    "MODEL_WARMING",
    "QUALITY_PROBE_PASSED",
    "ENDPOINT_READY",
    "FIRST_REQUEST_COMPLETED",
    "FAILED_IMAGE_ADMISSION",
    "FAILED_MODEL_VERIFICATION",
    "FAILED_STARTUP",
    "FAILED_COMPATIBILITY",
    "QUARANTINED",
)

DEPLOYMENT_TRANSITIONS: Tuple[Tuple[str, str], ...] = (
    ("BASELINE_VERIFIED", "INFRA_COMPONENTS_READY"),
    ("BASELINE_VERIFIED", "QUARANTINED"),
    ("INFRA_COMPONENTS_READY", "NAMESPACE_POLICY_READY"),
    ("NAMESPACE_POLICY_READY", "IMAGE_VERIFIED_PULLED"),
    ("NAMESPACE_POLICY_READY", "FAILED_IMAGE_ADMISSION"),
    ("IMAGE_VERIFIED_PULLED", "MODEL_DOWNLOADING"),
    ("MODEL_DOWNLOADING", "MODEL_VERIFIED"),
    ("MODEL_DOWNLOADING", "FAILED_MODEL_VERIFICATION"),
    ("MODEL_DOWNLOADING", "QUARANTINED"),
    ("MODEL_VERIFIED", "SERVICE_STARTING"),
    ("MODEL_VERIFIED", "FAILED_COMPATIBILITY"),
    ("SERVICE_STARTING", "MODEL_LOADING_COMPILING"),
    ("SERVICE_STARTING", "FAILED_STARTUP"),
    ("MODEL_LOADING_COMPILING", "MODEL_WARMING"),
    ("MODEL_LOADING_COMPILING", "FAILED_COMPATIBILITY"),
    ("MODEL_LOADING_COMPILING", "FAILED_STARTUP"),
    ("MODEL_WARMING", "QUALITY_PROBE_PASSED"),
    ("MODEL_WARMING", "FAILED_STARTUP"),
    ("QUALITY_PROBE_PASSED", "ENDPOINT_READY"),
    ("ENDPOINT_READY", "FIRST_REQUEST_COMPLETED"),
    ("ENDPOINT_READY", "QUARANTINED"),
    ("FAILED_MODEL_VERIFICATION", "QUARANTINED"),
    ("FAILED_COMPATIBILITY", "QUARANTINED"),
    ("FAILED_STARTUP", "QUARANTINED"),
    ("FAILED_IMAGE_ADMISSION", "QUARANTINED"),
)

#: Stages measured for the cold/warm distribution (§8 step 37).
STAGE_NAMES: Tuple[str, ...] = (
    "schedule",
    "image_pull",
    "artifact_download",
    "artifact_verify",
    "cache_commit",
    "model_load",
    "engine_compile",
    "warmup",
    "quality_probe",
    "ready",
    "first_request",
)

#: Negative injections of steps 32–34.
NEGATIVE_CASES: Tuple[str, ...] = (
    "CORRUPT_MODEL_HASH",
    "MISSING_SHARD",
    "WRONG_TOKENIZER",
    "WRONG_CONFIG",
    "WRONG_PRECISION",
    "DOWNLOAD_INTERRUPTED",
    "SLOW_STORAGE",
    "INCOMPATIBLE_ARCH",
    "INCOMPATIBLE_DRIVER",
    "INCOMPATIBLE_ENGINE_CONTRACT",
    "SUPPLY_CHAIN_GATE_FAILED",
)

#: Objects the failure cleanup of step 35 must account for.
RESIDUAL_KINDS: Tuple[str, ...] = (
    "staging_dir",
    "cache_lock",
    "pod",
    "job",
    "service_endpoint",
    "secret_mount",
    "device_allocation",
    "finalizer",
    "pvc",
)

#: Prerequisite checklist of step 1: what must NOT pre-exist for a clean run.
FORBIDDEN_IMPLICIT_STATE: Tuple[str, ...] = (
    "preinstalled_device_plugin",
    "preinstalled_observability_collector",
    "existing_hqsb_objects",
    "existing_hqsb_secrets",
    "warm_image_cache",
    "warm_model_cache",
    "manually_created_pvc",
    "cluster_admin_kubeconfig_in_pod",
)

DEPLOYMENT_STATE_MACHINE = rec.StateMachine(
    "deployment_bootstrap",
    DEPLOYMENT_STATES,
    DEPLOYMENT_TRANSITIONS,
    terminal_states=("FIRST_REQUEST_COMPLETED", "QUARANTINED"),
)


# ── clean boundary and preconditions ──────────────────────────────────────


@dataclass
class CleanScope:
    """Step 1: what already exists, what the workflow installs, what is forbidden."""

    clean_level: str = ""
    campaign_id: str = ""
    preexisting_components: Tuple[str, ...] = ()
    workflow_installed_components: Tuple[str, ...] = ()
    forbidden_implicit_state: Tuple[str, ...] = FORBIDDEN_IMPLICIT_STATE
    cluster_id: str = ""
    namespace: str = ""
    budget_seconds: int = 0
    budget_cost_units: float = 0.0
    cleanup_policy: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.clean_level not in CLEAN_LEVELS:
            problems.append(f"unknown clean level {self.clean_level!r} (must be one of {list(CLEAN_LEVELS)})")
        for name in ("campaign_id", "cluster_id", "namespace", "cleanup_policy"):
            if not getattr(self, name):
                problems.append(f"clean scope requires {name!r}")
        if self.budget_seconds <= 0:
            problems.append("a deployment budget in seconds is required (no unbounded deploy)")
        overlap = sorted(set(self.preexisting_components) & set(self.workflow_installed_components))
        if overlap:
            problems.append(
                f"components claimed both as pre-existing and workflow-installed: {overlap}"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "clean_level": self.clean_level,
            "campaign_id": self.campaign_id,
            "cluster_id": self.cluster_id,
            "namespace": self.namespace,
            "preexisting_components": list(self.preexisting_components),
            "workflow_installed_components": list(self.workflow_installed_components),
            "forbidden_implicit_state": list(self.forbidden_implicit_state),
            "budget_seconds": self.budget_seconds,
            "budget_cost_units": self.budget_cost_units,
            "cleanup_policy": self.cleanup_policy,
        }


@dataclass
class PreconditionCheck:
    """Step 7: a leftover artifact makes the run *contaminated*, not silently clean."""

    check_id: str
    expected: str = ""
    observed: str = ""
    contaminated: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "deployment_run_id": "",
            "check_id": self.check_id,
            "expected": self.expected,
            "observed": self.observed,
            "contaminated": self.contaminated,
        }


def evaluate_preconditions(
    observations: Mapping[str, Any], *, scope: CleanScope, required_absent: Sequence[str] = ()
) -> Dict[str, Any]:
    """Step 7: verify the clean baseline before deploying anything."""
    problems: List[str] = []
    checks: List[Dict[str, Any]] = []
    for name in required_absent or scope.forbidden_implicit_state:
        observed = observations.get(name, "absent")
        contaminated = str(observed) not in ("absent", "none", "", "False")
        checks.append(
            {"check_id": name, "expected": "absent", "observed": str(observed), "contaminated": contaminated}
        )
    contaminated = [row["check_id"] for row in checks if row["contaminated"]]
    if contaminated:
        problems.append(
            "clean preconditions violated (the run must be reported as contaminated, not cleaned up "
            "silently): " + ", ".join(contaminated)
        )
    problems.extend(scope.validate())
    return {"checks": checks, "contaminated": contaminated, "ok": not problems, "problems": problems,
            "verdict": "CLEAN" if not problems else "CONTAMINATED"}


# ── stage events and the readiness claim ──────────────────────────────────


@dataclass
class DeploymentStageEvent:
    """§10 ``DeploymentStageEvent``: the timeline unit of a deployment run."""

    deployment_run_id: str
    event_id: str
    stage: str
    attempt_id: str = "1"
    release_id: str = ""
    model_artifact_id: str = ""
    cluster_id: str = ""
    namespace: str = ""
    node_id: str = ""
    pod_id: str = ""
    container_id: str = ""
    device_ids: Tuple[str, ...] = ()
    state_from: str = ""
    state_to: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration_s: float = 0.0
    retry: int = 0
    status: str = ""
    reason_code: str = ""
    error_id: str = ""
    bytes_or_objects: int = 0
    cache_state: str = ""
    desired_generation: str = ""
    observed_generation: str = ""
    artifact_refs: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.stage not in STAGE_NAMES and self.state_to not in DEPLOYMENT_STATES:
            problems.append(f"unknown deployment stage {self.stage!r}")
        if self.state_to and self.state_to not in DEPLOYMENT_STATES:
            problems.append(f"unknown deployment state {self.state_to!r}")
        if self.state_from and self.state_to and not DEPLOYMENT_STATE_MACHINE.allowed(self.state_from, self.state_to):
            problems.append(f"illegal deployment transition {self.state_from} -> {self.state_to}")
        if self.cache_state and self.cache_state not in ("cold", "warm", "partial", ""):
            problems.append(f"unknown cache state {self.cache_state!r}")
        if self.duration_s < 0:
            problems.append("negative duration")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "deployment_run_id": self.deployment_run_id,
            "attempt_id": self.attempt_id,
            "event_id": self.event_id,
            "release_id": self.release_id,
            "model_artifact_id": self.model_artifact_id,
            "cluster_id": self.cluster_id,
            "namespace": self.namespace,
            "node_id": self.node_id,
            "pod_id": self.pod_id,
            "container_id": self.container_id,
            "device_ids": list(self.device_ids),
            "stage": self.stage,
            "state_from": self.state_from,
            "state_to": self.state_to,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "retry": self.retry,
            "status": self.status,
            "reason_code": self.reason_code,
            "error_id": self.error_id,
            "bytes_or_objects": self.bytes_or_objects,
            "cache_state": self.cache_state,
            "desired_generation": self.desired_generation,
            "observed_generation": self.observed_generation,
            "artifact_refs": list(self.artifact_refs),
        }


def deployment_timeline(events: Sequence[DeploymentStageEvent]) -> Dict[str, Any]:
    """Validate a recorded path through the §3 state machine and sum the stages."""
    problems: List[str] = []
    for event in events:
        problems.extend(event.validate())
    ordered = sorted(events, key=lambda event: (event.started_at, event.event_id))
    path = [ordered[0].state_from] if ordered and ordered[0].state_from else []
    path.extend(event.state_to for event in ordered if event.state_to)
    walk = DEPLOYMENT_STATE_MACHINE.walk(path) if path else {"ok": True}
    if not walk["ok"]:
        problems.append(
            f"deployment path took an illegal transition at index {walk['index']}: "
            f"{walk['state_from']} -> {walk['state_to']}"
        )
    stage_totals: Dict[str, float] = {}
    for event in events:
        if event.stage in STAGE_NAMES:
            stage_totals[event.stage] = stage_totals.get(event.stage, 0.0) + event.duration_s
    return {
        "events": len(events),
        "path": path,
        "stage_totals": dict(sorted(stage_totals.items())),
        "problems": problems,
        "ok": not problems,
    }


@dataclass
class ReadinessClaim:
    """§4 readiness semantics: nine conditions, not a live TCP port."""

    ready: bool = False
    release_id: str = ""
    model_artifact_id: str = ""
    backend_id: str = ""
    model_state: str = ""
    healthz_status: int = 0
    tokenizer_id: str = ""
    precision: str = ""
    warmup_complete: bool = False
    quality_probe_passed: bool = False
    minimum_capacity_available: bool = False
    release_digest_verified: bool = False
    not_draining: bool = True
    process_alive: bool = True
    engine_kernel_backend_identity_known: bool = False
    tokenizer_config_precision_compatible: bool = False
    endpoint_slice_targets: Tuple[str, ...] = ()

    def as_claim(self) -> Dict[str, Any]:
        payload = {
            "ready": self.ready,
            "release_id": self.release_id,
            "model_artifact_id": self.model_artifact_id,
            "backend_id": self.backend_id,
            "model_state": self.model_state,
            "healthz_status": self.healthz_status,
            "endpoint_slice_targets": list(self.endpoint_slice_targets),
        }
        for condition in rec.READINESS_CONDITIONS:
            if condition == "exact_model_active":
                # Readiness never derives from "the process is up": the model slot must
                # be the ACTIVE, quality-passed artifact of the release.
                payload[condition] = bool(self.ready) and self.model_state == "ACTIVE"
                continue
            payload[condition] = bool(getattr(self, condition, False))
        return payload


def readiness_verdict(claim: ReadinessClaim) -> Dict[str, Any]:
    """Apply §4 plus the §11 readiness invariants (verification before traffic)."""
    payload = claim.as_claim()
    validator = ct.validate_readiness_claim(payload)
    problems = [finding.detail for finding in validator.findings]
    if claim.ready and claim.model_state != "ACTIVE":
        problems.append(
            f"ready=true while model_state={claim.model_state or 'unset'} (only an ACTIVE, "
            "quality-passed model may accept traffic)"
        )
    if claim.ready and not claim.endpoint_slice_targets:
        problems.append("no EndpointSlice target recorded: readiness was never wired to traffic")
    return {
        "ready": claim.ready and not problems,
        "requested_ready": claim.ready,
        "problems": problems,
        "reason_codes": validator.reason_codes,
        "note": "readiness is a semantic claim: process alive + model active + warmup + quality + capacity",
    }


# ── pre-ready traffic exclusion and the first request ─────────────────────


def pre_ready_traffic_exclusion(attempts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 26: no formal request may reach a replica before it is ready."""
    rows: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    for attempt in attempts:
        row = {
            "deployment_run_id": attempt.get("deployment_run_id", ""),
            "probe_id": attempt.get("probe_id", ""),
            "request_id": attempt.get("request_id", ""),
            "reached_pod": bool(attempt.get("reached_pod", False)),
            "pod_ready": bool(attempt.get("pod_ready", False)),
            "status": attempt.get("status", ""),
        }
        expected = {
            "served": row["reached_pod"] and row["pod_ready"],
            "rejected": (not row["reached_pod"]) or row["status"] in ("503", "QUEUED", "REDIRECTED"),
        }
        row["ok"] = expected["served"] or expected["rejected"]
        rows.append(row)
        if not row["ok"]:
            violations.append(row)
    return {
        "rows": rows,
        "attempts": len(rows),
        "violations": violations,
        "ok": not violations,
        "reason": (
            ""
            if not violations
            else "a request was served by a replica before readiness (503/queue/redirect is allowed)"
        ),
    }


@dataclass
class FirstRequestRecord:
    """Step 27–28: a 200 is not a deployment success — the tokens must be right."""

    request_id: str
    release_id: str = ""
    model_artifact_id: str = ""
    backend_id: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    output_hash: str = ""
    reference_hash: str = ""
    ttft_s: float = 0.0
    tpot_s: float = 0.0
    e2e_s: float = 0.0
    status: str = ""
    stream_complete: bool = False
    quality_status: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.release_id or not self.model_artifact_id or not self.backend_id:
            problems.append("the first request must record release/model/actual backend identity")
        if not self.output_hash:
            problems.append("the first request must record the output hash (comparison against reference)")
        if self.status == "COMPLETED" and not self.stream_complete:
            problems.append("a completed streaming request must be complete (no truncated stream is a success)")
        if self.status == "COMPLETED" and self.reference_hash and self.output_hash != self.reference_hash:
            problems.append("the first request output differs from the frozen reference (deployment FAIL)")
        if self.status == "COMPLETED" and self.quality_status not in ("pass", ""):
            problems.append("the first request passed HTTP but failed the quality gate")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "deployment_run_id": "",
            "request_id": self.request_id,
            "release_id": self.release_id,
            "model_artifact_id": self.model_artifact_id,
            "backend_id": self.backend_id,
            "tokens": self.output_tokens,
            "output_hash": self.output_hash,
            "status": self.status,
        }


# ── cold/warm stage statistics and cleanup ───────────────────────────────


def cold_warm_stage_times(
    events: Sequence[DeploymentStageEvent], *, cache_state: str
) -> Dict[str, Any]:
    """Step 30/31/37: cold and warm runs are reported separately, never mixed."""
    if cache_state not in ("cold", "warm"):
        raise ConfigError(
            f"cache_state must be 'cold' or 'warm' (got {cache_state!r}); mixing them hides the cache dependency"
        )
    rows: List[Dict[str, Any]] = []
    for stage in STAGE_NAMES:
        samples = [event.duration_s for event in events if event.stage == stage]
        if not samples:
            continue
        ordered = sorted(samples)
        rows.append(
            {
                "deployment_run_id": events[0].deployment_run_id if events else "",
                "cache_state": cache_state,
                "stage": stage,
                "duration_s": round(sum(ordered) / len(ordered), 6),
                "samples": len(ordered),
            }
        )
    critical_path = max(rows, key=lambda row: row["duration_s"])["stage"] if rows else ""
    return {
        "cache_state": cache_state,
        "rows": rows,
        "time_to_ready_s": sum(row["duration_s"] for row in rows if row["stage"] in STAGE_NAMES[:10]),
        "critical_path_stage": critical_path,
        "note": "autoscaling must use decision→ready capacity, never Pod creation/running",
    }


def evaluate_failure_cleanup(
    *, scope: CleanScope, residuals: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Step 35: after a failed run nothing may stay behind (device/cache/endpoint)."""
    rows: List[Dict[str, Any]] = []
    leaked: List[Dict[str, Any]] = []
    for residual in residuals:
        kind = str(residual.get("kind", ""))
        if kind not in RESIDUAL_KINDS:
            raise ConfigError(f"unknown residual kind {kind!r}")
        row = {
            "deployment_run_id": residual.get("deployment_run_id", ""),
            "kind": kind,
            "object_id": residual.get("object_id", ""),
            "namespace": residual.get("namespace", scope.namespace),
            "released": bool(residual.get("released", False)),
            "reason": residual.get("reason", ""),
        }
        rows.append(row)
        if not row["released"]:
            leaked.append(row)
    return {
        "rows": rows,
        "residuals": len(rows),
        "leaked": leaked,
        "ok": not leaked,
        "reason": (
            ""
            if not leaked
            else "residual objects after a failed deployment: "
            + ", ".join(f"{row['kind']}:{row['object_id']}" for row in leaked)
        ),
    }


def run_negative_deployment_cases(
    cases: Sequence[Mapping[str, Any]], *, scope: CleanScope
) -> Dict[str, Any]:
    """Steps 32–34: every injected failure must fail closed *before* serving."""
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        kind = str(case.get("kind", ""))
        if kind not in NEGATIVE_CASES:
            raise ConfigError(f"unknown deployment negative case {kind!r}")
        observed = str(case.get("observed", "NOT_RUN"))
        rows.append(
            {
                "case_id": str(case.get("case_id", f"deploy-neg-{index:03d}")),
                "kind": kind,
                "expected": str(case.get("expected", "FAIL_CLOSED")),
                "observed": observed,
                "served_traffic": bool(case.get("served_traffic", False)),
                "first_bad_object": case.get("first_bad_object", ""),
                "cleanup_ok": bool(case.get("cleanup_ok", False)),
                "ok": observed in ("FAIL_CLOSED", "FAILED_AS_EXPECTED") and not case.get("served_traffic", False),
            }
        )
    failures = [row for row in rows if not row["ok"]]
    return {
        "rows": rows,
        "cases": len(rows),
        "failures": failures,
        "ok": not failures,
        "scope": scope.clean_level,
        "reason": (
            ""
            if not failures
            else "an injected failure was not fail-closed before traffic: "
            + ", ".join(row["kind"] for row in failures)
        ),
    }


@dataclass
class ManualIntervention:
    """Steps 36/38: any author-time command is a deviation, recorded not hidden."""

    deployment_run_id: str
    actor: str
    command: str
    reason: str = ""
    deviation: bool = True
    at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "deployment_run_id": self.deployment_run_id,
            "actor": self.actor,
            "command": self.command,
            "reason": self.reason,
            "deviation": self.deviation,
        }


def automation_verdict(interventions: Sequence[ManualIntervention]) -> Dict[str, Any]:
    """§1/RQ1: a workflow that needed ad-hoc commands is not automated."""
    deviations = [item for item in interventions if item.deviation]
    return {
        "interventions": len(interventions),
        "deviations": len(deviations),
        "automated": not deviations,
        "reason": (
            ""
            if not deviations
            else "the deployment required manual commands: "
            + "; ".join(f"{item.actor}: {item.command}" for item in deviations)
        ),
    }


def deployment_verdict(
    *,
    scope: CleanScope,
    preconditions: Mapping[str, Any],
    timeline: Mapping[str, Any],
    readiness: Mapping[str, Any],
    pre_ready: Mapping[str, Any],
    first_request: Optional[FirstRequestRecord],
    cleanup: Mapping[str, Any],
    interventions: Sequence[ManualIntervention],
) -> Dict[str, Any]:
    """Step 38: aggregate the clean-deployment verdict (with every blocker named)."""
    problems: List[str] = []
    if not preconditions.get("ok"):
        problems.append("clean preconditions not satisfied")
    if not timeline.get("ok"):
        problems.append("deployment timeline contains illegal transitions")
    if not readiness.get("ready"):
        problems.append("readiness was not reached semantically")
    if not pre_ready.get("ok"):
        problems.append("pre-ready traffic exclusion failed")
    if first_request is None:
        problems.append("no first request recorded")
    else:
        problems.extend(first_request.validate())
    if not cleanup.get("ok"):
        problems.append("failure cleanup left residuals")
    automated = automation_verdict(interventions)["automated"]
    if not automated:
        problems.append("the workflow is not automated (manual commands were needed)")
    return {
        "clean_level": scope.clean_level,
        "problems": problems,
        "verdict": "AUTOMATED" if not problems else "CONDITIONAL_OR_FAILED",
        "automated": automated,
        "note": "a clean *namespace* run may not claim infrastructure bootstrap (L3)",
    }


# ── protocol steps and smoke self-check ──────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 clean level 与验收边界", ("deployment:CleanScope", "deployment:CLEAN_LEVELS")),
    (2, "冻结 IaC/deployment source", ("deployment:IaCSource", "deployment:render_manifests")),
    (3, "冻结 ReleaseBundle 与模型", ("identity:ReleaseBundle", "deployment:bind_release")),
    (4, "冻结 startup/quality/first-request contract", ("deployment:StartupContract",)),
    (5, "冻结环境资源与预算", ("deployment:CleanScope", "campaign:SafetyPolicy")),
    (6, "建立 cluster baseline snapshot", ("deployment:cluster_baseline",)),
    (7, "验证 clean preconditions", ("deployment:evaluate_preconditions",)),
    (8, "创建隔离 namespace/project", ("deployment:NamespacePolicy",)),
    (9, "安装/验证基础组件", ("deployment:InstalledComponent",)),
    (10, "验证 node/device capability", ("scheduling:NodeInventory", "scheduling:validate_allocatable")),
    (11, "应用 storage/cache 资源", ("deployment:StorageRequest",)),
    (12, "注入最小 Secret/Config", ("deployment:SecretReference",)),
    (13, "部署 observability collector", ("observability:SemanticConvention",)),
    (14, "验证 image supply-chain admission", ("supply_chain:admit_deployment_digest",)),
    (15, "创建 HQSB workload/service", ("deployment:render_manifests", "deployment:WorkloadSpec")),
    (16, "记录 scheduling/pull", ("deployment:DeploymentStageEvent",)),
    (17, "启动模型 artifact downloader", ("artifacts:StagingDownload",)),
    (18, "校验下载结果", ("artifacts:verify_download",)),
    (19, "原子提交 verified cache", ("artifacts:atomic_commit",)),
    (20, "启动 service process", ("deployment:DeploymentStageEvent",)),
    (21, "加载模型与 runtime", ("artifacts:load_and_measure",)),
    (22, "执行 compile/engine/graph 初始化", ("artifacts:resolve_engine",)),
    (23, "执行 warmup", ("artifacts:WarmupResult",)),
    (24, "执行 startup correctness/quality probe", ("deployment:QualityProbe",)),
    (25, "验证 readiness transition", ("deployment:readiness_verdict", "deployment:ReadinessClaim")),
    (26, "验证 pre-ready traffic exclusion", ("deployment:pre_ready_traffic_exclusion",)),
    (27, "执行首个正式请求", ("deployment:FirstRequestRecord",)),
    (28, "核对首请求正确性", ("deployment:FirstRequestRecord.validate",)),
    (29, "验证资源/身份闭合", ("deployment:identity_closure",)),
    (30, "执行 warm-cache redeploy", ("deployment:cold_warm_stage_times",)),
    (31, "执行 cold-cache repeat", ("deployment:cold_warm_stage_times",)),
    (32, "注入损坏模型", ("deployment:run_negative_deployment_cases",)),
    (33, "注入下载中断/慢存储", ("deployment:run_negative_deployment_cases", "artifacts:interrupted_download_state")),
    (34, "注入不兼容 runtime/device", ("deployment:run_negative_deployment_cases", "deployment:validate_compatibility")),
    (35, "执行失败 cleanup", ("deployment:evaluate_failure_cleanup", "deployment:RESIDUAL_KINDS")),
    (36, "重复独立 bootstrap", ("deployment:ManualIntervention", "deployment:automation_verdict")),
    (37, "构建阶段时延/瓶颈分析", ("deployment:cold_warm_stage_times", "deployment:deployment_timeline")),
    (38, "生成 clean deployment verdict", ("deployment:deployment_verdict",)),
)


@dataclass
class IaCSource:
    """Step 2: the versioned workflow identity (chart/manifest/operator)."""

    iac_id: str
    tool: str = ""
    version: str = ""
    commit: str = ""
    chart_lock_digest: str = ""
    rendered_manifest_digest: str = ""
    operator_version: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("iac_id", "tool", "version", "commit"):
            if not getattr(self, name):
                problems.append(f"IaC source requires {name!r}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "iac_id": self.iac_id,
            "tool": self.tool,
            "version": self.version,
            "commit": self.commit,
            "chart_lock_digest": self.chart_lock_digest,
            "rendered_manifest_digest": self.rendered_manifest_digest,
            "operator_version": self.operator_version,
        }


@dataclass
class StartupContract:
    """Step 4: stage timeouts, retry and readiness contract for the service."""

    release_id: str
    stage_timeouts_s: Mapping[str, int] = field(default_factory=dict)
    readiness_conditions: Tuple[str, ...] = rec.READINESS_CONDITIONS
    quality_probe_tokens: Tuple[int, ...] = ()
    reference_output_hash: str = ""
    retry_policy: str = ""
    first_request_latency_is_gate: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if set(self.readiness_conditions) != set(rec.READINESS_CONDITIONS):
            problems.append("the readiness contract must cover every §4 condition")
        for stage in STAGE_NAMES:
            if stage not in self.stage_timeouts_s:
                problems.append(f"startup contract has no timeout for stage {stage!r}")
        if not self.quality_probe_tokens:
            problems.append("a quality probe with fixed token ids is required (no 'it answered' check)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "release_id": self.release_id,
            "stage_timeouts_s": dict(sorted(self.stage_timeouts_s.items())),
            "readiness_conditions": list(self.readiness_conditions),
            "quality_probe_tokens": list(self.quality_probe_tokens),
            "reference_output_hash": self.reference_output_hash,
            "retry_policy": self.retry_policy,
            "first_request_latency_is_gate": self.first_request_latency_is_gate,
        }


def validate_compatibility(contract: Mapping[str, Any], observed: Mapping[str, Any]) -> Dict[str, Any]:
    """Step 34: an incompatible runtime must fail fast (never a silent fallback path)."""
    problems: List[str] = []
    for field_name in ("arch", "driver", "runtime", "engine_contract"):
        expected = contract.get(field_name)
        actual = observed.get(field_name)
        if expected and actual and expected != actual:
            problems.append(f"{field_name}: expected {expected}, observed {actual}")
    if observed.get("silent_fallback") and not observed.get("fallback_reason"):
        problems.append("a fallback occurred without a recorded reason (silent fallback is forbidden)")
    return {"compatible": not problems, "problems": problems}


def render_manifests(
    *, release: idn.ReleaseBundle, namespace: str, workload_id: str, config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Step 15: the deployment document set, digest-pinned (no ``latest``)."""
    problems: List[str] = []
    problems.extend(release.validate())
    if config.get("image") and not idn.is_digest(str(config["image"])):
        problems.append("workload image must be digest-pinned")
    payload = {
        "release_id": release.release_id,
        "namespace": namespace,
        "workload_id": workload_id,
        "image_index_digest": release.image_index_digest,
        "image": config.get("image", ""),
        "resource_requests": config.get("resource_requests", {}),
        "security_context": config.get("security_context", {}),
        "probes": config.get("probes", {}),
        "termination_grace_period_seconds": config.get("termination_grace_period_seconds", 0),
        "affinity": config.get("affinity", {}),
        "labels": {
            "release_id": release.release_id,
            "model_artifact_id": release.model_artifact_id,
            "deployment_template": release.deployment_template_digest,
        },
    }
    return {"manifests": payload, "problems": problems, "ok": not problems}


def bind_release(
    *, release: idn.ReleaseBundle, model_artifact_digest: str, tokenizer_digest: str
) -> Dict[str, Any]:
    """Step 3: the release/model binding must agree with the artifact plane."""
    problems: List[str] = []
    if not idn.is_digest(model_artifact_digest):
        problems.append("model artifact must be content-addressed")
    if not idn.is_digest(tokenizer_digest):
        problems.append("tokenizer must be content-addressed")
    if release.model_artifact_id and model_artifact_digest and release.model_artifact_id not in model_artifact_digest:
        problems.append(
            "the release names a model artifact that does not match the downloaded digest "
            "(the deployment would serve an unverified model)"
        )
    return {
        "release_id": release.release_id,
        "model_artifact_digest": model_artifact_digest,
        "tokenizer_digest": tokenizer_digest,
        "ok": not problems,
        "problems": problems,
    }


def cluster_baseline(observations: Mapping[str, Any]) -> Dict[str, Any]:
    """Step 6: control plane/nodes/runtime/drivers/CRDs/storage/network snapshot."""
    required = (
        "control_plane_version",
        "api_versions",
        "nodes",
        "container_runtime",
        "device_plugin_version",
        "storage_classes",
        "network_classes",
        "existing_workloads",
    )
    missing = [name for name in required if not observations.get(name)]
    return {
        "observations": dict(sorted(observations.items())),
        "missing": missing,
        "ok": not missing,
        "reason": "" if not missing else "cluster baseline is incomplete: " + ", ".join(missing),
    }


@dataclass
class NamespacePolicy:
    """Step 8: the isolation objects created by the workflow, not by hand."""

    namespace: str
    labels: Mapping[str, str] = field(default_factory=dict)
    service_accounts: Tuple[str, ...] = ()
    roles: Tuple[str, ...] = ()
    resource_quota: Mapping[str, str] = field(default_factory=dict)
    limit_range: Mapping[str, str] = field(default_factory=dict)
    pod_security_level: str = ""
    network_policies: Tuple[str, ...] = ()
    ownership_metadata: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("namespace", "pod_security_level"):
            if not getattr(self, name):
                problems.append(f"namespace policy requires {name!r}")
        if not self.service_accounts:
            problems.append("a dedicated service account is required")
        if not self.resource_quota:
            problems.append("a ResourceQuota is required (§18 quota boundary)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "labels": dict(sorted(self.labels.items())),
            "service_accounts": list(self.service_accounts),
            "roles": list(self.roles),
            "resource_quota": dict(sorted(self.resource_quota.items())),
            "limit_range": dict(sorted(self.limit_range.items())),
            "pod_security_level": self.pod_security_level,
            "network_policies": list(self.network_policies),
            "ownership_metadata": dict(sorted(self.ownership_metadata.items())),
        }


@dataclass
class InstalledComponent:
    """Step 9: one platform component installed by the clean workflow."""

    name: str
    version: str = ""
    digest: str = ""
    kind: str = ""
    ready: bool = False
    events_ref: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.name:
            problems.append("component name is required")
        if not self.version:
            problems.append(f"component {self.name}: version must be pinned")
        if self.digest and not idn.is_digest(self.digest):
            problems.append(f"component {self.name}: digest {self.digest!r} is not a digest")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "kind": self.kind,
            "ready": self.ready,
            "events_ref": self.events_ref,
        }


@dataclass
class StorageRequest:
    """Step 11: declared storage only — no undeclared hostPath."""

    name: str
    access_mode: str = "ReadWriteOnce"
    size_bytes: int = 0
    storage_class: str = ""
    cache_path: str = ""
    retention_policy: str = ""
    host_path: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("name", "storage_class", "cache_path", "retention_policy"):
            if not getattr(self, name):
                problems.append(f"storage request requires {name!r}")
        if self.host_path:
            problems.append("hostPath volumes are not declared storage; use a PVC/object store")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "access_mode": self.access_mode,
            "size_bytes": self.size_bytes,
            "storage_class": self.storage_class,
            "cache_path": self.cache_path,
            "retention_policy": self.retention_policy,
            "host_path": self.host_path,
        }


@dataclass
class SecretReference:
    """Step 12: which config/secret mechanism provides credentials (values never logged)."""

    name: str
    mechanism: str = ""
    version: str = ""
    content_hash: str = ""
    mounted_as: str = ""
    rotated_by: str = ""
    value_logged: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("name", "mechanism", "content_hash"):
            if not getattr(self, name):
                problems.append(f"secret reference requires {name!r}")
        if self.value_logged:
            problems.append(f"the value of {self.name} must never be written to a log or manifest")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mechanism": self.mechanism,
            "version": self.version,
            "content_hash": self.content_hash,
            "mounted_as": self.mounted_as,
            "rotated_by": self.rotated_by,
            "value_logged": self.value_logged,
        }


@dataclass
class QualityProbe:
    """Step 24: fixed token ids, selected tensors and the actual backend."""

    release_id: str
    model_artifact_id: str
    token_ids: Tuple[int, ...] = ()
    expected_greedy_tokens: Tuple[int, ...] = ()
    observed_greedy_tokens: Tuple[int, ...] = ()
    tolerance: float = 0.0
    actual_backend: str = ""
    passed: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.token_ids:
            problems.append("quality probe needs fixed token ids")
        if not self.actual_backend:
            problems.append("quality probe must record the actual backend")
        if self.expected_greedy_tokens and self.expected_greedy_tokens != self.observed_greedy_tokens:
            problems.append("the startup quality probe produced different greedy tokens than the reference")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "release_id": self.release_id,
            "model_artifact_id": self.model_artifact_id,
            "token_ids": list(self.token_ids),
            "expected_greedy_tokens": list(self.expected_greedy_tokens),
            "observed_greedy_tokens": list(self.observed_greedy_tokens),
            "tolerance": self.tolerance,
            "actual_backend": self.actual_backend,
            "passed": self.passed,
        }


def identity_closure(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 29: pod→node→device and release/model/config ids must never be ``unknown``."""
    unknown: List[str] = []
    for index, record in enumerate(records):
        for field_name, value in record.items():
            if isinstance(value, str) and value.strip().lower() in ("unknown", "", "n/a"):
                unknown.append(f"row{index}.{field_name}")
    return {
        "records": len(records),
        "unknown_fields": unknown,
        "ok": not unknown,
        "reason": "" if not unknown else "identity closure has unknown fields: " + ", ".join(unknown[:10]),
    }


@dataclass
class WorkloadSpec:
    """Step 15: the workload as deployed (release/model/placement binding)."""

    workload_id: str
    release_id: str = ""
    model_artifact_id: str = ""
    replicas: int = 1
    resource_requests: Mapping[str, str] = field(default_factory=dict)
    probe_config_id: str = ""
    placement_plan_id: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "release_id": self.release_id,
            "model_artifact_id": self.model_artifact_id,
            "replicas": self.replicas,
            "resource_requests": dict(sorted(self.resource_requests.items())),
            "probe_config_id": self.probe_config_id,
            "placement_plan_id": self.placement_plan_id,
        }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the deployment contracts (smoke, not an experiment)."""
    checks: Dict[str, Any] = {}
    scope = CleanScope(
        clean_level="L1_CLEAN_NAMESPACE",
        campaign_id="smoke",
        cluster_id="cluster-a",
        namespace="hqsb-smoke",
        budget_seconds=600,
        cleanup_policy="delete-namespace",
    )
    checks["clean_scope_valid"] = scope.validate() == []

    overlapping = CleanScope(
        clean_level="L1_CLEAN_NAMESPACE",
        campaign_id="smoke",
        cluster_id="cluster-a",
        namespace="ns",
        budget_seconds=60,
        cleanup_policy="delete",
        preexisting_components=("device_plugin",),
        workflow_installed_components=("device_plugin",),
    )
    checks["overlap_rejected"] = any("both" in problem for problem in overlapping.validate())

    pre = evaluate_preconditions({"existing_hqsb_objects": "3 pods found"}, scope=scope)
    checks["contamination_detected"] = pre["verdict"] == "CONTAMINATED"

    claim = ReadinessClaim(ready=True, release_id="r1", model_artifact_id="m1", backend_id="cuda",
                           model_state="ACTIVE", release_digest_verified=True, warmup_complete=True,
                           quality_probe_passed=True, minimum_capacity_available=True,
                           engine_kernel_backend_identity_known=True,
                           tokenizer_config_precision_compatible=True)
    checks["readiness_requires_endpoint"] = readiness_verdict(claim)["ready"] is False
    claim.endpoint_slice_targets = ("pod-1",)
    checks["readiness_semantic_ok"] = readiness_verdict(claim)["ready"] is True

    healthz_only = ReadinessClaim(ready=True, healthz_status=200, model_state="ABSENT")
    checks["healthz_is_not_readiness"] = readiness_verdict(healthz_only)["ready"] is False

    served_before_ready = pre_ready_traffic_exclusion(
        [{"probe_id": "p1", "request_id": "req1", "reached_pod": True, "pod_ready": False, "status": "200"}]
    )
    checks["pre_ready_violation_detected"] = served_before_ready["ok"] is False

    residual = evaluate_failure_cleanup(
        scope=scope, residuals=[{"kind": "device_allocation", "object_id": "gpu-0", "released": False}]
    )
    checks["residual_detected"] = residual["ok"] is False
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "note": "接口自检；未连接任何集群，未执行任何部署",
    }
