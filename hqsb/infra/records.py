"""S13 vocabularies: statuses, lifecycle states, fault taxonomy, table schemas.

Everything the eleven S13 experiments may name in a machine-readable table is
defined *here*, so that:

* "not measured" can never be stored as the number ``0`` (:func:`numeric_value`);
* the three state machines of ``details/S13/README.md`` §9 (pod/process, model
  artifact, request) are data with validated transitions instead of prose;
* the evidence ladder of §19 (``DESIGN_ONLY`` … ``PRODUCTION_OBSERVED``) and the
  status propagation of §24 are constants a report can be checked against;
* the failure vocabulary of §26.3 (legal degradations) is explicit, so a
  reduced scope is recorded as ``N/A_BY_ADR``/``NOT_APPLICABLE`` and never as a
  quiet success.

Nothing here runs an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

SCHEMA_VERSION = "1.0.0"

# ── protocol statuses (docs/stage_experiments/README.md §3) ────────────────

STATUS_NOT_STARTED = "NOT_STARTED"
STATUS_RUNNING = "RUNNING"
STATUS_PASS = "PASS"
STATUS_PASS_NEGATIVE = "PASS_NEGATIVE"
STATUS_FAIL = "FAIL"
STATUS_BLOCKED = "BLOCKED"
STATUS_N_A_BY_ADR = "N/A_BY_ADR"

PROTOCOL_STATUSES: Tuple[str, ...] = (
    STATUS_NOT_STARTED,
    STATUS_RUNNING,
    STATUS_PASS,
    STATUS_PASS_NEGATIVE,
    STATUS_FAIL,
    STATUS_BLOCKED,
    STATUS_N_A_BY_ADR,
)

CONCLUSION_STATUSES: Tuple[str, ...] = (
    STATUS_PASS,
    STATUS_PASS_NEGATIVE,
    STATUS_FAIL,
    STATUS_BLOCKED,
    STATUS_N_A_BY_ADR,
)

#: Categories where a negative result may never replace a pass (handbook §3).
NEGATIVE_NOT_ALLOWED_CATEGORIES: Tuple[str, ...] = (
    "security",
    "artifact_compatibility",
    "correctness",
    "data_isolation",
    "quality_gate",
    "supply_chain",
)

#: S13-specific prerequisite states (``details/S13/README.md`` §4).
PREREQ_BLOCKED_PREREQUISITE = "BLOCKED_PREREQUISITE"
PREREQ_NOT_APPLICABLE_CAPABILITY = "NOT_APPLICABLE_CAPABILITY"
PREREQ_NOT_RUN_CLUSTER_UNAVAILABLE = "NOT_RUN_CLUSTER_UNAVAILABLE"
PREREQ_NOT_RUN_PERMISSION_DENIED = "NOT_RUN_PERMISSION_DENIED"
PREREQ_NOT_RUN_SAFETY_BOUNDARY = "NOT_RUN_SAFETY_BOUNDARY"
PREREQ_NOT_RUN_TOOL_UNAVAILABLE = "NOT_RUN_TOOL_UNAVAILABLE"

PREREQUISITE_STATES: Tuple[str, ...] = (
    PREREQ_BLOCKED_PREREQUISITE,
    PREREQ_NOT_APPLICABLE_CAPABILITY,
    PREREQ_NOT_RUN_CLUSTER_UNAVAILABLE,
    PREREQ_NOT_RUN_PERMISSION_DENIED,
    PREREQ_NOT_RUN_SAFETY_BOUNDARY,
    PREREQ_NOT_RUN_TOOL_UNAVAILABLE,
)

# ── evidence ladder (§19) ─────────────────────────────────────────────────

EVIDENCE_DESIGN_ONLY = "DESIGN_ONLY"
EVIDENCE_IMPLEMENTED_UNVERIFIED = "IMPLEMENTED_UNVERIFIED"
EVIDENCE_TESTED_SINGLE_RUN = "TESTED_SINGLE_RUN"
EVIDENCE_REPEATED_CONTROLLED = "REPEATED_CONTROLLED"
EVIDENCE_FAULT_VALIDATED = "FAULT_VALIDATED"
EVIDENCE_INDEPENDENTLY_REPRODUCED = "INDEPENDENTLY_REPRODUCED"
EVIDENCE_PRODUCTION_OBSERVED = "PRODUCTION_OBSERVED"

EVIDENCE_LEVELS: Tuple[str, ...] = (
    EVIDENCE_DESIGN_ONLY,
    EVIDENCE_IMPLEMENTED_UNVERIFIED,
    EVIDENCE_TESTED_SINGLE_RUN,
    EVIDENCE_REPEATED_CONTROLLED,
    EVIDENCE_FAULT_VALIDATED,
    EVIDENCE_INDEPENDENTLY_REPRODUCED,
    EVIDENCE_PRODUCTION_OBSERVED,
)

#: Ordered rank: levels may only be *downgraded*, never silently upgraded.
EVIDENCE_RANK: Mapping[str, int] = {level: index for index, level in enumerate(EVIDENCE_LEVELS)}

#: A test cluster can reach at most ``FAULT_VALIDATED``; the two levels above it
#: require an actual production observation (§19).
MAX_EVIDENCE_WITHOUT_PRODUCTION = EVIDENCE_FAULT_VALIDATED

#: What a level requires, used by :func:`evidence_claim_allowed`.
EVIDENCE_REQUIREMENTS: Mapping[str, str] = {
    EVIDENCE_DESIGN_ONLY: "设计/协议存在，无实现运行证据",
    EVIDENCE_IMPLEMENTED_UNVERIFIED: "实现存在且通过本层自检，但未在目标环境运行",
    EVIDENCE_TESTED_SINGLE_RUN: "目标环境单次运行记录（含 raw）",
    EVIDENCE_REPEATED_CONTROLLED: "多 episode/多时段重复且有对照",
    EVIDENCE_FAULT_VALIDATED: "故障注入/演练验证降级与恢复",
    EVIDENCE_INDEPENDENTLY_REPRODUCED: "第二会话/独立环境复现同一结论",
    EVIDENCE_PRODUCTION_OBSERVED: "真实生产观测（非测试集群）",
}

# ── three state machines (§9) ─────────────────────────────────────────────

POD_LIFECYCLE_STATES: Tuple[str, ...] = (
    "PENDING",
    "CONTAINER_STARTING",
    "STARTUP_PROBE_PENDING",
    "ALIVE_NOT_READY",
    "READY_ACCEPTING",
    "DRAINING_NOT_ACCEPTING",
    "TERMINATING",
    "TERMINATED",
    "FORCED_KILL",
)

POD_LIFECYCLE_TRANSITIONS: Tuple[Tuple[str, str], ...] = (
    ("PENDING", "CONTAINER_STARTING"),
    ("CONTAINER_STARTING", "STARTUP_PROBE_PENDING"),
    ("STARTUP_PROBE_PENDING", "ALIVE_NOT_READY"),
    ("STARTUP_PROBE_PENDING", "TERMINATING"),
    ("ALIVE_NOT_READY", "READY_ACCEPTING"),
    ("ALIVE_NOT_READY", "DRAINING_NOT_ACCEPTING"),
    ("READY_ACCEPTING", "DRAINING_NOT_ACCEPTING"),
    ("READY_ACCEPTING", "TERMINATING"),
    ("DRAINING_NOT_ACCEPTING", "TERMINATING"),
    ("TERMINATING", "TERMINATED"),
    ("TERMINATING", "FORCED_KILL"),
)

ARTIFACT_LIFECYCLE_STATES: Tuple[str, ...] = (
    "ABSENT",
    "STAGING_DOWNLOAD",
    "DOWNLOADED_UNVERIFIED",
    "VERIFYING",
    "VERIFIED_IMMUTABLE",
    "LOADING",
    "ENGINE_COMPILED_OR_RESOLVED",
    "WARMED",
    "QUALITY_CAPACITY_PASSED",
    "ACTIVE",
    "RETIRING",
    "CACHED_INACTIVE",
    "QUARANTINED",
    "GC_ELIGIBLE",
)

ARTIFACT_LIFECYCLE_TRANSITIONS: Tuple[Tuple[str, str], ...] = (
    ("ABSENT", "STAGING_DOWNLOAD"),
    ("STAGING_DOWNLOAD", "DOWNLOADED_UNVERIFIED"),
    ("STAGING_DOWNLOAD", "ABSENT"),
    ("DOWNLOADED_UNVERIFIED", "VERIFYING"),
    ("VERIFYING", "VERIFIED_IMMUTABLE"),
    ("VERIFYING", "QUARANTINED"),
    ("VERIFIED_IMMUTABLE", "LOADING"),
    ("LOADING", "ENGINE_COMPILED_OR_RESOLVED"),
    ("ENGINE_COMPILED_OR_RESOLVED", "WARMED"),
    ("WARMED", "QUALITY_CAPACITY_PASSED"),
    ("QUALITY_CAPACITY_PASSED", "ACTIVE"),
    ("ACTIVE", "RETIRING"),
    ("RETIRING", "CACHED_INACTIVE"),
    ("RETIRING", "QUARANTINED"),
    ("ACTIVE", "QUARANTINED"),
    ("CACHED_INACTIVE", "GC_ELIGIBLE"),
    ("CACHED_INACTIVE", "LOADING"),
    ("QUARANTINED", "ABSENT"),
    ("QUARANTINED", "GC_ELIGIBLE"),
)

#: Only these artifact states may be loaded into the runtime / served (§3 of E13-04).
LOADABLE_ARTIFACT_STATES: Tuple[str, ...] = (
    "VERIFIED_IMMUTABLE",
    "LOADING",
    "ENGINE_COMPILED_OR_RESOLVED",
    "WARMED",
    "QUALITY_CAPACITY_PASSED",
    "ACTIVE",
    "CACHED_INACTIVE",
)
#: Only these may serve traffic.
SERVABLE_ARTIFACT_STATES: Tuple[str, ...] = ("ACTIVE",)

REQUEST_LIFECYCLE_STATES: Tuple[str, ...] = (
    "RECEIVED",
    "AUTHORIZED",
    "ADMITTED",
    "REJECTED",
    "QUEUED",
    "BATCHED_OR_EXECUTING",
    "STREAMING",
    "COMPLETED",
    "CANCELLED",
    "TIMED_OUT",
    "FAILED",
    "ACCOUNTED_RELEASED",
)

REQUEST_LIFECYCLE_TRANSITIONS: Tuple[Tuple[str, str], ...] = (
    ("RECEIVED", "AUTHORIZED"),
    ("RECEIVED", "REJECTED"),
    ("AUTHORIZED", "ADMITTED"),
    ("AUTHORIZED", "REJECTED"),
    ("ADMITTED", "QUEUED"),
    ("ADMITTED", "BATCHED_OR_EXECUTING"),
    ("QUEUED", "BATCHED_OR_EXECUTING"),
    ("QUEUED", "CANCELLED"),
    ("QUEUED", "TIMED_OUT"),
    ("QUEUED", "FAILED"),
    ("BATCHED_OR_EXECUTING", "STREAMING"),
    ("BATCHED_OR_EXECUTING", "COMPLETED"),
    ("BATCHED_OR_EXECUTING", "CANCELLED"),
    ("BATCHED_OR_EXECUTING", "TIMED_OUT"),
    ("BATCHED_OR_EXECUTING", "FAILED"),
    ("STREAMING", "COMPLETED"),
    ("STREAMING", "CANCELLED"),
    ("STREAMING", "TIMED_OUT"),
    ("STREAMING", "FAILED"),
    ("COMPLETED", "ACCOUNTED_RELEASED"),
    ("CANCELLED", "ACCOUNTED_RELEASED"),
    ("TIMED_OUT", "ACCOUNTED_RELEASED"),
    ("FAILED", "ACCOUNTED_RELEASED"),
    ("REJECTED", "ACCOUNTED_RELEASED"),
)

#: Terminal request statuses — exactly one per request (§11 invariant 4 of E13-05).
REQUEST_TERMINAL_STATES: Tuple[str, ...] = (
    "COMPLETED",
    "CANCELLED",
    "TIMED_OUT",
    "FAILED",
    "REJECTED",
)

#: Request completion semantics of E13-05 §4 (declared, not assumed).
REQUEST_COMPLETION_SEMANTICS: Tuple[str, ...] = (
    "COMPLETED_EXACT_VERSION",
    "CANCELLED_EXPLICITLY",
    "RETRYABLE_BEFORE_FIRST_TOKEN",
    "PARTIAL_STREAM_NONRETRYABLE",
    "RESUMABLE_WITH_PROTOCOL_TOKEN",
    "FAILED_FORCED_TERMINATION",
)

# ── readiness semantics (E13-02 §4) ───────────────────────────────────────

READINESS_CONDITIONS: Tuple[str, ...] = (
    "process_alive",
    "release_digest_verified",
    "exact_model_active",
    "tokenizer_config_precision_compatible",
    "engine_kernel_backend_identity_known",
    "warmup_complete",
    "quality_probe_passed",
    "minimum_capacity_available",
    "not_draining",
)

#: Probe kinds must stay separated (E13-05 §2); a single TCP-alive endpoint for
#: all three is explicitly a FAIL condition.
PROBE_KINDS: Tuple[str, ...] = ("startup", "readiness", "liveness")

# ── admission (§12 of README, E13-06 §4) ──────────────────────────────────

ADMISSION_DECISIONS: Tuple[str, ...] = (
    "ADMIT_NOW",
    "QUEUE_BOUNDED",
    "REJECT_RETRYABLE_OVERLOAD",
    "REJECT_INVALID_OR_TOO_LARGE",
    "ROUTE_TO_OTHER_CAPACITY",
    "DEGRADE_EXPLICITLY",
)

#: ``DEGRADE_EXPLICITLY`` is only legal when the degradation is declared and the
#: quality contract still holds (§5 of E13-09).
EXPLICIT_DEGRADATIONS: Tuple[str, ...] = (
    "QUALITY_DECLARED",
    "FEATURE_DISABLED",
    "ROUTE_ALTERNATE_DECLARED",
)

RESOURCE_MEMORY_COMPONENTS: Tuple[str, ...] = (
    "M_weights_resident",
    "M_engine_kernel_graph",
    "M_KV_active",
    "M_workspace_dynamic",
    "M_allocator_fragmentation_reserved",
    "M_runtime_scheduler_cache",
    "M_collective_buffers",
    "M_system_and_safety_reserve",
)

# ── autoscaling (E13-07 §4) ───────────────────────────────────────────────

AUTOSCALING_POLICIES: Tuple[str, ...] = (
    "STATIC_S12_CAPACITY",
    "CPU_ONLY",
    "ACCELERATOR_UTIL",
    "QUEUE_REQUESTS",
    "QUEUE_TOKEN_WORK_OR_AGE",
    "MULTI_SIGNAL_SLO_AWARE",
    "PREDICTIVE_OR_PREWARM",
)

AUTOSCALING_ACTIONS: Tuple[str, ...] = ("SCALE_UP", "SCALE_DOWN", "HOLD", "BLOCKED")

#: Episodes every autoscaling comparison must cover (§9 steps 16–23).
AUTOSCALING_EPISODES: Tuple[str, ...] = (
    "STEADY_LOW",
    "STEADY_HIGH",
    "STEP_UP",
    "SHORT_BURST",
    "RAMP_UP",
    "PERIODIC_BURST",
    "MIXED_LENGTH",
    "STEP_DOWN",
    "REPEATED_BURST",
)

# ── observability (E13-08) ────────────────────────────────────────────────

OBSERVABILITY_LAYERS: Tuple[str, ...] = (
    "client",
    "gateway_auth_quota",
    "admission_router_queue",
    "scheduler_batch",
    "prefill_runtime",
    "decode_runtime",
    "op_kernel_device",
    "stream_write_backpressure",
    "distributed_collective",
    "accelerator_telemetry",
    "system_host",
    "storage_network",
    "control_plane",
)

SIGNAL_CLASSES: Tuple[str, ...] = ("metrics", "logs", "traces", "profiles", "events")

#: Labels that must never appear on a Prometheus metric (unbounded cardinality).
FORBIDDEN_METRIC_LABELS: Tuple[str, ...] = (
    "request_id",
    "trace_id",
    "tenant_id",
    "user_id",
    "session_id",
    "prompt",
    "completion",
    "token_text",
    "model_path",
    "pod_uid",
    "device_uuid",
    "api_key",
)

#: Identity attributes that belong in traces/logs instead of metric labels.
TRACE_ONLY_ATTRIBUTES: Tuple[str, ...] = (
    "request_id",
    "trace_id",
    "tenant_pseudonym",
    "release_id",
    "model_artifact_id",
    "backend_id",
    "fault_id",
    "rollout_id",
)

ALERT_SEVERITIES: Tuple[str, ...] = ("critical", "warning", "info")
ALERT_STATES: Tuple[str, ...] = ("firing", "resolved", "suppressed", "pending")

#: Alert/rule values in the shipped templates are *policy defaults*; the marker is
#: asserted by a test so a policy default can never be mistaken for a measurement.
POLICY_DEFAULT_MARKER = "POLICY_DEFAULT_UNVERIFIED"

# ── fault injection (E13-09 §2, §5) ───────────────────────────────────────

FAULT_LAYERS: Tuple[str, ...] = (
    "process",
    "pod_container",
    "node_kubelet",
    "accelerator_backend",
    "device_plugin",
    "cache_artifact",
    "storage",
    "network_dns",
    "oom_memory",
    "thermal_power",
    "control_plane_dependency",
)

FAULT_MECHANISMS: Tuple[str, ...] = (
    "SIGKILL_PROCESS",
    "DEADLOCK_NO_PROGRESS",
    "POD_DELETE",
    "POD_EVICTION",
    "NODE_UNAVAILABLE",
    "BACKEND_ERROR_INJECTION",
    "DEVICE_UNHEALTHY",
    "PLUGIN_RESTART",
    "CACHE_CORRUPTION",
    "STALE_LOCK",
    "ARTIFACT_SHARD_MISSING",
    "OBJECT_STORE_LATENCY",
    "OBJECT_STORE_ERROR",
    "DISK_FULL_ENOSPC",
    "NETWORK_LATENCY",
    "NETWORK_LOSS",
    "DNS_FAILURE",
    "COLLECTIVE_RANK_FAILURE",
    "DEVICE_OOM",
    "HOST_MEMORY_PRESSURE",
    "RESOURCE_LEAK",
    "THERMAL_THROTTLE",
    "POWER_CAP",
    "METRIC_STALE",
    "CONTROLLER_RESTART",
)

DEGRADATION_STRATEGIES: Tuple[str, ...] = (
    "MASKED_REDUNDANCY",
    "RETRY_BOUNDED",
    "ROUTE_ALTERNATE",
    "ADMISSION_SHED",
    "DEGRADE_EXPLICIT_QUALITY_OR_FEATURE",
    "PAUSE_NOT_READY",
    "ROLLBACK",
    "FAIL_CLOSED",
    "MANUAL_ESCALATION",
    "UNRECOVERABLE_WITHIN_SCOPE",
)

#: Strategies that change user-visible semantics and therefore need an explicit
#: declaration + quality re-check (never a silent fallback).
SEMANTIC_CHANGING_DEGRADATIONS: Tuple[str, ...] = (
    "DEGRADE_EXPLICIT_QUALITY_OR_FEATURE",
    "ROUTE_ALTERNATE",
    "MANUAL_ESCALATION",
    "UNRECOVERABLE_WITHIN_SCOPE",
)

FAULT_VERDICTS: Tuple[str, ...] = (
    "MASKED",
    "DEGRADED_AS_DESIGNED",
    "RECOVERED_AUTOMATIC",
    "RECOVERED_MANUAL",
    "FAILED_SLO",
    "UNRECOVERABLE_WITHIN_SCOPE",
    "NOT_RUN_UNAUTHORIZED",
)

# ── canary (E13-10 §2, §4) ────────────────────────────────────────────────

CANARY_STATES: Tuple[str, ...] = (
    "CANDIDATE_REGISTERED",
    "SUPPLY_CHAIN_PASSED",
    "OFFLINE_CORRECTNESS_QUALITY_PASSED",
    "DEPLOYED_NOT_READY",
    "WARMED_READY_NO_TRAFFIC",
    "SHADOW_OR_SMOKE",
    "CANARY_EXPOSURE_STAGE_1",
    "CANARY_EXPOSURE_STAGE_2",
    "CANARY_EXPOSURE_STAGE_3",
    "PROMOTED",
    "STOPPED_FAILED",
    "ROLLBACK_INITIATED",
    "CONTROL_RESTORED",
    "POSTCHECK",
    "INCONCLUSIVE_HOLD",
    "EXPIRED",
)

CANARY_DECISIONS: Tuple[str, ...] = ("CONTINUE", "PROMOTE", "HOLD", "STOP", "ROLLBACK", "INCONCLUSIVE")

#: Gate hierarchy of §4; a later gate can never compensate an earlier failure.
CANARY_GATES: Tuple[str, ...] = (
    "G0_SUPPLY_CHAIN_COMPATIBILITY",
    "G1_OFFLINE_CORRECTNESS",
    "G2_MODEL_QUALITY_TOKEN_SEMANTICS",
    "G3_STARTUP_READINESS_HEALTH",
    "G4_ONLINE_ERROR_AVAILABILITY_STREAM",
    "G5_LATENCY_SLO_GOODPUT",
    "G6_MEMORY_KV_OOM_LEAK_THERMAL",
    "G7_COST_ENERGY_REPLICA",
    "G8_PROMOTION_POST_DEPLOY",
)

#: Hard gates stop a candidate immediately, without waiting for a performance sample.
CANARY_HARD_GATES: Tuple[str, ...] = (
    "G0_SUPPLY_CHAIN_COMPATIBILITY",
    "G1_OFFLINE_CORRECTNESS",
    "G2_MODEL_QUALITY_TOKEN_SEMANTICS",
    "G3_STARTUP_READINESS_HEALTH",
    "G4_ONLINE_ERROR_AVAILABILITY_STREAM",
)

ASSIGNMENT_UNITS: Tuple[str, ...] = (
    "REQUEST",
    "SESSION_STICKY",
    "TENANT_STRATIFIED",
    "WORKLOAD_BUCKET_STRATIFIED",
    "NODE_DEVICE_BLOCKED",
)

# ── multi-tenancy (E13-11 §6) ─────────────────────────────────────────────

TENANT_INVARIANTS: Tuple[str, ...] = (
    "MT-I01",
    "MT-I02",
    "MT-I03",
    "MT-I04",
    "MT-I05",
    "MT-I06",
    "MT-I07",
    "MT-I08",
    "MT-I09",
    "MT-I10",
    "MT-I11",
    "MT-I12",
)

TENANT_INVARIANT_TEXT: Mapping[str, str] = {
    "MT-I01": "认证身份与租户上下文不可由客户端任意改写",
    "MT-I02": "租户 A 不能读取、列举、修改、删除或执行租户 B 的对象",
    "MT-I03": "普通工作负载不能读取不属于它的 Secret 或自动获得高权限 token",
    "MT-I04": "默认拒绝网络策略下，未显式允许的连接失败",
    "MT-I05": "未验证或未授权的模型、适配器和缓存内容不能进入服务态",
    "MT-I06": "租户资源使用不超过静态配额和动态预算，且并发扣减无竞态突破",
    "MT-I07": "一个租户的过载不造成其他受保护租户无界 SLO 劣化",
    "MT-I08": "日志、指标、Trace、错误响应和诊断包不暴露凭据或跨租户内容",
    "MT-I09": "设备、模型驻留槽位和临时存储归属可审计，不可被其他租户窃取",
    "MT-I10": "权限、策略、配额、发布和拒绝事件具有完整审计链",
    "MT-I11": "凭据撤销或轮换后，旧凭据在规定窗口内失效",
    "MT-I12": "测试结束后资源计数、额度、连接和缓存引用恢复一致",
}

TENANT_SUBJECTS: Tuple[str, ...] = (
    "tenant-a-user",
    "tenant-a-runtime",
    "tenant-b-user",
    "tenant-b-runtime",
    "platform-readonly",
    "release-controller",
    "attacker-untrusted",
    "break-glass-admin",
)

SECURITY_CASE_KINDS: Tuple[str, ...] = (
    "AUTHN",
    "AUTHZ",
    "NETWORK",
    "SECRET_EXPOSURE",
    "ARTIFACT_CACHE",
    "DEVICE_ISOLATION",
    "QUOTA_ACCOUNTING",
    "QUOTA_RACE",
    "ABUSE",
    "NOISY_NEIGHBOR",
    "TELEMETRY_REDACTION",
    "AUDIT",
    "ROTATION_RECOVERY",
)

SECURITY_VERDICTS: Tuple[str, ...] = ("PASS", "FAIL", "INCONCLUSIVE", "NOT_RUN", "INVALID")

#: Security cases are expected to be denied; a successful attack call is a FAIL.
DENY_EXPECTATIONS: Tuple[str, ...] = ("deny", "allow", "fail_closed")

# ── image / model negative-scan categories (E13-01) ───────────────────────

SECRET_SCAN_SURFACES: Tuple[str, ...] = (
    "source_tree",
    "build_context",
    "ci_logs",
    "build_args",
    "image_history",
    "image_layers",
    "image_final_filesystem",
    "config_files",
)

MODEL_FILE_PATTERNS: Tuple[str, ...] = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.gguf",
    "*.onnx",
    "tokenizer.json",
    "*.model",
)

# ── missingness semantics ─────────────────────────────────────────────────

MISSING_NOT_APPLICABLE_CAPABILITY = "NOT_APPLICABLE_CAPABILITY"
MISSING_NOT_RUN_PREREQUISITE = "NOT_RUN_PREREQUISITE"
MISSING_NOT_RUN_TOOL_UNAVAILABLE = "NOT_RUN_TOOL_UNAVAILABLE"
MISSING_NOT_RUN_CLUSTER_UNAVAILABLE = "NOT_RUN_CLUSTER_UNAVAILABLE"
MISSING_NOT_RUN_PERMISSION_DENIED = "NOT_RUN_PERMISSION_DENIED"
MISSING_NOT_RUN_SAFETY_BOUNDARY = "NOT_RUN_SAFETY_BOUNDARY"
MISSING_RUN_FAILED = "RUN_FAILED"
MISSING_QUALITY_GATE_FAILED = "QUALITY_GATE_FAILED"
MISSING_MEASUREMENT_UNAVAILABLE = "MEASUREMENT_UNAVAILABLE"
MISSING_SCANNER_UNAVAILABLE = "SCANNER_UNAVAILABLE"
MISSING_REGISTRY_UNAVAILABLE = "REGISTRY_UNAVAILABLE"
MISSING_LINEAGE_INVALID = "LINEAGE_INVALID"

MISSINGNESS_CODES: Tuple[str, ...] = (
    MISSING_NOT_APPLICABLE_CAPABILITY,
    MISSING_NOT_RUN_PREREQUISITE,
    MISSING_NOT_RUN_TOOL_UNAVAILABLE,
    MISSING_NOT_RUN_CLUSTER_UNAVAILABLE,
    MISSING_NOT_RUN_PERMISSION_DENIED,
    MISSING_NOT_RUN_SAFETY_BOUNDARY,
    MISSING_RUN_FAILED,
    MISSING_QUALITY_GATE_FAILED,
    MISSING_MEASUREMENT_UNAVAILABLE,
    MISSING_SCANNER_UNAVAILABLE,
    MISSING_REGISTRY_UNAVAILABLE,
    MISSING_LINEAGE_INVALID,
)


def numeric_value(value: Any, *, state: str = "") -> Optional[float]:
    """Return a number or refuse: a missing state is never the value ``0``.

    ``None``/``""``/``"N/A"`` and every explicit missingness code are rejected;
    zero itself is a legitimate measurement and passes through unchanged.
    """
    if isinstance(value, str) and value.strip() in MISSINGNESS_CODES:
        raise ConfigError(
            f"missing state {value!r} must not be coerced into a number", details={"field": "value"}
        )
    if state:
        if state not in MISSINGNESS_CODES:
            raise ConfigError(f"unknown missingness state {state!r}")
        raise ConfigError(f"{state} carries no numeric value")
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ConfigError("missing value must be represented by a missingness code, not None/''")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"value {value!r} is not numeric") from exc


# ── state machines ────────────────────────────────────────────────────────


@dataclass
class StateMachine:
    """A validated finite state machine (three of them run in every experiment)."""

    name: str
    states: Tuple[str, ...]
    transitions: Tuple[Tuple[str, str], ...]
    terminal_states: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        unknown = sorted({state for pair in self.transitions for state in pair} - set(self.states))
        if unknown:
            problems.append(f"{self.name}: transitions reference unknown states {unknown}")
        unknown_terminal = sorted(set(self.terminal_states) - set(self.states))
        if unknown_terminal:
            problems.append(f"{self.name}: unknown terminal states {unknown_terminal}")
        if not self.terminal_states:
            problems.append(f"{self.name}: no terminal state declared")
        return problems

    def allowed(self, state_from: str, state_to: str) -> bool:
        return (state_from, state_to) in self.transitions

    def assert_transition(self, state_from: str, state_to: str) -> None:
        if state_from not in self.states:
            raise ConfigError(f"{self.name}: unknown state {state_from!r}")
        if state_to not in self.states:
            raise ConfigError(f"{self.name}: unknown state {state_to!r}")
        if not self.allowed(state_from, state_to):
            raise ConfigError(
                f"{self.name}: illegal transition {state_from} -> {state_to}",
                details={"field": "transition"},
            )

    def walk(self, path: Sequence[str]) -> Dict[str, Any]:
        """Validate a recorded path; returns the first illegal step, if any."""
        for index in range(1, len(path)):
            if not self.allowed(path[index - 1], path[index]):
                return {
                    "ok": False,
                    "index": index,
                    "state_from": path[index - 1],
                    "state_to": path[index],
                    "reason": "illegal transition",
                }
        return {"ok": True, "index": -1, "state_from": "", "state_to": "", "reason": ""}


POD_STATE_MACHINE = StateMachine(
    "pod_lifecycle",
    POD_LIFECYCLE_STATES,
    POD_LIFECYCLE_TRANSITIONS,
    terminal_states=("TERMINATED", "FORCED_KILL"),
)

ARTIFACT_STATE_MACHINE = StateMachine(
    "artifact_lifecycle",
    ARTIFACT_LIFECYCLE_STATES,
    ARTIFACT_LIFECYCLE_TRANSITIONS,
    terminal_states=("GC_ELIGIBLE",),
)

REQUEST_STATE_MACHINE = StateMachine(
    "request_lifecycle",
    REQUEST_LIFECYCLE_STATES,
    REQUEST_LIFECYCLE_TRANSITIONS,
    terminal_states=("ACCOUNTED_RELEASED",),
)

STATE_MACHINES: Mapping[str, StateMachine] = {
    machine.name: machine for machine in (POD_STATE_MACHINE, ARTIFACT_STATE_MACHINE, REQUEST_STATE_MACHINE)
}


def validate_state_machines() -> List[str]:
    problems: List[str] = []
    for machine in STATE_MACHINES.values():
        problems.extend(machine.validate())
    return problems


# ── experiment status propagation (§24) ───────────────────────────────────


@dataclass(frozen=True)
class PropagationRule:
    """If ``experiment_id`` is not PASS, ``blocked_claims`` may not be claimed."""

    experiment_id: str
    status_when_bad: str
    blocked_claims: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status_when_bad": self.status_when_bad,
            "blocked_claims": list(self.blocked_claims),
        }


PROPAGATION_RULES: Tuple[PropagationRule, ...] = (
    PropagationRule(
        "E13-01",
        STATUS_FAIL,
        ("release admitted into a cluster", "supply-chain identity is trustworthy"),
    ),
    PropagationRule(
        "E13-02",
        STATUS_FAIL,
        ("repeatable clean deployment", "deployment automation"),
    ),
    PropagationRule(
        "E13-03",
        STATUS_FAIL,
        ("performance conclusions on the placed device", "multi-tenant isolation",
         "distributed conclusions"),
    ),
    PropagationRule("E13-04", STATUS_FAIL, ("model upgrade", "canary activation", "rollback")),
    PropagationRule(
        "E13-05",
        STATUS_FAIL,
        ("rolling upgrade safety", "canary safety", "fault recovery acceptance"),
    ),
    PropagationRule(
        "E13-06",
        STATUS_FAIL,
        ("autoscaling cannot protect SLO/OOM during cold start",),
    ),
    PropagationRule("E13-07", STATUS_FAIL, ("elasticity", "scale-to-zero")),
    PropagationRule(
        "E13-08",
        STATUS_FAIL,
        ("high-confidence RCA", "automatic release gates"),
    ),
    PropagationRule("E13-09", STATUS_FAIL, ("production reliability claim",)),
    PropagationRule("E13-10", STATUS_FAIL, ("automatic release",)),
    PropagationRule(
        "E13-11",
        STATUS_FAIL,
        ("shared-cluster multi-tenant security",),
    ),
)


def propagate_statuses(statuses: Mapping[str, str]) -> Dict[str, Any]:
    """Apply §24 propagation: a bad upstream experiment blocks the dependent claim."""
    unknown = sorted(set(statuses) - {rule.experiment_id for rule in PROPAGATION_RULES})
    if unknown:
        raise ConfigError(f"unknown experiment ids in status map: {unknown}")
    blocked: List[Dict[str, Any]] = []
    for rule in PROPAGATION_RULES:
        status = statuses.get(rule.experiment_id, STATUS_NOT_STARTED)
        if status in (STATUS_PASS, STATUS_PASS_NEGATIVE):
            continue
        blocked.append(
            {
                "experiment_id": rule.experiment_id,
                "status": status,
                "blocked_claims": list(rule.blocked_claims),
            }
        )
    return {
        "blocked": blocked,
        "blocked_claim_count": sum(len(row["blocked_claims"]) for row in blocked),
        "all_green": not blocked,
    }


# ── table schemas (the machine-readable output of every experiment) ───────

#: ``table name → required fields``; validation refuses a row that drops a field.
TABLE_SCHEMAS: Mapping[str, Tuple[str, ...]] = {
    # E13-01 supply chain
    "release_bundle": (
        "release_id", "source_commit", "image_index_digest", "platform_image_digests",
        "model_artifact_id", "tokenizer_id", "deployment_template_digest", "status",
    ),
    "oci_dag_diff": ("campaign_id", "build_id", "level", "key", "left", "right", "equal"),
    "filesystem_package_diff": ("campaign_id", "path", "state", "left", "right", "left_type", "right_type"),
    "nondeterminism": ("campaign_id", "level", "key", "source", "explanation"),
    "sbom_component": ("sbom_id", "component_id", "name", "version", "supplier", "license", "hash", "relationship"),
    "vulnerability_finding": (
        "finding_id", "component_id", "cve_id", "severity", "cvss", "fix_available",
        "reachability", "status", "evidence_ref",
    ),
    "vulnerability_exception": ("finding_id", "owner", "risk", "compensating_control", "expires_at", "retest_trigger"),
    "secret_finding": ("finding_id", "surface", "pattern", "redacted_value", "action"),
    "license_finding": ("component_id", "license_id", "kind", "obligation", "status"),
    "provenance_statement": ("provenance_id", "builder_id", "subject_digest", "build_definition_hash", "resolved_dependencies", "digest"),
    "attestation_verification": ("attestation_id", "subject_digest", "signer_identity", "issuer", "status", "reason"),
    "runtime_security_context": ("image_digest", "uid", "gid", "read_only_rootfs", "capabilities", "seccomp", "write_paths"),
    "negative_test_result": ("case_id", "kind", "expected", "observed", "gate_decision", "reason"),
    # E13-02 deployment
    "deployment_stage_event": (
        "deployment_run_id", "attempt_id", "event_id", "release_id", "model_artifact_id",
        "cluster_id", "namespace", "stage", "state_from", "state_to", "started_at", "ended_at",
        "status", "reason_code",
    ),
    "precondition_check": ("deployment_run_id", "check_id", "expected", "observed", "contaminated"),
    "pre_ready_attempt": ("deployment_run_id", "probe_id", "request_id", "reached_pod", "pod_ready", "status"),
    "first_request_event": ("deployment_run_id", "request_id", "release_id", "model_artifact_id", "backend_id", "tokens", "output_hash", "status"),
    "cold_warm_stage_time": ("deployment_run_id", "cache_state", "stage", "duration_s", "samples"),
    "residual_inventory": ("deployment_run_id", "kind", "object_id", "namespace", "released", "reason"),
    "manual_intervention": ("deployment_run_id", "actor", "command", "reason", "deviation"),
    # E13-03 scheduling
    "placement_plan": (
        "placement_plan_id", "workload_id", "required_vendor", "required_arch", "min_memory_bytes",
        "device_count", "sharing_mode", "required_node_labels", "topology_policy", "rank_to_device",
    ),
    "capability_label": ("node_id", "label", "value", "source", "owner", "ttl_s", "protected"),
    "scheduler_event": ("placement_run_id", "pod_uid", "phase", "reason", "node_id", "timestamp"),
    "placement_evidence": (
        "placement_run_id", "placement_plan_id", "pod_uid", "requested_resource_and_capability",
        "scheduled_node", "allocated_resource_name", "physical_or_partition_device_ids",
        "runtime_visible_device_ids", "actual_execution_device_ids", "verdict",
    ),
    "topology_cpu_memory_numa": ("placement_run_id", "pod_uid", "cpuset", "cpu_numa", "memory_numa", "device_numa", "link_domain"),
    "unauthorized_access": ("placement_run_id", "from_tenant", "target", "action", "expected", "observed", "verdict"),
    # E13-04 artifacts
    "artifact_lifecycle_event": (
        "event_id", "attempt_id", "artifact_id", "content_root", "cache_location_id",
        "state_from", "state_to", "timestamp", "actor", "generation",
    ),
    "cache_key": ("artifact_id", "content_root", "tokenizer_id", "quant_artifact_id", "engine_artifact_id", "target_abi", "cache_key"),
    "verification_result": ("attempt_id", "file", "expected_hash", "observed_hash", "status", "first_bad_object"),
    "compatibility_result": ("attempt_id", "artifact_id", "check", "expected", "observed", "status"),
    "lease_record": ("lease_id", "artifact_id", "holder", "state", "acquired_at", "expires_at", "released"),
    "activation_generation": ("generation_id", "artifact_id", "linearization_ts", "active", "request_or_session_refs"),
    "request_version_consistency": ("request_id", "generation_id", "artifact_id", "status", "mixed_version"),
    "gc_plan": ("gc_run_id", "artifact_id", "state", "pinned", "eligible", "reason"),
    "rollback_timeline": ("rollback_id", "from_artifact_id", "to_artifact_id", "decision_ts", "routed_ts", "baseline_restored_ts"),
    # E13-05 lifecycle
    "probe_result": ("episode_id", "probe_kind", "endpoint", "result", "reason", "latency_ms"),
    "signal_hook_exit": ("episode_id", "event", "timestamp", "detail"),
    "request_lifecycle_event": (
        "request_id", "trace_id", "tenant_pseudonym", "release_id", "model_artifact_id",
        "state", "timestamp", "status", "error_id", "resource_release_status",
    ),
    "token_integrity": ("request_id", "expected_tokens", "observed_tokens", "prefix_ok", "status", "model_version"),
    "retry_duplicate": ("request_id", "attempt_id", "reason", "duplicate", "retryable", "client_visible"),
    "drain_timeline": ("episode_id", "delete_ts", "prestop_ts", "sigterm_ts", "not_ready_ts", "exit_ts", "forced_kill"),
    "resource_release": ("episode_id", "resource", "released", "residual", "reason"),
    "rolling_availability": ("episode_id", "available_replicas", "ready_replicas", "surge", "unavailable", "slo_ok"),
    # E13-06 capacity
    "memory_ledger": ("run_id", "rank_id", "component", "bytes", "source", "state"),
    "per_request_kv": ("run_id", "request_id", "model_layers", "prompt_tokens", "output_tokens", "kv_bytes", "block_rounding_bytes"),
    "admission_decision": (
        "decision_id", "request_id", "timestamp", "policy_version", "prompt_tokens",
        "requested_output_tokens", "predicted_incremental_kv_bytes", "usable_memory_bytes",
        "safety_margin_bytes", "decision", "reason_code",
    ),
    "admission_policy_comparison": ("policy_id", "accepted", "rejected", "false_admit", "false_reject", "goodput_ratio", "decision_overhead_ms"),
    "prediction_residual": ("run_id", "workload_bucket", "rank_id", "component", "predicted_bytes", "actual_bytes", "residual", "explanation"),
    "false_decision": ("run_id", "request_id", "kind", "counterfactual_method", "confidence", "reason"),
    "margin_validation": ("holdout_id", "margin_bytes", "oom_free", "false_reject_rate", "validated"),
    # E13-07 autoscaling
    "metric_sample": ("sample_id", "metric_name", "value", "unit", "event_ts", "export_ts", "scrape_ts", "age_s"),
    "autoscaling_decision": (
        "decision_id", "episode_id", "timestamp", "policy_version", "metric_name_values",
        "metric_ages", "current_replicas", "ready_replicas", "desired_raw", "desired_stabilized",
        "action", "reason", "controller_generation",
    ),
    "replica_lifecycle": ("episode_id", "pod_uid", "created_ts", "scheduled_ts", "ready_ts", "terminated_ts", "forced"),
    "control_metrics": ("episode_id", "policy_id", "detection_delay_s", "decision_to_ready_s", "overshoot", "oscillation_count", "settling_s"),
    "cost_resource": ("episode_id", "policy_id", "replica_minutes", "device_minutes", "cold_starts", "cache_hit_ratio_state"),
    # E13-08 observability
    "semantic_convention": ("name", "signal_class", "unit", "attributes", "required", "version"),
    "metric_catalog_entry": ("metric_name", "layer", "unit", "labels", "boundary", "sli_or_diagnostic"),
    "cardinality_report": ("metric_name", "series_count", "top_labels", "growth_rate_per_hour", "policy_action"),
    "trace_coverage": ("trace_id", "request_id", "hops_expected", "hops_observed", "unknown_hops", "release_id", "device_id"),
    "alert_rule": ("alert_id", "user_impact", "query", "for_duration", "severity", "owner", "runbook_id", "threshold_source"),
    "alert_result": ("alert_id", "case_id", "detected", "notified", "resolved", "false_positive", "mttd_s"),
    "overhead_ab": ("case_id", "instrumentation_level", "overhead_pct", "tail_impact"),
    "rca_record": (
        "rca_id", "case_id", "investigator_id", "blind_status", "symptom", "started_at",
        "ended_at", "identified_layer", "root_cause", "scope", "confidence", "ground_truth_match",
    ),
    "telemetry_failure": ("case_id", "component", "detection_signal", "detected", "silently_zero"),
    "redaction_scan": ("case_id", "signal_class", "canary_id", "leaked", "location"),
    # E13-09 faults
    "fault_spec": (
        "fault_case_id", "hypothesis", "layer", "mechanism", "target_selector", "resolved_targets",
        "blast_radius", "safety_policy_id", "expected_detection", "expected_degradation",
        "expected_recovery", "abort_threshold",
    ),
    "fault_injection_event": ("episode_id", "fault_case_id", "planned_ts", "effective_ts", "end_ts", "mechanism", "intensity", "status"),
    "fault_ground_truth": ("episode_id", "fault_case_id", "effective_ts", "probe_evidence", "hidden_from_investigator"),
    "reliability_metrics": ("episode_id", "mttd_s", "mttm_s", "mttr_s", "slo_violation_area", "error_budget_spent", "manual_steps"),
    "retry_fallback": ("episode_id", "request_id", "retries", "retry_amplification", "fallback", "fallback_declared", "quality_checked"),
    "state_diff": ("episode_id", "object_kind", "object_id", "pre_state", "post_state", "consistent"),
    "residual_watch": ("episode_id", "window_s", "observation", "anomaly", "action"),
    "postmortem": ("episode_id", "timeline", "impact", "root_cause", "contributing_factors", "actions", "owner"),
    # E13-10 canary
    "canary_candidate_identity": ("candidate_id", "release_id", "kind", "declared_changes", "ground_truth"),
    "canary_transition": ("canary_run_id", "state_from", "state_to", "timestamp", "rule_id", "evidence_count", "reason"),
    "canary_assignment": ("request_id", "canary_run_id", "arm", "unit", "actual_release_id", "actual_backend", "contaminated"),
    "canary_metric": ("canary_run_id", "stage", "metric", "arm", "value", "unit", "sample_count", "interval_low", "interval_high"),
    "canary_decision_row": (
        "decision_id", "canary_run_id", "timestamp", "stage", "control_release_id", "candidate_release_id",
        "traffic_fraction", "information_fraction", "decision", "reason_codes",
    ),
    "canary_gate_result": ("canary_run_id", "stage", "gate_id", "status", "hard_gate", "evidence_count", "reason"),
    "canary_rollback": ("rollback_id", "canary_run_id", "decision_ts", "traffic_stopped_ts", "baseline_restored_ts", "residual_objects"),
    "canary_error_rates": ("canary_run_id", "bad_kind", "detected", "detection_stage", "false_negative", "exposure_at_detection"),
    "override_audit": ("override_id", "actor", "action", "reason", "expires_at", "counts_as_automatic"),
    # E13-11 security
    "threat_model": ("scope_id", "tenant_definition", "attacker_capability", "trusted_components", "protected_assets", "non_goals", "failure_policy"),
    "tenant_identity_map": ("tenant_id", "namespace", "service_account", "api_key_id", "model_ids", "quota_profile"),
    "rbac_graph": ("subject", "role", "verbs", "resources", "scope", "risk", "risk_reason"),
    "security_case": (
        "case_id", "kind", "subject", "authenticated_tenant", "claimed_tenant", "action", "resource",
        "path", "expected", "observed_status", "decision_point", "audit_event_id", "verdict",
    ),
    "quota_ledger": ("tenant_id", "resource", "limit", "reserved", "committed", "released", "balance", "oversell"),
    "quota_race_trial": ("trial_id", "concurrency", "gateway_replicas", "accepted", "rejected", "max_oversell", "final_balance"),
    "abuse_trial": ("trial_id", "kind", "tenant_id", "bounded", "rejected_before_resource", "neighbor_impact"),
    "noisy_neighbor_trial": ("trial_id", "victim_tenant", "attacker_tenant", "metric", "victim_alone", "victim_with_attacker", "interference_ratio", "percentile"),
    "telemetry_redaction": ("case_id", "signal_class", "canary_id", "subject", "leaked", "location"),
    "audit_coverage": ("event_kind", "emitted", "fields_complete", "lag_s", "correlatable"),
    "recovery_state": ("case_id", "resource", "pre_value", "post_value", "consistent", "window_s"),
}

S13_TABLE_SCHEMAS: Mapping[str, Tuple[str, ...]] = TABLE_SCHEMAS


def validate_rows(table: str, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Validate table rows against :data:`TABLE_SCHEMAS`; problems are returned."""
    if table not in TABLE_SCHEMAS:
        raise ConfigError(f"unknown S13 table {table!r}")
    required = TABLE_SCHEMAS[table]
    problems: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        missing = [field for field in required if field not in row]
        if missing:
            problems.append({"row": index, "table": table, "missing_fields": missing})
        for field_name, value in row.items():
            if isinstance(value, str) and value in MISSINGNESS_CODES:
                continue
            if value is None:
                problems.append(
                    {
                        "row": index,
                        "table": table,
                        "missing_fields": [field_name],
                        "reason": "use a missingness code instead of None",
                    }
                )
    return problems


@dataclass
class CoverageReport:
    """Aggregate table coverage for one run (used by the acceptance report)."""

    table_counts: Dict[str, int] = field(default_factory=dict)

    def record(self, table: str, rows: int) -> None:
        if table not in TABLE_SCHEMAS:
            raise ConfigError(f"unknown S13 table {table!r}")
        self.table_counts[table] = self.table_counts.get(table, 0) + rows

    def empty_tables(self) -> List[str]:
        return sorted(name for name in TABLE_SCHEMAS if not self.table_counts.get(name))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tables": len(TABLE_SCHEMAS),
            "populated": len(self.table_counts),
            "table_counts": dict(sorted(self.table_counts.items())),
            "empty_tables": self.empty_tables(),
        }
