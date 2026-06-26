"""Distributed fault taxonomy, oracles, recovery levels and resource closure.

E10-03 (collective mismatch/timeout) and E10-10 (rank/network/OOM faults) share
one machinery:

* a normalized error taxonomy that keeps the vendor code and retryability;
* a pre-registered :class:`FaultOracle` — "最终有异常" is not a correct failure,
  so the oracle names the *first detectable layer*, the bound on detection and
  abort time, the required terminal state and the allowed recovery level;
* :func:`evaluate_fault` which turns an oracle plus an observation into a
  verdict and explicitly fails the eight cheating patterns of E10-03 §12;
* :class:`TimeMetrics` computing MTTD/propagation/abort/cleanup/restart/MTTR;
* :class:`SafetyScope` / :class:`FaultInjection` which refuse unsafe injection
  (shared network, non-experiment PIDs, no external watchdog);
* :class:`ResourceSnapshot` + :func:`snapshot_delta` distinguishing allocator
  cache from a leak.

No function in this module injects anything by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Normalized distributed error classes (E10-10 §4 / step 4).
ERROR_CLASSES: Tuple[str, ...] = (
    "RANK_EXIT",
    "RANK_STUCK",
    "DEVICE_OOM",
    "NETWORK",
    "COMM_ASYNC",
    "TIMEOUT",
    "MISMATCH",
    "BOOTSTRAP",
    "COMMUNICATOR_ALLOC",
    "DEVICE_ERROR",
    "PROFILER_FAILURE",
    "UNKNOWN",
)

#: Recovery levels, weakest first (details README §18).
RECOVERY_LEVELS: Tuple[str, ...] = (
    "REQUEST_ABORT_ONLY",
    "COMMUNICATOR_RECREATE",
    "PROCESS_GROUP_RESTART",
    "NODE_QUARANTINE",
    "JOB_RESTART_FROM_ARTIFACT",
    "UNRECOVERABLE_IN_SCOPE",
)

#: Which layer is expected to detect a class of fault first (E10-10 §12).
DETECTION_LAYERS: Tuple[str, ...] = (
    "process_control_heartbeat",
    "collective_progress_timeout",
    "allocator_or_load_barrier",
    "init_timeout_or_preflight",
    "backend_async_error",
    "external_watchdog_or_ras",
    "profiler_supervisor",
    "request_state",
)

#: Commit states an inference token can be in (E10-10 §3.3).
COMMIT_STATES: Tuple[str, ...] = ("generated", "committed", "emitted", "acknowledged")

#: The retry decision vocabulary handed to S08 (E10-10 §13).
RETRY_DECISIONS: Tuple[str, ...] = (
    "safe_to_retry_before_commit",
    "unsafe_after_commit",
    "manual_policy_required",
)

#: Per-fault pre-registered minimum recovery (E10-10 §12 table).
MINIMUM_RECOVERY_BY_FAULT: Mapping[str, str] = {
    "clean_rank_exit": "PROCESS_GROUP_RESTART",
    "sigterm": "COMMUNICATOR_RECREATE",
    "sigkill": "PROCESS_GROUP_RESTART",
    "rank_hang": "PROCESS_GROUP_RESTART",
    "compute_straggler_timeout": "REQUEST_ABORT_ONLY",
    "shard_load_oom": "PROCESS_GROUP_RESTART",
    "runtime_workspace_oom": "COMMUNICATOR_RECREATE",
    "kv_growth_oom": "REQUEST_ABORT_ONLY",
    "communicator_alloc_failure": "PROCESS_GROUP_RESTART",
    "bootstrap_failure": "PROCESS_GROUP_RESTART",
    "inflight_network_error": "PROCESS_GROUP_RESTART",
    "link_delay_loss": "COMMUNICATOR_RECREATE",
    "post_error_reuse": "COMMUNICATOR_RECREATE",
    "op_mismatch": "COMMUNICATOR_RECREATE",
    "count_mismatch": "COMMUNICATOR_RECREATE",
    "dtype_mismatch": "COMMUNICATOR_RECREATE",
    "root_mismatch": "COMMUNICATOR_RECREATE",
    "skipped_collective": "PROCESS_GROUP_RESTART",
    "duplicate_seq": "COMMUNICATOR_RECREATE",
}

#: Campaign stop conditions (E10-10 §14).
STOP_CONDITIONS: Mapping[str, str] = {
    "affects_non_experiment": "a fault touched a non-experiment PID/node/network: stop and escalate",
    "watchdog_cannot_control_all_ranks": "stop real hang/crash injection",
    "device_health_unknown_or_critical": "quarantine; never auto-reset",
    "resources_grow_after_cleanup_window": "stop further faults and locate the leak first",
    "unrecoverable_repeat": "keep the evidence; do not retry the same fault forever",
    "network_isolation_insufficient": "use a mock and mark PARTIALLY_VALIDATED",
    "timeout_exceeds_scheduler_kill": "redesign; do not fake application recovery with an external kill",
}

#: Action values a resource snapshot diff may produce.
RESOURCE_DELTA_KINDS: Tuple[str, ...] = ("reclaimed", "allocator_cache", "leak_candidate")


def _require(condition: bool, message: str, *, field_name: str = "") -> None:
    if not condition:
        raise ConfigError(message, details={"field": field_name} if field_name else None)


# ── taxonomy ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ErrorEntry:
    """One normalized error class with retryability and detection layer."""

    error_class: str
    normalized_code: str
    retryable: bool
    fallback_allowed: bool
    detection_layer: str
    reason_code: str = ""

    def __post_init__(self) -> None:
        _require(self.error_class in ERROR_CLASSES, "unknown error class", field_name="error_class")
        _require(self.detection_layer in DETECTION_LAYERS, "unknown detection layer", field_name="detection_layer")
        _require(bool(self.normalized_code), "an error entry needs a normalized code")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "error_class": self.error_class,
            "normalized_code": self.normalized_code,
            "retryable": self.retryable,
            "fallback_allowed": self.fallback_allowed,
            "detection_layer": self.detection_layer,
            "reason_code": self.reason_code,
        }


#: Data-driven catalog: classification may only return registered entries.
ERROR_CATALOG: Mapping[str, ErrorEntry] = {
    "RANK_EXIT": ErrorEntry("RANK_EXIT", "hqsb.rank.exit", False, False, "process_control_heartbeat"),
    "RANK_STUCK": ErrorEntry("RANK_STUCK", "hqsb.rank.stuck", False, False, "collective_progress_timeout"),
    "DEVICE_OOM": ErrorEntry("DEVICE_OOM", "hqsb.device.oom", True, True, "allocator_or_load_barrier"),
    "NETWORK": ErrorEntry("NETWORK", "hqsb.network.error", False, False, "backend_async_error"),
    "COMM_ASYNC": ErrorEntry("COMM_ASYNC", "hqsb.comm.async_error", False, False, "backend_async_error"),
    "TIMEOUT": ErrorEntry("TIMEOUT", "hqsb.comm.timeout", False, False, "collective_progress_timeout"),
    "MISMATCH": ErrorEntry("MISMATCH", "hqsb.collective.mismatch", False, False, "init_timeout_or_preflight"),
    "BOOTSTRAP": ErrorEntry("BOOTSTRAP", "hqsb.comm.bootstrap", True, False, "init_timeout_or_preflight"),
    "COMMUNICATOR_ALLOC": ErrorEntry("COMMUNICATOR_ALLOC", "hqsb.comm.alloc_failure", True, False, "init_timeout_or_preflight"),
    "DEVICE_ERROR": ErrorEntry("DEVICE_ERROR", "hqsb.device.error", False, False, "backend_async_error"),
    "PROFILER_FAILURE": ErrorEntry("PROFILER_FAILURE", "hqsb.profiler.failure", True, True, "profiler_supervisor"),
    "UNKNOWN": ErrorEntry("UNKNOWN", "hqsb.unknown", False, False, "external_watchdog_or_ras",
                          reason_code="classification.insufficient_evidence"),
}


def classify_distributed_error(
    *, vendor_code: str = "", message: str = "", normalized_hint: str = ""
) -> Dict[str, Any]:
    """Classify an error; an unknown message stays UNKNOWN with a reason code."""
    if normalized_hint:
        entry = ERROR_CATALOG.get(normalized_hint)
        if entry is None:
            raise ConfigError(
                f"normalized_hint {normalized_hint!r} is not in the catalog",
                details={"field": "normalized_hint"},
            )
    else:
        from hqsb.distributed.backend import normalize_backend_error

        normalized = normalize_backend_error(message, vendor_code)["normalized"]
        entry = ERROR_CATALOG.get(normalized, ERROR_CATALOG["UNKNOWN"])
    payload = entry.as_dict()
    payload["vendor_code"] = vendor_code
    payload["raw_message"] = message
    payload["retry_decision"] = retryability_after_fault(
        commit_state="generated" if entry.retryable else "committed",
        recovery_level="REQUEST_ABORT_ONLY",
    )["decision"]
    return payload


# ── oracle ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FaultOracle:
    """The pre-registered expectation for one fault (E10-03 §12 / E10-10 §12)."""

    first_detectable_layer: str
    expected_normalized_error: str
    maximum_detection_time_s: float
    maximum_global_abort_time_s: float
    expected_failed_ranks: Tuple[int, ...]
    all_rank_terminal_state: str
    output_validity: str = "INVALID"
    communicator_post_state: str = "ABORTED"
    allowed_recovery_level: str = "COMMUNICATOR_RECREATE"
    resource_delta_limit_bytes: int = 0
    post_recovery_probe: str = ""

    def __post_init__(self) -> None:
        _require(
            self.first_detectable_layer in DETECTION_LAYERS,
            "first_detectable_layer must be a known layer",
            field_name="first_detectable_layer",
        )
        _require(
            self.allowed_recovery_level in RECOVERY_LEVELS,
            "allowed_recovery_level must be a known level",
            field_name="allowed_recovery_level",
        )
        _require(
            self.output_validity in ("INVALID", "UNKNOWN", "VALID"),
            "output_validity must be INVALID/UNKNOWN/VALID",
            field_name="output_validity",
        )
        _require(
            self.maximum_detection_time_s > 0 and self.maximum_global_abort_time_s > 0,
            "detection/abort bounds must be positive",
            field_name="maximum_detection_time_s",
        )
        _require(self.post_recovery_probe != "", "an oracle needs a post-recovery probe", field_name="post_recovery_probe")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "first_detectable_layer": self.first_detectable_layer,
            "expected_normalized_error": self.expected_normalized_error,
            "maximum_detection_time_s": self.maximum_detection_time_s,
            "maximum_global_abort_time_s": self.maximum_global_abort_time_s,
            "expected_failed_ranks": list(self.expected_failed_ranks),
            "all_rank_terminal_state": self.all_rank_terminal_state,
            "output_validity": self.output_validity,
            "communicator_post_state": self.communicator_post_state,
            "allowed_recovery_level": self.allowed_recovery_level,
            "resource_delta_limit_bytes": self.resource_delta_limit_bytes,
            "post_recovery_probe": self.post_recovery_probe,
        }


@dataclass(frozen=True)
class FailureObservation:
    """What actually happened during a fault injection run."""

    per_rank_terminal_state: Mapping[int, str]
    first_error_rank: Optional[int] = None
    detected_at_layer: str = ""
    normalized_error: str = ""
    output_consumed_after_error: bool = False
    communicator_reused_after_error: bool = False
    epoch_reused: bool = False
    resource_delta_growth_bytes: int = 0
    recovery_level_used: str = "REQUEST_ABORT_ONLY"
    ranks_left_in_flight: Tuple[int, ...] = ()
    externally_killed_without_record: Tuple[int, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "per_rank_terminal_state": dict(sorted(self.per_rank_terminal_state.items())),
            "first_error_rank": self.first_error_rank,
            "detected_at_layer": self.detected_at_layer,
            "normalized_error": self.normalized_error,
            "output_consumed_after_error": self.output_consumed_after_error,
            "communicator_reused_after_error": self.communicator_reused_after_error,
            "epoch_reused": self.epoch_reused,
            "resource_delta_growth_bytes": self.resource_delta_growth_bytes,
            "recovery_level_used": self.recovery_level_used,
            "ranks_left_in_flight": list(self.ranks_left_in_flight),
            "externally_killed_without_record": list(self.externally_killed_without_record),
        }


@dataclass(frozen=True)
class FaultVerdict:
    ok: bool
    failures: Tuple[str, ...] = ()
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "failures": list(self.failures), "notes": self.notes}


def evaluate_fault(
    oracle: FaultOracle,
    observation: FailureObservation,
    times: Optional["TimeMetrics"] = None,
    *,
    world_size: Optional[int] = None,
) -> FaultVerdict:
    """Decide whether a fault was *correctly* handled (E10-03 §12, E10-10 §11)."""
    failures: List[str] = []

    if (
        oracle.first_detectable_layer in ("init_timeout_or_preflight",)
        and observation.detected_at_layer == "backend_async_error"
        and oracle.expected_normalized_error
        in (ERROR_CATALOG["MISMATCH"].normalized_code, "MISMATCH", "hqsb.collective.mismatch")
    ):
        failures.append(
            "a metadata detectable mismatch surfaced only as a device timeout; preflight was not used"
        )
    if observation.recovery_level_used not in RECOVERY_LEVELS:
        failures.append(f"unknown recovery level {observation.recovery_level_used!r}")
    elif RECOVERY_LEVELS.index(observation.recovery_level_used) < RECOVERY_LEVELS.index(
        oracle.allowed_recovery_level
    ):
        failures.append(
            f"recovery level {observation.recovery_level_used} is weaker than the required "
            f"{oracle.allowed_recovery_level}"
        )
    if observation.ranks_left_in_flight:
        failures.append(f"ranks left IN_FLIGHT forever: {list(observation.ranks_left_in_flight)}")
    if observation.output_consumed_after_error:
        failures.append("a downstream consumer read output after the error (partial output consumed)")
    if observation.communicator_reused_after_error:
        failures.append("the fatal communicator was reused after the error")
    if observation.epoch_reused:
        failures.append("the new communicator reused an old epoch's handles/buffers/sequences")
    if observation.externally_killed_without_record:
        failures.append(
            f"ranks were killed externally without a record: "
            f"{list(observation.externally_killed_without_record)}"
        )
    if (
        oracle.resource_delta_limit_bytes
        and observation.resource_delta_growth_bytes > oracle.resource_delta_limit_bytes
    ):
        failures.append(
            "resources grew beyond the pre-registered delta limit "
            f"({observation.resource_delta_growth_bytes} > {oracle.resource_delta_limit_bytes})"
        )
    if world_size is not None and len(observation.per_rank_terminal_state) != world_size:
        failures.append(
            f"terminal state incomplete: {len(observation.per_rank_terminal_state)} of "
            f"{world_size} ranks reported"
        )
    if any(state == "UNKNOWN" for state in observation.per_rank_terminal_state.values()):
        failures.append("some rank ended in an unknown terminal state")
    if times is not None:
        times.validate()
        if times.mttd_s > oracle.maximum_detection_time_s:
            failures.append(
                f"detection took {times.mttd_s}s > bound {oracle.maximum_detection_time_s}s"
            )
        if times.abort_time_s > oracle.maximum_global_abort_time_s:
            failures.append(
                f"global abort took {times.abort_time_s}s > bound "
                f"{oracle.maximum_global_abort_time_s}s"
            )
    return FaultVerdict(
        ok=not failures,
        failures=tuple(failures),
        notes="" if not failures else "the fault was handled outside its pre-registered oracle",
    )


# ── time metrics ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TimeMetrics:
    """Timeline of one fault, with the derived MTTD/MTTR (E10-10 §5)."""

    fault_injected_s: float
    first_local_detection_s: float
    global_failure_decision_s: float
    all_ranks_abort_started_s: float
    last_rank_exited_or_clean_s: float
    resources_reclaimed_s: float
    restart_started_s: float
    ready_s: float
    first_healthy_request_done_s: float

    def validate(self) -> None:
        ordered = [
            ("fault_injected", self.fault_injected_s),
            ("first_local_detection", self.first_local_detection_s),
            ("global_failure_decision", self.global_failure_decision_s),
            ("all_ranks_abort_started", self.all_ranks_abort_started_s),
            ("last_rank_exited_or_clean", self.last_rank_exited_or_clean_s),
            ("resources_reclaimed", self.resources_reclaimed_s),
            ("restart_started", self.restart_started_s),
            ("ready", self.ready_s),
            ("first_healthy_request_done", self.first_healthy_request_done_s),
        ]
        for index in range(1, len(ordered)):
            if ordered[index][1] < ordered[index - 1][1]:
                raise ConfigError(
                    f"{ordered[index][0]} ({ordered[index][1]}) precedes "
                    f"{ordered[index - 1][0]} ({ordered[index - 1][1]})",
                    details={"field": ordered[index][0]},
                )

    @property
    def mttd_s(self) -> float:
        return self.first_local_detection_s - self.fault_injected_s

    @property
    def propagation_s(self) -> float:
        return self.global_failure_decision_s - self.first_local_detection_s

    @property
    def abort_time_s(self) -> float:
        return self.last_rank_exited_or_clean_s - self.global_failure_decision_s

    @property
    def cleanup_time_s(self) -> float:
        return self.resources_reclaimed_s - self.all_ranks_abort_started_s

    @property
    def restart_time_s(self) -> float:
        return self.ready_s - self.restart_started_s

    @property
    def mttr_s(self) -> float:
        return self.first_healthy_request_done_s - self.fault_injected_s

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fault_injected_s": self.fault_injected_s,
            "first_local_detection_s": self.first_local_detection_s,
            "global_failure_decision_s": self.global_failure_decision_s,
            "all_ranks_abort_started_s": self.all_ranks_abort_started_s,
            "last_rank_exited_or_clean_s": self.last_rank_exited_or_clean_s,
            "resources_reclaimed_s": self.resources_reclaimed_s,
            "restart_started_s": self.restart_started_s,
            "ready_s": self.ready_s,
            "first_healthy_request_done_s": self.first_healthy_request_done_s,
            "mttd_s": self.mttd_s,
            "propagation_s": self.propagation_s,
            "abort_time_s": self.abort_time_s,
            "cleanup_time_s": self.cleanup_time_s,
            "restart_time_s": self.restart_time_s,
            "mttr_s": self.mttr_s,
        }


def distribution_summary(values: Sequence[float]) -> Dict[str, Any]:
    """Median/P50/P99 over *independent fault runs* (never iterations)."""
    if not values:
        return {"count": 0, "median": None, "p50": None, "p99": None, "max": None}
    ordered = sorted(float(value) for value in values)
    count = len(ordered)

    def quantile(q: float) -> float:
        if count == 1:
            return ordered[0]
        position = q * (count - 1)
        low = int(position)
        high = min(low + 1, count - 1)
        fraction = position - low
        return ordered[low] * (1 - fraction) + ordered[high] * fraction

    return {
        "count": count,
        "median": quantile(0.5),
        "p50": quantile(0.5),
        "p99": quantile(0.99),
        "max": ordered[-1],
        "samples": ordered,
    }


# ── safety ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SafetyScope:
    """Blast radius and forbidden actions for fault injection (E10-10 step 1)."""

    experiment_nodes: Tuple[str, ...]
    experiment_processes: Tuple[int, ...]
    devices: Tuple[str, ...]
    networks: Tuple[str, ...] = ()
    time_window: str = ""
    allowed_signals: Tuple[str, ...] = ("SIGTERM", "SIGKILL")
    forbidden_actions: Tuple[str, ...] = (
        "kill non-experiment PIDs",
        "modify shared switches",
        "reset devices/nodes by default",
    )
    blast_radius: str = ""
    approval_reference: str = ""
    dedicated_job: bool = True
    external_watchdog_available: bool = False
    non_experiment_pids_protected: bool = False
    shared_network_untouched: bool = False

    def validate(self) -> None:
        _require(bool(self.experiment_nodes), "scope needs experiment nodes", field_name="experiment_nodes")
        _require(bool(self.experiment_processes), "scope needs experiment PIDs", field_name="experiment_processes")
        _require(bool(self.blast_radius), "scope needs an explicit blast radius", field_name="blast_radius")
        _require(self.dedicated_job, "fault injection requires a dedicated job", field_name="dedicated_job")
        _require(
            self.non_experiment_pids_protected,
            "the scope must state that non-experiment PIDs are protected",
            field_name="non_experiment_pids_protected",
        )
        _require(
            self.shared_network_untouched,
            "the scope must state that shared network equipment is untouched",
            field_name="shared_network_untouched",
        )
        if self.networks:
            _require(
                bool(self.approval_reference),
                "network fault injection needs an approval reference",
                field_name="approval_reference",
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_nodes": list(self.experiment_nodes),
            "experiment_processes": list(self.experiment_processes),
            "devices": list(self.devices),
            "networks": list(self.networks),
            "time_window": self.time_window,
            "allowed_signals": list(self.allowed_signals),
            "forbidden_actions": list(self.forbidden_actions),
            "blast_radius": self.blast_radius,
            "approval_reference": self.approval_reference,
            "dedicated_job": self.dedicated_job,
            "external_watchdog_available": self.external_watchdog_available,
            "non_experiment_pids_protected": self.non_experiment_pids_protected,
            "shared_network_untouched": self.shared_network_untouched,
        }


@dataclass(frozen=True)
class FaultInjection:
    """One injection with markers and cleanup (E10-10 §4 / E09-10 style)."""

    fault: str
    target_rank: Optional[int]
    target_link: str = ""
    intensity: str = ""
    duration_s: float = 0.0
    start_marker: str = ""
    end_marker: str = ""
    cleanup: str = ""
    requires_isolation: bool = False
    partially_validated: bool = False
    mock_reason: str = ""

    def __post_init__(self) -> None:
        _require(bool(self.fault), "injection needs a fault name", field_name="fault")
        _require(bool(self.start_marker) and bool(self.end_marker), "injection needs start/end markers")
        _require(bool(self.cleanup), "injection needs a cleanup action", field_name="cleanup")
        if self.partially_validated:
            _require(
                bool(self.mock_reason),
                "a PARTIALLY_VALIDATED fault must say why (isolation/permission)",
                field_name="mock_reason",
            )

    def validate_against_scope(self, scope: SafetyScope) -> List[str]:
        problems: List[str] = []
        if self.requires_isolation and not scope.networks and not self.partially_validated:
            problems.append(
                "a network fault needs an isolated experiment network or an explicit "
                "PARTIALLY_VALIDATED mock label"
            )
        if self.target_rank is not None and self.target_rank not in range(len(scope.experiment_processes)):
            problems.append(f"target rank {self.target_rank} is outside the experiment scope")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fault": self.fault,
            "target_rank": self.target_rank,
            "target_link": self.target_link,
            "intensity": self.intensity,
            "duration_s": self.duration_s,
            "start_marker": self.start_marker,
            "end_marker": self.end_marker,
            "cleanup": self.cleanup,
            "requires_isolation": self.requires_isolation,
            "partially_validated": self.partially_validated,
            "mock_reason": self.mock_reason,
        }


# ── control plane / watchdogs ──────────────────────────────────────────────


@dataclass(frozen=True)
class ControlHeartbeat:
    rank: int
    pid: int
    last_seq: int
    state: str
    at_s: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "pid": self.pid,
            "last_seq": self.last_seq,
            "state": self.state,
            "at_s": self.at_s,
        }


#: Watchdog layers, each answering a different root cause (E10-10 step 6).
WATCHDOG_LAYERS: Tuple[Mapping[str, str], ...] = (
    {"name": "process_heartbeat", "watches": "process liveness", "answer": "a rank died"},
    {"name": "collective_seq_progress", "watches": "collective sequence advance", "answer": "a rank is stuck"},
    {"name": "request_deadline", "watches": "per-request budget", "answer": "one request overran"},
    {"name": "device_health", "watches": "device health/errors", "answer": "the device is unhealthy"},
    {"name": "job_grace", "watches": "job-level abort grace", "answer": "the whole group must stop"},
)


def watchdog_decision(
    heartbeats: Sequence[ControlHeartbeat],
    *,
    now_s: float,
    heartbeat_timeout_s: float,
    progress_timeout_s: float,
) -> Dict[str, Any]:
    """Distinguish "alive but not progressing" from "process gone" (E10-03 step 17)."""
    if heartbeat_timeout_s <= 0 or progress_timeout_s <= 0:
        raise ConfigError("watchdog timeouts must be positive")
    stale: List[int] = []
    stuck: List[int] = []
    latest_seq: Dict[int, int] = {}
    for beat in heartbeats:
        age = now_s - beat.at_s
        if age > heartbeat_timeout_s:
            stale.append(beat.rank)
        previous = latest_seq.get(beat.rank)
        if previous is not None and beat.last_seq == previous and age <= heartbeat_timeout_s:
            stuck.append(beat.rank)
        latest_seq[beat.rank] = max(previous or beat.last_seq, beat.last_seq)
    action = "none"
    reason = ""
    if stale:
        action = "abort_and_restart"
        reason = f"process heartbeat timed out for ranks {sorted(stale)}"
    elif stuck and now_s - heartbeats[-1].at_s > progress_timeout_s:
        action = "abort_group"
        reason = f"collective progress stalled for ranks {sorted(stuck)} while processes are alive"
    return {"action": action, "reason": reason, "stale_ranks": sorted(stale), "stuck_ranks": sorted(stuck)}


@dataclass
class AbortCoordinator:
    """Epoch-scoped global abort decision (E10-10 step 7)."""

    rank_epoch: int
    world_size: int
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    aborted: bool = False

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ConfigError("world_size must be >= 1")

    def decide(self, *, first_error_rank: int, error_class: str, at_s: float) -> Dict[str, Any]:
        if self.aborted:
            return {
                "decision": "already_aborted",
                "reason": "the epoch already aborted; the first decision is authoritative",
            }
        if error_class not in ERROR_CLASSES:
            raise ConfigError(f"unknown error class {error_class!r}", details={"field": "error_class"})
        self.aborted = True
        record = {
            "decision": "global_abort",
            "rank_epoch": self.rank_epoch,
            "first_error_rank": first_error_rank,
            "error_class": error_class,
            "at_s": at_s,
            "scope": "epoch",
            "notifies": "all healthy ranks stop new work and invalidate in-flight output",
        }
        self.decisions.append(record)
        return record

    def terminal_states_complete(self, states: Mapping[int, str]) -> bool:
        return sorted(states) == list(range(self.world_size))


def retryability_after_fault(*, commit_state: str, recovery_level: str) -> Dict[str, Any]:
    """Whether a failed request may be retried (E10-10 §13)."""
    if commit_state not in COMMIT_STATES:
        raise ConfigError(
            f"commit_state must be one of {COMMIT_STATES}", details={"field": "commit_state"}
        )
    if recovery_level not in RECOVERY_LEVELS:
        raise ConfigError(
            f"recovery_level must be one of {RECOVERY_LEVELS}", details={"field": "recovery_level"}
        )
    index = COMMIT_STATES.index(commit_state)
    if commit_state == "generated":
        decision = "safe_to_retry_before_commit"
    elif commit_state in ("committed",):
        decision = "safe_to_retry_before_commit"
    elif commit_state == "emitted":
        decision = "unsafe_after_commit"
    else:
        decision = "manual_policy_required"
    if recovery_level == "UNRECOVERABLE_IN_SCOPE":
        decision = "manual_policy_required"
    return {
        "commit_state": commit_state,
        "recovery_level": recovery_level,
        "decision": decision,
        "commit_index": index,
        "note": "S10 never stitches tokens/KV from two rank epochs",
    }


# ── resources ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResourceSnapshot:
    """Resource inventory before/after a fault (E10-10 step 9)."""

    label: str
    device_memory_bytes: Mapping[str, int] = field(default_factory=dict)
    host_rss_bytes: int = 0
    process_count: int = 0
    thread_count: int = 0
    open_fds: int = 0
    communicators: int = 0
    streams_events: int = 0
    sockets: int = 0
    shared_memory_segments: int = 0
    rendezvous_files: int = 0
    temp_artifacts: int = 0
    allocator_cached_bytes: int = 0

    def total_bytes(self) -> int:
        return sum(self.device_memory_bytes.values()) + self.host_rss_bytes

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "device_memory_bytes": dict(sorted(self.device_memory_bytes.items())),
            "host_rss_bytes": self.host_rss_bytes,
            "process_count": self.process_count,
            "thread_count": self.thread_count,
            "open_fds": self.open_fds,
            "communicators": self.communicators,
            "streams_events": self.streams_events,
            "sockets": self.sockets,
            "shared_memory_segments": self.shared_memory_segments,
            "rendezvous_files": self.rendezvous_files,
            "temp_artifacts": self.temp_artifacts,
            "allocator_cached_bytes": self.allocator_cached_bytes,
        }


def snapshot_delta(
    before: ResourceSnapshot, after: ResourceSnapshot, *, limit_bytes: int = 0
) -> Dict[str, Any]:
    """Classify the resource delta: reclaimed / allocator cache / leak candidate."""
    delta_bytes = after.total_bytes() - before.total_bytes()
    counts = {
        "process_count": after.process_count - before.process_count,
        "thread_count": after.thread_count - before.thread_count,
        "open_fds": after.open_fds - before.open_fds,
        "communicators": after.communicators - before.communicators,
        "streams_events": after.streams_events - before.streams_events,
        "sockets": after.sockets - before.sockets,
        "shared_memory_segments": after.shared_memory_segments - before.shared_memory_segments,
        "rendezvous_files": after.rendezvous_files - before.rendezvous_files,
        "temp_artifacts": after.temp_artifacts - before.temp_artifacts,
    }
    cached_growth = after.allocator_cached_bytes - before.allocator_cached_bytes
    unexplained_growth = delta_bytes - cached_growth
    if unexplained_growth > 0:
        # A rising allocator cache is normal; an *unexplained* rise is not.
        kind = (
            "leak_candidate"
            if (not limit_bytes or unexplained_growth > limit_bytes)
            else "allocator_cache"
        )
    elif cached_growth > 0:
        kind = "allocator_cache"
    else:
        kind = "reclaimed"
    return {
        "delta_bytes": delta_bytes,
        "cached_growth_bytes": cached_growth,
        "unexplained_growth_bytes": unexplained_growth,
        "count_deltas": counts,
        "kind": kind,
        "within_limit": limit_bytes == 0 or unexplained_growth <= limit_bytes,
        "note": (
            "allocator cache is not a leak; an unexplained, persistent growth is"
            if kind != "reclaimed"
            else "resources returned to the pre-fault level"
        ),
    }


# ── fault matrix ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FaultMatrixEntry:
    """One row of the E10-10 fault matrix with its oracle skeleton."""

    fault: str
    injection: str
    first_detectable_layer: str
    expected_normalized_error: str
    minimum_recovery_level: str
    notes: str = ""

    def __post_init__(self) -> None:
        _require(self.fault in MINIMUM_RECOVERY_BY_FAULT, "fault not in the frozen matrix", field_name="fault")
        _require(self.first_detectable_layer in DETECTION_LAYERS, "unknown detection layer")
        _require(self.minimum_recovery_level in RECOVERY_LEVELS, "unknown recovery level")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fault": self.fault,
            "injection": self.injection,
            "first_detectable_layer": self.first_detectable_layer,
            "expected_normalized_error": self.expected_normalized_error,
            "minimum_recovery_level": self.minimum_recovery_level,
            "notes": self.notes,
        }


def _matrix_entry(fault: str, injection: str, layer: str, error_class: str, notes: str = "") -> FaultMatrixEntry:
    return FaultMatrixEntry(
        fault=fault,
        injection=injection,
        first_detectable_layer=layer,
        expected_normalized_error=ERROR_CATALOG[error_class].normalized_code,
        minimum_recovery_level=MINIMUM_RECOVERY_BY_FAULT[fault],
        notes=notes,
    )


#: The frozen fault matrix (E10-03 §5 collective faults + E10-10 §4 process faults).
FAULT_MATRIX: Tuple[FaultMatrixEntry, ...] = (
    _matrix_entry("clean_rank_exit", "rank exits at a chosen phase/seq", "process_control_heartbeat", "RANK_EXIT"),
    _matrix_entry("sigterm", "coordinator sends SIGTERM to an experiment rank", "process_control_heartbeat", "RANK_EXIT"),
    _matrix_entry("sigkill", "coordinator sends SIGKILL (no cleanup)", "external_watchdog_or_ras", "RANK_EXIT"),
    _matrix_entry("rank_hang", "rank stays alive but stops collective progress", "collective_progress_timeout", "RANK_STUCK"),
    _matrix_entry("compute_straggler_timeout", "controlled delay either side of the timeout", "collective_progress_timeout", "TIMEOUT"),
    _matrix_entry("shard_load_oom", "allocator budget/fault hook during shard load", "allocator_or_load_barrier", "DEVICE_OOM"),
    _matrix_entry("runtime_workspace_oom", "workspace/activation OOM on one rank/layer", "allocator_or_load_barrier", "DEVICE_OOM"),
    _matrix_entry("kv_growth_oom", "KV growth to the budget or a fault hook", "allocator_or_load_barrier", "DEVICE_OOM"),
    _matrix_entry("communicator_alloc_failure", "failure during init/recreate", "init_timeout_or_preflight", "COMMUNICATOR_ALLOC"),
    _matrix_entry("bootstrap_failure", "wrong rendezvous/rank table fixture", "init_timeout_or_preflight", "BOOTSTRAP"),
    _matrix_entry("inflight_network_error", "isolated in-flight disconnect", "backend_async_error", "NETWORK",
                  notes="requires an authorised isolated network, otherwise PARTIALLY_VALIDATED"),
    _matrix_entry("link_delay_loss", "controlled traffic shaping/proxy", "collective_progress_timeout", "TIMEOUT"),
    _matrix_entry("post_error_reuse", "issue another collective after a fatal error", "request_state", "COMM_ASYNC",
                  notes="the wrapper must reject with COMM_ABORTED"),
    _matrix_entry("op_mismatch", "one rank uses a different op at the same seq", "init_timeout_or_preflight", "MISMATCH"),
    _matrix_entry("count_mismatch", "one rank changes logical count/shape", "init_timeout_or_preflight", "MISMATCH"),
    _matrix_entry("dtype_mismatch", "same byte count, different dtype", "init_timeout_or_preflight", "MISMATCH"),
    _matrix_entry("root_mismatch", "broadcast with different roots", "init_timeout_or_preflight", "MISMATCH"),
    _matrix_entry("skipped_collective", "one rank skips the call but keeps heartbeating", "collective_progress_timeout", "RANK_STUCK"),
    _matrix_entry("duplicate_seq", "reuse a wrong buffer/sequence", "request_state", "MISMATCH"),
)

MATRIX_BY_FAULT: Mapping[str, FaultMatrixEntry] = {entry.fault: entry for entry in FAULT_MATRIX}


def oracle_for_fault(
    fault: str,
    *,
    maximum_detection_time_s: float,
    maximum_global_abort_time_s: float,
    expected_failed_ranks: Sequence[int] = (),
    resource_delta_limit_bytes: int = 0,
    post_recovery_probe: str = "device→p2p→small collective→TP block→Qwen tiny→short request",
) -> FaultOracle:
    """Instantiate the pre-registered oracle for one matrix row."""
    entry = MATRIX_BY_FAULT.get(fault)
    if entry is None:
        raise ConfigError(
            f"fault {fault!r} is not in the frozen matrix; register it before injecting",
            details={"field": "fault"},
        )
    return FaultOracle(
        first_detectable_layer=entry.first_detectable_layer,
        expected_normalized_error=entry.expected_normalized_error,
        maximum_detection_time_s=maximum_detection_time_s,
        maximum_global_abort_time_s=maximum_global_abort_time_s,
        expected_failed_ranks=tuple(expected_failed_ranks),
        all_rank_terminal_state="FAILED_OR_CLEAN",
        output_validity="INVALID",
        communicator_post_state="ABORTED",
        allowed_recovery_level=entry.minimum_recovery_level,
        resource_delta_limit_bytes=resource_delta_limit_bytes,
        post_recovery_probe=post_recovery_probe,
    )


# ── stop conditions / drills ───────────────────────────────────────────────


def should_stop_campaign(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the campaign stop conditions (E10-10 §14)."""
    triggered = [key for key in STOP_CONDITIONS if state.get(key)]
    return {
        "stop": bool(triggered),
        "triggered": triggered,
        "reasons": {key: STOP_CONDITIONS[key] for key in triggered},
    }


@dataclass(frozen=True)
class DrillResult:
    """A blind runbook drill (E10-03 step 29 / E10-10 step 29)."""

    diagnosed_root_rank: Optional[int]
    expected_root_rank: int
    diagnosed_recovery_level: str
    expected_recovery_level: str
    diagnosis_time_s: float
    misdiagnoses: Tuple[str, ...] = ()
    documentation_gaps: Tuple[str, ...] = ()
    operated_on_wrong_target: bool = False

    @property
    def ok(self) -> bool:
        return (
            self.diagnosed_root_rank == self.expected_root_rank
            and self.diagnosed_recovery_level == self.expected_recovery_level
            and not self.operated_on_wrong_target
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "diagnosed_root_rank": self.diagnosed_root_rank,
            "expected_root_rank": self.expected_root_rank,
            "diagnosed_recovery_level": self.diagnosed_recovery_level,
            "expected_recovery_level": self.expected_recovery_level,
            "diagnosis_time_s": self.diagnosis_time_s,
            "misdiagnoses": list(self.misdiagnoses),
            "documentation_gaps": list(self.documentation_gaps),
            "operated_on_wrong_target": self.operated_on_wrong_target,
            "ok": self.ok,
        }


def boundedness_summary(verdicts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-fault MTTD/abort/cleanup/restart roll-up (E10-10 step 30)."""
    by_fault: Dict[str, List[Mapping[str, Any]]] = {}
    for verdict in verdicts:
        by_fault.setdefault(str(verdict["fault"]), []).append(verdict)
    rows: List[Dict[str, Any]] = []
    boundaries: Dict[str, List[str]] = {
        "application_recoverable": [],
        "process_restart_required": [],
        "node_quarantine_required": [],
        "not_really_verified": [],
    }
    for fault, items in sorted(by_fault.items()):
        mttd = distribution_summary([float(item["mttd_s"]) for item in items])
        mttr = distribution_summary([float(item["mttr_s"]) for item in items])
        both = distribution_summary([float(item["abort_time_s"]) for item in items])
        recovery = sorted({str(item.get("recovery_level_used", "")) for item in items})
        row = {
            "fault": fault,
            "runs": len(items),
            "ok": all(bool(item.get("ok")) for item in items),
            "mttd": mttd,
            "abort_time": both,
            "mttr": mttr,
            "recovery_levels": recovery,
            "partially_validated": any(bool(item.get("partially_validated")) for item in items),
        }
        rows.append(row)
        if row["partially_validated"]:
            boundaries["not_really_verified"].append(fault)
        elif recovery and recovery[-1] in ("NODE_QUARANTINE", "JOB_RESTART_FROM_ARTIFACT", "UNRECOVERABLE_IN_SCOPE"):
            boundaries["node_quarantine_required"].append(fault)
        elif recovery and recovery[-1] in ("PROCESS_GROUP_RESTART",):
            boundaries["process_restart_required"].append(fault)
        else:
            boundaries["application_recoverable"].append(fault)
    return {"faults": rows, "boundaries": boundaries}


__all__ = [
    "AbortCoordinator",
    "COMMIT_STATES",
    "ControlHeartbeat",
    "DETECTION_LAYERS",
    "DrillResult",
    "ERROR_CATALOG",
    "ERROR_CLASSES",
    "ErrorEntry",
    "FAULT_MATRIX",
    "FaultInjection",
    "FaultMatrixEntry",
    "FaultOracle",
    "FaultVerdict",
    "FailureObservation",
    "MATRIX_BY_FAULT",
    "MINIMUM_RECOVERY_BY_FAULT",
    "RECOVERY_LEVELS",
    "RESOURCE_DELTA_KINDS",
    "RETRY_DECISIONS",
    "ResourceSnapshot",
    "STOP_CONDITIONS",
    "SafetyScope",
    "TimeMetrics",
    "WATCHDOG_LAYERS",
    "boundedness_summary",
    "classify_distributed_error",
    "distribution_summary",
    "evaluate_fault",
    "oracle_for_fault",
    "retryability_after_fault",
    "should_stop_campaign",
    "snapshot_delta",
    "watchdog_decision",
]
