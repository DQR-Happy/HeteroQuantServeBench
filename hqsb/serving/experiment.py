"""S08 experiment scaffolding: prerequisites, run layout, evidence, verdicts.

Like its S05/S06/S07 counterparts, the one central property is that **this
module cannot produce a conclusion by default**:

* :func:`check_prerequisites` looks for the *evidence* the S08 protocol
  demands (a passing S07 P0 verdict, a real two-Backend registry, frozen
  request fixtures, a pre-registered SLO, a recorded topology, a load-generator
  calibration and a frozen environment fingerprint) and names the path it
  looked for;
* :meth:`RunDirectory.write_verdict` refuses PASS/FAIL/PASS_NEGATIVE unless the
  caller explicitly enables execution, the prerequisites are satisfied **and**
  raw samples exist;
* the SLO/spec documents ship as *templates*: the SLO helper refuses to
  evaluate goodput until the owner freezes them.

Two checks deliberately live under ``docs/stage_experiments/`` (the protocol
tree this scaffolding never writes to), so S08's own interface work cannot
unlock S08's own experiments.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.experiment_io import RunStorage, run_directory_path
from hqsb.core.errors import ConfigError
from hqsb.runtime.experiment import EXPERIMENT_RECORD_FIELDS  # reused field list

#: Protocol statuses (docs/stage_experiments/README.md §3).
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

CONCLUSION_STATUSES: Tuple[str, ...] = (STATUS_PASS, STATUS_PASS_NEGATIVE, STATUS_FAIL)

EXPERIMENTS: Tuple[str, ...] = tuple(f"E08-{index:02d}" for index in range(1, 12))

STAGE = "S08"

#: details README §26 run layout.
RUN_LAYOUT: Tuple[str, ...] = (
    "environment",
    "service_config",
    "backend_registry",
    "slo",
    "workload",
    "client",
    "gateway",
    "admission",
    "queue",
    "routing",
    "cache",
    "backend",
    "stream",
    "faults",
    "metrics",
    "traces",
    "logs",
    "profiler",
    "correctness",
    "performance",
    "resource",
    "commands",
    "stdout",
    "stderr",
)

#: Files the §26 layout places at the run root.
RUN_ROOT_FILES: Tuple[str, ...] = (
    "preregistration.json",
    "evidence_manifest.json",
    "report.md",
)

#: Where a real S08 execution records its evidence (never written by this module).
S08_EVIDENCE_DIR = os.path.join("docs", "stage_experiments", "S08")
S07_VERDICT_GLOB = os.path.join("docs", "stage_experiments", "S07", "**", "verdict.json")

BACKEND_REGISTRY_NAME = "backend_registry.json"
SLO_SPEC_NAME = "slo_spec.json"
TOPOLOGY_NAME = "topology.json"
LOADGEN_CALIBRATION_NAME = "loadgen_calibration.json"

REQUEST_FIXTURE_CANDIDATES: Tuple[str, ...] = (
    os.path.join("docs", "stage_experiments", "S08", "s08_request_fixtures.json"),
    os.path.join("docs", "stage_experiments", "S04.5", "s07_request_fixtures.json"),
    os.path.join("benchmarks", "workloads", "runtime", "frozen_request_spec.json"),
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── prerequisites ──────────────────────────────────────────────────────────


@dataclass
class PrerequisiteCheck:
    """One prerequisite with its evidence pointer and verdict."""

    name: str
    satisfied: bool
    evidence: str = ""
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "satisfied": self.satisfied,
            "evidence": self.evidence,
            "reason": self.reason,
        }


@dataclass
class PrerequisiteStatus:
    """Aggregate prerequisite status for S08."""

    stage: str
    checks: List[PrerequisiteCheck] = field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return all(check.satisfied for check in self.checks)

    @property
    def missing(self) -> List[str]:
        return [check.name for check in self.checks if not check.satisfied]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "satisfied": self.satisfied,
            "missing": self.missing,
            "checks": [check.as_dict() for check in self.checks],
        }


def _s07_passing_verdicts(root: str) -> List[Dict[str, Any]]:
    passing: List[Dict[str, Any]] = []
    for path in glob.glob(os.path.join(root, S07_VERDICT_GLOB), recursive=True):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") in (STATUS_PASS, STATUS_PASS_NEGATIVE):
            payload["_path"] = path
            passing.append(payload)
    return passing


def check_prerequisites(root: str) -> PrerequisiteStatus:
    """Inspect the repository for the S08 evidence chain (never for intent)."""
    checks: List[PrerequisiteCheck] = []

    s07 = _s07_passing_verdicts(root)
    checks.append(
        PrerequisiteCheck(
            name="s07_p0_verdicts",
            satisfied=bool(s07),
            evidence=s07[0]["_path"] if s07 else "",
            reason=(
                ""
                if s07
                else "no S07 verdict.json with PASS/PASS_NEGATIVE; the Backend Contract, "
                "request state, cancel, metrics and C7 trace S08 builds on are unverified. "
                "S07 interface scaffolding is NOT execution evidence"
            ),
        )
    )

    registry_path = os.path.join(root, S08_EVIDENCE_DIR, BACKEND_REGISTRY_NAME)
    registry_ok = False
    registry_note = (
        f"no two-Backend registry evidence at {registry_path}; a service must register at "
        "least two real instances (distinct handles) before routing can be claimed. "
        "Scaffolding under hqsb/serving cannot satisfy it"
    )
    if os.path.isfile(registry_path):
        try:
            with open(registry_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            instances = payload.get("instances", payload)
            ids = {item.get("instance_id") for item in instances} if isinstance(instances, list) else set(instances)
            handles = {
                item.get("runtime_commit") or item.get("model_epoch")
                for item in (instances if isinstance(instances, list) else [])
            }
            registry_ok = len(ids) >= 2 and len(handles) >= 2
            if not registry_ok:
                registry_note = (
                    "the registry evidence lists fewer than two distinct instances or two "
                    "names pointing at one handle"
                )
        except (OSError, json.JSONDecodeError) as exc:
            registry_note = f"registry evidence is unreadable: {exc}"
    checks.append(
        PrerequisiteCheck(
            name="two_backend_registry_evidence",
            satisfied=registry_ok,
            evidence=registry_path if registry_ok else "",
            reason="" if registry_ok else registry_note,
        )
    )

    fixtures = [
        os.path.join(root, relative)
        for relative in REQUEST_FIXTURE_CANDIDATES
        if os.path.isfile(os.path.join(root, relative))
    ]
    checks.append(
        PrerequisiteCheck(
            name="frozen_request_fixtures",
            satisfied=bool(fixtures),
            evidence=fixtures[0] if fixtures else "",
            reason=(
                ""
                if fixtures
                else "no frozen tokenized request fixtures (model/tokenizer/template/"
                "precision/sampling/stop); without them the client and the service would "
                "each build their own requests"
            ),
        )
    )

    slo_path = os.path.join(root, S08_EVIDENCE_DIR, SLO_SPEC_NAME)
    slo_frozen = False
    slo_note = (
        f"no pre-registered SLO at {slo_path}; the shipped configs/serving/slo_spec.yaml is "
        "a template and cannot evaluate goodput"
    )
    if os.path.isfile(slo_path):
        try:
            with open(slo_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            slo_frozen = payload.get("preregistration_status") == "frozen"
            if not slo_frozen:
                slo_note = "the recorded SLO is not frozen"
        except (OSError, json.JSONDecodeError) as exc:
            slo_note = f"SLO evidence is unreadable: {exc}"
    checks.append(
        PrerequisiteCheck(
            name="preregistered_slo",
            satisfied=slo_frozen,
            evidence=slo_path if slo_frozen else "",
            reason="" if slo_frozen else slo_note,
        )
    )

    topology_path = os.path.join(root, S08_EVIDENCE_DIR, TOPOLOGY_NAME)
    checks.append(
        PrerequisiteCheck(
            name="recorded_topology",
            satisfied=os.path.isfile(topology_path),
            evidence=topology_path if os.path.isfile(topology_path) else "",
            reason=(
                ""
                if os.path.isfile(topology_path)
                else f"no recorded service/loadgen topology at {topology_path}; without it "
                "'the client is not the bottleneck' is an assumption"
            ),
        )
    )

    calibration_path = os.path.join(root, S08_EVIDENCE_DIR, LOADGEN_CALIBRATION_NAME)
    checks.append(
        PrerequisiteCheck(
            name="loadgen_calibration",
            satisfied=os.path.isfile(calibration_path),
            evidence=calibration_path if os.path.isfile(calibration_path) else "",
            reason=(
                ""
                if os.path.isfile(calibration_path)
                else f"no load-generator calibration at {calibration_path}; the client's own "
                "ceiling must be measured before capacity can be attributed to the service"
            ),
        )
    )

    # Only the protocol tree counts: fingerprints under ``experiment_results`` are
    # written by this scaffolding and must not unlock its own gate.
    fingerprints = glob.glob(
        os.path.join(root, "docs", "stage_experiments", "*", "**", "environment_fingerprint.json"),
        recursive=True,
    )
    checks.append(
        PrerequisiteCheck(
            name="frozen_environment_fingerprint",
            satisfied=bool(fingerprints),
            evidence=sorted(fingerprints)[0] if fingerprints else "",
            reason=(
                ""
                if fingerprints
                else "no recorded environment fingerprint in docs/stage_experiments/**; a "
                "fingerprint written by this scaffolding under experiment_results/ does not "
                "count (the same trap S06/S07 closed)"
            ),
        )
    )

    schema_ok = True
    schema_note = ""
    try:
        from hqsb.serving import telemetry as serving_telemetry

        coverage = serving_telemetry.coverage_summary()
        schema_ok = bool(coverage["c6"]["ok"] and coverage["c7"]["ok"])
        if not schema_ok:
            schema_note = (
                "the C6/C7 projection does not cover the required S08 fields "
                f"(c6 missing={coverage['c6']['missing']})"
            )
    except Exception as exc:  # noqa: BLE001 - a broken contract must block, not crash
        schema_ok = False
        schema_note = f"{type(exc).__name__}: {exc}"
    checks.append(
        PrerequisiteCheck(
            name="c6_c7_schema_available",
            satisfied=schema_ok,
            evidence="hqsb/serving/telemetry.py" if schema_ok else "",
            reason=schema_note,
        )
    )
    return PrerequisiteStatus(stage=STAGE, checks=checks)


# ── preregistration ────────────────────────────────────────────────────────


@dataclass
class Preregistration:
    """The frozen S08 experiment plan (details README §23.1)."""

    experiment_id: str
    question: str
    hypothesis: str
    service_versions: Mapping[str, str] = field(default_factory=dict)
    model_artifact_hash: str = ""
    protocol_profile_hash: str = ""
    slo_spec_hash: str = ""
    arrival_trace_hash: str = ""
    payload_trace_hash: str = ""
    policy_config_hash: str = ""
    primary_metric: str = ""
    guardrails: Tuple[str, ...] = ()
    load_points: Tuple[float, ...] = ()
    runs: int = 0
    independent_processes: int = 3
    measurement_window_sec: float = 0.0
    exclusions: Tuple[str, ...] = ()
    stop_conditions: Tuple[str, ...] = ()
    allowed_claims: Tuple[str, ...] = ()
    claim_boundary: str = ""
    hardware: str = ""
    notes: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.experiment_id not in EXPERIMENTS:
            raise ConfigError(
                f"unknown experiment id {self.experiment_id!r}; expected one of "
                f"{list(EXPERIMENTS)}"
            )
        for name in ("question", "hypothesis"):
            if not getattr(self, name):
                raise ConfigError(
                    f"preregistration for {self.experiment_id} needs a non-empty {name!r}; "
                    "a hypothesis written after seeing the results is not a preregistration"
                )
        if not self.claim_boundary:
            raise ConfigError(
                f"preregistration for {self.experiment_id} needs an explicit claim_boundary "
                "(what may NOT be claimed from this experiment)",
                details={"field": "claim_boundary"},
            )
        if self.independent_processes < 3:
            raise ConfigError(
                "at least three independent service processes are required "
                "(details README §23.2)",
                details={"field": "independent_processes"},
            )
        if not self.measurement_window_sec:
            raise ConfigError(
                "the preregistration must fix the measurement window",
                details={"field": "measurement_window_sec"},
            )

    def to_json(self) -> str:
        payload = {
            "schema_version": "1.0.0",
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "service_versions": dict(self.service_versions),
            "model_artifact_hash": self.model_artifact_hash,
            "protocol_profile_hash": self.protocol_profile_hash,
            "slo_spec_hash": self.slo_spec_hash,
            "arrival_trace_hash": self.arrival_trace_hash,
            "payload_trace_hash": self.payload_trace_hash,
            "policy_config_hash": self.policy_config_hash,
            "primary_metric": self.primary_metric,
            "guardrails": list(self.guardrails),
            "load_points": list(self.load_points),
            "runs": self.runs,
            "independent_processes": self.independent_processes,
            "measurement_window_sec": self.measurement_window_sec,
            "exclusions": list(self.exclusions),
            "stop_conditions": list(self.stop_conditions),
            "allowed_claims": list(self.allowed_claims),
            "claim_boundary": self.claim_boundary,
            "hardware": self.hardware,
            "notes": self.notes,
            "created_at": self.created_at or _utc_now(),
        }
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)

    @property
    def prereg_hash(self) -> str:
        return _sha256_text(self.to_json())


# ── handbook §4 unified record ─────────────────────────────────────────────


@dataclass
class ExperimentRecord:
    """The unified per-experiment record (handbook §4), stage S08.

    The field list is imported from :mod:`hqsb.runtime.experiment` so the two
    stages cannot drift; the validation rules are the handbook's.
    """

    experiment_id: str
    question: str
    hypothesis: str
    run_id: str = ""
    status: str = STATUS_NOT_STARTED
    stage: str = STAGE
    git_commit: str = ""
    git_dirty: bool = False
    model_manifest_sha256: str = ""
    config_sha256: str = ""
    operator_or_binary_sha256: str = ""
    environment_uri: str = ""
    hardware_and_power_mode: str = ""
    requested_implementation: str = ""
    actual_implementation: str = ""
    fallback_reason: str = ""
    controls: Mapping[str, Any] = field(default_factory=dict)
    independent_variables: Mapping[str, Any] = field(default_factory=dict)
    correctness_metrics: Mapping[str, Any] = field(default_factory=dict)
    performance_samples_uri: str = ""
    profile_artifacts_uri: str = ""
    started_at: str = ""
    ended_at: str = ""
    decision: str = ""
    limitations: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.experiment_id not in EXPERIMENTS:
            raise ConfigError(
                f"unknown experiment id {self.experiment_id!r}; expected one of "
                f"{list(EXPERIMENTS)}"
            )
        if self.status not in PROTOCOL_STATUSES:
            raise ConfigError(
                f"unknown status {self.status!r}; protocol allows {list(PROTOCOL_STATUSES)}"
            )
        for name in ("question", "hypothesis"):
            if not getattr(self, name):
                raise ConfigError(
                    f"the experiment record needs a non-empty {name!r} (handbook §5.1)",
                    details={"field": name},
                )
        if self.actual_implementation and self.requested_implementation:
            if (
                self.actual_implementation != self.requested_implementation
                and not self.fallback_reason
            ):
                raise ConfigError(
                    "actual implementation differs from the requested one without a "
                    "fallback_reason (handbook §5.7)",
                    details={"field": "fallback_reason"},
                )
        if self.status in CONCLUSION_STATUSES:
            if not self.decision:
                raise ConfigError(
                    f"a {self.status} record needs a decision line", details={"field": "decision"}
                )
            if not (self.performance_samples_uri or self.profile_artifacts_uri):
                raise ConfigError(
                    "a conclusion must point at its raw evidence; '未运行不得声明 PASS' "
                    "(handbook §4)",
                    details={"field": "performance_samples_uri"},
                )

    @property
    def executed(self) -> bool:
        return bool(self.started_at or self.ended_at)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "model_manifest_sha256": self.model_manifest_sha256,
            "config_sha256": self.config_sha256,
            "operator_or_binary_sha256": self.operator_or_binary_sha256,
            "environment_uri": self.environment_uri,
            "hardware_and_power_mode": self.hardware_and_power_mode,
            "requested_implementation": self.requested_implementation,
            "actual_implementation": self.actual_implementation,
            "fallback_reason": self.fallback_reason,
            "controls": dict(self.controls),
            "independent_variables": dict(self.independent_variables),
            "correctness_metrics": dict(self.correctness_metrics),
            "performance_samples_uri": self.performance_samples_uri,
            "profile_artifacts_uri": self.profile_artifacts_uri,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "decision": self.decision,
            "limitations": list(self.limitations),
        }

    def to_json(self) -> str:
        payload = dict(self.as_dict())
        payload["schema_version"] = "1.0.0"
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


def template_experiment_record(
    experiment_id: str, *, status: str, reason: str
) -> ExperimentRecord:
    """A record template the executor fills in; it can never carry a conclusion."""
    if status in CONCLUSION_STATUSES:
        raise ConfigError(
            "a template cannot be written with a conclusion status; run the experiment and "
            "build the record from real evidence",
            details={"field": "status"},
        )
    return ExperimentRecord(
        experiment_id=experiment_id,
        question="see docs/stage_experiments/details/S08/<file>",
        hypothesis="see docs/stage_experiments/details/S08/<file>",
        status=status,
        limitations=[
            "template only; the executor must fill environment/model/config hashes, "
            "implementations, controls, metrics and raw evidence URIs",
            reason,
        ],
    )


# ── evidence manifest (control plane §26.3 + S08 extras) ───────────────────


@dataclass
class EvidenceManifest:
    """Run evidence manifest with the S08-specific fields."""

    run_id: str
    stage: str = STAGE
    experiment_id: str = ""
    claim_ids: Sequence[str] = ()
    git_commit: str = ""
    git_dirty: bool = False
    model_manifest_sha256: str = ""
    config_uri: str = ""
    config_sha256: str = ""
    operator_artifacts: Sequence[Mapping[str, Any]] = ()
    environment_uri: str = ""
    environment_sha256: str = ""
    commands: Sequence[str] = ()
    raw_artifacts: Sequence[Mapping[str, Any]] = ()
    normalized_artifacts: Sequence[Mapping[str, Any]] = ()
    reports: Sequence[str] = ()
    correctness_status: str = "not_run"
    claim_level: str = "SOURCE"
    limitations: Sequence[str] = ()
    # S08 extras (details README §3 / §26)
    service_id: str = ""
    service_version: str = ""
    service_commit: str = ""
    protocol_profile_hash: str = ""
    backend_registry_uri: str = ""
    backend_identities: Sequence[Mapping[str, Any]] = ()
    slo_spec_uri: str = ""
    slo_spec_hash: str = ""
    payload_trace_hash: str = ""
    arrival_trace_hash: str = ""
    topology_uri: str = ""
    loadgen_version: str = ""
    loadgen_calibration_uri: str = ""
    policy_versions: Mapping[str, str] = field(default_factory=dict)
    per_request_metrics_uri: str = ""
    wire_events_uri: str = ""
    trace_uri: str = ""
    fault_timeline_uri: str = ""
    actual_precision: str = ""
    memory_bytes: float = 0.0
    energy_joules: float = 0.0

    def to_json(self) -> str:
        payload = {
            "schema_version": "1.0.0",
            "run_id": self.run_id,
            "stage": self.stage,
            "experiment_id": self.experiment_id,
            "claim_ids": list(self.claim_ids),
            "git": {"commit": self.git_commit, "dirty": self.git_dirty},
            "model_artifact": {"manifest_sha256": self.model_manifest_sha256},
            "config": {"uri": self.config_uri, "sha256": self.config_sha256},
            "operator_artifacts": [dict(item) for item in self.operator_artifacts],
            "environment": {"uri": self.environment_uri, "sha256": self.environment_sha256},
            "commands": list(self.commands),
            "raw_artifacts": [dict(item) for item in self.raw_artifacts],
            "normalized_artifacts": [dict(item) for item in self.normalized_artifacts],
            "reports": list(self.reports),
            "correctness_status": self.correctness_status,
            "claim_level": self.claim_level,
            "limitations": list(self.limitations),
            "service": {
                "service_id": self.service_id,
                "version": self.service_version,
                "commit": self.service_commit,
                "protocol_profile_hash": self.protocol_profile_hash,
                "backend_registry_uri": self.backend_registry_uri,
                "backend_identities": [dict(item) for item in self.backend_identities],
                "slo_spec_uri": self.slo_spec_uri,
                "slo_spec_hash": self.slo_spec_hash,
                "payload_trace_hash": self.payload_trace_hash,
                "arrival_trace_hash": self.arrival_trace_hash,
                "topology_uri": self.topology_uri,
                "loadgen_version": self.loadgen_version,
                "loadgen_calibration_uri": self.loadgen_calibration_uri,
                "policy_versions": dict(self.policy_versions),
                "per_request_metrics_uri": self.per_request_metrics_uri,
                "wire_events_uri": self.wire_events_uri,
                "trace_uri": self.trace_uri,
                "fault_timeline_uri": self.fault_timeline_uri,
                "actual_precision": self.actual_precision,
                "memory_bytes": self.memory_bytes,
                "energy_joules": self.energy_joules,
            },
        }
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


# ── run directory ──────────────────────────────────────────────────────────


class RunDirectory(RunStorage):
    """Creates and manages one ``experiment_results/S08/<E>/<run_id>/`` tree."""

    def __init__(self, root: str, experiment_id: str, run_id: str) -> None:
        if experiment_id not in EXPERIMENTS:
            raise ConfigError(f"unknown experiment id {experiment_id!r}")
        if not run_id:
            raise ConfigError("run_id must not be empty")
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.path = run_directory_path(root, STAGE, experiment_id, run_id)

    def create(self) -> str:
        os.makedirs(self.path, exist_ok=True)
        for sub in RUN_LAYOUT:
            os.makedirs(os.path.join(self.path, sub), exist_ok=True)
        return self.path





    def write_preregistration(self, prereg: Preregistration) -> str:
        return self.write_text("preregistration.json", prereg.to_json())

    def write_experiment_record(self, record: ExperimentRecord) -> str:
        return self.write_text("experiment_record.json", record.to_json())

    def write_evidence_manifest(self, manifest: EvidenceManifest) -> str:
        return self.write_text("evidence_manifest.json", manifest.to_json())

    def raw_file_count(self) -> int:
        for candidate in ("client", "performance", "gateway", "stream"):
            path = os.path.join(self.path, candidate)
            if os.path.isdir(path):
                count = sum(len(files) for _root, _dirs, files in os.walk(path))
                if count:
                    return count
        return 0

    def write_report_skeleton(self, experiment_id: str, title: str) -> str:
        """Write a report stub that cannot be mistaken for a result."""
        skeleton = (
            f"# {experiment_id}: {title}\n\n"
            f"> 状态：**NOT_RUN**（本文件由 `hqsb.serving.experiment` 生成的骨架；\n"
            f"> 未执行任何实验，未产出任何数值）。\n\n"
            "## 1. 预注册\n\n见 `preregistration.json`。\n\n"
            "## 2. 环境与身份\n\n见 `environment_fingerprint.json`、`evidence_manifest.json`；"
            "未执行时为占位。\n\n"
            "## 3. 原始数据\n\n`client/`、`gateway/`、`stream/`、`performance/` 均为空，"
            "直到实验真正运行。\n\n"
            "## 4. 结论\n\n无。实验未执行，禁止填写任何容量/延迟/吞吐/goodput/命中率结论。\n"
        )
        return self.write_text("report.md", skeleton)

    def write_verdict(
        self,
        *,
        status: str,
        reason: str,
        prerequisites: PrerequisiteStatus,
        executed: bool,
        allow_execute: bool = False,
        raw_samples: int = 0,
        limitations: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Write a verdict — or refuse when the run cannot support one."""
        if status not in PROTOCOL_STATUSES:
            raise ConfigError(
                f"unknown status {status!r}; protocol allows {list(PROTOCOL_STATUSES)}"
            )
        effective = status
        effective_reason = reason
        if status in CONCLUSION_STATUSES:
            if not allow_execute:
                effective = STATUS_BLOCKED
                effective_reason = (
                    "refusing to emit an experimental conclusion: execution is disabled "
                    "(pass allow_execute=True to the driver and run the experiment properly)"
                )
            elif not prerequisites.satisfied:
                effective = STATUS_BLOCKED
                effective_reason = "prerequisites unsatisfied: " + ", ".join(prerequisites.missing)
            elif not executed or raw_samples <= 0:
                effective = STATUS_BLOCKED
                effective_reason = (
                    "no raw samples recorded; a conclusion without raw evidence is not allowed"
                )
        payload = {
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": effective,
            "requested_status": status,
            "reason": effective_reason,
            "executed": executed,
            "raw_samples": raw_samples,
            "prerequisites": prerequisites.as_dict(),
            "limitations": list(limitations),
            "written_at": _utc_now(),
        }
        self.write_json("verdict.json", payload)
        return payload

    def write_status(
        self,
        status: str,
        reason: str,
        prerequisites: PrerequisiteStatus,
        executed: bool = False,
    ) -> Dict[str, Any]:
        """Write a non-conclusion status (NOT_STARTED / RUNNING / BLOCKED)."""
        if status in CONCLUSION_STATUSES:
            raise ConfigError(f"use write_verdict for conclusion statuses, got {status!r}")
        payload = {
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": status,
            "reason": reason,
            "executed": executed,
            "prerequisites": prerequisites.as_dict(),
            "written_at": _utc_now(),
        }
        self.write_json("status.json", payload)
        return payload


# ── environment / git ──────────────────────────────────────────────────────


def environment_fingerprint(extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Collect a machine-readable environment record (never contains secrets)."""
    payload: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
        "collected_at": _utc_now(),
        "scope": "development host; not a Jetson production measurement",
    }
    for module_name, key in (("torch", "torch"), ("triton", "triton"), ("yaml", "pyyaml")):
        try:
            module = __import__(module_name)
            payload[key] = str(getattr(module, "__version__", ""))
        except ImportError:
            payload[key] = "not installed"
    try:
        from hqsb.serving import transport_http

        payload["http_transport"] = {
            "implementation": "stdlib.http.server",
            "config": transport_http.HttpTransportConfig().as_dict(),
        }
    except Exception as exc:  # noqa: BLE001 - the fingerprint must never crash
        payload["http_transport"] = {"error": f"{type(exc).__name__}: {exc}"}
    if extra:
        payload.update(dict(extra))
    return payload


def git_state(root: str) -> Dict[str, Any]:
    """Return ``{commit, dirty}``; a missing git binary is reported, not fatal."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False
        )
        return {
            "commit": commit.stdout.strip() if commit.returncode == 0 else "",
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else False,
            "error": "" if commit.returncode == 0 else commit.stderr.strip(),
        }
    except FileNotFoundError:
        return {"commit": "", "dirty": False, "error": "git not installed"}


def interface_only_run(
    root: str, experiment_id: str, *, run_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a run directory and record a non-conclusion status."""
    prerequisites = check_prerequisites(root)
    resolved_run_id = run_id or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run = RunDirectory(root, experiment_id, resolved_run_id)
    run.create()
    run.write_json("environment_fingerprint.json", environment_fingerprint())
    run.write_json("prerequisites.json", prerequisites.as_dict())
    status = STATUS_NOT_STARTED if prerequisites.satisfied else STATUS_BLOCKED
    reason = (
        "interface scaffolding only; no experiment was executed"
        if prerequisites.satisfied
        else "prerequisites unsatisfied: " + ", ".join(prerequisites.missing)
    )
    return {
        "run_dir": run.path,
        "status": run.write_status(status, reason, prerequisites)["status"],
        "prerequisites": prerequisites.as_dict(),
    }


__all__ = [
    "BACKEND_REGISTRY_NAME",
    "CONCLUSION_STATUSES",
    "EXPERIMENTS",
    "EXPERIMENT_RECORD_FIELDS",
    "EvidenceManifest",
    "ExperimentRecord",
    "LOADGEN_CALIBRATION_NAME",
    "PROTOCOL_STATUSES",
    "Preregistration",
    "PrerequisiteCheck",
    "PrerequisiteStatus",
    "REQUEST_FIXTURE_CANDIDATES",
    "RUN_LAYOUT",
    "RUN_ROOT_FILES",
    "RunDirectory",
    "S08_EVIDENCE_DIR",
    "SLO_SPEC_NAME",
    "STAGE",
    "STATUS_BLOCKED",
    "STATUS_FAIL",
    "STATUS_N_A_BY_ADR",
    "STATUS_NOT_STARTED",
    "STATUS_PASS",
    "STATUS_PASS_NEGATIVE",
    "STATUS_RUNNING",
    "TOPOLOGY_NAME",
    "check_prerequisites",
    "environment_fingerprint",
    "git_state",
    "interface_only_run",
    "template_experiment_record",
]
