"""S10 experiment scaffolding: prerequisites, run layout, evidence, verdicts.

Like its S05–S09 counterparts, the central property is that **this module cannot
produce a conclusion by default**:

* :func:`check_prerequisites` looks for the *evidence* the S10 protocol demands
  (a passing S07 P0 verdict, an S08 trace reference, at least two real
  accelerators, a frozen topology manifest and backend identity, a single-card
  model-core reference, a frozen model/workload pair and a protocol-tree
  environment fingerprint) and names the path it looked for;
* :meth:`RunDirectory.write_verdict` refuses PASS/FAIL/PASS_NEGATIVE unless the
  caller explicitly enables execution, the prerequisites are satisfied **and**
  raw samples exist;
* two checks only read ``docs/stage_experiments/**`` — the protocol tree this
  scaffolding never writes to — so the interface layer cannot unlock its own
  experiments.

The CPU loopback/simulation helpers are *smoke* self-checks: they carry
``claim_allowed() == False`` and are never an experiment result.
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

EXPERIMENTS: Tuple[str, ...] = tuple(f"E10-{index:02d}" for index in range(1, 11))

STAGE = "S10"

#: Generic run layout covering every per-experiment requirement of the details.
RUN_LAYOUT: Tuple[str, ...] = (
    "environment",
    "topology",
    "placement",
    "ranks",
    "groups",
    "specs",
    "harness",
    "cases",
    "capabilities",
    "correctness",
    "golden",
    "faults",
    "watchdog",
    "cleanup",
    "recovery",
    "metrics",
    "ledger",
    "benchmark",
    "bandwidth",
    "models",
    "scaling",
    "decomposition",
    "memory",
    "traces",
    "clocks",
    "network",
    "injections",
    "attribution",
    "analysis",
    "confirmation",
    "decision",
    "commands",
    "stdout",
    "stderr",
)

#: Files the layout places at the run root.
RUN_ROOT_FILES: Tuple[str, ...] = (
    "preregistration.json",
    "experiment_record.json",
    "evidence_manifest.json",
    "report.md",
)

#: Where a real S10 execution records its evidence (never written by this module).
S10_EVIDENCE_DIR = os.path.join("docs", "stage_experiments", "S10")
S07_VERDICT_GLOB = os.path.join("docs", "stage_experiments", "S07", "**", "verdict.json")
S08_TRACE_GLOB = os.path.join("docs", "stage_experiments", "S08", "**", "*.jsonl")
S02_REFERENCE_GLOB = os.path.join("docs", "stage_experiments", "S02", "**", "*.json")

TOPOLOGY_MANIFEST_NAME = "manifest.canonical.json"
TOPOLOGY_HASH_NAME = "manifest.sha256"
ACCELERATOR_INVENTORY_NAME = "accelerators.json"
BACKEND_IDENTITY_NAME = "backend_identity.json"
MODEL_WORKLOAD_NAME = "model_workload.json"

#: Minimum accelerators for a real S10 experiment (details S10 README §5).
MINIMUM_ACCELERATORS = 2


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
    """Aggregate prerequisite status for S10."""

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


def _passing_verdicts(root: str, pattern: str) -> List[Dict[str, Any]]:
    passing: List[Dict[str, Any]] = []
    for path in glob.glob(os.path.join(root, pattern), recursive=True):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("status") in (
            STATUS_PASS,
            STATUS_PASS_NEGATIVE,
        ):
            payload["_path"] = path
            passing.append(payload)
    return passing


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _accelerator_count(payload: Optional[Mapping[str, Any]]) -> int:
    if not payload:
        return 0
    for key in ("accelerators", "devices", "gpus", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    count = payload.get("accelerator_count")
    return int(count) if isinstance(count, int) else 0


def check_prerequisites(root: str) -> PrerequisiteStatus:
    """Inspect the repository for the S10 evidence chain (never for intent)."""
    checks: List[PrerequisiteCheck] = []

    s07 = _passing_verdicts(root, S07_VERDICT_GLOB)
    checks.append(
        PrerequisiteCheck(
            name="s07_p0_verdicts",
            satisfied=bool(s07),
            evidence=s07[0]["_path"] if s07 else "",
            reason=(
                ""
                if s07
                else "no S07 verdict.json with PASS/PASS_NEGATIVE; the runtime/KV/scheduling "
                "semantics S10 builds on are unverified. S07 interface scaffolding is NOT "
                "execution evidence"
            ),
        )
    )

    s08_traces = sorted(
        glob.glob(os.path.join(root, S08_TRACE_GLOB), recursive=True)
    )
    checks.append(
        PrerequisiteCheck(
            name="s08_trace_reference",
            satisfied=bool(s08_traces),
            evidence=s08_traces[0] if s08_traces else "",
            reason=(
                ""
                if s08_traces
                else "no recorded S08 tokenized request/service trace under "
                "docs/stage_experiments/S08/**; without it the distributed runs cannot be "
                "related to the served workload"
            ),
        )
    )

    accelerator_path = os.path.join(root, S10_EVIDENCE_DIR, ACCELERATOR_INVENTORY_NAME)
    accelerator_payload = _load_json(accelerator_path)
    accelerator_count = _accelerator_count(accelerator_payload)
    examples_path = os.path.join(root, S10_EVIDENCE_DIR, "E10-01", "topology", ACCELERATOR_INVENTORY_NAME)
    if not accelerator_count:
        accelerator_payload = _load_json(examples_path)
        accelerator_count = _accelerator_count(accelerator_payload)
        if accelerator_count:
            accelerator_path = examples_path
    checks.append(
        PrerequisiteCheck(
            name="two_accelerators_available",
            satisfied=accelerator_count >= MINIMUM_ACCELERATORS,
            evidence=accelerator_path if accelerator_count >= MINIMUM_ACCELERATORS else "",
            reason=(
                ""
                if accelerator_count >= MINIMUM_ACCELERATORS
                else f"no accelerator inventory with >= {MINIMUM_ACCELERATORS} real devices at "
                f"{accelerator_path}; a single device cannot produce a 2-rank collective or a "
                "TP=2 model loop (missing resources are not zeros)"
            ),
        )
    )

    manifest_candidates = [
        os.path.join(root, S10_EVIDENCE_DIR, "E10-01", "topology", TOPOLOGY_HASH_NAME),
        os.path.join(root, S10_EVIDENCE_DIR, TOPOLOGY_HASH_NAME),
    ]
    manifest_path = next((path for path in manifest_candidates if os.path.isfile(path)), "")
    checks.append(
        PrerequisiteCheck(
            name="frozen_topology_manifest",
            satisfied=bool(manifest_path),
            evidence=manifest_path,
            reason=(
                ""
                if manifest_path
                else "no sealed TopologyManifest hash (manifest.sha256) under "
                "docs/stage_experiments/S10/E10-01/topology/; E10-02 must not start on an "
                "unverified topology"
            ),
        )
    )

    backend_path = os.path.join(root, S10_EVIDENCE_DIR, "E10-02", "harness", BACKEND_IDENTITY_NAME)
    backend_payload = _load_json(backend_path) or _load_json(
        os.path.join(root, S10_EVIDENCE_DIR, BACKEND_IDENTITY_NAME)
    )
    backend_ok = bool(
        backend_payload
        and backend_payload.get("kind")
        and backend_payload.get("version")
        and str(backend_payload.get("version")).lower() not in ("latest", "unknown")
    )
    checks.append(
        PrerequisiteCheck(
            name="frozen_collective_backend",
            satisfied=backend_ok,
            evidence=backend_path if backend_ok else "",
            reason=(
                ""
                if backend_ok
                else f"no exact collective backend identity at {backend_path}; NCCL/HCCL version, "
                "framework and launcher must be frozen before any collective measurement"
            ),
        )
    )

    references = sorted(
        glob.glob(os.path.join(root, S02_REFERENCE_GLOB), recursive=True)
    ) + sorted(glob.glob(os.path.join(root, S07_VERDICT_GLOB), recursive=True))
    checks.append(
        PrerequisiteCheck(
            name="single_card_reference",
            satisfied=bool(references),
            evidence=references[0] if references else "",
            reason=(
                ""
                if references
                else "no single-device model-core reference for the same ModelArtifact; TP "
                "correctness (E10-04) compares against it and must not invent one"
            ),
        )
    )

    model_workload_path = os.path.join(root, S10_EVIDENCE_DIR, MODEL_WORKLOAD_NAME)
    model_workload = _load_json(model_workload_path)
    model_ok = bool(
        model_workload
        and model_workload.get("model_artifact_hash")
        and model_workload.get("workload_hash")
    )
    checks.append(
        PrerequisiteCheck(
            name="frozen_model_and_workload",
            satisfied=model_ok,
            evidence=model_workload_path if model_ok else "",
            reason=(
                ""
                if model_ok
                else f"no frozen ModelArtifact/workload pair at {model_workload_path}; scaling "
                "rows are only pairable when the model, precision, tokens and stop rules match"
            ),
        )
    )

    # Only the protocol tree counts: a fingerprint written by this scaffolding
    # under experiment_results/ must not unlock its own gate.
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
                "fingerprint written by this scaffolding does not count"
            ),
        )
    )

    schema_ok = True
    schema_note = ""
    try:
        coverage = _coverage_summary()
        schema_ok = bool(coverage["c6"]["ok"])
        if not schema_ok:
            schema_note = f"the C6 projection is missing {coverage['c6']['missing']}"
    except Exception as exc:  # noqa: BLE001 - a broken contract must block, not crash
        schema_ok = False
        schema_note = f"{type(exc).__name__}: {exc}"
    checks.append(
        PrerequisiteCheck(
            name="c6_c7_schema_available",
            satisfied=schema_ok,
            evidence="hqsb/distributed/telemetry.py" if schema_ok else "",
            reason=schema_note,
        )
    )
    return PrerequisiteStatus(stage=STAGE, checks=checks)


def _coverage_summary() -> Dict[str, Any]:
    from hqsb.distributed import telemetry as telemetry_mod

    return telemetry_mod.coverage_summary()


# ── preregistration ────────────────────────────────────────────────────────


@dataclass
class Preregistration:
    """The frozen S10 experiment plan (details README §16/§22)."""

    experiment_id: str
    question: str
    hypothesis: str
    backend: str = ""
    backend_version: str = ""
    world_size: int = 0
    node_count: int = 0
    rank_epoch: int = 0
    topology_manifest_hash: str = ""
    placement_plan_hash: str = ""
    parallel_plan_hash: str = ""
    model_artifact_hash: str = ""
    workload_hash: str = ""
    precision: str = ""
    collective_grid: Tuple[str, ...] = ()
    primary_metric: str = ""
    guardrails: Tuple[str, ...] = ()
    independent_runs: int = 3
    exclusions: Tuple[str, ...] = ()
    stop_conditions: Tuple[str, ...] = ()
    allowed_claims: Tuple[str, ...] = ()
    claim_boundary: str = ""
    hardware: str = ""
    non_claims: Tuple[str, ...] = ()
    notes: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.experiment_id not in EXPERIMENTS:
            raise ConfigError(
                f"unknown experiment id {self.experiment_id!r}; expected one of {list(EXPERIMENTS)}"
            )
        for name in ("question", "hypothesis"):
            if not getattr(self, name):
                raise ConfigError(
                    f"preregistration for {self.experiment_id} needs a non-empty {name!r}; a "
                    "hypothesis written after the results is not a preregistration"
                )
        if not self.claim_boundary:
            raise ConfigError(
                f"preregistration for {self.experiment_id} needs an explicit claim_boundary "
                "(what may NOT be claimed)",
                details={"field": "claim_boundary"},
            )
        if not self.non_claims:
            raise ConfigError(
                "write down the non-claims (card counts / node counts / topologies not covered)",
                details={"field": "non_claims"},
            )
        if self.independent_runs < 3:
            raise ConfigError(
                "handbook §5.4 requires at least three independent process runs",
                details={"field": "independent_runs"},
            )
        if self.world_size < 1:
            raise ConfigError(
                "the preregistration must fix the world size", details={"field": "world_size"}
            )

    def to_json(self) -> str:
        payload = {
            "schema_version": "1.0.0",
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "world_size": self.world_size,
            "node_count": self.node_count,
            "rank_epoch": self.rank_epoch,
            "topology_manifest_hash": self.topology_manifest_hash,
            "placement_plan_hash": self.placement_plan_hash,
            "parallel_plan_hash": self.parallel_plan_hash,
            "model_artifact_hash": self.model_artifact_hash,
            "workload_hash": self.workload_hash,
            "precision": self.precision,
            "collective_grid": list(self.collective_grid),
            "primary_metric": self.primary_metric,
            "guardrails": list(self.guardrails),
            "independent_runs": self.independent_runs,
            "exclusions": list(self.exclusions),
            "stop_conditions": list(self.stop_conditions),
            "allowed_claims": list(self.allowed_claims),
            "claim_boundary": self.claim_boundary,
            "hardware": self.hardware,
            "non_claims": list(self.non_claims),
            "notes": self.notes,
            "created_at": self.created_at or _utc_now(),
        }
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)

    @property
    def prereg_hash(self) -> str:
        return _sha256_text(self.to_json())


# ── unified record (handbook §4) ───────────────────────────────────────────


@dataclass
class ExperimentRecord:
    """The unified per-experiment record (handbook §4), stage S10."""

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
                f"unknown experiment id {self.experiment_id!r}; expected one of {list(EXPERIMENTS)}"
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
                    "a conclusion must point at its raw evidence; 未运行不得声明 PASS (handbook §4)",
                    details={"field": "performance_samples_uri"},
                )
            if not self.model_manifest_sha256 or not self.config_sha256:
                raise ConfigError(
                    "a conclusion needs the model manifest and config hashes",
                    details={"field": "model_manifest_sha256"},
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
        question="see docs/stage_experiments/details/S10/<file>",
        hypothesis="see docs/stage_experiments/details/S10/<file>",
        status=status,
        limitations=[
            "template only; the executor must fill environment/model/config hashes, "
            "implementations, controls, metrics and raw evidence URIs",
            reason,
        ],
    )


# ── evidence manifest ──────────────────────────────────────────────────────


@dataclass
class EvidenceManifest:
    """Run evidence manifest with the S10-specific fields (details README §16)."""

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
    # S10 extras
    job_id: str = ""
    rank_epoch: int = 0
    world_size: int = 0
    node_count: int = 0
    backend: str = ""
    backend_version: str = ""
    topology_manifest_sha256: str = ""
    placement_plan_sha256: str = ""
    parallel_plan_sha256: str = ""
    workload_hash: str = ""
    precision_logical: str = ""
    precision_actual: str = ""
    scaleup_kind: str = ""
    speedup: Optional[float] = None
    efficiency: Optional[float] = None
    device_seconds: Optional[float] = None
    per_rank_artifacts_uri: str = ""
    collective_ledger_uri: str = ""
    profiler_metric_manifest_uri: str = ""
    trace_uri: str = ""
    fault_timeline_uri: str = ""
    recovery_level: str = ""
    mttd_s: Optional[float] = None
    mttr_s: Optional[float] = None

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
            "distributed": {
                "job_id": self.job_id,
                "rank_epoch": self.rank_epoch,
                "world_size": self.world_size,
                "node_count": self.node_count,
                "backend": self.backend,
                "backend_version": self.backend_version,
                "topology_manifest_sha256": self.topology_manifest_sha256,
                "placement_plan_sha256": self.placement_plan_sha256,
                "parallel_plan_sha256": self.parallel_plan_sha256,
                "workload_hash": self.workload_hash,
                "precision_logical": self.precision_logical,
                "precision_actual": self.precision_actual,
                "scaleup_kind": self.scaleup_kind,
                "speedup": self.speedup,
                "efficiency": self.efficiency,
                "device_seconds": self.device_seconds,
                "per_rank_artifacts_uri": self.per_rank_artifacts_uri,
                "collective_ledger_uri": self.collective_ledger_uri,
                "profiler_metric_manifest_uri": self.profiler_metric_manifest_uri,
                "trace_uri": self.trace_uri,
                "fault_timeline_uri": self.fault_timeline_uri,
                "recovery_level": self.recovery_level,
                "mttd_s": self.mttd_s,
                "mttr_s": self.mttr_s,
            },
        }
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


# ── run directory ──────────────────────────────────────────────────────────


class RunDirectory(RunStorage):
    """Creates and manages one ``experiment_results/S10/<E>/<run_id>/`` tree."""

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
        """Count raw artifacts, preferring the per-rank evidence directories."""
        for candidate in ("ranks", "benchmark", "correctness", "traces", "faults"):
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
            f"> 状态：**NOT_RUN**（本文件由 `hqsb.distributed.experiment` 生成的骨架；\n"
            f"> 未执行任何实验，未产出任何数值）。\n\n"
            "## 1. 预注册\n\n见 `preregistration.json`。\n\n"
            "## 2. 环境与身份\n\n见 `environment_fingerprint.json`、`evidence_manifest.json`；"
            "未执行时为占位。\n\n"
            "## 3. 原始数据\n\n`ranks/`、`benchmark/`、`correctness/`、`traces/` 均为空，"
            "直到实验真正运行。\n\n"
            "## 4. 结论\n\n无。实验未执行，禁止填写任何 collective/TP/scaling/overlap/故障结论。\n"
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
        "scope": "development host; not a multi-node production measurement",
    }
    for module_name, key in (
        ("torch", "torch"),
        ("yaml", "pyyaml"),
    ):
        try:
            module = __import__(module_name)
            payload[key] = str(getattr(module, "__version__", ""))
        except ImportError:
            payload[key] = "not installed"
    try:
        import torch  # noqa: F401 - lazy, optional

        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            payload["cuda_device_count"] = torch.cuda.device_count()
            payload["nccl_version"] = (
                torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else ""
            )
        else:
            payload["cuda_device_count"] = 0
    except Exception as exc:  # noqa: BLE001 - the fingerprint must never crash
        payload["cuda_probe"] = f"{type(exc).__name__}: {exc}"
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
    "ACCELERATOR_INVENTORY_NAME",
    "BACKEND_IDENTITY_NAME",
    "CONCLUSION_STATUSES",
    "EXPERIMENTS",
    "EXPERIMENT_RECORD_FIELDS",
    "EvidenceManifest",
    "ExperimentRecord",
    "MINIMUM_ACCELERATORS",
    "MODEL_WORKLOAD_NAME",
    "PROTOCOL_STATUSES",
    "Preregistration",
    "PrerequisiteCheck",
    "PrerequisiteStatus",
    "RUN_LAYOUT",
    "RUN_ROOT_FILES",
    "RunDirectory",
    "S02_REFERENCE_GLOB",
    "S07_VERDICT_GLOB",
    "S08_TRACE_GLOB",
    "S10_EVIDENCE_DIR",
    "STAGE",
    "STATUS_BLOCKED",
    "STATUS_FAIL",
    "STATUS_N_A_BY_ADR",
    "STATUS_NOT_STARTED",
    "STATUS_PASS",
    "STATUS_PASS_NEGATIVE",
    "STATUS_RUNNING",
    "TOPOLOGY_HASH_NAME",
    "TOPOLOGY_MANIFEST_NAME",
    "check_prerequisites",
    "environment_fingerprint",
    "git_state",
    "interface_only_run",
    "template_experiment_record",
]
