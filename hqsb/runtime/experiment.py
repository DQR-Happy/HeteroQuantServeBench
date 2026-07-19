"""S07 experiment scaffolding: prerequisites, run layout, evidence, verdicts.

Like its S05/S06 counterparts, the scaffolding has one central property: **it
cannot produce a conclusion by default**.

* :func:`check_prerequisites` inspects the repository for the *evidence* the S07
  protocol demands (S04.5 M4 marker, S05 quality/kernel verdicts, S06 stable
  capability verdicts, frozen request fixtures, a runtime capability probe and a
  main-runtime selection, plus a recorded environment fingerprint) and names the
  path it looked for;
* :meth:`RunDirectory.write_verdict` refuses PASS/FAIL/PASS_NEGATIVE unless the
  caller explicitly enables execution, the prerequisites are satisfied **and**
  raw samples exist;
* :meth:`RunDirectory.write_report_skeleton` writes a ``NOT_RUN`` stub with no
  numbers, so an unexecuted report can never be mistaken for a result;
* the run layout mirrors details README §18 and the evidence manifest carries the
  S07 fields of §16.

Two prerequisites deliberately live under ``docs/stage_experiments/S07/`` — the
protocol tree that this scaffolding never writes to — so S07's own interface
work cannot unlock S07's own experiments.
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

EXPERIMENTS: Tuple[str, ...] = tuple(f"E07-{index:02d}" for index in range(1, 11))

STAGE = "S07"

#: Details-README §18 run layout.
RUN_LAYOUT: Tuple[str, ...] = (
    "commands",
    "stdout",
    "stderr",
    "environment",
    "model_backend",
    "request_trace",
    "requests",
    "iterations",
    "scheduler",
    "kv",
    "prefix",
    "graph",
    "kernels",
    "correctness",
    "performance",
    "memory",
    "energy",
    "errors",
    "profiler",
)

#: The marker an actual S04.5 execution leaves behind.  S07's scaffolding lives
#: in a different package, but the marker is shared so the S04.5 gate has one
#: definition (see the S06 development report §2.3).
S04_5_EVIDENCE_MARKER = os.path.join("hqsb", "integration", "s04_5_evidence.json")

#: Where a real S07 execution records the probe and the primary-runtime choice.
#: The protocol tree is never written by this scaffolding.
S07_EVIDENCE_DIR = os.path.join("docs", "stage_experiments", "S07")

CAPABILITY_PROBE_NAME = "capability_probe.json"
MAIN_RUNTIME_SELECTION_NAME = "main_runtime_selection.json"

#: Candidate locations for the frozen request fixtures (model/tokenizer/sampling).
REQUEST_FIXTURE_CANDIDATES: Tuple[str, ...] = (
    os.path.join("docs", "stage_experiments", "S04.5", "s07_request_fixtures.json"),
    os.path.join("benchmarks", "workloads", "runtime", "frozen_request_spec.json"),
    os.path.join("docs", "stage_experiments", "S07", "s07_request_fixtures.json"),
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
    """Aggregate prerequisite status for S07."""

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


def _find_verdicts(root: str, stage: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(root, "docs", "stage_experiments", stage, "**", "verdict.json")
    verdicts: List[Dict[str, Any]] = []
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        payload["_path"] = path
        verdicts.append(payload)
    return verdicts


def check_prerequisites(root: str) -> PrerequisiteStatus:
    """Inspect the repository for the S07 evidence chain (never for intent)."""
    checks: List[PrerequisiteCheck] = []

    marker = os.path.join(root, S04_5_EVIDENCE_MARKER)
    checks.append(
        PrerequisiteCheck(
            name="s04_5_model_reintegration_evidence",
            satisfied=os.path.isfile(marker),
            evidence=marker if os.path.isfile(marker) else "",
            reason=(
                ""
                if os.path.isfile(marker)
                else (
                    "no S04.5 execution marker at hqsb/integration/s04_5_evidence.json; "
                    "the runtime experiments need the real Qwen semantics and the six "
                    "frozen workloads first"
                )
            ),
        )
    )

    s05_passing = [
        item
        for item in _find_verdicts(root, "S05")
        if item.get("status") in (STATUS_PASS, STATUS_PASS_NEGATIVE)
    ]
    checks.append(
        PrerequisiteCheck(
            name="s05_quality_and_kernel_evidence",
            satisfied=bool(s05_passing),
            evidence=s05_passing[0]["_path"] if s05_passing else "",
            reason=(
                ""
                if s05_passing
                else "no S05 verdict.json with PASS/PASS_NEGATIVE; the precision the "
                "runtime must execute is not quality-approved yet"
            ),
        )
    )

    s06_passing = [
        item
        for item in _find_verdicts(root, "S06")
        if item.get("status") in (STATUS_PASS, STATUS_PASS_NEGATIVE)
    ]
    checks.append(
        PrerequisiteCheck(
            name="s06_stable_capability_evidence",
            satisfied=bool(s06_passing),
            evidence=s06_passing[0]["_path"] if s06_passing else "",
            reason=(
                ""
                if s06_passing
                else "no S06 verdict.json with PASS/PASS_NEGATIVE; the compiled "
                "callable/guard/cache/fallback chain S07 consumes is unverified. "
                "S06 interface scaffolding in hqsb/integration is NOT execution "
                "evidence and must not be read as such"
            ),
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
                else "no frozen tokenized request fixtures (model/tokenizer/chat "
                "template/sampling/stop/input token IDs); without them each runtime "
                "would tokenize its own text and the comparison would be meaningless"
            ),
        )
    )

    probe = os.path.join(root, S07_EVIDENCE_DIR, CAPABILITY_PROBE_NAME)
    checks.append(
        PrerequisiteCheck(
            name="runtime_capability_probe",
            satisfied=os.path.isfile(probe),
            evidence=probe if os.path.isfile(probe) else "",
            reason=(
                ""
                if os.path.isfile(probe)
                else f"no runtime capability probe at {probe}; the probe must be "
                "produced on the target machine by the experiment run — scaffolding "
                "in hqsb/runtime cannot satisfy it"
            ),
        )
    )

    selection = os.path.join(root, S07_EVIDENCE_DIR, MAIN_RUNTIME_SELECTION_NAME)
    checks.append(
        PrerequisiteCheck(
            name="main_runtime_selection",
            satisfied=os.path.isfile(selection),
            evidence=selection if os.path.isfile(selection) else "",
            reason=(
                ""
                if os.path.isfile(selection)
                else f"no main-runtime selection at {selection}; the source-level "
                "primary runtime must be chosen from a capability probe, not assumed "
                "by a document (details README §6). Scaffolding in hqsb/runtime "
                "cannot satisfy it"
            ),
        )
    )

    # Only the protocol tree counts: run directories under ``experiment_results``
    # are written by this very scaffolding, so accepting them would let S07
    # unlock its own experiments (the same trap S06 closed for S04.5).
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
                else "no recorded environment fingerprint (Python/torch/CUDA/runtime "
                "version/commit) in docs/stage_experiments/**; a fingerprint written "
                "by this scaffolding under experiment_results/ does not count, and a "
                "run without a frozen toolchain identity cannot be reproduced"
            ),
        )
    )

    schema_note = ""
    schema_ok = True
    try:
        from hqsb.runtime import telemetry

        coverage = telemetry.c6_c7_summary()
        # The control plane requires runtime metrics to align with C2 as well.
        schema_ok = bool(
            coverage["c6"]["ok"] and coverage["c7"]["ok"] and coverage["c2"]["ok"]
        )
        if not schema_ok:
            schema_note = (
                "the C2/C6/C7 projection does not cover the required S07 fields "
                f"(c2 missing={coverage['c2']['missing']}, "
                f"c6 missing={coverage['c6']['missing']}, "
                f"c7 invalid={coverage['c7']['invalid']})"
            )
    except Exception as exc:  # noqa: BLE001 - a broken contract must block, not crash
        schema_ok = False
        schema_note = f"{type(exc).__name__}: {exc}"
    checks.append(
        PrerequisiteCheck(
            name="c6_c7_schema_available",
            satisfied=schema_ok,
            evidence="hqsb/runtime/telemetry.py" if schema_ok else "",
            reason=schema_note,
        )
    )

    return PrerequisiteStatus(stage=STAGE, checks=checks)


# ── preregistration ────────────────────────────────────────────────────────


@dataclass
class Preregistration:
    """The frozen S07 experiment plan (details README §13.1)."""

    experiment_id: str
    question: str
    hypothesis: str
    backend_roles: Mapping[str, str] = field(default_factory=dict)
    runtime_versions: Mapping[str, str] = field(default_factory=dict)
    model_artifact_hash: str = ""
    precision: str = ""
    request_trace_hash: str = ""
    scheduler_policy: str = ""
    kv_policy: str = ""
    prefix_policy: str = ""
    graph_policy: str = ""
    quality_gate: str = ""
    performance_metrics: Tuple[str, ...] = ()
    repeats: int = 0
    independent_processes: int = 3
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
                    f"preregistration for {self.experiment_id} needs a non-empty "
                    f"{name!r}; a hypothesis written after seeing results is not a "
                    "preregistration"
                )
        if not self.claim_boundary:
            raise ConfigError(
                f"preregistration for {self.experiment_id} needs an explicit "
                "claim_boundary (what may NOT be claimed from this experiment)",
                details={"field": "claim_boundary"},
            )
        if self.independent_processes < 3:
            raise ConfigError(
                "at least three independent runtime processes are required "
                "(details README §13.3)",
                details={"field": "independent_processes"},
            )

    def to_json(self) -> str:
        payload = {
            "schema_version": "1.0.0",
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "backend_roles": dict(self.backend_roles),
            "runtime_versions": dict(self.runtime_versions),
            "model_artifact_hash": self.model_artifact_hash,
            "precision": self.precision,
            "request_trace_hash": self.request_trace_hash,
            "scheduler_policy": self.scheduler_policy,
            "kv_policy": self.kv_policy,
            "prefix_policy": self.prefix_policy,
            "graph_policy": self.graph_policy,
            "quality_gate": self.quality_gate,
            "performance_metrics": list(self.performance_metrics),
            "repeats": self.repeats,
            "independent_processes": self.independent_processes,
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


# ── evidence manifest (details README §16 + control plane §26.3) ───────────


#: The unified per-experiment record of the execution handbook §4.  The names and
#: order come from that document; ``fallback_reason`` is added because §5.7
#: requires requested/actual implementations to carry a reason code.
EXPERIMENT_RECORD_FIELDS: Tuple[str, ...] = (
    "stage",
    "experiment_id",
    "status",
    "question",
    "hypothesis",
    "run_id",
    "git_commit",
    "git_dirty",
    "model_manifest_sha256",
    "config_sha256",
    "operator_or_binary_sha256",
    "environment_uri",
    "hardware_and_power_mode",
    "requested_implementation",
    "actual_implementation",
    "fallback_reason",
    "controls",
    "independent_variables",
    "correctness_metrics",
    "performance_samples_uri",
    "profile_artifacts_uri",
    "started_at",
    "ended_at",
    "decision",
    "limitations",
)


@dataclass
class ExperimentRecord:
    """One experiment's unified record (handbook §4 / §5.7).

    The validation rules are the handbook's: a record must state the question and
    a falsifiable hypothesis, an automatic selection must record
    requested/actual/reason, and a conclusion must point at raw evidence.  The
    class therefore refuses to describe a result that has no raw samples behind
    it — which is exactly what "未运行不得声明 PASS" means in code.
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
                f"unknown status {self.status!r}; protocol allows "
                f"{list(PROTOCOL_STATUSES)}"
            )
        for name in ("question", "hypothesis"):
            if not getattr(self, name):
                raise ConfigError(
                    f"the experiment record needs a non-empty {name!r}; a record "
                    "without a falsifiable question cannot be executed (handbook §5.1)",
                    details={"field": name},
                )
        if self.actual_implementation and self.requested_implementation:
            if (
                self.actual_implementation != self.requested_implementation
                and not self.fallback_reason
            ):
                raise ConfigError(
                    "actual implementation differs from the requested one without a "
                    "fallback_reason; automatic selection must record "
                    "requested/actual/reason (handbook §5.7)",
                    details={"field": "fallback_reason"},
                )
        if self.status in CONCLUSION_STATUSES:
            if not self.decision:
                raise ConfigError(
                    f"a {self.status} record needs a decision line",
                    details={"field": "decision"},
                )
            if not (self.performance_samples_uri or self.profile_artifacts_uri):
                raise ConfigError(
                    "a conclusion must point at its raw evidence "
                    "(performance_samples_uri or profile_artifacts_uri); "
                    "'未运行不得声明 PASS' (handbook §4)",
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
            "a template cannot be written with a conclusion status; run the "
            "experiment and build the record from real evidence",
            details={"field": "status"},
        )
    return ExperimentRecord(
        experiment_id=experiment_id,
        question="see docs/stage_experiments/details/S07/<file>",
        hypothesis="see docs/stage_experiments/details/S07/<file>",
        status=status,
        started_at="",
        ended_at="",
        decision="",
        limitations=[
            "template only; the executor must fill environment/model/config hashes, "
            "implementations, controls, metrics and raw evidence URIs",
            reason,
        ],
    )


@dataclass
class EvidenceManifest:
    """Run evidence manifest with the S07-specific fields."""

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
    # S07 extras (details README §16)
    backend_id: str = ""
    runtime_version: str = ""
    runtime_commit: str = ""
    requested_capability: Mapping[str, str] = field(default_factory=dict)
    actual_capability: Mapping[str, str] = field(default_factory=dict)
    capability_reasons: Mapping[str, str] = field(default_factory=dict)
    request_trace_hash: str = ""
    scheduler_policy: str = ""
    kv_policy: str = ""
    prefix_policy: str = ""
    graph_policy: str = ""
    per_request_metrics_uri: str = ""
    iteration_ledger_uri: str = ""
    actual_precision: str = ""
    observed_kernel: str = ""
    observed_attention_backend: str = ""
    failure: str = ""
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
            "runtime": {
                "backend_id": self.backend_id,
                "version": self.runtime_version,
                "commit": self.runtime_commit,
                "capability": {
                    "requested": dict(self.requested_capability),
                    "actual": dict(self.actual_capability),
                    "reasons": dict(self.capability_reasons),
                },
                "request_trace_hash": self.request_trace_hash,
                "policies": {
                    "scheduler": self.scheduler_policy,
                    "kv": self.kv_policy,
                    "prefix": self.prefix_policy,
                    "graph": self.graph_policy,
                },
                "per_request_metrics_uri": self.per_request_metrics_uri,
                "iteration_ledger_uri": self.iteration_ledger_uri,
                "actual_precision": self.actual_precision,
                "observed_kernel": self.observed_kernel,
                "observed_attention_backend": self.observed_attention_backend,
                "failure": self.failure,
                "memory_bytes": self.memory_bytes,
                "energy_joules": self.energy_joules,
            },
        }
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


# ── run directory ──────────────────────────────────────────────────────────


class RunDirectory(RunStorage):
    """Creates and manages one ``experiment_results/S07/<E>/<run_id>/`` tree."""

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
        for candidate in ("requests", "performance", "iterations"):
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
            f"> 状态：**NOT_RUN**（本文件由 `hqsb.runtime.experiment` 生成的骨架；\n"
            f"> 未执行任何实验，未产出任何数值）。\n\n"
            "## 1. 预注册\n\n"
            "见 `preregistration.json`。\n\n"
            "## 2. 环境与身份\n\n"
            "见 `environment_fingerprint.json`、`evidence_manifest.json`；未执行时为占位。\n\n"
            "## 3. 原始数据\n\n"
            "`requests/`、`iterations/`、`kv/`、`prefix/`、`graph/`、`performance/` 均为空，"
            "直到实验真正运行。\n\n"
            "## 4. 结论\n\n"
            "无。实验未执行，禁止填写任何容量/吞吐/延迟/命中率结论。\n"
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
                    "refusing to emit an experimental conclusion: execution is "
                    "disabled (pass allow_execute=True to the driver and run the "
                    "experiment properly)"
                )
            elif not prerequisites.satisfied:
                effective = STATUS_BLOCKED
                effective_reason = (
                    "prerequisites unsatisfied: " + ", ".join(prerequisites.missing)
                )
            elif not executed or raw_samples <= 0:
                effective = STATUS_BLOCKED
                effective_reason = (
                    "no raw samples recorded; a conclusion without raw evidence is "
                    "not allowed"
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
    }
    for module_name, key in (("torch", "torch"), ("triton", "triton")):
        try:
            module = __import__(module_name)
            payload[key] = str(getattr(module, "__version__", ""))
        except ImportError:
            payload[key] = "not installed"
    try:
        from hqsb.runtime import adapter as adapter_mod

        payload["runtime_engine_probe"] = [
            probe.as_dict() for probe in adapter_mod.probe_environment()
        ]
    except Exception as exc:  # noqa: BLE001 - probe must never crash the fingerprint
        payload["runtime_engine_probe"] = [{"error": f"{type(exc).__name__}: {exc}"}]
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            payload["cuda"] = {
                "name": properties.name,
                "capability": [properties.major, properties.minor],
                "total_memory_bytes": int(properties.total_memory),
            }
    except ImportError:  # pragma: no cover - CPU-minimal
        pass
    if extra:
        payload.update(dict(extra))
    return payload


def git_state(root: str) -> Dict[str, Any]:
    """Return ``{commit, dirty}``; a missing git binary is reported, not fatal."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "commit": commit.stdout.strip() if commit.returncode == 0 else "",
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else False,
            "error": "" if commit.returncode == 0 else commit.stderr.strip(),
        }
    except FileNotFoundError:
        return {"commit": "", "dirty": False, "error": "git not installed"}


def interface_only_run(
    root: str,
    experiment_id: str,
    *,
    run_id: Optional[str] = None,
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
    "CAPABILITY_PROBE_NAME",
    "CONCLUSION_STATUSES",
    "EXPERIMENTS",
    "EXPERIMENT_RECORD_FIELDS",
    "EvidenceManifest",
    "ExperimentRecord",
    "MAIN_RUNTIME_SELECTION_NAME",
    "PROTOCOL_STATUSES",
    "Preregistration",
    "PrerequisiteCheck",
    "PrerequisiteStatus",
    "REQUEST_FIXTURE_CANDIDATES",
    "RUN_LAYOUT",
    "RunDirectory",
    "S04_5_EVIDENCE_MARKER",
    "S07_EVIDENCE_DIR",
    "STAGE",
    "STATUS_BLOCKED",
    "STATUS_FAIL",
    "STATUS_N_A_BY_ADR",
    "STATUS_NOT_STARTED",
    "STATUS_PASS",
    "STATUS_PASS_NEGATIVE",
    "STATUS_RUNNING",
    "check_prerequisites",
    "environment_fingerprint",
    "git_state",
    "interface_only_run",
]
