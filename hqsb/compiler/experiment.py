"""S11 experiment scaffolding: prerequisites, preregistration, evidence, verdicts.

The scaffolding implements the full "preregister → collect → write → decide"
structure every S11 experiment needs, while making it *impossible by default*
to emit a conclusion:

* ``check_prerequisites`` reads the repository for real evidence paths (never
  for intent) and reports each check with its evidence pointer and reason;
* ``RunDirectory`` writes only under ``experiment_results/S11/...`` — never
  into ``docs/stage_experiments`` (the protocol tree is read-only);
* ``write_verdict`` refuses PASS/FAIL/PASS_NEGATIVE unless execution was
  explicitly allowed, the prerequisites are satisfied *and* raw samples exist;
* ``EvidenceManifest`` carries the S11 identity fields (graph/semantic/compile
  identities, IR levels, target/registry/cache/autotune/cost-model ids).

Nothing here runs an experiment.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import IR_LEVELS, canonical_json, sha256_text

# ── statuses and layout ────────────────────────────────────────────────────

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

STAGE = "S11"

EXPERIMENTS: Tuple[str, ...] = tuple(f"E11-{index:02d}" for index in range(1, 11))

RUN_LAYOUT: Tuple[str, ...] = (
    "identity",
    "capture",
    "passes",
    "lowering",
    "compiler",
    "autotune",
    "cost_model",
    "cache",
    "correctness",
    "benchmark",
    "profile",
    "commands",
    "stdout",
    "stderr",
)

RUN_ROOT_FILES: Tuple[str, ...] = (
    "preregistration.json",
    "experiment_record.json",
    "evidence_manifest.json",
    "verdict.json",
    "status.json",
    "report.md",
)

#: Evidence globs used by the prerequisite checks (protocol tree first).
S01_VERDICT_GLOB = "docs/stage_experiments/S01/*/raw*/verdict.json"
S02_VERDICT_GLOB = "docs/stage_experiments/S02/*/raw*/verdict.json"
S03_VERDICT_GLOB = "docs/stage_experiments/S03/*/raw*/verdict.json"
S04_VERDICT_GLOB = "docs/stage_experiments/S04/*/raw*/verdict.json"
S05_VERDICT_GLOB = "docs/stage_experiments/S05/*/raw*/verdict.json"
S06_VERDICT_GLOB = "docs/stage_experiments/S06/*/raw*/verdict.json"
DEV_BASELINE_GLOB = "reports/dev/*/gate*/s04_backend_baseline.json"
ENVIRONMENT_GLOB = "reports/dev/*/gate0_environment/environment.json"
FINGERPRINT_GLOB = "docs/stage_experiments/*/**/environment_fingerprint.json"


def _sha256_text(text: str) -> str:
    return sha256_text(text)


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
    required: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "satisfied": self.satisfied,
            "required": self.required,
            "evidence": self.evidence,
            "reason": self.reason,
        }


@dataclass
class PrerequisiteStatus:
    """Aggregate prerequisite status for S11."""

    stage: str
    checks: List[PrerequisiteCheck] = field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return all(check.satisfied for check in self.checks if check.required)

    @property
    def missing(self) -> List[str]:
        return [check.name for check in self.checks if check.required and not check.satisfied]

    @property
    def advisory(self) -> List[str]:
        return [check.name for check in self.checks if not check.required and not check.satisfied]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "satisfied": self.satisfied,
            "missing": self.missing,
            "advisory": self.advisory,
            "checks": [check.as_dict() for check in self.checks],
        }


def _verdict_status(payload: Mapping[str, Any]) -> str:
    """Verdict files across stages use ``status``, ``overall`` or ``passed``."""
    if "status" in payload:
        return str(payload["status"])
    if "overall" in payload:
        return str(payload["overall"])
    passed = payload.get("passed")
    if passed is True:
        return STATUS_PASS
    if passed is False:
        return STATUS_FAIL
    return ""


def _passing_verdicts(root: str, pattern: str) -> List[Dict[str, Any]]:
    passing: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(root, pattern), recursive=True)):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and _verdict_status(payload) in (
            STATUS_PASS,
            STATUS_PASS_NEGATIVE,
            "True",
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


def check_prerequisites(root: str) -> PrerequisiteStatus:
    """Inspect the repository for the S11 evidence chain (never for intent)."""
    checks: List[PrerequisiteCheck] = []

    s01 = _passing_verdicts(root, S01_VERDICT_GLOB)
    checks.append(
        PrerequisiteCheck(
            name="s01_contracts_and_identity",
            satisfied=bool(s01),
            evidence=s01[0]["_path"] if s01 else "",
            reason=(
                ""
                if s01
                else "no S01 PASS verdict for C1–C7/registry/schema: the contracts S11 builds "
                "on are unverified"
            ),
        )
    )

    s02 = _passing_verdicts(root, S02_VERDICT_GLOB)
    model_config = os.path.join(root, "configs", "models", "qwen3_1_7b.yaml")
    checks.append(
        PrerequisiteCheck(
            name="s02_qwen_reference_replayable",
            satisfied=bool(s02) and os.path.isfile(model_config),
            evidence=s02[0]["_path"] if s02 else model_config,
            reason=(
                ""
                if s02 and os.path.isfile(model_config)
                else "S02 verdicts and/or the frozen model config are missing: without a "
                "replayable Qwen reference the capture census has no denominator"
            ),
        )
    )

    s03 = _passing_verdicts(root, S03_VERDICT_GLOB)
    s04 = _passing_verdicts(root, S04_VERDICT_GLOB)
    dev_baseline = sorted(glob.glob(os.path.join(root, DEV_BASELINE_GLOB)))
    checks.append(
        PrerequisiteCheck(
            name="s03_s04_kernel_hardware_evidence",
            satisfied=bool(s03 or s04 or dev_baseline),
            evidence=(s03[0]["_path"] if s03 else (s04[0]["_path"] if s04 else (dev_baseline[0] if dev_baseline else ""))),
            reason=(
                ""
                if (s03 or s04 or dev_baseline)
                else "no S03/S04 PASS verdict and no development baseline JSON: the lowering "
                "targets have no correctness evidence to bind to"
            ),
        )
    )

    s06 = _passing_verdicts(root, S06_VERDICT_GLOB)
    checks.append(
        PrerequisiteCheck(
            name="s06_pattern_model_level_correctness",
            satisfied=bool(s06),
            evidence=s06[0]["_path"] if s06 else "",
            reason=(
                ""
                if s06
                else "no S06 PASS verdict: the real Qwen pattern has not reached model-level "
                "correctness through the custom op, so E11-02/E11-03 hero-path acceptance "
                "cannot be executed (toy graphs are not a substitute)"
            ),
        )
    )

    s05 = _passing_verdicts(root, S05_VERDICT_GLOB)
    checks.append(
        PrerequisiteCheck(
            name="s05_quant_artifact_chain",
            satisfied=bool(s05),
            evidence=s05[0]["_path"] if s05 else "",
            reason=(
                ""
                if s05
                else "no S05 PASS verdict: the quantized pattern branch is "
                "NOT_APPLICABLE_CAPABILITY until QuantArtifact/quality-gate evidence exists"
            ),
            required=False,
        )
    )

    fingerprint_candidates = sorted(
        glob.glob(os.path.join(root, FINGERPRINT_GLOB), recursive=True)
    ) + sorted(glob.glob(os.path.join(root, ENVIRONMENT_GLOB)))
    frozen_environment = _first_fingerprint_with_versions(fingerprint_candidates)
    checks.append(
        PrerequisiteCheck(
            name="frozen_compiler_environment",
            satisfied=bool(frozen_environment),
            evidence=frozen_environment,
            reason=(
                ""
                if frozen_environment
                else "no environment fingerprint with frozen torch/triton/arch values: compile "
                "identity would be incomplete"
            ),
        )
    )

    checks.append(
        PrerequisiteCheck(
            name="independent_evidence_dirs",
            satisfied=os.path.isfile(os.path.join(root, "hqsb", "compiler", "experiment.py")),
            evidence=os.path.join("hqsb", "compiler", "experiment.py"),
            reason=(
                ""
                if os.path.isfile(os.path.join(root, "hqsb", "compiler", "experiment.py"))
                else "the run-directory scaffolding is unavailable"
            ),
        )
    )

    driver = os.path.join(root, "scripts", "compiler", "run_e11.py")
    checks.append(
        PrerequisiteCheck(
            name="stable_cli_runner_for_gates",
            satisfied=os.path.isfile(driver),
            evidence=driver,
            reason=(
                ""
                if os.path.isfile(driver)
                else "no stable CLI driver: correctness/benchmark/cache-reset/fault-injection "
                "must be callable from one entry point"
            ),
        )
    )

    tools = _probe_export_tools()
    checks.append(
        PrerequisiteCheck(
            name="ir_binary_export_capability",
            satisfied=not tools["missing"],
            evidence=",".join(sorted(tools["present"])),
            reason=(
                ""
                if not tools["missing"]
                else "missing tools for IR/binary/profile export: " + ", ".join(tools["missing"])
            ),
            required=False,
        )
    )

    return PrerequisiteStatus(stage=STAGE, checks=checks)


def _first_fingerprint_with_versions(candidates: Sequence[str]) -> str:
    required = ("torch", "triton")
    for path in candidates:
        payload = _load_json(path)
        if not payload:
            continue
        blob = json.dumps(payload, sort_keys=True)
        if all(name in blob for name in required):
            return path
    return ""


def _probe_export_tools() -> Dict[str, Any]:
    names = ("nvcc", "cuobjdump", "nvdisasm", "nsys", "ncu")
    present = []
    missing = []
    for name in names:
        found = _which(name)
        if found:
            present.append(f"{name}={found}")
        else:
            missing.append(f"{name}:NOT_RUN_TOOL_UNAVAILABLE")
    return {"present": present, "missing": missing}


def _which(name: str) -> str:
    from shutil import which

    return which(name) or ""


# ── preregistration / experiment record / evidence manifest ────────────────


@dataclass
class Preregistration:
    """Written *before* a run; the hypothesis must be falsifiable."""

    experiment_id: str
    run_id: str
    hypothesis: str
    primary_metric: str
    thresholds: Mapping[str, Any]
    claim_boundary: str
    non_claims: Sequence[str]
    frozen_inputs: Mapping[str, str] = field(default_factory=dict)
    deviations: Sequence[str] = ()
    created_at: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.experiment_id not in EXPERIMENTS:
            problems.append(f"unknown experiment id {self.experiment_id!r}")
        for name in ("run_id", "hypothesis", "primary_metric", "claim_boundary"):
            if not getattr(self, name):
                problems.append(f"preregistration missing {name!r}")
        if not self.thresholds:
            problems.append("thresholds must be frozen before the run (no post-hoc gates)")
        if not self.non_claims:
            problems.append("preregistration must state what the run will NOT claim")
        return problems

    def payload(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "hypothesis": self.hypothesis,
            "primary_metric": self.primary_metric,
            "thresholds": dict(sorted(self.thresholds.items())),
            "claim_boundary": self.claim_boundary,
            "non_claims": list(self.non_claims),
            "frozen_inputs": dict(sorted(self.frozen_inputs.items())),
            "deviations": list(self.deviations),
            "created_at": self.created_at or _utc_now(),
        }

    def to_json(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, indent=2, ensure_ascii=False)

    def prereg_hash(self) -> str:
        return _sha256_text(canonical_json(self.payload()))


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
    """The handbook §4 record; every field is present even when empty."""

    experiment_id: str
    status: str = STATUS_NOT_STARTED
    question: str = ""
    hypothesis: str = ""
    run_id: str = ""
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

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.experiment_id not in EXPERIMENTS:
            problems.append(f"unknown experiment id {self.experiment_id!r}")
        if self.status not in PROTOCOL_STATUSES:
            problems.append(f"unknown status {self.status!r}")
        if self.actual_implementation and self.requested_implementation != self.actual_implementation:
            if not self.fallback_reason:
                problems.append(
                    "requested != actual without a fallback reason (silent fallback is forbidden)"
                )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": STAGE,
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
            "controls": dict(sorted(self.controls.items())),
            "independent_variables": dict(sorted(self.independent_variables.items())),
            "correctness_metrics": dict(sorted(self.correctness_metrics.items())),
            "performance_samples_uri": self.performance_samples_uri,
            "profile_artifacts_uri": self.profile_artifacts_uri,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "decision": self.decision,
            "limitations": list(self.limitations),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2, ensure_ascii=False)


def template_experiment_record(experiment_id: str, *, run_id: str = "", commit: str = "") -> ExperimentRecord:
    return ExperimentRecord(experiment_id=experiment_id, run_id=run_id, git_commit=commit)


# ── evidence manifest (§26.3 + S11 extensions) ─────────────────────────────

EVIDENCE_MANIFEST_FIELDS: Tuple[str, ...] = (
    "schema_version",
    "run_id",
    "stage",
    "claim_ids",
    "git",
    "model_artifact",
    "config",
    "operator_artifacts",
    "environment",
    "commands",
    "raw_artifacts",
    "normalized_artifacts",
    "reports",
    "correctness_status",
    "claim_level",
    "limitations",
    # S11 extensions
    "capture_mode",
    "graph_identity",
    "semantic_identity",
    "compile_identity",
    "ir_levels",
    "target_snapshot_sha256",
    "lowering_registry_digest",
    "selected_lowering",
    "actual_kernel",
    "guard_set_id",
    "artifact_manifest_id",
    "cache_key",
    "cache_layer",
    "cache_hit",
    "autotune_session_id",
    "cost_model_id",
    "compile_phase_times",
)

CLAIM_LEVELS: Tuple[str, ...] = ("SOURCE", "TEST", "RUNTIME", "MODEL", "SERVICE", "PORTABLE")


@dataclass
class EvidenceManifest:
    run_id: str
    stage: str = STAGE
    claim_ids: Tuple[str, ...] = ()
    git: Mapping[str, Any] = field(default_factory=dict)
    model_artifact: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    operator_artifacts: Sequence[Mapping[str, Any]] = ()
    environment: Mapping[str, Any] = field(default_factory=dict)
    commands: Sequence[Sequence[str]] = ()
    raw_artifacts: Sequence[Mapping[str, Any]] = ()
    normalized_artifacts: Sequence[Mapping[str, Any]] = ()
    reports: Sequence[str] = ()
    correctness_status: str = "not_run"
    claim_level: str = "SOURCE"
    limitations: Sequence[str] = ()
    capture_mode: str = ""
    graph_identity: str = ""
    semantic_identity: str = ""
    compile_identity: str = ""
    ir_levels: Tuple[str, ...] = ()
    target_snapshot_sha256: str = ""
    lowering_registry_digest: str = ""
    selected_lowering: str = ""
    actual_kernel: str = ""
    guard_set_id: str = ""
    artifact_manifest_id: str = ""
    cache_key: str = ""
    cache_layer: str = ""
    cache_hit: Optional[bool] = None
    autotune_session_id: str = ""
    cost_model_id: str = ""
    compile_phase_times: Mapping[str, float] = field(default_factory=dict)
    schema_version: str = "1.0.0"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.run_id:
            problems.append("manifest needs a run_id")
        if self.claim_level not in CLAIM_LEVELS:
            problems.append(f"unknown claim level {self.claim_level!r}")
        for level in self.ir_levels:
            if level not in IR_LEVELS:
                problems.append(f"unknown IR level in manifest: {level!r}")
        for artifact in list(self.raw_artifacts) + list(self.normalized_artifacts):
            if "uri" not in artifact or "sha256" not in artifact:
                problems.append(f"artifact row without uri/sha256: {artifact}")
        return problems

    def payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "stage": self.stage,
            "claim_ids": list(self.claim_ids),
            "git": dict(sorted(self.git.items())),
            "model_artifact": dict(sorted(self.model_artifact.items())),
            "config": dict(sorted(self.config.items())),
            "operator_artifacts": [dict(item) for item in self.operator_artifacts],
            "environment": dict(sorted(self.environment.items())),
            "commands": [list(item) for item in self.commands],
            "raw_artifacts": [dict(item) for item in self.raw_artifacts],
            "normalized_artifacts": [dict(item) for item in self.normalized_artifacts],
            "reports": list(self.reports),
            "correctness_status": self.correctness_status,
            "claim_level": self.claim_level,
            "limitations": list(self.limitations),
            "capture_mode": self.capture_mode,
            "graph_identity": self.graph_identity,
            "semantic_identity": self.semantic_identity,
            "compile_identity": self.compile_identity,
            "ir_levels": list(self.ir_levels),
            "target_snapshot_sha256": self.target_snapshot_sha256,
            "lowering_registry_digest": self.lowering_registry_digest,
            "selected_lowering": self.selected_lowering,
            "actual_kernel": self.actual_kernel,
            "guard_set_id": self.guard_set_id,
            "artifact_manifest_id": self.artifact_manifest_id,
            "cache_key": self.cache_key,
            "cache_layer": self.cache_layer,
            "cache_hit": self.cache_hit,
            "autotune_session_id": self.autotune_session_id,
            "cost_model_id": self.cost_model_id,
            "compile_phase_times": dict(sorted(self.compile_phase_times.items())),
        }

    def to_json(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, indent=2, ensure_ascii=False)

    def as_dict(self) -> Dict[str, Any]:
        return self.payload()


# ── run directory ──────────────────────────────────────────────────────────


class RunDirectory:
    """Creates and manages one ``experiment_results/S11/<E>/<run_id>/`` tree."""

    def __init__(self, root: str, experiment_id: str, run_id: str) -> None:
        if experiment_id not in EXPERIMENTS:
            raise ConfigError(f"unknown experiment id {experiment_id!r}")
        if not run_id:
            raise ConfigError("run_id must not be empty")
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.path = os.path.abspath(
            os.path.join(root, "experiment_results", STAGE, experiment_id, run_id)
        )

    def create(self) -> str:
        os.makedirs(self.path, exist_ok=True)
        for sub in RUN_LAYOUT:
            os.makedirs(os.path.join(self.path, sub), exist_ok=True)
        return self.path

    def write_json(self, relative: str, payload: Any) -> str:
        path = os.path.join(self.path, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
        return path

    def write_text(self, relative: str, text: str) -> str:
        path = os.path.join(self.path, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def write_jsonl(self, relative: str, rows: Sequence[Mapping[str, Any]]) -> str:
        path = os.path.join(self.path, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
        return path

    def record_command(
        self,
        index: int,
        command: Sequence[str],
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
    ) -> Dict[str, Any]:
        name = f"{index:02d}_{'_'.join(part for part in command[:3] if part)}"[:60]
        self.write_json(
            os.path.join("commands", f"{name}.json"),
            {"command": list(command), "returncode": returncode, "cwd": os.getcwd()},
        )
        if stdout:
            self.write_text(os.path.join("stdout", f"{name}.stdout"), stdout)
        if stderr:
            self.write_text(os.path.join("stderr", f"{name}.stderr"), stderr)
        return {
            "command": list(command),
            "returncode": returncode,
            "stdout": f"stdout/{name}.stdout" if stdout else "",
            "stderr": f"stderr/{name}.stderr" if stderr else "",
        }

    def write_preregistration(self, prereg: Preregistration) -> str:
        return self.write_text("preregistration.json", prereg.to_json())

    def write_experiment_record(self, record: ExperimentRecord) -> str:
        return self.write_text("experiment_record.json", record.to_json())

    def write_evidence_manifest(self, manifest: EvidenceManifest) -> str:
        return self.write_text("evidence_manifest.json", manifest.to_json())

    def raw_file_count(self) -> int:
        """Count raw artifacts, preferring the compiler/benchmark directories."""
        for candidate in (
            "compiler",
            "capture",
            "passes",
            "lowering",
            "autotune",
            "cost_model",
            "cache",
            "correctness",
            "benchmark",
            "profile",
        ):
            candidate_path = os.path.join(self.path, candidate)
            if os.path.isdir(candidate_path):
                count = sum(len(files) for _root, _dirs, files in os.walk(candidate_path))
                if count:
                    return count
        return 0

    def write_report_skeleton(self, experiment_id: str, title: str) -> str:
        """Write a report stub that cannot be mistaken for a result."""
        skeleton = (
            f"# {experiment_id}: {title}\n\n"
            f"> 状态：**NOT_RUN**（本文件由 `hqsb.compiler.experiment` 生成的骨架；\n"
            f"> 未执行任何实验，未产出任何数值）。\n\n"
            "## 1. 预注册\n\n见 `preregistration.json`。\n\n"
            "## 2. 环境与身份\n\n见 `environment_fingerprint.json`、`evidence_manifest.json`；"
            "未执行时为占位。\n\n"
            "## 3. 原始数据\n\n`capture/`、`passes/`、`lowering/`、`compiler/`、`autotune/`、"
            "`cache/` 均为空，直到实验真正运行。\n\n"
            "## 4. 结论\n\n无。实验未执行，禁止填写任何 capture/lowering/autotune/cache 结论。\n"
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
    }
    for module_name, key in (("torch", "torch"), ("triton", "triton")):
        payload[key] = _optional_module_version(module_name)
    payload["cuda_driver"] = _cuda_driver_version()
    payload["tools"] = _probe_export_tools()
    if extra:
        payload["extra"] = dict(extra)
    return payload


def _optional_module_version(name: str) -> str:
    try:
        import importlib

        module = importlib.import_module(name)
    except Exception:
        return ""
    return str(getattr(module, "__version__", ""))


def _cuda_driver_version() -> str:
    if not _which("nvidia-smi"):
        return ""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = completed.stdout.strip().splitlines()
    return lines[0].strip() if lines else ""


def git_state(root: str) -> Dict[str, Any]:
    """Commit + dirty flag + diff hash (never rewrites anything)."""
    commit = _git(root, ["rev-parse", "HEAD"])
    status = _git(root, ["status", "--porcelain"])
    diff = _git(root, ["diff", "--no-color"])
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "dirty_files": sorted(line[3:] for line in status.splitlines() if line.strip()),
        "diff_sha256": _sha256_text(diff) if diff else "",
    }


def _git(root: str, args: Sequence[str]) -> str:
    if not _which("git"):
        return ""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def interface_only_run(
    root: str, experiment_id: str, *, run_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a run directory that records *interfaces only* (no experiment)."""
    from hqsb.compiler import interface_map

    rid = run_id or f"interface_only_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run = RunDirectory(root, experiment_id, rid)
    run.create()
    status = check_prerequisites(root)
    run.write_json("environment_fingerprint.json", environment_fingerprint())
    run.write_json("prerequisites.json", status.as_dict())
    run.write_json("interface_map.json", interface_map.mapping_for(experiment_id).as_dict())
    run.write_report_skeleton(experiment_id, interface_map.mapping_for(experiment_id).title)
    verdict = run.write_verdict(
        status=STATUS_BLOCKED,
        reason=(
            "interface-only scaffolding run: no experiment executed"
            + ("" if status.satisfied else "; prerequisites unsatisfied: " + ", ".join(status.missing))
        ),
        prerequisites=status,
        executed=False,
        allow_execute=False,
        raw_samples=0,
    )
    return {"run_dir": run.path, "verdict": verdict, "prerequisites": status.as_dict()}
