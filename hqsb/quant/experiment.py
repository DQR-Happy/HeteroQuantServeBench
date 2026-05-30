"""Experiment scaffolding: preregistration, run layout, evidence manifest, verdicts.

S05 experiments are *not* executed by this repository state: the hard
prerequisite (S04.5 M4) is unmet. The scaffolding therefore has one central
property — **it cannot produce a conclusion by default**:

* :func:`check_prerequisites` reads the repository for the S04.5 M4 evidence
  and reports what is missing; the result feeds every verdict;
* :func:`finalize_status` maps ``(prerequisites, executed, evidence)`` to one
  of the protocol statuses, and returns ``BLOCKED`` whenever the prerequisite
  chain is incomplete — regardless of how much interface code exists;
* :meth:`RunDirectory.write_verdict` refuses to write a PASS/FAIL/PASS_NEGATIVE
  verdict unless the caller passes ``allow_execute=True`` *and* the
  prerequisites are satisfied *and* the run recorded at least one raw sample;
* the run directory layout and the evidence manifest follow the S05 protocol
  (``preregistration.json``, ``evidence_manifest.json``, ``raw/``,
  ``normalized/``, ``quant_artifacts/``, ...), so a future execution lands in
  the structure the protocol expects.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hqsb.core.errors import ConfigError

#: Protocol statuses (docs/stage_experiments/README.md §3).
STATUS_NOT_STARTED = "NOT_STARTED"
STATUS_RUNNING = "RUNNING"
STATUS_PASS = "PASS"
STATUS_PASS_NEGATIVE = "PASS_NEGATIVE"
STATUS_FAIL = "FAIL"
STATUS_BLOCKED = "BLOCKED"
STATUS_N_A_BY_ADR = "N/A_BY_ADR"

PROTOCOL_STATUSES = (
    STATUS_NOT_STARTED,
    STATUS_RUNNING,
    STATUS_PASS,
    STATUS_PASS_NEGATIVE,
    STATUS_FAIL,
    STATUS_BLOCKED,
    STATUS_N_A_BY_ADR,
)

#: Statuses that require real execution evidence and are therefore refused by
#: default.
CONCLUSION_STATUSES = (STATUS_PASS, STATUS_PASS_NEGATIVE, STATUS_FAIL)

EXPERIMENTS = tuple(f"E05-{index:02d}" for index in range(1, 11))

STAGE = "S05"

#: The run directory layout from details README §13 (created eagerly).
RUN_LAYOUT = (
    "commands",
    "stdout",
    "stderr",
    "raw",
    "normalized",
    "quant_artifacts",
    "compiler",
    "profiler",
    "plots",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── prerequisites ─────────────────────────────────────────────────────────


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
    """Aggregate prerequisite status for the S05 stage."""

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


def check_prerequisites(root: str) -> PrerequisiteStatus:
    """Inspect the repository for the S04.5 M4 prerequisite chain.

    The checks are deliberately about *evidence*, not about intent:

    * an integration module that can replace an operator in a real model
      (``hqsb/integration`` or the torch-op binding the S04.5 plan names);
    * the S04.5 experiment raw/evidence directories;
    * the S04.5 acceptance report;
    * a frozen FP16 six-workload baseline bound to the current model hash.

    A missing item is reported with the path that was looked for, so the
    reason can be verified by hand (E05-02 §3: any missing condition makes the
    run exploratory only).
    """
    checks: List[PrerequisiteCheck] = []

    integration_candidates = [
        os.path.join(root, "hqsb", "integration"),
        os.path.join(root, "hqsb", "integration", "torch_ops.py"),
        os.path.join(root, "hqsb", "integration", "module_swap.py"),
    ]
    existing_integration = [path for path in integration_candidates if os.path.exists(path)]
    checks.append(
        PrerequisiteCheck(
            name="s04_5_operator_integration",
            satisfied=bool(existing_integration),
            evidence=existing_integration[0] if existing_integration else "",
            reason=(
                ""
                if existing_integration
                else "no hqsb/integration module found; the reversible operator "
                "replacement that S05 depends on does not exist in the tree"
            ),
        )
    )

    raw_root = os.path.join(root, "docs", "stage_experiments", "S04.5")
    raw_evidence = os.path.isdir(raw_root)
    checks.append(
        PrerequisiteCheck(
            name="s04_5_experiment_evidence",
            satisfied=raw_evidence,
            evidence=raw_root if raw_evidence else "",
            reason="" if raw_evidence else f"no S04.5 experiment evidence under {raw_root}",
        )
    )

    acceptance_candidates = [
        os.path.join(root, "docs", "reports", "S04.5_阶段验收报告.md"),
        os.path.join(root, "docs", "reports", "S04p5_阶段验收报告.md"),
    ]
    acceptance = [path for path in acceptance_candidates if os.path.exists(path)]
    checks.append(
        PrerequisiteCheck(
            name="s04_5_acceptance_report",
            satisfied=bool(acceptance),
            evidence=acceptance[0] if acceptance else "",
            reason=(
                ""
                if acceptance
                else "no S04.5 acceptance report; the M4 claim is unverified"
            ),
        )
    )

    baseline_candidates = [
        os.path.join(root, "docs", "stage_experiments", "S04.5", "six_workload_fp16_baseline.json"),
        os.path.join(root, "benchmarks", "normalized", "six_workload_fp16_baseline.json"),
    ]
    baseline = [path for path in baseline_candidates if os.path.exists(path)]
    checks.append(
        PrerequisiteCheck(
            name="six_workload_fp16_baseline",
            satisfied=bool(baseline),
            evidence=baseline[0] if baseline else "",
            reason=(
                ""
                if baseline
                else "no frozen six-workload FP16 baseline bound to the current "
                "model hash; without it quantization error cannot be separated "
                "from operator re-integration error"
            ),
        )
    )

    return PrerequisiteStatus(stage=STAGE, checks=checks)


# ── preregistration ───────────────────────────────────────────────────────


@dataclass
class Preregistration:
    """The frozen experiment plan (details README §9.1)."""

    experiment_id: str
    question: str
    hypothesis: str
    controls: Dict[str, Any] = field(default_factory=dict)
    independent_variables: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    gates: Dict[str, Any] = field(default_factory=dict)
    guard_band: Dict[str, float] = field(default_factory=dict)
    seeds: Sequence[int] = ()
    repeats: int = 0
    independent_processes: int = 0
    exclusions: Sequence[str] = ()
    stop_conditions: Sequence[str] = ()
    allowed_claims: Sequence[str] = ()
    hardware: str = ""
    backend: str = ""
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
                    f"preregistration"
                )

    def to_json(self) -> str:
        payload = {
            "schema_version": "1.0.0",
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "controls": self.controls,
            "independent_variables": self.independent_variables,
            "metrics": self.metrics,
            "gates": self.gates,
            "guard_band": self.guard_band,
            "seeds": list(self.seeds),
            "repeats": self.repeats,
            "independent_processes": self.independent_processes,
            "exclusions": list(self.exclusions),
            "stop_conditions": list(self.stop_conditions),
            "allowed_claims": list(self.allowed_claims),
            "hardware": self.hardware,
            "backend": self.backend,
            "notes": self.notes,
            "created_at": self.created_at or _utc_now(),
        }
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)

    @property
    def prereg_hash(self) -> str:
        return _sha256_text(self.to_json())


# ── run directory and evidence ────────────────────────────────────────────


@dataclass
class EvidenceManifest:
    """Run evidence manifest (control plane §26.3)."""

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
        }
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


class RunDirectory:
    """Creates and manages one ``experiment_results/S05/<E>/<run_id>/`` tree."""

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

    def record_command(self, index: int, command: Sequence[str], stdout: str = "", stderr: str = "", returncode: int = 0) -> Dict[str, Any]:
        """Persist one command with its streams (details README §4)."""
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

    def write_evidence_manifest(self, manifest: EvidenceManifest) -> str:
        return self.write_text("evidence_manifest.json", manifest.to_json())

    def raw_file_count(self) -> int:
        raw = os.path.join(self.path, "raw")
        if not os.path.isdir(raw):
            return 0
        return sum(len(files) for _root, _dirs, files in os.walk(raw))

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
        """Write a verdict — or refuse when the run cannot support one.

        Refusal conditions (all of them deliberate):

        * ``status`` is a conclusion status and ``allow_execute`` is False
          (the default), so interface work can never emit a result;
        * prerequisites are unsatisfied (→ the written status is ``BLOCKED``);
        * a conclusion is requested without raw samples.
        """
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
                    "no raw samples recorded; a conclusion without raw evidence "
                    "is not allowed"
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

    def write_status(self, status: str, reason: str, prerequisites: PrerequisiteStatus, executed: bool = False) -> Dict[str, Any]:
        """Write a non-conclusion status (NOT_STARTED / RUNNING / BLOCKED)."""
        if status in CONCLUSION_STATUSES:
            raise ConfigError(
                f"use write_verdict for conclusion statuses, got {status!r}"
            )
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


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── environment and git helpers ───────────────────────────────────────────


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
    """Create a run directory and record a non-conclusion status.

    This is what the CLI does by default: it materialises the structure the
    protocol expects, records the prerequisite report, and writes
    ``NOT_STARTED`` (or ``BLOCKED`` when a prerequisite is missing) — never a
    result.
    """
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
    "CONCLUSION_STATUSES",
    "EXPERIMENTS",
    "EvidenceManifest",
    "PROTOCOL_STATUSES",
    "Preregistration",
    "PrerequisiteCheck",
    "PrerequisiteStatus",
    "RUN_LAYOUT",
    "RunDirectory",
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
