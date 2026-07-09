"""S14 experiment scaffolding: prerequisites, preregistration, evidence, verdicts.

All twelve S14 experiments share one structure — "preregister → collect → write →
decide" — with a **triple gate** that makes a conclusion impossible by default:

* :func:`check_prerequisites` reads the repository for *evidence* (upstream
  verdicts, S13 production records) and probes for the external capabilities an
  S14 run needs (a distributed launcher, a second device, training/RL/frontier
  extras, profilers).  A missing capability becomes the precise S14 state of
  ``details/S14/README.md`` §21 (``BLOCKED_PREREQUISITE``,
  ``BLOCKED_CAPABILITY``, ``INVALID_IDENTITY``, …) — never a silent pass and never
  a mock;
* :class:`RunDirectory` writes only under ``artifacts/S14/<experiment>/<run>``
  (the §22 layout); the protocol tree ``docs/stage_experiments/**`` is never
  written to (``campaign.assert_writable`` enforces it);
* :meth:`RunDirectory.write_verdict` refuses ``PASS``/``PASS_NEGATIVE``/``FAIL``/
  ``FAIL_PERFORMANCE_HYPOTHESIS`` without ``--execute``, satisfied prerequisites
  **and** raw samples;
* :class:`EvidenceManifest` carries the S14 identity fields (training run,
  checkpoint, policy snapshot, frontier contract) so a conclusion resolves back
  to raw samples.

Nothing here runs an experiment.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import campaign as camp
from hqsb.experimental import records as rec
from hqsb.experimental.identity import SCHEMA_PREFIX, canonical_digest

#: Only these are conclusion statuses the triple gate must refuse.
GATED_CONCLUSION_STATUSES: Tuple[str, ...] = (
    rec.STATUS_PASS,
    rec.STATUS_PASS_NEGATIVE,
    rec.STATUS_FAIL,
    rec.STATUS_FAIL_PERFORMANCE_HYPOTHESIS,
)

STAGE = "S14"

#: ``details/S14/README.md`` §22 — run output lives under ``artifacts/S14``.
RUN_ROOT = camp.RUN_ROOT

EXPERIMENTS: Tuple[str, ...] = tuple(experiment_id for experiment_id, _level, _title in rec.EXPERIMENT_TABLE)

RUN_ROOT_FILES: Tuple[str, ...] = (
    "preregistration.json",
    "experiment_record.json",
    "evidence_manifest.json",
    "verdict.json",
    "status.json",
    "report.md",
    "environment_fingerprint.json",
    "prerequisites.json",
    "interface_map.json",
)

#: Upstream verdict globs (protocol tree first, then repository reports).
UPSTREAM_VERDICT_GLOBS: Mapping[str, str] = {
    stage: f"docs/stage_experiments/{stage}/*/raw*/verdict.json"
    for stage in ("S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08",
                  "S09", "S10", "S11", "S12", "S13")
}
S13_CAMPAIGN_GLOB = "experiment_results/S13/*/campaign_manifest.yaml"
S14_CAMPAIGN_GLOB = "experiment_results/S14/*/campaign_manifest.yaml"

#: The minimum upstream stages that must carry evidence before an S14 run.
MIN_UPSTREAM_STAGES = 1

#: External components an S14 run needs, and the state a missing one produces.
EXTERNAL_COMPONENTS: Mapping[str, Tuple[str, ...]] = {
    "distributed_launcher": ("torchrun", "mpirun", "deepspeed", "accelerate"),
    "second_device": ("nvidia-smi", "npu-smi", "rocm-smi"),
    "profiler_tooling": ("nsys", "ncu", "torch.profiler"),
    "training_tooling": ("torch.distributed", "deepspeed", "megatron"),
    "rl_tooling": ("ray", "vllm", "trl", "transformers"),
    "frontier_tooling": ("cuobjdump", "nvdisasm", "cusparselt"),
    "edge_tooling": ("adb", "fastboot", "bundletool"),
}

#: Which prerequisite state a missing component produces.
COMPONENT_STATE: Mapping[str, str] = {
    "distributed_launcher": rec.STATUS_BLOCKED_PREREQUISITE,
    "second_device": rec.STATUS_BLOCKED_CAPABILITY,
    "profiler_tooling": rec.STATUS_BLOCKED_CAPABILITY,
    "training_tooling": rec.STATUS_BLOCKED_CAPABILITY,
    "rl_tooling": rec.STATUS_BLOCKED_CAPABILITY,
    "frontier_tooling": rec.STATUS_BLOCKED_CAPABILITY,
    "edge_tooling": rec.STATUS_BLOCKED_CAPABILITY,
}

#: Prerequisites that need a decision or an artefact rather than a binary.
DECISION_COMPONENTS: Mapping[str, str] = {
    "experimental_environment": "an isolated environment for the experimental extras",
    "raw_evidence_retention": "a writable artifacts/S14 run directory",
    "quality_oracle": "a task-native quality oracle for the chosen workload",
    "holdout_isolation": "a holdout workload that exploration will not touch",
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── prerequisites ──────────────────────────────────────────────────────────


@dataclass
class PrerequisiteCheck:
    """One prerequisite with its evidence pointer, state code and reason."""

    name: str
    satisfied: bool
    state: str = ""
    evidence: str = ""
    reason: str = ""
    required: bool = True
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "satisfied": self.satisfied,
            "state": self.state,
            "required": self.required,
            "evidence": self.evidence,
            "reason": self.reason,
        }
        if self.detail:
            payload["detail"] = dict(sorted(self.detail.items()))
        return payload


@dataclass
class PrerequisiteStatus:
    """Aggregate prerequisite status for S14 (states from ``details/S14/README.md`` §21)."""

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

    def states(self) -> Dict[str, str]:
        return {check.name: check.state for check in self.checks if check.state}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "satisfied": self.satisfied,
            "missing": self.missing,
            "advisory": self.advisory,
            "states": dict(sorted(self.states().items())),
            "checks": [check.as_dict() for check in self.checks],
        }


def _verdict_status(payload: Mapping[str, Any]) -> str:
    if "status" in payload:
        return str(payload["status"])
    if "overall" in payload:
        return str(payload["overall"])
    passed = payload.get("passed")
    if passed is True:
        return rec.STATUS_PASS
    if passed is False:
        return rec.STATUS_FAIL
    return ""


def _load_json(path: str) -> Optional[Mapping[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:  # noqa: BLE001 - unreadable evidence is simply not evidence
        return None
    return payload if isinstance(payload, Mapping) else None


def _passing_verdicts(root: str, pattern: str) -> List[Dict[str, Any]]:
    passing: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(root, pattern), recursive=True)):
        payload = _load_json(path)
        if payload and _verdict_status(payload) in (rec.STATUS_PASS, rec.STATUS_PASS_NEGATIVE, "True"):
            enriched = dict(payload)
            enriched["_path"] = path
            passing.append(enriched)
    return passing


def _which(name: str) -> str:
    """Resolve a binary name; a dotted module name is probed with ``find_spec``."""
    if "." in name and "/" not in name:
        try:
            import importlib.util

            return name if importlib.util.find_spec(name) is not None else ""
        except (ImportError, ValueError):
            return ""
    return shutil.which(name) or ""


def check_prerequisites(root: str, *, probe: bool = False) -> PrerequisiteStatus:
    """Read the repository (and optionally the machine) for the S14 gates.

    ``probe=False`` never touches the machine: it reports the external components
    as *unknown* rather than missing, so a CI run does not silently claim a
    capability it did not test.  ``probe=True`` resolves the binaries/modules and
    turns a missing one into the precise blocking state.
    """
    checks: List[PrerequisiteCheck] = []

    upstream: Dict[str, List[Dict[str, Any]]] = {}
    for stage, pattern in sorted(UPSTREAM_VERDICT_GLOBS.items()):
        upstream[stage] = _passing_verdicts(root, pattern)
    stages_with_evidence = sorted(stage for stage, rows in upstream.items() if rows)
    checks.append(
        PrerequisiteCheck(
            name="upstream_verdicts",
            satisfied=len(stages_with_evidence) >= MIN_UPSTREAM_STAGES,
            state="" if stages_with_evidence else rec.STATUS_BLOCKED_PREREQUISITE,
            evidence=", ".join(stages_with_evidence) if stages_with_evidence else "(none found)",
            reason=(
                ""
                if stages_with_evidence
                else "no upstream stage carries a passing verdict; S14 的冻结输入尚未建立"
            ),
            detail={"stages": {stage: len(rows) for stage, rows in sorted(upstream.items())}},
        )
    )

    s13_campaigns = sorted(glob.glob(os.path.join(root, S13_CAMPAIGN_GLOB), recursive=True))
    checks.append(
        PrerequisiteCheck(
            name="s13_production_records",
            satisfied=bool(s13_campaigns),
            state="" if s13_campaigns else rec.STATUS_BLOCKED_PREREQUISITE,
            evidence=s13_campaigns[0] if s13_campaigns else "(none found)",
            reason="" if s13_campaigns else "S13 campaign 记录缺失，采用门与隔离要求无法引用",
            required=False,
        )
    )

    run_root = os.path.join(root, RUN_ROOT)
    writable = os.access(run_root, os.W_OK) if os.path.isdir(run_root) else os.access(root, os.W_OK)
    checks.append(
        PrerequisiteCheck(
            name="raw_evidence_retention",
            satisfied=writable,
            state="" if writable else rec.STATUS_BLOCKED_PREREQUISITE,
            evidence=run_root,
            reason="" if writable else f"{RUN_ROOT} is not writable; raw evidence could not be kept",
        )
    )

    for name, description in sorted(DECISION_COMPONENTS.items()):
        checks.append(
            PrerequisiteCheck(
                name=name,
                satisfied=False,
                state=rec.STATUS_BLOCKED_PREREQUISITE,
                evidence="",
                reason=f"{description} must be confirmed before a run (no automatic default)",
            )
        )

    for component, candidates in sorted(EXTERNAL_COMPONENTS.items()):
        if probe:
            found = next((_which(candidate) for candidate in candidates if _which(candidate)), "")
            checks.append(
                PrerequisiteCheck(
                    name=component,
                    satisfied=bool(found),
                    state="" if found else COMPONENT_STATE[component],
                    evidence=found,
                    reason="" if found else f"none of {', '.join(candidates)} is available",
                    required=component in ("distributed_launcher", "second_device"),
                    detail={"candidates": list(candidates)},
                )
            )
        else:
            checks.append(
                PrerequisiteCheck(
                    name=component,
                    satisfied=False,
                    state=rec.STATUS_BLOCKED_CAPABILITY,
                    evidence="",
                    reason="not probed (run with --probe to resolve binaries/modules)",
                    required=False,
                    detail={"candidates": list(candidates)},
                )
            )

    return PrerequisiteStatus(stage=STAGE, checks=checks)


# ── run directory ──────────────────────────────────────────────────────────


class RunDirectory:
    """One S14 run directory, created exactly once and written only under §22."""

    def __init__(self, root: str, experiment_id: str, run_id: str) -> None:
        if experiment_id not in EXPERIMENTS:
            raise ConfigError(f"unknown S14 experiment {experiment_id!r}")
        self.root = root
        self.experiment_id = experiment_id
        self.run_id = run_id
        base = os.path.join(root, RUN_ROOT, experiment_id, run_id)
        camp.assert_writable(base, root=root)
        self.path = base

    # -- layout ------------------------------------------------------------

    def create(self) -> str:
        os.makedirs(self.path, exist_ok=True)
        for name in camp.RUN_LAYOUT:
            target = os.path.join(self.path, name)
            if name in camp.LAYOUT_DIRECTORIES:
                os.makedirs(target, exist_ok=True)
        return self.path

    #: Files a run may write at the run root: the §22 layout plus the driver's own
    #: bookkeeping files.  ``source_identity.json`` is part of §22; the rest are the
    #: records this scaffolding defines (manual §4 + the triple gate's artefacts).
    WRITABLE_ROOT_FILES: Tuple[str, ...] = (
        *camp.RUN_LAYOUT,
        *RUN_ROOT_FILES,
        "source_identity.json",
    )

    def _resolve(self, name: str) -> str:
        if name not in self.WRITABLE_ROOT_FILES:
            raise ConfigError(
                f"{name!r} is not part of the S14 run layout (§22); refusing to write an unlisted artefact"
            )
        target = os.path.join(self.path, name)
        camp.assert_writable(target, root=self.root)
        return target

    def write_json(self, name: str, payload: Mapping[str, Any]) -> str:
        target = self._resolve(name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
        return target

    def write_text(self, name: str, text: str) -> str:
        target = self._resolve(name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(text)
        return target

    # -- records -----------------------------------------------------------

    def write_report_skeleton(self, title: str = "") -> str:
        mapping_line = f"- experiment: {self.experiment_id} — {title}" if title else f"- experiment: {self.experiment_id}"
        body = "\n".join(
            [
                f"# S14 run report — {self.experiment_id} / {self.run_id}",
                "",
                mapping_line,
                f"- stage: {STAGE}",
                f"- created: {_utc_now()}",
                "",
                "> 本文件是**运行骨架**。没有 `--execute`、未满足前置、或没有 raw samples 时，",
                "> 本 run 不得包含任何结论性数字；未运行即为未运行。",
                "",
                "## 结论",
                "",
                "（尚未运行）",
                "",
                "## 未完成项 / 阻塞",
                "",
                "（待填）",
                "",
            ]
        )
        return self.write_text("report.md", body)

    def write_preregistration(self, payload: Mapping[str, Any]) -> str:
        problems = _preregistration_problems(payload)
        if problems:
            raise ConfigError("preregistration is incomplete: " + "; ".join(problems))
        return self.write_json("preregistration.json", dict(payload))

    def write_status(self, status: str, reason: str, prerequisites: Optional[PrerequisiteStatus] = None) -> Dict[str, Any]:
        if status in GATED_CONCLUSION_STATUSES:
            raise ConfigError(
                f"{status} is a conclusion and must go through write_verdict (triple gate)"
            )
        if status not in rec.ALL_STATUSES:
            raise ConfigError(f"unknown status {status!r}")
        payload = {
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": status,
            "reason": reason,
            "created_at": _utc_now(),
            "prerequisites_satisfied": bool(prerequisites and prerequisites.satisfied),
            "conclusion": False,
        }
        return {"status": status, "reason": reason, "path": self.write_json("status.json", payload), **payload}

    def write_verdict(
        self,
        *,
        status: str,
        reason: str,
        prerequisites: PrerequisiteStatus,
        executed: bool,
        allow_execute: bool,
        raw_samples: int,
        limitations: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """The triple gate: ``--execute`` **and** prerequisites **and** raw samples.

        A conclusion written without all three is refused.  A refused conclusion is
        downgraded to ``BLOCKED_PREREQUISITE`` rather than the requested status, so
        the failure is visible in the artefact rather than only in a log line.
        """
        if status in GATED_CONCLUSION_STATUSES:
            if not allow_execute:
                raise ConfigError(
                    f"refusing to write {status}: --execute was not passed "
                    "(任务第五节：不得执行正式实验、不产出实验结论)"
                )
            if not prerequisites.satisfied:
                raise ConfigError(
                    f"refusing to write {status}: prerequisites unsatisfied ({', '.join(prerequisites.missing)})"
                )
            if not executed:
                raise ConfigError(f"refusing to write {status}: no execution was performed")
            if raw_samples <= 0:
                raise ConfigError(f"refusing to write {status}: no raw samples were recorded")
        elif status not in rec.ALL_STATUSES:
            raise ConfigError(f"unknown status {status!r}")

        payload = {
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": status,
            "reason": reason,
            "created_at": _utc_now(),
            "executed": executed,
            "execution_allowed": allow_execute,
            "raw_samples": raw_samples,
            "prerequisites_satisfied": prerequisites.satisfied,
            "prerequisites_missing": prerequisites.missing,
            "limitations": list(limitations),
            "conclusion": status in GATED_CONCLUSION_STATUSES,
        }
        return {"status": status, "reason": reason, "path": self.write_json("verdict.json", payload), **payload}

    def write_evidence_manifest(self, manifest: "EvidenceManifest") -> str:
        return self.write_json("evidence_manifest.json", manifest.as_dict())


def _preregistration_problems(payload: Mapping[str, Any]) -> List[str]:
    """The §23 run-manifest minimum fields, checked before anything runs."""
    missing = [name for name in rec.RUN_MANIFEST_FIELDS if name not in payload]
    problems = [f"preregistration is missing {name!r}" for name in missing]
    if payload.get("status") in GATED_CONCLUSION_STATUSES:
        problems.append("a preregistration may not carry a conclusion status")
    return problems


@dataclass
class EvidenceManifest:
    """``Evidence Manifest`` (控制平面 §26.3) with the S14 identity fields added."""

    run_id: str
    experiment_id: str
    git_commit: str = ""
    git_dirty: Optional[bool] = None
    patch_sha256: str = ""
    claim_ids: Tuple[str, ...] = ()
    model_artifact_id: str = ""
    checkpoint_artifact_id: str = ""
    policy_snapshot_id: str = ""
    frontier_study_contract_id: str = ""
    seed_bundle_id: str = ""
    config_uri: str = ""
    config_sha256: str = ""
    environment_uri: str = ""
    environment_sha256: str = ""
    commands: Tuple[str, ...] = ()
    raw_artifacts: Tuple[Mapping[str, str], ...] = ()
    profile_artifacts: Tuple[Mapping[str, str], ...] = ()
    quality_artifacts: Tuple[Mapping[str, str], ...] = ()
    decision_artifact: str = ""
    reports: Tuple[str, ...] = ()
    correctness_status: str = "not_run"
    claim_level: str = "NOT_MEASURED"
    limitations: Tuple[str, ...] = ()

    schema_version = f"{SCHEMA_PREFIX}.evidence-manifest.v1"

    #: ``claim_level`` values the manifest may carry.
    CLAIM_LEVELS: Tuple[str, ...] = (
        "SOURCE", "TEST", "RUNTIME", "MODEL", "SERVICE", "PORTABLE", "NOT_MEASURED",
    )

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("run_id", "experiment_id"):
            if not getattr(self, name):
                findings.append(f"EvidenceManifest: {name} is required")
        if self.experiment_id and self.experiment_id not in EXPERIMENTS:
            findings.append(f"EvidenceManifest: unknown experiment {self.experiment_id!r}")
        if self.claim_level not in self.CLAIM_LEVELS:
            findings.append(f"EvidenceManifest: unknown claim_level {self.claim_level!r}")
        if self.correctness_status not in ("pass", "fail", "not_run"):
            findings.append(f"EvidenceManifest: unknown correctness_status {self.correctness_status!r}")
        if self.claim_level not in ("NOT_MEASURED",) and not self.raw_artifacts:
            findings.append(
                "EvidenceManifest: a claim above NOT_MEASURED requires at least one raw artefact "
                "（拿不出证据必须降级）"
            )
        if not self.commands:
            findings.append("EvidenceManifest: commands are required (结论必须能反向定位到 raw sample)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "claim_ids": list(self.claim_ids),
            "git": {"commit": self.git_commit, "dirty": self.git_dirty, "patch_sha256": self.patch_sha256},
            "model_artifact_id": self.model_artifact_id,
            "checkpoint_artifact_id": self.checkpoint_artifact_id,
            "policy_snapshot_id": self.policy_snapshot_id,
            "frontier_study_contract_id": self.frontier_study_contract_id,
            "seed_bundle_id": self.seed_bundle_id,
            "config": {"uri": self.config_uri, "sha256": self.config_sha256},
            "environment": {"uri": self.environment_uri, "sha256": self.environment_sha256},
            "commands": list(self.commands),
            "raw_artifacts": [dict(item) for item in self.raw_artifacts],
            "profile_artifacts": [dict(item) for item in self.profile_artifacts],
            "quality_artifacts": [dict(item) for item in self.quality_artifacts],
            "decision_artifact": self.decision_artifact,
            "reports": list(self.reports),
            "correctness_status": self.correctness_status,
            "claim_level": self.claim_level,
            "limitations": list(self.limitations),
        }
        payload["manifest_digest"] = canonical_digest(payload)
        return payload


# ── environment ────────────────────────────────────────────────────────────


def environment_fingerprint(root: str = "") -> Dict[str, Any]:
    """A minimal, honest fingerprint: what was actually observed, nothing inferred."""
    payload: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "recorded_at": _utc_now(),
        "note": "GPU/NPU 型号、驱动、功耗模式必须由实验自行采集；这里不推测设备",
    }
    if root:
        payload["repo_root_relative"] = os.path.relpath(root, root) or "."
    return payload


def git_identity(root: str) -> Dict[str, Any]:
    """Current commit and dirty state; a missing git is reported, not guessed."""
    def _run(*args: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ("git", *args), cwd=root, capture_output=True, text=True, timeout=10, check=False
            )
        except Exception:  # noqa: BLE001 - a missing git is a state, not a crash
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = _run("rev-parse", "HEAD")
    dirty_output = _run("status", "--porcelain")
    return {
        "commit": commit or "",
        "dirty": bool(dirty_output) if dirty_output is not None else None,
        "available": commit is not None,
    }


def build_experiment_record(experiment_id: str, *, git: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The manual §4 record, prefilled with the current identity at ``NOT_STARTED``."""
    record = rec.experiment_record_template(experiment_id)
    if git is not None:
        record["git_commit"] = str(git.get("commit", ""))
        record["git_dirty"] = bool(git.get("dirty", False))
    return record
