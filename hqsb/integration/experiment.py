"""S06 experiment scaffolding: prerequisites, run layout, evidence, verdicts.

Like its S05 counterpart (``hqsb.quant.experiment``), the scaffolding has one
central property: **it cannot produce a conclusion by default**.

* :func:`check_prerequisites` inspects the repository for the *evidence* the S06
  protocol demands (S04.5 M4, S05 P0, frozen six-workload baseline, recorded
  environment fingerprint, C1–C7 availability) and reports every missing item
  with the path it looked for;
* :meth:`RunDirectory.write_verdict` refuses to write PASS/FAIL/PASS_NEGATIVE
  unless the caller explicitly enables execution, the prerequisites are
  satisfied **and** raw samples exist;
* the run layout mirrors the S06 data layout (``graph/before``, ``graph/after``,
  ``graph/guards``, ``graph/breaks``, ``compile/phases``, ``compile/generated``,
  ``compile/cache``, ``dispatch``, ``correctness``, ``performance``, ``memory``,
  ``profiler``, ``traces``, ``errors``, ``operator``, ``environment``);
* :meth:`RunDirectory.write_report_skeleton` writes the experiment report stub
  with ``NOT_RUN`` placeholders and **no numbers**, so an unexecuted report can
  never be mistaken for a result.
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

CONCLUSION_STATUSES = (STATUS_PASS, STATUS_PASS_NEGATIVE, STATUS_FAIL)

EXPERIMENTS = tuple(f"E06-{index:02d}" for index in range(1, 12))

STAGE = "S06"

#: Run layout from the S06 details README §16.
RUN_LAYOUT: Tuple[str, ...] = (
    "commands",
    "stdout",
    "stderr",
    "environment",
    "operator",
    "graph/before",
    "graph/after",
    "graph/guards",
    "graph/breaks",
    "compile/phases",
    "compile/generated",
    "compile/cache",
    "dispatch",
    "correctness",
    "performance",
    "memory",
    "profiler",
    "traces",
    "errors",
)

#: The marker an actual S04.5 execution must leave behind.  S06's own
#: scaffolding lives in the same package, so a bare directory can never be
#: accepted as S04.5 evidence.
S04_5_EVIDENCE_MARKER = os.path.join("hqsb", "integration", "s04_5_evidence.json")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    """Aggregate prerequisite status for S06."""

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
    """Inspect the repository for the S06 evidence chain (never for intent).

    Every check names the path it looked for, so the reason is verifiable by
    hand.  S06's own interface scaffolding is explicitly **not** accepted as
    upstream evidence.
    """
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
                    "S06 interface scaffolding in hqsb/integration is NOT S04.5 M4 "
                    "evidence and must not be read as such"
                )
            ),
        )
    )

    raw_root = os.path.join(root, "docs", "stage_experiments", "S04.5")
    checks.append(
        PrerequisiteCheck(
            name="s04_5_experiment_evidence",
            satisfied=os.path.isdir(raw_root),
            evidence=raw_root if os.path.isdir(raw_root) else "",
            reason="" if os.path.isdir(raw_root) else f"no S04.5 experiment evidence under {raw_root}",
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
            reason="" if acceptance else "no S04.5 acceptance report; the M4 claim is unverified",
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
                else "no frozen six-workload FP16 baseline; without it a compile-path "
                "difference cannot be separated from a model-level difference"
            ),
        )
    )

    s05_verdicts = _find_verdicts(root, "S05")
    s05_passing = [
        item
        for item in s05_verdicts
        if item.get("status") in (STATUS_PASS, STATUS_PASS_NEGATIVE)
    ]
    checks.append(
        PrerequisiteCheck(
            name="s05_p0_evidence",
            satisfied=bool(s05_passing),
            evidence=s05_passing[0]["_path"] if s05_passing else "",
            reason=(
                ""
                if s05_passing
                else "no S05 verdict.json with PASS/PASS_NEGATIVE; S05 P0 is not complete, "
                "so the quant route cannot be consumed"
            ),
        )
    )

    fingerprints = glob.glob(
        os.path.join(root, "docs", "stage_experiments", "*", "**", "environment_fingerprint.json"),
        recursive=True,
    ) + glob.glob(
        os.path.join(root, "experiment_results", "S04.5", "**", "environment_fingerprint.json"),
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
                else "no recorded environment fingerprint (Python/torch/CUDA/compiler/ABI/arch); "
                "a run without a frozen toolchain identity cannot be reproduced"
            ),
        )
    )

    # C1–C7 availability is a code-level check and is satisfied in this tree; it
    # is kept here so a schema regression blocks execution instead of surfacing
    # at report time.
    schema_note = ""
    schema_ok = True
    try:
        from hqsb.integration import telemetry

        coverage = telemetry.c6_c7_summary()
        schema_ok = bool(coverage["c6"]["ok"] and coverage["c7"]["ok"])
        if not schema_ok:
            schema_note = "C6/C7 projection does not cover the required S06 fields"
    except Exception as exc:  # noqa: BLE001 - a broken contract must block, not crash
        schema_ok = False
        schema_note = f"{type(exc).__name__}: {exc}"
    checks.append(
        PrerequisiteCheck(
            name="c6_c7_schema_available",
            satisfied=schema_ok,
            evidence="hqsb/integration/telemetry.py" if schema_ok else "",
            reason=schema_note,
        )
    )

    return PrerequisiteStatus(stage=STAGE, checks=checks)


# ── preregistration ───────────────────────────────────────────────────────


@dataclass
class Preregistration:
    """The frozen S06 experiment plan (details README §15.1)."""

    experiment_id: str
    question: str
    hypothesis: str
    capture_mode: str = ""
    operator_schema_hash: str = ""
    pattern: str = ""
    dynamic_policy: str = ""
    shape_sequence: Tuple[str, ...] = ()
    cache_state: str = ""
    quality_tolerance: Mapping[str, Any] = field(default_factory=dict)
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
                    f"preregistration"
                )
        if not self.claim_boundary:
            raise ConfigError(
                f"preregistration for {self.experiment_id} needs an explicit "
                "claim_boundary (what may NOT be claimed from this experiment)",
                details={"field": "claim_boundary"},
            )

    def to_json(self) -> str:
        payload = {
            "schema_version": "1.0.0",
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "capture_mode": self.capture_mode,
            "operator_schema_hash": self.operator_schema_hash,
            "pattern": self.pattern,
            "dynamic_policy": self.dynamic_policy,
            "shape_sequence": list(self.shape_sequence),
            "cache_state": self.cache_state,
            "quality_tolerance": dict(self.quality_tolerance),
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


# ── evidence manifest ─────────────────────────────────────────────────────


@dataclass
class EvidenceManifest:
    """Run evidence manifest (control plane §26.3 + S06 extras)."""

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
    # S06 extras (details README §14)
    compile_mode: str = ""
    graph_identity: str = ""
    compile_identity: str = ""
    graph_count: int = 0
    break_count: int = 0
    cache_layer: str = ""
    cache_hit: bool = False
    compile_phase_times: Mapping[str, float] = field(default_factory=dict)
    requested_lowering: str = ""
    actual_lowering: str = ""
    observed_kernel: str = ""
    fallback_reason: str = ""

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
            "graph_compile": {
                "compile_mode": self.compile_mode,
                "graph_identity": self.graph_identity,
                "compile_identity": self.compile_identity,
                "graph_count": self.graph_count,
                "break_count": self.break_count,
                "cache_layer": self.cache_layer,
                "cache_hit": self.cache_hit,
                "compile_phase_times": dict(self.compile_phase_times),
                "requested_lowering": self.requested_lowering,
                "actual_lowering": self.actual_lowering,
                "observed_kernel": self.observed_kernel,
                "fallback_reason": self.fallback_reason,
            },
        }
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


# ── run directory ─────────────────────────────────────────────────────────


class RunDirectory:
    """Creates and manages one ``experiment_results/S06/<E>/<run_id>/`` tree."""

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

    def write_evidence_manifest(self, manifest: EvidenceManifest) -> str:
        return self.write_text("evidence_manifest.json", manifest.to_json())

    def raw_file_count(self) -> int:
        raw = os.path.join(self.path, "raw")
        if not os.path.isdir(raw):
            raw = os.path.join(self.path, "performance")
        if not os.path.isdir(raw):
            return 0
        return sum(len(files) for _root, _dirs, files in os.walk(raw))

    def write_report_skeleton(self, experiment_id: str, title: str) -> str:
        """Write a report stub that cannot be mistaken for a result."""
        skeleton = (
            f"# {experiment_id}: {title}\n\n"
            f"> 状态：**NOT_RUN**（本文件由 `hqsb.integration.experiment` 生成的骨架；\n"
            f"> 未执行任何实验，未产出任何数值）。\n\n"
            "## 1. 预注册\n\n"
            "见 `preregistration.json`。\n\n"
            "## 2. 环境与身份\n\n"
            "见 `environment_fingerprint.json`、`evidence_manifest.json`；未执行时为占位。\n\n"
            "## 3. 原始数据\n\n"
            "`raw/`、`performance/`、`profiler/`、`traces/` 均为空，直到实验真正运行。\n\n"
            "## 4. 结论\n\n"
            "无。实验未执行，禁止填写任何设备/性能/正确性结论。\n"
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
        """Write a verdict — or refuse when the run cannot support one.

        Refusal conditions (all deliberate):

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


# ── environment / git ─────────────────────────────────────────────────────


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
    "CONCLUSION_STATUSES",
    "EXPERIMENTS",
    "EvidenceManifest",
    "PROTOCOL_STATUSES",
    "Preregistration",
    "PrerequisiteCheck",
    "PrerequisiteStatus",
    "RUN_LAYOUT",
    "RunDirectory",
    "S04_5_EVIDENCE_MARKER",
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
