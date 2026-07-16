"""S15 experiment scaffolding: prerequisites, §22 package, evidence, verdicts.

All eleven S15 experiments share one structure — "preregister → collect → write →
decide" — with a **quadruple gate** that makes a conclusion impossible by default:

* :func:`check_prerequisites` reads the repository for *evidence* (S00–S14
  acceptance records, a frozen release candidate, a claim ledger) and probes for
  the external capabilities an S15 run needs (a clean environment builder, a
  documentation builder, packaging/scanner tools, an accelerator, network
  access, a reviewer, participants).  A missing capability becomes the precise
  S15 state of ``details/S15/README.md`` §13.5 (``BLOCKED``, ``INVALID``) — never
  a silent pass and never a fabricated value;
* :class:`RunDirectory` writes only under ``artifacts/S15/<experiment>/<run>``
  and creates the §22 uniform data package; the protocol tree
  ``docs/stage_experiments/**`` is never written to (``campaign.assert_writable``
  enforces it);
* :meth:`RunDirectory.write_verdict` refuses ``PASS``/``FAIL`` without
  ``--execute``, satisfied prerequisites **and** raw samples;
* additionally (S15-specific) a conclusion must be bound to a *release
  candidate* and a *claim ledger digest*: a public claim that cannot name which
  frozen candidate it belongs to is exactly the orphan the stage exists to catch.

Nothing here runs an experiment or publishes anything.
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

from hqsb.release import campaign as camp
from hqsb.release import records as rec
from hqsb.release.identity import SCHEMA_PREFIX, canonical_digest, is_digest

#: Conclusion statuses the gate must refuse without the four conditions.
GATED_CONCLUSION_STATUSES: Tuple[str, ...] = rec.CONCLUSION_STATUSES

STAGE = "S15"

#: Run output lives under ``artifacts/S15`` (§22 + S14 precedent).
RUN_ROOT = camp.RUN_ROOT

EXPERIMENTS: Tuple[str, ...] = rec.EXPERIMENT_IDS

#: Extra root files the driver writes besides the §22 package.
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
    "source_identity.json",
)

#: A stage counts as frozen when one of these exists (report or acceptance record).
UPSTREAM_ACCEPTANCE_GLOBS: Mapping[str, Tuple[str, ...]] = {
    stage: (
        f"docs/reports/{stage}_阶段验收报告.md",
        f"docs/reports/{stage}_开发报告.md",
        f"docs/stage_experiments/{stage}/*/acceptance.json",
        f"experiment_results/{stage}/*/campaign_manifest.yaml",
    )
    for stage in rec.UPSTREAM_STAGES
}

#: The minimum number of upstream stages that must carry written acceptance.
MIN_UPSTREAM_STAGES = 8

#: External components an S15 run needs, and the state a missing one produces.
EXTERNAL_COMPONENTS: Mapping[str, Tuple[str, ...]] = {
    "clean_environment_builder": ("docker", "podman", "python3 -m venv"),
    "documentation_builder": ("mkdocs", "sphinx-build", "mdbook"),
    "packaging_tooling": ("python3 -m build", "twine"),
    "scanner_tooling": ("gitleaks", "trivy", "pip-audit", "syft"),
    "recording_tooling": ("ffmpeg", "asciinema"),
    "accelerator_device": ("nvidia-smi", "npu-smi", "rocm-smi"),
    "network_access": ("curl", "wget"),
    "browser_tooling": ("chromium", "google-chrome", "playwright"),
}

#: Which prerequisite state a missing component produces.
COMPONENT_STATE: Mapping[str, str] = {
    "clean_environment_builder": rec.STATUS_BLOCKED,
    "documentation_builder": rec.STATUS_BLOCKED,
    "packaging_tooling": rec.STATUS_BLOCKED,
    "scanner_tooling": rec.STATUS_BLOCKED,
    "recording_tooling": rec.STATUS_BLOCKED,
    "accelerator_device": rec.STATUS_BLOCKED,
    "network_access": rec.STATUS_BLOCKED,
    "browser_tooling": rec.STATUS_BLOCKED,
}

#: Prerequisites that need a decision or an artefact rather than a binary.
DECISION_COMPONENTS: Mapping[str, str] = {
    "release_candidate": "a frozen ReleaseCandidateSnapshot (details/S15/README.md §7.1)",
    "claim_ledger": "a verified claim ledger to render from (E15-01)",
    "public_artifact": "a built wheel/sdist or container to consume (E15-02/E15-05)",
    "evidence_bundle": "a public evidence bundle with per-file rights (E15-05/E15-06)",
    "upstream_authorization": "explicit human authorisation for an upstream submission (E15-10 step 25)",
    "reviewer_resource": "an independent reviewer who never worked on HQSB (E15-09 step 2)",
    "participant_resource": "2–3 target-role participants without prior exposure (E15-11 step 2–4)",
    "release_authorization": "explicit human GO/NO-GO for publishing (E15-05 step 41)",
}

#: Decision components whose absence only blocks the experiments that need them.
OPTIONAL_DECISION_COMPONENTS: Tuple[str, ...] = (
    "reviewer_resource",
    "participant_resource",
)


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
    """Aggregate prerequisite status for S15 (states from §13.5)."""

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

    def for_experiment(self, experiment_id: str) -> "PrerequisiteStatus":
        """The subset of checks that actually gates one experiment.

        The full stage gate is ``G0`` (upstream freeze) plus the capabilities the
        experiment itself declares; the reviewer/participant resources gate only
        E15-09/E15-11, so a missing reviewer does not mark E15-01 ``BLOCKED`` for
        the wrong reason.
        """
        if experiment_id not in rec.EXPERIMENT_IDS:
            raise ConfigError(f"unknown S15 experiment {experiment_id!r}")
        needs_reviewer = experiment_id in ("E15-09", "E15-10")
        needs_participant = experiment_id == "E15-11"
        subset: List[PrerequisiteCheck] = []
        for check in self.checks:
            if check.name == "upstream_freeze":
                subset.append(check)
                continue
            if check.name == "reviewer_resource" and not needs_reviewer:
                continue
            if check.name == "participant_resource" and not needs_participant:
                continue
            subset.append(check)
        return PrerequisiteStatus(stage=self.stage, checks=subset)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "satisfied": self.satisfied,
            "missing": self.missing,
            "advisory": self.advisory,
            "states": dict(sorted(self.states().items())),
            "checks": [check.as_dict() for check in self.checks],
        }


def _which(name: str) -> str:
    """Resolve a binary name; a ``python -m x`` / dotted module is probed with find_spec."""
    if name.startswith("python3 -m "):
        module = name.split(" ", 2)[-1]
        try:
            import importlib.util

            return name if importlib.util.find_spec(module) is not None else ""
        except (ImportError, ValueError):
            return ""
    if "." in name and "/" not in name:
        try:
            import importlib.util

            return name if importlib.util.find_spec(name) is not None else ""
        except (ImportError, ValueError):
            return ""
    return shutil.which(name) or ""


def _upstream_acceptance(root: str) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {}
    for stage, patterns in sorted(UPSTREAM_ACCEPTANCE_GLOBS.items()):
        hits: List[str] = []
        for pattern in patterns:
            hits.extend(sorted(glob.glob(os.path.join(root, pattern))))
        found[stage] = hits
    return found


def check_prerequisites(root: str, *, probe: bool = False) -> PrerequisiteStatus:
    """Read the repository (and optionally the machine) for the S15 gates.

    ``probe=False`` never touches the machine: it reports the external components
    as *unknown* rather than missing, so a CI run does not silently claim a
    capability it did not test.  ``probe=True`` resolves the binaries/modules and
    turns a missing one into the blocking state.
    """
    checks: List[PrerequisiteCheck] = []

    upstream = _upstream_acceptance(root)
    stages_with_evidence = sorted(stage for stage, hits in upstream.items() if hits)
    checks.append(
        PrerequisiteCheck(
            name="upstream_freeze",
            satisfied=len(stages_with_evidence) >= MIN_UPSTREAM_STAGES,
            state="" if len(stages_with_evidence) >= MIN_UPSTREAM_STAGES else rec.STATUS_BLOCKED,
            evidence=", ".join(stages_with_evidence) if stages_with_evidence else "(none found)",
            reason=(
                ""
                if len(stages_with_evidence) >= MIN_UPSTREAM_STAGES
                else (
                    f"only {len(stages_with_evidence)} of {len(rec.UPSTREAM_STAGES)} upstream stages carry written "
                    f"acceptance/development records; S15 的冻结输入（G0）尚未建立"
                )
            ),
            detail={"stages": {stage: len(hits) for stage, hits in sorted(upstream.items())}},
        )
    )

    # G0 additionally wants the *S15* frozen objects: candidate + ledger.
    candidate_hits = sorted(glob.glob(os.path.join(root, "artifacts/S15/*/*/release_candidate_ref.json")))
    checks.append(
        PrerequisiteCheck(
            name="release_candidate",
            satisfied=bool(candidate_hits),
            state="" if candidate_hits else rec.STATUS_BLOCKED,
            evidence=candidate_hits[0] if candidate_hits else "(none found)",
            reason="" if candidate_hits else "no frozen ReleaseCandidateSnapshot exists yet (§7.1)",
        )
    )
    ledger_hits = sorted(glob.glob(os.path.join(root, "artifacts/S15/E15-01/*/claim_ledger.yaml")))
    checks.append(
        PrerequisiteCheck(
            name="claim_ledger",
            satisfied=bool(ledger_hits),
            state="" if ledger_hits else rec.STATUS_BLOCKED,
            evidence=ledger_hits[0] if ledger_hits else "(none found)",
            reason="" if ledger_hits else "no verified claim ledger exists yet (E15-01)",
        )
    )

    artifact_hits: List[str] = []
    for pattern in ("artifacts/S15/E15-05/*/artifacts/*", "dist/*.whl", "dist/*.tar.gz"):
        artifact_hits.extend(sorted(glob.glob(os.path.join(root, pattern))))
    checks.append(
        PrerequisiteCheck(
            name="public_artifact",
            satisfied=bool(artifact_hits),
            state="" if artifact_hits else rec.STATUS_BLOCKED,
            evidence=artifact_hits[0] if artifact_hits else "(none found)",
            reason="" if artifact_hits else "no built wheel/sdist/container exists to consume (E15-02 step 12)",
        )
    )

    bundle_hits = sorted(glob.glob(os.path.join(root, "artifacts/S15/E15-05/*/artifact_manifest.json")))
    checks.append(
        PrerequisiteCheck(
            name="evidence_bundle",
            satisfied=bool(bundle_hits),
            state="" if bundle_hits else rec.STATUS_BLOCKED,
            evidence=bundle_hits[0] if bundle_hits else "(none found)",
            reason="" if bundle_hits else "no public evidence bundle with per-file rights exists (§7.2)",
            required=False,
        )
    )

    for name, description in sorted(DECISION_COMPONENTS.items()):
        if name in ("release_candidate", "claim_ledger", "public_artifact", "evidence_bundle"):
            continue
        checks.append(
            PrerequisiteCheck(
                name=name,
                satisfied=False,
                state=rec.STATUS_BLOCKED,
                evidence="",
                reason=f"{description} must be confirmed before a run (no automatic default)",
                required=name not in OPTIONAL_DECISION_COMPONENTS,
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
                    required=component in ("clean_environment_builder", "documentation_builder"),
                    detail={"candidates": list(candidates)},
                )
            )
        else:
            checks.append(
                PrerequisiteCheck(
                    name=component,
                    satisfied=False,
                    state=rec.STATUS_BLOCKED,
                    evidence="",
                    reason="not probed (run with --probe to resolve binaries/modules)",
                    required=False,
                    detail={"candidates": list(candidates)},
                )
            )

    return PrerequisiteStatus(stage=STAGE, checks=checks)


# ── run directory ──────────────────────────────────────────────────────────


class RunDirectory:
    """One S15 run directory: the §22 package, written exactly once."""

    def __init__(self, root: str, experiment_id: str, run_id: str) -> None:
        if experiment_id not in EXPERIMENTS:
            raise ConfigError(f"unknown S15 experiment {experiment_id!r}")
        self.root = root
        self.experiment_id = experiment_id
        self.run_id = run_id
        layout = camp.run_layout(experiment_id, run_id, root=root)
        base = layout["_base"]
        camp.assert_writable(base, root=root)
        self.path = base
        self._layout = layout

    # -- layout ------------------------------------------------------------

    def create(self) -> str:
        os.makedirs(self.path, exist_ok=True)
        for name in rec.RUN_LAYOUT:
            target = os.path.join(self.path, name)
            if name in camp.LAYOUT_DIRECTORIES:
                os.makedirs(target, exist_ok=True)
        for name in rec.EXPERIMENT_LAYOUT_EXTRA.get(self.experiment_id, ()):
            os.makedirs(os.path.join(self.path, name), exist_ok=True)
        return self.path

    @property
    def writable_root_files(self) -> Tuple[str, ...]:
        return tuple(rec.RUN_LAYOUT) + tuple(RUN_ROOT_FILES)

    def _resolve(self, name: str) -> str:
        if name not in self.writable_root_files:
            raise ConfigError(
                f"{name!r} is not part of the S15 run layout (§22); refusing to write an unlisted artefact"
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
                f"# S15 run report — {self.experiment_id} / {self.run_id}",
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

    def write_findings(self, payload: Mapping[str, Any]) -> str:
        return self.write_json("findings.json", dict(payload))

    def write_limitations(self, text: str) -> str:
        return self.write_text("limitations.md", text)

    def write_acceptance(self, gate_results: Mapping[str, str], *, notes: str = "") -> Dict[str, Any]:
        """``acceptance.json`` — per-gate verdicts, never a single overall PASS (§22)."""
        if not gate_results:
            raise ConfigError("acceptance.json must record the gates one by one, not a single overall verdict")
        invalid = {gate: verdict for gate, verdict in gate_results.items() if verdict not in rec.ALL_STATUSES}
        if invalid:
            raise ConfigError(f"acceptance.json carries non-status verdicts: {invalid}")
        payload = {
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "gates": {gate: gate_results[gate] for gate in sorted(gate_results)},
            "notes": notes,
            "created_at": _utc_now(),
        }
        payload["acceptance_digest"] = canonical_digest(payload)
        return {"path": self.write_json("acceptance.json", payload), **payload}

    def write_status(self, status: str, reason: str, prerequisites: Optional[PrerequisiteStatus] = None) -> Dict[str, Any]:
        if status in GATED_CONCLUSION_STATUSES:
            raise ConfigError(f"{status} is a conclusion and must go through write_verdict (quadruple gate)")
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
        candidate_id: str = "",
        claim_ledger_sha256: str = "",
        limitations: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """The quadruple gate: ``--execute`` **and** prerequisites **and** raw
        samples **and** (for a conclusion) a release candidate + claim ledger.

        A conclusion written without all four is refused.  A refused conclusion is
        not silently downgraded either: the caller gets the error, and the artefact
        on disk keeps the non-conclusion status it was given.
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
            if not candidate_id:
                raise ConfigError(
                    f"refusing to write {status}: no release candidate id — a public claim must name the "
                    "frozen candidate it belongs to (§7.1)"
                )
            if not is_digest(claim_ledger_sha256):
                raise ConfigError(
                    f"refusing to write {status}: the claim ledger digest is missing/invalid — rendering "
                    "must read a frozen ledger (§8.1)"
                )
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
            "release_candidate_id": candidate_id,
            "claim_ledger_sha256": claim_ledger_sha256,
            "limitations": list(limitations),
            "conclusion": status in GATED_CONCLUSION_STATUSES,
        }
        return {"status": status, "reason": reason, "path": self.write_json("verdict.json", payload), **payload}

    def write_evidence_manifest(self, manifest: "EvidenceManifest") -> str:
        return self.write_json("evidence_manifest.yaml", manifest.as_dict())


def _preregistration_problems(payload: Mapping[str, Any]) -> List[str]:
    """The manual §4 fields, checked before anything runs.

    S15 adds the two identity fields every later artefact references:
    ``release_candidate_id`` and ``claim_ids``.
    """
    missing = [name for name in rec.RUN_MANIFEST_FIELDS if name not in payload]
    problems = [f"preregistration is missing {name!r}" for name in missing]
    if payload.get("status") in GATED_CONCLUSION_STATUSES:
        problems.append("a preregistration may not carry a conclusion status")
    if not payload.get("release_candidate_id"):
        problems.append("preregistration is missing 'release_candidate_id' (S15 identity)")
    return problems


@dataclass
class EvidenceManifest:
    """``Evidence Manifest`` (control-plane §26.3) with the S15 identity fields."""

    run_id: str
    experiment_id: str
    git_commit: str = ""
    git_dirty: Optional[bool] = None
    patch_sha256: str = ""
    claim_ids: Tuple[str, ...] = ()
    release_candidate_id: str = ""
    claim_ledger_sha256: str = ""
    evidence_graph_root: str = ""
    public_channels: Tuple[str, ...] = ()
    reviewer_ids: Tuple[str, ...] = ()
    config_uri: str = ""
    config_sha256: str = ""
    environment_uri: str = ""
    environment_sha256: str = ""
    commands: Tuple[str, ...] = ()
    raw_artifacts: Tuple[Mapping[str, str], ...] = ()
    normalized_artifacts: Tuple[Mapping[str, str], ...] = ()
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
        if self.claim_level != "NOT_MEASURED" and not self.raw_artifacts:
            findings.append(
                "EvidenceManifest: a claim above NOT_MEASURED requires at least one raw artefact "
                "（拿不出证据必须降级）"
            )
        if not self.commands:
            findings.append("EvidenceManifest: commands are required (结论必须能反向定位到 raw sample)")
        if self.claim_level != "NOT_MEASURED" and not self.release_candidate_id:
            findings.append("EvidenceManifest: a measured claim must name its release candidate")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "claim_ids": list(self.claim_ids),
            "git": {"commit": self.git_commit, "dirty": self.git_dirty, "patch_sha256": self.patch_sha256},
            "release_candidate_id": self.release_candidate_id,
            "claim_ledger_sha256": self.claim_ledger_sha256,
            "evidence_graph_root": self.evidence_graph_root,
            "public_channels": list(self.public_channels),
            "reviewer_ids": list(self.reviewer_ids),
            "config": {"uri": self.config_uri, "sha256": self.config_sha256},
            "environment": {"uri": self.environment_uri, "sha256": self.environment_sha256},
            "commands": list(self.commands),
            "raw_artifacts": [dict(item) for item in self.raw_artifacts],
            "normalized_artifacts": [dict(item) for item in self.normalized_artifacts],
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
        "note": "GPU/NPU 型号、驱动与本机安装的扫描器必须由实验自行采集；这里不推测设备或工具链",
    }
    if root:
        payload["repo_relative_root"] = "."
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


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the scaffolding (labelled smoke, not an experiment)."""
    problems: List[str] = []
    run = RunDirectory("/tmp/hqsb-s15-smoke", "E15-01", "interface_only")
    created = run.create()
    for entry in rec.RUN_LAYOUT:
        if entry in camp.LAYOUT_DIRECTORIES:
            if not os.path.isdir(os.path.join(created, entry)):
                problems.append(f"§22 directory {entry!r} was not created")
        elif entry not in run.writable_root_files:
            problems.append(f"§22 file {entry!r} is not writable through the run directory")
    for extra in rec.EXPERIMENT_LAYOUT_EXTRA.get("E15-01", ()):
        if not os.path.isdir(os.path.join(created, extra)):
            problems.append(f"experiment-specific directory {extra!r} was not created")
    status = run.write_status(rec.STATUS_BLOCKED, "prerequisites unsatisfied", None)
    blocked_ok = status["status"] == rec.STATUS_BLOCKED and status["conclusion"] is False
    if not blocked_ok:
        problems.append("a BLOCKED status was not written as a non-conclusion")
    refused = 0
    try:
        run.write_verdict(
            status=rec.STATUS_PASS,
            reason="smoke attempt",
            prerequisites=PrerequisiteStatus(stage=STAGE, checks=[]),
            executed=False,
            allow_execute=False,
            raw_samples=0,
        )
    except ConfigError:
        refused += 1
    try:
        run.write_json("unlisted_artefact.json", {})
    except ConfigError:
        refused += 1
    if refused != 2:
        problems.append(f"expected two refusals (conclusion, unlisted file), got {refused}")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "run_dir": created,
        "layout_entries": len(rec.RUN_LAYOUT),
        "refusals": refused,
        "problems": problems,
    }
