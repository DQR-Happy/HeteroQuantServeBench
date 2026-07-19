"""S13 experiment scaffolding: prerequisites, preregistration, evidence, verdicts.

Mirrors the S12 scaffolding so the eleven S13 experiments share one structure:
"preregister → collect → write → decide" with a **triple gate** that makes a
conclusion impossible by default:

* :func:`check_prerequisites` reads the repository for *evidence* (S08 serving,
  S12 capacity/campaign data, upstream PASS verdicts) and probes for the external
  capabilities an S13 run needs (cluster access, container engine, scanners,
  telemetry backend).  A missing capability becomes the precise S13 state of
  ``details/S13/README.md`` §4 (``NOT_RUN_CLUSTER_UNAVAILABLE``,
  ``SCANNER_UNAVAILABLE``, ``NOT_RUN_PERMISSION_DENIED``, …) — never a silent pass;
* :class:`RunDirectory` writes only under ``experiment_results/S13/...``; the
  protocol tree ``docs/stage_experiments/**`` is never written to;
* :meth:`RunDirectory.write_verdict` refuses ``PASS``/``FAIL``/``PASS_NEGATIVE``
  without ``--execute``, satisfied prerequisites **and** raw samples;
* :class:`EvidenceManifest` carries the S13 identity fields (release/image/cluster/
  fault/canary/tenant) and only accepts an evidence level its payload supports.

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

from hqsb.core.experiment_io import RunStorage, run_directory_path
from hqsb.core.errors import ConfigError
from hqsb.infra import records as rec

#: Only these three are conclusion statuses the triple gate must refuse.
GATED_CONCLUSION_STATUSES: Tuple[str, ...] = (
    rec.STATUS_PASS,
    rec.STATUS_PASS_NEGATIVE,
    rec.STATUS_FAIL,
)

STAGE = "S13"

EXPERIMENTS: Tuple[str, ...] = tuple(f"E13-{index:02d}" for index in range(1, 12))

#: One run directory holds raw *and* derived material for one experiment run.
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

#: Upstream evidence globs (protocol tree first, then repository reports).
S08_VERDICT_GLOB = "docs/stage_experiments/S08/*/raw*/verdict.json"
S12_VERDICT_GLOB = "docs/stage_experiments/S12/*/raw*/verdict.json"
S12_CAMPAIGN_GLOB = "experiment_results/S12/*/campaign_manifest.json"
UPSTREAM_VERDICT_GLOBS: Mapping[str, str] = {
    "S00": "docs/stage_experiments/S00/*/raw*/verdict.json",
    "S01": "docs/stage_experiments/S01/*/raw*/verdict.json",
    "S02": "docs/stage_experiments/S02/*/raw*/verdict.json",
    "S03": "docs/stage_experiments/S03/*/raw*/verdict.json",
    "S04": "docs/stage_experiments/S04/*/raw*/verdict.json",
    "S05": "docs/stage_experiments/S05/*/raw*/verdict.json",
    "S06": "docs/stage_experiments/S06/*/raw*/verdict.json",
    "S07": "docs/stage_experiments/S07/*/raw*/verdict.json",
    "S08": "docs/stage_experiments/S08/*/raw*/verdict.json",
    "S09": "docs/stage_experiments/S09/*/raw*/verdict.json",
    "S10": "docs/stage_experiments/S10/*/raw*/verdict.json",
    "S11": "docs/stage_experiments/S11/*/raw*/verdict.json",
    "S12": "docs/stage_experiments/S12/*/raw*/verdict.json",
}
JETSON_EVIDENCE_GLOB = "reports/jetson/**/*.json"
S13_CAMPAIGN_GLOB = "experiment_results/S13/*/campaign_manifest.yaml"

#: Minimum upstream stages that must carry a PASS verdict before a *release* claim.
MIN_UPSTREAM_STAGES = 1

#: External components an S13 run needs; each maps to an S13 prerequisite state.
EXTERNAL_COMPONENTS: Mapping[str, Tuple[str, ...]] = {
    "container_tooling": ("docker", "podman", "buildah", "nerdctl", "buildctl"),
    "cluster_tooling": ("kubectl", "helm", "helmfile", "kustomize"),
    "registry_tooling": ("cosign", "crane", "skopeo"),
    "scan_tooling": ("syft", "grype", "trivy", "grype-db"),
    "telemetry_tooling": ("promtool", "otelcol", "prometheus"),
}

#: Which prerequisite state a missing component produces.
COMPONENT_STATE: Mapping[str, str] = {
    "container_tooling": rec.PREREQ_NOT_RUN_TOOL_UNAVAILABLE,
    "cluster_tooling": rec.PREREQ_NOT_RUN_CLUSTER_UNAVAILABLE,
    "registry_tooling": rec.MISSING_REGISTRY_UNAVAILABLE,
    "scan_tooling": rec.MISSING_SCANNER_UNAVAILABLE,
    "telemetry_tooling": rec.PREREQ_NOT_RUN_TOOL_UNAVAILABLE,
}

#: Prerequisites that need *permission* rather than a binary.
PERMISSION_COMPONENTS: Mapping[str, str] = {
    "cluster_admin_scope": "namespace-scoped RBAC for the test namespace",
    "registry_push_scope": "push rights to an isolated registry",
    "kubeconfig_context": "a dedicated test context (never the production context)",
    "fault_authorization": "written authorization for destructive faults in the test namespace",
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
    """Aggregate prerequisite status for S13 (states from ``details/S13/README.md`` §4)."""

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


def _passing_verdicts(root: str, pattern: str) -> List[Dict[str, Any]]:
    passing: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(root, pattern), recursive=True)):
        payload = _load_json(path)
        if payload and _verdict_status(payload) in (rec.STATUS_PASS, rec.STATUS_PASS_NEGATIVE, "True"):
            payload = dict(payload)
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


def _which(name: str) -> str:
    from shutil import which

    return which(name) or ""


def _probe_components() -> Dict[str, Dict[str, Any]]:
    probed: Dict[str, Dict[str, Any]] = {}
    for component, binaries in EXTERNAL_COMPONENTS.items():
        present = [f"{name}={_which(name)}" for name in binaries if _which(name)]
        probed[component] = {
            "present": present,
            "missing": [] if present else [f"{name}:{COMPONENT_STATE[component]}" for name in binaries],
            "state": "" if present else COMPONENT_STATE[component],
        }
    return probed


def check_prerequisites(root: str, *, probe: bool = True) -> PrerequisiteStatus:
    """Inspect the repository (and optionally the machine) for S13 readiness.

    Repository checks look for *evidence*, never for intent: a report claiming a
    feature is not evidence, a PASS verdict or a campaign manifest is.
    """
    checks: List[PrerequisiteCheck] = []

    stage_hits: Dict[str, str] = {}
    for stage, pattern in UPSTREAM_VERDICT_GLOBS.items():
        hits = _passing_verdicts(root, pattern)
        if hits:
            stage_hits[stage] = hits[0]["_path"]
    usable = len(stage_hits)
    checks.append(
        PrerequisiteCheck(
            name="upstream_evidence_chain",
            satisfied=usable >= MIN_UPSTREAM_STAGES,
            state="" if usable >= MIN_UPSTREAM_STAGES else rec.PREREQ_BLOCKED_PREREQUISITE,
            evidence=",".join(f"{stage}={path}" for stage, path in sorted(stage_hits.items())),
            reason=(
                ""
                if usable >= MIN_UPSTREAM_STAGES
                else f"only {usable} upstream stage(s) carry a PASS verdict; S13 inherits identity/quality "
                "contracts and may not substitute a theoretical value"
            ),
            detail={"usable_stages": usable, "stages": sorted(stage_hits)},
        )
    )

    s08 = _passing_verdicts(root, S08_VERDICT_GLOB)
    checks.append(
        PrerequisiteCheck(
            name="s08_service_contract",
            satisfied=bool(s08),
            state="" if s08 else rec.PREREQ_BLOCKED_PREREQUISITE,
            evidence=s08[0]["_path"] if s08 else "",
            reason=(
                ""
                if s08
                else "no S08 PASS verdict: without a real service/streaming/admission implementation S13 "
                "could only deploy a mock (§3.1 of details/S13/README.md)"
            ),
        )
    )

    s12 = sorted(glob.glob(os.path.join(root, S12_CAMPAIGN_GLOB))) + _passing_verdicts(root, S12_VERDICT_GLOB)
    checks.append(
        PrerequisiteCheck(
            name="s12_capacity_and_quality_baseline",
            satisfied=bool(s12),
            state="" if s12 else rec.PREREQ_BLOCKED_PREREQUISITE,
            evidence=str(s12[0]) if s12 else "",
            reason=(
                ""
                if s12
                else "no S12 capacity/quality baseline: E13-06 would have nothing to reconcile and "
                "E13-10 no thresholds to compare against"
            ),
        )
    )

    model_config = os.path.join(root, "configs", "models", "qwen3_1_7b.yaml")
    checks.append(
        PrerequisiteCheck(
            name="model_artifact_contract",
            satisfied=os.path.isfile(model_config),
            state="" if os.path.isfile(model_config) else rec.PREREQ_BLOCKED_PREREQUISITE,
            evidence=model_config,
            reason="" if os.path.isfile(model_config) else "frozen model artifact config is missing",
        )
    )

    infra_dir = os.path.join(root, "infra")
    checks.append(
        PrerequisiteCheck(
            name="deployment_assets_present",
            satisfied=os.path.isdir(infra_dir),
            state="" if os.path.isdir(infra_dir) else rec.PREREQ_BLOCKED_PREREQUISITE,
            evidence=infra_dir,
            reason="" if os.path.isdir(infra_dir) else "no infra/ asset tree (containers/helm/observability)",
        )
    )

    driver = os.path.join(root, "scripts", "infra", "run_e13.py")
    checks.append(
        PrerequisiteCheck(
            name="stable_cli_runner_for_gates",
            satisfied=os.path.isfile(driver),
            evidence=driver,
            reason="" if os.path.isfile(driver) else "no stable CLI driver for the S13 gates",
        )
    )

    if probe:
        components = _probe_components()
        for name, entry in components.items():
            checks.append(
                PrerequisiteCheck(
                    name=name,
                    satisfied=not entry["missing"],
                    state=entry["state"],
                    evidence=",".join(sorted(entry["present"])),
                    reason=(
                        ""
                        if not entry["missing"]
                        else "missing tooling: " + ", ".join(entry["missing"])
                    ),
                )
            )
    checks.append(
        PrerequisiteCheck(
            name="fault_authorization_declared",
            satisfied=_has_fault_authorization(root),
            state="" if _has_fault_authorization(root) else rec.PREREQ_NOT_RUN_SAFETY_BOUNDARY,
            evidence="configs/infra/experiment_spec.yaml" if _has_fault_authorization(root) else "",
            reason=(
                ""
                if _has_fault_authorization(root)
                else "no declared safety policy: destructive fault experiments stay NOT_RUN_SAFETY_BOUNDARY"
            ),
        )
    )
    return PrerequisiteStatus(stage=STAGE, checks=checks)


def _has_fault_authorization(root: str) -> bool:
    path = os.path.join(root, "configs", "infra", "experiment_spec.yaml")
    if not os.path.isfile(path):
        return False
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is a core dependency
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(document, dict) and bool(document.get("safety_policy_required", False))


# ── preregistration ───────────────────────────────────────────────────────


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
    release_ids: Sequence[str] = ()
    cluster_ids: Sequence[str] = ()
    fault_case_ids: Sequence[str] = ()
    frozen_inputs: Mapping[str, str] = field(default_factory=dict)
    exclusion_policy: Sequence[str] = ()
    abort_policy: Sequence[str] = ()
    safety_policy_id: str = ""
    repetitions: int = 0
    randomization_policy: str = ""
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
        if not self.abort_policy:
            problems.append("the abort/kill-switch policy must be frozen before a destructive experiment")
        if not self.safety_policy_id:
            problems.append("a safety policy id is required (§20.2)")
        if self.repetitions <= 0:
            problems.append("repetitions must be planned")
        if not self.release_ids:
            problems.append("the release identity under test must be frozen")
        return problems

    def payload(self) -> Dict[str, Any]:
        return {
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "hypothesis": self.hypothesis,
            "primary_metric": self.primary_metric,
            "thresholds": dict(sorted(self.thresholds.items())),
            "claim_boundary": self.claim_boundary,
            "non_claims": list(self.non_claims),
            "release_ids": list(self.release_ids),
            "cluster_ids": list(self.cluster_ids),
            "fault_case_ids": list(self.fault_case_ids),
            "frozen_inputs": dict(sorted(self.frozen_inputs.items())),
            "exclusion_policy": list(self.exclusion_policy),
            "abort_policy": list(self.abort_policy),
            "safety_policy_id": self.safety_policy_id,
            "repetitions": self.repetitions,
            "randomization_policy": self.randomization_policy,
            "deviations": list(self.deviations),
            "created_at": self.created_at or _utc_now(),
        }

    def to_json(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, indent=2, ensure_ascii=False)

    def prereg_hash(self) -> str:
        from hqsb.infra.identity import hash_payload

        return hash_payload(self.payload())


# ── experiment record (handbook §4 + S13 fields) ─────────────────────────


@dataclass
class ExperimentRecord:
    """The handbook record plus the S13 identity/governance fields."""

    experiment_id: str
    status: str = rec.STATUS_NOT_STARTED
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
    # S13 additions
    release_id: str = ""
    image_index_digest: str = ""
    cluster_id: str = ""
    namespace: str = ""
    model_artifact_id: str = ""
    observability_schema_version: str = ""
    capacity_policy_version: str = ""
    autoscaling_policy_version: str = ""
    canary_policy_id: str = ""
    fault_case_ids: Sequence[str] = ()
    tenant_scope: str = ""
    evidence_level: str = rec.EVIDENCE_DESIGN_ONLY
    evidence_axis_levels: Mapping[str, str] = field(default_factory=dict)
    safety_policy_id: str = ""
    abort_events: Sequence[str] = ()
    manual_interventions: Sequence[str] = ()
    allowed_claims: Sequence[str] = ()
    forbidden_claims: Sequence[str] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.experiment_id not in EXPERIMENTS:
            problems.append(f"unknown experiment id {self.experiment_id!r}")
        if self.status not in rec.PROTOCOL_STATUSES:
            problems.append(f"unknown status {self.status!r}")
        if self.actual_implementation and self.requested_implementation != self.actual_implementation:
            if not self.fallback_reason:
                problems.append("requested != actual without a fallback reason (silent fallback is forbidden)")
        if self.evidence_level not in rec.EVIDENCE_LEVELS:
            problems.append(f"unknown evidence level {self.evidence_level!r}")
        for axis, level in self.evidence_axis_levels.items():
            if level not in rec.EVIDENCE_LEVELS:
                problems.append(f"unknown evidence level {level!r} for axis {axis!r}")
        if not self.allowed_claims:
            problems.append("the allowed claims must be recorded (they bound the report wording)")
        if not self.forbidden_claims:
            problems.append("the forbidden claims must be recorded")
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
            "release_id": self.release_id,
            "image_index_digest": self.image_index_digest,
            "cluster_id": self.cluster_id,
            "namespace": self.namespace,
            "model_artifact_id": self.model_artifact_id,
            "observability_schema_version": self.observability_schema_version,
            "capacity_policy_version": self.capacity_policy_version,
            "autoscaling_policy_version": self.autoscaling_policy_version,
            "canary_policy_id": self.canary_policy_id,
            "fault_case_ids": list(self.fault_case_ids),
            "tenant_scope": self.tenant_scope,
            "evidence_level": self.evidence_level,
            "evidence_axis_levels": dict(sorted(self.evidence_axis_levels.items())),
            "safety_policy_id": self.safety_policy_id,
            "abort_events": list(self.abort_events),
            "manual_interventions": list(self.manual_interventions),
            "allowed_claims": list(self.allowed_claims),
            "forbidden_claims": list(self.forbidden_claims),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2, ensure_ascii=False)


def template_experiment_record(experiment_id: str, *, run_id: str = "", commit: str = "") -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=experiment_id,
        run_id=run_id,
        git_commit=commit,
        allowed_claims=("接口层能力成立",),
        forbidden_claims=("未运行实验前不得声称任何生产化/可靠性/安全结论",),
    )


# ── evidence manifest ────────────────────────────────────────────────────

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
    # S13 extensions
    "campaign_id",
    "release_id",
    "image_index_digest",
    "cluster_id",
    "namespace",
    "deployment_run_ids",
    "artifact_lifecycle_ids",
    "lifecycle_episode_ids",
    "capacity_campaign_id",
    "autoscaling_campaign_id",
    "observability_campaign_id",
    "fault_run_ids",
    "canary_run_ids",
    "security_scope_id",
    "evidence_level",
    "evidence_axis_levels",
    "safety_policy_id",
    "tenant_scope",
    "localization_scope",
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
    campaign_id: str = ""
    release_id: str = ""
    image_index_digest: str = ""
    cluster_id: str = ""
    namespace: str = ""
    deployment_run_ids: Tuple[str, ...] = ()
    artifact_lifecycle_ids: Tuple[str, ...] = ()
    lifecycle_episode_ids: Tuple[str, ...] = ()
    capacity_campaign_id: str = ""
    autoscaling_campaign_id: str = ""
    observability_campaign_id: str = ""
    fault_run_ids: Tuple[str, ...] = ()
    canary_run_ids: Tuple[str, ...] = ()
    security_scope_id: str = ""
    evidence_level: str = rec.EVIDENCE_DESIGN_ONLY
    evidence_axis_levels: Mapping[str, str] = field(default_factory=dict)
    safety_policy_id: str = ""
    tenant_scope: str = "single_tenant"
    localization_scope: str = ""
    schema_version: str = "1.0.0"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.run_id:
            problems.append("manifest needs a run_id")
        if self.claim_level not in CLAIM_LEVELS:
            problems.append(f"unknown claim level {self.claim_level!r}")
        if self.evidence_level not in rec.EVIDENCE_LEVELS:
            problems.append(f"unknown evidence level {self.evidence_level!r}")
        for axis, level in self.evidence_axis_levels.items():
            if level not in rec.EVIDENCE_LEVELS:
                problems.append(f"unknown evidence level {level!r} for axis {axis!r}")
        for artifact in list(self.raw_artifacts) + list(self.normalized_artifacts):
            if "uri" not in artifact or "sha256" not in artifact:
                problems.append(f"artifact row without uri/sha256: {artifact}")
        if rec.EVIDENCE_RANK[self.evidence_level] >= rec.EVIDENCE_RANK[rec.EVIDENCE_TESTED_SINGLE_RUN]:
            if not self.raw_artifacts:
                problems.append(
                    f"evidence level {self.evidence_level} requires raw artifacts (a run without raw data "
                    "cannot support a tested claim)"
                )
            if not self.release_id:
                problems.append(f"evidence level {self.evidence_level} requires a release identity")
        if self.correctness_status not in ("pass", "fail", "not_run"):
            problems.append(f"unknown correctness status {self.correctness_status!r}")
        if self.claim_level in ("SERVICE", "PORTABLE") and not self.cluster_id:
            problems.append(
                f"claim level {self.claim_level} needs the cluster identity it was observed on"
            )
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
            "campaign_id": self.campaign_id,
            "release_id": self.release_id,
            "image_index_digest": self.image_index_digest,
            "cluster_id": self.cluster_id,
            "namespace": self.namespace,
            "deployment_run_ids": list(self.deployment_run_ids),
            "artifact_lifecycle_ids": list(self.artifact_lifecycle_ids),
            "lifecycle_episode_ids": list(self.lifecycle_episode_ids),
            "capacity_campaign_id": self.capacity_campaign_id,
            "autoscaling_campaign_id": self.autoscaling_campaign_id,
            "observability_campaign_id": self.observability_campaign_id,
            "fault_run_ids": list(self.fault_run_ids),
            "canary_run_ids": list(self.canary_run_ids),
            "security_scope_id": self.security_scope_id,
            "evidence_level": self.evidence_level,
            "evidence_axis_levels": dict(sorted(self.evidence_axis_levels.items())),
            "safety_policy_id": self.safety_policy_id,
            "tenant_scope": self.tenant_scope,
            "localization_scope": self.localization_scope,
        }

    def to_json(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, indent=2, ensure_ascii=False)

    def as_dict(self) -> Dict[str, Any]:
        return self.payload()


# ── run directory and the triple gate ────────────────────────────────────


class RunDirectory(RunStorage):
    """Creates and manages one ``experiment_results/S13/<E>/<run_id>/`` tree."""

    def __init__(self, root: str, experiment_id: str, run_id: str) -> None:
        if experiment_id not in EXPERIMENTS:
            raise ConfigError(f"unknown experiment id {experiment_id!r}")
        if not run_id:
            raise ConfigError("run_id must not be empty")
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.root = root
        self.path = run_directory_path(root, STAGE, experiment_id, run_id)

    def create(self, *, subdirs: Sequence[str] = ()) -> str:
        os.makedirs(self.path, exist_ok=True)
        for sub in subdirs:
            os.makedirs(os.path.join(self.path, sub), exist_ok=True)
        return self.path





    def write_preregistration(self, prereg: Preregistration) -> str:
        problems = prereg.validate()
        if problems:
            raise ConfigError("invalid preregistration: " + "; ".join(problems))
        return self.write_text("preregistration.json", prereg.to_json())

    def write_experiment_record(self, record: ExperimentRecord) -> str:
        problems = record.validate()
        if problems:
            raise ConfigError("invalid experiment record: " + "; ".join(problems))
        return self.write_text("experiment_record.json", record.to_json())

    def write_evidence_manifest(self, manifest: EvidenceManifest) -> str:
        problems = manifest.validate()
        if problems:
            raise ConfigError("invalid evidence manifest: " + "; ".join(problems))
        return self.write_text("evidence_manifest.json", manifest.to_json())

    def raw_file_count(self) -> int:
        """Count raw evidence files (a verdict needs raw data, not directories)."""
        for candidate in (
            "raw", "supply_chain", "cluster", "deployment", "artifacts", "lifecycle",
            "capacity", "autoscaling", "observability", "faults", "canary", "security",
        ):
            base = os.path.join(self.path, candidate)
            if os.path.isdir(base):
                count = sum(len(files) for _root, _dirs, files in os.walk(base))
                if count:
                    return count
        return 0

    def write_report_skeleton(self, experiment_id: str, title: str) -> str:
        skeleton = (
            f"# {experiment_id}: {title}\n\n"
            "> 状态：**NOT_RUN**（本文件由 `hqsb.infra.experiment` 生成的骨架；\n"
            "> 未执行任何实验，未产出任何数值）。\n\n"
            "## 1. 预注册\n\n见 `preregistration.json`。\n\n"
            "## 2. 环境与身份\n\n见 `environment_fingerprint.json`、`evidence_manifest.json`；"
            "未执行时为占位。\n\n"
            "## 3. 原始数据\n\n本 run 下的 `raw/`…`security/` 目录均为空，直到实验真正运行。\n\n"
            "## 4. 结论\n\n无。实验未执行，禁止填写任何部署/供应链/容量/可观测/故障/canary/安全结论。\n"
        )
        return self.write_text("report.md", skeleton)

    def write_verdict(self, *, status: str, reason: str, prerequisites: PrerequisiteStatus,
                      executed: bool, allow_execute: bool = False, raw_samples: int = 0,
                      limitations: Sequence[str] = (), evidence_level: str = "") -> Dict[str, Any]:
        """Write a verdict — or refuse when the run cannot support one (triple gate)."""
        if status not in rec.PROTOCOL_STATUSES:
            raise ConfigError(f"unknown status {status!r}; protocol allows {list(rec.PROTOCOL_STATUSES)}")
        effective = status
        effective_reason = reason
        if status in GATED_CONCLUSION_STATUSES:
            if not allow_execute:
                effective = rec.STATUS_BLOCKED
                effective_reason = (
                    "refusing to emit an experimental conclusion: execution is disabled "
                    "(pass allow_execute=True and run the experiment properly)"
                )
            elif not prerequisites.satisfied:
                effective = rec.STATUS_BLOCKED
                effective_reason = "prerequisites unsatisfied: " + ", ".join(prerequisites.missing)
            elif not executed or raw_samples <= 0:
                effective = rec.STATUS_BLOCKED
                effective_reason = (
                    "no raw samples recorded; a conclusion without raw evidence is not allowed"
                )
        if evidence_level and evidence_level not in rec.EVIDENCE_LEVELS:
            raise ConfigError(f"unknown evidence level {evidence_level!r}")
        payload = {
            "stage": STAGE,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": effective,
            "requested_status": status,
            "reason": effective_reason,
            "executed": executed,
            "raw_samples": raw_samples,
            "evidence_level": evidence_level or rec.EVIDENCE_DESIGN_ONLY,
            "prerequisites": prerequisites.as_dict(),
            "limitations": list(limitations),
            "written_at": _utc_now(),
        }
        self.write_json("verdict.json", payload)
        return payload

    def write_status(self, status: str, reason: str, prerequisites: PrerequisiteStatus,
                     executed: bool = False) -> Dict[str, Any]:
        """Write a non-conclusion status (NOT_STARTED / RUNNING / BLOCKED)."""
        if status in rec.CONCLUSION_STATUSES:
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


# ── environment and git state ────────────────────────────────────────────


def environment_fingerprint(extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Machine-readable environment record (never contains secrets)."""
    payload: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
        "collected_at": _utc_now(),
        "arch": platform.machine(),
    }
    payload["container_tooling"] = {name: _which(name) for name in EXTERNAL_COMPONENTS["container_tooling"]}
    payload["cluster_tooling"] = {name: _which(name) for name in EXTERNAL_COMPONENTS["cluster_tooling"]}
    payload["scan_tooling"] = {name: _which(name) for name in EXTERNAL_COMPONENTS["scan_tooling"]}
    payload["kubectl_context"] = _kubectl_context()
    if extra:
        payload["extra"] = dict(extra)
    return payload


def _kubectl_context() -> Dict[str, Any]:
    """Best-effort current context; the *name* only, never credentials."""
    if not _which("kubectl"):
        return {"available": False, "state": rec.PREREQ_NOT_RUN_TOOL_UNAVAILABLE}
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv
            ["kubectl", "config", "current-context"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "state": rec.PREREQ_NOT_RUN_PERMISSION_DENIED}
    context = completed.stdout.strip()
    return {"available": bool(context), "context": context, "cluster_reachable": None}


def git_state(root: str) -> Dict[str, Any]:
    """Commit + dirty flag + diff hash (never rewrites anything)."""
    from hqsb.infra.identity import sha256_text

    commit = _git(root, ["rev-parse", "HEAD"])
    status = _git(root, ["status", "--porcelain"])
    diff = _git(root, ["diff", "--no-color"])
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "dirty_files": sorted(line[3:] for line in status.splitlines() if line.strip()),
        "diff_sha256": sha256_text(diff) if diff else "",
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


def interface_only_run(root: str, experiment_id: str, *, run_id: Optional[str] = None,
                       probe: bool = False) -> Dict[str, Any]:
    """Create a run directory that records *interfaces only* (no experiment)."""
    from hqsb.infra import interface_map

    rid = run_id or f"interface_only_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run = RunDirectory(root, experiment_id, rid)
    run.create()
    status = check_prerequisites(root, probe=probe)
    run.write_json("environment_fingerprint.json", environment_fingerprint())
    run.write_json("prerequisites.json", status.as_dict())
    run.write_json("interface_map.json", interface_map.mapping_for(experiment_id).as_dict())
    run.write_report_skeleton(experiment_id, interface_map.mapping_for(experiment_id).title)
    verdict = run.write_verdict(
        status=rec.STATUS_BLOCKED,
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


def experiment_titles() -> Dict[str, str]:
    """Experiment id → title, resolved through the interface map (single source)."""
    from hqsb.infra import interface_map

    return {mapping.experiment_id: mapping.title for mapping in interface_map.EXPERIMENTS}
