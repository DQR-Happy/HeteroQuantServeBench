"""S12 experiment scaffolding: prerequisites, preregistration, evidence, verdicts.

The scaffolding implements the full "preregister → collect → write → decide"
structure every S12 experiment needs while making it *impossible by default* to
emit a conclusion:

* :func:`check_prerequisites` reads the repository for real evidence paths
  (never for intent) and reports each check with its evidence pointer;
* :class:`RunDirectory` writes only under ``experiment_results/S12/...`` — never
  into ``docs/stage_experiments`` (the protocol tree is read-only);
* :meth:`RunDirectory.write_verdict` refuses ``PASS``/``FAIL``/``PASS_NEGATIVE``
  unless execution was explicitly allowed, the prerequisites are satisfied *and*
  raw samples exist; otherwise the status is downgraded to ``BLOCKED`` with a
  reason;
* :class:`EvidenceManifest` carries the S12 identity fields (comparison /
  candidate / capability / layer / energy / cost / lineage / regeneration
  level) and only accepts an evidence level that its payload can support.

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
from hqsb.evaluation.campaign import ARTIFACT_LAYOUT, ArtifactLayout
from hqsb.evaluation.identity import canonical_json, sha256_text
from hqsb.evaluation.records import (
    CONCLUSION_STATUSES,
    EVIDENCE_LEVELS,
    PROTOCOL_STATUSES,
    REGENERATION_LEVELS,
    RESULT_CLASSES,
    STATUS_BLOCKED,
    STATUS_FAIL,
    STATUS_NOT_STARTED,
    STATUS_PASS,
    STATUS_PASS_NEGATIVE,
)

#: Only these three are *performance/conclusion* statuses that the triple gate
#: must refuse; ``BLOCKED`` and ``N/A_BY_ADR`` are written with an explicit
#: reason and never silently downgraded to "execution disabled".
GATED_CONCLUSION_STATUSES: Tuple[str, ...] = (
    STATUS_PASS,
    STATUS_PASS_NEGATIVE,
    STATUS_FAIL,
)

STAGE = "S12"

EXPERIMENTS: Tuple[str, ...] = tuple(f"E12-{index:02d}" for index in range(1, 11))

#: One run directory holds raw *and* derived material for one experiment run.
RUN_LAYOUT: Tuple[str, ...] = ARTIFACT_LAYOUT

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
UPSTREAM_VERDICT_GLOBS: Mapping[str, str] = {
    "S03": "docs/stage_experiments/S03/*/raw*/verdict.json",
    "S04": "docs/stage_experiments/S04/*/raw*/verdict.json",
    "S05": "docs/stage_experiments/S05/*/raw*/verdict.json",
    "S06": "docs/stage_experiments/S06/*/raw*/verdict.json",
    "S07": "docs/stage_experiments/S07/*/raw*/verdict.json",
    "S08": "docs/stage_experiments/S08/*/raw*/verdict.json",
    "S09": "docs/stage_experiments/S09/*/raw*/verdict.json",
    "S10": "docs/stage_experiments/S10/*/raw*/verdict.json",
    "S11": "docs/stage_experiments/S11/*/raw*/verdict.json",
}
DEV_BASELINE_GLOB = "reports/dev/*/gate*/s04_backend_baseline.json"
ENVIRONMENT_GLOB = "reports/dev/*/gate0_environment/environment.json"
FINGERPRINT_GLOB = "docs/stage_experiments/*/**/environment_fingerprint.json"
JETSON_EVIDENCE_GLOB = "reports/jetson/**/*.json"
PRICE_SNAPSHOT_GLOB = "experiment_results/S12/*/cost/price_snapshots/*.json"
METER_CAPABILITY_GLOB = "experiment_results/S12/*/energy/capability/*.json"

#: Minimum number of upstream stages (S03–S11) that must carry a PASS verdict
#: before S12 may claim a cross-hardware comparison.
MIN_UPSTREAM_STAGES = 2

#: S12 needs at least three hardware types, or two types with two architectures
#: (details/S12/README.md §24.2).
MIN_HARDWARE_TYPES = 3
MIN_HARDWARE_TYPES_TWO_ARCH = 2
MIN_ARCHITECTURES = 2


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
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "satisfied": self.satisfied,
            "required": self.required,
            "evidence": self.evidence,
            "reason": self.reason,
        }
        if self.detail:
            payload["detail"] = dict(sorted(self.detail.items()))
        return payload


@dataclass
class PrerequisiteStatus:
    """Aggregate prerequisite status for S12."""

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
    """Inspect the repository for the S12 evidence chain (never for intent)."""
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
                else "no S01 PASS verdict: ModelArtifact/WorkloadSpec/BenchmarkResult contracts, "
                "schema validation and artifact identity are the join keys of every S12 cell"
            ),
        )
    )

    s02 = _passing_verdicts(root, S02_VERDICT_GLOB)
    model_config = os.path.join(root, "configs", "models", "qwen3_1_7b.yaml")
    checks.append(
        PrerequisiteCheck(
            name="s02_model_workload_quality_freeze",
            satisfied=bool(s02) and os.path.isfile(model_config),
            evidence=s02[0]["_path"] if s02 else model_config,
            reason=(
                ""
                if s02 and os.path.isfile(model_config)
                else "S02 verdicts and/or the frozen Qwen model config are missing: without a "
                "replayable reference, workload and quality gate there is no common estimand"
            ),
        )
    )

    stage_hits: Dict[str, str] = {}
    for stage, pattern in UPSTREAM_VERDICT_GLOBS.items():
        hits = _passing_verdicts(root, pattern)
        if hits:
            stage_hits[stage] = hits[0]["_path"]
    usable_stages = len(stage_hits)
    checks.append(
        PrerequisiteCheck(
            name="s03_s11_upstream_evidence_chain",
            satisfied=usable_stages >= MIN_UPSTREAM_STAGES,
            evidence=",".join(f"{stage}={path}" for stage, path in sorted(stage_hits.items())),
            reason=(
                ""
                if usable_stages >= MIN_UPSTREAM_STAGES
                else f"only {usable_stages} of S03–S11 carry a PASS verdict "
                f"(need >= {MIN_UPSTREAM_STAGES}); missing upstream evidence may not be replaced "
                "by a theoretical value"
            ),
            detail={"usable_stages": usable_stages, "stages": sorted(stage_hits)},
        )
    )

    hardware = _hardware_coverage(root, stage_hits)
    coverage_ok = int(hardware["types"]) >= MIN_HARDWARE_TYPES or (
        int(hardware["types"]) >= MIN_HARDWARE_TYPES_TWO_ARCH
        and int(hardware["architectures"]) >= MIN_ARCHITECTURES
    )
    checks.append(
        PrerequisiteCheck(
            name="multi_hardware_coverage",
            satisfied=coverage_ok,
            evidence=",".join(hardware["evidence"]),
            reason=(
                ""
                if coverage_ok
                else "S12 needs at least three hardware types, or two types with two "
                "architectures ({}, {} found); a single-platform study cannot satisfy the "
                "stage gate".format(hardware["types"], hardware["architectures"])
            ),
            detail={"types": hardware["types"], "architectures": hardware["architectures"]},
        )
    )

    fingerprint_candidates = sorted(
        glob.glob(os.path.join(root, FINGERPRINT_GLOB), recursive=True)
    ) + sorted(glob.glob(os.path.join(root, ENVIRONMENT_GLOB)))
    frozen_environment = _first_fingerprint_with_versions(fingerprint_candidates)
    checks.append(
        PrerequisiteCheck(
            name="frozen_evaluation_environment",
            satisfied=bool(frozen_environment),
            evidence=frozen_environment,
            reason=(
                ""
                if frozen_environment
                else "no environment fingerprint with frozen torch/driver/arch values: "
                "hardware identity, power mode and clock policy would be incomplete"
            ),
        )
    )

    tools = _probe_export_tools()
    checks.append(
        PrerequisiteCheck(
            name="profiler_and_export_tooling",
            satisfied=not tools["missing"],
            evidence=",".join(sorted(tools["present"])),
            reason=(
                ""
                if not tools["missing"]
                else "missing tooling for profile/roofline attribution: " + ", ".join(tools["missing"])
            ),
            required=False,
            detail={"missing": tools["missing"]},
        )
    )

    meter_capabilities = sorted(glob.glob(os.path.join(root, METER_CAPABILITY_GLOB), recursive=True))
    checks.append(
        PrerequisiteCheck(
            name="power_meter_capability_evidence",
            satisfied=bool(meter_capabilities),
            evidence=meter_capabilities[0] if meter_capabilities else "",
            reason=(
                ""
                if meter_capabilities
                else "no E12-02 telemetry capability record: platforms without a verified meter "
                "must stay MEASUREMENT_UNAVAILABLE (TDP is not a measurement)"
            ),
            required=False,
        )
    )

    price_snapshots = sorted(glob.glob(os.path.join(root, PRICE_SNAPSHOT_GLOB), recursive=True))
    checks.append(
        PrerequisiteCheck(
            name="price_source_snapshots",
            satisfied=bool(price_snapshots),
            evidence=price_snapshots[0] if price_snapshots else "",
            reason=(
                ""
                if price_snapshots
                else "no dated price snapshot: TCO/cost claims are blocked (performance and "
                "energy conclusions may still be published)"
            ),
            required=False,
        )
    )

    checks.append(
        PrerequisiteCheck(
            name="independent_evidence_dirs",
            satisfied=os.path.isfile(os.path.join(root, "hqsb", "evaluation", "campaign.py")),
            evidence=os.path.join("hqsb", "evaluation", "campaign.py"),
            reason=(
                ""
                if os.path.isfile(os.path.join(root, "hqsb", "evaluation", "campaign.py"))
                else "the campaign/run-directory scaffolding is unavailable"
            ),
        )
    )

    driver = os.path.join(root, "scripts", "evaluation", "run_e12.py")
    checks.append(
        PrerequisiteCheck(
            name="stable_cli_runner_for_gates",
            satisfied=os.path.isfile(driver),
            evidence=driver,
            reason=(
                ""
                if os.path.isfile(driver)
                else "no stable CLI driver: comparability/capability/benchmark/energy/cost gates "
                "must be callable from one entry point"
            ),
        )
    )

    return PrerequisiteStatus(stage=STAGE, checks=checks)


def _hardware_coverage(root: str, stage_hits: Mapping[str, str]) -> Dict[str, Any]:
    """Count declared hardware instances of *distinct* type and architecture.

    Deliberately conservative: only explicit evidence files count.  A device
    name in prose does not, and two instances of the same SKU are one type.
    """
    found: Dict[str, str] = {}
    evidence: List[str] = []

    jetson = sorted(glob.glob(os.path.join(root, JETSON_EVIDENCE_GLOB), recursive=True))
    if jetson:
        found["jetson_orin|sm_87"] = jetson[0]
        evidence.append(jetson[0])

    dev_baseline = sorted(glob.glob(os.path.join(root, DEV_BASELINE_GLOB)))
    if dev_baseline:
        found["rtx3090|sm_86"] = dev_baseline[0]
        evidence.append(dev_baseline[0])

    if "S09" in stage_hits:
        found["ascend|npu"] = stage_hits["S09"]
        evidence.append(stage_hits["S09"])

    if "S10" in stage_hits:
        found["multi_device|topology"] = stage_hits["S10"]
        evidence.append(stage_hits["S10"])

    if "S03" in stage_hits or "S04" in stage_hits:
        found["cuda_kernel|sm"] = stage_hits.get("S03") or stage_hits.get("S04", "")
        evidence.append(found["cuda_kernel|sm"])

    types = len({name.split("|")[0] for name in found})
    architectures = len({name.split("|")[1] for name in found})
    return {"types": types, "architectures": architectures, "instances": sorted(found), "evidence": evidence}


def _first_fingerprint_with_versions(candidates: Sequence[str]) -> str:
    required = ("torch", "arch")
    for path in candidates:
        payload = _load_json(path)
        if not payload:
            continue
        blob = json.dumps(payload, sort_keys=True)
        if all(name in blob for name in required):
            return path
    return ""


def _probe_export_tools() -> Dict[str, Any]:
    names = ("nvcc", "cuobjdump", "nvdisasm", "nsys", "ncu", "nvidia-smi", "tegrastats")
    present: List[str] = []
    missing: List[str] = []
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


# ── preregistration / experiment record / evidence manifest ───────────────


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
    comparison_ids: Sequence[str] = ()
    candidate_ids: Sequence[str] = ()
    layers: Sequence[str] = ()
    frozen_inputs: Mapping[str, str] = field(default_factory=dict)
    exclusion_policy: Sequence[str] = ()
    normalization_policy: Sequence[str] = ()
    deviations: Sequence[str] = ()
    created_at: str = ""

    def validate(self) -> List[str]:
        from hqsb.evaluation.layers import LAYERS

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
        if not self.comparison_ids:
            problems.append("a preregistration must reference the frozen comparison ids")
        if not self.exclusion_policy:
            problems.append("the outlier/exclusion policy must be frozen before looking at results")
        unknown_layers = sorted(set(self.layers) - set(LAYERS))
        if unknown_layers:
            problems.append(f"unknown layers in preregistration: {unknown_layers}")
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
            "comparison_ids": list(self.comparison_ids),
            "candidate_ids": list(self.candidate_ids),
            "layers": list(self.layers),
            "frozen_inputs": dict(sorted(self.frozen_inputs.items())),
            "exclusion_policy": list(self.exclusion_policy),
            "normalization_policy": list(self.normalization_policy),
            "deviations": list(self.deviations),
            "created_at": self.created_at or _utc_now(),
        }

    def to_json(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, indent=2, ensure_ascii=False)

    def prereg_hash(self) -> str:
        return sha256_text(canonical_json(self.payload()))


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
    # S12 additions
    "comparison_ids",
    "candidate_ids",
    "layers",
    "cell_status",
    "metric_status",
    "claim_status",
    "evidence_levels",
)


@dataclass
class ExperimentRecord:
    """The handbook §4 record plus S12's cell/metric/claim statuses."""

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
    comparison_ids: Sequence[str] = ()
    candidate_ids: Sequence[str] = ()
    layers: Sequence[str] = ()
    cell_status: str = "NOT_RUN"
    metric_status: str = "MISSING"
    claim_status: str = "NOT_PUBLISHABLE"
    evidence_levels: Mapping[str, str] = field(default_factory=dict)

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
        for axis, level in self.evidence_levels.items():
            if level not in EVIDENCE_LEVELS:
                problems.append(f"unknown evidence level {level!r} for axis {axis!r}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        payload = {
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
            "comparison_ids": list(self.comparison_ids),
            "candidate_ids": list(self.candidate_ids),
            "layers": list(self.layers),
            "cell_status": self.cell_status,
            "metric_status": self.metric_status,
            "claim_status": self.claim_status,
            "evidence_levels": dict(sorted(self.evidence_levels.items())),
        }
        return payload

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2, ensure_ascii=False)


def template_experiment_record(experiment_id: str, *, run_id: str = "", commit: str = "") -> ExperimentRecord:
    return ExperimentRecord(experiment_id=experiment_id, run_id=run_id, git_commit=commit)


# ── evidence manifest (§26.3 + S12 extensions) ────────────────────────────

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
    # S12 extensions
    "campaign_id",
    "comparison_ids",
    "candidate_ids",
    "platform_instance_ids",
    "layers",
    "capability_evidence_ids",
    "capability_status",
    "comparability_status",
    "quality_status",
    "stability_status",
    "energy_measurement_ids",
    "meter_boundary",
    "cost_result_ids",
    "price_source_ids",
    "maturity_record_ids",
    "measurement_boundary_id",
    "result_class",
    "regeneration_level",
    "lineage_root",
    "entity_refs",
    "artifact_aggregate_root",
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
    comparison_ids: Tuple[str, ...] = ()
    candidate_ids: Tuple[str, ...] = ()
    platform_instance_ids: Tuple[str, ...] = ()
    layers: Tuple[str, ...] = ()
    capability_evidence_ids: Tuple[str, ...] = ()
    capability_status: str = ""
    comparability_status: str = ""
    quality_status: str = ""
    stability_status: str = ""
    energy_measurement_ids: Tuple[str, ...] = ()
    meter_boundary: str = ""
    cost_result_ids: Tuple[str, ...] = ()
    price_source_ids: Tuple[str, ...] = ()
    maturity_record_ids: Tuple[str, ...] = ()
    measurement_boundary_id: str = ""
    result_class: str = "MEASURED"
    regeneration_level: str = "R0"
    lineage_root: str = ""
    entity_refs: Tuple[str, ...] = ()
    artifact_aggregate_root: str = ""
    schema_version: str = "1.0.0"

    def validate(self) -> List[str]:
        from hqsb.evaluation.layers import LAYERS

        problems: List[str] = []
        if not self.run_id:
            problems.append("manifest needs a run_id")
        if self.claim_level not in CLAIM_LEVELS:
            problems.append(f"unknown claim level {self.claim_level!r}")
        if self.result_class not in RESULT_CLASSES:
            problems.append(f"unknown result class {self.result_class!r}")
        if self.regeneration_level not in REGENERATION_LEVELS:
            problems.append(f"unknown regeneration level {self.regeneration_level!r}")
        unknown_layers = sorted(set(self.layers) - set(LAYERS))
        if unknown_layers:
            problems.append(f"unknown layers in manifest: {unknown_layers}")
        for artifact in list(self.raw_artifacts) + list(self.normalized_artifacts):
            if "uri" not in artifact or "sha256" not in artifact:
                problems.append(f"artifact row without uri/sha256: {artifact}")
        if self.regeneration_level in ("R2", "R3", "R4") and not self.lineage_root:
            problems.append(
                f"regeneration level {self.regeneration_level} needs a lineage root "
                "(otherwise the claim is not reproducible)"
            )
        if self.result_class == "MEASURED" and self.correctness_status not in ("pass", "not_run"):
            problems.append(
                "a MEASURED result must not carry failing correctness: quality precedes performance"
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
            "comparison_ids": list(self.comparison_ids),
            "candidate_ids": list(self.candidate_ids),
            "platform_instance_ids": list(self.platform_instance_ids),
            "layers": list(self.layers),
            "capability_evidence_ids": list(self.capability_evidence_ids),
            "capability_status": self.capability_status,
            "comparability_status": self.comparability_status,
            "quality_status": self.quality_status,
            "stability_status": self.stability_status,
            "energy_measurement_ids": list(self.energy_measurement_ids),
            "meter_boundary": self.meter_boundary,
            "cost_result_ids": list(self.cost_result_ids),
            "price_source_ids": list(self.price_source_ids),
            "maturity_record_ids": list(self.maturity_record_ids),
            "measurement_boundary_id": self.measurement_boundary_id,
            "result_class": self.result_class,
            "regeneration_level": self.regeneration_level,
            "lineage_root": self.lineage_root,
            "entity_refs": list(self.entity_refs),
            "artifact_aggregate_root": self.artifact_aggregate_root,
        }

    def to_json(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, indent=2, ensure_ascii=False)

    def as_dict(self) -> Dict[str, Any]:
        return self.payload()


# ── run directory ──────────────────────────────────────────────────────────


class RunDirectory(RunStorage):
    """Creates and manages one ``experiment_results/S12/<E>/<run_id>/`` tree."""

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
        """Count raw evidence files, preferring the benchmark/energy directories."""
        for candidate in (
            "benchmark/raw",
            "benchmark",
            "capability/evidence",
            "energy/raw_power",
            "repeatability/telemetry",
            "cost/source_snapshots",
            "maturity/sessions",
            "lineage",
            "inventory",
            "comparability",
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
            f"> 状态：**NOT_RUN**（本文件由 `hqsb.evaluation.experiment` 生成的骨架；\n"
            f"> 未执行任何实验，未产出任何数值）。\n\n"
            "## 1. 预注册\n\n见 `preregistration.json`。\n\n"
            "## 2. 环境与身份\n\n见 `environment_fingerprint.json`、`evidence_manifest.json`；"
            "未执行时为占位。\n\n"
            "## 3. 原始数据\n\n`benchmark/`、`capability/`、`repeatability/`、`energy/`、"
            "`cost/`、`maturity/` 均为空，直到实验真正运行。\n\n"
            "## 4. 结论\n\n无。实验未执行，禁止填写任何可比性/capability/性能/能耗/成本/"
            "成熟度/lineage 结论。\n"
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
        cell_status: str = "",
        metric_status: str = "",
        claim_status: str = "",
    ) -> Dict[str, Any]:
        """Write a verdict — or refuse when the run cannot support one."""
        if status not in PROTOCOL_STATUSES:
            raise ConfigError(
                f"unknown status {status!r}; protocol allows {list(PROTOCOL_STATUSES)}"
            )
        effective = status
        effective_reason = reason
        if status in GATED_CONCLUSION_STATUSES:
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
            "cell_status": cell_status,
            "metric_status": metric_status,
            "claim_status": claim_status,
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
    for module_name in ("torch", "triton", "transformers"):
        payload[module_name] = _optional_module_version(module_name)
    payload["cuda_driver"] = _cuda_driver_version()
    payload["arch"] = payload["machine"]
    payload["power_mode"] = _power_mode()
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


def _power_mode() -> str:
    """Best-effort power-mode string; empty when the platform cannot report it."""
    for command in (["nvpmodel", "-q"], ["tegrastats", "--help"]):
        if not _which(command[0]):
            continue
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv
                command,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        text = (completed.stdout or "").strip().splitlines()
        if text:
            return text[0][:120]
    return ""


def git_state(root: str) -> Dict[str, Any]:
    """Commit + dirty flag + diff hash (never rewrites anything)."""
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


def interface_only_run(
    root: str, experiment_id: str, *, run_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a run directory that records *interfaces only* (no experiment)."""
    from hqsb.evaluation import interface_map

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


def campaign_layout(root: str, campaign_id: str) -> ArtifactLayout:
    """Convenience accessor so callers do not re-implement the layout rules."""
    return ArtifactLayout(root, campaign_id)
