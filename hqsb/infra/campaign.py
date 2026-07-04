"""S13 campaign layout, manifest and execution-safety policy (§20–§21).

``details/S13/README.md`` §21 fixes the *logical* artifact tree of an S13 campaign
(``artifacts/S13/<campaign_id>/…``).  As in S12, the logical hierarchy is kept
exactly while the physical root follows the repository convention
(``experiment_results/S13/<campaign_id>/``); the drift is registered in the
development report instead of being silently renamed.

§20.2 additionally requires that every destructive experiment runs under a signed
scope: authorized namespace/cluster, explicit device/node range, kill switch,
maximum duration/cost/replica count, error-budget cap and protected credentials.
:class:`SafetyPolicy` is that record and :func:`validate_safety_policy` refuses an
incomplete one *before* any injection is scheduled.

Nothing here creates a directory or runs anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.infra import identity as idn
from hqsb.infra import records as rec

SCHEMA_VERSION = "1.0.0"

STAGE = "S13"

#: Logical layout of §21, verbatim (order preserved for report readability).
ARTIFACT_LAYOUT: Tuple[str, ...] = (
    "preregistration",
    "supply_chain/builds",
    "supply_chain/images",
    "supply_chain/sbom",
    "supply_chain/provenance",
    "supply_chain/vulnerability",
    "supply_chain/secrets",
    "supply_chain/licenses",
    "supply_chain/gates",
    "cluster/inventory",
    "cluster/iac",
    "cluster/device_plugins",
    "cluster/policies",
    "cluster/events",
    "deployment/manifests",
    "deployment/timelines",
    "deployment/states",
    "artifacts/model_manifests",
    "artifacts/downloads",
    "artifacts/cache",
    "artifacts/activation",
    "lifecycle/probes",
    "lifecycle/requests",
    "lifecycle/drain",
    "lifecycle/rollouts",
    "capacity/predictions",
    "capacity/admissions",
    "capacity/sweeps",
    "autoscaling/metrics",
    "autoscaling/decisions",
    "autoscaling/replicas",
    "observability/metrics",
    "observability/logs",
    "observability/traces",
    "observability/profiles",
    "observability/alerts",
    "faults/specs",
    "faults/injections",
    "faults/recoveries",
    "faults/postmortems",
    "canary/assignments",
    "canary/analyses",
    "canary/decisions",
    "canary/rollback",
    "security/identities",
    "security/authorization",
    "security/quota",
    "security/network",
    "security/redaction",
    "security/abuse",
    "lineage",
    "reports/runbooks",
)

#: Files the campaign root must carry (raw evidence cannot be reconstructed later).
ROOT_FILES: Tuple[str, ...] = (
    "campaign_manifest.yaml",
    "preregistration/questions.yaml",
    "preregistration/releases.parquet",
    "preregistration/workloads.parquet",
    "preregistration/slos.yaml",
    "preregistration/thresholds.yaml",
    "preregistration/safety.yaml",
    "lineage/entities.parquet",
    "lineage/activities.parquet",
    "lineage/edges.parquet",
    "reports/acceptance.json",
)

#: §21 report names; the reliability/security reports are experiment-dependent and
#: therefore expected to be *absent* until the experiments run.
REQUIRED_REPORTS: Tuple[str, ...] = (
    "reports/production_architecture.md",
    "reports/runbooks/index.md",
)
EXPERIMENT_DEPENDENT_REPORTS: Tuple[str, ...] = (
    "reports/reliability_report.md",
    "reports/security_supply_chain.md",
)

#: Logical §21 root → physical repository root (registered drift).
LOGICAL_ROOT_TEMPLATE = "artifacts/S13/{campaign_id}"
PHYSICAL_ROOT_TEMPLATE = "experiment_results/S13/{campaign_id}"


def logical_root(campaign_id: str) -> str:
    return LOGICAL_ROOT_TEMPLATE.format(campaign_id=campaign_id)


def physical_root(root: str, campaign_id: str) -> str:
    if not campaign_id:
        raise ConfigError("campaign_id must not be empty")
    return os.path.abspath(os.path.join(root, "experiment_results", STAGE, campaign_id))


# ── campaign manifest ─────────────────────────────────────────────────────


@dataclass
class CampaignManifest:
    """Freezes §20.1 before any experiment runs."""

    campaign_id: str
    stage: str = STAGE
    created_at: str = ""
    release_ids: Tuple[str, ...] = ()
    image_index_digests: Mapping[str, str] = field(default_factory=dict)
    model_artifact_ids: Tuple[str, ...] = ()
    cluster_ids: Tuple[str, ...] = ()
    cluster_clean_level: str = ""
    workloads: Tuple[str, ...] = ()
    primary_sli: Tuple[str, ...] = ()
    slo_targets: Mapping[str, Any] = field(default_factory=dict)
    capacity_policy_id: str = ""
    autoscaling_policy_ids: Tuple[str, ...] = ()
    canary_policy_ids: Tuple[str, ...] = ()
    fault_case_ids: Tuple[str, ...] = ()
    observability_schema_version: str = ""
    planned_repetitions: int = 0
    randomization_policy: str = ""
    telemetry_gaps: Tuple[str, ...] = ()
    limitations: Tuple[str, ...] = ()
    safety_policy_id: str = ""
    allowed_claims: Tuple[str, ...] = ()
    forbidden_claims: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.campaign_id:
            problems.append("campaign_id is required")
        for digest in self.image_index_digests.values():
            if not idn.is_digest(digest):
                problems.append(f"image index digest {digest!r} is not a digest")
        if not self.safety_policy_id:
            problems.append("a campaign must reference a safety policy (§20.2)")
        if not self.allowed_claims:
            problems.append("allowed_claims must be frozen before the run")
        if not self.forbidden_claims:
            problems.append("the non-claims of the campaign must be frozen before the run")
        if not self.primary_sli:
            problems.append("primary SLI list must be frozen before the run")
        if not self.slo_targets:
            problems.append("SLO targets must be frozen before the run (no post-hoc thresholds)")
        if self.planned_repetitions <= 0:
            problems.append("repetitions must be planned before the run")
        return problems

    def payload(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": self.stage,
            "campaign_id": self.campaign_id,
            "created_at": self.created_at,
            "release_ids": list(self.release_ids),
            "image_index_digests": dict(sorted(self.image_index_digests.items())),
            "model_artifact_ids": list(self.model_artifact_ids),
            "cluster_ids": list(self.cluster_ids),
            "cluster_clean_level": self.cluster_clean_level,
            "workloads": list(self.workloads),
            "primary_sli": list(self.primary_sli),
            "slo_targets": dict(sorted(self.slo_targets.items())),
            "capacity_policy_id": self.capacity_policy_id,
            "autoscaling_policy_ids": list(self.autoscaling_policy_ids),
            "canary_policy_ids": list(self.canary_policy_ids),
            "fault_case_ids": list(self.fault_case_ids),
            "observability_schema_version": self.observability_schema_version,
            "planned_repetitions": self.planned_repetitions,
            "randomization_policy": self.randomization_policy,
            "telemetry_gaps": list(self.telemetry_gaps),
            "limitations": list(self.limitations),
            "safety_policy_id": self.safety_policy_id,
            "allowed_claims": list(self.allowed_claims),
            "forbidden_claims": list(self.forbidden_claims),
        }

    def manifest_hash(self) -> str:
        return idn.hash_payload(self.payload())

    def as_dict(self) -> Dict[str, Any]:
        return self.payload()


def campaign_manifest(**kwargs: Any) -> CampaignManifest:
    manifest = CampaignManifest(**kwargs)
    problems = manifest.validate()
    if problems:
        raise ConfigError("invalid campaign manifest: " + "; ".join(problems))
    return manifest


# ── execution safety (§20.2) ──────────────────────────────────────────────


@dataclass
class SafetyPolicy:
    """The authorized scope every destructive experiment must stay inside."""

    safety_policy_id: str
    cluster_id: str = ""
    namespaces: Tuple[str, ...] = ()
    authorized_nodes: Tuple[str, ...] = ()
    authorized_devices: Tuple[str, ...] = ()
    max_targets: int = 1
    max_duration_s: int = 0
    max_replicas: int = 0
    max_cost_units: float = 0.0
    error_budget_cap: float = 0.0
    kill_switch: str = ""
    operator: str = ""
    protected_assets: Tuple[str, ...] = (
        "registry_credentials",
        "model_store_credentials",
        "production_namespaces",
        "other_tenants_data",
    )
    cleanup_required: bool = True
    target_inventory_recorded: bool = False
    fault_runs_separated_from_baseline: bool = False
    abort_on_external_impact: bool = True

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("safety_policy_id", "cluster_id", "kill_switch", "operator"):
            if not getattr(self, name):
                problems.append(f"safety policy requires {name!r}")
        if not self.namespaces:
            problems.append("at least one authorized namespace is required")
        if self.max_targets <= 0:
            problems.append("max_targets must be positive (no unbounded blast radius)")
        if self.max_duration_s <= 0:
            problems.append("max_duration_s must be positive")
        if self.max_replicas <= 0:
            problems.append("max_replicas must be positive")
        if self.error_budget_cap <= 0:
            problems.append("error_budget_cap must be positive (fault runs may not burn unbounded budget)")
        if not self.target_inventory_recorded:
            problems.append(
                "the target inventory must be recorded before destructive faults "
                "(otherwise the blast radius is unknown)"
            )
        if not self.fault_runs_separated_from_baseline:
            problems.append("fault runs must be separated from baseline runs")
        if any("*" in node for node in self.authorized_nodes):
            problems.append("authorized_nodes may not contain wildcards")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "safety_policy_id": self.safety_policy_id,
            "cluster_id": self.cluster_id,
            "namespaces": list(self.namespaces),
            "authorized_nodes": list(self.authorized_nodes),
            "authorized_devices": list(self.authorized_devices),
            "max_targets": self.max_targets,
            "max_duration_s": self.max_duration_s,
            "max_replicas": self.max_replicas,
            "max_cost_units": self.max_cost_units,
            "error_budget_cap": self.error_budget_cap,
            "kill_switch": self.kill_switch,
            "operator": self.operator,
            "protected_assets": list(self.protected_assets),
            "cleanup_required": self.cleanup_required,
            "target_inventory_recorded": self.target_inventory_recorded,
            "fault_runs_separated_from_baseline": self.fault_runs_separated_from_baseline,
            "abort_on_external_impact": self.abort_on_external_impact,
        }


def validate_safety_policy(policy: SafetyPolicy) -> List[str]:
    return policy.validate()


def authorize_targets(
    policy: SafetyPolicy, requested: Sequence[str]
) -> Dict[str, Any]:
    """Resolve requested targets against the authorized inventory.

    An empty/unknown target list is refused: §20.2 forbids destructive actions on
    unresolved selectors, and *silently* shrinking the target set would change the
    experiment.
    """
    problems = policy.validate()
    if problems:
        return {"authorized": False, "resolved": [], "reason": "invalid safety policy: " + "; ".join(problems)}
    if not requested:
        return {"authorized": False, "resolved": [], "reason": "no targets requested"}
    if any("*" in target or "?" in target for target in requested):
        return {"authorized": False, "resolved": [], "reason": "wildcard targets are not permitted"}
    inventory = set(policy.authorized_nodes) | set(policy.authorized_devices) | set(policy.namespaces)
    unknown = sorted(set(requested) - inventory)
    if unknown:
        return {
            "authorized": False,
            "resolved": [],
            "unknown": unknown,
            "reason": "targets outside the authorized inventory: " + ", ".join(unknown),
        }
    if len(requested) > policy.max_targets:
        return {
            "authorized": False,
            "resolved": list(requested),
            "reason": f"{len(requested)} targets exceed the authorized blast radius {policy.max_targets}",
        }
    return {"authorized": True, "resolved": list(requested), "reason": ""}


# ── campaign directory ────────────────────────────────────────────────────


class CampaignDirectory:
    """Materialises the §21 layout under ``experiment_results/S13/<id>/``."""

    def __init__(self, root: str, campaign_id: str) -> None:
        if not campaign_id:
            raise ConfigError("campaign_id must not be empty")
        self.campaign_id = campaign_id
        self.root = root
        self.logical = logical_root(campaign_id)
        self.path = physical_root(root, campaign_id)

    def create(self) -> str:
        os.makedirs(self.path, exist_ok=True)
        for relative in ARTIFACT_LAYOUT:
            os.makedirs(os.path.join(self.path, relative), exist_ok=True)
        return self.path

    def path_for(self, relative: str) -> str:
        target = os.path.abspath(os.path.join(self.path, relative))
        if target != self.path and not target.startswith(self.path + os.sep):
            raise ConfigError(f"refusing to write outside the campaign directory: {relative!r}")
        return target

    def write_json(self, relative: str, payload: Any) -> str:
        import json

        path = self.path_for(relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
        return path

    def write_text(self, relative: str, text: str) -> str:
        path = self.path_for(relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def layout_status(self, *, with_campaign_manifest: bool = False) -> Dict[str, Any]:
        """Which §21 directories exist; missing entries are listed, never ignored."""
        missing = [
            name
            for name in ARTIFACT_LAYOUT
            if not os.path.isdir(os.path.join(self.path, name))
        ]
        root_missing = [
            name
            for name in ROOT_FILES
            if not os.path.isfile(os.path.join(self.path, name))
            and (with_campaign_manifest or name != "campaign_manifest.yaml")
        ]
        return {
            "campaign_id": self.campaign_id,
            "logical_root": self.logical,
            "physical_root": self.path,
            "directories": len(ARTIFACT_LAYOUT),
            "missing_directories": missing,
            "missing_root_files": root_missing,
            "complete": not missing and not root_missing,
        }


def validate_layout(root: str, campaign_id: str) -> List[str]:
    """Report missing logical directories for an existing campaign."""
    base = physical_root(root, campaign_id)
    if not os.path.isdir(base):
        return [f"campaign directory missing: {base}"]
    return [name for name in ARTIFACT_LAYOUT if not os.path.isdir(os.path.join(base, name))]


def report_obligations(*, experiments_executed: Sequence[str]) -> Dict[str, Any]:
    """Which reports the campaign may carry given the executed experiments.

    ``reliability_report.md`` and ``security_supply_chain.md`` may only be written
    once E13-09/E13-11 (and their prerequisites) have actually run, so the driver
    records the obligation instead of shipping an empty file that looks like a
    result.
    """
    required = list(REQUIRED_REPORTS)
    conditional: List[str] = []
    if "E13-09" in experiments_executed:
        conditional.append("reports/reliability_report.md")
    if "E13-01" in experiments_executed or "E13-11" in experiments_executed:
        conditional.append("reports/security_supply_chain.md")
    return {
        "required_always": required,
        "conditional": sorted(set(conditional)),
        "blocked": sorted(set(EXPERIMENT_DEPENDENT_REPORTS) - set(conditional)),
        "note": (
            "experiment-dependent reports are listed as blocked until the corresponding "
            "experiment has raw evidence; an empty report file must not imply a result"
        ),
    }


def expected_evidence_summary(campaign_id: str) -> Dict[str, Any]:
    """Static description of what a complete campaign would contain (no numbers)."""
    return {
        "campaign_id": campaign_id,
        "logical_root": logical_root(campaign_id),
        "physical_root_template": PHYSICAL_ROOT_TEMPLATE,
        "directories": len(ARTIFACT_LAYOUT),
        "tables": len(rec.TABLE_SCHEMAS),
        "evidence_levels": list(rec.EVIDENCE_LEVELS),
    }


def campaign_preregistration_document(manifest: CampaignManifest) -> Dict[str, Any]:
    """The six preregistration documents of §20.1 as one payload map."""
    return {
        "preregistration/questions.yaml": {
            "campaign_id": manifest.campaign_id,
            "primary_sli": list(manifest.primary_sli),
            "allowed_claims": list(manifest.allowed_claims),
            "forbidden_claims": list(manifest.forbidden_claims),
        },
        "preregistration/releases.parquet": {
            "release_ids": list(manifest.release_ids),
            "image_index_digests": dict(sorted(manifest.image_index_digests.items())),
            "model_artifact_ids": list(manifest.model_artifact_ids),
        },
        "preregistration/workloads.parquet": {"workloads": list(manifest.workloads)},
        "preregistration/slos.yaml": {"slo_targets": dict(sorted(manifest.slo_targets.items()))},
        "preregistration/thresholds.yaml": {
            "capacity_policy_id": manifest.capacity_policy_id,
            "autoscaling_policy_ids": list(manifest.autoscaling_policy_ids),
            "canary_policy_ids": list(manifest.canary_policy_ids),
        },
        "preregistration/safety.yaml": {
            "safety_policy_id": manifest.safety_policy_id,
            "fault_case_ids": list(manifest.fault_case_ids),
            "planned_repetitions": manifest.planned_repetitions,
            "randomization_policy": manifest.randomization_policy,
            "telemetry_gaps": list(manifest.telemetry_gaps),
        },
    }


def default_layout_subdirs() -> Tuple[str, ...]:
    """Expose the layout read-only for tests and the interface map."""
    return ARTIFACT_LAYOUT


def stage_summary() -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "layout_directories": len(ARTIFACT_LAYOUT),
        "required_root_files": len(ROOT_FILES),
        "tables": len(rec.TABLE_SCHEMAS),
        "experiments": list(_experiment_ids()),
    }


def _experiment_ids() -> Sequence[str]:
    return tuple(f"E13-{index:02d}" for index in range(1, 12))


def evaluate_fault_injection(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Not defined here.

    The S12 layer owns fault *localization* for evaluation artifacts
    (``hqsb.evaluation.lineage.evaluate_fault_injection``).  S13 keeps its own
    fault-injection contract in :mod:`hqsb.infra.faults`; this module deliberately
    does not shadow that concept, and the symbol exists only to raise a clear
    error if a caller confuses the two layers.
    """
    raise ConfigError(
        "campaign.evaluate_fault_injection does not exist: use hqsb.infra.faults for S13 "
        "fault episodes or hqsb.evaluation.lineage for S12 evaluation-artifact lineage"
    )
