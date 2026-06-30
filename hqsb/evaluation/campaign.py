"""S12 campaign manifest, upstream evidence ledger and status propagation.

A campaign is the unit that E12-10 later has to be able to rebuild: it freezes
the comparison groups, the candidate universe, the environment, and the state
of every upstream artefact it consumes.

Two structures matter:

``UpstreamEvidence``
    each consumed S02–S11 artefact gets one of ``VERIFIED`` /
    ``VERIFIED_WITH_LIMITS`` / ``UNVERIFIED`` / ``MISSING`` / ``INCOMPATIBLE``;
    only the first two may enter a formal comparison.  A missing artefact may
    **not** be replaced by a theoretical value.
``StatusMatrix``
    ``experiment_status`` / ``cell_status`` / ``metric_status`` / ``claim_status``
    are tracked separately so a local failure cannot be flattened into one
    global green tick.

The logical artefact layout of ``details/S12/README.md`` §21 is materialised by
:class:`ArtifactLayout`; it writes under ``experiment_results/S12/<campaign>/``
(the repository convention used by S08/S10/S11) and never into
``docs/stage_experiments`` (the protocol tree is read-only).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.identity import canonical_json, sha256_text, stable_id
from hqsb.evaluation.records import (
    MISSINGNESS_CODES,
    PROTOCOL_STATUSES,
    STATUS_BLOCKED,
    STATUS_NOT_STARTED,
    StatusMatrix,
    UPSTREAM_COMPARABLE_STATES,
    UPSTREAM_EVIDENCE_STATES,
)

STAGE = "S12"

CAMPAIGN_SCHEMA_VERSION = "1.0.0"

#: Logical artefact layout (§21).  The names are the schema responsibilities and
#: must not be collapsed even if the physical paths move.
ARTIFACT_LAYOUT: Tuple[str, ...] = (
    "preregistration",
    "inventory",
    "comparability/contracts",
    "capability/evidence",
    "benchmark/operator",
    "benchmark/model_core",
    "benchmark/service",
    "benchmark/distributed",
    "benchmark/raw",
    "benchmark/normalized",
    "repeatability/telemetry",
    "model/peaks",
    "model/roofline",
    "model/amdahl",
    "model/residuals",
    "energy/calibration",
    "energy/raw_power",
    "energy/aligned_windows",
    "energy/summaries",
    "cost/source_snapshots",
    "cost/assumptions",
    "cost/sensitivity",
    "pareto/profiles",
    "pareto/frontiers",
    "pareto/recommendations",
    "maturity/sessions",
    "maturity/incidents",
    "maturity/rubric",
    "lineage/validation",
    "reports/figures",
    "reports/tables",
    "commands",
    "stdout",
    "stderr",
)

#: Logical layer → physical directory (S12 建議目录结构的物理映射).
LAYER_DIRECTORIES: Mapping[str, str] = {
    "operator": "benchmark/operator",
    "model_core": "benchmark/model_core",
    "service": "benchmark/service",
    "distributed": "benchmark/distributed",
}

RUN_ROOT_FILES: Tuple[str, ...] = (
    "campaign_manifest.json",
    "upstream_evidence.json",
    "status_matrix.json",
    "acceptance.json",
    "limitations.md",
)

ACCEPTANCE_FIELDS: Tuple[str, ...] = (
    "campaign_id",
    "stage",
    "experiment_status",
    "cell_status",
    "metric_status",
    "claim_status",
    "comparable_groups",
    "blocked_items",
    "limitations",
    "claim_scope",
    "written_at",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── upstream evidence ─────────────────────────────────────────────────────


@dataclass
class UpstreamEvidence:
    """One consumed upstream artefact and its admissibility state."""

    upstream_stage: str
    experiment_id: str
    artifact_uri: str
    state: str
    reason: str = ""
    schema_ok: bool = False
    hash_ok: bool = False
    verified_at: str = ""
    limits: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.state not in UPSTREAM_EVIDENCE_STATES:
            problems.append(f"unknown upstream evidence state {self.state!r}")
        if not self.upstream_stage:
            problems.append("upstream_stage is required")
        if self.state in UPSTREAM_COMPARABLE_STATES:
            if not self.artifact_uri:
                problems.append("a usable upstream artefact needs a URI")
            if not self.hash_ok:
                problems.append("a usable upstream artefact needs a verified hash")
            if self.state == "VERIFIED_WITH_LIMITS" and not self.limits:
                problems.append("VERIFIED_WITH_LIMITS must state its limits explicitly")
        if self.state in ("MISSING", "INCOMPATIBLE", "UNVERIFIED") and not self.reason:
            problems.append(f"state {self.state} requires a reason (no silent gaps)")
        return problems

    @property
    def usable(self) -> bool:
        return self.state in UPSTREAM_COMPARABLE_STATES

    def as_dict(self) -> Dict[str, Any]:
        return {
            "upstream_stage": self.upstream_stage,
            "experiment_id": self.experiment_id,
            "artifact_uri": self.artifact_uri,
            "state": self.state,
            "reason": self.reason,
            "schema_ok": self.schema_ok,
            "hash_ok": self.hash_ok,
            "verified_at": self.verified_at,
            "limits": list(self.limits),
        }


def inventory_upstream_evidence(rows: Sequence[UpstreamEvidence]) -> Dict[str, Any]:
    """Aggregate an evidence inventory; unusable entries stay visible."""
    problems: List[str] = []
    usable: List[str] = []
    for row in rows:
        problems.extend(f"{row.upstream_stage}/{row.experiment_id}: {item}" for item in row.validate())
        if row.usable:
            usable.append(f"{row.upstream_stage}/{row.experiment_id}")
    return {
        "total": len(rows),
        "usable": sorted(usable),
        "unusable": sorted(
            f"{row.upstream_stage}/{row.experiment_id}" for row in rows if not row.usable
        ),
        "problems": problems,
        "ok": not problems,
    }


def require_upstream(states: Mapping[str, str], stage: str) -> None:
    """Fail closed when a required stage is not in a comparable state."""
    state = states.get(stage, "MISSING")
    if state not in UPSTREAM_COMPARABLE_STATES:
        raise ConfigError(
            f"upstream stage {stage} is {state}: it may not enter a formal comparison "
            f"(a theoretical value is not a substitute)",
            details={"stage": stage, "state": state},
        )


# ── campaign manifest ─────────────────────────────────────────────────────


@dataclass
class CampaignManifest:
    """Frozen campaign identity (E12-10 step 1)."""

    campaign_id: str
    created_at: str = ""
    schema_version: str = CAMPAIGN_SCHEMA_VERSION
    policy_version: str = "s12_policy_1.0.0"
    git: Mapping[str, Any] = field(default_factory=dict)
    frozen_inputs: Mapping[str, str] = field(default_factory=dict)
    comparison_group_ids: Tuple[str, ...] = ()
    candidate_ids: Tuple[str, ...] = ()
    platform_instance_ids: Tuple[str, ...] = ()
    contract_hashes: Mapping[str, str] = field(default_factory=dict)
    head_documents: Mapping[str, str] = field(default_factory=dict)
    upstream_evidence_summary: Mapping[str, Any] = field(default_factory=dict)
    owners: Mapping[str, str] = field(default_factory=dict)
    limitations: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.campaign_id:
            problems.append("campaign_id is required")
        if not self.frozen_inputs:
            problems.append("a campaign must freeze its inputs (commit/config/model/environment)")
        for group_id, digest in self.contract_hashes.items():
            if len(digest) != 64:
                problems.append(f"contract hash for {group_id} is not a sha256 digest")
        if self.upstream_evidence_summary.get("problems"):
            problems.extend(self.upstream_evidence_summary["problems"])
        return problems

    def payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": STAGE,
            "campaign_id": self.campaign_id,
            "policy_version": self.policy_version,
            "created_at": self.created_at or _utc_now(),
            "git": dict(sorted(self.git.items())),
            "frozen_inputs": dict(sorted(self.frozen_inputs.items())),
            "comparison_group_ids": list(self.comparison_group_ids),
            "candidate_ids": list(self.candidate_ids),
            "platform_instance_ids": list(self.platform_instance_ids),
            "contract_hashes": dict(sorted(self.contract_hashes.items())),
            "head_documents": dict(sorted(self.head_documents.items())),
            "upstream_evidence_summary": self.upstream_evidence_summary,
            "owners": dict(sorted(self.owners.items())),
            "limitations": list(self.limitations),
        }

    def to_json(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, indent=2, ensure_ascii=False)

    def sha256(self) -> str:
        return sha256_text(canonical_json(self.payload()))


def new_campaign_id(seed: Mapping[str, Any]) -> str:
    """Deterministic campaign id from its frozen inputs (never a display name)."""
    return stable_id("s12camp", seed)


# ── status propagation ────────────────────────────────────────────────────


def propagate_experiment_status(
    *,
    scope_id: str,
    scope_kind: str,
    unsatisfied_prerequisites: Sequence[str] = (),
    executed: bool = False,
    raw_samples: int = 0,
    allow_execute: bool = False,
    quality_failed: bool = False,
    stable: Optional[bool] = None,
    lineage_ok: Optional[bool] = None,
    reason: str = "",
) -> StatusMatrix:
    """Derive the four statuses from *facts*, never from optimism.

    Ordering follows ``details/S12/README.md`` §3.2 and §16: evidence validity →
    comparability → quality → stability → lineage.  Every degradation carries a
    reason string so the acceptance report can cite it.
    """
    experiment_status = STATUS_NOT_STARTED
    cell_status = "NOT_RUN"
    metric_status = "MISSING"
    claim_status = "NOT_PUBLISHABLE"
    reasons: List[str] = [reason] if reason else []

    if unsatisfied_prerequisites or not allow_execute:
        experiment_status = STATUS_BLOCKED
        cell_status = "BLOCKED"
        metric_status = "MISSING"
        claim_status = "NOT_PUBLISHABLE"
        if unsatisfied_prerequisites:
            reasons.append("prerequisites unsatisfied: " + ", ".join(sorted(unsatisfied_prerequisites)))
        if not allow_execute:
            reasons.append("execution not allowed by the driver (no conclusion may be emitted)")
    elif not executed or raw_samples <= 0:
        experiment_status = STATUS_BLOCKED
        cell_status = "BLOCKED"
        reasons.append("no raw samples recorded")
    else:
        experiment_status = "RUNNING"
        cell_status = "RAN"
        metric_status = "PRELIMINARY"
        claim_status = "NOT_PUBLISHABLE"
        if quality_failed:
            cell_status = "QUALITY_FAILED"
            metric_status = "INSUFFICIENT"
            reasons.append("quality gate failed: the cell may not enter performance/energy/cost/Pareto")
        if stable is False:
            cell_status = "UNSTABLE" if cell_status == "RAN" else cell_status
            metric_status = "UNSTABLE"
            reasons.append("repeatability gate not met: evidence downgraded, high-confidence claims blocked")
        if lineage_ok is False:
            metric_status = "INVALID"
            claim_status = "NOT_PUBLISHABLE"
            reasons.append("lineage invalid: affected points may not be published even if plausible")
        if cell_status == "RAN" and metric_status == "PRELIMINARY":
            claim_status = "PUBLISHABLE_WITH_LIMITS"
        if stable is True and lineage_ok is True and not quality_failed:
            claim_status = "PUBLISHABLE"

    matrix = StatusMatrix(
        scope_id=scope_id,
        scope_kind=scope_kind,
        experiment_status=experiment_status,
        cell_status=cell_status,
        metric_status=metric_status,
        claim_status=claim_status,
        reason="; ".join(reasons),
    )
    problems = matrix.validate()
    if problems:
        raise ConfigError("invalid status matrix: " + "; ".join(problems))
    return matrix


def matrix_from_mapping(payload: Mapping[str, Any]) -> StatusMatrix:
    return StatusMatrix(
        scope_id=str(payload.get("scope_id", "")),
        scope_kind=str(payload.get("scope_kind", "campaign")),
        experiment_status=str(payload.get("experiment_status", STATUS_NOT_STARTED)),
        cell_status=str(payload.get("cell_status", "NOT_RUN")),
        metric_status=str(payload.get("metric_status", "MISSING")),
        claim_status=str(payload.get("claim_status", "NOT_PUBLISHABLE")),
        reason=str(payload.get("reason", "")),
    )


def status_matrix_payload(matrices: Sequence[StatusMatrix]) -> Dict[str, Any]:
    problems: List[str] = []
    for matrix in matrices:
        problems.extend(f"{matrix.scope_id}: {item}" for item in matrix.validate())
    return {
        "stage": STAGE,
        "rows": [matrix.as_dict() for matrix in matrices],
        "ok": not problems,
        "problems": problems,
    }


# ── acceptance decision ───────────────────────────────────────────────────


@dataclass
class AcceptanceDecision:
    """``decision/acceptance.json`` — never a single global green tick."""

    campaign_id: str
    experiment_status: str = STATUS_NOT_STARTED
    cell_status: str = "NOT_RUN"
    metric_status: str = "MISSING"
    claim_status: str = "NOT_PUBLISHABLE"
    comparable_groups: int = 0
    blocked_items: Tuple[str, ...] = ()
    limitations: Tuple[str, ...] = ()
    claim_scope: Tuple[str, ...] = ()
    written_at: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.campaign_id:
            problems.append("campaign_id is required")
        if self.experiment_status not in PROTOCOL_STATUSES:
            problems.append(f"unknown experiment status {self.experiment_status!r}")
        if self.experiment_status == "PASS" and not self.comparable_groups:
            problems.append("a PASS needs at least one real comparable comparison group")
        if not self.limitations:
            problems.append("limitations must be stated explicitly (never a silent pass)")
        return problems

    def payload(self) -> Dict[str, Any]:
        payload = {
            "campaign_id": self.campaign_id,
            "stage": STAGE,
            "experiment_status": self.experiment_status,
            "cell_status": self.cell_status,
            "metric_status": self.metric_status,
            "claim_status": self.claim_status,
            "comparable_groups": self.comparable_groups,
            "blocked_items": list(self.blocked_items),
            "limitations": list(self.limitations),
            "claim_scope": list(self.claim_scope),
            "written_at": self.written_at or _utc_now(),
        }
        missing = [name for name in ACCEPTANCE_FIELDS if name not in payload]
        if missing:
            raise ConfigError(f"acceptance payload is missing {missing}")
        return payload

    def to_json(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, indent=2, ensure_ascii=False)


# ── missingness policy ────────────────────────────────────────────────────


@dataclass
class MissingnessPolicy:
    """How a missing cell is represented downstream (never ``0``)."""

    policy_id: str = "s12_missingness_1.0.0"
    zero_fill_allowed: bool = False
    allowed_states: Tuple[str, ...] = MISSINGNESS_CODES
    notes: str = (
        "missing is an evidence state, not a number; NOT_APPLICABLE_CAPABILITY and "
        "MEASUREMENT_UNAVAILABLE must stay distinguishable all the way into cost/Pareto"
    )

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.zero_fill_allowed:
            problems.append("zero fill is forbidden for every S12 claim path")
        unknown = sorted(set(self.allowed_states) - set(MISSINGNESS_CODES))
        if unknown:
            problems.append(f"unknown missingness states: {unknown}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "zero_fill_allowed": self.zero_fill_allowed,
            "allowed_states": list(self.allowed_states),
            "notes": self.notes,
        }


# ── artefact layout ───────────────────────────────────────────────────────


class ArtifactLayout:
    """Materialises the campaign's logical layout under a physical root."""

    def __init__(self, root: str, campaign_id: str) -> None:
        if not campaign_id:
            raise ConfigError("campaign_id must not be empty")
        self.campaign_id = campaign_id
        self.path = os.path.abspath(
            os.path.join(root, "experiment_results", STAGE, campaign_id)
        )

    def create(self) -> str:
        os.makedirs(self.path, exist_ok=True)
        for relative in ARTIFACT_LAYOUT:
            os.makedirs(os.path.join(self.path, relative), exist_ok=True)
        return self.path

    def layer_dir(self, layer: str) -> str:
        if layer not in LAYER_DIRECTORIES:
            raise ConfigError(f"unknown layer {layer!r}")
        return os.path.join(self.path, LAYER_DIRECTORIES[layer])

    def path_for(self, relative: str) -> str:
        target = os.path.abspath(os.path.join(self.path, relative))
        if not target.startswith(self.path + os.sep) and target != self.path:
            raise ConfigError(f"refusing to write outside the campaign directory: {relative!r}")
        return target

    def write_json(self, relative: str, payload: Any) -> str:
        path = self.path_for(relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
        return path

    def write_jsonl(self, relative: str, rows: Sequence[Mapping[str, Any]]) -> str:
        path = self.path_for(relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
        return path

    def write_text(self, relative: str, text: str) -> str:
        path = self.path_for(relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def relative_files(self) -> List[str]:
        rows: List[str] = []
        for base, dirs, files in os.walk(self.path):
            dirs.sort()
            for name in sorted(files):
                rows.append(os.path.relpath(os.path.join(base, name), self.path))
        return rows


def required_layout_entries() -> Tuple[str, ...]:
    return ARTIFACT_LAYOUT


def validate_layout(root: str, campaign_id: str) -> List[str]:
    """Report logical directories that do not exist (never silently skip)."""
    base = os.path.join(root, "experiment_results", STAGE, campaign_id)
    if not os.path.isdir(base):
        return [f"campaign directory missing: {base}"]
    return [name for name in ARTIFACT_LAYOUT if not os.path.isdir(os.path.join(base, name))]


def missingness_default(key: str) -> str:
    """Stable default state used by templates (documented, never ad hoc)."""
    mapping = {
        "performance": "NOT_RUN_PREREQUISITE",
        "energy": "MEASUREMENT_UNAVAILABLE",
        "cost": "PRICE_UNAVAILABLE",
        "capability": "NOT_APPLICABLE_CAPABILITY",
        "quality": "QUALITY_GATE_FAILED",
        "lineage": "LINEAGE_INVALID",
    }
    if key not in mapping:
        raise ConfigError(f"unknown missingness category {key!r}")
    return mapping[key]
