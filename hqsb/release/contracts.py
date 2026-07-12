"""The frozen S15 objects: release candidate, evidence bundle, acceptance, claims.

``details/S15/README.md`` §7 freezes three objects before anything else happens,
and §8/§19 add two more that every public artefact depends on:

* :class:`ReleaseCandidateSnapshot` — *which* immutable input this audit is
  about.  Once frozen, every experiment evidence references its id; fixing any
  source, document or evidence file requires a **new** candidate (§7.1);
* :class:`PublicEvidenceBundle` — *what* a third party receives (§7.2), with the
  hard exclusion rules: sensitive data, private models, non-redistributable data,
  host paths and credentials must be excluded and explained by a stub;
* :class:`FinalAcceptanceDecision` — the GO/NO_GO/CONDITIONAL verdict (§7.3),
  which may not bypass the claim ledger (§7.3 note);
* :class:`ClaimRecord` — the controlled data object every document, figure, demo
  caption and resume bullet renders from (§8.1), with the ten qualification
  gates of §10;
* :class:`ContributionRecord` — the human/agent/third-party boundary (§19), so
  "calling vLLM" is never written as "implementing vLLM".

The module also implements the cross-cutting invariants the manual states once
and every experiment must honour: quality before performance (manual §5.2),
no silent degradation (manual §5.7), channel rendering limits (``E15-01`` §11)
and the credibility product rule of §1 (a missing factor lowers the claim, it
does not get averaged away).

Nothing here executes an experiment or produces a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release.identity import (
    CHANNEL_MINIMUM_LEVEL,
    CHANNEL_MIRRORS_CLAIM,
    CLAIM_STATES,
    EVIDENCE_LEVELS,
    PUBLIC_CHANNELS,
    SCHEMA_PREFIX,
    canonical_digest,
    channel_level_ok,
    is_digest,
    level_index,
    required_support,
)
from hqsb.release import records as rec

#: Bundle sections of §7.2, in order.
EVIDENCE_BUNDLE_SECTIONS: Tuple[str, ...] = (
    "identity",
    "protocols",
    "raw",
    "normalized",
    "analysis",
    "figures",
    "reports",
    "release",
    "reproduction",
    "communication",
)

#: Categories that may never be published in an evidence bundle (§7.2).
FORBIDDEN_BUNDLE_CATEGORIES: Tuple[str, ...] = (
    "secret",
    "credential",
    "private_model_weight",
    "non_redistributable_data",
    "host_absolute_path",
    "personal_data",
)

#: Reason codes an automatic selection must record (manual §5.7).
DEGRADATION_REASON_CODES: Tuple[str, ...] = (
    "DEVICE_UNAVAILABLE",
    "DEPENDENCY_MISSING",
    "DEPENDENCY_INCOMPATIBLE",
    "CAPABILITY_UNSUPPORTED",
    "FEATURE_DISABLED_BY_CONFIG",
    "ARTIFACT_MISSING",
    "ARTIFACT_HASH_MISMATCH",
    "LICENSE_WITHHELD",
    "BUDGET_EXCEEDED",
    "REVIEWER_UNAVAILABLE",
)

#: Severity of each claim-gate failure code (``E15-01`` §10).
FAILURE_SEVERITY: Mapping[str, str] = {
    "IDENTITY_INCOMPLETE": "P1",
    "SEMANTIC_AMBIGUOUS": "P1",
    "QUALITY_DISQUALIFIED": "P0",
    "MEASUREMENT_INVALID": "P0",
    "FALLBACK_MIXED": "P0",
    "ORPHAN": "P0",
    "CORRUPTED": "P0",
    "STALE": "P1",
    "ATTRIBUTION_ERROR": "P0",
    "CHANNEL_CONFLICT": "P0",
}

#: P0 failures that block release and resume material unconditionally (§10).
P0_FAILURE_CODES: Tuple[str, ...] = tuple(
    code for code, severity in FAILURE_SEVERITY.items() if severity == "P0"
)


# ── §7.1 ReleaseCandidateSnapshot ────────────────────────────────────────────


@dataclass
class ReleaseCandidateSnapshot:
    """One immutable set of S15 audit inputs (§7.1).

    A candidate is immutable by contract: any repaired source, document or
    evidence file produces a new candidate, and silently overwriting the id is
    exactly what the protocol forbids.
    """

    candidate_id: str
    source_repository: str = ""
    source_commit: str = ""
    tree_dirty: bool = True
    submodules: Tuple[str, ...] = ()
    contract_versions: Mapping[str, str] = field(default_factory=dict)
    upstream_stage_acceptance: Tuple[str, ...] = ()
    claim_ledger_id: str = ""
    evidence_graph_root: str = ""
    release_policy_id: str = ""
    created_at_utc: str = ""
    status: str = rec.STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.release-candidate.v1"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.candidate_id:
            problems.append("ReleaseCandidateSnapshot: candidate_id is required")
        if not self.source_commit or len(self.source_commit) < 40:
            problems.append(
                "ReleaseCandidateSnapshot: a full source commit is required (a tag or short sha is not identity)"
            )
        if self.tree_dirty:
            problems.append("ReleaseCandidateSnapshot: a dirty tree cannot be frozen (E15-05 step 2)")
        missing = [f"C{index}" for index in range(1, 8) if f"C{index}" not in self.contract_versions]
        if missing:
            problems.append(f"ReleaseCandidateSnapshot: contract versions missing for {', '.join(missing)}")
        if not self.claim_ledger_id:
            problems.append("ReleaseCandidateSnapshot: claim_ledger_id is required (rendering reads the ledger)")
        if not self.evidence_graph_root:
            problems.append("ReleaseCandidateSnapshot: evidence_graph_root is required")
        elif not is_digest(self.evidence_graph_root):
            problems.append("ReleaseCandidateSnapshot: evidence_graph_root must be sha256:<hex>")
        if self.status not in rec.ALL_STATUSES:
            problems.append(f"ReleaseCandidateSnapshot: unknown status {self.status!r}")
        return problems

    @property
    def candidate_digest(self) -> str:
        return canonical_digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "tree_dirty": self.tree_dirty,
            "submodules": list(self.submodules),
            "contract_versions": {key: self.contract_versions[key] for key in sorted(self.contract_versions)},
            "upstream_stage_acceptance": list(self.upstream_stage_acceptance),
            "claim_ledger_id": self.claim_ledger_id,
            "evidence_graph_root": self.evidence_graph_root,
            "release_policy_id": self.release_policy_id,
            "created_at_utc": self.created_at_utc,
            "status": self.status,
        }
        if include_digest:
            payload["candidate_digest"] = canonical_digest(payload)
        return payload


def new_candidate(
    previous: Optional[ReleaseCandidateSnapshot], *, candidate_id: str, source_commit: str, **kwargs: Any
) -> ReleaseCandidateSnapshot:
    """Derive a **new** candidate after a fix; the old id is never reused.

    ``E15-09`` step 34 and ``E15-05`` §8 invariant 10 require a fixed problem to
    produce a new candidate (with new digests) rather than an in-place edit, so
    the previous id must differ.
    """
    if previous is not None and previous.candidate_id == candidate_id:
        raise ConfigError(
            "a fix must generate a new release candidate id; reusing "
            f"{candidate_id!r} would overwrite the frozen audit baseline (E15-09 step 34)"
        )
    return ReleaseCandidateSnapshot(candidate_id=candidate_id, source_commit=source_commit, **kwargs)


# ── §7.2 PublicEvidenceBundle ────────────────────────────────────────────────


@dataclass
class BundleEntry:
    """One file of the public evidence bundle (§7.2 ``manifest.json`` row)."""

    path: str
    uri: str
    sha256: str
    size_bytes: int = 0
    media_type: str = ""
    role: str = ""
    license: str = ""
    sensitivity: str = "public"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.path or not self.uri:
            findings.append("BundleEntry: path and uri are required")
        if not is_digest(self.sha256):
            findings.append(f"BundleEntry({self.path}): sha256 must be sha256:<hex>")
        if self.sensitivity not in ("public", "redacted", "withheld"):
            findings.append(f"BundleEntry({self.path}): unknown sensitivity {self.sensitivity!r}")
        if self.sensitivity == "withheld" and self.uri and not self.uri.startswith("stub:"):
            findings.append(
                f"BundleEntry({self.path}): a withheld artefact must be represented by a stub URI "
                "(§7.2: 缺失由公开 stub 解释)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "role": self.role,
            "license": self.license,
            "sensitivity": self.sensitivity,
        }


@dataclass
class PublicEvidenceBundle:
    """The bundle a third party receives (§7.2).

    It is **not** a zip of the local ``reports/`` tree: entries carry a role and a
    license, and anything sensitive/private/non-redistributable is represented by
    a stub with an explanation instead of being shipped.
    """

    bundle_id: str
    candidate_id: str = ""
    entries: Tuple[BundleEntry, ...] = ()
    excluded_categories: Mapping[str, str] = field(default_factory=dict)
    retention_policy: str = ""
    status: str = rec.STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.evidence-bundle.v1"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.bundle_id:
            findings.append("PublicEvidenceBundle: bundle_id is required")
        if not self.entries:
            findings.append("PublicEvidenceBundle: an empty bundle is not a bundle")
        for entry in self.entries:
            findings.extend(entry.problems())
        unknown = [name for name in self.excluded_categories if name not in FORBIDDEN_BUNDLE_CATEGORIES]
        if unknown:
            findings.append(f"PublicEvidenceBundle: unknown exclusion categories {', '.join(sorted(unknown))}")
        # §7.2: every forbidden category that applies must be explained, not silent.
        for category, explanation in self.excluded_categories.items():
            if not explanation or len(explanation) < 10:
                findings.append(
                    f"PublicEvidenceBundle: exclusion {category!r} needs a real explanation "
                    "(the reader must know what is missing and why)"
                )
        present_roles = {entry.role for entry in self.entries}
        for section in ("identity", "raw", "reports"):
            if section not in present_roles:
                findings.append(f"PublicEvidenceBundle: no entry with role {section!r} (bundle incomplete)")
        return findings

    @property
    def manifest_digest(self) -> str:
        return canonical_digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "candidate_id": self.candidate_id,
            "entries": [entry.as_dict() for entry in self.entries],
            "excluded_categories": {key: self.excluded_categories[key] for key in sorted(self.excluded_categories)},
            "retention_policy": self.retention_policy,
            "status": self.status,
        }
        if include_digest:
            payload["manifest_digest"] = canonical_digest(payload)
        return payload


def bundle_from_inventory(
    bundle_id: str,
    candidate_id: str,
    files: Mapping[str, str],
    *,
    roles: Mapping[str, str],
    licenses: Mapping[str, str],
    exclusions: Mapping[str, str] = {},
) -> PublicEvidenceBundle:
    """Build a bundle from a ``{path: sha256}`` inventory with role/license maps.

    A file without a role or a license is refused: an evidence bundle whose files
    have no rights decision is the "compress the local tree" failure of §7.2.
    """
    entries: List[BundleEntry] = []
    for path in sorted(files):
        if path not in roles:
            raise ConfigError(f"bundle entry {path!r} has no role; refusing to bundle an unlabelled file")
        if path not in licenses:
            raise ConfigError(f"bundle entry {path!r} has no license decision (模型数据许可必须单独裁决)")
        entries.append(
            BundleEntry(
                path=path,
                uri=f"bundle://{bundle_id}/{path}",
                sha256=files[path],
                role=roles[path],
                license=licenses[path],
            )
        )
    return PublicEvidenceBundle(
        bundle_id=bundle_id,
        candidate_id=candidate_id,
        entries=tuple(entries),
        excluded_categories=dict(exclusions),
    )


# ── §7.3 FinalAcceptanceDecision ─────────────────────────────────────────────


@dataclass
class FinalAcceptanceDecision:
    """The final GO/NO_GO/CONDITIONAL verdict (§7.3)."""

    candidate_id: str
    experiment_outcomes: Mapping[str, str] = field(default_factory=dict)
    blocking_findings: Tuple[str, ...] = ()
    public_claim_ids: Tuple[str, ...] = ()
    withheld_claim_ids: Tuple[str, ...] = ()
    release_decision: str = "NO_GO"
    resume_decision: str = "NO_GO"
    signed_by: Tuple[str, ...] = ()
    decided_at_utc: str = ""

    schema_version = f"{SCHEMA_PREFIX}.acceptance.v1"

    DECISIONS: Tuple[str, ...] = ("GO", "NO_GO", "CONDITIONAL")

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.candidate_id:
            findings.append("FinalAcceptanceDecision: candidate_id is required")
        unknown = [name for name in self.experiment_outcomes if name not in rec.EXPERIMENT_IDS]
        if unknown:
            findings.append(f"FinalAcceptanceDecision: unknown experiments {', '.join(sorted(unknown))}")
        invalid = [
            name
            for name, status in self.experiment_outcomes.items()
            if status not in rec.ALL_STATUSES
        ]
        if invalid:
            findings.append(f"FinalAcceptanceDecision: invalid outcomes {', '.join(sorted(invalid))}")
        if self.release_decision not in self.DECISIONS:
            findings.append(f"FinalAcceptanceDecision: unknown release_decision {self.release_decision!r}")
        if self.resume_decision not in self.DECISIONS:
            findings.append(f"FinalAcceptanceDecision: unknown resume_decision {self.resume_decision!r}")
        # §25.2: GO requires every P0 experiment to PASS; E15-10 may be the one
        # honest exception (technical preview), which is what CONDITIONAL is for.
        if self.release_decision == "GO":
            not_passed = [
                name
                for name in rec.EXPERIMENT_IDS
                if name != "E15-11" and self.experiment_outcomes.get(name) != rec.STATUS_PASS
            ]
            if not_passed:
                findings.append(
                    "FinalAcceptanceDecision: GO requires every P0 experiment to PASS; not passed: "
                    + ", ".join(not_passed)
                )
            if self.blocking_findings:
                findings.append("FinalAcceptanceDecision: GO with blocking findings is contradictory")
            if not self.signed_by:
                findings.append("FinalAcceptanceDecision: GO must be signed by a human")
        # Both decisions read the same ledger (§7.3 note).
        overlap = sorted(set(self.public_claim_ids) & set(self.withheld_claim_ids))
        if overlap:
            findings.append(
                f"FinalAcceptanceDecision: claims cannot be public and withheld at once: {', '.join(overlap)}"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "experiment_outcomes": {key: self.experiment_outcomes[key] for key in sorted(self.experiment_outcomes)},
            "blocking_findings": list(self.blocking_findings),
            "public_claim_ids": list(self.public_claim_ids),
            "withheld_claim_ids": list(self.withheld_claim_ids),
            "release_decision": self.release_decision,
            "resume_decision": self.resume_decision,
            "signed_by": list(self.signed_by),
            "decided_at_utc": self.decided_at_utc,
        }
        payload["decision_digest"] = canonical_digest(payload)
        return payload


# ── §8 ClaimRecord ───────────────────────────────────────────────────────────


@dataclass
class ClaimScope:
    """The applicability domain of a claim (§8.1 ``scope``)."""

    model_artifact_id: Optional[str] = None
    workload_spec_id: Optional[str] = None
    operator_spec_id: Optional[str] = None
    backend_id: Optional[str] = None
    hardware_ids: Tuple[str, ...] = ()
    precision: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_artifact_id": self.model_artifact_id,
            "workload_spec_id": self.workload_spec_id,
            "operator_spec_id": self.operator_spec_id,
            "backend_id": self.backend_id,
            "hardware_ids": list(self.hardware_ids),
            "precision": self.precision,
        }


@dataclass
class ClaimEffect:
    """The quantitative part of a claim: point, unit, interval, sample unit."""

    point: Optional[float] = None
    unit: Optional[str] = None
    interval: Optional[Tuple[float, float]] = None
    sample_unit: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.point is not None and not self.unit:
            findings.append("ClaimEffect: a point estimate without a unit is not auditable")
        if self.interval is not None:
            low, high = self.interval
            if low > high:
                findings.append("ClaimEffect: interval bounds are inverted")
            if self.point is not None and not (low <= self.point <= high):
                findings.append("ClaimEffect: the point estimate lies outside its own interval")
        if self.point is not None and not self.sample_unit:
            findings.append("ClaimEffect: the sample unit (run/request/token) must be named")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "point": self.point,
            "unit": self.unit,
            "interval": list(self.interval) if self.interval else None,
            "sample_unit": self.sample_unit,
        }


@dataclass
class ClaimRecord:
    """A public claim as a controlled data object (§8.1).

    Documents, figures, demo captions and resume bullets render *from* this
    record; they never keep their own copy of the number (§8.1 closing note).
    """

    claim_id: str
    canonical_text_zh: str = ""
    canonical_text_en: str = ""
    claim_type: str = ""
    evidence_level: str = "PLANNED"
    scope: ClaimScope = field(default_factory=ClaimScope)
    baseline_id: Optional[str] = None
    estimand: Optional[str] = None
    effect: ClaimEffect = field(default_factory=ClaimEffect)
    evidence_refs: Tuple[str, ...] = ()
    limitations: Tuple[str, ...] = ()
    owner: str = ""
    valid_from_commit: str = ""
    invalidated_by: Tuple[str, ...] = ()
    status: str = "DRAFT"
    raw_refs: Tuple[str, ...] = ()
    gate_results: Mapping[str, str] = field(default_factory=dict)
    channels: Tuple[str, ...] = ()

    schema_version = f"{SCHEMA_PREFIX}.claim.v1"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.claim_id.startswith("CLM-"):
            findings.append(f"ClaimRecord: claim_id must start with 'CLM-', got {self.claim_id!r}")
        if self.claim_type not in rec.CLAIM_TYPES:
            findings.append(f"ClaimRecord: unknown claim_type {self.claim_type!r}")
        if self.evidence_level not in EVIDENCE_LEVELS:
            findings.append(f"ClaimRecord: unknown evidence_level {self.evidence_level!r}")
        if self.status not in CLAIM_STATES:
            findings.append(f"ClaimRecord: unknown status {self.status!r}")
        if not self.owner:
            findings.append("ClaimRecord: an owner is required (§8.1)")
        if not self.valid_from_commit:
            findings.append("ClaimRecord: valid_from_commit is required (evidence has a time window)")
        if not self.canonical_text_zh or not self.canonical_text_en:
            findings.append("ClaimRecord: both canonical texts are required (双语渲染同一事实)")
        findings.extend(self.effect.problems())
        # Evidence: URI and digest travel together (E15-01 §3.2).
        if self.status == "VERIFIED":
            if not self.evidence_refs:
                findings.append("ClaimRecord: VERIFIED requires at least one evidence reference")
            if not self.raw_refs and self.effect.point is not None:
                findings.append(
                    "ClaimRecord: a numeric VERIFIED claim must reach raw samples "
                    "(E15-01 §11: 数字必须可点击到 raw)"
                )
            if not self.baseline_id and self.claim_type == "performance":
                findings.append("ClaimRecord: a performance claim needs a baseline identity")
            if self.claim_type == "performance" and not self.estimand:
                findings.append("ClaimRecord: a performance claim needs an estimand")
        if self.effect.point is not None and self.claim_type == "performance" and self.status == "VERIFIED":
            if self.effect.interval is None:
                findings.append("ClaimRecord: a VERIFIED performance claim needs an interval (§5.4)")
        return findings

    def gate_failures(self) -> List[str]:
        """The ten qualification gates of ``E15-01`` §10, as failure codes.

        Every gate is evaluated and **kept**, even when the claim ends up VERIFIED:
        the protocol explicitly forbids deleting the intermediate evidence.  A gate
        that cannot be evaluated from the record's own fields is reported as its
        failure code — "unknown" is not a pass.
        """
        failures: List[str] = []
        scope = self.scope
        # Identity
        if not (scope.model_artifact_id or scope.operator_spec_id or self.claim_type == "attribution"):
            if self.claim_type in ("performance", "quality", "portability", "reliability"):
                failures.append("IDENTITY_INCOMPLETE")
        # Semantics
        if not self.estimand and self.claim_type in ("performance", "quality", "memory", "energy_or_cost", "portability"):
            failures.append("SEMANTIC_AMBIGUOUS")
        # Correctness (quality before performance, manual §5.2)
        correctness = self.gate_results.get("correctness")
        if self.claim_type == "performance" and correctness != "pass":
            failures.append("QUALITY_DISQUALIFIED")
        # Measurement
        if self.claim_type == "performance":
            effect = self.effect
            if effect.point is None or not effect.sample_unit or effect.interval is None:
                failures.append("MEASUREMENT_INVALID")
        # Actual path
        actual = self.gate_results.get("actual_path")
        if self.claim_type in ("performance", "correctness") and actual not in ("pass",):
            failures.append("FALLBACK_MIXED")
        # Evidence / Integrity — URI and digest travel together (E15-01 §3.2)
        for ref in self.evidence_refs:
            marker = ref.find("sha256:")
            if marker < 0:
                failures.append("ORPHAN")
                continue
            digest = ref[marker : marker + len("sha256:") + 64]
            if not is_digest(digest):
                failures.append("CORRUPTED")
        if not self.evidence_refs:
            failures.append("ORPHAN")
        # Freshness
        if self.status == "STALE":
            failures.append("STALE")
        # Attribution
        if not self.owner:
            failures.append("ATTRIBUTION_ERROR")
        # Rendering
        for channel in self.channels:
            if channel not in PUBLIC_CHANNELS:
                failures.append("CHANNEL_CONFLICT")
                continue
            if channel in CHANNEL_MIRRORS_CLAIM:
                continue
            if not channel_level_ok(channel, self.evidence_level, self.evidence_level):
                failures.append("CHANNEL_CONFLICT")
        return sorted(set(failures))

    def severity_failures(self) -> Dict[str, List[str]]:
        """Group the gate failures by severity (§10)."""
        grouped: Dict[str, List[str]] = {}
        for code in self.gate_failures():
            grouped.setdefault(FAILURE_SEVERITY.get(code, "P2"), []).append(code)
        return {level: sorted(codes) for level, codes in sorted(grouped.items())}

    def blocks_release(self) -> bool:
        """A P0 failure blocks release *and* resume material (§10)."""
        return any(code in P0_FAILURE_CODES for code in self.gate_failures())

    def render(self, channel: str) -> Dict[str, Any]:
        """Render the record for a channel without touching the fact fields.

        The channel template may change wording and length; the fact fields are
        copied verbatim from the canonical record (§8.1: 避免同一数字人工复制十次后漂移).
        """
        if channel not in PUBLIC_CHANNELS:
            raise ConfigError(f"unknown public channel {channel!r}")
        if channel in CHANNEL_MIRRORS_CLAIM:
            permitted = self.status == "VERIFIED"
        else:
            permitted = self.status == "VERIFIED" and channel_level_ok(
                self.evidence_level, CHANNEL_MINIMUM_LEVEL[channel]
            )
        return {
            "channel": channel,
            "claim_id": self.claim_id,
            "text_zh": self.canonical_text_zh,
            "text_en": self.canonical_text_en,
            "claim_type": self.claim_type,
            "evidence_level": self.evidence_level,
            "scope": self.scope.as_dict(),
            "baseline_id": self.baseline_id,
            "estimand": self.estimand,
            "effect": self.effect.as_dict(),
            "limitations": list(self.limitations),
            "evidence_refs": list(self.evidence_refs),
            "status": self.status,
            "permitted": permitted,
        }

    def transition(self, new_status: str) -> str:
        """Enforce the state machine of §8.2 and return the new status.

        * ``DRAFT`` → ``VERIFIED`` (gates pass), ``REJECTED`` (unsupported),
          ``RETRACTED`` (abandoned);
        * ``VERIFIED`` → ``STALE`` (dependency/commit/schema/scope change),
          ``RETRACTED`` (contradiction discovered);
        * ``STALE``/``REJECTED``/``RETRACTED`` are terminal for the current
          revision — revalidation happens as a *new* revision, and history is
          never deleted.
        """
        if new_status not in CLAIM_STATES:
            raise ConfigError(f"unknown claim state {new_status!r}")
        allowed: Mapping[str, Tuple[str, ...]] = {
            "DRAFT": ("VERIFIED", "REJECTED", "RETRACTED"),
            "VERIFIED": ("STALE", "RETRACTED"),
            "STALE": ("VERIFIED", "RETRACTED"),
            "REJECTED": ("RETRACTED",),
            "RETRACTED": (),
        }
        if new_status not in allowed[self.status]:
            raise ConfigError(
                f"claim state transition {self.status} → {new_status} is not allowed by §8.2 "
                f"(allowed: {', '.join(allowed[self.status]) or 'none'})"
            )
        if new_status == "VERIFIED":
            failures = self.gate_failures()
            if failures:
                raise ConfigError(
                    "refusing to mark the claim VERIFIED: gates not passed: " + ", ".join(failures)
                )
        self.status = new_status
        return new_status

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "claim_id": self.claim_id,
            "canonical_text_zh": self.canonical_text_zh,
            "canonical_text_en": self.canonical_text_en,
            "claim_type": self.claim_type,
            "evidence_level": self.evidence_level,
            "scope": self.scope.as_dict(),
            "baseline_id": self.baseline_id,
            "estimand": self.estimand,
            "effect": self.effect.as_dict(),
            "evidence_refs": list(self.evidence_refs),
            "limitations": list(self.limitations),
            "owner": self.owner,
            "valid_from_commit": self.valid_from_commit,
            "invalidated_by": list(self.invalidated_by),
            "status": self.status,
            "raw_refs": list(self.raw_refs),
            "gate_results": {key: self.gate_results[key] for key in sorted(self.gate_results)},
            "channels": list(self.channels),
        }
        payload["claim_digest"] = canonical_digest(payload)
        return payload


# ── §19 ContributionRecord ───────────────────────────────────────────────────


@dataclass
class ContributionRecord:
    """Who did what, for one artefact (§19).

    The record exists so that "调用 vLLM" cannot be written as "实现 vLLM" and an
    agent-drafted patch cannot be silently promoted to unreviewed human work.
    """

    artifact_id: str
    roles: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    review_evidence: Tuple[str, ...] = ()
    accepted_by: str = ""
    limitations: Tuple[str, ...] = ()

    schema_version = f"{SCHEMA_PREFIX}.contribution.v1"

    #: Role vocabulary of §19.
    ROLE_GROUPS: Tuple[str, ...] = ("human", "coding_agent", "third_party")
    HUMAN_ROLES: Tuple[str, ...] = (
        "problem_selection",
        "architecture_decision",
        "experiment_design_review",
        "result_interpretation",
    )
    AGENT_ROLES: Tuple[str, ...] = ("draft_generation", "mechanical_refactor", "test_scaffolding")
    THIRD_PARTY_ROLES: Tuple[str, ...] = ("framework_runtime", "profiler", "upstream_kernel")

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.artifact_id:
            findings.append("ContributionRecord: artifact_id is required")
        unknown = [group for group in self.roles if group not in self.ROLE_GROUPS]
        if unknown:
            findings.append(f"ContributionRecord: unknown role groups {', '.join(sorted(unknown))}")
        for group, roles in self.roles.items():
            allowed = {
                "human": self.HUMAN_ROLES,
                "coding_agent": self.AGENT_ROLES,
                "third_party": self.THIRD_PARTY_ROLES,
            }.get(group, ())
            invalid = [role for role in roles if role not in allowed]
            if invalid:
                findings.append(f"ContributionRecord: unknown {group} roles {', '.join(sorted(invalid))}")
        human = self.roles.get("human", ())
        if "result_interpretation" not in human:
            findings.append(
                "ContributionRecord: a human must own result interpretation (§19: 不得把 Agent 产出直接算作个人实现)"
            )
        if not self.accepted_by:
            findings.append("ContributionRecord: accepted_by (a human) is required")
        if "architecture_decision" not in human and "problem_selection" not in human:
            findings.append("ContributionRecord: the human contribution boundary must include problem or design work")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "roles": {group: list(roles) for group, roles in sorted(self.roles.items())},
            "review_evidence": list(self.review_evidence),
            "accepted_by": self.accepted_by,
            "limitations": list(self.limitations),
        }
        payload["record_digest"] = canonical_digest(payload)
        return payload


# ── cross-cutting invariants ─────────────────────────────────────────────────


def check_quality_before_performance(
    *, claim_type: str, correctness_status: str, performance_eligible: bool
) -> List[str]:
    """Manual §5.2: a failed correctness gate stops every performance claim."""
    findings: List[str] = []
    if claim_type == "performance" and performance_eligible and correctness_status != "pass":
        findings.append(
            "performance eligibility claimed while correctness is "
            f"{correctness_status!r}; correctness precedes performance (manual §5.2)"
        )
    return findings


def check_no_silent_degradation(
    *, requested: str, actual: str, reason_code: str, reason: str
) -> List[str]:
    """Manual §5.7: every fallback records requested/actual/reason."""
    findings: List[str] = []
    if requested != actual:
        if not reason_code:
            findings.append("a fallback is declared but no reason code is recorded (manual §5.7)")
        elif reason_code not in DEGRADATION_REASON_CODES:
            findings.append(f"unknown fallback reason code {reason_code!r}")
        if not reason or len(reason) < 8:
            findings.append("a fallback must explain itself in the reason text, not only by a code")
    elif reason_code:
        findings.append("a reason code is recorded although requested == actual; the record is inconsistent")
    return findings


def check_evidence_level_upgrade(
    *, claimed_level: str, cited_levels: Sequence[str]
) -> List[str]:
    """A claim may not cite only lower levels than the one it advertises.

    ``E15-01`` §3.3: "高层必须引用低层所需证据，不能跳级".  Concretely, a
    ``RUNTIME`` claim must cite at least a ``TEST`` reference; a ``MODEL`` claim
    must cite ``RUNTIME``; and so on.
    """
    findings: List[str] = []
    if not cited_levels:
        findings.append("a claim at level above PLANNED must cite evidence; none was supplied")
        return findings
    target = level_index(claimed_level)
    best = max(level_index(level) for level in cited_levels)
    if best < target:
        needed = required_support(claimed_level)
        findings.append(
            f"claimed level {claimed_level} requires supporting evidence up to "
            f"{needed[-1] if needed else 'SOURCE'}; cited levels max out at "
            f"{EVIDENCE_LEVELS[best]}"
        )
    return findings


def check_bilingual_fact_consistency(zh: Mapping[str, Any], en: Mapping[str, Any]) -> List[str]:
    """Structured fact diff of two language renderings (§8.1, ``E15-04`` step 33).

    Only the *fact fields* are compared; navigation/explanation differences are
    allowed.  A differing number, unit, model, hardware, workload, baseline,
    evidence level, status or limitation is a critical diff.
    """
    fact_fields = (
        "claim_id",
        "point",
        "unit",
        "interval",
        "sample_unit",
        "model_artifact_id",
        "hardware_ids",
        "workload_spec_id",
        "baseline_id",
        "evidence_level",
        "status",
        "limitations",
    )
    findings: List[str] = []
    for field_name in fact_fields:
        left = zh.get(field_name)
        right = en.get(field_name)
        if isinstance(left, list):
            left = sorted(map(str, left))
        if isinstance(right, list):
            right = sorted(map(str, right))
        if left != right:
            findings.append(f"bilingual fact diff on {field_name!r}: {left!r} != {right!r}")
    return findings


def check_credibility_factors(present: Mapping[str, bool]) -> Dict[str, Any]:
    """§1: credibility is a product — a missing factor lowers the claim level.

    Returns the list of missing factors plus the highest claim level the present
    factors support.  The mapping is deliberately *not* a score: there is no
    averaging that turns 9/10 into "almost true".
    """
    missing = [factor for factor in rec.CREDIBILITY_FACTORS if not present.get(factor, False)]
    support = "PLANNED"
    if not present.get("claim_truthfulness", False):
        support = "PLANNED"
    elif missing:
        support = "SOURCE"
    else:
        support = "PORTABLE"
    return {
        "missing_factors": missing,
        "supported_level": support,
        "downgrade_required": bool(missing),
        "note": "任一乘项缺失都必须降低公开声明等级（details/S15/README.md §1）",
    }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the contract layer (labelled smoke, not an experiment)."""
    problems: List[str] = []
    candidate = ReleaseCandidateSnapshot(
        candidate_id="cand-1",
        source_commit="0" * 40,
        tree_dirty=False,
        contract_versions={f"C{index}": "1.0.0" for index in range(1, 8)},
        claim_ledger_id="ledger-1",
        evidence_graph_root="sha256:" + "1" * 64,
    )
    problems.extend(candidate.validate())
    decision = FinalAcceptanceDecision(
        candidate_id="cand-1",
        experiment_outcomes={name: rec.STATUS_BLOCKED for name in rec.EXPERIMENT_IDS},
        release_decision="GO",
        resume_decision="NO_GO",
    )
    negative_controls = {
        "go_with_blocked_experiments": decision.problems(),
        "candidate_reuse": [],
        "silent_degradation": check_no_silent_degradation(
            requested="triton", actual="torch_reference", reason_code="", reason=""
        ),
        "level_jump": check_evidence_level_upgrade(claimed_level="SERVICE", cited_levels=["SOURCE"]),
        "bilingual_drift": check_bilingual_fact_consistency({"point": 1.0, "unit": "ms"}, {"point": 1.4, "unit": "ms"}),
    }
    try:
        new_candidate(candidate, candidate_id="cand-1", source_commit="0" * 40)
    except ConfigError as exc:
        negative_controls["candidate_reuse"] = [str(exc)]
    for name, findings in negative_controls.items():
        if not findings:
            problems.append(f"negative control {name!r} was not rejected")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "bundle_sections": len(EVIDENCE_BUNDLE_SECTIONS),
        "credibility_factors": len(rec.CREDIBILITY_FACTORS),
        "p0_failure_codes": list(P0_FAILURE_CODES),
        "negative_controls_rejected": sum(1 for findings in negative_controls.values() if findings),
        "problems": problems,
    }
