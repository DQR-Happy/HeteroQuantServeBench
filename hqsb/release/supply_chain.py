"""E15-05 — tag → release artifacts, provenance, SBOM and licensing.

Protocol: ``docs/stage_experiments/details/S15/E15-05_release_supply_chain_provenance.md``
(45 steps) and ``details/S15/README.md`` §14–§16.

The module models the supply chain as data so that "the release page has
attachments" can never stand in for "third parties can verify where the binaries
came from":

* :class:`ReleasePolicy` — version semantics (SemVer), asset kinds, immutable
  publication, retraction and risk acceptance rules (step 1, §14, §15);
* :class:`BuildDefinition` — workflow/action digests, base image digest,
  toolchain lock, build args and network policy, all pinned (steps 7–9);
* :class:`ArtifactManifest` + :func:`verify_artifact_manifest` — the canonical
  inventory with per-asset role, license, sensitivity and provenance/SBOM refs
  (steps 10, 16, 33);
* :class:`Provenance` + :func:`verify_provenance` — SLSA-style subject/materials/
  builder/parameters, verified **at the consumer** with the eight checks of §9.2
  (steps 17–18, 38);
* :class:`SbomDocument` + :func:`verify_sbom_coverage` — SPDX-ish components with
  the "requirements.txt is not an SBOM" rule and a coverage spot-check against
  actual files/linkage (steps 19–22, 39);
* :class:`VulnerabilityReport` + :func:`dispose_vulnerabilities` — scanner/database
  timestamps, severity, applicability and a disposition per finding, because a
  "0 vulnerabilities" headline ages (steps 23–24, §9.1);
* :class:`LicenseInventory` and :func:`audit_model_data_rights` — source,
  third-party, model, data and media rights are decided **separately** (§16);
* :func:`verify_secret_scan`, :func:`verify_pii_scan`,
  :func:`check_archive_boundary` — steps 31–33 with the canary requirement
  ("scanner returned 0" over an empty scope is INVALID);
* :func:`compare_rebuilds` and :func:`attribute_nondeterminism` — reproducibility
  vs. trustworthiness (§3.3), never conflated;
* :func:`check_immutable_publication`, :func:`retraction_plan`,
  :func:`archive_citation` — steps 41–43;
* :func:`supply_chain_go_no_go` — the per-asset qualification matrix of §9.

Nothing here builds a wheel, signs anything, scans anything or uploads anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release import records as rec
from hqsb.release.identity import SCHEMA_PREFIX, canonical_digest, is_digest

#: Asset kinds the inventory must cover (§9 asset matrix).
ASSET_KINDS: Tuple[str, ...] = rec.ReleaseArtifactRecord.ARTIFACT_TYPES

#: Additional non-package assets that are part of a release (§9).
EXTRA_ASSETS: Tuple[str, ...] = ("provenance", "sbom", "checksums", "notices", "demo_video", "docs_site")

#: Finding dispositions (§9.1).
FINDING_DISPOSITIONS: Tuple[str, ...] = rec.FINDING_DISPOSITIONS

#: License decisions.
LICENSE_DECISIONS: Tuple[str, ...] = ("ALLOW", "DENY", "CONDITIONAL")

#: Vulnerability decisions.
VULNERABILITY_DECISIONS: Tuple[str, ...] = ("PASS", "BLOCK", "ACCEPTED_RISK")

#: Rights that must be decided for every non-source artefact (§16).
RIGHTS_KINDS: Tuple[str, ...] = (
    "download",
    "use",
    "modify",
    "quantize",
    "derive",
    "redistribute",
)

#: Non-determinism sources of step 35.
NONDETERMINISM_SOURCES: Tuple[str, ...] = (
    "timestamp",
    "archive_order",
    "build_id",
    "path",
    "compiler_seed",
    "real_content_difference",
)

#: Provenance consumer checks (§9.2): all eight, none optional.
PROVENANCE_CONSUMER_CHECKS: Tuple[str, ...] = (
    "subject_digest_matches_asset",
    "source_repository_matches_official",
    "source_commit_matches_tag",
    "builder_in_allowlist",
    "build_definition_matches_policy",
    "external_parameters_approved",
    "attestation_signature_valid",
    "verification_hard_fails_on_mismatch",
)

#: The supply-chain gates of §23 / control-plane §28 that a candidate release must run.
RELEASE_GATES: Tuple[str, ...] = (
    "schema_validation",
    "tests",
    "command_smoke",
    "link_check",
    "claim_scan",
    "matrix_diff",
    "regeneration",
    "checksum_manifest",
    "install_verification",
    "sbom_parse",
    "license_policy",
    "secret_scan",
    "vulnerability_scan",
    "provenance_verification",
    "bundle_completeness",
)

#: SemVer shape accepted by the policy (§14).
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


@dataclass
class ReleasePolicy:
    """The frozen release policy of step 1."""

    policy_id: str
    versioning: str = "semver"
    pre_release_allowed: bool = True
    asset_kinds: Tuple[str, ...] = ASSET_KINDS
    immutable_assets: bool = True
    retraction_requires_new_version: bool = True
    model_weights_redistributed: bool = False
    signing: str = ""
    retention_policy: str = ""
    risk_acceptance_requires: Tuple[str, ...] = ("owner", "reason", "scope", "remediation", "expiry")

    schema_version = f"{SCHEMA_PREFIX}.release-policy.v1"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.policy_id:
            findings.append("ReleasePolicy: policy_id is required")
        if self.versioning != "semver":
            findings.append(f"ReleasePolicy: unknown versioning scheme {self.versioning!r}")
        if not self.immutable_assets:
            findings.append("ReleasePolicy: published assets must be immutable (§14)")
        if not self.retraction_requires_new_version:
            findings.append("ReleasePolicy: a fix must be a new version, never an in-place replacement")
        if self.model_weights_redistributed:
            findings.append("ReleasePolicy: model weights must not be redistributed by default (§16)")
        for kind in self.asset_kinds:
            if kind not in ASSET_KINDS:
                findings.append(f"ReleasePolicy: unknown asset kind {kind!r}")
        for name in ("owner", "expiry"):
            if name not in self.risk_acceptance_requires:
                findings.append(f"ReleasePolicy: risk acceptance must require {name!r}")
        return findings


def validate_version(*, version: str, change_kind: str, policy: ReleasePolicy) -> List[str]:
    """SemVer semantics of step 4 (the number must mean something)."""
    findings: List[str] = []
    if not _SEMVER.match(version):
        findings.append(f"version {version!r} is not a SemVer 2.0.0 string")
    major, minor, patch = (int(part) for part in version.split("-")[0].split("+")[0].split("."))
    if major == 0 and policy.pre_release_allowed and "-" not in version:
        findings.append("a 0.y.z release should say whether it is a pre-release (public API is not yet stable)")
    if change_kind == "breaking" and not (major > 0 or "-" in version):
        findings.append("a breaking change on a released major version requires a major bump or an explicit 0.y.z warning")
    if change_kind == "feature" and patch != 0:
        findings.append("a backwards-compatible feature should bump the minor version, not the patch")
    if change_kind == "docs_only" and (major, minor, patch) == (0, 0, 0):
        findings.append("a docs-only change still needs a version")
    return findings


@dataclass
class BuildDefinition:
    """Pinned build definition (steps 7–9)."""

    definition_id: str
    workflow: str = ""
    action_digests: Tuple[str, ...] = ()
    base_image_digest: str = ""
    toolchain_lock: str = ""
    build_args: Mapping[str, str] = field(default_factory=dict)
    network_policy: str = ""
    builder_id: str = ""
    permissions: Mapping[str, str] = field(default_factory=dict)
    isolated_builder: bool = False
    author_home_mounted: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.workflow or not self.builder_id:
            findings.append("BuildDefinition: workflow and builder_id are required")
        for digest in self.action_digests:
            if not is_digest(digest):
                findings.append(f"BuildDefinition: action pin {digest!r} is not an immutable digest (no @main)")
        if not self.base_image_digest or not is_digest(self.base_image_digest):
            findings.append("BuildDefinition: the base image must be pinned by digest, not by tag")
        if not self.toolchain_lock:
            findings.append("BuildDefinition: the toolchain lock is required")
        if not self.isolated_builder:
            findings.append("BuildDefinition: builds must run in an isolated/hosted builder (step 9)")
        if self.author_home_mounted:
            findings.append("BuildDefinition: the author's home must not be mounted into the builder")
        for scope in self.permissions:
            if scope not in ("contents", "packages", "id-token", "attestations", "actions"):
                findings.append(f"BuildDefinition: unknown permission scope {scope!r}")
        if self.permissions.get("contents") in ("write", "write-all") and not self.permissions.get("id-token"):
            findings.append("BuildDefinition: write permissions without OIDC identity are not least-privilege")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "definition_id": self.definition_id,
            "workflow": self.workflow,
            "action_digests": list(self.action_digests),
            "base_image_digest": self.base_image_digest,
            "toolchain_lock": self.toolchain_lock,
            "build_args": {key: self.build_args[key] for key in sorted(self.build_args)},
            "network_policy": self.network_policy,
            "builder_id": self.builder_id,
            "permissions": {key: self.permissions[key] for key in sorted(self.permissions)},
            "isolated_builder": self.isolated_builder,
            "author_home_mounted": self.author_home_mounted,
        }


@dataclass
class AssetRecord:
    """One release asset row (steps 10, 16)."""

    logical_name: str
    asset_type: str
    version: str
    media_type: str
    size_bytes: int
    sha256: str
    role: str = ""
    platform: str = ""
    license_decision: str = "ALLOW"
    sensitivity: str = "public"
    provenance_ref: str = ""
    sbom_ref: str = ""
    attestation_ref: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("logical_name", "asset_type", "version", "media_type", "role"):
            if not getattr(self, name):
                findings.append(f"AssetRecord: {name} is required for every asset")
        if self.asset_type not in ASSET_KINDS + EXTRA_ASSETS:
            findings.append(f"AssetRecord({self.logical_name}): unknown asset type {self.asset_type!r}")
        if self.size_bytes < 0:
            findings.append(f"AssetRecord({self.logical_name}): negative size")
        if not is_digest(self.sha256):
            findings.append(f"AssetRecord({self.logical_name}): sha256 must be sha256:<hex>")
        if self.license_decision not in LICENSE_DECISIONS:
            findings.append(f"AssetRecord({self.logical_name}): unknown license decision {self.license_decision!r}")
        if self.sensitivity not in ("public", "redacted", "withheld"):
            findings.append(f"AssetRecord({self.logical_name}): unknown sensitivity {self.sensitivity!r}")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "asset_type": self.asset_type,
            "version": self.version,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "role": self.role,
            "platform": self.platform,
            "license_decision": self.license_decision,
            "sensitivity": self.sensitivity,
            "provenance_ref": self.provenance_ref,
            "sbom_ref": self.sbom_ref,
            "attestation_ref": self.attestation_ref,
        }


class ArtifactManifest:
    """The canonical, authoritative asset inventory (step 16)."""

    def __init__(self, *, release_version: str, tag: str, source_commit: str, assets: Sequence[AssetRecord]) -> None:
        self.release_version = release_version
        self.tag = tag
        self.source_commit = source_commit
        self.assets = list(assets)

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.source_commit or len(self.source_commit) < 40:
            findings.append("ArtifactManifest: a full source commit is required")
        if not self.release_version or not self.tag:
            findings.append("ArtifactManifest: release version and tag are required")
        if not self.assets:
            findings.append("ArtifactManifest: an empty inventory is not an inventory")
        for asset in self.assets:
            findings.extend(asset.problems())
        names = [asset.logical_name for asset in self.assets]
        if len(set(names)) != len(names):
            findings.append("ArtifactManifest: duplicate asset names")
        digests = [asset.sha256 for asset in self.assets]
        if len(set(digests)) != len(digests):
            findings.append("ArtifactManifest: two assets share a digest (a checksum file cannot identify them)")
        return findings

    @property
    def manifest_digest(self) -> str:
        return canonical_digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": f"{SCHEMA_PREFIX}.artifact-manifest.v1",
            "release_version": self.release_version,
            "tag": self.tag,
            "source_commit": self.source_commit,
            "assets": [asset.as_dict() for asset in self.assets],
        }
        if include_digest:
            payload["manifest_digest"] = canonical_digest(payload)
        return payload


def verify_artifact_manifest(manifest: ArtifactManifest, *, rehash: Mapping[str, str]) -> List[str]:
    """Consumer-side per-file digest verification (steps 16, 36)."""
    findings = list(manifest.problems())
    for asset in manifest.assets:
        actual = rehash.get(asset.logical_name)
        if actual is None:
            findings.append(f"{asset.logical_name}: not downloadable at the declared location")
        elif actual != asset.sha256:
            findings.append(f"{asset.logical_name}: digest mismatch ({actual} != {asset.sha256})")
    return findings


@dataclass
class Provenance:
    """SLSA-style build provenance (step 17)."""

    provenance_id: str
    subject_digest: str
    source_repository: str = ""
    source_commit: str = ""
    builder_id: str = ""
    build_definition_digest: str = ""
    external_parameters: Mapping[str, Any] = field(default_factory=dict)
    internal_parameters: Mapping[str, Any] = field(default_factory=dict)
    started_at_utc: str = ""
    finished_at_utc: str = ""
    slsa_level_claimed: str = ""
    slsa_requirements_checked: Tuple[str, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.provenance_id or not self.builder_id:
            findings.append("Provenance: provenance_id and builder_id are required")
        if not is_digest(self.subject_digest):
            findings.append("Provenance: the subject must be the asset digest, not a file name")
        if not self.source_commit or len(self.source_commit) < 40:
            findings.append("Provenance: the source commit must be the full SHA the tag points to")
        if not self.build_definition_digest or not is_digest(self.build_definition_digest):
            findings.append("Provenance: the build definition must be bound by digest")
        if self.slsa_level_claimed and not self.slsa_requirements_checked:
            findings.append(
                "Provenance: claiming a SLSA level without checking its requirements is exactly the "
                "'certification by assertion' the protocol forbids"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_PREFIX}.provenance.v1",
            "provenance_id": self.provenance_id,
            "subject_digest": self.subject_digest,
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "builder_id": self.builder_id,
            "build_definition_digest": self.build_definition_digest,
            "external_parameters": {key: self.external_parameters[key] for key in sorted(self.external_parameters)},
            "internal_parameters": {key: self.internal_parameters[key] for key in sorted(self.internal_parameters)},
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "slsa_level_claimed": self.slsa_level_claimed,
            "slsa_requirements_checked": list(self.slsa_requirements_checked),
        }


def verify_provenance(
    provenance: Provenance,
    *,
    consumer: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """The eight consumer checks of §9.2; a failure is a hard fail, not a warning."""
    checks: Dict[str, bool] = {}
    checks["subject_digest_matches_asset"] = consumer.get("asset_digest") == provenance.subject_digest
    checks["source_repository_matches_official"] = consumer.get("official_repository") == provenance.source_repository
    checks["source_commit_matches_tag"] = consumer.get("tag_commit") == provenance.source_commit
    checks["builder_in_allowlist"] = provenance.builder_id in tuple(policy.get("builder_allowlist", ()))
    checks["build_definition_matches_policy"] = (
        provenance.build_definition_digest == consumer.get("expected_build_definition_digest")
    )
    approved_sources = set(policy.get("approved_parameter_sources", ()))
    checks["external_parameters_approved"] = all(
        str(source) in approved_sources for source in provenance.external_parameters.values()
    )
    checks["attestation_signature_valid"] = bool(consumer.get("attestation_signature_valid", False))
    checks["verification_hard_fails_on_mismatch"] = bool(consumer.get("hard_fail_on_mismatch", True))
    failed = sorted(name for name, ok in checks.items() if not ok)
    return {
        "checks": checks,
        "failed": failed,
        "ok": not failed,
        "note": "验证失败是 hard fail，不退回“checksum 对就行”；provenance 也不是“没有漏洞”的保证",
    }


@dataclass
class SbomComponent:
    """One SBOM component (steps 19–21)."""

    name: str
    version: str
    purl: str = ""
    sha256: str = ""
    relationship: str = "DEPENDS_ON"
    scope: str = "runtime"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.name or not self.version:
            findings.append("SbomComponent: name and version are required")
        if not self.purl:
            findings.append(f"SbomComponent({self.name}): a purl (or equivalent) is required to identify the package")
        if self.scope not in ("runtime", "build", "test", "system", "native"):
            findings.append(f"SbomComponent({self.name}): unknown scope {self.scope!r}")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "purl": self.purl,
            "sha256": self.sha256,
            "relationship": self.relationship,
            "scope": self.scope,
        }


@dataclass
class SbomDocument:
    """SPDX-ish SBOM with declared coverage (steps 19–22)."""

    asset_id: str
    format: str = "spdx-json"
    components: Tuple[SbomComponent, ...] = ()
    covers: Tuple[str, ...] = ()
    generated_at_utc: str = ""
    scanner: str = ""
    scanner_version: str = ""

    COVERAGE_KINDS: Tuple[str, ...] = ("direct", "transitive", "system", "native")

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.asset_id:
            findings.append("SbomDocument: asset_id is required (one SBOM per asset digest)")
        if not self.components:
            findings.append("SbomDocument: an empty SBOM is not coverage")
        for component in self.components:
            findings.extend(component.problems())
        for kind in self.covers:
            if kind not in self.COVERAGE_KINDS:
                findings.append(f"SbomDocument: unknown coverage kind {kind!r}")
        if not self.scanner or not self.scanner_version:
            findings.append("SbomDocument: the scanner identity/version must be recorded")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_PREFIX}.sbom.v1",
            "asset_id": self.asset_id,
            "format": self.format,
            "components": [component.as_dict() for component in self.components],
            "covers": list(self.covers),
            "generated_at_utc": self.generated_at_utc,
            "scanner": self.scanner,
            "scanner_version": self.scanner_version,
        }


def verify_sbom_coverage(sbom: SbomDocument, *, spot_check: Mapping[str, Sequence[str]]) -> List[str]:
    """Format *and* coverage: parse plus a spot-check against actual files/linkage (step 22)."""
    findings = list(sbom.problems())
    listed = {component.name for component in sbom.components}
    for kind, expected in sorted(spot_check.items()):
        missing = [name for name in expected if name not in listed]
        if missing:
            findings.append(f"SBOM coverage misses {kind} components: {', '.join(missing)}")
    if sbom.format == "requirements.txt":
        findings.append("requirements.txt is not an SBOM (no relationships, no hashes, no system components)")
    return findings


@dataclass
class VulnerabilityReport:
    """Scanner output with timestamps and dispositions (steps 23–24)."""

    scanner: str
    scanner_version: str
    database_timestamp_utc: str
    findings: Tuple[Mapping[str, Any], ...] = ()
    scanned_scope: Tuple[str, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.scanner or not self.scanner_version or not self.database_timestamp_utc:
            findings.append(
                "VulnerabilityReport: scanner identity, version and database timestamp are required "
                "(a scan is a time-dependent observation)"
            )
        if not self.scanned_scope:
            findings.append(
                "VulnerabilityReport: an empty scope with zero findings is INVALID, not a clean result "
                "(manual §23 closing note)"
            )
        for item in self.findings:
            for name in ("cve", "severity", "component", "applicability", "disposition"):
                if name not in item:
                    findings.append(f"VulnerabilityReport: finding {item.get('cve', '?')} lacks {name!r}")
                    break
            if item.get("disposition") == "ACCEPTED_RISK" and not all(
                item.get(name) for name in ("owner", "reason", "scope", "remediation", "expiry")
            ):
                findings.append(
                    f"VulnerabilityReport: accepted risk on {item.get('cve', '?')} needs owner/reason/scope/"
                    "remediation/expiry"
                )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scanner": self.scanner,
            "scanner_version": self.scanner_version,
            "database_timestamp_utc": self.database_timestamp_utc,
            "scanned_scope": list(self.scanned_scope),
            "findings": [dict(item) for item in self.findings],
        }


def dispose_vulnerabilities(report: VulnerabilityReport) -> Dict[str, Any]:
    """Summarise findings by severity and disposition (step 24)."""
    findings = list(report.problems())
    by_severity: Dict[str, int] = {}
    by_disposition: Dict[str, int] = {}
    blocking: List[str] = []
    for item in report.findings:
        severity = str(item.get("severity", "unknown"))
        by_severity[severity] = by_severity.get(severity, 0) + 1
        disposition = str(item.get("disposition", "OPEN"))
        by_disposition[disposition] = by_disposition.get(disposition, 0) + 1
        if severity in ("critical", "high") and disposition in ("OPEN", "BLOCK"):
            blocking.append(str(item.get("cve", "?")))
    return {
        "findings": findings,
        "by_severity": {key: by_severity[key] for key in sorted(by_severity)},
        "by_disposition": {key: by_disposition[key] for key in sorted(by_disposition)},
        "blocking": sorted(blocking),
        "ok": not findings and not blocking,
    }


# ── licensing (steps 25–30) ──────────────────────────────────────────────────


@dataclass
class LicenseInventory:
    """Source / third-party / per-file license audit (steps 25–27)."""

    root_license: str = ""
    package_metadata_license: str = ""
    spdx_expression: str = ""
    copyright_owner: str = ""
    per_file_covered: bool = False
    per_file_exceptions: Tuple[str, ...] = ()
    vendored_components: Tuple[Mapping[str, str], ...] = ()
    notices_path: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.root_license:
            findings.append("LicenseInventory: the root LICENSE is required")
        if not self.package_metadata_license:
            findings.append("LicenseInventory: the package metadata license is required")
        if self.package_metadata_license and self.root_license:
            if self.package_metadata_license.strip().lower() not in self.root_license.strip().lower():
                findings.append(
                    "LicenseInventory: README/LICENSE and package metadata disagree on the license "
                    "(E15-05 step 25)"
                )
        if not self.copyright_owner:
            findings.append("LicenseInventory: the copyright owner must be named")
        if not self.per_file_covered:
            findings.append(
                "LicenseInventory: a root LICENSE does not automatically cover vendored third-party code; "
                "per-file coverage (REUSE-style or equivalent) is required (step 26)"
            )
        for component in self.vendored_components:
            for name in ("name", "source_url", "version", "license", "modifications"):
                if not component.get(name):
                    findings.append(f"LicenseInventory: vendored component {component.get('name', '?')} lacks {name!r}")
        if self.vendored_components and not self.notices_path:
            findings.append("LicenseInventory: THIRD_PARTY_NOTICES is required when vendored code exists")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "root_license": self.root_license,
            "package_metadata_license": self.package_metadata_license,
            "spdx_expression": self.spdx_expression,
            "copyright_owner": self.copyright_owner,
            "per_file_covered": self.per_file_covered,
            "per_file_exceptions": list(self.per_file_exceptions),
            "vendored_components": [dict(item) for item in self.vendored_components],
            "notices_path": self.notices_path,
        }


@dataclass
class RightsDecision:
    """A per-artifact rights decision (steps 28–30)."""

    artifact_id: str
    artifact_kind: str
    rights: Mapping[str, bool] = field(default_factory=dict)
    conditions: str = ""
    redistribution_allowed: bool = False
    acquisition_steps: str = ""
    manifest_only: bool = False

    KINDS: Tuple[str, ...] = ("model", "tokenizer", "chat_template", "adapter", "quantized_weights", "dataset", "prompt_set", "image", "font", "recording")

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.artifact_kind not in self.KINDS:
            findings.append(f"RightsDecision: unknown artifact kind {self.artifact_kind!r}")
        missing = [name for name in RIGHTS_KINDS if name not in self.rights]
        if missing:
            findings.append(f"RightsDecision({self.artifact_id}): rights not decided for {', '.join(missing)}")
        if not self.redistribution_allowed and not self.manifest_only:
            findings.append(
                f"RightsDecision({self.artifact_id}): redistribution is not allowed, so only a manifest/hash/"
                "acquisition path may be published (never the file itself)"
            )
        if self.manifest_only and not self.acquisition_steps:
            findings.append(f"RightsDecision({self.artifact_id}): a manifest-only artifact needs acquisition steps")
        if self.redistribution_allowed and self.artifact_kind in ("model", "quantized_weights", "adapter"):
            findings.append(
                f"RightsDecision({self.artifact_id}): weights are marked redistributable; that requires an "
                "explicit licence review, not a default"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "rights": {key: bool(self.rights.get(key, False)) for key in RIGHTS_KINDS},
            "conditions": self.conditions,
            "redistribution_allowed": self.redistribution_allowed,
            "acquisition_steps": self.acquisition_steps,
            "manifest_only": self.manifest_only,
        }


def audit_model_data_rights(decisions: Sequence[RightsDecision]) -> Dict[str, Any]:
    """Rights are decided per artifact, and model/data licenses are separate from code (§16)."""
    problems: List[str] = []
    for decision in decisions:
        problems.extend(decision.problems())
    if not decisions:
        problems.append("no rights decisions were made; 'downloadable' is not 'redistributable' (§16)")
    return {
        "decisions": [decision.as_dict() for decision in decisions],
        "problems": problems,
        "ok": not problems,
    }


# ── secrets, PII, archive boundary (steps 31–33) ─────────────────────────────

#: Scan scopes the secret scan must cover (step 31).
SECRET_SCAN_SCOPES: Tuple[str, ...] = (
    "git_history",
    "tag_tree",
    "build_context",
    "container_layers",
    "logs",
    "sbom",
    "docs",
    "release_bundle",
)


def verify_secret_scan(
    *, scope: Sequence[str], findings: Sequence[Mapping[str, Any]], canary_injected: bool, canary_detected: bool
) -> List[str]:
    """Secret scan *with* a canary: detection power must be proven (step 31)."""
    problems: List[str] = []
    for required in SECRET_SCAN_SCOPES:
        if required not in scope:
            problems.append(f"secret scan does not cover {required!r}; a partial scan is not a clean result")
    if not canary_injected:
        problems.append("no canary secret was injected; the scan's detection power is unproven")
    elif not canary_detected:
        problems.append("the injected canary secret was not detected; the scan result cannot be trusted")
    for item in findings:
        if str(item.get("severity", "")).upper() == "P0":
            problems.append(
                f"P0 secret finding {item.get('location', '?')} must be removed and rotated; risk acceptance "
                "is not available for secrets (E15-05 §9.1)"
            )
    return problems


def verify_pii_scan(*, texts: Mapping[str, str], reviewed: bool) -> List[str]:
    """PII / path / internal-host scan (step 32)."""
    patterns = {
        "absolute_home": r"(?:/root/|/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/)",
        "ip_address": r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+\.\d+\b",
        "device_serial": r"(?i)serial[-_ ]?number\s*[:=]\s*\S+",
        "personal_prompt": r"(?i)(?:my name is|我叫|my email is)\s*\S+",
    }
    problems: List[str] = []
    for name, text in sorted(texts.items()):
        for kind, pattern in patterns.items():
            if re.search(pattern, text):
                problems.append(f"{name}: possible {kind}; adjudicate before publishing")
    if problems and not reviewed:
        problems.append("PII/path findings must be adjudicated by a human; an unreviewed hit blocks the release")
    return problems


def check_archive_boundary(*, members: Sequence[str], forbidden: Sequence[str] = ()) -> List[str]:
    """What a wheel/sdist/bundle/image must not contain (step 33)."""
    defaults = (
        ".git/",
        "__pycache__/",
        ".pytest_cache/",
        "core.",
        "id_rsa",
        ".env",
        ".safetensors",
        ".whl",
        ".nsys-rep",
        ".ncu-rep",
    )
    patterns = tuple(forbidden) + defaults
    problems: List[str] = []
    for member in members:
        lowered = member.lower()
        for pattern in patterns:
            if pattern.lower() in lowered:
                problems.append(
                    f"{member}: matches forbidden archive pattern {pattern!r} "
                    "(.gitignore is not a packaging filter)"
                )
    if not members:
        problems.append("archive boundary check was given no members; an empty listing proves nothing")
    return problems


# ── reproducibility and consumer verification (steps 34–40) ─────────────────


def compare_rebuilds(first: Mapping[str, str], second: Mapping[str, str]) -> Dict[str, Any]:
    """Two builds: byte-identical or not, per artifact (step 34)."""
    results: Dict[str, str] = {}
    for name in sorted(set(first) | set(second)):
        left = first.get(name)
        right = second.get(name)
        if left is None or right is None:
            results[name] = "missing_in_one_build"
        elif left == right:
            results[name] = "byte_identical"
        else:
            results[name] = "differs"
    identical = [name for name, verdict in results.items() if verdict == "byte_identical"]
    return {
        "results": results,
        "byte_identical": sorted(identical),
        "reproducible_level": (
            "byte_reproducible" if identical and len(identical) == len(results) else "content_reproducible_only"
        ),
        "note": "可复现构建与可信构建是正交性质；不得用其中一个替代另一个（E15-05 §3.3）",
    }


def attribute_nondeterminism(*, differing: Sequence[str], sources: Mapping[str, str]) -> List[str]:
    """Attribute each difference to a registered source (step 35)."""
    problems: List[str] = []
    for name in differing:
        source = sources.get(name)
        if not source:
            problems.append(f"{name}: difference not attributed to any source; 'the build tool' is not an explanation")
            continue
        if source not in NONDETERMINISM_SOURCES:
            problems.append(f"{name}: unknown non-determinism source {source!r}")
        elif source == "real_content_difference":
            problems.append(
                f"{name}: the difference is a real content difference, which invalidates the reproducibility "
                "claim rather than explaining it"
            )
    return problems


def consumer_verify(*, kind: str, steps: Mapping[str, bool]) -> List[str]:
    """Consumer verification per artifact kind (steps 36–40)."""
    required: Mapping[str, Tuple[str, ...]] = {
        "wheel": ("install", "import", "sample", "upgrade", "version_identity"),
        "sdist": ("install", "import", "sample", "version_identity"),
        "container": ("pull_by_digest", "non_root", "read_only_boundary", "health", "sample"),
        "provenance": PROVENANCE_CONSUMER_CHECKS,
        "sbom_notice": ("downloadable", "parsable", "maps_to_asset_digest"),
        "release_notes": ("version_matches", "install_command_matches", "checksums_match", "claims_match_ledger"),
    }
    if kind not in required:
        raise ConfigError(f"unknown consumer verification kind {kind!r}; known: {', '.join(sorted(required))}")
    problems = [f"{kind}: consumer verification step {name!r} was not performed" for name in required[kind] if not steps.get(name)]
    return problems


def check_immutable_publication(*, draft_first: bool, replaced_assets: Sequence[str]) -> List[str]:
    """Draft → upload → verify → publish, and never replace a published asset (step 41)."""
    problems: List[str] = []
    if not draft_first:
        problems.append("the release was published without a draft/verify step")
    for asset in replaced_assets:
        problems.append(
            f"{asset}: a published asset was replaced in place; a fix requires a new version and a "
            "retraction note (E15-05 §8 invariant 10)"
        )
    return problems


def retraction_plan(*, reason: str, kind: str, advisory: str, new_version: str, audit_record: str) -> List[str]:
    """The retraction path: NO-GO, advisory, patch release, audit record (step 42)."""
    problems: List[str] = []
    if kind not in ("digest_mismatch", "secret", "license", "cve", "quality", "identity"):
        problems.append(f"unknown retraction trigger {kind!r}")
    if not reason:
        problems.append("a retraction needs a reason that a reader can verify")
    if not advisory:
        problems.append("a retraction needs a public advisory (deleting the asset silently is not a retraction)")
    if not new_version:
        problems.append("a retraction needs a fixed version to point users at")
    if not audit_record:
        problems.append("a retraction needs an audit record; attachments must not simply disappear")
    return problems


@dataclass
class ArchiveCitation:
    """Long-term citation metadata (step 43)."""

    citation_file: str = "CITATION.cff"
    version: str = ""
    authors: Tuple[str, ...] = ()
    license: str = ""
    source_url: str = ""
    release_url: str = ""
    archive_uri: str = ""
    doi: str = ""
    adr_id: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("version", "license", "source_url", "release_url"):
            if not getattr(self, name):
                findings.append(f"ArchiveCitation: {name} is required")
        if not self.authors:
            findings.append("ArchiveCitation: authors/contributors are required")
        if self.archive_uri and not self.adr_id:
            findings.append("ArchiveCitation: the permanent-URI choice must be recorded in an ADR (E15-05 step 43)")
        if "main" in self.source_url or "latest" in self.release_url:
            findings.append("ArchiveCitation: a permanent record may not point at a floating main/latest")
        return findings


def supply_chain_go_no_go(
    *,
    manifest: ArtifactManifest,
    build_definition: BuildDefinition,
    provenance_ok: bool,
    sbom_findings: Sequence[str],
    secret_findings: Sequence[str],
    license_findings: Sequence[str],
    rights_findings: Sequence[str],
    vulnerability: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """The per-asset release qualification matrix of §9, as a verdict."""
    blocking: List[str] = []
    if manifest.problems():
        blocking.extend(manifest.problems())
    if build_definition.problems():
        blocking.extend(build_definition.problems())
    if not provenance_ok:
        blocking.append("provenance did not verify at the consumer")
    for name, findings in (
        ("sbom", sbom_findings),
        ("secret", secret_findings),
        ("license", license_findings),
        ("model_data_rights", rights_findings),
    ):
        for finding in findings:
            if "P0" in finding or "must" in finding or "requires" in finding:
                blocking.append(f"{name}: {finding}")
    if vulnerability is not None and vulnerability.get("blocking"):
        blocking.append(f"vulnerability: {', '.join(vulnerability['blocking'])}")
    return {
        "decision": "GO" if not blocking else "NO_GO",
        "blocking_findings": blocking,
        "gates": {gate: ("not_run" if any("not performed" in item for item in blocking) else "pending") for gate in RELEASE_GATES},
        "note": "任何资产变化必须新版本；同版本资产不得被替换",
    }


def freeze_release(
    *, policy: ReleasePolicy, manifest: ArtifactManifest, records: Sequence[rec.ReleaseArtifactRecord]
) -> Dict[str, Any]:
    """Freeze the supply-chain record (step 45)."""
    problems = list(policy.problems()) + list(manifest.problems())
    for record in records:
        problems.extend(record.validate())
    if problems:
        raise ConfigError("refusing to freeze the release: " + "; ".join(problems[:5]))
    return {
        "schema_version": f"{SCHEMA_PREFIX}.release-supply-chain.v1",
        "policy": policy.policy_id,
        "manifest_digest": manifest.manifest_digest,
        "assets": len(manifest.assets),
        "records": [record.as_dict() for record in records],
    }


# ── PROTOCOL_STEPS (45) ─────────────────────────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 ReleasePolicy", ("supply_chain.ReleasePolicy",)),
    (2, "冻结 ReleaseCandidateSnapshot", ("contracts.ReleaseCandidateSnapshot", "contracts.new_candidate")),
    (3, "执行 release GO/NO-GO 前置检查", ("supply_chain.supply_chain_go_no_go", "experiment.check_prerequisites")),
    (4, "选择版本号并验证语义", ("supply_chain.validate_version",)),
    (5, "生成并审查 changelog", ("supply_chain.ReleasePolicy", "telemetry.ActionLog")),
    (6, "验证 tag 目标和保护策略", ("supply_chain.validate_version", "campaign.PROHIBITED_ACTIONS")),
    (7, "冻结 build definition", ("supply_chain.BuildDefinition",)),
    (8, "最小化 builder 权限", ("supply_chain.BuildDefinition.problems",)),
    (9, "验证构建隔离", ("supply_chain.BuildDefinition", "campaign.REQUIRED_ISOLATION")),
    (10, "建立 release asset inventory", ("supply_chain.AssetRecord", "supply_chain.EXTRA_ASSETS")),
    (11, "构建 sdist", ("supply_chain.AssetRecord", "telemetry.ActionLog")),
    (12, "从 sdist 构建 wheel", ("supply_chain.AssetRecord", "supply_chain.consumer_verify")),
    (13, "构建 container image", ("supply_chain.AssetRecord", "supply_chain.consumer_verify")),
    (14, "构建架构/后端变体", ("records.ReleaseArtifactRecord.ARTIFACT_TYPES", "supply_chain.AssetRecord")),
    (15, "构建 sample/evidence bundle", ("contracts.PublicEvidenceBundle", "supply_chain.ArtifactManifest")),
    (16, "生成 canonical artifact manifest", ("supply_chain.ArtifactManifest", "supply_chain.verify_artifact_manifest")),
    (17, "生成构建 provenance", ("supply_chain.Provenance",)),
    (18, "生成 artifact attestation", ("supply_chain.Provenance.problems", "supply_chain.verify_provenance")),
    (19, "生成软件包 SBOM", ("supply_chain.SbomDocument", "supply_chain.SbomComponent")),
    (20, "生成 container SBOM", ("supply_chain.SbomDocument", "supply_chain.SbomDocument.COVERAGE_KINDS")),
    (21, "审计 accelerator binary 依赖", ("supply_chain.SbomComponent", "hero_replay.check_binary_compatibility")),
    (22, "验证 SBOM 语法与覆盖", ("supply_chain.verify_sbom_coverage",)),
    (23, "运行漏洞扫描", ("supply_chain.VulnerabilityReport",)),
    (24, "执行漏洞适用性裁决", ("supply_chain.dispose_vulnerabilities", "records.FINDING_DISPOSITIONS")),
    (25, "审计源码许可证", ("supply_chain.LicenseInventory",)),
    (26, "执行逐文件许可覆盖", ("supply_chain.LicenseInventory.problems",)),
    (27, "审计第三方源码与 notices", ("supply_chain.LicenseInventory", "supply_chain.AssetRecord")),
    (28, "审计模型与 tokenizer 权利", ("supply_chain.RightsDecision", "supply_chain.audit_model_data_rights")),
    (29, "审计数据与 prompt 权利", ("supply_chain.RightsDecision", "supply_chain.RIGHTS_KINDS")),
    (30, "审计图片、字体和录屏素材", ("supply_chain.RightsDecision.KINDS", "demo.DemoRecordingManifest")),
    (31, "运行 secret 扫描", ("supply_chain.verify_secret_scan", "supply_chain.SECRET_SCAN_SCOPES")),
    (32, "运行 PII/路径/内网扫描", ("supply_chain.verify_pii_scan", "docs_gate.scan_privacy_paths")),
    (33, "检查 archive 内容边界", ("supply_chain.check_archive_boundary",)),
    (34, "执行独立重复构建", ("supply_chain.compare_rebuilds",)),
    (35, "归因非确定性", ("supply_chain.attribute_nondeterminism", "supply_chain.NONDETERMINISM_SOURCES")),
    (36, "执行 wheel/sdist 消费验证", ("supply_chain.consumer_verify", "quickstart.check_core_import")),
    (37, "执行 container 消费验证", ("supply_chain.consumer_verify",)),
    (38, "执行 provenance 消费端验证", ("supply_chain.verify_provenance", "supply_chain.PROVENANCE_CONSUMER_CHECKS")),
    (39, "执行 SBOM/notice 消费验证", ("supply_chain.verify_sbom_coverage", "supply_chain.consumer_verify")),
    (40, "验证 release notes 与实际资产", ("supply_chain.consumer_verify", "docs_gate.check_version_consistency")),
    (41, "验证不可变发布流程", ("supply_chain.check_immutable_publication",)),
    (42, "验证撤回和安全修复路径", ("supply_chain.retraction_plan", "records.FINDING_DISPOSITIONS")),
    (43, "归档可引用版本", ("supply_chain.ArchiveCitation",)),
    (44, "执行最终独立供应链审查", ("supply_chain.supply_chain_go_no_go", "claims.ContextAdjudication")),
    (45, "签发 ReleaseSupplyChainRecord", ("supply_chain.freeze_release", "records.ReleaseArtifactRecord")),
)

TITLE = "Tag→Release 制品、Provenance、SBOM 与许可证"
LEVEL = "P0"
CLAIM_BOUNDARY = (
    "E15-05 证明发布资产可绑定、可验证、可合法分发；不证明源码无恶意或无漏洞"
)


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the supply-chain interfaces (labelled smoke)."""
    problems: List[str] = []
    policy = ReleasePolicy(policy_id="pol-1", signing="sigstore-keyless", retention_policy="immutable archive")
    problems.extend(policy.problems())
    if not validate_version(version="1.2.3", change_kind="feature", policy=policy):
        problems.append("a feature change on the patch position was accepted")
    definition = BuildDefinition(
        definition_id="bd-1",
        workflow=".github/workflows/release.yml",
        action_digests=("sha256:" + "a" * 64,),
        base_image_digest="sha256:" + "b" * 64,
        toolchain_lock="uv.lock",
        builder_id="github-hosted",
        isolated_builder=True,
        permissions={"contents": "write", "id-token": "write"},
    )
    problems.extend(definition.problems())
    floating = BuildDefinition(definition_id="bd-2", workflow="w", builder_id="b", isolated_builder=True)
    if not floating.problems():
        problems.append("a floating build definition was accepted")
    manifest = ArtifactManifest(
        release_version="0.1.0",
        tag="v0.1.0",
        source_commit="c" * 40,
        assets=(
            AssetRecord(
                logical_name="hqsb-0.1.0-py3-none-any.whl",
                asset_type="wheel",
                version="0.1.0",
                media_type="application/zip",
                size_bytes=1024,
                sha256="sha256:" + "d" * 64,
                role="package",
            ),
        ),
    )
    problems.extend(manifest.problems())
    provenance = Provenance(
        provenance_id="prov-1",
        subject_digest="sha256:" + "d" * 64,
        source_repository="https://github.com/example/hqsb",
        source_commit="c" * 40,
        builder_id="github-hosted",
        build_definition_digest="sha256:" + "e" * 64,
    )
    result = verify_provenance(
        provenance,
        consumer={
            "asset_digest": provenance.subject_digest,
            "official_repository": provenance.source_repository,
            "tag_commit": provenance.source_commit,
            "expected_build_definition_digest": "sha256:" + "e" * 64,
            "attestation_signature_valid": True,
        },
        policy={"builder_allowlist": ("github-hosted",)},
    )
    if not result["ok"]:
        problems.append(f"a valid provenance failed verification: {result['failed']}")
    swapped = verify_provenance(
        provenance,
        consumer={"asset_digest": "sha256:" + "f" * 64, "hard_fail_on_mismatch": True},
        policy={"builder_allowlist": ("github-hosted",)},
    )
    if swapped["ok"]:
        problems.append("a tampered asset digest passed provenance verification")
    scan = verify_secret_scan(scope=("git_history",), findings=(), canary_injected=False, canary_detected=False)
    if not scan:
        problems.append("a partial secret scan without a canary was accepted")
    boundary = check_archive_boundary(members=["hqsb/__init__.py", "repo/.git/config"])
    if not boundary:
        problems.append("a .git directory inside an archive was accepted")
    rights = RightsDecision(artifact_id="model-1", artifact_kind="model", rights={key: False for key in RIGHTS_KINDS})
    if not rights.problems():
        problems.append("a redistribution-denied model without a manifest-only plan was accepted")
    rebuild = compare_rebuilds({"a": "sha256:" + "1" * 64}, {"a": "sha256:" + "2" * 64})
    if rebuild["reproducible_level"] != "content_reproducible_only":
        problems.append("a differing rebuild was reported as byte-reproducible")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "asset_kinds": len(ASSET_KINDS),
        "release_gates": len(RELEASE_GATES),
        "rights_kinds": len(RIGHTS_KINDS),
        "problems": problems,
    }
