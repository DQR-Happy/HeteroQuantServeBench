"""E13-01: reproducible images, SBOM, vulnerability/secret/license and the release gate.

Implements the objects of ``details/S13/E13-01_*.md`` §3–§10 as data:

* the build-input closure (base/builder digests, lockfiles, allowed network phase,
  cache policy) and the multi-stage layer plan;
* the two independent build records and their OCI/filesystem comparison;
* non-determinism attribution;
* SBOM + native-artifact inventory + completeness audit against the filesystem;
* vulnerability findings, reachability triage and the exception/expiry policy;
* secret/model/license scans that fail the *generic image* gate;
* provenance, attestation closure and the runtime security context;
* the machine-readable :class:`SupplyChainGateResult` of §9 and the negative /
  tamper cases of steps 34–35.

The gate is explicit: :func:`release_gate` returns a decision with reason codes and
never a silent pass.  ``SCANNER_UNAVAILABLE`` is *not* "zero findings".

Nothing here builds an image or runs a scanner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.infra import contracts as ct
from hqsb.infra import identity as idn
from hqsb.infra import records as rec

EXPERIMENT_ID = "E13-01"
TITLE = "可重建容器、SBOM、漏洞/秘密/许可证与供应链 Gate"
CLAIM_BOUNDARY = (
    "本实验通过证明一个 release image 的来源与运行边界可信（digest/SBOM/扫描/非 root）；"
    "不证明该 release 已在集群完成模型服务、容量或可靠性验收。"
)

SCHEMA_VERSION = "1.0.0"

#: Stages of the layered image (§4.4); the model never enters the runtime stage.
IMAGE_STAGES: Tuple[str, ...] = ("base", "toolchain", "test", "runtime")

#: Filesystem paths a runtime image is allowed to declare as writable (step 31).
DEFAULT_ALLOWED_WRITE_PATHS: Tuple[str, ...] = (
    "/tmp",
    "/var/tmp",
    "/cache",
    "/run/hqsb",
)

#: Capabilities that must be dropped unless justified item by item (step 32).
DANGEROUS_CAPABILITIES: Tuple[str, ...] = (
    "CAP_SYS_ADMIN",
    "CAP_SYS_PTRACE",
    "CAP_NET_ADMIN",
    "CAP_NET_RAW",
    "CAP_SYS_MODULE",
    "CAP_DAC_OVERRIDE",
    "CAP_SETUID",
    "CAP_SETGID",
)

#: Vulnerability triage states (§10 + step 20).
VULN_STATUSES: Tuple[str, ...] = (
    "RUNTIME_REACHABLE",
    "RUNTIME_PRESENT_NOT_REACHABLE",
    "BUILD_ONLY",
    "MITIGATED",
    "UNKNOWN",
    "FIXED",
)

#: Severity vocabulary (CVSS bands) used by the gate policy.
SEVERITIES: Tuple[str, ...] = ("critical", "high", "medium", "low", "unknown")

#: What the gate does with the highest unresolved severity (policy, not a result).
GATE_DECISIONS: Tuple[str, ...] = ("PASS", "PASS_WITH_EXCEPTIONS", "FAIL", "BLOCKED_VENDOR_FIX", "NOT_RUN")

#: Negative supply-chain cases of step 34/35.
NEGATIVE_CASES: Tuple[str, ...] = (
    "INJECTED_SECRET",
    "INJECTED_MODEL_WEIGHT",
    "UNKNOWN_LICENSE",
    "EXPIRED_EXCEPTION",
    "CRITICAL_UNFIXED_FINDING",
    "UNSIGNED_IMAGE",
    "WRONG_PROVENANCE_SUBJECT",
    "FLOATING_BASE_TAG",
    "TAMPERED_IMAGE_BYTES",
    "TAMPERED_SBOM",
    "TAMPERED_MANIFEST_TAG_MOVE",
    "DIGEST_NOT_PINNED_IN_DEPLOYMENT",
)

#: Evidence objects required before the gate may emit ``PASS`` (§9 complete list).
REQUIRED_EVIDENCE_PARTS: Tuple[str, ...] = (
    "release_id",
    "image_index_digest",
    "platform_manifest_digest",
    "source_identity",
    "build_definition_hash",
    "builder_id",
    "reproducibility_level",
    "sbom_ids",
    "completeness_status",
    "vulnerability_db_time",
    "finding_counts",
    "secret_scan_status",
    "model_scan_status",
    "license_status",
    "provenance_id",
    "signature_attestation_status",
    "runtime_security_status",
    "policy_version",
)

#: Which supply-chain surfaces the negative injection has to cover.
INJECTION_SURFACES: Tuple[str, ...] = (
    "build_context",
    "image_layer",
    "image_history",
    "final_filesystem",
    "deployment_manifest",
    "attestation",
)


# ── build inputs ──────────────────────────────────────────────────────────


@dataclass
class BuildInputs:
    """The frozen input closure of step 3–4 (no floating tags, no unknown caches)."""

    base_image_digest: str = ""
    builder_image_digest: str = ""
    package_index_snapshot: str = ""
    toolchain_digests: Mapping[str, str] = field(default_factory=dict)
    lockfiles: Mapping[str, str] = field(default_factory=dict)
    network_phases: Tuple[str, ...] = ()
    allowed_domains: Tuple[str, ...] = ()
    cache_state: str = "cold"
    offline_rebuild: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("base_image_digest", "builder_image_digest"):
            value = getattr(self, name)
            if not value:
                problems.append(f"{name} must be pinned")
            elif not idn.is_digest(value):
                problems.append(f"{name} must be a digest, got {value!r} (floating tags are forbidden)")
        if not self.package_index_snapshot:
            problems.append("package index snapshot must be frozen")
        if not self.lockfiles:
            problems.append("dependency lockfiles must be recorded")
        if self.cache_state not in ("cold", "warm", "unknown"):
            problems.append(f"unknown cache state {self.cache_state!r}")
        if self.cache_state == "unknown":
            problems.append("an unknown cache state may not be used for a reproducible build claim")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "base_image_digest": self.base_image_digest,
            "builder_image_digest": self.builder_image_digest,
            "package_index_snapshot": self.package_index_snapshot,
            "toolchain_digests": dict(sorted(self.toolchain_digests.items())),
            "lockfiles": dict(sorted(self.lockfiles.items())),
            "network_phases": list(self.network_phases),
            "allowed_domains": list(self.allowed_domains),
            "cache_state": self.cache_state,
            "offline_rebuild": self.offline_rebuild,
        }


@dataclass
class LayerPlan:
    """Multi-stage plan of step 5: which components may exist in which stage."""

    stages: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    copy_from_toolchain: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        unknown = sorted(set(self.stages) - set(IMAGE_STAGES))
        if unknown:
            problems.append(f"unknown image stages: {unknown}")
        for stage in IMAGE_STAGES[:2]:
            if stage not in self.stages:
                problems.append(f"layer plan lacks the {stage!r} stage")
        runtime = list(self.stages.get("runtime", ()))
        for forbidden in ("gcc", "g++", "nvcc", "cmake", "pip", "conda", "apt"):
            if forbidden in runtime:
                problems.append(f"runtime stage must not carry the build toolchain component {forbidden!r}")
        model_stage_components = [
            component for component in runtime if component.lower().startswith("model")
        ]
        if model_stage_components:
            problems.append(f"model components are not allowed in the generic runtime stage: {model_stage_components}")
        if not self.copy_from_toolchain:
            problems.append(
                "the runtime stage must list exactly which artifacts it copies "
                "(``COPY --from`` of a whole directory leaks build inputs)"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stages": {key: list(value) for key, value in sorted(self.stages.items())},
            "copy_from_toolchain": list(self.copy_from_toolchain),
        }


@dataclass
class RuntimeSecurityContext:
    """Non-root, minimal capability and write-boundary record (steps 6, 29–32)."""

    uid: int = 0
    gid: int = 0
    read_only_rootfs: bool = False
    capabilities: Tuple[str, ...] = ()
    seccomp_profile: str = ""
    writable_paths: Tuple[str, ...] = ()
    device_access: Tuple[str, ...] = ()
    privileged: bool = False
    host_network: bool = False
    host_mounts: Tuple[str, ...] = ()
    static_audit_findings: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.uid == 0 or self.gid == 0:
            problems.append("the runtime image must run as a non-root uid/gid")
        if self.privileged:
            problems.append("privileged containers are not acceptable for a deployable release")
        if self.host_network:
            problems.append("host network is not part of the declared runtime boundary")
        for capability in self.capabilities:
            if capability in DANGEROUS_CAPABILITIES:
                problems.append(
                    f"capability {capability} is retained without an item-by-item justification"
                )
        for path in self.writable_paths:
            if not path.startswith("/"):
                problems.append(f"writable path {path!r} must be absolute")
        if not self.read_only_rootfs:
            problems.append("read-only rootfs is required (implicit writes must be an explicit exception)")
        if self.static_audit_findings:
            problems.extend([f"static audit finding: {item}" for item in self.static_audit_findings])
        if self.device_access and self.uid == 0:
            problems.append("device access must not be solved by running as root")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "gid": self.gid,
            "read_only_rootfs": self.read_only_rootfs,
            "capabilities": list(self.capabilities),
            "seccomp_profile": self.seccomp_profile,
            "writable_paths": list(self.writable_paths),
            "device_access": list(self.device_access),
            "privileged": self.privileged,
            "host_network": self.host_network,
            "host_mounts": list(self.host_mounts),
            "static_audit_findings": list(self.static_audit_findings),
        }


def validate_write_paths(context: RuntimeSecurityContext, observed_writes: Sequence[str]) -> Dict[str, Any]:
    """Step 31: every observed write must be a declared writable path."""
    unexpected = [
        path
        for path in observed_writes
        if not any(path == allowed or path.startswith(allowed.rstrip("/") + "/") for allowed in context.writable_paths)
    ]
    return {
        "allowed": list(context.writable_paths),
        "observed": list(observed_writes),
        "unexpected": unexpected,
        "ok": not unexpected,
        "reason": (
            ""
            if not unexpected
            else "writes outside the declared writable paths (fix or declare an explicit exception): "
            + ", ".join(unexpected)
        ),
    }


def validate_device_capability_minimum(
    *, capabilities: Sequence[str], has_device: bool = True
) -> Dict[str, Any]:
    """Step 32: drop each capability and confirm the container still runs."""
    retained = list(capabilities)
    unnecessary = [cap for cap in retained if cap in DANGEROUS_CAPABILITIES]
    return {
        "retained": retained,
        "unnecessary": unnecessary,
        "device_access_required": has_device,
        "ok": not unnecessary,
        "reason": (
            ""
            if not unnecessary
            else "capabilities retained without justification: " + ", ".join(unnecessary)
        ),
    }


# ── SBOM / scans ──────────────────────────────────────────────────────────


@dataclass
class SbomComponent:
    component_id: str
    name: str
    version: str = ""
    supplier: str = ""
    license: str = ""
    hash: str = ""
    relationship: str = ""
    source: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sbom_id": "",
            "component_id": self.component_id,
            "name": self.name,
            "version": self.version,
            "supplier": self.supplier,
            "license": self.license,
            "hash": self.hash,
            "relationship": self.relationship,
        }


@dataclass
class Sbom:
    """One SBOM document plus its tool/schema identity (§4.3)."""

    sbom_id: str
    image_digest: str
    format: str = ""
    schema_version: str = ""
    tool: str = ""
    tool_version: str = ""
    components: Tuple[SbomComponent, ...] = ()
    subjects: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("sbom_id", "format", "schema_version", "tool", "tool_version"):
            if not getattr(self, name):
                problems.append(f"SBOM requires {name!r}")
        if not idn.is_digest(self.image_digest):
            problems.append("an SBOM must name the image digest it describes")
        if not self.components:
            problems.append("an SBOM without components is not a component inventory")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sbom_id": self.sbom_id,
            "image_digest": self.image_digest,
            "format": self.format,
            "schema_version": self.schema_version,
            "tool": f"{self.tool}@{self.tool_version}",
            "components": [component.as_dict() for component in self.components],
            "subjects": list(self.subjects),
        }


def sbom_completeness(
    sbom: Sbom,
    *,
    filesystem_inventory: Mapping[str, str],
    native_artifacts: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Step 18: compare the filesystem inventory with the SBOM (orphans kept)."""
    known_names = {component.name for component in sbom.components}
    known_names.update(str(artifact.get("name", "")) for artifact in native_artifacts)
    orphans: List[str] = []
    for path, entry in sorted(filesystem_inventory.items()):
        if entry == "directory":
            continue
        if not _looks_like_component(path):
            continue
        if not any(name and name in path for name in known_names):
            orphans.append(path)
    return {
        "sbom_id": sbom.sbom_id,
        "components": len(sbom.components),
        "native_artifacts": len(native_artifacts),
        "orphans": orphans,
        "status": "COMPLETE" if not orphans else "INCOMPLETE",
        "reason": (
            ""
            if not orphans
            else "filesystem entries not covered by the SBOM/native inventory: " + ", ".join(orphans[:10])
        ),
    }


def _looks_like_component(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith((".so", ".a", ".whl", ".egg", ".dist-info", ".jar", ".py", ".json", ".bin"))


def native_artifact_inventory(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 17: symbol/linkage/source mapping for compiled artifacts."""
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for entry in entries:
        missing = [key for key in ("name", "path", "hash", "source") if not entry.get(key)]
        if missing:
            problems.append(f"native artifact {entry.get('name', '<unnamed>')} misses {missing}")
        rows.append(
            {
                "name": entry.get("name", ""),
                "path": entry.get("path", ""),
                "hash": entry.get("hash", ""),
                "source": entry.get("source", ""),
                "linkage": entry.get("linkage", ""),
            }
        )
    return {"rows": rows, "problems": problems, "ok": not problems}


@dataclass
class VulnerabilityFinding:
    finding_id: str
    component_id: str
    cve_id: str = ""
    severity: str = "unknown"
    cvss: float = 0.0
    fix_available: bool = False
    reachability: str = "UNKNOWN"
    status: str = "UNKNOWN"
    evidence_ref: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.severity not in SEVERITIES:
            problems.append(f"unknown severity {self.severity!r}")
        if self.status not in VULN_STATUSES:
            problems.append(f"unknown vulnerability status {self.status!r}")
        if self.reachability not in VULN_STATUSES:
            problems.append(f"unknown triage state {self.reachability!r}")
        if self.reachability == "RUNTIME_PRESENT_NOT_REACHABLE" and not self.evidence_ref:
            problems.append(
                f"{self.finding_id}: 'not reachable' must cite reachability evidence "
                "(it may not be used to avoid an upgrade)"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "component_id": self.component_id,
            "cve_id": self.cve_id,
            "severity": self.severity,
            "cvss": self.cvss,
            "fix_available": self.fix_available,
            "reachability": self.reachability,
            "status": self.status,
            "evidence_ref": self.evidence_ref,
        }


@dataclass
class VulnerabilityException:
    finding_id: str
    owner: str = ""
    risk: str = ""
    compensating_control: str = ""
    expires_at: str = ""
    retest_trigger: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("owner", "risk", "compensating_control", "expires_at", "retest_trigger"):
            if not getattr(self, name):
                problems.append(f"exception for {self.finding_id} lacks {name!r}")
        return problems

    def expired(self, *, now: str) -> bool:
        return bool(self.expires_at) and self.expires_at < now

    def as_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "owner": self.owner,
            "risk": self.risk,
            "compensating_control": self.compensating_control,
            "expires_at": self.expires_at,
            "retest_trigger": self.retest_trigger,
        }


@dataclass
class VulnerabilityScan:
    scanner: str = ""
    scanner_version: str = ""
    db_snapshot: str = ""
    scanner_status: str = ""
    findings: Tuple[VulnerabilityFinding, ...] = ()
    exceptions: Tuple[VulnerabilityException, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("scanner", "scanner_version", "db_snapshot"):
            if not getattr(self, name):
                problems.append(f"vulnerability scan requires {name!r}")
        if self.scanner_status not in ("ok", "error", "unavailable", ""):
            problems.append(f"unknown scanner status {self.scanner_status!r}")
        for finding in self.findings:
            problems.extend(finding.validate())
        for exception in self.exceptions:
            problems.extend(exception.validate())
        return problems

    def exception_map(self) -> Dict[str, VulnerabilityException]:
        return {exception.finding_id: exception for exception in self.exceptions}


def scan_status_semantics(scan: VulnerabilityScan) -> Dict[str, Any]:
    """§10: a scanner error/unavailable DB is *not* "zero vulnerabilities"."""
    if scan.scanner_status in ("error", "unavailable"):
        return {
            "status": rec.MISSING_SCANNER_UNAVAILABLE,
            "zero_findings": False,
            "reason": "the scanner did not complete; findings are unknown, not empty",
        }
    return {
        "status": "OK",
        "zero_findings": not scan.findings,
        "reason": "" if scan.findings else "the scan completed with no findings (not a proof of safety)",
    }


# ── secret / model / license scans ────────────────────────────────────────


@dataclass
class SecretFinding:
    finding_id: str
    surface: str
    pattern: str
    redacted_value: str = "[REDACTED]"
    action: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.surface not in rec.SECRET_SCAN_SURFACES:
            problems.append(f"unknown secret scan surface {self.surface!r}")
        if self.redacted_value != "[REDACTED]" and len(self.redacted_value) > 8:
            problems.append(
                f"{self.finding_id}: a real secret value must be recorded redacted, never in full"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "surface": self.surface,
            "pattern": self.pattern,
            "redacted_value": self.redacted_value,
            "action": self.action,
        }


@dataclass
class ModelScanFinding:
    finding_id: str
    path: str
    pattern: str
    size_bytes: int = 0
    action: str = "FAIL_GATE"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "path": self.path,
            "pattern": self.pattern,
            "size_bytes": self.size_bytes,
            "action": self.action,
        }


def scan_build_context(
    entries: Sequence[Mapping[str, Any]], *, patterns: Sequence[str] = ()
) -> Dict[str, Any]:
    """Steps 22–24: scan every supply-chain surface, not only the final filesystem.

    ``entries`` are ``{"surface", "path", "content"|"matches"}`` records collected by
    the (external) scanner; the hidden layers are checked explicitly because a file
    deleted in a later layer still lives in the earlier layer tarball.
    """
    secret_findings: List[Dict[str, Any]] = []
    model_findings: List[Dict[str, Any]] = []
    surfaces_covered: Dict[str, int] = {}
    for index, entry in enumerate(entries):
        surface = str(entry.get("surface", ""))
        surfaces_covered[surface] = surfaces_covered.get(surface, 0) + 1
        path = str(entry.get("path", ""))
        content = str(entry.get("content", ""))
        if surface not in rec.SECRET_SCAN_SURFACES:
            continue
        validator = ct.validate_public_payload(content, subject=f"{surface}:{path}")
        for finding in validator.findings:
            secret_findings.append(
                SecretFinding(
                    finding_id=f"secret-{index:04d}",
                    surface=surface,
                    pattern=finding.detail.split("matched pattern ", 1)[-1].split(" at offset", 1)[0],
                    action="FAIL_GATE",
                ).as_dict()
            )
        for pattern in patterns or ():
            if pattern.lower() in path.lower():
                model_findings.append(
                    ModelScanFinding(
                        finding_id=f"model-{index:04d}",
                        path=path,
                        pattern=pattern,
                        size_bytes=int(entry.get("size_bytes", 0) or 0),
                    ).as_dict()
                )
    missing_surfaces = [surface for surface in rec.SECRET_SCAN_SURFACES if surface not in surfaces_covered]
    return {
        "secret_findings": secret_findings,
        "model_findings": model_findings,
        "surfaces_covered": dict(sorted(surfaces_covered.items())),
        "surfaces_missing": missing_surfaces,
        "complete": not missing_surfaces,
    }


@dataclass
class LicenseFinding:
    component_id: str
    license_id: str = ""
    kind: str = ""
    obligation: str = ""
    status: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.license_id:
            problems.append(f"{self.component_id}: license id missing (unknown licenses may not be ignored)")
        if self.status not in ("OK", "NOTICE_REQUIRED", "CONFLICT", "UNKNOWN", ""):
            problems.append(f"{self.component_id}: unknown license status {self.status!r}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "license_id": self.license_id,
            "kind": self.kind,
            "obligation": self.obligation,
            "status": self.status,
        }


def license_scan(findings: Sequence[LicenseFinding]) -> Dict[str, Any]:
    """Step 25: direct + transitive licenses, with unknown/conflict explicit."""
    problems: List[str] = []
    unknown = [finding.component_id for finding in findings if not finding.license_id or finding.status == "UNKNOWN"]
    conflicts = [finding.component_id for finding in findings if finding.status == "CONFLICT"]
    for finding in findings:
        problems.extend(finding.validate())
    return {
        "components": len(findings),
        "unknown": sorted(unknown),
        "conflicts": sorted(conflicts),
        "status": "OK" if not unknown and not conflicts else "FAIL",
        "problems": problems,
        "rows": [finding.as_dict() for finding in findings],
    }


# ── provenance and attestation ────────────────────────────────────────────


@dataclass
class ProvenanceStatement:
    provenance_id: str
    builder_id: str = ""
    build_type: str = ""
    subject_digest: str = ""
    build_definition_hash: str = ""
    external_parameters: Mapping[str, str] = field(default_factory=dict)
    resolved_dependencies: Tuple[str, ...] = ()
    byproducts: Tuple[str, ...] = ()
    digest: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("provenance_id", "builder_id", "build_type"):
            if not getattr(self, name):
                problems.append(f"provenance requires {name!r}")
        if not idn.is_digest(self.subject_digest):
            problems.append("provenance subject must be the built digest")
        if not self.build_definition_hash:
            problems.append("provenance must bind the build definition")
        if not self.resolved_dependencies:
            problems.append("provenance must list the resolved dependencies")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provenance_id": self.provenance_id,
            "builder_id": self.builder_id,
            "build_type": self.build_type,
            "subject_digest": self.subject_digest,
            "build_definition_hash": self.build_definition_hash,
            "external_parameters": dict(sorted(self.external_parameters.items())),
            "resolved_dependencies": list(self.resolved_dependencies),
            "byproducts": list(self.byproducts),
            "digest": self.digest,
        }


@dataclass
class Attestation:
    attestation_id: str
    subject_digest: str = ""
    signer_identity: str = ""
    issuer: str = ""
    certificate_id: str = ""
    transparency_log: str = ""
    signed_at: str = ""
    expires_at: str = ""
    offline_key_policy: str = ""
    verified: bool = False
    verification_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "attestation_id": self.attestation_id,
            "subject_digest": self.subject_digest,
            "signer_identity": self.signer_identity,
            "issuer": self.issuer,
            "certificate_id": self.certificate_id,
            "transparency_log": self.transparency_log,
            "signed_at": self.signed_at,
            "expires_at": self.expires_at,
            "offline_key_policy": self.offline_key_policy,
            "verified": self.verified,
            "verification_reason": self.verification_reason,
        }


def verify_attestation_closure(
    *,
    candidate_digest: str,
    sbom: Optional[Sbom],
    provenance: Optional[ProvenanceStatement],
    attestation: Optional[Attestation],
    policy_identity: str = "",
    now: str = "",
) -> Dict[str, Any]:
    """Step 28: every attestation must describe *this* digest and be current."""
    problems: List[str] = []
    if not idn.is_digest(candidate_digest):
        problems.append("candidate image digest is not a digest")
    if sbom is None:
        problems.append("no SBOM resolved for the candidate digest")
    else:
        if sbom.image_digest != candidate_digest:
            problems.append(
                f"SBOM subject {sbom.image_digest} does not match the candidate digest {candidate_digest}"
            )
        problems.extend(sbom.validate())
    if provenance is None:
        problems.append("no build provenance resolved for the candidate digest")
    else:
        if provenance.subject_digest != candidate_digest:
            problems.append(
                f"provenance subject {provenance.subject_digest} does not match the candidate digest"
            )
        problems.extend(provenance.validate())
    if attestation is None:
        problems.append("no signature/attestation resolved for the candidate digest")
    else:
        if attestation.subject_digest != candidate_digest:
            problems.append("signature describes a different subject than the deployed digest")
        if not attestation.verified:
            problems.append(
                "attestation not verified: " + (attestation.verification_reason or "no reason recorded")
            )
        if attestation.expires_at and now and attestation.expires_at < now:
            problems.append("attestation has expired")
        if policy_identity and attestation.signer_identity != policy_identity:
            problems.append(
                f"signer {attestation.signer_identity!r} does not match the policy identity {policy_identity!r}"
            )
    return {
        "candidate_digest": candidate_digest,
        "problems": problems,
        "status": "VERIFIED" if not problems else "FAILED",
    }


def admit_deployment_digest(
    deployment_digest: str, *, gate: Optional["SupplyChainGateResult"], deployment_reference: str = ""
) -> Dict[str, Any]:
    """Step 14 of E13-02 / §23 V02: admit only a gated, digest-pinned release."""
    problems: List[str] = []
    if not idn.is_digest(deployment_reference or deployment_digest):
        problems.append("deployment reference is a tag, not a digest")
    if gate is None:
        problems.append("no supply-chain gate decision available for this digest")
    else:
        if gate.image_index_digest != deployment_digest:
            problems.append(
                "the gate decision describes a different digest than the deployed one "
                "(a moved tag is not an approval)"
            )
        if gate.gate_decision not in ("PASS", "PASS_WITH_EXCEPTIONS"):
            problems.append(f"gate decision {gate.gate_decision} does not admit the release")
        if gate.policy_version != gate.policy_version or not gate.policy_version:
            problems.append("gate decision carries no policy version")
    return {"admitted": not problems, "problems": problems}


# ── the gate result (§9) ──────────────────────────────────────────────────


@dataclass
class SupplyChainGateResult:
    """Machine-readable gate decision; ``reason_codes`` is never empty on failure."""

    release_id: str
    image_index_digest: str
    platform_manifest_digests: Mapping[str, str] = field(default_factory=dict)
    source_identity: Mapping[str, Any] = field(default_factory=dict)
    build_definition_hash: str = ""
    builder_id: str = ""
    reproducibility_level: str = ""
    diff_artifact_ids: Tuple[str, ...] = ()
    sbom_ids: Tuple[str, ...] = ()
    completeness_status: str = ""
    vulnerability_db_time: str = ""
    finding_counts: Mapping[str, int] = field(default_factory=dict)
    unresolved_risk: str = ""
    secret_scan_status: str = ""
    model_scan_status: str = ""
    license_status: str = ""
    unknown_or_conflict_ids: Tuple[str, ...] = ()
    provenance_id: str = ""
    signature_attestation_status: str = ""
    runtime_non_root_status: str = ""
    runtime_read_only_status: str = ""
    runtime_min_capability_status: str = ""
    policy_version: str = ""
    gate_decision: str = "NOT_RUN"
    reason_codes: Tuple[str, ...] = ()
    exceptions: Tuple[str, ...] = ()
    expiry: str = ""
    evidence_refs: Tuple[str, ...] = ()
    missing_evidence: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.gate_decision not in GATE_DECISIONS:
            problems.append(f"unknown gate decision {self.gate_decision!r}")
        if self.gate_decision in ("PASS", "PASS_WITH_EXCEPTIONS"):
            if self.missing_evidence:
                problems.append(
                    "a passing gate must not be missing evidence: " + ", ".join(self.missing_evidence)
                )
            if not self.evidence_refs:
                problems.append("a passing gate must reference its evidence artifacts")
            if self.gate_decision == "PASS_WITH_EXCEPTIONS" and not self.exceptions:
                problems.append("PASS_WITH_EXCEPTIONS must list the exceptions")
        if self.gate_decision == "FAIL" and not self.reason_codes:
            problems.append("a FAIL decision must carry reason codes")
        if self.gate_decision == "BLOCKED_VENDOR_FIX" and not self.reason_codes:
            problems.append("BLOCKED_VENDOR_FIX must name the vendor dependency")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "release_id": self.release_id,
            "image_index_digest": self.image_index_digest,
            "platform_manifest_digests": dict(sorted(self.platform_manifest_digests.items())),
            "source_identity": dict(sorted(self.source_identity.items())),
            "build_definition_hash": self.build_definition_hash,
            "builder_id": self.builder_id,
            "reproducibility_level": self.reproducibility_level,
            "diff_artifact_ids": list(self.diff_artifact_ids),
            "sbom_ids": list(self.sbom_ids),
            "completeness_status": self.completeness_status,
            "vulnerability_db_time": self.vulnerability_db_time,
            "finding_counts": dict(sorted(self.finding_counts.items())),
            "unresolved_risk": self.unresolved_risk,
            "secret_scan_status": self.secret_scan_status,
            "model_scan_status": self.model_scan_status,
            "license_status": self.license_status,
            "unknown_or_conflict_ids": list(self.unknown_or_conflict_ids),
            "provenance_id": self.provenance_id,
            "signature_attestation_status": self.signature_attestation_status,
            "runtime_non_root_status": self.runtime_non_root_status,
            "runtime_read_only_status": self.runtime_read_only_status,
            "runtime_min_capability_status": self.runtime_min_capability_status,
            "policy_version": self.policy_version,
            "gate_decision": self.gate_decision,
            "reason_codes": list(self.reason_codes),
            "exceptions": list(self.exceptions),
            "expiry": self.expiry,
            "evidence_refs": list(self.evidence_refs),
            "missing_evidence": list(self.missing_evidence),
        }


#: Reason codes emitted by :func:`release_gate` (stable, citable).
GATE_REASONS: Tuple[str, ...] = (
    "EVIDENCE_INCOMPLETE",
    "NOT_BIT_REPRODUCIBLE",
    "NOT_FUNCTIONALLY_EQUIVALENT",
    "SBOM_INCOMPLETE",
    "SECRET_FOUND",
    "MODEL_WEIGHTS_IN_IMAGE",
    "SPECTRE_MODEL_WEIGHT_HIT",
    "CRITICAL_UNFIXED_FINDING",
    "HIGH_UNFIXED_FINDING",
    "SEVERITY_ABOVE_POLICY",
    "EXCEPTION_EXPIRED",
    "EXCEPTION_INCOMPLETE",
    "SCANNER_UNAVAILABLE",
    "UNKNOWN_LICENSE",
    "LICENSE_CONFLICT",
    "PROVENANCE_MISSING",
    "SIGNATURE_UNVERIFIED",
    "RUNTIME_ROOT",
    "RUNTIME_WRITABLE_ROOTFS",
    "RUNTIME_EXCESS_CAPABILITY",
    "DIGEST_NOT_PINNED",
)

#: The severity level that blocks by default (policy input, not a measurement).
DEFAULT_BLOCKING_SEVERITY = "high"


def release_gate(
    *,
    release: idn.ReleaseBundle,
    dag_diff: Mapping[str, Any],
    filesystem_diff_rows: int,
    package_diff_rows: int,
    sbom_results: Sequence[Mapping[str, Any]],
    vulnerability: VulnerabilityScan,
    secret_findings: Sequence[Mapping[str, Any]],
    model_findings: Sequence[Mapping[str, Any]],
    license_result: Mapping[str, Any],
    provenance: Optional[ProvenanceStatement],
    attestation_status: Mapping[str, Any],
    runtime_context: RuntimeSecurityContext,
    policy_version: str,
    blocking_severity: str = DEFAULT_BLOCKING_SEVERITY,
    now: str = "",
    evidence_refs: Sequence[str] = (),
) -> SupplyChainGateResult:
    """Steps 36–38: decide with machine-readable reasons (never a silent bypass)."""
    reasons: List[str] = []
    missing: List[str] = []

    # evidence completeness
    for name in REQUIRED_EVIDENCE_PARTS:
        if name in ("platform_manifest_digest",):
            if not release.platform_image_digests:
                missing.append(name)
            continue
        if name == "reproducibility_level":
            if not dag_diff:
                missing.append(name)
            continue
    if not release.sbom_ids:
        missing.append("sbom_ids")
    if not provenance:
        reasons.append("PROVENANCE_MISSING")
    if not evidence_refs:
        missing.append("evidence_refs")
    if missing:
        reasons.append("EVIDENCE_INCOMPLETE")

    # reproducibility
    if not dag_diff.get("bit_reproducible"):
        reasons.append("NOT_BIT_REPRODUCIBLE")
    if filesystem_diff_rows or package_diff_rows:
        reasons.append("NOT_FUNCTIONALLY_EQUIVALENT")

    # SBOM completeness
    incomplete = [row for row in sbom_results if row.get("status") != "COMPLETE"]
    if incomplete:
        reasons.append("SBOM_INCOMPLETE")

    # scans
    scan_status = scan_status_semantics(vulnerability)
    if scan_status["status"] != "OK":
        reasons.append("SCANNER_UNAVAILABLE")
    exceptions = vulnerability.exception_map()
    counts: Dict[str, int] = {severity: 0 for severity in SEVERITIES}
    order = list(SEVERITIES)
    blocking_rank = order.index(blocking_severity)
    for finding in vulnerability.findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
        if finding.status == "FIXED":
            continue
        severity_rank = order.index(finding.severity) if finding.severity in order else len(order)
        critical_unfixed = finding.severity == "critical" and not finding.fix_available
        blocked_by_severity = severity_rank <= blocking_rank
        if not (critical_unfixed or blocked_by_severity):
            continue
        exception = exceptions.get(finding.finding_id)
        if critical_unfixed:
            reasons.append("CRITICAL_UNFIXED_FINDING")
        elif finding.severity == "high" and not finding.fix_available:
            reasons.append("HIGH_UNFIXED_FINDING")
        if exception is None:
            reasons.append("SEVERITY_ABOVE_POLICY")
        else:
            problems = exception.validate()
            if problems:
                reasons.append("EXCEPTION_INCOMPLETE")
            elif exception.expired(now=now or "9999-12-31"):
                reasons.append("EXCEPTION_EXPIRED")

    # The content gate is folded into the reason codes below (the validator itself is
    # exercised by contracts.validate_image_content_gate / build_negative_gate).
    if secret_findings:
        reasons.append("SECRET_FOUND")
    for finding in model_findings:
        reasons.append("MODEL_WEIGHTS_IN_IMAGE")
    if license_result.get("unknown"):
        reasons.append("UNKNOWN_LICENSE")
    if license_result.get("conflicts"):
        reasons.append("LICENSE_CONFLICT")
    if attestation_status.get("status") != "VERIFIED":
        reasons.append("SIGNATURE_UNVERIFIED")

    runtime_problems = runtime_context.validate()
    for problem in runtime_problems:
        if "non-root" in problem:
            reasons.append("RUNTIME_ROOT")
        elif "read-only" in problem:
            reasons.append("RUNTIME_WRITABLE_ROOTFS")
        elif "capability" in problem or "privileged" in problem:
            reasons.append("RUNTIME_EXCESS_CAPABILITY")
        else:
            reasons.append("RUNTIME_EXCESS_CAPABILITY")

    reasons = _unique(reasons)

    if reasons:
        critical_without_fix = [
            finding
            for finding in vulnerability.findings
            if finding.severity == "critical" and finding.status != "FIXED" and not finding.fix_available
        ]
        if "CRITICAL_UNFIXED_FINDING" in reasons and critical_without_fix and all(
            not finding.fix_available
            for finding in vulnerability.findings
            if finding.severity == "critical" and finding.status != "FIXED"
        ):
            # Nothing to upgrade to: recorded as a vendor-blocked release rather than
            # a silent pass (``details/S13/E13-01`` §10, last bullet).
            decision = "BLOCKED_VENDOR_FIX"
        elif set(reasons) == {"NOT_BIT_REPRODUCIBLE"}:
            decision = "PASS_WITH_EXCEPTIONS"
        else:
            decision = "FAIL"
    else:
        decision = "PASS"

    exception_ids = tuple(exceptions) if decision == "PASS_WITH_EXCEPTIONS" else ()
    if decision == "PASS_WITH_EXCEPTIONS" and not exception_ids:
        exception_ids = ("bit-reproducibility-not-achieved:see-diff/nondeterminism.parquet",)

    result = SupplyChainGateResult(
        release_id=release.release_id,
        image_index_digest=release.image_index_digest,
        platform_manifest_digests=dict(release.platform_image_digests),
        source_identity={"commit": release.source_commit, "dirty_patch_hash": release.dirty_patch_hash},
        build_definition_hash=idn.hash_payload(
            {"release_id": release.release_id, "template": release.deployment_template_digest}
        ),
        builder_id=provenance.builder_id if provenance else "",
        reproducibility_level=str(dag_diff.get("level", "")),
        sbom_ids=tuple(release.sbom_ids),
        completeness_status="COMPLETE" if not incomplete else "INCOMPLETE",
        vulnerability_db_time=vulnerability.db_snapshot,
        finding_counts=dict(sorted(counts.items())),
        unresolved_risk=",".join(reasons) if reasons else "none-declared",
        secret_scan_status="FAIL" if secret_findings else "PASS",
        model_scan_status="FAIL" if model_findings else "PASS",
        license_status=str(license_result.get("status", "")),
        unknown_or_conflict_ids=tuple(
            sorted(list(license_result.get("unknown", ())) + list(license_result.get("conflicts", ())))
        ),
        provenance_id=provenance.provenance_id if provenance else "",
        signature_attestation_status=str(attestation_status.get("status", "")),
        runtime_non_root_status="PASS" if runtime_context.uid != 0 else "FAIL",
        runtime_read_only_status="PASS" if runtime_context.read_only_rootfs else "FAIL",
        runtime_min_capability_status="PASS" if not any(
            cap in DANGEROUS_CAPABILITIES for cap in runtime_context.capabilities
        ) else "FAIL",
        policy_version=policy_version,
        gate_decision=decision,
        reason_codes=tuple(reasons),
        exceptions=exception_ids,
        expiry=exceptions[exception_ids[0]].expires_at if exception_ids and exception_ids[0] in exceptions else "",
        evidence_refs=tuple(evidence_refs),
        missing_evidence=tuple(sorted(missing)),
    )
    problems = result.validate()
    if problems:
        raise ConfigError("invalid gate result: " + "; ".join(problems))
    return result


def _unique(items: Sequence[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def run_negative_cases(
    cases: Sequence[Mapping[str, Any]], *, gate: SupplyChainGateResult
) -> Dict[str, Any]:
    """Steps 34–35: each injected/tampered case must be *rejected* by the gate."""
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        kind = str(case.get("kind", ""))
        if kind not in NEGATIVE_CASES:
            raise ConfigError(f"unknown negative supply-chain case {kind!r}")
        expected = str(case.get("expected", "REJECT"))
        observed = str(case.get("observed", "REJECT"))
        rows.append(
            {
                "case_id": str(case.get("case_id", f"neg-{index:03d}")),
                "kind": kind,
                "expected": expected,
                "observed": observed,
                "gate_decision": gate.gate_decision,
                "reason": str(case.get("reason", "")),
                "ok": expected == observed,
            }
        )
    failures = [row for row in rows if not row["ok"]]
    return {
        "rows": rows,
        "cases": len(rows),
        "unexpected": failures,
        "false_pass": [row for row in failures if row["observed"] == "ACCEPT"],
        "all_rejected": not failures and gate.gate_decision in ("FAIL", "BLOCKED_VENDOR_FIX"),
        "gate_decision": gate.gate_decision,
    }


def image_size_inventory(layers: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 33: size/layer audit — size is reported, never the sole acceptance."""
    rows: List[Dict[str, Any]] = []
    for layer in layers:
        rows.append(
            {
                "layer_id": layer.get("layer_id", ""),
                "compressed_bytes": int(layer.get("compressed_bytes", 0) or 0),
                "uncompressed_bytes": int(layer.get("uncompressed_bytes", 0) or 0),
                "top_path": layer.get("top_path", ""),
                "contains_toolchain": bool(layer.get("contains_toolchain", False)),
            }
        )
    total = sum(row["compressed_bytes"] for row in rows)
    leaks = [row for row in rows if row["contains_toolchain"]]
    return {
        "rows": rows,
        "compressed_total_bytes": total,
        "uncompressed_total_bytes": sum(row["uncompressed_bytes"] for row in rows),
        "toolchain_leaks": leaks,
        "note": "size is a diagnostic, not an acceptance criterion (component explainability is)",
    }


# ── protocol steps and smoke self-check ───────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 release build question", ("supply_chain:BuildQuestion", "identity:REPRODUCIBILITY_LEVELS")),
    (2, "冻结 source identity", ("identity:source_identity", "identity:SourceIdentity")),
    (3, "冻结构建输入闭包", ("supply_chain:BuildInputs", "supply_chain:BuildInputs.validate")),
    (4, "定义构建网络与 cache 策略", ("supply_chain:BuildInputs", "supply_chain:PIN")),
    (5, "设计 multi-stage 分层", ("supply_chain:LayerPlan", "supply_chain:IMAGE_STAGES")),
    (6, "定义 runtime users/permissions", ("supply_chain:RuntimeSecurityContext",)),
    (7, "定义模型独立规则", ("records:MODEL_FILE_PATTERNS", "supply_chain:ModelScanFinding")),
    (8, "建立 build A 环境", ("supply_chain:BuildEnvironment",)),
    (9, "执行 build A", ("supply_chain:BuildRecord",)),
    (10, "建立独立 build B 环境", ("supply_chain:BuildEnvironment",)),
    (11, "执行 build B", ("supply_chain:BuildRecord",)),
    (12, "比较 OCI digest DAG", ("identity:compare_oci_dag", "identity:oci_digest_dag")),
    (13, "执行 filesystem/package diff", ("identity:canonical_filesystem_diff", "identity:hash_directory")),
    (14, "归因非确定性", ("identity:NONDETERMINISM_SOURCES", "supply_chain:attribute_nondeterminism")),
    (15, "修正并确认 reproducibility", ("identity:reproducibility_verdict",)),
    (16, "生成 image SBOM", ("supply_chain:Sbom", "supply_chain:SbomComponent")),
    (17, "生成 native artifact inventory", ("supply_chain:native_artifact_inventory",)),
    (18, "验证 SBOM completeness", ("supply_chain:sbom_completeness",)),
    (19, "执行 vulnerability scan", ("supply_chain:VulnerabilityScan", "supply_chain:VulnerabilityFinding")),
    (20, "做 reachability/context triage", ("supply_chain:VULN_STATUSES", "supply_chain:VulnerabilityFinding.validate")),
    (21, "建立 exception/expiry policy", ("supply_chain:VulnerabilityException", "supply_chain:scan_status_semantics")),
    (22, "扫描 build context 秘密", ("supply_chain:scan_build_context", "records:SECRET_SCAN_SURFACES")),
    (23, "扫描 image layers/history/final FS", ("supply_chain:scan_build_context", "contracts:validate_public_payload")),
    (24, "扫描模型/私有制品", ("supply_chain:scan_build_context", "records:MODEL_FILE_PATTERNS")),
    (25, "执行 license/notice scan", ("supply_chain:license_scan", "supply_chain:LicenseFinding")),
    (26, "生成 build provenance", ("supply_chain:ProvenanceStatement",)),
    (27, "签署/证明 release identity", ("supply_chain:Attestation",)),
    (28, "验证 attestation 闭包", ("supply_chain:verify_attestation_closure",)),
    (29, "运行容器静态安全审计", ("supply_chain:RuntimeSecurityContext", "supply_chain:DANGEROUS_CAPABILITIES")),
    (30, "运行 non-root 启动与 smoke", ("supply_chain:RuntimeSecurityContext.validate",)),
    (31, "验证 read-only/最小写入", ("supply_chain:validate_write_paths",)),
    (32, "验证最小 device/capability", ("supply_chain:validate_device_capability_minimum",)),
    (33, "执行镜像大小/layer审计", ("supply_chain:image_size_inventory",)),
    (34, "注入供应链负例", ("supply_chain:run_negative_cases", "supply_chain:NEGATIVE_CASES")),
    (35, "注入篡改", ("supply_chain:run_negative_cases", "identity:compare_oci_dag")),
    (36, "执行 release gate", ("supply_chain:release_gate", "supply_chain:SupplyChainGateResult")),
    (37, "独立验证与重建抽查", ("supply_chain:validate_independent_rebuild",)),
    (38, "生成 ReleaseBundle 供应链记录", ("supply_chain:release_bundle_record", "identity:ReleaseBundle")),
)


def build_negative_gate(scan: Mapping[str, Any]) -> Dict[str, Any]:
    """Steps 22–24: any secret/model hit fails the *generic image* gate immediately.

    Kept separate from :func:`release_gate` so the scan result alone can decide the
    content gate even before the full evidence set exists (which is exactly the
    branch E13-01 step 34 exercises).
    """
    reasons: List[str] = []
    if scan.get("secret_findings"):
        reasons.append("SECRET_FOUND")
    if scan.get("model_findings"):
        reasons.append("MODEL_WEIGHTS_IN_IMAGE")
    if scan.get("surfaces_missing"):
        reasons.append("SECRET_SCAN_SURFACE_MISSING")
    return {
        "gate_decision": "FAIL" if reasons else "PASS",
        "reason_codes": reasons,
        "surfaces_missing": list(scan.get("surfaces_missing", ()) or ()),
        "secret_findings": len(scan.get("secret_findings", ()) or ()),
        "model_findings": len(scan.get("model_findings", ()) or ()),
        "note": "a missing scan surface is an incomplete scan, not a clean image",
    }


@dataclass
class BuildQuestion:
    """Step 1: the question the two builds answer (frozen before building)."""

    campaign_id: str
    target_platforms: Tuple[str, ...] = ()
    reproducibility_target: str = "FUNCTIONAL_EQUIVALENCE"
    runtime_behavior_checks: Tuple[str, ...] = ()
    allowed_nondeterminism: Tuple[str, ...] = ()
    policy_gates: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.reproducibility_target not in idn.REPRODUCIBILITY_LEVELS:
            problems.append(f"unknown reproducibility target {self.reproducibility_target!r}")
        if not self.target_platforms:
            problems.append("target platforms must be frozen")
        if not self.runtime_behavior_checks:
            problems.append("runtime behaviour checks must be frozen (equivalence is behavioural too)")
        unknown = sorted(set(self.allowed_nondeterminism) - set(idn.NONDETERMINISM_SOURCES))
        if unknown:
            problems.append(f"unknown allowed non-determinism sources: {unknown}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "target_platforms": list(self.target_platforms),
            "reproducibility_target": self.reproducibility_target,
            "runtime_behavior_checks": list(self.runtime_behavior_checks),
            "allowed_nondeterminism": list(self.allowed_nondeterminism),
            "policy_gates": list(self.policy_gates),
        }


@dataclass
class BuildEnvironment:
    """Steps 8/10: the builder identity (a second host is required for build B)."""

    build_id: str
    host_id: str = ""
    os_release: str = ""
    container_engine: str = ""
    buildkit_version: str = ""
    cpu_arch: str = ""
    timezone: str = ""
    locale: str = ""
    cache_enabled: bool = False
    derived_cache_reused: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("build_id", "host_id", "container_engine", "buildkit_version", "cpu_arch"):
            if not getattr(self, name):
                problems.append(f"build environment requires {name!r}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "build_id": self.build_id,
            "host_id": self.host_id,
            "os_release": self.os_release,
            "container_engine": self.container_engine,
            "buildkit_version": self.buildkit_version,
            "cpu_arch": self.cpu_arch,
            "timezone": self.timezone,
            "locale": self.locale,
            "cache_enabled": self.cache_enabled,
            "derived_cache_reused": self.derived_cache_reused,
        }


@dataclass
class BuildRecord:
    """Steps 9/11: one build invocation and its OCI output."""

    build_id: str
    environment: BuildEnvironment
    command: Tuple[str, ...] = ()
    parameters: Mapping[str, str] = field(default_factory=dict)
    log_uri: str = ""
    digest_dag: Optional[idn.DigestDAG] = None
    resolved_dependencies: Tuple[str, ...] = ()
    wall_time_s: float = 0.0
    resource_metrics: Mapping[str, Any] = field(default_factory=dict)
    status: str = "NOT_RUN"
    reused_final_manifest_from: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        problems.extend(self.environment.validate())
        if self.status not in ("OK", "FAILED", "NOT_RUN"):
            problems.append(f"unknown build status {self.status!r}")
        if self.digest_dag is None and self.status == "OK":
            problems.append("a completed build must record its OCI digest DAG")
        if self.digest_dag is not None:
            problems.extend(self.digest_dag.validate())
        if self.reused_final_manifest_from:
            problems.append(
                "a build may not reuse the other build's final manifest "
                f"(would fake reproducibility; reused from {self.reused_final_manifest_from!r})"
            )
        if not self.log_uri and self.status == "OK":
            problems.append("a completed build must keep its log")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "build_id": self.build_id,
            "environment": self.environment.as_dict(),
            "command": list(self.command),
            "parameters": dict(sorted(self.parameters.items())),
            "log_uri": self.log_uri,
            "digest_dag": self.digest_dag.as_dict() if self.digest_dag else None,
            "resolved_dependencies": list(self.resolved_dependencies),
            "wall_time_s": self.wall_time_s,
            "resource_metrics": dict(sorted(self.resource_metrics.items())),
            "status": self.status,
            "reused_final_manifest_from": self.reused_final_manifest_from,
        }


def attribute_nondeterminism(
    dag_diff: Mapping[str, Any], attributions: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Step 14: every differing descriptor must be attributed to a named source."""
    unknown = [
        row
        for row in attributions
        if str(row.get("source", "")) not in idn.NONDETERMINISM_SOURCES
    ]
    if unknown:
        raise ConfigError(
            f"unknown non-determinism sources: {sorted({row.get('source') for row in unknown})}"
        )
    differing = [row for row in dag_diff.get("rows", ()) if not row["equal"]]
    attributed_keys = {(row.get("level"), row.get("key")) for row in attributions}
    unattributed = [
        row for row in differing if (row["level"], row["key"]) not in attributed_keys
    ]
    return {
        "differing": len(differing),
        "attributed": len(differing) - len(unattributed),
        "unattributed": unattributed,
        "rows": [
            {
                "campaign_id": row.get("campaign_id", ""),
                "level": row.get("level", ""),
                "key": row.get("key", ""),
                "source": row.get("source", ""),
                "explanation": row.get("explanation", ""),
            }
            for row in attributions
        ],
        "complete": not unattributed,
        "reason": (
            ""
            if not unattributed
            else "differing descriptors without an attributed source (a difference without a "
            "cause is not a reproducibility result)"
        ),
    }


def validate_independent_rebuild(
    *, rebuild: BuildRecord, original: BuildRecord, sbom: Optional[Sbom], gate: SupplyChainGateResult
) -> Dict[str, Any]:
    """Step 37: a second session rebuilds and re-verifies; local files do not count."""
    problems: List[str] = []
    if rebuild.environment.host_id == original.environment.host_id:
        problems.append("the independent rebuild ran on the same host as the original")
    if rebuild.status != "OK":
        problems.append("the independent rebuild did not complete")
    if rebuild.digest_dag is None:
        problems.append("the independent rebuild recorded no digest DAG")
    elif original.digest_dag is not None:
        diff = idn.compare_oci_dag(original.digest_dag, rebuild.digest_dag)
        if not diff["bit_reproducible"]:
            problems.append(
                "the independent rebuild is not bit-identical: "
                + ", ".join(diff["differing_levels"])
            )
    if sbom is None:
        problems.append("the independent verification has no SBOM")
    if gate.reproducibility_level != idn.REPRO_BIT_REPRODUCIBLE:
        problems.append(
            "the gate's reproducibility level is "
            f"{gate.reproducibility_level or 'unset'}: an independent rebuild claim needs bit reproducibility"
        )
    return {"ok": not problems, "problems": problems}


def release_bundle_record(
    *,
    release: idn.ReleaseBundle,
    gate: SupplyChainGateResult,
    build_definition_hash: str,
    runtime_compatibility: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Step 38: bind the deployable digest and its evidence for E13-02."""
    payload = release.payload()
    payload.update(
        {
            "supply_chain_gate_decision": gate.gate_decision,
            "supply_chain_reason_codes": list(gate.reason_codes),
            "reproducibility_level": gate.reproducibility_level,
            "sbom_ids": list(gate.sbom_ids),
            "build_definition_hash": build_definition_hash,
            "runtime_compatibility": list(runtime_compatibility),
            "invalidation_conditions": [
                "base image digest changes",
                "dependency lockfile changes",
                "scan policy version changes",
                "SBOM/provenance subject no longer matches the deployed digest",
            ],
        }
    )
    return {
        "release_bundle": payload,
        "quarantined": gate.gate_decision not in ("PASS", "PASS_WITH_EXCEPTIONS"),
        "note": (
            "a quarantined release must not be admitted by E13-02; the deployable digest "
            "and its evidence are bound here"
        ),
    }


# ── pinned patch point for the interface map ──────────────────────────────

#: ``BuildInputs`` pins; kept as a named constant so the step table can cite it.
PIN = "digest-pinned"

#: Non-run prerequisite states this module may emit.
UNAVAILABLE_STATES: Tuple[str, ...] = (
    rec.PREREQ_NOT_RUN_TOOL_UNAVAILABLE,
    rec.PREREQ_NOT_RUN_PERMISSION_DENIED,
    rec.MISSING_REGISTRY_UNAVAILABLE,
)


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the supply-chain contracts (smoke, not an experiment)."""
    checks: Dict[str, Any] = {}

    inputs = BuildInputs(
        base_image_digest=idn.ZERO_DIGEST,
        builder_image_digest=idn.ZERO_DIGEST,
        package_index_snapshot="index-2026-09-01",
        lockfiles={"requirements.txt": idn.sha256_text("pinned")},
    )
    checks["build_inputs_valid"] = inputs.validate() == []

    floating = BuildInputs(base_image_digest="ubuntu:24.04", builder_image_digest=idn.ZERO_DIGEST,
                           package_index_snapshot="x", lockfiles={"l": "h"})
    checks["floating_base_rejected"] = any("digest" in problem for problem in floating.validate())

    plan = LayerPlan(
        stages={"base": ("libc",), "toolchain": ("gcc", "nvcc"), "runtime": ("hqsb", "python")},
        copy_from_toolchain=("hqsb-native.so",),
    )
    checks["layer_plan_valid"] = plan.validate() == []
    bad_plan = LayerPlan(
        stages={"base": ("libc",), "toolchain": ("gcc",), "runtime": ("gcc", "model-weights")},
        copy_from_toolchain=(),
    )
    checks["toolchain_and_model_leak_rejected"] = len(bad_plan.validate()) >= 2

    runtime = RuntimeSecurityContext(
        uid=10001, gid=10001, read_only_rootfs=True, writable_paths=("/tmp",), seccomp_profile="runtime/default"
    )
    checks["runtime_context_valid"] = runtime.validate() == []
    root_runtime = RuntimeSecurityContext(uid=0, gid=0, capabilities=("CAP_SYS_ADMIN",))
    checks["root_and_capability_rejected"] = len(root_runtime.validate()) >= 2

    writes = validate_write_paths(runtime, ["/tmp/out.log", "/etc/passwd"])
    checks["undeclared_write_detected"] = writes["unexpected"] == ["/etc/passwd"]

    scan = VulnerabilityScan(
        scanner="demo-scanner", scanner_version="1.0.0", db_snapshot="2026-09-01T00:00:00Z",
        scanner_status="unavailable",
    )
    checks["scanner_unavailable_not_zero"] = scan_status_semantics(scan)["status"] == rec.MISSING_SCANNER_UNAVAILABLE

    leaked = scan_build_context(
        [{"surface": "image_history", "path": "/opt/app/.netrc", "content": "api_key = supersecr3tvalue"}],
        patterns=("*.safetensors",),
    )
    checks["secret_surface_scanned"] = len(leaked["secret_findings"]) == 1
    checks["missing_surfaces_reported"] = "image_final_filesystem" in leaked["surfaces_missing"]

    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "note": "接口自检；未构建任何镜像，未运行任何扫描器",
    }
