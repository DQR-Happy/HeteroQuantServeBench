"""Release / OCI / source identity for S13 (``details/S13/README.md`` §6, §7, §22.1).

The S13 experiments deploy a *versioned* ``ReleaseBundle``, never ``latest``.  This
module is the single place where:

* a deployment reference is rejected when it is a mutable tag instead of a digest
  (:func:`require_digest_reference` — the OCI descriptor model of
  ``manifest/config/layers`` is content-addressed);
* the image digest DAG (index → platform manifest → config → layers → DiffIDs) is
  built and compared (:func:`oci_digest_dag`, :func:`compare_oci_dag`) so E13-01 can
  separate *build repeatability* / *functional equivalence* / *bit reproducibility*
  instead of claiming "we built it twice";
* the source identity (commit + dirty patch hash) and the ``ReleaseBundle`` field set
  of §6 are frozen as data, with validation that refuses an incomplete identity
  (``details/S13/README.md`` §23: "身份不完整时 release 不得 ready").

Nothing here builds, pulls or runs anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

SCHEMA_VERSION = "1.0.0"

#: ``<algorithm>:<hex>`` — the only form accepted as a deployment identity.
DIGEST_PATTERN = re.compile(r"^(?P<algorithm>[a-z0-9]+(?:[.+_-][a-z0-9]+)*):(?P<value>[a-f0-9]{32,})$")

#: Media types the DAG walk understands (unknown ones are reported, not dropped).
MEDIA_TYPE_INDEX = "application/vnd.oci.image.index.v1+json"
MEDIA_TYPE_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
MEDIA_TYPE_CONFIG = "application/vnd.oci.image.config.v1+json"
MEDIA_TYPE_LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"

#: Descriptor kinds inside an image DAG (order = closure order of §7).
DAG_LEVELS: Tuple[str, ...] = ("index", "manifest", "config", "layer")

#: Differential categories E13-01 must attribute (§7, §14 of the experiment file).
NONDETERMINISM_SOURCES: Tuple[str, ...] = (
    "timestamp",
    "random_build_id",
    "absolute_path",
    "package_index",
    "compression",
    "archive_order",
    "compiler_or_driver_jit",
    "unpinned_dependency",
    "unknown",
)

#: Reproducibility levels: three *distinct* claims that must not be conflated.
REPRO_BUILD_REPEATABILITY = "BUILD_REPEATABILITY"
REPRO_FUNCTIONAL_EQUIVALENCE = "FUNCTIONAL_EQUIVALENCE"
REPRO_BIT_REPRODUCIBLE = "BIT_REPRODUCIBLE"
REPRO_NOT_REPRODUCIBLE = "NOT_REPRODUCIBLE"

REPRODUCIBILITY_LEVELS: Tuple[str, ...] = (
    REPRO_BUILD_REPEATABILITY,
    REPRO_FUNCTIONAL_EQUIVALENCE,
    REPRO_BIT_REPRODUCIBLE,
    REPRO_NOT_REPRODUCIBLE,
)

#: Fields that ``details/S13/README.md`` §6 requires in a ReleaseBundle.
RELEASE_BUNDLE_FIELDS: Tuple[str, ...] = (
    "release_id",
    "source_commit",
    "dirty_patch_hash",
    "image_index_digest",
    "platform_image_digests",
    "build_provenance_id",
    "sbom_ids",
    "scan_policy_id",
    "service_config_hash",
    "feature_flags_hash",
    "model_artifact_id",
    "tokenizer_id",
    "quant_artifact_id",
    "engine_artifact_id",
    "kernel_bundle_id",
    "compiler_artifact_id",
    "runtime_driver_compatibility_contract",
    "workload_quality_baseline_id",
    "capacity_policy_id",
    "autoscaling_policy_id",
    "observability_schema_version",
    "deployment_template_digest",
    "rollback_parent_release_id",
    "created_at",
    "status",
)

#: Identity axes that must be present before a release may be called *ready*
#: (``details/S13/README.md`` §23 first bullet).  Optional components stay optional.
REQUIRED_IDENTITY_FIELDS: Tuple[str, ...] = (
    "release_id",
    "source_commit",
    "image_index_digest",
    "platform_image_digests",
    "service_config_hash",
    "model_artifact_id",
    "tokenizer_id",
    "deployment_template_digest",
)

RELEASE_STATUSES: Tuple[str, ...] = (
    "DRAFT",
    "SUPPLY_CHAIN_PENDING",
    "SUPPLY_CHAIN_PASSED",
    "QUARANTINED",
    "DEPLOYABLE",
    "RETIRED",
)

#: ``sha256`` example digest used by tests and templates (never a measured value).
ZERO_DIGEST = "sha256:" + "0" * 64


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(payload: Any) -> str:
    """Deterministic JSON used for every identity hash in this layer."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def hash_payload(payload: Any) -> str:
    """Hash a structured payload through its canonical JSON form."""
    return sha256_text(canonical_json(payload))


def hash_file(path: str) -> str:
    """Streaming file hash; missing files raise instead of hashing nothing."""
    if not os.path.isfile(path):
        raise ConfigError(f"cannot hash missing file: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def hash_directory(path: str, *, patterns: Sequence[str] = ("**/*",)) -> Dict[str, str]:
    """Hash a directory tree into ``{relative_path: digest}`` (sorted, symlink-safe).

    Directories and symlinks are recorded as such so that a "canonical filesystem
    diff" (E13-01 step 13) can compare path/type/hash without following links.
    """
    if not os.path.isdir(path):
        raise ConfigError(f"not a directory: {path}")
    import glob

    entries: Dict[str, str] = {}
    for pattern in patterns:
        for candidate in sorted(glob.glob(os.path.join(path, pattern), recursive=True)):
            relative = os.path.relpath(candidate, path)
            if os.path.islink(candidate):
                entries[relative] = "symlink:" + os.readlink(candidate)
            elif os.path.isdir(candidate):
                entries[relative + "/"] = "directory"
            elif os.path.isfile(candidate):
                entries[relative] = hash_file(candidate)
    return entries


# ── digests and references ─────────────────────────────────────────────────


def parse_digest(reference: str) -> Tuple[str, str]:
    """Split ``sha256:<hex>`` into ``(algorithm, value)``; malformed input raises."""
    match = DIGEST_PATTERN.match(reference or "")
    if not match:
        raise ConfigError(
            f"not a content digest: {reference!r} (expected '<algorithm>:<hex>')",
            details={"field": "digest"},
        )
    return match.group("algorithm"), match.group("value")


def is_digest(reference: str) -> bool:
    return bool(DIGEST_PATTERN.match(reference or ""))


def require_digest_reference(reference: str, *, field_name: str = "image") -> str:
    """Refuse a mutable tag where a deployment identity is required (§23 bullet 2)."""
    if not is_digest(reference):
        raise ConfigError(
            f"{field_name} must be pinned by digest, got {reference!r} "
            "(a tag is mutable and cannot be a deployment identity)",
            details={"field": field_name},
        )
    return reference


@dataclass(frozen=True)
class Descriptor:
    """One OCI content descriptor (digest + size + media type + annotations)."""

    media_type: str
    digest: str
    size: int = 0
    annotations: Mapping[str, str] = field(default_factory=dict)
    kind: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.media_type:
            problems.append("descriptor without media_type")
        try:
            parse_digest(self.digest)
        except ConfigError as exc:
            problems.append(str(exc))
        if self.size < 0:
            problems.append(f"negative size for {self.digest}")
        if self.kind and self.kind not in DAG_LEVELS:
            problems.append(f"unknown descriptor kind {self.kind!r}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "media_type": self.media_type,
            "digest": self.digest,
            "size": self.size,
            "kind": self.kind,
        }
        if self.annotations:
            payload["annotations"] = dict(sorted(self.annotations.items()))
        return payload


@dataclass
class DigestDAG:
    """The closure ``index → platform manifest → config/layers`` of one image."""

    index_digest: str
    descriptors: Tuple[Descriptor, ...] = ()
    platform_manifests: Mapping[str, str] = field(default_factory=dict)
    platform_configs: Mapping[str, str] = field(default_factory=dict)
    platform_layers: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    diff_ids: Tuple[str, ...] = ()
    history: Tuple[str, ...] = ()
    annotations: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> List[str]:
        problems: List[str] = []
        try:
            parse_digest(self.index_digest)
        except ConfigError as exc:
            problems.append(f"index: {exc}")
        for descriptor in self.descriptors:
            problems.extend(descriptor.validate())
        for platform, digest in self.platform_manifests.items():
            if not platform:
                problems.append("platform manifest entry without a platform key")
            try:
                parse_digest(digest)
            except ConfigError as exc:
                problems.append(f"manifest[{platform}]: {exc}")
        if not self.platform_manifests:
            problems.append("an image DAG needs at least one platform manifest")
        return problems

    def layer_digests(self) -> Tuple[str, ...]:
        ordered: List[str] = []
        for platform in sorted(self.platform_layers):
            ordered.extend(self.platform_layers[platform])
        return tuple(ordered)

    def closure(self) -> Tuple[str, ...]:
        """All digests reachable from the index (deduplicated, deterministic)."""
        digests: List[str] = [self.index_digest]
        digests.extend(self.platform_manifests[platform] for platform in sorted(self.platform_manifests))
        digests.extend(self.platform_configs[platform] for platform in sorted(self.platform_configs))
        digests.extend(self.layer_digests())
        digests.extend(self.diff_ids)
        seen: Dict[str, None] = {}
        for digest in digests:
            seen.setdefault(digest, None)
        return tuple(seen)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index_digest": self.index_digest,
            "descriptors": [descriptor.as_dict() for descriptor in self.descriptors],
            "platform_manifests": dict(sorted(self.platform_manifests.items())),
            "platform_configs": dict(sorted(self.platform_configs.items())),
            "platform_layers": {key: list(value) for key, value in sorted(self.platform_layers.items())},
            "diff_ids": list(self.diff_ids),
            "history": list(self.history),
            "annotations": dict(sorted(self.annotations.items())),
        }


def oci_digest_dag(
    index_digest: str,
    *,
    platform_manifests: Mapping[str, str],
    platform_configs: Mapping[str, str],
    platform_layers: Mapping[str, Sequence[str]],
    diff_ids: Sequence[str] = (),
    history: Sequence[str] = (),
    annotations: Optional[Mapping[str, str]] = None,
) -> DigestDAG:
    """Build an image DAG from an explicit OCI layout inventory."""
    descriptors: List[Descriptor] = [Descriptor(MEDIA_TYPE_INDEX, index_digest, kind="index")]
    for platform, digest in sorted(platform_manifests.items()):
        descriptors.append(Descriptor(MEDIA_TYPE_MANIFEST, digest, kind="manifest"))
        config = platform_configs.get(platform)
        if config:
            descriptors.append(Descriptor(MEDIA_TYPE_CONFIG, config, kind="config"))
        for layer in platform_layers.get(platform, ()):
            descriptors.append(Descriptor(MEDIA_TYPE_LAYER, layer, kind="layer"))
    dag = DigestDAG(
        index_digest=index_digest,
        descriptors=tuple(descriptors),
        platform_manifests=dict(platform_manifests),
        platform_configs=dict(platform_configs),
        platform_layers={key: tuple(value) for key, value in platform_layers.items()},
        diff_ids=tuple(diff_ids),
        history=tuple(history),
        annotations=dict(annotations or {}),
    )
    problems = dag.validate()
    if problems:
        raise ConfigError("invalid image digest DAG: " + "; ".join(problems))
    return dag


def compare_oci_dag(left: DigestDAG, right: DigestDAG) -> Dict[str, Any]:
    """Compare two builds layer by layer and classify the difference.

    The verdict is *not* a boolean: bit reproducibility requires every descriptor
    and DiffID to be identical, while a filesystem-equivalent rebuild may still
    differ in layers (compression, ordering) as long as the canonical filesystem
    diff is empty (E13-01 steps 12–15).
    """
    rows: List[Dict[str, Any]] = []

    def record(level: str, key: str, a: str, b: str) -> None:
        rows.append(
            {
                "level": level,
                "key": key,
                "left": a,
                "right": b,
                "equal": a == b,
            }
        )

    record("index", "index_digest", left.index_digest, right.index_digest)
    for platform in sorted(set(left.platform_manifests) | set(right.platform_manifests)):
        record(
            "manifest",
            f"manifest[{platform}]",
            left.platform_manifests.get(platform, ""),
            right.platform_manifests.get(platform, ""),
        )
        record(
            "config",
            f"config[{platform}]",
            left.platform_configs.get(platform, ""),
            right.platform_configs.get(platform, ""),
        )
        left_layers = list(left.platform_layers.get(platform, ()))
        right_layers = list(right.platform_layers.get(platform, ()))
        record("layer", f"layers[{platform}].count", str(len(left_layers)), str(len(right_layers)))
        for index in range(max(len(left_layers), len(right_layers))):
            record(
                "layer",
                f"layer[{platform}][{index}]",
                left_layers[index] if index < len(left_layers) else "",
                right_layers[index] if index < len(right_layers) else "",
            )
    for index in range(max(len(left.diff_ids), len(right.diff_ids))):
        record(
            "layer",
            f"diff_id[{index}]",
            left.diff_ids[index] if index < len(left.diff_ids) else "",
            right.diff_ids[index] if index < len(right.diff_ids) else "",
        )
    for index in range(max(len(left.history), len(right.history))):
        record(
            "history",
            f"history[{index}]",
            left.history[index] if index < len(left.history) else "",
            right.history[index] if index < len(right.history) else "",
        )

    differing_levels = sorted({row["level"] for row in rows if not row["equal"]})
    bit_reproducible = not differing_levels
    return {
        "rows": rows,
        "differing_levels": differing_levels,
        "bit_reproducible": bit_reproducible,
        "diff_counts": {level: sum(1 for row in rows if row["level"] == level and not row["equal"]) for level in DAG_LEVELS + ("history",)},
        # Never upgrade a claim: a structural diff alone cannot prove the images are
        # functionally equivalent — that needs the filesystem/package diff path.
        "level": REPRO_BIT_REPRODUCIBLE if bit_reproducible else REPRO_BUILD_REPEATABILITY,
        "next_step": (
            ""
            if bit_reproducible
            else "attribute each differing descriptor and record the canonical "
            "filesystem/package diff before claiming functional equivalence"
        ),
    }


def reproducibility_verdict(
    *,
    build_a_ok: bool,
    build_b_ok: bool,
    dag_diff: Mapping[str, Any],
    filesystem_diff_rows: int,
    package_diff_rows: int,
    nondeterminism_sources: Sequence[str] = (),
) -> Dict[str, Any]:
    """Classify a rebuild pair into the four §7 levels (no silent upgrade)."""
    unknown = sorted(set(nondeterminism_sources) - set(NONDETERMINISM_SOURCES))
    if unknown:
        raise ConfigError(f"unknown non-determinism sources: {unknown}")
    if not (build_a_ok and build_b_ok):
        level = REPRO_NOT_REPRODUCIBLE
        why = "at least one build did not complete successfully"
    elif dag_diff.get("bit_reproducible"):
        level = REPRO_BIT_REPRODUCIBLE
        why = "all index/manifest/config/layer digests and DiffIDs are identical"
    elif filesystem_diff_rows == 0 and package_diff_rows == 0 and nondeterminism_sources:
        level = REPRO_FUNCTIONAL_EQUIVALENCE
        why = "digests differ but the canonical filesystem/package diff is empty"
    elif filesystem_diff_rows == 0 and package_diff_rows == 0:
        level = REPRO_BUILD_REPEATABILITY
        why = (
            "digests differ and no non-determinism source has been attributed yet; "
            "equivalence is not claimed until the diff is explained"
        )
    else:
        level = REPRO_NOT_REPRODUCIBLE
        why = "filesystem/package differences remain between the two builds"
    return {
        "level": level,
        "reason": why,
        "bit_reproducible": bool(dag_diff.get("bit_reproducible")),
        "differing_levels": list(dag_diff.get("differing_levels", [])),
        "filesystem_diff_rows": filesystem_diff_rows,
        "package_diff_rows": package_diff_rows,
        "nondeterminism_sources": list(nondeterminism_sources),
    }


def canonical_filesystem_diff(
    left: Mapping[str, str], right: Mapping[str, str]
) -> List[Dict[str, Any]]:
    """Canonical path/type/hash diff of two trees (E13-01 step 13).

    Differences are *kept*: the caller records them in
    ``diff/filesystem_packages.parquet`` instead of dropping "equal" rows silently
    producing a false clean bill of health.
    """
    differences: List[Dict[str, Any]] = []
    for path in sorted(set(left) | set(right)):
        left_value = left.get(path)
        right_value = right.get(path)
        if left_value == right_value:
            continue
        state = "ADDED" if left_value is None else "REMOVED" if right_value is None else "CHANGED"
        differences.append(
            {
                "path": path,
                "state": state,
                "left": left_value or "",
                "right": right_value or "",
                "left_type": _tree_entry_type(left_value),
                "right_type": _tree_entry_type(right_value),
            }
        )
    return differences


def _tree_entry_type(value: Optional[str]) -> str:
    if value is None:
        return "absent"
    if value == "directory":
        return "directory"
    if value.startswith("symlink:"):
        return "symlink"
    return "file"


# ── source identity ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SourceIdentity:
    """``commit`` + ``dirty`` state; a dirty release is explicit, never silent."""

    commit: str
    dirty: bool = False
    dirty_files: Tuple[str, ...] = ()
    diff_sha256: str = ""
    submodules: Mapping[str, str] = field(default_factory=dict)
    lockfiles: Mapping[str, str] = field(default_factory=dict)
    generated_sources: Tuple[str, ...] = ()
    version_tag: str = ""

    def dirty_patch_hash(self) -> str:
        return self.diff_sha256 if self.dirty else ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.commit:
            problems.append("source identity requires a commit")
        if self.dirty and not self.diff_sha256:
            problems.append(
                "a dirty source tree must record the diff hash (a dirty release may only be "
                "experimental and must be auditable)"
            )
        if self.dirty and not self.dirty_files:
            problems.append("a dirty source tree must list the modified files")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "commit": self.commit,
            "dirty": self.dirty,
            "dirty_files": list(self.dirty_files),
            "dirty_patch_hash": self.dirty_patch_hash(),
            "diff_sha256": self.diff_sha256,
            "submodules": dict(sorted(self.submodules.items())),
            "lockfiles": dict(sorted(self.lockfiles.items())),
            "generated_sources": list(self.generated_sources),
            "version_tag": self.version_tag,
        }


def source_identity(
    commit: str,
    *,
    dirty_files: Sequence[str] = (),
    diff_text: str = "",
    submodules: Optional[Mapping[str, str]] = None,
    lockfiles: Optional[Mapping[str, str]] = None,
    generated_sources: Sequence[str] = (),
    version_tag: str = "",
) -> SourceIdentity:
    """Freeze the source identity of a release build (E13-01 step 2)."""
    identity = SourceIdentity(
        commit=commit,
        dirty=bool(dirty_files),
        dirty_files=tuple(sorted(dirty_files)),
        diff_sha256=sha256_text(diff_text) if dirty_files and diff_text else "",
        submodules=dict(submodules or {}),
        lockfiles=dict(lockfiles or {}),
        generated_sources=tuple(generated_sources),
        version_tag=version_tag,
    )
    problems = identity.validate()
    if problems:
        raise ConfigError("invalid source identity: " + "; ".join(problems))
    return identity


# ── release bundle ─────────────────────────────────────────────────────────


@dataclass
class ReleaseBundle:
    """The deployment identity unit of §6 (models never enter the runtime image)."""

    release_id: str = ""
    source_commit: str = ""
    dirty_patch_hash: str = ""
    image_index_digest: str = ""
    platform_image_digests: Mapping[str, str] = field(default_factory=dict)
    build_provenance_id: str = ""
    sbom_ids: Tuple[str, ...] = ()
    scan_policy_id: str = ""
    service_config_hash: str = ""
    feature_flags_hash: str = ""
    model_artifact_id: str = ""
    tokenizer_id: str = ""
    quant_artifact_id: str = ""
    engine_artifact_id: str = ""
    kernel_bundle_id: str = ""
    compiler_artifact_id: str = ""
    runtime_driver_compatibility_contract: str = ""
    workload_quality_baseline_id: str = ""
    capacity_policy_id: str = ""
    autoscaling_policy_id: str = ""
    observability_schema_version: str = ""
    deployment_template_digest: str = ""
    rollback_parent_release_id: str = ""
    created_at: str = ""
    status: str = "DRAFT"
    model_embedded_in_image: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.status not in RELEASE_STATUSES:
            problems.append(f"unknown release status {self.status!r}")
        for name in REQUIRED_IDENTITY_FIELDS:
            value = getattr(self, name)
            if not value:
                problems.append(f"missing required identity field {name!r}")
        for field_name in ("image_index_digest", "deployment_template_digest"):
            value = getattr(self, field_name)
            if value and not is_digest(value):
                problems.append(f"{field_name} must be a digest, got {value!r}")
        for platform, digest in self.platform_image_digests.items():
            if not platform:
                problems.append("platform_image_digests has an empty platform key")
            if not is_digest(digest):
                problems.append(f"platform image for {platform!r} is not a digest: {digest!r}")
        if self.model_embedded_in_image:
            problems.append(
                "model weights must not be embedded in the generic runtime image "
                "(S13 §3: model and image change independently)"
            )
        # The provenance/signature relation of §8: an SBOM id is not a signature.
        if self.status in ("DEPLOYABLE", "SUPPLY_CHAIN_PASSED") and not self.sbom_ids:
            problems.append("a deployable release needs at least one SBOM id")
        return problems

    def identity_complete(self) -> bool:
        return not [
            name for name in REQUIRED_IDENTITY_FIELDS if not getattr(self, name)
        ]

    def payload(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "release_id": self.release_id,
            "source_commit": self.source_commit,
            "dirty_patch_hash": self.dirty_patch_hash,
            "image_index_digest": self.image_index_digest,
            "platform_image_digests": dict(sorted(self.platform_image_digests.items())),
            "build_provenance_id": self.build_provenance_id,
            "sbom_ids": list(self.sbom_ids),
            "scan_policy_id": self.scan_policy_id,
            "service_config_hash": self.service_config_hash,
            "feature_flags_hash": self.feature_flags_hash,
            "model_artifact_id": self.model_artifact_id,
            "tokenizer_id": self.tokenizer_id,
            "quant_artifact_id": self.quant_artifact_id,
            "engine_artifact_id": self.engine_artifact_id,
            "kernel_bundle_id": self.kernel_bundle_id,
            "compiler_artifact_id": self.compiler_artifact_id,
            "runtime_driver_compatibility_contract": self.runtime_driver_compatibility_contract,
            "workload_quality_baseline_id": self.workload_quality_baseline_id,
            "capacity_policy_id": self.capacity_policy_id,
            "autoscaling_policy_id": self.autoscaling_policy_id,
            "observability_schema_version": self.observability_schema_version,
            "deployment_template_digest": self.deployment_template_digest,
            "rollback_parent_release_id": self.rollback_parent_release_id,
            "created_at": self.created_at,
            "status": self.status,
        }

    def bundle_hash(self) -> str:
        return hash_payload(self.payload())

    def as_dict(self) -> Dict[str, Any]:
        return self.payload()


def release_bundle(**kwargs: Any) -> ReleaseBundle:
    """Construct a bundle and fail fast on an incomplete/invalid identity."""
    bundle = ReleaseBundle(**kwargs)
    problems = bundle.validate()
    if problems:
        raise ConfigError("invalid ReleaseBundle: " + "; ".join(problems))
    return bundle


def validate_release_change_scope(
    control: ReleaseBundle, candidate: ReleaseBundle, *, declared_changes: Sequence[str]
) -> Dict[str, Any]:
    """E13-10 step 1 / §23: only pre-registered differences may exist.

    ``declared_changes`` names the identity fields the canary is allowed to vary
    (e.g. ``image_index_digest`` for a code change).  Everything else must match
    between control and candidate, otherwise the comparison is not attributable.
    """
    allowed = set(declared_changes)
    unknown = sorted(allowed - set(RELEASE_BUNDLE_FIELDS) - {"platform_image_digests"})
    if unknown:
        raise ConfigError(f"declared change scope names unknown fields: {unknown}")
    left = control.payload()
    right = candidate.payload()
    differences: List[Dict[str, Any]] = []
    for name in RELEASE_BUNDLE_FIELDS:
        if name in ("release_id", "created_at", "status"):
            continue
        if left.get(name) != right.get(name):
            differences.append(
                {
                    "field": name,
                    "control": left.get(name),
                    "candidate": right.get(name),
                    "declared": name in allowed,
                }
            )
    undeclared = [row for row in differences if not row["declared"]]
    return {
        "declared_changes": sorted(allowed),
        "differences": differences,
        "undeclared_differences": undeclared,
        "single_variable_ok": not undeclared,
        "reason": (
            ""
            if not undeclared
            else "control and candidate differ in undeclared fields: "
            + ", ".join(row["field"] for row in undeclared)
        ),
    }


def validate_rollback_chain(
    bundles: Iterable[ReleaseBundle], *, active_release_id: str
) -> Dict[str, Any]:
    """Walk the ``rollback_parent_release_id`` chain to a known-good root.

    A rollback target that is missing (or itself references an unknown release)
    blocks the production rollback claim (§11 gate 8 of E13-04).
    """
    by_id = {bundle.release_id: bundle for bundle in bundles}
    if active_release_id not in by_id:
        raise ConfigError(f"unknown active release {active_release_id!r}")
    chain: List[str] = []
    missing: List[str] = []
    seen: Dict[str, None] = {}
    current = by_id[active_release_id]
    while True:
        if current.release_id in seen:
            missing.append(f"cycle at {current.release_id}")
            break
        seen.setdefault(current.release_id, None)
        chain.append(current.release_id)
        parent = current.rollback_parent_release_id
        if not parent:
            break
        if parent not in by_id:
            missing.append(parent)
            break
        current = by_id[parent]
    known_good = [release_id for release_id in chain if by_id[release_id].status == "DEPLOYABLE"]
    return {
        "active_release_id": active_release_id,
        "chain": chain,
        "missing_targets": missing,
        "known_good_targets": known_good,
        "rollback_available": bool(known_good) and not missing,
    }


def platform_matrix(bundles: Iterable[ReleaseBundle]) -> Dict[str, Any]:
    """Aggregate which platforms each release can run on (index → digests)."""
    rows: List[Dict[str, Any]] = []
    for bundle in bundles:
        for platform, digest in sorted(bundle.platform_image_digests.items()):
            rows.append(
                {
                    "release_id": bundle.release_id,
                    "platform": platform,
                    "image_digest": digest,
                    "rollback_parent": bundle.rollback_parent_release_id,
                }
            )
    platforms = sorted({row["platform"] for row in rows})
    return {"rows": rows, "platforms": platforms, "release_count": len({row["release_id"] for row in rows})}
