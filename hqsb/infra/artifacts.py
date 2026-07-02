"""E13-04: model artifact download, cache, atomic activation, switch and rollback.

Implements ``details/S13/E13-04_*.md`` as data:

* the content-addressed cache key of §4.1 (model root **plus** tokenizer, quant,
  engine/kernel artifact and target ABI — a name is not an identity);
* staging → verify → atomic commit, with the ``complete`` marker written only
  after the data is durable (§4.2);
* leases/pins and the GC plan of §4.4 (active/inflight/canary/rollback-retained
  artifacts are never collected);
* the activation linearization point and per-request version consistency
  (§4.3, gate §11) including KV/prefix isolation;
* rollback to a known-good artifact with its verification gate;
* the fault injections of steps 28–35 (missing/tampered shard, tokenizer/quant
  mismatch, interrupted download, storage timeout, ENOSPC, dead lock owner,
  concurrent GC/load/switch).

Nothing here downloads or deletes anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.infra import identity as idn
from hqsb.infra import records as rec

EXPERIMENT_ID = "E13-04"
TITLE = "模型制品下载、Cache、原子激活、版本切换与回滚"
CLAIM_BOUNDARY = (
    "本实验通过证明模型制品生命周期与版本一致性；不证明 rolling 请求无损、canary 判断正确或"
    "跨区域灾备（由 E13-05/E13-10 验证）。"
)

SCHEMA_VERSION = "1.0.0"

#: Lifecycle states come from ``records`` so the three state machines stay one source.
LIFECYCLE_STATES: Tuple[str, ...] = rec.ARTIFACT_LIFECYCLE_STATES
STATE_MACHINE = rec.ARTIFACT_STATE_MACHINE

#: Cache key components (§4.1) — every one of them changes the content identity.
CACHE_KEY_FIELDS: Tuple[str, ...] = (
    "model_root",
    "tokenizer_id",
    "config_id",
    "quant_artifact_id",
    "engine_artifact_id",
    "kernel_bundle_id",
    "target_arch",
    "runtime_abi",
)

#: Layout of the cache (§4.2/§4.4).
CACHE_AREAS: Tuple[str, ...] = ("staging", "verified", "quarantine", "active_index", "leases", "gc_metadata")

#: Verification checks of steps 10–11.
VERIFICATION_CHECKS: Tuple[str, ...] = (
    "file_size",
    "file_hash",
    "manifest_hash",
    "aggregate_root",
    "path_safety",
    "duplicate_files",
    "extra_files",
    "permissions",
)

COMPATIBILITY_CHECKS: Tuple[str, ...] = (
    "tokenizer_matches_config",
    "config_matches_weights",
    "precision_supported",
    "quant_artifact_matches_weights",
    "engine_target_matches_arch",
    "kernel_bundle_abi_matches_runtime",
    "runtime_driver_contract",
)

#: Fault injections of steps 28–35.
FAULT_CASES: Tuple[str, ...] = (
    "MISSING_SHARD",
    "TAMPERED_SHARD",
    "TOKENIZER_MISMATCH",
    "CONFIG_MISMATCH",
    "QUANT_MISMATCH",
    "DOWNLOAD_INTERRUPTED",
    "PROCESS_CRASH_BEFORE_COMMIT",
    "STORAGE_TIMEOUT",
    "STORAGE_STALE_READ",
    "DISK_FULL_ENOSPC",
    "LOCK_OWNER_DEATH",
    "CONCURRENT_GC_WITH_LOAD",
    "CONCURRENT_GC_WITH_SWITCH",
    "CONCURRENT_GC_WITH_ROLLBACK",
)

#: Which lease holders protect an artifact from GC (§4.4).
PIN_KINDS: Tuple[str, ...] = ("active", "inflight", "canary_candidate", "rollback_retained", "pinned_manual")

#: Reasons GC may remove an artifact.
GC_REASONS: Tuple[str, ...] = (
    "UNUSED_BEYOND_RETENTION",
    "QUARANTINED",
    "SUPERSEDED_AND_UNPINNED",
    "FAILED_VERIFICATION",
    "CAMPAIGN_CLEANUP",
)


# ── cache identity ────────────────────────────────────────────────────────


@dataclass
class CacheKey:
    """The content-addressed cache key of §4.1 (never a model *name*)."""

    artifact_id: str
    content_root: str = ""
    model_root: str = ""
    tokenizer_id: str = ""
    config_id: str = ""
    quant_artifact_id: str = ""
    engine_artifact_id: str = ""
    kernel_bundle_id: str = ""
    target_arch: str = ""
    runtime_abi: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("artifact_id", "model_root", "tokenizer_id", "config_id", "target_arch", "runtime_abi"):
            if not getattr(self, name):
                problems.append(
                    f"cache key requires {name!r}: caching by model name alone lets different "
                    "tokenizer/precision/engine variants collide"
                )
        if self.content_root and not idn.is_digest(self.content_root):
            problems.append("content root must be a digest")
        return problems

    def key(self) -> str:
        problems = self.validate()
        if problems:
            raise ConfigError("invalid cache key: " + "; ".join(problems))
        return idn.hash_payload({name: getattr(self, name) for name in CACHE_KEY_FIELDS})

    def as_dict(self) -> Dict[str, Any]:
        payload = {name: getattr(self, name) for name in CACHE_KEY_FIELDS}
        payload["artifact_id"] = self.artifact_id
        payload["cache_key"] = self.key() if not self.validate() else ""
        return payload


def cache_layout_manifest(*, root: str, quota_bytes: int, watermark_ratio: float) -> Dict[str, Any]:
    """Step 3: the cache layout and its watermarks (a full disk is a design input)."""
    if not root.startswith("/"):
        raise ConfigError("cache root must be an absolute path")
    if quota_bytes <= 0:
        raise ConfigError("cache quota must be positive (no unbounded cache)")
    if not 0.0 < watermark_ratio < 1.0:
        raise ConfigError("watermark ratio must be inside (0, 1)")
    return {
        "root": root,
        "areas": list(CACHE_AREAS),
        "quota_bytes": quota_bytes,
        "gc_watermark": int(quota_bytes * watermark_ratio),
        "permissions": {"staging": "0700", "verified": "0500", "quarantine": "0700"},
        "note": "quarantine and staging are never part of the active path",
    }


# ── lifecycle events ─────────────────────────────────────────────────────


@dataclass
class ArtifactLifecycleEvent:
    """§10 ``ArtifactLifecycleEvent``."""

    event_id: str
    attempt_id: str
    artifact_id: str
    state_to: str
    content_root: str = ""
    release_id: str = ""
    model_artifact_id: str = ""
    tokenizer_id: str = ""
    quant_artifact_id: str = ""
    engine_artifact_id: str = ""
    kernel_bundle_id: str = ""
    cache_location_id: str = ""
    state_from: str = ""
    timestamp: str = ""
    actor: str = ""
    generation: str = ""
    lock_lease_id: str = ""
    bytes_or_chunk: int = 0
    etag: str = ""
    verification_status: str = ""
    active_or_inactive: str = "inactive"
    request_or_session_refs: Tuple[str, ...] = ()
    status: str = ""
    error_id: str = ""
    retry: int = 0
    durability_status: str = ""
    evidence_refs: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.state_to not in LIFECYCLE_STATES:
            problems.append(f"unknown artifact state {self.state_to!r}")
        if self.state_from and not STATE_MACHINE.allowed(self.state_from, self.state_to):
            problems.append(f"illegal artifact transition {self.state_from} -> {self.state_to}")
        if self.state_from in ("VERIFYING", "VERIFIED_IMMUTABLE") and not self.verification_status:
            problems.append("verification status must be recorded for verify/load transitions")
        if self.state_to == "VERIFIED_IMMUTABLE" and self.durability_status not in ("durable", ""):
            problems.append(
                "a verified entry may only be committed after the data is durable "
                "(marker-before-data is the classic partial-cache bug)"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "attempt_id": self.attempt_id,
            "artifact_id": self.artifact_id,
            "content_root": self.content_root,
            "cache_location_id": self.cache_location_id,
            "state_from": self.state_from,
            "state_to": self.state_to,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "generation": self.generation,
        }


def _result(constraint_id: str, findings: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {"constraint_id": constraint_id, "ok": not findings, "findings": list(findings)}


def assert_loadable(state: str) -> Dict[str, Any]:
    """§3/§5 invariant: an unverified or partial artifact is never loadable."""
    findings: List[Dict[str, Any]] = []
    if state not in rec.LOADABLE_ARTIFACT_STATES:
        findings.append(
            {
                "reason_code": "UNVERIFIED_ARTIFACT_LOAD",
                "state": state,
                "detail": "only a verified/committed artifact may be loaded by the runtime",
            }
        )
    return _result("A-01", findings)


def assert_servable(state: str, *, quality_passed: bool) -> Dict[str, Any]:
    """Only a quality-passed ``ACTIVE`` artifact may serve traffic."""
    findings: List[Dict[str, Any]] = []
    if state not in rec.SERVABLE_ARTIFACT_STATES:
        findings.append(
            {"reason_code": "NON_ACTIVE_SERVED", "state": state, "detail": "only ACTIVE may serve"}
        )
    if not quality_passed:
        findings.append(
            {
                "reason_code": "QUALITY_NOT_PASSED_SERVED",
                "state": state,
                "detail": "activation requires the quality/capacity probe to pass",
            }
        )
    return _result("A-02", findings)


# ── staging, verification, atomic commit ─────────────────────────────────


@dataclass
class StagingDownload:
    """Step 9/17/30: an attempt-scoped staging download (never the active path)."""

    attempt_id: str
    artifact_id: str
    uri: str = ""
    staging_path: str = ""
    active_path: str = ""
    bytes_expected: int = 0
    bytes_downloaded: int = 0
    chunks: int = 0
    retries: int = 0
    etag: str = ""
    interrupted: bool = False
    resumed: bool = False
    status: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.staging_path or not self.active_path:
            problems.append("staging and active paths must both be declared")
        elif self.staging_path == self.active_path:
            problems.append(
                "downloads must not write the active path directly (a partial file would be a cache hit)"
            )
        if not self.uri:
            problems.append("artifact URI must be recorded")
        if self.bytes_downloaded > self.bytes_expected > 0:
            problems.append("downloaded more bytes than the manifest declares")
        if self.interrupted and not self.status:
            problems.append("an interrupted download must record its terminal status")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "artifact_id": self.artifact_id,
            "uri": self.uri,
            "staging_path": self.staging_path,
            "bytes_expected": self.bytes_expected,
            "bytes_downloaded": self.bytes_downloaded,
            "chunks": self.chunks,
            "retries": self.retries,
            "etag": self.etag,
            "interrupted": self.interrupted,
            "resumed": self.resumed,
            "status": self.status,
        }


@dataclass
class FileVerification:
    path: str
    expected_hash: str = ""
    observed_hash: str = ""
    expected_size: int = 0
    observed_size: int = 0
    status: str = ""
    first_bad_object: str = ""

    def ok(self) -> bool:
        return (
            self.expected_hash == self.observed_hash
            and self.expected_size == self.observed_size
            and self.status in ("OK", "PASS")
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": "",
            "file": self.path,
            "expected_hash": self.expected_hash,
            "observed_hash": self.observed_hash,
            "status": self.status,
            "first_bad_object": self.first_bad_object,
        }


def verify_download(
    *,
    download: StagingDownload,
    files: Sequence[FileVerification],
    aggregate_root_expected: str,
    aggregate_root_observed: str,
    manifest_expected: str = "",
    manifest_observed: str = "",
) -> Dict[str, Any]:
    """Steps 10–12: per-file + aggregate verification before any commit."""
    problems = list(download.validate())
    bad = [entry for entry in files if not entry.ok()]
    if bad:
        problems.append(
            "files failed verification; the first bad object is "
            f"{bad[0].path} ({bad[0].status or 'hash/size mismatch'})"
        )
    if aggregate_root_expected != aggregate_root_observed:
        problems.append(
            f"aggregate root mismatch: expected {aggregate_root_expected} observed {aggregate_root_observed}"
        )
    if manifest_expected and manifest_expected != manifest_observed:
        problems.append("manifest hash mismatch (the artifact describes different content)")
    if download.bytes_downloaded != download.bytes_expected and download.bytes_expected:
        problems.append(
            f"incomplete download: {download.bytes_downloaded}/{download.bytes_expected} bytes"
        )
    return {
        "attempt_id": download.attempt_id,
        "files": len(files),
        "files_failed": len(bad),
        "first_bad_object": bad[0].path if bad else "",
        "problems": problems,
        "verification_status": "OK" if not problems else "FAILED",
        "may_commit": not problems,
    }


# ── compatibility and atomic commit ──────────────────────────────────────


def check_compatibility(
    *, cache_key: CacheKey, observed: Mapping[str, str], required_checks: Sequence[str] = COMPATIBILITY_CHECKS
) -> Dict[str, Any]:
    """Step 11: tokenizer/config/precision/quant/engine/ABI are semantic gates."""
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for check in required_checks:
        if check not in COMPATIBILITY_CHECKS:
            raise ConfigError(f"unknown compatibility check {check!r}")
        expected = _compat_expectation(cache_key, check)
        actual = observed.get(check, "")
        status = "OK" if (expected == "" or actual == expected) else "MISMATCH"
        if status == "MISMATCH":
            problems.append(f"{check}: expected {expected}, observed {actual}")
        rows.append(
            {
                "attempt_id": observed.get("attempt_id", ""),
                "artifact_id": cache_key.artifact_id,
                "check": check,
                "expected": expected,
                "observed": actual,
                "status": status,
            }
        )
    return {
        "artifact_id": cache_key.artifact_id,
        "rows": rows,
        "problems": problems,
        "status": "COMPATIBLE" if not problems else "QUARANTINE",
    }


def _compat_expectation(cache_key: CacheKey, check: str) -> str:
    mapping = {
        "tokenizer_matches_config": cache_key.tokenizer_id,
        "config_matches_weights": cache_key.config_id,
        "precision_supported": "",
        "quant_artifact_matches_weights": cache_key.quant_artifact_id,
        "engine_target_matches_arch": cache_key.target_arch,
        "kernel_bundle_abi_matches_runtime": cache_key.runtime_abi,
        "runtime_driver_contract": "",
    }
    return mapping.get(check, "")


@dataclass
class AtomicCommit:
    """Step 12: staging → verified via rename/metadata transaction, then the marker."""

    attempt_id: str
    artifact_id: str
    staging_path: str = ""
    verified_path: str = ""
    marker: str = ""
    data_durable: bool = False
    marker_written_after_data: bool = True
    rescanned_after_commit: bool = False
    committed_at: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.data_durable:
            problems.append("commit requires durable data (fsync/object-store durability)")
        if not self.marker_written_after_data:
            problems.append("the complete marker must be written after the data is durable")
        if not self.rescanned_after_commit:
            problems.append("the committed entry must be re-scanned to confirm the marker matches the content")
        if self.staging_path and self.staging_path == self.verified_path:
            problems.append("verified and staging paths must differ (no in-place verification)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event_id": f"commit-{self.attempt_id}",
            "attempt_id": self.attempt_id,
            "artifact_id": self.artifact_id,
            "content_root": "",
            "state_from": "VERIFYING",
            "state_to": "VERIFIED_IMMUTABLE",
            "timestamp": self.committed_at,
            "actor": "artifact-manager",
            "generation": "",
        }


def atomic_commit(
    *,
    commit: AtomicCommit,
    verification: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    cache_key: CacheKey,
) -> Dict[str, Any]:
    """Steps 12–13: commit only after verification *and* compatibility pass."""
    problems = list(commit.validate())
    problems.extend(cache_key.validate())
    if not verification.get("may_commit"):
        problems.append("verification did not pass; a partial/corrupt artifact may never be committed")
    if compatibility.get("status") != "COMPATIBLE":
        problems.append("compatibility failed; the artifact goes to quarantine instead of the verified store")
    return {
        "artifact_id": commit.artifact_id,
        "cache_key": cache_key.key() if not cache_key.validate() else "",
        "committed": not problems,
        "state": "VERIFIED_IMMUTABLE" if not problems else "QUARANTINED",
        "problems": problems,
    }


# ── leases, pins and garbage collection ──────────────────────────────────


@dataclass
class LeaseRecord:
    """Step 33: a lease with an owner and an expiry (a dead owner must not deadlock)."""

    lease_id: str
    artifact_id: str
    holder: str = ""
    state: str = "HELD"
    acquired_at: str = ""
    expires_at: str = ""
    released: bool = False
    pin_kind: str = ""
    crashed_holder_recovered: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("lease_id", "artifact_id", "holder", "acquired_at", "expires_at"):
            if not getattr(self, name):
                problems.append(f"lease requires {name!r}")
        if self.pin_kind and self.pin_kind not in PIN_KINDS:
            problems.append(f"unknown pin kind {self.pin_kind!r}")
        if self.state == "EXPIRED" and not self.crashed_holder_recovered:
            problems.append(
                "an expired lease must record whether a surviving owner recovered it "
                "(otherwise double-commit or permanent deadlock is unproven)"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "artifact_id": self.artifact_id,
            "holder": self.holder,
            "state": self.state,
            "acquired_at": self.acquired_at,
            "expires_at": self.expires_at,
            "released": self.released,
        }


def single_flight_outcome(attempts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 18: concurrent replicas must produce **one** verified entry, not a race."""
    commits = [row for row in attempts if row.get("committed")]
    keys = {row.get("cache_key", "") for row in commits}
    waiters_failed = [row for row in attempts if row.get("role") == "waiter" and row.get("failed_without_propagation")]
    problems: List[str] = []
    if len(commits) > 1:
        problems.append(
            f"{len(commits)} concurrent commits for the same artifact (single-flight/lock not enforced)"
        )
    if len(keys) > 1:
        problems.append("concurrent attempts produced different cache keys for the same artifact")
    if waiters_failed:
        problems.append("a waiter failed without propagating the failure to the caller")
    partial_hits = [row for row in attempts if row.get("partial_hit")]
    if partial_hits:
        problems.append("a partial staging entry was treated as a cache hit")
    return {
        "attempts": len(attempts),
        "commits": len(commits),
        "distinct_keys": len(keys),
        "partial_hits": len(partial_hits),
        "ok": not problems,
        "problems": problems,
    }


def gc_plan(
    *, entries: Sequence[Mapping[str, Any]], watermark_bytes: int, used_bytes: int
) -> Dict[str, Any]:
    """Step 34: dry-run GC — only unpinned, eligible entries may be removed."""
    rows: List[Dict[str, Any]] = []
    protected: List[Dict[str, Any]] = []
    for entry in entries:
        pins = [pin for pin in entry.get("pins", ()) or () if pin in PIN_KINDS]
        eligible = (
            entry.get("state") in ("CACHED_INACTIVE", "QUARANTINED")
            and not pins
            and bool(entry.get("reason"))
        )
        row = {
            "gc_run_id": entry.get("gc_run_id", ""),
            "artifact_id": entry.get("artifact_id", ""),
            "state": entry.get("state", ""),
            "pinned": bool(pins),
            "eligible": eligible,
            "reason": entry.get("reason", "") or ("pinned:" + ",".join(sorted(pins)) if pins else ""),
        }
        rows.append(row)
        if pins:
            protected.append(row)
    reclaimable = sum(int(entry.get("bytes", 0) or 0) for entry, row in zip(entries, rows) if row["eligible"])
    return {
        "rows": rows,
        "protected": protected,
        "reclaimable_bytes": reclaimable,
        "watermark_bytes": watermark_bytes,
        "used_bytes": used_bytes,
        "needs_gc": used_bytes >= watermark_bytes,
        "note": "an active/inflight/canary/rollback artifact must never be collected",
    }


def gc_safety_check(
    *, deleted: Sequence[str], protected: Sequence[str], active: str, rollback_target: str
) -> Dict[str, Any]:
    """Steps 34/35: use-after-delete and lost rollback targets are hard failures."""
    problems: List[str] = []
    deleted_set = set(deleted)
    for protected_id in protected:
        if protected_id in deleted_set:
            problems.append(f"GC deleted a pinned artifact: {protected_id}")
    if active and active in deleted_set:
        problems.append("GC deleted the active artifact")
    if rollback_target and rollback_target in deleted_set:
        problems.append("GC deleted the known-good rollback target (production rollback would be impossible)")
    return {
        "deleted": sorted(deleted_set),
        "problems": problems,
        "ok": not problems,
        "reason": "" if not problems else "; ".join(problems),
    }


# ── activation, switch and rollback ──────────────────────────────────────


@dataclass
class ActivationGeneration:
    """Step 16/22: the atomic routing generation and its linearization instant."""

    generation_id: str
    artifact_id: str
    activation_mode: str = ""
    linearization_ts: str = ""
    command_returned_ts: str = ""
    active: bool = False
    quality_passed: bool = False
    capacity_available: bool = False
    request_or_session_refs: Tuple[str, ...] = ()
    retiring_refs: Tuple[str, ...] = ()

    def validate(self, *, prior: Optional["ActivationGeneration"] = None) -> List[str]:
        problems: List[str] = []
        if self.activation_mode not in ("POD_ROLLING", "IN_PROCESS_SLOT"):
            problems.append(
                f"activation mode {self.activation_mode!r} must be declared "
                "(pod-level rolling and in-process slot switching have different inflight semantics)"
            )
        if not self.linearization_ts:
            problems.append("the linearization instant must be recorded (the command return is not the effect)")
        if not self.quality_passed:
            problems.append("a version may only become ACTIVE after warmup/quality/capacity passed")
        if not self.capacity_available:
            problems.append("activation requires enough capacity to serve the new version")
        if prior is not None and prior.artifact_id == self.artifact_id:
            problems.append("the new generation points at the same artifact (a no-op switch is not a switch)")
        if self.command_returned_ts and self.linearization_ts and self.command_returned_ts < self.linearization_ts:
            problems.append("the command returned before the linearization instant: the effect is unverified")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "artifact_id": self.artifact_id,
            "linearization_ts": self.linearization_ts,
            "active": self.active,
            "request_or_session_refs": list(self.request_or_session_refs),
        }


def activate_version(
    *,
    cache_key: CacheKey,
    commit: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    warmup: WarmupResult,
    capacity_available: bool,
    prior: Optional[ActivationGeneration],
    generation_id: str,
    linearization_ts: str,
    command_returned_ts: str = "",
    existing_active_lease_ids: Sequence[str] = (),
    retired_lease_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    """Step 15/16/21/25: activate a version only through the §11 gate.

    The gate is data-driven so it can be applied to both the pod-rolling and the
    in-process-slot activation modes; it returns the (unwritten) generation record
    plus the reasons a switch would be refused.
    """
    problems: List[str] = list(commit.get("problems", []) or [])
    problems.extend(cache_key.validate())
    if not commit.get("committed"):
        problems.append("the candidate version is not committed to the verified store")
    if compatibility.get("status") != "COMPATIBLE":
        problems.append("the candidate version did not pass the compatibility gate")
    problems.extend(warmup.validate())
    if not capacity_available:
        problems.append("not enough capacity to serve the candidate version")
    generation = ActivationGeneration(
        generation_id=generation_id,
        artifact_id=cache_key.artifact_id,
        activation_mode="POD_ROLLING" if commit.get("committed") else "IN_PROCESS_SLOT",
        linearization_ts=linearization_ts,
        command_returned_ts=command_returned_ts,
        active=True,
        quality_passed=warmup.quality_status == "pass",
        capacity_available=capacity_available,
        retiring_refs=tuple(retired_lease_ids),
    )
    problems.extend(generation.validate(prior=prior))
    rollback_target_retained = bool(prior) and (prior.artifact_id not in set(retired_lease_ids))
    if not rollback_target_retained:
        problems.append(
            "the previous known-good version must keep its lease/pin during the activation window "
            "(otherwise a rollback has no target)"
        )
    if not existing_active_lease_ids:
        problems.append("the activation must hold a lease/pin for the new active generation")
    return {
        "artifact_id": cache_key.artifact_id,
        "generation": generation.as_dict(),
        "activated": not problems,
        "rollback_target_retained": rollback_target_retained,
        "problems": problems,
    }


def validate_request_version_consistency(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 23: one request uses exactly one immutable model identity."""
    problems: List[str] = []
    observations: List[Dict[str, Any]] = []
    for row in rows:
        versions = list(row.get("artifact_ids_seen", ()) or ())
        mixed = len(set(versions)) > 1
        entry = {
            "request_id": row.get("request_id", ""),
            "generation_id": row.get("generation_id", ""),
            "artifact_id": versions[0] if versions else "",
            "status": row.get("status", ""),
            "mixed_version": mixed,
        }
        observations.append(entry)
        if mixed:
            problems.append(
                f"request {entry['request_id']} saw {sorted(set(versions))}: sessions must not mix versions"
            )
        if row.get("status") == "INFLIGHT" and not row.get("inflight_policy"):
            problems.append(f"request {entry['request_id']}: inflight behaviour across the switch is undeclared")
    return {"rows": observations, "problems": problems, "ok": not problems}


def validate_kv_prefix_isolation(
    *, cache_keys: Sequence[Mapping[str, Any]], cross_version_hits: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Step 24: KV/prefix cache keys must include model/tokenizer/precision."""
    problems: List[str] = []
    for key in cache_keys:
        missing = [
            field_name
            for field_name in ("model_root", "tokenizer_id", "precision")
            if not key.get(field_name)
        ]
        if missing:
            problems.append(
                f"cache key {key.get('cache_key_id', '?')} omits {missing}: version A state could be reused by B"
            )
    if cross_version_hits:
        problems.append(
            f"{len(cross_version_hits)} prefix/KV hits crossed a version boundary "
            "(semantic corruption with a plausible-looking cache hit)"
        )
    return {"keys": len(cache_keys), "cross_version_hits": len(cross_version_hits), "problems": problems,
            "ok": not problems}


@dataclass
class RollbackPolicy:
    """Step 5: the known-good selection and retention rules of a rollback."""

    policy_id: str
    known_good_selection: str = ""
    retention_days: int = 0
    retention_bytes: int = 0
    trigger: str = ""
    max_rollback_s: int = 0
    verification_gate: Tuple[str, ...] = ()
    artifact_lease_required: bool = True

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("policy_id", "known_good_selection", "trigger"):
            if not getattr(self, name):
                problems.append(f"rollback policy requires {name!r}")
        if self.retention_days <= 0 and self.retention_bytes <= 0:
            problems.append("a rollback target needs a retention window (otherwise it may be collected)")
        if self.max_rollback_s <= 0:
            problems.append("a maximum rollback time must be pre-registered")
        if not self.verification_gate:
            problems.append("the rollback verification gate must be pre-registered")
        if not self.artifact_lease_required:
            problems.append(
                "the rollback target must hold a lease/pin for the whole activation window"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "known_good_selection": self.known_good_selection,
            "retention_days": self.retention_days,
            "retention_bytes": self.retention_bytes,
            "trigger": self.trigger,
            "max_rollback_s": self.max_rollback_s,
            "verification_gate": list(self.verification_gate),
            "artifact_lease_required": self.artifact_lease_required,
        }


def rollback_timeline(
    *,
    rollback_id: str,
    from_artifact: str,
    to_artifact: str,
    decision_ts: str,
    routed_ts: str,
    baseline_restored_ts: str,
    quality_verified: bool,
    resource_state_consistent: bool,
    policy: RollbackPolicy,
    residual_artifacts: Sequence[str],
) -> Dict[str, Any]:
    """Steps 26–27: a rollback is only complete when identity/quality/resources are back."""
    problems = list(policy.validate())
    if not decision_ts or not routed_ts or not baseline_restored_ts:
        problems.append("the rollback timeline must record decision → routing → baseline restored")
    if not quality_verified:
        problems.append("rollback must re-verify quality (a Running pod is not a restored baseline)")
    if not resource_state_consistent:
        problems.append("rollback must verify resource/state consistency")
    quarantined = [item for item in residual_artifacts if item not in (from_artifact, "")]
    if not quarantined and from_artifact:
        problems.append(
            "the failed version must be quarantined/inactive after the rollback, not left as the active pointer"
        )
    return {
        "rollback_id": rollback_id,
        "from_artifact": from_artifact,
        "to_artifact": to_artifact,
        "decision_ts": decision_ts,
        "routed_ts": routed_ts,
        "baseline_restored_ts": baseline_restored_ts,
        "quarantined": quarantined,
        "problems": problems,
        "ok": not problems,
    }


# ── fault injections (steps 28–35) ───────────────────────────────────────


def run_fault_cases(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Steps 28–35: each injection must fail closed and leave a recoverable state."""
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        kind = str(case.get("kind", ""))
        if kind not in FAULT_CASES:
            raise ConfigError(f"unknown artifact fault case {kind!r}")
        served = bool(case.get("served_traffic", False))
        recovered = bool(case.get("recovered", False))
        rows.append(
            {
                "case_id": str(case.get("case_id", f"artifact-fault-{index:03d}")),
                "kind": kind,
                "expected": str(case.get("expected", "FAIL_CLOSED_AND_RECOVERABLE")),
                "first_bad_object": case.get("first_bad_object", ""),
                "served_traffic": served,
                "recovered": recovered,
                "residual_state": case.get("residual_state", ""),
                "ok": (not served) and recovered,
            }
        )
    failures = [row for row in rows if not row["ok"]]
    return {
        "rows": rows,
        "cases": len(rows),
        "failures": failures,
        "ok": not failures,
        "reason": (
            ""
            if not failures
            else "artifact fault cases that served traffic or failed to recover: "
            + ", ".join(row["kind"] for row in failures)
        ),
    }


def interrupted_download_state(
    *, download: StagingDownload, restarted: bool, cache_hit_after_restart: bool
) -> Dict[str, Any]:
    """Step 30: after a crash the partial staging entry is resumed or cleaned, never a hit."""
    problems = list(download.validate())
    if not download.interrupted:
        problems.append("this case is about an interrupted download; mark interrupted=True")
    if not restarted:
        problems.append("the restart of the process must be recorded")
    if cache_hit_after_restart:
        problems.append("a partial staging entry was served as a cache hit after the restart")
    return {
        "attempt_id": download.attempt_id,
        "resumed": download.resumed,
        "problems": problems,
        "ok": not problems,
    }


# ── protocol steps and smoke self-check ──────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 artifact identity contract", ("artifacts:CacheKey", "artifacts:CACHE_KEY_FIELDS")),
    (2, "冻结生命周期状态机", ("records:ARTIFACT_STATE_MACHINE", "artifacts:ArtifactLifecycleEvent.validate")),
    (3, "冻结 cache layout", ("artifacts:cache_layout_manifest", "artifacts:CACHE_AREAS")),
    (4, "冻结 activation semantics", ("artifacts:ActivationGeneration",)),
    (5, "冻结 rollback policy", ("artifacts:RollbackPolicy",)),
    (6, "冻结 workload/quality contract", ("deployment:QualityProbe", "artifacts:assert_servable")),
    (7, "盘点存储与原子能力", ("artifacts:StorageCapability",)),
    (8, "清理任务专用 cold cache", ("artifacts:cache_pre_inventory",)),
    (9, "执行单副本 cold download", ("artifacts:StagingDownload",)),
    (10, "执行逐文件与聚合校验", ("artifacts:verify_download", "artifacts:FileVerification")),
    (11, "执行 compatibility 校验", ("artifacts:check_compatibility", "artifacts:COMPATIBILITY_CHECKS")),
    (12, "原子提交 verified entry", ("artifacts:atomic_commit", "artifacts:AtomicCommit")),
    (13, "加载并记录内存", ("artifacts:load_and_measure",)),
    (14, "编译/解析 engine artifact", ("artifacts:resolve_engine",)),
    (15, "执行 warmup/quality/capacity probe", ("artifacts:WarmupResult", "deployment:QualityProbe")),
    (16, "激活 baseline 版本 A", ("artifacts:ActivationGeneration", "artifacts:assert_servable")),
    (17, "执行 warm cache restart", ("artifacts:cache_layout_manifest",)),
    (18, "执行并发副本下载", ("artifacts:single_flight_outcome",)),
    (19, "执行并发不同版本下载", ("artifacts:CacheKey.key", "artifacts:single_flight_outcome")),
    (20, "准备版本 B", ("artifacts:atomic_commit", "artifacts:assert_loadable")),
    (21, "运行切换前对照流量", ("artifacts:ActivationGeneration",)),
    (22, "触发 A→B 原子切换", ("artifacts:ActivationGeneration.validate",)),
    (23, "验证新旧请求版本一致性", ("artifacts:validate_request_version_consistency",)),
    (24, "验证 KV/prefix cache 隔离", ("artifacts:validate_kv_prefix_isolation",)),
    (25, "验证切换质量/性能", ("deployment:QualityProbe", "capacity:AdmissionDecision")),
    (26, "触发 B→A 回滚", ("artifacts:RollbackPolicy", "artifacts:rollback_timeline")),
    (27, "验证回滚闭环", ("artifacts:rollback_timeline",)),
    (28, "注入缺失/篡改 shard", ("artifacts:run_fault_cases", "artifacts:verify_download")),
    (29, "注入 tokenizer/config/quant mismatch", ("artifacts:run_fault_cases", "artifacts:check_compatibility")),
    (30, "注入下载中断/进程崩溃", ("artifacts:interrupted_download_state",)),
    (31, "注入存储超时/陈旧读取", ("artifacts:run_fault_cases",)),
    (32, "注入磁盘满/水位", ("artifacts:run_fault_cases", "artifacts:gc_plan")),
    (33, "测试 lock owner死亡", ("artifacts:LeaseRecord",)),
    (34, "测试 GC 与保留", ("artifacts:gc_plan", "artifacts:gc_safety_check")),
    (35, "测试并发 GC/load/switch", ("artifacts:gc_safety_check", "artifacts:single_flight_outcome")),
    (36, "重复跨进程/节点", ("artifacts:CacheKey.key", "artifacts:single_flight_outcome")),
    (37, "验证 lineage 与审计", ("artifacts:ArtifactLifecycleEvent", "telemetry:project_request_lifecycle_event")),
    (38, "形成 artifact lifecycle verdict", ("artifacts:artifact_lifecycle_verdict",)),
)


@dataclass
class StorageCapability:
    """Step 7: what the underlying storage actually guarantees."""

    kind: str
    atomic_rename: bool = False
    fsync_supported: bool = False
    lock_supported: bool = False
    quota_bytes: int = 0
    failure_model: str = ""
    consistency_model: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("kind", "failure_model", "consistency_model"):
            if not getattr(self, name):
                problems.append(f"storage capability requires {name!r}")
        if not (self.atomic_rename or self.lock_supported):
            problems.append(
                "the storage offers neither atomic rename nor locking: a partial artifact could become visible"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "atomic_rename": self.atomic_rename,
            "fsync_supported": self.fsync_supported,
            "lock_supported": self.lock_supported,
            "quota_bytes": self.quota_bytes,
            "failure_model": self.failure_model,
            "consistency_model": self.consistency_model,
        }


@dataclass
class WarmupResult:
    """Step 15: warmup/quality/capacity probe outcome (the activation gate)."""

    artifact_id: str
    shapes: Tuple[str, ...] = ()
    rounds: int = 0
    stable_per_round: bool = False
    quality_status: str = ""
    capacity_available: bool = False
    actual_backend: str = ""
    kv_bytes: int = 0

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.shapes:
            problems.append("warmup must use frozen shapes")
        if self.rounds <= 0:
            problems.append("warmup rounds must be recorded (a single round is not a warmup)")
        if not self.stable_per_round:
            problems.append("warmup must show per-round stability (memory growth indicates a leak)")
        if self.quality_status != "pass":
            problems.append("quality must pass before a version may become ACTIVE")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "shapes": list(self.shapes),
            "rounds": self.rounds,
            "stable_per_round": self.stable_per_round,
            "quality_status": self.quality_status,
            "capacity_available": self.capacity_available,
            "actual_backend": self.actual_backend,
            "kv_bytes": self.kv_bytes,
        }


def load_and_measure(
    *, artifact_id: str, host_bytes: int, device_bytes: int, shards: int, engine_cache_hit: bool
) -> Dict[str, Any]:
    """Step 13: load from the verified path and record the footprint."""
    problems: List[str] = []
    if host_bytes <= 0 or device_bytes <= 0:
        problems.append("both host and device memory must be recorded for a load")
    if shards <= 0:
        problems.append("shard count must be recorded")
    return {
        "artifact_id": artifact_id,
        "host_bytes": host_bytes,
        "device_bytes": device_bytes,
        "shards": shards,
        "engine_cache_hit": engine_cache_hit,
        "problems": problems,
        "ok": not problems,
    }


def resolve_engine(
    *, artifact_id: str, target_arch: str, engine_target: str, rebuilt: bool, overwrote_entry: bool
) -> Dict[str, Any]:
    """Step 14: an incompatible engine binary is rebuilt as a *new* artifact."""
    problems: List[str] = []
    if engine_target != target_arch and not rebuilt:
        problems.append(
            f"engine artifact targets {engine_target} but the platform is {target_arch}: "
            "an incompatible binary must be rejected or rebuilt explicitly"
        )
    if overwrote_entry:
        problems.append("rebuilding must not overwrite the original cache entry (identity is content-addressed)")
    return {
        "artifact_id": artifact_id,
        "target_arch": target_arch,
        "engine_target": engine_target,
        "rebuilt": rebuilt,
        "problems": problems,
        "ok": not problems,
    }


def cache_pre_inventory(*, campaign_scope: str, entries: Sequence[str], shared_entries: Sequence[str]) -> Dict[str, Any]:
    """Step 8: only campaign-scoped entries may be removed (never shared data)."""
    outside = sorted(set(entries) - {campaign_scope})
    if outside:
        raise ConfigError(
            f"refusing to clean entries outside the campaign scope: {outside} "
            "(shared/other-tenant data must not be touched)"
        )
    return {
        "campaign_scope": campaign_scope,
        "entries": sorted(entries),
        "shared_entries_untouched": sorted(shared_entries),
        "note": "cold-cache preparation is limited to campaign-scoped entries",
    }


def artifact_lifecycle_verdict(
    *,
    verification_rows: int,
    compatibility_status: str,
    commit: Mapping[str, Any],
    switch: Mapping[str, Any],
    rollback: Mapping[str, Any],
    gc_safety: Mapping[str, Any],
    faults: Mapping[str, Any],
) -> Dict[str, Any]:
    """Step 38: which lifecycle claims the stage may make."""
    problems: List[str] = []
    if verification_rows <= 0:
        problems.append("no verification rows: nothing was verified")
    if compatibility_status != "COMPATIBLE":
        problems.append("compatibility gate did not pass")
    if not commit.get("committed"):
        problems.append("no artifact was atomically committed")
    if not switch.get("ok"):
        problems.append("version switch consistency is unproven")
    if not rollback.get("ok"):
        problems.append("rollback did not restore the known-good baseline")
    if not gc_safety.get("ok"):
        problems.append("GC safety violated")
    if not faults.get("ok"):
        problems.append("artifact fault cases incomplete")
    return {
        "problems": problems,
        "verdict": "PASSABLE_AT_CODE_LEVEL" if not problems else "BLOCKED",
        "activation_mode": switch.get("activation_mode", ""),
        "note": "cache/switch/rollback safety domains come from the executed experiment, not from this verdict",
    }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the artifact lifecycle contracts (smoke, not an experiment)."""
    checks: Dict[str, Any] = {}
    key = CacheKey(
        artifact_id="qwen3-1.7b", content_root=idn.ZERO_DIGEST, model_root="qwen3-1.7b@rev1",
        tokenizer_id="tok-1", config_id="cfg-1", target_arch="sm_86", runtime_abi="cuda-12.4",
    )
    checks["cache_key_valid"] = key.validate() == []
    checks["cache_key_is_content_addressed"] = key.key().startswith("sha256:")

    weak = CacheKey(artifact_id="qwen3-1.7b", model_root="qwen3-1.7b")
    checks["name_only_key_rejected"] = len(weak.validate()) >= 3

    download = StagingDownload(
        attempt_id="a1", artifact_id="qwen3-1.7b", uri="s3://models/rev1", staging_path="/cache/staging/a1",
        active_path="/cache/verified/root", bytes_expected=100, bytes_downloaded=100,
    )
    checks["staging_not_active"] = download.validate() == []

    unsafe = StagingDownload(
        attempt_id="a2", artifact_id="q", uri="s3://m", staging_path="/cache/active", active_path="/cache/active"
    )
    checks["active_path_write_rejected"] = any("active path" in problem for problem in unsafe.validate())

    bad_file = FileVerification(path="model-00002.safetensors", expected_hash="sha256:" + "a" * 64,
                                observed_hash="sha256:" + "b" * 64, expected_size=10, observed_size=10,
                                status="MISMATCH")
    verification = verify_download(
        download=download, files=[bad_file], aggregate_root_expected="r", aggregate_root_observed="r"
    )
    checks["corrupt_shard_detected"] = verification["may_commit"] is False

    commit = AtomicCommit(
        attempt_id="a1", artifact_id="qwen3-1.7b", staging_path="/cache/staging/a1",
        verified_path="/cache/verified/x", marker="/cache/verified/x/.complete", data_durable=True,
        rescanned_after_commit=True,
    )
    checks["commit_requires_durability"] = commit.validate() == []

    early_marker = AtomicCommit(attempt_id="a3", artifact_id="q", data_durable=False,
                                marker_written_after_data=False, rescanned_after_commit=False)
    checks["marker_before_data_rejected"] = len(early_marker.validate()) >= 3

    gc = gc_plan(
        entries=[
            {"artifact_id": "active-1", "state": "ACTIVE", "pins": ["active"], "bytes": 10, "reason": ""},
            {"artifact_id": "old-1", "state": "CACHED_INACTIVE", "pins": [], "bytes": 10,
             "reason": "UNUSED_BEYOND_RETENTION"},
        ],
        watermark_bytes=5,
        used_bytes=20,
    )
    checks["gc_protects_pins"] = [row["artifact_id"] for row in gc["protected"]] == ["active-1"]

    safety = gc_safety_check(
        deleted=["active-1"], protected=["active-1"], active="active-1", rollback_target="rollback-1"
    )
    checks["gc_deleting_active_detected"] = safety["ok"] is False

    inconsistent = validate_request_version_consistency(
        [{"request_id": "r1", "artifact_ids_seen": ["v1", "v2"], "status": "COMPLETED"}]
    )
    checks["mixed_version_detected"] = inconsistent["ok"] is False

    faults = run_fault_cases([{"kind": "MISSING_SHARD", "served_traffic": False, "recovered": True}])
    checks["fault_case_ok"] = faults["ok"] is True
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "note": "接口自检；未下载、未提交、未删除任何制品",
    }