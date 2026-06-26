"""Rank identity, run identity and communicator membership (S10 core).

Details README §6 is explicit: ``global_rank``, ``local_rank``, ``node_rank``,
``group_rank`` and ``device_id`` must never be conflated, and every log line
must carry ``global/local/group rank + node + PID + device UUID + communicator
ID``.  This module defines those records, the checks that make conflating them
impossible, and the run identity every rank must agree on (details README §16).

Nothing here talks to a device: the records come from
:mod:`hqsb.distributed.probes` or from a real launcher.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Run status vocabulary of the S10 protocol (details README §16).
RUN_STATUSES: Tuple[str, ...] = (
    "NOT_RUN",
    "PASS",
    "PASS_NEGATIVE",
    "FAIL",
    "INCONCLUSIVE",
    "UNSUPPORTED",
)

#: Communicator backend kinds (details README §16).
BACKEND_KINDS: Tuple[str, ...] = ("nccl", "hccl", "gloo", "other")

#: Group types a ParallelPlan may create (details README §6.2).
GROUP_TYPES: Tuple[str, ...] = ("tp", "pp", "ep", "cp", "dp", "world")

#: Communicator lifecycle states (details E10-03 §3.1).
COMMUNICATOR_STATES: Tuple[str, ...] = (
    "INIT",
    "READY",
    "ENQUEUED",
    "IN_FLIGHT",
    "COMPLETED",
    "ERROR",
    "ABORTING",
    "ABORTED",
    "DESTROYED",
)

#: Asynchronous error policies a group may declare.
ASYNC_ERROR_POLICIES: Tuple[str, ...] = ("abort_communicator", "query_and_recreate")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunIdentity:
    """The public run identity every rank must agree on (details README §16)."""

    run_id: str
    experiment_id: str
    job_id: str
    world_size: int
    node_count: int
    backend: str
    backend_version: str
    rank_epoch: int = 0
    parallel_plan_hash: str = ""
    topology_manifest_hash: str = ""
    model_artifact_hash: str = ""
    workload_hash: str = ""
    precision_logical: str = ""
    precision_actual: str = ""
    git_commit: str = ""
    config_hash: str = ""
    status: str = "NOT_RUN"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ConfigError("run identity needs a run_id", details={"field": "run_id"})
        if not self.experiment_id.startswith("E10-"):
            raise ConfigError(
                f"experiment_id {self.experiment_id!r} is not an S10 experiment",
                details={"field": "experiment_id"},
            )
        if self.world_size < 1:
            raise ConfigError("world_size must be >= 1", details={"field": "world_size"})
        if self.node_count < 1:
            raise ConfigError("node_count must be >= 1", details={"field": "node_count"})
        if self.backend not in BACKEND_KINDS:
            raise ConfigError(
                f"backend must be one of {BACKEND_KINDS}", details={"field": "backend"}
            )
        if (
            not self.backend_version
            or self.backend_version.strip().lower() in ("latest", "unknown", "unset", "n/a")
        ):
            raise ConfigError(
                "the backend version must be recorded exactly (no 'latest'/'unknown'/blank)",
                details={"field": "backend_version"},
            )
        if self.rank_epoch < 0:
            raise ConfigError("rank_epoch must be >= 0", details={"field": "rank_epoch"})
        if self.status not in RUN_STATUSES:
            raise ConfigError(
                f"status must be one of {RUN_STATUSES}", details={"field": "status"}
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "job_id": self.job_id,
            "rank_epoch": self.rank_epoch,
            "world_size": self.world_size,
            "node_count": self.node_count,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "parallel_plan_hash": self.parallel_plan_hash,
            "topology_manifest_hash": self.topology_manifest_hash,
            "model_artifact_hash": self.model_artifact_hash,
            "workload_hash": self.workload_hash,
            "precision_logical": self.precision_logical,
            "precision_actual": self.precision_actual,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "status": self.status,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2, ensure_ascii=False)

    @property
    def identity_hash(self) -> str:
        return _sha256_text(self.canonical_json())


#: Fields that must be identical across every rank of a job.
SHARED_IDENTITY_FIELDS: Tuple[str, ...] = (
    "run_id",
    "experiment_id",
    "job_id",
    "world_size",
    "node_count",
    "backend",
    "backend_version",
    "rank_epoch",
    "parallel_plan_hash",
    "topology_manifest_hash",
    "model_artifact_hash",
    "workload_hash",
    "precision_logical",
    "precision_actual",
    "git_commit",
    "config_hash",
)


def compare_run_identity(identities: Sequence[RunIdentity]) -> Dict[str, Any]:
    """A multi-rank aggregator must not stitch different jobs together (§16)."""
    violations: List[str] = []
    if not identities:
        return {"ok": False, "violations": ["no rank identity recorded"], "reference": ""}
    reference = identities[0]
    for item in identities[1:]:
        for name in SHARED_IDENTITY_FIELDS:
            if getattr(item, name) != getattr(reference, name):
                violations.append(
                    f"rank run identity differs on {name!r}: "
                    f"{getattr(reference, name)!r} vs {getattr(item, name)!r}"
                )
    return {
        "ok": not violations,
        "violations": violations,
        "reference": reference.run_id,
        "shared_fields": list(SHARED_IDENTITY_FIELDS),
    }


@dataclass(frozen=True)
class RankIdentity:
    """One process' actual binding, reported from inside the process."""

    global_rank: int
    local_rank: int
    node_rank: int
    group_rank: Mapping[str, int] = field(default_factory=dict)
    node_id: str = ""
    pid: int = 0
    device_uuid: str = ""
    device_index: int = -1
    cpu_affinity: Tuple[int, ...] = ()
    numa_memory_policy: str = ""
    visible_nics: Tuple[str, ...] = ()
    current_stream: str = ""
    rank_epoch: int = 0
    communicator_ids: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
            "node_rank": self.node_rank,
            "group_rank": dict(sorted(self.group_rank.items())),
            "node_id": self.node_id,
            "pid": self.pid,
            "device_uuid": self.device_uuid,
            "device_index": self.device_index,
            "cpu_affinity": list(self.cpu_affinity),
            "numa_memory_policy": self.numa_memory_policy,
            "visible_nics": list(self.visible_nics),
            "current_stream": self.current_stream,
            "rank_epoch": self.rank_epoch,
            "communicator_ids": dict(sorted(self.communicator_ids.items())),
        }


def validate_rank_identities(
    identities: Sequence[RankIdentity],
    *,
    expected_world_size: Optional[int] = None,
    one_rank_per_device: bool = True,
) -> Dict[str, Any]:
    """Aggregator checks: duplicates, gaps, shared devices, world size (step 22)."""
    issues: List[str] = []
    ranks = [item.global_rank for item in identities]
    if len(set(ranks)) != len(ranks):
        issues.append("duplicate global_rank")
    if ranks:
        expected = (
            expected_world_size if expected_world_size is not None else max(ranks) + 1
        )
        missing = sorted(set(range(expected)) - set(ranks))
        if missing:
            issues.append(f"rank gap: missing {missing}")
        if expected_world_size is not None and len(ranks) != expected_world_size:
            issues.append(
                f"observed {len(ranks)} ranks but planned world_size is {expected_world_size}"
            )
    if one_rank_per_device:
        devices: Dict[str, List[int]] = {}
        for item in identities:
            devices.setdefault(item.device_uuid, []).append(item.global_rank)
        for device, holders in sorted(devices.items()):
            if len(holders) > 1:
                issues.append(f"device {device} is bound by ranks {sorted(holders)}")
    if any(not item.device_uuid for item in identities):
        issues.append("some rank reports no device UUID (local index is not identity)")
    return {
        "ok": not issues,
        "issues": issues,
        "one_rank_per_device": one_rank_per_device,
        "world_size_observed": len(identities),
    }


@dataclass(frozen=True)
class GroupMembership:
    """One communicator/process group declaration (details README §6.2)."""

    group_id: str
    group_type: str
    ordered_global_ranks: Tuple[int, ...]
    backend: str
    version: str
    creation_epoch: int
    timeout_s: float
    async_error_policy: str = "abort_communicator"
    topology_scope: str = ""
    creation_sequence: int = 0
    state: str = "INIT"

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ConfigError("group needs a group_id", details={"field": "group_id"})
        if self.group_type not in GROUP_TYPES:
            raise ConfigError(
                f"group_type must be one of {GROUP_TYPES}", details={"field": "group_type"}
            )
        if not self.ordered_global_ranks:
            raise ConfigError("group needs ordered members", details={"field": "ordered_global_ranks"})
        if len(set(self.ordered_global_ranks)) != len(self.ordered_global_ranks):
            raise ConfigError(
                "group members must be unique", details={"field": "ordered_global_ranks"}
            )
        if self.backend not in BACKEND_KINDS:
            raise ConfigError(
                f"backend must be one of {BACKEND_KINDS}", details={"field": "backend"}
            )
        if self.async_error_policy not in ASYNC_ERROR_POLICIES:
            raise ConfigError(
                f"async_error_policy must be one of {ASYNC_ERROR_POLICIES}",
                details={"field": "async_error_policy"},
            )
        if self.timeout_s <= 0:
            raise ConfigError("group timeout must be positive", details={"field": "timeout_s"})
        if self.state not in COMMUNICATOR_STATES:
            raise ConfigError(
                f"state must be one of {COMMUNICATOR_STATES}", details={"field": "state"}
            )

    @property
    def ordered_group_hash(self) -> str:
        return _sha256_text(",".join(str(rank) for rank in self.ordered_global_ranks))

    def group_rank_of(self, global_rank: int) -> int:
        """Group rank is the index in *this* group, never the global rank."""
        try:
            return self.ordered_global_ranks.index(global_rank)
        except ValueError as exc:
            raise ConfigError(
                f"global rank {global_rank} is not a member of group {self.group_id}",
                details={"field": "global_rank"},
            ) from exc

    def member_ranks(self) -> Tuple[int, ...]:
        return tuple(self.ordered_global_ranks)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "group_type": self.group_type,
            "ordered_global_ranks": list(self.ordered_global_ranks),
            "backend": self.backend,
            "version": self.version,
            "creation_epoch": self.creation_epoch,
            "timeout_s": self.timeout_s,
            "async_error_policy": self.async_error_policy,
            "topology_scope": self.topology_scope,
            "creation_sequence": self.creation_sequence,
            "state": self.state,
        }


def validate_group_membership(
    groups: Sequence[GroupMembership],
    *,
    observed_ranks: Optional[Iterable[int]] = None,
    expected_creation_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Membership + creation-order drift detection (details E10-01 step 23)."""
    issues: List[str] = []
    ids = [group.group_id for group in groups]
    if len(set(ids)) != len(ids):
        issues.append("duplicate group_id")
    for group in groups:
        if len(group.ordered_global_ranks) < 1:
            issues.append(f"group {group.group_id} has no members")
    observed = set(observed_ranks) if observed_ranks is not None else None
    if observed is not None:
        for group in groups:
            missing = sorted(set(group.ordered_global_ranks) - observed)
            if missing:
                issues.append(f"group {group.group_id} lists unobserved ranks {missing}")
    if expected_creation_order is not None:
        actual_order = [
            group.group_id
            for group in sorted(groups, key=lambda item: item.creation_sequence)
        ]
        if list(expected_creation_order) != actual_order:
            issues.append(
                "group creation order drift: planned "
                f"{list(expected_creation_order)} observed {actual_order}"
            )
    return {"ok": not issues, "issues": issues, "groups": len(groups)}


def compare_planned_actual_groups(
    planned: Sequence[GroupMembership], actual: Sequence[GroupMembership]
) -> Dict[str, Any]:
    """Planned/actual group diff (never trust the launcher's intent)."""
    planned_by_id = {group.group_id: group for group in planned}
    actual_by_id = {group.group_id: group for group in actual}
    mismatches: List[str] = []
    for group_id in sorted(set(planned_by_id) | set(actual_by_id)):
        left = planned_by_id.get(group_id)
        right = actual_by_id.get(group_id)
        if left is None or right is None:
            mismatches.append(f"group {group_id} exists only on one side")
            continue
        if tuple(left.ordered_global_ranks) != tuple(right.ordered_global_ranks):
            mismatches.append(
                f"group {group_id} ordered ranks differ: "
                f"{list(left.ordered_global_ranks)} vs {list(right.ordered_global_ranks)}"
            )
    return {"ok": not mismatches, "mismatches": mismatches}


def allocator_group_order_check(
    groups: Sequence[GroupMembership], planned_order: Sequence[str]
) -> Dict[str, Any]:
    """Multi-group ordering rule (details E10-03 §3.2): creation/usage order must match."""
    return validate_group_membership(
        groups, expected_creation_order=planned_order
    )


__all__ = [
    "ASYNC_ERROR_POLICIES",
    "BACKEND_KINDS",
    "COMMUNICATOR_STATES",
    "GROUP_TYPES",
    "GroupMembership",
    "RUN_STATUSES",
    "RankIdentity",
    "RunIdentity",
    "SHARED_IDENTITY_FIELDS",
    "allocator_group_order_check",
    "compare_planned_actual_groups",
    "compare_run_identity",
    "validate_group_membership",
    "validate_rank_identities",
]
