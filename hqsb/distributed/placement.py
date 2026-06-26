"""PlacementPlan: parallel coordinates → rank → device/CPU/NIC (E10-01).

The plan is an *intent*; the acceptance criterion is the comparison with what
the processes actually bound to (details README §7: "环境变量写了 device 0
不等于进程实际只访问 device 0").

Everything here is pure data plus validation: building the plan, generating the
launcher/rank table from the same object, comparing planned vs actual rank
identities, and constructing a legal alternative placement used as a sanity
control for the topology graph.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.distributed.ranks import GroupMembership, RankIdentity

#: One-rank-per-device is the default experiment invariant (details README §12).
DEFAULT_INVARIANTS: Tuple[str, ...] = (
    "unique(global_rank)",
    "unique(device_uuid)",
    "planned_world_size == observed_rank_count",
    "planned_fast_path_edges_not_down_or_unknown",
    "actual_device_uuid == planned_device_uuid",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParallelCoordinate:
    """The (tp, pp, ep, cp) coordinates of a rank."""

    tp: int = 0
    pp: int = 0
    ep: int = 0
    cp: int = 0

    def __post_init__(self) -> None:
        for name in ("tp", "pp", "ep", "cp"):
            if getattr(self, name) < 0:
                raise ConfigError(
                    f"parallel coordinate {name} must be >= 0", details={"field": name}
                )

    def as_dict(self) -> Dict[str, int]:
        return {"tp": self.tp, "pp": self.pp, "ep": self.ep, "cp": self.cp}

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.tp, self.pp, self.ep, self.cp)


@dataclass(frozen=True)
class RankPlacement:
    """One rank's planned placement plus, later, its actual binding."""

    global_rank: int
    local_rank: int
    node_rank: int
    coordinate: ParallelCoordinate
    planned_device_uuid: str
    actual_device_uuid: str = ""
    cpu_affinity: Tuple[int, ...] = ()
    numa_memory_policy: str = ""
    preferred_nic: str = ""
    communicator_ids: Tuple[str, ...] = ()
    launcher_env_digest: str = ""

    @property
    def binding_ok(self) -> bool:
        return bool(self.actual_device_uuid) and self.actual_device_uuid == self.planned_device_uuid

    def as_dict(self) -> Dict[str, Any]:
        return {
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
            "node_rank": self.node_rank,
            "coordinate": self.coordinate.as_dict(),
            "planned_device_uuid": self.planned_device_uuid,
            "actual_device_uuid": self.actual_device_uuid,
            "cpu_affinity": list(self.cpu_affinity),
            "numa_memory_policy": self.numa_memory_policy,
            "preferred_nic": self.preferred_nic,
            "communicator_ids": list(self.communicator_ids),
            "launcher_env_digest": self.launcher_env_digest,
        }


@dataclass
class PlacementPlan:
    """A versioned placement plan (details E10-01 §5)."""

    plan_id: str
    world_size: int
    node_count: int
    entries: Tuple[RankPlacement, ...]
    source_topology_hash: str = ""
    high_bandwidth_domains: Mapping[str, Tuple[int, ...]] = field(default_factory=dict)
    policy_notes: str = ""
    created_at: str = ""
    schema_version: str = "1.0.0"

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.world_size < 1:
            errors.append("world_size must be >= 1")
        if self.node_count < 1:
            errors.append("node_count must be >= 1")
        if len(self.entries) != self.world_size:
            errors.append(
                f"plan lists {len(self.entries)} entries for world_size {self.world_size}"
            )
        ranks = [entry.global_rank for entry in self.entries]
        if len(set(ranks)) != len(ranks):
            errors.append("duplicate global_rank in plan")
        if sorted(ranks) != list(range(self.world_size)):
            errors.append("planned global ranks are not contiguous [0, world_size)")
        devices = [entry.planned_device_uuid for entry in self.entries]
        if len(set(devices)) != len(devices):
            errors.append("one-rank-per-device plan binds a device twice")
        if any(not entry.planned_device_uuid for entry in self.entries):
            errors.append("every planned rank needs a device UUID")
        for rank in self.high_bandwidth_domains.values():
            unknown = sorted(set(rank) - set(ranks))
            if unknown:
                errors.append(f"high-bandwidth domain references unknown ranks {unknown}")
        return errors

    def entry_for(self, global_rank: int) -> RankPlacement:
        for entry in self.entries:
            if entry.global_rank == global_rank:
                return entry
        raise ConfigError(f"rank {global_rank} is not in plan {self.plan_id}")

    def planned_device_by_rank(self) -> Dict[int, str]:
        return {entry.global_rank: entry.planned_device_uuid for entry in self.entries}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "world_size": self.world_size,
            "node_count": self.node_count,
            "source_topology_hash": self.source_topology_hash,
            "high_bandwidth_domains": {
                name: list(ranks) for name, ranks in sorted(self.high_bandwidth_domains.items())
            },
            "policy_notes": self.policy_notes,
            "created_at": self.created_at,
            "entries": [entry.as_dict() for entry in self.entries],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2, ensure_ascii=False)

    @property
    def sha256(self) -> str:
        return _sha256_text(self.canonical_json())


def build_rank_table(plan: PlacementPlan) -> List[Dict[str, Any]]:
    """Generate the launcher rank table from the one plan (never hand-written twice)."""
    rows: List[Dict[str, Any]] = []
    for entry in sorted(plan.entries, key=lambda item: item.global_rank):
        rows.append(
            {
                "global_rank": entry.global_rank,
                "local_rank": entry.local_rank,
                "node_rank": entry.node_rank,
                "device_uuid": entry.planned_device_uuid,
                "coordinate": entry.coordinate.as_dict(),
                "communicator_ids": list(entry.communicator_ids),
            }
        )
    return rows


#: Launcher kinds a rank table can be rendered for.
LAUNCHER_KINDS: Tuple[str, ...] = ("torchrun", "mpirun", "hccl_hostfile", "local")

#: Environment keys that carry placement intent (recorded as a digest, not paths).
LAUNCHER_ENV_KEYS: Tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NCCL_DEBUG",
    "HCCL_IF_IP",
)


def launcher_config(plan: PlacementPlan, kind: str = "torchrun") -> Dict[str, Any]:
    """Render the launcher mapping for a plan (details E10-01 step 20)."""
    if kind not in LAUNCHER_KINDS:
        raise ConfigError(
            f"launcher kind must be one of {LAUNCHER_KINDS}", details={"field": "kind"}
        )
    table = build_rank_table(plan)
    payload: Dict[str, Any] = {
        "kind": kind,
        "world_size": plan.world_size,
        "nodes": plan.node_count,
        "rank_table": table,
        "plan_hash": plan.sha256,
        "env_keys_expected": list(LAUNCHER_ENV_KEYS),
    }
    if kind == "mpirun":
        payload["hostfile_hint"] = "one slot per planned device; see rank_table"
    if kind == "hccl_hostfile":
        payload["hostfile_hint"] = "device ids come from the plan, not from local indices"
    return payload


def compare_planned_actual(
    plan: PlacementPlan, identities: Sequence[RankIdentity]
) -> Dict[str, Any]:
    """Planned vs actual binding comparison (the acceptance criterion)."""
    mismatches: List[Dict[str, Any]] = []
    observed = {item.global_rank: item for item in identities}
    for entry in plan.entries:
        item = observed.get(entry.global_rank)
        if item is None:
            mismatches.append({"global_rank": entry.global_rank, "issue": "rank missing"})
            continue
        if item.device_uuid != entry.planned_device_uuid:
            mismatches.append(
                {
                    "global_rank": entry.global_rank,
                    "issue": "device mismatch",
                    "planned": entry.planned_device_uuid,
                    "actual": item.device_uuid,
                }
            )
        if item.local_rank != entry.local_rank:
            mismatches.append(
                {
                    "global_rank": entry.global_rank,
                    "issue": "local rank mismatch",
                    "planned": entry.local_rank,
                    "actual": item.local_rank,
                }
            )
        if entry.cpu_affinity and tuple(item.cpu_affinity) != tuple(entry.cpu_affinity):
            mismatches.append(
                {
                    "global_rank": entry.global_rank,
                    "issue": "cpu affinity mismatch",
                    "planned": list(entry.cpu_affinity),
                    "actual": list(item.cpu_affinity),
                }
            )
    extra = sorted(set(observed) - {entry.global_rank for entry in plan.entries})
    if extra:
        mismatches.append({"global_rank": extra, "issue": "unplanned ranks observed"})
    return {
        "ok": not mismatches,
        "mismatches": mismatches,
        "planned_world_size": plan.world_size,
        "observed_rank_count": len(identities),
    }


def validate_communicator_membership(
    plan: PlacementPlan, groups: Sequence[GroupMembership]
) -> Dict[str, Any]:
    """Every group member must be a planned rank and vice versa where required."""
    issues: List[str] = []
    planned = {entry.global_rank for entry in plan.entries}
    for group in groups:
        unknown = sorted(set(group.ordered_global_ranks) - planned)
        if unknown:
            issues.append(f"group {group.group_id} references unplanned ranks {unknown}")
        for entry in plan.entries:
            if entry.global_rank in group.ordered_global_ranks:
                if group.group_id not in entry.communicator_ids:
                    issues.append(
                        f"rank {entry.global_rank} is a member of {group.group_id} but the plan "
                        "does not list that communicator"
                    )
    return {"ok": not issues, "issues": issues}


@dataclass(frozen=True)
class PlacementSanityExpectation:
    """What the alternative placement is allowed to claim (sanity, not speed)."""

    baseline_edges: Tuple[str, ...]
    alternative_edges: Tuple[str, ...]
    expected_direction: str  # "slower_or_equal" | "no_difference_expected"
    reason_if_no_difference: str = ""
    is_performance_claim: bool = False

    def validate(self) -> None:
        if self.is_performance_claim:
            raise ConfigError(
                "the alternative placement control is a topology sanity check, not a speed "
                "conclusion (details E10-01 step 25)",
                details={"field": "is_performance_claim"},
            )
        if self.expected_direction == "no_difference_expected" and not self.reason_if_no_difference:
            raise ConfigError(
                "a 'no difference' expectation must state why (small messages, algorithm "
                "bypass, measurement limits)",
                details={"field": "reason_if_no_difference"},
            )


def alternative_placement(
    plan: PlacementPlan, *, swap_ranks: Tuple[int, int], reason: str
) -> Tuple[PlacementPlan, PlacementSanityExpectation]:
    """Build a legal slower/remote-placement control by swapping two ranks."""
    if not reason:
        raise ConfigError("an alternative placement needs a written reason")
    left, right = swap_ranks
    entries = {entry.global_rank: entry for entry in plan.entries}
    if left not in entries or right not in entries:
        raise ConfigError(f"swap ranks {swap_ranks} must both exist in the plan")
    a, b = entries[left], entries[right]
    new_entries = list(plan.entries)
    new_entries[new_entries.index(a)] = _replace_placement(
        a, global_rank=right, local_rank=b.local_rank, node_rank=b.node_rank,
        planned_device_uuid=b.planned_device_uuid,
    )
    new_entries[new_entries.index(b)] = _replace_placement(
        b, global_rank=left, local_rank=a.local_rank, node_rank=a.node_rank,
        planned_device_uuid=a.planned_device_uuid,
    )
    alternative = PlacementPlan(
        plan_id=f"{plan.plan_id}-alt-{left}{right}",
        world_size=plan.world_size,
        node_count=plan.node_count,
        entries=tuple(sorted(new_entries, key=lambda item: item.global_rank)),
        source_topology_hash=plan.source_topology_hash,
        high_bandwidth_domains=plan.high_bandwidth_domains,
        policy_notes=reason,
        created_at=plan.created_at,
    )
    expectation = PlacementSanityExpectation(
        baseline_edges=(),
        alternative_edges=(),
        expected_direction="no_difference_expected",
        reason_if_no_difference=(
            "the control is a direction sanity check; a null result does not prove the "
            "topology has no effect"
        ),
    )
    return alternative, expectation


def _replace_placement(entry: RankPlacement, **changes: Any) -> RankPlacement:
    payload = entry.as_dict()
    payload.update(changes)
    payload["coordinate"] = entry.coordinate
    return RankPlacement(
        global_rank=payload["global_rank"],
        local_rank=payload["local_rank"],
        node_rank=payload["node_rank"],
        coordinate=payload["coordinate"],
        planned_device_uuid=payload["planned_device_uuid"],
        actual_device_uuid=payload["actual_device_uuid"],
        cpu_affinity=tuple(payload["cpu_affinity"]),
        numa_memory_policy=payload["numa_memory_policy"],
        preferred_nic=payload["preferred_nic"],
        communicator_ids=tuple(payload["communicator_ids"]),
        launcher_env_digest=payload["launcher_env_digest"],
    )


def plan_digest(entries_env: Mapping[str, str]) -> str:
    """Digest the placement-relevant launcher environment (no absolute paths)."""
    payload = {key: entries_env.get(key, "") for key in LAUNCHER_ENV_KEYS}
    return _sha256_text(json.dumps(payload, sort_keys=True))


__all__ = [
    "DEFAULT_INVARIANTS",
    "LAUNCHER_ENV_KEYS",
    "LAUNCHER_KINDS",
    "ParallelCoordinate",
    "PlacementPlan",
    "PlacementSanityExpectation",
    "RankPlacement",
    "alternative_placement",
    "build_rank_table",
    "compare_planned_actual",
    "launcher_config",
    "plan_digest",
    "validate_communicator_membership",
]
