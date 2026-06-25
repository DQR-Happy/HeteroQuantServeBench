"""Read-only probes and preflight fixtures for topology/placement (E10-01).

Two things live here:

1. **Probe execution** — a tiny executor abstraction
   (:class:`SubprocessExecutor` for real hosts, :class:`FixtureExecutor` for
   tests and for hosts where the tool is absent).  Every probe result carries a
   four-state status; a missing tool yields ``UNAVAILABLE(reason)`` and can never
   be mistaken for a measured value.  All commands are read-only.
2. **Preflight fixtures** — the *negative* mapping cases of details E10-01
   step 26 (duplicate device, unknown UUID, rank gap, wrong group order).  They
   must be rejected before a communicator is initialised, and they need no real
   communication to run, so they can be exercised on a CPU-only host.

Nothing here mutates shared state: no traffic shaping, no device reset, no
killing processes (those live in :mod:`hqsb.distributed.faults` with a safety
scope).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.distributed.placement import PlacementPlan
from hqsb.distributed.ranks import GroupMembership, RankIdentity
from hqsb.distributed.topology import unavailable

#: Probe outcomes: a probe either observed something, was not applicable, could
#: not reach the tool, or failed.  Never collapse these into a boolean.
PROBE_STATUSES: Tuple[str, ...] = ("PASS", "UNSUPPORTED", "UNVERIFIED", "UNAVAILABLE", "FAIL")

#: P2P capability states (details E10-01 step 13).
P2P_STATES: Tuple[str, ...] = ("SUPPORTED", "UNSUPPORTED", "UNKNOWN")

#: The read-only inventory commands the CUDA/NPU branches use.  ``required``
#: commands block the manifest; the rest degrade to UNAVAILABLE.
PROBE_COMMANDS: Mapping[str, Mapping[str, Any]] = {
    "device_query": {"command": ("nvidia-smi", "--query-gpu=index,uuid,serial,name,pci.bus_id,memory.total", "--format=csv,noheader"), "required": True, "branch": "cuda_nccl"},
    "npu_query": {"command": ("npu-smi", "info"), "required": False, "branch": "ascend_hccl"},
    "cpu_numa": {"command": ("lscpu", "--parse=CPU,NODE,SOCKET,CACHE"), "required": True, "branch": "both"},
    "numa_hardware": {"command": ("numactl", "--hardware"), "required": False, "branch": "both"},
    "pcie_tree": {"command": ("lspci", "-vv", "-nn"), "required": False, "branch": "both"},
    "topo_matrix": {"command": ("nvidia-smi", "topo", "-m"), "required": False, "branch": "cuda_nccl"},
    "nic_ports": {"command": ("ip", "-j", "link", "show"), "required": False, "branch": "both"},
    "rdma_links": {"command": ("rdma", "link", "show"), "required": False, "branch": "both"},
    "rdma_devices": {"command": ("ibv_devinfo", "-l"), "required": False, "branch": "both"},
    "hccl_config": {"command": ("cat", "/usr/local/Ascend/hccl_config.json"), "required": False, "branch": "ascend_hccl"},
}

#: Environment keys that describe the backend runtime (details E10-01 step 11).
BACKEND_ENV_KEYS: Tuple[str, ...] = (
    "NCCL_DEBUG",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_IB_HCA",
    "NCCL_SOCKET_IFNAME",
    "NCCL_P2P_DISABLE",
    "NCCL_NVLS_ENABLE",
    "HCCL_ALGO",
    "HCCL_EXEC_TIMEOUT",
    "HCCL_IF_IP",
)


@dataclass(frozen=True)
class ProbeOutcome:
    """One probe run: status, raw output pointer and a structured value/reason."""

    name: str
    status: str
    command: Tuple[str, ...] = ()
    stdout_path: str = ""
    value: Any = ""
    reason: str = ""
    returncode: Optional[int] = None

    def __post_init__(self) -> None:
        if self.status not in PROBE_STATUSES:
            raise ConfigError(
                f"probe status must be one of {PROBE_STATUSES}", details={"field": "status"}
            )
        if self.status in ("UNAVAILABLE", "UNVERIFIED", "FAIL") and not self.reason:
            raise ConfigError(
                f"probe {self.name} reported {self.status} without a reason (no silent gaps)",
                details={"field": "reason"},
            )

    @property
    def ok(self) -> bool:
        return self.status == "PASS"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "command": list(self.command),
            "stdout_path": self.stdout_path,
            "value": self.value,
            "reason": self.reason,
            "returncode": self.returncode,
        }


class CommandExecutor:
    """Executes read-only commands (or refuses to, in dry-run/test mode)."""

    def run(self, command: Sequence[str], timeout_s: float = 10.0) -> ProbeOutcome:  # pragma: no cover - interface
        raise NotImplementedError


class SubprocessExecutor(CommandExecutor):
    """Runs the command for real; a missing tool is UNAVAILABLE, never an exception."""

    def __init__(self, *, allow_network_tools: bool = False) -> None:
        self.allow_network_tools = allow_network_tools

    def run(self, command: Sequence[str], timeout_s: float = 10.0) -> ProbeOutcome:
        name = command[0] if command else "unnamed"
        try:
            completed = subprocess.run(
                list(command), capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except FileNotFoundError:
            return ProbeOutcome(
                name=name,
                status="UNAVAILABLE",
                command=tuple(command),
                reason=f"{name} is not installed on this host",
            )
        except subprocess.TimeoutExpired:
            return ProbeOutcome(
                name=name,
                status="FAIL",
                command=tuple(command),
                reason=f"{name} timed out after {timeout_s}s",
            )
        if completed.returncode != 0:
            return ProbeOutcome(
                name=name,
                status="UNAVAILABLE",
                command=tuple(command),
                reason=f"{name} exited {completed.returncode}: {completed.stderr.strip()[:200]}",
                returncode=completed.returncode,
            )
        return ProbeOutcome(
            name=name,
            status="PASS",
            command=tuple(command),
            value=completed.stdout,
            returncode=0,
        )


class FixtureExecutor(CommandExecutor):
    """Returns recorded outputs; used by tests and by tool-less hosts."""

    def __init__(self, fixtures: Mapping[str, str]) -> None:
        self.fixtures = dict(fixtures)

    def run(self, command: Sequence[str], timeout_s: float = 10.0) -> ProbeOutcome:
        key = " ".join(command)
        if key in self.fixtures:
            return ProbeOutcome(
                name=command[0], status="PASS", command=tuple(command), value=self.fixtures[key]
            )
        return ProbeOutcome(
            name=command[0] if command else "unnamed",
            status="UNAVAILABLE",
            command=tuple(command),
            reason=f"no fixture recorded for {key!r}",
        )


def run_inventory_probes(
    executor: CommandExecutor, *, branch: str
) -> Dict[str, ProbeOutcome]:
    """Run the read-only inventory probe set for one branch (step 1/3/4/6/7/9/10/11)."""
    if branch not in ("cuda_nccl", "ascend_hccl", "both"):
        raise ConfigError(f"unknown branch {branch!r}", details={"field": "branch"})
    outcomes: Dict[str, ProbeOutcome] = {}
    for name, spec in PROBE_COMMANDS.items():
        if spec["branch"] not in (branch, "both"):
            outcomes[name] = ProbeOutcome(
                name=name,
                status="UNSUPPORTED",
                reason=f"probe belongs to the {spec['branch']} branch",
            )
            continue
        outcomes[name] = executor.run(tuple(spec["command"]))
    return outcomes


def probe_chain_status(outcomes: Mapping[str, ProbeOutcome]) -> Dict[str, Any]:
    """First non-pass probe, with the required set separated from the optional set."""
    required = {name for name, spec in PROBE_COMMANDS.items() if spec["required"]}
    first_non_pass = ""
    blockers: List[str] = []
    for name in sorted(outcomes):
        outcome = outcomes[name]
        if outcome.status != "PASS":
            first_non_pass = first_non_pass or name
            if name in required:
                blockers.append(f"{name}:{outcome.status}")
    return {
        "statuses": {name: outcome.status for name, outcome in sorted(outcomes.items())},
        "first_non_pass": first_non_pass,
        "required_blockers": blockers,
        "ok": not blockers,
    }


def capture_backend_env(
    environ: Mapping[str, str], *, backend: str, version: str
) -> Dict[str, Any]:
    """Record every actually-effective communication env var (step 11)."""
    if backend not in ("nccl", "hccl", "gloo", "other"):
        raise ConfigError(f"unknown backend {backend!r}", details={"field": "backend"})
    env = {key: environ[key] for key in BACKEND_ENV_KEYS if key in environ}
    return {
        "backend": backend,
        "version": version,
        "env_vars": env,
        "unset_means_default": sorted(set(BACKEND_ENV_KEYS) - set(env)),
        "defaults_explained_by": "the locked backend version's documentation",
    }


# ── probe matrices ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class P2PCapability:
    """One ordered device pair's P2P capability (never inferred from PCIe/NIC)."""

    src_uuid: str
    dst_uuid: str
    read: str = "UNKNOWN"
    write: str = "UNKNOWN"
    atomic: str = "UNKNOWN"
    reason: str = ""

    def __post_init__(self) -> None:
        for name in ("read", "write", "atomic"):
            if getattr(self, name) not in P2P_STATES:
                raise ConfigError(
                    f"P2P {name} must be one of {P2P_STATES}", details={"field": name}
                )
        if "UNKNOWN" in (self.read, self.write, self.atomic) and not self.reason:
            raise ConfigError(
                "an UNKNOWN P2P capability needs a reason (probe not run/permission)",
                details={"field": "reason"},
            )

    @property
    def reachable(self) -> bool:
        return all(getattr(self, name) == "SUPPORTED" for name in ("read", "write"))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "src_uuid": self.src_uuid,
            "dst_uuid": self.dst_uuid,
            "read": self.read,
            "write": self.write,
            "atomic": self.atomic,
            "reachable": self.reachable,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CopyProbeRow:
    """Device-to-device copy microprobe (step 14) — a link sanity check only."""

    src_uuid: str
    dst_uuid: str
    message_bytes: int
    latency_us: float
    bandwidth_gbps: float
    hop_class: str = ""
    source_tool: str = ""

    def validate(self) -> None:
        if self.message_bytes <= 0 or self.latency_us <= 0:
            raise ConfigError(
                "copy probe rows need positive message size and latency",
                details={"field": "message_bytes"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "src_uuid": self.src_uuid,
            "dst_uuid": self.dst_uuid,
            "message_bytes": self.message_bytes,
            "latency_us": self.latency_us,
            "bandwidth_gbps": self.bandwidth_gbps,
            "hop_class": self.hop_class,
            "source_tool": self.source_tool,
            "note": "device copy sanity; this is NOT an E10-02 collective curve",
        }


@dataclass(frozen=True)
class HostDeviceProbe:
    """Host↔device NUMA probe (step 15)."""

    accelerator_uuid: str
    numa_binding: str  # "local" | "remote"
    pinned_copy_gbps: float
    submission_latency_us: float
    correct_binding: bool
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accelerator_uuid": self.accelerator_uuid,
            "numa_binding": self.numa_binding,
            "pinned_copy_gbps": self.pinned_copy_gbps,
            "submission_latency_us": self.submission_latency_us,
            "correct_binding": self.correct_binding,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RdmaReachability:
    """NIC/RDMA reachability probe (step 16): IP vs RDMA vs device-direct."""

    src_node: str
    dst_node: str
    ip_reachable: bool
    rdma_reachable: str = "UNVERIFIED"  # bool-like states: True/False/"UNVERIFIED"
    device_direct_reachable: str = "UNVERIFIED"
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "src_node": self.src_node,
            "dst_node": self.dst_node,
            "ip_reachable": self.ip_reachable,
            "rdma_reachable": self.rdma_reachable,
            "device_direct_reachable": self.device_direct_reachable,
            "reason": self.reason,
            "note": "reachability is not a verified data path",
        }


@dataclass(frozen=True)
class DataPathEvidence:
    """What transport a minimal collective actually used (step 17)."""

    collective: str
    src_node: str
    dst_node: str
    transport: str
    source: str
    verified: bool
    raw_uri: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "collective": self.collective,
            "src_node": self.src_node,
            "dst_node": self.dst_node,
            "transport": self.transport,
            "source": self.source,
            "verified": self.verified,
            "raw_uri": self.raw_uri,
        }


def rdma_claim_guard(reachability: RdmaReachability, data_path: Optional[DataPathEvidence]) -> Dict[str, Any]:
    """A NIC being present/up does not make the RDMA data path verified (§9)."""
    if not reachability.ip_reachable:
        return {"claim": "not_reachable", "reason": "IP reachability failed"}
    if data_path is None or not data_path.verified:
        return {
            "claim": "network_reachable_only",
            "reason": "RDMA/device-direct path not verified by a data-path probe",
        }
    if data_path.transport.lower() in ("tcp", "socket", "ethernet"):
        return {"claim": "network_reachable_only", "reason": f"transport is {data_path.transport}"}
    return {"claim": "data_path_verified", "reason": f"transport={data_path.transport}"}


#: The probe matrix of details E10-01 §6 (positive + negative/boundary probes).
PROBE_MATRIX: Tuple[Mapping[str, str], ...] = (
    {"layer": "inventory", "positive": "all devices uniquely enumerated", "negative": "duplicate/missing UUID fixture"},
    {"layer": "pcie_fabric", "positive": "link up, width/speed", "negative": "degraded-width / link-down read-only detection"},
    {"layer": "p2p", "positive": "every pair read/write", "negative": "unreachable pair marked UNSUPPORTED"},
    {"layer": "numa", "positive": "CPU/memory affinity", "negative": "cross-socket placement"},
    {"layer": "nic_rdma", "positive": "port/GID/state/capability", "negative": "TCP fallback / invisible device"},
    {"layer": "launcher", "positive": "one rank per device", "negative": "remap, duplicate binding, omission"},
    {"layer": "process_group", "positive": "ordered rank list", "negative": "planned/actual group difference"},
    {"layer": "reproducibility", "positive": "new job isomorphic", "negative": "allowed field-level diff"},
)


# ── preflight fixtures (negative mapping cases, step 26) ───────────────────


#: Reject codes produced before communicator initialisation.
PREFLIGHT_CODES: Tuple[str, ...] = (
    "DUPLICATE_DEVICE",
    "UNKNOWN_DEVICE_UUID",
    "RANK_GAP",
    "WORLD_SIZE_MISMATCH",
    "GROUP_ORDER_MISMATCH",
    "UNPLANNED_RANK",
)


@dataclass(frozen=True)
class PreflightRejection:
    code: str
    detail: str
    global_rank: Optional[int] = None
    group_id: str = ""

    def __post_init__(self) -> None:
        if self.code not in PREFLIGHT_CODES:
            raise ConfigError(
                f"unknown preflight code {self.code!r}", details={"field": "code"}
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "global_rank": self.global_rank,
            "group_id": self.group_id,
            "detected_before_communicator_init": True,
        }


@dataclass
class PreflightResult:
    ok: bool
    rejections: Tuple[PreflightRejection, ...] = ()
    checked_ranks: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checked_ranks": self.checked_ranks,
            "rejections": [item.as_dict() for item in self.rejections],
        }


def preflight_rank_mapping(
    plan: PlacementPlan,
    identities: Sequence[RankIdentity],
    *,
    known_device_uuids: Optional[Sequence[str]] = None,
    one_rank_per_device: bool = True,
) -> PreflightResult:
    """Run the mapping fixtures before any communicator exists."""
    rejections: List[PreflightRejection] = []

    ranks = [item.global_rank for item in identities]
    if len(set(ranks)) != len(ranks):
        rejections.append(PreflightRejection("RANK_GAP", f"duplicate global ranks {sorted(ranks)}"))
    expected = set(range(plan.world_size))
    missing = sorted(expected - set(ranks))
    if missing:
        rejections.append(
            PreflightRejection("RANK_GAP", f"missing global ranks {missing}")
        )
    extra = sorted(set(ranks) - expected)
    if extra:
        rejections.append(
            PreflightRejection("UNPLANNED_RANK", f"ranks outside the plan: {extra}")
        )
    if len(identities) != plan.world_size:
        rejections.append(
            PreflightRejection(
                "WORLD_SIZE_MISMATCH",
                f"observed {len(identities)} ranks, plan says {plan.world_size}",
            )
        )

    if one_rank_per_device:
        by_device: Dict[str, List[int]] = {}
        for item in identities:
            by_device.setdefault(item.device_uuid, []).append(item.global_rank)
        for device, holders in sorted(by_device.items()):
            if len(holders) > 1:
                rejections.append(
                    PreflightRejection(
                        "DUPLICATE_DEVICE",
                        f"device {device} bound by ranks {sorted(holders)}",
                        global_rank=sorted(holders)[0],
                    )
                )

    planned_uuids = set(plan.planned_device_by_rank().values())
    known = set(known_device_uuids) if known_device_uuids is not None else None
    for item in identities:
        if known is not None and item.device_uuid not in known:
            rejections.append(
                PreflightRejection(
                    "UNKNOWN_DEVICE_UUID",
                    f"rank {item.global_rank} reports uuid {item.device_uuid!r} that the topology "
                    "manifest does not contain",
                    global_rank=item.global_rank,
                )
            )
        if item.device_uuid not in planned_uuids:
            rejections.append(
                PreflightRejection(
                    "UNKNOWN_DEVICE_UUID",
                    f"rank {item.global_rank} device is not in the plan",
                    global_rank=item.global_rank,
                )
            )
    return PreflightResult(
        ok=not rejections, rejections=tuple(rejections), checked_ranks=len(identities)
    )


def preflight_group_order(
    groups: Sequence[GroupMembership], planned_order: Sequence[str]
) -> PreflightResult:
    """Reject a group creation-order mismatch before it can deadlock (§3.2)."""
    rejections: List[PreflightRejection] = []
    actual = [group.group_id for group in sorted(groups, key=lambda item: item.creation_sequence)]
    if list(planned_order) != actual:
        rejections.append(
            PreflightRejection(
                "GROUP_ORDER_MISMATCH",
                f"planned order {list(planned_order)}, observed {actual}",
            )
        )
    return PreflightResult(ok=not rejections, rejections=tuple(rejections), checked_ranks=0)


def mapping_fixture_cases() -> Tuple[Dict[str, Any], ...]:
    """The four negative fixtures of step 26, described for the driver/smoke path."""
    return (
        {"fixture": "duplicate_device", "expected_code": "DUPLICATE_DEVICE"},
        {"fixture": "unknown_uuid", "expected_code": "UNKNOWN_DEVICE_UUID"},
        {"fixture": "rank_gap", "expected_code": "RANK_GAP"},
        {"fixture": "wrong_group_order", "expected_code": "GROUP_ORDER_MISMATCH"},
    )


def make_duplicate_device_identities(plan: PlacementPlan) -> Tuple[RankIdentity, ...]:
    """Build the duplicate-device fixture without touching hardware."""
    entries = plan.entries[: min(2, len(plan.entries))]
    if len(entries) < 2:
        raise ConfigError("the duplicate-device fixture needs at least two planned ranks")
    first, second = entries[0], entries[1]
    return (
        RankIdentity(
            global_rank=first.global_rank,
            local_rank=first.local_rank,
            node_rank=first.node_rank,
            node_id="fixture-node",
            pid=1000,
            device_uuid=first.planned_device_uuid,
        ),
        RankIdentity(
            global_rank=second.global_rank,
            local_rank=second.local_rank,
            node_rank=second.node_rank,
            node_id="fixture-node",
            pid=1001,
            device_uuid=first.planned_device_uuid,  # deliberately wrong
        ),
    )


__all__ = [
    "BACKEND_ENV_KEYS",
    "CommandExecutor",
    "CopyProbeRow",
    "DataPathEvidence",
    "FixtureExecutor",
    "HostDeviceProbe",
    "P2PCapability",
    "P2P_STATES",
    "PREFLIGHT_CODES",
    "PROBE_COMMANDS",
    "PROBE_MATRIX",
    "PROBE_STATUSES",
    "PreflightRejection",
    "PreflightResult",
    "ProbeOutcome",
    "RdmaReachability",
    "SubprocessExecutor",
    "capture_backend_env",
    "make_duplicate_device_identities",
    "mapping_fixture_cases",
    "preflight_group_order",
    "preflight_rank_mapping",
    "probe_chain_status",
    "rdma_claim_guard",
    "run_inventory_probes",
    "unavailable",
]
