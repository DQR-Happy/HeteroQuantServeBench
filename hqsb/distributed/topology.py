"""Topology identity: the machine-readable physical fact base for S10 (E10-01).

``TopologyGraph`` is not a screenshot: every node and edge carries a *source*,
an observation timestamp and a confidence level, and every field that cannot be
queried is recorded as ``UNAVAILABLE(reason)`` rather than ``0`` (details
README §4.2, §7).

The module provides:

* the manifest records (host/NUMA/accelerator/PCIe/fabric/NIC/RDMA/software);
* schema validation (dangling nodes, inconsistent duplicate edges, unauthorised
  ``UNAVAILABLE``);
* canonical JSON + SHA-256 sealing, so later runs can judge equality and diff
  drift field by field;
* the six-level drift classification of details README §12 with the downstream
  action attached to each level;
* the hard registration invariants of §12 (unique ranks, one rank per device,
  planned world size, group membership, fast-path edges not DOWN/UNKNOWN).

Nothing in this module runs a probe by itself: it normalises and validates what
:mod:`hqsb.distributed.probes` collected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

#: Prefix used for every value that could not be observed.
UNAVAILABLE_PREFIX = "UNAVAILABLE("

#: The only branch values the scope may declare (details E10-01 step 1).
BRANCHES: Tuple[str, ...] = ("cuda_nccl", "ascend_hccl", "both")

#: Node kinds of the directed multigraph (details README §7).
NODE_KINDS: Tuple[str, ...] = (
    "accelerator",
    "cpu_numa",
    "host_memory",
    "pcie_switch",
    "nvswitch_or_hccs",
    "nic_port",
    "network_fabric",
    "host",
)

#: Edge types of the directed multigraph (details README §7).
EDGE_TYPES: Tuple[str, ...] = (
    "pcie",
    "nvlink",
    "hccs",
    "roce_rdma",
    "ethernet",
    "cpu_interconnect",
)

#: Edge confidence levels, ascending (details E10-01 step 18).
EDGE_CONFIDENCE: Tuple[str, ...] = (
    "DECLARED",
    "QUERY_VERIFIED",
    "DATA_PATH_VERIFIED",
    "MEASURED",
    "DEGRADED",
)

#: Link states; ``DOWN``/``UNKNOWN`` may never be used as a fast path.
LINK_STATES: Tuple[str, ...] = ("UP", "DOWN", "DEGRADED", "UNKNOWN")

#: Drift levels with the downstream action (details README §12).
DRIFT_ACTIONS: Mapping[str, str] = {
    "IDENTITY_ONLY": "keep the diff; do not invalidate performance results",
    "HEALTH_CONTEXT": "include/exclude by the pre-registered health range",
    "PLACEMENT_RELEVANT": "re-verify rank placement and a minimal collective",
    "PERFORMANCE_RELEVANT": "old E10-02/05 performance is not pairable; re-measure",
    "COMPATIBILITY_BREAKING": "re-run the E10-01..03 gates",
    "TOPOLOGY_CLASS_CHANGE": "new experiment family; do not connect old curves",
}

#: Field path (canonical JSON pointer-ish prefix) → drift level.
#: Order matters: the first matching rule wins.
DRIFT_FIELD_RULES: Tuple[Tuple[str, str], ...] = (
    ("host.scheduler_job_id", "IDENTITY_ONLY"),
    ("host.pid", "IDENTITY_ONLY"),
    ("collected_at", "IDENTITY_ONLY"),
    ("host.boot_time", "IDENTITY_ONLY"),
    ("health.", "HEALTH_CONTEXT"),
    ("accelerators[].temperature_c", "HEALTH_CONTEXT"),
    ("numa[].memory_free_bytes", "HEALTH_CONTEXT"),
    ("edges[].error_counter", "HEALTH_CONTEXT"),
    ("visibility.", "PLACEMENT_RELEVANT"),
    ("placement.", "PLACEMENT_RELEVANT"),
    ("accelerators[].visible_index", "PLACEMENT_RELEVANT"),
    ("nics[].rail", "PLACEMENT_RELEVANT"),
    ("affinity[].", "PLACEMENT_RELEVANT"),
    ("edges[].p2p_", "PLACEMENT_RELEVANT"),
    ("pcie[]", "PERFORMANCE_RELEVANT"),
    ("fabric[]", "PERFORMANCE_RELEVANT"),
    ("edges[].status", "PERFORMANCE_RELEVANT"),
    ("edges[].confidence", "PERFORMANCE_RELEVANT"),
    ("edges[].measured_", "PERFORMANCE_RELEVANT"),
    ("edges[].source_tool", "PERFORMANCE_RELEVANT"),
    ("edges[].error_counter", "HEALTH_CONTEXT"),
    ("nics[].speed_gbps", "PERFORMANCE_RELEVANT"),
    ("nics[].state", "PERFORMANCE_RELEVANT"),
    ("rdma.", "PERFORMANCE_RELEVANT"),
    ("software.backend", "COMPATIBILITY_BREAKING"),
    ("software.backend_version", "COMPATIBILITY_BREAKING"),
    ("accelerators[].driver", "COMPATIBILITY_BREAKING"),
    ("accelerators[].firmware", "COMPATIBILITY_BREAKING"),
    ("scope.node_scope", "TOPOLOGY_CLASS_CHANGE"),
    ("scope.branch", "TOPOLOGY_CLASS_CHANGE"),
    ("fabric[].fabric_type", "TOPOLOGY_CLASS_CHANGE"),
    ("nodes[].kind", "TOPOLOGY_CLASS_CHANGE"),
)

#: Degraded/unknown edge policies (details E10-01 step 27).
DEGRADED_ACTIONS: Tuple[str, ...] = ("DENY", "AVOID", "EXPLICIT_ALLOW")


def unavailable(reason: str) -> str:
    """Build an explicit ``UNAVAILABLE(reason)`` marker."""
    if not reason:
        raise ConfigError("an UNAVAILABLE marker needs a reason (never a silent blank)")
    return f"{UNAVAILABLE_PREFIX}{reason})"


def is_unavailable(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(UNAVAILABLE_PREFIX)


def _require(condition: bool, message: str, *, field_name: str = "") -> None:
    if not condition:
        raise ConfigError(message, details={"field": field_name} if field_name else None)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    """Deterministic JSON used for hashing and field-level diffing."""
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


def manifest_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256_text(canonical_json(payload))


def redact_identity(text: str, aliases: Optional[Mapping[str, str]] = None) -> str:
    """Redact sensitive fragments while keeping a stable alias.

    Details E10-01 step 2 asks for IP/user fields to be masked but a stable
    alias kept, so a re-collected manifest stays comparable.
    """
    if not text:
        return text
    aliases = dict(aliases or {})
    out = text
    for secret, alias in aliases.items():
        if secret:
            out = out.replace(secret, alias)
    return out


# ── scope / host / NUMA ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ObservationScope:
    """The declared experiment scope (details E10-01 step 1)."""

    branch: str
    node_scope: str  # "single_node" | "multi_node"
    declared_at: str = ""
    notes: str = ""
    missing_resource_branches: Tuple[str, ...] = ()

    def validate(self) -> None:
        _require(self.branch in BRANCHES, f"branch must be one of {BRANCHES}", field_name="branch")
        _require(
            self.node_scope in ("single_node", "multi_node"),
            "node_scope must be single_node or multi_node",
            field_name="node_scope",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "branch": self.branch,
            "node_scope": self.node_scope,
            "declared_at": self.declared_at,
            "notes": self.notes,
            "missing_resource_branches": list(self.missing_resource_branches),
        }


@dataclass(frozen=True)
class HostIdentity:
    """Host/scheduler identity (details E10-01 step 2)."""

    node_id: str
    host_alias: str
    cpu_arch: str
    os_release: str
    kernel: str
    container_digest: str = ""
    scheduler_job_id: str = ""
    boot_time: str = ""
    collected_at: str = ""
    redacted_fields: Tuple[str, ...] = ()

    def validate(self) -> None:
        _require(bool(self.node_id), "host identity needs a node_id", field_name="node_id")
        _require(bool(self.host_alias), "host identity needs a stable alias", field_name="host_alias")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host_alias": self.host_alias,
            "cpu_arch": self.cpu_arch,
            "os_release": self.os_release,
            "kernel": self.kernel,
            "container_digest": self.container_digest,
            "scheduler_job_id": self.scheduler_job_id,
            "boot_time": self.boot_time,
            "collected_at": self.collected_at,
            "redacted_fields": list(self.redacted_fields),
        }


@dataclass(frozen=True)
class NumaNode:
    """One CPU socket / NUMA node (details E10-01 step 3)."""

    node_id: int
    socket: int
    logical_cpus: Tuple[int, ...]
    memory_total_bytes: Any
    memory_free_bytes: Any
    distance: Mapping[int, Any] = field(default_factory=dict)
    online: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "socket": self.socket,
            "logical_cpus": list(self.logical_cpus),
            "memory_total_bytes": self.memory_total_bytes,
            "memory_free_bytes": self.memory_free_bytes,
            "distance": {str(k): v for k, v in self.distance.items()},
            "online": self.online,
        }


@dataclass
class NumaTopology:
    """CPU socket/NUMA graph plus a complete core → NUMA mapping."""

    nodes: Tuple[NumaNode, ...]
    core_to_numa: Mapping[int, int]

    def validate(self) -> List[str]:
        errors: List[str] = []
        ids = [node.node_id for node in self.nodes]
        if len(set(ids)) != len(ids):
            errors.append("duplicate NUMA node ids")
        for node in self.nodes:
            if not node.logical_cpus:
                errors.append(f"NUMA node {node.node_id} lists no logical CPU")
            for cpu in node.logical_cpus:
                mapped = self.core_to_numa.get(cpu)
                if mapped is None:
                    errors.append(f"logical CPU {cpu} has no NUMA mapping")
                elif mapped != node.node_id:
                    errors.append(
                        f"logical CPU {cpu} maps to NUMA {mapped} but is listed under {node.node_id}"
                    )
        for cpu, numa in self.core_to_numa.items():
            if numa not in ids:
                errors.append(f"core_to_numa references unknown NUMA node {numa} for CPU {cpu}")
        return errors

    def as_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [node.as_dict() for node in self.nodes],
            "core_to_numa": {str(k): v for k, v in sorted(self.core_to_numa.items())},
        }


# ── accelerators / PCIe / fabric / affinity / NIC / RDMA / software ────────


@dataclass(frozen=True)
class AcceleratorRecord:
    """One physical accelerator (details E10-01 step 4)."""

    device_id: int
    sku: str
    uuid: str
    serial: str = ""
    pci_bdf: str = ""
    memory_bytes: Any = ""
    firmware: str = ""
    driver: str = ""
    health: str = "UNKNOWN"
    temperature_c: Any = ""
    in_use_by_other_process: bool = False
    visible_index: Any = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "sku": self.sku,
            "uuid": self.uuid,
            "serial": self.serial,
            "pci_bdf": self.pci_bdf,
            "memory_bytes": self.memory_bytes,
            "firmware": self.firmware,
            "driver": self.driver,
            "health": self.health,
            "temperature_c": self.temperature_c,
            "in_use_by_other_process": self.in_use_by_other_process,
            "visible_index": self.visible_index,
        }


def detect_identity_problems(records: Sequence[AcceleratorRecord]) -> List[str]:
    """Duplicate/missing accelerator identity detection (E10-01 step 4)."""
    problems: List[str] = []
    uuids: Dict[str, int] = {}
    for record in records:
        if not record.uuid or is_unavailable(record.uuid):
            problems.append(f"device {record.device_id} has no usable UUID")
            continue
        uuids[record.uuid] = uuids.get(record.uuid, 0) + 1
    for uuid, count in sorted(uuids.items()):
        if count > 1:
            problems.append(f"UUID {uuid} appears {count} times")
    return problems


@dataclass(frozen=True)
class VisibleDeviceAudit:
    """Visible-device remapping audit (details E10-01 step 5).

    ``launcher_env`` is what the launcher wrote (e.g.
    ``CUDA_VISIBLE_DEVICES=1,3``); ``runtime_visible`` is what the process sees;
    ``current_device_uuid`` is what the process actually bound to.  Guessing a
    physical card from a local index is explicitly forbidden.
    """

    launcher_env: Mapping[str, str]
    runtime_visible: Tuple[str, ...]
    current_device_uuid: str
    local_index: int

    def audit(self) -> List[str]:
        issues: List[str] = []
        if not self.runtime_visible:
            issues.append("runtime reports no visible device")
        if self.local_index < 0 or self.local_index >= max(len(self.runtime_visible), 1):
            issues.append(f"local index {self.local_index} outside the visible list")
        elif self.runtime_visible[self.local_index] != self.current_device_uuid:
            issues.append(
                "local index resolves to "
                f"{self.runtime_visible[self.local_index]} but the process reports "
                f"{self.current_device_uuid}"
            )
        return issues

    def as_dict(self) -> Dict[str, Any]:
        return {
            "launcher_env": dict(self.launcher_env),
            "runtime_visible": list(self.runtime_visible),
            "current_device_uuid": self.current_device_uuid,
            "local_index": self.local_index,
        }


@dataclass(frozen=True)
class PcieLink:
    """One PCIe path segment (details E10-01 step 6)."""

    endpoint_bdf: str
    switch_bdf: str = ""
    root_complex: str = ""
    generation: Any = ""
    nominal_width: Any = ""
    negotiated_width: Any = ""
    nominal_speed_gts: Any = ""
    negotiated_speed_gts: Any = ""
    hop_class: str = ""
    aer_status: str = "UNKNOWN"
    error_count: int = 0

    def degraded(self) -> bool:
        if self.aer_status not in ("OK", "UNKNOWN", ""):
            return True
        try:
            return int(self.negotiated_width) < int(self.nominal_width)
        except (TypeError, ValueError):
            return False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "endpoint_bdf": self.endpoint_bdf,
            "switch_bdf": self.switch_bdf,
            "root_complex": self.root_complex,
            "generation": self.generation,
            "nominal_width": self.nominal_width,
            "negotiated_width": self.negotiated_width,
            "nominal_speed_gts": self.nominal_speed_gts,
            "negotiated_speed_gts": self.negotiated_speed_gts,
            "hop_class": self.hop_class,
            "aer_status": self.aer_status,
            "error_count": self.error_count,
        }


@dataclass(frozen=True)
class FabricLink:
    """NVLink/NVSwitch/HCCS link (details E10-01 step 7)."""

    src_device: str
    dst_device: str
    fabric_type: str
    ports: int = 0
    state: str = "UNKNOWN"
    width: Any = ""
    speed_gbps: Any = ""
    remote: str = ""
    error_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "src_device": self.src_device,
            "dst_device": self.dst_device,
            "fabric_type": self.fabric_type,
            "ports": self.ports,
            "state": self.state,
            "width": self.width,
            "speed_gbps": self.speed_gbps,
            "remote": self.remote,
            "error_count": self.error_count,
        }


@dataclass(frozen=True)
class AffinityRecord:
    """Accelerator ↔ CPU/memory NUMA affinity (details E10-01 step 8)."""

    accelerator_uuid: str
    cpu_numa_node: Any
    memory_numa_node: Any
    distance: Any = ""
    cross_socket: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accelerator_uuid": self.accelerator_uuid,
            "cpu_numa_node": self.cpu_numa_node,
            "memory_numa_node": self.memory_numa_node,
            "distance": self.distance,
            "cross_socket": self.cross_socket,
        }


@dataclass(frozen=True)
class NicRecord:
    """One NIC port (details E10-01 step 9)."""

    interface: str
    pci_bdf: str = ""
    numa_node: Any = ""
    port: int = 0
    link_layer: str = ""
    speed_gbps: Any = ""
    mtu: Any = ""
    state: str = "UNKNOWN"
    gid: str = ""
    rail: str = ""
    error_counters: Mapping[str, int] = field(default_factory=dict)
    reachable: str = "UNKNOWN"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "interface": self.interface,
            "pci_bdf": self.pci_bdf,
            "numa_node": self.numa_node,
            "port": self.port,
            "link_layer": self.link_layer,
            "speed_gbps": self.speed_gbps,
            "mtu": self.mtu,
            "state": self.state,
            "gid": self.gid,
            "rail": self.rail,
            "error_counters": dict(self.error_counters),
            "reachable": self.reachable,
        }


@dataclass(frozen=True)
class RdmaStackRecord:
    """RDMA/RoCE/IB software stack (details E10-01 step 10)."""

    rdma_core_version: str = ""
    ofed_version: str = ""
    driver: str = ""
    firmware: str = ""
    ucx_version: str = ""
    mpi_version: str = ""
    pfc_visible: Any = ""
    ecn_visible: Any = ""
    traffic_class: Any = ""
    container_capabilities: Tuple[str, ...] = ()
    device_direct_available: Any = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rdma_core_version": self.rdma_core_version,
            "ofed_version": self.ofed_version,
            "driver": self.driver,
            "firmware": self.firmware,
            "ucx_version": self.ucx_version,
            "mpi_version": self.mpi_version,
            "pfc_visible": self.pfc_visible,
            "ecn_visible": self.ecn_visible,
            "traffic_class": self.traffic_class,
            "container_capabilities": list(self.container_capabilities),
            "device_direct_available": self.device_direct_available,
        }


@dataclass(frozen=True)
class BackendRuntimeConfig:
    """NCCL/HCCL runtime configuration (details E10-01 step 11)."""

    backend: str
    version: str
    env_vars: Mapping[str, str] = field(default_factory=dict)
    interface_selection: Any = ""
    algorithm_overrides: Mapping[str, str] = field(default_factory=dict)
    protocol_overrides: Mapping[str, str] = field(default_factory=dict)
    debug_level: str = ""
    defaults_explained_by: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "version": self.version,
            "env_vars": dict(sorted(self.env_vars.items())),
            "interface_selection": self.interface_selection,
            "algorithm_overrides": dict(sorted(self.algorithm_overrides.items())),
            "protocol_overrides": dict(sorted(self.protocol_overrides.items())),
            "debug_level": self.debug_level,
            "defaults_explained_by": self.defaults_explained_by,
        }


# ── the graph ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TopologyNode:
    node_id: str
    kind: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "kind": self.kind, "attributes": dict(self.attributes)}


@dataclass(frozen=True)
class TopologyEdge:
    """A directed edge with evidence and confidence (details README §7)."""

    src: str
    dst: str
    edge_type: str
    direction: str = "bidirectional"
    nominal_capacity: Any = ""
    negotiated_capacity: Any = ""
    hop_class: str = ""
    numa_distance: Any = ""
    p2p_read: str = "UNKNOWN"
    p2p_write: str = "UNKNOWN"
    p2p_atomic: str = "UNKNOWN"
    rdma_capable: Any = "UNKNOWN"
    status: str = "UNKNOWN"
    error_counter: int = 0
    source_tool: str = ""
    observed_at: str = ""
    confidence: str = "DECLARED"
    evidence_uri: str = ""
    measured_latency_us: Any = ""
    measured_bandwidth_gbps: Any = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "edge_type": self.edge_type,
            "direction": self.direction,
            "nominal_capacity": self.nominal_capacity,
            "negotiated_capacity": self.negotiated_capacity,
            "hop_class": self.hop_class,
            "numa_distance": self.numa_distance,
            "p2p_read": self.p2p_read,
            "p2p_write": self.p2p_write,
            "p2p_atomic": self.p2p_atomic,
            "rdma_capable": self.rdma_capable,
            "status": self.status,
            "error_counter": self.error_counter,
            "source_tool": self.source_tool,
            "observed_at": self.observed_at,
            "confidence": self.confidence,
            "evidence_uri": self.evidence_uri,
            "measured_latency_us": self.measured_latency_us,
            "measured_bandwidth_gbps": self.measured_bandwidth_gbps,
        }

    def is_fast_path_candidate(self) -> bool:
        return self.edge_type in ("nvlink", "hccs", "roce_rdma")


@dataclass
class TopologyManifest:
    """The normalised, hashed topology record handed to E10-02..E10-10."""

    scope: ObservationScope
    host: HostIdentity
    numa: NumaTopology
    accelerators: Tuple[AcceleratorRecord, ...]
    pcie: Tuple[PcieLink, ...] = ()
    fabric: Tuple[FabricLink, ...] = ()
    affinity: Tuple[AffinityRecord, ...] = ()
    nics: Tuple[NicRecord, ...] = ()
    rdma: Optional[RdmaStackRecord] = None
    software: Tuple[BackendRuntimeConfig, ...] = ()
    visible_audits: Tuple[VisibleDeviceAudit, ...] = ()
    nodes: Tuple[TopologyNode, ...] = ()
    edges: Tuple[TopologyEdge, ...] = ()
    raw_evidence: Mapping[str, str] = field(default_factory=dict)
    collected_at: str = ""
    schema_version: str = "1.0.0"

    # ── schema ──
    def validate(self) -> List[str]:
        errors: List[str] = []
        self.scope.validate()
        self.host.validate()
        errors.extend(self.numa.validate())

        node_ids = {node.node_id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            errors.append("duplicate topology node_id")
        for node in self.nodes:
            if node.kind not in NODE_KINDS:
                errors.append(f"node {node.node_id} has unknown kind {node.kind!r}")

        seen: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for edge in self.edges:
            if edge.src not in node_ids:
                errors.append(f"edge {edge.src}->{edge.dst} dangles: unknown src node")
            if edge.dst not in node_ids:
                errors.append(f"edge {edge.src}->{edge.dst} dangles: unknown dst node")
            if edge.edge_type not in EDGE_TYPES:
                errors.append(f"edge {edge.src}->{edge.dst} has unknown type {edge.edge_type!r}")
            if edge.status not in LINK_STATES:
                errors.append(f"edge {edge.src}->{edge.dst} has unknown status {edge.status!r}")
            if edge.confidence not in EDGE_CONFIDENCE:
                errors.append(
                    f"edge {edge.src}->{edge.dst} has unknown confidence {edge.confidence!r}"
                )
            if edge.confidence in ("MEASURED", "DATA_PATH_VERIFIED"):
                if is_unavailable(edge.source_tool) or not edge.source_tool:
                    errors.append(
                        f"edge {edge.src}->{edge.dst} claims {edge.confidence} without a source tool"
                    )
                if edge.confidence == "MEASURED" and (
                    is_unavailable(edge.measured_latency_us) or edge.measured_latency_us == ""
                ):
                    errors.append(
                        f"edge {edge.src}->{edge.dst} claims MEASURED without a measured latency"
                    )
            key = (edge.src, edge.dst, edge.edge_type, edge.direction)
            previous = seen.get(key)
            payload = edge.as_dict()
            if previous is not None and previous != payload:
                errors.append(
                    f"duplicate edge {key} with inconsistent payload "
                    "(same endpoints/type/direction, different fields)"
                )
            seen[key] = payload
        if not self.accelerators:
            errors.append("manifest enumerates zero accelerators")
        errors.extend(detect_identity_problems(self.accelerators))
        for audit in self.visible_audits:
            errors.extend(audit.audit())
        return errors

    # ── serialisation ──
    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "collected_at": self.collected_at,
            "scope": self.scope.as_dict(),
            "host": self.host.as_dict(),
            "numa": self.numa.as_dict(),
            "accelerators": [item.as_dict() for item in self.accelerators],
            "pcie": [item.as_dict() for item in self.pcie],
            "fabric": [item.as_dict() for item in self.fabric],
            "affinity": [item.as_dict() for item in self.affinity],
            "nics": [item.as_dict() for item in self.nics],
            "rdma": self.rdma.as_dict() if self.rdma else None,
            "software": [item.as_dict() for item in self.software],
            "visible_audits": [item.as_dict() for item in self.visible_audits],
            "nodes": [item.as_dict() for item in self.nodes],
            "edges": [item.as_dict() for item in self.edges],
            "raw_evidence": dict(sorted(self.raw_evidence.items())),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.as_dict())

    @property
    def sha256(self) -> str:
        return _sha256_text(self.canonical_json())

    def degraded_edges(self) -> List[TopologyEdge]:
        return [
            edge
            for edge in self.edges
            if edge.status in ("DOWN", "DEGRADED", "UNKNOWN")
            or edge.confidence == "DEGRADED"
        ]


# ── drift ──────────────────────────────────────────────────────────────────


def classify_drift(field_path: str) -> str:
    """Map a changed field path to one of the six drift levels (§12)."""
    for prefix, level in DRIFT_FIELD_RULES:
        if field_path == prefix or field_path.startswith(prefix):
            return level
        # ``accelerators[].x`` matches ``accelerators[3].x``
        if "[]" in prefix:
            head, _, tail = prefix.partition("[]")
            if field_path.startswith(head + "[") and field_path.endswith(tail) and tail:
                return level
    return "PLACEMENT_RELEVANT"


def _flatten(payload: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(payload, Mapping):
        for key in sorted(payload):
            flat.update(_flatten(payload[key], f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            flat.update(_flatten(item, f"{prefix}[{index}]"))
    else:
        flat[prefix] = payload
    return flat


@dataclass(frozen=True)
class DriftField:
    path: str
    left: Any
    right: Any
    level: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "left": self.left,
            "right": self.right,
            "level": self.level,
            "downstream_action": DRIFT_ACTIONS[self.level],
        }


def diff_manifests(left: TopologyManifest, right: TopologyManifest) -> Dict[str, Any]:
    """Field-level manifest diff grouped by drift level (E10-01 step 28)."""
    flat_left = _flatten(left.as_dict())
    flat_right = _flatten(right.as_dict())
    paths = sorted(set(flat_left) | set(flat_right))
    fields: List[DriftField] = []
    for path in paths:
        if flat_left.get(path) != flat_right.get(path):
            fields.append(
                DriftField(
                    path=path,
                    left=flat_left.get(path, "<missing>"),
                    right=flat_right.get(path, "<missing>"),
                    level=classify_drift(path),
                )
            )
    by_level: Dict[str, List[Dict[str, Any]]] = {level: [] for level in DRIFT_ACTIONS}
    for item in fields:
        by_level[item.level].append(item.as_dict())
    highest = max(
        (
            level
            for level, rows in by_level.items()
            if rows
        ),
        key=lambda level: list(DRIFT_ACTIONS).index(level),
        default="IDENTITY_ONLY",
    )
    return {
        "same_topology_class": "TOPOLOGY_CLASS_CHANGE" not in {
            item.level for item in fields
        },
        "highest_level": highest,
        "action": DRIFT_ACTIONS[highest],
        "field_count": len(fields),
        "levels": by_level,
        "reuse_allowed": highest
        in ("IDENTITY_ONLY", "HEALTH_CONTEXT"),
    }


# ── degraded link policy ───────────────────────────────────────────────────


@dataclass(frozen=True)
class DegradedLinkPolicy:
    """How an edge that is DOWN/DEGRADED/UNKNOWN is treated (step 27)."""

    action: str = "DENY"
    explicit_allow_edges: Tuple[Tuple[str, str], ...] = ()
    reason: str = ""

    def validate(self) -> None:
        _require(
            self.action in DEGRADED_ACTIONS,
            f"degraded action must be one of {DEGRADED_ACTIONS}",
            field_name="action",
        )
        if self.action == "EXPLICIT_ALLOW" and not self.reason:
            raise ConfigError(
                "EXPLICIT_ALLOW on a degraded/unknown link needs a written reason",
                details={"field": "reason"},
            )

    def decide(self, edge: TopologyEdge) -> Dict[str, Any]:
        self.validate()
        bad = edge.status in ("DOWN", "DEGRADED", "UNKNOWN") or edge.confidence == "DEGRADED"
        if not bad:
            return {"edge": f"{edge.src}->{edge.dst}", "action": "ALLOW", "reason": "healthy"}
        if self.action == "EXPLICIT_ALLOW" and (edge.src, edge.dst) in self.explicit_allow_edges:
            return {
                "edge": f"{edge.src}->{edge.dst}",
                "action": "EXPLICIT_ALLOW",
                "reason": self.reason,
            }
        return {
            "edge": f"{edge.src}->{edge.dst}",
            "action": self.action,
            "reason": f"status={edge.status} confidence={edge.confidence}",
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "explicit_allow_edges": [list(item) for item in self.explicit_allow_edges],
            "reason": self.reason,
        }


def audit_degraded_edges(
    manifest: TopologyManifest, policy: DegradedLinkPolicy
) -> Dict[str, Any]:
    decisions = [policy.decide(edge) for edge in manifest.edges]
    blocked = [row for row in decisions if row["action"] in ("DENY", "AVOID")]
    return {"decisions": decisions, "blocked": blocked, "ok": not blocked}


# ── hard invariants / E10-02 gate ──────────────────────────────────────────


#: The machine-checked registration invariants of details README §12.
HARD_INVARIANTS: Tuple[str, ...] = (
    "unique(global_rank)",
    "unique(process_id within job)",
    "unique(device_uuid for one-rank-per-device plan)",
    "planned_world_size == observed_rank_count",
    "all(group members exist in observed ranks)",
    "all(planned fast-path edges are not DOWN/UNKNOWN)",
    "actual_device_uuid(rank) == planned_device_uuid(rank)",
)


def check_hard_invariants(
    *,
    identities: Sequence[Any],
    planned_world_size: Optional[int] = None,
    one_rank_per_device: bool = True,
    groups: Sequence[Any] = (),
    manifest: Optional[TopologyManifest] = None,
    planned_device_by_rank: Optional[Mapping[int, str]] = None,
) -> Dict[str, Any]:
    """Check the seven hard invariants over observed rank identities."""
    violations: List[str] = []

    ranks = [getattr(item, "global_rank") for item in identities]
    if len(set(ranks)) != len(ranks):
        violations.append("unique(global_rank): duplicate global rank observed")

    pids = [(getattr(item, "node_id", ""), getattr(item, "pid", 0)) for item in identities]
    if len(set(pids)) != len(pids):
        violations.append("unique(process_id within job): duplicate (node, pid) observed")

    if one_rank_per_device:
        devices = [getattr(item, "device_uuid", "") for item in identities]
        if len(set(devices)) != len(devices):
            violations.append(
                "unique(device_uuid for one-rank-per-device plan): two ranks share a device"
            )

    if planned_world_size is not None and len(identities) != planned_world_size:
        violations.append(
            f"planned_world_size == observed_rank_count: planned {planned_world_size}, "
            f"observed {len(identities)}"
        )

    observed_ranks = set(ranks)
    for group in groups:
        members = set(getattr(group, "ordered_global_ranks", ()))
        missing = sorted(members - observed_ranks)
        if missing:
            violations.append(
                f"all(group members exist in observed ranks): group "
                f"{getattr(group, 'group_id', '?')} lists {missing}"
            )

    if planned_device_by_rank:
        for item in identities:
            planned = planned_device_by_rank.get(getattr(item, "global_rank"))
            actual = getattr(item, "device_uuid", "")
            if planned is not None and planned != actual:
                violations.append(
                    f"actual_device_uuid(rank) == planned_device_uuid(rank): rank "
                    f"{getattr(item, 'global_rank')} planned {planned}, actual {actual}"
                )

    if manifest is not None:
        blocked = [
            f"{edge.src}->{edge.dst}"
            for edge in manifest.edges
            if edge.is_fast_path_candidate() and edge.status in ("DOWN", "UNKNOWN")
        ]
        if blocked:
            violations.append(
                f"all(planned fast-path edges are not DOWN/UNKNOWN): {blocked}"
            )

    return {
        "ok": not violations,
        "violations": violations,
        "invariants": list(HARD_INVARIANTS),
    }


@dataclass(frozen=True)
class GateDecision:
    """A gate verdict handed from one experiment to the next."""

    ready: bool
    gate: str
    blockers: Tuple[str, ...] = ()
    evidence: Tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "ready": self.ready,
            "blockers": list(self.blockers),
            "evidence": list(self.evidence),
            "reason": self.reason,
        }


def gate_for_collective_probe(
    manifest: TopologyManifest,
    *,
    policy: Optional[DegradedLinkPolicy] = None,
    placement_ok: bool = False,
    data_path_verified: bool = False,
) -> GateDecision:
    """E10-01 → E10-02 gate (details E10-01 step 30)."""
    blockers: List[str] = []
    errors = manifest.validate()
    if errors:
        blockers.append(f"manifest schema: {errors[0]}")
    if not placement_ok:
        blockers.append("planned/actual placement not verified")
    if not data_path_verified:
        blockers.append("actual communication data path not verified")
    effective_policy = policy or DegradedLinkPolicy(action="DENY")
    degraded = audit_degraded_edges(manifest, effective_policy)
    if not degraded["ok"]:
        blockers.append(
            "degraded/unknown fast-path edge without an explicit policy: "
            + ", ".join(row["edge"] for row in degraded["blocked"][:3])
        )
    return GateDecision(
        ready=not blockers,
        gate="E10-01 -> E10-02",
        blockers=tuple(blockers),
        evidence=(f"manifest.sha256={manifest.sha256[:12]}",),
        reason="" if not blockers else "collective sweep must not start on unverified topology",
    )


__all__ = [
    "AcceleratorRecord",
    "AffinityRecord",
    "BackendRuntimeConfig",
    "BRANCHES",
    "DEGRADED_ACTIONS",
    "DRIFT_ACTIONS",
    "DRIFT_FIELD_RULES",
    "DegradedLinkPolicy",
    "DriftField",
    "EDGE_CONFIDENCE",
    "EDGE_TYPES",
    "FabricLink",
    "GateDecision",
    "HARD_INVARIANTS",
    "HostIdentity",
    "LINK_STATES",
    "NODE_KINDS",
    "NicRecord",
    "NumaNode",
    "NumaTopology",
    "ObservationScope",
    "PcieLink",
    "RdmaStackRecord",
    "TopologyEdge",
    "TopologyManifest",
    "TopologyNode",
    "UNAVAILABLE_PREFIX",
    "VisibleDeviceAudit",
    "audit_degraded_edges",
    "canonical_json",
    "check_hard_invariants",
    "classify_drift",
    "detect_identity_problems",
    "diff_manifests",
    "gate_for_collective_probe",
    "is_unavailable",
    "manifest_sha256",
    "redact_identity",
    "unavailable",
]
