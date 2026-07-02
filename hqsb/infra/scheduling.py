"""E13-03: accelerator placement, capability labels, NUMA/topology and isolation.

Implements ``details/S13/E13-03_*.md`` as data:

* the ``PlacementPlan`` contract of §4 (hard capability vs soft preference);
* node/device inventory and the capability-label schema with provenance, owner
  and TTL — a label the workload can rewrite is not a protection (§3.2, step 4);
* the full ``planned → scheduled → allocated → runtime visible → actually used``
  chain (a resource name is not a capability);
* CPU/cpuset/NUMA, link domain, P2P and rank mapping;
* exclusive/partition/shared semantics and the unauthorized-access evidence;
* negative placements that must be refused *before* the model loads.

Nothing here talks to a scheduler: every function validates records produced by
the (external) cluster tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

EXPERIMENT_ID = "E13-03"
TITLE = "Accelerator 资源调度、Capability/NUMA/Topology 与隔离"
CLAIM_BOUNDARY = (
    "本实验通过证明资源落位与隔离符合声明；不证明模型生命周期、容量控制、autoscaling 或多租户"
    "整体安全（由 E13-04/E13-06/E13-07/E13-11 验证）。"
)

SCHEMA_VERSION = "1.0.0"

#: Scheduling phases recorded per pod (§3.1: filter → score → bind → runtime verify).
SCHEDULER_PHASES: Tuple[str, ...] = ("PENDING", "FILTERED", "SCORED", "BOUND", "ALLOCATED", "RUNTIME_VERIFIED", "UNSCHEDULABLE")

#: How a device is shared (§11 of README; the three modes have different claims).
SHARING_MODES: Tuple[str, ...] = ("EXCLUSIVE", "PARTITION_MIG_OR_VNPU", "TIME_SLICING_SHARED")

#: Partition profiles are opaque identifiers; the profile must be recorded verbatim.
PARTITION_PROFILES: Tuple[str, ...] = ("MIG_1g.5gb", "MIG_2g.10gb", "MIG_3g.20gb", "VNPU_1C", "VNPU_2C", "NONE")

#: Topology Manager policies (§11): the node policy changes admission semantics.
TOPOLOGY_POLICIES: Tuple[str, ...] = ("none", "best-effort", "restricted", "single-numa-node")

#: Link domains a rank mapping may rely on.
LINK_DOMAINS: Tuple[str, ...] = ("NVLink", "PCIe", "XGMI", "HCCS", "SYS", "UNKNOWN")

#: Label classes; only the ``protected`` ones may not be written by a workload.
LABEL_CLASSES: Tuple[str, ...] = ("vendor", "product", "arch", "memory", "precision", "topology", "health", "isolation")

#: Capability labels a workload/kubelet must not be able to forge (step 4/26).
PROTECTED_LABEL_CLASSES: Tuple[str, ...] = ("arch", "memory", "precision", "topology", "health", "isolation")

#: Negative placement cases of steps 23–29.
NEGATIVE_CASES: Tuple[str, ...] = (
    "WRONG_VENDOR",
    "WRONG_ARCH",
    "PRECISION_UNSUPPORTED",
    "MEMORY_INSUFFICIENT",
    "PARTITION_PROFILE_UNAVAILABLE",
    "MISSING_TOLERATION",
    "RESERVED_NODE_TAINT",
    "MISSING_CAPABILITY_LABEL",
    "FORGED_CAPABILITY_LABEL",
    "UNHEALTHY_DEVICE",
    "PLUGIN_RESTART",
    "RANK_DEVICE_MISMATCH",
    "CROSS_NAMESPACE_DEVICE_ACCESS",
    "RESOURCE_RELEASE_LEAK",
)

#: Isolation layers that must be verified separately (§3.4).
ISOLATION_LAYERS: Tuple[str, ...] = (
    "scheduler_accounting",
    "device_visibility",
    "driver_context_memory",
    "hardware_partition",
    "shared_fairness",
    "cache_storage_artifact",
    "telemetry_information",
)

#: Performance metrics a locality A/B may compare (effect, not a single profile).
LOCALITY_METRICS: Tuple[str, ...] = (
    "host_device_copy_bandwidth",
    "ttft",
    "tpot",
    "goodput",
    "host_cpu_utilization",
    "collective_time",
    "compute_wait_time",
)


# ── inventory and capability labels ──────────────────────────────────────


@dataclass
class DeviceRecord:
    """One accelerator (physical device, partition or virtual slice)."""

    device_id: str
    vendor: str = ""
    product: str = ""
    arch: str = ""
    memory_bytes: int = 0
    driver_version: str = ""
    firmware_version: str = ""
    partition_profile: str = "NONE"
    health: str = "unknown"
    numa_node: int = -1
    link_domain: str = "UNKNOWN"
    peer_devices: Tuple[str, ...] = ()
    clocks_mhz: int = 0
    power_cap_w: float = 0.0
    allocatable: bool = True
    source: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("device_id", "vendor", "product", "arch", "driver_version"):
            if not getattr(self, name):
                problems.append(f"device record requires {name!r}")
        if self.memory_bytes <= 0:
            problems.append(f"device {self.device_id}: memory must be positive")
        if self.partition_profile not in PARTITION_PROFILES:
            problems.append(f"device {self.device_id}: unknown partition profile {self.partition_profile!r}")
        if self.link_domain not in LINK_DOMAINS:
            problems.append(f"device {self.device_id}: unknown link domain {self.link_domain!r}")
        if not self.source:
            problems.append(
                f"device {self.device_id}: inventory must record its source (plugin/runtime/vendor CLI)"
            )
        if self.health == "unknown":
            problems.append(
                f"device {self.device_id}: health is unknown — an unknown-health device may not be "
                "advertised as allocatable"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "vendor": self.vendor,
            "product": self.product,
            "arch": self.arch,
            "memory_bytes": self.memory_bytes,
            "driver_version": self.driver_version,
            "firmware_version": self.firmware_version,
            "partition_profile": self.partition_profile,
            "health": self.health,
            "numa_node": self.numa_node,
            "link_domain": self.link_domain,
            "peer_devices": list(self.peer_devices),
            "clocks_mhz": self.clocks_mhz,
            "power_cap_w": self.power_cap_w,
            "allocatable": self.allocatable,
            "source": self.source,
        }


@dataclass
class NodeInventory:
    """Node/CPU/NUMA/RAM plus the devices attached to it (step 2)."""

    node_id: str
    cpu_arch: str = ""
    cpu_count: int = 0
    cpu_numa_nodes: int = 0
    memory_bytes: int = 0
    runtime_version: str = ""
    kubelet_version: str = ""
    topology_policy: str = "none"
    cpu_manager_policy: str = ""
    memory_manager_policy: str = ""
    labels: Mapping[str, str] = field(default_factory=dict)
    taints: Tuple[str, ...] = ()
    devices: Tuple[DeviceRecord, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("node_id", "cpu_arch", "runtime_version", "kubelet_version"):
            if not getattr(self, name):
                problems.append(f"node inventory requires {name!r}")
        if self.topology_policy not in TOPOLOGY_POLICIES:
            problems.append(f"node {self.node_id}: unknown topology policy {self.topology_policy!r}")
        if self.cpu_manager_policy == "":
            problems.append(f"node {self.node_id}: CPU Manager policy must be recorded (it is never a default)")
        for device in self.devices:
            problems.extend(device.validate())
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "cpu_arch": self.cpu_arch,
            "cpu_count": self.cpu_count,
            "cpu_numa_nodes": self.cpu_numa_nodes,
            "memory_bytes": self.memory_bytes,
            "runtime_version": self.runtime_version,
            "kubelet_version": self.kubelet_version,
            "topology_policy": self.topology_policy,
            "cpu_manager_policy": self.cpu_manager_policy,
            "memory_manager_policy": self.memory_manager_policy,
            "labels": dict(sorted(self.labels.items())),
            "taints": list(self.taints),
            "devices": [device.as_dict() for device in self.devices],
        }


@dataclass
class CapabilityLabel:
    """Step 3/4: a node capability label with owner, source, TTL and protection."""

    node_id: str
    label: str
    value: str
    label_class: str
    source: str = ""
    owner: str = ""
    ttl_s: int = 0
    protected: bool = False
    verified_at: str = ""
    write_subjects: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.label_class not in LABEL_CLASSES:
            problems.append(f"unknown label class {self.label_class!r}")
        for name in ("node_id", "label", "value", "source", "owner"):
            if not getattr(self, name):
                problems.append(f"label {self.label!r} requires {name!r}")
        if self.ttl_s <= 0:
            problems.append(f"label {self.label!r} needs a TTL (a capability label may expire)")
        if self.label_class in PROTECTED_LABEL_CLASSES:
            if not self.protected:
                problems.append(
                    f"label {self.label!r} of class {self.label_class!r} must be protected: a workload "
                    "that can rewrite it defeats placement and isolation"
                )
            if self.write_subjects and any(subject != "kubelet" for subject in self.write_subjects):
                problems.append(
                    f"label {self.label!r}: only the kubelet/platform controller may write a protected label, "
                    f"got {list(self.write_subjects)}"
                )
        if not self.verified_at:
            problems.append(f"label {self.label!r} has no verification timestamp (provenance must be checkable)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "value": self.value,
            "source": self.source,
            "owner": self.owner,
            "ttl_s": self.ttl_s,
            "protected": self.protected,
        }


def validate_label_provenance(
    labels: Sequence[CapabilityLabel], *, now: str = "", forged: Sequence[str] = ()
) -> Dict[str, Any]:
    """Steps 4/26: protected labels must be trustworthy and current."""
    problems: List[str] = []
    for label in labels:
        problems.extend(label.validate())
        if label.verified_at and now and label.verified_at > now:
            problems.append(f"label {label.label!r} is verified in the future ({label.verified_at})")
    forged_hits = [name for name in forged if name in {label.label for label in labels}]
    if forged_hits:
        problems.append(
            "a workload-owned writer was able to modify protected labels: " + ", ".join(sorted(forged_hits))
        )
    return {
        "labels": len(labels),
        "protected": sorted(label.label for label in labels if label.protected),
        "forged": sorted(forged_hits),
        "ok": not problems,
        "problems": problems,
    }


# ── placement plan and evidence ──────────────────────────────────────────


@dataclass
class PlacementPlan:
    """§4 ``PlacementPlan``: hard capability first, preference second."""

    placement_plan_id: str
    workload_id: str
    required_vendor: str = ""
    required_product: str = ""
    required_arch: str = ""
    min_memory_bytes: int = 0
    required_precision: Tuple[str, ...] = ()
    device_count: int = 0
    partition_profile: str = "NONE"
    sharing_mode: str = "EXCLUSIVE"
    required_node_labels: Mapping[str, str] = field(default_factory=dict)
    required_taints_tolerated: Tuple[str, ...] = ()
    cpu_request: str = ""
    cpu_numa_policy: str = ""
    topology_policy_required: str = ""
    link_domain_required: str = ""
    p2p_required: bool = False
    rank_to_device: Mapping[str, str] = field(default_factory=dict)
    anti_affinity: Tuple[str, ...] = ()
    cache_locality_preference: str = ""
    isolation_policy: str = ""
    unschedulable_behavior: str = "PENDING_WITH_REASON"
    soft_preferences: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("placement_plan_id", "workload_id", "required_vendor", "required_arch"):
            if not getattr(self, name):
                problems.append(f"placement plan requires {name!r}")
        if self.min_memory_bytes <= 0:
            problems.append("minimum device memory is a hard capability and must be positive")
        if self.device_count <= 0:
            problems.append("device count must be positive")
        if self.partition_profile not in PARTITION_PROFILES:
            problems.append(f"unknown partition profile {self.partition_profile!r}")
        if self.sharing_mode not in SHARING_MODES:
            problems.append(f"unknown sharing mode {self.sharing_mode!r}")
        if self.required_arch and self.required_arch in self.soft_preferences:
            problems.append(
                "the target architecture appears as a soft preference: hard capability must not be "
                "expressed as affinity"
            )
        if self.link_domain_required and self.link_domain_required not in LINK_DOMAINS:
            problems.append(f"unknown link domain {self.link_domain_required!r}")
        if self.device_count > 1 and not self.rank_to_device:
            problems.append("multi-device placement must declare the rank→device mapping")
        if self.p2p_required and not self.link_domain_required:
            problems.append("a P2P requirement needs a link domain to verify against")
        if self.sharing_mode != "EXCLUSIVE" and not self.isolation_policy:
            problems.append(
                "a shared/partitioned plan must state its isolation policy (shared capacity may not be "
                "used as exclusive capacity)"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "placement_plan_id": self.placement_plan_id,
            "workload_id": self.workload_id,
            "required_vendor": self.required_vendor,
            "required_arch": self.required_arch,
            "min_memory_bytes": self.min_memory_bytes,
            "device_count": self.device_count,
            "sharing_mode": self.sharing_mode,
            "required_node_labels": dict(sorted(self.required_node_labels.items())),
            "topology_policy": self.topology_policy_required,
            "rank_to_device": dict(sorted(self.rank_to_device.items())),
        }


@dataclass
class PlacementEvidence:
    """§9 ``PlacementEvidence``: the whole chain, ending at the *used* device."""

    placement_run_id: str
    placement_plan_id: str
    pod_uid: str
    requested_resource_and_capability: str = ""
    scheduled_node: str = ""
    scheduler_profile: str = ""
    filter_score_reasons: Tuple[str, ...] = ()
    allocated_resource_name: str = ""
    physical_or_partition_device_ids: Tuple[str, ...] = ()
    runtime_visible_device_ids: Tuple[str, ...] = ()
    actual_execution_device_ids: Tuple[str, ...] = ()
    cpu_set: str = ""
    cpu_numa: int = -1
    memory_numa: int = -1
    device_numa: int = -1
    link_domain: str = ""
    p2p_verified: bool = False
    rank_mapping: Mapping[str, str] = field(default_factory=dict)
    plugin_version: str = ""
    topology_policy_version: str = ""
    health_status: str = ""
    sharing_status: str = ""
    isolation_status: str = ""
    performance_result_ids: Tuple[str, ...] = ()
    negative_case_id: str = ""
    verdict: str = ""
    reason: str = ""
    evidence_refs: Tuple[str, ...] = ()

    def validate(self, plan: Optional[PlacementPlan] = None) -> List[str]:
        problems: List[str] = []
        for name in ("placement_run_id", "placement_plan_id", "pod_uid", "scheduled_node"):
            if not getattr(self, name):
                problems.append(f"placement evidence requires {name!r}")
        if not self.actual_execution_device_ids:
            problems.append(
                "the actual execution device is unknown: seeing a device and *using* it are different claims"
            )
        if len(self.runtime_visible_device_ids) != len(set(self.runtime_visible_device_ids)):
            problems.append("duplicate runtime-visible device ids")
        if set(self.actual_execution_device_ids) - set(self.runtime_visible_device_ids):
            problems.append("an execution device was not visible to the container (impossible mapping)")
        if set(self.runtime_visible_device_ids) - set(self.physical_or_partition_device_ids):
            problems.append(
                "the container sees devices that were not allocated to it (visibility is not isolation)"
            )
        if plan is not None:
            problems.extend(plan.validate())
            if plan.device_count != len(self.physical_or_partition_device_ids):
                problems.append(
                    f"plan asks for {plan.device_count} device(s) but {len(self.physical_or_partition_device_ids)} "
                    "were allocated"
                )
            if plan.link_domain_required and self.link_domain and plan.link_domain_required != self.link_domain:
                problems.append(
                    f"link domain mismatch: plan={plan.link_domain_required} actual={self.link_domain}"
                )
            if plan.p2p_required and not self.p2p_verified:
                problems.append("the plan requires P2P but no P2P verification was recorded")
            if plan.sharing_mode == "EXCLUSIVE" and self.sharing_status == "SHARED_WITH_OTHER_POD":
                problems.append("an exclusive plan was placed on a device shared with another pod")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "placement_run_id": self.placement_run_id,
            "placement_plan_id": self.placement_plan_id,
            "pod_uid": self.pod_uid,
            "requested_resource_and_capability": self.requested_resource_and_capability,
            "scheduled_node": self.scheduled_node,
            "allocated_resource_name": self.allocated_resource_name,
            "physical_or_partition_device_ids": list(self.physical_or_partition_device_ids),
            "runtime_visible_device_ids": list(self.runtime_visible_device_ids),
            "actual_execution_device_ids": list(self.actual_execution_device_ids),
            "verdict": self.verdict,
        }


# ── filter/score/bind and negative placements ────────────────────────────


def validate_allocatable(nodes: Sequence[NodeInventory]) -> Dict[str, Any]:
    """Step 8: health/missing devices must not stay advertised as allocatable."""
    problems: List[str] = []
    suspicious: List[Dict[str, Any]] = []
    for node in nodes:
        problems.extend(node.validate())
        for device in node.devices:
            if device.allocatable and device.health not in ("healthy", "OK", "ok"):
                suspicious.append({"node_id": node.node_id, "device_id": device.device_id, "health": device.health})
    if suspicious:
        problems.append(
            "devices advertised as allocatable while unhealthy/unknown: "
            + ", ".join(f"{row['node_id']}/{row['device_id']}({row['health']})" for row in suspicious)
        )
    return {"nodes": len(nodes), "suspicious": suspicious, "ok": not problems, "problems": problems}


def filter_nodes(plan: PlacementPlan, nodes: Sequence[NodeInventory]) -> Dict[str, Any]:
    """Step 9/23/24: the hard-constraint filter, with a reason per rejected node."""
    candidates: List[Dict[str, Any]] = []
    for node in nodes:
        reasons: List[str] = []
        # Hard capability is checked on the *device inventory* first: a node label is
        # a fast path and may be absent or stale, while the inventory records the
        # product/vendor/arch/memory the runtime will actually see (E13-03 §3.2).
        usable = [
            device
            for device in node.devices
            if device.health in ("healthy", "OK", "ok")
            and device.memory_bytes >= plan.min_memory_bytes
            and (plan.partition_profile == "NONE" or device.partition_profile == plan.partition_profile)
            and (not plan.required_vendor or device.vendor == plan.required_vendor)
            and (not plan.required_arch or device.arch == plan.required_arch)
        ]
        if plan.required_vendor and not usable and node.labels.get("vendor") != plan.required_vendor:
            reasons.append(
                f"vendor {node.labels.get('vendor', '<unset>')} != {plan.required_vendor} "
                "and no matching device in the inventory"
            )
        if plan.required_arch and not usable and node.labels.get("accel.arch") != plan.required_arch:
            reasons.append(
                f"accelerator arch {node.labels.get('accel.arch', '<unset>')} != {plan.required_arch} "
                "and no matching device in the inventory"
            )
        for label, value in plan.required_node_labels.items():
            if node.labels.get(label) != value:
                reasons.append(f"label {label}={node.labels.get(label, '<unset>')} != {value}")
        for taint in node.taints:
            if taint not in plan.required_taints_tolerated:
                reasons.append(f"taint {taint} not tolerated")
        if len(usable) < plan.device_count:
            reasons.append(
                f"only {len(usable)} usable device(s) for {plan.device_count} request(s) at "
                f">= {plan.min_memory_bytes} bytes / profile {plan.partition_profile}"
            )
        if plan.link_domain_required and not any(
            device.link_domain == plan.link_domain_required for device in usable
        ):
            reasons.append(f"no device in link domain {plan.link_domain_required}")
        candidates.append(
            {
                "node_id": node.node_id,
                "feasible": not reasons,
                "reasons": reasons,
                "usable_devices": [device.device_id for device in usable],
            }
        )
    feasible = [row for row in candidates if row["feasible"]]
    return {
        "candidates": candidates,
        "feasible": [row["node_id"] for row in feasible],
        "unschedulable_reasons": {
            row["node_id"]: row["reasons"] for row in candidates if not row["feasible"]
        },
        "no_feasible_node": not feasible,
        "note": "an unschedulable negative case is a *correct* outcome, with a precise reason",
    }


def validate_rank_mapping(plan: PlacementPlan, evidence: PlacementEvidence) -> Dict[str, Any]:
    """Steps 15/29: rank→device order matters for collectives; mismatch must fail."""
    problems: List[str] = []
    for rank, device in plan.rank_to_device.items():
        observed = evidence.rank_mapping.get(rank)
        if observed is None:
            problems.append(f"rank {rank}: no observed device (a silent device reorder corrupts collectives)")
        elif observed != device:
            problems.append(f"rank {rank}: planned {device} but observed {observed}")
    extra = sorted(set(evidence.rank_mapping) - set(plan.rank_to_device))
    if extra:
        problems.append(f"observed ranks not in the plan: {extra}")
    return {"ok": not problems, "problems": problems}


def unauthorized_device_access(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Steps 30/35: any successful cross-namespace/tenant device access is a FAIL."""
    results: List[Dict[str, Any]] = []
    successes: List[Dict[str, Any]] = []
    for row in rows:
        entry = {
            "placement_run_id": row.get("placement_run_id", ""),
            "from_tenant": row.get("from_tenant", ""),
            "target": row.get("target", ""),
            "action": row.get("action", ""),
            "expected": row.get("expected", "deny"),
            "observed": row.get("observed", ""),
            "verdict": "PASS" if row.get("observed") == row.get("expected", "deny") else "FAIL",
        }
        results.append(entry)
        if entry["verdict"] == "FAIL":
            successes.append(entry)
    return {
        "rows": results,
        "attempts": len(results),
        "violations": successes,
        "unauthorized_success_rate": (len(successes) / len(results)) if results else 0.0,
        "ok": not successes,
        "reason": (
            ""
            if not successes
            else "unauthorized access succeeded: "
            + ", ".join(f"{row['from_tenant']}->{row['target']}" for row in successes)
        ),
    }


def validate_resource_release(
    *, before: Mapping[str, Any], after_release: Mapping[str, Any], expected_release_s: float
) -> Dict[str, Any]:
    """Step 31: allocation/context/memory must return to the baseline after termination."""
    problems: List[str] = []
    for key, value in before.items():
        observed = after_release.get(key)
        if observed is None:
            problems.append(f"{key}: no post-termination observation")
            continue
        if isinstance(value, (int, float)) and isinstance(observed, (int, float)) and observed > value:
            problems.append(f"{key}: {observed} > pre-run {value} (residual allocation)")
    return {
        "before": dict(sorted(before.items())),
        "after": dict(sorted(after_release.items())),
        "release_within_s": expected_release_s,
        "ok": not problems,
        "problems": problems,
    }


def locality_ab(
    rows: Sequence[Mapping[str, Any]], *, metric: str, repetitions: int = 3
) -> Dict[str, Any]:
    """Steps 13/34: aligned vs remote locality must be compared as an effect.

    Only one factor may change (locality); a different device model changes two.
    """
    if metric not in LOCALITY_METRICS:
        raise ConfigError(f"unknown locality metric {metric!r}")
    seen: Dict[str, List[float]] = {"aligned": [], "remote": []}
    device_models: Dict[str, set] = {"aligned": set(), "remote": set()}
    for row in rows:
        arm = str(row.get("arm", ""))
        if arm not in seen:
            continue
        seen[arm].append(float(row.get("value", 0.0)))
        device_models[arm].add(str(row.get("device_model", "")))
    problems: List[str] = []
    if len(seen["aligned"]) < repetitions or len(seen["remote"]) < repetitions:
        problems.append(
            f"locality A/B needs >= {repetitions} samples per arm (got "
            f"{len(seen['aligned'])}/{len(seen['remote'])})"
        )
    if device_models["aligned"] != device_models["remote"]:
        problems.append(
            "the two arms used different device models: locality and hardware would be confounded"
        )
    return {
        "metric": metric,
        "samples": {arm: len(values) for arm, values in seen.items()},
        "medians": {arm: (sorted(values)[len(values) // 2] if values else None) for arm, values in seen.items()},
        "device_models": {arm: sorted(models) for arm, models in device_models.items()},
        "ok": not problems,
        "problems": problems,
    }


def run_negative_placements(
    cases: Sequence[Mapping[str, Any]], *, plan: PlacementPlan
) -> Dict[str, Any]:
    """Steps 23–29: every negative case must be refused *before* the model loads."""
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        kind = str(case.get("kind", ""))
        if kind not in NEGATIVE_CASES:
            raise ConfigError(f"unknown negative placement case {kind!r}")
        observed = str(case.get("observed", "NOT_RUN"))
        rows.append(
            {
                "case_id": str(case.get("case_id", f"place-neg-{index:03d}")),
                "kind": kind,
                "expected": str(case.get("expected", "UNSCHEDULABLE_OR_REJECTED")),
                "observed": observed,
                "reason": str(case.get("reason", "")),
                "model_loaded": bool(case.get("model_loaded", False)),
                "ok": observed in ("UNSCHEDULABLE_OR_REJECTED", "REJECTED", "FAILED_CLOSED")
                and not case.get("model_loaded", False),
            }
        )
    failures = [row for row in rows if not row["ok"]]
    return {
        "rows": rows,
        "cases": len(rows),
        "failures": failures,
        "ok": not failures,
        "placement_plan_id": plan.placement_plan_id,
        "reason": (
            ""
            if not failures
            else "a wrong/no-capability placement ran anyway: " + ", ".join(row["kind"] for row in failures)
        ),
    }


def isolation_verdict(
    *,
    plan: PlacementPlan,
    exclusive_rows: Sequence[Mapping[str, Any]],
    shared_rows: Sequence[Mapping[str, Any]],
    unauthorized: Mapping[str, Any],
    telemetry_visibility: Mapping[str, Any],
) -> Dict[str, Any]:
    """Steps 20–22/35: isolation claims are only made for the layers actually tested."""
    problems: List[str] = []
    verified: List[str] = []
    if plan.sharing_mode == "EXCLUSIVE":
        second_pod = [row for row in exclusive_rows if row.get("second_pod_admitted")]
        if second_pod:
            problems.append("an exclusive device admitted a second pod requesting the same resource")
        else:
            verified.append("scheduler_accounting")
    else:
        if not shared_rows:
            problems.append(
                f"sharing mode {plan.sharing_mode} declared but no concurrent sharing measurement exists"
            )
        else:
            verified.append("shared_fairness")
    if unauthorized.get("ok"):
        verified.append("device_visibility")
    else:
        problems.append("cross-tenant device access succeeded")
    if telemetry_visibility.get("ok"):
        verified.append("telemetry_information")
    else:
        problems.append("telemetry exposes another tenant's device identity")
    if not plan.isolation_policy:
        problems.append("no isolation policy declared on the placement plan")
    return {
        "sharing_mode": plan.sharing_mode,
        "verified_layers": sorted(set(verified)),
        "unverified_layers": sorted(set(ISOLATION_LAYERS) - set(verified)),
        "ok": not problems,
        "problems": problems,
        "note": "only the verified layers may appear in a production claim",
    }


# ── protocol steps and smoke self-check ──────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 placement questions", ("scheduling:PlacementPlan", "scheduling:NEGATIVE_CASES")),
    (2, "冻结 cluster/device inventory", ("scheduling:NodeInventory", "scheduling:DeviceRecord")),
    (3, "定义 capability label schema", ("scheduling:CapabilityLabel", "scheduling:LABEL_CLASSES")),
    (4, "保护可信 labels", ("scheduling:validate_label_provenance", "scheduling:PROTECTED_LABEL_CLASSES")),
    (5, "冻结 device plugin/DRA 配置", ("scheduling:DevicePluginConfig",)),
    (6, "冻结 kubelet topology/CPU/memory policy", ("scheduling:NodeInventory", "scheduling:TOPOLOGY_POLICIES")),
    (7, "建立 PlacementPlan", ("scheduling:PlacementPlan",)),
    (8, "验证 allocatable inventory", ("scheduling:validate_allocatable",)),
    (9, "运行单设备合法 placement", ("scheduling:filter_nodes", "scheduling:SchedulerEvent")),
    (10, "验证容器内 device 映射", ("scheduling:PlacementEvidence",)),
    (11, "验证 actual backend 使用", ("scheduling:PlacementEvidence.validate",)),
    (12, "验证 CPU/cpuset/NUMA", ("scheduling:TopologyRecord",)),
    (13, "运行 locality A/B", ("scheduling:locality_ab",)),
    (14, "验证 Topology Manager 行为", ("scheduling:TopologyPolicyResult",)),
    (15, "运行多设备合法 placement", ("scheduling:validate_rank_mapping",)),
    (16, "验证 P2P/link topology", ("scheduling:P2PResult",)),
    (17, "运行 topology A/B", ("scheduling:locality_ab", "scheduling:LOCALITY_METRICS")),
    (18, "验证 anti-affinity/spread", ("scheduling:anti_affinity_check",)),
    (19, "验证 partition profile", ("scheduling:PARTITION_PROFILES", "scheduling:DeviceRecord")),
    (20, "验证 exclusive 模式", ("scheduling:isolation_verdict",)),
    (21, "验证 shared/time-slicing 模式", ("scheduling:isolation_verdict", "scheduling:SHARING_MODES")),
    (22, "测邻居干扰", ("scheduling:neighbor_interference",)),
    (23, "测试 wrong vendor/arch", ("scheduling:run_negative_placements",)),
    (24, "测试 memory/partition不足", ("scheduling:run_negative_placements",)),
    (25, "测试 taint/toleration错误", ("scheduling:run_negative_placements",)),
    (26, "测试缺失/伪造 label", ("scheduling:run_negative_placements", "scheduling:validate_label_provenance")),
    (27, "测试 unhealthy device", ("scheduling:run_negative_placements", "scheduling:validate_allocatable")),
    (28, "测试 device plugin restart", ("scheduling:plugin_restart_recovery",)),
    (29, "测试 rank/device mismatch", ("scheduling:validate_rank_mapping",)),
    (30, "测试跨 namespace 越权", ("scheduling:unauthorized_device_access",)),
    (31, "验证资源释放", ("scheduling:validate_resource_release",)),
    (32, "验证 placement observability", ("scheduling:PlacementEvidence", "observability:SemanticConvention")),
    (33, "计算 scheduling 指标", ("scheduling:scheduling_metrics",)),
    (34, "计算 locality/topology效应", ("scheduling:locality_ab",)),
    (35, "计算隔离/共享指标", ("scheduling:isolation_verdict", "scheduling:unauthorized_device_access")),
    (36, "执行 policy regression", ("scheduling:policy_regression",)),
    (37, "独立复验", ("scheduling:validate_resource_release",)),
    (38, "形成 placement/isolation verdict", ("scheduling:placement_verdict",)),
)


@dataclass
class DevicePluginConfig:
    """Step 5: plugin/DRA identity, resource names and sharing behaviour."""

    plugin_name: str
    image_digest: str = ""
    resource_names: Tuple[str, ...] = ()
    allocation_behavior: str = ""
    sharing_behavior: str = ""
    health_behavior: str = ""
    version: str = ""
    feature_gates: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("plugin_name", "version", "allocation_behavior", "sharing_behavior", "health_behavior"):
            if not getattr(self, name):
                problems.append(f"device plugin config requires {name!r}")
        if not self.resource_names:
            problems.append("resource names must be recorded explicitly")
        if self.image_digest and not self.image_digest.startswith("sha256:"):
            problems.append("plugin image must be digest-pinned")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "image_digest": self.image_digest,
            "resource_names": list(self.resource_names),
            "allocation_behavior": self.allocation_behavior,
            "sharing_behavior": self.sharing_behavior,
            "health_behavior": self.health_behavior,
            "version": self.version,
            "feature_gates": list(self.feature_gates),
        }


@dataclass
class SchedulerEvent:
    """Step 9: one scheduling phase transition with its reason."""

    placement_run_id: str
    pod_uid: str
    phase: str
    timestamp: str = ""
    node_id: str = ""
    reason: str = ""
    filter_reasons: Tuple[str, ...] = ()
    score: float = 0.0

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.phase not in SCHEDULER_PHASES:
            problems.append(f"unknown scheduler phase {self.phase!r}")
        if self.phase == "UNSCHEDULABLE" and not self.reason:
            problems.append("an unschedulable phase must carry a precise reason")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "placement_run_id": self.placement_run_id,
            "pod_uid": self.pod_uid,
            "phase": self.phase,
            "reason": self.reason,
            "node_id": self.node_id,
            "timestamp": self.timestamp,
        }


@dataclass
class TopologyRecord:
    """Step 12: CPU/cpuset/NUMA/memory locality of one pod."""

    placement_run_id: str
    pod_uid: str
    cpuset: str = ""
    cpu_numa: int = -1
    memory_numa: int = -1
    device_numa: int = -1
    link_domain: str = ""
    irq_affinity: str = ""
    nic_numa: int = -1

    def validate(self, *, strict_locality: bool = False) -> List[str]:
        problems: List[str] = []
        if not self.cpuset:
            problems.append("cpuset must be recorded (CPU locality is part of placement)")
        if self.cpu_numa < 0 or self.device_numa < 0:
            problems.append("CPU and device NUMA nodes must both be known")
        elif strict_locality and self.cpu_numa != self.device_numa:
            problems.append(
                f"strict locality required but CPU NUMA {self.cpu_numa} != device NUMA {self.device_numa}"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "placement_run_id": self.placement_run_id,
            "pod_uid": self.pod_uid,
            "cpuset": self.cpuset,
            "cpu_numa": self.cpu_numa,
            "memory_numa": self.memory_numa,
            "device_numa": self.device_numa,
            "link_domain": self.link_domain,
        }


@dataclass
class TopologyPolicyResult:
    """Step 14: the Topology Manager policy that was actually in effect."""

    node_id: str
    policy: str = ""
    scope: str = ""
    admitted: bool = False
    hints: Tuple[str, ...] = ()
    reason: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.policy not in TOPOLOGY_POLICIES:
            problems.append(f"unknown topology policy {self.policy!r}")
        if not self.scope:
            problems.append("topology scope must be recorded (container/pod)")
        if not self.admitted and not self.reason:
            problems.append("a rejected topology admission must record its reason")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "policy": self.policy,
            "scope": self.scope,
            "admitted": self.admitted,
            "hints": list(self.hints),
            "reason": self.reason,
        }


@dataclass
class P2PResult:
    """Step 16: peer access / collective microbenchmark per link domain."""

    placement_run_id: str
    device_pairs: Tuple[Tuple[str, str], ...] = ()
    link_domain: str = ""
    peer_access_ok: bool = False
    collective_kind: str = ""
    collective_time_us: float = 0.0
    link_counters: Mapping[str, float] = field(default_factory=dict)

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.device_pairs:
            problems.append("P2P verification needs the device pairs it measured")
        if self.link_domain not in LINK_DOMAINS or self.link_domain == "UNKNOWN":
            problems.append("P2P verification must name a real link domain")
        if self.peer_access_ok and not self.link_counters:
            problems.append("a positive peer-access claim needs link counters, not only an API return")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "placement_run_id": self.placement_run_id,
            "device_pairs": [list(pair) for pair in self.device_pairs],
            "link_domain": self.link_domain,
            "peer_access_ok": self.peer_access_ok,
            "collective_kind": self.collective_kind,
            "collective_time_us": self.collective_time_us,
            "link_counters": dict(sorted(self.link_counters.items())),
        }


def anti_affinity_check(
    *, replicas: Sequence[Mapping[str, Any]], hard_spread: bool, required_domains: int = 1
) -> Dict[str, Any]:
    """Step 18: hard spread rules must hold; soft rules are reported as preference."""
    placement = [str(row.get("node_id", "")) for row in replicas]
    nodes = sorted(set(placement))
    problems: List[str] = []
    if hard_spread and len(nodes) < min(required_domains, len(replicas)):
        problems.append(
            f"hard spread requires {required_domains} distinct node(s), got {len(nodes)} for {len(replicas)} replica(s)"
        )
    return {
        "replicas": len(replicas),
        "distinct_nodes": len(nodes),
        "hard_spread": hard_spread,
        "ok": not problems,
        "problems": problems,
    }


def neighbor_interference(
    *, baseline: float, with_neighbor: float, metric: str, direction: str = "higher_is_worse"
) -> Dict[str, Any]:
    """Step 22: neighbour impact is quantified, never averaged away globally."""
    if metric not in LOCALITY_METRICS:
        raise ConfigError(f"unknown interference metric {metric!r}")
    ratio = (with_neighbor / baseline) if baseline else float("inf")
    worse = ratio > 1.0 if direction == "higher_is_worse" else ratio < 1.0
    return {
        "metric": metric,
        "baseline": baseline,
        "with_neighbor": with_neighbor,
        "interference_ratio": ratio,
        "degraded": worse,
        "note": "report the victim's own metric; a global average hides a noisy neighbour",
    }


def plugin_restart_recovery(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 28: a plugin restart must not lose existing allocation identity."""
    problems: List[str] = []
    for row in rows:
        if not row.get("allocation_identity_preserved", False):
            problems.append(
                f"device {row.get('device_id', '?')}: allocation identity lost across a plugin restart"
            )
        if row.get("new_scheduling_blocked") and not row.get("reason"):
            problems.append("new scheduling was blocked without a recorded reason")
    return {"rows": list(rows), "ok": not problems, "problems": problems}


def scheduling_metrics(events: Sequence[SchedulerEvent]) -> Dict[str, Any]:
    """Step 33: pending/schedule/allocation timing and the unschedulable rate."""
    problems: List[str] = []
    for event in events:
        problems.extend(event.validate())
    total = len(events)
    unschedulable = [event for event in events if event.phase == "UNSCHEDULABLE"]
    return {
        "events": total,
        "unschedulable": len(unschedulable),
        "unschedulable_rate": (len(unschedulable) / total) if total else 0.0,
        "unschedulable_reasons": sorted({event.reason for event in unschedulable if event.reason}),
        "problems": problems,
    }


def policy_regression(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 36: synthetic cluster objects keep the hard/soft rules from regressing."""
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        filter_result = filter_nodes(case["plan"], case["nodes"])
        observed = "FEASIBLE" if filter_result["feasible"] else "UNSCHEDULABLE"
        rows.append(
            {
                "case_id": str(case.get("case_id", f"reg-{index:03d}")),
                "expected": str(case.get("expected", "")),
                "observed": observed,
                "ok": str(case.get("expected", "")) == observed,
                "reasons": filter_result["unschedulable_reasons"],
            }
        )
    failures = [row for row in rows if not row["ok"]]
    return {"rows": rows, "cases": len(rows), "failures": failures, "ok": not failures}


def placement_verdict(
    *,
    plan: PlacementPlan,
    evidence: PlacementEvidence,
    allocatable: Mapping[str, Any],
    negatives: Mapping[str, Any],
    isolation: Mapping[str, Any],
    release: Mapping[str, Any],
) -> Dict[str, Any]:
    """Step 38: which pools/placement/isolation level the stage may claim."""
    problems: List[str] = []
    problems.extend(plan.validate())
    problems.extend(evidence.validate(plan))
    if not allocatable.get("ok"):
        problems.append("device inventory is not trustworthy (unhealthy devices were advertised)")
    if not negatives.get("ok"):
        problems.append("negative placements were not refused before model load")
    if not isolation.get("ok"):
        problems.append("isolation claims are not fully verified")
    if not release.get("ok"):
        problems.append("device resources were not released after termination")
    return {
        "placement_plan_id": plan.placement_plan_id,
        "sharing_mode": plan.sharing_mode,
        "verified_isolation_layers": list(isolation.get("verified_layers", [])),
        "problems": problems,
        "verdict": "PASSABLE_AT_CODE_LEVEL" if not problems else "BLOCKED",
        "note": (
            "a multi-device claim additionally requires rank/link/P2P evidence "
            "and may not inherit a single-device conclusion"
        ),
    }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the placement contracts (smoke, not an experiment)."""
    checks: Dict[str, Any] = {}
    device = DeviceRecord(
        device_id="GPU-0", vendor="NVIDIA", product="RTX3090", arch="sm_86", memory_bytes=24 << 30,
        driver_version="550.54", partition_profile="NONE", health="healthy", numa_node=0,
        link_domain="PCIe", source="device_plugin",
    )
    node = NodeInventory(
        node_id="node-1", cpu_arch="x86_64/sm_86", runtime_version="containerd-1.7",
        kubelet_version="v1.30", topology_policy="single-numa-node", cpu_manager_policy="static",
        labels={"vendor": "NVIDIA", "accel.arch": "sm_86", "accel.memory.gb": "24"},
        devices=(device,),
    )
    checks["node_inventory_valid"] = node.validate() == []

    unchecked = DeviceRecord(device_id="GPU-1", vendor="NVIDIA", product="RTX3090", arch="sm_86",
                             memory_bytes=24 << 30, driver_version="550.54", health="unknown", source="plugin")
    checks["unknown_health_flagged"] = any("health" in problem for problem in unchecked.validate())

    forged = CapabilityLabel(
        node_id="node-1", label="accel.arch", value="sm_90", label_class="arch", source="node",
        owner="sre", ttl_s=3600, protected=False, verified_at="2026-09-19T00:00:00Z",
    )
    checks["protected_label_enforced"] = any("protected" in problem for problem in forged.validate())

    plan = PlacementPlan(
        placement_plan_id="p1", workload_id="w1", required_vendor="NVIDIA", required_arch="sm_86",
        min_memory_bytes=16 << 30, device_count=1, partition_profile="NONE", sharing_mode="EXCLUSIVE",
        topology_policy_required="single-numa-node", rank_to_device={"0": "GPU-0"},
    )
    checks["plan_valid"] = plan.validate() == []

    filtered = filter_nodes(plan, [node])
    checks["feasible_node_found"] = filtered["feasible"] == ["node-1"]

    wrong = PlacementPlan(
        placement_plan_id="p2", workload_id="w1", required_vendor="NVIDIA", required_arch="sm_90",
        min_memory_bytes=80 << 30, device_count=2, rank_to_device={"0": "a", "1": "b"},
    )
    checks["wrong_capability_unschedulable"] = filter_nodes(wrong, [node])["no_feasible_node"] is True

    evidence = PlacementEvidence(
        placement_run_id="run-1", placement_plan_id="p1", pod_uid="pod-1", scheduled_node="node-1",
        allocated_resource_name="nvidia.com/gpu", physical_or_partition_device_ids=("GPU-0",),
        runtime_visible_device_ids=("GPU-0",), actual_execution_device_ids=("GPU-0",),
        rank_mapping={"0": "GPU-0"}, p2p_verified=False,
    )
    checks["evidence_chain_ok"] = evidence.validate(plan) == []

    leaky = PlacementEvidence(
        placement_run_id="run-1", placement_plan_id="p1", pod_uid="pod-1", scheduled_node="node-1",
        physical_or_partition_device_ids=("GPU-0",), runtime_visible_device_ids=("GPU-0", "GPU-1"),
        actual_execution_device_ids=("GPU-0",),
    )
    checks["visibility_leak_detected"] = any(
        "not allocated" in problem for problem in leaky.validate(plan)
    )

    unauthorized = unauthorized_device_access(
        [{"from_tenant": "tenant-a", "target": "GPU-1", "action": "open", "expected": "deny", "observed": "deny"}]
    )
    checks["unauthorized_denied_ok"] = unauthorized["ok"] is True
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "checks": checks,
        "note": "接口自检；未连接任何集群，未提交任何 pod",
    }
