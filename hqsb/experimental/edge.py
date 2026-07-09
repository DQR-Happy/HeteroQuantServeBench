"""E14-08 — Android/ARM/edge runtime adapter and the adoption decision.

Optional P2: ``N/A_BY_ADR`` when not selected.  Two routes are equally legitimate
(§2):

* **route A — a real adapter**: a target device exists, so a minimal C4 adapter is
  implemented and its conversion, dispatch, quality, performance, resources and
  failures are measured on the device;
* **route B — an evidence-based technology map**: no target device exists, so a
  versioned capability/operator/format/risk/value analysis is produced together
  with an explicit ``DO_NOT_ADOPT`` / ``BLOCKED_DEVICE`` ADR.  Route B may
  complete the *decision research* but must not produce latency/power/adapter
  claims.

Two measurement rules shape the interfaces:

* **delegate partition decides the real coverage** (§4.1): "NPU is enabled" does
  not prove the graph runs on the NPU, so the per-op backend, the fallback ops and
  the boundary copies are recorded;
* **edge timing has three regimes** (§4.2): cold first-run, warm short-run and
  sustained thermal steady-state.  A short boost run extrapolated to sustained
  performance is exactly what :func:`thermal_steady_state` refuses.

Interfaces provided:

* :class:`RouteChoice`, :class:`PlatformRuntimeChoice` (steps 1–2),
  :class:`DeviceFingerprint` (step 4);
* :class:`RuntimeCapabilityMap` / :func:`op_coverage_map` (steps 9–10);
* :func:`estimate_memory_and_package` (step 11), :func:`realtime_feasibility`
  (step 12) — estimates that must never be written as measurements;
* :func:`go_no_go` (step 13), :class:`ConversionDag` (steps 14–16);
* :class:`EdgeAdapterContract` — capability/load/run/unload/health and error
  mapping, with lazy import (step 17);
* :func:`partition_placement` (step 21), :func:`device_correctness` (step 22);
* :func:`cold_warm_sustained` (steps 24–26), :func:`thread_affinity_scan`
  (step 27), :func:`shape_scan` (step 28), :func:`memory_lifecycle` (step 29);
* :func:`power_energy` (step 30) — calibrated source or ``unavailable``;
* :func:`package_and_startup` (step 31), :func:`background_disturbance`
  (step 32);
* :func:`unsupported_op`, :func:`corrupted_model_check`, :func:`low_memory_cancel`
  (steps 33–35);
* :func:`technology_map` (step 37), :class:`EdgeAdoptionDecision` (step 40).

Nothing here contacts a device or builds a binary; the arithmetic is over supplied
partitions, latencies and packet sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.contracts import AdoptionDecision
from hqsb.experimental.identity import SCHEMA_PREFIX, canonical_digest, is_digest

EXPERIMENT_ID = "E14-08"
TITLE = "Android/ARM/端侧 Runtime Adapter 与采用决策"
LEVEL = "P2"

CLAIM_BOUNDARY = (
    "路线 A 只在真实设备有实测时成立；路线 B 只产生采用/不采用 ADR，**不得**产生任何设备性能或 "
    "adapter 支持 claim。优秀结果可以是证据充分的“不做”（E14-08 §12/§13）。"
)

#: The candidate platform–runtime combinations; exactly one primary (§7 step 2).
PLATFORM_RUNTIMES: Tuple[str, ...] = (
    "android_tflite",
    "android_nnapi",
    "android_mnn",
    "android_ncnn",
    "android_qnn",
    "linux_arm_onnxruntime",
)

#: Op placement outcomes (step 10/21).
OP_PLACEMENT: Tuple[str, ...] = ("supported", "partitioned", "fallback", "unsupported")

#: The three timing regimes that may never be mixed (§4.2).
TIMING_MODES: Tuple[str, ...] = rec.EDGE_MODES

#: Evidence levels; route B may only reach ``MAP_ONLY`` (§7 step 40).
EVIDENCE_LEVELS: Tuple[str, ...] = rec.EDGE_EVIDENCE_LEVELS

#: Technology-map axes that must all carry a verdict (step 37).
MAP_AXES: Tuple[str, ...] = (
    "capability",
    "operator_gap",
    "conversion",
    "device",
    "licence",
    "maintenance",
    "job_value",
)

#: Package contributions that decide whether a build is shippable (step 31).
PACKAGE_COMPONENTS: Tuple[str, ...] = (
    "model_bytes",
    "runtime_library_bytes",
    "native_library_bytes",
    "assets_bytes",
    "app_code_bytes",
)

#: Header fields a real device must supply before any measurement (step 4).
DEVICE_FINGERPRINT_FIELDS: Tuple[str, ...] = (
    "device_id",
    "soc",
    "cpu",
    "gpu",
    "npu",
    "ram_bytes",
    "os_version",
    "firmware",
    "runtime_version",
    "thermal_design",
)


@dataclass(frozen=True)
class RouteChoice:
    """Step 1: A or B, decided by whether a real device, permission and tooling exist."""

    route: str
    has_device: bool
    has_measurement_tooling: bool
    has_permission: bool
    rationale: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.route not in rec.EDGE_ROUTES:
            findings.append(f"RouteChoice: route {self.route!r} must be one of {', '.join(rec.EDGE_ROUTES)}")
        needs_device = self.route == rec.EDGE_ROUTES[0]
        if needs_device and not (self.has_device and self.has_measurement_tooling and self.has_permission):
            findings.append(
                "RouteChoice: route A requires a real device, measurement tooling and permission; "
                "without them the honest route is B (无设备时用模拟器数字填真实性能 = FAIL)"
            )
        if not needs_device and self.has_device and self.rationale and "no device" in self.rationale.lower():
            findings.append("RouteChoice: route B is justified by 'no device' although a device is available")
        if not self.rationale:
            findings.append("RouteChoice: a rationale is required")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "has_device": self.has_device,
            "has_measurement_tooling": self.has_measurement_tooling,
            "has_permission": self.has_permission,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class PlatformRuntimeChoice:
    """Step 2: one combination, with the alternatives named."""

    platform: str
    runtime: str
    combination_id: str
    rejected_alternatives: Tuple[str, ...] = ()
    rationale: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.combination_id not in PLATFORM_RUNTIMES:
            findings.append(
                f"PlatformRuntimeChoice: {self.combination_id!r} must be one of "
                f"{', '.join(PLATFORM_RUNTIMES)}"
            )
        for name in ("platform", "runtime"):
            if not getattr(self, name):
                findings.append(f"PlatformRuntimeChoice: {name} is required")
        if len(self.rejected_alternatives) < 2:
            findings.append(
                "PlatformRuntimeChoice: at least two alternatives must be rejected explicitly "
                "(列出六个框架却没有一条深入 = FAIL)"
            )
        if not self.rationale:
            findings.append("PlatformRuntimeChoice: a rationale is required")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "runtime": self.runtime,
            "combination_id": self.combination_id,
            "rejected_alternatives": list(self.rejected_alternatives),
            "rationale": self.rationale,
        }


@dataclass
class DeviceFingerprint:
    """Step 4: everything needed to reproduce a device measurement."""

    fields: Mapping[str, Any] = field(default_factory=dict)
    evidence_level: str = "MAP_ONLY"

    def problems(self) -> List[str]:
        findings: List[str] = []
        missing = [name for name in DEVICE_FINGERPRINT_FIELDS if not self.fields.get(name)]
        if missing:
            findings.append(f"DeviceFingerprint: missing {', '.join(missing)} (只写“ARM 手机”不可复现)")
        if self.evidence_level not in EVIDENCE_LEVELS:
            findings.append(
                f"DeviceFingerprint: evidence_level {self.evidence_level!r} must be one of "
                f"{', '.join(EVIDENCE_LEVELS)}"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fields": {key: self.fields[key] for key in sorted(self.fields)},
            "evidence_level": self.evidence_level,
        }


@dataclass
class RuntimeCapabilityMap:
    """Step 9: formats, dtypes, dynamic shape, custom ops, delegates, profiler."""

    runtime_id: str
    formats: Tuple[str, ...] = ()
    dtypes: Tuple[str, ...] = ()
    dynamic_shape: bool = False
    custom_ops: Tuple[str, ...] = ()
    delegates: Tuple[str, ...] = ()
    profiler_available: bool = False
    source_version: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("runtime_id", "source_version"):
            if not getattr(self, name):
                findings.append(f"RuntimeCapabilityMap: {name} is required (当前版本，而非营销支持列表)")
        if not self.formats:
            findings.append("RuntimeCapabilityMap: model formats must be listed")
        if not self.dtypes:
            findings.append("RuntimeCapabilityMap: dtypes must be listed")
        if not self.delegates:
            findings.append("RuntimeCapabilityMap: delegates/backends must be listed")
        if not self.profiler_available:
            findings.append(
                "RuntimeCapabilityMap: without a profiler the actual partition cannot be proven (step 20)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "runtime_id": self.runtime_id,
            "formats": list(self.formats),
            "dtypes": list(self.dtypes),
            "dynamic_shape": self.dynamic_shape,
            "custom_ops": list(self.custom_ops),
            "delegates": list(self.delegates),
            "profiler_available": self.profiler_available,
            "source_version": self.source_version,
        }


def op_coverage_map(
    *, ops: Sequence[Mapping[str, Any]], runtime_id: str, source_version: str
) -> Dict[str, Any]:
    """Step 10: model operator × shape against runtime/backend support.

    ``E14-08`` §10: 只判断模型文件能否打开 is not a coverage answer; each op must be
    labelled supported/partitioned/fallback/unsupported with its shape.
    """
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []
    counts = {level: 0 for level in OP_PLACEMENT}
    for op in ops:
        placement = str(op.get("placement", ""))
        if placement not in OP_PLACEMENT:
            problems.append(f"op {op.get('op')!r} has unknown placement {placement!r}")
            continue
        for key in ("op", "shape", "dtype"):
            if key not in op:
                problems.append(f"op {op.get('op')!r} is missing {key!r}")
                break
        else:
            counts[placement] += 1
            rows.append(dict(op))
    total = sum(counts.values())
    covered = counts["supported"] + counts["partitioned"]
    return {
        "runtime_id": runtime_id,
        "source_version": source_version,
        "rows": rows,
        "counts": counts,
        "coverage": covered / total if total else 0.0,
        "fallback_count": counts["fallback"],
        "unsupported_count": counts["unsupported"],
        "problems": problems,
        "note": "supported/partitioned/fallback/unsupported 必须逐 op 标注（step 10）",
    }


def estimate_memory_and_package(
    *, weights_bytes: int, kv_bytes: int, activation_bytes: int, workspace_bytes: int,
    runtime_library_bytes: int, app_assets_bytes: int,
) -> Dict[str, Any]:
    """Step 11: an **estimate**, marked as such, that a later measurement replaces.

    ``E14-08`` §10: 只看模型文件大小 misses KV/activation/workspace and the runtime
    library; returning all of them together keeps the peak honest.
    """
    for name, value in (
        ("weights_bytes", weights_bytes), ("kv_bytes", kv_bytes), ("activation_bytes", activation_bytes),
        ("workspace_bytes", workspace_bytes), ("runtime_library_bytes", runtime_library_bytes),
        ("app_assets_bytes", app_assets_bytes),
    ):
        if value < 0:
            raise ConfigError(f"{name} must be >= 0")
    runtime_peak = weights_bytes + kv_bytes + activation_bytes + workspace_bytes
    package = weights_bytes + runtime_library_bytes + app_assets_bytes
    return {
        "evidence_level": "MAP_ONLY",
        "runtime_peak_bytes": runtime_peak,
        "package_bytes": package,
        "components": {
            "weights_bytes": weights_bytes,
            "kv_bytes": kv_bytes,
            "activation_bytes": activation_bytes,
            "workspace_bytes": workspace_bytes,
            "runtime_library_bytes": runtime_library_bytes,
            "app_assets_bytes": app_assets_bytes,
        },
        "note": "这是估算，不是实测；在采用决策中必须标 MAP_ONLY（step 11/12）",
    }


def realtime_feasibility(
    *, deadline_ms: float, estimated_latency_ms: float, estimated_energy_j: float, thermal_budget_j: float
) -> Dict[str, Any]:
    """Step 12: a prior from arithmetic, not a measurement (§7 step 12).

    ``E14-08`` §10: 进入明显不可行的无界开发 is the cost this avoids, so an infeasible
    estimate blocks development *before* the adapter is written.
    """
    problems: List[str] = []
    if deadline_ms <= 0:
        problems.append("deadline_ms must be positive")
    margin = (deadline_ms - estimated_latency_ms) / deadline_ms if deadline_ms else 0.0
    over_energy = estimated_energy_j > thermal_budget_j
    if margin < 0:
        problems.append(
            f"estimated latency {estimated_latency_ms} ms exceeds the {deadline_ms} ms deadline by "
            f"{abs(margin) * 100:.1f}%"
        )
    if over_energy:
        problems.append(
            f"estimated energy {estimated_energy_j} J exceeds the thermal budget {thermal_budget_j} J"
        )
    return {
        "evidence_level": "MAP_ONLY",
        "deadline_ms": deadline_ms,
        "estimated_latency_ms": estimated_latency_ms,
        "margin_ratio": margin,
        "estimated_energy_j": estimated_energy_j,
        "thermal_budget_j": thermal_budget_j,
        "feasible": not problems,
        "problems": problems,
    }


def go_no_go(
    *, route_choice: RouteChoice, feasibility: Mapping[str, Any], memory_estimate: Mapping[str, Any],
    device_ram_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Step 13: the hard gate that turns an infeasible candidate into route B.

    ``E14-08`` §10: sunk-cost 驱动继续 is the failure; an unmet hard gate switches to
    the technology map and a rejection ADR instead of "one more try".
    """
    problems: List[str] = []
    if not feasibility.get("feasible"):
        problems.append("the realtime/thermal gate is not met")
    peak = int(memory_estimate.get("runtime_peak_bytes", 0))
    if device_ram_bytes is not None:
        if peak >= device_ram_bytes:
            problems.append(
                f"estimated runtime peak {peak} bytes does not fit device RAM {device_ram_bytes} bytes"
            )
        headroom = (device_ram_bytes - peak) / device_ram_bytes if device_ram_bytes else 0.0
    else:
        headroom = None
        problems.append("device RAM is unknown (MAP_ONLY); the fit cannot be assessed")
    if route_choice.route == rec.EDGE_ROUTES[0] and problems:
        action = "switch_to_route_B"
    elif route_choice.route == rec.EDGE_ROUTES[0]:
        action = "proceed_to_route_A"
    else:
        action = "technology_map_only"
    return {
        "action": action,
        "headroom_ratio": headroom,
        "problems": problems,
        "note": "硬门不满足即转技术地图与不采用 ADR，不得因沉没成本继续（step 13）",
    }


@dataclass
class ConversionDag:
    """Steps 14–16: export → rewrite → quant → compile → package, plus correctness."""

    dag_id: str
    nodes: Tuple[Mapping[str, Any], ...] = ()
    host_correctness_passed: bool = False
    dynamic_axes_preserved: bool = False
    state_io_declared: bool = False
    metadata_complete: bool = False

    NODE_KINDS: Tuple[str, ...] = ("export", "rewrite", "quant", "compile", "package")

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.dag_id:
            findings.append("ConversionDag: dag_id is required")
        if not self.nodes:
            findings.append("ConversionDag: no nodes recorded (一个 GUI 转换工具无日志 = FAIL)")
        for index, node in enumerate(self.nodes):
            kind = str(node.get("kind", ""))
            if kind not in self.NODE_KINDS:
                findings.append(f"ConversionDag node {index}: unknown kind {kind!r}")
            for key in ("tool", "version", "digest"):
                if not node.get(key):
                    findings.append(f"ConversionDag node {index} ({kind}): {key!r} is required")
            digest = str(node.get("digest", ""))
            if digest and not is_digest(digest):
                findings.append(f"ConversionDag node {index}: digest must be sha256:<hex>")
        if not self.dynamic_axes_preserved:
            findings.append("ConversionDag: static export loses dynamic length (动态 shape 降级必须声明)")
        if not self.state_io_declared:
            findings.append("ConversionDag: state/KV inputs and outputs must be declared")
        if not self.metadata_complete:
            findings.append("ConversionDag: op/dtype/metadata inventory is incomplete")
        if not self.host_correctness_passed:
            findings.append(
                "ConversionDag: host/reference correctness must pass before device work "
                "(否则把设备错误与转换错误混在一起)"
            )
        return findings

    def digest(self) -> str:
        return canonical_digest([dict(node) for node in self.nodes])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dag_id": self.dag_id,
            "nodes": [dict(node) for node in self.nodes],
            "host_correctness_passed": self.host_correctness_passed,
            "dynamic_axes_preserved": self.dynamic_axes_preserved,
            "state_io_declared": self.state_io_declared,
            "metadata_complete": self.metadata_complete,
            "digest": self.digest(),
        }


@dataclass
class EdgeAdapterContract:
    """Step 17: the minimal C4 adapter surface, with lazy import and error mapping."""

    adapter_id: str
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    operations: Tuple[str, ...] = ()
    import_lazy: bool = True
    error_classes: Mapping[str, str] = field(default_factory=dict)

    REQUIRED_OPERATIONS: Tuple[str, ...] = (
        "capability", "load", "run", "unload", "health", "actual_backend",
    )

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.adapter_id:
            findings.append("EdgeAdapterContract: adapter_id is required")
        missing = [name for name in self.REQUIRED_OPERATIONS if name not in self.operations]
        if missing:
            findings.append(f"EdgeAdapterContract: missing operations: {', '.join(missing)}")
        if not self.import_lazy:
            findings.append(
                "EdgeAdapterContract: the platform SDK must be imported lazily "
                "(平台 SDK 进入 core 是 FAIL，E14-08 step 36)"
            )
        if not self.capabilities:
            findings.append("EdgeAdapterContract: a capability descriptor is required")
        if not self.error_classes:
            findings.append("EdgeAdapterContract: runtime errors must be mapped to structured classes")
        for name in ("unsupported_op", "out_of_memory", "delegate_failure"):
            if name not in self.error_classes:
                findings.append(f"EdgeAdapterContract: error class {name!r} is not mapped")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "capabilities": dict(sorted(self.capabilities.items())),
            "operations": list(self.operations),
            "import_lazy": self.import_lazy,
            "error_classes": {key: self.error_classes[key] for key in sorted(self.error_classes)},
        }


def partition_placement(
    *, partitions: Sequence[Mapping[str, Any]], requested_backend: str
) -> Dict[str, Any]:
    """Step 21: per-op backend, fallback ops and boundary copies.

    ``E14-08`` §10: delegate enabled 等于全图 NPU is the misconception; the fraction
    of ops that actually ran on the accelerator is what a claim may rest on.
    """
    problems: List[str] = []
    on_device = 0
    total = 0
    boundary_copies = 0
    fallback_ops: List[str] = []
    for partition in partitions:
        ops = int(partition.get("op_count", 0))
        if ops <= 0:
            problems.append(f"partition {partition.get('partition_id')!r} has no ops")
            continue
        total += ops
        backend = str(partition.get("actual_backend", ""))
        if not backend:
            problems.append(f"partition {partition.get('partition_id')!r} does not record its backend")
        elif backend == requested_backend:
            on_device += ops
        else:
            fallback_ops.append(str(partition.get("partition_id", "")))
        boundary_copies += int(partition.get("boundary_copies", 0))
    if not partitions:
        problems.append("no partitions recorded")
    return {
        "requested_backend": requested_backend,
        "ops": total,
        "ops_on_requested_backend": on_device,
        "accelerated_op_fraction": on_device / total if total else 0.0,
        "fallback_partitions": fallback_ops,
        "boundary_copies": boundary_copies,
        "problems": problems,
        "note": "actual_backend 未知即不得声称 NPU 生效（step 20/21）",
    }


def device_correctness(
    *, reference: Mapping[str, Any], device: Mapping[str, Any], tolerance: Mapping[str, float]
) -> Dict[str, Any]:
    """Step 22: tensor/token/task reconciliation against the reference, before timing."""
    problems: List[str] = []
    fields = ("logits_max_abs", "top1_token_match", "task_metric")
    missing = [name for name in fields if name not in device]
    if missing:
        problems.append(f"device correctness does not report: {', '.join(missing)}")
    for name in fields:
        if name not in device or name not in reference:
            continue
        allowed = tolerance.get(name)
        if allowed is None:
            problems.append(f"no preregistered tolerance for {name!r}")
            continue
        if name == "top1_token_match":
            if not device[name]:
                problems.append("the device's top-1 token does not match the reference")
        else:
            delta = abs(float(device[name]) - float(reference[name]))
            if delta > allowed:
                problems.append(f"{name} differs by {delta:.6f} > {allowed:.6f}")
    return {
        "reference": dict(reference),
        "device": dict(device),
        "problems": problems,
        "quality_eligible": not problems,
    }


def cpu_baseline_same_device(
    *, cpu_latency_ms: Optional[float], device_latency_ms: Optional[float], device_id: str
) -> Dict[str, Any]:
    """Step 23: the baseline runs on the *same* device, not on a desktop CPU.

    ``E14-08`` §10: 桌面 CPU 作为移动 NPU speedup baseline is the error this refuses;
    a missing same-device baseline makes the comparison unavailable rather than
    substituted.
    """
    problems: List[str] = []
    if cpu_latency_ms is None:
        return {
            "device_id": device_id,
            "available": False,
            "problems": ["no same-device CPU baseline was measured; a desktop CPU may not substitute"],
            "speedup": None,
        }
    if device_latency_ms is None:
        problems.append("no device latency supplied")
    speedup = None
    if cpu_latency_ms and device_latency_ms:
        speedup = cpu_latency_ms / device_latency_ms
    return {
        "device_id": device_id,
        "available": not problems,
        "cpu_latency_ms": cpu_latency_ms,
        "device_latency_ms": device_latency_ms,
        "speedup": speedup,
        "problems": problems,
    }


def cold_warm_sustained(
    *,
    cold: Mapping[str, Any],
    warm: Mapping[str, Any],
    sustained: Mapping[str, Any],
    steady_window_s: float,
    evidence_level: str,
) -> Dict[str, Any]:
    """Steps 24–26: the three regimes, separated and all required.

    ``E14-08`` §10: 只测短 boost 外推持续性能 is a FAIL, so a missing sustained window
    is reported as missing rather than replaced by the warm number.
    """
    problems: List[str] = []
    if evidence_level not in EVIDENCE_LEVELS:
        problems.append(f"unknown evidence level {evidence_level!r}")
    if evidence_level == "MAP_ONLY":
        return {
            "evidence_level": evidence_level,
            "modes": {},
            "problems": ["MAP_ONLY evidence may not carry latency measurements"],
            "complete": False,
        }
    for label, dataset in (("cold", cold), ("warm", warm), ("sustained", sustained)):
        if not dataset:
            problems.append(f"the {label} regime was not measured")
            continue
        for key in ("latency_ms", "repeat"):
            if key not in dataset:
                problems.append(f"the {label} regime does not report {key!r}")
    if steady_window_s <= 0:
        problems.append("the sustained run has no steady window (短 boost 外推是不合格的)")
    if warm.get("latency_ms") and sustained.get("latency_ms"):
        degradation = float(sustained["latency_ms"]) / float(warm["latency_ms"])
        if degradation > 1.1:
            problems.append(
                f"steady-state latency is {degradation:.3f}× the warm latency: thermal throttling must be "
                "named in the claim, not averaged"
            )
    else:
        degradation = None
    return {
        "evidence_level": evidence_level,
        "modes": {"cold": dict(cold), "warm": dict(warm), "sustained": dict(sustained)},
        "steady_window_s": steady_window_s,
        "throttle_ratio": degradation,
        "problems": problems,
        "complete": not problems,
    }


def thermal_steady_state(
    samples: Sequence[Mapping[str, Any]], *, window_s: float, clock_drop_ratio: float
) -> Dict[str, Any]:
    """Step 26: frequencies, temperature and latency over time, with a steady window.

    ``E14-08`` §10: 短 boost 外推持续性能 is the failure, so the steady window is
    located from the samples and its clock/temperature state is reported.
    """
    problems: List[str] = []
    if not samples:
        return {"error": "no thermal samples supplied"}
    required = ("t_ns", "latency_ms", "temperature_c", "clock_hz", "average_power_w")
    for index, sample in enumerate(samples):
        missing = [key for key in required if key not in sample]
        if missing:
            problems.append(f"thermal sample {index} is missing {', '.join(missing)}")
    early = [sample for sample in samples if float(sample.get("t_ns", 0)) <= window_s * 1e9 * 0.2]
    late = [sample for sample in samples if float(sample.get("t_ns", 0)) >= window_s * 1e9 * 0.8]
    if not early or not late:
        problems.append("the run is too short to locate a steady window")
    clock_first = max((float(sample.get("clock_hz", 0)) for sample in early), default=0.0)
    clock_late = min((float(sample.get("clock_hz", 0)) for sample in late), default=0.0)
    drop = 1.0 if not clock_first else 1.0 - clock_late / clock_first
    if drop > clock_drop_ratio:
        problems.append(
            f"the sustained clock fell {drop * 100:.1f}% below its start: the run is thermally limited"
        )
    max_temp = max((float(sample.get("temperature_c", 0)) for sample in samples), default=0.0)
    return {
        "samples": len(samples),
        "window_s": window_s,
        "steady_latency_ms": late[-1].get("latency_ms") if late else None,
        "clock_drop_ratio": drop,
        "max_temperature_c": max_temp,
        "thermally_limited": drop > clock_drop_ratio,
        "problems": problems,
    }


def thread_affinity_scan(rows: Sequence[Mapping[str, Any]], *, preset: Sequence[str]) -> Dict[str, Any]:
    """Step 27: threads/cores/delegates over a *preregistered* set.

    ``E14-08`` §10: 选最快设置后不报告耗电和温度 is the failure, so power and
    temperature travel with every row.
    """
    problems: List[str] = []
    for row in rows:
        for key in ("configuration", "threads", "latency_ms", "average_power_w", "temperature_c"):
            if key not in row:
                problems.append(f"affinity row {row.get('configuration', '<unnamed>')} is missing {key!r}")
                break
    unbudgeted = sorted({str(row.get("configuration")) for row in rows} - set(preset))
    if unbudgeted:
        problems.append(f"configurations outside the preregistered set: {', '.join(unbudgeted)}")
    return {"rows": len(rows), "preset": list(preset), "problems": problems}


def shape_scan(rows: Sequence[Mapping[str, Any]], *, required: Sequence[str]) -> Dict[str, Any]:
    """Step 28: typical and boundary workloads, and whether the partition changed."""
    problems: List[str] = []
    partition_changes = 0
    for row in rows:
        missing = [key for key in required if key not in row]
        if missing:
            problems.append(f"shape row {row.get('shape', '<unnamed>')} is missing {', '.join(missing)}")
            continue
        if row.get("partition_changed"):
            partition_changes += 1
    return {
        "rows": len(rows),
        "partition_changes": partition_changes,
        "problems": problems,
        "note": "一个 shape 的 delegate 覆盖不能外推全部动态输入（step 28）",
    }


def memory_lifecycle(phases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Step 29: load/compile/prefill/decode RSS, PSS, native and device, plus release."""
    problems: List[str] = []
    required = ("phase", "rss_bytes", "pss_bytes", "native_bytes", "device_bytes")
    peak = 0
    peak_phase = ""
    for phase in phases:
        missing = [key for key in required if key not in phase]
        if missing:
            problems.append(f"memory phase {phase.get('phase', '<unnamed>')} is missing {', '.join(missing)}")
            continue
        total = int(phase["rss_bytes"]) + int(phase["native_bytes"]) + int(phase["device_bytes"])
        if total > peak:
            peak = total
            peak_phase = str(phase["phase"])
        if phase.get("released") is None and phase.get("phase") in ("unload", "after_request"):
            problems.append(f"phase {phase['phase']} does not state whether memory was released")
    if not phases:
        problems.append("no memory phases recorded")
    return {
        "phases": len(phases),
        "peak_bytes": peak,
        "peak_phase": peak_phase,
        "problems": problems,
        "note": "只读 Java heap 或单一 API 会低估峰值（step 29）",
    }


def power_energy(
    *, samples: Sequence[Mapping[str, Any]], source: str, calibrated: bool
) -> Dict[str, Any]:
    """Step 30: a calibrated/official interface, or an explicit ``unavailable``.

    ``E14-08`` §10: 功率估算冒充实测 is a FAIL, so an uncalibrated source produces
    ``available=False`` with the reason instead of a number.
    """
    problems: List[str] = []
    if source not in ("battery_counter", "device_rail", "external_meter", "vendor_tool", "estimate", "unavailable"):
        problems.append(f"unknown power source {source!r}")
    if source == "unavailable":
        return {
            "available": False,
            "source": source,
            "reason_code": "DEVICE_UNAVAILABLE",
            "problems": [],
            "note": "接口不可用时标 unavailable，不得用软件估算冒充（step 30）",
        }
    if source == "estimate":
        return {
            "available": False,
            "source": source,
            "reason_code": "NOT_IMPLEMENTED",
            "problems": ["an estimate is not a measurement"],
        }
    if not calibrated:
        problems.append(f"source {source!r} is not declared calibrated")
    if not samples:
        problems.append("no power samples supplied")
    energies: List[float] = []
    for index, sample in enumerate(samples):
        for key in ("t_ns", "average_power_w"):
            if key not in sample:
                problems.append(f"power sample {index} is missing {key!r}")
                break
        else:
            energies.append(float(sample["average_power_w"]))
    total_energy = None
    if len(samples) >= 2:
        first = float(samples[0]["t_ns"])
        last = float(samples[-1]["t_ns"])
        duration_s = (last - first) / 1e9
        total_energy = (sum(energies) / len(energies)) * duration_s if energies else None
    return {
        "available": not problems,
        "source": source,
        "calibrated": calibrated,
        "samples": len(samples),
        "energy_j": total_energy,
        "problems": problems,
    }


def package_and_startup(
    *, components: Mapping[str, int], install_time_s: Optional[float], startup_time_s: Optional[float]
) -> Dict[str, Any]:
    """Step 31: package composition and the cold startup experience."""
    problems: List[str] = []
    missing = [name for name in PACKAGE_COMPONENTS if name not in components]
    if missing:
        problems.append(f"package inventory is missing {', '.join(missing)}")
    total = sum(int(components.get(name, 0)) for name in PACKAGE_COMPONENTS)
    if install_time_s is None:
        problems.append("install time is unmeasured")
    if startup_time_s is None:
        problems.append("startup time is unmeasured")
    return {
        "components": {name: int(components.get(name, 0)) for name in PACKAGE_COMPONENTS},
        "total_bytes": total,
        "install_time_s": install_time_s,
        "startup_time_s": startup_time_s,
        "problems": problems,
        "note": "性能好但发布包不可接受也必须进入决策（step 31）",
    }


def background_disturbance(rows: Sequence[Mapping[str, Any]], *, deadline_ms: float) -> Dict[str, Any]:
    """Step 32: controlled background load, so jitter and deadline misses are visible."""
    problems: List[str] = []
    for row in rows:
        for key in ("condition", "p50_ms", "p99_ms", "deadline_miss_rate", "jitter_ms"):
            if key not in row:
                problems.append(f"background row {row.get('condition', '<unnamed>')} is missing {key!r}")
                break
    worst = max((float(row.get("deadline_miss_rate", 0.0)) for row in rows), default=0.0)
    if worst > 0.01:
        problems.append(f"deadline miss rate reached {worst:.3%} under background load")
    return {
        "conditions": len(rows),
        "worst_deadline_miss_rate": worst,
        "deadline_ms": deadline_ms,
        "problems": problems,
        "note": "实验室空机平均值不能代表真实端侧（step 32）",
    }


# ── robustness and isolation (steps 33–36) ─────────────────────────────────


def unsupported_op(
    *, op: str, placement: str, rejected: bool, fell_back: bool, reason_code: str, actual_backend: str
) -> Dict[str, Any]:
    """Step 33: an unsupported op must be refused or explicitly fallen back.

    ``E14-08`` §10: partial CPU fallback 不披露 causes a long tail that no average
    latency reveals.
    """
    problems: List[str] = []
    if placement not in OP_PLACEMENT:
        problems.append(f"unknown placement {placement!r}")
    if placement in ("fallback", "unsupported") and not (rejected or fell_back):
        problems.append(f"op {op!r} is {placement} but neither rejected nor fallen back")
    if fell_back and not reason_code:
        problems.append(f"op {op!r} fell back without a reason code")
    if placement == "supported" and not actual_backend:
        problems.append(f"op {op!r} claims support without an actual backend")
    return {
        "op": op,
        "placement": placement,
        "reason_code": reason_code,
        "actual_backend": actual_backend,
        "disclosed": bool(reason_code),
        "problems": problems,
        "ok": not problems,
    }


def corrupted_model_check(
    *, checksum_valid: bool, runtime_version_matches: bool, rejected_before_load: bool
) -> Dict[str, Any]:
    """Step 34: a corrupt or wrong-version model must be refused before loading."""
    problems: List[str] = []
    if not rejected_before_load:
        problems.append(
            "no pre-load validation recorded (设备崩溃或接受错误格式 = FAIL)"
        )
    if not checksum_valid and not rejected_before_load:
        problems.append("a checksum mismatch reached the runtime")
    if not runtime_version_matches and not rejected_before_load:
        problems.append("a runtime version mismatch reached the runtime")
    return {
        "checksum_valid": checksum_valid,
        "runtime_version_matches": runtime_version_matches,
        "rejected_before_load": rejected_before_load,
        "problems": problems,
        "safe": not problems,
    }


def low_memory_cancel(events: Sequence[Mapping[str, Any]]) -> List[str]:
    """Step 35: admission, structured errors, release and the next request's health."""
    problems: List[str] = []
    for event in events:
        phase = str(event.get("phase", ""))
        if phase not in ("admission", "load", "run", "cancel", "after_recovery"):
            problems.append(f"low-memory phase {phase!r} is not documented")
        if event.get("structured_error") is False:
            problems.append(f"{phase}: the failure was not reported as a structured error")
        if event.get("resource_released") is False:
            problems.append(f"{phase}: resources were not released")
        if phase == "after_recovery" and event.get("next_request_ok") is False:
            problems.append("the request after recovery failed: residual state was not cleaned")
    if not events:
        problems.append("no low-memory cases recorded")
    return problems


def core_isolation_check(*, core_install_clean: bool, imports_platform_sdk: bool, ci_installs_edge_extra: bool) -> List[str]:
    """Step 36: the edge toolchain must not enter the core wheel, import graph or CI.

    ``E14-08`` §10: 平台 SDK 进入 core binds the main package to a mobile toolchain,
    so this is a gate rather than a guideline (see also ``E14-01``).
    """
    problems: List[str] = []
    if imports_platform_sdk:
        problems.append("a core module imports the platform SDK (must be a lazy import inside the adapter)")
    if not core_install_clean:
        problems.append("the core install is not clean of the edge extra")
    if ci_installs_edge_extra:
        problems.append("a core CI job installs the edge extra")
    return problems


def technology_map(
    axes: Mapping[str, Mapping[str, Any]], *, route: str
) -> Dict[str, Any]:
    """Step 37: every axis with a verdict *and* a source, or an explicit ``unknown``.

    ``E14-08`` §10: 技术地图的 unknown 填成支持 is the failure; the map therefore
    requires a source per axis and reports the unknowns separately.
    """
    problems: List[str] = []
    if route not in rec.EDGE_ROUTES:
        problems.append(f"technology_map: unknown route {route!r}")
    rows: List[Dict[str, Any]] = []
    unknown: List[str] = []
    for axis in MAP_AXES:
        entry = axes.get(axis)
        if entry is None:
            problems.append(f"technology_map: axis {axis!r} is missing")
            continue
        if not entry.get("source") or not entry.get("version"):
            problems.append(f"technology_map: axis {axis!r} has no authoritative source/version")
        verdict = str(entry.get("verdict", ""))
        if verdict == "unknown":
            unknown.append(axis)
        elif not verdict:
            problems.append(f"technology_map: axis {axis!r} has no verdict")
        rows.append({"axis": axis, **dict(entry)})
    if unknown and route == rec.EDGE_ROUTES[1] and not problems:
        pass  # route B may conclude "unknown" for some axes; that is the honest answer
    return {
        "route": route,
        "rows": rows,
        "unknown_axes": unknown,
        "problems": problems,
        "note": "unknown 必须保留为 unknown，不得填成支持（step 37）",
    }


def edge_execution_record(
    *,
    device_id: Optional[str],
    runtime_id: str,
    model_artifact_id: str,
    input_id: str,
    mode: str,
    requested_backend: str,
    actual_backend: str,
    accelerated_op_fraction: Optional[float],
    fallback_ops: Sequence[str],
    latency_ms: Optional[float],
    memory_peak_bytes: Optional[int],
    average_power_w: Optional[float],
    energy_j: Optional[float],
    temperature_c: Optional[float],
    quality_status: str,
    evidence_level: str,
) -> Dict[str, Any]:
    """``E14-08`` §8 ``EdgeExecutionRecord`` with the evidence level enforced."""
    problems: List[str] = []
    if mode not in TIMING_MODES:
        problems.append(f"mode {mode!r} must be one of {', '.join(TIMING_MODES)}")
    if evidence_level not in EVIDENCE_LEVELS:
        problems.append(f"evidence_level {evidence_level!r} must be one of {', '.join(EVIDENCE_LEVELS)}")
    if evidence_level == "DEVICE_MEASURED" and not device_id:
        problems.append("DEVICE_MEASURED requires a device id")
    if evidence_level == "MAP_ONLY" and any(
        value is not None for value in (latency_ms, average_power_w, energy_j, memory_peak_bytes)
    ):
        problems.append("MAP_ONLY evidence may not carry device measurements")
    if not actual_backend:
        problems.append("actual_backend is required (未知即不得声称 NPU 生效)")
    return {
        "schema_version": f"{SCHEMA_PREFIX}.e14-08.exec.v1",
        "device_id": device_id,
        "runtime_id": runtime_id,
        "model_artifact_id": model_artifact_id,
        "input_id": input_id,
        "mode": mode,
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "accelerated_op_fraction": accelerated_op_fraction,
        "fallback_ops": list(fallback_ops),
        "latency_ms": latency_ms,
        "memory_peak_bytes": memory_peak_bytes,
        "average_power_w": average_power_w,
        "energy_j": energy_j,
        "temperature_c": temperature_c,
        "quality_status": quality_status,
        "evidence_level": evidence_level,
        "problems": problems,
    }


# ── adoption (step 40) ─────────────────────────────────────────────────────


def edge_adoption(
    *,
    decision_id: str,
    route: str,
    evidence_level: str,
    correctness: Mapping[str, Any],
    timing: Mapping[str, Any],
    partition: Mapping[str, Any],
    package: Mapping[str, Any],
    isolation_problems: Sequence[str],
    technology_map_result: Mapping[str, Any],
    evidence_refs: Sequence[str],
) -> AdoptionDecision:
    """Step 40: route A needs device evidence; route B may only reject or block.

    ``E14-08`` §12: 因投入开发自动决定采用 is the failure — a route-A candidate with
    an unmet quality or timing gate is rejected, and route B can never reach an
    adoption decision.
    """
    problems: List[str] = []
    if route not in rec.EDGE_ROUTES:
        problems.append(f"unknown route {route!r}")
    if evidence_level not in EVIDENCE_LEVELS:
        problems.append(f"unknown evidence level {evidence_level!r}")
    problems.extend(isolation_problems)
    if route == rec.EDGE_ROUTES[0]:
        if evidence_level != "DEVICE_MEASURED":
            problems.append("route A requires DEVICE_MEASURED evidence; a technology map is not an adapter")
        if not correctness.get("quality_eligible"):
            problems.append("device correctness/quality gate failed")
        if not timing.get("complete"):
            problems.append("cold/warm/sustained timing is incomplete")
        if not partition.get("accelerated_op_fraction"):
            problems.append("no accelerated op fraction: 'NPU enabled' is not evidence")
        if package.get("problems"):
            problems.append("package/startup gate failed")
        decision = rec.ADOPT_EXPERIMENTAL if not problems else (
            rec.REJECT_QUALITY if not correctness.get("quality_eligible") else rec.BLOCKED_EVIDENCE
        )
        allowed: Tuple[str, ...] = (
            (f"adapter {route} 在 {evidence_level} 证据等级下通过质量与三态计时门",) if not problems else ()
        )
    else:
        if not technology_map_result.get("rows"):
            problems.append("route B requires a populated technology map")
        decision = rec.N_A_BY_ADR
        allowed = ()
    return AdoptionDecision(
        decision_id=decision_id,
        experiment_id=EXPERIMENT_ID,
        decision=decision,
        allowed_claims=allowed,
        forbidden_claims=(
            "模拟器数字冒充实测",
            "delegate enabled 等于全图 NPU",
            "桌面 CPU 作为移动 NPU speedup baseline",
            "功率估算冒充实测",
            "技术地图的 unknown 填成支持",
        ),
        quality_status=rec.STATUS_PASS if not problems else rec.STATUS_FAIL_QUALITY,
        performance_status=rec.STATUS_NOT_RUN,
        maturity=(
            rec.MATURITY_KERNEL_PROFILED
            if decision == rec.ADOPT_EXPERIMENTAL
            else rec.MATURITY_DESIGN_ONLY
        ),
        evidence_refs=tuple(evidence_refs),
        limitations=tuple(problems) + ("结论限定于一个平台—runtime 组合和一个设备",),
        reopened_if=("获得目标设备与测量工具时可从路线 B 转入路线 A",),
    )


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the E14-08 interfaces (labelled smoke, not an experiment)."""
    route = RouteChoice(
        route="ROUTE_B_TECHNOLOGY_MAP", has_device=False, has_measurement_tooling=False,
        has_permission=False, rationale="no target device available",
    )
    coverage = op_coverage_map(
        ops=[
            {"op": "conv2d", "shape": [1, 3, 224, 224], "dtype": "fp16", "placement": "supported"},
            {"op": "custom_silu", "shape": [1, 256], "dtype": "fp16", "placement": "fallback"},
        ],
        runtime_id="tflite", source_version="2.16",
    )
    estimate = estimate_memory_and_package(
        weights_bytes=100, kv_bytes=10, activation_bytes=5, workspace_bytes=5,
        runtime_library_bytes=8, app_assets_bytes=2,
    )
    feasibility = realtime_feasibility(
        deadline_ms=50.0, estimated_latency_ms=30.0, estimated_energy_j=1.0, thermal_budget_j=2.0
    )
    gate = go_no_go(
        route_choice=route, feasibility=feasibility, memory_estimate=estimate, device_ram_bytes=1024
    )
    partition = partition_placement(
        partitions=[
            {"partition_id": "p0", "op_count": 90, "actual_backend": "npu", "boundary_copies": 2},
            {"partition_id": "p1", "op_count": 10, "actual_backend": "cpu", "boundary_copies": 1},
        ],
        requested_backend="npu",
    )
    record = edge_execution_record(
        device_id=None, runtime_id="tflite", model_artifact_id="m", input_id="i", mode="warm",
        requested_backend="npu", actual_backend="unknown", accelerated_op_fraction=None,
        fallback_ops=(), latency_ms=None, memory_peak_bytes=None, average_power_w=None,
        energy_j=None, temperature_c=None, quality_status=rec.STATUS_NOT_RUN,
        evidence_level="MAP_ONLY",
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "route_problems": route.problems(),
        "coverage": round(coverage["coverage"], 4),
        "runtime_peak_bytes": estimate["runtime_peak_bytes"],
        "feasible": feasibility["feasible"],
        "go_no_go_action": gate["action"],
        "accelerated_op_fraction": round(partition["accelerated_op_fraction"], 4),
        "map_only_record_problems": len(record["problems"]),
        "platform_runtimes": len(PLATFORM_RUNTIMES),
        "map_axes": len(MAP_AXES),
    }


# ── protocol step table (40 steps of details/S14/E14-08) ───────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "用 ADR 选择路线 A 或 B", ("edge:RouteChoice", "records:EDGE_ROUTES")),
    (2, "冻结一个平台—runtime 组合", ("edge:PlatformRuntimeChoice", "edge:PLATFORM_RUNTIMES")),
    (3, "冻结端侧任务", ("edge:realtime_feasibility", "frontier:BaselinePair")),
    (4, "冻结设备身份", ("edge:DeviceFingerprint", "edge:DEVICE_FINGERPRINT_FIELDS")),
    (5, "冻结 ModelArtifact 与许可", ("contracts:ServingModelArtifact", "edge:ConversionDag")),
    (6, "冻结输入与质量门", ("edge:device_correctness", "frontier:QualityGate")),
    (7, "冻结性能/功耗边界", ("edge:TIMING_MODES", "edge:cold_warm_sustained")),
    (8, "冻结安全与系统边界", ("campaign:REQUIRED_ISOLATION", "edge:RouteChoice.has_permission")),
    (9, "建立 runtime capability map", ("edge:RuntimeCapabilityMap", "edge:RuntimeCapabilityMap.problems")),
    (10, "建立 op coverage map", ("edge:op_coverage_map", "edge:OP_PLACEMENT")),
    (11, "估计内存和包体", ("edge:estimate_memory_and_package", "edge:PACKAGE_COMPONENTS")),
    (12, "估计实时和能耗可行性", ("edge:realtime_feasibility", "frontier:PredictionModel")),
    (13, "执行 Go/No-Go 决策", ("edge:go_no_go", "frontier:StopRules")),
    (14, "建立转换 DAG（路线 A）", ("edge:ConversionDag", "edge:ConversionDag.NODE_KINDS")),
    (15, "验证转换 inventory/shape", ("edge:ConversionDag.dynamic_axes_preserved",
                                       "edge:ConversionDag.state_io_declared")),
    (16, "验证 host/reference correctness", ("edge:ConversionDag.host_correctness_passed", "parity:evaluate_gates")),
    (17, "实现最小 C4 adapter（路线 A）", ("edge:EdgeAdapterContract", "edge:EdgeAdapterContract.REQUIRED_OPERATIONS")),
    (18, "构建目标应用/二进制", ("edge:package_and_startup", "identity:file_inventory")),
    (19, "部署到清洁目标设备", ("edge:DeviceFingerprint.evidence_level", "campaign:isolation_clause")),
    (20, "运行设备 capability probe", ("edge:partition_placement", "edge:RuntimeCapabilityMap.profiler_available")),
    (21, "验证逐 op/partition placement", ("edge:partition_placement", "edge:OP_PLACEMENT")),
    (22, "运行设备 correctness/quality", ("edge:device_correctness", "contracts:check_quality_before_performance")),
    (23, "建立 CPU baseline", ("edge:cpu_baseline_same_device", "frontier:BaselinePair.shared_fields")),
    (24, "测 cold load/compile/first inference", ("edge:cold_warm_sustained", "edge:TIMING_MODES")),
    (25, "测 warm single-stream", ("edge:cold_warm_sustained", "edge:edge_execution_record")),
    (26, "测 sustained thermal steady-state", ("edge:thermal_steady_state", "edge:cold_warm_sustained")),
    (27, "测线程/亲和性扫描", ("edge:thread_affinity_scan", "frontier:AblationMatrix")),
    (28, "测 shape/context/batch 扫描", ("edge:shape_scan", "frontier:WorkloadStrata")),
    (29, "测内存生命周期", ("edge:memory_lifecycle", "contracts:check_resource_ledger")),
    (30, "测功率/能耗", ("edge:power_energy", "records:REASON_CODES")),
    (31, "测应用包体和启动体验", ("edge:package_and_startup", "frontier:CostDenominator")),
    (32, "测后台/系统扰动", ("edge:background_disturbance", "frontier:StopRules")),
    (33, "测试 unsupported op/shape", ("edge:unsupported_op", "edge:OP_PLACEMENT")),
    (34, "测试损坏/错版本模型", ("edge:corrupted_model_check", "records:STATUS_INVALID_IDENTITY")),
    (35, "测试低内存和取消", ("edge:low_memory_cancel", "records:STATUS_FAIL_RECOVERY")),
    (36, "验证 core 依赖隔离", ("edge:core_isolation_check", "dependencies:analyse_import_trace")),
    (37, "形成路线 B 技术地图", ("edge:technology_map", "edge:MAP_AXES")),
    (38, "在 holdout input/steady run 确认", ("frontier:WorkloadStrata.holdout_id", "edge:thermal_steady_state")),
    (39, "跨安装/运行重复", ("records:EXPERIMENT_UNITS", "frontier:StatisticsPlan.minimum_repeats")),
    (40, "形成 EdgeAdoptionDecision", ("edge:edge_adoption", "contracts:AdoptionDecision.validate")),
)