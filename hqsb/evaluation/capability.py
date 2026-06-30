"""E12-02: capability registry, probe evidence matrix and invalidation rules.

"Supported" is not a boolean and not a spec sheet.  This module separates four
levels — ``DECLARED`` (vendor documentation), ``DISCOVERED`` (an API enumerated
it), ``VERIFIED`` (a minimal probe actually compiled, executed, synchronised,
checked and recorded its actual backend) and ``BENCHMARKED`` (E12-03 ran the
formal workload) — and it refuses to answer a probe failure with "unsupported":

* a missing component, a permission denial, a timeout, a compile failure and a
  numerical mismatch stay distinguishable;
* a silent fallback can make a *feature* fallback-capable, but it can never
  verify the requested feature;
* evidence carries an invalidation key (driver/runtime/framework/container/arch/
  permission) so a software upgrade cannot keep an old ``VERIFIED`` alive.

Nothing here probes hardware or executes an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.identity import sha256_text, stable_id
from hqsb.evaluation.records import TABLE_SCHEMAS

EXPERIMENT_ID = "E12-02"
TITLE = "跨平台 Capability 自动探测与运行证据矩阵"
CLAIM_BOUNDARY = (
    "本实验通过只证明当前 SUT 能执行所需最小能力，"
    "不证明其性能、稳定性、能效或成本优越；未验证能力不得写成 supported。"
)

SCHEMA_VERSION = "1.0.0"

# ── vocabulary ────────────────────────────────────────────────────────────

FEATURE_LAYERS: Tuple[str, ...] = (
    "platform",
    "numeric",
    "kernel",
    "runtime",
    "service",
    "distributed",
    "telemetry",
)

CAPABILITY_DECLARED = "DECLARED"
CAPABILITY_DISCOVERED = "DISCOVERED"
CAPABILITY_VERIFIED = "VERIFIED"
CAPABILITY_BENCHMARKED = "BENCHMARKED"

CAPABILITY_LEVELS: Tuple[str, ...] = (
    CAPABILITY_DECLARED,
    CAPABILITY_DISCOVERED,
    CAPABILITY_VERIFIED,
    CAPABILITY_BENCHMARKED,
)

LEVEL_ORDER: Mapping[str, int] = {name: index for index, name in enumerate(CAPABILITY_LEVELS)}

#: Only these levels may be reported as ``supported`` in a capability matrix.
SUPPORTING_LEVELS: Tuple[str, ...] = (CAPABILITY_VERIFIED, CAPABILITY_BENCHMARKED)

#: Probe failure taxonomy (§5.4).  A failure is *classified*, never generalised.
PROBE_FAILURE_CATEGORIES: Tuple[str, ...] = (
    "UNSUPPORTED_BY_HARDWARE",
    "UNSUPPORTED_BY_RUNTIME",
    "MISSING_COMPONENT",
    "VERSION_INCOMPATIBLE",
    "PERMISSION_DENIED",
    "RESOURCE_UNAVAILABLE",
    "COMPILE_FAILED",
    "LOAD_FAILED",
    "EXECUTION_FAILED",
    "TIMEOUT",
    "NUMERICAL_MISMATCH",
    "PROBE_BUG",
    "UNKNOWN",
)

#: Failure categories that say nothing about the *device*.
DEVICE_INDEPENDENT_FAILURES: Tuple[str, ...] = (
    "MISSING_COMPONENT",
    "VERSION_INCOMPATIBLE",
    "PERMISSION_DENIED",
    "RESOURCE_UNAVAILABLE",
    "TIMEOUT",
    "PROBE_BUG",
    "UNKNOWN",
)

INVALIDATION_TRIGGERS: Tuple[str, ...] = (
    "driver_change",
    "runtime_change",
    "framework_change",
    "container_image_change",
    "feature_implementation_change",
    "arch_change",
    "permission_change",
    "device_instance_change",
    "host_change",
)

#: Changes that do *not* invalidate an evidence row, with the reason why.
INVALIDATION_WHITELIST: Mapping[str, str] = {
    "benchmark_workload_change": "the probe result does not depend on the E12-03 workload",
    "display_name_change": "reporting metadata only",
    "price_change": "an economic input, unrelated to device capability",
}


def meets(level: str, required: str) -> bool:
    if level not in LEVEL_ORDER or required not in LEVEL_ORDER:
        raise ConfigError(f"unknown capability level ({level!r} / {required!r})")
    return LEVEL_ORDER[level] >= LEVEL_ORDER[required]


# ── feature registry ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Feature:
    """One fine-grained capability requirement."""

    feature_id: str
    layer: str
    description: str
    success_criteria: str
    evidence_ttl_s: int = 30 * 24 * 3600
    depends_on: Tuple[str, ...] = ()
    criticality: str = "P0"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.layer not in FEATURE_LAYERS:
            problems.append(f"unknown feature layer {self.layer!r}")
        if not self.success_criteria:
            problems.append(f"{self.feature_id} has no success criteria")
        if self.evidence_ttl_s <= 0:
            problems.append(f"{self.feature_id} needs a positive evidence TTL")
        if self.criticality not in ("P0", "P1", "P2"):
            problems.append(f"unknown criticality {self.criticality!r}")
        if self.feature_id in self.depends_on:
            problems.append(f"{self.feature_id} depends on itself")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "layer": self.layer,
            "description": self.description,
            "inputs": "frozen probe inputs (see probe spec)",
            "success_criteria": self.success_criteria,
            "evidence_ttl": self.evidence_ttl_s,
            "depends_on": list(self.depends_on),
            "required_by_group": "",
            "criticality": self.criticality,
        }


DTYPES: Tuple[str, ...] = ("fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4")

_DTYPE_ASPECTS: Tuple[Tuple[str, str], ...] = (
    ("storage", "allocate → initialise → copy → synchronise → checksum matches"),
    ("elementwise", "add/mul on small non-degenerate tensors within the frozen tolerance"),
    ("reduction_accum_fp32", "reduction with fp32 accumulation matches the CPU reference"),
    ("gemm_native", "minimal GEMM uses the native matrix path (instruction path observable)"),
)


def _numeric_features() -> Tuple[Feature, ...]:
    rows: List[Feature] = []
    for dtype in DTYPES:
        for aspect, criteria in _DTYPE_ASPECTS:
            depends = () if aspect == "storage" else (f"{dtype}_storage",)
            rows.append(
                Feature(
                    feature_id=f"{dtype}_{aspect}",
                    layer="numeric",
                    description=f"{dtype} {aspect.replace('_', ' ')}",
                    success_criteria=criteria,
                    depends_on=depends,
                    evidence_ttl_s=90 * 24 * 3600,
                )
            )
    rows.append(
        Feature(
            feature_id="int8_dequant_path",
            layer="numeric",
            description="INT8 packing/scale/zero-point dequantisation",
            success_criteria="dequant output within tolerance and packing documented",
            depends_on=("int8_storage",),
        )
    )
    rows.append(
        Feature(
            feature_id="int4_pack_unpack",
            layer="numeric",
            description="INT4 packing layout and unpack path",
            success_criteria="packed layout round-trips and dequant matches the reference",
            depends_on=("int4_storage",),
        )
    )
    return tuple(rows)


def _kernel_features() -> Tuple[Feature, ...]:
    rows = [
        ("cuda_kernel_toolchain", "compile→load→launch→sync→check for a CUDA kernel", ()),
        ("triton_kernel_toolchain", "compile→load→launch→sync→check for a Triton kernel", ()),
        ("cutlass_kernel_toolchain", "compile→load→launch→sync→check for a CUTLASS kernel", ()),
        ("ascendc_kernel_toolchain", "compile→load→launch→sync→check for an AscendC kernel", ()),
        ("aot_binary_load", "prebuilt binary/code object loads on the target arch", ()),
        ("arch_target_rejection", "an incompatible arch target is rejected, not silently recompiled", ()),
        ("kernel_profiler_counters", "kernel counters/timeline can be collected", ()),
        ("graph_capture_path", "graph capture succeeds and the actual graph is traceable", ()),
    ]
    return tuple(
        Feature(
            feature_id=name,
            layer="kernel",
            description=description,
            success_criteria="the probe records compile/load/launch/sync/check stages plus actual backend",
            depends_on=depends,
        )
        for name, description, depends in rows
    )


def _runtime_features() -> Tuple[Feature, ...]:
    rows = [
        ("model_load", "load the frozen ModelArtifact with hash verification", ("bf16_storage",)),
        ("attention_kv_path", "minimal prefill/decode attention with KV read/write", ("bf16_gemm_native",)),
        ("runtime_engine", "requested runtime engine actually executes", ("model_load",)),
        ("dynamic_shape_path", "dynamic shapes/lowering/guard are visible in a trace", ("graph_capture_path",)),
        ("quant_engine", "a quantised engine executes the quantised artifact", ("int8_dequant_path",)),
        ("fallback_visibility", "requested vs actual backend is observable in metadata/trace", ("runtime_engine",)),
    ]
    return tuple(
        Feature(
            feature_id=name,
            layer="runtime",
            description=description,
            success_criteria="selected tensors/logits/tokens and the actual engine are recorded",
            depends_on=depends,
        )
        for name, description, depends in rows
    )


def _service_features() -> Tuple[Feature, ...]:
    rows = [
        ("service_start", "cold start → ready with a recorded stage breakdown", ("runtime_engine",)),
        ("service_streaming", "streaming and non-streaming requests return the same token ids", ("service_start",)),
        ("service_open_loop", "open-loop request-rate generation with a fixed trace", ("service_start",)),
        ("service_metrics", "metrics/trace correlation per request", ("service_start",)),
        ("service_cancellation", "cancellation/timeout semantics are observable", ("service_start",)),
        ("continuous_batching", "continuous batching is enabled and observable", ("service_start",)),
        ("prefix_cache", "prefix/prefix-cache reuse is observable", ("service_start",)),
    ]
    return tuple(
        Feature(
            feature_id=name,
            layer="service",
            description=description,
            success_criteria="capability only: the probe proves the feature works, not that it is fast",
            depends_on=depends,
        )
        for name, description, depends in rows
    )


def _distributed_features() -> Tuple[Feature, ...]:
    ops = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
    rows = [
        Feature(
            feature_id=f"bf16_collective_{op}",
            layer="distributed",
            description=f"bf16 {op} correctness-first collective",
            success_criteria="result matches the reference and async errors are surfaced",
            depends_on=("topology_discovery",),
        )
        for op in ops
    ]
    rows.extend(
        [
            Feature(
                feature_id="topology_discovery",
                layer="distributed",
                description="physical/logical topology and rank→device map",
                success_criteria="topology and rank mapping are recorded",
                criticality="P0",
            ),
            Feature(
                feature_id="p2p_access",
                layer="distributed",
                description="peer access / link state",
                success_criteria="peer access state is readable and reported",
                depends_on=("topology_discovery",),
            ),
            Feature(
                feature_id="parallel_plan_support",
                layer="distributed",
                description="shard/parallel plan prerequisites",
                success_criteria=(
                    "a minimal multi-rank forward completes with cross-rank model identity"
                ),
                depends_on=("bf16_collective_all_reduce",),
                criticality="P1",
            ),
        ]
    )
    return tuple(rows)


def _telemetry_features() -> Tuple[Feature, ...]:
    rows: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
        ("telemetry_power", "instantaneous/averaged power field with semantics", "accelerator.power_W", ()),
        ("telemetry_energy_accumulator", "cumulative energy counter with wrap/reset semantics", "accelerator.energy_J", ("telemetry_power",)),
        ("telemetry_temperature", "temperature field", "accelerator.temperature_C", ()),
        ("telemetry_clock", "core/memory clock fields", "accelerator.clock_mhz", ()),
        ("telemetry_throttle", "throttle reason bitmask", "accelerator.throttle_reasons", ()),
        ("telemetry_ecc", "ECC/RAS counters", "accelerator.ecc_errors", ()),
        ("telemetry_permission", "collector runs without elevated permission", "accelerator.power_W", ("telemetry_power",)),
    )
    return tuple(
        Feature(
            feature_id=name,
            layer="telemetry",
            description=description,
            success_criteria=(
                f"canonical field {canonical} has boundary/unit/semantics/resolution and can be sampled"
            ),
            depends_on=depends,
            evidence_ttl_s=30 * 24 * 3600,
        )
        for name, description, canonical, depends in rows
    )


def _platform_features() -> Tuple[Feature, ...]:
    rows: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
        ("device_discovery", "vendor + runtime device enumeration", "device count/arch/UUID recorded and cross-checked", ()),
        ("memory_capacity_readable", "memory capacity/type/ECC readable", "capacity and type recorded", ("device_discovery",)),
        ("interconnect_discovery", "PCIe/NVLink/XGMI/NUMA topology readable", "link state recorded", ("device_discovery",)),
        ("partition_visibility", "partition/MIG/virtualisation state readable", "partition state recorded", ("device_discovery",)),
    )
    return tuple(
        Feature(
            feature_id=name,
            layer="platform",
            description=description,
            success_criteria=criteria,
            depends_on=depends,
            evidence_ttl_s=90 * 24 * 3600,
        )
        for name, description, criteria, depends in rows
    )


FEATURE_REGISTRY: Tuple[Feature, ...] = (
    _platform_features()
    + _numeric_features()
    + _kernel_features()
    + _runtime_features()
    + _service_features()
    + _distributed_features()
    + _telemetry_features()
)

FEATURE_BY_ID: Mapping[str, Feature] = {row.feature_id: row for row in FEATURE_REGISTRY}


def feature(feature_id: str) -> Feature:
    try:
        return FEATURE_BY_ID[feature_id]
    except KeyError as exc:
        raise ConfigError(f"unknown feature {feature_id!r}") from exc


def features_for_layer(layer: str) -> Tuple[Feature, ...]:
    return tuple(row for row in FEATURE_REGISTRY if row.layer == layer)


def registry_problems() -> List[str]:
    problems: List[str] = []
    for row in FEATURE_REGISTRY:
        problems.extend(f"{row.feature_id}: {item}" for item in row.validate())
        for dependency in row.depends_on:
            if dependency not in FEATURE_BY_ID:
                problems.append(f"{row.feature_id} depends on unknown feature {dependency!r}")
    if len(FEATURE_REGISTRY) < 40:
        problems.append("the registry is too coarse: S12 requires fine-grained feature ids")
    problems.extend(f"duplicate feature id {name}" for name in _duplicates(row.feature_id for row in FEATURE_REGISTRY))
    return problems


def _duplicates(values: Iterable[str]) -> Tuple[str, ...]:
    seen: set = set()
    dupes: List[str] = []
    for value in values:
        if value in seen:
            dupes.append(value)
        seen.add(value)
    return tuple(sorted(set(dupes)))


TRIGGER_FIELD: Mapping[str, str] = {
    "driver_change": "driver",
    "runtime_change": "runtime",
    "framework_change": "framework",
    "container_image_change": "container_image",
    "feature_implementation_change": "feature_implementation",
    "arch_change": "arch",
    "permission_change": "permission",
    "device_instance_change": "host",
    "host_change": "host",
}


def dependency_order(feature_ids: Sequence[str]) -> Tuple[str, ...]:
    """Topological order, dependencies first; a cycle fails closed.

    Transitive dependencies are included even when they were not requested, so
    a probe plan can never be built on an unverified prerequisite.
    """
    for name in feature_ids:
        feature(name)
    resolved: List[str] = []
    visiting: set = set()

    def visit(name: str) -> None:
        if name in resolved:
            return
        if name in visiting:
            raise ConfigError(f"feature dependency cycle at {name!r}")
        visiting.add(name)
        for dependency in feature(name).depends_on:
            visit(dependency)
        visiting.discard(name)
        resolved.append(name)

    for name in sorted(feature_ids):
        visit(name)
    return tuple(resolved)


# ── probe classification ──────────────────────────────────────────────────


def classify_probe_outcome(
    *,
    component_present: bool = True,
    permission_denied: bool = False,
    timeout: bool = False,
    build_ok: bool = False,
    load_ok: bool = False,
    execution_ok: bool = False,
    sync_ok: bool = False,
    numerical_ok: bool = True,
    hardware_supports: Optional[bool] = None,
    runtime_supports: Optional[bool] = None,
    signal: Optional[str] = None,
    exit_code: int = 0,
) -> Dict[str, Any]:
    """Classify a probe outcome; never generalise into ``unsupported``."""
    if not component_present:
        return _failure("MISSING_COMPONENT", "the component/library is not installed")
    if permission_denied:
        return _failure("PERMISSION_DENIED", "the probe was denied access")
    if timeout:
        return _failure("TIMEOUT", "the probe exceeded its frozen timeout")
    if not build_ok:
        return _failure("COMPILE_FAILED", f"build failed (exit={exit_code}, signal={signal})")
    if not load_ok:
        return _failure("LOAD_FAILED", "artifact/module could not be loaded")
    if not execution_ok:
        if hardware_supports is False:
            return _failure("UNSUPPORTED_BY_HARDWARE", "the device rejects the operation")
        if runtime_supports is False:
            return _failure("UNSUPPORTED_BY_RUNTIME", "the runtime rejects the operation")
        return _failure("EXECUTION_FAILED", f"execution failed (exit={exit_code}, signal={signal})")
    if not sync_ok:
        return _failure("EXECUTION_FAILED", "the probe could not be synchronised")
    if not numerical_ok:
        return _failure("NUMERICAL_MISMATCH", "the output does not match the reference")
    return {
        "status": "OK",
        "failure_category": "",
        "reason": "",
        "verified": True,
        "note": "execute+sync+correctness recorded; actual backend must still be evidenced",
    }


def _failure(category: str, reason: str) -> Dict[str, Any]:
    if category not in PROBE_FAILURE_CATEGORIES:
        raise ConfigError(f"unknown probe failure category {category!r}")
    return {
        "status": "FAIL",
        "failure_category": category,
        "reason": reason,
        "verified": False,
        "device_conclusion_allowed": category not in DEVICE_INDEPENDENT_FAILURES,
    }


# ── evidence ──────────────────────────────────────────────────────────────


@dataclass
class CapabilityEvidence:
    """One capability evidence row (§10 minimum fields)."""

    evidence_id: str
    platform_instance_id: str
    feature_id: str
    declared_status: str = ""
    declared_source_id: str = ""
    discovered_status: str = ""
    discovery_artifact_id: str = ""
    verified_status: str = ""
    benchmarked_status: str = ""
    probe_spec_hash: str = ""
    command_or_api: str = ""
    input_hash: str = ""
    requested_backend: str = ""
    actual_backend: str = ""
    build_status: str = ""
    load_status: str = ""
    execution_status: str = ""
    sync_status: str = ""
    correctness_status: str = ""
    error_metrics: Mapping[str, Any] = field(default_factory=dict)
    exit_code: int = 0
    signal: str = ""
    timeout: bool = False
    failure_category: str = ""
    reason: str = ""
    versions: Mapping[str, str] = field(default_factory=dict)
    started_at: str = ""
    ended_at: str = ""
    valid_until: str = ""
    invalidation_key: str = ""
    artifact_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.evidence_id:
            self.evidence_id = stable_id(
                "capev",
                {
                    "platform": self.platform_instance_id,
                    "feature": self.feature_id,
                    "probe": self.probe_spec_hash,
                    "input": self.input_hash,
                },
            )

    def level(self) -> str:
        if self.benchmarked_status == "yes":
            return CAPABILITY_BENCHMARKED
        if self.verified_status == "yes":
            return CAPABILITY_VERIFIED
        if self.discovered_status == "yes":
            return CAPABILITY_DISCOVERED
        if self.declared_status == "yes":
            return CAPABILITY_DECLARED
        return "NONE"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.feature_id not in FEATURE_BY_ID:
            problems.append(f"unknown feature {self.feature_id!r}")
        if self.verified_status == "yes":
            for name in ("execution_status", "sync_status", "correctness_status"):
                if getattr(self, name) != "pass":
                    problems.append(
                        f"VERIFIED requires {name}=pass (got {getattr(self, name)!r}): "
                        "a compile or allocation is not an execution proof"
                    )
            if not self.actual_backend:
                problems.append("VERIFIED requires the actual backend (requested is not enough)")
            if not self.invalidation_key:
                problems.append("VERIFIED requires an invalidation key (versions/permissions)")
        if self.declared_status == "yes" and not self.declared_source_id:
            problems.append("DECLARED without a source id (URL/version/timestamp) is invalid")
        if self.failure_category and self.failure_category not in PROBE_FAILURE_CATEGORIES:
            problems.append(f"unknown failure category {self.failure_category!r}")
        if self.requested_backend and self.actual_backend and self.requested_backend != self.actual_backend:
            if not self.reason:
                problems.append(
                    "requested != actual requires a reason; a silent fallback may not verify the "
                    "requested feature"
                )
        return problems

    @property
    def fallback_only(self) -> bool:
        return bool(
            self.requested_backend
            and self.actual_backend
            and self.requested_backend != self.actual_backend
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "platform_instance_id": self.platform_instance_id,
            "feature_id": self.feature_id,
            "declared_status": self.declared_status,
            "discovered_status": self.discovered_status,
            "verified_status": self.verified_status,
            "benchmarked_status": self.benchmarked_status,
            "probe_spec_hash": self.probe_spec_hash,
            "input_hash": self.input_hash,
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "correctness_status": self.correctness_status,
            "failure_category": self.failure_category,
            "reason": self.reason,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "valid_until": self.valid_until,
            "invalidation_key": self.invalidation_key,
            "artifact_refs": list(self.artifact_refs),
            "level": self.level(),
        }


def evidence_upgrade_rule(evidence: CapabilityEvidence) -> Dict[str, Any]:
    """What level the evidence may claim, and why it may not go higher."""
    problems = evidence.validate()
    if problems:
        return {"level": "NONE", "allowed": False, "problems": problems, "reason": problems[0]}
    if evidence.discovered_status != "yes":
        return {
            "level": CAPABILITY_DECLARED if evidence.declared_status == "yes" else "NONE",
            "allowed": False,
            "reason": "discovery has not been recorded: an enumerated or executed proof is required",
        }
    if evidence.verified_status != "yes":
        return {
            "level": CAPABILITY_DISCOVERED,
            "allowed": False,
            "reason": "execute+sync+correctness+actual backend are required before VERIFIED",
        }
    if evidence.fallback_only:
        return {
            "level": CAPABILITY_DISCOVERED,
            "allowed": False,
            "reason": "executed through a fallback: the requested feature is fallback-capable, not verified",
        }
    return {
        "level": CAPABILITY_VERIFIED,
        "allowed": True,
        "reason": "",
        "benchmarked_upgrade": "BENCHMARKED only after E12-03 runs the formal workload",
    }


class CapabilityMatrix:
    """The per-instance capability matrix; unsupported is never inferred."""

    def __init__(self) -> None:
        self._rows: Dict[Tuple[str, str], CapabilityEvidence] = {}

    def record(self, evidence: CapabilityEvidence) -> str:
        problems = evidence.validate()
        if problems:
            raise ConfigError("invalid capability evidence: " + "; ".join(problems))
        key = (evidence.platform_instance_id, evidence.feature_id)
        existing = self._rows.get(key)
        if existing is not None:
            old_level = existing.level()
            new_level = evidence.level()
            if old_level != "NONE" and LEVEL_ORDER.get(new_level, -1) < LEVEL_ORDER[old_level]:
                raise ConfigError(
                    f"refusing to overwrite evidence for {key} with a weaker record "
                    f"({old_level} -> {new_level}): re-probe instead of downgrading silently"
                )
        self._rows[key] = evidence
        return evidence.evidence_id

    def get(self, platform_instance_id: str, feature_id: str) -> Optional[CapabilityEvidence]:
        return self._rows.get((platform_instance_id, feature_id))

    def status_for(self, platform_instance_id: str, feature_id: str) -> str:
        row = self.get(platform_instance_id, feature_id)
        return row.level() if row else "NONE"

    def supported(self, platform_instance_id: str, feature_id: str) -> bool:
        return self.status_for(platform_instance_id, feature_id) in SUPPORTING_LEVELS

    def as_rows(self) -> List[Dict[str, Any]]:
        return [self._rows[key].as_dict() for key in sorted(self._rows)]

    def evidence_gaps(self, *, platform_instance_id: str, required: Sequence[str]) -> List[Dict[str, Any]]:
        gaps: List[Dict[str, Any]] = []
        for name in required:
            level = self.status_for(platform_instance_id, name)
            if level in SUPPORTING_LEVELS:
                continue
            row = self.get(platform_instance_id, name)
            gaps.append(
                {
                    "platform_instance_id": platform_instance_id,
                    "feature_id": name,
                    "level": level,
                    "failure_category": row.failure_category if row else "",
                    "reason": row.reason if row else "no probe evidence recorded",
                }
            )
        return gaps


def dependency_satisfied(
    matrix: CapabilityMatrix, *, platform_instance_id: str, feature_id: str
) -> Dict[str, Any]:
    """A dependency that is not VERIFIED blocks the dependent feature."""
    row = feature(feature_id)
    missing = [
        dependency
        for dependency in row.depends_on
        if matrix.status_for(platform_instance_id, dependency) not in SUPPORTING_LEVELS
    ]
    return {
        "feature_id": feature_id,
        "depends_on": list(row.depends_on),
        "unsatisfied": sorted(missing),
        "satisfied": not missing,
    }


# ── invalidation ──────────────────────────────────────────────────────────


def invalidate(
    rows: Sequence[CapabilityEvidence],
    triggers: Sequence[str],
    *,
    whitelist: Mapping[str, str] = INVALIDATION_WHITELIST,
) -> Tuple[Dict[str, Any], ...]:
    """Which evidence rows expire given the observed environment changes."""
    unknown = [
        name
        for name in triggers
        if name not in INVALIDATION_TRIGGERS and name not in whitelist
    ]
    if unknown:
        raise ConfigError(
            f"undeclared invalidation trigger(s) {unknown}: an unrelated change needs a whitelist reason"
        )
    effective = [name for name in triggers if name in INVALIDATION_TRIGGERS]
    expired: List[Dict[str, Any]] = []
    if not effective:
        return ()
    fields = sorted({TRIGGER_FIELD[name] for name in effective})
    for row in rows:
        if not row.invalidation_key:
            expired.append(
                {
                    "evidence_id": row.evidence_id,
                    "trigger": "missing_invalidation_key",
                    "reason": "no key to compare",
                }
            )
            continue
        parts = row.invalidation_key.split("|")
        hit = sorted({name for name in fields if any(part.startswith(f"{name}=") for part in parts)})
        if hit:
            expired.append(
                {
                    "evidence_id": row.evidence_id,
                    "platform_instance_id": row.platform_instance_id,
                    "feature_id": row.feature_id,
                    "trigger": ",".join(hit),
                    "reason": "environment changed after the probe: re-probe before use",
                }
            )
    return tuple(expired)


def invalidation_key(versions: Mapping[str, str], *, arch: str, permission: str) -> str:
    """Stable key of the environment facts an evidence row depends on.

    The key is a readable ``field=hash`` list so that an invalidation trigger can
    be matched to *which* fact changed, rather than only reporting
    "something changed".
    """
    fields = {
        "driver": versions.get("driver", ""),
        "runtime": versions.get("runtime", ""),
        "framework": versions.get("framework", ""),
        "container_image": versions.get("container_image", ""),
        "feature_implementation": versions.get("feature_implementation", ""),
        "arch": arch,
        "permission": permission,
        "host": versions.get("host", ""),
    }
    return "|".join(
        f"{name}={sha256_text(str(fields[name]))[:16]}" for name in sorted(fields)
    )


def permission_change_detected(old_permission: str, new_permission: str) -> bool:
    return old_permission != new_permission


# ── coverage join with E12-01 ─────────────────────────────────────────────


def coverage_join(
    requirements_by_candidate: Mapping[str, Mapping[str, Sequence[str]]],
    matrix: CapabilityMatrix,
    *,
    platform_by_candidate: Mapping[str, str],
) -> Tuple[Dict[str, Any], ...]:
    """Per comparison cell: runnable / blocked / not_applicable + exact gaps."""
    rows: List[Dict[str, Any]] = []
    for candidate_id, per_layer in sorted(requirements_by_candidate.items()):
        platform = platform_by_candidate.get(candidate_id, "")
        for layer, required in sorted(per_layer.items()):
            if not required:
                rows.append(
                    {
                        "comparison_group_id": "",
                        "candidate_id": candidate_id,
                        "layer": layer,
                        "required_feature_ids": [],
                        "missing_feature_ids": [],
                        "join_status": "not_applicable",
                        "reason": "no capability requirements: layer is NOT_APPLICABLE_CAPABILITY",
                    }
                )
                continue
            gaps = matrix.evidence_gaps(platform_instance_id=platform, required=required)
            rows.append(
                {
                    "comparison_group_id": "",
                    "candidate_id": candidate_id,
                    "layer": layer,
                    "required_feature_ids": list(required),
                    "missing_feature_ids": [row["feature_id"] for row in gaps],
                    "join_status": "runnable" if not gaps else "blocked",
                    "reason": "" if not gaps else "; ".join(
                        f"{row['feature_id']}:{row['reason']}" for row in gaps
                    ),
                }
            )
    return tuple(rows)


# ── negative probes ───────────────────────────────────────────────────────


def negative_probe_cases() -> Tuple[Dict[str, Any], ...]:
    """Fault injections whose classification must stay distinguishable."""
    cases = (
        ("hidden_device", "MISSING_COMPONENT"),
        ("removed_library", "MISSING_COMPONENT"),
        ("wrong_arch_binary", "LOAD_FAILED"),
        ("no_permission", "PERMISSION_DENIED"),
        ("timeout", "TIMEOUT"),
        ("device_oom", "RESOURCE_UNAVAILABLE"),
        ("bad_input", "EXECUTION_FAILED"),
        ("wrong_reference", "NUMERICAL_MISMATCH"),
    )
    return tuple(
        {
            "case_id": name,
            "injection": name,
            "expected_category": category,
            "expected_verdict": "FAIL",
        }
        for name, category in cases
    )


def evaluate_negative_probes(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Every injected fault must be classified as expected, not as UNKNOWN."""
    rows: List[Dict[str, Any]] = []
    for row in results:
        expected = str(row.get("expected_category", ""))
        observed = str(row.get("observed_category", ""))
        rows.append(
            {
                "case_id": str(row.get("case_id", "")),
                "expected_category": expected,
                "observed_category": observed,
                "detected": expected == observed,
                "generic_unknown": observed == "UNKNOWN",
                "artifact_refs": list(row.get("artifact_refs", ())),
            }
        )
    return {
        "rows": rows,
        "cases": len(rows),
        "detected": sum(1 for row in rows if row["detected"]),
        "generic_unknown": sum(1 for row in rows if row["generic_unknown"]),
        "ok": bool(rows) and all(row["detected"] for row in rows),
    }


def probe_spec(
    feature_id: str,
    *,
    harness_hash: str,
    command_or_api: str,
    inputs: Mapping[str, Any],
    timeout_s: float = 60.0,
    resource_limits: Optional[Mapping[str, Any]] = None,
    reference_impl: str = "",
) -> Dict[str, Any]:
    """Frozen probe spec; the hash is what evidence binds to."""
    payload = {
        "feature_id": feature_id,
        "harness_hash": harness_hash,
        "command_or_api": command_or_api,
        "inputs": dict(sorted(inputs.items())),
        "timeout_s": timeout_s,
        "resource_limits": dict(sorted((resource_limits or {}).items())),
        "reference_impl": reference_impl,
    }
    from hqsb.evaluation.identity import canonical_json

    spec_hash = sha256_text(canonical_json(payload))
    return {
        "probe_spec_id": stable_id("probe", payload),
        "probe_spec_hash": spec_hash,
        **payload,
    }


def probe_plan(feature_ids: Sequence[str], *, harness_hash: str) -> Tuple[Dict[str, Any], ...]:
    """One spec per requested feature, in dependency order."""
    ordered = dependency_order(tuple(set(feature_ids) | {dep for name in feature_ids for dep in feature(name).depends_on}))
    rows: List[Dict[str, Any]] = []
    for name in ordered:
        row = feature(name)
        rows.append(
            probe_spec(
                name,
                harness_hash=harness_hash,
                command_or_api=f"probe:{name}",
                inputs={"success_criteria": row.success_criteria},
                reference_impl="frozen reference implementation (hash pinned by the harness)",
            )
        )
    return tuple(rows)


def needs_probe(level: str, *, required: str = CAPABILITY_VERIFIED) -> bool:
    return not meets(level, required)


def table_schemas() -> Mapping[str, Tuple[str, ...]]:
    """The E12-02 tables a run must be able to write."""
    return {
        name: TABLE_SCHEMAS[name]
        for name in sorted(TABLE_SCHEMAS)
        if name.startswith("e12_02.")
    }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only callability check; never an experiment."""
    matrix = CapabilityMatrix()
    verified = CapabilityEvidence(
        evidence_id="",
        platform_instance_id="plat_demo",
        feature_id="bf16_storage",
        declared_status="yes",
        declared_source_id="vendor-doc#bf16",
        discovered_status="yes",
        verified_status="yes",
        probe_spec_hash="a" * 64,
        input_hash="b" * 64,
        requested_backend="cuda",
        actual_backend="cuda",
        build_status="pass",
        load_status="pass",
        execution_status="pass",
        sync_status="pass",
        correctness_status="pass",
        invalidation_key=invalidation_key({"driver": "d1", "runtime": "r1"}, arch="sm_90", permission="user"),
    )
    matrix.record(verified)
    fallback = CapabilityEvidence(
        evidence_id="",
        platform_instance_id="plat_demo",
        feature_id="bf16_gemm_native",
        discovered_status="yes",
        requested_backend="cuda_tensor_core",
        actual_backend="cuda_fallback",
        reason="guard refused the native path",
        invalidation_key=invalidation_key({"driver": "d1"}, arch="sm_90", permission="user"),
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "registry_features": len(FEATURE_REGISTRY),
        "registry_problems": registry_problems(),
        "supported_after_probe": matrix.supported("plat_demo", "bf16_storage"),
        "unprobed_is_not_supported": matrix.supported("plat_demo", "fp8_gemm_native") is False,
        "memory_failure_is_not_unsupported": classify_probe_outcome(component_present=False)[
            "failure_category"
        ]
        == "MISSING_COMPONENT",
        "permission_is_distinct": classify_probe_outcome(permission_denied=True)["failure_category"]
        == "PERMISSION_DENIED",
        "numerical_is_distinct": classify_probe_outcome(
            build_ok=True, load_ok=True, execution_ok=True, sync_ok=True, numerical_ok=False
        )["failure_category"]
        == "NUMERICAL_MISMATCH",
        "fallback_does_not_verify_requested": evidence_upgrade_rule(fallback)["level"]
        == CAPABILITY_DISCOVERED,
        "verified_upgrade_allowed": evidence_upgrade_rule(verified)["allowed"],
    }


PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "读取 E12-01 capability requirements", ("candidates:capability_requirements", "capability:feature")),
    (2, "定义 feature_id registry", ("capability:FEATURE_REGISTRY", "capability:Feature", "capability:FEATURE_LAYERS")),
    (3, "冻结 probe harness", ("platform:probe_harness", "platform:ProbeHarness")),
    (4, "建立平台实例身份", ("platform:PlatformIdentity", "platform:PLATFORM_IDENTITY_FIELDS")),
    (5, "采集 OS/host/container", ("platform:SoftwareStack", "platform:SOFTWARE_STACK_FIELDS")),
    (6, "采集厂商软件栈", ("platform:SoftwareStack.stack_id", "capability:invalidation_key")),
    (7, "保存声明来源", ("capability:CapabilityEvidence.declared_source_id", "records:TABLE_SCHEMAS")),
    (8, "运行设备发现 API", ("platform:identity_conflicts", "capability:classify_probe_outcome")),
    (9, "核对多来源身份", ("platform:identity_conflicts", "platform:PlatformIdentity.usable")),
    (10, "探测基本 allocation/copy", ("capability:classify_probe_outcome", "capability:CapabilityEvidence")),
    (11, "探测 elementwise/reduction", ("capability:Feature.success_criteria", "capability:CapabilityMatrix.record")),
    (12, "探测矩阵路径", ("capability:Feature.depends_on", "capability:dependency_satisfied")),
    (13, "探测低比特路径", ("capability:FEATURE_REGISTRY", "capability:evidence_upgrade_rule")),
    (14, "探测 attention/KV", ("capability:feature", "capability:probe_spec")),
    (15, "探测自定义 kernel toolchain", ("capability:probe_plan", "capability:probe_spec")),
    (16, "探测 AOT/JIT 与 arch target", ("capability:classify_probe_outcome", "capability:CapabilityEvidence.actual_backend")),
    (17, "探测 graph/compiler integration", ("capability:CapabilityMatrix.status_for", "capability:meets")),
    (18, "探测模型加载", ("capability:CapabilityMatrix.get", "capability:CapabilityEvidence.artifact_refs")),
    (19, "探测最小 model-core", ("capability:CapabilityEvidence.correctness_status", "capability:evidence_upgrade_rule")),
    (20, "探测服务启动与请求", ("capability:CapabilityMatrix", "capability:CapabilityEvidence")),
    (21, "探测 scheduler 特性", ("capability:feature", "capability:CapabilityMatrix.supported")),
    (22, "探测容量边界元数据", ("capability:CapabilityEvidence", "capability:classify_probe_outcome")),
    (23, "探测拓扑与 P2P", ("capability:dependency_satisfied", "capability:CapabilityMatrix.evidence_gaps")),
    (24, "探测 collective", ("capability:CapabilityMatrix.record", "capability:classify_probe_outcome")),
    (25, "探测分布式 model prerequisite", ("capability:dependency_order", "capability:CapabilityMatrix.status_for")),
    (26, "探测 profiler/tooling", ("capability:CapabilityEvidence", "platform:TelemetryFieldAdapter")),
    (27, "探测功率/能量字段", ("platform:canonicalize_telemetry_field", "platform:energy_capability")),
    (28, "探测温度/时钟/throttle/health", ("platform:ADAPTER_CATALOG", "platform:TelemetryFieldAdapter.validate")),
    (29, "验证 idle→load 响应", ("platform:telemetry_smoke_check", "platform:CANONICAL_TELEMETRY_FIELDS")),
    (30, "运行重复性 probe", ("capability:CapabilityEvidence.invalidation_key", "capability:probe_spec")),
    (31, "执行负向故障注入", ("capability:negative_probe_cases", "capability:evaluate_negative_probes")),
    (32, "核对 requested 与 actual backend", ("capability:CapabilityEvidence.fallback_only", "capability:evidence_upgrade_rule")),
    (33, "构建证据升级规则", ("capability:evidence_upgrade_rule", "capability:CAPABILITY_LEVELS")),
    (34, "构建失效规则", ("capability:invalidate", "capability:INVALIDATION_TRIGGERS")),
    (35, "与 E12-01 做 coverage join", ("capability:coverage_join", "capability:CapabilityMatrix.evidence_gaps")),
    (36, "形成 capability acceptance report", ("capability:table_schemas", "campaign:AcceptanceDecision")),
)
