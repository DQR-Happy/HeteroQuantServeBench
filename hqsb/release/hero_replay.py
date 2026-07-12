"""E15-03 — target GPU/NPU hero-story replay.

Protocol: ``docs/stage_experiments/details/S15/E15-03_accelerator_hero_story_replay.md``
(45 steps).

This is the technical reproduction gate: the frozen hero story is replayed in a
*new* environment, and the question is not "did the number match to the
nanosecond" but "do identity, correctness, actual path, effect and mechanism
still hold, and if not, is the claim updated rather than the figure".

The module implements:

* :class:`HeroStoryContract` — the narrative nodes, claim ids, estimand, minimum
  meaningful effect, guard band, quality gate and failure policy, frozen before
  the first measurement (steps 1–3);
* :class:`EnvironmentFingerprint`, :class:`EnvironmentDiff`,
  :func:`classify_comparability` — exact/compatible/conditional/incomparable,
  decided *before* the results are seen (steps 4–6, 36);
* :class:`DeviceHealth`, :class:`ProfilerAvailability` — the "is this a valid
  measurement environment at all" checks (steps 7–8, 24);
* :class:`ArtifactIdentity`, :func:`check_binary_compatibility`,
  :func:`check_workload_tokens`, :func:`check_operator_spec` — the identity and
  compatibility gates (steps 9–13, 37);
* :class:`CapabilityPreflight` — per workload point: supported or not, with the
  selected implementation and the reason; unsupported points are *reported*, not
  silently dropped (step 14);
* :class:`CorrectnessMatrix` — operator / block / model / service layers with the
  per-layer metrics and the rule that quality stops performance aggregation
  (steps 15–18);
* :class:`ActualPathEvidence` — dispatcher log, trace events, kernel/collective
  names, binary hash and profiler activity, plus the **negative control** that
  proves the instrumentation can tell candidate from fallback (steps 19–20, 38);
* :class:`ConfirmatoryPlan` — strata, cold/hot boundary, ABBA interleaving and
  the stability window (steps 21–24);
* :class:`AmdahlModel` — ``S_max = 1 / ((1 - p) + p / s)`` with integration
  overhead and the error decomposition of step 35;
* :class:`BlockedEstimate` and :func:`compare_with_release` — the paired/blocked
  effect with CI and the guard-band verdict (steps 33–34);
* :func:`replay_verdict` — the §10 consistency matrix, returning one of
  ``CONFIRMED`` … ``MECHANISM_NOT_REPRODUCED``;
* :func:`claim_actions_for` and :func:`check_stop_rules` — step 42 and §10.2: a
  contradicted result updates the claim, it does not delete it.

Nothing here runs a benchmark, loads a model or touches a device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.release import records as rec
from hqsb.release.identity import SCHEMA_PREFIX, canonical_digest, is_digest

#: The five narrative nodes the hero story must be able to show.
NARRATIVE_NODES: Tuple[str, ...] = (
    "reference_baseline",
    "profile_bottleneck",
    "candidate_implementation",
    "model_integration",
    "measured_effect",
)

#: The minimum profiling reconciliation table (``E15-03`` §10.1).
PROFILE_LAYERS: Tuple[str, ...] = ("operator", "block", "model", "service", "resource")

#: Layers that may be marked ``N/A_BY_CLAIM_SCOPE`` when the claim does not reach them.
OPTIONAL_PROFILE_LAYERS: Tuple[str, ...] = ("block", "service")

#: Comparability classes (step 5).
COMPARABILITY_CLASSES: Tuple[str, ...] = rec.COMPARABILITY_CLASSES

#: Verdicts of the §10 consistency matrix.
REPLAY_VERDICTS: Tuple[str, ...] = rec.REPLAY_VERDICTS

#: Amdahl prediction error components (step 35).
AMDAHL_ERROR_COMPONENTS: Tuple[str, ...] = (
    "hotspot_share",
    "dispatch",
    "fusion",
    "memory",
    "compile",
    "cache",
    "queue",
    "communication",
    "measurement_noise",
)

#: Proof kinds an actual-path record can carry (§3.3 / step 19).
ACTUAL_PATH_PROOFS: Tuple[str, ...] = (
    "dispatcher_record",
    "trace_event",
    "kernel_name",
    "collective_name",
    "binary_hash",
    "profiler_activity",
    "fallback_reason",
)

#: Stop and protection rules (§10.2).
STOP_RULES: Tuple[Mapping[str, str], ...] = (
    {"rule": "correctness_or_quality_gate_failed", "action": "stop confirmatory performance aggregation"},
    {"rule": "device_errors_or_throttle", "action": "abort and record the environment state"},
    {"rule": "oom", "action": "do not silently shorten the sequence; a new workload/claim version is required"},
    {"rule": "profiler_disturbance", "action": "exclude the profiled window from performance samples"},
    {"rule": "budget_exceeded", "action": "report uncertainty; do not keep running until significant"},
)


@dataclass
class HeroStoryContract:
    """The frozen contract of steps 1–2."""

    candidate_id: str
    hero_claim_id: str
    auxiliary_claim_ids: Tuple[str, ...] = ()
    narrative_nodes: Tuple[str, ...] = NARRATIVE_NODES
    primary_estimand: str = ""
    minimum_meaningful_effect: Optional[float] = None
    guard_band: Optional[float] = None
    quality_gate: str = ""
    failure_policy: str = ""
    intended_difference: str = ""

    schema_version = f"{SCHEMA_PREFIX}.hero-story-contract.v1"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.candidate_id or not self.hero_claim_id:
            findings.append("HeroStoryContract: candidate_id and hero_claim_id are required")
        if len(self.auxiliary_claim_ids) != 2:
            findings.append(
                "HeroStoryContract: the protocol asks for exactly two auxiliary claims, frozen with the hero one"
            )
        missing_nodes = [node for node in NARRATIVE_NODES if node not in self.narrative_nodes]
        if missing_nodes:
            findings.append(f"HeroStoryContract: narrative nodes missing: {', '.join(missing_nodes)}")
        if not self.primary_estimand:
            findings.append("HeroStoryContract: the primary estimand must be written before the replay")
        if self.minimum_meaningful_effect is None:
            findings.append("HeroStoryContract: the minimum meaningful effect must be pre-registered")
        if self.guard_band is None or self.guard_band <= 0:
            findings.append(
                "HeroStoryContract: the guard band must be pre-registered and positive "
                "(widening it after seeing the result is the failure mode E15-03 forbids)"
            )
        if not self.quality_gate or not self.failure_policy:
            findings.append("HeroStoryContract: the quality gate and the failure policy are required")
        if not self.intended_difference:
            findings.append("HeroStoryContract: the unique intended difference from the baseline is required")
        return findings


@dataclass
class EnvironmentFingerprint:
    """Accelerator environment identity (step 6)."""

    environment_id: str
    device_sku: str = ""
    device_count: int = 0
    memory_bytes_per_device: int = 0
    topology: str = ""
    driver_version: str = ""
    runtime_version: str = ""
    firmware: str = ""
    compiler: str = ""
    framework: str = ""
    kernel_libraries: Tuple[str, ...] = ()
    power_mode: str = ""
    clock_mode: str = ""
    container_digest: str = ""
    reused_author_state: bool = False

    REQUIRED: Tuple[str, ...] = (
        "device_sku",
        "driver_version",
        "runtime_version",
        "framework",
        "power_mode",
    )

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in self.REQUIRED:
            if not getattr(self, name):
                findings.append(f"EnvironmentFingerprint: {name} is required (same SKU is not the same environment)")
        if self.device_count <= 0:
            findings.append("EnvironmentFingerprint: device_count must be positive")
        if self.reused_author_state:
            findings.append(
                "EnvironmentFingerprint: the replay consumed author state (editable install/cache/uncommitted file); "
                "this is not a new environment (step 4)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "device_sku": self.device_sku,
            "device_count": self.device_count,
            "memory_bytes_per_device": self.memory_bytes_per_device,
            "topology": self.topology,
            "driver_version": self.driver_version,
            "runtime_version": self.runtime_version,
            "firmware": self.firmware,
            "compiler": self.compiler,
            "framework": self.framework,
            "kernel_libraries": list(self.kernel_libraries),
            "power_mode": self.power_mode,
            "clock_mode": self.clock_mode,
            "container_digest": self.container_digest,
            "reused_author_state": self.reused_author_state,
        }


def classify_comparability(original: EnvironmentFingerprint, replay: EnvironmentFingerprint) -> str:
    """exact / compatible / conditional / incomparable, decided before results (step 5).

    The classification names *which* fields differ, so the comparison strength is
    a property of the environments rather than of the numbers they produced.
    """
    problems = list(original.problems()) + list(replay.problems())
    if problems:
        raise ConfigError("cannot classify comparability of an incomplete environment: " + "; ".join(problems[:3]))
    same = all(
        getattr(original, name) == getattr(replay, name)
        for name in ("device_sku", "driver_version", "runtime_version", "framework", "power_mode", "clock_mode")
    )
    if same and original.compiler == replay.compiler and original.container_digest == replay.container_digest:
        return "exact"
    soft_fields = ("driver_version", "runtime_version", "framework", "compiler", "container_digest")
    soft_diff = [name for name in soft_fields if getattr(original, name) != getattr(replay, name)]
    hard_diff = [
        name
        for name in ("device_sku", "power_mode", "clock_mode")
        if getattr(original, name) != getattr(replay, name)
    ]
    if not hard_diff and soft_diff:
        return "compatible"
    if hard_diff and len(hard_diff) <= 2:
        return "conditional"
    return "incomparable"


def environment_diff(original: EnvironmentFingerprint, replay: EnvironmentFingerprint) -> Dict[str, Any]:
    """Field-by-field diff with an explicit comparability class (step 36)."""
    klass = classify_comparability(original, replay)
    differences: Dict[str, Dict[str, Any]] = {}
    for name in original.as_dict():
        left = getattr(original, name)
        right = getattr(replay, name)
        if left != right:
            differences[name] = {"original": left, "replay": right}
    return {
        "comparability": klass,
        "differences": differences,
        "note": "只对已有证据的因素做因果判断；相关变化不能用口头归因",
    }


@dataclass
class DeviceHealth:
    """Device health and isolation before measuring (steps 7, 24)."""

    error_counts: Mapping[str, int] = field(default_factory=dict)
    temperature_c: Optional[float] = None
    clock_mhz: Optional[float] = None
    power_limit_w: Optional[float] = None
    persistence_mode: bool = False
    mig_or_virtualized: bool = False
    foreign_processes: Tuple[str, ...] = ()
    shared_tenant: bool = False
    idle_achieved: bool = False
    stability_window_ok: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        errors = {name: count for name, count in self.error_counts.items() if count}
        if errors:
            findings.append(f"device error counters are non-zero: {errors!r}")
        if any(count is None for count in self.error_counts.values()):
            findings.append("device error counters were not read (unknown is not zero)")
        if self.temperature_c is None or self.clock_mhz is None:
            findings.append("temperature and clock must be recorded before the run")
        if self.foreign_processes:
            findings.append(f"other processes share the device: {self.foreign_processes!r}")
        if self.shared_tenant:
            findings.append("the device is shared with another tenant; results are not comparable")
        if not self.stability_window_ok:
            findings.append("the pre-run stability window was not confirmed (throttling can reverse the direction)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "error_counts": {key: int(self.error_counts[key]) for key in sorted(self.error_counts)},
            "temperature_c": self.temperature_c,
            "clock_mhz": self.clock_mhz,
            "power_limit_w": self.power_limit_w,
            "persistence_mode": self.persistence_mode,
            "mig_or_virtualized": self.mig_or_virtualized,
            "foreign_processes": list(self.foreign_processes),
            "shared_tenant": self.shared_tenant,
            "idle_achieved": self.idle_achieved,
            "stability_window_ok": self.stability_window_ok,
        }


@dataclass
class ProfilerAvailability:
    """Profiler readiness (step 8): without it there is no actual-path proof."""

    system_profiler: str = ""
    kernel_profiler: str = ""
    permissions_ok: bool = False
    version: str = ""
    fallback_plan: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.system_profiler or not self.kernel_profiler:
            if not self.fallback_plan:
                findings.append(
                    "no system/kernel profiler is available and no agreed degradation was recorded "
                    "(structurally this is BLOCKED, not a silent downgrade)"
                )
        if not self.permissions_ok:
            findings.append("profiler permissions are not confirmed; a run that discovers this late wastes the block")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "system_profiler": self.system_profiler,
            "kernel_profiler": self.kernel_profiler,
            "permissions_ok": self.permissions_ok,
            "version": self.version,
            "fallback_plan": self.fallback_plan,
        }


@dataclass
class ArtifactIdentity:
    """Model / workload / operator / binary identity (steps 9–13)."""

    model_artifact_id: str = ""
    model_manifest_sha256: str = ""
    tokenizer_sha256: str = ""
    workload_spec_id: str = ""
    token_sha256: str = ""
    operator_spec_id: str = ""
    operator_spec_sha256: str = ""
    binary_sha256: str = ""
    source_commit: str = ""
    release_artifact_sha256: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in (
            "model_artifact_id",
            "model_manifest_sha256",
            "workload_spec_id",
            "token_sha256",
            "operator_spec_id",
            "binary_sha256",
            "source_commit",
        ):
            value = getattr(self, name)
            if not value:
                findings.append(f"ArtifactIdentity: {name} is required (a config name is not an identity)")
        for name in ("model_manifest_sha256", "token_sha256", "binary_sha256", "operator_spec_sha256", "release_artifact_sha256"):
            value = getattr(self, name)
            if value and not is_digest(value):
                findings.append(f"ArtifactIdentity: {name} must be sha256:<hex>")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def check_binary_compatibility(*, requires: Mapping[str, Any], provides: Mapping[str, Any]) -> List[str]:
    """Optional dependency / ABI / compute-capability compatibility (step 10)."""
    findings: List[str] = []
    if requires.get("compute_capability") and provides.get("compute_capability"):
        if requires["compute_capability"] != provides["compute_capability"]:
            findings.append(
                "compute capability mismatch: "
                f"binary needs {requires['compute_capability']}, device provides {provides['compute_capability']}"
            )
    if requires.get("abi") and provides.get("abi") and requires["abi"] != provides["abi"]:
        findings.append(f"ABI mismatch: {requires['abi']} vs {provides['abi']}")
    for name in ("driver_min", "runtime_min"):
        required = requires.get(name)
        available = provides.get(name)
        if required and available and _version_tuple(available) < _version_tuple(required):
            findings.append(f"{name} not satisfied: needs {required}, has {available}")
    for extra in requires.get("extras", ()):
        if extra not in provides.get("extras", ()):
            findings.append(f"optional extra {extra!r} is not installed (feature-scoped failure expected)")
    return findings


def _version_tuple(value: Any) -> Tuple[int, ...]:
    parts: List[int] = []
    for chunk in str(value).replace("-", ".").split("."):
        digits = "".join(char for char in chunk if char.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_workload_tokens(*, declared: Mapping[str, Any], observed: Mapping[str, Any]) -> List[str]:
    """Tokenisation and workload identity (step 12)."""
    findings: List[str] = []
    for name in ("token_sha256", "isl", "osl", "stop_rules", "seed"):
        if name not in declared or name not in observed:
            findings.append(f"workload token check: {name!r} is not recorded on both sides")
            continue
        left = declared[name]
        right = observed[name]
        if isinstance(left, (list, tuple)):
            left = list(left)
        if isinstance(right, (list, tuple)):
            right = list(right)
        if left != right:
            findings.append(f"workload token drift on {name!r}: {left!r} != {right!r}")
    if declared.get("batch") != observed.get("batch") or declared.get("concurrency") != observed.get("concurrency"):
        findings.append("comparing throughput at different batch/concurrency is not a comparison")
    return findings


def check_operator_spec(*, declared: Mapping[str, Any], observed: Mapping[str, Any]) -> List[str]:
    """OperatorSpec / binary identity (step 13)."""
    findings: List[str] = []
    for name in ("equation", "epsilon", "dtype", "layout", "alignment", "stream", "workspace_bytes", "fallback_policy"):
        if name not in declared:
            findings.append(f"operator spec: {name!r} is not declared (an undeclared field cannot be compared)")
            continue
        if name in observed and observed[name] != declared[name]:
            findings.append(f"operator spec drift on {name!r}: {declared[name]!r} != {observed[name]!r}")
    return findings


@dataclass
class CapabilityPreflight:
    """Per-point capability with reasons (step 14)."""

    points: Tuple[Mapping[str, Any], ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.points:
            findings.append("CapabilityPreflight: no workload points were checked")
        for point in self.points:
            if "point_id" not in point or "supported" not in point:
                findings.append("CapabilityPreflight: every point needs an id and a supported flag")
                continue
            if not point["supported"] and not point.get("reason"):
                findings.append(f"CapabilityPreflight({point['point_id']}): an unsupported point needs a reason")
            if point["supported"] and not point.get("selected_implementation"):
                findings.append(
                    f"CapabilityPreflight({point['point_id']}): a supported point must name the selected implementation"
                )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {"points": [dict(point) for point in self.points]}


def evaluate_correctness_matrix(
    *,
    layer: str,
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, float],
    first_mismatch_index: Optional[int] = None,
) -> Dict[str, Any]:
    """One layer of the correctness matrix (steps 16–18).

    ``layer`` is one of ``operator`` / ``block`` / ``model`` / ``service``.  The
    result is a per-layer verdict; the *rule* that a failure here stops the
    performance aggregation is enforced by :func:`replay_verdict`.
    """
    if layer not in ("operator", "block", "model", "service"):
        raise ConfigError(f"unknown correctness layer {layer!r}")
    if not metrics:
        raise ConfigError(f"correctness layer {layer!r} has no metrics (an unmeasured layer is not a pass)")
    findings: List[str] = []
    for name, threshold in sorted(thresholds.items()):
        value = metrics.get(name)
        if value is None:
            findings.append(f"{layer}: metric {name!r} was not measured")
            continue
        if name in ("max_abs_error", "mean_abs_error", "rmse"):
            ok = float(value) <= threshold
        elif name in ("cosine", "token_agreement"):
            ok = float(value) >= threshold
        else:
            ok = float(value) <= threshold
        if not ok:
            findings.append(f"{layer}: {name}={value} breaches the pre-registered threshold {threshold}")
    if first_mismatch_index is not None and first_mismatch_index >= 0 and layer == "model":
        findings.append(f"{layer}: first mismatch at index {first_mismatch_index} (locate the layer, do not average it away)")
    return {"layer": layer, "metrics": dict(sorted(metrics.items())), "findings": findings, "ok": not findings}


@dataclass
class ActualPathEvidence:
    """Proof that the candidate actually ran (steps 19–20, 38)."""

    proofs: Mapping[str, Any] = field(default_factory=dict)
    binary_sha256: str = ""
    negative_control: Optional[Mapping[str, Any]] = None

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.proofs:
            findings.append("ActualPathEvidence: no proof was recorded (a config file naming a backend is not evidence)")
        unknown = [name for name in self.proofs if name not in ACTUAL_PATH_PROOFS]
        if unknown:
            findings.append(f"ActualPathEvidence: unknown proof kinds {', '.join(sorted(unknown))}")
        if self.binary_sha256 and not is_digest(self.binary_sha256):
            findings.append("ActualPathEvidence: binary_sha256 must be sha256:<hex>")
        if self.negative_control is None:
            findings.append(
                "ActualPathEvidence: the forced-fallback negative control is missing; without it the "
                "instrumentation has no proven detection power (step 20)"
            )
        else:
            distinguishable = bool(self.negative_control.get("distinguished"))
            if not distinguishable:
                findings.append("ActualPathEvidence: the negative control did not distinguish candidate from fallback")
        return findings

    def candidate_hit(self) -> bool:
        return any(
            self.proofs.get(name) not in (None, "", False, "reference", "fallback")
            for name in ("dispatcher_record", "kernel_name", "collective_name", "profiler_activity")
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "proofs": {key: self.proofs[key] for key in sorted(self.proofs)},
            "binary_sha256": self.binary_sha256,
            "negative_control": dict(self.negative_control) if self.negative_control else None,
            "candidate_hit": self.candidate_hit(),
        }


@dataclass
class ConfirmatoryPlan:
    """Strata, cold/hot boundary, interleaving and stability (steps 21–24)."""

    strata: Tuple[str, ...] = ()
    adverse_boundary_included: bool = False
    cold_start_measured: bool = False
    steady_state_measured: bool = False
    compile_and_autotune_billed: bool = False
    interleave_order: str = ""
    independent_process_restarts: int = 0
    minimum_repeats: int = 0

    INTERLEAVE_ORDERS: Tuple[str, ...] = ("ABBA", "randomised", "blocked")

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.strata:
            findings.append("ConfirmatoryPlan: confirmatory strata must be frozen, not chosen after the fact")
        if not self.adverse_boundary_included:
            findings.append("ConfirmatoryPlan: at least one unfavourable boundary point must be included (step 21)")
        if not (self.cold_start_measured and self.steady_state_measured):
            findings.append("ConfirmatoryPlan: cold-start and steady-state must be reported separately (step 22)")
        if not self.compile_and_autotune_billed:
            findings.append("ConfirmatoryPlan: compile/autotune cost must be billed explicitly, not hidden (step 22)")
        if self.interleave_order not in self.INTERLEAVE_ORDERS:
            findings.append(f"ConfirmatoryPlan: unknown interleave order {self.interleave_order!r}")
        if self.independent_process_restarts < 1:
            findings.append("ConfirmatoryPlan: baseline/candidate must be interleaved in independent process blocks")
        if self.minimum_repeats < 3:
            findings.append("ConfirmatoryPlan: at least three independent repetitions are required (manual §5.4)")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strata": list(self.strata),
            "adverse_boundary_included": self.adverse_boundary_included,
            "cold_start_measured": self.cold_start_measured,
            "steady_state_measured": self.steady_state_measured,
            "compile_and_autotune_billed": self.compile_and_autotune_billed,
            "interleave_order": self.interleave_order,
            "independent_process_restarts": self.independent_process_restarts,
            "minimum_repeats": self.minimum_repeats,
        }


@dataclass
class AmdahlModel:
    """The Amdahl prediction *and* its error decomposition (steps 31–32, 35)."""

    hotspot_share: float
    local_speedup: float
    integration_overhead: float = 0.0
    measured_model_effect: Optional[float] = None
    error_components: Mapping[str, float] = field(default_factory=dict)
    recomputed_in_new_environment: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not 0.0 <= self.hotspot_share <= 1.0:
            findings.append("AmdahlModel: hotspot share must be in [0, 1]")
        if self.local_speedup <= 0:
            findings.append("AmdahlModel: local speedup must be positive")
        if self.integration_overhead < 0:
            findings.append("AmdahlModel: integration overhead may not be negative")
        if not self.recomputed_in_new_environment:
            findings.append(
                "AmdahlModel: the hotspot share must be recomputed in the new environment, not copied "
                "from the original profile (step 31)"
            )
        unknown = [name for name in self.error_components if name not in AMDAHL_ERROR_COMPONENTS]
        if unknown:
            findings.append(f"AmdahlModel: unknown error components {', '.join(sorted(unknown))}")
        return findings

    def ideal_upper_bound(self) -> float:
        """``S_max = 1 / ((1 - p) + p / s)`` — the ideal upper bound."""
        denominator = (1.0 - self.hotspot_share) + self.hotspot_share / self.local_speedup
        return 1.0 / denominator

    def predicted(self) -> float:
        """The prediction the story must be judged against, overhead included."""
        return max(1.0, self.ideal_upper_bound() - self.integration_overhead)

    def prediction_interval(self) -> Tuple[float, float]:
        predicted = self.predicted()
        return (1.0 + (predicted - 1.0) * 0.5, predicted)

    def residual(self) -> Optional[float]:
        if self.measured_model_effect is None:
            return None
        return float(self.measured_model_effect) - self.predicted()

    def explain_residual(self) -> Dict[str, Any]:
        """Attribute the residual to the registered components (step 35).

        The residual must be *covered* by the named components; "framework
        overhead" without a number is exactly what the protocol refuses.
        """
        residual = self.residual()
        if residual is None:
            return {"residual": None, "explained": 0.0, "unexplained": None, "ok": False}
        explained = float(sum(self.error_components.values()))
        unexplained = residual - explained
        return {
            "residual": residual,
            "explained": explained,
            "unexplained": unexplained,
            "ok": abs(unexplained) <= 0.25 * max(abs(residual), 1e-9),
            "components": {key: self.error_components[key] for key in sorted(self.error_components)},
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hotspot_share": self.hotspot_share,
            "local_speedup": self.local_speedup,
            "integration_overhead": self.integration_overhead,
            "ideal_upper_bound": self.ideal_upper_bound(),
            "predicted": self.predicted(),
            "measured_model_effect": self.measured_model_effect,
            "residual": self.residual(),
            "recomputed_in_new_environment": self.recomputed_in_new_environment,
        }


@dataclass
class BlockedEstimate:
    """A paired/blocked estimate with its interval and unit (step 33)."""

    blocks: Tuple[Mapping[str, float], ...] = ()
    unit: str = "run_block"
    minimum_meaningful_effect: Optional[float] = None

    def problems(self) -> List[str]:
        findings: List[str] = []
        if len(self.blocks) < 3:
            findings.append("BlockedEstimate: fewer than three independent blocks; the interval would be noise")
        if self.unit != "run_block":
            findings.append(
                f"BlockedEstimate: the experimental unit must be the run block, got {self.unit!r} "
                "(requests/tokens are observations inside a block)"
            )
        for block in self.blocks:
            if "baseline" not in block or "candidate" not in block:
                findings.append("BlockedEstimate: every block needs a baseline and a candidate measurement")
                break
        return findings

    def effect(self) -> Dict[str, Any]:
        """Paired difference per block, with mean and a t-style interval."""
        differences = [block["candidate"] - block["baseline"] for block in self.blocks]
        n = len(differences)
        if n == 0:
            return {"point": None, "interval": None, "n": 0}
        mean = sum(differences) / n
        if n > 1:
            variance = sum((value - mean) ** 2 for value in differences) / (n - 1)
            standard_error = math.sqrt(variance / n)
            half_width = 1.96 * standard_error if n >= 3 else float("nan")
            interval: Optional[Tuple[float, float]] = (mean - half_width, mean + half_width) if n >= 3 else None
        else:
            interval = None
        crosses_zero = bool(interval and interval[0] <= 0.0 <= interval[1])
        below_minimum = bool(
            self.minimum_meaningful_effect is not None and abs(mean) < self.minimum_meaningful_effect
        )
        return {
            "point": mean,
            "interval": interval,
            "n": n,
            "crosses_zero": crosses_zero,
            "below_minimum_effect": below_minimum,
            "raw_differences": differences,
        }


def compare_with_release(
    *, contract: HeroStoryContract, effect: Mapping[str, Any], release_effect: float
) -> Dict[str, Any]:
    """Guard-band comparison of the replay against the released number (step 34)."""
    point = effect.get("point")
    if point is None:
        return {"verdict": "INCONCLUSIVE", "reason": "no effect was estimated", "within_guard_band": False}
    delta = float(point) - float(release_effect)
    band = float(contract.guard_band or 0.0)
    same_direction = (point > 1.0 and release_effect > 1.0) or (point < 1.0 and release_effect < 1.0)
    within = abs(delta) <= band
    return {
        "release_effect": release_effect,
        "replay_effect": point,
        "absolute_delta": abs(delta),
        "guard_band": band,
        "within_guard_band": within,
        "same_direction": same_direction,
        "verdict": "consistent" if within else ("direction_consistent" if same_direction else "contradicted"),
    }


def check_stop_rules(
    *, correctness_ok: bool, device_ok: bool, oom: bool, profiler_disturbed: bool, budget_remaining: bool
) -> List[Dict[str, str]]:
    """Which stop rules fire for this run (§10.2)."""
    fired: List[Dict[str, str]] = []
    table = {rule["rule"]: rule["action"] for rule in STOP_RULES}
    if not correctness_ok:
        fired.append({"rule": "correctness_or_quality_gate_failed", "action": table["correctness_or_quality_gate_failed"]})
    if not device_ok:
        fired.append({"rule": "device_errors_or_throttle", "action": table["device_errors_or_throttle"]})
    if oom:
        fired.append({"rule": "oom", "action": table["oom"]})
    if profiler_disturbed:
        fired.append({"rule": "profiler_disturbance", "action": table["profiler_disturbance"]})
    if not budget_remaining:
        fired.append({"rule": "budget_exceeded", "action": table["budget_exceeded"]})
    return fired


def replay_verdict(
    *,
    correctness_ok: bool,
    quality_ok: bool,
    actual_path_ok: bool,
    within_guard_band: bool,
    direction_same: bool,
    mechanism_consistent: bool,
    interval_crosses_zero: bool,
) -> str:
    """The §10 consistency matrix, as a function."""
    if not (correctness_ok and quality_ok):
        return "QUALITY_DISQUALIFIED"
    if not actual_path_ok:
        return "INVALID_PATH"
    if interval_crosses_zero:
        return "INCONCLUSIVE"
    if within_guard_band and mechanism_consistent:
        return "CONFIRMED"
    if direction_same and mechanism_consistent:
        return "CONDITIONAL_CONFIRMED"
    if not direction_same and mechanism_consistent:
        return "CONTRADICTED_OR_SCOPE_CHANGE"
    if not direction_same and not mechanism_consistent:
        return "CONTRADICTED_UNATTRIBUTED"
    if not mechanism_consistent:
        return "MECHANISM_NOT_REPRODUCED"
    return "INCONCLUSIVE"


def claim_actions_for(verdict: str) -> Tuple[str, ...]:
    """What the verdict requires of the claim ledger (step 42, §10)."""
    mapping: Mapping[str, Tuple[str, ...]] = {
        "CONFIRMED": (),
        "CONDITIONAL_CONFIRMED": ("conditional_scope_recorded", "environment_condition_written_into_claim"),
        "INCONCLUSIVE": ("effect_range_widened", "no_speedup_wording"),
        "CONTRADICTED_OR_SCOPE_CHANGE": ("scope_narrowed_or_stale", "claim_revision_recorded"),
        "CONTRADICTED_UNATTRIBUTED": ("claim_retracted_pending_attribution", "figure_withdrawn"),
        "INVALID_PATH": ("actual_path_claim_blocked", "no_speedup_wording"),
        "QUALITY_DISQUALIFIED": ("performance_claim_withdrawn", "quality_finding_opened"),
        "MECHANISM_NOT_REPRODUCED": ("mechanism_claim_removed", "claim_revision_recorded"),
        "BLOCKED": ("claim_unchanged", "blocked_state_recorded"),
    }
    if verdict not in mapping:
        raise ConfigError(f"unknown replay verdict {verdict!r}")
    return mapping[verdict]


def profile_reconciliation_rows(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The §10.1 before/after table, with ``N/A_BY_CLAIM_SCOPE`` allowed only where declared."""
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for record in records:
        layer = str(record.get("layer", ""))
        if layer not in PROFILE_LAYERS:
            problems.append(f"unknown profile layer {layer!r}")
            continue
        status = str(record.get("status", ""))
        if status == "N/A_BY_CLAIM_SCOPE" and layer not in OPTIONAL_PROFILE_LAYERS:
            problems.append(f"layer {layer!r} may not be N/A: the claim reaches it")
        if status not in ("measured", "N/A_BY_CLAIM_SCOPE"):
            problems.append(
                f"layer {layer!r} is left blank; a reached layer without run-id join is EVIDENCE_UNJOINABLE"
            )
        rows.append(dict(record))
    return {"rows": rows, "problems": problems}


def one_click_replay_command(*, experiment_id: str, claim_id: str, environment_id: str) -> str:
    """The auditable one-command entry (step 43).

    "One click" must not hide the stages: the command keeps every stage output and
    failure code visible, which is why it is a *driver* invocation and not a script
    that swallows them.
    """
    return (
        "python3 scripts/release/run_e15.py --experiment "
        f"{experiment_id} --execute --run-id replay_{environment_id}_{claim_id.lower()}"
    )


def freeze_replay(
    *,
    contract: HeroStoryContract,
    environment: EnvironmentFingerprint,
    record: rec.HeroReplayResult,
    health: DeviceHealth,
) -> Dict[str, Any]:
    """Freeze the replay record (step 45)."""
    problems = list(contract.problems()) + list(environment.problems()) + list(health.problems()) + list(record.validate())
    if problems:
        raise ConfigError("refusing to freeze the replay: " + "; ".join(problems[:5]))
    return {
        "schema_version": f"{SCHEMA_PREFIX}.hero-replay.v1",
        "contract": contract.schema_version,
        "environment": environment.as_dict(),
        "record": record.as_dict(),
        "record_digest": canonical_digest(record.as_dict()),
    }


# ── PROTOCOL_STEPS (45) ─────────────────────────────────────────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "选择唯一主 Hero Claim", ("hero_replay.HeroStoryContract", "claims.adjudicate_release_eligibility")),
    (2, "冻结 HeroStoryContract", ("hero_replay.HeroStoryContract", "hero_replay.NARRATIVE_NODES")),
    (3, "冻结原始发布证据", ("identity.content_address_aggregate", "claims.EvidenceGraph")),
    (4, "选择新目标环境", ("hero_replay.EnvironmentFingerprint",)),
    (5, "定义环境可比性类别", ("hero_replay.classify_comparability", "hero_replay.COMPARABILITY_CLASSES")),
    (6, "采集加速器环境指纹", ("hero_replay.EnvironmentFingerprint.as_dict",)),
    (7, "验证设备健康和隔离", ("hero_replay.DeviceHealth",)),
    (8, "验证 profiler 可用性", ("hero_replay.ProfilerAvailability",)),
    (9, "从候选 release 安装", ("records.ReleaseArtifactRecord", "identity.EvidenceRef")),
    (10, "验证可选依赖与二进制兼容", ("hero_replay.check_binary_compatibility",)),
    (11, "获取模型和 tokenizer", ("hero_replay.ArtifactIdentity",)),
    (12, "验证 workload tokens", ("hero_replay.check_workload_tokens",)),
    (13, "验证 OperatorSpec 和 candidate binary", ("hero_replay.check_operator_spec", "hero_replay.ArtifactIdentity")),
    (14, "执行 capability preflight", ("hero_replay.CapabilityPreflight",)),
    (15, "建立 reference 语义基线", ("hero_replay.evaluate_correctness_matrix",)),
    (16, "运行 operator correctness matrix", ("hero_replay.evaluate_correctness_matrix",)),
    (17, "运行 block/model differential correctness", ("hero_replay.evaluate_correctness_matrix",)),
    (18, "运行服务语义门", ("hero_replay.evaluate_correctness_matrix", "records.HeroReplayResult.quality_status")),
    (19, "确认 actual path", ("hero_replay.ActualPathEvidence", "hero_replay.ACTUAL_PATH_PROOFS")),
    (20, "运行故意 fallback 负对照", ("hero_replay.ActualPathEvidence.problems",)),
    (21, "冻结 confirmatory workload strata", ("hero_replay.ConfirmatoryPlan",)),
    (22, "冻结冷/热边界", ("hero_replay.ConfirmatoryPlan",)),
    (23, "冻结运行 block 与交错顺序", ("hero_replay.ConfirmatoryPlan.INTERLEAVE_ORDERS",)),
    (24, "记录运行前稳定性窗口", ("hero_replay.DeviceHealth", "hero_replay.DeviceHealth.problems")),
    (25, "采集 operator micro raw", ("telemetry.ActionLog", "records.HeroReplayResult")),
    (26, "采集 block/model raw", ("records.HeroReplayResult", "telemetry.SegmentTiming")),
    (27, "采集 service raw", ("records.HeroReplayResult.primary_effect",)),
    (28, "采集资源与能效", ("telemetry.SegmentTiming", "records.HeroReplayResult")),
    (29, "采集系统时间线 Profile", ("hero_replay.profile_reconciliation_rows", "figures.LineageLayer")),
    (30, "采集 kernel counter Profile", ("hero_replay.profile_reconciliation_rows", "hero_replay.PROFILE_LAYERS")),
    (31, "重算原始热点占比", ("hero_replay.AmdahlModel",)),
    (32, "计算预注册 Amdahl 预测", ("hero_replay.AmdahlModel.predicted", "hero_replay.AmdahlModel.prediction_interval")),
    (33, "估计 primary effect", ("hero_replay.BlockedEstimate", "hero_replay.BlockedEstimate.effect")),
    (34, "比较原报告与重放结果", ("hero_replay.compare_with_release",)),
    (35, "分析 Amdahl 预测误差", ("hero_replay.AmdahlModel.explain_residual", "hero_replay.AMDAHL_ERROR_COMPONENTS")),
    (36, "分析环境差异", ("hero_replay.environment_diff",)),
    (37, "注入 artifact mismatch", ("hero_replay.ArtifactIdentity.problems", "identity.EvidenceRef")),
    (38, "注入错误后端或 silent fallback", ("hero_replay.ActualPathEvidence", "contracts.check_no_silent_degradation")),
    (39, "注入 correctness failure", ("hero_replay.evaluate_correctness_matrix", "contracts.check_quality_before_performance")),
    (40, "执行最小跨版本敏感性", ("hero_replay.environment_diff", "campaign.REQUIRED_ISOLATION")),
    (41, "裁决结论一致性", ("hero_replay.replay_verdict", "hero_replay.REPLAY_VERDICTS")),
    (42, "更新 Claim Ledger", ("hero_replay.claim_actions_for", "claims.revise_claim")),
    (43, "生成一键 Hero Replay", ("hero_replay.one_click_replay_command", "experiment.RunDirectory.write_verdict")),
    (44, "独立复核 raw→report", ("figures.independent_review_task", "figures.rebuild_chain")),
    (45, "冻结 HeroReplayRecord", ("hero_replay.freeze_replay", "records.HeroReplayResult")),
)

TITLE = "目标 GPU/NPU Hero Story 独立重放"
LEVEL = "P0"
CLAIM_BOUNDARY = "E15-03 是作者侧/受控新环境的重放，不等于独立第三方复现（E15-09 才验证独立性）"


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the hero-replay interfaces (labelled smoke)."""
    problems: List[str] = []
    contract = HeroStoryContract(
        candidate_id="cand-1",
        hero_claim_id="CLM-abc",
        auxiliary_claim_ids=("CLM-x", "CLM-y"),
        primary_estimand="tokens/s",
        minimum_meaningful_effect=0.05,
        guard_band=0.10,
        quality_gate="greedy token agreement >= 0.99",
        failure_policy="update the claim, never the figure",
        intended_difference="custom RMSNorm kernel instead of the reference implementation",
    )
    problems.extend(contract.problems())
    model = AmdahlModel(hotspot_share=0.3, local_speedup=2.0, integration_overhead=0.02, recomputed_in_new_environment=True)
    expected = 1.0 / ((1 - 0.3) + 0.3 / 2.0)
    if abs(model.ideal_upper_bound() - expected) > 1e-9:
        problems.append("the Amdahl upper bound is wrong")
    copied = AmdahlModel(hotspot_share=0.3, local_speedup=2.0, recomputed_in_new_environment=False)
    if not copied.problems():
        problems.append("a copied hotspot share was accepted")
    path = ActualPathEvidence(proofs={"kernel_name": "hqsb_rmsnorm_v2"}, binary_sha256="sha256:" + "0" * 64)
    if not path.problems():
        problems.append("an actual-path proof without a negative control was accepted")
    estimate = BlockedEstimate(
        blocks=({"baseline": 100.0, "candidate": 120.0}, {"baseline": 101.0, "candidate": 118.0}, {"baseline": 99.0, "candidate": 119.0}),
        minimum_meaningful_effect=0.01,
    )
    effect = estimate.effect()
    if effect["n"] != 3 or effect["crosses_zero"]:
        problems.append("the blocked estimate is wrong")
    verdict = replay_verdict(
        correctness_ok=True,
        quality_ok=True,
        actual_path_ok=False,
        within_guard_band=True,
        direction_same=True,
        mechanism_consistent=True,
        interval_crosses_zero=False,
    )
    if verdict != "INVALID_PATH":
        problems.append(f"a fallback run produced verdict {verdict!r} instead of INVALID_PATH")
    if not claim_actions_for("QUALITY_DISQUALIFIED"):
        problems.append("a quality failure did not require claim actions")
    stopped = check_stop_rules(correctness_ok=False, device_ok=True, oom=False, profiler_disturbed=False, budget_remaining=True)
    if not stopped:
        problems.append("a correctness failure did not fire a stop rule")
    return {
        "status": "smoke",
        "claim_allowed": False,
        "narrative_nodes": len(NARRATIVE_NODES),
        "verdicts": len(REPLAY_VERDICTS),
        "amdahl_upper_bound": model.ideal_upper_bound(),
        "problems": problems,
    }
