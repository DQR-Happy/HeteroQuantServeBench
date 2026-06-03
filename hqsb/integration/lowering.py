"""Lowering registry, allocation accounting and benefit attribution (E06-07).

The core performance experiment must show a chain, not a node count
(E06-07 §1):

    pattern hit → lowering selected → target kernel observed
      → intermediate/launch reduced → block/model phase explained

This module provides:

* :class:`LoweringTarget` + :class:`LoweringRegistry` — capability-driven
  selection with an explicit priority rule.  A rule may mention *semantics and
  capability*, never a model/module name: selecting by module path is exactly
  what E06-11 fails (E06-07 §5 "if the rule is a hard-coded module name, E06-11
  must fail");
* :class:`LoweringDecision` — selected target, rejected targets with reasons,
  generated artifact and observed kernel;
* allocation accounting (theoretical intermediate bytes vs measured) with an
  itemised residual;
* Amdahl-style prediction and an attribution report so a shortfall is explained
  rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError, RegistryError

# ── targets and capability ────────────────────────────────────────────────


@dataclass(frozen=True)
class LoweringCapability:
    """What a lowering target can execute (declared, then verified)."""

    dtypes: Tuple[str, ...] = ("float16", "bfloat16", "float32")
    layouts: Tuple[str, ...] = ("contiguous",)
    min_rank: int = 1
    max_rank: int = 4
    min_m: int = 1
    max_m: int = 0  # 0 == unbounded
    arch: Tuple[str, ...] = ()
    requires_group_size: bool = False
    group_sizes: Tuple[int, ...] = ()
    supports_dynamic: bool = True
    stream_aware: bool = True
    max_workspace_bytes: int = 0

    def accepts(self, request: "LoweringRequest") -> Tuple[bool, str]:
        if request.dtype not in self.dtypes:
            return False, "DTYPE"
        if request.layout not in self.layouts:
            return False, "STRIDE_LAYOUT"
        if not self.min_rank <= request.rank <= self.max_rank:
            return False, "SHAPE"
        if not self.min_m <= request.m:
            return False, "SHAPE"
        if self.max_m and request.m > self.max_m:
            return False, "SHAPE"
        if self.arch and request.arch and request.arch not in self.arch:
            return False, "BACKEND_CAPABILITY"
        if not self.supports_dynamic and request.dynamic:
            return False, "DYNAMIC_CONSTRAINT"
        if self.requires_group_size:
            if request.group_size is None or request.group_size not in self.group_sizes:
                return False, "QUANT_POLICY"
        if request.workspace_bytes and request.workspace_bytes > self.max_workspace_bytes:
            return False, "SHAPE"
        return True, ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dtypes": list(self.dtypes),
            "layouts": list(self.layouts),
            "rank": [self.min_rank, self.max_rank],
            "m_range": [self.min_m, self.max_m],
            "arch": list(self.arch),
            "group_sizes": list(self.group_sizes),
            "supports_dynamic": self.supports_dynamic,
            "stream_aware": self.stream_aware,
            "max_workspace_bytes": self.max_workspace_bytes,
        }


@dataclass(frozen=True)
class LoweringRequest:
    """What the graph node needs, expressed in capability terms only."""

    pattern_id: str
    pattern_version: str = ""
    node: str = ""
    module_path: str = ""
    op: str = ""
    dtype: str = "float16"
    layout: str = "contiguous"
    rank: int = 2
    m: int = 1
    symbolic_shape: str = ""
    actual_shape: Tuple[int, ...] = ()
    arch: str = ""
    group_size: Optional[int] = None
    dynamic: bool = False
    workspace_bytes: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_version": self.pattern_version,
            "node": self.node,
            "module_path": self.module_path,
            "op": self.op,
            "dtype": self.dtype,
            "layout": self.layout,
            "rank": self.rank,
            "m": self.m,
            "symbolic_shape": self.symbolic_shape,
            "actual_shape": list(self.actual_shape),
            "arch": self.arch,
            "group_size": self.group_size,
            "dynamic": self.dynamic,
            "workspace_bytes": self.workspace_bytes,
        }


@dataclass(frozen=True)
class PriorityRule:
    """Ordering rule: a *named policy*, never a per-module special case.

    ``key`` is an explicit, documented tag such as ``"static_m1"`` or
    ``"prefill_large_m"``; the rule must not be keyed on module names.
    """

    name: str
    tags: Tuple[str, ...] = ()
    prefer: Tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.name != self.name.lower():  # pragma: no cover - defensive
            raise ConfigError(f"priority rule name {self.name!r} must be lower-case")

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "tags": list(self.tags), "prefer": list(self.prefer), "note": self.note}


@dataclass(frozen=True)
class LoweringTarget:
    """One lowering provider with its capability and kernel identity."""

    name: str
    provider: str
    capability: LoweringCapability = field(default_factory=LoweringCapability)
    kernel_symbol: str = ""
    artifact_kind: str = "generated"  # "generated" | "prebuilt" | "reference"
    version: str = "1.0.0"
    reference: bool = False
    priority: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "capability": self.capability.as_dict(),
            "kernel_symbol": self.kernel_symbol,
            "artifact_kind": self.artifact_kind,
            "version": self.version,
            "reference": self.reference,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class RejectedTarget:
    target: str
    reason: str
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "reason": self.reason, "detail": self.detail}


@dataclass(frozen=True)
class LoweringDecision:
    """One decision, with everything E06-07 §5 requires."""

    request: LoweringRequest
    selected: Optional[LoweringTarget]
    rejected: Tuple[RejectedTarget, ...] = ()
    rule: Optional[PriorityRule] = None
    rule_reason: str = ""
    generated_artifact: str = ""
    observed_kernel: str = ""
    fallback: bool = False
    fallback_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.selected is not None

    @property
    def selected_name(self) -> str:
        return self.selected.name if self.selected else ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request.as_dict(),
            "selected": self.selected.as_dict() if self.selected else None,
            "rejected": [item.as_dict() for item in self.rejected],
            "rule": self.rule.as_dict() if self.rule else None,
            "rule_reason": self.rule_reason,
            "generated_artifact": self.generated_artifact,
            "observed_kernel": self.observed_kernel,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
        }


class LoweringRegistry:
    """Capability + priority selection; refuses model-name keyed rules."""

    def __init__(self, rules: Sequence[PriorityRule] = ()) -> None:
        self._targets: Dict[str, LoweringTarget] = {}
        self._rules: List[PriorityRule] = list(rules)

    def register(self, target: LoweringTarget) -> None:
        if target.name in self._targets:
            raise RegistryError(
                f"lowering target {target.name!r} is already registered",
                details={"field": "name"},
            )
        if not target.provider:
            raise ConfigError(
                f"target {target.name!r} must declare a provider",
                details={"field": "provider"},
            )
        self._targets[target.name] = target

    def add_rule(self, rule: PriorityRule) -> None:
        self._rules.append(rule)

    @property
    def targets(self) -> Tuple[LoweringTarget, ...]:
        return tuple(self._targets[name] for name in sorted(self._targets))

    def select(self, request: LoweringRequest) -> LoweringDecision:
        """Select a target by capability then by rule; record every rejection."""
        rejected: List[RejectedTarget] = []
        eligible: List[LoweringTarget] = []
        for target in self.targets:
            accepted, reason = target.capability.accepts(request)
            if accepted:
                eligible.append(target)
            else:
                rejected.append(RejectedTarget(target.name, reason, reason))
        if not eligible:
            return LoweringDecision(
                request=request,
                selected=None,
                rejected=tuple(rejected),
                fallback=True,
                fallback_reason="NO_ELIGIBLE_LOWERING_TARGET",
            )
        rule, rule_reason = self._match_rule(request, eligible)
        ordered = sorted(
            eligible,
            key=lambda target: (
                0 if (rule and target.name in rule.prefer) else 1,
                -target.priority,
                target.name,
            ),
        )
        return LoweringDecision(
            request=request,
            selected=ordered[0],
            rejected=tuple(rejected),
            rule=rule,
            rule_reason=rule_reason,
        )

    def _match_rule(
        self, request: LoweringRequest, eligible: Sequence[LoweringTarget]
    ) -> Tuple[Optional[PriorityRule], str]:
        for rule in self._rules:
            if rule.tags and request.op and request.op not in rule.tags:
                continue
            if rule.prefer and not any(target.name in rule.prefer for target in eligible):
                continue
            return rule, f"matched rule {rule.name!r}"
        return None, "no rule matched; default ordering by priority then name"

    def selection_audit(self) -> Dict[str, Any]:
        """Static check that no rule encodes a model/module name (E06-11 §5)."""
        forbidden_tokens = ("qwen", "llama", "model.", "layers.", "self_attn", "mlp.")  # hqsb-hardcode-allow: denylist vocabulary, not a selector
        findings: List[Dict[str, Any]] = []
        for rule in self._rules:
            payload = " ".join((rule.name, *rule.tags, *rule.prefer))
            for token in forbidden_tokens:
                if token in payload.lower():
                    findings.append(
                        {
                            "rule": rule.name,
                            "token": token,
                            "reason": "MODEL_NAME_SELECTOR",
                        }
                    )
        return {"ok": not findings, "findings": findings, "rules": len(self._rules)}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "targets": [target.as_dict() for target in self.targets],
            "rules": [rule.as_dict() for rule in self._rules],
        }


def frozen_registry() -> LoweringRegistry:
    """The declared targets for the S06 patterns (capability descriptions only)."""
    registry = LoweringRegistry(
        rules=(
            PriorityRule(
                name="decode_narrow_m",
                tags=("hqsb::dequant_linear",),
                prefer=("hqsb.triton.dequant_linear",),
                note="M=1/narrow decode prefers the fused dequant kernel",
            ),
            PriorityRule(
                name="prefill_wide_m",
                tags=("hqsb::dequant_linear",),
                prefer=("hqsb.cuda.dequant_linear",),
                note="wide prefill prefers the CUDA/CUTLASS route",
            ),
        )
    )
    registry.register(
        LoweringTarget(
            name="hqsb.cuda.fused_add_rms_norm",
            provider="cuda_shared_lib",
            capability=LoweringCapability(
                dtypes=("float16", "bfloat16", "float32"),
                arch=("sm_86", "sm_87"),
                min_m=1,
                max_m=0,
            ),
            kernel_symbol="hqsb_fused_add_rms_norm_v1",
            artifact_kind="prebuilt",
            priority=10,
        )
    )
    registry.register(
        LoweringTarget(
            name="hqsb.triton.fused_add_rms_norm",
            provider="triton",
            capability=LoweringCapability(
                dtypes=("float16", "bfloat16", "float32"),
                arch=("sm_86", "sm_87"),
            ),
            kernel_symbol="fused_add_rms_norm_kernel",
            artifact_kind="generated",
            priority=5,
        )
    )
    registry.register(
        LoweringTarget(
            name="hqsb.reference.fused_add_rms_norm",
            provider="composite",
            capability=LoweringCapability(
                dtypes=("float16", "bfloat16", "float32"),
                min_rank=1,
                max_rank=4,
                min_m=1,
            ),
            kernel_symbol="",
            artifact_kind="reference",
            reference=True,
            priority=-10,
        )
    )
    registry.register(
        LoweringTarget(
            name="hqsb.triton.dequant_linear",
            provider="triton",
            capability=LoweringCapability(
                dtypes=("float16", "bfloat16"),
                requires_group_size=True,
                group_sizes=(32, 64, 128),
                min_m=1,
                max_m=8,
            ),
            kernel_symbol="hqsb_dequant_gemm_kernel",
            artifact_kind="generated",
            priority=10,
        )
    )
    registry.register(
        LoweringTarget(
            name="hqsb.cuda.dequant_linear",
            provider="cuda_shared_lib",
            capability=LoweringCapability(
                dtypes=("float16", "bfloat16"),
                requires_group_size=True,
                group_sizes=(32, 64, 128),
                min_m=1,
            ),
            kernel_symbol="hqsb_dequant_linear_cuda",
            artifact_kind="prebuilt",
            priority=8,
        )
    )
    registry.register(
        LoweringTarget(
            name="hqsb.inductor.generated",
            provider="inductor",
            capability=LoweringCapability(
                dtypes=("float16", "bfloat16", "float32"),
                min_rank=1,
                max_rank=4,
                min_m=1,
                supports_dynamic=True,
            ),
            kernel_symbol="triton_poi_fused",
            artifact_kind="generated",
            priority=0,
        )
    )
    return registry


# ── allocation accounting ─────────────────────────────────────────────────


@dataclass(frozen=True)
class AllocationAccount:
    """Theoretical vs measured memory, with an itemised residual (E06-07 §8)."""

    intermediate_bytes_before: int
    intermediate_bytes_after: int
    workspace_bytes: int = 0
    materialized_dequant_bytes: int = 0
    contiguous_copies_bytes: int = 0
    allocator_allocated_bytes: int = 0
    allocator_reserved_bytes: int = 0
    peak_bytes: int = 0
    launch_count_before: int = 0
    launch_count_after: int = 0

    @property
    def theoretical_saving_bytes(self) -> int:
        return self.intermediate_bytes_before - self.intermediate_bytes_after

    @property
    def saving_accounted_bytes(self) -> int:
        return (
            self.theoretical_saving_bytes
            - self.workspace_bytes
            - self.materialized_dequant_bytes
            - self.contiguous_copies_bytes
        )

    @property
    def launch_delta(self) -> int:
        return self.launch_count_after - self.launch_count_before

    def reconcile(self, measured_saving_bytes: int) -> Dict[str, Any]:
        """Close the theoretical saving against the measured one.

        ``measured_saving_bytes`` is a *saving* (positive = fewer bytes than
        before).  A negative residual means the saving was over-predicted and
        must be explained rather than rounded away.
        """
        residual = measured_saving_bytes - self.saving_accounted_bytes
        return {
            "theoretical_saving_bytes": self.theoretical_saving_bytes,
            "accounted_saving_bytes": self.saving_accounted_bytes,
            "measured_saving_bytes": measured_saving_bytes,
            "residual_bytes": residual,
            "items": {
                "workspace": self.workspace_bytes,
                "materialized_dequant": self.materialized_dequant_bytes,
                "contiguous_copies": self.contiguous_copies_bytes,
            },
            "explained": residual == 0,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "intermediate_bytes_before": self.intermediate_bytes_before,
            "intermediate_bytes_after": self.intermediate_bytes_after,
            "theoretical_saving_bytes": self.theoretical_saving_bytes,
            "workspace_bytes": self.workspace_bytes,
            "materialized_dequant_bytes": self.materialized_dequant_bytes,
            "contiguous_copies_bytes": self.contiguous_copies_bytes,
            "allocator_allocated_bytes": self.allocator_allocated_bytes,
            "allocator_reserved_bytes": self.allocator_reserved_bytes,
            "peak_bytes": self.peak_bytes,
            "launch_count_before": self.launch_count_before,
            "launch_count_after": self.launch_count_after,
            "launch_delta": self.launch_delta,
        }


#: Element sizes for the theoretical byte model (no torch needed).
DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "uint8": 1,
    "int8": 1,
    "int32": 4,
}


def tensor_bytes(shape: Sequence[int], dtype: str) -> int:
    """Theoretical bytes of one intermediate tensor."""
    if dtype not in DTYPE_BYTES:
        raise ConfigError(
            f"unknown dtype {dtype!r} for the byte model",
            details={"field": "dtype", "allowed": sorted(DTYPE_BYTES)},
        )
    product = 1
    for dim in shape:
        product *= int(dim)
    return product * DTYPE_BYTES[dtype]


def intermediate_bytes(
    tensors: Sequence[Tuple[str, Sequence[int]]], dtype: str = "float16"
) -> int:
    """Sum of theoretical intermediate sizes (buffers the fused path can drop)."""
    return sum(tensor_bytes(shape, dtype) for _name, shape in tensors)


def fuse_saving_model(
    unfused_intermediates: Sequence[Tuple[str, Sequence[int]]],
    fused_intermediates: Sequence[Tuple[str, Sequence[int]]],
    dtype: str = "float16",
) -> Dict[str, int]:
    """Predicted intermediate traffic before/after, itemised by tensor name."""
    before = {name: tensor_bytes(shape, dtype) for name, shape in unfused_intermediates}
    after = {name: tensor_bytes(shape, dtype) for name, shape in fused_intermediates}
    return {
        "before_bytes": sum(before.values()),
        "after_bytes": sum(after.values()),
        "saving_bytes": sum(before.values()) - sum(after.values()),
        "removed_tensors": sorted(set(before) - set(after)),
    }


# ── Amdahl prediction and attribution ─────────────────────────────────────


@dataclass(frozen=True)
class AmdahlPrediction:
    """Amdahl bound and the simple additive prediction for one phase."""

    target_fraction: float  # p: share of the baseline phase time in the target
    speedup: float  # s: speedup of the target itself
    baseline_phase_ms: float
    added_overhead_ms: float = 0.0
    phase: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_fraction <= 1.0:
            raise ConfigError(
                f"target_fraction must be in [0,1], got {self.target_fraction}",
                details={"field": "target_fraction"},
            )
        if self.speedup <= 0:
            raise ConfigError(
                f"speedup must be positive, got {self.speedup}",
                details={"field": "speedup"},
            )

    @property
    def predicted_model_speedup(self) -> float:
        p = self.target_fraction
        return 1.0 / ((1.0 - p) + p / self.speedup)

    @property
    def predicted_delta_ms(self) -> float:
        return (
            self.baseline_phase_ms
            - self.baseline_phase_ms * self.target_fraction / self.speedup
            - self.added_overhead_ms
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "target_fraction": self.target_fraction,
            "speedup": self.speedup,
            "baseline_phase_ms": self.baseline_phase_ms,
            "added_overhead_ms": self.added_overhead_ms,
            "predicted_model_speedup": self.predicted_model_speedup,
            "predicted_delta_ms": self.predicted_delta_ms,
        }


@dataclass(frozen=True)
class AttributionReport:
    """Predicted vs actual with the residual attributed to named factors."""

    prediction: AmdahlPrediction
    actual_delta_ms: float
    factors: Mapping[str, float] = field(default_factory=dict)

    @property
    def residual_ms(self) -> float:
        unexplained = self.actual_delta_ms - self.prediction.predicted_delta_ms
        attributed = sum(self.factors.values())
        return unexplained - attributed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "predicted_delta_ms": self.prediction.predicted_delta_ms,
            "actual_delta_ms": self.actual_delta_ms,
            "factors": dict(self.factors),
            "residual_ms": self.residual_ms,
            "explained": abs(self.residual_ms) < 1e-9,
        }


@dataclass(frozen=True)
class AblationCase:
    """One ablation cell (E06-07 §11 step 18)."""

    name: str
    pattern_enabled: bool
    lowering_enabled: bool
    kernel_enabled: bool
    epilogue_enabled: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pattern_enabled": self.pattern_enabled,
            "lowering_enabled": self.lowering_enabled,
            "kernel_enabled": self.kernel_enabled,
            "epilogue_enabled": self.epilogue_enabled,
        }


def ablation_matrix() -> Tuple[AblationCase, ...]:
    """The frozen ablation set: one factor disabled at a time."""
    return (
        AblationCase("full", True, True, True, True),
        AblationCase("no_pattern", False, True, True, True),
        AblationCase("no_lowering", True, False, True, True),
        AblationCase("no_kernel", True, True, False, True),
        AblationCase("no_epilogue", True, True, True, False),
    )


__all__ = [
    "DTYPE_BYTES",
    "AblationCase",
    "AllocationAccount",
    "AmdahlPrediction",
    "AttributionReport",
    "LoweringCapability",
    "LoweringDecision",
    "LoweringRegistry",
    "LoweringRequest",
    "LoweringTarget",
    "PriorityRule",
    "RejectedTarget",
    "ablation_matrix",
    "frozen_registry",
    "fuse_saving_model",
    "intermediate_bytes",
    "tensor_bytes",
]
