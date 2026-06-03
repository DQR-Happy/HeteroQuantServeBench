"""Multi-path differential correctness scaffolding (E06-03).

S06 re-establishes its semantic floor before any automatic rewrite: the same
real model input must stay acceptable across reference, custom-op, explicit
fusion, compiled and HQSB-lowered paths (E06-03 §1).

The module provides the *machinery* of that comparison:

* the eight-row path matrix, each row isolating one factor;
* a tolerance registry keyed by level **and** dtype (never one global threshold,
  and never relaxed for a compiled path);
* metric computation reusing the repository's existing numerical-diff口径
  (:mod:`hqsb.benchmark.metrics`) plus the extra diagnostics the protocol asks
  for (NaN/Inf, first mismatch, cosine, relative L2);
* the frozen add+RMSNorm fusion semantics and a pure-Python reference so the
  *composition*, not the target kernel, is the oracle (E06-03 §4);
* state/mode-switch auditing, first-divergence localisation and an explicit
  correctness matrix in which a missing cell can never be averaged away.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError, SchemaError

# ── the path matrix ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class DifferentialPath:
    """One row of the path matrix (E06-03 §2)."""

    path_id: str
    graph: str
    operator: str
    kernel: str
    purpose: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "graph": self.graph,
            "operator": self.operator,
            "kernel": self.kernel,
            "purpose": self.purpose,
        }


PATHS: Tuple[DifferentialPath, ...] = (
    DifferentialPath("eager-reference", "original model", "ATen/composite", "PyTorch", "total oracle"),
    DifferentialPath("eager-custom", "no automatic rewrite", "HQSB op", "forced backend", "isolate registration/kernel"),
    DifferentialPath("eager-explicit-fused", "manual fused call", "HQSB fused", "forced", "isolate fusion math"),
    DifferentialPath("compile-baseline", "Dynamo/Inductor", "original ATen", "Inductor", "compiler control"),
    DifferentialPath("compile-custom-opaque", "custom op preserved", "HQSB", "selected", "compile composition"),
    DifferentialPath("compile-pattern-reference", "pattern hit", "reference lowering", "composite", "isolate rewrite"),
    DifferentialPath("compile-pattern-hqsb", "pattern hit", "HQSB lowering", "actual kernel", "full target path"),
    DifferentialPath("fallback", "HQSB requested but unsupported", "reference", "actual fallback", "semantic recovery"),
)

MATRIX_COLUMNS = ("operator", "block", "model_logits", "generation", "alias", "stream", "state")


@dataclass(frozen=True)
class DifferentialSpec:
    """The frozen binding every differential run must share (E06-03 step 1)."""

    model_id: str
    model_manifest_sha256: str
    input_token_hash: str
    quant_artifact_hash: str = ""
    paths: Tuple[str, ...] = tuple(path.path_id for path in PATHS)
    seed: int = 0
    kv_state: str = "empty"
    generation_config: str = ""
    tolerance_registry_hash: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        known = {path.path_id for path in PATHS}
        unknown = [item for item in self.paths if item not in known]
        if unknown:
            raise ConfigError(
                f"unknown differential path(s) {unknown}; allowed={sorted(known)}",
                details={"field": "paths"},
            )
        if not self.model_manifest_sha256:
            raise ConfigError(
                "a differential spec without the model manifest hash cannot be "
                "reproduced (performance/correctness runs must share identity)",
                details={"field": "model_manifest_sha256"},
            )

    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_manifest_sha256": self.model_manifest_sha256,
            "input_token_hash": self.input_token_hash,
            "quant_artifact_hash": self.quant_artifact_hash,
            "paths": list(self.paths),
            "seed": self.seed,
            "kv_state": self.kv_state,
            "generation_config": self.generation_config,
            "tolerance_registry_hash": self.tolerance_registry_hash,
            "notes": self.notes,
        }


# ── tolerance discipline ──────────────────────────────────────────────────


class Level:
    OPERATOR = "operator"
    FUSION_LOCAL = "fusion_local"
    BLOCK = "block"
    MODEL_LOGITS = "model_logits"
    GENERATION = "generation"

    ALL = (OPERATOR, FUSION_LOCAL, BLOCK, MODEL_LOGITS, GENERATION)


@dataclass(frozen=True)
class ToleranceSpec:
    """Pre-registered tolerance for one (level, dtype) pair."""

    level: str
    dtype: str
    max_abs: float
    rel_l2: float
    cosine: float = 0.9999
    kl_max: Optional[float] = None
    top_k: int = 0
    margin: Optional[float] = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.level not in Level.ALL:
            raise SchemaError(
                f"unknown level {self.level!r}",
                details={"field": "level", "allowed": list(Level.ALL)},
            )
        for name in ("max_abs", "rel_l2", "cosine"):
            value = getattr(self, name)
            if value <= 0:
                raise ConfigError(
                    f"{self.level}/{self.dtype}: {name} must be positive",
                    details={"field": name, "actual": value},
                )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "dtype": self.dtype,
            "max_abs": self.max_abs,
            "rel_l2": self.rel_l2,
            "cosine": self.cosine,
            "kl_max": self.kl_max,
            "top_k": self.top_k,
            "margin": self.margin,
            "note": self.note,
        }


class ToleranceRegistry:
    """No default threshold: a missing (level, dtype) entry is an error."""

    def __init__(self, specs: Sequence[ToleranceSpec]) -> None:
        self._specs: Dict[Tuple[str, str], ToleranceSpec] = {}
        for spec in specs:
            self._specs[(spec.level, spec.dtype)] = spec

    def get(self, level: str, dtype: str) -> ToleranceSpec:
        key = (level, dtype)
        if key not in self._specs:
            raise ConfigError(
                f"no pre-registered tolerance for level={level!r} dtype={dtype!r}; "
                "using a global default would silently loosen the gate",
                details={"field": "tolerance", "available": sorted(
                    f"{level}/{dtype}" for level, dtype in self._specs
                )},
            )
        return self._specs[key]

    @property
    def digest(self) -> str:
        payload = json.dumps(
            [self._specs[key].as_dict() for key in sorted(self._specs)], sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "digest": self.digest,
            "specs": [self._specs[key].as_dict() for key in sorted(self._specs)],
        }


def default_tolerance_registry() -> ToleranceRegistry:
    """Grounded in S03/S04.x registered thresholds (FP32 5e-4, FP16 2e-2) plus
    scale-free gates; the compiled path reuses these and never relaxes them."""
    return ToleranceRegistry(
        (
            ToleranceSpec(Level.OPERATOR, "float32", 5e-4, 1e-5, note="S03 registered threshold"),
            ToleranceSpec(Level.OPERATOR, "float16", 2e-2, 1e-2, note="S03/S04 registered threshold"),
            ToleranceSpec(Level.OPERATOR, "bfloat16", 3e-2, 2e-2),
            ToleranceSpec(
                Level.FUSION_LOCAL,
                "float16",
                2e-2,
                1e-2,
                note="fusion reassociation budget must be pre-registered, not discovered",
            ),
            ToleranceSpec(Level.FUSION_LOCAL, "float32", 5e-4, 1e-5),
            ToleranceSpec(Level.BLOCK, "float16", 5e-2, 3e-2, note="error accumulates across a block"),
            ToleranceSpec(Level.BLOCK, "float32", 2e-3, 1e-4),
            ToleranceSpec(
                Level.MODEL_LOGITS,
                "float16",
                0.5,
                0.05,
                kl_max=1e-3,
                top_k=5,
                margin=0.0,
                note="logit-scale gate; token metrics are reported separately",
            ),
            ToleranceSpec(
                Level.GENERATION,
                "float16",
                1.0,
                0.1,
                kl_max=5e-3,
                top_k=5,
                note="per-step gate; first divergence is also reported",
            ),
        )
    )


# ── metric computation ────────────────────────────────────────────────────

try:  # pragma: no cover - import shape is asserted by tests
    from hqsb.benchmark.metrics import numerical_diff_summary as _numerical_diff_summary
except Exception:  # noqa: BLE001 - fallback keeps the module importable standalone
    _numerical_diff_summary = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MetricReport:
    """Numerical comparison of one candidate against its oracle."""

    level: str
    dtype: str
    max_abs: float
    mean_abs: float
    rmse: float
    rel_l2: float
    cosine: float
    nan_count: int = 0
    inf_count: int = 0
    first_mismatch: Optional[int] = None
    observed_kernel: str = ""
    passed: bool = True
    failures: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "dtype": self.dtype,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "rmse": self.rmse,
            "rel_l2": self.rel_l2,
            "cosine": self.cosine,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "first_mismatch": self.first_mismatch,
            "observed_kernel": self.observed_kernel,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def _flatten(values: Any) -> List[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    flat: List[float] = []

    def walk(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for sub in item:
                walk(sub)
        else:
            flat.append(float(item))

    walk(values)
    return flat


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    na = math.sqrt(sum(a * a for a in left))
    nb = math.sqrt(sum(b * b for b in right))
    if na == 0 or nb == 0:
        return 0.0 if na != nb else 1.0
    return dot / (na * nb)


def compare_arrays(
    candidate: Any,
    oracle: Any,
    tolerance: ToleranceSpec,
    *,
    observed_kernel: str = "",
) -> MetricReport:
    """Field-level numerical comparison against the frozen tolerance.

    NaN/Inf is a hard safety failure independent of any numeric threshold
    (E06-03 §8), and the first mismatching index is retained so a failure can be
    localised instead of merely reported.
    """
    left = _flatten(candidate)
    right = _flatten(oracle)
    if len(left) != len(right):
        raise ConfigError(
            f"candidate/oracle element counts differ: {len(left)} vs {len(right)}",
            details={"field": "shape"},
        )
    nan_count = sum(1 for value in left if math.isnan(value))
    inf_count = sum(1 for value in left if math.isinf(value))
    diffs = [abs(a - b) for a, b in zip(left, right)]
    max_abs = max(diffs) if diffs else 0.0
    mean_abs = sum(diffs) / len(diffs) if diffs else 0.0
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else 0.0
    norm_oracle = math.sqrt(sum(b * b for b in right))
    rel_l2 = (math.sqrt(sum(d * d for d in diffs)) / norm_oracle) if norm_oracle else 0.0
    cosine = _cosine(left, right)
    first_mismatch = None
    for index, diff in enumerate(diffs):
        if diff > tolerance.max_abs:
            first_mismatch = index
            break
    failures: List[str] = []
    if nan_count or inf_count:
        failures.append("NAN_OR_INF")
    if max_abs > tolerance.max_abs:
        failures.append("MAX_ABS")
    if rel_l2 > tolerance.rel_l2:
        failures.append("REL_L2")
    if cosine < tolerance.cosine:
        failures.append("COSINE")
    return MetricReport(
        level=tolerance.level,
        dtype=tolerance.dtype,
        max_abs=max_abs,
        mean_abs=mean_abs,
        rmse=rmse,
        rel_l2=rel_l2,
        cosine=cosine,
        nan_count=nan_count,
        inf_count=inf_count,
        first_mismatch=first_mismatch,
        observed_kernel=observed_kernel,
        passed=not failures,
        failures=tuple(failures),
    )


def reference_diff_summary(candidate: Any, oracle: Any) -> Dict[str, Any]:
    """Repository-standard diff summary (kept as the shared metric口径)."""
    if _numerical_diff_summary is None:  # pragma: no cover - defensive
        raise ConfigError(
            "hqsb.benchmark.metrics.numerical_diff_summary is unavailable; the "
            "differential report must reuse the repository metric definitions",
            details={"field": "metrics"},
        )
    return dict(_numerical_diff_summary(oracle, candidate))


# ── frozen fusion semantics ───────────────────────────────────────────────


class ReadOrder:
    BEFORE_ROUNDING = "before_rounding"
    AFTER_ROUNDING = "after_rounding"

    ALL = (BEFORE_ROUNDING, AFTER_ROUNDING)


@dataclass(frozen=True)
class FusionSemantics:
    """The add+RMSNorm decisions the protocol requires to be frozen (E06-03 §4)."""

    name: str
    accumulation_dtype: str
    r_new_rounding: str
    normalized_reads: str
    residual_in_place: bool
    eps_position: str
    weight_broadcast: bool
    outputs_alias_inputs: bool
    extra_users_policy: str

    def __post_init__(self) -> None:
        if self.normalized_reads not in ReadOrder.ALL:
            raise SchemaError(
                f"{self.name}: normalized_reads must be one of {list(ReadOrder.ALL)}",
                details={"field": "normalized_reads"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "accumulation_dtype": self.accumulation_dtype,
            "r_new_rounding": self.r_new_rounding,
            "normalized_reads": self.normalized_reads,
            "residual_in_place": self.residual_in_place,
            "eps_position": self.eps_position,
            "weight_broadcast": self.weight_broadcast,
            "outputs_alias_inputs": self.outputs_alias_inputs,
            "extra_users_policy": self.extra_users_policy,
        }


FROZEN_ADD_RMSNORM_SEMANTICS = FusionSemantics(
    name="hqsb.fusion.add_rmsnorm.v1",
    accumulation_dtype="float32",
    r_new_rounding="round to the activation dtype once, before normalization",
    normalized_reads="after_rounding",
    residual_in_place=False,
    eps_position="inside the mean of squares (x^2 + eps)",
    weight_broadcast=True,
    outputs_alias_inputs=False,
    extra_users_policy="the fused node must expose updated_residual to every previous user",
)


def fused_add_rms_norm_reference(
    x: Sequence[float],
    residual: Sequence[float],
    weight: Sequence[float],
    eps: float = 1e-6,
    semantics: FusionSemantics = FROZEN_ADD_RMSNORM_SEMANTICS,
) -> Tuple[List[float], List[float]]:
    """Pure-Python FP64 reference following the frozen semantics.

    This is deliberately **not** the target kernel: using the kernel as its own
    reference cannot reveal a semantic difference (E06-03 §12).
    """
    if len(x) != len(residual) or len(weight) != len(x):
        raise ConfigError(
            f"reference shapes disagree: x={len(x)} residual={len(residual)} weight={len(weight)}",
            details={"field": "shape"},
        )
    if semantics.residual_in_place:
        raise ConfigError(
            "the frozen functional contract cannot be evaluated with an in-place "
            "residual; a mutable variant must be a separate private op",
            details={"field": "residual_in_place"},
        )
    updated = [a + b for a, b in zip(residual, x)]
    mean_square = sum(value * value for value in updated) / len(updated)
    rsqrt = 1.0 / math.sqrt(mean_square + eps)
    normalized = [value * rsqrt * scale for value, scale in zip(updated, weight)]
    return normalized, updated


def composed_add_rms_norm_reference(
    x: Sequence[float],
    residual: Sequence[float],
    weight: Sequence[float],
    eps: float = 1e-6,
) -> Tuple[List[float], List[float]]:
    """The naive composition of two separate ops (the unfused oracle)."""
    updated = [a + b for a, b in zip(residual, x)]
    mean_square = sum(value * value for value in updated) / len(updated)
    rsqrt = 1.0 / math.sqrt(mean_square + eps)
    return [value * rsqrt * scale for value, scale in zip(updated, weight)], updated


# ── state and mode switching ──────────────────────────────────────────────


@dataclass(frozen=True)
class StateSnapshot:
    """State identity before/after a mode switch (E06-03 §6)."""

    state_dict_hash: str = ""
    parameter_ids: Tuple[str, ...] = ()
    tied_weights: Tuple[Tuple[str, str], ...] = ()
    buffers_hash: str = ""
    training_mode: bool = False
    inference_mode: bool = True
    autocast: bool = False
    rng_state_hash: str = ""
    kv_cache_hash: str = ""
    compiled_wrapper_id: str = ""
    backend_enabled: bool = False
    output_hash: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state_dict_hash": self.state_dict_hash,
            "parameter_ids": list(self.parameter_ids),
            "tied_weights": [list(item) for item in self.tied_weights],
            "buffers_hash": self.buffers_hash,
            "training_mode": self.training_mode,
            "inference_mode": self.inference_mode,
            "autocast": self.autocast,
            "rng_state_hash": self.rng_state_hash,
            "kv_cache_hash": self.kv_cache_hash,
            "compiled_wrapper_id": self.compiled_wrapper_id,
            "backend_enabled": self.backend_enabled,
            "output_hash": self.output_hash,
        }


def compare_state(before: StateSnapshot, after: StateSnapshot) -> Dict[str, Any]:
    """Diff two state snapshots field by field (restore must be exact)."""
    left, right = before.as_dict(), after.as_dict()
    differences = [
        {"field": key, "before": left[key], "after": right[key]}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    ]
    return {"ok": not differences, "differences": differences}


@dataclass
class ModeSwitchAudit:
    """enable/disable/restore with an explicit wrapper-rebuild policy."""

    name: str
    records: List[Dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        action: str,
        *,
        state: StateSnapshot,
        expected_backend: str,
        actual_backend: str,
        wrapper_rebuilt: bool,
        output_matches_reference: bool,
    ) -> Dict[str, Any]:
        """Record one switch; a stale compiled wrapper is a hard failure.

        After ``disable`` the old compiled callable must not be used
        (E06-03 §12), which is why ``wrapper_rebuilt`` is mandatory on re-enable.
        """
        payload = {
            "action": action,
            "expected_backend": expected_backend,
            "actual_backend": actual_backend,
            "wrapper_rebuilt": wrapper_rebuilt,
            "output_matches_reference": output_matches_reference,
            "state": state.as_dict(),
            "ok": (
                expected_backend == actual_backend
                and output_matches_reference
                and (action != "enable" or wrapper_rebuilt)
            ),
        }
        self.records.append(payload)
        return payload

    @property
    def ok(self) -> bool:
        return all(record["ok"] for record in self.records)

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "records": list(self.records)}


# ── divergence localisation and the correctness matrix ────────────────────


@dataclass(frozen=True)
class DivergenceRecord:
    """The first node/layer/step whose error exceeded its threshold."""

    level: str
    node: str
    op: str
    module: str
    layer: int
    step: int
    metric: str
    value: float
    threshold: float
    observed_kernel: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "node": self.node,
            "op": self.op,
            "module": self.module,
            "layer": self.layer,
            "step": self.step,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "observed_kernel": self.observed_kernel,
        }


class DivergenceLocator:
    """Collects per-node errors and reports the first over-threshold one."""

    def __init__(self) -> None:
        self.entries: List[DivergenceRecord] = []

    def observe(
        self,
        *,
        level: str,
        node: str,
        op: str,
        module: str,
        layer: int,
        step: int,
        metric: str,
        value: float,
        threshold: float,
        observed_kernel: str = "",
    ) -> DivergenceRecord:
        record = DivergenceRecord(
            level=level,
            node=node,
            op=op,
            module=module,
            layer=layer,
            step=step,
            metric=metric,
            value=value,
            threshold=threshold,
            observed_kernel=observed_kernel,
        )
        self.entries.append(record)
        return record

    def first_over(self) -> Optional[DivergenceRecord]:
        for record in sorted(self.entries, key=lambda item: (item.step, item.layer, item.node)):
            if record.value > record.threshold:
                return record
        return None

    def as_dict(self) -> Dict[str, Any]:
        first = self.first_over()
        return {
            "observed": len(self.entries),
            "first_over": first.as_dict() if first else None,
            "entries": [item.as_dict() for item in self.entries],
        }


class CellStatus:
    PASS = "PASS"
    FAIL = "FAIL"
    MISSING = "MISSING"

    ALL = (PASS, FAIL, MISSING)


@dataclass
class CorrectnessMatrix:
    """path × level matrix; ``MISSING`` is explicit and never averaged away."""

    specs: DifferentialSpec
    cells: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)

    def mark(self, path: str, level: str, status: str, detail: Optional[Mapping[str, Any]] = None) -> None:
        if status not in CellStatus.ALL:
            raise ConfigError(
                f"unknown cell status {status!r}",
                details={"field": "status", "allowed": list(CellStatus.ALL)},
            )
        if path not in self.specs.paths:
            raise ConfigError(
                f"path {path!r} is not in the frozen differential spec",
                details={"field": "path"},
            )
        if level not in Level.ALL:
            raise ConfigError(
                f"unknown level {level!r}", details={"field": "level"}
            )
        self.cells[(path, level)] = {"status": status, "detail": dict(detail or {})}

    def missing_cells(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            sorted(
                (path, level)
                for path in self.specs.paths
                for level in Level.ALL
                if (path, level) not in self.cells
                or self.cells[(path, level)]["status"] == CellStatus.MISSING
            )
        )

    @property
    def complete(self) -> bool:
        return not self.missing_cells()

    @property
    def failed_cells(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            sorted(key for key, value in self.cells.items() if value["status"] == CellStatus.FAIL)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.specs.as_dict(),
            "spec_digest": self.specs.digest(),
            "complete": self.complete,
            "missing": [list(item) for item in self.missing_cells()],
            "failed": [list(item) for item in self.failed_cells],
            "cells": [
                {"path": key[0], "level": key[1], **value}
                for key, value in sorted(self.cells.items())
            ],
        }


def path_matrix_table() -> str:
    """Human-readable path matrix (also used by the interface-map report)."""
    lines = ["| path | graph | operator | kernel | purpose |", "|---|---|---|---|---|"]
    for item in PATHS:
        lines.append(
            f"| `{item.path_id}` | {item.graph} | {item.operator} | {item.kernel} | {item.purpose} |"
        )
    return "\n".join(lines)


__all__ = [
    "FROZEN_ADD_RMSNORM_SEMANTICS",
    "MATRIX_COLUMNS",
    "PATHS",
    "CellStatus",
    "CorrectnessMatrix",
    "DifferentialPath",
    "DifferentialSpec",
    "DivergenceLocator",
    "DivergenceRecord",
    "FusionSemantics",
    "Level",
    "MetricReport",
    "ModeSwitchAudit",
    "ReadOrder",
    "StateSnapshot",
    "ToleranceRegistry",
    "ToleranceSpec",
    "compare_arrays",
    "compare_state",
    "composed_add_rms_norm_reference",
    "default_tolerance_registry",
    "fused_add_rms_norm_reference",
    "path_matrix_table",
    "reference_diff_summary",
]
