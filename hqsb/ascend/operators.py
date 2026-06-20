"""Frozen operator semantics, CPU oracles and the shared error/tolerance gate.

Three operators carry S09 (E09-02 Add / Row-Reduce-Sum, E09-03/04 RMSNorm).  The
point of this module is that **all implementations read one definition**:

* the equation, epsilon position, accumulation dtype, empty-tensor policy,
  alias policy and NaN/Inf behaviour are frozen in a spec object with a hash
  (E09-02 step 2, E09-03 §2);
* the golden values come from a CPU FP64 oracle written independently of any
  device path — the protocol forbids using "the same framework NPU op" as the
  only golden (E09-02 step 3);
* the error metric set and the tolerance are shared.  :class:`ToleranceSpec` is
  keyed by ``(operator, dtype)`` **only**: there is no parameter through which a
  backend could obtain a looser gate, which is how "do not relax the threshold
  for Ascend" (E09-03 §6, §11) is enforced structurally rather than by review.

Everything here is pure Python on purpose.  The oracle and the comparator must
run on a CPU-only host so the correctness *protocol* is verifiable at M2 even
while the device path stays ``BLOCKED``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.benchmark.metrics import percentile
from hqsb.core.errors import ConfigError, UsageError
from hqsb.core.fingerprint import canonical_json, sha256_hex

OPERATOR_ADD = "hqsb.ascend.add"
OPERATOR_ROW_REDUCE_SUM = "hqsb.ascend.row_reduce_sum"
OPERATOR_RMSNORM = "hqsb.rms_norm"

OPERATORS: Tuple[str, ...] = (OPERATOR_ADD, OPERATOR_ROW_REDUCE_SUM, OPERATOR_RMSNORM)

#: Implementation roles of E09-03 §3.  ``ascend_reference``/``ascend_optimized``
#: are the migration targets; the rest exist so a comparison has a spine.
IMPLEMENTATION_ROLES: Tuple[str, ...] = (
    "oracle",
    "framework_reference",
    "cuda_reference",
    "cuda_optimized",
    "triton",
    "ascend_reference",
    "ascend_optimized",
    "native_framework",
)

FP32 = "fp32"
FP16 = "fp16"
BF16 = "bf16"

REL_FLOOR_DEFAULT = 1e-6


# ── operator specs ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OperatorSemantics:
    """The frozen C3-level semantics one implementation must honour.

    ``equation`` is prose *and* ``eps_position`` is structural: putting epsilon
    inside the square root or before the division changes the operator, and the
    protocol calls that out as an error a numeric gate cannot hide (E09-03 §2).
    """

    operator: str
    equation: str
    eps_position: str
    input_dtypes: Tuple[str, ...]
    accum_dtypes: Tuple[str, ...]
    output_dtypes: Tuple[str, ...]
    layouts: Tuple[str, ...]
    empty_policy: str
    alias_policy: str
    broadcast_policy: str
    axis_policy: str
    non_finite_policy: str
    tolerance_id: str
    spec_version: str = "1.0.0"
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.operator not in OPERATORS:
            raise ConfigError(
                f"unknown operator {self.operator!r}; S09 freezes {list(OPERATORS)}",
                details={"operator": self.operator},
            )
        if not self.equation or not self.eps_position:
            raise ConfigError(
                f"{self.operator}: equation and eps_position must be explicit — 'the usual "
                "RMSNorm' is not a specification",
                details={"operator": self.operator},
            )
        for name in ("empty_policy", "alias_policy", "non_finite_policy", "tolerance_id"):
            if not getattr(self, name):
                raise ConfigError(
                    f"{self.operator}: {name} must be stated; an undefined case is a silent "
                    "wrong answer waiting to happen",
                    details={"operator": self.operator, "field": name},
                )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "operator": self.operator,
            "spec_version": self.spec_version,
            "equation": self.equation,
            "eps_position": self.eps_position,
            "input_dtypes": list(self.input_dtypes),
            "accum_dtypes": list(self.accum_dtypes),
            "output_dtypes": list(self.output_dtypes),
            "layouts": list(self.layouts),
            "empty_policy": self.empty_policy,
            "alias_policy": self.alias_policy,
            "broadcast_policy": self.broadcast_policy,
            "axis_policy": self.axis_policy,
            "non_finite_policy": self.non_finite_policy,
            "tolerance_id": self.tolerance_id,
            "notes": list(self.notes),
        }

    @property
    def spec_hash(self) -> str:
        return sha256_hex(canonical_json(self.as_dict()))


ADD_SPEC = OperatorSemantics(
    operator=OPERATOR_ADD,
    equation="y_i = x1_i + x2_i for i in [0, N)",
    eps_position="not_applicable",
    input_dtypes=(FP32, FP16, BF16),
    accum_dtypes=(FP32,),
    output_dtypes=(FP32, FP16, BF16),
    layouts=("contiguous_nd",),
    empty_policy="reject: N=0 is refused by the host tiling path, not returned as an empty tensor",
    alias_policy="reject: in-place/aliased inputs are UNSUPPORTED in v0",
    broadcast_policy="reject: v0 forbids broadcasting so stride semantics cannot leak in",
    axis_policy="not_applicable",
    non_finite_policy="IEEE propagation: NaN/Inf are preserved, never clamped",
    tolerance_id="tol.elementwise",
    notes=(
        "identical shape and dtype for x1/x2/y",
        "overflow follows the chosen dtype and the framework reference; no silent clamp",
    ),
)

ROW_REDUCE_SUM_SPEC = OperatorSemantics(
    operator=OPERATOR_ROW_REDUCE_SUM,
    equation="y_r = sum_j x[r, j] for r in [0, R), j in [0, H)",
    eps_position="not_applicable",
    input_dtypes=(FP32, FP16, BF16),
    accum_dtypes=(FP32,),
    output_dtypes=(FP32,),
    layouts=("contiguous_nd",),
    empty_policy="reject: H=0 is refused; the spec does not silently choose the mathematical zero",
    alias_policy="reject",
    broadcast_policy="not_applicable",
    axis_policy="last axis only; any other axis is UNSUPPORTED",
    non_finite_policy="IEEE propagation: a single NaN makes the row NaN",
    tolerance_id="tol.reduction.fp32_accum",
    notes=(
        "FP16/BF16 inputs accumulate in FP32 (declared here, not discovered later)",
        "float addition is not associative: tile/core order changes the result within tolerance",
        "output shape is [R] — one fixed schema, not '[R] or [R,1]' per backend",
    ),
)

RMSNORM_SPEC = OperatorSemantics(
    operator=OPERATOR_RMSNORM,
    equation=(
        "mean_square = (1/H) * sum_j x_j^2; "
        "rstd = 1 / sqrt(mean_square + eps); "
        "y_j = x_j * rstd * gamma_j"
    ),
    eps_position="after the mean of squares, before the square root",
    input_dtypes=(FP32, FP16, BF16),
    accum_dtypes=(FP32,),
    output_dtypes=(FP32, FP16, BF16),
    layouts=("contiguous_nd",),
    empty_policy="reject: H=0 has no defined mean square",
    alias_policy="reject: y must not alias x or gamma",
    broadcast_policy="gamma is per-hidden-element [H]; a scalar gamma is a different operator",
    axis_policy="reduction over the last axis only",
    non_finite_policy="IEEE propagation; NaN/Inf rows are reported separately, not averaged into the error stats",
    tolerance_id="tol.rmsnorm",
    notes=(
        "LayerNorm (mean subtraction) or sqrt(mean(x^2 + eps)) is NOT this operator",
        "only the model-visible output is returned; rstd is an intermediate",
    ),
)

SPECS: Mapping[str, OperatorSemantics] = {
    OPERATOR_ADD: ADD_SPEC,
    OPERATOR_ROW_REDUCE_SUM: ROW_REDUCE_SUM_SPEC,
    OPERATOR_RMSNORM: RMSNORM_SPEC,
}


def spec_for(operator: str) -> OperatorSemantics:
    try:
        return SPECS[operator]
    except KeyError:
        raise UsageError(
            f"no frozen spec for {operator!r}; S09 freezes {list(OPERATORS)} and does not "
            "accept an ad-hoc equation",
            details={"operator": operator},
        ) from None


# ── CPU oracles (FP64) ───────────────────────────────────────────────────────


def oracle_add(x1: Sequence[float], x2: Sequence[float]) -> List[float]:
    if len(x1) != len(x2):
        raise UsageError(
            f"add oracle needs equal lengths, got {len(x1)} and {len(x2)}; broadcasting is "
            "rejected by the spec, not implemented by the oracle",
            details={"len_x1": len(x1), "len_x2": len(x2)},
        )
    return [float(a) + float(b) for a, b in zip(x1, x2)]


def oracle_row_reduce_sum(rows: Sequence[Sequence[float]], *, kahan: bool = False) -> List[float]:
    """Row sums in FP64.  ``kahan=True`` adds a compensated variant for diagnosis.

    The compensated oracle exists because "error grows with H" must be compared
    against a numerical error model, not answered by loosening the gate
    (E09-02 step 3, E09-03 §10).
    """
    out: List[float] = []
    for row in rows:
        if kahan:
            total = 0.0
            compensation = 0.0
            for value in row:
                y = float(value) - compensation
                t = total + y
                compensation = (t - total) - y
                total = t
            out.append(total)
        else:
            out.append(math.fsum(float(value) for value in row))
    return out


def oracle_rmsnorm(
    rows: Sequence[Sequence[float]],
    gamma: Sequence[float],
    eps: float,
    *,
    return_intermediates: bool = False,
) -> Any:
    """FP64 RMSNorm with the epsilon position the spec freezes.

    ``math.fsum`` gives an exactly-rounded square sum, so the oracle's own
    summation error is not part of the measured error budget.
    """
    if eps <= 0:
        raise UsageError(f"eps must be positive, got {eps}", details={"eps": eps})
    outputs: List[List[float]] = []
    mean_squares: List[float] = []
    rstds: List[float] = []
    for row in rows:
        hidden = len(row)
        if hidden == 0:
            raise UsageError("H=0 has no defined mean square; the spec rejects it")
        if len(gamma) != hidden:
            raise UsageError(
                f"gamma length {len(gamma)} != hidden {hidden}; gamma is per-hidden-element",
                details={"gamma": len(gamma), "hidden": hidden},
            )
        mean_square = math.fsum(float(value) * float(value) for value in row) / hidden
        rstd = 1.0 / math.sqrt(mean_square + eps)
        outputs.append([float(value) * rstd * float(weight) for value, weight in zip(row, gamma)])
        mean_squares.append(mean_square)
        rstds.append(rstd)
    if return_intermediates:
        return outputs, mean_squares, rstds
    return outputs


def oracle_rmsnorm_analytic(*, rows: int, hidden: int, value: float, gamma_value: float, eps: float) -> List[List[float]]:
    """Closed form for a constant row — the audit of E09-03 step 5.

    For ``x_j = v`` and ``gamma_j = g``: ``mean_square = v^2``, so
    ``y_j = v * g / sqrt(v^2 + eps)``.  A golden that puts epsilon in the wrong
    place fails this check even though random inputs would hide it.
    """
    expected = value * gamma_value / math.sqrt(value * value + eps)
    return [[expected] * hidden for _ in range(rows)]


# ── error metrics ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ErrorMetrics:
    """The full metric set of details README §7.2 / E09-03 §6.

    ``max_rel`` alone is not a verdict: near-zero references inflate it, so the
    relative floor, the absolute errors and the first bad index travel with it.
    """

    count: int
    max_abs: float
    mean_abs: float
    rmse: float
    p99_abs: float
    max_rel: float
    rel_floor: float
    cosine: float
    nan_count: int
    inf_count: int
    first_bad_index: int
    first_bad_detail: Mapping[str, Any] = field(default_factory=dict)
    row_rms_before: Tuple[float, ...] = ()
    row_rms_after_gamma_removed: Tuple[float, ...] = ()
    non_finite_reference: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "rmse": self.rmse,
            "p99_abs": self.p99_abs,
            "max_rel": self.max_rel,
            "rel_floor": self.rel_floor,
            "cosine": self.cosine,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "first_bad_index": self.first_bad_index,
            "first_bad_detail": dict(self.first_bad_detail),
            "row_rms_before": list(self.row_rms_before),
            "row_rms_after_gamma_removed": list(self.row_rms_after_gamma_removed),
            "non_finite_reference": self.non_finite_reference,
        }


def _flatten(values: Any) -> List[float]:
    flat: List[float] = []
    for item in values:
        if isinstance(item, (list, tuple)):
            flat.extend(_flatten(item))
        else:
            flat.append(float(item))
    return flat


def error_metrics(
    actual: Any,
    reference: Any,
    *,
    rel_floor: float = REL_FLOOR_DEFAULT,
    rows_for_invariants: Optional[Sequence[Sequence[float]]] = None,
    gamma_for_invariants: Optional[Sequence[float]] = None,
) -> ErrorMetrics:
    """Compare two tensors element-wise and report the full metric set."""
    got = _flatten(actual)
    want = _flatten(reference)
    if len(got) != len(want):
        raise UsageError(
            f"cannot compare {len(got)} values against {len(want)}; a shape mismatch is a "
            "failure, not a zero-padded comparison",
            details={"actual": len(got), "reference": len(want)},
        )
    if not got:
        raise UsageError("refusing to score an empty comparison; the spec rejects empty tensors")
    if rel_floor <= 0:
        raise UsageError(f"rel_floor must be positive, got {rel_floor}", details={"rel_floor": rel_floor})

    abs_errors: List[float] = []
    rel_errors: List[float] = []
    nan_count = 0
    inf_count = 0
    non_finite_reference = 0
    first_bad_index = -1
    first_bad_detail: Dict[str, Any] = {}
    dot = 0.0
    norm_actual = 0.0
    norm_reference = 0.0
    finite_abs: List[float] = []

    for index, (a, b) in enumerate(zip(got, want)):
        if math.isnan(a):
            nan_count += 1
        if math.isinf(a):
            inf_count += 1
        if math.isnan(b) or math.isinf(b):
            non_finite_reference += 1
        both_finite = all(math.isfinite(value) for value in (a, b))
        if both_finite:
            difference = abs(a - b)
            abs_errors.append(difference)
            finite_abs.append(difference)
            rel_errors.append(difference / max(abs(b), rel_floor))
            dot += a * b
            norm_actual += a * a
            norm_reference += b * b
            if first_bad_index < 0 and difference > 0:
                first_bad_index = index
                first_bad_detail = {"actual": a, "reference": b, "abs": difference}
        elif math.isnan(a) != math.isnan(b) or (math.isinf(a) and a != b):
            # A finite-vs-non-finite disagreement is a real error, not a skipped
            # element: NaN==NaN must not become a silent pass.
            if first_bad_index < 0:
                first_bad_index = index
                first_bad_detail = {"actual": a, "reference": b, "abs": float("inf")}
            abs_errors.append(float("inf"))

    max_abs = max(abs_errors) if abs_errors else 0.0
    mean_abs = (sum(finite_abs) / len(finite_abs)) if finite_abs else 0.0
    rmse = math.sqrt(sum(value * value for value in finite_abs) / len(finite_abs)) if finite_abs else 0.0
    p99_abs = percentile(finite_abs, 0.99) if finite_abs else 0.0
    max_rel = max(rel_errors) if rel_errors else 0.0
    cosine = (
        dot / math.sqrt(norm_actual * norm_reference)
        if norm_actual > 0 and norm_reference > 0
        else (1.0 if norm_actual == 0 and norm_reference == 0 else 0.0)
    )

    row_rms_before: List[float] = []
    row_rms_after: List[float] = []
    if rows_for_invariants:
        for row in rows_for_invariants:
            row_rms_before.append(math.sqrt(math.fsum(float(v) * float(v) for v in row) / len(row)) if row else 0.0)
        if gamma_for_invariants:
            for row in rows_for_invariants:
                stripped = [float(v) / float(g) if g else float(v) for v, g in zip(row, gamma_for_invariants)]
                row_rms_after.append(
                    math.sqrt(math.fsum(value * value for value in stripped) / len(stripped)) if stripped else 0.0
                )

    return ErrorMetrics(
        count=len(got),
        max_abs=max_abs,
        mean_abs=mean_abs,
        rmse=rmse,
        p99_abs=p99_abs,
        max_rel=max_rel,
        rel_floor=rel_floor,
        cosine=cosine,
        nan_count=nan_count,
        inf_count=inf_count,
        first_bad_index=first_bad_index,
        first_bad_detail=first_bad_detail,
        row_rms_before=tuple(row_rms_before),
        row_rms_after_gamma_removed=tuple(row_rms_after),
        non_finite_reference=non_finite_reference,
    )


# ── preregistered tolerance ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ToleranceSpec:
    """One preregistered gate, keyed by operator and dtype only.

    There is deliberately **no** ``backend`` field and no ``scale``/``relax``
    parameter.  The only way to obtain a looser gate is to register a new
    ``tolerance_id`` with its own justification, which is a visible document
    change rather than a call-site argument (E09-03 §6/§11).
    """

    tolerance_id: str
    operator: str
    dtype: str
    max_abs: float
    mean_abs: float
    rmse: float
    max_rel: float
    rel_floor: float
    cosine_min: float
    max_nan: int = 0
    max_inf: int = 0
    justification: str = ""
    preregistered: bool = False

    def __post_init__(self) -> None:
        for name in ("tolerance_id", "operator", "dtype", "justification"):
            if not getattr(self, name):
                raise ConfigError(
                    f"a tolerance without {name!r} cannot be audited; 'whatever passes' is not a gate",
                    details={"field": name},
                )
        if not self.preregistered:
            raise ConfigError(
                f"tolerance {self.tolerance_id!r} is not marked preregistered; a threshold chosen "
                "after seeing the results is not a gate (handbook §5.6)",
                details={"tolerance_id": self.tolerance_id},
            )
        if self.rel_floor <= 0:
            raise ConfigError("rel_floor must be positive", details={"rel_floor": self.rel_floor})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tolerance_id": self.tolerance_id,
            "operator": self.operator,
            "dtype": self.dtype,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "rmse": self.rmse,
            "max_rel": self.max_rel,
            "rel_floor": self.rel_floor,
            "cosine_min": self.cosine_min,
            "max_nan": self.max_nan,
            "max_inf": self.max_inf,
            "justification": self.justification,
            "preregistered": self.preregistered,
        }

    @property
    def hash(self) -> str:
        return sha256_hex(canonical_json(self.as_dict()))


@dataclass(frozen=True)
class MetricVerdict:
    metric: str
    observed: float
    limit: float
    ok: bool
    comparator: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "observed": self.observed,
            "limit": self.limit,
            "ok": self.ok,
            "comparator": self.comparator,
        }


@dataclass(frozen=True)
class ToleranceVerdict:
    tolerance_id: str
    operator: str
    dtype: str
    implementation: str
    actual_backend: str
    passed: bool
    verdicts: Tuple[MetricVerdict, ...]
    metrics: Mapping[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tolerance_id": self.tolerance_id,
            "operator": self.operator,
            "dtype": self.dtype,
            "implementation": self.implementation,
            "actual_backend": self.actual_backend,
            "passed": self.passed,
            "failed_metrics": [item.metric for item in self.verdicts if not item.ok],
            "verdicts": [item.as_dict() for item in self.verdicts],
            "metrics": dict(self.metrics),
        }


def evaluate_tolerance(
    metrics: ErrorMetrics,
    spec: ToleranceSpec,
    *,
    implementation: str,
    actual_backend: str,
) -> ToleranceVerdict:
    """Apply one shared gate.  ``implementation``/``actual_backend`` are recorded
    for provenance only — they cannot influence the limits."""
    checks = (
        MetricVerdict("max_abs", metrics.max_abs, spec.max_abs, metrics.max_abs <= spec.max_abs, "<="),
        MetricVerdict("mean_abs", metrics.mean_abs, spec.mean_abs, metrics.mean_abs <= spec.mean_abs, "<="),
        MetricVerdict("rmse", metrics.rmse, spec.rmse, metrics.rmse <= spec.rmse, "<="),
        MetricVerdict("p99_abs", metrics.p99_abs, spec.max_abs, metrics.p99_abs <= spec.max_abs, "<="),
        MetricVerdict("max_rel", metrics.max_rel, spec.max_rel, metrics.max_rel <= spec.max_rel, "<="),
        MetricVerdict("cosine", metrics.cosine, spec.cosine_min, metrics.cosine >= spec.cosine_min, ">="),
        MetricVerdict("nan_count", float(metrics.nan_count), float(spec.max_nan), metrics.nan_count <= spec.max_nan, "<="),
        MetricVerdict("inf_count", float(metrics.inf_count), float(spec.max_inf), metrics.inf_count <= spec.max_inf, "<="),
    )
    if metrics.rel_floor != spec.rel_floor:
        raise UsageError(
            f"metrics were computed with rel_floor={metrics.rel_floor} but the gate declares "
            f"{spec.rel_floor}; max_rel is not comparable across floors",
            details={"metrics_rel_floor": metrics.rel_floor, "spec_rel_floor": spec.rel_floor},
        )
    return ToleranceVerdict(
        tolerance_id=spec.tolerance_id,
        operator=spec.operator,
        dtype=spec.dtype,
        implementation=implementation,
        actual_backend=actual_backend,
        passed=all(item.ok for item in checks),
        verdicts=checks,
        metrics=metrics.as_dict(),
    )


def tolerance_key(spec: ToleranceSpec) -> Tuple[str, str]:
    """The only two axes a gate may depend on."""
    return (spec.operator, spec.dtype)


def audit_no_backend_relaxation(specs: Iterable[ToleranceSpec]) -> Dict[str, Any]:
    """Prove one ``(operator, dtype)`` pair never has two different gates.

    This is the structural form of "do not relax the threshold for Ascend": if a
    second, looser gate appeared for the same pair, the audit fails.
    """
    seen: Dict[Tuple[str, str], ToleranceSpec] = {}
    conflicts: List[Dict[str, Any]] = []
    for spec in specs:
        key = tolerance_key(spec)
        previous = seen.get(key)
        if previous is None:
            seen[key] = spec
            continue
        if previous.as_dict() != spec.as_dict():
            conflicts.append(
                {
                    "operator": key[0],
                    "dtype": key[1],
                    "ids": sorted({previous.tolerance_id, spec.tolerance_id}),
                    "diff": {
                        field_name: (getattr(previous, field_name), getattr(spec, field_name))
                        for field_name in ("max_abs", "mean_abs", "rmse", "max_rel", "cosine_min")
                        if getattr(previous, field_name) != getattr(spec, field_name)
                    },
                }
            )
    return {"ok": not conflicts, "pairs": len(seen), "conflicts": conflicts}


# ── framework reference (lazy; torch is an optional extra) ───────────────────


def pytorch_reference_rmsnorm(
    rows: Sequence[Sequence[float]], gamma: Sequence[float], eps: float, dtype: str
) -> Dict[str, Any]:
    """``framework_reference`` role, imported lazily.

    Returns a structured ``NOT_AVAILABLE`` reason rather than raising, so a
    CPU-minimal install can still enumerate the implementation roster
    (E09-03 step 2: a missing implementation is ``NOT_AVAILABLE``, never a name
    standing in for something that does not exist).
    """
    try:
        import torch  # noqa: PLC0415 - optional extra, imported on demand
    except ImportError:
        return {
            "role": "framework_reference",
            "status": "NOT_AVAILABLE",
            "reason": "torch is not installed (optional extra 'benchmark'); the CPU oracle remains the golden",
        }
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    tensor = torch.tensor([list(row) for row in rows], dtype=torch_dtype)
    weights = torch.tensor(list(gamma), dtype=torch_dtype)
    hidden = tensor.shape[-1]
    variance = tensor.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    normalized = tensor * torch.rsqrt(variance + eps)
    output = (normalized * weights.to(torch.float32)).to(torch_dtype)
    return {
        "role": "framework_reference",
        "status": "computed",
        "torch_version": torch.__version__,
        "dtype": dtype,
        "hidden": int(hidden),
        "output": output.to(torch.float32).tolist(),
    }


def implementation_roster(availability: Mapping[str, str]) -> List[Dict[str, Any]]:
    """E09-03 step 2: list every role with its identity or ``NOT_AVAILABLE``."""
    roster: List[Dict[str, Any]] = []
    for role in IMPLEMENTATION_ROLES:
        identity = availability.get(role, "")
        roster.append(
            {
                "role": role,
                "status": "IDENTIFIED" if identity else "NOT_AVAILABLE",
                "identity": identity,
                "note": (
                    ""
                    if identity
                    else "a role without an artifact identity must stay NOT_AVAILABLE; a name is not an implementation"
                ),
            }
        )
    return roster


__all__ = [
    "ADD_SPEC",
    "BF16",
    "ErrorMetrics",
    "FP16",
    "FP32",
    "IMPLEMENTATION_ROLES",
    "MetricVerdict",
    "OPERATORS",
    "OPERATOR_ADD",
    "OPERATOR_ROW_REDUCE_SUM",
    "OPERATOR_RMSNORM",
    "OperatorSemantics",
    "REL_FLOOR_DEFAULT",
    "RMSNORM_SPEC",
    "ROW_REDUCE_SUM_SPEC",
    "SPECS",
    "ToleranceSpec",
    "ToleranceVerdict",
    "audit_no_backend_relaxation",
    "error_metrics",
    "evaluate_tolerance",
    "implementation_roster",
    "oracle_add",
    "oracle_rmsnorm",
    "oracle_rmsnorm_analytic",
    "oracle_row_reduce_sum",
    "pytorch_reference_rmsnorm",
    "spec_for",
    "tolerance_key",
]
