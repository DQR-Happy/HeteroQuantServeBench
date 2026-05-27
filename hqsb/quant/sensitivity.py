"""Layer sensitivity, outlier analysis and interaction (E05-05 §5–§8).

Four distinct questions must not be conflated (E05-05 §3):

1. **single-quant**: quantize unit *i* while everything else stays FP16 →
   direct damage;
2. **leave-one-out**: start from all-W4 and restore unit *i* to W8/FP16 →
   marginal recovery *in that failing context*;
3. **cumulative**: add low-bit units in a defined order (layer index, harm
   order, recovery-per-byte order, random controls) → path dependence;
4. **pairwise interaction**: ``loss(i,j) - loss(i) - loss(j) + loss(FP16)`` →
   whether a cost model may assume additivity.

The module also covers the outlier/weight-error views (weight RMSE, layer
output error, saturation, scale utilization, salient-channel agreement) and
the *measured* cost table, since a sensitivity ranking is worthless if the
cheap-looking restoration is not executable.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.quant.units import InterventionUnit

#: Intervention kinds (E05-05 §8).
SINGLE_QUANT = "single_quant"
LEAVE_ONE_OUT = "leave_one_out"
CUMULATIVE = "cumulative"
PAIRWISE = "pairwise"

INTERVENTION_KINDS = (SINGLE_QUANT, LEAVE_ONE_OUT, CUMULATIVE, PAIRWISE)

#: Cumulative orders (E05-05 §8.4). Random controls are required.
ORDER_LAYER = "layer_index"
ORDER_DIRECT_HARM = "direct_harm"
ORDER_RECOVERY_PER_BYTE = "recovery_per_byte"
ORDER_RANDOM = "random_control"


@dataclass
class UnitIntervention:
    """One planned intervention (a config for a unit, or a set of units)."""

    intervention_id: str
    kind: str
    unit_ids: Tuple[str, ...]
    configs: Tuple[Dict[str, Any], ...]
    order: str = ""
    seed: Optional[int] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in INTERVENTION_KINDS:
            raise ConfigError(
                f"unknown intervention kind {self.kind!r}; supported: "
                f"{list(INTERVENTION_KINDS)}"
            )
        if not self.unit_ids:
            raise ConfigError("an intervention must name at least one unit")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "kind": self.kind,
            "unit_ids": list(self.unit_ids),
            "configs": [dict(config) for config in self.configs],
            "order": self.order,
            "seed": self.seed,
            "notes": self.notes,
        }


@dataclass
class InterventionResult:
    """Measured outcome of one intervention (local + model level)."""

    intervention_id: str
    kind: str
    unit_ids: Tuple[str, ...]
    local_error: Dict[str, float] = field(default_factory=dict)
    logit_kl: float = float("nan")
    ppl_delta: float = float("nan")
    task_delta: float = float("nan")
    slice_deltas: Dict[str, float] = field(default_factory=dict)
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    artifact_bytes: int = 0
    device_bytes: int = 0
    phase_latency_ms: Dict[str, float] = field(default_factory=dict)
    observed_kernel: str = ""
    fallback_reason: str = ""
    gate_verdict: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "kind": self.kind,
            "unit_ids": list(self.unit_ids),
            "local_error": dict(self.local_error),
            "logit_kl": self.logit_kl,
            "ppl_delta": self.ppl_delta,
            "task_delta": self.task_delta,
            "slice_deltas": dict(self.slice_deltas),
            "ci": [self.ci_low, self.ci_high],
            "artifact_bytes": self.artifact_bytes,
            "device_bytes": self.device_bytes,
            "phase_latency_ms": dict(self.phase_latency_ms),
            "observed_kernel": self.observed_kernel,
            "fallback_reason": self.fallback_reason,
            "gate_verdict": self.gate_verdict,
        }


def plan_single_quant(
    units: Sequence[InterventionUnit], *, quant_config: Mapping[str, Any]
) -> List[UnitIntervention]:
    """One intervention per unit: everything FP16 except unit *i*."""
    return [
        UnitIntervention(
            intervention_id=f"single:{unit.unit_id}",
            kind=SINGLE_QUANT,
            unit_ids=(unit.unit_id,),
            configs=(dict(quant_config),),
            notes="all other units stay FP16",
        )
        for unit in sorted(units, key=lambda item: item.unit_id)
    ]


def plan_leave_one_out(
    units: Sequence[InterventionUnit],
    *,
    baseline_config: Mapping[str, Any],
    restore_config: Mapping[str, Any],
) -> List[UnitIntervention]:
    """One intervention per unit: all-W4 except unit *i* restored."""
    return [
        UnitIntervention(
            intervention_id=f"loo:{unit.unit_id}",
            kind=LEAVE_ONE_OUT,
            unit_ids=(unit.unit_id,),
            configs=(dict(restore_config),),
            notes=f"baseline = all units at {dict(baseline_config)}; this unit restored",
        )
        for unit in sorted(units, key=lambda item: item.unit_id)
    ]


def plan_cumulative(
    units: Sequence[InterventionUnit],
    *,
    quant_config: Mapping[str, Any],
    order: str,
    scores: Optional[Mapping[str, float]] = None,
    seed: int = 0,
) -> List[UnitIntervention]:
    """Cumulative path: quantize units one at a time in ``order``.

    ``order`` semantics:

    * ``layer_index`` — ascending layer then unit id (the natural path);
    * ``direct_harm`` — ascending direct-harm score (least harmful first);
    * ``recovery_per_byte`` — descending recovery per extra byte (cheapest
      recovery first, i.e. restore order reversed for quantization);
    * ``random_control`` — a seeded shuffle, repeated by the caller with
      several seeds (E05-05 §8.4 requires random controls).
    """
    ordered = sorted(units, key=lambda item: (item.layer_index or 0, item.unit_id))
    if order == ORDER_DIRECT_HARM:
        if scores is None:
            raise ConfigError("direct_harm order requires a scores mapping")
        ordered = sorted(
            units, key=lambda item: (scores.get(item.unit_id, 0.0), item.unit_id)
        )
    elif order == ORDER_RECOVERY_PER_BYTE:
        if scores is None:
            raise ConfigError("recovery_per_byte order requires a scores mapping")
        ordered = sorted(
            units,
            key=lambda item: (-scores.get(item.unit_id, 0.0), item.unit_id),
        )
    elif order == ORDER_RANDOM:
        ordered = list(units)
        random.Random(seed).shuffle(ordered)
    elif order != ORDER_LAYER:
        raise ConfigError(
            f"unknown cumulative order {order!r}; supported: "
            f"{[ORDER_LAYER, ORDER_DIRECT_HARM, ORDER_RECOVERY_PER_BYTE, ORDER_RANDOM]}"
        )
    interventions: List[UnitIntervention] = []
    for step in range(1, len(ordered) + 1):
        subset = ordered[:step]
        interventions.append(
            UnitIntervention(
                intervention_id=f"cumulative:{order}:{step:03d}",
                kind=CUMULATIVE,
                unit_ids=tuple(unit.unit_id for unit in subset),
                configs=(dict(quant_config),),
                order=order,
                seed=seed if order == ORDER_RANDOM else None,
                notes=f"first {step} units quantized in {order} order",
            )
        )
    return interventions


def plan_pairwise(
    units: Sequence[InterventionUnit],
    *,
    quant_config: Mapping[str, Any],
    top_k: int,
    random_control_pairs: int = 2,
    seed: int = 0,
) -> List[UnitIntervention]:
    """Pairwise interventions over the top-K units plus random controls.

    The random controls are what make a "non-additivity" claim interpretable:
    without them, a large interaction on the top pairs could just be the
    typical interaction size.
    """
    ordered = list(units)
    top = ordered[:top_k]
    rng = random.Random(seed)
    controls: List[Tuple[InterventionUnit, InterventionUnit]] = []
    if len(ordered) >= 2:
        for _ in range(random_control_pairs):
            left, right = rng.sample(ordered, 2)
            controls.append((left, right))
    interventions: List[UnitIntervention] = []
    for index, left in enumerate(top):
        for right in top[index + 1 :]:
            interventions.append(
                UnitIntervention(
                    intervention_id=f"pair:{left.unit_id}|{right.unit_id}",
                    kind=PAIRWISE,
                    unit_ids=(left.unit_id, right.unit_id),
                    configs=(dict(quant_config),),
                    notes="top-K pair",
                )
            )
    for left, right in controls:
        interventions.append(
            UnitIntervention(
                intervention_id=f"pair-control:{left.unit_id}|{right.unit_id}",
                kind=PAIRWISE,
                unit_ids=(left.unit_id, right.unit_id),
                configs=(dict(quant_config),),
                seed=seed,
                notes="random control pair",
            )
        )
    return interventions


# ── error views (E05-05 §5/§7) ────────────────────────────────────────────


def weight_error_metrics(original: Sequence[float], quantized: Sequence[float]) -> Dict[str, float]:
    """Weight-space reconstruction metrics (representation error only)."""
    from hqsb.benchmark.metrics import numerical_diff_summary

    if len(original) != len(quantized):
        raise ConfigError(
            f"weight vectors must match in length, got {len(original)}/{len(quantized)}"
        )
    summary = numerical_diff_summary(
        [float(value) for value in original], [float(value) for value in quantized]
    )
    if not summary:
        return {}
    summary["relative_rmse"] = summary["rmse"] / (
        math.sqrt(
            sum(float(value) ** 2 for value in original) / max(1, len(original))
        )
        or 1.0
    )
    return summary


def layer_output_error(
    activations: Sequence[float],
    weight_delta: Sequence[float],
    *,
    rows: int,
    cols: int,
) -> Dict[str, float]:
    """``ΔY = X(W_hat - W)`` energy and its per-row worst (E05-05 §5.2).

    Computed in float64 with pure Python so it is usable as an oracle before
    any kernel exists. ``energy_ratio`` relates the error energy to the
    activation energy, which is what makes "small ΔW but large ΔY" visible.
    """
    if len(activations) % cols != 0:
        raise ConfigError(
            f"activation length {len(activations)} is not a multiple of K={cols}"
        )
    if len(weight_delta) != rows * cols:
        raise ConfigError(
            f"weight delta length {len(weight_delta)} != rows*cols={rows * cols}"
        )
    m = len(activations) // cols
    worst_row = 0.0
    total_energy = 0.0
    activation_energy = 0.0
    per_row: List[float] = []
    for row in range(m):
        x_row = activations[row * cols : (row + 1) * cols]
        activation_energy += sum(value * value for value in x_row)
        row_error = 0.0
        for n in range(rows):
            delta_row = weight_delta[n * cols : (n + 1) * cols]
            value = sum(a * b for a, b in zip(x_row, delta_row))
            row_error += value * value
            total_energy += value * value
        per_row.append(math.sqrt(row_error))
        worst_row = max(worst_row, math.sqrt(row_error))
    return {
        "error_l2": math.sqrt(total_energy),
        "error_rms": math.sqrt(total_energy / max(1, m * rows)),
        "worst_row_l2": worst_row,
        "activation_l2": math.sqrt(activation_energy),
        "energy_ratio": (
            math.sqrt(total_energy) / math.sqrt(activation_energy)
            if activation_energy
            else float("nan")
        ),
        "rows": m,
    }


def interaction(
    loss_both: float, loss_left: float, loss_right: float, loss_none: float
) -> float:
    """Second-order interaction (E05-05 §8.5).

    ``interaction(i,j) = loss(i,j) - loss(i) - loss(j) + loss(FP16)``.
    A near-zero value supports an additive cost model; a large one does not.
    """
    return loss_both - loss_left - loss_right + loss_none


def interaction_table(
    results: Sequence[InterventionResult],
    single_losses: Mapping[str, float],
    baseline_loss: float,
) -> List[Dict[str, Any]]:
    """Compute interactions for every pairwise result."""
    rows: List[Dict[str, Any]] = []
    for result in results:
        if result.kind != PAIRWISE or len(result.unit_ids) != 2:
            continue
        left, right = result.unit_ids
        if left not in single_losses or right not in single_losses:
            rows.append(
                {
                    "unit_i": left,
                    "unit_j": right,
                    "expected_additive": None,
                    "observed": result.ppl_delta,
                    "interaction": None,
                    "reason": "missing single-unit loss for one of the pair",
                }
            )
            continue
        expected = single_losses[left] + single_losses[right]
        observed = result.ppl_delta
        rows.append(
            {
                "unit_i": left,
                "unit_j": right,
                "expected_additive": expected,
                "observed": observed,
                "interaction": interaction(
                    observed + baseline_loss,
                    single_losses[left] + baseline_loss,
                    single_losses[right] + baseline_loss,
                    baseline_loss,
                ),
            }
        )
    return rows


def sensitivity_ranking(results: Sequence[InterventionResult]) -> List[Dict[str, Any]]:
    """Rank units by direct harm / marginal recovery / recovery per byte.

    The ranking is *descriptive*; it is never used as the policy by itself
    (E05-05 §9: the policy is a constrained optimisation over the measured
    cost table).
    """
    rows: List[Dict[str, Any]] = []
    for result in results:
        if result.kind not in (SINGLE_QUANT, LEAVE_ONE_OUT):
            continue
        rows.append(
            {
                "unit_id": result.unit_ids[0],
                "kind": result.kind,
                "ppl_delta": result.ppl_delta,
                "logit_kl": result.logit_kl,
                "ci": [result.ci_low, result.ci_high],
                "artifact_bytes": result.artifact_bytes,
                "gate_verdict": result.gate_verdict,
            }
        )
    return rows


def recovery_per_byte(results: Sequence[InterventionResult]) -> Dict[str, float]:
    """Recovery per extra artifact byte, used as a cumulative order score.

    Only leave-one-out results with a *positive* measured recovery and a
    positive extra-byte cost contribute; others are absent (not zero), so a
    missing measurement cannot masquerade as a cheap restoration.
    """
    scores: Dict[str, float] = {}
    for result in results:
        if result.kind != LEAVE_ONE_OUT or result.artifact_bytes <= 0:
            continue
        recovery = -result.ppl_delta  # PPL lower is better → positive = recovered
        if recovery <= 0:
            continue
        scores[result.unit_ids[0]] = recovery / result.artifact_bytes
    return scores


@dataclass
class CostRow:
    """Measured cost of one unit configuration (E05-05 §9.1)."""

    unit_id: str
    config: Dict[str, Any]
    artifact_bytes: int
    device_bytes: int
    prefill_latency_ms: float = float("nan")
    decode_latency_ms: float = float("nan")
    observed_kernel: str = ""
    fallback_reason: str = ""
    kernel_switch_count: int = 0
    workspace_bytes: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "config": dict(self.config),
            "artifact_bytes": self.artifact_bytes,
            "device_bytes": self.device_bytes,
            "prefill_latency_ms": self.prefill_latency_ms,
            "decode_latency_ms": self.decode_latency_ms,
            "observed_kernel": self.observed_kernel,
            "fallback_reason": self.fallback_reason,
            "kernel_switch_count": self.kernel_switch_count,
            "workspace_bytes": self.workspace_bytes,
        }


def cost_table_json(rows: Sequence[CostRow]) -> str:
    return json.dumps(
        [row.as_dict() for row in rows], sort_keys=True, indent=2, ensure_ascii=False
    )


def surrogate_validation(
    predicted: Sequence[float], observed: Sequence[float]
) -> Dict[str, Any]:
    """Validate a surrogate (additive) cost/quality model (E05-05 §10.3).

    Reports prediction error, rank correlation, worst-case residual and the
    pairs where additivity fails. A surrogate that cannot meet the
    pre-registered error bar must not be used to select a policy.
    """
    from hqsb.quant.stats import summarize_distribution

    if len(predicted) != len(observed):
        raise ConfigError(
            f"predicted ({len(predicted)}) and observed ({len(observed)}) must match"
        )
    residuals = [float(p) - float(o) for p, o in zip(predicted, observed)]
    summary = summarize_distribution(residuals)
    ranks_predicted = _ranks(predicted)
    ranks_observed = _ranks(observed)
    n = len(predicted)
    if n < 2:
        rank_correlation = float("nan")
    else:
        mean_p = sum(ranks_predicted) / n
        mean_o = sum(ranks_observed) / n
        numerator = sum(
            (a - mean_p) * (b - mean_o)
            for a, b in zip(ranks_predicted, ranks_observed)
        )
        denominator = math.sqrt(
            sum((a - mean_p) ** 2 for a in ranks_predicted)
            * sum((b - mean_o) ** 2 for b in ranks_observed)
        )
        rank_correlation = numerator / denominator if denominator else float("nan")
    worst_index = max(range(n), key=lambda index: abs(residuals[index])) if n else -1
    return {
        "n": n,
        "mean_residual": summary.mean,
        "rmse": summary.std,
        "max_abs_residual": summary.absmax,
        "worst_index": worst_index,
        "rank_correlation": rank_correlation,
        "residuals": residuals,
    }


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    for position, index in enumerate(order):
        ranks[index] = float(position)
    return ranks


__all__ = [
    "CUMULATIVE",
    "CostRow",
    "INTERVENTION_KINDS",
    "InterventionResult",
    "LEAVE_ONE_OUT",
    "ORDER_DIRECT_HARM",
    "ORDER_LAYER",
    "ORDER_RANDOM",
    "ORDER_RECOVERY_PER_BYTE",
    "PAIRWISE",
    "SINGLE_QUANT",
    "UnitIntervention",
    "cost_table_json",
    "interaction",
    "interaction_table",
    "layer_output_error",
    "plan_cumulative",
    "plan_leave_one_out",
    "plan_pairwise",
    "plan_single_quant",
    "recovery_per_byte",
    "sensitivity_ranking",
    "surrogate_validation",
    "weight_error_metrics",
]
