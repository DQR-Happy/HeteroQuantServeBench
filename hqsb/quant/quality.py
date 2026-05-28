"""Model-level quality evidence: logits, tokens, gates (E05-02 §7).

Four minimum levels are required (operator → block → logits → autoregressive),
and within the logits level "cosine alone is not enough": a high cosine can
still flip the decision margin. This module computes the full pre-registered
metric set:

* logit max absolute error, RMSE / normalized RMSE, cosine;
* KL and Jensen–Shannon divergence between the FP16 and quantized
  next-token distributions;
* top-1/top-k overlap, top-1 vs top-2 margin change, top-1 flips;
* first divergence position for a teacher-forced sequence;
* per-step records for cache-enabled autoregressive generation
  (context length, KV bytes, per-step KL/margin/token, NaN/Inf), so error
  accumulation is visible step by step.

Quality gates are vector constraints: a failing slice cannot be averaged
away, and a confidence interval crossing the pre-registered margin yields
``INCONCLUSIVE`` (never PASS) — see :func:`evaluate_quality_gate`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.quant import stats as qstats

#: Metric directions (E05-10 §6.1).
DIRECTIONS = {
    "logit_max_abs_error": qstats.LOWER_IS_BETTER,
    "logit_rmse": qstats.LOWER_IS_BETTER,
    "logit_normalized_rmse": qstats.LOWER_IS_BETTER,
    "logit_cosine": qstats.HIGHER_IS_BETTER,
    "kl_fp16_quant": qstats.LOWER_IS_BETTER,
    "js_divergence": qstats.LOWER_IS_BETTER,
    "top1_overlap": qstats.HIGHER_IS_BETTER,
    "topk_overlap": qstats.HIGHER_IS_BETTER,
    "margin_delta": qstats.LOWER_IS_BETTER,
    "top1_flip_rate": qstats.LOWER_IS_BETTER,
    "perplexity": qstats.LOWER_IS_BETTER,
    "task_accuracy": qstats.HIGHER_IS_BETTER,
    "first_divergence_position": qstats.HIGHER_IS_BETTER,
}


def _to_float_list(row) -> List[float]:
    if hasattr(row, "tolist"):
        return [float(value) for value in row.detach().reshape(-1).tolist()]
    return [float(value) for value in row]


def _softmax(row: Sequence[float]) -> List[float]:
    if not row:
        return []
    maximum = max(row)
    exps = [math.exp(value - maximum) for value in row]
    total = math.fsum(exps)
    if total == 0.0:  # pragma: no cover - exp cannot underflow to all-zero here
        return [0.0] * len(row)
    return [value / total for value in exps]


def _log_softmax(row: Sequence[float]) -> List[float]:
    if not row:
        return []
    maximum = max(row)
    shifted = [value - maximum for value in row]
    log_sum = math.log(math.fsum(math.exp(value) for value in shifted))
    return [value - log_sum for value in shifted]


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """``KL(p || q)`` in nats, computed stably from log-probabilities."""
    if len(p) != len(q):
        raise ConfigError(f"KL needs equal-length distributions, got {len(p)}/{len(q)}")
    log_p = _log_softmax(list(p))
    log_q = _log_softmax(list(q))
    total = 0.0
    for lp, lq in zip(log_p, log_q):
        probability = math.exp(lp)
        if probability == 0.0:
            continue
        total += probability * (lp - lq)
    return total


def jensen_shannon(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen–Shannon divergence (symmetric, bounded by ``ln 2``)."""
    if len(p) != len(q):
        raise ConfigError(f"JS needs equal-length distributions, got {len(p)}/{len(q)}")
    log_p = _log_softmax(list(p))
    log_q = _log_softmax(list(q))
    log_m = [
        math.log(0.5 * (math.exp(lp) + math.exp(lq))) for lp, lq in zip(log_p, log_q)
    ]
    left = 0.0
    right = 0.0
    for lp, lq, lm in zip(log_p, log_q, log_m):
        p_probability = math.exp(lp)
        q_probability = math.exp(lq)
        if p_probability:
            left += 0.5 * p_probability * (lp - lm)
        if q_probability:
            right += 0.5 * q_probability * (lq - lm)
    return left + right


def topk_indices(row: Sequence[float], k: int) -> List[int]:
    """Indices of the ``k`` largest values (ties broken by lower index)."""
    if k <= 0:
        raise ConfigError(f"k must be positive, got {k}")
    return sorted(range(len(row)), key=lambda index: (-row[index], index))[:k]


def overlap(left: Sequence[int], right: Sequence[int]) -> float:
    """Overlap fraction of two index sets relative to the requested size."""
    if not left:
        return float("nan")
    left_set, right_set = set(left), set(right)
    return len(left_set & right_set) / len(left_set)


def margin(row: Sequence[float]) -> float:
    """Top-1 minus top-2 logit (decision margin)."""
    if len(row) < 2:
        return float("nan")
    ordered = sorted(row, reverse=True)
    return ordered[0] - ordered[1]


@dataclass
class PositionMetrics:
    """Logit metrics at one sequence position."""

    position: int
    sample_id: str = ""
    logit_max_abs_error: float = float("nan")
    logit_rmse: float = float("nan")
    logit_normalized_rmse: float = float("nan")
    logit_cosine: float = float("nan")
    kl_fp16_quant: float = float("nan")
    js_divergence: float = float("nan")
    top1_overlap: float = float("nan")
    topk_overlap: float = float("nan")
    margin_baseline: float = float("nan")
    margin_candidate: float = float("nan")
    margin_delta: float = float("nan")
    top1_flipped: bool = False
    slice: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "sample_id": self.sample_id,
            "logit_max_abs_error": self.logit_max_abs_error,
            "logit_rmse": self.logit_rmse,
            "logit_normalized_rmse": self.logit_normalized_rmse,
            "logit_cosine": self.logit_cosine,
            "kl_fp16_quant": self.kl_fp16_quant,
            "js_divergence": self.js_divergence,
            "top1_overlap": self.top1_overlap,
            "topk_overlap": self.topk_overlap,
            "margin_baseline": self.margin_baseline,
            "margin_candidate": self.margin_candidate,
            "margin_delta": self.margin_delta,
            "top1_flipped": self.top1_flipped,
            "slice": self.slice,
        }


def position_metrics(
    baseline_row: Sequence[float],
    candidate_row: Sequence[float],
    *,
    position: int = 0,
    sample_id: str = "",
    top_k: int = 5,
    slice_name: str = "",
) -> PositionMetrics:
    """Compute the full per-position metric set (E05-02 §7.2)."""
    base = _to_float_list(baseline_row)
    cand = _to_float_list(candidate_row)
    if len(base) != len(cand):
        raise ConfigError(
            f"logits must have the same vocabulary size, got {len(base)}/{len(cand)}"
        )
    from hqsb.benchmark.metrics import numerical_diff_summary

    summary = numerical_diff_summary(base, cand)
    base_topk = topk_indices(base, top_k)
    cand_topk = topk_indices(cand, top_k)
    base_top1 = topk_indices(base, 1)[0]
    cand_top1 = topk_indices(cand, 1)[0]
    norm = math.sqrt(math.fsum(value * value for value in base)) or 1.0
    return PositionMetrics(
        position=position,
        sample_id=sample_id,
        logit_max_abs_error=summary["max_abs_error"],
        logit_rmse=summary["rmse"],
        logit_normalized_rmse=summary["rmse"] / norm,
        logit_cosine=summary["cosine_similarity"],
        kl_fp16_quant=kl_divergence(base, cand),
        js_divergence=jensen_shannon(base, cand),
        top1_overlap=1.0 if base_top1 == cand_top1 else 0.0,
        topk_overlap=overlap(base_topk, cand_topk),
        margin_baseline=margin(base),
        margin_candidate=margin(cand),
        margin_delta=abs(margin(cand) - margin(base)),
        top1_flipped=base_top1 != cand_top1,
        slice=slice_name,
    )


def teacher_forcing_report(
    baseline_logits: Sequence[Sequence[float]],
    candidate_logits: Sequence[Sequence[float]],
    *,
    sample_id: str = "",
    top_k: int = 5,
    slice_name: str = "",
) -> Dict[str, Any]:
    """Per-position report over one teacher-forced sequence (E05-02 §7.2).

    Uses the *same* token sequence for both models, so a divergence cannot
    hide behind autoregressive feedback. Returns every position record plus
    the first divergence position (``-1`` when none) and per-metric means.
    """
    if len(baseline_logits) != len(candidate_logits):
        raise ConfigError(
            f"teacher-forcing needs equal-length sequences, got "
            f"{len(baseline_logits)}/{len(candidate_logits)}"
        )
    records: List[PositionMetrics] = []
    for index, (base_row, cand_row) in enumerate(
        zip(baseline_logits, candidate_logits)
    ):
        records.append(
            position_metrics(
                base_row,
                cand_row,
                position=index,
                sample_id=sample_id,
                top_k=top_k,
                slice_name=slice_name,
            )
        )
    first_divergence = next(
        (
            record.position
            for record in records
            if record.top1_flipped
        ),
        -1,
    )
    numeric_fields = [
        "logit_max_abs_error",
        "logit_rmse",
        "logit_cosine",
        "kl_fp16_quant",
        "topk_overlap",
        "margin_delta",
    ]
    means = {}
    for field_name in numeric_fields:
        values = [getattr(record, field_name) for record in records]
        values = [value for value in values if not math.isnan(value)]
        means[field_name] = (math.fsum(values) / len(values)) if values else float("nan")
    return {
        "sample_id": sample_id,
        "slice": slice_name,
        "positions": len(records),
        "first_divergence_position": first_divergence,
        "top1_flip_count": sum(1 for record in records if record.top1_flipped),
        "means": means,
        "records": [record.as_dict() for record in records],
    }


@dataclass
class GenerationStep:
    """One decode step of a cache-enabled generation (E05-08 §9.3)."""

    step: int
    context_length: int
    token: int
    kv_bytes: int
    kl_fp16_quant: float = float("nan")
    margin_baseline: float = float("nan")
    margin_candidate: float = float("nan")
    logit_max_abs_error: float = float("nan")
    tpot_ms: float = float("nan")
    quant_dequant_ms: float = float("nan")
    nan_seen: bool = False
    inf_seen: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "context_length": self.context_length,
            "token": self.token,
            "kv_bytes": self.kv_bytes,
            "kl_fp16_quant": self.kl_fp16_quant,
            "margin_baseline": self.margin_baseline,
            "margin_candidate": self.margin_candidate,
            "logit_max_abs_error": self.logit_max_abs_error,
            "tpot_ms": self.tpot_ms,
            "quant_dequant_ms": self.quant_dequant_ms,
            "nan_seen": self.nan_seen,
            "inf_seen": self.inf_seen,
        }


def generation_report(
    steps: Sequence[GenerationStep],
    *,
    baseline_tokens: Sequence[int] = (),
    candidate_tokens: Sequence[int] = (),
) -> Dict[str, Any]:
    """Summarize an autoregressive run: divergence, NaN/Inf, quality flags.

    ``degeneration`` flags a repetition collapse (the same token 8+ times in a
    row over the last 32 steps) and ``early_stop`` reports a length below the
    requested one. Both are diagnostics, not verdicts.
    """
    if not steps:
        return {
            "steps": 0,
            "first_divergence_step": -1,
            "mean_kl": float("nan"),
            "nan_count": 0,
            "inf_count": 0,
            "kv_bytes_final": 0,
        }
    first_divergence = -1
    if baseline_tokens and candidate_tokens:
        for index, (left, right) in enumerate(zip(baseline_tokens, candidate_tokens)):
            if left != right:
                first_divergence = index
                break
    kl_values = [
        step.kl_fp16_quant for step in steps if not math.isnan(step.kl_fp16_quant)
    ]
    tokens = list(candidate_tokens) if candidate_tokens else [step.token for step in steps]
    degeneration = False
    if len(tokens) >= 8:
        tail = tokens[-32:]
        if len(set(tail[-8:])) == 1 and tail.count(tail[-1]) >= 8:
            degeneration = True
    return {
        "steps": len(steps),
        "first_divergence_step": first_divergence,
        "mean_kl": (math.fsum(kl_values) / len(kl_values)) if kl_values else float("nan"),
        "max_kl": max(kl_values) if kl_values else float("nan"),
        "nan_count": sum(1 for step in steps if step.nan_seen),
        "inf_count": sum(1 for step in steps if step.inf_seen),
        "kv_bytes_final": steps[-1].kv_bytes,
        "degeneration_suspected": degeneration,
        "records": [step.as_dict() for step in steps],
    }


# ── quality gate (E05-02 §7.4, E05-10 §5.3/§6) ────────────────────────────


@dataclass
class QualityBudget:
    """Pre-registered quality gate definition.

    ``margins`` maps a metric name to its allowed degradation; ``slices``
    lists the slice names that each must individually pass. ``hard_safety``
    lists checks that are absolute (NaN/Inf, illegal token, crash).
    """

    name: str
    margins: Dict[str, float]
    slices: Tuple[str, ...] = ()
    hard_safety: Tuple[str, ...] = ("no_nan_inf", "no_illegal_token", "no_crash")
    confidence: float = qstats.DEFAULT_CONFIDENCE
    resamples: int = qstats.DEFAULT_RESAMPLES
    seed: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "margins": dict(sorted(self.margins.items())),
            "slices": list(self.slices),
            "hard_safety": list(self.hard_safety),
            "confidence": self.confidence,
            "resamples": self.resamples,
            "seed": self.seed,
        }


def evaluate_quality_gate(
    budget: QualityBudget,
    *,
    paired_metrics: Mapping[str, Tuple[Sequence[float], Sequence[float]]],
    slice_metrics: Optional[
        Mapping[str, Mapping[str, Tuple[Sequence[float], Sequence[float]]]]
    ] = None,
    safety: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Any]:
    """Evaluate the vector quality gate.

    Args:
        budget: the pre-registered budget.
        paired_metrics: ``{metric: (baseline_values, candidate_values)}`` on
            paired items.
        slice_metrics: ``{slice: {metric: (baseline, candidate)}}`` — each
            listed slice must pass on its own.
        safety: ``{check: ok}`` for the hard-safety items.

    Returns a machine-readable report with a verdict per metric/slice, the
    overall verdict and the list of reasons. The overall verdict is ``FAIL``
    if any hard-safety check or any required metric fails, ``INCONCLUSIVE``
    if nothing failed but something is inconclusive, otherwise ``PASS``.
    """
    safety = dict(safety or {})
    missing_safety = [
        check for check in budget.hard_safety if check not in safety
    ]
    metric_results: Dict[str, Any] = {}
    verdicts: List[str] = []
    for metric, (baseline, candidate) in sorted(paired_metrics.items()):
        if metric not in DIRECTIONS:
            raise ConfigError(
                f"metric {metric!r} has no registered direction; refusing to "
                f"guess whether higher or lower is better"
            )
        direction = DIRECTIONS[metric]
        if metric not in budget.margins:
            metric_results[metric] = {
                "verdict": qstats.GATE_INCONCLUSIVE,
                "reason": "metric has no pre-registered margin in the budget",
            }
            verdicts.append(qstats.GATE_INCONCLUSIVE)
            continue
        verdict = qstats.non_inferiority_verdict(
            metric,
            direction,
            baseline,
            candidate,
            budget.margins[metric],
            confidence=budget.confidence,
            resamples=budget.resamples,
            seed=budget.seed,
        )
        metric_results[metric] = verdict.as_dict()
        verdicts.append(verdict.verdict)

    slice_results: Dict[str, Any] = {}
    if budget.slices:
        provided = dict(slice_metrics or {})
        for slice_name in budget.slices:
            if slice_name not in provided:
                slice_results[slice_name] = {
                    "verdict": qstats.GATE_INCONCLUSIVE,
                    "reason": "slice not measured",
                }
                verdicts.append(qstats.GATE_INCONCLUSIVE)
                continue
            slice_verdicts = []
            details: Dict[str, Any] = {}
            for metric, (baseline, candidate) in sorted(provided[slice_name].items()):
                margin = budget.margins.get(metric)
                if margin is None:
                    details[metric] = {
                        "verdict": qstats.GATE_INCONCLUSIVE,
                        "reason": "no pre-registered margin",
                    }
                    slice_verdicts.append(qstats.GATE_INCONCLUSIVE)
                    continue
                direction = DIRECTIONS.get(metric)
                if direction is None:
                    details[metric] = {
                        "verdict": qstats.GATE_INCONCLUSIVE,
                        "reason": "metric direction not registered",
                    }
                    slice_verdicts.append(qstats.GATE_INCONCLUSIVE)
                    continue
                verdict = qstats.non_inferiority_verdict(
                    metric,
                    direction,
                    baseline,
                    candidate,
                    margin,
                    confidence=budget.confidence,
                    resamples=budget.resamples,
                    seed=budget.seed,
                )
                details[metric] = verdict.as_dict()
                slice_verdicts.append(verdict.verdict)
            worst = _worst_verdict(slice_verdicts)
            slice_results[slice_name] = {"verdict": worst, "metrics": details}
            verdicts.append(worst)

    safety_failed = [check for check, ok in safety.items() if not ok]
    overall = _worst_verdict(verdicts) if verdicts else qstats.GATE_INCONCLUSIVE
    if safety_failed or missing_safety:
        overall = qstats.GATE_FAIL
    reasons: List[str] = []
    if safety_failed:
        reasons.append(f"hard safety failed: {sorted(safety_failed)}")
    if missing_safety:
        reasons.append(f"hard safety not measured: {sorted(missing_safety)}")
    reasons.extend(
        f"{metric}: {result['verdict']}"
        for metric, result in sorted(metric_results.items())
        if result["verdict"] != qstats.GATE_PASS
    )
    reasons.extend(
        f"slice {name}: {result['verdict']}"
        for name, result in sorted(slice_results.items())
        if result["verdict"] != qstats.GATE_PASS
    )
    return {
        "budget": budget.as_dict(),
        "overall_verdict": overall,
        "metrics": metric_results,
        "slices": slice_results,
        "safety": safety,
        "reasons": reasons,
        "note": (
            "INCONCLUSIVE is not a pass and must not be reported as one; add "
            "samples or narrow the claim"
        ),
    }


def _worst_verdict(verdicts: Iterable[str]) -> str:
    order = {qstats.GATE_PASS: 0, qstats.GATE_INCONCLUSIVE: 1, qstats.GATE_FAIL: 2}
    worst = qstats.GATE_PASS
    for verdict in verdicts:
        if order.get(verdict, 1) > order[worst]:
            worst = verdict
    return worst


def worst_slice_report(gate_report: Mapping[str, Any]) -> Optional[str]:
    """Name of the worst-performing slice (for the "worst-case" reporting rule)."""
    slices = gate_report.get("slices") or {}
    order = {qstats.GATE_PASS: 0, qstats.GATE_INCONCLUSIVE: 1, qstats.GATE_FAIL: 2}
    worst_name = None
    worst_rank = -1
    for name, result in sorted(slices.items()):
        rank = order.get(result.get("verdict", ""), 1)
        if rank > worst_rank:
            worst_rank = rank
            worst_name = name
    return worst_name


__all__ = [
    "DIRECTIONS",
    "GenerationStep",
    "PositionMetrics",
    "QualityBudget",
    "evaluate_quality_gate",
    "generation_report",
    "jensen_shannon",
    "kl_divergence",
    "margin",
    "overlap",
    "position_metrics",
    "teacher_forcing_report",
    "topk_indices",
    "worst_slice_report",
]
