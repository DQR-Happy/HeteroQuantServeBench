"""E02-09 decision primitives: Amdahl, Roofline, candidate scoring, S03 protocol.

E02-09 turns the S02 measurement artifacts into one falsifiable engineering
decision: *which* hotspot S03 should optimize, *why*, and *at most how much* it
is worth. This module holds the pure part of that work so the analysis is
reproducible offline from the stored E02-* raw evidence and testable without a
GPU:

* :func:`amdahl_oracle_table` / :func:`amdahl_oracle_check` — the protocol's
  1%/5%/20% table, used as an *independent oracle* for the formula.
* :func:`validate_fractions` / :func:`combine_amdahl` — the multi-fraction form
  ``1/[(1-Σf)+Σ(f_i/s_i)]`` with the reject rules the protocol demands
  (``f<0``, ``f>1``, ``s<1``, ``Σf>1`` must be refused, never averaged away).
* :func:`request_share_bound` — the upper bound of one request an optimization
  could touch (within-phase share × phase weight), explicitly documented as a
  bound and not a promised speedup.
* :func:`shape_weighted_speedup` — shape-weighted ``s`` from the E02-02 call
  matrix instead of an arithmetic mean over shapes.
* :func:`roofline_point` / :func:`classify_roofline` — useful FLOPs, moved
  bytes, arithmetic intensity, ceiling and bottleneck classification, keeping
  *theoretical FLOPs*, *modeled bytes* and *measured throughput percentages*
  as separate quantities.
* :func:`candidate_dimensions`, :func:`weighted_total_scores`,
  :func:`weight_sensitivity` — raw, auditable dimensions per candidate plus an
  optional frozen-weight total whose stability is reported instead of trusted.
* :func:`s03_protocol` — the frozen correctness gate, performance guard band,
  dispatcher keys, fallback and stop criteria handed to S03.
* :data:`S_SCENARIOS` / :data:`GUARD_BAND` / :data:`NOMINAL_ENVELOPE` — every
  pre-registered number, with its source, in one place.

The Amdahl formula itself is *not* re-implemented here: it already exists once
in :mod:`hqsb.benchmark.roofline` (``amdahl_speedup`` /
``amdahl_max_speedup``). This module only adds validation, combination and the
scenario/protocol layer on top, so a report never re-derives it by hand.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.benchmark.roofline import amdahl_max_speedup, amdahl_speedup
from hqsb.core.errors import ConfigError

# ── Pre-registered envelope and guard band ───────────────────────────────

#: Nominal Roofline envelope of the target device. These are the *datasheet*
#: model values already used by E02-07 §4.7; they are an envelope, not a
#: measurement, and E02-08 showed the governor parks the GPU at 306 MHz, so
#: absolute FLOP/s may only be compared against "the clock actually observed".
NOMINAL_ENVELOPE: Dict[str, Any] = {
    "device": "Jetson Orin Nano Super 8GB (Ampere, sm_87)",
    "peak_fp16_flops": 67e12,
    "peak_dram_bandwidth": 68e9,
    "ridge_point_flop_per_byte": 67e12 / 68e9,
    "observed_gpu_clock_hz": {"dynamic_governor_floor": 306e6, "fixed_max": 1020e6},
    "sources": {
        "peak_fp16_flops": "datasheet value used by E02-07 §4.7 (nominal, not measured)",
        "peak_dram_bandwidth": "datasheet value used by E02-07 §4.7 (nominal, not measured)",
        "observed_gpu_clock_hz": "E02-08 §4.6 measured frequency distributions",
    },
    "caveats": [
        "nominal envelopes assume a boost clock the kernel never saw (E02-07 §4.7)",
        "NCU flushes caches between replay passes, so replay bytes != steady-state bytes",
        "DRAM byte counters are unavailable on this Tegra build (E02-07 §4.9), so "
        "the DRAM side is a modeled lower bound",
    ],
}

#: Pre-registered performance guard band. The two independent noise sources
#: measured in S02 are (a) run-to-run spread of the un-instrumented ordinary
#: baseline (E02-03: TTFT < 2%, E2E < 2.5%) and (b) the profiler's own
#: perturbation (+2%..+8%, E02-07 §4.10). A micro improvement smaller than the
#: ordinary-baseline spread cannot be distinguished from noise.
GUARD_BAND: Dict[str, Any] = {
    "independent_processes_min": 3,
    "micro_min_relative_improvement": 0.05,
    "requires_ci_lower_bound_above": 1.00,
    "noise_floor_relative": 0.025,
    "rationale": (
        "5% >= 2x the largest measured ordinary-baseline run-to-run spread "
        "(E02-03: TTFT <2%, E2E <2.5%); profiled windows are excluded from "
        "accept/reject because E02-07 §4.10 measures +2%..+8% perturbation"
    ),
    "sources": {
        "ordinary_baseline_spread": "E02-03 raw/verdict.json per-run P50s",
        "profiler_perturbation": "E02-07 §4.10 perturbation.json",
    },
}

#: Scenario ``s`` values per candidate class. E02-09 has *not* implemented a
#: kernel, so these are MODEL values with a stated source, never reported as
#: RUNTIME speedups.
S_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "rmsnorm_teaching": {
        "conservative": 1.25,
        "neutral": 2.0,
        "optimistic": 4.0,
        "source": "MODEL: reduction/vectorization teaching line; E00-04 pilot magnitude",
    },
    "attention_softmax": {
        "conservative": 1.25,
        "neutral": 2.0,
        "optimistic": 4.0,
        "source": (
            "MODEL: latency/occupancy-limited reduction (E02-07 §4.6 P1: 4096 "
            "waves/SM, No Eligible 72.4%, both throughput roofs <30%)"
        ),
    },
    "gemm_library": {
        "conservative": 1.05,
        "neutral": 1.15,
        "optimistic": 1.30,
        "source": (
            "MODEL: library re-selection / epilogue / low-bit prep on a kernel "
            "already at 87-97% Memory throughput (E02-07 §4.6 D1/P2); no "
            "from-scratch GEMM is in scope (E03-09 §7.4)"
        ),
    },
    "elementwise_fusion": {
        "conservative": 1.10,
        "neutral": 1.30,
        "optimistic": 1.60,
        "source": "MODEL: launch/memory-traffic reduction for aten::mul/cat/copy_ chains",
    },
}

#: Frozen weighting rule for the *optional* composite score. The protocol
#: forbids hiding raw dimensions behind an unauditable total, so the total is
#: only produced together with a sensitivity analysis, and any ranking flip is
#: reported as "candidates are close" instead of a unique answer.
DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "evidence_share": 0.30,
    "expected_end_to_end_gain": 0.25,
    "workload_coverage": 0.15,
    "safe_reintegration_probability": 0.15,
    "reuse_value_later_stages": 0.15,
}


# ── Amdahl ───────────────────────────────────────────────────────────────


def amdahl_ceiling(fraction: float) -> float:
    """Maximum overall speedup when ``fraction`` of the time is removed."""
    return amdahl_max_speedup(fraction)


def validate_fractions(shares: Iterable[float], *, tol: float = 1e-9) -> float:
    """Validate a set of non-overlapping fractions and return their sum.

    Raises:
        ConfigError: If any share is outside ``[0, 1]`` or the sum exceeds 1
            (which would double count a parent and its children).
    """
    total = 0.0
    for share in shares:
        if share < 0.0 or share > 1.0:
            raise ConfigError(f"share must be in [0, 1], got {share}")
        total += share
    if total > 1.0 + tol:
        raise ConfigError(
            f"shares sum to {total}, which is > 1: the partition overlaps "
            "(parent and child shares must not be added together)"
        )
    return total


def amdahl_oracle_table() -> Dict[str, Any]:
    """Recompute the protocol's example table with the single implementation.

    The expected values are the ones printed in
    ``details/S02/E02-09_hotspot_decision_and_amdahl.md`` §3 (and §11 for the
    ``s=1.25`` row); they are *independently* hand-derived there, so comparing
    the two catches percent/fraction and off-by-one mistakes.
    """
    fractions = (0.01, 0.05, 0.20)
    factors = (2.0, 4.0)
    rows: List[Dict[str, Any]] = []
    for fraction in fractions:
        row: Dict[str, Any] = {
            "f": fraction,
            "s2": amdahl_speedup(fraction, 2.0),
            "s4": amdahl_speedup(fraction, 4.0),
            "s_infinite": amdahl_ceiling(fraction),
        }
        rows.append(row)
    return {
        "fractions": list(fractions),
        "factors": list(factors),
        "rows": rows,
        "extra": {"f": 0.20, "s": 1.25, "speedup": amdahl_speedup(0.20, 1.25)},
    }


#: Hand-computed expectations from the protocol document (independent oracle).
AMDAHL_ORACLE_EXPECTED: Dict[str, float] = {
    "f=0.01,s=2": 1.00503,
    "f=0.01,s=4": 1.00756,
    "f=0.01,s=inf": 1.01010,
    "f=0.05,s=2": 1.02564,
    "f=0.05,s=4": 1.03896,
    "f=0.05,s=inf": 1.05263,
    "f=0.20,s=2": 1.11111,
    "f=0.20,s=4": 1.17647,
    "f=0.20,s=inf": 1.25000,
    "f=0.20,s=1.25": 1.04167,
}


def amdahl_oracle_check(*, rel_tol: float = 5e-4) -> Dict[str, Any]:
    """Compare the implementation against the hand-computed oracle.

    Returns a machine-checkable verdict plus the per-row deltas, so the report
    can state that the formula and the units were verified instead of assumed.
    """
    table = amdahl_oracle_table()
    actual: Dict[str, float] = {}
    for row in table["rows"]:
        label = f"{row['f']:.2f}"
        actual[f"f={label},s=2"] = row["s2"]
        actual[f"f={label},s=4"] = row["s4"]
        actual[f"f={label},s=inf"] = row["s_infinite"]
    extra = table["extra"]
    actual[f"f={extra['f']:.2f},s={extra['s']}"] = extra["speedup"]

    deltas: Dict[str, Any] = {}
    mismatches: List[str] = []
    for key, expected in AMDAHL_ORACLE_EXPECTED.items():
        got = actual.get(key)
        if got is None:
            mismatches.append(f"{key}: missing")
            continue
        delta = abs(got - expected)
        deltas[key] = {"expected": expected, "actual": got, "abs_delta": delta}
        if delta > rel_tol:
            mismatches.append(f"{key}: expected {expected}, got {got}")
    return {"passed": not mismatches, "rel_tol": rel_tol, "deltas": deltas, "mismatches": mismatches}


def combine_amdahl(speedups: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
    """Overall speedup for several non-overlapping accelerated fractions.

    Args:
        speedups: ``(fraction, speedup)`` pairs sharing one baseline.

    Returns:
        Dict with the combined speedup, the untouched fraction, and the
        per-item ceiling, or raises :class:`ConfigError` on an illegal input.
    """
    fractions = [f for f, _ in speedups]
    covered = validate_fractions(fractions)
    items: List[Dict[str, Any]] = []
    denominator = 1.0 - covered
    for fraction, factor in speedups:
        if factor < 1.0:
            raise ConfigError(f"speedup_factor must be >= 1, got {factor}")
        denominator += fraction / factor
        items.append(
            {
                "fraction": fraction,
                "speedup": factor,
                "ceiling_if_infinite": amdahl_ceiling(fraction),
            }
        )
    return {
        "speedup": 1.0 / denominator,
        "covered_fraction": covered,
        "untouched_fraction": 1.0 - covered,
        "items": items,
        "note": "all fractions must share one baseline and must not overlap",
    }


def request_share_bound(phase_share: float, phase_weight: float) -> float:
    """Upper bound of one request an optimization could touch.

    ``phase_share`` is the within-phase device-work share (mode-robust);
    ``phase_weight`` is that phase's share of one request's wall clock. The
    product is a *bound*: overlapping kernels and host-bound idle mean
    removing device work does not remove the same amount of wall clock.
    """
    if not (0.0 <= phase_share <= 1.0):
        raise ConfigError(f"phase_share must be in [0, 1], got {phase_share}")
    if not (0.0 <= phase_weight <= 1.0):
        raise ConfigError(f"phase_weight must be in [0, 1], got {phase_weight}")
    return phase_share * phase_weight


def shape_weighted_speedup(
    calls: Sequence[float],
    t_old_us: Sequence[float],
    t_new_us: Sequence[float],
) -> Dict[str, Any]:
    """Shape-weighted speedup ``Σ(calls·t_old) / Σ(calls·t_new)``.

    Averaging per-shape speedups would weight a one-call shape like a
    hundreds-of-calls shape; the protocol therefore requires the call matrix
    (from E02-02) and the two costs. Shapes with no selected implementation
    must enter ``t_new_us`` with the *fallback* latency, never be dropped.
    """
    if not (len(calls) == len(t_old_us) == len(t_new_us)):
        raise ConfigError(
            "calls, t_old_us and t_new_us must have the same length, got "
            f"{len(calls)}, {len(t_old_us)}, {len(t_new_us)}"
        )
    if not calls:
        raise ConfigError("shape_weighted_speedup needs at least one shape")
    total_old = 0.0
    total_new = 0.0
    rows: List[Dict[str, Any]] = []
    for count, old, new in zip(calls, t_old_us, t_new_us):
        if count <= 0:
            raise ConfigError(f"call count must be positive, got {count}")
        if old <= 0.0 or new <= 0.0:
            raise ConfigError(f"latencies must be positive, got old={old}, new={new}")
        total_old += count * old
        total_new += count * new
        rows.append(
            {
                "calls": count,
                "t_old_us": old,
                "t_new_us": new,
                "shape_speedup": old / new,
                "old_cost_us": count * old,
                "new_cost_us": count * new,
            }
        )
    return {
        "t_old_target_us": total_old,
        "t_new_target_us": total_new,
        "s_weighted": total_old / total_new,
        "rows": rows,
    }


# ── Roofline ─────────────────────────────────────────────────────────────


def roofline_point(
    *,
    bytes_moved: float,
    duration_us: float,
    useful_flops: Optional[float] = None,
    memory_level: str = "dram",
    peak_flops: float = NOMINAL_ENVELOPE["peak_fp16_flops"],
    peak_bandwidth: float = NOMINAL_ENVELOPE["peak_dram_bandwidth"],
    measured_memory_throughput_pct: Optional[float] = None,
    measured_compute_throughput_pct: Optional[float] = None,
    l2_hit_rate_pct: Optional[float] = None,
    ceiling_source: str = "nominal_datasheet",
) -> Dict[str, Any]:
    """One Roofline point, keeping FLOPs / bytes / measured percents apart.

    ``bytes_moved`` must be the traffic of the *declared* ``memory_level``
    (modeled DRAM traffic if ``memory_level='dram'``); comparing L2 bytes with
    a DRAM roof is exactly the mistake the protocol forbids.
    """
    if duration_us <= 0:
        raise ConfigError(f"duration_us must be > 0, got {duration_us}")
    if bytes_moved <= 0:
        raise ConfigError(f"bytes_moved must be > 0, got {bytes_moved}")
    seconds = duration_us / 1e6
    achieved_bytes_per_s = bytes_moved / seconds
    point: Dict[str, Any] = {
        "memory_level": memory_level,
        "bytes_moved": bytes_moved,
        "duration_us": duration_us,
        "achieved_bytes_per_s": achieved_bytes_per_s,
        "bandwidth_ceiling_fraction": (
            achieved_bytes_per_s / peak_bandwidth if peak_bandwidth else None
        ),
        "measured_memory_throughput_pct": measured_memory_throughput_pct,
        "measured_compute_throughput_pct": measured_compute_throughput_pct,
        "l2_hit_rate_pct": l2_hit_rate_pct,
        "ceiling_source": ceiling_source,
        "caveats": list(NOMINAL_ENVELOPE["caveats"]),
    }
    if useful_flops is not None:
        if useful_flops < 0:
            raise ConfigError(f"useful_flops must be >= 0, got {useful_flops}")
        intensity = useful_flops / bytes_moved
        achieved_flops = useful_flops / seconds
        bound = min(peak_flops, peak_bandwidth * intensity)
        point.update(
            {
                "useful_flops": useful_flops,
                "achieved_flops": achieved_flops,
                "arithmetic_intensity_flop_per_byte": intensity,
                "roofline_bound_flops": bound,
                "roofline_efficiency": achieved_flops / bound if bound else 0.0,
                "ridge_point_flop_per_byte": (
                    peak_flops / peak_bandwidth if peak_bandwidth else None
                ),
            }
        )
    else:
        point.update(
            {
                "useful_flops": None,
                "arithmetic_intensity_flop_per_byte": None,
                "arithmetic_intensity_note": (
                    "no useful-FLOP model for this kernel class; the byte side "
                    "is reported instead of inventing a FLOP count"
                ),
            }
        )
    point["classification"] = classify_roofline(
        arithmetic_intensity=point.get("arithmetic_intensity_flop_per_byte"),
        measured_memory_throughput_pct=measured_memory_throughput_pct,
        measured_compute_throughput_pct=measured_compute_throughput_pct,
    )
    return point


def classify_roofline(
    *,
    arithmetic_intensity: Optional[float],
    measured_memory_throughput_pct: Optional[float],
    measured_compute_throughput_pct: Optional[float],
    saturation_pct: float = 75.0,
) -> str:
    """Classify a measured kernel into one of the protocol's four buckets.

    Uses the *measured* throughput percentages when available, because the
    nominal envelope assumes a clock this board never runs at (E02-08 §4.6).
    Never claims a root cause the counters do not support.
    """
    mem = measured_memory_throughput_pct
    comp = measured_compute_throughput_pct
    if mem is None and comp is None:
        return "insufficient_evidence"
    if mem is not None and mem >= saturation_pct and (comp is None or mem >= comp):
        return "throughput_memory_limited"
    if comp is not None and comp >= saturation_pct:
        return "throughput_compute_limited"
    if (mem is not None and mem < saturation_pct) and (
        comp is not None and comp < saturation_pct
    ):
        return "launch_or_latency_limited"
    return "mixed_or_insufficient"


# ── Candidate dimensions and (optional) scored ranking ────────────────────


def candidate_dimensions(
    *,
    name: str,
    phase: str,
    share_low: float,
    share_high: float,
    call_count: Optional[int],
    shapes: Sequence[str],
    bottleneck: str,
    expected_speedup: Mapping[str, float],
    ceiling: float,
    reference_complexity: str,
    integration_risk: str,
    library_baseline: str,
    reuse_value: str,
    workload_coverage: float,
    evidence: Sequence[str],
    notes: str = "",
) -> Dict[str, Any]:
    """One candidate's *raw* dimensions, before any weighting is applied.

    The protocol requires the raw values to be recorded first, so a reader can
    audit the decision without trusting a hidden composite score.
    """
    # ``share_low``/``share_high`` are a range, not a partition, so each bound
    # is checked individually instead of through ``validate_fractions``.
    for bound in (share_low, share_high):
        if bound < 0.0 or bound > 1.0:
            raise ConfigError(f"share bound must be in [0, 1], got {bound}")
    if share_high < share_low:
        raise ConfigError(
            f"share_high ({share_high}) must be >= share_low ({share_low})"
        )
    if not (0.0 <= workload_coverage <= 1.0):
        raise ConfigError(f"workload_coverage must be in [0, 1], got {workload_coverage}")
    if ceiling < 1.0:
        raise ConfigError(f"ceiling must be >= 1, got {ceiling}")
    return {
        "name": name,
        "phase": phase,
        "share_low": share_low,
        "share_high": share_high,
        "call_count": call_count,
        "shapes": list(shapes),
        "bottleneck": bottleneck,
        "expected_speedup": dict(expected_speedup),
        "amdahl_ceiling": ceiling,
        "reference_complexity": reference_complexity,
        "integration_risk": integration_risk,
        "library_baseline": library_baseline,
        "reuse_value": reuse_value,
        "workload_coverage": workload_coverage,
        "evidence": list(evidence),
        "notes": notes,
    }


def _dimension_scores(candidate: Mapping[str, Any]) -> Dict[str, float]:
    """Map raw dimensions onto the frozen 0..1 scale used by the total."""
    high = float(candidate["share_high"])
    neutral = float(candidate.get("expected_speedup", {}).get("neutral", 1.0))
    # A 2x shape-local speedup saturates the "gain" dimension; this is a
    # normalisation, not a claim that 2x is achievable.
    gain = min(max((neutral - 1.0) / 1.0, 0.0), 1.0)
    risk_penalty = {"low": 1.0, "medium": 0.6, "high": 0.3}.get(
        str(candidate.get("integration_risk", "high")).lower(), 0.3
    )
    reuse = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(
        str(candidate.get("reuse_value", "low")).lower(), 0.3
    )
    return {
        "evidence_share": min(high, 1.0),
        "expected_end_to_end_gain": gain,
        "workload_coverage": float(candidate["workload_coverage"]),
        "safe_reintegration_probability": risk_penalty,
        "reuse_value_later_stages": reuse,
    }


def weighted_total_scores(
    candidates: Sequence[Mapping[str, Any]],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Frozen-weight composite score, always with its per-dimension breakdown."""
    applied = dict(weights or DEFAULT_SCORE_WEIGHTS)
    total_weight = sum(applied.values())
    if total_weight <= 0:
        raise ConfigError("weights must sum to a positive number")
    ranked: List[Dict[str, Any]] = []
    for candidate in candidates:
        scores = _dimension_scores(candidate)
        for key in applied:
            if key not in scores:
                raise ConfigError(f"unknown score dimension {key!r}")
        total = sum(applied[key] * scores[key] for key in applied) / total_weight
        ranked.append(
            {"name": candidate["name"], "total": total, "dimensions": scores}
        )
    ranked.sort(key=lambda item: item["total"], reverse=True)
    return {
        "weights": applied,
        "weight_source": "pre-registered in hqsb.benchmark.hotspot_decision.DEFAULT_SCORE_WEIGHTS",
        "ranking": ranked,
    }


def weight_sensitivity(
    candidates: Sequence[Mapping[str, Any]],
    *,
    variants: Optional[Sequence[Mapping[str, float]]] = None,
) -> Dict[str, Any]:
    """Check whether the ranking survives a reasonable change of weights.

    If the top choice flips easily, the honest report is "candidates are
    close", not a unique answer.
    """
    base = dict(DEFAULT_SCORE_WEIGHTS)
    if variants is None:
        variants = (
            {"evidence_share": 0.5, "expected_end_to_end_gain": 0.5},
            {"evidence_share": 0.15, "safe_reintegration_probability": 0.5},
            {"workload_coverage": 0.4, "reuse_value_later_stages": 0.4},
        )
    results: List[Dict[str, Any]] = []
    for variant in variants:
        merged = dict(base)
        merged.update(variant)
        total = sum(merged.values())
        normalised = {key: value / total for key, value in merged.items()}
        outcome = weighted_total_scores(candidates, weights=normalised)
        results.append(
            {
                "variant": dict(variant),
                "ranking": [row["name"] for row in outcome["ranking"]],
                "top": outcome["ranking"][0]["name"] if outcome["ranking"] else None,
            }
        )
    tops = [row["top"] for row in results]
    base_top = weighted_total_scores(candidates)["ranking"][0]["name"]
    flipped = any(top != base_top for top in tops)
    return {
        "base_top": base_top,
        "variants": results,
        "top_flips": flipped,
        "verdict": (
            "candidates are close: the frozen-weight choice is not robust"
            if flipped
            else "the frozen-weight choice is stable across the tested weights"
        ),
    }


# ── S03 protocol handed over ─────────────────────────────────────────────


def s03_protocol() -> Dict[str, Any]:
    """Frozen correctness gate, performance guard band and stop criteria.

    Tolerance numbers here are *ceilings pre-registered before S03 ran*; E03-01
    resolves them per dtype/shape into the C3 ``OperatorSpec`` and may tighten
    them with a documented derivation, but loosening them after seeing results
    is forbidden by the protocol.
    """
    fp16_quantum = 2.0 ** -11  # ~4.88e-4 relative, FP16 significand
    return {
        "correctness": {
            "rmsnorm": {
                "equation": "y = x * rsqrt(mean(x^2) + eps) * w, reduction over the last dim H",
                "dtype": {"in": "float16", "weight": "float16", "accum": "float32", "out": "float16"},
                "epsilon_source": "model config (Qwen3RMSNorm), frozen per run",
                "reference": "FP64 CPU oracle + independent FP32 framework oracle (E03-01)",
                "metrics": ["max_abs", "mean_abs", "RMSE", "cosine", "L2rel"],
                "tolerance_ceiling": {
                    "atol_fp16_out": 8 * fp16_quantum,
                    "rtol_fp16_out": 2e-2,
                    "l2rel": 1e-3,
                    "cosine_min": 0.9999,
                    "derivation": "FP16 out quantum 2^-11; ceiling = 8 quanta + FP32 accumulation order budget",
                },
                "shapes_required": [
                    "real: hidden=2048 (input/post_attention_layernorm), head_dim=128 with Hq=16 / Hkv=8 (q_norm/k_norm)",
                    "synthetic boundary: H in {100,128,512,6144,8192, odd}",
                    "rows: 1, smallest decode batch, small batch, B*I prefill flatten, max safe rows",
                ],
            },
            "second_hotspot": {
                "declared_boundary": "frozen in the decision record; the operator boundary (not the whole attention block) is part of the C3 spec",
                "reference": "independent oracle for the declared boundary; vendor/library results may cross-check but never be the only reference (E03-09 §6.1)",
                "metrics": ["max_abs", "mean_abs", "RMSE", "cosine", "L2rel", "first_mismatch"],
                "tolerance_ceiling": {
                    "atol_fp16_out": 1.6e-2,
                    "rtol_fp16_out": 5e-2,
                    "l2rel": 2e-2,
                    "cosine_min": 0.9995,
                    "derivation": "reduction length up to 2048 accumulate in FP32 then one output rounding; ceiling must be widened in E03-01 only with a derivation, never to admit a wrong result",
                },
                "domain_requirements": [
                    "masked / padding positions and all-masked policy",
                    "sequence boundaries and odd lengths",
                    "GQA head mapping if the boundary spans heads",
                    "layout / stride / alignment support-or-reject",
                ],
            },
            "hard_gates": [
                "current / non-default stream: no implicit default-stream serialisation (E03-06)",
                "compute-sanitizer: no OOB / race / leak on supported cases (E03-07)",
                "invalid input fails before launch with a stable reason code",
                "forced variant must actually be the executed variant (no silent fallback)",
            ],
            "model_level_gate_deferred_to_s04_5": (
                "operator-level PASS does not imply model-level PASS; token parity and "
                "first-token logits (rtol/atol 1e-3, hqsb/benchmark/correctness.py) "
                "are checked at S04.5"
            ),
            "correctness_never_tradeable_for_speed": True,
        },
        "performance": {
            "guard_band": dict(GUARD_BAND),
            "reporting_rules": [
                "report shape-weighted and single-point-best separately",
                "compile/tuning cost reported separately from steady kernel time",
                "cold library init separate from warm steady-state",
                "ordinary benchmark and profiler runs stored separately",
            ],
        },
        "dispatcher": {
            "keys": [
                "operator id / version",
                "shape class (decode M=1 vs prefill large M; hidden vs head_dim)",
                "dtype / accumulation",
                "layout / alignment / tail",
                "stream / workspace",
                "architecture capability",
            ],
            "fallback": "unknown or unfavourable case returns to the framework reference",
            "no_hidden_defaults": True,
        },
        "stop_criteria": [
            "correctness/safety failure: stop the performance claim and fix first (never PASS_NEGATIVE)",
            "ceiling too low: after the pre-registered exploration budget, PASS_NEGATIVE with retained evidence",
            "gain not above guard band: stop adding versions",
            "resource/ABI/integration infeasible: record the boundary and the alternative route",
            "already inside the explainable roof/Amdahl interval: hand to S04/S04.5 instead of micro-tuning further",
            "engineering budget exceeded with low new hypothesis value: stop and report",
        ],
        "pass_negative_scope": (
            "PASS_NEGATIVE applies only to valid, correct and sufficiently sampled "
            "performance hypotheses; stream/sanitizer/correctness failures are FAIL"
        ),
    }


# ── Decision record ──────────────────────────────────────────────────────


def decision_record(
    *,
    selected: Sequence[Mapping[str, Any]],
    deferred: Sequence[Mapping[str, Any]],
    rank_one_handling: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble the Hotspot Decision Record (protocol §12)."""
    if not selected:
        raise ConfigError("a decision record needs at least one selected line")
    for entry in selected:
        for key in ("name", "phase", "share_range", "amdahl_ceiling", "route", "stop_criteria"):
            if key not in entry:
                raise ConfigError(f"selected entry {entry.get('name')!r} missing {key!r}")
    return {
        "selected": [dict(entry) for entry in selected],
        "rank_one_handling": dict(rank_one_handling),
        "deferred_or_not_selected": [dict(entry) for entry in deferred],
        "provenance": dict(provenance),
        "revisitable": (
            "the decision is not permanent truth: if the model, attention path, "
            "runtime or precision changes, re-profile and recompute the shares"
        ),
    }


__all__ = [
    "AMDAHL_ORACLE_EXPECTED",
    "DEFAULT_SCORE_WEIGHTS",
    "GUARD_BAND",
    "NOMINAL_ENVELOPE",
    "S_SCENARIOS",
    "amdahl_ceiling",
    "amdahl_oracle_check",
    "amdahl_oracle_table",
    "candidate_dimensions",
    "classify_roofline",
    "combine_amdahl",
    "decision_record",
    "request_share_bound",
    "roofline_point",
    "s03_protocol",
    "shape_weighted_speedup",
    "validate_fractions",
    "weight_sensitivity",
    "weighted_total_scores",
]
