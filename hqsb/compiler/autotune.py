"""Autotune search spaces, legality filtering, budgets, holdout and cost.

Protocol anchor: ``details/S11/E11-06_autotune_search_holdout_budget.md``.
Autotune is treated as a *budgeted experiment*, not "turn on max_autotune":

* the search space is a semantic object (knobs with meaning, constraints with
  categories and reasons, candidate identity, seed);
* three gates (static legality → compile/resource → correctness) run *before*
  benchmarking, and an invalid candidate is never timed (latency=∞ is not an
  observation);
* budgets B0–B3 are frozen with wall/device/compile/disk limits and an
  early-stop rule, and all methods use the same budget;
* tuning/validation/holdout are separated by semantic groups, and a winner
  must be confirmed in a new process;
* search cost, steady gain and break-even are reported separately.

Nothing here runs a benchmark: the module builds the plan, validates the
records and computes the metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import canonical_json, sha256_text
from hqsb.compiler.records import (
    AUTOTUNE_SPLITS,
    AutotuneTrialRecord,
    amortization_row,
    break_even_calls,
)

# ── search space (E11-06 §5, steps 3–6) ────────────────────────────────────

KNOB_TYPES: Tuple[str, ...] = ("int", "float", "choice", "bool")

CONSTRAINT_CATEGORIES: Tuple[str, ...] = ("semantic", "target", "resource", "shape", "alignment")

STATIC_REJECT_REASONS: Tuple[str, ...] = (
    "AUTOTUNE_INVALID:semantic_illegal",
    "AUTOTUNE_INVALID:target_unsupported",
    "AUTOTUNE_INVALID:resource_infeasible",
    "AUTOTUNE_INVALID:shape_incompatible",
    "AUTOTUNE_INVALID:alignment_required",
    "AUTOTUNE_INVALID:vector_alignment",
)


@dataclass
class Knob:
    name: str
    kind: str
    values: Tuple[Any, ...]
    meaning: str

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.kind not in KNOB_TYPES:
            problems.append(f"unknown knob type {self.kind!r}")
        if not self.values:
            problems.append(f"{self.name}: a knob without values cannot be searched")
        if not self.meaning:
            problems.append(f"{self.name}: knobs must state the mechanism they control")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.kind,
            "values": list(self.values),
            "meaning": self.meaning,
        }


@dataclass
class SearchConstraint:
    expression: str
    category: str
    reason: str
    predicate: Optional[Any] = None  # callable(config) -> bool

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.category not in CONSTRAINT_CATEGORIES:
            problems.append(f"unknown constraint category {self.category!r}")
        if not self.reason:
            problems.append(f"{self.expression}: constraints must explain their reason")
        return problems

    def allows(self, config: Mapping[str, Any]) -> bool:
        if self.predicate is None:
            raise ConfigError(
                f"constraint {self.expression!r} has no predicate: it cannot be evaluated, and "
                "unknown must not silently pass"
            )
        return bool(self.predicate(config))

    def as_dict(self) -> Dict[str, Any]:
        return {"expression": self.expression, "category": self.category, "reason": self.reason}


@dataclass
class SearchSpaceSpec:
    """The frozen search space of one task (identity must be versioned)."""

    space_id: str
    semantic_op: str
    version: str
    target_id: str
    shape_domain_id: str
    knobs: Tuple[Knob, ...]
    constraints: Tuple[SearchConstraint, ...]
    candidate_generator_version: str = "1"
    default_candidate: Mapping[str, Any] = field(default_factory=dict)
    heuristic_policy: str = ""
    max_candidates: int = 0
    sampling_strategy: str = "full"
    seed: int = 0
    correctness_policy_id: str = ""
    benchmark_policy_id: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("space_id", "semantic_op", "version", "target_id", "shape_domain_id"):
            if not getattr(self, name):
                problems.append(f"search space missing {name!r}")
        if not self.knobs:
            problems.append("a search space without knobs is not a search space")
        problems.extend(problem for knob in self.knobs for problem in knob.validate())
        problems.extend(problem for item in self.constraints for problem in item.validate())
        if not self.default_candidate:
            problems.append("the search space must declare its default candidate")
        if not self.correctness_policy_id or not self.benchmark_policy_id:
            problems.append("correctness/benchmark policies must be named before tuning")
        if self.max_candidates <= 0:
            problems.append("max_candidates must bound the generated set")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "search_space_id": self.space_id,
            "semantic_op": self.semantic_op,
            "version": self.version,
            "target_id": self.target_id,
            "shape_domain_id": self.shape_domain_id,
            "knobs": [knob.as_dict() for knob in self.knobs],
            "constraints": [item.as_dict() for item in self.constraints],
            "candidate_generator_version": self.candidate_generator_version,
            "default_candidate": dict(sorted(self.default_candidate.items())),
            "heuristic_policy": self.heuristic_policy,
            "max_candidates": self.max_candidates,
            "sampling_strategy": self.sampling_strategy,
            "seed": self.seed,
            "correctness_policy_id": self.correctness_policy_id,
            "benchmark_policy_id": self.benchmark_policy_id,
        }

    def digest(self) -> str:
        return sha256_text(canonical_json(self.as_dict()))


def candidate_identity(space: SearchSpaceSpec, config: Mapping[str, Any], *, kernel_build_id: str = "") -> str:
    """Candidate identity = config + kernel source/build + target + flags."""
    payload = {
        "space": space.space_id,
        "space_version": space.version,
        "config": dict(sorted(config.items())),
        "kernel_build_id": kernel_build_id,
        "target_id": space.target_id,
        "generator": space.candidate_generator_version,
    }
    return sha256_text(canonical_json(payload))[:16]


@dataclass
class GeneratedCandidate:
    config: Mapping[str, Any]
    identity: str
    status: str = "generated"  # generated | invalid
    reject_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_identity": self.identity,
            "config": dict(sorted(self.config.items())),
            "status": self.status,
            "reject_reason": self.reject_reason,
        }


def generate_candidates(space: SearchSpaceSpec) -> List[GeneratedCandidate]:
    """Enumerate the full space, then apply static constraints (steps 8–9)."""
    problems = space.validate()
    if problems:
        raise ConfigError("invalid search space: " + "; ".join(problems))
    configs: List[Dict[str, Any]] = [{}]
    for knob in space.knobs:
        configs = [{**config, knob.name: value} for config in configs for value in knob.values]
    rows: List[GeneratedCandidate] = []
    for config in configs:
        identity = candidate_identity(space, config)
        failed = [item for item in space.constraints if not item.allows(config)]
        if failed:
            rows.append(
                GeneratedCandidate(
                    config=config,
                    identity=identity,
                    status="invalid",
                    reject_reason=f"AUTOTUNE_INVALID:{failed[0].category}",
                )
            )
        else:
            rows.append(GeneratedCandidate(config=config, identity=identity))
    if len(rows) > space.max_candidates:
        raise ConfigError(
            f"generated {len(rows)} candidates > max_candidates {space.max_candidates}: "
            "budget must be declared before generating"
        )
    return rows


def filter_false_reject_audit(
    candidates: Sequence[GeneratedCandidate],
    *,
    compile_fn: Mapping[str, str],
    sample_per_reason: int = 2,
) -> Dict[str, Any]:
    """Verify a sample of rejected candidates really is infeasible (step 9)."""
    rejected: Dict[str, List[GeneratedCandidate]] = {}
    for row in candidates:
        if row.status == "invalid":
            rejected.setdefault(row.reject_reason, []).append(row)
    rows: List[Dict[str, Any]] = []
    for reason, items in sorted(rejected.items()):
        for item in items[:sample_per_reason]:
            outcome = compile_fn.get(item.identity, "not_probed")
            rows.append(
                {
                    "candidate_identity": item.identity,
                    "reason": reason,
                    "probe_outcome": outcome,
                    "false_reject": outcome == "compiled",
                }
            )
    return {
        "sampled": len(rows),
        "false_rejects": [row for row in rows if row["false_reject"]],
        "families": sorted(rejected),
        "reason_coverage": len(rejected),
        "note": "rejected candidates are never reported as ordinary search cost",
    }


# ── budget ladder (steps 11, 18–22) ────────────────────────────────────────


@dataclass
class Budget:
    name: str
    max_generated: int
    max_compiled: int
    max_measured: int
    wall_clock_s: float
    device_seconds: float
    compile_cpu_seconds: float
    disk_bytes: int
    early_stop: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.early_stop:
            problems.append(f"{self.name}: an early-stop rule is mandatory")
        for name in (
            "max_generated",
            "max_compiled",
            "max_measured",
            "wall_clock_s",
            "device_seconds",
            "compile_cpu_seconds",
            "disk_bytes",
        ):
            value = getattr(self, name)
            if value is None or value < 0:
                problems.append(f"{self.name}: {name} must be a non-negative bound")
        if self.max_measured > self.max_compiled:
            problems.append(f"{self.name}: measured cannot exceed compiled")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_generated": self.max_generated,
            "max_compiled": self.max_compiled,
            "max_measured": self.max_measured,
            "wall_clock_s": self.wall_clock_s,
            "device_seconds": self.device_seconds,
            "compile_cpu_seconds": self.compile_cpu_seconds,
            "disk_bytes": self.disk_bytes,
            "early_stop": self.early_stop,
        }


def budget_ladder(
    *, space_size: int, wall_per_candidate_s: float = 1.0
) -> List[Budget]:
    """B0 default → B1 small → B2 medium → B3 exhaustive (monotone)."""
    return [
        Budget("B0_default", 1, 1, 1, max(wall_per_candidate_s, 1.0), 1.0, 1.0, 1 << 20, "n/a"),
        Budget(
            "B1_small",
            min(space_size, 8),
            min(space_size, 6),
            min(space_size, 4),
            max(8 * wall_per_candidate_s, 8.0),
            16.0,
            60.0,
            1 << 24,
            "stop when top-1 is unchanged for 2 batches",
        ),
        Budget(
            "B2_medium",
            min(space_size, 32),
            min(space_size, 24),
            min(space_size, 16),
            max(32 * wall_per_candidate_s, 32.0),
            64.0,
            240.0,
            1 << 26,
            "stop when confirmed best is unchanged for 3 batches",
        ),
        Budget(
            "B3_exhaustive",
            space_size,
            space_size,
            space_size,
            max(space_size * wall_per_candidate_s, 64.0),
            256.0,
            960.0,
            1 << 28,
            "exhaustive: no early stop",
        ),
    ]


def check_budget_fairness(rows: Sequence[Tuple[str, Budget]]) -> Dict[str, Any]:
    """All compared methods must use the same measurement budget (step 21)."""
    by_method = {name: budget.as_dict() for name, budget in rows}
    signatures = {
        name: (
            budget.max_measured,
            budget.device_seconds,
            budget.wall_clock_s,
        )
        for name, budget in rows
    }
    distinct = {value for value in signatures.values()}
    return {
        "methods": by_method,
        "fair": len(distinct) <= 1,
        "note": "comparing methods with different budgets is a protocol violation",
    }


# ── splits (steps 10, 25–27) ───────────────────────────────────────────────

SPLIT_STRATEGIES: Tuple[str, ...] = (
    "group_60_20_20",
    "leave_one_shape_group_out",
)

HOLDOUT_KINDS: Tuple[str, ...] = (
    "interpolation",
    "boundary_extrapolation",
    "layout_dtype",
    "target",
)


@dataclass
class ShapeGroup:
    group_id: str
    graph_family: str
    regime: str
    bucket: str
    dtype: str
    layout: str
    arch: str

    def key(self) -> str:
        return "|".join(
            (self.graph_family, self.regime, self.bucket, self.dtype, self.layout, self.arch)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "graph_family": self.graph_family,
            "regime": self.regime,
            "bucket": self.bucket,
            "dtype": self.dtype,
            "layout": self.layout,
            "arch": self.arch,
        }


def build_split_manifest(
    groups: Sequence[ShapeGroup],
    *,
    strategy: str = "group_60_20_20",
    holdout_id: str = "",
) -> Dict[str, Any]:
    """Assign whole groups to tuning/validation/holdout (never split a group)."""
    if strategy not in SPLIT_STRATEGIES:
        raise ConfigError(f"unknown split strategy {strategy!r}")
    ordered = sorted(groups, key=lambda item: item.key())
    if len({group.key() for group in ordered}) != len(ordered):
        raise ConfigError("duplicate shape group keys: lineage must be unique before splitting")
    if strategy == "leave_one_shape_group_out":
        if not holdout_id or holdout_id not in {group.group_id for group in ordered}:
            raise ConfigError("leave-one-group-out needs a declared holdout group id")
        tuning = [g.group_id for g in ordered if g.group_id != holdout_id]
        return {
            "strategy": strategy,
            "manifest": {"tuning": tuning, "validation": [], "holdout": [holdout_id]},
            "groups": [group.as_dict() for group in ordered],
            "hash": sha256_text(canonical_json([group.as_dict() for group in ordered])),
        }
    n = len(ordered)
    n_holdout = max(1, n // 5)
    n_validation = max(1, n // 5)
    holdout = [g.group_id for g in ordered[-n_holdout:]]
    validation = [g.group_id for g in ordered[-n_holdout - n_validation : -n_holdout]]
    tuning = [g.group_id for g in ordered[: n - n_holdout - n_validation]]
    return {
        "strategy": strategy,
        "manifest": {"tuning": tuning, "validation": validation, "holdout": holdout},
        "groups": [group.as_dict() for group in ordered],
        "hash": sha256_text(canonical_json([group.as_dict() for group in ordered])),
        "rule": "same canonical graph / adjacent shape lineage must not cross splits",
    }


def split_leak_check(manifest: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    sets = {name: set(ids) for name, ids in manifest.items()}
    overlaps = {
        f"{left}∩{right}": sorted(sets[left] & sets[right])
        for left in sets
        for right in sets
        if left < right and sets[left] & sets[right]
    }
    return {
        "overlaps": overlaps,
        "leak_free": not overlaps,
        "note": "random-row splits are diagnostics only and never the primary result",
    }


# ── trial sandbox and reset (steps 12–16) ──────────────────────────────────


@dataclass
class TrialSandbox:
    """Isolation contract for one candidate (timeout, limits, capture)."""

    timeout_s: float
    memory_limit_mb: int
    output_dir: str
    resets_state: bool = True

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.timeout_s <= 0:
            problems.append("sandbox timeout must be positive")
        if self.memory_limit_mb <= 0:
            problems.append("sandbox memory limit must be positive")
        if not self.output_dir:
            problems.append("sandbox must write to an isolated output directory")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timeout_s": self.timeout_s,
            "memory_limit_mb": self.memory_limit_mb,
            "output_dir": self.output_dir,
            "resets_state": self.resets_state,
            "contract": (
                "a failing candidate is isolated: timeout/OOM/illegal access must not terminate "
                "the session or pollute later trials"
            ),
        }


def reset_protocol(*, tensors: Sequence[str], rng: bool = True, kv: bool = False) -> Dict[str, Any]:
    """Per-trial reset: which tensors/state are restored and how it is verified."""
    return {
        "reset_tensors": list(tensors),
        "reset_rng": rng,
        "reset_kv": kv,
        "verification": "hash each reset tensor before the trial and compare after restore",
        "rule": (
            "Triton-style autotune reruns candidates; without reset the later candidates see "
            "mutated data and the ranking is invalid"
        ),
    }


def verify_reset(hashes_before: Mapping[str, str], hashes_after: Mapping[str, str]) -> Dict[str, Any]:
    diff = sorted(
        key
        for key in set(hashes_before) | set(hashes_after)
        if hashes_before.get(key) != hashes_after.get(key)
    )
    return {"mismatched": diff, "ok": not diff}


def measurement_schedule(
    candidates: Sequence[str], *, seed: int, batches: int = 3
) -> Dict[str, Any]:
    """Randomised/rotated measurement order (step 17) with a reproducible seed."""
    if not candidates:
        raise ConfigError("no candidates to schedule")
    import random

    rng = random.Random(seed)
    order: List[str] = []
    pool = list(candidates)
    for _ in range(batches):
        batch = list(pool)
        rng.shuffle(batch)
        order.extend(batch)
    return {
        "seed": seed,
        "batches": batches,
        "order": order,
        "note": "fixed order lets whichever candidate runs last inherit a warmer/steadier clock",
    }


def correctness_before_benchmark(trial: AutotuneTrialRecord) -> Dict[str, Any]:
    """Gate: a candidate is benchmarked only after it passes correctness."""
    problems = trial.validate()
    ordered = not (
        trial.correctness_status != "pass" and bool(trial.raw_timing_samples)
    )
    return {
        "trial": trial.as_dict(),
        "compliant": ordered and not problems,
        "problems": problems,
        "rule": "benchmark precedes correctness is a protocol violation (step 15)",
    }


# ── oracle, winners, confirmation (steps 21–24) ────────────────────────────


def per_shape_oracle(
    *, shape_group: str, rows: Sequence[AutotuneTrialRecord], oracle_scope: str
) -> Dict[str, Any]:
    """Best *observed* candidate under the frozen measurement protocol."""
    eligible = [
        row
        for row in rows
        if row.shape_group == shape_group
        and row.legality_status == "legal"
        and row.compile_status == "ok"
        and row.correctness_status == "pass"
        and row.raw_timing_samples
    ]
    if not eligible:
        return {
            "shape_group": shape_group,
            "best": "",
            "best_median": None,
            "oracle_scope": oracle_scope,
            "reason": "no measured legal candidate",
        }
    best = min(eligible, key=lambda row: _median(row.raw_timing_samples))
    return {
        "shape_group": shape_group,
        "best": candidate_identity_from_trial(best),
        "best_median": round(_median(best.raw_timing_samples), 6),
        "oracle_scope": oracle_scope,
        "candidates_considered": len(eligible),
        "note": "this oracle is the best of the frozen candidate set, not a hardware optimum",
    }


def candidate_identity_from_trial(trial: AutotuneTrialRecord) -> str:
    return sha256_text(canonical_json(dict(sorted(trial.candidate_config.items()))))[:16]


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ConfigError("median of an empty sample set")
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _ci95(values: Sequence[float]) -> Tuple[float, float]:
    import math

    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    half = 1.96 * math.sqrt(variance) / math.sqrt(n)
    return (mean - half, mean + half)


def select_winners(rows: Sequence[AutotuneTrialRecord], *, top_k: int = 3) -> Dict[str, Any]:
    """Median + CI based provisional winners; overlapping CIs are a tie."""
    by_candidate: Dict[str, List[AutotuneTrialRecord]] = {}
    for row in rows:
        if row.correctness_status == "pass" and row.raw_timing_samples:
            by_candidate.setdefault(candidate_identity_from_trial(row), []).append(row)
    stats: List[Dict[str, Any]] = []
    for identity, items in by_candidate.items():
        samples = [value for item in items for value in item.raw_timing_samples]
        low, high = _ci95(samples)
        stats.append(
            {
                "candidate_identity": identity,
                "median": round(_median(samples), 6),
                "ci95_low": None if low != low else round(low, 6),
                "ci95_high": None if high != high else round(high, 6),
                "samples": len(samples),
                "config": dict(sorted(items[0].candidate_config.items())),
            }
        )
    stats.sort(key=lambda item: item["median"])
    winners = stats[:top_k]
    ties: List[Dict[str, Any]] = []
    if len(stats) >= 2:
        first = stats[0]
        for other in stats[1:]:
            if (
                first["ci95_high"] is not None
                and other["ci95_low"] is not None
                and other["ci95_low"] <= first["ci95_high"]
            ):
                ties.append(
                    {
                        "a": first["candidate_identity"],
                        "b": other["candidate_identity"],
                        "rule": "overlapping CI: report a tie instead of a ranking",
                    }
                )
    return {
        "provisional": winners,
        "ties": ties,
        "tie_policy": (
            "on ties choose the lower-resource / faster-compiling candidate; do not report a "
            "winner's-curse ordering from single minimum samples"
        ),
        "requires_confirmation": True,
    }


def confirmation_plan(
    *, top_k: Sequence[str], default_candidates: Sequence[str], runs_per_cell: int = 3
) -> Dict[str, Any]:
    if runs_per_cell < 3:
        raise ConfigError("confirmation needs at least 3 independent runs per cell")
    return {
        "candidates": list(top_k) + [item for item in default_candidates if item not in top_k],
        "runs_per_cell": runs_per_cell,
        "process": "new process",
        "order": "randomised across candidates",
        "samples": "fresh samples only: search samples are never reused for confirmation",
        "recompile_or_load": "recompile, or load the exact artifact with a verified hash",
    }


# ── holdout evaluation and cost (steps 25–30, §10) ─────────────────────────


def holdout_evaluate(
    *,
    kind: str,
    shared_winner_id: str,
    rows: Sequence[Mapping[str, Any]],
    regret_threshold: float = 0.10,
    default_key: str = "default_latency_ms",
    winner_key: str = "winner_latency_ms",
    oracle_key: str = "oracle_latency_ms",
) -> Dict[str, Any]:
    """Evaluate a shared winner on unseen shapes/regimes with a pre-registered gate."""
    if kind not in HOLDOUT_KINDS:
        raise ConfigError(f"unknown holdout kind {kind!r}")
    per_case: List[Dict[str, Any]] = []
    for row in rows:
        default = row.get(default_key)
        winner = row.get(winner_key)
        oracle = row.get(oracle_key)
        slowdown = None
        regret = None
        if default and winner is not None:
            slowdown = winner / default - 1.0
        if oracle and winner is not None:
            regret = winner / oracle - 1.0
        per_case.append(
            {
                **dict(row),
                "slowdown_vs_default": None if slowdown is None else round(slowdown, 4),
                "regret_vs_oracle": None if regret is None else round(regret, 4),
                "exceeds_threshold": None if regret is None else regret > regret_threshold,
            }
        )
    exceeded = [row for row in per_case if row["exceeds_threshold"]]
    return {
        "kind": kind,
        "shared_winner": shared_winner_id,
        "cases": per_case,
        "exceeded": [row.get("case") for row in exceeded],
        "regret_threshold": regret_threshold,
        "verdict": "PASS" if not exceeded else "FALLBACK_REQUIRED",
        "fallback": (
            "per-shape tuning, a bucket-specific winner, or the default candidate; the failed "
            "holdout result is kept, not edited"
        ),
    }


def search_cost_breakdown(
    *,
    generation_s: float,
    static_filter_s: float,
    build_compile_s: float,
    correctness_s: float,
    benchmark_s: float,
    confirmation_s: float,
    cache_write_s: float,
    host_cpu_seconds: float,
    device_seconds: float,
    disk_bytes: int,
) -> Dict[str, Any]:
    stages = {
        "generation": generation_s,
        "static_filter": static_filter_s,
        "build_compile": build_compile_s,
        "correctness": correctness_s,
        "benchmark": benchmark_s,
        "confirmation": confirmation_s,
        "cache_write": cache_write_s,
    }
    return {
        "stages": {key: round(value, 6) for key, value in stages.items()},
        "total_wall_s": round(sum(stages.values()), 6),
        "host_cpu_seconds": host_cpu_seconds,
        "device_seconds": device_seconds,
        "disk_bytes": disk_bytes,
        "note": "search cost is reported separately from steady runtime and from model benefit",
    }


def autotune_metrics(rows: Sequence[AutotuneTrialRecord]) -> Dict[str, Any]:
    legal = [row for row in rows if row.legality_status == "legal"]
    compiled = [row for row in legal if row.compile_status == "ok"]
    correct = [row for row in compiled if row.correctness_status == "pass"]
    measured = [row for row in correct if row.raw_timing_samples]
    return {
        "generated": len(rows),
        "statically_legal": len(legal),
        "compiled": len(compiled),
        "correct": len(correct),
        "measured": len(measured),
        "compile_success_rate": round(len(compiled) / max(1, len(legal)), 4),
        "correct_rate": round(len(correct) / max(1, len(compiled)), 4),
        "valid_measure_rate": round(len(measured) / max(1, len(correct)), 4),
        "invalid_reasons": _reason_histogram(row for row in rows if row.legality_status == "invalid"),
        "compile_failures": _reason_histogram(
            row for row in legal if row.compile_status not in ("ok", "")
        ),
        "correctness_failures": _reason_histogram(
            row for row in compiled if row.correctness_status not in ("pass", "")
        ),
        "rule": "each failure class has its own meaning and must not be collapsed into cost",
    }


def _reason_histogram(rows: Iterable[AutotuneTrialRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        reason = row.reject_reason or row.compile_status or row.correctness_status or "unlabelled"
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def quality_cost_curve(
    *, rows: Sequence[Mapping[str, Any]], oracle_latency_ms: Optional[float]
) -> Dict[str, Any]:
    """Budget → best confirmed latency / regret / overhead (step 22)."""
    curve = []
    for row in rows:
        budget = row.get("budget", "")
        best = row.get("best_confirmed_median_ms")
        regret = None
        if best is not None and oracle_latency_ms:
            regret = round(best / oracle_latency_ms - 1.0, 4)
        curve.append(
            {
                "budget": budget,
                "device_seconds": row.get("device_seconds"),
                "trial_count": row.get("trial_count"),
                "best_confirmed_median_ms": best,
                "regret_vs_oracle": regret,
                "compile_failures": row.get("compile_failures"),
                "correctness_failures": row.get("correctness_failures"),
            }
        )
    return {
        "curve": curve,
        "oracle_latency_ms": oracle_latency_ms,
        "oracle_scope": "frozen candidate set under the frozen measurement protocol",
    }


def amortization(
    *,
    phase: str,
    search_cost_s: float,
    per_call_saving_s: float,
    call_count: float,
) -> Dict[str, Any]:
    """Break-even per phase; ``NEVER`` when there is no per-call saving."""
    row = amortization_row(
        phase=phase,
        cold_total_s=search_cost_s,
        warm_load_s=0.0,
        baseline_steady_s=per_call_saving_s + 0.001,
        compiled_steady_s=0.001,
        call_count=call_count,
    )
    calls = break_even_calls(search_cost_s, per_call_saving_s)
    return {
        **row,
        "search_cost_s": search_cost_s,
        "break_even_calls": None if calls is None else round(calls, 3),
        "per_phase_rule": "prefill and decode have different call frequencies: never average them",
    }


# ── tuning database and policy (steps 31–32) ───────────────────────────────

TUNING_DB_KEY_FIELDS: Tuple[str, ...] = (
    "semantic_op",
    "schema_version",
    "shape_domain_id",
    "target_id",
    "target_arch",
    "compiler_versions",
    "kernel_build_id",
    "search_space_id",
    "search_space_version",
    "measurement_policy_id",
    "correctness_policy_id",
)


def tuning_database_record(
    *,
    key: Mapping[str, Any],
    winner_config: Mapping[str, Any],
    winner_identity: str,
    confirmation_status: str,
    invalidation_dependencies: Sequence[str],
    selection_status: str = "confirmed",
) -> Dict[str, Any]:
    missing = [name for name in TUNING_DB_KEY_FIELDS if name not in key]
    if missing:
        raise ConfigError(f"tuning DB key missing fields: {missing}")
    return {
        "key": {name: key[name] for name in TUNING_DB_KEY_FIELDS},
        "winner_config": dict(sorted(winner_config.items())),
        "winner_identity": winner_identity,
        "confirmation_status": confirmation_status,
        "selection_status": selection_status,
        "invalidation_dependencies": list(invalidation_dependencies),
        "record_digest": sha256_text(
            canonical_json({"key": {name: key[name] for name in TUNING_DB_KEY_FIELDS},
                            "winner": winner_identity})
        ),
    }


def autotune_policy_document(
    *,
    per_shape_tune_when: Sequence[str],
    shared_config_when: Sequence[str],
    budget: Budget,
    holdout_failure_action: str,
    online_tuning_forbidden: bool = True,
) -> Dict[str, Any]:
    return {
        "per_shape_tune_when": list(per_shape_tune_when),
        "shared_config_when": list(shared_config_when),
        "budget": budget.as_dict(),
        "holdout_failure_action": holdout_failure_action,
        "online_tuning_forbidden": online_tuning_forbidden,
        "stateful_kernel_rule": (
            "stateful kernels are never tuned online: candidates would mutate residual/KV as a "
            "side effect of the search"
        ),
    }


def holdout_kinds_table() -> List[Dict[str, str]]:
    return [
        {"kind": "interpolation", "definition": "unseen shape in the same regime"},
        {"kind": "boundary_extrapolation", "definition": "tile/bucket edges and longer legal shapes"},
        {"kind": "layout_dtype", "definition": "only when the config claims to be shareable"},
        {"kind": "target", "definition": "only with real second-architecture data"},
    ]


def split_summary(rows: Sequence[AutotuneTrialRecord]) -> Dict[str, Any]:
    counts: Dict[str, int] = {name: 0 for name in AUTOTUNE_SPLITS}
    for row in rows:
        counts[row.split] = counts.get(row.split, 0) + 1
    return {"counts": counts, "total": len(rows)}
