"""Cost model: features, splits, baselines, oracle regret and safe fallback.

Protocol anchor: ``details/S11/E11-07_cost_model_regret_safe_selection.md``.
The primary metric is **chosen-vs-oracle regret on an untouched holdout**, not
prediction loss.  The single hard rule is §18 of the details README:

    a cost model may only rank candidates that already passed semantic
    legality, target capability, guards and correctness evidence — it can
    never make an illegal candidate legal.

Implemented instruments:

* feature schema with availability time (pre-compile / post-compile /
  post-run) and an explicit forbidden list for leakage audits;
* dataset units, robust labels (median/CI ties) and group-based splits with an
  "untouched final test" guard;
* baseline evaluators (default/heuristic/random/global-best/nearest/oracle);
* small deterministic model families (constant, linear regression, pairwise
  ranker) so a holdout comparison is possible without heavy dependencies;
* regret/ranking metrics, confidence + OOD detection and risk-coverage;
* a fail-closed selection policy (no legal candidate → semantic fallback;
  OOD/missing → heuristic; low confidence → benchmark top-k or fallback);
* adversarial tests: illegal-candidate injection, corrupt model artifact,
  selection overhead accounting and the deployment verdict split into
  "methodology pass" vs "deployment pass".

No training is performed at import time and no numbers are fabricated: all
fitters return their own training/holdout split information.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import canonical_json, sha256_text

# ── feature schema (E11-07 §5, step 4) ─────────────────────────────────────

FEATURE_AVAILABILITY: Tuple[str, ...] = ("pre_compile", "post_compile", "post_run")

FEATURE_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "workload_shape": (
        "B",
        "M",
        "N",
        "K",
        "H",
        "head_dim",
        "q_len",
        "kv_len",
        "tail_remainder",
        "ratio",
        "phase",
        "call_frequency_bucket",
        "symbolic_range_width",
    ),
    "dtype_layout": (
        "input_dtype",
        "output_dtype",
        "accum_dtype",
        "element_bytes",
        "stride_class",
        "contiguous",
        "alignment",
        "reduction_axis",
        "quant_layout",
    ),
    "candidate_schedule": (
        "backend",
        "tile_m",
        "tile_n",
        "tile_k",
        "num_warps",
        "num_stages",
        "vector_width",
        "fusion_group_size",
        "reduction_strategy",
        "workspace_bytes",
        "specialization_breadth",
    ),
    "target": (
        "arch_family",
        "sm_count",
        "warp_size",
        "register_limit",
        "shared_limit",
        "memory_bandwidth_class",
        "cache_class",
        "supported_instructions",
    ),
    "ir_static_resource": (
        "op_count",
        "estimated_flops",
        "estimated_bytes",
        "arithmetic_intensity",
        "live_buffers",
        "buffer_reuse",
        "static_registers",
        "static_shared",
        "static_local",
        "code_size",
        "expected_waves",
    ),
}

#: Fields that must never enter a pre-run model (leakage list, step 5).
LEAKAGE_FORBIDDEN: Tuple[str, ...] = (
    "actual_latency",
    "measured_latency",
    "latency_ms",
    "winner",
    "winner_flag",
    "oracle_rank",
    "oracle_latency",
    "post_run_counters",
    "ncu_metrics",
    "artifact_path_label",
    "test_group_id",
    "split_id",
    "cache_key",
)

#: Fields whose *absence* forces a fallback (critical features).
CRITICAL_PRE_COMPILE_FEATURES: Tuple[str, ...] = ("input_dtype", "phase", "H")


@dataclass
class FeatureField:
    name: str
    group: str
    availability: str
    unit: str = ""
    missing_policy: str = "unknown"
    target_specific: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.group not in FEATURE_GROUPS:
            problems.append(f"{self.name}: unknown feature group {self.group!r}")
        if self.availability not in FEATURE_AVAILABILITY:
            problems.append(f"{self.name}: unknown availability {self.availability!r}")
        if self.name in LEAKAGE_FORBIDDEN:
            problems.append(f"{self.name}: listed in the leakage-forbidden set")
        if self.missing_policy not in ("unknown", "fallback", "reject"):
            problems.append(f"{self.name}: missing policy must be unknown/fallback/reject")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "availability": self.availability,
            "unit": self.unit,
            "missing_policy": self.missing_policy,
            "target_specific": self.target_specific,
        }


@dataclass
class FeatureSchema:
    version: str
    fields: Tuple[FeatureField, ...]

    def validate(self) -> List[str]:
        problems = [problem for field_ in self.fields for problem in field_.validate()]
        names = [field_.name for field_ in self.fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        problems.extend(f"duplicate feature {name!r}" for name in duplicates)
        if not self.version:
            problems.append("feature schema must be versioned")
        return problems

    def digest(self) -> str:
        return sha256_text(canonical_json([field_.as_dict() for field_ in self.fields]))

    def by_name(self, name: str) -> FeatureField:
        for field_ in self.fields:
            if field_.name == name:
                return field_
        raise ConfigError(f"unknown feature {name!r}")

    def missing_critical(self, row: Mapping[str, Any]) -> List[str]:
        return [
            name
            for name in CRITICAL_PRE_COMPILE_FEATURES
            if name in {field_.name for field_ in self.fields}
            and self.by_name(name).availability == "pre_compile"
            and row.get(name) in (None, "")
        ]


def default_feature_schema() -> FeatureSchema:
    fields: List[FeatureField] = []
    for group, names in FEATURE_GROUPS.items():
        for name in names:
            availability = "pre_compile"
            if name in ("static_registers", "static_shared", "static_local", "code_size",
                        "expected_waves", "workspace_bytes"):
                availability = "post_compile"
            if name in ("post_run_counters",):
                availability = "post_run"
            fields.append(
                FeatureField(
                    name=name,
                    group=group,
                    availability=availability,
                    missing_policy="fallback" if name in CRITICAL_PRE_COMPILE_FEATURES else "unknown",
                )
            )
    return FeatureSchema(version="1.0.0", fields=tuple(fields))


def leakage_audit(rows: Sequence[Mapping[str, Any]], schema: FeatureSchema) -> Dict[str, Any]:
    """Static audit of dataset rows/field names against the forbidden list."""
    problems: List[Dict[str, Any]] = []
    known = {field_.name for field_ in schema.fields}
    for index, row in enumerate(rows):
        for key in row:
            if key in LEAKAGE_FORBIDDEN:
                problems.append({"row": index, "field": key, "issue": "forbidden feature present"})
            elif key not in known and key not in ("label", "tie", "task_id", "shape_group", "split"):
                problems.append({"row": index, "field": key, "issue": "undeclared feature"})
    return {
        "rows": len(rows),
        "problems": problems,
        "clean": not problems,
        "rule": (
            "feature availability must be decided before selection: a pre-compile model may "
            "not see post-run counters"
        ),
    }


# ── labels and splits (steps 3, 6–7) ───────────────────────────────────────


@dataclass
class DatasetUnit:
    task_id: str
    shape_group: str
    target_id: str
    candidate_id: str
    features: Mapping[str, Any]
    samples: Tuple[float, ...]
    split: str = "tuning"

    def median(self) -> float:
        ordered = sorted(self.samples)
        n = len(ordered)
        if n == 0:
            raise ConfigError("unit without samples cannot produce a label")
        mid = n // 2
        return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0

    def ci95(self) -> Tuple[float, float]:
        n = len(self.samples)
        if n < 2:
            return (self.median(), self.median())
        mean = sum(self.samples) / n
        variance = sum((value - mean) ** 2 for value in self.samples) / (n - 1)
        half = 1.96 * math.sqrt(variance) / math.sqrt(n)
        return (mean - half, mean + half)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "shape_group": self.shape_group,
            "target_id": self.target_id,
            "candidate_id": self.candidate_id,
            "features": dict(sorted(self.features.items())),
            "samples": list(self.samples),
            "split": self.split,
            "label_median": round(self.median(), 6),
        }


def build_labels(
    units: Sequence[DatasetUnit], *, tie_overlap: bool = True, normalized: bool = True
) -> Dict[str, Any]:
    """Robust labels: median, normalised cost and CI-overlap ties (step 6)."""
    by_task: Dict[str, List[DatasetUnit]] = {}
    for unit in units:
        by_task.setdefault(unit.task_id, []).append(unit)
    rows: List[Dict[str, Any]] = []
    for task_id, items in sorted(by_task.items()):
        default_median = next((item.median() for item in items if item.candidate_id == "default"), None)
        for item in items:
            low, high = item.ci95()
            rows.append(
                {
                    "task_id": task_id,
                    "shape_group": item.shape_group,
                    "candidate_id": item.candidate_id,
                    "label_median": round(item.median(), 6),
                    "normalized_vs_default": (
                        None
                        if not normalized or not default_median
                        else round(item.median() / default_median, 6)
                    ),
                    "ci95": [round(low, 6), round(high, 6)],
                    "oracle_rank": sorted(
                        (other.median(), other.candidate_id) for other in items
                    ).index((item.median(), item.candidate_id)),
                }
            )
    ties: List[Dict[str, Any]] = []
    if tie_overlap:
        for task_id, items in sorted(by_task.items()):
            for index, left in enumerate(items):
                for right in items[index + 1 :]:
                    low_a, high_a = left.ci95()
                    low_b, high_b = right.ci95()
                    if low_a <= high_b and low_b <= high_a:
                        ties.append(
                            {
                                "task_id": task_id,
                                "a": left.candidate_id,
                                "b": right.candidate_id,
                                "rule": "CI overlap: label as tie instead of manufacturing an order",
                            }
                        )
    return {"rows": rows, "ties": ties, "tie_count": len(ties)}


SPLIT_STRATEGIES: Tuple[str, ...] = (
    "random_row_diagnostic",
    "leave_shape_group_out",
    "leave_regime_out",
    "leave_graph_pattern_out",
    "leave_arch_out",
)

PRIMARY_SPLITS: Tuple[str, ...] = (
    "leave_shape_group_out",
    "leave_regime_out",
    "leave_graph_pattern_out",
    "leave_arch_out",
)


@dataclass
class SplitManifest:
    strategy: str
    train: Tuple[str, ...]
    validation: Tuple[str, ...]
    final_test: Tuple[str, ...]
    locked: bool = False
    final_test_consumed: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.strategy not in SPLIT_STRATEGIES:
            problems.append(f"unknown split strategy {self.strategy!r}")
        if self.strategy == "random_row_diagnostic":
            problems.append(
                "random-row splits are leakage-sensitive diagnostics, not a primary result"
            )
        groups = [set(self.train), set(self.validation), set(self.final_test)]
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                if left & right:
                    problems.append(f"split overlap: {sorted(left & right)}")
        if not self.locked:
            problems.append("the split must be locked before training")
        return problems

    def consume_final_test(self) -> None:
        if self.final_test_consumed:
            raise ConfigError("the final test has already been consumed; it is no longer untouched")
        self.final_test_consumed = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "train": list(self.train),
            "validation": list(self.validation),
            "final_test": list(self.final_test),
            "locked": self.locked,
            "final_test_consumed": self.final_test_consumed,
        }


def build_split(
    *,
    strategy: str,
    group_keys: Mapping[str, str],
    holdout_keys: Sequence[str],
    validation_keys: Sequence[str] = (),
) -> SplitManifest:
    """Build a group-level split; a group never spans two splits."""
    if strategy not in PRIMARY_SPLITS:
        raise ConfigError(
            f"{strategy!r} cannot be the primary split; the primary split must be group-based"
        )
    holdout = set(holdout_keys)
    validation = set(validation_keys)
    if holdout & validation:
        raise ConfigError("holdout and validation groups overlap")
    train = tuple(
        unit for unit, key in sorted(group_keys.items()) if key not in holdout | validation
    )
    validation_units = tuple(
        unit for unit, key in sorted(group_keys.items()) if key in validation
    )
    test_units = tuple(unit for unit, key in sorted(group_keys.items()) if key in holdout)
    return SplitManifest(
        strategy=strategy,
        train=train,
        validation=validation_units,
        final_test=test_units,
        locked=True,
    )


# ── baselines and models (steps 8–12) ──────────────────────────────────────

BASELINES: Tuple[str, ...] = (
    "default",
    "heuristic",
    "random_legal",
    "global_best",
    "nearest_shape",
    "oracle",
)

STRONG_BASELINES: Tuple[str, ...] = ("default", "heuristic", "oracle")


def heuristic_pick(candidates: Sequence[str], features: Mapping[str, Any]) -> str:
    """Transparent rule: decode (q_len==1) prefers the small-tile candidate."""
    if features.get("phase") == "decode" and "small_tile" in candidates:
        return "small_tile"
    if "default" in candidates:
        return "default"
    return sorted(candidates)[0]


def evaluate_baseline(
    *,
    name: str,
    candidates: Sequence[str],
    features: Mapping[str, Any],
    latencies: Mapping[str, float],
    seed: int = 0,
    history: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Evaluate one baseline selector against the frozen candidate set."""
    if name not in BASELINES:
        raise ConfigError(f"unknown baseline {name!r}")
    legal = [item for item in candidates if latencies.get(item) is not None]
    if not legal:
        raise ConfigError("no legal candidate with a latency: baseline cannot be evaluated")
    if name == "default":
        chosen = "default" if "default" in legal else sorted(legal)[0]
    elif name == "heuristic":
        chosen = heuristic_pick(legal, features)
    elif name == "random_legal":
        chosen = random.Random(seed).choice(sorted(legal))
    elif name == "global_best":
        chosen = min(legal, key=lambda item: latencies[item])
    elif name == "nearest_shape":
        chosen = (history or {}).get(str(features.get("shape_group", "")), "")
        if chosen not in legal:
            chosen = "default" if "default" in legal else sorted(legal)[0]
    else:  # oracle
        chosen = min(legal, key=lambda item: latencies[item])
    return {
        "baseline": name,
        "chosen": chosen,
        "latency": latencies[chosen],
        "oracle_latency": min(latencies[item] for item in legal),
        "regret": round(latencies[chosen] / min(latencies[item] for item in legal) - 1.0, 6),
    }


def evaluate_baselines(
    *,
    candidates: Sequence[str],
    features: Mapping[str, Any],
    latencies: Mapping[str, float],
    seed: int = 0,
) -> Dict[str, Any]:
    rows = [
        evaluate_baseline(
            name=name, candidates=candidates, features=features, latencies=latencies, seed=seed
        )
        for name in BASELINES
    ]
    oracle = next(row for row in rows if row["baseline"] == "oracle")
    return {
        "rows": rows,
        "oracle_latency": oracle["latency"],
        "note": "beating random_legal alone is not evidence; the strong baselines are default/heuristic/oracle",
    }


class Predictor:
    """Minimal predictor protocol (fit/predict/describe)."""

    name = "predictor"

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "Predictor":  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, row: Mapping[str, Any]) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class ConstantPredictor(Predictor):
    """Predicts the training mean: the sanity floor for every comparison."""

    name = "constant"

    def __init__(self) -> None:
        self.value = 0.0
        self.trained = False

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "ConstantPredictor":
        values = [float(row["label"]) for row in rows]
        self.value = sum(values) / len(values) if values else 0.0
        self.trained = True
        return self

    def predict(self, row: Mapping[str, Any]) -> float:
        if not self.trained:
            raise ConfigError("predictor used before fit")
        return self.value

    def describe(self) -> Dict[str, Any]:
        return {"model": "constant", "value": self.value, "trained": self.trained}


class LinearRanker(Predictor):
    """Standardised linear regression on numeric features (deterministic)."""

    name = "linear"

    def __init__(self) -> None:
        self.weights: Dict[str, float] = {}
        self.means: Dict[str, float] = {}
        self.scales: Dict[str, float] = {}
        self.bias = 0.0
        self.trained = False

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "LinearRanker":
        if not rows:
            raise ConfigError("cannot fit on an empty training set")
        keys = sorted(
            key
            for key in rows[0]
            if key not in ("label", "tie", "split", "task_id", "shape_group", "candidate_id")
            and isinstance(rows[0][key], (int, float))
        )
        for key in keys:
            values = [float(row.get(key, 0.0)) for row in rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            self.means[key] = mean
            self.scales[key] = math.sqrt(variance) or 1.0
        weights = {key: 0.0 for key in keys}
        labels = [float(row["label"]) for row in rows]
        bias = sum(labels) / len(labels)
        lr = 0.05
        for _ in range(200):
            grad = {key: 0.0 for key in keys}
            grad_bias = 0.0
            for row, label in zip(rows, labels):
                prediction = bias + sum(
                    weights[key] * self._standardise(key, row) for key in keys
                )
                error = prediction - label
                grad_bias += error
                for key in keys:
                    grad[key] += error * self._standardise(key, row)
            scale = 1.0 / len(rows)
            bias -= lr * grad_bias * scale
            for key in keys:
                weights[key] -= lr * grad[key] * scale
        self.weights = weights
        self.bias = bias
        self.trained = True
        return self

    def _standardise(self, key: str, row: Mapping[str, Any]) -> float:
        value = row.get(key, self.means[key])
        if not isinstance(value, (int, float)):
            value = self.means[key]
        return (float(value) - self.means[key]) / self.scales[key]

    def predict(self, row: Mapping[str, Any]) -> float:
        if not self.trained:
            raise ConfigError("predictor used before fit")
        return self.bias + sum(
            weight * self._standardise(key, row) for key, weight in self.weights.items()
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "model": "linear",
            "bias": self.bias,
            "weights": dict(sorted(self.weights.items())),
            "trained": self.trained,
        }


class PairwiseRanker(Predictor):
    """Perceptron-style pairwise ranker over candidate feature deltas."""

    name = "pairwise"

    def __init__(self, *, seed: int = 0) -> None:
        self.weights: Dict[str, float] = {}
        self.trained = False
        self.seed = seed

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "PairwiseRanker":
        by_task: Dict[str, List[Mapping[str, Any]]] = {}
        for row in rows:
            by_task.setdefault(str(row["task_id"]), []).append(row)
        keys = sorted(
            key
            for key in (rows[0] if rows else {})
            if key not in ("label", "tie", "split", "task_id", "shape_group", "candidate_id")
            and isinstance(rows[0][key], (int, float))
        )
        weights = {key: 0.0 for key in keys}
        pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for items in by_task.values():
            ordered = sorted(items, key=lambda item: float(item["label"]))
            for index, better in enumerate(ordered):
                for worse in ordered[index + 1 :]:
                    pairs.append((better, worse))
        rng = random.Random(self.seed)
        for _ in range(50):
            rng.shuffle(pairs)
            for better, worse in pairs:
                margin = sum(
                    weights[key]
                    * (float(worse.get(key, 0.0)) - float(better.get(key, 0.0)))
                    for key in keys
                )
                # perceptron update whenever the pair is not yet ordered correctly
                # (margin <= 0 covers the all-zero initialisation)
                if margin <= 0:
                    for key in keys:
                        weights[key] += 0.01 * (
                            float(better.get(key, 0.0)) - float(worse.get(key, 0.0))
                        )
        self.weights = weights
        self.trained = True
        return self

    def predict(self, row: Mapping[str, Any]) -> float:
        if not self.trained:
            raise ConfigError("predictor used before fit")
        score = sum(
            weight * float(row.get(key, 0.0)) for key, weight in self.weights.items()
        )
        return -score  # lower score ⇒ lower predicted cost

    def describe(self) -> Dict[str, Any]:
        return {"model": "pairwise", "weights": dict(sorted(self.weights.items()))}


def model_families(seed: int = 0) -> List[Predictor]:
    return [ConstantPredictor(), LinearRanker(), PairwiseRanker(seed=seed)]


# ── metrics (E11-07 §11) ───────────────────────────────────────────────────


def relative_regret(chosen_latency: float, oracle_latency: float) -> Optional[float]:
    if oracle_latency <= 0:
        return None
    return chosen_latency / oracle_latency - 1.0


@dataclass
class RegretReport:
    per_case: Tuple[Mapping[str, Any], ...]
    weights: Tuple[float, ...] = ()
    catastrophic_threshold: float = 0.5

    def summary(self) -> Dict[str, Any]:
        regrets = [row["regret"] for row in self.per_case if row.get("regret") is not None]
        if not regrets:
            return {
                "cases": 0,
                "note": "no case with a measured oracle: regret is UNAVAILABLE",
            }
        weights = list(self.weights) or [1.0] * len(regrets)
        if len(weights) != len(regrets):
            raise ConfigError("weights must match the number of cases")
        weight_sum = sum(weights)
        weighted = sum(w * r for w, r in zip(weights, regrets)) / weight_sum
        ordered = sorted(regrets)
        p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
        catastrophic = [r for r in regrets if r > self.catastrophic_threshold]
        return {
            "cases": len(regrets),
            "mean": round(sum(regrets) / len(regrets), 6),
            "median": round(ordered[len(ordered) // 2], 6),
            "p95": round(p95, 6),
            "max": round(max(regrets), 6),
            "weighted_mean": round(weighted, 6),
            "catastrophic_threshold": self.catastrophic_threshold,
            "catastrophic_rate": round(len(catastrophic) / len(regrets), 6),
            "rule": "p95/max/catastrophic are reported next to the mean; the mean hides long tails",
        }


def speedup_capture(*, baseline: float, selected: float, oracle: float) -> Optional[float]:
    """Fraction of the oracle's achievable gain captured (None if no headroom)."""
    headroom = baseline - oracle
    if headroom <= 0:
        return None
    return (baseline - selected) / headroom


def ranking_metrics(
    *,
    predictions: Sequence[Tuple[str, float]],
    true_latencies: Mapping[str, float],
    k: int = 2,
) -> Dict[str, Any]:
    """top-1/top-k accuracy, pairwise accuracy and tie-aware Kendall tau."""
    ordered = sorted(predictions, key=lambda item: item[1])
    predicted_order = [name for name, _ in ordered]
    true_order = [name for name, _ in sorted(true_latencies.items(), key=lambda item: item[1])]
    oracle = true_order[0]
    top1 = predicted_order[0] == oracle if predicted_order else False
    topk = oracle in predicted_order[:k]
    pairs = [(a, b) for index, a in enumerate(true_order) for b in true_order[index + 1 :]]
    concordant = 0
    ties = 0
    for left, right in pairs:
        if true_latencies.get(left) == true_latencies.get(right):
            ties += 1
            continue
        rank_left = predicted_order.index(left)
        rank_right = predicted_order.index(right)
        if rank_left < rank_right:
            concordant += 1
    comparable = max(1, len(pairs) - ties)
    return {
        "top1": top1,
        "topk": topk,
        "k": k,
        "pairwise_accuracy": round(concordant / comparable, 4),
        "pair_count": len(pairs),
        "ties": ties,
        "kendall_tau_tie_aware": round(
            (concordant - (comparable - concordant)) / comparable, 4
        ),
        "note": "ranking metrics are diagnostics; the primary metric stays oracle regret",
    }


# ── confidence, OOD and fallback (steps 14, 23–26) ─────────────────────────


@dataclass
class ConfidenceModel:
    """Top-2 margin plus a train-fitted feature range for OOD detection."""

    margins: Tuple[float, ...] = ()
    threshold: Optional[float] = None
    feature_ranges: Mapping[str, Tuple[float, float]] = field(default_factory=dict)

    def calibrate(self, rows: Sequence[Mapping[str, Any]], threshold: float) -> "ConfidenceModel":
        self.threshold = threshold
        self.margins = tuple(float(row["top2_margin"]) for row in rows if "top2_margin" in row)
        return self

    def fit_ranges(self, rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> "ConfidenceModel":
        ranges: Dict[str, Tuple[float, float]] = {}
        for key in keys:
            values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
            if values:
                ranges[key] = (min(values), max(values))
        self.feature_ranges = ranges
        return self

    def ood_score(self, row: Mapping[str, Any]) -> Tuple[float, bool]:
        violations = [
            key
            for key, (low, high) in self.feature_ranges.items()
            if isinstance(row.get(key), (int, float)) and not (low <= float(row[key]) <= high)
        ]
        return (float(len(violations)), bool(violations))

    def risk_coverage(self, cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Risk-coverage curve: coverage down ⇒ tail regret must go down."""
        if self.threshold is None:
            raise ConfigError("confidence threshold must be calibrated before use")
        rows: List[Dict[str, Any]] = []
        thresholds = sorted({float(case.get("confidence", 0.0)) for case in cases} | {self.threshold})
        for threshold in thresholds:
            covered = [case for case in cases if float(case.get("confidence", 0.0)) >= threshold]
            regrets = [
                float(case["regret"]) for case in covered if case.get("regret") is not None
            ]
            rows.append(
                {
                    "threshold": threshold,
                    "coverage": round(len(covered) / max(1, len(cases)), 4),
                    "mean_regret": round(sum(regrets) / len(regrets), 6) if regrets else None,
                    "p95_regret": (
                        round(
                            sorted(regrets)[
                                min(len(regrets) - 1, math.ceil(0.95 * len(regrets)) - 1)
                            ],
                            6,
                        )
                        if regrets
                        else None
                    ),
                    "max_regret": round(max(regrets), 6) if regrets else None,
                }
            )
        monotone = _is_risk_monotone(rows)
        return {
            "threshold": self.threshold,
            "rows": rows,
            "risk_monotone": monotone,
            "note": (
                "if lowering coverage does not lower tail regret, the confidence signal is "
                "not useful and must not gate deployment"
            ),
        }


def _is_risk_monotone(rows: Sequence[Mapping[str, Any]], tolerance: float = 1e-9) -> bool:
    """Risk-coverage monotonicity: more coverage may not lower tail risk.

    Rows are ordered by ascending coverage, so the tail risk must be
    non-decreasing: if a *smaller* covered set had a *higher* p95 regret, the
    confidence signal is anti-correlated with risk and must not gate
    deployment.
    """
    previous: Optional[float] = None
    for row in sorted(rows, key=lambda item: item["coverage"]):
        value = row.get("p95_regret")
        if value is None:
            continue
        if previous is not None and value < previous - tolerance:
            return False
        previous = value
    return True


# ── selection policy (steps 24–25, §3.5) ───────────────────────────────────

FALLBACK_ACTIONS: Tuple[str, ...] = (
    "semantic_fallback",
    "heuristic_default",
    "benchmark_topk",
    "model_selected",
)

POLICY_STEPS: Tuple[str, ...] = (
    "no_legal_candidates",
    "ood_or_missing_feature",
    "low_confidence",
    "model_selected",
)


def select_with_policy(
    *,
    eligible: Sequence[str],
    predictions: Mapping[str, float],
    confidence: float,
    confidence_threshold: float,
    ood: bool,
    missing_critical: Sequence[str] = (),
    topk_benchmark_candidates: Sequence[str] = (),
    heuristic_choice: str = "default",
) -> Dict[str, Any]:
    """Fail-closed decision chain; the model never widens the eligible set."""
    if not eligible:
        return {
            "action": "semantic_fallback",
            "chosen": "",
            "reason": "no legal candidate: fall back to the reference/original subgraph",
            "fallback_triggered": True,
        }
    if ood or missing_critical:
        return {
            "action": "heuristic_default",
            "chosen": heuristic_choice if heuristic_choice in eligible else sorted(eligible)[0],
            "reason": (
                "OOD input or missing critical feature"
                if ood
                else f"missing critical features: {list(missing_critical)}"
            ),
            "fallback_triggered": True,
        }
    if confidence < confidence_threshold:
        if topk_benchmark_candidates:
            return {
                "action": "benchmark_topk",
                "chosen": "",
                "candidates": list(topk_benchmark_candidates),
                "reason": "low confidence: measure a few candidates instead of trusting the model",
                "fallback_triggered": True,
            }
        return {
            "action": "heuristic_default",
            "chosen": heuristic_choice if heuristic_choice in eligible else sorted(eligible)[0],
            "reason": "low confidence without a measurement budget",
            "fallback_triggered": True,
        }
    legal_predictions = {
        name: value for name, value in predictions.items() if name in eligible
    }
    if not legal_predictions:
        return {
            "action": "heuristic_default",
            "chosen": heuristic_choice if heuristic_choice in eligible else sorted(eligible)[0],
            "reason": "model produced no prediction for the eligible set",
            "fallback_triggered": True,
        }
    chosen = min(sorted(legal_predictions), key=lambda name: legal_predictions[name])
    return {
        "action": "model_selected",
        "chosen": chosen,
        "reason": "model selected within the legal, evidence-covered set",
        "fallback_triggered": False,
    }


def illegal_candidate_injection(
    *,
    eligible: Sequence[str],
    illegal: str,
    mock_predictions: Mapping[str, float],
    policy_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Adversarial test: the cheapest illegal candidate must still be blocked."""
    decision = select_with_policy(
        eligible=eligible, predictions=mock_predictions, **policy_kwargs  # type: ignore[arg-type]
    )
    blocked = decision["chosen"] != illegal
    return {
        "illegal_candidate": illegal,
        "decision": decision,
        "blocked": blocked,
        "rule": (
            "the eligible set is computed upstream from legality/capability/guard/evidence; a "
            "cost model has no authority to re-admit a candidate"
        ),
    }


@dataclass
class ModelArtifactGuard:
    """Version/hash guard for loading a cost-model artifact (step 26)."""

    expected_hash: str
    expected_schema_version: str
    expected_feature_version: str

    def check(
        self, *, artifact_hash: str, schema_version: str, feature_version: str
    ) -> Dict[str, Any]:
        problems: List[str] = []
        if artifact_hash != self.expected_hash:
            problems.append("artifact hash mismatch")
        if schema_version != self.expected_schema_version:
            problems.append("schema version mismatch")
        if feature_version != self.expected_feature_version:
            problems.append("feature version mismatch")
        return {
            "ok": not problems,
            "problems": problems,
            "action": "use heuristic/default" if problems else "load model",
            "rule": "a corrupt or stale model is disabled, never loaded 'to see if it works'",
        }


def selection_overhead(
    *,
    feature_extraction_us: float,
    model_load_us: float,
    inference_us: float,
    candidate_filter_us: float,
    decision_us: float,
    kernel_latency_us: Optional[float] = None,
) -> Dict[str, Any]:
    total = feature_extraction_us + model_load_us + inference_us + candidate_filter_us + decision_us
    ratio = None if kernel_latency_us in (None, 0) else round(total / kernel_latency_us, 6)
    return {
        "components_us": {
            "feature_extraction": feature_extraction_us,
            "model_load": model_load_us,
            "inference": inference_us,
            "candidate_filter": candidate_filter_us,
            "decision": decision_us,
        },
        "total_us": total,
        "selection_overhead_ratio": ratio,
        "note": (
            "for a small decode kernel the selection overhead can dominate: it is reported per "
            "phase, not averaged away"
        ),
    }


# ── deployment verdict (step 32) ───────────────────────────────────────────


@dataclass
class DeploymentCriteria:
    max_weighted_regret: float
    max_p95_regret: float
    max_catastrophic_rate: float
    min_improvement_over_heuristic: float
    confidence_risk_monotone: bool
    overhead_ratio_limit: float = 0.05

    def evaluate(self, summary: Mapping[str, Any], *, overhead_ratio: Optional[float]) -> Dict[str, Any]:
        failures: List[Dict[str, Any]] = []
        checks = {
            "weighted_regret": (summary.get("weighted_mean"), self.max_weighted_regret, "<="),
            "p95_regret": (summary.get("p95"), self.max_p95_regret, "<="),
            "catastrophic_rate": (summary.get("catastrophic_rate"), self.max_catastrophic_rate, "<="),
            "improvement_over_heuristic": (
                summary.get("improvement_over_heuristic"),
                self.min_improvement_over_heuristic,
                ">=",
            ),
            "overhead_ratio": (overhead_ratio, self.overhead_ratio_limit, "<="),
        }
        for name, (value, threshold, direction) in checks.items():
            if value is None:
                failures.append({"check": name, "value": None, "threshold": threshold, "reason": "UNAVAILABLE"})
                continue
            ok = value <= threshold if direction == "<=" else value >= threshold
            if not ok:
                failures.append({"check": name, "value": value, "threshold": threshold})
        if not self.confidence_risk_monotone:
            failures.append(
                {"check": "confidence_risk_monotone", "value": False, "threshold": True}
            )
        return {
            "deploy": not failures,
            "failures": failures,
            "criteria": {
                "max_weighted_regret": self.max_weighted_regret,
                "max_p95_regret": self.max_p95_regret,
                "max_catastrophic_rate": self.max_catastrophic_rate,
                "min_improvement_over_heuristic": self.min_improvement_over_heuristic,
                "overhead_ratio_limit": self.overhead_ratio_limit,
            },
            "note": (
                "methodology PASS and deployment PASS are separate verdicts; a complete study "
                "with a negative deployment decision is a valid result"
            ),
        }


def methodology_verdict(
    *,
    dataset_reproducible: bool,
    final_test_untouched_until_frozen: bool,
    baselines_complete: bool,
    regret_reported: bool,
    risk_coverage_reported: bool,
    illegal_injection_blocked: bool,
    corrupt_model_fallback: bool,
    dispatch_replay_ok: bool,
    overhead_reported: bool,
) -> Dict[str, Any]:
    checks = {
        "dataset_reproducible": dataset_reproducible,
        "final_test_untouched_until_frozen": final_test_untouched_until_frozen,
        "baselines_complete": baselines_complete,
        "regret_reported": regret_reported,
        "risk_coverage_reported": risk_coverage_reported,
        "illegal_injection_blocked": illegal_injection_blocked,
        "corrupt_model_fallback": corrupt_model_fallback,
        "dispatch_replay_ok": dispatch_replay_ok,
        "overhead_reported": overhead_reported,
    }
    failing = sorted(name for name, ok in checks.items() if not ok)
    return {
        "checks": checks,
        "failing": failing,
        "methodology_pass": not failing,
        "deployment_pass": None,
        "note": "deployment is decided separately by DeploymentCriteria.evaluate()",
    }


def error_case_study(
    *, case_id: str, kind: str, missing_feature: str, linked_mechanism: str = ""
) -> Dict[str, Any]:
    if kind not in ("max_regret", "low_confidence_success", "high_confidence_failure"):
        raise ConfigError(f"unknown case-study kind {kind!r}")
    return {
        "case_id": case_id,
        "kind": kind,
        "missing_feature": missing_feature,
        "linked_mechanism": linked_mechanism,
        "note": "high-confidence failures must be studied, never averaged away",
    }


def dataset_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(canonical_json([dict(sorted(row.items())) for row in rows]))
