"""Dynamic shapes, guards, variants, recompile accounting and variant safety.

Protocol anchor: ``details/S11/E11-05_dynamic_shape_guard_recompile_fallback.md``.
The goal is *not* "zero recompiles" and *not* "recompilation is fine": every
variant must have an explicit legal domain, every guard failure must have an
explainable action, and no input may ever execute an incompatible binary.

Implemented instruments:

* the guard taxonomy with per-category failure actions (§4) and expression
  builders from symbolic metadata (step 4);
* ``Variant`` (semantic identity, target, code/binary, guard domain, priority)
  and ``VariantRegistry.lookup`` which *refuses* to call "same graph hash" a
  hit when any required guard is false (§3.3, step 26);
* domain coverage/overlap analysis with priority (step 7);
* compile accounting that requires every new compile to map to a guard/path/
  domain (steps 22, 27) with the ``unexplained_compile_rate`` metric;
* strategy comparison S0–S4 with a deployment-horizon total cost (steps 9–14,
  28–29);
* concurrency and variant-limit safety (steps 23–25);
* the frozen default dynamic policy document (step 32).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import canonical_json, sha256_text
from hqsb.compiler.records import GuardEventRecord

# ── taxonomy (E11-05 §4) ───────────────────────────────────────────────────

GUARD_CATEGORIES: Tuple[str, ...] = (
    "semantic_input",
    "python_control",
    "shape_range",
    "relational",
    "layout",
    "dtype_device",
    "target",
    "performance",
    "state",
)

DEFAULT_ACTION_BY_CATEGORY: Mapping[str, str] = {
    "semantic_input": "error_or_semantic_fallback",
    "python_control": "new_graph_or_break",
    "shape_range": "another_bucket_or_recompile_or_fallback",
    "relational": "error_or_fallback",
    "layout": "copy_or_another_variant_or_fallback",
    "dtype_device": "compatible_candidate_or_fallback",
    "target": "not_registered_or_another_candidate",
    "performance": "generic_variant_no_semantic_error",
    "state": "runtime_state_path_never_stale_binary",
}

#: Guard kinds that may be *dropped* only with a correctness argument; kept
#: explicit so a "minimality" review cannot silently remove a semantic guard.
SEMANTIC_GUARD_CATEGORIES: Tuple[str, ...] = ("semantic_input", "relational", "state")
PERFORMANCE_GUARD_CATEGORIES: Tuple[str, ...] = ("performance",)


@dataclass
class GuardSpec:
    """One guard: expression, category, source and the action on failure."""

    guard_id: str
    category: str
    expression: str
    source: str
    evaluator: Optional[Callable[[Mapping[str, Any]], bool]] = None
    semantic_required: bool = True
    origin: str = "user"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.category not in GUARD_CATEGORIES:
            problems.append(f"unknown guard category {self.category!r}")
        if not self.expression:
            problems.append(f"{self.guard_id}: guard needs an expression")
        if not self.source:
            problems.append(f"{self.guard_id}: guard needs a source (module/line/analysis)")
        if self.category in PERFORMANCE_GUARD_CATEGORIES and self.semantic_required:
            problems.append(
                f"{self.guard_id}: a performance guard may not be declared semantically required"
            )
        if self.category in SEMANTIC_GUARD_CATEGORIES and not self.semantic_required:
            problems.append(
                f"{self.guard_id}: a {self.category} guard may not be dropped as 'performance only'"
            )
        return problems

    def evaluate(self, inputs: Mapping[str, Any]) -> bool:
        if self.evaluator is not None:
            return bool(self.evaluator(inputs))
        raise ConfigError(
            f"guard {self.guard_id!r} has no evaluator: a guard that cannot be evaluated "
            "must not silently return True"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "guard_id": self.guard_id,
            "category": self.category,
            "expression": self.expression,
            "source": self.source,
            "semantic_required": self.semantic_required,
            "origin": self.origin,
            "failure_action": DEFAULT_ACTION_BY_CATEGORY[self.category],
        }


def range_guard(
    *, guard_id: str, symbol: str, lower: int, upper: int, source: str, category: str = "shape_range"
) -> GuardSpec:
    """``lower <= symbol <= upper`` as an evaluable guard."""

    def _eval(inputs: Mapping[str, Any]) -> bool:
        value = inputs.get(symbol)
        return isinstance(value, int) and lower <= value <= upper

    return GuardSpec(
        guard_id=guard_id,
        category=category,
        expression=f"{lower} <= {symbol} <= {upper}",
        source=source,
        evaluator=_eval,
        semantic_required=category in SEMANTIC_GUARD_CATEGORIES,
    )


def equality_guard(
    *, guard_id: str, name: str, expected: Any, source: str, category: str = "dtype_device"
) -> GuardSpec:
    def _eval(inputs: Mapping[str, Any]) -> bool:
        return inputs.get(name) == expected

    return GuardSpec(
        guard_id=guard_id,
        category=category,
        expression=f"{name} == {expected!r}",
        source=source,
        evaluator=_eval,
        semantic_required=category in SEMANTIC_GUARD_CATEGORIES,
    )


def divisibility_guard(
    *, guard_id: str, symbol: str, divisor: int, source: str
) -> GuardSpec:
    def _eval(inputs: Mapping[str, Any]) -> bool:
        value = inputs.get(symbol)
        return isinstance(value, int) and value % divisor == 0

    return GuardSpec(
        guard_id=guard_id,
        category="shape_range",
        expression=f"{symbol} % {divisor} == 0",
        source=source,
        evaluator=_eval,
    )


def alignment_guard(*, guard_id: str, symbol: str, alignment: int, source: str) -> GuardSpec:
    def _eval(inputs: Mapping[str, Any]) -> bool:
        value = inputs.get(symbol)
        return isinstance(value, int) and value % alignment == 0

    return GuardSpec(
        guard_id=guard_id,
        category="layout",
        expression=f"{symbol} % {alignment} == 0",
        source=source,
        evaluator=_eval,
        semantic_required=False,  # alignment is a capability condition, not semantics
    )


# ── variants and lookup (E11-05 §3.3 / steps 5–6, 26) ──────────────────────


@dataclass
class Variant:
    """``V_i = (semantic_identity, target, code/binary, guard_domain, priority)``."""

    variant_id: str
    semantic_identity: str
    compile_identity: str
    target_id: str
    artifact_id: str
    guards: Tuple[GuardSpec, ...]
    priority: int = 0
    created_reason: str = ""
    parent_compile_id: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("variant_id", "semantic_identity", "compile_identity", "target_id"):
            if not getattr(self, name):
                problems.append(f"variant missing {name!r}")
        if not self.guards:
            problems.append(
                f"{self.variant_id}: a variant with an empty guard domain would accept any input"
            )
        if not self.created_reason:
            problems.append(f"{self.variant_id}: new variants must record why they were created")
        for guard in self.guards:
            problems.extend(guard.validate())
        return problems

    def evaluate(self, inputs: Mapping[str, Any]) -> "GuardEvaluation":
        results = []
        for guard in self.guards:
            value = guard.evaluate(inputs)
            results.append(
                {
                    "guard_id": guard.guard_id,
                    "category": guard.category,
                    "expression": guard.expression,
                    "outcome": value,
                    "semantic_required": guard.semantic_required,
                }
            )
        return GuardEvaluation(
            variant_id=self.variant_id,
            results=tuple(results),
            all_true=all(row["outcome"] for row in results),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "semantic_identity": self.semantic_identity,
            "compile_identity": self.compile_identity,
            "target_id": self.target_id,
            "artifact_id": self.artifact_id,
            "guards": [guard.as_dict() for guard in self.guards],
            "priority": self.priority,
            "created_reason": self.created_reason,
            "parent_compile_id": self.parent_compile_id,
        }


@dataclass
class GuardEvaluation:
    variant_id: str
    results: Tuple[Mapping[str, Any], ...]
    all_true: bool

    @property
    def failed(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(row for row in self.results if not row["outcome"])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "all_true": self.all_true,
            "failed_guards": [dict(row) for row in self.failed],
            "results": [dict(row) for row in self.results],
        }


@dataclass
class LookupResult:
    """Outcome of a variant lookup: ``hit`` requires *all* guards true."""

    outcome: str  # hit | miss | fallback
    variant_id: str = ""
    reason: str = ""
    failed_guards: Tuple[str, ...] = ()
    considered: Tuple[str, ...] = ()
    artifact_compatible: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "variant_id": self.variant_id,
            "reason": self.reason,
            "failed_guards": list(self.failed_guards),
            "considered": list(self.considered),
            "artifact_compatible": self.artifact_compatible,
        }


class VariantRegistry:
    """Ordered variant lookup with strict guard/artifact semantics."""

    def __init__(self, *, fallback_available: bool = True) -> None:
        self._variants: Dict[str, Variant] = {}
        self.fallback_available = fallback_available

    def add(self, variant: Variant) -> None:
        problems = variant.validate()
        if problems:
            raise ConfigError(f"invalid variant: {'; '.join(problems)}")
        if variant.variant_id in self._variants:
            raise ConfigError(f"duplicate variant id {variant.variant_id!r}")
        self._variants[variant.variant_id] = variant

    def variants(self) -> List[Variant]:
        return sorted(self._variants.values(), key=lambda item: (-item.priority, item.variant_id))

    def lookup(
        self,
        *,
        semantic_identity: str,
        inputs: Mapping[str, Any],
        artifact_compatible: Callable[[str], bool] = lambda artifact_id: True,
        accept_guard_false: bool = False,
    ) -> LookupResult:
        """Find a variant only if every required guard is true.

        ``accept_guard_false`` exists solely for negative testing: production
        paths must never set it (guard false + reuse is an automatic FAIL).
        """
        considered: List[str] = []
        failed: List[str] = []
        for variant in self.variants():
            if variant.semantic_identity != semantic_identity:
                continue
            considered.append(variant.variant_id)
            evaluation = variant.evaluate(inputs)
            if not evaluation.all_true and not accept_guard_false:
                failed.extend(row["guard_id"] for row in evaluation.failed)
                continue
            if not artifact_compatible(variant.artifact_id):
                failed.append(f"{variant.variant_id}:artifact_incompatible")
                continue
            if not evaluation.all_true:
                failed.extend(row["guard_id"] for row in evaluation.failed)
            return LookupResult(
                outcome="hit",
                variant_id=variant.variant_id,
                reason="all guards true and artifact compatible",
                considered=tuple(considered),
                artifact_compatible=True,
            )
        if considered:
            return LookupResult(
                outcome="miss" if self.fallback_available else "fallback",
                reason=(
                    "no variant with all guards true; guard failures recorded (a matching graph "
                    "hash is not a cache hit)"
                ),
                failed_guards=tuple(sorted(set(failed))),
                considered=tuple(considered),
            )
        return LookupResult(
            outcome="miss",
            reason="no variant with this semantic identity",
            considered=(),
        )


# ── domain analysis (step 7) ───────────────────────────────────────────────


def domain_coverage(
    variants: Sequence[Variant],
    domain_samples: Sequence[Mapping[str, Any]],
    *,
    fallback_available: bool = True,
) -> Dict[str, Any]:
    """Enumerate a finite domain: coverage, holes, overlaps and priorities."""
    covered = 0
    holes: List[int] = []
    overlapping: List[Dict[str, Any]] = []
    for index, sample in enumerate(domain_samples):
        matches = [variant for variant in variants if variant.evaluate(sample).all_true]
        if matches:
            covered += 1
        else:
            holes.append(index)
        if len(matches) > 1:
            priorities = [variant.priority for variant in matches]
            overlapping.append(
                {
                    "sample_index": index,
                    "variants": [variant.variant_id for variant in matches],
                    "priorities": priorities,
                    "deterministic": len(set(priorities)) == len(priorities),
                }
            )
    unresolved = [row for row in overlapping if not row["deterministic"]]
    return {
        "samples": len(domain_samples),
        "covered": covered,
        "coverage": round(covered / max(1, len(domain_samples)), 4),
        "holes": holes,
        "holes_have_fallback": fallback_available,
        "overlaps": overlapping,
        "overlaps_without_priority_order": unresolved,
        "ok": not unresolved and (not holes or fallback_available),
        "rule": (
            "overlaps require a deterministic priority; holes require a fallback or an "
            "explicit input error"
        ),
    }


# ── compile ledger (steps 22, 27) ──────────────────────────────────────────


@dataclass
class CompileEvent:
    """One compilation, mapped to the guard/path/domain that triggered it."""

    compile_id: str
    variant_id: str
    trigger: str  # guard_failure | new_shape_bucket | python_path | state_epoch | explicit
    triggering_guard_ids: Tuple[str, ...] = ()
    domain: str = ""
    compile_time_s: float = 0.0
    code_size_bytes: int = 0
    cache_entries: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "compile_id": self.compile_id,
            "variant_id": self.variant_id,
            "trigger": self.trigger,
            "triggering_guard_ids": list(self.triggering_guard_ids),
            "domain": self.domain,
            "compile_time_s": self.compile_time_s,
            "code_size_bytes": self.code_size_bytes,
            "cache_entries": self.cache_entries,
        }


COMPILE_TRIGGERS: Tuple[str, ...] = (
    "guard_failure",
    "new_shape_bucket",
    "python_path",
    "state_epoch",
    "explicit",
    "unexplained",
)


def compile_explainability(events: Sequence[CompileEvent], *, calls: int) -> Dict[str, Any]:
    """``unexplained_compile_rate`` must be 0 (E11-05 step 27)."""
    unexplained = [
        event.compile_id
        for event in events
        if event.trigger == "unexplained"
        or (event.trigger == "guard_failure" and not event.triggering_guard_ids)
        or not event.domain
    ]
    total_cost = sum(event.compile_time_s for event in events)
    return {
        "calls": calls,
        "compiles": len(events),
        "recompile_rate": round(len(events) / max(1, calls), 6),
        "unexplained": unexplained,
        "unexplained_compile_rate": round(len(unexplained) / max(1, len(events)), 6),
        "total_compile_cost_s": round(total_cost, 6),
        "compile_amortized_latency_s": round(total_cost / max(1, calls), 6),
        "ok": not unexplained,
        "rule": "every new compile must map to a guard/path/domain; otherwise the cause is unknown",
    }


def wrong_reuse_audit(
    dispatch_rows: Sequence[Mapping[str, Any]],
    guard_events: Sequence[GuardEventRecord],
) -> Dict[str, Any]:
    """Join dispatch rows with guard evaluations: wrong reuse must be zero."""
    events_by_variant: Dict[str, List[GuardEventRecord]] = {}
    for event in guard_events:
        events_by_variant.setdefault(event.variant_id, []).append(event)
    violations: List[Dict[str, Any]] = []
    for row in dispatch_rows:
        variant = str(row.get("variant_id", ""))
        for event in events_by_variant.get(variant, ()):
            if event.outcome is False and event.action == "reuse":
                violations.append(
                    {
                        "dispatch": dict(row),
                        "guard_id": event.guard_id,
                        "expression": event.expression,
                    }
                )
    return {
        "dispatches": len(dispatch_rows),
        "guard_events": len(guard_events),
        "wrong_reuse_count": len(violations),
        "violations": violations,
        "ok": not violations,
        "rule": "any guard false→same incompatible binary execution is an immediate FAIL",
    }


# ── strategies and total cost (steps 9–14, 28–29) ──────────────────────────

STRATEGIES: Tuple[str, ...] = (
    "S0_static",
    "S1_automatic",
    "S2_declared_bounded",
    "S3_broad_dynamic",
    "S4_manual_buckets",
)

MIN_STRATEGIES_FOR_COMPARISON = 3


@dataclass
class StrategyObservation:
    """One strategy's measured/simulated row (compile and runtime separated)."""

    strategy: str
    distinct_variants: int
    compile_events: int
    compile_cost_s: float
    first_hit_latency_ms: Optional[float]
    steady_latency_ms: float
    guard_failure_rate: float
    fallback_rate: float
    code_cache_bytes: int
    correctness_ok: bool = False

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.strategy not in STRATEGIES:
            problems.append(f"unknown strategy {self.strategy!r}")
        if not self.correctness_ok:
            problems.append(
                f"{self.strategy}: a strategy row without correctness is not comparable"
            )
        if self.steady_latency_ms <= 0:
            problems.append(f"{self.strategy}: steady latency must be positive")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "distinct_variants": self.distinct_variants,
            "compile_events": self.compile_events,
            "compile_cost_s": self.compile_cost_s,
            "first_hit_latency_ms": self.first_hit_latency_ms,
            "steady_latency_ms": self.steady_latency_ms,
            "guard_failure_rate": self.guard_failure_rate,
            "fallback_rate": self.fallback_rate,
            "code_cache_bytes": self.code_cache_bytes,
            "correctness_ok": self.correctness_ok,
        }


def compare_strategies(
    rows: Sequence[StrategyObservation], *, requests: int, horizon_requests: int = 0
) -> Dict[str, Any]:
    """Total cost over a deployment horizon (compile + runtime), not steady only."""
    if requests <= 0:
        raise ConfigError("requests must be positive to compute total cost")
    horizon = horizon_requests or requests
    table: List[Dict[str, Any]] = []
    problems: List[str] = []
    for row in rows:
        problems.extend(row.validate())
        total_runtime_s = row.steady_latency_ms / 1000.0 * horizon
        total = row.compile_cost_s + total_runtime_s
        table.append(
            {
                **row.as_dict(),
                "total_cost_s": round(total, 6),
                "compile_share": round(row.compile_cost_s / total, 4) if total > 0 else None,
                "amortized_latency_ms": round(total / horizon * 1000.0, 6),
            }
        )
    table.sort(key=lambda item: item["total_cost_s"])
    return {
        "requests": requests,
        "horizon_requests": horizon,
        "rows": table,
        "ranking_by_total_cost": [row["strategy"] for row in table],
        "problems": problems,
        "fair_comparison": len({row.strategy for row in rows}) >= MIN_STRATEGIES_FOR_COMPARISON,
        "rule": (
            "static variants may win long-run if traffic is concentrated and caches persist; "
            "generic dynamic may lose specialisation — decision needs the total cost"
        ),
    }


# ── concurrency / budget (steps 23–25) ─────────────────────────────────────


@dataclass
class ConcurrencyPlan:
    """Concurrent first-trigger plan: no duplicate storm, no half-baked entry."""

    threads: int
    same_variant: bool
    expected_new_compiles: int
    expect_single_publish: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "threads": self.threads,
            "same_variant": self.same_variant,
            "expected_new_compiles": self.expected_new_compiles,
            "expect_single_publish": self.expect_single_publish,
        }


def evaluate_concurrency(
    *,
    plan: ConcurrencyPlan,
    compiles: int,
    published_entries: int,
    half_baked_reads: int,
    correctness_ok: bool,
) -> Dict[str, Any]:
    storm = plan.same_variant and compiles > plan.expected_new_compiles
    return {
        "plan": plan.as_dict(),
        "compiles": compiles,
        "published_entries": published_entries,
        "half_baked_reads": half_baked_reads,
        "duplicate_storm": storm,
        "correctness_ok": correctness_ok,
        "ok": not storm and half_baked_reads == 0 and correctness_ok,
        "rule": (
            "readers may only observe committed entries; duplicate compiles are allowed but "
            "must be recorded as cost, half-baked reads are never allowed"
        ),
    }


@dataclass
class VariantBudget:
    """Compiler variant limit (framework-dependent) with fail-closed behaviour."""

    max_variants: int
    behaviour_on_exceed: str = "fallback"

    def __post_init__(self) -> None:
        if self.max_variants <= 0:
            raise ConfigError("max_variants must be positive")
        if self.behaviour_on_exceed not in ("fallback", "error"):
            raise ConfigError(
                "variant limit behaviour must be 'fallback' or 'error'; silent eager is not "
                "an allowed state"
            )

    def exceed_action(self, variants: int) -> Dict[str, Any]:
        exceeded = variants > self.max_variants
        return {
            "variants": variants,
            "limit": self.max_variants,
            "exceeded": exceeded,
            "action": self.behaviour_on_exceed if exceeded else "continue",
            "must_report_compiled_false": exceeded,
            "rule": (
                "after the limit the system falls back or errors; reporting the run as "
                "'compiled' would hide the fallback"
            ),
        }


# ── default dynamic policy (step 32) ───────────────────────────────────────


def dynamic_policy_document(
    *,
    default_strategy: str,
    supported_domain: Mapping[str, Any],
    variant_budget: VariantBudget,
    compile_budget_s: float,
    known_overspecialization: Sequence[str] = (),
    monitoring_thresholds: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Frozen, machine-readable default policy + monitoring thresholds."""
    if default_strategy not in STRATEGIES:
        raise ConfigError(f"unknown default strategy {default_strategy!r}")
    thresholds = dict(monitoring_thresholds or {})
    return {
        "default_strategy": default_strategy,
        "supported_domain": dict(sorted(supported_domain.items())),
        "variant_budget": variant_budget.max_variants,
        "variant_limit_behaviour": variant_budget.behaviour_on_exceed,
        "compile_budget_s": compile_budget_s,
        "known_overspecialization": list(known_overspecialization),
        "monitoring_thresholds": thresholds,
        "fallback": "reference lowering, then original subgraph, then explicit error",
        "rule": (
            "a performance guard must never be reported as a semantic input error; unknown "
            "shapes fall back rather than reuse a stale variant"
        ),
    }


def policy_digest(document: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(document))


def guard_minimality_review(
    guards: Sequence[GuardSpec],
    *,
    removed: Sequence[str] = (),
    replacement_domains: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Review proposed guard removal.

    A guard may only be dropped when (a) it is not semantic and (b) the caller
    declares the replacement variant domain that now covers those inputs.
    Dropping a guard to cut recompiles without a replacement is exactly the
    "wrong binary reuse" failure mode.
    """
    replacements = dict(replacement_domains or {})
    by_id = {guard.guard_id: guard for guard in guards}
    blocked: List[Dict[str, str]] = []
    for guard_id in removed:
        guard = by_id.get(guard_id)
        if guard is None:
            blocked.append({"guard_id": guard_id, "reason": "unknown guard"})
            continue
        if guard.semantic_required or guard.category in SEMANTIC_GUARD_CATEGORIES:
            blocked.append(
                {
                    "guard_id": guard_id,
                    "reason": (
                        f"category {guard.category} is semantic; removal needs a proof, "
                        "not a benchmark"
                    ),
                }
            )
            continue
        if guard_id not in replacements:
            blocked.append(
                {
                    "guard_id": guard_id,
                    "reason": (
                        "no replacement variant domain declared: removing the guard would "
                        "let the old binary handle inputs it never proved legal"
                    ),
                }
            )
    return {
        "removed": list(removed),
        "replacement_domains": replacements,
        "blocked": blocked,
        "ok": not blocked,
        "allowed_removals": [
            guard_id for guard_id in removed if guard_id not in {row["guard_id"] for row in blocked}
        ],
    }


def variant_count_is_explainable(events: Iterable[CompileEvent]) -> Dict[str, Any]:
    reasons: Dict[str, int] = {}
    for event in events:
        reasons[event.trigger] = reasons.get(event.trigger, 0) + 1
    return {
        "by_trigger": dict(sorted(reasons.items())),
        "explainable": all(key != "unexplained" for key in reasons),
        "note": "the minimum variant count is not a goal; every count must have a cause",
    }
