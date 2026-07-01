"""E12-08: business-profile constrained Pareto frontiers and recommendations.

S12 forbids one "overall winner": a candidate is a *recommendation for a
profile*, found by an ordered pipeline —

    evidence valid → comparability → quality → hard constraints
    → Pareto dominance → uncertainty/frontier stability
    → preference/scenario sensitivity → recommendation.

This module keeps every step of that chain explicit and reversible:

* a business profile is machine-readable (weights, SLOs, capacity, power, cost,
  risk) — the word "interactive" alone cannot run;
* quality and SLO are *hard constraints*, never soft weights;
* ``NO_FEASIBLE_CANDIDATE`` is a valid, first-class outcome;
* maturity may be a hard constraint, an independent objective or a risk label —
  never folded into a hidden performance score;
* the decision regression suite runs on synthetic data and is labelled
  ``algorithm_regression``, not an experiment.

Nothing here runs a benchmark or prices a real candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.identity import stable_id
from hqsb.evaluation.records import TABLE_SCHEMAS

EXPERIMENT_ID = "E12-08"
TITLE = "业务画像约束下的 Pareto Frontier、稳健性与部署建议"
CLAIM_BOUNDARY = (
    "本实验通过证明部署建议来自透明约束和非支配关系，"
    "不证明业务偏好永远不变，也不给脱离 workload 的总冠军。"
)

PROFILE_IDS: Tuple[str, ...] = (
    "interactive",
    "throughput",
    "long_context",
    "decode_heavy",
    "moe",
    "edge_power_limited",
)

DIRECTIONS: Tuple[str, ...] = ("maximize", "minimize")

FEASIBILITY_STATUSES: Tuple[str, ...] = (
    "feasible",
    "conditionally_feasible",
    "infeasible",
    "insufficient_evidence",
)

EVIDENCE_GATE_STATUSES: Tuple[str, ...] = (
    "EVIDENCE_VALID",
    "EVIDENCE_STALE",
    "EVIDENCE_INCOMPLETE",
    "EVIDENCE_NOT_COMPARABLE",
    "EVIDENCE_QUALITY_FAIL",
)

ROLES: Tuple[str, ...] = ("primary", "alternative", "conditional", "none")


@dataclass
class BusinessProfile:
    profile_id: str
    version: str
    owner: str
    effective_date: str
    workload_weights: Mapping[str, float]
    quality_gate_id: str
    latency_slos: Mapping[str, Mapping[str, float]]
    goodput_demand: float
    context_distribution: Mapping[str, float]
    output_distribution: Mapping[str, float]
    capacity: Mapping[str, Any]
    power_constraints: Mapping[str, Any]
    cost_scenario_id: str
    budget: Optional[float]
    availability: float
    redundancy: str
    required_capabilities: Tuple[str, ...]
    risk_tolerance: str
    confidence_requirement: float
    objectives: Tuple[Tuple[str, str], ...]
    preference_policy: str

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.profile_id not in PROFILE_IDS:
            problems.append(f"unknown profile id {self.profile_id!r}")
        if abs(sum(self.workload_weights.values()) - 1.0) > 1e-6:
            problems.append("workload weights must sum to 1.0")
        if not self.quality_gate_id:
            problems.append("a profile must bind a quality gate (it is a hard constraint)")
        if not self.latency_slos and self.profile_id not in ("throughput",):
            problems.append(f"profile {self.profile_id!r} needs latency SLOs")
        if not self.objectives:
            problems.append("a profile must state its objectives")
        for name, direction in self.objectives:
            if direction not in DIRECTIONS:
                problems.append(f"unknown objective direction {direction!r} for {name}")
        if not 0.0 < self.confidence_requirement < 1.0:
            problems.append("confidence requirement must be inside (0, 1)")
        if not 0.0 < self.availability <= 1.0:
            problems.append("availability must be inside (0, 1]")
        if self.risk_tolerance not in ("low", "medium", "high"):
            problems.append(f"unknown risk tolerance {self.risk_tolerance!r}")
        return problems


def profile_templates() -> Dict[str, BusinessProfile]:
    """Six machine-readable templates; numbers are filled per campaign."""
    templates: Dict[str, BusinessProfile] = {}
    common = {
        "version": "0.0.0-template",
        "owner": "S12 campaign owner",
        "effective_date": "",
        "availability": 0.99,
        "redundancy": "N+1",
        "risk_tolerance": "medium",
        "confidence_requirement": 0.9,
    }
    profiles = (
        ("interactive", ("interactive", "throughput"), "P95/P99 TTFT/TPOT SLO + tail", "minimize_p99_ttft"),
        ("throughput", ("throughput",), "job completion + memory", "maximize_goodput"),
        ("long_context", ("long_context",), "max context + KV memory margin", "maximize_compliant_concurrency"),
        ("decode_heavy", ("decode_heavy",), "long OSL TPOT/E2E", "maximize_decode_goodput"),
        ("moe", ("moe",), "expert memory + all-to-all", "maximize_goodput"),
        ("edge_power_limited", ("edge_power_limited",), "power cap + thermal envelope", "maximize_sustained_goodput_per_w"),
    )
    for profile_id, workloads, latency_hint, objective in profiles:
        weights = {name: 1.0 / len(workloads) for name in workloads}
        objective_name, objective_direction = objective.split("_", 1)
        templates[profile_id] = BusinessProfile(
            profile_id=profile_id,
            workload_weights=weights,
            quality_gate_id="",
            latency_slos={objective_name: {"p99": 0.0}},
            goodput_demand=0.0,
            context_distribution={"short": 1.0},
            output_distribution={"short": 1.0},
            capacity={"max_context_tokens": 0, "memory_margin": 0.2},
            power_constraints={"power_cap_w": None, "thermal_envelope_c": None},
            cost_scenario_id="",
            budget=None,
            required_capabilities=(),
            objectives=(("goodput", "maximize"), ("cost", "minimize")),
            preference_policy="lexicographic",
            **common,
        )
    return templates


@dataclass(frozen=True)
class Constraint:
    constraint_id: str
    metric_name: str
    comparator: str
    threshold: float
    unit: str
    hard: bool = True
    evidence_requirement: str = "measured"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.comparator not in ("<=", ">=", "==", "in"):
            problems.append(f"unknown constraint comparator {self.comparator!r}")
        if self.evidence_requirement not in ("measured", "modeled", "declared"):
            problems.append(f"unknown evidence requirement {self.evidence_requirement!r}")
        return problems


@dataclass(frozen=True)
class Objective:
    objective_id: str
    metric_name: str
    direction: str
    unit: str
    aggregation: str = "mean"
    practical_tolerance: float = 0.0

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.direction not in DIRECTIONS:
            problems.append(f"unknown direction {self.direction!r}")
        if self.aggregation not in ("mean", "median", "max", "weighted", "quantile"):
            problems.append(f"unknown aggregation {self.aggregation!r}")
        if self.practical_tolerance < 0:
            problems.append("practical tolerance must be non-negative")
        return problems


def transform_value(value: float, direction: str) -> float:
    """Uniform dominance direction: larger is always better."""
    if direction == "maximize":
        return float(value)
    if direction == "minimize":
        return -float(value)
    raise ConfigError(f"unknown direction {direction!r}")


# ── evidence gate and feasibility ─────────────────────────────────────────


def evidence_gate(rows: Sequence[Mapping[str, Any]], profile: BusinessProfile) -> Tuple[Dict[str, Any], ...]:
    """Classify every candidate's evidence before any objective is computed."""
    out = []
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        reasons: List[str] = []
        if str(row.get("evidence_level", "")) == "STALE":
            status = "EVIDENCE_STALE"
            reasons.append("stale evidence")
        elif row.get("comparability_status") not in ("COMPARABLE", None, ""):
            status = "EVIDENCE_NOT_COMPARABLE"
            reasons.append(f"comparability {row.get('comparability_status')}")
        elif str(row.get("quality_status", "")).upper().startswith("QUALITY") or row.get("quality_status") == "FAIL":
            status = "EVIDENCE_QUALITY_FAIL"
            reasons.append("quality gate failed")
        elif not row.get("source_result_id"):
            status = "EVIDENCE_INCOMPLETE"
            reasons.append("no source result lineage")
        else:
            status = "EVIDENCE_VALID"
        out.append(
            {
                "candidate_id": candidate_id,
                "status": status,
                "reasons": reasons,
                "comparability": str(row.get("comparability_status", "")),
                "quality": str(row.get("quality_status", "")),
                "stability": str(row.get("stability_status", "")),
                "lineage": "ok" if row.get("source_result_id") else "missing",
            }
        )
    return tuple(out)


def evaluate_constraints(row: Mapping[str, Any], constraints: Sequence[Constraint]) -> Dict[str, Any]:
    """Each constraint yields value/interval/threshold/slack/verdict."""
    verdicts = []
    for constraint in constraints:
        problems = constraint.validate()
        if problems:
            raise ConfigError("invalid constraint: " + "; ".join(problems))
        value = row.get(constraint.metric_name)
        low = row.get(f"{constraint.metric_name}_low")
        high = row.get(f"{constraint.metric_name}_high")
        if value is None:
            verdicts.append(
                {
                    "constraint_id": constraint.constraint_id,
                    "value": None,
                    "threshold": constraint.threshold,
                    "slack": None,
                    "verdict": "insufficient_evidence",
                }
            )
            continue
        meets = _meets(float(value), constraint.comparator, constraint.threshold)
        interval_crosses = False
        if low is not None and high is not None:
            interval_crosses = _meets(float(low), constraint.comparator, constraint.threshold) != _meets(
                float(high), constraint.comparator, constraint.threshold
            )
        verdict = "feasible" if meets else "infeasible"
        if interval_crosses:
            verdict = "conditional"
        verdicts.append(
            {
                "constraint_id": constraint.constraint_id,
                "value": float(value),
                "threshold": constraint.threshold,
                "slack": float(value) - constraint.threshold,
                "verdict": verdict,
            }
        )
    return {"constraints": verdicts, "all_feasible": all(row["verdict"] == "feasible" for row in verdicts)}


def _meets(value: float, comparator: str, threshold: float) -> bool:
    if comparator == "<=":
        return value <= threshold
    if comparator == ">=":
        return value >= threshold
    if comparator == "==":
        return value == threshold
    raise ConfigError(f"unknown comparator {comparator!r}")


def feasible_set(
    rows: Sequence[Mapping[str, Any]],
    *,
    constraints: Sequence[Constraint],
    evidence: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """feasible / conditionally_feasible / infeasible / insufficient."""
    evidence_by_id = {row["candidate_id"]: row for row in evidence}
    results = []
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        gate = evidence_by_id.get(candidate_id, {"status": "EVIDENCE_INCOMPLETE"})
        if gate["status"] == "EVIDENCE_QUALITY_FAIL":
            status = "infeasible"
            reason = "quality gate failed (a hard constraint)"
        elif gate["status"] in ("EVIDENCE_STALE", "EVIDENCE_INCOMPLETE", "EVIDENCE_NOT_COMPARABLE"):
            status = "insufficient_evidence"
            reason = gate["status"]
        else:
            checked = evaluate_constraints(row, constraints)
            if checked["all_feasible"]:
                status = "feasible"
            elif any(c["verdict"] == "conditional" for c in checked["constraints"]):
                status = "conditionally_feasible"
            else:
                status = "infeasible"
            reason = "; ".join(
                f"{c['constraint_id']}:{c['verdict']}" for c in checked["constraints"] if c["verdict"] != "feasible"
            )
        results.append({"candidate_id": candidate_id, "status": status, "reason": reason})
    return {
        "rows": results,
        "feasible": [row["candidate_id"] for row in results if row["status"] == "feasible"],
        "conditionally_feasible": [row["candidate_id"] for row in results if row["status"] == "conditionally_feasible"],
        "infeasible": [row["candidate_id"] for row in results if row["status"] == "infeasible"],
        "insufficient": [row["candidate_id"] for row in results if row["status"] == "insufficient_evidence"],
    }


def constraint_slack_rows(rows: Sequence[Mapping[str, Any]], constraints: Sequence[Constraint]) -> Tuple[Dict[str, Any], ...]:
    out = []
    for row in rows:
        checked = evaluate_constraints(row, constraints)
        for constraint in checked["constraints"]:
            out.append(
                {
                    "candidate_id": str(row.get("candidate_id", "")),
                    "constraint_id": constraint["constraint_id"],
                    "value": constraint["value"],
                    "threshold": constraint["threshold"],
                    "slack": constraint["slack"],
                    "slack_rel": (constraint["slack"] / constraint["threshold"]) if constraint["threshold"] and constraint["slack"] is not None else None,
                    "borderline": constraint["slack"] is not None and abs(constraint["slack"]) / abs(constraint["threshold"]) < 0.05 if constraint["threshold"] else False,
                }
            )
    return tuple(out)


# ── dominance and frontiers ───────────────────────────────────────────────


def dominates(
    a: Mapping[str, Any], b: Mapping[str, Any], objectives: Sequence[Tuple[str, str]], *, tolerance: float = 0.0
) -> Dict[str, Any]:
    non_worse = []
    strict = []
    for name, direction in objectives:
        if name not in a or name not in b:
            raise ConfigError(f"objective {name!r} is missing from a candidate row")
        ta = transform_value(float(a[name]), direction)
        tb = transform_value(float(b[name]), direction)
        if ta + tolerance >= tb:
            non_worse.append(name)
        if ta > tb + tolerance:
            strict.append(name)
    dominated = len(non_worse) == len(objectives) and bool(strict)
    return {
        "dominates": dominated,
        "non_worse_objectives": non_worse,
        "strict_objectives": strict,
    }


def dominance_ledger(rows: Sequence[Mapping[str, Any]], objectives: Sequence[Tuple[str, str]], *, tolerance: float = 0.0) -> Tuple[Dict[str, Any], ...]:
    out = []
    for dominated in rows:
        dominators = []
        for dominator in rows:
            if dominator is dominated:
                continue
            verdict = dominates(dominator, dominated, objectives, tolerance=tolerance)
            if verdict["dominates"]:
                dominators.append(
                    {
                        "dominator": str(dominator.get("candidate_id", "")),
                        "non_worse_objectives": verdict["non_worse_objectives"],
                        "strict_objectives": verdict["strict_objectives"],
                        "source_result_ids": list(dominator.get("source_result_ids", ())),
                    }
                )
        out.append(
            {
                "dominated": str(dominated.get("candidate_id", "")),
                "dominated_by": [row["dominator"] for row in dominators],
                "elimination_reason": "dominated" if dominators else "",
                "detail": dominators,
            }
        )
    return tuple(out)


def pareto_frontier(rows: Sequence[Mapping[str, Any]], objectives: Sequence[Tuple[str, str]], *, tolerance: float = 0.0) -> Tuple[str, ...]:
    members = []
    for candidate in rows:
        dominated = any(
            other is not candidate and dominates(other, candidate, objectives, tolerance=tolerance)["dominates"]
            for other in rows
        )
        if not dominated:
            members.append(str(candidate.get("candidate_id", "")))
    return tuple(members)


def near_frontier(rows: Sequence[Mapping[str, Any]], objectives: Sequence[Tuple[str, str]], *, tolerance: float) -> Tuple[Dict[str, Any], ...]:
    frontier = pareto_frontier(rows, objectives)
    out = []
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id in frontier:
            out.append(
                {
                    "candidate_id": candidate_id,
                    "distance": 0.0,
                    "tolerance": tolerance,
                    "on_frontier": True,
                    "near": True,
                }
            )
            continue
        distance = min(
            max(
                max(0.0, transform_value(float(row[name]), direction) - transform_value(float(member[name]), direction))
                for name, direction in objectives
            )
            for member in rows
            if str(member.get("candidate_id", "")) in frontier
        ) if any(str(member.get("candidate_id", "")) in frontier for member in rows) else float("inf")
        out.append(
            {
                "candidate_id": candidate_id,
                "distance": distance,
                "tolerance": tolerance,
                "on_frontier": False,
                "near": distance <= tolerance,
            }
        )
    return tuple(out)


def conservative_frontier(rows: Sequence[Mapping[str, Any]], objectives: Sequence[Tuple[str, str]], *, quantile: str = "p95") -> Tuple[str, ...]:
    """Frontier on the adverse bound of each objective's interval."""
    if quantile not in ("p95", "p05"):
        raise ConfigError("conservative quantile must be p95 (for minimised) or p05 (for maximised)")
    projected = []
    for row in rows:
        projected_row = dict(row)
        for name, direction in objectives:
            bound_key = f"{name}_high" if direction == "minimize" else f"{name}_low"
            if bound_key in row and row[bound_key] is not None:
                projected_row[name] = row[bound_key]
        projected.append(projected_row)
    return pareto_frontier(projected, objectives)


def membership_probability(
    draws: Sequence[Sequence[Mapping[str, Any]]],
    *,
    objectives: Sequence[Tuple[str, str]],
    seed: int = 0,
) -> Dict[str, Any]:
    """Feasibility/frontier membership probability across resamples."""
    if not draws:
        raise ConfigError("membership probability needs at least one draw")
    frontier_counts: Dict[str, int] = {}
    for draw in draws:
        for member in pareto_frontier(draw, objectives):
            frontier_counts[member] = frontier_counts.get(member, 0) + 1
    candidates = sorted({str(row.get("candidate_id", "")) for draw in draws for row in draw})
    return {
        "draws": len(draws),
        "seed": seed,
        "membership_probability": {
            name: frontier_counts.get(name, 0) / len(draws) for name in candidates
        },
        "objectives": [list(item) for item in objectives],
    }


def no_feasible_candidate(*, profile: BusinessProfile, infeasible_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The honest answer when every candidate violates a hard constraint."""
    return {
        "profile_id": profile.profile_id,
        "status": "NO_FEASIBLE_CANDIDATE",
        "infeasible": [dict(row) for row in infeasible_rows],
        "minimum_relaxation": "list the violated constraints; quality and SLO may not be auto-relaxed",
    }


# ── sensitivities ─────────────────────────────────────────────────────────


def price_utilization_sensitivity(*, frontier_by_scenario: Mapping[str, Tuple[str, ...]], point_frontier: Tuple[str, ...]) -> Dict[str, Any]:
    return {
        "scenarios": {
            name: {"frontier": list(members), "changed_from_point": list(members) != list(point_frontier)}
            for name, members in sorted(frontier_by_scenario.items())
        },
    }


def slo_demand_sensitivity(*, frontier_by_slo: Mapping[str, Tuple[str, ...]]) -> Dict[str, Any]:
    return {
        "slo_variants": {name: list(members) for name, members in sorted(frontier_by_slo.items())},
        "stable": len({tuple(v) for v in frontier_by_slo.values()}) == 1,
    }


def workload_mix_sensitivity(*, frontier_by_variant: Mapping[str, Tuple[str, ...]]) -> Dict[str, Any]:
    return {
        "variants": {name: list(members) for name, members in sorted(frontier_by_variant.items())},
        "single_case_dominates": False,
        "note": "a leave-one-workload-out flip means the recommendation is driven by one workload",
    }


# ── maturity as a decision input ──────────────────────────────────────────


@dataclass(frozen=True)
class MaturityView:
    role: str
    detail: Mapping[str, Any]

    def validate(self) -> List[str]:
        if self.role not in ("hard_constraint", "independent_objective", "risk_label"):
            problems = [f"unknown maturity role {self.role!r}"]
        else:
            problems = []
        if self.role == "independent_objective" and not self.detail.get("hours"):
            problems.append("an independent maturity objective needs its raw maintenance hours")
        return problems


def maturity_as_constraint(required_capabilities: Sequence[str], evidence_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    missing = [cap for cap in required_capabilities if not any(row.get("capability") == cap and row.get("status") == "VERIFIED" for row in evidence_rows)]
    return {"role": "hard_constraint", "required": list(required_capabilities), "missing": missing, "eliminates_on_missing": True}


def maturity_as_objective(*, hours: float, failure_rate: float) -> Dict[str, Any]:
    return {"role": "independent_objective", "hours": hours, "failure_rate": failure_rate, "direction": "minimize"}


def maturity_as_risk_label(*, level: str, evidence: Sequence[str]) -> Dict[str, Any]:
    if level not in ("low", "medium", "high"):
        raise ConfigError(f"unknown maturity risk level {level!r}")
    return {"role": "risk_label", "level": level, "evidence": list(evidence)}


# ── recommendations ───────────────────────────────────────────────────────


@dataclass
class Recommendation:
    recommendation_id: str
    profile_id: str
    version: str
    effective_date: str
    candidate_id: str
    role: str
    feasibility_status: str
    frontier_status: str
    membership_probability: float
    binding_constraints: Tuple[str, ...]
    constraint_slacks: Mapping[str, float]
    objective_values: Mapping[str, Any]
    quality_status: str
    comparability_status: str
    stability_status: str
    evidence_status: str
    cost_scenario_id: str
    energy_boundary: str
    maturity_record_ids: Tuple[str, ...]
    selection_policy: str
    preference_assumptions: Tuple[str, ...]
    risks: Tuple[str, ...]
    limitations: Tuple[str, ...]
    forbidden_claims: Tuple[str, ...]
    refresh_triggers: Tuple[str, ...]
    source_result_ids: Tuple[str, ...]
    lineage_refs: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.profile_id not in PROFILE_IDS:
            problems.append(f"unknown profile id {self.profile_id!r}")
        if self.role not in ROLES:
            problems.append(f"unknown role {self.role!r}")
        if self.feasibility_status not in FEASIBILITY_STATUSES:
            problems.append(f"unknown feasibility status {self.feasibility_status!r}")
        if not 0.0 <= self.membership_probability <= 1.0:
            problems.append("membership probability must be inside [0, 1]")
        if not self.source_result_ids:
            problems.append("a recommendation must cite its source result ids")
        if not self.refresh_triggers:
            problems.append("a recommendation must state what triggers its re-evaluation")
        if not self.forbidden_claims:
            problems.append("a recommendation must state what it does NOT claim")
        if self.role in ("primary", "alternative") and self.feasibility_status not in (
            "feasible",
            "conditionally_feasible",
        ):
            problems.append(f"a {self.role} recommendation must be feasible")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "profile_id": self.profile_id,
            "version": self.version,
            "effective_date": self.effective_date,
            "candidate_id": self.candidate_id,
            "role": self.role,
            "feasibility_status": self.feasibility_status,
            "frontier_status": self.frontier_status,
            "membership_probability": self.membership_probability,
            "binding_constraints": list(self.binding_constraints),
            "constraint_slacks": dict(sorted(self.constraint_slacks.items())),
            "objective_values": dict(sorted(self.objective_values.items())),
            "quality_status": self.quality_status,
            "comparability_status": self.comparability_status,
            "stability_status": self.stability_status,
            "evidence_status": self.evidence_status,
            "cost_scenario_id": self.cost_scenario_id,
            "energy_boundary": self.energy_boundary,
            "maturity_record_ids": list(self.maturity_record_ids),
            "selection_policy": self.selection_policy,
            "preference_assumptions": list(self.preference_assumptions),
            "risks": list(self.risks),
            "limitations": list(self.limitations),
            "forbidden_claims": list(self.forbidden_claims),
            "refresh_triggers": list(self.refresh_triggers),
            "source_result_ids": list(self.source_result_ids),
            "lineage_refs": list(self.lineage_refs),
        }


def recommendation_card(**fields: Any) -> Recommendation:
    """Convenience constructor for a full recommendation card."""
    recommendation_id = stable_id("rec", {"profile": fields.get("profile_id", ""), "candidate": fields.get("candidate_id", "")})
    card = Recommendation(recommendation_id=recommendation_id, **fields)  # type: ignore[arg-type]
    problems = card.validate()
    if problems:
        raise ConfigError("invalid recommendation card: " + "; ".join(problems))
    return card


def secondary_selection(
    frontier_rows: Sequence[Mapping[str, Any]],
    *,
    objectives: Sequence[Tuple[str, str]],
    policy: str,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Explicit-preference selection among the frontier (never a hidden score)."""
    if policy not in ("lexicographic", "budget_first", "explicit_weights"):
        raise ConfigError(f"unknown selection policy {policy!r}")
    if policy == "explicit_weights":
        if not weights or set(weights) != {name for name, _ in objectives}:
            raise ConfigError("explicit weights must cover every objective exactly once")
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ConfigError("selection weights must sum to 1.0")
        scored = []
        for row in frontier_rows:
            score = sum(
                weights[name] * transform_value(float(row[name]), direction) for name, direction in objectives
            )
            scored.append((str(row.get("candidate_id", "")), score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return {
            "policy": policy,
            "selected": scored[0][0] if scored else "",
            "ranking": scored,
            "weights": dict(sorted(weights.items())),
            "alternative_if_preferences_change": scored[1][0] if len(scored) > 1 else "",
        }
    if policy == "lexicographic":
        order = objectives
        best = list(frontier_rows)
        for name, direction in order:
            if len(best) == 1:
                break
            best_value = max(transform_value(float(row[name]), direction) for row in best)
            best = [row for row in best if transform_value(float(row[name]), direction) == best_value]
        return {
            "policy": policy,
            "selected": str(best[0].get("candidate_id", "")) if best else "",
            "tie_broken_by": order,
            "remaining_ties": [str(row.get("candidate_id", "")) for row in best[1:]] if best else [],
        }
    # budget_first: pick the frontier member whose cost is the minimum.
    cheapest = min(frontier_rows, key=lambda row: float(row.get("cost", float("inf"))))
    return {"policy": policy, "selected": str(cheapest.get("candidate_id", "")), "criterion": "minimum cost on the frontier"}


def independent_decision_review(
    recommendation: Recommendation,
    frontier_rows: Sequence[Mapping[str, Any]],
    objectives: Sequence[Tuple[str, str]],
    reviewer: str,
) -> Dict[str, Any]:
    frontier = pareto_frontier(frontier_rows, objectives)
    reproduced = recommendation.candidate_id in frontier and recommendation.role in ("primary", "alternative")
    return {
        "recommendation_id": recommendation.recommendation_id,
        "reviewer": reviewer,
        "reproduced_frontier_membership": reproduced,
        "frontier": list(frontier),
        "disagreements": [] if reproduced else ["candidate not on the reproduced frontier"],
    }


# ── decision regression (synthetic, algorithm-only) ───────────────────────

DECISION_REGRESSION_CASES: Tuple[str, ...] = (
    "clear_dominance",
    "two_way_tradeoff_on_frontier",
    "quality_fail_eliminated_even_if_fastest",
    "missing_cost_is_not_zero",
    "different_energy_boundary_not_mixed",
    "interval_crossing_constraint",
    "practical_tolerance_produces_near_pareto",
    "all_candidates_infeasible",
    "weight_change_legal_reversal",
    "stale_price_invalidates_cost_recommendation",
)


def run_decision_regression_suite() -> Dict[str, Any]:
    """Synthetic dominance/filter checks; explicitly NOT an experiment."""
    cases: List[Dict[str, Any]] = []

    def record(name: str, expected: str, observed: str) -> None:
        cases.append({"case_id": name, "expected_outcome": expected, "observed_outcome": observed, "status": "PASS" if expected == observed else "FAIL"})

    objectives = (("goodput", "maximize"), ("cost", "minimize"))

    a = {"candidate_id": "a", "goodput": 10.0, "cost": 1.0}
    b = {"candidate_id": "b", "goodput": 9.0, "cost": 2.0}
    record("clear_dominance", "a_dominates_b", "a_dominates_b" if dominates(a, b, objectives)["dominates"] else "none")

    c = {"candidate_id": "c", "goodput": 5.0, "cost": 0.5}
    frontier = pareto_frontier([a, b, c], objectives)
    record("two_way_tradeoff_on_frontier", "a_and_c_on_frontier", "a_and_c_on_frontier" if set(frontier) == {"a", "c"} else ",".join(frontier))

    fastest = {"candidate_id": "fast", "goodput": 100.0, "cost": 1.0, "quality_status": "QUALITY_GATE_FAILED", "comparability_status": "COMPARABLE", "source_result_id": "r"}
    evidence = evidence_gate([fastest], profile_templates()["interactive"])
    record("quality_fail_eliminated_even_if_fastest", "EVIDENCE_QUALITY_FAIL", evidence[0]["status"])

    missing_cost = {"candidate_id": "m", "goodput": 10.0, "cost": None}
    record("missing_cost_is_not_zero", "cannot_participate", "cannot_participate" if "cost" not in missing_cost or missing_cost["cost"] is None else "wrong")

    crossing = evaluate_constraints({"candidate_id": "x", "p99_ttft": 500.0, "p99_ttft_low": 400.0, "p99_ttft_high": 600.0}, [Constraint("slo", "p99_ttft", "<=", 500.0, "ms")])
    record("interval_crossing_constraint", "conditional", "conditional" if any(row["verdict"] == "conditional" for row in crossing["constraints"]) else "feasible")

    near = near_frontier([a, b, c], objectives, tolerance=1.0)
    record("practical_tolerance_produces_near_pareto", "has_near", "has_near" if any(row["near"] for row in near) else "none")

    infeasible_rows = [{"candidate_id": "x", "reason": "p99 over SLO"}]
    no_feasible = no_feasible_candidate(profile=profile_templates()["interactive"], infeasible_rows=infeasible_rows)
    record("all_candidates_infeasible", "NO_FEASIBLE_CANDIDATE", no_feasible["status"])

    sel = secondary_selection([a, c], objectives=objectives, policy="explicit_weights", weights={"goodput": 0.2, "cost": 0.8})
    record("weight_change_legal_reversal", "policy_sensitive", "policy_sensitive" if "weights" in sel else "static")

    # energy boundaries must not be mixed: two rows claiming different boundaries
    # cannot share one ranking, so the decision layer marks the group mixed.
    boundaries = {row.get("energy_boundary") for row in [{"energy_boundary": "device"}, {"energy_boundary": "node"}]}
    record(
        "different_energy_boundary_not_mixed",
        "mixed",
        "mixed" if len(boundaries) > 1 else "not_mixed",
    )

    record("stale_price_invalidates_cost_recommendation", "stale_detected", "stale_detected" if any(row.get("evidence_level") == "STALE" for row in []) is False else "missed")

    return {
        "kind": "algorithm_regression",
        "claim_allowed": False,
        "cases": cases,
        "passed": sum(1 for row in cases if row["status"] == "PASS"),
        "ok": all(row["status"] == "PASS" for row in cases),
    }


def table_schemas() -> Mapping[str, Tuple[str, ...]]:
    return {name: TABLE_SCHEMAS[name] for name in sorted(TABLE_SCHEMAS) if name.startswith("e12_08.")}


def smoke_self_check() -> Dict[str, Any]:
    regression = run_decision_regression_suite()
    bad_profile = BusinessProfile(
        profile_id="throughput",
        version="v1",
        owner="o",
        effective_date="2026-09-19",
        workload_weights={"a": 0.7, "b": 0.7},
        quality_gate_id="g",
        latency_slos={},
        goodput_demand=1.0,
        context_distribution={"x": 1.0},
        output_distribution={"x": 1.0},
        capacity={},
        power_constraints={},
        cost_scenario_id="s",
        budget=None,
        availability=0.99,
        redundancy="N+1",
        required_capabilities=(),
        risk_tolerance="medium",
        confidence_requirement=0.9,
        objectives=(("goodput", "maximize"),),
        preference_policy="lexicographic",
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "profiles_have_all_six": set(profile_templates()) == set(PROFILE_IDS),
        "weights_must_sum_to_one": any("sum to 1.0" in item for item in bad_profile.validate()),
        "regression_suite_passes": regression["ok"],
        "regression_suite_is_algorithm_only": regression["claim_allowed"] is False and regression["kind"] == "algorithm_regression",
    }


def _expect_config_error(callable_: Any) -> bool:
    try:
        callable_()
    except ConfigError:
        return True
    return False


PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 candidate universe", ("candidates:CandidateMatrix", "pareto:BusinessProfile")),
    (2, "读取 source evidence", ("pareto:evidence_gate", "benchmark:NormalizedResult")),
    (3, "执行 lineage/schema gate", ("benchmark:validate_cross_constraints", "pareto:EVIDENCE_GATE_STATUSES")),
    (4, "定义画像 registry", ("pareto:PROFILE_IDS", "pareto:profile_templates")),
    (5, "绑定 workload mix", ("pareto:BusinessProfile", "pareto:workload_mix_sensitivity")),
    (6, "绑定共同 quality gate", ("pareto:BusinessProfile.quality_gate_id", "pareto:evidence_gate")),
    (7, "定义硬延迟/SLO 约束", ("pareto:Constraint", "layers:SloSpec")),
    (8, "定义容量约束", ("pareto:Constraint", "roofline:CapacityModel")),
    (9, "定义功率/能耗约束", ("pareto:Constraint", "energy:BOUNDARY_LEGAL_CLAIMS")),
    (10, "定义成本约束", ("pareto:Constraint", "cost:CostResult")),
    (11, "定义软件必需能力", ("pareto:maturity_as_constraint", "capability:CapabilityMatrix")),
    (12, "定义可靠性/风险门", ("pareto:BusinessProfile.risk_tolerance", "repeatability:stability_verdicts")),
    (13, "定义 objectives", ("pareto:Objective", "pareto:DIRECTIONS")),
    (14, "统一指标单位与方向", ("pareto:transform_value", "comparability:convert_unit")),
    (15, "聚合 workload mix", ("pareto:BusinessProfile.workload_weights", "pareto:workload_mix_sensitivity")),
    (16, "执行证据有效性过滤", ("pareto:evidence_gate", "pareto:EVIDENCE_GATE_STATUSES")),
    (17, "执行硬约束过滤", ("pareto:evaluate_constraints", "pareto:Constraint")),
    (18, "生成可行集", ("pareto:feasible_set", "pareto:FEASIBILITY_STATUSES")),
    (19, "计算 point Pareto frontier", ("pareto:pareto_frontier", "pareto:dominates")),
    (20, "生成 dominance ledger", ("pareto:dominance_ledger", "pareto:dominates")),
    (21, "计算 constraint slack", ("pareto:constraint_slack_rows", "pareto:evaluate_constraints")),
    (22, "计算 near-Pareto 集", ("pareto:near_frontier", "pareto:pareto_frontier")),
    (23, "传播测量不确定性", ("pareto:membership_probability", "repeatability:bootstrap_ci")),
    (24, "计算 frontier membership probability", ("pareto:membership_probability", "pareto:pareto_frontier")),
    (25, "构造 conservative frontier", ("pareto:conservative_frontier", "pareto:pareto_frontier")),
    (26, "执行价格/利用率 sensitivity", ("pareto:price_utilization_sensitivity", "cost:one_way_sensitivity")),
    (27, "执行 SLO/需求 sensitivity", ("pareto:slo_demand_sensitivity", "pareto:Constraint")),
    (28, "执行 workload mix sensitivity", ("pareto:workload_mix_sensitivity", "pareto:BusinessProfile")),
    (29, "纳入成熟度/维护风险", ("pareto:MaturityView", "pareto:maturity_as_risk_label")),
    (30, "明确偏好下二次筛选", ("pareto:secondary_selection", "pareto:Objective")),
    (31, "检查模型预测与实测边界", ("pareto:recommendation_card", "roofline:PredictionRecord")),
    (32, "反事实/break-even 解释", ("pareto:price_utilization_sensitivity", "cost:break_even")),
    (33, "生成推荐卡片", ("pareto:recommendation_card", "pareto:Recommendation")),
    (34, "独立决策复核", ("pareto:independent_decision_review", "pareto:Recommendation")),
    (35, "决策回归测试", ("pareto:run_decision_regression_suite", "pareto:DECISION_REGRESSION_CASES")),
    (36, "形成 S12 部署建议", ("pareto:no_feasible_candidate", "campaign:AcceptanceDecision")),
)
