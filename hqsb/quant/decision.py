"""E05-10: candidate registry, five gates, Pareto and recommendations.

The final S05 experiment is a *decision* experiment. Its structure is enforced
here:

1. :class:`CandidateRegistry` — every candidate has a unique identity bound to
   model/tokenizer revision, method/config hash, scheme, coverage, artifact
   hashes, runtime, hardware fingerprint and result-set hashes; a missing
   required field marks the candidate ``INCOMPLETE`` (never filled with a
   default, never scored as zero);
2. :func:`evaluate_gates` — the five gates in order (correctness → artifact →
   quality → execution → measurement); a failed gate removes the candidate
   from the deployment front but keeps it in the waterflow table;
3. :func:`point_pareto` / :func:`uncertainty_pareto` — per
   hardware × workload fronts with point estimates and bootstrap resampling;
4. :func:`apply_scenarios` — the A–E deployment scenarios; if no candidate
   satisfies the constraints the answer is "no feasible solution", not a
   relaxed quality gate;
5. :func:`offline_amortization` — amortized offline cost with sensitivities;
6. :func:`recommendation_matrix` — recommended / conditional / research-only /
   rejected / incomplete lists with hardware, SLO, quality boundary, fallback
   and limitations;
7. :func:`release_bundle_manifest` / :func:`regression_thresholds` — what S06
   consumes.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.quant import stats as qstats

#: Candidate status values.
COMPLETE = "complete"
INCOMPLETE = "incomplete"

#: Recommendation classes (E05-10 §16 step 19).
RECOMMENDED = "recommended"
CONDITIONAL = "conditional"
RESEARCH_ONLY = "research_only"
REJECTED_CLAIM = "rejected"
INCOMPLETE_CLASS = "incomplete"

#: Objective dimensions (all minimised; quality is a gate, not a score).
OBJECTIVES = (
    "device_peak_bytes",
    "ttft_ms",
    "tpot_ms",
    "joules_per_token",
    "offline_cost_s",
)

REQUIRED_IDENTITY_FIELDS = (
    "candidate_id",
    "model_revision",
    "method",
    "config_hash",
    "bits",
    "scheme_hash",
    "module_coverage",
    "calibration_hash",
    "canonical_artifact_hash",
    "packed_variant_hash",
    "runtime",
    "kernel_provider",
    "hardware_fingerprint",
    "quality_result_hash",
    "performance_result_hash",
    "fallback_policy",
)


@dataclass
class Candidate:
    """One deployment candidate with its evidence links."""

    identity: Dict[str, Any]
    gates: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    energy: Dict[str, Any] = field(default_factory=dict)
    lifecycle: Dict[str, Any] = field(default_factory=dict)
    coverage: Dict[str, Any] = field(default_factory=dict)
    evidence_links: List[Dict[str, str]] = field(default_factory=list)
    recommendation: str = ""
    reason: str = ""

    @property
    def candidate_id(self) -> str:
        return str(self.identity.get("candidate_id", ""))

    def completeness(self) -> Dict[str, Any]:
        """Audit required identity/metric fields; missing → INCOMPLETE."""
        missing_identity = [
            name
            for name in REQUIRED_IDENTITY_FIELDS
            if self.identity.get(name) in (None, "", [])
        ]
        required_metrics = {
            "device_peak_bytes": self.memory.get("device_peak_bytes"),
            "ttft_ms": self.metrics.get("ttft_ms"),
            "tpot_ms": self.metrics.get("tpot_ms"),
        }
        missing_metrics = [
            name for name, value in required_metrics.items() if value is None
        ]
        status = COMPLETE if not missing_identity and not missing_metrics else INCOMPLETE
        return {
            "status": status,
            "missing_identity_fields": missing_identity,
            "missing_metrics": missing_metrics,
            "evidence_links": len(self.evidence_links),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "identity": dict(self.identity),
            "completeness": self.completeness(),
            "gates": dict(self.gates),
            "metrics": dict(self.metrics),
            "quality": dict(self.quality),
            "execution": dict(self.execution),
            "memory": dict(self.memory),
            "energy": dict(self.energy),
            "lifecycle": dict(self.lifecycle),
            "coverage": dict(self.coverage),
            "evidence_links": list(self.evidence_links),
            "recommendation": self.recommendation,
            "reason": self.reason,
        }


@dataclass
class CandidateRegistry:
    """Candidate registry with uniqueness and evidence-link enforcement."""

    decision_spec_hash: str = ""
    candidates: Dict[str, Candidate] = field(default_factory=dict)
    hardware: str = ""
    workload: str = ""

    def add(self, candidate: Candidate) -> None:
        if not candidate.candidate_id:
            raise ConfigError("candidate_id must not be empty")
        if candidate.candidate_id in self.candidates:
            raise ConfigError(
                f"duplicate candidate_id {candidate.candidate_id!r}; candidates "
                f"must be unique per hardware/workload/policy"
            )
        self.candidates[candidate.candidate_id] = candidate

    def from_records(self, records: Sequence[Mapping[str, Any]]) -> "CandidateRegistry":
        """Build from raw dicts (the shape a driver collects)."""
        for record in records:
            self.add(
                Candidate(
                    identity=dict(record.get("identity", {})),
                    gates=dict(record.get("gates", {})),
                    metrics=dict(record.get("metrics", {})),
                    quality=dict(record.get("quality", {})),
                    execution=dict(record.get("execution", {})),
                    memory=dict(record.get("memory", {})),
                    energy=dict(record.get("energy", {})),
                    lifecycle=dict(record.get("lifecycle", {})),
                    coverage=dict(record.get("coverage", {})),
                    evidence_links=list(record.get("evidence_links", [])),
                )
            )
        return self

    def completeness_audit(self) -> List[Dict[str, Any]]:
        rows = []
        for candidate in self.candidates.values():
            audit = candidate.completeness()
            rows.append({"candidate_id": candidate.candidate_id, **audit})
        return rows

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision_spec_hash": self.decision_spec_hash,
            "hardware": self.hardware,
            "workload": self.workload,
            "candidates": [candidate.as_dict() for candidate in self.candidates.values()],
        }


def evaluate_gates(registry: CandidateRegistry) -> Dict[str, Any]:
    """Evaluate the five gates for every candidate, in order.

    Gate inputs are the *candidate's own records*; a gate without evidence is
    INCONCLUSIVE. A failed correctness/artifact/quality gate removes the
    candidate from the deployment front; a failed execution gate keeps it as
    ``quality_only``; a failed measurement gate makes it ``research_only``.
    """
    rows: List[Dict[str, Any]] = []
    for candidate in registry.candidates.values():
        completeness = candidate.completeness()
        quality = candidate.quality.get("gate", {})
        execution = candidate.execution
        measurement = candidate.metrics.get("measurement_gate", {})
        correctness = candidate.gates.get("correctness", {})
        artifact = candidate.gates.get("artifact", {})
        verdicts = {
            "correctness": _verdict(correctness),
            "artifact": _verdict(artifact),
            "quality": _verdict(quality),
            "execution": _execution_verdict(execution),
            "measurement": _verdict(measurement),
        }
        failed = [name for name, value in verdicts.items() if value == qstats.GATE_FAIL]
        inconclusive = [
            name for name, value in verdicts.items() if value == qstats.GATE_INCONCLUSIVE
        ]
        if completeness["status"] == INCOMPLETE:
            classification = INCOMPLETE_CLASS
        elif any(name in failed for name in ("correctness", "artifact", "quality")):
            classification = REJECTED_CLAIM
        elif "execution" in failed:
            classification = RESEARCH_ONLY  # algorithm-quality only
        elif "measurement" in failed:
            classification = RESEARCH_ONLY
        elif inconclusive:
            classification = INCOMPLETE_CLASS
        else:
            classification = RECOMMENDED
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "completeness": completeness["status"],
                "verdicts": verdicts,
                "failed_gates": failed,
                "inconclusive_gates": inconclusive,
                "classification": classification,
            }
        )
    by_class: Dict[str, int] = {}
    for row in rows:
        by_class[row["classification"]] = by_class.get(row["classification"], 0) + 1
    return {
        "order": ["correctness", "artifact", "quality", "execution", "measurement"],
        "rows": rows,
        "by_classification": dict(sorted(by_class.items())),
        "note": (
            "a failed gate removes the candidate from the deployment front but "
            "keeps it in the waterflow table; INCONCLUSIVE is not a pass"
        ),
    }


def _verdict(report: Mapping[str, Any]) -> str:
    if not report:
        return qstats.GATE_INCONCLUSIVE
    if report.get("verdict"):
        return str(report["verdict"])
    if report.get("passed") is True:
        return qstats.GATE_PASS
    if report.get("passed") is False:
        return qstats.GATE_FAIL
    return qstats.GATE_INCONCLUSIVE


def _execution_verdict(execution: Mapping[str, Any]) -> str:
    """The execution gate needs low-bit evidence, not just a passing API call."""
    if not execution:
        return qstats.GATE_INCONCLUSIVE
    if execution.get("fallback_reason"):
        return qstats.GATE_FAIL
    if execution.get("low_bit_executed") is True and execution.get("observed_kernel"):
        return qstats.GATE_PASS
    if execution.get("observed_kernel") and execution.get("claimed_bits") in (None, 16):
        # FP16 or storage-only evidence cannot support a low-bit claim.
        return qstats.GATE_FAIL
    return qstats.GATE_FAIL


def _deployment_eligible(registry: CandidateRegistry) -> List[Candidate]:
    """Candidates whose five gates passed (deployment front input)."""
    rows = {row["candidate_id"]: row for row in evaluate_gates(registry)["rows"]}
    eligible = []
    for candidate in registry.candidates.values():
        row = rows[candidate.candidate_id]
        if row["failed_gates"] or row["inconclusive_gates"]:
            continue
        eligible.append(candidate)
    return eligible


# ── Pareto (E05-10 §8/§9) ─────────────────────────────────────────────────


def dominates(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    objectives: Sequence[str] = OBJECTIVES,
) -> bool:
    """True when ``left`` is no worse in all objectives and strictly better in one."""
    at_least_one_better = False
    for name in objectives:
        left_value = left.get(name)
        right_value = right.get(name)
        if left_value is None or right_value is None:
            raise ConfigError(
                f"objective {name!r} missing for a candidate; a missing objective "
                f"cannot be treated as zero or as best"
            )
        if left_value > right_value:
            return False
        if left_value < right_value:
            at_least_one_better = True
    return at_least_one_better


def point_pareto(
    registry: CandidateRegistry,
    *,
    objectives: Sequence[str] = OBJECTIVES,
    hardware: Optional[str] = None,
    workload: Optional[str] = None,
) -> Dict[str, Any]:
    """Point-estimate Pareto front per hardware × workload (no averaging)."""
    groups: Dict[Tuple[str, str], List[Candidate]] = {}
    for candidate in _deployment_eligible(registry):
        identity = candidate.identity
        key = (
            str(identity.get("hardware_fingerprint", "unknown")),
            str(identity.get("workload", "unknown")),
        )
        if hardware and key[0] != hardware:
            continue
        if workload and key[1] != workload:
            continue
        groups.setdefault(key, []).append(candidate)
    fronts: List[Dict[str, Any]] = []
    for (hardware_key, workload_key), candidates in sorted(groups.items()):
        vectors = {
            candidate.candidate_id: {
                name: _value(candidate, name) for name in objectives
            }
            for candidate in candidates
        }
        front = []
        dominated = []
        for candidate_id, vector in vectors.items():
            is_dominated = any(
                other_id != candidate_id and dominates(other, vector, objectives=objectives)
                for other_id, other in vectors.items()
            )
            (dominated if is_dominated else front).append(candidate_id)
        fronts.append(
            {
                "hardware": hardware_key,
                "workload": workload_key,
                "candidates": sorted(vectors),
                "pareto_front": sorted(front),
                "dominated": sorted(dominated),
                "objective_vectors": vectors,
                "objectives": list(objectives),
            }
        )
    return {"fronts": fronts, "objectives": list(objectives)}


def _value(candidate: Candidate, objective: str) -> Optional[float]:
    if objective in candidate.metrics and candidate.metrics[objective] is not None:
        return float(candidate.metrics[objective])
    if objective in candidate.memory and candidate.memory[objective] is not None:
        return float(candidate.memory[objective])
    if objective in candidate.energy and candidate.energy[objective] is not None:
        return float(candidate.energy[objective])
    if objective in candidate.lifecycle and candidate.lifecycle[objective] is not None:
        return float(candidate.lifecycle[objective])
    return None


@dataclass
class UncertaintyFront:
    """Result of resampling candidates into the front (E05-10 §9)."""

    resamples: int
    inclusion_probability: Dict[str, float]
    robustly_dominated: List[str]
    likely_dominated: List[str]
    non_dominated: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "resamples": self.resamples,
            "inclusion_probability": dict(sorted(self.inclusion_probability.items())),
            "robustly_dominated": sorted(self.robustly_dominated),
            "likely_dominated": sorted(self.likely_dominated),
            "non_dominated": sorted(self.non_dominated),
        }


def uncertainty_pareto(
    registry: CandidateRegistry,
    *,
    objectives: Sequence[str] = OBJECTIVES,
    metric_ci: Optional[Mapping[str, Mapping[str, Tuple[float, float]]]] = None,
    resamples: int = 500,
    seed: int = 0,
) -> UncertaintyFront:
    """Bootstrap the Pareto front over the reported confidence intervals.

    ``metric_ci[candidate_id][objective] = (low, high)``. Each resample draws a
    value from a symmetric interval around the point estimate and recomputes
    the front. The resulting inclusion probability is a *stability
    description*, not a new score (E05-10 §9).
    """
    eligible = _deployment_eligible(registry)
    if not eligible:
        return UncertaintyFront(resamples=0, inclusion_probability={}, robustly_dominated=[], likely_dominated=[], non_dominated=[])
    metric_ci = dict(metric_ci or {})
    rng = random.Random(seed)
    inclusion = {candidate.candidate_id: 0 for candidate in eligible}
    dominated_count = {candidate.candidate_id: 0 for candidate in eligible}
    for _ in range(resamples):
        vectors: Dict[str, Dict[str, float]] = {}
        for candidate in eligible:
            vector: Dict[str, float] = {}
            for objective in objectives:
                point = _value(candidate, objective)
                if point is None:
                    raise ConfigError(
                        f"candidate {candidate.candidate_id!r} lacks objective "
                        f"{objective!r} for the uncertainty front"
                    )
                low, high = metric_ci.get(candidate.candidate_id, {}).get(
                    objective, (point, point)
                )
                half = max(0.0, (high - low) / 2.0)
                vector[objective] = max(0.0, point + rng.uniform(-half, half))
            vectors[candidate.candidate_id] = vector
        for candidate_id, vector in vectors.items():
            if any(
                other_id != candidate_id
                and dominates(other, vector, objectives=objectives)
                for other_id, other in vectors.items()
            ):
                dominated_count[candidate_id] += 1
            else:
                inclusion[candidate_id] += 1
    inclusion_probability = {
        candidate_id: count / resamples for candidate_id, count in inclusion.items()
    }
    robustly_dominated = [
        candidate_id
        for candidate_id, count in dominated_count.items()
        if count == resamples
    ]
    likely_dominated = [
        candidate_id
        for candidate_id, count in dominated_count.items()
        if 0 < count < resamples
    ]
    non_dominated = [
        candidate_id for candidate_id, probability in inclusion_probability.items() if probability > 0
    ]
    return UncertaintyFront(
        resamples=resamples,
        inclusion_probability=inclusion_probability,
        robustly_dominated=robustly_dominated,
        likely_dominated=likely_dominated,
        non_dominated=non_dominated,
    )


# ── scenarios, amortization and recommendations (E05-10 §11/§16) ─────────


@dataclass
class Scenario:
    """One deployment scenario definition (E05-10 §11)."""

    name: str
    constraints: Dict[str, float]
    minimize: Tuple[str, ...] = ("tpot_ms",)
    description: str = ""


SCENARIO_MEMORY_CONSTRAINED = Scenario(
    name="memory_constrained",
    constraints={"device_peak_bytes_max": 0.0},
    minimize=("tpot_ms", "joules_per_token"),
    description="model+runtime+KV must fit the device budget with a safety margin",
)
SCENARIO_INTERACTIVE = Scenario(
    name="interactive_low_latency",
    constraints={"ttft_ms_max": 0.0, "tpot_p95_ms_max": 0.0},
    minimize=("joules_per_token", "device_peak_bytes"),
    description="TTFT and p95 TPOT SLO must hold before energy is compared",
)
SCENARIO_BATCH_PREFILL = Scenario(
    name="batch_prefill",
    constraints={"prefill_tokens_per_s_min": 0.0},
    minimize=("prefill_tokens_per_s",),
    description="large-M prefill throughput; W4 fused dequant is not assumed to win",
)
SCENARIO_LONG_CONTEXT = Scenario(
    name="long_context",
    constraints={"max_context_tokens_min": 0.0},
    minimize=("tpot_ms", "device_peak_bytes"),
    description="requires the E05-08 KV gate",
)
SCENARIO_PORTABILITY = Scenario(
    name="portability",
    constraints={"compatibility_rank_max": 1.0},
    minimize=("compatibility_rank", "offline_cost_s"),
    description="prefers direct/repack over requantize and penalises fallbacks",
)


def apply_scenarios(
    registry: CandidateRegistry,
    scenarios: Sequence[Scenario],
    *,
    objective_values: Optional[Mapping[str, Mapping[str, float]]] = None,
    quality_verdicts: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Filter candidates by scenario constraints; report "no feasible solution".

    Quality is always a precondition: a candidate that failed the quality gate
    is never considered, whatever the scenario.
    """
    objective_values = dict(objective_values or {})
    quality_verdicts = dict(quality_verdicts or {})
    # Validate constraint syntax up front, before touching any candidate: a
    # malformed SLO must be refused regardless of the candidate set.
    for scenario in scenarios:
        for constraint in scenario.constraints:
            if not (constraint.endswith("_min") or constraint.endswith("_max")):
                raise ConfigError(
                    f"scenario {scenario.name!r} constraint {constraint!r} must "
                    f"end in _min or _max"
                )
    results: List[Dict[str, Any]] = []
    for scenario in scenarios:
        feasible: List[str] = []
        rejected: List[Dict[str, Any]] = []
        for candidate in registry.candidates.values():
            candidate_id = candidate.candidate_id
            quality = quality_verdicts.get(candidate_id)
            if quality is None:
                quality = _verdict(candidate.quality.get("gate", {}))
            if quality != qstats.GATE_PASS:
                rejected.append(
                    {"candidate_id": candidate_id, "reason": f"quality gate {quality}"}
                )
                continue
            values = {
                **objective_values.get(candidate_id, {}),
                **{
                    key: value
                    for key, value in {
                        "device_peak_bytes": candidate.memory.get("device_peak_bytes"),
                        "ttft_ms": candidate.metrics.get("ttft_ms"),
                        "tpot_p95_ms": candidate.metrics.get("tpot_p95_ms"),
                        "prefill_tokens_per_s": candidate.metrics.get("prefill_tokens_per_s"),
                        "max_context_tokens": candidate.memory.get("max_context_tokens"),
                        "compatibility_rank": candidate.execution.get("compatibility_rank"),
                        "offline_cost_s": candidate.lifecycle.get("offline_cost_s"),
                    }.items()
                    if value is not None
                },
            }
            violated = []
            for constraint, bound in scenario.constraints.items():
                if constraint.endswith("_max"):
                    key = constraint[: -len("_max")]
                    if key in values and values[key] > bound:
                        violated.append(
                            {"constraint": constraint, "value": values[key], "bound": bound}
                        )
                elif constraint.endswith("_min"):
                    key = constraint[: -len("_min")]
                    if key in values and values[key] < bound:
                        violated.append(
                            {"constraint": constraint, "value": values[key], "bound": bound}
                        )
                else:
                    raise ConfigError(
                        f"constraint {constraint!r} must end in _min or _max"
                    )
            if violated:
                rejected.append(
                    {"candidate_id": candidate_id, "reason": "constraint violated", "violations": violated}
                )
                continue
            feasible.append(candidate_id)
        ranking: List[Dict[str, Any]] = []
        if feasible:
            sort_keys = [name for name in scenario.minimize]
            decorated = []
            for candidate_id in feasible:
                values = objective_values.get(candidate_id, {})
                key_values = []
                complete = True
                for key in sort_keys:
                    value = values.get(key)
                    if value is None:
                        complete = False
                        break
                    key_values.append(value)
                decorated.append((complete, key_values, candidate_id))
            decorated.sort(key=lambda item: (not item[0], item[1] if item[0] else [], item[2]))
            ranking = [
                {"candidate_id": candidate_id, "complete_ranking_data": complete}
                for complete, _values, candidate_id in decorated
            ]
        results.append(
            {
                "scenario": scenario.name,
                "description": scenario.description,
                "constraints": dict(scenario.constraints),
                "feasible": sorted(feasible),
                "ranking": ranking,
                "rejected": rejected,
                "outcome": (
                    "feasible_set_found" if feasible else "no_feasible_candidate"
                ),
                "note": (
                    "no feasible candidate means exactly that; the quality gate is "
                    "never relaxed to manufacture a winner"
                ),
            }
        )
    return {"scenarios": results}


def offline_amortization(
    offline_cost_s: float,
    *,
    request_volumes: Sequence[int],
    unamortized_kept: bool = True,
) -> Dict[str, Any]:
    """Amortize the offline cost over several request volumes (E05-10 §7.4)."""
    if offline_cost_s < 0:
        raise ConfigError("offline cost must be non-negative")
    rows = []
    for volume in sorted(set(int(value) for value in request_volumes)):
        if volume <= 0:
            raise ConfigError(f"request volume must be positive, got {volume}")
        rows.append(
            {
                "expected_requests": volume,
                "amortized_cost_s_per_request": offline_cost_s / volume,
            }
        )
    return {
        "offline_cost_s": offline_cost_s,
        "unamortized_kept": unamortized_kept,
        "sensitivity": rows,
        "note": (
            "the unamortized value is always reported; a single optimistic "
            "request count must not be the only number"
        ),
    }


class RecommendationClass:
    """Recommendation classes (strings, exported for table-driven code)."""

    RECOMMENDED = RECOMMENDED
    CONDITIONAL = CONDITIONAL
    RESEARCH_ONLY = RESEARCH_ONLY
    REJECTED = REJECTED_CLAIM
    INCOMPLETE = INCOMPLETE_CLASS


def recommendation_matrix(
    registry: CandidateRegistry,
    *,
    pareto: Optional[Mapping[str, Any]] = None,
    scenarios: Optional[Mapping[str, Any]] = None,
    quality_verdicts: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Build the deployment decision table (E05-10 §19).

    Each row carries the candidate, its recommendation class, the hard
    constraints it was selected under, the evidence links, and the
    conditions/fallbacks. Rejected and incomplete candidates are listed
    explicitly (never dropped).
    """
    quality_verdicts = dict(quality_verdicts or {})
    fronts = {
        (front["hardware"], front["workload"]): {candidate_id: front for candidate_id in front["pareto_front"]}
        for front in (pareto or {}).get("fronts", [])
    }
    rows: List[Dict[str, Any]] = []
    for candidate in registry.candidates.values():
        completeness = candidate.completeness()
        gate_verdicts = evaluate_gates(registry)
        gate_row = next(
            row
            for row in gate_verdicts["rows"]
            if row["candidate_id"] == candidate.candidate_id
        )
        if completeness["status"] == INCOMPLETE:
            classification = INCOMPLETE_CLASS
            reason = f"missing evidence: {completeness['missing_identity_fields']} {completeness['missing_metrics']}"
        elif gate_row["failed_gates"]:
            classification = REJECTED_CLAIM
            reason = f"failed gates: {gate_row['failed_gates']}"
        elif gate_row["inconclusive_gates"]:
            classification = INCOMPLETE_CLASS
            reason = f"inconclusive gates: {gate_row['inconclusive_gates']}"
        else:
            quality = quality_verdicts.get(
                candidate.candidate_id,
                _verdict(candidate.quality.get("gate", {})),
            )
            on_front = any(
                candidate.candidate_id in front
                for front in fronts.values()
            )
            if quality == qstats.GATE_PASS and on_front:
                classification = RECOMMENDED
                reason = "passed all gates and lies on a Pareto front"
            elif quality == qstats.GATE_PASS:
                classification = CONDITIONAL
                reason = "passed gates but dominated on every measured front"
            else:
                classification = RESEARCH_ONLY
                reason = "algorithm-quality evidence only"
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "recommendation": classification,
                "reason": reason,
                "hardware": candidate.identity.get("hardware_fingerprint"),
                "workload": candidate.identity.get("workload"),
                "quality_gate": candidate.quality.get("gate", {}),
                "fallback_policy": candidate.identity.get("fallback_policy"),
                "evidence_links": list(candidate.evidence_links),
                "limitations": list(candidate.identity.get("known_limitations", [])),
                "metrics": {
                    "device_peak_bytes": candidate.memory.get("device_peak_bytes"),
                    "ttft_ms": candidate.metrics.get("ttft_ms"),
                    "tpot_ms": candidate.metrics.get("tpot_ms"),
                    "joules_per_token": candidate.energy.get("joules_per_token"),
                    "offline_cost_s": candidate.lifecycle.get("offline_cost_s"),
                },
            }
        )
    # Scenario feasibility is reported next to the candidate table.
    scenario_rows = (scenarios or {}).get("scenarios", [])
    by_class: Dict[str, int] = {}
    for row in rows:
        by_class[row["recommendation"]] = by_class.get(row["recommendation"], 0) + 1
    return {
        "rows": rows,
        "by_class": dict(sorted(by_class.items())),
        "scenarios": scenario_rows,
        "note": (
            "recommendations always carry hardware, workload/SLO, quality "
            "boundary and fallback; a missing feasible candidate is reported "
            "as such"
        ),
    }


# ── release bundle and regression thresholds (E05-10 §16 step 22/23) ─────


def regression_thresholds(
    *,
    quality_metrics: Mapping[str, Mapping[str, float]],
    memory_metrics: Mapping[str, float],
    performance_metrics: Mapping[str, float],
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Freeze the numbers later stages must not regress (S06/S07)."""
    return {
        "quality": {
            metric: {key: float(value) for key, value in metrics.items()}
            for metric, metrics in sorted(quality_metrics.items())
        },
        "memory_bytes": {key: float(value) for key, value in sorted(memory_metrics.items())},
        "performance_ms": {
            key: float(value) for key, value in sorted(performance_metrics.items())
        },
        "tolerance": float(tolerance),
        "note": (
            "thresholds are the frozen S05 baselines; a later optimisation that "
            "breaks one must return to the corresponding S05 gate rather than "
            "redefine the threshold"
        ),
    }


def release_bundle_manifest(
    *,
    run_id: str,
    artifacts: Sequence[Mapping[str, Any]],
    policies: Sequence[Mapping[str, Any]],
    compatibility_matrix_hash: str,
    result_registry_hash: str,
    figures: Sequence[str] = (),
    known_limitations: Sequence[str] = (),
    s06_interface: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the S05 release bundle manifest (E05-10 §16 step 22).

    The bundle lists *hashes and URIs*, never raw weights; the caller supplies
    the artifact records (canonical/packed hashes, schemas, layouts).
    """
    payload = {
        "schema_version": "1.0.0",
        "kind": "hqsb.s05.release_bundle",
        "run_id": run_id,
        "artifacts": [dict(artifact) for artifact in artifacts],
        "policies": [dict(policy) for policy in policies],
        "compatibility_matrix_hash": compatibility_matrix_hash,
        "result_registry_hash": result_registry_hash,
        "figures": list(figures),
        "known_limitations": list(known_limitations),
        "s06_interface": dict(s06_interface or {}),
    }
    payload["bundle_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def figure_points_from_table(
    table_rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
    group_key: str = "candidate_id",
) -> Dict[str, Any]:
    """Produce plot data strictly from the unified table (E05-10 §14).

    Each point keeps its `candidate_id` and source row index, so a figure can
    be traced back to raw samples. Missing values are skipped and counted.
    """
    points: List[Dict[str, Any]] = []
    skipped = 0
    for index, row in enumerate(table_rows):
        x_value = row.get(x_key)
        y_value = row.get(y_key)
        if x_value is None or y_value is None:
            skipped += 1
            continue
        points.append(
            {
                "x": float(x_value),
                "y": float(y_value),
                "group": row.get(group_key),
                "row_index": index,
                "workload_id": row.get("workload_id"),
                "run_id": row.get("run_id"),
            }
        )
    return {
        "x_key": x_key,
        "y_key": y_key,
        "points": points,
        "skipped_rows": skipped,
        "note": "figure points must keep row_index so the value can be re-derived",
    }


__all__ = [
    "COMPLETE",
    "CONDITIONAL",
    "Candidate",
    "CandidateRegistry",
    "INCOMPLETE",
    "INCOMPLETE_CLASS",
    "OBJECTIVES",
    "RECOMMENDED",
    "RECOMMENDATION_CLASSES",
    "REJECTED_CLAIM",
    "REQUIRED_IDENTITY_FIELDS",
    "RESEARCH_ONLY",
    "RecommendationClass",
    "SCENARIO_BATCH_PREFILL",
    "SCENARIO_INTERACTIVE",
    "SCENARIO_LONG_CONTEXT",
    "SCENARIO_MEMORY_CONSTRAINED",
    "SCENARIO_PORTABILITY",
    "Scenario",
    "UncertaintyFront",
    "apply_scenarios",
    "dominates",
    "evaluate_gates",
    "figure_points_from_table",
    "offline_amortization",
    "point_pareto",
    "recommendation_matrix",
    "regression_thresholds",
    "release_bundle_manifest",
    "uncertainty_pareto",
]

#: Alias kept for table-driven registration code.
RECOMMENDATION_CLASSES = (
    RECOMMENDED,
    CONDITIONAL,
    RESEARCH_ONLY,
    REJECTED_CLAIM,
    INCOMPLETE_CLASS,
)
