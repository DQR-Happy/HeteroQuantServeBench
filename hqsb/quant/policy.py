"""Mixed-precision policy: machine-readable output of E05-05 (E05-05 §9/§10).

A policy is a document, not a screenshot of a table:

    units:
      - unit_id: layers.3.mlp.down_proj
        bits: 8
        group_size: 128
        method: rtn
        packed_layout: hqsb.w8a16.rowmajor.nk.v1
        keep_fp16: false
    quality_gate: {...}
    cost: {...}
    provenance: {...}

It is validated against the intervention units before it can be exported: a
policy that assigns different configurations to modules the backend executes
as one fused unit, or that gives tied weights different bits, is *invalid*
(E05-05 §4/§14), and the validator says exactly which constraint was broken.

Search drivers are provided as **auditable baselines** (all-FP16/W8/W4,
heuristic skip, direct-harm greedy, recovery-per-byte greedy) plus a generic
budgeted greedy search. Every evaluated candidate — accepted or rejected —
goes into the trace, because "only the winner survives" is a forbidden
pattern (E05-05 §14).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.quant.units import InterventionUnit, UnitConfig

POLICY_SCHEMA_VERSION = "1.0.0"
POLICY_KIND = "hqsb.quant.mixed_precision_policy"

KEEP_FP16 = 16


@dataclass
class UnitPolicyEntry:
    """Configuration for one unit in a policy."""

    unit_id: str
    bits: int
    group_size: Optional[int] = None
    method: str = "rtn"
    packed_layout: str = ""
    keep_fp16: bool = False
    scheme_hash: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "bits": self.bits,
            "group_size": self.group_size,
            "method": self.method,
            "packed_layout": self.packed_layout,
            "keep_fp16": self.keep_fp16,
            "scheme_hash": self.scheme_hash,
        }


@dataclass
class PolicyValidation:
    """Result of validating a policy against the unit/capability constraints."""

    valid: bool
    problems: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"valid": self.valid, "problems": list(self.problems)}


@dataclass
class MixedPrecisionPolicy:
    """A versioned mixed-precision policy document."""

    name: str
    entries: List[UnitPolicyEntry]
    quality_gate: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    known_limitations: List[str] = field(default_factory=list)
    schema_version: str = POLICY_SCHEMA_VERSION
    kind: str = POLICY_KIND

    def entry_map(self) -> Dict[str, UnitPolicyEntry]:
        return {entry.unit_id: entry for entry in self.entries}

    def validate(self, units: Sequence[InterventionUnit]) -> PolicyValidation:
        """Check unit coverage, fused/tied consistency and capability legality."""
        problems: List[Dict[str, Any]] = []
        unit_map = {unit.unit_id: unit for unit in units}
        unknown = [entry.unit_id for entry in self.entries if entry.unit_id not in unit_map]
        for unit_id in unknown:
            problems.append(
                {
                    "kind": "unknown_unit",
                    "unit_id": unit_id,
                    "detail": "policy references a unit that does not exist",
                }
            )
        missing = [unit_id for unit_id in unit_map if unit_id not in self.entry_map()]
        for unit_id in missing:
            problems.append(
                {
                    "kind": "missing_unit",
                    "unit_id": unit_id,
                    "detail": "policy does not decide this unit",
                }
            )
        for entry in self.entries:
            unit = unit_map.get(entry.unit_id)
            if unit is None:
                continue
            if unit.constraint != "independent" and unit.shared_with:
                peers = [peer for peer in unit.shared_with if peer in self.entry_map()]
                for peer in peers:
                    other = self.entry_map()[peer]
                    if (other.bits, other.group_size, other.method) != (
                        entry.bits,
                        entry.group_size,
                        entry.method,
                    ):
                        problems.append(
                            {
                                "kind": "constraint_violation",
                                "unit_id": entry.unit_id,
                                "detail": (
                                    f"{unit.constraint} group requires one "
                                    f"configuration; {entry.unit_id} and {peer} differ"
                                ),
                            }
                        )
            legal = [
                config
                for config in unit.legal_configs
                if config.bits == entry.bits
                and config.group_size == entry.group_size
                and config.method == entry.method
            ]
            if entry.keep_fp16:
                continue
            if not legal:
                problems.append(
                    {
                        "kind": "illegal_config",
                        "unit_id": entry.unit_id,
                        "detail": (
                            f"bits={entry.bits} group={entry.group_size} "
                            f"method={entry.method!r} is not executable for this unit"
                        ),
                        "legal_configs": [
                            config.as_dict() for config in unit.legal_configs
                        ],
                    }
                )
        return PolicyValidation(valid=not problems, problems=problems)

    def to_yaml(self) -> str:
        import yaml

        payload = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "name": self.name,
            "units": [entry.as_dict() for entry in self.entries],
            "quality_gate": dict(self.quality_gate),
            "cost": dict(self.cost),
            "provenance": dict(self.provenance),
            "known_limitations": list(self.known_limitations),
        }
        return yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> "MixedPrecisionPolicy":
        import yaml

        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ConfigError("policy YAML must be a mapping")
        if payload.get("kind") != POLICY_KIND:
            raise ConfigError(
                f"policy kind {payload.get('kind')!r} is not {POLICY_KIND!r}"
            )
        if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise ConfigError(
                f"policy schema {payload.get('schema_version')!r} is not the "
                f"supported {POLICY_SCHEMA_VERSION!r}"
            )
        unknown = set(payload) - {
            "schema_version",
            "kind",
            "name",
            "units",
            "quality_gate",
            "cost",
            "provenance",
            "known_limitations",
        }
        if unknown:
            raise ConfigError(f"policy has unknown field(s): {sorted(unknown)}")
        entries = []
        for item in payload.get("units", []):
            unknown_entry = set(item) - set(UnitPolicyEntry("x", 4).as_dict())
            if unknown_entry:
                raise ConfigError(
                    f"policy unit entry has unknown field(s): {sorted(unknown_entry)}"
                )
            entries.append(UnitPolicyEntry(**item))
        return cls(
            name=str(payload.get("name", "policy")),
            entries=entries,
            quality_gate=dict(payload.get("quality_gate", {})),
            cost=dict(payload.get("cost", {})),
            provenance=dict(payload.get("provenance", {})),
            known_limitations=list(payload.get("known_limitations", [])),
        )

    @property
    def policy_hash(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "units": [entry.as_dict() for entry in self.entries],
                "quality_gate": self.quality_gate,
                "cost": self.cost,
                "provenance": self.provenance,
                "known_limitations": self.known_limitations,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> Dict[str, Any]:
        bits_histogram: Dict[str, int] = {}
        for entry in self.entries:
            label = "fp16" if entry.keep_fp16 else f"w{entry.bits}"
            bits_histogram[label] = bits_histogram.get(label, 0) + 1
        return {
            "name": self.name,
            "policy_hash": self.policy_hash,
            "units": len(self.entries),
            "bits_histogram": dict(sorted(bits_histogram.items())),
            "quality_gate": dict(self.quality_gate),
            "cost": dict(self.cost),
        }


# ── search drivers (E05-05 §10.1) ─────────────────────────────────────────


@dataclass
class CandidateEvaluation:
    """One evaluated candidate in a search trace."""

    candidate_id: str
    parent: Optional[str]
    change: Dict[str, Any]
    predicted: Dict[str, float] = field(default_factory=dict)
    measured: Dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    reject_reason: str = ""
    policy_hash: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent": self.parent,
            "change": dict(self.change),
            "predicted": dict(self.predicted),
            "measured": dict(self.measured),
            "accepted": self.accepted,
            "reject_reason": self.reject_reason,
            "policy_hash": self.policy_hash,
        }


@dataclass
class SearchTrace:
    """Full search record: every candidate, not only the winner."""

    search_id: str
    method: str
    budget: Dict[str, Any]
    evaluations: List[CandidateEvaluation] = field(default_factory=list)
    stop_reason: str = ""
    selected: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "search_id": self.search_id,
            "method": self.method,
            "budget": dict(self.budget),
            "evaluations": [item.as_dict() for item in self.evaluations],
            "stop_reason": self.stop_reason,
            "selected": self.selected,
        }


def baseline_policies(
    units: Sequence[InterventionUnit],
    *,
    w8: UnitConfig,
    w4: UnitConfig,
    heuristic_fp16_units: Optional[Sequence[str]] = None,
    direct_harm_scores: Optional[Mapping[str, float]] = None,
    recovery_per_byte_scores: Optional[Mapping[str, float]] = None,
) -> Dict[str, MixedPrecisionPolicy]:
    """The auditable baseline set (E05-05 §10.1).

    Returns ``all_fp16``, ``all_w8``, ``all_w4``, ``heuristic_skip`` and — when
    scores are provided — ``direct_harm_greedy`` and
    ``recovery_per_byte_greedy``. Greedy baselines are *baselines*, not assumed
    optima.
    """
    ordered = sorted(units, key=lambda item: item.unit_id)
    policies: Dict[str, MixedPrecisionPolicy] = {}

    def _policy(name: str, chooser: Callable[[InterventionUnit], UnitPolicyEntry]) -> MixedPrecisionPolicy:
        return MixedPrecisionPolicy(
            name=name,
            entries=[chooser(unit) for unit in ordered],
            provenance={"method": name, "units": len(ordered)},
        )

    def _fp16(unit: InterventionUnit) -> UnitPolicyEntry:
        return UnitPolicyEntry(unit_id=unit.unit_id, bits=KEEP_FP16, keep_fp16=True)

    def _config_entry(unit: InterventionUnit, config: UnitConfig) -> UnitPolicyEntry:
        return UnitPolicyEntry(
            unit_id=unit.unit_id,
            bits=config.bits,
            group_size=config.group_size,
            method=config.method,
            packed_layout=config.packed_layout,
            scheme_hash=config.scheme_hash,
        )

    policies["all_fp16"] = _policy("all_fp16", _fp16)
    policies["all_w8"] = _policy("all_w8", lambda unit: _config_entry(unit, w8))
    policies["all_w4"] = _policy("all_w4", lambda unit: _config_entry(unit, w4))

    skip = set(heuristic_fp16_units or ())
    policies["heuristic_skip"] = _policy(
        "heuristic_skip",
        lambda unit: _fp16(unit) if unit.unit_id in skip else _config_entry(unit, w4),
    )

    if direct_harm_scores is not None:
        ranked = sorted(
            ordered,
            key=lambda unit: (-direct_harm_scores.get(unit.unit_id, 0.0), unit.unit_id),
        )

        def _harm_chooser(unit: InterventionUnit) -> UnitPolicyEntry:
            position = ranked.index(unit)
            return _config_entry(unit, w8) if position < max(1, len(ranked) // 4) else _config_entry(unit, w4)

        policies["direct_harm_greedy"] = _policy("direct_harm_greedy", _harm_chooser)

    if recovery_per_byte_scores is not None:
        ranked = sorted(
            ordered,
            key=lambda unit: (
                -recovery_per_byte_scores.get(unit.unit_id, 0.0),
                unit.unit_id,
            ),
        )

        def _recovery_chooser(unit: InterventionUnit) -> UnitPolicyEntry:
            position = ranked.index(unit)
            return _config_entry(unit, w8) if position < max(1, len(ranked) // 4) else _config_entry(unit, w4)

        policies["recovery_per_byte_greedy"] = _policy(
            "recovery_per_byte_greedy", _recovery_chooser
        )
    return policies


def greedy_search(
    units: Sequence[InterventionUnit],
    *,
    w8: UnitConfig,
    w4: UnitConfig,
    evaluate: Callable[[MixedPrecisionPolicy], Dict[str, float]],
    quality_pass: Callable[[Dict[str, float]], bool],
    byte_budget: int,
    unit_bytes: Callable[[InterventionUnit, UnitConfig], int],
    search_id: str = "greedy",
    score_key: str = "quality_margin",
) -> Tuple[MixedPrecisionPolicy, SearchTrace]:
    """Greedy upgrade search: start from all-W4, upgrade units to W8 while it helps.

    Each step evaluates every single-unit upgrade (measured, not predicted),
    keeps the best *quality gain per extra byte* that still satisfies the
    quality gate, and stops when no upgrade improves the score, the byte
    budget is exhausted, or every unit is at W8. All evaluations — including
    rejected ones — are recorded in the trace.
    """
    ordered = sorted(units, key=lambda item: item.unit_id)
    current = {
        unit.unit_id: UnitPolicyEntry(
            unit_id=unit.unit_id,
            bits=w4.bits,
            group_size=w4.group_size,
            method=w4.method,
            packed_layout=w4.packed_layout,
            scheme_hash=w4.scheme_hash,
        )
        for unit in ordered
    }
    trace = SearchTrace(
        search_id=search_id,
        method="greedy_quality_per_byte",
        budget={"byte_budget": byte_budget, "score_key": score_key},
    )

    def _policy(entries: Mapping[str, UnitPolicyEntry]) -> MixedPrecisionPolicy:
        return MixedPrecisionPolicy(
            name=search_id,
            entries=[entries[unit.unit_id] for unit in ordered],
            provenance={"search_id": search_id, "method": "greedy"},
        )

    def _bytes(entries: Mapping[str, UnitPolicyEntry]) -> int:
        total = 0
        for unit in ordered:
            entry = entries[unit.unit_id]
            config = w8 if entry.bits == w8.bits else w4
            total += unit_bytes(unit, config)
        return total

    best_policy = _policy(current)
    best_metrics = evaluate(best_policy)
    trace.evaluations.append(
        CandidateEvaluation(
            candidate_id=f"{search_id}:seed",
            parent=None,
            change={"init": "all_w4"},
            measured=dict(best_metrics),
            accepted=quality_pass(best_metrics),
            reject_reason="" if quality_pass(best_metrics) else "seed policy fails quality gate",
            policy_hash=best_policy.policy_hash,
        )
    )
    if not quality_pass(best_metrics):
        trace.stop_reason = "seed policy already fails the quality gate"
        return best_policy, trace

    best_score = best_metrics.get(score_key, 0.0)
    while True:
        used_bytes = _bytes(current)
        candidates: List[Tuple[float, InterventionUnit, Dict[str, float], MixedPrecisionPolicy]] = []
        for unit in ordered:
            if current[unit.unit_id].bits == w8.bits:
                continue
            extra = unit_bytes(unit, w8) - unit_bytes(unit, w4)
            if used_bytes + extra > byte_budget:
                continue
            trial = dict(current)
            trial[unit.unit_id] = UnitPolicyEntry(
                unit_id=unit.unit_id,
                bits=w8.bits,
                group_size=w8.group_size,
                method=w8.method,
                packed_layout=w8.packed_layout,
                scheme_hash=w8.scheme_hash,
            )
            policy = _policy(trial)
            metrics = evaluate(policy)
            accepted = quality_pass(metrics)
            improvement = metrics.get(score_key, 0.0) - best_score
            ratio = improvement / extra if extra > 0 else float("inf")
            trace.evaluations.append(
                CandidateEvaluation(
                    candidate_id=f"{search_id}:upgrade:{unit.unit_id}",
                    parent=best_policy.policy_hash,
                    change={
                        "unit_id": unit.unit_id,
                        "bits": [current[unit.unit_id].bits, w8.bits],
                    },
                    measured=dict(metrics),
                    accepted=accepted and improvement > 0,
                    reject_reason=(
                        ""
                        if accepted and improvement > 0
                        else (
                            "quality gate failed"
                            if not accepted
                            else "no quality gain"
                        )
                    ),
                    policy_hash=policy.policy_hash,
                )
            )
            if accepted and improvement > 0:
                candidates.append((ratio, unit, metrics, policy))
        if not candidates:
            trace.stop_reason = "no single-unit upgrade improves the score within budget"
            break
        candidates.sort(key=lambda item: (-item[0], item[1].unit_id))
        _ratio, chosen_unit, metrics, policy = candidates[0]
        current[chosen_unit.unit_id] = UnitPolicyEntry(
            unit_id=chosen_unit.unit_id,
            bits=w8.bits,
            group_size=w8.group_size,
            method=w8.method,
            packed_layout=w8.packed_layout,
            scheme_hash=w8.scheme_hash,
        )
        best_policy = policy
        best_metrics = metrics
        best_score = metrics.get(score_key, best_score)
        if _bytes(current) >= byte_budget:
            trace.stop_reason = "byte budget exhausted"
            break
        if all(entry.bits == w8.bits for entry in current.values()):
            trace.stop_reason = "all units upgraded to w8"
            break
    trace.selected = best_policy.policy_hash
    trace.budget["final_bytes"] = _bytes(current)
    return best_policy, trace


def counterfactual_check(
    policy: MixedPrecisionPolicy,
    *,
    unit_id: str,
    measured_fn: Callable[[MixedPrecisionPolicy], Dict[str, float]],
    downgrade_bits: int,
    score_key: str,
    expect_direction: str = "degrade",
) -> Dict[str, Any]:
    """Counterfactual: does changing one unit move the metric as predicted?

    E05-05 §11 step 19 requires causal evidence for the retained layers: a
    policy that claims "unit X must stay at higher precision" should show that
    lowering it degrades the measured metric (and that raising it restores it).
    """
    before = measured_fn(policy)
    entry_map = policy.entry_map()
    if unit_id not in entry_map:
        raise ConfigError(f"unit {unit_id!r} is not part of this policy")
    modified = MixedPrecisionPolicy(
        name=f"{policy.name}:counterfactual:{unit_id}",
        entries=[
            UnitPolicyEntry(
                unit_id=entry.unit_id,
                bits=downgrade_bits if entry.unit_id == unit_id else entry.bits,
                group_size=(
                    entry_map[unit_id].group_size
                    if entry.unit_id == unit_id
                    else entry.group_size
                ),
                method=entry.method,
                packed_layout=entry.packed_layout,
                keep_fp16=False,
                scheme_hash=entry.scheme_hash,
            )
            for entry in policy.entries
        ],
        quality_gate=dict(policy.quality_gate),
        cost=dict(policy.cost),
        provenance={**policy.provenance, "counterfactual_of": policy.policy_hash},
    )
    after = measured_fn(modified)
    delta = after.get(score_key, float("nan")) - before.get(score_key, float("nan"))
    consistent = (
        delta < 0 if expect_direction == "degrade" else delta > 0
    )
    return {
        "unit_id": unit_id,
        "before": before,
        "after": after,
        "delta": delta,
        "expected_direction": expect_direction,
        "consistent": bool(consistent),
        "modified_policy_hash": modified.policy_hash,
    }


def selection_regret(
    trace: SearchTrace, *, best_measured: float, selected_measured: float
) -> Dict[str, Any]:
    """Report the search's selection regret and the trial count.

    ``regret = best_observed - selected`` over *all* evaluated candidates:
    hiding a better rejected candidate would otherwise be invisible
    (E05-05 §14 "只保存最终policy，不保存候选和轨迹").
    """
    observed = [
        evaluation.measured.get("score", float("nan"))
        for evaluation in trace.evaluations
        if "score" in evaluation.measured
    ]
    observed = [value for value in observed if not math.isnan(value)]
    return {
        "trials": len(trace.evaluations),
        "best_observed": max(observed) if observed else float("nan"),
        "selected_measured": selected_measured,
        "best_measured": best_measured,
        "regret_vs_best_observed": (
            (max(observed) - selected_measured) if observed else float("nan")
        ),
    }


__all__ = [
    "KEEP_FP16",
    "CandidateEvaluation",
    "MixedPrecisionPolicy",
    "POLICY_KIND",
    "POLICY_SCHEMA_VERSION",
    "PolicyValidation",
    "SearchTrace",
    "UnitPolicyEntry",
    "baseline_policies",
    "counterfactual_check",
    "greedy_search",
    "selection_regret",
]
