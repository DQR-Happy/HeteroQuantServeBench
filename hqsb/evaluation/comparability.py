"""E12-01: cell-by-cell comparability audit, verdicts and illegal-join guard.

The experiment answers "which cells may be compared, under which frozen
contract, and where does a difference change the question itself?".  It is the
admission gate of S12: a result that has no verdict may not be ranked, priced or
put on a Pareto frontier, no matter how complete its numbers look.

Design rules enforced here:

* four-state verdicts (``COMPARABLE`` / ``CONDITIONAL`` / ``NOT_COMPARABLE`` /
  ``INSUFFICIENT_EVIDENCE``) — comparability is not a boolean;
* ``CONDITIONAL`` requires a preregistered normalization *with* its allowed
  analyses and forbidden claims; without them the verdict degrades to
  ``NOT_COMPARABLE``;
* a missing state is an evidence state: it never carries a fabricated number;
* the rule engine never overrides a human disagreement — disagreements are
  recorded, not averaged away.

Nothing here executes an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.contracts import (
    AUDIT_DIMENSIONS,
    FIELD_ALLOWED_DIFFERENCE,
    FIELD_CONDITIONALLY_NORMALIZABLE,
    FIELD_FORBIDDEN_DIFFERENCE,
    FIELD_INVARIANT,
    FIELD_REPORT_ONLY,
    ComparisonContract,
    FieldDiff,
    NormalizationFormula,
    conditional_differences,
    diff_contracts,
    invariant_violations,
    reason_codes_for,
)
from hqsb.evaluation.identity import sha256_text, stable_id
from hqsb.evaluation.layers import LAYERS
from hqsb.evaluation.records import (
    COMPARABLE,
    COMPARABILITY_STATES,
    CONDITIONAL,
    INSUFFICIENT_EVIDENCE,
    MISSINGNESS_CODES,
    NOT_COMPARABLE,
    RANKABLE_STATES,
)

EXPERIMENT_ID = "E12-01"
TITLE = "候选矩阵逐格可比性审计与 Comparison Contract"
CLAIM_BOUNDARY = (
    "本实验通过只证明'比较问题被定义正确并有合格候选'，"
    "不证明任一候选更快、更省电或更便宜；不可比项不得进入排名/Pareto。"
)

POLICY_VERSION = "s12_policy_1.0.0"

#: Error codes returned by the illegal-join guard (stable strings).
INVALID_JOIN_ERROR_CODES: Tuple[str, ...] = (
    "JOIN_UNKNOWN_COMPARISON_GROUP",
    "JOIN_CONTRACT_VERSION_CHANGED",
    "JOIN_LAYER_MIXED",
    "JOIN_QUEUE_MIXED",
    "JOIN_QUALITY_FAILED",
    "JOIN_NOT_COMPARABLE",
    "JOIN_INSUFFICIENT_EVIDENCE",
    "JOIN_MISSING_STATE",
    "JOIN_ACTUAL_BACKEND_UNKNOWN",
    "JOIN_CONDITIONAL_NOT_OPTED_IN",
    "JOIN_UNIT_MISMATCH",
)

#: The protocol negative cases (§9 step 30 / §15 方法学 PASS #6).
NEGATIVE_CASE_KINDS: Mapping[str, str] = {
    "model_hash_differs": "两个 cell 使用不同权重的 ModelArtifact（FORBIDDEN_DIFFERENCE）",
    "tokenizer_differs": "tokenizer 不同 → tokens/s 分母不同",
    "eos_stop_differs": "一边 EOS 提前停止，实际工作量不同",
    "quality_gate_failed": "低精度候选未过共同质量门仍想排名",
    "timing_boundary_differs": "一边 client 边界，一边 model-core 边界",
    "warmup_state_differs": "一边冷启动，一边 steady state",
    "actual_backend_unknown": "requested=Triton 但 actual 未知",
    "unit_mismatch": "一边 ms，一边 µs（单位未审计）",
    "device_count_scaled": "单卡与多卡直接按设备数缩放",
    "missing_as_zero": "不支持项被填 0 进入成本/能耗最优",
    "precision_contract_incomplete": "只写 FP16，未区分权重/累加/KV",
    "workload_layer_mixed": "model-core 与 service 共用同一 latency 字段",
    "conditional_without_formula": "CONDITIONAL 没有归一化公式与禁止主张",
    "stale_quality_evidence": "模型/precision 变更后旧质量门未失效",
}


def negative_case_matrix() -> Tuple[Dict[str, Any], ...]:
    """The injectable protocol differences with their expected verdicts."""
    expected: Mapping[str, Tuple[str, str]] = {
        "model_hash_differs": (NOT_COMPARABLE, "MODEL_ARTIFACT_MISMATCH"),
        "tokenizer_differs": (NOT_COMPARABLE, "TOKENIZER_MISMATCH"),
        "eos_stop_differs": (NOT_COMPARABLE, "STOP_POLICY_MISMATCH"),
        "quality_gate_failed": (NOT_COMPARABLE, "QUALITY_GATE_FAILED"),
        "timing_boundary_differs": (NOT_COMPARABLE, "TIMING_BOUNDARY_MISMATCH"),
        "warmup_state_differs": (CONDITIONAL, "WARMUP_STATE_MISMATCH"),
        "actual_backend_unknown": (INSUFFICIENT_EVIDENCE, "ACTUAL_BACKEND_UNKNOWN"),
        "unit_mismatch": (INSUFFICIENT_EVIDENCE, "UNIT_MISMATCH"),
        "device_count_scaled": (CONDITIONAL, "DEVICE_COUNT_MISMATCH"),
        "missing_as_zero": (NOT_COMPARABLE, "WORKLOAD_SPEC_MISMATCH"),
        "precision_contract_incomplete": (INSUFFICIENT_EVIDENCE, "PRECISION_CONTRACT_INCOMPLETE"),
        "workload_layer_mixed": (NOT_COMPARABLE, "LAYER_MIXED"),
        "conditional_without_formula": (NOT_COMPARABLE, "POLICY_VIOLATION"),
        "stale_quality_evidence": (NOT_COMPARABLE, "QUALITY_EVIDENCE_STALE"),
    }
    rows = []
    for case_id, description in NEGATIVE_CASE_KINDS.items():
        verdict, reason = expected[case_id]
        rows.append(
            {
                "case_id": case_id,
                "difference_class": case_id,
                "expected_verdict": verdict,
                "expected_reason_code": reason,
                "description": description,
            }
        )
    return tuple(rows)


# ── field classification policy ───────────────────────────────────────────


@dataclass
class FieldClassificationPolicy:
    """Field classes with a version; unknown fields default to INVARIANT."""

    policy_version: str = POLICY_VERSION
    overrides: Mapping[str, str] = field(default_factory=dict)

    ALLOWED_CLASSES = (
        FIELD_INVARIANT,
        FIELD_ALLOWED_DIFFERENCE,
        FIELD_CONDITIONALLY_NORMALIZABLE,
        FIELD_REPORT_ONLY,
        FIELD_FORBIDDEN_DIFFERENCE,
    )

    def classify(self, field_path: str) -> str:
        if field_path in self.overrides:
            value = self.overrides[field_path]
            if value not in self.ALLOWED_CLASSES:
                raise ConfigError(f"unknown field class {value!r} for {field_path!r}")
            return value
        # Fail closed: an unclassified field is treated as invariant.
        return FIELD_INVARIANT

    def validate(self) -> List[str]:
        problems = [
            f"unknown field class {value!r} for {name!r}"
            for name, value in sorted(self.overrides.items())
            if value not in self.ALLOWED_CLASSES
        ]
        if not self.policy_version:
            problems.append("a field classification policy needs a version")
        return problems

    def as_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "field_path": name,
                "group": name.split(".")[0],
                "field_class": value,
                "policy_version": self.policy_version,
                "rationale": "explicit override",
            }
            for name, value in sorted(self.overrides.items())
        ]


# ── verdicts ──────────────────────────────────────────────────────────────


@dataclass
class ComparabilityVerdict:
    """One verdict row (§11 minimum fields)."""

    verdict_id: str = ""
    audit_id: str = ""
    comparison_group_id: str = ""
    candidate_a: str = ""
    candidate_b_or_reference: str = ""
    field_path: str = ""
    field_class: str = FIELD_INVARIANT
    value_a_hash: str = ""
    value_b_hash: str = ""
    match_status: str = "MATCH"
    verdict: str = INSUFFICIENT_EVIDENCE
    reason_code: str = ""
    evidence_refs: Tuple[str, ...] = ()
    normalization_formula_id: str = ""
    allowed_analyses: Tuple[str, ...] = ()
    forbidden_claims: Tuple[str, ...] = ()
    reviewer: str = ""
    reviewed_at: str = ""
    policy_version: str = POLICY_VERSION

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.verdict not in COMPARABILITY_STATES:
            problems.append(f"unknown comparability verdict {self.verdict!r}")
        if self.field_class not in FieldClassificationPolicy.ALLOWED_CLASSES:
            problems.append(f"unknown field class {self.field_class!r}")
        if self.verdict == COMPARABLE:
            if not self.comparison_group_id:
                problems.append("COMPARABLE must reference its comparison group/contract")
            if self.match_status == "DIFF" and self.field_class in (
                FIELD_INVARIANT,
                FIELD_FORBIDDEN_DIFFERENCE,
            ):
                problems.append(
                    "an invariant/forbidden difference cannot be a COMPARABLE verdict"
                )
        if self.verdict == CONDITIONAL:
            if not self.normalization_formula_id:
                problems.append("CONDITIONAL must reference a normalization formula")
            if not self.allowed_analyses:
                problems.append("CONDITIONAL must list the analyses it allows")
            if not self.forbidden_claims:
                problems.append("CONDITIONAL must state the claims it forbids")
        if self.verdict == NOT_COMPARABLE and self.allowed_analyses:
            problems.append("NOT_COMPARABLE must not carry ranking/allowed-analysis scope")
        if self.verdict == INSUFFICIENT_EVIDENCE and not self.reason_code:
            problems.append("INSUFFICIENT_EVIDENCE must state what is missing")
        if not self.policy_version:
            problems.append("a verdict must cite its policy version")
        return problems

    @property
    def rankable(self) -> bool:
        return self.verdict in RANKABLE_STATES

    def as_dict(self) -> Dict[str, Any]:
        return {
            "verdict_id": self.verdict_id,
            "audit_id": self.audit_id,
            "comparison_group_id": self.comparison_group_id,
            "candidate_a": self.candidate_a,
            "candidate_b_or_reference": self.candidate_b_or_reference,
            "field_path": self.field_path,
            "field_class": self.field_class,
            "value_a_hash": self.value_a_hash,
            "value_b_hash": self.value_b_hash,
            "match_status": self.match_status,
            "verdict": self.verdict,
            "reason_code": self.reason_code,
            "evidence_refs": list(self.evidence_refs),
            "normalization_formula_id": self.normalization_formula_id,
            "allowed_analyses": list(self.allowed_analyses),
            "forbidden_claims": list(self.forbidden_claims),
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "policy_version": self.policy_version,
        }


def decide_comparability(
    diffs: Sequence[FieldDiff],
    *,
    quality_ok: bool,
    evidence_complete: bool,
    normalization: Optional[NormalizationFormula] = None,
    group_id: str = "",
    candidate_a: str = "",
    candidate_b: str = "",
    policy_version: str = POLICY_VERSION,
    evidence_refs: Sequence[str] = (),
) -> ComparabilityVerdict:
    """The four-state rule engine (order: evidence → quality → diff → normalize).

    The order may not change: computing a weighted score first and "punishing"
    a quality failure with a low weight is exactly what S12 forbids.
    """
    base = {
        "audit_id": stable_id("audit", {"group": group_id, "a": candidate_a, "b": candidate_b}),
        "comparison_group_id": group_id,
        "candidate_a": candidate_a,
        "candidate_b_or_reference": candidate_b,
        "evidence_refs": tuple(evidence_refs),
        "policy_version": policy_version,
    }
    if not evidence_complete:
        verdict = ComparabilityVerdict(
            verdict_id=stable_id("vd", {"group": group_id, "a": candidate_a, "b": candidate_b, "r": "missing"}),
            verdict=INSUFFICIENT_EVIDENCE,
            reason_code="MISSING_EVIDENCE",
            **base,
        )
        return verdict
    if not quality_ok:
        return ComparabilityVerdict(
            verdict_id=stable_id("vd", {"group": group_id, "a": candidate_a, "b": candidate_b, "r": "quality"}),
            verdict=NOT_COMPARABLE,
            reason_code="QUALITY_GATE_FAILED",
            **base,
        )
    hard = invariant_violations(diffs)
    if hard:
        reason = hard[0].reason_code or "POLICY_VIOLATION"
        return ComparabilityVerdict(
            verdict_id=stable_id("vd", {"group": group_id, "a": candidate_a, "b": candidate_b, "r": reason}),
            field_path=hard[0].field_path,
            field_class=hard[0].field_class,
            value_a_hash=hard[0].value_a_hash,
            value_b_hash=hard[0].value_b_hash,
            match_status="DIFF",
            verdict=NOT_COMPARABLE,
            reason_code=reason,
            **base,
        )
    soft = conditional_differences(diffs)
    if soft:
        if normalization is None or normalization.validate():
            return ComparabilityVerdict(
                verdict_id=stable_id("vd", {"group": group_id, "a": candidate_a, "b": candidate_b, "r": "cond"}),
                field_path=soft[0].field_path,
                field_class=soft[0].field_class,
                match_status="DIFF",
                verdict=NOT_COMPARABLE,
                reason_code="POLICY_VIOLATION",
                **base,
            )
        return ComparabilityVerdict(
            verdict_id=stable_id("vd", {"group": group_id, "a": candidate_a, "b": candidate_b, "r": "conditional"}),
            field_path=soft[0].field_path,
            field_class=soft[0].field_class,
            value_a_hash=soft[0].value_a_hash,
            value_b_hash=soft[0].value_b_hash,
            match_status="DIFF",
            verdict=CONDITIONAL,
            reason_code=soft[0].reason_code,
            normalization_formula_id=normalization.formula_id,
            allowed_analyses=normalization.allowed_analyses,
            forbidden_claims=normalization.forbidden_claims,
            **base,
        )
    return ComparabilityVerdict(
        verdict_id=stable_id("vd", {"group": group_id, "a": candidate_a, "b": candidate_b, "r": "comparable"}),
        verdict=COMPARABLE,
        reason_code="",
        **base,
    )


# ── per-cell audit ────────────────────────────────────────────────────────


def _reclassify(policy: FieldClassificationPolicy, diff: FieldDiff) -> str:
    """Apply the audit policy to a field path, falling back to the catalog class.

    ``diff_contracts`` already classified the field from the frozen catalog; an
    explicit policy override wins, and an unknown path stays ``INVARIANT``
    (fail closed) unless the catalog recognised it.
    """
    short = diff.field_path.split(".")[-1]
    if short in policy.overrides:
        return policy.classify(short)
    if diff.field_class:
        return diff.field_class
    return policy.classify(diff.field_path)


@dataclass
class CellAudit:
    """One audited cell: its diffs, its missing fields and its verdict."""

    cell_id: str
    comparison_id: str
    candidate_id: str
    layer: str
    scenario: str
    diffs: Tuple[FieldDiff, ...] = ()
    missing: Tuple[str, ...] = ()
    unresolved: Tuple[str, ...] = ()
    verdict: ComparabilityVerdict = field(default_factory=ComparabilityVerdict)
    normalization_formula_id: str = ""
    evidence_refs: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "comparison_id": self.comparison_id,
            "candidate_id": self.candidate_id,
            "layer": self.layer,
            "scenario": self.scenario,
            "diffs": [diff.as_dict() for diff in self.diffs],
            "missing": list(self.missing),
            "unresolved": list(self.unresolved),
            "verdict": self.verdict.verdict,
            "reason_code": self.verdict.reason_code,
            "normalization_formula_id": self.normalization_formula_id,
            "evidence_refs": list(self.evidence_refs),
        }


CELL_REQUIRED_FIELDS: Tuple[str, ...] = (
    "cell_id",
    "comparison_id",
    "candidate_id",
    "layer",
    "scenario",
    "workload_spec_id",
    "measurement_contract_id",
    "quality_gate_id",
    "actual_backend",
)


def audit_cell(
    cell: Mapping[str, Any],
    contract: ComparisonContract,
    *,
    policy: FieldClassificationPolicy,
    quality_ok: bool = True,
    evidence_complete: bool = True,
    normalization: Optional[NormalizationFormula] = None,
    reference: Optional[Mapping[str, Any]] = None,
) -> CellAudit:
    """Audit one cell against the contract (or against a reference cell)."""
    problems = contract.validate()
    if problems:
        raise ConfigError("comparison contract is invalid: " + "; ".join(problems))
    problems = policy.validate()
    if problems:
        raise ConfigError("field classification policy is invalid: " + "; ".join(problems))

    missing_fields = [name for name in CELL_REQUIRED_FIELDS if cell.get(name) in (None, "")]
    unresolved: List[str] = []
    for name in missing_fields:
        if name in ("actual_backend",):
            unresolved.append("ACTUAL_BACKEND_UNKNOWN")
        elif name == "quality_gate_id":
            unresolved.append("QUALITY_GATE_MISMATCH")
        else:
            unresolved.append("MISSING_EVIDENCE")

    left = reference if reference is not None else contract.payload()["comparison_contract"]
    diffs = tuple(
        FieldDiff(
            field_path=diff.field_path,
            field_class=_reclassify(policy, diff),
            value_a_hash=diff.value_a_hash,
            value_b_hash=diff.value_b_hash,
            dimension=diff.dimension,
            match_status=diff.match_status,
            reason_code=diff.reason_code,
        )
        for diff in diff_contracts(contract, contract, value_a=left, value_b=cell)
    )
    verdict = decide_comparability(
        diffs,
        quality_ok=quality_ok and "QUALITY_GATE_FAILED" not in unresolved,
        evidence_complete=evidence_complete and not missing_fields,
        normalization=normalization,
        group_id=str(contract.comparison_id),
        candidate_a=str(cell.get("candidate_id", "")),
        candidate_b=str(cell.get("reference_candidate_id", "reference")),
    )
    return CellAudit(
        cell_id=str(cell.get("cell_id", "")),
        comparison_id=str(contract.comparison_id),
        candidate_id=str(cell.get("candidate_id", "")),
        layer=str(cell.get("layer", contract.workload.layer)),
        scenario=str(cell.get("scenario", contract.workload.scenario)),
        diffs=diffs,
        missing=tuple(missing_fields),
        unresolved=tuple(sorted(set(unresolved))),
        verdict=verdict,
        normalization_formula_id=verdict.normalization_formula_id,
        evidence_refs=tuple(verdict.evidence_refs),
    )


@dataclass
class AuditResult:
    """Aggregate of one audit: verdicts, histogram, gaps and action items."""

    audit_id: str
    cells: Tuple[CellAudit, ...] = ()
    policy_version: str = POLICY_VERSION

    def verdict_histogram(self) -> Dict[str, int]:
        counts: Dict[str, int] = {state: 0 for state in COMPARABILITY_STATES}
        for cell in self.cells:
            counts[cell.verdict.verdict] = counts.get(cell.verdict.verdict, 0) + 1
        return dict(sorted(counts.items()))

    def unresolved_dependencies(self) -> Tuple[str, ...]:
        rows: set = set()
        for cell in self.cells:
            rows.update(cell.unresolved)
        return tuple(sorted(rows))

    def action_items(self) -> Tuple[Dict[str, Any], ...]:
        return action_items_from_gaps(self.cells)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "policy_version": self.policy_version,
            "cells": [cell.as_dict() for cell in self.cells],
            "verdict_histogram": self.verdict_histogram(),
            "unresolved_dependencies": list(self.unresolved_dependencies()),
            "action_items": [dict(row) for row in self.action_items()],
        }


def run_audit(
    cells: Sequence[Mapping[str, Any]],
    contract: ComparisonContract,
    *,
    policy: FieldClassificationPolicy,
    quality_ok_by_cell: Optional[Mapping[str, bool]] = None,
    evidence_complete_by_cell: Optional[Mapping[str, bool]] = None,
    normalization: Optional[NormalizationFormula] = None,
) -> AuditResult:
    audits = []
    for cell in cells:
        cell_id = str(cell.get("cell_id", ""))
        audits.append(
            audit_cell(
                cell,
                contract,
                policy=policy,
                quality_ok=(quality_ok_by_cell or {}).get(cell_id, True),
                evidence_complete=(evidence_complete_by_cell or {}).get(cell_id, True),
                normalization=normalization,
            )
        )
    return AuditResult(
        audit_id=stable_id("audit", {"comparison": contract.comparison_id, "cells": sorted(str(c.get("cell_id", "")) for c in cells)}),
        cells=tuple(audits),
        policy_version=policy.policy_version,
    )


def action_items_from_gaps(cells: Sequence[CellAudit]) -> Tuple[Dict[str, Any], ...]:
    """Turn ``INSUFFICIENT_EVIDENCE`` into executable probe/quality/rerun tasks."""
    rows: List[Dict[str, Any]] = []
    for cell in cells:
        if cell.verdict.verdict not in (INSUFFICIENT_EVIDENCE, NOT_COMPARABLE):
            continue
        for reason in cell.unresolved or (cell.verdict.reason_code,):
            rows.append(
                {
                    "item_id": stable_id("gap", {"cell": cell.cell_id, "reason": reason}),
                    "cell_id": cell.cell_id,
                    "candidate_id": cell.candidate_id,
                    "reason": reason,
                    "action": _action_for_reason(reason),
                    "owner": "",
                    "priority": "P0",
                    "blocking": True,
                }
            )
    return tuple(sorted(rows, key=lambda row: (row["cell_id"], row["reason"])))


def _action_for_reason(reason: str) -> str:
    mapping = {
        "ACTUAL_BACKEND_UNKNOWN": "rerun with dispatch/kernel evidence (E12-03 step 11)",
        "QUALITY_GATE_FAILED": "rerun the shared quality gate; the cell stays out of performance",
        "QUALITY_GATE_MISMATCH": "restore the common quality gate id before comparing",
        "MISSING_EVIDENCE": "complete the missing identity/measurement fields",
        "PRECISION_CONTRACT_INCOMPLETE": "record weight/activation/accumulation/KV dtypes",
        "UNIT_MISMATCH": "re-express the metric in the contract unit and keep the raw value",
        "POLICY_VIOLATION": "preregister a normalization formula or shrink the comparison group",
    }
    return mapping.get(reason, "audit the field-level difference and rerun or shrink the group")


# ── audits: units, boundaries, quality dependency, actual backend ─────────


_UNIT_FAMILIES: Mapping[str, Tuple[str, ...]] = {
    "time": ("ns", "us", "ms", "s"),
    "bytes": ("B", "KiB", "MiB", "GiB"),
    "energy": ("J", "kJ", "Wh"),
    "power": ("W", "mW"),
    "rate": ("1/s", "request/s", "token/s"),
    "ratio": ("ratio", "%"),
}

_UNIT_SCALE: Mapping[str, float] = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "B": 1.0,
    "KiB": 1024.0,
    "MiB": 1024.0**2,
    "GiB": 1024.0**3,
    "J": 1.0,
    "kJ": 1000.0,
    "Wh": 3600.0,
    "W": 1.0,
    "mW": 1e-3,
    "ratio": 1.0,
    "%": 0.01,
}

_UNIT_CANONICAL: Mapping[str, str] = {
    "ns": "ns",
    "us": "ns",
    "ms": "ns",
    "s": "ns",
    "B": "B",
    "KiB": "B",
    "MiB": "B",
    "GiB": "B",
    "J": "J",
    "kJ": "J",
    "Wh": "J",
    "W": "W",
    "mW": "W",
    "ratio": "ratio",
    "%": "ratio",
}


def unit_family(unit: str) -> str:
    for family, units in _UNIT_FAMILIES.items():
        if unit in units:
            return family
    raise ConfigError(f"unknown unit {unit!r}")


def convert_unit(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """Convert a value, keeping the raw value and the formula in the row."""
    if unit_family(from_unit) != unit_family(to_unit):
        raise ConfigError(f"cannot convert {from_unit!r} to {to_unit!r}: different unit families")
    canonical_value = value * _UNIT_SCALE[from_unit]
    converted = canonical_value / _UNIT_SCALE[to_unit]
    return {
        "raw_value": value,
        "raw_unit": from_unit,
        "value": converted,
        "unit": to_unit,
        "canonical_unit": _UNIT_CANONICAL[to_unit],
        "formula": f"{value} {from_unit} -> {convert_expr(from_unit, to_unit)} -> {to_unit}",
    }


def convert_expr(from_unit: str, to_unit: str) -> str:
    return f"x * {_UNIT_SCALE[from_unit] / _UNIT_SCALE[to_unit]:.6g}"


def unit_audit(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    """Check ns/us/ms, bytes/GiB, W/J, request/token and device/system mixing."""
    findings: List[Dict[str, Any]] = []
    by_metric: Dict[str, set] = {}
    for row in rows:
        metric = str(row.get("metric_name", ""))
        unit = str(row.get("unit", ""))
        if metric and unit:
            by_metric.setdefault(metric, set()).add(unit)
    for metric, units in sorted(by_metric.items()):
        if len(units) > 1:
            families = {unit_family(unit) for unit in units}
            findings.append(
                {
                    "check_id": f"unit_mix:{metric}",
                    "metric_name": metric,
                    "units": sorted(units),
                    "convertible": len(families) == 1,
                    "status": "FAIL" if len(families) > 1 else "CONVERTIBLE",
                    "detail": (
                        "same metric reported in different unit families"
                        if len(families) > 1
                        else "unit conversion required, raw value must be preserved"
                    ),
                }
            )
    for row in rows:
        basis = str(row.get("normalization_basis", ""))
        denominator = str(row.get("denominator", ""))
        if basis.startswith("per_") and not denominator:
            findings.append(
                {
                    "check_id": f"denominator_missing:{row.get('row_id', '')}",
                    "metric_name": str(row.get("metric_name", "")),
                    "status": "FAIL",
                    "detail": "a per-* basis without a denominator cannot be audited",
                }
            )
    return tuple(findings)


def boundary_audit(
    rows: Sequence[Mapping[str, Any]],
    contract: ComparisonContract,
) -> Tuple[Dict[str, Any], ...]:
    """Same metric name must come from the same start/end events."""
    findings: List[Dict[str, Any]] = []
    boundaries = {
        str(row.get("metric_name", "")): str(row.get("boundary_id", ""))
        for row in rows
        if row.get("metric_name")
    }
    for metric, boundary in sorted(boundaries.items()):
        if not boundary:
            findings.append(
                {
                    "check_id": f"boundary_unproven:{metric}",
                    "metric_name": metric,
                    "boundary_id": "",
                    "status": "INSUFFICIENT_EVIDENCE",
                    "reason_code": "TIMING_BOUNDARY_MISMATCH",
                    "detail": "no trace evidence that the metric uses the contract boundary",
                }
            )
        elif boundary != contract.timing.boundary:
            findings.append(
                {
                    "check_id": f"boundary_mismatch:{metric}",
                    "metric_name": metric,
                    "boundary_id": boundary,
                    "status": "FAIL",
                    "reason_code": "TIMING_BOUNDARY_MISMATCH",
                    "detail": f"metric boundary {boundary!r} != contract boundary {contract.timing.boundary!r}",
                }
            )
    return tuple(findings)


def quality_dependency_audit(
    quality_rows: Sequence[Mapping[str, Any]],
    performance_rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    """A performance cell must cite a live quality gate result."""
    findings: List[Dict[str, Any]] = []
    live = {
        (str(row.get("candidate_id", "")), str(row.get("precision_contract_id", "")), str(row.get("model_artifact_id", ""))): row
        for row in quality_rows
        if str(row.get("status", "")) == "PASS"
    }
    for row in performance_rows:
        key = (
            str(row.get("candidate_id", "")),
            str(row.get("precision_contract_id", "")),
            str(row.get("model_artifact_id", "")),
        )
        match = live.get(key)
        if match is None:
            findings.append(
                {
                    "check_id": f"quality_missing:{row.get('cell_id', '')}",
                    "cell_id": str(row.get("cell_id", "")),
                    "status": "FAIL",
                    "reason_code": "QUALITY_GATE_FAILED",
                    "detail": "no passing quality result for this candidate/precision/model",
                }
            )
            continue
        if str(match.get("gate_id", "")) != str(row.get("quality_gate_id", "")):
            findings.append(
                {
                    "check_id": f"quality_stale:{row.get('cell_id', '')}",
                    "cell_id": str(row.get("cell_id", "")),
                    "status": "FAIL",
                    "reason_code": "QUALITY_EVIDENCE_STALE",
                    "detail": "quality result belongs to a different gate than the performance row",
                }
            )
    return tuple(findings)


def actual_backend_audit(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    """requested != actual must be explicit; unknown actual blocks comparison."""
    findings: List[Dict[str, Any]] = []
    for row in rows:
        requested = str(row.get("requested_backend", ""))
        actual = str(row.get("actual_backend", ""))
        reason = str(row.get("fallback_reason", ""))
        cell_id = str(row.get("cell_id", ""))
        if not actual:
            findings.append(
                {
                    "check_id": f"actual_unknown:{cell_id}",
                    "cell_id": cell_id,
                    "status": "INSUFFICIENT_EVIDENCE",
                    "reason_code": "ACTUAL_BACKEND_UNKNOWN",
                    "detail": "requested backend without dispatch/kernel evidence",
                }
            )
        elif requested and requested != actual and not reason:
            findings.append(
                {
                    "check_id": f"silent_fallback:{cell_id}",
                    "cell_id": cell_id,
                    "status": "FAIL",
                    "reason_code": "SILENT_FALLBACK",
                    "detail": f"requested {requested!r} but executed {actual!r} without a reason",
                }
            )
    return tuple(findings)


# ── illegal-join guard ────────────────────────────────────────────────────


@dataclass
class JoinDecision:
    """Result of trying to feed rows into a shared query / frontier."""

    accepted_ids: Tuple[str, ...] = ()
    rejected: Tuple[Mapping[str, Any], ...] = ()

    @property
    def accepted(self) -> Tuple[str, ...]:
        return self.accepted_ids

    @property
    def ok(self) -> bool:
        return not self.rejected

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accepted": list(self.accepted_ids),
            "rejected": [dict(row) for row in self.rejected],
            "all_rejected_codes": sorted({str(row.get("error_code", "")) for row in self.rejected}),
            "ok": self.ok,
        }


def illegal_join_guard(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_group_ids: Iterable[str],
    contract_by_group: Mapping[str, ComparisonContract],
    level: str = "ranking",
    opt_in_conditional_groups: Iterable[str] = (),
) -> JoinDecision:
    """Reject illegal rows before they can reach a ranking/Pareto query.

    ``level`` is ``ranking`` (only ``COMPARABLE`` passes) or ``conditional``
    (``CONDITIONAL`` groups that explicitly opted in also pass).
    """
    allowed = set(allowed_group_ids)
    opted_in = set(opt_in_conditional_groups)
    accepted: List[str] = []
    rejected: List[Mapping[str, Any]] = []
    for row in rows:
        row_id = str(row.get("cell_id", row.get("result_id", "")))
        group_id = str(row.get("comparison_group_id", row.get("comparison_id", "")))
        verdict = str(row.get("comparability_status", row.get("verdict", "")))
        error = ""
        if group_id not in allowed:
            error = "JOIN_UNKNOWN_COMPARISON_GROUP"
        elif group_id in contract_by_group:
            contract = contract_by_group[group_id]
            expected_hash = contract.sha256()
            seen_hash = str(row.get("contract_sha256", ""))
            if seen_hash and seen_hash != expected_hash:
                error = "JOIN_CONTRACT_VERSION_CHANGED"
            elif str(contract.workload.layer) not in ("", str(row.get("layer", ""))):
                error = "JOIN_LAYER_MIXED"
        if not error:
            status = str(row.get("status", ""))
            quality = str(row.get("quality_status", ""))
            if quality.upper().startswith("QUALITY") or quality == "FAIL":
                error = "JOIN_QUALITY_FAILED"
            elif status in MISSINGNESS_CODES or row.get("missing_reason"):
                error = "JOIN_MISSING_STATE"
            elif verdict == "":
                error = "JOIN_INSUFFICIENT_EVIDENCE"
            elif verdict == NOT_COMPARABLE:
                error = "JOIN_NOT_COMPARABLE"
            elif verdict == INSUFFICIENT_EVIDENCE:
                error = "JOIN_INSUFFICIENT_EVIDENCE"
            elif verdict == CONDITIONAL and group_id not in opted_in:
                error = "JOIN_CONDITIONAL_NOT_OPTED_IN"
            elif level == "ranking" and verdict == CONDITIONAL:
                error = "JOIN_CONDITIONAL_NOT_OPTED_IN"
            elif not row.get("actual_backend"):
                error = "JOIN_ACTUAL_BACKEND_UNKNOWN"
        if error:
            rejected.append({"row_id": row_id, "comparison_group_id": group_id, "error_code": error})
        else:
            accepted.append(row_id)
    return JoinDecision(accepted_ids=tuple(accepted), rejected=tuple(rejected))


def resolution_lossless(raw_rows: Sequence[Mapping[str, Any]], normalized_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """A conditional normalization must keep raw and normalized side by side."""
    raw_ids = {str(row.get("row_id", "")) for row in raw_rows}
    missing_raw = [
        str(row.get("row_id", ""))
        for row in normalized_rows
        if str(row.get("source_row_id", "")) not in raw_ids
    ]
    return {
        "raw_rows": len(raw_rows),
        "normalized_rows": len(normalized_rows),
        "normalized_without_raw": sorted(missing_raw),
        "lossless": not missing_raw,
    }


# ── reviewer records and suite manifests ──────────────────────────────────


@dataclass
class ReviewerRecord:
    """One independent review of a comparison group (§9 step 28)."""

    review_id: str
    comparison_group_id: str
    reviewer: str
    reviewed_at: str
    verdict: str
    disagreements: Tuple[str, ...] = ()
    evidence_refs: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.verdict not in COMPARABILITY_STATES:
            problems.append(f"unknown reviewer verdict {self.verdict!r}")
        if not self.reviewer:
            problems.append("a review needs a reviewer id")
        if self.disagreements and not self.evidence_refs:
            problems.append("a disagreement must cite the field-level evidence")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "comparison_group_id": self.comparison_group_id,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "verdict": self.verdict,
            "disagreements": list(self.disagreements),
            "evidence_refs": list(self.evidence_refs),
        }


def reviewer_record(
    comparison_group_id: str,
    *,
    reviewer: str,
    verdict: str,
    reviewed_at: str = "",
    disagreements: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
) -> ReviewerRecord:
    return ReviewerRecord(
        review_id=stable_id("rev", {"group": comparison_group_id, "reviewer": reviewer, "verdict": verdict}),
        comparison_group_id=comparison_group_id,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        verdict=verdict,
        disagreements=tuple(disagreements),
        evidence_refs=tuple(evidence_refs),
    )


def reviewer_agreement(records: Sequence[ReviewerRecord]) -> Dict[str, Any]:
    """Agreement per group; disagreements are kept, never averaged away."""
    by_group: Dict[str, List[ReviewerRecord]] = {}
    problems: List[str] = []
    for record in records:
        problems.extend(f"{record.review_id}: {item}" for item in record.validate())
        by_group.setdefault(record.comparison_group_id, []).append(record)
    rows = []
    for group, items in sorted(by_group.items()):
        verdicts = {item.verdict for item in items}
        disagreements = sorted({d for item in items for d in item.disagreements})
        rows.append(
            {
                "comparison_group_id": group,
                "reviewers": sorted(item.reviewer for item in items),
                "independent_reviews": len(items),
                "agreement": len(verdicts) == 1,
                "verdicts": sorted(verdicts),
                "disagreements": disagreements,
                "resolved_by": "rule clarification or smaller group (never by relaxing invariants)",
            }
        )
    return {
        "rows": rows,
        "groups": len(rows),
        "fully_agreed": sum(1 for row in rows if row["agreement"]),
        "problems": problems,
        "ok": not problems,
    }


@dataclass
class ComparisonGroup:
    """A frozen group of cells that may be analysed together."""

    comparison_group_id: str
    comparison_id: str
    layer: str
    scenario: str
    member_candidate_ids: Tuple[str, ...]
    contract_sha256: str
    verdict: str
    allowed_differences: Tuple[str, ...] = ()
    forbidden_claims: Tuple[str, ...] = ()
    downstream_scope: str = "ranking"
    suite_manifest_id: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.layer not in LAYERS:
            problems.append(f"unknown layer {self.layer!r}")
        if self.verdict not in COMPARABILITY_STATES:
            problems.append(f"unknown group verdict {self.verdict!r}")
        if len(self.member_candidate_ids) < 2 and self.verdict == COMPARABLE:
            problems.append("a comparable group needs at least two members")
        if len(self.contract_sha256) != 64:
            problems.append("the group must cite the contract sha256")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "comparison_group_id": self.comparison_group_id,
            "comparison_id": self.comparison_id,
            "layer": self.layer,
            "scenario": self.scenario,
            "member_candidate_ids": list(self.member_candidate_ids),
            "contract_sha256": self.contract_sha256,
            "verdict": self.verdict,
            "allowed_differences": list(self.allowed_differences),
            "forbidden_claims": list(self.forbidden_claims),
            "downstream_scope": self.downstream_scope,
            "suite_manifest_id": self.suite_manifest_id,
        }


def build_comparison_groups(
    audits: Sequence[CellAudit],
    contract: ComparisonContract,
    *,
    group_id: Optional[str] = None,
) -> Tuple[ComparisonGroup, ...]:
    """Group the audited cells by verdict and freeze the admissible members."""
    gid = group_id or f"grp_{contract.comparison_id}"
    verdicts = {cell.verdict.verdict for cell in audits}
    rows: List[ComparisonGroup] = []
    for verdict in sorted(verdicts):
        members = tuple(sorted(cell.candidate_id for cell in audits if cell.verdict.verdict == verdict))
        rows.append(
            ComparisonGroup(
                comparison_group_id=f"{gid}:{verdict}",
                comparison_id=contract.comparison_id,
                layer=contract.workload.layer,
                scenario=contract.workload.scenario,
                member_candidate_ids=members,
                contract_sha256=contract.sha256(),
                verdict=verdict,
                allowed_differences=tuple(contract.allowed_differences),
                forbidden_claims=(
                    ("no ranking", "no cost/energy objective")
                    if verdict != COMPARABLE
                    else ("no absolute hardware ranking outside the frozen contract",)
                ),
                downstream_scope="ranking" if verdict == COMPARABLE else "diagnostic_only",
                suite_manifest_id="",
            )
        )
    return tuple(rows)


def suite_manifest(
    group: ComparisonGroup,
    contract: ComparisonContract,
    member_cells: Sequence[str],
) -> Dict[str, Any]:
    """Freeze the immutable suite manifest that E12-03 consumes."""
    payload = {
        "comparison_group_id": group.comparison_group_id,
        "contract_sha256": contract.sha256(),
        "member_cells": sorted(str(cell) for cell in member_cells),
        "policy_version": POLICY_VERSION,
    }
    manifest_id = stable_id("suite", payload)
    return {
        "suite_manifest_id": manifest_id,
        "comparison_group_id": group.comparison_group_id,
        "contract_sha256": payload["contract_sha256"],
        "member_cells": payload["member_cells"],
        "frozen_at": "",
        "invalidated_by": "",
        "manifest_hash": sha256_text(str(payload)),
    }


def suite_manifest_valid(
    manifest: Mapping[str, Any],
    contract: ComparisonContract,
    member_cells: Sequence[str],
) -> bool:
    """A changed contract or membership invalidates the manifest."""
    return (
        str(manifest.get("contract_sha256", "")) == contract.sha256()
        and sorted(str(cell) for cell in member_cells) == list(manifest.get("member_cells", []))
    )


def estimate_verdict_histogram(verdicts: Sequence[ComparabilityVerdict]) -> Dict[str, int]:
    counts: Dict[str, int] = {state: 0 for state in COMPARABILITY_STATES}
    for verdict in verdicts:
        counts[verdict.verdict] = counts.get(verdict.verdict, 0) + 1
    return dict(sorted(counts.items()))


def audit_dimensions() -> Tuple[str, ...]:
    """Re-export the 16 audit dimensions (kept importable from this module)."""
    return tuple(AUDIT_DIMENSIONS)


def reason_codes(dimension: str) -> Tuple[str, ...]:
    return tuple(reason_codes_for(dimension))


# ── smoke self-check (never an experiment) ────────────────────────────────


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only check that the interfaces are callable and fail closed."""
    from hqsb.evaluation.contracts import default_contract

    contract = default_contract()
    policy = FieldClassificationPolicy()
    cells = [
        {
            "cell_id": "c1",
            "comparison_id": contract.comparison_id,
            "candidate_id": "cand_a",
            "layer": contract.workload.layer,
            "scenario": contract.workload.scenario,
            "workload_spec_id": contract.workload.workload_spec_id,
            "measurement_contract_id": "mc1",
            "quality_gate_id": contract.quality.gate_id,
            "actual_backend": "cuda",
        },
        {
            "cell_id": "c2",
            "comparison_id": contract.comparison_id,
            "candidate_id": "cand_b",
            "layer": contract.workload.layer,
            "scenario": contract.workload.scenario,
            "workload_spec_id": contract.workload.workload_spec_id,
            "measurement_contract_id": "mc1",
            "quality_gate_id": contract.quality.gate_id,
            "actual_backend": "",
        },
    ]
    audit = run_audit(cells, contract, policy=policy)
    join = illegal_join_guard(
        [
            {
                "cell_id": "c1",
                "comparison_group_id": f"grp_{contract.comparison_id}",
                "comparability_status": COMPARABLE,
                "quality_status": "pass",
                "actual_backend": "cuda",
                "layer": contract.workload.layer,
                "contract_sha256": contract.sha256(),
            },
            {
                "cell_id": "c2",
                "comparison_group_id": f"grp_{contract.comparison_id}",
                "comparability_status": INSUFFICIENT_EVIDENCE,
                "quality_status": "pass",
                "actual_backend": "",
                "layer": contract.workload.layer,
                "contract_sha256": contract.sha256(),
            },
        ],
        allowed_group_ids={f"grp_{contract.comparison_id}"},
        contract_by_group={f"grp_{contract.comparison_id}": contract},
    )
    cases = negative_case_matrix()
    return {
        "status": "smoke",
        "claim_allowed": False,
        "negative_cases": len(cases),
        "audit_cells": len(audit.cells),
        "verdict_histogram": audit.verdict_histogram(),
        "actual_backend_unknown_cell": "ACTUAL_BACKEND_UNKNOWN" in audit.unresolved_dependencies(),
        "illegal_join_rejected_codes": sorted(
            {str(row.get("error_code")) for row in join.rejected}
        ),
        "clean_row_accepted": "c1" in join.accepted,
        "unit_conversion_keeps_raw": convert_unit(1.5, "ms", "ns")["raw_value"] == 1.5,
        "policy_default_is_invariant": FieldClassificationPolicy().classify("unclassified.field")
        == FIELD_INVARIANT,
    }


PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结评估问题（四层 estimand）", ("candidates:estimand_registry", "candidates:Estimand", "layers:LAYERS")),
    (2, "冻结候选 universe", ("candidates:CandidateMatrix", "candidates:CandidateIdentity")),
    (3, "为候选生成稳定身份", ("candidates:CandidateIdentity.candidate_id", "identity:stable_id")),
    (4, "盘点上游 evidence", ("candidates:UpstreamEvidenceRow", "candidates:inventory_for_candidates", "campaign:UpstreamEvidence")),
    (5, "冻结 ModelArtifact", ("candidates:CANDIDATE_IDENTITY_FIELDS", "contracts:SemanticIdentity")),
    (6, "冻结 token 语义", ("layers:TokenAccounting", "contracts:SemanticIdentity")),
    (7, "冻结生成语义", ("contracts:SemanticIdentity", "records:MISSINGNESS_CODES")),
    (8, "定义 precision contract", ("contracts:QualityClause", "contracts:ComparisonContract")),
    (9, "绑定 QuantArtifact", ("candidates:CandidateIdentity", "records:MISSING_NOT_APPLICABLE_CAPABILITY")),
    (10, "冻结共同 correctness gate", ("contracts:ComparisonContract", "contracts:QualityClause")),
    (11, "冻结共同 quality gate", ("contracts:QualityClause", "comparability:quality_dependency_audit")),
    (12, "建立 workload registry", ("layers:SCENARIOS", "contracts:WorkloadClause")),
    (13, "定义四层 measurement boundary", ("layers:MEASUREMENT_BOUNDARIES", "layers:MeasurementBoundary")),
    (14, "冻结 token/request accounting", ("layers:TokenAccounting", "layers:TOKEN_ACCOUNTING_FIELDS")),
    (15, "冻结 warmup/cooldown/cache policy", ("contracts:TimingClause", "layers:MEASUREMENT_BOUNDARIES")),
    (16, "冻结 backend 证明", ("comparability:actual_backend_audit", "telemetry:S12ResultFields")),
    (17, "冻结服务负载合同", ("layers:LOAD_MODES", "layers:SloSpec")),
    (18, "冻结分布式合同", ("layers:SCENARIOS", "layers:ESTIMANDS")),
    (19, "冻结系统状态合同", ("contracts:FIELD_CATALOG", "records:TABLE_SCHEMAS")),
    (20, "冻结统计合同", ("contracts:StatisticsClause", "repeatability:ReplicationHierarchy")),
    (21, "建立字段分类", ("contracts:FIELD_CLASSES", "comparability:FieldClassificationPolicy")),
    (22, "生成两两或组内 diff", ("contracts:diff_contracts", "contracts:FieldDiff")),
    (23, "执行单位审计", ("comparability:unit_audit", "comparability:convert_unit")),
    (24, "执行边界审计", ("comparability:boundary_audit", "layers:MeasurementBoundary")),
    (25, "执行 quality 依赖审计", ("comparability:quality_dependency_audit", "records:MISSING_QUALITY_GATE_FAILED")),
    (26, "执行 actual-backend 审计", ("comparability:actual_backend_audit", "telemetry:project_c6")),
    (27, "生成初步机器裁决", ("comparability:decide_comparability", "comparability:ComparabilityVerdict")),
    (28, "独立人工复核", ("comparability:reviewer_record", "comparability:ReviewerRecord")),
    (29, "裁决分歧", ("comparability:reviewer_agreement", "contracts:NormalizationFormula")),
    (30, "注入协议负例", ("comparability:negative_case_matrix", "comparability:NEGATIVE_CASE_KINDS")),
    (31, "验证非法 join 防护", ("comparability:illegal_join_guard", "comparability:INVALID_JOIN_ERROR_CODES")),
    (32, "验证条件比较", ("comparability:resolution_lossless", "contracts:NormalizationFormula")),
    (33, "生成 comparison groups", ("comparability:build_comparison_groups", "comparability:ComparisonGroup")),
    (34, "生成缺口计划", ("comparability:action_items_from_gaps", "records:TABLE_SCHEMAS")),
    (35, "锁定 E12-03 输入", ("comparability:suite_manifest", "comparability:suite_manifest_valid")),
    (36, "形成验收与限制报告", ("comparability:AuditResult", "campaign:AcceptanceDecision")),
)
