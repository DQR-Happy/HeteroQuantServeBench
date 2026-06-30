"""Frozen configuration documents for S12 with strict key validation.

The twelve ``configs/evaluation/*.yaml`` documents are the single source of the
*frozen vocabularies* the reports cite (verdict states, capability levels,
failure taxonomies, boundaries, scenario kinds, …).  ``audit_documents``
compares every document against the code constants field by field, so a document
that drifts from the implementation fails a test instead of silently disagreeing
with the report.

The YAML documents deliberately carry *structure and vocabulary, not numbers*:
price, electricity and measurement values are campaign inputs, never baked into
a frozen document.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml

from hqsb.core.errors import ConfigError

KIND_COMPARABILITY = "hqsb.evaluation.comparability_spec"
KIND_CAPABILITY = "hqsb.evaluation.capability_spec"
KIND_BENCHMARK = "hqsb.evaluation.benchmark_spec"
KIND_REPEATABILITY = "hqsb.evaluation.repeatability_spec"
KIND_ROOFLINE = "hqsb.evaluation.roofline_spec"
KIND_ENERGY = "hqsb.evaluation.energy_spec"
KIND_COST = "hqsb.evaluation.cost_spec"
KIND_PARETO = "hqsb.evaluation.pareto_spec"
KIND_MATURITY = "hqsb.evaluation.maturity_spec"
KIND_LINEAGE = "hqsb.evaluation.lineage_spec"
KIND_CAMPAIGN = "hqsb.evaluation.campaign_spec"
KIND_EXPERIMENT = "hqsb.evaluation.experiment_spec"

KINDS: Tuple[str, ...] = (
    KIND_COMPARABILITY,
    KIND_CAPABILITY,
    KIND_BENCHMARK,
    KIND_REPEATABILITY,
    KIND_ROOFLINE,
    KIND_ENERGY,
    KIND_COST,
    KIND_PARETO,
    KIND_MATURITY,
    KIND_LINEAGE,
    KIND_CAMPAIGN,
    KIND_EXPERIMENT,
)

COMMON_KEYS: Tuple[str, ...] = ("kind", "name", "description", "notes", "schema_version")

ALLOWED_KEYS: Mapping[str, Tuple[str, ...]] = {
    KIND_COMPARABILITY: COMMON_KEYS
    + ("verdict_states", "field_classes", "audit_dimensions", "reason_codes", "negative_cases", "layers"),
    KIND_CAPABILITY: COMMON_KEYS
    + ("capability_levels", "feature_layers", "failure_categories", "invalidation_triggers", "telemetry_fields", "registry_min_features"),
    KIND_BENCHMARK: COMMON_KEYS
    + ("layers", "load_modes", "allowed_normalizations", "forbidden_normalizations", "validator_constraints", "estimators"),
    KIND_REPEATABILITY: COMMON_KEYS
    + ("stability_dimensions", "health_labels", "contamination_labels", "exclusion_decisions", "stability_verdicts"),
    KIND_ROOFLINE: COMMON_KEYS
    + ("peak_evidence_levels", "memory_levels", "bottleneck_classes", "residual_classes", "operation_definitions"),
    KIND_ENERGY: COMMON_KEYS
    + ("energy_boundaries", "window_policies", "meter_semantics", "idle_states", "power_quality_checks"),
    KIND_COST: COMMON_KEYS
    + ("scenario_kinds", "purchase_models", "cloud_component_keys", "owned_component_keys", "density_metrics", "double_count_checks"),
    KIND_PARETO: COMMON_KEYS
    + ("profile_ids", "directions", "feasibility_statuses", "evidence_gate_statuses", "roles", "decision_regression_cases"),
    KIND_MATURITY: COMMON_KEYS
    + ("maturity_dimensions", "rubric_levels", "failure_taxonomy", "tool_coverage_areas", "task_count"),
    KIND_LINEAGE: COMMON_KEYS
    + ("entity_types", "activity_types", "relations", "regeneration_levels", "required_lineage_objects"),
    KIND_CAMPAIGN: COMMON_KEYS
    + ("upstream_evidence_states", "result_classes", "evidence_levels", "missingness_codes"),
    KIND_EXPERIMENT: COMMON_KEYS
    + ("experiments", "statuses", "raw_tables", "run_root", "rules"),
}


def load_yaml_document(path: str) -> Dict[str, Any]:
    """Load one spec document with strict kind/key validation (no silent keys)."""
    if not os.path.isfile(path):
        raise ConfigError(f"spec file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ConfigError(f"{path}: document must be a mapping")
    kind = document.get("kind")
    if kind not in ALLOWED_KEYS:
        raise ConfigError(f"{path}: unknown kind {kind!r}", details={"field": "kind"})
    unknown = sorted(set(document) - set(ALLOWED_KEYS[kind]))
    if unknown:
        raise ConfigError(f"{path}: unknown keys {unknown}", details={"field": "keys"})
    if not document.get("name"):
        raise ConfigError(f"{path}: name is required", details={"field": "name"})
    return document


def _kind_file(directory: str, kind: str) -> str:
    return os.path.join(directory, kind.split(".")[-1] + ".yaml")


@dataclass
class EvaluationSpecs:
    documents: Mapping[str, Dict[str, Any]] = field(default_factory=dict)
    source_dir: str = ""

    @classmethod
    def load(cls, directory: str) -> "EvaluationSpecs":
        if not os.path.isdir(directory):
            raise ConfigError(f"spec directory not found: {directory}")
        documents: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        for kind in KINDS:
            path = _kind_file(directory, kind)
            if not os.path.isfile(path):
                missing.append(kind)
                continue
            document = load_yaml_document(path)
            if document.get("kind") != kind:
                raise ConfigError(f"{path}: kind {document.get('kind')!r} != expected {kind!r}")
            documents[kind] = document
        if missing:
            raise ConfigError(f"missing spec documents: {missing}", details={"field": "specs_dir"})
        return cls(documents=documents, source_dir=directory)

    def document(self, kind: str) -> Dict[str, Any]:
        try:
            return self.documents[kind]
        except KeyError as exc:
            raise ConfigError(f"spec {kind!r} not loaded") from exc

    def audit(self) -> List[Dict[str, Any]]:
        return audit_documents(self.documents)


def _audit(kind: str, ok: bool, problems: Sequence[str]) -> Dict[str, Any]:
    return {"kind": kind, "ok": ok and not problems, "problems": list(problems)}


def _compare_sets(kind: str, field_name: str, document: Mapping[str, Any], expected: Sequence[Any]) -> Dict[str, Any]:
    actual = document.get(field_name)
    if not isinstance(actual, list):
        return _audit(kind, False, [f"{field_name}: expected a list"])
    actual_values = [str(value) for value in actual]
    expected_values = [str(value) for value in expected]
    missing = sorted(set(expected_values) - set(actual_values))
    extra = sorted(set(actual_values) - set(expected_values))
    problems = []
    if missing:
        problems.append(f"{field_name}: missing {missing}")
    if extra:
        problems.append(f"{field_name}: undocumented {extra}")
    return _audit(kind, not problems, problems)


def audit_documents(documents: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Field-by-field audit of every document against the code constants."""
    from hqsb.evaluation import (
        benchmark,
        capability,
        comparability,
        cost,
        energy,
        layers,
        lineage,
        maturity,
        pareto,
        platform,
        records,
        repeatability,
        roofline,
        telemetry,
    )
    from hqsb.evaluation import contracts

    reports: List[Dict[str, Any]] = []

    doc = documents[KIND_COMPARABILITY]
    problems: List[str] = []
    for field_name, expected in (
        ("verdict_states", records.COMPARABILITY_STATES),
        ("field_classes", contracts.FIELD_CLASSES),
        ("audit_dimensions", contracts.AUDIT_DIMENSIONS),
        ("reason_codes", contracts.ALL_REASON_CODES),
        ("negative_cases", tuple(comparability.NEGATIVE_CASE_KINDS)),
        ("layers", layers.LAYERS),
    ):
        report = _compare_sets(KIND_COMPARABILITY, field_name, doc, expected)
        problems.extend(report["problems"])
    reports.append(_audit(KIND_COMPARABILITY, not problems, problems))

    doc = documents[KIND_CAPABILITY]
    problems = []
    for field_name, expected in (
        ("capability_levels", capability.CAPABILITY_LEVELS),
        ("feature_layers", capability.FEATURE_LAYERS),
        ("failure_categories", capability.PROBE_FAILURE_CATEGORIES),
        ("invalidation_triggers", capability.INVALIDATION_TRIGGERS),
        ("telemetry_fields", platform.CANONICAL_TELEMETRY_FIELDS),
    ):
        report = _compare_sets(KIND_CAPABILITY, field_name, doc, expected)
        problems.extend(report["problems"])
    if int(doc.get("registry_min_features", 0)) < 40:
        problems.append("registry_min_features must be at least 40")
    reports.append(_audit(KIND_CAPABILITY, not problems, problems))

    doc = documents[KIND_BENCHMARK]
    problems = []
    for field_name, expected in (
        ("layers", layers.LAYERS),
        ("load_modes", layers.LOAD_MODES),
        ("allowed_normalizations", layers.ALLOWED_NORMALIZATIONS),
        ("forbidden_normalizations", layers.FORBIDDEN_NORMALIZATIONS),
        ("validator_constraints", tuple(row["constraint_id"] for row in benchmark.VALIDATOR_CONSTRAINTS)),
        ("estimators", benchmark.ESTIMATORS),
    ):
        report = _compare_sets(KIND_BENCHMARK, field_name, doc, expected)
        problems.extend(report["problems"])
    reports.append(_audit(KIND_BENCHMARK, not problems, problems))

    doc = documents[KIND_REPEATABILITY]
    problems = []
    for field_name, expected in (
        ("stability_dimensions", repeatability.STABILITY_DIMENSIONS),
        ("health_labels", repeatability.HEALTH_LABELS),
        ("contamination_labels", repeatability.CONTAMINATION_LABELS),
        ("exclusion_decisions", repeatability.EXCLUSION_DECISIONS),
        ("stability_verdicts", repeatability.STABILITY_VERDICTS),
    ):
        report = _compare_sets(KIND_REPEATABILITY, field_name, doc, expected)
        problems.extend(report["problems"])
    reports.append(_audit(KIND_REPEATABILITY, not problems, problems))

    doc = documents[KIND_ROOFLINE]
    problems = []
    for field_name, expected in (
        ("peak_evidence_levels", roofline.PEAK_EVIDENCE_LEVELS),
        ("memory_levels", roofline.MEMORY_LEVELS),
        ("bottleneck_classes", roofline.BOTTLENECK_CLASSES),
        ("residual_classes", roofline.RESIDUAL_CLASSES),
        ("operation_definitions", tuple(row.semantic_op for row in roofline.OPERATION_DEFINITIONS)),
    ):
        report = _compare_sets(KIND_ROOFLINE, field_name, doc, expected)
        problems.extend(report["problems"])
    reports.append(_audit(KIND_ROOFLINE, not problems, problems))

    doc = documents[KIND_ENERGY]
    problems = []
    for field_name, expected in (
        ("energy_boundaries", energy.ENERGY_BOUNDARIES),
        ("window_policies", energy.WINDOW_POLICIES),
        ("meter_semantics", energy.METER_SEMANTICS),
        ("idle_states", energy.IDLE_STATES),
        ("power_quality_checks", energy.POWER_QUALITY_CHECKS),
    ):
        report = _compare_sets(KIND_ENERGY, field_name, doc, expected)
        problems.extend(report["problems"])
    reports.append(_audit(KIND_ENERGY, not problems, problems))

    doc = documents[KIND_COST]
    problems = []
    for field_name, expected in (
        ("scenario_kinds", cost.SCENARIO_KINDS),
        ("purchase_models", cost.PURCHASE_MODELS),
        ("cloud_component_keys", cost.CLOUD_COMPONENT_KEYS),
        ("owned_component_keys", cost.OWNED_COMPONENT_KEYS),
        ("density_metrics", cost.DENSITY_METRICS),
        ("double_count_checks", cost.DOUBLE_COUNT_CHECKS),
    ):
        report = _compare_sets(KIND_COST, field_name, doc, expected)
        problems.extend(report["problems"])
    reports.append(_audit(KIND_COST, not problems, problems))

    doc = documents[KIND_PARETO]
    problems = []
    for field_name, expected in (
        ("profile_ids", pareto.PROFILE_IDS),
        ("directions", pareto.DIRECTIONS),
        ("feasibility_statuses", pareto.FEASIBILITY_STATUSES),
        ("evidence_gate_statuses", pareto.EVIDENCE_GATE_STATUSES),
        ("roles", pareto.ROLES),
        ("decision_regression_cases", pareto.DECISION_REGRESSION_CASES),
    ):
        report = _compare_sets(KIND_PARETO, field_name, doc, expected)
        problems.extend(report["problems"])
    reports.append(_audit(KIND_PARETO, not problems, problems))

    doc = documents[KIND_MATURITY]
    problems = []
    for field_name, expected in (
        ("maturity_dimensions", maturity.MATURITY_DIMENSIONS),
        ("rubric_levels", tuple(str(level) for level in maturity.RUBRIC_LEVELS)),
        ("failure_taxonomy", maturity.FAILURE_TAXONOMY),
        ("tool_coverage_areas", maturity.TOOL_COVERAGE_AREAS),
    ):
        report = _compare_sets(KIND_MATURITY, field_name, doc, expected)
        problems.extend(report["problems"])
    if int(doc.get("task_count", 0)) != len(maturity.STANDARD_TASKS):
        problems.append(f"task_count must be {len(maturity.STANDARD_TASKS)}")
    reports.append(_audit(KIND_MATURITY, not problems, problems))

    doc = documents[KIND_LINEAGE]
    problems = []
    for field_name, expected in (
        ("entity_types", lineage.ENTITY_TYPES),
        ("activity_types", lineage.ACTIVITY_TYPES),
        ("relations", lineage.RELATIONS),
        ("regeneration_levels", records.REGENERATION_LEVELS),
        ("required_lineage_objects", lineage.REQUIRED_LINEAGE_OBJECTS),
    ):
        report = _compare_sets(KIND_LINEAGE, field_name, doc, expected)
        problems.extend(report["problems"])
    reports.append(_audit(KIND_LINEAGE, not problems, problems))

    doc = documents[KIND_CAMPAIGN]
    problems = []
    for field_name, expected in (
        ("upstream_evidence_states", records.UPSTREAM_EVIDENCE_STATES),
        ("result_classes", records.RESULT_CLASSES),
        ("evidence_levels", records.EVIDENCE_LEVELS),
        ("missingness_codes", records.MISSINGNESS_CODES),
    ):
        report = _compare_sets(KIND_CAMPAIGN, field_name, doc, expected)
        problems.extend(report["problems"])
    reports.append(_audit(KIND_CAMPAIGN, not problems, problems))

    doc = documents[KIND_EXPERIMENT]
    problems = []
    expected_experiments = tuple(f"E12-{index:02d}" for index in range(1, 11))
    report = _compare_sets(KIND_EXPERIMENT, "experiments", doc, expected_experiments)
    problems.extend(report["problems"])
    report = _compare_sets(KIND_EXPERIMENT, "statuses", doc, records.PROTOCOL_STATUSES)
    problems.extend(report["problems"])
    tables = doc.get("raw_tables")
    if tables is not None:
        report = _compare_sets(KIND_EXPERIMENT, "raw_tables", doc, tuple(telemetry.S12_TABLE_SCHEMAS))
        problems.extend(report["problems"])
    reports.append(_audit(KIND_EXPERIMENT, not problems, problems))

    return reports


def audit_all_ok(reports: Sequence[Mapping[str, Any]]) -> bool:
    return all(report["ok"] for report in reports)


def default_spec_dir(root: str) -> str:
    return os.path.join(root, "configs", "evaluation")
