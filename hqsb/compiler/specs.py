"""Frozen configuration documents for S11 with strict key validation.

The twelve ``configs/compiler/*.yaml`` documents are the *single source of
frozen policy values* the reports cite (capture axes, guard taxonomy, pattern
contracts, capability reasons, lowering policies, budgets, feature groups,
cache/key/invalidation matrices, portable-stack roles, AI-gate classes and
tolerances).  ``audit_documents`` compares every document against the code
constants field by field, so a document that drifts from the implementation
fails a test instead of silently disagreeing with the report.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml

from hqsb.core.errors import ConfigError

KIND_CAPTURE_SPEC = "hqsb.compiler.capture_spec"
KIND_GUARD_SPEC = "hqsb.compiler.guard_spec"
KIND_PATTERN_SPEC = "hqsb.compiler.pattern_spec"
KIND_TARGET_SPEC = "hqsb.compiler.target_spec"
KIND_LOWERING_SPEC = "hqsb.compiler.lowering_spec"
KIND_AUTOTUNE_SPEC = "hqsb.compiler.autotune_spec"
KIND_COSTMODEL_SPEC = "hqsb.compiler.costmodel_spec"
KIND_CACHE_SPEC = "hqsb.compiler.cache_spec"
KIND_PORTABLE_SPEC = "hqsb.compiler.portable_spec"
KIND_AIGATE_SPEC = "hqsb.compiler.aigate_spec"
KIND_TOLERANCE_SPEC = "hqsb.compiler.tolerance_spec"
KIND_EXPERIMENT_SPEC = "hqsb.compiler.experiment_spec"

KINDS: Tuple[str, ...] = (
    KIND_CAPTURE_SPEC,
    KIND_GUARD_SPEC,
    KIND_PATTERN_SPEC,
    KIND_TARGET_SPEC,
    KIND_LOWERING_SPEC,
    KIND_AUTOTUNE_SPEC,
    KIND_COSTMODEL_SPEC,
    KIND_CACHE_SPEC,
    KIND_PORTABLE_SPEC,
    KIND_AIGATE_SPEC,
    KIND_TOLERANCE_SPEC,
    KIND_EXPERIMENT_SPEC,
)

COMMON_KEYS: Tuple[str, ...] = ("kind", "name", "description", "notes", "schema_version")

ALLOWED_KEYS: Mapping[str, Tuple[str, ...]] = {
    KIND_CAPTURE_SPEC: COMMON_KEYS
    + (
        "capture_modes",
        "regions",
        "workloads",
        "layout_cases",
        "backend_modes",
        "repeat_cases",
        "break_reason_codes",
        "coverage_metrics",
        "metadata_dimensions",
        "trace_phases",
        "missing_reasons",
    ),
    KIND_GUARD_SPEC: COMMON_KEYS
    + (
        "categories",
        "actions",
        "strategies",
        "variant_limit_behaviours",
        "compile_triggers",
        "semantic_categories",
    ),
    KIND_PATTERN_SPEC: COMMON_KEYS
    + (
        "patterns",
        "pattern_version",
        "predicate_order",
        "match_classes",
        "mutation_axes",
        "allowed_eps",
        "decomposition_cast_ops",
        "role_order",
        "forbidden_relaxations",
    ),
    KIND_TARGET_SPEC: COMMON_KEYS
    + (
        "device_kinds",
        "capability_reasons",
        "export_layers",
        "required_snapshot_fields",
        "arch_features",
        "required_snapshot_rules",
    ),
    KIND_LOWERING_SPEC: COMMON_KEYS
    + (
        "execution_kinds",
        "policies",
        "evidence_kinds",
        "min_evidence_kinds",
        "pre_launch_checks",
        "runtime_error_policies",
        "correctness_order",
        "performance_order",
        "injectable_failures",
    ),
    KIND_AUTOTUNE_SPEC: COMMON_KEYS
    + (
        "constraint_categories",
        "static_reject_reasons",
        "split_strategies",
        "holdout_kinds",
        "trial_splits",
        "budget_names",
        "tuning_db_key_fields",
        "knob_types",
        "rules",
    ),
    KIND_COSTMODEL_SPEC: COMMON_KEYS
    + (
        "feature_groups",
        "availability",
        "leakage_forbidden",
        "critical_features",
        "baselines",
        "strong_baselines",
        "split_strategies",
        "primary_splits",
        "fallback_actions",
        "policy_steps",
        "catastrophic_threshold",
        "rules",
    ),
    KIND_CACHE_SPEC: COMMON_KEYS
    + (
        "layers",
        "key_fields",
        "non_key_fields",
        "entry_states",
        "read_reject_reasons",
        "invalidation_matrix",
        "corruption_cases",
        "timing_boundaries",
        "event_kinds",
        "eviction_strategies",
    ),
    KIND_PORTABLE_SPEC: COMMON_KEYS
    + (
        "stacks",
        "legality_modes",
        "legal_statuses",
        "mapping_dimensions",
        "unmapped_policies",
        "schedule_steps",
        "bridge_kinds",
        "role_dimensions",
        "adoption_options",
        "scope_ceiling",
    ),
    KIND_AIGATE_SPEC: COMMON_KEYS
    + (
        "candidate_classes",
        "expected_gate_by_class",
        "gate_order",
        "banned_imports",
        "banned_patterns",
        "admission_required_fields",
        "fast_p_thresholds",
        "claim_boundaries",
    ),
    KIND_TOLERANCE_SPEC: COMMON_KEYS
    + (
        "policy_id",
        "policy_owner",
        "dtype_tolerances",
        "rules",
        "forbidden_relaxations",
    ),
    KIND_EXPERIMENT_SPEC: COMMON_KEYS
    + ("experiments", "run_root", "run_layout", "statuses", "raw_tables", "rules"),
}


def _require(condition: bool, message: str, *, field_name: str = "") -> None:
    if not condition:
        raise ConfigError(message, details={"field": field_name} if field_name else {})


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
    _require(bool(document.get("name")), f"{path}: name is required", field_name="name")
    return document


def _kind_file(directory: str, kind: str) -> str:
    return os.path.join(directory, kind.split(".")[-1] + ".yaml")


@dataclass
class CompilerSpecs:
    """All compiled-in spec documents, loaded from one directory."""

    documents: Mapping[str, Dict[str, Any]] = field(default_factory=dict)
    source_dir: str = ""

    @classmethod
    def load(cls, directory: str) -> "CompilerSpecs":
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
                raise ConfigError(
                    f"{path}: kind {document.get('kind')!r} != expected {kind!r}"
                )
            documents[kind] = document
        if missing:
            raise ConfigError(
                f"missing spec documents: {missing}", details={"field": "specs_dir"}
            )
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


def _compare_sets(
    kind: str, field_name: str, document: Mapping[str, Any], expected: Sequence[str]
) -> Dict[str, Any]:
    actual = document.get(field_name)
    if not isinstance(actual, list):
        return _audit(kind, False, [f"{field_name}: expected a list"])
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    problems = []
    if missing:
        problems.append(f"{field_name}: missing {missing}")
    if extra:
        problems.append(f"{field_name}: undocumented {extra}")
    return _audit(kind, not problems, problems)


def audit_documents(documents: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Field-by-field audit of every document against the code constants."""
    from hqsb.compiler import (
        aigate,
        autotune,
        cache,
        capture,
        costmodel,
        guards,
        lowering,
        pattern_library,
        portable,
        targets,
        telemetry,
    )

    reports: List[Dict[str, Any]] = []

    doc = documents[KIND_CAPTURE_SPEC]
    problems: List[str] = []
    for field_name, expected in (
        ("capture_modes", capture.CAPTURE_MODES),
        ("regions", capture.REGIONS),
        ("workloads", capture.WORKLOADS),
        ("layout_cases", capture.LAYOUT_CASES),
        ("backend_modes", capture.BACKEND_MODES),
        ("repeat_cases", capture.REPEAT_CASES),
        ("break_reason_codes", tuple(capture.BREAK_REASON_CATALOG)),
        ("coverage_metrics", capture.COVERAGE_METRICS),
        ("metadata_dimensions", capture.METADATA_DIMENSIONS),
        ("trace_phases", capture.TRACE_PHASES),
        ("missing_reasons", capture.MISSING_REASONS),
    ):
        report = _compare_sets(KIND_CAPTURE_SPEC, field_name, doc, expected)
        problems.extend(report["problems"])
    reports.append(_audit(KIND_CAPTURE_SPEC, not problems, problems))

    doc = documents[KIND_GUARD_SPEC]
    problems = []
    for field_name, expected in (
        ("categories", guards.GUARD_CATEGORIES),
        ("actions", guards.DEFAULT_ACTION_BY_CATEGORY),
        ("strategies", guards.STRATEGIES),
        ("variant_limit_behaviours", ("fallback", "error")),
        ("compile_triggers", guards.COMPILE_TRIGGERS),
        ("semantic_categories", guards.SEMANTIC_GUARD_CATEGORIES),
    ):
        if field_name == "actions":
            actual = doc.get(field_name)
            if isinstance(actual, dict):
                missing = sorted(set(guards.GUARD_CATEGORIES) - set(actual))
                extra = sorted(set(actual) - set(guards.GUARD_CATEGORIES))
                problems.extend(
                    [f"actions: missing {missing}"] if missing else []
                )
                problems.extend([f"actions: undocumented {extra}"] if extra else [])
            else:
                problems.append("actions: expected a mapping")
            continue
        report = _compare_sets(KIND_GUARD_SPEC, field_name, doc, expected)
        problems.extend(report["problems"])
    reports.append(_audit(KIND_GUARD_SPEC, not problems, problems))

    doc = documents[KIND_PATTERN_SPEC]
    problems = []
    for field_name, expected in (
        ("patterns", tuple(contract.pattern_id for contract in pattern_library.frozen_contracts())),
        ("predicate_order", pattern_library.PREDICATE_ORDER),
        ("match_classes", tuple(row[0] for row in pattern_library.MATCH_CLASS_TABLE)),
        ("mutation_axes", tuple(row[0] for row in pattern_library.NEAR_MISS_MUTATION_AXES)),
        ("allowed_eps", tuple(str(value) for value in (1e-5, 1e-6, 1e-4))),
        ("decomposition_cast_ops", pattern_library.DECOMPOSITION_VARIANTS),
        ("role_order", pattern_library.ROLE_ORDER_RESIDUAL_RMSNORM),
    ):
        report = _compare_sets(KIND_PATTERN_SPEC, field_name, doc, expected)
        problems.extend(report["problems"])
    if doc.get("pattern_version") != pattern_library.PATTERN_VERSION:
        problems.append("pattern_version mismatch")
    reports.append(_audit(KIND_PATTERN_SPEC, not problems, problems))

    doc = documents[KIND_TARGET_SPEC]
    problems = []
    for field_name, expected in (
        ("device_kinds", targets.BACKEND_KINDS),
        ("capability_reasons", targets.CAPABILITY_REASONS),
        ("required_snapshot_fields", targets.TargetSnapshot.__dataclass_fields__),  # type: ignore[attr-defined]
    ):
        report = _compare_sets(KIND_TARGET_SPEC, field_name, doc, tuple(expected))
        problems.extend(report["problems"])
    reports.append(_audit(KIND_TARGET_SPEC, not problems, problems))

    doc = documents[KIND_LOWERING_SPEC]
    problems = []
    for field_name, expected in (
        ("execution_kinds", lowering.EXECUTION_KINDS),
        ("policies", lowering.POLICIES),
        ("evidence_kinds", lowering.EVIDENCE_KINDS),
        ("pre_launch_checks", lowering.PRE_LAUNCH_CHECKS),
        ("runtime_error_policies", lowering.RUNTIME_ERROR_POLICIES),
        ("correctness_order", lowering.CORRECTNESS_ORDER),
        ("performance_order", lowering.PERFORMANCE_ORDER),
        ("injectable_failures", lowering.INJECTABLE_LOWERING_FAILURES),
    ):
        report = _compare_sets(KIND_LOWERING_SPEC, field_name, doc, expected)
        problems.extend(report["problems"])
    if doc.get("min_evidence_kinds") != lowering.MIN_EVIDENCE_KINDS:
        problems.append("min_evidence_kinds mismatch")
    reports.append(_audit(KIND_LOWERING_SPEC, not problems, problems))

    doc = documents[KIND_AUTOTUNE_SPEC]
    problems = []
    for field_name, expected in (
        ("constraint_categories", autotune.CONSTRAINT_CATEGORIES),
        ("static_reject_reasons", autotune.STATIC_REJECT_REASONS),
        ("split_strategies", autotune.SPLIT_STRATEGIES),
        ("holdout_kinds", autotune.HOLDOUT_KINDS),
        ("trial_splits", autotune.AUTOTUNE_SPLITS),
        ("tuning_db_key_fields", autotune.TUNING_DB_KEY_FIELDS),
        ("knob_types", autotune.KNOB_TYPES),
    ):
        report = _compare_sets(KIND_AUTOTUNE_SPEC, field_name, doc, expected)
        problems.extend(report["problems"])
    budgets = [entry.name for entry in autotune.budget_ladder(space_size=4)]
    report = _compare_sets(KIND_AUTOTUNE_SPEC, "budget_names", doc, budgets)
    problems.extend(report["problems"])
    reports.append(_audit(KIND_AUTOTUNE_SPEC, not problems, problems))

    doc = documents[KIND_COSTMODEL_SPEC]
    problems = []
    for field_name, expected in (
        ("feature_groups", tuple(costmodel.FEATURE_GROUPS)),
        ("availability", costmodel.FEATURE_AVAILABILITY),
        ("leakage_forbidden", costmodel.LEAKAGE_FORBIDDEN),
        ("critical_features", costmodel.CRITICAL_PRE_COMPILE_FEATURES),
        ("baselines", costmodel.BASELINES),
        ("strong_baselines", costmodel.STRONG_BASELINES),
        ("split_strategies", costmodel.SPLIT_STRATEGIES),
        ("primary_splits", costmodel.PRIMARY_SPLITS),
        ("fallback_actions", costmodel.FALLBACK_ACTIONS),
        ("policy_steps", costmodel.POLICY_STEPS),
    ):
        report = _compare_sets(KIND_COSTMODEL_SPEC, field_name, doc, expected)
        problems.extend(report["problems"])
    reports.append(_audit(KIND_COSTMODEL_SPEC, not problems, problems))

    doc = documents[KIND_CACHE_SPEC]
    problems = []
    for field_name, expected in (
        ("layers", cache.CACHE_LAYERS),
        ("key_fields", tuple(cache.KEY_FIELD_REASONS)),
        ("non_key_fields", cache.NON_KEY_FIELDS),
        ("entry_states", cache.ENTRY_STATES),
        ("read_reject_reasons", cache.READ_REJECT_REASONS),
        ("corruption_cases", cache.CORRUPTION_CASES),
        ("timing_boundaries", cache.TIMING_BOUNDARIES),
        ("event_kinds", cache.CACHE_EVENT_KINDS),
        ("eviction_strategies", ("lru", "ttl", "quota")),
    ):
        report = _compare_sets(KIND_CACHE_SPEC, field_name, doc, expected)
        problems.extend(report["problems"])
    matrix = doc.get("invalidation_matrix")
    if isinstance(matrix, list):
        expected_changes = [row[0] for row in cache.INVALIDATION_MATRIX]
        actual_changes = [
            row.get("change") if isinstance(row, dict) else row for row in matrix
        ]
        if sorted(actual_changes) != sorted(expected_changes):
            problems.append("invalidation_matrix does not match the code matrix")
    else:
        problems.append("invalidation_matrix: expected a list")
    reports.append(_audit(KIND_CACHE_SPEC, not problems, problems))

    doc = documents[KIND_PORTABLE_SPEC]
    problems = []
    for field_name, expected in (
        ("stacks", portable.STACKS),
        ("legality_modes", portable.LEGALITY_MODES),
        ("legal_statuses", portable.LEGAL_STATUSES),
        ("unmapped_policies", portable.UNMAPPED_POLICIES),
        ("schedule_steps", portable.SCHEDULE_STEPS),
        ("bridge_kinds", portable.BRIDGE_KINDS),
        ("role_dimensions", portable.ROLE_DIMENSIONS),
        ("adoption_options", portable.ADOPTION_OPTIONS),
        ("mapping_dimensions", tuple(row[0] for row in portable.MAPPING_ROWS)),
    ):
        report = _compare_sets(KIND_PORTABLE_SPEC, field_name, doc, expected)
        problems.extend(report["problems"])
    reports.append(_audit(KIND_PORTABLE_SPEC, not problems, problems))

    doc = documents[KIND_AIGATE_SPEC]
    problems = []
    for field_name, expected in (
        ("candidate_classes", tuple(aigate.CANDIDATE_CLASSES)),
        ("gate_order", aigate.GATE_ORDER),
        ("banned_imports", aigate.BANNED_IMPORTS),
        ("admission_required_fields", aigate.ADMISSION_REQUIRED_FIELDS),
    ):
        report = _compare_sets(KIND_AIGATE_SPEC, field_name, doc, expected)
        problems.extend(report["problems"])
    gate_map = doc.get("expected_gate_by_class")
    if isinstance(gate_map, dict):
        if gate_map != dict(aigate.EXPECTED_GATE_BY_CLASS):
            problems.append("expected_gate_by_class does not match the code map")
    else:
        problems.append("expected_gate_by_class: expected a mapping")
    reports.append(_audit(KIND_AIGATE_SPEC, not problems, problems))

    doc = documents[KIND_TOLERANCE_SPEC]
    problems = []
    if doc.get("policy_id") != "common_s06":
        problems.append("tolerance policy_id must reference the S06 common policy")
    if not doc.get("forbidden_relaxations"):
        problems.append("forbidden_relaxations must be enumerated")
    reports.append(_audit(KIND_TOLERANCE_SPEC, not problems, problems))

    doc = documents[KIND_EXPERIMENT_SPEC]
    problems = []
    expected_experiments = [f"E11-{index:02d}" for index in range(1, 11)]
    report = _compare_sets(KIND_EXPERIMENT_SPEC, "experiments", doc, expected_experiments)
    problems.extend(report["problems"])
    tables = doc.get("raw_tables")
    if tables is not None:
        report = _compare_sets(
            KIND_EXPERIMENT_SPEC, "raw_tables", doc, tuple(telemetry.S11_TABLE_SCHEMAS)
        )
        problems.extend(report["problems"])
    reports.append(_audit(KIND_EXPERIMENT_SPEC, not problems, problems))
    return reports


def audit_all_ok(reports: Sequence[Mapping[str, Any]]) -> bool:
    return all(report["ok"] for report in reports)


def default_spec_dir(root: str) -> str:
    return os.path.join(root, "configs", "compiler")
