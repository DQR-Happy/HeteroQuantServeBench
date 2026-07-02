"""Frozen configuration documents for S13 with strict key validation.

The ``configs/infra/*.yaml`` documents are the single source of the *frozen
vocabularies* an S13 report cites (release statuses, reproducibility levels, clean
levels, sharing modes, lifecycle states, admission policies, autoscaling policies,
observability layers, fault taxonomy, canary gates, tenant invariants, tables).
:func:`audit_documents` compares every document against the code constants field by
field, so a document that drifts from the implementation fails a test instead of
silently disagreeing with the report.

The documents carry *structure and vocabulary, not measurements*: thresholds,
traffic fractions, replicas, cost values and durations are campaign inputs, never
baked into a frozen document.  Where a template needs an illustrative value it is
marked ``POLICY_DEFAULT_UNVERIFIED`` and the marker is asserted by a test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml

from hqsb.core.errors import ConfigError

KIND_RELEASE_IDENTITY = "hqsb.infra.release_identity_spec"
KIND_SUPPLY_CHAIN = "hqsb.infra.supply_chain_spec"
KIND_DEPLOYMENT = "hqsb.infra.deployment_spec"
KIND_SCHEDULING = "hqsb.infra.scheduling_spec"
KIND_ARTIFACT = "hqsb.infra.artifact_spec"
KIND_LIFECYCLE = "hqsb.infra.lifecycle_spec"
KIND_CAPACITY = "hqsb.infra.capacity_admission_spec"
KIND_AUTOSCALING = "hqsb.infra.autoscaling_spec"
KIND_OBSERVABILITY = "hqsb.infra.observability_spec"
KIND_FAULT = "hqsb.infra.fault_spec"
KIND_CANARY = "hqsb.infra.canary_spec"
KIND_SECURITY = "hqsb.infra.security_spec"
KIND_EXPERIMENT = "hqsb.infra.experiment_spec"

KINDS: Tuple[str, ...] = (
    KIND_RELEASE_IDENTITY,
    KIND_SUPPLY_CHAIN,
    KIND_DEPLOYMENT,
    KIND_SCHEDULING,
    KIND_ARTIFACT,
    KIND_LIFECYCLE,
    KIND_CAPACITY,
    KIND_AUTOSCALING,
    KIND_OBSERVABILITY,
    KIND_FAULT,
    KIND_CANARY,
    KIND_SECURITY,
    KIND_EXPERIMENT,
)

COMMON_KEYS: Tuple[str, ...] = ("kind", "name", "description", "notes", "schema_version", "status")

ALLOWED_KEYS: Mapping[str, Tuple[str, ...]] = {
    KIND_RELEASE_IDENTITY: COMMON_KEYS
    + ("release_statuses", "reproducibility_levels", "required_identity_fields", "nondeterminism_sources", "bundle_fields"),
    KIND_SUPPLY_CHAIN: COMMON_KEYS
    + ("image_stages", "gate_decisions", "vuln_statuses", "severities", "secret_surfaces", "model_file_patterns",
       "negative_cases", "required_evidence_parts", "blocking_severity"),
    KIND_DEPLOYMENT: COMMON_KEYS
    + ("clean_levels", "deployment_states", "stage_names", "residual_kinds", "negative_cases", "forbidden_implicit_state"),
    KIND_SCHEDULING: COMMON_KEYS
    + ("sharing_modes", "partition_profiles", "topology_policies", "link_domains", "label_classes",
       "protected_label_classes", "isolation_layers", "negative_cases", "locality_metrics"),
    KIND_ARTIFACT: COMMON_KEYS
    + ("lifecycle_states", "cache_key_fields", "cache_areas", "verification_checks", "compatibility_checks",
       "pin_kinds", "gc_reasons", "fault_cases"),
    KIND_LIFECYCLE: COMMON_KEYS
    + ("probe_kinds", "probe_semantics", "termination_stages", "completion_semantics", "readiness_only_reasons",
       "release_targets", "negative_cases", "coverage_paths"),
    KIND_CAPACITY: COMMON_KEYS
    + ("memory_components", "admission_policies", "admission_decisions", "reason_codes", "sweep_stages",
       "budget_features", "fault_cases"),
    KIND_AUTOSCALING: COMMON_KEYS
    + ("policies", "metric_units", "leading_signals", "episodes", "actions", "failure_cases",
       "scale_down_protections"),
    KIND_OBSERVABILITY: COMMON_KEYS
    + ("layers", "signal_classes", "forbidden_metric_labels", "trace_only_attributes", "sampling_kinds",
       "always_keep_cases", "telemetry_components", "base_units", "latency_boundaries", "alert_severities",
       "policy_default_marker"),
    KIND_FAULT: COMMON_KEYS
    + ("layers", "mechanisms", "degradation_strategies", "verdicts", "pre_injection_gates",
       "service_invariants", "resource_invariants", "fault_contract_fields"),
    KIND_CANARY: COMMON_KEYS
    + ("states", "decisions", "gates", "hard_gates", "assignment_units", "candidate_kinds", "metric_directions",
       "override_actions", "forbidden_override_actions", "promotion_path"),
    KIND_SECURITY: COMMON_KEYS
    + ("invariants", "invariant_text", "case_kinds", "verdicts", "subjects", "k8s_verbs", "rbac_risks",
       "static_quota_resources", "dynamic_budgets", "abuse_kinds", "audit_event_kinds", "deny_case_kinds",
       "negative_result_wording"),
    KIND_EXPERIMENT: COMMON_KEYS
    + ("experiments", "statuses", "evidence_levels", "missingness_codes", "prerequisite_states", "tables",
       "run_root", "removed_validation", "safety_policy_required", "rules", "claim_levels"),
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
class InfraSpecs:
    documents: Mapping[str, Dict[str, Any]] = field(default_factory=dict)
    source_dir: str = ""

    @classmethod
    def load(cls, directory: str) -> "InfraSpecs":
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


def _compare(kind: str, field_name: str, document: Mapping[str, Any], expected: Sequence[Any]) -> Dict[str, Any]:
    actual = document.get(field_name)
    if not isinstance(actual, list):
        return {"kind": kind, "ok": False, "problems": [f"{field_name}: expected a list"]}
    actual_values = [str(value) for value in actual]
    expected_values = [str(value) for value in expected]
    missing = sorted(set(expected_values) - set(actual_values))
    extra = sorted(set(actual_values) - set(expected_values))
    problems = []
    if missing:
        problems.append(f"{field_name}: missing {missing}")
    if extra:
        problems.append(f"{field_name}: undocumented {extra}")
    return {"kind": kind, "ok": not problems, "problems": problems}


def _compare_keys(kind: str, field_name: str, document: Mapping[str, Any], expected: Sequence[Any]) -> Dict[str, Any]:
    """Compare the *key set* of a mapping field (units, directions, invariant text)."""
    actual = document.get(field_name)
    if not isinstance(actual, dict):
        return {"kind": kind, "ok": False, "problems": [f"{field_name}: expected a mapping"]}
    return _compare(kind, field_name, {field_name: sorted(actual)}, [str(value) for value in expected])


def audit_documents(documents: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Field-by-field audit of every document against the code constants."""
    from hqsb.infra import (
        artifacts,
        autoscaling,
        canary,
        capacity,
        deployment,
        experiment,
        faults,
        identity,
        lifecycle,
        observability,
        records,
        scheduling,
        security,
        supply_chain,
    )

    reports: List[Dict[str, Any]] = []

    doc = documents[KIND_RELEASE_IDENTITY]
    problems: List[str] = []
    for field_name, expected in (
        ("release_statuses", identity.RELEASE_STATUSES),
        ("reproducibility_levels", identity.REPRODUCIBILITY_LEVELS),
        ("required_identity_fields", identity.REQUIRED_IDENTITY_FIELDS),
        ("nondeterminism_sources", identity.NONDETERMINISM_SOURCES),
        ("bundle_fields", identity.RELEASE_BUNDLE_FIELDS),
    ):
        problems.extend(_compare(KIND_RELEASE_IDENTITY, field_name, doc, expected)["problems"])
    reports.append(_audit(KIND_RELEASE_IDENTITY, not problems, problems))

    doc = documents[KIND_SUPPLY_CHAIN]
    problems = []
    for field_name, expected in (
        ("image_stages", supply_chain.IMAGE_STAGES),
        ("gate_decisions", supply_chain.GATE_DECISIONS),
        ("vuln_statuses", supply_chain.VULN_STATUSES),
        ("severities", supply_chain.SEVERITIES),
        ("secret_surfaces", records.SECRET_SCAN_SURFACES),
        ("model_file_patterns", records.MODEL_FILE_PATTERNS),
        ("negative_cases", supply_chain.NEGATIVE_CASES),
        ("required_evidence_parts", supply_chain.REQUIRED_EVIDENCE_PARTS),
    ):
        problems.extend(_compare(KIND_SUPPLY_CHAIN, field_name, doc, expected)["problems"])
    if doc.get("blocking_severity") not in supply_chain.SEVERITIES:
        problems.append("blocking_severity must be one of the declared severities")
    reports.append(_audit(KIND_SUPPLY_CHAIN, not problems, problems))

    doc = documents[KIND_DEPLOYMENT]
    problems = []
    for field_name, expected in (
        ("clean_levels", deployment.CLEAN_LEVELS),
        ("deployment_states", deployment.DEPLOYMENT_STATES),
        ("stage_names", deployment.STAGE_NAMES),
        ("residual_kinds", deployment.RESIDUAL_KINDS),
        ("negative_cases", deployment.NEGATIVE_CASES),
        ("forbidden_implicit_state", deployment.FORBIDDEN_IMPLICIT_STATE),
    ):
        problems.extend(_compare(KIND_DEPLOYMENT, field_name, doc, expected)["problems"])
    reports.append(_audit(KIND_DEPLOYMENT, not problems, problems))

    doc = documents[KIND_SCHEDULING]
    problems = []
    for field_name, expected in (
        ("sharing_modes", scheduling.SHARING_MODES),
        ("partition_profiles", scheduling.PARTITION_PROFILES),
        ("topology_policies", scheduling.TOPOLOGY_POLICIES),
        ("link_domains", scheduling.LINK_DOMAINS),
        ("label_classes", scheduling.LABEL_CLASSES),
        ("protected_label_classes", scheduling.PROTECTED_LABEL_CLASSES),
        ("isolation_layers", scheduling.ISOLATION_LAYERS),
        ("negative_cases", scheduling.NEGATIVE_CASES),
        ("locality_metrics", scheduling.LOCALITY_METRICS),
    ):
        problems.extend(_compare(KIND_SCHEDULING, field_name, doc, expected)["problems"])
    reports.append(_audit(KIND_SCHEDULING, not problems, problems))

    doc = documents[KIND_ARTIFACT]
    problems = []
    for field_name, expected in (
        ("lifecycle_states", records.ARTIFACT_LIFECYCLE_STATES),
        ("cache_key_fields", artifacts.CACHE_KEY_FIELDS),
        ("cache_areas", artifacts.CACHE_AREAS),
        ("verification_checks", artifacts.VERIFICATION_CHECKS),
        ("compatibility_checks", artifacts.COMPATIBILITY_CHECKS),
        ("pin_kinds", artifacts.PIN_KINDS),
        ("gc_reasons", artifacts.GC_REASONS),
        ("fault_cases", artifacts.FAULT_CASES),
    ):
        problems.extend(_compare(KIND_ARTIFACT, field_name, doc, expected)["problems"])
    reports.append(_audit(KIND_ARTIFACT, not problems, problems))

    doc = documents[KIND_LIFECYCLE]
    problems = []
    for field_name, expected in (
        ("probe_kinds", records.PROBE_KINDS),
        ("termination_stages", lifecycle.TERMINATION_STAGES),
        ("completion_semantics", records.REQUEST_COMPLETION_SEMANTICS),
        ("readiness_only_reasons", lifecycle.READINESS_ONLY_REASONS),
        ("release_targets", lifecycle.RELEASE_TARGETS),
        ("negative_cases", lifecycle.NEGATIVE_CASES),
        ("coverage_paths", lifecycle.COVERAGE_PATHS),
    ):
        problems.extend(_compare(KIND_LIFECYCLE, field_name, doc, expected)["problems"])
    declared_semantics = doc.get("probe_semantics", {})
    if sorted(declared_semantics) != sorted(lifecycle.PROBE_SEMANTICS):
        problems.append("probe_semantics must define exactly the three probe kinds (startup/readiness/liveness)")
    reports.append(_audit(KIND_LIFECYCLE, not problems, problems))

    doc = documents[KIND_CAPACITY]
    problems = []
    for field_name, expected in (
        ("memory_components", records.RESOURCE_MEMORY_COMPONENTS),
        ("admission_policies", capacity.ADMISSION_POLICIES),
        ("admission_decisions", records.ADMISSION_DECISIONS),
        ("reason_codes", capacity.REASON_CODES),
        ("sweep_stages", capacity.SWEEP_STAGES),
        ("budget_features", capacity.BUDGET_FEATURES),
        ("fault_cases", capacity.FAULT_CASES),
    ):
        problems.extend(_compare(KIND_CAPACITY, field_name, doc, expected)["problems"])
    reports.append(_audit(KIND_CAPACITY, not problems, problems))

    doc = documents[KIND_AUTOSCALING]
    problems = []
    for field_name, expected in (
        ("policies", autoscaling.POLICIES),
        ("leading_signals", autoscaling.LEADING_SIGNALS),
        ("episodes", records.AUTOSCALING_EPISODES),
        ("actions", records.AUTOSCALING_ACTIONS),
        ("failure_cases", autoscaling.FAILURE_CASES),
        ("scale_down_protections", autoscaling.SCALE_DOWN_PROTECTIONS),
    ):
        problems.extend(_compare(KIND_AUTOSCALING, field_name, doc, expected)["problems"])
    problems.extend(
        _compare_keys(KIND_AUTOSCALING, "metric_units", doc, tuple(autoscaling.METRIC_UNITS))["problems"]
    )
    reports.append(_audit(KIND_AUTOSCALING, not problems, problems))

    doc = documents[KIND_OBSERVABILITY]
    problems = []
    for field_name, expected in (
        ("layers", records.OBSERVABILITY_LAYERS),
        ("signal_classes", records.SIGNAL_CLASSES),
        ("forbidden_metric_labels", records.FORBIDDEN_METRIC_LABELS),
        ("trace_only_attributes", records.TRACE_ONLY_ATTRIBUTES),
        ("sampling_kinds", observability.SAMPLING_KINDS),
        ("always_keep_cases", observability.ALWAYS_KEEP_CASES),
        ("telemetry_components", observability.TELEMETRY_COMPONENTS),
        ("base_units", observability.BASE_UNITS),
        ("latency_boundaries", observability.LATENCY_BOUNDARIES),
        ("alert_severities", records.ALERT_SEVERITIES),
    ):
        problems.extend(_compare(KIND_OBSERVABILITY, field_name, doc, expected)["problems"])
    if doc.get("policy_default_marker") != records.POLICY_DEFAULT_MARKER:
        problems.append("policy_default_marker must equal the code marker POLICY_DEFAULT_UNVERIFIED")
    reports.append(_audit(KIND_OBSERVABILITY, not problems, problems))

    doc = documents[KIND_FAULT]
    problems = []
    for field_name, expected in (
        ("layers", records.FAULT_LAYERS),
        ("mechanisms", records.FAULT_MECHANISMS),
        ("degradation_strategies", records.DEGRADATION_STRATEGIES),
        ("verdicts", records.FAULT_VERDICTS),
        ("pre_injection_gates", faults.PRE_INJECTION_GATES),
        ("service_invariants", faults.SERVICE_INVARIANTS),
        ("resource_invariants", faults.RESOURCE_INVARIANTS),
        ("fault_contract_fields", faults.FAULT_CONTRACT_FIELDS),
    ):
        problems.extend(_compare(KIND_FAULT, field_name, doc, expected)["problems"])
    reports.append(_audit(KIND_FAULT, not problems, problems))

    doc = documents[KIND_CANARY]
    problems = []
    for field_name, expected in (
        ("states", records.CANARY_STATES),
        ("decisions", records.CANARY_DECISIONS),
        ("gates", records.CANARY_GATES),
        ("hard_gates", records.CANARY_HARD_GATES),
        ("assignment_units", records.ASSIGNMENT_UNITS),
        ("candidate_kinds", canary.CANDIDATE_KINDS),
        ("override_actions", canary.OVERRIDE_ACTIONS),
        ("forbidden_override_actions", canary.FORBIDDEN_OVERRIDE_ACTIONS),
        ("promotion_path", canary.PROMOTION_PATH),
    ):
        problems.extend(_compare(KIND_CANARY, field_name, doc, expected)["problems"])
    problems.extend(
        _compare_keys(KIND_CANARY, "metric_directions", doc, tuple(canary.METRIC_DIRECTIONS))["problems"]
    )
    reports.append(_audit(KIND_CANARY, not problems, problems))

    doc = documents[KIND_SECURITY]
    problems = []
    for field_name, expected in (
        ("invariants", records.TENANT_INVARIANTS),
        ("case_kinds", records.SECURITY_CASE_KINDS),
        ("verdicts", records.SECURITY_VERDICTS),
        ("subjects", records.TENANT_SUBJECTS),
        ("k8s_verbs", security.K8S_VERBS),
        ("rbac_risks", security.RBAC_RISKS),
        ("static_quota_resources", security.STATIC_QUOTA_RESOURCES),
        ("dynamic_budgets", security.DYNAMIC_BUDGETS),
        ("abuse_kinds", security.ABUSE_KINDS),
        ("audit_event_kinds", security.AUDIT_EVENT_KINDS),
        ("deny_case_kinds", security.DENY_CASE_KINDS),
    ):
        problems.extend(_compare(KIND_SECURITY, field_name, doc, expected)["problems"])
    problems.extend(
        _compare_keys(KIND_SECURITY, "invariant_text", doc, tuple(records.TENANT_INVARIANT_TEXT))["problems"]
    )
    if doc.get("negative_result_wording") != security.NEGATIVE_RESULT_WORDING:
        problems.append("negative_result_wording must match the code constant verbatim")
    reports.append(_audit(KIND_SECURITY, not problems, problems))

    doc = documents[KIND_EXPERIMENT]
    problems = []
    for field_name, expected in (
        ("experiments", tuple(f"E13-{index:02d}" for index in range(1, 12))),
        ("statuses", records.PROTOCOL_STATUSES),
        ("evidence_levels", records.EVIDENCE_LEVELS),
        ("missingness_codes", records.MISSINGNESS_CODES),
        ("prerequisite_states", records.PREREQUISITE_STATES),
        ("tables", tuple(records.TABLE_SCHEMAS)),
        ("claim_levels", experiment.CLAIM_LEVELS),
    ):
        problems.extend(_compare(KIND_EXPERIMENT, field_name, doc, expected)["problems"])
    if not doc.get("safety_policy_required", False):
        problems.append(
            "safety_policy_required must be true: the S13 prerequisite gate reads it to decide whether "
            "destructive faults may run"
        )
    if not doc.get("removed_validation"):
        problems.append("removed_validation must state which validations were deliberately not deleted")
    reports.append(_audit(KIND_EXPERIMENT, not problems, problems))

    return reports


def audit_all_ok(reports: Sequence[Mapping[str, Any]]) -> bool:
    return all(report["ok"] for report in reports)


def default_spec_dir(root: str) -> str:
    return os.path.join(root, "configs", "infra")
