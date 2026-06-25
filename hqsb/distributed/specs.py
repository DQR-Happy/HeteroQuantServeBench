"""Strict loading and contract auditing of the frozen S10 configuration.

Every S10 configuration is a *frozen artifact*: the observation scope, the
placement rules, the collective grid, the timeout policy, the TP plan policy,
the scaling protocol, the overlap schedules, the PP/CP/SP rubric, the MoE spec,
the trace/profiler spec, the fault matrix and the shared tolerances.  The loader
refuses unknown keys instead of ignoring them, and :func:`audit_documents`
compares the documents with the code field by field so configuration and
behaviour cannot drift apart silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml

from hqsb.core.errors import ConfigError

KIND_TOPOLOGY_SPEC = "hqsb.distributed.topology_spec"
KIND_PLACEMENT_SPEC = "hqsb.distributed.placement_spec"
KIND_COLLECTIVE_SPEC = "hqsb.distributed.collective_spec"
KIND_TIMEOUT_SPEC = "hqsb.distributed.timeout_spec"
KIND_PARALLEL_PLAN_SPEC = "hqsb.distributed.parallel_plan_spec"
KIND_SCALING_SPEC = "hqsb.distributed.scaling_spec"
KIND_OVERLAP_SPEC = "hqsb.distributed.overlap_spec"
KIND_BOUNDARY_SPEC = "hqsb.distributed.boundary_spec"
KIND_MOE_SPEC = "hqsb.distributed.moe_spec"
KIND_TRACE_SPEC = "hqsb.distributed.trace_spec"
KIND_FAULT_SPEC = "hqsb.distributed.fault_spec"
KIND_TOLERANCE_SPEC = "hqsb.distributed.tolerance_spec"

KINDS: Tuple[str, ...] = (
    KIND_TOPOLOGY_SPEC,
    KIND_PLACEMENT_SPEC,
    KIND_COLLECTIVE_SPEC,
    KIND_TIMEOUT_SPEC,
    KIND_PARALLEL_PLAN_SPEC,
    KIND_SCALING_SPEC,
    KIND_OVERLAP_SPEC,
    KIND_BOUNDARY_SPEC,
    KIND_MOE_SPEC,
    KIND_TRACE_SPEC,
    KIND_FAULT_SPEC,
    KIND_TOLERANCE_SPEC,
)

#: Keys allowed on every document, independent of its kind.
COMMON_KEYS: Tuple[str, ...] = ("kind", "description", "name", "notes")

#: Per-kind allowed keys (unknown keys are refused, never ignored).
ALLOWED_KEYS: Mapping[str, Tuple[str, ...]] = {
    KIND_TOPOLOGY_SPEC: (
        "schema_version",
        "branches",
        "node_kinds",
        "edge_types",
        "edge_confidence",
        "link_states",
        "drift_levels",
        "degraded_policy",
        "hard_invariants",
        "raw_probe_commands",
        "redaction_rules",
    ),
    KIND_PLACEMENT_SPEC: (
        "schema_version",
        "invariants",
        "launcher_kinds",
        "launcher_env_keys",
        "high_bandwidth_domain_policy",
        "alternative_placement",
        "planned_actual_requirements",
    ),
    KIND_COLLECTIVE_SPEC: (
        "schema_version",
        "ops",
        "required_error_hashed",
        "count_semantics",
        "size_grid",
        "dtypes",
        "algorithm_groups",
        "order_policy",
        "independent_jobs",
        "bandwidth_formulas",
        "stop_rules",
        "official_tool_crosscheck",
    ),
    KIND_TIMEOUT_SPEC: (
        "schema_version",
        "preregistration_status",
        "preregistration_note",
        "families",
        "timeouts",
        "multiplier",
        "minimum_operational_timeout_s",
        "maximum_timeout_s",
        "healthy_p99_source",
        "watchdog_layers",
        "delayed_rank_ladder",
        "cleanup_order",
    ),
    KIND_PARALLEL_PLAN_SPEC: (
        "schema_version",
        "kv_strategies",
        "replication_policies",
        "embedding_strategies",
        "unsupported_policies",
        "shard_axes",
        "collective_triggers",
        "correctness_layers",
        "max_tp_degree",
        "require_direct_shard_load",
        "vocab_parallel_requirement",
    ),
    KIND_SCALING_SPEC: (
        "schema_version",
        "preregistration_status",
        "scaling_kinds",
        "weak_definitions",
        "device_counts",
        "node_families",
        "capacity_ladder",
        "oom_margin_fraction",
        "independent_runs",
        "block_order",
        "pairability_fields",
        "verdicts",
        "stop_rules",
        "non_extrapolation",
    ),
    KIND_OVERLAP_SPEC: (
        "schema_version",
        "schedules",
        "chunk_strategies",
        "stream_policy",
        "race_probe",
        "abba_blocks",
        "guardrails",
        "causal_matrix",
        "clock_uncertainty_ms",
        "profiler_runs_separate",
    ),
    KIND_BOUNDARY_SPEC: (
        "schema_version",
        "activation",
        "candidate_kinds",
        "cp_sp_algorithms",
        "rubric_factors",
        "rubric_table",
        "adopt_criteria",
        "re_evaluation_triggers",
        "probe_chain",
        "unsupported_cases",
    ),
    KIND_MOE_SPEC: (
        "schema_version",
        "claim_levels",
        "claim_level_table",
        "skew_profiles",
        "overflow_policies",
        "routing_stats_fields",
        "placement_kinds",
        "holdout_divergence_threshold",
        "l4_requires_real_artifact",
    ),
    KIND_TRACE_SPEC: (
        "schema_version",
        "modes",
        "clock_domains",
        "metric_manifest_fields",
        "root_cause_classes",
        "evidence_matrix",
        "injection_kinds",
        "unmapped_ratio_threshold",
        "deep_counter_policy",
        "synchronised_clocks_required",
    ),
    KIND_FAULT_SPEC: (
        "schema_version",
        "preregistration_status",
        "error_classes",
        "recovery_levels",
        "detection_layers",
        "fault_matrix",
        "stop_conditions",
        "required_oracle_fields",
        "isolation_requirements",
        "partial_validation_policy",
    ),
    KIND_TOLERANCE_SPEC: (
        "schema_version",
        "preregistration_status",
        "collective_tolerances",
        "model_tolerances",
        "no_backend_relaxation",
        "rel_floor",
        "gates",
    ),
}


def _require(condition: bool, message: str, *, field_name: str = "") -> None:
    if not condition:
        raise ConfigError(message, details={"field": field_name} if field_name else None)


def load_yaml_document(path: str) -> Dict[str, Any]:
    """Load one YAML document and refuse unknown/missing/bad-kind keys."""
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: the document must be a mapping")
    kind = payload.get("kind")
    if kind not in KINDS:
        raise ConfigError(
            f"{path}: kind must be one of {list(KINDS)}, got {kind!r}", details={"field": "kind"}
        )
    allowed = set(COMMON_KEYS) | set(ALLOWED_KEYS[kind])
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ConfigError(
            f"{path}: unknown keys {unknown} for kind {kind!r} (silently ignoring them is not "
            "allowed)",
            details={"field": unknown[0]},
        )
    return payload


@dataclass
class DistributedSpecs:
    """All loaded S10 documents plus their per-document audit results."""

    documents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    audits: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(audit["ok"] for audit in self.audits) and len(self.documents) == len(KINDS)

    def document(self, kind: str) -> Dict[str, Any]:
        if kind not in self.documents:
            raise ConfigError(f"document {kind!r} was not loaded", details={"field": "kind"})
        return self.documents[kind]

    @classmethod
    def load(cls, directory: str) -> "DistributedSpecs":
        if not os.path.isdir(directory):
            raise ConfigError(f"configuration directory {directory!r} does not exist")
        instances = cls()
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(directory, filename)
            document = load_yaml_document(path)
            instances.documents[document["kind"]] = document
        missing = sorted(set(KINDS) - set(instances.documents))
        if missing:
            raise ConfigError(
                f"missing configuration documents: {missing} (expected one per kind)"
            )
        instances.audits = audit_documents(instances.documents)
        return instances


def _audit(kind: str, ok: bool, problems: Sequence[str]) -> Dict[str, Any]:
    return {"kind": kind, "ok": ok, "problems": list(problems)}


def audit_documents(documents: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Compare every document with the code that will consume it."""
    from hqsb.distributed import boundary as boundary_mod
    from hqsb.distributed import collectives as collectives_mod
    from hqsb.distributed import faults as faults_mod
    from hqsb.distributed import moe as moe_mod
    from hqsb.distributed import overlap as overlap_mod
    from hqsb.distributed import parallel_plan as plan_mod
    from hqsb.distributed import placement as placement_mod
    from hqsb.distributed import probes as probes_mod
    from hqsb.distributed import scaling as scaling_mod
    from hqsb.distributed import sequence as sequence_mod
    from hqsb.distributed import topology as topology_mod
    from hqsb.distributed import traces as traces_mod

    audits: List[Dict[str, Any]] = []

    def compare_list(kind: str, key: str, expected: Sequence[Any]) -> Dict[str, Any]:
        """Compare a document field with a code constant.

        The field may be a list of names (``ops``) or a mapping whose *keys* are
        the names (``stop_rules``/``rubric_table``): both must match the code
        exactly, so a renamed rule cannot drift away from the implementation.
        """
        document = documents[kind]
        actual = document.get(key)
        problems: List[str] = []
        if isinstance(actual, Mapping):
            actual_set = [str(item) for item in actual]
        elif isinstance(actual, (list, tuple)):
            actual_set = [str(item) for item in actual]
        else:
            problems.append(f"{key} must be a list or mapping")
            actual_set = []
        expected_set = [str(item) for item in expected]
        missing = sorted(set(expected_set) - set(actual_set))
        unknown = sorted(set(actual_set) - set(expected_set))
        if missing:
            problems.append(f"{key} is missing {missing}")
        if unknown:
            problems.append(f"{key} lists unknown values {unknown}")
        return _audit(kind, not problems, problems)

    topology_doc = documents[KIND_TOPOLOGY_SPEC]
    audits.append(compare_list(KIND_TOPOLOGY_SPEC, "branches", topology_mod.BRANCHES))
    audits.append(compare_list(KIND_TOPOLOGY_SPEC, "node_kinds", topology_mod.NODE_KINDS))
    audits.append(compare_list(KIND_TOPOLOGY_SPEC, "edge_types", topology_mod.EDGE_TYPES))
    audits.append(
        compare_list(KIND_TOPOLOGY_SPEC, "edge_confidence", topology_mod.EDGE_CONFIDENCE)
    )
    audits.append(compare_list(KIND_TOPOLOGY_SPEC, "link_states", topology_mod.LINK_STATES))
    audits.append(compare_list(KIND_TOPOLOGY_SPEC, "drift_levels", list(topology_mod.DRIFT_ACTIONS)))
    audits.append(
        compare_list(
            KIND_TOPOLOGY_SPEC,
            "raw_probe_commands",
            list(probes_mod.PROBE_COMMANDS),
        )
    )
    degraded_action = str(topology_doc.get("degraded_policy", {}).get("action", ""))
    audits.append(
        _audit(
            KIND_TOPOLOGY_SPEC,
            degraded_action in topology_mod.DEGRADED_ACTIONS,
            [] if degraded_action in topology_mod.DEGRADED_ACTIONS else ["degraded_policy.action invalid"],
        )
    )

    audits.append(
        compare_list(KIND_PLACEMENT_SPEC, "invariants", placement_mod.DEFAULT_INVARIANTS)
    )
    audits.append(compare_list(KIND_PLACEMENT_SPEC, "launcher_kinds", placement_mod.LAUNCHER_KINDS))
    audits.append(
        compare_list(KIND_PLACEMENT_SPEC, "launcher_env_keys", placement_mod.LAUNCHER_ENV_KEYS)
    )

    audits.append(compare_list(KIND_COLLECTIVE_SPEC, "ops", collectives_mod.OPS))
    audits.append(
        compare_list(
            KIND_COLLECTIVE_SPEC, "required_error_hashed", collectives_mod.REQUIRED_OP_ERROR_HASHED
        )
    )
    audits.append(compare_list(KIND_COLLECTIVE_SPEC, "dtypes", list(collectives_mod.DTYPE_BYTES)))
    audits.append(compare_list(KIND_COLLECTIVE_SPEC, "stop_rules", list(collectives_mod.STOP_RULES)))
    formulas = documents[KIND_COLLECTIVE_SPEC].get("bandwidth_formulas", [])
    formula_ids = {
        str(item.get("id", "")) for item in formulas if isinstance(item, Mapping)
    }
    expected_ids = {entry["id"] for entry in collectives_mod.FORMULA_REGISTRY.values()}
    audits.append(
        _audit(
            KIND_COLLECTIVE_SPEC,
            expected_ids <= formula_ids,
            [] if expected_ids <= formula_ids else [f"missing formula ids {sorted(expected_ids - formula_ids)}"],
        )
    )

    timeout_doc = documents[KIND_TIMEOUT_SPEC]
    timeout_fields = set(timeout_doc.get("timeouts", {}))
    expected_timeout_fields = {
        "init_timeout_s",
        "collective_timeout_s",
        "watchdog_heartbeat_timeout_s",
        "job_kill_grace_s",
        "post_cleanup_probe_timeout_s",
    }
    audits.append(
        _audit(
            KIND_TIMEOUT_SPEC,
            expected_timeout_fields <= timeout_fields,
            []
            if expected_timeout_fields <= timeout_fields
            else [f"timeouts missing {sorted(expected_timeout_fields - timeout_fields)}"],
        )
    )
    audits.append(
        compare_list(KIND_TIMEOUT_SPEC, "families", sequence_mod.TIMEOUT_FAMILIES)
    )
    audits.append(
        _audit(
            KIND_TIMEOUT_SPEC,
            str(timeout_doc.get("preregistration_status", "")) in ("template", "frozen"),
            ["preregistration_status must be template|frozen"],
        )
    )

    audits.append(compare_list(KIND_PARALLEL_PLAN_SPEC, "kv_strategies", plan_mod.KV_STRATEGIES))
    audits.append(
        compare_list(KIND_PARALLEL_PLAN_SPEC, "replication_policies", plan_mod.REPLICATION_POLICIES)
    )
    audits.append(
        compare_list(
            KIND_PARALLEL_PLAN_SPEC, "embedding_strategies", plan_mod.EMBEDDING_STRATEGIES
        )
    )
    audits.append(
        compare_list(KIND_PARALLEL_PLAN_SPEC, "unsupported_policies", plan_mod.UNSUPPORTED_POLICIES)
    )
    audits.append(
        compare_list(KIND_PARALLEL_PLAN_SPEC, "correctness_layers", plan_mod.MODEL_CORRECTNESS_LAYERS)
    )

    audits.append(compare_list(KIND_SCALING_SPEC, "scaling_kinds", scaling_mod.SCALING_KINDS))
    audits.append(
        compare_list(KIND_SCALING_SPEC, "weak_definitions", scaling_mod.WEAK_DEFINITIONS)
    )
    audits.append(compare_list(KIND_SCALING_SPEC, "verdicts", scaling_mod.SCALING_VERDICTS))
    audits.append(compare_list(KIND_SCALING_SPEC, "stop_rules", list(scaling_mod.STOP_RULES)))
    audits.append(
        compare_list(KIND_SCALING_SPEC, "pairability_fields", scaling_mod.PAIRABILITY_FIELDS)
    )

    audits.append(compare_list(KIND_OVERLAP_SPEC, "schedules", overlap_mod.SCHEDULE_KINDS))
    audits.append(
        compare_list(KIND_OVERLAP_SPEC, "chunk_strategies", overlap_mod.CHUNK_STRATEGIES)
    )
    audits.append(
        compare_list(KIND_OVERLAP_SPEC, "guardrails", overlap_mod.OVERLAP_GUARDRAILS)
    )

    audits.append(compare_list(KIND_BOUNDARY_SPEC, "candidate_kinds", boundary_mod.CANDIDATE_KINDS))
    audits.append(
        compare_list(KIND_BOUNDARY_SPEC, "cp_sp_algorithms", boundary_mod.CP_SP_ALGORITHMS)
    )
    audits.append(compare_list(KIND_BOUNDARY_SPEC, "rubric_factors", boundary_mod.RUBRIC_FACTORS))
    audits.append(
        compare_list(KIND_BOUNDARY_SPEC, "adopt_criteria", boundary_mod.ADOPT_REJECT_CRITERIA)
    )
    audits.append(compare_list(KIND_BOUNDARY_SPEC, "probe_chain", boundary_mod.PROBE_CHAIN))

    audits.append(compare_list(KIND_MOE_SPEC, "claim_levels", moe_mod.CLAIM_LEVELS))
    audits.append(compare_list(KIND_MOE_SPEC, "skew_profiles", moe_mod.SKEW_PROFILES))
    audits.append(compare_list(KIND_MOE_SPEC, "overflow_policies", moe_mod.OVERFLOW_POLICIES))
    audits.append(
        compare_list(KIND_MOE_SPEC, "routing_stats_fields", moe_mod.ROUTING_STATS_FIELDS)
    )
    audits.append(compare_list(KIND_MOE_SPEC, "placement_kinds", moe_mod.PLACEMENT_KINDS))

    audits.append(compare_list(KIND_TRACE_SPEC, "modes", traces_mod.TRACE_MODES))
    audits.append(compare_list(KIND_TRACE_SPEC, "clock_domains", traces_mod.CLOCK_DOMAINS))
    audits.append(
        compare_list(KIND_TRACE_SPEC, "root_cause_classes", traces_mod.ROOT_CAUSE_CLASSES)
    )
    audits.append(
        compare_list(KIND_TRACE_SPEC, "injection_kinds", traces_mod.INJECTION_KINDS)
    )

    audits.append(compare_list(KIND_FAULT_SPEC, "error_classes", faults_mod.ERROR_CLASSES))
    audits.append(compare_list(KIND_FAULT_SPEC, "recovery_levels", faults_mod.RECOVERY_LEVELS))
    audits.append(
        compare_list(KIND_FAULT_SPEC, "detection_layers", faults_mod.DETECTION_LAYERS)
    )
    fault_doc = documents[KIND_FAULT_SPEC]
    matrix_rows = fault_doc.get("fault_matrix", [])
    matrix_faults = {
        str(item.get("fault", "")) for item in matrix_rows if isinstance(item, Mapping)
    }
    expected_faults = set(faults_mod.MINIMUM_RECOVERY_BY_FAULT)
    audits.append(
        _audit(
            KIND_FAULT_SPEC,
            expected_faults <= matrix_faults,
            []
            if expected_faults <= matrix_faults
            else [f"fault_matrix missing {sorted(expected_faults - matrix_faults)}"],
        )
    )

    tolerance_doc = documents[KIND_TOLERANCE_SPEC]
    ops_covered = {str(item.get("op", "")) for item in tolerance_doc.get("collective_tolerances", []) if isinstance(item, Mapping)}
    audits.append(
        _audit(
            KIND_TOLERANCE_SPEC,
            set(collectives_mod.REQUIRED_OP_ERROR_HASHED) <= ops_covered,
            []
            if set(collectives_mod.REQUIRED_OP_ERROR_HASHED) <= ops_covered
            else [
                "collective_tolerances must cover "
                f"{sorted(set(collectives_mod.REQUIRED_OP_ERROR_HASHED) - ops_covered)}"
            ],
        )
    )
    audits.append(
        _audit(
            KIND_TOLERANCE_SPEC,
            bool(tolerance_doc.get("no_backend_relaxation")),
            ["no_backend_relaxation must be true: one gate per (op, dtype), no per-backend relaxation"],
        )
    )
    return audits


__all__ = [
    "ALLOWED_KEYS",
    "COMMON_KEYS",
    "DistributedSpecs",
    "KINDS",
    "KIND_BOUNDARY_SPEC",
    "KIND_COLLECTIVE_SPEC",
    "KIND_FAULT_SPEC",
    "KIND_MOE_SPEC",
    "KIND_OVERLAP_SPEC",
    "KIND_PARALLEL_PLAN_SPEC",
    "KIND_PLACEMENT_SPEC",
    "KIND_SCALING_SPEC",
    "KIND_TIMEOUT_SPEC",
    "KIND_TOLERANCE_SPEC",
    "KIND_TOPOLOGY_SPEC",
    "KIND_TRACE_SPEC",
    "audit_documents",
    "load_yaml_document",
]
