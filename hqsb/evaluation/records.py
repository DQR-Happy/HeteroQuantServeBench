"""Shared vocabularies, missingness semantics and table schemas for S12.

Every S12 experiment writes machine-readable tables; this module is the single
place where their *names*, *required fields* and *status vocabularies* are
defined, so that

* a table row that silently drops a required field fails validation instead of
  passing as "mostly complete";
* "not measured" can never be stored as the number ``0``
  (``numeric_value`` refuses a numeric payload for a missing state);
* the status propagation of ``details/S12/README.md`` §3.2 is data, not prose.

Nothing here executes an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from hqsb.core.errors import ConfigError

SCHEMA_VERSION = "1.0.0"

# ── protocol statuses (stage_experiments/README.md §3) ─────────────────────

STATUS_NOT_STARTED = "NOT_STARTED"
STATUS_RUNNING = "RUNNING"
STATUS_PASS = "PASS"
STATUS_PASS_NEGATIVE = "PASS_NEGATIVE"
STATUS_FAIL = "FAIL"
STATUS_BLOCKED = "BLOCKED"
STATUS_N_A_BY_ADR = "N/A_BY_ADR"

PROTOCOL_STATUSES: Tuple[str, ...] = (
    STATUS_NOT_STARTED,
    STATUS_RUNNING,
    STATUS_PASS,
    STATUS_PASS_NEGATIVE,
    STATUS_FAIL,
    STATUS_BLOCKED,
    STATUS_N_A_BY_ADR,
)

CONCLUSION_STATUSES: Tuple[str, ...] = (
    STATUS_PASS,
    STATUS_PASS_NEGATIVE,
    STATUS_FAIL,
    STATUS_BLOCKED,
    STATUS_N_A_BY_ADR,
)

#: Categories for which a negative result may not replace a pass
#: (handbook §3: safety, artifact compatibility, correctness, data isolation,
#: quality gates).
NEGATIVE_NOT_ALLOWED_CATEGORIES: Tuple[str, ...] = (
    "safety",
    "artifact_compatibility",
    "correctness",
    "data_isolation",
    "quality_gate",
)

# ── result classes (details/S12/README.md §3.3) ────────────────────────────

RESULT_CLASS_MEASURED = "MEASURED"
RESULT_CLASS_MODELED = "MODELED"
RESULT_CLASS_SCENARIO = "SCENARIO"

RESULT_CLASSES: Tuple[str, ...] = (
    RESULT_CLASS_MEASURED,
    RESULT_CLASS_MODELED,
    RESULT_CLASS_SCENARIO,
)

#: Only ``MEASURED`` results may fill a ``measured_value`` column.
MEASURED_COLUMNS: Tuple[str, ...] = ("measured_value", "measured_interval", "measured_result_id")
MODELED_COLUMNS: Tuple[str, ...] = ("predicted_value", "predicted_interval", "prediction_id")
SCENARIO_COLUMNS: Tuple[str, ...] = ("scenario_value", "assumption_ids", "scenario_id")

# ── evidence levels (§19) ──────────────────────────────────────────────────

EVIDENCE_LEVELS: Tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4")

EVIDENCE_LEVEL_CRITERIA: Mapping[str, str] = {
    "L0": "规格或文档声明（厂商/理论上限）",
    "L1": "本机 probe 通过（最小能力已验证）",
    "L2": "单次有效 benchmark run",
    "L3": "多进程/多时段重复 + raw lineage",
    "L4": "多设备/独立复现或受控对照",
}

#: Performance, energy, cost and maturity each carry their own level; a
#: performance L3 must not upgrade a price estimate.  This tuple names the
#: independent axes.
EVIDENCE_AXES: Tuple[str, ...] = ("performance", "quality", "energy", "cost", "maturity", "stable")

# ── upstream evidence states (§2) ─────────────────────────────────────────

UPSTREAM_VERIFIED = "VERIFIED"
UPSTREAM_VERIFIED_WITH_LIMITS = "VERIFIED_WITH_LIMITS"
UPSTREAM_UNVERIFIED = "UNVERIFIED"
UPSTREAM_MISSING = "MISSING"
UPSTREAM_INCOMPATIBLE = "INCOMPATIBLE"

UPSTREAM_EVIDENCE_STATES: Tuple[str, ...] = (
    UPSTREAM_VERIFIED,
    UPSTREAM_VERIFIED_WITH_LIMITS,
    UPSTREAM_UNVERIFIED,
    UPSTREAM_MISSING,
    UPSTREAM_INCOMPATIBLE,
)

#: Only these two states may enter a formal comparison.
UPSTREAM_COMPARABLE_STATES: Tuple[str, ...] = (UPSTREAM_VERIFIED, UPSTREAM_VERIFIED_WITH_LIMITS)

# ── comparability verdicts (§5) ───────────────────────────────────────────

COMPARABLE = "COMPARABLE"
CONDITIONAL = "CONDITIONAL"
NOT_COMPARABLE = "NOT_COMPARABLE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

COMPARABILITY_STATES: Tuple[str, ...] = (
    COMPARABLE,
    CONDITIONAL,
    NOT_COMPARABLE,
    INSUFFICIENT_EVIDENCE,
)

#: States allowed to enter ranking / Pareto (``CONDITIONAL`` only with an
#: explicit opt-in and a frozen normalization formula).
RANKABLE_STATES: Tuple[str, ...] = (COMPARABLE,)
CONDITIONAL_OPT_IN_STATES: Tuple[str, ...] = (COMPARABLE, CONDITIONAL)

# ── missingness semantics (§5.1) ──────────────────────────────────────────

MISSING_NOT_APPLICABLE_CAPABILITY = "NOT_APPLICABLE_CAPABILITY"
MISSING_NOT_RUN_PREREQUISITE = "NOT_RUN_PREREQUISITE"
MISSING_NOT_RUN_TOOL_UNAVAILABLE = "NOT_RUN_TOOL_UNAVAILABLE"
MISSING_RUN_FAILED = "RUN_FAILED"
MISSING_QUALITY_GATE_FAILED = "QUALITY_GATE_FAILED"
MISSING_NOT_COMPARABLE = "NOT_COMPARABLE"
MISSING_MEASUREMENT_UNAVAILABLE = "MEASUREMENT_UNAVAILABLE"
MISSING_PRICE_UNAVAILABLE = "PRICE_UNAVAILABLE"
MISSING_LINEAGE_INVALID = "LINEAGE_INVALID"
MISSING_NO_FEASIBLE_CANDIDATE = "NO_FEASIBLE_CANDIDATE"

MISSINGNESS_CODES: Tuple[str, ...] = (
    MISSING_NOT_APPLICABLE_CAPABILITY,
    MISSING_NOT_RUN_PREREQUISITE,
    MISSING_NOT_RUN_TOOL_UNAVAILABLE,
    MISSING_RUN_FAILED,
    MISSING_QUALITY_GATE_FAILED,
    MISSING_NOT_COMPARABLE,
    MISSING_MEASUREMENT_UNAVAILABLE,
    MISSING_PRICE_UNAVAILABLE,
    MISSING_LINEAGE_INVALID,
    MISSING_NO_FEASIBLE_CANDIDATE,
)

#: Which missing states are *structural* (the cell can never be filled) versus
#: *recoverable* (a rerun/probe/price refresh can fill it).
MISSINGNESS_NATURE: Mapping[str, str] = {
    MISSING_NOT_APPLICABLE_CAPABILITY: "structural",
    MISSING_NOT_RUN_PREREQUISITE: "recoverable",
    MISSING_NOT_RUN_TOOL_UNAVAILABLE: "recoverable",
    MISSING_RUN_FAILED: "recoverable",
    MISSING_QUALITY_GATE_FAILED: "structural_until_quality_changes",
    MISSING_NOT_COMPARABLE: "structural_for_this_group",
    MISSING_MEASUREMENT_UNAVAILABLE: "structural",
    MISSING_PRICE_UNAVAILABLE: "recoverable",
    MISSING_LINEAGE_INVALID: "recoverable",
    MISSING_NO_FEASIBLE_CANDIDATE: "structural_for_this_profile",
}

# ── cell / metric / claim statuses (§3.2) ─────────────────────────────────

CELL_STATUSES: Tuple[str, ...] = (
    "RAN",
    "NOT_RUN",
    "BLOCKED",
    "NOT_APPLICABLE",
    "QUALITY_FAILED",
    "UNSTABLE",
    "CONDITIONAL",
    "UNMEASURABLE",
)

METRIC_STATUSES: Tuple[str, ...] = (
    "VALID",
    "PRELIMINARY",
    "UNSTABLE",
    "INSUFFICIENT",
    "MISSING",
    "INVALID",
)

CLAIM_STATUSES: Tuple[str, ...] = (
    "PUBLISHABLE",
    "PUBLISHABLE_WITH_LIMITS",
    "NOT_PUBLISHABLE",
    "ORPHAN",
    "WITHDRAWN",
)

#: ``details/S12/README.md`` §3.2 status propagation, as data.
STATUS_PROPAGATION: Tuple[Mapping[str, str], ...] = (
    {
        "source_status": "E12-01 NOT_COMPARABLE",
        "scope": "comparison_edge",
        "effect": "禁止该 comparison edge；候选仍可参加其他合法组",
    },
    {
        "source_status": "E12-02 capability missing",
        "scope": "layer_or_feature",
        "effect": "对应 layer/feature 记 NOT_APPLICABLE 或 BLOCKED；其他 feature 可继续",
    },
    {
        "source_status": "E12-03 quality fail",
        "scope": "candidate_cell",
        "effect": "该 candidate×precision×workload 不进性能/energy/cost/Pareto",
    },
    {
        "source_status": "E12-04 unstable",
        "scope": "cell_evidence",
        "effect": "保留 point 与区间，证据降级；高置信推荐受阻",
    },
    {
        "source_status": "E12-05 high residual",
        "scope": "model_claim",
        "effect": "不阻止实测结果，但禁止高置信机理/外推",
    },
    {
        "source_status": "E12-06 meter unavailable",
        "scope": "energy_claim",
        "effect": "性能可继续，energy objective/claim 缺失",
    },
    {
        "source_status": "E12-07 price unavailable/stale",
        "scope": "cost_claim",
        "effect": "性能/能效可继续，TCO/cost recommendation 失效",
    },
    {
        "source_status": "E12-09 maturity evidence incomplete",
        "scope": "recommendation",
        "effect": "technical frontier 可生成，deployable recommendation 降级",
    },
    {
        "source_status": "E12-10 lineage invalid",
        "scope": "point_claim",
        "effect": "受影响 point/claim 不得发布，即使数字看起来合理",
    },
)

# ── artifact / validation statuses ────────────────────────────────────────

VALIDATION_STATUSES: Tuple[str, ...] = ("PASS", "FAIL", "SKIPPED", "NOT_RUN", "NOT_APPLICABLE")

#: Exact/tolerance comparison outcomes used by E12-10 regeneration.
DIFF_KINDS: Tuple[str, ...] = ("EXACT", "TOLERANCE", "PRESENTATION", "SEMANTIC_CHANGE", "MISSING")

# ── regeneration grades (details/S12/README.md §7 / E12-10 §7) ────────────

REGENERATION_LEVELS: Tuple[str, ...] = ("R0", "R1", "R2", "R3", "R4")

REGENERATION_CRITERIA: Mapping[str, str] = {
    "R0": "文件存在但 lineage 不完整",
    "R1": "raw/derived hash 与引用可验证",
    "R2": "可从 raw 重建 normalized 数字",
    "R3": "可从 normalized/analysis 重建图表/报告",
    "R4": "独立 clean environment 完成 R2/R3 并解释允许差异",
}

#: Formal S12 reports need R3 for every claim and R4 for the sampled key results.
MIN_REGENERATION_FOR_REPORT = "R3"
MIN_REGENERATION_FOR_KEY_RESULT = "R4"

# ── table schemas ─────────────────────────────────────────────────────────
#
# Table names are the *logical* names of the mandatory outputs in the ten
# experiment files; each entry lists the required columns a row must carry.

_T_IDENTITY = ("campaign_id", "created_at", "identity_schema_version")
_T_ACTOR = ("run_id", "experiment_id", "git_commit", "git_dirty")

TABLE_SCHEMAS: Mapping[str, Tuple[str, ...]] = {
    # E12-01
    "e12_01.estimands": ("estimand_id", "layer", "scenario", "object", "boundary", "population", "estimator", "unit"),
    "e12_01.candidates": (
        "candidate_id", "platform_id", "hardware_sku", "device_count", "topology_id",
        "software_stack_id", "backend_id", "model_artifact_id", "tokenizer_id",
        "precision_contract_id", "workload_spec_id", "measurement_contract_id",
        "comparison_id", "raw_manifest_hash", "display_name",
    ),
    "e12_01.upstream_evidence": (
        "candidate_id", "upstream_stage", "experiment_id", "artifact_uri", "state",
        "reason", "schema_ok", "hash_ok", "verified_at",
    ),
    "e12_01.field_classification": ("field_path", "group", "field_class", "policy_version", "rationale"),
    "e12_01.field_diff": (
        "diff_id", "comparison_id", "candidate_a", "candidate_b", "field_path", "field_class",
        "value_a_hash", "value_b_hash", "match_status", "reason_code",
    ),
    "e12_01.machine_verdicts": (
        "verdict_id", "comparison_id", "candidate_a", "candidate_b_or_reference", "field_path",
        "field_class", "verdict", "reason_code", "evidence_refs", "policy_version",
    ),
    "e12_01.reviewer_records": (
        "review_id", "comparison_id", "reviewer", "reviewed_at", "verdict", "disagreements", "evidence_refs",
    ),
    "e12_01.final_verdicts": (
        "verdict_id", "comparison_id", "candidate_a", "candidate_b_or_reference", "verdict",
        "reason_code", "normalization_formula_id", "allowed_analyses", "forbidden_claims", "policy_version",
    ),
    "e12_01.negative_cases": ("case_id", "difference_class", "expected_verdict", "expected_reason_code", "payload"),
    "e12_01.negative_results": ("case_id", "observed_verdict", "observed_reason_code", "detected", "artifact_refs"),
    "e12_01.validator_results": ("check_id", "constraint", "status", "detail", "affected_ids"),
    "e12_01.comparison_groups": (
        "comparison_group_id", "comparison_id", "layer", "scenario", "member_candidate_ids",
        "contract_sha256", "verdict", "allowed_differences", "downstream_scope", "suite_manifest_id",
    ),
    "e12_01.suite_manifests": ("suite_manifest_id", "comparison_group_id", "contract_sha256", "member_cell_ids", "frozen_at", "invalidated_by"),
    "e12_01.action_items": ("item_id", "verdict_id", "candidate_id", "action", "owner", "priority", "blocking"),
    # E12-02
    "e12_02.features": (
        "feature_id", "layer", "description", "inputs", "success_criteria", "evidence_ttl",
        "depends_on", "required_by_group", "criticality",
    ),
    "e12_02.platform_instances": (
        "platform_instance_id", "vendor", "sku", "revision", "device_count", "partition", "memory_bytes",
        "interconnect", "host_id", "identity_conflicts", "collected_at",
    ),
    "e12_02.software_stack": (
        "platform_instance_id", "firmware", "driver", "runtime", "compiler", "framework", "containers",
        "os_kernel", "abi_version", "captured_at",
    ),
    "e12_02.declared_sources": ("declared_source_id", "feature_id", "url", "document_version", "retrieved_at", "excerpt_hash"),
    "e12_02.discovered": ("discovery_id", "platform_instance_id", "api_or_command", "raw_artifact_uri", "normalized_fields", "status"),
    "e12_02.probes_specs": ("probe_spec_id", "feature_id", "command_or_api", "inputs", "timeout_s", "resource_limits", "reference_impl"),
    "e12_02.probes_results": (
        "probe_result_id", "probe_spec_id", "platform_instance_id", "feature_id", "exit_code", "signal", "timeout",
        "build_status", "load_status", "execution_status", "sync_status", "failure_category", "reason",
    ),
    "e12_02.probes_dispatch": ("probe_result_id", "requested_backend", "actual_backend", "fallback_reason", "evidence_kinds", "confirmed"),
    "e12_02.numeric_errors": ("probe_result_id", "metric", "value", "tolerance", "status", "reference_hash"),
    "e12_02.telemetry_fields": (
        "field_id", "platform_instance_id", "canonical_field", "vendor_field", "semantics", "boundary",
        "unit", "resolution", "sample_period", "availability", "permission", "raw_artifact_uri",
    ),
    "e12_02.telemetry_smoke": ("series_id", "field_id", "t0", "t1", "variance_observed", "monotonic", "stale_detected"),
    "e12_02.negative_results": ("case_id", "injection", "expected_category", "observed_category", "detected", "artifact_refs"),
    "e12_02.evidence": (
        "evidence_id", "platform_instance_id", "feature_id", "declared_status", "discovered_status",
        "verified_status", "benchmarked_status", "probe_spec_hash", "input_hash", "requested_backend",
        "actual_backend", "correctness_status", "started_at", "ended_at", "valid_until",
        "invalidation_key", "artifact_refs",
    ),
    "e12_02.dependencies": ("feature_id", "depends_on_feature_id", "dependency_kind", "satisfied", "reason"),
    "e12_02.invalidation": ("trigger", "affected_feature_ids", "whitelist_reason", "policy_version"),
    "e12_02.coverage_join": ("comparison_group_id", "candidate_id", "required_feature_ids", "missing_feature_ids", "join_status", "reason"),
    # E12-03
    "e12_03.run_matrix": (
        "cell_id", "comparison_group_id", "candidate_id", "layer", "scenario", "workload_spec_id",
        "measurement_contract_id", "planned_repetitions", "run_family_id", "status",
    ),
    "e12_03.observations": (
        "observation_id", "run_id", "sample_id", "timestamp_monotonic_ns", "candidate_id", "comparison_id",
        "layer", "scenario", "workload_spec_id", "requested_backend", "actual_backend", "fallback_reason",
        "input_tokens", "requested_output_tokens", "accepted_output_tokens", "latency_component", "latency_ns",
        "status", "error_code", "memory_boundary", "memory_bytes", "quality_gate_id", "quality_status",
        "source_artifact_refs",
    ),
    "e12_03.normalized_results": (
        "normalized_result_id", "source_observation_ids", "transform_activity_id", "metric_name", "value", "unit",
        "direction", "estimator", "interval_low", "interval_high", "confidence_level", "normalization_basis",
        "denominator_value", "comparability_status", "quality_status", "evidence_level", "missing_reason", "limitations",
    ),
    "e12_03.operator_correctness": ("case_id", "shape", "dtype", "layout", "reference_id", "tolerance", "max_abs_err", "max_rel_err", "status"),
    "e12_03.operator_samples": ("sample_id", "case_id", "implementation_id", "iteration", "latency_ns", "sync_boundary", "status"),
    "e12_03.operator_dispatch": ("case_id", "requested_backend", "actual_backend", "symbol", "library_algorithm", "fallback_reason", "launch_params"),
    "e12_03.model_core_correctness": ("case_id", "workload_spec_id", "selected_tensors", "greedy_tokens", "token_count_status", "kv_state_status", "status"),
    "e12_03.model_core_requests": ("request_id", "workload_spec_id", "batch", "isl", "requested_osl", "accepted_osl", "phase", "status", "error_code"),
    "e12_03.model_core_tokens": ("token_event_id", "request_id", "token_index", "visible_time_ns", "phase", "accepted", "draft"),
    "e12_03.model_core_memory": ("request_id", "weights_bytes", "kv_bytes", "workspace_bytes", "allocator_allocated", "allocator_reserved", "graph_cache_bytes", "peak_bytes", "steady_bytes", "missing_reason"),
    "e12_03.service_requests": ("request_id", "arrival_ns", "admission_ns", "execution_start_ns", "complete_ns", "status", "slo_compliant", "error_code"),
    "e12_03.service_tokens": ("token_event_id", "request_id", "token_index", "visible_time_ns", "accepted", "draft", "rejected"),
    "e12_03.service_scheduler": ("event_id", "t_ns", "queue_depth", "active_requests", "waiting_requests", "batch_tokens", "kv_blocks", "admission_decisions", "rejects"),
    "e12_03.distributed_correctness": ("case_id", "op", "dtype", "device_count", "topology_id", "reference_id", "max_abs_err", "status"),
    "e12_03.distributed_collectives": ("case_id", "op", "message_bytes", "rank_count", "algorithm", "comm_ns", "compute_ns", "wait_ns", "error"),
    "e12_03.distributed_scaling": ("cell_id", "scaling_kind", "device_count", "global_workload", "local_workload", "wall_seconds", "device_seconds", "memory_margin", "baseline_run_id"),
    "e12_03.system_telemetry": ("run_id", "t_ns", "temperature_c", "clock_mhz", "power_w", "throttle_flags", "cpu_load", "swap_used", "oom_events", "neighbor_processes"),
    "e12_03.schema_results": ("check_id", "table", "row_id", "constraint", "status", "detail"),
    "e12_03.run_estimates": ("run_id", "cell_id", "metric_name", "estimator", "value", "unit", "sample_count", "inclusion_status", "interval_low", "interval_high"),
    "e12_03.comparison_matrix": ("cell_id", "comparison_group_id", "candidate_id", "layer", "metric_name", "raw_value", "raw_unit", "normalized_value", "normalized_unit", "denominator", "status"),
    "e12_03.cross_layer_conversion": ("chain_id", "candidate_id", "workload_spec_id", "operator_effect", "op_time_share", "dispatch_hit_rate", "predicted_model_core_effect", "observed_model_core_effect", "observed_service_effect", "observed_distributed_effect", "residual_status"),
    # E12-04
    "e12_04.replication": ("level", "unit_name", "min_count", "role", "primary_inference_unit"),
    "e12_04.thresholds": ("metric_name", "layer", "precision_half_width_rel", "practical_cv", "max_drift", "anomaly_rate_max", "practical_effect", "pilot_source", "frozen_at"),
    "e12_04.blocks": ("block_id", "day", "time_block", "candidate_order", "seed", "host_id", "device_ids", "schedule_version"),
    "e12_04.randomization": ("schedule_version", "seed", "design", "balance_metric", "notes"),
    "e12_04.telemetry_accelerator": ("run_id", "t_ns", "power_w", "energy_j", "temperature_c", "clock_mhz", "throttle_flags", "sampling_gap_s"),
    "e12_04.telemetry_system": ("run_id", "t_ns", "cpu_load", "ram_used", "swap_used", "io_wait", "network_bytes", "neighbor_processes"),
    "e12_04.clock_alignment": ("stream_id", "clock", "offset_ns", "error_bound_ns", "method", "status"),
    "e12_04.health": ("run_id", "phase", "device_errors", "link_state", "temperature_c", "power_cap_w", "background_processes", "healthy", "labels"),
    "e12_04.run_estimates": ("run_id", "cell_id", "metric_name", "value", "sample_count", "health_labels", "inclusion_status", "exclusion_rule_id"),
    "e12_04.injections": ("injection_id", "kind", "target", "control_parameters", "safety_limit", "expected_labels"),
    "e12_04.injection_results": ("injection_id", "detected", "observed_labels", "false_negative", "artifact_refs"),
    "e12_04.anomalies": ("event_id", "run_id", "label", "threshold", "observed_value", "rule_id", "rule_version", "detected_at"),
    "e12_04.exclusions": ("run_id", "rule_id", "rule_version", "detected_event", "threshold", "observed_value", "pre_registered", "decision", "performance_blind_basis", "reviewer", "artifact_refs", "effect_on_estimate", "sensitivity_result"),
    "e12_04.within_run": ("run_id", "metric_name", "median", "cv", "mad", "iqr", "autocorr", "sample_count"),
    "e12_04.variance_components": ("cell_id", "metric_name", "component", "estimate", "method", "block_count", "interval_low", "interval_high"),
    "e12_04.drift": ("cell_id", "metric_name", "covariate", "slope", "interval_low", "interval_high", "change_point", "rule_version"),
    "e12_04.sensitivity": ("cell_id", "metric_name", "scenario", "estimate", "interval_low", "interval_high", "depends_on_exclusion"),
    "e12_04.rank_frontier_stability": ("cell_id", "metric_name", "membership_probability", "rank_probability", "bootstrap_count", "seed", "threshold"),
    "e12_04.confirmation": ("cell_id", "metric_name", "confirmation_value", "interval_low", "interval_high", "protocol_deviation", "verdict"),
    "e12_04.verdicts": ("cell_id", "metric_name", "stability_verdict", "interval_low", "interval_high", "dominant_variation_source", "anomaly_rate", "evidence_level", "remeasure_condition"),
    # E12-05
    "e12_05.calibration_split": ("cell_id", "split", "reason", "frozen_at"),
    "e12_05.spec_sources": ("source_id", "resource", "value", "unit", "source_type", "url", "retrieved_at", "assumptions"),
    "e12_05.theoretical_peaks": ("peak_id", "platform_instance_id", "resource", "dtype", "value", "unit", "evidence_level", "assumptions", "source_id"),
    "e12_05.roof_samples": ("sample_id", "roof_id", "kind", "level", "dtype", "value", "unit", "temperature_c", "clock_mhz", "power_w", "achieved_plateau", "sample_count"),
    "e12_05.sustainable_estimates": ("roof_id", "platform_instance_id", "kind", "level", "dtype", "value", "unit", "interval_low", "interval_high", "evidence_level", "measured_at"),
    "e12_05.operations": ("ops_definition_id", "semantic_op", "formula", "dtype", "accumulation", "path", "notes"),
    "e12_05.traffic_logical": ("traffic_id", "cell_id", "level", "logical_bytes", "definition", "version"),
    "e12_05.traffic_measured": ("traffic_id", "cell_id", "level", "measured_bytes", "counter_source", "counter_semantics", "artifact_refs"),
    "e12_05.capacity_model": ("capacity_id", "candidate_id", "weights_bytes", "kv_bytes", "workspace_bytes", "allocator_overhead_bytes", "comm_bytes", "margin_bytes", "predicted_peak_bytes", "measured_peak_bytes", "error_bytes", "missing_reason"),
    "e12_05.roofline_points": ("point_id", "cell_id", "ai_logical", "ai_measured", "compute_roof_id", "bandwidth_roof_ids", "bound_latency_ns", "measured_latency_ns", "utilization_theoretical", "utilization_sustainable", "actual_backend", "quality_status", "stability_status"),
    "e12_05.phase_shares": ("cell_id", "phase", "component", "share", "source", "stability_status"),
    "e12_05.predictions": ("prediction_id", "model_version", "target_cell_id", "split", "operations_definition_id", "traffic_definition_id", "compute_roof_id", "bandwidth_roof_ids", "predicted_value", "unit", "interval_low", "interval_high", "residual_status"),
    "e12_05.service_capacity": ("cell_id", "service_time_s", "batch_efficiency", "kv_capacity", "queue_model", "predicted_arrival_max", "predicted_goodput", "assumptions", "interval_low", "interval_high"),
    "e12_05.distributed_critical_path": ("cell_id", "device_count", "parallel_plan_id", "compute_ns", "collective_ns", "wait_ns", "idle_ns", "overlap_ns", "predicted_runtime_ns", "measured_runtime_ns"),
    "e12_05.predicted_vs_measured": ("prediction_id", "measured_result_id", "predicted_value", "measured_value", "unit", "absolute_error", "relative_error", "log_error", "validation_split", "pass_threshold"),
    "e12_05.residuals": ("residual_id", "prediction_id", "class", "evidence_refs", "confidence", "resolved", "followup_ablation_id"),
    "e12_05.ablations": ("ablation_id", "cell_id", "changed_factor", "expected_shift", "observed_shift", "single_factor", "alternative_explanations", "status"),
    "e12_05.prediction_intervals": ("prediction_id", "method", "interval_low", "interval_high", "confidence_level", "inputs_uncertainty"),
    "e12_05.bottleneck_verdicts": ("verdict_id", "cell_id", "bottleneck_class", "evidence_refs", "headroom_rel", "predicted_upper_bound", "confidence", "forbidden_extrapolations"),
    # E12-06
    "e12_06.meters": ("meter_id", "platform_instance_id", "api_or_device", "firmware_version", "tool_version", "channel", "unit", "calibrated_at", "calibration_status", "task_mapping"),
    "e12_06.meter_capabilities": ("meter_id", "field", "semantics", "boundary", "unit", "resolution", "sample_period", "availability", "permission", "wrap_or_reset"),
    "e12_06.clock_alignment": ("stream_id", "offset_ns", "averaging_window_s", "method", "uncertainty_ns", "validated_at"),
    "e12_06.step_response": ("meter_id", "step_kind", "reported_delay_s", "method", "uncertainty_s"),
    "e12_06.accumulator_vs_integral": ("run_id", "meter_id", "accumulator_delta_j", "integral_j", "difference_j", "tolerance_j", "explained_by"),
    "e12_06.collector_overhead": ("meter_id", "collector_state", "latency_delta_rel", "cpu_delta_rel", "threshold", "action"),
    "e12_06.raw_power": ("run_id", "meter_id", "t_ns", "power_w", "energy_j", "status", "quality_flags"),
    "e12_06.workload_events": ("run_id", "t_ns", "event", "request_id", "token_count", "phase"),
    "e12_06.requests_tokens": ("run_id", "admitted", "completed", "failed", "backlog", "retried", "prompt_tokens", "generated_tokens", "accepted_tokens", "draft_tokens", "rejected_tokens", "compliant_requests", "compliant_tokens", "closed"),
    "e12_06.telemetry_flags": ("run_id", "t_ns", "flag", "detail", "repaired"),
    "e12_06.windows": ("window_id", "run_id", "policy", "raw_t0", "raw_t1", "aligned_t0", "aligned_t1", "gap_fraction", "sample_count"),
    "e12_06.energy_total": ("energy_measurement_id", "run_id", "meter_id", "boundary", "method", "total_energy_j", "sample_count", "gap_fraction", "uncertainty_low", "uncertainty_high", "quality_status"),
    "e12_06.energy_incremental": ("energy_measurement_id", "idle_power_w", "idle_estimator", "incremental_energy_j", "clip_policy", "idle_drift"),
    "e12_06.efficiency": ("energy_measurement_id", "j_per_request", "j_per_token", "tokens_per_j", "goodput_per_w", "denominator_requests", "denominator_tokens", "temperature_range", "clock_range", "throttle_flags"),
    "e12_06.repeatability": ("cell_id", "metric_name", "value", "cv", "interval_low", "interval_high", "process_count", "day_count", "meter_drift"),
    "e12_06.sensitivity": ("cell_id", "dimension", "value", "metric_name", "estimate", "notes"),
    "e12_06.verdicts": ("cell_id", "metric_name", "status", "boundary", "interval_low", "interval_high", "thermal_risk", "limitations", "allowed_for_cost"),
    # E12-07
    "e12_07.price_sources": ("price_source_id", "provider", "source_type", "url", "retrieved_at", "effective_from", "effective_to", "region", "currency", "tax_policy", "confidence"),
    "e12_07.price_snapshots": ("snapshot_id", "price_source_id", "sku", "instance", "included_resources", "purchase_model", "commitment", "billing_granularity", "unit_price", "unit", "content_hash", "snapshot_path"),
    "e12_07.price_normalized": ("snapshot_id", "price_source_id", "base_currency", "per_hour", "per_month", "fx_source_id", "normalized_at"),
    "e12_07.deployment_units": ("candidate_id", "deployment_unit", "device_count", "host_resources", "storage", "network", "mapping_evidence"),
    "e12_07.assumptions": ("assumption_id", "candidate_id", "kind", "value", "unit", "source", "range_low", "range_high"),
    "e12_07.qualified_points": ("cell_id", "candidate_id", "profile_id", "goodput", "slo_compliant", "stability_status", "quality_status", "source_result_id", "stranded_capacity_rel"),
    "e12_07.replica_plans": ("plan_id", "candidate_id", "profile_id", "demand", "replicas", "redundancy", "availability", "utilization", "stranded_capacity"),
    "e12_07.components": ("cost_result_id", "component", "value", "unit", "period", "source_ids", "formula_version"),
    "e12_07.cost_per": ("cost_result_id", "candidate_id", "profile_id", "scenario_id", "effective_date", "currency", "compliant_requests", "compliant_tokens", "cost_per_request", "cost_per_token", "cost_per_million_tokens", "total_cost", "period"),
    "e12_07.density": ("cost_result_id", "candidate_id", "metric", "value", "unit", "denominator", "missing_reason"),
    "e12_07.uncertainty": ("cost_result_id", "method", "interval_low", "interval_high", "confidence_level", "propagated_inputs"),
    "e12_07.sensitivity_one_way": ("cost_result_id", "parameter", "low_value", "high_value", "low_cost", "high_cost", "elasticity", "rank"),
    "e12_07.sensitivity_joint": ("cost_result_id", "scenario_id", "method", "assumptions", "cost_low", "cost_high", "rank_changes"),
    "e12_07.break_even": ("candidate_a", "candidate_b", "profile_id", "parameter", "threshold_value", "feasible_range", "direction"),
    "e12_07.double_count": ("audit_id", "check", "status", "detail", "affected_results"),
    "e12_07.recalculation": ("cost_result_id", "checker", "method", "recomputed_total", "difference", "unit_ok", "currency_ok", "period_ok", "status"),
    "e12_07.verdicts": ("cost_result_id", "candidate_id", "scenario_id", "status", "evidence_level", "main_sensitivity", "valid_until", "refresh_trigger", "limitations"),
    # E12-08
    "e12_08.profiles": ("profile_id", "version", "owner", "effective_date", "quality_gate_id", "cost_scenario_id", "risk_tolerance", "confidence_requirement"),
    "e12_08.workload_weights": ("profile_id", "workload_spec_id", "weight", "source", "validated"),
    "e12_08.constraints": ("profile_id", "constraint_id", "metric_name", "comparator", "threshold", "unit", "hard", "evidence_requirement"),
    "e12_08.objectives": ("profile_id", "objective_id", "metric_name", "direction", "unit", "aggregation", "practical_tolerance"),
    "e12_08.source_results": ("profile_id", "candidate_id", "metric_name", "value", "unit", "interval_low", "interval_high", "source_result_id", "evidence_level", "status"),
    "e12_08.evidence_gate": ("profile_id", "candidate_id", "status", "reasons", "comparability", "quality", "stability", "capability", "lineage"),
    "e12_08.feasibility": ("profile_id", "candidate_id", "status", "violated_constraints", "conditional_constraints", "reason"),
    "e12_08.constraint_slack": ("profile_id", "candidate_id", "constraint_id", "value", "threshold", "slack", "slack_rel", "borderline"),
    "e12_08.point_frontiers": ("profile_id", "candidate_id", "frontier_status", "objective_values", "dominated_by", "near_frontier", "membership_probability"),
    "e12_08.dominance_edges": ("profile_id", "dominator", "dominated", "non_worse_objectives", "strict_objectives", "source_result_ids"),
    "e12_08.near_frontiers": ("profile_id", "candidate_id", "distance", "tolerance", "source_result_ids"),
    "e12_08.conservative_frontiers": ("profile_id", "candidate_id", "quantile", "frontier_status", "objective_values", "differs_from_point"),
    "e12_08.membership": ("profile_id", "candidate_id", "feasible_probability", "frontier_probability", "selection_probability", "failure_constraint_frequency", "bootstrap_count", "seed"),
    "e12_08.price_utilization_sensitivity": ("profile_id", "scenario_id", "candidate_id", "frontier_status", "changed_from_point", "break_even_ref"),
    "e12_08.slo_demand_sensitivity": ("profile_id", "slo_variant", "candidate_id", "feasibility", "frontier_status", "region_notes"),
    "e12_08.workload_mix_sensitivity": ("profile_id", "variant", "candidate_id", "frontier_status", "dominant_workload", "leave_one_out_effect"),
    "e12_08.break_even_regions": ("profile_id", "candidate_a", "candidate_b", "parameter", "threshold", "action"),
    "e12_08.recommendations": (
        "recommendation_id", "profile_id", "version", "effective_date", "candidate_id", "role", "feasibility_status",
        "frontier_status", "membership_probability", "binding_constraints", "constraint_slacks", "objective_values",
        "quality_status", "comparability_status", "stability_status", "evidence_status", "cost_scenario_id",
        "energy_boundary", "maturity_record_ids", "selection_policy", "preference_assumptions", "risks",
        "limitations", "forbidden_claims", "refresh_triggers", "source_result_ids", "lineage_refs",
    ),
    "e12_08.reviewer_records": ("recommendation_id", "reviewer", "reviewed_at", "reproduced", "disagreements", "evidence_refs"),
    "e12_08.regression_tests": ("case_id", "scenario", "expected_outcome", "observed_outcome", "status"),
    # E12-09
    "e12_09.tasks": ("task_id", "dimension", "description", "inputs", "expected_output", "success_criteria", "time_budget_s", "allowed_help", "termination_condition", "applicability"),
    "e12_09.operators": ("operator_or_session_id", "experience_band", "platform_familiarity", "independent", "notes"),
    "e12_09.sessions": ("session_id", "candidate_id", "operator_or_session_id", "clean_or_warm", "task_id", "start", "end", "wall_seconds", "active_seconds", "blocked_seconds", "rework_seconds", "success_status", "correctness_status", "actual_backend", "failure_ids", "retry_count", "manual_step_count", "documentation_source_ids", "workaround_ids", "artifact_refs", "rubric_dimension_levels", "reviewer_ids", "limitations", "privacy_redaction_status"),
    "e12_09.events": ("event_id", "session_id", "t", "kind", "command_or_action", "exit_code", "manual_edit", "decision", "artifact_ref"),
    "e12_09.incidents": ("incident_id", "session_id", "category", "severity", "root_cause_confidence", "evidence_refs"),
    "e12_09.recovery_steps": ("incident_id", "step_index", "step", "duration_s", "successful", "evidence_ref"),
    "e12_09.workarounds": ("workaround_id", "kind", "owner", "upstream_issue", "first_version", "last_verified_version", "risk", "automation_status", "maintenance_hours", "removal_condition"),
    "e12_09.docs_sources": ("source_id", "candidate_id", "kind", "version", "url_or_ref", "used_in_session"),
    "e12_09.docs_gaps": ("gap_id", "candidate_id", "source_id", "kind", "detail", "severity", "evidence_ref"),
    "e12_09.tools_coverage": ("candidate_id", "area", "tool", "can_observe", "can_export", "can_correlate", "can_automate", "limitation"),
    "e12_09.upgrades": ("upgrade_id", "candidate_id", "component", "from_version", "to_version", "breaking_changes", "migration_hours", "regression_status", "rollback_verified"),
    "e12_09.task_metrics": ("session_id", "task_id", "success_status", "wall_seconds", "active_seconds", "blocked_seconds", "rework_seconds", "manual_step_count", "failure_count", "retry_count", "first_correct_inference_seconds"),
    "e12_09.rubric_anchors": ("dimension", "level", "anchor", "evidence_kind"),
    "e12_09.rubric_ratings": ("session_id", "task_id", "dimension", "level", "not_evaluated", "rationale", "reviewer", "evidence_refs"),
    "e12_09.agreement": ("dimension", "exact_agreement", "adjacent_agreement", "disputed_items", "resolution", "reviewer_count"),
    "e12_09.learning_curve": ("candidate_id", "task_id", "session_ordinal", "experience_band", "active_seconds", "failure_count", "manual_step_count"),
    "e12_09.maintenance": ("candidate_id", "component", "active_hours_low", "active_hours_high", "frequency", "basis", "workaround_ids"),
    "e12_09.verdicts": ("candidate_id", "dimension", "level", "coverage", "strengths", "gaps", "confidence", "forbidden_claims", "deployable_constraints"),
    # E12-10
    "e12_10.entities": ("entity_id", "entity_type", "logical_uri", "byte_hash", "canonical_hash", "aggregate_root", "size", "mime", "schema_version", "storage_locations", "created_at", "status"),
    "e12_10.activities": ("activity_id", "transform_type", "code_artifact_id", "environment_id", "entrypoint", "parameter_entity_id", "seed_policy", "input_entity_ids", "output_entity_ids", "started_at", "ended_at", "status", "log_entity_id"),
    "e12_10.edges": ("edge_id", "relation", "source_entity_id", "target_entity_id", "activity_id", "parameters_hash", "created_at", "validator_status"),
    "e12_10.storage_locations": ("entity_id", "kind", "location", "readable", "verified_at"),
    "e12_10.claims": ("claim_id", "text_ref", "experiment_id", "evidence_level", "status", "limitations", "source_result_ids", "figure_or_table_ids", "orphan"),
    "e12_10.schema_results": ("check_id", "entity_or_activity_id", "schema_name", "status", "detail"),
    "e12_10.hash_results": ("check_id", "entity_id", "expected_hash", "actual_hash", "kind", "status", "detail"),
    "e12_10.semantic_cross_checks": ("check_id", "subject_id", "constraint", "status", "detail"),
    "e12_10.dag_integrity": ("check_id", "kind", "status", "detail", "affected_ids"),
    "e12_10.dashboard_points": ("point_id", "view_or_query", "view_version", "filters", "normalized_result_ids", "format", "displayed_rounding", "label"),
    "e12_10.sample_plan": ("stratum", "population", "sampled", "selection_rule", "covers_critical_claims"),
    "e12_10.spot_checks": ("point_id", "stratum", "reverse_trace_complete", "regenerated", "diff_kind", "status", "detail"),
    "e12_10.reverse_trace": ("point_id", "hops", "query_time_s", "manual_steps", "missing_entities", "complete"),
    "e12_10.regeneration_diff": ("object_id", "kind", "diff_kind", "tolerance", "expected", "observed", "explanation", "policy_version"),
    "e12_10.fault_injection": ("case_id", "injection", "expected_locator", "observed_locator", "localised", "downstream_claims"),
    "e12_10.coverage": ("experiment_id", "claim_id", "point_id", "regeneration_level", "lineage_complete", "spot_check_status"),
    "e12_10.orphan_claims": ("claim_id", "reason", "affected_report", "action"),
    # campaign level
    "campaign.upstream_evidence": ("campaign_id", "upstream_stage", "experiment_id", "artifact_uri", "state", "reason", "verified_at"),
    "campaign.status_matrix": ("scope_id", "scope_kind", "experiment_status", "cell_status", "metric_status", "claim_status", "reason"),
    "campaign.acceptance": ("campaign_id", "experiment_status", "cell_status", "metric_status", "claim_status", "limitations", "written_at"),
}

# ── helpers ───────────────────────────────────────────────────────────────


def missing_fields(table: str, row: Mapping[str, Any]) -> List[str]:
    """Required columns absent from ``row`` (missing keys, not empty values)."""
    schema = TABLE_SCHEMAS.get(table)
    if schema is None:
        raise ConfigError(f"unknown S12 table {table!r}", details={"table": table})
    return [name for name in schema if name not in row]


def validate_table_row(table: str, row: Mapping[str, Any]) -> List[str]:
    """Validate one row: required columns present and no fake numeric missing."""
    problems: List[str] = []
    for name in missing_fields(table, row):
        problems.append(f"{table}: row is missing required field {name!r}")
    status = row.get("status") or row.get("missing_reason") or row.get("quality_status")
    if isinstance(status, str) and is_missing_status(status):
        for column in MEASURED_COLUMNS + ("value", "raw_value", "normalized_value"):
            if column in row and isinstance(row[column], (int, float)):
                problems.append(
                    f"{table}: missing state {status!r} must not carry a numeric {column!r}"
                )
    return problems


def is_missing_status(status: str) -> bool:
    return status in MISSINGNESS_CODES or status in (
        MISSING_QUALITY_GATE_FAILED,
        "NOT_RUN",
        "NOT_APPLICABLE",
        "UNMEASURABLE",
        "MISSING",
        "UNKNOWN",
    )


def numeric_value(
    value: Optional[float],
    status: str,
    *,
    field: str = "value",
) -> float:
    """Return ``value`` or raise when a missing state pretends to be a number.

    ``0`` is a valid measurement; *missing* is an evidence state.  Allowing a
    missing cell to default to ``0`` is exactly how "not supported" turns into
    "most power efficient" in a ranking, so this helper fails closed.
    """
    if is_missing_status(status) or value is None:
        raise ConfigError(
            f"{field} is unavailable (status={status or 'MISSING'}); missing data must not "
            f"be written as a number",
            details={"status": status, "field": field},
        )
    return float(value)


@dataclass
class StatusMatrix:
    """``experiment_status`` / ``cell_status`` / ``metric_status`` / ``claim_status``.

    The acceptance report of S12 must carry all four: a single global green
    tick would erase local failures (details README §3.2).
    """

    scope_id: str
    scope_kind: str = "campaign"
    experiment_status: str = STATUS_NOT_STARTED
    cell_status: str = "NOT_RUN"
    metric_status: str = "MISSING"
    claim_status: str = "NOT_PUBLISHABLE"
    reason: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.experiment_status not in PROTOCOL_STATUSES:
            problems.append(f"unknown experiment status {self.experiment_status!r}")
        if self.cell_status not in CELL_STATUSES:
            problems.append(f"unknown cell status {self.cell_status!r}")
        if self.metric_status not in METRIC_STATUSES:
            problems.append(f"unknown metric status {self.metric_status!r}")
        if self.claim_status not in CLAIM_STATUSES:
            problems.append(f"unknown claim status {self.claim_status!r}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "scope_kind": self.scope_kind,
            "experiment_status": self.experiment_status,
            "cell_status": self.cell_status,
            "metric_status": self.metric_status,
            "claim_status": self.claim_status,
            "reason": self.reason,
        }


def propagate_status(source_status: str) -> Optional[Mapping[str, str]]:
    """Look up one status-propagation rule by its source description."""
    for rule in STATUS_PROPAGATION:
        if rule["source_status"] == source_status:
            return rule
    return None


def table_names() -> Tuple[str, ...]:
    return tuple(sorted(TABLE_SCHEMAS))


def tables_for_experiment(experiment_id: str) -> Tuple[str, ...]:
    prefix = experiment_id.lower().replace("e12-", "e12_") + "."
    return tuple(name for name in sorted(TABLE_SCHEMAS) if name.startswith(prefix))


def rows_to_manifest(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Project table rows onto the manifest form used by aggregate roots."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "role": str(row.get("role", row.get("table", ""))),
                "path": str(row.get("path", "")),
                "byte_hash": str(row.get("byte_hash", row.get("sha256", ""))),
                "size": int(row.get("size", 0)),
            }
        )
    return out
