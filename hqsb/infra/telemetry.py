"""S13 telemetry projections: the §22 core schemas and table coverage.

The eleven S13 experiments all write the same *shapes* of evidence.  This module
gives those shapes a single home:

* projections from the S13 experiment records to the core schemas of
  ``details/S13/README.md`` §22 (``ReleaseIdentity``, ``DeploymentEvent``,
  ``RequestLifecycleEvent``, ``AdmissionDecision``, ``AutoscalingDecision``,
  ``FaultExperiment``, ``CanaryDecision``);
* :func:`validate_table_rows` / :func:`coverage_report`, which refuse a row that
  silently drops a required field and list the tables a run still has to fill;
* a :class:`ExperimentRecordProjection` that keeps the identity fields of the
  handbook record (``docs/stage_experiments/README.md`` §4) next to the S13 ones.

Nothing here writes raw data or emits a conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.infra import identity as idn
from hqsb.infra import records as rec

SCHEMA_VERSION = "1.0.0"

#: The core schemas of §22, in the order the README lists them.
CORE_SCHEMAS: Tuple[str, ...] = (
    "ReleaseIdentity",
    "DeploymentEvent",
    "RequestLifecycleEvent",
    "AdmissionDecision",
    "AutoscalingDecision",
    "FaultExperiment",
    "CanaryDecision",
)

#: Required fields per core schema (§22.1–§22.7).
CORE_SCHEMA_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "ReleaseIdentity": (
        "release_id", "source_commit", "dirty_patch_hash", "image_index_digest",
        "image_platform_digests", "model_artifact_id", "tokenizer_id", "config_hash",
        "deployment_digest", "sbom_ids", "provenance_id", "signature_id", "scan_ids",
        "created_at", "parent_release", "status",
    ),
    "DeploymentEvent": (
        "deployment_run_id", "event_id", "timestamp", "cluster_id", "node_id", "namespace",
        "workload_id", "pod_id", "container_id", "release_id", "desired_generation",
        "observed_generation", "state_from", "state_to", "reason", "controller", "artifact_refs",
    ),
    "RequestLifecycleEvent": (
        "request_id", "trace_id", "tenant_pseudonym", "release_id", "model_artifact_id",
        "backend_id", "pod_id", "node_id", "device_id", "rank_id", "state", "timestamp",
        "prompt_tokens", "output_tokens", "queue_seconds", "batch_size", "kv_bytes", "status",
        "error_id", "cancel_reason", "retry_count", "resource_release_status",
    ),
    "AdmissionDecision": (
        "decision_id", "request_id", "timestamp", "predicted_prompt_output_tokens",
        "predicted_KV_memory", "current_queue_token_work", "usable_memory", "safety_margin",
        "SLO_risk", "policy_version", "decision", "reason_code", "actual_outcome",
    ),
    "AutoscalingDecision": (
        "decision_id", "timestamp", "policy_version", "metric_values_and_ages", "desired_raw",
        "desired_stabilized", "current_replicas", "desired_replicas", "ready_replicas",
        "available_replicas", "scale_limit", "cooldown", "reason", "actual_ready_at",
        "slo_consequence", "cost_consequence",
    ),
    "FaultExperiment": (
        "fault_run_id", "hypothesis", "target", "mechanism", "blast_radius", "safety_policy",
        "start", "end", "expected_detection", "expected_degradation", "expected_recovery",
        "observed_MTTD", "observed_MTTM", "observed_MTTR", "sli_impact", "slo_impact",
        "error_budget_impact", "state_invariants", "resource_invariants", "cleanup", "verdict",
    ),
    "CanaryDecision": (
        "canary_run_id", "control_release", "candidate_release", "assignment_policy",
        "traffic_fraction", "exposure", "quality_metrics", "correctness_metrics", "error_metrics",
        "slo_metrics", "performance_metrics", "resource_metrics", "sequential_rule", "thresholds",
        "evidence_count", "decision", "reason", "rollback_id", "final_active_release",
    ),
}

#: Required subset of each core schema.  The remaining §22 fields are *optional in
#: principle* (a clean tree has no dirty-patch hash, the first release has no parent,
#: a hash policy may replace a signature) and are reported separately so an empty
#: optional field is visible without being treated as a missing identity.
REQUIRED_CORE_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "ReleaseIdentity": (
        "release_id", "source_commit", "image_index_digest", "image_platform_digests",
        "model_artifact_id", "tokenizer_id", "config_hash", "deployment_digest", "sbom_ids",
        "provenance_id", "status",
    ),
    "DeploymentEvent": (
        "deployment_run_id", "event_id", "cluster_id", "namespace", "release_id", "state_to",
    ),
    "RequestLifecycleEvent": (
        "request_id", "trace_id", "release_id", "model_artifact_id", "state", "timestamp",
    ),
    "AdmissionDecision": (
        "decision_id", "request_id", "policy_version", "decision", "reason_code",
    ),
    "AutoscalingDecision": (
        "decision_id", "policy_version", "metric_values_and_ages", "current_replicas",
        "desired_stabilized", "action",
    ),
    "FaultExperiment": (
        "fault_run_id", "hypothesis", "mechanism", "blast_radius", "safety_policy",
        "expected_recovery", "verdict",
    ),
    "CanaryDecision": (
        "canary_run_id", "control_release", "candidate_release", "decision",
    ),
}


def _partition_missing(schema: str, payload: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """Split empty fields into ``(required_missing, optional_missing)``."""
    fields = CORE_SCHEMA_FIELDS[schema]
    required = set(REQUIRED_CORE_FIELDS.get(schema, fields))
    required_missing: List[str] = []
    optional_missing: List[str] = []
    for name in fields:
        if payload.get(name) not in ("", None, [], {}):
            continue
        (required_missing if name in required else optional_missing).append(name)
    return required_missing, optional_missing


#: Which experiments are expected to fill which table (used by the coverage audit).
TABLE_OWNERS: Mapping[str, Tuple[str, ...]] = {
    "release_bundle": ("E13-01",),
    "oci_dag_diff": ("E13-01",),
    "filesystem_package_diff": ("E13-01",),
    "nondeterminism": ("E13-01",),
    "sbom_component": ("E13-01",),
    "vulnerability_finding": ("E13-01",),
    "vulnerability_exception": ("E13-01",),
    "secret_finding": ("E13-01",),
    "license_finding": ("E13-01",),
    "provenance_statement": ("E13-01",),
    "attestation_verification": ("E13-01",),
    "runtime_security_context": ("E13-01",),
    "negative_test_result": ("E13-01", "E13-02", "E13-03", "E13-04", "E13-05"),
    "deployment_stage_event": ("E13-02",),
    "precondition_check": ("E13-02",),
    "pre_ready_attempt": ("E13-02",),
    "first_request_event": ("E13-02",),
    "cold_warm_stage_time": ("E13-02", "E13-07"),
    "residual_inventory": ("E13-02",),
    "manual_intervention": ("E13-02", "E13-05", "E13-09"),
    "placement_plan": ("E13-03",),
    "capability_label": ("E13-03",),
    "scheduler_event": ("E13-03",),
    "placement_evidence": ("E13-03",),
    "topology_cpu_memory_numa": ("E13-03",),
    "unauthorized_access": ("E13-03", "E13-11"),
    "artifact_lifecycle_event": ("E13-04",),
    "cache_key": ("E13-04",),
    "verification_result": ("E13-04", "E13-09"),
    "compatibility_result": ("E13-04",),
    "lease_record": ("E13-04",),
    "activation_generation": ("E13-04", "E13-10"),
    "request_version_consistency": ("E13-04", "E13-10"),
    "gc_plan": ("E13-04",),
    "rollback_timeline": ("E13-04", "E13-10"),
    "probe_result": ("E13-05",),
    "signal_hook_exit": ("E13-05",),
    "request_lifecycle_event": ("E13-05", "E13-06", "E13-08"),
    "token_integrity": ("E13-05", "E13-10"),
    "retry_duplicate": ("E13-05", "E13-09"),
    "drain_timeline": ("E13-05",),
    "resource_release": ("E13-05", "E13-09"),
    "rolling_availability": ("E13-05", "E13-07"),
    "memory_ledger": ("E13-06",),
    "per_request_kv": ("E13-06",),
    "admission_decision": ("E13-06",),
    "admission_policy_comparison": ("E13-06",),
    "prediction_residual": ("E13-06",),
    "false_decision": ("E13-06",),
    "margin_validation": ("E13-06",),
    "metric_sample": ("E13-07", "E13-08"),
    "autoscaling_decision": ("E13-07",),
    "replica_lifecycle": ("E13-07",),
    "control_metrics": ("E13-07",),
    "cost_resource": ("E13-07",),
    "semantic_convention": ("E13-08",),
    "metric_catalog_entry": ("E13-08",),
    "cardinality_report": ("E13-08",),
    "trace_coverage": ("E13-08",),
    "alert_rule": ("E13-08",),
    "alert_result": ("E13-08",),
    "overhead_ab": ("E13-08",),
    "rca_record": ("E13-08",),
    "telemetry_failure": ("E13-08",),
    "redaction_scan": ("E13-08", "E13-11"),
    "fault_spec": ("E13-09",),
    "fault_injection_event": ("E13-09",),
    "fault_ground_truth": ("E13-09",),
    "reliability_metrics": ("E13-09",),
    "retry_fallback": ("E13-09",),
    "state_diff": ("E13-09",),
    "residual_watch": ("E13-09",),
    "postmortem": ("E13-09",),
    "canary_candidate_identity": ("E13-10",),
    "canary_transition": ("E13-10",),
    "canary_assignment": ("E13-10",),
    "canary_metric": ("E13-10",),
    "canary_decision_row": ("E13-10",),
    "canary_gate_result": ("E13-10",),
    "canary_rollback": ("E13-10",),
    "canary_error_rates": ("E13-10",),
    "override_audit": ("E13-10",),
    "threat_model": ("E13-11",),
    "tenant_identity_map": ("E13-11",),
    "rbac_graph": ("E13-11",),
    "security_case": ("E13-11",),
    "quota_ledger": ("E13-11",),
    "quota_race_trial": ("E13-11",),
    "abuse_trial": ("E13-11",),
    "noisy_neighbor_trial": ("E13-11",),
    "telemetry_redaction": ("E13-11",),
    "audit_coverage": ("E13-11",),
    "recovery_state": ("E13-11",),
}

#: Handbook §4 record fields that every S13 experiment record must keep.
HANDBOOK_RECORD_FIELDS: Tuple[str, ...] = (
    "stage", "experiment_id", "status", "question", "hypothesis", "run_id", "git_commit",
    "git_dirty", "model_manifest_sha256", "config_sha256", "operator_or_binary_sha256",
    "environment_uri", "hardware_and_power_mode", "requested_implementation",
    "actual_implementation", "controls", "independent_variables", "correctness_metrics",
    "performance_samples_uri", "profile_artifacts_uri", "started_at", "ended_at", "decision",
    "limitations",
)

#: S13 additions to the handbook record (identity + evidence + governance).
S13_RECORD_FIELDS: Tuple[str, ...] = (
    "release_id",
    "image_index_digest",
    "cluster_id",
    "namespace",
    "model_artifact_id",
    "observability_schema_version",
    "capacity_policy_version",
    "autoscaling_policy_version",
    "canary_policy_id",
    "fault_case_ids",
    "tenant_scope",
    "evidence_level",
    "evidence_axis_levels",
    "safety_policy_id",
    "abort_events",
    "manual_interventions",
    "allowed_claims",
    "forbidden_claims",
)


def validate_table_rows(table: str, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Thin wrapper so callers do not import ``records`` directly."""
    return rec.validate_rows(table, rows)


def coverage_report(counts: Mapping[str, int]) -> Dict[str, Any]:
    """Which tables a run has filled; the empty ones are listed, never implied."""
    unknown = sorted(set(counts) - set(rec.TABLE_SCHEMAS))
    if unknown:
        raise ConfigError(f"unknown S13 tables in the coverage report: {unknown}")
    empty = sorted(name for name in rec.TABLE_SCHEMAS if counts.get(name, 0) == 0)
    by_experiment: Dict[str, List[str]] = {}
    for table, owners in TABLE_OWNERS.items():
        for owner in owners:
            by_experiment.setdefault(owner, []).append(table)
    return {
        "tables": len(rec.TABLE_SCHEMAS),
        "filled": len([name for name, value in counts.items() if value]),
        "empty": empty,
        "by_experiment": {key: sorted(value) for key, value in sorted(by_experiment.items())},
        "note": "an empty table list is the honest state before the corresponding experiment runs",
    }


def experiment_tables(experiment_id: str) -> Tuple[str, ...]:
    """The tables one experiment is expected to fill (empty table ⇒ not yet run)."""
    if experiment_id not in {owner for owners in TABLE_OWNERS.values() for owner in owners}:
        raise ConfigError(f"unknown experiment id {experiment_id!r}")
    return tuple(sorted(table for table, owners in TABLE_OWNERS.items() if experiment_id in owners))


# ── projections ──────────────────────────────────────────────────────────


def project_release_identity(bundle: Any, *, config_hash: str = "", sbom_ids: Sequence[str] = (),
                             provenance_id: str = "", signature_id: str = "",
                             scan_ids: Sequence[str] = (), created_at: str = "",
                             parent_release: str = "") -> Dict[str, Any]:
    """§22.1 ``ReleaseIdentity`` projection from a :class:`ReleaseBundle`."""
    payload = {
        "release_id": bundle.release_id,
        "source_commit": bundle.source_commit,
        "dirty_patch_hash": bundle.dirty_patch_hash,
        "image_index_digest": bundle.image_index_digest,
        "image_platform_digests": dict(bundle.platform_image_digests),
        "model_artifact_id": bundle.model_artifact_id,
        "tokenizer_id": bundle.tokenizer_id,
        "config_hash": config_hash or bundle.service_config_hash,
        "deployment_digest": bundle.deployment_template_digest,
        "sbom_ids": list(sbom_ids or bundle.sbom_ids),
        "provenance_id": provenance_id or bundle.build_provenance_id,
        "signature_id": signature_id,
        "scan_ids": list(scan_ids),
        "created_at": created_at or bundle.created_at,
        "parent_release": parent_release or bundle.rollback_parent_release_id,
        "status": bundle.status,
    }
    missing, optional = _partition_missing("ReleaseIdentity", payload)
    return {
        "row": payload,
        "missing": missing,
        "optional_missing": optional,
        "complete": not missing,
    }


def project_deployment_event(event: Any) -> Dict[str, Any]:
    """§22.2 ``DeploymentEvent`` projection from a stage event."""
    payload = {
        "deployment_run_id": event.deployment_run_id,
        "event_id": event.event_id,
        "timestamp": event.started_at,
        "cluster_id": event.cluster_id,
        "node_id": event.node_id,
        "namespace": event.namespace,
        "workload_id": event.stage,
        "pod_id": event.pod_id,
        "container_id": event.container_id,
        "release_id": event.release_id,
        "desired_generation": event.desired_generation,
        "observed_generation": event.observed_generation,
        "state_from": event.state_from,
        "state_to": event.state_to,
        "reason": event.reason_code,
        "controller": event.stage,
        "artifact_refs": list(event.artifact_refs),
    }
    missing, optional = _partition_missing("DeploymentEvent", payload)
    return {"row": payload, "missing": missing, "optional_missing": optional, "complete": not missing}


def project_request_lifecycle_event(*, request_id: str, trace_id: str, state: str, timestamp: float,
                                    tenant_pseudonym: str, release_id: str, model_artifact_id: str,
                                    **extra: Any) -> Dict[str, Any]:
    """§22.3 ``RequestLifecycleEvent`` (high-cardinality identity belongs here, not in metrics)."""
    payload: Dict[str, Any] = {
        "request_id": request_id,
        "trace_id": trace_id,
        "tenant_pseudonym": tenant_pseudonym,
        "release_id": release_id,
        "model_artifact_id": model_artifact_id,
        "backend_id": extra.get("backend_id", ""),
        "pod_id": extra.get("pod_id", ""),
        "node_id": extra.get("node_id", ""),
        "device_id": extra.get("device_id", ""),
        "rank_id": extra.get("rank_id", ""),
        "state": state,
        "timestamp": timestamp,
        "prompt_tokens": extra.get("prompt_tokens", ""),
        "output_tokens": extra.get("output_tokens", ""),
        "queue_seconds": extra.get("queue_seconds", ""),
        "batch_size": extra.get("batch_size", ""),
        "kv_bytes": extra.get("kv_bytes", ""),
        "status": extra.get("status", ""),
        "error_id": extra.get("error_id", ""),
        "cancel_reason": extra.get("cancel_reason", ""),
        "retry_count": extra.get("retry_count", ""),
        "resource_release_status": extra.get("resource_release_status", ""),
    }
    missing, optional = _partition_missing("RequestLifecycleEvent", payload)
    return {"row": payload, "missing": missing, "optional_missing": optional, "complete": not missing}


def project_admission_decision(decision: Any) -> Dict[str, Any]:
    """§22.4 ``AdmissionDecision`` projection."""
    payload = {
        "decision_id": decision.decision_id,
        "request_id": decision.request_id,
        "timestamp": decision.timestamp,
        "predicted_prompt_output_tokens": decision.prompt_tokens + decision.requested_output_tokens,
        "predicted_KV_memory": decision.predicted_incremental_kv_bytes,
        "current_queue_token_work": decision.queue_tokens,
        "usable_memory": decision.usable_memory_bytes,
        "safety_margin": decision.safety_margin_bytes,
        "SLO_risk": decision.predicted_slo_risk,
        "policy_version": decision.policy_version,
        "decision": decision.decision,
        "reason_code": decision.reason_code,
        "actual_outcome": decision.outcome,
    }
    missing, optional = _partition_missing("AdmissionDecision", payload)
    return {"row": payload, "missing": missing, "optional_missing": optional, "complete": not missing}


def project_autoscaling_decision(decision: Any) -> Dict[str, Any]:
    """§22.5 ``AutoscalingDecision`` projection."""
    payload = {
        "decision_id": decision.decision_id,
        "timestamp": decision.timestamp,
        "policy_version": decision.policy_version,
        "metric_values_and_ages": {
            metric: {"value": value, "age_s": decision.metric_ages.get(metric, "")}
            for metric, value in decision.metric_name_values.items()
        },
        "desired_raw": decision.desired_raw,
        "desired_stabilized": decision.desired_stabilized,
        "current_replicas": decision.current_replicas,
        "desired_replicas": decision.desired_stabilized,
        "ready_replicas": decision.ready_replicas,
        "available_replicas": decision.available_replicas,
        "scale_limit": decision.rate_limit,
        "cooldown": decision.cooldown_s,
        "reason": decision.reason,
        "actual_ready_at": decision.ready_at,
        "slo_consequence": decision.slo_consequence,
        "cost_consequence": "",
    }
    missing, optional = _partition_missing("AutoscalingDecision", payload)
    return {"row": payload, "missing": missing, "optional_missing": optional, "complete": not missing}


def project_fault_experiment(*, spec: Any, episode: Any, invariants: Mapping[str, Any]) -> Dict[str, Any]:
    """§22.6 ``FaultExperiment`` projection."""
    payload = {
        "fault_run_id": episode.episode_id,
        "hypothesis": spec.hypothesis,
        "target": ", ".join(spec.resolved_targets),
        "mechanism": spec.mechanism,
        "blast_radius": spec.blast_radius,
        "safety_policy": spec.safety_policy_id,
        "start": episode.effective_ts,
        "end": episode.end_ts,
        "expected_detection": spec.expected_detection,
        "expected_degradation": spec.expected_degradation,
        "expected_recovery": spec.expected_recovery,
        "observed_MTTD": episode.mttd(),
        "observed_MTTM": episode.mttm(),
        "observed_MTTR": episode.mttr(),
        "sli_impact": episode.slo_impact,
        "slo_impact": episode.slo_impact,
        "error_budget_impact": episode.error_budget_spent,
        "state_invariants": dict(sorted(invariants.get("state", {}).items())),
        "resource_invariants": dict(sorted(invariants.get("resource", {}).items())),
        "cleanup": episode.cleanup_status,
        "verdict": episode.verdict,
    }
    missing, optional = _partition_missing("FaultExperiment", payload)
    return {"row": payload, "missing": missing, "optional_missing": optional, "complete": not missing}


def project_canary_decision(row: Any, *, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """§22.7 ``CanaryDecision`` projection."""
    payload = {
        "canary_run_id": row.canary_run_id,
        "control_release": row.control_release_id,
        "candidate_release": row.candidate_release_id,
        "assignment_policy": row.assignment_policy,
        "traffic_fraction": row.traffic_fraction,
        "exposure": row.eligible_count,
        "quality_metrics": metrics.get("quality", {}),
        "correctness_metrics": metrics.get("correctness", {}),
        "error_metrics": metrics.get("error", {}),
        "slo_metrics": metrics.get("slo", {}),
        "performance_metrics": metrics.get("performance", {}),
        "resource_metrics": metrics.get("resource", {}),
        "sequential_rule": row.sequential_boundary,
        "thresholds": metrics.get("thresholds", {}),
        "evidence_count": row.eligible_count,
        "decision": row.decision,
        "reason": ", ".join(row.reason_codes),
        "rollback_id": row.rollback_id,
        "final_active_release": row.final_active_release,
    }
    missing, optional = _partition_missing("CanaryDecision", payload)
    return {"row": payload, "missing": missing, "optional_missing": optional, "complete": not missing}


@dataclass
class ExperimentRecordProjection:
    """Handbook §4 fields plus the S13 identity/governance additions."""

    experiment_id: str
    record: Mapping[str, Any] = field(default_factory=dict)

    def missing_fields(self) -> List[str]:
        required = HANDBOOK_RECORD_FIELDS + S13_RECORD_FIELDS
        return [name for name in required if self.record.get(name) in ("", None, [], {})]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "fields": dict(sorted(self.record.items())),
            "missing": self.missing_fields(),
            "complete": not self.missing_fields(),
        }


def evidence_axis_levels(levels: Mapping[str, str]) -> Dict[str, Any]:
    """§19 evidence levels per axis; an axis may not inherit another axis's level."""
    unknown = {axis: level for axis, level in levels.items() if level not in rec.EVIDENCE_LEVELS}
    if unknown:
        raise ConfigError(f"unknown evidence levels: {sorted(unknown.items())}")
    capped: List[str] = []
    for axis, level in levels.items():
        if rec.EVIDENCE_RANK[level] > rec.EVIDENCE_RANK[rec.MAX_EVIDENCE_WITHOUT_PRODUCTION]:
            capped.append(axis)
    return {
        "axes": dict(sorted(levels.items())),
        "requires_production_evidence": capped,
        "note": (
            "a test cluster campaign reaches at most FAULT_VALIDATED; INDEPENDENTLY_REPRODUCED and "
            "PRODUCTION_OBSERVED need real production observation"
        ),
    }


def identity_closure_check(rows: Sequence[Mapping[str, Any]], *, required: Sequence[str]) -> Dict[str, Any]:
    """Every report row must resolve release/model/cluster/namespace identity."""
    problems: List[str] = []
    for index, row in enumerate(rows):
        for field_name in required:
            if row.get(field_name) in ("", None, "unknown"):
                problems.append(f"row {index}: {field_name} is unresolved")
    return {"rows": len(rows), "problems": problems, "ok": not problems}


def schema_audit() -> Dict[str, Any]:
    """Static audit of the projection definitions (no data needed)."""
    problems: List[str] = []
    for schema, fields in CORE_SCHEMA_FIELDS.items():
        duplicates = sorted({name for name in fields if fields.count(name) > 1})
        if duplicates:
            problems.append(f"{schema}: duplicate field names {duplicates}")
    missing_owners = sorted(set(rec.TABLE_SCHEMAS) - set(TABLE_OWNERS))
    if missing_owners:
        problems.append(f"tables without an owning experiment: {missing_owners}")
    unknown_owners = sorted(set(TABLE_OWNERS) - set(rec.TABLE_SCHEMAS))
    if unknown_owners:
        problems.append(f"table owners reference unknown tables: {unknown_owners}")
    return {
        "schemas": len(CORE_SCHEMA_FIELDS),
        "tables": len(rec.TABLE_SCHEMAS),
        "problems": problems,
        "ok": not problems,
    }


#: Logical URI helper for evidence references (mirrors the S12 convention).
def logical_uri(campaign_id: str, kind: str, name: str) -> str:
    return f"s13://{campaign_id}/{kind}/{name}"


def hash_row(row: Mapping[str, Any]) -> str:
    """Content hash of one evidence row (so a report point can cite it)."""
    return idn.hash_payload(dict(row))


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the projection layer (smoke, not an experiment)."""
    checks: Dict[str, Any] = {}
    audit = schema_audit()
    checks["schema_audit_ok"] = audit["ok"] is True

    report = coverage_report({})
    checks["empty_coverage_is_honest"] = report["filled"] == 0 and len(report["empty"]) == len(rec.TABLE_SCHEMAS)

    tables = experiment_tables("E13-01")
    checks["experiment_tables_resolved"] = "release_bundle" in tables and "security_case" not in tables

    levels = evidence_axis_levels({"supply_chain": rec.EVIDENCE_IMPLEMENTED_UNVERIFIED})
    checks["evidence_levels_validated"] = levels["axes"]["supply_chain"] == rec.EVIDENCE_IMPLEMENTED_UNVERIFIED

    closure = identity_closure_check(
        [{"release_id": "r1", "model_artifact_id": "unknown", "cluster_id": "c1"}],
        required=("release_id", "model_artifact_id", "cluster_id"),
    )
    checks["identity_closure_detects_unknown"] = closure["ok"] is False
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": "S13",
        "checks": checks,
        "note": "接口自检；未写入任何 raw 数据",
    }
