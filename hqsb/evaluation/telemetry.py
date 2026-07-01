"""C6/C7 projection, table schemas and coverage audit for S12.

S12 produces *cross-hardware* evidence that must land back in the shared
contracts: ``C6 BenchmarkResult`` (identity, raw samples, correctness, resource,
selected backend) and ``C7 TraceEvent`` (what happened during one request).
Unlike S11, an S12 result row additionally has to carry the *comparison*
identity, the capability/comparability/stability statuses and — when relevant —
the meter boundary, the price source ids and the lineage root; otherwise the
"cross-platform" claim has nothing to stand on.

An incomplete field is reported as missing, never fabricated, and a row whose
``status`` is a missing state may not carry a numeric payload (that is how
"unsupported" silently becomes "most power efficient").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.identity import canonical_json, sha256_text
from hqsb.evaluation.records import (
    CLAIM_STATUSES,
    COMPARABILITY_STATES,
    MISSINGNESS_CODES,
    RESULT_CLASSES,
    TABLE_SCHEMAS,
    is_missing_status,
)

# ── additional raw tables produced by S12 ─────────────────────────────────

DISPATCH_ROW_FIELDS: Tuple[str, ...] = (
    "dispatch_id",
    "cell_id",
    "candidate_id",
    "layer",
    "requested_backend",
    "eligible_backends",
    "selected_backend",
    "actual_backend",
    "actual_candidate_id",
    "fallback_reason",
    "artifact_hash",
    "evidence_kinds",
    "trace_id",
)

CAPABILITY_EVIDENCE_ROW_FIELDS: Tuple[str, ...] = (
    "evidence_id",
    "platform_instance_id",
    "feature_id",
    "declared_status",
    "discovered_status",
    "verified_status",
    "benchmarked_status",
    "probe_spec_hash",
    "requested_backend",
    "actual_backend",
    "correctness_status",
    "failure_category",
    "reason",
    "valid_until",
)

NORMALIZED_METRIC_ROW_FIELDS: Tuple[str, ...] = (
    "normalized_result_id",
    "cell_id",
    "candidate_id",
    "layer",
    "metric_name",
    "value",
    "unit",
    "direction",
    "estimator",
    "interval_low",
    "interval_high",
    "confidence_level",
    "replication_unit",
    "normalization_basis",
    "denominator_value",
    "comparability_status",
    "quality_status",
    "stability_status",
    "evidence_level",
    "missing_reason",
    "source_observation_ids",
)

ENERGY_ROW_FIELDS: Tuple[str, ...] = (
    "energy_measurement_id",
    "run_id",
    "cell_id",
    "meter_id",
    "boundary",
    "window_policy_id",
    "method",
    "total_energy_j",
    "incremental_energy_j",
    "idle_power_w",
    "compliant_requests",
    "compliant_tokens",
    "j_per_request",
    "j_per_token",
    "tokens_per_j",
    "goodput_per_w",
    "uncertainty_low",
    "uncertainty_high",
    "quality_status",
    "artifact_refs",
)

COST_ROW_FIELDS: Tuple[str, ...] = (
    "cost_result_id",
    "candidate_id",
    "profile_id",
    "scenario_id",
    "effective_date",
    "period",
    "region",
    "currency",
    "deployment_unit",
    "replicas",
    "utilization",
    "qualified_goodput_result_id",
    "energy_result_id",
    "price_source_ids",
    "assumption_ids",
    "total_cost",
    "compliant_requests",
    "compliant_tokens",
    "cost_per_request",
    "cost_per_token",
    "cost_per_million_tokens",
    "interval_low",
    "interval_high",
    "evidence_level",
    "limitations",
)

RECOMMENDATION_ROW_FIELDS: Tuple[str, ...] = (
    "recommendation_id",
    "profile_id",
    "version",
    "effective_date",
    "candidate_id",
    "role",
    "feasibility_status",
    "frontier_status",
    "membership_probability",
    "binding_constraints",
    "objective_values",
    "quality_status",
    "comparability_status",
    "stability_status",
    "evidence_status",
    "cost_scenario_id",
    "energy_boundary",
    "selection_policy",
    "risks",
    "limitations",
    "forbidden_claims",
    "refresh_triggers",
    "source_result_ids",
    "lineage_refs",
)

LINEAGE_EDGE_ROW_FIELDS: Tuple[str, ...] = (
    "edge_id",
    "relation",
    "source_entity_id",
    "target_entity_id",
    "activity_id",
    "parameters_hash",
    "created_at",
    "validator_status",
)

#: All raw tables S12 writes, with their minimal field lists.
S12_TABLE_SCHEMAS: Mapping[str, Tuple[str, ...]] = {
    **TABLE_SCHEMAS,
    "s12_dispatch_row": DISPATCH_ROW_FIELDS,
    "s12_capability_evidence_row": CAPABILITY_EVIDENCE_ROW_FIELDS,
    "s12_normalized_metric_row": NORMALIZED_METRIC_ROW_FIELDS,
    "s12_energy_row": ENERGY_ROW_FIELDS,
    "s12_cost_row": COST_ROW_FIELDS,
    "s12_recommendation_row": RECOMMENDATION_ROW_FIELDS,
    "s12_lineage_edge_row": LINEAGE_EDGE_ROW_FIELDS,
}


def validate_table_row(table: str, row: Mapping[str, Any]) -> List[str]:
    """Validate a row: unknown keys, missing keys and fake numeric missing."""
    if table not in S12_TABLE_SCHEMAS:
        raise ConfigError(f"unknown table {table!r}; known: {sorted(S12_TABLE_SCHEMAS)}")
    required = set(S12_TABLE_SCHEMAS[table])
    problems = sorted(f"{table}: unknown key {key!r}" for key in set(row) - required)
    problems.extend(f"{table}: missing key {key!r}" for key in sorted(required - set(row)))
    status = row.get("status") or row.get("missing_reason") or row.get("quality_status")
    if isinstance(status, str) and is_missing_status(status):
        for column in ("value", "measured_value", "raw_value", "normalized_value", "latency_ns"):
            if isinstance(row.get(column), (int, float)):
                problems.append(
                    f"{table}: missing state {status!r} must not carry a numeric {column!r}"
                )
    return problems


def missing_fields(table: str, row: Mapping[str, Any]) -> List[str]:
    if table not in S12_TABLE_SCHEMAS:
        raise ConfigError(f"unknown table {table!r}")
    return sorted(set(S12_TABLE_SCHEMAS[table]) - set(row))


def table_hashes(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, str]:
    """Content hash per table (used in manifests to bind raw evidence)."""
    return {
        name: sha256_text(canonical_json([dict(sorted(row.items())) for row in rows]))
        for name, rows in sorted(tables.items())
    }


def coverage_summary(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """How much of each table is actually filled (never silent about gaps)."""
    rows: List[Dict[str, Any]] = []
    for name, table_rows in sorted(tables.items()):
        schema = S12_TABLE_SCHEMAS.get(name)
        if schema is None:
            raise ConfigError(f"unknown table {name!r} in coverage summary")
        empty_cells = 0
        total_cells = 0
        for row in table_rows:
            for column in schema:
                total_cells += 1
                if row.get(column) in (None, "", [], {}):
                    empty_cells += 1
        rows.append(
            {
                "table": name,
                "rows": len(table_rows),
                "columns": len(schema),
                "empty_cells": empty_cells,
                "fill_ratio": (1.0 - empty_cells / total_cells) if total_cells else 0.0,
            }
        )
    return {"tables": rows, "total_rows": sum(row["rows"] for row in rows)}


# ── C6 projection ──────────────────────────────────────────────────────────


@dataclass
class S12ResultFields:
    """The S12-specific part of a C6 result row (E12-03 step 35)."""

    comparison_id: str = ""
    candidate_id: str = ""
    platform_instance_id: str = ""
    layer: str = ""
    scenario: str = ""
    workload_spec_id: str = ""
    measurement_boundary_id: str = ""
    requested_backend: str = ""
    actual_backend: str = ""
    fallback_reason: str = ""
    quality_status: str = "not_run"
    comparability_status: str = ""
    capability_status: str = ""
    stability_status: str = ""
    evidence_level: str = "L0"
    result_class: str = "MEASURED"
    regeneration_level: str = "R0"
    meter_boundary: str = ""
    energy_measurement_id: str = ""
    cost_result_id: str = ""
    price_source_ids: Tuple[str, ...] = ()
    maturity_record_ids: Tuple[str, ...] = ()
    lineage_root: str = ""
    trace_id: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("comparison_id", "candidate_id", "layer"):
            if not getattr(self, name):
                problems.append(f"{name} is required for a cross-hardware result row")
        if self.actual_backend and self.requested_backend and self.actual_backend != self.requested_backend:
            if not self.fallback_reason:
                problems.append(
                    "actual != requested requires a fallback reason (no silent fallback)"
                )
        if self.comparability_status and self.comparability_status not in COMPARABILITY_STATES:
            problems.append(f"unknown comparability status {self.comparability_status!r}")
        if self.result_class not in RESULT_CLASSES:
            problems.append(f"unknown result class {self.result_class!r}")
        if self.result_class == "MEASURED" and isinstance(self.quality_status, str) and is_missing_status(self.quality_status):
            problems.append(
                f"a MEASURED result may not carry quality status {self.quality_status!r}: "
                "quality precedes performance"
            )
        if self.energy_measurement_id and not self.meter_boundary:
            problems.append("an energy result must state its meter boundary")
        if self.cost_result_id and not self.price_source_ids:
            problems.append("a cost result must reference at least one price source")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "candidate_id": self.candidate_id,
            "platform_instance_id": self.platform_instance_id,
            "layer": self.layer,
            "scenario": self.scenario,
            "workload_spec_id": self.workload_spec_id,
            "measurement_boundary_id": self.measurement_boundary_id,
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "fallback_reason": self.fallback_reason,
            "quality_status": self.quality_status,
            "comparability_status": self.comparability_status,
            "capability_status": self.capability_status,
            "stability_status": self.stability_status,
            "evidence_level": self.evidence_level,
            "result_class": self.result_class,
            "regeneration_level": self.regeneration_level,
            "meter_boundary": self.meter_boundary,
            "energy_measurement_id": self.energy_measurement_id,
            "cost_result_id": self.cost_result_id,
            "price_source_ids": list(self.price_source_ids),
            "maturity_record_ids": list(self.maturity_record_ids),
            "lineage_root": self.lineage_root,
            "trace_id": self.trace_id,
        }


def project_c6(fields: S12ResultFields) -> Dict[str, Any]:
    """Map S12 fields onto the C6 result projection with a coverage statement."""
    problems = fields.validate()
    projected = {
        "run_id": fields.comparison_id,
        "candidate_id": fields.candidate_id,
        "platform_instance_id": fields.platform_instance_id,
        "layer": fields.layer,
        "scenario": fields.scenario,
        "workload_spec_id": fields.workload_spec_id,
        "measurement_boundary_id": fields.measurement_boundary_id,
        "requested_implementation": fields.requested_backend,
        "actual_implementation": fields.actual_backend,
        "fallback_reason": fields.fallback_reason,
        "quality_status": fields.quality_status,
        "comparability_status": fields.comparability_status,
        "capability_status": fields.capability_status,
        "stability_status": fields.stability_status,
        "evidence_level": fields.evidence_level,
        "result_class": fields.result_class,
        "regeneration_level": fields.regeneration_level,
        "meter_boundary": fields.meter_boundary,
        "energy_measurement_id": fields.energy_measurement_id,
        "cost_result_id": fields.cost_result_id,
        "lineage_root": fields.lineage_root,
    }
    filled = sorted(key for key, value in projected.items() if value not in (None, "", [], {}))
    return {
        "fields": projected,
        "filled_fields": filled,
        "empty_fields": sorted(set(projected) - set(filled)),
        "problems": problems,
        "ok": not problems,
    }


# ── C7 projection ──────────────────────────────────────────────────────────

C7_KIND_MAP: Mapping[str, str] = {
    "comparability_verdict": "policy",
    "field_diff": "policy",
    "capability_probe": "capability",
    "capability_invalidation": "capability",
    "benchmark_run": "kernel",
    "service_request": "service",
    "distributed_run": "service",
    "normalize": "metric",
    "stability_check": "metric",
    "energy_window": "energy",
    "energy_integration": "energy",
    "cost_calculation": "cost",
    "pareto_decision": "decision",
    "recommendation": "decision",
    "maturity_session": "process",
    "lineage_transform": "lineage",
    "regeneration": "lineage",
}

#: Event kinds with no C7 mapping are reported, not silently dropped.
C7_UNMAPPED_KINDS: Tuple[str, ...] = ("unknown",)


@dataclass
class CampaignTraceRecord:
    """One C7-shaped trace record for a campaign event."""

    span_id: str
    trace_id: str
    parent_span_id: str = ""
    event_kind: str = ""
    c7_kind: str = ""
    cell_id: str = ""
    comparison_id: str = ""
    candidate_id: str = ""
    started_at_ns: int = 0
    ended_at_ns: int = 0
    status: str = "ok"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.span_id or not self.trace_id:
            problems.append("a trace record needs span_id and trace_id")
        if self.event_kind and self.event_kind not in C7_KIND_MAP:
            problems.append(f"unmapped event kind {self.event_kind!r} (reported, never dropped)")
        if self.c7_kind and self.event_kind and C7_KIND_MAP.get(self.event_kind) != self.c7_kind:
            problems.append(
                f"c7_kind {self.c7_kind!r} disagrees with the kind map for {self.event_kind!r}"
            )
        if self.ended_at_ns and self.started_at_ns and self.ended_at_ns < self.started_at_ns:
            problems.append("ended_at_ns precedes started_at_ns")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "c7_kind": self.c7_kind or C7_KIND_MAP.get(self.event_kind, ""),
            "event_kind": self.event_kind,
            "cell_id": self.cell_id,
            "comparison_id": self.comparison_id,
            "candidate_id": self.candidate_id,
            "started_at_ns": self.started_at_ns,
            "ended_at_ns": self.ended_at_ns,
            "status": self.status,
            "attributes": dict(sorted(self.attributes.items())),
        }


def project_c7(record: CampaignTraceRecord) -> Dict[str, Any]:
    problems = record.validate()
    projected = record.as_dict()
    filled = sorted(key for key, value in projected.items() if value not in (None, "", [], {}))
    return {
        "fields": projected,
        "filled_fields": filled,
        "empty_fields": sorted(set(projected) - set(filled)),
        "problems": problems,
        "ok": not problems,
    }


def span_chain_check(records: Sequence[CampaignTraceRecord]) -> Dict[str, Any]:
    """Every span must be reachable from a root and share one trace id."""
    by_span = {record.span_id: record for record in records}
    roots = [record for record in records if not record.parent_span_id]
    orphans = [
        record.span_id
        for record in records
        if record.parent_span_id and record.parent_span_id not in by_span
    ]
    trace_ids = {record.trace_id for record in records}
    return {
        "spans": len(records),
        "roots": len(roots),
        "orphans": sorted(orphans),
        "trace_ids": sorted(trace_ids),
        "single_trace": len(trace_ids) <= 1,
        "chain_ok": bool(roots) and not orphans and len(trace_ids) <= 1,
    }


def claim_status_rows(claims: Mapping[str, str]) -> Dict[str, Any]:
    """Validate claim statuses before they reach the acceptance report."""
    problems = [
        f"unknown claim status {status!r} for {claim_id}"
        for claim_id, status in sorted(claims.items())
        if status not in CLAIM_STATUSES
    ]
    return {"claims": dict(sorted(claims.items())), "problems": problems, "ok": not problems}


def missingness_audit(rows: Sequence[Mapping[str, Any]], *, column: str = "status") -> Dict[str, Any]:
    """Count missing states; a state is never collapsed into ``null`` or ``0``."""
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(column, "UNKNOWN"))
        counts[value] = counts.get(value, 0) + 1
    structured = {key: value for key, value in counts.items() if key in MISSINGNESS_CODES}
    return {
        "counts": dict(sorted(counts.items())),
        "structured_missing": structured,
        "unstructured_statuses": sorted(set(counts) - set(MISSINGNESS_CODES)),
        "has_structured_missingness": bool(structured),
    }


def table_names() -> Tuple[str, ...]:
    return tuple(sorted(S12_TABLE_SCHEMAS))


def schema_for(table: str) -> Tuple[str, ...]:
    try:
        return S12_TABLE_SCHEMAS[table]
    except KeyError as exc:
        raise ConfigError(f"unknown table {table!r}") from exc


def empty_row(table: str) -> Dict[str, Any]:
    """A schema-shaped empty row used by templates (all keys present, all empty)."""
    return {column: "" for column in schema_for(table)}


def validate_rows(table: str, rows: Sequence[Mapping[str, Any]]) -> List[str]:
    problems: List[str] = []
    for index, row in enumerate(rows):
        problems.extend(f"row {index}: {item}" for item in validate_table_row(table, row))
    return problems
