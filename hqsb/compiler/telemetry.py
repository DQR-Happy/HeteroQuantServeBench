"""C6/C7 projection, raw-table schemas and coverage audit for S11.

S11 produces compile-side evidence that must land back in the shared contracts:
``C6 BenchmarkResult`` (identity, raw samples, correctness, resource, selected
backend) and ``C7 TraceEvent`` (what happened during one request).  This module
defines the projection fields, validates raw rows against their schemas and
audits which contract fields the S11 tables can fill — an incomplete field is
reported as missing, never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.compiler.identity import canonical_json, sha256_text
from hqsb.compiler.records import TABLE_SCHEMAS

# ── additional raw tables produced by S11 ──────────────────────────────────

ARTIFACT_INDEX_FIELDS: Tuple[str, ...] = (
    "artifact_id",
    "run_id",
    "compile_id",
    "parent_artifact_ids",
    "ir_level",
    "format",
    "schema_version",
    "raw_hash",
    "canonical_hash",
    "target_arch",
    "consumer",
    "verification_status",
)

LOWERING_DECISION_FIELDS: Tuple[str, ...] = (
    "compile_id",
    "op_instance_id",
    "semantic_op",
    "schema_version",
    "source_ir_id",
    "targeted_ir_id",
    "target_snapshot_id",
    "candidate_id",
    "semantic_legal",
    "capability_outcome",
    "artifact_compatible",
    "guard_outcome",
    "evidence_covered",
    "predicted_cost",
    "selection_policy",
    "selected",
    "reject_reason",
    "fallback_id",
    "materialize_status",
    "actual_dispatch_id",
)

DISPATCH_EVENT_FIELDS: Tuple[str, ...] = (
    "dispatch_id",
    "compile_id",
    "variant_id",
    "requested",
    "eligible",
    "selected",
    "actual",
    "actual_candidate_id",
    "fallback_reason",
    "artifact_hash",
    "guard_outcome",
    "trace_id",
    "latency_us",
)

CACHE_EVENT_FIELDS: Tuple[str, ...] = (
    "layer",
    "kind",
    "key",
    "entry_id",
    "reason_code",
    "bytes",
    "latency_us",
    "compile_occurred",
)

CAPTURE_BREAK_FIELDS: Tuple[str, ...] = (
    "break_id",
    "frame_id",
    "reason_code",
    "source_file",
    "source_line",
    "graph_before_id",
    "graph_after_id",
    "fallback_action",
)

GUARD_DEFINITION_FIELDS: Tuple[str, ...] = (
    "guard_id",
    "category",
    "expression",
    "source",
    "semantic_required",
    "failure_action",
)

VERIFIER_ISSUE_FIELDS: Tuple[str, ...] = (
    "graph_id",
    "code",
    "detail",
    "op_id",
    "severity",
    "blocks_compilation",
)

#: All raw tables S11 writes, with their minimal field lists.
S11_TABLE_SCHEMAS: Mapping[str, Tuple[str, ...]] = {
    **TABLE_SCHEMAS,
    "artifact_index": ARTIFACT_INDEX_FIELDS,
    "lowering_decision": LOWERING_DECISION_FIELDS,
    "dispatch_event": DISPATCH_EVENT_FIELDS,
    "cache_event": CACHE_EVENT_FIELDS,
    "capture_break": CAPTURE_BREAK_FIELDS,
    "guard_definition": GUARD_DEFINITION_FIELDS,
    "verifier_issue": VERIFIER_ISSUE_FIELDS,
}


def validate_table_row(table: str, row: Mapping[str, Any]) -> List[str]:
    """Validate a row against its table schema (unknown keys are rejected)."""
    if table not in S11_TABLE_SCHEMAS:
        raise ConfigError(
            f"unknown table {table!r}; known: {sorted(S11_TABLE_SCHEMAS)}"
        )
    required = set(S11_TABLE_SCHEMAS[table])
    return sorted(f"{table}: unknown key {key!r}" for key in set(row) - required)


def missing_fields(table: str, row: Mapping[str, Any]) -> List[str]:
    required = set(S11_TABLE_SCHEMAS[table])
    return sorted(required - set(row))


def table_hashes(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, str]:
    """Content hash per table (used in manifests to bind raw evidence)."""
    return {
        name: sha256_text(canonical_json([dict(sorted(row.items())) for row in rows]))
        for name, rows in sorted(tables.items())
    }


# ── C6 projection ──────────────────────────────────────────────────────────


@dataclass
class S11ResultFields:
    """The S11-specific part of a C6 result row (E11-03 step 16)."""

    compile_id: str = ""
    compile_mode: str = ""
    graph_identity: str = ""
    semantic_identity: str = ""
    compile_identity: str = ""
    ir_levels: Tuple[str, ...] = ()
    rewrite_count: int = 0
    guard_set_id: str = ""
    guard_outcome: str = ""
    cache_layer: str = ""
    cache_hit: Optional[bool] = None
    compile_phase_times: Mapping[str, float] = field(default_factory=dict)
    requested_lowering: str = ""
    selected_lowering: str = ""
    actual_lowering: str = ""
    observed_kernel: str = ""
    artifact_hash: str = ""
    fallback_reason: str = ""
    autotune_id: str = ""
    cost_model_id: str = ""
    variant_id: str = ""
    trace_id: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "compile_id": self.compile_id,
            "compile_mode": self.compile_mode,
            "graph_identity": self.graph_identity,
            "semantic_identity": self.semantic_identity,
            "compile_identity": self.compile_identity,
            "ir_levels": list(self.ir_levels),
            "rewrite_count": self.rewrite_count,
            "guard_set_id": self.guard_set_id,
            "guard_outcome": self.guard_outcome,
            "cache_layer": self.cache_layer,
            "cache_hit": self.cache_hit,
            "compile_phase_times": dict(sorted(self.compile_phase_times.items())),
            "requested_lowering": self.requested_lowering,
            "selected_lowering": self.selected_lowering,
            "actual_lowering": self.actual_lowering,
            "observed_kernel": self.observed_kernel,
            "artifact_hash": self.artifact_hash,
            "fallback_reason": self.fallback_reason,
            "autotune_id": self.autotune_id,
            "cost_model_id": self.cost_model_id,
            "variant_id": self.variant_id,
            "trace_id": self.trace_id,
        }

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.compile_id:
            problems.append("compile_id is required for a C6 projection")
        if self.actual_lowering and self.selected_lowering and self.actual_lowering != self.selected_lowering:
            if not self.fallback_reason:
                problems.append(
                    "actual != selected requires a fallback reason (no silent fallback, handbook §5.7)"
                )
        if self.cache_hit is True and not self.cache_layer:
            problems.append("a cache hit must name the layer")
        return problems


def project_c6(fields: S11ResultFields) -> Dict[str, Any]:
    """Map S11 fields onto the C6 result projection with a coverage statement."""
    problems = fields.validate()
    projected = {
        "run_id": fields.compile_id,
        "compile_mode": fields.compile_mode,
        "graph_identity": fields.graph_identity,
        "compile_identity": fields.compile_identity,
        "ir_levels": list(fields.ir_levels),
        "requested_implementation": fields.requested_lowering,
        "selected_implementation": fields.selected_lowering,
        "actual_implementation": fields.actual_lowering,
        "observed_kernel": fields.observed_kernel,
        "fallback_reason": fields.fallback_reason,
        "cache_layer": fields.cache_layer,
        "cache_hit": fields.cache_hit,
        "compile_phase_times": dict(sorted(fields.compile_phase_times.items())),
        "artifact_hash": fields.artifact_hash,
        "autotune_id": fields.autotune_id,
        "cost_model_id": fields.cost_model_id,
    }
    filled = sorted(key for key, value in projected.items() if value not in (None, "", [], {}))
    return {
        "fields": projected,
        "filled_fields": filled,
        "empty_fields": sorted(set(projected) - set(filled)),
        "problems": problems,
        "ok": not problems,
    }


def to_benchmark_result(fields: S11ResultFields, *, timestamp: float, **metadata: Any):
    """Export C6; ``project_c6`` remains the field-coverage inspection API."""
    from hqsb.core.contracts.projection import result_from_projection

    return result_from_projection(
        project_c6(fields), namespace="s11", run_id=fields.compile_id,
        timestamp=timestamp, **metadata,
    )


# ── C7 projection ──────────────────────────────────────────────────────────

C7_KIND_MAP: Mapping[str, str] = {
    "capture": "cache",
    "graph_break": "cache",
    "pass_rewrite": "kernel",
    "lowering_decision": "kernel",
    "dispatch": "kernel",
    "compile": "cache",
    "cache_event": "cache",
    "guard_event": "cache",
    "autotune_trial": "cache",
    "cost_model_decision": "cache",
    "fallback": "output",
}


@dataclass
class CompilerTraceRecord:
    """One S11 trace row before projection to a C7 event."""

    kind: str
    trace_id: str
    span_id: str
    timestamp_ns: int
    name: str = ""
    parent_span_id: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_event(self) -> Dict[str, Any]:
        if self.kind not in C7_KIND_MAP:
            raise ConfigError(f"unknown compiler trace kind {self.kind!r}")
        return {
            "event_type": C7_KIND_MAP[self.kind],
            "timestamp_ns": self.timestamp_ns,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id or None,
            "name": self.name or self.kind,
            "attributes": {"compiler.kind": self.kind, **dict(self.attributes)},
        }


def project_c7(records: Sequence[CompilerTraceRecord]) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    problems: List[str] = []
    for record in records:
        try:
            events.append(record.to_event())
        except ConfigError as exc:
            problems.append(str(exc))
    covered = sorted({event["attributes"]["compiler.kind"] for event in events})
    return {
        "events": events,
        "kinds_covered": covered,
        "problems": problems,
        "ok": not problems and bool(events),
        "rule": (
            "the span chain must let a compile decision be joined back to a request/prefill/decode "
            "span; orphaned compiler events are a trace gap"
        ),
    }


def span_chain_check(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Every non-root span must name a parent that exists in the trace."""
    ids = {str(event.get("span_id")) for event in events}
    rootless = [
        str(event.get("span_id"))
        for event in events
        if event.get("parent_span_id") and str(event.get("parent_span_id")) not in ids
    ]
    return {
        "events": len(events),
        "dangling_parents": rootless,
        "ok": not rootless,
    }


# ── coverage audit ─────────────────────────────────────────────────────────


def coverage_summary() -> Dict[str, Any]:
    """State which C6/C7 fields and raw tables the S11 layer can fill."""
    c6 = project_c6(S11ResultFields(compile_id="probe"))
    c7 = project_c7(
        [
            CompilerTraceRecord(kind=kind, trace_id="t", span_id="s", timestamp_ns=1)
            for kind in C7_KIND_MAP
        ]
    )
    return {
        "c6_fields": len(c6["fields"]),
        "c6_required_present": "run_id" in c6["fields"],
        "c7_kinds": sorted(C7_KIND_MAP),
        "c7_ok": c7["ok"],
        "tables": sorted(S11_TABLE_SCHEMAS),
        "table_count": len(S11_TABLE_SCHEMAS),
        "ok": c6["ok"] and c7["ok"],
        "note": (
            "this is a schema-coverage statement, not an experimental result: it says which "
            "fields the layer *can* fill, not that anything was measured"
        ),
    }


def result_field_rows() -> List[Dict[str, str]]:
    """Human-readable C6 field mapping for the development report."""
    return [
        {"field": "run_id", "source": "CompileRun.run_id / compile_id"},
        {"field": "selected_implementation", "source": "LoweringDecision.selected"},
        {"field": "actual_implementation", "source": "DispatchRow.actual (fallback recorded)"},
        {"field": "observed_kernel", "source": "ActualDispatchEvidence (profiler/binary)"},
        {"field": "compile_phase_times", "source": "BackendPlan step timings → cost keys"},
        {"field": "cache_layer/cache_hit", "source": "cache telemetry + ReadResult"},
        {"field": "artifact_hash", "source": "ArtifactIdentity.canonical_hash"},
        {"field": "guard_outcome", "source": "GuardEvaluation/GuardEventRecord"},
        {"field": "autotune_id/cost_model_id", "source": "tuning DB record / model artifact"},
    ]
