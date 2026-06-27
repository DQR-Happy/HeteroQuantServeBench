"""C6/C7 projection for the distributed layer plus the unified data tables.

The frozen contracts are :class:`~hqsb.core.contracts.result.BenchmarkResult`
(C6) and :class:`~hqsb.core.contracts.trace.TraceEvent` (C7).  Like S06/S07/S08,
the distributed layer projects its evidence onto them **without changing them**:
distributed payloads go into the ``summary["s10"]`` namespace and into event
attributes, and a coverage audit makes a silently dropped field visible.

The module also defines the four unified data tables of details README §17
(topology edge / collective sample / parallel event / scaling row) with a row
validator, so a report cannot mix column sets across platforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.contracts.result import BenchmarkResult, CorrectnessReport, EnvironmentInfo
from hqsb.core.contracts.trace import TraceEvent, TraceEventType
from hqsb.core.errors import ConfigError

#: Namespace used inside the frozen C6 ``summary`` map.
C6_NAMESPACE = "s10"

#: Distributed fields the C6 projection must carry (details README §17.4).
S10_C6_FIELDS: Tuple[str, ...] = (
    "run_id",
    "world_size",
    "node_count",
    "tp_degree",
    "pp_degree",
    "ep_degree",
    "cp_degree",
    "topology_manifest_hash",
    "placement_plan_hash",
    "parallel_plan_hash",
    "backend",
    "backend_version",
    "rank_epoch",
    "scaling_kind",
    "speedup",
    "efficiency",
    "device_seconds",
    "compute_ms",
    "comm_ms",
    "overlap_ms",
    "wait_ms",
    "idle_ms",
    "collective_summary",
    "memory_summary",
    "failure_summary",
)

#: The distributed span chain, in causal order (details E10-09 §4).
SPAN_CHAIN: Tuple[str, ...] = (
    "request",
    "iteration",
    "phase",
    "token_step",
    "layer",
    "module",
    "collective",
    "kernel",
    "transport",
)

#: Which C7 event type each distributed span maps to (only frozen enum members).
KIND_TO_EVENT_TYPE: Mapping[str, TraceEventType] = {
    "request": TraceEventType.QUEUE,
    "iteration": TraceEventType.DECODE,
    "phase": TraceEventType.PREFILL,
    "token_step": TraceEventType.DECODE,
    "layer": TraceEventType.KERNEL,
    "module": TraceEventType.KERNEL,
    "collective": TraceEventType.COLLECTIVE,
    "kernel": TraceEventType.KERNEL,
    "transport": TraceEventType.NETWORK,
    "rank": TraceEventType.QUEUE,
    "placement": TraceEventType.QUEUE,
    "cache": TraceEventType.CACHE,
}

#: Attributes a distributed C7 event must carry (details E10-09 step 5).
REQUIRED_EVENT_ATTRIBUTES: Tuple[str, ...] = (
    "run_id",
    "global_rank",
    "rank_epoch",
    "device_uuid",
    "phase",
    "group_id",
    "collective_seq",
    "parallel_plan_hash",
)


@dataclass
class S10ResultFields:
    """The distributed payload the C6 projection carries."""

    run_id: str = ""
    world_size: int = 0
    node_count: int = 0
    tp_degree: int = 0
    pp_degree: int = 0
    ep_degree: int = 0
    cp_degree: int = 0
    topology_manifest_hash: str = ""
    placement_plan_hash: str = ""
    parallel_plan_hash: str = ""
    backend: str = ""
    backend_version: str = ""
    rank_epoch: int = 0
    scaling_kind: str = ""
    speedup: Optional[float] = None
    efficiency: Optional[float] = None
    device_seconds: Optional[float] = None
    compute_ms: Optional[float] = None
    comm_ms: Optional[float] = None
    overlap_ms: Optional[float] = None
    wait_ms: Optional[float] = None
    idle_ms: Optional[float] = None
    collective_summary: Mapping[str, Any] = field(default_factory=dict)
    memory_summary: Mapping[str, Any] = field(default_factory=dict)
    failure_summary: Mapping[str, Any] = field(default_factory=dict)
    correctness_status: str = "not_run"
    raw_artifacts: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "world_size": self.world_size,
            "node_count": self.node_count,
            "tp_degree": self.tp_degree,
            "pp_degree": self.pp_degree,
            "ep_degree": self.ep_degree,
            "cp_degree": self.cp_degree,
            "topology_manifest_hash": self.topology_manifest_hash,
            "placement_plan_hash": self.placement_plan_hash,
            "parallel_plan_hash": self.parallel_plan_hash,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "rank_epoch": self.rank_epoch,
            "scaling_kind": self.scaling_kind,
            "speedup": self.speedup,
            "efficiency": self.efficiency,
            "device_seconds": self.device_seconds,
            "compute_ms": self.compute_ms,
            "comm_ms": self.comm_ms,
            "overlap_ms": self.overlap_ms,
            "wait_ms": self.wait_ms,
            "idle_ms": self.idle_ms,
            "collective_summary": dict(self.collective_summary),
            "memory_summary": dict(self.memory_summary),
            "failure_summary": dict(self.failure_summary),
            "correctness_status": self.correctness_status,
            "raw_artifacts": dict(sorted(self.raw_artifacts.items())),
        }


def project_c6(
    run_id: str,
    fields: S10ResultFields,
    *,
    environment: Optional[EnvironmentInfo] = None,
    git_commit: Optional[str] = None,
    git_dirty: Optional[bool] = None,
    model_artifact_hash: Optional[str] = None,
    config_hash: Optional[str] = None,
    raw_samples: Sequence[Mapping[str, Any]] = (),
    summary: Optional[Mapping[str, Any]] = None,
    correctness: Optional[CorrectnessReport] = None,
) -> BenchmarkResult:
    """Project the distributed payload onto the frozen C6 contract (no new fields)."""
    import time

    merged_summary: Dict[str, Any] = dict(summary or {})
    merged_summary[C6_NAMESPACE] = fields.as_dict()
    merged_summary["claim_level"] = "SOURCE"
    if correctness is None:
        correctness = CorrectnessReport(
            passed=fields.correctness_status == "pass",
            method="distributed correctness gate (see the S10 reports)",
            details={"status": fields.correctness_status, "distributed": True},
        )
    if environment is None:
        environment = EnvironmentInfo(
            platform="distributed-run",
            device=f"world_size={fields.world_size} nodes={fields.node_count}",
            framework_versions={fields.backend: fields.backend_version} if fields.backend else {},
        )
    return BenchmarkResult(
        run_id=run_id or fields.run_id or "unset-run",
        timestamp=time.time(),
        environment=environment,
        git_commit=git_commit,
        git_dirty=git_dirty,
        model_artifact_hash=model_artifact_hash,
        config_hash=config_hash,
        raw_samples=[dict(item) for item in raw_samples],
        summary=merged_summary,
        correctness=correctness,
        artifact_links=dict(fields.raw_artifacts),
    )


def project_c7(
    fields: S10ResultFields,
    *,
    records: Sequence[Mapping[str, Any]] = (),
    run_id: str = "",
    trace_id: str = "",
) -> List[TraceEvent]:
    """Project distributed span records onto C7 events (namespaced attributes)."""
    events: List[TraceEvent] = []
    for index, record in enumerate(records):
        kind = str(record.get("kind", "collective"))
        trace_kind = KIND_TO_EVENT_TYPE.get(kind)
        if trace_kind is None:
            raise ConfigError(
                f"unknown distributed span kind {kind!r}; register it in KIND_TO_EVENT_TYPE first",
                details={"field": "kind"},
            )
        attributes: Dict[str, Any] = {
            name: record.get(name) for name in REQUIRED_EVENT_ATTRIBUTES
        }
        attributes["run_id"] = record.get("run_id", run_id or fields.run_id)
        attributes["distributed"] = {
            "world_size": fields.world_size,
            "tp_degree": fields.tp_degree,
            "group_id": record.get("group_id", ""),
            "collective_seq": record.get("collective_seq", -1),
            "transport": record.get("transport", ""),
            "wait_reason": record.get("wait_reason", ""),
        }
        attributes["s10_kind"] = kind
        events.append(
            TraceEvent(
                event_type=trace_kind,
                timestamp_ns=int(record.get("start_ns") or 0),
                trace_id=str(record.get("trace_id", trace_id or run_id or fields.run_id)),
                span_id=str(record.get("span_id", f"{run_id or fields.run_id}-{index}")),
                parent_span_id=record.get("parent_span_id") or None,
                name=str(record.get("name", kind)),
                attributes=attributes,
            )
        )
    return events


def coverage_summary() -> Dict[str, Any]:
    """Audit that the projection carries every declared distributed field."""
    present = set(S10ResultFields().as_dict())
    required = set(S10_C6_FIELDS)
    missing = sorted(required - present)
    extra = sorted(present - required)
    return {
        "c6": {"ok": not missing, "missing": missing, "extra": extra},
        "c7": {"ok": True, "required_attributes": list(REQUIRED_EVENT_ATTRIBUTES)},
    }


# ── unified data tables (details README §17) ────────────────────────────────


#: Unified topology edge table (details README §17.1).
TOPO_EDGE_FIELDS: Tuple[str, ...] = (
    "run_id",
    "src_node",
    "dst_node",
    "edge_type",
    "nominal_width",
    "negotiated_speed",
    "status",
    "p2p_read",
    "p2p_write",
    "rdma_capable",
    "measured_latency_us",
    "measured_bandwidth_GBps",
    "source_tool",
    "timestamp",
    "evidence_uri",
)

#: Unified collective sample table (details README §17.2).
COLLECTIVE_SAMPLE_FIELDS: Tuple[str, ...] = (
    "run_id",
    "rank",
    "group_id",
    "collective_seq",
    "op",
    "count",
    "dtype",
    "payload_bytes",
    "inplace",
    "algorithm",
    "protocol",
    "stream_id",
    "enqueue_us",
    "completion_us",
    "latency_us",
    "algbw_GBps",
    "busbw_GBps",
    "correct",
    "input_hash",
    "output_hash",
    "error",
)

#: Unified parallel event table (details README §17.3).
PARALLEL_EVENT_FIELDS: Tuple[str, ...] = (
    "run_id",
    "request_id",
    "phase",
    "token_step",
    "layer",
    "parallel_kind",
    "tp_rank",
    "pp_rank",
    "ep_rank",
    "event_type",
    "collective_seq",
    "tensor_role",
    "shape",
    "dtype",
    "payload_bytes",
    "start_ns",
    "end_ns",
    "stream",
    "device",
    "wait_reason",
    "actual_backend",
)

#: Unified scaling row table (details README §17.4).
SCALING_ROW_FIELDS: Tuple[str, ...] = (
    "run_id",
    "world_size",
    "node_count",
    "tp",
    "pp",
    "ep",
    "cp",
    "global_batch",
    "per_rank_work",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "ttft_ms",
    "tpot_ms",
    "throughput",
    "speedup",
    "efficiency",
    "per_rank_memory",
    "compute_ms",
    "comm_ms",
    "overlap_ms",
    "wait_ms",
    "idle_ms",
    "failure_count",
    "status",
)

#: Table name → required columns.
TABLE_SCHEMAS: Mapping[str, Tuple[str, ...]] = {
    "topology_edge": TOPO_EDGE_FIELDS,
    "collective_sample": COLLECTIVE_SAMPLE_FIELDS,
    "parallel_event": PARALLEL_EVENT_FIELDS,
    "scaling_row": SCALING_ROW_FIELDS,
}


def validate_table_row(table: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a row against the unified table schema (unknown/missing columns)."""
    if table not in TABLE_SCHEMAS:
        raise ConfigError(
            f"unknown table {table!r}; expected one of {sorted(TABLE_SCHEMAS)}",
            details={"field": "table"},
        )
    required = set(TABLE_SCHEMAS[table])
    present = set(row)
    missing = sorted(required - present)
    extra = sorted(present - required)
    return {
        "table": table,
        "ok": not missing and not extra,
        "missing": missing,
        "extra": extra,
        "columns": len(required),
    }


def table_hashes(rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, str]:
    """Content hashes so a normalized table can be referenced from a manifest."""
    import hashlib
    import json

    hashes: Dict[str, str] = {}
    for table, rows in sorted(rows_by_table.items()):
        payload = json.dumps([dict(row) for row in rows], sort_keys=True, ensure_ascii=False)
        hashes[table] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return hashes


__all__ = [
    "C6_NAMESPACE",
    "COLLECTIVE_SAMPLE_FIELDS",
    "KIND_TO_EVENT_TYPE",
    "PARALLEL_EVENT_FIELDS",
    "REQUIRED_EVENT_ATTRIBUTES",
    "S10_C6_FIELDS",
    "S10ResultFields",
    "SCALING_ROW_FIELDS",
    "SPAN_CHAIN",
    "TABLE_SCHEMAS",
    "TOPO_EDGE_FIELDS",
    "coverage_summary",
    "project_c6",
    "project_c7",
    "table_hashes",
    "validate_table_row",
]
