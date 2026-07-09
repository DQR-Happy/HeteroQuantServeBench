"""S14 projection onto the C6 (result) and C7 (trace) contracts.

S14 adds concepts that C1–C7 does not carry natively — a training step, a
rollout trajectory, a policy publish event, a KV episode, a modality phase.
``details/S14/README.md`` §7 says not to force them into C1–C7 but to
**establish a stable reference**; this module is that reference layer:

* :data:`C6_EXTENSION_FIELDS` / :data:`C7_EXTENSION_FIELDS` name the extra
  columns and their owning experiment;
* :func:`project_training_step`, :func:`project_conversion_node`,
  :func:`project_trajectory_event`, :func:`project_speculation_cycle`,
  :func:`project_moe_layer`, :func:`project_long_context_episode`,
  :func:`project_sparse_execution`, :func:`project_modality_phase`,
  :func:`project_workflow_event`, :func:`project_edge_record` produce a C6/C7
  shaped row;
* :func:`modality_metric_name` refuses to express an image/audio workload in
  ``tokens_per_second`` (``E14-06`` step 8 / §11 "用 token/s 表示所有任务").

Every projection is *label preserving*: a missing observation becomes ``None``
plus a ``missing`` entry, never ``0``.  No projection computes a derived
measurement — the arithmetic belongs to the experiment, not to the projector.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.identity import canonical_digest

SCHEMA_VERSION = "1.0.0"

#: S14's additions to the C6 result contract.
C6_EXTENSION_FIELDS: Mapping[str, Tuple[str, str]] = {
    "training_run_id": ("E14-02", "哪次训练产生了这一行"),
    "training_step": ("E14-02", "训练步（不是独立重复单位）"),
    "rank": ("E14-02", "数据并行 rank"),
    "world_size": ("E14-02", "实际 world size（不是 launcher 参数）"),
    "policy_snapshot_id": ("E14-04", "生成该样本的策略版本"),
    "staleness_steps": ("E14-04", "consume_step - produce_policy_step"),
    "frontier_study_contract_id": ("E14-05", "该行所属的冻结协议"),
    "intended_difference": ("E14-05", "baseline→candidate 的唯一预期差异"),
    "actual_implementation": ("E14-01", "实际执行路径（缺失即 INVALID_IDENTITY）"),
    "capability_reason": ("E14-01", "capability 判定原因码"),
    "logical_sparsity": ("E14-F4", "零元素比例（≠ 执行稀疏度）"),
    "pattern_compliance": ("E14-F4", "满足硬件 group pattern 的比例"),
    "kv_logical_bytes": ("E14-F3", "KV 逻辑字节"),
    "kv_physical_bytes": ("E14-F3", "KV 物理字节（含 block/对齐）"),
    "kv_metadata_bytes": ("E14-F3", "KV scale/zero/索引元数据字节"),
    "truncated_tokens": ("E14-F3", "被截断的 token 数（非零即不支持该长度）"),
    "acceptance": ("E14-F1", "接受率（不等于加速）"),
    "tokens_per_expert": ("E14-F2", "每 expert token 数（分布，不是均值）"),
    "imbalance_cv": ("E14-F2", "expert 负载变异系数"),
    "imbalance_gini": ("E14-F2", "expert 负载 Gini"),
    "max_to_mean_ratio": ("E14-F2", "最热 expert/mean"),
    "task_native_metric": ("E14-06", "任务原生指标名（不是 token/s）"),
    "evidence_level": ("E14-08", "MAP_ONLY | DEVICE_MEASURED"),
    "idempotency_key_hash": ("E14-07", "工具调用的幂等键"),
}

#: S14's additions to the C7 trace contract.
C7_EXTENSION_FIELDS: Mapping[str, Tuple[str, str]] = {
    "lifecycle_stage": (
        "E14-02/04",
        "training_step | rollout_request | trajectory | weight_publish | weight_load | "
        "weight_active | runtime_iteration | kernel | collective | service_request | service_stream",
    ),
    "span_kind": ("E14-07", "model | tool | orchestration | memory | queue | retry_wait"),
    "waiting": ("E14-07", "该 span 是否主要是等待而非执行"),
    "on_critical_path": ("E14-07", "是否在关键路径（span 求和不等于 E2E）"),
    "clock_domain": ("S14 README §24.1", "单调时钟域；跨进程/节点必须声明对齐策略"),
    "modality_phase": ("E14-06", "preprocess | encode | project | denoise | decode | postprocess"),
    "policy_snapshot_id": ("E14-04", "该 span 由哪一版策略产生"),
    "expert_rank": ("E14-F2", "expert→rank placement 版本"),
}

#: The lifecycle stages of ``S14 README`` §24.1, in timeline order.
TIMELINE_STAGES: Tuple[str, ...] = (
    "training_step",
    "rollout_request",
    "trajectory",
    "weight_publish",
    "weight_load",
    "weight_active",
    "runtime_iteration",
    "kernel",
    "collective",
    "service_request",
    "service_stream",
)

#: The span kinds of ``E14-07`` §4.3.
SPAN_KINDS: Tuple[str, ...] = ("model", "tool", "orchestration", "memory", "queue", "retry_wait")

#: Task-native metric names (``E14-06`` §3.1), per modality.
MODALITY_METRICS: Mapping[str, Tuple[str, ...]] = {
    "vlm": ("vqa_accuracy", "grounding_iou", "ocr_cer", "generation_quality"),
    "diffusion": ("image_fid", "clip_score", "prompt_alignment", "human_preference"),
    "audio": ("wer", "cer", "audio_quality_mos", "real_time_factor"),
}

#: Metric names that must never be used for a non-LLM workload.
LLM_ONLY_METRICS: Tuple[str, ...] = ("tokens_per_second", "ttft", "tpot", "itl", "acceptance")


def modality_metric_name(modality: str, metric: str) -> str:
    """Validate a task-native metric name (``E14-06`` step 8 / §10).

    Refusing ``tokens_per_second`` for an image/audio workload is deliberate:
    reusing the LLM unit is how a cross-morphology study degenerates into a
    framework wrapper (``E14-06`` §10).
    """
    if modality not in MODALITY_METRICS:
        raise ConfigError(
            f"unknown modality {modality!r}; must be one of {', '.join(rec.MULTIMODAL_MODALITIES)}"
        )
    if metric in LLM_ONLY_METRICS:
        raise ConfigError(
            f"metric {metric!r} is an LLM-only unit and may not express a {modality} workload "
            "(E14-06 §10: 用 token/s 表示所有任务 是 FAIL)"
        )
    if metric not in MODALITY_METRICS[modality]:
        raise ConfigError(
            f"metric {metric!r} is not declared for {modality}; declared: "
            f"{', '.join(MODALITY_METRICS[modality])}"
        )
    return metric


def _project(
    *, schema: str, identity: Mapping[str, Any], fields: Mapping[str, Tuple[str, str]], owner: str
) -> Dict[str, Any]:
    missing: List[str] = []
    row: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "projection_owner": owner}
    for name, (source, _description) in fields.items():
        if name in identity:
            row[name] = identity[name]
            if identity[name] is None:
                missing.append(name)
        else:
            row[name] = None
            missing.append(name)
    row["c6_or_c7_schema"] = schema
    row["missing"] = missing
    return row


def project_training_step(step: Mapping[str, Any]) -> Dict[str, Any]:
    """E14-02 step 24/25 → C6/C7 row (a training step is not a repetition unit)."""
    row = _project(
        schema="C7.TrainingStepTrace",
        identity=step,
        fields={
            key: C6_EXTENSION_FIELDS[key]
            for key in (
                "training_run_id",
                "training_step",
                "rank",
                "world_size",
                "actual_implementation",
                "capability_reason",
            )
        },
        owner="E14-02",
    )
    row["timing_ms"] = step.get("timing_ms")
    row["collective_bytes"] = step.get("collective_bytes")
    row["memory_bytes"] = step.get("memory_bytes")
    row["parameter_digest_after"] = step.get("parameter_digest_after")
    return row


def project_conversion_node(node: Mapping[str, Any]) -> Dict[str, Any]:
    """E14-03 ``ConversionNodeEvent`` → C6 row."""
    row = _project(
        schema="C6.ConversionNode",
        identity=node,
        fields={key: C6_EXTENSION_FIELDS[key] for key in ("actual_implementation", "capability_reason")},
        owner="E14-03",
    )
    for name in ("conversion_graph_id", "node_id", "tool_commit", "config_hash", "input_artifact_ids",
                 "output_artifact_id", "tensor_count_in", "tensor_count_out", "mapping_digest",
                 "validation_status"):
        row[name] = node.get(name)
    return row


def project_trajectory_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    """E14-04 ``PolicyDataflowEvent`` → C7 row."""
    row = _project(
        schema="C7.PolicyDataflow",
        identity=event,
        fields={
            key: C6_EXTENSION_FIELDS[key]
            for key in ("policy_snapshot_id", "staleness_steps", "actual_implementation")
        },
        owner="E14-04",
    )
    for name in ("event_id", "stage", "trajectory_ids", "producer_policy_snapshot_id",
                 "consumer_training_step", "attempt", "queue_age_ms", "input_digest", "output_digest"):
        row[name] = event.get(name)
    return row


def project_speculation_cycle(cycle: Mapping[str, Any]) -> Dict[str, Any]:
    """E14-F1 ``SpeculationCycle`` → C6 row (acceptance plus its full cost)."""
    row = _project(
        schema="C6.SpeculationCycle",
        identity=cycle,
        fields={
            key: C6_EXTENSION_FIELDS[key]
            for key in ("acceptance", "actual_implementation", "capability_reason")
        },
        owner="E14-F1",
    )
    row["accepted_count"] = cycle.get("accepted_count")
    row["proposed_tokens"] = len(cycle.get("proposed_token_ids") or ())
    for name in ("proposer_artifact_id", "target_artifact_id", "reject_position", "draft_time_ms",
                 "verify_time_ms", "rollback_time_ms", "target_calls", "kv_written_bytes",
                 "kv_discarded_bytes"):
        row[name] = cycle.get(name)
    return row


def project_moe_layer(layer: Mapping[str, Any]) -> Dict[str, Any]:
    """E14-F2 ``MoELayerEvent`` → C6 row (distribution, not a mean)."""
    row = _project(
        schema="C6.MoELayer",
        identity=layer,
        fields={
            key: C6_EXTENSION_FIELDS[key]
            for key in (
                "tokens_per_expert",
                "imbalance_cv",
                "imbalance_gini",
                "max_to_mean_ratio",
                "actual_implementation",
            )
        },
        owner="E14-F2",
    )
    for name in ("request_id", "layer", "router_version", "top_k", "expert_to_rank", "dropped_tokens",
                 "rerouted_tokens", "dispatch_bytes_per_peer", "dispatch_ms", "expert_compute_ms_per_rank",
                 "combine_ms", "slowest_rank", "actual_kernels"):
        row[name] = layer.get(name)
    return row


def project_long_context_episode(episode: Mapping[str, Any]) -> Dict[str, Any]:
    """E14-F3 ``LongContextEpisode`` → C6 row (capacity, quality and latency together)."""
    row = _project(
        schema="C6.LongContextEpisode",
        identity=episode,
        fields={
            key: C6_EXTENSION_FIELDS[key]
            for key in (
                "kv_logical_bytes",
                "kv_physical_bytes",
                "kv_metadata_bytes",
                "truncated_tokens",
                "actual_implementation",
            )
        },
        owner="E14-F3",
    )
    for name in ("episode_id", "model_artifact_id", "candidate_config_id", "submitted_tokens",
                 "accepted_tokens", "processed_tokens", "output_tokens", "prefill_ms", "ttft_ms",
                 "tpot_ms", "quality", "actual_backend"):
        row[name] = episode.get(name)
    return row


def project_sparse_execution(record: Mapping[str, Any]) -> Dict[str, Any]:
    """E14-F4 ``SparseExecutionRecord`` → C6 row."""
    row = _project(
        schema="C6.SparseExecution",
        identity=record,
        fields={
            key: C6_EXTENSION_FIELDS[key]
            for key in ("logical_sparsity", "pattern_compliance", "actual_implementation")
        },
        owner="E14-F4",
    )
    for name in ("operator_id", "source_artifact_id", "sparse_artifact_id", "pattern", "shape", "dtype",
                 "compressed_value_bytes", "metadata_bytes", "actual_kernel", "fallback", "latency_ms",
                 "correctness_status", "quality_eligible"):
        row[name] = record.get(name)
    return row


def project_modality_phase(phase: Mapping[str, Any]) -> Dict[str, Any]:
    """E14-06 ``ModalityPhaseEvent`` → C7 row."""
    row = _project(
        schema="C7.ModalityPhase",
        identity=phase,
        fields={key: C7_EXTENSION_FIELDS[key] for key in ("modality_phase",)},
        owner="E14-06",
    )
    row["task_native_metric"] = phase.get("task_native_metric")
    for name in ("request_id", "modality", "input_shape", "output_shape", "actual_backend",
                 "actual_kernels", "start_ns", "end_ns", "device_memory_peak_bytes",
                 "quality_artifact_id"):
        row[name] = phase.get(name)
    return row


def project_workflow_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    """E14-07 ``WorkflowStateEvent`` → C7 row."""
    row = _project(
        schema="C7.WorkflowState",
        identity=event,
        fields={key: C7_EXTENSION_FIELDS[key] for key in ("idempotency_key_hash",)},
        owner="E14-07",
    )
    for name in ("workflow_id", "task_id", "tenant_id", "old_state", "new_state", "attempt",
                 "model_artifact_id", "tool_contract_id", "trace_id", "timestamp_ns", "cost_accumulated"):
        row[name] = event.get(name)
    return row


def project_edge_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """E14-08 ``EdgeExecutionRecord`` → C6 row (evidence level is part of the row)."""
    row = _project(
        schema="C6.EdgeExecution",
        identity=record,
        fields={key: C6_EXTENSION_FIELDS[key] for key in ("evidence_level",)},
        owner="E14-08",
    )
    for name in ("device_id", "runtime_id", "model_artifact_id", "input_id", "mode",
                 "requested_backend", "actual_backend", "accelerated_op_fraction", "fallback_ops",
                 "latency_ms", "memory_peak_bytes", "average_power_w", "energy_j", "temperature_c",
                 "quality_status"):
        row[name] = record.get(name)
    if row.get("evidence_level") == "MAP_ONLY" and any(
        row.get(name) is not None for name in ("latency_ms", "average_power_w", "energy_j", "memory_peak_bytes")
    ):
        row.setdefault("violations", []).append(
            "MAP_ONLY evidence may not carry device measurements (E14-08 §2 路线 B 不产生 latency/power claim)"
        )
    return row


#: Projector registry used by the driver and the tests.
PROJECTORS: Mapping[str, Any] = {
    "training_step": project_training_step,
    "conversion_node": project_conversion_node,
    "trajectory_event": project_trajectory_event,
    "speculation_cycle": project_speculation_cycle,
    "moe_layer": project_moe_layer,
    "long_context_episode": project_long_context_episode,
    "sparse_execution": project_sparse_execution,
    "modality_phase": project_modality_phase,
    "workflow_event": project_workflow_event,
    "edge_record": project_edge_record,
}


def project(kind: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    if kind not in PROJECTORS:
        raise ConfigError(f"unknown projection {kind!r}; known: {', '.join(sorted(PROJECTORS))}")
    return PROJECTORS[kind](payload)


def validate_table_rows(table: str, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Check rows against the declared columns of :data:`records.TABLE_SCHEMAS`.

    Returns one finding per offending row; ``unknown`` columns are refused so a
    typo cannot silently become a null in the normalized output.
    """
    if table not in rec.TABLE_SCHEMAS:
        raise ConfigError(f"unknown S14 table {table!r}")
    expected = set(rec.table_columns(table))
    findings: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        unknown = sorted(set(row) - expected)
        missing = sorted(expected - set(row))
        if unknown:
            findings.append({"row": index, "problem": "unknown columns", "columns": unknown})
        if missing:
            findings.append({"row": index, "problem": "missing columns", "columns": missing})
    return findings


def identity_closure(rows: Sequence[Mapping[str, Any]], *, required: Sequence[str]) -> Dict[str, Any]:
    """``E14-03``/``E14-04``: every measured row must resolve to its lineage ids.

    A row that cannot be traced back to its artifacts is not evidence, it is a
    number (``E14-03`` §8.1: 路径名代替 artifact identity 是 FAIL).
    """
    unresolved: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        absent = [name for name in required if not row.get(name)]
        if absent:
            unresolved.append({"row": index, "missing": absent})
    return {
        "rows": len(rows),
        "required": list(required),
        "unresolved": unresolved,
        "closed": not unresolved,
    }


def coverage_report(counts: Mapping[str, int]) -> Dict[str, Any]:
    """Coverage of the timeline stages / span kinds / layers, with the gaps named."""
    timeline = {stage: int(counts.get(stage, 0)) for stage in TIMELINE_STAGES}
    spans = {kind: int(counts.get(kind, 0)) for kind in SPAN_KINDS}
    layers = {layer: int(counts.get(layer, 0)) for layer in rec.PROFILE_LAYERS}
    return {
        "timeline": timeline,
        "span_kinds": spans,
        "profile_layers": layers,
        "missing_timeline_stages": [name for name, value in timeline.items() if value == 0],
        "missing_span_kinds": [name for name, value in spans.items() if value == 0],
        "missing_profile_layers": [name for name, value in layers.items() if value == 0],
        "layers_covered": sum(1 for value in layers.values() if value),
    }


def hash_row(row: Mapping[str, Any]) -> str:
    """Content address one projected row (raw→normalized link, ``E14-03`` §8.1)."""
    return canonical_digest({key: row[key] for key in sorted(row)})


def logical_uri(campaign_id: str, kind: str, name: str) -> str:
    if not campaign_id or not name:
        raise ConfigError("logical_uri requires a campaign id and a name")
    return f"s14://{campaign_id}/{kind}/{name}"


def schema_audit() -> Dict[str, Any]:
    """The C6/C7 extension surface, so the report can cite it without hand-copying."""
    return {
        "c6_extension_fields": len(C6_EXTENSION_FIELDS),
        "c7_extension_fields": len(C7_EXTENSION_FIELDS),
        "timeline_stages": list(TIMELINE_STAGES),
        "span_kinds": list(SPAN_KINDS),
        "projectors": sorted(PROJECTORS),
        "modalities": {name: list(metrics) for name, metrics in sorted(MODALITY_METRICS.items())},
        "llm_only_metrics": list(LLM_ONLY_METRICS),
    }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the projection layer (labelled smoke, not an experiment)."""
    checks: Dict[str, Any] = {"status": "smoke", "claim_allowed": False}
    checks["c6_fields"] = len(C6_EXTENSION_FIELDS)
    checks["c7_fields"] = len(C7_EXTENSION_FIELDS)
    checks["projectors"] = sorted(PROJECTORS)
    refused = False
    try:
        modality_metric_name("audio", "tokens_per_second")
    except ConfigError:
        refused = True
    checks["llm_metric_refused_for_audio"] = refused
    edge = project_edge_record({"evidence_level": "MAP_ONLY", "latency_ms": 1.0})
    checks["map_only_violation_detected"] = bool(edge.get("violations"))
    return checks
