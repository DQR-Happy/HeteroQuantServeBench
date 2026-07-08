"""E14-03 — checkpoint → conversion/merge → quant → runtime semantic parity.

Core judgement: a legal E14-02 checkpoint becomes a ``ServingModelArtifact``
through an explicit, replayable conversion graph; tensor mapping, logits, greedy
tokens and task quality pass **layer by layer**; wrong shard/tokenizer/config/
precision/adapter/version is refused *before* serving; every conversion can be
traced back to code, config and input artefacts.

Why the gates are layered (§3.3): greedy tokens are discrete, so a small logit
difference can hide behind a large top-1 margin, while a tiny legal difference at
a near-tie can flip a token.  Comparing only total file hashes cannot locate an
error, and comparing only final text can be masked by tokenisation — hence
``inventory → tensor → block → logits/top-k → token → quality → runtime load``.

Interfaces provided:

* :class:`ConversionNodeEvent` — the §7 record of one graph node;
* :class:`TensorMappingRow`, :func:`validate_mapping` — the explicit mapping
  table, with one-to-one / many-to-one legality (steps 11, 20);
* :class:`ParityGate` / :func:`evaluate_gates` — the seven layers as data, with
  the first failing layer named (steps 20–25, 31);
* :class:`AdapterSemantics` — unmerged baseline, merge and duplicate-merge
  protection (steps 16–18);
* :func:`negative_artifact_matrix` — the six wrong-artefact fixtures of §PASS
  (steps 33–36);
* :func:`check_atomic_publish`, :func:`check_conversion_cache` — concurrency,
  cache and version cross-talk (steps 37–38);
* :class:`TrainServeParityVerdict` — the step-40 ruling.

Nothing here converts a tensor or loads a model; the Runtime load path and the
quant node delegate to C1/C4/C5 and are addressed by identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


from hqsb.experimental import records as rec
from hqsb.experimental.contracts import ServingModelArtifact, artifact_hierarchy
from hqsb.experimental.identity import SCHEMA_PREFIX, canonical_digest, is_digest

EXPERIMENT_ID = "E14-03"
TITLE = "Checkpoint→转换/Merge→Quant→Runtime 的训推一致性"
LEVEL = "P0"

CLAIM_BOUNDARY = (
    "只证明转换 DAG、分层语义门与负向拒绝；不证明训练改善质量，也不证明转换后的 runtime 更快"
    "（E14-03 §12）。"
)

#: The ordered semantic layers (``E14-03`` §3.3 / §9).
PARITY_LAYERS: Tuple[str, ...] = (
    "inventory",
    "tensor",
    "block",
    "logits",
    "token",
    "quality",
    "runtime_load",
)

#: The six wrong-artefact families that must fail closed (``E14-03`` PASS list).
NEGATIVE_ARTIFACT_FAMILIES: Tuple[str, ...] = (
    "missing_or_corrupt_shard",
    "wrong_tokenizer_or_template",
    "wrong_config_or_rope",
    "wrong_precision_or_quant_metadata",
    "wrong_runtime_or_engine_version",
    "wrong_adapter_or_base_revision",
)

#: Reference paths of §5.  ``R0`` is the source of truth; every other path is a
#: controlled step away from it, so a drift can be attributed to one edge.
REFERENCE_PATHS: Tuple[str, ...] = (
    "R0_training_native_eval",
    "R1_full_precision_export",
    "R2_adapter_merged_or_unmerged",
    "R3_quantized_runtime",
    "R4_compiled_engine",
)

#: The conversion nodes of the frozen DAG (``E14-03`` step 3).
CONVERSION_NODES: Tuple[str, ...] = (
    "gather_reshard",
    "rename",
    "transpose",
    "fuse_qkv",
    "merge_adapter",
    "cast",
    "quantize",
    "pack",
    "engine_build",
    "publish",
)

#: Tensor transforms an edge may declare (``E14-03`` §3.2).
TRANSFORMS: Tuple[str, ...] = (
    "identity",
    "slice",
    "concat",
    "split",
    "permute",
    "transpose",
    "cast",
    "quantize",
    "merge_delta",
    "pack",
)

#: Dtypes an edge may declare.
DTYPES: Tuple[str, ...] = ("fp32", "tf32", "bf16", "fp16", "fp8", "int8", "int4", "uint8", "bool")


@dataclass
class ConversionNodeEvent:
    """``E14-03`` §7 ``ConversionNodeEvent`` (schema ``…e14-03.node.v1``)."""

    conversion_graph_id: str
    node_id: str
    tool_commit: str = ""
    config_hash: str = ""
    input_artifact_ids: Tuple[str, ...] = ()
    output_artifact_id: str = ""
    tensor_count_in: int = 0
    tensor_count_out: int = 0
    mapping_digest: str = ""
    validation_status: str = rec.STATUS_NOT_RUN
    resource_usage: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: Tuple[str, ...] = ()

    schema_version = f"{SCHEMA_PREFIX}.e14-03.node.v1"

    def validate(self) -> List[str]:
        findings: List[str] = []
        for name in ("conversion_graph_id", "node_id", "output_artifact_id"):
            if not getattr(self, name):
                findings.append(f"ConversionNodeEvent: {name} is required")
        if self.node_id and self.node_id not in CONVERSION_NODES:
            findings.append(
                f"ConversionNodeEvent: unknown node {self.node_id!r}; the DAG nodes are "
                f"{', '.join(CONVERSION_NODES)}"
            )
        if not self.tool_commit:
            findings.append("ConversionNodeEvent: tool_commit is required (转换必须能反查源码)")
        if not self.config_hash:
            findings.append("ConversionNodeEvent: config_hash is required")
        if not self.input_artifact_ids:
            findings.append("ConversionNodeEvent: at least one input artifact id is required")
        for digest in (*self.input_artifact_ids, self.output_artifact_id, self.mapping_digest):
            if digest and not is_digest(digest):
                findings.append(f"ConversionNodeEvent: {digest!r} is not a sha256 digest")
        if self.tensor_count_in and not self.tensor_count_out:
            findings.append("ConversionNodeEvent: tensors were consumed but none produced")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "conversion_graph_id": self.conversion_graph_id,
            "node_id": self.node_id,
            "tool_commit": self.tool_commit,
            "config_hash": self.config_hash,
            "input_artifact_ids": list(self.input_artifact_ids),
            "output_artifact_id": self.output_artifact_id,
            "tensor_count_in": self.tensor_count_in,
            "tensor_count_out": self.tensor_count_out,
            "mapping_digest": self.mapping_digest,
            "validation_status": self.validation_status,
            "resource_usage": {key: self.resource_usage[key] for key in sorted(self.resource_usage)},
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass
class TensorMappingRow:
    """One row of the explicit mapping table (``E14-03`` step 11)."""

    source_name: str
    target_name: str
    transform: str = "identity"
    axis: int = 0
    slice_spec: str = ""
    permute: Tuple[int, ...] = ()
    dtype_from: str = ""
    dtype_to: str = ""
    shape_before: Tuple[int, ...] = ()
    shape_after: Tuple[int, ...] = ()
    source_digest: str = ""
    source_ranks: Tuple[int, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("source_name", "target_name"):
            if not getattr(self, name):
                findings.append(f"mapping row is missing {name!r}")
        if self.transform not in TRANSFORMS:
            findings.append(f"mapping row {self.source_name} has unknown transform {self.transform!r}")
        if self.transform == "permute" and not self.permute:
            findings.append(f"mapping row {self.source_name}: permute without a permutation")
        if self.permute and sorted(self.permute) != list(range(len(self.permute))):
            findings.append(f"mapping row {self.source_name}: permute {self.permute} is not a permutation")
        for name in ("dtype_from", "dtype_to"):
            value = getattr(self, name)
            if value and value not in DTYPES:
                findings.append(f"mapping row {self.source_name}: unknown dtype {value!r} in {name}")
        if self.shape_before and self.shape_after and self.transform == "identity":
            if self.shape_before != self.shape_after:
                findings.append(
                    f"mapping row {self.source_name}: identity transform changes the shape "
                    f"{self.shape_before} -> {self.shape_after}"
                )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "target_name": self.target_name,
            "transform": self.transform,
            "axis": self.axis,
            "slice": self.slice_spec,
            "permute": list(self.permute),
            "dtype_from": self.dtype_from,
            "dtype_to": self.dtype_to,
            "shape_before": list(self.shape_before),
            "shape_after": list(self.shape_after),
            "source_digest": self.source_digest,
            "source_ranks": list(self.source_ranks),
        }


def validate_mapping(
    rows: Sequence[TensorMappingRow],
    *,
    expected_targets: Sequence[str] = (),
    allow_many_to_one: bool = False,
) -> Dict[str, Any]:
    """Steps 11/20: every required parameter has exactly one legal source.

    A ``regex`` rename that silently overwrites a same-named tensor is the
    failure this refuses (§28 error: regex rename 覆盖同名 tensor).
    """
    problems: List[str] = []
    by_target: Dict[str, List[TensorMappingRow]] = {}
    by_source: Dict[str, List[TensorMappingRow]] = {}
    for row in rows:
        problems.extend(row.problems())
        by_target.setdefault(row.target_name, []).append(row)
        by_source.setdefault(row.source_name, []).append(row)
    if not allow_many_to_one:
        for target, mapping_rows in sorted(by_target.items()):
            if len(mapping_rows) > 1:
                problems.append(
                    f"target {target} has {len(mapping_rows)} sources (silent overwrite)"
                )
    duplicated_sources = sorted(
        name for name, mapping_rows in by_source.items() if len(mapping_rows) > 1
    )
    missing = sorted(set(expected_targets) - set(by_target))
    extra = sorted(set(by_target) - set(expected_targets)) if expected_targets else []
    if missing:
        problems.append(f"{len(missing)} expected target tensors are unmapped: {', '.join(missing[:5])}")
    if extra:
        problems.append(f"{len(extra)} produced targets are not in the architecture spec: {', '.join(extra[:5])}")
    return {
        "rows": len(rows),
        "targets": len(by_target),
        "sources": len(by_source),
        "many_to_one_allowed": allow_many_to_one,
        "missing_targets": missing,
        "extra_targets": extra,
        "duplicate_source_usage": duplicated_sources,
        "mapping_digest": canonical_digest([row.as_dict() for row in rows]),
        "problems": problems,
        "ok": not problems,
    }


@dataclass
class ParityGate:
    """One of the seven layers, with its preregistered tolerance and result."""

    layer: str
    tolerance: Mapping[str, float] = field(default_factory=dict)
    status: str = rec.STATUS_NOT_RUN
    detail: Mapping[str, Any] = field(default_factory=dict)

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.layer not in PARITY_LAYERS:
            findings.append(f"unknown parity layer {self.layer!r}")
        if not self.tolerance:
            findings.append(
                f"layer {self.layer}: tolerance must be preregistered (冻结精度与容差，不事后放宽)"
            )
        if self.status not in rec.ALL_STATUSES:
            findings.append(f"layer {self.layer}: unknown status {self.status!r}")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "tolerance": {key: self.tolerance[key] for key in sorted(self.tolerance)},
            "status": self.status,
            "detail": dict(self.detail),
        }


def evaluate_gates(gates: Sequence[ParityGate]) -> Dict[str, Any]:
    """Steps 20–25/31: stop at the first failing layer and name it.

    Layers after a failure are reported ``NOT_RUN`` rather than "passed by
    omission", because a later text-level match cannot repair an earlier tensor
    mismatch (``E14-03`` §9: tensor 一致而 logits 不一致 / logits 小偏差导致
    token 分歧).
    """
    problems: List[str] = []
    for gate in gates:
        problems.extend(gate.problems())
    ordered = {gate.layer: gate for gate in gates}
    missing_layers = [layer for layer in PARITY_LAYERS if layer not in ordered]
    first_failure: Optional[str] = None
    blocked: List[str] = []
    for layer in PARITY_LAYERS:
        gate = ordered.get(layer)
        if gate is None:
            continue
        if first_failure is not None:
            blocked.append(layer)
            continue
        if gate.status in (rec.STATUS_FAIL, rec.STATUS_FAIL_CORRECTNESS, rec.STATUS_FAIL_QUALITY):
            first_failure = layer
    return {
        "layers": len(ordered),
        "expected_layers": len(PARITY_LAYERS),
        "missing_layers": missing_layers,
        "first_failure": first_failure,
        "layers_after_failure": blocked,
        "all_passed": not problems and not missing_layers and first_failure is None
        and all(gate.status == rec.STATUS_PASS for gate in gates),
        "problems": problems,
        "digest": canonical_digest([gate.as_dict() for gate in gates]),
    }


# ── adapter merge semantics (steps 7, 16–18) ───────────────────────────────


@dataclass
class AdapterSemantics:
    """The frozen adapter/merge contract (``E14-03`` step 7)."""

    adapter_id: str = ""
    base_artifact_id: str = ""
    target_modules: Tuple[str, ...] = ()
    rank: int = 0
    alpha: float = 0.0
    merge_dtype: str = "fp32"
    fan_in_axis: int = 0
    fan_out_axis: int = 1
    merged: bool = False
    merge_count: int = 0
    in_place: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.adapter_id and not self.base_artifact_id:
            return ["adapter semantics is not frozen: record the adapter/base identity or declare N/A explicitly"]
        if self.adapter_id:
            for name in ("base_artifact_id",):
                if not getattr(self, name):
                    findings.append(f"adapter {self.adapter_id}: {name} is required")
            if not self.target_modules:
                findings.append(f"adapter {self.adapter_id}: target_modules are required")
            if self.rank <= 0:
                findings.append(f"adapter {self.adapter_id}: rank must be positive")
            if self.merge_dtype not in DTYPES:
                findings.append(f"adapter {self.adapter_id}: unknown merge_dtype {self.merge_dtype!r}")
            if self.merge_count > 1:
                findings.append(
                    f"adapter {self.adapter_id}: merged {self.merge_count} times "
                    "(重复 merge 会二次缩放 delta)"
                )
            if self.merged and self.in_place:
                findings.append(
                    f"adapter {self.adapter_id}: in-place merge destroys the only unmerged oracle"
                )
            if self.merged and self.alpha == 0.0:
                findings.append(f"adapter {self.adapter_id}: alpha is 0; the delta would vanish silently")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "base_artifact_id": self.base_artifact_id,
            "target_modules": list(self.target_modules),
            "rank": self.rank,
            "alpha": self.alpha,
            "merge_dtype": self.merge_dtype,
            "fan_in_axis": self.fan_in_axis,
            "fan_out_axis": self.fan_out_axis,
            "merged": self.merged,
            "merge_count": self.merge_count,
            "in_place": self.in_place,
        }


def compare_merged_unmerged(
    *,
    unmerged_layer_diffs: Sequence[Mapping[str, Any]],
    merged_layer_diffs: Sequence[Mapping[str, Any]],
    abs_tol: float,
) -> Dict[str, Any]:
    """Step 18: merged vs unmerged must be attributed to merge, not to a kernel.

    ``E14-03`` §9: 检查 merge dtype、fan-in/out、alpha 和重复 merge；两个路径必须
    使用相同 backend/精度，否则无法归因。
    """
    if len(unmerged_layer_diffs) != len(merged_layer_diffs):
        return {"error": "both paths must report the same set of layers"}
    rows: List[Dict[str, Any]] = []
    for left, right in zip(unmerged_layer_diffs, merged_layer_diffs):
        layer = str(left.get("layer", right.get("layer", "")))
        delta = float(merged_layer_diffs[merged_layer_diffs.index(right)].get("max_abs", 0.0))
        rows.append({"layer": layer, "max_abs": delta, "within": abs(delta) <= abs_tol})
    return {
        "layers": rows,
        "all_within": all(row["within"] for row in rows),
        "abs_tol": abs_tol,
        "first_divergence": next((row["layer"] for row in rows if not row["within"]), None),
    }


# ── negative artefacts, publish, cache, cost (steps 33–39) ──────────────────


@dataclass(frozen=True)
class NegativeArtifactCase:
    """One wrong-artefact fixture and the earliest gate that must reject it."""

    family: str
    mutation: str
    earliest_detector: str
    structured_reason: str
    isolation_required: str = "复制制品后再篡改，不破坏原始 artifact（E14-03 step 33）"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "mutation": self.mutation,
            "earliest_detector": self.earliest_detector,
            "structured_reason": self.structured_reason,
            "isolation_required": self.isolation_required,
        }


def negative_artifact_matrix() -> Tuple[NegativeArtifactCase, ...]:
    """Steps 33–36: the six families that must fail closed before serving.

    ``E14-03`` §10 forbids testing only random corruption: a same-shape wrong
    *version* is the case that a shape check cannot catch, so each family names
    the identity gate rather than a numerical symptom.
    """
    return (
        NegativeArtifactCase(
            family="missing_or_corrupt_shard",
            mutation="删除或篡改一项分片/文件",
            earliest_detector="inventory 完整性 + aggregate root（converter 侧，最早合法检查点）",
            structured_reason="INVALID_IDENTITY",
        ),
        NegativeArtifactCase(
            family="wrong_tokenizer_or_template",
            mutation="替换 tokenizer / special token / chat template",
            earliest_detector="模型身份校验：tokenizer digest 不属于权重 hash",
            structured_reason="INVALID_IDENTITY",
        ),
        NegativeArtifactCase(
            family="wrong_config_or_rope",
            mutation="改变 heads / RoPE / 层数",
            earliest_detector="architecture 参数规范比对（step 10）",
            structured_reason="INVALID_PROTOCOL",
        ),
        NegativeArtifactCase(
            family="wrong_precision_or_quant_metadata",
            mutation="改变 dtype / quant metadata",
            earliest_detector="precision contract + C5 QuantArtifact 校验",
            structured_reason="INVALID_IDENTITY",
        ),
        NegativeArtifactCase(
            family="wrong_runtime_or_engine_version",
            mutation="以错误设备/runtime 加载 engine",
            earliest_detector="compatibility digest 校验",
            structured_reason="INVALID_IDENTITY",
        ),
        NegativeArtifactCase(
            family="wrong_adapter_or_base_revision",
            mutation="同 shape 但错误 base revision / target module",
            earliest_detector="provenance/identity gate（形状检查抓不到）",
            structured_reason="INVALID_IDENTITY",
        ),
    )


def check_atomic_publish(
    events: Sequence[Mapping[str, Any]], *, active_pointer_reads: Sequence[Mapping[str, Any]] = ()
) -> List[str]:
    """Step 37: no consumer may observe a partially written artefact.

    A publish must be staging → verify → atomic pointer swap; a read that lands
    between two writes is the version cross-talk the step forbids.
    """
    problems: List[str] = []
    for event in events:
        if not event.get("staging_dir"):
            problems.append(f"publish {event.get('publish_id', '<unnamed>')}: no staging directory")
        if not event.get("completion_marker"):
            problems.append(f"publish {event.get('publish_id', '<unnamed>')}: no completion marker")
        if not event.get("atomic_swap"):
            problems.append(
                f"publish {event.get('publish_id', '<unnamed>')}: active pointer is not swapped atomically"
            )
        if event.get("wrote_in_place"):
            problems.append(
                f"publish {event.get('publish_id', '<unnamed>')}: wrote into the live artefact in place"
            )
    for read in active_pointer_reads:
        if not read.get("resolved_version"):
            problems.append("a consumer read the active pointer without resolving a version")
        if read.get("saw_partial"):
            problems.append(
                f"consumer {read.get('consumer', '<unnamed>')} observed a partially written artefact"
            )
    return problems


def check_conversion_cache(
    *, cold: Mapping[str, Any], warm: Mapping[str, Any], semantic_fields: Sequence[str]
) -> Dict[str, Any]:
    """Step 38: a cache hit must not change semantics, and must name its key.

    ``E14-03`` §10: 复用未知旧制品伪造速度和一致性 is the failure; a warm run whose
    key cannot be explained is reported even when its outputs match.
    """
    problems: List[str] = []
    for field_name in semantic_fields:
        if cold.get(field_name) != warm.get(field_name):
            problems.append(
                f"field {field_name!r} differs between cold and warm: cold={cold.get(field_name)!r} "
                f"warm={warm.get(field_name)!r}"
            )
    for field_name in ("cache_key", "cache_hit", "cache_reason"):
        if field_name not in warm:
            problems.append(f"warm run does not record {field_name!r} (命中来源不可追踪)")
    if warm.get("cache_hit") and not warm.get("cache_reason"):
        problems.append("cache hit without a reason code")
    return {
        "fields": len(semantic_fields),
        "problems": problems,
        "consistent": not problems,
        "cold_digest": canonical_digest({key: cold.get(key) for key in sorted(semantic_fields)}),
        "warm_digest": canonical_digest({key: warm.get(key) for key in sorted(semantic_fields)}),
    }


#: Resource dimensions a conversion/load must report (``E14-03`` step 39).
CONVERSION_RESOURCE_KEYS: Tuple[str, ...] = (
    "wall_ms",
    "cpu_ms",
    "host_memory_peak_bytes",
    "device_memory_peak_bytes",
    "disk_bytes_written",
    "network_bytes",
)


def audit_conversion_resources(timeline: Mapping[str, Any]) -> List[str]:
    """Step 39: a conversion that OOMs or fills the disk is not a success story."""
    findings: List[str] = []
    for key in CONVERSION_RESOURCE_KEYS:
        if key not in timeline:
            findings.append(f"conversion resource timeline is missing {key!r} (unknown is not zero)")
        elif timeline[key] is None:
            findings.append(f"conversion resource {key!r} is unmeasured; record it explicitly")
    if not timeline.get("failure_cleanup"):
        findings.append("no failure-cleanup record: 失败后是否清理临时目录必须可查")
    return findings


# ── verdict (step 40) ──────────────────────────────────────────────────────


@dataclass
class TrainServeParityVerdict:
    """Step 40 — per-node and per-layer rulings, never one blanket PASS."""

    verdict_id: str
    decision: str = rec.STATUS_NOT_RUN
    node_status: Mapping[str, str] = field(default_factory=dict)
    gate_summary: Mapping[str, Any] = field(default_factory=dict)
    source_to_serving_lineage: Tuple[str, ...] = ()
    sub_path_status: Mapping[str, str] = field(default_factory=dict)
    negative_cases: Tuple[Mapping[str, Any], ...] = ()
    limitations: Tuple[str, ...] = ()

    schema_version = f"{SCHEMA_PREFIX}.e14-03.verdict.v1"

    #: Sub-paths that must each carry their own status (``E14-03`` §11).
    REQUIRED_SUB_PATHS: Tuple[str, ...] = ("full_precision", "adapter", "quant", "engine", "runtime_load")

    def validate(self) -> List[str]:
        findings: List[str] = []
        if self.decision not in rec.ALL_STATUSES:
            findings.append(f"TrainServeParityVerdict: unknown decision {self.decision!r}")
        for node, status in self.node_status.items():
            if node not in CONVERSION_NODES:
                findings.append(f"TrainServeParityVerdict: unknown conversion node {node!r}")
            if status not in rec.ALL_STATUSES:
                findings.append(f"TrainServeParityVerdict: node {node} has unknown status {status!r}")
        if self.decision == rec.STATUS_PASS:
            missing = [name for name in self.REQUIRED_SUB_PATHS if name not in self.sub_path_status]
            if missing:
                findings.append(
                    "TrainServeParityVerdict: PASS requires a status for every sub-path "
                    f"(一个总 PASS 不能隐藏 quant/engine 子路径失败); missing: {', '.join(missing)}"
                )
            if self.gate_summary.get("all_passed") is not True:
                findings.append("TrainServeParityVerdict: PASS requires all seven parity layers to pass")
            if not self.negative_cases:
                findings.append("TrainServeParityVerdict: PASS requires the wrong-artefact cases to be rejected")
            if not self.source_to_serving_lineage:
                findings.append("TrainServeParityVerdict: PASS requires a resolvable source→serving lineage")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "verdict_id": self.verdict_id,
            "decision": self.decision,
            "node_status": {key: self.node_status[key] for key in sorted(self.node_status)},
            "gate_summary": dict(self.gate_summary),
            "source_to_serving_lineage": list(self.source_to_serving_lineage),
            "sub_path_status": {key: self.sub_path_status[key] for key in sorted(self.sub_path_status)},
            "negative_cases": [dict(item) for item in self.negative_cases],
            "limitations": list(self.limitations),
            "digest": canonical_digest(
                {
                    "verdict_id": self.verdict_id,
                    "decision": self.decision,
                    "sub_path_status": {k: self.sub_path_status[k] for k in sorted(self.sub_path_status)},
                    "gate_digest": self.gate_summary.get("digest", ""),
                }
            ),
        }


def required_serving_artifact_fields() -> Dict[str, Any]:
    """The target contract of step 2, plus the artifact chain of §7.3.

    Exposed as data so the report can cite the target contract instead of
    paraphrasing it, and so a missing field is a failing test rather than a
    paragraph.
    """
    artifact = ServingModelArtifact(
        serving_artifact_id="serving::sha256:" + "0" * 64,
        source_checkpoint_id="checkpoint::sha256:" + "0" * 64,
        conversion_graph_id="conversion-graph::sha256:" + "0" * 64,
    )
    return {
        "artifact_chain": artifact_hierarchy(),
        "pre_quant_chain": artifact_hierarchy(quantized=False, compiled=False),
        "full_chain": artifact_hierarchy(quantized=True, compiled=True),
        "target_fields": sorted(artifact.as_dict()),
    }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the E14-03 interfaces (labelled smoke, not an experiment)."""
    mapping = validate_mapping(
        [
            TensorMappingRow("model.layers.0.q_proj.weight", "layers.0.attn.qkv.weight", "concat",
                             shape_before=(896, 896), shape_after=(2688, 896)),
            TensorMappingRow("model.layers.0.k_proj.weight", "layers.0.attn.qkv.weight", "concat",
                             shape_before=(128, 896), shape_after=(2688, 896)),
        ],
        expected_targets=("layers.0.attn.qkv.weight",),
        allow_many_to_one=True,
    )
    overwrite = validate_mapping(
        [
            TensorMappingRow("a", "t"),
            TensorMappingRow("b", "t"),
        ],
        expected_targets=("t",),
    )
    gates = [
        ParityGate(layer="inventory", tolerance={"missing": 0}, status=rec.STATUS_PASS),
        ParityGate(layer="tensor", tolerance={"max_abs": 1e-5}, status=rec.STATUS_FAIL_CORRECTNESS),
        ParityGate(layer="block", tolerance={"max_abs": 1e-4}, status=rec.STATUS_NOT_RUN),
    ]
    adapter = AdapterSemantics(
        adapter_id="a1", base_artifact_id="b1", target_modules=("q_proj",), rank=8, alpha=16.0
    )
    cache = check_conversion_cache(
        cold={"tensor_digest": "x"}, warm={"tensor_digest": "x", "cache_key": "k", "cache_hit": True,
                                           "cache_reason": "key match"},
        semantic_fields=("tensor_digest",),
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "many_to_one_ok": mapping["ok"],
        "silent_overwrite_rejected": not overwrite["ok"],
        "first_failing_layer": evaluate_gates(gates)["first_failure"],
        "layers_after_failure": evaluate_gates(gates)["layers_after_failure"],
        "adapter_problems": adapter.problems(),
        "negative_cases": len(negative_artifact_matrix()),
        "cache_consistent": cache["consistent"],
        "parity_layers": len(PARITY_LAYERS),
    }


# ── result accessors ───────────────────────────────────────────────────────

def mapping_missing_targets(result: Mapping[str, Any]) -> Tuple[str, ...]:
    """Step 10: architecture-required tensors that no mapping row produces."""
    return tuple(result.get("missing_targets", ()))


def gates_first_failure(result: Mapping[str, Any]) -> Optional[str]:
    """Step 23: the earliest semantic layer that failed (later layers never repair it)."""
    return result.get("first_failure")


# ── protocol step table (40 steps of details/S14/E14-03) ───────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "冻结 source CheckpointArtifact", ("contracts:CheckpointArtifact", "contracts:CheckpointArtifact.aggregate_root")),
    (2, "冻结目标 ServingModelArtifact 契约", ("parity:required_serving_artifact_fields",
                                              "contracts:artifact_hierarchy")),
    (3, "冻结转换 DAG", ("parity:ConversionNodeEvent", "parity:CONVERSION_NODES", "identity:LineageChain")),
    (4, "冻结唯一 intended path", ("parity:REFERENCE_PATHS", "identity:LineageChain.validate")),
    (5, "冻结 reference workload", ("telemetry:project_conversion_node", "records:TABLE_SCHEMAS")),
    (6, "冻结精度与容差", ("parity:ParityGate.tolerance", "records:REASON_CODES")),
    (7, "冻结 adapter/merge 语义", ("parity:AdapterSemantics", "parity:AdapterSemantics.problems")),
    (8, "冻结 quant 前提", ("contracts:ServingModelArtifact.quant_artifact_id", "records:REASON_CODES")),
    (9, "盘点 source inventory", ("identity:file_inventory", "identity:content_address_aggregate")),
    (10, "建立 architecture 参数规范", ("identity:artifact_ref", "parity:mapping_missing_targets")),
    (11, "生成显式 mapping table", ("parity:TensorMappingRow", "parity:validate_mapping")),
    (12, "执行 source 原生 eval baseline", ("parity:REFERENCE_PATHS", "telemetry:project_conversion_node")),
    (13, "校验 source checkpoint 可完整加载", ("contracts:CheckpointArtifact.missing_state",
                                                "training:CheckpointInventory.verify")),
    (14, "执行分片 gather/reshard", ("parity:TRANSFORMS", "training:reshard_capability")),
    (15, "执行 rename/transpose/fusion", ("parity:TensorMappingRow.permute", "parity:validate_mapping")),
    (16, "执行 adapter unmerged 路径", ("parity:AdapterSemantics.merged", "parity:REFERENCE_PATHS")),
    (17, "执行 adapter merge", ("parity:AdapterSemantics.merge_count", "parity:AdapterSemantics.merge_dtype")),
    (18, "比较 merged 与 unmerged", ("parity:compare_merged_unmerged", "parity:ParityGate.tolerance")),
    (19, "执行 full-precision 通用格式导出", ("contracts:artifact_hierarchy", "identity:LineageChain.add")),
    (20, "验证输出 inventory 完整性", ("parity:validate_mapping", "contracts:ServingModelArtifact.aggregate_root")),
    (21, "运行 tensor-level diff", ("training:reconcile_tensors", "parity:ParityGate")),
    (22, "运行 block-level parity", ("parity:PARITY_LAYERS", "parity:evaluate_gates")),
    (23, "运行 logits/top-k parity", ("parity:gates_first_failure", "parity:PARITY_LAYERS")),
    (24, "运行 greedy token parity", ("parity:PARITY_LAYERS", "records:TABLE_SCHEMAS")),
    (25, "运行 full-precision task quality", ("contracts:check_quality_before_performance",
                                               "parity:ParityGate.status")),
    (26, "执行可选 quant 节点", ("contracts:ServingModelArtifact.quant_artifact_id",
                                  "contracts:ServingModelArtifact.precision_contract_id")),
    (27, "验证 quant tensor/scale/packing", ("parity:TensorMappingRow.dtype_to", "parity:DTYPES")),
    (28, "运行 quant correctness/quality gate", ("contracts:check_quality_before_performance",
                                                  "records:STATUS_FAIL_QUALITY")),
    (29, "执行 runtime load", ("contracts:ServingModelArtifact.runtime_id",
                                "contracts:check_actual_path_recorded")),
    (30, "执行可选 engine build", ("contracts:ServingModelArtifact.engine_artifact_id",
                                    "contracts:ServingModelArtifact.compatibility_digest")),
    (31, "验证 runtime/engine 语义", ("parity:evaluate_gates", "contracts:ServingModelArtifact.status")),
    (32, "验证 serving readiness gate", ("contracts:ServingModelArtifact.quality_evidence_id",
                                          "records:STATUS_INVALID_IDENTITY")),
    (33, "注入缺失/损坏 shard", ("parity:negative_artifact_matrix", "identity:content_address_aggregate")),
    (34, "注入错误 tokenizer/chat template", ("parity:negative_artifact_matrix",
                                               "contracts:ServingModelArtifact.tokenizer_artifact_id")),
    (35, "注入错误 config/precision/version", ("parity:negative_artifact_matrix",
                                                "contracts:ServingModelArtifact.config_artifact_id")),
    (36, "注入错误 adapter/base 组合", ("parity:negative_artifact_matrix", "parity:AdapterSemantics.base_artifact_id")),
    (37, "验证并发转换和原子发布", ("parity:check_atomic_publish", "identity:LineageChain.validate")),
    (38, "验证重复转换与 cache", ("parity:check_conversion_cache", "identity:canonical_digest")),
    (39, "测转换/加载资源和成本", ("parity:audit_conversion_resources", "parity:CONVERSION_RESOURCE_KEYS")),
    (40, "形成 TrainServeParityVerdict", ("parity:TrainServeParityVerdict",
                                           "parity:TrainServeParityVerdict.validate")),
)
