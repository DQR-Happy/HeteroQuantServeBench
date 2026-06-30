"""Comparison contracts and field classification (E12-01 §4, §6, §7).

A comparison contract is the frozen answer to "what is this estimand?":

``semantic_identity`` / ``quality`` / ``workload`` / ``timing`` /
``statistics`` / ``normalization``.

The contract is *machine-validated* (types, required fields, enums, units and
mutually exclusive options) rather than "YAML that looks right", and every
field of the candidate matrix is classified before any diff is computed:

``INVARIANT``
    Must be identical inside a comparison group; a difference makes the cells
    ``NOT_COMPARABLE``.
``ALLOWED_DIFFERENCE``
    Exactly the candidate difference the research question is about.
``CONDITIONALLY_NORMALIZABLE``
    May be handled by a *preregistered* stratification/denominator; yields
    ``CONDITIONAL`` with a formula, an allowed-analysis list and forbidden
    claims.
``REPORT_ONLY``
    Recorded, never used to decide comparability (e.g. display names).
``FORBIDDEN_DIFFERENCE``
    A difference that invalidates the whole comparison regardless of any
    normalization (e.g. different model revision).

Nothing here executes an experiment; the diff engine is CPU-only and
deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.evaluation.identity import canonical_hash, canonical_json, sha256_text

CONTRACT_SCHEMA_VERSION = "1.0.0"

# ── field classes ─────────────────────────────────────────────────────────

FIELD_INVARIANT = "INVARIANT"
FIELD_ALLOWED_DIFFERENCE = "ALLOWED_DIFFERENCE"
FIELD_CONDITIONALLY_NORMALIZABLE = "CONDITIONALLY_NORMALIZABLE"
FIELD_REPORT_ONLY = "REPORT_ONLY"
FIELD_FORBIDDEN_DIFFERENCE = "FORBIDDEN_DIFFERENCE"

FIELD_CLASSES: Tuple[str, ...] = (
    FIELD_INVARIANT,
    FIELD_ALLOWED_DIFFERENCE,
    FIELD_CONDITIONALLY_NORMALIZABLE,
    FIELD_REPORT_ONLY,
    FIELD_FORBIDDEN_DIFFERENCE,
)

# ── audit dimensions (E12-01 §6) ──────────────────────────────────────────

AUDIT_DIMENSIONS: Tuple[str, ...] = (
    "model_identity",
    "tokenizer",
    "input",
    "generation",
    "precision",
    "quant",
    "correctness",
    "quality",
    "workload",
    "timing",
    "warmup_cache",
    "backend",
    "service",
    "distributed",
    "system",
    "statistics",
)

#: Reason codes per dimension: used by verdicts, exclusion ledgers and the
#: illegal-join guard.  Stable strings, never free text.
REASON_CODES: Mapping[str, Tuple[str, ...]] = {
    "model_identity": ("MODEL_ARTIFACT_MISMATCH", "MODEL_REVISION_MISMATCH", "MODEL_CONFIG_MISMATCH"),
    "tokenizer": ("TOKENIZER_MISMATCH", "CHAT_TEMPLATE_MISMATCH", "SPECIAL_TOKEN_MISMATCH"),
    "input": ("INPUT_TOKEN_IDS_MISMATCH", "ATTENTION_MASK_MISMATCH", "PADDING_MISMATCH"),
    "generation": ("SAMPLING_POLICY_MISMATCH", "STOP_POLICY_MISMATCH", "OUTPUT_LENGTH_MISMATCH"),
    "precision": ("PRECISION_CONTRACT_INCOMPLETE", "PRECISION_PATH_MISMATCH", "ACCUMULATION_MISMATCH"),
    "quant": ("QUANT_ARTIFACT_MISMATCH", "QUANT_CALIBRATION_MISMATCH", "QUANT_PACKING_MISMATCH"),
    "correctness": ("CORRECTNESS_GATE_MISMATCH", "TOLERANCE_RELAXED", "REFERENCE_MISMATCH"),
    "quality": ("QUALITY_GATE_FAILED", "QUALITY_GATE_MISMATCH", "QUALITY_EVIDENCE_STALE"),
    "workload": ("WORKLOAD_SPEC_MISMATCH", "LAYER_MIXED", "ISL_OSL_MISMATCH"),
    "timing": ("TIMING_BOUNDARY_MISMATCH", "SYNCHRONISATION_MISMATCH", "CLOCK_MISMATCH"),
    "warmup_cache": ("WARMUP_STATE_MISMATCH", "CACHE_STATE_MISMATCH", "COMPILE_STATE_MISMATCH"),
    "backend": ("ACTUAL_BACKEND_UNKNOWN", "ACTUAL_BACKEND_MISMATCH", "SILENT_FALLBACK"),
    "service": ("SERVICE_LOAD_MODE_MISMATCH", "ARRIVAL_PROCESS_MISMATCH", "SLO_MISMATCH", "TOKEN_ACCOUNTING_MISMATCH"),
    "distributed": ("DISTRIBUTED_TOPOLOGY_MISMATCH", "PARALLEL_PLAN_MISMATCH", "DEVICE_COUNT_MISMATCH"),
    "system": ("POWER_POLICY_MISMATCH", "CLOCK_POLICY_MISMATCH", "PARTITION_MISMATCH", "THERMAL_STATE_MISMATCH"),
    "statistics": ("REPLICATION_UNIT_MISMATCH", "STOPPING_RULE_MISMATCH", "ESTIMATOR_MISMATCH", "INTERVAL_METHOD_MISMATCH"),
    "general": ("MISSING_EVIDENCE", "UNIT_MISMATCH", "CONTRACT_VERSION_CHANGED", "POLICY_VIOLATION"),
}

ALL_REASON_CODES: Tuple[str, ...] = tuple(
    sorted({code for codes in REASON_CODES.values() for code in codes})
)


def reason_codes_for(dimension: str) -> Tuple[str, ...]:
    return REASON_CODES.get(dimension, ())


# ── field catalog ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FieldSpec:
    """One auditable matrix field plus its default classification."""

    field_path: str
    dimension: str
    field_class: str
    rationale: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field_path": self.field_path,
            "dimension": self.dimension,
            "field_class": self.field_class,
            "rationale": self.rationale,
        }


def _specs(rows: Iterable[Tuple[str, str, str, str]]) -> Tuple[FieldSpec, ...]:
    return tuple(FieldSpec(*row) for row in rows)


FIELD_CATALOG: Tuple[FieldSpec, ...] = _specs(
    [
        # model identity
        ("model_artifact_id", "model_identity", FIELD_INVARIANT, "同一比较组必须使用同一 ModelArtifact"),
        ("model.weights_root_hash", "model_identity", FIELD_INVARIANT, "权重根 hash 变化即换制品"),
        ("model.revision", "model_identity", FIELD_FORBIDDEN_DIFFERENCE, "revision 不同改变被测量对象"),
        ("tokenizer_id", "tokenizer", FIELD_INVARIANT, "tokenizer 不同则 tokens/s 不同分母"),
        ("tokenizer.chat_template_hash", "tokenizer", FIELD_INVARIANT, "模板改变输入语义"),
        ("input_token_ids_hash", "input", FIELD_INVARIANT, "固定文本不等于固定 token"),
        ("generation.stop_policy", "generation", FIELD_INVARIANT, "EOS 提前停止改变实际工作量"),
        ("generation.output_token_accounting", "generation", FIELD_INVARIANT, "分母口径必须一致"),
        ("precision.weight_dtype", "precision", FIELD_INVARIANT, "权重 dtype 属于被测对象"),
        ("precision.activation_dtype", "precision", FIELD_INVARIANT, ""),
        ("precision.accumulation_dtype", "precision", FIELD_INVARIANT, "累加精度不可只写 FP16"),
        ("precision.kv_dtype", "precision", FIELD_INVARIANT, ""),
        ("quant_artifact_id", "quant", FIELD_ALLOWED_DIFFERENCE, "量化候选是被比较对象"),
        ("quality_gate_id", "quality", FIELD_INVARIANT, "质量门必须共同"),
        ("quality.threshold_policy", "quality", FIELD_INVARIANT, "平台不得私自放宽"),
        ("correctness_gate_id", "correctness", FIELD_INVARIANT, ""),
        ("workload_spec_id", "workload", FIELD_INVARIANT, "工作量必须相同"),
        ("workload.layer", "workload", FIELD_FORBIDDEN_DIFFERENCE, "不同层不是同一估计量"),
        ("workload.scenario", "workload", FIELD_INVARIANT, ""),
        ("workload.batch", "workload", FIELD_ALLOWED_DIFFERENCE, "合法 sweep 维度"),
        ("workload.concurrency", "workload", FIELD_ALLOWED_DIFFERENCE, "closed-loop 场景维度"),
        ("workload.request_rate", "workload", FIELD_ALLOWED_DIFFERENCE, "open-loop 场景维度"),
        ("timing.clock", "timing", FIELD_INVARIANT, "monotonic 与 wall 不可混用"),
        ("timing.boundary", "timing", FIELD_INVARIANT, "client/server/model-core 边界不可混"),
        ("timing.synchronisation", "timing", FIELD_INVARIANT, ""),
        ("warmup.policy", "warmup_cache", FIELD_INVARIANT, "冷启动与稳态不可混排"),
        ("warmup.compile_state", "warmup_cache", FIELD_CONDITIONALLY_NORMALIZABLE, "可分层（cold/steady）但不可拼接最优"),
        ("cache.state", "warmup_cache", FIELD_CONDITIONALLY_NORMALIZABLE, ""),
        ("backend.requested", "backend", FIELD_ALLOWED_DIFFERENCE, "候选实现差异"),
        ("backend.actual", "backend", FIELD_CONDITIONALLY_NORMALIZABLE, "actual 未知时不得比较"),
        ("backend.fallback_reason", "backend", FIELD_REPORT_ONLY, "记录用，不参与裁决"),
        ("service.load_mode", "service", FIELD_FORBIDDEN_DIFFERENCE, "open-loop 与 closed-loop 不是同一估计量"),
        ("service.arrival_process", "service", FIELD_INVARIANT, ""),
        ("service.slo", "service", FIELD_INVARIANT, "goodput 依赖统一 SLO"),
        ("service.retry_policy", "service", FIELD_INVARIANT, ""),
        ("distributed.device_count", "distributed", FIELD_CONDITIONALLY_NORMALIZABLE, "scaling study 可条件比较"),
        ("distributed.topology_id", "distributed", FIELD_ALLOWED_DIFFERENCE, "拓扑是被测 SUT 的一部分"),
        ("distributed.parallel_plan_id", "distributed", FIELD_ALLOWED_DIFFERENCE, ""),
        ("distributed.scaling_kind", "distributed", FIELD_INVARIANT, "strong/weak/capacity 不可混"),
        ("system.power_cap_w", "system", FIELD_INVARIANT, "功耗上限改变性能"),
        ("system.clock_policy", "system", FIELD_INVARIANT, ""),
        ("system.partition", "system", FIELD_INVARIANT, "MIG/partition 改变容量"),
        ("system.thermal_start_c", "system", FIELD_REPORT_ONLY, "作为协变量记录"),
        ("statistics.unit_of_replication", "statistics", FIELD_INVARIANT, "token 不能冒充独立重复"),
        ("statistics.estimator", "statistics", FIELD_INVARIANT, ""),
        ("statistics.interval_method", "statistics", FIELD_INVARIANT, ""),
        ("statistics.stopping_rule", "statistics", FIELD_INVARIANT, ""),
        ("contract.policy_version", "statistics", FIELD_REPORT_ONLY, ""),
        ("display_name", "model_identity", FIELD_REPORT_ONLY, "显示名不是 join key"),
    ]
)

FIELD_BY_PATH: Mapping[str, FieldSpec] = {spec.field_path: spec for spec in FIELD_CATALOG}


def field_spec(field_path: str) -> Optional[FieldSpec]:
    return FIELD_BY_PATH.get(field_path)


def classify_field(field_path: str) -> str:
    spec = FIELD_BY_PATH.get(field_path)
    if spec is None:
        # Unknown fields are invariant by default: fail closed rather than
        # letting an unclassified field silently differ.
        return FIELD_INVARIANT
    return spec.field_class


# ── contract ──────────────────────────────────────────────────────────────


@dataclass
class NormalizationFormula:
    """A ``CONDITIONAL`` verdict's frozen normalization (E12-01 §7)."""

    formula_id: str
    formula: str
    inputs: Tuple[str, ...] = ()
    conditions: Tuple[str, ...] = ()
    residual_limitations: Tuple[str, ...] = ()
    allowed_analyses: Tuple[str, ...] = ()
    forbidden_claims: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.formula_id:
            problems.append("normalization formula needs an id")
        if not self.formula:
            problems.append("normalization formula needs a definition")
        if not self.conditions:
            problems.append("CONDITIONAL normalization must state its condition")
        if not self.allowed_analyses:
            problems.append("CONDITIONAL normalization must list allowed analyses")
        if not self.forbidden_claims:
            problems.append("CONDITIONAL normalization must state forbidden claims")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "formula_id": self.formula_id,
            "formula": self.formula,
            "inputs": list(self.inputs),
            "conditions": list(self.conditions),
            "residual_limitations": list(self.residual_limitations),
            "allowed_analyses": list(self.allowed_analyses),
            "forbidden_claims": list(self.forbidden_claims),
        }


@dataclass
class SemanticIdentity:
    model_artifact_id: str
    tokenizer_id: str
    chat_template_hash: str
    input_token_ids_hash: str
    stop_policy: str
    output_token_accounting: str = "accepted_generated_tokens"

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in (
            "model_artifact_id",
            "tokenizer_id",
            "chat_template_hash",
            "input_token_ids_hash",
            "stop_policy",
        ):
            if not getattr(self, name):
                problems.append(f"semantic_identity.{name} is required")
        if self.output_token_accounting not in (
            "accepted_generated_tokens",
            "generated_before_stop",
            "served_output_tokens",
        ):
            problems.append(
                f"unknown output_token_accounting {self.output_token_accounting!r}"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_artifact_id": self.model_artifact_id,
            "tokenizer_id": self.tokenizer_id,
            "chat_template_hash": self.chat_template_hash,
            "input_token_ids_hash": self.input_token_ids_hash,
            "stop_policy": self.stop_policy,
            "output_token_accounting": self.output_token_accounting,
        }


@dataclass
class QualityClause:
    gate_id: str
    metrics: Tuple[str, ...] = ()
    threshold_policy: str = ""
    split: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.gate_id:
            problems.append("quality.gate_id is required")
        if not self.metrics:
            problems.append("quality.metrics must be enumerated")
        if not self.threshold_policy:
            problems.append("quality.threshold_policy is required (no per-platform relaxation)")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "metrics": list(self.metrics),
            "threshold_policy": self.threshold_policy,
            "split": self.split,
        }


@dataclass
class WorkloadClause:
    workload_spec_id: str
    layer: str
    scenario: str
    isl: Optional[int] = None
    osl: Optional[int] = None
    batch: Optional[int] = None
    concurrency: Optional[int] = None
    request_rate: Optional[float] = None
    weights: Mapping[str, float] = field(default_factory=dict)

    def validate(self) -> List[str]:
        from hqsb.evaluation.layers import LAYERS, SCENARIOS

        problems: List[str] = []
        if not self.workload_spec_id:
            problems.append("workload.workload_spec_id is required")
        if self.layer not in LAYERS:
            problems.append(f"unknown layer {self.layer!r}")
        elif self.scenario not in SCENARIOS.get(self.layer, ()):
            problems.append(f"scenario {self.scenario!r} does not belong to layer {self.layer!r}")
        if self.weights:
            total = sum(float(value) for value in self.weights.values())
            if abs(total - 1.0) > 1e-6:
                problems.append(f"workload weights must sum to 1.0, got {total}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "workload_spec_id": self.workload_spec_id,
            "layer": self.layer,
            "scenario": self.scenario,
            "isl": self.isl,
            "osl": self.osl,
            "batch": self.batch,
            "concurrency": self.concurrency,
            "request_rate": self.request_rate,
            "weights": dict(sorted(self.weights.items())),
        }


@dataclass
class TimingClause:
    clock: str = "monotonic"
    boundary: str = ""
    synchronisation: str = ""
    warmup_policy: str = ""
    cooldown_policy: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.clock not in ("monotonic", "wall", "device_event"):
            problems.append(f"unknown timing clock {self.clock!r}")
        for name in ("boundary", "synchronisation", "warmup_policy", "cooldown_policy"):
            if not getattr(self, name):
                problems.append(f"timing.{name} must be frozen before the run")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "clock": self.clock,
            "boundary": self.boundary,
            "synchronisation": self.synchronisation,
            "warmup_policy": self.warmup_policy,
            "cooldown_policy": self.cooldown_policy,
        }


@dataclass
class StatisticsClause:
    unit_of_replication: str = ""
    stopping_rule: str = ""
    estimator: str = ""
    interval_method: str = ""
    confidence_level: float = 0.95
    min_repetitions: int = 3

    def validate(self) -> List[str]:
        problems: List[str] = []
        for name in ("unit_of_replication", "stopping_rule", "estimator", "interval_method"):
            if not getattr(self, name):
                problems.append(f"statistics.{name} must be frozen before the run")
        if not 0.0 < self.confidence_level < 1.0:
            problems.append("statistics.confidence_level must be inside (0, 1)")
        if self.min_repetitions < 3:
            problems.append(
                "statistics.min_repetitions must be >= 3 (handbook §5: >= 3 independent processes)"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "unit_of_replication": self.unit_of_replication,
            "stopping_rule": self.stopping_rule,
            "estimator": self.estimator,
            "interval_method": self.interval_method,
            "confidence_level": self.confidence_level,
            "min_repetitions": self.min_repetitions,
        }


@dataclass
class NormalizationClause:
    allowed: Tuple[str, ...] = ()
    forbidden: Tuple[str, ...] = ()

    def validate(self) -> List[str]:
        from hqsb.evaluation.layers import ALLOWED_NORMALIZATIONS, FORBIDDEN_NORMALIZATIONS

        problems: List[str] = []
        unknown = sorted(set(self.allowed) - set(ALLOWED_NORMALIZATIONS))
        if unknown:
            problems.append(f"unknown normalization kinds: {unknown}")
        illegal = sorted(set(self.forbidden) - set(FORBIDDEN_NORMALIZATIONS))
        if illegal:
            problems.append(f"undeclared forbidden normalizations: {illegal}")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {"allowed": list(self.allowed), "forbidden": list(self.forbidden)}


@dataclass
class ComparisonContract:
    """Frozen answer to "what estimand do these cells estimate?"."""

    comparison_id: str
    schema_version: str = CONTRACT_SCHEMA_VERSION
    semantic_identity: SemanticIdentity = None  # type: ignore[assignment]
    quality: QualityClause = None  # type: ignore[assignment]
    workload: WorkloadClause = None  # type: ignore[assignment]
    timing: TimingClause = field(default_factory=TimingClause)
    statistics: StatisticsClause = field(default_factory=StatisticsClause)
    normalization: NormalizationClause = field(default_factory=NormalizationClause)
    invariants: Tuple[str, ...] = ()
    allowed_differences: Tuple[str, ...] = ()
    policy_version: str = "s12_policy_1.0.0"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.comparison_id:
            problems.append("comparison_id is required")
        for name in ("semantic_identity", "quality", "workload"):
            clause = getattr(self, name)
            if clause is None:
                problems.append(f"contract.{name} is required before a comparison group is frozen")
                continue
            problems.extend(clause.validate())
        problems.extend(self.timing.validate())
        problems.extend(self.statistics.validate())
        problems.extend(self.normalization.validate())
        for field_path in list(self.invariants) + list(self.allowed_differences):
            if field_path not in FIELD_BY_PATH:
                problems.append(f"unknown contract field path {field_path!r}")
        both = sorted(set(self.invariants) & set(self.allowed_differences))
        if both:
            problems.append(f"fields cannot be invariant and an allowed difference at once: {both}")
        return problems

    def payload(self) -> Dict[str, Any]:
        return {
            "comparison_contract": {
                "schema_version": self.schema_version,
                "comparison_id": self.comparison_id,
                "semantic_identity": _as_dict(self.semantic_identity),
                "quality": _as_dict(self.quality),
                "workload": _as_dict(self.workload),
                "timing": _as_dict(self.timing),
                "statistics": _as_dict(self.statistics),
                "normalization": _as_dict(self.normalization),
                "invariants": sorted(self.invariants),
                "allowed_differences": sorted(self.allowed_differences),
                "policy_version": self.policy_version,
            }
        }

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.payload(), sort_keys=True, allow_unicode=True)

    def sha256(self) -> str:
        return sha256_text(canonical_json(self.payload()))

    def canonical_hash(self) -> str:
        return canonical_hash(self.payload())


def _as_dict(value: Any) -> Any:
    return value.as_dict() if hasattr(value, "as_dict") else value


def contract_from_mapping(payload: Mapping[str, Any]) -> ComparisonContract:
    """Build a contract from its serialised form (round-trip safe)."""
    body = payload.get("comparison_contract", payload)
    if not isinstance(body, Mapping):
        raise ConfigError("comparison contract must be a mapping")
    semantic = body.get("semantic_identity") or {}
    quality = body.get("quality") or {}
    workload = body.get("workload") or {}
    timing = body.get("timing") or {}
    statistics = body.get("statistics") or {}
    normalization = body.get("normalization") or {}
    return ComparisonContract(
        comparison_id=str(body.get("comparison_id", "")),
        schema_version=str(body.get("schema_version", CONTRACT_SCHEMA_VERSION)),
        semantic_identity=SemanticIdentity(
            model_artifact_id=str(semantic.get("model_artifact_id", "")),
            tokenizer_id=str(semantic.get("tokenizer_id", "")),
            chat_template_hash=str(semantic.get("chat_template_hash", "")),
            input_token_ids_hash=str(semantic.get("input_token_ids_hash", "")),
            stop_policy=str(semantic.get("stop_policy", "")),
            output_token_accounting=str(
                semantic.get("output_token_accounting", "accepted_generated_tokens")
            ),
        ),
        quality=QualityClause(
            gate_id=str(quality.get("gate_id", "")),
            metrics=tuple(quality.get("metrics", ())),
            threshold_policy=str(quality.get("threshold_policy", "")),
            split=str(quality.get("split", "")),
        ),
        workload=WorkloadClause(
            workload_spec_id=str(workload.get("workload_spec_id", "")),
            layer=str(workload.get("layer", "")),
            scenario=str(workload.get("scenario", "")),
            isl=workload.get("isl"),
            osl=workload.get("osl"),
            batch=workload.get("batch"),
            concurrency=workload.get("concurrency"),
            request_rate=workload.get("request_rate"),
            weights=dict(workload.get("weights", {}) or {}),
        ),
        timing=TimingClause(
            clock=str(timing.get("clock", "monotonic")),
            boundary=str(timing.get("boundary", "")),
            synchronisation=str(timing.get("synchronisation", "")),
            warmup_policy=str(timing.get("warmup_policy", "")),
            cooldown_policy=str(timing.get("cooldown_policy", "")),
        ),
        statistics=StatisticsClause(
            unit_of_replication=str(statistics.get("unit_of_replication", "")),
            stopping_rule=str(statistics.get("stopping_rule", "")),
            estimator=str(statistics.get("estimator", "")),
            interval_method=str(statistics.get("interval_method", "")),
            confidence_level=float(statistics.get("confidence_level", 0.95)),
            min_repetitions=int(statistics.get("min_repetitions", 3)),
        ),
        normalization=NormalizationClause(
            allowed=tuple(normalization.get("allowed", ())),
            forbidden=tuple(normalization.get("forbidden", ())),
        ),
        invariants=tuple(body.get("invariants", ())),
        allowed_differences=tuple(body.get("allowed_differences", ())),
        policy_version=str(body.get("policy_version", "s12_policy_1.0.0")),
    )


def default_contract(
    comparison_id: str = "cmp_s12_example",
    *,
    layer: str = "model_core",
    scenario: str = "decode",
) -> ComparisonContract:
    """A fully populated contract used by fixtures, smoke tests and templates."""
    from hqsb.evaluation.layers import ALLOWED_NORMALIZATIONS, FORBIDDEN_NORMALIZATIONS

    return ComparisonContract(
        comparison_id=comparison_id,
        semantic_identity=SemanticIdentity(
            model_artifact_id="qwen3-1.7b@frozen",
            tokenizer_id="qwen3-tokenizer@frozen",
            chat_template_hash="0" * 64,
            input_token_ids_hash="1" * 64,
            stop_policy="eos_or_max_new_tokens",
            output_token_accounting="accepted_generated_tokens",
        ),
        quality=QualityClause(
            gate_id="qwen3_lm_eval_smoke",
            metrics=("perplexity_delta", "task_accuracy_delta"),
            threshold_policy="common_s06",
            split="holdout",
        ),
        workload=WorkloadClause(
            workload_spec_id="ws_decode_512_128",
            layer=layer,
            scenario=scenario,
            isl=512,
            osl=128,
            batch=1,
        ),
        timing=TimingClause(
            clock="monotonic",
            boundary="model_core_forward",
            synchronisation="device_event",
            warmup_policy="load+compile+5_warmup",
            cooldown_policy="idle_30s",
        ),
        statistics=StatisticsClause(
            unit_of_replication="process",
            stopping_rule="min_3_processes_then_ci_width_2pct",
            estimator="median",
            interval_method="run_level_bootstrap",
        ),
        normalization=NormalizationClause(
            allowed=ALLOWED_NORMALIZATIONS,
            forbidden=FORBIDDEN_NORMALIZATIONS,
        ),
        invariants=tuple(
            spec.field_path for spec in FIELD_CATALOG if spec.field_class == FIELD_INVARIANT
        ),
        allowed_differences=tuple(
            spec.field_path
            for spec in FIELD_CATALOG
            if spec.field_class == FIELD_ALLOWED_DIFFERENCE
        ),
    )


# ── contract diff ─────────────────────────────────────────────────────────


@dataclass
class FieldDiff:
    """One field-level difference between two contract snapshots."""

    field_path: str
    field_class: str
    value_a_hash: str
    value_b_hash: str
    dimension: str
    match_status: str
    reason_code: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field_path": self.field_path,
            "field_class": self.field_class,
            "value_a_hash": self.value_a_hash,
            "value_b_hash": self.value_b_hash,
            "dimension": self.dimension,
            "match_status": self.match_status,
            "reason_code": self.reason_code,
        }


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key in sorted(payload):
        value = payload[key]
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


def diff_contracts(
    contract_a: ComparisonContract,
    contract_b: ComparisonContract,
    *,
    value_a: Optional[Mapping[str, Any]] = None,
    value_b: Optional[Mapping[str, Any]] = None,
) -> List[FieldDiff]:
    """Canonical field diff of two contracts (no textual diff).

    ``value_a``/``value_b`` allow auditing *candidate* field values against the
    contract (e.g. the actual backend of a run); when omitted the contract
    snapshots themselves are diffed.
    """
    left = _flatten(value_a if value_a is not None else contract_a.payload()["comparison_contract"])
    right = _flatten(value_b if value_b is not None else contract_b.payload()["comparison_contract"])
    rows: List[FieldDiff] = []
    for path in sorted(set(left) | set(right)):
        a = left.get(path)
        b = right.get(path)
        match = canonical_json(a) == canonical_json(b)
        if match:
            continue
        spec = field_spec(path) or field_spec(path.replace("comparison_contract.", ""))
        dimension = spec.dimension if spec else _dimension_for_path(path)
        field_class = spec.field_class if spec else FIELD_INVARIANT
        rows.append(
            FieldDiff(
                field_path=path,
                field_class=field_class,
                value_a_hash=sha256_text(canonical_json(a)),
                value_b_hash=sha256_text(canonical_json(b)),
                dimension=dimension,
                match_status="DIFF",
                reason_code=_reason_for(dimension, field_class),
            )
        )
    return rows


def _dimension_for_path(path: str) -> str:
    head = path.split(".")[0].replace("comparison_contract.", "")
    aliases = {
        "semantic_identity": "model_identity",
        "invariants": "statistics",
        "allowed_differences": "statistics",
        "policy_version": "statistics",
    }
    if head in aliases:
        return aliases[head]
    return head if head in AUDIT_DIMENSIONS else "general"


def _reason_for(dimension: str, field_class: str) -> str:
    codes = reason_codes_for(dimension) or reason_codes_for("general")
    if field_class == FIELD_FORBIDDEN_DIFFERENCE:
        return codes[0]
    if field_class == FIELD_CONDITIONALLY_NORMALIZABLE:
        return codes[-1]
    return codes[0]


def invariant_violations(diffs: Sequence[FieldDiff]) -> List[FieldDiff]:
    """Differences that make a group ``NOT_COMPARABLE`` outright."""
    return [
        diff
        for diff in diffs
        if diff.field_class in (FIELD_INVARIANT, FIELD_FORBIDDEN_DIFFERENCE)
    ]


def conditional_differences(diffs: Sequence[FieldDiff]) -> List[FieldDiff]:
    return [diff for diff in diffs if diff.field_class == FIELD_CONDITIONALLY_NORMALIZABLE]
