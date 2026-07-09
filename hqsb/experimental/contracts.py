"""Cross-cutting S14 invariants and the seven unified evidence objects.

``details/S14/README.md`` §7 defines the evidence objects every S14 experiment
must be able to produce, and §8–§19 defines the invariants each one has to
satisfy.  The objects are *contracts*: they can be constructed, validated and
serialised on the CPU, but none of them contains, computes or fabricates a
measurement.  A field with no observation stays ``None`` (the protocol's
"missing is a state, never a zero" rule, §21).

The seven objects:

===========================  ==========================================
object                       answers
===========================  ==========================================
``TrainingRunArtifact``      哪次训练产生了什么（§7.1）
``CheckpointArtifact``       checkpoint 的完整状态与保存语义（§7.2）
``ServingModelArtifact``     checkpoint→转换/merge/quant 后的推理制品（§7.3）
``PolicySnapshot``           rollout 由哪一版策略生成（§7.4）
``TrajectoryRecord``         一条样本的策略/reward/environment 身份（§7.5）
``FrontierStudyContract``    前沿实验究竟改了什么、测什么（§7.6）
``AdoptionDecision``         采用/不采用及其允许声明（§7.7）
===========================  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.identity import (
    SCHEMA_PREFIX,
    SeedBundle,
    canonical_digest,
    content_address_aggregate,
    is_digest,
)

# ── small shared validators ────────────────────────────────────────────────


def _require(payload: Mapping[str, Any], names: Sequence[str], *, owner: str) -> List[str]:
    return [name for name in names if payload.get(name) in (None, "", [], {})]


def _check_status(status: str, *, owner: str, allow_conclusion: bool = False) -> List[str]:
    if status not in rec.ALL_STATUSES:
        return [f"{owner}: unknown status {status!r}"]
    if not allow_conclusion and status in rec.CONCLUSION_STATUSES:
        return [
            f"{owner}: {status} is a conclusion; contracts carry NOT_RUN/initial states only "
            "(the driver writes conclusions, and only with --execute + prerequisites + raw samples)"
        ]
    return []


def _check_digest(value: str, *, name: str) -> List[str]:
    if value and not is_digest(value):
        return [f"{name} must be sha256:<hex>, got {value!r}"]
    return []


# ── §7.1 TrainingRunArtifact ───────────────────────────────────────────────


@dataclass
class TrainingRunArtifact:
    """§7.1 — which training run produced what."""

    training_run_id: str
    source_commit: str
    base_model_artifact_id: str
    algorithm: str = "sft"
    data_artifact_id: str = ""
    distributed_plan_id: str = ""
    seed_bundle: Optional[SeedBundle] = None
    precision_contract_id: str = ""
    optimizer_identity: str = ""
    step_range: Tuple[int, int] = (0, 0)
    checkpoint_ids: Tuple[str, ...] = ()
    metric_artifact_uri: str = ""
    profile_artifact_uri: str = ""
    status: str = rec.STATUS_NOT_RUN
    limitations: Tuple[str, ...] = ()

    schema_version = f"{SCHEMA_PREFIX}.training-run.v1"

    def validate(self) -> List[str]:
        problems = _check_status(self.status, owner="TrainingRunArtifact")
        if not self.training_run_id:
            problems.append("TrainingRunArtifact: training_run_id is required (a path is not an identity)")
        if not self.source_commit:
            problems.append("TrainingRunArtifact: source_commit is required")
        for name in ("base_model_artifact_id",):
            if not self.base_model_artifact_id:
                problems.append(f"TrainingRunArtifact: {name} is required")
        if self.algorithm not in rec.POSTTRAINING_ALGORITHMS + ("other",):
            problems.append(
                f"TrainingRunArtifact: algorithm {self.algorithm!r} must be one of "
                f"{', '.join(rec.POSTTRAINING_ALGORITHMS + ('other',))}"
            )
        if self.seed_bundle is None:
            problems.append("TrainingRunArtifact: seed_bundle is required (a bare seed is not reproducible)")
        start, end = self.step_range
        if start < 0 or end < start:
            problems.append(f"TrainingRunArtifact: step_range {self.step_range} is not a valid [start, end]")
        for checkpoint in self.checkpoint_ids:
            for problem in _check_digest(checkpoint, name="checkpoint_ids"):
                problems.append(f"TrainingRunArtifact: {problem}")
        if not self.checkpoint_ids and self.status == rec.STATUS_PASS:
            problems.append("TrainingRunArtifact: PASS without any checkpoint id is not a train->serve closure")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "training_run_id": self.training_run_id,
            "source_commit": self.source_commit,
            "base_model_artifact_id": self.base_model_artifact_id,
            "data_artifact_id": self.data_artifact_id,
            "algorithm": self.algorithm,
            "distributed_plan_id": self.distributed_plan_id,
            "seed_bundle": self.seed_bundle.as_dict() if self.seed_bundle else None,
            "precision_contract_id": self.precision_contract_id,
            "optimizer_identity": self.optimizer_identity,
            "step_range": list(self.step_range),
            "checkpoint_ids": list(self.checkpoint_ids),
            "metric_artifact_uri": self.metric_artifact_uri,
            "profile_artifact_uri": self.profile_artifact_uri,
            "status": self.status,
            "limitations": list(self.limitations),
        }


# ── §7.2 CheckpointArtifact ────────────────────────────────────────────────


@dataclass
class CheckpointArtifact:
    """§7.2 — a checkpoint is a state, not a path."""

    checkpoint_id: str
    training_run_id: str
    step: int
    inventory: Mapping[str, str] = field(default_factory=dict)
    optimizer_identity: str = ""
    scheduler_state: bool = False
    scaler_state: bool = False
    consumed_samples: int = 0
    rng_state_present: bool = False
    parallel_topology_digest: str = ""
    tokenizer_artifact_id: str = ""
    config_artifact_id: str = ""
    chat_template_digest: str = ""
    adapter_semantics: str = ""
    format_version: str = ""
    tool_versions: Mapping[str, str] = field(default_factory=dict)
    reshard_supported: bool = False
    atomic: bool = False
    status: str = rec.STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.checkpoint.v1"

    def aggregate_root(self) -> str:
        """One digest over the full inventory (order independent, content bound)."""
        return content_address_aggregate(self.inventory)

    def missing_state(self) -> List[str]:
        """The state components ``E14-02`` step 29 requires to call it complete."""
        absent: List[str] = []
        if not self.inventory:
            absent.append("model tensors")
        if not self.optimizer_identity:
            absent.append("optimizer tensors/state")
        if not self.scheduler_state:
            absent.append("lr scheduler")
        if not self.scaler_state:
            absent.append("gradient scaler")
        if not self.rng_state_present:
            absent.append("RNG states per rank/device/dataloader")
        if not self.parallel_topology_digest:
            absent.append("parallel/sharding topology")
        if not self.tokenizer_artifact_id:
            absent.append("tokenizer")
        if not self.config_artifact_id:
            absent.append("config")
        return absent

    def validate(self) -> List[str]:
        problems = _check_status(self.status, owner="CheckpointArtifact")
        if not self.checkpoint_id:
            problems.append("CheckpointArtifact: checkpoint_id is required")
        if self.step < 0:
            problems.append("CheckpointArtifact: step must be >= 0")
        if self.inventory:
            try:
                self.aggregate_root()
            except ConfigError as exc:
                problems.append(f"CheckpointArtifact: {exc}")
        # A checkpoint that claims PASS but cannot continue training is exactly
        # the §28 error 2 ("只保存 model weights 却声称可恢复训练").
        if self.status == rec.STATUS_PASS:
            absent = self.missing_state()
            if absent:
                problems.append(f"CheckpointArtifact: PASS while missing {', '.join(absent)}")
        if self.reshard_supported and not self.parallel_topology_digest:
            problems.append("CheckpointArtifact: reshard claimed without a recorded source topology")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "training_run_id": self.training_run_id,
            "step": self.step,
            "optimizer_identity": self.optimizer_identity,
            "scheduler_state": self.scheduler_state,
            "scaler_state": self.scaler_state,
            "consumed_samples": self.consumed_samples,
            "rng_state_present": self.rng_state_present,
            "parallel_topology_digest": self.parallel_topology_digest,
            "tokenizer_artifact_id": self.tokenizer_artifact_id,
            "config_artifact_id": self.config_artifact_id,
            "chat_template_digest": self.chat_template_digest,
            "adapter_semantics": self.adapter_semantics,
            "format_version": self.format_version,
            "tool_versions": dict(sorted(self.tool_versions.items())),
            "reshard_supported": self.reshard_supported,
            "atomic": self.atomic,
            "status": self.status,
            "inventory": dict(sorted(self.inventory.items())),
        }
        if self.inventory:
            payload["aggregate_root"] = self.aggregate_root()
        return payload


# ── §7.3 ServingModelArtifact ──────────────────────────────────────────────


def artifact_hierarchy(*, quantized: bool = False, compiled: bool = False) -> List[str]:
    """The S14 artifact chain (dependency order) for one serving artifact."""
    chain = ["base-checkpoint", "unmerged-or-merged-model", "full-precision-export"]
    if quantized:
        chain.append("quantized-artifact")
    if compiled:
        chain.append("compiled-engine")
    chain.append("serving-model-artifact")
    return chain


@dataclass
class ServingModelArtifact:
    """§7.3 — what the Runtime actually loads, and where it came from."""

    serving_artifact_id: str
    source_checkpoint_id: str
    conversion_graph_id: str
    converter_identity: str = ""
    tensor_mapping_digest: str = ""
    tokenizer_artifact_id: str = ""
    config_artifact_id: str = ""
    precision_contract_id: str = ""
    quant_artifact_id: str = ""
    engine_artifact_id: str = ""
    inventory: Mapping[str, str] = field(default_factory=dict)
    compatibility_digest: str = ""
    quality_evidence_id: str = ""
    c1_model_artifact_id: str = ""
    runtime_id: str = ""
    status: str = rec.STATUS_NOT_RUN
    limitations: Tuple[str, ...] = ()

    schema_version = f"{SCHEMA_PREFIX}.serving-model.v1"

    def aggregate_root(self) -> str:
        return content_address_aggregate(self.inventory)

    def validate(self) -> List[str]:
        problems = _check_status(self.status, owner="ServingModelArtifact")
        for name in ("serving_artifact_id", "source_checkpoint_id", "conversion_graph_id"):
            if not getattr(self, name):
                problems.append(f"ServingModelArtifact: {name} is required (an output path is not an identity)")
        for name in ("converter_identity", "tokenizer_artifact_id", "config_artifact_id"):
            if not getattr(self, name):
                problems.append(f"ServingModelArtifact: {name} is part of the model identity and is required")
        if self.quant_artifact_id and not self.precision_contract_id:
            problems.append("ServingModelArtifact: a quantized artifact must reference its precision contract (C5)")
        if self.engine_artifact_id and not self.compatibility_digest:
            problems.append(
                "ServingModelArtifact: a compiled engine must record device/arch/runtime compatibility "
                "(E14-03 step 30: 不能把 engine 当跨硬件通用文件)"
            )
        if self.status == rec.STATUS_PASS and not self.quality_evidence_id:
            problems.append("ServingModelArtifact: PASS without quality evidence is not a serving parity claim")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "serving_artifact_id": self.serving_artifact_id,
            "source_checkpoint_id": self.source_checkpoint_id,
            "conversion_graph_id": self.conversion_graph_id,
            "converter_identity": self.converter_identity,
            "tensor_mapping_digest": self.tensor_mapping_digest,
            "tokenizer_artifact_id": self.tokenizer_artifact_id,
            "config_artifact_id": self.config_artifact_id,
            "precision_contract_id": self.precision_contract_id,
            "quant_artifact_id": self.quant_artifact_id,
            "engine_artifact_id": self.engine_artifact_id,
            "compatibility_digest": self.compatibility_digest,
            "quality_evidence_id": self.quality_evidence_id,
            "c1_model_artifact_id": self.c1_model_artifact_id,
            "runtime_id": self.runtime_id,
            "status": self.status,
            "limitations": list(self.limitations),
            "inventory": dict(sorted(self.inventory.items())),
        }
        if self.inventory:
            payload["aggregate_root"] = self.aggregate_root()
        return payload


# ── §7.4 PolicySnapshot ────────────────────────────────────────────────────


@dataclass
class PolicySnapshot:
    """§7.4 — which policy version generated a rollout."""

    policy_snapshot_id: str
    source_checkpoint_id: str
    optimizer_step: int
    weight_aggregate_digest: str = ""
    tokenizer_artifact_id: str = ""
    generation_config_digest: str = ""
    rollout_engine_id: str = ""
    rollout_backend: str = ""
    rollout_precision: str = ""
    loaded_at_ns: int = 0
    active_from_ns: int = 0
    active_until_ns: int = 0
    transfer_event_id: str = ""
    transfer_verified: bool = False
    status: str = rec.STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.policy-snapshot.v1"

    def is_active_at(self, timestamp_ns: int) -> bool:
        """Half-open ``[active_from, active_until)``; never "latest wins"."""
        if self.active_from_ns and timestamp_ns < self.active_from_ns:
            return False
        if self.active_until_ns and timestamp_ns >= self.active_until_ns:
            return False
        return True

    def validate(self) -> List[str]:
        problems = _check_status(self.status, owner="PolicySnapshot")
        for name in ("policy_snapshot_id", "source_checkpoint_id"):
            if not getattr(self, name):
                problems.append(f"PolicySnapshot: {name} is required")
        if self.optimizer_step < 0:
            problems.append("PolicySnapshot: optimizer_step must be >= 0")
        if not self.weight_aggregate_digest:
            problems.append("PolicySnapshot: weight_aggregate_digest is required (E14-04 step 23)")
        elif not is_digest(self.weight_aggregate_digest):
            problems.append("PolicySnapshot: weight_aggregate_digest must be sha256:<hex>")
        if not self.generation_config_digest:
            problems.append("PolicySnapshot: generation_config_digest is required (sampling affects the distribution)")
        if self.active_until_ns and self.active_from_ns and self.active_until_ns < self.active_from_ns:
            problems.append("PolicySnapshot: active_until_ns precedes active_from_ns")
        if self.transfer_event_id and not self.transfer_verified:
            problems.append(
                "PolicySnapshot: a weight transfer event must be verified in the inference engine "
                "(E14-04 step 23: 传输 API 成功不等于部分 rank/参数已更新)"
            )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_snapshot_id": self.policy_snapshot_id,
            "source_checkpoint_id": self.source_checkpoint_id,
            "optimizer_step": self.optimizer_step,
            "weight_aggregate_digest": self.weight_aggregate_digest,
            "tokenizer_artifact_id": self.tokenizer_artifact_id,
            "generation_config_digest": self.generation_config_digest,
            "rollout_engine_id": self.rollout_engine_id,
            "rollout_backend": self.rollout_backend,
            "rollout_precision": self.rollout_precision,
            "loaded_at_ns": self.loaded_at_ns,
            "active_from_ns": self.active_from_ns,
            "active_until_ns": self.active_until_ns,
            "transfer_event_id": self.transfer_event_id,
            "transfer_verified": self.transfer_verified,
            "status": self.status,
        }


# ── §7.5 TrajectoryRecord ──────────────────────────────────────────────────


@dataclass
class TrajectoryRecord:
    """§7.5 — one sample's full policy/reward/environment lineage."""

    trajectory_id: str
    sample_id: str
    policy_snapshot_id: str
    prompt_token_hash: str = ""
    response_token_hash: str = ""
    sampling_config_hash: str = ""
    reference_snapshot_id: Optional[str] = None
    reward_definition_id: str = ""
    environment_version: str = ""
    logprob_artifact_uri: str = ""
    reward_components: Mapping[str, float] = field(default_factory=dict)
    termination_reason: str = "eos"
    produced_at_step: int = 0
    consumed_at_step: Optional[int] = None
    staleness_steps: Optional[int] = None
    valid: bool = True
    rejection_reason: str = ""
    status: str = rec.STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.trajectory.v1"

    TERMINATION_REASONS: Tuple[str, ...] = ("eos", "length", "tool", "error", "cancel")

    def compute_staleness(self) -> Optional[int]:
        if self.consumed_at_step is None:
            return None
        return self.consumed_at_step - self.produced_at_step

    def validate(self, *, max_staleness: Optional[int] = None) -> List[str]:
        problems = _check_status(self.status, owner="TrajectoryRecord")
        for name in ("trajectory_id", "sample_id", "policy_snapshot_id"):
            if not getattr(self, name):
                problems.append(f"TrajectoryRecord: {name} is required (样本无 policy 身份即 FAIL)")
        for name in ("prompt_token_hash", "response_token_hash", "sampling_config_hash"):
            value = getattr(self, name)
            if value and not is_digest(value):
                problems.append(f"TrajectoryRecord: {name} must be sha256:<hex>")
        if self.termination_reason not in self.TERMINATION_REASONS:
            problems.append(
                f"TrajectoryRecord: termination_reason {self.termination_reason!r} must be one of "
                f"{', '.join(self.TERMINATION_REASONS)}"
            )
        if self.termination_reason == "error" and self.valid and self.reward_components:
            problems.append(
                "TrajectoryRecord: an errored trajectory must not carry reward components "
                "(E14-04 step 33: 超时样本不得被赋默认 reward 进入训练)"
            )
        if self.consumed_at_step is not None and self.consumed_at_step < self.produced_at_step:
            problems.append("TrajectoryRecord: consumed_at_step precedes produced_at_step")
        expected = self.compute_staleness()
        if expected is not None:
            if expected < 0:
                problems.append(f"TrajectoryRecord: negative staleness {expected}")
            if self.staleness_steps is not None and self.staleness_steps != expected:
                problems.append(
                    f"TrajectoryRecord: recorded staleness {self.staleness_steps} != computed {expected} "
                    "(staleness_steps = consume_step - produce_policy_step)"
                )
            if max_staleness is not None and expected > max_staleness:
                problems.append(
                    f"TrajectoryRecord: staleness {expected} exceeds the preregistered bound {max_staleness}"
                )
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "sample_id": self.sample_id,
            "policy_snapshot_id": self.policy_snapshot_id,
            "reference_snapshot_id": self.reference_snapshot_id,
            "reward_definition_id": self.reward_definition_id,
            "environment_version": self.environment_version,
            "prompt_token_hash": self.prompt_token_hash,
            "response_token_hash": self.response_token_hash,
            "sampling_config_hash": self.sampling_config_hash,
            "logprob_artifact_uri": self.logprob_artifact_uri,
            "reward_components": dict(sorted(self.reward_components.items())),
            "termination_reason": self.termination_reason,
            "produced_at_step": self.produced_at_step,
            "consumed_at_step": self.consumed_at_step,
            "staleness_steps": self.staleness_steps,
            "valid": self.valid,
            "rejection_reason": self.rejection_reason,
            "status": self.status,
        }


def check_batch_version_consistency(
    trajectories: Sequence[TrajectoryRecord], *, declared_snapshot_id: str
) -> List[str]:
    """``E14-04`` step 29 — refuse a silently mixed-version batch.

    A batch may mix versions only when it *declares* that it does; a batch that
    declares one policy and contains another is the failure mode the protocol
    forbids, so it is reported rather than averaged away.
    """
    problems: List[str] = []
    actual = sorted({item.policy_snapshot_id for item in trajectories})
    if declared_snapshot_id and any(snapshot != declared_snapshot_id for snapshot in actual):
        problems.append(
            f"batch declares policy {declared_snapshot_id} but contains {', '.join(actual)} "
            "(mixed-version batch without explicit stratification)"
        )
    if not declared_snapshot_id and len(actual) > 1:
        problems.append(
            f"batch mixes {len(actual)} policy snapshots without declaring a stratified design: {', '.join(actual)}"
        )
    return problems


# ── §7.6 FrontierStudyContract ─────────────────────────────────────────────


@dataclass
class FrontierStudyContract:
    """§7.6 / ``E14-05`` step 40 — the frozen, hashed study protocol."""

    study_id: str
    selected_branch: str
    primary_estimand: str
    primary_hypothesis: str
    minimum_effect: str
    baseline_id: str
    candidate_id: str
    intended_difference: str
    quality_gate_id: str
    source_versions: Tuple[str, ...] = ()
    workload_strata: Tuple[str, ...] = ()
    profile_layers: Tuple[str, ...] = ()
    negative_controls: Tuple[str, ...] = ()
    exploration_budget: Mapping[str, Any] = field(default_factory=dict)
    holdout_id: str = ""
    stop_rules: Tuple[str, ...] = ()
    adoption_rule_id: str = ""
    forbidden_claims: Tuple[str, ...] = ()
    status: str = rec.STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.frontier-study.v1"

    def contract_sha256(self) -> str:
        return canonical_digest(self.as_dict(include_hash=False))

    def validate(self) -> List[str]:
        problems = _check_status(self.status, owner="FrontierStudyContract")
        if self.selected_branch not in rec.FRONTIER_BRANCH_NAMES:
            problems.append(
                f"FrontierStudyContract: selected_branch {self.selected_branch!r} must be exactly one of "
                f"{', '.join(rec.FRONTIER_BRANCH_NAMES)}（只允许一个主分支）"
            )
        required = {
            "primary_estimand": self.primary_estimand,
            "primary_hypothesis": self.primary_hypothesis,
            "minimum_effect": self.minimum_effect,
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "intended_difference": self.intended_difference,
            "quality_gate_id": self.quality_gate_id,
            "holdout_id": self.holdout_id,
            "adoption_rule_id": self.adoption_rule_id,
        }
        for name, value in required.items():
            if not value:
                problems.append(f"FrontierStudyContract: {name} is required")
        if self.baseline_id and self.baseline_id == self.candidate_id:
            problems.append("FrontierStudyContract: baseline and candidate must differ by one intended change")
        if not self.source_versions:
            problems.append("FrontierStudyContract: at least one paper/official source version is required")
        if not self.workload_strata:
            problems.append("FrontierStudyContract: workload_strata are required (a single friendly shape is not a study)")
        if not self.profile_layers:
            problems.append("FrontierStudyContract: at least two profile layers are required (E14-05 step 24)")
        elif len(set(self.profile_layers)) < 2:
            problems.append("FrontierStudyContract: profile_layers must cover at least two distinct layers")
        unknown_layers = [layer for layer in self.profile_layers if layer not in rec.PROFILE_LAYERS]
        if unknown_layers:
            problems.append(f"FrontierStudyContract: unknown profile layers {', '.join(unknown_layers)}")
        if not self.negative_controls:
            problems.append(
                "FrontierStudyContract: negative controls are required (E14-05 step 21: 避免门禁永远只见正常样本)"
            )
        if not self.exploration_budget:
            problems.append("FrontierStudyContract: an exploration budget is required (unbounded search is a FAIL)")
        if not self.stop_rules:
            problems.append("FrontierStudyContract: stop/abort rules are required")
        return problems

    def as_dict(self, *, include_hash: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "selected_branch": self.selected_branch,
            "source_versions": list(self.source_versions),
            "primary_estimand": self.primary_estimand,
            "primary_hypothesis": self.primary_hypothesis,
            "minimum_effect": self.minimum_effect,
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "intended_difference": self.intended_difference,
            "quality_gate_id": self.quality_gate_id,
            "workload_strata": list(self.workload_strata),
            "profile_layers": list(self.profile_layers),
            "negative_controls": list(self.negative_controls),
            "exploration_budget": dict(sorted(self.exploration_budget.items())),
            "holdout_id": self.holdout_id,
            "stop_rules": list(self.stop_rules),
            "adoption_rule_id": self.adoption_rule_id,
            "forbidden_claims": list(self.forbidden_claims),
            "status": self.status,
        }
        if include_hash:
            payload["contract_sha256"] = canonical_digest(payload)
        return payload


def supersede_contract(
    previous: FrontierStudyContract, updated: FrontierStudyContract
) -> Dict[str, Any]:
    """``E14-05`` step 40 — a protocol edit creates a new version.

    Overwriting the old protocol would erase the research-drift evidence, so the
    old hash is retained and the *confirmation eligibility* is invalidated.
    """
    previous_hash = previous.contract_sha256()
    new_hash = updated.contract_sha256()
    return {
        "previous_contract_sha256": previous_hash,
        "new_contract_sha256": new_hash,
        "changed": previous_hash != new_hash,
        "previous_confirmation_eligible": False,
        "reason": "protocol version superseded; previous confirmatory runs are no longer eligible",
    }


# ── §7.7 AdoptionDecision ──────────────────────────────────────────────────


@dataclass
class AdoptionDecision:
    """§7.7 — adoption is a multi-objective decision, not ``speedup > 1``."""

    decision_id: str
    experiment_id: str
    decision: str
    allowed_claims: Tuple[str, ...] = ()
    forbidden_claims: Tuple[str, ...] = ()
    quality_status: str = rec.STATUS_NOT_RUN
    performance_status: str = rec.STATUS_NOT_RUN
    maturity: str = rec.MATURITY_DESIGN_ONLY
    evidence_refs: Tuple[str, ...] = ()
    limitations: Tuple[str, ...] = ()
    reopened_if: Tuple[str, ...] = ()

    schema_version = f"{SCHEMA_PREFIX}.adoption.v1"

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.decision not in rec.ADOPTION_DECISIONS:
            problems.append(
                f"AdoptionDecision: decision {self.decision!r} must be one of {', '.join(rec.ADOPTION_DECISIONS)}"
            )
        if self.maturity not in rec.MATURITY_LEVELS:
            problems.append(f"AdoptionDecision: unknown maturity {self.maturity!r}")
        if self.decision in rec.NON_ADOPTION_DECISIONS:
            if self.allowed_claims:
                problems.append(
                    f"AdoptionDecision: {self.decision} may not carry allowed_claims "
                    "(未选方向不能产生已支持 claim)"
                )
            if not self.reopened_if and self.decision != rec.BLOCKED_EVIDENCE:
                problems.append(
                    f"AdoptionDecision: {self.decision} must state the condition that would reopen the branch"
                )
        if self.decision == rec.ADOPT_CORE and self.maturity != rec.MATURITY_ADOPTED_CORE:
            problems.append("AdoptionDecision: ADOPT_CORE requires maturity ADOPTED_CORE (through S13/S15 regression)")
        if self.decision == rec.ADOPT_EXPERIMENTAL and rec.maturity_rank(self.maturity) < rec.maturity_rank(
            rec.MATURITY_QUALITY_VERIFIED
        ):
            problems.append("AdoptionDecision: ADOPT_EXPERIMENTAL requires quality-verified evidence at minimum")
        if self.decision in (rec.RESEARCH_ONLY, rec.REJECT_NO_BENEFIT) and not self.evidence_refs:
            problems.append(
                f"AdoptionDecision: {self.decision} must reference raw evidence (a negative result needs data too)"
            )
        if self.decision == rec.REJECT_QUALITY and self.quality_status not in (
            rec.STATUS_FAIL_QUALITY,
            rec.STATUS_FAIL,
        ):
            problems.append("AdoptionDecision: REJECT_QUALITY requires a recorded quality failure")
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "experiment_id": self.experiment_id,
            "decision": self.decision,
            "allowed_claims": list(self.allowed_claims),
            "forbidden_claims": list(self.forbidden_claims),
            "quality_status": self.quality_status,
            "performance_status": self.performance_status,
            "maturity": self.maturity,
            "evidence_refs": list(self.evidence_refs),
            "limitations": list(self.limitations),
            "reopened_if": list(self.reopened_if),
        }


# ── cross-experiment invariants (§8–§19) ───────────────────────────────────


def check_quality_before_performance(
    *, correctness_status: str, quality_status: str, performance_eligible: bool
) -> List[str]:
    """``S14 README`` §19: quality gates precede any performance eligibility.

    Ordering is enforced, not advised: a performance row whose correctness or
    quality gate did not pass is refused (``E14-03`` §8.5, ``E14-F2`` step 36).
    """
    problems: List[str] = []
    if performance_eligible and correctness_status != rec.STATUS_PASS:
        problems.append(
            f"performance is marked eligible while correctness is {correctness_status!r}; "
            "correctness gate must pass first"
        )
    if performance_eligible and quality_status not in (rec.STATUS_PASS, ""):
        problems.append(
            f"performance is marked eligible while quality is {quality_status!r}; quality gate must pass first"
        )
    return problems


def check_actual_path_recorded(sample: Mapping[str, Any]) -> List[str]:
    """``S14 README`` §24.2 — a performance sample without its actual path is invalid.

    A fallback that did not record ``actual_backend``/``actual_kernel`` makes the
    run ``INVALID_IDENTITY`` (§21), never a fast result.
    """
    problems: List[str] = []
    for field_name in ("requested_implementation", "actual_implementation"):
        if not sample.get(field_name):
            problems.append(f"performance sample is missing {field_name!r} (requested/actual must both be recorded)")
    if not sample.get("feature_flags") and not sample.get("capability_reason"):
        problems.append(
            "performance sample records neither feature flags nor a capability reason: "
            "a silent fallback is indistinguishable from a real path"
        )
    return problems


def check_negative_control_coverage(controls: Sequence[str]) -> List[str]:
    """``S14 README`` §18.2 — the required negative controls."""
    required = {
        "valid_baseline": ("baseline", "valid_baseline", "reference"),
        "candidate": ("candidate",),
        "feature_off": ("feature_off", "flag_off", "disabled"),
        "wrong_artifact": ("wrong_artifact", "wrong_config", "wrong_version", "mismatch"),
        "unsupported_shape": ("unsupported_shape", "unsupported_capability"),
    }
    present = {item.lower() for item in controls}
    problems: List[str] = []
    for name, aliases in required.items():
        if not any(alias in present for alias in aliases):
            problems.append(f"negative-control set is missing '{name}' (accepted spellings: {', '.join(aliases)})")
    return problems


def check_profile_layering(layers: Sequence[str], *, minimum: int = 2) -> List[str]:
    """``S14 README`` §17: every mainline experiment covers at least two layers."""
    problems: List[str] = []
    unique = [layer for layer in dict.fromkeys(layers) if layer]
    if len(unique) < minimum:
        problems.append(f"at least {minimum} profile layers are required, got {len(unique)}")
    unknown = [layer for layer in unique if layer not in rec.PROFILE_LAYERS]
    if unknown:
        problems.append(f"unknown profile layers: {', '.join(unknown)}")
    return problems


def check_resource_ledger(ledger: Mapping[str, Any]) -> List[str]:
    """``S14 README`` §24.3 — the resource ledger that must accompany a claim.

    Missing entries stay ``None`` ("unknown"), but the *keys* must exist, so an
    unmeasured dimension is visible instead of implied to be zero.
    """
    required = (
        "device_memory_allocated_bytes",
        "device_memory_reserved_bytes",
        "device_memory_peak_bytes",
        "host_memory_bytes",
        "pinned_memory_bytes",
        "kv_or_cache_bytes",
        "collective_bytes",
        "power_w",
        "energy_j",
        "device_count",
    )
    return [f"resource ledger is missing key {name!r}" for name in required if name not in ledger]


def check_no_silent_degradation(events: Sequence[Mapping[str, Any]]) -> List[str]:
    """Hard rule 3 of the task: every fallback records requested/actual/reason."""
    problems: List[str] = []
    for index, event in enumerate(events):
        if not event.get("fallback"):
            continue
        for name in ("requested", "actual", "reason_code"):
            if not event.get(name):
                problems.append(f"fallback event {index} records no {name!r}: silent degradation is forbidden")
        reason = str(event.get("reason_code", ""))
        if reason and reason not in rec.REASON_CODES:
            problems.append(f"fallback event {index} uses unknown reason code {reason!r}")
    return problems


def check_noncapture_matrix(captures: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """``E14-01`` §3.3 / §8.2 — import purity is a testable property.

    ``captures`` maps ``environment_id → observation``; a core install that
    creates a device context, forks a worker, downloads a model or writes a
    cache is a FAIL regardless of how fast it imported.
    """
    problems: List[str] = []
    for environment, observation in sorted(captures.items()):
        for field_name in ("device_context_created", "child_process_count", "network_access", "cache_written"):
            if field_name not in observation:
                problems.append(f"{environment}: purity observation is missing {field_name!r}")
        if observation.get("device_context_created") is True:
            problems.append(f"{environment}: import created a device context")
        if int(observation.get("child_process_count", 0) or 0) > 0:
            problems.append(f"{environment}: import forked a child process")
        if observation.get("network_access"):
            problems.append(f"{environment}: import performed network access")
        if observation.get("cache_written"):
            problems.append(f"{environment}: import wrote to a cache")
    return problems


def core_golden_equivalence(
    core_only: Mapping[str, Any], all_extras_off: Mapping[str, Any], *, fields: Sequence[str]
) -> List[str]:
    """``E14-01`` steps 27/34 — feature-off behaviour must equal core-only.

    Compared field by field so the diff names the surface that drifted
    (C6/C7 output, stdout/stderr, exit code, written files — step 34).
    """
    problems: List[str] = []
    for field_name in fields:
        left, right = core_only.get(field_name), all_extras_off.get(field_name)
        if left != right:
            problems.append(f"field {field_name!r} differs: core-only={left!r} all-extras-off={right!r}")
    return problems


#: The seven objects, for the driver's ``--objects`` listing and the tests.
EVIDENCE_OBJECTS: Mapping[str, str] = {
    "TrainingRunArtifact": "details/S14/README.md §7.1",
    "CheckpointArtifact": "details/S14/README.md §7.2",
    "ServingModelArtifact": "details/S14/README.md §7.3",
    "PolicySnapshot": "details/S14/README.md §7.4",
    "TrajectoryRecord": "details/S14/README.md §7.5",
    "FrontierStudyContract": "details/S14/README.md §7.6",
    "AdoptionDecision": "details/S14/README.md §7.7",
}
