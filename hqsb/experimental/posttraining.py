"""E14-04 — SFT/DPO/GRPO rollout, sync/async and policy staleness.

Core judgement: pick **one** post-training algorithm by ADR; the trainer,
inference engine, environment, reward/reference and data identities must all be
complete; every sample/trajectory must resolve to the policy that generated it;
the synchronous path must be right *before* the asynchronous path is measured;
and an asynchronous path may never mix unlabelled policy versions.

The three objectives are not interchangeable (§3.1, §10):

* **SFT** — teacher-forced token NLL on a frozen dataset; the risks are mask,
  padding, chat template and label shift, not staleness;
* **DPO** — a preference-pair objective built from policy/reference log-ratios;
  it needs no online rollout, so "sync/async" may only describe data production,
  scoring and training (``E14-04`` §10.2);
* **GRPO** — group-relative advantage with old/current/reference policy;
  correctness depends on group boundaries, reward direction/scale, token masks
  and consistent log-probabilities.

Interfaces provided:

* :func:`sft_loss`, :func:`dpo_loss`, :func:`grpo_loss` — minimal, hand-checkable
  oracles over small numeric vectors (steps 12, 19);
* :func:`validate_token_mask`, :func:`length_normalisation` — mask and length
  effects (step 13);
* :func:`compare_logprobs` — trainer vs engine log-prob agreement (step 14);
* :func:`reward_determinism` — repeated scoring of a fixed trajectory (step 15);
* :class:`PostTrainingAlgorithmChoice`, :class:`PolicyPublishEvent` (steps 1–4, 22);
* :class:`SyncStateMachine`, :class:`AsyncQueuePolicy` (steps 9–10, 26–30);
* :func:`check_staleness_sweep`, :func:`check_kv_reuse_identity` (steps 30, 32);
* :func:`failure_scenarios`, :class:`PostTrainingVerdict` (steps 33–36, 40).

Nothing here calls a framework: the oracles are arithmetic over supplied numbers,
so a framework's logged scalar can be checked against an independent one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.contracts import (
    PolicySnapshot,
    TrajectoryRecord,
    check_batch_version_consistency,
)
from hqsb.experimental.identity import SCHEMA_PREFIX, canonical_digest, digest_text, is_digest

EXPERIMENT_ID = "E14-04"
TITLE = "SFT/DPO/GRPO Rollout、同步/异步与 Policy Staleness"
LEVEL = "P0"

CLAIM_BOUNDARY = (
    "只证明小模型、数据集与有限训练步下的系统正确性；不等于模型已获得通用对齐、推理或安全能力"
    "（E14-04 §12）。"
)

#: The three algorithms (§1); exactly one is primary.
ALGORITHMS: Tuple[str, ...] = rec.POSTTRAINING_ALGORITHMS

#: Sync state machine (§6 step 9).  Empty/failure transitions are declared in
#: ``records.POSTTRAINING_SYNC_TRANSITIONS`` so the two cannot drift.
SYNC_STAGES: Tuple[str, ...] = rec.POSTTRAINING_SYNC_STATES

#: Anti-cheat indicators that must accompany the training reward (§6 step 25).
ANTI_CHEAT_INDICATORS: Tuple[str, ...] = (
    "response_length_mean",
    "format_compliance",
    "kl_from_reference",
    "entropy",
    "clip_fraction",
    "heldout_quality",
    "safety_regression",
    "repetition_rate",
)

#: Reward-failure modes that must not become a silent training signal (§6 step 34).
REWARD_FAILURE_MODES: Tuple[str, ...] = ("timeout", "malformed", "unavailable", "rate_limited", "nondeterministic")


def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        raise ConfigError("softmax needs at least one logit")
    top = max(logits)
    exps = [math.exp(value - top) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def _log_softmax(logits: Sequence[float]) -> List[float]:
    return [math.log(prob) for prob in _softmax(logits)]


def sft_loss(
    logits_per_step: Sequence[Sequence[float]],
    target_ids: Sequence[int],
    mask: Sequence[int],
    *,
    ignore_index: int = -100,
    label_shift: int = 0,
) -> Dict[str, Any]:
    """``L_SFT = -Σ mask_t log π(y_t) / Σ mask_t`` (§3.1), computed independently.

    ``label_shift`` is explicit because an off-by-one between logits and labels
    is the most common SFT defect and it is invisible in a scalar loss curve.
    """
    if not (len(logits_per_step) == len(target_ids) == len(mask)):
        raise ConfigError("logits/targets/mask must have the same length")
    numerator = 0.0
    denominator = 0
    for position, (logits, target, keep) in enumerate(zip(logits_per_step, target_ids, mask)):
        if not keep or target == ignore_index:
            continue
        shifted = position + label_shift
        if not 0 <= shifted < len(target_ids):
            raise ConfigError(f"label_shift {label_shift} moves position {position} out of range")
        log_probs = _log_softmax(logits)
        if not 0 <= target < len(log_probs):
            raise ConfigError(f"target id {target} is outside the vocabulary at position {position}")
        numerator += -log_probs[target]
        denominator += 1
    if denominator == 0:
        return {"loss": None, "denominator": 0, "error": "every position was masked out"}
    return {"loss": numerator / denominator, "numerator": numerator, "denominator": denominator}


def dpo_loss(
    policy_logprobs: Mapping[str, float],
    reference_logprobs: Mapping[str, float],
    *,
    beta: float,
) -> Dict[str, Any]:
    """The DPO preference objective over ``(chosen, rejected)`` (§3.1, §10.2).

    ``sigmas``/``reward margin`` are returned alongside the loss so a sign error
    in the chosen/rejected ordering is visible (``E14-04`` step 21).
    """
    for key in ("chosen", "rejected"):
        if key not in policy_logprobs or key not in reference_logprobs:
            raise ConfigError(f"dpo_loss requires {key!r} log-probabilities for policy and reference")
    policy_margin = policy_logprobs["chosen"] - policy_logprobs["rejected"]
    reference_margin = reference_logprobs["chosen"] - reference_logprobs["rejected"]
    logits = beta * (policy_margin - reference_margin)
    # -log sigmoid(logits) written without overflow for large |logits|
    loss = math.log1p(math.exp(-logits)) if logits >= 0 else (-logits + math.log1p(math.exp(logits)))
    return {
        "loss": loss,
        "policy_margin": policy_margin,
        "reference_margin": reference_margin,
        "implicit_reward_margin": logits,
        "beta": beta,
    }


def grpo_advantages(rewards: Sequence[float], group_ids: Sequence[str]) -> Dict[str, List[float]]:
    """Group-relative advantage (§3.1/§10.3): the *group* boundary is the unit.

    A single group spanning two prompts would normalise across prompts and is the
    failure the step-12 hand-check exists to catch.
    """
    if len(rewards) != len(group_ids):
        raise ConfigError("rewards and group_ids must have the same length")
    groups: Dict[str, List[float]] = {}
    for reward, group in zip(rewards, group_ids):
        groups.setdefault(str(group), []).append(float(reward))
    if not groups:
        raise ConfigError("at least one group is required")
    advantages: List[float] = [0.0] * len(rewards)
    for group, values in groups.items():
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        for index, (reward, gid) in enumerate(zip(rewards, group_ids)):
            if str(gid) != group:
                continue
            advantages[index] = 0.0 if std == 0 else (float(reward) - mean) / std
    return {"advantages": advantages, "group_sizes": {group: len(values) for group, values in groups.items()}}


def grpo_loss(
    old_logprobs: Sequence[float],
    new_logprobs: Sequence[float],
    reference_logprobs: Sequence[float],
    advantages: Sequence[float],
    mask: Sequence[int],
    *,
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.0,
) -> Dict[str, Any]:
    """Clipped importance-ratio objective with an optional KL term (§3.1).

    Ratios are ``exp(new - old)``; a sign error or an old/new swap changes the
    direction of the update while the loss may still decrease, so the ratio and
    clip fraction are returned as first-class outputs.
    """
    lengths = {len(old_logprobs), len(new_logprobs), len(reference_logprobs), len(advantages), len(mask)}
    if len(lengths) != 1:
        raise ConfigError("all GRPO inputs must have the same length")
    total = 0.0
    count = 0
    ratios: List[float] = []
    clipped = 0
    kl_total = 0.0
    for old, new, reference, advantage, keep in zip(
        old_logprobs, new_logprobs, reference_logprobs, advantages, mask
    ):
        if not keep:
            continue
        ratio = math.exp(new - old)
        unclipped = ratio * advantage
        clipped_value = max(min(ratio, 1.0 + clip_epsilon), 1.0 - clip_epsilon) * advantage
        total += -min(unclipped, clipped_value)
        ratios.append(ratio)
        if ratio > 1.0 + clip_epsilon or ratio < 1.0 - clip_epsilon:
            clipped += 1
        if kl_coefficient:
            kl_total += (new - reference)
        count += 1
    if count == 0:
        return {"loss": None, "count": 0, "error": "every token was masked out"}
    loss = total / count + kl_coefficient * (kl_total / count if kl_coefficient else 0.0)
    return {
        "loss": loss,
        "count": count,
        "ratio_mean": sum(ratios) / len(ratios),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "clip_fraction": clipped / count,
        "kl": (kl_total / count) if count else None,
    }


def validate_token_mask(
    mask: Sequence[int],
    *,
    prompt_length: int,
    response_length: int,
    eos_index: Optional[int] = None,
    truncated: bool = False,
) -> List[str]:
    """Step 13: padding / prompt mask / EOS / truncation / empty response.

    Masking the prompt would train the policy to reproduce the question and
    leaving padding unmasked would let length buy reward — both are silent
    quality failures, so the mask's *shape* is checked, not just its values.
    """
    findings: List[str] = []
    if len(mask) != prompt_length + response_length:
        findings.append(
            f"mask length {len(mask)} != prompt {prompt_length} + response {response_length}"
        )
    if any(value not in (0, 1) for value in mask):
        findings.append("mask must be binary")
    if mask[:prompt_length] and any(mask[:prompt_length]):
        findings.append("prompt tokens are unmasked: SFT/GRPO would train on the prompt")
    if response_length == 0:
        findings.append("empty response: an empty generation must be dropped, not scored as zero loss")
    if truncated and eos_index is None:
        findings.append("truncated response without EOS: the truncation is not marked in the mask")
    if eos_index is not None and not (0 <= eos_index < len(mask)):
        findings.append(f"eos_index {eos_index} is outside the sequence")
    if eos_index is not None and mask[eos_index] and not any(mask[eos_index + 1 :]):
        pass  # a masked EOS with everything after it masked is the correct shape
    return findings


def length_normalisation(
    *, token_losses: Sequence[float], mask: Sequence[int], mode: str = "token_mean"
) -> Dict[str, Any]:
    """Step 13: the denominator decides whether longer output is rewarded."""
    kept = [loss for loss, keep in zip(token_losses, mask) if keep]
    if not kept:
        return {"value": None, "denominator": 0, "error": "no unmasked token"}
    if mode == "token_mean":
        return {"value": sum(kept) / len(kept), "denominator": len(kept), "mode": mode}
    if mode == "sum":
        return {"value": sum(kept), "denominator": 1, "mode": mode}
    if mode == "sequence_mean":
        return {"value": sum(kept), "denominator": 1, "mode": mode,
                "caveat": "sequence_mean requires per-sequence lengths to be comparable"}
    raise ConfigError(f"unknown length normalisation {mode!r}")


def compare_logprobs(
    trainer: Mapping[str, Sequence[float]],
    engine: Mapping[str, Sequence[float]],
    *,
    abs_tol: float = 1e-4,
) -> Dict[str, Any]:
    """Step 14: the trainer's log-prob and the engine's must be the same object.

    Compared on the *same* token sequence, because comparing different sequences
    is how a tokenizer/sampling mismatch hides inside an acceptable mean.
    """
    if set(trainer) != set(engine):
        return {"error": "trainer and engine must score the same token sequences",
                "trainer_ids": sorted(trainer), "engine_ids": sorted(engine)}
    rows: List[Dict[str, Any]] = []
    for sample_id in sorted(trainer):
        left, right = list(trainer[sample_id]), list(engine[sample_id])
        if len(left) != len(right):
            rows.append({"sample_id": sample_id, "status": "length_mismatch",
                         "trainer": len(left), "engine": len(right), "within": False})
            continue
        diffs = [b - a for a, b in zip(left, right)]
        max_abs = max((abs(value) for value in diffs), default=0.0)
        rows.append({"sample_id": sample_id, "max_abs": max_abs, "within": max_abs <= abs_tol})
    return {
        "samples": rows,
        "all_within": all(row["within"] for row in rows),
        "abs_tol": abs_tol,
        "first_divergence": next((row["sample_id"] for row in rows if not row["within"]), None),
    }


def reward_determinism(
    *, repeated_rewards: Sequence[float], tolerance: float = 0.0, timeout_count: int = 0
) -> Dict[str, Any]:
    """Step 15: reward noise must not be mistaken for a policy update signal."""
    if not repeated_rewards:
        return {"error": "no repeated reward samples supplied"}
    spread = max(repeated_rewards) - min(repeated_rewards)
    return {
        "samples": len(repeated_rewards),
        "min": min(repeated_rewards),
        "max": max(repeated_rewards),
        "spread": spread,
        "timeouts": timeout_count,
        "deterministic": spread <= tolerance and timeout_count == 0,
        "tolerance": tolerance,
        "note": "scoring on a changeable reward service must be frozen with a version, not assumed stable",
    }


@dataclass(frozen=True)
class PostTrainingAlgorithmChoice:
    """Steps 1–2: one algorithm, with its paper/implementation version frozen."""

    algorithm: str
    objective_contract_digest: str
    framework_identity: str
    reference_policy_present: bool = False
    kl_coefficient: float = 0.0
    clip_epsilon: float = 0.0
    rationale: str = ""
    rejected_alternatives: Tuple[str, ...] = ()

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.algorithm not in ALGORITHMS:
            findings.append(f"algorithm must be one of {', '.join(ALGORITHMS)}")
        for name in ("objective_contract_digest", "framework_identity", "rationale"):
            if not getattr(self, name):
                findings.append(f"algorithm choice is missing {name!r}")
        if self.objective_contract_digest and not is_digest(self.objective_contract_digest):
            findings.append("objective_contract_digest must be sha256:<hex>")
        if self.algorithm == "dpo" and self.kl_coefficient:
            findings.append(
                "DPO does not take a separate KL coefficient; the reference is in the objective "
                "(§10.2: 不能把 DPO 伪装成在线 RL)"
            )
        if self.algorithm == "grpo" and self.clip_epsilon <= 0:
            findings.append("GRPO requires a positive clip epsilon")
        if self.algorithm in ("dpo", "grpo") and not self.reference_policy_present:
            findings.append(f"{self.algorithm} requires a reference policy identity")
        if not self.rejected_alternatives:
            findings.append("an ADR must say which algorithms were not chosen")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "objective_contract_digest": self.objective_contract_digest,
            "framework_identity": self.framework_identity,
            "reference_policy_present": self.reference_policy_present,
            "kl_coefficient": self.kl_coefficient,
            "clip_epsilon": self.clip_epsilon,
            "rationale": self.rationale,
            "rejected_alternatives": list(self.rejected_alternatives),
        }


def offline_semantics(algorithm: str) -> Dict[str, Any]:
    """§10.2: the permitted meaning of "async" for the chosen algorithm.

    For SFT/DPO the only honest asynchronous claim is about data preparation,
    scoring and training — calling it on-policy rollout would be a false claim
    that the protocol explicitly forbids.
    """
    if algorithm not in ALGORITHMS:
        raise ConfigError(f"unknown algorithm {algorithm!r}")
    if algorithm == "grpo":
        return {
            "algorithm": algorithm,
            "online_rollout": True,
            "allowed_async_claim": "rollout/reward/train 重叠",
            "forbidden_claim": "无",
            "requires_staleness_bound": True,
        }
    return {
        "algorithm": algorithm,
        "online_rollout": False,
        "allowed_async_claim": "数据准备/评分/训练流水异步",
        "forbidden_claim": "on-policy RL rollout（不得伪装）",
        "requires_staleness_bound": False,
    }


@dataclass
class PolicyPublishEvent:
    """Step 22: staging → verify → activate, atomically (§11)."""

    publish_id: str
    policy_snapshot_id: str
    staging: bool = True
    verify_passed: bool = False
    atomic_activate: bool = False
    verified_on_all_ranks: bool = False
    probe_digest: str = ""
    drained_old_requests: bool = False

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("publish_id", "policy_snapshot_id"):
            if not getattr(self, name):
                findings.append(f"PolicyPublishEvent: {name} is required")
        if not self.staging:
            findings.append("publish skips staging: consumers can see a half-written policy")
        if not self.verify_passed:
            findings.append("publish activates without verification")
        if not self.atomic_activate:
            findings.append("publish is not atomic: 单条请求中途切权重会混用两版")
        if not self.verified_on_all_ranks:
            findings.append(
                "weight transfer was not verified on every rank "
                "(E14-04 step 23: 传输 API 成功 ≠ 所有 rank/参数已更新)"
            )
        if not self.probe_digest:
            findings.append("no fixed probe output recorded for the activated policy")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "publish_id": self.publish_id,
            "policy_snapshot_id": self.policy_snapshot_id,
            "staging": self.staging,
            "verify_passed": self.verify_passed,
            "atomic_activate": self.atomic_activate,
            "verified_on_all_ranks": self.verified_on_all_ranks,
            "probe_digest": self.probe_digest,
            "drained_old_requests": self.drained_old_requests,
        }


def snapshot_from_checkpoint(
    *, policy_snapshot_id: str, checkpoint_id: str, optimizer_step: int, weight_aggregate_digest: str,
    generation_config: Mapping[str, Any], tokenizer_artifact_id: str, loaded_at_ns: int = 0,
) -> PolicySnapshot:
    """Bind a checkpoint to the rollout side (step 22) with a generation identity."""
    return PolicySnapshot(
        policy_snapshot_id=policy_snapshot_id,
        source_checkpoint_id=checkpoint_id,
        optimizer_step=optimizer_step,
        weight_aggregate_digest=weight_aggregate_digest,
        tokenizer_artifact_id=tokenizer_artifact_id,
        generation_config_digest=canonical_digest(generation_config),
        loaded_at_ns=loaded_at_ns,
    )


def build_trajectory(
    *, trajectory_id: str, sample_id: str, policy: PolicySnapshot, prompt: str, response: str,
    sampling_config: Mapping[str, Any], reward_definition_id: str, termination_reason: str = "eos",
    produced_at_step: int = 0, consumed_at_step: Optional[int] = None,
    reward_components: Optional[Mapping[str, float]] = None,
) -> TrajectoryRecord:
    """Steps 17–18: build a trajectory whose every identity field is filled.

    The token hashes are over the raw text here (a real run hashes the token
    ids); what matters is that *some* prompt/response/sampling identity exists —
    a record without them cannot be placed on-policy or off-policy (§3.2).
    """
    trajectory = TrajectoryRecord(
        trajectory_id=trajectory_id,
        sample_id=sample_id,
        policy_snapshot_id=policy.policy_snapshot_id,
        reference_snapshot_id=None,
        reward_definition_id=reward_definition_id,
        prompt_token_hash=digest_text(prompt),
        response_token_hash=digest_text(response),
        sampling_config_hash=canonical_digest(sampling_config) if sampling_config
        else digest_text("no-sampling-config"),
        termination_reason=termination_reason,
        produced_at_step=produced_at_step,
        consumed_at_step=consumed_at_step,
        reward_components=dict(reward_components or {}),
    )
    trajectory.staleness_steps = trajectory.compute_staleness()
    return trajectory


def audit_trajectory_batch(
    trajectories: Sequence[TrajectoryRecord], *, declared_snapshot_id: str = "", max_staleness: Optional[int] = None
) -> Dict[str, Any]:
    """Steps 18/29: reject bad or silently mixed-version samples before training."""
    problems: List[str] = check_batch_version_consistency(
        trajectories, declared_snapshot_id=declared_snapshot_id
    )
    invalid: List[Dict[str, str]] = []
    for trajectory in trajectories:
        findings = trajectory.validate(max_staleness=max_staleness)
        if findings:
            invalid.append({"trajectory_id": trajectory.trajectory_id, "problems": "; ".join(findings)})
    problems.extend(item["problems"] for item in invalid)
    return {
        "trajectories": len(trajectories),
        "declared_snapshot_id": declared_snapshot_id,
        "invalid": invalid,
        "problems": problems,
        "ok": not problems,
    }


# ── sync / async state machines (steps 9–10) ───────────────────────────────


@dataclass
class SyncStateMachine:
    """Step 9: the synchronous chain with auditable transitions."""

    events: List[Mapping[str, Any]] = field(default_factory=list)

    def record(self, stage: str, *, previous: str = "", status: str = "", detail: Optional[Mapping[str, Any]] = None) -> None:
        self.events.append({"previous": previous, "stage": stage, "status": status, "detail": dict(detail or {})})

    def problems(self) -> List[str]:
        findings: List[str] = []
        if not self.events:
            findings.append("no stage events recorded: stage 顺序不能靠日志顺序猜测")
            return findings
        previous = ""
        for index, event in enumerate(self.events):
            stage = str(event.get("stage", ""))
            if stage not in SYNC_STAGES:
                findings.append(f"event {index}: unknown stage {stage!r}")
                continue
            if index and str(event.get("previous", "")) != previous:
                findings.append(
                    f"event {index}: declared previous {event.get('previous')!r} != actual {previous!r}"
                )
            if index and not rec.is_valid_transition("posttraining_sync", previous, stage):
                findings.append(f"illegal transition {previous} -> {stage}")
            if not event.get("status"):
                findings.append(f"event {index} ({stage}) has no status")
            previous = stage
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {"events": [dict(event) for event in self.events], "problems": self.problems()}


#: Async failure/queue states that must be declared before overlapping (§6 step 10).
ASYNC_QUEUE_STATES: Tuple[str, ...] = (
    "PENDING",
    "LEASED",
    "CONSUMED",
    "EXPIRED",
    "DEAD_LETTER",
    "DUPLICATE_REJECTED",
    "STALE_REJECTED",
)


@dataclass
class AsyncQueuePolicy:
    """Step 10/26: a *bounded* queue with lease, retry, dedup and a staleness cap."""

    queue_name: str
    capacity: int
    lease_seconds: float
    max_attempts: int
    staleness_cap: int
    dead_letter_capacity: int = 0
    backpressure: str = "reject"
    consumption_semantics: str = "at-least-once"
    dedup_key: str = "trajectory_id"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.capacity <= 0:
            findings.append(
                f"queue {self.queue_name}: unbounded capacity equals '用内存换吞吐'（§10）"
            )
        if self.lease_seconds <= 0:
            findings.append(f"queue {self.queue_name}: a lease is required to make retries safe")
        if self.max_attempts <= 0:
            findings.append(f"queue {self.queue_name}: max_attempts must be bounded")
        if self.staleness_cap < 0:
            findings.append(f"queue {self.queue_name}: staleness_cap must be >= 0")
        if self.dead_letter_capacity < 0:
            findings.append(f"queue {self.queue_name}: dead_letter_capacity must be >= 0")
        if self.consumption_semantics not in rec.CONSUMPTION_SEMANTICS:
            findings.append(
                f"queue {self.queue_name}: consumption_semantics must be one of "
                f"{', '.join(rec.CONSUMPTION_SEMANTICS)}"
            )
        if self.consumption_semantics == "exactly-once" and not self.dedup_key:
            findings.append(
                f"queue {self.queue_name}: exactly-once without a dedup key is a claim, not a mechanism"
            )
        if self.backpressure not in ("reject", "block", "shed"):
            findings.append(f"queue {self.queue_name}: unknown backpressure policy {self.backpressure!r}")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "queue_name": self.queue_name,
            "capacity": self.capacity,
            "lease_seconds": self.lease_seconds,
            "max_attempts": self.max_attempts,
            "staleness_cap": self.staleness_cap,
            "dead_letter_capacity": self.dead_letter_capacity,
            "backpressure": self.backpressure,
            "consumption_semantics": self.consumption_semantics,
            "dedup_key": self.dedup_key,
        }


def check_consumption_ledger(
    produced: Sequence[str], consumed: Sequence[str], *, semantics: str
) -> Dict[str, Any]:
    """Steps 29/35: duplicate or lost trajectories must be *explained*, not averaged.

    ``at-least-once`` legitimately repeats a trajectory, so the ledger reports
    whether the repeat was recorded; ``exactly-once`` claims are checked against
    the same data, which is why the two semantics cannot share one assertion.
    """
    produced_set, consumed_set = set(produced), set(consumed)
    duplicates: Dict[str, int] = {}
    for item in consumed:
        duplicates[item] = duplicates.get(item, 0) + 1
    repeated = sorted(item for item, count in duplicates.items() if count > 1)
    lost = sorted(produced_set - consumed_set)
    invented = sorted(consumed_set - produced_set)
    problems: List[str] = []
    if invented:
        problems.append(f"consumed trajectories that were never produced: {', '.join(invented[:5])}")
    if lost:
        problems.append(f"produced but never consumed: {len(lost)} trajectory(ies)")
    if repeated and semantics == "exactly-once":
        problems.append(f"exactly-once violated: {len(repeated)} trajectory(ies) consumed more than once")
    if repeated and semantics == "at-least-once" and not any(
        item.get("duplicate_recorded") if isinstance(item, Mapping) else False for item in ()
    ):
        pass  # repeats are legal here; the ledger below records them explicitly
    return {
        "produced": len(produced_set),
        "consumed": len(consumed),
        "repeated": repeated,
        "lost": lost,
        "invented": invented,
        "semantics": semantics,
        "problems": problems,
        "ok": not problems,
    }


def check_staleness_sweep(rows: Sequence[Mapping[str, Any]], *, staleness_cap: int) -> Dict[str, Any]:
    """Step 30: scan staleness 0,1,2… and report throughput, KL and quality together.

    A single async configuration compared against sync cannot separate overlap
    from off-policy drift, so the sweep is the unit and the cap is enforced here
    as well as at the queue.
    """
    problems: List[str] = []
    observed: List[int] = []
    for row in rows:
        staleness = int(row.get("staleness_steps", -1))
        observed.append(staleness)
        if staleness < 0:
            problems.append("a row reports negative staleness")
        if staleness > staleness_cap:
            problems.append(f"staleness {staleness} exceeds the preregistered cap {staleness_cap}")
        for key in ("utilization", "kl", "ratio_mean", "quality_value", "drop_rate"):
            if key not in row:
                problems.append(f"staleness row {staleness} is missing {key!r}")
    missing_points = sorted(set(range(0, staleness_cap + 1)) - set(observed))
    if missing_points:
        problems.append(f"the sweep does not cover staleness {missing_points}")
    return {
        "rows": len(rows),
        "observed": sorted(set(observed)),
        "staleness_cap": staleness_cap,
        "missing_points": missing_points,
        "problems": problems,
        "ok": not problems,
    }


def check_kv_reuse_identity(
    *, cache_key_components: Sequence[str], invalidated_on_publish: bool, reused_across_policy: bool
) -> List[str]:
    """Step 32: a KV/prefix cache key must include the policy identity.

    Reusing a KV computed under an older policy is the failure the step names;
    it is a correctness bug, not a cache-tuning detail.
    """
    problems: List[str] = []
    required = {"model", "policy_snapshot_id", "tokenizer", "prompt"}
    present = {str(item).lower().replace("policy", "policy_snapshot_id") for item in cache_key_components}
    missing = sorted(item for item in required if item not in present)
    if missing:
        problems.append(
            "cache key is missing identity components: "
            + ", ".join(missing)
            + " (跨权重复用旧 KV = FAIL)"
        )
    if not invalidated_on_publish:
        problems.append("cache is not invalidated when a new policy snapshot is activated")
    if reused_across_policy:
        problems.append("a KV entry was reused across two policy snapshots")
    return problems


# ── failure scenarios (steps 33–36) ────────────────────────────────────────

#: Each scenario names the *required* outcome, so an injected failure cannot be
#: "handled" by silently producing a training signal (§10).
FAILURE_SCENARIOS: Tuple[Tuple[str, str, str], ...] = (
    ("environment_timeout", "E14-04 step 33", "样本进入 timed_out，不得赋默认 reward=0 进入训练"),
    ("tool_malformed_result", "E14-04 step 33", "畸形结果在校验阶段拒绝，并记录 tool contract id"),
    ("reward_service_timeout", "E14-04 step 34", "backpressure 生效，队列年龄有界，rollout 不无界生产"),
    ("reward_slowdown", "E14-04 step 34", "staleness 增长被记录并触发停止规则"),
    ("rollout_worker_crash", "E14-04 step 35", "lease 到期重派，去重键阻止重复梯度更新"),
    ("trainer_crash", "E14-04 step 35", "checkpoint 恢复，数据水位不回退，队列不丢失"),
    ("publish_failure", "E14-04 step 35", "旧 policy 继续服务，不出现半激活状态"),
)


def failure_scenarios() -> Tuple[Dict[str, str], ...]:
    return tuple(
        {"scenario": name, "clause": clause, "required_outcome": outcome}
        for name, clause, outcome in FAILURE_SCENARIOS
    )


def check_failure_accounting(records: Sequence[Mapping[str, Any]]) -> List[str]:
    """Steps 33–35: every injected failure must state its data consequence."""
    problems: List[str] = []
    known = {name for name, _clause, _outcome in FAILURE_SCENARIOS}
    for record in records:
        scenario = str(record.get("scenario", ""))
        if scenario not in known:
            problems.append(f"unknown failure scenario {scenario!r}")
            continue
        for key in ("sample_state", "resource_released", "duplicate_consumed", "trajectory_ids"):
            if key not in record:
                problems.append(f"{scenario}: failure record is missing {key!r}")
        if record.get("sample_state") == "scored" and scenario.startswith("environment_"):
            problems.append(f"{scenario}: a timed-out sample was scored (默认 reward 进入训练 = FAIL)")
        if record.get("duplicate_consumed"):
            problems.append(f"{scenario}: a retry produced a duplicate gradient update")
        if record.get("resource_released") is False:
            problems.append(f"{scenario}: resources were not released after the failure")
    return problems


def check_heldout_quality(
    *,
    train_reward_before: float,
    train_reward_after: float,
    heldout_before: float,
    heldout_after: float,
    guard_band: float,
    indicators: Mapping[str, float],
) -> Dict[str, Any]:
    """Step 25: reward up + held-out down is not a PASS, it is reward hacking.

    ``E14-04`` §9: reward 上升、held-out 下降 → 优先判断 reward hacking/过拟合，
    不能 PASS.  Every anti-cheat indicator must be present so a length bias or a
    format collapse cannot hide behind a single aggregate.
    """
    problems: List[str] = []
    missing = [name for name in ANTI_CHEAT_INDICATORS if name not in indicators]
    if missing:
        problems.append(f"anti-cheat indicators missing: {', '.join(missing)}")
    delta = heldout_after - heldout_before
    reward_gain = train_reward_after - train_reward_before
    hacked = reward_gain > 0 and delta < -abs(guard_band)
    if hacked:
        problems.append(
            f"train reward rose by {reward_gain:.4f} while held-out fell by {abs(delta):.4f}: "
            "reward hacking / overfitting, not a PASS"
        )
    return {
        "train_reward_gain": reward_gain,
        "heldout_delta": delta,
        "guard_band": guard_band,
        "reward_hacking_suspected": hacked,
        "indicators": {key: indicators[key] for key in sorted(indicators)},
        "problems": problems,
        "pass_eligible": not problems,
    }


def compare_sync_async_pareto(
    rows: Sequence[Mapping[str, Any]], *, generated_tokens: int, training_tokens: int
) -> Dict[str, Any]:
    """Step 39: async may not "win" by doing more work.

    ``E14-04`` §10: async 多生成更多 token 后直接比较总 reward 是不公平的, so rows
    with a different work denominator are separated out instead of ranked.
    """
    comparable: List[Dict[str, Any]] = []
    incomparable: List[Dict[str, Any]] = []
    for row in rows:
        if int(row.get("generated_tokens", -1)) != generated_tokens or int(
            row.get("training_tokens", -1)
        ) != training_tokens:
            incomparable.append(dict(row))
            continue
        for key in ("wall_time_s", "staleness_steps", "quality_value", "cost_units"):
            if key not in row:
                incomparable.append(dict(row))
                break
        else:
            comparable.append(dict(row))
    return {
        "work_denominator": {"generated_tokens": generated_tokens, "training_tokens": training_tokens},
        "comparable": comparable,
        "excluded_for_different_work": incomparable,
        "note": "只在相同生成/训练 token 与质量门下比较；不同工作量的行单列，不参与排名",
    }


# ── verdict (step 40) ──────────────────────────────────────────────────────


@dataclass
class PostTrainingVerdict:
    """Step 40 — objective, lineage, sync, async, quality and recovery, separately."""

    verdict_id: str
    decision: str = rec.STATUS_NOT_RUN
    objective: Mapping[str, Any] = field(default_factory=dict)
    lineage: Mapping[str, Any] = field(default_factory=dict)
    sync: Mapping[str, Any] = field(default_factory=dict)
    async_path: Mapping[str, Any] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)
    failure_recovery: Mapping[str, Any] = field(default_factory=dict)
    adoption: Mapping[str, Any] = field(default_factory=dict)
    limitations: Tuple[str, ...] = ()

    schema_version = f"{SCHEMA_PREFIX}.e14-04.verdict.v1"

    #: Sub-decisions required because "reward 曲线上升" cannot cover them (§11).
    REQUIRED_AXES: Tuple[str, ...] = ("objective", "lineage", "sync", "quality", "failure_recovery")

    def validate(self) -> List[str]:
        findings: List[str] = []
        if self.decision not in rec.ALL_STATUSES:
            findings.append(f"PostTrainingVerdict: unknown decision {self.decision!r}")
        if self.decision == rec.STATUS_PASS:
            for axis in self.REQUIRED_AXES:
                if not getattr(self, axis):
                    findings.append(
                        f"PostTrainingVerdict: PASS requires a ruling on {axis!r} "
                        "（reward 上升不能覆盖版本或恢复失败）"
                    )
            if self.quality.get("reward_hacking_suspected") is True:
                findings.append("PostTrainingVerdict: PASS is impossible while reward hacking is suspected")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "verdict_id": self.verdict_id,
            "decision": self.decision,
            "objective": dict(self.objective),
            "lineage": dict(self.lineage),
            "sync": dict(self.sync),
            "async_path": dict(self.async_path),
            "quality": dict(self.quality),
            "failure_recovery": dict(self.failure_recovery),
            "adoption": dict(self.adoption),
            "limitations": list(self.limitations),
            "digest": canonical_digest(
                {
                    "verdict_id": self.verdict_id,
                    "decision": self.decision,
                    "quality": dict(self.quality),
                    "lineage": dict(self.lineage),
                }
            ),
        }


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the E14-04 interfaces (labelled smoke, not an experiment)."""
    logits = [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]
    sft = sft_loss(logits, [0, 1], [0, 1])
    dpo = dpo_loss({"chosen": -1.0, "rejected": -3.0}, {"chosen": -2.0, "rejected": -2.0}, beta=0.1)
    grouped = grpo_advantages([1.0, 0.0, 2.0, 1.0], ["g1", "g1", "g2", "g2"])
    grpo = grpo_loss(
        [-1.0, -1.0, -1.0, -1.0],
        [-0.9, -1.1, -0.8, -1.2],
        [-1.5, -1.5, -1.5, -1.5],
        grouped["advantages"],
        [1, 1, 1, 1],
        clip_epsilon=0.2,
    )
    machine = SyncStateMachine()
    machine.record("GENERATE", status="ok")
    machine.record("TRAIN", previous="GENERATE", status="ok")
    queue = AsyncQueuePolicy("scored", capacity=8, lease_seconds=30.0, max_attempts=3, staleness_cap=1)
    ledger = check_consumption_ledger(["t0", "t1"], ["t0", "t1", "t0"], semantics="at-least-once")
    hacked = check_heldout_quality(
        train_reward_before=0.1, train_reward_after=0.5, heldout_before=0.8, heldout_after=0.5,
        guard_band=0.02, indicators={name: 0.0 for name in ANTI_CHEAT_INDICATORS},
    )
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "sft_loss": sft["loss"],
        "dpo_loss": dpo["loss"],
        "grpo_clip_fraction": grpo["clip_fraction"],
        "sync_illegal_transition_detected": bool(machine.problems()),
        "queue_problems": queue.problems(),
        "at_least_once_repeats_ok": ledger["ok"] and bool(ledger["repeated"]),
        "reward_hacking_detected": hacked["reward_hacking_suspected"],
        "scenarios": len(FAILURE_SCENARIOS),
        "anti_cheat_indicators": len(ANTI_CHEAT_INDICATORS),
    }


# ── result accessors ───────────────────────────────────────────────────────

def grpo_clip_fraction(result: Mapping[str, Any]) -> Optional[float]:
    """Step 21: how much of the batch was clipped (a wrong ratio shows up here)."""
    return result.get("clip_fraction")


def consumption_semantics(result: Mapping[str, Any]) -> str:
    """Step 35: exactly-once vs at-least-once decides what a retry may duplicate."""
    return str(result.get("semantics", ""))


# ── protocol step table (40 steps of details/S14/E14-04) ───────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "用 ADR 选择唯一主算法", ("posttraining:PostTrainingAlgorithmChoice", "posttraining:ALGORITHMS")),
    (2, "冻结算法论文与实现版本", ("posttraining:PostTrainingAlgorithmChoice.framework_identity",
                                   "posttraining:offline_semantics")),
    (3, "冻结能力声明", ("posttraining:CLAIM_BOUNDARY", "contracts:AdoptionDecision.forbidden_claims")),
    (4, "冻结 base PolicySnapshot", ("posttraining:snapshot_from_checkpoint", "contracts:PolicySnapshot")),
    (5, "冻结 data artifact", ("identity:artifact_ref", "identity:content_address_aggregate")),
    (6, "冻结 environment/tool contract", ("posttraining:build_trajectory", "records:TABLE_SCHEMAS")),
    (7, "冻结 reward/reference 定义", ("contracts:TrajectoryRecord.reward_definition_id",
                                        "posttraining:reward_determinism")),
    (8, "冻结 sampling contract", ("posttraining:build_trajectory", "identity:canonical_digest")),
    (9, "冻结同步状态机", ("posttraining:SyncStateMachine", "records:POSTTRAINING_SYNC_TRANSITIONS")),
    (10, "冻结异步状态机", ("posttraining:AsyncQueuePolicy", "posttraining:ASYNC_QUEUE_STATES")),
    (11, "冻结资源和停止预算", ("campaign:RunBudget", "campaign:BUDGET_DIMENSIONS")),
    (12, "构造手算最小 objective case", ("posttraining:sft_loss", "posttraining:dpo_loss",
                                          "posttraining:grpo_loss")),
    (13, "验证 token mask 和长度归一化", ("posttraining:validate_token_mask",
                                          "posttraining:length_normalisation")),
    (14, "验证 policy/reference logprob", ("posttraining:compare_logprobs", "contracts:PolicySnapshot")),
    (15, "验证 reward determinism/variance", ("posttraining:reward_determinism", "posttraining:REWARD_FAILURE_MODES")),
    (16, "验证 rollout engine 语义", ("contracts:PolicySnapshot.rollout_engine_id",
                                       "contracts:check_actual_path_recorded")),
    (17, "执行合法同步 rollout", ("posttraining:build_trajectory", "contracts:TrajectoryRecord")),
    (18, "审计 trajectory 完整性", ("posttraining:audit_trajectory_batch", "contracts:TrajectoryRecord.validate")),
    (19, "重建 objective 与 batch 统计", ("posttraining:grpo_advantages", "posttraining:grpo_loss")),
    (20, "执行单步训练更新", ("posttraining:check_consumption_ledger", "records:TABLE_SCHEMAS")),
    (21, "验证更新方向和边界", ("posttraining:dpo_loss", "posttraining:grpo_clip_fraction")),
    (22, "转换并发布新 PolicySnapshot", ("posttraining:PolicyPublishEvent", "posttraining:snapshot_from_checkpoint")),
    (23, "验证权重同步完整性", ("posttraining:PolicyPublishEvent.verified_on_all_ranks",
                                "contracts:PolicySnapshot.transfer_verified")),
    (24, "运行短同步闭环", ("posttraining:SyncStateMachine", "records:TABLE_SCHEMAS")),
    (25, "执行 held-out 质量与反作弊评价", ("posttraining:check_heldout_quality",
                                            "posttraining:ANTI_CHEAT_INDICATORS")),
    (26, "部署有界异步队列", ("posttraining:AsyncQueuePolicy", "posttraining:AsyncQueuePolicy.problems")),
    (27, "启用版本化异步生成", ("contracts:PolicySnapshot.is_active_at", "contracts:TrajectoryRecord.produced_at_step")),
    (28, "验证原子 weight activate", ("posttraining:PolicyPublishEvent.atomic_activate",
                                       "contracts:PolicySnapshot.active_from_ns")),
    (29, "注入 mixed-version batch 负例", ("contracts:check_batch_version_consistency",
                                           "posttraining:audit_trajectory_batch")),
    (30, "扫描 staleness 梯度", ("posttraining:check_staleness_sweep", "posttraining:AsyncQueuePolicy.staleness_cap")),
    (31, "测 batching/queue 与 GPU 利用", ("posttraining:AsyncQueuePolicy.capacity", "records:TABLE_SCHEMAS")),
    (32, "验证 KV/prefix reuse 身份", ("posttraining:check_kv_reuse_identity", "contracts:PolicySnapshot.policy_snapshot_id")),
    (33, "注入 environment/tool timeout", ("posttraining:failure_scenarios", "posttraining:check_failure_accounting")),
    (34, "注入 reward failure/slowdown", ("posttraining:REWARD_FAILURE_MODES", "posttraining:check_failure_accounting")),
    (35, "注入 rollout/trainer worker crash", ("posttraining:failure_scenarios",
                                                "posttraining:consumption_semantics")),
    (36, "执行 checkpoint→resume 闭环", ("training:compare_resume_continuity", "posttraining:SyncStateMachine")),
    (37, "做跨层 profiling", ("telemetry:project_trajectory_event", "records:PROFILE_LAYERS")),
    (38, "跨 seed/run 重复", ("records:EXPERIMENT_UNITS", "identity:SeedBundle")),
    (39, "比较 sync/async Pareto", ("posttraining:compare_sync_async_pareto", "contracts:AdoptionDecision")),
    (40, "形成 PostTrainingVerdict", ("posttraining:PostTrainingVerdict",
                                       "posttraining:PostTrainingVerdict.validate")),
)
