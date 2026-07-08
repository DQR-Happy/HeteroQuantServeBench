"""E14-F1 — speculative/MTP acceptance, extra computation and service scheduling.

Conditional P0: it is the main branch only when ``E14-05`` selects F1, otherwise
``N/A_BY_ADR``.  The core judgement is that a candidate is compared against
target-only under the same target, generation semantics, quality, workload,
device and service policy, with **every** cost counted: proposed/accepted/
resampled tokens, draft/target/verify/KV/scheduler cost, and the kernel/bytes/
FLOPs of the work that was thrown away.

Theoretical basis (§3):

* classic speculative sampling preserves the target distribution through
  correct accept/reject plus **residual sampling** — that is what "exact" means,
  and it is checked here on small probability vectors rather than assumed;
* greedy mode reduces to a longest-prefix match with an explicit first-reject,
  bonus token and KV rollback;
* the cost is a *cycle* cost, and ``acceptance ↑`` does **not** imply
  ``T_cycle / E[A] ↓``.

Interfaces provided:

* :class:`TargetProposerBinding` — artifact/tokenizer/vocabulary compatibility
  (steps 3–5);
* :class:`GenerationContract` — greedy vs sampling, stop rules, counters
  (steps 6–7);
* :func:`greedy_prefix_match`, :func:`accept_reject_test`,
  :func:`residual_distribution`, :func:`expected_accepted_tokens` — the
  hand-checkable oracles (steps 14–16);
* :class:`SpeculationCycle` — the §7 record, with KV actions and target calls
  (steps 17–19);
* :func:`acceptance_by_position`, :func:`stratify_by_domain` (step 23);
* :func:`cost_model`, :func:`reconstruct_cycle_cost` (steps 9, 37);
* :func:`wasted_work`, :func:`low_acceptance_guard` (steps 31–33);
* :func:`proposer_failure_fallback`, :func:`check_cancel_release` (steps 34–35);
* :class:`F1AdoptionDecision` (step 40).

Nothing here runs an engine: the oracles are arithmetic over supplied logits and
token ids, so an engine's aggregate acceptance rate can be checked against an
independent computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

from hqsb.experimental import records as rec
from hqsb.experimental.contracts import AdoptionDecision
from hqsb.experimental.identity import SCHEMA_PREFIX, is_digest

EXPERIMENT_ID = "E14-F1"
TITLE = "Speculative/MTP Acceptance、额外计算与服务调度"
LEVEL = "条件 P0"

CLAIM_BOUNDARY = (
    "只在 F1 为唯一主分支时成立；质量与 actual-path 门通过前，acceptance 数字不构成加速声明。"
    "无收益但门禁与归因完整时可形成 PASS_NEGATIVE/REJECT_NO_BENEFIT（E14-F1 §12）。"
)

#: Proposer mechanisms; ``E14-05`` picks one as primary (step 2).
PROPOSER_KINDS: Tuple[str, ...] = ("classic_draft", "mtp_head", "medusa", "tree", "ngram")

#: Generation modes; greedy and sampling may never share a results table (§10).
GENERATION_MODES: Tuple[str, ...] = ("greedy", "sampling")

#: Reasons a proposal word ends (§6 step 18).
STOP_REASONS: Tuple[str, ...] = ("accepted_all", "rejected", "eos", "max_tokens", "cancelled", "error")

#: Strata that must be reported separately (step 23).
ACCEPTANCE_STRATA: Tuple[str, ...] = ("domain", "context_length", "output_length", "entropy_bin")

#: Cost components of one speculation cycle (§3.2, step 30).
CYCLE_COST_COMPONENTS: Tuple[str, ...] = (
    "draft_time_ms",
    "target_verify_time_ms",
    "rollback_time_ms",
    "scheduler_time_ms",
    "kv_copy_time_ms",
)

#: KV actions a cycle may perform (step 19).
KV_ACTIONS: Tuple[str, ...] = ("write", "commit", "discard", "copy", "none")


@dataclass(frozen=True)
class TargetProposerBinding:
    """Steps 3–5: both artefacts plus the tokenizer/vocabulary contract."""

    target_artifact_id: str
    proposer_artifact_id: str
    proposer_kind: str
    target_tokenizer_id: str
    proposer_tokenizer_id: str
    target_vocab_size: int
    proposer_vocab_size: int
    token_id_mapping: str = "identity"
    proposer_precision: str = ""
    proposer_device: str = ""
    proposer_weight_digest: str = ""

    def problems(self) -> List[str]:
        findings: List[str] = []
        for name in ("target_artifact_id", "proposer_artifact_id", "target_tokenizer_id",
                     "proposer_tokenizer_id"):
            if not getattr(self, name):
                findings.append(f"TargetProposerBinding: {name} is required")
        if self.proposer_kind not in PROPOSER_KINDS:
            findings.append(
                f"TargetProposerBinding: proposer_kind {self.proposer_kind!r} must be one of "
                f"{', '.join(PROPOSER_KINDS)}"
            )
        if self.target_tokenizer_id and self.proposer_tokenizer_id:
            if self.target_tokenizer_id != self.proposer_tokenizer_id and self.token_id_mapping == "identity":
                findings.append(
                    "TargetProposerBinding: different tokenizers with an identity mapping "
                    "(文本相同但 token 序列不同 —— 必须有显式映射或拒绝)"
                )
        if self.target_vocab_size and self.proposer_vocab_size:
            if self.proposer_vocab_size > self.target_vocab_size:
                findings.append(
                    f"TargetProposerBinding: proposer vocab {self.proposer_vocab_size} exceeds target "
                    f"{self.target_vocab_size}; draft tokens outside the target vocabulary are unmappable"
                )
        if self.proposer_weight_digest and not is_digest(self.proposer_weight_digest):
            findings.append("TargetProposerBinding: proposer_weight_digest must be sha256:<hex>")
        if not self.proposer_precision:
            findings.append("TargetProposerBinding: proposer precision is part of its identity")
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target_artifact_id": self.target_artifact_id,
            "proposer_artifact_id": self.proposer_artifact_id,
            "proposer_kind": self.proposer_kind,
            "target_tokenizer_id": self.target_tokenizer_id,
            "proposer_tokenizer_id": self.proposer_tokenizer_id,
            "target_vocab_size": self.target_vocab_size,
            "proposer_vocab_size": self.proposer_vocab_size,
            "token_id_mapping": self.token_id_mapping,
            "proposer_precision": self.proposer_precision,
            "proposer_device": self.proposer_device,
            "proposer_weight_digest": self.proposer_weight_digest,
        }


@dataclass(frozen=True)
class GenerationContract:
    """Step 6: the generation semantics both paths must share."""

    mode: str
    max_tokens: int
    eos_token_id: Optional[int] = None
    stop_sequences: Tuple[str, ...] = ()
    seed: Optional[int] = None
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    accepted_token_counting: str = "committed_tokens"

    def problems(self) -> List[str]:
        findings: List[str] = []
        if self.mode not in GENERATION_MODES:
            findings.append(f"GenerationContract: mode {self.mode!r} must be greedy or sampling")
        if self.max_tokens <= 0:
            findings.append("GenerationContract: max_tokens must be positive")
        if self.mode == "greedy":
            if self.temperature not in (0.0, 1.0):
                findings.append(
                    "GenerationContract: greedy with temperature != 0/1 blurs the greedy/sampling boundary"
                )
            if self.seed is not None:
                findings.append("GenerationContract: greedy mode does not need a seed; recording one hides the mode")
        else:
            if self.seed is None:
                findings.append("GenerationContract: sampling mode requires a seed to be reproducible")
            if self.temperature <= 0:
                findings.append("GenerationContract: sampling mode requires temperature > 0")
        if self.eos_token_id is None and not self.stop_sequences:
            findings.append(
                "GenerationContract: no EOS or stop rule means the two paths can stop at different points "
                "(早停规则不同会让延迟比较失真)"
            )
        if self.accepted_token_counting not in ("committed_tokens", "accepted_only", "includes_bonus"):
            findings.append(
                f"GenerationContract: unknown accepted_token_counting {self.accepted_token_counting!r}"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "max_tokens": self.max_tokens,
            "eos_token_id": self.eos_token_id,
            "stop_sequences": list(self.stop_sequences),
            "seed": self.seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "accepted_token_counting": self.accepted_token_counting,
        }


# ── hand-checkable oracles (steps 14–16) ───────────────────────────────────


def greedy_prefix_match(
    proposed: Sequence[int], target_tokens: Sequence[int]
) -> Dict[str, Any]:
    """Greedy acceptance is a longest-prefix match plus one bonus token (§3.1).

    ``target_tokens`` are the target's own greedy tokens for the positions the
    proposals occupy, so ``accepted`` is the first index where they differ and
    ``bonus`` is the target token at that index — the token a correct
    implementation must emit instead of the rejected proposal.
    """
    if not proposed:
        return {"accepted": 0, "reject_position": 0, "bonus": None, "committed": []}
    accepted = 0
    for index, token in enumerate(proposed):
        if index >= len(target_tokens):
            break
        if int(token) == int(target_tokens[index]):
            accepted += 1
        else:
            break
    reject_position = accepted if accepted < len(proposed) else None
    bonus = None
    if reject_position is not None and reject_position < len(target_tokens):
        bonus = int(target_tokens[reject_position])
    committed = [int(token) for token in proposed[:accepted]]
    if bonus is not None:
        committed.append(bonus)
    return {
        "accepted": accepted,
        "reject_position": reject_position,
        "bonus": bonus,
        "committed": committed,
    }


def residual_distribution(target_probs: Sequence[float], draft_probs: Sequence[float]) -> List[float]:
    """``normalise(max(0, p - q))`` — the exact-sampling residual (§3.1).

    Using the target distribution itself (or the draft's) instead of the residual
    still "works" but changes the output distribution, which is why the oracle
    returns the vector rather than a single accepted token.
    """
    if len(target_probs) != len(draft_probs):
        raise ConfigError("target_probs and draft_probs must have the same length")
    if not target_probs:
        raise ConfigError("probability vectors must be non-empty")
    positive = [max(0.0, float(p) - float(q)) for p, q in zip(target_probs, draft_probs)]
    total = sum(positive)
    if total <= 0.0:
        # Degenerate case: p == q everywhere, so a rejection cannot occur.
        return [0.0 for _ in positive]
    return [value / total for value in positive]


def accept_reject_test(
    *, target_probs: Sequence[float], draft_probs: Sequence[float], token: int, uniform: float
) -> Dict[str, Any]:
    """One speculative accept/reject decision (``E14-F1`` §3.1).

    Accept iff ``u <= min(1, p(token)/q(token))``; otherwise the token is
    resampled from the residual.  ``uniform`` is supplied rather than drawn, so a
    test can pin an exact case (step 14: 用小词表 logits 验证实现).
    """
    if not 0.0 <= uniform < 1.0:
        raise ConfigError("uniform must be in [0, 1)")
    if not 0 <= token < len(target_probs):
        raise ConfigError(f"token {token} is outside the distribution")
    p = float(target_probs[token])
    q = float(draft_probs[token])
    if q <= 0.0:
        accepted = p <= 0.0
        threshold = 1.0 if p <= 0.0 else 0.0
    else:
        threshold = min(1.0, p / q)
        accepted = uniform <= threshold
    return {
        "token": token,
        "p": p,
        "q": q,
        "threshold": threshold,
        "uniform": uniform,
        "accepted": accepted,
        "path": "accept" if accepted else "reject",
    }


def expected_accepted_tokens(acceptance_by_position: Sequence[float], *, gamma: Optional[int] = None) -> float:
    """``E[A]`` for one cycle: the sum of per-position acceptance probabilities.

    For exact sampling the joint acceptance of position *i* is the product of the
    per-position probabilities, so the expectation is their running product; a
    naive ``sum`` overstates the benefit and is the reason acceptance is reported
    by position rather than as one aggregate (§6 step 17).
    """
    if not acceptance_by_position:
        return 0.0
    if gamma is not None and gamma != len(acceptance_by_position):
        raise ConfigError("gamma must match the number of positions")
    product = 1.0
    total = 0.0
    for probability in acceptance_by_position:
        if not 0.0 <= probability <= 1.0:
            raise ConfigError(f"acceptance probability {probability} is outside [0, 1]")
        product *= probability
        total += product
    return total


def acceptance_by_position(cycles: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Step 17: per-position acceptance, not one aggregate percentage."""
    proposals: Dict[int, int] = {}
    accepts: Dict[int, int] = {}
    for cycle in cycles:
        proposed = cycle.get("proposed_token_ids") or []
        accepted = int(cycle.get("accepted_count", 0))
        for position in range(len(proposed)):
            proposals[position] = proposals.get(position, 0) + 1
            if position < accepted:
                accepts[position] = accepts.get(position, 0) + 1
    return [
        {
            "position": position,
            "proposed": proposals[position],
            "accepted": accepts.get(position, 0),
            "acceptance": accepts.get(position, 0) / proposals[position] if proposals[position] else 0.0,
        }
        for position in sorted(proposals)
    ]


def stratify_by_domain(cycles: Sequence[Mapping[str, Any]], *, key: str = "domain") -> Dict[str, Any]:
    """Step 23: a global mean hides workloads that cannot benefit (§9)."""
    if key not in ACCEPTANCE_STRATA:
        raise ConfigError(f"stratum key {key!r} must be one of {', '.join(ACCEPTANCE_STRATA)}")
    groups: Dict[str, Dict[str, float]] = {}
    for cycle in cycles:
        name = str(cycle.get(key, "<unstratified>"))
        bucket = groups.setdefault(name, {"cycles": 0.0, "proposed": 0.0, "accepted": 0.0})
        bucket["cycles"] += 1
        bucket["proposed"] += len(cycle.get("proposed_token_ids") or [])
        bucket["accepted"] += int(cycle.get("accepted_count", 0))
    return {
        key: name,
        "strata": {
            name: {
                "cycles": int(values["cycles"]),
                "proposed": int(values["proposed"]),
                "accepted": int(values["accepted"]),
                "acceptance": values["accepted"] / values["proposed"] if values["proposed"] else 0.0,
            }
            for name, values in sorted(groups.items())
        },
        "note": "某 domain 受益必须限定于该 domain/entropy，不写普遍加速（E14-F1 §9）",
    }


@dataclass
class SpeculationCycle:
    """``E14-F1`` §7 ``SpeculationCycle`` (schema ``…e14-f1.cycle.v1``)."""

    request_id: str
    cycle_index: int
    target_artifact_id: str
    proposer_artifact_id: str
    proposed_token_ids: Tuple[int, ...] = ()
    accepted_count: int = 0
    reject_position: Optional[int] = None
    committed_token_ids: Tuple[int, ...] = ()
    draft_time_ms: Optional[float] = None
    verify_time_ms: Optional[float] = None
    rollback_time_ms: Optional[float] = None
    scheduler_time_ms: Optional[float] = None
    kv_copy_time_ms: Optional[float] = None
    target_calls: int = 0
    kv_written_bytes: int = 0
    kv_discarded_bytes: int = 0
    actual_backend: str = ""
    state: str = "PROPOSED"
    status: str = rec.STATUS_NOT_RUN

    schema_version = f"{SCHEMA_PREFIX}.e14-f1.cycle.v1"

    def expected_committed(self) -> List[int]:
        """What a correct greedy implementation must commit for this cycle."""
        return [int(token) for token in self.proposed_token_ids[: self.accepted_count]]

    def validate(self) -> List[str]:
        findings: List[str] = []
        for name in ("request_id", "target_artifact_id", "proposer_artifact_id"):
            if not getattr(self, name):
                findings.append(f"SpeculationCycle: {name} is required")
        if self.cycle_index < 0:
            findings.append("SpeculationCycle: cycle_index must be >= 0")
        if self.state not in rec.SPECULATION_CYCLE_STATES:
            findings.append(
                f"SpeculationCycle: state {self.state!r} must be one of "
                f"{', '.join(rec.SPECULATION_CYCLE_STATES)}"
            )
        if self.accepted_count > len(self.proposed_token_ids):
            findings.append("SpeculationCycle: accepted_count exceeds the number of proposals")
        if self.reject_position is not None:
            if self.reject_position != self.accepted_count:
                findings.append(
                    f"SpeculationCycle: reject_position {self.reject_position} != accepted_count "
                    f"{self.accepted_count} (首个拒绝位置必须紧接已接受前缀)"
                )
        elif self.accepted_count != len(self.proposed_token_ids):
            findings.append(
                "SpeculationCycle: no reject position but not every proposal was accepted"
            )
        if self.state == "COMMITTED" and not self.committed_token_ids:
            findings.append("SpeculationCycle: a committed cycle must record its committed tokens")
        if self.accepted_count and self.target_calls <= 0:
            findings.append("SpeculationCycle: accepted tokens require at least one target call")
        if self.kv_discarded_bytes and not self.rollback_time_ms:
            findings.append(
                "SpeculationCycle: KV was discarded but no rollback time was recorded "
                "(rollback 成本必须计入，否则 K 值虚降)"
            )
        for name in CYCLE_COST_COMPONENTS:
            if getattr(self, name) is None:
                findings.append(f"SpeculationCycle: cost component {name!r} is unmeasured")
        if not self.actual_backend:
            findings.append(
                "SpeculationCycle: actual_backend is required (缺失即 INVALID_IDENTITY，无法证明命中 speculative 路径)"
            )
        return findings

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "cycle_index": self.cycle_index,
            "target_artifact_id": self.target_artifact_id,
            "proposer_artifact_id": self.proposer_artifact_id,
            "proposed_token_ids": list(self.proposed_token_ids),
            "accepted_count": self.accepted_count,
            "reject_position": self.reject_position,
            "committed_token_ids": list(self.committed_token_ids),
            "draft_time_ms": self.draft_time_ms,
            "verify_time_ms": self.verify_time_ms,
            "rollback_time_ms": self.rollback_time_ms,
            "scheduler_time_ms": self.scheduler_time_ms,
            "kv_copy_time_ms": self.kv_copy_time_ms,
            "target_calls": self.target_calls,
            "kv_written_bytes": self.kv_written_bytes,
            "kv_discarded_bytes": self.kv_discarded_bytes,
            "actual_backend": self.actual_backend,
            "state": self.state,
            "status": self.status,
        }


def cycle_committed_tokens(cycle: SpeculationCycle) -> int:
    """Step 17: the number of tokens a cycle actually added to the output.

    ``accepted_count`` alone omits the bonus token, which is why the count is a
    function rather than a field the caller can get wrong.
    """
    return len(cycle.committed_token_ids)


def cycle_cost_ms(cycle: SpeculationCycle) -> float:
    """Step 30: the full cycle cost — every component, no wall-time shortcut."""
    return sum(
        float(getattr(cycle, name) or 0.0)
        for name in CYCLE_COST_COMPONENTS
    )


def effective_tpot(cycles: Sequence[SpeculationCycle]) -> Dict[str, Any]:
    """``effective_TPOT ≈ T_cycle / E[A]`` (§3.2), with the extras included.

    Rejected proposals still cost draft and verify time, so a high acceptance rate
    with an expensive cycle is a regression; the ratio is computed from the
    supplied cycles instead of being asserted.
    """
    if not cycles:
        return {"error": "no cycles supplied"}
    total_cost = sum(cycle_cost_ms(cycle) for cycle in cycles)
    total_committed = sum(cycle_committed_tokens(cycle) for cycle in cycles)
    if total_committed == 0:
        return {"error": "no tokens were committed", "total_cost_ms": total_cost}
    return {
        "cycles": len(cycles),
        "total_cost_ms": total_cost,
        "total_committed_tokens": total_committed,
        "effective_tpot_ms": total_cost / total_committed,
        "target_calls": sum(cycle.target_calls for cycle in cycles),
        "committed_per_target_call": total_committed / max(
            1, sum(cycle.target_calls for cycle in cycles)
        ),
    }


# ── cost model, waste, guards (steps 9, 31–34) ─────────────────────────────


@dataclass(frozen=True)
class SpeculationCostModel:
    """Step 9: the break-even written *before* the run (§10 forbids a post-hoc one)."""

    draft_ms_per_token: float
    verify_ms_per_token: float
    target_decode_ms_per_token: float
    acceptance_by_position: Tuple[float, ...]
    gamma: int
    rollback_ms: float = 0.0
    scheduler_ms: float = 0.0

    def problems(self) -> List[str]:
        findings: List[str] = []
        if len(self.acceptance_by_position) != self.gamma:
            findings.append(
                f"cost model: {len(self.acceptance_by_position)} acceptance positions for gamma {self.gamma}"
            )
        for name in ("draft_ms_per_token", "verify_ms_per_token", "target_decode_ms_per_token"):
            value = getattr(self, name)
            if value <= 0:
                findings.append(f"cost model: {name} must be positive (an assumed 0 makes break-even infinite)")
        if any(probability < 0.0 or probability > 1.0 for probability in self.acceptance_by_position):
            findings.append("cost model: acceptance probabilities must be in [0, 1]")
        return findings

    def predict(self) -> Dict[str, Any]:
        """Predicted ``T_cycle``, ``E[A]`` and the resulting TPOT ratio."""
        cycle = (
            self.draft_ms_per_token * self.gamma
            + self.verify_ms_per_token * self.gamma
            + self.rollback_ms
            + self.scheduler_ms
        )
        expected = expected_accepted_tokens(self.acceptance_by_position, gamma=self.gamma)
        predicted_tpot = cycle / expected if expected > 0 else float("inf")
        break_even = predicted_tpot < self.target_decode_ms_per_token
        return {
            "gamma": self.gamma,
            "predicted_cycle_ms": cycle,
            "expected_accepted": expected,
            "predicted_tpot_ms": predicted_tpot,
            "target_only_tpot_ms": self.target_decode_ms_per_token,
            "break_even": break_even,
            "required_acceptance_for_break_even": _required_acceptance(
                cycle, self.target_decode_ms_per_token, self.gamma
            ),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "draft_ms_per_token": self.draft_ms_per_token,
            "verify_ms_per_token": self.verify_ms_per_token,
            "target_decode_ms_per_token": self.target_decode_ms_per_token,
            "acceptance_by_position": list(self.acceptance_by_position),
            "gamma": self.gamma,
            "rollback_ms": self.rollback_ms,
            "scheduler_ms": self.scheduler_ms,
        }


def _required_acceptance(cycle_ms: float, target_tpot_ms: float, gamma: int) -> Optional[float]:
    """The uniform per-position acceptance needed to break even.

    Returned so a run whose measured acceptance is below it can be *explained*
    rather than reported as an unexplained regression.  ``None`` means the cycle
    is already too expensive for any acceptance (an important early stop).
    """
    if target_tpot_ms <= 0 or cycle_ms <= 0:
        return None
    required_expected = cycle_ms / target_tpot_ms
    if required_expected >= gamma:
        return None
    if required_expected <= 0:
        return 0.0
    # solve  sum_{i=1..gamma} a^i = required_expected  for a in (0, 1)
    low, high = 0.0, 1.0 - 1e-12
    for _ in range(200):
        mid = (low + high) / 2.0
        total = 0.0
        power = 1.0
        for _index in range(gamma):
            power *= mid
            total += power
        if total < required_expected:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def wasted_work(cycles: Sequence[SpeculationCycle], *, bytes_per_token: int = 0) -> Dict[str, Any]:
    """Step 31: rejected proposals are real work that was thrown away.

    Reporting only accepted tokens hides the cost the candidate introduced, so
    the discarded proposals, their KV writes and their padding are accounted.
    """
    if not cycles:
        return {"error": "no cycles supplied"}
    proposed = sum(len(cycle.proposed_token_ids) for cycle in cycles)
    accepted = sum(cycle.accepted_count for cycle in cycles)
    rejected = proposed - accepted
    discarded_kv = sum(cycle.kv_discarded_bytes for cycle in cycles)
    return {
        "proposed": proposed,
        "accepted": accepted,
        "rejected": rejected,
        "wasted_ratio": rejected / proposed if proposed else 0.0,
        "kv_discarded_bytes": discarded_kv,
        "wasted_bytes_if_unaccounted": rejected * bytes_per_token,
        "note": "被丢弃的 proposal 对应真实 FLOPs/bytes/KV 写入，必须计入成本（step 31）",
    }


def low_acceptance_guard(
    *, measured_acceptance_by_position: Sequence[float], model: SpeculationCostModel, safety_margin: float = 0.1
) -> Dict[str, Any]:
    """Step 33: the system must not stay slower than target-only without a guard.

    ``E14-F1`` §9: 系统在最坏条件持续比 target-only 更差而无保护 is a FAIL, so the
    guard states the acceptance floor and the action taken below it.
    """
    predicted = model.predict()
    required = predicted["required_acceptance_for_break_even"]
    measured = expected_accepted_tokens(measured_acceptance_by_position) / max(
        1, len(measured_acceptance_by_position)
    )
    below = required is None or measured < required * (1.0 - safety_margin)
    return {
        "required_acceptance": required,
        "measured_mean_acceptance": measured,
        "safety_margin": safety_margin,
        "below_break_even": below,
        "action": "fallback_to_target_only" if below else "continue",
        "reason_code": "QUALITY_GATE_FAILED" if below else "",
        "note": "低于 break-even 时必须回退并在 trace 中记录 actual path",
    }


def propose_with_entropy(
    *, entropy_bits: Sequence[float], top1_margins: Sequence[float], positions: int
) -> Dict[str, Any]:
    """Steps 23/24: a truncation policy that depends on difficulty, not on a constant.

    Reporting "acceptance is higher on easy prompts" is only useful if *difficulty*
    is measured; this pairs each position with the target's entropy and top-1
    margin so the stratification is reproducible.
    """
    if len(entropy_bits) < positions or len(top1_margins) < positions:
        raise ConfigError("entropy_bits and top1_margins must cover every proposed position")
    tiers: List[Dict[str, Any]] = []
    for index in range(positions):
        entropy = float(entropy_bits[index])
        margin = float(top1_margins[index])
        tiers.append(
            {
                "position": index,
                "entropy_bits": entropy,
                "top1_margin": margin,
                "tier": "low_entropy" if entropy < 1.0 else ("mid_entropy" if entropy < 3.0 else "high_entropy"),
                "confident": margin > 1.0,
            }
        )
    return {
        "positions": tiers,
        "confidence_fraction": sum(1 for tier in tiers if tier["confident"]) / len(tiers),
        "note": "接受率与 entropy/top-1 margin 一起报告，才能说明哪里可获益（step 23）",
    }


def proposer_failure_fallback(
    *, failure: str, target_only_tokens: Sequence[int], speculative_tokens: Sequence[int]
) -> Dict[str, Any]:
    """Step 34: a proposer failure must degrade to target-only *without* changing output.

    ``E14-F1`` PASS requires 低 acceptance、取消和 proposer failure 有界处理; losing
    or duplicating a token during failover is a FAIL even if the final text
    happens to look plausible.
    """
    known = ("timeout", "unavailable", "shape_error", "version_mismatch", "quality_degraded")
    if failure not in known:
        raise ConfigError(f"unknown proposer failure {failure!r}; known: {', '.join(known)}")
    identical = list(target_only_tokens) == list(speculative_tokens)
    return {
        "failure": failure,
        "fallback": "target_only",
        "reason_code": "DEPENDENCY_ABSENT" if failure in ("unavailable", "timeout") else "SHAPE_UNSUPPORTED",
        "tokens_identical": identical,
        "status": rec.STATUS_NOT_RUN if identical else rec.STATUS_FAIL_CORRECTNESS,
        "note": "中途丢请求或重复 token 即 FAIL；fallback 必须记录 requested/actual",
    }


def check_cancel_release(events: Sequence[Mapping[str, Any]]) -> List[str]:
    """Step 35: a cancelled request must not keep device work or KV alive."""
    problems: List[str] = []
    for event in events:
        stage = str(event.get("stage", "<unnamed>"))
        if not {"proposal", "verify", "streaming"} & {stage}:
            problems.append(f"cancel event {stage!r} is not at a documented stage")
        if event.get("device_work_remaining"):
            problems.append(f"cancel at {stage}: device work was not released")
        if event.get("kv_blocks_leaked"):
            problems.append(f"cancel at {stage}: KV blocks leaked")
        if event.get("duplicate_token_emitted"):
            problems.append(f"cancel at {stage}: a token was emitted after cancellation")
    return problems


def check_quality_gate(
    *, paired_delta: float, guard_band: float, mode: str, distribution_test_passed: Optional[bool]
) -> Dict[str, Any]:
    """Step 36: quality first, and "exact" is a claim that needs evidence.

    ``E14-F1`` §9: theoretical exact 但实现 token 分歧 → 判 correctness fail,
    不能用任务分数接近掩盖.  In sampling mode an "exact" claim without a
    distribution test is therefore refused rather than accepted by default.
    """
    problems: List[str] = []
    if mode not in GENERATION_MODES:
        problems.append(f"unknown generation mode {mode!r}")
    if paired_delta < -abs(guard_band):
        problems.append(
            f"paired quality delta {paired_delta:.4f} exceeds the guard band {guard_band:.4f}"
        )
    if mode == "sampling" and distribution_test_passed is None:
        problems.append(
            "sampling mode without a distribution test cannot claim exactness "
            "(单 seed token 相同不构成证据)"
        )
    if mode == "sampling" and distribution_test_passed is False:
        problems.append("the sampling distribution test failed: correctness fail, not a quality trade-off")
    return {
        "mode": mode,
        "paired_delta": paired_delta,
        "guard_band": guard_band,
        "distribution_test_passed": distribution_test_passed,
        "problems": problems,
        "performance_eligible": not problems,
    }


def reconcile_prediction(*, predicted_tpot_ms: float, measured_tpot_ms: float) -> Dict[str, Any]:
    """Step 37: rebuild the measured TPOT from the model and report the residual.

    ``E14-F1`` §12 requires the收益 or 退化 to be explainable by profile/trace; an
    unresolved residual is recorded as such instead of being narrated away.
    """
    if predicted_tpot_ms <= 0:
        return {"error": "predicted_tpot_ms must be positive"}
    residual = measured_tpot_ms - predicted_tpot_ms
    return {
        "predicted_tpot_ms": predicted_tpot_ms,
        "measured_tpot_ms": measured_tpot_ms,
        "residual_ms": residual,
        "residual_ratio": residual / predicted_tpot_ms,
        "explained": abs(residual) <= 0.1 * predicted_tpot_ms,
        "note": "端到端变化必须由 draft/verify/accepted/KV/scheduler 重建；解释不了就写 residual",
    }


# ── adoption (step 40) ─────────────────────────────────────────────────────


def f1_adoption(
    *,
    decision_id: str,
    quality: Mapping[str, Any],
    effective: Mapping[str, Any],
    model: SpeculationCostModel,
    stratified: Mapping[str, Any],
    complexity: Mapping[str, Any],
    evidence_refs: Sequence[str],
) -> AdoptionDecision:
    """Step 40: a peak speedup does not decide adoption (§10).

    Quality failure outranks any performance result, and a benefit confined to
    one stratum becomes ``ADOPT_EXPERIMENTAL`` rather than ``ADOPT_CORE``.
    """
    problems: List[str] = []
    if not quality.get("performance_eligible"):
        problems.append("quality gate not passed: performance rows are ineligible")
    if effective.get("error"):
        problems.append(f"no usable cycle data: {effective['error']}")
    strata = (stratified.get("strata") or {})
    beneficial = [name for name, row in strata.items() if row.get("accepted")]
    if len(beneficial) <= 1 and beneficial:
        decision = rec.ADOPT_EXPERIMENTAL
        allowed = (f"在 stratum {beneficial[0]} 内观察到 effect",)
    elif beneficial:
        decision = rec.RESEARCH_ONLY
        allowed = ("acceptance 与成本可重建",)
    else:
        decision = rec.BLOCKED_EVIDENCE
        allowed = ()
    if problems:
        decision = rec.REJECT_QUALITY
    return AdoptionDecision(
        decision_id=decision_id,
        experiment_id=EXPERIMENT_ID,
        decision=decision,
        allowed_claims=allowed,
        forbidden_claims=(
            "acceptance 高 ⇒ 更快",
            "单请求加速外推到服务",
            "把 prefix cache 收益算作 speculative 收益",
        ),
        quality_status=rec.STATUS_PASS if quality.get("performance_eligible") else rec.STATUS_FAIL_QUALITY,
        performance_status=rec.STATUS_NOT_RUN,
        maturity=rec.MATURITY_SEMANTIC_VERIFIED if not problems else rec.MATURITY_SOURCE_INTEGRATED,
        evidence_refs=tuple(evidence_refs),
        limitations=tuple(problems) + (
            f"预测 break-even 需要 acceptance {model.predict()['required_acceptance_for_break_even']}",
        ),
        reopened_if=("acceptable workload 与质量门同时满足时可以重新评估",),
    )


def smoke_self_check() -> Dict[str, Any]:
    """CPU-only self-check of the E14-F1 interfaces (labelled smoke, not an experiment)."""
    greedy = greedy_prefix_match([7, 8, 9], [7, 8, 5])
    residual = residual_distribution([0.5, 0.5], [0.9, 0.1])
    test = accept_reject_test(target_probs=[0.5, 0.5], draft_probs=[0.9, 0.1], token=0, uniform=0.5)
    acceptance = [0.8, 0.6, 0.4]
    expected = expected_accepted_tokens(acceptance, gamma=3)
    binding = TargetProposerBinding(
        target_artifact_id="serving::sha256:" + "0" * 64,
        proposer_artifact_id="serving::sha256:" + "1" * 64,
        proposer_kind="classic_draft",
        target_tokenizer_id="tok-a",
        proposer_tokenizer_id="tok-b",
        target_vocab_size=151936,
        proposer_vocab_size=151936,
        proposer_precision="bf16",
    )
    contract = GenerationContract(mode="sampling", max_tokens=64, eos_token_id=1, seed=7, temperature=0.7)
    model = SpeculationCostModel(
        draft_ms_per_token=0.2, verify_ms_per_token=0.6, target_decode_ms_per_token=3.0,
        acceptance_by_position=tuple(acceptance), gamma=3,
    )
    guard = low_acceptance_guard(measured_acceptance_by_position=[0.1, 0.05, 0.02], model=model)
    return {
        "status": "smoke",
        "claim_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "greedy_accepted": greedy["accepted"],
        "greedy_bonus": greedy["bonus"],
        "residual": [round(value, 4) for value in residual],
        "accept_decision": test["path"],
        "expected_accepted": round(expected, 4),
        "binding_problems": binding.problems(),
        "contract_mode": contract.mode,
        "predicted_break_even": model.predict()["break_even"],
        "guard_action": guard["action"],
        "strata_keys": list(ACCEPTANCE_STRATA),
    }


# ── protocol step table (40 steps of details/S14/E14-F1) ───────────────────

PROTOCOL_STEPS: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = (
    (1, "绑定 E14-05 协议", ("frontier:issue_contract", "frontier:contract_hash")),
    (2, "冻结 speculative 机制版本", ("speculative:PROPOSER_KINDS", "speculative:TargetProposerBinding.proposer_kind")),
    (3, "冻结 target ModelArtifact", ("contracts:ServingModelArtifact", "contracts:ServingModelArtifact.quality_evidence_id")),
    (4, "冻结 proposer artifact", ("speculative:TargetProposerBinding", "identity:artifact_ref")),
    (5, "验证 tokenizer/vocabulary compatibility", ("speculative:TargetProposerBinding.problems",
                                                    "speculative:TargetProposerBinding.token_id_mapping")),
    (6, "冻结生成语义", ("speculative:GenerationContract", "speculative:GENERATION_MODES")),
    (7, "冻结 workload strata", ("speculative:ACCEPTANCE_STRATA", "frontier:WorkloadStrata")),
    (8, "冻结 service load", ("frontier:BaselinePair", "records:EXPERIMENT_UNITS")),
    (9, "预注册理论成本模型", ("speculative:SpeculationCostModel", "speculative:SpeculationCostModel.predict")),
    (10, "运行 capability/actual-path probe", ("contracts:check_actual_path_recorded", "records:STATUS_INVALID_IDENTITY")),
    (11, "建立 target-only correctness baseline", ("speculative:greedy_prefix_match", "parity:REFERENCE_PATHS")),
    (12, "建立 target-only 性能基线", ("contracts:check_resource_ledger", "telemetry:coverage_report")),
    (13, "建立 proposer 独立能力基线", ("speculative:TargetProposerBinding", "speculative:propose_with_entropy")),
    (14, "构造手算 acceptance case", ("speculative:accept_reject_test", "speculative:residual_distribution")),
    (15, "验证 greedy 逐 token parity", ("speculative:greedy_prefix_match", "speculative:SpeculationCycle.committed_token_ids")),
    (16, "验证 sampling 分布语义", ("speculative:residual_distribution", "speculative:check_quality_gate")),
    (17, "插桩 proposal/verification 状态机", ("speculative:SpeculationCycle", "speculative:acceptance_by_position")),
    (18, "验证 EOS/stop/长度边界", ("speculative:STOP_REASONS", "speculative:GenerationContract.eos_token_id")),
    (19, "验证 KV commit/rollback", ("speculative:KV_ACTIONS", "speculative:SpeculationCycle.kv_discarded_bytes")),
    (20, "运行主参数 γ/steps 扫描", ("speculative:SpeculationCostModel.gamma", "frontier:StatisticsPlan")),
    (21, "运行 tree width/depth 消融", ("frontier:AblationMatrix", "speculative:PROPOSER_KINDS")),
    (22, "运行 draft size/precision 消融", ("speculative:TargetProposerBinding.proposer_precision",
                                            "frontier:CostDenominator")),
    (23, "运行 domain/entropy 分层", ("speculative:stratify_by_domain", "speculative:propose_with_entropy")),
    (24, "运行 context/output 长度分层", ("speculative:stratify_by_domain", "frontier:WorkloadStrata.holdout_id")),
    (25, "运行单请求与固定 batch", ("speculative:effective_tpot", "records:TABLE_SCHEMAS")),
    (26, "运行多并发 continuous batching", ("speculative:effective_tpot", "contracts:check_resource_ledger")),
    (27, "运行 open-loop SLO 曲线", ("records:PROFILE_LAYERS", "contracts:AdoptionDecision.performance_status")),
    (28, "验证长短/高低 acceptance 公平性", ("speculative:stratify_by_domain", "frontier:StopRules")),
    (29, "验证 prefix cache 交互", ("posttraining:check_kv_reuse_identity", "speculative:wasted_work")),
    (30, "采集 kernel/系统 profile", ("speculative:cycle_cost_ms", "records:PROFILE_LAYER_FIELDS")),
    (31, "计算计算与内存浪费", ("speculative:wasted_work", "speculative:SpeculationCycle.kv_written_bytes")),
    (32, "测显存、能耗和设备成本", ("contracts:check_resource_ledger", "frontier:CostDenominator")),
    (33, "注入低 acceptance/错误 proposer", ("speculative:low_acceptance_guard", "speculative:proposer_failure_fallback")),
    (34, "注入 proposer failure/timeout", ("speculative:proposer_failure_fallback", "campaign:REQUIRED_ISOLATION")),
    (35, "验证 cancel/backpressure/drain", ("speculative:check_cancel_release", "speculative:STOP_REASONS")),
    (36, "运行任务质量门", ("speculative:check_quality_gate", "contracts:check_quality_before_performance")),
    (37, "对账预测与实测", ("speculative:reconcile_prediction", "speculative:SpeculationCostModel.predict")),
    (38, "在 holdout workloads 确认", ("frontier:WorkloadStrata.holdout_id", "frontier:StatisticsPlan")),
    (39, "跨 run/time block 重复", ("records:EXPERIMENT_UNITS", "frontier:StatisticsPlan.minimum_repeats")),
    (40, "形成 F1 AdoptionDecision", ("speculative:f1_adoption", "contracts:AdoptionDecision.validate")),
)
