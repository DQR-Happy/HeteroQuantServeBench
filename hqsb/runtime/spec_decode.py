"""Speculative decoding / MTP contract and benefit model (E07-08, P1).

E07-08 is a P1 experiment: it is only required when the project actually claims
speculative decoding or MTP.  Two consequences shape this module:

* **No claim by default.**  :func:`claim_status` returns ``NOT_RUN``/``NOT_CLAIMED``
  unless someone executed the experiment and produced the evidence; the module
  itself never enables a claim.
* **The algorithm must be named.**  "draft then let the target pick" does not
  preserve the target distribution.  :class:`SpecAlgorithm` therefore enumerates
  ``greedy``, ``strict_sampling`` and ``mtp``, and
  :func:`assert_mtp_does_not_borrow` refuses a report that quotes the strict
  speculative-sampling guarantee for an MTP path.

The acceptance mathematics uses :class:`fractions.Fraction`, so the acceptance
probability ``min(1, p/q)`` and the residual distribution
``normalize(max(0, p - q))`` are checked *exactly* rather than to a tolerance —
that is the difference between a verified sampler and an approximate one.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError, SchemaError

# ── algorithms ─────────────────────────────────────────────────────────────

GREEDY = "greedy"
STRICT_SAMPLING = "strict_sampling"
MTP = "mtp"

SPEC_ALGORITHMS: Tuple[str, ...] = (GREEDY, STRICT_SAMPLING, MTP)


@dataclass(frozen=True)
class DraftTargetIdentity:
    """Identity/compatibility of the draft/target pair (E07-08 §4)."""

    target_model_id: str
    target_revision: str
    target_tokenizer_id: str
    target_vocab_size: int
    target_precision: str
    draft_model_id: str = ""
    draft_revision: str = ""
    draft_tokenizer_id: str = ""
    draft_vocab_size: int = 0
    token_mapping_verified: bool = False
    kv_layout: str = "paged"
    max_context: int = 0
    gamma: int = 3

    def __post_init__(self) -> None:
        if not self.target_model_id or not self.target_revision:
            raise ConfigError("the target artifact must be identified")
        if self.gamma < 1:
            raise ConfigError("gamma must be >= 1", details={"field": "gamma"})
        if self.draft_model_id and self.draft_vocab_size <= 0:
            raise ConfigError(
                "a draft model must declare its vocabulary size; otherwise a silent "
                "token-mapping mismatch cannot be detected",
                details={"field": "draft_vocab_size"},
            )

    @property
    def uses_draft_model(self) -> bool:
        return bool(self.draft_model_id)

    def compatibility(self) -> Dict[str, Any]:
        """Reject an incompatible pair; the caller must not proceed quietly."""
        problems: List[str] = []
        if self.uses_draft_model:
            if self.draft_vocab_size != self.target_vocab_size and not (
                self.token_mapping_verified
            ):
                problems.append(
                    f"draft vocab {self.draft_vocab_size} != target vocab "
                    f"{self.target_vocab_size} without a verified token mapping"
                )
            if (
                self.draft_tokenizer_id
                and self.draft_tokenizer_id != self.target_tokenizer_id
                and not self.token_mapping_verified
            ):
                problems.append(
                    "draft and target use different tokenizers without a verified "
                    "mapping"
                )
        return {
            "compatible": not problems,
            "problems": problems,
            "identity": self.as_dict(),
        }

    def require_compatible(self) -> None:
        report = self.compatibility()
        if not report["compatible"]:
            raise SchemaError(
                "draft/target pair is incompatible: " + "; ".join(report["problems"]),
                details={"fields": report["problems"]},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target_model_id": self.target_model_id,
            "target_revision": self.target_revision,
            "target_tokenizer_id": self.target_tokenizer_id,
            "target_vocab_size": self.target_vocab_size,
            "target_precision": self.target_precision,
            "draft_model_id": self.draft_model_id,
            "draft_revision": self.draft_revision,
            "draft_tokenizer_id": self.draft_tokenizer_id,
            "draft_vocab_size": self.draft_vocab_size,
            "token_mapping_verified": self.token_mapping_verified,
            "kv_layout": self.kv_layout,
            "max_context": self.max_context,
            "gamma": self.gamma,
        }


# ── acceptance / residual mathematics (exact rationals) ────────────────────


def acceptance_probability(target_p: Fraction, draft_q: Fraction) -> Fraction:
    """``min(1, p/q)`` — undefined when the draft assigns zero probability."""
    if draft_q == 0:
        raise ConfigError(
            "acceptance probability is undefined when the draft probability is 0",
            details={"field": "draft_q"},
        )
    if not 0 <= target_p <= 1 or not 0 <= draft_q <= 1:
        raise ConfigError("probabilities must be in [0, 1]")
    return min(Fraction(1), target_p / draft_q)


def residual_distribution(
    target_p: Sequence[Fraction], draft_q: Sequence[Fraction]
) -> Tuple[Fraction, ...]:
    """``normalize(max(0, p - q))`` — the distribution a rejection samples from."""
    if len(target_p) != len(draft_q):
        raise ConfigError(
            "target and draft distributions must share the vocabulary size",
            details={"field": "distribution"},
        )
    if not target_p:
        raise ConfigError("an empty distribution cannot be normalised")
    raw = [max(Fraction(0), p - q) for p, q in zip(target_p, draft_q)]
    total = sum(raw)
    if total == 0:
        raise ConfigError(
            "the residual distribution is degenerate (p <= q everywhere); the "
            "algorithm is not applicable to this pair",
            details={"field": "distribution"},
        )
    return tuple(value / total for value in raw)


def residual_is_normalized(residual: Sequence[Fraction]) -> bool:
    return bool(residual) and sum(residual) == 1


@dataclass(frozen=True)
class AcceptanceAudit:
    """One hand-computed acceptance case (E07-08 steps 4–5)."""

    case: str
    target_p: Tuple[Fraction, ...]
    draft_q: Tuple[Fraction, ...]
    sampled_uniform: Fraction
    expected_accepted: bool
    expected_next_token: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case,
            "target_p": [str(value) for value in self.target_p],
            "draft_q": [str(value) for value in self.draft_q],
            "sampled_uniform": str(self.sampled_uniform),
            "expected_accepted": self.expected_accepted,
            "expected_next_token": self.expected_next_token,
        }


def verify_acceptance(
    target_p: Sequence[Fraction],
    draft_q: Sequence[Fraction],
    proposed_token: int,
    uniform_draw: Fraction,
) -> Dict[str, Any]:
    """Apply the strict acceptance rule for one proposed token."""
    if not 0 <= proposed_token < len(target_p):
        raise ConfigError("proposed token is outside the vocabulary")
    if not 0 <= uniform_draw <= 1:
        raise ConfigError("the uniform draw must be in [0, 1]")
    probability = acceptance_probability(target_p[proposed_token], draft_q[proposed_token])
    accepted = uniform_draw <= probability
    residual = residual_distribution(target_p, draft_q)
    return {
        "proposed_token": proposed_token,
        "acceptance_probability": str(probability),
        "accepted": accepted,
        "residual_normalized": residual_is_normalized(residual),
        "residual": [str(value) for value in residual],
    }


def golden_acceptance_cases() -> Tuple[AcceptanceAudit, ...]:
    """All-accept / first-reject / none / near-tie hand-computed cases."""
    p = (Fraction(1, 2), Fraction(3, 10), Fraction(1, 5))
    q = (Fraction(1, 2), Fraction(1, 2), Fraction(0))
    return (
        AcceptanceAudit(
            case="all_accept",
            target_p=p,
            draft_q=q,
            sampled_uniform=Fraction(1, 10),
            expected_accepted=True,
            expected_next_token=0,
        ),
        AcceptanceAudit(
            case="first_reject",
            target_p=p,
            draft_q=q,
            sampled_uniform=Fraction(9, 10),
            expected_accepted=False,
            expected_next_token=1,
        ),
        AcceptanceAudit(
            case="draft_zero_probability",
            target_p=p,
            draft_q=q,
            sampled_uniform=Fraction(1, 2),
            expected_accepted=True,
            expected_next_token=2,
        ),
    )


# ── cycles, commit/rollback, benefit model ─────────────────────────────────


@dataclass(frozen=True)
class CycleRecord:
    """One draft→verify→accept→rollback cycle (E07-08 §5/§8)."""

    cycle_index: int
    proposed: int
    accepted: int
    rejected_index: int = -1
    correction_token_committed: bool = False
    advanced_tokens: int = 0
    target_calls: int = 1
    draft_ms: float = 0.0
    verify_ms: float = 0.0
    accept_ms: float = 0.0
    rollback_ms: float = 0.0
    scheduler_ms: float = 0.0
    draft_kv_blocks: int = 0
    target_kv_blocks: int = 0
    eos_reached: bool = False

    def __post_init__(self) -> None:
        if self.proposed < 0 or self.accepted < 0:
            raise ConfigError("proposed/accepted counts must be non-negative")
        if self.accepted > self.proposed:
            raise ConfigError(
                "accepted tokens cannot exceed proposed tokens",
                details={"field": "accepted"},
            )
        expected_advance = self.accepted + (1 if self.correction_token_committed else 0)
        if self.advanced_tokens != expected_advance:
            raise ConfigError(
                "advanced_tokens must equal accepted + (correction or bonus token): "
                f"{self.advanced_tokens} != {expected_advance}. Conflating 'accepted' "
                "with 'advanced' inflates the speedup (E07-08 §12)",
                details={"field": "advanced_tokens"},
            )

    @property
    def cycle_time_ms(self) -> float:
        return (
            self.draft_ms
            + self.verify_ms
            + self.accept_ms
            + self.rollback_ms
            + self.scheduler_ms
        )

    @property
    def acceptance_rate(self) -> float:
        if self.proposed == 0:
            return 0.0
        return self.accepted / self.proposed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cycle_index": self.cycle_index,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "rejected_index": self.rejected_index,
            "correction_token_committed": self.correction_token_committed,
            "advanced_tokens": self.advanced_tokens,
            "target_calls": self.target_calls,
            "cycle_time_ms": self.cycle_time_ms,
            "acceptance_rate": self.acceptance_rate,
            "draft_kv_blocks": self.draft_kv_blocks,
            "target_kv_blocks": self.target_kv_blocks,
            "eos_reached": self.eos_reached,
        }


def verify_greedy_exactness(
    vanilla_tokens: Sequence[int],
    speculative_tokens: Sequence[int],
    *,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Greedy speculative decoding must match vanilla token-for-token."""
    limit = min(len(vanilla_tokens), len(speculative_tokens), max_new_tokens)
    first_mismatch = -1
    for index in range(limit):
        if vanilla_tokens[index] != speculative_tokens[index]:
            first_mismatch = index
            break
    truncated = len(speculative_tokens) < min(len(vanilla_tokens), max_new_tokens)
    return {
        "ok": first_mismatch < 0 and not truncated,
        "first_mismatch_index": first_mismatch,
        "compared_tokens": limit,
        "vanilla_length": len(vanilla_tokens),
        "speculative_length": len(speculative_tokens),
        "truncated": truncated,
    }


def distribution_gate(
    left_samples: Sequence[float],
    right_samples: Sequence[float],
    *,
    pre_registered_bound: float,
    min_seeds: int = 8,
) -> Dict[str, Any]:
    """Sampling-mode quality gate; too few seeds stays ``INCONCLUSIVE``."""
    if pre_registered_bound <= 0:
        raise ConfigError("the distribution gate needs a pre-registered bound")
    if len(left_samples) < min_seeds or len(right_samples) < min_seeds:
        return {
            "verdict": "INCONCLUSIVE",
            "reason": f"need {min_seeds} seeds per side, got {len(left_samples)}/"
            f"{len(right_samples)}",
            "bound": pre_registered_bound,
        }
    left = sum(left_samples) / len(left_samples)
    right = sum(right_samples) / len(right_samples)
    delta = left - right
    return {
        "verdict": "PASS" if abs(delta) <= pre_registered_bound else "FAIL",
        "delta": delta,
        "bound": pre_registered_bound,
        "seeds_left": len(left_samples),
        "seeds_right": len(right_samples),
    }


def kv_commit_rollback_audit(
    *,
    proposed_positions: int,
    accepted_positions: int,
    committed_positions: int,
    rolled_back_positions: int,
    stale_token_detected: bool = False,
) -> Dict[str, Any]:
    """Token/KV conservation is a hard gate for E07-08 (§7)."""
    problems: List[str] = []
    if accepted_positions > proposed_positions:
        problems.append("accepted positions exceed proposed positions")
    if committed_positions not in (accepted_positions, accepted_positions + 1):
        problems.append(
            "committed positions must be the accepted tokens plus at most one "
            "correction/bonus token"
        )
    if rolled_back_positions != proposed_positions - accepted_positions:
        problems.append(
            "rolled-back positions must equal proposed - accepted "
            f"({rolled_back_positions} != {proposed_positions - accepted_positions})"
        )
    if stale_token_detected:
        problems.append("a rejected draft token survived into the committed output")
    return {
        "ok": not problems,
        "problems": problems,
        "proposed_positions": proposed_positions,
        "accepted_positions": accepted_positions,
        "committed_positions": committed_positions,
        "rolled_back_positions": rolled_back_positions,
    }


@dataclass(frozen=True)
class BenefitModel:
    """Cycle-cost model of E07-08 §8, computed from measured cycle records."""

    cycles: Tuple[CycleRecord, ...]
    vanilla_tpot_ms: float

    def __post_init__(self) -> None:
        if not self.cycles:
            raise ConfigError("the benefit model needs at least one cycle")
        if self.vanilla_tpot_ms <= 0:
            raise ConfigError("vanilla_tpot_ms must be positive")

    @property
    def total_cycle_time_ms(self) -> float:
        return sum(cycle.cycle_time_ms for cycle in self.cycles)

    @property
    def total_advanced(self) -> int:
        return sum(cycle.advanced_tokens for cycle in self.cycles)

    @property
    def effective_tpot_ms(self) -> float:
        if self.total_advanced == 0:
            raise ConfigError(
                "no token was advanced: effective TPOT is undefined, report the "
                "negative result instead of a division by zero"
            )
        return self.total_cycle_time_ms / self.total_advanced

    @property
    def target_calls_per_output_token(self) -> float:
        calls = sum(cycle.target_calls for cycle in self.cycles)
        if self.total_advanced == 0:
            return float("inf")
        return calls / self.total_advanced

    @property
    def accepted_over_proposed(self) -> float:
        proposed = sum(cycle.proposed for cycle in self.cycles)
        accepted = sum(cycle.accepted for cycle in self.cycles)
        return accepted / proposed if proposed else 0.0

    def speedup(self) -> float:
        return self.vanilla_tpot_ms / self.effective_tpot_ms

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cycles": [cycle.as_dict() for cycle in self.cycles],
            "total_cycle_time_ms": self.total_cycle_time_ms,
            "total_advanced": self.total_advanced,
            "effective_tpot_ms": self.effective_tpot_ms,
            "target_calls_per_output_token": self.target_calls_per_output_token,
            "accepted_over_proposed": self.accepted_over_proposed,
            "vanilla_tpot_ms": self.vanilla_tpot_ms,
            "speedup": self.speedup(),
            "note": (
                "a high acceptance rate does not imply a speedup: the draft cost is "
                "inside cycle_time_ms"
            ),
        }


def gamma_sweep(gammas: Sequence[int]) -> Dict[str, Any]:
    """Pre-registered gamma candidates, including 1 as the control."""
    if not gammas or any(gamma < 1 for gamma in gammas):
        raise ConfigError("gamma candidates must be >= 1")
    if 1 not in gammas:
        raise ConfigError(
            "a gamma sweep must include 1: without the control, the measured effect "
            "cannot be separated from the verification overhead",
            details={"field": "gammas"},
        )
    return {
        "gammas": sorted(set(gammas)),
        "control": 1,
        "note": "gamma is frozen before results; choosing it afterwards is prohibited",
    }


# ── MTP contract and claim gate ────────────────────────────────────────────


@dataclass(frozen=True)
class MtpContract:
    """MTP must declare its own quality contract (E07-08 §6)."""

    heads: int
    quality_contract: str
    borrows_strict_sampling_guarantee: bool = False
    target_distribution_preserved: bool = False

    def __post_init__(self) -> None:
        if self.heads < 1:
            raise ConfigError("MTP needs at least one head")
        if not self.quality_contract:
            raise ConfigError(
                "MTP must state its quality contract explicitly; an unspecified "
                "contract cannot be validated",
                details={"field": "quality_contract"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "heads": self.heads,
            "quality_contract": self.quality_contract,
            "borrows_strict_sampling_guarantee": self.borrows_strict_sampling_guarantee,
            "target_distribution_preserved": self.target_distribution_preserved,
        }


def assert_mtp_does_not_borrow(contract: MtpContract) -> None:
    """Refuse to attribute the strict-sampling guarantee to an MTP path."""
    if contract.borrows_strict_sampling_guarantee and not contract.target_distribution_preserved:
        raise SchemaError(
            "an MTP path may not quote the strict speculative-sampling distribution "
            "guarantee unless it proves target-distribution preservation itself "
            "(E07-08 §6)",
            details={"field": "borrows_strict_sampling_guarantee"},
        )


@dataclass(frozen=True)
class SpeculativeEvidence:
    """Evidence required before a speculative claim (E07-08 §13)."""

    algorithm_frozen: bool = False
    identity_verified: bool = False
    vanilla_oracle_compared: bool = False
    acceptance_verified: bool = False
    kv_rollback_audited: bool = False
    cycle_cost_decomposed: bool = False
    workload_slices_measured: bool = False
    actual_path_recorded: bool = False

    @property
    def complete(self) -> bool:
        return all(
            (
                self.algorithm_frozen,
                self.identity_verified,
                self.vanilla_oracle_compared,
                self.acceptance_verified,
                self.kv_rollback_audited,
                self.cycle_cost_decomposed,
                self.workload_slices_measured,
                self.actual_path_recorded,
            )
        )

    def missing(self) -> List[str]:
        names = (
            "algorithm_frozen",
            "identity_verified",
            "vanilla_oracle_compared",
            "acceptance_verified",
            "kv_rollback_audited",
            "cycle_cost_decomposed",
            "workload_slices_measured",
            "actual_path_recorded",
        )
        return [name for name in names if not getattr(self, name)]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_frozen": self.algorithm_frozen,
            "identity_verified": self.identity_verified,
            "vanilla_oracle_compared": self.vanilla_oracle_compared,
            "acceptance_verified": self.acceptance_verified,
            "kv_rollback_audited": self.kv_rollback_audited,
            "cycle_cost_decomposed": self.cycle_cost_decomposed,
            "workload_slices_measured": self.workload_slices_measured,
            "actual_path_recorded": self.actual_path_recorded,
            "missing": self.missing(),
        }


def claim_status(
    evidence: SpeculativeEvidence, *, executed: bool, algorithm: str
) -> Dict[str, Any]:
    """E07-08 is P1: without an explicit claim the status is ``NOT_RUN``."""
    if algorithm not in SPEC_ALGORITHMS:
        raise ConfigError(
            f"unknown speculative algorithm {algorithm!r}",
            details={"allowed": list(SPEC_ALGORITHMS)},
        )
    if not executed:
        return {
            "status": "NOT_RUN",
            "reason": (
                "no speculative/MTP claim is made; E07-08 is P1 and is only required "
                "when such a claim exists"
            ),
            "evidence": evidence.as_dict(),
        }
    if not evidence.complete:
        return {
            "status": "NOT_CLAIMED",
            "reason": "evidence incomplete: " + ", ".join(evidence.missing()),
            "evidence": evidence.as_dict(),
        }
    return {"status": "CLAIMED", "reason": "", "evidence": evidence.as_dict()}


def workload_slices() -> Tuple[Dict[str, str], ...]:
    """The slice list of E07-08 §9; the acceptance distribution is reported per slice."""
    return (
        {"slice": "predictable", "description": "high-predictability continuation"},
        {"slice": "code", "description": "code-like output"},
        {"slice": "math", "description": "arithmetic/reasoning output"},
        {"slice": "chat", "description": "open-ended chat output"},
        {"slice": "eos_near", "description": "generation ending soon"},
        {"slice": "adversarial", "description": "low-acceptance output"},
    )


__all__ = [
    "AcceptanceAudit",
    "BenefitModel",
    "CycleRecord",
    "DraftTargetIdentity",
    "GREEDY",
    "MTP",
    "MtpContract",
    "SPEC_ALGORITHMS",
    "STRICT_SAMPLING",
    "SpeculativeEvidence",
    "acceptance_probability",
    "assert_mtp_does_not_borrow",
    "claim_status",
    "distribution_gate",
    "gamma_sweep",
    "golden_acceptance_cases",
    "kv_commit_rollback_audit",
    "residual_distribution",
    "residual_is_normalized",
    "verify_acceptance",
    "verify_greedy_exactness",
    "workload_slices",
]
