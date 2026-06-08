"""Backend semantic parity oracles (E07-01 §5–§9).

E07-01 is the gate that decides whether later experiments may compare anything
at all, so the oracles here are strict about *what may be concluded*:

* the **greedy** oracle compares token IDs step by step and localises the first
  divergence; logits are compared with a pre-registered tolerance when the
  backend exposes them;
* the **sampling** oracle refuses single-seed exact-match reasoning (different
  runtimes consume RNG differently), and instead compares the requested
  statistic across many seeds, returning ``INCONCLUSIVE`` when the sample size
  cannot support a conclusion;
* the **parameter-effect** matrix requires every *supported* field to actually
  change the output or to carry telemetry evidence; "the config was accepted"
  is not evidence that it took effect;
* the **unsupported** matrix turns a silent default into a failure: an
  unsupported field must be rejected or explicitly fallen back, never
  "successfully ignored".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import CapabilityError, ConfigError
from hqsb.runtime.adapter import GenerationResult, StreamChunk, validate_stream
from hqsb.runtime.metrics import distribution_summary, percentile
from hqsb.runtime.request import (
    CLAIMABLE_STATES,
    REQUEST_FIELDS,
    CapabilityReport,
    RequestSpec,
)

# ── greedy oracle ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GreedyParityReport:
    """Step-by-step comparison of two greedy runs of the same request."""

    request_id: str
    reference_tokens: Tuple[int, ...]
    candidate_tokens: Tuple[int, ...]
    first_divergence_index: int
    length_equal: bool
    token_ids_equal: bool
    logits_available: bool
    max_logit_abs_diff: float = 0.0
    tolerance: float = 0.0
    finish_reason_equal: bool = True
    notes: Tuple[str, ...] = ()

    @property
    def tokens_match(self) -> bool:
        return self.token_ids_equal and self.length_equal

    @property
    def logits_within_tolerance(self) -> bool:
        if not self.logits_available:
            return True
        return self.max_logit_abs_diff <= self.tolerance

    @property
    def ok(self) -> bool:
        return self.tokens_match and self.logits_within_tolerance and self.finish_reason_equal

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "reference_tokens": len(self.reference_tokens),
            "candidate_tokens": len(self.candidate_tokens),
            "first_divergence_index": self.first_divergence_index,
            "length_equal": self.length_equal,
            "token_ids_equal": self.token_ids_equal,
            "logits_available": self.logits_available,
            "max_logit_abs_diff": self.max_logit_abs_diff,
            "tolerance": self.tolerance,
            "finish_reason_equal": self.finish_reason_equal,
            "ok": self.ok,
            "notes": list(self.notes),
        }


def compare_greedy(
    reference: GenerationResult,
    candidate: GenerationResult,
    *,
    tolerance: float = 0.0,
    reference_logits: Optional[Sequence[Sequence[float]]] = None,
    candidate_logits: Optional[Sequence[Sequence[float]]] = None,
) -> GreedyParityReport:
    """Compare two greedy generations (E07-01 §5)."""
    if reference.request_id != candidate.request_id:
        raise ConfigError(
            "greedy parity must compare the same request id; comparing two "
            "requests would mix request variance into the result",
            details={"field": "request_id"},
        )
    left = tuple(reference.token_ids)
    right = tuple(candidate.token_ids)
    divergence = -1
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            divergence = index
            break
    notes: List[str] = []
    if divergence < 0 and len(left) != len(right):
        divergence = min(len(left), len(right))
        notes.append("sequences match up to the shorter length only")
    logits_available = reference_logits is not None and candidate_logits is not None
    max_diff = 0.0
    if logits_available:
        if len(reference_logits) != len(candidate_logits):
            notes.append("logit step counts differ; only matching steps compared")
        for left_step, right_step in zip(reference_logits, candidate_logits):
            if len(left_step) != len(right_step):
                raise ConfigError(
                    "logit vectors have different vocabulary sizes",
                    details={"field": "logits"},
                )
            for a, b in zip(left_step, right_step):
                diff = abs(float(a) - float(b))
                max_diff = max(max_diff, diff)
        if max_diff > tolerance:
            notes.append(
                f"logits differ by {max_diff:.6g} > tolerance {tolerance:.6g}; the "
                "token sequence may still match, but the numerical path differs"
            )
    return GreedyParityReport(
        request_id=reference.request_id,
        reference_tokens=left,
        candidate_tokens=right,
        first_divergence_index=divergence,
        length_equal=len(left) == len(right),
        token_ids_equal=left == right,
        logits_available=logits_available,
        max_logit_abs_diff=max_diff,
        tolerance=tolerance,
        finish_reason_equal=reference.finish_reason == candidate.finish_reason,
        notes=tuple(notes),
    )


# ── sampling oracle ────────────────────────────────────────────────────────

SAMPLING_VERDICTS: Tuple[str, ...] = ("PASS", "FAIL", "INCONCLUSIVE")


@dataclass(frozen=True)
class SamplingComparison:
    """Distribution-level comparison across multiple seeds (E07-01 §5)."""

    request_id: str
    mode: str
    seeds_left: int
    seeds_right: int
    statistic: str
    left_summary: Mapping[str, Any]
    right_summary: Mapping[str, Any]
    statistic_delta: float
    pre_registered_bound: float
    verdict: str
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("external_sampler", "multi_seed_distribution"):
            raise ConfigError(
                "sampling comparison must declare whether an external sampler made "
                "the draws identical or whether distributions are compared",
                details={"field": "mode"},
            )
        if self.verdict not in SAMPLING_VERDICTS:
            raise ConfigError(f"unknown sampling verdict {self.verdict!r}")
        if self.mode == "multi_seed_distribution" and min(
            self.seeds_left, self.seeds_right
        ) < 2:
            raise ConfigError(
                "a single seed cannot support a distribution claim; different "
                "runtimes consume RNG differently (E07-01 §5)",
                details={"field": "seeds"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "mode": self.mode,
            "seeds_left": self.seeds_left,
            "seeds_right": self.seeds_right,
            "statistic": self.statistic,
            "left": dict(self.left_summary),
            "right": dict(self.right_summary),
            "statistic_delta": self.statistic_delta,
            "pre_registered_bound": self.pre_registered_bound,
            "verdict": self.verdict,
            "notes": list(self.notes),
        }


def compare_sampling_distribution(
    *,
    request_id: str,
    left_samples: Sequence[float],
    right_samples: Sequence[float],
    statistic: str = "median",
    pre_registered_bound: float,
    min_seeds: int = 4,
) -> SamplingComparison:
    """Compare a pre-registered statistic across seeds.

    ``left_samples``/``right_samples`` are per-seed values of the statistic (for
    example the median first-token logprob, or a task score).  With too few
    seeds the verdict is ``INCONCLUSIVE`` — never "equal".
    """
    if pre_registered_bound <= 0:
        raise ConfigError(
            "a distribution comparison needs a positive pre-registered bound; "
            "otherwise 'no difference detected' silently becomes 'equivalent'",
            details={"field": "pre_registered_bound"},
        )
    if statistic not in ("median", "mean", "p95", "match_rate"):
        raise ConfigError(f"unknown statistic {statistic!r}", details={"field": "statistic"})
    if len(left_samples) < min_seeds or len(right_samples) < min_seeds:
        return SamplingComparison(
            request_id=request_id,
            mode="multi_seed_distribution",
            seeds_left=len(left_samples),
            seeds_right=len(right_samples),
            statistic=statistic,
            left_summary={},
            right_summary={},
            statistic_delta=0.0,
            pre_registered_bound=pre_registered_bound,
            verdict="INCONCLUSIVE",
            notes=(
                f"only {len(left_samples)}/{len(right_samples)} seeds; "
                f"{min_seeds} are required before a distribution claim",
            ),
        )
    left_value = _statistic(left_samples, statistic)
    right_value = _statistic(right_samples, statistic)
    delta = left_value - right_value
    return SamplingComparison(
        request_id=request_id,
        mode="multi_seed_distribution",
        seeds_left=len(left_samples),
        seeds_right=len(right_samples),
        statistic=statistic,
        left_summary=distribution_summary(left_samples).as_dict(),
        right_summary=distribution_summary(right_samples).as_dict(),
        statistic_delta=delta,
        pre_registered_bound=pre_registered_bound,
        verdict="PASS" if abs(delta) <= pre_registered_bound else "FAIL",
    )


def _statistic(values: Sequence[float], statistic: str) -> float:
    if statistic == "median":
        return percentile(values, 0.5)
    if statistic == "mean":
        return sum(values) / len(values)
    if statistic == "p95":
        return percentile(values, 0.95)
    return sum(1 for value in values if value > 0) / len(values)


def assert_no_single_seed_claim(report: SamplingComparison) -> None:
    """Guard the listed anti-pattern: too few seeds cannot prove equality.

    Two distinct refusals: a single seed per side is *never* evidence, and an
    ``INCONCLUSIVE`` report (fewer seeds than the pre-registered minimum) cannot
    be quoted as equivalence either.
    """
    if report.mode != "multi_seed_distribution":
        return
    if min(report.seeds_left, report.seeds_right) < 2:
        raise CapabilityError(
            "refusing a single-seed sampling comparison: equality of one sampled "
            "sequence is not evidence that two runtimes implement the same "
            "distribution",
            details={"field": "seeds"},
        )
    if report.verdict == "INCONCLUSIVE":
        raise CapabilityError(
            f"refusing an equivalence claim from {report.seeds_left}/"
            f"{report.seeds_right} seeds: the comparison is inconclusive, not equal",
            details={"field": "seeds"},
        )


# ── streaming, stop and length boundaries ──────────────────────────────────


def compare_streaming(
    reference: GenerationResult,
    chunks: Sequence[StreamChunk],
) -> Dict[str, Any]:
    """Streaming must reproduce the non-streaming token sequence (§6)."""
    report = validate_stream(
        chunks, request_id=reference.request_id, expected_token_ids=reference.token_ids
    )
    concatenated = tuple(chunk.token_id for chunk in chunks)
    report["concatenated_equal"] = concatenated == tuple(reference.token_ids)
    report["final_flag_present"] = bool(chunks) and chunks[-1].final
    report["ok"] = bool(report["ok"] and report["concatenated_equal"])
    return report


@dataclass(frozen=True)
class BoundaryCase:
    """One stop/EOS/length boundary case (E07-01 step 10)."""

    case: str
    max_new_tokens: int
    min_new_tokens: int = 0
    eos_token_id: Optional[int] = None
    expected_finish_reason: str = "length"
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "eos_token_id": self.eos_token_id,
            "expected_finish_reason": self.expected_finish_reason,
            "description": self.description,
        }


def boundary_cases() -> Tuple[BoundaryCase, ...]:
    """The pre-registered boundary set; every backend runs the same cases."""
    return (
        BoundaryCase(
            case="eos_before_max",
            max_new_tokens=32,
            eos_token_id=2,
            expected_finish_reason="stop",
            description="EOS arrives early; length must not be reported",
        ),
        BoundaryCase(
            case="max_tokens_hit",
            max_new_tokens=8,
            expected_finish_reason="length",
            description="max_new_tokens reached without EOS",
        ),
        BoundaryCase(
            case="zero_token",
            max_new_tokens=1,
            min_new_tokens=0,
            expected_finish_reason="length",
            description="a one-token request: the smallest legal output",
        ),
        BoundaryCase(
            case="min_tokens_protects_eos",
            max_new_tokens=4,
            min_new_tokens=3,
            eos_token_id=2,
            expected_finish_reason="length",
            description="EOS before min_new_tokens must be suppressed",
        ),
    )


# ── parameter effect / unsupported negative matrices ───────────────────────


@dataclass(frozen=True)
class ParameterEffect:
    """Whether changing one supported field actually changed something."""

    field: str
    before_value: Any
    after_value: Any
    output_changed: bool
    telemetry_evidence: str = ""
    oracle_expectation: str = "output_changes"
    verdict: str = ""

    def __post_init__(self) -> None:
        if self.oracle_expectation not in ("output_changes", "output_identical"):
            raise ConfigError(
                "a parameter effect must state whether the oracle expects the output "
                "to change; otherwise every observation can be rationalised"
            )
        if not self.verdict:
            object.__setattr__(self, "verdict", self._compute_verdict())

    def _compute_verdict(self) -> str:
        if self.oracle_expectation == "output_changes":
            if self.output_changed:
                return "EFFECTIVE"
            if self.telemetry_evidence:
                return "EFFECTIVE_TELEMETRY_ONLY"
            return "INEFFECTIVE"
        if self.output_changed:
            return "UNEXPECTED_EFFECT"
        return "EFFECTIVE"

    @property
    def ok(self) -> bool:
        return self.verdict in ("EFFECTIVE", "EFFECTIVE_TELEMETRY_ONLY")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "before_value": self.before_value,
            "after_value": self.after_value,
            "output_changed": self.output_changed,
            "telemetry_evidence": self.telemetry_evidence,
            "oracle_expectation": self.oracle_expectation,
            "verdict": self.verdict,
            "ok": self.ok,
        }


def parameter_effect_matrix(
    capability: CapabilityReport, effects: Sequence[ParameterEffect]
) -> Dict[str, Any]:
    """Every claimable field must appear in the matrix (E07-01 step 11)."""
    claimable = [
        name
        for name, entry in capability.fields.items()
        if entry.state in CLAIMABLE_STATES
    ]
    covered = {effect.field for effect in effects}
    missing = sorted(set(claimable) - covered)
    unclaimed = sorted(covered - set(REQUEST_FIELDS))
    ineffective = [effect.field for effect in effects if not effect.ok]
    return {
        "backend": capability.backend_id,
        "ok": not missing and not unclaimed and not ineffective,
        "claimable_fields": sorted(claimable),
        "missing": missing,
        "unknown_fields": unclaimed,
        "ineffective": ineffective,
        "rows": [effect.as_dict() for effect in effects],
    }


@dataclass(frozen=True)
class UnsupportedObservation:
    """What a backend did when asked for an unsupported field."""

    field: str
    declared_state: str
    observed_behaviour: str
    reason_exposed: str = ""

    ALLOWED = (
        "REJECTED",
        "EXPLICIT_FALLBACK",
        "EMULATED",
        "CONSTRAINED",
        "SILENT_IGNORE",
    )

    def __post_init__(self) -> None:
        if self.observed_behaviour not in UnsupportedObservation.ALLOWED:
            raise ConfigError(
                f"unknown observed behaviour {self.observed_behaviour!r}",
                details={"allowed": list(UnsupportedObservation.ALLOWED)},
            )

    @property
    def ok(self) -> bool:
        if self.declared_state in CLAIMABLE_STATES:
            return True
        return self.observed_behaviour in (
            "REJECTED",
            "EXPLICIT_FALLBACK",
            "EMULATED",
            "CONSTRAINED",
        ) and bool(self.reason_exposed)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "declared_state": self.declared_state,
            "observed_behaviour": self.observed_behaviour,
            "reason_exposed": self.reason_exposed,
            "ok": self.ok,
        }


def unsupported_negative_matrix(
    capability: CapabilityReport, observations: Sequence[UnsupportedObservation]
) -> Dict[str, Any]:
    """Unsupported fields must be refused or fall back *with a reason* (§9 step 12)."""
    expect = {
        name
        for name, entry in capability.fields.items()
        if entry.state not in CLAIMABLE_STATES
    }
    covered = {observation.field for observation in observations}
    missing = sorted(expect - covered)
    failures = [observation.field for observation in observations if not observation.ok]
    return {
        "backend": capability.backend_id,
        "ok": not missing and not failures,
        "expected_unsupported": sorted(expect),
        "missing_observations": missing,
        "silent_or_unexplained": failures,
        "rows": [observation.as_dict() for observation in observations],
        "note": (
            "a field that is accepted and ignored is a failure even when the output "
            "looks right; the experiment cannot be repaired afterwards"
        ),
    }


# ── capability / parity matrix and error recovery ──────────────────────────


def capability_parity_matrix(
    requests: Sequence[RequestSpec],
    reports: Mapping[str, CapabilityReport],
) -> List[Dict[str, Any]]:
    """One row per backend × field, with NA and an explicit reason."""
    if len(reports) < 2:
        raise ConfigError(
            "a parity matrix needs at least two backends; a single column cannot "
            "show a difference",
            details={"field": "backends"},
        )
    require_prefix = any(request.prefix_cache_enabled for request in requests)
    rows: List[Dict[str, Any]] = []
    for backend_id, report in sorted(reports.items()):
        report = CapabilityReport(backend_id=backend_id, fields=dict(report.fields))
        for name in REQUEST_FIELDS:
            entry = report.fields.get(name)
            if entry is None:
                rows.append(
                    {
                        "backend": backend_id,
                        "field": name,
                        "state": "UNKNOWN",
                        "comparable": False,
                        "reason": "no probe result: field is UNKNOWN and ignored",
                    }
                )
                continue
            comparable = entry.state in CLAIMABLE_STATES
            if name == "prefix_cache" and not require_prefix:
                comparable = True
            rows.append(
                {
                    "backend": backend_id,
                    "field": name,
                    "state": entry.state,
                    "comparable": comparable,
                    "reason": entry.reason or entry.constraint,
                }
            )
    # Backends must agree on identity/quality before any performance comparison.
    return rows


@dataclass(frozen=True)
class ErrorRecoveryStep:
    """One step of an error-recovery sequence."""

    step: str
    operation: str
    expected: str
    observed: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "operation": self.operation,
            "expected": self.expected,
            "observed": self.observed,
        }


def error_recovery_sequence() -> Tuple[ErrorRecoveryStep, ...]:
    """load/generate failure followed by a healthy request (E07-01 step 17)."""
    return (
        ErrorRecoveryStep("1", "load(bad artifact)", "structured failure"),
        ErrorRecoveryStep("2", "close()", "resources released, no exception leak"),
        ErrorRecoveryStep("3", "load(good artifact)", "loads successfully"),
        ErrorRecoveryStep("4", "generate(valid request)", "healthy output"),
        ErrorRecoveryStep(
            "5", "compare with pre-failure golden", "identical token sequence"
        ),
    )


def error_recovery_report(steps: Sequence[ErrorRecoveryStep]) -> Dict[str, Any]:
    expected = error_recovery_sequence()
    executed = {step.step: step for step in steps}
    missing = [step.step for step in expected if step.step not in executed]
    return {
        "ok": not missing,
        "missing_steps": missing,
        "steps": [step.as_dict() for step in steps],
    }


__all__ = [
    "BoundaryCase",
    "ErrorRecoveryStep",
    "GreedyParityReport",
    "ParameterEffect",
    "SAMPLING_VERDICTS",
    "SamplingComparison",
    "UnsupportedObservation",
    "assert_no_single_seed_claim",
    "boundary_cases",
    "capability_parity_matrix",
    "compare_greedy",
    "compare_sampling_distribution",
    "compare_streaming",
    "error_recovery_report",
    "error_recovery_sequence",
    "parameter_effect_matrix",
    "unsupported_negative_matrix",
]
