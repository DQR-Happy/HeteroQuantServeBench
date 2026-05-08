"""Correctness and determinism gates for benchmark results.

These functions compare a measured run against a golden reference (or
against sibling repetitions) and produce a *machine-readable* verdict with
a precise first-mismatch location, so a numerical regression is diagnosable
without eyeballing token lists.

Golden reference contract (see ``benchmarks/schemas/golden_reference_schema.json``):

* ``generated_tokens`` — the exact token ID sequence produced by greedy
  decoding;
* ``first_token`` — the top-K logits of the first token for soft numerical
  comparison;
* a stable ``schema_version``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from hqsb.core.errors import SchemaError

# Relative tolerance for first-token logit comparison. Logits are FP16 on
# device; comparing with too-strict absolute tolerance would flag noise.
_DEFAULT_LOGIT_RTOL = 1e-3
_DEFAULT_LOGIT_ATOL = 1e-3


def hash_token_sequence(token_ids: Sequence[int]) -> str:
    """Return a stable SHA256 hex digest of a token ID sequence.

    The sequence is serialized as compact JSON so ``[1, 2]`` and ``[12]``
    hash differently.
    """
    payload = json.dumps(list(token_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ComparisonResult:
    """Outcome of comparing a measured sequence against a reference."""

    passed: bool
    method: str
    details: Dict[str, Any] = field(default_factory=dict)


def compare_token_sequence(
    actual: Sequence[int],
    expected: Sequence[int],
) -> ComparisonResult:
    """Compare a measured token sequence against a golden sequence.

    Returns a :class:`ComparisonResult` whose ``details`` records the first
    mismatch index and the differing values, if any.
    """
    if list(actual) == list(expected):
        return ComparisonResult(
            passed=True,
            method="token_sequence",
            details={
                "length": len(actual),
                "actual_sha256": hash_token_sequence(actual),
                "expected_sha256": hash_token_sequence(expected),
            },
        )

    first_mismatch = None
    for i, (a, e) in enumerate(zip(actual, expected)):
        if a != e:
            first_mismatch = i
            break
    if first_mismatch is None:
        # One sequence is a prefix of the other.
        first_mismatch = min(len(actual), len(expected))

    return ComparisonResult(
        passed=False,
        method="token_sequence",
        details={
            "first_mismatch_index": first_mismatch,
            "actual_length": len(actual),
            "expected_length": len(expected),
            "actual_token": (
                actual[first_mismatch]
                if first_mismatch < len(actual)
                else None
            ),
            "expected_token": (
                expected[first_mismatch]
                if first_mismatch < len(expected)
                else None
            ),
            "actual_sha256": hash_token_sequence(actual),
            "expected_sha256": hash_token_sequence(expected),
        },
    )


def compare_first_token_logits(
    actual: Sequence[float],
    expected: Sequence[float],
    *,
    rtol: float = _DEFAULT_LOGIT_RTOL,
    atol: float = _DEFAULT_LOGIT_ATOL,
) -> ComparisonResult:
    """Compare first-token logits within relative/absolute tolerance.

    The comparison follows the ``numpy.allclose`` convention: elementwise
    ``|a - e| <= atol + rtol * |e|``.
    """
    if len(actual) != len(expected):
        return ComparisonResult(
            passed=False,
            method="first_token_logits",
            details={
                "reason": "length_mismatch",
                "actual_length": len(actual),
                "expected_length": len(expected),
            },
        )

    max_abs = 0.0
    max_rel = 0.0
    for a, e in zip(actual, expected):
        diff = abs(a - e)
        max_abs = max(max_abs, diff)
        denom = atol + rtol * abs(e)
        if denom > 0:
            max_rel = max(max_rel, diff / denom)

    passed = all(
        abs(a - e) <= atol + rtol * abs(e) for a, e in zip(actual, expected)
    )
    return ComparisonResult(
        passed=passed,
        method="first_token_logits",
        details={
            "length": len(actual),
            "max_abs_error": max_abs,
            "max_relative_error": max_rel,
            "rtol": rtol,
            "atol": atol,
        },
    )


def verify_determinism(
    repetitions: Sequence[Sequence[int]],
) -> ComparisonResult:
    """Verify that all repetitions produced the same token sequence.

    Args:
        repetitions: One token ID sequence per repetition.

    Returns:
        A :class:`ComparisonResult` whose ``details`` records the number of
        distinct sequence hashes (1 == fully deterministic).
    """
    if not repetitions:
        return ComparisonResult(
            passed=False,
            method="determinism",
            details={"reason": "no_repetitions"},
        )

    hashes = [hash_token_sequence(seq) for seq in repetitions]
    distinct = set(hashes)
    return ComparisonResult(
        passed=len(distinct) == 1,
        method="determinism",
        details={
            "repetitions": len(repetitions),
            "distinct_sequence_hashes": len(distinct),
            "sequence_sha256": hashes[0],
        },
    )


def compare_against_golden(
    golden: Mapping[str, Any],
    *,
    generated_tokens: Optional[Sequence[int]] = None,
    first_token_logits: Optional[Sequence[float]] = None,
) -> ComparisonResult:
    """Compare a measured run against a golden reference document.

    At least one of ``generated_tokens`` or ``first_token_logits`` must be
    provided. The returned result aggregates both comparisons when both are
    given.

    Raises:
        SchemaError: If ``golden`` lacks a recognized schema version.
    """
    if "schema_version" not in golden:
        raise SchemaError("golden document is missing 'schema_version'")

    token_result: Optional[ComparisonResult] = None
    logit_result: Optional[ComparisonResult] = None

    if generated_tokens is not None:
        expected_tokens = golden.get("generated_tokens")
        if expected_tokens is None:
            raise SchemaError("golden document is missing 'generated_tokens'")
        token_result = compare_token_sequence(
            generated_tokens, list(expected_tokens)
        )

    if first_token_logits is not None:
        first_token = golden.get("first_token", {}) or {}
        expected_logits = first_token.get("logits")
        if expected_logits is None:
            raise SchemaError(
                "golden document is missing 'first_token.logits'"
            )
        logit_result = compare_first_token_logits(
            first_token_logits, list(expected_logits)
        )

    if token_result is None and logit_result is None:
        raise SchemaError(
            "must provide generated_tokens or first_token_logits to compare"
        )

    passed = all(
        r.passed for r in (token_result, logit_result) if r is not None
    )
    details: Dict[str, Any] = {}
    if token_result is not None:
        details["tokens"] = token_result.details
    if logit_result is not None:
        details["first_token_logits"] = logit_result.details

    return ComparisonResult(
        passed=passed,
        method="golden",
        details=details,
    )


__all__ = [
    "ComparisonResult",
    "compare_against_golden",
    "compare_first_token_logits",
    "compare_token_sequence",
    "hash_token_sequence",
    "verify_determinism",
]
