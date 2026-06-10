"""Runtime graph buckets and attention backends (E07-06).

E07-06 runs a 2×2 factorial — submission mode (eager / CUDA Graph) × attention
backend (runtime default / candidate) — inside the runtime, with the scheduler,
KV layout and precision held fixed.  This module provides:

* :class:`GraphSpec` / :class:`RouteDecision` — which bucket a runtime shape
  lands in, and what happens when it lands outside every bucket;
* :class:`ReplayRecord` / :func:`replay_distinctness` — the guard against the
  "replay the same input twice and call it correct" mistake;
* :class:`AttentionCandidate` / :func:`check_attention_support` — capability is
  decided per shape/dtype/layout/phase, never inferred from a config string;
* :func:`factorial_matrix` / :func:`two_factor_analysis` — main effects and the
  interaction term, with unsupported cells kept as ``UNSUPPORTED`` instead of
  being patched into a neighbouring path;
* :func:`phase_split_report` — prefill and decode are reported separately, and
  a blended "TPS" is refused;
* :func:`claim_status` — CUDA Graph is a claimable capability only with complete
  evidence; the default state is ``NOT_RUN``.

The runtime reuses the S06 CUDA-Graph *contract* through
:func:`inherit_from_e06_08` (a function-level import so this module stays
importable without the integration package being loaded).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import CapabilityError, ConfigError

# ── submission modes and buckets ───────────────────────────────────────────

EAGER = "eager"
CUDA_GRAPH = "cuda_graph"

SUBMISSION_MODES: Tuple[str, ...] = (EAGER, CUDA_GRAPH)

OUT_OF_BUCKET_POLICIES: Tuple[str, ...] = ("fallback_non_graph", "recapture")

PHASES: Tuple[str, ...] = ("prefill", "decode")


@dataclass(frozen=True)
class GraphBucketSpec:
    """One capture bucket; a runtime shape either lands in it or it does not."""

    name: str
    max_sequences: int
    max_tokens: int
    context_bucket: int = 0

    def __post_init__(self) -> None:
        if self.max_sequences <= 0 or self.max_tokens <= 0:
            raise ConfigError(
                "a graph bucket needs positive max_sequences and max_tokens",
                details={"field": self.name},
            )

    def contains(self, sequences: int, tokens: int) -> bool:
        return sequences <= self.max_sequences and tokens <= self.max_tokens

    @property
    def padding_tokens(self) -> int:
        """Worst-case padding this bucket can impose (must be reported)."""
        return self.max_tokens

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_sequences": self.max_sequences,
            "max_tokens": self.max_tokens,
            "context_bucket": self.context_bucket,
        }


@dataclass(frozen=True)
class GraphSpec:
    """Frozen graph-capture specification of the runtime."""

    buckets: Tuple[GraphBucketSpec, ...]
    out_of_bucket_policy: str = "fallback_non_graph"
    allow_recapture: bool = False
    max_graphs: int = 4
    capture_stream: str = "capture_origin_stream"
    includes_input_copy: bool = True
    includes_output_copy: bool = True
    claim_cuda_graph: bool = False

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ConfigError("a graph spec needs at least one bucket")
        if self.out_of_bucket_policy not in OUT_OF_BUCKET_POLICIES:
            raise ConfigError(
                f"unknown out_of_bucket_policy {self.out_of_bucket_policy!r}",
                details={"field": "out_of_bucket_policy"},
            )
        if len(self.buckets) > self.max_graphs:
            raise ConfigError(
                f"{len(self.buckets)} buckets exceed max_graphs={self.max_graphs}; "
                "the runtime would silently recapture",
                details={"field": "max_graphs"},
            )

    def resolve(self, *, sequences: int, tokens: int, phase: str) -> "RouteDecision":
        """Decide the bucket for one runtime shape."""
        if phase not in PHASES:
            raise ConfigError(f"unknown phase {phase!r}", details={"field": "phase"})
        if sequences <= 0 or tokens <= 0:
            raise ConfigError("sequences/tokens must be positive")
        eligible = [
            bucket
            for bucket in self.buckets
            if (bucket.context_bucket == 0 or bucket.context_bucket >= tokens)
            and bucket.contains(sequences, tokens)
        ]
        if not eligible:
            return RouteDecision(
                phase=phase,
                sequences=sequences,
                tokens=tokens,
                bucket="",
                status="OUT_OF_BUCKET",
                fallback=self.out_of_bucket_policy,
                reason=(
                    f"no bucket covers sequences={sequences}, tokens={tokens}; the "
                    f"runtime must use {self.out_of_bucket_policy} and record it"
                ),
                padding_tokens=0,
            )
        chosen = min(
            eligible, key=lambda bucket: (bucket.max_tokens, bucket.max_sequences)
        )
        padding = max(chosen.max_tokens - tokens, 0)
        return RouteDecision(
            phase=phase,
            sequences=sequences,
            tokens=tokens,
            bucket=chosen.name,
            status="HIT",
            fallback="",
            reason="",
            padding_tokens=padding,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "buckets": [bucket.as_dict() for bucket in self.buckets],
            "out_of_bucket_policy": self.out_of_bucket_policy,
            "allow_recapture": self.allow_recapture,
            "max_graphs": self.max_graphs,
            "capture_stream": self.capture_stream,
            "includes_input_copy": self.includes_input_copy,
            "includes_output_copy": self.includes_output_copy,
            "claim_cuda_graph": self.claim_cuda_graph,
        }


@dataclass(frozen=True)
class RouteDecision:
    """Bucket resolution result; ``fallback`` is empty only for a real hit."""

    phase: str
    sequences: int
    tokens: int
    bucket: str
    status: str
    fallback: str
    reason: str
    padding_tokens: int

    def __post_init__(self) -> None:
        if self.status not in ("HIT", "OUT_OF_BUCKET"):
            raise ConfigError(f"unknown route status {self.status!r}")
        if self.status == "HIT" and self.fallback:
            raise ConfigError("a bucket hit must not carry a fallback")
        if self.status == "OUT_OF_BUCKET" and not self.reason:
            raise ConfigError("an out-of-bucket decision needs a reason")

    @property
    def actual_mode(self) -> str:
        return CUDA_GRAPH if self.status == "HIT" else EAGER

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "sequences": self.sequences,
            "tokens": self.tokens,
            "bucket": self.bucket,
            "status": self.status,
            "fallback": self.fallback,
            "reason": self.reason,
            "padding_tokens": self.padding_tokens,
            "actual_mode": self.actual_mode,
        }


# ── replay records ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReplayRecord:
    """One replay with its input identity and phase timings (E07-06 §8)."""

    bucket: str
    inputs_hash: str
    replay_index: int
    warmup: bool = False
    capture_ms: float = 0.0
    first_replay_ms: float = 0.0
    steady_ms: float = 0.0
    cpu_gap_ms: float = 0.0
    pool_bytes: float = 0.0
    workspace_bytes: float = 0.0
    correct: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bucket": self.bucket,
            "inputs_hash": self.inputs_hash,
            "replay_index": self.replay_index,
            "warmup": self.warmup,
            "capture_ms": self.capture_ms,
            "first_replay_ms": self.first_replay_ms,
            "steady_ms": self.steady_ms,
            "cpu_gap_ms": self.cpu_gap_ms,
            "pool_bytes": self.pool_bytes,
            "workspace_bytes": self.workspace_bytes,
            "correct": self.correct,
        }


def replay_distinctness(records: Sequence[ReplayRecord]) -> Dict[str, Any]:
    """Replays of one bucket must receive *different* inputs to be meaningful."""
    by_bucket: Dict[str, List[ReplayRecord]] = {}
    for record in records:
        by_bucket.setdefault(record.bucket, []).append(record)
    problems: List[str] = []
    for bucket, bucket_records in sorted(by_bucket.items()):
        hashes = [record.inputs_hash for record in bucket_records]
        if len(bucket_records) > 1 and len(set(hashes)) == 1:
            problems.append(
                f"bucket {bucket}: all {len(bucket_records)} replays used the same "
                "inputs; a stale-buffer bug would be invisible"
            )
    return {
        "ok": not problems,
        "problems": problems,
        "buckets": sorted(by_bucket),
        "replays": len(records),
    }


def capture_cost_break_even(
    *, capture_ms: float, eager_ms: float, steady_ms: float
) -> Dict[str, Any]:
    """Replays needed to amortise capture; no point when steady >= eager."""
    if eager_ms <= 0 or steady_ms <= 0:
        raise ConfigError("timings must be positive")
    saving = eager_ms - steady_ms
    if saving <= 0:
        return {
            "break_even_replays": None,
            "saving_per_replay_ms": saving,
            "reason": "steady replay is not faster than eager; capture cannot amortise",
        }
    return {
        "break_even_replays": int(-(-capture_ms // saving)),
        "saving_per_replay_ms": saving,
        "reason": "",
    }


# ── attention backends ─────────────────────────────────────────────────────

MASK_KINDS: Tuple[str, ...] = ("causal", "none")


@dataclass(frozen=True)
class AttentionCandidate:
    """One attention implementation with the shapes it actually supports."""

    name: str
    provider: str
    version: str
    dtypes: Tuple[str, ...]
    kv_dtypes: Tuple[str, ...]
    head_dim: int
    supports_gqa: bool
    mask: str
    phases: Tuple[str, ...]
    paged_layout: bool
    max_context: int
    alignment: int = 1
    graph_capturable: bool = False
    workspace_bytes: float = 0.0
    fallback_target: str = "runtime_default"

    def __post_init__(self) -> None:
        if self.mask not in MASK_KINDS:
            raise ConfigError(
                f"unknown mask kind {self.mask!r}",
                details={"field": "mask", "allowed": list(MASK_KINDS)},
            )
        for phase in self.phases:
            if phase not in PHASES:
                raise ConfigError(f"unknown phase {phase!r}", details={"field": "phases"})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "version": self.version,
            "dtypes": list(self.dtypes),
            "kv_dtypes": list(self.kv_dtypes),
            "head_dim": self.head_dim,
            "supports_gqa": self.supports_gqa,
            "mask": self.mask,
            "phases": list(self.phases),
            "paged_layout": self.paged_layout,
            "max_context": self.max_context,
            "alignment": self.alignment,
            "graph_capturable": self.graph_capturable,
            "workspace_bytes": self.workspace_bytes,
            "fallback_target": self.fallback_target,
        }


@dataclass(frozen=True)
class AttentionRequest:
    """The shape/phase a caller wants attention for."""

    dtype: str
    kv_dtype: str
    head_dim: int
    query_heads: int
    kv_heads: int
    context: int
    phase: str
    paged: bool
    sequences: int = 1
    graph_capture: bool = False

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ConfigError(f"unknown phase {self.phase!r}", details={"field": "phase"})
        if self.kv_heads <= 0 or self.query_heads <= 0:
            raise ConfigError("head counts must be positive")
        if self.sequences <= 0:
            raise ConfigError("sequences must be positive")

    @property
    def is_gqa(self) -> bool:
        return self.kv_heads != self.query_heads


@dataclass(frozen=True)
class AttentionSupport:
    """Support decision with the failing field, never a bare boolean."""

    candidate: str
    supported: bool
    fallback: str
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.supported and not self.reason:
            raise ConfigError(
                "an unsupported attention choice must say why; 'not supported' alone "
                "cannot be planned around",
                details={"field": "reason"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate,
            "supported": self.supported,
            "fallback": self.fallback,
            "reason": self.reason,
        }


def check_attention_support(
    candidate: AttentionCandidate, request: AttentionRequest
) -> AttentionSupport:
    """Per-field support check; ``API configured ≠ kernel ran`` (E07-06 §5)."""
    reasons: List[str] = []
    if request.dtype not in candidate.dtypes:
        reasons.append(f"dtype {request.dtype} not in {list(candidate.dtypes)}")
    if request.kv_dtype not in candidate.kv_dtypes:
        reasons.append(f"kv dtype {request.kv_dtype} not in {list(candidate.kv_dtypes)}")
    if request.head_dim != candidate.head_dim:
        reasons.append(f"head_dim {request.head_dim} != {candidate.head_dim}")
    if request.is_gqa and not candidate.supports_gqa:
        reasons.append("GQA requested but the candidate does not support GQA")
    if request.phase not in candidate.phases:
        reasons.append(f"phase {request.phase} not in {list(candidate.phases)}")
    if request.paged and not candidate.paged_layout:
        reasons.append("paged KV requested but the candidate is contiguous-only")
    if request.context > candidate.max_context:
        reasons.append(
            f"context {request.context} exceeds max_context {candidate.max_context}"
        )
    if candidate.alignment > 1 and request.head_dim % candidate.alignment:
        reasons.append(
            f"head_dim {request.head_dim} is not a multiple of alignment "
            f"{candidate.alignment}"
        )
    if request.graph_capture and not candidate.graph_capturable:
        reasons.append("graph capture requested but the candidate is not capturable")
    if reasons:
        return AttentionSupport(
            candidate=candidate.name,
            supported=False,
            fallback=candidate.fallback_target,
            reason="; ".join(reasons),
        )
    return AttentionSupport(candidate=candidate.name, supported=True, fallback="")


def attention_matrix(
    candidates: Sequence[AttentionCandidate],
    requests: Sequence[AttentionRequest],
) -> List[Dict[str, Any]]:
    """Capability matrix over candidates × requests (E07-06 step 2)."""
    rows: List[Dict[str, Any]] = []
    for request in requests:
        for candidate in candidates:
            decision = check_attention_support(candidate, request)
            rows.append(
                {
                    "phase": request.phase,
                    "context": request.context,
                    "paged": request.paged,
                    **decision.as_dict(),
                }
            )
    return rows


# ── factorial analysis ─────────────────────────────────────────────────────

CELL_OK = "OK"
CELL_UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class FactorialCell:
    """One (submission, attention) combination."""

    submission: str
    attention: str
    status: str = CELL_OK
    metrics: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        if self.submission not in SUBMISSION_MODES:
            raise ConfigError(f"unknown submission mode {self.submission!r}")
        if self.status not in (CELL_OK, CELL_UNSUPPORTED):
            raise ConfigError(f"unknown cell status {self.status!r}")
        if self.status == CELL_UNSUPPORTED and not self.reason:
            raise ConfigError(
                "an unsupported factorial cell must carry its reason; otherwise it "
                "will be averaged into a neighbouring path (E07-06 §9)",
                details={"field": "reason"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "submission": self.submission,
            "attention": self.attention,
            "status": self.status,
            "metrics": dict(self.metrics),
            "reason": self.reason,
        }


def factorial_matrix(
    submission_modes: Sequence[str],
    attention_backends: Sequence[str],
    cells: Sequence[FactorialCell],
) -> Dict[str, Any]:
    """Build the 2×2 matrix, keeping unsupported combinations visible."""
    expected = [
        (submission, attention)
        for submission in submission_modes
        for attention in attention_backends
    ]
    provided = {(cell.submission, cell.attention): cell for cell in cells}
    missing = [pair for pair in expected if pair not in provided]
    unsupported = [
        cell.as_dict() for cell in cells if cell.status == CELL_UNSUPPORTED
    ]
    return {
        "ok": not missing,
        "expected_cells": [list(pair) for pair in expected],
        "missing_cells": [list(pair) for pair in missing],
        "unsupported_cells": unsupported,
        "cells": [cell.as_dict() for cell in cells],
        "note": (
            "an unsupported combination is reported as UNSUPPORTED and never "
            "replaced by the nearest working path"
        ),
    }


@dataclass(frozen=True)
class TwoFactorEffects:
    """Main effects of graph/attention plus their interaction."""

    metric: str
    graph_effect: float
    attention_effect: float
    interaction: float
    unsupported_cells: Tuple[str, ...]
    extrapolation_allowed: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "graph_effect": self.graph_effect,
            "attention_effect": self.attention_effect,
            "interaction": self.interaction,
            "unsupported_cells": list(self.unsupported_cells),
            "extrapolation_allowed": self.extrapolation_allowed,
        }


def two_factor_analysis(cells: Sequence[FactorialCell], *, metric: str) -> TwoFactorEffects:
    """Classic 2×2 decomposition, refused when a cell is unsupported.

    Sign convention (fixed here so it cannot be argued after the fact):

    * ``graph_effect = value(cuda_graph) - value(eager)`` averaged over the two
      attention choices — a **negative** value for a lower-is-better metric means
      the graph helped;
    * ``attention_effect = value(candidate) - value(default)`` averaged over the
      two submission modes, using the attention order the caller declared;
    * ``interaction`` is the difference of those differences.

    Effects are computed from the *declared* ordering of the cells rather than
    from an alphabetical sort, so swapping candidate names cannot flip a sign.
    """
    by_key = {
        (cell.submission, cell.attention): cell
        for cell in cells
        if cell.status == CELL_OK
    }
    unsupported = [
        f"{cell.submission}×{cell.attention}"
        for cell in cells
        if cell.status == CELL_UNSUPPORTED
    ]
    submissions = _ordered_unique(cell.submission for cell in cells)
    attentions = _ordered_unique(cell.attention for cell in cells)
    need = [
        (submission, attention) for submission in submissions for attention in attentions
    ]
    incomplete = len(submissions) != 2 or len(attentions) != 2 or any(
        pair not in by_key for pair in need
    )
    if incomplete:
        return TwoFactorEffects(
            metric=metric,
            graph_effect=0.0,
            attention_effect=0.0,
            interaction=0.0,
            unsupported_cells=tuple(unsupported),
            extrapolation_allowed=False,
        )
    (s0, s1) = submissions
    (a0, a1) = attentions

    def value(submission: str, attention: str) -> float:
        cell = by_key[(submission, attention)]
        if metric not in cell.metrics:
            raise ConfigError(
                f"metric {metric!r} is missing from cell {submission}×{attention}; "
                "a decomposition cannot invent the missing number",
                details={"field": metric},
            )
        return float(cell.metrics[metric])

    graph_effect = ((value(s1, a0) - value(s0, a0)) + (value(s1, a1) - value(s0, a1))) / 2
    attention_effect = (
        (value(s0, a1) - value(s0, a0)) + (value(s1, a1) - value(s1, a0))
    ) / 2
    interaction = (value(s1, a1) - value(s1, a0)) - (value(s0, a1) - value(s0, a0))
    return TwoFactorEffects(
        metric=metric,
        graph_effect=graph_effect,
        attention_effect=attention_effect,
        interaction=interaction,
        unsupported_cells=tuple(unsupported),
        extrapolation_allowed=True,
    )


def _ordered_unique(values: Sequence[str]) -> List[str]:
    """Deduplicate while preserving the caller's order (never sort a factor)."""
    seen: List[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


# ── phase separation and claim gate ────────────────────────────────────────


@dataclass(frozen=True)
class PhaseMetrics:
    """Prefill/decode metrics for one configuration."""

    configuration: str
    prefill_ms: float
    decode_ms: float
    prefill_tokens: int
    decode_tokens: int
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    memory_bytes: float = 0.0
    graph_bucket_hits: int = 0
    graph_bucket_misses: int = 0

    @property
    def prefill_tokens_per_s(self) -> float:
        if self.prefill_ms <= 0:
            return 0.0
        return self.prefill_tokens / (self.prefill_ms / 1000.0)

    @property
    def decode_tokens_per_s(self) -> float:
        if self.decode_ms <= 0:
            return 0.0
        return self.decode_tokens / (self.decode_ms / 1000.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "configuration": self.configuration,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "memory_bytes": self.memory_bytes,
            "graph_bucket_hits": self.graph_bucket_hits,
            "graph_bucket_misses": self.graph_bucket_misses,
            "prefill_tokens_per_s": self.prefill_tokens_per_s,
            "decode_tokens_per_s": self.decode_tokens_per_s,
        }


def phase_split_report(rows: Sequence[PhaseMetrics]) -> Dict[str, Any]:
    """Refuse to blend prefill and decode into one number (E07-06 §3)."""
    if not rows:
        raise ConfigError("a phase split report needs at least one row")
    blended: List[str] = []
    for row in rows:
        if row.prefill_ms and row.decode_ms:
            blended.append(row.configuration)
    return {
        "rows": [row.as_dict() for row in rows],
        "phases_reported_separately": True,
        "rows_with_both_phases": blended,
        "note": (
            "prefill and decode are never averaged into a single TPS; the caller "
            "must quote the phase it measured"
        ),
    }


class ClaimStatus:
    """Claim gate states; an unrun capability is never a PASS."""

    NOT_RUN = "NOT_RUN"
    NOT_CLAIMED = "NOT_CLAIMED"
    CLAIMED = "CLAIMED"


@dataclass(frozen=True)
class GraphClaimEvidence:
    """Evidence required before CUDA Graph may be claimed (E07-06 steps 3–20)."""

    eligibility_checked: bool = False
    capture_recorded: bool = False
    multi_input_replay_verified: bool = False
    correctness_gate_passed: bool = False
    fallback_recorded: bool = False
    memory_accounted: bool = False
    replay_distinct: bool = False

    @property
    def complete(self) -> bool:
        return all(
            (
                self.eligibility_checked,
                self.capture_recorded,
                self.multi_input_replay_verified,
                self.correctness_gate_passed,
                self.fallback_recorded,
                self.memory_accounted,
                self.replay_distinct,
            )
        )

    def missing(self) -> List[str]:
        names = (
            "eligibility_checked",
            "capture_recorded",
            "multi_input_replay_verified",
            "correctness_gate_passed",
            "fallback_recorded",
            "memory_accounted",
            "replay_distinct",
        )
        return [name for name in names if not getattr(self, name)]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "eligibility_checked": self.eligibility_checked,
            "capture_recorded": self.capture_recorded,
            "multi_input_replay_verified": self.multi_input_replay_verified,
            "correctness_gate_passed": self.correctness_gate_passed,
            "fallback_recorded": self.fallback_recorded,
            "memory_accounted": self.memory_accounted,
            "replay_distinct": self.replay_distinct,
            "missing": self.missing(),
        }


def claim_status(
    evidence: GraphClaimEvidence, *, executed: bool, spec: Optional[GraphSpec] = None
) -> Dict[str, Any]:
    """Decide whether the runtime may claim CUDA Graph (default: no).

    Order matters: execution, then evidence completeness (which names the missing
    pieces), then the frozen spec preference, then the claim.  A spec that
    declines to claim must not hide which evidence was missing.
    """
    if not executed:
        return {
            "status": ClaimStatus.NOT_RUN,
            "reason": "no graph experiment was executed; a capability that was not "
            "run cannot be claimed",
            "evidence": evidence.as_dict(),
        }
    if not evidence.complete:
        return {
            "status": ClaimStatus.NOT_CLAIMED,
            "reason": "evidence incomplete: " + ", ".join(evidence.missing()),
            "evidence": evidence.as_dict(),
        }
    if spec is not None and not spec.claim_cuda_graph:
        return {
            "status": ClaimStatus.NOT_CLAIMED,
            "reason": "the frozen graph spec declares claim_cuda_graph=false",
            "evidence": evidence.as_dict(),
        }
    return {
        "status": ClaimStatus.CLAIMED,
        "reason": "",
        "evidence": evidence.as_dict(),
    }


def inherit_from_e06_08() -> Dict[str, Any]:
    """Reuse the S06 CUDA-Graph contract instead of re-inventing it.

    Imported lazily so ``hqsb.runtime`` stays usable without the integration
    package being loaded; when the dependency direction forbids the import the
    caller gets a structured reason rather than an ImportError.
    """
    try:
        from hqsb.integration import cuda_graph as integration_cuda_graph
    except Exception as exc:  # noqa: BLE001 - a missing layer must not crash the caller
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "contract": {},
        }
    spec = integration_cuda_graph.GraphSpec
    output_contract = integration_cuda_graph.OutputContract
    return {
        "available": True,
        "reason": "",
        "contract": {
            "module": "hqsb.integration.cuda_graph",
            "graph_spec": spec.__name__,
            "output_contracts": list(output_contract.ALL),
            "claim_gate": "hqsb.integration.cuda_graph.claim_status",
            "note": (
                "S07 owns the runtime-side buckets and replay accounting; the "
                "capture/output-contract semantics come from S06"
            ),
        },
    }


def attention_capability_error(decision: AttentionSupport) -> CapabilityError:
    """Turn an unsupported attention decision into a structured capability error."""
    return CapabilityError(
        f"attention candidate {decision.candidate!r} is unsupported: {decision.reason}",
        details={
            "candidate": decision.candidate,
            "fallback": decision.fallback,
            "reason": decision.reason,
        },
    )


def phase_request_matrix(
    contexts: Sequence[int],
    sequence_counts: Sequence[int],
    phase: str,
) -> List[AttentionRequest]:
    """Shape matrix builder used by E07-06 step 11."""
    if phase not in PHASES:
        raise ConfigError(f"unknown phase {phase!r}")
    if not contexts or not sequence_counts:
        raise ConfigError("the shape matrix needs at least one context and one count")
    rows: List[AttentionRequest] = []
    for context in contexts:
        for sequences in sequence_counts:
            rows.append(
                AttentionRequest(
                    dtype="float16",
                    kv_dtype="float16",
                    head_dim=128,
                    query_heads=16,
                    kv_heads=8,
                    context=context,
                    phase=phase,
                    paged=True,
                    sequences=sequences,
                    graph_capture=phase == "decode" and sequences <= 8,
                )
            )
    return rows


__all__ = [
    "CELL_OK",
    "CELL_UNSUPPORTED",
    "CUDA_GRAPH",
    "ClaimStatus",
    "EAGER",
    "FactorialCell",
    "GraphBucketSpec",
    "GraphClaimEvidence",
    "GraphSpec",
    "MASK_KINDS",
    "OUT_OF_BUCKET_POLICIES",
    "PHASES",
    "PhaseMetrics",
    "ReplayRecord",
    "RouteDecision",
    "SUBMISSION_MODES",
    "TwoFactorEffects",
    "AttentionCandidate",
    "AttentionRequest",
    "AttentionSupport",
    "attention_capability_error",
    "attention_matrix",
    "capture_cost_break_even",
    "check_attention_support",
    "claim_status",
    "factorial_matrix",
    "inherit_from_e06_08",
    "phase_request_matrix",
    "phase_split_report",
    "replay_distinctness",
    "two_factor_analysis",
]
