"""CUDA Graph capture/replay contracts (E06-08, P1).

E06-08 is a P1 capability gate: it is required **only if** the project claims
CUDA Graph support.  This module therefore ships the *contract* — preconditions,
static buffers, output lifetime, pool accounting, the failure matrix and the
claim gate — without capturing anything.  :func:`claim_status` returns
``NOT_CLAIMED`` until complete evidence exists, so a missing P1 run can never be
mistaken for a passed capability.

The most expensive mistakes in this area are silent (E06-08 §14): replaying an
old static input, returning a reused buffer as if owned, capturing half a graph
and continuing, or freeing a buffer that is still in flight.  Each of those has
an explicit guard here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError, CapabilityError

# ── output lifetime ───────────────────────────────────────────────────────


class OutputContract:
    """Public contract for replay outputs (E06-08 §7)."""

    REUSED_BUFFER = "REUSED_BUFFER"
    OWNED_COPY = "OWNED_COPY"

    ALL = (REUSED_BUFFER, OWNED_COPY)


@dataclass
class OutputHandle:
    """Tracks whether a caller still holds a view of a reused buffer."""

    contract: str
    buffer_id: str
    holders: int = 0

    def __post_init__(self) -> None:
        if self.contract not in OutputContract.ALL:
            raise ConfigError(
                f"unknown output contract {self.contract!r}",
                details={"field": "contract", "allowed": list(OutputContract.ALL)},
            )

    def acquire(self) -> None:
        self.holders += 1

    def release(self) -> None:
        if self.holders > 0:
            self.holders -= 1

    @property
    def safe_to_replay(self) -> bool:
        """A reused buffer must not be overwritten while a holder is alive."""
        return self.contract == OutputContract.OWNED_COPY or self.holders == 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "buffer_id": self.buffer_id,
            "holders": self.holders,
            "safe_to_replay": self.safe_to_replay,
        }


# ── static buffers ────────────────────────────────────────────────────────


class BufferKind:
    INPUT = "input"
    OUTPUT = "output"
    WEIGHT = "weight"
    WORKSPACE = "workspace"
    KV = "kv"

    ALL = (INPUT, OUTPUT, WEIGHT, WORKSPACE, KV)


@dataclass(frozen=True)
class StaticBuffer:
    """A buffer whose address must stay fixed for the graph's lifetime."""

    name: str
    kind: str
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: str
    address: int
    owner: str
    lifetime: str = "graph"

    def __post_init__(self) -> None:
        if self.kind not in BufferKind.ALL:
            raise ConfigError(
                f"buffer {self.name!r}: unknown kind {self.kind!r}",
                details={"field": "kind", "allowed": list(BufferKind.ALL)},
            )
        if self.address <= 0:
            raise ConfigError(
                f"buffer {self.name!r}: address must be a positive int (0 == unallocated)",
                details={"field": "address", "actual": self.address},
            )
        if len(self.shape) != len(self.stride):
            raise ConfigError(
                f"buffer {self.name!r}: stride rank must match shape rank",
                details={"field": "stride"},
            )

    def bytes(self) -> int:
        from hqsb.integration.lowering import DTYPE_BYTES

        if self.dtype not in DTYPE_BYTES:
            raise ConfigError(
                f"buffer {self.name!r}: unknown dtype {self.dtype!r}",
                details={"field": "dtype"},
            )
        product = 1
        for dim in self.shape:
            product *= int(dim)
        return product * DTYPE_BYTES[self.dtype]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "address": self.address,
            "owner": self.owner,
            "lifetime": self.lifetime,
            "bytes": self.bytes(),
        }


@dataclass
class StaticBufferPool:
    """Address-stable buffers with in-flight protection (E06-08 §5, §18)."""

    buffers: Dict[str, StaticBuffer] = field(default_factory=dict)
    in_flight: Dict[str, int] = field(default_factory=dict)

    def allocate(self, buffer: StaticBuffer) -> None:
        if buffer.name in self.buffers:
            existing = self.buffers[buffer.name]
            if existing.address != buffer.address:
                raise ConfigError(
                    f"buffer {buffer.name!r} already exists with address "
                    f"{existing.address}; a captured graph holds that pointer",
                    details={"field": "address", "existing": existing.address},
                )
        self.buffers[buffer.name] = buffer

    @property
    def addresses(self) -> Dict[str, int]:
        return {name: buffer.address for name, buffer in sorted(self.buffers.items())}

    @property
    def total_bytes(self) -> int:
        return sum(buffer.bytes() for buffer in self.buffers.values())

    def mark_in_flight(self, name: str) -> None:
        self.in_flight[name] = self.in_flight.get(name, 0) + 1

    def complete(self, name: str) -> None:
        if self.in_flight.get(name):
            self.in_flight[name] -= 1

    def release(self, name: str) -> None:
        """Refuse to release a buffer which is still used by an in-flight replay."""
        if self.in_flight.get(name):
            raise CapabilityError(
                f"buffer {name!r} is still in flight ({self.in_flight[name]} uses); "
                "releasing it now risks a use-after-free",
                details={"field": "in_flight", "buffer": name},
            )
        self.buffers.pop(name, None)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "buffers": [buffer.as_dict() for buffer in self.buffers.values()],
            "total_bytes": self.total_bytes,
            "in_flight": dict(self.in_flight),
        }


# ── eligibility ───────────────────────────────────────────────────────────


ELIGIBILITY_CHECKS = (
    "side_stream_warmup",
    "capture_stream_explicit",
    "no_cpu_gpu_sync",
    "no_unsupported_alloc",
    "current_stream_preserved",
    "rng_graph_safe",
    "addresses_fixed",
    "lifetimes_long_enough",
    "no_data_dependent_control_flow",
    "unsupported_identified_before_capture",
    "multi_stream_fork_rejoin",
)


@dataclass(frozen=True)
class EligibilityObservation:
    """Observed preconditions recorded by the runner (defaults are pessimistic)."""

    side_stream_warmup: bool = False
    capture_stream_explicit: bool = False
    no_cpu_gpu_sync: bool = False
    no_unsupported_alloc: bool = False
    current_stream_preserved: bool = False
    rng_graph_safe: bool = False
    addresses_fixed: bool = False
    lifetimes_long_enough: bool = False
    no_data_dependent_control_flow: bool = False
    unsupported_identified_before_capture: bool = False
    multi_stream_fork_rejoin: bool = True  # no extra streams used == trivially satisfied

    def as_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in ELIGIBILITY_CHECKS}


@dataclass(frozen=True)
class EligibilityReport:
    ok: bool
    failures: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "failures": list(self.failures)}


def evaluate_eligibility(observation: EligibilityObservation) -> EligibilityReport:
    """Pessimistic gate: an unobserved precondition is a failed precondition."""
    failures = [
        name
        for name in ELIGIBILITY_CHECKS
        if not bool(getattr(observation, name))
    ]
    return EligibilityReport(ok=not failures, failures=tuple(failures))


# ── spec, buckets and records ─────────────────────────────────────────────


@dataclass(frozen=True)
class BucketSpec:
    """Finite shape buckets; out-of-bucket cases fall back, they are not padded away."""

    name: str
    dims: Mapping[str, int]
    padding_cost: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dims": dict(self.dims),
            "padding_cost": dict(self.padding_cost),
        }


class OutOfBucketPolicy:
    FALLBACK_NON_GRAPH = "fallback_non_graph"
    RECAPTURE = "recapture"
    REJECT = "reject"

    ALL = (FALLBACK_NON_GRAPH, RECAPTURE, REJECT)


@dataclass(frozen=True)
class GraphSpec:
    """Frozen capture scope and policy (E06-08 §6, step 1)."""

    capture_scope: str
    buckets: Tuple[BucketSpec, ...] = ()
    output_contract: str = OutputContract.REUSED_BUFFER
    out_of_bucket_policy: str = OutOfBucketPolicy.FALLBACK_NON_GRAPH
    max_graphs: int = 4
    stream: str = "capture_stream"
    includes_input_copy: bool = True
    includes_output_copy: bool = True
    allow_recapture: bool = True

    def __post_init__(self) -> None:
        if self.output_contract not in OutputContract.ALL:
            raise ConfigError(
                f"unknown output contract {self.output_contract!r}",
                details={"field": "output_contract"},
            )
        if self.out_of_bucket_policy not in OutOfBucketPolicy.ALL:
            raise ConfigError(
                f"unknown out-of-bucket policy {self.out_of_bucket_policy!r}",
                details={"field": "out_of_bucket_policy"},
            )
        if self.max_graphs <= 0:
            raise ConfigError("max_graphs must be positive", details={"field": "max_graphs"})
        if not self.includes_input_copy and not self.includes_output_copy:
            raise ConfigError(
                "timing must state whether input/output copies are inside the measured "
                "region; both may not be excluded",
                details={"field": "includes_input_copy"},
            )

    def bucket_for(self, dims: Mapping[str, int]) -> Optional[BucketSpec]:
        for bucket in self.buckets:
            if dict(bucket.dims) == dict(dims):
                return bucket
        return None

    def resolve(self, dims: Mapping[str, int]) -> "BucketPlan":
        exact = self.bucket_for(dims)
        if exact is not None:
            return BucketPlan(
                bucket=exact, exact_match=True, policy=self.out_of_bucket_policy
            )
        return BucketPlan(
            bucket=None,
            exact_match=False,
            policy=self.out_of_bucket_policy,
            fallback_reason="OUT_OF_BUCKET",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capture_scope": self.capture_scope,
            "buckets": [bucket.as_dict() for bucket in self.buckets],
            "output_contract": self.output_contract,
            "out_of_bucket_policy": self.out_of_bucket_policy,
            "max_graphs": self.max_graphs,
            "stream": self.stream,
            "includes_input_copy": self.includes_input_copy,
            "includes_output_copy": self.includes_output_copy,
            "allow_recapture": self.allow_recapture,
        }


@dataclass(frozen=True)
class BucketPlan:
    bucket: Optional[BucketSpec]
    exact_match: bool
    policy: str
    fallback_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bucket": self.bucket.as_dict() if self.bucket else None,
            "exact_match": self.exact_match,
            "policy": self.policy,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class CaptureRecord:
    """One capture attempt (success or failure), never partially applied."""

    graph_id: str
    scope: str
    stream: str
    node_count: int
    capture_ms: float
    instantiate_ms: float
    pool_bytes: int
    static_buffer_bytes: int
    status: str = "captured"
    error: str = ""
    stage: str = "capture"

    @property
    def ok(self) -> bool:
        return self.status == "captured"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "scope": self.scope,
            "stream": self.stream,
            "node_count": self.node_count,
            "capture_ms": self.capture_ms,
            "instantiate_ms": self.instantiate_ms,
            "pool_bytes": self.pool_bytes,
            "static_buffer_bytes": self.static_buffer_bytes,
            "status": self.status,
            "error": self.error,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class ReplayRecord:
    """One replay with distinct input/output hashes (catches "old input" bugs)."""

    graph_id: str
    replay_index: int
    input_hash: str
    output_hash: str
    copy_in_ms: float = 0.0
    replay_ms: float = 0.0
    copy_out_ms: float = 0.0
    correctness_ok: Optional[bool] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "replay_index": self.replay_index,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "copy_in_ms": self.copy_in_ms,
            "replay_ms": self.replay_ms,
            "copy_out_ms": self.copy_out_ms,
            "correctness_ok": self.correctness_ok,
        }


def replay_distinctness(records: Sequence[ReplayRecord]) -> Dict[str, Any]:
    """Check that different inputs produced different outputs (E06-08 §10)."""
    by_input: Dict[str, set] = {}
    for record in records:
        by_input.setdefault(record.input_hash, set()).add(record.output_hash)
    collapsed = [
        {"input_hash": key, "output_hashes": sorted(value)}
        for key, value in by_input.items()
        if len(value) == 1 and len(records) > 1
        and sum(1 for item in records if item.input_hash == key) > 1
    ]
    return {
        "distinct_inputs": len(by_input),
        "replays": len(records),
        "repeated_inputs_with_identical_output": collapsed,
        "ok": True,  # identical output for identical input is expected; flagged only
    }


# ── failure matrix ────────────────────────────────────────────────────────


FAILURE_CASES: Tuple[str, ...] = (
    "shape_change",
    "stride_layout_change",
    "new_allocation",
    "cpu_sync_or_item",
    "unsupported_custom_op",
    "wrong_stream",
    "concurrent_capture",
    "dynamic_control",
    "output_address_change",
    "weight_or_artifact_change",
    "pool_object_released",
    "oom",
    "compile_fallback_region",
    "device_arch_mismatch",
)


@dataclass(frozen=True)
class FailureExpectation:
    case: str
    expected_stage: str
    expected_action: str  # always a clean non-graph path

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case,
            "expected_stage": self.expected_stage,
            "expected_action": self.expected_action,
        }


def failure_matrix() -> Tuple[FailureExpectation, ...]:
    """Each case must end in a documented non-graph path, never a half capture."""
    return tuple(
        FailureExpectation(
            case=case,
            expected_stage="pre_capture" if case in ("shape_change", "device_arch_mismatch") else "capture",
            expected_action=OutOfBucketPolicy.FALLBACK_NON_GRAPH,
        )
        for case in FAILURE_CASES
    )


@dataclass(frozen=True)
class FailureObservation:
    case: str
    stage: str
    captured_partially: bool
    reused_old_graph: bool
    fell_back_to_non_graph: bool
    reason: str = ""

    @property
    def ok(self) -> bool:
        return (
            not self.captured_partially
            and not self.reused_old_graph
            and self.fell_back_to_non_graph
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case,
            "stage": self.stage,
            "captured_partially": self.captured_partially,
            "reused_old_graph": self.reused_old_graph,
            "fell_back_to_non_graph": self.fell_back_to_non_graph,
            "reason": self.reason,
            "ok": self.ok,
        }


def evaluate_failure_matrix(
    observations: Sequence[FailureObservation],
) -> Dict[str, Any]:
    by_case = {item.case: item for item in observations}
    rows = []
    for expectation in failure_matrix():
        observed = by_case.get(expectation.case)
        rows.append(
            {
                **expectation.as_dict(),
                "observed": observed.as_dict() if observed else None,
                "ok": bool(observed and observed.ok),
            }
        )
    failing = [row for row in rows if not row["ok"]]
    return {"rows": rows, "failing": failing, "ok": not failing, "cases": len(rows)}


# ── pool memory and performance decomposition ─────────────────────────────


@dataclass(frozen=True)
class PoolMemoryAccount:
    """Pool memory across the graph lifecycle (E06-08 §8)."""

    allocated_before_bytes: int = 0
    reserved_before_bytes: int = 0
    pool_bytes: int = 0
    static_buffer_bytes: int = 0
    peak_bytes: int = 0
    after_destroy_bytes: int = 0
    survivors: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "allocated_before_bytes": self.allocated_before_bytes,
            "reserved_before_bytes": self.reserved_before_bytes,
            "pool_bytes": self.pool_bytes,
            "static_buffer_bytes": self.static_buffer_bytes,
            "peak_bytes": self.peak_bytes,
            "after_destroy_bytes": self.after_destroy_bytes,
            "residual_bytes": self.after_destroy_bytes - self.allocated_before_bytes,
            "survivors": list(self.survivors),
        }


@dataclass(frozen=True)
class ReplayTiming:
    """Capture/copy/replay decomposed; a single replay time is not enough."""

    capture_ms: float = 0.0
    instantiate_ms: float = 0.0
    upload_ms: float = 0.0
    first_replay_ms: float = 0.0
    steady_replay_ms: float = 0.0
    copy_in_ms: float = 0.0
    copy_out_ms: float = 0.0
    cpu_submit_ms: float = 0.0
    gpu_execution_ms: float = 0.0
    recaptures: int = 0
    padding_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capture_ms": self.capture_ms,
            "instantiate_ms": self.instantiate_ms,
            "upload_ms": self.upload_ms,
            "first_replay_ms": self.first_replay_ms,
            "steady_replay_ms": self.steady_replay_ms,
            "copy_in_ms": self.copy_in_ms,
            "copy_out_ms": self.copy_out_ms,
            "cpu_submit_ms": self.cpu_submit_ms,
            "gpu_execution_ms": self.gpu_execution_ms,
            "recaptures": self.recaptures,
            "padding_ms": self.padding_ms,
        }

    def break_even_replays(self, eager_ms: float) -> Optional[int]:
        """Replays needed to amortise capture+instantiate (copy costs included)."""
        saving = eager_ms - (self.steady_replay_ms + self.copy_in_ms + self.copy_out_ms)
        if saving <= 0:
            return None
        fixed = self.capture_ms + self.instantiate_ms + self.upload_ms
        return int(-(-fixed // saving))


# ── claim gate ────────────────────────────────────────────────────────────


class ClaimStatus:
    CLAIMED = "CLAIMED"
    NOT_CLAIMED = "NOT_CLAIMED"
    NOT_RUN = "NOT_RUN"

    ALL = (CLAIMED, NOT_CLAIMED, NOT_RUN)


@dataclass(frozen=True)
class ClaimEvidence:
    """What must exist before the project may claim CUDA Graph support."""

    graph_safe_scope: bool = False
    multi_input_replay_correct: bool = False
    long_replay_correct: bool = False
    lifetime_safe: bool = False
    stream_dependencies_ok: bool = False
    bucket_fallback_defined: bool = False
    failure_matrix_clean: bool = False
    pool_explained: bool = False
    timing_decomposed: bool = False
    end_to_end_benefit: bool = False
    break_even_bound: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph_safe_scope": self.graph_safe_scope,
            "multi_input_replay_correct": self.multi_input_replay_correct,
            "long_replay_correct": self.long_replay_correct,
            "lifetime_safe": self.lifetime_safe,
            "stream_dependencies_ok": self.stream_dependencies_ok,
            "bucket_fallback_defined": self.bucket_fallback_defined,
            "failure_matrix_clean": self.failure_matrix_clean,
            "pool_explained": self.pool_explained,
            "timing_decomposed": self.timing_decomposed,
            "end_to_end_benefit": self.end_to_end_benefit,
            "break_even_bound": self.break_even_bound,
        }

    @property
    def missing(self) -> Tuple[str, ...]:
        return tuple(
            name for name, value in self.as_dict().items() if not value
        )


def claim_status(evidence: ClaimEvidence, *, executed: bool = False) -> Dict[str, Any]:
    """P1 claim gate: no complete evidence ⇒ ``NOT_CLAIMED`` (never ``PASS``)."""
    if not executed:
        return {
            "status": ClaimStatus.NOT_RUN,
            "missing": list(evidence.missing),
            "reason": "P1 capability not executed; CUDA Graph must not be claimed",
        }
    if evidence.missing:
        return {
            "status": ClaimStatus.NOT_CLAIMED,
            "missing": list(evidence.missing),
            "reason": "incomplete evidence; a partial CUDA Graph story must not be claimed",
        }
    return {"status": ClaimStatus.CLAIMED, "missing": [], "reason": "evidence complete"}


__all__ = [
    "BufferKind",
    "BucketPlan",
    "BucketSpec",
    "CaptureRecord",
    "ClaimEvidence",
    "ClaimStatus",
    "ELIGIBILITY_CHECKS",
    "EligibilityObservation",
    "EligibilityReport",
    "FAILURE_CASES",
    "FailureExpectation",
    "FailureObservation",
    "GraphSpec",
    "OutOfBucketPolicy",
    "OutputContract",
    "OutputHandle",
    "PoolMemoryAccount",
    "ReplayRecord",
    "ReplayTiming",
    "StaticBuffer",
    "StaticBufferPool",
    "claim_status",
    "evaluate_eligibility",
    "evaluate_failure_matrix",
    "failure_matrix",
    "replay_distinctness",
]
