"""Collective semantics, oracles, bandwidth formulas and cost models (E10-02).

Everything in this module is *definition* and *pure computation*:

* ``CollectiveSpec`` freezes what ``count``/payload mean per API, so a
  ReduceScatter total input count cannot be silently compared with an AllGather
  per-rank count (details E10-02 step 3);
* rank-coded generators and CPU oracles validate correctness independently of
  the backend (steps 4–6);
* the bandwidth registry keeps the NCCL-tests bus corrections as *named
  formulas with IDs*, and any derived HCCL column is explicitly renamed
  ``normalized_nccltests_bus_correction`` — it never impersonates a vendor
  number (details E10-02 §3.3);
* the α–β fit is piecewise, refuses to cross a detected crossover, and keeps
  residuals;
* :class:`LoopbackCollectiveExecutor` is a CPU test double for smoke self-checks
  only: ``claim_allowed()`` is always ``False``.

The module never runs a real collective.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.distributed.backend import CapabilityMatrix, CapabilityKey

#: The five required collectives plus AllToAllV (details E10-02 step 3).
OPS: Tuple[str, ...] = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "all_to_all_v",
    "broadcast",
)

#: The five collectives that must have numerical correctness + per-rank hashes.
REQUIRED_OP_ERROR_HASHED: Tuple[str, ...] = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast",
)

#: Element sizes for the dtypes the harness understands.
DTYPE_BYTES: Mapping[str, int] = {
    "int8": 1,
    "uint8": 1,
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "int32": 4,
    "fp64": 8,
    "int64": 8,
}

#: Payload definition per API (details E10-02 §3.3, README §8).
COUNT_SEMANTICS: Mapping[str, str] = {
    "all_reduce": "count = elements per rank; every rank sends and receives `count`",
    "all_gather": "count = elements contributed per rank; output has p*count, concatenated by group rank",
    "reduce_scatter": "total_input_count = p * per_rank_recv_count; each rank keeps its 1/p segment",
    "all_to_all": "count = per-rank per-destination elements; symmetric matrix",
    "all_to_all_v": "send_counts[src][dst] may be unbalanced; global send == global recv",
    "broadcast": "count = elements at the root; root is the only producer",
}

#: Formula IDs so a derived number can be recomputed later (details E10-02 step 20).
FORMULA_REGISTRY: Mapping[str, Mapping[str, Any]] = {
    "algbw": {
        "id": "hqsb.algbw.v1",
        "definition": "algbw = logical_payload_bytes / latency_s",
        "units": "GB/s (10^9 bytes/s)",
    },
    "bus_correction": {
        "id": "hqsb.bus_correction.v1",
        "definition": "nccl-tests correction factor per op and world size",
        "source": "NVIDIA nccl-tests PERFORMANCE.md",
    },
    "derived_hccl_column": {
        "id": "hqsb.normalized_nccltests_bus_correction.v1",
        "definition": (
            "derived column for cross-table analysis only; it must not be reported as an "
            "HCCL-native bus bandwidth or as physical link traffic"
        ),
    },
    "alpha_beta": {
        "id": "hqsb.alpha_beta.v1",
        "definition": "T ≈ alpha + bytes * beta, fitted per segment with residuals",
    },
}

#: Matrix size grid categories (details E10-02 §4.2).
SIZE_CATEGORIES: Tuple[str, ...] = (
    "latency",
    "transition",
    "bandwidth",
    "model_neighbourhood",
    "boundary",
)

#: Allowed case-status values for a grid row.
CASE_STATUSES: Tuple[str, ...] = ("PLANNED", "FILTERED", "DONE", "INVALID")

#: Order policies for a clean sweep (details E10-02 step 19).
ORDER_POLICIES: Tuple[str, ...] = ("randomized_block", "up_then_down", "down_then_up")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── specs ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CollectiveSpec:
    """One collective's frozen API semantics (details E10-02 step 3)."""

    op: str
    count_semantics: str
    inplace: bool = False
    root: Optional[int] = None
    reduce_op: str = "sum"
    dtype: str = "fp16"
    requires_exact_division: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ConfigError(f"unknown collective {self.op!r}", details={"field": "op"})
        if self.op == "broadcast" and self.root is None:
            raise ConfigError("broadcast needs a root", details={"field": "root"})
        if self.op != "broadcast" and self.root is not None:
            raise ConfigError(
                f"{self.op} does not take a root", details={"field": "root"}
            )
        if self.dtype not in DTYPE_BYTES:
            raise ConfigError(f"unknown dtype {self.dtype!r}", details={"field": "dtype"})

    def payload_bytes(self, *, numel_per_rank: int) -> int:
        """Logical payload of *one rank's* API call (the algbw numerator)."""
        return numel_per_rank * DTYPE_BYTES[self.dtype]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "count_semantics": self.count_semantics,
            "inplace": self.inplace,
            "root": self.root,
            "reduce_op": self.reduce_op,
            "dtype": self.dtype,
            "requires_exact_division": self.requires_exact_division,
            "notes": self.notes,
        }


def spec_for(op: str, *, dtype: str = "fp16", root: Optional[int] = None, inplace: bool = False) -> CollectiveSpec:
    if op not in OPS:
        raise ConfigError(f"unknown collective {op!r}; expected one of {OPS}", details={"field": "op"})
    return CollectiveSpec(
        op=op,
        count_semantics=COUNT_SEMANTICS[op],
        inplace=inplace,
        root=root if op == "broadcast" else None,
        dtype=dtype,
        requires_exact_division=op in ("reduce_scatter", "all_to_all"),
    )


# ── bandwidth formulas ─────────────────────────────────────────────────────


def bus_correction(op: str, world_size: int) -> float:
    """NCCL-tests bus correction (details README §9.3)."""
    if op not in OPS:
        raise ConfigError(f"unknown collective {op!r}", details={"field": "op"})
    if world_size < 1:
        raise ConfigError("world_size must be >= 1", details={"field": "world_size"})
    p = world_size
    if op == "all_reduce":
        return 2.0 * (p - 1) / p
    if op in ("reduce_scatter", "all_gather", "all_to_all", "all_to_all_v"):
        return (p - 1) / p
    return 1.0  # broadcast


def algbw_gbps(payload_bytes: int, latency_s: float) -> float:
    if latency_s <= 0:
        raise ConfigError("latency must be positive", details={"field": "latency_s"})
    return payload_bytes / latency_s / 1e9


def busbw_gbps(payload_bytes: int, latency_s: float, op: str, world_size: int) -> float:
    return algbw_gbps(payload_bytes, latency_s) * bus_correction(op, world_size)


def bandwidth_row(
    *,
    op: str,
    world_size: int,
    numel_per_rank: int,
    dtype: str,
    latency_us: float,
    algorithm: str,
    protocol: str,
    source: str = "hqsb",
) -> Dict[str, Any]:
    """Build one bandwidth row with the formula IDs attached (step 20)."""
    if dtype not in DTYPE_BYTES:
        raise ConfigError(f"unknown dtype {dtype!r}", details={"field": "dtype"})
    payload = numel_per_rank * DTYPE_BYTES[dtype]
    latency_s = latency_us * 1e-6
    algbw = algbw_gbps(payload, latency_s)
    corrected = algbw * bus_correction(op, world_size)
    row: Dict[str, Any] = {
        "op": op,
        "world_size": world_size,
        "numel_per_rank": numel_per_rank,
        "dtype": dtype,
        "payload_bytes": payload,
        "latency_us": latency_us,
        "algbw_GBps": algbw,
        "correction_factor": bus_correction(op, world_size),
        "algorithm": algorithm,
        "protocol": protocol,
        "source": source,
        "algbw_formula_id": FORMULA_REGISTRY["algbw"]["id"],
        "correction_formula_id": FORMULA_REGISTRY["bus_correction"]["id"],
    }
    if source == "hqsb":
        row["busbw_GBps"] = corrected
    else:
        # vendor-native fields are preserved; the NCCL correction is a *derived*
        # column and must be named as such (details E10-02 §3.3).
        row["normalized_nccltests_bus_correction"] = corrected
        row["busbw_GBps"] = ""
    return row


# ── generators / oracles ───────────────────────────────────────────────────


def rank_coded_pattern(rank: int, index: int, iteration: int = 0, seed: int = 0) -> int:
    """Deterministic, parseable element (details E10-02 step 4).

    The pattern encodes rank and index so a stale buffer or a wrong rank's
    buffer is detectable from the value itself.
    """
    if rank < 0 or index < 0:
        raise ConfigError("rank and index must be >= 0")
    return ((rank + 1) * 1_000_003 + (index + 1) * 7_919 + (iteration + 1) * 104_729 + seed) % 65_521


def vector_for_rank(rank: int, numel: int, *, iteration: int = 0, seed: int = 0) -> List[int]:
    return [rank_coded_pattern(rank, index, iteration, seed) for index in range(numel)]


def oracle_all_reduce(inputs: Sequence[Sequence[int]], *, reduce_op: str = "sum") -> List[int]:
    if not inputs:
        raise ConfigError("all_reduce needs at least one input")
    length = len(inputs[0])
    for item in inputs:
        if len(item) != length:
            raise ConfigError("all ranks must contribute the same element count")
    if reduce_op == "sum":
        return [sum(item[index] for item in inputs) for index in range(length)]
    if reduce_op == "max":
        return [max(item[index] for item in inputs) for index in range(length)]
    if reduce_op == "min":
        return [min(item[index] for item in inputs) for index in range(length)]
    raise ConfigError(f"unsupported reduce_op {reduce_op!r}", details={"field": "reduce_op"})


def oracle_all_gather(
    inputs: Sequence[Sequence[int]], *, ordered_ranks: Optional[Sequence[int]] = None
) -> List[int]:
    """Concatenation follows *group rank order*, not global rank order.

    ``ordered_ranks`` maps concatenation position → contributing rank index, so
    a permuted (non-natural) group order is validated explicitly: using global
    rank as the concatenation order is a classic silent error (E10-02 step 14).
    """
    if ordered_ranks is not None:
        if sorted(ordered_ranks) != list(range(len(inputs))):
            raise ConfigError(
                f"ordered_ranks {list(ordered_ranks)} is not a permutation of "
                f"[0, {len(inputs)})",
                details={"field": "ordered_ranks"},
            )
        reordered = [inputs[rank] for rank in ordered_ranks]
        return [value for chunk in reordered for value in chunk]
    return [value for chunk in inputs for value in chunk]


def oracle_reduce_scatter(inputs: Sequence[Sequence[int]], rank: int) -> List[int]:
    """Reduce element-wise, then keep only this rank's contiguous 1/p segment."""
    reduced = oracle_all_reduce(inputs)
    p = len(inputs)
    if len(reduced) % p:
        raise ConfigError(
            "reduce_scatter requires an element count divisible by world size",
            details={"field": "numel"},
        )
    segment = len(reduced) // p
    return reduced[rank * segment : (rank + 1) * segment]


def oracle_broadcast(inputs: Sequence[Sequence[int]], root: int) -> List[int]:
    if root < 0 or root >= len(inputs):
        raise ConfigError(f"root {root} outside [0, {len(inputs)})", details={"field": "root"})
    return list(inputs[root])


def oracle_all_to_all(chunks: Sequence[Sequence[Sequence[int]]]) -> List[List[int]]:
    """``chunks[src][dst]`` → ``received[dst][src]`` (permutation, no reduction)."""
    p = len(chunks)
    for row in chunks:
        if len(row) != p:
            raise ConfigError("all_to_all requires a p x p chunk matrix")
    return [[list(chunks[src][dst]) for src in range(p)] for dst in range(p)]


def oracle_all_to_all_v(
    sends: Mapping[Tuple[int, int], Sequence[int]],
    *,
    world_size: int,
) -> Dict[Tuple[int, int], List[int]]:
    """AllToAllV: verify offset/count semantics, return ``received[dst, src]``."""
    received: Dict[Tuple[int, int], List[int]] = {}
    send_total = 0
    recv_total = 0
    for (src, dst), payload in sends.items():
        if not (0 <= src < world_size and 0 <= dst < world_size):
            raise ConfigError(f"({src}, {dst}) outside world size {world_size}")
        received[(dst, src)] = list(payload)
        send_total += len(payload)
    recv_total = sum(len(item) for item in received.values())
    if send_total != recv_total:
        raise ConfigError(
            f"AllToAllV conservation violated: send {send_total} != recv {recv_total}"
        )
    return received


# ── comparison ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VectorComparison:
    ok: bool
    max_abs: float = 0.0
    mean_abs: float = 0.0
    rmse: float = 0.0
    max_rel: float = 0.0
    cosine: float = 1.0
    first_bad_index: int = -1
    nan_count: int = 0
    inf_count: int = 0
    rel_floor: float = 1e-6
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "rmse": self.rmse,
            "max_rel": self.max_rel,
            "cosine": self.cosine,
            "first_bad_index": self.first_bad_index,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "rel_floor": self.rel_floor,
            "reason": self.reason,
        }


def compare_vectors(
    expected: Sequence[float],
    actual: Sequence[float],
    *,
    tolerance: float,
    rel_floor: float = 1e-6,
) -> VectorComparison:
    """Pure-Python numerical comparison with first-error localisation."""
    if len(expected) != len(actual):
        return VectorComparison(
            ok=False,
            reason=f"length mismatch: {len(expected)} vs {len(actual)}",
            rel_floor=rel_floor,
        )
    if not expected:
        return VectorComparison(ok=True, rel_floor=rel_floor)
    nan_count = sum(1 for value in actual if isinstance(value, float) and math.isnan(value))
    inf_count = sum(1 for value in actual if isinstance(value, float) and math.isinf(value))
    diffs = [abs(float(e) - float(a)) for e, a in zip(expected, actual)]
    max_abs = max(diffs)
    mean_abs = sum(diffs) / len(diffs)
    rmse = math.sqrt(sum(diff * diff for diff in diffs) / len(diffs))
    rels = [
        diff / max(abs(float(e)), rel_floor)
        for diff, e in zip(diffs, expected)
    ]
    max_rel = max(rels)
    dot = sum(float(e) * float(a) for e, a in zip(expected, actual))
    norm_e = math.sqrt(sum(float(e) ** 2 for e in expected))
    norm_a = math.sqrt(sum(float(a) ** 2 for a in actual))
    cosine = dot / (norm_e * norm_a) if norm_e and norm_a else 1.0
    first_bad = -1
    for index, diff in enumerate(diffs):
        if diff > tolerance or rels[index] > max(tolerance, rel_floor):
            first_bad = index
            break
    if nan_count or inf_count:
        ok = False
    else:
        ok = max_abs <= tolerance and max_rel <= max(tolerance, rel_floor)
    return VectorComparison(
        ok=ok,
        max_abs=max_abs,
        mean_abs=mean_abs,
        rmse=rmse,
        max_rel=max_rel,
        cosine=cosine,
        first_bad_index=first_bad,
        nan_count=nan_count,
        inf_count=inf_count,
        rel_floor=rel_floor,
    )


# ── guard / buffer ownership ───────────────────────────────────────────────


@dataclass
class GuardedBuffer:
    """Input/output buffer with poison guard regions (details E10-02 step 6)."""

    name: str
    values: List[int]
    guard_slots: int = 4
    guard_value: int = -999_999

    def __post_init__(self) -> None:
        if self.guard_slots < 1:
            raise ConfigError("guard_slots must be >= 1")

    def build(self) -> List[int]:
        return [self.guard_value] * self.guard_slots + list(self.values) + [self.guard_value] * self.guard_slots

    def check_guard(self, buffer: Sequence[int]) -> Dict[str, Any]:
        expected = self.build()
        overruns: List[int] = []
        for index in list(range(self.guard_slots)) + list(
            range(len(expected) - self.guard_slots, len(expected))
        ):
            if index >= len(buffer):
                overruns.append(index)
                continue
            if buffer[index] != self.guard_value:
                overruns.append(index)
        return {"name": self.name, "overruns": overruns, "ok": not overruns}

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "numel": len(self.values), "guard_slots": self.guard_slots}


def alias_check(inplace: bool, input_name: str, output_name: str) -> Dict[str, Any]:
    """In-place/out-of-place alias rule (details E10-02 step 6)."""
    if inplace and input_name != output_name:
        raise ConfigError(
            "an in-place collective must use the same buffer for input and output",
            details={"field": "inplace"},
        )
    if not inplace and input_name == output_name:
        raise ConfigError(
            "an out-of-place collective must not alias its input and output",
            details={"field": "inplace"},
        )
    return {"inplace": inplace, "input": input_name, "output": output_name, "ok": True}


# ── grids ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MessageSizeCase:
    payload_bytes: int
    numel: int
    dtype: str
    category: str
    status: str = "PLANNED"
    filtered_reason: str = ""

    def __post_init__(self) -> None:
        if self.category not in SIZE_CATEGORIES:
            raise ConfigError(
                f"category must be one of {SIZE_CATEGORIES}", details={"field": "category"}
            )
        if self.status not in CASE_STATUSES:
            raise ConfigError(
                f"status must be one of {CASE_STATUSES}", details={"field": "status"}
            )
        if self.status == "FILTERED" and not self.filtered_reason:
            raise ConfigError(
                "a filtered case must record why (memory/capability)",
                details={"field": "filtered_reason"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "payload_bytes": self.payload_bytes,
            "numel": self.numel,
            "dtype": self.dtype,
            "category": self.category,
            "status": self.status,
            "filtered_reason": self.filtered_reason,
        }


def message_size_grid(
    *,
    dtype: str,
    min_bytes: int = 8,
    max_bytes: int = 1 << 27,
    factor: float = 2.0,
    model_payloads: Sequence[int] = (),
    include_non_power_of_two: bool = True,
    memory_limit_bytes: Optional[int] = None,
) -> List[MessageSizeCase]:
    """Log sweep + model payload neighbourhood + legal boundaries (step 7)."""
    if dtype not in DTYPE_BYTES:
        raise ConfigError(f"unknown dtype {dtype!r}", details={"field": "dtype"})
    if min_bytes <= 0 or max_bytes < min_bytes:
        raise ConfigError("invalid byte range", details={"field": "min_bytes"})
    element = DTYPE_BYTES[dtype]
    raw_sizes: List[int] = []
    size = min_bytes
    while size <= max_bytes:
        raw_sizes.append(int(size))
        size = max(size + 1, int(size * factor))
    for payload in model_payloads:
        for candidate in (payload // 2, payload, payload * 2):
            if min_bytes <= candidate <= max_bytes:
                raw_sizes.append(int(candidate))
    if include_non_power_of_two:
        for base in (min_bytes, max_bytes):
            neighbour = base + element * 3
            if min_bytes <= neighbour <= max_bytes:
                raw_sizes.append(neighbour)

    cases: List[MessageSizeCase] = []
    seen: set = set()
    for size in sorted(set(raw_sizes)):
        if size in seen:
            continue
        seen.add(size)
        numel = max(1, size // element)
        payload = numel * element
        if size <= 4 * 1024:
            category = "latency"
        elif size <= 1 << 20:
            category = "transition"
        elif size <= max_bytes:
            category = "bandwidth"
        else:  # pragma: no cover - guarded by the range
            category = "boundary"
        if any(
            payload in (value // 2, value, value * 2) for value in model_payloads
        ) and model_payloads:
            category = "model_neighbourhood"
        status = "PLANNED"
        reason = ""
        if memory_limit_bytes is not None and payload * 4 > memory_limit_bytes:
            status = "FILTERED"
            reason = f"buffers would need {payload * 4} bytes > limit {memory_limit_bytes}"
        cases.append(
            MessageSizeCase(
                payload_bytes=payload,
                numel=numel,
                dtype=dtype,
                category=category,
                status=status,
                filtered_reason=reason,
            )
        )
    return cases


@dataclass(frozen=True)
class RankCase:
    world_size: int
    node_count: int
    placement_hash: str
    topology_family: str
    available: bool = True
    missing_reason: str = ""

    def __post_init__(self) -> None:
        if self.world_size < 1 or self.node_count < 1:
            raise ConfigError("world_size/node_count must be >= 1")
        if not self.available and not self.missing_reason:
            raise ConfigError(
                "an unavailable rank case must record a resource reason (never zero-filled)",
                details={"field": "missing_reason"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "world_size": self.world_size,
            "node_count": self.node_count,
            "placement_hash": self.placement_hash,
            "topology_family": self.topology_family,
            "available": self.available,
            "missing_reason": self.missing_reason,
        }


def rank_grid(
    *,
    available_ranks: Sequence[int],
    node_families: Sequence[Tuple[int, int]],
    placement_hashes: Mapping[int, str],
    missing_reason: str = "resource unavailable",
) -> List[RankCase]:
    """Single-node and multi-node cells stay separate (step 8)."""
    cases: List[RankCase] = []
    for world_size in available_ranks:
        for node_count, ranks in node_families:
            available = world_size <= ranks and world_size in available_ranks
            family = f"nodes={node_count}"
            cases.append(
                RankCase(
                    world_size=world_size,
                    node_count=node_count,
                    placement_hash=placement_hashes.get(world_size, ""),
                    topology_family=family,
                    available=available,
                    missing_reason="" if available else f"{missing_reason}: {node_count} node(s) provide {ranks} ranks",
                )
            )
    return cases


def dtype_capability_probe(
    matrix: CapabilityMatrix,
    *,
    op: str,
    dtypes: Sequence[str],
    backend: str,
) -> List[Dict[str, Any]]:
    """Expose the capability matrix per (op, dtype) without guessing (step 9)."""
    rows: List[Dict[str, Any]] = []
    for dtype in dtypes:
        if dtype not in DTYPE_BYTES:
            raise ConfigError(f"unknown dtype {dtype!r}", details={"field": "dtype"})
        decision = matrix.decide(CapabilityKey(op=op, dtype=dtype, backend=backend))
        row = {"op": op, "dtype": dtype, "backend": backend}
        row.update(decision.as_dict())
        rows.append(row)
    return rows


# ── correctness sweep ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CorrectnessRow:
    """Per-rank correctness record (details E10-02 step 13 / README §17.2)."""

    op: str
    rank: int
    numel: int
    dtype: str
    input_hash: str
    output_hash: str
    comparison: VectorComparison
    collective_seq: int = 0
    algorithm: str = ""
    protocol: str = ""
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "op": self.op,
            "rank": self.rank,
            "numel": self.numel,
            "dtype": self.dtype,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "collective_seq": self.collective_seq,
            "algorithm": self.algorithm,
            "protocol": self.protocol,
            "error": self.error,
        }
        payload.update(self.comparison.as_dict())
        return payload


def correctness_row(
    *,
    spec: CollectiveSpec,
    rank: int,
    inputs: Sequence[Sequence[int]],
    actual: Sequence[float],
    tolerance: float,
    collective_seq: int = 0,
    algorithm: str = "",
    protocol: str = "",
) -> CorrectnessRow:
    """Build an AllReduce correctness row from a rank-coded input set."""
    expected = oracle_all_reduce(inputs, reduce_op=spec.reduce_op)
    comparison = compare_vectors([float(v) for v in expected], actual, tolerance=tolerance)
    return CorrectnessRow(
        op=spec.op,
        rank=rank,
        numel=len(expected),
        dtype=spec.dtype,
        input_hash=_sha256_text(repr([list(item) for item in inputs])),
        output_hash=_sha256_text(repr([float(v) for v in actual])),
        comparison=comparison,
        collective_seq=collective_seq,
        algorithm=algorithm,
        protocol=protocol,
    )


def should_stop_performance(rows: Sequence[CorrectnessRow]) -> Dict[str, Any]:
    """Correctness failure stops that communicator's performance runs (step 13)."""
    bad = [row for row in rows if not row.comparison.ok]
    return {
        "stop": bool(bad),
        "first_failure": bad[0].as_dict() if bad else None,
        "reason": (
            "correctness gate failed; performance numbers from this communicator are debug-only"
            if bad
            else ""
        ),
    }


# ── latency sweep ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LatencySweepPlan:
    """Order/clamp policy for a clean sweep (details E10-02 step 19)."""

    order_policy: str = "randomized_block"
    warmup_calls: int = 5
    repeats: int = 20
    independent_jobs: int = 3
    per_case_barrier: bool = False
    barrier_note: str = ""
    profiler_enabled: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.order_policy not in ORDER_POLICIES:
            raise ConfigError(
                f"order_policy must be one of {ORDER_POLICIES}", details={"field": "order_policy"}
            )
        if self.per_case_barrier and not self.barrier_note:
            raise ConfigError(
                "if barriers are used, say so explicitly — they pollute per-iteration latency",
                details={"field": "barrier_note"},
            )
        if self.profiler_enabled:
            raise ConfigError(
                "profiler runs are separate from the clean benchmark (details E10-02 step 25)",
                details={"field": "profiler_enabled"},
            )
        if self.independent_jobs < 3:
            raise ConfigError(
                "handbook §5.4 requires at least three independent process runs",
                details={"field": "independent_jobs"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "order_policy": self.order_policy,
            "warmup_calls": self.warmup_calls,
            "repeats": self.repeats,
            "independent_jobs": self.independent_jobs,
            "per_case_barrier": self.per_case_barrier,
            "barrier_note": self.barrier_note,
            "profiler_enabled": self.profiler_enabled,
            "seed": self.seed,
        }


# ── cost model ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AlphaBetaSegment:
    lower_bytes: int
    upper_bytes: int
    alpha_us: float
    beta_us_per_byte: float
    residual_rms_us: float
    points: int
    formula_id: str = "hqsb.alpha_beta.v1"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lower_bytes": self.lower_bytes,
            "upper_bytes": self.upper_bytes,
            "alpha_us": self.alpha_us,
            "beta_us_per_byte": self.beta_us_per_byte,
            "residual_rms_us": self.residual_rms_us,
            "points": self.points,
            "formula_id": self.formula_id,
        }

    def predict_us(self, payload_bytes: int) -> float:
        return self.alpha_us + self.beta_us_per_byte * payload_bytes


def _least_squares(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float]:
    n = len(points)
    if n < 2:
        raise ConfigError("a fit needs at least two points")
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    if denom == 0:
        raise ConfigError("all payload sizes are identical; cannot fit a slope")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in points]
    rms = math.sqrt(sum(r * r for r in residuals) / n)
    return intercept, slope, rms


def fit_alpha_beta_segments(
    points: Sequence[Mapping[str, float]],
    *,
    segment_boundaries: Sequence[int],
) -> List[AlphaBetaSegment]:
    """Piecewise α–β fit; a single line across a crossover is refused (§step 21)."""
    if len(segment_boundaries) < 2:
        raise ConfigError(
            "declare at least two segment boundaries (a single line across the algorithm "
            "crossover is not allowed)",
            details={"field": "segment_boundaries"},
        )
    boundaries = sorted(segment_boundaries)
    segments: List[AlphaBetaSegment] = []
    for index in range(len(boundaries) - 1):
        low, high = boundaries[index], boundaries[index + 1]
        subset = [
            (float(row["payload_bytes"]), float(row["latency_us"]))
            for row in points
            if low <= float(row["payload_bytes"]) <= high
        ]
        if len(subset) < 2:
            continue
        alpha, beta, rms = _least_squares(subset)
        segments.append(
            AlphaBetaSegment(
                lower_bytes=low,
                upper_bytes=high,
                alpha_us=alpha,
                beta_us_per_byte=beta,
                residual_rms_us=rms,
                points=len(subset),
            )
        )
    return segments


def detect_crossover(
    points: Sequence[Mapping[str, float]],
    *,
    min_slope_ratio: float = 1.5,
) -> List[Dict[str, Any]]:
    """Find algorithm/protocol slope changes from raw points (step 22)."""
    ordered = sorted(points, key=lambda row: float(row["payload_bytes"]))
    crossovers: List[Dict[str, Any]] = []
    slopes: List[Tuple[int, float]] = []
    for index in range(1, len(ordered)):
        previous = ordered[index - 1]
        current = ordered[index]
        dx = float(current["payload_bytes"]) - float(previous["payload_bytes"])
        if dx <= 0:
            continue
        dy = float(current["latency_us"]) - float(previous["latency_us"])
        slopes.append((index, dy / dx))
    for index in range(1, len(slopes)):
        previous_slope = slopes[index - 1][1]
        current_slope = slopes[index][1]
        if previous_slope <= 0:
            continue
        ratio = current_slope / previous_slope
        if ratio > min_slope_ratio or ratio < 1.0 / min_slope_ratio:
            row = ordered[slopes[index][0]]
            crossovers.append(
                {
                    "payload_bytes": row["payload_bytes"],
                    "before_slope_us_per_byte": previous_slope,
                    "after_slope_us_per_byte": current_slope,
                    "algorithm": row.get("algorithm", ""),
                    "protocol": row.get("protocol", ""),
                    "needs_denser_sampling": True,
                }
            )
    return crossovers


def rank_scaling_table(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Compare the same op/size across world sizes with the right correction (step 23)."""
    table: List[Dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        entry["correction_factor"] = bus_correction(
            str(row["op"]), int(row["world_size"])
        )
        if "algbw_GBps" in row:
            entry["busbw_GBps"] = float(row["algbw_GBps"]) * entry["correction_factor"]
        entry["note"] = "hierarchical algorithms: keep node_count in the row"
        table.append(entry)
    return sorted(table, key=lambda item: (str(item["op"]), int(item["world_size"]), int(item["payload_bytes"])))


def placement_effect_table(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Baseline vs alternative placement, per case (step 24)."""
    table: List[Dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        entry["attribution_hint"] = (
            "connect the difference to a concrete edge and data path; do not label it "
            "'NVLink vs PCIe'"
        )
        table.append(entry)
    return table


# ── model message projection ───────────────────────────────────────────────


@dataclass(frozen=True)
class MessageProjection:
    phase: str
    layer: Any
    op: str
    payload_bytes: int
    predicted_latency_us: float
    near_crossover: bool = False
    crossover_note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "layer": self.layer,
            "op": self.op,
            "payload_bytes": self.payload_bytes,
            "predicted_latency_us": self.predicted_latency_us,
            "near_crossover": self.near_crossover,
            "crossover_note": self.crossover_note,
        }


def project_model_messages(
    expected_events: Sequence[Mapping[str, Any]],
    segments: Sequence[AlphaBetaSegment],
    *,
    crossover_bytes: Sequence[int] = (),
    crossover_window: float = 0.25,
) -> List[MessageProjection]:
    """Project the E10-04 ledger payloads onto the measured curve (step 28)."""
    rows: List[MessageProjection] = []
    for event in expected_events:
        payload = int(event["payload_bytes"])
        chosen: Optional[AlphaBetaSegment] = None
        for segment in segments:
            if segment.lower_bytes <= payload <= segment.upper_bytes:
                chosen = segment
                break
        if chosen is None:
            raise ConfigError(
                f"payload {payload} is outside every fitted segment; extend the grid rather than "
                "extrapolating",
                details={"field": "payload_bytes"},
            )
        near = any(
            abs(payload - point) <= crossover_window * max(point, 1)
            for point in crossover_bytes
        )
        rows.append(
            MessageProjection(
                phase=str(event.get("phase", "")),
                layer=event.get("layer", ""),
                op=str(event.get("collective", "")),
                payload_bytes=payload,
                predicted_latency_us=chosen.predict_us(payload),
                near_crossover=near,
                crossover_note=(
                    "payload sits near an algorithm crossover: densify sampling before using it"
                    if near
                    else ""
                ),
            )
        )
    return rows


# ── crosscheck / stop rules ────────────────────────────────────────────────


def crosscheck_with_official_tools(
    hqsb_rows: Sequence[Mapping[str, Any]],
    official_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compare harness vs official tool, requiring every delta to be explained (step 27)."""
    official_index = {
        (row["op"], int(row["world_size"]), int(row["payload_bytes"])): row
        for row in official_rows
    }
    comparisons: List[Dict[str, Any]] = []
    unexplained: List[Dict[str, Any]] = []
    for row in hqsb_rows:
        key = (row["op"], int(row["world_size"]), int(row["payload_bytes"]))
        official = official_index.get(key)
        if official is None:
            comparisons.append({"key": key, "status": "NO_OFFICIAL_COUNTERPART", "explained": False})
            continue
        hqsb_latency = float(row["latency_us"])
        official_latency = float(official["latency_us"])
        ratio = hqsb_latency / official_latency if official_latency else float("inf")
        explanation = row.get("delta_explanation", "")
        entry = {
            "key": key,
            "hqsb_latency_us": hqsb_latency,
            "official_latency_us": official_latency,
            "ratio": ratio,
            "explained": bool(explanation),
            "delta_explanation": explanation,
        }
        comparisons.append(entry)
        if abs(ratio - 1.0) > 0.15 and not explanation:
            unexplained.append(entry)
    return {"comparisons": comparisons, "unexplained": unexplained, "ok": not unexplained}


#: Stop/expand rules of details E10-02 §11, as data the driver consults.
STOP_RULES: Mapping[str, str] = {
    "correctness_failure": "stop that communicator's performance runs immediately",
    "repeated_timeout_or_device_error": "stop the job, go to E10-03/E10-10, do not retry blindly",
    "variance_above_threshold": "add independent jobs, not more iterations in the same process",
    "near_memory_limit": "stop growing the message size; record the capacity",
    "bandwidth_plateau_not_seen": "extend size if resources allow, else report 'no plateau observed'",
    "crossover_between_two_coarse_points": "densify in that interval without changing the confirmation set",
    "profiler_overhead_too_high": "shrink the profile window; keep the clean result",
}


def apply_stop_rules(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Translate observed state into a stop/continue decision with a reason."""
    triggered = [key for key in STOP_RULES if state.get(key)]
    return {
        "stop": bool(triggered),
        "triggered": triggered,
        "reasons": {key: STOP_RULES[key] for key in triggered},
    }


# ── CPU loopback test double ───────────────────────────────────────────────


@dataclass
class LoopbackResult:
    op: str
    outputs_by_rank: Mapping[int, List[int]]
    simulated: bool = True
    collective_seq: int = 0
    error: str = ""

    def claim_allowed(self) -> bool:
        """A simulated collective never supports a performance/quality claim."""
        return False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "ranks": sorted(self.outputs_by_rank),
            "simulated": self.simulated,
            "claim_allowed": self.claim_allowed(),
            "collective_seq": self.collective_seq,
            "error": self.error,
            "note": "CPU loopback test double; not a device measurement",
        }


class LoopbackCollectiveExecutor:
    """Executes the collective *semantics* on the CPU, for smoke self-checks only."""

    def __init__(self, world_size: int) -> None:
        if world_size < 1:
            raise ConfigError("world_size must be >= 1", details={"field": "world_size"})
        self.world_size = world_size
        self._seq = 0

    def execute(
        self,
        spec: CollectiveSpec,
        *,
        inputs: Mapping[int, Sequence[int]],
        root: Optional[int] = None,
    ) -> LoopbackResult:
        self._seq += 1
        ordered = [list(inputs[rank]) for rank in sorted(inputs)]
        if len(ordered) != self.world_size:
            return LoopbackResult(
                op=spec.op,
                outputs_by_rank={},
                collective_seq=self._seq,
                error=f"expected {self.world_size} rank inputs, got {len(ordered)}",
            )
        try:
            if spec.op == "all_reduce":
                reduced = oracle_all_reduce(ordered, reduce_op=spec.reduce_op)
                outputs = {rank: list(reduced) for rank in range(self.world_size)}
            elif spec.op == "all_gather":
                gathered = oracle_all_gather(ordered)
                outputs = {rank: list(gathered) for rank in range(self.world_size)}
            elif spec.op == "reduce_scatter":
                outputs = {
                    rank: oracle_reduce_scatter(ordered, rank) for rank in range(self.world_size)
                }
            elif spec.op == "broadcast":
                effective_root = spec.root if root is None else root
                if effective_root is None:
                    raise ConfigError("broadcast needs a root")
                broadcasted = oracle_broadcast(ordered, int(effective_root))
                outputs = {rank: list(broadcasted) for rank in range(self.world_size)}
            else:
                return LoopbackResult(
                    op=spec.op,
                    outputs_by_rank={},
                    collective_seq=self._seq,
                    error=f"{spec.op} needs a chunk/count matrix; use oracle_all_to_all",
                )
        except ConfigError as exc:
            return LoopbackResult(
                op=spec.op, outputs_by_rank={}, collective_seq=self._seq, error=str(exc)
            )
        return LoopbackResult(op=spec.op, outputs_by_rank=outputs, collective_seq=self._seq)


#: Unified collective sample columns (details README §17.2).
COLLECTIVE_SAMPLE_FIELDS: Tuple[str, ...] = (
    "run_id",
    "rank",
    "group_id",
    "collective_seq",
    "op",
    "count",
    "dtype",
    "payload_bytes",
    "inplace",
    "algorithm",
    "protocol",
    "stream_id",
    "enqueue_us",
    "completion_us",
    "latency_us",
    "algbw_GBps",
    "busbw_GBps",
    "correct",
    "input_hash",
    "output_hash",
    "error",
)


__all__ = [
    "ALPHA_BETA_SEGMENT",
    "AlphaBetaSegment",
    "CASE_STATUSES",
    "COLLECTIVE_SAMPLE_FIELDS",
    "COUNT_SEMANTICS",
    "CorrectnessRow",
    "DTYPE_BYTES",
    "FORMULA_REGISTRY",
    "GuardedBuffer",
    "LatencySweepPlan",
    "LoopbackCollectiveExecutor",
    "LoopbackResult",
    "MessageProjection",
    "MessageSizeCase",
    "OPS",
    "ORDER_POLICIES",
    "REQUIRED_OP_ERROR_HASHED",
    "RankCase",
    "SIZE_CATEGORIES",
    "STOP_RULES",
    "VectorComparison",
    "alias_check",
    "algbw_gbps",
    "apply_stop_rules",
    "bandwidth_row",
    "bus_correction",
    "busbw_gbps",
    "compare_vectors",
    "correctness_row",
    "crosscheck_with_official_tools",
    "detect_crossover",
    "dtype_capability_probe",
    "fit_alpha_beta_segments",
    "message_size_grid",
    "oracle_all_gather",
    "oracle_all_reduce",
    "oracle_all_to_all",
    "oracle_all_to_all_v",
    "oracle_broadcast",
    "oracle_reduce_scatter",
    "placement_effect_table",
    "project_model_messages",
    "rank_coded_pattern",
    "rank_grid",
    "rank_scaling_table",
    "should_stop_performance",
    "spec_for",
    "vector_for_rank",
]

#: Backwards-compatible alias used by early drafts of the interface map.
ALPHA_BETA_SEGMENT = AlphaBetaSegment
