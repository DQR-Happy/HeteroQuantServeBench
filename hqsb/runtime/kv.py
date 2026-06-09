"""Paged KV geometry, block lifecycle, fragmentation and capacity (E07-03).

The module is deliberately split into three layers so a report cannot blur them:

1. **Geometry** (:class:`KVGeometry`, :func:`blocks_for`) — pure arithmetic from
   the model config and the block size;
2. **Lifecycle** (:class:`BlockPool`) — an explicit state machine with
   refcounts, owners and checkable invariants (free ⇒ no live owner, write
   before allocate, no access after free, evict never touches active/shared,
   cancel/finish releases or legally caches);
3. **Accounting** (:class:`MemoryReconciliation`) — predicted versus measured
   memory, where every byte must land in a *named* class.  The details README
   §12 forbids calling the residual "fragmentation": residues are reported as
   ``unexplained`` and fail the reconciliation.

Capacity/OOM handling is also explicit: :data:`OOM_ACTION_ORDER` is fixed and
bounded, so "retry forever" is not expressible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

# ── geometry ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KVGeometry:
    """KV-cache geometry taken from the *runtime schema*, never hand-copied.

    ``bytes_per_token = 2 × L × H_kv × D_h × element_bytes`` plus any per-token
    quantisation side data (scales/zeros), which must be counted separately
    (details README §2 and §10).
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    element_bytes: int
    kv_dtype: str = "float16"
    quant_scale_bytes_per_token: float = 0.0
    layout: str = "layer_major"
    group: str = "full_attention"

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_kv_heads", "head_dim", "element_bytes"):
            if getattr(self, name) <= 0:
                raise ConfigError(
                    f"KVGeometry.{name} must be positive",
                    details={"field": name},
                )
        if self.quant_scale_bytes_per_token < 0:
            raise ConfigError("quant side bytes cannot be negative")

    @property
    def payload_bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.element_bytes

    @property
    def bytes_per_token(self) -> float:
        """Payload plus quantisation side data actually stored per token."""
        return float(self.payload_bytes_per_token) + self.quant_scale_bytes_per_token

    def payload_bytes(self, tokens: int, block_size: int) -> int:
        return int(round(self.payload_bytes_per_token * self.slots_for(tokens, block_size)))

    def bytes_for(self, tokens: int, block_size: int) -> float:
        return self.bytes_per_token * self.slots_for(tokens, block_size)

    def slots_for(self, tokens: int, block_size: int) -> int:
        return blocks_for(tokens, block_size) * block_size

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "element_bytes": self.element_bytes,
            "kv_dtype": self.kv_dtype,
            "quant_scale_bytes_per_token": self.quant_scale_bytes_per_token,
            "layout": self.layout,
            "group": self.group,
            "payload_bytes_per_token": self.payload_bytes_per_token,
            "bytes_per_token": self.bytes_per_token,
        }


def blocks_for(tokens: int, block_size: int) -> int:
    """``ceil(tokens / P)``; a negative or zero request is refused, not padded."""
    if tokens < 0:
        raise ConfigError("token count must be non-negative")
    if block_size <= 0:
        raise ConfigError("block size must be positive")
    return math.ceil(tokens / block_size)


def block_boundary_points(block_size: int, blocks: int = 2) -> Tuple[int, ...]:
    """The P-1/P/P+1/2P-1/2P boundary set used by E07-03 step 5."""
    if block_size <= 1 or blocks < 1:
        raise ConfigError("block_size must be > 1 and blocks >= 1")
    points: List[int] = []
    for index in range(1, blocks + 1):
        base = index * block_size
        for point in (base - 1, base, base + 1):
            if point > 0 and point not in points:
                points.append(point)
    return tuple(points)


# ── fragmentation definitions (E07-03 §3) ──────────────────────────────────

#: Named byte/space classes.  Everything that is not payload must be labelled;
#: "everything else is fragmentation" is explicitly disallowed.
FRAGMENT_CLASSES: Tuple[str, ...] = (
    "internal_slot_waste",
    "cached_inactive_blocks",
    "block_table_metadata",
    "refcount_hash_metadata",
    "page_alignment",
    "allocator_reserve",
    "workspace",
    "graph_static_buffers",
    "non_kv_runtime",
)


@dataclass(frozen=True)
class KVRequestAccounting:
    """Per-request block reservation (internal fragmentation lives here)."""

    request_id: str
    live_tokens: int
    reserved_tokens: int
    block_size: int
    blocks: int
    slots: int

    def __post_init__(self) -> None:
        if self.reserved_tokens < self.live_tokens:
            raise ConfigError(
                "reserved_tokens cannot be smaller than live_tokens; a request "
                "cannot hold fewer slots than it has live tokens",
                details={"field": "reserved_tokens"},
            )
        if self.blocks != blocks_for(self.reserved_tokens, self.block_size):
            raise ConfigError(
                "blocks must equal ceil(reserved_tokens / block_size)",
                details={"field": "blocks"},
            )
        if self.slots != self.blocks * self.block_size:
            raise ConfigError("slots must equal blocks × block_size")

    @property
    def internal_fragment_tokens(self) -> int:
        """Allocated slots minus live useful tokens (prefix-tail waste)."""
        return self.slots - self.live_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "live_tokens": self.live_tokens,
            "reserved_tokens": self.reserved_tokens,
            "block_size": self.block_size,
            "blocks": self.blocks,
            "slots": self.slots,
            "internal_fragment_tokens": self.internal_fragment_tokens,
        }


@dataclass
class MemoryReconciliation:
    """Predicted vs measured KV memory with every byte named (E07-03 §6, §14)."""

    geometry: KVGeometry
    block_size: int
    active_tokens: int
    named: Dict[str, float] = field(default_factory=dict)
    measured_framework_allocated: float = 0.0
    measured_framework_reserved: float = 0.0
    measured_device_process: float = 0.0
    tolerance_bytes: float = 0.0

    def declare(self, klass: str, bytes_value: float) -> None:
        if klass not in FRAGMENT_CLASSES:
            raise ConfigError(
                f"unknown memory class {klass!r}; a byte that cannot be named must "
                "be reported as unexplained, not folded into a vague category",
                details={"field": klass, "allowed": list(FRAGMENT_CLASSES)},
            )
        self.named[klass] = float(bytes_value)

    @property
    def predicted_payload_bytes(self) -> float:
        return self.geometry.bytes_for(self.active_tokens, self.block_size)

    @property
    def predicted_total_bytes(self) -> float:
        return self.predicted_payload_bytes + sum(self.named.values())

    @property
    def residual_bytes(self) -> float:
        if self.measured_framework_reserved:
            return self.measured_framework_reserved - self.predicted_total_bytes
        return self.measured_device_process - self.predicted_total_bytes

    def explained(self) -> bool:
        """True only when the residual is within the declared tolerance."""
        if self.tolerance_bytes <= 0:
            raise ConfigError(
                "reconciliation needs a non-zero tolerance; otherwise every run is "
                "'explained' by construction",
                details={"field": "tolerance_bytes"},
            )
        return abs(self.residual_bytes) <= self.tolerance_bytes

    def as_rows(self) -> List[Dict[str, Any]]:
        rows = [
            {"class": "payload", "bytes": self.predicted_payload_bytes},
        ]
        rows.extend(
            {"class": name, "bytes": value} for name, value in sorted(self.named.items())
        )
        rows.append({"class": "unexplained_residual", "bytes": self.residual_bytes})
        return rows

    def as_dict(self) -> Dict[str, Any]:
        return {
            "block_size": self.block_size,
            "active_tokens": self.active_tokens,
            "predicted_payload_bytes": self.predicted_payload_bytes,
            "predicted_total_bytes": self.predicted_total_bytes,
            "measured_framework_allocated": self.measured_framework_allocated,
            "measured_framework_reserved": self.measured_framework_reserved,
            "measured_device_process": self.measured_device_process,
            "residual_bytes": self.residual_bytes,
            "tolerance_bytes": self.tolerance_bytes,
            "explained": self.explained(),
            "rows": self.as_rows(),
        }


# ── block lifecycle ────────────────────────────────────────────────────────


class BlockState:
    """Paged block states (E07-03 §4); extra states stay string-typed."""

    FREE = "FREE"
    ALLOCATED = "ALLOCATED"
    WRITTEN = "WRITTEN"
    ACTIVE = "ACTIVE"
    SHARED_CACHED = "SHARED_CACHED"
    EVICTABLE = "EVICTABLE"
    EVICTING = "EVICTING"
    COPY_ON_WRITE = "COPY_ON_WRITE"
    PREEMPTED = "PREEMPTED"


ALLOWED_BLOCK_TRANSITIONS: Mapping[str, Tuple[str, ...]] = {
    BlockState.FREE: (BlockState.ALLOCATED,),
    BlockState.ALLOCATED: (BlockState.WRITTEN, BlockState.FREE),
    BlockState.WRITTEN: (BlockState.ACTIVE, BlockState.FREE),
    BlockState.ACTIVE: (
        BlockState.SHARED_CACHED,
        BlockState.EVICTABLE,
        BlockState.COPY_ON_WRITE,
        BlockState.PREEMPTED,
        BlockState.FREE,
    ),
    BlockState.SHARED_CACHED: (BlockState.ACTIVE, BlockState.EVICTABLE, BlockState.FREE),
    BlockState.EVICTABLE: (BlockState.EVICTING, BlockState.ACTIVE),
    BlockState.EVICTING: (BlockState.FREE,),
    BlockState.COPY_ON_WRITE: (BlockState.ACTIVE, BlockState.FREE),
    BlockState.PREEMPTED: (BlockState.ACTIVE, BlockState.EVICTABLE, BlockState.FREE),
}


@dataclass
class BlockRecord:
    """One physical block with owner, refcount, token range and audit trail."""

    block_id: int
    state: str = BlockState.FREE
    owner_request: str = ""
    refcount: int = 0
    token_start: int = 0
    token_count: int = 0
    content_hash: str = ""
    device_address: Optional[int] = None
    history: List[str] = field(default_factory=list)
    readers: List[str] = field(default_factory=list)

    @property
    def has_live_owner(self) -> bool:
        return self.refcount > 0 and bool(self.owner_request)

    def transition(self, new_state: str, reason: str) -> None:
        allowed = ALLOWED_BLOCK_TRANSITIONS.get(self.state, ())
        if new_state not in allowed:
            raise ConfigError(
                f"illegal block transition {self.state} → {new_state} for block "
                f"{self.block_id} ({reason}); only {allowed} are permitted",
                details={"block_id": self.block_id, "state": self.state},
            )
        self.history.append(f"{self.state}->{new_state}:{reason}")
        self.state = new_state

    def as_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "state": self.state,
            "owner_request": self.owner_request,
            "refcount": self.refcount,
            "token_start": self.token_start,
            "token_count": self.token_count,
            "content_hash": self.content_hash,
            "device_address": self.device_address,
            "readers": list(self.readers),
            "history": list(self.history),
        }


class BlockPool:
    """A deterministic paged block pool with refcounts and audit events.

    This is the *contract* the runtime's KV manager is probed against: the same
    invariants are asserted by E07-03 step 2 and re-used by E07-05 (shared
    prefix refcounts) and E07-09 (failure recovery).
    """

    #: Event log of the pool (allocate/share/release/evict/…), declared here so
    #: the experiment interface map can address it; ``__init__`` assigns a fresh
    #: per-instance list.
    events: List[Dict[str, Any]]
    blocks: Dict[int, "BlockRecord"]

    def __init__(self, total_blocks: int, block_size: int) -> None:
        if total_blocks <= 0 or block_size <= 0:
            raise ConfigError("block pool needs positive total_blocks and block_size")
        self.block_size = block_size
        self.blocks: Dict[int, BlockRecord] = {
            index: BlockRecord(block_id=index) for index in range(total_blocks)
        }
        self.events: List[Dict[str, Any]] = []

    # ── events ────────────────────────────────────────────────────────────

    def _emit(self, kind: str, **payload: Any) -> Dict[str, Any]:
        event = {"kind": kind, **payload}
        self.events.append(event)
        return event

    def free_blocks(self) -> List[int]:
        return [
            block_id
            for block_id, record in self.blocks.items()
            if record.state == BlockState.FREE
        ]

    def used_blocks(self) -> List[int]:
        return [
            block_id
            for block_id, record in self.blocks.items()
            if record.state != BlockState.FREE
        ]

    def cached_blocks(self) -> List[int]:
        return [
            block_id
            for block_id, record in self.blocks.items()
            if record.state in (BlockState.SHARED_CACHED, BlockState.EVICTABLE)
        ]

    # ── lifecycle operations ──────────────────────────────────────────────

    def allocate(
        self, request_id: str, token_start: int, token_count: int, *, reuse: Sequence[int] = ()
    ) -> List[int]:
        """Allocate the blocks needed for ``token_count`` tokens, then write them."""
        if not request_id:
            raise ConfigError("allocation needs an owner request id")
        needed = blocks_for(token_count, self.block_size)
        chosen: List[int] = list(reuse)
        if len(chosen) > needed:
            raise ConfigError(
                "cannot reuse more blocks than the request needs",
                details={"field": "reuse"},
            )
        for block_id in chosen:
            record = self.blocks[block_id]
            record.transition(BlockState.ACTIVE, "reuse")
            record.readers.append(request_id)
            self._emit("reuse", request_id=request_id, block_id=block_id)
        for block_id in self.free_blocks():
            if len(chosen) == needed:
                break
            record = self.blocks[block_id]
            record.transition(BlockState.ALLOCATED, "allocate")
            record.owner_request = request_id
            record.refcount = 1
            if request_id not in record.readers:
                record.readers.append(request_id)
            record.token_start = token_start + len(chosen) * self.block_size
            record.token_count = min(
                self.block_size, max(token_count - len(chosen) * self.block_size, 0)
            )
            record.transition(BlockState.WRITTEN, "write")
            record.content_hash = f"blk{block_id}-{record.token_start}"
            chosen.append(block_id)
            self._emit("allocate", request_id=request_id, block_id=block_id)
        if len(chosen) < needed:
            raise ConfigError(
                f"block pool exhausted: needed {needed} blocks, only {len(chosen)} "
                "available (this is the predictable-admission case, not a leak)",
                details={"field": "total_blocks"},
            )
        for block_id in chosen[len(reuse):]:
            self.blocks[block_id].transition(BlockState.ACTIVE, "activate")
        return chosen

    def share(self, block_id: int, request_id: str) -> None:
        """Share a cached block with another request (refcount += 1)."""
        record = self._require_live(block_id)
        record.refcount += 1
        if request_id not in record.readers:
            record.readers.append(request_id)
        if record.state == BlockState.EVICTABLE:
            record.transition(BlockState.ACTIVE, "reader-attached")
        self._emit("share", request_id=request_id, block_id=block_id, refcount=record.refcount)

    def release(self, request_id: str, block_ids: Optional[Sequence[int]] = None) -> None:
        """Release a request's references (finish/cancel/timeout path)."""
        targets = list(block_ids) if block_ids is not None else self.blocks_for(request_id)
        for block_id in targets:
            record = self.blocks[block_id]
            if request_id in record.readers:
                record.readers.remove(request_id)
            record.refcount = max(record.refcount - 1, 0)
            self._emit("release", request_id=request_id, block_id=block_id)
            if record.refcount == 0:
                if record.owner_request == request_id:
                    record.owner_request = ""
                if record.state == BlockState.PREEMPTED:
                    record.transition(BlockState.EVICTABLE, "release-preempted")
                elif record.state in (BlockState.ACTIVE, BlockState.SHARED_CACHED):
                    record.transition(
                        BlockState.EVICTABLE if record.content_hash else BlockState.FREE,
                        "release",
                    )
                elif record.state == BlockState.EVICTABLE:
                    continue
                else:
                    record.transition(BlockState.FREE, "release")

    def blocks_for(self, request_id: str) -> List[int]:
        return [
            block_id
            for block_id, record in self.blocks.items()
            if request_id in record.readers or record.owner_request == request_id
        ]

    def mark_cached(self, block_id: int, request_id: str) -> None:
        """Finish with a legal retain: the block stays as shared cache."""
        record = self.blocks[block_id]
        if request_id not in record.readers:
            raise ConfigError(
                "only a reader can hand a block to the prefix cache",
                details={"block_id": block_id},
            )
        self._emit("mark_cached", request_id=request_id, block_id=block_id)
        record.transition(BlockState.SHARED_CACHED, "prefix-cache-retain")

    def evict(self, block_id: int) -> None:
        """Evict an eligible block; active/shared readers are never touched."""
        record = self.blocks[block_id]
        if record.state == BlockState.ACTIVE or record.refcount > 0:
            raise ConfigError(
                f"refusing to evict block {block_id}: state={record.state} "
                f"refcount={record.refcount} (active/shared blocks must never be "
                "evicted)",
                details={"block_id": block_id},
            )
        if record.state != BlockState.EVICTABLE:
            raise ConfigError(
                f"block {block_id} is not evictable (state={record.state})",
                details={"block_id": block_id},
            )
        record.transition(BlockState.EVICTING, "evict")
        record.content_hash = ""
        record.owner_request = ""
        record.transition(BlockState.FREE, "evict-done")
        self._emit("evict", block_id=block_id)

    # ── invariants ────────────────────────────────────────────────────────

    def _require_live(self, block_id: int) -> BlockRecord:
        record = self.blocks.get(block_id)
        if record is None:
            raise ConfigError(f"unknown block id {block_id}", details={"block_id": block_id})
        if record.state == BlockState.FREE:
            raise ConfigError(
                f"access to freed block {block_id}: use-after-free is a lifecycle "
                "bug, not a cache miss",
                details={"block_id": block_id},
            )
        return record

    def invariant_report(self) -> Dict[str, Any]:
        """Check the E07-03 §4 invariants over the current pool state."""
        problems: List[str] = []
        for block_id, record in self.blocks.items():
            if record.state == BlockState.FREE and record.has_live_owner:
                problems.append(f"block {block_id}: free but has a live owner")
            if record.state == BlockState.FREE and record.readers:
                problems.append(f"block {block_id}: free but still has readers")
            if record.state in (BlockState.ALLOCATED, BlockState.WRITTEN):
                problems.append(
                    f"block {block_id}: left in transient state {record.state}"
                )
            if record.refcount < 0:
                problems.append(f"block {block_id}: negative refcount")
            if record.state in (BlockState.ACTIVE, BlockState.SHARED_CACHED) and not (
                record.owner_request or record.readers or record.content_hash
            ):
                problems.append(f"block {block_id}: live state without owner/hash")
            if record.state != BlockState.FREE and record.refcount != len(record.readers):
                problems.append(
                    f"block {block_id}: refcount {record.refcount} != readers "
                    f"{len(record.readers)}"
                )
        return {
            "ok": not problems,
            "problems": problems,
            "free": len(self.free_blocks()),
            "used": len(self.used_blocks()),
            "cached": len(self.cached_blocks()),
            "blocks": len(self.blocks),
        }

    # ── fault fixtures (E07-03 step 17) ───────────────────────────────────

    def inject_double_free(self, request_id: str) -> str:
        """Release twice; the second release must be diagnosable, not silent."""
        self.release(request_id)
        before = self.invariant_report()
        self.release(request_id)
        after = self.invariant_report()
        return (
            "DOUBLE_FREE_ABSORBED"
            if before["ok"] and after["ok"]
            else "DOUBLE_FREE_CORRUPTED_POOL"
        )

    def inject_refcount_error(self, block_id: int) -> None:
        record = self.blocks[block_id]
        record.refcount += 1  # deliberate corruption for the fault fixture
        self._emit("inject_refcount_error", block_id=block_id)

    def inject_stale_id_access(self, block_id: int) -> str:
        """Access a block id after it was freed (E07-03 step 17).

        Returns ``STALE_ID_REJECTED`` when the pool correctly refuses the access,
        ``BLOCK_STILL_LIVE`` when the id was never freed (the fixture is not
        exercising a stale id), and ``STALE_ID_ACCEPTED`` when a freed block is
        reached — the only outcome that counts as a defect.
        """
        record = self.blocks.get(block_id)
        if record is None:
            return "STALE_ID_REJECTED"
        if record.state != BlockState.FREE:
            return "BLOCK_STILL_LIVE"
        try:
            self._require_live(block_id)
        except ConfigError:
            return "STALE_ID_REJECTED"
        return "STALE_ID_ACCEPTED"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "block_size": self.block_size,
            "blocks": [record.as_dict() for record in self.blocks.values()],
            "events": list(self.events),
            "invariants": self.invariant_report(),
        }


# ── eviction policies ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvictionCandidate:
    block_id: int
    last_use_iteration: int
    reuse_count: int
    bytes_value: float
    pinned: bool = False


def select_evictions(
    candidates: Iterable[EvictionCandidate], policy: str, limit: int
) -> List[EvictionCandidate]:
    """Select evictions under a named policy (bounded comparison, E07-05 §7)."""
    pool = [candidate for candidate in candidates if not candidate.pinned]
    if policy == "lru":
        ordered = sorted(pool, key=lambda item: (item.last_use_iteration, item.block_id))
    elif policy == "lru_reuse_aware":
        ordered = sorted(
            pool,
            key=lambda item: (item.reuse_count, item.last_use_iteration, item.block_id),
        )
    else:
        raise ConfigError(
            f"unknown eviction policy {policy!r}; S07 compares a bounded set of "
            "policies, it does not run an unbounded algorithm contest",
            details={"field": "policy"},
        )
    return ordered[: max(limit, 0)]


# ── capacity and OOM policy ────────────────────────────────────────────────


@dataclass(frozen=True)
class KVCapacityModel:
    """Theoretical capacity, safe admission and the OOM point (E07-03 §8)."""

    geometry: KVGeometry
    block_size: int
    blocks_total: int
    non_kv_resident_bytes: float
    tolerance_bytes: float = 0.0
    watermark: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 < self.watermark <= 1.0:
            raise ConfigError("watermark must be in (0, 1]")

    @property
    def payload_capacity_bytes(self) -> float:
        return self.blocks_total * self.block_size * self.geometry.bytes_per_token

    def theoretical_max_tokens(self) -> int:
        return self.blocks_total * self.block_size

    def safe_admission_tokens(self) -> int:
        return int(self.theoretical_max_tokens() * self.watermark)

    def max_concurrent_requests(self, mean_tokens: int) -> int:
        if mean_tokens <= 0:
            raise ConfigError("mean_tokens must be positive")
        return self.safe_admission_tokens() // mean_tokens

    def predicted_blocks(self, tokens: int) -> int:
        return blocks_for(tokens, self.block_size)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "block_size": self.block_size,
            "blocks_total": self.blocks_total,
            "payload_capacity_bytes": self.payload_capacity_bytes,
            "theoretical_max_tokens": self.theoretical_max_tokens(),
            "safe_admission_tokens": self.safe_admission_tokens(),
            "watermark": self.watermark,
            "non_kv_resident_bytes": self.non_kv_resident_bytes,
        }


@dataclass(frozen=True)
class CapacityBoundary:
    """Bisected capacity/OOM boundary under a deterministic probe."""

    safe_tokens: int
    oom_tokens: int
    probe_points: Tuple[int, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "safe_tokens": self.safe_tokens,
            "oom_tokens": self.oom_tokens,
            "probe_points": list(self.probe_points),
            "gap_tokens": self.oom_tokens - self.safe_tokens,
        }


def find_capacity_boundary(
    probe: Callable[[int], bool], *, low: int = 0, high: int
) -> CapacityBoundary:
    """Bisect ``probe`` (True = survived) to the last safe / first OOM point.

    A fresh process per probe point is the caller's responsibility (E07-03 §8);
    this helper only guarantees a deterministic, reproducible boundary.
    """
    if high <= low:
        raise ConfigError("capacity search needs high > low")
    if not probe(low):
        raise ConfigError(
            "the capacity probe already fails at the lower bound; the run is not a "
            "clean measurement (state was polluted by an earlier OOM)",
            details={"field": "low"},
        )
    points: List[int] = []
    safe, oom = low, high
    if probe(high):
        points.append(high)
        raise ConfigError(
            "capacity probe survived the upper bound; widen the search range "
            "instead of reporting an unbounded capacity",
            details={"field": "high"},
        )
    points.append(high)
    while oom - safe > 1:
        middle = (safe + oom) // 2
        points.append(middle)
        if probe(middle):
            safe = middle
        else:
            oom = middle
    return CapacityBoundary(safe_tokens=safe, oom_tokens=oom, probe_points=tuple(points))


#: OOM classes (E07-09 §5).  The action order below is fixed and bounded.
OOM_KINDS: Tuple[str, ...] = (
    "predictable_admission_reject",
    "preallocation_failure",
    "execution_oom",
    "fragmentation_induced",
    "prefix_cache_pressure",
    "graph_pool_oom",
    "actual_leak",
)

#: The vocabulary of OOM actions (ordering used by the frozen config).
OOM_ACTION_ORDER: Tuple[str, ...] = (
    "reject",
    "evict_eligible_cache",
    "preempt",
    "reduce_batch",
    "fallback",
    "fail",
)

#: Per-kind ladders.  Each ladder is finite and **ends in a terminal action**
#: (``reject`` or ``fail``): a policy that can run out of attempts without
#: deciding is exactly the unbounded retry the protocol forbids.
OOM_LADDERS: Mapping[str, Tuple[str, ...]] = {
    "predictable_admission_reject": ("reject",),
    "actual_leak": ("fail",),
    "graph_pool_oom": ("fallback", "fail"),
    "prefix_cache_pressure": ("evict_eligible_cache", "fail"),
    "fragmentation_induced": ("evict_eligible_cache", "preempt", "fail"),
    "preallocation_failure": ("reduce_batch", "preempt", "fail"),
    "execution_oom": ("reduce_batch", "evict_eligible_cache", "preempt", "fail"),
}

#: How many actions may be attempted before the request must fail.  Bounded on
#: purpose: an unbounded retry loop is a design defect, not resilience.
MAX_OOM_ATTEMPTS = max(len(ladder) for ladder in OOM_LADDERS.values())


def oom_action(kind: str, attempt: int) -> str:
    """Deterministic, bounded OOM action for the ``attempt``-th try."""
    if kind not in OOM_KINDS:
        raise ConfigError(
            f"unknown OOM kind {kind!r}",
            details={"field": "kind", "allowed": list(OOM_KINDS)},
        )
    if attempt < 0:
        raise ConfigError("attempt index must be non-negative")
    if attempt >= MAX_OOM_ATTEMPTS:
        raise ConfigError(
            f"OOM handling exceeded {MAX_OOM_ATTEMPTS} attempts for {kind!r}; "
            "the policy must fail deterministically instead of retrying forever",
            details={"field": "attempt"},
        )
    ladder = OOM_LADDERS[kind]
    return ladder[attempt] if attempt < len(ladder) else "fail"


@dataclass(frozen=True)
class ContextLimitCheck:
    """Over-long context must be refused at the earliest knowable layer."""

    requested_tokens: int
    model_max: int
    runtime_max: int
    kv_capacity_tokens: int

    @property
    def limits(self) -> Dict[str, int]:
        return {
            "model_max": self.model_max,
            "runtime_max": self.runtime_max,
            "kv_capacity_tokens": self.kv_capacity_tokens,
        }

    @property
    def rejects(self) -> bool:
        return self.requested_tokens > min(self.limits.values())

    @property
    def earliest_rejecting_layer(self) -> str:
        return min(self.limits, key=lambda name: self.limits[name])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested_tokens": self.requested_tokens,
            "limits": self.limits,
            "rejects": self.rejects,
            "earliest_rejecting_layer": self.earliest_rejecting_layer,
            "headroom_tokens": min(self.limits.values()) - self.requested_tokens,
        }


# ── long-run resource statistics ───────────────────────────────────────────


@dataclass(frozen=True)
class ResourceSlope:
    """Slope of a resource series with a bounded/steady/growing verdict."""

    name: str
    slope_per_cycle: float
    ci_low: float
    ci_high: float
    verdict: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "slope_per_cycle": self.slope_per_cycle,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "verdict": self.verdict,
        }


def resource_slope(
    name: str, samples: Sequence[float], *, tolerance: float
) -> ResourceSlope:
    """Least-squares slope with a plateau/bounded-cache distinction.

    ``tolerance`` is the pre-registered per-cycle drift budget; a positive slope
    inside the budget is a *bounded cache*, not a leak, and a positive slope
    outside it is ``GROWING`` (a leak candidate that blocks the PASS).
    """
    if len(samples) < 3:
        raise ConfigError("a resource slope needs at least three samples")
    if tolerance <= 0:
        raise ConfigError("resource slope needs a positive tolerance budget")
    n = len(samples)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(samples) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ConfigError("degenerate series: all x identical")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, samples)) / denominator
    residuals = [y - (mean_y + slope * (x - mean_x)) for x, y in zip(xs, samples)]
    spread = max(residuals) - min(residuals) if residuals else 0.0
    if abs(slope) <= tolerance:
        verdict = "STEADY" if spread <= tolerance * n else "BOUNDED_CACHE"
    elif slope > 0:
        verdict = "GROWING"
    else:
        verdict = "DECREASING"
    return ResourceSlope(
        name=name,
        slope_per_cycle=slope,
        ci_low=slope - spread,
        ci_high=slope + spread,
        verdict=verdict,
    )


__all__ = [
    "ALLOWED_BLOCK_TRANSITIONS",
    "OOM_LADDERS",
    "BlockPool",
    "BlockRecord",
    "BlockState",
    "CapacityBoundary",
    "ContextLimitCheck",
    "EvictionCandidate",
    "FRAGMENT_CLASSES",
    "KVGeometry",
    "KVRequestAccounting",
    "KVCapacityModel",
    "MAX_OOM_ATTEMPTS",
    "MemoryReconciliation",
    "OOM_ACTION_ORDER",
    "OOM_KINDS",
    "ResourceSlope",
    "block_boundary_points",
    "blocks_for",
    "find_capacity_boundary",
    "oom_action",
    "resource_slope",
    "select_evictions",
]
