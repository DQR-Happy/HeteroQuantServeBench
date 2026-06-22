"""A CPU **test double** of the Ascend execution model — not an Ascend result.

.. warning::

   Nothing in this module runs on an NPU.  :class:`SimulatedDevice` models
   ``CopyIn(GM→UB) / Compute / CopyOut(UB→GM)`` in pure Python so the *host-side*
   half of the contract can be verified on a CPU-only host:

   * the serialized :class:`~hqsb.ascend.tiling.TilingData` round-trips and the
     device-side parse agrees with the host-side one (E09-02 step 6; §11 "let host
     and device each keep an unversioned struct");
   * tail masking never reads or writes outside the legal GM range (step 10);
   * the multi-tile loop terminates and every row is owned by exactly one core
     (steps 11/12/15);
   * the UB budget is charged before launch, including the extra slots a
     ``buffer_count=2`` candidate needs (step 7, E09-04 §3.2);
   * inputs and guard regions are byte-identical afterwards (step 5).

   ``SimResult.claim_allowed()`` is hard-wired to ``False`` and every row is
   stamped ``simulated=True``: these numbers must never appear in a report as
   Ascend correctness, latency or bandwidth.  The real device path is
   ``ops/ascend`` plus the E09-01 probes, and it stays ``BLOCKED`` here.

What the double models, and what it does not
--------------------------------------------
It models *addressing, partitioning, tail validity and memory budgets*.  It does
**not** model time: there is no asynchronous execution, no frequency, no
CopyIn/Compute overlap and therefore no latency, bandwidth or pipeline-hole
number to report.  ``statuses["sync_complete"]`` is the literal string
``"simulated"`` so a report generator cannot mistake it for an observed device
completion.  Overlap is a timing property and belongs to msprof on real silicon
(E09-04 step 19).

The value of the double is that it turns "the tiling math looks right" into a
falsifiable CPU test, and that it fails loudly on exactly the bug classes the
protocol warns about (E09-02 §11): reading past GM and relying on the padding
happening to be zero; ``block_dim`` larger than the work producing an underflow;
a correct-looking result that overran the buffer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hqsb.core.errors import BackendError, ConfigError, UsageError
from hqsb.ascend.operators import (
    OPERATOR_ADD,
    OPERATOR_ROW_REDUCE_SUM,
    OPERATOR_RMSNORM,
)
from hqsb.ascend.test_vectors import _round_to_dtype
from hqsb.ascend.tiling import (
    TILING_KEYS,
    TILING_STRUCT_SIZE,
    TilingData,
    UbBudget,
    dtype_bytes,
)

SIMULATION_DISCLAIMER = (
    "CPU test double of the Ascend execution model. Not a device measurement: no NPU, "
    "CANN toolchain or msprof was involved, and no number produced here may be reported "
    "as Ascend correctness, latency, bandwidth or energy."
)

#: Neutral element per reduction — a padded tail must not change the result
#: (E09-02 step 16).  A ``max`` reduction would need a different one, which is
#: why this is a table rather than a hard-coded ``0.0``.
NEUTRAL_ELEMENTS: Mapping[str, float] = {"sum": 0.0, "add": 0.0, "square_sum": 0.0}

POISON_BYTE = 0xA5

SENTINEL = float("nan")


class DeviceMemoryFault(BackendError):
    """An out-of-range GM access, a UB overrun or a queue protocol error."""


@dataclass(frozen=True)
class GmAllocation:
    """One named global-memory buffer with a hard length."""

    name: str
    offset: int
    length: int
    element_bytes: int

    @property
    def end(self) -> int:
        return self.offset + self.length * self.element_bytes


class SimulatedGlobalMemory:
    """Bounds-checked flat device memory.

    Unallocated bytes hold poison, so "the kernel read past the end and got lucky
    because the padding was zero" becomes a visible fault instead of a pass.
    """

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ConfigError("GM capacity must be positive", details={"capacity_bytes": capacity_bytes})
        self.capacity_bytes = capacity_bytes
        self._bytes = bytearray([POISON_BYTE] * capacity_bytes)
        self._allocations: Dict[str, GmAllocation] = {}
        self._cursor = 0
        self.violations: List[Dict[str, Any]] = []
        self.read_bytes = 0
        self.write_bytes = 0

    def allocate(self, name: str, values: Sequence[float], *, element_bytes: int) -> GmAllocation:
        if name in self._allocations:
            raise ConfigError(f"GM buffer {name!r} is already allocated", details={"name": name})
        allocation = GmAllocation(
            name=name, offset=self._cursor, length=len(values), element_bytes=element_bytes
        )
        if allocation.end > self.capacity_bytes:
            raise DeviceMemoryFault(
                f"GM allocation {name!r} of {len(values)} elements does not fit in "
                f"{self.capacity_bytes} B (needs {allocation.end} B); the real path reports DEVICE_OOM",
                details={"name": name, "needed": allocation.end, "capacity": self.capacity_bytes},
            )
        self._allocations[name] = allocation
        self._cursor = allocation.end
        for index, value in enumerate(values):
            self.store(allocation, index, value, accounted=False)
        return allocation

    def allocation(self, name: str) -> GmAllocation:
        try:
            return self._allocations[name]
        except KeyError:
            raise UsageError(f"unknown GM buffer {name!r}", details={"name": name}) from None

    def _byte_range(self, allocation: GmAllocation, index: int) -> Tuple[int, int]:
        start = allocation.offset + index * allocation.element_bytes
        return start, start + allocation.element_bytes

    def store(self, allocation: GmAllocation, index: int, value: float, *, accounted: bool = True) -> None:
        start, end = self._byte_range(allocation, index)
        if index < 0 or end > allocation.end:
            self.violations.append(
                {
                    "kind": "out_of_bounds_write",
                    "buffer": allocation.name,
                    "index": index,
                    "byte_range": [start, end],
                    "allocation_end": allocation.end,
                }
            )
            raise DeviceMemoryFault(
                f"write to {allocation.name}[{index}] is outside the allocation "
                f"(bytes {start}:{end}, allocation ends at {allocation.end})",
                details={"buffer": allocation.name, "index": index},
            )
        self._bytes[start:end] = _encode(value, allocation.element_bytes)
        if accounted:
            self.write_bytes += allocation.element_bytes

    def load(self, allocation: GmAllocation, index: int) -> float:
        start, end = self._byte_range(allocation, index)
        if index < 0 or end > allocation.end:
            self.violations.append(
                {
                    "kind": "out_of_bounds_read",
                    "buffer": allocation.name,
                    "index": index,
                    "byte_range": [start, end],
                    "allocation_end": allocation.end,
                }
            )
            raise DeviceMemoryFault(
                f"read of {allocation.name}[{index}] is outside the allocation "
                f"(bytes {start}:{end}, allocation ends at {allocation.end})",
                details={"buffer": allocation.name, "index": index},
            )
        self.read_bytes += allocation.element_bytes
        return _decode(bytes(self._bytes[start:end]), allocation.element_bytes)

    def snapshot(self, name: str) -> List[float]:
        allocation = self.allocation(name)
        return [self.load(allocation, index) for index in range(allocation.length)]

    def audit(self) -> Dict[str, Any]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "allocated_bytes": self._cursor,
            "allocations": {
                name: {"offset": item.offset, "length": item.length, "element_bytes": item.element_bytes}
                for name, item in sorted(self._allocations.items())
            },
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "violation_count": len(self.violations),
            "violations": self.violations,
        }


def _encode(value: float, element_bytes: int) -> bytes:
    import struct

    if element_bytes == 4:
        return struct.pack("<f", value)
    if element_bytes == 2:
        try:
            return struct.pack("<e", value)
        except (OverflowError, ValueError):
            return struct.pack("<e", float("inf") if value > 0 else float("-inf"))
    if element_bytes == 8:
        return struct.pack("<d", value)
    raise UsageError(f"no encoding for {element_bytes}-byte elements", details={"element_bytes": element_bytes})


def _decode(payload: bytes, element_bytes: int) -> float:
    import struct

    if element_bytes == 4:
        return struct.unpack("<f", payload)[0]
    if element_bytes == 2:
        return struct.unpack("<e", payload)[0]
    if element_bytes == 8:
        return struct.unpack("<d", payload)[0]
    raise UsageError(f"no decoding for {element_bytes}-byte elements", details={"element_bytes": element_bytes})


class SimulatedUb:
    """On-chip buffer with a hard budget and a high-water mark."""

    def __init__(self, budget: UbBudget) -> None:
        self.budget = budget
        self._live: Dict[str, int] = {}
        self.high_water_bytes = 0
        self.overruns: List[Dict[str, Any]] = []

    @property
    def in_use_bytes(self) -> int:
        return sum(self._live.values())

    def allocate(self, tag: str, nbytes: int) -> int:
        if nbytes < 0:
            raise UsageError(f"negative UB allocation for {tag!r}", details={"tag": tag})
        if tag in self._live:
            raise DeviceMemoryFault(
                f"UB tag {tag!r} is already live; reusing it without freeing would alias two tiles",
                details={"tag": tag},
            )
        projected = self.in_use_bytes + nbytes
        if projected > self.budget.usable_bytes:
            self.overruns.append(
                {"tag": tag, "requested": nbytes, "in_use": self.in_use_bytes, "usable": self.budget.usable_bytes}
            )
            raise DeviceMemoryFault(
                f"UB overrun allocating {tag!r}: {projected} B requested, {self.budget.usable_bytes} B usable "
                f"(method: {self.budget.method})",
                details={"tag": tag, "projected": projected, "usable": self.budget.usable_bytes},
            )
        self._live[tag] = nbytes
        self.high_water_bytes = max(self.high_water_bytes, projected)
        return nbytes

    def free(self, tag: str) -> None:
        if tag not in self._live:
            raise DeviceMemoryFault(
                f"freeing UB tag {tag!r} that is not live — a double free or a leaked handle",
                details={"tag": tag},
            )
        del self._live[tag]

    def audit(self) -> Dict[str, Any]:
        return {
            "usable_bytes": self.budget.usable_bytes,
            "high_water_bytes": self.high_water_bytes,
            "still_live": dict(self._live),
            "leaked": bool(self._live),
            "overruns": self.overruns,
        }


class SimulatedQueue:
    """A TQue-style bounded producer/consumer queue.

    Depth equals ``buffer_count``.  Enqueueing past the depth models the deadlock
    a double-buffered loop hits when Compute never drains (E09-02 step 11).
    """

    def __init__(self, depth: int, name: str) -> None:
        if depth < 1:
            raise ConfigError(f"queue {name!r} depth must be >= 1, got {depth}", details={"depth": depth})
        self.depth = depth
        self.name = name
        self._items: List[Any] = []
        self.enque_count = 0
        self.deque_count = 0
        self.blocked_enque = 0
        self.underflow = 0

    def enque(self, item: Any) -> None:
        if len(self._items) >= self.depth:
            self.blocked_enque += 1
            raise DeviceMemoryFault(
                f"queue {self.name!r} is full at depth {self.depth}; a producer that never waits "
                "for the consumer deadlocks",
                details={"queue": self.name, "depth": self.depth},
            )
        self._items.append(item)
        self.enque_count += 1

    def deque(self) -> Any:
        if not self._items:
            self.underflow += 1
            raise DeviceMemoryFault(
                f"queue {self.name!r} is empty; the consumer ran ahead of the producer",
                details={"queue": self.name},
            )
        self.deque_count += 1
        return self._items.pop(0)

    def audit(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "depth": self.depth,
            "enque_count": self.enque_count,
            "deque_count": self.deque_count,
            "blocked_enque": self.blocked_enque,
            "underflow": self.underflow,
            "drained": not self._items,
        }


@dataclass(frozen=True)
class SimResult:
    """Outcome of one simulated launch."""

    kernel: str
    tiling: TilingData
    output: Tuple[float, ...]
    statuses: Mapping[str, Any]
    accounting: Mapping[str, Any]
    faults: Tuple[Mapping[str, Any], ...] = ()
    simulated: bool = True

    @property
    def ok(self) -> bool:
        return not self.faults and bool(self.statuses.get("launch_completed"))

    def claim_allowed(self) -> bool:
        """Always ``False`` — mirrors ``hqsb.serving.dummy_backend``."""
        return False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kernel": self.kernel,
            "simulated": True,
            "disclaimer": SIMULATION_DISCLAIMER,
            "claim_allowed": False,
            "ok": self.ok,
            "tiling": self.tiling.as_dict(),
            "statuses": dict(self.statuses),
            "accounting": dict(self.accounting),
            "faults": [dict(item) for item in self.faults],
            "output_summary": {
                "count": len(self.output),
                "nan_count": sum(1 for value in self.output if math.isnan(value)),
                "inf_count": sum(1 for value in self.output if math.isinf(value)),
            },
        }


@dataclass
class SimulatedDevice:
    """One simulated NPU: GM, UB budget, bounded queues and byte accounting."""

    ub_budget: UbBudget
    gm_capacity_bytes: int = 1 << 28
    max_cores: int = 48
    dtype: str = "fp32"
    accum_dtype: str = "fp32"

    def __post_init__(self) -> None:
        if self.max_cores < 1:
            raise ConfigError("max_cores must be >= 1", details={"max_cores": self.max_cores})

    # ── launch ──────────────────────────────────────────────────────────────

    def launch(
        self,
        kernel: str,
        tiling_payload: bytes,
        *,
        inputs: Mapping[str, Sequence[float]],
        gamma: Optional[Sequence[float]] = None,
        eps: float = 1e-6,
        reread_x: bool = True,
        integrity_snapshot: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> SimResult:
        """Run one launch from **serialized** tiling bytes.

        Parsing bytes rather than accepting a :class:`TilingData` object is the
        point: it proves the ABI round-trip, not just the Python dataclass.

        ``integrity_snapshot`` names GM buffers whose contents must be unchanged
        afterwards (inputs, and any guard regions the caller allocated as their
        own buffers).
        """
        if len(tiling_payload) != TILING_STRUCT_SIZE:
            raise ConfigError(
                f"tiling payload is {len(tiling_payload)} bytes, the ABI expects {TILING_STRUCT_SIZE}",
                details={"got": len(tiling_payload), "expected": TILING_STRUCT_SIZE},
            )
        tiling = TilingData.from_bytes(tiling_payload)
        if kernel not in (OPERATOR_ADD, OPERATOR_ROW_REDUCE_SUM, OPERATOR_RMSNORM):
            raise UsageError(
                f"the device model implements {OPERATOR_ADD}, {OPERATOR_ROW_REDUCE_SUM} and "
                f"{OPERATOR_RMSNORM}; got {kernel!r}",
                details={"kernel": kernel},
            )
        if tiling.block_dim > self.max_cores:
            raise DeviceMemoryFault(
                f"block_dim={tiling.block_dim} exceeds the {self.max_cores} cores this device model has",
                details={"block_dim": tiling.block_dim, "max_cores": self.max_cores},
            )
        if tiling.tiling_key not in TILING_KEYS:
            raise DeviceMemoryFault(
                f"unknown TilingKey {tiling.tiling_key}; the device must fail rather than take a "
                "default path (E09-04 step 13)",
                details={"tiling_key": tiling.tiling_key},
            )

        item_bytes = dtype_bytes(self.dtype)
        gm = SimulatedGlobalMemory(self.gm_capacity_bytes)
        ub = SimulatedUb(self.ub_budget)
        faults: List[Dict[str, Any]] = []
        queues: List[SimulatedQueue] = []

        buffers: Dict[str, GmAllocation] = {}
        try:
            for name, values in inputs.items():
                buffers[name] = gm.allocate(name, list(values), element_bytes=item_bytes)
            if gamma is not None:
                buffers["gamma"] = gm.allocate("gamma", list(gamma), element_bytes=item_bytes)
            buffers["y"] = gm.allocate(
                "y", [SENTINEL] * _output_length(kernel, tiling), element_bytes=item_bytes
            )
            if kernel == OPERATOR_ADD and "x1" in buffers and "x2" in buffers:
                if buffers["x1"].length != buffers["x2"].length:
                    raise UsageError(
                        "add requires equal-length inputs; broadcasting is rejected by the spec",
                        details={"x1": buffers["x1"].length, "x2": buffers["x2"].length},
                    )
            if kernel == OPERATOR_RMSNORM:
                if gamma is None:
                    raise UsageError("rmsnorm requires gamma", details={"kernel": kernel})
                if buffers["gamma"].length != tiling.hidden:
                    raise UsageError(
                        f"gamma length {buffers['gamma'].length} != hidden {tiling.hidden}",
                        details={"gamma": buffers["gamma"].length, "hidden": tiling.hidden},
                    )

            if kernel == OPERATOR_ADD:
                self._run_add(gm, ub, buffers, tiling, queues, faults)
            elif kernel == OPERATOR_ROW_REDUCE_SUM:
                self._run_row_reduce_sum(gm, ub, buffers, tiling, faults)
            else:
                self._run_rmsnorm(gm, ub, buffers, tiling, eps, reread_x, faults)
            output = gm.snapshot("y")
        except (DeviceMemoryFault, UsageError) as exc:
            faults.append(
                {
                    "kind": type(exc).__name__,
                    "message": str(exc),
                    "details": getattr(exc, "details", {}),
                }
            )
            output = gm.snapshot("y") if "y" in buffers else []

        integrity = _audit_integrity(gm, integrity_snapshot or {}, faults)
        unwritten = [index for index, value in enumerate(output) if math.isnan(value) and _expected_written(kernel, tiling, index)]
        if unwritten:
            faults.append(
                {"kind": "output_not_fully_written", "count": len(unwritten), "indices": unwritten[:16]}
            )

        statuses = {
            "launch_completed": not faults,
            "host_enqueue": "ok",
            # The model has no asynchronous execution, so it must not claim a
            # synchronized device completion it never observed.
            "sync_complete": "simulated",
            "tiling_key": TILING_KEYS[tiling.tiling_key],
            "ub_leaked": ub.audit()["leaked"],
            "queues_drained": all(queue.audit()["drained"] for queue in queues),
        }
        accounting = {
            "gm": gm.audit(),
            "ub": ub.audit(),
            "queues": [queue.audit() for queue in queues],
            "core_histogram": tiling.core_histogram(),
            "row_coverage": tiling.row_coverage_audit(),
            "integrity": integrity,
            "logical_bytes": logical_bytes(kernel, tiling, item_bytes),
            "byte_models": byte_models(kernel, tiling, self.dtype, reread_x=reread_x),
        }
        return SimResult(
            kernel=kernel,
            tiling=tiling,
            output=tuple(output),
            statuses=statuses,
            accounting=accounting,
            faults=tuple(faults),
        )

    # ── kernels ─────────────────────────────────────────────────────────────

    def _run_add(
        self,
        gm: SimulatedGlobalMemory,
        ub: SimulatedUb,
        buffers: Mapping[str, GmAllocation],
        tiling: TilingData,
        queues: List[SimulatedQueue],
        faults: List[Dict[str, Any]],
    ) -> None:
        x1, x2, out = buffers["x1"], buffers["x2"], buffers["y"]
        tile = tiling.tile_elems
        item_bytes = dtype_bytes(self.dtype)
        neutral = NEUTRAL_ELEMENTS["add"]
        for core in range(tiling.block_dim):
            rows_here = tiling.rows_for_core(core)
            if rows_here == 0:
                # An idle core is legal (rows < block_dim) but must do nothing;
                # a negative length here is the underflow E09-02 §11 warns about.
                continue
            start = tiling.start_row(core) * tiling.hidden
            length = rows_here * tiling.hidden
            # ``buffer_count`` slots are held for the whole core loop: that is the
            # UB price of double buffering, charged whether or not it pays off.
            for slot in range(tiling.buffer_count):
                ub.allocate(f"add_x1_c{core}_s{slot}", tile * item_bytes)
                ub.allocate(f"add_x2_c{core}_s{slot}", tile * item_bytes)
                ub.allocate(f"add_y_c{core}_s{slot}", tile * item_bytes)
            queue = SimulatedQueue(tiling.buffer_count, f"add_in_c{core}")
            queues.append(queue)
            loop_count, tail = _loop_and_tail(length, tile)
            if loop_count != tiling.loop_count:
                faults.append(
                    {
                        "kind": "loop_count_mismatch",
                        "core": core,
                        "host_loop_count": tiling.loop_count,
                        "device_loop_count": loop_count,
                        "detail": "host tiling and device loop disagree; the ABI is not shared",
                    }
                )
            for iteration in range(loop_count):
                offset = iteration * tile
                count = tile if iteration < loop_count - 1 else (tail or tile)
                valid = min(count, length - offset)
                left = [
                    gm.load(x1, start + offset + index) if index < valid else neutral
                    for index in range(count)
                ]
                right = [
                    gm.load(x2, start + offset + index) if index < valid else neutral
                    for index in range(count)
                ]
                queue.enque((left, right))
                first, second = queue.deque()
                computed = [
                    _round_to_dtype(a + b, self.dtype) for a, b in zip(first, second)
                ]
                # CopyOut writes only the legal range: a padded tail lane must
                # never reach GM (E09-02 step 10).
                for index in range(valid):
                    gm.store(out, start + offset + index, computed[index])
            for slot in range(tiling.buffer_count):
                ub.free(f"add_x1_c{core}_s{slot}")
                ub.free(f"add_x2_c{core}_s{slot}")
                ub.free(f"add_y_c{core}_s{slot}")

    def _run_row_reduce_sum(
        self,
        gm: SimulatedGlobalMemory,
        ub: SimulatedUb,
        buffers: Mapping[str, GmAllocation],
        tiling: TilingData,
        faults: List[Dict[str, Any]],
    ) -> None:
        src, out = buffers["x"], buffers["y"]
        hidden = tiling.hidden
        tile = tiling.tile_elems
        item_bytes = dtype_bytes(self.dtype)
        accum_bytes = dtype_bytes(self.accum_dtype)
        neutral = NEUTRAL_ELEMENTS["sum"]
        for core in range(tiling.block_dim):
            rows_here = tiling.rows_for_core(core)
            first_row = tiling.start_row(core)
            if rows_here == 0:
                continue
            for slot in range(tiling.buffer_count):
                ub.allocate(f"red_x_c{core}_s{slot}", min(tile, hidden) * item_bytes)
            ub.allocate(f"red_acc_c{core}", accum_bytes)
            for local in range(rows_here):
                row = first_row + local
                base = row * hidden
                # Explicitly initialised per row: reusing on-chip memory would
                # carry the previous row's residue (E09-02 step 14).
                accumulator = 0.0
                loop_count, tail = _loop_and_tail(hidden, tile)
                for iteration in range(loop_count):
                    offset = iteration * tile
                    count = tile if iteration < loop_count - 1 else (tail or tile)
                    valid = min(count, hidden - offset)
                    staged = [
                        gm.load(src, base + offset + index) if index < valid else neutral
                        for index in range(count)
                    ]
                    accumulator += math.fsum(staged)
                gm.store(out, row, _round_to_dtype(accumulator, self.accum_dtype))
            ub.free(f"red_acc_c{core}")
            for slot in range(tiling.buffer_count):
                ub.free(f"red_x_c{core}_s{slot}")
        if faults:  # pragma: no cover - faults are appended by the caller
            return

    def _run_rmsnorm(
        self,
        gm: SimulatedGlobalMemory,
        ub: SimulatedUb,
        buffers: Mapping[str, GmAllocation],
        tiling: TilingData,
        eps: float,
        reread_x: bool,
        faults: List[Dict[str, Any]],
    ) -> None:
        src, weights, out = buffers["x"], buffers["gamma"], buffers["y"]
        hidden = tiling.hidden
        tile = tiling.tile_elems
        item_bytes = dtype_bytes(self.dtype)
        accum_bytes = dtype_bytes(self.accum_dtype)
        if eps <= 0:
            raise UsageError(f"eps must be positive, got {eps}", details={"eps": eps})
        for core in range(tiling.block_dim):
            rows_here = tiling.rows_for_core(core)
            first_row = tiling.start_row(core)
            if rows_here == 0:
                continue
            for slot in range(tiling.buffer_count):
                ub.allocate(f"rms_x_c{core}_s{slot}", min(tile, hidden) * item_bytes)
                ub.allocate(f"rms_y_c{core}_s{slot}", min(tile, hidden) * item_bytes)
            ub.allocate(f"rms_sq_c{core}", accum_bytes)
            if not reread_x:
                # Keeping the row on chip removes the second GM read but costs UB
                # the host budget must already have accounted for.
                ub.allocate(f"rms_keep_c{core}", hidden * item_bytes)
            for local in range(rows_here):
                row = first_row + local
                base = row * hidden
                accumulator = 0.0
                loop_count, tail = _loop_and_tail(hidden, tile)
                if loop_count != tiling.loop_count:
                    faults.append(
                        {
                            "kind": "loop_count_mismatch",
                            "core": core,
                            "row": row,
                            "host_loop_count": tiling.loop_count,
                            "device_loop_count": loop_count,
                        }
                    )
                cached: List[float] = []
                # Pass 1: square and accumulate in the declared accumulation dtype.
                for iteration in range(loop_count):
                    offset = iteration * tile
                    count = tile if iteration < loop_count - 1 else (tail or tile)
                    valid = min(count, hidden - offset)
                    staged = [
                        gm.load(src, base + offset + index) if index < valid else 0.0
                        for index in range(count)
                    ][:valid]
                    if not reread_x:
                        cached.extend(staged)
                    accumulator += math.fsum(value * value for value in staged)
                mean_square = accumulator / hidden
                # eps after the mean of squares and before the sqrt: the position
                # the frozen spec declares, not whichever is convenient.
                rstd = 1.0 / math.sqrt(mean_square + eps)
                cursor = 0
                for iteration in range(loop_count):
                    offset = iteration * tile
                    count = tile if iteration < loop_count - 1 else (tail or tile)
                    valid = min(count, hidden - offset)
                    if reread_x:
                        staged = [
                            gm.load(src, base + offset + index) if index < valid else 0.0
                            for index in range(count)
                        ][:valid]
                    else:
                        staged = cached[cursor : cursor + valid]
                        cursor += valid
                    # gamma is indexed by the in-row position j, never by a flat
                    # global index (E09-03 step 13).
                    for index in range(valid):
                        weight = gm.load(weights, offset + index)
                        gm.store(
                            out,
                            base + offset + index,
                            _round_to_dtype(staged[index] * rstd * weight, self.dtype),
                        )
            if not reread_x:
                ub.free(f"rms_keep_c{core}")
            ub.free(f"rms_sq_c{core}")
            for slot in range(tiling.buffer_count):
                ub.free(f"rms_x_c{core}_s{slot}")
                ub.free(f"rms_y_c{core}_s{slot}")


# ── helpers ──────────────────────────────────────────────────────────────────


def _expected_written(kernel: str, tiling: TilingData, index: int) -> bool:
    """Only indices inside the logical output may stay at the sentinel."""
    length = _output_length(kernel, tiling)
    return 0 <= index < length


def _output_length(kernel: str, tiling: TilingData) -> int:
    if kernel == OPERATOR_ROW_REDUCE_SUM:
        return tiling.rows
    return tiling.rows * tiling.hidden


def _loop_and_tail(length: int, tile: int) -> Tuple[int, int]:
    if tile < 1:
        raise DeviceMemoryFault("tile_elems must be >= 1", details={"tile_elems": tile})
    if length < 0:
        raise DeviceMemoryFault(
            f"negative per-core length {length}; block_dim larger than the work must be "
            "rejected by host tiling, not discovered here",
            details={"length": length},
        )
    loop_count = length // tile
    tail = length % tile
    if tail:
        loop_count += 1
    return loop_count, tail


def logical_bytes(kernel: str, tiling: TilingData, item_bytes: int) -> Dict[str, Any]:
    """The *logical minimum* byte model of details README §8.3.

    This is the shared cross-backend denominator.  It is deliberately separate
    from the simulator's own access counters, which describe this model's
    behaviour and must never be reported as profiler-observed traffic.
    """
    elements = tiling.rows * tiling.hidden
    if kernel == OPERATOR_ADD:
        return {"formula": "bytes(x1) + bytes(x2) + bytes(y)", "bytes": 3 * elements * item_bytes}
    if kernel == OPERATOR_ROW_REDUCE_SUM:
        return {
            "formula": "bytes(x) + bytes(y); partial-workspace traffic is listed separately",
            "bytes": elements * item_bytes + tiling.rows * item_bytes,
            "partial_workspace_bytes": 0,
        }
    return {
        "formula": "2*bytes(x) + bytes(gamma_effective) + bytes(y) when x is re-read",
        "bytes": 3 * elements * item_bytes + tiling.hidden * item_bytes,
        "gamma_read_effective_bytes": tiling.hidden * item_bytes,
        "note": "whether gamma is re-read per row or cached changes this term; report both models",
    }


def byte_models(
    kernel: str, tiling: TilingData, dtype: str, *, reread_x: bool = True
) -> Dict[str, Any]:
    """E09-03 §7: the three byte denominators, published side by side.

    Reporting only the most favourable one is prohibited, so all three come back
    together and the third is explicitly ``UNAVAILABLE`` without a device.
    """
    item_bytes = dtype_bytes(dtype)
    logical = logical_bytes(kernel, tiling, item_bytes)
    elements = tiling.rows * tiling.hidden
    if kernel == OPERATOR_RMSNORM:
        x_reads = 2 if reread_x else 1
        gamma_term = tiling.hidden * item_bytes * (1 if reread_x else tiling.rows)
        implementation = x_reads * elements * item_bytes + gamma_term + elements * item_bytes
        formula = (
            "x re-read from GM: 2*bytes(x) + bytes(gamma) + bytes(y)"
            if reread_x
            else "x held on chip: bytes(x) + bytes(gamma)*rows + bytes(y)"
        )
    else:
        implementation = logical["bytes"]
        formula = logical["formula"]
    return {
        "logical_minimum": logical,
        "implementation_estimated": {"formula": formula, "bytes": implementation},
        "profiler_observed": {
            "status": "UNAVAILABLE",
            "reason": "no msprof on this host; a simulated counter would be fabricated traffic",
        },
        "note": "each denominator yields its own GB/s; publishing one alone is prohibited (E09-03 §7)",
    }


def _audit_integrity(
    gm: SimulatedGlobalMemory, snapshot: Mapping[str, Sequence[float]], faults: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare recorded buffers with their post-launch contents."""
    mutated: Dict[str, List[int]] = {}
    for name, expected in snapshot.items():
        if name not in gm._allocations:
            faults.append({"kind": "integrity_buffer_missing", "buffer": name})
            continue
        current = gm.snapshot(name)
        diffs = [
            index
            for index, value in enumerate(expected)
            if index < len(current) and not _same(value, current[index])
        ]
        if diffs:
            mutated[name] = diffs[:16]
            faults.append(
                {
                    "kind": "input_or_guard_mutated",
                    "buffer": name,
                    "count": len(diffs),
                    "indices": diffs[:16],
                }
            )
    return {"checked": sorted(snapshot), "mutated": mutated, "ok": not mutated}


def _same(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


__all__ = [
    "DeviceMemoryFault",
    "GmAllocation",
    "NEUTRAL_ELEMENTS",
    "POISON_BYTE",
    "SENTINEL",
    "SIMULATION_DISCLAIMER",
    "SimResult",
    "SimulatedDevice",
    "SimulatedGlobalMemory",
    "SimulatedQueue",
    "SimulatedUb",
    "byte_models",
    "logical_bytes",
]
