"""Static / continuous / chunked-prefill scheduling and ledger (E07-04).

What this module is
-------------------
A **deterministic discrete-event scheduler** over already-tokenized requests.
It answers the *structural* questions E07-04 asks — which request is scheduled
in which iteration, how many prefill/decode tokens it receives, when a chunk
boundary falls, which requests are preempted, whether the KV capacity is
respected, whether tokens are conserved, how fair the schedule is, and where
congestion begins — without a GPU and without inventing performance numbers.

What this module is **not**
---------------------------
It is not a benchmark.  The simulator emits iteration units, not milliseconds:
every time field is either zero or copied from a real run that the experiment
supplied.  Simulated iteration counts are labelled as structure in the reports
(`simulated=true` in every payload) and never quoted as throughput.

Two invariants are enforced while simulating, because E07-04 §8 makes them hard
gates:

* **chunk boundaries** never duplicate or drop a prompt token — a request's
  computed positions only ever move forward, and a chunk is exactly the
  contiguous range ``[computed, computed + scheduled)``;
* **token conservation** holds every iteration
  (``previous + scheduled - rollback == new``), so a preemption/recompute cannot
  quietly lose or invent positions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError
from hqsb.runtime import kv as kv_mod
from hqsb.runtime.trace import IterationLedgerEntry, conservation_audit

# ── batching modes (details README §11 / E07-04 §2) ────────────────────────

STATIC = "static"
CONTINUOUS = "continuous"
CHUNKED_PREFILL = "chunked_prefill"

BATCH_MODES: Tuple[str, ...] = (STATIC, CONTINUOUS, CHUNKED_PREFILL)


@dataclass(frozen=True)
class SchedulerSpec:
    """The frozen scheduler configuration of one trace replay."""

    mode: str
    max_batched_tokens: int
    max_sequences: int
    block_size: int
    kv_blocks: int
    chunk_size: int = 0
    decode_priority: bool = False
    long_prefill_threshold: int = 0
    preemption_policy: str = "recompute_longest"
    admission_reserve_full_isl: bool = True
    watermark: float = 0.9
    prefix_cache: bool = False

    def __post_init__(self) -> None:
        if self.mode not in BATCH_MODES:
            raise ConfigError(
                f"unknown batching mode {self.mode!r}",
                details={"field": "mode", "allowed": list(BATCH_MODES)},
            )
        for name in ("max_batched_tokens", "max_sequences", "block_size", "kv_blocks"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive", details={"field": name})
        if self.mode == CHUNKED_PREFILL and self.chunk_size <= 0:
            raise ConfigError(
                "chunked prefill needs a positive chunk_size; otherwise the mode "
                "name promises a behaviour the scheduler does not implement",
                details={"field": "chunk_size"},
            )
        if self.mode != CHUNKED_PREFILL and self.chunk_size:
            raise ConfigError(
                "chunk_size is only meaningful for chunked prefill; setting it "
                "elsewhere would silently change the experiment",
                details={"field": "chunk_size"},
            )
        if self.preemption_policy not in ("none", "recompute_longest", "recompute_newest"):
            raise ConfigError(
                f"unknown preemption policy {self.preemption_policy!r}",
                details={"field": "preemption_policy"},
            )
        if not 0.0 < self.watermark <= 1.0:
            raise ConfigError("watermark must be in (0, 1]")

    @property
    def kv_token_capacity(self) -> int:
        return int(self.kv_blocks * self.block_size * self.watermark)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "max_batched_tokens": self.max_batched_tokens,
            "max_sequences": self.max_sequences,
            "block_size": self.block_size,
            "kv_blocks": self.kv_blocks,
            "kv_token_capacity": self.kv_token_capacity,
            "chunk_size": self.chunk_size,
            "decode_priority": self.decode_priority,
            "long_prefill_threshold": self.long_prefill_threshold,
            "preemption_policy": self.preemption_policy,
            "admission_reserve_full_isl": self.admission_reserve_full_isl,
            "watermark": self.watermark,
            "prefix_cache": self.prefix_cache,
        }


@dataclass(frozen=True)
class SimRequest:
    """One request in a model-core trace (E07-04 §5)."""

    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    submit_iteration: int = 0
    shared_prefix_tokens: int = 0
    cancel_at_iteration: Optional[int] = None
    priority: int = 0

    def __post_init__(self) -> None:
        if self.prompt_tokens <= 0:
            raise ConfigError("prompt_tokens must be positive")
        if self.max_new_tokens <= 0:
            raise ConfigError("max_new_tokens must be positive")
        if self.submit_iteration < 0:
            raise ConfigError("submit_iteration must be non-negative")
        if not 0 <= self.shared_prefix_tokens <= self.prompt_tokens:
            raise ConfigError(
                "shared_prefix_tokens must be within the prompt length",
                details={"field": "shared_prefix_tokens"},
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "max_new_tokens": self.max_new_tokens,
            "submit_iteration": self.submit_iteration,
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "cancel_at_iteration": self.cancel_at_iteration,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class RequestTrace:
    """A named, hashable collection of requests (E07-04 §5 / details §13.2)."""

    name: str
    requests: Tuple[SimRequest, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.requests:
            raise ConfigError("a trace needs at least one request")
        ids = [request.request_id for request in self.requests]
        if len(set(ids)) != len(ids):
            raise ConfigError("trace request ids must be unique")

    @property
    def trace_hash(self) -> str:
        payload = [request.as_dict() for request in self.requests]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def submit_spread(self) -> int:
        return max(
            (request.submit_iteration for request in self.requests), default=0
        ) - min((request.submit_iteration for request in self.requests), default=0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "trace_hash": self.trace_hash,
            "requests": [request.as_dict() for request in self.requests],
        }


def homogeneous_trace(
    name: str, *, count: int, prompt_tokens: int, max_new_tokens: int, **kwargs: Any
) -> RequestTrace:
    return RequestTrace(
        name=name,
        requests=tuple(
            SimRequest(
                request_id=f"{name}-{index}",
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            for index in range(count)
        ),
    )


def mixed_trace(
    name: str,
    *,
    short_count: int,
    long_count: int,
    short_prompt: int,
    long_prompt: int,
    max_new_tokens: int,
    long_first: bool = True,
    **kwargs: Any,
) -> RequestTrace:
    """Pre-registered short/long mixture; the ratio is fixed by the spec."""
    requests: List[SimRequest] = []
    if long_first:
        requests.extend(
            SimRequest(
                request_id=f"{name}-long-{index}",
                prompt_tokens=long_prompt,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            for index in range(long_count)
        )
        requests.extend(
            SimRequest(
                request_id=f"{name}-short-{index}",
                prompt_tokens=short_prompt,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            for index in range(short_count)
        )
    else:
        requests.extend(
            SimRequest(
                request_id=f"{name}-short-{index}",
                prompt_tokens=short_prompt,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            for index in range(short_count)
        )
        requests.extend(
            SimRequest(
                request_id=f"{name}-long-{index}",
                prompt_tokens=long_prompt,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            for index in range(long_count)
        )
    return RequestTrace(name=name, requests=tuple(requests))


def shared_prefix_trace(
    name: str,
    *,
    count: int,
    shared_prefix: int,
    suffix_choices: Sequence[int],
    max_new_tokens: int,
    **kwargs: Any,
) -> RequestTrace:
    """A common root with divergent suffixes (E07-05 §5 prefix-tree case)."""
    if not suffix_choices:
        raise ConfigError("shared-prefix trace needs at least one suffix length")
    requests = []
    for index in range(count):
        suffix = int(suffix_choices[index % len(suffix_choices)])
        requests.append(
            SimRequest(
                request_id=f"{name}-{index}",
                prompt_tokens=shared_prefix + suffix,
                max_new_tokens=max_new_tokens,
                shared_prefix_tokens=shared_prefix,
                **kwargs,
            )
        )
    return RequestTrace(name=name, requests=tuple(requests))


def staggered_trace(
    name: str,
    *,
    requests: Sequence[SimRequest],
    spacing: int,
) -> RequestTrace:
    """Re-submit a trace with a fixed offset between consecutive submits."""
    if spacing < 0:
        raise ConfigError("spacing must be non-negative")
    shifted = [
        SimRequest(
            request_id=request.request_id,
            prompt_tokens=request.prompt_tokens,
            max_new_tokens=request.max_new_tokens,
            submit_iteration=index * spacing,
            shared_prefix_tokens=request.shared_prefix_tokens,
            cancel_at_iteration=request.cancel_at_iteration,
            priority=request.priority,
        )
        for index, request in enumerate(requests)
    ]
    return RequestTrace(name=name, requests=tuple(shifted))


# ── simulation ─────────────────────────────────────────────────────────────


@dataclass
class _RequestRun:
    """Mutable per-request simulation state."""

    spec: SimRequest
    state: str = "WAITING"
    computed: int = 0
    output_tokens: int = 0
    admitted_iteration: int = -1
    first_token_iteration: int = -1
    finish_iteration: int = -1
    preemptions: int = 0
    recomputed_positions: int = 0
    cancelled: bool = False
    schedule: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.state in ("FINISHED", "CANCELLED")

    @property
    def prefilled(self) -> bool:
        return self.computed >= self.spec.prompt_tokens


@dataclass
class RequestOutcome:
    """Per-request outcome of one simulated replay."""

    request_id: str
    admitted_iteration: int
    first_token_iteration: int
    finish_iteration: int
    prompt_tokens: int
    output_tokens: int
    computed_positions: int
    recomputed_positions: int
    preemptions: int
    cancelled: bool
    queue_iterations: int
    wait_iterations: int

    @property
    def e2e_iterations(self) -> int:
        if self.finish_iteration < 0:
            return -1
        return self.finish_iteration - (self.admitted_iteration if self.admitted_iteration >= 0 else 0)

    @property
    def ttft_iterations(self) -> int:
        if self.first_token_iteration < 0:
            return -1
        return self.first_token_iteration - (self.admitted_iteration if self.admitted_iteration >= 0 else 0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "admitted_iteration": self.admitted_iteration,
            "first_token_iteration": self.first_token_iteration,
            "finish_iteration": self.finish_iteration,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "computed_positions": self.computed_positions,
            "recomputed_positions": self.recomputed_positions,
            "preemptions": self.preemptions,
            "cancelled": self.cancelled,
            "queue_iterations": self.queue_iterations,
            "wait_iterations": self.wait_iterations,
            "e2e_iterations": self.e2e_iterations,
            "ttft_iterations": self.ttft_iterations,
        }


@dataclass
class SimulationResult:
    """Structure-only result; every payload carries ``simulated=True``."""

    trace: RequestTrace
    spec: SchedulerSpec
    iterations: List[IterationLedgerEntry]
    outcomes: List[RequestOutcome]
    padding_slots: int
    max_concurrent_running: int
    kv_peak_blocks: int
    simulated: bool = True

    @property
    def makespan_iterations(self) -> int:
        return len(self.iterations)

    def conservation(self) -> Dict[str, Any]:
        return conservation_audit(self.iterations)

    def fairness(self) -> Dict[str, Any]:
        completed = [
            outcome
            for outcome in self.outcomes
            if outcome.finish_iteration >= 0 and not outcome.cancelled
        ]
        throughputs = [
            outcome.output_tokens / max(outcome.e2e_iterations, 1) for outcome in completed
        ]
        slowdowns = [
            outcome.e2e_iterations / max(_solo_estimate(outcome), 1) for outcome in completed
        ]
        return {
            "jain_index": fairness_jain(throughputs) if throughputs else 0.0,
            "normalized_slowdown_mean": (
                sum(slowdowns) / len(slowdowns) if slowdowns else 0.0
            ),
            "completed": len(completed),
            "simulated": True,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "simulated": True,
            "trace": self.trace.as_dict(),
            "spec": self.spec.as_dict(),
            "makespan_iterations": self.makespan_iterations,
            "padding_slots": self.padding_slots,
            "max_concurrent_running": self.max_concurrent_running,
            "kv_peak_blocks": self.kv_peak_blocks,
            "conservation": self.conservation(),
            "fairness": self.fairness(),
            "outcomes": [outcome.as_dict() for outcome in self.outcomes],
            "iterations": [entry.as_dict() for entry in self.iterations],
        }


def _solo_estimate(outcome: RequestOutcome) -> int:
    return outcome.prompt_tokens + outcome.output_tokens


class SchedulerSimulator:
    """Deterministic replay of one :class:`RequestTrace` under one spec."""

    def __init__(self, spec: SchedulerSpec, trace: RequestTrace) -> None:
        self.spec = spec
        self.trace = trace
        self.runs: Dict[str, _RequestRun] = {
            request.request_id: _RequestRun(spec=request) for request in trace.requests
        }
        if spec.mode == STATIC:
            self.static_batch: List[str] = [
                request.request_id for request in trace.requests
            ][: spec.max_sequences]
        else:
            self.static_batch = []
        self.iterations: List[IterationLedgerEntry] = []
        self.padding_slots = 0
        self.kv_peak_blocks = 0
        self._iteration = 0

    # ── helpers ───────────────────────────────────────────────────────────

    def reserve_blocks(self, run: _RequestRun) -> int:
        """Blocks a run holds against the pool.

        With ``admission_reserve_full_isl`` (the default) a run reserves its whole
        prompt *and* its declared output budget at admission, so a long prefill can
        never over-admit and then thrash (E07-03 step 13 / E07-04 step 13).  The
        alternative policy reserves only the positions already computed, which is
        the behaviour an over-committing runtime shows.
        """
        if self.spec.admission_reserve_full_isl:
            target = run.spec.prompt_tokens + run.spec.max_new_tokens
        else:
            target = run.computed
        return kv_mod.blocks_for(target, self.spec.block_size)

    def _blocks_held(self, runs: Sequence[_RequestRun]) -> int:
        return sum(self.reserve_blocks(run) for run in runs)

    def _capacity_ok(self, extra: _RequestRun, running: Sequence[_RequestRun]) -> bool:
        held = self._blocks_held(list(running) + [extra])
        return held <= int(self.spec.kv_blocks * self.spec.watermark)

    def _admit(self, run: _RequestRun) -> None:
        run.state = "RUNNING"
        run.admitted_iteration = self._iteration

    def _preempt(self, victim: _RequestRun, running: List[_RequestRun]) -> int:
        dropped = victim.computed
        victim.preemptions += 1
        victim.recomputed_positions += dropped
        victim.computed = 0
        victim.output_tokens = 0
        victim.first_token_iteration = -1
        victim.state = "WAITING"
        victim.admitted_iteration = -1
        if victim in running:
            running.remove(victim)
        return dropped

    def _select_victim(self, running: Sequence[_RequestRun]) -> Optional[_RequestRun]:
        if self.spec.preemption_policy == "none" or not running:
            return None
        if self.spec.preemption_policy == "recompute_newest":
            return min(running, key=lambda run: run.computed)
        return max(running, key=lambda run: run.computed)

    # ── one iteration ─────────────────────────────────────────────────────

    def step(self) -> Optional[IterationLedgerEntry]:
        spec = self.spec
        iteration = self._iteration
        scheduled: Dict[str, int] = {}
        prefill_tokens = 0
        decode_tokens = 0
        rollback = 0
        previous_total = sum(run.computed for run in self.runs.values())
        budget = spec.max_batched_tokens
        running: List[_RequestRun] = [
            run for run in self.runs.values() if run.state == "RUNNING"
        ]
        waiting: List[_RequestRun] = [
            run for run in self.runs.values() if run.state == "WAITING"
        ]
        preempted: List[str] = []

        # 1. cancels are observed before scheduling (they free KV immediately).
        for run in list(waiting) + list(running):
            if (
                run.spec.cancel_at_iteration is not None
                and iteration >= run.spec.cancel_at_iteration
                and not run.done
            ):
                run.cancelled = True
                run.state = "CANCELLED"
                run.finish_iteration = iteration
                if run in running:
                    running.remove(run)
                if run in waiting:
                    waiting.remove(run)
                rollback += run.computed
                run.computed = 0

        # 2. admission.
        for run in list(waiting):
            if run.spec.submit_iteration > iteration:
                continue
            if len(running) >= spec.max_sequences:
                break
            if not self._capacity_ok(run, running):
                victim = self._select_victim(running)
                if victim is None:
                    continue
                rollback += self._preempt(victim, running)
                preempted.append(victim.spec.request_id)
                if not self._capacity_ok(run, running):
                    continue
            self._admit(run)
            running.append(run)

        # 3. static mode: only the frozen batch participates.
        if spec.mode == STATIC:
            batch = [self.runs[rid] for rid in self.static_batch]
            active = [
                run
                for run in batch
                if not run.done and run.spec.submit_iteration <= iteration
            ]
            idle_slots = len(batch) - len(active)
            self.padding_slots += max(idle_slots, 0)
            for run in batch:
                if run.done or run.spec.submit_iteration > iteration:
                    continue
                if run not in running:
                    self._admit(run)
                    running.append(run)
            running = [run for run in running if run in batch]

        # 4. scheduling decisions.
        order = sorted(
            running,
            key=lambda run: (
                # decode priority first, then admission order, then priority field
                0 if (spec.decode_priority and run.prefilled) else 1,
                run.admitted_iteration,
                -run.spec.priority,
                run.spec.request_id,
            ),
        )
        for run in order:
            if run.done or run.state not in ("RUNNING", "PREEMPTED"):
                continue
            if not run.prefilled:
                remaining = run.spec.prompt_tokens - run.computed
                if spec.mode == CHUNKED_PREFILL:
                    allowance = min(remaining, spec.chunk_size, budget)
                else:
                    allowance = min(remaining, budget)
                if allowance <= 0:
                    continue
                # The block budget was reserved at admission, so growth inside the
                # reservation cannot over-commit; exceeding it is a bookkeeping bug.
                if run.computed + allowance > self.reserve_blocks(run) * self.spec.block_size:
                    raise ConfigError(
                        f"request {run.spec.request_id} would exceed its reserved KV "
                        "budget; admission accounting and scheduling disagree",
                        details={"request_id": run.spec.request_id},
                    )
                run.computed += allowance
                run.schedule.append((iteration, allowance))
                scheduled[run.spec.request_id] = allowance
                prefill_tokens += allowance
                budget -= allowance
                if run.prefilled and run.first_token_iteration < 0:
                    # First token in the same iteration as the last prefill chunk;
                    # it costs a computed position, so the ledger must count it too.
                    run.output_tokens = 1
                    run.first_token_iteration = iteration
                    run.schedule.append((iteration, 1))
                    scheduled[run.spec.request_id] = allowance + 1
                    run.computed += 1
                    decode_tokens += 1
                    budget -= 1
            else:
                if budget < 1:
                    continue
                run.computed += 1
                run.output_tokens += 1
                run.schedule.append((iteration, 1))
                scheduled[run.spec.request_id] = scheduled.get(run.spec.request_id, 0) + 1
                decode_tokens += 1
                budget -= 1

        # 5. completion.
        finished: List[str] = []
        for run in list(running):
            if run.cancelled:
                continue
            if run.prefilled and run.output_tokens >= run.spec.max_new_tokens:
                run.state = "FINISHED"
                run.finish_iteration = iteration
                finished.append(run.spec.request_id)
                running.remove(run)

        new_total = sum(run.computed for run in self.runs.values())
        # Blocks are held by *running* requests only: a finished or cancelled
        # request has already released its reservation in the ledger.
        blocks_used = self._blocks_held(running)
        entry = IterationLedgerEntry(
            iteration=iteration,
            waiting=tuple(
                run.spec.request_id
                for run in self.runs.values()
                if run.state == "WAITING" and run.spec.submit_iteration <= iteration
            ),
            running=tuple(run.spec.request_id for run in running),
            preempted=tuple(preempted),
            finished=tuple(finished),
            scheduled_tokens=scheduled,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            token_budget=spec.max_batched_tokens,
            sequence_budget=spec.max_sequences,
            kv_free_blocks=max(spec.kv_blocks - blocks_used, 0),
            kv_used_blocks=blocks_used,
            model_input_m=len(scheduled),
            previous_computed_positions=previous_total,
            new_computed_positions=new_total,
            rollback_positions=rollback,
            oom_or_preemption="preempt" if preempted else "",
        )
        if entry.conservation_residual() != 0:
            raise ConfigError(
                "scheduler ledger violates token conservation at iteration "
                f"{iteration}: residual={entry.conservation_residual()}",
                details={"iteration": iteration},
            )
        self.kv_peak_blocks = max(self.kv_peak_blocks, entry.kv_used_blocks)
        self.iterations.append(entry)
        self._iteration += 1
        return entry

    def run(self, *, max_iterations: int) -> SimulationResult:
        """Replay until every request is done or ``max_iterations`` is reached."""
        if max_iterations <= 0:
            raise ConfigError("max_iterations must be positive")
        max_concurrent = 0
        for _ in range(max_iterations):
            self.step()
            active = sum(
                1
                for run in self.runs.values()
                if run.state in ("RUNNING", "PREFILLING", "DECODING")
            )
            max_concurrent = max(max_concurrent, active)
            if all(run.done for run in self.runs.values()):
                break
        outcomes: List[RequestOutcome] = []
        for run in self.runs.values():
            queue_iterations = max(
                (run.admitted_iteration if run.admitted_iteration >= 0 else self._iteration)
                - run.spec.submit_iteration,
                0,
            )
            outcomes.append(
                RequestOutcome(
                    request_id=run.spec.request_id,
                    admitted_iteration=run.admitted_iteration,
                    first_token_iteration=run.first_token_iteration,
                    finish_iteration=run.finish_iteration,
                    prompt_tokens=run.spec.prompt_tokens,
                    output_tokens=run.output_tokens,
                    # Positions the model actually computed: final prompt + decode
                    # plus everything thrown away by preemption (never deleted from
                    # the ledger, E07 details §13.5).
                    computed_positions=run.computed + run.recomputed_positions,
                    recomputed_positions=run.recomputed_positions,
                    preemptions=run.preemptions,
                    cancelled=run.cancelled,
                    queue_iterations=queue_iterations,
                    wait_iterations=(
                        (run.first_token_iteration - run.spec.submit_iteration)
                        if run.first_token_iteration >= 0
                        else -1
                    ),
                )
            )
        return SimulationResult(
            trace=self.trace,
            spec=self.spec,
            iterations=list(self.iterations),
            outcomes=outcomes,
            padding_slots=self.padding_slots,
            max_concurrent_running=max_concurrent,
            kv_peak_blocks=self.kv_peak_blocks,
        )


def simulate(
    spec: SchedulerSpec, trace: RequestTrace, *, max_iterations: int = 512
) -> SimulationResult:
    return SchedulerSimulator(spec, trace).run(max_iterations=max_iterations)


# ── correctness helpers ────────────────────────────────────────────────────


def chunk_boundaries(prompt_tokens: int, chunk_size: int) -> Tuple[Tuple[int, int], ...]:
    """Contiguous half-open chunk ranges; together they tile ``[0, prompt)``."""
    if prompt_tokens <= 0 or chunk_size <= 0:
        raise ConfigError("chunk boundaries need a positive prompt and chunk size")
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < prompt_tokens:
        end = min(start + chunk_size, prompt_tokens)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def chunk_coverage_audit(
    prompt_tokens: int, chunk_size: int, scheduled: Sequence[Tuple[int, int]]
) -> Dict[str, Any]:
    """Verify the scheduled chunk offsets tile the prompt exactly once."""
    expected = chunk_boundaries(prompt_tokens, chunk_size)
    problems: List[str] = []
    if len(scheduled) != len(expected):
        problems.append(f"expected {len(expected)} chunks, got {len(scheduled)}")
    cursor = 0
    for index, (start, count) in enumerate(scheduled):
        if start != cursor:
            problems.append(f"chunk {index} starts at {start}, expected {cursor}")
        cursor = start + count
    if cursor != prompt_tokens:
        problems.append(f"chunks cover {cursor} tokens, prompt has {prompt_tokens}")
    return {"ok": not problems, "problems": problems, "chunks": list(scheduled)}


def fairness_jain(values: Sequence[float]) -> float:
    """Jain's fairness index; 1.0 is perfectly fair (E07-04 §6)."""
    if not values:
        raise ConfigError("Jain index needs at least one value")
    if any(value < 0 for value in values):
        raise ConfigError("Jain index is defined for non-negative values")
    denominator = len(values) * sum(value * value for value in values)
    if denominator == 0:
        return 1.0
    return (sum(values) ** 2) / denominator


@dataclass(frozen=True)
class CongestionReport:
    """Where the deterministic trace stops being comfortably served."""

    congestion_iteration: Optional[int]
    reasons: Tuple[str, ...]
    waiting_growth_iterations: int
    budget_saturation: float
    kv_pressure: float
    preemption_iterations: Tuple[int, ...]

    @property
    def congested(self) -> bool:
        return self.congestion_iteration is not None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "congested": self.congested,
            "congestion_iteration": self.congestion_iteration,
            "reasons": list(self.reasons),
            "waiting_growth_iterations": self.waiting_growth_iterations,
            "budget_saturation": self.budget_saturation,
            "kv_pressure": self.kv_pressure,
            "preemption_iterations": list(self.preemption_iterations),
            "simulated": True,
        }


def detect_congestion(
    result: SimulationResult,
    *,
    waiting_growth_run: int = 3,
    budget_saturation_threshold: float = 0.95,
) -> CongestionReport:
    """Mark the first iteration where queuing, saturation or thrash appears."""
    waiting_growth = 0
    congestion_iteration: Optional[int] = None
    reasons: List[str] = []
    preemption_iterations: List[int] = []
    saturated = 0
    for entry in result.iterations:
        if entry.scheduled_total >= entry.token_budget * budget_saturation_threshold:
            saturated += 1
        if entry.waiting:
            waiting_growth += 1
        else:
            waiting_growth = 0
        if entry.oom_or_preemption:
            preemption_iterations.append(entry.iteration)
        if congestion_iteration is None:
            if waiting_growth >= waiting_growth_run:
                congestion_iteration = entry.iteration
                reasons.append(
                    f"waiting queue grew for {waiting_growth} consecutive iterations"
                )
            elif entry.scheduled_total >= entry.token_budget:
                congestion_iteration = entry.iteration
                reasons.append("token budget saturated")
            elif entry.oom_or_preemption:
                congestion_iteration = entry.iteration
                reasons.append("KV pressure forced preemption")
    budget_saturation = saturated / max(len(result.iterations), 1)
    kv_pressure = result.kv_peak_blocks / max(result.spec.kv_blocks, 1)
    return CongestionReport(
        congestion_iteration=congestion_iteration,
        reasons=tuple(reasons),
        waiting_growth_iterations=waiting_growth,
        budget_saturation=budget_saturation,
        kv_pressure=kv_pressure,
        preemption_iterations=tuple(preemption_iterations),
    )


def strategy_curve(
    specs: Sequence[SchedulerSpec], trace: RequestTrace, *, max_iterations: int = 512
) -> List[Dict[str, Any]]:
    """Run one trace under several specs to build the strategy curve (§9 step 18)."""
    if len(specs) < 2:
        raise ConfigError(
            "a strategy curve needs at least two configurations (a curve of one "
            "point is a single measurement, not a comparison)"
        )
    rows = []
    for spec in specs:
        result = simulate(spec, trace, max_iterations=max_iterations)
        rows.append(
            {
                "mode": spec.mode,
                "chunk_size": spec.chunk_size,
                "max_batched_tokens": spec.max_batched_tokens,
                "max_sequences": spec.max_sequences,
                "makespan_iterations": result.makespan_iterations,
                "padding_slots": result.padding_slots,
                "kv_peak_blocks": result.kv_peak_blocks,
                "congestion": detect_congestion(result).as_dict(),
                "simulated": True,
            }
        )
    return rows


__all__ = [
    "BATCH_MODES",
    "CHUNKED_PREFILL",
    "CONTINUOUS",
    "CongestionReport",
    "RequestOutcome",
    "RequestTrace",
    "STATIC",
    "SchedulerSimulator",
    "SchedulerSpec",
    "SimRequest",
    "SimulationResult",
    "chunk_boundaries",
    "chunk_coverage_audit",
    "detect_congestion",
    "fairness_jain",
    "homogeneous_trace",
    "mixed_trace",
    "shared_prefix_trace",
    "simulate",
    "staggered_trace",
    "strategy_curve",
]
