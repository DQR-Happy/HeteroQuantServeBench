"""Experiment step → code interface map (the "no missing capability" proof).

Each S07 experiment document lists twenty concrete steps.  This module records,
for **every** step, which HQSB interface implements it, and
:func:`resolve_interfaces` verifies that each referenced symbol actually imports
— so the map cannot rot into documentation.  A test
(``tests/unit/runtime/test_s07_experiment_scaffolding.py``) runs the
verification.

Maturity labels follow the control plane (M0–M7):

* ``M1`` — source exists (interface implemented, structured errors);
* ``M2`` — covered by automated tests in this tree;
* ``M3``+ — requires real runtime/hardware evidence, which this repository state
  cannot produce (the S04.5 M4 and S06 P0 prerequisites are unmet), so no entry
  is labelled above M2.

Drivers live in ``scripts/runtime/run_e07.py`` and are *not* executed by this
map.  Step counts come from ``docs/stage_experiments/details/S07/E07-*.md``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class StepMapping:
    """One experiment step and the interfaces that implement it."""

    step: int
    title: str
    interfaces: Tuple[str, ...]
    maturity: str = "M2"
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "title": self.title,
            "interfaces": list(self.interfaces),
            "maturity": self.maturity,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ExperimentMapping:
    """All steps of one experiment plus its driver entry."""

    experiment_id: str
    title: str
    driver: str
    steps: Tuple[StepMapping, ...]
    fixture: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "driver": self.driver,
            "fixture": self.fixture,
            "steps": [step.as_dict() for step in self.steps],
        }


def _steps(pairs: Sequence[Tuple[int, str, Sequence[str], str]]) -> Tuple[StepMapping, ...]:
    """Compact constructor: ``(step, title, interfaces, notes)``."""
    return tuple(
        StepMapping(step=step, title=title, interfaces=tuple(interfaces), notes=notes)
        for step, title, interfaces, notes in pairs
    )


E07_01 = ExperimentMapping(
    experiment_id="E07-01",
    title="backend semantic parity (load/warmup/generate/stream/cancel/metrics/close)",
    driver="scripts/runtime/run_e07.py --experiment E07-01",
    fixture="hqsb.runtime.request.BackendSpec / adapter.probe_environment",
    steps=_steps(
        [
            (1, "freeze BackendSpec and versions", ["hqsb.runtime.request.BackendSpec", "hqsb.runtime.request.assert_single_primary", "hqsb.runtime.adapter.AdapterRegistry.register", "hqsb.runtime.adapter.probe_environment"], "exactly one source-level primary is enforced"),
            (2, "freeze model/tokenizer/sampling", ["hqsb.runtime.request.ModelIdentity", "hqsb.runtime.request.SamplingSpec", "hqsb.runtime.request.StopSpec", "hqsb.runtime.request.token_hash"], "comparison unit is the token ID sequence"),
            (3, "implement adapter schema", ["hqsb.runtime.adapter.RuntimeAdapter", "hqsb.runtime.adapter.AdapterRegistration", "hqsb.runtime.adapter.AdapterState"], "business code never calls a private runtime API"),
            (4, "capability probe", ["hqsb.runtime.request.CapabilityReport.declare", "hqsb.runtime.request.CapabilityField", "hqsb.runtime.request.CAPABILITY_STATES", "hqsb.runtime.adapter.select_primary_engine"], "state/constraint/reason per field; UNKNOWN blocks a claim"),
            (5, "load identity test", ["hqsb.runtime.adapter.RuntimeAdapter.verify_identity", "hqsb.runtime.request.ModelIdentity.as_dict"], "a different revision is refused, never auto-selected"),
            (6, "warmup boundary test", ["hqsb.runtime.adapter.RuntimeAdapter.warmup", "hqsb.runtime.trace.OverheadObservation"], "compile/graph/cache side effects kept out of steady state"),
            (7, "greedy single-request parity", ["hqsb.runtime.parity.compare_greedy", "hqsb.runtime.parity.GreedyParityReport"], "first diverging step is localised"),
            (8, "sampling semantics", ["hqsb.runtime.parity.compare_sampling_distribution", "hqsb.runtime.parity.assert_no_single_seed_claim"], "a single seed never proves distribution equality"),
            (9, "streaming concatenation", ["hqsb.runtime.parity.compare_streaming", "hqsb.runtime.adapter.validate_stream", "hqsb.runtime.adapter.StreamChunk"], "order/duplication/final flag are checked"),
            (10, "stop/EOS/length boundaries", ["hqsb.runtime.parity.boundary_cases", "hqsb.runtime.parity.BoundaryCase", "hqsb.runtime.request.StopSpec"], "EOS, max tokens and min tokens cases are pre-registered"),
            (11, "per-parameter effect", ["hqsb.runtime.parity.parameter_effect_matrix", "hqsb.runtime.parity.ParameterEffect"], "every claimable field must change output or show telemetry"),
            (12, "unsupported negative tests", ["hqsb.runtime.parity.unsupported_negative_matrix", "hqsb.runtime.parity.UnsupportedObservation", "hqsb.runtime.request.resolve_request"], "silently ignored fields fail"),
            (13, "cancel semantics", ["hqsb.runtime.adapter.CancelRecord", "hqsb.runtime.adapter.RuntimeAdapter.cancel", "hqsb.runtime.failure.CancelTimeline"], "requested/observed/done plus the in-flight policy"),
            (14, "metrics consistency", ["hqsb.runtime.request.ResolvedConfig", "hqsb.runtime.metrics.RequestTimeline", "hqsb.runtime.comparison.recompute_metrics"], "requested vs actual is recorded, not inferred"),
            (15, "close / double close", ["hqsb.runtime.adapter.ALLOWED_ADAPTER_TRANSITIONS", "hqsb.runtime.adapter.DummyRuntimeAdapter.close"], "idempotent close; old handle unusable"),
            (16, "repeated load-close", ["hqsb.runtime.adapter.run_load_close_scenarios", "hqsb.runtime.adapter.load_close_scenarios"], "bridges to the E07-09 long-run precondition"),
            (17, "error recovery", ["hqsb.runtime.parity.error_recovery_sequence", "hqsb.runtime.parity.error_recovery_report"], "a healthy request after a failure must match the golden"),
            (18, "fresh-process re-verification", ["hqsb.runtime.experiment.interface_only_run", "hqsb.runtime.experiment.environment_fingerprint"], "no inherited model/cache/RNG state"),
            (19, "capability/parity matrix", ["hqsb.runtime.parity.capability_parity_matrix", "hqsb.runtime.request.CapabilityReport.matrix"], "incomparable rows keep an explicit NA reason"),
            (20, "verdict", ["hqsb.runtime.experiment.RunDirectory.write_verdict", "hqsb.runtime.experiment.ExperimentRecord", "hqsb.runtime.telemetry.project_c6"], "semantic failures never enter the performance experiments; the handbook §4 record can never carry a conclusion without raw evidence"),
        ]
    ),
)

E07_02 = ExperimentMapping(
    experiment_id="E07-02",
    title="source-level hot path and C7 trace (request → scheduler → runner → KV → kernel)",
    driver="scripts/runtime/run_e07.py --experiment E07-02",
    fixture="hqsb.runtime.trace.SpanCollector / telemetry.TraceCollector",
    steps=_steps(
        [
            (1, "freeze source and build", ["hqsb.runtime.experiment.git_state", "hqsb.runtime.experiment.environment_fingerprint"], "commit/flags/dependencies recorded before instrumenting"),
            (2, "hypothesised call chain", ["hqsb.runtime.trace.SPAN_KINDS", "hqsb.runtime.trace.call_chain_tree"], "the expectation is a hypothesis, not a result"),
            (3, "define the C7 span schema", ["hqsb.runtime.telemetry.TraceRecord", "hqsb.runtime.telemetry.to_trace_events", "hqsb.runtime.telemetry.KIND_TO_EVENT_TYPE"], "request/iteration/KV/kernel correlation"),
            (4, "instrument request state", ["hqsb.runtime.trace.RequestStateMachine", "hqsb.runtime.trace.ALLOWED_TRANSITIONS", "hqsb.runtime.trace.StateTransitionRecord"], "every transition carries a reason"),
            (5, "instrument scheduler", ["hqsb.runtime.trace.IterationLedgerEntry", "hqsb.runtime.scheduler.SchedulerSimulator.step"], "queue, budget and decision recorded per iteration"),
            (6, "instrument KV manager", ["hqsb.runtime.kv.BlockPool.events", "hqsb.runtime.kv.BlockRecord.history"], "allocate/free/evict/refcount events"),
            (7, "instrument model runner", ["hqsb.runtime.trace.SpanCollector.emit", "hqsb.runtime.trace.Span", "hqsb.runtime.graph_route.RouteDecision"], "batch metadata, shape, graph and attention route"),
            (8, "correlate GPU kernels", ["hqsb.runtime.trace.kernel_join", "hqsb.runtime.trace.ClockCalibration"], "a kernel without a parent span is reported, not hidden"),
            (9, "instrument sampling/output", ["hqsb.runtime.trace.Span.token_index", "hqsb.runtime.trace.SpanCollector.for_request"], "token index ties an output token to an iteration"),
            (10, "single-request prefill", ["hqsb.runtime.trace.conservation_audit", "hqsb.runtime.metrics.RequestTimeline"], "the simplest chain and token conservation"),
            (11, "long decode", ["hqsb.runtime.metrics.RequestTimeline.itl_ms", "hqsb.runtime.metrics.distribution_summary"], "repeated iterations and KV growth"),
            (12, "concurrent mixed trace", ["hqsb.runtime.scheduler.simulate", "hqsb.runtime.trace.SpanCollector.join_audit"], "dynamic join/exit and preemption"),
            (13, "prefix/graph path", ["hqsb.runtime.prefix_cache.PrefixCache.lookup", "hqsb.runtime.graph_route.GraphSpec.resolve"], "spans for the reuse and bucket paths"),
            (14, "cancel/failure path", ["hqsb.runtime.trace.RequestStateMachine.require_clean_finish", "hqsb.runtime.failure.CancelTimeline"], "the state machine reaches cleanup"),
            (15, "cross-thread context", ["hqsb.runtime.telemetry.TraceCollector.emit", "hqsb.runtime.trace.SpanCollector.join_audit"], "request_id is explicit, never thread-local"),
            (16, "host/device clocks", ["hqsb.runtime.trace.ClockCalibration.host_device_delta_ns", "hqsb.runtime.trace.ClockCalibration.method"], "different clocks are never subtracted directly"),
            (17, "instrumentation overhead", ["hqsb.runtime.trace.instrumentation_overhead", "hqsb.runtime.trace.OverheadThresholds", "hqsb.runtime.trace.INSTRUMENTATION_LEVELS"], "off/minimal/full/profiler measured separately"),
            (18, "state machine and call chain figures", ["hqsb.runtime.trace.state_machine_graph", "hqsb.runtime.trace.call_chain_tree"], "generated from raw events"),
            (19, "hot path table", ["hqsb.runtime.trace.hot_path_table", "hqsb.runtime.trace.HotPathRow"], "split by phase and by CPU/GPU/queue"),
            (20, "manual replay of a representative request", ["hqsb.runtime.telemetry.trace_join_check", "hqsb.runtime.telemetry.chain_coverage"], "no broken span chain"),
        ]
    ),
)

E07_03 = ExperimentMapping(
    experiment_id="E07-03",
    title="paged KV capacity, fragmentation, sharing and lifetime",
    driver="scripts/runtime/run_e07.py --experiment E07-03",
    fixture="hqsb.runtime.kv.KVGeometry / BlockPool / KVCapacityModel",
    steps=_steps(
        [
            (1, "freeze the KV schema", ["hqsb.runtime.kv.KVGeometry", "hqsb.runtime.kv.FRAGMENT_CLASSES", "hqsb.runtime.specs.audit_kv_spec"], "L/H_kv/D_h/dtype/layout/group/block metadata"),
            (2, "export the block manager", ["hqsb.runtime.kv.BlockPool", "hqsb.runtime.kv.BlockState", "hqsb.runtime.kv.ALLOWED_BLOCK_TRANSITIONS"], "state, pool capacity and allocation policy"),
            (3, "calibrate empty and model memory", ["hqsb.runtime.kv.MemoryReconciliation", "hqsb.runtime.kv.MemoryReconciliation.declare"], "non-KV resident memory separated from the KV model"),
            (4, "single-request per-token growth", ["hqsb.runtime.kv.KVRequestAccounting", "hqsb.runtime.kv.blocks_for"], "predicted blocks/bytes checked against the block table"),
            (5, "block boundary points", ["hqsb.runtime.kv.block_boundary_points", "hqsb.runtime.kv.BlockPool.allocate"], "P-1/P/P+1/2P-1/2P/tail"),
            (6, "block size sweep", ["hqsb.runtime.kv.KVCapacityModel", "hqsb.runtime.kv.KVGeometry.bytes_for"], "same request trace, same memory budget"),
            (7, "context/output sweep", ["hqsb.runtime.kv.KVRequestAccounting.internal_fragment_tokens", "hqsb.runtime.kv.MemoryReconciliation.as_rows"], "short and long context plus long decode"),
            (8, "batch and length distribution sweep", ["hqsb.runtime.scheduler.mixed_trace", "hqsb.runtime.kv.KVCapacityModel.max_concurrent_requests"], "uniform, mixed and adversarial tails"),
            (9, "allocate/free", ["hqsb.runtime.kv.BlockPool.release", "hqsb.runtime.kv.BlockPool.invariant_report"], "finish/cancel/timeout paths"),
            (10, "shared prefix refcount", ["hqsb.runtime.kv.BlockPool.share", "hqsb.runtime.prefix_cache.PrefixCache.release"], "two readers, one exits, all exit"),
            (11, "eviction", ["hqsb.runtime.kv.select_evictions", "hqsb.runtime.kv.BlockPool.evict", "hqsb.runtime.prefix_cache.PrefixCache.evict"], "only eligible cached blocks; active output unchanged"),
            (12, "preemption/recompute", ["hqsb.runtime.scheduler.SchedulerSpec.preemption_policy", "hqsb.runtime.scheduler.RequestOutcome.recomputed_positions"], "released blocks and conserved tokens"),
            (13, "chunked prefill", ["hqsb.runtime.scheduler.chunk_coverage_audit", "hqsb.runtime.scheduler.CHUNKED_PREFILL"], "reserve per chunk; no over-admission thrash"),
            (14, "metadata and fragmentation", ["hqsb.runtime.kv.MemoryReconciliation.explained", "hqsb.runtime.kv.MemoryReconciliation.residual_bytes"], "internal/external/cache/allocator separated"),
            (15, "attention access", ["hqsb.runtime.graph_route.AttentionRequest", "hqsb.runtime.graph_route.phase_split_report"], "block size effect on the kernel and TPOT"),
            (16, "maximum capacity", ["hqsb.runtime.kv.find_capacity_boundary", "hqsb.runtime.kv.KVCapacityModel.safe_admission_tokens"], "bisection with a fresh state per point"),
            (17, "fault injection", ["hqsb.runtime.kv.BlockPool.inject_double_free", "hqsb.runtime.kv.BlockPool.inject_refcount_error", "hqsb.runtime.kv.BlockPool.inject_stale_id_access"], "double free, stale id, refcount error, evict race"),
            (18, "long-run loop", ["hqsb.runtime.kv.resource_slope", "hqsb.runtime.failure.resource_slope_report"], "no growth or stale reference after allocate/free/evict"),
            (19, "select a safe configuration", ["hqsb.runtime.kv.ContextLimitCheck", "hqsb.runtime.kv.KVCapacityModel.watermark"], "capacity, latency and length distribution jointly"),
            (20, "capacity/lifetime figures", ["hqsb.runtime.kv.BlockPool.as_dict", "hqsb.runtime.kv.MemoryReconciliation.as_dict", "hqsb.runtime.telemetry.chain_coverage"], "generated from raw events"),
        ]
    ),
)

E07_04 = ExperimentMapping(
    experiment_id="E07-04",
    title="static / continuous / chunked-prefill scheduling and token budgets",
    driver="scripts/runtime/run_e07.py --experiment E07-04",
    fixture="hqsb.runtime.scheduler.SchedulerSpec / RequestTrace",
    steps=_steps(
        [
            (1, "freeze SchedulerSpec", ["hqsb.runtime.scheduler.SchedulerSpec", "hqsb.runtime.specs.build_scheduler_spec", "hqsb.runtime.scheduler.BATCH_MODES", "hqsb.runtime.telemetry.c2_alignment"], "policy/budgets/traces/fairness/congestion bound first; workload semantics stay aligned with C2"),
            (2, "replay a single request", ["hqsb.runtime.scheduler.simulate", "hqsb.runtime.trace.conservation_audit"], "scheduling must not change the E07-01 semantics"),
            (3, "static baseline", ["hqsb.runtime.scheduler.STATIC", "hqsb.runtime.scheduler.SimulationResult.padding_slots"], "padding and empty slots recorded"),
            (4, "continuous baseline", ["hqsb.runtime.scheduler.CONTINUOUS", "hqsb.runtime.scheduler.SchedulerSimulator"], "dynamic join/exit on the same submit trace"),
            (5, "chunked prefill off/on", ["hqsb.runtime.scheduler.CHUNKED_PREFILL", "hqsb.runtime.scheduler.chunk_boundaries"], "other budgets held fixed"),
            (6, "chunk size sweep", ["hqsb.runtime.scheduler.strategy_curve", "hqsb.runtime.scheduler.chunk_coverage_audit"], "small/medium/large and full prefill"),
            (7, "token budget sweep", ["hqsb.runtime.scheduler.SchedulerSimulator.step", "hqsb.runtime.trace.IterationLedgerEntry.budget_utilization"], "under-fed, balanced and saturated"),
            (8, "max sequences sweep", ["hqsb.runtime.scheduler.SchedulerSpec.max_sequences", "hqsb.runtime.scheduler.SimulationResult.max_concurrent_running"], "token vs sequence limits separated"),
            (9, "concurrency sweep", ["hqsb.runtime.scheduler.detect_congestion", "hqsb.runtime.scheduler.CongestionReport"], "up to the congestion/safety boundary"),
            (10, "short/long mixture", ["hqsb.runtime.scheduler.mixed_trace", "hqsb.runtime.comparison.request_classes"], "head-of-line blocking and decode stall"),
            (11, "arrival order swap", ["hqsb.runtime.scheduler.staggered_trace", "hqsb.runtime.scheduler.RequestTrace.trace_hash"], "separates policy effects from arrival luck"),
            (12, "decode priority", ["hqsb.runtime.scheduler.SchedulerSpec.decode_priority", "hqsb.runtime.scheduler.SchedulerSpec.long_prefill_threshold"], "long TTFT vs decode ITL trade-off"),
            (13, "admission/watermark", ["hqsb.runtime.scheduler.SchedulerSpec.admission_reserve_full_isl", "hqsb.runtime.kv.KVCapacityModel.watermark"], "prevents chunked-prefill over-admission and KV thrash"),
            (14, "preemption", ["hqsb.runtime.scheduler.RequestOutcome.preemptions", "hqsb.runtime.kv.OOM_ACTION_ORDER"], "recompute cost and quality"),
            (15, "cancel", ["hqsb.runtime.scheduler.SimRequest.cancel_at_iteration", "hqsb.runtime.trace.IterationLedgerEntry.finished"], "batch/token/KV conservation"),
            (16, "iteration and kernel collection", ["hqsb.runtime.trace.IterationLedgerEntry.kernels", "hqsb.runtime.trace.kernel_join"], "connects to E07-02 for GPU/CPU explanation"),
            (17, "independent-process repetition", ["hqsb.runtime.metrics.paired_effect", "hqsb.runtime.policy_ab.block_schedule"], "ABBA/randomised order"),
            (18, "statistical curves and CI", ["hqsb.runtime.metrics.distribution_summary", "hqsb.runtime.metrics.RunLevelSamples"], "run-level samples, not per-token samples"),
            (19, "locate the congestion point", ["hqsb.runtime.scheduler.detect_congestion", "hqsb.runtime.scheduler.fairness_jain"], "queue/KV/GPU/scheduler evidence together"),
            (20, "choose a conditional policy", ["hqsb.runtime.scheduler.SimulationResult.as_dict", "hqsb.runtime.scheduler.SchedulerSpec.as_dict"], "recommendation bound to a workload distribution"),
        ]
    ),
)

E07_05 = ExperimentMapping(
    experiment_id="E07-05",
    title="prefix cache key, correctness, locality, benefit and eviction",
    driver="scripts/runtime/run_e07.py --experiment E07-05",
    fixture="hqsb.runtime.prefix_cache.PrefixCache / PrefixKey",
    steps=_steps(
        [
            (1, "freeze PrefixSpec", ["hqsb.runtime.prefix_cache.PrefixCacheSpec", "hqsb.runtime.specs.build_prefix_spec", "hqsb.runtime.prefix_cache.CACHE_KEY_FIELDS"], "key fields, hash, block/tail, refcount, eviction, isolation"),
            (2, "cache-off golden", ["hqsb.runtime.prefix_cache.CacheCorrectnessComparison", "hqsb.runtime.parity.compare_greedy"], "logits/tokens/KV summary per fixture"),
            (3, "verify key construction", ["hqsb.runtime.prefix_cache.build_prefix_key", "hqsb.runtime.prefix_cache.PrefixKey.compute_digest", "hqsb.runtime.prefix_cache.token_digest"], "built from final token IDs and model identity"),
            (4, "block boundary hits", ["hqsb.runtime.prefix_cache.PrefixCache.lookup", "hqsb.runtime.prefix_cache.LookupResult"], "P-1/P/P+1 and multi-block"),
            (5, "prefix length sweep", ["hqsb.runtime.scheduler.shared_prefix_trace", "hqsb.runtime.prefix_cache.build_prefix_key"], "same total prompt, varying shared part"),
            (6, "reuse and locality sweep", ["hqsb.runtime.prefix_cache.CacheEntryRecord.reuse_count", "hqsb.runtime.prefix_cache.CacheEntryRecord.last_use_iteration"], "repetition count and spacing"),
            (7, "prefix tree fork", ["hqsb.runtime.prefix_cache.PrefixKey.parent_chain_digest", "hqsb.runtime.prefix_cache.PrefixCache.insert"], "common root, different suffix, refcount check"),
            (8, "wrong-identity negative tests", ["hqsb.runtime.prefix_cache.identity_negative_fixtures", "hqsb.runtime.prefix_cache.IDENTITY_FIELDS"], "model/tokenizer/template/quant/rope/layout"),
            (9, "hash collision fixture", ["hqsb.runtime.prefix_cache.forced_collision_case", "hqsb.runtime.prefix_cache.COLLISION_POLICIES"], "a forced digest collision is rejected, not served"),
            (10, "multi KV group hit", ["hqsb.runtime.prefix_cache.LookupResult.block_group_hits", "hqsb.runtime.prefix_cache.PrefixCache.lookup"], "intersection over groups, no single-group exaggeration"),
            (11, "concurrent readers", ["hqsb.runtime.prefix_cache.CacheEntryRecord.readers", "hqsb.runtime.prefix_cache.PrefixCache.invariant_report"], "shared block while one request finishes/cancels"),
            (12, "eviction pressure", ["hqsb.runtime.prefix_cache.PrefixCache.evict", "hqsb.runtime.kv.select_evictions"], "LRU vs one candidate policy"),
            (13, "eviction race", ["hqsb.runtime.kv.BlockPool.evict", "hqsb.runtime.prefix_cache.CacheEntryRecord.refcount"], "active/refcount protection"),
            (14, "chunked prefill interaction", ["hqsb.runtime.scheduler.chunk_coverage_audit", "hqsb.runtime.prefix_cache.LookupResult"], "cached tokens, chunks and budget conserved"),
            (15, "graph/attention interaction", ["hqsb.runtime.graph_route.GraphSpec.resolve", "hqsb.runtime.graph_route.RouteDecision.actual_mode"], "shape/bucket/actual path after a hit"),
            (16, "lookup and bookkeeping cost", ["hqsb.runtime.prefix_cache.NetSavingModel", "hqsb.runtime.prefix_cache.PrefixCache.lookups"], "CPU, GPU and hash cost"),
            (17, "net TTFT/throughput/memory/energy", ["hqsb.runtime.comparison.recompute_metrics", "hqsb.runtime.metrics.paired_effect"], "multi-request trace with CI"),
            (18, "reset / version invalidate", ["hqsb.runtime.prefix_cache.PrefixCache.reset", "hqsb.runtime.prefix_cache.PrefixCacheSpec.cache_layout_version"], "model or schema change clears/isolates"),
            (19, "long-run hit/evict loop", ["hqsb.runtime.prefix_cache.PrefixCache.invariant_report", "hqsb.runtime.kv.resource_slope"], "no stale reference; cache size bounded"),
            (20, "locality curve", ["hqsb.runtime.prefix_cache.CacheEntryRecord.as_dict", "hqsb.runtime.metrics.distribution_summary"], "hit length/reuse/cache bytes → saved time"),
        ]
    ),
)

E07_06 = ExperimentMapping(
    experiment_id="E07-06",
    title="eager/CUDA Graph and attention backend in prefill and decode",
    driver="scripts/runtime/run_e07.py --experiment E07-06",
    fixture="hqsb.runtime.graph_route.GraphSpec / AttentionCandidate",
    steps=_steps(
        [
            (1, "freeze the graph × attention matrix", ["hqsb.runtime.graph_route.GraphSpec", "hqsb.runtime.graph_route.factorial_matrix", "hqsb.runtime.specs.build_graph_spec"], "scheduler/KV/precision and the trace are locked"),
            (2, "capability probe", ["hqsb.runtime.graph_route.check_attention_support", "hqsb.runtime.graph_route.attention_matrix", "hqsb.runtime.graph_route.AttentionSupport"], "supported/reason per shape"),
            (3, "eager default baseline", ["hqsb.runtime.graph_route.EAGER", "hqsb.runtime.graph_route.phase_split_report"], "prefill/decode and kernel evidence"),
            (4, "eager candidate", ["hqsb.runtime.graph_route.AttentionCandidate", "hqsb.runtime.graph_route.attention_capability_error"], "isolates the attention effect"),
            (5, "graph default", ["hqsb.runtime.graph_route.CUDA_GRAPH", "hqsb.runtime.graph_route.GraphClaimEvidence"], "isolates the submission effect"),
            (6, "graph candidate", ["hqsb.runtime.graph_route.two_factor_analysis", "hqsb.runtime.graph_route.TwoFactorEffects"], "observes the interaction"),
            (7, "single attention oracle", ["hqsb.runtime.graph_route.phase_request_matrix", "hqsb.runtime.parity.compare_greedy"], "real Q/K/V and block table"),
            (8, "block/model/long generation", ["hqsb.runtime.parity.GreedyParityReport", "hqsb.runtime.metrics.distribution_summary"], "four-level quality gate"),
            (9, "multi-input replay", ["hqsb.runtime.graph_route.ReplayRecord", "hqsb.runtime.graph_route.replay_distinctness"], "stale static buffers are detected"),
            (10, "active sequence sweep", ["hqsb.runtime.graph_route.GraphBucketSpec", "hqsb.runtime.graph_route.GraphSpec.resolve"], "bucket coverage and CPU overhead"),
            (11, "context/chunk sweep", ["hqsb.runtime.scheduler.chunk_boundaries", "hqsb.runtime.graph_route.PhaseMetrics"], "prefill and decode reported separately"),
            (12, "block size / KV layout sweep", ["hqsb.runtime.kv.KVGeometry.layout", "hqsb.runtime.kv.KVGeometry.bytes_per_token"], "kept consistent with E07-03"),
            (13, "prefix cache path", ["hqsb.runtime.prefix_cache.PrefixCache.lookup", "hqsb.runtime.graph_route.RouteDecision"], "a cached boundary does not break attention/graph"),
            (14, "bucket switch/reuse", ["hqsb.runtime.graph_route.GraphSpec.out_of_bucket_policy", "hqsb.runtime.graph_route.capture_cost_break_even"], "no address/state pollution from an old graph"),
            (15, "unsupported/fallback", ["hqsb.runtime.graph_route.AttentionSupport.fallback", "hqsb.runtime.graph_route.check_attention_support"], "actual path and reason recorded"),
            (16, "CPU/GPU timeline", ["hqsb.runtime.trace.ClockCalibration", "hqsb.runtime.trace.Span"], "proves launch/attention bottlenecks"),
            (17, "memory pool/workspace", ["hqsb.runtime.graph_route.ReplayRecord.pool_bytes", "hqsb.runtime.kv.MemoryReconciliation"], "benefit and capacity reported together"),
            (18, "independent-process repetition", ["hqsb.runtime.metrics.RunLevelSamples", "hqsb.runtime.policy_ab.block_schedule"], "cold/capture/steady separated"),
            (19, "two-factor analysis", ["hqsb.runtime.graph_route.two_factor_analysis", "hqsb.runtime.graph_route.FactorialCell"], "graph main effect, attention main effect, interaction"),
            (20, "applicability boundary", ["hqsb.runtime.graph_route.phase_split_report", "hqsb.runtime.graph_route.claim_status"], "by phase/shape/hardware, with the P1 claim gate"),
        ]
    ),
)

E07_07 = ExperimentMapping(
    experiment_id="E07-07",
    title="strict single-variable A/B of one runtime policy change",
    driver="scripts/runtime/run_e07.py --experiment E07-07",
    fixture="hqsb.runtime.policy_ab.Adr / SelectionGate",
    steps=_steps(
        [
            (1, "freeze the baseline run", ["hqsb.runtime.experiment.EvidenceManifest", "hqsb.runtime.experiment.git_state"], "environment, commit and raw evidence"),
            (2, "write the ADR/hypothesis", ["hqsb.runtime.policy_ab.Adr", "hqsb.runtime.policy_ab.SelectionGate.require_ok", "hqsb.runtime.policy_ab.SelectionGate"], "frozen before any B result is seen"),
            (3, "design the minimal patch", ["hqsb.runtime.policy_ab.rollback_plan", "hqsb.runtime.policy_ab.CANDIDATE_POLICIES"], "feature flag plus reference path preserved"),
            (4, "code-review the invariants", ["hqsb.runtime.trace.RequestStateMachine.require_clean_finish", "hqsb.runtime.kv.BlockPool.invariant_report"], "state machine, concurrency, error, lifetime"),
            (5, "unit/property tests", ["hqsb.runtime.metrics.paired_effect", "hqsb.runtime.policy_ab.ablation_matrix"], "policy functions and boundaries"),
            (6, "integration correctness", ["hqsb.runtime.parity.compare_greedy", "hqsb.runtime.prefix_cache.CacheCorrectnessComparison"], "replays the E07-01/03/04/05/06 gates"),
            (7, "fault/cancel/OOM", ["hqsb.runtime.failure.frozen_matrix", "hqsb.runtime.failure.failure_matrix_table"], "the patch must not break recovery"),
            (8, "generate A/B artifacts", ["hqsb.runtime.policy_ab.AbIdentity", "hqsb.runtime.policy_ab.require_identity_equal"], "same base commit; diff/hash auditable"),
            (9, "pilot run for debugging only", ["hqsb.runtime.policy_ab.pilot_separation"], "pilot data never enters the final matrix"),
            (10, "freeze the final matrix", ["hqsb.runtime.policy_ab.block_schedule", "hqsb.runtime.policy_ab.schedule_balance"], "configs, traces, repeats, exclusions"),
            (11, "execute ABBA/randomised blocks", ["hqsb.runtime.policy_ab.block_schedule", "hqsb.runtime.experiment.RunDirectory.record_command"], "independent processes and warm state"),
            (12, "normal traces", ["hqsb.runtime.scheduler.homogeneous_trace", "hqsb.runtime.scheduler.mixed_trace"], "the primary deployment distribution"),
            (13, "adversarial traces", ["hqsb.runtime.scheduler.staggered_trace", "hqsb.runtime.failure.frozen_matrix"], "expected regressions and safety boundaries"),
            (14, "internal metrics", ["hqsb.runtime.trace.IterationLedgerEntry.as_dict", "hqsb.runtime.kv.MemoryReconciliation.as_dict"], "the proximate causal chain"),
            (15, "representative profile", ["hqsb.runtime.trace.hot_path_table", "hqsb.runtime.trace.kernel_join"], "CPU/GPU/kernel explanation"),
            (16, "ablation", ["hqsb.runtime.policy_ab.ablation_matrix", "hqsb.runtime.policy_ab.causal_chain_check"], "disabling a sub-mechanism or threshold"),
            (17, "effect size and CI", ["hqsb.runtime.policy_ab.measure_ab", "hqsb.runtime.policy_ab.AbEffect"], "paired, sliced by request class"),
            (18, "causal chain verification", ["hqsb.runtime.policy_ab.causal_chain_check"], "patch → near-cause → phase → end to end"),
            (19, "regression envelope", ["hqsb.runtime.policy_ab.RegressionRow", "hqsb.runtime.policy_ab.regression_envelope"], "which workloads get worse and by how much"),
            (20, "verdict: merge/rollback/PASS_NEGATIVE", ["hqsb.runtime.policy_ab.decide", "hqsb.runtime.policy_ab.DECISIONS"], "the pre-registered primary is not changed afterwards"),
        ]
    ),
)

E07_08 = ExperimentMapping(
    experiment_id="E07-08",
    title="speculative decoding / MTP acceptance, rollback and benefit (P1)",
    driver="scripts/runtime/run_e07.py --experiment E07-08",
    fixture="hqsb.runtime.spec_decode.DraftTargetIdentity / CycleRecord",
    steps=_steps(
        [
            (1, "freeze the algorithm", ["hqsb.runtime.spec_decode.SPEC_ALGORITHMS", "hqsb.runtime.specs.build_spec_decode_contract", "hqsb.runtime.spec_decode.DraftTargetIdentity"], "greedy/strict sampling/MTP plus the commit/rollback rule"),
            (2, "freeze target/draft identity", ["hqsb.runtime.spec_decode.DraftTargetIdentity.compatibility", "hqsb.runtime.spec_decode.DraftTargetIdentity.require_compatible"], "tokenizer/vocab/precision compatibility"),
            (3, "implement the vanilla oracle", ["hqsb.runtime.spec_decode.verify_greedy_exactness", "hqsb.runtime.parity.compare_greedy"], "same target and sampling"),
            (4, "hand-computed single cycle", ["hqsb.runtime.spec_decode.golden_acceptance_cases", "hqsb.runtime.spec_decode.AcceptanceAudit"], "all accept/first reject/none/EOS"),
            (5, "verify acceptance/residual", ["hqsb.runtime.spec_decode.acceptance_probability", "hqsb.runtime.spec_decode.residual_distribution", "hqsb.runtime.spec_decode.residual_is_normalized"], "exact rational normalisation"),
            (6, "verify greedy exactness", ["hqsb.runtime.spec_decode.verify_greedy_exactness", "hqsb.runtime.spec_decode.GREEDY"], "token-by-token, stop and long generation"),
            (7, "verify the sampling distribution", ["hqsb.runtime.spec_decode.distribution_gate", "hqsb.runtime.parity.compare_sampling_distribution"], "multiple seeds and pre-registered statistics"),
            (8, "verify KV commit/rollback", ["hqsb.runtime.spec_decode.kv_commit_rollback_audit", "hqsb.runtime.kv.BlockPool.release"], "block/refcount/token position conservation"),
            (9, "cancel/OOM during a cycle", ["hqsb.runtime.failure.CancelTimeline", "hqsb.runtime.failure.oom_sequence"], "cleanup after an interrupted cycle"),
            (10, "gamma sweep", ["hqsb.runtime.spec_decode.gamma_sweep", "hqsb.runtime.spec_decode.CycleRecord"], "includes gamma=1 as the control"),
            (11, "workload slice sweep", ["hqsb.runtime.spec_decode.workload_slices", "hqsb.runtime.metrics.distribution_summary"], "the acceptance distribution, not only its mean"),
            (12, "batch/context sweep", ["hqsb.runtime.scheduler.SchedulerSpec.max_sequences", "hqsb.runtime.spec_decode.BenefitModel"], "verify parallelism and KV cost"),
            (13, "decompose cycle time", ["hqsb.runtime.spec_decode.CycleRecord.cycle_time_ms", "hqsb.runtime.spec_decode.BenefitModel.as_dict"], "draft/verify/accept/rollback/scheduler"),
            (14, "target-call reduction", ["hqsb.runtime.spec_decode.BenefitModel.target_calls_per_output_token"], "per committed output token"),
            (15, "memory and energy", ["hqsb.runtime.spec_decode.CycleRecord.draft_kv_blocks", "hqsb.runtime.kv.KVGeometry.bytes_for"], "two models plus the temporary buffers"),
            (16, "graph interaction", ["hqsb.runtime.graph_route.GraphSpec.resolve", "hqsb.runtime.graph_route.claim_status"], "fixed gamma/bucket actual path and fallback"),
            (17, "long-run multi-request", ["hqsb.runtime.scheduler.simulate", "hqsb.runtime.kv.resource_slope"], "mixed acceptance in one batch without cross-talk"),
            (18, "independent-process statistics", ["hqsb.runtime.metrics.paired_effect", "hqsb.runtime.metrics.RunLevelSamples"], "paired traces and CI"),
            (19, "model vs measurement", ["hqsb.runtime.spec_decode.BenefitModel.speedup", "hqsb.runtime.comparison.recompute_metrics"], "the cycle formula explains the speedup"),
            (20, "verdict on applicable distributions", ["hqsb.runtime.spec_decode.claim_status", "hqsb.runtime.spec_decode.SpeculativeEvidence"], "no global claim; P1 defaults to NOT_RUN"),
        ]
    ),
)

E07_09 = ExperimentMapping(
    experiment_id="E07-09",
    title="cancel/timeout/OOM/long-context failure matrix and resource recovery",
    driver="scripts/runtime/run_e07.py --experiment E07-09",
    fixture="hqsb.runtime.failure.frozen_matrix / CancelTimeline",
    steps=_steps(
        [
            (1, "freeze FailureSpec", ["hqsb.runtime.failure.FailureCase", "hqsb.runtime.failure.frozen_matrix", "hqsb.runtime.failure.COMMON_INVARIANTS"], "state, injection point, expected action, thresholds"),
            (2, "healthy golden", ["hqsb.runtime.parity.compare_greedy", "hqsb.runtime.prefix_cache.CacheCorrectnessComparison"], "fixed requests, blocks and outputs"),
            (3, "cancel in every state", ["hqsb.runtime.failure.CancelTimeline", "hqsb.runtime.trace.RequestStateMachine"], "waiting/prefill/decode/final race"),
            (4, "chunk boundary cancel", ["hqsb.runtime.scheduler.chunk_coverage_audit", "hqsb.runtime.failure.CancelTimeline"], "no duplicated prompt token, no half-written KV"),
            (5, "shared prefix cancel", ["hqsb.runtime.kv.BlockPool.share", "hqsb.runtime.kv.BlockPool.release"], "refcount and other readers"),
            (6, "timeout in every phase", ["hqsb.runtime.failure.EXPECTED_ACTIONS", "hqsb.runtime.failure.FailureCase.extra_output_policy"], "scheduler observation and output boundary"),
            (7, "over-long context", ["hqsb.runtime.kv.ContextLimitCheck", "hqsb.runtime.failure.ContextAbuseCheck", "hqsb.runtime.failure.context_limit_layers"], "model/runtime/capacity limits; earliest rejection"),
            (8, "admission OOM", ["hqsb.runtime.kv.BlockPool.allocate", "hqsb.runtime.kv.OOM_KINDS"], "no partial allocation; full rollback"),
            (9, "KV allocation OOM", ["hqsb.runtime.kv.oom_action", "hqsb.runtime.failure.oom_sequence"], "block table and free list stay consistent"),
            (10, "execution/workspace OOM", ["hqsb.runtime.failure.oom_sequence", "hqsb.runtime.kv.MAX_OOM_ATTEMPTS"], "isolated process; bounded action order"),
            (11, "prefix/graph pressure", ["hqsb.runtime.prefix_cache.PrefixCache.evict", "hqsb.runtime.graph_route.claim_status"], "evict/fallback ordering"),
            (12, "compile/attention failure", ["hqsb.runtime.graph_route.attention_capability_error", "hqsb.runtime.failure.run_separation_plan"], "deterministic fallback"),
            (13, "same-batch isolation", ["hqsb.runtime.scheduler.SimulationResult.outcomes", "hqsb.runtime.failure.healthy_request_probe"], "a failing request must not affect the others"),
            (14, "concurrent free/evict", ["hqsb.runtime.failure.concurrency_release_scenarios", "hqsb.runtime.kv.BlockPool.invariant_report"], "race, deadlock and refcount"),
            (15, "repeated load-close", ["hqsb.runtime.failure.load_close_scenarios", "hqsb.runtime.adapter.run_load_close_scenarios"], "success/failure/close/double close"),
            (16, "long-run mixture", ["hqsb.runtime.failure.resource_slope_report", "hqsb.runtime.kv.resource_slope"], "fixed-seed stress sequence"),
            (17, "healthy request after a failure", ["hqsb.runtime.failure.healthy_request_probe"], "immediate and multi-round recovery"),
            (18, "resource slopes", ["hqsb.runtime.failure.SegmentSlopes", "hqsb.runtime.failure.leak_blocks_pass"], "KV/device/host/object/cleanup queue"),
            (19, "sanitizer/profiler representative run", ["hqsb.runtime.failure.run_separation_plan"], "kept apart from the ordinary stress run"),
            (20, "failure matrix", ["hqsb.runtime.failure.FailureOutcome", "hqsb.runtime.failure.failure_matrix_table"], "expected/actual, extra tokens, cleanup, recovery"),
        ]
    ),
)

E07_10 = ExperimentMapping(
    experiment_id="E07-10",
    title="fair reference/edge/cloud model-core comparison",
    driver="scripts/runtime/run_e07.py --experiment E07-10",
    fixture="hqsb.runtime.comparison.ComparisonSpec / ComparisonRow",
    steps=_steps(
        [
            (1, "freeze ComparisonSpec", ["hqsb.runtime.comparison.ComparisonSpec", "hqsb.runtime.specs.build_comparison_method", "hqsb.runtime.specs.make_comparison_spec"], "method frozen in YAML; run identity bound at execution time"),
            (2, "import the E07-01 capability results", ["hqsb.runtime.parity.capability_parity_matrix", "hqsb.runtime.request.CapabilityReport.missing_fields"], "missing/emulated/constrained stated explicitly"),
            (3, "verify identity", ["hqsb.runtime.comparison.BackendIdentity", "hqsb.runtime.adapter.RuntimeAdapter.verify_identity"], "model, tokenizer, precision, kernel"),
            (4, "common-denominator correctness", ["hqsb.runtime.comparison.common_denominator_rows", "hqsb.runtime.comparison.COMMON_DENOMINATOR_FEATURES"], "only the features every backend supports"),
            (5, "best-valid correctness", ["hqsb.runtime.comparison.best_valid_rows", "hqsb.runtime.comparison.ComparisonRow.quality_gate_passed"], "each backend's pre-registered best legal configuration"),
            (6, "load/build/compile", ["hqsb.runtime.comparison.PhaseRecord", "hqsb.runtime.comparison.cold_warm_report"], "the full cold phase is reported, not hidden"),
            (7, "single-request six workloads", ["hqsb.runtime.scheduler.homogeneous_trace", "hqsb.runtime.metrics.RequestTimeline"], "prefill/decode/long generation"),
            (8, "static runtime trace", ["hqsb.runtime.scheduler.STATIC", "hqsb.runtime.scheduler.simulate"], "identical batches"),
            (9, "continuous/chunked trace", ["hqsb.runtime.scheduler.CONTINUOUS", "hqsb.runtime.scheduler.CHUNKED_PREFILL"], "only the common feature table; the rest is NA"),
            (10, "prefix trace", ["hqsb.runtime.prefix_cache.PrefixCache.lookup", "hqsb.runtime.scheduler.shared_prefix_trace"], "off/on separated with identical key semantics"),
            (11, "graph/attention", ["hqsb.runtime.graph_route.GraphSpec.resolve", "hqsb.runtime.graph_route.check_attention_support"], "actual path and fallback"),
            (12, "capacity/KV", ["hqsb.runtime.kv.KVCapacityModel", "hqsb.runtime.kv.MemoryReconciliation.explained"], "same memory budget or explicitly different hardware"),
            (13, "cancel/OOM/close", ["hqsb.runtime.failure.failure_matrix_table", "hqsb.runtime.adapter.CancelRecord"], "reliability matrix"),
            (14, "energy", ["hqsb.runtime.comparison.ComparisonRow.metrics", "hqsb.runtime.telemetry.S07ResultFields.energy_joules"], "same hardware comparable; cross-hardware bound to the device"),
            (15, "independent processes / blocked order", ["hqsb.runtime.policy_ab.block_schedule", "hqsb.runtime.metrics.RunLevelSamples"], "raw samples preserved"),
            (16, "unified metric recomputation", ["hqsb.runtime.comparison.recompute_metrics"], "runtime self-reported TPS is never copied"),
            (17, "token denominator audit", ["hqsb.runtime.comparison.token_denominator_audit", "hqsb.runtime.metrics.TokenLedger"], "cached/spec/recompute/cancel cannot inflate throughput"),
            (18, "tier and Pareto", ["hqsb.runtime.comparison.ParetoPoint", "hqsb.runtime.comparison.pareto_front", "hqsb.runtime.comparison.BackendIdentity.tier"], "per hardware × workload; no cross-scenario mixing"),
            (19, "capability/limitations", ["hqsb.runtime.comparison.limitations_report", "hqsb.runtime.comparison.TIER_CLAIMS"], "unobservable and unsupported stated explicitly"),
            (20, "form the S08 backend interface", ["hqsb.runtime.comparison.s08_interface_surface", "hqsb.runtime.adapter.RuntimeAdapter.capability"], "only stable contract fields cross the boundary"),
        ]
    ),
)


EXPERIMENT_MAP: Tuple[ExperimentMapping, ...] = (
    E07_01,
    E07_02,
    E07_03,
    E07_04,
    E07_05,
    E07_06,
    E07_07,
    E07_08,
    E07_09,
    E07_10,
)


def mapping_for(experiment_id: str) -> ExperimentMapping:
    for mapping in EXPERIMENT_MAP:
        if mapping.experiment_id == experiment_id:
            return mapping
    raise KeyError(experiment_id)


def _resolve_attribute(obj: Any, attribute: str, symbol: str) -> Any:
    """Resolve one attribute, accepting dataclass fields as interfaces.

    A frozen dataclass field such as ``ComparisonRow.metrics`` is a real part of
    the contract, but it is an *instance* attribute and therefore not reachable
    through ``getattr`` on the class.  Returning the ``Field`` descriptor keeps
    the map able to point at the precise field a step needs without inventing
    module-level aliases.
    """
    try:
        return getattr(obj, attribute)
    except AttributeError as exc:
        field_obj = _dataclass_field(obj, attribute)
        if field_obj is not None:
            return field_obj
        annotation = _annotated_attribute(obj, attribute)
        if annotation is not None:
            return annotation
        raise AttributeError(f"{symbol!r}: {type(exc).__name__}: {exc}") from exc


def _dataclass_field(obj: Any, attribute: str) -> Optional[Any]:
    import dataclasses

    if not isinstance(obj, type) or not dataclasses.is_dataclass(obj):
        return None
    for field_obj in dataclasses.fields(obj):
        if field_obj.name == attribute:
            return field_obj
    return None


def _annotated_attribute(obj: Any, attribute: str) -> Optional[Any]:
    """Accept a class-body annotation as a declared interface.

    ``BlockPool.events`` is an instance attribute filled in ``__init__``; the
    annotation in the class body is the contract, so the map may point at it.
    """
    if not isinstance(obj, type):
        return None
    annotations = getattr(obj, "__annotations__", {})
    return annotations.get(attribute)


def resolve_interface(symbol: str) -> Any:
    """Import a dotted symbol, supporting nested attributes and dataclass fields.

    ``hqsb.runtime.prefix_cache.PrefixCache.lookup`` is resolved by importing the
    longest importable module prefix and walking the remaining attributes;
    ``hqsb.runtime.comparison.ComparisonRow.metrics`` resolves to the dataclass
    field.  Either way the map cannot reference a symbol that no longer exists.
    """
    parts = symbol.split(".")
    if len(parts) < 2:
        raise AttributeError(f"{symbol!r} is not a dotted path")
    last_error: Optional[BaseException] = None
    for index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:index])
        try:
            obj: Any = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:  # keep looking for a shorter prefix
            last_error = exc
            continue
        for attribute in parts[index:]:
            obj = _resolve_attribute(obj, attribute, symbol)
        return obj
    raise AttributeError(f"{symbol!r}: no importable module prefix ({last_error})")


def resolve_interfaces() -> Dict[str, Any]:
    """Verify that every referenced interface imports (the map cannot rot)."""
    failures: List[Dict[str, str]] = []
    interfaces: List[str] = []
    for mapping in EXPERIMENT_MAP:
        for step in mapping.steps:
            for symbol in step.interfaces:
                interfaces.append(symbol)
                try:
                    resolve_interface(symbol)
                except Exception as exc:  # noqa: BLE001 - report, never crash the audit
                    failures.append(
                        {
                            "experiment": mapping.experiment_id,
                            "step": str(step.step),
                            "symbol": symbol,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    unique = sorted(set(interfaces))
    return {
        "stage": "S07",
        "ok": not failures,
        "experiments": [mapping.experiment_id for mapping in EXPERIMENT_MAP],
        "steps": sum(len(mapping.steps) for mapping in EXPERIMENT_MAP),
        "interfaces": len(unique),
        "references": len(interfaces),
        "failures": failures,
    }


def mapping_table_markdown() -> str:
    """Full step→interface table (regenerated by the driver, never hand-edited)."""
    lines: List[str] = [
        "# S07 实验步骤 → 代码接口对照表",
        "",
        "> 由 `hqsb/runtime/interface_map.py` 生成；"
        "`scripts/runtime/run_e07.py --mode interface-map` 可重新生成。",
        "> 成熟度：M2 = 源码 + 本树自动化测试覆盖；M3+ 需真实 runtime/硬件证据（当前 BLOCKED）。",
        "",
    ]
    for mapping in EXPERIMENT_MAP:
        lines.append(f"## {mapping.experiment_id}：{mapping.title}")
        lines.append("")
        lines.append(f"- 驱动：`{mapping.driver}`")
        if mapping.fixture:
            lines.append(f"- 夹具：`{mapping.fixture}`")
        lines.append("")
        lines.append("| 步骤 | 内容 | 代码接口 | 成熟度 | 说明 |")
        lines.append("|---|---|---|---|---|")
        for step in mapping.steps:
            interfaces = "<br>".join(f"`{item}`" for item in step.interfaces)
            lines.append(
                f"| {step.step} | {step.title} | {interfaces} | {step.maturity} | {step.notes} |"
            )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "E07_01",
    "E07_02",
    "E07_03",
    "E07_04",
    "E07_05",
    "E07_06",
    "E07_07",
    "E07_08",
    "E07_09",
    "E07_10",
    "EXPERIMENT_MAP",
    "ExperimentMapping",
    "StepMapping",
    "mapping_for",
    "mapping_table_markdown",
    "resolve_interface",
    "resolve_interfaces",
]
