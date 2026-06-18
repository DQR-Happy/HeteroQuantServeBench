"""Experiment step → code interface map (the "no missing capability" proof).

Each S08 experiment document lists twenty-four concrete steps.  This module
records, for **every** step, which HQSB interface implements it, and
:func:`resolve_interfaces` verifies that each referenced symbol actually imports
— so the map cannot rot into documentation.  A test
(``tests/unit/serving/test_s08_experiment_scaffolding.py``) runs the
verification.

Maturity labels follow the control plane (M0–M7):

* ``M1`` — source exists (interface implemented, structured errors);
* ``M2`` — covered by automated tests in this tree;
* ``M3``+ — requires a real service/runtime/hardware run, which this repository
  state cannot produce (the S07 P0 and two-Backend prerequisites are unmet), so
  no entry is labelled above M2.

Drivers live in ``scripts/serving/run_e08.py`` and are *not* executed by this
map.  Step counts come from ``docs/stage_experiments/details/S08/E08-*.md``.
"""

from __future__ import annotations

import dataclasses
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


E08_01 = ExperimentMapping(
    experiment_id="E08-01",
    title="completion/chat, SSE, error, disconnect, timeout and cancel conformance",
    driver="scripts/serving/run_e08.py --experiment E08-01",
    fixture="hqsb.serving.dummy_backend.DummyServingBackend",
    steps=_steps(
        [
            (1, "freeze the compatibility statement", ["hqsb.serving.protocol.ProtocolProfile", "hqsb.serving.protocol.ProtocolProfile.schema_hash", "hqsb.serving.specs.KIND_PROTOCOL_PROFILE"], "upstream field, HQSB status, constraint and evidence type per row"),
            (2, "freeze schema and error catalog", ["hqsb.serving.protocol.ErrorCatalog", "hqsb.serving.protocol.ErrorEntry", "hqsb.serving.protocol.ErrorCatalog.validate"], "status/code/retryability/stage and default expansion"),
            (3, "model-free protocol fixture", ["hqsb.serving.dummy_backend.DummyServingBackend", "hqsb.serving.dummy_backend.DummyScript", "hqsb.serving.dummy_backend.frozen_identity"], "HTTP/SSE is isolated from model noise"),
            (4, "validate the loadgen/client parser", ["hqsb.serving.sse.parser_oracle_cases", "hqsb.serving.sse.fragmentation_cases", "hqsb.serving.sse.utf8_boundary_cases"], "split/sticky/truncated/invalid bytes"),
            (5, "completion non-stream normal path", ["hqsb.serving.protocol.validate_response_body", "hqsb.serving.protocol.canonicalize_request", "hqsb.serving.protocol.NormalizedRequest.expanded_defaults"], "default expansion, usage, finish reason, identity"),
            (6, "chat non-stream normal path", ["hqsb.serving.protocol.request_pipeline", "hqsb.serving.protocol.CanonicalBackendRequest", "hqsb.serving.dummy_backend.template_identity"], "roles, template version, token IDs, backend request"),
            (7, "completion stream normal path", ["hqsb.serving.sse.encode_data_frame", "hqsb.serving.sse.SseReassembler.feed", "hqsb.serving.sse.reconstruct"], "raw frames, sequence, write/read times, rebuilt output"),
            (8, "chat stream normal path", ["hqsb.serving.sse.validate_stream", "hqsb.serving.sse.encode_terminal", "hqsb.serving.sse.encode_usage_frame"], "first delta, content deltas, finish, usage, terminal order"),
            (9, "stream/non-stream pairing", ["hqsb.serving.sse.stream_vs_nonstream", "hqsb.serving.sse.require_no_silent_drop"], "greedy pairing on text/token/finish/usage"),
            (10, "stop/EOS/length boundaries", ["hqsb.serving.protocol.ProtocolProfile.finish_reasons", "hqsb.serving.slo.classify_failure", "hqsb.serving.gateway.RequestStateMachine.transition"], "stop across chunks, EOS, max tokens, empty output"),
            (11, "sampling fields take effect", ["hqsb.serving.protocol.canonicalize_request", "hqsb.serving.protocol.SAMPLING_RANGES", "hqsb.serving.gateway.TransitionRecord"], "the actual request spec proves a field is not silently ignored"),
            (12, "unknown/unsupported negative tests", ["hqsb.serving.protocol.validate_request", "hqsb.serving.protocol.ProtocolProfile.field_status", "hqsb.serving.protocol.evaluate_negative_case"], "unknown field, restricted combination, namespaced extension"),
            (13, "malformed/wrong-type negative tests", ["hqsb.serving.protocol.negative_corpus", "hqsb.serving.protocol.parse_json_body", "hqsb.serving.protocol.RequestRejected"], "table-driven JSON/array/null/range/Unicode cases"),
            (14, "model/context identity errors", ["hqsb.serving.protocol.resolve_model_alias", "hqsb.serving.protocol.canonicalize_request", "hqsb.serving.protocol.ModelResolution"], "unknown alias, wrong revision, over-long context"),
            (15, "backend error mapping", ["hqsb.serving.gateway.ServingGateway", "hqsb.serving.dummy_backend.DummyFault", "hqsb.serving.protocol.ErrorCatalog.wire_body"], "unavailable/OOM/timeout/internal status and cause chain"),
            (16, "waiting/admitted cancel", ["hqsb.serving.gateway.ServingGateway.request_cancel", "hqsb.serving.gateway.TransitionRecord", "hqsb.serving.dummy_backend.DummyBackendSession.cancel"], "one terminal transition and zero leaks"),
            (17, "prefill/decode cancel", ["hqsb.serving.pipeline.CANCEL_WINDOWS", "hqsb.serving.pipeline.StreamPipelineRecorder.declare_cancel", "hqsb.serving.pipeline.StreamPipelineRecorder.audit"], "cancelled before output, at first token, mid-stream, terminal race"),
            (18, "graceful disconnect", ["hqsb.serving.clients.ClientBehavior", "hqsb.serving.clients.simulate_client_read", "hqsb.serving.pipeline.DeliveryLedger.audit"], "client stops reading and closes; token delta accounted"),
            (19, "abrupt disconnect", ["hqsb.serving.clients.behavior_matrix", "hqsb.serving.dummy_backend.DummyServingBackend.as_dict", "hqsb.serving.transport.InProcessTransport"], "RST/process exit must not wait for a socket timeout"),
            (20, "deadline propagation", ["hqsb.serving.gateway.GatewayConfig.deadline_header", "hqsb.serving.timing.TimestampLedger.record", "hqsb.serving.slo.classify_failure"], "validation/queue/backend/stream deadline exhaustion"),
            (21, "concurrency isolation", ["hqsb.serving.gateway.RequestStateMachine", "hqsb.serving.pipeline.DeliveryLedger", "hqsb.serving.observability.MetricsRegistry.consistency_check"], "no id/token/error crossover between requests"),
            (22, "close/restart compatibility", ["hqsb.serving.gateway.ServingGateway.begin_drain", "hqsb.serving.gateway.ServingGateway.close", "hqsb.serving.gateway.ServingGateway.readiness"], "protocol stable across drain/restart; old requests keep their epoch"),
            (23, "real model/backend re-verification", ["hqsb.serving.dummy_backend.DummyServingBackend.claim_allowed", "hqsb.serving.router.BackendRecord", "hqsb.serving.experiment.Preregistration"], "runs only after the dummy matrix passes; requires a real Backend"),
            (24, "conformance matrix", ["hqsb.serving.protocol.ConformanceRow", "hqsb.serving.protocol.conformance_matrix", "hqsb.serving.telemetry.project_c6"], "per field/endpoint/lifecycle/backend verdict with raw URI"),
        ]
    ),
)

E08_02 = ExperimentMapping(
    experiment_id="E08-02",
    title="open-loop capacity, SLO cliff and maximum in-SLO goodput",
    driver="scripts/serving/run_e08.py --experiment E08-02",
    fixture="configs/serving/capacity_spec.yaml",
    steps=_steps(
        [
            (1, "freeze CapacitySpec and SLOSpec", ["hqsb.serving.specs.KIND_CAPACITY_SPEC", "hqsb.serving.slo.SLOSpec", "hqsb.serving.experiment.Preregistration"], "workload, window, load grid, failure and selection rules"),
            (2, "verify the E08-01 protocol gate", ["hqsb.serving.protocol.conformance_matrix", "hqsb.serving.slo.classify_failure", "hqsb.serving.gateway.RequestOutcome"], "representative stream/non-stream requests must pass"),
            (3, "freeze service and backend identity", ["hqsb.serving.experiment.EvidenceManifest", "hqsb.serving.experiment.git_state", "hqsb.serving.router.BackendRecord"], "commit/config/model/precision/policy/kernel and environment"),
            (4, "calibrate clock and loadgen", ["hqsb.serving.loadgen.noop_calibration", "hqsb.serving.loadgen.NoopCalibrationPoint", "hqsb.serving.loadgen.service_points_below_client_ceiling"], "no-op server ceiling, lag, parser, CPU/network/fd"),
            (5, "idle baseline", ["hqsb.serving.timing.TimestampLedger.derived", "hqsb.serving.timing.time_conservation", "hqsb.serving.slo.latency_summary"], "single request and very low QPS time decomposition"),
            (6, "closed-loop concurrency pilot", ["hqsb.serving.loadgen.LoadgenSpec", "hqsb.serving.loadgen.closed_loop_records", "hqsb.serving.loadgen.lock_little_check"], "service time, connection need, coarse capacity"),
            (7, "open-loop coarse up sweep", ["hqsb.serving.arrival.generate_arrival_trace", "hqsb.serving.loadgen.run_open_loop", "hqsb.serving.loadgen.SendRecord"], "pre-registered constant or Poisson schedule"),
            (8, "per-point loadgen validity", ["hqsb.serving.loadgen.loadgen_validity", "hqsb.serving.loadgen.ClientResources", "hqsb.serving.loadgen.INVALID_LABEL"], "scheduled/sent rate, lag/drop, client resources"),
            (9, "locate the initial SLO cliff", ["hqsb.serving.slo.LoadPointResult", "hqsb.serving.slo.tail_sample_sufficiency", "hqsb.serving.slo.hysteresis_report"], "goodput/P99/queue slope/reject joined"),
            (10, "fine sweep near the cliff", ["hqsb.serving.slo.max_slo_goodput", "hqsb.serving.slo.LoadPointResult.as_dict", "hqsb.serving.specs.KIND_CAPACITY_SPEC"], "no linear interpolation between sparse points"),
            (11, "high-to-low down sweep", ["hqsb.serving.slo.hysteresis_report", "hqsb.serving.admission.recovery_report", "hqsb.serving.arrival.burst_recovery_metrics"], "cache/queue/thermal hysteresis and slow recovery"),
            (12, "stream curve", ["hqsb.serving.timing.DERIVED_DURATIONS", "hqsb.serving.pipeline.default_layers", "hqsb.serving.observability.SpanRecorder"], "first frame, ITL, buffer and connection resources"),
            (13, "non-stream curve", ["hqsb.serving.protocol.validate_response_body", "hqsb.serving.timing.TimestampLedger.duration_ms", "hqsb.serving.slo.latency_summary"], "full response, E2E and body bytes"),
            (14, "stratify by request length", ["hqsb.serving.fairness.TenantOutcome", "hqsb.serving.slo.class_sample_sufficiency", "hqsb.serving.slo.latency_summary"], "short/long/prefill-heavy/decode-heavy tails"),
            (15, "stratify by prefix hit", ["hqsb.serving.cache_routing.CacheTelemetrySample", "hqsb.serving.cache_routing.prediction_vs_actual", "hqsb.serving.slo.LoadPointResult"], "hit ratio must not drift silently with load"),
            (16, "queue stability check", ["hqsb.serving.arrival.QueueSample", "hqsb.serving.pipeline.resource_slope", "hqsb.serving.observability.classify_time_breakdown"], "depth/age/inflight slope and Little bookkeeping"),
            (17, "locate the bottleneck layer", ["hqsb.serving.observability.SPAN_KINDS", "hqsb.serving.observability.classify_time_breakdown", "hqsb.serving.telemetry.SPAN_CHAIN"], "gateway/queue/runtime/GPU/socket decomposition"),
            (18, "independent process repeats", ["hqsb.serving.experiment.Preregistration.independent_processes", "hqsb.serving.service_ab.stratified_matrix", "hqsb.serving.slo.max_slo_goodput"], "randomised/blocked order with cold/warm records"),
            (19, "long confirmation run at G*", ["hqsb.serving.slo.LoadPointResult", "hqsb.serving.pipeline.long_run_pass", "hqsb.serving.admission.recovery_report"], "prove G* is not a brief cache/thermal accident"),
            (20, "failure/reject funnel re-check", ["hqsb.serving.slo.FunnelCounts", "hqsb.serving.slo.classify_failure", "hqsb.serving.slo.FunnelCounts.audit"], "offered→good conservation with reasons"),
            (21, "statistics and intervals", ["hqsb.serving.slo.percentile_from_raw", "hqsb.serving.service_ab.measure_ab", "hqsb.serving.observability.MetricsRegistry.terminal_events"], "run-level median/CI plus per-request quantiles"),
            (22, "compute maximum SLO goodput", ["hqsb.serving.slo.max_slo_goodput", "hqsb.serving.slo.SLOSpec.require_frozen", "hqsb.serving.slo.token_goodput"], "pre-registered constraints only"),
            (23, "reproduce the selected point", ["hqsb.serving.experiment.RunDirectory.write_verdict", "hqsb.serving.experiment.check_prerequisites", "hqsb.serving.experiment.interface_only_run"], "fresh service process and loadgen"),
            (24, "capacity runbook", ["hqsb.serving.experiment.RunDirectory.write_report_skeleton", "hqsb.serving.slo.hysteresis_report", "hqsb.serving.admission.recovery_report"], "safe region, cliff, warning signs, non-extrapolation"),
        ]
    ),
)

E08_03 = ExperimentMapping(
    experiment_id="E08-03",
    title="constant/Poisson/burst arrival: queue, tail and SLO sensitivity",
    driver="scripts/serving/run_e08.py --experiment E08-03",
    fixture="configs/serving/arrival_spec.yaml",
    steps=_steps(
        [
            (1, "freeze ArrivalSpec", ["hqsb.serving.arrival.ArrivalSpec", "hqsb.serving.arrival.DISTRIBUTIONS", "hqsb.serving.specs.KIND_ARRIVAL_SPEC"], "generators, seeds, mean rate, window, hash"),
            (2, "freeze the payload trace", ["hqsb.serving.loadgen.payload_trace_hash", "hqsb.serving.loadgen.LoadgenSpec.payload_trace_hash", "hqsb.serving.experiment.Preregistration.payload_trace_hash"], "order, length, output, tenant, prefix, sampling"),
            (3, "select E08-02 load bands", ["hqsb.serving.slo.LoadPointResult", "hqsb.serving.arrival.ArrivalSpec.load_bands", "hqsb.serving.slo.max_slo_goodput"], "pre-register low/mid/cliff/overload points"),
            (4, "generate arrival traces offline", ["hqsb.serving.arrival.generate_arrival_trace", "hqsb.serving.arrival.ArrivalTrace.content_hash", "hqsb.serving.arrival.replay_trace"], "all t_sched fixed before the run"),
            (5, "intended-arrival statistics", ["hqsb.serving.arrival.arrival_statistics", "hqsb.serving.arrival.burst_summary", "hqsb.serving.arrival.PEAK_WINDOWS_MS"], "mean, CV, peak windows, burst size/duration"),
            (6, "no-op server calibration", ["hqsb.serving.loadgen.noop_calibration", "hqsb.serving.sse.parser_oracle_cases", "hqsb.serving.clients.verify_script_effects"], "replay all three traces without event-loop distortion"),
            (7, "constant low-load control", ["hqsb.serving.arrival.constant_deltas", "hqsb.serving.timing.TimestampLedger", "hqsb.serving.slo.latency_summary"], "no queue, protocol correct, time闭环"),
            (8, "three distributions at low load", ["hqsb.serving.arrival.compare_arrival_traces", "hqsb.serving.arrival.fidelity_gate", "hqsb.serving.slo.goodput"], "small difference expected: a measurement falsification control"),
            (9, "three distributions at mid load", ["hqsb.serving.arrival.QueueSample", "hqsb.serving.slo.goodput", "hqsb.serving.observability.MetricsRegistry"], "paired payload, queue/batch/tail raw"),
            (10, "three distributions at cliff load", ["hqsb.serving.arrival.burst_recovery_metrics", "hqsb.serving.slo.LoadPointResult", "hqsb.serving.observability.RootCauseClaim"], "arrival spike to P99/SLO violation timing"),
            (11, "safe overload band", ["hqsb.serving.admission.BoundedQueue", "hqsb.serving.admission.PressureStateMachine", "hqsb.serving.admission.recovery_report"], "only with the bounded policy of E08-05"),
            (12, "burst intensity sweep", ["hqsb.serving.arrival.on_off_deltas", "hqsb.serving.arrival.batch_burst_deltas", "hqsb.serving.arrival.burst_summary"], "holding the mean rate while peak/duty change"),
            (13, "burst period sweep", ["hqsb.serving.arrival.on_off_deltas", "hqsb.serving.arrival.burst_recovery_metrics", "hqsb.serving.pipeline.CANCEL_WINDOWS"], "shorter than a prefill/decode cycle vs longer than recovery"),
            (14, "length-independent arrival", ["hqsb.serving.arrival.ArrivalSpec.comparability", "hqsb.serving.loadgen.LoadgenSpec", "hqsb.serving.experiment.Preregistration.load_points"], "keep arrival independent of payload class"),
            (15, "correlated burst extension", ["hqsb.serving.fairness.hol_trace", "hqsb.serving.fairness.HOL_CONSTRUCTS", "hqsb.serving.arrival.BurstinessPoint"], "long-request/tenant bursts as a compound scenario"),
            (16, "actual arrival fidelity gate", ["hqsb.serving.arrival.fidelity_gate", "hqsb.serving.arrival.ActualArrival", "hqsb.serving.loadgen.loadgen_validity"], "per-run intended/actual rate, lag/drop, peak"),
            (17, "rebuild the queue timeline", ["hqsb.serving.arrival.QueueSample", "hqsb.serving.slo.FunnelCounts", "hqsb.serving.pipeline.resource_slope"], "recompute depth and oldest age from events"),
            (18, "correlate runtime iterations", ["hqsb.serving.observability.Span.iteration", "hqsb.serving.telemetry.KIND_TO_EVENT_TYPE", "hqsb.serving.observability.classify_time_breakdown"], "batch, prefill/decode interference, KV, graph path"),
            (19, "tail and goodput statistics", ["hqsb.serving.slo.percentile_from_raw", "hqsb.serving.slo.goodput", "hqsb.serving.slo.tail_sample_sufficiency"], "P50/P95/P99, violations and goodput by class"),
            (20, "recovery metrics", ["hqsb.serving.arrival.burst_recovery_metrics", "hqsb.serving.admission.recovery_report", "hqsb.serving.arrival.QueueSample"], "burst end to queue/SLO/resource baseline"),
            (21, "independent process repeats", ["hqsb.serving.experiment.Preregistration.independent_processes", "hqsb.serving.arrival.ArrivalSpec.seeds", "hqsb.serving.service_ab.stratified_matrix"], "randomised distribution order, keep every failure"),
            (22, "paired statistical effects", ["hqsb.serving.service_ab.measure_ab", "hqsb.serving.slo.latency_summary", "hqsb.serving.service_ab.stratified_matrix"], "Poisson−constant and burst−constant run-level deltas"),
            (23, "mechanism chain verification", ["hqsb.serving.service_ab.mechanism_mediation_check", "hqsb.serving.arrival.burst_recovery_metrics", "hqsb.serving.observability.evidence_consistency"], "arrival variance → queue/batch → tail order"),
            (24, "arrival sensitivity map", ["hqsb.serving.arrival.arrival_sensitivity_map", "hqsb.serving.arrival.BurstinessPoint", "hqsb.serving.experiment.RunDirectory.write_report_skeleton"], "safe/sensitive/reject zones and non-extrapolation"),
        ]
    ),
)

E08_04 = ExperimentMapping(
    experiment_id="E08-04",
    title="mixed short/long, priority and multi-tenant fairness (HOL, starvation)",
    driver="scripts/serving/run_e08.py --experiment E08-04",
    fixture="configs/serving/workload_suite.yaml",
    steps=_steps(
        [
            (1, "freeze the tenant/class spec", ["hqsb.serving.fairness.TenantOutcome", "hqsb.serving.specs.KIND_WORKLOAD_SUITE", "hqsb.serving.slo.SLOSpec"], "classes, weights, priority, deadlines, lengths, SLO"),
            (2, "freeze the cost function", ["hqsb.serving.fairness.CostModel", "hqsb.serving.fairness.CostModel.cost", "hqsb.serving.policies.QueueEntry.estimated_cost"], "alpha/beta/gamma, prefix/recompute/cancel accounting"),
            (3, "deterministic policy interface", ["hqsb.serving.policies.QueuePolicy", "hqsb.serving.policies.PolicyDecision", "hqsb.serving.policies.policy_interface_audit"], "one input/output/decision log for every policy"),
            (4, "policy unit and property tests", ["hqsb.serving.policies.WeightedFairPolicy.normalized_service", "hqsb.serving.policies.DecisionLog.work_conserving_audit", "hqsb.serving.policies.build_policy"], "counter, weight, idle/new tenant, tie break, cancel"),
            (5, "per-class isolated baseline", ["hqsb.serving.fairness.slowdown", "hqsb.serving.fairness.isolated_baseline_check", "hqsb.serving.slo.class_sample_sufficiency"], "matched TTFT/TPOT/E2E/goodput for slowdown"),
            (6, "homogeneous control", ["hqsb.serving.policies.policy_interface_audit", "hqsb.serving.fairness.work_conserving_utilization", "hqsb.serving.fairness.fairness_report"], "equal-cost requests: policies should agree"),
            (7, "long-first HOL trace", ["hqsb.serving.fairness.hol_trace", "hqsb.serving.fairness.HOL_CONSTRUCTS", "hqsb.serving.policies.DecisionLog"], "a long request first, then a short burst"),
            (8, "short-stream with long insert", ["hqsb.serving.fairness.hol_trace", "hqsb.serving.policies.StrictPriorityPolicy", "hqsb.serving.fairness.max_continuous_wait"], "is the long request starved by a steady short stream"),
            (9, "prefill-heavy/decode-heavy mix", ["hqsb.serving.fairness.hol_trace", "hqsb.serving.observability.Span.kind", "hqsb.serving.slo.latency_summary"], "batch composition vs TTFT/TPOT interference"),
            (10, "equal-weight two tenants", ["hqsb.serving.fairness.fairness_report", "hqsb.serving.fairness.jain_index", "hqsb.serving.fairness.ideal_service_lag"], "backlogged share, lag, Jain, max wait"),
            (11, "unequal-weight tenants", ["hqsb.serving.fairness.normalized_service", "hqsb.serving.policies.WeightedFairPolicy", "hqsb.serving.fairness.fairness_report"], "1:2 / 1:4 entitlements, normalized service not raw tokens"),
            (12, "strict priority", ["hqsb.serving.policies.StrictPriorityPolicy", "hqsb.serving.fairness.starvation_events", "hqsb.serving.fairness.priority_inversion_report"], "high-priority gain plus low-priority starvation"),
            (13, "aging/minimum share", ["hqsb.serving.policies.WeightedFairPolicy.normalized_service", "hqsb.serving.fairness.fairness_report", "hqsb.serving.fairness.work_conserving_utilization"], "threshold sweep must not break work conservation"),
            (14, "unknown OSL prediction", ["hqsb.serving.policies.prediction_report", "hqsb.serving.admission.AdmissionCostModel.estimate", "hqsb.serving.fairness.CostModel"], "online features only; final error recorded"),
            (15, "malicious max_tokens", ["hqsb.serving.policies.max_tokens_abuse_check", "hqsb.serving.admission.AdmissionRequest.max_tokens", "hqsb.serving.slo.RequestSLOInput"], "declared huge, stops early: charge must not be inflatable"),
            (16, "prefix hit/miss accounting", ["hqsb.serving.fairness.cache_fairness_split", "hqsb.serving.cache_routing.prefix_saved_prefill_ms", "hqsb.serving.fairness.fairness_report"], "logical vs computed vs charged service"),
            (17, "cancel/timeout accounting", ["hqsb.serving.policies.QueuePolicy.refund", "hqsb.serving.pipeline.DeliveryLedger.waste", "hqsb.serving.fairness.fairness_report"], "consumed cost kept, unexecuted refunded, counters consistent"),
            (18, "burst tenant", ["hqsb.serving.fairness.fairness_report", "hqsb.serving.fairness.ideal_service_lag", "hqsb.serving.arrival.burst_recovery_metrics"], "one tenant's burst must not permanently hurt others"),
            (19, "capacity-neighbourhood replay", ["hqsb.serving.slo.LoadPointResult", "hqsb.serving.service_ab.stratified_matrix", "hqsb.serving.fairness.fairness_report"], "compare policies at mid/cliff, not only idle"),
            (20, "scheduler/runtime trace", ["hqsb.serving.observability.SpanRecorder", "hqsb.serving.policies.DecisionLog.as_rows", "hqsb.serving.telemetry.to_trace_events"], "connect decisions to scheduled tokens/batch/KV"),
            (21, "independent repeats", ["hqsb.serving.service_ab.block_schedule", "hqsb.serving.service_ab.schedule_balance", "hqsb.serving.experiment.Preregistration.independent_processes"], "randomise FIFO/priority/fair order"),
            (22, "fairness and guardrails", ["hqsb.serving.fairness.fairness_report", "hqsb.serving.fairness.DEFAULT_GUARDRAILS", "hqsb.serving.fairness.priority_inversion_report"], "Jain, lag, starvation, slowdown, per-class SLO"),
            (23, "adversarial re-check", ["hqsb.serving.fairness.hol_trace", "hqsb.serving.fairness.fairness_report", "hqsb.serving.policies.DecisionLog.work_conserving_audit"], "worst length/tenant/order case: no deadlock or starvation"),
            (24, "conditional policy map", ["hqsb.serving.service_ab.stratified_matrix", "hqsb.serving.service_ab.service_decision", "hqsb.serving.experiment.RunDirectory.write_report_skeleton"], "which policy for which load/mix/entitlement"),
        ]
    ),
)

E08_05 = ExperimentMapping(
    experiment_id="E08-05",
    title="admission, token budget, backpressure, reject and overload recovery",
    driver="scripts/serving/run_e08.py --experiment E08-05",
    fixture="configs/serving/admission_spec.yaml",
    steps=_steps(
        [
            (1, "freeze OverloadSpec", ["hqsb.serving.admission.PressureSpec", "hqsb.serving.admission.WAVEFORMS", "hqsb.serving.specs.KIND_ADMISSION_SPEC"], "G*, waveforms, thresholds, SLO, hard guards, recovery"),
            (2, "pressure state machine test", ["hqsb.serving.admission.PressureStateMachine", "hqsb.serving.admission.PressureSignal", "hqsb.serving.admission.PressureStateMachine.oscillation_report"], "enter/exit/hysteresis/min dwell on synthetic signals"),
            (3, "admission cost unit test", ["hqsb.serving.admission.AdmissionCostModel.estimate", "hqsb.serving.admission.AdmissionEstimate", "hqsb.serving.admission.AdmissionRequest.history_mean_output"], "length, prefix hit, max output, deadline, unknown OSL"),
            (4, "E08-02 safe baseline", ["hqsb.serving.slo.max_slo_goodput", "hqsb.serving.admission.recovery_report", "hqsb.serving.pipeline.resource_slope"], "stable queue/resource/SLO at G* as control"),
            (5, "very-large queue counter-example", ["hqsb.serving.admission.UnboundedQueueGuard", "hqsb.serving.admission.UnboundedQueueGuard.guard", "hqsb.serving.admission.SAFETY_INVARIANTS"], "short, guarded, killed at the threshold"),
            (6, "bounded request queue", ["hqsb.serving.admission.BoundedQueue", "hqsb.serving.admission.AdmissionDecision", "hqsb.serving.slo.FunnelCounts"], "fixed request cap: reject, tail, goodput, class bias"),
            (7, "token-aware admission", ["hqsb.serving.admission.BoundedQueue.try_admit", "hqsb.serving.admission.AdmissionCostModel.calibration", "hqsb.serving.admission.AdmissionEstimate.estimated_kv_tokens"], "token/KV budget and OOM headroom"),
            (8, "deadline-aware admission", ["hqsb.serving.admission.AdmissionEstimate.deadline_slack_ms", "hqsb.serving.slo.classify_failure", "hqsb.serving.slo.FunnelCounts.add_reason"], "false accept / false reject review"),
            (9, "ramp overload", ["hqsb.serving.admission.overload_plan", "hqsb.serving.admission.PressureStateMachine.observe", "hqsb.serving.admission.recovery_report"], "cross SOFT/HARD/SHEDDING, repeatable cliff"),
            (10, "step-up overload", ["hqsb.serving.admission.overload_plan", "hqsb.serving.admission.AdmissionDecision", "hqsb.serving.slo.LoadPointResult"], "first reject, queue peak, SLO damage, resource peak"),
            (11, "burst absorption", ["hqsb.serving.arrival.on_off_deltas", "hqsb.serving.admission.BoundedQueue.snapshot", "hqsb.serving.arrival.burst_recovery_metrics"], "which transient the queue can absorb"),
            (12, "sustained overload", ["hqsb.serving.admission.BoundedQueue", "hqsb.serving.slo.FunnelCounts.audit", "hqsb.serving.pipeline.resource_slope"], "admitted≈completed, bounded queue, stable reject"),
            (13, "step-down recovery", ["hqsb.serving.admission.recovery_report", "hqsb.serving.admission.PressureStateMachine.oscillation_report", "hqsb.serving.slo.LoadPointResult"], "time back to NORMAL for queue/resource/SLO/state"),
            (14, "backend capacity drop", ["hqsb.serving.circuit.CircuitBreaker", "hqsb.serving.admission.PressureStateMachine", "hqsb.serving.router.Telemetry"], "same offered load, lower service rate"),
            (15, "priority/tenant shedding", ["hqsb.serving.fairness.fairness_report", "hqsb.serving.admission.REJECT_CODES", "hqsb.serving.policies.StrictPriorityPolicy"], "low weight rejected first without undeclared discrimination"),
            (16, "invalid vs overload cost", ["hqsb.serving.protocol.ErrorCatalog.entry", "hqsb.serving.protocol.ErrorEntry.backend_request_created", "hqsb.serving.admission.safety_audit"], "rejections are cheap and create no backend state"),
            (17, "429/503/Retry-After conformance", ["hqsb.serving.admission.retry_after_header", "hqsb.serving.protocol.ErrorCatalog.wire_body", "hqsb.serving.admission.REJECT_CODES"], "per reject reason: status, body, header, retryability"),
            (18, "retry-off main experiment", ["hqsb.serving.loadgen.LoadgenSpec.no_retry", "hqsb.serving.admission.RetryBudget.amplification", "hqsb.serving.slo.LoadPointResult"], "the true overload curve without positive feedback"),
            (19, "controlled retry experiment", ["hqsb.serving.admission.RetryBudget.decide", "hqsb.serving.admission.RetryDecision", "hqsb.serving.admission.retry_after_header"], "no-backoff vs exponential+jitter vs budget"),
            (20, "attempt amplification", ["hqsb.serving.admission.RetryBudget.amplification", "hqsb.serving.faults.Attempt", "hqsb.serving.faults.attempt_lineage_audit"], "attempts, duplicate/late work, backend load per original"),
            (21, "slow-client pressure", ["hqsb.serving.pipeline.LayerState.pressure_ratio", "hqsb.serving.clients.behavior_matrix", "hqsb.serving.admission.PressureSignal.buffered_socket_bytes"], "socket buffered bytes as a pressure signal"),
            (22, "independent repeats and resets", ["hqsb.serving.experiment.Preregistration.independent_processes", "hqsb.serving.admission.recovery_report", "hqsb.serving.circuit.CircuitBreaker.counters"], "queue/circuit/cache must not leak between runs"),
            (23, "goodput/recovery Pareto", ["hqsb.serving.slo.max_slo_goodput", "hqsb.serving.fairness.fairness_report", "hqsb.serving.admission.recovery_report"], "compare reject rate, G*, P99, resource, recovery, fairness"),
            (24, "overload runbook", ["hqsb.serving.experiment.RunDirectory.write_report_skeleton", "hqsb.serving.admission.SAFETY_INVARIANTS", "hqsb.serving.admission.recovery_report"], "thresholds, reasons, expected degradation, stop rules"),
        ]
    ),
)

E08_06 = ExperimentMapping(
    experiment_id="E08-06",
    title="multi-backend routing by model/capability/health/SLO/cost",
    driver="scripts/serving/run_e08.py --experiment E08-06",
    fixture="configs/serving/routing_spec.yaml",
    steps=_steps(
        [
            (1, "freeze RoutingSpec", ["hqsb.serving.router.ScoreSpec", "hqsb.serving.router.HARD_FILTER_ORDER", "hqsb.serving.specs.KIND_ROUTING_SPEC"], "registry schema, hard filters, score, tie break, TTL, fallback"),
            (2, "import S07 capability/identity", ["hqsb.serving.router.BackendRecord", "hqsb.serving.router.BackendRecord.model_identity", "hqsb.runtime.comparison.s08_interface_surface"], "artifact, actual precision/kernel, limits"),
            (3, "two real backends", ["hqsb.serving.router.BackendRegistry.register", "hqsb.serving.router.RegistrySnapshot.ids", "hqsb.serving.dummy_backend.DummyServingBackend"], "two instances/handles, load/warmup/readiness"),
            (4, "registry schema and atomicity", ["hqsb.serving.router.BackendRegistry.replace_generation", "hqsb.serving.router.RegistrySnapshot", "hqsb.serving.router.ScoreSpec.telemetry_ttl_ms"], "generation update, duplicate id, bad capability, stale epoch"),
            (5, "hard-filter unit tests", ["hqsb.serving.router.hard_filter", "hqsb.serving.router.FilterOutcome", "hqsb.serving.router.RouteRequest"], "positive/negative candidate sets per filter"),
            (6, "score unit/property tests", ["hqsb.serving.router.score_candidates", "hqsb.serving.router.ScoredCandidate.normalized", "hqsb.serving.router.ScoreFeature.missing_policy"], "units, normalisation, NaN/missing, tie break"),
            (7, "model alias/artifact routing", ["hqsb.serving.router.RouteRequest.model_identity", "hqsb.serving.router.route", "hqsb.serving.router.route_vs_actual"], "different alias/revision → actual response identity"),
            (8, "precision/quality routing", ["hqsb.serving.router.RouteRequest.quality_class", "hqsb.serving.router.hard_filter", "hqsb.serving.protocol.ErrorCatalog.entry"], "an unapproved quantized backend is not a candidate"),
            (9, "context capability routing", ["hqsb.serving.router.RouteRequest.prompt_tokens", "hqsb.serving.router.BackendRecord.max_context_tokens", "hqsb.serving.protocol.canonicalize_request"], "short/long/over-limit inputs"),
            (10, "protocol feature routing", ["hqsb.serving.router.BackendRecord.features", "hqsb.serving.router.RouteRequest.stream", "hqsb.serving.router.RouteRequest.logprobs"], "stream/cancel/logprobs go only to exact support"),
            (11, "healthy baseline routing", ["hqsb.serving.router.route", "hqsb.serving.router.RouteDecision.scored", "hqsb.serving.slo.LoadPointResult"], "route distribution, score, SLO"),
            (12, "single-backend high load", ["hqsb.serving.router.Telemetry", "hqsb.serving.router.score_candidates", "hqsb.serving.router.route_vs_actual"], "least-load/SLO score shifts traffic without breaking identity"),
            (13, "dynamic capability change", ["hqsb.serving.router.BackendRegistry.replace_generation", "hqsb.serving.router.telemetry_freshness_report", "hqsb.serving.router.RouteDecision.selected_route_epoch"], "withdraw/add a feature: TTL, epoch, in-flight boundary"),
            (14, "readiness/model reload", ["hqsb.serving.circuit.HealthState", "hqsb.serving.circuit.HealthState.classification", "hqsb.serving.router.RouteDecision.selected_model_epoch"], "a new epoch takes no traffic before warmup"),
            (15, "health degrade/circuit open", ["hqsb.serving.circuit.CircuitBreaker", "hqsb.serving.circuit.breaker_matrix", "hqsb.serving.router.hard_filter"], "exclusion, stale window, recovery probe"),
            (16, "telemetry stale/missing", ["hqsb.serving.router.telemetry_freshness_report", "hqsb.serving.router.score_candidates", "hqsb.serving.router.ScoreSpec.missing_telemetry_never_zero"], "frozen/delayed telemetry → conservative rule"),
            (17, "tight/relaxed SLO", ["hqsb.serving.router.RouteRequest.deadline_ns", "hqsb.serving.router.RouteDecision", "hqsb.serving.router.ScoreSpec.slo_feasibility_terms"], "feasibility prediction and selection change"),
            (18, "cost/SLO conflict", ["hqsb.serving.router.BackendRecord.cost_units", "hqsb.serving.router.route", "hqsb.serving.slo.LoadPointResult"], "cheap-slow vs. expensive-fast → conditional Pareto"),
            (19, "no feasible backend", ["hqsb.serving.router.RouteDecision.no_feasible", "hqsb.serving.router.route", "hqsb.serving.protocol.ErrorCatalog.entry"], "stable reject, no silent fallback"),
            (20, "allowed degradation", ["hqsb.serving.router.fallback_plan", "hqsb.serving.router.PERMISSION_REQUIRED_LEVELS", "hqsb.serving.router.RouteDecision.fallback_level"], "only with explicit permission; actual identity visible"),
            (21, "route-vs-actual conservation", ["hqsb.serving.router.route_vs_actual", "hqsb.serving.gateway.RequestOutcome.backend_id", "hqsb.serving.gateway.RequestOutcome.model_epoch"], "per request: selected vs executed instance/epoch"),
            (22, "concurrency and long run", ["hqsb.serving.router.BackendRegistry.replace_generation", "hqsb.serving.circuit.transition_is_recomputable", "hqsb.serving.pipeline.resource_slope"], "registry/health updates racing with requests"),
            (23, "independent repeats and calibration", ["hqsb.serving.router.ScoredCandidate.slo_feasible", "hqsb.serving.service_ab.measure_ab", "hqsb.serving.service_ab.stratified_matrix"], "predicted-vs-actual SLO and CI"),
            (24, "routing decision matrix", ["hqsb.serving.router.RouteDecision.as_dict", "hqsb.serving.router.fallback_plan", "hqsb.serving.experiment.RunDirectory.write_report_skeleton"], "candidates, selection, result, limits, runbook"),
        ]
    ),
)

E08_07 = ExperimentMapping(
    experiment_id="E08-07",
    title="prefix locality, cache-aware routing and load skew",
    driver="scripts/serving/run_e08.py --experiment E08-07",
    fixture="configs/serving/cache_routing_spec.yaml",
    steps=_steps(
        [
            (1, "freeze CacheRoutingSpec", ["hqsb.serving.cache_routing.JointWeights", "hqsb.serving.cache_routing.CACHE_POLICIES", "hqsb.serving.specs.KIND_CACHE_ROUTING_SPEC"], "identity, telemetry, four policies, score, TTL, workload, SLO"),
            (2, "import S07 cache correctness", ["hqsb.serving.cache_routing.IDENTITY_FIELDS", "hqsb.runtime.prefix_cache.PrefixCache", "hqsb.serving.cache_routing.block_identity_second_check"], "key/block/refcount/eviction already pass; the service does not patch them"),
            (3, "token-level prefix corpus", ["hqsb.serving.cache_routing.PrefixEntry", "hqsb.serving.cache_routing.PrefixMatcher.insert", "hqsb.serving.loadgen.payload_trace_hash"], "exact/partial/branch/near-match/wrong-version fixtures hashed"),
            (4, "fix the initial cache state", ["hqsb.serving.cache_routing.CacheTelemetrySample.cache_epoch", "hqsb.serving.cache_routing.TelemetryFaultInjector", "hqsb.serving.cache_routing.telemetry_freshness"], "cold/seeded/warm inventory per instance"),
            (5, "verify the router prefix matcher", ["hqsb.serving.cache_routing.PrefixMatcher.match", "hqsb.serving.cache_routing.matcher_oracle", "hqsb.serving.sse.utf8_boundary_cases"], "matched length vs. an independent token oracle"),
            (6, "verify the backend second check", ["hqsb.serving.cache_routing.block_identity_second_check", "hqsb.serving.cache_routing.identity_negative_fixtures", "hqsb.serving.cache_routing.version_invalidation_check"], "a forged/colliding fingerprint must not hit"),
            (7, "random/round-robin baseline", ["hqsb.serving.cache_routing.cache_route", "hqsb.serving.cache_routing.CacheRouteDecision", "hqsb.serving.slo.goodput"], "accidental hits, load and prefill across the workload"),
            (8, "least-load baseline", ["hqsb.serving.cache_routing.cache_route", "hqsb.serving.cache_routing.skew_report", "hqsb.serving.slo.goodput"], "low skew with lower locality"),
            (9, "pure affinity counter-example", ["hqsb.serving.cache_routing.cache_route", "hqsb.serving.cache_routing.skew_report", "hqsb.serving.fairness.fairness_report"], "maximum hit rate: hotspot queue, tail, eviction, fairness"),
            (10, "joint policy main experiment", ["hqsb.serving.cache_routing.cache_route", "hqsb.serving.cache_routing.CacheRouteDecision.candidates", "hqsb.serving.cache_routing.JointWeights.as_dict"], "frozen score with per-request predicted cost"),
            (11, "prefix length sweep", ["hqsb.serving.cache_routing.prefix_saved_prefill_ms", "hqsb.serving.cache_routing.NetValueModel", "hqsb.serving.timing.TimestampLedger"], "does saved prefill cover lookup and queue cost"),
            (12, "reuse count/distance sweep", ["hqsb.serving.cache_routing.PrefixEntry.last_access_ns", "hqsb.serving.cache_routing.PrefixEntry.frequency", "hqsb.serving.cache_routing.NetValueModel"], "one reuse to hot prefix, eviction interaction"),
            (13, "cache capacity sweep", ["hqsb.serving.cache_routing.CacheTelemetrySample.pressure_ratio", "hqsb.serving.cache_routing.skew_report", "hqsb.serving.slo.LoadPointResult"], "pressure, pollution, eviction, recompute, oscillation"),
            (14, "zero-locality falsification", ["hqsb.serving.cache_routing.zero_locality_falsification", "hqsb.serving.cache_routing.cache_route", "hqsb.serving.service_ab.measure_ab"], "the joint policy must not invent a benefit"),
            (15, "hot-prefix skew", ["hqsb.serving.cache_routing.skew_report", "hqsb.serving.fairness.fairness_report", "hqsb.serving.cache_routing.cache_route"], "one hot prefix saturating an instance"),
            (16, "burst locality", ["hqsb.serving.arrival.on_off_deltas", "hqsb.serving.cache_routing.TelemetryFaultInjector", "hqsb.serving.cache_routing.cache_route"], "is affinity worth it for a short burst"),
            (17, "telemetry delay/drop/reorder", ["hqsb.serving.cache_routing.TelemetryFaultInjector.apply", "hqsb.serving.cache_routing.contradictory_updates", "hqsb.serving.cache_routing.telemetry_freshness"], "stale/missing/contradictory updates: safety and degradation"),
            (18, "version/model invalidation", ["hqsb.serving.cache_routing.version_invalidation_check", "hqsb.serving.cache_routing.CacheIdentity.digest", "hqsb.serving.cache_routing.TenantSharingPolicy"], "old cache must be unmatchable"),
            (19, "tenant sharing policy", ["hqsb.serving.cache_routing.TenantSharingPolicy.allows", "hqsb.serving.cache_routing.tenant_isolation_fixture", "hqsb.serving.observability.redact_text"], "same/cross tenant fixtures; no prompt leakage"),
            (20, "cancel/evict race", ["hqsb.serving.cache_routing.PrefixMatcher.evict", "hqsb.serving.pipeline.CANCEL_WINDOWS", "hqsb.serving.cache_routing.CachePredictionOutcome"], "the backend must miss/recover, not mis-reference"),
            (21, "cliff-load paired repeats", ["hqsb.serving.service_ab.block_schedule", "hqsb.serving.service_ab.measure_ab", "hqsb.serving.experiment.Preregistration.independent_processes"], "same arrival/payload trace, randomised policies"),
            (22, "prediction calibration", ["hqsb.serving.cache_routing.prediction_vs_actual", "hqsb.serving.cache_routing.CachePredictionOutcome.actual_matched_tokens", "hqsb.serving.cache_routing.telemetry_freshness"], "predicted hit/saved time vs. actual, stale rate"),
            (23, "causal chain verification", ["hqsb.serving.service_ab.mechanism_mediation_check", "hqsb.serving.cache_routing.NetValueModel", "hqsb.serving.observability.evidence_consistency"], "route→reuse→prefill→TTFT/goodput/fairness"),
            (24, "locality-load policy map", ["hqsb.serving.cache_routing.skew_report", "hqsb.serving.service_ab.stratified_matrix", "hqsb.serving.experiment.RunDirectory.write_report_skeleton"], "policy applicability by load/prefix/reuse/capacity"),
        ]
    ),
)

E08_08 = ExperimentMapping(
    experiment_id="E08-08",
    title="slow reader, disconnect, batch cancel and graceful shutdown lifecycle",
    driver="scripts/serving/run_e08.py --experiment E08-08",
    fixture="configs/serving/stream_lifecycle_spec.yaml",
    steps=_steps(
        [
            (1, "freeze StreamLifecycleSpec", ["hqsb.serving.pipeline.LayerState", "hqsb.serving.pipeline.CANCEL_WINDOWS", "hqsb.serving.specs.KIND_STREAM_LIFECYCLE_SPEC"], "per-layer buffers, timeouts, linearisation, drain, resources"),
            (2, "verify the E08-01 SSE oracle", ["hqsb.serving.sse.validate_stream", "hqsb.serving.sse.reconstruct", "hqsb.serving.sse.require_no_silent_drop"], "fast reader: raw frames, concatenation, usage, terminal"),
            (3, "instrument the token/byte pipeline", ["hqsb.serving.pipeline.FramePipelineEvent", "hqsb.serving.pipeline.FRAME_STAGES", "hqsb.serving.pipeline.StreamPipelineRecorder.record"], "ready→queue→serialize→write→client-read per frame"),
            (4, "calibrate the client reader", ["hqsb.serving.clients.behavior_matrix", "hqsb.serving.clients.simulate_client_read", "hqsb.serving.clients.verify_script_effects"], "prove delay/bandwidth/stall/RST really happen"),
            (5, "fast-reader baseline", ["hqsb.serving.pipeline.summarize_layers", "hqsb.serving.timing.DERIVED_DURATIONS", "hqsb.serving.slo.latency_summary"], "minimal buffer, write latency, TTFT/ITL, resource baseline"),
            (6, "fixed-delay slow reader", ["hqsb.serving.clients.ClientBehavior.read_delay_ms", "hqsb.serving.pipeline.LayerState.pressure_ratio", "hqsb.serving.pipeline.StreamPipelineRecorder.audit"], "which layer accumulates the backlog"),
            (7, "bandwidth-limited slow reader", ["hqsb.serving.clients.ClientBehavior.bandwidth_bytes_per_s", "hqsb.serving.sse.utf8_boundary_cases", "hqsb.serving.pipeline.default_layers"], "bytes/s limit with small/large chunks and CJK payloads"),
            (8, "periodic stall/resume", ["hqsb.serving.clients.ClientBehavior.stall_period_ms", "hqsb.serving.clients.simulate_client_read", "hqsb.serving.pipeline.CANCEL_WINDOWS"], "short pause absorbed, long pause triggers the policy"),
            (9, "never-read client", ["hqsb.serving.clients.ClientBehavior.never_read_after_headers", "hqsb.serving.transport.InProcessTransport", "hqsb.serving.gateway.ServingGateway"], "socket/app caps, timeout and cancel"),
            (10, "graceful disconnect", ["hqsb.serving.clients.ClientBehavior.read_frames_then_close", "hqsb.serving.pipeline.DeliveryLedger.as_dict", "hqsb.serving.dummy_backend.DummyServingBackend.as_dict"], "read N frames then FIN: last token, detection, backend stop"),
            (11, "abrupt disconnect", ["hqsb.serving.clients.ClientBehavior.abort_after_frames", "hqsb.serving.gateway.ServingGateway", "hqsb.serving.transport.WriteRecord"], "RST/kill client: no dependence on a clean terminal handshake"),
            (12, "cancel in every pipeline window", ["hqsb.serving.pipeline.StreamPipelineRecorder.declare_cancel", "hqsb.serving.pipeline.ALLOWED_AFTER_CANCEL", "hqsb.serving.pipeline.StreamPipelineRecorder.audit"], "queue/ready/serialize/write/final race"),
            (13, "batch cancel", ["hqsb.serving.gateway.ServingGateway.request_cancel", "hqsb.serving.pipeline.DeliveryLedger", "hqsb.serving.observability.MetricsRegistry.consistency_check"], "same batch/different tenants: no cross contamination"),
            (14, "slow + normal mix", ["hqsb.serving.clients.slow_ratio_plan", "hqsb.serving.clients.collateral_report", "hqsb.serving.fairness.fairness_report"], "normal-client P99/goodput collateral"),
            (15, "per-connection cap", ["hqsb.serving.pipeline.LayerState.cap_bytes", "hqsb.serving.transport.InProcessTransport.record_backpressure", "hqsb.serving.pipeline.summarize_layers"], "a single connection is handled, not the whole service"),
            (16, "global pressure coupling", ["hqsb.serving.admission.PressureSignal.buffered_socket_bytes", "hqsb.serving.admission.PressureStateMachine.observe", "hqsb.serving.admission.recovery_report"], "total buffered bytes trigger E08-05 pressure"),
            (17, "long-output KV pressure", ["hqsb.serving.pipeline.DeliveryLedger.waste", "hqsb.serving.clients.ClientReadReport", "hqsb.serving.slo.token_goodput"], "KV slot hold, generated-discarded tokens, energy"),
            (18, "healthy request after disconnect", ["hqsb.serving.fairness.fairness_report", "hqsb.serving.timing.TimestampLedger.derived", "hqsb.serving.pipeline.resource_slope"], "queue/resource/latency return to baseline"),
            (19, "graceful drain normal path", ["hqsb.serving.gateway.ServingGateway.begin_drain", "hqsb.serving.gateway.ServingGateway.readiness", "hqsb.serving.gateway.ServingGateway.close"], "in-flight completes; new requests rejected, none reach the backend"),
            (20, "drain deadline exceeded", ["hqsb.serving.gateway.ServingGateway.force_cancel", "hqsb.serving.pipeline.resource_slope", "hqsb.serving.gateway.ServingGateway.close"], "long stream/never-read: bounded force cancel and cleanup"),
            (21, "repeated shutdown/close", ["hqsb.serving.gateway.ServingGateway.close", "hqsb.serving.gateway.ServingGateway.begin_drain", "hqsb.serving.pipeline.long_run_pass"], "concurrent signals, double close, close after failure"),
            (22, "long-run leak check", ["hqsb.serving.pipeline.resource_slope", "hqsb.serving.pipeline.long_run_pass", "hqsb.serving.clients.slow_ratio_plan"], "task/fd/RSS/KV/socket slope over many rounds"),
            (23, "independent repeats and profile", ["hqsb.serving.observability.overhead_report", "hqsb.serving.experiment.Preregistration.independent_processes", "hqsb.serving.observability.SamplingPolicy"], "timing runs separated from full-trace/profile runs"),
            (24, "lifecycle/buffer map", ["hqsb.serving.pipeline.summarize_layers", "hqsb.serving.experiment.RunDirectory.write_report_skeleton", "hqsb.serving.specs.KIND_STREAM_LIFECYCLE_SPEC"], "owner/cap/wakeup/cancel/cleanup per layer"),
        ]
    ),
)

E08_09 = ExperimentMapping(
    experiment_id="E08-09",
    title="backend crash/hang/OOM, cache/network faults, circuit breaker and recovery",
    driver="scripts/serving/run_e08.py --experiment E08-09",
    fixture="configs/serving/fault_spec.yaml",
    steps=_steps(
        [
            (1, "freeze FaultSpec and policies", ["hqsb.serving.faults.FaultSpec", "hqsb.serving.faults.FaultSpecEntry", "hqsb.serving.circuit.CircuitSpec"], "trigger, commit phase, expected behaviour, thresholds, safety"),
            (2, "verify injector precision", ["hqsb.serving.dummy_backend.DummyFault", "hqsb.serving.dummy_backend.detect_fault_applied", "hqsb.serving.faults.injector_precision"], "dummy target first: time, scope, recovery, no collateral"),
            (3, "healthy golden", ["hqsb.serving.sse.validate_stream", "hqsb.serving.router.route_vs_actual", "hqsb.serving.slo.goodput"], "two backends, cache, stream/non-stream stable"),
            (4, "circuit unit/property tests", ["hqsb.serving.circuit.CircuitBreaker", "hqsb.serving.circuit.transition_is_recomputable", "hqsb.serving.circuit.EXCLUDED_FAILURE_CLASSES"], "counting, window, open/half-open, probes, epoch/reset"),
            (5, "pre-submit backend unavailable", ["hqsb.serving.faults.retry_policy_after_fault", "hqsb.serving.router.hard_filter", "hqsb.serving.dummy_backend.DummyFault"], "safe re-route/reject with zero backend residue"),
            (6, "load/warmup failure", ["hqsb.serving.circuit.HealthState.classification", "hqsb.serving.router.RouteDecision.selected_model_epoch", "hqsb.serving.router.hard_filter"], "an unready epoch never enters the feasible set"),
            (7, "crash before commit", ["hqsb.serving.faults.commit_phase", "hqsb.serving.faults.Attempt", "hqsb.serving.router.RouteDecision.selected_instance_id"], "retry/failover with attempt lineage, cancel and actual identity"),
            (8, "crash after stream commit", ["hqsb.serving.faults.retry_policy_after_fault", "hqsb.serving.protocol.ErrorCatalog.entry", "hqsb.serving.sse.validate_stream"], "explicit incomplete/error; no transparent re-generation"),
            (9, "backend hang/no progress", ["hqsb.serving.circuit.CircuitBreaker.record_failure", "hqsb.serving.circuit.RecoveryTimeline.metrics", "hqsb.serving.pipeline.resource_slope"], "deadline/watchdog MTTD, isolation, slot/KV release, collateral"),
            (10, "backend slow/straggler", ["hqsb.serving.router.Telemetry", "hqsb.serving.circuit.RecoveryTimeline", "hqsb.serving.service_ab.evaluate_guardrails"], "not automatically a failure: avoid over-flapping"),
            (11, "runtime OOM", ["hqsb.serving.faults.FaultSpecEntry", "hqsb.serving.admission.AdmissionCostModel.calibration", "hqsb.serving.circuit.CircuitBreaker.counters"], "admission miss vs. execution OOM; circuit/retry/fallback/cleanup"),
            (12, "malformed/identity mismatch", ["hqsb.serving.circuit.QUARANTINE_CLASSES", "hqsb.serving.circuit.CircuitBreaker.clear_quarantine", "hqsb.serving.gateway.ServingGateway"], "hard quarantine: never hand a wrong-model response to the client"),
            (13, "cache unavailable", ["hqsb.serving.cache_routing.cache_route", "hqsb.serving.cache_routing.telemetry_freshness", "hqsb.serving.cache_routing.TelemetryFaultInjector"], "bypass/recompute, cache is not a hard dependency"),
            (14, "cache stale/corrupt", ["hqsb.serving.cache_routing.block_identity_second_check", "hqsb.serving.cache_routing.version_invalidation_check", "hqsb.serving.cache_routing.contradictory_updates"], "wrong identity refused before use; quarantine/invalidation recorded"),
            (15, "gateway→backend connect failure", ["hqsb.serving.faults.retry_policy_after_fault", "hqsb.serving.admission.RetryBudget.decide", "hqsb.serving.protocol.ErrorCatalog.entry"], "pre-commit retry, backoff, route filter, error mapping"),
            (16, "mid-request network interruption", ["hqsb.serving.faults.commit_phase", "hqsb.serving.gateway.RequestOutcome.detached_reason", "hqsb.serving.pipeline.DeliveryLedger"], "inject before/after the first token; commit-aware behaviour"),
            (17, "disconnect vs. backend fault race", ["hqsb.serving.gateway.TransitionRecord", "hqsb.serving.circuit.CircuitBreaker.counters", "hqsb.serving.faults.blast_radius_report"], "one terminal reason; counters not double counted"),
            (18, "retry storm counter-example", ["hqsb.serving.admission.RetryBudget.amplification", "hqsb.serving.admission.RetryBudget", "hqsb.serving.admission.UnboundedQueueGuard"], "no jitter/no budget vs. controlled, stop at the guard"),
            (19, "healthy backend collateral", ["hqsb.serving.faults.blast_radius_report", "hqsb.serving.faults.AffectedRequest", "hqsb.serving.admission.PressureStateMachine"], "after failover: queue/goodput of the healthy backend"),
            (20, "recovery/half-open probe", ["hqsb.serving.circuit.CircuitBreaker.allow_request", "hqsb.serving.circuit.RecoveryTimeline", "hqsb.serving.router.BackendRecord.route_epoch"], "new epoch takes limited probes then ramps traffic"),
            (21, "repeated flap", ["hqsb.serving.circuit.flap_report", "hqsb.serving.circuit.CircuitBreaker.counters", "hqsb.serving.pipeline.resource_slope"], "oscillation, backoff, resources, epoch ABA"),
            (22, "long-run mixed faults", ["hqsb.serving.faults.InjectionPlan", "hqsb.serving.faults.fault_matrix_report", "hqsb.serving.pipeline.resource_slope"], "pre-generated fault timeline; resource slope and error accumulation"),
            (23, "independent repeats and root-cause trace", ["hqsb.serving.observability.SpanRecorder", "hqsb.serving.observability.RootCauseClaim", "hqsb.serving.experiment.Preregistration.independent_processes"], "at least three runs; one full trace per fault class"),
            (24, "fault matrix/runbook", ["hqsb.serving.faults.fault_matrix_report", "hqsb.serving.circuit.RecoveryTimeline.as_dict", "hqsb.serving.experiment.RunDirectory.write_report_skeleton"], "detection, blast radius, retry/fallback, MTTD/MTTR, cleanup"),
        ]
    ),
)

E08_10 = ExperimentMapping(
    experiment_id="E08-10",
    title="gateway → queue → runtime → prefill/decode → kernel observability",
    driver="scripts/serving/run_e08.py --experiment E08-10",
    fixture="configs/serving/telemetry_spec.yaml",
    steps=_steps(
        [
            (1, "freeze TelemetrySpec", ["hqsb.serving.observability.ID_RELATIONSHIPS", "hqsb.serving.observability.METRIC_KINDS", "hqsb.serving.specs.KIND_TELEMETRY_SPEC"], "ids, spans, metrics/logs, clocks, sampling, redaction, overhead gate"),
            (2, "schema/context unit tests", ["hqsb.serving.observability.parse_traceparent", "hqsb.serving.observability.local_trace_context", "hqsb.serving.observability.SpanRecorder.join_audit"], "legal/illegal traceparent, parent-child, async/thread propagation"),
            (3, "dummy full-chain fixture", ["hqsb.serving.gateway.ServingGateway", "hqsb.serving.dummy_backend.DummyServingBackend", "hqsb.serving.observability.SpanRecorder"], "controlled delay/error per stage; trace vs. metric/log agreement"),
            (4, "clock/offset calibration", ["hqsb.serving.timing.ClockDomain", "hqsb.serving.timing.cross_domain_delta_ns", "hqsb.serving.observability.CLOCK_DOMAINS"], "monotonic/wall offsets, uncertainty, GPU correlation method"),
            (5, "instrument HTTP/validation", ["hqsb.serving.observability.SpanRecorder.emit", "hqsb.serving.transport.TransportRequest.received_ns", "hqsb.serving.timing.TIMESTAMP_FIELDS"], "receive/body/parse/schema/template/tokenization boundaries"),
            (6, "instrument admission/queue", ["hqsb.serving.gateway.TransitionRecord.queue_depth", "hqsb.serving.admission.PressureEvent", "hqsb.serving.observability.METRIC_KINDS"], "enqueue/dequeue/reject, depth/age, policy decision"),
            (7, "instrument routing/attempts", ["hqsb.serving.router.RouteDecision", "hqsb.serving.circuit.CircuitEvent", "hqsb.serving.faults.Attempt"], "candidates/filter/score/selected, attempt/fallback/circuit"),
            (8, "bridge the S07 trace", ["hqsb.serving.observability.Span.iteration", "hqsb.serving.telemetry.KIND_TO_EVENT_TYPE", "hqsb.runtime.telemetry.SPAN_CHAIN"], "connect backend request to scheduler iteration/batch/KV/kernel"),
            (9, "instrument SSE/socket/client", ["hqsb.serving.pipeline.FramePipelineEvent", "hqsb.serving.sse.SseFrame.index", "hqsb.serving.clients.ClientReadReport"], "token-ready→serialize→write→client-read→terminal"),
            (10, "normal request golden trace", ["hqsb.serving.observability.SpanRecorder.critical_path", "hqsb.serving.timing.time_conservation", "hqsb.serving.observability.evidence_consistency"], "manual span replay: state, token, time conservation"),
            (11, "queue-slow trace", ["hqsb.serving.observability.classify_time_breakdown", "hqsb.serving.slo.LoadPointResult", "hqsb.serving.arrival.QueueSample"], "separate gateway from Runtime service time"),
            (12, "runtime-slow trace", ["hqsb.serving.observability.SPAN_KINDS", "hqsb.serving.telemetry.to_trace_events", "hqsb.serving.observability.RootCauseClaim"], "long prefill/decode or slow kernel, drill into iteration/shape"),
            (13, "cache hit/miss pairing", ["hqsb.serving.cache_routing.prediction_vs_actual", "hqsb.serving.cache_routing.CacheTelemetrySample", "hqsb.serving.observability.classify_time_breakdown"], "route/cache telemetry vs. actual reuse and prefill time"),
            (14, "slow-client trace", ["hqsb.serving.pipeline.StreamPipelineRecorder.audit", "hqsb.serving.clients.ClientReadReport.client_read_ns", "hqsb.serving.observability.SAMPLING_MODES"], "separate token-ready from socket/client-read delay and buffer"),
            (15, "fault/retry trace", ["hqsb.serving.faults.Attempt", "hqsb.serving.circuit.CircuitEvent", "hqsb.serving.circuit.RecoveryTimeline"], "fault injection → circuit → attempt → fallback → client error → cleanup"),
            (16, "metric counter conservation", ["hqsb.serving.observability.MetricsRegistry.terminal_events", "hqsb.serving.observability.MetricsRegistry.consistency_check", "hqsb.serving.slo.FunnelCounts"], "recompute received/admitted/completed/good/error/retry"),
            (17, "histogram vs. raw percentiles", ["hqsb.serving.observability.Histogram.count", "hqsb.serving.observability.Histogram.quantile", "hqsb.serving.slo.percentile_from_raw"], "buckets cover the SLO; no aggregate summary quantiles"),
            (18, "log/trace consistency", ["hqsb.serving.observability.LogRecord", "hqsb.serving.observability.redaction_audit", "hqsb.serving.observability.evidence_consistency"], "route/error/state logs find their span; epoch and reason match"),
            (19, "exemplar/trace drill-down", ["hqsb.serving.observability.Histogram.exemplars", "hqsb.serving.observability.SpanRecorder.chain_for", "hqsb.serving.telemetry.trace_join_check"], "from a P99/error metric to a specific trace and backend"),
            (20, "trace loss/cardinality pressure", ["hqsb.serving.observability.FORBIDDEN_LABELS", "hqsb.serving.observability.SamplingPolicy", "hqsb.serving.observability.MetricsRegistry.counter"], "exporter drop, log queue, label cardinality at capacity"),
            (21, "minimal/full overhead A/B", ["hqsb.serving.observability.overhead_report", "hqsb.serving.observability.OverheadObservation", "hqsb.serving.observability.SamplingPolicy.keep"], "off/minimal/full/profile perturbation on P99/goodput/CPU/bytes"),
            (22, "blind P99 request", ["hqsb.serving.observability.slow_request_selection_rule", "hqsb.serving.observability.RootCauseClaim.verdict", "hqsb.serving.observability.classify_time_breakdown"], "selected by the pre-registered rule, not by convenience"),
            (23, "counterfactual root cause", ["hqsb.serving.observability.RootCauseClaim.counterfactual_supported", "hqsb.serving.service_ab.mechanism_mediation_check", "hqsb.serving.observability.evidence_consistency"], "change the single cause or use a matched fast request"),
            (24, "dashboard/trace runbook", ["hqsb.serving.experiment.RunDirectory.write_report_skeleton", "hqsb.serving.observability.MetricsRegistry.exposition", "hqsb.serving.observability.SPAN_KINDS"], "alert→class→trace→queue/runtime/kernel→action"),
        ]
    ),
)

E08_11 = ExperimentMapping(
    experiment_id="E08-11",
    title="service-level strict A/B of one scheduler/cache/routing policy",
    driver="scripts/serving/run_e08.py --experiment E08-11",
    fixture="hqsb.serving.service_ab.BottleneckEvidence",
    steps=_steps(
        [
            (1, "aggregate E08-01…10 baseline evidence", ["hqsb.serving.service_ab.BottleneckEvidence", "hqsb.serving.service_ab.bottleneck_table", "hqsb.serving.service_ab.BASELINE_EVIDENCE_SOURCES"], "bottleneck table citing accessible raw/trace/report only"),
            (2, "select the single main problem", ["hqsb.serving.service_ab.SERVICE_CANDIDATES", "hqsb.serving.service_ab.selection_gate_from_table", "hqsb.runtime.policy_ab.SelectionGate"], "impact, evidence strength, risk, measurability, scope"),
            (3, "ADR and falsifiable hypothesis", ["hqsb.runtime.policy_ab.Adr", "hqsb.serving.experiment.Preregistration", "hqsb.serving.service_ab.SERVICE_DECISIONS"], "mechanism, primary metric, near cause, guardrails, rollback"),
            (4, "freeze baseline A", ["hqsb.serving.experiment.git_state", "hqsb.serving.experiment.EvidenceManifest", "hqsb.serving.slo.SLOSpec.require_frozen"], "lock source/config/artifact and replay the baseline"),
            (5, "minimal treatment B", ["hqsb.runtime.policy_ab.require_identity_equal", "hqsb.serving.service_ab.mechanism_mediation_check", "hqsb.serving.router.RouteDecision.policy_version"], "only the target policy patch, with requested/actual path evidence"),
            (6, "code-review invariants", ["hqsb.serving.admission.SAFETY_INVARIANTS", "hqsb.serving.pipeline.DeliveryLedger.audit", "hqsb.serving.gateway.ALLOWED_REQUEST_TRANSITIONS"], "protocol, identity, deadline, fairness, cancel, resource, thread safety"),
            (7, "unit/property/simulation tests", ["hqsb.serving.policies.policy_interface_audit", "hqsb.serving.circuit.breaker_matrix", "hqsb.serving.admission.PressureStateMachine"], "deterministic decision/boundary/NaN/counter/epoch/tie-break"),
            (8, "protocol/correctness gate", ["hqsb.serving.protocol.conformance_matrix", "hqsb.serving.sse.validate_stream", "hqsb.serving.router.route_vs_actual"], "both arms pass the E08-01 matrix and S07 identity"),
            (9, "fault/resource gate", ["hqsb.serving.faults.retry_policy_after_fault", "hqsb.serving.pipeline.resource_slope", "hqsb.serving.clients.behavior_matrix"], "cancel/OOM/disconnect/failure paths before scaling up"),
            (10, "freeze A/B artifacts", ["hqsb.serving.experiment.EvidenceManifest.config_sha256", "hqsb.serving.specs.ServingSpecs.load", "hqsb.serving.experiment.Preregistration.policy_config_hash"], "unique diff, build/config hash, no code change after this point"),
            (11, "pilot for variance and tooling", ["hqsb.runtime.policy_ab.pilot_separation", "hqsb.serving.loadgen.noop_calibration", "hqsb.serving.slo.tail_sample_sufficiency"], "effect direction, runtime, tail samples, loadgen capacity"),
            (12, "freeze the final matrix", ["hqsb.serving.service_ab.StratifiedResult", "hqsb.serving.service_ab.stratified_matrix", "hqsb.serving.specs.KIND_CAPACITY_SPEC"], "target/neutral/holdout, load bands, runs, order, stop rules"),
            (13, "reset the initial state", ["hqsb.serving.admission.PressureStateMachine", "hqsb.serving.circuit.CircuitBreaker", "hqsb.serving.cache_routing.CacheTelemetrySample"], "clear queue/circuit/cache or load one snapshot"),
            (14, "ABBA/randomised blocks", ["hqsb.runtime.policy_ab.block_schedule", "hqsb.runtime.policy_ab.schedule_balance", "hqsb.serving.observability.overhead_report"], "same arrival/payload trace, thermal/clock monitored"),
            (15, "target traces", ["hqsb.serving.slo.goodput", "hqsb.serving.policies.DecisionLog", "hqsb.serving.pipeline.summarize_layers"], "full client/queue/policy/backend/resource raw"),
            (16, "neutral controls", ["hqsb.serving.service_ab.measure_ab", "hqsb.serving.fairness.fairness_report", "hqsb.serving.slo.latency_summary"], "no fake benefit or fixed overhead in unrelated scenarios"),
            (17, "holdout/adversarial", ["hqsb.serving.arrival.BurstinessPoint", "hqsb.serving.fairness.hol_trace", "hqsb.serving.cache_routing.zero_locality_falsification"], "burst/length/tenant/cache/fault generalisation and degradation"),
            (18, "representative full trace/profile", ["hqsb.serving.observability.SpanRecorder", "hqsb.serving.observability.overhead_report", "hqsb.serving.service_ab.mechanism_mediation_check"], "matched requests verify the actual path/near cause"),
            (19, "recompute the offered→good funnel", ["hqsb.serving.slo.FunnelCounts", "hqsb.serving.slo.classify_failure", "hqsb.serving.loadgen.offered_load_funnel"], "including reject/timeout/failure/cancel and intended/actual"),
            (20, "primary and guardrail effects", ["hqsb.serving.service_ab.measure_ab", "hqsb.serving.service_ab.evaluate_guardrails", "hqsb.serving.service_ab.guardrail_summary"], "paired run effects, CI, practical margin, tails, per-class"),
            (21, "mediation chain check", ["hqsb.serving.service_ab.mechanism_mediation_check", "hqsb.serving.observability.classify_time_breakdown", "hqsb.serving.router.RouteDecision.as_dict"], "policy → queue/cache/route → Runtime → client outcome"),
            (22, "regression envelope/ablation", ["hqsb.runtime.policy_ab.regression_envelope", "hqsb.runtime.policy_ab.ablation_matrix", "hqsb.serving.service_ab.stratified_matrix"], "benefit/null/regression regions, marked exploratory"),
            (23, "independent confirmation", ["hqsb.serving.experiment.interface_only_run", "hqsb.runtime.policy_ab.pilot_separation", "hqsb.serving.experiment.check_prerequisites"], "fresh process, pre-registered best/negative point, reverse order"),
            (24, "verdict merge/rollback/negative", ["hqsb.serving.service_ab.service_decision", "hqsb.serving.service_ab.rollback_for_service", "hqsb.serving.experiment.RunDirectory.write_verdict"], "PASS_POSITIVE/PASS_NEGATIVE/FAIL/INCONCLUSIVE and ADR update"),
        ]
    ),
)

EXPERIMENTS: Tuple[ExperimentMapping, ...] = (
    E08_01,
    E08_02,
    E08_03,
    E08_04,
    E08_05,
    E08_06,
    E08_07,
    E08_08,
    E08_09,
    E08_10,
    E08_11,
)

MAPPINGS: Dict[str, ExperimentMapping] = {
    mapping.experiment_id: mapping for mapping in EXPERIMENTS
}


def mapping_for(experiment_id: str) -> ExperimentMapping:
    if experiment_id not in MAPPINGS:
        raise KeyError(
            f"unknown experiment {experiment_id!r}; expected one of {list(MAPPINGS)}"
        )
    return MAPPINGS[experiment_id]


def _resolve(symbol: str) -> Optional[str]:
    """Import ``module.Class.method`` and return the error message (or None)."""
    parts = symbol.split(".")
    for split in range(len(parts), 1, -1):
        candidate = ".".join(parts[:split])
        if not (candidate.startswith("hqsb") or candidate.startswith("ops")):
            break
        try:
            module = importlib.import_module(candidate)
        except ImportError:
            continue
        target: Any = module
        path = parts[split:]
        for index, attribute in enumerate(path):
            try:
                target = getattr(target, attribute)
            except AttributeError as exc:
                # a dataclass field without a default is an instance attribute, so
                # ``getattr(Class, field)`` fails even though the field exists
                if dataclasses.is_dataclass(target) and index == len(path) - 1:
                    if attribute in {field.name for field in dataclasses.fields(target)}:
                        return None
                return f"{type(exc).__name__}: {exc}"
        return None
    return f"ModuleNotFoundError: no module component of {symbol!r} is importable"


def resolve_interfaces() -> Dict[str, Any]:
    """Verify every referenced symbol really imports (the map cannot rot)."""
    references: List[Tuple[str, int, str]] = [
        (mapping.experiment_id, step.step, symbol)
        for mapping in EXPERIMENTS
        for step in mapping.steps
        for symbol in step.interfaces
    ]
    unique = sorted({symbol for _experiment, _step, symbol in references})
    failures: List[Dict[str, Any]] = []
    for symbol in unique:
        error = _resolve(symbol)
        if error:
            failures.append(
                {
                    "symbol": symbol,
                    "error": error,
                    "used_by": [
                        f"{experiment}:{step}"
                        for experiment, step, candidate in references
                        if candidate == symbol
                    ][:3],
                }
            )
    return {
        "experiments": len(EXPERIMENTS),
        "steps": sum(len(mapping.steps) for mapping in EXPERIMENTS),
        "references": len(references),
        "interfaces": len(unique),
        "ok": not failures,
        "failures": failures,
    }


def mapping_table_markdown() -> str:
    """Human-readable table for the development report and the driver."""
    lines = [
        "| 实验 | 步骤数 | 驱动入口 | 关键接口（示例） |",
        "|---|---|---|---|",
    ]
    for mapping in EXPERIMENTS:
        sample = ", ".join(
            interface.split(".")[-1] for interface in mapping.steps[0].interfaces[:3]
        )
        lines.append(
            f"| {mapping.experiment_id} {mapping.title} | {len(mapping.steps)} | "
            f"`{mapping.driver}` | `{sample}` |"
        )
    total = sum(len(mapping.steps) for mapping in EXPERIMENTS)
    lines.append(f"| **合计** | **{total}** | — | — |")
    return "\n".join(lines) + "\n"


def step_table_for(experiment_id: str) -> str:
    mapping = mapping_for(experiment_id)
    lines = [f"### {mapping.experiment_id}: {mapping.title}", "", f"驱动：`{mapping.driver}`", ""]
    lines.append("| 步骤 | 步骤内容 | 代码接口 | 成熟度 |")
    lines.append("|---|---|---|---|")
    for step in mapping.steps:
        interfaces = "<br>".join(f"`{item}`" for item in step.interfaces)
        lines.append(f"| {step.step} | {step.title} | {interfaces} | {step.maturity} |")
    return "\n".join(lines) + "\n"


_resolve.__doc__ = "Import one dotted symbol and return the error message (or None)."

__all__ = [
    "E08_01",
    "E08_02",
    "E08_03",
    "E08_04",
    "E08_05",
    "E08_06",
    "E08_07",
    "E08_08",
    "E08_09",
    "E08_10",
    "E08_11",
    "EXPERIMENTS",
    "ExperimentMapping",
    "MAPPINGS",
    "StepMapping",
    "mapping_for",
    "mapping_table_markdown",
    "resolve_interfaces",
    "step_table_for",
]
