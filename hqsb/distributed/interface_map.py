"""Experiment step → code interface map (the "no missing capability" proof).

``docs/stage_experiments/details/S10/E10-01..E10-10`` each list thirty concrete
steps.  This module records, for **every** step, which HQSB interface implements
it, and :func:`resolve_interfaces` verifies that each referenced symbol actually
imports — so the map cannot rot into documentation.  A test
(``tests/unit/distributed/test_s10_experiment_scaffolding.py``) runs the
verification.

Maturity labels follow the control plane (M0–M7):

* ``M1`` — source exists (interface implemented, structured errors);
* ``M2`` — covered by automated tests in this tree;
* ``M3``+ — requires a real multi-accelerator run, which this repository state
  cannot produce (no second device, no S07 P0 verdict, no sealed topology
  manifest), so no entry is labelled above M2.

Drivers live in ``scripts/distributed/run_e10.py`` and are *not* executed here.
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


def _steps(pairs: Sequence[Tuple[int, str, Sequence[str]]]) -> Tuple[StepMapping, ...]:
    """Compact constructor: ``(step, title, interfaces)``."""
    return tuple(
        StepMapping(step=step, title=title, interfaces=tuple(interfaces))
        for step, title, interfaces in pairs
    )


E10_01 = ExperimentMapping(
    experiment_id="E10-01",
    title="topology identity, link reachability and rank placement",
    driver="scripts/distributed/run_e10.py --experiment E10-01",
    fixture="configs/distributed/topology_spec.yaml + placement_spec.yaml",
    steps=_steps(
        [
            (1, "freeze the experiment scope", ["hqsb.distributed.topology.ObservationScope", "hqsb.distributed.topology.BRANCHES"]),
            (2, "freeze host and scheduler identity", ["hqsb.distributed.topology.HostIdentity", "hqsb.distributed.topology.redact_identity"]),
            (3, "collect the CPU socket/NUMA graph", ["hqsb.distributed.topology.NumaTopology", "hqsb.distributed.topology.NumaNode"]),
            (4, "enumerate accelerator physical identity", ["hqsb.distributed.topology.AcceleratorRecord", "hqsb.distributed.topology.detect_identity_problems"]),
            (5, "audit visible-device remapping", ["hqsb.distributed.topology.VisibleDeviceAudit"]),
            (6, "collect the PCIe tree", ["hqsb.distributed.topology.PcieLink"]),
            (7, "collect the device fabric", ["hqsb.distributed.topology.FabricLink", "hqsb.distributed.topology.LINK_STATES"]),
            (8, "collect accelerator ↔ CPU/memory affinity", ["hqsb.distributed.topology.AffinityRecord"]),
            (9, "enumerate NICs and ports", ["hqsb.distributed.topology.NicRecord"]),
            (10, "collect the RDMA/RoCE/IB stack", ["hqsb.distributed.topology.RdmaStackRecord"]),
            (11, "collect the NCCL/HCCL runtime config", ["hqsb.distributed.topology.BackendRuntimeConfig", "hqsb.distributed.probes.capture_backend_env"]),
            (12, "generate the initial TopologyGraph", ["hqsb.distributed.topology.TopologyManifest", "hqsb.distributed.topology.TopologyNode", "hqsb.distributed.topology.TopologyEdge"]),
            (13, "run the accelerator-pair P2P probe", ["hqsb.distributed.probes.P2PCapability", "hqsb.distributed.probes.PROBE_MATRIX"]),
            (14, "run the device copy microprobe", ["hqsb.distributed.probes.CopyProbeRow"]),
            (15, "run the host↔device NUMA probe", ["hqsb.distributed.probes.HostDeviceProbe"]),
            (16, "run the NIC/RDMA reachability probe", ["hqsb.distributed.probes.RdmaReachability"]),
            (17, "verify the actual communication data path", ["hqsb.distributed.probes.DataPathEvidence", "hqsb.distributed.probes.rdma_claim_guard"]),
            (18, "assign topology edge confidence levels", ["hqsb.distributed.topology.EDGE_CONFIDENCE", "hqsb.distributed.topology.TopologyEdge"]),
            (19, "design the baseline PlacementPlan", ["hqsb.distributed.placement.PlacementPlan", "hqsb.distributed.placement.ParallelCoordinate"]),
            (20, "generate the rank table / launcher config", ["hqsb.distributed.placement.build_rank_table", "hqsb.distributed.placement.launcher_config"]),
            (21, "start the per-rank identity probe", ["hqsb.distributed.ranks.RankIdentity"]),
            (22, "verify one rank per device", ["hqsb.distributed.ranks.validate_rank_identities", "hqsb.distributed.placement.compare_planned_actual"]),
            (23, "verify process group membership", ["hqsb.distributed.ranks.GroupMembership", "hqsb.distributed.ranks.compare_planned_actual_groups"]),
            (24, "run the placement positive smoke", ["hqsb.distributed.placement.RankPlacement", "hqsb.distributed.probes.probe_chain_status"]),
            (25, "run the alternative placement control", ["hqsb.distributed.placement.alternative_placement", "hqsb.distributed.placement.PlacementSanityExpectation"]),
            (26, "inject mapping fixtures", ["hqsb.distributed.probes.preflight_rank_mapping", "hqsb.distributed.probes.make_duplicate_device_identities", "hqsb.distributed.probes.preflight_group_order", "hqsb.distributed.probes.mapping_fixture_cases"]),
            (27, "check degraded/unknown link policy", ["hqsb.distributed.topology.DegradedLinkPolicy", "hqsb.distributed.topology.audit_degraded_edges"]),
            (28, "reproduce with an independent new job", ["hqsb.distributed.topology.diff_manifests", "hqsb.distributed.topology.classify_drift"]),
            (29, "seal topology/placement hashes", ["hqsb.distributed.topology.TopologyManifest.sha256", "hqsb.distributed.topology.canonical_json", "hqsb.distributed.placement.PlacementPlan.sha256"]),
            (30, "form the E10-02 gate verdict", ["hqsb.distributed.topology.gate_for_collective_probe", "hqsb.distributed.topology.check_hard_invariants"]),
        ]
    ),
)

E10_02 = ExperimentMapping(
    experiment_id="E10-02",
    title="collective correctness, message-size curve and cost model",
    driver="scripts/distributed/run_e10.py --experiment E10-02",
    fixture="configs/distributed/collective_spec.yaml + tolerance_spec.yaml",
    steps=_steps(
        [
            (1, "read the E10-01 gate", ["hqsb.distributed.topology.gate_for_collective_probe"]),
            (2, "freeze backend and harness identity", ["hqsb.distributed.backend.BackendIdentity", "hqsb.distributed.backend.HarnessIdentity"]),
            (3, "define the CollectiveSpec", ["hqsb.distributed.collectives.spec_for", "hqsb.distributed.collectives.CollectiveSpec", "hqsb.distributed.collectives.COUNT_SEMANTICS"]),
            (4, "implement the rank-coded generator", ["hqsb.distributed.collectives.rank_coded_pattern", "hqsb.distributed.collectives.vector_for_rank"]),
            (5, "implement the CPU/host oracle", ["hqsb.distributed.collectives.oracle_all_reduce", "hqsb.distributed.collectives.oracle_all_gather", "hqsb.distributed.collectives.oracle_reduce_scatter", "hqsb.distributed.collectives.oracle_broadcast", "hqsb.distributed.collectives.oracle_all_to_all", "hqsb.distributed.collectives.oracle_all_to_all_v"]),
            (6, "implement guard/buffer ownership checks", ["hqsb.distributed.collectives.GuardedBuffer", "hqsb.distributed.collectives.alias_check"]),
            (7, "generate the message-size grid", ["hqsb.distributed.collectives.message_size_grid", "hqsb.distributed.collectives.MessageSizeCase"]),
            (8, "generate the rank/topology grid", ["hqsb.distributed.collectives.rank_grid", "hqsb.distributed.collectives.RankCase"]),
            (9, "probe dtype/op capability", ["hqsb.distributed.collectives.dtype_capability_probe", "hqsb.distributed.backend.CapabilityMatrix"]),
            (10, "freeze the auto/forced algorithm groups", ["hqsb.distributed.backend.algorithm_groups", "hqsb.distributed.backend.AlgoProtocolGroup", "hqsb.distributed.backend.RequestedActual"]),
            (11, "calibrate completion timing", ["hqsb.distributed.backend.TimingCalibration"]),
            (12, "measure init/first-call cost", ["hqsb.distributed.backend.ConnectionCostRecord"]),
            (13, "run the AllReduce correctness sweep", ["hqsb.distributed.collectives.correctness_row", "hqsb.distributed.collectives.should_stop_performance"]),
            (14, "run the AllGather correctness sweep", ["hqsb.distributed.collectives.oracle_all_gather", "hqsb.distributed.collectives.vector_for_rank"]),
            (15, "run the ReduceScatter correctness sweep", ["hqsb.distributed.collectives.oracle_reduce_scatter", "hqsb.distributed.collectives.COUNT_SEMANTICS"]),
            (16, "run the Broadcast root sweep", ["hqsb.distributed.collectives.oracle_broadcast", "hqsb.distributed.collectives.spec_for"]),
            (17, "run the AllToAll correctness sweep", ["hqsb.distributed.collectives.oracle_all_to_all"]),
            (18, "run the AllToAllV skew sweep", ["hqsb.distributed.collectives.oracle_all_to_all_v", "hqsb.distributed.collectives.compare_vectors"]),
            (19, "run the clean latency sweep", ["hqsb.distributed.collectives.LatencySweepPlan", "hqsb.distributed.collectives.ORDER_POLICIES"]),
            (20, "compute AlgBW/BusBW", ["hqsb.distributed.collectives.bandwidth_row", "hqsb.distributed.collectives.FORMULA_REGISTRY", "hqsb.distributed.collectives.bus_correction"]),
            (21, "fit the α–β segments", ["hqsb.distributed.collectives.fit_alpha_beta_segments", "hqsb.distributed.collectives.AlphaBetaSegment"]),
            (22, "identify algorithm/protocol crossovers", ["hqsb.distributed.collectives.detect_crossover"]),
            (23, "analyse rank scaling", ["hqsb.distributed.collectives.rank_scaling_table"]),
            (24, "analyse topology/placement effects", ["hqsb.distributed.collectives.placement_effect_table"]),
            (25, "collect representative-case profiles", ["hqsb.distributed.traces.TRACE_MODES", "hqsb.distributed.collectives.LatencySweepPlan"]),
            (26, "check CPU/network/device bottlenecks", ["hqsb.distributed.collectives.apply_stop_rules", "hqsb.distributed.collectives.STOP_RULES"]),
            (27, "cross-validate with the official tool", ["hqsb.distributed.collectives.crosscheck_with_official_tools"]),
            (28, "project the E10-04 model messages", ["hqsb.distributed.collectives.project_model_messages", "hqsb.distributed.collectives.MessageProjection"]),
            (29, "run the independent job confirmation", ["hqsb.distributed.collectives.apply_stop_rules", "hqsb.distributed.backend.RequestedActual"]),
            (30, "form the communication baseline verdict", ["hqsb.distributed.collectives.should_stop_performance", "hqsb.distributed.collectives.FORMULA_REGISTRY", "hqsb.distributed.collectives.LoopbackCollectiveExecutor"]),
        ]
    ),
)

E10_03 = ExperimentMapping(
    experiment_id="E10-03",
    title="collective sequence, mismatch, timeout and communicator cleanup",
    driver="scripts/distributed/run_e10.py --experiment E10-03",
    fixture="configs/distributed/timeout_spec.yaml + fault_spec.yaml",
    steps=_steps(
        [
            (1, "freeze the safety boundary", ["hqsb.distributed.faults.SafetyScope", "hqsb.distributed.sequence.TimeoutSpec"]),
            (2, "read the E10-01/02 gate", ["hqsb.distributed.topology.gate_for_collective_probe", "hqsb.distributed.collectives.apply_stop_rules"]),
            (3, "define the communicator state machine", ["hqsb.distributed.sequence.CommunicatorStateMachine", "hqsb.distributed.sequence.ALLOWED_TRANSITIONS"]),
            (4, "define sequence allocation", ["hqsb.distributed.sequence.SequenceAllocator"]),
            (5, "implement the per-rank call record", ["hqsb.distributed.sequence.CollectiveCallRecord", "hqsb.distributed.sequence.METADATA_FIELDS"]),
            (6, "implement the development metadata preflight", ["hqsb.distributed.sequence.preflight_check", "hqsb.distributed.sequence.PreflightResult"]),
            (7, "implement the external watchdog", ["hqsb.distributed.sequence.Watchdog", "hqsb.distributed.faults.WATCHDOG_LAYERS"]),
            (8, "implement the error propagation channel", ["hqsb.distributed.sequence.ErrorPropagationChannel"]),
            (9, "implement abort/destroy/cleanup ordering", ["hqsb.distributed.sequence.cleanup_plan", "hqsb.distributed.sequence.CommunicatorStateMachine.destroy_or_abort"]),
            (10, "establish the legal golden sequence", ["hqsb.distributed.sequence.GOLDEN_SEQUENCE"]),
            (11, "verify group-rank ordering", ["hqsb.distributed.ranks.GroupMembership.group_rank_of", "hqsb.distributed.collectives.oracle_all_gather"]),
            (12, "inject an operation mismatch", ["hqsb.distributed.faults.MATRIX_BY_FAULT", "hqsb.distributed.sequence.field_diffs"]),
            (13, "inject a count/shape mismatch", ["hqsb.distributed.sequence.field_diffs", "hqsb.distributed.sequence.Divergence"]),
            (14, "inject a dtype mismatch", ["hqsb.distributed.sequence.byte_counts_match_but_semantics_differ", "hqsb.distributed.sequence.SEMANTIC_CRITICAL_FIELDS"]),
            (15, "inject a root mismatch", ["hqsb.distributed.sequence.preflight_check", "hqsb.distributed.sequence.FieldDiff"]),
            (16, "inject a collective order swap", ["hqsb.distributed.sequence.preflight_check", "hqsb.distributed.sequence.Divergence"]),
            (17, "inject a skipped collective", ["hqsb.distributed.sequence.Watchdog", "hqsb.distributed.faults.watchdog_decision"]),
            (18, "inject a delayed-rank ladder", ["hqsb.distributed.sequence.delayed_rank_ladder", "hqsb.distributed.sequence.evaluate_delayed_case"]),
            (19, "inject a group creation order mismatch", ["hqsb.distributed.ranks.allocator_group_order_check", "hqsb.distributed.probes.preflight_group_order"]),
            (20, "inject concurrent multi-group order mismatch", ["hqsb.distributed.ranks.allocator_group_order_check", "hqsb.distributed.ranks.compare_planned_actual_groups"]),
            (21, "verify async error attribution", ["hqsb.distributed.faults.classify_distributed_error", "hqsb.distributed.backend.normalize_backend_error"]),
            (22, "verify partial buffers are not consumed", ["hqsb.distributed.faults.FailureObservation", "hqsb.distributed.faults.evaluate_fault"]),
            (23, "verify post-error reuse is rejected", ["hqsb.distributed.sequence.CommunicatorStateMachine.begin_collective", "hqsb.distributed.sequence.WRAPPER_ERROR_CODES"]),
            (24, "execute the all-rank abort", ["hqsb.distributed.faults.AbortCoordinator"]),
            (25, "audit resource cleanup", ["hqsb.distributed.faults.ResourceSnapshot", "hqsb.distributed.faults.snapshot_delta"]),
            (26, "recreate the communicator", ["hqsb.distributed.sequence.SequenceAllocator.reset_for_epoch", "hqsb.distributed.ranks.GroupMembership"]),
            (27, "verify error output completeness", ["hqsb.distributed.sequence.ErrorPropagationChannel.missing_ranks", "hqsb.distributed.sequence.ErrorPropagationChannel.as_dict"]),
            (28, "measure detection/abort/recovery distributions", ["hqsb.distributed.faults.TimeMetrics", "hqsb.distributed.faults.distribution_summary"]),
            (29, "run the blind diagnostic drill", ["hqsb.distributed.faults.DrillResult"]),
            (30, "form the TP safety gate", ["hqsb.distributed.sequence.tp_safety_gate", "hqsb.distributed.sequence.COLLECTIVE_FAULTS"]),
        ]
    ),
)

E10_04 = ExperimentMapping(
    experiment_id="E10-04",
    title="Qwen tensor parallel correctness and the communication ledger",
    driver="scripts/distributed/run_e10.py --experiment E10-04",
    fixture="configs/distributed/parallel_plan_spec.yaml",
    steps=_steps(
        [
            (1, "freeze the single-device reference", ["hqsb.distributed.parallel_plan.representative_run_plan", "hqsb.distributed.parallel_plan.RunPlan"]),
            (2, "parse the Qwen architecture census", ["hqsb.distributed.parallel_plan.QwenArchitectureCensus", "hqsb.distributed.parallel_plan.census_from_mapping"]),
            (3, "define the TP degree capability", ["hqsb.distributed.parallel_plan.tp_capability_matrix", "hqsb.distributed.parallel_plan.TpCapability"]),
            (4, "write the ParallelPlan schema", ["hqsb.distributed.parallel_plan.ParallelPlan", "hqsb.distributed.parallel_plan.PlanCollectiveEvent"]),
            (5, "derive the attention shard", ["hqsb.distributed.parallel_plan.derive_attention_shard", "hqsb.distributed.parallel_plan.AttentionShard"]),
            (6, "derive the MLP shard", ["hqsb.distributed.parallel_plan.derive_mlp_shard", "hqsb.distributed.parallel_plan.MlpShard"]),
            (7, "fix the norm/rope/residual state", ["hqsb.distributed.parallel_plan.default_replication_policies", "hqsb.distributed.parallel_plan.ReplicatedTensorPolicy"]),
            (8, "decide the embedding/LM-head strategy", ["hqsb.distributed.parallel_plan.embedding_strategy", "hqsb.distributed.parallel_plan.EmbeddingStrategyDecision"]),
            (9, "decide the KV cache ownership", ["hqsb.distributed.parallel_plan.kv_ownership", "hqsb.distributed.parallel_plan.kv_per_token_bytes", "hqsb.distributed.parallel_plan.KvOwnership"]),
            (10, "generate the sharded checkpoint", ["hqsb.distributed.parallel_plan.ShardSpec", "hqsb.distributed.parallel_plan.plan_axis_shards"]),
            (11, "verify the shard round-trip", ["hqsb.distributed.parallel_plan.verify_shard_ranges", "hqsb.distributed.parallel_plan.merge_flat", "hqsb.distributed.parallel_plan.RoundtripReport"]),
            (12, "verify direct shard loading", ["hqsb.distributed.parallel_plan.LoadPlan", "hqsb.distributed.parallel_plan.audit_direct_load"]),
            (13, "create the TP process groups", ["hqsb.distributed.ranks.GroupMembership", "hqsb.distributed.sequence.CommunicatorStateMachine"]),
            (14, "implement the activation layout state", ["hqsb.distributed.parallel_plan.ActivationLayout", "hqsb.distributed.parallel_plan.validate_collective_input_state"]),
            (15, "run the column-parallel unit test", ["hqsb.distributed.parallel_plan.column_parallel_case", "hqsb.distributed.collectives.compare_vectors"]),
            (16, "run the row-parallel unit test", ["hqsb.distributed.parallel_plan.row_parallel_case", "hqsb.distributed.parallel_plan.split_flat"]),
            (17, "run the attention block test", ["hqsb.distributed.parallel_plan.attention_block_case"]),
            (18, "run the MLP block test", ["hqsb.distributed.parallel_plan.mlp_block_case"]),
            (19, "run the layer-level TP test", ["hqsb.distributed.parallel_plan.layer_case", "hqsb.distributed.parallel_plan.first_deviation"]),
            (20, "run the tiny-model prefill", ["hqsb.distributed.parallel_plan.representative_run_plan"]),
            (21, "run the tiny-model multi-step decode", ["hqsb.distributed.parallel_plan.decode_alignment_plan"]),
            (22, "run the representative short prefill", ["hqsb.distributed.parallel_plan.representative_run_plan", "hqsb.distributed.parallel_plan.first_deviation"]),
            (23, "run the representative decode", ["hqsb.distributed.parallel_plan.decode_alignment_plan", "hqsb.distributed.ledger.expected_ledger"]),
            (24, "generate the theoretical communication ledger", ["hqsb.distributed.ledger.expected_ledger", "hqsb.distributed.ledger.ExpectedEvent", "hqsb.distributed.ledger.expected_totals"]),
            (25, "generate the observed ledger", ["hqsb.distributed.ledger.ObservedEvent", "hqsb.distributed.ledger.diff_ledger"]),
            (26, "diff expected ↔ observed", ["hqsb.distributed.ledger.diff_ledger", "hqsb.distributed.ledger.LEDGER_REASON_CODES", "hqsb.distributed.ledger.why_observed_can_differ"]),
            (27, "audit per-rank memory", ["hqsb.distributed.ledger.MemoryRow", "hqsb.distributed.ledger.reconcile_memory", "hqsb.distributed.ledger.rank_skew"]),
            (28, "verify unsupported/uneven shapes", ["hqsb.distributed.parallel_plan.unsupported_policy", "hqsb.distributed.parallel_plan.assert_no_truncation", "hqsb.distributed.parallel_plan.UnsupportedDecision"]),
            (29, "run the independent new-job confirmation", ["hqsb.distributed.parallel_plan.memory_model", "hqsb.distributed.parallel_plan.tp_plan_gate"]),
            (30, "form the scaling input verdict", ["hqsb.distributed.parallel_plan.tp_plan_gate", "hqsb.distributed.ledger.ledger_gate", "hqsb.distributed.parallel_plan.derive_plan"]),
        ]
    ),
)

E10_05 = ExperimentMapping(
    experiment_id="E10-05",
    title="strong/weak/capacity scaling, efficiency and the time decomposition",
    driver="scripts/distributed/run_e10.py --experiment E10-05",
    fixture="configs/distributed/scaling_spec.yaml",
    steps=_steps(
        [
            (1, "verify the E10-01..04 gates", ["hqsb.distributed.parallel_plan.tp_plan_gate", "hqsb.distributed.ledger.ledger_gate"]),
            (2, "freeze the scaling questions", ["hqsb.distributed.scaling.ScalingPreregistration"]),
            (3, "define the strong work unit", ["hqsb.distributed.scaling.WorkUnit"]),
            (4, "define the weak work unit", ["hqsb.distributed.scaling.WeakWorkUnit", "hqsb.distributed.scaling.WEAK_DEFINITIONS"]),
            (5, "define the capacity protocol", ["hqsb.distributed.scaling.CapacityProtocol"]),
            (6, "build the available resource matrix", ["hqsb.distributed.scaling.ResourceMatrix", "hqsb.distributed.scaling.ResourceMatrixCell"]),
            (7, "separate the node families", ["hqsb.distributed.scaling.topology_family_label", "hqsb.distributed.scaling.refuse_mixed_families"]),
            (8, "calibrate the single-device baseline", ["hqsb.distributed.scaling.baseline_calibration", "hqsb.distributed.scaling.Baseline"]),
            (9, "freeze the TP plan per degree", ["hqsb.distributed.parallel_plan.derive_plan", "hqsb.distributed.parallel_plan.ParallelPlan.sha256"]),
            (10, "verify correctness per degree", ["hqsb.distributed.parallel_plan.layer_case", "hqsb.distributed.parallel_plan.first_deviation"]),
            (11, "measure cold/load/init cost", ["hqsb.distributed.backend.ConnectionCostRecord"]),
            (12, "establish the warmup steady state", ["hqsb.distributed.traces.VariabilityBaseline"]),
            (13, "run the strong prefill sweep", ["hqsb.distributed.scaling.decomposition", "hqsb.distributed.scaling.TimeDecomposition"]),
            (14, "run the strong decode sweep", ["hqsb.distributed.scaling.bootstrap_ci"]),
            (15, "run the batch × TP sweep", ["hqsb.distributed.scaling.strong_speedup", "hqsb.distributed.scaling.SpeedupResult"]),
            (16, "run the sequence × TP sweep", ["hqsb.distributed.scaling.weak_efficiency"]),
            (17, "run the request/token weak scaling", ["hqsb.distributed.scaling.device_seconds"]),
            (18, "run the capacity scaling", ["hqsb.distributed.scaling.ResourceMatrix.degrees_for", "hqsb.distributed.scaling.CapacityProtocol"]),
            (19, "collect the per-rank memory breakdown", ["hqsb.distributed.ledger.MemoryRow", "hqsb.distributed.ledger.rank_skew"]),
            (20, "collect the collective ledger per degree", ["hqsb.distributed.ledger.expected_ledger", "hqsb.distributed.ledger.diff_ledger"]),
            (21, "compute strong speedup/efficiency", ["hqsb.distributed.scaling.strong_speedup", "hqsb.distributed.scaling.speedup_from_p0"]),
            (22, "compute weak efficiency", ["hqsb.distributed.scaling.weak_efficiency", "hqsb.distributed.scaling.WEAK_DEFINITIONS"]),
            (23, "decompose compute/comm/wait/idle", ["hqsb.distributed.scaling.decomposition", "hqsb.distributed.overlap.overlap_metrics_from_intervals"]),
            (24, "compare against the E10-02 cost model", ["hqsb.distributed.collectives.project_model_messages"]),
            (25, "fit the scaling model", ["hqsb.distributed.scaling.fit_scaling_model", "hqsb.distributed.scaling.ScalingFit"]),
            (26, "randomise blocks and repeat independently", ["hqsb.distributed.scaling.ScalingPreregistration.independent_runs"]),
            (27, "flag anomalous/straggler runs", ["hqsb.distributed.scaling.flag_anomalies", "hqsb.distributed.scaling.pick_fastest_is_forbidden"]),
            (28, "compute device-seconds", ["hqsb.distributed.scaling.device_seconds"]),
            (29, "run the confirmation matrix", ["hqsb.distributed.scaling.confirmation_plan", "hqsb.distributed.scaling.ConfirmationPlan"]),
            (30, "form the scaling verdict", ["hqsb.distributed.scaling.scaling_verdict", "hqsb.distributed.scaling.split_pairable", "hqsb.distributed.scaling.pairability"]),
        ]
    ),
)

E10_06 = ExperimentMapping(
    experiment_id="E10-06",
    title="communication–computation overlap, stream/event dependencies and chunk A/B",
    driver="scripts/distributed/run_e10.py --experiment E10-06",
    fixture="configs/distributed/overlap_spec.yaml",
    steps=_steps(
        [
            (1, "read the E10-04/05 evidence and pick cases", ["hqsb.distributed.scaling.TimeDecomposition", "hqsb.distributed.scaling.scaling_verdict"]),
            (2, "draw the tensor dependency DAG", ["hqsb.distributed.overlap.TensorDependencyDAG", "hqsb.distributed.overlap.DagNode"]),
            (3, "verify the legal overlap window", ["hqsb.distributed.overlap.legal_overlap_window", "hqsb.distributed.overlap.OverlapBound"]),
            (4, "freeze the A/B/C schedules", ["hqsb.distributed.overlap.ScheduleIdentity", "hqsb.distributed.overlap.SCHEDULE_KINDS", "hqsb.distributed.overlap.validate_only_variable"]),
            (5, "define the chunk/bucket grid", ["hqsb.distributed.overlap.chunk_candidates", "hqsb.distributed.overlap.ChunkCandidate", "hqsb.distributed.overlap.CHUNK_STRATEGIES"]),
            (6, "define the stream/group policy", ["hqsb.distributed.overlap.StreamPolicy", "hqsb.distributed.overlap.audit_stream_policy"]),
            (7, "implement the readiness event", ["hqsb.distributed.overlap.ReadinessEvent"]),
            (8, "implement the completion event/work wait", ["hqsb.distributed.overlap.CompletionEvent", "hqsb.distributed.overlap.validate_work_lifetime"]),
            (9, "implement the actual schedule trace", ["hqsb.distributed.overlap.ScheduleTraceEntry", "hqsb.distributed.overlap.compare_actual_vs_planned"]),
            (10, "run the blocking baseline correctness", ["hqsb.distributed.parallel_plan.layer_case", "hqsb.distributed.parallel_plan.first_deviation"]),
            (11, "run the async-no-overlap control", ["hqsb.distributed.overlap.validate_only_variable", "hqsb.distributed.overlap.ScheduleIdentity"]),
            (12, "run the single-chunk overlap correctness", ["hqsb.distributed.overlap.RaceProbePlan"]),
            (13, "run the multi-chunk correctness", ["hqsb.distributed.overlap.compare_actual_vs_planned", "hqsb.distributed.overlap.ScheduleTraceEntry"]),
            (14, "run the two-stream race probe", ["hqsb.distributed.overlap.RaceProbePlan", "hqsb.distributed.overlap.evaluate_race_probe"]),
            (15, "verify the absence of an implicit global sync", ["hqsb.distributed.overlap.global_sync_audit"]),
            (16, "run the clean no-overlap benchmark", ["hqsb.distributed.overlap.overlap_metrics_from_intervals"]),
            (17, "run the clean overlap benchmark", ["hqsb.distributed.overlap.OverlapMetrics"]),
            (18, "run the chunk sweep", ["hqsb.distributed.overlap.chunk_candidates", "hqsb.distributed.overlap.chunk_overhead"]),
            (19, "run the stream-priority sweep", ["hqsb.distributed.overlap.StreamPolicy", "hqsb.distributed.overlap.audit_stream_policy"]),
            (20, "collect the contrastive timeline", ["hqsb.distributed.overlap.Interval", "hqsb.distributed.overlap.merge_intervals"]),
            (21, "compute the interval-set overlap", ["hqsb.distributed.overlap.union_measure_ns", "hqsb.distributed.overlap.intersection_measure_ns", "hqsb.distributed.overlap.wall_covered_ns"]),
            (22, "decompose the exposed communication", ["hqsb.distributed.overlap.OverlapMetrics.exposed_comm_ms", "hqsb.distributed.overlap.OverlapMetrics"]),
            (23, "quantify resource contention", ["hqsb.distributed.overlap.contention_report"]),
            (24, "quantify chunk overhead", ["hqsb.distributed.overlap.chunk_overhead"]),
            (25, "compare against the theoretical bound", ["hqsb.distributed.overlap.compare_to_bound", "hqsb.distributed.overlap.OverlapBound"]),
            (26, "stratify prefill/decode", ["hqsb.distributed.overlap.phase_policy", "hqsb.distributed.overlap.PhaseShapePolicy"]),
            (27, "run the adversarial boundary cases", ["hqsb.distributed.overlap.chunk_candidates", "hqsb.distributed.overlap.COUNTERFACTUALS"]),
            (28, "run the independent ABBA confirmation", ["hqsb.distributed.overlap.abba_confirmation", "hqsb.distributed.overlap.AbbaConfirmation"]),
            (29, "form the shape/phase-aware policy", ["hqsb.distributed.overlap.phase_policy", "hqsb.distributed.overlap.PhaseShapePolicy"]),
            (30, "form the positive/negative verdict", ["hqsb.distributed.overlap.overlap_verdict", "hqsb.distributed.overlap.CAUSAL_MATRIX", "hqsb.distributed.overlap.OVERLAP_GUARDRAILS"]),
        ]
    ),
)

E10_07 = ExperimentMapping(
    experiment_id="E10-07",
    title="PP/CP/SP selection gate, minimal loop, bubble and memory boundary (P1)",
    driver="scripts/distributed/run_e10.py --experiment E10-07",
    fixture="configs/distributed/boundary_spec.yaml",
    steps=_steps(
        [
            (1, "decide whether the P1 activates", ["hqsb.distributed.boundary.activation_decision", "hqsb.distributed.boundary.ActivationDecision"]),
            (2, "read the E10-04/05/09 evidence", ["hqsb.distributed.traces.PUBLIC_SUMMARY_FIELDS", "hqsb.distributed.scaling.SCALING_ROW_FIELDS"]),
            (3, "freeze the candidate definitions", ["hqsb.distributed.boundary.CandidateDefinition", "hqsb.distributed.boundary.validate_candidates"]),
            (4, "run the capability gate", ["hqsb.distributed.boundary.capability_gate"]),
            (5, "build the analytical cost model", ["hqsb.distributed.boundary.analytical_costs", "hqsb.distributed.boundary.AnalyticalCosts"]),
            (6, "select one primary by the rubric", ["hqsb.distributed.boundary.score_candidates", "hqsb.distributed.boundary.select_primary", "hqsb.distributed.boundary.RUBRIC_TABLE"]),
            (7, "freeze the baseline", ["hqsb.distributed.scaling.baseline_calibration", "hqsb.distributed.scaling.Baseline"]),
            (8, "write the parallel plan", ["hqsb.distributed.boundary.PipelinePlan", "hqsb.distributed.boundary.ContextPlan"]),
            (9, "verify partition conservation", ["hqsb.distributed.boundary.verify_partition"]),
            (10, "establish the communication ledger", ["hqsb.distributed.ledger.ExpectedEvent", "hqsb.distributed.ledger.diff_ledger"]),
            (11, "implement the minimal two-rank path", ["hqsb.distributed.boundary.PipelinePlan", "hqsb.distributed.boundary.ContextPlan"]),
            (12, "verify single-microbatch correctness", ["hqsb.distributed.parallel_plan.layer_case", "hqsb.distributed.parallel_plan.first_deviation"]),
            (13, "verify multi-step greedy decode", ["hqsb.distributed.parallel_plan.decode_alignment_plan"]),
            (14, "run the representative short workload", ["hqsb.distributed.parallel_plan.representative_run_plan"]),
            (15, "generate the main-variable sweep", ["hqsb.distributed.boundary.microbatch_sweep", "hqsb.distributed.boundary.sequence_sweep"]),
            (16, "measure cold/init/transfer buffer", ["hqsb.distributed.backend.ConnectionCostRecord"]),
            (17, "run the clean performance sweep", ["hqsb.distributed.scaling.decomposition", "hqsb.distributed.scaling.bootstrap_ci"]),
            (18, "collect the per-rank timeline", ["hqsb.distributed.traces.PhaseBreakdown", "hqsb.distributed.traces.phase_breakdown"]),
            (19, "verify the theoretical ledger", ["hqsb.distributed.ledger.ledger_gate"]),
            (20, "analyse the PP stage balance", ["hqsb.distributed.boundary.stage_balance", "hqsb.distributed.boundary.StageBalanceRow"]),
            (21, "analyse the PP bubble", ["hqsb.distributed.boundary.bubble_metrics", "hqsb.distributed.boundary.pipeline_bubble_ideal"]),
            (22, "analyse CP/SP attention/KV", ["hqsb.distributed.boundary.cp_sp_metrics"]),
            (23, "analyse the memory benefit", ["hqsb.distributed.boundary.stage_partition_objective", "hqsb.distributed.boundary.stage_balance"]),
            (24, "analyse low/high concurrency", ["hqsb.distributed.boundary.microbatch_sweep", "hqsb.distributed.boundary.adversarial_cases"]),
            (25, "run the placement control", ["hqsb.distributed.placement.PlacementPlan", "hqsb.distributed.placement.alternative_placement"]),
            (26, "run the failure/unsupported smoke", ["hqsb.distributed.boundary.health_probe_plan", "hqsb.distributed.boundary.UNSUPPORTED_CASES"]),
            (27, "compute the benefit/cost versus TP", ["hqsb.distributed.boundary.tp_comparison_rows"]),
            (28, "run adversarial/holdout cases", ["hqsb.distributed.boundary.adversarial_cases"]),
            (29, "run the independent confirmation", ["hqsb.distributed.scaling.confirmation_plan", "hqsb.distributed.scaling.ConfirmationPlan"]),
            (30, "form the adopt/reject verdict", ["hqsb.distributed.boundary.adopt_reject_verdict", "hqsb.distributed.boundary.boundary_verdict", "hqsb.distributed.boundary.ADOPT_REJECT_CRITERIA"]),
        ]
    ),
)

E10_08 = ExperimentMapping(
    experiment_id="E10-08",
    title="MoE expert parallel, dispatch/combine and AllToAllV imbalance",
    driver="scripts/distributed/run_e10.py --experiment E10-08",
    fixture="configs/distributed/moe_spec.yaml",
    steps=_steps(
        [
            (1, "determine the completion level and claim", ["hqsb.distributed.moe.claim_gate", "hqsb.distributed.moe.CLAIM_LEVEL_TABLE"]),
            (2, "read the topology/AllToAll baseline", ["hqsb.distributed.moe.CountMatrix", "hqsb.distributed.collectives.spec_for"]),
            (3, "define the MoE operator spec", ["hqsb.distributed.moe.MoeOperatorSpec"]),
            (4, "define the RouteArtifact", ["hqsb.distributed.moe.generate_route_artifact", "hqsb.distributed.moe.RouteArtifact"]),
            (5, "implement the CPU dispatch oracle", ["hqsb.distributed.moe.dispatch_oracle", "hqsb.distributed.moe.DispatchResult"]),
            (6, "implement the CPU combine oracle", ["hqsb.distributed.moe.combine_oracle"]),
            (7, "construct the hand-computed cases", ["hqsb.distributed.moe.HAND_CASES"]),
            (8, "implement count exchange/validation", ["hqsb.distributed.moe.CountMatrix", "hqsb.distributed.moe.count_matrix", "hqsb.distributed.moe.compute_offsets"]),
            (9, "implement device pack/unpack", ["hqsb.distributed.moe.pack_plan"]),
            (10, "implement dispatch AllToAll(V)", ["hqsb.distributed.moe.DispatchSpec"]),
            (11, "implement the local expert stub", ["hqsb.distributed.moe.deterministic_stub_expert"]),
            (12, "implement combine AllToAll(V)", ["hqsb.distributed.moe.combine_oracle"]),
            (13, "run the L2 end-to-end correctness", ["hqsb.distributed.moe.dispatch_oracle", "hqsb.distributed.moe.combine_oracle"]),
            (14, "connect the real expert MLP", ["hqsb.distributed.moe.expert_mlp_plan", "hqsb.distributed.moe.MoeOperatorSpec"]),
            (15, "generate the fixed skew route set", ["hqsb.distributed.moe.SKEW_PROFILES", "hqsb.distributed.moe.generate_route_artifact"]),
            (16, "run the dispatch/combine micro sweep", ["hqsb.distributed.moe.pack_plan", "hqsb.distributed.moe.DispatchSpec", "hqsb.distributed.moe.imbalance_metrics"]),
            (17, "measure expert compute scaling", ["hqsb.distributed.moe.imbalance_metrics", "hqsb.distributed.moe.expert_mlp_plan"]),
            (18, "compute the imbalance metrics", ["hqsb.distributed.moe.imbalance_metrics", "hqsb.distributed.moe.gini", "hqsb.distributed.moe.entropy"]),
            (19, "verify the communication ledger", ["hqsb.distributed.moe.CountMatrix.audit", "hqsb.distributed.moe.padding_waste"]),
            (20, "compare fixed AllToAll vs AllToAllV", ["hqsb.distributed.moe.padding_waste"]),
            (21, "sweep capacity factor and overflow policy", ["hqsb.distributed.moe.capacity_policy_rows", "hqsb.distributed.moe.OVERFLOW_POLICIES"]),
            (22, "design the baseline placement", ["hqsb.distributed.moe.round_robin_placement", "hqsb.distributed.moe.ExpertPlacement"]),
            (23, "design the topology/load-aware placement", ["hqsb.distributed.moe.topology_aware_placement"]),
            (24, "run the placement A/B", ["hqsb.distributed.moe.placement_ab_rows"]),
            (25, "test the unseen route holdout", ["hqsb.distributed.moe.holdout_check"]),
            (26, "collect the multi-rank profile", ["hqsb.distributed.traces.TRACE_MODES", "hqsb.distributed.traces.MetricManifest"]),
            (27, "run the prefill/decode phase cases", ["hqsb.distributed.moe.phase_cases"]),
            (28, "run the real-model gate (L4)", ["hqsb.distributed.moe.l4_gate"]),
            (29, "run the independent confirmation", ["hqsb.distributed.scaling.confirmation_plan", "hqsb.distributed.moe.holdout_check"]),
            (30, "form the graded verdict", ["hqsb.distributed.moe.moe_verdict", "hqsb.distributed.moe.claim_gate"]),
        ]
    ),
)

E10_09 = ExperimentMapping(
    experiment_id="E10-09",
    title="multi-rank trace alignment, straggler, synchronisation amplification",
    driver="scripts/distributed/run_e10.py --experiment E10-09",
    fixture="configs/distributed/trace_spec.yaml",
    steps=_steps(
        [
            (1, "freeze the profiling questions", ["hqsb.distributed.experiment.Preregistration", "hqsb.distributed.experiment.EXPERIMENTS"]),
            (2, "select the baseline cases", ["hqsb.distributed.scaling.TimeDecomposition", "hqsb.distributed.traces.PUBLIC_SUMMARY_FIELDS"]),
            (3, "freeze the trace modes", ["hqsb.distributed.traces.TRACE_MODES"]),
            (4, "record the metric manifest", ["hqsb.distributed.traces.MetricManifest"]),
            (5, "implement the full-chain correlation", ["hqsb.distributed.traces.CorrelationChain", "hqsb.distributed.traces.DistributedTraceEvent"]),
            (6, "implement clock calibration", ["hqsb.distributed.traces.ClockCalibration", "hqsb.distributed.traces.calibrate_offsets"]),
            (7, "verify collective event pairing", ["hqsb.distributed.traces.pair_collective_events", "hqsb.distributed.traces.PairedCollective"]),
            (8, "measure profiler overhead", ["hqsb.distributed.traces.MetricManifest", "hqsb.distributed.traces.TRACE_MODES"]),
            (9, "collect the healthy baseline all-rank trace", ["hqsb.distributed.traces.VariabilityBaseline"]),
            (10, "build the unified timeline", ["hqsb.distributed.traces.DistributedTraceEvent", "hqsb.distributed.traces.TRACE_EVENT_FIELDS"]),
            (11, "compute the per-rank phase breakdown", ["hqsb.distributed.traces.PhaseBreakdown", "hqsb.distributed.traces.phase_breakdown"]),
            (12, "compute arrival/completion skew", ["hqsb.distributed.traces.arrival_completion_skew"]),
            (13, "map to the communication matrix", ["hqsb.distributed.traces.communication_matrix"]),
            (14, "establish the baseline variability", ["hqsb.distributed.traces.VariabilityBaseline.threshold_ms"]),
            (15, "inject a rank arrival delay", ["hqsb.distributed.traces.injection_plan"]),
            (16, "inject a rank compute slowdown", ["hqsb.distributed.traces.injection_plan"]),
            (17, "inject a host submission delay", ["hqsb.distributed.traces.injection_plan"]),
            (18, "inject expert skew", ["hqsb.distributed.moe.SKEW_PROFILES", "hqsb.distributed.traces.injection_plan"]),
            (19, "use an alternative slow topology", ["hqsb.distributed.placement.alternative_placement", "hqsb.distributed.traces.injection_plan"]),
            (20, "optionally shape the isolated link", ["hqsb.distributed.traces.link_shaping_plan"]),
            (21, "verify the root-cause classifier", ["hqsb.distributed.traces.RootCauseClassifier", "hqsb.distributed.traces.classifier_scores"]),
            (22, "trace the synchronisation amplification", ["hqsb.distributed.traces.amplification_metrics"]),
            (23, "analyse the prefill/decode difference", ["hqsb.distributed.traces.PhaseBreakdown", "hqsb.distributed.traces.trace_verdict"]),
            (24, "verify the overlap attribution", ["hqsb.distributed.traces.overlap_attribution"]),
            (25, "attribute the scaling degradation", ["hqsb.distributed.traces.scaling_degradation_attribution"]),
            (26, "check the unmapped/unknown time", ["hqsb.distributed.traces.unmapped_events", "hqsb.distributed.traces.unmapped_ratio"]),
            (27, "generate the diagnosis runbook", ["hqsb.distributed.traces.straggler_runbook_lines"]),
            (28, "run the independent confirmation trace", ["hqsb.distributed.traces.RootCauseClassifier", "hqsb.distributed.traces.classifier_scores"]),
            (29, "export the public summary", ["hqsb.distributed.traces.public_summary_rows", "hqsb.distributed.telemetry.TABLE_SCHEMAS"]),
            (30, "form the attribution verdict", ["hqsb.distributed.traces.trace_verdict", "hqsb.distributed.traces.EVIDENCE_MATRIX"]),
        ]
    ),
)

E10_10 = ExperimentMapping(
    experiment_id="E10-10",
    title="rank/network/OOM faults, bounded failure and recovery boundaries",
    driver="scripts/distributed/run_e10.py --experiment E10-10",
    fixture="configs/distributed/fault_spec.yaml",
    steps=_steps(
        [
            (1, "freeze the safety scope and blast radius", ["hqsb.distributed.faults.SafetyScope", "hqsb.distributed.faults.FaultInjection"]),
            (2, "read the E10-03/09 gate", ["hqsb.distributed.sequence.tp_safety_gate", "hqsb.distributed.traces.trace_verdict"]),
            (3, "establish the healthy baseline", ["hqsb.distributed.faults.ResourceSnapshot", "hqsb.distributed.experiment.environment_fingerprint"]),
            (4, "define the distributed error taxonomy", ["hqsb.distributed.faults.ERROR_CLASSES", "hqsb.distributed.faults.classify_distributed_error", "hqsb.distributed.faults.ErrorEntry"]),
            (5, "implement the independent control plane", ["hqsb.distributed.faults.ControlHeartbeat", "hqsb.distributed.faults.WATCHDOG_LAYERS"]),
            (6, "implement the multi-level watchdog", ["hqsb.distributed.faults.watchdog_decision"]),
            (7, "implement the global abort coordinator", ["hqsb.distributed.faults.AbortCoordinator"]),
            (8, "implement commit/invalidation semantics", ["hqsb.distributed.faults.COMMIT_STATES", "hqsb.distributed.faults.retryability_after_fault"]),
            (9, "implement resource snapshots", ["hqsb.distributed.faults.ResourceSnapshot", "hqsb.distributed.faults.snapshot_delta"]),
            (10, "inject a clean rank exit", ["hqsb.distributed.faults.oracle_for_fault", "hqsb.distributed.faults.FAULT_MATRIX"]),
            (11, "inject SIGTERM", ["hqsb.distributed.faults.FaultInjection", "hqsb.distributed.faults.evaluate_fault"]),
            (12, "inject SIGKILL", ["hqsb.distributed.faults.FaultOracle", "hqsb.distributed.faults.evaluate_fault"]),
            (13, "inject a rank hang", ["hqsb.distributed.faults.watchdog_decision", "hqsb.distributed.faults.ERROR_CATALOG"]),
            (14, "inject a compute straggler timeout", ["hqsb.distributed.sequence.delayed_rank_ladder", "hqsb.distributed.sequence.evaluate_delayed_case"]),
            (15, "inject a shard-load OOM", ["hqsb.distributed.faults.oracle_for_fault", "hqsb.distributed.faults.MINIMUM_RECOVERY_BY_FAULT"]),
            (16, "inject a workspace/activation OOM", ["hqsb.distributed.faults.FailureObservation", "hqsb.distributed.faults.evaluate_fault"]),
            (17, "inject a KV growth OOM", ["hqsb.distributed.faults.retryability_after_fault", "hqsb.distributed.faults.COMMIT_STATES"]),
            (18, "inject a communicator allocation failure", ["hqsb.distributed.faults.ERROR_CATALOG", "hqsb.distributed.faults.oracle_for_fault"]),
            (19, "inject a bootstrap/connect failure", ["hqsb.distributed.faults.oracle_for_fault", "hqsb.distributed.faults.FaultOracle"]),
            (20, "inject an in-flight network disconnect", ["hqsb.distributed.faults.FaultInjection", "hqsb.distributed.faults.SafetyScope.validate"]),
            (21, "inject link delay/loss", ["hqsb.distributed.traces.link_shaping_plan", "hqsb.distributed.faults.FaultInjection"]),
            (22, "verify error propagation to every rank", ["hqsb.distributed.sequence.ErrorPropagationChannel", "hqsb.distributed.faults.AbortCoordinator.terminal_states_complete"]),
            (23, "verify communicator abort", ["hqsb.distributed.sequence.CommunicatorStateMachine.fail", "hqsb.distributed.ranks.COMMUNICATOR_STATES"]),
            (24, "verify process and resource cleanup", ["hqsb.distributed.faults.snapshot_delta", "hqsb.distributed.faults.RESOURCE_DELTA_KINDS"]),
            (25, "attempt an in-process communicator recreate", ["hqsb.distributed.sequence.SequenceAllocator.reset_for_epoch", "hqsb.distributed.faults.RECOVERY_LEVELS"]),
            (26, "execute the process-group restart", ["hqsb.distributed.faults.RECOVERY_LEVELS", "hqsb.distributed.experiment.RunDirectory"]),
            (27, "run the layered health probe", ["hqsb.distributed.boundary.health_probe_plan", "hqsb.distributed.boundary.PROBE_CHAIN"]),
            (28, "verify the retry/idempotency budget", ["hqsb.distributed.faults.retryability_after_fault", "hqsb.distributed.faults.RETRY_DECISIONS"]),
            (29, "run the blind runbook drill", ["hqsb.distributed.faults.DrillResult"]),
            (30, "form the boundedness verdict", ["hqsb.distributed.faults.boundedness_summary", "hqsb.distributed.faults.should_stop_campaign", "hqsb.distributed.faults.TimeMetrics"]),
        ]
    ),
)

EXPERIMENTS: Tuple[ExperimentMapping, ...] = (
    E10_01,
    E10_02,
    E10_03,
    E10_04,
    E10_05,
    E10_06,
    E10_07,
    E10_08,
    E10_09,
    E10_10,
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
    lines = [
        f"### {mapping.experiment_id}: {mapping.title}",
        "",
        f"驱动：`{mapping.driver}`",
        "",
    ]
    lines.append("| 步骤 | 步骤内容 | 代码接口 | 成熟度 |")
    lines.append("|---|---|---|---|")
    for step in mapping.steps:
        interfaces = "<br>".join(f"`{item}`" for item in step.interfaces)
        lines.append(f"| {step.step} | {step.title} | {interfaces} | {step.maturity} |")
    return "\n".join(lines) + "\n"


__all__ = [
    "E10_01",
    "E10_02",
    "E10_03",
    "E10_04",
    "E10_05",
    "E10_06",
    "E10_07",
    "E10_08",
    "E10_09",
    "E10_10",
    "EXPERIMENTS",
    "ExperimentMapping",
    "MAPPINGS",
    "StepMapping",
    "mapping_for",
    "mapping_table_markdown",
    "resolve_interfaces",
    "step_table_for",
]
