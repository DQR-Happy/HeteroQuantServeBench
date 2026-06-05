"""Experiment step → code interface map (the "no missing capability" proof).

Each S06 experiment document lists concrete steps ("具体实验步骤").  This module
records, for **every** step, which HQSB interface implements it, and
:func:`resolve_interfaces` verifies that each referenced symbol actually imports
— so the map cannot rot into documentation.  A test
(``tests/unit/integration/test_experiment_interface_map.py``) runs the
verification.

Maturity labels follow the control plane (M0–M7):

* ``M1`` — source exists (interface implemented, structured errors);
* ``M2`` — covered by automated tests in this tree;
* ``M3``+ — requires real hardware/model evidence, which this repository state
  cannot produce (the S04.5 M4 prerequisite is unmet), so no entry is labelled
  above M2.

Drivers live in ``scripts/integration/run_e06.py`` and are *not* executed by
this map.  Step counts come from the protocol files
``docs/stage_experiments/details/S06/E06-*.md``.
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


E06_01 = ExperimentMapping(
    experiment_id="E06-01",
    title="custom op registration, dispatcher, redispatch and conflicts",
    driver="scripts/integration/run_e06.py --experiment E06-01",
    fixture="hqsb.integration.specs.frozen_schemas / dispatch.frozen_p0_matrix",
    steps=_steps(
        [
            (1, "freeze OperatorSpec", ["hqsb.integration.specs.schema_from_operator_spec", "hqsb.integration.specs.OpSchema.schema_hash", "hqsb.integration.policies.load_operator_contracts"], "mutation/alias/policy cannot be defaulted"),
            (2, "single schema owner", ["hqsb.integration.specs.audit_schema_owners", "hqsb.integration.specs.assert_single_owner", "hqsb.integration.specs.OwnerAudit"], "python and C++ definition sites are counted"),
            (3, "register CPU/composite reference", ["hqsb.integration.dispatch.RegistrationMatrix.declare_schema", "hqsb.integration.dispatch.frozen_p0_matrix", "hqsb.integration.differential.composed_add_rms_norm_reference"], "reference never calls the custom op"),
            (4, "register CUDA and internal backend route", ["hqsb.integration.dispatch.RegistrationRecord", "hqsb.integration.dispatch.RegistrationMatrix.register", "hqsb.integration.lowering.frozen_registry"], "Triton/quant stay an internal route, not a key"),
            (5, "register Meta/Fake, autocast, autograd policy", ["hqsb.integration.meta.frozen_contracts", "hqsb.integration.specs.OpSchema.autocast_policy", "hqsb.integration.dispatch.DispatchKey"], "policy is explicit per op"),
            (6, "dispatch table snapshot", ["hqsb.integration.dispatch.snapshot_from_matrix", "hqsb.integration.dispatch.DispatchSnapshot.diff", "hqsb.integration.dispatch.RegistrationMatrix.matrix_hash"], "snapshots are generated, never hand-written"),
            (7, "basic device/dtype matrix", ["hqsb.integration.dispatch.select_implementation", "hqsb.integration.dispatch.CapabilityRequest", "hqsb.integration.dispatch.KernelObservation"], "requested/actual/kernel recorded together"),
            (8, "redispatch path test", ["hqsb.integration.dispatch.RedispatchGuard", "hqsb.integration.dispatch.redispatch_keyset", "hqsb.integration.dispatch.RedispatchError"], "recursion and key-set loss are refused"),
            (9, "alias/mutation oracle", ["hqsb.integration.specs.OpSchema.validate", "hqsb.integration.specs.AliasSpec", "hqsb.integration.specs.MutationSpec"], "schema must match the kernel's writes"),
            (10, "functionalization", ["hqsb.integration.specs.OpSchema.alias_signature", "hqsb.integration.meta.TensorMeta.alias_of_input", "hqsb.integration.meta.compare_metadata"], "functional contract for the fused op"),
            (11, "inference/autograd negative tests", ["hqsb.integration.specs.OpSchema.autograd_policy", "hqsb.integration.specs.AUTOGRAD_POLICIES", "hqsb.integration.taxonomy.classify_exception"], "no silent detach"),
            (12, "autocast matrix", ["hqsb.integration.specs.OpSchema.autocast_policy", "hqsb.integration.specs.AUTOCAST_POLICIES", "hqsb.integration.dispatch.DispatchKey.AUTOCAST_CUDA"], "input/accumulation/output dtype recorded"),
            (13, "opcheck", ["hqsb.integration.dispatch.opcheck_plan", "hqsb.integration.dispatch.OpcheckItem", "hqsb.integration.meta.contract_table"], "subtests saved individually, not as one PASS"),
            (14, "repeated load", ["hqsb.integration.dispatch.RegistrationAction.IDEMPOTENT", "hqsb.integration.dispatch.RegistrationMatrix.register", "hqsb.integration.experiment.RunDirectory.record_command"], "idempotent or explicitly refused"),
            (15, "namespace/schema conflict injection", ["hqsb.integration.dispatch.RegistrationAction", "hqsb.integration.dispatch.CLAIMED_NAMESPACES", "hqsb.integration.specs.NAMESPACE"], "no last-load-wins"),
            (16, "fresh process and spawn", ["hqsb.integration.experiment.interface_only_run", "hqsb.integration.experiment.environment_fingerprint", "hqsb.integration.dispatch.RegistrationMatrix.as_dict"], "no parent-process registration state"),
            (17, "ABI negative matrix", ["hqsb.integration.abi.preload_check", "hqsb.integration.abi.simulate_identity", "hqsb.integration.abi.ABI_REQUIREMENTS", "hqsb.integration.abi.BinaryManifest"], "refused before dlopen/kernel"),
            (18, "fallback and strict mode", ["hqsb.integration.dispatch.FallbackPolicy", "hqsb.integration.dispatch.RouteDecision", "hqsb.integration.taxonomy.FallbackRegistry"], "strict mode must fail loudly"),
            (19, "real Qwen invocation", ["hqsb.integration.adapter.model_census", "hqsb.integration.telemetry.TraceCollector.emit", "hqsb.integration.dispatch.KernelObservation"], "module call joins the dispatch span"),
            (20, "registration audit table", ["hqsb.integration.dispatch.DispatchSnapshot.as_dict", "hqsb.integration.telemetry.project_c6", "hqsb.integration.specs.schema_table"], "generated from raw snapshots and cases"),
        ]
    ),
)

E06_02 = ExperimentMapping(
    experiment_id="E06-02",
    title="Meta/FakeTensor, export and symbolic shape",
    driver="scripts/integration/run_e06.py --experiment E06-02",
    fixture="hqsb.integration.meta.frozen_contracts / graph.from_node_sequence",
    steps=_steps(
        [
            (1, "freeze metadata contract", ["hqsb.integration.meta.contract_by_name", "hqsb.integration.meta.frozen_contracts", "hqsb.integration.meta.MetadataContract.as_dict"], "shape/dtype/stride/alias/error per op"),
            (2, "hand-computed golden", ["hqsb.integration.meta.SymInt", "hqsb.integration.meta.dims_evaluate", "hqsb.integration.meta.TensorMeta"], "golden is independent of the real op"),
            (3, "direct Meta", ["hqsb.integration.meta.MetadataContract.infer_outputs", "hqsb.integration.meta.FakeCallSpy"], "no CUDA kernel, no payload read"),
            (4, "FakeTensorMode", ["hqsb.integration.meta.TensorMeta.as_dict", "hqsb.integration.meta.FakeCallSpy.record_device_api", "hqsb.integration.meta.MetadataContract.infer_outputs"], "logical device preserved"),
            (5, "real-vs-fake oracle", ["hqsb.integration.meta.compare_metadata", "hqsb.integration.meta.MetadataDiff.as_dict"], "field-by-field, not shape-only"),
            (6, "alias/mutation oracle", ["hqsb.integration.meta.TensorMeta.alias_of_input", "hqsb.integration.specs.OpSchema.alias_signature", "hqsb.integration.meta.compare_metadata"], "fake agrees with schema and real"),
            (7, "stride/layout matrix", ["hqsb.integration.meta.OutputRule.stride_policy", "hqsb.integration.meta.TensorMeta.is_contiguous", "hqsb.integration.meta.MetadataError"], "preserve / materialize / reject is explicit"),
            (8, "symbolic batch/ISL", ["hqsb.integration.meta.SymInt.symbol", "hqsb.integration.meta.SymInt.symbols", "hqsb.integration.meta.dims_repr"], "example values are not baked in"),
            (9, "quant logical/physical shape", ["hqsb.integration.meta.dequant_linear_contract", "hqsb.integration.meta.OutputRule", "hqsb.integration.patterns.QuantDescriptor"], "packed bytes never become logical K"),
            (10, "opcheck fake subtests", ["hqsb.integration.dispatch.opcheck_plan", "hqsb.integration.meta.FakeCallSpy"], "each fake subtest recorded separately"),
            (11, "FX shape propagation", ["hqsb.integration.graph.from_node_sequence", "hqsb.integration.graph.GraphNode.meta", "hqsb.integration.meta.GuardSet"], "node metadata + failing node saved"),
            (12, "Dynamo capture backend", ["hqsb.integration.graph.from_fx_graph_module", "hqsb.integration.graph.CaptureMode", "hqsb.integration.guards.CompileLedger.record"], "capture mode and graph saved"),
            (13, "export strict/non-strict", ["hqsb.integration.graph.CaptureMode.EXPORT_STRICT", "hqsb.integration.graph.CaptureMode.EXPORT_NON_STRICT", "hqsb.integration.meta.GuardSet.as_dict"], "modes are never mixed"),
            (14, "compile consumer", ["hqsb.integration.lowering.frozen_registry", "hqsb.integration.lowering.LoweringRequest", "hqsb.integration.meta.GuardSet.data_dependent"], "missing fake would show up as a break"),
            (15, "verify no real allocation/kernel", ["hqsb.integration.meta.FakeCallSpy.clean", "hqsb.integration.meta.FakeCallSpy.as_dict"], "allocation delta alone is not enough"),
            (16, "invalid metadata negative tests", ["hqsb.integration.meta.MetadataContract.validate_inputs", "hqsb.integration.meta.MetadataError", "hqsb.integration.meta.Constraint"], "diagnosed at the agreed stage"),
            (17, "symbolic bounds boundaries", ["hqsb.integration.guards.DynamicDimensionSpec", "hqsb.integration.guards.dynamic_correctness_points", "hqsb.integration.guards.resolve_compiler_config"], "out-of-bounds is refused, not padded"),
            (18, "fresh-process replay", ["hqsb.integration.cache.graph_identity", "hqsb.integration.experiment.environment_fingerprint", "hqsb.integration.experiment.interface_only_run"], "fake rules rebuild identically"),
            (19, "real Qwen graph metadata audit", ["hqsb.integration.adapter.model_census", "hqsb.integration.graph.GraphDiff", "hqsb.integration.meta.compare_metadata"], "before/after node metadata preserved"),
            (20, "metadata difference table", ["hqsb.integration.meta.MetadataDiff.as_dict", "hqsb.integration.meta.contract_table", "hqsb.integration.telemetry.dump_json"], "table generated from raw data"),
        ]
    ),
)

E06_03 = ExperimentMapping(
    experiment_id="E06-03",
    title="eager/custom/fused/compiled four-level differential correctness",
    driver="scripts/integration/run_e06.py --experiment E06-03",
    fixture="hqsb.integration.differential.default_tolerance_registry",
    steps=_steps(
        [
            (1, "freeze DifferentialSpec", ["hqsb.integration.differential.DifferentialSpec", "hqsb.integration.differential.default_tolerance_registry", "hqsb.integration.differential.ToleranceRegistry"], "model/input/quant/tolerance bound before results"),
            (2, "replay eager reference", ["hqsb.integration.differential.compare_arrays", "hqsb.integration.differential.reference_diff_summary", "hqsb.integration.differential.PATHS"], "golden drift detected, not papered over"),
            (3, "mode switch and restore", ["hqsb.integration.differential.ModeSwitchAudit", "hqsb.integration.differential.StateSnapshot", "hqsb.integration.differential.compare_state"], "no stale compiled wrapper after disable"),
            (4, "eager custom operator", ["hqsb.integration.dispatch.select_implementation", "hqsb.integration.dispatch.FallbackPolicy", "hqsb.integration.differential.DifferentialPath"], "one factor changed at a time"),
            (5, "explicit fused reference", ["hqsb.integration.differential.FROZEN_ADD_RMSNORM_SEMANTICS", "hqsb.integration.differential.composed_add_rms_norm_reference", "hqsb.integration.differential.FusionSemantics"], "composition, not the target kernel, is the oracle"),
            (6, "eager fused kernel", ["hqsb.integration.differential.fused_add_rms_norm_reference", "hqsb.integration.differential.compare_arrays"], "alias/dtype/stream recorded with the numbers"),
            (7, "compile baseline", ["hqsb.integration.guards.CompileLedger", "hqsb.integration.guards.FiveWayCounts", "hqsb.integration.cache.PhaseTimer"], "compiler-only difference quantified"),
            (8, "compile custom opaque", ["hqsb.integration.graph.CaptureMode", "hqsb.integration.dispatch.DispatchTableEntry", "hqsb.integration.telemetry.S06ResultFields"], "actual dispatch after compilation"),
            (9, "pattern + reference lowering", ["hqsb.integration.patterns.apply_rewrite", "hqsb.integration.lowering.LoweringDecision", "hqsb.integration.lowering.LoweringRegistry"], "rewrite isolated from lowering"),
            (10, "pattern + HQSB lowering", ["hqsb.integration.lowering.LoweringRegistry.select", "hqsb.integration.dispatch.KernelObservation", "hqsb.integration.telemetry.TraceCollector"], "observed kernel recorded"),
            (11, "operator random/boundary differential", ["hqsb.integration.differential.compare_arrays", "hqsb.integration.differential.ToleranceRegistry.get", "hqsb.integration.meta.MetadataError"], "per dtype/level thresholds"),
            (12, "single block node-by-node", ["hqsb.integration.differential.DivergenceLocator", "hqsb.integration.differential.DivergenceRecord", "hqsb.integration.graph.GraphNode"], "first inconsistent node retained"),
            (13, "full model teacher forcing", ["hqsb.integration.differential.Level.MODEL_LOGITS", "hqsb.integration.differential.CorrectnessMatrix", "hqsb.integration.differential.ToleranceSpec"], "logit metrics, not only final text"),
            (14, "cache-enabled generation", ["hqsb.integration.differential.Level.GENERATION", "hqsb.integration.telemetry.TraceCollector.emit", "hqsb.integration.differential.DivergenceLocator.observe"], "per-step evidence and first divergence"),
            (15, "autocast/stream matrix", ["hqsb.integration.lifecycle.audit_streams", "hqsb.integration.lifecycle.StreamEvent", "hqsb.integration.specs.OpSchema.autocast_policy"], "no double cast, no changed current stream"),
            (16, "fallback differential", ["hqsb.integration.dispatch.FallbackPolicy.resolve", "hqsb.integration.taxonomy.FallbackRegistry.resolve", "hqsb.integration.taxonomy.requested_actual_chain"], "fallback output matches the reference"),
            (17, "cold/warm/cache replay", ["hqsb.integration.cache.CacheState", "hqsb.integration.cache.PhaseTimer", "hqsb.integration.cache.CompileIdentity"], "correctness independent of trace state"),
            (18, "quantized path", ["hqsb.integration.patterns.QuantDescriptor", "hqsb.integration.differential.compare_arrays", "hqsb.integration.adapter.identity_collision_check"], "S05 quality gate still referenced"),
            (19, "first-divergence localisation", ["hqsb.integration.differential.DivergenceLocator.first_over", "hqsb.integration.graph.GraphNode", "hqsb.integration.telemetry.TraceCollector"], "graph node → op → layer → step"),
            (20, "correctness matrix", ["hqsb.integration.differential.CorrectnessMatrix", "hqsb.integration.differential.CellStatus", "hqsb.integration.differential.path_matrix_table"], "missing cells stay explicit"),
        ]
    ),
)

E06_04 = ExperimentMapping(
    experiment_id="E06-04",
    title="pattern coverage, reject reasons and semantic safety",
    driver="scripts/integration/run_e06.py --experiment E06-04",
    fixture="hqsb.integration.patterns.frozen_patterns / standard_mutations",
    steps=_steps(
        [
            (1, "freeze PatternSpec", ["hqsb.integration.patterns.PatternSpec", "hqsb.integration.patterns.Structure", "hqsb.integration.policies.audit_pattern_declarations"], "IR level/capture mode/predicates bound"),
            (2, "positive golden graphs", ["hqsb.integration.graph.from_node_sequence", "hqsb.integration.graph.Graph.structural_hash", "hqsb.integration.patterns.PatternContext"], "equivalence proven by E06-03"),
            (3, "negative corpus", ["hqsb.integration.patterns.standard_mutations", "hqsb.integration.patterns.Mutation", "hqsb.integration.patterns.RejectReason"], "one semantic factor per case"),
            (4, "six-workload before graph", ["hqsb.integration.graph.CaptureMode", "hqsb.integration.graph.Graph.as_dict", "hqsb.integration.guards.ordered_shape_trace"], "capture mode/decomposition/module stack saved"),
            (5, "enumerate structural candidates", ["hqsb.integration.patterns.find_matches", "hqsb.integration.patterns.Match"], "no rewrite in this step"),
            (6, "run semantic predicates", ["hqsb.integration.patterns.evaluate_candidate", "hqsb.integration.patterns.PredicateResult", "hqsb.integration.patterns.PredicateSpec"], "audit mode evaluates all factors"),
            (7, "alias/side-effect analysis", ["hqsb.integration.graph.Graph.users_of", "hqsb.integration.graph.GraphNode.is_impure", "hqsb.integration.patterns.PatternContext"], "extra users and impure nodes checked"),
            (8, "quant/capability validation", ["hqsb.integration.patterns.QuantDescriptor", "hqsb.integration.dispatch.OperatorCapability.supports", "hqsb.integration.dispatch.CapabilityRequest"], "layout mismatch caught before the kernel"),
            (9, "apply rewrite to a copy", ["hqsb.integration.patterns.apply_rewrite", "hqsb.integration.graph.GraphDiff", "hqsb.integration.patterns.RewriteRecord"], "the original graph is never polluted"),
            (10, "local differential", ["hqsb.integration.differential.compare_arrays", "hqsb.integration.differential.FROZEN_ADD_RMSNORM_SEMANTICS", "hqsb.integration.differential.CorrectnessMatrix"], "E06-03 oracle per pattern version"),
            (11, "block/model validation", ["hqsb.integration.differential.Level.BLOCK", "hqsb.integration.differential.Level.MODEL_LOGITS", "hqsb.integration.differential.CorrectnessMatrix.mark"], "every real hit passes, not a sample"),
            (12, "verify lowering hit", ["hqsb.integration.lowering.LoweringRegistry.select", "hqsb.integration.lowering.LoweringDecision.as_dict", "hqsb.integration.guards.EventKind.LOWERING_SELECT"], "PATTERN_HIT ≠ LOWERING_SELECTED ≠ KERNEL_OBSERVED"),
            (13, "dynamic call coverage", ["hqsb.integration.patterns.RuntimeUse", "hqsb.integration.patterns.coverage_report", "hqsb.integration.telemetry.TraceCollector"], "static nodes mapped to call counts"),
            (14, "four coverage numbers", ["hqsb.integration.patterns.CoverageReport", "hqsb.integration.patterns.coverage_report", "hqsb.integration.graph.graph_summary"], "node/call/time/model reported separately"),
            (15, "false-positive mutation testing", ["hqsb.integration.patterns.mutation_report", "hqsb.integration.patterns.standard_mutations", "hqsb.integration.patterns.MutationOutcome"], "predicates must still reject"),
            (16, "version/capture drift", ["hqsb.integration.patterns.PatternRegistry", "hqsb.integration.graph.CaptureMode", "hqsb.integration.patterns.RejectReason.VERSION_UNSUPPORTED"], "unknown shapes default to reject"),
            (17, "manual review of unknown", ["hqsb.integration.patterns.DecisionStatus.UNKNOWN_NEEDS_REVIEW", "hqsb.integration.patterns.Decision.as_dict"], "review evidence kept, never a substitute for tests"),
            (18, "coverage/reject report", ["hqsb.integration.patterns.CoverageReport.as_dict", "hqsb.integration.patterns.Decision.as_dict", "hqsb.integration.telemetry.dump_json"], "every eligible/hit/reject links to graph and case"),
        ]
    ),
)

E06_05 = ExperimentMapping(
    experiment_id="E06-05",
    title="dynamic shape, guards, graph breaks and recompile storms",
    driver="scripts/integration/run_e06.py --experiment E06-05",
    fixture="hqsb.integration.guards.ordered_shape_trace",
    steps=_steps(
        [
            (1, "freeze DynamicSpec", ["hqsb.integration.guards.DynamicDimensionSpec", "hqsb.integration.guards.resolve_compiler_config", "hqsb.integration.policies.load_compile_policy"], "bounds/buckets/policy/storm thresholds"),
            (2, "static baseline", ["hqsb.integration.guards.DynamicPolicy.STATIC", "hqsb.integration.guards.CompilerConfig", "hqsb.integration.cache.PhaseTimer"], "per-shape specialisation as the upper bound"),
            (3, "locked-version default policy", ["hqsb.integration.guards.DynamicPolicy.AUTO", "hqsb.integration.guards.resolve_compiler_config"], "resolved config is recorded, not assumed"),
            (4, "dynamic=True / explicit bounds", ["hqsb.integration.guards.DynamicPolicy.DYNAMIC", "hqsb.integration.guards.DynamicPolicy.EXPLICIT_BOUNDS", "hqsb.integration.guards.DynamicDimensionSpec.bucket_for"], "symbols, constraints and kernels recorded"),
            (5, "phase split", ["hqsb.integration.guards.DynamicPolicy.PHASE_SPLIT", "hqsb.integration.guards.CompilerConfig.as_dict"], "prefill and decode compiled separately"),
            (6, "bucket policy", ["hqsb.integration.guards.DynamicPolicy.BUCKETED", "hqsb.integration.cuda_graph.BucketPlan", "hqsb.integration.guards.DynamicDimensionSpec.buckets"], "padding cost is measured, not ignored"),
            (7, "ordered shape trace replay", ["hqsb.integration.guards.ordered_shape_trace", "hqsb.integration.guards.TraceRequestRecord", "hqsb.integration.guards.CompileLedger.record"], "per-request guard/break/recompile/cache events"),
            (8, "repeat an old shape", ["hqsb.integration.guards.reuse_evidence", "hqsb.integration.guards.TraceRequestRecord.graph_variant"], "variant reuse must be provable"),
            (9, "stride/layout sweep", ["hqsb.integration.meta.OutputRule.stride_policy", "hqsb.integration.dispatch.CapabilityRequest.layout", "hqsb.integration.guards.GuardKind.STRIDE"], "copy vs fallback is explicit"),
            (10, "python/config state sweep", ["hqsb.integration.guards.GuardKind.PYTHON_SCALAR", "hqsb.integration.guards.GuardRecord.introduced_by", "hqsb.integration.guards.audit_guard_minimality"], "unrelated-object guards located"),
            (11, "quant artifact / kernel capability sweep", ["hqsb.integration.patterns.QuantDescriptor.artifact_hash", "hqsb.integration.cache.graph_identity", "hqsb.integration.lowering.LoweringCapability"], "layout changes must invalidate"),
            (12, "graph break audit", ["hqsb.integration.guards.EventKind.GRAPH_BREAK", "hqsb.integration.guards.CompileLedger.five_way", "hqsb.integration.guards.CompileLedger.reasons_for"], "fullgraph and partial modes kept apart"),
            (13, "trigger recompile limit", ["hqsb.integration.guards.StormObservation.recompile_limit_reached", "hqsb.integration.guards.evaluate_storm", "hqsb.integration.guards.StormThresholds"], "isolated fixture, behaviour recorded"),
            (14, "boundary correctness", ["hqsb.integration.guards.dynamic_correctness_points", "hqsb.integration.differential.compare_arrays", "hqsb.integration.differential.DivergenceLocator"], "lower/middle/upper/out-of-bounds"),
            (15, "real distribution trace", ["hqsb.integration.guards.ShapeTracePoint", "hqsb.integration.guards.TraceRequestRecord.as_dict", "hqsb.integration.experiment.RunDirectory.record_command"], "synthetic distribution is labelled as such"),
            (16, "guard overhead", ["hqsb.integration.guards.GuardReview", "hqsb.integration.telemetry.S06ResultFields", "hqsb.integration.cache.PhaseTimer"], "host overhead separated from device time"),
            (17, "transition tail latency", ["hqsb.integration.guards.StormObservation", "hqsb.integration.guards.evaluate_storm"], "first new shape and fallback reported separately"),
            (18, "storm root cause", ["hqsb.integration.guards.StormReport.as_dict", "hqsb.integration.guards.CompileLedger", "hqsb.integration.guards.evaluate_storm"], "classified by shape/stride/object/scalar"),
            (19, "select policy", ["hqsb.integration.guards.evaluate_storm", "hqsb.integration.cache.compute_break_even", "hqsb.integration.guards.resolve_compiler_config"], "correctness → storm → steady trade-off"),
            (20, "new-process re-verification", ["hqsb.integration.cache.CacheState.WARM_NEW_PROCESS_LOCAL", "hqsb.integration.cache.CacheManifest", "hqsb.integration.experiment.environment_fingerprint"], "policy independent of old-process state"),
        ]
    ),
)

E06_06 = ExperimentMapping(
    experiment_id="E06-06",
    title="compile cold start, cache layers, invalidation and corruption",
    driver="scripts/integration/run_e06.py --experiment E06-06",
    fixture="hqsb.integration.cache / policies.load_cache_spec",
    steps=_steps(
        [
            (1, "freeze CacheSpec", ["hqsb.integration.policies.load_cache_spec", "hqsb.integration.cache.CacheState", "hqsb.integration.cache.default_invalidation_matrix"], "layers/states/identity/invalidation frozen"),
            (2, "task-specific cache directory", ["hqsb.integration.cache.atomic_write_text", "hqsb.integration.cache.save_manifest", "hqsb.integration.experiment.RunDirectory.create"], "no shared /tmp residue"),
            (3, "COLD-EMPTY", ["hqsb.integration.cache.CacheState.COLD_EMPTY", "hqsb.integration.cache.PhaseTimer", "hqsb.integration.cache.CompilePhase"], "all phases and artifacts recorded"),
            (4, "WARM-SAME-PROCESS", ["hqsb.integration.cache.CacheState.WARM_SAME_PROCESS", "hqsb.integration.cache.PhaseTimer.record", "hqsb.integration.cache.CacheEntry"], "first/second/steady separated"),
            (5, "WARM-NEW-PROCESS-LOCAL", ["hqsb.integration.cache.validate_entry", "hqsb.integration.cache.load_manifest", "hqsb.integration.cache.CacheState.WARM_NEW_PROCESS_LOCAL"], "disk entry actually read and validated"),
            (6, "CACHE-DISABLED", ["hqsb.integration.cache.CacheState.DISABLED", "hqsb.integration.cache.PhaseTimer"], "control for the cache lookup cost"),
            (7, "six workload / shape policy", ["hqsb.integration.guards.ordered_shape_trace", "hqsb.integration.cache.CacheManifest.layers", "hqsb.integration.cache.CacheSizeTracker.evaluate"], "graph variants and reuse per workload"),
            (8, "factor-by-factor invalidation", ["hqsb.integration.cache.evaluate_invalidation", "hqsb.integration.cache.InvalidationObservation", "hqsb.integration.cache.InvalidationFactor"], "one factor per run, expected action frozen"),
            (9, "model/weight/quant identity", ["hqsb.integration.cache.graph_identity", "hqsb.integration.patterns.QuantDescriptor.artifact_hash", "hqsb.integration.cache.CompileIdentity.digest"], "same shape must not bind new weights"),
            (10, "schema/binary/ABI", ["hqsb.integration.cache.IntegrityFailure.STALE_SCHEMA_ABI", "hqsb.integration.abi.preload_check", "hqsb.integration.cache.validate_entry"], "old binary refused by the new schema"),
            (11, "pytorch/triton/cuda/arch", ["hqsb.integration.cache.CompileIdentity", "hqsb.integration.abi.CompatibilityMatrix", "hqsb.integration.abi.ABI_REQUIREMENTS"], "unrun combinations count only as schema negatives"),
            (12, "corrupt cache", ["hqsb.integration.cache.CorruptionFixture", "hqsb.integration.cache.validate_entry", "hqsb.integration.cache.IntegrityFailure"], "validated before execution, golden untouched"),
            (13, "concurrent writers/readers", ["hqsb.integration.cache.CacheLock", "hqsb.integration.cache.CacheLockError", "hqsb.integration.cache.atomic_write_bytes"], "only complete entries are visible"),
            (14, "writer interruption / disk error", ["hqsb.integration.cache.CorruptionFixture.stale_lock", "hqsb.integration.cache.CacheEntry.complete", "hqsb.integration.cache.validate_entry"], "atomicity, cleanup and recovery"),
            (15, "imported/remote cache", ["hqsb.integration.cache.CacheState.IMPORTED_REMOTE", "hqsb.integration.policies.CacheSpecDocument.imported_remote_claimed"], "otherwise NOT_CLAIMED"),
            (16, "cache growth", ["hqsb.integration.cache.CacheSizeTracker.evaluate", "hqsb.integration.cache.evict", "hqsb.integration.cache.CacheManifest.total_bytes"], "entry/byte growth against the cap"),
            (17, "break-even", ["hqsb.integration.cache.compute_break_even", "hqsb.integration.cache.BreakEven.as_dict"], "per workload, no hidden cold cost"),
            (18, "correctness and observed kernel", ["hqsb.integration.differential.compare_arrays", "hqsb.integration.dispatch.KernelObservation", "hqsb.integration.cache.validate_entry"], "hit/miss/recompile paths re-checked"),
            (19, "invalidation matrix", ["hqsb.integration.cache.evaluate_invalidation", "hqsb.integration.telemetry.dump_json", "hqsb.integration.cache.InvalidationAction"], "generated from raw, unfavourable rows kept"),
            (20, "fresh-host boundary note", ["hqsb.integration.policies.CacheSpecDocument.as_dict", "hqsb.integration.experiment.environment_fingerprint", "hqsb.integration.cache.CacheState"], "new-process ≠ portable cache"),
        ]
    ),
)

E06_07 = ExperimentMapping(
    experiment_id="E06-07",
    title="fusion/lowering to real kernels, allocation and model benefit",
    driver="scripts/integration/run_e06.py --experiment E06-07",
    fixture="hqsb.integration.lowering.frozen_registry / ablation_matrix",
    steps=_steps(
        [
            (1, "freeze FusionExperimentSpec", ["hqsb.integration.lowering.frozen_registry", "hqsb.integration.differential.DifferentialSpec", "hqsb.integration.policies.load_compile_policy"], "pattern/lowering/quality gate/workloads bound"),
            (2, "replay E06-03 correctness", ["hqsb.integration.differential.CorrectnessMatrix", "hqsb.integration.differential.compare_arrays", "hqsb.integration.differential.default_tolerance_registry"], "no performance number before the gate"),
            (3, "eager unfused baseline", ["hqsb.integration.lowering.intermediate_bytes", "hqsb.integration.lowering.AllocationAccount", "hqsb.integration.guards.CompileLedger"], "kernels/launches/allocations outside the graph too"),
            (4, "compile inductor baseline", ["hqsb.integration.lowering.LoweringRegistry.select", "hqsb.integration.lowering.LoweringTarget.artifact_kind", "hqsb.integration.graph.CaptureMode.INDUCTOR"], "do not re-credit upstream fusion"),
            (5, "eager explicit fused", ["hqsb.integration.differential.fused_add_rms_norm_reference", "hqsb.integration.lowering.AllocationAccount.launch_delta"], "upper bound without rewrite"),
            (6, "pattern + reference lowering", ["hqsb.integration.patterns.apply_rewrite", "hqsb.integration.lowering.LoweringTarget.reference", "hqsb.integration.lowering.LoweringDecision.as_dict"], "validates the rewrite alone"),
            (7, "pattern + inductor lowering", ["hqsb.integration.lowering.LoweringTarget.kernel_symbol", "hqsb.integration.lowering.LoweringDecision.generated_artifact", "hqsb.integration.guards.EventKind.LOWERING_SELECT"], "generated code/IR saved"),
            (8, "pattern + HQSB lowering", ["hqsb.integration.lowering.LoweringRegistry.select", "hqsb.integration.dispatch.KernelObservation", "hqsb.integration.telemetry.S06ResultFields.observed_kernel"], "dispatch/capability/artifact/kernel chain"),
            (9, "micro shape sweep", ["hqsb.integration.lowering.tensor_bytes", "hqsb.integration.lowering.fuse_saving_model", "hqsb.integration.lowering.LoweringRequest"], "decode M=1, small M, prefill, tail"),
            (10, "block scaling", ["hqsb.integration.lowering.AllocationAccount", "hqsb.integration.lowering.AmdahlPrediction", "hqsb.integration.guards.CompileLedger"], "1/N/all block accumulation"),
            (11, "six-workload model test", ["hqsb.integration.experiment.EvidenceManifest", "hqsb.integration.guards.StormThresholds", "hqsb.integration.lowering.LoweringRequest"], "3 processes and CI, ordinary timing"),
            (12, "representative profile", ["hqsb.integration.dispatch.KernelObservation", "hqsb.integration.telemetry.TraceCollector.emit", "hqsb.integration.experiment.RUN_LAYOUT"], "CPU/GPU timeline per phase"),
            (13, "intermediate and allocation accounting", ["hqsb.integration.lowering.AllocationAccount.reconcile", "hqsb.integration.lowering.fuse_saving_model", "hqsb.integration.lowering.tensor_bytes"], "theoretical bytes closed against allocator data"),
            (14, "compile cost and code size", ["hqsb.integration.cache.compute_break_even", "hqsb.integration.cache.CacheEntry.size_bytes", "hqsb.integration.cache.PhaseTimer"], "connected to the E06-06 cache"),
            (15, "dynamic policy impact", ["hqsb.integration.guards.resolve_compiler_config", "hqsb.integration.guards.DynamicPolicy", "hqsb.integration.lowering.LoweringCapability.supports_dynamic"], "static/dynamic/bucket compared"),
            (16, "capability reject", ["hqsb.integration.lowering.RejectedTarget", "hqsb.integration.lowering.LoweringRegistry.select", "hqsb.integration.dispatch.FallbackPolicy"], "unsupported shapes fall back safely"),
            (17, "quantized path truthfulness", ["hqsb.integration.patterns.QuantDescriptor", "hqsb.integration.lowering.LoweringCapability.requires_group_size", "hqsb.integration.differential.compare_arrays"], "packed hash + observed low-bit kernel"),
            (18, "ablation", ["hqsb.integration.lowering.ablation_matrix", "hqsb.integration.lowering.AblationCase", "hqsb.integration.lowering.AttributionReport"], "pattern/lowering/kernel/epilogue switched off"),
            (19, "Amdahl / bytes attribution", ["hqsb.integration.lowering.AmdahlPrediction", "hqsb.integration.lowering.AttributionReport", "hqsb.integration.lowering.AllocationAccount.reconcile"], "prediction-vs-actual residual explained"),
            (20, "verdict", ["hqsb.integration.lowering.LoweringDecision.as_dict", "hqsb.integration.differential.CorrectnessMatrix.complete", "hqsb.integration.telemetry.project_c6"], "correctness/safety/execution/benefit judged separately"),
        ]
    ),
)

E06_08 = ExperimentMapping(
    experiment_id="E06-08",
    title="CUDA Graph capture and replay (P1)",
    driver="scripts/integration/run_e06.py --experiment E06-08",
    fixture="hqsb.integration.cuda_graph / policies.load_graph_spec",
    steps=_steps(
        [
            (1, "freeze GraphSpec", ["hqsb.integration.policies.load_graph_spec", "hqsb.integration.cuda_graph.GraphSpec", "hqsb.integration.cuda_graph.BucketSpec"], "scope/stream/buffers/output contract/pool policy"),
            (2, "confirm E06-07 path passes", ["hqsb.integration.differential.CorrectnessMatrix", "hqsb.integration.lowering.LoweringDecision"], "no capture of an incorrect callable"),
            (3, "non-graph baseline", ["hqsb.integration.cuda_graph.ReplayTiming", "hqsb.integration.guards.CompileLedger", "hqsb.integration.cache.PhaseTimer"], "eager/compile steady with timelines"),
            (4, "pre-allocate static buffers", ["hqsb.integration.cuda_graph.StaticBuffer", "hqsb.integration.cuda_graph.StaticBufferPool.allocate", "hqsb.integration.cuda_graph.StaticBufferPool.addresses"], "addresses, ownership and lifetime recorded"),
            (5, "side-stream warmup", ["hqsb.integration.cuda_graph.EligibilityObservation.side_stream_warmup", "hqsb.integration.cuda_graph.evaluate_eligibility"], "lazy init never enters the capture"),
            (6, "capture and instantiate", ["hqsb.integration.cuda_graph.CaptureRecord", "hqsb.integration.cuda_graph.PoolMemoryAccount", "hqsb.integration.cuda_graph.evaluate_eligibility"], "DAG/nodes/timing/pool/error saved"),
            (7, "multi-input replay correctness", ["hqsb.integration.cuda_graph.ReplayRecord", "hqsb.integration.cuda_graph.replay_distinctness", "hqsb.integration.differential.compare_arrays"], "catches replaying stale inputs"),
            (8, "long replay", ["hqsb.integration.cuda_graph.ReplayTiming", "hqsb.integration.lifecycle.robust_slope", "hqsb.integration.cuda_graph.ReplayRecord"], "decode-heavy repeat with resources"),
            (9, "output retention/overwrite", ["hqsb.integration.cuda_graph.OutputHandle", "hqsb.integration.cuda_graph.OutputContract", "hqsb.integration.cuda_graph.StaticBufferPool.mark_in_flight"], "borrowed vs owned is a public contract"),
            (10, "non-default/multiple streams", ["hqsb.integration.cuda_graph.EligibilityObservation.multi_stream_fork_rejoin", "hqsb.integration.lifecycle.audit_streams", "hqsb.integration.lifecycle.StreamEvent"], "fork/rejoin, no implicit global sync"),
            (11, "shape buckets", ["hqsb.integration.cuda_graph.GraphSpec.resolve", "hqsb.integration.cuda_graph.BucketPlan", "hqsb.integration.guards.DynamicDimensionSpec.bucket_for"], "out-of-bucket falls back, padding measured"),
            (12, "failure matrix", ["hqsb.integration.cuda_graph.failure_matrix", "hqsb.integration.cuda_graph.evaluate_failure_matrix", "hqsb.integration.cuda_graph.FailureObservation"], "no half capture, no old graph reuse"),
            (13, "pool memory", ["hqsb.integration.cuda_graph.PoolMemoryAccount", "hqsb.integration.cuda_graph.StaticBufferPool.total_bytes"], "create/capture/replay/destroy/trim curve"),
            (14, "multi-graph / pool sharing", ["hqsb.integration.cuda_graph.GraphSpec.max_graphs", "hqsb.integration.lifecycle.LifecycleMachine", "hqsb.integration.cuda_graph.PoolMemoryAccount"], "aliasing checked before sharing"),
            (15, "CPU/GPU timeline", ["hqsb.integration.cuda_graph.ReplayTiming.as_dict", "hqsb.integration.guards.CompileLedger", "hqsb.integration.lifecycle.audit_streams"], "benefit attributed to launch gaps"),
            (16, "end-to-end six-workload subset", ["hqsb.integration.guards.ordered_shape_trace", "hqsb.integration.cuda_graph.OutOfBucketPolicy", "hqsb.integration.experiment.RUN_LAYOUT"], "inapplicable cases keep their fallback reason"),
            (17, "capture/recapture amortisation", ["hqsb.integration.cuda_graph.ReplayTiming.break_even_replays", "hqsb.integration.cuda_graph.GraphSpec.allow_recapture"], "bucket hit rate and request distribution"),
            (18, "destroy and in-flight", ["hqsb.integration.cuda_graph.StaticBufferPool.release", "hqsb.integration.lifecycle.TeardownPlan.validate", "hqsb.integration.lifecycle.PostCloseExpectation"], "no use-after-free"),
            (19, "new-process re-verification", ["hqsb.integration.experiment.environment_fingerprint", "hqsb.integration.cuda_graph.claim_status", "hqsb.integration.cache.CacheState"], "graph objects are not portable artifacts"),
            (20, "claim verdict", ["hqsb.integration.cuda_graph.claim_status", "hqsb.integration.cuda_graph.ClaimEvidence", "hqsb.integration.cuda_graph.ClaimStatus"], "P1 unrun ⇒ NOT_CLAIMED"),
        ]
    ),
)

E06_09 = ExperimentMapping(
    experiment_id="E06-09",
    title="input/compile/ABI faults, error mapping and deterministic fallback",
    driver="scripts/integration/run_e06.py --experiment E06-09",
    fixture="hqsb.integration.taxonomy / abi",
    steps=_steps(
        [
            (1, "freeze error taxonomy", ["hqsb.integration.taxonomy.frozen_taxonomy", "hqsb.integration.taxonomy.REASON_CODES", "hqsb.integration.taxonomy.ErrorTaxonomy.uncovered_stages"], "stable code/stage/severity/retryable/fallback/message"),
            (2, "freeze state invariants", ["hqsb.integration.taxonomy.kv_invariants", "hqsb.integration.taxonomy.StateInvariant", "hqsb.integration.lifecycle.ResourceSpec"], "what must not change on failure"),
            (3, "golden successful request", ["hqsb.integration.experiment.Preregistration", "hqsb.integration.experiment.check_prerequisites", "hqsb.integration.differential.DifferentialSpec"], "fixtures derive from a replayable request"),
            (4, "inject device/dtype/shape/layout", ["hqsb.integration.taxonomy.FailureRecord", "hqsb.integration.meta.MetadataError", "hqsb.integration.dispatch.CapabilityRequest"], "validation stage and zero launches"),
            (5, "inject model/quant mismatch", ["hqsb.integration.taxonomy.ErrorStage.MODEL_QUANT_IDENTITY", "hqsb.integration.patterns.QuantDescriptor.artifact_hash", "hqsb.integration.adapter.identity_collision_check"], "caught before graph/lowering binding"),
            (6, "inject fake/capture/pattern faults", ["hqsb.integration.meta.MetadataError", "hqsb.integration.patterns.evaluate_candidate", "hqsb.integration.graph.Graph.copy"], "no partial replacement, reference still usable"),
            (7, "inject compile/codegen/link faults", ["hqsb.integration.cache.CorruptionFixture", "hqsb.integration.abi.preload_check", "hqsb.integration.taxonomy.FailureRecord"], "isolated cache/artifact, cleanup checked"),
            (8, "inject cache corruption", ["hqsb.integration.cache.validate_entry", "hqsb.integration.cache.CorruptionFixture.bit_flip", "hqsb.integration.cache.IntegrityFailure"], "bad entry never executed"),
            (9, "inject ABI matrix", ["hqsb.integration.abi.CompatibilityMatrix.check", "hqsb.integration.abi.simulate_identity", "hqsb.integration.abi.BinaryManifest.symbol_audit"], "manifest rejects before load"),
            (10, "inject workspace/OOM", ["hqsb.integration.taxonomy.ReasonCodeSpec", "hqsb.integration.lowering.LoweringCapability.max_workspace_bytes"], "predictable vs runtime OOM separated"),
            (11, "inject sync/async kernel errors", ["hqsb.integration.taxonomy.ErrorStage.ASYNC_DEVICE", "hqsb.integration.taxonomy.FailureRecord.sync_async", "hqsb.integration.taxonomy.deterministic_failure"], "isolated process, attribution boundary"),
            (12, "verify mutable state abort", ["hqsb.integration.taxonomy.TransactionalPlan.abort", "hqsb.integration.taxonomy.TransactionalPlan.validate_completion", "hqsb.integration.taxonomy.kv_invariants"], "no half token / partial residual"),
            (13, "fallback priority", ["hqsb.integration.taxonomy.FallbackRegistry.disable", "hqsb.integration.taxonomy.FallbackRegistry.resolve", "hqsb.integration.dispatch.FallbackPolicy"], "per-level disable with a reason"),
            (14, "strict mode", ["hqsb.integration.taxonomy.FallbackRegistry.resolve", "hqsb.integration.dispatch.FallbackPolicy.strict", "hqsb.integration.dispatch.RouteDecision"], "must fail, never fall back silently"),
            (15, "repeat the same fault", ["hqsb.integration.taxonomy.deterministic_failure", "hqsb.integration.experiment.RunDirectory.record_command", "hqsb.integration.cache.validate_entry"], "identical decision fields across replays"),
            (16, "recover a legal request", ["hqsb.integration.taxonomy.TransactionalPlan.commit", "hqsb.integration.lifecycle.LifecycleMachine.transition", "hqsb.integration.differential.ModeSwitchAudit"], "no dirty graph/cache/allocator state"),
            (17, "concurrent request isolation", ["hqsb.integration.lifecycle.audit_streams", "hqsb.integration.taxonomy.FailureRecord.partial_state", "hqsb.integration.cuda_graph.OutputHandle"], "or an explicit serial boundary"),
            (18, "verify the C6/C7 chain", ["hqsb.integration.taxonomy.requested_actual_chain", "hqsb.integration.telemetry.project_c6", "hqsb.integration.telemetry.to_trace_events"], "requested→failure→fallback→actual→result"),
            (19, "user error message audit", ["hqsb.integration.taxonomy.sanitize_message", "hqsb.integration.taxonomy.ReasonCodeSpec.user_message", "hqsb.integration.taxonomy.FailureRecord.sanitized_message"], "actionable, no internal leakage"),
            (20, "expected/actual matrix", ["hqsb.integration.taxonomy.FailureRecord.as_dict", "hqsb.integration.taxonomy.frozen_taxonomy", "hqsb.integration.telemetry.dump_json"], "all cases kept, including unexpected successes"),
        ]
    ),
)

E06_10 = ExperimentMapping(
    experiment_id="E06-10",
    title="long-run enable/disable, shape/stream alternation and lifecycle",
    driver="scripts/integration/run_e06.py --experiment E06-10",
    fixture="hqsb.integration.lifecycle / policies.load_resource_spec",
    steps=_steps(
        [
            (1, "freeze ResourceSpec", ["hqsb.integration.policies.load_resource_spec", "hqsb.integration.lifecycle.frozen_resource_specs", "hqsb.integration.lifecycle.ResourceSpec"], "owner/create/destroy/cap/threshold"),
            (2, "baseline empty process", ["hqsb.integration.experiment.environment_fingerprint", "hqsb.integration.lifecycle.ResourceSnapshot", "hqsb.integration.lifecycle.LivenessProbe"], "RSS/FD/threads/allocator before import"),
            (3, "per-state snapshots", ["hqsb.integration.lifecycle.LifecycleMachine.transition", "hqsb.integration.lifecycle.ResourceSnapshot.as_dict", "hqsb.integration.lifecycle.frozen_resource_specs"], "REGISTERED→…→ACTIVE deltas"),
            (4, "single enable/disable/close", ["hqsb.integration.lifecycle.EnableDisableAudit", "hqsb.integration.lifecycle.audit_enable_disable", "hqsb.integration.lifecycle.PostCloseExpectation"], "state machine and correctness first"),
            (5, "fixed long-run trace", ["hqsb.integration.lifecycle.segment_snapshots", "hqsb.integration.lifecycle.ResourceSnapshot", "hqsb.integration.experiment.RunDirectory.write_json"], "mode/shape/stream/resources/output per cycle"),
            (6, "reach shape/graph cache cap", ["hqsb.integration.cache.CacheSizeTracker.evaluate", "hqsb.integration.cache.evict", "hqsb.integration.lifecycle.evaluate_leak"], "bounded growth, reuse and eviction"),
            (7, "alternate default/non-default streams", ["hqsb.integration.lifecycle.audit_streams", "hqsb.integration.lifecycle.StreamEvent", "hqsb.integration.cuda_graph.EligibilityObservation"], "events, dependencies, global sync"),
            (8, "alternate supported/fallback", ["hqsb.integration.taxonomy.FallbackRegistry.resolve", "hqsb.integration.lifecycle.ResourceSnapshot", "hqsb.integration.lifecycle.evaluate_leak"], "failure paths reclaim as well as success"),
            (9, "periodic compile/load faults", ["hqsb.integration.cache.CorruptionFixture", "hqsb.integration.abi.preload_check", "hqsb.integration.lifecycle.LivenessProbe"], "temp files, caches, objects"),
            (10, "multiple model instances", ["hqsb.integration.adapter.identity_collision_check", "hqsb.integration.cache.graph_identity", "hqsb.integration.lifecycle.ResourceSnapshot"], "read-only artifacts shared, state separate"),
            (11, "optional CUDA Graph loop", ["hqsb.integration.cuda_graph.StaticBufferPool.release", "hqsb.integration.cuda_graph.PoolMemoryAccount", "hqsb.integration.lifecycle.LifecycleMachine"], "create/replay/destroy/pool"),
            (12, "monitor python liveness", ["hqsb.integration.lifecycle.LivenessProbe", "hqsb.integration.lifecycle.LivenessProbe.collect", "hqsb.integration.lifecycle.liveness_factory"], "weakrefs, hooks, GC"),
            (13, "monitor native resources", ["hqsb.integration.lifecycle.robust_slope", "hqsb.integration.lifecycle.ResourceSnapshot"], "RSS/pinned/FD/threads/temp"),
            (14, "monitor device resources", ["hqsb.integration.lifecycle.frozen_resource_specs", "hqsb.integration.lifecycle.ResourceSnapshot"], "allocated/reserved/workspace/events/KV"),
            (15, "collect sync timeline", ["hqsb.integration.lifecycle.audit_streams", "hqsb.integration.guards.CompileLedger.record", "hqsb.integration.guards.EventKind.KERNEL"], "long-run and short profile kept apart"),
            (16, "execute teardown", ["hqsb.integration.lifecycle.TeardownPlan.validate", "hqsb.integration.lifecycle.TeardownPlan.as_dict"], "in-flight wait before release"),
            (17, "verify post-close", ["hqsb.integration.lifecycle.PostCloseExpectation", "hqsb.integration.lifecycle.PostCloseExpectation.ok"], "contract failure, no use-after-free"),
            (18, "rebuild the second round", ["hqsb.integration.lifecycle.LifecycleMachine.path", "hqsb.integration.lifecycle.segment_snapshots", "hqsb.integration.lifecycle.ResourceSnapshot"], "no accumulation across rounds"),
            (19, "slope / change point statistics", ["hqsb.integration.lifecycle.robust_slope", "hqsb.integration.lifecycle.detect_plateau", "hqsb.integration.lifecycle.evaluate_leak"], "warmup/steady/teardown fitted separately"),
            (20, "fault localisation", ["hqsb.integration.lifecycle.evaluate_leak", "hqsb.integration.lifecycle.LeakReport.as_dict", "hqsb.integration.telemetry.TraceCollector"], "owner/span linked, not a screenshot"),
        ]
    ),
)

E06_11 = ExperimentMapping(
    experiment_id="E06-11",
    title="second model / dummy backend reuse and anti-hard-coding",
    driver="scripts/integration/run_e06.py --experiment E06-11",
    fixture="hqsb.integration.adapter / policies.load_reuse_spec",
    steps=_steps(
        [
            (1, "freeze ReuseSpec", ["hqsb.integration.policies.load_reuse_spec", "hqsb.integration.adapter.AdapterRegistration", "hqsb.integration.policies.ReuseSpecDocument"], "route, differences, adapter boundary, conditions"),
            (2, "snapshot core interfaces", ["hqsb.integration.adapter.core_contract_snapshot", "hqsb.integration.adapter.CoreContractSnapshot"], "public contract, registries, patterns, rules"),
            (3, "static hard-code scan", ["hqsb.integration.adapter.hardcode_scan", "hqsb.integration.adapter.hardcode_report", "hqsb.integration.adapter.HARDCODE_RULES"], "hit classification, not automatic bug"),
            (4, "second model module/graph census", ["hqsb.integration.adapter.model_census", "hqsb.integration.adapter.ModuleCensusEntry", "hqsb.integration.graph.from_fx_graph_module"], "runtime census, never hand-written"),
            (5, "implement model adapter", ["hqsb.integration.adapter.ModelAdapter", "hqsb.integration.adapter.AdapterRegistration"], "discovery/mapping only; no core pass copied"),
            (6, "dummy backend", ["hqsb.integration.adapter.DummyBackendAdapter", "hqsb.integration.adapter.BackendAdapter", "hqsb.integration.adapter.DummyBackendAdapter.performance_claim_allowed"], "C4 capability/execute/error/trace"),
            (7, "register operator and fake", ["hqsb.integration.specs.frozen_schemas", "hqsb.integration.meta.frozen_contracts", "hqsb.integration.dispatch.frozen_p0_matrix"], "reuse E06-01/02; no model-specific schema"),
            (8, "pattern dry scan", ["hqsb.integration.patterns.scan_graph", "hqsb.integration.patterns.Decision", "hqsb.integration.patterns.find_matches"], "candidates and rejects before any change"),
            (9, "operator/block correctness", ["hqsb.integration.differential.compare_arrays", "hqsb.integration.differential.CorrectnessMatrix", "hqsb.integration.differential.default_tolerance_registry"], "reuse the E06-03 oracle"),
            (10, "model/generation", ["hqsb.integration.differential.Level.MODEL_LOGITS", "hqsb.integration.differential.Level.GENERATION", "hqsb.integration.experiment.EvidenceManifest"], "fixed artifact and inputs"),
            (11, "compile/dynamic/cache", ["hqsb.integration.guards.resolve_compiler_config", "hqsb.integration.cache.CompileIdentity", "hqsb.integration.cache.validate_entry"], "minimal E06-05/06 matrix"),
            (12, "fallback/error", ["hqsb.integration.taxonomy.FallbackRegistry.resolve", "hqsb.integration.taxonomy.frozen_taxonomy", "hqsb.integration.dispatch.FallbackPolicy"], "differences map onto standard reasons"),
            (13, "dummy spy verification", ["hqsb.integration.adapter.SpyEvent", "hqsb.integration.adapter.DummyBackendAdapter.as_dict", "hqsb.integration.telemetry.TraceCollector"], "call order, capability, actual route, C6/C7"),
            (14, "Qwen regression", ["hqsb.integration.differential.DifferentialSpec", "hqsb.integration.patterns.coverage_report", "hqsb.integration.experiment.check_prerequisites"], "S06 P0 cases must not regress"),
            (15, "compare pattern coverage", ["hqsb.integration.patterns.CoverageReport", "hqsb.integration.patterns.RuntimeUse", "hqsb.integration.patterns.coverage_report"], "eligible/hit/reject/time kept apart"),
            (16, "compare code changes", ["hqsb.integration.adapter.ChangeBudget", "hqsb.integration.adapter.summarize_changes", "hqsb.integration.adapter.code_change_from_git"], "core/adapter/test diff classified"),
            (17, "lifecycle", ["hqsb.integration.lifecycle.LifecycleMachine", "hqsb.integration.adapter.identity_collision_check", "hqsb.integration.cache.graph_identity"], "registry/cache keys must not collide"),
            (18, "same-name/same-shape difference injection", ["hqsb.integration.adapter.identity_collision_check", "hqsb.integration.cache.CacheEntry.as_dict", "hqsb.integration.cache.validate_entry"], "no compiled-graph reuse across targets"),
            (19, "independent config registration", ["hqsb.integration.policies.load_reuse_spec", "hqsb.integration.adapter.AdapterRegistration.changes_global_default", "hqsb.integration.policies.load_directory"], "no global default pollution"),
            (20, "reuse audit", ["hqsb.integration.adapter.ReuseMatrix", "hqsb.integration.adapter.reuse_matrix_template", "hqsb.integration.adapter.summarize_changes"], "reused/extended/refused stated per item"),
        ]
    ),
)


EXPERIMENT_MAP: Tuple[ExperimentMapping, ...] = (
    E06_01,
    E06_02,
    E06_03,
    E06_04,
    E06_05,
    E06_06,
    E06_07,
    E06_08,
    E06_09,
    E06_10,
    E06_11,
)


def mapping_for(experiment_id: str) -> ExperimentMapping:
    for mapping in EXPERIMENT_MAP:
        if mapping.experiment_id == experiment_id:
            return mapping
    raise KeyError(experiment_id)


def resolve_interface(symbol: str) -> Any:
    """Import a dotted symbol, supporting nested attributes.

    ``hqsb.integration.specs.OpSchema.schema_hash`` is resolved by importing the
    longest importable module prefix and then walking the remaining attributes,
    so the map can point at a class method or an enum member (which is often the
    precise interface a step needs) without inventing module-level aliases.
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
        try:
            for attribute in parts[index:]:
                obj = getattr(obj, attribute)
        except AttributeError as exc:
            raise AttributeError(
                f"{symbol!r}: {type(exc).__name__}: {exc}"
            ) from exc
        return obj
    raise AttributeError(
        f"{symbol!r}: no importable module prefix ({last_error})"
    )


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
        "stage": "S06",
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
        "# S06 实验步骤 → 代码接口对照表",
        "",
        "> 由 `hqsb/integration/interface_map.py` 生成；"
        "`scripts/integration/run_e06.py --mode interface-map` 可重新生成。",
        "> 成熟度：M2 = 源码 + 本树自动化测试覆盖；M3+ 需真实硬件/模型证据（当前 BLOCKED）。",
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
    "E06_01",
    "E06_02",
    "E06_03",
    "E06_04",
    "E06_05",
    "E06_06",
    "E06_07",
    "E06_08",
    "E06_09",
    "E06_10",
    "E06_11",
    "EXPERIMENT_MAP",
    "ExperimentMapping",
    "StepMapping",
    "mapping_for",
    "mapping_table_markdown",
    "resolve_interface",
    "resolve_interfaces",
]
