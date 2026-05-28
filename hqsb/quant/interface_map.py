"""Experiment step → code interface map (the "no missing capability" proof).

Each S05 experiment document lists concrete steps ("具体实验步骤"). This module
records, for **every** step, which HQSB interface implements it, and
:func:`resolve_interfaces` verifies that each referenced symbol actually
imports — so the map cannot rot into documentation. A test
(``tests/unit/quant/test_interface_map.py``) runs the verification.

Maturity labels follow the control plane (M0–M7):

* ``M1`` — source exists (interface implemented, structured errors);
* ``M2`` — covered by automated tests in this tree;
* ``M3``+ — requires real hardware/model evidence, which this repository
  state cannot produce (the S04.5 M4 prerequisite is unmet), so no entry is
  labelled above M2.

Drivers live in ``scripts/quant/run_e05.py`` and are *not* executed by this
map.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


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


E05_01 = ExperimentMapping(
    experiment_id="E05-01",
    title="RTN quant math, packing and QuantArtifact",
    driver="scripts/quant/run_e05.py --experiment E05-01",
    fixture="hqsb.quant.fixtures.build_tiny_artifact / save_golden_artifacts",
    steps=_steps(
        [
            (1, "freeze the spec", ["hqsb.quant.spec.QuantScheme", "hqsb.quant.spec.integer_range", "hqsb.quant.spec.groups_per_row", "hqsb.quant.spec.tail_length"], "range/round/zero-group/constant-group/axis/dtype/pack/NaN policies are single-valued"),
            (2, "hand-computed golden", ["hqsb.quant.golden.quantize_exact", "hqsb.quant.golden.build_golden_vectors"], "exact-rational oracle independent of the float reference"),
            (3, "FP64/FP32 reference", ["hqsb.quant.rtn.quantize", "hqsb.quant.rtn.dequantize", "hqsb.quant.golden.cross_check_with_reference"], "two independent implementations; cross-check reported"),
            (4, "factorial matrix", ["hqsb.quant.spec.factorial_matrix", "hqsb.quant.spec.FactorialCase"], "illegal cells are pre-registered with expected rejects"),
            (5, "quantization correctness", ["hqsb.quant.rtn.reconstruction_report", "hqsb.quant.rtn.clamp_free_error_bound", "hqsb.quant.rtn.requantize_idempotence_check"], "clamp count, MSE/max/cosine/SQNR/first mismatch, input untouched"),
            (6, "pack/unpack", ["hqsb.quant.packing.pack_canonical", "hqsb.quant.packing.unpack_canonical", "hqsb.quant.packing.unpack_independent", "hqsb.quant.packing.validate_codes"], "all codes, odd count, tail, padding, sign extension"),
            (7, "save/load round-trip in a new process", ["hqsb.quant.artifact.QuantArtifactDocument.save", "hqsb.quant.artifact.load_document", "hqsb.quant.artifact.validate_artifact_dir"], "identity hash is stable and timestamp-independent"),
            (8, "cross-implementation", ["hqsb.quant.packing.unpack_independent", "hqsb.quant.artifact.verify_variant_against_canonical", "hqsb.quant.golden.cross_check_with_reference"], "packer and unpacker never share one index implementation"),
            (9, "illegal artifacts", ["hqsb.quant.faults.build_fault_matrix", "hqsb.quant.faults.run_fault_matrix"], "schema/model/quant/pack/capability mutations are refused pre-launch"),
            (10, "offline cost only", ["hqsb.quant.calibration.OfflineCost", "hqsb.quant.model_eval.PhaseTimer"], "offline quantize/pack/save/load timing is kept out of inference numbers"),
            (11, "size accounting", ["hqsb.quant.artifact.SizeBreakdown", "hqsb.quant.artifact.QuantArtifactDocument.size_breakdown", "hqsb.quant.artifact.SizeBreakdown.reconcile", "hqsb.quant.model_eval.measure_artifact_memory"], "predicted vs measured with an itemized residual"),
            (12, "golden vector package", ["hqsb.quant.golden.golden_package_json", "hqsb.quant.golden.golden_package_hash"], "reusable by E05-04/06/09"),
        ]
    ),
)

E05_02 = ExperimentMapping(
    experiment_id="E05-02",
    title="weight-only model quality/memory/performance baseline",
    driver="scripts/quant/run_e05.py --experiment E05-02",
    fixture="hqsb.quant.apply.swap_weights with a tiny torch model (unit test)",
    steps=_steps(
        [
            (1, "freeze ExperimentSpec", ["hqsb.quant.experiment.Preregistration", "hqsb.quant.experiment.environment_fingerprint", "hqsb.quant.experiment.git_state"], "spec hash freezes the protocol before results"),
            (2, "verify S04.5 M4 and FP16 reference", ["hqsb.quant.experiment.check_prerequisites", "hqsb.quant.experiment.PrerequisiteStatus"], "missing M4 evidence is reported, not assumed"),
            (3, "enumerate quantization coverage", ["hqsb.quant.coverage.CoveragePolicy", "hqsb.quant.coverage.enumerate_weight_rows", "hqsb.quant.coverage.coverage_summary", "hqsb.quant.apply.plan_model_quantization"], "per-tensor rows with exclusions and reasons"),
            (4, "offline W8/W4 generation", ["hqsb.quant.rtn.quantize", "hqsb.quant.artifact.QuantArtifactDocument.from_quantized", "hqsb.quant.artifact.CalibrationProvenance"], "scale/zero/saturation/reconstruction per tensor"),
            (5, "pack-save-load in a new process", ["hqsb.quant.packing.pack_kernel_variant", "hqsb.quant.artifact.load_document"], "artifact, model and packed hashes are cross-checked"),
            (6, "fake-dequant control", ["hqsb.quant.apply.fake_quant_weight", "hqsb.quant.apply.swap_weights", "hqsb.quant.execution.classify_execution"], "labelled fake_quant, excluded from low-bit performance claims"),
            (7, "operator and block alignment", ["hqsb.quant.apply.attach_capture_hooks", "hqsb.quant.apply.detach_capture_hooks", "hqsb.quant.quality.teacher_forcing_report"], "first over-budget layer and its input sample are retained"),
            (8, "teacher-forcing logit evaluation", ["hqsb.quant.quality.position_metrics", "hqsb.quant.quality.teacher_forcing_report"], "per-position KL/top-k/margin, not only final text"),
            (9, "PPL, tasks and robustness slices", ["hqsb.quant.model_eval.perplexity_from_logits", "hqsb.quant.model_eval.perplexity_torch", "hqsb.quant.quality.evaluate_quality_gate"], "fixed tokenizer/mask/stride/denominator; per-slice reporting"),
            (10, "cache-enabled autoregressive generation", ["hqsb.quant.quality.GenerationStep", "hqsb.quant.quality.generation_report"], "per-step KL/margin/token and first divergence"),
            (11, "artifact and load memory", ["hqsb.quant.model_eval.MemoryLadder", "hqsb.quant.model_eval.device_memory_ladder", "hqsb.quant.model_eval.measure_artifact_memory"], "split into artifact/canonical/packed/host/device/peak"),
            (12, "six-workload phase performance", ["hqsb.quant.model_eval.PhaseTimer", "hqsb.quant.model_eval.PHASES", "hqsb.quant.execution.ExecutionRecord"], "offline/cold-load/first-compile/steady kept apart"),
            (13, "profiler evidence", ["hqsb.quant.execution.ExecutionRecord.profiler_trace_hash", "hqsb.quant.execution.claim_audit"], "kernel symbol, launch count and fallback verified"),
            (14, "order and cache ablation", ["hqsb.quant.experiment.RunDirectory.record_command", "hqsb.quant.model_eval.PhaseTimer"], "run order/cold-warm recorded per repetition"),
            (15, "unified result table", ["hqsb.quant.model_eval.build_unified_table", "hqsb.quant.model_eval.TABLE_FIELDS"], "figures are generated from the table, never hand-copied"),
            (16, "adjudicate by gates", ["hqsb.quant.model_eval.gate_order", "hqsb.quant.execution.claim_audit"], "quality first, then execution reality, then performance/memory"),
        ]
    ),
)

E05_03 = ExperimentMapping(
    experiment_id="E05-03",
    title="calibration representativeness, sample efficiency and generalization",
    driver="scripts/quant/run_e05.py --experiment E05-03",
    fixture="hqsb.quant.calibration.SampleRecord/SplitManifest (synthetic pool in unit tests)",
    steps=_steps(
        [
            (1, "freeze DataSpec", ["hqsb.quant.calibration.DataSpec", "hqsb.quant.calibration.DataSpec.data_spec_hash"], "sources/revisions/splits/dedup/tokenizer/template/buckets/seeds/thresholds"),
            (2, "build candidate pool", ["hqsb.quant.calibration.SampleRecord", "hqsb.quant.calibration.SplitManifest", "hqsb.quant.calibration.token_ids_hash"], "sample manifest without copying private text"),
            (3, "leakage/near-duplicate audit", ["hqsb.quant.calibration.audit_leakage", "hqsb.quant.calibration.minhash_similarity"], "id/text-hash/n-gram/parent/template/benchmark checks with counts"),
            (4, "deployment distribution profile", ["hqsb.quant.calibration.SplitManifest.summary", "hqsb.quant.stats.summarize_distribution"], "language/domain/length/template differences made explicit"),
            (5, "factor sampling", ["hqsb.quant.calibration.draw_subset", "hqsb.quant.calibration.bucket_of", "hqsb.quant.calibration.sample_pool"], "pre-registered sample ids + token hashes; subset swap is detectable"),
            (6, "replay RTN control", ["hqsb.quant.rtn.quantize", "hqsb.quant.artifact.QuantArtifactDocument.identity_hash"], "RTN must not vary with the calibration subset"),
            (7, "collect method statistics", ["hqsb.quant.calibration.collect_activation_stats", "hqsb.quant.calibration.merge_module_stats"], "mask/padding/effective count verified; padding never enters stats"),
            (8, "generate independent artifacts", ["hqsb.quant.artifact.QuantArtifactDocument.save", "hqsb.quant.calibration.OfflineCost"], "per-subset artifact + offline time/host/device peak"),
            (9, "statistics and artifact replay", ["hqsb.quant.artifact.load_document", "hqsb.quant.artifact.validate_artifact_dir"], "repeatable quantization or an explicit randomness source"),
            (10, "local quality", ["hqsb.quant.quality.position_metrics", "hqsb.quant.sensitivity.weight_error_metrics"], "operator/block tests locate the first harmed layer"),
            (11, "policy-validation", ["hqsb.quant.stats.paired_bootstrap_ci", "hqsb.quant.stats.cluster_bootstrap_ci"], "learning curves with seed variance and CI"),
            (12, "select minimum sufficient budget", ["hqsb.quant.calibration.select_minimum_sufficient", "hqsb.quant.calibration.CandidateResult"], "pre-registered saturation rule; no final-evaluation input"),
            (13, "freeze the strategy", ["hqsb.quant.policy.MixedPrecisionPolicy.policy_hash", "hqsb.quant.calibration.SelectionDecision"], "policy hash + immutable calibration ids"),
            (14, "one-shot final evaluation", ["hqsb.quant.quality.evaluate_quality_gate", "hqsb.quant.stats.non_inferiority_verdict"], "frozen candidates only; no re-selection"),
            (15, "OOD/stress", ["hqsb.quant.activation.drift_report", "hqsb.quant.quality.evaluate_quality_gate"], "failure boundaries for deployment alerts"),
            (16, "offline cost", ["hqsb.quant.calibration.OfflineCost", "hqsb.quant.calibration.total_offline_cost", "hqsb.quant.calibration.amortized_cost_per_request"], "per-phase time and peaks for E05-10"),
            (17, "sensitivity explanation", ["hqsb.quant.stats.stable_subset_agreement", "hqsb.quant.stats.jaccard", "hqsb.quant.sensitivity.layer_output_error"], "scale/Hessian/salient-channel stability vs quality"),
            (18, "audit report", ["hqsb.quant.calibration.SplitManifest.manifest_hash", "hqsb.quant.calibration.SelectionDecision.as_dict", "hqsb.quant.model_eval.build_unified_table"], "every curve derived from raw manifests and parquet tables"),
        ]
    ),
)

E05_04 = ExperimentMapping(
    experiment_id="E05-04",
    title="GPTQ/AWQ/SmoothQuant adapters with a common protocol",
    driver="scripts/quant/run_e05.py --experiment E05-04",
    fixture="hqsb.quant.adapters.SourceTensorRecord (synthetic source record)",
    steps=_steps(
        [
            (1, "freeze MethodMatrix", ["hqsb.quant.adapters.base.MethodMatrix", "hqsb.quant.adapters.base.LEVELS"], "methods, versions, tunables, scope, calibration budget, levels"),
            (2, "lock third-party dependencies", ["hqsb.quant.adapters.base.capability_probe", "hqsb.quant.adapters.methods.adapter_availability_report"], "wheel/commit/flags/licence recorded; unavailable → blocked with reason"),
            (3, "resolve all defaults", ["hqsb.quant.adapters.base.SourceMethodConfig", "hqsb.quant.adapters.base.SourceMethodConfig.config_hash"], "implicit defaults expanded into a hashed resolved config"),
            (4, "build the model mapping", ["hqsb.quant.coverage.enumerate_weight_rows", "hqsb.quant.adapters.base.SourceMethodConfig"], "quantized/skipped/unsupported/fused/shared listed"),
            (5, "unquantized transform equivalence", ["hqsb.quant.adapters.base.transform_equivalence_check", "hqsb.quant.activation.SmoothTransform"], "scale folding verified before any quantization"),
            (6, "collect statistics per E05-03", ["hqsb.quant.calibration.collect_activation_stats", "hqsb.quant.calibration.SplitManifest.manifest_hash"], "identical frozen calibration manifest for every method"),
            (7, "run offline quantization", ["hqsb.quant.adapters.methods.MethodAdapter.convert_record", "hqsb.quant.calibration.OfflineCost"], "stage times, memory, failures and search traces kept"),
            (8, "export canonical QuantArtifact", ["hqsb.quant.artifact.QuantArtifactDocument.from_quantized", "hqsb.quant.artifact.QuantArtifactDocument.canonical_hash"], "field mapping + qvalue/scale/zero conversion + provenance"),
            (9, "source↔canonical equivalence", ["hqsb.quant.adapters.methods.MethodAdapter.equivariance_check", "hqsb.quant.adapters.base.dequantized_reference"], "bit unpack difference fails before model evaluation"),
            (10, "new-process round-trip", ["hqsb.quant.artifact.load_document", "hqsb.quant.artifact.verify_variant_against_canonical"], "saved artifact reloaded and re-executed"),
            (11, "Level A quality", ["hqsb.quant.apply.fake_quant_weight", "hqsb.quant.quality.teacher_forcing_report"], "common fake-dequant path isolates the algorithm"),
            (12, "build common packed variants", ["hqsb.quant.adapters.base.PackedVariantBuilder", "hqsb.quant.packing.pack_kernel_variant"], "semantic incompatibility is reported, never silently repacked"),
            (13, "Level B common kernel", ["ops.quant.executors.get_executor", "hqsb.quant.execution.claim_audit"], "same kernel for all algorithms; observed symbol verified"),
            (14, "Level C native runtime", ["hqsb.quant.adapters.base.RuntimeCapabilityBinding", "hqsb.quant.execution.ExecutionRecord.fallback_reason"], "native rewrite/cache/kernel/fallback recorded"),
            (15, "six-workload profiling", ["ops.quant.microbench.benchmark_callable", "hqsb.quant.model_eval.PhaseTimer"], "representative prefill/decode shapes deeply profiled"),
            (16, "final evaluation", ["hqsb.quant.quality.evaluate_quality_gate"], "frozen strategies only; no alpha/group re-tuning"),
            (17, "Level D cost summary", ["hqsb.quant.adapters.base.AdapterResult", "hqsb.quant.calibration.total_offline_cost"], "calibration/search/quant/pack/load/compile/failure rates separated"),
            (18, "attribution ablations", ["hqsb.quant.sensitivity.interaction", "hqsb.quant.decision.release_bundle_manifest"], "same algorithm different kernel / different algorithm same kernel"),
            (19, "adapter audit table", ["hqsb.quant.adapters.base.audit_field_mapping", "hqsb.quant.adapters.methods.MethodAdapter.field_mappings"], "every source field mapped/preserved/derived/unsupported"),
        ]
    ),
)

E05_05 = ExperimentMapping(
    experiment_id="E05-05",
    title="layer sensitivity, outliers and mixed precision",
    driver="scripts/quant/run_e05.py --experiment E05-05",
    fixture="hqsb.quant.units.build_units (synthetic module list before a real model)",
    steps=_steps(
        [
            (1, "freeze intervention granularity", ["hqsb.quant.units.build_units", "hqsb.quant.units.InterventionUnit", "hqsb.quant.units.UnitConfig"], "units come from the module tree ∩ backend capability, fused/shared recorded"),
            (2, "freeze data and quality budget", ["hqsb.quant.calibration.DataSpec", "hqsb.quant.quality.QualityBudget"], "reuses E05-03 splits; search budget declared"),
            (3, "all-FP16/W8/W4 anchors", ["hqsb.quant.policy.baseline_policies"], "anchors recomputed, not inherited"),
            (4, "baseline activation/outlier statistics", ["hqsb.quant.calibration.collect_activation_stats", "hqsb.quant.stats.summarize_distribution"], "hook does not change outputs or timing paths"),
            (5, "single-unit artifacts", ["hqsb.quant.sensitivity.plan_single_quant", "hqsb.quant.units.units_from_coverage"], "one unit quantized, everything else FP16; shared weights handled atomically"),
            (6, "frozen-input operator test", ["hqsb.quant.sensitivity.layer_output_error", "hqsb.quant.apply.attach_capture_hooks"], "real stored activations, per-unit direct output error"),
            (7, "teacher-forcing model test", ["hqsb.quant.quality.teacher_forcing_report", "hqsb.quant.sensitivity.InterventionResult"], "block error/logit KL/PPL proxy per intervention"),
            (8, "leave-one-out artifacts", ["hqsb.quant.sensitivity.plan_leave_one_out"], "recovery and extra bytes recorded per unit"),
            (9, "bit/group/method ablation on sensitive units", ["hqsb.quant.units.UnitConfig", "hqsb.quant.policy.UnitPolicyEntry"], "distinguish bit vs group vs outlier vs algorithm"),
            (10, "cumulative paths", ["hqsb.quant.sensitivity.plan_cumulative", "hqsb.quant.sensitivity.ORDER_DIRECT_HARM", "hqsb.quant.sensitivity.ORDER_RECOVERY_PER_BYTE", "hqsb.quant.sensitivity.ORDER_RANDOM"], "layer/harm/recovery/random orders quantify path dependence"),
            (11, "pairwise interaction", ["hqsb.quant.sensitivity.plan_pairwise", "hqsb.quant.sensitivity.interaction_table"], "interaction formula plus random control pairs"),
            (12, "measured cost table", ["hqsb.quant.sensitivity.CostRow", "hqsb.quant.sensitivity.cost_table_json"], "artifact/device bytes, phase latency, kernel, fallback, switches"),
            (13, "policy search", ["hqsb.quant.policy.greedy_search", "hqsb.quant.policy.SearchTrace"], "every candidate with parent/prediction/measurement/reject reason"),
            (14, "candidate front", ["hqsb.quant.decision.point_pareto", "hqsb.quant.decision.uncertainty_pareto"], "several non-dominated candidates kept"),
            (15, "new-process rebuild", ["hqsb.quant.policy.MixedPrecisionPolicy.from_yaml", "hqsb.quant.policy.MixedPrecisionPolicy.validate"], "no dependency on an in-memory patch order"),
            (16, "six workloads", ["hqsb.quant.model_eval.PhaseTimer", "hqsb.quant.execution.ExecutionRecord"], "coverage/kernel/memory/TTFT/TPOT/TPS/energy per candidate"),
            (17, "freeze candidates", ["hqsb.quant.policy.MixedPrecisionPolicy.policy_hash", "hqsb.quant.decision.recommendation_matrix"], "candidate set and ranking rule locked before final evaluation"),
            (18, "final evaluation and long generation", ["hqsb.quant.quality.generation_report", "hqsb.quant.quality.evaluate_quality_gate"], "all frozen candidates reported, failures included"),
            (19, "causal attribution", ["hqsb.quant.policy.counterfactual_check"], "counterfactual bit change on retained layers"),
            (20, "machine-readable policy", ["hqsb.quant.policy.MixedPrecisionPolicy.to_yaml", "hqsb.quant.policy.MixedPrecisionPolicy.validate"], "module selector/bit/group/method/packed requirement/fallback/provenance"),
        ]
    ),
)

E05_06 = ExperimentMapping(
    experiment_id="E05-06",
    title="real W8/W4 GEMM, fused dequant and model-path hit",
    driver="scripts/quant/run_e05.py --experiment E05-06",
    fixture="ops.quant.w4a16_triton.compile_probe + hqsb.quant.fixtures",
    steps=_steps(
        [
            (1, "freeze KernelSpec", ["ops.quant.capability.probe_low_bit_capability", "ops.quant.executors.executor_matrix"], "provider/version/arch/scheme/layout/shape/workspace/fallback"),
            (2, "export the real linear census", ["hqsb.quant.coverage.enumerate_weight_rows", "hqsb.quant.model_eval.PhaseTimer"], "module→M/N/K/stride/frequency/time from the six workloads"),
            (3, "capability matrix", ["hqsb.quant.compat.check_compatibility", "hqsb.quant.compat.KernelCapability"], "supported/direct/repack/fallback/reject probed, not read from docs"),
            (4, "generate test artifacts", ["hqsb.quant.fixtures.build_tiny_artifact", "hqsb.quant.artifact.QuantArtifactDocument.add_variant"], "canonical and packed hashes for random/boundary/real tensors"),
            (5, "independent pack/unpack verification", ["hqsb.quant.packing.unpack_independent", "hqsb.quant.artifact.verify_variant_against_canonical"], "CPU and kernel-side decoders do not share index code"),
            (6, "kernel oracle", ["hqsb.quant.oracle.kernel_oracle", "hqsb.quant.oracle.correctness_report"], "high-precision dequant-GEMM reference before any timing"),
            (7, "memory safety", ["ops.quant.safety.guarded_buffer", "ops.quant.safety.sanitizer_command", "ops.quant.safety.sanitizer_report"], "guards plus sanitizer; a missing tool is reported as a gap"),
            (8, "microbenchmark", ["ops.quant.microbench.benchmark_callable", "ops.quant.microbench.MicrobenchResult"], "FP16/storage-only/mature/HQSB fused under one protocol"),
            (9, "dequant decomposition", ["ops.quant.microbench.dequant_decomposition_plan", "ops.quant.executors.get_executor"], "unpack-only … fused … FP16 paths, launch and bytes recorded"),
            (10, "decode profile", ["ops.quant.microbench.workspace_report", "hqsb.quant.execution.ExecutionRecord"], "M=1 and small-M real projections analysed"),
            (11, "prefill profile", ["hqsb.quant.model_eval.PhaseTimer", "hqsb.quant.execution.ExecutionRecord"], "large-M reuse/compute utilisation and dequant share"),
            (12, "tail/alignment ablation", ["hqsb.quant.packing.plan_kernel_layout", "hqsb.quant.packing.packed_row_bytes"], "boundary shapes are not silent fallbacks; padding cost observed"),
            (13, "single-layer real model hook-up", ["hqsb.quant.apply.swap_weights", "ops.quant.executors.get_executor"], "operator/block/logit verified with the module→kernel mapping"),
            (14, "expand to a model policy", ["hqsb.quant.apply.swap_weights", "hqsb.quant.execution.claim_audit"], "per-layer dispatch and coverage recorded"),
            (15, "six-workload correctness", ["hqsb.quant.quality.teacher_forcing_report", "hqsb.quant.quality.generation_report"], "quantization oracle vs kernel output"),
            (16, "six-workload performance/memory/energy", ["hqsb.quant.model_eval.PhaseTimer", "hqsb.quant.model_eval.device_memory_ladder"], "≥3 processes, phases separated, kernel contribution recorded"),
            (17, "order and process replication", ["hqsb.quant.model_eval.measurement_gate"], "excludes autotune/cache/clock ordering bias"),
            (18, "negative capability tests", ["hqsb.quant.compat.check_compatibility", "hqsb.quant.faults.run_fault_matrix"], "unsupported layout/group/shape/arch refused or explicitly repacked"),
            (19, "causal report", ["hqsb.quant.sensitivity.surrogate_validation", "hqsb.quant.decision.figure_points_from_table"], "micro speedup × call frequency vs end-to-end (Amdahl)"),
            (20, "adjudicate", ["hqsb.quant.model_eval.gate_order", "hqsb.quant.decision.evaluate_gates"], "correctness/authenticity/statistics/model-hit judged separately"),
        ]
    ),
)

E05_07 = ExperimentMapping(
    experiment_id="E05-07",
    title="W8A8 static/dynamic activation quant (P1)",
    driver="scripts/quant/run_e05.py --experiment E05-07",
    fixture="hqsb.quant.activation.quantize_activation (pure-Python unit tests)",
    steps=_steps(
        [
            (1, "freeze ActivationQuantSpec", ["hqsb.quant.activation.ActivationQuantSpec", "hqsb.quant.activation.scheme_for_activation"], "static/dynamic, axis, qrange, round, clip, dtype, NaN, kernel/fallback"),
            (2, "freeze calibrated candidates", ["hqsb.quant.calibration.DataSpec", "hqsb.quant.activation.compute_static_scales"], "clip/alpha/granularity grid pre-registered on policy-validation"),
            (3, "FP16 activation profile", ["hqsb.quant.calibration.collect_activation_stats"], "range/percentile/outlier/drift across six workloads and positions"),
            (4, "generate static scales", ["hqsb.quant.activation.compute_static_scales", "hqsb.quant.artifact.CalibrationProvenance"], "count, statistics hash and calibration provenance"),
            (5, "dynamic scale reference", ["hqsb.quant.activation.quantize_activation"], "per-token/per-row reduction, mask and scale verified on CPU"),
            (6, "quantizer boundary tests", ["hqsb.quant.activation.quantize_activation", "hqsb.quant.activation.ActivationQuantResult"], "zero/constant/extrema/outlier/NaN/empty/tail/rounding ties"),
            (7, "SmoothQuant equivalence", ["hqsb.quant.activation.compute_smooth_scale", "hqsb.quant.adapters.base.transform_equivalence_check"], "unquantized equivalence and graph hash before W8A8"),
            (8, "fake W8A8", ["hqsb.quant.apply.fake_quant_weight", "hqsb.quant.quality.teacher_forcing_report"], "isolates activation+weight quantization quality"),
            (9, "W8A8 kernel oracle", ["hqsb.quant.activation.w8a8_reference", "hqsb.quant.activation.int32_accumulate"], "exact integer accumulation vs kernel; scale broadcast/epilogue"),
            (10, "actual instruction/kernel verification", ["hqsb.quant.execution.classify_execution", "ops.quant.executors.executor_matrix"], "symbol/provider/profile/fallback; FP16 restore is not W8A8"),
            (11, "microbenchmark decomposition", ["hqsb.quant.activation.OnlineCostBreakdown", "ops.quant.microbench.benchmark_callable"], "static/dynamic/granularity/fusion, prefill and decode separated"),
            (12, "saturation and quality curves", ["hqsb.quant.activation.drift_report", "hqsb.quant.quality.evaluate_quality_gate"], "whole sweep reported on policy-validation, not only the best point"),
            (13, "freeze candidates", ["hqsb.quant.policy.MixedPrecisionPolicy.validate"], "only quality-passing, kernel-supported configurations"),
            (14, "model operator/block/logit", ["hqsb.quant.apply.attach_capture_hooks", "hqsb.quant.quality.position_metrics"], "scale applied at the right module and position"),
            (15, "six workloads", ["hqsb.quant.model_eval.PhaseTimer", "hqsb.quant.execution.ExecutionRecord"], "quality/TTFT/TPOT/TPS/memory/energy plus per-layer saturation"),
            (16, "distribution drift stress", ["hqsb.quant.activation.drift_report", "hqsb.quant.quality.evaluate_quality_gate"], "same/other domain, short/long, outlier-heavy; static failures visible"),
            (17, "long generation", ["hqsb.quant.quality.GenerationStep", "hqsb.quant.quality.generation_report"], "dynamic scale sequence, per-step KL, NaN/Inf, context-dependent cost"),
            (18, "online cost attribution", ["hqsb.quant.activation.OnlineCostBreakdown", "hqsb.quant.sensitivity.surrogate_validation"], "extra quant time vs GEMM saving aligned with Amdahl"),
            (19, "new-process/graph-mode replication", ["hqsb.quant.artifact.load_document", "hqsb.quant.experiment.RunDirectory"], "scale state and artifact reload independent of the old process"),
            (20, "P1 claim adjudication", ["hqsb.quant.model_eval.gate_order", "hqsb.quant.execution.claim_audit"], "W8A8 claims need quality, real kernel, online benefit and a stated domain"),
        ]
    ),
)

E05_08 = ExperimentMapping(
    experiment_id="E05-08",
    title="KV cache INT8/INT4 and long context (P1)",
    driver="scripts/quant/run_e05.py --experiment E05-08",
    fixture="hqsb.quant.kv.attention_comparison_report (pure-Python attention oracle)",
    steps=_steps(
        [
            (1, "freeze KVQuantSpec", ["hqsb.quant.kv.KvQuantSpec"], "K/V bits, granularity, axis, block/page, scale dtype, RoPE point, residual window, layout"),
            (2, "export the real cache schema", ["hqsb.quant.kv.KvQuantSpec.as_dict", "hqsb.quant.model_eval.MemoryLadder"], "L/H_kv/D_h/layout/stride/page/write point/dtype from the runtime"),
            (3, "FP16 capacity model", ["hqsb.quant.kv.fp16_baseline_bytes", "hqsb.quant.kv.KvCapacity.reconcile"], "align with the runtime before deriving INT8/INT4"),
            (4, "K/V distributions", ["hqsb.quant.calibration.collect_activation_stats", "hqsb.quant.kv.quantize_kv_tensor"], "layer/head/position/workload/domain; K vs V; pre/post-RoPE explicit"),
            (5, "pre-registered candidates", ["hqsb.quant.kv.context_generation_matrix"], "granularity/clip/block/residual chosen on policy-validation only"),
            (6, "quantizer and pack verification", ["hqsb.quant.kv.quantize_kv_tensor", "hqsb.quant.packing.pack_canonical"], "synthetic boundaries plus real KV, tail/page boundary, cross-step append"),
            (7, "fake KV quality", ["hqsb.quant.kv.attention_reference", "hqsb.quant.kv.attention_comparison_report"], "q-deq K/V through a common attention reference"),
            (8, "attention oracle", ["hqsb.quant.kv.attention_comparison_report"], "QK scores, softmax KL/JS, output error localise propagation"),
            (9, "actual cache write path", ["hqsb.quant.kv.KvStepRecord", "hqsb.quant.execution.ExecutionRecord"], "per-token address/page/hash after quantize/pack/write"),
            (10, "actual attention read path", ["hqsb.quant.kv.generation_sweep"], "read from the quantized cache; no hidden FP16 shadow cache"),
            (11, "observed kernel verification", ["hqsb.quant.execution.claim_audit", "ops.quant.executors.get_executor"], "write/quant/dequant/attention kernels and memcpy related to module/step"),
            (12, "context-length sweep", ["hqsb.quant.kv.context_generation_matrix", "hqsb.quant.model_eval.PhaseTimer"], "correctness before performance; effective/allocated tokens recorded"),
            (13, "generation-length sweep", ["hqsb.quant.kv.generation_sweep"], "accumulation, TPOT and energy vs output length"),
            (14, "K/V granularity ablation", ["hqsb.quant.kv.KvQuantSpec", "hqsb.quant.stats.paired_bootstrap_ci"], "K and V changed separately; no paper transplant"),
            (15, "metadata and fragmentation", ["hqsb.quant.kv.KvCapacity", "hqsb.quant.model_eval.MemoryLadder"], "payload/metadata/padding/fragmentation/workspace from the allocator"),
            (16, "maximum usable context", ["hqsb.quant.kv.max_context_search"], "fixed safety margin, every probe recorded"),
            (17, "final quality", ["hqsb.quant.quality.evaluate_quality_gate"], "frozen strategy evaluated once on final and long-context tasks"),
            (18, "six workloads", ["hqsb.quant.model_eval.PhaseTimer"], "all six run plus the pre-registered long-context matrix"),
            (19, "write/read cost attribution", ["hqsb.quant.kv.KvStepRecord", "ops.quant.microbench.benchmark_callable"], "profile splits quant-write/dequant-read/attention/page management"),
            (20, "new-process and sequence isolation", ["hqsb.quant.kv.generation_sweep", "hqsb.quant.experiment.interface_only_run"], "cache reset, cross-request contamination, scale state, reload"),
            (21, "adjudicate", ["hqsb.quant.decision.evaluate_gates"], "quality/real cache/capacity closure/performance/applicability judged separately"),
        ]
    ),
)

E05_09 = ExperimentMapping(
    experiment_id="E05-09",
    title="artifact compatibility, reload, repack and recovery",
    driver="scripts/quant/run_e05.py --experiment E05-09",
    fixture="hqsb.quant.fixtures.save_golden_artifacts",
    steps=_steps(
        [
            (1, "freeze schema and compatibility policy", ["hqsb.quant.compat.CompatibilityStatus", "hqsb.quant.compat.FALLBACK_MODES", "hqsb.quant.compat.MigrationRule"], "versioned required/optional/critical fields, migrations, decisions, error codes, modes"),
            (2, "generate golden artifacts", ["hqsb.quant.fixtures.save_golden_artifacts", "hqsb.quant.artifact.QuantArtifactDocument.identity_hash"], "RTN W8/W4 with canonical plus packed variants and expected hashes"),
            (3, "new-process golden load", ["hqsb.quant.artifact.load_document", "hqsb.quant.artifact.validate_artifact_dir"], "clean process, full validation, operator/model output, observed kernel"),
            (4, "CPU metadata validation", ["hqsb.quant.artifact.load_manifest", "hqsb.quant.compat.check_compatibility"], "schema/identity/hash/capability plan without loading payloads"),
            (5, "cross-device load", ["hqsb.quant.compat.check_compatibility", "hqsb.quant.compat.KernelCapability"], "supported path tested; unsupported path refused or repacked; no fabricated multi-GPU claim"),
            (6, "version matrix", ["hqsb.quant.compat.plan_migration"], "old reader/new artifact, new reader/old artifact, minor migration, major rejection"),
            (7, "integrity fault injection", ["hqsb.quant.faults.run_fault_matrix", "hqsb.quant.faults.matrix_plan_json"], "truncation/bit flips/missing/concurrent writes on copies only"),
            (8, "model mismatch", ["hqsb.quant.faults.MutationCase", "hqsb.quant.compat.check_compatibility"], "per-field tampering localised to a field or tensor"),
            (9, "quant mismatch", ["hqsb.quant.faults.build_fault_matrix"], "bit/group/axis/scale/zero/tail/method refused before pack or kernel"),
            (10, "packing mismatch", ["hqsb.quant.artifact.verify_variant_against_canonical", "hqsb.quant.compat.check_compatibility"], "layout/byte order/packed tensor/parent hash cannot be misinterpreted"),
            (11, "kernel capability mismatch", ["hqsb.quant.compat.check_compatibility"], "unsupported arch/shape/group/dtype/ABI act before dispatch"),
            (12, "lossless repack", ["hqsb.quant.compat.repack", "hqsb.quant.compat.RepackProvenance"], "invariants preserved, new provenance, kernel oracle re-checked"),
            (13, "illegal 'repack'", ["hqsb.quant.compat.is_requantize"], "bit/group/scale changes classified as requantize, not repack"),
            (14, "explicit fallback", ["hqsb.quant.compat._apply_mode_policy", "hqsb.quant.execution.EXPLICIT_FALLBACK"], "allowed and forbidden modes both tested; result metadata changes"),
            (15, "atomic save faults", ["hqsb.quant.artifact.QuantArtifactDocument.save"], "readers only see a complete old or new version"),
            (16, "resource failure", ["hqsb.quant.compat.check_compatibility", "ops.quant.safety.sanitizer_report"], "workspace/OOM/extension failures release resources without dirty cache"),
            (17, "cross-process output equality", ["hqsb.quant.artifact.load_document", "hqsb.quant.oracle.correctness_report"], "same artifact, same dispatch, same tolerant output"),
            (18, "end-to-end model reload", ["hqsb.quant.apply.swap_weights", "hqsb.quant.execution.claim_audit"], "representative prefill/decode quality, kernel, fallback"),
            (19, "error localisation audit", ["hqsb.quant.faults.summarize_fault_matrix"], "field path, expected/actual summary, artifact id, reason code"),
            (20, "compatibility matrix", ["hqsb.quant.compat.compatibility_matrix_rows", "hqsb.quant.compat.matrix_to_json"], "direct/repack/requantize/fallback/reject generated from raw cases"),
        ]
    ),
)

E05_10 = ExperimentMapping(
    experiment_id="E05-10",
    title="quality-constrained memory/speed/energy Pareto decision",
    driver="scripts/quant/run_e05.py --experiment E05-10",
    fixture="hqsb.quant.decision.CandidateRegistry (synthetic candidates in unit tests)",
    steps=_steps(
        [
            (1, "freeze DecisionSpec", ["hqsb.quant.decision.CandidateRegistry", "hqsb.quant.decision.OBJECTIVES", "hqsb.quant.quality.QualityBudget"], "hardware/workload/SLO/quality margin/slices/objectives/statistics/dominance/exclusions"),
            (2, "build the candidate registry", ["hqsb.quant.decision.Candidate", "hqsb.quant.decision.REQUIRED_IDENTITY_FIELDS"], "identity bound to method/config/artifact/kernel/hardware/result hashes"),
            (3, "completeness audit", ["hqsb.quant.decision.CandidateRegistry.completeness_audit"], "missing fields mark INCOMPLETE; no interpolation"),
            (4, "correctness/artifact gate", ["hqsb.quant.decision.evaluate_gates"], "failed candidates stay in the waterflow table"),
            (5, "quality gate", ["hqsb.quant.quality.evaluate_quality_gate", "hqsb.quant.stats.non_inferiority_verdict"], "paired non-inferiority with CI and per-slice checks"),
            (6, "execution gate", ["hqsb.quant.decision._execution_verdict", "hqsb.quant.execution.claim_audit"], "algorithm-quality and deployment candidates listed separately"),
            (7, "measurement gate", ["hqsb.quant.model_eval.measurement_gate"], "process count, warmup, sync, environment stability, raw completeness"),
            (8, "unify units and directions", ["hqsb.quant.model_eval.TABLE_FIELDS", "hqsb.quant.quality.DIRECTIONS"], "original values kept, only directions are normalised"),
            (9, "recompute statistics from raw", ["hqsb.quant.stats.paired_bootstrap_ci", "hqsb.quant.stats.bootstrap_ci"], "summary is recomputed, not trusted from per-experiment files"),
            (10, "baseline consistency", ["hqsb.quant.experiment.Preregistration", "hqsb.quant.stats.summarize_distribution"], "FP16 drift must stay inside the pre-registered band"),
            (11, "theoretical vs measured resources", ["hqsb.quant.artifact.SizeBreakdown.reconcile", "hqsb.quant.kv.KvCapacity.reconcile", "hqsb.quant.sensitivity.surrogate_validation"], "byte/speedup/Amdahl closure with the residual reported"),
            (12, "causal ablations", ["hqsb.quant.sensitivity.interaction_table", "hqsb.quant.sensitivity.plan_leave_one_out"], "bit/algorithm/policy/kernel/activation/KV/runtime controls"),
            (13, "group by hardware × workload", ["hqsb.quant.decision.point_pareto"], "no cross-scenario averaging"),
            (14, "point-estimate Pareto", ["hqsb.quant.decision.point_pareto"], "hard-gate-passing candidates only"),
            (15, "uncertainty-aware Pareto", ["hqsb.quant.decision.uncertainty_pareto", "hqsb.quant.decision.UncertaintyFront"], "bootstrap front inclusion probability and dominance classes"),
            (16, "apply deployment SLOs", ["hqsb.quant.decision.apply_scenarios", "hqsb.quant.decision.Scenario"], "scenarios A–E; infeasible sets reported as such"),
            (17, "offline amortization", ["hqsb.quant.decision.offline_amortization"], "several request volumes; unamortized value kept"),
            (18, "failure/fallback risk", ["hqsb.quant.compat.CompatibilityStatus", "hqsb.quant.execution.EXPLICIT_FALLBACK"], "migration/unsupported-shape/OOM/fallback/calibration-shift/long-context risks"),
            (19, "choose the recommendation set", ["hqsb.quant.decision.recommendation_matrix"], "recommended/conditional/research-only/rejected/incomplete"),
            (20, "decision figures and tables", ["hqsb.quant.decision.figure_points_from_table", "hqsb.quant.model_eval.build_unified_table"], "all figures from versioned scripts and raw rows"),
            (21, "independent re-check", ["hqsb.quant.decision.evaluate_gates", "hqsb.quant.decision.point_pareto"], "a second pass recomputes gates/fronts and spot-checks raw rows"),
            (22, "freeze the S05 release bundle", ["hqsb.quant.decision.release_bundle_manifest"], "artifacts/policies/compatibility/registry/figures/limitations/S06 interface"),
            (23, "define regression thresholds", ["hqsb.quant.decision.regression_thresholds"], "quality/memory/phase-performance baselines for S06/S07"),
            (24, "final adjudication", ["hqsb.quant.experiment.RunDirectory.write_verdict", "hqsb.quant.experiment.check_prerequisites"], "S05 completion by P0 gates, not by finding a faster option"),
        ]
    ),
)

EXPERIMENT_MAP: Tuple[ExperimentMapping, ...] = (
    E05_01,
    E05_02,
    E05_03,
    E05_04,
    E05_05,
    E05_06,
    E05_07,
    E05_08,
    E05_09,
    E05_10,
)


def mapping_for(experiment_id: str) -> ExperimentMapping:
    for mapping in EXPERIMENT_MAP:
        if mapping.experiment_id == experiment_id:
            return mapping
    raise KeyError(f"no mapping for {experiment_id!r}")


def iter_interface_symbols() -> List[str]:
    symbols: List[str] = []
    for mapping in EXPERIMENT_MAP:
        for step in mapping.steps:
            symbols.extend(step.interfaces)
    return sorted(set(symbols))


def resolve_interface(symbol: str):
    """Resolve a dotted reference such as ``mod.Class.method``.

    The longest importable prefix is imported, then the remaining components
    are walked with ``getattr`` — so class attributes and methods are checked
    too, not just module-level names.
    """
    if "." not in symbol:
        raise ValueError(f"{symbol!r} is not a dotted reference")
    parts = symbol.split(".")
    module = None
    index = len(parts)
    while index > 0:
        candidate = ".".join(parts[:index])
        try:
            module = importlib.import_module(candidate)
            break
        except ModuleNotFoundError as exc:
            if exc.name != candidate and not candidate.startswith(str(exc.name) + "."):
                raise
            index -= 1
    if module is None:
        raise ModuleNotFoundError(f"no importable prefix in {symbol!r}")
    target = module
    for part in parts[index:]:
        target = getattr(target, part)
    return target


def resolve_interfaces() -> Dict[str, Any]:
    """Verify that every mapped interface imports (the map cannot rot)."""
    failures: List[Dict[str, str]] = []
    symbols = iter_interface_symbols()
    for symbol in symbols:
        try:
            resolve_interface(symbol)
        except Exception as exc:  # noqa: BLE001 - any failure is a broken mapping
            failures.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
    steps_total = sum(len(mapping.steps) for mapping in EXPERIMENT_MAP)
    return {
        "experiments": [mapping.experiment_id for mapping in EXPERIMENT_MAP],
        "steps": steps_total,
        "interfaces": len(symbols),
        "failures": failures,
        "ok": not failures,
    }


def mapping_table_markdown() -> str:
    """Render the step→interface table used in the development report."""
    lines: List[str] = []
    for mapping in EXPERIMENT_MAP:
        lines.append(f"### {mapping.experiment_id} — {mapping.title}")
        lines.append("")
        lines.append(f"Driver: `{mapping.driver}`  ")
        if mapping.fixture:
            lines.append(f"Fixture/self-check: `{mapping.fixture}`")
        lines.append("")
        lines.append("| 步骤 | 能力 | 接口落点 | 成熟度 |")
        lines.append("|---|---|---|---|")
        for step in mapping.steps:
            interfaces = "<br>".join(f"`{symbol}`" for symbol in step.interfaces)
            lines.append(
                f"| {step.step} | {step.title} | {interfaces} | {step.maturity} |"
            )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "EXPERIMENT_MAP",
    "E05_01",
    "E05_02",
    "E05_03",
    "E05_04",
    "E05_05",
    "E05_06",
    "E05_07",
    "E05_08",
    "E05_09",
    "E05_10",
    "ExperimentMapping",
    "StepMapping",
    "iter_interface_symbols",
    "mapping_for",
    "mapping_table_markdown",
    "resolve_interface",
    "resolve_interfaces",
]
