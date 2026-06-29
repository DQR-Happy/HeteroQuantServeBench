"""「实验步骤 → 代码接口」对照表 for all ten S11 experiments (320 steps).

Each experiment's 32 protocol steps are mapped to concrete interfaces in this
package; ``resolve_interfaces`` imports every referenced symbol so the mapping
cannot rot into documentation — a renamed function fails the audit instead of
silently pointing at nothing.

Entries are data only: no interface here executes an experiment.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hqsb.core.errors import ConfigError

MODULE_BY_PREFIX: Dict[str, str] = {
    "identity": "hqsb.compiler.identity",
    "ir": "hqsb.compiler.ir",
    "records": "hqsb.compiler.records",
    "capture": "hqsb.compiler.capture",
    "guards": "hqsb.compiler.guards",
    "pattern_library": "hqsb.compiler.pattern_library",
    "rewrite": "hqsb.compiler.rewrite",
    "targets": "hqsb.compiler.targets",
    "lowering": "hqsb.compiler.lowering",
    "backend": "hqsb.compiler.backend",
    "codegen": "hqsb.compiler.codegen",
    "autotune": "hqsb.compiler.autotune",
    "costmodel": "hqsb.compiler.costmodel",
    "cache": "hqsb.compiler.cache",
    "portable": "hqsb.compiler.portable",
    "aigate": "hqsb.compiler.aigate",
    "telemetry": "hqsb.compiler.telemetry",
    "specs": "hqsb.compiler.specs",
    "experiment": "hqsb.compiler.experiment",
}


@dataclass
class StepMapping:
    """One protocol step: its title and the interfaces that implement it."""

    index: int
    title: str
    interfaces: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "title": self.title, "interfaces": list(self.interfaces)}


@dataclass
class ExperimentMapping:
    """One experiment: 32 steps, a driver entry and the claim boundary."""

    experiment_id: str
    title: str
    level: str = "P0"
    claim_boundary: str = ""
    steps: Tuple[StepMapping, ...] = ()

    @property
    def driver(self) -> str:
        return f"scripts/compiler/run_e11.py --experiment {self.experiment_id}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "level": self.level,
            "driver": self.driver,
            "claim_boundary": self.claim_boundary,
            "steps": [step.as_dict() for step in self.steps],
        }


def _steps(pairs: Sequence[Tuple[int, str, Sequence[str]]]) -> Tuple[StepMapping, ...]:
    return tuple(StepMapping(index=index, title=title, interfaces=tuple(interfaces)) for index, title, interfaces in pairs)


E11_01 = ExperimentMapping(
    experiment_id="E11-01",
    title="真实 Qwen Graph Capture、Graph Break、Guard 与 Symbolic Domain",
    claim_boundary="捕获边界被完整测量；不证明 rewrite 正确、lowering 已执行或有性能收益",
    steps=_steps(
        [
            (1, "冻结实验身份", ["experiment:RunDirectory", "experiment:environment_fingerprint", "experiment:git_state", "identity:ArtifactIdentity"]),
            (2, "检查前置证据", ["experiment:check_prerequisites", "experiment:PrerequisiteStatus"]),
            (3, "冻结 Qwen 模块结构", ["capture:CaptureRequest", "capture:REGIONS"]),
            (4, "冻结 phase 与状态边界", ["capture:CaptureRequest", "capture:WORKLOADS"]),
            (5, "建立 eager execution census", ["capture:EagerCensus", "capture:EagerCensusEntry"]),
            (6, "实现无优化 Dynamo debug backend", ["capture:DebugBackend", "capture:DebugBackendResult", "backend:BackendHooks"]),
            (7, "定义 graph 序列化与 canonicalization", ["ir:IRGraph", "ir:IRGraph.to_text", "identity:content_hash"]),
            (8, "定义 source lineage 采集", ["capture:SourceLineage", "capture:MISSING_REASONS"]),
            (9, "定义 tensor metadata 采集", ["capture:TensorMetadataRecord", "capture:compare_fake_runtime"]),
            (10, "定义 effect metadata", ["ir:EffectSet", "ir:EFFECT_STATES"]),
            (11, "运行 single-block Dynamo capture", ["capture:CaptureRequest", "capture:BreakAudit", "rewrite:build_graph"]),
            (12, "运行 block fullgraph probe", ["capture:GraphBreak", "capture:BREAK_REASON_CATALOG"]),
            (13, "运行 block export", ["ir:import_fx_graph", "ir:ShapeConstraint", "identity:IR_EXPORT_ATEN"]),
            (14, "运行完整 prefill capture", ["capture:CaptureRequest", "capture:EagerCensus", "capture:BreakAudit"]),
            (15, "运行 single-step decode capture", ["capture:CaptureRequest", "capture:TRACE_PHASES"]),
            (16, "运行 multi-step decode capture", ["capture:ShapeTraceStep", "capture:ordered_shape_trace"]),
            (17, "执行六 workload 独立矩阵", ["capture:capture_matrix", "capture:WORKLOADS", "capture:REPEAT_CASES"]),
            (18, "执行有序 shape trace", ["capture:ordered_shape_trace", "capture:TRACE_EXPECTATIONS", "guards:CompileEvent"]),
            (19, "测试 batch 变化", ["ir:SymbolicDim", "guards:Variant"]),
            (20, "测试 ISL/KV length 变化", ["ir:ShapeConstraint", "guards:GuardSpec"]),
            (21, "测试 stride/layout 变化", ["capture:LAYOUT_CASES", "ir:TensorType", "guards:alignment_guard"]),
            (22, "测试 dtype/precision 分支", ["ir:DTYPE_BYTES", "guards:equality_guard"]),
            (23, "比较 custom op 开关", ["capture:BACKEND_MODES", "capture:GraphBreak"]),
            (24, "注入 missing fake/meta 负例", ["capture:missing_fake_negative", "capture:GraphBreak.validate"]),
            (25, "注入 data-dependent 与 unsupported Python 负例", ["capture:data_dependent_negative", "capture:BreakAudit"]),
            (26, "收集框架原生 trace", ["capture:NativeTracePlan", "capture:NativeTracePlan.for_torch"]),
            (27, "统一原生日志与 HQSB 事件", ["capture:join_native_and_hqsb", "telemetry:CompilerTraceRecord", "telemetry:span_chain_check"]),
            (28, "计算 coverage 与 metadata completeness", ["capture:CoverageReport", "capture:MetadataCompleteness", "capture:coverage_claim_guard"]),
            (29, "做同进程确定性复验", ["capture:repeatability_check", "ir:IRGraph.canonical_hash"]),
            (30, "做独立冷进程复验", ["capture:repeatability_check", "experiment:RunDirectory"]),
            (31, "选择并冻结 hero graph", ["capture:HeroGraphManifest", "identity:graph_identity"]),
            (32, "形成捕获边界裁决", ["capture:CoverageReport.claim_rules", "records:CaseState", "experiment:RunDirectory.write_verdict"]),
        ]
    ),
)

E11_02 = ExperimentMapping(
    experiment_id="E11-02",
    title="语义 Pattern Rewrite、Near-miss 拒绝与 Pass 幂等性",
    claim_boundary="重写合法性；不代表 candidate 更快或已 lower 到自定义 kernel",
    steps=_steps(
        [
            (1, "冻结输入图与语义版本", ["rewrite:build_graph", "identity:graph_identity", "records:PatternDecisionRecord"]),
            (2, "选择首个真实 pattern", ["pattern_library:residual_add_rmsnorm_contract", "pattern_library:contract_by_id"]),
            (3, "写数学与状态合同", ["pattern_library:PatternContract", "pattern_library:composed_add_rmsnorm_reference"]),
            (4, "定义 semantic pattern ID", ["pattern_library:PatternSignature", "pattern_library:residual_rmsnorm_signature", "pattern_library:contract_digest"]),
            (5, "定义 structural matcher", ["rewrite:find_candidates", "rewrite:StructuralCandidate", "pattern_library:ROLE_CHAIN_RESIDUAL_RMSNORM"]),
            (6, "定义 dtype/cast predicates", ["rewrite:evaluate_predicates", "pattern_library:PREDICATE_CATALOG"]),
            (7, "定义 shape/reduction predicates", ["rewrite:evaluate_predicates", "ir:ShapeConstraint"]),
            (8, "定义 epsilon predicates", ["rewrite:ALLOWED_EPS", "rewrite:evaluate_predicates"]),
            (9, "定义 users/liveness predicates", ["rewrite:evaluate_predicates", "rewrite:Decision"]),
            (10, "定义 alias/mutation/effect predicates", ["ir:EffectSet.risk_state", "rewrite:evaluate_predicates"]),
            (11, "定义 layout predicates", ["rewrite:evaluate_predicates", "lowering:preflight_checks"]),
            (12, "实现 replacement builder", ["rewrite:apply_rewrite", "ir:IROp"]),
            (13, "实现 proof record", ["rewrite:PredicateResult", "records:PatternDecisionRecord.validate"]),
            (14, "实现 graph verifier", ["ir:IRVerifier", "ir:verify_graph", "telemetry:VERIFIER_ISSUE_FIELDS"]),
            (15, "建立真实 positive corpus", ["pattern_library:corpus_plan", "rewrite:build_corpus_rows"]),
            (16, "建立 synthetic positive corpus", ["pattern_library:residual_add_rmsnorm_graph", "pattern_library:DECOMPOSITION_VARIANTS"]),
            (17, "建立 near-miss mutation operators", ["pattern_library:NEAR_MISS_MUTATION_AXES", "pattern_library:mutation_axes_table"]),
            (18, "建立 hard negative corpus", ["pattern_library:CORPUS_CLASSES", "pattern_library:MATCH_CLASS_TABLE"]),
            (19, "执行 matcher-only 分析", ["rewrite:find_candidates", "rewrite:StructuralCandidate.as_dict"]),
            (20, "执行 positive rewrite", ["rewrite:apply_rewrite", "rewrite:provenance_audit"]),
            (21, "执行 near-miss 拒绝测试", ["rewrite:evaluate_corpus", "rewrite:CorpusRow.expectation_met"]),
            (22, "测试 extra-user preservation", ["rewrite:evaluate_predicates", "rewrite:provenance_audit"]),
            (23, "测试 mutation/alias 边界", ["ir:EffectSet", "rewrite:atomicity_report"]),
            (24, "做 operator differential correctness", ["pattern_library:composed_add_rmsnorm_reference", "pattern_library:compare_reference_paths"]),
            (25, "做 block differential correctness", ["lowering:CorrectnessEvidence", "lowering:EvidenceIndex"]),
            (26, "做 model prefill correctness", ["lowering:CORRECTNESS_ORDER", "lowering:next_gate"]),
            (27, "做多步 decode correctness", ["lowering:CORRECTNESS_ORDER", "lowering:performance_allowed"]),
            (28, "运行 pass 第二次/多次", ["rewrite:idempotence_report", "rewrite:metadata_diff"]),
            (29, "测试 pass 顺序与固定点", ["rewrite:run_pipeline", "rewrite:DEFAULT_PIPELINE", "rewrite:PassSpec"]),
            (30, "注入 pass 中途失败", ["rewrite:atomicity_report", "rewrite:INJECTION_POINTS"]),
            (31, "独立进程确定性复验", ["rewrite:determinism_report", "rewrite:pass_identity"]),
            (32, "形成 pass 裁决与交付", ["rewrite:decision_rows", "rewrite:PassOutcome", "experiment:RunDirectory.write_verdict"]),
        ]
    ),
)

E11_03 = ExperimentMapping(
    experiment_id="E11-03",
    title="Graph→HQSB IR→Lowering Registry→真实 Kernel 闭环",
    claim_boundary="最小编译闭环成立；不代表动态 shape 优化、autotune 泛化或 cache 生命周期完整",
    steps=_steps(
        [
            (1, "冻结 hero graph 与 reference", ["capture:HeroGraphManifest", "lowering:CorrectnessEvidence", "identity:identity_chain_status"]),
            (2, "选择首个 target kernel", ["lowering:custom_kernel_entry", "lowering:LoweringEntry"]),
            (3, "定义 HQSB canonical IR schema", ["ir:IRGraph", "ir:IROp", "ir:IRValue"]),
            (4, "定义 targeted IR schema", ["ir:IRGraph", "lowering:LoweringDecision", "guards:Variant"]),
            (5, "实现 IR serialization 与 canonical hash", ["ir:IRGraph.to_json", "ir:IRGraph.from_dict", "ir:IRGraph.canonical_hash"]),
            (6, "实现 IR verifier", ["ir:IRVerifier", "ir:VerifierReport"]),
            (7, "实现 FX/Export→canonical IR importer", ["ir:import_fx_graph", "ir:import_op_sequence"]),
            (8, "建立 target snapshot", ["targets:TargetSnapshot", "targets:probe_target", "targets:cuda_target_snapshot"]),
            (9, "定义 lowering registry schema", ["lowering:LoweringRegistry", "lowering:LoweringEntry", "lowering:LoweringRegistry.snapshot"]),
            (10, "注册 reference lowering", ["lowering:reference_lowering_entry", "lowering:materialize"]),
            (11, "注册 custom kernel lowering", ["lowering:custom_kernel_entry", "lowering:EvidenceIndex"]),
            (12, "实现 capability analysis", ["targets:CapabilityRequirement", "targets:capability_matrix", "targets:CAPABILITY_REASONS"]),
            (13, "实现 runtime guard builder", ["guards:GuardSpec", "guards:range_guard", "guards:alignment_guard"]),
            (14, "实现 deterministic baseline selector", ["lowering:parse_policy", "lowering:evaluate_candidates", "lowering:CandidateEvaluation"]),
            (15, "实现 callable/materialization", ["lowering:materialize", "lowering:MaterializedPlan", "backend:HQSBBackend"]),
            (16, "实现 compile/runtime telemetry", ["lowering:DispatchTelemetry", "lowering:DispatchRow", "telemetry:project_c6"]),
            (17, "导出所有层级 IR/artifact", ["identity:ArtifactIdentity", "identity:artifact_index_rows", "backend:emit_artifact_manifest"]),
            (18, "验证 IR round-trip/重建", ["ir:round_trip", "lowering:rebuild_plan"]),
            (19, "运行 reference lowering", ["lowering:evaluate_candidates", "lowering:materialize"]),
            (20, "运行 forced custom operator test", ["lowering:parse_policy", "lowering:preflight_checks"]),
            (21, "运行 Qwen block test", ["lowering:performance_allowed", "lowering:CORRECTNESS_ORDER"]),
            (22, "运行 Qwen prefill", ["lowering:compile_breakdown", "records:CompileRun"]),
            (23, "运行多步 decode", ["guards:GuardEventRecord", "records:CaseState"]),
            (24, "做 actual kernel 双重确认", ["lowering:ActualDispatchEvidence", "lowering:EVIDENCE_KINDS", "lowering:disable_ablation_plan"]),
            (25, "测 framework default compile 对照", ["backend:BackendHooks", "backend:torch_compile_invocation", "backend:assert_not_debug_backend"]),
            (26, "测 compile-time breakdown", ["lowering:compile_breakdown", "records:merge_step_timings", "records:COMPILER_COST_KEYS"]),
            (27, "测 steady performance", ["codegen:ProfilePlan", "codegen:amdahl_attribution"]),
            (28, "测试 guard-false shape/layout/dtype", ["guards:VariantRegistry", "guards:LookupResult", "guards:domain_coverage"]),
            (29, "测试 capability/ABI/version 不兼容", ["lowering:inject_lowering_failure", "lowering:INJECTABLE_LOWERING_FAILURES"]),
            (30, "测试 compile 与 runtime failure", ["lowering:SideEffectGate", "lowering:SideEffectGate.on_runtime_error"]),
            (31, "独立冷进程重建 hero path", ["lowering:rebuild_plan", "experiment:RunDirectory", "identity:LineageGraph"]),
            (32, "形成闭环 verdict", ["lowering:success_status_for", "codegen:reconstruction_check", "experiment:RunDirectory.write_verdict"]),
        ]
    ),
)

E11_04 = ExperimentMapping(
    experiment_id="E11-04",
    title="Before/After IR、Generated Code、PTX/SASS 与硬件行为归因",
    claim_boundary="至少一个编译决策可被工程证据解释；不证明该解释适用于其他模型/架构",
    steps=_steps(
        [
            (1, "冻结正式 compile/run IDs", ["identity:ArtifactIdentity", "identity:LineageGraph"]),
            (2, "预注册 case 选择规则", ["experiment:Preregistration", "codegen:MechanismHypothesis"]),
            (3, "建立一因子对照", ["codegen:ablation_matrix", "codegen:AblationCase"]),
            (4, "确认 correctness 等价", ["lowering:performance_allowed", "lowering:CORRECTNESS_ORDER"]),
            (5, "重复无 profiler timing", ["codegen:ProfilePlan", "records:amortization_row"]),
            (6, "导出 before/after source graph", ["ir:diff_graphs", "ir:IRDiff"]),
            (7, "导出 canonical/targeted HQSB IR", ["ir:graph_summary", "lowering:LoweringDecision"]),
            (8, "导出 Inductor/scheduler IR", ["codegen:toolchain_export_plan", "codegen:EXPORT_LAYERS"]),
            (9, "导出 generated source", ["codegen:GeneratedSourceArtifact", "codegen:hash_all_sources"]),
            (10, "导出 backend compiler IR", ["codegen:toolchain_export_plan", "targets:toolchain_export_capability"]),
            (11, "提取正式 binary", ["identity:ArtifactIdentity.identity_hash", "cache:EntryStore.read"]),
            (12, "反汇编与 symbol mapping", ["codegen:toolchain_export_plan", "lowering:ActualDispatchEvidence"]),
            (13, "生成结构化 IR diff", ["codegen:cross_layer_diff_rows", "ir:diff_graphs"]),
            (14, "生成 memory-plan ledger", ["codegen:MemoryLedgerRow", "codegen:memory_plan_diff"]),
            (15, "生成 launch ledger", ["codegen:LaunchLedgerRow", "codegen:launch_ledger_diff"]),
            (16, "采集系统 timeline", ["codegen:ProfilePlan", "codegen:ProfilePlan.commands"]),
            (17, "选择 profile kernel", ["lowering:DispatchRow", "codegen:ProfilePlan"]),
            (18, "采集 launch/resource 指标", ["codegen:parse_resource_usage", "codegen:RESOURCE_FIELDS"]),
            (19, "采集 memory workload 指标", ["codegen:parse_resource_usage", "codegen:occupancy_limits"]),
            (20, "采集 compute/scheduler 指标", ["codegen:occupancy_limits", "codegen:MechanismHypothesis"]),
            (21, "控制 profiler replay 风险", ["codegen:ProfilePlan.validate", "lowering:SideEffectGate"]),
            (22, "对齐理论 bytes/FLOPs", ["codegen:occupancy_limits", "codegen:memory_plan_diff"]),
            (23, "计算 launch/intermediate 节省", ["codegen:launch_ledger_diff", "codegen:memory_plan_diff"]),
            (24, "分析 register/shared/occupancy", ["codegen:parse_resource_usage", "codegen:occupancy_limits"]),
            (25, "分析访存与指令", ["codegen:parse_resource_usage", "codegen:MechanismHypothesis"]),
            (26, "建立机制假设表", ["codegen:MechanismHypothesis", "codegen:mechanism_verdict_table"]),
            (27, "运行单因子 ablation", ["codegen:evaluate_ablation", "codegen:ablation_matrix"]),
            (28, "跨 shape 验证机制", ["codegen:MechanismHypothesis", "targets:target_field_diff"]),
            (29, "连接 phase/model 指标", ["codegen:amdahl_attribution", "records:amortization_row"]),
            (30, "独立 confirmation run", ["experiment:RunDirectory", "codegen:reconstruction_check"]),
            (31, "验证 artifact 重建", ["codegen:reconstruction_plan", "codegen:reconstruction_check"]),
            (32, "形成 attribution verdict", ["codegen:mechanism_verdict_table", "experiment:RunDirectory.write_verdict"]),
        ]
    ),
)

E11_05 = ExperimentMapping(
    experiment_id="E11-05",
    title="Dynamic Shape、Symbolic Guard、Recompile、Variant 与 Fallback Safety",
    claim_boundary="动态行为安全、可解释且有界；不证明 autotune winner 泛化或 cost model 有效",
    steps=_steps(
        [
            (1, "冻结 identity 与基线", ["experiment:git_state", "identity:compile_identity", "guards:Variant"]),
            (2, "枚举动态变量", ["ir:SymbolicDim", "ir:DIM_ORIGINS"]),
            (3, "区分 phase shape 语义", ["ir:ShapeConstraint", "capture:TRACE_PHASES"]),
            (4, "分类现有 guards", ["guards:GUARD_CATEGORIES", "guards:DEFAULT_ACTION_BY_CATEGORY"]),
            (5, "定义 variant schema", ["guards:Variant", "guards:Variant.validate"]),
            (6, "实现 guard event telemetry", ["records:GuardEventRecord", "guards:GuardEvaluation"]),
            (7, "建立 variant overlap/coverage analyzer", ["guards:domain_coverage", "guards:VariantRegistry"]),
            (8, "冻结 shape sequence", ["capture:ShapeTraceStep", "capture:ordered_shape_trace"]),
            (9, "运行 S0 static 对照", ["guards:StrategyObservation", "guards:compare_strategies"]),
            (10, "运行 S1 automatic 对照", ["guards:STRATEGIES", "guards:compare_strategies"]),
            (11, "设计 S2 bounded constraints", ["ir:ShapeConstraint", "guards:range_guard"]),
            (12, "运行 S2 bounded", ["guards:compare_strategies", "guards:compile_explainability"]),
            (13, "运行 S3 broad dynamic", ["guards:compare_strategies", "guards:StrategyObservation"]),
            (14, "设计 S4 manual buckets", ["guards:STRATEGIES", "guards:dynamic_policy_document"]),
            (15, "运行 batch sweep", ["guards:range_guard", "guards:VariantRegistry.lookup"]),
            (16, "运行 prefill ISL sweep", ["guards:domain_coverage", "guards:GuardSpec"]),
            (17, "运行 decode past-length sweep", ["guards:CompileEvent", "guards:compile_explainability"]),
            (18, "运行混合 batch/length trace", ["guards:VariantRegistry", "guards:LookupResult"]),
            (19, "运行 stride/layout sweep", ["guards:alignment_guard", "guards:LookupResult"]),
            (20, "运行 dtype/device negative", ["guards:equality_guard", "guards:VariantRegistry.lookup"]),
            (21, "运行 tail/resource boundary", ["guards:divisibility_guard", "guards:VariantBudget"]),
            (22, "测试返回旧 shape", ["guards:VariantRegistry.lookup", "guards:compile_explainability"]),
            (23, "测试 variant limit", ["guards:VariantBudget", "guards:VariantBudget.exceed_action"]),
            (24, "测试超界 fallback", ["guards:domain_coverage", "guards:LookupResult"]),
            (25, "测试 concurrent calls", ["guards:ConcurrencyPlan", "guards:evaluate_concurrency"]),
            (26, "验证每个 actual dispatch 的 guard", ["guards:wrong_reuse_audit", "lowering:DispatchRow"]),
            (27, "计算 variant/compile 可解释性", ["guards:compile_explainability", "guards:variant_count_is_explainable"]),
            (28, "比较策略 correctness/performance", ["guards:compare_strategies", "guards:StrategyObservation"]),
            (29, "计算真实分布总成本", ["guards:compare_strategies", "records:amortization_row"]),
            (30, "独立 holdout shape 验证", ["guards:domain_coverage", "autotune:holdout_evaluate"]),
            (31, "跨进程/缓存复验", ["cache:EntryStore", "cache:CacheKeySpec"]),
            (32, "冻结 dynamic policy", ["guards:dynamic_policy_document", "guards:policy_digest", "experiment:RunDirectory.write_verdict"]),
        ]
    ),
)

E11_06 = ExperimentMapping(
    experiment_id="E11-06",
    title="Autotune Search Space、合法性过滤、独立 Holdout 与搜索预算",
    claim_boundary="搜索过程科学可复核；不要求 autotune 一定胜出（无收益但方法完整是有效负结果）",
    steps=_steps(
        [
            (1, "冻结 task/target/domain", ["autotune:SearchSpaceSpec", "lowering:LoweringRegistry", "guards:Variant"]),
            (2, "选择调优对象", ["autotune:SearchSpaceSpec.validate", "lowering:LoweringDecision"]),
            (3, "枚举有物理含义的 knobs", ["autotune:Knob", "codegen:MechanismHypothesis"]),
            (4, "定义候选 identity", ["autotune:candidate_identity", "identity:compile_identity"]),
            (5, "定义静态语义约束", ["autotune:SearchConstraint", "autotune:CONSTRAINT_CATEGORIES"]),
            (6, "定义 target/resource 约束", ["targets:CapabilityRequirement", "autotune:SearchConstraint"]),
            (7, "建立 default/heuristic baselines", ["autotune:budget_ladder", "costmodel:heuristic_pick"]),
            (8, "生成完整候选清单", ["autotune:generate_candidates", "autotune:GeneratedCandidate"]),
            (9, "验证过滤器 false reject", ["autotune:filter_false_reject_audit", "autotune:STATIC_REJECT_REASONS"]),
            (10, "冻结 shape split", ["autotune:ShapeGroup", "autotune:build_split_manifest", "autotune:split_leak_check"]),
            (11, "冻结预算梯度", ["autotune:Budget", "autotune:Budget.validate"]),
            (12, "实现 trial sandbox", ["autotune:TrialSandbox", "records:AutotuneTrialRecord"]),
            (13, "实现输入/状态重置", ["autotune:reset_protocol", "autotune:verify_reset"]),
            (14, "编译候选", ["autotune:AutotuneTrialRecord", "records:AUTOTUNE_TRIAL_FIELDS"]),
            (15, "执行 pre-benchmark correctness", ["autotune:correctness_before_benchmark", "records:AutotuneTrialRecord.validate"]),
            (16, "执行短稳定性 smoke", ["records:AutotuneTrialRecord", "autotune:measurement_schedule"]),
            (17, "设计候选测量顺序", ["autotune:measurement_schedule", "autotune:verify_reset"]),
            (18, "运行 B0 default", ["autotune:budget_ladder", "autotune:check_budget_fairness"]),
            (19, "运行 B1 小预算搜索", ["autotune:Budget", "autotune:autotune_metrics"]),
            (20, "运行 B2 中预算搜索", ["autotune:autotune_metrics", "autotune:quality_cost_curve"]),
            (21, "运行 B3/exhaustive oracle", ["autotune:per_shape_oracle", "autotune:check_budget_fairness"]),
            (22, "计算 budget-quality 曲线", ["autotune:quality_cost_curve", "autotune:autotune_metrics"]),
            (23, "选 provisional winners", ["autotune:select_winners", "autotune:candidate_identity_from_trial"]),
            (24, "独立 confirmation", ["autotune:confirmation_plan", "records:AutotuneTrialRecord"]),
            (25, "运行 interpolation holdout", ["autotune:holdout_evaluate", "autotune:HOLDOUT_KINDS"]),
            (26, "运行 boundary/extrapolation holdout", ["autotune:holdout_evaluate", "autotune:split_summary"]),
            (27, "运行 layout/dtype holdout", ["autotune:holdout_evaluate", "autotune:holdout_kinds_table"]),
            (28, "分析 winner sensitivity", ["autotune:select_winners", "codegen:evaluate_ablation"]),
            (29, "计算总搜索成本", ["autotune:search_cost_breakdown", "records:COMPILER_COST_KEYS"]),
            (30, "计算 amortization", ["autotune:amortization", "records:break_even_calls"]),
            (31, "写入版本化 tuning DB", ["autotune:tuning_database_record", "autotune:TUNING_DB_KEY_FIELDS"]),
            (32, "形成 autotune policy", ["autotune:autotune_policy_document", "experiment:RunDirectory.write_verdict"]),
        ]
    ),
)

E11_07 = ExperimentMapping(
    experiment_id="E11-07",
    title="Cost Model、Candidate Ranking、Oracle Regret 与低置信安全回退",
    claim_boundary="在冻结候选空间与声明域内，自动选择的 regret 与失败保护经过独立评价",
    steps=_steps(
        [
            (1, "冻结研究问题与 primary metric", ["experiment:Preregistration", "costmodel:RegretReport"]),
            (2, "绑定 E11-06 数据版本", ["costmodel:dataset_digest", "costmodel:DatasetUnit"]),
            (3, "定义 dataset unit 与 group", ["costmodel:DatasetUnit", "autotune:ShapeGroup"]),
            (4, "冻结 feature schema", ["costmodel:FeatureSchema", "costmodel:default_feature_schema", "costmodel:FEATURE_GROUPS"]),
            (5, "执行 leakage audit", ["costmodel:leakage_audit", "costmodel:LEAKAGE_FORBIDDEN"]),
            (6, "构造 labels/ties", ["costmodel:build_labels", "costmodel:DatasetUnit.ci95"]),
            (7, "冻结 train/validation/final-test split", ["costmodel:SplitManifest", "costmodel:build_split"]),
            (8, "建立 simple heuristic", ["costmodel:heuristic_pick", "costmodel:STRONG_BASELINES"]),
            (9, "实现 baseline evaluators", ["costmodel:evaluate_baselines", "costmodel:evaluate_baseline"]),
            (10, "选择最小模型族", ["costmodel:model_families", "costmodel:LinearRanker", "costmodel:PairwiseRanker"]),
            (11, "建立 train-only preprocessing", ["costmodel:FeatureSchema.missing_critical", "costmodel:ConstantPredictor"]),
            (12, "训练 regression/ranking candidates", ["costmodel:Predictor.fit", "costmodel:LinearRanker.fit"]),
            (13, "在 validation 选模型", ["costmodel:RegretReport.summary", "costmodel:ranking_metrics"]),
            (14, "校准 confidence", ["costmodel:ConfidenceModel", "costmodel:ConfidenceModel.risk_coverage"]),
            (15, "冻结 final model/policy", ["costmodel:ModelArtifactGuard", "costmodel:select_with_policy"]),
            (16, "运行 final shape holdout", ["costmodel:RegretReport", "costmodel:relative_regret"]),
            (17, "运行 regime/workload holdout", ["costmodel:RegretReport.summary", "costmodel:select_with_policy"]),
            (18, "运行 graph/pattern holdout", ["costmodel:methodology_verdict", "pattern_library:frozen_contracts"]),
            (19, "运行 arch holdout", ["targets:target_field_diff", "costmodel:methodology_verdict"]),
            (20, "计算 ranking metrics", ["costmodel:ranking_metrics", "costmodel:speedup_capture"]),
            (21, "计算 oracle regret", ["costmodel:RegretReport.summary", "autotune:per_shape_oracle"]),
            (22, "计算 speedup capture", ["costmodel:speedup_capture", "records:amortization_row"]),
            (23, "评估 confidence risk-coverage", ["costmodel:ConfidenceModel.risk_coverage", "costmodel:_is_risk_monotone"]),
            (24, "测试 OOD/缺失 feature", ["costmodel:select_with_policy", "costmodel:FeatureSchema.missing_critical"]),
            (25, "注入非法低成本候选", ["costmodel:illegal_candidate_injection", "lowering:CandidateEvaluation"]),
            (26, "测试模型 artifact 损坏/版本错", ["costmodel:ModelArtifactGuard.check", "cache:EntryStore.read"]),
            (27, "测在线选择开销", ["costmodel:selection_overhead", "records:COMPILER_COST_KEYS"]),
            (28, "测 compile 前/后两级模型", ["costmodel:FeatureField", "lowering:compile_breakdown"]),
            (29, "做 feature ablation", ["codegen:evaluate_ablation", "costmodel:FEATURE_GROUPS"]),
            (30, "做误差案例剖析", ["costmodel:error_case_study", "codegen:MechanismHypothesis"]),
            (31, "新进程 replay selected decisions", ["lowering:DispatchRow", "lowering:rebuild_plan"]),
            (32, "形成部署/不部署裁决", ["costmodel:DeploymentCriteria", "costmodel:methodology_verdict", "experiment:RunDirectory.write_verdict"]),
        ]
    ),
)

E11_08 = ExperimentMapping(
    experiment_id="E11-08",
    title="编译制品 Cache、跨进程命中、版本失效、并发发布与坏缓存安全",
    claim_boundary="本地/声明环境的 cache 生命周期安全；不等同供应链签名或跨版本长期兼容",
    steps=_steps(
        [
            (1, "冻结正式 compile identity", ["identity:compile_identity", "cache:CacheKeySpec"]),
            (2, "枚举 cache 层", ["cache:CACHE_LAYERS", "cache:layer_dependency_map"]),
            (3, "定义 HQSB cache key schema", ["cache:default_key_spec", "cache:KEY_FIELD_REASONS", "cache:evaluate_key_vectors"]),
            (4, "定义 entry manifest/state machine", ["cache:EntryManifest", "cache:ENTRY_STATES", "cache:ALLOWED_STATE_TRANSITIONS"]),
            (5, "实现安全 reader", ["cache:EntryStore.read", "cache:ReadResult", "cache:READ_REJECT_REASONS"]),
            (6, "实现事务 writer", ["cache:EntryStore.publish", "cache:EntryStore.cleanup_temp"]),
            (7, "实现 cache telemetry", ["cache:CacheEvent", "cache:CacheTelemetry"]),
            (8, "定义 cache reset profiles", ["cache:timing_boundaries", "cache:TIMING_BOUNDARIES"]),
            (9, "运行 C0 完全冷编译", ["cache:timing_boundaries", "lowering:compile_breakdown"]),
            (10, "运行 C1 同进程内存命中", ["cache:CacheTelemetry", "cache:cache_metrics"]),
            (11, "运行 C2 新进程磁盘命中", ["cache:EntryStore.read", "cache:cache_metrics"]),
            (12, "运行 C3 prebuilt/package load", ["cache:EntryStore.read", "cache:EntryManifest"]),
            (13, "运行 guard-domain reuse", ["cache:EntryStore.read", "guards:VariantRegistry.lookup"]),
            (14, "运行 non-semantic change 测试", ["cache:key_stability", "cache:NON_KEY_FIELDS"]),
            (15, "运行 model/constant/schema 变化", ["cache:evaluate_invalidation", "cache:INVALIDATION_MATRIX"]),
            (16, "运行 pass/rewrite 变化", ["rewrite:pass_identity", "cache:evaluate_invalidation"]),
            (17, "运行 kernel/ABI 变化", ["lowering:LoweringEntry", "cache:evaluate_invalidation"]),
            (18, "运行 compiler/toolchain 变化", ["identity:require_frozen_versions", "cache:evaluate_invalidation"]),
            (19, "运行 target arch/features 变化", ["targets:TargetSnapshot", "cache:evaluate_invalidation"]),
            (20, "运行 autotune/cost-model 变化", ["autotune:tuning_database_record", "costmodel:ModelArtifactGuard"]),
            (21, "注入 metadata truncation/unknown schema", ["cache:inject_corruption", "cache:EntryStore.read"]),
            (22, "注入 payload truncation/bit flip", ["cache:inject_corruption", "cache:CORRUPTION_CASES"]),
            (23, "注入 metadata/payload swap", ["cache:inject_corruption", "cache:EntryManifest.validate"]),
            (24, "注入 incomplete commit/temp residue", ["cache:inject_corruption", "cache:EntryStore.cleanup_temp"]),
            (25, "测试 permission/disk-full", ["cache:EntryStore.read", "cache:ReadResult"]),
            (26, "测试同 key 并发 writer", ["cache:reader_writer_plan", "cache:evaluate_concurrency"]),
            (27, "测试 reader-writer 并发", ["cache:evaluate_concurrency", "cache:reader_writer_plan"]),
            (28, "测试 key omission detector", ["cache:key_omission_detector", "cache:CacheKeySpec.compute"]),
            (29, "测 cache size/eviction", ["cache:EvictionPolicy", "cache:EvictionPolicy.plan"]),
            (30, "计算 cold/warm/steady 收益", ["cache:cache_metrics", "records:amortization_row"]),
            (31, "独立宿主/容器兼容验证", ["cache:cache_policy_document", "cache:layer_dependency_map"]),
            (32, "冻结 cache policy 与 verdict", ["cache:cache_policy_document", "cache:cache_policy_digest", "experiment:RunDirectory.write_verdict"]),
        ]
    ),
)

E11_09 = ExperimentMapping(
    experiment_id="E11-09",
    title="TVM Relax/TensorIR 或 MLIR 的可迁移 Graph→Schedule→Target Lowering",
    claim_boundary="概念可迁移验证；绝不因完成一个 runnable op 就声称具备通用 MLIR/TVM 编译器",
    steps=_steps(
        [
            (1, "冻结 primary stack 与版本", ["portable:StackSelection", "portable:STACKS"]),
            (2, "冻结 semantic task", ["pattern_library:residual_add_rmsnorm_contract", "lowering:CorrectnessEvidence"]),
            (3, "定义 scope ceiling", ["portable:SCOPE_CEILING", "portable:StackSelection.validate"]),
            (4, "编写语义映射表", ["portable:semantic_mapping_table", "portable:MAPPING_ROWS"]),
            (5, "建立 target capability", ["targets:CapabilityRequirement", "portable:semantic_mapping_table"]),
            (6, "构造 high-level IR", ["portable:semantic_mapping_table", "identity:content_hash"]),
            (7, "运行 parser/verifier", ["portable:LegalizationTarget", "portable:LegalizationTarget.validate"]),
            (8, "实现/应用 pattern rewrite", ["pattern_library:PatternSignature", "portable:semantic_mapping_table"]),
            (9, "定义 legalization target", ["portable:LegalizationTarget", "portable:LEGAL_STATUSES"]),
            (10, "执行 analysis-only legality", ["portable:analysis_only_legality", "portable:LEGALITY_MODES"]),
            (11, "执行 legalization", ["portable:LegalizationTarget.classify", "portable:UNMAPPED_POLICIES"]),
            (12, "导出 unscheduled tensor/loop IR", ["portable:LoopIRSpec", "portable:verify_loop_ir"]),
            (13, "运行 loop-IR verifier/structural checks", ["portable:verify_loop_ir", "portable:LoopIRSpec.validate"]),
            (14, "定义 manual schedule", ["portable:ScheduleStep", "portable:SCHEDULE_STEPS"]),
            (15, "应用 schedule 并保存 trace", ["portable:ScheduleTrace", "portable:ScheduleStep.validate"]),
            (16, "验证 schedule 后 IR", ["portable:verify_loop_ir", "portable:ScheduleTrace.validate"]),
            (17, "可选 search-based schedule", ["portable:ScheduleTrace", "autotune:budget_ladder"]),
            (18, "执行 target lowering pipeline", ["portable:PassTraceEntry", "portable:pass_pipeline_trace"]),
            (19, "导出 target IR/generated code", ["portable:pass_pipeline_trace", "codegen:GeneratedSourceArtifact"]),
            (20, "构建 binary/runtime module", ["identity:ArtifactIdentity", "portable:pass_pipeline_trace"]),
            (21, "建立 PyTorch bridge", ["portable:BridgeContract", "portable:BRIDGE_KINDS"]),
            (22, "验证 bridge copy/sync", ["portable:bridge_audit_plan", "portable:bridge_overhead"]),
            (23, "运行 operator correctness", ["lowering:CorrectnessEvidence", "portable:verify_loop_ir"]),
            (24, "运行 negative capability", ["portable:negative_capability_case", "portable:analysis_only_legality"]),
            (25, "运行 Qwen block/subgraph integration", ["lowering:CORRECTNESS_ORDER", "portable:BridgeContract.validate"]),
            (26, "测 compile breakdown", ["lowering:compile_breakdown", "records:COMPILER_COST_KEYS"]),
            (27, "测 runtime performance", ["codegen:ProfilePlan", "portable:bridge_overhead"]),
            (28, "做 target profile", ["codegen:parse_resource_usage", "codegen:occupancy_limits"]),
            (29, "独立重放 pass/schedule", ["portable:replay_schedule", "portable:pass_pipeline_trace"]),
            (30, "统计开发工程成本", ["portable:development_cost_rubric", "portable:COST_RUBRIC_ITEMS"]),
            (31, "形成概念/角色对照", ["portable:role_comparison", "portable:concept_summary_rows", "portable:ROLE_DIMENSIONS"]),
            (32, "做采用裁决", ["portable:adoption_decision", "portable:ADOPTION_OPTIONS", "experiment:RunDirectory.write_verdict"]),
        ]
    ),
)

E11_10 = ExperimentMapping(
    experiment_id="E11-10",
    title="AI/Agent 生成 Kernel 候选的零信任质量门、对抗测试与人工审核",
    level="P1",
    claim_boundary="只有方法学 PASS 无 admitted candidate 时，只能声称协议建立与错误候选被拦截",
    steps=_steps(
        [
            (1, "冻结是否执行与 claim scope", ["aigate:claim_boundary", "experiment:Preregistration"]),
            (2, "冻结 task package", ["aigate:TaskPackage", "aigate:TaskPackage.digest"]),
            (3, "冻结生成配置", ["aigate:GenerationConfig", "aigate:GenerationConfig.validate"]),
            (4, "定义 provenance schema", ["aigate:ProvenanceRecord", "aigate:PROVENANCE_FIELDS"]),
            (5, "建立隔离 evaluator", ["aigate:SandboxPolicy", "aigate:SandboxPolicy.validate"]),
            (6, "锁定可信 harness", ["aigate:HarnessLock", "aigate:HarnessLock.validate"]),
            (7, "建立 wrong-candidate corpus", ["aigate:WrongCorpusEntry", "aigate:wrong_corpus_template", "aigate:EXPECTED_GATE_BY_CLASS"]),
            (8, "建立 correct controls", ["aigate:CANDIDATE_CLASSES", "aigate:wrong_corpus_template"]),
            (9, "实现 G0 provenance/immutability", ["aigate:gate_g0_provenance", "aigate:GATE_ORDER"]),
            (10, "实现 G1 static policy", ["aigate:gate_g1_static_policy", "aigate:BANNED_IMPORTS", "aigate:BANNED_PATTERNS"]),
            (11, "实现 G2 isolated compile", ["aigate:gate_g2_isolated_compile", "aigate:SandboxPolicy"]),
            (12, "验证 compile artifact identity", ["identity:ArtifactIdentity", "cache:EntryManifest"]),
            (13, "实现 G3 sanitizer/memory gate", ["aigate:gate_g3_sanitizer_memory", "aigate:GateResult"]),
            (14, "实现 effect/state observers", ["ir:EffectSet", "aigate:gate_g6_state_concurrency"]),
            (15, "实现 G4 public correctness", ["aigate:gate_g4_public_correctness", "lowering:CorrectnessEvidence"]),
            (16, "设计隐藏 shape 分布", ["aigate:HiddenShapeDistribution", "aigate:HiddenShapeDistribution.validate"]),
            (17, "设计隐藏 value 分布", ["aigate:HiddenValueDistribution", "aigate:HiddenValueDistribution.validate"]),
            (18, "设计 metamorphic tests", ["aigate:metamorphic_tests", "aigate:gate_g5_hidden_correctness"]),
            (19, "运行 G5 hidden correctness", ["aigate:gate_g5_hidden_correctness", "aigate:GateResult"]),
            (20, "运行 G6 repeated/nondeterminism", ["aigate:gate_g6_state_concurrency", "costmodel:DatasetUnit.ci95"]),
            (21, "运行 dynamic/guard/fallback", ["guards:VariantRegistry", "lowering:preflight_checks"]),
            (22, "实现 G7 measurement integrity", ["aigate:gate_g7_measurement_integrity", "aigate:HarnessLock"]),
            (23, "运行强 baseline", ["costmodel:STRONG_BASELINES", "costmodel:evaluate_baselines"]),
            (24, "运行多 shape 性能", ["autotune:measurement_schedule", "autotune:select_winners"]),
            (25, "计算 correct-and-fast 指标", ["aigate:fast_p", "aigate:resource_pareto"]),
            (26, "检查资源交换", ["aigate:resource_pareto", "codegen:occupancy_limits"]),
            (27, "运行 E11-03 integration", ["lowering:LoweringRegistry", "aigate:gate_g9_compiler_integration"]),
            (28, "运行 cache/corruption/failure", ["cache:EntryStore.publish", "cache:inject_corruption"]),
            (29, "执行迭代反馈对照", ["aigate:iteration_lineage", "aigate:GenerationConfig"]),
            (30, "独立人类 review", ["aigate:gate_g10_review", "aigate:human_review_checklist"]),
            (31, "执行 admission negative tests", ["aigate:validate_admission", "aigate:ADMISSION_REQUIRED_FIELDS"]),
            (32, "形成候选/admission verdict", ["aigate:candidate_verdict", "aigate:adversarial_detection_matrix", "aigate:claim_boundary"]),
        ]
    ),
)

EXPERIMENTS: Tuple[ExperimentMapping, ...] = (
    E11_01,
    E11_02,
    E11_03,
    E11_04,
    E11_05,
    E11_06,
    E11_07,
    E11_08,
    E11_09,
    E11_10,
)


def mapping_for(experiment_id: str) -> ExperimentMapping:
    for mapping in EXPERIMENTS:
        if mapping.experiment_id == experiment_id:
            return mapping
    raise ConfigError(f"unknown experiment id {experiment_id!r}")


def _resolve(symbol: str) -> Optional[str]:
    """Resolve ``module:symbol`` (or ``module:Cls.method``) to an import path."""
    if ":" not in symbol:
        return f"malformed reference (missing ':'): {symbol}"
    prefix, _, attribute = symbol.partition(":")
    module_name = MODULE_BY_PREFIX.get(prefix)
    if module_name is None:
        return f"unknown module prefix {prefix!r}"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import failure is the error report
        return f"cannot import {module_name}: {type(exc).__name__}: {exc}"
    target: Any = module
    for part in attribute.split("."):
        if not hasattr(target, part):
            return f"{module_name} has no attribute {part!r} (from {symbol})"
        target = getattr(target, part)
    return None


def resolve_interfaces() -> Dict[str, Any]:
    """Import-check every mapped interface; failures are reported, never hidden."""
    failures: List[Dict[str, str]] = []
    steps = 0
    references = 0
    unique: set = set()
    for mapping in EXPERIMENTS:
        for step in mapping.steps:
            steps += 1
            for symbol in step.interfaces:
                references += 1
                unique.add(symbol)
                problem = _resolve(symbol)
                if problem:
                    failures.append(
                        {
                            "experiment_id": mapping.experiment_id,
                            "step": step.index,
                            "symbol": symbol,
                            "problem": problem,
                        }
                    )
    return {
        "experiments": len(EXPERIMENTS),
        "steps": steps,
        "interfaces": len(unique),
        "references": references,
        "failures": failures,
        "ok": not failures,
    }


def mapping_table_markdown() -> str:
    lines = [
        "| 实验 | 级别 | 步骤数 | 驱动入口 | 覆盖能力（示例） |",
        "|---|---|---|---|---|",
    ]
    for mapping in EXPERIMENTS:
        sample = ", ".join(mapping.steps[0].interfaces[:2])
        lines.append(
            f"| {mapping.experiment_id} | {mapping.level} | {len(mapping.steps)} | "
            f"`{mapping.driver}` | {sample} … |"
        )
    return "\n".join(lines)


def step_table_for(experiment_id: str) -> str:
    mapping = mapping_for(experiment_id)
    lines = [
        f"### {mapping.experiment_id} {mapping.title}",
        "",
        f"claim boundary: {mapping.claim_boundary}",
        "",
        "| # | 步骤 | 代码接口 |",
        "|---|---|---|",
    ]
    for step in mapping.steps:
        lines.append(f"| {step.index} | {step.title} | {', '.join(step.interfaces)} |")
    return "\n".join(lines)


def total_steps() -> int:
    return sum(len(mapping.steps) for mapping in EXPERIMENTS)
