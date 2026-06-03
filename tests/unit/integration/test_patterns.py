"""Pattern matching, predicates, rewrite and coverage tests (E06-04)."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError, SchemaError
from hqsb.integration import graph, patterns


def build_positive_graph() -> graph.Graph:
    return graph.from_node_sequence(
        [
            {"name": "x", "op": "graph_input"},
            {"name": "residual", "op": "graph_input"},
            {"name": "add", "op": "aten.add.Tensor", "args": ["x", "residual"]},
            {"name": "norm", "op": "hqsb.rms_norm", "args": ["add", "weight", 1e-6], "output": True},
        ],
        capture_mode=graph.CaptureMode.DYNAMO,
        ir_level=graph.IRLevel.DYNAMO_FX,
    )


def base_context(fixture: graph.Graph, **overrides) -> patterns.PatternContext:
    payload = {
        "graph": fixture,
        "declared_eps": 1e-6,
        "pattern_eps": 1e-6,
        "norm_axis": -1,
        "hidden_size": 64,
        "weight_shapes": {"weight": (64,)},
        "weight_dtype": "float16",
        "supported_versions": ("1.0.0",),
        "version": "1.0.0",
    }
    payload.update(overrides)
    return patterns.PatternContext(**payload)


@pytest.mark.unit
class TestStructureMatching:
    def test_positive_graph_yields_one_candidate(self):
        fixture = build_positive_graph()
        matches = patterns.find_matches(fixture, patterns.residual_add_rmsnorm_pattern().structures[0])
        assert len(matches) == 1
        assert matches[0].bindings == {"add": "add", "norm": "norm"}

    def test_missing_edge_yields_no_candidate(self):
        fixture = graph.from_node_sequence(
            [
                {"name": "x", "op": "graph_input"},
                {"name": "w", "op": "graph_input"},
                {"name": "add", "op": "aten.add.Tensor", "args": ["x", "x"]},
                {"name": "norm", "op": "hqsb.rms_norm", "args": ["w", "w"], "output": True},
            ],
            capture_mode=graph.CaptureMode.DYNAMO,
            ir_level=graph.IRLevel.DYNAMO_FX,
        )
        assert patterns.find_matches(fixture, patterns.residual_add_rmsnorm_pattern().structures[0]) == ()

    def test_ir_level_mismatch_is_not_scanned(self):
        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        scan = patterns.scan_graph(
            spec,
            graph.Graph(
                nodes=fixture.nodes,
                outputs=fixture.outputs,
                capture_mode=graph.CaptureMode.DYNAMO,
                ir_level=graph.IRLevel.INDUCTOR_PRE_FUSION,
            ),
            lambda _graph, _match: base_context(fixture),
        )
        assert scan == ()


@pytest.mark.unit
class TestPredicates:
    def test_all_predicates_pass_on_the_positive_case(self):
        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        decisions = patterns.scan_graph(spec, fixture, lambda _g, _m: base_context(fixture))
        assert len(decisions) == 1
        assert decisions[0].status == patterns.DecisionStatus.HIT
        assert decisions[0].reason == "PREDICATES_PASSED"
        assert all(item.passed for item in decisions[0].predicate_results)

    def test_predicates_are_all_evaluated_in_audit_mode(self):
        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        match = patterns.find_matches(fixture, spec.structures[0])[0]
        ctx = base_context(fixture, pattern_eps=1e-3, add_alpha=0.5)
        decision = patterns.evaluate_candidate(spec, match, ctx, short_circuit=False)
        assert decision.status == patterns.DecisionStatus.REJECT
        assert len(decision.predicate_results) == len(spec.predicates)

    def test_short_circuit_stops_at_the_first_failure(self):
        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        match = patterns.find_matches(fixture, spec.structures[0])[0]
        ctx = base_context(fixture, pattern_eps=1e-3)
        decision = patterns.evaluate_candidate(spec, match, ctx, short_circuit=True)
        assert len(decision.predicate_results) == 1
        assert decision.reason == patterns.RejectReason.SEMANTIC_EPS

    def test_extra_user_is_rejected_with_the_field(self):
        fixture = build_positive_graph()
        extra = fixture.with_node(
            graph.GraphNode(name="side", op="aten.clone", args=(graph.Ref("norm"),))
        )
        spec = patterns.residual_add_rmsnorm_pattern()
        match = patterns.find_matches(extra, spec.structures[0])[0]
        decision = patterns.evaluate_candidate(spec, match, base_context(extra))
        assert decision.status == patterns.DecisionStatus.REJECT
        assert decision.reason == patterns.RejectReason.EXTRA_USER

    def test_impure_node_inside_the_match_is_rejected(self):
        fixture = graph.from_node_sequence(
            [
                {"name": "x", "op": "graph_input"},
                {"name": "r", "op": "graph_input"},
                {"name": "add", "op": "aten.add.Tensor", "args": ["x", "r"], "is_impure": True},
                {"name": "norm", "op": "hqsb.rms_norm", "args": ["add", "w"], "output": True},
            ],
            capture_mode=graph.CaptureMode.DYNAMO,
            ir_level=graph.IRLevel.DYNAMO_FX,
        )
        spec = patterns.residual_add_rmsnorm_pattern()
        match = patterns.find_matches(fixture, spec.structures[0])[0]
        decision = patterns.evaluate_candidate(spec, match, base_context(fixture), short_circuit=False)
        assert decision.status == patterns.DecisionStatus.REJECT
        assert decision.reason == patterns.RejectReason.SIDE_EFFECT

    def test_norm_axis_mismatch_is_rejected(self):
        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        match = patterns.find_matches(fixture, spec.structures[0])[0]
        decision = patterns.evaluate_candidate(
            spec, match, base_context(fixture, norm_axis=0), short_circuit=False
        )
        assert decision.reason == patterns.RejectReason.SEMANTIC_NORM

    def test_weight_shape_mismatch_is_rejected(self):
        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        match = patterns.find_matches(fixture, spec.structures[0])[0]
        decision = patterns.evaluate_candidate(
            spec,
            match,
            base_context(fixture, weight_shapes={"weight": (32,)}),
            short_circuit=False,
        )
        assert decision.reason == patterns.RejectReason.SHAPE

    def test_capability_failure_is_rejected(self):
        from hqsb.integration import dispatch

        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        match = patterns.find_matches(fixture, spec.structures[0])[0]
        capability = dispatch.OperatorCapability(
            op="hqsb::fused_add_rms_norm", provider="cuda_shared_lib", dtypes=("float16",)
        )
        request = dispatch.CapabilityRequest(
            op="hqsb::fused_add_rms_norm", dtype="float64", rank=2, device="cuda"
        )
        decision = patterns.evaluate_candidate(
            spec, match, base_context(fixture, request=request, capability=capability),
            short_circuit=False,
        )
        assert decision.reason == patterns.RejectReason.BACKEND_CAPABILITY

    def test_unknown_version_is_rejected(self):
        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        match = patterns.find_matches(fixture, spec.structures[0])[0]
        decision = patterns.evaluate_candidate(
            spec, match, base_context(fixture, supported_versions=()), short_circuit=False
        )
        assert decision.reason == patterns.RejectReason.VERSION_UNSUPPORTED

    def test_unknown_reject_reason_is_refused(self):
        with pytest.raises(SchemaError):
            patterns.PredicateSpec("x", "semantic", "NOT_A_REASON", "f", "e")

    def test_predicate_without_implementation_is_refused(self):
        with pytest.raises(SchemaError):
            patterns.PatternSpec(
                pattern_id="hqsb.pattern.fake",
                version="1.0.0",
                ir_level=graph.IRLevel.DYNAMO_FX,
                capture_modes=("dynamo",),
                decomposition_set="none",
                semantics="none",
                structures=patterns.residual_add_rmsnorm_pattern().structures,
                predicates=(patterns.PredicateSpec("not_implemented", "semantic", "DTYPE", "f", "e"),),
            )


@pytest.mark.unit
class TestMutationTesting:
    def test_standard_mutations_are_all_rejected_with_the_expected_reason(self):
        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        match = patterns.find_matches(fixture, spec.structures[0])[0]
        outcomes = patterns.mutation_report(
            spec, match, base_context(fixture), patterns.standard_mutations()
        )
        assert outcomes
        for outcome in outcomes:
            assert outcome.ok, outcome.as_dict()


@pytest.mark.unit
class TestRewrite:
    def _rewrite(self):
        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        decision = patterns.scan_graph(spec, fixture, lambda _g, _m: base_context(fixture))[0]

        def replacement(_decision, _ctx):
            add = fixture.node("add")
            norm = fixture.node("norm")
            return graph.GraphNode(
                name="fused_norm",
                op="hqsb.fused_add_rms_norm",
                args=(*add.args, norm.args[1]),
            )

        return fixture, decision, patterns.apply_rewrite(fixture, decision, base_context(fixture), replacement)

    def test_rewrite_happens_on_a_copy(self):
        fixture, _decision, result = self._rewrite()
        assert fixture.has("add") and fixture.has("norm")
        assert not result.graph.has("add")
        assert result.graph.has("fused_norm")

    def test_rewrite_record_and_diff_are_persisted(self):
        _fixture, _decision, result = self._rewrite()
        payload = result.as_dict()
        assert payload["record"]["matched_nodes"] == ["add", "norm"]
        assert payload["record"]["new_op"] == "hqsb.fused_add_rms_norm"
        assert payload["record"]["preserved_metadata"]["capture_mode"] == "dynamo"
        assert payload["diff"]["node_delta"] == -1

    def test_rewrite_refuses_a_non_hit_decision(self):
        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        match = patterns.find_matches(fixture, spec.structures[0])[0]
        decision = patterns.evaluate_candidate(
            spec, match, base_context(fixture, pattern_eps=1e-3), short_circuit=False
        )
        with pytest.raises(ConfigError):
            patterns.apply_rewrite(
                fixture,
                decision,
                base_context(fixture),
                lambda _d, _c: graph.GraphNode(name="x", op="hqsb.fused_add_rms_norm"),
            )

    def test_rewrite_refuses_a_name_collision(self):
        fixture, decision, _result = self._rewrite()
        with pytest.raises(ConfigError):
            patterns.apply_rewrite(
                fixture,
                decision,
                base_context(fixture),
                lambda _d, _c: graph.GraphNode(name="add", op="hqsb.fused_add_rms_norm"),
            )


@pytest.mark.unit
class TestCoverage:
    def test_coverage_math(self):
        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        decisions = patterns.scan_graph(spec, fixture, lambda _g, _m: base_context(fixture))
        hit = decisions[0]
        rejected = patterns.evaluate_candidate(
            spec,
            hit.candidate,
            base_context(fixture, pattern_eps=1e-3),
            short_circuit=False,
        )
        runtime = (
            patterns.RuntimeUse(hit.candidate.case_id, calls=5, total_time_ms=10.0, workload="tiny"),
            patterns.RuntimeUse(rejected.candidate.case_id, calls=1, total_time_ms=2.0, workload="short"),
        )
        report = patterns.coverage_report([hit, rejected], runtime, workloads_total=2)
        payload = report.as_dict()
        assert payload["counts"]["eligible_nodes"] == 1
        assert payload["counts"]["rewritten_nodes"] == 1
        assert payload["call_coverage"] == 1.0
        assert payload["time_coverage"] == 1.0
        assert payload["model_coverage"] == 0.5
        assert payload["rejects"] == 1
        assert payload["false_positives"] == 0

    def test_false_positives_are_counted_from_ground_truth(self):
        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        decisions = patterns.scan_graph(spec, fixture, lambda _g, _m: base_context(fixture))
        truth = {decisions[0].candidate.case_id: False}
        report = patterns.coverage_report(decisions, (), ground_truth=truth)
        assert report.false_positives == 1
        assert report.precision == 0.0

    def test_coverage_without_runtime_data_reports_zero_calls(self):
        fixture = build_positive_graph()
        spec = patterns.residual_add_rmsnorm_pattern()
        decisions = patterns.scan_graph(spec, fixture, lambda _g, _m: base_context(fixture))
        report = patterns.coverage_report(decisions)
        assert report.call_coverage == 0.0


@pytest.mark.unit
class TestPatternRegistry:
    def test_frozen_registry_has_two_patterns(self):
        registry = patterns.frozen_registry()
        assert len(registry.all()) == 2
        assert registry.ir_levels == ("dynamo_fx",)

    def test_duplicate_registration_is_refused(self):
        registry = patterns.PatternRegistry()
        spec = patterns.residual_add_rmsnorm_pattern()
        registry.register(spec)
        with pytest.raises(SchemaError):
            registry.register(spec)

    def test_identical_replacement_is_idempotent(self):
        registry = patterns.PatternRegistry()
        registry.register(patterns.residual_add_rmsnorm_pattern())
        registry.register(patterns.residual_add_rmsnorm_pattern(), replace_existing=True)

    def test_unknown_lookup_is_refused(self):
        with pytest.raises(SchemaError):
            patterns.frozen_registry().get("hqsb.pattern.nope", "1.0.0")

    def test_pattern_signature_is_stable(self):
        first = patterns.residual_add_rmsnorm_pattern().signature
        second = patterns.residual_add_rmsnorm_pattern().signature
        assert first == second
