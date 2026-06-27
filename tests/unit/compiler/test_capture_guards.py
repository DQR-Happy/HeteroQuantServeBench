"""Coverage for the capture census (E11-01) and guard/variant safety (E11-05)."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.compiler import capture as cap
from hqsb.compiler import guards as gd
from hqsb.compiler import records as rec


@pytest.mark.unit
class TestCaptureMatrix:
    def test_matrix_covers_regions_modes_and_negative_cells(self) -> None:
        rows = cap.capture_matrix()
        assert {row.region for row in rows} >= {"block", "prefill", "decode_step"}
        assert {row.capture_mode for row in rows} == set(cap.CAPTURE_MODES)
        assert any(row.layout_case == "unsupported_layout" for row in rows)

    def test_operator_only_cells_are_rejected_as_p0(self) -> None:
        request = cap.CaptureRequest(
            request_id="r",
            region="operator",
            capture_mode=cap.MODE_DYNAMO_DEBUG,
            workload="tiny",
        )
        assert any("operator" in problem for problem in request.validate())

    def test_unknown_axis_values_are_rejected(self) -> None:
        request = cap.CaptureRequest(
            request_id="r", region="block", capture_mode="magic", workload="tiny"
        )
        assert any("capture_mode" in problem for problem in request.validate())


@pytest.mark.unit
class TestBreakAudit:
    def _break(self, **overrides: object) -> cap.GraphBreak:
        payload = dict(
            break_id="b1",
            frame_id="f1",
            reason_code="CAPTURE_GRAPH_BREAK",
            native_reason="unsupported call",
            source_file="model.py",
            source_line=42,
            fallback_action="eager_island",
            graph_before_id="g1",
        )
        payload.update(overrides)
        return cap.GraphBreak(**payload)  # type: ignore[arg-type]

    def test_located_break_passes(self) -> None:
        assert cap.BreakAudit((self._break(),)).summary()["passes_localisation"] is True

    def test_unknown_break_is_not_acceptable(self) -> None:
        audit = cap.BreakAudit((self._break(reason_code="UNKNOWN_BREAK"),))
        assert audit.summary()["passes_localisation"] is False

    def test_missing_source_line_is_flagged(self) -> None:
        audit = cap.BreakAudit((self._break(source_line=0),))
        assert audit.summary()["unlocated"] == ["b1"]

    def test_reason_histogram_is_sorted(self) -> None:
        audit = cap.BreakAudit((self._break(), self._break(break_id="b2", reason_code="MISSING_FAKE_META")))
        assert list(audit.by_reason()) == ["CAPTURE_GRAPH_BREAK", "MISSING_FAKE_META"]

    def test_negative_fixtures_are_structured(self) -> None:
        assert cap.missing_fake_negative(op_name="hqsb::x").validate() == []
        assert cap.data_dependent_negative(source_file="m.py", source_line=3).validate() == []


@pytest.mark.unit
class TestMetadata:
    def test_source_lineage_needs_a_missing_reason_when_incomplete(self) -> None:
        lineage = cap.SourceLineage(node_id="n1", native_op="aten.add")
        assert any("missing_reason" in problem for problem in lineage.validate())
        ok = cap.SourceLineage(
            node_id="n1", native_op="aten.add", missing_reason="not_exposed_by_framework"
        )
        assert ok.validate() == []

    def test_tensor_metadata_completeness(self) -> None:
        tensor = cap.TensorMetadataRecord(
            tensor_id="t1",
            dtype="fp16",
            shape=(1, 8),
            symbolic_expression=("B", "H"),
            symbolic_range=((1, 8), (1, 4096)),
            stride=(8, 1),
            layout="strided",
            device="cuda:0",
            runtime_sampled=True,
        )
        assert all(tensor.completeness().values())

    def test_fake_runtime_comparison(self) -> None:
        fake = cap.TensorMetadataRecord(tensor_id="t", dtype="fp16", shape=(1, 8), layout="strided", device="cuda:0")
        runtime = cap.TensorMetadataRecord(tensor_id="t", dtype="fp16", shape=(1, 16), layout="strided", device="cuda:0")
        report = cap.compare_fake_runtime(fake, runtime)
        assert report["ok"] is False
        assert report["mismatches"][0]["field"] == "shape"

    def test_metadata_completeness_rates_and_thresholds(self) -> None:
        report = cap.MetadataCompleteness.from_records(
            lineages=[cap.SourceLineage(node_id="n", native_op="add", file="f", line=1)],
            tensors=[cap.TensorMetadataRecord(tensor_id="t", dtype="fp16", shape=(1,), layout="strided")],
            effect_states=["PROVED_SAFE", "UNKNOWN_UNSAFE"],
        )
        rates = report.rates()
        assert rates["source"] == 1.0
        assert rates["effect"] == 0.5
        assert report.meets({"effect": 0.75})["ok"] is False
        assert report.meets({"effect": 0.5})["ok"] is True


@pytest.mark.unit
class TestCoverage:
    def test_unavailable_metric_is_not_substituted(self) -> None:
        report = cap.CoverageReport(
            captured_ops=10,
            observed_ops=10,
            unavailable={"weighted_time_coverage": "profiler correlation unavailable"},
        )
        assert report.metrics()["op_count_coverage"] == 1.0
        assert report.metrics()["weighted_time_coverage"] is None
        guard = cap.coverage_claim_guard(report.metrics(), claim="time")
        assert guard["allowed"] is False

    def test_phase_coverage_is_per_phase(self) -> None:
        report = cap.CoverageReport(
            captured_phase_time={"prefill": 0.4, "decode": 0.6},
            phase_time={"prefill": 1.0, "decode": 1.0},
        )
        assert report.phase_coverage() == {"decode": 0.6, "prefill": 0.4}

    def test_op_count_cannot_stand_in_for_time_claims(self) -> None:
        report = cap.CoverageReport(captured_ops=5, observed_ops=10)
        assert cap.coverage_claim_guard(report.metrics(), claim="op_count")["allowed"] is True
        assert cap.coverage_claim_guard(report.metrics(), claim="hotspot")["allowed"] is False

    def test_unknown_claim_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            cap.coverage_claim_guard({}, claim="vibes")


@pytest.mark.unit
class TestCensusAndBackend:
    def test_census_requires_a_measured_denominator(self) -> None:
        census = cap.EagerCensus()
        census.add(
            cap.EagerCensusEntry(
                module_path="block.0",
                op_name="aten.add",
                dtype="fp16",
                shape=(1, 8),
                stride=(8, 1),
                device="cuda:0",
                time_us=12.0,
            )
        )
        assert census.ready is False
        with pytest.raises(ConfigError):
            census.to_coverage(captured_ops=1, captured_time_s=0.0)
        census.measurement_method = "torch.profiler"
        assert census.ready is True
        coverage = census.to_coverage(captured_ops=1, captured_time_s=1e-6)
        assert coverage.observed_ops == 1

    def test_hotspot_rows_are_ranked(self) -> None:
        census = cap.EagerCensus(measurement_method="profiler")
        for index, time_us in enumerate((5.0, 50.0, 1.0)):
            census.add(
                cap.EagerCensusEntry(
                    module_path=f"m{index}",
                    op_name="aten.mul",
                    dtype="fp16",
                    shape=(1, 8),
                    stride=(8, 1),
                    device="cuda:0",
                    time_us=time_us,
                )
            )
        assert census.hotspot_rows(top_k=1)[0]["time_us"] == 50.0

    def test_debug_backend_returns_forward_and_flags_itself(self) -> None:
        backend = cap.DebugBackend(capture_fn=lambda gm, inputs: {"graph_id": "g1"})
        graph_module = type("GM", (), {"forward": lambda self, *a: 1.0})()
        callable_ = backend(graph_module, [1, 2])
        assert callable_() == 1.0
        result = backend.result("g1")
        assert result.is_debug_backend is True
        assert result.returned_callable == "gm.forward"
        assert result.rewrites_applied == 0 and result.lowerings_selected == 0


@pytest.mark.unit
class TestTraceAndRepeatability:
    def test_ordered_trace_requires_the_frozen_sequence(self) -> None:
        steps = [
            cap.ShapeTraceStep(phase=phase, shapes={"B": 1}) for phase in cap.TRACE_PHASES
        ]
        assert len(cap.ordered_shape_trace(steps)) == len(cap.TRACE_PHASES)
        with pytest.raises(ConfigError):
            cap.ordered_shape_trace(list(reversed(steps)))

    def test_native_trace_plan_requires_recent_torch(self) -> None:
        old = cap.NativeTracePlan.for_torch("1.13.0")
        assert old.gaps
        new = cap.NativeTracePlan.for_torch("2.8.0")
        assert new.commands and not new.gaps

    def test_native_join_reports_unmapped_rows(self) -> None:
        report = cap.join_native_and_hqsb(
            [{"correlation_id": "a"}], [{"correlation_id": "a"}, {"correlation_id": "b"}]
        )
        assert report["ok"] is False
        assert report["unmapped_hqsb"] == ["b"]

    def test_repeatability_requires_stable_canonical_hash(self) -> None:
        report = cap.repeatability_check(
            [
                {"canonical_hash": "x", "raw_hash": "1", "breaks": [], "guards": []},
                {"canonical_hash": "x", "raw_hash": "2", "breaks": [], "guards": []},
            ]
        )
        assert report["ok"] is True
        assert report["raw_hashes_differ"] is True

    def test_hero_manifest_requires_sites_constraints_and_versions(self) -> None:
        manifest = cap.HeroGraphManifest(
            graph_id="g",
            source_graph_id="g",
            canonical_hash="h",
            capture_mode=cap.MODE_DYNAMO_DEBUG,
            region="block",
            workload="short",
            dtype="fp16",
        )
        problems = manifest.validate()
        assert any("pattern site" in problem for problem in problems)
        assert any("constraints" in problem for problem in problems)
        filled = cap.HeroGraphManifest(
            graph_id="g",
            source_graph_id="g",
            canonical_hash="h",
            capture_mode=cap.MODE_DYNAMO_DEBUG,
            region="block",
            workload="short",
            dtype="fp16",
            constraints=({"constraint_id": "c"},),
            pattern_sites=({"site": "s"},),
            compiler_versions={"torch": "2.8.0"},
            capture_scope_claim="prefill capture only",
        )
        assert filled.validate() == []
        assert filled.digest()


@pytest.mark.unit
class TestGuardTaxonomy:
    def test_guard_spec_validation(self) -> None:
        good = gd.range_guard(guard_id="B", symbol="B", lower=1, upper=8, source="policy.yaml")
        assert good.validate() == []
        bad_category = gd.GuardSpec(guard_id="x", category="vibes", expression="x", source="s")
        assert any("category" in problem for problem in bad_category.validate())

    def test_performance_guard_cannot_be_semantic(self) -> None:
        guard = gd.GuardSpec(
            guard_id="p",
            category="performance",
            expression="M<=16",
            source="heuristic",
            semantic_required=True,
            evaluator=lambda inputs: True,
        )
        assert any("performance" in problem for problem in guard.validate())

    def test_semantic_guard_cannot_be_dropped_as_performance_only(self) -> None:
        guard = gd.GuardSpec(
            guard_id="s",
            category="semantic_input",
            expression="rank==2",
            source="contract",
            semantic_required=False,
            evaluator=lambda inputs: True,
        )
        assert any("semantic" in problem for problem in guard.validate())

    def test_unevaluable_guard_refuses_to_return_true(self) -> None:
        guard = gd.GuardSpec(guard_id="g", category="shape_range", expression="B<=8", source="s")
        with pytest.raises(ConfigError):
            guard.evaluate({"B": 4})


@pytest.mark.unit
class TestVariantRegistry:
    def _variant(self, variant_id: str = "v1", priority: int = 1) -> gd.Variant:
        return gd.Variant(
            variant_id=variant_id,
            semantic_identity="sem",
            compile_identity="cid",
            target_id="cpu",
            artifact_id="art",
            guards=(
                gd.range_guard(guard_id=f"{variant_id}_b", symbol="B", lower=1, upper=8, source="s"),
                gd.equality_guard(guard_id=f"{variant_id}_d", name="dtype", expected="fp16", source="s"),
            ),
            priority=priority,
            created_reason="test",
        )

    def test_hit_requires_all_guards_true(self) -> None:
        registry = gd.VariantRegistry()
        registry.add(self._variant())
        assert registry.lookup(semantic_identity="sem", inputs={"B": 4, "dtype": "fp16"}).outcome == "hit"
        miss = registry.lookup(semantic_identity="sem", inputs={"B": 99, "dtype": "fp16"})
        assert miss.outcome == "miss"
        assert miss.failed_guards == ("v1_b",)

    def test_artifact_incompatibility_is_a_miss(self) -> None:
        registry = gd.VariantRegistry()
        registry.add(self._variant())
        result = registry.lookup(
            semantic_identity="sem",
            inputs={"B": 4, "dtype": "fp16"},
            artifact_compatible=lambda artifact_id: False,
        )
        assert result.outcome == "miss"
        assert any("artifact_incompatible" in item for item in result.failed_guards)

    def test_guard_false_reuse_is_possible_only_explicitly(self) -> None:
        registry = gd.VariantRegistry()
        registry.add(self._variant())
        forced = registry.lookup(
            semantic_identity="sem", inputs={"B": 99, "dtype": "fp16"}, accept_guard_false=True
        )
        assert forced.outcome == "hit"  # only reachable through the negative-test switch

    def test_duplicate_variant_and_empty_domain_are_rejected(self) -> None:
        registry = gd.VariantRegistry()
        registry.add(self._variant())
        with pytest.raises(ConfigError):
            registry.add(self._variant())
        with pytest.raises(ConfigError):
            registry.add(
                gd.Variant(
                    variant_id="v2",
                    semantic_identity="sem",
                    compile_identity="cid",
                    target_id="cpu",
                    artifact_id="art",
                    guards=(),
                    created_reason="r",
                )
            )

    def test_domain_coverage_reports_holes_and_priority_conflicts(self) -> None:
        registry_rows = [self._variant("v1", priority=1), self._variant("v2", priority=1)]
        report = gd.domain_coverage(
            registry_rows, [{"B": 4, "dtype": "fp16"}, {"B": 99, "dtype": "fp16"}]
        )
        assert report["coverage"] == 0.5
        assert report["holes"] == [1]
        assert report["overlaps_without_priority_order"]

    def test_domain_coverage_accepts_distinct_priorities(self) -> None:
        rows = [self._variant("v1", priority=1), self._variant("v2", priority=2)]
        report = gd.domain_coverage(rows, [{"B": 4, "dtype": "fp16"}])
        assert report["ok"] is True


@pytest.mark.unit
class TestCompileAccounting:
    def test_unexplained_compiles_fail_the_audit(self) -> None:
        explained = gd.CompileEvent(
            compile_id="c1", variant_id="v1", trigger="guard_failure", triggering_guard_ids=("g",), domain="B<=8"
        )
        unexplained = gd.CompileEvent(compile_id="c2", variant_id="v2", trigger="unexplained")
        report = gd.compile_explainability([explained, unexplained], calls=10)
        assert report["unexplained"] == ["c2"]
        assert report["ok"] is False

    def test_wrong_reuse_is_detected(self) -> None:
        events = [
            rec.GuardEventRecord(
                run_id="r",
                frame_id="f",
                variant_id="v1",
                guard_id="g",
                expression="B<=8",
                source="policy",
                category="shape_range",
                actual_values={"B": 16},
                outcome=False,
                action="reuse",
            )
        ]
        audit = gd.wrong_reuse_audit([{"variant_id": "v1"}], events)
        assert audit["wrong_reuse_count"] == 1
        assert audit["ok"] is False

    def test_guard_event_validation_rejects_false_reuse(self) -> None:
        event = rec.GuardEventRecord(
            run_id="r",
            frame_id="f",
            variant_id="v",
            guard_id="g",
            expression="B<=8",
            source="s",
            category="shape_range",
            actual_values={},
            outcome=False,
            action="reuse",
        )
        assert any("incompatible binary" in problem for problem in event.validate())

    def test_strategy_comparison_uses_total_cost(self) -> None:
        # S0 pays more compiles up front but wins over the 1000-request horizon
        rows = [
            gd.StrategyObservation("S0_static", 8, 8, 4.0, 3.0, 6.0, 0.0, 0.0, 1024, correctness_ok=True),
            gd.StrategyObservation("S2_declared_bounded", 2, 2, 1.0, 5.0, 11.0, 0.1, 0.0, 512, correctness_ok=True),
            gd.StrategyObservation("S4_manual_buckets", 3, 3, 1.5, 4.0, 10.5, 0.05, 0.0, 640, correctness_ok=True),
        ]
        report = gd.compare_strategies(rows, requests=100, horizon_requests=1000)
        assert report["fair_comparison"] is True
        assert report["ranking_by_total_cost"][0] == "S0_static"
        assert report["rows"][0]["total_cost_s"] < report["rows"][-1]["total_cost_s"]

    def test_strategy_rows_without_correctness_are_flagged(self) -> None:
        row = gd.StrategyObservation("S0_static", 1, 1, 1.0, 1.0, 1.0, 0.0, 0.0, 1)
        assert any("correctness" in problem for problem in row.validate())

    def test_variant_budget_fails_closed(self) -> None:
        budget = gd.VariantBudget(max_variants=2)
        assert budget.exceed_action(2)["action"] == "continue"
        exceeded = budget.exceed_action(3)
        assert exceeded["action"] == "fallback" and exceeded["must_report_compiled_false"] is True
        with pytest.raises(ConfigError):
            gd.VariantBudget(max_variants=2, behaviour_on_exceed="silent")

    def test_concurrency_evaluation(self) -> None:
        plan = gd.ConcurrencyPlan(threads=4, same_variant=True, expected_new_compiles=1)
        ok = gd.evaluate_concurrency(
            plan=plan, compiles=1, published_entries=1, half_baked_reads=0, correctness_ok=True
        )
        assert ok["ok"] is True
        storm = gd.evaluate_concurrency(
            plan=plan, compiles=3, published_entries=3, half_baked_reads=0, correctness_ok=True
        )
        assert storm["duplicate_storm"] is True

    def test_guard_removal_requires_a_replacement_domain(self) -> None:
        guards = (gd.range_guard(guard_id="B", symbol="B", lower=1, upper=8, source="s"),)
        blocked = gd.guard_minimality_review(guards, removed=["B"])
        assert blocked["ok"] is False
        allowed = gd.guard_minimality_review(
            guards, removed=["B"], replacement_domains={"B": "generic_variant"}
        )
        assert allowed["ok"] is True

    def test_policy_document_is_digestible_and_explicit(self) -> None:
        document = gd.dynamic_policy_document(
            default_strategy="S2_declared_bounded",
            supported_domain={"B": "1..8"},
            variant_budget=gd.VariantBudget(max_variants=8),
            compile_budget_s=60.0,
            monitoring_thresholds={"recompiles_per_1000": 5},
        )
        assert document["fallback"].startswith("reference lowering")
        assert gd.policy_digest(document) == gd.policy_digest(dict(reversed(list(document.items()))))
