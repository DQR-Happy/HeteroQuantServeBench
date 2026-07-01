"""Unit tests for the E12-10 evidence lineage and regeneration machinery."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.evaluation import lineage as ln
from hqsb.evaluation.identity import EntityRef, canonical_hash, logical_uri

pytestmark = pytest.mark.unit


def _entity(entity_id: str, entity_type: str, byte_hash: str = "") -> EntityRef:
    return EntityRef(
        entity_id=entity_id,
        entity_type=entity_type,
        logical_uri=logical_uri("camp", entity_type, entity_id),
        byte_hash=byte_hash,
        canonical_hash=canonical_hash({"id": entity_id}),
        status="REGISTERED",
    )


class TestRegistries:
    def test_raw_evidence_is_append_only(self) -> None:
        registry = ln.EntityRegistry()
        registry.register(_entity("raw_1", "raw", byte_hash="a" * 64))
        with pytest.raises(ConfigError):
            registry.register(_entity("raw_1", "raw", byte_hash="b" * 64))

    def test_identical_re_registration_is_idempotent(self) -> None:
        registry = ln.EntityRegistry()
        registry.register(_entity("raw_1", "raw", byte_hash="a" * 64))
        registry.register(_entity("raw_1", "raw", byte_hash="a" * 64))
        assert len(registry.by_type("raw")) == 1

    def test_unknown_relation_is_rejected(self) -> None:
        assert ln.validate_relation("derived_from") == []
        assert ln.validate_relation("cousin_of")


class TestEvidenceDAG:
    def _dag(self) -> tuple[ln.EntityRegistry, ln.EvidenceDAG]:
        registry = ln.EntityRegistry()
        registry.register(_entity("raw_1", "raw", byte_hash="a" * 64))
        registry.register(_entity("norm_1", "derived"))
        edge = ln.EvidenceEdge(edge_id="e1", relation="derived_from", source_entity_id="raw_1", target_entity_id="norm_1")
        return registry, ln.EvidenceDAG(registry, [edge])

    def test_connected_dag_is_valid(self) -> None:
        _, dag = self._dag()
        assert dag.validate() == ()

    def test_dangling_reference_is_localised(self) -> None:
        registry = ln.EntityRegistry()
        registry.register(_entity("raw_1", "raw"))
        edge = ln.EvidenceEdge(edge_id="e1", relation="derived_from", source_entity_id="raw_1", target_entity_id="missing")
        dag = ln.EvidenceDAG(registry, [edge])
        problems = dag.validate()
        assert any(row["kind"] == "dangling" and "missing" in row["detail"] for row in problems)

    def test_cycle_is_detected(self) -> None:
        registry = ln.EntityRegistry()
        registry.register(_entity("a", "derived"))
        registry.register(_entity("b", "derived"))
        edges = [
            ln.EvidenceEdge(edge_id="e1", relation="derived_from", source_entity_id="a", target_entity_id="b"),
            ln.EvidenceEdge(edge_id="e2", relation="derived_from", source_entity_id="b", target_entity_id="a"),
        ]
        dag = ln.EvidenceDAG(registry, edges)
        assert any(row["kind"] == "cycle" for row in dag.validate())

    def test_orphan_nodes_are_reported(self) -> None:
        registry = ln.EntityRegistry()
        registry.register(_entity("a", "derived"))
        registry.register(_entity("b", "derived"))
        edge = ln.EvidenceEdge(edge_id="e1", relation="derived_from", source_entity_id="a", target_entity_id="b")
        # register a third, unconnected entity
        registry.register(_entity("c", "derived"))
        dag = ln.EvidenceDAG(registry, [edge])
        assert any(row["kind"] == "orphan" and "c" in row["affected_ids"] for row in dag.validate())

    def test_reverse_trace_and_downstream(self) -> None:
        registry = ln.EntityRegistry()
        for name in ("raw_1", "norm_1", "fig_1"):
            registry.register(_entity(name, "derived"))
        edges = [
            ln.EvidenceEdge(edge_id="e1", relation="derived_from", source_entity_id="raw_1", target_entity_id="norm_1"),
            ln.EvidenceEdge(edge_id="e2", relation="rendered_as", source_entity_id="norm_1", target_entity_id="fig_1"),
        ]
        dag = ln.EvidenceDAG(registry, edges)
        trace = dag.reverse_trace("fig_1")
        assert set(trace["ancestors"]) == {"norm_1", "raw_1"}
        assert set(dag.downstream("raw_1")) == {"norm_1", "fig_1"}


class TestClaimsAndPoints:
    def test_orphan_claim_is_flagged(self) -> None:
        registry = ln.ClaimRegistry()
        orphan = registry.register_claim(
            "c1",
            experiment_id="E12-03",
            evidence_level="L0",
            status="NOT_PUBLISHABLE",
            source_result_ids=(),
            figure_or_table_ids=(),
            limitations=(),
        )
        assert orphan["orphan"] is True
        assert [row["claim_id"] for row in registry.orphan_claims()] == ["c1"]
        assert registry.publishable_claims() == ()

    def test_publishable_claim_needs_sources(self) -> None:
        registry = ln.ClaimRegistry()
        registry.register_claim(
            "c1",
            experiment_id="E12-03",
            evidence_level="L2",
            status="PUBLISHABLE",
            source_result_ids=("norm_1",),
            figure_or_table_ids=("fig_1",),
            limitations=("preliminary",),
        )
        assert [row["claim_id"] for row in registry.publishable_claims()] == ["c1"]

    def test_point_requires_result_ids(self) -> None:
        point = ln.DashboardPoint(
            point_id="p1", view_or_query="q", view_version="v1", filters=(), normalized_result_ids=(), format="table", displayed_rounding="2"
        )
        assert any("result ids" in item for item in point.validate())
        good = ln.DashboardPoint(
            point_id="p1", view_or_query="q", view_version="v1", filters=(), normalized_result_ids=("norm_1",), format="table", displayed_rounding="2"
        )
        assert good.validate() == []

    def test_point_references_must_resolve(self) -> None:
        registry = ln.EntityRegistry()
        registry.register(_entity("norm_1", "derived"))
        point = ln.DashboardPoint(
            point_id="p1", view_or_query="q", view_version="v1", filters=(), normalized_result_ids=("ghost",), format="table", displayed_rounding="2"
        )
        problems = ln.validate_point(point, registry)
        assert any("unknown result" in item for item in problems)


class TestValidationAndDiff:
    def test_semantic_cross_checks_require_contracts_and_sources(self) -> None:
        findings = ln.semantic_cross_checks(
            [
                {"id": "r1", "comparability_status": "COMPARABLE"},
                {"id": "e1", "energy_measurement_id": "e"},
                {"id": "c1", "cost_result_id": "c"},
                {"id": "rec1", "recommendation_id": "r"},
            ]
        )
        assert len(findings) == 4
        ids = {row["check_id"] for row in findings}
        assert "semantic_comparable_without_contract" in ids
        assert "semantic_energy_without_window" in ids
        assert "semantic_cost_without_price" in ids
        assert "semantic_recommendation_without_frontier" in ids

    def test_unit_closure_check_catches_non_closing_rows(self) -> None:
        ok = ln.unit_closure_check([{"id": "e1", "total_energy_j": 40.0, "accepted_compliant_tokens": 20, "j_per_token": 2.0}])
        assert ok["ok"] is True
        bad = ln.unit_closure_check([{"id": "e2", "total_energy_j": 40.0, "accepted_compliant_tokens": 20, "j_per_token": 3.0}])
        assert bad["ok"] is False

    def test_diff_uses_per_transform_tolerance(self) -> None:
        policy = ln.DiffPolicy(tolerances={"float_generic": 1e-6})
        assert ln.diff_objects({"x": 1.0}, {"x": 1.0 + 1e-9}, policy=policy) == ()
        assert ln.diff_objects({"x": 1.0}, {"x": 1.1}, policy=policy)
        assert ln.diff_objects({"x": 1.0}, {"y": 2.0}, policy=policy)[0]["kind"] == "MISSING"

    def test_regeneration_plan_and_verdict(self) -> None:
        registry = ln.EntityRegistry()
        registry.register(_entity("raw_1", "raw", byte_hash="a" * 64))
        registry.register(_entity("norm_1", "derived"))
        dag = ln.EvidenceDAG(registry, [ln.EvidenceEdge(edge_id="e", relation="derived_from", source_entity_id="raw_1", target_entity_id="norm_1")])
        plan = ln.plan_regeneration(dag=dag, stages=("normalize", "render"))
        assert plan["dag_ok"] is True
        run = ln.RegenerationRun(disable_derived_cache=True)
        run.record_step("rebuild", "PASS")
        verdict = ln.regeneration_verdict(regeneration=run, diffs=[])
        assert verdict["ok"] is True
        assert verdict["derived_cache_disabled"] is True

    def test_semantic_change_fails_regeneration(self) -> None:
        run = ln.RegenerationRun()
        run.record_step("rebuild", "PASS")
        verdict = ln.regeneration_verdict(
            regeneration=run, diffs=[{"kind": "SEMANTIC_CHANGE", "field": "x"}]
        )
        assert verdict["ok"] is False


class TestSpotChecksAndFaults:
    def test_spot_checks_require_regeneration(self) -> None:
        report = ln.evaluate_spot_checks(
            [
                {"point_id": "p1", "stratum": "model_core", "reverse_trace_complete": True, "regenerated": True, "diff_kind": "EXACT"},
                {"point_id": "p2", "stratum": "cost", "reverse_trace_complete": False, "regenerated": False, "diff_kind": ""},
            ]
        )
        assert report["ok"] is False
        assert report["passed"] == 1

    def test_fault_injection_must_localise(self) -> None:
        results = [
            {"case_id": "raw_tampered", "expected_locator": "entity byte hash", "observed_locator": "raw_1.hash", "downstream_claims": ["c1"], "aggregate_only": False}
        ]
        assert ln.evaluate_fault_injection(results)["ok"] is True
        vague = [
            {"case_id": "raw_tampered", "expected_locator": "entity byte hash", "observed_locator": "", "downstream_claims": [], "aggregate_only": True}
        ]
        report = ln.evaluate_fault_injection(vague)
        assert report["ok"] is False

    def test_required_objects_are_reported_when_missing(self) -> None:
        registry = ln.EntityRegistry()
        registry.register(_entity("raw_1", "raw"))
        missing = ln.assert_required_objects_documented(registry)
        assert "comparison_contract" in missing

    def test_coverage_report_and_ready(self) -> None:
        coverage = ln.coverage_report(
            rows=[{"regeneration_level": "R3"}], claims=[{"claim_id": "c1"}]
        )
        assert ln.report_ready(coverage) is True
        not_ready = ln.coverage_report(rows=[{"regeneration_level": "R1"}], claims=[])
        assert ln.report_ready(not_ready) is False


class TestSmoke:
    def test_protocol_steps_are_complete(self) -> None:
        assert len(ln.PROTOCOL_STEPS) == 36
        assert all(interfaces for _, _, interfaces in ln.PROTOCOL_STEPS)

    def test_smoke_self_check_is_labelled(self) -> None:
        result = ln.smoke_self_check()
        assert result["status"] == "smoke"
        assert result["claim_allowed"] is False
        assert result["append_only_rejects_overwrite"] is True
        assert result["dag_valid_when_edges_connect"] is True
        assert result["orphan_claims_flagged"] is True
        assert result["reverse_trace_finds_ancestors"] is True
        assert result["point_requires_result_ids"] is True
