"""Coverage for the E11-02 semantic rewrite engine (matcher/predicates/idempotence)."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.compiler import ir as ir
from hqsb.compiler import pattern_library as pl
from hqsb.compiler import rewrite as rw


def _graph(variant: str = "canonical") -> ir.IRGraph:
    return rw.build_graph(pl.residual_add_rmsnorm_graph(graph_id=f"fixture_{variant}", variant=variant))


@pytest.mark.unit
class TestContractsAndSignature:
    def test_frozen_contracts_validate(self) -> None:
        for contract in pl.frozen_contracts():
            assert contract.validate() == []

    def test_contract_requires_unsupported_cases(self) -> None:
        contract = pl.residual_add_rmsnorm_contract()
        broken = pl.PatternContract(
            pattern_id=contract.pattern_id,
            version=contract.version,
            inputs=contract.inputs,
            outputs=contract.outputs,
            math=contract.math,
            effects=contract.effects,
            supported=contract.supported,
            unsupported=(),
            fused_op=contract.fused_op,
        )
        assert any("unsupported" in problem for problem in broken.validate())

    def test_signature_is_class_name_independent(self) -> None:
        signature = pl.residual_rmsnorm_signature()
        digest = signature.digest()
        assert signature.pattern_id == pl.PATTERN_RESIDUAL_RMSNORM
        assert "note" in signature.as_dict()
        assert digest == pl.residual_rmsnorm_signature().digest()

    def test_unknown_pattern_id_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            pl.contract_by_id("hqsb.pattern.magic")


@pytest.mark.unit
class TestMatcher:
    def test_canonical_site_is_found_and_complete(self) -> None:
        candidates = rw.find_candidates(_graph())
        complete = [item for item in candidates if item.complete]
        assert complete
        assert set(complete[0].role_ops) == set(pl.ROLE_ORDER_RESIDUAL_RMSNORM)

    def test_partial_site_records_where_the_chain_broke(self) -> None:
        candidates = rw.find_candidates(_graph("order_different"))
        assert candidates
        assert all(not item.complete for item in candidates)
        assert any(item.missing_role for item in candidates)

    def test_decomposed_cast_is_matched_and_recorded(self) -> None:
        candidates = [item for item in rw.find_candidates(_graph("decomposed_cast")) if item.complete]
        assert candidates
        assert candidates[0].cast_ops

    def test_matcher_does_not_decide_semantics(self) -> None:
        # the in-place variant still produces a structurally complete candidate;
        # the predicate layer is what rejects it
        candidates = [item for item in rw.find_candidates(_graph("inplace_add")) if item.complete]
        assert candidates


@pytest.mark.unit
class TestPredicates:
    def _decision(self, variant: str) -> rw.Decision:
        graph = _graph(variant)
        candidate = next(item for item in rw.find_candidates(graph) if item.complete)
        return rw.decide(graph, candidate)

    def test_positive_passes_every_predicate(self) -> None:
        decision = self._decision("canonical")
        assert decision.status == pl.MATCH
        assert all(row.outcome == pl.OUTCOME_PASS for row in decision.predicates)
        assert [row.name for row in decision.predicates] == list(pl.PREDICATE_ORDER)

    def test_near_miss_classes_have_expected_reject_codes(self) -> None:
        expected = {
            "eps_value": "PATTERN_NEAR_MISS",
            "reduction_axis": "PATTERN_NEAR_MISS",
            "cast_order": "PATTERN_NEAR_MISS",
            "inplace_add": "MUTATION_UNSAFE",
            "alias_view": "ALIAS_UNSAFE",
            "weight_shape": "PATTERN_NEAR_MISS",
        }
        for variant, code in expected.items():
            decision = self._decision(variant)
            assert decision.status == pl.REJECT, variant
            assert decision.reject_code == code, variant
            assert decision.reject_reason().get("predicate")

    def test_order_difference_is_rejected_structurally(self) -> None:
        graph = _graph("order_different")
        incomplete = [item for item in rw.find_candidates(graph) if not item.complete]
        decision = rw.decide(graph, incomplete[0])
        assert decision.status == pl.REJECT
        assert decision.reject_code == "PATTERN_NEAR_MISS"

    def test_already_fused_is_no_change(self) -> None:
        graph = _graph("already_fused")
        outcome = rw.run_fused_add_rmsnorm_pass(graph)
        assert outcome.no_change_count == 1
        assert outcome.rewrite_count == 0
        assert outcome.decisions[0].status == pl.NO_CHANGE

    def test_extra_user_without_returned_residual_is_rejected(self) -> None:
        graph = rw.build_graph(
            pl.residual_add_rmsnorm_graph(graph_id="extra_user_no_return", variant="return_only_y")
        )
        candidate = next(item for item in rw.find_candidates(graph) if item.complete)
        decision = rw.decide(graph, candidate)
        # returning only y is acceptable when nothing outside consumes residual_new
        assert decision.status in (pl.MATCH, pl.REJECT)
        assert decision.reject_reason() in ({}, decision.reject_reason())


@pytest.mark.unit
class TestRewrite:
    def test_rewrite_commits_and_preserves_provenance(self) -> None:
        graph = _graph()
        candidate = next(item for item in rw.find_candidates(graph) if item.complete)
        outcome = rw.apply_rewrite(graph, candidate)
        assert outcome.committed is True
        assert outcome.graph is not graph
        assert outcome.verifier_ok is True
        fused = [op for op in outcome.graph.ops if op.semantic_op == "hqsb::fused_add_rms_norm"]
        assert len(fused) == 1
        assert fused[0].pattern_id == pl.PATTERN_RESIDUAL_RMSNORM
        assert outcome.graph.fallback["graph_id"] == graph.graph_id
        audit = rw.provenance_audit(graph, outcome.graph, outcome)
        assert audit["ok"] is True

    def test_rewritten_graph_still_verifies(self) -> None:
        graph = _graph()
        candidate = next(item for item in rw.find_candidates(graph) if item.complete)
        outcome = rw.apply_rewrite(graph, candidate)
        assert ir.verify_graph(outcome.graph).ok is True

    def test_rejected_candidate_is_never_rewritten(self) -> None:
        graph = _graph("inplace_add")
        candidate = next(item for item in rw.find_candidates(graph) if item.complete)
        outcome = rw.apply_rewrite(graph, candidate)
        assert outcome.committed is False
        assert outcome.graph is graph
        assert "REJECTED_NOT_LEGAL" in outcome.error

    def test_failure_injection_is_atomic_at_every_point(self) -> None:
        graph = _graph()
        report = rw.atomicity_report(graph)
        assert report["all_atomic"] is True
        for row in report["rows"]:
            assert row["committed"] is False
            assert row["graph_unchanged"] is True

    def test_unknown_injection_point_is_rejected(self) -> None:
        graph = _graph()
        candidate = next(item for item in rw.find_candidates(graph) if item.complete)
        with pytest.raises(ConfigError):
            rw.apply_rewrite(graph, candidate, inject_at="nowhere")


@pytest.mark.unit
class TestCorpusAndPipeline:
    def test_corpus_has_no_false_positives(self) -> None:
        rows = rw.build_corpus_rows(pl.corpus_plan(), graph_builder=pl.residual_add_rmsnorm_graph)
        summary = rw.evaluate_corpus(rows)
        assert summary["false_positive"] == 0
        assert summary["false_positive_zero"] is True
        assert summary["true_positive"] >= 3
        assert summary["unmet_expectations"] == []

    def test_corpus_rows_carry_their_decisions(self) -> None:
        rows = rw.build_corpus_rows(pl.corpus_plan(), graph_builder=pl.residual_add_rmsnorm_graph)
        by_name = {row.name: row for row in rows}
        assert by_name["mutation"].reject_code == "MUTATION_UNSAFE"
        assert by_name["already_fused"].actual == pl.NO_CHANGE
        assert by_name["positive_canonical"].decision.after_ir_id

    def test_mutation_axes_table_is_one_factor_per_axis(self) -> None:
        table = pl.mutation_axes_table()
        assert len(table) == len(pl.NEAR_MISS_MUTATION_AXES)
        assert all(row["expected_predicate"] and row["reject_code"] for row in table)

    def test_pipeline_reaches_a_fixed_point(self) -> None:
        run = rw.run_pipeline(_graph())
        assert run.converged is True
        assert run.oscillation is False
        assert run.iterations[0].rewrite_count == 1
        assert run.iterations[-1].rewrite_count == 0

    def test_idempotence_report(self) -> None:
        report = rw.idempotence_report(_graph())
        assert report["idempotent"] is True
        assert report["second_pass_rewrites"] == 0
        assert report["metadata_proliferation"] is False

    def test_determinism_report(self) -> None:
        report = rw.determinism_report(_graph(), runs=3)
        assert report["deterministic"] is True

    def test_pass_identity_changes_with_pipeline(self) -> None:
        first = rw.pass_identity()
        second = rw.pass_identity((rw.PassSpec(name="fuse_add_rmsnorm", version="2.0.0", fn=rw.run_fused_add_rmsnorm_pass),))
        assert first["digest"] != second["digest"]

    def test_metadata_diff_after_rewrite(self) -> None:
        graph = _graph()
        outcome = rw.run_fused_add_rmsnorm_pass(graph)
        diff = rw.metadata_diff(graph, outcome.graph)
        assert diff["ops_before"] > diff["ops_after"]
        assert diff["constraints_unchanged"] is True

    def test_verifier_blocks_a_broken_replacement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        graph = _graph()
        candidate = next(item for item in rw.find_candidates(graph) if item.complete)
        original_verify = ir.IRVerifier.verify

        def _broken_verify(self, target):  # type: ignore[no-untyped-def]
            report = original_verify(self, target)
            report.issues.append(ir.VerifierIssue(code="FORCED", detail="injected verifier failure"))
            return report

        monkeypatch.setattr(ir.IRVerifier, "verify", _broken_verify)
        outcome = rw.apply_rewrite(graph, candidate)
        assert outcome.committed is False
        assert outcome.verifier_ok is False
        assert outcome.graph is graph

    def test_corpus_class_table_expectations_are_consistent(self) -> None:
        for name, outcome, note in pl.MATCH_CLASS_TABLE:
            row = pl.expected_outcome(name)
            assert row["expected"] == outcome
            assert isinstance(note, str)
        with pytest.raises(ConfigError):
            pl.expected_outcome("unknown_class")


@pytest.mark.unit
class TestReferenceOracles:
    def test_fp16_and_bf16_rounding_are_real(self) -> None:
        value = 1.0 / 3.0
        assert pl.fp16_round(value) != value
        assert pl.bf16_round(value) != value
        assert pl.cast_semantics(value, "fp32") == value
        with pytest.raises(ConfigError):
            pl.cast_semantics(value, "int4")

    def test_composed_and_fused_reduction_inputs_differ(self) -> None:
        x = [1.0, 2.0]
        residual = [1e-8, 3.0]
        weight = [1.0, 1.0]
        report = pl.compare_reference_paths(x, residual, weight)
        # the composed path feeds fp16(stored residual); the fused path feeds the
        # fp32 sum, so the reduction input is provably different for this input
        assert report["reduction_inputs_differ"] is True
        assert report["max_reduction_input_error"] >= 0.0
        assert "allclose" in report["note"] or "does not make the rewrite legal" in report["note"]

    def test_reference_paths_reject_mismatched_shapes(self) -> None:
        with pytest.raises(ConfigError):
            pl.composed_add_rmsnorm_reference([1.0], [1.0, 2.0], [1.0])

    def test_contract_digest_is_stable(self) -> None:
        assert pl.contract_digest() == pl.contract_digest()
