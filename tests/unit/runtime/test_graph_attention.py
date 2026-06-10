"""E07-06 graph buckets, attention capability, factorial analysis, claim gate."""

from __future__ import annotations

import pytest

from hqsb.core.errors import CapabilityError, ConfigError
from hqsb.runtime import graph_route as G


def _spec(**overrides) -> G.GraphSpec:
    buckets = (
        G.GraphBucketSpec(name="decode_b1_s1", max_sequences=1, max_tokens=1),
        G.GraphBucketSpec(name="prefill_b1_s512", max_sequences=1, max_tokens=512),
    )
    payload = {"buckets": overrides.pop("buckets", buckets), "max_graphs": 4}
    payload.update(overrides)
    return G.GraphSpec(**payload)


def _candidate(**overrides) -> G.AttentionCandidate:
    payload = {
        "name": "runtime_default",
        "provider": "runtime",
        "version": "1.0.0",
        "dtypes": ("float16",),
        "kv_dtypes": ("float16",),
        "head_dim": 128,
        "supports_gqa": True,
        "mask": "causal",
        "phases": ("prefill", "decode"),
        "paged_layout": True,
        "max_context": 4096,
    }
    payload.update(overrides)
    return G.AttentionCandidate(**payload)


def _request(**overrides) -> G.AttentionRequest:
    payload = {
        "dtype": "float16",
        "kv_dtype": "float16",
        "head_dim": 128,
        "query_heads": 16,
        "kv_heads": 8,
        "context": 1024,
        "phase": "decode",
        "paged": True,
    }
    payload.update(overrides)
    return G.AttentionRequest(**payload)


@pytest.mark.unit
class TestGraphSpec:
    def test_bucket_hit_reports_actual_mode(self):
        decision = _spec().resolve(sequences=1, tokens=1, phase="decode")
        assert decision.status == "HIT"
        assert decision.actual_mode == G.CUDA_GRAPH
        assert decision.fallback == ""

    def test_out_of_bucket_requires_a_reason_and_falls_back(self):
        decision = _spec().resolve(sequences=1, tokens=4096, phase="prefill")
        assert decision.status == "OUT_OF_BUCKET"
        assert decision.fallback == "fallback_non_graph"
        assert decision.reason
        assert decision.actual_mode == G.EAGER

    def test_padding_is_reported_for_the_chosen_bucket(self):
        decision = _spec().resolve(sequences=1, tokens=100, phase="prefill")
        assert decision.padding_tokens == 412

    def test_bucket_count_may_not_exceed_max_graphs(self):
        with pytest.raises(ConfigError):
            _spec(max_graphs=1)

    def test_unknown_out_of_bucket_policy_refused(self):
        with pytest.raises(ConfigError):
            _spec(out_of_bucket_policy="ignore")

    def test_unknown_phase_refused(self):
        with pytest.raises(ConfigError):
            _spec().resolve(sequences=1, tokens=1, phase="both")

    def test_zero_shape_refused(self):
        with pytest.raises(ConfigError):
            _spec().resolve(sequences=0, tokens=1, phase="decode")


@pytest.mark.unit
class TestReplayAccounting:
    def test_replay_distinctness_requires_different_inputs(self):
        records = (
            G.ReplayRecord(bucket="b", inputs_hash="a", replay_index=0),
            G.ReplayRecord(bucket="b", inputs_hash="b", replay_index=1),
        )
        assert G.replay_distinctness(records)["ok"]

    def test_single_input_replays_are_flagged(self):
        records = (
            G.ReplayRecord(bucket="b", inputs_hash="a", replay_index=0),
            G.ReplayRecord(bucket="b", inputs_hash="a", replay_index=1),
        )
        report = G.replay_distinctness(records)
        assert not report["ok"]
        assert report["problems"]

    def test_break_even_is_none_when_capture_cannot_amortise(self):
        report = G.capture_cost_break_even(capture_ms=100.0, eager_ms=10.0, steady_ms=10.0)
        assert report["break_even_replays"] is None
        assert report["reason"]

    def test_break_even_rounds_up(self):
        report = G.capture_cost_break_even(capture_ms=100.0, eager_ms=10.0, steady_ms=9.0)
        assert report["break_even_replays"] == 100

    def test_negative_timing_refused(self):
        with pytest.raises(ConfigError):
            G.capture_cost_break_even(capture_ms=1.0, eager_ms=0.0, steady_ms=1.0)


@pytest.mark.unit
class TestAttentionCapability:
    def test_supported_shape_passes(self):
        assert G.check_attention_support(_candidate(), _request()).supported

    def test_each_unsupported_field_produces_a_reason(self):
        assert not G.check_attention_support(
            _candidate(), _request(context=8192)
        ).supported
        assert not G.check_attention_support(
            _candidate(paged_layout=False), _request(paged=True)
        ).supported
        assert not G.check_attention_support(
            _candidate(supports_gqa=False), _request()
        ).supported
        assert not G.check_attention_support(
            _candidate(dtypes=("float32",)), _request()
        ).supported

    def test_alignment_is_enforced(self):
        decision = G.check_attention_support(
            _candidate(alignment=16, head_dim=120), _request(head_dim=120)
        )
        assert not decision.supported
        assert "alignment" in decision.reason

    def test_graph_capture_requires_capturable_candidate(self):
        decision = G.check_attention_support(
            _candidate(graph_capturable=False), _request(graph_capture=True)
        )
        assert not decision.supported

    def test_unknown_mask_refused_at_construction(self):
        with pytest.raises(ConfigError):
            _candidate(mask="sliding")

    def test_capability_error_factory_carries_the_fallback(self):
        error = G.attention_capability_error(
            G.AttentionSupport(
                candidate="c", supported=False, fallback="runtime_default", reason="no"
            )
        )
        assert isinstance(error, CapabilityError)
        assert error.details["fallback"] == "runtime_default"
        assert error.details["reason"] == "no"

    def test_unsupported_decision_requires_a_reason(self):
        with pytest.raises(ConfigError):
            G.AttentionSupport(candidate="c", supported=False, fallback="d")

    def test_matrix_is_generated_for_every_candidate_and_shape(self):
        rows = G.attention_matrix([_candidate()], G.phase_request_matrix([128, 1024], [1, 4], "decode"))
        assert len(rows) == 4


@pytest.mark.unit
class TestFactorial:
    def _cells(self, **overrides) -> tuple:
        metric = overrides.pop("metric", 10.0)
        return (
            G.FactorialCell(
                submission="eager", attention="default", metrics={"tpot_ms": metric}
            ),
            G.FactorialCell(
                submission="eager", attention="candidate", metrics={"tpot_ms": metric - 1}
            ),
            G.FactorialCell(
                submission="cuda_graph", attention="default", metrics={"tpot_ms": metric - 2}
            ),
            G.FactorialCell(
                submission="cuda_graph", attention="candidate", metrics={"tpot_ms": metric - 2.5}
            ),
        )

    def test_matrix_requires_every_cell(self):
        cells = self._cells()
        matrix = G.factorial_matrix(
            ["eager", "cuda_graph"], ["default", "candidate"], cells
        )
        assert matrix["ok"]

    def test_missing_cell_is_reported(self):
        cells = self._cells()[:3]
        matrix = G.factorial_matrix(
            ["eager", "cuda_graph"], ["default", "candidate"], cells
        )
        assert not matrix["ok"]
        assert matrix["missing_cells"] == [["cuda_graph", "candidate"]]

    def test_unsupported_cell_must_carry_a_reason(self):
        with pytest.raises(ConfigError):
            G.FactorialCell(
                submission="eager", attention="candidate", status=G.CELL_UNSUPPORTED
            )

    def test_two_factor_analysis_decomposes_effects(self):
        effects = G.two_factor_analysis(self._cells(), metric="tpot_ms")
        assert effects.extrapolation_allowed
        assert effects.graph_effect == pytest.approx(-1.75)
        assert effects.attention_effect == pytest.approx(-0.75)
        assert effects.interaction == pytest.approx(0.5)

    def test_analysis_refuses_to_extrapolate_across_unsupported_cells(self):
        cells = list(self._cells()[:3]) + [
            G.FactorialCell(
                submission="cuda_graph",
                attention="candidate",
                status=G.CELL_UNSUPPORTED,
                reason="not capturable",
            )
        ]
        effects = G.two_factor_analysis(cells, metric="tpot_ms")
        assert not effects.extrapolation_allowed
        assert effects.unsupported_cells

    def test_missing_metric_is_not_invented(self):
        with pytest.raises(ConfigError):
            G.two_factor_analysis(self._cells(), metric="ttft_ms")


@pytest.mark.unit
class TestPhaseAndClaim:
    def test_phases_are_reported_separately(self):
        rows = [
            G.PhaseMetrics(
                configuration="eager",
                prefill_ms=100.0,
                decode_ms=200.0,
                prefill_tokens=512,
                decode_tokens=64,
            )
        ]
        report = G.phase_split_report(rows)
        assert report["phases_reported_separately"]
        assert report["rows"][0]["prefill_tokens_per_s"] == pytest.approx(5120.0)
        assert report["rows"][0]["decode_tokens_per_s"] == pytest.approx(320.0)

    def test_phase_split_needs_rows(self):
        with pytest.raises(ConfigError):
            G.phase_split_report([])

    def test_claim_without_execution_is_not_run(self):
        status = G.claim_status(G.GraphClaimEvidence(), executed=False, spec=_spec())
        assert status["status"] == G.ClaimStatus.NOT_RUN

    def test_claim_with_incomplete_evidence_is_not_claimed(self):
        status = G.claim_status(
            G.GraphClaimEvidence(eligibility_checked=True),
            executed=True,
            spec=_spec(),
        )
        assert status["status"] == G.ClaimStatus.NOT_CLAIMED
        assert "capture_recorded" in status["reason"]

    def test_spec_can_hold_the_claim_back_even_with_full_evidence(self):
        evidence = G.GraphClaimEvidence(
            eligibility_checked=True,
            capture_recorded=True,
            multi_input_replay_verified=True,
            correctness_gate_passed=True,
            fallback_recorded=True,
            memory_accounted=True,
            replay_distinct=True,
        )
        assert G.claim_status(evidence, executed=True, spec=_spec())["status"] == (
            G.ClaimStatus.NOT_CLAIMED
        )
        assert G.claim_status(
            evidence, executed=True, spec=_spec(claim_cuda_graph=True)
        )["status"] == G.ClaimStatus.CLAIMED

    def test_e06_08_inheritance_is_reported(self):
        report = G.inherit_from_e06_08()
        if report["available"]:
            assert report["contract"]["module"] == "hqsb.integration.cuda_graph"
            assert report["contract"]["output_contracts"]
        else:  # pragma: no cover - only when the integration package is absent
            assert report["reason"]
