"""Unit tests for the E12-03 four-layer replay data contract."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.evaluation import benchmark as bm
from hqsb.evaluation.layers import SloSpec, TokenAccounting

pytestmark = pytest.mark.unit


def _observation(**overrides: object) -> bm.Observation:
    payload = {
        "observation_id": "",
        "run_id": "run-1",
        "sample_id": "s1",
        "timestamp_monotonic_ns": 10,
        "candidate_id": "cand_a",
        "comparison_id": "cmp",
        "layer": "model_core",
        "scenario": "decode",
        "workload_spec_id": "ws_decode",
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "latency_component": "core_tpot",
        "latency_ns": 1_000_000,
        "quality_gate_id": "gate",
        "quality_status": "pass",
    }
    payload.update(overrides)
    return bm.Observation(**payload)  # type: ignore[arg-type]


class TestObservation:
    def test_valid_observation_has_a_stable_id(self) -> None:
        first = _observation()
        second = _observation()
        assert first.observation_id == second.observation_id
        assert first.validate() == []

    def test_unknown_layer_is_rejected(self) -> None:
        assert any("unknown layer" in item for item in _observation(layer="chip").validate())

    def test_metric_from_another_layer_is_rejected(self) -> None:
        problems = _observation(latency_component="client_ttft").validate()
        assert any("belongs to layer" in item for item in problems)

    def test_failed_run_must_not_carry_numbers(self) -> None:
        problems = _observation(status="RUN_FAILED", latency_ns=123.0).validate()
        assert any("must not carry numbers" in item for item in problems)
        assert _observation(status="RUN_FAILED", latency_ns=None, error_code="E_OOM").validate() == []

    def test_missing_actual_backend_blocks_the_row(self) -> None:
        problems = _observation(actual_backend="").validate()
        assert any("actual_backend is required" in item for item in problems)

    def test_silent_fallback_is_rejected(self) -> None:
        problems = _observation(requested_backend="triton", actual_backend="eager").validate()
        assert any("fallback reason" in item for item in problems)


class TestObservationStore:
    def test_append_is_idempotent_for_identical_rows(self) -> None:
        store = bm.ObservationStore()
        first = store.append(_observation())
        assert store.append(_observation()) == first
        assert len(store) == 1

    def test_raw_evidence_is_append_only(self) -> None:
        store = bm.ObservationStore()
        store.append(_observation())
        with pytest.raises(ConfigError):
            store.append(_observation(latency_ns=2_000_000))

    def test_invalid_observation_cannot_be_stored(self) -> None:
        store = bm.ObservationStore()
        with pytest.raises(ConfigError):
            store.append(_observation(layer="chip"))

    def test_by_run_filters(self) -> None:
        store = bm.ObservationStore()
        store.append(_observation())
        store.append(_observation(run_id="run-2", sample_id="s2"))
        assert len(store.by_run("run-1")) == 1


class TestNormalize:
    def test_forbidden_basis_is_refused(self) -> None:
        with pytest.raises(ConfigError):
            bm.normalize(
                [_observation()],
                metric_name="core_tpot",
                estimator="median",
                basis="tdp_as_energy",
                quality_status="pass",
                comparability_status="COMPARABLE",
            )

    def test_unknown_basis_is_refused(self) -> None:
        with pytest.raises(ConfigError):
            bm.normalize(
                [_observation()],
                metric_name="core_tpot",
                estimator="median",
                basis="per_vibes",
                quality_status="pass",
                comparability_status="COMPARABLE",
            )

    def test_per_token_basis_needs_a_denominator(self) -> None:
        with pytest.raises(ConfigError):
            bm.normalize(
                [_observation()],
                metric_name="core_tpot",
                estimator="median",
                basis="per_accepted_token",
                quality_status="pass",
                comparability_status="COMPARABLE",
            )

    def test_normalized_result_keeps_sources_and_denominator(self) -> None:
        result = bm.normalize(
            [_observation()],
            metric_name="core_tpot",
            estimator="median",
            basis="per_accepted_token",
            denominator=50.0,
            quality_status="pass",
            comparability_status="COMPARABLE",
        )
        assert result.value == pytest.approx(1_000_000 / 50.0)
        assert result.source_observation_ids
        assert result.denominator_value == 50.0
        assert result.validate() == []

    def test_failed_only_run_yields_a_missing_state_not_zero(self) -> None:
        result = bm.normalize(
            [_observation(status="RUN_FAILED", latency_ns=None, error_code="E")],
            metric_name="core_tpot",
            estimator="median",
            basis="per_accepted_token",
            denominator=10.0,
            quality_status="pass",
            comparability_status="",
        )
        assert result.value is None
        assert result.missing_reason == "NOT_RUN_PREREQUISITE"
        assert result.validate() == []

    def test_normalized_result_validation_rules(self) -> None:
        interval = bm.NormalizedResult(
            normalized_result_id="n1",
            source_observation_ids=("o1",),
            transform_activity_id="a1",
            metric_name="core_tpot",
            value=1.0,
            unit="ns",
            direction="lower_is_better",
            estimator="median",
            interval_low=0.9,
            interval_high=1.1,
            interval_method="",
            confidence_level=0.0,
            replication_unit="",
            normalization_basis="per_request",
            quality_status="pass",
        )
        problems = interval.validate()
        assert any("declared method" in item for item in problems)
        assert any("replication unit" in item for item in problems)
        assert any("confidence level" in item for item in problems)

    def test_quality_failed_result_may_not_carry_a_value(self) -> None:
        row = bm.NormalizedResult(
            normalized_result_id="n1",
            source_observation_ids=("o1",),
            transform_activity_id="a1",
            metric_name="core_tpot",
            value=1.0,
            unit="ns",
            direction="lower_is_better",
            estimator="median",
            normalization_basis="per_request",
            quality_status="QUALITY_GATE_FAILED",
        )
        assert any("quality-failed" in item for item in row.validate())


class TestCrossConstraints:
    def test_clean_row_passes_every_constraint(self) -> None:
        row = {
            "status": "OK",
            "layer": "model_core",
            "actual_backend": "cuda",
            "comparability_status": "COMPARABLE",
            "contract_sha256": "a" * 64,
            "quality_status": "pass",
        }
        assert bm.validate_cross_constraints(row) == ()

    def test_each_constraint_has_a_trigger(self) -> None:
        triggered = {
            "C1": bm.validate_cross_constraints({"status": "RUN_FAILED", "value": 0.0}),
            "C2": bm.validate_cross_constraints({"comparability_status": "COMPARABLE"}),
            "C3": bm.validate_cross_constraints({"capability_level": "BENCHMARKED"}),
            "C4": bm.validate_cross_constraints({"energy_measurement_id": "e1"}),
            "C5": bm.validate_cross_constraints({"cost_result_id": "c1"}),
            "C6": bm.validate_cross_constraints({"derived": True}),
            "C7": bm.validate_cross_constraints({"status": "OK"}),
            "C8": bm.validate_cross_constraints(
                {"quality_status": "QUALITY_GATE_FAILED", "in_pareto": True, "actual_backend": "cuda"}
            ),
            "C9": bm.validate_cross_constraints({"layer": "distributed", "actual_backend": "cuda"}),
            "C10": bm.validate_cross_constraints(
                {"interval_low": 1.0, "interval_high": 2.0, "actual_backend": "cuda"}
            ),
            "C11": bm.validate_cross_constraints({"point_id": "p1"}),
        }
        for prefix, findings in triggered.items():
            assert findings, prefix
            assert any(row["check_id"].startswith(prefix + "_") for row in findings), prefix

    def test_matrix_requires_rows_and_no_violations(self) -> None:
        assert bm.validate_matrix_rows([])["ok"] is False
        report = bm.validate_matrix_rows([{"status": "OK", "actual_backend": "cuda"}])
        assert report["ok"] is True
        assert len(report["checks"]) == len(bm.VALIDATOR_CONSTRAINTS)

    def test_layer_row_requires_that_layer_fields(self) -> None:
        problems = bm.validate_layer_row("distributed", {"metric_name": "global_goodput"})
        assert any("device_count" in item for item in problems)


class TestRunPlanning:
    def test_fewer_than_three_runs_is_refused(self) -> None:
        with pytest.raises(ConfigError):
            bm.plan_cells(
                comparison_group_id="g",
                candidate_id="c",
                layer="model_core",
                scenario="decode",
                workload_spec_id="ws",
                measurement_contract_id="mc",
                planned_repetitions=2,
            )

    def test_run_ids_are_unique_and_stable(self) -> None:
        plans = bm.plan_cells(
            comparison_group_id="g",
            candidate_id="c",
            layer="model_core",
            scenario="decode",
            workload_spec_id="ws",
            measurement_contract_id="mc",
            planned_repetitions=3,
        )
        assert len({plan.run_id for plan in plans}) == 3
        again = bm.plan_cells(
            comparison_group_id="g",
            candidate_id="c",
            layer="model_core",
            scenario="decode",
            workload_spec_id="ws",
            measurement_contract_id="mc",
            planned_repetitions=3,
        )
        assert [plan.run_id for plan in again] == [plan.run_id for plan in plans]

    def test_unknown_layer_is_refused(self) -> None:
        with pytest.raises(ConfigError):
            bm.plan_cells(
                comparison_group_id="g",
                candidate_id="c",
                layer="chip",
                scenario="decode",
                workload_spec_id="ws",
                measurement_contract_id="mc",
            )


class TestStages:
    def test_cold_start_stages_are_separate_from_steady_state(self) -> None:
        ledger = bm.ColdStartLedger("run-1")
        ledger.record_stage("model_load", started_ns=0, ended_ns=100)
        ledger.record_stage("warmup", started_ns=100, ended_ns=200)
        report = bm.steady_state_excludes_cold_start(ledger, steady_start_ns=150)
        assert report["excludes_cold_start"] is False
        assert bm.steady_state_excludes_cold_start(ledger, steady_start_ns=250)["excludes_cold_start"]

    def test_cold_start_stages_are_not_overwritten(self) -> None:
        ledger = bm.ColdStartLedger("run-1")
        ledger.record_stage("model_load", started_ns=0, ended_ns=100)
        with pytest.raises(ConfigError):
            ledger.record_stage("model_load", started_ns=0, ended_ns=50)
        with pytest.raises(ConfigError):
            ledger.record_stage("not_a_stage", started_ns=0, ended_ns=1)

    def test_warmup_converges_or_reports_failure(self) -> None:
        machine = bm.WarmupStateMachine(rounds=3, tolerance_rel=0.05)
        assert machine.observe(100.0, backend="cuda") is None
        assert machine.observe(101.0, backend="cuda") is None
        converged = machine.observe(102.0, backend="cuda")
        assert converged is not None and converged.converged is True

        drifting = bm.WarmupStateMachine(rounds=2, tolerance_rel=0.01)
        drifting.observe(100.0, backend="cuda")
        drifting.observe(200.0, backend="cuda")
        drifting.observe(100.0, backend="cuda")
        failed = drifting.observe(200.0, backend="cuda")
        assert failed is not None and failed.converged is False
        assert "did not converge" in failed.reason

    def test_warmup_rejects_bad_parameters(self) -> None:
        with pytest.raises(ConfigError):
            bm.WarmupStateMachine(rounds=0)
        with pytest.raises(ConfigError):
            bm.WarmupStateMachine(tolerance_rel=1.0)

    def test_thermal_gate_marks_instead_of_starting(self) -> None:
        assert bm.thermal_gate({"temperature_c": 50.0})["action"] == "PROCEED"
        hot = bm.thermal_gate({"temperature_c": 95.0})
        assert hot["action"] == "WAIT_OR_MARK"
        unknown = bm.thermal_gate({})
        assert unknown["action"] == "WAIT_OR_MARK"

    def test_run_health_labels(self) -> None:
        healthy = bm.run_health_check("run-1", {})
        assert healthy["healthy"] is True
        assert "HEALTHY_INCLUDED" in healthy["labels"]
        dirty = bm.run_health_check(
            "run-2",
            {"throttle_flags": "sw_thermal", "ram_used_ratio": 0.95, "telemetry_gap_s": 3},
            oom_events=1,
            neighbor_processes=("neighbour-job",),
            backlog_requests=5,
        )
        assert dirty["healthy"] is False
        assert "UNKNOWN_ANOMALY" not in dirty["labels"]


class TestEstimatesAndMatrices:
    def test_run_estimates_ignore_failed_samples(self) -> None:
        rows = [_observation(), _observation(sample_id="s2", latency_ns=3_000_000)]
        rows.append(_observation(sample_id="s3", status="RUN_FAILED", latency_ns=None, error_code="E"))
        estimates = bm.run_estimates(rows)
        assert len(estimates) == 1
        assert estimates[0]["sample_count"] == 2
        assert estimates[0]["value"] == pytest.approx(2_000_000.0)

    def test_comparison_matrix_keeps_raw_and_normalized(self) -> None:
        estimates = [{"run_id": "run-1", "cell_id": "cell-1", "metric_name": "core_tpot", "value": 100.0, "unit": "ns"}]
        rows = bm.comparison_matrix_rows(
            estimates,
            cell_meta={"cell-1": {"candidate_id": "c", "layer": "model_core", "denominator_value": 10.0}},
        )
        assert rows[0]["raw_value"] == 100.0
        assert rows[0]["normalized_value"] == pytest.approx(10.0)
        assert rows[0]["status"] == "VALID"

    def test_cross_layer_conversion_requires_explicit_residual(self) -> None:
        row = bm.cross_layer_conversion_row(
            chain_id="chain-1",
            candidate_id="cand_a",
            workload_spec_id="ws",
            operator_effect=1.4,
            op_time_share=0.3,
            dispatch_hit_rate=0.8,
            observed_model_core_effect=1.05,
        )
        assert row["predicted_model_core_effect"] == pytest.approx(1.096)
        assert row["residual_status"] == "UNEXPLAINED"
        without_effect = bm.cross_layer_conversion_row(
            chain_id="chain-2",
            candidate_id="cand_a",
            workload_spec_id="ws",
            operator_effect=None,
            op_time_share=None,
            dispatch_hit_rate=None,
            observed_model_core_effect=None,
        )
        assert without_effect["residual_status"] == "NOT_APPLICABLE"

    def test_cross_layer_conversion_rejects_impossible_hit_rate(self) -> None:
        with pytest.raises(ConfigError):
            bm.cross_layer_conversion_row(
                chain_id="chain-3",
                candidate_id="cand_a",
                workload_spec_id="ws",
                operator_effect=1.2,
                op_time_share=0.2,
                dispatch_hit_rate=1.5,
                observed_model_core_effect=1.0,
            )

    def test_goodput_excludes_over_slo_requests(self) -> None:
        slo = SloSpec(slo_id="interactive", ttft_ms=100.0, tpot_ms=50.0)
        requests = [
            {"request_id": "r1", "status": "ok", "ttft_ms": 10.0, "tpot_ms": 5.0, "accepted_output_tokens": 10},
            {"request_id": "r2", "status": "ok", "ttft_ms": 500.0, "tpot_ms": 5.0, "accepted_output_tokens": 10},
        ]
        report = bm.service_goodput_rows(requests, slo, measurement_seconds=2.0)
        assert report["compliant_requests"] == 1
        assert report["goodput"] == pytest.approx(0.5)
        assert report["token_goodput"] == pytest.approx(5.0)

    def test_overload_recovery_detects_hidden_backlog(self) -> None:
        requests = [{"status": "backlog"}, {"status": "ok"}]
        hidden = bm.overload_recovery_check(
            requests, measurement_end_ns=10, drain_deadline_ns=50
        )
        assert hidden["status"] == "BACKLOG_CONTAMINATION"
        drained = bm.overload_recovery_check(
            requests, measurement_end_ns=100, drain_deadline_ns=100
        )
        assert drained["status"] == "ok"

    def test_token_accounting_closure(self) -> None:
        closed = TokenAccounting(
            run_id="run-1",
            prompt_tokens=10,
            requested_output_tokens=20,
            generated_tokens_before_stop=20,
            accepted_output_tokens=20,
            served_output_tokens=20,
            admitted_requests=1,
            completed_requests=1,
        )
        assert bm.token_accounting_closes(closed)["ok"] is True


class TestSmoke:
    def test_protocol_steps_are_complete(self) -> None:
        assert len(bm.PROTOCOL_STEPS) == 36
        assert all(interfaces for _, _, interfaces in bm.PROTOCOL_STEPS)

    def test_smoke_self_check_is_labelled(self) -> None:
        result = bm.smoke_self_check()
        assert result["status"] == "smoke"
        assert result["claim_allowed"] is False
        assert result["forbidden_basis_rejected"] is True
        assert result["failed_row_with_number_rejected"] is True
