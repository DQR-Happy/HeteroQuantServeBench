"""E07-02 request state machine, spans, iteration ledger and hot path."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError, SchemaError
from hqsb.runtime import trace as T


def _completed_machine() -> T.RequestStateMachine:
    machine = T.RequestStateMachine("r0")
    for state in (
        T.RequestState.VALIDATED,
        T.RequestState.WAITING,
        T.RequestState.ADMITTED,
        T.RequestState.PREFILLING,
        T.RequestState.DECODING,
        T.RequestState.FINISHED,
    ):
        machine.transition(state)
    return machine


@pytest.mark.unit
class TestRequestStateMachine:
    def test_legal_path_is_recorded_with_reasons(self):
        machine = _completed_machine()
        assert machine.state == T.RequestState.FINISHED
        assert len(machine.transitions) == 6
        assert machine.transitions[0].old == T.RequestState.CREATED

    def test_illegal_transition_refused(self):
        with pytest.raises(SchemaError):
            T.RequestStateMachine("r0").transition(T.RequestState.DECODING)

    def test_terminal_state_only_advances_to_cleaned(self):
        machine = _completed_machine()
        with pytest.raises(SchemaError):
            machine.transition(T.RequestState.DECODING)

    def test_re_admission_after_cancel_is_detected(self):
        machine = T.RequestStateMachine("r0")
        machine.transition(T.RequestState.VALIDATED)
        machine.transition(T.RequestState.WAITING)
        machine.transition(T.RequestState.CANCEL_REQUESTED)
        machine.transition(T.RequestState.CANCELLED)
        # Force an illegal path by bypassing the guard, to prove the audit sees it.
        machine.state = T.RequestState.WAITING
        machine.transition(T.RequestState.ADMITTED)
        assert machine.re_admitted_after_cancel()

    def test_require_clean_finish_rejects_a_request_left_in_flight(self):
        machine = T.RequestStateMachine("r0")
        machine.transition(T.RequestState.VALIDATED)
        machine.transition(T.RequestState.WAITING)
        with pytest.raises(SchemaError):
            machine.require_clean_finish()

    def _decoding_machine(self) -> T.RequestStateMachine:
        machine = T.RequestStateMachine("r0")
        for state in (
            T.RequestState.VALIDATED,
            T.RequestState.WAITING,
            T.RequestState.ADMITTED,
            T.RequestState.PREFILLING,
            T.RequestState.DECODING,
        ):
            machine.transition(state)
        return machine

    def test_emitted_after_cancel_is_reported(self):
        machine = self._decoding_machine()
        machine.transition(T.RequestState.CANCEL_REQUESTED, timestamp_ns=100)
        tokens = [
            {"token_index": 0, "timestamp_ns": 90},
            {"token_index": 1, "timestamp_ns": 110},
        ]
        assert machine.emitted_after_cancel(tokens) == [1]

    def test_in_flight_tokens_may_be_allowed_by_contract(self):
        machine = self._decoding_machine()
        machine.transition(T.RequestState.CANCEL_REQUESTED, timestamp_ns=100)
        tokens = [{"token_index": 0, "timestamp_ns": 110, "in_flight_allowed": True}]
        assert machine.emitted_after_cancel(tokens) == []

    def test_cancel_after_finish_is_an_illegal_transition(self):
        with pytest.raises(SchemaError):
            _completed_machine().transition(T.RequestState.CANCEL_REQUESTED)


@pytest.mark.unit
class TestSpans:
    def _collector(self) -> T.SpanCollector:
        collector = T.SpanCollector(run_id="run")
        parent = collector.emit(
            "scheduler_iteration",
            request_id="r0",
            source_symbol="runtime.scheduler.step",
            end_ns=10,
        )
        collector.emit(
            "model_runner",
            request_id="r0",
            parent_span_id=parent.span_id,
            source_symbol="runtime.runner.forward",
            start_ns=2,
            end_ns=8,
            iteration=0,
        )
        return collector

    def test_join_audit_passes_for_a_well_formed_trace(self):
        assert self._collector().join_audit()["ok"]

    def test_missing_source_symbol_is_reported(self):
        collector = self._collector()
        collector.emit("cleanup", request_id="r0")
        audit = collector.join_audit()
        assert not audit["ok"]
        assert audit["missing_source_symbol"]

    def test_cross_request_parent_is_detected(self):
        collector = T.SpanCollector(run_id="run")
        parent = collector.emit("scheduler_iteration", request_id="r0", source_symbol="s")
        collector.emit(
            "model_runner", request_id="r1", parent_span_id=parent.span_id, source_symbol="s"
        )
        assert collector.join_audit()["cross_request_parent"]

    def test_span_must_not_end_before_it_starts(self):
        with pytest.raises(SchemaError):
            T.Span(kind="model_runner", span_id="s", start_ns=5, end_ns=1)

    def test_unknown_span_kind_refused(self):
        with pytest.raises(SchemaError):
            T.Span(kind="mystery", span_id="s")

    def test_sensitive_attributes_are_redacted(self):
        collector = T.SpanCollector(run_id="run")
        span = collector.emit(
            "request",
            request_id="r0",
            source_symbol="runtime.request",
            attributes={"prompt": "secret prompt", "tokens": 12},
        )
        assert span.attributes["prompt"].startswith("<redacted:")
        assert span.attributes["tokens"] == 12

    def test_long_strings_are_truncated(self):
        redacted = T.redact_attributes({"note": "x" * 600})
        assert redacted["note"].startswith("<truncated:")

    def test_call_chain_tree_reports_dangling_parents(self):
        collector = T.SpanCollector(run_id="run")
        collector.emit(
            "model_runner", request_id="r0", parent_span_id="missing", source_symbol="s"
        )
        tree = T.call_chain_tree(collector.spans, "r0")
        assert not tree["ok"]
        assert tree["dangling_parents"] == ["model_runner-1"]


@pytest.mark.unit
class TestIterationLedger:
    def test_conservation_identity_holds(self):
        entry = T.IterationLedgerEntry(
            iteration=0,
            scheduled_tokens={"r0": 4, "r1": 2},
            token_budget=8,
            previous_computed_positions=10,
            new_computed_positions=16,
        )
        assert entry.conservation_residual() == 0
        assert T.conservation_audit([entry])["ok"]

    def test_rollback_is_subtracted(self):
        entry = T.IterationLedgerEntry(
            iteration=1,
            scheduled_tokens={"r0": 4},
            token_budget=8,
            previous_computed_positions=16,
            rollback_positions=6,
            new_computed_positions=14,
        )
        assert entry.conservation_residual() == 0

    def test_conservation_violation_is_reported(self):
        entry = T.IterationLedgerEntry(
            iteration=2,
            scheduled_tokens={"r0": 4},
            token_budget=8,
            previous_computed_positions=0,
            new_computed_positions=9,
        )
        audit = T.conservation_audit([entry])
        assert not audit["ok"]
        assert audit["offenders"][0]["residual"] == -5

    def test_budget_utilization_needs_a_budget(self):
        entry = T.IterationLedgerEntry(iteration=0, token_budget=0)
        with pytest.raises(ConfigError):
            entry.budget_utilization

    def test_payload_labels_kv_and_route(self):
        entry = T.IterationLedgerEntry(
            iteration=3,
            scheduled_tokens={"r0": 1},
            token_budget=4,
            kv_free_blocks=2,
            kv_used_blocks=6,
            prefix_hits=16,
            graph_bucket="decode_b1_s1",
            attention_backend="runtime_default",
        )
        payload = entry.as_dict()
        assert payload["graph_bucket"] == "decode_b1_s1"
        assert payload["prefix_hits"] == 16
        assert payload["scheduled_total"] == 1


@pytest.mark.unit
class TestClocksAndOverhead:
    def test_clock_calibration_requires_a_method(self):
        with pytest.raises(ConfigError):
            T.ClockCalibration(host_reference_ns=0, device_reference_ns=0, skew_ns=0, method="")

    def test_device_span_is_ordered(self):
        clock = T.ClockCalibration(
            host_reference_ns=100, device_reference_ns=0, skew_ns=5, method="smoke"
        )
        assert clock.device_span_ns(10, 30) == 20
        assert clock.device_to_host_ns(10) == 110
        with pytest.raises(ConfigError):
            clock.device_span_ns(30, 10)

    def test_host_device_delta_removes_the_skew(self):
        clock = T.ClockCalibration(
            host_reference_ns=0, device_reference_ns=0, skew_ns=2, method="smoke"
        )
        delta = clock.host_device_delta_ns(
            host_start_ns=0, host_end_ns=20, device_start_ns=0, device_end_ns=10
        )
        assert delta == 8

    def test_full_trace_is_never_a_timing_level(self):
        report = T.instrumentation_overhead(
            [
                T.OverheadObservation(level="off", cpu_ms=10.0, gpu_ms=1.0, ttft_ms=20.0, tpot_ms=5.0),
                T.OverheadObservation(level="minimal", cpu_ms=10.1, gpu_ms=1.0, ttft_ms=20.1, tpot_ms=5.0),
                T.OverheadObservation(level="full", cpu_ms=30.0, gpu_ms=1.0, ttft_ms=40.0, tpot_ms=9.0),
            ],
            thresholds=T.OverheadThresholds(),
        )
        rows = {row["level"]: row for row in report["rows"]}
        assert rows["full"]["usable_for_timing"] is False
        assert rows["profiler"]["observed"] is False

    def test_overhead_needs_an_off_baseline(self):
        with pytest.raises(ConfigError):
            T.instrumentation_overhead(
                [T.OverheadObservation(level="minimal", cpu_ms=1.0, gpu_ms=0.0, ttft_ms=1.0, tpot_ms=1.0)],
                thresholds=T.OverheadThresholds(),
            )

    def test_unknown_instrumentation_level_refused(self):
        with pytest.raises(ConfigError):
            T.OverheadObservation(level="verbose", cpu_ms=0.0, gpu_ms=0.0, ttft_ms=0.0, tpot_ms=0.0)


@pytest.mark.unit
class TestReports:
    def test_state_machine_graph_is_generated_from_raw(self):
        graph = T.state_machine_graph([_completed_machine()])
        assert graph["generated_from_raw"]
        assert {"from": "CREATED", "to": "VALIDATED", "count": 1} in graph["edges"]

    def test_hot_path_table_sorts_by_total_time_and_notes_call_counts(self):
        rows = [
            T.HotPathRow(
                kind="sample",
                source_symbol="runtime.sample",
                calls=1000,
                cpu_self_ms=1.0,
                cpu_inclusive_ms=1.0,
                gpu_ms=0.0,
                p95_ms=0.1,
                queue_wait_ms=0.0,
            ),
            T.HotPathRow(
                kind="attention",
                source_symbol="runtime.attention",
                calls=5,
                cpu_self_ms=2.0,
                cpu_inclusive_ms=20.0,
                gpu_ms=30.0,
                p95_ms=10.0,
                queue_wait_ms=1.0,
            ),
        ]
        table = T.hot_path_table(rows, phase="decode")
        assert table["rows"][0]["kind"] == "attention"
        assert table["notes"]

    def test_hot_path_table_requires_a_phase(self):
        with pytest.raises(ConfigError):
            T.hot_path_table([], phase="both")

    def test_kernel_join_reports_unbound_kernels(self):
        collector = T.SpanCollector(run_id="run")
        span = collector.emit("attention", request_id="r0", source_symbol="s")
        report = T.kernel_join(
            collector.spans,
            [
                {"kernel": "flash_attn", "span_id": span.span_id, "device_ns": 5},
                {"kernel": "mystery", "span_id": "nope", "device_ns": 5},
            ],
        )
        assert not report["ok"]
        assert report["unbound_kernels"] == ["mystery"]
