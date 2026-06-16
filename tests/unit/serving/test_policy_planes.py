"""Policy plane tests: SLO/funnel, arrival/loadgen, fairness, policies, admission."""

from __future__ import annotations

import os

import pytest

from hqsb.core.errors import ConfigError
from hqsb.serving import admission, arrival, fairness, loadgen, policies, slo

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CONFIG = os.path.join(_REPO_ROOT, "configs", "serving")


@pytest.mark.unit
class TestSloFunnel:
    def test_slo_template_refuses_evaluation(self):
        import yaml

        with open(os.path.join(_CONFIG, "slo_spec.yaml"), encoding="utf-8") as handle:
            spec = slo.SLOSpec.from_document(yaml.safe_load(handle))
        assert not spec.frozen
        with pytest.raises(ConfigError):
            spec.require_frozen()
        result = slo.max_slo_goodput([], spec)
        assert result["status"] == slo.NOT_EVALUATED

    def test_good_predicate_requires_all_four(self):
        import yaml

        with open(os.path.join(_CONFIG, "slo_spec.yaml"), encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        payload["preregistration_status"] = "frozen"
        spec = slo.SLOSpec.from_document(payload)
        row = slo.RequestSLOInput(
            request_id="r0",
            request_class="interactive",
            protocol_success=True,
            quality_identity_ok=True,
            client_ttft_ms=1000.0,  # interactive ttft_ms = 500 -> violated
            tpot_ms=10.0,
            client_e2e_ms=500.0,
        )
        assert slo.evaluate_request(row, spec).good is False
        row = slo.RequestSLOInput(
            request_id="r1",
            request_class="interactive",
            protocol_success=False,
            quality_identity_ok=True,
            client_ttft_ms=10.0,
            tpot_ms=1.0,
            client_e2e_ms=50.0,
        )
        assert slo.evaluate_request(row, spec).good is False

    def test_funnel_is_monotone_and_attributed(self):
        funnel = slo.FunnelCounts()
        funnel.add("offered", 10)
        funnel.add("client_attempted", 9)
        funnel.add("gateway_received", 9)
        funnel.add("valid", 8)
        funnel.add("admitted", 8)
        funnel.add("backend_started", 7)
        funnel.add("completed_success", 6)
        funnel.add("slo_good", 6)
        funnel.add_reason("invalid_request", 1)
        funnel.add_reason("backend_failure", 1)
        # offered - slo_good = 4, but only 2 reasons recorded -> audit fails
        audit = funnel.audit()
        assert not audit["ok"]
        funnel.add_reason("overload_reject", 1)
        funnel.add_reason("slo_violation", 1)
        assert funnel.audit()["ok"]

    def test_classify_failure_never_becomes_a_success(self):
        import yaml

        with open(os.path.join(_CONFIG, "slo_spec.yaml"), encoding="utf-8") as handle:
            spec = slo.SLOSpec.from_document(yaml.safe_load(handle))
        assert slo.classify_failure("service_overloaded", spec) == "rejected"
        assert slo.classify_failure("client_cancelled", spec) == "cancel"
        assert slo.classify_failure("backend_unavailable", spec) == "backend_failure"
        assert slo.classify_failure("protocol_failure", spec) == "protocol_failure"
        with pytest.raises(ConfigError):
            slo.classify_failure("made_up_code", spec)


@pytest.mark.unit
class TestArrival:
    def _spec(self):
        import yaml

        with open(os.path.join(_CONFIG, "arrival_spec.yaml"), encoding="utf-8") as handle:
            return arrival.ArrivalSpec.from_document(yaml.safe_load(handle))

    def test_trace_is_replayable_and_hashed(self):
        spec = self._spec()
        a = arrival.generate_arrival_trace(
            spec, distribution="poisson", mean_rate=8.0, count=200, seed=1
        )
        b = arrival.generate_arrival_trace(
            spec, distribution="poisson", mean_rate=8.0, count=200, seed=1
        )
        c = arrival.generate_arrival_trace(
            spec, distribution="poisson", mean_rate=8.0, count=200, seed=2
        )
        assert a.content_hash() == b.content_hash()
        assert a.content_hash() != c.content_hash()

    def test_five_distributions_are_mean_comparable(self):
        spec = self._spec()
        traces = [
            arrival.generate_arrival_trace(
                spec, distribution=name, mean_rate=8.0, count=400, seed=7
            )
            for name in ("constant", "poisson", "on_off_burst", "batch_burst", "compound_poisson")
        ]
        assert arrival.compare_arrival_traces(traces)["ok"]

    def test_batch_burst_keeps_the_mean_rate(self):
        deltas, _ = arrival.batch_burst_deltas(8.0, 64, batch_size=8, seed=1)
        assert abs(sum(deltas) - 64 / 8.0) < 1e-6

    def test_fidelity_gate_labels_loadgen_invalid(self):
        spec = self._spec()
        actual = [arrival.ActualArrival(scheduled_ns=0, sent_ns=1_000_000_000)]
        # a lag far above the tolerance must be flagged
        report = arrival.fidelity_gate(actual, spec=spec, mean_rate=8.0)
        assert report["label"] == arrival.INVALID_LABEL

    def test_negative_inter_arrival_refused(self):
        with pytest.raises(ConfigError):
            arrival.poisson_deltas(0.0, 10, seed=1)


@pytest.mark.unit
class TestLoadgen:
    def test_no_retry_is_enforced(self):
        with pytest.raises(ConfigError):
            loadgen.LoadgenSpec(mode="open_loop", no_retry=False)

    def test_open_loop_drops_are_recorded(self):
        def send(index, payload_id, scheduled_ns):
            if index == 2:
                return None, "event_loop_saturated"
            return scheduled_ns, ""

        records = loadgen.run_open_loop(
            __import__("hqsb.serving.arrival", fromlist=["ArrivalTrace"]).ArrivalTrace(
                distribution="constant",
                seed=0,
                mean_rate=10.0,
                deltas_sec=(0.1, 0.1, 0.1),
                scheduled_ns=(0, 100_000_000, 200_000_000),
                payload_ids=(0, 1, 2),
            ),
            base_ns=0,
            send=send,
        )
        assert records[2].dropped is True
        assert records[2].drop_reason == "event_loop_saturated"

    def test_loadgen_validity_flags_saturated_client(self):
        resources = loadgen.ClientResources(cpu_ratio=0.9)
        report = loadgen.loadgen_validity(
            [], spec=loadgen.LoadgenSpec(mode="open_loop", no_retry=True), resources=resources
        )
        assert report["label"] == loadgen.INVALID_LABEL


@pytest.mark.unit
class TestFairness:
    def test_jain_index(self):
        assert fairness.jain_index([1.0, 1.0]) == pytest.approx(1.0)
        assert fairness.jain_index([1.0, 0.0]) == pytest.approx(0.5)

    def test_cost_model_and_lag(self):
        model = fairness.CostModel(alpha=1.0, beta=2.0, gamma=0.5, version="s08.cost.v1")
        assert model.cost(
            uncached_input_tokens=10, committed_output_tokens=5, recomputed_positions=2
        ) == pytest.approx(21.0)
        lag = fairness.ideal_service_lag(
            {"a": 100.0, "b": 0.0}, {"a": 1.0, "b": 1.0}, total_served=100.0
        )
        assert lag["a"] == 50.0 and lag["b"] == -50.0

    def test_hol_traces_have_all_constructs(self):
        assert set(fairness.HOL_CONSTRUCTS)
        for construct in fairness.HOL_CONSTRUCTS:
            entries = fairness.hol_trace(construct)
            assert entries, construct

    def test_slowdown_needs_isolated_baseline(self):
        assert fairness.slowdown(100.0, None) is None
        assert fairness.slowdown(100.0, 50.0) == pytest.approx(2.0)


@pytest.mark.unit
class TestPolicies:
    def test_policies_share_one_interface(self):
        fifo = policies.build_policy("fifo")
        sp = policies.build_policy("strict_priority")
        wf = policies.build_policy("weighted_fair")
        queue = [
            policies.QueueEntry(
                request_id="low",
                tenant="a",
                request_class="batch",
                priority="low",
                enqueued_ns=0,
                estimated_cost=100,
            ),
            policies.QueueEntry(
                request_id="high",
                tenant="b",
                request_class="interactive",
                priority="high",
                enqueued_ns=10,
                estimated_cost=100,
            ),
        ]
        report = policies.policy_interface_audit([fifo, sp, wf], queue, now_ns=100)
        assert report["ok"], report["problems"]
        assert sp.select(list(queue), now_ns=100).selected_request_id == "high"
        assert fifo.select(list(queue), now_ns=100).selected_request_id == "low"

    def test_work_conserving_audit(self):
        log = policies.DecisionLog()
        report = log.work_conserving_audit([(100, 3, False)])
        assert not report["ok"]
        report = log.work_conserving_audit([(100, 3, True)])
        assert report["ok"]

    def test_refund_only_unexecuted(self):
        wf = policies.build_policy("weighted_fair")
        entry = policies.QueueEntry(
            request_id="r", tenant="a", request_class="c", priority="normal",
            enqueued_ns=0, estimated_cost=100, actual_cost=100, charged_cost=100,
        )
        wf._charge(entry)
        wf.refund(entry, unexecuted_fraction=0.4)
        assert entry.charged_cost == pytest.approx(60.0)


@pytest.mark.unit
class TestAdmission:
    def _spec(self):
        import yaml

        with open(os.path.join(_CONFIG, "admission_spec.yaml"), encoding="utf-8") as handle:
            return admission.PressureSpec.from_document(yaml.safe_load(handle))

    def test_pressure_state_machine_dwell(self):
        spec = self._spec()
        machine = admission.PressureStateMachine(spec)
        signal = admission.PressureSignal(queue_depth_ratio=0.9)
        for _ in range(spec.min_dwell_windows - 1):
            assert machine.observe(signal, monotonic_ns=0) == []
        events = machine.observe(signal, monotonic_ns=0)
        assert machine.state != "NORMAL" and events

    def test_bounded_queue_rejects_over_cap(self):
        queue = admission.BoundedQueue(max_requests=2, max_tokens=1000, max_bytes=1000)
        assert queue.try_admit(request_id="a", token_cost=100).admitted
        assert queue.try_admit(request_id="b", token_cost=100).admitted
        decision = queue.try_admit(request_id="c", token_cost=100)
        assert not decision.admitted and decision.code == "queue_full"

    def test_unbounded_queue_has_a_kill_guard(self):
        guard = admission.UnboundedQueueGuard(max_duration_s=1.0, kill_guard_requests=5)
        for index in range(6):
            guard.admitted += 1
        report = guard.guard(now_ns=int(2e9))
        assert report["stopped"] and "guard" in report["reason"]

    def test_retry_budget_respects_commit_boundary(self):
        spec = self._spec()
        budget = admission.RetryBudget(spec.retry, seed=1)
        budget.register_original()
        assert budget.decide(
            request_id="r", attempt=1, elapsed_ms=0, committed=False, retryable=True
        ).retry is True
        assert budget.decide(
            request_id="r", attempt=1, elapsed_ms=0, committed=True, retryable=True
        ).retry is False
