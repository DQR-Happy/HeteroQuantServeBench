"""E10-03 / E10-10 interface tests: communicator safety, oracles, recovery bounds."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.distributed import faults as ft
from hqsb.distributed import sequence as sq


def _record(**overrides) -> sq.CollectiveCallRecord:
    payload = {
        "run_id": "r",
        "rank_epoch": 0,
        "global_rank": 0,
        "group_rank": 0,
        "group_id": "tp",
        "ordered_group_hash": "h",
        "collective_seq": 0,
        "op": "all_reduce",
        "reduce_op": "sum",
        "logical_count": 1024,
        "dtype": "fp16",
        "input_shape_hash": "i",
        "output_shape_hash": "o",
        "stream_id": "s",
        "callsite": "layer.0",
    }
    payload.update(overrides)
    return sq.CollectiveCallRecord(**payload)


@pytest.mark.unit
class TestStateMachine:
    def test_legal_sequence_and_post_error_rejection(self):
        machine = sq.CommunicatorStateMachine("tp")
        machine.transition("READY")
        machine.begin_collective(0)
        machine.complete()
        machine.fail("mismatch")
        assert machine.state == "ABORTED"
        with pytest.raises(ConfigError):
            machine.begin_collective(1)

    def test_illegal_transition_is_refused(self):
        machine = sq.CommunicatorStateMachine("tp")
        with pytest.raises(ConfigError):
            machine.transition("COMPLETED")

    def test_sequence_must_increase(self):
        machine = sq.CommunicatorStateMachine("tp")
        machine.transition("READY")
        machine.begin_collective(1)
        machine.complete()
        with pytest.raises(ConfigError):
            machine.begin_collective(1)

    def test_destroy_or_abort_handles_a_failed_communicator(self):
        machine = sq.CommunicatorStateMachine("tp")
        machine.transition("READY")
        machine.fail("boom")
        machine.destroy_or_abort()
        assert machine.state == "DESTROYED"

    def test_destroy_refuses_a_failed_state(self):
        # a communicator in ERROR must be aborted before it can be destroyed;
        # destroying it directly would hide the failure from the state machine
        machine = sq.CommunicatorStateMachine("tp")
        machine.transition("READY")
        machine.transition("ERROR", reason="boom")
        with pytest.raises(ConfigError):
            machine.destroy()
        machine.destroy_or_abort()
        assert machine.state == "DESTROYED"


@pytest.mark.unit
class TestSequenceAndPreflight:
    def test_sequence_namespace_resets_with_a_new_epoch(self):
        allocator = sq.SequenceAllocator("tp", rank_epoch=0)
        assert allocator.next() == 0
        allocator.reset_for_epoch(1)
        assert allocator.next() == 0
        with pytest.raises(ConfigError):
            allocator.reset_for_epoch(1)

    def test_preflight_finds_the_first_divergence_field(self):
        left = _record()
        right = _record(global_rank=1, group_rank=1, logical_count=2048)
        result = sq.preflight_check({0: left, 1: right})
        assert result.ok is False
        assert result.divergence.field == "logical_count"
        assert result.handshake_is_not_a_barrier is True

    def test_dtype_mismatch_is_not_masked_by_equal_bytes(self):
        left = _record(dtype="fp16", logical_count=1024)  # 2048 bytes
        right = _record(dtype="fp32", logical_count=512)  # 2048 bytes
        assert sq.byte_counts_match_but_semantics_differ(left, right) is True
        result = sq.preflight_check({0: left, 1: right})
        assert result.byte_counts_match_but_semantics_differ is True

    def test_preflight_only_runs_in_debug_or_fault_mode(self):
        with pytest.raises(ConfigError):
            sq.preflight_check({0: _record(), 1: _record()}, mode="performance")

    def test_identical_records_pass_the_preflight(self):
        result = sq.preflight_check({0: _record(), 1: _record(global_rank=1, group_rank=1)})
        assert result.ok is True


@pytest.mark.unit
class TestTimeoutsAndWatchdog:
    def test_timeout_is_clamped_and_needs_a_healthy_p99(self):
        spec = sq.TimeoutSpec(
            init_timeout_s=10.0, collective_timeout_s=1.0, watchdog_heartbeat_timeout_s=1.0,
            job_kill_grace_s=5.0, post_cleanup_probe_timeout_s=5.0, multiplier=3.0,
            minimum_operational_timeout_s=0.5, maximum_timeout_s=30.0,
        )
        assert spec.compute_collective_timeout(healthy_p99_s=0.1) == pytest.approx(0.5)
        assert spec.compute_collective_timeout(healthy_p99_s=100.0) == pytest.approx(30.0)
        with pytest.raises(ConfigError):
            spec.compute_collective_timeout(healthy_p99_s=0.0)

    def test_timeout_may_not_be_edited_after_the_fault(self):
        spec = sq.TimeoutSpec(
            init_timeout_s=10.0, collective_timeout_s=1.0, watchdog_heartbeat_timeout_s=1.0,
            job_kill_grace_s=5.0, post_cleanup_probe_timeout_s=5.0, multiplier=2.0,
            minimum_operational_timeout_s=0.5, maximum_timeout_s=30.0,
        )
        with pytest.raises(ConfigError):
            spec.refuse_post_fault_edit(edited_after_fault=True)

    def test_watchdog_distinguishes_dead_from_stuck(self):
        watchdog = sq.Watchdog(heartbeat_timeout_s=5.0, progress_timeout_s=1.0)
        watchdog.record(rank=0, pid=1, last_seq=7, state="IN_FLIGHT", at_s=10.0)
        watchdog.record(rank=1, pid=2, last_seq=7, state="IN_FLIGHT", at_s=10.0)
        decision = watchdog.evaluate(now_s=20.0)
        assert decision["action"] == "abort_and_restart"
        assert decision["stale_ranks"] == [0, 1]

    def test_delayed_rank_ladder_separates_false_positives(self):
        ladder = sq.delayed_rank_ladder(1.0)
        assert {case["case"] for case in ladder} >= {"below_threshold", "just_above"}
        near = sq.evaluate_delayed_case(delay_s=0.9, threshold_s=1.0, observed="completes")
        assert near["ok"] is True and near["false_positive"] is False
        false_positive = sq.evaluate_delayed_case(
            delay_s=0.9, threshold_s=1.0, observed="bounded failure"
        )
        assert false_positive["false_positive"] is True


@pytest.mark.unit
class TestPropagationAndCleanup:
    def test_error_propagation_requires_every_rank(self):
        channel = sq.ErrorPropagationChannel(world_size=3)
        channel.report(rank=0, error_class="MISMATCH", group_id="tp", collective_seq=4)
        assert channel.missing_ranks() == [1, 2]
        assert channel.complete is False
        channel.report(rank=1, error_class="TIMEOUT", group_id="tp", collective_seq=4)
        channel.report(rank=2, error_class="TIMEOUT", group_id="tp", collective_seq=4)
        assert channel.complete is True

    def test_cleanup_order_is_the_reverse_dependency_order(self):
        plan = sq.cleanup_plan(["world", "tp", "ep"], dependency_order=["world", "tp", "ep"])
        assert plan["order"] == ["ep", "tp", "world"]

    def test_cleanup_rejects_unknown_group(self):
        with pytest.raises(ConfigError):
            sq.cleanup_plan(["tp"], dependency_order=["tp", "ep"])

    def test_tp_safety_gate_lists_blockers(self):
        gate = sq.tp_safety_gate(
            golden_ok=True, faults_bounded=False, cleanup_ok=True, recreate_ok=True
        )
        assert gate["ready"] is False and gate["blockers"]


@pytest.mark.unit
class TestFaultOracle:
    def _oracle(self) -> ft.FaultOracle:
        return ft.oracle_for_fault(
            "op_mismatch", maximum_detection_time_s=5.0, maximum_global_abort_time_s=10.0
        )

    def _times(self) -> ft.TimeMetrics:
        return ft.TimeMetrics(
            fault_injected_s=0.0, first_local_detection_s=0.5, global_failure_decision_s=1.0,
            all_ranks_abort_started_s=1.2, last_rank_exited_or_clean_s=1.5,
            resources_reclaimed_s=2.0, restart_started_s=2.5, ready_s=4.0,
            first_healthy_request_done_s=5.0,
        )

    def _good(self) -> ft.FailureObservation:
        return ft.FailureObservation(
            per_rank_terminal_state={0: "ABORTED", 1: "ABORTED"},
            first_error_rank=0,
            detected_at_layer="init_timeout_or_preflight",
            normalized_error="hqsb.collective.mismatch",
            recovery_level_used="COMMUNICATOR_RECREATE",
        )

    def test_correct_failure_passes(self):
        assert ft.evaluate_fault(self._oracle(), self._good(), self._times(), world_size=2).ok

    def test_reuse_after_error_fails(self):
        observation = ft.FailureObservation(
            per_rank_terminal_state={0: "ABORTED", 1: "ABORTED"},
            detected_at_layer="init_timeout_or_preflight",
            normalized_error="hqsb.collective.mismatch",
            recovery_level_used="COMMUNICATOR_RECREATE",
            communicator_reused_after_error=True,
        )
        verdict = ft.evaluate_fault(self._oracle(), observation, self._times(), world_size=2)
        assert verdict.ok is False
        assert any("reused" in failure for failure in verdict.failures)

    def test_partial_output_consumption_fails(self):
        observation = ft.FailureObservation(
            per_rank_terminal_state={0: "ABORTED", 1: "ABORTED"},
            detected_at_layer="init_timeout_or_preflight",
            normalized_error="hqsb.collective.mismatch",
            recovery_level_used="COMMUNICATOR_RECREATE",
            output_consumed_after_error=True,
        )
        assert ft.evaluate_fault(self._oracle(), observation, self._times(), world_size=2).ok is False

    def test_weak_recovery_level_fails(self):
        observation = ft.FailureObservation(
            per_rank_terminal_state={0: "ABORTED", 1: "ABORTED"},
            detected_at_layer="init_timeout_or_preflight",
            normalized_error="hqsb.collective.mismatch",
            recovery_level_used="REQUEST_ABORT_ONLY",
        )
        verdict = ft.evaluate_fault(self._oracle(), observation, self._times(), world_size=2)
        assert any("weaker" in failure for failure in verdict.failures)

    def test_detection_bound_is_enforced(self):
        slow = ft.TimeMetrics(
            fault_injected_s=0.0, first_local_detection_s=30.0, global_failure_decision_s=31.0,
            all_ranks_abort_started_s=32.0, last_rank_exited_or_clean_s=33.0,
            resources_reclaimed_s=34.0, restart_started_s=35.0, ready_s=36.0,
            first_healthy_request_done_s=37.0,
        )
        verdict = ft.evaluate_fault(self._oracle(), self._good(), slow, world_size=2)
        assert any("detection took" in failure for failure in verdict.failures)

    def test_metadata_detectable_error_may_not_surface_as_timeout(self):
        oracle = ft.oracle_for_fault(
            "dtype_mismatch", maximum_detection_time_s=5.0, maximum_global_abort_time_s=10.0
        )
        observation = ft.FailureObservation(
            per_rank_terminal_state={0: "ABORTED", 1: "ABORTED"},
            detected_at_layer="backend_async_error",
            normalized_error="hqsb.collective.mismatch",
            recovery_level_used="COMMUNICATOR_RECREATE",
        )
        verdict = ft.evaluate_fault(oracle, observation, self._times(), world_size=2)
        assert any("preflight was not used" in failure for failure in verdict.failures)

    def test_unknown_fault_is_refused(self):
        with pytest.raises(ConfigError):
            ft.oracle_for_fault("not_in_matrix", maximum_detection_time_s=1, maximum_global_abort_time_s=1)


@pytest.mark.unit
class TestSafetyAndRecovery:
    def _scope(self, **overrides) -> ft.SafetyScope:
        payload = {
            "experiment_nodes": ("n0",),
            "experiment_processes": (101, 102),
            "devices": ("GPU-0", "GPU-1"),
            "blast_radius": "one dedicated job on n0",
            "dedicated_job": True,
            "non_experiment_pids_protected": True,
            "shared_network_untouched": True,
        }
        payload.update(overrides)
        return ft.SafetyScope(**payload)

    def test_scope_requires_blast_radius_and_protections(self):
        self._scope().validate()
        with pytest.raises(ConfigError):
            self._scope(blast_radius="").validate()
        with pytest.raises(ConfigError):
            self._scope(non_experiment_pids_protected=False).validate()

    def test_network_fault_needs_an_approval_reference(self):
        with pytest.raises(ConfigError):
            self._scope(networks=("eth0",), approval_reference="").validate()
        self._scope(networks=("eth0",), approval_reference="CHG-1").validate()

    def test_injection_needs_markers_and_cleanup(self):
        with pytest.raises(ConfigError):
            ft.FaultInjection(fault="sigkill", target_rank=0, start_marker="", end_marker="x", cleanup="c")
        injection = ft.FaultInjection(
            fault="sigkill", target_rank=0, start_marker="s", end_marker="e", cleanup="c"
        )
        assert injection.validate_against_scope(self._scope()) == []

    def test_network_injection_requires_isolation_or_a_mock_label(self):
        injection = ft.FaultInjection(
            fault="inflight_network_error", target_rank=0, start_marker="s", end_marker="e",
            cleanup="c", requires_isolation=True,
        )
        assert injection.validate_against_scope(self._scope())
        mocked = ft.FaultInjection(
            fault="inflight_network_error", target_rank=0, start_marker="s", end_marker="e",
            cleanup="c", requires_isolation=True, partially_validated=True,
            mock_reason="no isolated network; mock disconnect",
        )
        assert mocked.validate_against_scope(self._scope()) == []

    def test_recovery_level_ordering(self):
        assert ft.RECOVERY_LEVELS.index("PROCESS_GROUP_RESTART") > ft.RECOVERY_LEVELS.index(
            "REQUEST_ABORT_ONLY"
        )
        assert set(ft.MINIMUM_RECOVERY_BY_FAULT) == {entry.fault for entry in ft.FAULT_MATRIX}

    def test_retryability_after_commit(self):
        assert ft.retryability_after_fault(
            commit_state="generated", recovery_level="REQUEST_ABORT_ONLY"
        )["decision"] == "safe_to_retry_before_commit"
        assert ft.retryability_after_fault(
            commit_state="emitted", recovery_level="PROCESS_GROUP_RESTART"
        )["decision"] == "unsafe_after_commit"
        with pytest.raises(ConfigError):
            ft.retryability_after_fault(commit_state="nope", recovery_level="REQUEST_ABORT_ONLY")


@pytest.mark.unit
class TestResourceClosure:
    def test_allocator_cache_is_not_a_leak(self):
        before = ft.ResourceSnapshot(label="before", device_memory_bytes={"GPU-0": 1000})
        after = ft.ResourceSnapshot(
            label="after", device_memory_bytes={"GPU-0": 1100}, allocator_cached_bytes=100
        )
        delta = ft.snapshot_delta(before, after, limit_bytes=10)
        assert delta["kind"] == "allocator_cache"
        assert delta["within_limit"] is True

    def test_unexplained_growth_is_a_leak_candidate(self):
        before = ft.ResourceSnapshot(label="before", device_memory_bytes={"GPU-0": 1000})
        after = ft.ResourceSnapshot(
            label="after", device_memory_bytes={"GPU-0": 2000}, process_count=2
        )
        delta = ft.snapshot_delta(before, after, limit_bytes=10)
        assert delta["kind"] == "leak_candidate"
        assert delta["count_deltas"]["process_count"] == 2

    def test_timeseries_must_be_monotonic(self):
        with pytest.raises(ConfigError):
            ft.TimeMetrics(
                fault_injected_s=0.0, first_local_detection_s=5.0, global_failure_decision_s=1.0,
                all_ranks_abort_started_s=2.0, last_rank_exited_or_clean_s=3.0,
                resources_reclaimed_s=4.0, restart_started_s=5.0, ready_s=6.0,
                first_healthy_request_done_s=7.0,
            ).validate()

    def test_boundedness_summary_separates_boundaries(self):
        verdicts = [
            {"fault": "op_mismatch", "ok": True, "mttd_s": 0.5, "abort_time_s": 1.0,
             "mttr_s": 5.0, "recovery_level_used": "COMMUNICATOR_RECREATE"},
            {"fault": "inflight_network_error", "ok": True, "mttd_s": 1.0, "abort_time_s": 2.0,
             "mttr_s": 9.0, "recovery_level_used": "PROCESS_GROUP_RESTART",
             "partially_validated": True},
        ]
        summary = ft.boundedness_summary(verdicts)
        assert summary["boundaries"]["not_really_verified"] == ["inflight_network_error"]
        assert summary["faults"][0]["runs"] == 1

    def test_stop_conditions_trigger(self):
        decision = ft.should_stop_campaign({"affects_non_experiment": True})
        assert decision["stop"] is True and decision["reasons"]

    def test_abort_coordinator_is_epoch_scoped(self):
        coordinator = ft.AbortCoordinator(rank_epoch=0, world_size=2)
        first = coordinator.decide(first_error_rank=0, error_class="MISMATCH", at_s=1.0)
        second = coordinator.decide(first_error_rank=1, error_class="TIMEOUT", at_s=2.0)
        assert first["decision"] == "global_abort"
        assert second["decision"] == "already_aborted"
        assert coordinator.terminal_states_complete({0: "ABORTED", 1: "ABORTED"}) is True
        assert coordinator.terminal_states_complete({0: "ABORTED"}) is False
