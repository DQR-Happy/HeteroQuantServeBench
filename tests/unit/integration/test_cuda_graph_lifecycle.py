"""CUDA Graph contracts and lifecycle/resource accounting (E06-08/E06-10)."""

from __future__ import annotations

import gc

import pytest

from hqsb.core.errors import CapabilityError, ConfigError
from hqsb.integration import cuda_graph, lifecycle


@pytest.mark.unit
class TestOutputLifetime:
    def test_reused_buffer_is_unsafe_while_held(self):
        handle = cuda_graph.OutputHandle(
            contract=cuda_graph.OutputContract.REUSED_BUFFER, buffer_id="out"
        )
        handle.acquire()
        assert handle.safe_to_replay is False
        handle.release()
        assert handle.safe_to_replay is True

    def test_owned_copy_is_always_safe(self):
        handle = cuda_graph.OutputHandle(
            contract=cuda_graph.OutputContract.OWNED_COPY, buffer_id="out"
        )
        handle.acquire()
        assert handle.safe_to_replay is True

    def test_unknown_contract_refused(self):
        with pytest.raises(ConfigError):
            cuda_graph.OutputHandle(contract="MAYBE", buffer_id="out")


@pytest.mark.unit
class TestStaticBufferPool:
    def _buffer(self, name="inp", address=1000, kind=cuda_graph.BufferKind.INPUT):
        return cuda_graph.StaticBuffer(
            name=name,
            kind=kind,
            shape=(1, 64),
            stride=(64, 1),
            dtype="float16",
            address=address,
            owner="graph_manager",
        )

    def test_addresses_are_fixed(self):
        pool = cuda_graph.StaticBufferPool()
        pool.allocate(self._buffer())
        with pytest.raises(ConfigError):
            pool.allocate(self._buffer(address=2000))
        assert pool.addresses == {"inp": 1000}

    def test_in_flight_buffer_cannot_be_released(self):
        pool = cuda_graph.StaticBufferPool()
        pool.allocate(self._buffer())
        pool.mark_in_flight("inp")
        with pytest.raises(CapabilityError):
            pool.release("inp")
        pool.complete("inp")
        pool.release("inp")
        assert pool.addresses == {}

    def test_total_bytes_accounting(self):
        pool = cuda_graph.StaticBufferPool()
        pool.allocate(self._buffer())
        pool.allocate(self._buffer(name="out", address=2000, kind=cuda_graph.BufferKind.OUTPUT))
        assert pool.total_bytes == 256

    def test_invalid_buffer_refused(self):
        with pytest.raises(ConfigError):
            cuda_graph.StaticBuffer(
                name="b",
                kind="magic",
                shape=(1,),
                stride=(1,),
                dtype="float16",
                address=1,
                owner="o",
            )
        with pytest.raises(ConfigError):
            cuda_graph.StaticBuffer(
                name="b",
                kind=cuda_graph.BufferKind.INPUT,
                shape=(1,),
                stride=(1,),
                dtype="float16",
                address=0,
                owner="o",
            )
        with pytest.raises(ConfigError):
            cuda_graph.StaticBuffer(
                name="b",
                kind=cuda_graph.BufferKind.INPUT,
                shape=(1, 2),
                stride=(1,),
                dtype="float16",
                address=1,
                owner="o",
            )


@pytest.mark.unit
class TestEligibility:
    def test_empty_observation_is_not_eligible(self):
        report = cuda_graph.evaluate_eligibility(cuda_graph.EligibilityObservation())
        assert not report.ok
        assert "side_stream_warmup" in report.failures
        assert len(report.failures) == len(cuda_graph.ELIGIBILITY_CHECKS) - 1

    def test_full_observation_is_eligible(self):
        report = cuda_graph.evaluate_eligibility(
            cuda_graph.EligibilityObservation(
                side_stream_warmup=True,
                capture_stream_explicit=True,
                no_cpu_gpu_sync=True,
                no_unsupported_alloc=True,
                current_stream_preserved=True,
                rng_graph_safe=True,
                addresses_fixed=True,
                lifetimes_long_enough=True,
                no_data_dependent_control_flow=True,
                unsupported_identified_before_capture=True,
            )
        )
        assert report.ok
        assert report.failures == ()


@pytest.mark.unit
class TestGraphSpecAndBuckets:
    def _spec(self) -> cuda_graph.GraphSpec:
        return cuda_graph.GraphSpec(
            capture_scope="decode_step_block",
            buckets=(
                cuda_graph.BucketSpec(name="b1s1", dims={"batch": 1, "sequence": 1}),
                cuda_graph.BucketSpec(name="b1s32", dims={"batch": 1, "sequence": 32}),
            ),
            output_contract=cuda_graph.OutputContract.REUSED_BUFFER,
        )

    def test_exact_bucket_match(self):
        plan = self._spec().resolve({"batch": 1, "sequence": 1})
        assert plan.exact_match is True
        assert plan.bucket is not None

    def test_out_of_bucket_falls_back(self):
        plan = self._spec().resolve({"batch": 4, "sequence": 512})
        assert plan.exact_match is False
        assert plan.fallback_reason == "OUT_OF_BUCKET"
        assert plan.policy == cuda_graph.OutOfBucketPolicy.FALLBACK_NON_GRAPH

    def test_unknown_policy_refused(self):
        with pytest.raises(ConfigError):
            cuda_graph.GraphSpec(capture_scope="x", out_of_bucket_policy="pray")

    def test_timing_must_state_what_is_counted(self):
        with pytest.raises(ConfigError):
            cuda_graph.GraphSpec(
                capture_scope="x", includes_input_copy=False, includes_output_copy=False
            )


@pytest.mark.unit
class TestReplay:
    def test_distinctness_report(self):
        records = (
            cuda_graph.ReplayRecord("g", 0, "in-a", "out-a"),
            cuda_graph.ReplayRecord("g", 1, "in-b", "out-b"),
        )
        report = cuda_graph.replay_distinctness(records)
        assert report["distinct_inputs"] == 2
        assert report["replays"] == 2

    def test_break_even_replays(self):
        timing = cuda_graph.ReplayTiming(
            capture_ms=100.0,
            instantiate_ms=20.0,
            steady_replay_ms=5.0,
            copy_in_ms=1.0,
            copy_out_ms=1.0,
        )
        # fixed 120 ms / saving (20 - 7) = 13 ms ⇒ 10 replays (rounded up)
        assert timing.break_even_replays(eager_ms=20.0) == 10

    def test_break_even_absent_when_copies_dominate(self):
        timing = cuda_graph.ReplayTiming(
            capture_ms=100.0, steady_replay_ms=5.0, copy_in_ms=20.0, copy_out_ms=20.0
        )
        assert timing.break_even_replays(eager_ms=20.0) is None


@pytest.mark.unit
class TestFailureMatrix:
    def test_every_case_has_an_expectation(self):
        matrix = cuda_graph.failure_matrix()
        assert len(matrix) == len(cuda_graph.FAILURE_CASES)
        assert all(item.expected_action == cuda_graph.OutOfBucketPolicy.FALLBACK_NON_GRAPH for item in matrix)

    def test_partial_capture_is_a_failure(self):
        observations = (
            cuda_graph.FailureObservation(
                case="shape_change",
                stage="pre_capture",
                captured_partially=True,
                reused_old_graph=False,
                fell_back_to_non_graph=True,
            ),
        )
        report = cuda_graph.evaluate_failure_matrix(observations)
        assert not report["ok"]
        assert report["failing"][0]["case"] == "shape_change"

    def test_old_graph_reuse_is_a_failure(self):
        observations = (
            cuda_graph.FailureObservation(
                case="wrong_stream",
                stage="capture",
                captured_partially=False,
                reused_old_graph=True,
                fell_back_to_non_graph=True,
            ),
        )
        report = cuda_graph.evaluate_failure_matrix(observations)
        assert not report["ok"]

    def test_missing_cases_are_not_ok(self):
        report = cuda_graph.evaluate_failure_matrix(())
        assert not report["ok"]
        assert len(report["failing"]) == len(cuda_graph.FAILURE_CASES)

    def test_clean_matrix_passes(self):
        observations = tuple(
            cuda_graph.FailureObservation(
                case=expectation.case,
                stage=expectation.expected_stage,
                captured_partially=False,
                reused_old_graph=False,
                fell_back_to_non_graph=True,
            )
            for expectation in cuda_graph.failure_matrix()
        )
        assert cuda_graph.evaluate_failure_matrix(observations)["ok"] is True


@pytest.mark.unit
class TestClaimGate:
    def test_unexecuted_p1_is_not_run(self):
        report = cuda_graph.claim_status(cuda_graph.ClaimEvidence(), executed=False)
        assert report["status"] == cuda_graph.ClaimStatus.NOT_RUN
        assert "must not be claimed" in report["reason"]

    def test_partial_evidence_is_not_claimed(self):
        report = cuda_graph.claim_status(
            cuda_graph.ClaimEvidence(graph_safe_scope=True), executed=True
        )
        assert report["status"] == cuda_graph.ClaimStatus.NOT_CLAIMED
        assert report["missing"]

    def test_complete_evidence_can_be_claimed(self):
        evidence = cuda_graph.ClaimEvidence(
            graph_safe_scope=True,
            multi_input_replay_correct=True,
            long_replay_correct=True,
            lifetime_safe=True,
            stream_dependencies_ok=True,
            bucket_fallback_defined=True,
            failure_matrix_clean=True,
            pool_explained=True,
            timing_decomposed=True,
            end_to_end_benefit=True,
            break_even_bound=True,
        )
        report = cuda_graph.claim_status(evidence, executed=True)
        assert report["status"] == cuda_graph.ClaimStatus.CLAIMED


@pytest.mark.unit
class TestLifecycleMachine:
    def test_legal_path(self):
        machine = lifecycle.LifecycleMachine(name="compiled")
        for state in (
            lifecycle.LifecycleState.REGISTERED,
            lifecycle.LifecycleState.MODEL_ATTACHED,
            lifecycle.LifecycleState.COMPILED,
            lifecycle.LifecycleState.ACTIVE,
            lifecycle.LifecycleState.DISABLED,
            lifecycle.LifecycleState.CLOSED,
            lifecycle.LifecycleState.COLLECTED,
        ):
            machine.transition(state)
        assert machine.path[-1] == lifecycle.LifecycleState.COLLECTED
        assert len(machine.transitions) == 7

    def test_illegal_transition_refused(self):
        machine = lifecycle.LifecycleMachine(name="compiled")
        machine.transition(lifecycle.LifecycleState.REGISTERED)
        with pytest.raises(ConfigError):
            machine.transition(lifecycle.LifecycleState.ACTIVE)

    def test_unknown_state_refused(self):
        with pytest.raises(ConfigError):
            lifecycle.LifecycleMachine(name="x", state="SOMEWHERE")


@pytest.mark.unit
class TestResourceSpecs:
    def test_frozen_specs_have_owners_and_caps(self):
        specs = lifecycle.frozen_resource_specs()
        assert len(specs) >= 10
        for spec in specs:
            assert spec.owner
            assert spec.create_span and spec.destroy_span
        capped = [spec for spec in specs if spec.cache_cap_entries]
        assert capped

    def test_missing_owner_refused(self):
        with pytest.raises(ConfigError):
            lifecycle.ResourceSpec(
                name="x",
                resource_class=lifecycle.ResourceClass.DEVICE,
                owner="",
                create_span="a",
                destroy_span="b",
            )

    def test_unknown_class_refused(self):
        with pytest.raises(ConfigError):
            lifecycle.ResourceSpec(
                name="x", resource_class="quantum", owner="o", create_span="a", destroy_span="b"
            )


@pytest.mark.unit
class TestLeakStatistics:
    def _series(self, values):
        return tuple(
            lifecycle.ResourceSnapshot(cycle=index, allocated_bytes=value)
            for index, value in enumerate(values)
        )

    def test_flat_series_is_steady(self):
        spec = lifecycle.ResourceSpec(
            name="workspace",
            resource_class=lifecycle.ResourceClass.DEVICE,
            owner="runtime",
            create_span="a",
            destroy_span="b",
        )
        report = lifecycle.evaluate_leak(spec, self._series([100, 100, 100, 100, 100]), warmup_cycles=1)
        assert report.verdict in (lifecycle.LeakVerdict.STEADY, lifecycle.LeakVerdict.BOUNDED_CACHE)

    def test_growing_series_is_detected(self):
        spec = lifecycle.ResourceSpec(
            name="workspace",
            resource_class=lifecycle.ResourceClass.DEVICE,
            owner="runtime",
            create_span="a",
            destroy_span="b",
        )
        report = lifecycle.evaluate_leak(
            spec, self._series([100, 200, 300, 400, 500]), warmup_cycles=0, steady_cycles=5
        )
        assert report.verdict == lifecycle.LeakVerdict.GROWING
        assert report.slope.crosses_zero is False

    def test_insufficient_data_is_reported_not_guessed(self):
        spec = lifecycle.ResourceSpec(
            name="workspace",
            resource_class=lifecycle.ResourceClass.DEVICE,
            owner="runtime",
            create_span="a",
            destroy_span="b",
        )
        report = lifecycle.evaluate_leak(spec, self._series([100, 100]), warmup_cycles=0, steady_cycles=2)
        assert report.verdict == lifecycle.LeakVerdict.INSUFFICIENT_DATA

    def test_plateau_is_detected(self):
        series = self._series([100, 200, 300, 300, 300, 300])
        point = lifecycle.detect_plateau(series, "allocated_bytes", tolerance=0.0)
        assert point is not None
        assert point.kind == "plateau"

    def test_segmentation_keeps_teardown_separate(self):
        series = self._series([1, 2, 3, 4, 5, 6])
        segments = lifecycle.segment_snapshots(series, warmup_cycles=1, steady_cycles=3)
        assert len(segments["warmup"]) == 1
        assert len(segments["steady"]) == 3
        assert len(segments["teardown"]) == 2

    def test_bad_segmentation_refused(self):
        with pytest.raises(ConfigError):
            lifecycle.segment_snapshots(self._series([1, 2]), warmup_cycles=-1, steady_cycles=1)

    def test_unknown_metric_refused(self):
        with pytest.raises(ConfigError):
            lifecycle.robust_slope(self._series([1, 2, 3]), "vibes")


@pytest.mark.unit
class TestLivenessAndStreams:
    def test_liveness_probe_detects_collection(self):
        class Payload:
            pass

        probe = lifecycle.LivenessProbe(name="artifact")
        payload = Payload()
        probe.track("payload", payload)
        assert probe.alive() == ("payload",)
        del payload
        gc.collect()
        assert probe.alive() == ()
        assert probe.dead() == ("payload",)

    def test_probe_does_not_keep_the_object_alive(self):
        class Payload:
            pass

        probe = lifecycle.liveness_factory(Payload)
        assert probe.alive() == ()

    def test_stream_audit_flags_global_sync(self):
        events = (
            lifecycle.StreamEvent(name="sync", kind="sync"),
            lifecycle.StreamEvent(name="sync", kind="sync"),
        )
        report = lifecycle.audit_streams(events, expected_calls=1)
        assert report["implicit_global_sync"] is True
        assert not report["ok"]

    def test_stream_audit_flags_unbalanced_events(self):
        events = (
            lifecycle.StreamEvent(name="e", kind="create", created=True),
            lifecycle.StreamEvent(name="e", kind="record"),
        )
        report = lifecycle.audit_streams(events)
        assert report["unbalanced_events"] == 1
        assert not report["ok"]

    def test_balanced_events_pass(self):
        events = (
            lifecycle.StreamEvent(name="e", kind="create", created=True),
            lifecycle.StreamEvent(name="e", kind="destroy", destroyed=True),
        )
        assert lifecycle.audit_streams(events)["ok"] is True


@pytest.mark.unit
class TestEnableDisableAndTeardown:
    def test_disable_must_really_disable(self):
        stale = lifecycle.EnableDisableAudit(
            request_index=0,
            mode="disabled",
            actual_backend="hqsb.compiled.fused",
            old_callable_reachable=True,
        )
        assert not stale.ok
        report = lifecycle.audit_enable_disable((stale,))
        assert not report["ok"]

    def test_enable_requires_a_rebuilt_wrapper(self):
        record = lifecycle.EnableDisableAudit(
            request_index=1,
            mode="enabled",
            actual_backend="hqsb.compiled.fused",
            restore_ok=False,
        )
        assert not record.ok

    def test_teardown_refuses_release_while_in_flight(self):
        plan = lifecycle.TeardownPlan(wait_in_flight=False)
        result = plan.validate(in_flight=3)
        assert not result["ok"]
        assert "RELEASED_WHILE_IN_FLIGHT" in result["problems"]

    def test_teardown_with_wait_is_ok(self):
        assert lifecycle.TeardownPlan().validate(in_flight=2)["ok"] is True

    def test_post_close_must_fail_per_contract(self):
        expectation = lifecycle.PostCloseExpectation(
            name="compiled_callable", expected_error="RuntimeError: closed"
        )
        assert not expectation.ok
        invoked = lifecycle.PostCloseExpectation(
            name="compiled_callable",
            expected_error="RuntimeError: closed",
            observed_error="RuntimeError: closed",
            invoked=True,
        )
        assert invoked.ok
