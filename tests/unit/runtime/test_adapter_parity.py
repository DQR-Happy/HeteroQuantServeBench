"""E07-01 adapter contract, probing and semantic parity oracles."""

from __future__ import annotations

import pytest

from hqsb.core.errors import BackendError, CapabilityError, ConfigError
from hqsb.runtime import adapter as A
from hqsb.runtime import parity as P
from hqsb.runtime import request as R


def _spec(role: str = "reference", backend_id: str = "smoke") -> R.BackendSpec:
    return R.BackendSpec(
        backend_id=backend_id,
        role=role,
        version="1.0.0",
        commit="c0",
        source_identity="src",
        adapter_module="hqsb.runtime.adapter",
    )


def _identity() -> R.ModelIdentity:
    return R.ModelIdentity(
        model_id="Qwen/Qwen3-1.7B",
        model_manifest_sha256="a" * 64,
        revision="rev-1",
        tokenizer_id="Qwen/Qwen3-1.7B",
        chat_template_hash="tpl",
        precision="float16",
    )


def _request(**overrides) -> R.RequestSpec:
    payload = {
        "request_id": "r0",
        "identity": _identity(),
        "input_token_ids": (1, 2, 3, 4),
        "sampling": R.SamplingSpec(mode="greedy"),
        "stop": R.StopSpec(max_new_tokens=4),
    }
    payload.update(overrides)
    return R.RequestSpec(**payload)


def _result(**overrides) -> A.GenerationResult:
    payload = {
        "request_id": "r0",
        "token_ids": (11, 12, 13),
        "finish_reason": "length",
    }
    payload.update(overrides)
    return A.GenerationResult(**payload)


@pytest.mark.unit
class TestProbe:
    def test_candidate_engines_are_probed_without_importing_them(self):
        probes = A.probe_environment()
        assert {probe.engine for probe in probes} == set(A.CANDIDATE_ENGINES)
        for probe in probes:
            assert probe.state in (
                A.PROBE_NOT_INSTALLED,
                A.PROBE_INSTALLED_NOT_VERIFIED,
                A.PROBE_AVAILABLE,
                A.PROBE_UNKNOWN,
            )

    def test_an_unknown_engine_is_not_installed(self):
        probe = A.probe_environment(("definitely_not_a_real_engine",))[0]
        assert probe.state == A.PROBE_NOT_INSTALLED
        assert not probe.usable

    def test_primary_selection_needs_one_verified_engine(self):
        selection = A.select_primary_engine(
            [A.AdapterProbeResult("vllm", A.PROBE_NOT_INSTALLED, "vllm")]
        )
        assert not selection["ok"]
        assert selection["selected"] == ""

    def test_primary_selection_reports_ambiguity(self):
        selection = A.select_primary_engine(
            [
                A.AdapterProbeResult("vllm", A.PROBE_AVAILABLE, "vllm"),
                A.AdapterProbeResult("sglang", A.PROBE_AVAILABLE, "sglang"),
            ]
        )
        assert not selection["ok"]


@pytest.mark.unit
class TestDummyAdapter:
    def test_lifecycle_transitions_are_enforced(self):
        dummy = A.DummyRuntimeAdapter(_spec())
        with pytest.raises(BackendError):
            dummy.generate(_request())
        dummy.load(_identity())
        with pytest.raises(BackendError):
            dummy.transition(A.AdapterState.CLOSED)

    def test_load_verifies_identity(self):
        dummy = A.DummyRuntimeAdapter(_spec())
        report = dummy.load(_identity())
        assert dummy.verify_identity(_identity(), report)["ok"]
        with pytest.raises(BackendError):
            dummy.verify_identity(_identity(), {**report, "revision": "other"})

    def test_close_is_idempotent(self):
        dummy = A.DummyRuntimeAdapter(_spec())
        dummy.load(_identity())
        first = dummy.close()
        second = dummy.close()
        assert first["closed"] and second["already_closed"]

    def test_generate_after_close_is_refused(self):
        dummy = A.DummyRuntimeAdapter(_spec())
        dummy.load(_identity())
        dummy.warmup(_request())
        dummy.close()
        with pytest.raises(BackendError):
            dummy.generate(_request())

    def test_performance_claims_are_never_allowed(self):
        assert not A.DummyRuntimeAdapter(_spec()).performance_claim_allowed()["allowed"]

    def test_streaming_is_refused_with_a_structured_reason(self):
        dummy = A.DummyRuntimeAdapter(_spec())
        with pytest.raises(CapabilityError):
            dummy.stream(_request())

    def test_scenario_matrix_covers_success_and_failure_paths(self):
        results = A.run_load_close_scenarios(lambda: A.DummyRuntimeAdapter(_spec()))
        cases = {item["case"] for item in results}
        assert cases == {item["case"] for item in A.load_close_scenarios()}
        assert all(item["ok"] for item in results)


@pytest.mark.unit
class TestRegistry:
    def test_duplicate_registration_refused(self):
        registry = A.AdapterRegistry()
        registry.register(
            A.AdapterRegistration(name="reference", factory="f", spec=_spec())
        )
        with pytest.raises(ConfigError):
            registry.register(
                A.AdapterRegistration(name="reference", factory="g", spec=_spec())
            )

    def test_unknown_adapter_refused(self):
        with pytest.raises(CapabilityError):
            A.AdapterRegistry().resolve("vllm")

    def test_unusable_probe_blocks_resolution(self):
        registry = A.AdapterRegistry()
        registry.register(
            A.AdapterRegistration(
                name="vllm",
                factory="f",
                spec=_spec(role="cloud_primary", backend_id="vllm"),
                engine="vllm",
                probe=A.AdapterProbeResult("vllm", A.PROBE_NOT_INSTALLED, "vllm"),
            )
        )
        with pytest.raises(CapabilityError):
            registry.resolve("vllm")


@pytest.mark.unit
class TestStreamAndCancelContracts:
    def test_stream_order_and_final_flag_checked(self):
        chunks = (
            A.StreamChunk(request_id="r0", token_index=0, token_id=11, timestamp_ns=1),
            A.StreamChunk(request_id="r0", token_index=1, token_id=12, timestamp_ns=2, final=True),
        )
        assert A.validate_stream(chunks, request_id="r0")["ok"]

    def test_duplicate_index_is_reported(self):
        chunks = (
            A.StreamChunk(request_id="r0", token_index=0, token_id=11, timestamp_ns=1),
            A.StreamChunk(request_id="r0", token_index=0, token_id=12, timestamp_ns=2, final=True),
        )
        report = A.validate_stream(chunks, request_id="r0")
        assert not report["ok"]

    def test_missing_final_flag_is_reported(self):
        chunks = (A.StreamChunk(request_id="r0", token_index=0, token_id=11, timestamp_ns=1),)
        assert not A.validate_stream(chunks, request_id="r0")["ok"]

    def test_wrong_request_id_is_reported(self):
        chunks = (
            A.StreamChunk(request_id="r1", token_index=0, token_id=11, timestamp_ns=1, final=True),
        )
        assert not A.validate_stream(chunks, request_id="r0")["ok"]

    def test_cancel_requires_an_in_flight_policy(self):
        with pytest.raises(ConfigError):
            A.CancelRecord(
                request_id="r0",
                t_requested_ns=0,
                t_observed_ns=1,
                t_done_ns=2,
                in_flight_policy="unspecified",
            )

    def test_cancel_observation_latency(self):
        record = A.CancelRecord(
            request_id="r0",
            t_requested_ns=0,
            t_observed_ns=1_000_000,
            t_done_ns=2_000_000,
            in_flight_policy="discard_output",
        )
        assert record.observation_latency_ms == pytest.approx(1.0)


@pytest.mark.unit
class TestGenerationResult:
    def test_unknown_finish_reason_refused(self):
        with pytest.raises(ConfigError):
            _result(finish_reason="timeout")

    def test_empty_stop_needs_an_explicit_flag(self):
        with pytest.raises(ConfigError):
            _result(token_ids=(), finish_reason="stop")
        allowed = _result(
            token_ids=(), finish_reason="stop", raw={"allow_empty_stop": True}
        )
        assert allowed.finish_reason == "stop"


@pytest.mark.unit
class TestParityOracles:
    def test_greedy_parity_localises_the_first_divergence(self):
        report = P.compare_greedy(_result(), _result(token_ids=(11, 12, 99)))
        assert not report.ok
        assert report.first_divergence_index == 2

    def test_greedy_parity_refuses_mismatched_request_ids(self):
        with pytest.raises(ConfigError):
            P.compare_greedy(_result(), _result(request_id="r1"))

    def test_logit_tolerance_is_reported(self):
        report = P.compare_greedy(
            _result(),
            _result(),
            tolerance=0.01,
            reference_logits=[[0.0, 1.0]],
            candidate_logits=[[0.0, 1.5]],
        )
        assert report.logits_available
        assert not report.logits_within_tolerance

    def test_single_seed_distribution_claim_is_refused(self):
        with pytest.raises(ConfigError):
            P.compare_sampling_distribution(
                request_id="r0",
                left_samples=[0.1],
                right_samples=[0.2],
                pre_registered_bound=0.05,
            )

    def test_too_few_seeds_yields_inconclusive(self):
        report = P.compare_sampling_distribution(
            request_id="r0",
            left_samples=[0.1, 0.2],
            right_samples=[0.1, 0.2],
            pre_registered_bound=0.05,
            min_seeds=4,
        )
        assert report.verdict == "INCONCLUSIVE"
        with pytest.raises(CapabilityError):
            P.assert_no_single_seed_claim(report)

    def test_distribution_comparison_with_enough_seeds_decides(self):
        report = P.compare_sampling_distribution(
            request_id="r0",
            left_samples=[1.0, 1.0, 1.0, 1.0],
            right_samples=[1.0, 1.0, 1.0, 1.0],
            pre_registered_bound=0.1,
        )
        assert report.verdict == "PASS"

    def test_streaming_comparison_detects_a_truncated_stream(self):
        reference = _result()
        chunks = (
            A.StreamChunk(request_id="r0", token_index=0, token_id=11, timestamp_ns=1, final=True),
        )
        assert not P.compare_streaming(reference, chunks)["ok"]

    def test_boundary_cases_are_pre_registered(self):
        cases = {case.case for case in P.boundary_cases()}
        assert {"eos_before_max", "max_tokens_hit", "zero_token", "min_tokens_protects_eos"} == cases


@pytest.mark.unit
class TestParameterAndUnsupportedMatrices:
    def _capability(self) -> R.CapabilityReport:
        report = R.CapabilityReport(backend_id="smoke")
        report.declare("precision", "SUPPORTED_EXACT")
        report.declare(
            "prefix_cache",
            "UNSUPPORTED_REJECT",
            reason="no KV store",
        )
        return report

    def test_every_claimable_field_must_appear(self):
        covered = P.parameter_effect_matrix(
            self._capability(),
            [
                P.ParameterEffect(
                    field="precision", before_value="float16", after_value="bfloat16", output_changed=True
                )
            ],
        )
        assert covered["ok"]
        assert covered["claimable_fields"] == ["precision"]
        uncovered = P.parameter_effect_matrix(self._capability(), [])
        assert not uncovered["ok"]
        assert uncovered["missing"] == ["precision"]

    def test_an_ineffective_field_is_rejected(self):
        matrix = P.parameter_effect_matrix(
            self._capability(),
            [
                P.ParameterEffect(
                    field="precision", before_value="a", after_value="b", output_changed=False
                )
            ],
        )
        assert not matrix["ok"]
        assert matrix["ineffective"] == ["precision"]

    def test_telemetry_only_effect_is_accepted_with_evidence(self):
        effect = P.ParameterEffect(
            field="precision",
            before_value="a",
            after_value="b",
            output_changed=False,
            telemetry_evidence="kernel name changed",
        )
        assert effect.verdict == "EFFECTIVE_TELEMETRY_ONLY"

    def test_unexpected_effect_is_flagged(self):
        effect = P.ParameterEffect(
            field="seed",
            before_value=1,
            after_value=2,
            output_changed=True,
            oracle_expectation="output_identical",
        )
        assert effect.verdict == "UNEXPECTED_EFFECT"
        assert not effect.ok

    def test_silent_ignore_fails_the_unsupported_matrix(self):
        matrix = P.unsupported_negative_matrix(
            self._capability(),
            [
                P.UnsupportedObservation(
                    field="prefix_cache",
                    declared_state="UNSUPPORTED_REJECT",
                    observed_behaviour="SILENT_IGNORE",
                )
            ],
        )
        assert not matrix["ok"]
        assert matrix["silent_or_unexplained"] == ["prefix_cache"]

    def test_explicit_rejection_passes(self):
        matrix = P.unsupported_negative_matrix(
            self._capability(),
            [
                P.UnsupportedObservation(
                    field="prefix_cache",
                    declared_state="UNSUPPORTED_REJECT",
                    observed_behaviour="REJECTED",
                    reason_exposed="no KV store",
                )
            ],
        )
        assert matrix["ok"]

    def test_parity_matrix_needs_two_backends(self):
        with pytest.raises(ConfigError):
            P.capability_parity_matrix([_request()], {"a": self._capability()})

    def test_parity_matrix_marks_unprobed_fields(self):
        rows = P.capability_parity_matrix(
            [_request()],
            {"a": self._capability(), "b": R.CapabilityReport(backend_id="b")},
        )
        streaming = [row for row in rows if row["field"] == "streaming"]
        assert streaming and all(row["state"] == "UNKNOWN" for row in streaming)

    def test_error_recovery_sequence_is_complete(self):
        steps = P.error_recovery_sequence()
        assert P.error_recovery_report(steps)["ok"]
        assert not P.error_recovery_report(steps[:-1])["ok"]
