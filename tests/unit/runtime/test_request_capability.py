"""E07-01 request semantics and capability negotiation."""

from __future__ import annotations

import pytest

from hqsb.core.errors import CapabilityError, ConfigError
from hqsb.runtime import request as R


def _identity(**overrides) -> R.ModelIdentity:
    payload = {
        "model_id": "Qwen/Qwen3-1.7B",
        "model_manifest_sha256": "a" * 64,
        "revision": "rev-1",
        "tokenizer_id": "Qwen/Qwen3-1.7B",
        "chat_template_hash": "tpl",
        "precision": "float16",
    }
    payload.update(overrides)
    return R.ModelIdentity(**payload)


def _request(**overrides) -> R.RequestSpec:
    payload = {
        "request_id": "r0",
        "identity": _identity(),
        "input_token_ids": (1, 2, 3, 4),
        "sampling": R.SamplingSpec(mode="greedy"),
        "stop": R.StopSpec(max_new_tokens=8),
    }
    payload.update(overrides)
    return R.RequestSpec(**payload)


def _capability(**states: str) -> R.CapabilityReport:
    report = R.CapabilityReport(backend_id="b0")
    for field, state in states.items():
        kwargs = {}
        if state in ("UNSUPPORTED_REJECT", "UNSUPPORTED_FALLBACK", "EMULATED", "UNKNOWN"):
            kwargs["reason"] = f"{field} is {state}"
        if state == "SUPPORTED_WITH_CONSTRAINT":
            kwargs["constraint"] = f"{field} limited"
        if state == "EMULATED":
            kwargs["emulated_cost_note"] = "measured in run"
        report.declare(field, state, **kwargs)
    return report


@pytest.mark.unit
class TestModelIdentityAndRequest:
    def test_identity_requires_every_provenance_field(self):
        with pytest.raises(ConfigError):
            R.ModelIdentity(
                model_id="m",
                model_manifest_sha256="",
                revision="r",
                tokenizer_id="t",
                chat_template_hash="c",
                precision="float16",
            )

    def test_request_rejects_empty_token_ids(self):
        with pytest.raises(ConfigError):
            _request(input_token_ids=())

    def test_request_rejects_negative_tokens(self):
        with pytest.raises(ConfigError):
            _request(input_token_ids=(1, -2))

    def test_request_hash_is_stable_and_identity_sensitive(self):
        first = _request()
        second = _request()
        assert first.request_hash == second.request_hash
        other = _request(identity=_identity(model_id="other/model"))
        assert first.request_hash != other.request_hash

    def test_input_token_hash_depends_on_order(self):
        left = _request(input_token_ids=(1, 2, 3))
        right = _request(input_token_ids=(3, 2, 1))
        assert left.input_token_hash != right.input_token_hash


@pytest.mark.unit
class TestSamplingAndStop:
    def test_greedy_requires_zero_temperature(self):
        with pytest.raises(ConfigError):
            R.SamplingSpec(mode="greedy", temperature=0.7)

    def test_sampling_requires_positive_temperature(self):
        with pytest.raises(ConfigError):
            R.SamplingSpec(mode="sampling", temperature=0.0)

    def test_unknown_mode_refused(self):
        with pytest.raises(ConfigError):
            R.SamplingSpec(mode="beam")

    def test_gateway_stop_layer_is_out_of_scope(self):
        with pytest.raises(ConfigError):
            R.StopSpec(max_new_tokens=4, stop_string_layer="gateway")

    def test_min_tokens_must_fit_max_tokens(self):
        with pytest.raises(ConfigError):
            R.StopSpec(max_new_tokens=2, min_new_tokens=5)


@pytest.mark.unit
class TestCapabilityReport:
    def test_unsupported_state_needs_a_reason(self):
        with pytest.raises(ConfigError):
            R.CapabilityField(name="prefix_cache", state="UNSUPPORTED_REJECT")

    def test_emulated_state_needs_measured_cost(self):
        with pytest.raises(ConfigError):
            R.CapabilityField(name="prefix_cache", state="EMULATED", reason="emulated")

    def test_constrained_state_needs_constraint(self):
        with pytest.raises(ConfigError):
            R.CapabilityField(
                name="precision", state="SUPPORTED_WITH_CONSTRAINT", reason=""
            )

    def test_unknown_field_declaration_refused(self):
        report = R.CapabilityReport(backend_id="b0")
        with pytest.raises(ConfigError):
            report.declare("not_a_request_field", "SUPPORTED_EXACT")

    def test_missing_probe_result_is_unknown_and_not_supported(self):
        report = _capability(model_artifact="SUPPORTED_EXACT")
        assert report.state_of("streaming") == R.UNKNOWN
        with pytest.raises(CapabilityError):
            report.require_supported("streaming")

    def test_rejected_field_cannot_be_required(self):
        report = _capability(prefix_cache="UNSUPPORTED_REJECT")
        with pytest.raises(CapabilityError):
            report.require_supported("prefix_cache")

    def test_missing_fields_lists_unprobed_entries(self):
        report = _capability(model_artifact="SUPPORTED_EXACT")
        missing = report.missing_fields()
        assert "streaming" in missing and "model_artifact" not in missing


@pytest.mark.unit
class TestResolvedConfig:
    def test_silent_parameter_change_refused(self):
        config = R.ResolvedConfig(backend_id="b0")
        with pytest.raises(ConfigError):
            config.record("max_new_tokens", 32, 16)
        config.record("max_new_tokens", 32, 16, reason="runtime cap")
        assert config.changed_parameters()[0].name == "max_new_tokens"

    def test_resolve_request_records_requested_and_actual(self):
        report = _capability(
            model_artifact="SUPPORTED_EXACT",
            input_token_ids="SUPPORTED_EXACT",
            max_new_tokens="SUPPORTED_EXACT",
            precision="SUPPORTED_EXACT",
            sampling_mode="SUPPORTED_EXACT",
        )
        resolved = R.resolve_request(_request(), report)
        assert resolved.changed_parameters() == []
        assert resolved.parameters["precision"].actual == "float16"

    def test_resolve_request_refuses_unprobed_streaming(self):
        report = _capability(
            model_artifact="SUPPORTED_EXACT",
            input_token_ids="SUPPORTED_EXACT",
            max_new_tokens="SUPPORTED_EXACT",
            precision="SUPPORTED_EXACT",
            sampling_mode="SUPPORTED_EXACT",
        )
        with pytest.raises(CapabilityError):
            R.resolve_request(_request(streaming=True), report)

    def test_sampling_fallback_is_explicit(self):
        report = _capability(
            model_artifact="SUPPORTED_EXACT",
            input_token_ids="SUPPORTED_EXACT",
            max_new_tokens="SUPPORTED_EXACT",
            precision="SUPPORTED_EXACT",
            sampling_mode="UNSUPPORTED_FALLBACK",
        )
        resolved = R.resolve_request(
            _request(sampling=R.SamplingSpec(mode="sampling", temperature=0.8)), report
        )
        changed = {item.name for item in resolved.changed_parameters()}
        assert changed == {"sampling_mode"}

    def test_sampling_rejected_raises(self):
        report = _capability(
            model_artifact="SUPPORTED_EXACT",
            input_token_ids="SUPPORTED_EXACT",
            max_new_tokens="SUPPORTED_EXACT",
            precision="SUPPORTED_EXACT",
            sampling_mode="UNSUPPORTED_REJECT",
        )
        with pytest.raises(CapabilityError):
            R.resolve_request(
                _request(sampling=R.SamplingSpec(mode="sampling", temperature=0.5)),
                report,
            )


@pytest.mark.unit
class TestBackendSpec:
    def test_exactly_one_primary_required(self):
        from hqsb.runtime import adapter as A

        base = dict(
            version="1.0.0",
            commit="c0",
            source_identity="src",
            adapter_module="hqsb.runtime.adapter",
        )
        with pytest.raises(ConfigError):
            R.assert_single_primary(
                [
                    R.BackendSpec(backend_id="reference", role="reference", **base),
                    R.BackendSpec(backend_id="edge", role="edge", **base),
                ]
            )
        selected = R.assert_single_primary(
            [
                R.BackendSpec(backend_id="reference", role="reference", **base),
                R.BackendSpec(backend_id="cloud", role="cloud_primary", **base),
            ]
        )
        assert selected == "cloud"
        assert A.ALLOWED_ADAPTER_TRANSITIONS["CREATED"] == ("LOADING", "FAILED")

    def test_unknown_role_refused(self):
        with pytest.raises(ConfigError):
            R.BackendSpec(
                backend_id="x",
                role="fastest",
                version="1",
                commit="c",
                source_identity="s",
                adapter_module="m",
            )
