"""Protocol plane tests: schema, error catalog, SSE framing and parser oracle."""

from __future__ import annotations

import json
import os

import pytest
import yaml

from hqsb.core.errors import ConfigError
from hqsb.serving import protocol as P
from hqsb.serving import sse

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs", "serving")


def load_profile():
    with open(os.path.join(_CONFIG_DIR, "protocol_profile.yaml"), encoding="utf-8") as handle:
        return P.ProtocolProfile.from_document(yaml.safe_load(handle))


def load_catalog():
    with open(os.path.join(_CONFIG_DIR, "error_catalog.yaml"), encoding="utf-8") as handle:
        return P.ErrorCatalog.from_document(yaml.safe_load(handle))


CHAT = "/v1/chat/completions"


@pytest.mark.unit
class TestProtocolProfile:
    def test_profile_hashes_and_endpoints(self):
        profile = load_profile()
        assert set(profile.endpoints) == {"/v1/completions", CHAT}
        assert len(profile.schema_hash) == 64
        assert profile.schema_hash == load_profile().schema_hash

    def test_unsupported_features_are_not_claimed(self):
        profile = load_profile()
        for feature in ("tools_and_function_calling", "responses_api", "logprobs"):
            with pytest.raises(ConfigError):
                profile.require_implemented(feature)

    def test_unknown_endpoint_is_refused(self):
        with pytest.raises(ConfigError):
            load_profile().request_fields("/v1/embeddings")


@pytest.mark.unit
class TestErrorCatalog:
    def test_catalog_is_internally_consistent(self):
        audit = load_catalog().validate()
        assert audit["ok"], audit["problems"]

    def test_retryability_respects_the_commit_boundary(self):
        catalog = load_catalog()
        assert catalog.is_retryable("service_overloaded") is True
        assert catalog.is_retryable("invalid_json") is False
        # conditional: retryable before the stream commits, never after
        assert catalog.is_retryable("backend_unavailable", committed=False) is True
        assert catalog.is_retryable("backend_unavailable", committed=True) is False

    def test_unknown_code_is_refused(self):
        with pytest.raises(ConfigError):
            load_catalog().entry("made_up_code")

    def test_overload_and_invalid_are_not_interchangeable(self):
        catalog = load_catalog()
        for code in catalog.overload_codes():
            assert catalog.entry(code).http_status in (429, 503)
        assert catalog.entry("unknown_model").http_status == 404


@pytest.mark.unit
class TestValidation:
    def _validate(self, payload, endpoint=CHAT):
        return P.validate_request(
            payload, endpoint=endpoint, profile=load_profile(), catalog=load_catalog()
        )

    def test_defaults_are_expanded_and_recorded(self):
        normalized = self._validate({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        assert normalized.stream is False
        assert normalized.expanded_defaults["max_tokens"] == 256
        assert normalized.normalized["temperature"] == 1.0

    def test_unknown_field_is_refused_not_dropped(self):
        with pytest.raises(P.RequestRejected) as excinfo:
            self._validate(
                {"model": "m", "messages": [{"role": "user", "content": "hi"}], "foo": 1}
            )
        assert excinfo.value.issue.code == "unknown_field"
        assert excinfo.value.issue.path == "$.foo"

    def test_default_max_tokens_can_be_overridden(self):
        normalized = self._validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
        )
        assert normalized.normalized["max_tokens"] == 8
        assert "max_tokens" not in normalized.expanded_defaults

    def test_negative_corpus_is_table_driven(self):
        corpus = P.negative_corpus()
        assert len(corpus) >= 15
        profile, catalogue = load_profile(), load_catalog()
        registry = {"m": {"model_id": "m", "precision": "float16"}}
        for case in corpus:
            with pytest.raises(P.RequestRejected) as excinfo:
                P.evaluate_negative_case(
                    case, profile=profile, catalog=catalogue, model_registry=registry
                )
            assert excinfo.value.issue.code == case.expected_code, case.name

    def test_unknown_model_and_context_budget(self):
        with pytest.raises(P.RequestRejected) as excinfo:
            P.resolve_model_alias("nope", {"m": {"model_id": "m"}})
        assert excinfo.value.issue.code == "unknown_model"

        profile = load_profile()
        normalized = self._validate({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        resolution = P.ModelResolution(alias="m", identity={"model_id": "m"})
        long_tokenizer = lambda _text: list(range(profile.limits["max_prompt_tokens"] + 1))  # noqa: E731
        with pytest.raises(P.RequestRejected) as excinfo:
            P.canonicalize_request(
                normalized,
                request_id="r0",
                resolution=resolution,
                tokenizer=long_tokenizer,
                chat_template=lambda messages: "x",
                profile=profile,
            )
        assert excinfo.value.issue.code == "context_length_exceeded"

    def test_requested_normalized_actual_views(self):
        profile = load_profile()
        normalized = self._validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
        canonical = P.canonicalize_request(
            normalized,
            request_id="r0",
            resolution=P.ModelResolution(alias="m", identity={"model_id": "m", "precision": "float16"}),
            tokenizer=lambda text: tuple(range(len(text))),
            chat_template=lambda messages: "".join(item["content"] for item in messages),
            profile=profile,
        )
        assert canonical.requested["messages"][0]["content"] == "hi"
        assert canonical.normalized["max_tokens"] == 256
        assert canonical.actual["input_token_ids"] == [0, 1]
        assert canonical.actual["template_applied"] is True
        assert canonical.actual["truncation_applied"] is False
        assert canonical.stream is True

    def test_response_oracle_detects_problems(self):
        profile = load_profile()
        good = {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "text": "hi",
                    "message": {"role": "assistant", "content": "hi"},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        assert P.validate_response_body(good, profile=profile) == []
        broken = dict(good)
        broken["usage"] = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 5}
        broken["choices"] = [{**good["choices"][0], "finish_reason": "exploded"}]
        problems = P.validate_response_body(broken, profile=profile)
        assert any("total_tokens" in item for item in problems)
        assert any("finish_reason" in item for item in problems)

    def test_conformance_matrix_requires_raw_evidence(self):
        rows = (
            P.ConformanceRow("chat_nonstream", CHAT, "stream_false", "dummy", "PASS", raw_uri=""),
            P.ConformanceRow("chat_stream", CHAT, "stream_true", "dummy", "UNSUPPORTED"),
        )
        report = P.conformance_matrix(rows)
        assert not report["ok"]
        assert any("raw URI" in item for item in report["problems"])
        assert any("reason" in item for item in report["problems"])


@pytest.mark.unit
class TestSseCodec:
    def _frames(self):
        payloads = [
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "model": "m",
                "choices": [
                    {"index": 0, "delta": {"content": "he", "token_ids": [1]}, "finish_reason": None}
                ],
            },
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "model": "m",
                "choices": [
                    {"index": 0, "delta": {"content": "llo", "token_ids": [2]}, "finish_reason": None}
                ],
            },
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "model": "m",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        raw = b"".join(sse.encode_data_frame(item) for item in payloads) + sse.encode_terminal()
        return raw, list(sse.parse_stream(raw)["parsed"])

    def test_framing_and_reconstruction(self):
        raw, frames = self._frames()
        assert sse.parse_stream(raw)["complete"]
        audit = sse.validate_stream(frames)
        assert audit["ok"], audit["problems"]
        rebuilt = sse.reconstruct(frames)
        assert rebuilt["text"] == "hello"
        assert rebuilt["token_ids"] == [1, 2]
        assert rebuilt["finish_reason"] == "stop"

    def test_terminal_must_be_unique_and_last(self):
        _, frames = self._frames()
        duplicated = list(frames) + [frames[-1]]
        assert not sse.validate_stream(duplicated)["ok"]
        reordered = [frames[-1]] + frames[:-1]
        assert not sse.validate_stream(reordered)["ok"]

    def test_truncated_and_unknown_fields_are_reported(self):
        raw, _ = self._frames()
        assert not sse.parse_stream(raw[:-2])["complete"]
        _, frames = self._frames()
        broken = list(frames)
        broken[0] = sse.SseFrame(
            index=0, raw=b"data: {", data="{", parse_error="invalid JSON in SSE data"
        )
        assert not sse.validate_stream(broken)["ok"]

    def test_reassembler_tolerates_tcp_boundaries(self):
        raw, frames = self._frames()
        for case in sse.fragmentation_cases(raw, splits=(1, 5, 42)):
            assert case["complete"], case["errors"]
            assert len(case["frames"]) == len(frames)

    def test_utf8_split_inside_a_code_point(self):
        results = sse.utf8_boundary_cases()
        assert all(item["ok"] for item in results), results

    def test_heartbeat_is_not_a_token(self):
        raw = (
            sse.encode_heartbeat("ping")
            + sse.encode_data_frame(
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "model": "m",
                    "choices": [
                        {"index": 0, "delta": {"content": "a", "token_ids": [7]}, "finish_reason": None}
                    ],
                }
            )
            + sse.encode_terminal()
        )
        report = sse.parse_stream(raw)
        assert report["complete"]
        audit = sse.validate_stream(report["parsed"])
        assert audit["ok"] and audit["heartbeats"] == 1
        assert sse.reconstruct(report["parsed"])["token_ids"] == [7]

    def test_stream_vs_nonstream_pairing(self):
        _, frames = self._frames()
        body = {
            "choices": [
                {"index": 0, "text": "hello", "token_ids": [1, 2], "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        assert sse.stream_vs_nonstream(frames, body)["ok"]
        mismatched = json.loads(json.dumps(body))
        mismatched["choices"][0]["text"] = "hellp"
        assert not sse.stream_vs_nonstream(frames, mismatched)["ok"]

    def test_parser_oracle_cases(self):
        raw, _ = self._frames()
        report = sse.parser_oracle_cases(raw)
        assert report["baseline_ok"]
        assert report["fragmented_all_ok"]
        assert report["sticky_detected_extra_frames"]
        assert report["truncated_reported"]
        assert report["unknown_field_reported"]
        assert report["doubled_blank_tolerated"]

    def test_silent_token_drop_is_refused(self):
        _, frames = self._frames()
        sse.require_no_silent_drop(frames, 2)
        with pytest.raises(ConfigError):
            sse.require_no_silent_drop(frames, 3)
