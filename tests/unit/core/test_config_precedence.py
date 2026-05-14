"""Unit tests for E01-02 config precedence, provenance, redaction & identity.

These freeze the loader behaviors verified by the E01-02 experiment so the
CI keeps enforcing them: fixed precedence, per-field source map, strict YAML,
semantic hash scope, secret redaction, and cross-field validation.
"""

from __future__ import annotations

import copy

import pytest

from hqsb.core.config import BenchmarkConfig, ConfigLoader
from hqsb.core.errors import ConfigError


def _defaults() -> dict:
    return {
        "benchmark": {
            "model": "audit/model",
            "model_source": "modelscope",
            "backend": "dummy",
            "dtype": "float16",
            "attention_backend": "eager",
            "batch_size": 1,
            "warmup": True,
            "repetitions": 1,
        },
        "workloads": [
            {"name": "tiny", "input_tokens": 32, "output_tokens": 16},
        ],
    }


def _loader() -> ConfigLoader:
    return ConfigLoader(BenchmarkConfig)


@pytest.mark.unit
class TestPrecedence:
    def test_cli_beats_env_beats_file_beats_defaults(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("benchmark:\n  batch_size: 2\n", encoding="utf-8")
        res = _loader().load_resolved(
            defaults=_defaults(),
            path=str(path),
            environ={"HQSB_BENCHMARK__BATCH_SIZE": "4"},
            cli={"benchmark": {"batch_size": 8}},
        )
        assert res.config.benchmark.batch_size == 8
        assert res.source_map["benchmark.batch_size"] == "cli"

    def test_multi_layer_mix_preserves_uncovered_fields(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(
            "benchmark:\n  backend: file-backend\n", encoding="utf-8"
        )
        res = _loader().load_resolved(
            defaults=_defaults(),
            path=str(path),
            environ={"HQSB_BENCHMARK__REPETITIONS": "4"},
            cli={"benchmark": {"batch_size": 8}},
        )
        assert res.config.benchmark.backend == "file-backend"
        assert res.config.benchmark.repetitions == 4
        assert res.config.benchmark.batch_size == 8
        assert res.config.benchmark.dtype == "float16"  # from defaults, uncovered
        assert res.source_map["benchmark.backend"] == "file"
        assert res.source_map["benchmark.repetitions"] == "env"
        assert res.source_map["benchmark.batch_size"] == "cli"
        assert res.source_map["benchmark.dtype"] == "defaults"


@pytest.mark.unit
class TestStrictYaml:
    def test_duplicate_key_rejected(self, tmp_path):
        path = tmp_path / "dup.yaml"
        path.write_text("benchmark:\n  batch_size: 2\n  batch_size: 3\n", encoding="utf-8")
        with pytest.raises(ConfigError) as exc:
            _loader().load_resolved(defaults=_defaults(), path=str(path), environ={})
        assert exc.value.details["error_code"] == "duplicate_yaml_key"


@pytest.mark.unit
class TestPerLayerValidation:
    def test_low_layer_invalid_type_masked_by_high_still_rejected(self, tmp_path):
        path = tmp_path / "low.yaml"
        path.write_text("benchmark:\n  batch_size: not-an-int\n", encoding="utf-8")
        with pytest.raises(ConfigError) as exc:
            _loader().load_resolved(
                defaults=_defaults(),
                path=str(path),
                environ={},
                cli={"benchmark": {"batch_size": 8}},
            )
        assert exc.value.details["source"] == "file"
        assert exc.value.details["error_code"] == "type_error"

    def test_unknown_field_localized_to_source(self):
        with pytest.raises(ConfigError) as exc:
            _loader().load_resolved(
                defaults=_defaults(),
                environ={},
                cli={"benchmark": {"ols_typo": 64}},
            )
        assert exc.value.details["source"] == "cli"
        assert exc.value.details["error_code"] == "unknown_field"


@pytest.mark.unit
class TestSemanticHash:
    def test_operational_fields_excluded(self):
        base = _loader().load_resolved(defaults=_defaults(), environ={})
        changed = _loader().load_resolved(
            defaults=_defaults(),
            environ={},
            cli={"run": {"run_id": "run_123", "output_dir": "reports/other"}},
        )
        assert base.config_hash == changed.config_hash

    def test_real_change_differs(self):
        base = _loader().load_resolved(defaults=_defaults(), environ={})
        changed = _loader().load_resolved(
            defaults=_defaults(), environ={}, cli={"benchmark": {"batch_size": 8}}
        )
        assert base.config_hash != changed.config_hash


@pytest.mark.unit
class TestRedaction:
    def test_secret_redacted_everywhere(self):
        sentinel = "SECRET_TOKEN_123"
        res = _loader().load_resolved(
            defaults=_defaults(),
            environ={},
            cli={"secrets": {"modelscope_token": sentinel}},
        )
        assert res.public_view["secrets"]["modelscope_token"] == "<redacted>"
        assert sentinel not in json_dumps(res.public_view)
        assert sentinel not in json_dumps(res.semantic_payload)
        assert sentinel not in json_dumps(res.layer_inputs)

    def test_secret_not_in_error_message(self):
        sentinel = "SECRET_TOKEN_456"
        with pytest.raises(ConfigError) as exc:
            _loader().load_resolved(
                defaults=_defaults(),
                environ={},
                cli={
                    "secrets": {"modelscope_token": sentinel},
                    "benchmark": {"batch_size": "bad"},
                },
            )
        assert sentinel not in str(exc.value)


@pytest.mark.unit
class TestCrossField:
    def test_decode_heavy_must_be_decode_bound(self):
        with pytest.raises(ConfigError) as exc:
            BenchmarkConfig.model_validate({
                "benchmark": {
                    "model": "audit/model", "model_source": "modelscope",
                    "backend": "dummy", "dtype": "float16",
                },
                "workloads": [
                    {"name": "decode_heavy", "input_tokens": 256, "output_tokens": 128},
                ],
            })
        assert exc.value.details["error_code"] == "cross_field_conflict"


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj)
