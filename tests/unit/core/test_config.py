"""Unit tests for the layered config loader and config hashing.

Covers precedence (defaults < file < env < CLI), unknown-field rejection,
type checking, and deterministic hash stability.

Tests pass ``environ={}`` explicitly where they are not exercising the
environment layer, to isolate them from any ambient ``HQSB_*`` variables
in the host environment.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from hqsb.core.config import ConfigLoader, config_hash, deep_merge
from hqsb.core.errors import ConfigError


class _BenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = ""
    dtype: str = "float16"
    batch_size: int = 1
    warmup: bool = True


@pytest.mark.unit
class TestDeepMerge:
    def test_nested_merge(self):
        base = {"a": 1, "nested": {"x": 1, "y": 2}}
        override = {"b": 2, "nested": {"y": 20}}
        assert deep_merge(base, override) == {
            "a": 1, "b": 2, "nested": {"x": 1, "y": 20},
        }

    def test_base_not_mutated(self):
        base = {"a": {"x": 1}}
        deep_merge(base, {"a": {"y": 2}})
        assert base == {"a": {"x": 1}}


@pytest.mark.unit
class TestConfigLoader:
    def test_defaults_only(self):
        loader = ConfigLoader(_BenchConfig)
        config = loader.load(defaults={"model_id": "m"}, environ={})
        assert config.model_id == "m"
        assert config.batch_size == 1

    def test_file_overrides_defaults(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("model_id: from-file\nbatch_size: 8\n", encoding="utf-8")
        loader = ConfigLoader(_BenchConfig)
        config = loader.load(
            defaults={"model_id": "from-default"}, path=str(path), environ={}
        )
        assert config.model_id == "from-file"
        assert config.batch_size == 8

    def test_env_overrides_file(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("batch_size: 8\n", encoding="utf-8")
        loader = ConfigLoader(_BenchConfig)
        environ = {"HQSB_BATCH_SIZE": "16"}
        config = loader.load(path=str(path), environ=environ)
        assert config.batch_size == 16

    def test_env_coerces_scalar_types(self):
        loader = ConfigLoader(_BenchConfig)
        environ = {
            "HQSB_BATCH_SIZE": "4",
            "HQSB_WARMUP": "false",
            "HQSB_MODEL_ID": '"json-string"',
        }
        config = loader.load(environ=environ)
        assert config.batch_size == 4
        assert config.warmup is False
        assert config.model_id == "json-string"

    def test_cli_has_highest_precedence(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("batch_size: 8\n", encoding="utf-8")
        loader = ConfigLoader(_BenchConfig)
        config = loader.load(
            path=str(path),
            environ={"HQSB_BATCH_SIZE": "16"},
            cli={"batch_size": 32},
        )
        assert config.batch_size == 32

    def test_unknown_field_rejected(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("unknown_key: 1\n", encoding="utf-8")
        loader = ConfigLoader(_BenchConfig)
        with pytest.raises(ConfigError):
            loader.load(path=str(path), environ={})

    def test_missing_file_raises_config_error(self):
        loader = ConfigLoader(_BenchConfig)
        with pytest.raises(ConfigError):
            loader.load(path="/nonexistent/config.yaml", environ={})

    def test_wrong_type_rejected(self):
        loader = ConfigLoader(_BenchConfig)
        with pytest.raises(ConfigError):
            loader.load(cli={"batch_size": "not-an-int"}, environ={})


@pytest.mark.unit
class TestConfigHash:
    def test_hash_is_deterministic(self):
        loader = ConfigLoader(_BenchConfig)
        c1, h1 = loader.load_with_hash(cli={"batch_size": 4}, environ={})
        c2, h2 = loader.load_with_hash(cli={"batch_size": 4}, environ={})
        assert h1 == h2

    def test_hash_differs_by_value(self):
        loader = ConfigLoader(_BenchConfig)
        _, h1 = loader.load_with_hash(cli={"batch_size": 4}, environ={})
        _, h2 = loader.load_with_hash(cli={"batch_size": 8}, environ={})
        assert h1 != h2

    def test_hash_is_64_hex(self):
        loader = ConfigLoader(_BenchConfig)
        _, h = loader.load_with_hash(environ={})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_config_hash_helper(self):
        assert config_hash(_BenchConfig(batch_size=4)) == config_hash(
            _BenchConfig(batch_size=4)
        )
