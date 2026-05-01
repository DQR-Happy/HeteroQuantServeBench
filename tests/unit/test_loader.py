"""Unit tests for the model loader path helpers and negative paths.

These tests validate diagnostic failure behavior (missing directories,
missing required files) without downloading or loading any model weights.
"""

from __future__ import annotations

import os

import pytest

from hqsb.models.loader import _resolve_path, _validate_model_directory, load_qwen3


class TestResolvePath:
    def test_expands_home(self):
        resolved = _resolve_path("~/models/hqsb/Qwen3-1.7B")
        assert resolved == os.path.abspath(
            os.path.expanduser("~/models/hqsb/Qwen3-1.7B")
        )

    def test_expands_env_var(self, monkeypatch):
        monkeypatch.setenv("HQSB_TEST_ROOT", "/tmp/hqsb-test")
        assert _resolve_path("$HQSB_TEST_ROOT/model") == "/tmp/hqsb-test/model"

    def test_returns_absolute(self):
        assert os.path.isabs(_resolve_path("relative/path"))


class TestValidateModelDirectory:
    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _validate_model_directory(str(tmp_path / "does-not-exist"))

    def test_missing_required_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError) as excinfo:
            _validate_model_directory(str(tmp_path))
        message = str(excinfo.value)
        assert "config.json" in message

    def test_valid_directory_passes(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "tokenizer.json").write_text("{}")
        # Must not raise.
        _validate_model_directory(str(tmp_path))


class TestLoadQwen3NegativePath:
    def test_missing_model_raises_before_loading(self):
        with pytest.raises(FileNotFoundError):
            load_qwen3("/nonexistent/hqsb/model/path")
