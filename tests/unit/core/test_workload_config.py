"""Unit tests for workload YAML single-source-of-truth loading."""

from __future__ import annotations

import os

import pytest

from hqsb.benchmark.workload_config import (
    load_workload_dicts,
    load_workload_specs,
    workload_specs_by_name,
)
from hqsb.core.errors import ConfigError

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_REAL_CONFIG = os.path.join(
    _REPO_ROOT, "configs", "benchmarks", "jetson_qwen3_fp16.yaml"
)


@pytest.mark.unit
class TestLoadWorkloadSpecs:
    def test_loads_real_config(self):
        specs = load_workload_specs(_REAL_CONFIG)
        assert len(specs) == 6
        names = [s.name for s in specs]
        assert names == [
            "tiny", "short", "balanced",
            "long_prefill", "decode_heavy", "long_balanced",
        ]

    def test_real_config_values(self):
        specs = workload_specs_by_name(load_workload_specs(_REAL_CONFIG))
        assert specs["tiny"].input_tokens == 32
        assert specs["tiny"].output_tokens == 16
        assert specs["long_balanced"].input_tokens == 2048

    def test_missing_file_raises(self):
        with pytest.raises(ConfigError):
            load_workload_specs("/nonexistent/workloads.yaml")

    def test_missing_workloads_key_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("benchmark:\n  model: Qwen/Qwen3-1.7B\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_workload_specs(str(path))

    def test_invalid_entry_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "workloads:\n  - name: zero\n    input_tokens: 0\n    output_tokens: 1\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError):
            load_workload_specs(str(path))

    def test_empty_workloads_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("workloads: []\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_workload_specs(str(path))


@pytest.mark.unit
class TestLoadWorkloadDicts:
    def test_returns_raw_dicts(self):
        dicts = load_workload_dicts(_REAL_CONFIG)
        assert len(dicts) == 6
        assert dicts[0]["name"] == "tiny"
        assert dicts[0]["input_tokens"] == 32
