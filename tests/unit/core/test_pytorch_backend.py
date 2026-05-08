"""Unit tests for PyTorchBackend contract compliance (no model weights).

These tests exercise the backend's contract surface and the pure mapping
helpers without loading a real model or touching the GPU.
"""

from __future__ import annotations

import pytest

from hqsb.backends import PyTorchBackend, make_pytorch_backend
from hqsb.core.contracts import ModelArtifact, WorkloadSpec
from hqsb.core.errors import BackendError


@pytest.fixture
def backend() -> PyTorchBackend:
    return PyTorchBackend(model_path="/nonexistent/model")


@pytest.fixture
def artifact() -> ModelArtifact:
    return ModelArtifact(
        model_id="Qwen/Qwen3-1.7B",
        source="modelscope",
        architecture="Qwen3ForCausalLM",
        dtype="float16",
    )


@pytest.mark.unit
class TestContractSurface:
    def test_name(self, backend):
        assert backend.name == "pytorch"

    def test_capabilities(self, backend):
        cap = backend.capabilities()
        assert cap.supports_dtype("float16")
        assert cap.max_batch == 1
        assert cap.name == "pytorch"

    def test_health_unloaded(self, backend):
        assert backend.health() is False

    def test_generate_unloaded_raises(self, backend, artifact):
        workload = WorkloadSpec(name="x", input_tokens=8, output_tokens=4)
        with pytest.raises(BackendError):
            backend.generate(workload, None)

    def test_load_rejects_wrong_type(self, backend):
        with pytest.raises(TypeError):
            backend.load(object())

    def test_warmup_rejects_wrong_type(self, backend):
        with pytest.raises(TypeError):
            backend.warmup(object())

    def test_close_unloaded_is_noop(self, backend):
        backend.close()  # must not raise
        assert backend.health() is False


@pytest.mark.unit
class TestMappingHelpers:
    def test_to_sample(self):
        result = {
            "input_tokens": 8,
            "output_tokens": 4,
            "generated_token_ids": [1, 2, 3, 4],
            "prefill_forward_ms": 10.0,
            "first_token_selection_ms": 0.5,
            "raw_itl_ms": [2.0, 3.0, 4.0],
            "peak_cuda_allocated_mb": 100.0,
            "peak_cuda_reserved_mb": 120.0,
        }
        sample = PyTorchBackend._to_sample(result)
        assert sample.input_tokens == 8
        assert sample.output_tokens == 4
        assert sample.generated_token_ids == [1, 2, 3, 4]
        assert sample.itl_ms == [2.0, 3.0, 4.0]
        assert sample.peak_cuda_allocated_mb == 100.0

    def test_backend_metrics(self):
        result = {
            "load_time_s": 12.3,
            "kv_cache": {"total_bytes": 1024},
            "model_weight_bytes": 2048,
            "process_rss_bytes": 4096,
            "process_swap_bytes": 0,
        }
        metrics = PyTorchBackend._backend_metrics(result)
        assert metrics["load_time_s"] == 12.3
        assert metrics["kv_cache"]["total_bytes"] == 1024
        assert metrics["model_weight_bytes"] == 2048


@pytest.mark.unit
class TestFactory:
    def test_make_pytorch_backend(self):
        assert isinstance(make_pytorch_backend(), PyTorchBackend)
