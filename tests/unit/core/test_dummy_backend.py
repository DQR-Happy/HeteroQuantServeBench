"""Unit tests for DummyBackend and the backend-interface benchmark engine.

Validates the S01 acceptance criterion: a dummy backend can register, run,
and produce a versioned BenchmarkResult purely from the Contract, with no
model weights or GPU.
"""

from __future__ import annotations

import json

import pytest

from hqsb.backends import DummyBackend, make_dummy_backend
from hqsb.core.contracts import (
    EnvironmentInfo,
    ModelArtifact,
    WorkloadSpec,
)
from hqsb.core.errors import CapabilityError
from hqsb.core.registry import RegistryHub
from hqsb.benchmark.engine import BenchmarkEngine, run_backend


@pytest.fixture
def artifact() -> ModelArtifact:
    return ModelArtifact(
        model_id="Qwen/Qwen3-1.7B",
        source="modelscope",
        architecture="Qwen3ForCausalLM",
        dtype="float16",
    )


@pytest.fixture
def workload() -> WorkloadSpec:
    return WorkloadSpec(
        name="short",
        input_tokens=128,
        output_tokens=32,
        repetitions=3,
        seed=42,
    )


@pytest.mark.unit
class TestDummyBackend:
    def test_capabilities(self):
        cap = DummyBackend().capabilities()
        assert cap.name == "dummy"
        assert cap.supports_dtype("float16")
        assert cap.max_batch == 1

    def test_generate_is_deterministic(self, workload):
        backend = DummyBackend()
        out1 = backend.generate(workload, None)
        out2 = backend.generate(workload, None)
        tokens1 = [s.generated_token_ids for s in out1.samples]
        tokens2 = [s.generated_token_ids for s in out2.samples]
        assert tokens1 == tokens2

    def test_load_rejects_wrong_type(self):
        backend = DummyBackend()
        with pytest.raises(TypeError):
            backend.load(object())

    def test_lifecycle(self, artifact, workload):
        backend = DummyBackend()
        assert backend.health() is True
        backend.load(artifact)
        backend.warmup(workload)
        assert backend.metrics()["loaded"] is True
        backend.close()
        assert backend.health() is False

    def test_factory(self):
        assert isinstance(make_dummy_backend(), DummyBackend)


@pytest.mark.unit
class TestBenchmarkEngine:
    def test_run_produces_result(self, artifact, workload):
        result = BenchmarkEngine(DummyBackend()).run(workload, artifact=artifact)
        assert result.schema_version == "1.0.0"
        assert result.run_id.startswith("run_")
        assert result.correctness.passed is True
        assert result.correctness.method == "determinism"
        assert len(result.raw_samples) == 3
        assert result.workload.input_tokens == 128
        assert result.model_artifact_hash is not None
        assert result.config_hash is not None

    def test_run_summary_has_expected_metrics(self, artifact, workload):
        result = BenchmarkEngine(DummyBackend()).run(workload, artifact=artifact)
        assert "itl" in result.summary
        assert result.summary["itl"]["mean_ms"] == pytest.approx(10.0)

    def test_result_serializes_to_json(self, artifact, workload):
        result = BenchmarkEngine(DummyBackend()).run(workload, artifact=artifact)
        data = json.loads(result.model_dump_json())
        assert data["schema_version"] == "1.0.0"
        assert data["correctness"]["passed"] is True

    def test_capability_check_rejects_batch(self, artifact, workload):
        workload = workload.model_copy(update={"batch_size": 8})
        with pytest.raises(CapabilityError):
            BenchmarkEngine(DummyBackend()).run(workload, artifact=artifact)

    def test_capability_check_rejects_dtype(self, workload):
        artifact = ModelArtifact(
            model_id="m", source="modelscope",
            architecture="a", dtype="int4",
        )
        with pytest.raises(CapabilityError):
            BenchmarkEngine(DummyBackend()).run(workload, artifact=artifact)

    def test_capability_check_can_be_disabled(self, workload):
        backend = DummyBackend()
        result = BenchmarkEngine(backend).run(
            workload.model_copy(update={"batch_size": 8}),
            check_capabilities=False,
        )
        assert result.raw_samples  # still produced output

    def test_run_backend_wrapper(self, workload):
        result = run_backend(DummyBackend(), workload)
        assert result.correctness.passed is True


@pytest.mark.unit
class TestRegistryIntegration:
    def test_dummy_backend_registers_and_runs(self, artifact, workload):
        hub = RegistryHub()
        hub.backends.register("dummy", make_dummy_backend)
        backend = hub.backends.get("dummy")()
        result = BenchmarkEngine(backend).run(workload, artifact=artifact)
        assert result.correctness.passed is True
