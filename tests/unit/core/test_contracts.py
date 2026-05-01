"""Unit tests for the C1–C7 contract models.

Covers valid construction, missing-field rejection, unknown-field rejection,
and schema-version stamping.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hqsb.core.contracts import (
    BackendCapability,
    BenchmarkResult,
    EnvironmentInfo,
    GenerationSample,
    ModelArtifact,
    OperatorSpec,
    QuantArtifact,
    TensorSpec,
    TraceEvent,
    TraceEventType,
    WorkloadSpec,
)


def _artifact(**overrides) -> ModelArtifact:
    payload = {
        "model_id": "Qwen/Qwen3-1.7B",
        "source": "modelscope",
        "architecture": "Qwen3ForCausalLM",
        "dtype": "float16",
    }
    payload.update(overrides)
    return ModelArtifact(**payload)


@pytest.mark.unit
class TestModelArtifact:
    def test_valid_minimal(self):
        m = _artifact()
        assert m.model_id == "Qwen/Qwen3-1.7B"
        assert m.schema_version == "1.0.0"

    def test_missing_required_rejected(self):
        with pytest.raises(ValidationError):
            ModelArtifact(model_id="x", source="modelscope", dtype="float16")

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            _artifact(bogus=1)

    def test_optional_fields_default(self):
        m = _artifact()
        assert m.file_hashes == {}
        assert m.layout == "dense"
        assert m.quantization is None


@pytest.mark.unit
class TestWorkloadSpec:
    def test_valid(self):
        w = WorkloadSpec(name="short", input_tokens=128, output_tokens=32)
        assert w.input_tokens == 128
        assert w.schema_version == "1.0.0"

    def test_zero_input_rejected(self):
        with pytest.raises(ValidationError):
            WorkloadSpec(name="x", input_tokens=0, output_tokens=1)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            WorkloadSpec(name="x", input_tokens=1, output_tokens=1, bogus=True)


@pytest.mark.unit
class TestOperatorSpec:
    def test_valid(self):
        op = OperatorSpec(
            name="rmsnorm",
            semantic_version="1.0.0",
            inputs=[TensorSpec(name="x", dtype="float32", shape=[-1, 1024])],
            outputs=[TensorSpec(name="y", dtype="float32", shape=[-1, 1024])],
            implementation="v0_shared",
        )
        assert op.name == "rmsnorm"
        assert op.inputs[0].dtype == "float32"

    def test_negative_workspace_rejected(self):
        with pytest.raises(ValidationError):
            OperatorSpec(
                name="op", semantic_version="1.0.0",
                inputs=[], outputs=[], implementation="v0",
                workspace_bytes=-1,
            )


@pytest.mark.unit
class TestBackendCapability:
    def test_supports_dtype(self):
        cap = BackendCapability(name="dummy", supported_dtypes=["float16"])
        assert cap.supports_dtype("float16")
        assert not cap.supports_dtype("int4")


@pytest.mark.unit
class TestQuantArtifact:
    def test_valid(self):
        q = QuantArtifact(algorithm="rtn", bits=8, granularity="per-channel")
        assert q.bits == 8
        assert q.symmetric is True

    def test_bits_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            QuantArtifact(algorithm="rtn", bits=9, granularity="per-channel")


@pytest.mark.unit
class TestTraceEvent:
    def test_valid(self):
        e = TraceEvent(
            event_type=TraceEventType.DECODE,
            timestamp_ns=123,
            trace_id="t1",
            span_id="s1",
        )
        assert e.event_type == TraceEventType.DECODE
        assert e.attributes == {}


@pytest.mark.unit
class TestBenchmarkResult:
    def test_valid_minimal(self):
        r = BenchmarkResult(run_id="run_1", timestamp=1.0, environment=EnvironmentInfo())
        assert r.schema_version == "1.0.0"
        assert r.raw_samples == []

    def test_workload_roundtrip(self):
        w = WorkloadSpec(name="short", input_tokens=128, output_tokens=32)
        r = BenchmarkResult(
            run_id="run_1", timestamp=1.0,
            environment=EnvironmentInfo(), workload=w,
        )
        assert r.workload.input_tokens == 128
