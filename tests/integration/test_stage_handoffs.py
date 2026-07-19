"""Cross-stage regressions: execute real adapters against adjacent contracts.

These CPU tests use synthetic tokens and injected model execution, never model
weights. They verify handoffs, not hardware performance or scientific verdicts.
"""

from dataclasses import replace

import pytest

from hqsb.backends.dummy import DummyBackend
from hqsb.benchmark.engine import BenchmarkEngine
from hqsb.core.contracts import ModelArtifact, WorkloadSpec
from hqsb.core.errors import BackendError, CapabilityError
from hqsb.runtime.adapter import AdapterState, ReferenceRuntimeAdapter
from hqsb.runtime.request import BackendSpec, ModelIdentity, RequestSpec, StopSpec

pytestmark = pytest.mark.integration


def artifact():
    return ModelArtifact(
        model_id="test-model",
        source="local",
        revision="r1",
        architecture="test",
        dtype="float16",
    )


def request(model):
    return RequestSpec(
        request_id="handoff",
        input_token_ids=(9, 3, 7),
        identity=ModelIdentity(
            model_id=model.model_id,
            model_manifest_sha256=model.identity_hash(),
            revision=model.revision,
            tokenizer_id="test-tokenizer",
            chat_template_hash="",
            precision=model.dtype,
        ),
        stop=StopSpec(max_new_tokens=3, ignore_eos=True),
    )


def adapter(backend):
    return ReferenceRuntimeAdapter(
        backend,
        BackendSpec(
            backend_id=backend.name,
            role="reference",
            version="1",
            commit="test",
            source_identity="test",
            adapter_module="hqsb.runtime.adapter",
        ),
    )


def test_c4_backend_runs_through_s07_request_bridge():
    model = artifact()
    req = request(model)
    runtime = adapter(DummyBackend())
    runtime.load(req.identity, artifact=model)
    runtime.warmup(req)
    result = runtime.generate(req)
    assert len(result.token_ids) == 3
    assert result.usage == {"input_tokens": 3, "output_tokens": 3}
    assert result.timings["decode_total_ms"] == 20.0
    assert runtime.close()["closed"]


@pytest.mark.parametrize("field", ["cancel", "timeout", "streaming"])
def test_reference_adapter_does_not_advertise_unimplemented_operations(field):
    report = adapter(DummyBackend()).capability()
    assert report.state_of(field) == "UNSUPPORTED_REJECT"


def test_reference_adapter_failed_load_can_be_cleaned_up():
    model = artifact()
    runtime = adapter(DummyBackend(fail_at="load"))
    with pytest.raises(BackendError):
        runtime.load(request(model).identity, artifact=model)
    assert runtime.state == AdapterState.FAILED
    assert runtime.close()["closed"]


def test_reference_adapter_rejects_unimplemented_timeout_before_execution():
    model = artifact()
    req = request(model)
    runtime = adapter(DummyBackend())
    runtime.load(req.identity, artifact=model)
    with pytest.raises(CapabilityError):
        runtime.generate(replace(req, timeout_s=1.0))


def test_benchmark_distinguishes_decode_tail_from_output_throughput():
    result = BenchmarkEngine(DummyBackend()).run(
        WorkloadSpec(name="clock", input_tokens=3, output_tokens=3, repetitions=2)
    )
    assert result.summary["decode_tokens_per_s"] == pytest.approx(100.0)
    assert result.summary["output_tokens_per_s"] == pytest.approx(3 / 0.0401)


def test_benchmark_checks_context_before_loading():
    backend = DummyBackend()
    with pytest.raises(CapabilityError):
        BenchmarkEngine(backend).run(
            WorkloadSpec(name="too-long", input_tokens=4090, output_tokens=10),
            artifact=artifact(),
        )
    assert backend.call_counts["load"] == 0


def test_pytorch_backend_keeps_explicit_request_tokens_and_warmup_count(monkeypatch):
    torch = pytest.importorskip("torch")
    from hqsb.backends import pytorch

    backend = pytorch.PyTorchBackend()
    backend._model = object()
    backend._tokenizer = object()
    monkeypatch.setattr(backend, "_device", lambda: "cpu")
    seen = []

    def execute(model, inputs, output_tokens):
        seen.append(inputs["input_ids"].tolist())
        return {
            "input_tokens": 3,
            "output_tokens": output_tokens,
            "generated_token_ids": list(range(output_tokens)),
            "prefill_forward_ms": 2.0,
            "first_token_selection_ms": 1.0,
            "raw_itl_ms": [1.0] * (output_tokens - 1),
            "peak_cuda_allocated_mb": 0.0,
            "peak_cuda_reserved_mb": 0.0,
        }

    monkeypatch.setattr(pytorch, "benchmark_model_core", execute)
    monkeypatch.setattr(pytorch, "cuda_memory_snapshot", lambda: {})
    workload = WorkloadSpec(
        name="ids",
        input_tokens=3,
        output_tokens=2,
        token_ids=[9, 3, 7],
        warmup=2,
        repetitions=2,
    )
    backend.warmup(workload)
    output = backend.generate(workload, (9, 3, 7))
    assert seen == [[[9, 3, 7]]] * 4
    assert len(output.samples) == 2
    assert torch.equal(torch.tensor(seen[0]), torch.tensor([[9, 3, 7]]))


def test_pytorch_backend_does_not_reuse_same_name_different_revision(monkeypatch):
    pytest.importorskip("torch")
    from hqsb.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()
    backend._model = object()
    backend._artifact = artifact()
    with pytest.raises(BackendError, match="close"):
        backend.load(artifact().model_copy(update={"revision": "r2"}))


def test_partial_runtime_quality_does_not_become_c6_pass():
    from hqsb.runtime.telemetry import S07ResultFields, project_c6

    result = project_c6("partial-quality", S07ResultFields(quality_status="partial"))
    assert result.correctness.passed is False
    assert result.correctness.details["runtime_quality_status"] == "partial"


def test_c4_forwards_explicit_manifest_policy_and_cpu_staging(monkeypatch):
    pytest.importorskip("torch")
    from hqsb.backends.pytorch import PyTorchBackend
    from hqsb.models import loader

    observed = {}

    def load(path, **kwargs):
        observed.update(kwargs)
        return object(), object(), 0.1

    monkeypatch.setattr(loader, "load_qwen3", load)
    backend = PyTorchBackend(
        verify_manifest="frozen.txt",
        manifest_allow_extra=("model_sha256_manifest.txt",),
        cpu_staging=True,
    )
    backend.load(artifact())
    assert observed["verify_manifest"] == "frozen.txt"
    assert observed["allow_extra"] == ("model_sha256_manifest.txt",)
    assert observed["cpu_staging"] is True
    assert "strict_extra" not in observed  # loader's strict default stays enabled


def test_compiler_and_evaluation_export_real_c6_and_c7_contracts():
    from hqsb.compiler import telemetry as compiler
    from hqsb.core.contracts import BenchmarkResult, TraceEvent
    from hqsb.evaluation import telemetry as evaluation

    compiled = compiler.to_benchmark_result(
        compiler.S11ResultFields(compile_id="compile-1"), timestamp=1.0
    )
    evaluated = evaluation.to_benchmark_result(
        evaluation.S12ResultFields(
            comparison_id="comparison-1",
            candidate_id="candidate-1",
            layer="model-core",
            quality_status="pass",
        ),
        timestamp=2.0,
    )
    for result, namespace in ((compiled, "s11"), (evaluated, "s12")):
        restored = BenchmarkResult.model_validate_json(result.model_dump_json())
        assert namespace in restored.summary
        assert (
            restored.correctness is None
        )  # projection is not independent quality evidence
        assert restored.raw_samples == []
    for kind in evaluation.C7_KIND_MAP:
        event = evaluation.to_trace_event(
            evaluation.CampaignTraceRecord(
                span_id=kind,
                trace_id="trace",
                event_kind=kind,
                started_at_ns=12,
            )
        )
        restored = TraceEvent.model_validate_json(event.model_dump_json())
        assert restored.attributes["s12"]["event_kind"] == kind


def test_invalid_projection_cannot_be_published_as_c6():
    from hqsb.compiler.telemetry import S11ResultFields, to_benchmark_result
    from hqsb.core.errors import ConfigError

    with pytest.raises(ConfigError):
        to_benchmark_result(S11ResultFields(), timestamp=1.0)


def test_backend_trace_survives_c6_serialization():
    from hqsb.core.contracts import BenchmarkResult, TraceEvent

    result = BenchmarkEngine(DummyBackend()).run(
        WorkloadSpec(name="trace", input_tokens=3, output_tokens=3), artifact=artifact()
    )
    restored = BenchmarkResult.model_validate_json(result.model_dump_json())
    trace = restored.summary["trace"]
    assert trace["run_id"] == restored.run_id
    assert trace["events"] and trace["trace_ids"]
    assert {
        TraceEvent.model_validate(event).trace_id for event in trace["events"]
    } == set(trace["trace_ids"])


def test_repeated_malformed_samples_cannot_pass_determinism():
    from hqsb.core.contracts import GenerationOutput, GenerationSample

    class MalformedBackend(DummyBackend):
        def generate(self, workload, inputs):
            return GenerationOutput(
                samples=[GenerationSample(input_tokens=3, output_tokens=3)]
            )

    result = BenchmarkEngine(MalformedBackend()).run(
        WorkloadSpec(name="bad", input_tokens=3, output_tokens=3)
    )
    assert not result.correctness.passed
    assert result.correctness.details["malformed_samples"] == [0]
