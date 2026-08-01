"""Observation contracts use fake clocks/collectors, never an accelerator."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from hqsb.backends.observation import ObservationRecorder, OperatorCapture


def test_host_phase_closes_on_exception_and_off_records_nothing(monkeypatch):
    ticks = iter([5.0, 5.005, 5.011])
    record = ObservationRecorder(clock=lambda: next(ticks))
    with pytest.raises(ValueError):
        with record.phase("prefill"):
            raise ValueError("model failed")
    phase = record.data["phases"][0]
    assert phase["start_ms"] == pytest.approx(5)
    assert phase["duration_ms"] == pytest.approx(6)
    assert phase["source"] == "host_monotonic"
    off = ObservationRecorder("off")
    with off.phase("absent"):
        pass
    off.token(index=1)
    off.snapshot("absent", lambda: pytest.fail("off must not probe memory"))
    assert (
        off.data["phases"] == off.data["tokens"] == off.data["memory_snapshots"] == []
    )


def test_missing_allocator_is_null_not_zero(monkeypatch):
    monkeypatch.setattr("hqsb.backends.observation.host_memory", lambda: {})
    record = ObservationRecorder()
    record.snapshot("cpu", lambda: {})
    assert record.data["memory_snapshots"][0]["allocated_bytes"] is None
    assert record.data["memory_snapshots"][0]["reserved_bytes"] is None


class FakeProfiler:
    def __init__(self, activities, *, cuda_events=False, stop_error=False):
        self.activities = activities
        self.cuda_events = cuda_events
        self.stop_error = stop_error
        self.stopped = 0

    def start(self):
        pass

    def stop(self):
        self.stopped += 1
        if self.stop_error:
            raise RuntimeError("collector disconnected")

    def events(self):
        return [SimpleNamespace(device_type="CUDA" if self.cuda_events else "CPU")]

    def key_averages(self, **kwargs):
        assert kwargs == {"group_by_input_shape": True}
        return [
            SimpleNamespace(
                key="aten::mm",
                count=2,
                cpu_time_total=4000,
                self_cpu_time_total=2000,
                device_time_total=3000,
                self_device_time_total=2500,
                input_shapes=[[2, 3], [3, 4]],
            )
        ]

    def export_chrome_trace(self, path):
        from pathlib import Path

        Path(path).write_text('{"traceEvents": []}')


def torch_stub(
    *, support_cuda=True, cuda_events=False, start_error=False, stop_error=False
):
    collectors = []

    def factory(*, activities, record_shapes, profile_memory, with_stack):
        assert record_shapes and profile_memory and not with_stack
        if start_error and "CUDA" in activities:
            raise RuntimeError("CUPTI permission denied")
        collector = FakeProfiler(
            activities, cuda_events=cuda_events, stop_error=stop_error
        )
        collectors.append(collector)
        return collector

    return SimpleNamespace(
        profiler=SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CPU="CPU", CUDA="CUDA"),
            supported_activities=lambda: {"CPU", "CUDA"} if support_cuda else {"CPU"},
            profile=factory,
            record_function=lambda _: nullcontext(),
        )
    ), collectors


def test_capture_bound_and_real_operator_units(tmp_path):
    torch, collectors = torch_stub(cuda_events=True)
    capture = OperatorCapture(
        torch, enabled=True, capture_dir=str(tmp_path), capture_id="r1"
    )
    capture.start()
    for _ in range(30):
        capture.step_completed()
    capture.finish(total_tokens=30)
    capture.save_summary()
    assert collectors[0].stopped == 1
    assert capture.summary["coverage"]["decode_steps"] == 8
    assert capture.summary["coverage"]["total_output_tokens"] == 30
    assert capture.summary["status"] == "complete"
    assert capture.summary["operators"][0]["cuda_ms"] == 3
    assert capture.summary["operators"][0]["self_cuda_ms"] == 2.5
    assert capture.summary["operators"][0]["cpu_ms"] == 4
    assert capture.summary["trace_available"] is True
    assert json.loads((tmp_path / "profile.json").read_text())["capture_id"] == "r1"


@pytest.mark.parametrize("start_error,support_cuda", [(True, True), (False, False)])
def test_cpu_fallback_never_fabricates_gpu_times(start_error, support_cuda):
    torch, collectors = torch_stub(start_error=start_error, support_cuda=support_cuda)
    capture = OperatorCapture(torch, enabled=True)
    capture.start()
    capture.step_completed()
    capture.finish()
    assert collectors[-1].activities == ["CPU"]
    assert capture.summary["status"] == "partial"
    assert capture.summary["operators"][0]["cuda_ms"] is None
    assert capture.summary["operators"][0]["self_cuda_ms"] is None
    assert capture.summary["activities"] == ["CPU"]
    assert capture.summary["limitations"]


def test_profiler_failure_does_not_replace_inference_failure():
    torch, _ = torch_stub(stop_error=True)
    capture = OperatorCapture(torch, enabled=True)
    capture.start()
    with pytest.raises(ValueError, match="inference failed"):
        try:
            raise ValueError("inference failed")
        finally:
            capture.finish()
    assert capture.summary["status"] == "error"
    assert "collector disconnected" in capture.summary["limitations"][0]


def test_oversize_trace_is_removed_and_declared(tmp_path):
    torch, _ = torch_stub(cuda_events=True)
    capture = OperatorCapture(torch, enabled=True, capture_dir=str(tmp_path))
    capture.MAX_TRACE_BYTES = 1
    capture.start()
    capture.finish()
    assert not (tmp_path / "trace.json").exists()
    assert capture.summary["trace_available"] is False
    assert "retained artifact limit" in capture.summary["limitations"][0]


def test_relative_capture_directory_rejected():
    with pytest.raises(ValueError, match="absolute"):
        OperatorCapture(None, capture_dir="../../escape")


def test_module_hooks_are_removed_when_forward_is_interrupted():
    torch, _ = torch_stub(cuda_events=True)
    removed = []

    class Module:
        def register_forward_pre_hook(self, hook):
            self.before = hook
            return SimpleNamespace(remove=lambda: removed.append("before"))

        def register_forward_hook(self, hook, **kwargs):
            self.after = hook
            return SimpleNamespace(remove=lambda: removed.append("after"))

    layer = Module()
    model = SimpleNamespace(named_modules=lambda: [("model.layers.0", layer)])
    capture = OperatorCapture(torch, enabled=True)
    capture.start()
    capture.attach_modules(model)
    layer.before(layer, ())
    # Simulate an older PyTorch that does not call the post hook on failure.
    capture.finish()
    assert removed == ["before", "after"]
    assert capture._ranges == []
    assert capture.summary["coverage"]["module_names"] == ["model.layers.0"]


def tiny_quant_model():
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(128, 4, bias=False)
    model.lm_head = torch.nn.Linear(4, 2, bias=False)
    return model


@pytest.mark.parametrize("bits,group_size", [(4, 128), (8, None)])
def test_quantization_job_commits_without_mutating_loaded_model(
    tmp_path, bits, group_size
):
    import threading

    from hqsb.backends.quantization import quantize_loaded_model

    torch = pytest.importorskip("torch")
    model = tiny_quant_model()
    original = [parameter.detach().clone() for parameter in model.parameters()]
    rows = list(
        quantize_loaded_model(
            model,
            {
                "artifact_dir": str(tmp_path / "artifact"),
                "bits": bits,
                "group_size": group_size,
            },
            threading.Event(),
            {
                "hash": "unverified-metadata:test",
                "revision": "test",
                "scope": "metadata_only",
            },
        )
    )
    result = rows[-1]
    assert result["finish_reason"] == "stop"
    assert result["metrics"]["quantization"]["execution_path"] == "storage_only"
    assert result["metrics"]["quantization"]["native_deployment_available"] is False
    assert result["manifest"]["coverage"]["selected_tensor_count"] == 1
    assert (tmp_path / "artifact" / "manifest.json").is_file()
    assert not (tmp_path / "artifact.partial").exists()
    assert all(
        torch.equal(before, after)
        for before, after in zip(original, model.parameters())
    )
    assert all("text" not in row for row in rows)


def test_cancelled_quantization_removes_partial_artifact(tmp_path):
    import threading

    from hqsb.backends.quantization import quantize_loaded_model

    event = threading.Event()
    event.set()
    rows = list(
        quantize_loaded_model(
            tiny_quant_model(),
            {"artifact_dir": str(tmp_path / "cancelled"), "bits": 8},
            event,
            {"hash": "test", "revision": "test", "scope": "test"},
        )
    )
    assert rows[-1]["finish_reason"] == "cancelled"
    assert rows[-1]["artifact_id"] is None
    assert not (tmp_path / "cancelled").exists()
    assert not (tmp_path / "cancelled.partial").exists()


def test_chunk_cancellation_never_writes_completed_manifest(tmp_path):
    model = tiny_quant_model()
    from hqsb.quant.model_weight_only import (
        QuantizationCancelled,
        save_model_quant_artifact,
    )

    progress = []
    with pytest.raises(QuantizationCancelled):
        save_model_quant_artifact(
            model,
            tmp_path,
            bits=8,
            group_size=None,
            source_model_hash="test",
            source_revision="test",
            row_chunk=1,
            progress=progress.append,
            cancel_check=lambda: bool(progress),
        )
    assert progress[0]["rows_processed"] == 1
    assert not (tmp_path / "manifest.json").exists()


def test_quantization_does_not_overwrite_existing_artifact(tmp_path):
    import threading

    from hqsb.backends.quantization import quantize_loaded_model

    with pytest.raises(FileExistsError):
        list(
            quantize_loaded_model(
                tiny_quant_model(),
                {"artifact_dir": str(tmp_path), "bits": 8},
                threading.Event(),
                {"hash": "test", "revision": "test", "scope": "test"},
            )
        )
