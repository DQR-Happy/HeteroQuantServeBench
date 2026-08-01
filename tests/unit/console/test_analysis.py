"""HTTP analysis tests with explicit worker doubles; never hardware evidence."""

import builtins
import hashlib
import io
import json
import threading
import time
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hqsb.console.app import PREFIX, create_app
from hqsb.console import captures as captures_module
from hqsb.console.captures import CaptureRepository, safe_directory
from hqsb.console.config import DeploymentConfig, Settings
from hqsb.console.service import ConsoleService
from hqsb.console.store import TERMINAL


class AnalysisActorDouble:
    """Controlled acknowledgements exercise the API, not quantization math."""

    def __init__(self, config):
        self.config = config
        self.calls = []
        self.quant_started = threading.Event()
        self.generate_started = threading.Event()
        self.quant_release = threading.Event()
        self.generate_release = threading.Event()
        self.quant_release.set()
        self.generate_release.set()
        self.fail_quantization = False
        self.closed = False

    @staticmethod
    def _wait(release, is_cancelled):
        while not release.wait(0.005):
            if is_cancelled():
                return False
        return not is_cancelled()

    def execute(self, kind, payload, on_output, is_cancelled, timeout_s):
        self.calls.append((kind, dict(payload)))
        if kind == "load":
            return {"test_only": True, "memory": {"allocated": 100}}
        if kind == "unload":
            return {"test_only": True}
        if kind == "generate":
            self.generate_started.set()
            if not self._wait(self.generate_release, is_cancelled):
                return {"finish_reason": "cancelled", "test_only": True}
            on_output({"text": "test output", "output_tokens": 1})
            return {
                "finish_reason": "length",
                "text": "test output",
                "metrics": {"output_tokens": 1, "measurement_profile": "test_double"},
            }
        if kind == "quantize":
            # A deliberately recognizable fake manifest. Production artifacts
            # must come from the isolated runtime, not these test fixtures.
            directory = Path(payload["artifact_dir"])
            directory.mkdir()
            (directory / "manifest.json").write_text(
                json.dumps({"schema": "test_only", "bits": payload["bits"]})
            )
            on_output(
                {
                    "kind": "progress",
                    "completed_tensors": 1,
                    "total_tensors": 2,
                    "test_only": True,
                }
            )
            self.quant_started.set()
            if not self._wait(self.quant_release, is_cancelled):
                return {"finish_reason": "cancelled", "test_only": True}
            if self.fail_quantization:
                raise RuntimeError("deliberate quantization test-double failure")
            return {
                "test_only": True,
                "execution_label": "storage_only",
                "native_low_bit_kernel": False,
                "quality": "not_evaluated",
                "metrics": {"measurement_profile": "test_double"},
            }
        raise AssertionError(f"Unexpected test operation: {kind}")

    def close(self):
        self.closed = True
        self.quant_release.set()
        self.generate_release.set()


@pytest.fixture
def api(tmp_path, monkeypatch):
    # Deterministic budget inputs; tests never depend on current GPU/host load.
    monkeypatch.setattr(
        "hqsb.console.service.host_memory",
        lambda label="current": {
            "label": label,
            "available_bytes": 8 * 1024**3,
            "source": "test_double",
        },
    )
    monkeypatch.setattr(
        "hqsb.console.service.shutil.disk_usage",
        lambda path: SimpleNamespace(free=16 * 1024**3),
    )
    settings = Settings(
        data_dir=tmp_path / "state",
        evidence_root=tmp_path,
        web_dist=tmp_path / "dist",
        deployments=[
            DeploymentConfig(id="local", name="test local"),
            DeploymentConfig(
                id="external",
                name="test external",
                provider="openai",
                base_url="http://127.0.0.1:9/v1",
            ),
        ],
    )
    service = ConsoleService(settings, actor_factory=AnalysisActorDouble)
    app = create_app(settings, "analysis-test-token", service=service, monitor=False)
    with TestClient(
        app, headers={"Authorization": "Bearer analysis-test-token"}
    ) as client:
        yield SimpleNamespace(
            client=client,
            service=service,
            settings=settings,
            actor=service.actors["local"],
        )


def post(api, path, body=None, *, key=None):
    return api.client.post(
        PREFIX + path, json=body, headers={"Idempotency-Key": key or uuid.uuid4().hex}
    )


def finished(api, run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        row = api.service.store.get(run_id)
        if row["state"] in TERMINAL:
            return row
        time.sleep(0.005)
    raise AssertionError("Test-double operation did not reach terminal state")


def load(api, deployment="local"):
    response = post(api, f"/deployments/{deployment}/load")
    assert response.status_code == 202, response.text
    assert finished(api, response.json()["id"])["state"] == "completed"
    return next(
        row["epoch"] for row in api.service.deployments() if row["id"] == deployment
    )


def generation(epoch, **overrides):
    return {
        "deployment_id": "local",
        "expected_epoch": epoch,
        "messages": [{"role": "user", "content": "private input sentinel 71893"}],
        "max_output_tokens": 8,
        "save_input": False,
        **overrides,
    }


def quantization(epoch, **overrides):
    return {"deployment_id": "local", "expected_epoch": epoch, "bits": 4, **overrides}


def recorded_run(api, *, kind="generate", state="completed", metrics=None):
    run, _ = api.service.store.create(
        kind, {"test_only": True}, uuid.uuid4().hex, "test-digest"
    )
    return api.service.store.update(
        run["id"], {"state": state, "metrics": metrics or {}}
    )


def put_trace(api, run, events):
    directory = api.settings.data_dir / "captures" / run["id"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    return path


def test_resources_and_capabilities_use_no_torch_import(api, monkeypatch):
    original_import = builtins.__import__

    def deny_device_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "triton", "torch_npu"}:
            raise AssertionError(f"Control-plane resource query imported {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", deny_device_import)
        response = api.client.get(PREFIX + "/resources")
        capabilities = api.client.get(PREFIX + "/capabilities")
    assert response.status_code == capabilities.status_code == 200
    body = response.json()
    assert body["host"]["memory_scope"] == "control_host"
    assert body["processes"][0]["role"] == "console_api"
    assert body["limitations"]
    items = {row["id"]: row for row in capabilities.json()["items"]}
    assert items["memory_accounting"]["status"] == "available"
    assert items["byte_register_trace"]["status"] == "unavailable"
    assert items["native_low_bit"]["status"] == "unavailable"
    assert items["ncu_live"]["status"] == "unavailable"


def test_old_request_has_no_fabricated_capture_and_running_is_pending(api):
    old = recorded_run(api)
    response = api.client.get(PREFIX + f"/runs/{old['id']}/analysis")
    assert response.status_code == 200
    assert response.json()["observation"] is None
    assert response.json()["trace"]["status"] == "not_collected"
    assert api.client.get(PREFIX + f"/runs/{old['id']}/trace").status_code == 404
    running = recorded_run(api, state="running")
    analysis = api.client.get(PREFIX + f"/runs/{running['id']}/analysis").json()
    assert analysis["trace"]["status"] == "pending"
    assert (
        api.client.get(PREFIX + f"/runs/{running['id']}/trace/events").status_code
        == 409
    )
    assert api.client.get(PREFIX + f"/runs/{running['id']}/bundle").status_code == 409


def test_dataflow_endpoint_uses_real_copy_events_and_source_hash(api):
    run = recorded_run(
        api,
        metrics={
            "observation": {
                "profile": {
                    "activities": ["CPU", "CUDA"],
                    "coverage": {"decode_steps": 8},
                }
            }
        },
    )
    path = put_trace(
        api,
        run,
        [
            {
                "ph": "X",
                "cat": "gpu_memcpy",
                "name": "Memcpy HtoD",
                "ts": 10,
                "dur": 3,
                "args": {"bytes": 512, "correlation": 7},
            },
            {
                "ph": "X",
                "cat": "gpu_memcpy",
                "name": "Memcpy DtoH",
                "ts": 20,
                "dur": 4,
                "args": {"bytes": 8},
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaMemcpyAsync",
                "ts": 9,
                "dur": 8,
                "args": {"bytes": 512},
            },
            {"ph": "X", "cat": "kernel", "name": "GEMM", "ts": 15, "dur": 3},
        ],
    )
    response = api.client.get(PREFIX + f"/runs/{run['id']}/dataflow")
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == run["id"]
    assert body["totals"]["copy_events"] == 2
    assert body["totals"]["kernel_events"] == 1
    assert body["totals"]["bytes"] == 520
    assert (
        body["coverage"]["trace_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert body["coverage"]["profile_window"]["decode_steps"] == 8


def test_dataflow_missing_pending_and_unauthorized_states(api):
    run = recorded_run(api)
    route = PREFIX + f"/runs/{run['id']}/dataflow"
    response = api.client.get(route)
    assert response.status_code == 200
    assert response.json()["status"] == "not_collected"
    assert response.json()["totals"]["bytes"] is None
    assert (
        api.client.get(route, headers={"Authorization": "Bearer invalid"}).status_code
        == 401
    )
    pending = recorded_run(api, state="running")
    assert api.client.get(PREFIX + f"/runs/{pending['id']}/dataflow").status_code == 409


def test_dataflow_rejects_symlink_as_missing_capture(api, tmp_path):
    run = recorded_run(api)
    path = put_trace(api, run, [])
    external = tmp_path / "unrelated.json"
    external.write_text('{"traceEvents":[]}')
    path.unlink()
    path.symlink_to(external)
    response = api.client.get(PREFIX + f"/runs/{run['id']}/dataflow")
    assert response.status_code == 200
    assert response.json()["status"] == "not_collected"


def test_trace_normalization_filters_pagination_and_download_digest(api):
    run = recorded_run(api)
    events = [
        {
            "ph": "X",
            "ts": 3000,
            "dur": 500,
            "name": "CUDA GEMM",
            "cat": "kernel",
            "args": {"correlation": 7, "stream": 3, "Concrete Inputs": "secret scalar"},
        },
        {"ph": "X", "ts": 1000, "dur": 1500, "name": "aten::mm", "cat": "cpu_op"},
        {"ph": "X", "ts": 4000, "dur": 250, "name": "copy", "cat": "gpu_memcpy"},
        {"ph": "M", "ts": 0, "name": "metadata", "args": {}},
        {"ph": "X", "ts": float("nan"), "dur": 1},
        {"ph": "X", "ts": 1, "dur": float("inf")},
        {"ph": "X", "ts": 1, "dur": -1},
    ]
    path = put_trace(api, run, events)
    route = PREFIX + f"/runs/{run['id']}/trace"
    page = api.client.get(route + "/events", params={"limit": 1}).json()
    assert page["total"] == 3 and page["next_offset"] == 1
    assert page["items"][0]["start_ms"] == 0
    assert page["items"][0]["duration_ms"] == 1.5
    assert page["summary"]["status"] == "partial"
    assert page["summary"]["excluded_or_truncated_events"] == 3
    tail = api.client.get(route + "/events", params={"offset": 2, "limit": 1}).json()
    assert tail["next_offset"] is None
    assert tail["items"][0]["name"] == "copy"
    filtered = api.client.get(
        route + "/events",
        params={
            "category": "kernel",
            "search": "gemm",
            "start_ms": 2.25,
            "end_ms": 2.5,
        },
    ).json()
    assert filtered["total"] == 1
    assert filtered["items"][0]["args"] == {"correlation": 7, "stream": 3}
    assert "secret scalar" not in json.dumps(filtered)
    response = api.client.get(route)
    assert response.status_code == 200
    assert response.content == path.read_bytes()
    assert (
        response.headers["x-content-sha256"]
        == hashlib.sha256(response.content).hexdigest()
    )


@pytest.mark.parametrize(
    "params",
    [
        {"start_ms": "nan"},
        {"end_ms": "inf"},
        {"start_ms": 2, "end_ms": 1},
        {"offset": -1},
        {"limit": 0},
        {"limit": 1001},
        {"search": "x" * 201},
    ],
)
def test_trace_query_limits_reject_nonfinite_and_invalid_ranges(api, params):
    run = recorded_run(api)
    response = api.client.get(PREFIX + f"/runs/{run['id']}/trace/events", params=params)
    assert response.status_code == 422


def test_trace_args_with_nonfinite_values_never_break_json_response(api):
    run = recorded_run(api)
    put_trace(
        api,
        run,
        [
            {
                "ph": "X",
                "ts": 0,
                "dur": 1,
                "args": {"bytes": float("nan"), "grid": [float("inf")]},
            }
        ],
    )
    response = api.client.get(PREFIX + f"/runs/{run['id']}/trace/events")
    assert response.status_code == 200
    json.dumps(response.json(), allow_nan=False)
    assert response.json()["items"][0]["args"] == {"bytes": None, "grid": [None]}


@pytest.mark.parametrize("chunk_bytes", [1, 7, 64])
def test_trace_stream_parser_handles_utf8_numbers_and_nested_fake_keys(
    api, monkeypatch, chunk_bytes
):
    monkeypatch.setattr(captures_module, "TRACE_CHUNK_BYTES", chunk_bytes)
    run = recorded_run(api)
    path = put_trace(api, run, [])
    actual_name = '真实 €😀 kernel; quoted "traceEvents": [ ]'
    document = {
        "metadata": {
            "traceEvents": [
                {"ph": "X", "ts": 0, "dur": 1, "name": "not a top-level event"}
            ],
            "description": 'literal "traceEvents": [{"name":"also not an event"}]',
            "number": -1.234e45,
        },
        "traceEvents": [
            {
                "ph": "X",
                "ts": 1.25e16,
                "dur": 1234.5,
                "name": actual_name,
                "cat": "kernel",
            }
        ],
        "displayTimeUnit": "ms",
    }
    path.write_bytes(json.dumps(document, ensure_ascii=False).encode("utf-8"))
    response = api.client.get(PREFIX + f"/runs/{run['id']}/trace/events")
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["status"] == "available"
    assert body["total"] == 1 and body["items"][0]["name"] == actual_name
    assert body["items"][0]["start_ms"] == 0
    assert body["items"][0]["duration_ms"] == 1.2345


def test_large_trace_remains_pageable_without_whole_file_reads(api, monkeypatch):
    run = recorded_run(api)
    path = put_trace(api, run, [])
    event = json.dumps(
        {
            "ph": "X",
            "ts": 1000,
            "dur": 1,
            "name": "bounded",
            "cat": "kernel",
            "unused": "x" * 9216,
        }
    )
    count = 4096
    # Real >36 MiB input, emitted incrementally by this test as well. The
    # unused payload must not survive normalized event projection.
    with path.open("w") as stream:
        stream.write('{"traceEvents":[')
        for index in range(count):
            if index:
                stream.write(",")
            stream.write(event)
        stream.write("]}")
    assert 36 * 1024**2 < path.stat().st_size < 64 * 1024**2
    monkeypatch.setattr(captures_module, "MAX_EVENTS", 8)

    def disallow_whole_file_read(self):
        raise AssertionError("Trace parsing must not call Path.read_bytes")

    monkeypatch.setattr(Path, "read_bytes", disallow_whole_file_read)
    response = api.client.get(
        PREFIX + f"/runs/{run['id']}/trace/events", params={"offset": 6, "limit": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 8 and len(body["items"]) == 2
    assert body["next_offset"] is None
    assert body["summary"]["status"] == "partial"
    assert body["summary"]["excluded_or_truncated_events"] == count - 8
    assert all("unused" not in row and len(row["args"]) == 0 for row in body["items"])


@pytest.mark.parametrize(
    "raw",
    [
        '{"traceEvents":[],"traceEvents":[]}',
        '{"traceEvents":[{"ph":"X","ts":0,"dur":1},]}',
        '{"traceEvents":[{"ph":"X","ts":0,"dur":1}],"deep":'
        + "[" * 2000
        + "0"
        + "]" * 2000
        + "}",
        '{"traceEvents":{}}',
        '{"traceEvents":[]} extra',
    ],
)
def test_invalid_stream_containers_and_overdeep_metadata_degrade(api, raw):
    run = recorded_run(api)
    path = put_trace(api, run, [])
    # A malformed metadata tail must invalidate already-decoded events;
    # recursive decoding failures must not escape as an API 500.
    path.write_text(raw)
    response = api.client.get(PREFIX + f"/runs/{run['id']}/trace/events")
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["status"] == "invalid"
    assert body["items"] == [] and body["total"] == 0


def test_single_trace_event_exceeding_budget_is_rejected(api, monkeypatch):
    monkeypatch.setattr(captures_module, "MAX_TRACE_VALUE_BYTES", 256)
    run = recorded_run(api)
    put_trace(api, run, [{"ph": "X", "ts": 0, "dur": 1, "name": "x" * 512}])
    response = api.client.get(PREFIX + f"/runs/{run['id']}/trace/events")
    assert response.status_code == 200
    assert response.json()["summary"]["status"] == "invalid"
    assert response.json()["items"] == []


def test_trace_paths_and_symlinks_cannot_escape_capture_root(api, tmp_path):
    root = api.settings.data_dir / "captures"
    root.mkdir(parents=True)
    outside = tmp_path / "private-trace"
    outside.mkdir()
    (outside / "trace.json").write_text('{"secret":"outside capture root"}')
    run = recorded_run(api)
    (root / run["id"]).symlink_to(outside, target_is_directory=True)
    assert api.client.get(PREFIX + f"/runs/{run['id']}/trace").status_code == 404
    second = recorded_run(api)
    (root / second["id"]).mkdir()
    (root / second["id"] / "trace.json").symlink_to(outside / "trace.json")
    assert api.client.get(PREFIX + f"/runs/{second['id']}/trace").status_code == 404
    for key in (
        "../private-trace",
        "run_" + "a" * 32 + "/../private-trace",
        "/etc",
        "run_invalid",
    ):
        with pytest.raises(KeyError):
            safe_directory(root, key)
        with pytest.raises(KeyError):
            CaptureRepository(root).trace_file(key)


def test_optimization_and_bundle_preserve_privacy_and_file_hashes(api):
    epoch = load(api)
    response = post(api, "/requests", generation(epoch))
    assert response.status_code == 202
    run = finished(api, response.json()["id"])
    report = api.client.get(PREFIX + f"/runs/{run['id']}/optimization").json()
    assert report["status"] == "hypotheses_only"
    assert any(row["id"] == "capture_needed" for row in report["findings"])
    markdown = api.client.get(
        PREFIX + f"/runs/{run['id']}/optimization", params={"format": "markdown"}
    )
    assert markdown.status_code == 200 and "待验证" in markdown.text
    bundle = api.client.get(PREFIX + f"/runs/{run['id']}/bundle")
    assert bundle.status_code == 200
    with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["raw_trace_included"] is False
        assert manifest["raw_trace_download"] is None
        for entry in manifest["files"]:
            contents = archive.read(entry["path"])
            assert len(contents) == entry["bytes"]
            assert hashlib.sha256(contents).hexdigest() == entry["sha256"]
        persisted = json.loads(archive.read("run.json"))
        assert "messages" not in persisted["config"]
        assert len(persisted["config"]["input_sha256"]) == 64
        for name in archive.namelist():
            assert b"private input sentinel 71893" not in archive.read(name)


def test_quantization_requires_ready_local_provider_and_current_epoch(api):
    unloaded = post(api, "/quantization/jobs", quantization("not-loaded"))
    assert unloaded.status_code == 409
    assert unloaded.json()["error"]["code"] == "DEPLOYMENT_NOT_READY"
    external_epoch = load(api, "external")
    external = post(
        api,
        "/quantization/jobs",
        quantization(external_epoch, deployment_id="external"),
    )
    assert external.status_code == 422
    assert external.json()["error"]["code"] == "UNSUPPORTED_PROVIDER"
    load(api)
    stale = post(api, "/quantization/jobs", quantization("old-epoch"))
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "STALE_EPOCH"


@pytest.mark.parametrize(
    "bits,group", [(4, 16), (4, 0), (4, 256), (8, 128), (8, 32), (3, None)]
)
def test_quantization_rejects_incompatible_group_sizes(api, bits, group):
    response = post(
        api, "/quantization/jobs", quantization("unused", bits=bits, group_size=group)
    )
    assert response.status_code == 422
    assert not api.actor.calls


@pytest.mark.parametrize("body_field", ["capture_dir", "artifact_dir"])
def test_clients_cannot_choose_server_artifact_paths(api, body_field):
    for path, payload in (
        ("/requests", generation("unused", **{body_field: "/tmp/client-controlled"})),
        (
            "/quantization/jobs",
            quantization("unused", **{body_field: "/tmp/client-controlled"}),
        ),
    ):
        assert post(api, path, payload).status_code == 422
    assert not api.actor.calls


def test_quantization_progress_idempotency_and_manifest_gate(api):
    epoch = load(api)
    api.actor.quant_release.clear()
    payload = quantization(epoch)
    response = post(api, "/quantization/jobs", payload, key="same-quantization")
    assert response.status_code == 202
    run_id = response.json()["id"]
    assert api.actor.quant_started.wait(2)
    assert (
        next(row for row in api.service.deployments() if row["id"] == "local")["state"]
        == "quantizing"
    )
    duplicate = post(api, "/quantization/jobs", payload, key="same-quantization")
    assert duplicate.status_code == 202 and duplicate.json()["id"] == run_id
    conflict = post(
        api, "/quantization/jobs", quantization(epoch, bits=8), key="same-quantization"
    )
    assert conflict.status_code == 409
    blocked = post(api, "/requests", generation(epoch))
    assert blocked.status_code == 409
    assert (
        api.client.get(
            PREFIX + f"/quantization/artifacts/{run_id}/manifest"
        ).status_code
        == 409
    )
    events = api.service.store.events(run_id, 0)
    assert any(
        event["kind"] == "task.progress" and event["data"]["test_only"]
        for event in events
    )
    api.actor.quant_release.set()
    run = finished(api, run_id)
    assert run["state"] == "completed"
    assert run["config"]["group_size"] == 128
    quant_calls = [payload for kind, payload in api.actor.calls if kind == "quantize"]
    assert len(quant_calls) == 1
    assert (
        Path(quant_calls[0]["artifact_dir"]).parent
        == api.settings.data_dir / "artifacts"
    )
    assert "artifact_dir" not in run["config"]
    manifest = api.client.get(PREFIX + f"/quantization/artifacts/{run_id}/manifest")
    assert manifest.status_code == 200 and manifest.json()["schema"] == "test_only"
    assert (
        manifest.headers["x-content-sha256"]
        == hashlib.sha256(manifest.content).hexdigest()
    )
    items = api.client.get(PREFIX + "/quantization/artifacts").json()["items"]
    assert items[0]["native_deployment_available"] is False
    assert items[0]["quality"] == "not_evaluated"


def test_quantization_waits_for_running_generation_in_same_worker(api):
    epoch = load(api)
    api.actor.generate_release.clear()
    generation_response = post(api, "/requests", generation(epoch))
    assert generation_response.status_code == 202
    assert api.actor.generate_started.wait(2)
    quant = post(api, "/quantization/jobs", quantization(epoch))
    assert quant.status_code == 202
    assert not api.actor.quant_started.is_set()
    api.actor.generate_release.set()
    assert finished(api, generation_response.json()["id"])["state"] == "completed"
    assert finished(api, quant.json()["id"])["state"] == "completed"
    assert [kind for kind, _ in api.actor.calls] == ["load", "generate", "quantize"]


def test_w8_uses_per_channel_configuration_without_group_size(api):
    epoch = load(api)
    response = post(api, "/quantization/jobs", quantization(epoch, bits=8))
    assert response.status_code == 202
    run = finished(api, response.json()["id"])
    assert run["state"] == "completed"
    assert run["config"]["bits"] == 8 and run["config"]["group_size"] is None
    assert run["quality"] == "not_evaluated"


@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
def test_unfinished_quantization_artifacts_are_not_downloadable(api, outcome):
    epoch = load(api)
    api.actor.quant_release.clear()
    api.actor.fail_quantization = outcome == "failed"
    response = post(api, "/quantization/jobs", quantization(epoch))
    assert response.status_code == 202
    run_id = response.json()["id"]
    assert api.actor.quant_started.wait(2)
    # The fake worker has already written a partial manifest: state gates must
    # prevent this file from looking like a completed quantization artifact.
    assert (api.settings.data_dir / "artifacts" / run_id / "manifest.json").is_file()
    if outcome == "cancelled":
        assert post(api, f"/requests/{run_id}/cancel").status_code == 200
    else:
        api.actor.quant_release.set()
    assert finished(api, run_id)["state"] == outcome
    assert (
        api.client.get(
            PREFIX + f"/quantization/artifacts/{run_id}/manifest"
        ).status_code
        == 404
    )
    deployment = next(row for row in api.service.deployments() if row["id"] == "local")
    assert deployment["state"] == ("ready" if outcome == "cancelled" else "failed")


def test_manifest_symlink_is_not_downloadable_even_for_completed_run(api, tmp_path):
    run = recorded_run(api, kind="quantize")
    directory = api.settings.data_dir / "artifacts" / run["id"]
    directory.mkdir(parents=True)
    secret = tmp_path / "private.json"
    secret.write_text('{"private":true}')
    (directory / "manifest.json").symlink_to(secret)
    assert (
        api.client.get(
            PREFIX + f"/quantization/artifacts/{run['id']}/manifest"
        ).status_code
        == 404
    )
