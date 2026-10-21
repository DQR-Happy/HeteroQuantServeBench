"""Contracts and real HTTP/ownership tests; no GPU runtime imports."""
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient
from hqsb.backends.native_engine import NativeEngineProvider
from hqsb.console.app import create_app
from hqsb.console.config import DeploymentConfig, Settings
from hqsb.console.service import ConsoleService, ServiceError
from hqsb.core.native_engine import catalogue, launch_arguments, resolve_options
from hqsb.core.optimization import catalogue as optimization_catalogue
from hqsb.serving.native_benchmark import execute, validate


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_translation_and_layer_contract(engine):
    config = {"model": "test", "model_path": "/weights", "context_limit": 2048}
    options = resolve_options(engine, {"prefix_cache": True, "enforce_eager": False})
    argv = launch_arguments(engine, config, options, 9000)
    assert "--model" in argv if engine == "vllm" else "--model-path" in argv
    assert "--enforce-eager" not in argv and "--disable-cuda-graph" not in argv
    assert "--no-enable-prefix-caching" not in argv and "--disable-radix-cache" not in argv
    assert {2, 3, 4, 6, 7, 8, 9} <= {layer for row in catalogue(engine)["items"] for layer in row["layers"]}
    assert optimization_catalogue(engine)["engine"] == engine


@pytest.mark.parametrize("raw", [{"enforce_eager": "false"}, {"memory_fraction": 1.0},
                                     {"max_running_requests": True}, {"custom_arg": "--foo"},
                                     {"attention_backend": "invented"}, {"block_size": 3}])
def test_invalid_options_rejected(raw):
    with pytest.raises(ValueError):
        resolve_options("vllm", raw)


def test_benchmark_bounds():
    with pytest.raises(ValueError, match="EXPECTED_OUTPUT_COUNT"):
        validate({"expected_epoch": "e", "prompts": ["x"], "expected_outputs": ["x", "y"]})
    with pytest.raises(ValueError, match="EVIDENCE_LIMIT"):
        validate({"expected_epoch": "e", "rounds": 32, "max_output_tokens": 4096})


@pytest.fixture
def upstream():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.received.append((self.path, body))
            if body.get("prompt") == "reject":
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"context limit")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            choices = [{"delta": {"content": "4"}, "finish_reason": None}]
            if self.path.endswith("/completions") and not self.path.endswith("/chat/completions"):
                choices = [{"text": "4", "finish_reason": None}]
            events = [{"choices": choices}, {"choices": [{"finish_reason": "stop"}]},
                      {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}]
            for event in events:
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = NativeEngineProvider({"provider": "vllm", "model": "test", "input_mode": "chat"})
    provider.base_url = f"http://127.0.0.1:{server.server_port}"
    provider.receipt = {"requested": {"seed": 0}, "instance_id": "fixture"}
    provider.snapshot = lambda: {"scope": "fixture_external_boundary"}
    yield provider, server
    server.shutdown()
    thread.join()
    server.server_close()


def test_real_stream_and_shared_loadgen(upstream):
    provider, server = upstream
    plan = {"expected_epoch": "e", "prompts": ["2+2"], "expected_outputs": ["4"], "rounds": 4,
            "concurrency": 2, "fixed_output_length": True}
    result = execute(provider, plan, threading.Event())
    receipt = result["metrics"]["native_benchmark"]
    assert receipt["summary"]["completed"] == 4
    assert receipt["summary"]["output_tokens"] == 4
    assert receipt["summary"]["quality_exact_match_rate"] == 1
    assert receipt["summary"]["output_tokens_per_s"] > 0
    assert receipt["summary"]["ttft_ms"]["p99"] is None
    assert len(server.received) == 5  # one separate warmup
    assert all(body["ignore_eos"] and body["temperature"] == 0 for _, body in server.received)
    assert all(row["metrics"]["runtime_first_token_ms"] is None for row in receipt["requests"])


def test_completion_transport(upstream):
    provider, server = upstream
    events = list(provider.stream({"prompt": "2+2=", "max_output_tokens": 4, "remaining_ms": 1000}, threading.Event()))
    assert server.received[0][0] == "/v1/completions"
    assert events[-1]["text"] == "4"
    assert events[-1]["metrics"]["output_tokens"] == 1


def test_request_rejection_preserves_provider(upstream):
    provider, _ = upstream
    rejected = list(provider.generate({"prompt": "reject", "max_output_tokens": 4}, threading.Event()))[-1]
    assert rejected["request_error"]["code"] == "NATIVE_REQUEST_REJECTED"
    valid = list(provider.generate({"prompt": "2+2=", "max_output_tokens": 4}, threading.Event()))[-1]
    assert valid["text"] == "4"


def test_observation_off_skips_instance_and_process_sampling(upstream):
    provider, _ = upstream
    def forbidden():
        raise AssertionError("off must skip counter and procfs sampling")
    provider.snapshot = forbidden
    result = list(provider.generate({"prompt": "2+2=", "max_output_tokens": 4, "observation_mode": "off"}, threading.Event()))[-1]
    assert result["metrics"]["native_counter_collection"] == "off"
    assert "native_counters" not in result["metrics"]


def test_supervisor_eof_stops_upstream(tmp_path):
    pidfile = tmp_path / "pid"
    code = "import os,time,pathlib; pathlib.Path(" + repr(str(pidfile)) + ").write_text(str(os.getpid())); time.sleep(120)"
    owner = subprocess.Popen([sys.executable, "-m", "hqsb.backends.native_supervisor", sys.executable, "-c", code], stdin=subprocess.PIPE)
    try:
        for _ in range(100):
            if pidfile.exists():
                break
            time.sleep(.03)
        pid = int(pidfile.read_text())
        owner.stdin.close()
        owner.wait(timeout=10)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if owner.poll() is None:
            owner.stdin.close()
            owner.wait(timeout=10)


def test_supervisor_reaps_rank_ignoring_term(tmp_path):
    pidfile = tmp_path / "rank"
    rank = ("import signal,os,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "pathlib.Path(" + repr(str(pidfile)) + ").write_text(str(os.getpid())); time.sleep(120)")
    server = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(rank) + "]); time.sleep(120)"
    owner = subprocess.Popen([sys.executable, "-m", "hqsb.backends.native_supervisor", sys.executable, "-c", server], stdin=subprocess.PIPE)
    try:
        for _ in range(100):
            if pidfile.exists():
                break
            time.sleep(.03)
        pid = int(pidfile.read_text())
        owner.stdin.close()
        owner.wait(timeout=10)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if owner.poll() is None:
            owner.stdin.close()
            owner.wait(timeout=10)


def test_registration_persistence_and_load_contract(tmp_path):
    settings = Settings(data_dir=tmp_path / "state", evidence_root=tmp_path, web_dist=tmp_path,
        deployments=[DeploymentConfig(id="model", name="Test", manifest="/manifest", device_lock_path="/lock")])
    class Actor:
        def __init__(self, config):
            self.config = config
        def execute(self, kind, payload, *args):
            return {"engine_options": payload.get("engine_options")}
        def close(self):
            pass
    service = ConsoleService(settings, actor_factory=Actor)
    app = create_app(settings, service=service, monitor=False)
    with TestClient(app) as client:
        response = client.post("/api/console/v1/native-engines/deployments", json={"source_deployment_id": "model", "engine": "vllm"}, headers={"X-HQSB-Client": "console"})
        assert response.status_code == 201, response.text
        assert response.json()["id"] == "model-vllm"
        run = service.submit("load", "model-vllm", {"engine_options": {"prefix_cache": True}}, uuid.uuid4().hex)
        for _ in range(100):
            if service.store.get(run["id"])["state"] == "completed":
                break
            time.sleep(.01)
        state = service.states["model-vllm"]
        assert state["detail"]["engine_options"]["prefix_cache"] is True
        with pytest.raises(ServiceError, match="Only one"):
            service.submit("load", "model", {}, uuid.uuid4().hex)
        with pytest.raises(ServiceError, match="卸载"):
            service.submit("load", "model-vllm", {}, uuid.uuid4().hex)
    settings2 = settings.model_copy(update={"data_dir": settings.data_dir})
    fresh = ConsoleService(settings2, actor_factory=Actor)
    app2 = create_app(settings2, service=fresh, monitor=False)
    assert fresh.configs["model-vllm"].provider == "vllm"
    with TestClient(app2) as client:
        response = client.post("/api/console/v1/native-engines/deployments/model-vllm/retire",
                               json={}, headers={"X-HQSB-Client": "console"})
        assert response.status_code == 200, response.text
        assert json.loads((settings.data_dir / "native-deployments.json").read_text())["items"] == []
        assert fresh.store.get(run["id"])["state"] == "completed"


def test_comparison_rejects_mismatched_workload_and_hardware(tmp_path):
    settings = Settings(data_dir=tmp_path / "state", evidence_root=tmp_path, web_dist=tmp_path,
                        deployments=[])
    service = ConsoleService(settings)
    app = create_app(settings, service=service, monitor=False)
    def record(engine, speed):
        receipt = {"workload_sha256": "same", "measurement_profile": "http",
                   "engine": {"engine": engine, "model_manifest_sha256": "weights", "requested": resolve_options(engine, {}),
                              "hardware_inventory": {"availability": "measured", "csv": "gpu0"},
                              "launch_environment": {}},
                   "plan": {"expected_outputs": ["4"]},
                   "requests": [{"metrics": {"input_tokens": 5}}],
                   "summary": {"failed_or_cancelled": 0, "arrival_valid": True, "usage_complete": True,
                               "output_tokens_per_s": speed, "quality_exact_match_rate": 1}}
        run, _ = service.store.create("native_benchmark", {"deployment": {"input_mode": "chat"}}, uuid.uuid4().hex, "fixture")
        return service.store.update(run["id"], {"state": "completed", "metrics": {"native_benchmark": receipt}})
    a, b = record("vllm", 10), record("sglang", 20)
    with TestClient(app) as client:
        body = {"baseline_run_id": a["id"], "candidate_run_id": b["id"]}
        def compare():
            return client.post("/api/console/v1/native-engines/comparisons", json=body,
                               headers={"X-HQSB-Client": "console"}).json()
        assert compare()["output_tokens_per_s_ratio"] == 2
        b["metrics"]["native_benchmark"]["engine"]["requested"]["prefix_cache"] = True
        b["metrics"]["native_benchmark"]["summary"]["quality_exact_match_rate"] = .5
        service.store.update(b["id"], {"metrics": b["metrics"]})
        body["declared_changes"] = ["prefix_cache"]
        report = compare()
        assert report["comparable"] is True and report["quality_gate"]["status"] == "failed"
        assert report["eligible_for_adoption"] is False
        b["metrics"]["native_benchmark"]["workload_sha256"] = "different"
        b["metrics"]["native_benchmark"]["engine"]["hardware_inventory"]["csv"] = "other_gpu"
        service.store.update(b["id"], {"metrics": b["metrics"]})
        report = compare()
        assert report["comparable"] is False
        assert report["output_tokens_per_s_ratio"] is None
        assert set(report["mismatches"]) == {"workload_sha256", "hardware_inventory_unverified_or_different"}
