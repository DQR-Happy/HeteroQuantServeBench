"""Report semantics and admission isolation, without importing CUDA runtimes."""
import sqlite3
import sys
import os
import time
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from hqsb.console.app import create_app
from hqsb.console.config import Settings, DeploymentConfig
from hqsb.console.diagnostic_jobs import DiagnosticJobs
from hqsb.console.service import ConsoleService, ServiceError
from hqsb.core.diagnostics import DiagnosticPlan, profiler_arguments
from hqsb.core.native_engine import resolve_options
from hqsb.core.optimization import fingerprint
from hqsb.observation.nsight_reports import nsys_timeline, ncu_metrics
from hqsb.observation.storage import sha256


def test_timeline_overlapping_streams_are_unioned(tmp_path):
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            INSERT INTO StringIds VALUES(1,'attention'),(2,'cudaDeviceSynchronize');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER,demangledName INTEGER,correlationId INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(0,10,0,1,1,4),(5,15,0,2,1,5),(20,30,0,1,1,6),(0,100,1,1,1,7);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER,bytes INTEGER,copyKind INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(14,18,0,3,4096,1);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(1,31,2);
        ''')
    result = nsys_timeline(path)
    gpu = next(d for d in result["device_activity"] if d["device"] == "0")
    assert gpu["activity_union_ns"] == 28
    assert gpu["gap_within_activity_span_ns"] == 2
    assert result["top_kernels"][0]["summed_duration_ns"] == 100
    assert result["copies"][0]["bytes"] == 4096
    assert result["cuda_apis"][0]["summed_host_duration_ns"] == 30
    assert result["timeline"][0]["correlation_id"] == 4


def test_ncu_units_missing_values_and_process_scope():
    header = '"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    rows = '"0","101","gemm","lts__t_sector_hit_rate.pct","%","93.5"\n"0","101","gemm","dram__bytes.sum","byte","1,024"\n"0","102","gemm","dram__bytes.sum","byte","N/A"\n'
    result = ncu_metrics("==PROF==\n" + header + rows)
    assert len(result["kernels"]) == 2
    assert result["kernels"][0]["metrics"][0]["value"] == 93.5
    assert result["kernels"][0]["metrics"][1]["value"] == 1024
    assert result["kernels"][1]["metrics"][0]["value"] is None
    assert ncu_metrics("==WARNING== no kernels")["availability"] == "empty"


@pytest.mark.parametrize("bad", [{"tool": "shell"}, {"command": ["rm"]}, {"launch_count": 33}, {"kernel_filter": "["}])
def test_plan_rejects_unbounded_or_invalid_input(bad):
    with pytest.raises(ValidationError):
        DiagnosticPlan(source_run_id="run_" + "1" * 32, **bad)


def test_arguments_keep_regex_single_argv_and_warmup_excluded():
    plan = DiagnosticPlan(source_run_id="run_" + "1" * 32, tool="ncu", kernel_filter='x; echo "$HOME"')
    argv = profiler_arguments(plan, "/tool", "/output", ["/python", "runner"])
    assert 'regex:x; echo "$HOME"' in argv
    assert argv[argv.index("--profile-from-start") + 1] == "off"
    assert argv[argv.index("--target-processes") + 1] == "all"
    assert argv[argv.index("--cache-control") + 1] == "none"


@pytest.fixture
def setup(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    manifest = tmp_path / "manifest"
    manifest.write_text("fixture")
    config = DeploymentConfig(id="native", name="Fixture", provider="vllm", model_path=str(weights),
                              manifest=str(manifest), engine_python=sys.executable, device_lock_path=str(tmp_path / "lease"))
    settings = Settings(data_dir=tmp_path / "data", evidence_root=tmp_path, web_dist=tmp_path / "web", deployments=[config],
                        diagnostic_tools={"nsys": sys.executable}, artifact_budget_bytes=1024**3)
    svc = ConsoleService(settings)
    source, _ = svc.store.create("native_benchmark", {"deployment": config.public()}, "source", "fixture")
    workload = {"expected_epoch": "e", "prompts": ["test"], "rounds": 1}
    receipt = {"plan": workload, "workload_sha256": fingerprint({k: v for k, v in workload.items() if k != "expected_epoch"}),
               "engine": {"model_manifest_sha256": sha256(manifest), "requested": resolve_options("vllm", {})}, "requests": []}
    svc.store.update(source["id"], {"state": "completed", "metrics": {"native_benchmark": receipt}})
    jobs = DiagnosticJobs(settings, svc)
    yield settings, svc, jobs, source["id"]
    jobs.close()
    svc.close()


def test_admission_is_idempotent_and_blocks_concurrent_load(setup, monkeypatch):
    settings, svc, jobs, source = setup
    calls = []
    monkeypatch.setattr(jobs.executor, "submit", lambda *args: calls.append(args))
    first = jobs.start({"source_run_id": source}, "idempotent")
    assert jobs.start({"source_run_id": source}, "idempotent")["id"] == first["id"]
    assert len(calls) == 1
    with pytest.raises(ServiceError, match="独占"):
        svc.submit("load", "native", {}, "load")
    with pytest.raises(ServiceError):
        jobs.start({"source_run_id": source, "launch_count": 2}, "idempotent")


def test_resident_model_requires_explicit_unload(setup):
    _, svc, jobs, source = setup
    svc.states["native"]["state"] = "ready"
    with pytest.raises(ServiceError) as error:
        jobs.start({"source_run_id": source}, "busy")
    assert error.value.code == "DIAGNOSTIC_UNLOAD_REQUIRED"
    assert svc.pending == 0


def test_artifact_paths_and_private_launch_config(setup, tmp_path):
    _, svc, jobs, source = setup
    run, _ = svc.store.create("diagnostic", {}, "diag", "diag")
    root = jobs.root / run["id"]
    root.mkdir()
    (root / "input.json").write_text("private executable/config")
    (root / "receipt.json").write_text("{}")
    (root / "manifest.json").write_text(json.dumps({"artifacts": [{"path": "receipt.json", "bytes": 2, "sha256": sha256(root / "receipt.json")}]}))
    (root / "escape").symlink_to(tmp_path / "manifest")
    assert jobs.artifact(run["id"], "receipt.json").name == "receipt.json"
    for relative in ("input.json", "./input.json", "../x", "escape"):
        with pytest.raises(ServiceError):
            jobs.artifact(run["id"], relative)
    (root / "receipt.json").write_text("changed")
    with pytest.raises(ServiceError, match="封存清单"):
        jobs.artifact(run["id"], "receipt.json")


def test_api_exposes_contract_and_preserves_noauth(setup):
    settings, svc, jobs, source = setup
    with TestClient(create_app(settings, service=svc, monitor=False)) as client:
        assert client.get("/api/console/v1/diagnostics/catalog").status_code == 200
        assert client.post("/api/console/v1/diagnostics/jobs", headers={"Idempotency-Key": "a", "X-HQSB-Client": "console"},
                           json={"source_run_id": source, "command": ["sh"]}).status_code == 422


def test_cancel_reaps_real_profiler_process_tree_and_releases_owner(setup, tmp_path):
    settings, svc, jobs, source = setup
    tool = tmp_path / "profiler"
    tool.write_text(f'''#!{sys.executable}
import json, pathlib, subprocess, sys, time
if '--version' in sys.argv:
    print('fixture profiler 1'); raise SystemExit(0)
root = pathlib.Path(sys.argv[sys.argv.index('--output') + 1]).parent
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
(root / 'child-pid').write_text(str(child.pid))
(root / 'progress.json').write_text(json.dumps({{'stage': 'capturing'}}))
time.sleep(600)
''')
    tool.chmod(0o700)
    settings.diagnostic_tools["nsys"] = str(tool)
    run = jobs.start({"source_run_id": source}, "cancel")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not (jobs.root / run["id"] / "child-pid").exists():
        time.sleep(.05)
    pid = int((jobs.root / run["id"] / "child-pid").read_text())
    svc.cancel(run["id"])
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and svc.store.get(run["id"])["state"] not in {"cancelled", "failed"}:
        time.sleep(.05)
    final = svc.store.get(run["id"])
    assert final["state"] == "cancelled", final
    assert final["cleanup"] == "succeeded"
    assert svc.pending == 0 and svc.benchmark_owner is None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert json.loads((jobs.root / run["id"] / "manifest.json").read_text())["status"] == "cancelled"


def test_comparison_checks_cache_policy_and_retains_output_changes():
    from copy import deepcopy
    from hqsb.observation.diagnostic_comparison import compare
    a = {"run_id": "a", "tool": "nsys", "source_workload_sha256": "w", "capture_window": "steady",
         "replay_mode": "single_pass", "cache_control": "none", "clock_control": "none", "plan": {},
         "status": "completed", "cleanup": "succeeded", "replay_workloads_match_source": True,
         "replay_environment_matches_source": True, "replay_outputs_match_source": False,
         "source_engine": {"versions": {}, "model_manifest_sha256": "m", "hardware_inventory": {}, "requested": {"prefix_cache": False}},
         "panel": {"top_kernels": [{"name": "gemm", "device": "0", "launches": 2, "summed_duration_ns": 2000000}]}}
    b = deepcopy(a)
    b["run_id"] = "b"
    b["source_engine"]["requested"]["prefix_cache"] = True
    b["panel"]["top_kernels"][0]["launches"] = 1
    pair = compare(a, b)
    assert pair["comparable"] and pair["outputs_match_source"] == [False, False]
    assert pair["configuration_changes"]["prefix_cache"] == {"baseline": False, "candidate": True}
    assert pair["kernel_changes"][0]["candidate_launches"] == 1
    b["cache_control"] = "all"
    assert "cache_control" in compare(a, b)["mismatches"]


def test_permission_denial_stops_a_stuck_profiler_promptly(setup, tmp_path):
    settings, svc, jobs, source = setup
    tool = tmp_path / "denied-profiler"
    tool.write_text(f'''#!{sys.executable}
import sys, time
if '--version' in sys.argv:
    print('fixture profiler 1'); raise SystemExit(0)
print('==ERROR== ERR_NVGPUCTRPERM', flush=True)
time.sleep(600)
''')
    tool.chmod(0o700)
    settings.diagnostic_tools["ncu"] = str(tool)
    run = jobs.start({"source_run_id": source, "tool": "ncu"}, "permission")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and svc.store.get(run["id"])["state"] not in {"failed", "completed"}:
        time.sleep(.05)
    final = svc.store.get(run["id"])
    assert final["state"] == "failed"
    assert "GPU_COUNTER_PERMISSION_REQUIRED" in final["error"]
    assert final["cleanup"] == "succeeded" and svc.benchmark_owner is None
    assert jobs.catalog()["items"][1]["collection_permission"] == "permission_required"


def test_capture_window_surrounds_requests_and_excludes_warmup_snapshots():
    from contextlib import contextmanager
    from types import SimpleNamespace
    import threading
    from hqsb.serving.native_benchmark import execute
    order = []

    def stream(request, cancel, client=None):
        order.append("request")
        yield {"kind": "result", "text": "4", "finish_reason": "stop", "metrics": {
            "input_tokens": 1, "output_tokens": 1, "proxy_first_content_ms": 1, "runtime_e2e_ms": 2}}

    def snapshot():
        order.append("snapshot")
        return {}

    @contextmanager
    def window():
        order.append("start")
        try:
            yield
        finally:
            order.append("stop")

    provider = SimpleNamespace(config={"input_mode": "chat"}, stream=stream, snapshot=snapshot, receipt={})
    result = execute(provider, {"expected_epoch": "e", "prompts": ["x", "y"], "rounds": 1,
                              "warmup_requests": 1}, threading.Event(), capture_window=window)
    assert order == ["request", "snapshot", "start", "request", "request", "stop", "snapshot"]
    assert result["metrics"]["native_benchmark"]["measurement_profile"] == "native-common-http-diagnostic-v1"


def test_interrupted_attempt_keeps_engine_evidence_and_output_unavailable(tmp_path):
    engine_dir = tmp_path / "attempts" / "one" / "engine" / "instance"
    engine_dir.mkdir(parents=True)
    (engine_dir / "receipt.json").write_text(json.dumps({"instance_id": "instance"}))
    report = {"run_id": "d", "status": "failed"}
    source = {"metrics": {"native_benchmark": {"requests": [{"text": "4"}], "workload_sha256": "w", "engine": {}}}}
    DiagnosticJobs.seal(tmp_path, report, source)
    assert report["attempts"][0]["engine"]["instance_id"] == "instance"
    assert report["replay_outputs_complete"] is False
    assert report["replay_outputs_match_source"] is None


def test_ere_job_reuses_source_identity_and_scoped_executor(setup, monkeypatch):
    settings, svc, jobs, source = setup
    calls = []
    monkeypatch.setattr(jobs, "inventory", lambda tool: {"tool": tool, "installed": True, "executable": "/private/ere"})
    monkeypatch.setattr(jobs.executor, "submit", lambda *args: calls.append(args))
    before = svc.store.get(source)["metrics"]["native_benchmark"]["engine"]["requested"].copy()
    run = jobs.start({"source_run_id": source, "tool": "eretrace", "ere_kernel": "rms"}, "ere")
    assert calls[0][2]["_eretrace"]["runtime"] == "/private/ere"
    assert calls[0][2]["_eretrace"]["plan"]["ere_kernel"] == "rms"
    assert svc.store.get(source)["metrics"]["native_benchmark"]["engine"]["requested"] == before
    assert jobs.start({"source_run_id": source, "tool": "eretrace", "ere_kernel": "rms"}, "ere")["id"] == run["id"]
    assert len(calls) == 1
    svc.pending = 0
    svc.benchmark_owner = None


def test_ere_graph_source_requires_an_explicit_new_baseline(setup):
    settings, svc, jobs, source = setup
    metrics = svc.store.get(source)["metrics"]
    metrics["native_benchmark"]["engine"]["requested"]["enforce_eager"] = False
    svc.store.update(source, {"metrics": metrics})
    with pytest.raises(ServiceError, match="Eager"):
        jobs.start({"source_run_id": source, "tool": "eretrace"}, "graph")
    assert svc.pending == 0 and svc.benchmark_owner is None
    assert svc.store.get(source)["metrics"]["native_benchmark"]["engine"]["requested"]["enforce_eager"] is False
