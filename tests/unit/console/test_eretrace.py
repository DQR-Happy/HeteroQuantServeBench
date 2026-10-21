"""Exact formats, provenance, admission and comparison for the ERE adapter."""
from copy import deepcopy
import hashlib
import json
import struct

import numpy as np
import pytest

from hqsb.core.diagnostics import DiagnosticPlan
from hqsb.observation.eretrace_reports import frames, summarize, decode_reference
from hqsb.observation.eretrace_ptx import cached_ptx
from hqsb.observation.eretrace_tools import discover, validate_scope

DTYPE = np.dtype([("start", "<u8"), ("elapsed", "<u4"), ("sm", "<u4")])


def fixture(root, *, recovery=False, encoded=False, warps=4):
    root.mkdir(exist_ok=True)
    n = 3 * warps
    a = np.empty(n, dtype=DTYPE)
    a["start"] = 1000 + np.arange(n) * 2
    a["elapsed"] = np.arange(n) + 10
    a["sm"] = np.repeat(np.arange(3), warps)
    header = struct.pack("<10IQ", 3, 1, 1, warps * 32, 1, 1, 0, 1, 16, 32, 48)
    if encoded:
        bases = np.zeros(256, dtype="<u8")
        bases[:3] = a["start"][::warps]
        if warps <= 2:
            payload = struct.pack("<4I", 0, 3, 2, 1) + bases.tobytes()
            payload += b"".join(int(v).to_bytes(3, "little", signed=True) for v in (a["start"] - bases[a["sm"]]))
            payload += a["elapsed"].astype("<u2").tobytes() + a["sm"].astype("u1").tobytes()
        else:
            payload = struct.pack("<4I", 0, warps, 2, 5) + bases.tobytes()
            payload += b"".join(b"\0\0\0" + bytes([sm]) for sm in range(3))
            bitstream = 0
            locals_ = [int(a["start"][i] - bases[a["sm"][i]]) for i in range(n) if i % warps]
            for index, value in enumerate(locals_):
                bitstream |= (value & 4095) << (index * 12)
            payload += bitstream.to_bytes((len(locals_) * 12 + 7) // 8, "little")
            payload += a["elapsed"].astype("<u2").tobytes()
        word = 3 << 62 | len(payload)
        assert decode_reference(payload, n, warps).tobytes() == a.tobytes()
    else:
        payload = a.tobytes()
        word = (1 << 62 | len(payload)) if recovery else 0
    (root / "arena-0.nts").write_bytes(b"NTRF022\n" + header + struct.pack("<Q", word) + payload)
    (root / "hqsb-launches.jsonl").write_text(json.dumps({"captured": True, "kernel": "rms", "grid": [3, 1, 1], "block": [warps * 32, 1, 1]}) + "\n")
    (root / "coverage.json").write_text(json.dumps({"captured_launches": 1}))
    return a, header


@pytest.mark.parametrize("warps", [1, 2, 4, 5])
@pytest.mark.parametrize("encoded,recovery", [(False, False), (False, True), (True, False)])
def test_streaming_exact_fields_and_SM_cycles(tmp_path, warps, encoded, recovery):
    a, header = fixture(tmp_path, warps=warps, encoded=encoded, recovery=recovery)
    assert next(frames(tmp_path / "arena-0.nts"))["rows"].tobytes() == a.tobytes()
    panel = summarize(tmp_path)
    assert panel["counts"]["observed_records"] == len(a)
    assert panel["clock_domain"] == "clock64_per_SM"
    assert panel["decoded_artifacts"][0]["decoded_sha256"] == hashlib.sha256(header + a.tobytes()).hexdigest()
    assert sum(x["summed_elapsed_cycles"] for x in panel["sm_distribution"]) == int(a["elapsed"].sum())
    assert panel["launches"][0]["elapsed_cycles_max"] == int(a["elapsed"].max())


@pytest.mark.parametrize("cut", [1, 7, 12, 49, 65])
def test_partial_records_fail_instead_of_reporting_complete(tmp_path, cut):
    fixture(tmp_path, encoded=True)
    path = tmp_path / "arena-0.nts"
    path.write_bytes(path.read_bytes()[:-cut])
    with pytest.raises(ValueError, match="ERE_TRUNCATED"):
        summarize(tmp_path)


def test_lineage_budget_flags_and_runtime_identity(tmp_path):
    fixture(tmp_path, encoded=True)
    with pytest.raises(ValueError, match="ERE_FRAME_COUNT_BUDGET"):
        summarize(tmp_path, max_frames=0)
    (tmp_path / "coverage.json").write_text('{"captured_launches":2}')
    with pytest.raises(ValueError, match="ERE_COVERAGE_FRAME_COUNT"):
        summarize(tmp_path)
    path = tmp_path / "arena-0.nts"
    payload = bytearray(path.read_bytes()[64:])
    struct.pack_into("<I", payload, 0, 4)
    with pytest.raises(ValueError, match="UNACCEPTED"):
        decode_reference(payload, 12, 4)
    assert not discover(None)["installed"]
    (tmp_path / "binary").write_bytes(b"x")
    (tmp_path / "hqsb-runtime.json").write_text(json.dumps({"schema_version": "hqsb-eretrace-1", "runtime_files": {"binary": "changed"}}))
    assert "CHANGED" in discover(tmp_path)["reason"]


def test_PTX_requires_exact_binary_pair(tmp_path, monkeypatch):
    cache = tmp_path / "cache/hash"
    cache.mkdir(parents=True)
    image = tmp_path / "original.bin"
    image.write_bytes(b"\x7fELF123")
    (cache / "kernel.cubin").write_bytes(b"\x7fELF456")
    (cache / "kernel.ptx").write_text(".version 8.7")
    monkeypatch.setenv("HQSB_ERE_TRITON_CACHE", str(cache.parent))
    assert cached_ptx(image) is None
    (cache / "kernel.cubin").write_bytes(image.read_bytes())
    assert cached_ptx(image) == cache / "kernel.ptx"
    assert json.loads((tmp_path / "ptx-origin.json").read_text())["cubin_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()


@pytest.mark.parametrize("options,workload", [({"enforce_eager": False}, {}), ({"enforce_eager": True, "tensor_parallel_size": 2}, {}), ({"enforce_eager": True}, {"concurrency": 2})])
def test_scope_guard_preserves_upstream_configuration(options, workload):
    original = deepcopy(options)
    with pytest.raises(ValueError):
        validate_scope(options, workload)
    assert options == original


def test_plan_keeps_selection_narrow_and_typed():
    plan = DiagnosticPlan(source_run_id="run_" + "a" * 32, tool="eretrace", ere_codec="raw", ere_verify=True)
    assert plan.ere_codec == "raw"
    with pytest.raises(ValueError):
        DiagnosticPlan(source_run_id=plan.source_run_id, tool="eretrace", ere_kernel="a:b")


@pytest.mark.parametrize("reason", ["probe_failed_launches", "other_thread_launches"])
def test_partial_coverage_keeps_records_and_exposes_missing_scope(tmp_path, reason):
    fixture(tmp_path, encoded=True)
    (tmp_path / "coverage.json").write_text(json.dumps({"captured_launches": 1, reason: 1}))
    panel = summarize(tmp_path)
    assert panel["availability"] == "partial"
    assert panel["counts"]["observed_records"] == 12
    assert panel["decoded_artifacts"][0]["decoded_sha256"]


def test_launch_environment_is_isolated_and_uses_original_probe(tmp_path):
    from hqsb.observation.eretrace_tools import launch_environment
    driver = tmp_path / "driver"
    driver.write_bytes(b"driver")
    (tmp_path / "build").mkdir()
    (tmp_path / "build/block_sched.toml").write_text("original compiled probe")
    manifest = {"driver_library": str(driver), "driver_sha256": hashlib.sha256(b"driver").hexdigest(), "cuda_home": "/cuda"}
    (tmp_path / "hqsb-runtime.json").write_text(json.dumps(manifest))
    plan = DiagnosticPlan(source_run_id="run_" + "a" * 32, tool="eretrace", ere_codec="raw").model_dump()
    env = launch_environment(tmp_path, tmp_path / "capture", plan, tmp_path / "cache")
    assert env["NEUTRINO_ABLATION_MODE"] == "7"
    assert "NEUTRINO_DYNAMIC" not in env
    assert env["NEUTRINO_PROBES"] == "original compiled probe"
    assert env["HQSB_ERE_CODEC_RAW"] == "1"
    assert env["NEUTRINO_ARENA_CODEC_MIN_KIB"] == "16"
    assert not env["NEUTRINO_BENCHMARK"] == "1"
    assert str(tmp_path / "neutrino/build/preload.so") == env["LD_PRELOAD"]


def test_ERE_comparison_matches_signature_and_checks_probe_scope(tmp_path):
    from hqsb.observation.diagnostic_comparison import compare
    fixture(tmp_path, encoded=True)
    panel = summarize(tmp_path)
    a = {"run_id": "a", "tool": "eretrace", "source_workload_sha256": "w", "capture_window": "steady",
         "replay_mode": "single_pass", "cache_control": "none", "clock_control": "none",
         "plan": {"ere_kernel": "rms", "launch_count": 1, "ere_codec": "reference"},
         "status": "completed", "cleanup": "succeeded", "replay_workloads_match_source": True,
         "replay_environment_matches_source": True, "replay_outputs_match_source": True,
         "source_engine": {"versions": {}, "model_manifest_sha256": "m", "hardware_inventory": {}, "requested": {}},
         "panel": panel}
    b = deepcopy(a)
    b["run_id"] = "b"
    b["plan"]["ere_codec"] = "raw"
    b["panel"]["launches"][0]["elapsed_cycles_p95"] += 1
    result = compare(a, b)
    assert result["comparable"]
    assert next(x for x in result["metric_changes"] if x["metric"] == "elapsed_cycles_p95")["delta"] == 1
    b["plan"]["ere_kernel"] = "other"
    assert "selection:ere_kernel" in compare(a, b)["mismatches"]


def test_sglang_worker_profile_adapter_drains_on_the_original_stop(monkeypatch):
    from types import SimpleNamespace
    from hqsb.observation import sglang_profile_server as adapter
    events = []

    class Mixin:
        profiler_activities = ["CUDA_PROFILER"]
        profile_in_progress = False

        def start_profile(self, stage=None):
            events.append(("original_start", stage))
            self.profile_in_progress = True
            return SimpleNamespace(success=True)

        def stop_profile(self, stage=None):
            events.append(("original_stop", stage))
            self.profile_in_progress = False
            return SimpleNamespace(success=True)

    monkeypatch.setattr(adapter, "control", lambda start: events.append(("ere_control", start)))
    module = SimpleNamespace(SchedulerProfilerMixin=Mixin)
    adapter.adapt(module)
    adapter.adapt(module)
    obj = Mixin()
    obj.start_profile("prefill")
    obj.stop_profile("decode")
    assert events == [("original_start", "prefill"), ("ere_control", True), ("ere_control", False), ("original_stop", "decode")]
    obj.profiler_activities = ["CPU"]
    events.clear()
    obj.start_profile()
    obj.stop_profile()
    assert all(event[0] != "ere_control" for event in events)
