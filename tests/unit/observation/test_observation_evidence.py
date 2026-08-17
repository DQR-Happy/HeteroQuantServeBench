import json

import pytest

from hqsb.observation.contracts import ObservationPlan, metric
from hqsb.observation.storage import EvidenceReader, EvidenceWriter, publish, sha256
from hqsb.observation.analysis import analyze, compare


def test_plan_rejects_unbounded_or_unknown_inputs():
    with pytest.raises(ValueError):
        ObservationPlan(mode="gpu_physical")
    with pytest.raises(ValueError):
        ObservationPlan(max_pages=4097)
    with pytest.raises(ValueError):
        ObservationPlan(max_records=True)
    assert (
        metric("missing", None, "byte", source="test", scope="test")["availability"]
        == "not_collected"
    )
    assert (
        metric("zero", 0, "byte", source="test", scope="test")["availability"]
        == "observed"
    )


def writer(tmp_path, name="run"):
    return EvidenceWriter(tmp_path / name, {"run_id": name}, chunk_bytes=150)


def test_supervisor_publication_and_hash_checked_roundtrip(tmp_path):
    capture = writer(tmp_path)
    for i in range(12):
        assert capture.offer("test", {"index": i}) == "accepted"
    report = capture.close(execution_status="completed", cleanup_status="succeeded")
    assert report["counts"]["written"] == 12
    assert not (tmp_path / "run/observations/manifest.json").exists()
    with pytest.raises(ValueError, match="RUN_MISMATCH"):
        publish(tmp_path / "run", expected_run_id="other")
    publish(tmp_path / "run", expected_run_id="run")
    assert [
        r["payload"]["index"] for r in EvidenceReader(tmp_path / "run").records()
    ] == list(range(12))
    assert len(report["artifacts"]) > 1
    chunk = tmp_path / "run" / report["artifacts"][0]["path"]
    chunk.write_text(chunk.read_text() + " ")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        list(EvidenceReader(tmp_path / "run").records())


def test_record_and_byte_budgets_never_claim_complete(tmp_path):
    capture = EvidenceWriter(tmp_path / "bounded", {}, max_records=1, max_bytes=100)
    capture.offer("oversize", {"raw": "x" * 200})
    assert capture.offer("extra", {}) == "budget_exceeded"
    report = capture.close(execution_status="failed", cleanup_status="unknown")
    assert report["capture_status"] == "partial"
    assert report["counts"]["written"] == 0
    assert report["counts"]["accepted"] == 1
    assert report["counts"]["dropped"] == 1


def test_source_failure_and_partial_roundtrip(tmp_path):
    capture = writer(tmp_path)
    capture.offer("source_status", {"reason": "permission_denied"})
    first = capture.close(
        execution_status="completed",
        cleanup_status="succeeded",
        source_problems=["permission"],
    )
    assert (
        capture.close(execution_status="completed", cleanup_status="succeeded") == first
    )
    assert first["capture_status"] == "partial"
    publish(tmp_path / "run")
    assert (
        analyze(tmp_path / "run")["source_problems"][0]["reason"] == "permission_denied"
    )


def test_manifest_cannot_escape_evidence_root(tmp_path):
    capture = writer(tmp_path)
    capture.offer("test", {})
    capture.close(execution_status="completed", cleanup_status="succeeded")
    manifest = tmp_path / "run/observations/capture-manifest.json"
    data = json.loads(manifest.read_text())
    data["artifacts"][0]["path"] = "../../elsewhere"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="PATH_ESCAPE"):
        publish(tmp_path / "run")


def test_supervisor_failure_retains_worker_scope_and_forces_partial(tmp_path):
    capture = writer(tmp_path)
    capture.offer("boundary", {"label": "before_request"})
    capture.close(execution_status="failed", cleanup_status="succeeded")
    sealed = publish(tmp_path / "run", supervisor_failure="worker failed")
    assert sealed["capture_status"] == "partial"
    assert (
        sealed["execution_status"] == "failed"
        and sealed["cleanup_status"] == "succeeded"
    )
    assert sealed["counts"]["written"] == 1


def test_framework_trace_is_bound_to_the_sealed_stream(tmp_path):
    capture = writer(tmp_path)
    trace = tmp_path / "run/trace.json"
    trace.write_text('{"traceEvents":[]}')
    capture.offer("framework_profile", {"trace_sha256": sha256(trace)})
    capture.close(execution_status="completed", cleanup_status="succeeded")
    publish(tmp_path / "run")
    analyze(tmp_path / "run")
    trace.write_text('{"traceEvents":[{"changed":true}]}')
    with pytest.raises(ValueError, match="TRACE_HASH_MISMATCH"):
        analyze(tmp_path / "run")


def test_differences_preserve_missing_and_do_not_adopt():
    context = {
        "source": {"source_digest": "same"},
        "model_identity": "weights",
        "dtype": "fp16",
        "cleanup_policy": "same",
        "workload_hash": "same",
        "plan": {"mode": "basic"},
    }
    a = {
        "context": context,
        "capture_status": "complete",
        "metrics": [
            metric("allocated", 0, "byte", source="allocator", scope="allocator"),
            metric("pss", None, "byte", source="proc", scope="cpu"),
        ],
    }
    b = {
        **a,
        "metrics": [
            metric("allocated", 4, "byte", source="allocator", scope="allocator")
        ],
    }
    result = compare(a, b)
    assert result["comparable"]
    assert result["decision"] == "hold"
    assert result["differences"][0]["delta"] == 4
    assert result["differences"][1]["delta"] is None
    b = {**b, "context": {**context, "plan": {"mode": "memory"}}}
    assert "OBSERVATION_PLAN_MISMATCH" in compare(a, b)["reasons"]
    changed = {**a, "context": {**context, "cleanup_policy": "preserve_allocator"}}
    assert "CONTROL_MISMATCH:cleanup_policy" in compare(a, changed)["reasons"]
    assert compare(a, changed, intervention="cleanup_policy")["comparable"]
    changed["context"]["dtype"] = "int4"
    assert (
        "CONTROL_MISMATCH:dtype"
        in compare(a, changed, intervention="cleanup_policy")["reasons"]
    )


def test_periodic_sample_finishing_late_cannot_replace_cleanup_boundary(tmp_path):
    def sample(label, minor, pss):
        observed = lambda values: {"availability": "observed", "values": values}
        return {
            "label": label,
            "faults": {"minor": minor, "major": 0},
            "status": observed({"VmRSS": 20, "VmSwap": 0}),
            "smaps_rollup": observed({"Pss": pss})
            if pss is not None
            else {"availability": "not_collected", "values": {}},
            "system_memory": observed({"MemAvailable": 100}),
            "system_vmstat": observed({"pgfault": minor}),
            "psi": observed({}),
        }

    capture = writer(tmp_path)
    capture.offer("os_sample", sample("before_request", 5, 10))
    capture.offer("os_sample", sample("after_cleanup", 8, 30))
    capture.offer("os_sample", sample(None, 7, None))  # Earlier read, later enqueue.
    capture.close(execution_status="completed", cleanup_status="succeeded")
    publish(tmp_path / "run")
    result = analyze(tmp_path / "run")
    assert (
        next(
            m["value"]
            for m in result["metrics"]
            if m["metric_id"] == "Pss" and m["label"] == "after"
        )
        == 30
    )
    assert (
        next(
            m["value"]
            for m in result["metrics"]
            if m["metric_id"] == "cpu_minor_fault_delta"
        )
        == 3
    )
    assert result["os_source_availability"]["smaps_rollup"] == "observed"
