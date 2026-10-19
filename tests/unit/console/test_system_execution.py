"""L1 admission, output comparison and rollback lifecycle invariants."""

import json
from pathlib import Path
import pytest
from hqsb.core.system_execution import SystemPlan, summarize, validate_system
from hqsb.hardware.system import cuda_uuid, Nvidia


@pytest.mark.parametrize(
    "patch",
    [
        {"candidate": {"cpu_ids": [0, 0]}},
        {"rounds": 0},
        {"prompts": [""]},
        {"candidate": {"intra_threads": 0}},
        {"candidate": {"shell": "command"}},
        {"max_output_tokens": 129},
        {"candidate": {"cpu_ids": [-1]}},
    ],
)
def test_invalid_contract(patch):
    with pytest.raises(ValueError):
        SystemPlan.model_validate(patch)


def test_hardware_admission_is_explicit():
    hardware = {
        "devices": [{"power_constraints_w": [100, 350]}],
        "allowed_cpus": [0, 1],
        "numa_policy": {"available": False, "reason": "seccomp"},
        "numa_nodes": [],
    }
    for p, reason in [
        ({"device": 1}, "GPU_NOT_VISIBLE"),
        ({"candidate": {"cpu_ids": [2]}}, "OUTSIDE_EFFECTIVE"),
        ({"candidate": {"numa_node": 0}}, "NUMA_UNAVAILABLE"),
        ({"candidate": {"power_limit_w": 400}}, "OUTSIDE_HARDWARE"),
    ]:
        with pytest.raises(ValueError, match=reason):
            validate_system(SystemPlan.model_validate(p), hardware)
    validate_system(SystemPlan(), hardware)
    assert cuda_uuid("123") == cuda_uuid("GPU-123") == "GPU-123"


def test_quality_failure_does_not_hide_performance():
    def trial(label, rate, tokens, energy):
        return {
            "label": label,
            "requests": [{"token_ids": tokens}],
            "metrics": {"token_s": rate, "energy_j": energy},
        }

    d = summarize([trial("baseline", 10, [1], None), trial("candidate", 20, [2], 3)])
    assert not d["greedy_tokens_exact"]
    assert d["metrics"]["token_s"]["delta_percent"] == 100
    assert d["metrics"]["energy_j"]["delta"] is None
    assert summarize([])["repetitions_per_profile"] == [0, 0]


def test_no_permission_restore_can_confirm_unchanged_original(monkeypatch):
    n = object.__new__(Nvidia)
    monkeypatch.setattr(n, "power", lambda _: 350)
    # Equal readback does not need a privileged setter or pretend to change it.
    assert n.set_power("GPU-test", 350) == 350


def test_restart_journal_blocks_new_jobs_before_reserving_device(tmp_path, monkeypatch):
    from hqsb.console.config import Settings
    from hqsb.console.service import ConsoleService, ServiceError
    from hqsb.console.system_jobs import SystemJobs

    settings = Settings(
        data_dir=tmp_path / "console",
        evidence_root=tmp_path,
        web_dist=tmp_path / "web",
        deployments=[],
    )
    svc = ConsoleService(settings)
    jobs = SystemJobs(settings, svc)
    jobs.journal.write_text(
        '{"state":"restore_required","uuid":"GPU-old","original_power_w":350}'
    )
    monkeypatch.setattr(
        jobs,
        "hardware",
        lambda: {
            "devices": [{}],
            "allowed_cpus": [],
            "recovery_reason": "restore the recorded cap",
        },
    )
    try:
        with pytest.raises(ServiceError, match="restore the recorded cap"):
            jobs.start("tiny", SystemPlan(), "new")
        assert svc.pending == 0 and svc.benchmark_owner is None
        assert svc.store.list(10)["total"] == 0
        assert jobs.journal.exists()
    finally:
        jobs.close()
        svc.close()


def test_retry_race_reserves_once_and_conflicts_before_probe(tmp_path, monkeypatch):
    import struct
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from hqsb.console.config import Settings, DeploymentConfig
    from hqsb.console.service import ConsoleService, ServiceError
    from hqsb.console.system_jobs import SystemJobs

    class Actor:
        def __init__(self, _):
            pass

        def close(self):
            pass

    root = tmp_path / "model"
    root.mkdir()
    header = json.dumps({"w": {"shape": [1]}}).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    cfg = DeploymentConfig(
        id="tiny",
        name="tiny",
        model_path=str(root),
        manifest=str(tmp_path / "manifest"),
    )
    settings = Settings(
        data_dir=tmp_path / "console",
        evidence_root=tmp_path,
        web_dist=tmp_path / "web",
        deployments=[cfg],
    )
    svc = ConsoleService(settings, actor_factory=Actor)
    jobs = SystemJobs(settings, svc)
    jobs.executor.shutdown()

    class Queue:
        submitted = 0

        def submit(self, *args):
            self.submitted += 1

        def shutdown(self, **kwargs):
            pass

    jobs.executor = Queue()
    barrier = threading.Barrier(2)

    def hardware():
        barrier.wait(timeout=5)
        return {"devices": [{}], "allowed_cpus": [0], "recovery_reason": "recover"}

    monkeypatch.setattr(jobs, "hardware", hardware)
    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(jobs.start, "tiny", SystemPlan(), "retry") for _ in range(2)
            ]
            rows = [f.result(timeout=10) for f in futures]
        assert rows[0]["id"] == rows[1]["id"] and svc.pending == 1
        assert jobs.executor.submitted == 1
        monkeypatch.setattr(
            jobs, "hardware", lambda: pytest.fail("retry must not reprobe")
        )
        assert jobs.start("tiny", SystemPlan(), "retry")["id"] == rows[0]["id"]
        with pytest.raises(ServiceError, match="Key already used"):
            jobs.start("tiny", SystemPlan(rounds=1), "retry")
    finally:
        svc.pending = 0
        jobs.close()
        svc.close()


@pytest.mark.parametrize("failure", ["progress", "guardian_hang", "launch"])
def test_supervisor_failure_cancels_guard_before_publication(
    tmp_path, monkeypatch, failure
):
    import subprocess
    from hqsb.console.config import Settings
    from hqsb.console.service import ConsoleService
    from hqsb.console import system_jobs as module
    from hqsb.observation.storage import atomic_json

    settings = Settings(
        data_dir=tmp_path / "console",
        evidence_root=tmp_path,
        web_dist=tmp_path / "web",
        deployments=[],
    )
    svc = ConsoleService(settings)
    jobs = module.SystemJobs(settings, svc)
    run, _ = svc.store.create("system", {}, "test", "digest")
    svc.pending, svc.benchmark_owner = 1, "l1:" + run["id"]
    root = settings.data_dir / "captures" / run["id"]

    class Guardian:
        def __init__(self, args, **kwargs):
            if failure == "launch":
                raise OSError("launch denied")
            atomic_json(root / "progress.json", {})

        def poll(self):
            return None

        def wait(self, **kwargs):
            assert (root / "cancel").exists(), "must notify guardian before waiting"
            if failure == "guardian_hang":
                raise subprocess.TimeoutExpired("guardian", 20)
            atomic_json(
                root / "guard-result.json",
                {"status": "cancelled", "cleanup": "succeeded", "trials": []},
            )

    reader = module.read_json

    def read(path):
        if path.name == "progress.json":
            raise ValueError("progress corrupt")
        return reader(path)

    monkeypatch.setattr(module.subprocess, "Popen", Guardian)
    monkeypatch.setattr(module, "read_json", read)
    try:
        jobs.execute(run["id"], {}, SystemPlan(), {"devices": [{"uuid": "GPU-test"}]})
        r = svc.store.get(run["id"])
        assert (
            r["state"] == "failed" and svc.pending == 0 and svc.benchmark_owner is None
        )
        assert r["metrics"]["experiment_bundle"]["status"] == "partial"
        if failure == "guardian_hang":
            assert jobs.quarantined and r["cleanup"] != "succeeded"
        else:
            assert r["cleanup"] == "succeeded" and not jobs.quarantined
        assert ("launch denied" if failure == "launch" else "progress corrupt") in r[
            "error"
        ]
    finally:
        jobs.close()
        svc.close()


@pytest.mark.parametrize("failure", [None, "worker", "parent", "restore"])
def test_guard_restores_before_publishing_and_keeps_failed_journal(
    tmp_path, monkeypatch, failure
):
    from hqsb.backends import system_guard as g

    journal = tmp_path / "journal.json"
    original = 350
    events = []

    class Board:
        def __init__(self):
            self.cap = original

        def power(self, _):
            return self.cap

        def processes(self, _):
            return []

        def set_power(self, _, watts):
            assert journal.exists(), "write-ahead journal required before any mutation"
            events.append(watts)
            if failure == "restore" and len(events) > 1 and watts == original:
                raise RuntimeError("lost device during rollback")
            self.cap = watts
            return watts

        def close(self):
            pass

    class Worker:
        returncode = 0

        def __init__(self, args, **kwargs):
            i = int(args[-1])
            job = json.loads(Path(args[-2]).read_text())
            (tmp_path / f"trial-{i}.json").write_text(
                json.dumps(
                    {
                        "run_id": "test",
                        "status": "completed",
                        "trial": i,
                        "label": job["label"],
                    }
                )
            )
            (tmp_path / f"trial-{i}.log").write_text("worker failed")
            if failure == "worker":
                self.returncode = 1

        def poll(self):
            return self.returncode

    monkeypatch.setattr(g, "Nvidia", Board)
    monkeypatch.setattr(g.subprocess, "Popen", Worker)
    monkeypatch.setattr(
        g, "terminate", lambda p: events.append("terminate") if p else None
    )
    monkeypatch.setattr(
        g,
        "parent_identity",
        lambda _: "one" if failure != "parent" or not events else "gone",
    )
    plan = SystemPlan(
        rounds=1, baseline={"power_limit_w": 250}, candidate={"power_limit_w": 200}
    )
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            {
                "plan": plan.model_dump(),
                "run_id": "test",
                "uuid": "GPU-test",
                "journal": str(journal),
                "supervisor_pid": 123,
                "supervisor_start_ticks": "one",
                "config": {"device_lock_path": str(tmp_path / "device.lock")},
            }
        )
    )
    g.main(str(source))
    result = json.loads((tmp_path / "guard-result.json").read_text())
    if failure == "restore":
        assert result["cleanup"] == "failed" and journal.exists()
    else:
        assert result["cleanup"] == "succeeded" and not journal.exists()
        assert events[-1] == original
        if failure == "worker":
            assert result["status"] == "failed" and "worker failed" in result["error"]
        elif failure == "parent":
            assert result["status"] == "cancelled"
        else:
            assert [t["label"] for t in result["trials"]] == ["baseline", "candidate"]
            assert result["status"] == "completed"
