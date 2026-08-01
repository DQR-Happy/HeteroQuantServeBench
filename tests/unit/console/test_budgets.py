"""Regression checks for bounded list polling and queued artifact admission."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from hqsb.console.config import DeploymentConfig, Settings
from hqsb.console.service import ConsoleService, ServiceError
from hqsb.console.store import Store


def test_run_list_projects_details_without_changing_original(tmp_path):
    store = Store(tmp_path)
    run, _ = store.create("quantize", {"messages": [{"content": "private"}]}, "a", "b")
    store.update(
        run["id"],
        {
            "result": {"manifest": {"large": [1] * 1000}},
            "metrics": {
                "observation": {"operators": [1] * 1000},
                "kv_inventory": {"entries": [1] * 1000},
                "runtime_first_token_ms": 1.5,
                "quantization": {
                    "bytes": {"total": 7},
                    "weight_error": {
                        "nrmse": 0.1,
                        "per_tensor_metrics": {"large": [1] * 1000},
                        "per_tensor_wall_time_s": {"large": 1},
                    },
                },
            },
        },
    )
    try:
        row = store.list(kind="quantize")["items"][0]
        assert "result" not in row and "messages" not in row["config"]
        assert (
            "observation" not in row["metrics"] and "kv_inventory" not in row["metrics"]
        )
        assert row["metrics"]["runtime_first_token_ms"] == 1.5
        assert row["metrics"]["quantization"]["weight_error"] == {"nrmse": 0.1}
        assert store.get(run["id"])["result"]["manifest"]["large"] == [1] * 1000
    finally:
        store.close()


def test_budget_recheck_does_not_destroy_ready_model(tmp_path, monkeypatch):
    class Actor:
        def __init__(self, config):
            self.closed = False
            self.executed = False

        def execute(self, *args):
            self.executed = True
            raise AssertionError("Rejected capture must never start")

        def close(self):
            self.closed = True

    settings = Settings(
        data_dir=tmp_path / "state",
        evidence_root=tmp_path,
        web_dist=tmp_path / "dist",
        deployments=[DeploymentConfig(id="gpu", name="GPU")],
    )
    service = ConsoleService(settings, actor_factory=Actor)
    service.states["gpu"].update(state="ready", epoch="epoch")
    occupied, release = Event(), Event()
    executor: ThreadPoolExecutor = service.executors["gpu"]

    def hold_queue():
        occupied.set()
        assert release.wait(5)

    executor.submit(hold_queue)
    assert occupied.wait(1)
    calls = []

    def preflight(*args):
        calls.append(args)
        if len(calls) > 1:
            raise ServiceError("ARTIFACT_BUDGET", "Changed while queued")

    monkeypatch.setattr(service, "_preflight", preflight)
    try:
        run = service.submit(
            "generate",
            "gpu",
            {
                "expected_epoch": "epoch",
                "max_output_tokens": 4,
                "observation_mode": "operators",
                "deadline_ms": 5000,
            },
            "capture",
        )
        release.set()
        for _ in range(100):
            row = service.store.get(run["id"])
            if row["state"] == "failed":
                break
            time.sleep(0.01)
        assert row["state"] == "failed" and row["cleanup"] == "not_started"
        assert row["error_code"] == "ARTIFACT_BUDGET"
        assert service.states["gpu"]["state"] == "ready"
        assert not service.actors["gpu"].closed and not service.actors["gpu"].executed
    finally:
        release.set()
        service.close()


def test_failed_cleanup_still_finishes_request_and_records_unknown(tmp_path):
    class BrokenActor:
        def __init__(self, config):
            pass

        def execute(self, *args):
            raise RuntimeError("device failed")

        def close(self):
            raise RuntimeError("unable to confirm worker exit")

    settings = Settings(
        data_dir=tmp_path / "state",
        evidence_root=tmp_path,
        web_dist=tmp_path / "dist",
        deployments=[DeploymentConfig(id="gpu", name="GPU")],
    )
    service = ConsoleService(settings, actor_factory=BrokenActor)
    try:
        run = service.submit("load", "gpu", {}, "load")
        for _ in range(100):
            row = service.store.get(run["id"])
            if row["state"] == "failed":
                break
            time.sleep(0.01)
        assert row["state"] == "failed" and row["cleanup"] == "unknown"
        assert (
            "device failed" in row["error"]
            and "cleanup could not be confirmed" in row["error"]
        )
        assert service.states["gpu"]["state"] == "failed"
    finally:
        service.actors["gpu"].close = lambda: None
        service.close()
