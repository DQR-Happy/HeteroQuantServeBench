"""Control/storage tests. Actor double supplies no numerical or GPU evidence."""
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from hqsb.console.app import create_app
from hqsb.console.config import Settings, DeploymentConfig
from hqsb.console.scheduling_campaigns import SchedulingCampaigns
from hqsb.console.service import ConsoleService, ServiceError
from hqsb.observation.storage import EvidenceWriter


class Actor:
    def __init__(self, _):
        self.release = threading.Event()
        self.started = threading.Event()
        self.messages = []

    def execute(self, kind, payload, on_output, is_cancelled, timeout_s):
        if getattr(self, "fail_capability", False) and kind == "scheduling_capabilities":
            raise RuntimeError("lost worker")
        if kind == "scheduling_capabilities":
            return {"supported": True, "reasons": []}
        if kind != "scheduling":
            return {}
        writer = EvidenceWriter(payload["capture_dir"], {"run_id": payload["capture_id"]})
        self.started.set()
        while not self.release.wait(.005):
            if is_cancelled():
                break
        writer.offer("forward", {"members": ["a", "b"], "input_shape": [2, 1], "source": "test_double"})
        writer.close(execution_status="completed", cleanup_status="succeeded")
        return {"finish_reason": "cancelled" if is_cancelled() else "completed", "cleanup": "succeeded",
                "quality": "not_evaluated", "variants": []}

    def control(self, message):
        self.messages.append(message)

    def close(self):
        self.release.set()


@pytest.fixture
def env(tmp_path):
    settings = Settings(data_dir=tmp_path / "state", evidence_root=tmp_path, web_dist=tmp_path / "web",
        deployments=[DeploymentConfig(id="reference", name="test")])
    svc = ConsoleService(settings, actor_factory=Actor)
    manager = SchedulingCampaigns(settings, svc)
    load = svc.submit("load", "reference", {}, uuid.uuid4().hex)
    wait(svc, load["id"])
    yield settings, svc, manager, svc.actors["reference"]
    manager.close()
    svc.close()


def wait(svc, identity):
    for _ in range(1000):
        row = svc.store.get(identity)
        if row["state"] in {"completed", "cancelled", "failed"} and svc.pending == 0:
            return row
        time.sleep(.005)
    raise AssertionError("not finished")


def body(svc):
    return {"deployment_id": "reference", "expected_epoch": svc.states["reference"]["epoch"],
            "requests": [{"id": "a", "prompt": "private text"}], "modes": ["continuous"], "save_input": False}


def test_exclusive_idempotent_controls_privacy_and_publication(env):
    _, svc, manager, actor = env
    plan = body(svc)
    row = manager.start(plan, "one")
    assert actor.started.wait(1)
    assert manager.start(plan, "one")["id"] == row["id"]
    assert "prompt" not in row["config"]["requests"][0]
    with pytest.raises(ServiceError, match="Key already"):
        manager.start({**plan, "repeats": 2}, "one")
    with pytest.raises(ServiceError):
        svc.submit("generate", "reference", {}, "blocked")
    assert manager.cancel_request(row["id"], "a")["state"] == "queued"
    assert actor.messages[0]["run_id"] == row["id"]
    manager.submit_request(row["id"], {"id": "b", "prompt": "more private text"})
    assert "prompt" not in manager.get(row["id"])["config"]["external_requests"][0]
    actor.release.set()
    final = wait(svc, row["id"])
    # Queued HTTP ingress has no worker acceptance in this double: must not PASS.
    assert final["state"] == "failed"
    assert final["result"]["unacknowledged_ingress"] == ["b"]
    assert svc.states["reference"]["state"] == "ready" and svc.benchmark_owner is None
    records = manager.records(row["id"], "forward", 0, 1)
    assert len(records["items"]) == 1 and records["capture_status"] == "complete"
    path = next((svc.settings.data_dir / "captures" / row["id"] / "observations").glob("records-*.jsonl"))
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="HASH"):
        manager.records(row["id"], "forward", 0, 1)


def test_api_lists_new_run_kind_and_cancellation_releases_owner(env):
    settings, svc, manager, actor = env
    app = create_app(settings, service=svc, monitor=False)
    with TestClient(app) as client:
        headers = {"X-HQSB-Client": "console", "Idempotency-Key": "api-one"}
        response = client.post("/api/console/v1/scheduling-benchmarks", json=body(svc), headers=headers)
        assert response.status_code == 202
        identity = response.json()["id"]
        assert actor.started.wait(1)
        listing = client.get("/api/console/v1/runs").json()
        assert any(r["kind"] == "scheduling" for r in listing["items"])
        assert all("requests" not in r["config"] for r in listing["items"] if r["kind"] == "scheduling")
        assert client.post(f"/api/console/v1/scheduling-benchmarks/{identity}/cancel", json={}, headers=headers).status_code == 200
        final = wait(svc, identity)
        assert final["state"] == "cancelled" and svc.states["reference"]["state"] == "ready"
    # app lifespan owns and closes svc; the fixture close is idempotent.


def test_capability_failure_invalidates_stale_ready_state(env):
    _, svc, manager, actor = env
    actor.fail_capability = True
    with pytest.raises(ServiceError, match="reload"):
        manager.capabilities("reference")
    assert svc.states["reference"]["state"] == "failed"
    assert svc.states["reference"]["epoch"] is None and svc.pending == 0
