"""Console control-plane tests. TestActor is a test double, never a demo backend."""

import hashlib
import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from hqsb.console.app import create_app
from hqsb.console.config import DeploymentConfig, Settings
from hqsb.console.evidence import EvidenceCatalog
from hqsb.console.service import ConsoleService, ServiceError
from hqsb.console.store import Store


class TestActor:
    __test__ = False

    def __init__(self, config):
        self.closed = False
        self.config = config

    def execute(self, kind, payload, on_output, is_cancelled, timeout_s):
        if kind != "generate":
            return {"weight_verification": "test_only"}
        for n in range(30):
            if is_cancelled():
                return {"finish_reason": "cancelled", "text": "测试", "metrics": {}}
            on_output(
                {
                    "text": "测试" + str(n),
                    "output_tokens": n + 1,
                    "runtime_first_token_ms": 1.2,
                }
            )
            time.sleep(0.005)
        return {
            "finish_reason": "length",
            "text": "测试完成",
            "metrics": {"output_tokens": 30, "measurement_profile": "test_only"},
        }

    def close(self):
        self.closed = True


@pytest.fixture
def settings(tmp_path):
    raw = tmp_path / "docs/stage_experiments/S05/E05-02/raw"
    raw.mkdir(parents=True)
    (raw / "verdict.json").write_text(json.dumps({"overall": "BLOCKED"}))
    (raw / "summary.json").write_text(
        json.dumps({"performance": {"fp16": {"tiny": {"ttft_ms": {"median": 12}}}}})
    )
    return Settings(
        data_dir=tmp_path / "state",
        evidence_root=tmp_path,
        web_dist=tmp_path / "dist",
        deployments=[DeploymentConfig(id="reference", name="test")],
    )


def wait(service, run):
    for _ in range(500):
        row = service.store.get(run["id"])
        if row["state"] in {
            "completed",
            "cancelled",
            "failed",
            "timed_out",
            "interrupted",
        }:
            return row
        time.sleep(0.01)
    raise AssertionError("Test operation did not finish")


@pytest.fixture
def service(settings):
    svc = ConsoleService(settings, actor_factory=TestActor)
    yield svc
    svc.close()


def load(service):
    assert (
        wait(service, service.submit("load", "reference", {}, uuid.uuid4().hex))[
            "state"
        ]
        == "completed"
    )
    return service.deployments()[0]["epoch"]


def payload(epoch):
    return {
        "expected_epoch": epoch,
        "messages": [{"role": "user", "content": "private input"}],
        "max_output_tokens": 32,
        "deadline_ms": 5000,
        "save_input": False,
    }


def test_store_idempotency_sequence_and_restart(tmp_path):
    store = Store(tmp_path)
    run, new = store.create("generate", {}, "same", "digest")
    assert new
    again, new = store.create("generate", {}, "same", "digest")
    assert not new and again["id"] == run["id"]
    with pytest.raises(ValueError):
        store.create("generate", {}, "same", "different")
    store.update(run["id"], {"output": "one"}, "output.snapshot")
    store.close()
    store = Store(tmp_path)
    store.recover()
    assert store.get(run["id"])["state"] == "interrupted"
    assert [e["seq"] for e in store.events(run["id"], 0)] == [1, 2, 3]
    assert len(store.events(run["id"], 2)) == 1
    store.close()


def test_lifecycle_input_privacy_and_stale_epoch(service):
    with pytest.raises(ServiceError, match="Load and warm"):
        service.submit("generate", "reference", payload("stale"), "before-load")
    epoch = load(service)
    run = service.submit("generate", "reference", payload(epoch), "generation")
    assert "messages" not in run["config"] and len(run["config"]["input_sha256"]) == 64
    assert wait(service, run)["state"] == "completed"
    assert (
        wait(service, service.submit("unload", "reference", {}, "unload"))["state"]
        == "completed"
    )
    assert service.deployments()[0]["detail"] == {}
    reloading = service.submit("load", "reference", {}, "reload")
    assert reloading["config"]["actual_runtime"] == {}
    assert wait(service, reloading)["state"] == "completed"
    assert service.deployments()[0]["epoch"] != epoch
    with pytest.raises(ServiceError, match="Deployment changed"):
        service.submit("generate", "reference", payload(epoch), "stale")


def test_cancel_is_idempotent_and_completion_after_cleanup(service):
    epoch = load(service)
    run = service.submit("generate", "reference", payload(epoch), "cancel")
    service.cancel(run["id"])
    row = wait(service, run)
    assert row["state"] == "cancelled"
    assert row["cleanup"] in {"succeeded", "not_started"}
    assert service.cancel(run["id"])["state"] == "cancelled"
    assert service.deployments()[0]["state"] == "ready"


def test_deduplicate_load_and_reject_conflict(service):
    run = service.submit("load", "reference", {}, "one")
    wait(service, run)
    assert service.submit("load", "reference", {}, "one")["id"] == run["id"]
    with pytest.raises(ServiceError, match="Key already"):
        service.submit("unload", "reference", {}, "one")


def test_queue_backpressure_and_input_retention(service):
    epoch = load(service)
    service.settings.max_pending = 1
    data = {**payload(epoch), "save_input": True}
    run = service.submit("generate", "reference", data, "one")
    assert run["config"]["messages"] == data["messages"]
    with pytest.raises(ServiceError) as caught:
        service.submit("generate", "reference", data, "two")
    assert caught.value.status == 429
    wait(service, run)


def test_worker_failure_invalidates_deployment_and_preserves_error(service):
    load(service)

    def fail(*args):
        raise RuntimeError("controlled worker fault")

    service.actors["reference"].execute = fail
    run = service.submit(
        "generate", "reference", payload(service.deployments()[0]["epoch"]), "fault"
    )
    result = wait(service, run)
    assert result["state"] == "failed" and result["cleanup"] == "worker_terminated"
    assert service.deployments()[0]["epoch"] is None


def test_config_rejects_fake_quant_and_hides_server_secrets(settings):
    with pytest.raises(ValueError):
        DeploymentConfig(id="w4", name="false", precision="w4")
    with pytest.raises(ValueError):
        DeploymentConfig(id="remote", name="missing URL", provider="openai")
    cfg = DeploymentConfig(
        id="remote",
        name="remote",
        provider="openai",
        base_url="http://private/v1",
        api_key_env="SECRET",
    )
    assert not {"base_url", "api_key_env", "model_path"}.intersection(cfg.public())


def test_evidence_readonly_digest_and_symlink_guard(settings, tmp_path):
    catalog = EvidenceCatalog(settings.evidence_root)
    item = catalog.scan()[0]
    assert item["status"] == "BLOCKED"
    path, data, digest = catalog.file(item["id"])
    assert digest == hashlib.sha256(data).hexdigest()
    before = path.read_bytes()
    assert catalog.detail(item["id"])["content"]["overall"] == "BLOCKED"
    assert before == path.read_bytes()
    path.unlink()
    path.symlink_to("/etc/passwd")
    with pytest.raises(KeyError):
        catalog.file(item["id"])


@pytest.fixture
def client(settings):
    service = ConsoleService(settings, actor_factory=TestActor)
    app = create_app(
        settings, "test-token-12345678901234567890", service=service, monitor=False
    )
    with TestClient(app) as client:
        yield client, service


BASE = "/api/console/v1"
AUTH = {"Authorization": "Bearer test-token-12345678901234567890"}


def post(client, path, data=None, key=None):
    return client.post(
        BASE + path,
        json=data or {},
        headers={**AUTH, "Idempotency-Key": key or uuid.uuid4().hex},
    )


def test_auth_cookie_csrf_and_cross_origin(client):
    browser, _ = client
    assert browser.get(BASE + "/runs").status_code == 401
    assert (
        browser.post(BASE + "/session/login", json={"token": "wrong"}).status_code
        == 403
    )
    response = browser.post(
        BASE + "/session/login",
        json={"token": AUTH["Authorization"][7:]},
        headers={"X-HQSB-Client": "console"},
    )
    assert response.status_code == 200 and "HttpOnly" in response.headers["set-cookie"]
    assert browser.get(BASE + "/session").status_code == 200
    assert (
        browser.post(
            BASE + "/session/logout",
            json={},
            headers={"X-HQSB-Client": "console", "Origin": "http://evil"},
        ).status_code
        == 403
    )
    assert (
        browser.post(
            BASE + "/session/logout", json={}, headers={"X-HQSB-Client": "console"}
        ).status_code
        == 200
    )
    assert browser.get(BASE + "/runs").status_code == 401


def test_api_validation_and_lifecycle(client):
    browser, service = client
    assert post(browser, "/requests", {"deployment_id": "reference"}).status_code == 422
    run = post(browser, "/deployments/reference/load").json()
    wait(service, run)
    data = {"deployment_id": "reference", **payload(service.deployments()[0]["epoch"])}
    assert post(browser, "/requests", {**data, "precision": "w4"}).status_code == 422
    assert (
        post(browser, "/requests", {**data, "max_output_tokens": 513}).status_code
        == 422
    )
    assert (
        post(browser, "/requests", {**data, "deadline_ms": 500000}).status_code == 422
    )
    assert (
        post(
            browser,
            "/requests",
            {**data, "messages": [{"role": "system", "content": "only"}]},
        ).status_code
        == 422
    )
    run = post(browser, "/requests", data, "same").json()
    assert post(browser, "/requests", data, "same").json()["id"] == run["id"]
    assert (
        post(
            browser, "/requests", {**data, "max_output_tokens": 31}, "same"
        ).status_code
        == 409
    )
    wait(service, run)
    response = browser.get(BASE + f"/requests/{run['id']}/events?after=2", headers=AUTH)
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[0]["seq"] == 3 and events[-1]["kind"] == "completed"
    assert all(e["request_id"] == run["id"] for e in events)
    assert (
        browser.get(
            BASE + f"/requests/{run['id']}/events?after=10000", headers=AUTH
        ).status_code
        == 409
    )
    exported = browser.get(BASE + f"/runs/{run['id']}/export", headers=AUTH)
    assert exported.json()["output"] == "测试完成"
    assert "private input" not in exported.text
    assert browser.get(BASE + "/runs?limit=1000", headers=AUTH).status_code == 422


def test_comparability_and_historical_read(client):
    browser, service = client
    epoch = load(service)
    one = wait(service, service.submit("generate", "reference", payload(epoch), "one"))
    two = wait(service, service.submit("generate", "reference", payload(epoch), "two"))
    assert post(browser, "/comparisons", {"run_ids": [one["id"], two["id"]]}).json()[
        "comparable"
    ]
    three = wait(
        service,
        service.submit(
            "generate",
            "reference",
            {**payload(epoch), "max_output_tokens": 16},
            "three",
        ),
    )
    assert not post(
        browser, "/comparisons", {"run_ids": [one["id"], three["id"]]}
    ).json()["comparable"]
    result = browser.get(BASE + "/historical/performance", headers=AUTH).json()
    assert result["items"][0]["metrics"]["ttft_ms"] == 12
    item = browser.get(BASE + "/evidence", headers=AUTH).json()["items"][0]
    response = browser.get(BASE + "/evidence/" + item["id"] + "/download", headers=AUTH)
    assert (
        response.headers["X-Content-SHA256"]
        == hashlib.sha256(response.content).hexdigest()
    )


def test_api_import_keeps_heavy_runtime_lazy():
    import ast
    from pathlib import Path

    root = Path(__file__).parents[3]
    for path in (root / "hqsb/console").glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Import):
                assert all(
                    n.name.split(".")[0] not in {"torch", "triton", "transformers"}
                    for n in node.names
                )
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {
                    "torch",
                    "triton",
                    "transformers",
                }
