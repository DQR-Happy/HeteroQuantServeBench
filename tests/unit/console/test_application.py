"""L10 contracts, semantic cache isolation, routing and real controller lifetimes."""

import time
import pytest
from hqsb.core.application_plan import (
    ApplicationInput,
    ApplicationPlan,
    ApplicationPolicy,
)
from hqsb.serving.application_cache import ResultCache, result_key
from hqsb.serving.application_engine import ApplicationEngine, compare_variants


@pytest.mark.parametrize(
    "patch",
    [
        {"targets": []},
        {"targets": [{"id": "x", "deployment_id": "m"}] * 2},
        {
            "requests": [
                {
                    "id": "r",
                    "prompt": "p",
                    "messages": [{"role": "user", "content": "q"}],
                }
            ]
        },
        {"candidate": {"routing": "length_model"}},
        {"candidate": {"min_replicas": 2, "max_replicas": 1}},
        {"candidate": {"input_transform": "recent_messages"}},
        {"concurrency": 9},
    ],
)
def test_contract(patch):
    with pytest.raises(ValueError):
        ApplicationPlan.model_validate(
            {"targets": [{"id": "a", "deployment_id": "m"}], **patch}
        )


def test_exact_cache_all_invalidation_dimensions(monkeypatch):
    policy = ApplicationPolicy(result_cache=True, cache_max_entries=1)
    item = ApplicationInput(id="one", prompt="same")
    snapshot = {
        "epoch": "one",
        "identity": {"weights": "verified", "tokenizer": "v1", "precision": "fp16"},
    }
    key = result_key(item, policy, snapshot, False)
    for changed in [
        item.model_copy(update={"tenant": "other"}),
        item.model_copy(update={"prompt": "different"}),
        item.model_copy(update={"max_output_tokens": 2}),
    ]:
        assert result_key(changed, policy, snapshot, False) != key
    for changed in [
        {**snapshot, "epoch": "new"},
        {**snapshot, "identity": {"weights": "new"}},
    ]:
        assert result_key(item, policy, changed, False) != key
    assert result_key(item, policy, snapshot, True) != key
    cache = ResultCache(policy)
    cache.put(key, {"token_ids": [1, 2]})
    value = cache.get(key)
    value["token_ids"].append(3)
    assert cache.get(key)["token_ids"] == [1, 2]
    cache.put("another", {"token_ids": [9]})
    assert cache.get(key) is None and cache.summary()["entries"] == 1
    with cache.lock:
        until, row, size = cache.entries["another"]
        cache.entries["another"] = (time.monotonic() - 1, row, size)
    assert cache.get("another") is None and cache.summary()["bytes"] == 0


class Endpoint:
    def __init__(self, ready=True, weights="w"):
        self.ready, self.weights = ready, weights
        self.epoch, self.calls, self.loads, self.unloads = "e0", 0, 0, 0

    def snapshot(self):
        return {
            "state": "ready" if self.ready else "unloaded",
            "epoch": self.epoch,
            "identity": {"weights": self.weights},
            "prefix_supported": True,
        }

    def prepare(self, item, policy):
        return {
            "payload": {"prompt": item["prompt"]},
            "effective_input_sha256": "input",
            "input_tokens_before": 2,
            "input_tokens_after": 2,
            "effective_output_tokens": min(
                item["max_output_tokens"], policy["output_cap"] or 999
            ),
        }

    def generate(self, payload, key, deadline, cancelled):
        self.calls += 1
        time.sleep(0.005)
        return {
            "id": f"child-{self.calls}",
            "output": "result",
            "metrics": {
                "output_tokens": 2,
                "generated_token_ids": [1, 2],
                "input_tokens": 2,
            },
        }

    def load(self, key):
        self.loads += 1
        self.ready, self.epoch = True, f"e{self.loads}"
        return {"id": f"load-{self.loads}", "epoch": self.epoch}

    def unload(self, epoch, key, load_run_id):
        assert self.ready and epoch == self.epoch
        self.ready = False
        self.unloads += 1

    def clear_prefix(self):
        pass

    def begin_prefix(self):
        pass

    def end_prefix(self):
        pass


def test_run_directory_uses_indexes_and_status_is_small(tmp_path):
    from hqsb.console.store import Store

    store = Store(tmp_path)
    try:
        r, _ = store.create("application", {"large": "x" * 10000}, "key", "hash")
        assert store.status(r["id"]) == {
            "id": r["id"],
            "state": "queued",
            "cleanup": "pending",
            "error": None,
        }
        explain = store.db.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM runs WHERE json_extract(body,'$.kind')=? ORDER BY created DESC LIMIT 60",
            ("application",),
        ).fetchall()
        assert any("runs_kind_created_idx" in row[-1] for row in explain)
        assert not any("TEMP B-TREE" in row[-1] for row in explain)
    finally:
        store.close()


def plan(**patch):
    return ApplicationPlan.model_validate(
        {
            "targets": [{"id": "a", "deployment_id": "m"}],
            "concurrency": 2,
            "requests": [
                {
                    "id": str(i),
                    "prompt": "same",
                    "tenant": "other" if i == 3 else "default",
                }
                for i in range(4)
            ],
            **patch,
        }
    )


def engine(p, endpoints, cancelled=lambda: False):
    return ApplicationEngine(p, endpoints, cancelled, lambda _: None)


def test_real_engine_cache_avoids_forwards_and_isolates_tenant():
    p, endpoint = plan(), Endpoint()
    e = engine(p, {"a": endpoint})
    a, b = e.run(p.baseline, "baseline"), e.run(p.candidate, "candidate")
    assert a["metrics"]["backend_invocations"] == 4
    assert b["metrics"]["backend_invocations"] == 2 and b["cache"]["hits"] == 2
    hit = next(r for r in b["requests"] if r["cache"] == "hit")
    assert not hit["backend_invoked"] and hit["backend_generation_ms"] is None
    assert next(r for r in b["requests"] if r["tenant"] == "other")["cache"] == "miss"
    assert compare_variants(a, b)["all_tokens_exact"]


def test_managed_autoscale_really_loads_and_unloads_without_touching_existing():
    p = plan(targets=[{"id": "a", "deployment_id": "m", "lifecycle": "managed"}])
    endpoint = Endpoint(ready=False)
    e = engine(p, {"a": endpoint})
    result = e.run(ApplicationPolicy(autoscale=True), "test")
    assert endpoint.loads == endpoint.unloads == 1 and not endpoint.ready
    assert any(r["kind"] == "replica.loaded" for r in e.events)
    assert result["metrics"]["completed"] == 4
    resident = Endpoint()
    e = engine(plan(), {"a": resident})
    e.run(ApplicationPolicy(autoscale=True), "existing")
    assert resident.unloads == resident.loads == 0 and resident.ready


def test_idle_scale_down_and_reactivation():
    p = plan(
        targets=[{"id": "a", "deployment_id": "m", "lifecycle": "managed"}],
        requests=[
            {"id": "a", "prompt": "p"},
            {"id": "b", "prompt": "p", "arrival_ms": 250},
        ],
    )
    endpoint = Endpoint(ready=False)
    e = engine(p, {"a": endpoint})
    e.run(ApplicationPolicy(autoscale=True, idle_unload_ms=100), "idle")
    assert endpoint.loads == endpoint.unloads == 2
    assert any(r.get("reason") == "idle_scale_down" for r in e.events)


def test_cold_replica_load_does_not_block_ready_node():
    import threading

    short_done = threading.Event()

    class Ready(Endpoint):
        def generate(self, *args):
            short_done.set()
            return super().generate(*args)

    class Cold(Endpoint):
        def load(self, key):
            assert short_done.wait(2), "cold load held the global routing lock"
            return super().load(key)

    p = plan(
        targets=[
            {"id": "small", "deployment_id": "m", "role": "small"},
            {
                "id": "large",
                "deployment_id": "n",
                "machine_id": "remote",
                "role": "large",
                "lifecycle": "managed",
            },
        ],
        requests=[
            {"id": "long", "prompt": "a long input"},
            {"id": "short", "prompt": "p"},
        ],
    )
    doc = engine(p, {"small": Ready(), "large": Cold(ready=False)}).run(
        ApplicationPolicy(
            routing="length_model", short_input_chars=2, autoscale=True, max_replicas=2
        ),
        "load",
    )
    assert doc["metrics"]["completed"] == 2


def test_remote_uncertain_post_reuses_same_key_and_quarantines(monkeypatch):
    import httpx
    from hqsb.console.application_transport import RemoteEndpoint
    from hqsb.core.application_plan import ApplicationTarget

    calls = []
    errors = [True, False]

    def handle(request):
        calls.append(
            (str(request.url), request.headers["idempotency-key"], request.content)
        )
        if errors.pop(0):
            raise httpx.ReadTimeout("uncertain", request=request)
        return httpx.Response(202, json={"id": "same-child"})

    endpoint = object.__new__(RemoteEndpoint)
    endpoint.client = httpx.Client(
        base_url="http://node/api/console/v1", transport=httpx.MockTransport(handle)
    )
    endpoint.owner, endpoint.target, endpoint.quarantined = (
        "l10:run_" + "1" * 32,
        ApplicationTarget(id="a", deployment_id="m"),
        False,
    )
    endpoint.record = lambda *a, **k: None
    try:
        assert endpoint.operation("generate", {}, "stable")["id"] == "same-child"
        assert calls[0] == calls[1] and not endpoint.quarantined
        errors[:] = [True, True]
        with pytest.raises(httpx.ReadTimeout):
            endpoint.operation("generate", {}, "failed")
        assert endpoint.quarantined
        with pytest.raises(RuntimeError, match="UNCONFIRMED"):
            endpoint.release()
    finally:
        endpoint.close()


def test_prefix_scope_restores_prior_store_without_clearing_it(monkeypatch):
    import types
    import torch
    from hqsb.backends.application_input import prefix_scope
    from hqsb.backends.prefix_store import PrefixStore

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    old = PrefixStore()
    old.retained_bytes = 123
    provider = types.SimpleNamespace(prefix_store=old)
    prefix_scope(provider, "owner", True)
    assert provider.prefix_store is not old
    prefix_scope(provider, "owner", True)
    with pytest.raises(ValueError, match="OWNERSHIP"):
        prefix_scope(provider, "other", False)
    prefix_scope(provider, "owner", False)
    assert provider.prefix_store is old and old.retained_bytes == 123
    assert not hasattr(provider, "_l10_prefix_scope")


def test_model_router_and_alias_free_endpoint_selection():
    p = plan(
        targets=[
            {"id": "a", "deployment_id": "m", "role": "small"},
            {"id": "b", "deployment_id": "n", "machine_id": "remote", "role": "large"},
        ],
        requests=[
            {"id": "short", "prompt": "p"},
            {"id": "long", "prompt": "long text"},
        ],
    )
    a, b = Endpoint(weights="small"), Endpoint(weights="large")
    result = engine(p, {"a": a, "b": b}).run(
        ApplicationPolicy(routing="length_model", short_input_chars=2), "models"
    )
    assert {r["id"]: r["target"] for r in result["requests"]} == {
        "short": "a",
        "long": "b",
    }
    with pytest.raises(ValueError, match="IDENTICAL_MODELS"):
        engine(p, {"a": a, "b": b}).run(ApplicationPolicy(), "wrong")


def test_cancel_preserves_observed_rows_and_never_returns_cache_as_compute():
    stop = [False]

    class CancelEndpoint(Endpoint):
        def generate(self, *args):
            stop[0] = True
            raise TimeoutError("cancelled")

    p = plan(concurrency=1)
    e = engine(p, {"a": CancelEndpoint()}, lambda: stop[0])
    result = e.run(p.candidate, "cancel")
    assert result["metrics"]["failed"] == 4
    assert result["requests"] and result["metrics"]["computed_output_tokens"] == 0
    assert not compare_variants(result, result)["all_tokens_exact"]


def test_prepare_uses_real_tokenizer_and_preserves_system_messages():
    from types import SimpleNamespace
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from hqsb.backends.application_input import prepare

    tok = Tokenizer(
        models.WordLevel(
            {"unk": 0, "word": 1, "system": 2, "user": 3, "assistant": 4},
            unk_token="unk",
        )
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="unk")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}assistant"
    p = SimpleNamespace(
        tokenizer=tokenizer,
        model=SimpleNamespace(
            config=SimpleNamespace(model_type="llama", max_position_embeddings=128)
        ),
        config={"context_limit": 128, "max_output_tokens": 64},
    )
    item = ApplicationInput(id="a", prompt="word " * 80, max_output_tokens=32)
    result = prepare(
        p,
        {
            "input": item.model_dump(),
            "policy": ApplicationPolicy(
                input_transform="head_tail_tokens", input_token_cap=16, output_cap=4
            ).model_dump(),
        },
    )
    assert result["input_tokens_before"] == 80 and result["input_tokens_after"] <= 16
    assert result["effective_output_tokens"] == 4 and result["lossy"]
    item = ApplicationInput(
        id="chat",
        messages=[
            {"role": "system", "content": "word"},
            {"role": "user", "content": "word"},
            {"role": "assistant", "content": "word"},
            {"role": "user", "content": "word word"},
        ],
    )
    result = prepare(
        p,
        {
            "input": item.model_dump(),
            "policy": ApplicationPolicy(
                input_transform="recent_messages", recent_messages=1
            ).model_dump(),
        },
    )
    assert [m["role"] for m in result["payload"]["messages"]] == ["system", "user"]


def test_plan_editor_rejects_invalid_shapes_before_runtime(tmp_path):
    from fastapi.testclient import TestClient
    from hqsb.console.app import create_app
    from hqsb.console.config import Settings

    settings = Settings(
        auth_required=False,
        data_dir=tmp_path / "console",
        evidence_root=tmp_path,
        web_dist=tmp_path / "web",
        deployments=[],
    )
    with TestClient(create_app(settings, monitor=False)) as client:
        headers = {"X-HQSB-Client": "console"}
        response = client.post(
            "/api/console/v1/application/validate",
            json={"plan": plan().model_dump()},
            headers=headers,
        )
        assert response.status_code == 200
        broken = plan().model_dump()
        broken["requests"] = [None]
        assert (
            client.post(
                "/api/console/v1/application/validate",
                json={"plan": broken},
                headers=headers,
            ).status_code
            == 422
        )
        assert client.get("/api/console/v1/overview").json()["pending"] == 0


def test_alias_rejection_releases_physical_node_once_and_seals_failure(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from hqsb.console.config import Settings
    from hqsb.console.service import ConsoleService
    from hqsb.console import application_jobs as module

    counters = {"acquire": 0, "release": 0, "close": 0}

    class Remote:
        node_id = "same-physical-node"
        quarantined = False

        def __init__(self, connections, profile, target, owner, record):
            self.target = target

        def acquire(self):
            counters["acquire"] += 1

        def release(self):
            counters["release"] += 1

        def close(self):
            counters["close"] += 1

    settings = Settings(
        data_dir=tmp_path / "console",
        evidence_root=tmp_path,
        web_dist=tmp_path / "web",
        deployments=[],
    )
    svc = ConsoleService(settings)
    jobs = module.ApplicationJobs(
        settings, svc, SimpleNamespace(node_id="control"), object()
    )
    monkeypatch.setattr(module, "RemoteEndpoint", Remote)
    run, _ = svc.store.create("application", {}, "alias", "digest")
    svc.benchmark_owner = "l10:" + run["id"]
    p = plan(
        targets=[
            {"id": "a", "machine_id": "first", "deployment_id": "m"},
            {"id": "b", "machine_id": "second", "deployment_id": "m"},
        ]
    )
    try:
        jobs.execute(run["id"], p, {"a": object(), "b": object()})
        row = svc.store.get(run["id"])
        assert row["state"] == "failed" and row["cleanup"] == "succeeded"
        assert counters == {"acquire": 1, "release": 1, "close": 2}
        assert svc.benchmark_owner is None and not jobs.quarantined
        doc, manifest = jobs.read(run["id"])
        assert "ALIASED" in doc["error"] and manifest["status"] == "partial"
        (
            settings.data_dir / "captures" / run["id"] / "application-receipt.json"
        ).write_text("{}")
        with pytest.raises(ValueError, match="INTEGRITY"):
            jobs.read(run["id"])
    finally:
        jobs.close()
        svc.close()


def test_owned_unload_never_releases_user_resident_model(tmp_path):
    from hqsb.console.config import Settings, DeploymentConfig
    from hqsb.console.service import ConsoleService, ServiceError
    from hqsb.console.application_transport import LocalEndpoint
    from hqsb.core.application_plan import ApplicationTarget

    settings = Settings(
        data_dir=tmp_path / "console",
        evidence_root=tmp_path,
        web_dist=tmp_path / "web",
        deployments=[DeploymentConfig(id="m", name="m", manifest="some-manifest")],
    )
    svc = ConsoleService(settings)
    original, _ = svc.store.create(
        "load", {"benchmark_id": "another-owner"}, "original", "digest"
    )
    svc.states["m"].update(state="ready", epoch="original", load_run_id=original["id"])
    svc.benchmark_owner = "l10:owner"
    endpoint = LocalEndpoint(
        svc,
        ApplicationTarget(id="a", deployment_id="m", lifecycle="managed"),
        "l10:owner",
    )
    try:
        with pytest.raises(ServiceError, match="本实验"):
            endpoint.unload("original", "unload", original["id"])
        assert svc.states["m"]["state"] == "ready" and svc.pending == 0
    finally:
        svc.close()


def test_participant_retry_recovers_lifecycle_after_state_change(tmp_path):
    import hashlib
    import json
    from hqsb.console.config import Settings, DeploymentConfig
    from hqsb.console.service import ConsoleService, ServiceError
    from hqsb.console.application_transport import LocalEndpoint
    from hqsb.core.application_plan import ApplicationTarget

    svc = ConsoleService(
        Settings(
            data_dir=tmp_path / "data",
            evidence_root=tmp_path,
            web_dist=tmp_path / "web",
            deployments=[DeploymentConfig(id="m", name="m", manifest="manifest")],
        )
    )
    owner = "l10:owner"
    svc.benchmark_owner = owner
    endpoint = LocalEndpoint(
        svc, ApplicationTarget(id="a", deployment_id="m", lifecycle="managed"), owner
    )
    try:
        for kind, payload, state in [
            ("load", {}, "ready"),
            (
                "unload",
                {"expected_epoch": "original", "load_run_id": "original-load"},
                "unloaded",
            ),
        ]:
            key = owner + ":" + kind
            digest = hashlib.sha256(
                json.dumps([kind, "m", payload], sort_keys=True).encode()
            ).hexdigest()
            original, _ = svc.store.create(kind, {"benchmark_id": owner}, key, digest)
            svc.states["m"]["state"] = state
            assert endpoint.operation(kind, payload, kind)["id"] == original["id"]
            with pytest.raises(ServiceError, match="another payload"):
                endpoint.operation(kind, {**payload, "changed": True}, kind)
        assert svc.pending == 0
    finally:
        svc.close()
