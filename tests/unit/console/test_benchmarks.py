"""Fixed-suite and lifecycle contracts; actor double is never GPU evidence."""
import copy
import json
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from hqsb.console.app import PREFIX, create_app
from hqsb.console.benchmark_campaigns import BenchmarkCampaigns, FINAL
from hqsb.console.config import DeploymentConfig, Settings
from hqsb.console.service import ConsoleService, ServiceError
from hqsb.core.optimization import fingerprint
from hqsb.experiments.benchmark_suite import project_run, score, suite, summarize


def request(**changes):
    return {"deployment_id": "reference", "baseline_id": "framework-native",
            "candidate_ids": ["native-preserve-allocator"], "case_ids": ["arithmetic"],
            "repeats": 2, "output_tokens": 32, "diagnostics": False, **changes}


def wait(manager, identity):
    for _ in range(1000):
        document = manager.get(identity)
        if document["state"] in FINAL:
            return document
        time.sleep(.01)
    raise AssertionError("campaign did not terminate")


def run_fixture(case, profile="framework-native", cleanup="empty_cache", **changes):
    return {"id": uuid.uuid4().hex, "state": "completed", "cleanup": "succeeded", "output": "323",
            "finish_reason": "eos", "metrics": {"runtime_first_token_ms": 10, "runtime_e2e_ms": 20,
            "decode_tail_tokens_per_s": 100, "input_tokens": 60, "output_tokens": 2,
            "input_token_sha256": case["input_sha256"], "generated_token_ids": [1, 2],
            "optimization_receipt": {"status": "applied", "effective": {"kernel_profile": profile, "cleanup_policy": cleanup},
            "model_identity": "test-model", "source_digest": "test-source", "tokenizer_identity": "test-tokenizer"}}, **changes}


class ActorDouble:
    def __init__(self, config):
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls = []

    def execute(self, kind, payload, on_output, is_cancelled, timeout_s):
        self.calls.append((kind, payload))
        if kind != "generate":
            return {"weight_verification": "test-only"}
        self.started.set()
        while not self.release.wait(.01):
            if is_cancelled():
                return {"finish_reason": "cancelled", "text": "", "metrics": {}}
        fixture = run_fixture({"input_sha256": fingerprint(payload["messages"])})
        fixture["metrics"]["optimization_receipt"]["effective"] = payload["optimization"]["changes"]
        return {"finish_reason": "eos", "text": fixture["output"], "metrics": fixture["metrics"]}

    def close(self):
        self.release.set()


@pytest.fixture
def env(tmp_path):
    settings = Settings(data_dir=tmp_path / "state", evidence_root=tmp_path,
                        web_dist=tmp_path / "dist", deployments=[DeploymentConfig(id="reference", name="test")])
    actors = []
    def factory(config):
        actor = ActorDouble(config)
        actors.append(actor)
        return actor
    service = ConsoleService(settings, actor_factory=factory)
    manager = BenchmarkCampaigns(settings, service)
    yield settings, service, manager, actors
    manager.close()
    service.close()


def test_frozen_inputs_and_quality_do_not_confuse_agreement_with_correctness():
    first, second = suite(), suite()
    assert first == second and len(first["cases"]) == 4
    cases = {case["id"]: case for case in first["cases"]}
    assert all(case["input_sha256"] == fingerprint(case["messages"]) for case in cases.values())
    assert score(cases["arithmetic"], "323\n") is True
    assert score(cases["arithmetic"], "answer: 323") is False
    assert score(cases["structured"], '{"quantity":3,"sku":"AX17"}') is True
    assert score(cases["structured"], '{"quantity":3.0,"sku":"AX17"}') is False
    assert score(cases["structured"], '```json\n{}\n```') is False
    assert score(cases["explanation"], "anything") is None


def test_non_qwen_model_has_a_runnable_native_comparison_and_rejects_custom_load(env):
    _, service, manager, _ = env
    service.configs["reference"].model = "google/gemma-3-1b-it"
    catalog = manager.catalogue("reference")
    assert catalog["defaults"]["candidate_ids"] == ["native-preserve-allocator"]
    with pytest.raises(ServiceError, match="KERNEL_MODEL_SHAPE_UNSUPPORTED"):
        service.submit("load", "reference", {"kernel_profile": "qwen3-attention-math-v1"}, "incompatible")
    assert service.pending == 0


def test_report_excludes_warmups_and_blocks_unfair_ratios():
    case = suite()["cases"][0]
    schemes = [{"id": "base", "kernel_profile": "framework-native", "cleanup_policy": "empty_cache"},
               {"id": "candidate", "kernel_profile": "framework-native", "cleanup_policy": "preserve_allocator"}]
    plan = {"cases": [case], "schemes": schemes, "baseline_id": "base", "repeats": 1}
    records = []
    for scheme in schemes:
        run = run_fixture(case, cleanup=scheme["cleanup_policy"], output="incorrect")
        records.append(project_run(run, scheme_id=scheme["id"], case=case, role="measurement", repeat=0))
    warmup = copy.deepcopy(records[1])
    warmup.update(role="warmup", score=True)
    warmup["metrics"]["runtime_e2e_ms"] = 99999
    rows = summarize(plan, records + [warmup])["rows"]
    assert rows[1]["metrics"]["runtime_e2e_ms"] == 20
    assert rows[1]["task_accuracy"] == 0 and rows[1]["token_agreement"] == 1
    assert rows[1]["relative_e2e"] == 0
    records[1]["metrics"]["output_tokens"] = 1
    row = summarize(plan, records)["rows"][1]
    assert row["relative_e2e"] is None and "OUTPUT_LENGTH_CHANGED" in row["reasons"]
    records[1]["metrics"]["output_tokens"] = 2
    records[0]["receipt"]["status"] = "not_fully_applied"
    row = summarize(plan, records)["rows"][1]
    assert not row["comparable"] and "BASELINE_EXECUTION_NOT_CONFIRMED" in row["reasons"]
    records[0]["cleanup"] = "unknown"
    assert "BASELINE_INCOMPLETE" in summarize(plan, records)["rows"][1]["reasons"]
    records[0]["cleanup"] = "succeeded"
    records[0]["receipt"]["status"] = "applied"
    records[1]["metrics"]["runtime_e2e_ms"] = None
    row = summarize(plan, records)["rows"][1]
    assert row["relative_e2e"] is None and row["metrics"]["runtime_e2e_ms"] is None
    assert "TIMING_UNAVAILABLE" in row["reasons"]


def test_campaign_persists_idempotent_results_and_restores_unloaded(env):
    _, service, manager, actors = env
    created = manager.start(request(diagnostics=True), "same-key")
    done = wait(manager, created["id"])
    assert done["state"] == "completed", done
    assert len(done["records"]) == 8  # 2 warmups, 4 measurements, 2 independent diagnostics
    assert all(row["completed"] == 2 and row["task_accuracy"] == 1 for row in done["report"]["rows"])
    assert service.deployments()[0]["state"] == "unloaded"
    assert service.benchmark_owner is None
    assert manager.start(request(diagnostics=True), "same-key")["id"] == done["id"]
    assert sum(kind == "generate" for actor in actors for kind, _ in actor.calls) == 8
    assert len(manager.list()["items"]) == 1
    restored = BenchmarkCampaigns(manager.settings, service)
    assert restored.get(done["id"])["report_sha256"] == done["report_sha256"]
    with pytest.raises(ServiceError, match="提交键"):
        manager.start(request(), "same-key")
    path = manager.root / (done["id"] + ".json")
    raw = json.loads(path.read_text())
    raw["records"][0]["output"] = "tampered"
    path.write_text(json.dumps(raw))
    with pytest.raises(ServiceError, match="校验"):
        manager.get(done["id"])


def test_cancel_excludes_foreign_jobs_and_restores_ready_profile(env):
    _, service, manager, actors = env
    run = service.submit("load", "reference", {}, "initial-load")
    for _ in range(500):
        if service.store.get(run["id"])["state"] == "completed":
            break
        time.sleep(.01)
    actor = actors[0]
    actor.release.clear()
    created = manager.start(request(), "cancel-key")
    assert actor.started.wait(5)
    assert service.submit("load", "reference", {}, "initial-load")["id"] == run["id"]
    with pytest.raises(ServiceError, match="独占"):
        service.submit("unload", "reference", {}, "foreign-key")
    assert manager.cancel(created["id"])["state"] == "cancelling"
    done = wait(manager, created["id"])
    assert done["state"] == "cancelled", done
    assert all(row["completed"] == 0 for row in done["report"]["rows"])
    assert service.deployments()[0]["state"] == "ready"
    assert service.deployments()[0]["kernel_profile"] == "framework-native"
    assert service.benchmark_owner is None
    assert manager.cancel(created["id"])["state"] == "cancelled"


def test_restart_marks_interrupted_without_submitting(env):
    _, service, manager, _ = env
    done = wait(manager, manager.start(request(repeats=1), "restart-key")["id"])
    document = manager.get(done["id"])
    document.update(state="running", phase="measurement", current_run_id="accepted-before-crash")
    manager._save(document)
    partial = manager.root / (done["id"] + ".json.partial")
    partial.write_text('{"partial-write":')
    before = len(service.store.list()["items"])
    recovered = BenchmarkCampaigns(manager.settings, service).get(done["id"])
    assert recovered["state"] == "interrupted"
    assert recovered["current_run_id"] == "accepted-before-crash"
    assert not partial.exists()
    assert len(list(manager.root.glob("*.interrupted"))) == 1
    assert len(service.store.list()["items"]) == before


def test_http_validation_and_no_login_catalog(env):
    settings, service, _, _ = env
    app = create_app(settings, service=service, monitor=False)
    with TestClient(app) as client:
        response = client.get(PREFIX + "/benchmark-catalog", params={"deployment_id": "reference"})
        assert response.status_code == 200
        catalog = response.json()
        assert len(catalog["cases"]) == 4
        assert not next(row for row in catalog["schemes"] if row["id"] == "qwen3-norm-vector-v2")["availability"]["loadable"]
        headers = {"x-hqsb-client": "console", "Idempotency-Key": "validation"}
        for body in (request(repeats=0), request(output_tokens=129), request(case_ids=[]), request(case_ids=["unknown"]),
                     request(candidate_ids=["framework-native"]), request(candidate_ids=["qwen3-norm-vector-v2"])):
            response = client.post(PREFIX + "/benchmarks", json=body, headers=headers)
            assert response.status_code == 422, response.text
        assert service.pending == 0 and service.benchmark_owner is None
        assert client.get(PREFIX + "/benchmarks/not-a-valid-id").status_code == 422
        assert client.get(PREFIX + "/session").json()["auth_required"] is False


def test_profile_switch_loads_exact_candidate_and_returns_to_ready_baseline(env):
    _, service, manager, actors = env
    first = service.submit("load", "reference", {}, "initial")
    for _ in range(500):
        if service.store.get(first["id"])["state"] == "completed":
            break
        time.sleep(.01)
    done = wait(manager, manager.start(request(candidate_ids=["qwen3-attention-math-v1"], repeats=1), "profiles")["id"])
    assert done["state"] == "completed", done
    assert [row["payload"].get("kernel_profile") for row in done["lifecycle"] if row["kind"] == "load"] == ["qwen3-attention-math-v1", "framework-native"]
    assert service.deployments()[0]["kernel_profile"] == "framework-native"
    candidate = next(row for row in done["records"] if row["scheme_id"] == "qwen3-attention-math-v1")
    assert candidate["receipt"]["effective"]["kernel_profile"] == "qwen3-attention-math-v1"
    assert all(service.store.get(row["run_id"])["config"]["benchmark_id"] == done["id"] for row in done["lifecycle"])
