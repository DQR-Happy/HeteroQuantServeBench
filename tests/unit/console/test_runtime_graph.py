"""L3 contracts, real Dynamo capture, restoration and sealed evidence boundaries."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hqsb.core.runtime_plan import RuntimePlan, validate_runtime


def test_runtime_contract_rejects_unsafe_composition():
    for options in ({"method": "gptq"}, {"kv_bits": 2}, {"capture": True},
                    {"observation": "operators"}, {"privileged": True}, {"kernel": "custom"}):
        with pytest.raises(ValueError):
            validate_runtime({"mode": "inductor"}, **options)
    with pytest.raises(ValueError):
        RuntimePlan.model_validate({"mode": "inductor", "arbitrary_code": "unsafe"})
    with pytest.raises(ValueError):
        RuntimePlan.model_validate({"fullgraph": "false"})
    assert validate_runtime({"mode": "eager"}, method="gptq")["mode"] == "eager"


def test_nested_module_stack_joins_exact_inventory_path():
    import torch
    from hqsb.backends.runtime_graph import module_path
    from hqsb.backends.runtime_session import RuntimeSession

    known = {"model.layers.0.self_attn.q_proj"}
    raw = "L['self']._modules['model']._modules['layers']._modules['0'].self_attn.q_proj"
    assert module_path(raw, known) == "model.layers.0.self_attn.q_proj"
    assert module_path("L['self'].model.layers[0].self_attn.q_proj", known) in known
    assert module_path(raw.replace("q_proj", "unobserved"), known) is None

    class Nested(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.ModuleDict({"proj": torch.nn.Linear(4, 4)})])

        def forward(self, input_ids):
            return self.layers[0]["proj"](input_ids)

    model = Nested().eval()
    session = RuntimeSession(model, RuntimePlan(mode="dynamo_eager").model_dump())
    with torch.inference_mode(), session.activate({}):
        model(input_ids=torch.ones(2, 4))
    assert any(n["module"] == "layers.0.proj" for g in session.graphs for n in g["nodes"] if n["kind"] == "call_function")
    torch._dynamo.reset()


def test_real_capture_reuses_graph_links_modules_and_restores_forward(tmp_path):
    import torch
    from hqsb.backends.runtime_session import RuntimeSession

    class Fork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Linear(4, 4)
            self.right = torch.nn.Linear(4, 4)

        def forward(self, input_ids):
            return self.left(input_ids) + self.right(input_ids)

    model = Fork().eval()
    x = torch.randn(2, 4)
    other = x + 1
    expected = model(x)
    plan = RuntimePlan(mode="dynamo_eager").model_dump()
    session = RuntimeSession(model, plan)
    provider = SimpleNamespace(source_identity={"hash": "weights"}, tokenizer_identity="tokenizer",
                               source_environment={"source_digest": "source"})
    request = {"capture_id": "run_test", "expected_epoch": "epoch", "capture_dir": str(tmp_path)}
    with torch.inference_mode(), session.activate(request):
        torch.testing.assert_close(model(input_ids=x), expected)
        torch.testing.assert_close(model(input_ids=other), model.left(other) + model.right(other))
        summary = session.save(provider, request)
    assert "forward" not in model.__dict__
    assert summary["status"] == "invoked" and summary["new_graphs"] == 1
    assert summary["graph_invocations"] == 2 and summary["lowering"] == "none"
    doc = json.loads((tmp_path / "runtime-graph.json").read_text())
    assert {n["module"] for g in doc["graphs"] for n in g["nodes"]} >= {"left", "right"}
    assert doc["graphs"][0]["edges"]
    with torch.inference_mode(), session.activate({**request, "capture_id": "run_next"}):
        model(input_ids=x)
        assert session.summary()["new_graphs"] == 0
        assert session.summary()["graph_invocations"] == 1
    torch._dynamo.reset()


def test_forward_failure_has_no_eager_fallback_and_restores_model():
    import torch
    from hqsb.backends.runtime_session import RuntimeSession

    class Broken(torch.nn.Module):
        def forward(self, input_ids):
            raise RuntimeError("deliberate failure")

    model = Broken()
    session = RuntimeSession(model, RuntimePlan(mode="dynamo_eager").model_dump())
    with pytest.raises(RuntimeError, match="deliberate failure"), session.activate({}):
        model(input_ids=torch.ones(2))
    assert "forward" not in model.__dict__
    assert session.summary()["status"] == "failed"
    torch._dynamo.reset()


def test_fullgraph_failure_is_visible_not_eager():
    import torch
    from hqsb.backends.runtime_session import RuntimeSession

    class Break(torch.nn.Module):
        def forward(self, input_ids):
            return input_ids + input_ids.sum().item()

    model = Break()
    session = RuntimeSession(model, RuntimePlan(mode="dynamo_eager", fullgraph=True).model_dump())
    with pytest.raises(Exception), session.activate({}):
        model(input_ids=torch.ones(2))
    assert "forward" not in model.__dict__
    assert session.summary()["completed_forwards"] == 0
    torch._dynamo.reset()


def test_graph_metadata_budget_never_has_dangling_edges():
    import torch
    from hqsb.backends.runtime_graph import describe_graph

    gm = torch.fx.symbolic_trace(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()))
    graph = describe_graph(gm, set(dict(gm.named_modules())), graph_id="g1", phase="test", max_nodes=2)
    ids = {n["id"] for n in graph["nodes"]}
    assert graph["truncated"]
    assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])


def test_recompile_limit_reports_failure_instead_of_silent_eager():
    import torch
    from hqsb.backends.runtime_session import RuntimeSession

    class Shapes(torch.nn.Module):
        def forward(self, input_ids):
            return input_ids.sin()

    model = Shapes()
    session = RuntimeSession(model, RuntimePlan(mode="dynamo_eager", shape_policy="static").model_dump())
    with torch.inference_mode(), torch._dynamo.config.patch(recompile_limit=1):
        with pytest.raises(Exception) as error, session.activate({}):
            model(input_ids=torch.ones(2))
            model(input_ids=torch.ones(3))
    assert "limit" in str(error.value).lower()
    assert session.summary()["status"] == "failed"
    assert session.summary()["completed_forwards"] == 1
    assert "forward" not in model.__dict__
    torch._dynamo.reset()


def test_graph_route_requires_terminal_sealed_identity_and_integrity(tmp_path):
    from hqsb.console.app import create_app, PREFIX
    from hqsb.console.config import Settings, DeploymentConfig
    from hqsb.experiments.artifacts import seal

    app = create_app(Settings(data_dir=tmp_path / "state", evidence_root=tmp_path,
        web_dist=tmp_path / "dist", deployments=[DeploymentConfig(id="model", name="test", model_path=str(tmp_path))]), monitor=False)
    with TestClient(app) as client:
        store = app.state.service.store
        run, _ = store.create("generate", {"deployment": {"id": "model"}, "epoch": "epoch",
            "runtime_plan": {"mode": "eager"}}, "test", "digest")
        endpoint = PREFIX + f'/runs/{run["id"]}/runtime-graph'
        assert client.get(endpoint).status_code == 409
        store.update(run["id"], {"state": "completed"})
        directory = tmp_path / "state" / "captures" / run["id"]
        directory.mkdir(parents=True)
        (directory / "runtime-graph.json").write_text(json.dumps({"run_id": run["id"], "epoch": "epoch", "graphs": []}))
        (directory / "execution-receipt.json").write_text(json.dumps({"run_id": run["id"]}))
        assert client.get(endpoint).status_code == 409
        seal(directory, run_id=run["id"])
        assert client.get(endpoint).json()["publication"] == "supervisor_acknowledged"
        assert client.get(PREFIX + "/runtime/runs?deployment_id=model").json()["items"][0]["run_id"] == run["id"]
        (directory / "runtime-graph.json").write_text("{}")
        assert client.get(endpoint).status_code == 409


def test_compiled_run_cannot_be_registered_as_float_reference():
    from hqsb.experiments.quality_contract import is_floating_reference
    p = {"method": "none", "quantized_module_count": 0, "execution": "floating_reference",
         "compute_dtype": "float16", "kv": {"bits": None, "backend": "model_default"}}
    r = {"runtime": {"kernel": "framework-native"}, "precision_receipt": p}
    assert is_floating_reference({"receipt": r})
    r["runtime_receipt"] = {"plan": {"mode": "inductor"}}
    assert not is_floating_reference({"receipt": r})


def test_api_routes_preserve_selected_plan_and_node_deadline(tmp_path):
    from hqsb.console.app import create_app, PREFIX
    from hqsb.console.config import Settings, DeploymentConfig
    from hqsb.console.service import ConsoleService

    class Actor:
        def __init__(self, _):
            pass

        def execute(self, *_):
            return {"finish_reason": "length", "metrics": {}}

        def close(self):
            pass

    settings = Settings(data_dir=tmp_path / "state", evidence_root=tmp_path, web_dist=tmp_path,
        request_deadline_ms=42000, deployments=[DeploymentConfig(id="model", name="test")])
    service = ConsoleService(settings, actor_factory=Actor)
    service.states["model"].update(state="ready", epoch="epoch")
    plan = {"mode": "inductor", "shape_policy": "dynamic", "fullgraph": False}
    with TestClient(create_app(settings, service=service, monitor=False)) as client:
        limit = client.get(PREFIX + "/runtime/catalog").json()["limits"]["max_deadline_ms"]
        assert limit == 42000
        body = {"deployment_id": "model", "expected_epoch": "epoch", "prompt": "Hello",
                "runtime_plan": plan, "observation_mode": "off", "max_output_tokens": 2}
        headers = {"X-HQSB-Client": "console", "Idempotency-Key": "limit"}
        assert client.post(PREFIX + "/requests", json=body, headers=headers).status_code == 422
        response = client.post(PREFIX + "/requests", json={**body, "deadline_ms": limit}, headers=headers)
        assert response.status_code == 202, response.text
        assert response.json()["config"]["runtime_plan"] == plan
        response = client.post(PREFIX + "/precision/quality-jobs", json={
            "deployment_id": "model", "expected_epoch": "epoch", "runtime_plan": plan,
            "quality_plan": {"benchmark_id": "wikitext2-mini-v1"}},
            headers={**headers, "Idempotency-Key": "quality"})
        assert response.status_code == 202, response.text
        assert response.json()["config"]["runtime_plan"] == plan
