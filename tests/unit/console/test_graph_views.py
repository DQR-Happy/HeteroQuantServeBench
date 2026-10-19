"""Real exporter/compiler adapters plus integrity and diagnostic restrictions."""
import json
from types import SimpleNamespace

import pytest

from hqsb.core.graph_views import GraphCapture, validate_capture
from hqsb.backends.graph_formats import ir_graph, onnx_graph


def test_capture_contract_requires_separate_native_diagnostic():
    options = dict(runtime={"mode": "eager"}, native=True, privileged=False, observation="off")
    assert validate_capture({"kind": "onnx"}, **options)["kind"] == "onnx"
    for changed in ({"native": False}, {"privileged": True}, {"observation": "basic"},
                    {"quality": True}, {"tokens": 5}, {"runtime": None}, {"runtime": {"mode": "inductor"}}):
        with pytest.raises(ValueError):
            validate_capture({"kind": "onnx"}, **{**options, **changed})
    with pytest.raises(ValueError):
        GraphCapture(kind="arbitrary")


def test_ir_explicit_dependencies_not_order_or_fusion_guess():
    raw = """op0: SchedulerNode(ComputedBuffer)
op0.writes = OrderedSet([MemoryDep('buf0', c0, {c0: 4})])
op0.unmet_dependencies = OrderedSet([])

op1_op2: FusedSchedulerNode(SchedulerNode, SchedulerNode)
op1_op2.writes = OrderedSet([MemoryDep('buf2', c0, {c0: 4})])
op1_op2.unmet_dependencies = OrderedSet([MemoryDep('buf0', c0, {c0: 4})])
    op1: SchedulerNode(ComputedBuffer)
    op1.unmet_dependencies = OrderedSet([MemoryDep('internal', c0, {c0: 4})])
"""
    graph = ir_graph(raw)
    assert [n["id"] for n in graph["nodes"]] == ["op0", "op1_op2"]
    assert graph["edges"] == [{"source": "op0", "target": "op1_op2", "kind": "data", "buffer": "buf0"}]
    assert ir_graph("new unknown version")["parser_status"] == "unrecognized_or_empty"

    real_style = """op0_op1: FusedSchedulerNode(SchedulerNode,SchedulerNode)
op0_op1.writes = [MemoryDep('buf0', c0, {}), MemoryDep('buf1', c0, {})]
op0_op1.unmet_dependencies = []
op0_op1.snodes[0] =
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {})]
op0.unmet_dependencies = []
op0_op1.snodes[1] =
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {})]


op2: ExternKernelSchedulerNode(ExternKernelOut)
op2.writes = [StarDep(name='buf2', mode=None)]
op2.unmet_dependencies =
    [StarDep(name='buf1', mode=None)]
op2.met_dependencies = []
"""
    graph = ir_graph(real_style)
    assert [n["id"] for n in graph["nodes"]] == ["op0_op1", "op2"]
    assert graph["edges"] == [{"source":"op0_op1", "target":"op2", "kind":"data", "buffer":"buf1"}]


def test_real_onnx_export_keeps_weights_as_inputs_and_no_values(tmp_path):
    import torch
    import onnx
    from hqsb.backends.onnx_capture import capture

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 4)
            self.head = torch.nn.Linear(4, 16)

        def forward(self, input_ids, attention_mask, use_cache=False, return_dict=True):
            return SimpleNamespace(logits=self.head(self.embed(input_ids)) * attention_mask[..., None])

    model = Tiny().eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        reference = model(ids, torch.ones_like(ids)).logits
        entry = capture(model, model.forward, ids, torch.ones_like(ids), tmp_path)
        actual = model(ids, torch.ones_like(ids)).logits
    assert entry["status"] == "available", entry.get("error", entry)
    assert entry["checker"] == "passed"
    proto = onnx.load(tmp_path / entry["artifact"])
    assert not proto.graph.initializer
    assert len(proto.graph.input) > 2  # Resident weights are named graph inputs.
    assert any(n["target"].endswith("::Gemm") or n["target"].endswith("::MatMul") for n in entry["nodes"])
    assert "raw_data" not in json.dumps(entry)
    bounded = onnx_graph(proto, max_nodes=2)
    assert bounded["truncated"] and not bounded["edges"]
    torch.testing.assert_close(reference, actual, rtol=0, atol=0)


def test_actual_inductor_trace_and_cached_evidence_publication(tmp_path):
    import torch
    from hqsb.backends.runtime_session import RuntimeSession
    from hqsb.core.runtime_plan import RuntimePlan

    class Pointwise(torch.nn.Module):
        def forward(self, input_ids):
            return torch.sin(input_ids + 2) * 3

    model = Pointwise().eval()
    session = RuntimeSession(model, RuntimePlan(mode="inductor", shape_policy="static").model_dump(), "inductor")
    provider = SimpleNamespace(source_identity={"hash": "weights"}, tokenizer_identity="tokenizer",
                               source_environment={"source_digest": "source"})
    x = torch.randn(3, 4)
    expected = model(x)
    for identity in ("run_first", "run_second"):
        path = tmp_path / identity
        path.mkdir()
        request = {"capture_id": identity, "capture_dir": str(path), "expected_epoch": "epoch"}
        with torch.inference_mode(), session.activate(request):
            torch.testing.assert_close(model(input_ids=x), expected)
            session.save(provider, request)
        extra = json.loads((path / "graph-views.json").read_text())
        assert {v["kind"] for v in extra["views"]} == {"inductor_pre", "inductor_post", "generated_code"}
        assert all(v["status"] == "available" for v in extra["views"]), extra
        for view in extra["views"]:
            assert view["first_run_id"] == "run_first"
            assert (path / view["artifact"]).is_file()
        assert all(v.get("nodes") for v in extra["views"] if v["kind"] != "generated_code")
    assert "forward" not in model.__dict__
    torch._dynamo.reset()


def test_views_require_manifest_identity_and_bound_download_index(tmp_path):
    from fastapi.testclient import TestClient
    from hqsb.console.app import create_app, PREFIX
    from hqsb.console.config import Settings, DeploymentConfig
    from hqsb.experiments.artifacts import seal

    app = create_app(Settings(data_dir=tmp_path / "state", evidence_root=tmp_path,
        web_dist=tmp_path / "dist", deployments=[DeploymentConfig(id="model", name="test", model_path=str(tmp_path))]), monitor=False)
    with TestClient(app) as client:
        store = app.state.service.store
        run, _ = store.create("generate", {"deployment":{"id":"model"}, "epoch":"epoch", "runtime_plan":{"mode":"eager"}}, "test", "digest")
        endpoint = PREFIX + f'/runs/{run["id"]}/graph-views'
        assert client.get(endpoint).status_code == 409
        store.update(run["id"], {"state":"completed"})
        directory = tmp_path / "state" / "captures" / run["id"]
        directory.mkdir(parents=True)
        (directory / "runtime-graph.json").write_text(json.dumps({"run_id":run["id"], "epoch":"epoch", "source_digest":"source"}))
        (directory / "execution-receipt.json").write_text(json.dumps({"run_id":run["id"]}))
        (directory / "graph-debug-g1-0-ir_post_fusion.txt").write_text("actual compiler evidence")
        (directory / "graph-views.json").write_text(json.dumps({"run_id":run["id"], "epoch":"epoch", "source_digest":"source",
            "views":[{"kind":"inductor_post", "status":"available", "artifact":"graph-debug-g1-0-ir_post_fusion.txt"}]}))
        assert client.get(endpoint).status_code == 409
        seal(directory, run_id=run["id"])
        response = client.get(endpoint)
        assert response.status_code == 200, response.text
        view = response.json()["views"][0]
        download = client.get(PREFIX + f'/runs/{run["id"]}/experiment/artifacts/{view["artifact_index"]}')
        assert download.text == "actual compiler evidence"
        (directory / "graph-debug-g1-0-ir_post_fusion.txt").write_text("tampered")
        assert client.get(endpoint).status_code == 409
