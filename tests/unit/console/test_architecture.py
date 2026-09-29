"""Artifact and dependency evidence invariants, without real model allocations."""
import json
import math
import struct
import sys

import pytest
from fastapi.testclient import TestClient

from hqsb.models.architecture import inspect_architecture
from hqsb.models.architecture_graphs import graphs, layer_kinds
from hqsb.console.app import create_app, PREFIX
from hqsb.console.config import DeploymentConfig, Settings


def artifact(tmp_path, model_type="qwen3"):
    config = {"model_type":model_type,"num_hidden_layers":2,"hidden_size":8,
        "intermediate_size":16,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
        "vocab_size":10,"sliding_window_pattern":6}
    (tmp_path / "config.json").write_text(json.dumps(config))
    rows = {}
    for i in range(2):
        shapes = {"self_attn.q_proj.weight":[8,8],"self_attn.k_proj.weight":[4,8],"self_attn.v_proj.weight":[4,8],
            "self_attn.o_proj.weight":[8,8],"mlp.gate_proj.weight":[16,8],"mlp.up_proj.weight":[16,8],"mlp.down_proj.weight":[8,16],
            "input_layernorm.weight":[8],"post_attention_layernorm.weight":[8]}
        if model_type in {"qwen3","gemma3_text"}:
            shapes.update({"self_attn.q_norm.weight":[4],"self_attn.k_norm.weight":[4]})
        if model_type == "gemma3_text":
            shapes.update({"pre_feedforward_layernorm.weight":[8],"post_feedforward_layernorm.weight":[8]})
        for name, shape in shapes.items():
            rows[f"model.layers.{i}.{name}"] = {"shape":shape,"dtype":"F16","data_offsets":[0,0]}
    write_weights(tmp_path, rows)
    return config, rows


def write_weights(path, rows):
    offset = 0
    for row in rows.values():
        size = math.prod(row['shape']) * 2
        row['data_offsets'] = [offset,offset + size]
        offset += size
    data = json.dumps(rows).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q",len(data)) + data + bytes(offset))


@pytest.fixture(autouse=True)
def reviewed(monkeypatch):
    monkeypatch.setattr("hqsb.models.architecture.implementation", lambda kind: {"verified":kind in {"qwen2","qwen3","gemma3_text"},"reason":None,"version":"test-only"})


def test_headers_and_graph_are_available_without_importing_torch(tmp_path):
    artifact(tmp_path)
    before = "torch" in sys.modules
    result = inspect_architecture(str(tmp_path))
    assert ("torch" in sys.modules) == before
    assert result["status"] == "verified_static"
    assert len(result["layers"]) == 2
    assert len(result["graphs"]["attention"]["nodes"]) > 10
    assert result["shards"][0]["header_sha256"]


def test_missing_or_mismatched_weight_cannot_get_a_verified_dependency_graph(tmp_path):
    _, rows = artifact(tmp_path)
    del rows['model.layers.1.mlp.down_proj.weight']
    rows['model.layers.0.self_attn.k_proj.weight']['shape'] = [8,8]
    write_weights(tmp_path, rows)
    result = inspect_architecture(str(tmp_path))
    assert result["graphs"] is None and result["status"] == "metadata_only"
    assert any("shape" in issue for issue in result["issues"])
    assert any("down_proj" in issue for issue in result["issues"])


def test_unrecognized_family_gets_no_guessed_transformer_diagram(tmp_path):
    artifact(tmp_path, "unknown")
    result = inspect_architecture(str(tmp_path))
    assert result["graphs"] is None
    assert result["tensors"]


def test_truncated_shard_and_oversize_header_are_explicit_errors(tmp_path):
    artifact(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", 100))
    with pytest.raises(ValueError, match="截断"):
        inspect_architecture(str(tmp_path))
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", 20 * 1024**2))
    with pytest.raises(ValueError, match="预算"):
        inspect_architecture(str(tmp_path))


def test_shard_index_mismatch_prevents_complete_structure_claim(tmp_path):
    artifact(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{"model.layers.2.x":"missing.safetensors"}}))
    result = inspect_architecture(str(tmp_path))
    assert result["graphs"] is None
    assert any("分片不匹配" in issue for issue in result["issues"])


def test_family_norms_gating_and_layer_dependencies_are_distinct():
    qwen2, qwen3, gemma = (graphs(name,{}) for name in ["qwen2","qwen3","gemma3_text"])
    assert not any(row["id"] == "qn" for row in qwen2["attention"]["nodes"])
    assert any(row["module"] == "self_attn.q_norm" for row in qwen3["attention"]["nodes"])
    assert any(row["module"] == "post_feedforward_layernorm" for row in gemma["block"]["nodes"])
    assert next(row for row in gemma["mlp"]["nodes"] if row["id"] == "act")["label"] == "gelu_pytorch_tanh"
    assert {row["source"] for row in qwen3["mlp"]["edges"] if row["target"] == "mul"} == {"act","up"}
    assert layer_kinds({"model_type":"gemma3_text","sliding_window_pattern":6}, 7) == ["sliding_attention"] * 5 + ["full_attention","sliding_attention"]


def test_graphs_are_dags_and_parallel_branches_have_no_cross_dependency():
    for kind in ["qwen2","qwen3","gemma3_text"]:
        for graph in graphs(kind,{}).values():
            ids = {row['id'] for row in graph['nodes']}
            incoming = {node:0 for node in ids}
            for edge in graph['edges']:
                assert edge['source'] in ids and edge['target'] in ids
                incoming[edge['target']] += 1
            visited = set()
            while pending := {node for node, degree in incoming.items() if degree == 0} - visited:
                for node in pending:
                    visited.add(node)
                    for edge in graph['edges']:
                        if edge['source'] == node:
                            incoming[edge['target']] -= 1
            assert visited == ids


def test_route_is_read_only_and_does_not_load_a_worker(tmp_path):
    artifact(tmp_path)
    settings = Settings(data_dir=tmp_path/'state',evidence_root=tmp_path,web_dist=tmp_path/'dist',
        deployments=[DeploymentConfig(id='model',name='test',model_path=str(tmp_path))])
    app = create_app(settings, monitor=False)
    with TestClient(app) as client:
        response = client.get(PREFIX+'/deployments/model/architecture')
        assert response.status_code == 200
        assert response.json()['runtime']['state'] == 'unloaded'
        assert client.get(PREFIX+'/runs').json()['items'] == []
        assert client.get(PREFIX+'/deployments/missing/architecture').status_code == 404
