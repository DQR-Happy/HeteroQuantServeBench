from __future__ import annotations

import copy

import torch

from hqsb.quant.model_weight_only import (
    apply_model_quant_artifact,
    artifact_disk_usage,
    load_manifest,
    save_model_quant_artifact,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(5, 3, bias=False)
        self.lm_head = torch.nn.Linear(3, 7, bias=False)


def _model() -> TinyModel:
    model = TinyModel()
    with torch.no_grad():
        model.proj.weight.copy_(
            torch.tensor(
                [
                    [-1.0, -0.5, 0.0, 0.5, 1.0],
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        )
    return model


@torch.no_grad()
def test_model_artifact_w8_roundtrip_and_exclusion(tmp_path) -> None:
    source = _model()
    target = copy.deepcopy(source)
    original_lm_head = target.lm_head.weight.detach().clone()
    manifest = save_model_quant_artifact(
        source,
        tmp_path / "w8",
        bits=8,
        group_size=None,
        source_model_hash="model-hash",
        source_revision="revision",
        row_chunk=2,
    )
    target.proj.weight.zero_()
    result = apply_model_quant_artifact(
        target,
        tmp_path / "w8",
        expected_source_model_hash="model-hash",
        row_chunk=2,
    )
    assert result["bits"] == 8
    assert manifest["coverage"]["selected_tensor_count"] == 1
    assert torch.allclose(target.proj.weight, source.proj.weight, atol=0.01, rtol=0)
    assert torch.equal(target.lm_head.weight, original_lm_head)
    assert artifact_disk_usage(tmp_path / "w8")["qvalues"] == 15


@torch.no_grad()
def test_model_artifact_w4_odd_tail_identity_and_reload(tmp_path) -> None:
    source = _model()
    first = save_model_quant_artifact(
        source,
        tmp_path / "first",
        bits=4,
        group_size=4,
        source_model_hash="model-hash",
        source_revision="revision",
        row_chunk=2,
    )
    second = save_model_quant_artifact(
        source,
        tmp_path / "second",
        bits=4,
        group_size=4,
        source_model_hash="model-hash",
        source_revision="revision",
        row_chunk=3,
    )
    assert first["artifact_id"] == second["artifact_id"]
    assert load_manifest(tmp_path / "first")["artifact_id"] == first["artifact_id"]
    assert artifact_disk_usage(tmp_path / "first")["qvalues"] == 9

    target = copy.deepcopy(source)
    target.proj.weight.zero_()
    apply_model_quant_artifact(
        target,
        tmp_path / "first",
        expected_source_model_hash="model-hash",
        row_chunk=1,
    )
    assert torch.allclose(target.proj.weight, source.proj.weight, atol=0.08, rtol=0)


def test_model_artifact_rejects_wrong_source(tmp_path) -> None:
    model = _model()
    save_model_quant_artifact(
        model,
        tmp_path / "w8",
        bits=8,
        group_size=None,
        source_model_hash="model-hash",
        source_revision="revision",
    )
    try:
        apply_model_quant_artifact(
            model,
            tmp_path / "w8",
            expected_source_model_hash="wrong-hash",
        )
    except ValueError as exc:
        assert "source model hash" in str(exc)
    else:
        raise AssertionError("wrong source model hash was accepted")
