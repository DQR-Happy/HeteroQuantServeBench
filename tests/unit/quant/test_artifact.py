from __future__ import annotations

import json

import pytest

from hqsb.quant.legacy_e05_01.artifact import (
    ArtifactValidationError,
    load_quant_artifact,
    save_quant_artifact,
)
from hqsb.quant.legacy_e05_01.rtn import RtnSpec, dequantize, quantize


SOURCE_HASH = "a" * 64


def _artifact(path, *, symmetric=True):
    tensor = quantize(
        [[-7.0, -0.5, 0.0, 0.5, 7.0], [2.0, 2.0, 2.0, 2.0, 2.0]],
        RtnSpec(
            bits=4,
            symmetric=symmetric,
            granularity="per-group",
            group_size=4,
        ),
    )
    manifest = save_quant_artifact(
        path,
        tensor,
        source_model_hash=SOURCE_HASH,
        tensor_name="model.layers.0.mlp.down_proj.weight",
    )
    return tensor, manifest


@pytest.mark.unit
@pytest.mark.property
@pytest.mark.parametrize("symmetric", [True, False])
def test_save_load_round_trip_and_identity_stability(tmp_path, symmetric):
    original, first = _artifact(tmp_path / "first", symmetric=symmetric)
    _same, second = _artifact(tmp_path / "second", symmetric=symmetric)
    loaded, loaded_manifest = load_quant_artifact(tmp_path / "first")
    assert loaded.qvalues == original.qvalues
    assert loaded.scales == original.scales
    assert loaded.zero_points == original.zero_points
    assert dequantize(loaded) == dequantize(original)
    assert loaded_manifest["artifact_id"] == first["artifact_id"]
    assert second["artifact_id"] == first["artifact_id"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda value: value.update(version="9.0.0"), "VERSION_UNSUPPORTED"),
        (
            lambda value: value["identity"]["tensor"].update(original_layout="column-major"),
            "LAYOUT_UNSUPPORTED",
        ),
        (
            lambda value: value["identity"]["quantization"].update(axis=0),
            "GROUP_AXIS_INVALID",
        ),
    ],
)
def test_manifest_faults_rejected_before_use(tmp_path, mutation, reason):
    _artifact(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    value = json.loads(manifest_path.read_text())
    mutation(value)
    manifest_path.write_text(json.dumps(value))
    with pytest.raises(ArtifactValidationError) as caught:
        load_quant_artifact(tmp_path)
    assert caught.value.reason_code == reason


@pytest.mark.unit
def test_missing_scale_and_bad_q_checksum_rejected(tmp_path):
    _artifact(tmp_path)
    (tmp_path / "scales.bin").unlink()
    with pytest.raises(ArtifactValidationError) as caught:
        load_quant_artifact(tmp_path)
    assert caught.value.reason_code == "FILE_MISSING"

    _artifact(tmp_path)
    qpath = tmp_path / "qvalues.bin"
    payload = bytearray(qpath.read_bytes())
    payload[0] ^= 1
    qpath.write_bytes(payload)
    with pytest.raises(ArtifactValidationError) as caught:
        load_quant_artifact(tmp_path)
    assert caught.value.reason_code == "CHECKSUM_MISMATCH"


@pytest.mark.unit
def test_kernel_specific_layout_is_not_confused_with_canonical(tmp_path):
    _artifact(tmp_path)
    with pytest.raises(ArtifactValidationError) as caught:
        load_quant_artifact(tmp_path, expected_kernel_id="cutlass-sm87-w4")
    assert caught.value.reason_code == "KERNEL_INCOMPATIBLE"
