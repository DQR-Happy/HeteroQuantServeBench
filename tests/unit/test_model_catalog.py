"""ModelScope metadata selection is independent of model execution."""
import pytest

from hqsb.models.catalog import runtime_files


def entry(name, **changes):
    return {"Path": name, "Sha256": "a" * 64, "Size": 10, "Type": "blob", **changes}


def files():
    return [entry(name) for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")]


def test_selects_local_runtime_assets_without_code_or_alternate_checkpoints():
    selected = runtime_files(files() + [entry(name) for name in (
        "modeling_remote.py", "pytorch_model.bin", "optimizer.pt", "nested/model.safetensors",
        "../config.json", "/config.json", "a\\config.json")])
    assert {item["Path"] for item in selected} == {item["Path"] for item in files()}


@pytest.mark.parametrize("name", ["config.json", "model.safetensors"])
def test_publisher_hash_required_even_for_non_weight_assets(name):
    items = [entry(item["Path"], Sha256="" if item["Path"] == name else "a" * 64) for item in files()]
    with pytest.raises(ValueError, match="PUBLISHER_HASH_MISSING"):
        runtime_files(items)


def test_incomplete_or_duplicate_metadata_cannot_seal_a_snapshot():
    with pytest.raises(ValueError, match="WEIGHTS_MISSING"):
        runtime_files(files()[:-1])
    with pytest.raises(ValueError, match="RUNTIME_FILES_MISSING"):
        runtime_files(files()[1:])
    with pytest.raises(ValueError, match="DUPLICATE_FILE"):
        runtime_files(files() + [entry("config.json")])
