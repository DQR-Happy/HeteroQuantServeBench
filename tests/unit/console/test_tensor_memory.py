"""Small CPU tensors verify storage alias accounting; no CUDA initialization."""

from types import SimpleNamespace

import pytest

from hqsb.backends import tensor_memory
from hqsb.backends.tensor_memory import cache_inventory, parameter_inventory


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_parameters_buffers_and_tied_names_deduplicate_storage(torch):
    model = torch.nn.Module()
    storage = torch.arange(12, dtype=torch.float32)
    model.register_parameter("weight", torch.nn.Parameter(storage))
    model.register_parameter("tied", model.weight)
    model.register_parameter("view", torch.nn.Parameter(storage[3:9]))
    model.register_buffer("buffer_alias", storage[1:5])
    model.register_buffer("independent", torch.zeros(2, dtype=torch.int64))
    result = parameter_inventory(model)
    entries = {row["name"]: row for row in result["entries"]}
    assert result["availability"] == "complete"
    assert set(entries) == {"weight", "tied", "view", "buffer_alias", "independent"}
    assert result["logical_total_bytes"] == (12 + 12 + 6 + 4) * 4 + 2 * 8
    assert result["unique_storage_bytes"] == 12 * 4 + 2 * 8
    assert result["unique_storages"] == 2
    assert (
        entries["weight"]["storage_key"]
        == entries["view"]["storage_key"]
        == entries["buffer_alias"]["storage_key"]
    )
    assert entries["view"]["storage_offset"] == 3
    assert entries["view"]["storage_offset_bytes"] == 12
    assert entries["buffer_alias"]["storage_bytes"] == 48
    assert entries["weight"]["device"] == "cpu"
    assert all(
        "data_ptr" not in row and "values" not in row for row in result["entries"]
    )
    assert entries["weight"]["storage_key"] != str(storage.data_ptr())


def test_legacy_cache_views_share_storage_without_copy(torch):
    base = torch.zeros(1, 2, 8, 4)
    key, value = base[:, :, :4, :], base[:, :, 4:, :]
    cache = ((key, value), (key, value))
    before = (base.data_ptr(), key.data_ptr(), value.data_ptr(), base._version)
    result = cache_inventory(cache)
    assert result["layout"] == "legacy_layer_tuples"
    assert result["entry_count"] == 4
    assert result["unique_storage_bytes"] == base.numel() * base.element_size()
    assert result["logical_total_bytes"] == 2 * base.numel() * base.element_size()
    assert result["unique_storages"] == 1
    assert before == (base.data_ptr(), key.data_ptr(), value.data_ptr(), base._version)
    assert result["entries"][1]["storage_offset"] > 0


def test_hf_cache_representations_and_cross_attention(torch):
    key, value = torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)
    legacy_hf = SimpleNamespace(key_cache=[key], value_cache=[value])
    layer_hf = SimpleNamespace(layers=[SimpleNamespace(keys=key, values=value)])
    old = cache_inventory(legacy_hf)
    new = cache_inventory(layer_hf)
    assert old["layout"] == "hf_key_value_cache"
    assert new["layout"] == "hf_cache_layers"
    assert (
        old["unique_storage_bytes"]
        == new["unique_storage_bytes"]
        == 2 * key.numel() * key.element_size()
    )
    assert old["entries"][0]["storage_key"] == new["entries"][0]["storage_key"]
    cross = cache_inventory(((key, value, key, value),))
    assert [row["kind"] for row in cross["entries"]] == [
        "key",
        "value",
        "cross_key",
        "cross_value",
    ]
    assert cross["unique_storages"] == 2


def test_missing_and_unsupported_cache_do_not_claim_zero_memory():
    for cache in (
        None,
        object(),
        SimpleNamespace(key_cache=[None], value_cache=[None]),
    ):
        result = cache_inventory(cache)
        assert result["availability"] == "unavailable"
        assert result["unique_storage_bytes"] is None
        assert result["unobserved"]
    empty = cache_inventory(())
    assert empty["availability"] == "complete" and empty["unique_storage_bytes"] == 0


def test_unmaterialized_meta_tensors_have_unknown_storage(torch):
    model = torch.nn.Linear(2, 3, device="meta")
    result = parameter_inventory(model)
    assert result["availability"] == "partial"
    assert result["logical_total_bytes"] == (2 * 3 + 3) * 4
    assert result["storage_observed_entries"] == 0
    assert result["unique_storage_bytes"] is None
    assert all(row["storage_key"] is None for row in result["entries"])
    assert result["unobserved"]


def test_old_named_tensor_api_fallback_is_explicit(torch):
    weight = torch.zeros(2)

    class OlderModule:
        def named_parameters(self):
            return iter([("weight", weight)])

        def named_buffers(self):
            return iter([])

    result = parameter_inventory(OlderModule())
    assert result["unique_storage_bytes"] == 8
    assert result["availability"] == "partial"
    assert any("remove_duplicate=False" in item for item in result["limitations"])


def test_inventory_budget_is_shared_and_truncation_is_visible(torch, monkeypatch):
    monkeypatch.setattr(tensor_memory, "MAX_ENTRIES", 3)
    weight = torch.zeros(2)
    model = SimpleNamespace(
        named_parameters=lambda **kwargs: (
            (str(index), weight) for index in range(10_000)
        ),
        named_buffers=lambda **kwargs: iter([]),
    )
    result = parameter_inventory(model)
    assert result["entry_count"] == 3 and result["truncated"] is True
    assert result["availability"] == "partial" and result["unique_storage_bytes"] == 8
    cache = cache_inventory(((weight, weight),) * 10_000)
    assert cache["entry_count"] == 3 and cache["truncated"] is True
    assert cache["unique_storage_bytes"] == 8


def test_invalid_tensor_generator_is_bounded(monkeypatch):
    monkeypatch.setattr(tensor_memory, "MAX_ENTRIES", 3)
    attempted = []

    def invalid_parameters(**kwargs):
        for index in range(10_000):
            attempted.append(index)
            yield str(index), object()

    result = parameter_inventory(SimpleNamespace(named_parameters=invalid_parameters))
    assert len(attempted) == 4
    assert result["truncated"] is True
    assert result["availability"] == "unavailable"
    assert result["logical_total_bytes"] is None
