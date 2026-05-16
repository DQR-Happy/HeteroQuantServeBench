"""Unit tests for the E02-02 shape census collector.

The collector must record, per (module, phase): call counts, input/output
shapes, dtypes, strides, layouts, and contiguity — all read off live
tensors during a real forward pass, never hand-copied. These tests run on
CPU with a tiny synthetic module so they are part of the fast CI gate.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from hqsb.benchmark.shape_census import (
    ShapeCensusCollector,
    _flatten_tensors,
    _tensor_meta,
)


class _TinyNet(torch.nn.Module):
    """A two-layer net with one contiguous and one transposed input path."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def test_tensor_meta_contiguous():
    t = torch.randn(2, 4)
    meta = _tensor_meta(t)
    assert meta["shape"] == [2, 4]
    assert meta["dtype"] == "float32"
    assert meta["device"] == "cpu"
    assert meta["stride"] == [4, 1]
    assert meta["layout"] == "contiguous"
    assert meta["contiguous"] is True


def test_tensor_meta_non_contiguous():
    t = torch.randn(4, 4).t()  # transposed -> non-contiguous
    meta = _tensor_meta(t)
    assert meta["contiguous"] is False
    assert meta["layout"] == "non_contiguous"


def test_tensor_meta_dtype_name():
    meta = _tensor_meta(torch.zeros(1, dtype=torch.float16))
    assert meta["dtype"] == "float16"


def test_flatten_handles_containers_and_skips_non_tensors():
    out = []
    _flatten_tensors(
        (
            torch.ones(1),
            [torch.ones(2), {"k": torch.ones(3)}],
            None,
            42,
            "text",
            True,
        ),
        out,
    )
    assert [list(t.shape) for t in out] == [[1], [2], [3]]


def test_flatten_handles_dataclass():
    @dataclasses.dataclass
    class Out:
        logits: torch.Tensor
        name: str

    out = []
    _flatten_tensors(Out(logits=torch.ones(2, 3), name="x"), out)
    assert [list(t.shape) for t in out] == [[2, 3]]


class _FakeCache:
    """Minimal stand-in for transformers ``DynamicCache``."""

    def __init__(self):
        self.key_cache = [torch.ones(1, 2, 3, 4)]
        self.value_cache = [torch.ones(1, 2, 3, 4)]


def test_flatten_handles_dynamic_cache():
    out = []
    _flatten_tensors(_FakeCache(), out)
    assert [list(t.shape) for t in out] == [[1, 2, 3, 4], [1, 2, 3, 4]]


@pytest.mark.unit
class TestShapeCensusCollector:
    def test_records_call_count_and_io(self):
        model = _TinyNet()
        collector = ShapeCensusCollector()
        collector.attach(model)

        collector.set_phase("prefill")
        model(torch.randn(1, 4))
        collector.set_phase("decode")
        model(torch.randn(1, 4))
        model(torch.randn(1, 4))
        collector.detach()

        records = {r["module"]: r for r in collector.records()}

        # root + fc are registered; both phases exist for fc.
        fc_prefill = next(
            r for r in collector.records() if r["module"] == "fc" and r["phase"] == "prefill"
        )
        fc_decode = next(
            r for r in collector.records() if r["module"] == "fc" and r["phase"] == "decode"
        )
        assert fc_prefill["call_count"] == 1
        assert fc_decode["call_count"] == 2
        assert fc_prefill["module_type"] == "Linear"
        assert fc_prefill["input_shapes"] == ["[1, 4]"]
        assert fc_prefill["output_shapes"] == ["[1, 3]"]
        assert fc_prefill["input_contiguous"] is True
        assert fc_prefill["output_contiguous"] is True
        assert "root" in records

    def test_detach_stops_recording(self):
        model = _TinyNet()
        collector = ShapeCensusCollector()
        collector.attach(model)
        collector.detach()
        model(torch.randn(1, 4))
        assert collector.records() == []

    def test_records_are_phase_split(self):
        model = _TinyNet()
        collector = ShapeCensusCollector()
        collector.attach(model)
        collector.set_phase("prefill")
        model(torch.randn(1, 4))
        collector.set_phase("decode")
        model(torch.randn(1, 4))
        collector.detach()

        phases = {r["phase"] for r in collector.records() if r["module"] == "fc"}
        assert phases == {"prefill", "decode"}
