"""Unit tests for Roofline/Amdahl analysis primitives."""

from __future__ import annotations

import pytest

from hqsb.benchmark.roofline import (
    HotspotClass,
    RooflineModel,
    amdahl_max_speedup,
    amdahl_speedup,
    classify_hotspot,
    rank_hotspots,
)
from hqsb.core.errors import ConfigError


@pytest.fixture
def model() -> RooflineModel:
    return RooflineModel(
        name="test",
        peak_flops=1e12,
        peak_bandwidth=100e9,
    )


@pytest.mark.unit
class TestRooflineModel:
    def test_ridge_point(self, model):
        assert model.ridge_point() == pytest.approx(10.0)

    def test_bandwidth_bound(self, model):
        # AI=5 FLOP/byte -> 500 GFLOP/s (bandwidth bound)
        assert model.achievable_flops(5.0) == pytest.approx(500e9)

    def test_compute_bound(self, model):
        # AI=20 -> min(1e12, 2e12) = 1e12
        assert model.achievable_flops(20.0) == pytest.approx(1e12)
        assert model.is_compute_bound(20.0) is True

    def test_is_bandwidth_bound(self, model):
        assert model.is_compute_bound(5.0) is False

    def test_efficiency(self, model):
        # At AI=5, bound=500e9; achieved 250e9 -> 0.5
        assert model.efficiency(250e9, 5.0) == pytest.approx(0.5)

    def test_efficiency_zero_bound(self):
        m = RooflineModel("zero", peak_flops=0.0, peak_bandwidth=0.0)
        assert m.efficiency(0.0, 1.0) == 0.0

    def test_negative_ai_rejected(self, model):
        with pytest.raises(ConfigError):
            model.achievable_flops(-1.0)


@pytest.mark.unit
class TestAmdahl:
    def test_no_parallel_work(self):
        assert amdahl_speedup(0.0, 100.0) == pytest.approx(1.0)

    def test_infinite_speedup_half(self):
        assert amdahl_speedup(0.5, 1e9) == pytest.approx(2.0)

    def test_max_speedup(self):
        assert amdahl_max_speedup(0.5) == pytest.approx(2.0)
        assert amdahl_max_speedup(0.9) == pytest.approx(10.0)

    def test_invalid_fraction(self):
        with pytest.raises(ConfigError):
            amdahl_speedup(1.5, 2.0)

    def test_invalid_speedup(self):
        with pytest.raises(ConfigError):
            amdahl_speedup(0.5, 0.5)

    def test_max_speedup_full_fraction_rejected(self):
        with pytest.raises(ConfigError):
            amdahl_max_speedup(1.0)


@pytest.mark.unit
class TestClassifyHotspot:
    def test_gemm(self):
        assert classify_hotspot("aten::mm") == HotspotClass.GEMM_ATTENTION

    def test_attention(self):
        assert classify_hotspot("aten::scaled_dot_product_attention") == HotspotClass.GEMM_ATTENTION

    def test_norm(self):
        assert classify_hotspot("aten::rms_norm") == HotspotClass.ELEMENTWISE_REDUCTION

    def test_runtime_sync(self):
        assert classify_hotspot("cudaStreamSynchronize") == HotspotClass.RUNTIME_SYNC

    def test_unknown(self):
        assert classify_hotspot("mystery_kernel") == HotspotClass.OTHER


@pytest.mark.unit
class TestRankHotspots:
    def test_sorts_by_share_and_computes_amdahl(self):
        ranked = rank_hotspots(
            [
                {"name": "aten::mm", "time_share": 0.5},
                {"name": "aten::rms_norm", "time_share": 0.3},
                {"name": "aten::add", "time_share": 0.2},
            ]
        )
        assert ranked[0].name == "aten::mm"
        assert ranked[0].classification == HotspotClass.GEMM_ATTENTION
        assert ranked[0].amdahl_max == pytest.approx(2.0)
        assert ranked[-1].name == "aten::add"
