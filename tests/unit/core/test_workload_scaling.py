"""Unit tests for the E02-03 request-level scaling summary.

``request_summary`` aggregates per-request TTFT / TPOT / E2E and the three
throughputs (prefill / decode-tail / output) into P50/P95 summaries, keeping
request-level metrics separate from step-level ITL. These tests are pure
Python (no model, no GPU).
"""

from __future__ import annotations

import math

import pytest

from hqsb.benchmark.metrics import request_summary


def _request(prefill_ms=100.0, selection_ms=0.5, itl=None, isl=128, osl=32):
    return {
        "prefill_forward_ms": prefill_ms,
        "first_token_selection_ms": selection_ms,
        "raw_itl_ms": itl if itl is not None else [10.0] * (osl - 1),
        "input_tokens": isl,
        "output_tokens": osl,
    }


class TestRequestSummary:
    def test_ttft_is_prefill_plus_selection(self):
        s = request_summary([_request(prefill_ms=100.0, selection_ms=0.5)])
        # TTFT = 100.5 ms for the single request.
        assert s["ttft_ms"]["p50"] == pytest.approx(100.5)

    def test_tpot_is_decode_total_over_g_minus_1(self):
        # OSL=32 -> G-1 = 31 decode steps of 10 ms each => Tdecode = 310 ms.
        s = request_summary([_request()])
        assert s["tpot_ms"]["p50"] == pytest.approx(310.0 / 31.0)

    def test_e2e_is_ttft_plus_decode(self):
        s = request_summary([_request()])
        # 100.5 + 310 = 410.5 ms.
        assert s["e2e_ms"]["p50"] == pytest.approx(410.5)

    def test_decode_tail_tps(self):
        # (G-1) / Tdecode = 31 / 0.31 s = 100 tok/s.
        s = request_summary([_request()])
        assert s["decode_tokens_per_s"]["p50"] == pytest.approx(100.0)

    def test_output_tps(self):
        # G / Tgen = 32 / 0.4105 s.
        s = request_summary([_request()])
        assert s["output_tokens_per_s"]["p50"] == pytest.approx(32 / 0.4105)

    def test_prefill_tps(self):
        # I / prefill = 128 / 0.1 s = 1280 tok/s.
        s = request_summary([_request()])
        assert s["prefill_tokens_per_s"]["p50"] == pytest.approx(1280.0)

    def test_p50_and_p95_across_requests(self):
        requests = [
            _request(prefill_ms=100.0),
            _request(prefill_ms=200.0),
            _request(prefill_ms=300.0),
        ]
        s = request_summary(requests)
        assert s["ttft_ms"]["count"] == 3
        assert s["ttft_ms"]["p50"] == pytest.approx(200.5)
        # Position (3-1)*0.95 = 1.9 -> interpolate between 200.5 and 300.5.
        assert s["ttft_ms"]["p95"] == pytest.approx(200.5 + 0.9 * 100.0)

    def test_pooled_itl_is_step_level(self):
        # Two requests, each 31 steps of 10 ms => 62 pooled steps.
        s = request_summary([_request(), _request()])
        assert s["pooled_itl_ms"]["count"] == 62
        assert s["pooled_itl_ms"]["p50_ms"] == pytest.approx(10.0)

    def test_g_eq_1_tpot_is_nan_and_dropped(self):
        s = request_summary([_request(itl=[], osl=1)])
        assert s["tpot_ms"]["count"] == 0
        assert math.isnan(s["tpot_ms"]["p50"])

    def test_empty_requests(self):
        s = request_summary([])
        assert s["ttft_ms"]["count"] == 0
        assert math.isnan(s["ttft_ms"]["p50"])
        assert s["pooled_itl_ms"] == {}
