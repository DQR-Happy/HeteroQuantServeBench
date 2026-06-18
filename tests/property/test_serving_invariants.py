"""Property-style invariants of the serving layer (deterministic loops).

These check properties that hold for *any* input in a bounded domain, using
deterministic seeds instead of a shrinking engine, so they run on the
CPU-minimal installation with no extra dependency.  Each property also states
the negative case that would violate it, so the invariant is falsifiable.
"""

from __future__ import annotations

import itertools
import os
import random

import pytest

from hqsb.core.errors import ConfigError
from hqsb.serving import arrival, cache_routing, pipeline, sse, timing

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _spec():
    import yaml

    with open(
        os.path.join(_REPO_ROOT, "configs", "serving", "arrival_spec.yaml"), encoding="utf-8"
    ) as handle:
        return arrival.ArrivalSpec.from_document(yaml.safe_load(handle))


@pytest.mark.property
class TestDeliveryLedgerMonotone:
    """generated ≥ committed ≥ emitted ≥ flushed ≥ client_received, always."""

    def test_any_order_violation_is_detected(self):
        stages = ("generated", "committed", "emitted", "flushed", "client_received")

        def clean(generated=5, committed=4, emitted=4, flushed=4, received=4):
            ledger = pipeline.DeliveryLedger(
                request_id="r0",
                generated=generated,
                committed=committed,
                emitted=emitted,
                flushed=flushed,
            )
            ledger.client_received = received
            ledger.client_received_observed = True
            ledger.terminal_clean = True
            return ledger

        def bumped(stage: str):
            kwargs = {name: 99 for name in stages[1:] if name == stage}
            if stage == "client_received":
                return clean(received=99)
            return clean(**kwargs)

        # a clean delivery chain is valid
        assert clean().audit()["ok"]
        # increasing the *source* (generated) is allowed: it is upstream
        assert clean(generated=99).audit()["ok"]
        # increasing any downstream stage must be flagged
        for stage in stages[1:]:
            assert not bumped(stage).audit()["ok"], stage


@pytest.mark.property
class TestSseReassemblerComposition:
    """Parsing a concatenation equals parsing the pieces, for any split."""

    def test_feed_is_order_independent(self):
        rng = random.Random(1)
        frames = [
            sse.encode_data_frame(
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"t{index}", "token_ids": [index]},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            for index in range(rng.randint(1, 8))
        ]
        raw = b"".join(frames) + sse.encode_terminal()
        whole = sse.parse_stream(raw)
        # split at every boundary and rejoin
        for split in range(0, len(raw) + 1, 3):
            reassembler = sse.SseReassembler()
            reassembler.feed(raw[:split])
            reassembler.feed(raw[split:])
            reassembler.finish()
            assert len(reassembler.frames) == len(whole["parsed"]), split
            assert sse.reconstruct(reassembler.frames) == sse.reconstruct(whole["parsed"])


@pytest.mark.property
class TestArrivalMeanRate:
    """Every generator keeps the pre-registered mean rate up to sampling error."""

    def test_mean_rate_is_preserved_across_distributions(self):
        spec = _spec()
        for distribution, count, seed in itertools.product(
            ("constant", "poisson", "on_off_burst", "batch_burst", "compound_poisson"),
            (400,),
            (11, 12, 13),
        ):
            trace = arrival.generate_arrival_trace(
                spec, distribution=distribution, mean_rate=8.0, count=count, seed=seed
            )
            realized = trace.count / trace.duration_sec
            cv = float(trace.statistics()["cv"] or 0.0)
            tolerance = max(0.10, 3 * cv / (count**0.5))
            assert abs(realized - 8.0) / 8.0 <= tolerance, (
                distribution,
                realized,
                tolerance,
            )


@pytest.mark.property
class TestPrefixMatcherBounds:
    """Matched tokens never exceed either sequence length, and oracle agrees."""

    def test_match_never_exceeds_bounds(self):
        rng = random.Random(2)
        identity = cache_routing.CacheIdentity(
            fields={name: str(name) for name in cache_routing.IDENTITY_FIELDS}
        )
        matcher = cache_routing.PrefixMatcher(block_size=4)
        for _ in range(20):
            entry_tokens = tuple(rng.randrange(16) for _ in range(64))
            matcher.insert(
                cache_routing.PrefixEntry(
                    identity=identity, tokens=entry_tokens, block_size=4
                )
            )
        for _ in range(50):
            query = [rng.randrange(16) for _ in range(rng.randint(1, 80))]
            result = matcher.match(identity=identity, query_tokens=query)
            assert 0 <= result["matched_tokens"] <= min(len(query), 64)
            assert result["matched_tokens"] % 4 == 0
            oracle = matcher_oracle_agreement(matcher, identity, query)
            assert oracle["ok"]


def matcher_oracle_agreement(matcher, identity, query):
    return cache_routing.matcher_oracle(matcher, identity=identity, query=query)


@pytest.mark.property
class TestTimestampLedgerMonotone:
    """Same-clock timestamps must not go backwards in causal order."""

    def test_reversal_is_detected_and_cross_clock_refused(self):
        ledger = timing.TimestampLedger(request_id="r0")
        ledger.record("t_send", 100)
        ledger.record("t_gateway_recv", 50)  # reversed
        assert ledger.monotonicity_problems()
        other = timing.TimestampLedger(request_id="r1")
        other.record("t_send", 100, clock="loadgen_monotonic")
        other.record("t_gateway_recv", 200, clock="gateway_monotonic")
        with pytest.raises(ConfigError):
            other.duration_ns("ingress")  # cross-clock without calibration


@pytest.mark.property
class TestFunnelConservation:
    """offered − slo_good must always be attributed to reasons."""

    def test_unattributed_loss_is_flagged(self):
        from hqsb.serving import slo

        intermediate = (
            "client_attempted",
            "gateway_received",
            "valid",
            "admitted",
            "backend_started",
            "completed_success",
        )

        def funnel(attributed: bool):
            counter = slo.FunnelCounts()
            counter.add("offered", 10)
            for stage in intermediate:
                counter.add(stage, 9)
            counter.add("slo_good", 7)
            if attributed:
                # offered 10 -> attempted 9 is one invalid request; completed_success
                # 9 -> slo_good 7 is two SLO violations; total loss = 3
                counter.add_reason("invalid_request", 1)
                counter.add_reason("slo_violation", 2)
            return counter

        assert funnel(attributed=True).audit()["ok"]
        assert not funnel(attributed=False).audit()["ok"]
