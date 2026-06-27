"""Property-style invariants of the distributed layer (deterministic loops).

Each property holds for *any* input in a bounded domain and states the negative
case that would violate it, using deterministic seeds instead of a shrinking
engine so the suite runs on the CPU-minimal installation.
"""

from __future__ import annotations

import random

import pytest

from hqsb.distributed import collectives as co
from hqsb.distributed import faults as ft
from hqsb.distributed import ledger as lg
from hqsb.distributed import moe as mo
from hqsb.distributed import overlap as ov
from hqsb.distributed import parallel_plan as pp
from hqsb.distributed import sequence as sq
from hqsb.distributed import topology as tp


@pytest.mark.property
class TestBandwidthAlgebra:
    """busbw = algbw × correction, and the correction stays in (0, 2]."""

    def test_correction_and_bandwidth_for_any_size_and_world(self):
        rng = random.Random(20260919)
        for _ in range(50):
            op = rng.choice(co.OPS)
            world_size = rng.randint(1, 8)
            numel = rng.randint(1, 1 << 20)
            latency_us = rng.uniform(1.0, 10_000.0)
            correction = co.bus_correction(op, world_size)
            # the correction is 0 only for a degenerate world size of 1
            assert 0.0 <= correction <= 2.0
            if world_size >= 2:
                assert correction > 0.0
            row = co.bandwidth_row(
                op=op, world_size=world_size, numel_per_rank=numel, dtype="fp16",
                latency_us=latency_us, algorithm="auto", protocol="auto",
            )
            assert row["busbw_GBps"] == pytest.approx(row["algbw_GBps"] * correction)

    def test_a_zero_latency_is_refused(self):
        from hqsb.core.errors import ConfigError

        with pytest.raises(ConfigError):
            co.algbw_gbps(1024, 0.0)


@pytest.mark.property
class TestIntervalAlgebra:
    """union ≤ Σ durations; intersection ≤ min(side); merged intervals are disjoint."""

    @staticmethod
    def _interval(rng: random.Random) -> "ov.Interval":
        first = rng.randint(0, 1000)
        second = rng.randint(0, 1000)
        return ov.Interval(min(first, second), max(first, second))

    def test_union_and_intersection_bounds(self):
        rng = random.Random(7)
        for _ in range(40):
            left = [self._interval(rng) for _ in range(rng.randint(1, 6))]
            right = [self._interval(rng) for _ in range(rng.randint(1, 6))]
            union_left = ov.union_measure_ns(left)
            assert union_left <= sum(item.duration_ns for item in left)
            assert ov.intersection_measure_ns(left, right) <= min(
                union_left, ov.union_measure_ns(right)
            )
            merged = ov.merge_intervals(left)
            for earlier, later in zip(merged, merged[1:]):
                assert earlier.end_ns <= later.start_ns


@pytest.mark.property
class TestOverlapMetricsBounds:
    """exposed = comm − overlap ≥ 0 and overlap ≤ min(compute, comm)."""

    def test_metrics_hold_for_random_intervals(self):
        rng = random.Random(11)
        for _ in range(30):
            compute = [ov.Interval(0, rng.randint(1, 1000))]
            start = rng.randint(0, 1000)
            comm = [ov.Interval(start, start + rng.randint(1, 1000))]
            metrics = ov.overlap_metrics_from_intervals(compute, comm)
            assert metrics.exposed_comm_ms >= -1e-9
            assert metrics.overlap_ms <= min(metrics.compute_active_ms, metrics.comm_active_ms) + 1e-9
            if metrics.comm_active_ms:
                assert 0.0 <= metrics.overlap_fraction_comm <= 1.0


@pytest.mark.property
class TestShardRoundTrip:
    """Row/column shard split then merge reproduces the tensor exactly."""

    def test_round_trip_for_divisible_shapes(self):
        rng = random.Random(31337)
        checked = 0
        for _ in range(25):
            rows = rng.choice([1, 2, 4, 8])
            columns = rng.choice([4, 8, 16])
            shard_count = rng.choice([1, 2, 4])
            values = [rng.random() for _ in range(rows * columns)]
            for axis, extent in (("row", rows), ("column", columns)):
                if extent % shard_count:
                    continue  # a non-divisible axis is covered by the negative test below
                specs = pp.plan_axis_shards(
                    "w", (rows, columns), shard_axis=axis, shard_count=shard_count
                )
                shards = [pp.split_flat(values, (rows, columns), spec) for spec in specs]
                assert pp.merge_flat(shards, specs[0], (rows, columns)) == values
                checked += 1
        assert checked > 0

    def test_an_indivisible_axis_without_padding_is_refused(self):
        from hqsb.core.errors import ConfigError

        with pytest.raises(ConfigError):
            pp.plan_axis_shards("w", (5, 3), shard_axis="row", shard_count=2)


@pytest.mark.property
class TestMoeConservation:
    """tokens×top_k is conserved across dispatch → combine for every skew profile."""

    def test_conservation_for_every_profile_and_size(self):
        rng = random.Random(99)
        for profile in mo.SKEW_PROFILES:
            for _ in range(3):
                tokens = rng.randint(1, 32)
                world_size = rng.choice([2, 4])
                experts_per_rank = rng.choice([1, 2])
                top_k = rng.randint(1, 2)
                route = mo.generate_route_artifact(
                    artifact_id=f"p-{profile}", token_count=tokens, world_size=world_size,
                    experts_per_rank=experts_per_rank, top_k=top_k, profile=profile,
                    seed=rng.randint(0, 10_000),
                )
                placement = mo.round_robin_placement(
                    num_experts=world_size * experts_per_rank, world_size=world_size
                )
                matrix = mo.count_matrix(
                    route, expert_to_rank=placement.expert_to_rank, world_size=world_size
                )
                audit = matrix.audit()
                assert audit["ok"] is True
                assert sum(audit["send_tokens"]) == tokens * top_k

                dispatch = mo.dispatch_oracle(route)
                outputs = {
                    position: mo.deterministic_stub_expert(
                        position=position, expert_id=dispatch.expert_order[position],
                        hidden_size=4,
                    )
                    for position in range(dispatch.packed_count)
                }
                combined = mo.combine_oracle(dispatch, expert_outputs=outputs, hidden_size=4)
                assert combined["token_conservation"] is True
                assert len(combined["tokens"]) == tokens

    def test_imbalance_metrics_stay_in_range(self):
        rng = random.Random(5)
        for _ in range(20):
            counts = [rng.randint(0, 50) for _ in range(rng.randint(2, 8))]
            metrics = mo.imbalance_metrics(
                tokens_per_expert=counts, tokens_per_rank=[sum(counts)],
            )
            assert 0.0 <= metrics["gini"] < 1.0
            assert metrics["entropy"] >= 0.0


@pytest.mark.property
class TestLedgerAndMemoryConservation:
    """Ledger totals equal the sum of rows; memory buckets sum to the row total."""

    def test_memory_buckets_sum_to_total(self):
        rng = random.Random(17)
        for _ in range(20):
            row = lg.MemoryRow(
                rank=rng.randint(0, 3),
                weights=rng.randint(0, 1000),
                kv=rng.randint(0, 1000),
                activation=rng.randint(0, 1000),
                workspace=rng.randint(0, 1000),
                communicator_buffers=rng.randint(0, 1000),
                graph_pool=rng.randint(0, 1000),
                allocator_reserved=rng.randint(0, 1000),
                replicated_extra=rng.randint(0, 1000),
            )
            assert sum(row.buckets().values()) == row.total

    def test_empty_ledger_is_closed_and_conserved(self):
        result = lg.diff_ledger([], [])
        assert result.closed is True


@pytest.mark.property
class TestTimeoutClamp:
    """Derived timeouts always stay inside the pre-registered [min, max] band."""

    def test_clamp_for_any_healthy_p99(self):
        rng = random.Random(23)
        spec = sq.TimeoutSpec(
            init_timeout_s=30.0, collective_timeout_s=1.0, watchdog_heartbeat_timeout_s=1.0,
            job_kill_grace_s=5.0, post_cleanup_probe_timeout_s=5.0, multiplier=3.0,
            minimum_operational_timeout_s=0.5, maximum_timeout_s=60.0,
        )
        for _ in range(30):
            p99 = rng.uniform(0.001, 100.0)
            derived = spec.compute_collective_timeout(healthy_p99_s=p99)
            assert 0.5 <= derived <= 60.0

    def test_sequence_allocator_is_monotone(self):
        allocator = sq.SequenceAllocator("tp")
        previous = -1
        for _ in range(20):
            current = allocator.next()
            assert current == previous + 1
            previous = current


@pytest.mark.property
class TestTopologyAndFaultInvariants:
    """UNAVAILABLE never becomes a number; a fault verdict needs every rank."""

    def test_unavailable_round_trip(self):
        rng = random.Random(3)
        for _ in range(20):
            reason = f"reason-{rng.randint(0, 1000)}"
            marker = tp.unavailable(reason)
            assert tp.is_unavailable(marker)
            assert reason in marker

    def test_fault_verdict_requires_complete_rank_states(self):
        oracle = ft.oracle_for_fault(
            "op_mismatch", maximum_detection_time_s=5.0, maximum_global_abort_time_s=5.0
        )
        for world_size in (2, 4, 8):
            states = {rank: "ABORTED" for rank in range(world_size - 1)}
            observation = ft.FailureObservation(
                per_rank_terminal_state=states,
                detected_at_layer="init_timeout_or_preflight",
                normalized_error="hqsb.collective.mismatch",
                recovery_level_used="COMMUNICATOR_RECREATE",
            )
            verdict = ft.evaluate_fault(oracle, observation, world_size=world_size)
            assert verdict.ok is False
            assert any("terminal state incomplete" in failure for failure in verdict.failures)

    def test_preflight_passes_for_identical_semantics(self):
        base = {
            "run_id": "r", "rank_epoch": 0, "group_rank": 0, "group_id": "tp",
            "ordered_group_hash": "h", "collective_seq": 0, "op": "all_reduce",
            "reduce_op": "sum", "logical_count": 8, "dtype": "fp16",
            "input_shape_hash": "i", "output_shape_hash": "o", "stream_id": "s",
            "callsite": "layer.0",
        }
        for world_size in (2, 4):
            records = {
                rank: sq.CollectiveCallRecord(**{**base, "global_rank": rank, "group_rank": rank})
                for rank in range(world_size)
            }
            assert sq.preflight_check(records).ok is True
