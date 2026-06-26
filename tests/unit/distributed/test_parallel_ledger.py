"""E10-04 interface tests: census, TP capability, shards, plan, ledger, memory."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.distributed import ledger as lg
from hqsb.distributed import parallel_plan as pp


def _census(**overrides) -> pp.QwenArchitectureCensus:
    payload = {
        "family": "qwen3",
        "revision": "fixture",
        "num_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 8,
        "num_kv_heads": 4,
        "head_dim": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
    }
    payload.update(overrides)
    return pp.QwenArchitectureCensus(**payload)


def _plan(census: pp.QwenArchitectureCensus, degree: int = 2) -> pp.ParallelPlan:
    return pp.derive_plan(
        census,
        plan_id=f"plan-{degree}",
        model_manifest_sha256="fixture-model",
        degree=degree,
        ordered_ranks=list(range(degree)),
    )


@pytest.mark.unit
class TestCensus:
    def test_census_from_mapping_and_parameter_map(self):
        census = pp.census_from_mapping(
            {
                "num_hidden_layers": 2,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "head_dim": 8,
                "vocab_size": 128,
            }
        )
        shapes = census.parameter_shapes()
        assert shapes["model.layers.1.self_attn.k_proj.weight"] == (32, 64)
        assert shapes["model.layers.0.mlp.up_proj.weight"] == (128, 64)
        assert census.head_groups == 2

    def test_census_rejects_missing_fields(self):
        with pytest.raises(ConfigError):
            pp.census_from_mapping({"num_hidden_layers": 2})

    def test_gqa_grouping_is_validated(self):
        with pytest.raises(ConfigError):
            _census(num_kv_heads=3)


@pytest.mark.unit
class TestTpCapability:
    def test_supported_and_unsupported_degrees(self):
        rows = {row.degree: row for row in pp.tp_capability_matrix(_census(), [1, 2, 3], available_devices=4)}
        assert rows[1].supported and rows[2].supported
        assert rows[2].kv_strategy == "SPLIT_EVEN"
        assert rows[3].supported is False
        assert any("hidden_size" in reason for reason in rows[3].reasons)

    def test_replication_when_kv_heads_below_degree(self):
        census = _census(num_kv_heads=1, num_attention_heads=8)
        row = pp.tp_capability_matrix(census, [4], available_devices=4)[0]
        assert row.kv_strategy == "REPLICATE"
        assert any("does not shrink" in reason for reason in row.reasons)

    def test_missing_devices_block_the_degree(self):
        row = pp.tp_capability_matrix(_census(), [4], available_devices=2)[0]
        assert row.supported is False
        assert any("only 2 device" in reason for reason in row.reasons)


@pytest.mark.unit
class TestShardsAndPlan:
    def test_plan_covers_every_parameter(self):
        census = _census()
        plan = _plan(census)
        assert plan.validate(census) == []
        covered = {spec.param_name for spec in plan.parameter_shards}
        assert covered == set(census.parameter_shapes())

    def test_shard_ranges_cover_axis_without_overlap(self):
        specs = pp.plan_axis_shards("w", (6, 4), shard_axis="row", shard_count=2)
        audit = pp.verify_shard_ranges(specs, (6, 4))
        assert audit["ok"] is True
        with pytest.raises(ConfigError):
            pp.plan_axis_shards("w", (7, 4), shard_axis="row", shard_count=2)

    def test_split_merge_round_trip(self):
        values = [float(index) for index in range(12)]
        specs = pp.plan_axis_shards("w", (3, 4), shard_axis="row", shard_count=3)
        shards = [pp.split_flat(values, (3, 4), spec) for spec in specs]
        assert pp.merge_flat(shards, specs[0], (3, 4)) == values
        column_specs = pp.plan_axis_shards("w", (3, 4), shard_axis="column", shard_count=2)
        column_shards = [pp.split_flat(values, (3, 4), spec) for spec in column_specs]
        assert pp.merge_flat(column_shards, column_specs[0], (3, 4)) == values

    def test_plan_rejects_a_fallback_that_changes_semantics(self):
        census = _census()
        plan = _plan(census)
        broken = pp.ParallelPlan(
            plan_id="x",
            model_manifest_sha256="m",
            tp_degree=plan.tp_degree,
            ordered_ranks=plan.ordered_ranks,
            placement_hash="",
            parameter_shards=plan.parameter_shards,
            unsupported_predicate="",
            fallback_policy="copy",
        )
        errors = broken.validate(census)
        assert any("reject" in error for error in errors)
        assert any("unsupported-shape predicate" in error for error in errors)

    def test_attention_and_mlp_shards(self):
        census = _census()
        attention = pp.derive_attention_shard(census, degree=2, rank=0)
        assert attention.q_range == (0, 4) and attention.kv_range == (0, 2)
        mlp = pp.derive_mlp_shard(census, degree=2, rank=1)
        assert mlp.gate_range == (64, 128)

    def test_attention_shard_rejects_impossible_degree(self):
        census = _census(num_kv_heads=3, num_attention_heads=9)
        with pytest.raises(ConfigError):
            pp.derive_attention_shard(census, degree=2, rank=0)

    def test_kv_ownership_reports_replication(self):
        census = _census(num_kv_heads=1, num_attention_heads=8)
        ownership = pp.kv_ownership(census, degree=2, free_memory_bytes=1 << 30)
        assert ownership.replicated_kv_heads == 1
        assert ownership.per_token_bytes > 0
        assert ownership.predicted_capacity_tokens > 0


@pytest.mark.unit
class TestDirectLoadAndCases:
    def test_direct_load_audit_rejects_full_then_split(self):
        plan = pp.LoadPlan(mode="full_then_split", per_rank_peak_bytes=(1000, 1000))
        audit = pp.audit_direct_load(plan, full_model_bytes=1000)
        assert audit["ok"] is False
        good = pp.LoadPlan(mode="direct_shard_load", per_rank_peak_bytes=(500, 500))
        assert pp.audit_direct_load(good, full_model_bytes=1000)["ok"] is True

    def test_column_and_row_parallel_cases_match_the_reference(self):
        weight = [float(index) for index in range(12)]
        x = [1.0, 2.0, 3.0, 4.0]
        column = pp.column_parallel_case(weight, out_features=3, in_features=4, degree=3, x=x)
        assert column["max_abs"] == pytest.approx(0.0, abs=1e-9)
        row = pp.row_parallel_case(weight, out_features=3, in_features=4, degree=2, x=x)
        assert row["max_abs"] == pytest.approx(0.0, abs=1e-9)
        assert row["collective"] == "all_reduce"

    def test_attention_block_case_covers_every_head(self):
        result = pp.attention_block_case(hidden=64, num_heads=8, head_dim=8, degree=4)
        assert result["complete"] is True

    def test_mlp_block_case_completes(self):
        result = pp.mlp_block_case(
            hidden=2,
            intermediate=2,
            degree=2,
            x=[1.0, 1.0],
            gate=[1.0, 1.0, 1.0, 1.0],
            up=[1.0, 1.0, 1.0, 1.0],
            down=[1.0, 1.0, 1.0, 1.0],
        )
        assert len(result["merged"]) == 2

    def test_layer_case_locates_the_first_failing_layer(self):
        report = pp.layer_case({0: 0.0, 1: 0.5, 2: 0.6}, tolerance=0.1)
        assert report["first_failing_layer"] == 1

    def test_unsupported_policy_never_truncates(self):
        decision = pp.unsupported_policy(predicate="heads % degree", policy="UNEVEN", reason="tail heads")
        assert decision.policy == "UNEVEN"
        with pytest.raises(ConfigError):
            pp.assert_no_truncation(global_extent=8, covered_extent=4, predicate="heads")
        assert pp.assert_no_truncation(global_extent=6, covered_extent=8, predicate="heads")["ok"]


@pytest.mark.unit
class TestLedger:
    def _events(self):
        census = _census()
        plan = _plan(census)
        expected = lg.expected_ledger(plan, census, phase="decode", token_rows=1)
        observed = [
            lg.ObservedEvent(
                rank=0,
                group_id="tp",
                collective_seq=index,
                op=event.collective,
                tensor_role=event.tensor_role,
                phase=event.phase,
                layer=event.layer,
                payload_bytes=event.payload_bytes,
                dtype=event.dtype,
            )
            for index, event in enumerate(expected)
        ]
        return expected, observed

    def test_matching_ledger_is_closed(self):
        expected, observed = self._events()
        result = lg.diff_ledger(expected, observed)
        assert result.closed is True
        assert result.conservation_ok is True

    def test_unexplained_difference_keeps_the_ledger_open(self):
        expected, observed = self._events()
        shrunk = list(observed)
        shrunk[0] = lg.ObservedEvent(
            rank=0,
            group_id="tp",
            collective_seq=0,
            op=observed[0].op,
            tensor_role=observed[0].tensor_role,
            phase=observed[0].phase,
            layer=observed[0].layer,
            payload_bytes=observed[0].payload_bytes // 2,
            dtype=observed[0].dtype,
        )
        result = lg.diff_ledger(expected, shrunk)
        assert result.closed is False
        gate = lg.ledger_gate(result)
        assert gate["ready"] is False
        assert gate["unexplained_rows"]

    def test_explained_difference_closes_the_ledger(self):
        expected, observed = self._events()
        shrunk = list(observed)
        shrunk[0] = lg.ObservedEvent(
            rank=0,
            group_id="tp",
            collective_seq=0,
            op=observed[0].op,
            tensor_role=observed[0].tensor_role,
            phase=observed[0].phase,
            layer=observed[0].layer,
            payload_bytes=observed[0].payload_bytes // 2,
            dtype=observed[0].dtype,
        )
        key = (expected[0].phase, expected[0].layer, expected[0].collective)
        result = lg.diff_ledger(
            expected, shrunk, explanations={key: ("fusion", "bias/norm fused into the projection")}
        )
        assert result.closed is True
        assert "fusion" in result.reason_codes

    def test_a_duplicated_observed_event_keeps_the_ledger_open(self):
        expected, observed = self._events()
        result = lg.diff_ledger(expected, list(observed) + [observed[0]])
        # one expected call vs two observed calls: the row no longer matches and
        # has no reason code, so the ledger stays open even though the total
        # byte count merely changed.
        assert result.closed is False
        assert lg.ledger_gate(result)["ready"] is False

    def test_memory_reconciliation_and_skew(self):
        row = lg.MemoryRow(rank=0, weights=100, kv=10, activation=5, workspace=2, communicator_buffers=2)
        report = lg.reconcile_memory(
            {"weights": 100, "kv": 10, "activation": 5, "workspace": 2, "communicator_buffers": 2},
            row,
        )
        assert report["ok"] is True
        skewed = lg.rank_skew(
            [row, lg.MemoryRow(rank=1, weights=200, kv=10, activation=5, workspace=2, communicator_buffers=2)]
        )
        assert skewed["max_rank"] == 1

    def test_reason_catalogue_is_documented(self):
        reasons = {row["reason_code"] for row in lg.why_observed_can_differ()}
        assert set(lg.LEDGER_REASON_CODES) == reasons
