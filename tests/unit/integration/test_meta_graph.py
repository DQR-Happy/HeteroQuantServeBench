"""Meta/FakeTensor metadata and graph IR tests (E06-02 interface layer)."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError, SchemaError
from hqsb.integration import graph, meta


def _rms_norm_inputs() -> tuple:
    symbol = meta.SymInt.symbol("S")
    x = meta.TensorMeta(shape=(symbol, 64), dtype="float16", device="cuda", stride=(64, 1))
    weight = meta.TensorMeta(shape=(64,), dtype="float16", device="cuda", stride=(1,))
    return x, weight


@pytest.mark.unit
class TestSymbolicIntegers:
    def test_symbol_survives_arithmetic(self):
        s = meta.SymInt.symbol("S")
        expression = s * 2 + 4
        assert expression.symbols() == ("S",)
        assert expression.evaluate({"S": 8}) == 20
        assert not expression.is_concrete()

    def test_concrete_expression_evaluates_without_bindings(self):
        s = meta.SymInt.symbol("S")
        assert (s * 3).evaluate({"S": 0}) == 0
        assert meta.SymInt("add", (2, 3)).concrete_value() == 5

    def test_symbolic_expression_refuses_concrete_value(self):
        with pytest.raises(SchemaError):
            meta.SymInt.symbol("S").concrete_value()

    def test_bad_symbol_name_refused(self):
        with pytest.raises(SchemaError):
            meta.SymInt.symbol("bad name")

    def test_canonicalisation_is_order_insensitive(self):
        s = meta.SymInt.symbol("S")
        assert (s + 4).canonical() == (4 + s).canonical()


@pytest.mark.unit
class TestTensorMeta:
    def test_stride_rank_must_match_shape(self):
        with pytest.raises(SchemaError):
            meta.TensorMeta(shape=(2, 3), dtype="float16", device="cuda", stride=(3,))

    def test_contiguity_detection(self):
        contiguous = meta.TensorMeta(shape=(2, 3), dtype="float16", device="cuda", stride=(3, 1))
        transposed = meta.TensorMeta(shape=(3, 2), dtype="float16", device="cuda", stride=(1, 3))
        assert contiguous.is_contiguous
        assert not transposed.is_contiguous


@pytest.mark.unit
class TestMetadataContract:
    def test_infer_outputs_preserves_prefix_symbol(self):
        contract = meta.rms_norm_contract()
        x, weight = _rms_norm_inputs()
        outputs = contract.infer_outputs([x, weight], a_last=64, b_last=64, rank=2, weight_rank=1)
        assert meta.dims_repr(outputs[0].shape) == "[S, 64]"
        assert outputs[0].symbols() == ("S",)
        assert outputs[0].dtype == "float16"

    def test_output_does_not_alias_the_input(self):
        contract = meta.rms_norm_contract()
        outputs = contract.infer_outputs(
            list(_rms_norm_inputs()), a_last=64, b_last=64, rank=2, weight_rank=1
        )
        assert outputs[0].alias_of_input is None

    def test_fused_contract_returns_two_fresh_outputs(self):
        contract = meta.fused_add_rms_norm_contract()
        symbol = meta.SymInt.symbol("S")
        x = meta.TensorMeta(shape=(symbol, 64), dtype="float16", device="cuda", stride=(64, 1))
        residual = meta.TensorMeta(
            shape=(symbol, 64), dtype="float16", device="cuda", stride=(64, 1)
        )
        weight = meta.TensorMeta(shape=(64,), dtype="float16", device="cuda", stride=(1,))
        outputs = contract.infer_outputs(
            [x, residual, weight],
            a_last=64,
            b_last=64,
            rank=2,
            weight_rank=1,
            hidden=64,
            weight_len=64,
            weight_dtype="float16",
        )
        assert len(outputs) == 2
        assert all(item.alias_of_input is None for item in outputs)

    def test_mismatched_last_dim_is_refused(self):
        contract = meta.rms_norm_contract()
        x, weight = _rms_norm_inputs()
        with pytest.raises(meta.MetadataError) as excinfo:
            contract.infer_outputs([x, weight], a_last=64, b_last=32, rank=2, weight_rank=1)
        assert excinfo.value.reason == "SHAPE"
        assert excinfo.value.details["pre_allocation"] is True

    def test_rank_zero_input_is_refused(self):
        contract = meta.rms_norm_contract()
        scalar = meta.TensorMeta(shape=(), dtype="float16", device="cuda", stride=())
        weight = meta.TensorMeta(shape=(64,), dtype="float16", device="cuda", stride=(1,))
        with pytest.raises(meta.MetadataError) as excinfo:
            contract.infer_outputs([scalar, weight], a_last=0, b_last=64, rank=0, weight_rank=1)
        assert excinfo.value.reason == "RANK"

    def test_unsupported_dtype_is_refused_with_the_field(self):
        contract = meta.rms_norm_contract()
        x = meta.TensorMeta(shape=(1, 64), dtype="complex64", device="cuda", stride=(64, 1))
        weight = meta.TensorMeta(shape=(64,), dtype="complex64", device="cuda", stride=(1,))
        with pytest.raises(meta.MetadataError) as excinfo:
            contract.infer_outputs([x, weight], a_last=64, b_last=64, rank=2, weight_rank=1)
        assert excinfo.value.reason == "DTYPE"
        assert "inputs[0].dtype" in excinfo.value.field_name

    def test_dequant_linear_replaces_the_last_dim_with_logical_n(self):
        contract = meta.dequant_linear_contract()
        symbol = meta.SymInt.symbol("S")
        x = meta.TensorMeta(shape=(symbol, 96), dtype="float16", device="cuda", stride=(96, 1))
        packed = meta.TensorMeta(shape=(7, 24), dtype="uint8", device="cuda", stride=(24, 1))
        scales = meta.TensorMeta(shape=(7, 1), dtype="float16", device="cuda", stride=(1, 1))
        outputs = contract.infer_outputs(
            [x, packed, scales],
            a_last=96,
            b_last=96,
            k=96,
            group_size=48,
            tail_policy_allowed=True,
        )
        assert meta.dims_repr(outputs[0].shape) == "[S, N]"
        assert outputs[0].stride == (1,) or outputs[0].is_contiguous

    def test_explicit_stride_policy_rejects_non_contiguous(self):
        contract = meta.MetadataContract(
            op="hqsb::test",
            input_rule="x",
            outputs=(
                meta.OutputRule(name="out", shape_rule="same_as_input", stride_policy="reject"),
            ),
        )
        transposed = meta.TensorMeta(
            shape=(3, 2), dtype="float16", device="cuda", stride=(1, 3)
        )
        with pytest.raises(meta.MetadataError) as excinfo:
            contract.infer_outputs([transposed])
        assert excinfo.value.reason == "STRIDE_LAYOUT"

    def test_unknown_output_shape_rule_refused(self):
        contract = meta.MetadataContract(
            op="hqsb::test",
            input_rule="x",
            outputs=(meta.OutputRule(name="out", shape_rule="magic"),),
        )
        x = meta.TensorMeta(shape=(2, 2), dtype="float16", device="cuda", stride=(2, 1))
        with pytest.raises(SchemaError):
            contract.infer_outputs([x])

    def test_zero_size_dimension_policy(self):
        strict = meta.MetadataContract(
            op="hqsb::test",
            input_rule="x",
            outputs=(meta.OutputRule(name="out", shape_rule="same_as_input"),),
            allow_zero_size_dims=False,
        )
        empty = meta.TensorMeta(shape=(0, 4), dtype="float16", device="cuda", stride=(4, 1))
        with pytest.raises(meta.MetadataError):
            strict.infer_outputs([empty])


@pytest.mark.unit
class TestMetadataOracle:
    def test_compare_reports_each_mismatching_field(self):
        contract = meta.rms_norm_contract()
        real = contract.infer_outputs(
            list(_rms_norm_inputs()), a_last=64, b_last=64, rank=2, weight_rank=1
        )[0]
        fake = meta.TensorMeta(
            shape=real.shape,
            dtype="float32",
            device="cuda",
            stride=(128, 1),
        )
        diff = meta.compare_metadata(contract.op, real, fake)
        assert not diff.ok
        fields = {item["field"] for item in diff.mismatches}
        assert {"dtype", "stride"} <= fields

    def test_compare_is_ok_for_identical_metadata(self):
        contract = meta.rms_norm_contract()
        real = contract.infer_outputs(
            list(_rms_norm_inputs()), a_last=64, b_last=64, rank=2, weight_rank=1
        )[0]
        assert meta.compare_metadata(contract.op, real, real).ok


@pytest.mark.unit
class TestFakeCallSpy:
    def test_spy_flags_any_real_work(self):
        spy = meta.FakeCallSpy(op="hqsb::rms_norm")
        assert spy.clean
        spy.record_allocation(1024)
        assert not spy.clean
        assert spy.as_dict()["allocations_bytes"] == 1024

    def test_spy_flags_payload_reads(self):
        spy = meta.FakeCallSpy(op="hqsb::dequant_linear")
        spy.record_payload_read(4096)
        assert not spy.clean


@pytest.mark.unit
class TestGuardSet:
    def test_redundant_and_data_dependent_guards_are_visible(self):
        guards = meta.GuardSet()
        guard = meta.MetadataGuard(kind="dtype", expression="x.dtype == float16", source="fake")
        guards.add(guard)
        guards.add(guard)
        guards.add(
            meta.MetadataGuard(
                kind="data_dependent",
                expression="x.max() < 10",
                source="export",
                necessary=True,
            )
        )
        payload = guards.as_dict()
        assert payload["count"] == 3
        assert payload["data_dependent"] == 1
        assert payload["redundant"] == 2


@pytest.mark.unit
class TestGraphIR:
    def _fixture(self) -> graph.Graph:
        return graph.from_node_sequence(
            [
                {"name": "x", "op": "graph_input"},
                {"name": "w", "op": "graph_input"},
                {"name": "add", "op": "aten.add.Tensor", "args": ["x", "x"]},
                {"name": "norm", "op": "hqsb.rms_norm", "args": ["add", "w"], "output": True},
            ],
            capture_mode=graph.CaptureMode.DYNAMO,
            ir_level=graph.IRLevel.DYNAMO_FX,
        )

    def test_structural_hash_ignores_node_names(self):
        first = self._fixture()
        renamed = graph.from_node_sequence(
            [
                {"name": "a", "op": "graph_input"},
                {"name": "b", "op": "graph_input"},
                {"name": "c", "op": "aten.add.Tensor", "args": ["a", "a"]},
                {"name": "d", "op": "hqsb.rms_norm", "args": ["c", "b"], "output": True},
            ],
            capture_mode=graph.CaptureMode.DYNAMO,
            ir_level=graph.IRLevel.DYNAMO_FX,
        )
        assert first.structural_hash() == renamed.structural_hash()

    def test_structural_hash_changes_with_topology(self):
        first = self._fixture()
        different = graph.from_node_sequence(
            [
                {"name": "x", "op": "graph_input"},
                {"name": "w", "op": "graph_input"},
                {"name": "add", "op": "aten.add.Tensor", "args": ["x", "w"]},
                {"name": "norm", "op": "hqsb.rms_norm", "args": ["add", "w"], "output": True},
            ],
            capture_mode=graph.CaptureMode.DYNAMO,
            ir_level=graph.IRLevel.DYNAMO_FX,
        )
        assert first.structural_hash() != different.structural_hash()

    def test_unknown_capture_mode_is_refused(self):
        with pytest.raises(ConfigError):
            graph.Graph(capture_mode="vibes")

    def test_unknown_ir_level_is_refused(self):
        with pytest.raises(ConfigError):
            graph.Graph(ir_level="somewhere_between")

    def test_users_and_dead_nodes(self):
        fixture = self._fixture()
        assert fixture.user_count("add") == 1
        assert fixture.users_of("add")[0].name == "norm"
        assert fixture.dead_nodes() == ()

    def test_diff_reports_added_and_removed_nodes(self):
        before = self._fixture()
        after = before.with_node(
            graph.GraphNode(name="fused", op="hqsb.fused_add_rms_norm", args=before.node("add").args)
        ).without_node("add").without_node("norm")
        diff = before.diff(after)
        assert diff.removed
        assert diff.added
        assert diff.node_delta == -1

    def test_impure_nodes_are_flagged(self):
        fixture = graph.from_node_sequence(
            [
                {"name": "x", "op": "graph_input"},
                {"name": "print", "op": "aten._print", "args": ["x"], "is_impure": True},
            ],
            capture_mode=graph.CaptureMode.DYNAMO,
            ir_level=graph.IRLevel.DYNAMO_FX,
        )
        assert [node.name for node in fixture.impure_nodes()] == ["print"]

    def test_unknown_node_reference_refused(self):
        # ``$name`` is an explicit reference; a typo must not become a constant.
        with pytest.raises(ConfigError):
            graph.from_node_sequence([{"name": "a", "op": "op", "args": ["$not-a-node"]}])

    def test_plain_string_argument_stays_a_constant(self):
        fixture = graph.from_node_sequence(
            [{"name": "a", "op": "op", "args": ["weight", 1e-6]}],
            capture_mode=graph.CaptureMode.DYNAMO,
            ir_level=graph.IRLevel.DYNAMO_FX,
        )
        assert fixture.node("a").args == ("weight", 1e-6)

    def test_graph_summary_aggregates_ops(self):
        summary = graph.graph_summary([self._fixture(), self._fixture()])
        assert summary["graphs"] == 2
        assert summary["op_totals"]["aten.add.Tensor"] == 2
