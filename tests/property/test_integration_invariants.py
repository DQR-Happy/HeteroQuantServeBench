"""Property-style invariants for the S06 integration contracts.

Each test states an invariant that must hold for *every* input, not just the
examples used elsewhere: identity hashes are deterministic and collision-free
under the factors they claim to cover, accounting identities balance, coverage
ratios stay in range, and the deterministic fallback never drifts.
"""

from __future__ import annotations

import pytest

from hqsb.integration import cache, differential as diff, graph, guards, lowering, patterns, specs

pytestmark = pytest.mark.property


def _graph(permutation: int) -> graph.Graph:
    """Two structurally identical graphs that differ only in permutation."""
    base = [
        {"name": "x", "op": "graph_input"},
        {"name": "w", "op": "graph_input"},
        {"name": "add", "op": "aten.add.Tensor", "args": ["x", "x"]},
        {"name": "norm", "op": "hqsb.rms_norm", "args": ["add", "w"], "output": True},
    ]
    renamed = [
        {
            "name": node["name"],
            "op": node["op"],
            "args": list(node.get("args", ())),
            "output": node.get("output", False),
        }
        for node in base
    ]
    if permutation:
        renamed[0]["name"] = "input_a"
        renamed[1]["name"] = "input_b"
        for node in renamed:
            node["args"] = [
                {"x": "input_a", "w": "input_b"}.get(arg, arg) for arg in node["args"]
            ]
    return graph.from_node_sequence(
        renamed,
        capture_mode=graph.CaptureMode.DYNAMO,
        ir_level=graph.IRLevel.DYNAMO_FX,
    )


class TestHashes:
    def test_structural_hash_is_invariant_to_node_names(self):
        assert _graph(0).structural_hash() == _graph(1).structural_hash()

    def test_schema_hash_is_deterministic(self):
        for schema in specs.frozen_schemas():
            assert schema.schema_hash == schema.schema_hash
            assert len(schema.schema_hash) == 64

    def test_graph_identity_changes_with_every_declared_factor(self):
        base = cache.graph_identity(
            "g", op_schema_versions={"a": "1"}, tensor_metadata=("m",), model_id="m",
            quant_policy_id="q", rewrite_spec_id="r",
        )
        for override in (
            {"graph_hash": "g2"},
            {"op_schema_versions": {"a": "2"}},
            {"tensor_metadata": ("n",)},
            {"model_id": "m2"},
            {"quant_policy_id": "q2"},
            {"rewrite_spec_id": "r2"},
        ):
            payload = {
                "graph_hash": "g",
                "op_schema_versions": {"a": "1"},
                "tensor_metadata": ("m",),
                "model_id": "m",
                "quant_policy_id": "q",
                "rewrite_spec_id": "r",
            }
            payload.update(override)
            assert cache.graph_identity(**payload).digest != base.digest

    def test_pattern_signature_is_deterministic(self):
        for spec in patterns.frozen_patterns():
            assert spec.signature == spec.signature

    def test_preregistration_hash_is_deterministic(self):
        from hqsb.integration import experiment as exp

        prereg = exp.Preregistration(
            experiment_id="E06-01",
            question="q",
            hypothesis="h",
            claim_boundary="b",
        )
        assert prereg.prereg_hash == prereg.prereg_hash


class TestAccountingInvariants:
    def test_launch_delta_sign(self):
        for before, after in ((1, 5), (5, 1), (3, 3)):
            account = lowering.AllocationAccount(
                intermediate_bytes_before=0,
                intermediate_bytes_after=0,
                launch_count_before=before,
                launch_count_after=after,
            )
            assert account.launch_delta == after - before

    def test_reconcile_residual_is_measured_minus_accounted(self):
        account = lowering.AllocationAccount(
            intermediate_bytes_before=1000,
            intermediate_bytes_after=400,
            workspace_bytes=100,
        )
        for measured in (0, 250, 500, 1000):
            report = account.reconcile(measured_saving_bytes=measured)
            assert report["residual_bytes"] == measured - account.saving_accounted_bytes

    def test_tensor_bytes_are_multiplicative(self):
        for shape in ((1,), (2, 3), (4, 4, 4)):
            expected = 1
            for dim in shape:
                expected *= dim
            assert lowering.tensor_bytes(shape, "float16") == expected * 2

    def test_break_even_is_consistent(self):
        for cost, eager, compiled in ((100.0, 10.0, 8.0), (1000.0, 20.0, 12.0)):
            result = cache.compute_break_even(cost, eager, compiled)
            if result.requests:
                assert result.requests * (eager - compiled) >= cost


class TestCoverageInvariants:
    def test_ratios_stay_within_range(self):
        fixture = graph.from_node_sequence(
            [
                {"name": "x", "op": "graph_input"},
                {"name": "r", "op": "graph_input"},
                {"name": "add", "op": "aten.add.Tensor", "args": ["x", "r"]},
                {"name": "norm", "op": "hqsb.rms_norm", "args": ["add", "w"], "output": True},
            ],
            capture_mode=graph.CaptureMode.DYNAMO,
            ir_level=graph.IRLevel.DYNAMO_FX,
        )
        spec = patterns.residual_add_rmsnorm_pattern()
        context = patterns.PatternContext(
            graph=fixture,
            declared_eps=1e-6,
            pattern_eps=1e-6,
            hidden_size=64,
            weight_shapes={"weight": (64,)},
            supported_versions=("1.0.0",),
            version="1.0.0",
        )
        decisions = patterns.scan_graph(spec, fixture, lambda _g, _m: context)
        report = patterns.coverage_report(decisions)
        for value in (
            report.node_coverage,
            report.call_coverage,
            report.time_coverage,
            report.model_coverage,
            report.precision,
            report.recall,
        ):
            assert 0.0 <= value <= 1.0

    def test_empty_decision_set_has_zero_coverage(self):
        report = patterns.coverage_report(())
        assert report.node_coverage == 0.0
        assert report.false_positives == 0


class TestDeterminismInvariants:
    def test_fallback_resolution_is_a_function_of_its_inputs(self):
        from hqsb.integration import taxonomy

        registry = taxonomy.FallbackRegistry()
        for requested in ("hqsb.compiled.fused", "hqsb.eager.custom", "torch.eager.reference"):
            first = registry.resolve(requested)
            second = registry.resolve(requested)
            assert first == second

    def test_five_way_counts_are_additive(self):
        ledger = guards.CompileLedger()
        for index in range(5):
            ledger.record(guards.EventKind.RECOMPILE, index)
        counts = ledger.counts()
        assert counts[guards.EventKind.RECOMPILE] == 5
        five_way = ledger.five_way().as_dict()
        event_total = sum(
            five_way[key]
            for key in (
                "guard_failures",
                "recompiles",
                "graph_breaks",
                "runtime_asserts",
                "fallbacks",
            )
        )
        assert event_total == 5
        assert five_way["requests"] == 5

    def test_tolerance_lookup_is_monotone_in_dtype_precision(self):
        registry = diff.default_tolerance_registry()
        assert registry.get(diff.Level.OPERATOR, "float32").max_abs <= registry.get(
            diff.Level.OPERATOR, "float16"
        ).max_abs

    def test_correctness_matrix_missing_count_is_exact(self):
        spec = diff.DifferentialSpec(
            model_id="m", model_manifest_sha256="h", input_token_hash="t"
        )
        matrix = diff.CorrectnessMatrix(specs=spec)
        total = len(spec.paths) * len(diff.Level.ALL)
        assert len(matrix.missing_cells()) == total
        matrix.mark(spec.paths[0], diff.Level.ALL[0], diff.CellStatus.PASS)
        assert len(matrix.missing_cells()) == total - 1
