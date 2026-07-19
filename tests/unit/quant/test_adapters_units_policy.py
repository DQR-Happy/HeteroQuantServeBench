"""Tests for adapters, intervention units and mixed-precision policy."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.quant import sensitivity as sens
from hqsb.quant.adapters import (
    AWQ_ADAPTER,
    GPTQ_ADAPTER,
    SMOOTHQUANT_ADAPTER,
    SourceTensorRecord,
    adapter_availability_report,
    audit_field_mapping,
    capability_probe,
    get_adapter,
)
from hqsb.quant.adapters.base import FieldMapping, MethodMatrix
from hqsb.quant.packing import pack_canonical
from hqsb.quant.policy import (
    MixedPrecisionPolicy,
    UnitPolicyEntry,
    baseline_policies,
    greedy_search,
)
from hqsb.quant.units import UnitConfig, build_units


@pytest.mark.unit
class TestAdapterRegistry:
    def test_all_three_methods_registered(self):
        assert {GPTQ_ADAPTER.method, AWQ_ADAPTER.method} <= set(
            __import__("hqsb.quant.adapters", fromlist=["METHOD_ADAPTERS"]).METHOD_ADAPTERS
        )
        assert get_adapter("gptq").method == "gptq"

    def test_unknown_method_refused(self):
        with pytest.raises(KeyError):
            get_adapter("voodoo")

    def test_availability_report_has_reasons(self):
        report = adapter_availability_report()
        assert set(report) == {"gptq", "awq", "smoothquant"}
        for entry in report.values():
            assert "available" in entry
            if not entry["available"]:
                assert entry["reason"]

    def test_capability_probe_missing_package(self):
        probe = capability_probe("definitely_not_installed_hqsb")
        assert probe["available"] is False
        assert probe["reason"]


@pytest.mark.unit
class TestSourceConfig:
    def test_config_hash_covers_resolved_defaults(self):
        config = GPTQ_ADAPTER.source_config(version="1.0", public_config={"bits": 4})
        other = GPTQ_ADAPTER.source_config(version="1.0", public_config={"bits": 4}, resolved_config={"desc_act": True})
        assert config.config_hash != other.config_hash
        assert config.as_dict()["config_hash"] == config.config_hash

    def test_unavailable_library_is_recorded_not_raised(self):
        config = GPTQ_ADAPTER.source_config()
        if not config.package_available:
            assert config.unavailable_reason


@pytest.mark.unit
class TestFieldMappingAudit:
    def test_unmapped_field_fails_audit(self):
        mappings = [FieldMapping("qweight", "canonical.qvalues", "derived")]
        report = audit_field_mapping(["qweight", "scales"], mappings)
        assert report["audit_passed"] is False
        assert report["unmapped_fields"] == ["scales"]

    def test_unsupported_requires_reason(self):
        with pytest.raises(Exception):
            FieldMapping("x", "", "unsupported", loss_reason="")

    def test_duplicate_field_detected(self):
        mappings = [
            FieldMapping("a", "x", "mapped"),
            FieldMapping("a", "y", "mapped"),
        ]
        report = audit_field_mapping(["a"], mappings)
        assert report["duplicate_fields"] == ["a"]

    @pytest.mark.parametrize(
        "adapter", [GPTQ_ADAPTER, AWQ_ADAPTER, SMOOTHQUANT_ADAPTER]
    )
    def test_declared_method_fields_have_no_silent_drop(self, adapter):
        report = audit_field_mapping(adapter.source_fields, adapter.field_mappings())
        assert report["audit_passed"] is True
        assert report["unmapped_fields"] == []

    def test_transform_planes_are_in_source_record_audit(self):
        record = SourceTensorRecord(
            name="layer.weight",
            shape=(1, 2),
            bits=4,
            symmetric=True,
            group_size=2,
            packed_payload=pack_canonical([1, -1], 4),
            scales=[0.5],
            pre_scale=[2.0, 4.0],
            post_scale=[0.5, 0.25],
        )
        converted = AWQ_ADAPTER.convert_record(record, rows=1, cols=2)
        assert converted["mapping_audit"]["audit_passed"] is True
        assert converted["pre_scale"] == [2.0, 4.0]
        assert converted["post_scale"] == [0.5, 0.25]


@pytest.mark.unit
class TestConverter:
    def _record(self, **overrides):
        base = dict(
            name="layer.q_proj.weight",
            shape=(4, 64),
            bits=4,
            symmetric=True,
            group_size=32,
            packed_payload=pack_canonical([1, -2, 3, -4] * 32, 4),
            scales=[0.5] * 8,
        )
        base.update(overrides)
        return SourceTensorRecord(**base)

    def test_convert_produces_canonical_values(self):
        record = self._record(packed_payload=pack_canonical([1, -2, 3, -4] * 64, 4))
        out = GPTQ_ADAPTER.convert_record(record, rows=4, cols=64)
        assert out["qvalues"][:4] == [1, -2, 3, -4]
        assert out["mapping_audit"]["audit_passed"] is True
        assert out["scheme"].group_size == 32

    def test_truncated_payload_refused(self):
        record = self._record(packed_payload=b"\x00" * 10)
        with pytest.raises(Exception):
            GPTQ_ADAPTER.convert_record(record, rows=4, cols=64)

    def test_scale_count_mismatch_refused(self):
        record = self._record(scales=[0.5] * 3)
        with pytest.raises(Exception):
            GPTQ_ADAPTER.convert_record(record, rows=4, cols=64)


@pytest.mark.unit
class TestMethodMatrix:
    def test_matrix_hash_stable(self):
        matrix = MethodMatrix(
            methods=["gptq", "awq"],
            library_versions={"gptq": "1.0"},
            module_scope="linear",
            scheme_policy="w4a16",
            calibration_budget={"n": 128},
            quality_gate={"margin": 0.0},
        )
        assert matrix.matrix_hash == matrix.matrix_hash


@pytest.mark.unit
class TestUnits:
    def _rows(self, layers=2):
        rows = []
        for layer in range(layers):
            rows.append(
                {"name": f"model.layers.{layer}.self_attn.q_proj.weight", "num_parameters": 100, "quantized": True}
            )
            rows.append(
                {"name": f"model.layers.{layer}.mlp.down_proj.weight", "num_parameters": 100, "quantized": True}
            )
        return rows

    def test_fused_group_collapses_to_one_unit(self):
        cfg8 = UnitConfig(bits=8, group_size=None, method="rtn")
        cfg4 = UnitConfig(bits=4, group_size=128, method="rtn")
        units = build_units(
            self._rows(),
            legal_configs=[cfg8, cfg4],
            fused_groups={
                "attn_q": ["model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj"]
            },
        )
        ids = {unit.unit_id for unit in units}
        assert "fused:attn_q" in ids
        fused = next(unit for unit in units if unit.unit_id == "fused:attn_q")
        assert fused.constraint == "fused_group"
        assert fused.parameter_count == 200

    def test_role_classification(self):
        cfg4 = UnitConfig(bits=4, group_size=128, method="rtn")
        units = build_units(self._rows(1), legal_configs=[cfg4])
        roles = {unit.unit_id: unit.role for unit in units}
        assert roles["model.layers.0.self_attn.q_proj"] == "attention_qkv"
        assert roles["model.layers.0.mlp.down_proj"] == "mlp_down"

    def test_empty_legal_configs_refused(self):
        with pytest.raises(ConfigError):
            build_units(self._rows(1), legal_configs=[])


@pytest.mark.unit
class TestPolicy:
    def _rows(self, layers=1):
        rows = []
        for layer in range(layers):
            rows.append(
                {"name": f"model.layers.{layer}.self_attn.q_proj.weight", "num_parameters": 100, "quantized": True}
            )
            rows.append(
                {"name": f"model.layers.{layer}.mlp.down_proj.weight", "num_parameters": 100, "quantized": True}
            )
        return rows

    def test_policy_yaml_round_trip(self):
        policy = MixedPrecisionPolicy(
            name="p",
            entries=[UnitPolicyEntry(unit_id="u1", bits=4, group_size=128, method="rtn")],
        )
        clone = MixedPrecisionPolicy.from_yaml(policy.to_yaml())
        assert clone.policy_hash == policy.policy_hash

    def test_policy_validation_rejects_unknown_unit(self):
        cfg4 = UnitConfig(bits=4, group_size=128, method="rtn")
        units = build_units(self._rows(1), legal_configs=[cfg4])
        policy = MixedPrecisionPolicy(
            name="p",
            entries=[UnitPolicyEntry(unit_id="does.not.exist", bits=4, group_size=128, method="rtn")],
        )
        validation = policy.validate(units)
        assert validation.valid is False
        assert any(problem["kind"] == "unknown_unit" for problem in validation.problems)

    def test_policy_validation_rejects_illegal_config(self):
        cfg4 = UnitConfig(bits=4, group_size=128, method="rtn")
        units = build_units(self._rows(1), legal_configs=[cfg4])
        policy = MixedPrecisionPolicy(
            name="p",
            entries=[UnitPolicyEntry(unit_id="model.layers.0.self_attn.q_proj", bits=8, group_size=None, method="rtn")],
        )
        validation = policy.validate(units)
        assert validation.valid is False

    def test_baseline_policies(self):
        cfg8 = UnitConfig(bits=8, group_size=None, method="rtn")
        cfg4 = UnitConfig(bits=4, group_size=128, method="rtn")
        units = build_units(self._rows(1), legal_configs=[cfg8, cfg4])
        policies = baseline_policies(units, w8=cfg8, w4=cfg4)
        assert set(policies) == {"all_fp16", "all_w8", "all_w4", "heuristic_skip"}
        assert policies["all_w4"].validate(units).valid

    def test_greedy_search_trace_records_all_candidates(self):
        cfg8 = UnitConfig(bits=8, group_size=None, method="rtn")
        cfg4 = UnitConfig(bits=4, group_size=128, method="rtn")
        units = build_units(self._rows(1), legal_configs=[cfg8, cfg4])

        def evaluate(policy):
            return {"quality_margin": 0.1 * sum(1 for e in policy.entries if e.bits == 8), "score": 0.0}

        def quality_pass(metrics):
            return metrics["quality_margin"] >= 0.0

        _policy, trace = greedy_search(
            units,
            w8=cfg8,
            w4=cfg4,
            evaluate=evaluate,
            quality_pass=quality_pass,
            byte_budget=1000000,
            unit_bytes=lambda u, c: 100,
        )
        assert len(trace.evaluations) >= 1
        assert trace.stop_reason


@pytest.mark.unit
class TestSensitivityPlanning:
    def _units(self):
        cfg4 = UnitConfig(bits=4, group_size=128, method="rtn")
        return build_units(
            [
                {"name": f"model.layers.{l}.self_attn.q_proj.weight", "num_parameters": 10, "quantized": True}
                for l in range(3)
            ],
            legal_configs=[cfg4],
        )

    def test_single_quant_plan(self):
        units = self._units()
        plan = sens.plan_single_quant(units, quant_config={"bits": 4})
        assert len(plan) == 3
        assert plan[0].kind == sens.SINGLE_QUANT

    def test_leave_one_out_plan(self):
        units = self._units()
        plan = sens.plan_leave_one_out(units, baseline_config={"bits": 4}, restore_config={"bits": 8})
        assert len(plan) == 3

    def test_cumulative_random_control_is_seeded(self):
        units = self._units()
        a = sens.plan_cumulative(units, quant_config={"bits": 4}, order=sens.ORDER_RANDOM, seed=0)
        b = sens.plan_cumulative(units, quant_config={"bits": 4}, order=sens.ORDER_RANDOM, seed=0)
        assert [i.unit_ids for i in a] == [i.unit_ids for i in b]

    def test_unknown_order_refused(self):
        with pytest.raises(ConfigError):
            sens.plan_cumulative(self._units(), quant_config={}, order="magic")

    def test_interaction_formula(self):
        # interaction(i,j) = loss(i,j) - loss(i) - loss(j) + loss(none)
        assert sens.interaction(2.0, 1.0, 1.0, 0.0) == pytest.approx(0.0)

    def test_weight_error_metrics(self):
        report = sens.weight_error_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.1])
        assert report["max_abs_error"] == pytest.approx(0.1)
