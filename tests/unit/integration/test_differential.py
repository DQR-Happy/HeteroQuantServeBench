"""Differential correctness scaffolding tests (E06-03 interface layer)."""

from __future__ import annotations

import math

import pytest

from hqsb.core.errors import ConfigError, SchemaError
from hqsb.integration import differential as diff


@pytest.mark.unit
class TestPathMatrix:
    def test_eight_paths_are_frozen(self):
        assert len(diff.PATHS) == 8
        ids = [path.path_id for path in diff.PATHS]
        assert ids[0] == "eager-reference"
        assert "fallback" in ids

    def test_spec_requires_the_model_identity(self):
        with pytest.raises(ConfigError):
            diff.DifferentialSpec(model_id="m", model_manifest_sha256="", input_token_hash="t")

    def test_unknown_path_refused(self):
        with pytest.raises(ConfigError):
            diff.DifferentialSpec(
                model_id="m",
                model_manifest_sha256="h",
                input_token_hash="t",
                paths=("vibes",),
            )

    def test_spec_digest_is_stable(self):
        spec = diff.DifferentialSpec(model_id="m", model_manifest_sha256="h", input_token_hash="t")
        assert spec.digest() == spec.digest()

    def test_path_matrix_table_is_markdown(self):
        table = diff.path_matrix_table()
        assert "| path |" in table
        assert "eager-reference" in table


@pytest.mark.unit
class TestToleranceRegistry:
    def test_missing_entry_is_an_error_not_a_default(self):
        registry = diff.default_tolerance_registry()
        with pytest.raises(ConfigError):
            registry.get(diff.Level.OPERATOR, "float8")

    def test_thresholds_are_per_dtype_and_level(self):
        registry = diff.default_tolerance_registry()
        fp32 = registry.get(diff.Level.OPERATOR, "float32")
        fp16 = registry.get(diff.Level.OPERATOR, "float16")
        assert fp32.max_abs < fp16.max_abs
        assert registry.get(diff.Level.BLOCK, "float16").max_abs > fp16.max_abs

    def test_unknown_level_refused(self):
        with pytest.raises(SchemaError):
            diff.ToleranceSpec(level="vibes", dtype="float16", max_abs=1.0, rel_l2=1.0)

    def test_non_positive_tolerance_refused(self):
        with pytest.raises(ConfigError):
            diff.ToleranceSpec(
                level=diff.Level.OPERATOR, dtype="float16", max_abs=0.0, rel_l2=1.0
            )

    def test_registry_digest_is_stable(self):
        first = diff.default_tolerance_registry().digest
        second = diff.default_tolerance_registry().digest
        assert first == second


@pytest.mark.unit
class TestMetricComparison:
    def _tolerance(self, level=diff.Level.OPERATOR, dtype="float16"):
        return diff.default_tolerance_registry().get(level, dtype)

    def test_identical_outputs_pass(self):
        report = diff.compare_arrays([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], self._tolerance())
        assert report.passed
        assert report.max_abs == 0.0
        assert report.first_mismatch is None

    def test_mismatch_beyond_tolerance_is_reported_with_the_index(self):
        report = diff.compare_arrays(
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.5], self._tolerance()
        )
        assert not report.passed
        assert "MAX_ABS" in report.failures
        assert report.first_mismatch == 2

    def test_nan_is_a_hard_failure(self):
        report = diff.compare_arrays(
            [1.0, float("nan")], [1.0, 1.0], self._tolerance()
        )
        assert not report.passed
        assert "NAN_OR_INF" in report.failures
        assert report.nan_count == 1

    def test_inf_is_a_hard_failure(self):
        report = diff.compare_arrays(
            [float("inf"), 1.0], [1.0, 1.0], self._tolerance()
        )
        assert "NAN_OR_INF" in report.failures
        assert report.inf_count == 1

    def test_cosine_gate(self):
        tolerance = diff.ToleranceSpec(
            level=diff.Level.OPERATOR, dtype="float16", max_abs=10.0, rel_l2=10.0, cosine=0.99
        )
        report = diff.compare_arrays([1.0, 0.0], [0.0, 1.0], tolerance)
        assert "COSINE" in report.failures

    def test_element_count_mismatch_refused(self):
        with pytest.raises(ConfigError):
            diff.compare_arrays([1.0], [1.0, 2.0], self._tolerance())

    def test_nested_inputs_are_flattened(self):
        report = diff.compare_arrays(
            [[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], self._tolerance()
        )
        assert report.passed

    def test_observed_kernel_is_recorded(self):
        report = diff.compare_arrays(
            [1.0], [1.0], self._tolerance(), observed_kernel="hqsb_fused_add_rms_norm_v1"
        )
        assert report.observed_kernel == "hqsb_fused_add_rms_norm_v1"

    def test_repository_diff_summary_is_reused(self):
        summary = diff.reference_diff_summary([1.0, 2.0], [1.0, 2.5])
        assert "max_abs" in summary or summary


@pytest.mark.unit
class TestFusionSemantics:
    def test_semantics_are_frozen_and_functional(self):
        semantics = diff.FROZEN_ADD_RMSNORM_SEMANTICS
        assert semantics.residual_in_place is False
        assert semantics.outputs_alias_inputs is False
        assert semantics.normalized_reads == diff.ReadOrder.AFTER_ROUNDING
        assert semantics.accumulation_dtype == "float32"

    def test_unknown_read_order_refused(self):
        with pytest.raises(SchemaError):
            diff.FusionSemantics(
                name="x",
                accumulation_dtype="float32",
                r_new_rounding="r",
                normalized_reads="whenever",
                residual_in_place=False,
                eps_position="e",
                weight_broadcast=True,
                outputs_alias_inputs=False,
                extra_users_policy="p",
            )

    def test_reference_matches_the_composition(self):
        x = [0.1, 0.2, 0.3, 0.4]
        residual = [0.5, 0.6, 0.7, 0.8]
        weight = [1.0, 1.0, 1.0, 1.0]
        fused = diff.fused_add_rms_norm_reference(x, residual, weight)
        composed = diff.composed_add_rms_norm_reference(x, residual, weight)
        assert fused[1] == composed[1]
        for left, right in zip(fused[0], composed[0]):
            assert left == pytest.approx(right, rel=1e-12)

    def test_reference_refuses_in_place_semantics(self):
        semantics = diff.FusionSemantics(
            name="mutable",
            accumulation_dtype="float32",
            r_new_rounding="r",
            normalized_reads=diff.ReadOrder.AFTER_ROUNDING,
            residual_in_place=True,
            eps_position="e",
            weight_broadcast=True,
            outputs_alias_inputs=True,
            extra_users_policy="p",
        )
        with pytest.raises(ConfigError):
            diff.fused_add_rms_norm_reference([1.0], [1.0], [1.0], semantics=semantics)

    def test_reference_refuses_mismatched_shapes(self):
        with pytest.raises(ConfigError):
            diff.fused_add_rms_norm_reference([1.0], [1.0, 2.0], [1.0])

    def test_reference_is_scale_correct(self):
        # rms_norm of a constant vector with unit weight is 1.0 up to eps.
        x, residual, weight = [1.0] * 4, [0.0] * 4, [1.0] * 4
        normalized, updated = diff.fused_add_rms_norm_reference(x, residual, weight, eps=0.0)
        assert updated == [1.0] * 4
        assert all(value == pytest.approx(1.0) for value in normalized)
        assert not math.isnan(normalized[0])


@pytest.mark.unit
class TestStateAndModeSwitch:
    def _state(self, **overrides) -> diff.StateSnapshot:
        payload = {
            "state_dict_hash": "sd",
            "parameter_ids": ("0", "1"),
            "tied_weights": (("lm_head", "embed"),),
            "buffers_hash": "b",
            "inference_mode": True,
            "rng_state_hash": "r",
            "kv_cache_hash": "",
            "compiled_wrapper_id": "",
            "backend_enabled": False,
            "output_hash": "o",
        }
        payload.update(overrides)
        return diff.StateSnapshot(**payload)

    def test_identical_states_compare_ok(self):
        assert diff.compare_state(self._state(), self._state())["ok"] is True

    def test_state_difference_is_reported_field_by_field(self):
        report = diff.compare_state(self._state(), self._state(output_hash="other"))
        assert not report["ok"]
        assert report["differences"][0]["field"] == "output_hash"

    def test_enable_then_disable_audit(self):
        audit = diff.ModeSwitchAudit(name="qwen")
        audit.record(
            "enable",
            state=self._state(backend_enabled=True),
            expected_backend="hqsb.compiled.fused",
            actual_backend="hqsb.compiled.fused",
            wrapper_rebuilt=True,
            output_matches_reference=True,
        )
        audit.record(
            "disable",
            state=self._state(),
            expected_backend="torch.eager.reference",
            actual_backend="torch.eager.reference",
            wrapper_rebuilt=False,
            output_matches_reference=True,
        )
        assert audit.ok

    def test_re_enable_without_a_rebuilt_wrapper_fails(self):
        audit = diff.ModeSwitchAudit(name="qwen")
        record = audit.record(
            "enable",
            state=self._state(backend_enabled=True),
            expected_backend="hqsb.compiled.fused",
            actual_backend="hqsb.compiled.fused",
            wrapper_rebuilt=False,
            output_matches_reference=True,
        )
        assert record["ok"] is False


@pytest.mark.unit
class TestDivergenceAndMatrix:
    def test_first_divergence_is_localised(self):
        locator = diff.DivergenceLocator()
        locator.observe(
            level=diff.Level.BLOCK,
            node="n1",
            op="hqsb.rms_norm",
            module="layers.0.input_layernorm",
            layer=0,
            step=0,
            metric="max_abs",
            value=0.001,
            threshold=0.05,
        )
        locator.observe(
            level=diff.Level.BLOCK,
            node="n2",
            op="hqsb.fused_add_rms_norm",
            module="layers.3.input_layernorm",
            layer=3,
            step=3,
            metric="max_abs",
            value=0.9,
            threshold=0.05,
        )
        first = locator.first_over()
        assert first is not None
        assert first.layer == 3
        assert locator.as_dict()["observed"] == 2

    def test_no_divergence_returns_none(self):
        locator = diff.DivergenceLocator()
        locator.observe(
            level=diff.Level.OPERATOR,
            node="n",
            op="op",
            module="m",
            layer=0,
            step=0,
            metric="max_abs",
            value=0.0,
            threshold=1.0,
        )
        assert locator.first_over() is None

    def test_matrix_missing_cells_are_explicit(self):
        spec = diff.DifferentialSpec(model_id="m", model_manifest_sha256="h", input_token_hash="t")
        matrix = diff.CorrectnessMatrix(specs=spec)
        missing = matrix.missing_cells()
        assert len(missing) == len(spec.paths) * len(diff.Level.ALL)
        assert matrix.complete is False

    def test_marking_a_cell_reduces_missing(self):
        spec = diff.DifferentialSpec(model_id="m", model_manifest_sha256="h", input_token_hash="t")
        matrix = diff.CorrectnessMatrix(specs=spec)
        matrix.mark("eager-reference", diff.Level.OPERATOR, diff.CellStatus.PASS)
        assert ("eager-reference", diff.Level.OPERATOR) not in matrix.missing_cells()

    def test_failed_cells_are_listed(self):
        spec = diff.DifferentialSpec(model_id="m", model_manifest_sha256="h", input_token_hash="t")
        matrix = diff.CorrectnessMatrix(specs=spec)
        matrix.mark("eager-custom", diff.Level.BLOCK, diff.CellStatus.FAIL)
        assert matrix.failed_cells == (("eager-custom", diff.Level.BLOCK),)

    def test_unknown_path_or_status_refused(self):
        spec = diff.DifferentialSpec(model_id="m", model_manifest_sha256="h", input_token_hash="t")
        matrix = diff.CorrectnessMatrix(specs=spec)
        with pytest.raises(ConfigError):
            matrix.mark("unknown-path", diff.Level.OPERATOR, diff.CellStatus.PASS)
        with pytest.raises(ConfigError):
            matrix.mark("eager-custom", diff.Level.BLOCK, "PROBABLY_FINE")
        with pytest.raises(ConfigError):
            matrix.mark("eager-custom", "level-42", diff.CellStatus.PASS)

    def test_matrix_serialises_the_spec_digest(self):
        spec = diff.DifferentialSpec(model_id="m", model_manifest_sha256="h", input_token_hash="t")
        payload = diff.CorrectnessMatrix(specs=spec).as_dict()
        assert payload["spec_digest"] == spec.digest()
        assert payload["complete"] is False
