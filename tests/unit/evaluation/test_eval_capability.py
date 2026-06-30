"""Unit tests for the E12-02 capability registry and platform adapters."""

from __future__ import annotations

import pytest

from hqsb.core.errors import ConfigError
from hqsb.evaluation import capability as cap
from hqsb.evaluation import platform as plat

pytestmark = pytest.mark.unit


def _verified(**overrides: object) -> cap.CapabilityEvidence:
    payload = {
        "evidence_id": "",
        "platform_instance_id": "plat_a",
        "feature_id": "bf16_storage",
        "declared_status": "yes",
        "declared_source_id": "vendor-doc#bf16",
        "discovered_status": "yes",
        "verified_status": "yes",
        "probe_spec_hash": "a" * 64,
        "input_hash": "b" * 64,
        "requested_backend": "cuda",
        "actual_backend": "cuda",
        "build_status": "pass",
        "load_status": "pass",
        "execution_status": "pass",
        "sync_status": "pass",
        "correctness_status": "pass",
        "invalidation_key": "driver=d1|runtime=r1",
    }
    payload.update(overrides)
    return cap.CapabilityEvidence(**payload)  # type: ignore[arg-type]


class TestFeatureRegistry:
    def test_registry_is_fine_grained_and_consistent(self) -> None:
        assert len(cap.FEATURE_REGISTRY) >= 40
        assert cap.registry_problems() == []

    def test_bf16_is_not_a_single_boolean(self) -> None:
        ids = {row.feature_id for row in cap.FEATURE_REGISTRY}
        assert {"bf16_storage", "bf16_elementwise", "bf16_reduction_accum_fp32", "bf16_gemm_native"} <= ids

    def test_dependency_order_places_dependencies_first(self) -> None:
        order = cap.dependency_order(("bf16_gemm_native",))
        assert order.index("bf16_storage") < order.index("bf16_gemm_native")

    def test_dependency_cycle_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cyclic = cap.Feature(
            feature_id="cycle_a",
            layer="numeric",
            description="fixture",
            success_criteria="never",
            depends_on=("cycle_b",),
        )
        other = cap.Feature(
            feature_id="cycle_b",
            layer="numeric",
            description="fixture",
            success_criteria="never",
            depends_on=("cycle_a",),
        )
        monkeypatch.setitem(cap.FEATURE_BY_ID, "cycle_a", cyclic)
        monkeypatch.setitem(cap.FEATURE_BY_ID, "cycle_b", other)
        with pytest.raises(ConfigError):
            cap.dependency_order(("cycle_a",))

    def test_level_ordering(self) -> None:
        assert cap.meets("VERIFIED", "DISCOVERED") is True
        assert cap.meets("DISCOVERED", "VERIFIED") is False
        with pytest.raises(ConfigError):
            cap.meets("SOMETHING", "VERIFIED")


class TestProbeClassification:
    def test_failure_categories_stay_distinguishable(self) -> None:
        assert cap.classify_probe_outcome(component_present=False)["failure_category"] == "MISSING_COMPONENT"
        assert cap.classify_probe_outcome(permission_denied=True)["failure_category"] == "PERMISSION_DENIED"
        assert cap.classify_probe_outcome(timeout=True)["failure_category"] == "TIMEOUT"
        assert cap.classify_probe_outcome(build_ok=False)["failure_category"] == "COMPILE_FAILED"
        assert cap.classify_probe_outcome(build_ok=True, load_ok=False)["failure_category"] == "LOAD_FAILED"

    def test_failure_is_never_generalised_to_unsupported(self) -> None:
        naive = cap.classify_probe_outcome(component_present=False)
        assert naive["device_conclusion_allowed"] is False
        hardware = cap.classify_probe_outcome(
            build_ok=True, load_ok=True, execution_ok=False, hardware_supports=False
        )
        assert hardware["failure_category"] == "UNSUPPORTED_BY_HARDWARE"

    def test_numerical_mismatch_is_not_an_execution_failure(self) -> None:
        result = cap.classify_probe_outcome(
            build_ok=True, load_ok=True, execution_ok=True, sync_ok=True, numerical_ok=False
        )
        assert result["failure_category"] == "NUMERICAL_MISMATCH"


class TestCapabilityEvidence:
    def test_verified_requires_execution_sync_correctness_and_actual_backend(self) -> None:
        problems = _verified(execution_status="", actual_backend="").validate()
        assert any("execution_status" in item for item in problems)
        assert any("actual backend" in item for item in problems)

    def test_declared_needs_a_source(self) -> None:
        row = cap.CapabilityEvidence(
            evidence_id="",
            platform_instance_id="plat_a",
            feature_id="bf16_storage",
            declared_status="yes",
        )
        assert any("DECLARED without a source" in item for item in row.validate())

    def test_silent_fallback_needs_a_reason(self) -> None:
        row = _verified(actual_backend="eager", reason="")
        assert any("requested != actual" in item for item in row.validate())

    def test_upgrade_rule_stops_at_discovered_without_execution(self) -> None:
        row = cap.CapabilityEvidence(
            evidence_id="",
            platform_instance_id="plat_a",
            feature_id="bf16_storage",
            discovered_status="yes",
        )
        assert cap.evidence_upgrade_rule(row)["level"] == "DISCOVERED"

    def test_upgrade_rule_refuses_fallback_evidence(self) -> None:
        row = _verified(requested_backend="cuda_native", actual_backend="eager", reason="guard refused")
        result = cap.evidence_upgrade_rule(row)
        assert result["allowed"] is False
        assert "fallback" in result["reason"]

    def test_verified_evidence_is_accepted(self) -> None:
        assert cap.evidence_upgrade_rule(_verified())["allowed"] is True


class TestCapabilityMatrix:
    def test_unprobed_feature_is_not_supported(self) -> None:
        matrix = cap.CapabilityMatrix()
        assert matrix.supported("plat_a", "fp8_gemm_native") is False
        assert matrix.status_for("plat_a", "fp8_gemm_native") == "NONE"

    def test_verified_feature_is_supported(self) -> None:
        matrix = cap.CapabilityMatrix()
        matrix.record(_verified())
        assert matrix.supported("plat_a", "bf16_storage") is True

    def test_weakening_overwrite_is_rejected(self) -> None:
        matrix = cap.CapabilityMatrix()
        matrix.record(_verified())
        weaker = cap.CapabilityEvidence(
            evidence_id="",
            platform_instance_id="plat_a",
            feature_id="bf16_storage",
            discovered_status="yes",
        )
        with pytest.raises(ConfigError):
            matrix.record(weaker)

    def test_evidence_gaps_reports_reason(self) -> None:
        matrix = cap.CapabilityMatrix()
        matrix.record(_verified())
        gaps = matrix.evidence_gaps(
            platform_instance_id="plat_a", required=("bf16_storage", "bf16_gemm_native")
        )
        assert [row["feature_id"] for row in gaps] == ["bf16_gemm_native"]
        assert gaps[0]["reason"]

    def test_dependency_unsatisfied_blocks_dependent_feature(self) -> None:
        matrix = cap.CapabilityMatrix()
        report = cap.dependency_satisfied(
            matrix, platform_instance_id="plat_a", feature_id="bf16_gemm_native"
        )
        assert report["satisfied"] is False
        assert report["unsatisfied"] == ["bf16_storage"]


class TestInvalidation:
    def test_driver_change_expires_matching_evidence(self) -> None:
        row = _verified(invalidation_key=cap.invalidation_key({"driver": "d1"}, arch="sm_90", permission="user"))
        expired = cap.invalidate([row], ["driver_change"])
        assert expired and expired[0]["evidence_id"] == row.evidence_id

    def test_unrelated_change_is_whitelisted_with_a_reason(self) -> None:
        row = _verified(invalidation_key=cap.invalidation_key({"driver": "d1"}, arch="sm_90", permission="user"))
        assert cap.invalidate([row], ["price_change"]) == ()

    def test_undeclared_trigger_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            cap.invalidate([], ["something_new"])

    def test_missing_invalidation_key_expires(self) -> None:
        row = _verified(invalidation_key="")
        assert cap.invalidate([row], ["driver_change"])[0]["trigger"] == "missing_invalidation_key"


class TestCoverageAndNegatives:
    def test_coverage_join_marks_blocked_and_not_applicable(self) -> None:
        matrix = cap.CapabilityMatrix()
        matrix.record(_verified())
        rows = cap.coverage_join(
            {
                "cand_a": {"numeric": ("bf16_storage", "bf16_gemm_native"), "distributed": ()},
            },
            matrix,
            platform_by_candidate={"cand_a": "plat_a"},
        )
        statuses = {row["layer"]: row["join_status"] for row in rows}
        assert statuses["numeric"] == "blocked"
        assert statuses["distributed"] == "not_applicable"

    def test_negative_probes_must_be_classified_not_unknown(self) -> None:
        cases = cap.negative_probe_cases()
        results = [
            {
                "case_id": case["case_id"],
                "expected_category": case["expected_category"],
                "observed_category": case["expected_category"],
            }
            for case in cases
        ]
        assert cap.evaluate_negative_probes(results)["ok"] is True
        broken = [dict(results[0], observed_category="UNKNOWN")]
        report = cap.evaluate_negative_probes(broken)
        assert report["ok"] is False
        assert report["generic_unknown"] == 1

    def test_probe_plan_covers_dependencies(self) -> None:
        plan = cap.probe_plan(("bf16_gemm_native",), harness_hash="h" * 64)
        ids = [row["feature_id"] for row in plan]
        assert ids.index("bf16_storage") < ids.index("bf16_gemm_native")

    def test_probe_spec_hash_is_stable(self) -> None:
        first = cap.probe_spec("bf16_storage", harness_hash="h", command_or_api="cli", inputs={"a": 1})
        second = cap.probe_spec("bf16_storage", harness_hash="h", command_or_api="cli", inputs={"a": 1})
        assert first["probe_spec_hash"] == second["probe_spec_hash"]


class TestPlatformLayer:
    def test_platform_identity_id_is_stable_and_validated(self) -> None:
        identity = plat.PlatformIdentity(
            vendor="nvidia",
            sku="rtx-3090",
            revision="a1",
            device_count=1,
            memory_bytes=24 * 1024**3,
            interconnect="pcie",
            host_id="host-1",
        )
        assert identity.platform_instance_id.startswith("plat_")
        assert identity.validate() == []
        broken = plat.PlatformIdentity(vendor="nvidia", sku="", device_count=0, host_id="")
        assert len(broken.validate()) >= 2

    def test_conflicting_identity_sources_block_the_instance(self) -> None:
        conflicts = plat.identity_conflicts(
            {
                "os_pci": {"sku": "A100-SXM", "device_count": 1},
                "vendor_cli": {"sku": "A100-PCIe", "device_count": 2},
                "runtime_api": {"sku": "A100-SXM"},
            }
        )
        fields = {row["field"]: row["status"] for row in conflicts}
        assert fields["device_count"] == "CONFLICT"
        assert fields["sku"] == "CONFLICT"
        assert all(row["values"] for row in conflicts)

    def test_sources_missing_a_field_are_reported_as_unavailable(self) -> None:
        conflicts = plat.identity_conflicts(
            {
                "os_pci": {"sku": "A100", "memory_bytes": 80 * 1024**3},
                "vendor_cli": {"sku": "A100"},
            }
        )
        fields = {row["field"]: row["status"] for row in conflicts}
        assert fields["memory_bytes"] == "CONFLICT"

    def test_software_stack_requires_core_versions(self) -> None:
        stack = plat.SoftwareStack(driver="", runtime="cuda-12", framework="torch-2.8")
        assert any("driver" in item for item in stack.validate())

    def test_unmapped_vendor_field_stays_namespaced(self) -> None:
        row = plat.canonicalize_telemetry_field({"vendor_field": "mystery:counter", "value": 42})
        assert row["status"] == "MEASUREMENT_UNAVAILABLE"
        assert row["canonical_field"] == ""
        assert row["vendor.mystery:counter"] == 42

    def test_mapped_field_without_sample_is_unavailable_not_zero(self) -> None:
        row = plat.canonicalize_telemetry_field(
            {"vendor_field": "nvidia-smi:power.draw", "canonical_field": "accelerator.power_W", "value": None}
        )
        assert row["status"] == "MEASUREMENT_UNAVAILABLE"
        assert "value" not in row

    def test_energy_capability_lists_missing_fields(self) -> None:
        report = plat.energy_capability(["accelerator.power_W", "accelerator.energy_J"])
        assert report["status"] == "MEASUREMENT_UNAVAILABLE"
        assert "accelerator.clock_mhz" in report["missing_for_energy"]
        assert report["can_measure_energy"] is False

    def test_telemetry_smoke_detects_stale_constant_series(self) -> None:
        series = [{"t_ns": 1, "power_w": 100.0}, {"t_ns": 2, "power_w": 100.0}]
        report = plat.telemetry_smoke_check(series)
        assert report["constant_value_suspected_stale_cache"] is True
        assert report["note"].startswith("capability check only")

    def test_probe_harness_validation(self) -> None:
        harness = plat.probe_harness(
            source_uri="hqsb://S12/probes/bf16",
            source_hash="a" * 64,
            cli_or_api="probe:bf16_storage",
            reference_impl="cpu_reference@1",
        )
        assert harness.validate() == []
        assert harness.harness_id.startswith("harness_")


class TestSmokeAndSteps:
    def test_protocol_steps_are_complete(self) -> None:
        assert len(cap.PROTOCOL_STEPS) == 36
        assert all(interfaces for _, _, interfaces in cap.PROTOCOL_STEPS)

    def test_smoke_self_check_is_labelled(self) -> None:
        result = cap.smoke_self_check()
        assert result["status"] == "smoke"
        assert result["claim_allowed"] is False
        assert result["unprobed_is_not_supported"] is True
        assert result["registry_problems"] == []
