"""Adapter/anti-hardcoding and error-taxonomy tests (E06-09/E06-11)."""

from __future__ import annotations

import os

import pytest

from hqsb.core.errors import CapabilityError, ConfigError, SchemaError
from hqsb.integration import adapter, abi, taxonomy


@pytest.mark.unit
class TestHardcodeScanner:
    def _write(self, tmp_path, name, text):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_planted_violations_are_found_and_classified(self, tmp_path):
        path = self._write(
            tmp_path / "core",
            "leaky.py",
            "from transformers import Qwen3ForCausalLM\n"
            "num_hidden_layers = 28\n"
            "if backend == 'cuda':\n"
            "    pass\n"
            "hidden_size = 2048\n"
            "raise ValueError('unsupported shape')\n",
        )
        findings = adapter.hardcode_scan([os.path.dirname(path)], adapter_markers=())
        rules = {item.rule_id for item in findings}
        assert "HC-MODEL-CLASS" in rules
        assert "HC-FIXED-LAYERS" in rules
        assert "HC-BACKEND-BRANCH" in rules
        assert "HC-SHAPE-CONSTANT" in rules
        report = adapter.hardcode_report(findings)
        assert report["leaks"] >= 4
        assert not report["ok"]

    def test_adapter_hits_are_not_leaks(self, tmp_path):
        path = self._write(
            tmp_path / "adapters",
            "qwen_adapter.py",
            "MODEL_CLASS = 'Qwen3ForCausalLM'\n",
        )
        findings = adapter.hardcode_scan([os.path.dirname(path)])
        assert findings
        assert all(item.in_adapter for item in findings)
        assert adapter.hardcode_report(findings)["ok"] is True

    def test_clean_core_code_passes(self, tmp_path):
        path = self._write(
            tmp_path / "core",
            "clean.py",
            "def select(capability, request):\n"
            "    return capability.supports(request)\n",
        )
        findings = adapter.hardcode_scan([os.path.dirname(path)], adapter_markers=())
        assert adapter.hardcode_report(findings)["ok"] is True

    def test_real_integration_package_is_clean(self):
        findings = adapter.hardcode_scan([os.path.join(os.getcwd(), "hqsb", "integration")])
        report = adapter.hardcode_report(findings)
        assert report["ok"] is True, report["leak_details"]


@pytest.mark.unit
class TestChangeBudget:
    def test_change_classification(self):
        assert (
            adapter.classify_change("hqsb/integration/cache.py", "fix a cache key bug")
            == adapter.ChangeClass.GENERIC_BUG_FIX
        )
        assert (
            adapter.classify_change("hqsb/integration/graph.py", "add a new contract field")
            == adapter.ChangeClass.CONTRACT_EXTENSION
        )
        leak = adapter.classify_change(
            "hqsb/integration/adapter.py", "special-case the Qwen module path"
        )
        assert leak in (adapter.ChangeClass.MODEL_SPECIFIC_LEAK, adapter.ChangeClass.CONTRACT_EXTENSION)

    def test_unknown_change_class_refused(self):
        with pytest.raises(ConfigError):
            adapter.ChangeBudget(
                path="x.py", loc_added=1, loc_removed=0, change_class="whatever"
            )

    def test_summary_flags_core_leaks(self):
        budgets = (
            adapter.ChangeBudget(
                path="hqsb/integration/adapter.py",
                loc_added=10,
                loc_removed=0,
                change_class=adapter.ChangeClass.MODEL_SPECIFIC_LEAK,
                reason="special case",
            ),
            adapter.ChangeBudget(
                path="configs/x.yaml",
                loc_added=2,
                loc_removed=0,
                change_class=adapter.ChangeClass.CONTRACT_EXTENSION,
            ),
        )
        summary = adapter.summarize_changes(budgets)
        assert not summary["ok"]
        assert summary["core_leaks"][0]["path"].endswith("adapter.py")

    def test_budget_net_loc(self):
        budget = adapter.ChangeBudget(
            path="x.py", loc_added=10, loc_removed=3, change_class=adapter.ChangeClass.GENERIC_BUG_FIX
        )
        assert budget.net_loc == 7


@pytest.mark.unit
class TestDummyBackendAdapter:
    def test_spy_records_calls(self):
        backend = adapter.DummyBackendAdapter()
        backend.capability()
        backend.route({"dtype": "float16"})
        backend.execute({"x": 1})
        kinds = [event.kind for event in backend.spy]
        assert kinds == ["capability", "route", "execute"]
        assert len(backend.as_dict()["spy_events"]) == 3

    def test_unsupported_dtype_routes_to_reference(self):
        backend = adapter.DummyBackendAdapter()
        decision = backend.route({"dtype": "float64"})
        assert decision["fallback_used"] is True
        assert decision["reason"] == "DTYPE"

    def test_performance_claim_is_always_refused(self):
        backend = adapter.DummyBackendAdapter()
        claim = backend.performance_claim_allowed()
        assert claim["allowed"] is False
        assert "DUMMY_BACKEND" in claim["reason"]

    def test_error_mapping_is_structured(self):
        mapping = adapter.DummyBackendAdapter().error_mapping()
        assert mapping["missing_artifact"] == "QUANT_ARTIFACT_MISMATCH"


@pytest.mark.unit
class TestAdapterRegistration:
    def test_global_default_change_refused(self):
        with pytest.raises(ConfigError):
            adapter.AdapterRegistration(
                target="second",
                kind="model",
                adapter_path="hqsb/integration/adapters/second_model.py",
                changes_global_default=True,
            )

    def test_unknown_kind_refused(self):
        with pytest.raises(ConfigError):
            adapter.AdapterRegistration(target="x", kind="sorcery", adapter_path="p")

    def test_registration_is_config_only(self):
        registration = adapter.AdapterRegistration(
            target="dummy", kind="backend", adapter_path="hqsb/integration/adapter.py"
        )
        assert registration.changes_global_default is False
        assert registration.as_dict()["kind"] == "backend"


@pytest.mark.unit
class TestReuse:
    def test_identity_collision_detected(self):
        report = adapter.identity_collision_check(
            {"model_id": "qwen", "graph_identity": "g"},
            {"model_id": "second", "graph_identity": "g"},
        )
        assert not report["ok"]
        assert report["collisions"] == ["graph_identity"]

    def test_reuse_matrix_template_covers_core_items(self):
        matrix = adapter.reuse_matrix_template()
        items = {row.item for row in matrix.rows}
        assert {"operator_schema", "graph_pass", "lowering_registry"} <= items
        assert matrix.status_counts["REUSED"] == len(matrix.rows)

    def test_invalid_reuse_status_refused(self):
        with pytest.raises(ConfigError):
            adapter.ReuseRow(item="x", status="MAYBE")

    def test_core_contract_snapshot_has_patterns_and_rules(self):
        snapshot = adapter.core_contract_snapshot()
        assert "hqsb.pattern.residual_add_rmsnorm" in snapshot.patterns
        assert "HC-MODEL-CLASS" in snapshot.forbidden_hardcode_rules
        assert len(snapshot.digest) == 64


@pytest.mark.unit
class TestErrorTaxonomy:
    def test_every_stage_has_at_least_one_reason_code(self):
        taxonomy_spec = taxonomy.frozen_taxonomy()
        assert taxonomy_spec.uncovered_stages() == ()
        assert len(taxonomy_spec.by_stage()) == len(taxonomy.ErrorStage.ALL)

    def test_unknown_reason_code_refused(self):
        with pytest.raises(SchemaError):
            taxonomy.frozen_taxonomy().resolve("SOMETHING_HAPPENED")

    def test_reason_codes_carry_user_messages(self):
        for spec in taxonomy.REASON_CODES:
            assert spec.user_message
            assert spec.stage in taxonomy.ErrorStage.ALL

    def test_non_screaming_reason_code_refused(self):
        with pytest.raises(SchemaError):
            taxonomy.ReasonCodeSpec("lower_case", taxonomy.ErrorStage.API_SCHEMA, user_message="x")

    def test_fatal_codes_forbid_fallback(self):
        fatal = [spec for spec in taxonomy.REASON_CODES if spec.severity == "fatal"]
        assert fatal
        assert all(spec.fallback_allowed is False for spec in fatal)

    def test_exception_classification(self):
        assert taxonomy.classify_exception(CapabilityError("x"))[0] == "CAPABILITY_UNSUPPORTED"
        assert taxonomy.classify_exception(ValueError("x"))[0] == "SCHEMA_INVALID_ARGUMENT"
        assert taxonomy.classify_exception(MemoryError())[0] == "DEVICE_OOM"

    def test_message_sanitisation_removes_paths_and_addresses(self):
        message = taxonomy.sanitize_message(
            "failed at /root/models/qwen/config.json addr 0x7f9c3a1b2c3d weight tensor"
        )
        assert "/root/" not in message
        assert "0x7f9c3a1b2c3d" not in message
        assert "<redacted-text>" in message


@pytest.mark.unit
class TestFailureRecord:
    def _record(self, **overrides) -> taxonomy.FailureRecord:
        payload = {
            "error_id": "e1",
            "reason_code": "INPUT_DTYPE_UNSUPPORTED",
            "stage": taxonomy.ErrorStage.INPUT_VALIDATION,
            "requested": "hqsb.compiled.fused",
            "actual": "torch.eager.reference",
            "message": "input dtype float64 unsupported",
            "final_result": "fallback_ok",
        }
        payload.update(overrides)
        return taxonomy.FailureRecord(**payload)

    def test_unknown_stage_refused(self):
        with pytest.raises(SchemaError):
            self._record(stage="somewhere")

    def test_sync_async_must_be_declared(self):
        with pytest.raises(ConfigError):
            self._record(sync_async="maybe")

    def test_deterministic_replay_ignores_volatile_ids(self):
        first = self._record(error_id="e1")
        second = self._record(error_id="e2")
        report = taxonomy.deterministic_failure([first, second])
        assert report["ok"] is True
        assert report["replays"] == 2

    def test_replay_difference_is_reported(self):
        first = self._record()
        second = self._record(actual="explicit_error")
        report = taxonomy.deterministic_failure([first, second])
        assert report["ok"] is False
        assert report["differences"][0]["field"] == "actual"

    def test_requested_actual_chain_is_complete(self):
        chain = taxonomy.requested_actual_chain(self._record())
        assert chain["complete"] is True
        assert chain["fallback"] == ""

    def test_hash_state_is_stable(self):
        payload = {"kv": "a", "tokens": "b"}
        assert taxonomy.hash_state(payload) == taxonomy.hash_state(dict(payload))


@pytest.mark.unit
class TestTransactionalPlan:
    def _plan(self, shadow=("kv_cache",)) -> taxonomy.TransactionalPlan:
        return taxonomy.TransactionalPlan(
            name="decode_step",
            invariants=taxonomy.kv_invariants({"kv_cache": "h0", "token_sequence": "t0"}),
            shadow_outputs=list(shadow),
        )

    def test_happy_path_commits(self):
        plan = self._plan()
        plan.validate()
        plan.prepare()
        plan.execute()
        plan.validate_completion({"kv_cache": "h0", "token_sequence": "t0"})
        payload = plan.commit()
        assert payload["state"] == taxonomy.TransactionState.COMMITTED
        assert plan.committed is True

    def test_invariant_violation_aborts_and_discards_shadow_outputs(self):
        plan = self._plan()
        plan.validate()
        plan.prepare()
        plan.execute()
        with pytest.raises(ConfigError):
            plan.validate_completion({"kv_cache": "h1", "token_sequence": "t1"})
        assert plan.state == taxonomy.TransactionState.ABORTED
        assert plan.shadow_outputs == []

    def test_abort_discards_partial_outputs(self):
        plan = self._plan()
        plan.validate()
        plan.prepare()
        plan.execute()
        payload = plan.abort("kernel failed")
        assert payload["discarded_shadow_outputs"] == ["kv_cache"]

    def test_out_of_order_transitions_refused(self):
        plan = self._plan()
        with pytest.raises(ConfigError):
            plan.execute()

    def test_plan_without_invariants_refused(self):
        with pytest.raises(ConfigError):
            taxonomy.TransactionalPlan(name="x", invariants=[])


@pytest.mark.unit
class TestFallbackRegistry:
    def test_requested_available_is_used(self):
        registry = taxonomy.FallbackRegistry()
        decision = registry.resolve("hqsb.eager.custom")
        assert decision["actual"] == "hqsb.eager.custom"
        assert decision["fallback_used"] is False

    def test_level_is_disabled_with_a_reason(self):
        registry = taxonomy.FallbackRegistry()
        with pytest.raises(ConfigError):
            registry.disable("hqsb.eager.custom", "")
        registry.disable("hqsb.eager.custom", "broken kernel")
        decision = registry.resolve("hqsb.eager.custom")
        assert decision["actual"] == "torch.compile.reference"
        assert any("disabled" in step for step in decision["path"])

    def test_chain_never_upgrades(self):
        registry = taxonomy.FallbackRegistry()
        decision = registry.resolve("torch.eager.reference")
        # The last level is explicit_error: a lower-priority request cannot be
        # silently promoted to a higher one.
        assert decision["actual"] in ("torch.eager.reference", "explicit_error")

    def test_unknown_level_refused(self):
        with pytest.raises(ConfigError):
            taxonomy.FallbackRegistry().disable("nope", "reason")

    def test_strict_mode_fails_loudly(self):
        registry = taxonomy.FallbackRegistry(strict=True)
        decision = registry.resolve("hqsb.compiled.fused")
        assert decision["reason"] == "STRICT_MODE_NO_FALLBACK"
        assert decision["actual"] == "hqsb.compiled.fused"

    def test_fully_disabled_chain_reaches_explicit_error(self):
        registry = taxonomy.FallbackRegistry()
        for level in registry.levels[:-1]:
            registry.disable(level.name, "injected")
        decision = registry.resolve("hqsb.compiled.fused")
        assert decision["actual"] == "explicit_error"
        assert decision["reason"] == "NO_AVAILABLE_IMPLEMENTATION"


@pytest.mark.unit
class TestABI:
    def _identity(self, **overrides) -> abi.BuildIdentity:
        base = abi.simulate_identity()
        return abi.simulate_identity(base, **overrides)

    def test_compatible_identity_passes(self):
        report = abi.CompatibilityMatrix().check(self._identity())
        assert report.ok
        assert report.pre_load is True

    def test_arch_mismatch_is_detected_before_load(self):
        report = abi.CompatibilityMatrix().check(
            self._identity(gpu_arch="sm_90", fatbin_targets=("sm_86",))
        )
        assert not report.ok
        assert "ARCH_UNSUPPORTED" in report.codes()

    def test_abi_mismatch_is_detected(self):
        report = abi.CompatibilityMatrix().check(
            self._identity(cxx_abi_version="3.0.0", fatbin_targets=("sm_86",))
        )
        assert not report.ok
        assert "ABI_MISMATCH" in report.codes()

    def test_unknown_field_is_refused_by_default(self):
        identity = abi.simulate_identity(
            abi.simulate_identity(), unknown=("gpu_arch",), fatbin_targets=("sm_86",)
        )
        report = abi.CompatibilityMatrix().check(identity)
        assert not report.ok
        assert any(item["actual"] == "<unknown>" for item in report.failures)

    def test_unknown_field_allowed_only_when_explicit(self):
        identity = abi.simulate_identity(
            abi.simulate_identity(), unknown=("gpu_arch",), fatbin_targets=("sm_86",)
        )
        report = abi.CompatibilityMatrix(allow_unknown_fields=True).check(identity)
        assert report.ok

    def test_unknown_simulated_field_refused(self):
        with pytest.raises(ConfigError):
            abi.simulate_identity(nonsense="x")

    def test_binary_manifest_hash_and_symbols(self, tmp_path):
        payload = tmp_path / "libhqsb.so"
        payload.write_bytes(b"binary")
        import hashlib

        digest = hashlib.sha256(b"binary").hexdigest()
        manifest = abi.BinaryManifest(
            path=str(payload),
            sha256=digest,
            target_arch="sm_86",
            abi_version="1",
            required_symbols=("hqsb_rms_norm_forward", "hqsb_query_build_arch"),
            provided_symbols=("hqsb_rms_norm_forward",),
        )
        assert manifest.verify_file()["ok"] is True
        audit = manifest.symbol_audit()
        assert audit["ok"] is False
        assert audit["missing"] == ["hqsb_query_build_arch"]

    def test_binary_hash_mismatch_detected(self, tmp_path):
        payload = tmp_path / "libhqsb.so"
        payload.write_bytes(b"binary")
        manifest = abi.BinaryManifest(
            path=str(payload), sha256="0" * 64, target_arch="sm_86", abi_version="1"
        )
        result = manifest.verify_file()
        assert result["ok"] is False
        assert result["reason_code"] == "ABI_MISMATCH"

    def test_missing_binary_detected(self, tmp_path):
        manifest = abi.BinaryManifest(
            path=str(tmp_path / "missing.so"), sha256="0" * 64, target_arch="sm_86", abi_version="1"
        )
        assert manifest.verify_file()["reason_code"] == "LINK_LOAD_FAILED"

    def test_preload_check_combines_identity_and_manifest(self, tmp_path):
        payload = tmp_path / "libhqsb.so"
        payload.write_bytes(b"binary")
        import hashlib

        manifest = abi.BinaryManifest(
            path=str(payload),
            sha256=hashlib.sha256(b"binary").hexdigest(),
            target_arch="sm_86",
            abi_version="1",
        )
        good = abi.preload_check(self._identity(), manifest)
        assert good["ok"] is True
        bad = abi.preload_check(self._identity(gpu_arch="sm_90", fatbin_targets=("sm_86",)))
        assert bad["ok"] is False

    def test_wheel_tag_check(self):
        assert abi.py_abi_matches("3.12.3", "cp312") is True
        assert abi.py_abi_matches("3.10.0", "cp312") is False

    def test_identity_digest_is_stable(self):
        first = self._identity()
        second = self._identity()
        assert first.digest() == second.digest()
