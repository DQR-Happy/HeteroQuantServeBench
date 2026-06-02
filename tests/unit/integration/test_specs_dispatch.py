"""Schema and dispatcher boundary tests (E06-01 interface layer)."""

from __future__ import annotations

import os

import pytest

from hqsb.core.errors import ConfigError, RegistryError, SchemaError
from hqsb.integration import dispatch, specs


@pytest.mark.unit
class TestOperatorSchema:
    def test_frozen_schemas_are_valid_and_stable(self):
        schemas = specs.frozen_schemas()
        assert len(schemas) == 3
        for schema in schemas:
            schema.validate()
            assert schema.schema_hash == schema.schema_hash
            assert len(schema.schema_hash) == 64

    def test_schema_hash_changes_with_mutation(self):
        base = specs.fused_add_rms_norm_schema()
        mutated = specs.OpSchema(
            name=base.name,
            overload=base.overload,
            args=base.args,
            returns=base.returns,
            mutation=(specs.MutationSpec("residual", "in_place"),),
            aliases=(
                specs.AliasSpec(0, None, "no_alias"),
                specs.AliasSpec(1, "residual", "reuse_input"),
            ),
            inference_only=base.inference_only,
            autograd_policy=base.autograd_policy,
            autocast_policy=base.autocast_policy,
        )
        assert mutated.schema_hash != base.schema_hash
        mutated.validate()

    def test_validate_refuses_alias_without_mutation(self):
        schema = specs.OpSchema(
            name="hqsb::bad",
            overload="default",
            args=(specs.ArgSpec("x", "Tensor"),),
            returns=(specs.ReturnSpec("out"),),
            mutation=(),
            aliases=(specs.AliasSpec(0, "x", "reuse_input"),),
            inference_only=True,
            autograd_policy="INFERENCE_ONLY_ERROR",
            autocast_policy="FOLLOW_INPUT",
        )
        with pytest.raises(SchemaError):
            schema.validate()

    def test_validate_refuses_missing_alias_declaration(self):
        schema = specs.OpSchema(
            name="hqsb::bad2",
            overload="default",
            args=(specs.ArgSpec("x", "Tensor"),),
            returns=(specs.ReturnSpec("out"),),
            aliases=(),
            inference_only=True,
            autograd_policy="INFERENCE_ONLY_ERROR",
            autocast_policy="FOLLOW_INPUT",
        )
        with pytest.raises(SchemaError):
            schema.validate()

    def test_unknown_argument_kind_refused(self):
        with pytest.raises(SchemaError):
            specs.ArgSpec("x", "Tensor3D")

    def test_autocast_policy_requires_dtype_when_casting(self):
        with pytest.raises(SchemaError):
            specs.OpSchema(
                name="hqsb::bad3",
                overload="default",
                args=(specs.ArgSpec("x", "Tensor"),),
                returns=(specs.ReturnSpec("out"),),
                aliases=(specs.AliasSpec(0, None, "no_alias"),),
                autograd_policy="INFERENCE_ONLY_ERROR",
                autocast_policy="CAST_INPUTS_TO",
            )

    def test_registry_snapshot_digest_is_stable(self):
        first = specs.schema_registry_snapshot().digest
        second = specs.schema_registry_snapshot().digest
        assert first == second
        assert len(specs.schema_registry_snapshot().as_dict()) == 3

    def test_schema_from_operator_spec_requires_contract_fields(self):
        spec = {"name": "rms_norm", "inputs": [{"name": "x"}], "outputs": [{"name": "out"}]}
        with pytest.raises(SchemaError) as excinfo:
            specs.schema_from_operator_spec(spec)
        assert "contract" in str(excinfo.value)

    def test_schema_from_operator_spec_builds_a_valid_schema(self):
        spec = {
            "name": "rms_norm",
            "semantic_version": "1.0.0",
            "inputs": [{"name": "x", "dtype": "float16"}],
            "outputs": [{"name": "out", "dtype": "float16"}],
        }
        contract = {
            "mutation": [],
            "alias": [{"output_index": 0, "input_name": None, "kind": "no_alias"}],
            "inference_only": True,
            "autograd_policy": "INFERENCE_ONLY_ERROR",
            "autocast_policy": "FOLLOW_INPUT",
        }
        schema = specs.schema_from_operator_spec(spec, contract)
        schema.validate()
        assert schema.name == "hqsb::rms_norm"


@pytest.mark.unit
class TestSchemaOwnerAudit:
    def test_scan_reports_owners_and_fragments(self, tmp_path):
        python_file = tmp_path / "ops.py"
        python_file.write_text(
            "import torch\n"
            "torch.library.define('hqsb::rms_norm', '(Tensor x) -> Tensor')\n"
            "lib = torch.library.Library('hqsb', 'IMPL')\n",
            encoding="utf-8",
        )
        cpp_file = tmp_path / "ops.cpp"
        cpp_file.write_text(
            'TORCH_LIBRARY(hqsb, m) { m.def("rms_norm(Tensor x) -> Tensor"); }\n'
            'TORCH_LIBRARY_FRAGMENT(hqsb, m) { m.def("rms_norm_impl(Tensor x) -> Tensor"); }\n',
            encoding="utf-8",
        )
        audit = specs.audit_schema_owners([str(tmp_path)])
        assert audit.owners["hqsb::rms_norm"]
        assert any(site.owner_kind == "fragment" for site in audit.sites)

    def test_conflicting_owners_are_reported(self, tmp_path):
        for name in ("a.py", "b.py"):
            (tmp_path / name).write_text(
                "import torch\n"
                "torch.library.define('hqsb::rms_norm', '(Tensor x) -> Tensor')\n",
                encoding="utf-8",
            )
        audit = specs.audit_schema_owners([str(tmp_path)])
        assert audit.conflicts
        assert not audit.ok

    def test_missing_owner_is_refused(self, tmp_path):
        empty = tmp_path / "empty.py"
        empty.write_text("# no definition here\n", encoding="utf-8")
        audit = specs.audit_schema_owners([str(tmp_path)])
        with pytest.raises(ConfigError):
            specs.assert_single_owner([specs.rms_norm_schema()], audit)

    def test_real_tree_has_no_conflicting_owners(self):
        audit = specs.audit_schema_owners([os.path.join(os.getcwd(), "hqsb")])
        assert not audit.conflicts


@pytest.mark.unit
class TestRegistrationMatrix:
    def test_frozen_matrix_has_no_conflicts(self):
        matrix = dispatch.frozen_p0_matrix()
        assert matrix.conflicts == []
        for schema in specs.frozen_schemas():
            assert matrix.schemas[schema.name] == schema.schema_hash

    def test_duplicate_same_implementation_is_idempotent(self):
        matrix = dispatch.frozen_p0_matrix()
        record = matrix.records[0]
        outcome = matrix.register(record)
        assert outcome.action == dispatch.RegistrationAction.IDEMPOTENT

    def test_duplicate_different_implementation_is_rejected(self):
        matrix = dispatch.frozen_p0_matrix()
        outcome = matrix.register(
            dispatch.RegistrationRecord(
                qualified_name=specs.OP_RMS_NORM,
                key=dispatch.DispatchKey.CUDA,
                implementation="some_other_kernel",
                provider="cuda_shared_lib",
                library="libhqsb_ops",
                schema_hash=matrix.schemas[specs.OP_RMS_NORM],
            )
        )
        assert outcome.action == dispatch.RegistrationAction.REJECTED_DUPLICATE_CONFLICT
        assert not outcome.accepted

    def test_same_name_different_schema_is_rejected(self):
        matrix = dispatch.RegistrationMatrix()
        matrix.declare_schema("hqsb::rms_norm", "hash-a")
        outcome = matrix.declare_schema("hqsb::rms_norm", "hash-b")
        assert outcome.action == dispatch.RegistrationAction.REJECTED_SCHEMA_MISMATCH

    def test_namespace_collision_is_rejected(self):
        matrix = dispatch.RegistrationMatrix()
        outcome = matrix.register(
            dispatch.RegistrationRecord(
                qualified_name="hqsb::squatter",
                key=dispatch.DispatchKey.CUDA,
                implementation="third_party",
                provider="python",
                library="elsewhere",
                owner="third_party",
            )
        )
        assert outcome.action == dispatch.RegistrationAction.REJECTED_NAMESPACE_COLLISION

    def test_implementation_built_against_other_schema_is_rejected(self):
        matrix = dispatch.frozen_p0_matrix()
        outcome = matrix.register(
            dispatch.RegistrationRecord(
                qualified_name=specs.OP_RMS_NORM,
                key=dispatch.DispatchKey.CUDA,
                implementation="k",
                provider="cuda_shared_lib",
                library="libhqsb_ops",
                schema_hash="0" * 64,
            )
        )
        assert outcome.action == dispatch.RegistrationAction.REJECTED_SCHEMA_MISMATCH

    def test_unknown_dispatch_key_refused(self):
        with pytest.raises(RegistryError):
            dispatch.RegistrationRecord(
                qualified_name="hqsb::x",
                key="CUDA_BUT_FAKE",
                implementation="i",
                provider="python",
                library="l",
            )


@pytest.mark.unit
class TestDispatchSnapshots:
    def test_snapshot_diff_detects_changes(self):
        before = dispatch.snapshot_from_matrix("before", dispatch.frozen_p0_matrix())
        matrix = dispatch.frozen_p0_matrix()
        matrix.register(
            dispatch.RegistrationRecord(
                qualified_name=specs.OP_DEQUANT_LINEAR,
                key=dispatch.DispatchKey.CPU,
                implementation="cpu_dequant_reference",
                provider="cpp",
                library="libhqsb_ops",
            )
        )
        after = dispatch.snapshot_from_matrix("after", matrix)
        diff = before.diff(after)
        assert diff["before"] != diff["after"]
        assert specs.OP_DEQUANT_LINEAR in diff["changed"]

    def test_snapshot_digest_is_stable(self):
        first = dispatch.snapshot_from_matrix("s", dispatch.frozen_p0_matrix()).digest
        second = dispatch.snapshot_from_matrix("s", dispatch.frozen_p0_matrix()).digest
        assert first == second

    def test_opcheck_plan_covers_all_operators(self):
        plan = dispatch.opcheck_plan()
        names = {item.subtest for item in plan}
        assert {"schema_check", "test_schema", "test_faketensor"} <= names
        assert len(plan) >= 3 * 5


@pytest.mark.unit
class TestSelectionAndRedispatch:
    def _request(self, **overrides):
        payload = {
            "op": specs.OP_RMS_NORM,
            "dtype": "float16",
            "layout": "contiguous",
            "rank": 2,
            "device": "cuda",
            "arch": "sm_86",
        }
        payload.update(overrides)
        return dispatch.CapabilityRequest(**payload)

    def test_capability_driven_selection(self):
        capability = dispatch.OperatorCapability(
            op=specs.OP_RMS_NORM,
            provider="cuda_shared_lib",
            dtypes=("float16",),
            arch=("sm_86",),
            kernel_symbol="hqsb_rms_norm_v2",
        )
        result = dispatch.select_implementation(
            dispatch.frozen_p0_matrix(), capability, self._request()
        )
        assert result.fallback_used is False
        assert result.selected == "hqsb_cuda_rms_norm_dispatch"
        assert result.key == dispatch.DispatchKey.CUDA

    def test_unsupported_dtype_falls_back_to_composite(self):
        capability = dispatch.OperatorCapability(
            op=specs.OP_RMS_NORM,
            provider="cuda_shared_lib",
            dtypes=("float16",),
        )
        result = dispatch.select_implementation(
            dispatch.frozen_p0_matrix(), capability, self._request(dtype="float64")
        )
        assert result.fallback_used is True
        assert result.key == dispatch.DispatchKey.COMPOSITE_EXPLICIT_AUTOGRAD
        assert result.reason == "DTYPE"

    def test_choose_key_maps_devices(self):
        assert dispatch.choose_key(self._request(device="cpu")) == dispatch.DispatchKey.CPU
        assert dispatch.choose_key(self._request(device="meta")) == dispatch.DispatchKey.META
        assert dispatch.choose_key(self._request()) == dispatch.DispatchKey.CUDA

    def test_redispatch_keyset_drops_handled_keys(self):
        keyset = frozenset({dispatch.DispatchKey.CUDA, dispatch.DispatchKey.AUTOCAST_CUDA})
        remaining = dispatch.redispatch_keyset(keyset, frozenset({dispatch.DispatchKey.CUDA}))
        assert remaining == frozenset({dispatch.DispatchKey.AUTOCAST_CUDA})

    def test_redispatch_exhaustion_is_refused(self):
        keyset = frozenset({dispatch.DispatchKey.CUDA})
        with pytest.raises(dispatch.RedispatchError):
            dispatch.redispatch_keyset(keyset, keyset)

    def test_redispatch_recursion_is_refused(self):
        guard = dispatch.RedispatchGuard()
        keyset = frozenset({dispatch.DispatchKey.CUDA})
        with pytest.raises(dispatch.RedispatchError):
            with guard.enter("hqsb::rms_norm", keyset):
                with guard.enter("hqsb::rms_norm", keyset):
                    pass

    def test_redispatch_depth_is_bounded(self):
        guard = dispatch.RedispatchGuard(max_depth=1)
        with pytest.raises(dispatch.RedispatchError):
            with guard.enter("hqsb::a", frozenset({"CUDA"})):
                with guard.enter("hqsb::b", frozenset({"CPU"})):
                    pass


@pytest.mark.unit
class TestFallbackPolicy:
    def test_requested_available_is_not_a_fallback(self):
        policy = dispatch.FallbackPolicy()
        decision = policy.resolve("hqsb.eager.custom", ["hqsb.eager.custom"])
        assert decision.actual == "hqsb.eager.custom"
        assert decision.fallback_used is False

    def test_missing_capability_walks_down_the_chain(self):
        policy = dispatch.FallbackPolicy()
        decision = policy.resolve("hqsb.compiled.fused", ["torch.eager.reference"])
        assert decision.actual == "torch.eager.reference"
        assert decision.fallback_used is True
        assert decision.reason.startswith("FALLBACK_TO")

    def test_strict_mode_refuses_to_fall_back(self):
        policy = dispatch.FallbackPolicy(strict=True)
        decision = policy.resolve("hqsb.compiled.fused", ["torch.eager.reference"])
        assert decision.actual == "hqsb.compiled.fused"
        assert decision.reason == "STRICT_MODE_NO_FALLBACK"

    def test_disabled_capability_is_skipped_with_a_reason(self):
        policy = dispatch.FallbackPolicy(disabled_capabilities=frozenset({"hqsb.eager.custom"}))
        decision = policy.resolve("hqsb.eager.custom", ["hqsb.eager.custom", "torch.eager.reference"])
        assert decision.actual == "torch.eager.reference"

    def test_no_available_implementation_is_explicit(self):
        policy = dispatch.FallbackPolicy()
        decision = policy.resolve("hqsb.compiled.fused", [])
        assert decision.actual == "explicit_error"
        assert decision.reason == "NO_AVAILABLE_IMPLEMENTATION"

    def test_fallback_disabled_never_degrades_silently(self):
        policy = dispatch.FallbackPolicy(enabled=False)
        decision = policy.resolve("hqsb.compiled.fused", ["torch.eager.reference"])
        assert decision.reason == "FALLBACK_DISABLED"
        assert decision.actual == "explicit_error"

    def test_resolution_is_deterministic(self):
        policy = dispatch.FallbackPolicy()
        first = policy.resolve("hqsb.compiled.fused", ["torch.eager.reference"]).as_dict()
        second = policy.resolve("hqsb.compiled.fused", ["torch.eager.reference"]).as_dict()
        assert first == second
