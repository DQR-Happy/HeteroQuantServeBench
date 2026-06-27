"""Coverage for the S11 IR, identity and lineage interfaces.

Covers E11-01 steps 7–10 (serialisation, source/shape/effect metadata),
E11-03 steps 3–7, 17–18 (schema, verifier, lineage) and the protocol §5/§6
identity contracts.
"""

from __future__ import annotations

import json
import sys

import pytest

from hqsb.core.errors import ConfigError
from hqsb.compiler import identity as ident
from hqsb.compiler import ir as ir

REQUIRED = dict(
    artifact_id="a1",
    ir_level=ident.IR_HQSB_TARGETED,
    format="json",
    schema_version="1.0.0",
    run_id="r1",
    compile_id="c1",
    source_commit="deadbeef",
    target_triple="x86_64-linux-gnu",
    target_arch="sm_86",
    canonical_hash="0" * 64,
)


@pytest.mark.unit
class TestCanonicalisation:
    def test_canonical_json_is_key_sorted(self) -> None:
        assert ident.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_content_hash_ignores_noise_keys_but_records_them(self) -> None:
        first, stripped_first = ident.content_hash({"op": "add", "timestamp": 1, "pid": 7})
        second, stripped_second = ident.content_hash({"op": "add", "timestamp": 2, "pid": 9})
        assert first == second
        assert "timestamp" in " ".join(stripped_first)
        assert "pid" in " ".join(stripped_second)

    def test_content_hash_normalises_addresses_and_temp_names(self) -> None:
        first, stripped = ident.content_hash({"code": "mov 0x7ffe1234, t12"})
        second, _ = ident.content_hash({"code": "mov 0x55aa9988, t99"})
        assert first == second
        assert stripped

    def test_semantic_keys_are_never_noise(self) -> None:
        assert ident.check_noise_keys_do_not_touch_semantics() == []

    def test_semantic_change_moves_the_hash(self) -> None:
        first, _ = ident.content_hash({"op": "add", "dtype": "fp16"})
        second, _ = ident.content_hash({"op": "add", "dtype": "fp32"})
        assert first != second


@pytest.mark.unit
class TestVersionFreeze:
    def test_vague_versions_are_rejected(self) -> None:
        for value in ("", "latest", "main", "HEAD", "unknown", "stable"):
            assert ident.version_is_frozen(value) is False

    def test_numeric_versions_are_accepted(self) -> None:
        for value in ("2.8.0+cu128", "3.4.0", "12.4.131", "v1.2"):
            assert ident.version_is_frozen(value) is True

    def test_require_frozen_versions_lists_offenders(self) -> None:
        offenders = ident.require_frozen_versions({"torch": "2.8.0", "triton": "latest"})
        assert offenders == ["triton"]

    def test_dirty_patch_hash_ignores_diff_headers(self) -> None:
        text = "index 123..456\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n+change\n"
        assert ident.normalized_dirty_patch_hash(text) == ident.sha256_text("+change")


@pytest.mark.unit
class TestArtifactIdentity:
    def test_valid_identity_passes(self) -> None:
        assert ident.ArtifactIdentity(**REQUIRED).validate() == []

    def test_missing_required_field_is_reported(self) -> None:
        payload = dict(REQUIRED)
        payload["target_arch"] = ""
        problems = ident.ArtifactIdentity(**payload).validate()
        assert any("target_arch" in problem for problem in problems)

    def test_unknown_ir_level_is_rejected(self) -> None:
        payload = dict(REQUIRED, ir_level="MAGIC_IR")
        assert any("ir_level" in problem for problem in ident.ArtifactIdentity(**payload).validate())

    def test_bad_hash_format_is_rejected(self) -> None:
        payload = dict(REQUIRED, canonical_hash="not-a-hash")
        assert any("canonical_hash" in problem for problem in ident.ArtifactIdentity(**payload).validate())

    def test_volatile_level_requires_canonical_hash(self) -> None:
        payload = dict(REQUIRED, ir_level=ident.IR_PTX, canonical_hash="", raw_hash="a" * 64)
        assert any("volatile" in problem for problem in ident.ArtifactIdentity(**payload).validate())

    def test_identity_hash_excludes_volatile_fields(self) -> None:
        first = ident.ArtifactIdentity(**REQUIRED, created_at="t1", command="cmd1")
        second = ident.ArtifactIdentity(**REQUIRED, created_at="t2", command="cmd2")
        assert first.identity_hash() == second.identity_hash()

    def test_identity_hash_tracks_semantics(self) -> None:
        first = ident.ArtifactIdentity(**dict(REQUIRED, target_arch="sm_86"))
        second = ident.ArtifactIdentity(**dict(REQUIRED, target_arch="sm_87"))
        assert first.identity_hash() != second.identity_hash()

    def test_from_payload_rejects_unknown_keys(self) -> None:
        with pytest.raises(ConfigError):
            ident.ArtifactIdentity.from_payload({**REQUIRED, "mystery": 1})

    def test_artifact_index_rows_carry_identity_hash(self) -> None:
        rows = ident.artifact_index_rows([ident.ArtifactIdentity(**REQUIRED)])
        assert rows[0]["identity_hash"] == ident.ArtifactIdentity(**REQUIRED).identity_hash()


@pytest.mark.unit
class TestLineage:
    def _graph(self) -> ident.LineageGraph:
        graph = ident.LineageGraph()
        graph.add(ident.ArtifactIdentity(artifact_id="src", ir_level=ident.IR_SOURCE, format="py",
                                         schema_version="1.0.0", run_id="r", compile_id="c",
                                         source_commit="x", target_triple="t", target_arch="a",
                                         canonical_hash="1" * 64))
        graph.add(ident.ArtifactIdentity(artifact_id="fx", ir_level=ident.IR_DYNAMO_FX, format="json",
                                         schema_version="1.0.0", run_id="r", compile_id="c",
                                         source_commit="x", target_triple="t", target_arch="a",
                                         parent_artifact_ids=("src",), canonical_hash="2" * 64))
        graph.add(ident.ArtifactIdentity(artifact_id="tgt", ir_level=ident.IR_HQSB_TARGETED,
                                         format="json", schema_version="1.0.0", run_id="r",
                                         compile_id="c", source_commit="x", target_triple="t",
                                         target_arch="a", parent_artifact_ids=("fx",),
                                         canonical_hash="3" * 64))
        return graph

    def test_duplicate_artifact_id_is_rejected(self) -> None:
        graph = self._graph()
        with pytest.raises(ConfigError):
            graph.add(ident.ArtifactIdentity(**dict(REQUIRED, artifact_id="src")))

    def test_unknown_parent_is_reported(self) -> None:
        graph = ident.LineageGraph()
        graph.add(ident.ArtifactIdentity(**dict(REQUIRED, parent_artifact_ids=("missing",))))
        assert any("unknown parent" in problem for problem in graph.validate())

    def test_cycle_is_detected(self) -> None:
        graph = ident.LineageGraph()
        graph.add(ident.ArtifactIdentity(**dict(REQUIRED, artifact_id="x", parent_artifact_ids=("y",))))
        graph.add(ident.ArtifactIdentity(**dict(REQUIRED, artifact_id="y", parent_artifact_ids=("x",))))
        assert graph.cycles()
        assert any("cycle" in problem for problem in graph.validate())

    def test_coverage_reports_missing_levels(self) -> None:
        report = self._graph().coverage("tgt", (ident.IR_SOURCE, ident.IR_HQSB_CANONICAL))
        assert report["missing_levels"] == [ident.IR_HQSB_CANONICAL]
        assert report["complete"] is False

    def test_experiment_chain_status_uses_identity_chain(self) -> None:
        report = ident.identity_chain_status(self._graph(), "E11-02", "tgt")
        assert report["experiment_id"] == "E11-02"
        assert report["reachable_levels"] == [ident.IR_SOURCE, ident.IR_DYNAMO_FX, ident.IR_HQSB_TARGETED]

    def test_unknown_experiment_chain_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            ident.identity_required_levels_for_experiment("E11-99")


@pytest.mark.unit
class TestIdentities:
    def test_three_identities_are_distinct_and_stable(self) -> None:
        graph = ident.graph_identity(
            normalized_graph={"ops": ["add", "rmsnorm"]},
            op_schema_versions={"aten.add": "1.0.0"},
            constants_hash="c",
            tensor_metadata={"dtype": "fp16"},
            symbolic_constraints=("1<=B<=8",),
            effects=("functional",),
            capture_mode="dynamo_debug_backend",
            capture_version="2.8.0",
        )
        semantic = ident.semantic_identity(
            canonical_graph_identity=graph["digest"],
            model_artifact_id="qwen3-1.7b",
            quant_artifact_id="",
            numerical_contract="common_s06",
            state_contract="functional",
        )
        compile_key = ident.compile_identity(
            semantic_identity_digest=semantic["digest"],
            pass_pipeline=[{"name": "fuse", "version": "1.0.0"}],
            pass_options={},
            lowering_registry_digest="reg",
            selected_candidate_id="reference",
            kernel_build_ids=("build1",),
            compiler_versions={"torch": "2.8.0"},
            compiler_flags=("-O3",),
            target_triple="x86_64",
            target_arch="sm_86",
            target_features=("tensor_core",),
            abi_version="1",
            guard_domain="B<=8",
            autotune_id="",
            cost_model_id="",
        )
        assert len({graph["digest"], semantic["digest"], compile_key["digest"]}) == 3
        assert "audit key" in graph["note"]
        assert ident.compile_identity(
            semantic_identity_digest=semantic["digest"],
            pass_pipeline=[{"name": "fuse", "version": "1.0.0"}],
            pass_options={},
            lowering_registry_digest="reg",
            selected_candidate_id="reference",
            kernel_build_ids=("build1",),
            compiler_versions={"torch": "2.8.0"},
            compiler_flags=("-O3",),
            target_triple="x86_64",
            target_arch="sm_86",
            target_features=("tensor_core",),
            abi_version="1",
            guard_domain="B<=8",
            autotune_id="",
            cost_model_id="",
        )["digest"] == compile_key["digest"]

    def test_field_diff_lists_changed_fields(self) -> None:
        rows = ident.identity_field_diff({"a": 1, "b": 2}, {"a": 1, "b": 3})
        assert rows == [{"field": "b", "before": 2, "after": 3}]


@pytest.mark.unit
class TestIRTypes:
    def test_symbolic_dim_validation(self) -> None:
        good = ir.SymbolicDim(name="B", expression="B", lower=1, upper=8)
        assert good.validate() == []
        assert any("lower > upper" in p for p in ir.SymbolicDim(name="B", expression="B", lower=9, upper=1).validate())
        assert any("origin" in p for p in ir.SymbolicDim(name="B", expression="B", lower=1, upper=2, origin="wish").validate())
        assert any("unbounded" in p for p in ir.SymbolicDim(name="B", expression="B").validate())

    def test_constraint_validation(self) -> None:
        assert ir.ShapeConstraint("c", "H % 8 == 0", "variant", "kernel").validate() == []
        assert any(
            "category" in p for p in ir.ShapeConstraint("c", "H>0", "vibes", "kernel").validate()
        )

    def test_tensor_type_validation(self) -> None:
        assert ir.TensorType(dtype="fp16", dims=("B", "H"), stride=(8, 1)).validate() == []
        assert any("dtype" in p for p in ir.TensorType(dtype="float128").validate())
        assert any("layout" in p for p in ir.TensorType(dtype="fp16", layout="magic").validate())
        assert any(
            "same rank" in p
            for p in ir.TensorType(dtype="fp16", dims=("B", "H"), stride=(1,)).validate()
        )

    def test_effect_risk_states(self) -> None:
        safe = ir.EffectSet(evidence="schema", mutation="none")
        assert safe.risk_state() == "PROVED_SAFE"
        unsafe = ir.EffectSet(evidence="schema", mutation="inplace", writes=("ph_r",))
        assert unsafe.risk_state() == "PROVED_UNSAFE"
        unknown = ir.EffectSet(evidence="unknown", mutation="unknown", state_reason="schema gap")
        assert unknown.risk_state() == "UNKNOWN_UNSAFE"
        assert unknown.validate() == []

    def test_unknown_effects_require_a_reason(self) -> None:
        assert any("reason" in p for p in ir.EffectSet(evidence="unknown", mutation="unknown").validate())


def _linear_graph(graph_id: str = "g") -> ir.IRGraph:
    return ir.import_op_sequence(
        graph_id=graph_id,
        ops=[
            {
                "op_id": "o1",
                "semantic_op": "aten.add",
                "operands": ["x", "y"],
                "results": ["s"],
                "source": {"module_path": "fixture"},
                "mutation": "none",
                "effect_evidence": "schema",
            },
            {
                "op_id": "o2",
                "semantic_op": "aten.mul",
                "operands": ["s", "w"],
                "results": ["z"],
                "source": {"module_path": "fixture"},
                "mutation": "none",
                "effect_evidence": "schema",
            },
        ],
        inputs=[{"value_id": "x"}, {"value_id": "y"}, {"value_id": "w"}],
        outputs=["z"],
        constraints=(ir.ShapeConstraint("c1", "1 <= B <= 8", "variant", "user"),),
    )


@pytest.mark.unit
class TestVerifier:
    def test_valid_graph_passes(self) -> None:
        report = ir.verify_graph(_linear_graph())
        assert report.ok, report.as_dict()

    def test_cycle_is_rejected(self) -> None:
        graph = ir.IRGraph(
            graph_id="cyc",
            level=ident.IR_HQSB_CANONICAL,
            ops=(
                ir.IROp(
                    op_id="a",
                    semantic_op="aten.add",
                    operands=("b_out", "x"),
                    results=("a_out",),
                    source={"module_path": "f"},
                ),
                ir.IROp(
                    op_id="b",
                    semantic_op="aten.add",
                    operands=("a_out", "x"),
                    results=("b_out",),
                    source={"module_path": "f"},
                ),
            ),
            values=(
                ir.IRValue(value_id="x", is_input=True),
                ir.IRValue(value_id="a_out", producer="a"),
                ir.IRValue(value_id="b_out", producer="b"),
            ),
            inputs=("x",),
            outputs=("a_out",),
            constraints=(ir.ShapeConstraint("c", "B>0", "semantic", "user"),),
        )
        report = ir.verify_graph(graph)
        assert any(issue.code == "CYCLE" for issue in report.issues)

    def test_undefined_operand_is_reported(self) -> None:
        graph = ir.import_op_sequence(
            graph_id="undef",
            ops=[
                {
                    "op_id": "o1",
                    "semantic_op": "aten.add",
                    "operands": ["ghost", "x"],
                    "results": ["y"],
                    "source": {"module_path": "f"},
                }
            ],
            inputs=[{"value_id": "x"}],
            outputs=["y"],
            constraints=(ir.ShapeConstraint("c", "B>0", "semantic", "user"),),
        )
        assert any(issue.code == "UNDEFINED_OPERAND" for issue in ir.verify_graph(graph).issues)

    def test_missing_constraints_block_compilation(self) -> None:
        graph = ir.IRGraph(
            graph_id="noconstraints",
            level=ident.IR_HQSB_CANONICAL,
            ops=_linear_graph().ops,
            values=_linear_graph().values,
            inputs=_linear_graph().inputs,
            outputs=_linear_graph().outputs,
        )
        report = ir.verify_graph(graph)
        assert any(issue.code == "CONSTRAINTS_MISSING" for issue in report.issues)
        assert report.ok is False

    def test_targeted_graph_requires_candidates_selection_and_fallback(self) -> None:
        base = _linear_graph("targeted")
        graph = ir.IRGraph(
            graph_id=base.graph_id,
            level=ident.IR_HQSB_TARGETED,
            ops=base.ops,
            values=base.values,
            inputs=base.inputs,
            outputs=base.outputs,
            constraints=base.constraints,
        )
        codes = {issue.code for issue in ir.verify_graph(graph).issues}
        assert {"NO_CANDIDATES", "NO_SELECTION", "NO_FALLBACK"} <= codes

    def test_duplicate_and_dangling_ids_are_reported(self) -> None:
        graph = ir.IRGraph(
            graph_id="dup",
            level=ident.IR_HQSB_CANONICAL,
            ops=(
                ir.IROp(op_id="o", semantic_op="aten.add", operands=("x",), results=("y",), source={"module_path": "f"}),
                ir.IROp(op_id="o", semantic_op="aten.add", operands=("x",), results=("y",), source={"module_path": "f"}),
            ),
            values=(
                ir.IRValue(value_id="x", is_input=True),
                ir.IRValue(value_id="y", producer="o"),
                ir.IRValue(value_id="orphan"),
            ),
            inputs=("x",),
            outputs=("y",),
            constraints=(ir.ShapeConstraint("c", "B>0", "semantic", "user"),),
        )
        codes = {issue.code for issue in ir.verify_graph(graph).issues}
        assert "DUPLICATE_ID" in codes
        assert "DANGLING_VALUE" in codes


@pytest.mark.unit
class TestSerializationAndDiff:
    def test_round_trip_preserves_canonical_hash(self) -> None:
        graph = _linear_graph()
        rebuilt, same = ir.round_trip(graph)
        assert same is True
        assert rebuilt.canonical_hash()[0] == graph.canonical_hash()[0]

    def test_unknown_keys_are_rejected_on_load(self) -> None:
        payload = json.loads(_linear_graph().to_json())
        payload["mystery"] = 1
        with pytest.raises(ConfigError):
            ir.IRGraph.from_dict(payload)

    def test_text_dump_contains_ops_and_constraints(self) -> None:
        text = _linear_graph().to_text()
        assert "aten.add" in text and "constraint c1" in text

    def test_diff_detects_structure_and_constraints(self) -> None:
        before = _linear_graph("a")
        after = ir.IRGraph(
            graph_id="b",
            level=before.level,
            ops=before.ops[:-1],
            values=before.values,
            inputs=before.inputs,
            outputs=("s",),
            constraints=(),
        )
        diff = ir.diff_graphs(before, after)
        assert diff.ops_removed == ("o2",)
        assert diff.constraints_removed == ("c1",)
        assert diff.structure_changed is True

    def test_graph_summary_reports_counts_and_hashes(self) -> None:
        summary = ir.graph_summary(_linear_graph())
        assert summary["op_count"] == 2
        assert summary["canonical_hash"]

    def test_fx_importer_rejects_non_frontend_levels(self) -> None:
        with pytest.raises(ConfigError):
            ir.import_fx_graph(None, graph_id="g", level=ident.IR_HQSB_CANONICAL)

    def test_fx_importer_reports_missing_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ConfigError) as excinfo:
            ir.import_fx_graph(None, graph_id="g", level=ident.IR_DYNAMO_FX)
        assert excinfo.value.details.get("reason") == "NOT_INSTALLED"
