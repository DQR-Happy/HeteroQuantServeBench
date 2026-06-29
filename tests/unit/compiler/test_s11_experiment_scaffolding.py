"""Coverage for the S11 scaffolding: prerequisites, run directory, verdict refusal.

These tests pin the stage-level discipline: the scaffolding never writes into
``docs/stage_experiments``, never emits a conclusion on its own, and every
prerequisite check carries its evidence pointer.
"""

from __future__ import annotations

import json
import os

import pytest

from hqsb.core.errors import ConfigError
from hqsb.compiler import experiment as exp
from hqsb.compiler import interface_map as imap
from hqsb.compiler import specs as spec_mod
from hqsb.compiler import telemetry as tm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SPEC_DIR = os.path.join(REPO_ROOT, "configs", "compiler")


@pytest.mark.unit
class TestPrerequisites:
    def test_status_lists_every_check_with_evidence(self) -> None:
        status = exp.check_prerequisites(REPO_ROOT)
        assert status.stage == "S11"
        names = [check.name for check in status.checks]
        assert names == [
            "s01_contracts_and_identity",
            "s02_qwen_reference_replayable",
            "s03_s04_kernel_hardware_evidence",
            "s06_pattern_model_level_correctness",
            "s05_quant_artifact_chain",
            "frozen_compiler_environment",
            "independent_evidence_dirs",
            "stable_cli_runner_for_gates",
            "ir_binary_export_capability",
        ]
        for check in status.checks:
            if check.satisfied:
                assert check.evidence, check.name
            else:
                assert check.reason, check.name

    def test_optional_checks_are_advisory_not_blocking(self) -> None:
        status = exp.check_prerequisites(REPO_ROOT)
        optional = {check.name for check in status.checks if not check.required}
        assert {"s05_quant_artifact_chain", "ir_binary_export_capability"} <= optional
        for name in optional:
            assert name not in status.missing

    def test_self_provided_checks_are_satisfied_by_this_stage(self) -> None:
        status = exp.check_prerequisites(REPO_ROOT)
        by_name = {check.name: check for check in status.checks}
        assert by_name["independent_evidence_dirs"].satisfied is True
        assert by_name["stable_cli_runner_for_gates"].satisfied is True

    def test_status_serialises_missing_and_advisory(self) -> None:
        payload = exp.check_prerequisites(REPO_ROOT).as_dict()
        assert isinstance(payload["missing"], list)
        assert isinstance(payload["advisory"], list)
        assert payload["satisfied"] is False or payload["missing"] == []


@pytest.mark.unit
class TestRunDirectory:
    def test_layout_is_created_under_experiment_results(self, tmp_path) -> None:
        run = exp.RunDirectory(str(tmp_path), "E11-02", "run1")
        path = run.create()
        assert path.endswith(os.path.join("experiment_results", "S11", "E11-02", "run1"))
        for sub in exp.RUN_LAYOUT:
            assert os.path.isdir(os.path.join(path, sub))

    def test_unknown_experiment_and_empty_run_id_are_rejected(self, tmp_path) -> None:
        with pytest.raises(ConfigError):
            exp.RunDirectory(str(tmp_path), "E11-99", "run")
        with pytest.raises(ConfigError):
            exp.RunDirectory(str(tmp_path), "E11-01", "")

    def test_report_skeleton_states_not_run(self, tmp_path) -> None:
        run = exp.RunDirectory(str(tmp_path), "E11-03", "run")
        run.create()
        path = run.write_report_skeleton("E11-03", "title")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        assert "NOT_RUN" in text
        assert "禁止填写任何" in text

    def test_raw_file_count_starts_at_zero(self, tmp_path) -> None:
        run = exp.RunDirectory(str(tmp_path), "E11-04", "run")
        run.create()
        assert run.raw_file_count() == 0


@pytest.mark.unit
class TestVerdictRefusal:
    def _status(self, satisfied: bool) -> exp.PrerequisiteStatus:
        return exp.PrerequisiteStatus(
            stage="S11",
            checks=[exp.PrerequisiteCheck(name="x", satisfied=satisfied, evidence="e", reason="")],
        )

    def test_status_without_execute_refuses_conclusions(self, tmp_path) -> None:
        run = exp.RunDirectory(str(tmp_path), "E11-01", "run")
        run.create()
        verdict = run.write_verdict(
            status=exp.STATUS_PASS,
            reason="candidate",
            prerequisites=self._status(True),
            executed=True,
            allow_execute=False,
            raw_samples=10,
        )
        assert verdict["status"] == exp.STATUS_BLOCKED
        assert "execution is disabled" in verdict["reason"]
        assert verdict["requested_status"] == exp.STATUS_PASS

    def test_missing_prerequisites_refuse_conclusions(self, tmp_path) -> None:
        run = exp.RunDirectory(str(tmp_path), "E11-01", "run")
        run.create()
        verdict = run.write_verdict(
            status=exp.STATUS_PASS,
            reason="candidate",
            prerequisites=self._status(False),
            executed=True,
            allow_execute=True,
            raw_samples=10,
        )
        assert verdict["status"] == exp.STATUS_BLOCKED
        assert "prerequisites unsatisfied" in verdict["reason"]

    def test_missing_raw_samples_refuse_conclusions(self, tmp_path) -> None:
        run = exp.RunDirectory(str(tmp_path), "E11-01", "run")
        run.create()
        verdict = run.write_verdict(
            status=exp.STATUS_FAIL,
            reason="candidate",
            prerequisites=self._status(True),
            executed=True,
            allow_execute=True,
            raw_samples=0,
        )
        assert verdict["status"] == exp.STATUS_BLOCKED
        assert "no raw samples" in verdict["reason"]

    def test_conclusion_is_written_when_everything_is_satisfied(self, tmp_path) -> None:
        run = exp.RunDirectory(str(tmp_path), "E11-01", "run")
        run.create()
        verdict = run.write_verdict(
            status=exp.STATUS_PASS_NEGATIVE,
            reason="preregistered negative result",
            prerequisites=self._status(True),
            executed=True,
            allow_execute=True,
            raw_samples=12,
        )
        assert verdict["status"] == exp.STATUS_PASS_NEGATIVE
        assert os.path.isfile(os.path.join(run.path, "verdict.json"))

    def test_unknown_status_is_rejected(self, tmp_path) -> None:
        run = exp.RunDirectory(str(tmp_path), "E11-01", "run")
        run.create()
        with pytest.raises(ConfigError):
            run.write_verdict(
                status="MAYBE",
                reason="r",
                prerequisites=self._status(True),
                executed=True,
                allow_execute=True,
                raw_samples=1,
            )

    def test_write_status_refuses_conclusion_statuses(self, tmp_path) -> None:
        run = exp.RunDirectory(str(tmp_path), "E11-01", "run")
        run.create()
        with pytest.raises(ConfigError):
            run.write_status(exp.STATUS_PASS, "r", self._status(True))
        payload = run.write_status(exp.STATUS_NOT_STARTED, "interface-only", self._status(False))
        assert payload["status"] == exp.STATUS_NOT_STARTED


@pytest.mark.unit
class TestPreregistrationAndManifest:
    def test_preregistration_requires_thresholds_and_non_claims(self) -> None:
        prereg = exp.Preregistration(
            experiment_id="E11-02",
            run_id="r",
            hypothesis="h",
            primary_metric="m",
            thresholds={},
            claim_boundary="b",
            non_claims=(),
        )
        problems = prereg.validate()
        assert any("thresholds" in problem for problem in problems)
        assert any("NOT claim" in problem for problem in problems)

    def test_preregistration_hash_is_stable(self) -> None:
        prereg = exp.Preregistration(
            experiment_id="E11-02",
            run_id="r",
            hypothesis="near-miss sites are rejected",
            primary_metric="false_positive_count",
            thresholds={"false_positive": 0},
            claim_boundary="rewrite legality only",
            non_claims=("no speedup claim",),
        )
        assert prereg.validate() == []
        assert prereg.prereg_hash() == prereg.prereg_hash()

    def test_experiment_record_requires_fallback_reason(self) -> None:
        record = exp.ExperimentRecord(
            experiment_id="E11-03",
            requested_implementation="cuda_v2",
            actual_implementation="reference",
        )
        assert any("fallback reason" in problem for problem in record.validate())
        record.fallback_reason = "GUARD_FALSE"
        assert record.validate() == []

    def test_evidence_manifest_validates_hashes_and_levels(self) -> None:
        manifest = exp.EvidenceManifest(
            run_id="r",
            raw_artifacts=({"uri": "raw/x.jsonl"},),
            ir_levels=("MAGIC",),
            claim_level="MAGIC",
        )
        problems = manifest.validate()
        assert any("uri/sha256" in problem for problem in problems)
        assert any("IR level" in problem for problem in problems)
        assert any("claim level" in problem for problem in problems)

    def test_evidence_manifest_carries_s11_identity_fields(self) -> None:
        manifest = exp.EvidenceManifest(
            run_id="r",
            capture_mode="dynamo_debug_backend",
            graph_identity="g",
            compile_identity="c",
            ir_levels=("HQSB_TARGETED",),
            cache_hit=False,
            selected_lowering="reference",
        )
        assert manifest.validate() == []
        payload = json.loads(manifest.to_json())
        assert payload["graph_identity"] == "g"
        assert payload["cache_hit"] is False


@pytest.mark.unit
class TestInterfaceMapAndSpecs:
    def test_interface_map_resolves_every_reference(self) -> None:
        result = imap.resolve_interfaces()
        assert result["experiments"] == 10
        assert result["steps"] == 320
        assert result["failures"] == []
        assert result["ok"] is True

    def test_every_step_has_at_least_one_interface(self) -> None:
        for mapping in imap.EXPERIMENTS:
            assert len(mapping.steps) == 32
            for step in mapping.steps:
                assert step.interfaces, (mapping.experiment_id, step.index)

    def test_mapping_lookup_and_tables(self) -> None:
        assert imap.mapping_for("E11-05").level == "P0"
        assert imap.total_steps() == 320
        assert "E11-10" in imap.mapping_table_markdown()
        assert "claim boundary" in imap.step_table_for("E11-01")
        with pytest.raises(ConfigError):
            imap.mapping_for("E11-99")

    def test_specs_load_and_audit(self) -> None:
        documents = spec_mod.CompilerSpecs.load(SPEC_DIR)
        reports = documents.audit()
        assert len(documents.documents) == len(spec_mod.KINDS)
        assert spec_mod.audit_all_ok(reports), [r for r in reports if not r["ok"]]

    def test_spec_loading_rejects_unknown_keys(self, tmp_path) -> None:
        path = tmp_path / "capture_spec.yaml"
        path.write_text("kind: hqsb.compiler.capture_spec\nname: x\nmystery: 1\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            spec_mod.load_yaml_document(str(path))

    def test_missing_spec_directory_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ConfigError):
            spec_mod.CompilerSpecs.load(str(tmp_path / "nope"))

    def test_spec_audit_detects_drift(self) -> None:
        documents = dict(spec_mod.CompilerSpecs.load(SPEC_DIR).documents)
        tampered = dict(documents[spec_mod.KIND_GUARD_SPEC])
        tampered["categories"] = list(tampered["categories"])[:-1]
        reports = spec_mod.audit_documents({**documents, spec_mod.KIND_GUARD_SPEC: tampered})
        guard_report = next(report for report in reports if report["kind"] == spec_mod.KIND_GUARD_SPEC)
        assert guard_report["ok"] is False
        assert any("categories" in problem for problem in guard_report["problems"])


@pytest.mark.unit
class TestTelemetryProjection:
    def test_c6_projection_requires_fallback_reason(self) -> None:
        fields = tm.S11ResultFields(
            compile_id="c",
            selected_lowering="cuda_v2",
            actual_lowering="reference",
        )
        report = tm.project_c6(fields)
        assert report["ok"] is False
        assert any("fallback reason" in problem for problem in report["problems"])

    def test_c6_projection_lists_empty_fields(self) -> None:
        report = tm.project_c6(tm.S11ResultFields(compile_id="c"))
        assert "run_id" in report["filled_fields"]
        assert "observed_kernel" in report["empty_fields"]
        assert report["ok"] is True

    def test_c7_projection_maps_kinds_and_span_chain(self) -> None:
        records = [
            tm.CompilerTraceRecord(kind="capture", trace_id="t", span_id="s1", timestamp_ns=1),
            tm.CompilerTraceRecord(
                kind="dispatch", trace_id="t", span_id="s2", timestamp_ns=2, parent_span_id="s1"
            ),
        ]
        report = tm.project_c7(records)
        assert report["ok"] is True
        assert report["kinds_covered"] == ["capture", "dispatch"]
        assert tm.span_chain_check(report["events"])["ok"] is True
        dangling = tm.span_chain_check(
            [{"span_id": "s", "parent_span_id": "missing"}]
        )
        assert dangling["ok"] is False

    def test_unknown_trace_kind_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            tm.CompilerTraceRecord(kind="magic", trace_id="t", span_id="s", timestamp_ns=1).to_event()

    def test_table_schemas_cover_every_s11_table(self) -> None:
        assert set(tm.S11_TABLE_SCHEMAS) >= {
            "compile_run",
            "pattern_decision",
            "guard_event",
            "autotune_trial",
            "cost_model_decision",
            "artifact_index",
            "lowering_decision",
            "dispatch_event",
            "cache_event",
            "capture_break",
            "guard_definition",
            "verifier_issue",
        }

    def test_table_row_validation_rejects_unknown_and_missing_keys(self) -> None:
        assert tm.validate_table_row("cache_event", {"layer": "pass_ir"}) == []
        assert tm.validate_table_row("cache_event", {"layer": "pass_ir", "mystery": 1})
        missing = tm.missing_fields("cache_event", {"layer": "pass_ir"})
        assert "kind" in missing

    def test_table_hashes_are_deterministic(self) -> None:
        tables = {"cache_event": [{"layer": "pass_ir", "kind": "hit"}]}
        assert tm.table_hashes(tables) == tm.table_hashes(tables)

    def test_coverage_summary_is_a_schema_statement(self) -> None:
        summary = tm.coverage_summary()
        assert summary["ok"] is True
        assert summary["table_count"] == len(tm.S11_TABLE_SCHEMAS)
        assert "not an experimental result" in summary["note"]

    def test_unknown_table_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            tm.validate_table_row("magic", {})
