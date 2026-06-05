"""Tests for the S06 experiment scaffolding and the interface map.

These tests are *interface-layer* evidence (maturity M2): they prove the
processes are callable, the contracts are enforced and the negative paths are
refused.  They do not execute any S06 experiment and produce no experiment
numbers.
"""

from __future__ import annotations

import json
import os

import pytest

from hqsb.core.errors import ConfigError
from hqsb.integration import experiment as exp
from hqsb.integration import interface_map as imap
from hqsb.integration import policies


@pytest.mark.unit
class TestPrerequisites:
    def test_prerequisites_are_inspected_not_assumed(self):
        status = exp.check_prerequisites(os.getcwd())
        assert status.stage == "S06"
        assert isinstance(status.satisfied, bool)
        for check in status.checks:
            if not check.satisfied:
                assert check.reason, "a missing prerequisite must explain itself"

    def test_s06_scaffolding_is_not_accepted_as_s04_5_evidence(self):
        """The S04.5 gate must require its own marker, not a directory."""
        status = exp.check_prerequisites(os.getcwd())
        by_name = {check.name: check for check in status.checks}
        check = by_name["s04_5_model_reintegration_evidence"]
        assert check.evidence == "" or check.evidence.endswith("s04_5_evidence.json")
        assert "NOT S04.5" in check.reason or check.satisfied

    def test_c1_c7_schema_check_is_a_real_code_check(self):
        status = exp.check_prerequisites(os.getcwd())
        by_name = {check.name: check for check in status.checks}
        assert by_name["c6_c7_schema_available"].satisfied is True

    def test_current_tree_is_blocked(self):
        status = exp.check_prerequisites(os.getcwd())
        # Upstream evidence (S04.5/S05) is absent in this tree.
        assert not status.satisfied
        assert "s05_p0_evidence" in status.missing

    def test_status_has_stage_and_checks(self):
        payload = exp.check_prerequisites(os.getcwd()).as_dict()
        assert payload["stage"] == "S06"
        assert "missing" in payload


@pytest.mark.unit
class TestPreregistration:
    def test_hash_is_stable(self):
        prereg = exp.Preregistration(
            experiment_id="E06-01",
            question="q",
            hypothesis="h",
            claim_boundary="no performance claim from this run",
        )
        assert prereg.prereg_hash == prereg.prereg_hash

    def test_unknown_experiment_refused(self):
        with pytest.raises(ConfigError):
            exp.Preregistration(
                experiment_id="E99-01",
                question="q",
                hypothesis="h",
                claim_boundary="b",
            )

    def test_empty_hypothesis_refused(self):
        with pytest.raises(ConfigError):
            exp.Preregistration(
                experiment_id="E06-01", question="q", hypothesis="", claim_boundary="b"
            )

    def test_missing_claim_boundary_refused(self):
        with pytest.raises(ConfigError):
            exp.Preregistration(experiment_id="E06-01", question="q", hypothesis="h")


@pytest.mark.unit
class TestRunDirectory:
    def test_layout_matches_the_protocol(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E06-01", "run1")
        path = run.create()
        assert os.path.isdir(os.path.join(path, "graph", "before"))
        assert os.path.isdir(os.path.join(path, "compile", "cache"))
        for sub in exp.RUN_LAYOUT:
            assert os.path.isdir(os.path.join(path, sub))

    def test_verdict_refused_by_default(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E06-01", "run1")
        run.create()
        prerequisites = exp.check_prerequisites(os.getcwd())
        payload = run.write_verdict(
            status=exp.STATUS_PASS,
            reason="test",
            prerequisites=prerequisites,
            executed=False,
            allow_execute=False,
        )
        assert payload["status"] == exp.STATUS_BLOCKED
        assert payload["requested_status"] == exp.STATUS_PASS

    def test_verdict_refused_without_raw_samples(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E06-01", "run1")
        run.create()
        prerequisites = exp.PrerequisiteStatus(stage="S06", checks=[])
        payload = run.write_verdict(
            status=exp.STATUS_FAIL,
            reason="test",
            prerequisites=prerequisites,
            executed=True,
            allow_execute=True,
            raw_samples=0,
        )
        assert payload["status"] == exp.STATUS_BLOCKED
        assert "raw samples" in payload["reason"]

    def test_status_records_non_conclusion(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E06-02", "run1")
        run.create()
        prerequisites = exp.check_prerequisites(os.getcwd())
        payload = run.write_status(exp.STATUS_BLOCKED, "no execution", prerequisites)
        assert payload["status"] == exp.STATUS_BLOCKED
        assert os.path.isfile(os.path.join(run.path, "status.json"))

    def test_report_skeleton_contains_no_numbers(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E06-07", "run1")
        run.create()
        path = run.write_report_skeleton("E06-07", "fusion/lowering")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        assert "NOT_RUN" in text
        assert "无。实验未执行" in text

    def test_evidence_manifest_carries_s06_fields(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E06-06", "run1")
        run.create()
        manifest = exp.EvidenceManifest(
            run_id="run1",
            experiment_id="E06-06",
            compile_mode="dynamo",
            graph_identity="g",
            compile_identity="c",
            cache_layer="dynamo_code",
            cache_hit=True,
            requested_lowering="hqsb.cuda.fused_add_rms_norm",
            actual_lowering="hqsb.cuda.fused_add_rms_norm",
            observed_kernel="hqsb_fused_add_rms_norm_v1",
            fallback_reason="",
        )
        path = run.write_evidence_manifest(manifest)
        payload = json.loads(open(path, encoding="utf-8").read())
        assert payload["graph_compile"]["cache_hit"] is True
        assert payload["graph_compile"]["observed_kernel"]

    def test_interface_only_run_is_blocked(self, tmp_path):
        report = exp.interface_only_run(str(tmp_path), "E06-01", run_id="smoke")
        assert report["status"] in (exp.STATUS_NOT_STARTED, exp.STATUS_BLOCKED)
        assert report["run_dir"].endswith("smoke")


@pytest.mark.unit
class TestInterfaceMap:
    def test_every_interface_resolves(self):
        report = imap.resolve_interfaces()
        assert report["ok"] is True
        assert report["failures"] == []
        assert report["experiments"] == [f"E06-{i:02d}" for i in range(1, 12)]
        assert report["steps"] == 218
        assert report["interfaces"] > 300

    def test_each_experiment_has_the_protocol_step_count(self):
        expected = {
            "E06-01": 20,
            "E06-02": 20,
            "E06-03": 20,
            "E06-04": 18,
            "E06-05": 20,
            "E06-06": 20,
            "E06-07": 20,
            "E06-08": 20,
            "E06-09": 20,
            "E06-10": 20,
            "E06-11": 20,
        }
        for experiment_id, count in expected.items():
            mapping = imap.mapping_for(experiment_id)
            assert len(mapping.steps) == count, experiment_id
            assert mapping.driver
            assert [step.step for step in mapping.steps] == list(range(1, count + 1))

    def test_eleven_experiments_total(self):
        assert len(imap.EXPERIMENT_MAP) == 11

    def test_every_step_has_interfaces(self):
        for mapping in imap.EXPERIMENT_MAP:
            for step in mapping.steps:
                assert step.interfaces, f"{mapping.experiment_id} step {step.step}"
                assert step.maturity == "M2"

    def test_markdown_table_is_nonempty(self):
        markdown = imap.mapping_table_markdown()
        assert "E06-01" in markdown
        assert "E06-11" in markdown
        assert "| 步骤 |" in markdown

    def test_resolve_interface_bad_reference(self):
        with pytest.raises(AttributeError):
            imap.resolve_interface("hqsb.integration.interface_map.nonexistent_symbol")

    def test_resolve_interface_supports_nested_attributes(self):
        assert imap.resolve_interface(
            "hqsb.integration.specs.OpSchema.schema_hash"
        ) is not None


@pytest.mark.unit
class TestConfigs:
    def test_all_shipped_configs_load(self):
        directory = policies.default_config_dir(os.getcwd())
        documents = policies.load_directory(directory)
        kinds = {item["kind"] for item in documents}
        assert kinds == set(policies.KINDS)

    def test_pattern_declarations_match_the_code(self):
        directory = policies.default_config_dir(os.getcwd())
        documents = policies.load_directory(directory)
        by_kind = {item["kind"]: item["object"] for item in documents}
        audit = policies.audit_pattern_declarations(by_kind[policies.KIND_PATTERN_SPECS])
        assert audit["ok"] is True

    def test_operator_contracts_match_the_frozen_schemas(self):
        directory = policies.default_config_dir(os.getcwd())
        documents = policies.load_directory(directory)
        by_kind = {item["kind"]: item["object"] for item in documents}
        schemas = policies.build_schemas(by_kind[policies.KIND_OPERATOR_CONTRACTS])
        assert {schema.name for schema in schemas} == {
            "hqsb::rms_norm",
            "hqsb::fused_add_rms_norm",
            "hqsb::dequant_linear",
        }

    def test_compile_policy_is_not_strict_with_fallback(self):
        directory = policies.default_config_dir(os.getcwd())
        document = policies.load_compile_policy(
            os.path.join(directory, "compile_policy.yaml")
        )
        assert document.compiler_config.policy == document.policy
        assert not (document.strict and document.fallback_enabled)
