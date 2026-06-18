"""S08 experiment scaffolding: gates, verdict refusal, interface map."""

from __future__ import annotations

import json
import os

import pytest

from hqsb.core.errors import ConfigError
from hqsb.serving import experiment as exp
from hqsb.serving import interface_map as imap
from hqsb.serving import specs

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@pytest.mark.unit
class TestPrerequisites:
    def test_all_experiments_are_known(self):
        assert set(exp.EXPERIMENTS) == {
            f"E08-{index:02d}" for index in range(1, 12)
        }

    def test_prerequisites_are_currently_blocked(self):
        status = exp.check_prerequisites(_REPO_ROOT)
        # In this repository state the hard prerequisites are unmet, so the stage
        # must NOT report satisfied.  Only the C6/C7 schema availability is
        # guaranteed by this interface layer itself.
        assert not status.satisfied
        assert "s07_p0_verdicts" in status.missing
        assert "two_backend_registry_evidence" in status.missing

    def test_prerequisites_do_not_self_unlock(self):
        # The evidence directory is under docs/stage_experiments/S08 — never under
        # experiment_results — so this interface layer cannot satisfy its own gate.
        assert exp.S08_EVIDENCE_DIR.startswith(os.path.join("docs", "stage_experiments"))
        assert exp.S08_EVIDENCE_DIR != os.path.join("experiment_results", "S08")


@pytest.mark.unit
class TestVerdictRefusal:
    def test_template_record_cannot_carry_a_conclusion(self):
        with pytest.raises(ConfigError):
            exp.template_experiment_record("E08-01", status="PASS", reason="?")

    def test_verdict_refuses_conclusion_without_execute(self, tmp_path):
        status = exp.PrerequisiteStatus(stage="S08", checks=[])
        run = exp.RunDirectory(str(tmp_path), "E08-01", "r0")
        run.create()
        verdict = run.write_verdict(
            status="PASS",
            reason="should be refused",
            prerequisites=status,
            executed=True,
            raw_samples=10,
            allow_execute=False,
        )
        assert verdict["status"] == exp.STATUS_BLOCKED
        assert "execution is disabled" in verdict["reason"]

    def test_verdict_refuses_conclusion_without_prerequisites(self, tmp_path):
        status = exp.PrerequisiteStatus(
            stage="S08",
            checks=[exp.PrerequisiteCheck(name="s07_p0_verdicts", satisfied=False)],
        )
        run = exp.RunDirectory(str(tmp_path), "E08-01", "r0")
        run.create()
        verdict = run.write_verdict(
            status="PASS",
            reason="should be refused",
            prerequisites=status,
            executed=True,
            raw_samples=10,
            allow_execute=True,
        )
        assert verdict["status"] == exp.STATUS_BLOCKED

    def test_verdict_refuses_conclusion_without_raw_samples(self, tmp_path):
        status = exp.PrerequisiteStatus(stage="S08", checks=[])
        run = exp.RunDirectory(str(tmp_path), "E08-01", "r0")
        run.create()
        verdict = run.write_verdict(
            status="PASS",
            reason="should be refused",
            prerequisites=status,
            executed=False,
            raw_samples=0,
            allow_execute=True,
        )
        assert verdict["status"] == exp.STATUS_BLOCKED

    def test_non_conclusion_status_is_writable(self, tmp_path):
        status = exp.PrerequisiteStatus(stage="S08", checks=[])
        run = exp.RunDirectory(str(tmp_path), "E08-01", "r0")
        run.create()
        payload = run.write_status(exp.STATUS_NOT_STARTED, "nothing ran", status)
        assert payload["status"] == exp.STATUS_NOT_STARTED


@pytest.mark.unit
class TestRunDirectory:
    def test_layout_has_all_directories(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E08-01", "r0")
        run.create()
        for name in exp.RUN_LAYOUT:
            assert os.path.isdir(os.path.join(run.path, name))

    def test_report_skeleton_cannot_be_mistaken_for_a_result(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E08-01", "r0")
        run.create()
        path = run.write_report_skeleton("E08-01", "title")
        text = open(path, encoding="utf-8").read()
        assert "NOT_RUN" in text
        assert "未执行" in text

    def test_record_and_manifest_round_trip(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E08-01", "r0")
        run.create()
        record = exp.template_experiment_record("E08-01", status="NOT_STARTED", reason="tpl")
        path = run.write_experiment_record(record)
        parsed = json.load(open(path, encoding="utf-8"))
        assert parsed["experiment_id"] == "E08-01"
        assert parsed["status"] == "NOT_STARTED"
        manifest = exp.EvidenceManifest(run_id="r0", service_id="s", service_version="0.1")
        mpath = run.write_evidence_manifest(manifest)
        assert json.load(open(mpath, encoding="utf-8"))["service"]["service_id"] == "s"


@pytest.mark.unit
class TestInterfaceMap:
    def test_all_264_steps_are_mapped(self):
        assert sum(len(mapping.steps) for mapping in imap.EXPERIMENTS) == 264

    def test_every_referenced_symbol_resolves(self):
        report = imap.resolve_interfaces()
        assert report["ok"], report["failures"][:10]
        assert report["steps"] == 264
        assert report["experiments"] == 11

    def test_markdown_table_is_emitted(self):
        table = imap.mapping_table_markdown()
        assert "| 实验 |" in table
        assert "264" in table


@pytest.mark.unit
class TestSpecs:
    def test_specs_load_and_audit(self):
        loaded = specs.ServingSpecs.load(os.path.join(_REPO_ROOT, "configs", "serving"))
        assert len(loaded.documents) == 12
        assert loaded.ok, [
            audit for audit in loaded.audits if not audit["ok"]
        ]

    def test_specs_reject_unknown_keys(self, tmp_path):
        import yaml

        path = os.path.join(tmp_path, "bad.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "kind": specs.KIND_PROTOCOL_PROFILE,
                    "profile_version": "x",
                    "bogus_key": True,
                },
                handle,
            )
        with pytest.raises(ConfigError):
            specs.load_yaml_document(path)

    def test_specs_refuse_missing_document(self, tmp_path):
        with pytest.raises(ConfigError):
            specs.ServingSpecs.load(str(tmp_path))
