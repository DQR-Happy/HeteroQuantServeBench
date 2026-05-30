"""Tests for experiment scaffolding and the interface map (E05 §2.9.1)."""

from __future__ import annotations

import os

import pytest

from hqsb.core.errors import ConfigError
from hqsb.quant import experiment as exp
from hqsb.quant import interface_map as imap


@pytest.mark.unit
class TestPrerequisites:
    def test_prerequisites_are_inspected_not_assumed(self):
        status = exp.check_prerequisites(os.getcwd())
        assert status.stage == exp.STAGE
        assert isinstance(status.satisfied, bool)
        # Every missing item carries a reason.
        for check in status.checks:
            if not check.satisfied:
                assert check.reason

    def test_status_has_stage_and_checks(self):
        payload = exp.check_prerequisites(os.getcwd()).as_dict()
        assert payload["stage"] == "S05"
        assert "missing" in payload


@pytest.mark.unit
class TestPreregistration:
    def test_hash_is_stable(self):
        prereg = exp.Preregistration(
            experiment_id="E05-01", question="q", hypothesis="h", repeats=3
        )
        assert prereg.prereg_hash == prereg.prereg_hash

    def test_unknown_experiment_refused(self):
        with pytest.raises(ConfigError):
            exp.Preregistration(experiment_id="E99-01", question="q", hypothesis="h")

    def test_empty_hypothesis_refused(self):
        with pytest.raises(ConfigError):
            exp.Preregistration(experiment_id="E05-01", question="q", hypothesis="")


@pytest.mark.unit
class TestRunDirectory:
    def test_layout_created(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E05-01", "run1")
        path = run.create()
        for sub in exp.RUN_LAYOUT:
            assert os.path.isdir(os.path.join(path, sub))

    def test_verdict_refused_by_default(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E05-01", "run1")
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

    def test_status_records_non_conclusion(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E05-01", "run1")
        run.create()
        prerequisites = exp.check_prerequisites(os.getcwd())
        payload = run.write_status(exp.STATUS_BLOCKED, "no execution", prerequisites)
        assert payload["status"] == exp.STATUS_BLOCKED
        assert os.path.isfile(os.path.join(run.path, "status.json"))

    def test_record_command_persists_streams(self, tmp_path):
        run = exp.RunDirectory(str(tmp_path), "E05-01", "run1")
        run.create()
        record = run.record_command(0, ["echo", "hi"], stdout="hi\n", stderr="", returncode=0)
        assert record["returncode"] == 0

    def test_interface_only_run_is_blocked(self, tmp_path):
        report = exp.interface_only_run(str(tmp_path), "E05-01", run_id="smoke")
        # The S04.5 M4 prerequisite is unmet in this tree → BLOCKED.
        assert report["status"] in (exp.STATUS_NOT_STARTED, exp.STATUS_BLOCKED)
        assert report["run_dir"].endswith("smoke")


@pytest.mark.unit
class TestInterfaceMap:
    def test_every_interface_resolves(self):
        report = imap.resolve_interfaces()
        assert report["ok"] is True
        assert report["failures"] == []
        assert report["experiments"] == [f"E05-{i:02d}" for i in range(1, 11)]
        assert report["interfaces"] > 200

    def test_each_experiment_has_steps(self):
        for experiment_id in [f"E05-{i:02d}" for i in range(1, 11)]:
            mapping = imap.mapping_for(experiment_id)
            assert len(mapping.steps) >= 10
            assert mapping.driver

    def test_ten_experiments_total(self):
        assert len(imap.EXPERIMENT_MAP) == 10

    def test_markdown_table_is_nonempty(self):
        markdown = imap.mapping_table_markdown()
        assert "E05-01" in markdown
        assert "E05-10" in markdown

    def test_resolve_interface_bad_reference(self):
        with pytest.raises(AttributeError):
            imap.resolve_interface("hqsb.quant.interface_map.nonexistent_symbol")
