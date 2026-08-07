"""S09 evidence campaign: complete indexing without fabricated hardware claims."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hqsb.ascend import experiment as exp
from hqsb.ascend import interface_map
from hqsb.console.evidence import EvidenceCatalog


def _snapshot() -> exp.CapabilitySnapshot:
    manifest = {
        "schema_version": "1.0.0",
        "collected_at": "2026-09-21T00:00:00Z",
        "device_id": "UNAVAILABLE",
        "requested_backend": "ascend",
        "hardware": {
            "host_id": "fixture", "board_id": "fixture", "chip_sku": "UNAVAILABLE",
            "soc_version": "UNAVAILABLE", "device_count": 0,
            "logical_to_physical": {}, "visible_device_policy": "UNAVAILABLE",
            "cpu_arch": "aarch64", "os": "fixture", "kernel": "fixture",
            "container_runtime": "UNAVAILABLE", "image_digest": "UNAVAILABLE",
            "health": {"status": "UNAVAILABLE"},
        },
        "ascend_stack": {
            "firmware_version": "UNAVAILABLE", "firmware_components": {},
            "driver_version": "UNAVAILABLE", "driver_install_mode": "UNAVAILABLE",
            "driver_loaded_modules": [], "cann_toolkit_version": "UNAVAILABLE",
            "cann_runtime_version": "UNAVAILABLE", "cann_compiler_version": "UNAVAILABLE",
            "cann_ops_package_version": "UNAVAILABLE", "cann_kernel_package_version": "UNAVAILABLE",
            "set_env_source": "UNAVAILABLE", "library_resolution": {},
            "compiler_path": "UNAVAILABLE", "msprof_version": "UNAVAILABLE",
            "msprof_metric_sets": [], "install_roots": [],
        },
        "framework": {
            "python_version": "fixture", "pytorch_version": "UNAVAILABLE",
            "torch_npu_version": "UNAVAILABLE", "mindspore_version": "UNAVAILABLE",
            "atb_version": "UNAVAILABLE", "mindie_version": "UNAVAILABLE",
            "package_hashes": {}, "cpp_abi": "UNAVAILABLE", "compile_flags": [],
            "selected_integration_path": "UNAVAILABLE",
        },
        "project": {
            "git_commit": "fixture", "git_dirty": False,
            "source_patch_hash": "fixture", "model_artifact_hash": "UNAVAILABLE",
            "operator_spec_hash": "fixture", "quant_artifact_hash": "UNAVAILABLE",
        },
        "official_sources": [],
        "evidence": {"fixture": True},
    }
    command = {
        "argv": ["npu-smi", "info"], "returncode": 127, "stdout": "",
        "stderr": "not found", "duration_ms": 1.0, "timed_out": False,
    }
    return exp.CapabilitySnapshot(
        collected_at="2026-09-21T00:00:00Z",
        host={"board_model": "Jetson fixture", "machine": "aarch64"},
        commands=(command,),
        tool_paths={"npu-smi": "UNAVAILABLE"},
        device_nodes={"/dev/davinci0": False},
        python_modules={"torch": True, "torch_npu": False},
        stack_probe_summary={"statuses": {"device_query": "UNAVAILABLE"}},
        compatibility_manifest=manifest,
        compatibility_manifest_sha256="a" * 64,
        compatibility_schema={"ok": True, "problems": []},
        capabilities=(),
        ascend_ready=False,
        blocker_codes=("ASCEND_DEVICE_NOT_PRESENT", "CANN_ROOT_NOT_PRESENT"),
    )


def _detail_protocols(root: Path) -> None:
    details = root / "docs/stage_experiments/details/S09"
    details.mkdir(parents=True)
    for experiment_id in exp.EXPERIMENTS:
        lines = [f"# {experiment_id}", "", "## 具体实验步骤", ""]
        lines.extend(f"### 步骤 {step}：fixture step {step}" for step in range(1, 29))
        (details / f"{experiment_id}_fixture.md").write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.unit
def test_protocol_catalog_covers_ten_experiments_and_required_fields():
    assert tuple(exp.PROTOCOLS) == exp.EXPERIMENTS
    assert all(item.required_data for item in exp.PROTOCOLS.values())
    assert all(len(item.pass_criteria) == 7 for item in exp.PROTOCOLS.values())


@pytest.mark.unit
def test_interface_map_tracks_all_280_steps(tmp_path):
    _detail_protocols(tmp_path)
    report = interface_map.resolve_interfaces(tmp_path)
    assert report["ok"], report["failures"]
    assert report["experiments"] == 10
    assert report["steps"] == 280
    assert "does not mean" in report["notice"]


@pytest.mark.unit
def test_blocked_campaign_is_complete_and_frontend_addressable(tmp_path, monkeypatch):
    _detail_protocols(tmp_path)
    monkeypatch.setattr(exp, "collect_capability_snapshot", lambda _root: _snapshot())
    summary = exp.collect_campaign(tmp_path, run_id="fixture-run")

    assert summary["collector_status"] == "PASS"
    assert summary["stage_status"] == "BLOCKED"
    assert summary["stage_complete"] is False
    assert summary["experiment_statuses"]["E09-06"] == "N/A_BY_ADR"
    assert all(
        status == "BLOCKED"
        for experiment_id, status in summary["experiment_statuses"].items()
        if experiment_id != "E09-06"
    )

    for experiment_id in exp.EXPERIMENTS:
        experiment = tmp_path / "docs/stage_experiments/S09" / experiment_id
        verdict = json.loads((experiment / "raw/verdict.json").read_text())
        assert verdict["collector_status"] == "PASS"
        assert verdict["claim_allowed"] is False
        assert verdict["ascend_kernel_launched"] is False
        assert verdict["raw_samples"] == 0
        assert len((experiment / "raw/step_status.jsonl").read_text().splitlines()) == 28
        assert (experiment / f"{experiment_id}_实验报告.md").is_file()
        assert (experiment / "raw/evidence_manifest.json").is_file()

    catalog = EvidenceCatalog(tmp_path)
    items = catalog.scan(refresh=True)
    assert len(items) == 10
    assert {item["stage"] for item in items} == {"S09"}
    assert {item["experiment"] for item in items} == set(exp.EXPERIMENTS)
    assert {item["status"] for item in items} == {"BLOCKED", "N/A_BY_ADR"}
    assert all(any(ref["name"].endswith("_实验报告.md") for ref in item["files"]) for item in items)


@pytest.mark.unit
def test_cpu_smoke_is_explicitly_non_claiming():
    smoke = exp.smoke_self_check()
    assert smoke["status"] == "SMOKE_PASS"
    assert smoke["simulated"] is True
    assert smoke["claim_allowed"] is False

