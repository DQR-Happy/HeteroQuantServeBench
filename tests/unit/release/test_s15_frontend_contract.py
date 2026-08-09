"""S15 browser evidence uses the same read-only list/detail contract as prior stages."""

from __future__ import annotations

import json

from hqsb.console.evidence import EvidenceCatalog


def test_s15_verdict_and_report_are_indexable(tmp_path) -> None:
    experiment = tmp_path / "docs/stage_experiments/S15/E15-01"
    raw = experiment / "raw"
    raw.mkdir(parents=True)
    (raw / "verdict.json").write_text(
        json.dumps(
            {
                "stage": "S15",
                "experiment": "E15-01",
                "overall": "BLOCKED",
                "component_status": "PARTIAL_SCAN_PASS",
            }
        ),
        encoding="utf-8",
    )
    (raw / "claim_ledger.yaml").write_text('{"status":"DRAFT"}\n', encoding="utf-8")
    (experiment / "E15-01_实验报告.md").write_text("# E15-01\n", encoding="utf-8")

    catalog = EvidenceCatalog(tmp_path)
    items = catalog.scan(refresh=True)
    assert len(items) == 1
    assert items[0]["stage"] == "S15"
    assert items[0]["experiment"] == "E15-01"
    assert items[0]["status"] == "BLOCKED"
    assert {item["name"] for item in items[0]["files"]} == {
        "verdict.json",
        "claim_ledger.yaml",
        "E15-01_实验报告.md",
    }
    detail = catalog.detail(items[0]["id"])
    assert detail["content"]["component_status"] == "PARTIAL_SCAN_PASS"
