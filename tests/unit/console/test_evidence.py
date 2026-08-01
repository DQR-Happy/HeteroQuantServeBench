"""S05 evidence stays addressable and truthful through the browser contract."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hqsb.console.app import PREFIX, create_app
from hqsb.console.config import Settings
from hqsb.console.evidence import EvidenceCatalog, MAX_EVIDENCE_BYTES


def write_evidence(root, experiment, verdict):
    directory = root / "docs/stage_experiments/S05" / experiment
    raw = directory / "raw"
    (raw / "quality").mkdir(parents=True)
    (raw / "kernel").mkdir()
    (directory / "post_fix").mkdir()
    (directory / "model_diagnostics").mkdir()
    documents = {
        raw / "verdict.json": json.dumps(verdict),
        raw / "summary.json": json.dumps({"experiment": experiment}),
        raw / "quality/summary.json": json.dumps({"quality_gate": "FAIL"}),
        raw / "kernel/summary.json": json.dumps({"actual_kernel": None}),
        raw / "quality/slices.jsonl": '{"slice":"long","status":"FAIL"}\n',
        raw / "kernel/timings.csv": "phase,latency_ms\ndecode,1.25\n",
        raw / "command.txt": "recorded command only; never executed by the API\n",
        directory / f"{experiment}_实验报告.md": f"# {experiment}\n\n保留原始结论。\n",
        directory / "post_fix/verification.json": '{"regression_pass":true,"original_verdict_unchanged":true}',
        directory / "model_diagnostics/interventions.jsonl": '{"scope":"diagnostic","deployable":false}\n',
    }
    for path, content in documents.items():
        path.write_text(content, encoding="utf-8")
    return raw, documents


def test_s05_browser_list_preview_download_preserves_results(tmp_path):
    statuses = {
        "E05-05": "BLOCKED",
        "E05-06": "FAIL",
        "E05-07": "PASS_WITH_NEGATIVE_CONCLUSION",
        "E05-08": "not_run",
        "E05-09": "PASS",
        "E05-10": "BLOCKED",
    }
    documents = {}
    for experiment, status in statuses.items():
        _, files = write_evidence(
            tmp_path,
            experiment,
            {
                "overall": status,
                "blocking_ids": ["quality_gate"] if status == "BLOCKED" else [],
                "contract_fixture_overall": "PASS",
            },
        )
        documents.update(files)
    settings = Settings(
        data_dir=tmp_path / "state",
        evidence_root=tmp_path,
        web_dist=tmp_path / "dist",
        deployments=[],
    )
    app = create_app(
        settings,
        "test-evidence-token",
        service=SimpleNamespace(close=lambda: None),
        monitor=False,
    )
    headers = {"Authorization": "Bearer test-evidence-token"}
    with TestClient(app) as client:
        assert client.get(PREFIX + "/evidence").status_code == 401
        response = client.get(PREFIX + "/evidence", headers=headers)
        assert response.status_code == 200
        items = response.json()["items"]
        assert {row["experiment"]: row["status"] for row in items} == statuses
        refs = [ref for row in items for ref in row["files"]]
        assert len({ref["id"] for ref in refs}) == len(documents)
        assert len({ref["relative_path"] for ref in refs}) == len(documents)
        # Same basenames from quality/kernel and different experiments must
        # remain distinct; the UI uses these IDs to open its evidence drawer.
        assert sum(ref["name"] == "summary.json" for ref in refs) == 18
        for ref in refs:
            path = tmp_path / ref["relative_path"]
            expected = documents[path].encode("utf-8")
            detail = client.get(PREFIX + f"/evidence/{ref['id']}", headers=headers)
            assert detail.status_code == 200
            body = detail.json()
            assert body["sha256"] == hashlib.sha256(expected).hexdigest()
            assert body["bytes"] == len(expected)
            assert body["content"] == (
                json.loads(expected)
                if path.suffix == ".json"
                else expected.decode("utf-8")
            )
            download = client.get(
                PREFIX + f"/evidence/{ref['id']}/download", headers=headers
            )
            assert download.status_code == 200
            assert download.content == expected
            assert download.headers["X-Content-SHA256"] == body["sha256"]
            assert path.read_bytes() == expected
        assert client.get(PREFIX + "/evidence/unknown", headers=headers).status_code == 404


def test_nested_evidence_rejects_escaped_binary_and_oversized_files(tmp_path):
    raw, documents = write_evidence(tmp_path, "E05-09", {"overall": "BLOCKED"})
    private = tmp_path / "operator-private.txt"
    private.write_text("not an experiment attachment")
    (raw / "quality/escaped.txt").symlink_to(private)
    (raw / "quality/tensor.npy").write_bytes(b"not browser text")
    (raw / "quality/trace.nsys-rep").write_bytes(b"native profiler data")
    with (raw / "quality/oversized.json").open("wb") as stream:
        stream.truncate(MAX_EVIDENCE_BYTES + 1)
    catalog = EvidenceCatalog(tmp_path)
    row = catalog.scan()[0]
    assert {tmp_path / ref["relative_path"] for ref in row["files"]} == set(documents)
    escaped_id = hashlib.sha256(b"operator-private.txt").hexdigest()[:24]
    with pytest.raises(KeyError):
        catalog.file(escaped_id)


def test_attachment_revalidated_after_indexing(tmp_path):
    raw, _ = write_evidence(tmp_path, "E05-06", {"overall": "FAIL"})
    catalog = EvidenceCatalog(tmp_path)
    refs = {ref["relative_path"]: ref for ref in catalog.scan()[0]["files"]}
    nested = raw / "quality/summary.json"
    key = refs[str(nested.relative_to(tmp_path))]["id"]
    private = tmp_path / "other-experiment.json"
    private.write_text('{"private":true}')
    nested.unlink()
    nested.symlink_to(private)
    with pytest.raises(KeyError):
        catalog.file(key)
    nested.unlink()
    with nested.open("wb") as stream:
        stream.truncate(MAX_EVIDENCE_BYTES + 1)
    with pytest.raises(ValueError, match="8 MB"):
        catalog.file(key)


def test_not_run_fixture_pass_does_not_become_experiment_pass(tmp_path):
    raw, _ = write_evidence(
        tmp_path,
        "E05-08",
        {"status": "not_run", "contract_fixture_overall": "PASS"},
    )
    catalog = EvidenceCatalog(tmp_path)
    before = catalog.scan()[0]
    assert before["status"] == "not_run"
    (raw / "verdict.json").write_text('{"contract_fixture_overall":"PASS"}')
    after = catalog.scan(refresh=True)[0]
    assert after["status"] == "UNKNOWN"
    assert after["id"] == before["id"]
    assert [ref["id"] for ref in after["files"]] == [
        ref["id"] for ref in before["files"]
    ]
