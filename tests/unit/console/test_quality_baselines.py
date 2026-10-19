"""Durable floating baselines, strict identities, scalar deltas and historical backfill."""

import json
import math
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from hqsb.console.app import create_app
from hqsb.console.config import Settings
from hqsb.console.quality_baselines import QualityBaselines
from hqsb.console.service import ConsoleService
from hqsb.console.store import Store
from hqsb.experiments.artifacts import seal
from hqsb.experiments.quality_contract import compare_scores


def saved(store, directory, *, method="gptq", nll=3, change=None, precision=None):
    run, _ = store.create(
        "generate",
        {"deployment": {"model": "fixture"}, "experiment_role": "quality"},
        str(store.list()["total"]),
        str(store.list()["total"]),
    )
    identity = run["id"]
    quality = {
        "run_id": identity,
        "status": "complete",
        "score_mode": "corpus_nll",
        "benchmark_id": "fixture",
        "dataset_sha256": "dataset",
        "protocol": {"name": "fixture", "max_tokens": 3},
        "positions": [],
        "samples": [
            {
                "id": "sample",
                "slice": "article",
                "input_sha256": "input",
                "text_sha256": "text",
                "input_tokens": 3,
                "scored_tokens": 2,
                "nll_sum": nll,
            }
        ],
        "mean_nll": nll / 2,
        "perplexity": math.exp(nll / 2),
        "scored_tokens": 2,
        "document_count": 1,
    }
    receipt = {
        "run_id": identity,
        "status": "applied",
        "execution_status": "completed",
        "cleanup_status": "succeeded",
        "model_identity": {"hash": "model"},
        "tokenizer_identity": "tokenizer",
        "source_digest": "code",
        "effective": {"attention_backend": "eager"},
        "runtime": {"kernel": "framework-native"},
        "precision_receipt": {
            "method": method,
            "compute_dtype": "float16",
            "kv": {"bits": None, "backend": "model_default"},
            "execution": "floating_reference"
            if method == "none"
            else "fused_dequant_weight_only",
            "quantized_module_count": 0 if method == "none" else 4,
            **(precision or {}),
        },
    }
    if change:
        change(receipt, quality)
    folder = directory / "captures" / identity
    folder.mkdir(parents=True)
    for name, data in [
        ("quality-result.json", quality),
        ("execution-receipt.json", receipt),
    ]:
        (folder / name).write_text(json.dumps(data))
    seal(folder, run_id=identity)
    store.update(
        identity,
        {
            "state": "completed",
            "cleanup": "succeeded",
            "metrics": {
                "quality_evaluation": {"status": "complete"},
                "optimization_receipt": receipt,
            },
        },
    )
    return identity


def test_backfills_older_references_pins_and_persists_until_explicit_replacement(
    tmp_path,
):
    store = Store(tmp_path)
    reference = saved(store, tmp_path, method="none", nll=2)
    for _ in range(32):
        saved(store, tmp_path)
    candidate = saved(store, tmp_path)
    repo = QualityBaselines(tmp_path, store)
    result = repo.for_run(candidate)
    assert result["baseline"]["run_id"] == reference
    assert result["baseline"]["compute_dtype"] == "float16"
    assert result["quality"]["ppl_relative_increase_percent"] == pytest.approx(
        (math.exp(0.5) - 1) * 100
    )
    assert result["quality"]["mean_nll_increase"] == 0.5
    assert result["quality"]["status"] == "measured_no_acceptance_limits"
    assert QualityBaselines(tmp_path, store).for_run(reference)["status"] == "baseline"
    later = saved(
        store,
        tmp_path,
        method="none",
        nll=1,
        precision={"compute_dtype": "float32", "matmul_allow_tf32": False},
    )
    assert repo.for_run(candidate)["baseline"]["run_id"] == reference
    assert repo.register(later)["status"] == "baseline"
    assert repo.for_run(candidate)["baseline"]["compute_dtype"] == "float32"
    document = json.loads(
        next((tmp_path / "quality-baselines").glob("*.json")).read_text()
    )
    assert [x["run_id"] for x in document["history"]] == [reference, later]
    store.close()


@pytest.mark.parametrize(
    "field", ["model", "tokenizer", "source", "data", "protocol", "input", "packages"]
)
def test_never_mixes_contexts(tmp_path, field):
    store = Store(tmp_path)
    saved(store, tmp_path, method="none")

    def change(r, q):
        if field == "model":
            r["model_identity"] = {"hash": "other"}
        if field == "tokenizer":
            r["tokenizer_identity"] = "other"
        if field == "source":
            r["source_digest"] = "other"
        if field == "data":
            q["dataset_sha256"] = "other"
        if field == "protocol":
            q["protocol"]["max_tokens"] = 5
        if field == "input":
            q["samples"][0]["input_sha256"] = "other"
        if field == "packages":
            r["source"] = {"packages": {"torch": "other-version"}}

    candidate = saved(store, tmp_path, change=change)
    assert QualityBaselines(tmp_path, store).for_run(candidate)["status"] == "missing"
    store.close()


@pytest.mark.parametrize(
    "precision",
    [
        {"kv": {"bits": 2}},
        {"quantized_module_count": 1},
        {"artifact_id": "artifact"},
        {"compute_dtype": "float32", "matmul_allow_tf32": True},
    ],
)
def test_reference_rejects_quantized_kv_artifact_partial_quantization_and_tf32(
    tmp_path, precision
):
    store = Store(tmp_path)
    identity = saved(store, tmp_path, method="none", precision=precision)
    with pytest.raises(ValueError, match="UNQUANTIZED"):
        QualityBaselines(tmp_path, store).register(identity)
    store.close()


def test_corrupted_pinned_evidence_does_not_silently_choose_a_new_reference(tmp_path):
    store = Store(tmp_path)
    first = saved(store, tmp_path, method="none")
    repo = QualityBaselines(tmp_path, store)
    repo.register(first)
    saved(store, tmp_path, method="none", nll=2)
    candidate = saved(store, tmp_path)
    (tmp_path / "captures" / first / "quality-result.json").write_text("{}")
    result = repo.for_run(candidate)
    assert result["status"] == "unavailable"
    assert result["baseline_run_id"] == first
    store.close()


def test_concurrent_discovery_keeps_one_reference_and_one_history_entry(tmp_path):
    store = Store(tmp_path)
    reference = saved(store, tmp_path, method="none")
    candidate = saved(store, tmp_path)
    repo = QualityBaselines(tmp_path, store)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(repo.for_run, [candidate] * 4))
    assert all(r["baseline"]["run_id"] == reference for r in results)
    doc = json.loads(next((tmp_path / "quality-baselines").glob("*.json")).read_text())
    assert len(doc["history"]) == 1
    store.close()


def test_api_autolinks_and_only_explicit_floating_records_can_replace_reference(
    tmp_path,
):
    settings = Settings(
        data_dir=tmp_path, evidence_root=tmp_path, web_dist=tmp_path, deployments=[]
    )
    service = ConsoleService(settings)
    reference = saved(service.store, tmp_path, method="none", nll=2)
    candidate = saved(service.store, tmp_path)
    with TestClient(create_app(settings, service=service, monitor=False)) as client:
        prefix = "/api/console/v1"
        response = client.get(prefix + f"/runs/{candidate}/quality-baseline")
        assert response.status_code == 200, response.text
        assert response.json()["baseline"]["run_id"] == reference
        assert (
            client.post(
                prefix + "/precision/quality-baselines",
                json={"run_id": candidate},
                headers={"X-HQSB-Client": "console"},
            ).status_code
            == 409
        )
        response = client.post(
            prefix + "/precision/quality-baselines",
            json={"run_id": reference},
            headers={"X-HQSB-Client": "console"},
        )
        assert response.status_code == 201, response.text
        assert (
            client.get(
                prefix + "/runs/run_" + "0" * 32 + "/quality-baseline"
            ).status_code
            == 404
        )


def test_extreme_finite_nll_does_not_emit_infinite_json_percentages(tmp_path):
    store = Store(tmp_path)
    saved(store, tmp_path, method="none", nll=2)
    candidate = saved(store, tmp_path, nll=1416)
    result = QualityBaselines(tmp_path, store).for_run(candidate)
    assert result["quality"]["mean_nll_increase"] == 707
    assert result["quality"]["ppl_relative_increase_percent"] is None
    assert (
        "ppl_relative_increase_percent" in result["quality"]["numeric_overflow_fields"]
    )
    json.dumps(result, allow_nan=False)
    store.close()


def test_custom_scalar_accuracy_uses_percentage_points_and_leakage_is_rejected(
    tmp_path,
):
    from hqsb.experiments.artifacts import load_bundle

    store = Store(tmp_path)

    def custom(r, q):
        q.pop("score_mode")
        q.pop("benchmark_id")
        q["task_accuracy"] = 0.9

    reference = saved(store, tmp_path, method="none", nll=2, change=custom)
    candidate = saved(store, tmp_path, change=custom)
    left = load_bundle(tmp_path / "captures" / reference)
    right = load_bundle(tmp_path / "captures" / candidate)
    right["quality"]["task_accuracy"] = 0.8
    result = compare_scores(left, right)
    assert result["task_accuracy_drop_percentage_points"] == pytest.approx(10)
    right["receipt"]["precision_receipt"]["calibration_text_sha256"] = ["text"]
    with pytest.raises(ValueError, match="LEAKAGE"):
        compare_scores(left, right)
    store.close()
