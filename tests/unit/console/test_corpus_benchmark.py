"""PPL math, pinned selection, protocol identity and sealed API comparisons."""

import json
import math

import pytest
from fastapi.testclient import TestClient

from hqsb.console.app import create_app
from hqsb.console.config import DeploymentConfig, Settings
from hqsb.console.service import ConsoleService
from hqsb.core.optimization import validate_quality_plan
from hqsb.core import corpus_benchmark as corpus
from hqsb.experiments.artifacts import seal


def test_pinned_data_and_resolved_plan_are_immutable():
    data = corpus.load_benchmark()
    assert len(data["samples"]) == len({s["article"] for s in data["samples"]}) == 8
    assert [s["stratum"] for s in data["samples"]] == list(range(8))
    assert data["source"]["variant"] == "wikitext-2-raw-v1"
    assert (
        data["source"]["split"] == "test" and not data["selection"]["model_scores_used"]
    )
    plan = validate_quality_plan({"benchmark_id": corpus.BENCHMARK_ID})
    assert plan["max_tokens"] == 64 and plan["score_mode"] == "corpus_nll"
    assert validate_quality_plan(plan) == plan
    for change in ({"samples": []}, {"max_tokens": 128}, {"benchmark_sha256": "bad"}):
        with pytest.raises(ValueError, match="PLAN_MISMATCH"):
            validate_quality_plan({**plan, **change})
    with pytest.raises(ValueError, match="UNKNOWN"):
        validate_quality_plan({"benchmark_id": "unknown"})


def test_subset_builder_does_not_treat_embedded_equations_as_article_titles():
    from scripts.console.build_wikitext2_mini import build

    rows = []
    for index in range(8):
        rows.extend(
            [
                "",
                f"= Article {index} =",
                "",
                "A sentence about science. " * 30,
                "= 1 henry and capacitance C =",
                "with the formula continuing here.",
                "",
            ]
        )
    result = build(rows, {"split": "test"})
    assert result["selection"]["article_count"] == 8
    assert {s["article"] for s in result["samples"]} == {
        f"Article {i}" for i in range(8)
    }
    assert build(rows, {"split": "test"}) == result


def test_token_weighted_ppl_is_not_average_document_ppl():
    result = corpus.aggregate(
        [{"scored_tokens": 1, "nll_sum": 2.0}, {"scored_tokens": 3, "nll_sum": 0.0}]
    )
    assert result["mean_nll"] == 0.5
    assert result["perplexity"] == pytest.approx(math.exp(0.5))
    assert result["perplexity"] != pytest.approx((math.exp(2) + 1) / 2)
    overflow = corpus.aggregate([{"scored_tokens": 1, "nll_sum": 1000.0}])
    assert overflow["perplexity"] is None and overflow["perplexity_overflow"]
    for invalid in (
        [],
        [{"scored_tokens": 0, "nll_sum": 0}],
        [{"scored_tokens": 1, "nll_sum": float("nan")}],
    ):
        with pytest.raises(ValueError):
            corpus.aggregate(invalid)


def quality(run_id="run", nll=2.0):
    return {
        "run_id": run_id,
        "score_mode": "corpus_nll",
        "benchmark_id": corpus.BENCHMARK_ID,
        "dataset_sha256": corpus.DATA_SHA256,
        "status": "complete",
        "positions": [],
        "protocol": {"name": "unit-reference", "forward_mode": "incremental_decode"},
        "samples": [
            {
                "id": "sample",
                "slice": "article",
                "input_sha256": "tokens",
                "text_sha256": "text",
                "scored_tokens": 2,
                "nll_sum": nll,
            }
        ],
    }


def test_comparison_reports_delta_and_only_applies_predeclared_limits():
    result = corpus.compare(quality(), quality(nll=3.0))
    assert result["mean_nll_increase"] == 0.5
    assert result["ppl_ratio"] == pytest.approx(math.exp(0.5))
    assert result["status"] == "measured_no_acceptance_limits"
    assert corpus.compare(quality(), quality(nll=3.0), 0.1)["status"] == "failed"
    assert corpus.compare(quality(), quality(nll=3.0), 0.5)["status"] == "passed"
    for change in (
        {"benchmark_id": "another-benchmark"},
        {"dataset_sha256": "other"},
        {"protocol": {"forward_mode": "full_prefill"}},
        {"score_mode": "custom"},
    ):
        with pytest.raises(ValueError):
            corpus.compare(quality(), {**quality(), **change})
    other = quality()
    other["samples"][0]["input_sha256"] = "different-tokenizer-result"
    with pytest.raises(ValueError, match="TOKEN_OR_SAMPLE"):
        corpus.compare(quality(), other)


class Actor:
    def __init__(self, _):
        pass

    def execute(self, *_):
        return {"finish_reason": "length", "metrics": {}}

    def close(self):
        pass


def test_builtin_api_resolves_real_samples_preserves_idempotency_and_limits(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        evidence_root=tmp_path,
        web_dist=tmp_path,
        deployments=[DeploymentConfig(id="test", name="test")],
    )
    service = ConsoleService(settings, actor_factory=Actor)
    service.states["test"].update(state="ready", epoch="epoch")
    with TestClient(create_app(settings, service=service, monitor=False)) as client:
        catalogue = client.get("/api/console/v1/precision/quality-benchmarks")
        assert (
            catalogue.status_code == 200
            and catalogue.json()["items"][0]["sample_count"] == 8
        )
        body = {
            "deployment_id": "test",
            "expected_epoch": "epoch",
            "quality_plan": {"benchmark_id": corpus.BENCHMARK_ID},
            "ppl_limits": {"max_mean_nll_increase": 0.1},
        }
        headers = {"X-HQSB-Client": "console", "Idempotency-Key": "same"}
        first = client.post(
            "/api/console/v1/precision/quality-jobs", json=body, headers=headers
        )
        assert first.status_code == 202, first.text
        duplicate = client.post(
            "/api/console/v1/precision/quality-jobs", json=body, headers=headers
        )
        assert duplicate.json()["id"] == first.json()["id"]
        config = first.json()["config"]
        assert (
            len(config["quality_plan"]["samples"]) == 8
            and config["ppl_limits"]["max_mean_nll_increase"] == 0.1
        )
        assert "text" not in config["quality_plan"]["samples"][0]
        invalid = {**body, "quality_plan": {"samples": [{"text": "custom text"}]}}
        assert (
            client.post(
                "/api/console/v1/precision/quality-jobs", json=invalid, headers=headers
            ).status_code
            == 422
        )


@pytest.mark.parametrize(
    "candidate_limit,status", [(None, 201), ({"max_mean_nll_increase": 0.1}, 409)]
)
def test_sealed_corpus_comparison_never_requires_full_logits(
    tmp_path, candidate_limit, status
):
    settings = Settings(
        data_dir=tmp_path, evidence_root=tmp_path, web_dist=tmp_path, deployments=[]
    )
    service = ConsoleService(settings, actor_factory=Actor)
    ids = []
    for index in range(2):
        run, _ = service.store.create(
            "generate",
            {"quality_limits": None, "ppl_limits": candidate_limit if index else None},
            str(index),
            str(index),
        )
        identity = run["id"]
        folder = tmp_path / "captures" / identity
        folder.mkdir(parents=True)
        result = quality(identity, 2 + index)
        document = {
            "run_id": identity,
            "model_identity": {"hash": "same"},
            "tokenizer_identity": {"hash": "same"},
            "source_digest": "same-source",
            "effective": {"attention_backend": "eager"},
            "precision_receipt": {},
        }
        (folder / "quality-result.json").write_text(json.dumps(result))
        (folder / "execution-receipt.json").write_text(json.dumps(document))
        seal(folder, run_id=identity)
        service.store.update(identity, {"state": "completed", "cleanup": "succeeded"})
        ids.append(identity)
    with TestClient(create_app(settings, service=service, monitor=False)) as client:
        response = client.post(
            "/api/console/v1/precision/comparisons",
            json={"baseline_run_id": ids[0], "candidate_run_ids": ids[1:]},
            headers={"X-HQSB-Client": "console"},
        )
        assert response.status_code == status, response.text
        if status == 201:
            assert response.json()["candidates"][0]["quality"][
                "ppl_ratio"
            ] == pytest.approx(math.exp(0.5))
