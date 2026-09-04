import array
import copy

import pytest

from hqsb.core.optimization import catalogue, fingerprint, resolve_plan, validate_quality_plan
from hqsb.evaluation.optimization_comparison import performance_gate
from hqsb.experiments.artifacts import load_bundle, seal
from hqsb.experiments.comparison import compare_campaign
from hqsb.experiments.plans import role_request, validate_spec
from hqsb.observation.contracts import resolve_observation
from hqsb.observation.storage import atomic_json, sha256

DEFAULTS = {"cleanup_policy": "empty_cache", "attention_backend": "eager", "output_budget": 8}


def spec():
    return {"baseline": {"changes": {}}, "candidate": {"changes": {"cleanup_policy": "preserve_allocator"}},
            "declared_changes": ["cleanup_policy"], "performance_pairs": 1,
            "quality_plan": {"samples": [{"text": "Hello GPU world.", "expected": "ok"}]}}


def test_catalogue_covers_layers_without_promising_execution():
    document = catalogue()
    assert set(document["layers"]) == set(range(1, 11))
    assert next(i for i in document["items"] if i["id"] == "paged_kv")["runtime_capability"] == "unavailable"
    assert all(i["runtime_capability"] != "available" for i in catalogue("openai")["items"])


@pytest.mark.parametrize("changes", [{"paged_kv": True}, {"output_budget": True}, {"output_budget": 0},
                                     {"cleanup_policy": "pretend"}, {"unknown": 1}])
def test_invalid_and_unimplemented_measures_rejected(changes):
    with pytest.raises(ValueError):
        resolve_plan({"changes": changes}, DEFAULTS)


def test_resolution_is_not_application_proof():
    result = resolve_plan({"changes": {"cleanup_policy": "preserve_allocator"}}, DEFAULTS)
    assert result["status"] == "resolved_not_applied"
    assert result["resolved"]["cleanup_policy"] == "preserve_allocator"
    with pytest.raises(ValueError, match="REQUIRES_RELOAD"):
        resolve_plan({"changes": {"attention_backend": "sdpa"}}, DEFAULTS, deployment_attention="eager")
    with pytest.raises(ValueError, match="PROVIDER"):
        resolve_plan({"changes": {"output_budget": 4}}, DEFAULTS, provider="openai")


def test_composite_sources_and_caps():
    plan = resolve_observation({"sources": ["allocator", "pages", "operators"], "max_pages": 2}, "basic")
    assert all(plan.collects(source) for source in ("allocator", "pages", "operators"))
    assert plan.max_pages == 2
    with pytest.raises(ValueError):
        resolve_observation({"sources": ["pages"]}, "off")
    with pytest.raises(ValueError):
        resolve_observation({"sources": ["pages", "pages"]}, "basic")


def test_registration_enforces_declared_change_and_dataset_bounds():
    plan = validate_spec(spec())
    assert role_request(plan, "baseline", "quality")["observation_mode"] == "off"
    bad = spec()
    bad["declared_changes"] = []
    with pytest.raises(ValueError, match="UNDECLARED"):
        validate_spec(bad)
    with pytest.raises(ValueError):
        validate_quality_plan({"samples": [{"text": "x"}], "max_tokens": 100})
    with pytest.raises(ValueError):
        validate_quality_plan({"samples": [{"text": "x", "script": "run"}]})


def bundle(tmp_path, run_id, plan, arm, role, logits=None, pid=1, effective_override=None):
    path = tmp_path / run_id
    path.mkdir()
    requested = role_request(plan, arm, role)
    resolved = resolve_plan(requested["optimization"], {**DEFAULTS, "output_budget": requested["max_output_tokens"]})
    receipt = {**resolved, "status": "applied", "effective": resolved["resolved"], "run_id": run_id,
               "measurement_role": role, "model_identity": {"scope": "verified_manifest_at_load", "hash": "weights"},
               "tokenizer_identity": "tokenizer", "output_token_hash": "equal",
               "source": {"source_digest": "source", "process": {"pid": pid, "start_ticks": pid}},
               "proof": {"input_token_sha256": "input", "output_tokens": 8, "runtime_e2e_ms": 100.0,
                         "cleanup_call": "torch.cuda.empty_cache" if resolved["resolved"]["cleanup_policy"] == "empty_cache" else "empty_cache_not_called"},
               "experiment_links": {"campaign_id": plan["experiment_id"], "spec_sha256": plan["spec_sha256"], "arm": arm, "role": role}}
    if effective_override:
        receipt["effective"] = {**receipt["effective"], **effective_override}
    atomic_json(path / "execution-receipt.json", receipt)
    if logits is not None:
        raw_path = path / "quality-logits-0000.f32"
        raw_path.write_bytes(array.array("f", logits).tobytes())
        quality = {"run_id": run_id, "status": "complete", "dataset_sha256": "dataset",
                   "protocol": {"top_k": 2, "generation_tokens": resolved["resolved"]["output_budget"]}, "task_accuracy": 1.0, "perplexity": 2.0,
                   "positions": [{"sample_id": "0", "slice": "all", "position": 0, "vocab_size": len(logits),
                                  "path": raw_path.name, "bytes": raw_path.stat().st_size,
                                  "sha256": sha256(raw_path), "nll": 1.0}],
                   "samples": [{"output_token_sha256": "same"}]}
        atomic_json(path / "quality-result.json", quality)
    seal(path, run_id=run_id)
    return path


def test_sealed_artifacts_detect_mutation_and_wrong_identity(tmp_path):
    plan = {**validate_spec(spec()), "experiment_id": "exp_test"}
    path = bundle(tmp_path, "run_a", plan, "baseline", "quality", [1, 2, 3])
    assert load_bundle(path)["manifest"]["status"] == "complete"
    (path / "quality-logits-0000.f32").write_bytes(b"bad")
    with pytest.raises(ValueError, match="INTEGRITY"):
        load_bundle(path)


def test_quality_reports_full_positions_and_does_not_auto_adopt(tmp_path):
    plan = {**validate_spec(spec()), "experiment_id": "exp_test"}
    a = bundle(tmp_path, "run_a", plan, "baseline", "quality", [1, 2, 3], pid=1)
    b = bundle(tmp_path, "run_b", plan, "candidate", "quality", [1, 2, 3], pid=2)
    pair = {"role": "quality", "baseline_dir": a, "candidate_dir": b}
    report = compare_campaign(plan, [pair])
    assert report["comparable"] and report["quality_status"] == "passed"
    assert report["quality"][0]["positions"][0]["logit_max_abs_error"] == 0
    assert report["decision"] == "hold" and report["performance"]["status"] == "not_evaluated"


def test_quality_failure_and_registered_plan_mismatch_hold(tmp_path):
    plan = {**validate_spec(spec()), "experiment_id": "exp_test"}
    a = bundle(tmp_path, "run_a", plan, "baseline", "quality", [1, 2, 3])
    b = bundle(tmp_path, "run_b", plan, "candidate", "quality", [9, 1, 0])
    pair = {"role": "quality", "baseline_dir": a, "candidate_dir": b}
    assert compare_campaign(plan, [pair])["quality_status"] == "failed"
    changed = copy.deepcopy(plan)
    changed["spec"]["candidate"] = {"changes": {"cleanup_policy": "empty_cache"}}
    changed["spec_sha256"] = fingerprint(changed["spec"])
    report = compare_campaign(changed, [pair])
    assert "REGISTERED_PLAN_MISMATCH:candidate" in report["reasons"]
    assert report["decision"] == "hold"


def test_self_reported_applied_does_not_override_actual_effect(tmp_path):
    plan = {**validate_spec(spec()), "experiment_id": "exp_test"}
    a = bundle(tmp_path, "run_a", plan, "baseline", "quality", [1, 2, 3])
    b = bundle(tmp_path, "run_b", plan, "candidate", "quality", [1, 2, 3],
               effective_override={"cleanup_policy": "empty_cache"})
    report = compare_campaign(plan, [{"role": "quality", "baseline_dir": a, "candidate_dir": b}])
    assert "REGISTERED_EFFECT_NOT_MATCHED:candidate" in report["reasons"]
    assert report["decision"] == "hold"


def test_declared_output_budget_preserves_teacher_forced_comparison(tmp_path):
    raw = spec()
    raw["candidate"] = {"changes": {"output_budget": 8}}
    raw["declared_changes"] = ["output_budget"]
    # Baseline output budget for performance is eight, so declare a real change.
    raw["output_tokens"] = 4
    plan = {**validate_spec(raw), "experiment_id": "exp_test"}
    a = bundle(tmp_path, "run_a", plan, "baseline", "quality", [1, 2, 3])
    b = bundle(tmp_path, "run_b", plan, "candidate", "quality", [1, 2, 3])
    report = compare_campaign(plan, [{"role": "quality", "baseline_dir": a, "candidate_dir": b}])
    assert report["quality_status"] == "passed"
    assert report["quality"][0]["generation_budgets"] == [4, 8]
    assert report["decision"] == "hold"


def test_partial_and_path_traversal_are_not_complete(tmp_path):
    path = tmp_path / "run_x"
    path.mkdir()
    manifest = seal(path, run_id="run_x")
    assert manifest["status"] == "partial"
    manifest["artifacts"] = [{"path": "../escape", "bytes": 1, "sha256": "x"}]
    atomic_json(path / "experiment-manifest.json", manifest)
    with pytest.raises(ValueError, match="PATH"):
        load_bundle(path)


def test_performance_requires_independent_process_pairs():
    assert performance_gate([-0.1] * 10, maximum=0, independent=True)["status"] == "passed"
    assert performance_gate([-0.1] * 10, maximum=0, independent=False)["status"] == "insufficient_evidence"
    assert performance_gate([0.1] * 10, maximum=0, independent=True)["status"] == "failed"


def test_performance_uses_sealed_proof_not_unbound_result_values(tmp_path):
    raw = spec()
    raw["performance_pairs"] = 10
    plan = {**validate_spec(raw), "experiment_id": "exp_test"}
    pairs = []
    for index in range(10):
        a = bundle(tmp_path, f"run_a{index}", plan, "baseline", "performance", pid=2 * index)
        b = bundle(tmp_path, f"run_b{index}", plan, "candidate", "performance", pid=2 * index + 1)
        pairs.append({"role": "performance", "baseline_dir": a, "candidate_dir": b,
                      "baseline_metrics": {"runtime_e2e_ms": 100}, "candidate_metrics": {"runtime_e2e_ms": 1}})
    report = compare_campaign(plan, pairs)
    assert report["performance"]["median_relative_e2e"] == 0
    assert report["performance"]["pairs"] == 10
    assert report["decision"] == "hold"  # Quality and mechanisms still have no evidence.


def test_source_patch_must_match_declared_file_hashes(tmp_path):
    raw = spec()
    raw["control_changes"] = {"source_files": {"hqsb/example.py": {"baseline": "a" * 64, "candidate": "b" * 64}}}
    plan = {**validate_spec(raw), "experiment_id": "exp_test"}
    a = bundle(tmp_path, "run_a", plan, "baseline", "quality", [1, 2, 3])
    b = bundle(tmp_path, "run_b", plan, "candidate", "quality", [1, 2, 3])
    report = compare_campaign(plan, [{"role": "quality", "baseline_dir": a, "candidate_dir": b}])
    assert "SOURCE_PATCH_SCOPE_MISMATCH" in report["reasons"]
    assert report["decision"] == "hold"


def test_frozen_plan_cannot_be_changed_after_registration():
    plan = validate_spec(spec())
    plan["spec"]["quality_limits"]["max_logit_abs_error"] = 100
    with pytest.raises(ValueError, match="PLAN_INTEGRITY"):
        compare_campaign(plan, [])


@pytest.mark.parametrize("field,value", [("max_relative_e2e", True), ("output_contract", []),
                                        ("control_changes", {"source_files": {"../escape": {}}})])
def test_bad_experiment_fields_are_validation_errors(field, value):
    raw = spec()
    raw[field] = value
    with pytest.raises(ValueError):
        validate_spec(raw)


def test_experiment_api_authentication_registration_and_missing_evidence(tmp_path):
    from fastapi.testclient import TestClient
    from hqsb.console.app import PREFIX, create_app
    from hqsb.console.config import DeploymentConfig, Settings

    settings = Settings(auth_required=True, data_dir=tmp_path / "store", evidence_root=tmp_path, web_dist=tmp_path / "web",
                        deployments=[DeploymentConfig(id="reference", name="reference")])
    with TestClient(create_app(settings, "secret", monitor=False)) as client:
        assert client.get(PREFIX + "/optimization-catalog").status_code == 401
        assert client.get(PREFIX + "/experiments").status_code == 401
        client.headers["Authorization"] = "Bearer secret"
        created = client.post(PREFIX + "/experiments/plans", json={"spec": spec()})
        assert created.status_code == 201
        experiment_id = created.json()["experiment_id"]
        listing = client.get(PREFIX + "/experiments?limit=1").json()
        assert listing["total"] == 1 and listing["items"][0]["spec_sha256"] == created.json()["spec_sha256"]
        assert client.get(PREFIX + "/experiments?offset=1").json()["items"] == []
        assert client.get(PREFIX + "/experiments?limit=51").status_code == 422
        assert client.get(PREFIX + f"/experiments/{experiment_id}/jobs").json() == {"items": []}
        report = client.get(PREFIX + f"/experiments/{experiment_id}/report").json()
        assert report["completeness"] == "partial" and report["decision"] == "hold"
        assert report["quality_status"] == "not_evaluated"
        assert client.get(PREFIX + "/experiments/invalid/plan").status_code == 422
        malformed = {**spec(), "declared_changes": [["cleanup_policy"]]}
        assert client.post(PREFIX + "/experiments/plans", json={"spec": malformed}).status_code == 422
        from hqsb.experiments.artifacts import read_json
        path = settings.data_dir / "experiments" / experiment_id / "plan.json"
        document = read_json(path)
        document["spec"]["output_tokens"] = 20
        atomic_json(path, document)
        assert client.get(PREFIX + f"/experiments/{experiment_id}/plan").status_code == 409
        assert client.get(PREFIX + "/experiments").status_code == 409
        assert client.get(PREFIX + f"/experiments/{experiment_id}/jobs").status_code == 409
