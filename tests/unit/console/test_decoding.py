"""L7 numerical semantics, real cached verification, and control-plane contracts."""

import copy
import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from hqsb.core.decoding_plan import DecodingPlan, validate_decoding, public_plan
from hqsb.backends.decoding_sampling import Sampler, ChoiceTrie
from hqsb.backends.decoding_engine import Decoder, lookup, DecodingStopped


def model(seed=3):
    torch.manual_seed(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=256,
            eos_token_id=None,
        )
    ).eval()


def reference(m, history, count):
    ids = []
    cache = DynamicCache(config=m.config)
    with torch.inference_mode():
        current = history
        for _ in range(count):
            out = m(
                input_ids=torch.tensor([current]), past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            token = int(out.logits[0, -1].argmax())
            ids.append(token)
            current = [token]
    return ids


@pytest.mark.parametrize(
    "value",
    [
        {"sampling": "greedy", "temperature": 0.7},
        {"method": "assistant"},
        {"method": "prompt_lookup", "sampling": "multinomial"},
        {"method": "prompt_lookup", "choices": ["x"]},
        {"method": "autoregressive", "draft_tokens": 8},
        {"choices": ["x", "x"]},
        {"constraint_backend": "mask"},
        {"method": "unknown"},
        {"lookup_min_ngram": 4, "lookup_max_ngram": 2},
    ],
)
def test_no_silently_ignored_or_incompatible_controls(value):
    with pytest.raises(ValueError):
        DecodingPlan.model_validate(value)


def test_execution_boundaries_and_private_choice_plan():
    p = DecodingPlan(choices=["private label"])
    assert "private label" not in str(public_plan(p))
    request = {"max_output_tokens": 16, "prompt": "Classify:"}
    assert validate_decoding(p, native=True, modified=False, request=request)["choices"]
    for extra in (
        {"memory_plan": {}},
        {"runtime_plan": {}},
        {"experiment_role": "quality"},
        {"fixed_output_length": True},
        {"observation_mode": "operators"},
        {"max_output_tokens": 513},
    ):
        with pytest.raises(ValueError):
            validate_decoding(
                p, native=True, modified=False, request={**request, **extra}
            )
    with pytest.raises(ValueError):
        validate_decoding({}, native=True, modified=True, request=request)


def test_sampling_reference_filter_order_penalty_and_local_randomness():
    p = DecodingPlan(
        sampling="multinomial",
        temperature=0.7,
        top_k=3,
        top_p=0.8,
        repetition_penalty=1.2,
    )
    sampler = Sampler(p, "cpu", {0})
    logits = torch.tensor([-2.0, 1.0, 3.0, 2.0, 0.5])
    scores, _ = sampler.scores(logits, [0, 2], [2])
    expected = logits.clone()
    expected[0] *= 1.2
    expected[2] /= 1.2
    expected /= 0.7
    expected[expected < torch.topk(expected, 3).values[-1]] = -torch.inf
    ordered, idx = expected.sort(descending=True)
    remove = ordered.softmax(-1).cumsum(-1) > 0.8
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    expected[idx[remove]] = -torch.inf
    torch.testing.assert_close(scores, expected)
    state = torch.random.get_rng_state().clone()
    a = Sampler(p, "cpu", {0})
    b = Sampler(p, "cpu", {0})
    assert [a.select(logits, [], []) for _ in range(40)] == [
        b.select(logits, [], []) for _ in range(40)
    ]
    assert torch.equal(state, torch.random.get_rng_state())
    with pytest.raises(ValueError):
        a.select(torch.tensor([float("nan"), 0.0]), [], [])
    assert (
        Sampler(DecodingPlan(min_new_tokens=2), "cpu", {2}).select(logits, [], []) != 2
    )


class Tokenizer:
    def encode(self, text, **_):
        return list(map(int, text.split()))


def test_constraint_mask_and_gather_share_tie_break_and_terminal_semantics():
    for backend in ("mask", "gather"):
        p = DecodingPlan(choices=["2", "2 3"], constraint_backend=backend)
        trie = ChoiceTrie(Tokenizer(), p.choices, {0}, 6, 0)
        s = Sampler(p, "cpu", {0}, trie)
        logits = torch.ones(16)
        assert s.select(logits, [], []) == 2
        assert s.select(logits, [2], [2]) == 0
        assert trie.valid([2, 0]) and trie.valid([2, 3, 0]) and not trie.valid([2])
    with pytest.raises(ValueError):
        ChoiceTrie(Tokenizer(), ["2 3"], {0}, 2, 0)
    with pytest.raises(ValueError):
        ChoiceTrie(Tokenizer(), ["2"], {0}, 4, 2)


def test_lookup_only_proposes_existing_following_tokens_within_budget():
    p = DecodingPlan(method="prompt_lookup", lookup_min_ngram=2, lookup_max_ngram=3)
    assert lookup([1, 2, 3, 4, 5, 1, 2, 3], p, 4) == [4, 5, 1, 2]
    assert lookup([1, 2, 3], p, 4) == []
    assert lookup([1, 2, 3, 4, 1, 2, 3], p, 0) == []


@pytest.mark.parametrize("method", ["autoregressive", "prompt_lookup", "assistant"])
@pytest.mark.parametrize("count", [1, 2, 17])
def test_actual_hf_model_matches_independent_serial_reference_and_cache_invariant(
    method, count
):
    target = model()
    draft = model(11) if method == "assistant" else None
    p = DecodingPlan(
        method=method, **({"draft_deployment_id": "tiny"} if draft else {})
    )
    history = [1, 2, 3, 1, 2, 3]
    expected = reference(target, history, count)
    d = Decoder(
        target, p, Sampler(p, "cpu", set()), history, count, set(), lambda: None, draft
    )
    output = [t for group in d.rounds() for t in group]
    assert output == expected
    assert d.cache.get_seq_length() == len(history) + count - 1
    assert all(
        r["target_cache_tokens"] == r["committed_tokens"] - 1 for r in d.stats["rounds"]
    )
    if method == "assistant" and count > 2:
        assert (
            d.summary()["draft_forward_calls"] > 0
            and d.summary()["verification_calls"] > 0
        )
        assert d.summary()["discarded_proposal_tokens"] > 0
    d.close()
    assert d.cache is None and (not d.draft or d.draft.cache is None)


def test_full_acceptance_bonus_eos_and_cancellation_do_not_reuse_future_states():
    target = model()
    p = DecodingPlan(method="assistant", draft_deployment_id="same")
    d = Decoder(
        target,
        p,
        Sampler(p, "cpu", set()),
        [1, 2, 3],
        12,
        set(),
        lambda: None,
        copy.deepcopy(target),
    )
    assert [t for g in d.rounds() for t in g] == reference(target, [1, 2, 3], 12)
    assert d.summary()["accepted_tokens"] > 0
    assert d.summary()["target_forward_calls"] < 12
    eos = reference(target, [1, 2, 3], 1)[0]
    d = Decoder(
        target,
        p,
        Sampler(p, "cpu", {eos}),
        [1, 2, 3],
        12,
        {eos},
        lambda: None,
        copy.deepcopy(target),
    )
    assert [t for g in d.rounds() for t in g] == [eos]
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls > 3:
            raise DecodingStopped("cancelled")

    d = Decoder(
        target,
        p,
        Sampler(p, "cpu", set()),
        [1, 2, 3],
        12,
        set(),
        check,
        copy.deepcopy(target),
    )
    with pytest.raises(DecodingStopped):
        list(d.rounds())
    d.close()
    assert d.cache is None and d.draft.cache is None


def test_service_rejects_unverified_combinations_before_dispatch(tmp_path):
    from hqsb.console.service import ConsoleService, ServiceError
    from hqsb.console.config import Settings, DeploymentConfig

    class Actor:
        def __init__(self, _):
            pass

        def close(self):
            pass

        def execute(self, *_):
            raise AssertionError("Rejected plan must not execute")

    svc = ConsoleService(
        Settings(
            data_dir=tmp_path,
            evidence_root=tmp_path,
            web_dist=tmp_path / "web",
            deployments=[DeploymentConfig(id="test", name="test")],
        ),
        actor_factory=Actor,
    )
    svc.states["test"].update(state="ready", epoch="e", precision="float16", detail={})
    try:
        body = {
            "prompt": "a",
            "max_output_tokens": 8,
            "expected_epoch": "e",
            "decoding_plan": {"method": "assistant", "draft_deployment_id": "missing"},
            "observation_mode": "off",
        }
        with pytest.raises(ServiceError, match="REGISTERED"):
            svc.submit("generate", "test", body, "unknown")
        svc.states["test"]["structure_artifact_id"] = "l5_" + "a" * 32
        with pytest.raises(ServiceError, match="RESTORE"):
            svc.submit("generate", "test", {**body, "decoding_plan": {}}, "modified")
        assert svc.pending == 0
    finally:
        svc.close()


def test_comparison_rejects_unapplied_or_tampered_tokens_and_sampling_changes():
    from hqsb.console.decoding_routes import compare
    from hqsb.core.optimization import fingerprint

    plan = public_plan({})
    run = {
        "id": "a",
        "state": "completed",
        "cleanup": "succeeded",
        "config": {
            "observation_mode": "off",
            "fixed_output_length": True,
            "actual_runtime": {"device_name": "test-gpu"},
        },
        "metrics": {"generated_token_ids": [1, 2], "output_tokens": 2},
    }
    receipt = {
        "status": "applied",
        "cleanup_status": "succeeded",
        "model_identity": {"hash": "source"},
        "tokenizer_identity": "tok",
        "source_digest": "runtime",
        "runtime": {"kernel": "framework-native"},
        "precision_receipt": {"plan_sha256": "float"},
        "effective": {"attention_backend": "sdpa", "cleanup_policy": "empty_cache"},
        "proof": {"input_token_sha256": "input", "requested_token_limit": 2},
        "output_token_hash": fingerprint([1, 2]),
        "decoding_receipt": {"status": "invoked", "plan": plan, "statistics": {}},
    }
    a = copy.deepcopy(run)
    b = copy.deepcopy(run)
    b["id"] = "b"
    left = {"manifest": {"status": "complete"}, "receipt": copy.deepcopy(receipt)}
    right = copy.deepcopy(left)
    right["receipt"]["decoding_receipt"]["plan"]["method"] = "prompt_lookup"
    assert compare(a, b, left, right)["comparable"]
    right["receipt"]["status"] = "not_fully_applied"
    assert (
        "applied_execution_receipts_required" in compare(a, b, left, right)["reasons"]
    )
    right["receipt"]["status"] = "applied"
    b["metrics"]["generated_token_ids"] = [1, 3]
    result = compare(a, b, left, right)
    assert (
        "sealed_output_token_identity_mismatch" in result["reasons"]
        and result["quality"]["first_mismatch"] == 1
    )
    b["metrics"]["generated_token_ids"] = [1, 2]
    right["receipt"]["decoding_receipt"]["plan"]["sampling"] = "multinomial"
    assert (
        "different_sampling_or_constraint_semantics"
        in compare(a, b, left, right)["reasons"]
    )


def test_decoding_sidecar_is_supervisor_sealed_and_tamper_detected(tmp_path):
    from hqsb.experiments.artifacts import seal, read_manifest
    from hqsb.observation.storage import atomic_json

    root = tmp_path / "run_test"
    root.mkdir()
    atomic_json(root / "execution-receipt.json", {"run_id": "run_test"})
    atomic_json(
        root / "decoding-receipt.json", {"run_id": "run_test", "status": "invoked"}
    )
    manifest = seal(root, run_id="run_test", status="complete")
    assert "decoding-receipt.json" in {a["path"] for a in manifest["artifacts"]}
    atomic_json(
        root / "decoding-receipt.json", {"run_id": "run_test", "status": "forged"}
    )
    with pytest.raises(ValueError, match="INTEGRITY"):
        read_manifest(root)


def test_global_optimization_inventory_links_the_real_l7_entrypoint():
    from hqsb.core.optimization import catalogue

    item = next(i for i in catalogue()["items"] if i["id"] == "speculative_decode")
    assert item["implementation_status"] == "request_api_only"
    assert (
        item["entrypoint"] == "/decoding" and item["scope"] == "request_decoding_plan"
    )


def test_draft_budget_uses_destination_dtype_and_rejects_quantized_source(tmp_path):
    import json
    import torch
    from safetensors.torch import save_file
    from hqsb.backends.decoding_draft import planned_parameter_bytes

    config = tmp_path / "config.json"
    config.write_text("{}")
    with pytest.raises(ValueError, match="SAFETENSORS_CHECKPOINT"):
        planned_parameter_bytes(tmp_path, 4)
    save_file(
        {"weight": torch.zeros((3, 5), dtype=torch.float16)},
        tmp_path / "model.safetensors",
    )
    assert planned_parameter_bytes(tmp_path, 2) == 30
    assert planned_parameter_bytes(tmp_path, 4) == 60
    config.write_text(json.dumps({"quantization_config": {"quant_method": "gptq"}}))
    with pytest.raises(ValueError, match="UNQUANTIZED_CHECKPOINT"):
        planned_parameter_bytes(tmp_path, 4)


def test_off_mode_l7_still_reserves_disk_for_mandatory_receipts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hqsb.console.service import ConsoleService, ServiceError
    from hqsb.console.config import Settings, DeploymentConfig

    class Actor:
        def __init__(self, _):
            pass

        def close(self):
            pass

    cfg = DeploymentConfig(id="target", name="target")
    svc = ConsoleService(
        Settings(
            data_dir=tmp_path,
            evidence_root=tmp_path,
            web_dist=tmp_path / "web",
            deployments=[cfg],
        ),
        actor_factory=Actor,
    )
    monkeypatch.setattr(
        "hqsb.console.service.shutil.disk_usage", lambda _: SimpleNamespace(free=0)
    )
    monkeypatch.setattr(
        "hqsb.console.service.host_memory",
        lambda *args: {"available_bytes": 16 * 1024**3},
    )
    try:
        with pytest.raises(ServiceError, match="free disk"):
            svc._preflight(
                "generate", cfg, {"observation_mode": "off", "decoding_plan": {}}
            )
        # Legacy off mode writes no mandatory L7 receipt and retains its contract.
        svc._preflight("generate", cfg, {"observation_mode": "off"})
        assert svc.pending == 0
    finally:
        svc.close()
