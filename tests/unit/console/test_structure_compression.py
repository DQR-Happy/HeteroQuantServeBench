"""Actual numerical transformations, compact reload and CPU-job ownership."""

import copy
import json
import time
import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from hqsb.core.structure_plan import StructurePlan
from hqsb.backends.structure_transforms import (
    apply,
    prune_weight,
    select,
    activation_stats,
)
from hqsb.backends.structure_artifacts import export, load, verify


def tiny():
    torch.manual_seed(3)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            tie_word_embeddings=True,
        )
    ).eval()
    t = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "Once": 1, "upon": 2, "a": 3, "time": 4, "fox": 5, "test": 6},
            unk_token="[UNK]",
        )
    )
    t.pre_tokenizer = Whitespace()
    return model, PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]")


def test_contract_rejects_ignored_calibration_or_invalid_patterns():
    for value in (
        {"method": "wanda"},
        {"method": "mlp_channels", "pattern": "2:4"},
        {"method": "magnitude", "pattern": "2:4", "sparsity": 0.4},
        {"method": "svd_low_rank", "calibration": ["test"]},
        {"method": "mlp_channels", "targets": ["x", "x"]},
        {"method": "unknown"},
    ):
        with pytest.raises(ValueError):
            StructurePlan.model_validate(value)


def test_weight_patterns_and_wanda_activation_axis():
    w = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    pruned, mask = prune_weight(w, 0.5, "2:4")
    assert mask.reshape(-1, 4).sum(-1).tolist() == [2, 2]
    assert pruned.tolist() == [[0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 7.0, 8.0]]
    ordinary, _ = prune_weight(w, 0.5, "unstructured")
    activated, _ = prune_weight(
        w,
        0.5,
        "unstructured",
        torch.tensor([100.0, 100.0, 100.0, 100.0, 0.01, 0.01, 0.01, 0.01]),
    )
    assert not torch.equal(ordinary, activated)
    with pytest.raises(ValueError):
        prune_weight(torch.ones(2, 5), 0.5, "2:4")


def test_gated_channel_coupling_matches_masked_reference():
    model, _ = tiny()
    name = "model.layers.0.mlp"
    original = copy.deepcopy(model.get_submodule(name))
    rows = apply(
        model,
        [name],
        StructurePlan(method="mlp_channels", keep_ratio=0.5),
        lambda: None,
        lambda *args: None,
    )
    row = rows[0]
    inputs = torch.randn(2, 3, 16)
    intermediate = original.act_fn(original.gate_proj(inputs)) * original.up_proj(
        inputs
    )
    keep = torch.zeros(32)
    keep[row["kept_channels"]] = 1
    expected = original.down_proj(intermediate * keep)
    torch.testing.assert_close(model.get_submodule(name)(inputs), expected)
    assert model.get_submodule(name).gate_proj.out_features == 16
    assert model.get_submodule(name).up_proj.out_features == 16
    assert model.get_submodule(name).down_proj.in_features == 16


def test_svd_runs_two_compact_matrices_with_no_dense_weight():
    model, _ = tiny()
    name = "model.layers.0.mlp.down_proj"
    original = model.get_submodule(name)
    with torch.no_grad():
        original.weight.copy_(torch.randn(16, 2) @ torch.randn(2, 32))
    x = torch.randn(2, 5, 32)
    expected = original(x)
    rows = apply(
        model,
        [name],
        StructurePlan(method="svd_low_rank", keep_ratio=0.8),
        lambda: None,
        lambda *args: None,
    )
    compact = model.get_submodule(name)
    torch.testing.assert_close(compact(x), expected, rtol=2e-5, atol=2e-5)
    assert not hasattr(compact, "weight")
    assert rows[0]["factor_parameters"] < rows[0]["original_parameters"]


@pytest.mark.parametrize(
    "method", ["mlp_channels", "svd_low_rank", "magnitude", "wanda"]
)
def test_actual_compact_roundtrip_identity_ties_and_tamper(tmp_path, method):
    model, tokenizer = tiny()
    before = sum(p.numel() for p in model.parameters())
    plan = StructurePlan(
        method=method, calibration=["Once upon a time fox"] if method == "wanda" else []
    )
    names = select(model, plan)
    stats, hashes = (
        activation_stats(model, tokenizer, names, plan, lambda: None, 128)
        if method == "wanda"
        else ({}, [])
    )
    rows = apply(model, names, plan, lambda: None, lambda *args: None, stats)
    with torch.inference_mode():
        expected = model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits
    doc = export(
        model,
        tokenizer,
        tmp_path / "artifact",
        {
            "artifact_id": "l5_" + "a" * 32,
            "source_model_identity": {"hash": "source"},
            "transforms": rows,
        },
    )
    doc["publication"] = "supervisor_acknowledged"
    (tmp_path / "artifact/l5-manifest.json").write_text(json.dumps(doc))
    _, restored, _ = load(tmp_path / "artifact", torch.float32, "sdpa")
    with torch.inference_mode():
        torch.testing.assert_close(
            restored(input_ids=torch.tensor([[1, 2, 3, 4]])).logits,
            expected,
            rtol=1e-5,
            atol=1e-5,
        )
    assert (
        restored.lm_head.weight.data_ptr()
        == restored.model.embed_tokens.weight.data_ptr()
    )
    after = sum(p.numel() for p in restored.parameters())
    assert (
        after < before
        if method in {"mlp_channels", "svd_low_rank"}
        else after == before
    )
    if method == "wanda":
        assert hashes
    weights = tmp_path / "artifact/weights.safetensors"
    with weights.open("r+b") as f:
        f.seek(-1, 2)
        b = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([b[0] ^ 1]))
    with pytest.raises(ValueError, match="INTEGRITY"):
        verify(tmp_path / "artifact")


def test_supervised_cpu_job_is_idempotent_exclusive_and_leaves_original_untouched(
    tmp_path,
    monkeypatch,
):
    import hashlib
    from hqsb.console.config import Settings, DeploymentConfig
    from hqsb.console.service import ConsoleService, ServiceError
    from hqsb.console.structure_jobs import StructureJobs
    from types import SimpleNamespace
    import shutil

    monkeypatch.setattr(
        shutil, "disk_usage", lambda _: SimpleNamespace(free=768 * 1024**2)
    )

    model, tokenizer = tiny()
    root = tmp_path / "source"
    model.save_pretrained(root)
    tokenizer.save_pretrained(root)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
    }
    manifest = tmp_path / "source.sha256"
    manifest.write_text(
        "".join(f"{sha} {name}\n" for name, sha in sorted(hashes.items()))
    )
    settings = Settings(
        data_dir=tmp_path / "console",
        evidence_root=tmp_path,
        web_dist=tmp_path / "web",
        deployments=[
            DeploymentConfig(
                id="tiny", name="tiny", model_path=str(root), manifest=str(manifest)
            )
        ],
    )

    class NoGPUActor:
        def __init__(self, _):
            pass

        def close(self):
            pass

        def execute(self, *args):
            raise AssertionError("CPU creation must not call the live GPU actor")

    svc = ConsoleService(settings, actor_factory=NoGPUActor)
    svc.states["tiny"].update(
        state="ready",
        epoch="original",
        precision="float16",
        detail={"parameter_bytes": sum(p.numel() * 2 for p in model.parameters())},
    )
    jobs = StructureJobs(settings, svc)
    try:
        plan = {"method": "mlp_channels", "targets": ["model.layers.0.mlp"]}
        with monkeypatch.context() as limited:
            limited.setattr(
                "hqsb.models.model_admission.effective_host_available", lambda *_: 0
            )
            with pytest.raises(ServiceError, match="容器可用内存"):
                jobs.start("tiny", "original", plan, "container-bound")
            assert svc.pending == 0 and svc.benchmark_owner is None
        row = jobs.start("tiny", "original", plan, "build")
        assert jobs.start("tiny", "original", plan, "build")["id"] == row["id"]
        with pytest.raises(ServiceError):
            jobs.start("tiny", "original", {**plan, "keep_ratio": 0.4}, "build")
        with pytest.raises(ServiceError):
            svc.submit("unload", "tiny", {}, "blocked")
        end = time.monotonic() + 60
        while svc.pending and time.monotonic() < end:
            time.sleep(0.05)
        result = svc.store.get(row["id"])
        assert result["state"] == "completed", result.get("error")
        doc = jobs.manifest(result["metrics"]["structure_artifact"]["artifact_id"])
        assert doc["publication"] == "supervisor_acknowledged"
        assert (
            doc["statistics"]["parameters_after"]
            < doc["statistics"]["parameters_before"]
        )
        assert (
            svc.states["tiny"]["state"] == "ready"
            and svc.states["tiny"]["epoch"] == "original"
        )
        assert not svc.benchmark_owner and not svc.pending
        assert {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
        } == hashes
        cancelled = jobs.start("tiny", "original", plan, "cancelled")
        svc.cancel(cancelled["id"])
        end = time.monotonic() + 30
        while svc.pending and time.monotonic() < end:
            time.sleep(0.05)
        assert svc.store.get(cancelled["id"])["state"] == "cancelled"
        assert len(list(jobs.root.glob("l5_*/l5-manifest.json"))) == 1
    finally:
        jobs.close()
        svc.close()


def test_failed_child_reap_releases_queue_and_reports_unknown_cleanup(
    tmp_path, monkeypatch
):
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace
    from hqsb.console.config import Settings, DeploymentConfig
    from hqsb.console.service import ConsoleService
    from hqsb.console.structure_jobs import StructureJobs
    import shutil

    monkeypatch.setattr(
        shutil, "disk_usage", lambda _: SimpleNamespace(free=2 * 1024**3)
    )
    monkeypatch.setattr(
        "hqsb.backends.observation.host_memory",
        lambda: {"host_available_bytes": 8 * 1024**3},
    )
    settings = Settings(
        data_dir=tmp_path / "state",
        evidence_root=tmp_path,
        web_dist=tmp_path / "web",
        deployments=[
            DeploymentConfig(
                id="tiny", name="tiny", manifest=str(tmp_path / "source.sha256")
            )
        ],
    )

    class Actor:
        def __init__(self, _):
            pass

        def close(self):
            pass

        def execute(self, *args):
            raise AssertionError("CPU cancellation must not invoke GPU actor")

    svc = ConsoleService(settings, actor_factory=Actor)
    svc.states["tiny"].update(
        state="ready",
        epoch="original",
        precision="float16",
        detail={"parameter_bytes": 1024},
    )
    jobs = StructureJobs(settings, svc)

    class UnreapableChild:
        returncode = None

        def __init__(self, args, **kwargs):
            svc.cancel(Path(args[-1]).parent.name)

        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("unreapable-test-child", timeout)

    monkeypatch.setattr(subprocess, "Popen", UnreapableChild)
    try:
        row = jobs.start("tiny", "original", {"method": "mlp_channels"}, "failed-reap")
        deadline = time.monotonic() + 10
        while svc.pending and time.monotonic() < deadline:
            time.sleep(0.01)
        result = svc.store.get(row["id"])
        assert result["state"] == "failed" and result["cleanup"] == "unknown"
        assert "L5_CLEANUP_FAILED" in result["error"]
        assert svc.pending == 0 and svc.benchmark_owner is None and jobs.stop.is_set()
        assert not (jobs.jobs / row["id"] / "input.json").exists()
        assert (
            svc.states["tiny"]["state"] == "ready"
            and svc.states["tiny"]["epoch"] == "original"
        )
    finally:
        jobs.close()
        svc.close()
