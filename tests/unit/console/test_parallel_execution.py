"""L9 contracts and real multi-process native-model execution on the target."""

import json
import os
import subprocess
import sys
import pytest
from hqsb.core.parallel_execution import ParallelPlan, validate_hardware
from hqsb.observation.storage import sha256


@pytest.mark.parametrize(
    "value",
    [
        {"strategy": "tp"},
        {"devices": [0, 0]},
        {"strategy": "single", "devices": [0, 1]},
        {"device_type": "cpu"},
        {"prompts": [""]},
        {"strategy": "dp", "devices": [0, 1], "prompts": ["one"]},
        {"strategy": "prefill_decode", "devices": [0, 1, 2]},
        {
            "strategy": "layer_shard",
            "devices": [0, 1],
            "device_type": "cpu",
            "precision": "float32",
        },
        {"strategy": "ep"},
    ],
)
def test_invalid_parallel_contracts(value):
    with pytest.raises(ValueError):
        ParallelPlan.model_validate(value)


def test_no_fake_cuda_ranks_or_silent_fallback():
    p = ParallelPlan(strategy="tp", devices=[0, 1])
    with pytest.raises(ValueError, match="NOT_VISIBLE"):
        validate_hardware(p, {"devices": [{}], "nccl": True, "gloo": True})
    assert not ParallelPlan().trace_collectives
    with pytest.raises(ValueError, match="TRACE_LIMIT"):
        ParallelPlan(trace_collectives=True, max_output_tokens=128)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers

    root = tmp_path_factory.mktemp("l9-model")
    torch.manual_seed(17)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            eos_token_id=None,
        )
    )
    model.save_pretrained(root)
    vocab = {f"t{i}": i for i in range(32)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="t0", bos_token="t1"
    ).save_pretrained(root)
    manifest = root.parent / "l9-tiny.sha256"
    manifest.write_text(
        "".join(
            f"{sha256(p)}  {p.name}\n" for p in sorted(root.iterdir()) if p.is_file()
        )
    )
    return {
        "model_path": str(root),
        "manifest": str(manifest),
        "attention": "eager",
        "context_limit": 128,
    }


def run(tmp_path, checkpoint, strategy):
    n = 1 if strategy == "single" else 2
    plan = ParallelPlan(
        strategy=strategy,
        device_type="cpu",
        devices=list(range(n)),
        precision="float32",
        prompts=["t2 t3 t4", "t7 t8 t9"],
        max_output_tokens=7,
        trace_collectives=True,
    )
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps({"run_id": "test", "config": checkpoint, "plan": plan.model_dump()})
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={n}",
            "--max-restarts=0",
            "--module",
            "hqsb.backends.parallel_rank",
            str(path),
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        },
        capture_output=True,
        text=True,
        timeout=80,
    )
    assert result.returncode == 0, result.stderr[-9000:]
    return [json.loads((tmp_path / f"rank-{i}.json").read_text()) for i in range(n)]


def test_real_cpu_dp_tp_and_kv_handoff_match_serial(tmp_path, checkpoint):
    base = run(tmp_path / "single", checkpoint, "single")[0]
    actual = lambda rs: sorted(
        [(r["index"], r["token_ids"]) for rank in rs for r in rank["requests"]]
    )
    expected = actual([base])
    dp = run(tmp_path / "dp", checkpoint, "dp")
    assert actual(dp) == expected and all(len(r["requests"]) == 1 for r in dp)
    assert dp[0]["pid"] != dp[1]["pid"]
    tp = run(tmp_path / "tp", checkpoint, "tp")
    assert all(actual([r]) == expected for r in tp)
    assert all(
        any(t["global_shape"] != t["local_shape"] for t in r["parameters"]["tensors"])
        for r in tp
    )
    assert any("all" in e["name"].lower() for r in tp for e in r["collectives"])
    pd = run(tmp_path / "pd", checkpoint, "prefill_decode")
    assert not pd[0]["requests"] and actual([pd[1]]) == expected
    assert all(r["handoff"]["kv_payload_bytes"] > 0 for r in pd[1]["requests"])


def test_supervisor_idempotency_sealing_and_cancel(tmp_path, checkpoint, monkeypatch):
    import time
    from hqsb.console.config import Settings, DeploymentConfig
    from hqsb.console.service import ConsoleService
    from hqsb.console.parallel_jobs import ParallelJobs

    class Actor:
        def __init__(self, _):
            pass

        def close(self):
            pass

    cfg = DeploymentConfig(id="tiny", name="tiny", **checkpoint)
    settings = Settings(
        data_dir=tmp_path / "console",
        evidence_root=tmp_path,
        web_dist=tmp_path / "web",
        deployments=[cfg],
    )
    svc = ConsoleService(settings, actor_factory=Actor)
    jobs = ParallelJobs(settings, svc)
    monkeypatch.setattr(
        jobs, "hardware", lambda: {"devices": [], "nccl": False, "gloo": True}
    )
    plan = ParallelPlan(
        device_type="cpu",
        precision="float32",
        prompts=["t2 t3 t4"],
        max_output_tokens=3,
        trace_collectives=False,
    )

    def wait(identity):
        end = time.monotonic() + 50
        while time.monotonic() < end:
            result = svc.store.get(identity)
            if (
                result["state"] in {"completed", "failed", "cancelled", "timed_out"}
                and svc.pending == 0
            ):
                return result
            time.sleep(0.05)
        raise TimeoutError(identity)

    try:
        run = jobs.start("tiny", plan, "unique")
        assert jobs.start("tiny", plan, "unique")["id"] == run["id"]
        result = wait(run["id"])
        assert result["state"] == "completed", result.get("error")
        doc, manifest = jobs.read(run["id"])
        assert doc["status"] == "applied" and manifest["status"] == "complete"
        assert doc["requests"][0]["token_ids"]
        second = jobs.start("tiny", plan, "cancel")
        svc.cancel(second["id"])
        cancelled = wait(second["id"])
        assert cancelled["state"] == "cancelled" and cancelled["cleanup"] == "succeeded"
        assert not svc.pending and svc.benchmark_owner is None
        directory = settings.data_dir / "captures" / run["id"]
        (directory / "parallel-receipt.json").write_text("{}")
        with pytest.raises(ValueError, match="INTEGRITY"):
            jobs.read(run["id"])
    finally:
        jobs.close()
        svc.close()


def test_l9_inventory_and_exact_comparison_guard():
    from hqsb.core.optimization import catalogue
    from hqsb.console.parallel_routes import compare
    import copy

    entry = next(x for x in catalogue()["items"] if x["id"] == "parallel_plan")
    assert (
        entry["entrypoint"] == "/parallel"
        and entry["implementation_status"] == "session_api_only"
    )
    doc = {
        "status": "applied",
        "cleanup": "succeeded",
        "model_identity": "m",
        "tokenizer_identity": "t",
        "source_digest": "s",
        "plan": ParallelPlan().model_dump(),
        "ranks": [{"versions": {"torch": "x"}}],
        "requests": [{"index": 0, "input_token_hash": "in", "token_ids": [1, 2]}],
    }
    assert compare(doc, copy.deepcopy(doc))["comparable"]
    other = copy.deepcopy(doc)
    other["plan"]["strategy"] = "dp_hosts"
    result = compare(doc, other)
    assert result["comparable"] and not result["performance_comparable"]
    assert (
        result["performance_scope"]["output_tokens_per_s"] == "clock_boundary_mismatch"
    )
    b = copy.deepcopy(doc)
    b["requests"][0]["token_ids"] = [1, 3]
    assert not compare(doc, b)["comparable"]
    b = copy.deepcopy(doc)
    b["source_digest"] = "other"
    assert not compare(doc, b)["comparable"]


def test_same_key_retries_overlapping_slow_probe_only_reserve_once(
    tmp_path, checkpoint, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from hqsb.console.config import Settings, DeploymentConfig
    from hqsb.console.service import ConsoleService
    from hqsb.console.parallel_jobs import ParallelJobs

    class Actor:
        def __init__(self, _):
            pass

        def close(self):
            pass

    cfg = DeploymentConfig(id="tiny", name="tiny", **checkpoint)
    svc = ConsoleService(
        Settings(
            data_dir=tmp_path / "store",
            evidence_root=tmp_path,
            web_dist=tmp_path / "web",
            deployments=[cfg],
        ),
        actor_factory=Actor,
    )
    jobs = ParallelJobs(svc.settings, svc)
    barrier = threading.Barrier(2)
    release = threading.Event()
    calls = []

    def probe():
        barrier.wait(timeout=5)
        return {"devices": [], "gloo": True, "nccl": False}

    def no_device_execution(identity, *_):
        calls.append(identity)
        release.wait(10)
        with svc.lock:
            svc.pending -= 1
            svc.benchmark_owner = None

    monkeypatch.setattr(jobs, "hardware", probe)
    monkeypatch.setattr(jobs, "execute", no_device_execution)
    try:
        plan = ParallelPlan(device_type="cpu", precision="float32", prompts=["t2"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(jobs.start, "tiny", plan, "same-retry-key")
                for _ in range(2)
            ]
            rows = [f.result(timeout=10) for f in futures]
        assert rows[0]["id"] == rows[1]["id"] and svc.pending == 1
        release.set()
        jobs.close()
        assert calls == [rows[0]["id"]]
    finally:
        release.set()
        jobs.close()
        svc.close()
