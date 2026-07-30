#!/usr/bin/env python3
"""Remote Qwen block intervention screening on four frozen policy samples.

This is exploratory fake-dequant teacher forcing, not a selected mixed policy,
full PPL/task evaluation, low-bit execution, or six-workload performance claim.
Original E05-05/raw is never modified.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT/"docs/stage_experiments/S05/E05-05/model_diagnostics"
POLICY = ROOT/"docs/stage_experiments/S05/E05-03/raw/data/policy_validation_manifest.jsonl"
ARTIFACT = ROOT/"docs/stage_experiments/S05/E05-02/raw/artifacts/rtn_w4"
MODEL = Path("~/models/hqsb/Qwen3-1.7B").expanduser()
MODEL_MANIFEST = ROOT/"docs/benchmark/model_sha256_manifest.txt"
ROLES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n")
    tmp.replace(path)


def append(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n")


def read(path):
    return json.loads(Path(path).read_text())


def load_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def canonical_ids_hash(ids):
    # E05-03 calibration.token_ids_hash uses unsigned little-endian uint32s.
    digest = hashlib.sha256()
    for token in ids:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def rebuild_policy(tokenizer):
    candidates = load_rows(POLICY)
    selected, seen = [], set()
    for row in candidates:
        if row["split"] != "policy-validation":
            raise ValueError("Policy manifest contains a non-policy sample")
        key = row["domain"], row["length_bucket"]
        if key in seen:
            continue
        seen.add(key)
        relative = row["parent_id"]
        # Only the selected policy source files are read; never build E03's full corpus.
        revision, expected_sha = row["revision"].split(";sha256:")
        current = ROOT/relative
        if current.is_file() and sha(current) == expected_sha:
            payload = current.read_bytes(); source_mode = "current file exact hash"
        else:
            git_revision = revision.removeprefix("git:")
            payload = subprocess.check_output(["git", "show", f"{git_revision}:{relative}"], cwd=ROOT)
            source_mode = "frozen git source because working tree changed"
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"Frozen source hash mismatch for {relative}")
        source_file = OUT/"sources"/(expected_sha+".txt")
        source_file.parent.mkdir(parents=True, exist_ok=True); source_file.write_bytes(payload)
        all_ids = tokenizer(payload.decode("utf-8", errors="replace"), add_special_tokens=False)["input_ids"]
        offset = int(row["sample_id"].rsplit(":", 2)[1])
        ids = all_ids[offset:offset+row["num_tokens"]]
        observed_hash = canonical_ids_hash(ids)
        if len(ids) != row["num_tokens"] or observed_hash != row["token_hash"]:
            raise ValueError(f"Policy token hash mismatch: {row['sample_id']}")
        selected.append({**row, "token_ids": ids, "source_mode": source_mode,
                         "source_sha256_verified": actual_sha,
                         "token_hash_verified": observed_hash, "offset": offset,
                         "source_snapshot": str(source_file.relative_to(OUT))})
    if len(selected) != 4:
        raise ValueError(f"Expected four domain x length representatives, got {len(selected)}")
    write(OUT/"policy_samples.json", {"selection": "first manifest row per (domain,length_bucket); frozen before intervention",
                                      "source_manifest": str(POLICY.relative_to(ROOT)), "source_sha256": sha(POLICY),
                                      "final_evaluation_read": False, "samples": selected})
    return selected


def prepare():
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("Run on Jetson through scripts/remote_run.sh")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=True)
    rebuild_policy(tokenizer)
    spec = {"experiment_id": "E05-05", "scope": "exploratory model teacher-forcing block screening",
            "frozen_at_utc": now(), "model": str(MODEL), "model_manifest_sha256": sha(MODEL_MANIFEST),
            "quant_artifact_manifest_sha256": sha(ARTIFACT/"manifest.json"),
            "runner_sha256": sha(__file__), "intervention_unit": "whole transformer block, seven Linear projections",
            "module_roles": list(ROLES), "blocks": list(range(28)),
            "conditions": ["fp16"]+[f"single_w4_block_{i:02d}" for i in range(28)]+["all_w4"]+[f"loo_fp16_block_{i:02d}" for i in range(28)],
            "condition_count": 58, "sample_count": 4, "condition_sample_count": 232,
            "scheme": "reuse E05-02 canonical symmetric INT4 group128 artifact; whole-weight FP16 dequant before forward",
            "quality": "all non-final positions teacher-forcing next-token NLL, KL(ref||candidate), logit cosine, top1",
            "hidden": "propagated block output error vs FP16 same tokens; not frozen-input local error",
            "scope_limits": ["no final set read", "no policy selection", "no full-dataset PPL/task", "no free-running generation",
                             "no native low-bit kernel", "no six-workload steady timing or energy claim"],
            "memory": "one loaded FP16 model; restore original parameters one module at a time from safetensors; reference logits on disk",
            "model_forward": {"use_cache": False, "attention_backend": "eager", "dtype": "float16", "dropout": "eval"},
            "screening_metric": "mean teacher-forcing KL across four fixed policy samples; ranking is diagnostic only",
            "performance_fields": "instrumented forward wall time is diagnostic and includes hidden hooks; not latency claim"}
    write(OUT/"spec.json", spec)
    (OUT/"collector_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"phase": "prepare", "samples_verified": 4, "conditions": 58}), flush=True)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def execute():
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("Run on Jetson through scripts/remote_run.sh")
    if (OUT/"quality/fp16.jsonl").exists():
        raise RuntimeError("Refusing to overwrite an existing screening run")
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from hqsb.models.loader import load_qwen3
    from hqsb.quant.model_weight_only import load_manifest, _safe_name
    torch.set_num_threads(2); torch.manual_seed(50521)
    free, total = torch.cuda.mem_get_info()
    if free < 4_400_000_000:
        raise RuntimeError(f"Insufficient safe free memory for isolated model diagnostic: {free}")
    spec = read(OUT/"spec.json")
    if spec["model_manifest_sha256"] != sha(MODEL_MANIFEST) or spec["quant_artifact_manifest_sha256"] != sha(ARTIFACT/"manifest.json"):
        raise ValueError("Model or artifact manifest drifted after prepare")
    tload = time.perf_counter()
    tokenizer, model, reported_load = load_qwen3(str(MODEL), dtype=torch.float16,
                                               attention_backend="eager", verify_manifest=str(MODEL_MANIFEST),
                                               allow_extra=("model_sha256_manifest.txt",), cpu_staging=True)
    model.eval()
    samples = rebuild_policy(tokenizer)
    if any(p.device.type != "cuda" or p.dtype != torch.float16 for p in model.parameters()):
        raise RuntimeError("Expected complete FP16 CUDA model; offload is not accepted")
    env = {"started_at_utc": now(), "hostname": platform.node(), "platform": platform.platform(),
           "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
           "free_bytes_before_load": free, "total_bytes": total, "load_wall_s": time.perf_counter()-tload,
           "reported_load_s": reported_load, "model_parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()),
           "model_parameter_device": "cuda", "model_parameter_dtype": "float16", "runner_sha256": sha(__file__),
           "model_manifest_sha256": sha(MODEL_MANIFEST), "quant_artifact_manifest_sha256": sha(ARTIFACT/"manifest.json")}
    write(OUT/"environment.json", env)
    named = dict(model.named_modules())
    blocks = [named[f"model.layers.{i}"] for i in range(28)]
    names = [[f"model.layers.{i}.{role}" for role in ROLES] for i in range(28)]
    manifest = load_manifest(ARTIFACT)
    if manifest["source"]["model_sha256"] != sha(MODEL_MANIFEST):
        raise ValueError("RTN artifact source identity mismatch")
    by_name = {entry["name"]: (idx, entry) for idx, entry in enumerate(manifest["tensors"])}
    if len(by_name) != 196 or set(by_name) != {n for bb in names for n in bb}:
        raise ValueError("Expected exact 28x7 projection artifact coverage")
    packed_bytes = {}
    validated = []
    for name, (idx, entry) in by_name.items():
        directory = ARTIFACT/"tensors"/_safe_name(idx, name)
        for key in ("qvalues", "scales"):
            rec = entry["files"][key]; p = directory/rec["path"]
            if p.stat().st_size != rec["bytes"] or sha(p) != rec["sha256"]:
                raise ValueError(f"Invalid source packed payload: {name}/{key}")
        packed_bytes[name] = sum(entry["files"][key]["bytes"] for key in ("qvalues", "scales"))
        validated.append({"name": name, "shape": entry["source_shape"], "packed_bytes": packed_bytes[name],
                          "files": entry["files"], "relative_directory": str(directory.relative_to(ROOT))})
    write(OUT/"source_artifact_validation.json", {"artifact_id": manifest["artifact_id"], "tensors": validated,
                                                  "validated_payload_count": 392, "all_passed": True})
    # Persistent file handles only map checkpoint pages; no full CUDA backup exists.
    weight_index = MODEL/"model.safetensors.index.json"
    if weight_index.exists():
        shard_map = read(weight_index)["weight_map"]
    else:
        with safe_open(str(MODEL/"model.safetensors"), framework="pt", device="cpu") as f:
            shard_map = {key: "model.safetensors" for key in f.keys()}
    state = {"condition": None, "sample_index": None, "baseline": False}
    active_w4 = set()
    baseline_hashes = {}
    hidden_rows = []
    forward_count = 0
    with contextlib.ExitStack() as stack:
        shards = {name: stack.enter_context(safe_open(str(MODEL/name), framework="pt", device="cpu"))
                  for name in sorted(set(shard_map.values()))}

        @torch.inference_mode()
        def install_block(index, mode):
            start = time.perf_counter(); modules = []
            for name in names[index]:
                weight = named[name].weight
                if mode == "fp16":
                    key = name+".weight"
                    original = shards[shard_map[key]].get_tensor(key).to(dtype=torch.float16)
                    expected_hash = tensor_hash(original)
                    weight.copy_(original)
                    actual_hash = tensor_hash(weight)
                    if actual_hash != expected_hash:
                        raise ValueError(f"FP16 restore hash mismatch for {name}")
                    previous_hash = baseline_hashes.get(name)
                    if previous_hash is not None and previous_hash != actual_hash:
                        raise ValueError(f"Checkpoint-cast FP16 differs from baseline: {name}")
                    baseline_hashes[name] = actual_hash
                    modules.append({"name": name, "mode": mode, "restored_sha256": actual_hash, "exact": True})
                    del original
                else:
                    if name not in baseline_hashes:
                        baseline_hashes[name] = tensor_hash(weight)
                    idx, entry = by_name[name]
                    r, c = map(int, entry["source_shape"]); g = 128; ng = math.ceil(c/g)
                    directory = ARTIFACT/"tensors"/_safe_name(idx, name)
                    packed = np.memmap(directory/entry["files"]["qvalues"]["path"], dtype=np.uint8, mode="r", shape=(r, math.ceil(c/2)))
                    scales = np.memmap(directory/entry["files"]["scales"]["path"], dtype="<f4", mode="r", shape=(r, ng))
                    for begin in range(0, r, 256):
                        end = min(begin+256, r)
                        p = torch.from_numpy(np.array(packed[begin:end], copy=True)).to("cuda")
                        lo, hi = (p & 15).to(torch.int8), (p >> 4).to(torch.int8)
                        unsigned = torch.stack((lo, hi), -1).flatten(-2)[..., :c]
                        q = torch.where(unsigned >= 8, unsigned-16, unsigned)
                        s = torch.from_numpy(np.array(scales[begin:end], copy=True)).to("cuda")
                        deq = (q.reshape(end-begin, ng, g).float()*s[..., None]).reshape(end-begin, c)
                        weight[begin:end].copy_(deq.to(torch.float16))
                    modules.append({"name": name, "mode": mode, "canonical_payload_sha256": entry["files"]["qvalues"]["sha256"],
                                    "dequantized_fp16_sha256": tensor_hash(weight), "packed_bytes": packed_bytes[name]})
                    del p, lo, hi, unsigned, q, s, deq, packed, scales
            if mode == "w4":
                active_w4.add(index)
            else:
                active_w4.discard(index)
            torch.cuda.synchronize()
            append(OUT/"interventions.jsonl", {"at_utc": now(), "block": index, "mode": mode,
                                               "active_w4_blocks": sorted(active_w4), "modules": modules,
                                               "installation_wall_s": time.perf_counter()-start})

        def block_hook(index):
            def hook(module, inputs, output):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                tensor = tensor.detach()
                refpath = OUT/f"reference/sample_{state['sample_index']}/block_{index:02d}.npy"
                if state["baseline"]:
                    refpath.parent.mkdir(parents=True, exist_ok=True)
                    np.save(refpath, tensor.cpu().numpy())
                    tf = tensor.float()
                    hidden_rows.append({"condition": state["condition"], "sample_index": state["sample_index"],
                                        "block": index, "absmax": float(tf.abs().max()), "rms": float(tf.square().mean().sqrt()),
                                        "shape": list(tensor.shape), "reference_file": str(refpath.relative_to(OUT))})
                else:
                    reference = torch.from_numpy(np.load(refpath)).to("cuda").float()
                    tf = tensor.float(); delta = tf-reference
                    norm = reference.square().sum().sqrt()
                    hidden_rows.append({"condition": state["condition"], "sample_index": state["sample_index"],
                                        "block": index, "max_abs": float(delta.abs().max()), "rmse": float(delta.square().mean().sqrt()),
                                        "relative_l2": float(delta.square().sum().sqrt()/norm.clamp_min(1e-30)),
                                        "cosine": float(F.cosine_similarity(reference.flatten()[None], tf.flatten()[None], dim=-1)[0]),
                                        "absmax": float(tf.abs().max()), "rms": float(tf.square().mean().sqrt()),
                                        "kind": "propagated full-model hidden output error"})
            return hook

        def activation_hook(name):
            def hook(module, inputs):
                if not state["baseline"]:
                    return
                t = inputs[0].detach().float()
                append(OUT/"activation_fp16.jsonl", {"sample_index": state["sample_index"], "module": name,
                                                       "shape": list(t.shape), "min": float(t.min()), "max": float(t.max()),
                                                       "absmax": float(t.abs().max()), "rms": float(t.square().mean().sqrt()),
                                                       "finite": bool(torch.isfinite(t).all()), "scope": "real FP16 projection input"})
            return hook
        hooks = [b.register_forward_hook(block_hook(i)) for i, b in enumerate(blocks)]
        hooks += [named[n].register_forward_pre_hook(activation_hook(n)) for bb in names for n in bb]

        @torch.inference_mode()
        def evaluate(label):
            nonlocal forward_count
            state["condition"] = label; state["baseline"] = label == "fp16"
            started = time.perf_counter()
            for si, sample in enumerate(samples):
                state["sample_index"] = si; hidden_rows.clear()
                ids = torch.tensor([sample["token_ids"]], dtype=torch.long, device="cuda")
                torch.cuda.synchronize(); t0 = time.perf_counter()
                result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                torch.cuda.synchronize(); elapsed = time.perf_counter()-t0
                logits = result.logits[0]
                refpath = OUT/f"reference/sample_{si}/logits.npy"
                if state["baseline"]:
                    np.save(refpath, logits.detach().cpu().numpy())
                ref_np = np.load(refpath, mmap_mode="r")
                count = ids.shape[1]-1
                per_position = []; nll_sum = ref_nll_sum = kl_sum = cos_sum = 0.; agree = 0
                for begin in range(0, count, 16):
                    end = min(count, begin+16)
                    candidate = logits[begin:end].float()
                    reference = torch.from_numpy(np.array(ref_np[begin:end], copy=True)).to("cuda").float()
                    labels = ids[0, begin+1:end+1]
                    lp = F.log_softmax(candidate, -1); rp = F.log_softmax(reference, -1)
                    nll = -lp.gather(1, labels[:, None]).squeeze(1)
                    ref_nll = -rp.gather(1, labels[:, None]).squeeze(1)
                    kl = (rp.exp()*(rp-lp)).sum(-1)
                    cos = F.cosine_similarity(candidate, reference, -1)
                    eq = candidate.argmax(-1) == reference.argmax(-1)
                    nll_sum += float(nll.sum()); ref_nll_sum += float(ref_nll.sum()); kl_sum += float(kl.sum()); cos_sum += float(cos.sum()); agree += int(eq.sum())
                    for j in range(end-begin):
                        per_position.append({"position": begin+j, "next_token": int(labels[j]), "nll": float(nll[j]),
                                             "fp16_nll": float(ref_nll[j]), "kl": float(kl[j]), "cosine": float(cos[j]), "top1_agreement": bool(eq[j])})
                    del candidate, reference, lp, rp, nll, ref_nll, kl, cos, eq
                if not all(math.isfinite(v) for v in (nll_sum, ref_nll_sum, kl_sum, cos_sum)):
                    raise ValueError(f"Non-finite model quality at {label}/{si}")
                quantized_names = [n for b in active_w4 for n in names[b]]
                covered_fp16 = sum(named[n].weight.numel()*2 for n in quantized_names)
                hypothetical_packed = env["model_parameter_bytes"]-covered_fp16+sum(packed_bytes[n] for n in quantized_names)
                row = {"condition": label, "sample_index": si, "sample_id": sample["sample_id"], "domain": sample["domain"],
                       "length_bucket": sample["length_bucket"], "input_tokens": ids.shape[1], "scored_tokens": count,
                       "token_hash": sample["token_hash"], "active_w4_blocks": sorted(active_w4),
                       "next_token_nll": nll_sum/count, "fp16_next_token_nll": ref_nll_sum/count,
                       "delta_nll": (nll_sum-ref_nll_sum)/count, "teacher_kl_mean": kl_sum/count,
                       "mean_logit_cosine": cos_sum/count, "top1_agreement": agree/count,
                       "sample_teacher_ppl": math.exp(nll_sum/count), "sample_teacher_ppl_ratio": math.exp((nll_sum-ref_nll_sum)/count),
                       "instrumented_forward_wall_s": elapsed, "runtime_weight_bytes": env["model_parameter_bytes"],
                       "logical_packed_policy_bytes": hypothetical_packed,
                       "memory_claim": "runtime remains all FP16; logical packed bytes are representation accounting only",
                       "cuda_allocated_bytes": torch.cuda.memory_allocated(), "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "positions": per_position}
                append(OUT/f"quality/{label}.jsonl", row)
                for hrow in hidden_rows:
                    append(OUT/f"hidden/{label}.jsonl", hrow)
                forward_count += 1
                del logits, result, ids, ref_np
            summary = {"condition": label, "finished_at_utc": now(), "wall_s": time.perf_counter()-started,
                       "forward_count_cumulative": forward_count, "active_w4_blocks": sorted(active_w4)}
            append(OUT/"progress.jsonl", summary)
            print(json.dumps(summary), flush=True)

        try:
            evaluate("fp16")
            for b in range(28):
                install_block(b, "w4"); evaluate(f"single_w4_block_{b:02d}"); install_block(b, "fp16")
            for b in range(28):
                install_block(b, "w4")
            evaluate("all_w4")
            for b in range(28):
                install_block(b, "fp16"); evaluate(f"loo_fp16_block_{b:02d}"); install_block(b, "w4")
            for b in range(28):
                install_block(b, "fp16")
            write(OUT/"restore_validation.json", {"all_fp16_restored": not active_w4, "restored_projection_count": len(baseline_hashes),
                                                  "original_fp16_sha256": baseline_hashes,
                                                  "method": "checkpoint tensor cast to FP16, exact byte SHA256 comparison after every restore",
                                                  "model_instance_scope": "private diagnostic process only; no console model mutated"})
        finally:
            for hook in hooks:
                hook.remove()
    env["completed_at_utc"] = now(); env["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    env["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write(OUT/"environment.json", env)
    write(OUT/"execution_status.json", {"status": "COMPLETED", "conditions": 58, "forward_passes": forward_count,
                                        "final_evaluation_read": False, "at_utc": now()})
    print(json.dumps({"phase": "execute", "completed": True, "forward_passes": forward_count}), flush=True)


def finalize():
    spec = read(OUT/"spec.json")
    summary = []
    for condition in spec["conditions"]:
        p = OUT/f"quality/{condition}.jsonl"
        rows = load_rows(p) if p.exists() else []
        if len(rows) != 4:
            raise ValueError(f"Incomplete condition {condition}: {len(rows)}/4")
        total_tokens = sum(r["scored_tokens"] for r in rows)
        metrics = {field: sum(r[field] for r in rows)/len(rows) for field in
                   ("next_token_nll", "delta_nll", "teacher_kl_mean", "mean_logit_cosine", "top1_agreement", "sample_teacher_ppl_ratio")}
        summary.append({"condition": condition, "samples": 4, "scored_tokens": total_tokens,
                        **metrics, "token_weighted_nll": sum(r["next_token_nll"]*r["scored_tokens"] for r in rows)/total_tokens,
                        "logical_packed_policy_bytes": rows[0]["logical_packed_policy_bytes"],
                        "runtime_weight_bytes": rows[0]["runtime_weight_bytes"],
                        "aggregation": "arithmetic mean over four fixed samples; no CI/generalization claim"})
    by_name = {r["condition"]: r for r in summary}
    all_w4 = by_name["all_w4"]
    singles = sorted((r for r in summary if r["condition"].startswith("single_")), key=lambda r: r["teacher_kl_mean"], reverse=True)
    loos = sorted((dict(r, kl_recovery=all_w4["teacher_kl_mean"]-r["teacher_kl_mean"],
                       nll_recovery=all_w4["next_token_nll"]-r["next_token_nll"]) for r in summary if r["condition"].startswith("loo_")),
                  key=lambda r: r["kl_recovery"], reverse=True)
    write(OUT/"summary.json", {"experiment_id": "E05-05", "scope": "real model exploratory block screening on four policy samples",
                              "status": "SCREENING_COMPLETED", "formal_experiment_overall": "BLOCKED",
                              "conditions": summary, "single_block_ranking": singles, "leave_one_out_ranking": loos,
                              "final_evaluation_read": False, "deployable_policy_selected": False,
                              "remaining": ["module-level separate effects", "pair/cumulative interactions", "budget-constrained policy and random control",
                                            "policy freeze", "heldout final/task/free-running", "six-workload performance", "actual low-bit kernel policy"]})
    write(OUT/"verdict.json", {"overall": "SCREENING_COMPLETED", "formal_experiment_overall": "BLOCKED",
                              "complete_conditions": 58, "samples_per_condition": 4, "model_forwards": 232,
                              "passed": {"policy_source_hash": True, "policy_token_hash": True, "artifact_payload_hash": True,
                                         "single_block_all_28": True, "leave_one_out_all_28": True,
                                         "fp16_restore_exact": read(OUT/"restore_validation.json")["all_fp16_restored"]},
                              "not_claimed": ["full E05-05 PASS", "final quality", "selected mixed policy", "native W4 execution", "runtime memory reduction", "stable model latency"]})
    write(OUT/"EVIDENCE_MANIFEST.json", {"experiment_id": "E05-05", "subexperiment": "model_diagnostics",
                                        "created_at_utc": now(), "original_raw_modified": False,
                                        "files": [{"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": sha(p)}
                                                  for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "EVIDENCE_MANIFEST.json"]})
    print(json.dumps({"phase": "finalize", "conditions": len(summary), "top_single": singles[:3], "top_loo": loos[:3]}, ensure_ascii=False), flush=True)


def verify():
    manifest = read(OUT/"EVIDENCE_MANIFEST.json")
    for record in manifest["files"]:
        p = OUT/record["path"]
        assert p.stat().st_size == record["bytes"] and sha(p) == record["sha256"], p
    assert len(read(OUT/"summary.json")["conditions"]) == 58
    print(json.dumps({"phase": "verify", "files": len(manifest["files"]), "passed": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "execute", "finalize", "verify"))
    args = parser.parse_args()
    try:
        globals()[args.phase]()
    except Exception as exc:
        if args.phase == "execute":
            write(OUT/"execution_status.json", {"status": "FAILED", "exception": type(exc).__name__, "message": str(exc), "at_utc": now()})
        raise
