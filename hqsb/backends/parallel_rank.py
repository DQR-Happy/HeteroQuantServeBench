"""torchrun entrypoint. A rank owns weights, request KV and its process group."""

import contextlib
from datetime import timedelta
import json
import os
from pathlib import Path
import time
from hqsb.core.parallel_execution import ParallelPlan
from hqsb.core.optimization import fingerprint
from hqsb.observation.storage import atomic_json, sha256
from hqsb.backends.parallel_model import load, parameters, prefill, generate


def main(path):
    from hqsb.backends.parallel_launcher import parent_lifetime

    parent_lifetime()
    import torch
    import torch.distributed as dist
    from hqsb.backends.parallel_handoff import transfer

    job = json.loads(Path(path).read_text())
    plan = ParallelPlan.model_validate(job["plan"])
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    directory = Path(path).parent
    if plan.device_type == "cuda":
        torch.cuda.set_device(rank if plan.strategy != "layer_shard" else 0)
    group = world > 1
    if group:
        backend = (
            "nccl"
            if plan.device_type == "cuda" and plan.strategy in {"tp", "dp"}
            else "gloo"
        )
        dist.init_process_group(
            backend=backend, timeout=timedelta(seconds=min(60, plan.timeout_s))
        )
    model = None
    cache = None
    profiler = None
    doc = {
        "schema_version": "l9-rank-v1",
        "rank": rank,
        "world_size": world,
        "pid": os.getpid(),
        "run_id": job["run_id"],
        "status": "running",
        "group_backend": dist.get_backend() if group else None,
        "requests": [],
    }

    def progress(phase, completed=0):
        atomic_json(
            directory / f"progress-{rank}.json",
            {"rank": rank, "phase": phase, "completed_requests": completed},
        )

    try:
        progress("loading")
        torch.manual_seed(0)
        start = time.monotonic()
        model, tok, device, placement = load(job["config"], plan, rank)
        doc.update(
            load_ms=(time.monotonic() - start) * 1000,
            parameters=parameters(model),
            tp_plan=getattr(model, "_tp_plan", None),
            layer_placement=placement,
            load_strategy=model._l9_load_strategy,
        )
        doc["model_manifest_sha256"] = sha256(
            Path(os.path.expandvars(job["config"]["manifest"])).expanduser()
        )
        doc["tokenizer_sha256"] = fingerprint(
            json.loads(tok.backend_tokenizer.to_str())
        )
        doc["versions"] = {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
        }
        if plan.device_type == "cuda":
            doc["devices"] = (
                [
                    {"id": i, "name": torch.cuda.get_device_name(i)}
                    for i in range(len(plan.devices))
                ]
                if plan.strategy == "layer_shard"
                else [{"id": rank, "name": torch.cuda.get_device_name(rank)}]
            )
            for item in doc["devices"]:
                torch.cuda.reset_peak_memory_stats(item["id"])
        else:
            doc["devices"] = [{"id": rank, "name": "CPU diagnostic process; not a GPU"}]
        encoded = [tok.encode(p, add_special_tokens=True) for p in plan.prompts]
        limit = min(
            job["config"]["context_limit"],
            getattr(
                model.config, "max_position_embeddings", job["config"]["context_limit"]
            ),
        )
        if any(
            not ids or len(ids) > 512 or len(ids) + plan.max_output_tokens > limit
            for ids in encoded
        ):
            raise ValueError("L9_INPUT_OR_CONTEXT_BUDGET_EXCEEDED")
        if plan.strategy == "tp":
            if not any(
                p["global_shape"] != p["local_shape"]
                for p in doc["parameters"]["tensors"]
            ):
                raise ValueError("L9_TP_REQUESTED_BUT_NO_ACTUAL_SHARD_FOUND")
        if group:
            dist.barrier()
        progress("executing")
        started = time.monotonic()
        if plan.trace_collectives:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            )
        with profiler if profiler else contextlib.nullcontext():
            for index, ids in enumerate(encoded):
                if plan.strategy == "dp" and index % world != rank:
                    continue
                with torch.inference_mode():
                    if plan.strategy == "prefill_decode":
                        pstart = time.monotonic()
                        cache, first = (
                            prefill(model, ids, device) if rank == 0 else (None, None)
                        )
                        if device != "cpu":
                            torch.cuda.synchronize()
                        prefill_ms = (
                            (time.monotonic() - pstart) * 1000 if rank == 0 else None
                        )
                        cache, first, handoff = transfer(
                            cache, first, model, device, rank
                        )
                        kv_ready_ms = (time.monotonic() - pstart) * 1000
                        if rank == 1:
                            result, cache = generate(
                                model, tok, ids, device, plan, cache, first
                            )
                            result["handoff"] = handoff
                            result["first_token_ms"] += kv_ready_ms
                            result["first_token_scope"] = (
                                "decode-rank wait-for-KV plus local first token; not ingress TTFT"
                            )
                            doc["requests"].append({"index": index, **result})
                        else:
                            doc.setdefault("prefill", []).append(
                                {"index": index, "prefill_ms": prefill_ms, **handoff}
                            )
                    else:
                        result, cache = generate(model, tok, ids, device, plan)
                        doc["requests"].append({"index": index, **result})
                    cache = None
                    progress("executing", len(doc["requests"]) if rank else index + 1)
            if group:
                dist.barrier()
        doc["workload_ms"] = (time.monotonic() - started) * 1000
        doc["collectives"] = (
            [
                {
                    "name": e.key,
                    "calls": e.count,
                    "cpu_total_us": e.cpu_time_total,
                    "timing_scope": "profiler CPU event; may include enqueue, not completed GPU communication time",
                }
                for e in profiler.key_averages()
                if any(
                    k in e.key.lower()
                    for k in (
                        "c10d",
                        "nccl",
                        "gloo",
                        "allreduce",
                        "allgather",
                        "reduce_scatter",
                        "send",
                        "recv",
                    )
                )
            ]
            if profiler
            else None
        )
        if plan.device_type == "cuda":
            doc["allocator"] = [
                {
                    "id": d["id"],
                    "allocated_bytes": torch.cuda.memory_allocated(d["id"]),
                    "reserved_bytes": torch.cuda.memory_reserved(d["id"]),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(d["id"]),
                }
                for d in doc["devices"]
            ]
        doc["status"] = "completed"
        progress("completed", len(doc["requests"]))
    finally:
        cache = None
        model = None
        if group and dist.is_initialized():
            dist.destroy_process_group()
        doc["cleanup"] = "process_group_destroyed_and_process_exit_releases_weights"
        atomic_json(directory / f"rank-{rank}.json", doc)


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
