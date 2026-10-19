"""One fresh process per L1 profile: apply before torch pools/model allocation."""

import json
import os
from pathlib import Path
import resource
import threading
import time
from hqsb.core.system_execution import SystemPlan, SystemProfile
from hqsb.observation.storage import atomic_json, sha256
from hqsb.hardware.system import numa_policy, Nvidia, read, cuda_uuid


def process_sample():
    r = resource.getrusage(resource.RUSAGE_SELF)
    import psutil

    return {
        "rss_bytes": psutil.Process().memory_info().rss,
        "minor_faults": r.ru_minflt,
        "major_faults": r.ru_majflt,
        "voluntary_switches": r.ru_nvcsw,
        "involuntary_switches": r.ru_nivcsw,
        "cpu_seconds": r.ru_utime + r.ru_stime,
        "numa_maps": (read("/proc/self/numa_maps") or "")[:32768],
    }


def main(path, trial):
    from hqsb.backends.parallel_launcher import parent_lifetime

    parent_lifetime()
    job = json.loads(Path(path).read_text())
    plan = SystemPlan.model_validate(job["plan"])
    from hqsb.observation.identity import environment

    source = environment(Path.cwd())
    if source["source_digest"] != job["source_digest"]:
        raise RuntimeError("L1_SOURCE_CHANGED_BEFORE_TRIAL")
    profile = SystemProfile.model_validate(job["profile"])
    before_affinity = sorted(os.sched_getaffinity(0))
    if profile.cpu_ids is not None:
        if not set(profile.cpu_ids) <= set(before_affinity):
            raise ValueError("L1_CPU_AFFINITY_CHANGED_SINCE_ADMISSION")
        os.sched_setaffinity(0, profile.cpu_ids)
    numa = numa_policy(profile.numa_node) if profile.numa_node is not None else None
    # New threads inherit affinity and memory policy. Pools are configured before
    # model execution; interop is write-once, hence separate fresh processes.
    os.environ.update(
        OMP_NUM_THREADS=str(profile.intra_threads),
        MKL_NUM_THREADS=str(profile.intra_threads),
        TOKENIZERS_PARALLELISM="false",
    )
    import torch
    from hqsb.core.parallel_execution import ParallelPlan
    from hqsb.backends.parallel_model import load, generate

    torch.set_num_threads(profile.intra_threads)
    torch.set_num_interop_threads(profile.interop_threads)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    p = ParallelPlan(
        precision=plan.precision,
        prompts=plan.prompts,
        max_output_tokens=plan.max_output_tokens,
        fixed_output_length=True,
    )
    manifest_path = Path(os.path.expandvars(job["config"]["manifest"])).expanduser()
    manifest_hash = sha256(manifest_path)
    started = time.monotonic()
    model, tok, device, _ = load(job["config"], p, 0)
    encoded = [tok.encode(s, add_special_tokens=True) for s in plan.prompts]
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
        raise ValueError("L1_INPUT_CONTEXT_BUDGET_EXCEEDED")
    warmup = p.model_copy(update={"max_output_tokens": 2})
    generate(model, tok, encoded[0], device, warmup)
    torch.cuda.synchronize()
    load_ms = (time.monotonic() - started) * 1000
    torch.cuda.reset_peak_memory_stats()
    nvml, nvml_reason = None, None
    try:
        nvml = Nvidia()
    except Exception as exc:
        nvml_reason = str(exc)
    samples, stop = [], threading.Event()

    def take():
        if nvml:
            samples.append({"time_s": time.monotonic(), **nvml.sample(job["uuid"])})

    def sample_loop():
        while not stop.wait(0.25):
            if len(samples) < 1000:
                take()

    thread = threading.Thread(target=sample_loop, daemon=True)
    pre = process_sample()
    take()
    start = time.monotonic()
    thread.start()
    try:
        requests = []
        for i, ids in enumerate(encoded):
            result, cache = generate(model, tok, ids, device, p)
            requests.append({"index": i, **result})
            del cache
        torch.cuda.synchronize()
        workload_ms = (time.monotonic() - start) * 1000
    finally:
        stop.set()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("L1_SAMPLER_DID_NOT_EXIT")
        take()
        if nvml:
            nvml.close()
    post = process_sample()
    affinity = {
        t.name: sorted(os.sched_getaffinity(int(t.name)))
        for t in Path("/proc/self/task").iterdir()
    }
    expected = sorted(profile.cpu_ids or before_affinity)
    if any(a != expected for a in affinity.values()):
        raise RuntimeError("L1_THREAD_AFFINITY_READBACK_MISMATCH")
    counter = [s.get("total_energy_mj") for s in (samples[:1] + samples[-1:])]
    energy = (
        (counter[1] - counter[0]) / 1000
        if len(counter) == 2 and None not in counter and counter[1] >= counter[0]
        else None
    )
    power_pairs = list(zip(samples, samples[1:]))
    energy_estimate = (
        sum(
            (b["time_s"] - a["time_s"]) * (a["power_w"] + b["power_w"]) / 2
            for a, b in power_pairs
        )
        if power_pairs
        and all(
            a.get("power_w") is not None and b.get("power_w") is not None
            for a, b in power_pairs
        )
        else None
    )
    doc = {
        "run_id": job["run_id"],
        "source_digest": source["source_digest"],
        "trial": trial,
        "label": job["label"],
        "status": "completed",
        "requested": profile.model_dump(),
        "effective": {
            "cpu_ids": expected,
            "thread_affinities": affinity,
            "intra_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "numa_policy": numa,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_uuid": cuda_uuid(torch.cuda.get_device_properties(0).uuid),
            "power_limit_w": job.get("effective_power_w"),
        },
        "model_manifest_sha256": manifest_hash,
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
        },
        "load_and_warmup_ms": load_ms,
        "requests": requests,
        "samples": samples,
        "process_before": pre,
        "process_after": post,
        "metrics": {
            "workload_ms": workload_ms,
            "token_s": sum(len(r["token_ids"]) for r in requests)
            * 1000
            / sum(r["generation_ms"] for r in requests),
            "first_token_ms": sum(r["first_token_ms"] for r in requests)
            / len(requests),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "rss_bytes": post["rss_bytes"],
            "energy_j": energy,
            "energy_estimate_j": energy_estimate,
            **{
                k + "_delta": post[k] - pre[k]
                for k in (
                    "minor_faults",
                    "major_faults",
                    "voluntary_switches",
                    "involuntary_switches",
                    "cpu_seconds",
                )
            },
        },
        "energy_scope": "NVML board counter during measured workload, includes other board activity; no load/warmup, no CPU energy",
        "sampling": {
            "period_s": 0.25,
            "scope": "device-wide telemetry, not process utilization",
            "unavailable": nvml_reason,
            "counter_window_ms": (samples[-1]["time_s"] - samples[0]["time_s"]) * 1000
            if samples
            else None,
        },
    }
    if doc["effective"]["cuda_uuid"] != job["uuid"]:
        raise RuntimeError("L1_ACTUAL_GPU_UUID_MISMATCH")
    if sha256(manifest_path) != manifest_hash:
        raise RuntimeError("L1_MODEL_MANIFEST_CHANGED_DURING_TRIAL")
    atomic_json(Path(path).parent / f"trial-{trial}.json", doc)


if __name__ == "__main__":
    import sys

    main(sys.argv[1], int(sys.argv[2]))
