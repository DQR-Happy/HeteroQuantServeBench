"""L9 process-group supervision and device lease. No torch in the API process."""

from concurrent.futures import ThreadPoolExecutor
import json
import fcntl
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from hqsb.console.service import ServiceError
from hqsb.console.store import TERMINAL
from hqsb.core.parallel_execution import ParallelPlan, validate_hardware
from hqsb.core.optimization import fingerprint
from hqsb.observation.storage import atomic_json, sha256
from hqsb.experiments.artifacts import seal, read_manifest, read_json


def weight_budget(root, size):
    elements = 0
    for path in root.glob("*.safetensors"):
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 8 * 1024**2:
                raise ValueError("L9_WEIGHT_HEADER_BUDGET")
            header = json.loads(f.read(n))
        for k, v in header.items():
            if k != "__metadata__":
                count = 1
                for dim in v["shape"]:
                    count *= dim
                elements += count
    if not elements:
        raise ValueError("L9_LOCAL_SAFETENSORS_REQUIRED")
    return elements * size


def mapped_visibility(devices):
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = inherited.split(",") if inherited is not None else None
    return ",".join(visible[i] if visible is not None else str(i) for i in devices)


def terminate(process):
    """The torchrun agent and all ranks share this private session/process group."""
    if process is None:
        return
    import psutil

    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # Also terminate orphan ranks when the agent has already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)
    _, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in alive):
        raise RuntimeError("L9_RANK_CLEANUP_NOT_CONFIRMED")


class ParallelJobs:
    def __init__(self, settings, service):
        self.settings, self.service = settings, service
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="l9")
        self.stop = threading.Event()
        self.probe_lock = threading.Lock()
        self.probed = None
        self.quarantined = False

    def hardware(self):
        with self.probe_lock:
            if self.probed and time.monotonic() - self.probed[0] < 30:
                return self.probed[1]
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "hqsb.backends.parallel_probe"],
                    capture_output=True,
                    text=True,
                    timeout=25,
                    check=True,
                )
                result = json.loads(proc.stdout)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                raise ServiceError(
                    "L9_HARDWARE_PROBE_FAILED",
                    (str(exc) + ": " + str(getattr(exc, "stderr", ""))[-1000:])[:1400],
                    503,
                ) from exc
            self.probed = (time.monotonic(), result)
            return result

    def start(self, deployment_id, plan, key):
        plan = ParallelPlan.model_validate(plan)
        svc = self.service
        digest = fingerprint(["l9", deployment_id, plan.model_dump()])
        with svc.lock:
            with svc.store.lock:
                old = svc.store.db.execute(
                    "SELECT digest,run_id FROM idempotency WHERE key=?", (key,)
                ).fetchone()
            if old:
                if old[0] != digest:
                    raise ServiceError("IDEMPOTENCY_CONFLICT", "Key already used")
                return svc.store.get(old[1])
        try:
            validate_hardware(plan, self.hardware())
        except ValueError as exc:
            raise ServiceError("INVALID_PARALLEL_PLAN", str(exc), 422) from exc
        with svc.lock:
            # A slow first hardware probe can overlap a same-key HTTP retry.
            # Recheck after reacquiring the mutation lock, before busy checks.
            with svc.store.lock:
                old = svc.store.db.execute(
                    "SELECT digest,run_id FROM idempotency WHERE key=?", (key,)
                ).fetchone()
            if old:
                if old[0] != digest:
                    raise ServiceError("IDEMPOTENCY_CONFLICT", "Key already used")
                return svc.store.get(old[1])
            svc._reserve()
            if self.quarantined:
                raise ServiceError(
                    "L9_CLEANUP_QUARANTINE",
                    "上次分布式进程清理未确认。请先检查节点残留进程并重启服务后恢复执行。",
                )
            if self.stop.is_set():
                raise ServiceError("SHUTTING_DOWN", "Service closing", 503)
            cfg = svc.configs.get(deployment_id)
            if (
                cfg is None
                or cfg.provider != "pytorch"
                or not cfg.manifest
                or self.settings.privileged_worker
            ):
                raise ServiceError(
                    "L9_LOCAL_VERIFIED_SOURCE_REQUIRED",
                    "需要登记过 manifest 的本机非特权 PyTorch 源模型。",
                    422,
                )
            if svc.pending or svc.benchmark_owner:
                raise ServiceError("DEVICE_BUSY", "请在运行记录中等待当前任务结束。")
            if any(
                d["state"] not in {"unloaded", "failed"}
                for d in svc.states.values()
                if d["provider"] == "pytorch"
            ):
                raise ServiceError(
                    "L9_UNLOAD_RESIDENT_MODELS",
                    "请先在设备与部署页卸载驻留模型；L9 独立进程负责分布式权重生命周期。",
                )
            root = Path(os.path.expandvars(cfg.model_path)).expanduser()
            try:
                weights = weight_budget(root, 4 if plan.precision == "float32" else 2)
            except (OSError, ValueError) as exc:
                raise ServiceError("L9_SOURCE_UNAVAILABLE", str(exc), 422) from exc
            from hqsb.backends.observation import host_memory
            from hqsb.models.model_admission import effective_host_available

            available = effective_host_available(
                host_memory().get("host_available_bytes"), Path("/sys/fs/cgroup")
            )
            # Ordinary CUDA replicas stage swappable CPU tensors before moving
            # the whole model to GPU; CPU diagnostics also keep host workspace.
            required_host = (
                weights * plan.ranks * (2 if plan.device_type == "cpu" else 1)
                + 512 * 1024**2
            )
            if available is None or available < required_host:
                raise ServiceError(
                    "L9_HOST_MEMORY_BUDGET",
                    "完整 CPU 暂存副本的保守预算超过主机/cgroup 可用内存。",
                )
            self.settings.data_dir.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(self.settings.data_dir).free < 512 * 1024**2:
                raise ServiceError(
                    "L9_DISK_BUDGET", "请在机器连接与存储页检查至少 512 MiB 证据余量。"
                )
            used = sum(
                p.stat().st_size
                for p in (self.settings.data_dir / "captures").rglob("*")
                if p.is_file()
            )
            if used + 64 * 1024**2 > self.settings.artifact_budget_bytes:
                raise ServiceError("L9_ARTIFACT_QUOTA", "请归档既有观测制品后重试。")
            run, _ = svc.store.create(
                "distributed",
                {
                    "deployment": cfg.public(),
                    "parallel_plan": plan.model_dump(),
                    "weight_bytes_estimate": weights,
                    "execution_scope": "single_host"
                    if plan.device_type == "cuda"
                    else "cpu_multi_process_diagnostic",
                },
                key,
                digest,
            )
            svc.benchmark_owner = "l9:" + run["id"]
            svc.pending += 1
            self.executor.submit(self.execute, run["id"], cfg.model_dump(), plan)
            return run

    def execute(self, identity, config, plan):
        svc = self.service
        directory = self.settings.data_dir / "captures" / identity
        process = None
        final = "failed"
        error = None
        ranks = []
        started = time.monotonic()
        locks = []
        try:
            directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            with svc.lock:
                if (
                    self.stop.is_set()
                    or svc.store.get(identity)["state"] == "cancel_requested"
                ):
                    final = "cancelled"
                    return
                svc.store.update(identity, {"state": "running"}, "started")
            if plan.device_type == "cuda":
                base = Path(
                    config.get("device_lock_path")
                    or str(Path.home() / ".cache/hqsb/device0.lock")
                )
                base.parent.mkdir(parents=True, exist_ok=True)
                for i in sorted(plan.devices):
                    handle = (
                        base if i == 0 else base.with_name(base.name + f".gpu{i}")
                    ).open("a")
                    locks.append(handle)
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_json(
                directory / "input.json",
                {"run_id": identity, "config": config, "plan": plan.model_dump()},
            )
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": mapped_visibility(plan.devices)
                if plan.device_type == "cuda"
                else "",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            }
            args = [
                sys.executable,
                "-m",
                "hqsb.backends.parallel_launcher",
                "--standalone",
                f"--nproc-per-node={plan.ranks}",
                "--max-restarts=0",
                "--module",
                "hqsb.backends.parallel_rank",
                str(directory / "input.json"),
            ]
            if plan.ranks == 1:
                args = [
                    sys.executable,
                    "-m",
                    "hqsb.backends.parallel_rank",
                    str(directory / "input.json"),
                ]
            with (directory / "worker.log").open("wb") as log:
                process = subprocess.Popen(
                    args,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
                last_progress = None
                while process.poll() is None:
                    cancelled = (
                        self.stop.is_set()
                        or svc.store.get(identity)["state"] == "cancel_requested"
                    )
                    if cancelled or time.monotonic() - started > plan.timeout_s:
                        final = "cancelled" if cancelled else "timed_out"
                        terminate(process)
                        break
                    progress = [
                        read_json(p) for p in sorted(directory.glob("progress-*.json"))
                    ]
                    if progress != last_progress:
                        svc.store.update(
                            identity,
                            {"progress": progress},
                            "parallel.progress",
                            {"ranks": progress},
                        )
                        last_progress = progress
                    self.stop.wait(0.15)
            if final not in {"cancelled", "timed_out"}:
                if process.returncode:
                    with (directory / "worker.log").open("rb") as f:
                        f.seek(max(0, (directory / "worker.log").stat().st_size - 2200))
                        error = f.read().decode(errors="replace")
                    raise ValueError(
                        f"L9_RANK_PROCESS_FAILED(exit={process.returncode}): " + error
                    )
                ranks = [
                    read_json(directory / f"rank-{i}.json") for i in range(plan.ranks)
                ]
                if any(
                    r["status"] != "completed" or r["run_id"] != identity for r in ranks
                ):
                    raise ValueError("L9_ALL_RANKS_COMPLETE_REQUIRED")
                final = "completed"
        except Exception as exc:
            error = str(exc)[:2600]
            final = "failed"
        finally:
            cleanup = "succeeded"
            try:
                terminate(process)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                cleanup = "failed"
                self.quarantined = True
                error = str(exc)
                final = "failed"
            try:
                # Never publish complete evidence until the private process group is gone.
                from hqsb.observation.identity import environment

                source = environment(Path.cwd())
                requests = []
                if ranks:
                    if plan.strategy == "dp":
                        requests = sorted(
                            [x for r in ranks for x in r["requests"]],
                            key=lambda r: r["index"],
                        )
                    else:
                        requests = ranks[1 if plan.strategy == "prefill_decode" else 0][
                            "requests"
                        ]
                    if [r["index"] for r in requests] != list(range(len(plan.prompts))):
                        raise ValueError("L9_REQUEST_PARTITION_INCOMPLETE")
                    if plan.strategy == "tp" and any(
                        [r["token_ids"] for r in rank["requests"]]
                        != [r["token_ids"] for r in requests]
                        for rank in ranks[1:]
                    ):
                        raise ValueError("L9_TP_RANK_TOKEN_DIVERGENCE")
                workload = max((r["workload_ms"] for r in ranks), default=0)
                count = sum(len(r["token_ids"]) for r in requests)
                doc = {
                    "schema_version": "l9-v1",
                    "run_id": identity,
                    "status": "applied"
                    if final == "completed" and cleanup == "succeeded"
                    else final,
                    "plan": plan.model_dump(),
                    "ranks": ranks,
                    "requests": requests,
                    "source_digest": source["source_digest"],
                    "cleanup": cleanup,
                    "model_identity": ranks[0]["model_manifest_sha256"]
                    if ranks
                    else None,
                    "tokenizer_identity": ranks[0]["tokenizer_sha256"]
                    if ranks
                    else None,
                    "workload_ms": workload or None,
                    "output_tokens": count,
                    "output_tokens_per_s": count * 1000 / workload
                    if workload
                    else None,
                    "job_total_ms": (time.monotonic() - started) * 1000,
                    "quality_scope": "actual greedy tokens per input, no task accuracy claim",
                    "timing_scope": "slowest rank workload wall time incl rank barriers; job total includes launcher, loading and exit",
                    "limitations": [
                        "No cross-host clock subtraction. Profiler CPU events are not GPU-completed NCCL timings.",
                        "Layer sharding executes sequentially; it is not overlapped pipeline scheduling.",
                        "DP is inference replica/request sharding, not training DDP gradient synchronization.",
                    ],
                }
                atomic_json(directory / "parallel-receipt.json", doc)
                atomic_json(
                    directory / "execution-receipt.json",
                    {
                        "run_id": identity,
                        "status": doc["status"],
                        "source_digest": source["source_digest"],
                        "model_identity": doc["model_identity"],
                        "cleanup_status": cleanup,
                        "parallel_receipt_sha256": sha256(
                            directory / "parallel-receipt.json"
                        ),
                    },
                )
                manifest = seal(
                    directory,
                    run_id=identity,
                    status="complete" if final == "completed" else "partial",
                    error=error,
                )
                svc.store.update(
                    identity,
                    {
                        "state": final,
                        "cleanup": cleanup,
                        "error": error,
                        "metrics": {
                            "parallel_receipt": doc,
                            "output_tokens": count,
                            "output_tokens_per_s": doc["output_tokens_per_s"],
                            "worker_request_total_ms": doc["job_total_ms"],
                            "experiment_bundle": manifest,
                        },
                    },
                    final,
                )
            except Exception as exc:
                svc.store.update(
                    identity,
                    {"state": "failed", "cleanup": cleanup, "error": str(exc)[:1600]},
                    "error",
                )
            finally:
                with svc.lock:
                    svc.pending -= 1
                    if svc.benchmark_owner == "l9:" + identity:
                        svc.benchmark_owner = None
                for handle in locks:
                    handle.close()

    def read(self, identity):
        run = self.service.store.get(identity)
        if run["kind"] != "distributed" or run["state"] not in TERMINAL:
            raise ServiceError(
                "L9_TERMINAL_RUN_REQUIRED", "请等待分布式任务结束与清理。"
            )
        directory = self.settings.data_dir / "captures" / identity
        manifest = read_manifest(directory)
        doc = read_json(directory / "parallel-receipt.json")
        if (
            doc["run_id"] != identity
            or doc != run["metrics"].get("parallel_receipt")
            or "parallel-receipt.json" not in {a["path"] for a in manifest["artifacts"]}
        ):
            raise ServiceError("L9_INTEGRITY", "L9 receipt identity mismatch")
        return doc, manifest

    def close(self):
        self.stop.set()
        self.executor.shutdown(wait=True)
