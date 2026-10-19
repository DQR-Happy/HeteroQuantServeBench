"""L5 process supervision. Owns private CPU jobs, never the live GPU model."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from hqsb.console.service import ServiceError
from hqsb.core.structure_plan import StructurePlan
from hqsb.core.optimization import fingerprint
from hqsb.observation.storage import atomic_json, sha256
from hqsb.backends.structure_artifacts import resolve, verify


def read_manifest(root, store, identity):
    directory = resolve(root, identity)
    file = directory / "l5-manifest.json"
    if not file.exists() or file.is_symlink() or file.stat().st_size > 4 * 1024**2:
        raise ValueError("L5_MANIFEST_INVALID")
    try:
        doc = json.loads(file.read_text())
        run = store.get(doc["creation_run_id"])
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError("L5_ARTIFACT_REGISTRY_UNAVAILABLE") from exc
    if (
        run["state"] != "completed"
        or doc.get("artifact_id") != identity
        or doc.get("publication") != "supervisor_acknowledged"
        or sha256(file) != run["metrics"]["structure_artifact"]["manifest_sha256"]
    ):
        raise ValueError("L5_CREATION_RECEIPT_IDENTITY")
    return doc


class StructureJobs:
    def __init__(self, settings, service):
        self.settings, self.service = settings, service
        self.root = settings.data_dir / "l5-artifacts"
        self.jobs = settings.data_dir / "l5-jobs"
        self.stop = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="l5-cpu")

    def start(self, deployment_id, epoch, plan, key):
        plan = StructurePlan.model_validate(plan)
        digest = fingerprint([deployment_id, epoch, plan.model_dump()])
        svc = self.service
        with svc.lock:
            with svc.store.lock:
                old = svc.store.db.execute(
                    "SELECT digest,run_id FROM idempotency WHERE key=?", (key,)
                ).fetchone()
            if old:
                if old[0] != digest:
                    raise ServiceError(
                        "IDEMPOTENCY_CONFLICT", "Key already used for another payload"
                    )
                return svc.store.get(old[1])
            if self.stop.is_set() or svc.closed:
                raise ServiceError("SHUTTING_DOWN", "Service closing", 503)
            if deployment_id not in svc.configs:
                raise ServiceError("NOT_FOUND", "Unknown deployment", 404)
            state, cfg = svc.states[deployment_id], svc.configs[deployment_id]
            if state["state"] != "ready" or state["epoch"] != epoch:
                raise ServiceError(
                    "STALE_OR_UNLOADED_MODEL", "请先在设备页加载并刷新原模型。"
                )
            if svc.pending or svc.benchmark_owner:
                raise ServiceError(
                    "DEVICE_BUSY", "请等待当前任务完成后再创建压缩制品。"
                )
            if (
                cfg.provider != "pytorch"
                or self.settings.privileged_worker
                or state.get("kernel_profile", "framework-native") != "framework-native"
                or (state.get("quantization") or {}).get("method", "none") != "none"
                or ((state.get("quantization") or {}).get("kv") or {}).get("bits")
                or state.get("structure_artifact_id")
            ):
                raise ServiceError(
                    "L5_ORIGINAL_NATIVE_FLOAT_MODEL_REQUIRED",
                    "请在结构页恢复原始浮点模型与框架原生算子；压缩任务不叠加量化或已有 L5 制品。",
                    422,
                )
            if not cfg.manifest:
                raise ServiceError(
                    "L5_VERIFIED_SOURCE_REQUIRED", "请先完成源模型制品校验。", 422
                )
            from hqsb.backends.observation import host_memory
            from hqsb.models.model_admission import effective_host_available

            required = state["detail"].get("parameter_bytes", 0) * 4 + 1024**3
            available = effective_host_available(
                host_memory().get("host_available_bytes"), Path("/sys/fs/cgroup")
            )
            if available is None or required > available:
                raise ServiceError(
                    "L5_HOST_MEMORY_BUDGET",
                    "CPU 独立副本及工作空间估计超过主机/容器可用内存。",
                )
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            artifact_reserve = state["detail"].get("parameter_bytes", 0) + 64 * 1024**2
            if (
                used + artifact_reserve > self.settings.artifact_budget_bytes
                or shutil.disk_usage(self.root).free < artifact_reserve + 512 * 1024**2
            ):
                raise ServiceError(
                    "L5_ARTIFACT_BUDGET",
                    "L5 制品空间或磁盘余量不足，请检查目标机器磁盘。",
                )
            safe = plan.model_dump(exclude={"calibration"})
            safe.update(
                calibration_text_sha256=[
                    hashlib.sha256(s.encode()).hexdigest() for s in plan.calibration
                ],
                calibration_count=len(plan.calibration),
            )
            run, _ = svc.store.create(
                "compress",
                {
                    "deployment": cfg.public(),
                    "epoch": epoch,
                    "structure_plan": safe,
                    "creation_device": "cpu_isolated",
                    "load_run_id": state.get("load_run_id"),
                },
                key,
                digest,
            )
            svc.benchmark_owner = "l5:" + run["id"]
            svc.pending += 1
            self.executor.submit(
                self._execute,
                run["id"],
                cfg.model_dump(),
                state.get("precision", cfg.precision),
                plan.model_dump(),
            )
            return run

    def _execute(self, identity, config, precision, plan):
        svc = self.service
        artifact_id = "l5_" + identity.removeprefix("run_")
        stage = self.root / (artifact_id + ".partial")
        destination = resolve(self.root, artifact_id)
        directory = self.jobs / identity
        process = None
        final = "failed"
        try:
            directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            deadline = time.monotonic() + self.settings.quantization_timeout_s
            payload = {
                "config": config,
                "plan": plan,
                "precision": precision,
                "stage": str(stage),
                "artifact_id": artifact_id,
                "progress": str(directory / "progress.json"),
                "deadline": deadline,
            }
            atomic_json(directory / "input.json", payload)
            with svc.lock:
                if (
                    svc.store.get(identity)["state"] == "cancel_requested"
                    or self.stop.is_set()
                ):
                    svc.store.update(
                        identity,
                        {"state": "cancelled", "cleanup": "not_started"},
                        "cancelled",
                    )
                    return
                svc.store.update(identity, {"state": "running"}, "started")
            with (directory / "worker.log").open("wb") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "hqsb.backends.structure_job",
                        str(directory / "input.json"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": "",
                        "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2",
                    },
                )
                last = None
                while process.poll() is None:
                    cancelled = (
                        svc.store.get(identity)["state"] == "cancel_requested"
                        or self.stop.is_set()
                    )
                    if cancelled or time.monotonic() > deadline:
                        final = "cancelled" if cancelled else "timed_out"
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                        break
                    progress = directory / "progress.json"
                    if progress.exists():
                        value = json.loads(progress.read_text())
                        if value != last:
                            svc.store.update(
                                identity, {"progress": value}, "task.progress", value
                            )
                            last = value
                    self.stop.wait(0.2)
            if final in {"cancelled", "timed_out"}:
                svc.store.update(
                    identity, {"state": final, "cleanup": "succeeded"}, final
                )
                return
            if process.returncode:
                with (directory / "worker.log").open("rb") as stream:
                    stream.seek(
                        max(0, (directory / "worker.log").stat().st_size - 1800)
                    )
                    raise ValueError(
                        "L5_WORKER_FAILED:" + stream.read().decode(errors="replace")
                    )
            if (
                svc.store.get(identity)["state"] == "cancel_requested"
                or self.stop.is_set()
            ):
                svc.store.update(
                    identity,
                    {"state": "cancelled", "cleanup": "succeeded"},
                    "cancelled",
                )
                return
            doc = verify(stage, published=False)
            used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            if used > self.settings.artifact_budget_bytes:
                raise ValueError("L5_ARTIFACT_STORE_BUDGET")
            with svc.lock:
                if (
                    svc.store.get(identity)["state"] == "cancel_requested"
                    or self.stop.is_set()
                ):
                    svc.store.update(
                        identity,
                        {"state": "cancelled", "cleanup": "succeeded"},
                        "cancelled",
                    )
                    return
                doc.update(
                    publication="supervisor_acknowledged", creation_run_id=identity
                )
                atomic_json(stage / "l5-manifest.json", doc)
                stage.rename(destination)
                digest = sha256(destination / "l5-manifest.json")
                svc.store.update(
                    identity,
                    {
                        "state": "completed",
                        "cleanup": "succeeded",
                        "metrics": {
                            "structure_artifact": {
                                "artifact_id": artifact_id,
                                "manifest_sha256": digest,
                                "statistics": doc["statistics"],
                                "transforms": doc["transforms"],
                                "disk_bytes": doc["disk_bytes"],
                                "creation_seconds": doc["creation_seconds"],
                                "publication": doc["publication"],
                            }
                        },
                    },
                    "completed",
                )
        except Exception as exc:
            svc.store.update(
                identity,
                {
                    "state": "failed",
                    "cleanup": "succeeded"
                    if process is None or process.poll() is not None
                    else "unknown",
                    "error": str(exc)[:2000],
                },
                "error",
            )
        finally:
            cleanup_errors = []
            try:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                if stage.exists():
                    shutil.rmtree(stage)
            except Exception as exc:
                # A failed kill/wait must not strand the global device queue.
                # Keep uncertain stage files and disable further CPU builds.
                cleanup_errors.append(str(exc))
            finally:
                try:
                    (directory / "input.json").unlink(missing_ok=True)
                except OSError as exc:
                    cleanup_errors.append(str(exc))
                try:
                    if cleanup_errors:
                        self.stop.set()
                        previous = svc.store.get(identity).get("error") or ""
                        svc.store.update(
                            identity,
                            {
                                "cleanup": "unknown",
                                "error": (
                                    previous
                                    + "; L5_CLEANUP_FAILED: "
                                    + "; ".join(cleanup_errors)
                                )[:2000],
                            },
                            "cleanup.error",
                        )
                finally:
                    with svc.lock:
                        if svc.benchmark_owner == "l5:" + identity:
                            svc.benchmark_owner = None
                        svc.pending -= 1

    def manifest(self, identity):
        return read_manifest(self.root, self.service.store, identity)

    def close(self):
        self.stop.set()
        self.executor.shutdown(wait=True, cancel_futures=False)
