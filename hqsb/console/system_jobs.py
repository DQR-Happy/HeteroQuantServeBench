"""L1 admission, guardian lifecycle and sealed paired evidence. No torch here."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from hqsb.console.service import ServiceError
from hqsb.console.store import TERMINAL
from hqsb.models.source_budget import weight_budget
from hqsb.backends.process_supervision import parent_identity
from hqsb.core.system_execution import SystemPlan, summarize, validate_system
from hqsb.core.optimization import fingerprint
from hqsb.observation.storage import atomic_json, sha256
from hqsb.experiments.artifacts import seal, read_manifest, read_json


class SystemJobs:
    def __init__(self, settings, service):
        self.settings, self.service = settings, service
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="l1")
        self.stop = threading.Event()
        self.probe_lock = threading.Lock()
        self.probed = None
        self.quarantined = False

    def hardware(self):
        with self.probe_lock:
            if not self.probed or time.monotonic() - self.probed[0] > 30:
                try:
                    proc = subprocess.run(
                        [sys.executable, "-m", "hqsb.backends.system_probe"],
                        capture_output=True,
                        text=True,
                        timeout=25,
                        check=True,
                    )
                    self.probed = (time.monotonic(), json.loads(proc.stdout))
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    raise ServiceError(
                        "L1_PROBE_FAILED",
                        str(exc) + str(getattr(exc, "stderr", ""))[-1000:],
                        503,
                    ) from exc
            return {
                **self.probed[1],
                "quarantined": self.quarantined or self.journal.exists(),
                "recovery_reason": "存在未确认清理或功耗恢复日志。请在执行节点检查 .console/l1-power-journal.json，恢复其 UUID 的原始功耗并回读；确认守护进程已退出后归档日志并重启服务。",
            }

    @property
    def journal(self):
        return self.settings.data_dir / "l1-power-journal.json"

    def start(self, deployment_id, plan, key):
        plan = SystemPlan.model_validate(plan)
        svc = self.service
        digest = fingerprint(["l1", deployment_id, plan.model_dump()])

        def existing():
            with svc.store.lock:
                old = svc.store.db.execute(
                    "SELECT digest,run_id FROM idempotency WHERE key=?", (key,)
                ).fetchone()
            if old:
                if old[0] != digest:
                    raise ServiceError("IDEMPOTENCY_CONFLICT", "Key already used")
                return svc.store.get(old[1])

        with svc.lock:
            old = existing()
            if old:
                return old
        hardware = self.hardware()
        try:
            validate_system(plan, hardware)
        except ValueError as exc:
            raise ServiceError("INVALID_SYSTEM_PLAN", str(exc), 422) from exc
        with svc.lock:
            old = existing()
            if old:
                return old
            svc._reserve()
            if self.quarantined or self.journal.exists():
                raise ServiceError("L1_RECOVERY_REQUIRED", hardware["recovery_reason"])
            if self.stop.is_set():
                raise ServiceError("SHUTTING_DOWN", "Service closing", 503)
            if svc.pending or svc.benchmark_owner:
                raise ServiceError("DEVICE_BUSY", "请在运行记录中等待或取消当前任务。")
            if any(
                d["state"] not in {"unloaded", "failed"}
                for d in svc.states.values()
                if d["provider"] == "pytorch"
            ):
                raise ServiceError(
                    "L1_UNLOAD_RESIDENT_MODELS",
                    "请在设备与部署页卸载驻留模型，L1 独立进程负责权重生命周期。",
                )
            cfg = svc.configs.get(deployment_id)
            if (
                cfg is None
                or cfg.provider != "pytorch"
                or not cfg.manifest
                or self.settings.privileged_worker
            ):
                raise ServiceError(
                    "L1_VERIFIED_NATIVE_SOURCE_REQUIRED",
                    "请在模型页登记带 manifest 的本机原始浮点 PyTorch 模型；不支持特权 worker。",
                    422,
                )
            root = Path(os.path.expandvars(cfg.model_path)).expanduser()
            from hqsb.backends.observation import host_memory
            from hqsb.models.model_admission import effective_host_available

            try:
                weights = weight_budget(root, 4 if plan.precision == "float32" else 2)
            except (OSError, ValueError) as exc:
                raise ServiceError("L1_SOURCE_BUDGET_FAILED", str(exc), 422) from exc
            available = effective_host_available(
                host_memory().get("host_available_bytes"), Path("/sys/fs/cgroup")
            )
            if available is None or available < weights + 512 * 1024**2:
                raise ServiceError(
                    "L1_HOST_MEMORY_BUDGET", "CPU 暂存预算超过主机/cgroup 可用内存。"
                )
            self.settings.data_dir.mkdir(parents=True, exist_ok=True)
            used = sum(
                p.stat().st_size
                for p in (self.settings.data_dir / "captures").rglob("*")
                if p.is_file()
            )
            if (
                shutil.disk_usage(self.settings.data_dir).free < 512 * 1024**2
                or used + 32 * 1024**2 > self.settings.artifact_budget_bytes
            ):
                raise ServiceError(
                    "L1_DISK_BUDGET",
                    "请在机器与存储页保留至少 512 MiB 可用空间和 32 MiB 观测配额。",
                )
            run, _ = svc.store.create(
                "system",
                {"deployment": cfg.public(), "system_plan": plan.model_dump()},
                key,
                digest,
            )
            svc.pending += 1
            svc.benchmark_owner = "l1:" + run["id"]
            self.executor.submit(
                self.execute, run["id"], cfg.model_dump(), plan, hardware
            )
            return run

    def execute(self, identity, config, plan, hardware):
        svc = self.service
        directory = self.settings.data_dir / "captures" / identity
        process, error, doc = None, None, None
        final, cleanup = "failed", "pending"
        try:
            from hqsb.observation.identity import environment

            source = environment(Path.cwd())
            directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            with svc.lock:
                if (
                    self.stop.is_set()
                    or svc.store.get(identity)["state"] == "cancel_requested"
                ):
                    final, cleanup = "cancelled", "succeeded"
                else:
                    svc.store.update(identity, {"state": "running"}, "started")
                    atomic_json(
                        directory / "input.json",
                        {
                            "run_id": identity,
                            "config": config,
                            "plan": plan.model_dump(),
                            "uuid": hardware["devices"][plan.device]["uuid"],
                            "journal": str(self.journal.resolve()),
                            "source_digest": source["source_digest"],
                            "supervisor_pid": os.getpid(),
                            "supervisor_start_ticks": parent_identity(os.getpid()),
                        },
                    )
                    with (directory / "guard.log").open("wb") as log:
                        # Guardian intentionally survives API death to roll back.
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "hqsb.backends.system_guard",
                                str(directory / "input.json"),
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
            if process:
                started = time.monotonic()
                progress = None
                while process.poll() is None:
                    if (
                        self.stop.is_set()
                        or svc.store.get(identity)["state"] == "cancel_requested"
                    ):
                        (directory / "cancel").touch()
                    if time.monotonic() - started > plan.timeout_s + 60:
                        (directory / "cancel").touch()
                        self.quarantined = True
                        raise RuntimeError(
                            "L1_GUARD_EXIT_UNCONFIRMED: inspect guardian and power journal before restart"
                        )
                    if (directory / "progress.json").exists():
                        current = read_json(directory / "progress.json")
                        if current != progress:
                            svc.store.update(
                                identity,
                                {"progress": current},
                                "system.progress",
                                current,
                            )
                            progress = current
                    self.stop.wait(0.2) if not self.stop.is_set() else time.sleep(0.2)
                if process.returncode:
                    self.quarantined = True
                    raise RuntimeError(
                        "L1_GUARD_CRASH: inspect guard.log and recovery journal"
                    )
                doc = read_json(directory / "guard-result.json")
                final, cleanup, error = doc["status"], doc["cleanup"], doc.get("error")
                if final == "completed":
                    trials = doc["trials"]
                    if len(trials) != plan.rounds * 2:
                        raise RuntimeError("L1_INCOMPLETE_TRIAL_SET")
                    signatures = [
                        fingerprint(
                            [
                                t["model_manifest_sha256"],
                                t["versions"],
                                t["effective"]["cuda_uuid"],
                                [r["input_token_hash"] for r in t["requests"]],
                            ]
                        )
                        for t in trials
                    ]
                    if len(set(signatures)) != 1:
                        raise RuntimeError("L1_MODEL_RUNTIME_OR_INPUT_IDENTITY_CHANGED")
                if cleanup != "succeeded":
                    self.quarantined = True
        except Exception as exc:
            error = str(exc)[:2600]
            final = "failed"
            if process is None:
                cleanup = "succeeded"  # No child or device-global mutation began.
            if process and process.poll() is None:
                # A control-plane/readback failure must also stop the workload.
                # Never SIGKILL the guardian: it owns global-state rollback.
                try:
                    (directory / "cancel").touch()
                    process.wait(timeout=20)
                    recovered = read_json(directory / "guard-result.json")
                    doc, cleanup = recovered, recovered["cleanup"]
                except (OSError, ValueError, KeyError, subprocess.SubprocessError):
                    self.quarantined = True
            if cleanup != "succeeded":
                self.quarantined = True
        finally:
            try:
                from hqsb.observation.identity import environment

                source = environment(Path.cwd())
                if doc and any(
                    t.get("source_digest") != source["source_digest"]
                    for t in doc.get("trials", [])
                ):
                    final, error = "failed", "L1_SOURCE_CHANGED_DURING_EXPERIMENT"
                receipt = {
                    "schema_version": "l1-v1",
                    "run_id": identity,
                    "status": "applied" if final == "completed" else final,
                    "cleanup": cleanup,
                    "error": error,
                    "plan": plan.model_dump(),
                    "hardware": hardware,
                    "source_digest": source["source_digest"],
                    **(doc or {}),
                }
                receipt["status"] = "applied" if final == "completed" else final
                receipt["error"] = error
                receipt["comparison"] = summarize(receipt.get("trials", []))
                receipt["comparison"]["complete_pair"] = (
                    final == "completed" and cleanup == "succeeded"
                )
                atomic_json(directory / "system-receipt.json", receipt)
                atomic_json(
                    directory / "execution-receipt.json",
                    {
                        "run_id": identity,
                        "status": receipt["status"],
                        "cleanup_status": cleanup,
                        "source_digest": source["source_digest"],
                        "system_receipt_sha256": sha256(
                            directory / "system-receipt.json"
                        ),
                    },
                )
                manifest = seal(
                    directory,
                    run_id=identity,
                    status="complete"
                    if final == "completed" and cleanup == "succeeded"
                    else "partial",
                    error=error,
                )
                svc.store.update(
                    identity,
                    {
                        "state": final,
                        "cleanup": cleanup,
                        "error": error,
                        "quality": (
                            "passed"
                            if receipt["comparison"]["greedy_tokens_exact"]
                            else "failed"
                        )
                        if final == "completed"
                        else "not_evaluated",
                        "metrics": {
                            "system_receipt": receipt,
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
                    if svc.benchmark_owner == "l1:" + identity:
                        svc.benchmark_owner = None

    def read(self, identity):
        run = self.service.store.get(identity)
        if run["kind"] != "system" or run["state"] not in TERMINAL:
            raise ServiceError(
                "L1_TERMINAL_RUN_REQUIRED", "请等待硬件系统实验完成并确认恢复。"
            )
        root = self.settings.data_dir / "captures" / identity
        manifest = read_manifest(root)
        doc = read_json(root / "system-receipt.json")
        if (
            doc["run_id"] != identity
            or doc != run["metrics"].get("system_receipt")
            or "system-receipt.json" not in {a["path"] for a in manifest["artifacts"]}
        ):
            raise ServiceError("L1_INTEGRITY", "System receipt identity mismatch")
        return doc, manifest

    def close(self):
        self.stop.set()
        self.executor.shutdown(wait=True)
