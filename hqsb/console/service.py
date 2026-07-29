"""Deployment lifecycle, persistent requests and bounded device execution."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from hqsb.console.store import Store, TERMINAL
from hqsb.console.workers import Actor
from hqsb.console.captures import safe_directory
from hqsb.console.resources import host_memory


class ServiceError(Exception):
    def __init__(self, code, message, status=409):
        super().__init__(message)
        self.code, self.status = code, status


class ConsoleService:
    def __init__(self, settings, actor_factory=None):
        self.settings = settings
        self.store = Store(settings.data_dir)
        self.store.recover()
        self.lock = threading.RLock()
        self.pending = 0
        self.closed = False
        self.configs = {x.id: x for x in settings.deployments}
        self.states = {
            x.id: {
                **x.public(),
                "state": "unloaded",
                "epoch": None,
                "detail": {},
                "last_error": None,
            }
            for x in settings.deployments
        }

        def make_actor(config):
            if actor_factory is not None:
                return actor_factory(config.model_dump())
            if settings.privileged_worker and config.provider == "pytorch":
                from hqsb.console.privileged_worker import PrivilegedActor

                return PrivilegedActor(config.model_dump(), data_dir=settings.data_dir)
            return Actor(config.model_dump())

        self.actors = {x.id: make_actor(x) for x in settings.deployments}
        self.executors = {
            x.id: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"hqsb-{x.id}")
            for x in settings.deployments
        }

    def deployments(self):
        with self.lock:
            # Tensor inventories have their own on-demand endpoint; do not copy
            # hundreds of rows into every overview poll or inference config.
            rows = []
            for state in self.states.values():
                row = {key: value for key, value in state.items() if key != "latest_kv"}
                row["detail"] = {
                    key: value
                    for key, value in state["detail"].items()
                    if key != "parameter_inventory"
                }
                rows.append(row)
            return json.loads(json.dumps(rows))

    def memory_inventory(self, deployment_id):
        with self.lock:
            row = self.states[deployment_id]
            return json.loads(
                json.dumps(
                    {
                        "schema_version": 1,
                        "deployment_id": deployment_id,
                        "source": "load_time_snapshot",
                        "sampled_at": row.get("loaded_at"),
                        "parameters": row.get("detail", {}).get("parameter_inventory"),
                        "latest_kv": row.get("latest_kv"),
                        "limitations": [
                            "权重按实际存储去重；逻辑字节和底层 storage 字节含义不同。",
                            "KV 来自最近一次请求结束前快照，并非当前仍驻留；临时 activation/workspace/驱动分配不在此账本。",
                            "storage_key 是当前进程内匿名标识，不是物理地址或跨进程稳定身份。",
                        ],
                    }
                )
            )

    def _reserve(self):
        if self.closed:
            raise ServiceError("SHUTTING_DOWN", "Service is shutting down", 503)
        if self.pending >= self.settings.max_pending:
            raise ServiceError("QUEUE_FULL", "Request queue is full", 429)

    def _preflight(self, kind, cfg, payload):
        mode = payload.get("observation_mode", "basic")
        if mode not in {"off", "basic", "operators"}:
            raise ServiceError("INVALID_OBSERVATION", "Unknown observation mode", 422)
        if kind == "quantize" or mode == "operators":
            if cfg.provider != "pytorch":
                raise ServiceError(
                    "UNSUPPORTED_PROVIDER",
                    "此操作需要本地 PyTorch worker，外部 HTTP 接口不能采内部算子或制作权重。",
                    422,
                )
            minimum = (
                self.settings.artifact_min_available_bytes
                if kind == "quantize"
                else self.settings.profile_min_available_bytes
            )
            available = host_memory().get("available_bytes")
            if available is not None and available < minimum:
                raise ServiceError(
                    "MEMORY_BUDGET",
                    f"Available memory {available} B is below the {minimum} B diagnostic budget",
                    409,
                )
            reserve = 512 * 1024**2
            if kind == "quantize":
                # Bound packed matrices + FP32 scales from the load-time ledger.
                # Include every 2-D parameter, even those the converter excludes.
                inventory = self.states[cfg.id]["detail"].get("parameter_inventory", {})
                estimate = 64 * 1024**2  # manifest / transaction bookkeeping
                for tensor in inventory.get("entries", []):
                    shape = tensor.get("shape", [])
                    if len(shape) == 2:
                        rows, columns = shape
                        packed = (rows * columns * payload["bits"] + 7) // 8
                        groups = (
                            (columns + payload["group_size"] - 1)
                            // payload["group_size"]
                            if payload["bits"] == 4
                            else 1
                        )
                        estimate += packed + rows * groups * 4
                reserve = max(2 * 1024**3, estimate)
            if shutil.disk_usage(self.settings.data_dir).free < reserve:
                raise ServiceError(
                    "DISK_BUDGET",
                    "Insufficient free disk for bounded diagnostic artifacts",
                    409,
                )
            used = sum(
                p.stat().st_size
                for name in ("captures", "artifacts")
                for p in (self.settings.data_dir / name).rglob("*")
                if p.is_file() and not p.is_symlink()
            )
            if used + reserve > self.settings.artifact_budget_bytes:
                raise ServiceError(
                    "ARTIFACT_BUDGET",
                    "Diagnostic artifact quota would be exceeded; archive existing captures first",
                    409,
                )

    def submit(self, kind, deployment_id, payload, key):
        with self.lock:
            if deployment_id not in self.configs:
                raise ServiceError("NOT_FOUND", "Deployment is not configured", 404)
            digest = hashlib.sha256(
                json.dumps([kind, deployment_id, payload], sort_keys=True).encode()
            ).hexdigest()
            # Look up existing operations before capability/state checks: retried load remains idempotent.
            with self.store.lock:
                old = self.store.db.execute(
                    "SELECT digest,run_id FROM idempotency WHERE key=?", (key,)
                ).fetchone()
            if old:
                if old[0] != digest:
                    raise ServiceError(
                        "IDEMPOTENCY_CONFLICT", "Key already used for another payload"
                    )
                return self.store.get(old[1])
            self._reserve()
            state, cfg = self.states[deployment_id], self.configs[deployment_id]
            if kind in {"generate", "quantize"}:
                if state["state"] != "ready":
                    raise ServiceError(
                        "DEPLOYMENT_NOT_READY", "Load and warm the deployment first"
                    )
                if payload.get("expected_epoch") != state["epoch"]:
                    raise ServiceError(
                        "STALE_EPOCH", "Deployment changed; refresh before submitting"
                    )
                if (
                    kind == "generate"
                    and payload["max_output_tokens"] > cfg.max_output_tokens
                ):
                    raise ServiceError(
                        "TOKEN_LIMIT",
                        "Output length exceeds deployment capability",
                        422,
                    )
                if kind == "quantize" and (
                    payload.get("bits") not in (4, 8)
                    or (
                        payload["bits"] == 4
                        and payload.get("group_size") not in (32, 64, 128)
                    )
                    or (payload["bits"] == 8 and payload.get("group_size") is not None)
                ):
                    raise ServiceError(
                        "QUANT_CONFIG",
                        "W4 requires group 32/64/128; W8 uses per-channel scaling",
                        422,
                    )
                self._preflight(kind, cfg, payload)
            elif kind == "load":
                if state["state"] not in {"unloaded", "failed"}:
                    raise ServiceError(
                        "DEPLOYMENT_BUSY", "Unload this deployment before loading again"
                    )
                if cfg.provider == "pytorch" and any(
                    x["provider"] == "pytorch"
                    and x["state"] not in {"unloaded", "failed"}
                    for k, x in self.states.items()
                    if k != deployment_id
                ):
                    raise ServiceError(
                        "DEVICE_BUSY", "Only one local model may own this GPU"
                    )
            elif kind == "unload":
                if state["state"] != "ready":
                    raise ServiceError(
                        "DEPLOYMENT_NOT_READY", "Only a ready deployment can be drained"
                    )
            else:
                raise ServiceError(
                    "UNSUPPORTED_JOB", "This job kind is not implemented", 422
                )
            safe_payload = dict(payload)
            if "messages" in safe_payload:
                messages = safe_payload.pop("messages")
                safe_payload["input_sha256"] = hashlib.sha256(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest()
                if payload.get("save_input"):
                    safe_payload["messages"] = messages
            config = {
                "deployment": cfg.public(),
                "epoch": state["epoch"],
                **safe_payload,
                "load_run_id": state.get("load_run_id") if kind != "load" else None,
                "actual_runtime": json.loads(
                    json.dumps(
                        {
                            key: value
                            for key, value in state["detail"].items()
                            if key not in {"observation", "parameter_inventory"}
                        }
                    )
                )
                if kind != "load"
                else {},
                "generation_policy": {
                    "sampling": "greedy",
                    "temperature": 0,
                    "chat_template_owner": "local_tokenizer"
                    if cfg.provider == "pytorch"
                    else "upstream",
                    "enable_thinking": False
                    if cfg.provider == "pytorch"
                    else "upstream_defined",
                },
            }
            run, _ = self.store.create(kind, config, key, digest)
            if kind == "load":
                state.update(state="loading", last_error=None)
            if kind == "unload":
                state["state"] = "draining"
            if kind == "quantize":
                state["state"] = "quantizing"
            self.pending += 1
            submitted = time.monotonic()
            self.executors[deployment_id].submit(
                self._execute, run["id"], kind, deployment_id, payload, submitted
            )
            return run

    def _execute(self, run_id, kind, deployment_id, payload, submitted):
        actor = self.actors[deployment_id]
        state = self.states[deployment_id]
        started, first_content = time.monotonic(), None
        resource_before = host_memory("before_" + kind)
        try:
            with self.lock:
                if self.store.get(run_id)["state"] == "cancel_requested":
                    if kind == "load":
                        state["state"] = "unloaded"
                    if kind == "unload":
                        state["state"] = "ready"
                    if kind == "quantize":
                        state["state"] = "ready"
                    self.store.update(
                        run_id,
                        {"state": "cancelled", "cleanup": "not_started"},
                        "cancelled",
                    )
                    return
                if kind in {"generate", "quantize"}:
                    try:
                        # Queued captures may have consumed disk since submit.
                        # A pre-execution rejection must preserve the ready model.
                        self._preflight(kind, self.configs[deployment_id], payload)
                    except ServiceError as exc:
                        if kind == "quantize":
                            state["state"] = "ready"
                        self.store.update(
                            run_id,
                            {
                                "state": "failed",
                                "cleanup": "not_started",
                                "error": str(exc),
                                "error_code": exc.code,
                            },
                            "error",
                        )
                        return
                self.store.update(
                    run_id,
                    {
                        "state": "running",
                        "started_at": time.time(),
                        "metrics": {
                            "queue_ms": (started - submitted) * 1000,
                            "resource_before": resource_before,
                        },
                    },
                )
            remaining = (
                self.settings.quantization_timeout_s * 1000
                if kind == "quantize"
                else payload.get("deadline_ms", self.settings.request_deadline_ms)
            ) - (started - submitted) * 1000
            if kind in {"generate", "quantize"} and remaining <= 0:
                if kind == "quantize":
                    with self.lock:
                        state["state"] = "ready"
                self.store.update(
                    run_id, {"state": "timed_out", "cleanup": "not_started"}, "error"
                )
                return

            def on_output(event):
                nonlocal first_content
                if event.get("kind") == "progress":
                    self.store.update(
                        run_id, {"progress": event}, "task.progress", event
                    )
                    return
                if event.get("text") and first_content is None:
                    first_content = (time.monotonic() - submitted) * 1000
                current = self.store.get(run_id)
                metrics = {
                    **current["metrics"],
                    **{
                        k: v
                        for k, v in event.items()
                        if k
                        in {"input_tokens", "output_tokens", "runtime_first_token_ms"}
                    },
                    "console_first_content_ms": first_content,
                }
                self.store.update(
                    run_id,
                    {"output": event.get("text", ""), "metrics": metrics},
                    "output.snapshot",
                    {"text": event.get("text", ""), "metrics": metrics},
                )

            worker_payload = {**payload, "remaining_ms": max(1, remaining)}
            if kind == "quantize":
                directory = safe_directory(self.settings.data_dir / "artifacts", run_id)
                directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                worker_payload.update(artifact_dir=str(directory), artifact_id=run_id)
            elif kind == "generate" and payload.get("observation_mode") == "operators":
                directory = safe_directory(self.settings.data_dir / "captures", run_id)
                directory.mkdir(parents=True, exist_ok=False, mode=0o700)
                worker_payload.update(capture_dir=str(directory), capture_id=run_id)
            result = actor.execute(
                kind,
                worker_payload,
                on_output,
                lambda: self.closed
                or self.store.get(run_id)["state"] == "cancel_requested",
                max(1, remaining / 1000) if kind in {"generate", "quantize"} else 300,
            )
            stopped = result.get("finish_reason") in {"cancelled", "timed_out"}
            if kind == "load" and not stopped:
                with self.lock:
                    state.update(
                        state="ready",
                        epoch=uuid.uuid4().hex,
                        detail=result,
                        last_error=None,
                        load_run_id=run_id,
                        loaded_at=time.time(),
                    )
            elif kind == "unload" or result.get("worker_terminated"):
                with self.lock:
                    state.update(
                        state="unloaded", epoch=None, detail={}, load_run_id=None
                    )
            elif kind == "quantize":
                with self.lock:
                    state["state"] = "ready"
            current = self.store.get(run_id)
            metrics = {
                **current["metrics"],
                **result.get("metrics", {}),
                "console_first_content_ms": first_content,
                "console_e2e_ms": (time.monotonic() - submitted) * 1000,
                "resource_after": host_memory("after_" + kind),
            }
            with self.lock:
                lifecycle = state.setdefault("memory_lifecycle", [])
                lifecycle.extend([resource_before, metrics["resource_after"]])
                del lifecycle[:-20]
                if kind == "generate":
                    state["latest_observation"] = {
                        "run_id": run_id,
                        "sampled_at": time.time(),
                        "memory": metrics.get("memory", {}),
                        "source": "request_end_snapshot",
                    }
                    state["latest_kv"] = {
                        "run_id": run_id,
                        "sampled_at": time.time(),
                        "inventory": metrics.get("kv_inventory"),
                    }
                elif kind in {"load", "unload"}:
                    state["latest_observation"] = None
                    state["latest_kv"] = None
            patch = {
                "state": result["finish_reason"] if stopped else "completed",
                "metrics": metrics,
                "cleanup": "worker_terminated"
                if result.get("worker_terminated")
                else "succeeded",
                "output": result.get("text", current["output"]),
                "finish_reason": result.get("finish_reason"),
                "result": result
                if kind != "generate"
                else {"finish_reason": result.get("finish_reason")},
            }
            self.store.update(
                run_id, patch, "completed" if not stopped else patch["state"]
            )
        except Exception as exc:
            cleanup = "worker_terminated"
            error = str(exc)[:800]
            try:
                actor.close()
            except Exception as cleanup_exc:
                cleanup = "unknown"
                error = (
                    f"{error}; worker cleanup could not be confirmed: {cleanup_exc}"[
                        :1200
                    ]
                )
            with self.lock:
                state.update(
                    state="failed",
                    epoch=None,
                    detail={},
                    load_run_id=None,
                    last_error=error,
                )
            self.store.update(
                run_id,
                {
                    "state": "failed",
                    "cleanup": cleanup,
                    "error": error,
                },
                "error",
            )
        finally:
            with self.lock:
                self.pending -= 1

    def cancel(self, run_id):
        with self.lock:
            run = self.store.get(run_id)
            if run["state"] not in TERMINAL and run["state"] != "cancel_requested":
                return self.store.update(run_id, {"state": "cancel_requested"})
            return run

    def close(self):
        with self.lock:
            self.closed = True
        for executor in self.executors.values():
            executor.shutdown(wait=True)
        for actor in self.actors.values():
            actor.close()
        self.store.close()
