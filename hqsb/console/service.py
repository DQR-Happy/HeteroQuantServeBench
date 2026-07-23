"""Deployment lifecycle, persistent requests and bounded device execution."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from hqsb.console.store import Store, TERMINAL
from hqsb.console.workers import Actor


class ServiceError(Exception):
    def __init__(self, code, message, status=409):
        super().__init__(message)
        self.code, self.status = code, status


class ConsoleService:
    def __init__(self, settings, actor_factory=Actor):
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
        self.actors = {
            x.id: actor_factory(x.model_dump()) for x in settings.deployments
        }
        self.executors = {
            x.id: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"hqsb-{x.id}")
            for x in settings.deployments
        }

    def deployments(self):
        with self.lock:
            return json.loads(json.dumps(list(self.states.values())))

    def _reserve(self):
        if self.closed:
            raise ServiceError("SHUTTING_DOWN", "Service is shutting down", 503)
        if self.pending >= self.settings.max_pending:
            raise ServiceError("QUEUE_FULL", "Request queue is full", 429)

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
            if kind == "generate":
                if state["state"] != "ready":
                    raise ServiceError(
                        "DEPLOYMENT_NOT_READY", "Load and warm the deployment first"
                    )
                if payload.get("expected_epoch") != state["epoch"]:
                    raise ServiceError(
                        "STALE_EPOCH", "Deployment changed; refresh before submitting"
                    )
                if payload["max_output_tokens"] > cfg.max_output_tokens:
                    raise ServiceError(
                        "TOKEN_LIMIT",
                        "Output length exceeds deployment capability",
                        422,
                    )
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
                "actual_runtime": json.loads(json.dumps(state["detail"]))
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
        try:
            with self.lock:
                if self.store.get(run_id)["state"] == "cancel_requested":
                    if kind == "load":
                        state["state"] = "unloaded"
                    if kind == "unload":
                        state["state"] = "ready"
                    self.store.update(
                        run_id,
                        {"state": "cancelled", "cleanup": "not_started"},
                        "cancelled",
                    )
                    return
                self.store.update(
                    run_id,
                    {
                        "state": "running",
                        "started_at": time.time(),
                        "metrics": {"queue_ms": (started - submitted) * 1000},
                    },
                )
            remaining = (
                payload.get("deadline_ms", self.settings.request_deadline_ms)
                - (started - submitted) * 1000
            )
            if kind == "generate" and remaining <= 0:
                self.store.update(
                    run_id, {"state": "timed_out", "cleanup": "not_started"}, "error"
                )
                return

            def on_output(event):
                nonlocal first_content
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
            result = actor.execute(
                kind,
                worker_payload,
                on_output,
                lambda: self.closed
                or self.store.get(run_id)["state"] == "cancel_requested",
                max(1, remaining / 1000) if kind == "generate" else 300,
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
                    )
            elif kind == "unload" or result.get("worker_terminated"):
                with self.lock:
                    state.update(
                        state="unloaded", epoch=None, detail={}, load_run_id=None
                    )
            current = self.store.get(run_id)
            metrics = {
                **current["metrics"],
                **result.get("metrics", {}),
                "console_first_content_ms": first_content,
                "console_e2e_ms": (time.monotonic() - submitted) * 1000,
            }
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
            actor.close()
            with self.lock:
                state.update(
                    state="failed",
                    epoch=None,
                    detail={},
                    load_run_id=None,
                    last_error=str(exc)[:800],
                )
            self.store.update(
                run_id,
                {
                    "state": "failed",
                    "cleanup": "worker_terminated",
                    "error": str(exc)[:800],
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
