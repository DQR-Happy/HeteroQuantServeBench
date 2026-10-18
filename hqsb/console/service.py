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
        self.benchmark_owner = None
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
            actor_config = {**config.model_dump(), "l4_artifact_root": str(settings.data_dir / "l4-artifacts"),
                            "l4_artifact_budget_bytes": settings.artifact_budget_bytes}
            if actor_factory is not None:
                return actor_factory(actor_config)
            if settings.privileged_worker and config.provider == "pytorch":
                from hqsb.console.privileged_worker import PrivilegedActor

                return PrivilegedActor(config.model_dump(), data_dir=settings.data_dir)
            return Actor(actor_config)

        self._make_actor = make_actor
        self.actors = {x.id: make_actor(x) for x in settings.deployments}
        self.executors = {
            x.id: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"hqsb-{x.id}")
            for x in settings.deployments
        }

    def register_deployment(self, config):
        """Publish a verified model config without loading it or changing GPU ownership."""
        with self.lock:
            if self.closed:
                raise ServiceError("SHUTTING_DOWN", "Service is shutting down", 503)
            if config.id in self.configs:
                if self.configs[config.id] != config:
                    raise ServiceError("DEPLOYMENT_ID_CONFLICT", "Deployment ID already belongs to another artifact")
                return
            if len(self.configs) >= 16:
                raise ServiceError("DEPLOYMENT_LIMIT", "At most sixteen registered deployments")
            actor = self._make_actor(config)
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"hqsb-{config.id}")
            self.configs[config.id] = config
            self.actors[config.id] = actor
            self.executors[config.id] = executor
            self.states[config.id] = {**config.public(), "state": "unloaded", "epoch": None, "detail": {}, "last_error": None}

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
                    if key not in {"parameter_inventory", "module_inventory"}
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
        if mode not in {"off", "basic", "operators", "memory", "pages"}:
            raise ServiceError("INVALID_OBSERVATION", "Unknown observation mode", 422)
        if kind == "quantize" or payload.get("capture_architecture") or payload.get("experiment_role") == "quality" or mode in {"operators", "memory", "pages"} or (mode == "basic" and cfg.provider == "pytorch"):
            if cfg.provider != "pytorch":
                raise ServiceError(
                    "UNSUPPORTED_PROVIDER",
                    "此操作需要本地 PyTorch worker，外部 HTTP 接口不能采内部算子或制作权重。",
                    422,
                )
            minimum = (
                self.settings.artifact_min_available_bytes
                if kind == "quantize"
                else 128 * 1024**2 if mode == "basic" else self.settings.profile_min_available_bytes
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
                for name in ("captures", "artifacts", "experiments")
                for p in (self.settings.data_dir / name).rglob("*")
                if p.is_file() and not p.is_symlink()
            )
            if used + reserve > self.settings.artifact_budget_bytes:
                raise ServiceError(
                    "ARTIFACT_BUDGET",
                    "Diagnostic artifact quota would be exceeded; archive existing captures first",
                    409,
                )

    def claim_benchmark(self, owner, deployment_id):
        with self.lock:
            self._reserve()
            if self.benchmark_owner or self.pending:
                raise ServiceError("BENCHMARK_BUSY", "请等待当前任务或性能对照结束后再开始。")
            if deployment_id not in self.configs or self.configs[deployment_id].provider != "pytorch":
                raise ServiceError("BENCHMARK_PROVIDER", "固定输入对照目前需要本地 PyTorch 部署。", 422)
            if any(row["provider"] == "pytorch" and row["id"] != deployment_id and row["state"] not in {"unloaded", "failed"} for row in self.states.values()):
                raise ServiceError("DEVICE_BUSY", "其他部署正占用本地模型设备。")
            self.benchmark_owner = owner

    def release_benchmark(self, owner):
        with self.lock:
            if self.benchmark_owner == owner:
                self.benchmark_owner = None

    def token_budget(self, deployment_id, payload):
        """Ephemeral preview on the same serial worker; never starts inference."""
        with self.lock:
            self._reserve()
            if self.benchmark_owner or self.pending:
                raise ServiceError("DEVICE_BUSY", "任务正在执行，请稍后重新检查输入额度。")
            cfg = self.configs.get(deployment_id)
            state = self.states.get(deployment_id)
            if cfg is None or cfg.provider != "pytorch":
                raise ServiceError("UNSUPPORTED_PROVIDER", "输入额度检查需要本地模型。", 422)
            if state["state"] != "ready" or payload["expected_epoch"] != state["epoch"]:
                raise ServiceError("STALE_EPOCH", "模型尚未就绪或已变化，请刷新后重新检查。")
            request = {**payload, "input_mode": "completion" if payload.get("prompt") is not None else "chat"}
            if cfg.input_mode == "completion" and request["input_mode"] != "completion":
                raise ServiceError("INPUT_MODE", "该模型需要原始文本续写。", 422)
            self.pending += 1
            def preview():
                actor = self.actors[deployment_id]
                try:
                    result = actor.execute("token_budget", request, lambda _: None, lambda: self.closed, 15)
                    if result.get("worker_terminated"):
                        raise RuntimeError("输入额度检查 worker 已终止")
                    return result
                except Exception:
                    try:
                        actor.close()
                    finally:
                        with self.lock:
                            state.update(state="failed", epoch=None, detail={}, load_run_id=None,
                                         last_error="输入额度检查失败，请重新加载模型。")
                    raise
                finally:
                    with self.lock:
                        self.pending -= 1
            future = self.executors[deployment_id].submit(preview)
        result = future.result()
        if result.get("request_error"):
            raise ServiceError("INVALID_INPUT", result["request_error"], 422)
        return result

    def submit(self, kind, deployment_id, payload, key, *, benchmark_owner=None):
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
            if self.benchmark_owner and self.benchmark_owner != benchmark_owner:
                raise ServiceError("BENCHMARK_BUSY", "性能对照正在独占执行设备，请在性能与质量对照页等待或取消。")
            self._reserve()
            state, cfg = self.states[deployment_id], self.configs[deployment_id]
            if kind == "generate":
                if payload.get("capture_architecture") and (cfg.provider != "pytorch" or payload["max_output_tokens"] > 4 or payload.get("experiment_role")):
                    raise ServiceError("ARCHITECTURE_CAPTURE", "运行架构采集需要本地模型，输出预算最多 4 tokens，且单独作为诊断任务执行。", 422)
                completion = payload.get("prompt") is not None
                if completion:
                    if cfg.provider != "pytorch" or payload.get("messages") or not isinstance(payload["prompt"], str) or not payload["prompt"].strip() or len(payload["prompt"]) > 32768:
                        raise ServiceError("INPUT_MODE", "原始文本续写需要非空 prompt，不能同时提交聊天消息，且仅支持本地 PyTorch Provider", 422)
                elif cfg.input_mode == "completion":
                    raise ServiceError("INPUT_MODE", "该部署没有官方聊天模板，请使用原始文本续写 prompt", 422)
                payload = {**payload, "input_mode": "completion" if completion else "chat"}
                from hqsb.core.optimization import resolve_plan, validate_quality_plan
                from hqsb.observation.contracts import resolve_observation

                try:
                    resolved = resolve_plan(payload.get("optimization"), {
                        "cleanup_policy": payload.get("cleanup_policy", "empty_cache"),
                        "attention_backend": cfg.attention, "output_budget": payload["max_output_tokens"]},
                        provider=cfg.provider, deployment_attention=cfg.attention,
                        deployment_kernel_profile=state.get("kernel_profile", "framework-native"))
                    resolve_observation(payload.get("observation_plan"), payload.get("observation_mode", "basic"))
                    if payload.get("experiment_role") == "quality":
                        validate_quality_plan(payload.get("quality_plan"))
                        if resolved["resolved"]["output_budget"] > 32:
                            raise ValueError("QUALITY_GENERATION_LIMIT")
                    payload = {**payload, "max_output_tokens": resolved["resolved"]["output_budget"]}
                    if payload.get("capture_architecture") and payload["max_output_tokens"] > 4:
                        raise ValueError("ARCHITECTURE_CAPTURE_OUTPUT_LIMIT")
                except (ValueError, KeyError) as exc:
                    raise ServiceError("INVALID_EXPERIMENT_PLAN", str(exc), 422) from exc
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
                        f"请求输出预算 {payload['max_output_tokens']} 超过部署输出上限 {cfg.max_output_tokens} tokens；"
                        f"部署输入＋输出总窗口为 {cfg.context_limit} tokens",
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
                if kind == "quantize" and (state.get("precision", cfg.precision) != "float16" or state.get("quantization", {}).get("method", "none") != "none"):
                    raise ServiceError("QUANT_PRECISION_UNSUPPORTED", "当前交互量化制品通路仅验收了 FP16 源模型", 422)
                self._preflight(kind, cfg, payload)
            elif kind == "load":
                from hqsb.core.kernel_profiles import profile, requires_native
                from hqsb.core.precision import PrecisionPlan

                try:
                    selected = payload.get("kernel_profile", cfg.kernel_profile)
                    precision = payload.get("precision", cfg.precision)
                    quantization = PrecisionPlan.model_validate(payload.get("quantization", {}))
                    if self.settings.privileged_worker and (quantization.method != "none" or quantization.kv_bits):
                        raise ValueError("L4_WORKER_UNSUPPORTED: quantization requires an unprivileged local worker")
                    profile(selected)
                    if set(payload) - {"kernel_profile", "precision", "quantization"} or (cfg.provider != "pytorch" and payload):
                        raise ValueError("KERNEL_PROFILE_LOAD_INVALID")
                    if precision not in {"float32", "float16", "bfloat16"}:
                        raise ValueError("L4_PRECISION_UNSUPPORTED")
                    if selected != "framework-native" and (precision != "float16" or quantization.method != "none" or quantization.kv_bits):
                        raise ValueError("L4_CUSTOM_L2_KERNEL_NOT_QUANTIZATION_VERIFIED")
                    if profile(selected)["operators"]:
                        from hqsb.console.kernel_inventory import model_kernel_compatibility

                        incompatible = model_kernel_compatibility(cfg)
                        if incompatible:
                            raise ValueError(incompatible)
                    if requires_native(selected):
                        from hqsb.console.kernel_inventory import native_inventory

                        inventory = native_inventory(cfg)
                        if inventory["status"] != "files_verified":
                            raise ValueError(inventory["reason"])
                except ValueError as exc:
                    raise ServiceError("INVALID_KERNEL_PROFILE", str(exc), 422) from exc
                if cfg.provider == "pytorch":
                    # A failed live worker may retain its last attempted profile.
                    # Always send the resolved selection, including the default.
                    payload = {**payload, "kernel_profile": selected,
                               "precision": precision}
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
            if benchmark_owner is not None:
                # Accepted lifecycle jobs remain traceable even if the manager
                # exits between SQLite acceptance and its journal update.
                safe_payload["benchmark_id"] = benchmark_owner
            if "quality_plan" in safe_payload:
                quality_plan = safe_payload["quality_plan"]
                safe_payload["quality_plan"] = {**quality_plan, "samples": [
                    {"id": sample.get("id", str(index)), "text_sha256": hashlib.sha256(sample["text"].encode()).hexdigest(),
                     "expected_present": "expected" in sample, "slice": sample.get("slice", "all")}
                    for index, sample in enumerate(quality_plan["samples"])]}
            if "messages" in safe_payload:
                messages = safe_payload.pop("messages")
                safe_payload["input_sha256"] = hashlib.sha256(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest()
                if payload.get("save_input"):
                    safe_payload["messages"] = messages
            if "prompt" in safe_payload:
                prompt = safe_payload.pop("prompt")
                safe_payload["input_sha256"] = hashlib.sha256(
                    json.dumps({"input_mode": "completion", "prompt": prompt}, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest()
                if payload.get("save_input"):
                    safe_payload["prompt"] = prompt
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
                            if key not in {"observation", "parameter_inventory", "module_inventory"}
                        }
                    )
                )
                if kind != "load"
                else {},
                "generation_policy": {
                    "input_mode": payload.get("input_mode", cfg.input_mode),
                    "sampling": "greedy",
                    "temperature": 0,
                    "chat_template_owner": "none_raw_text" if payload.get("input_mode") == "completion" else "local_tokenizer"
                    if cfg.provider == "pytorch"
                    else "upstream",
                    "enable_thinking": "not_applicable" if payload.get("input_mode") == "completion" else state["detail"].get("chat_template_policy", {}).get("enable_thinking", "model_template_default")
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
        directory = None
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
            elif kind == "generate" and self.configs[deployment_id].provider == "pytorch" and (payload.get("capture_architecture") or payload.get("observation_mode", "basic") != "off" or payload.get("optimization") is not None or payload.get("experiment_role") is not None):
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
            if kind == "generate" and directory is not None and (directory / "execution-receipt.json").exists():
                from hqsb.experiments.artifacts import seal

                result.setdefault("metrics", {})["experiment_bundle"] = seal(
                    directory, run_id=run_id, status="complete" if result.get("finish_reason") not in {"cancelled", "timed_out", "rejected"} else "partial")
            if kind == "generate" and result.get("metrics", {}).get("observation_bundle"):
                from hqsb.observation.storage import publish

                try:
                    manifest = publish(directory, expected_run_id=run_id)
                    result["metrics"]["observation_bundle"].update(
                        publication=manifest["publication"], capture_status=manifest["capture_status"])
                except (OSError, ValueError) as exc:
                    result["metrics"]["observation_bundle"].update(
                        publication="failed", capture_status="partial", error=str(exc)[:400])
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
                        kernel_profile=payload.get("kernel_profile", self.configs[deployment_id].kernel_profile),
                        precision=payload.get("precision", self.configs[deployment_id].precision),
                        quantization={k: v for k, v in result.get("precision_receipt", {"method": "none"}).items()
                                      if k in {"method", "compute_dtype", "artifact_id", "quantized_module_count", "execution", "kv", "plan_sha256"}},
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
            if result.get("request_error"):
                patch.update(state="failed", error=result["request_error"]["message"],
                             error_code=result["request_error"]["code"])
            self.store.update(
                run_id, patch, "error" if result.get("request_error") else "completed" if not stopped else patch["state"]
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
            failure_metrics = self.store.get(run_id)["metrics"]
            if kind == "generate" and directory is not None:
                if cleanup != "unknown":
                    from hqsb.experiments.artifacts import seal

                    try:
                        seal(directory, run_id=run_id, status="partial", error=error)
                    except (ValueError, OSError):
                        pass  # The original failure remains authoritative; no complete bundle is published.
                bundle = {"capture_status": "partial", "publication": "unsealed_interrupted"}
                if cleanup != "unknown":
                    try:
                        from hqsb.observation.storage import publish

                        manifest = publish(directory, expected_run_id=run_id, supervisor_failure=error)
                        bundle.update(publication=manifest["publication"], counts=manifest["counts"])
                    except (OSError, ValueError) as capture_error:
                        bundle["error"] = str(capture_error)[:400]
                failure_metrics = {**failure_metrics, "observation_bundle": bundle}
            self.store.update(
                run_id,
                {
                    "state": "failed",
                    "cleanup": cleanup,
                    "error": error,
                    "metrics": failure_metrics,
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
