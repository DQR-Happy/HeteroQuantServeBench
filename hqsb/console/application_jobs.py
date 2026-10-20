"""L10 experiment admission, endpoint leases and immutable evidence publication."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import threading
import time
from hqsb.console.service import ServiceError
from hqsb.console.store import TERMINAL
from hqsb.console.application_transport import LocalEndpoint, RemoteEndpoint
from hqsb.core.application_plan import ApplicationPlan
from hqsb.core.optimization import fingerprint
from hqsb.experiments.artifacts import seal, read_manifest, read_json
from hqsb.observation.storage import atomic_json, sha256
from hqsb.serving.application_engine import ApplicationEngine, compare_variants


class ApplicationJobs:
    def __init__(self, settings, service, registry, connections):
        self.settings, self.service = settings, service
        self.registry, self.connections = registry, connections
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="l10")
        self.stop = threading.Event()
        self.quarantined = False

    def catalog(self):
        return {
            "schema_version": "l10-v1",
            "node_id": self.registry.node_id,
            "lease_owner": self.service.benchmark_owner,
            "pending": self.service.pending,
            "quarantined": self.quarantined,
            "default": ApplicationPlan(
                targets=[
                    {
                        "id": "primary",
                        "deployment_id": next(
                            iter(self.service.configs), "qwen-reference"
                        ),
                    }
                ]
            ).model_dump(),
            "routing": [
                "round_robin",
                "least_pending",
                "cache_affinity",
                "length_model",
            ],
            "limits": [
                "One target per physical node, one active request per endpoint; this is not L8 continuous batching.",
                "Result cache is exact, tenant/epoch/runtime isolated and cold for each A/B variant. It stores CPU results, not KV.",
                "Cache affinity means exact-result locality. Optional L6 prefix reuse is validated separately by the worker.",
                "Managed autoscaling loads/unloads registered model replicas on existing nodes; no VM/pod provisioning or Kubernetes HPA.",
                "Lossy prompt processing and model/budget routing need independent task-quality evaluation; output agreement alone is not quality.",
                "No automatic reroute/retry after uncertain execution. Remote cleanup/leases may require operator recovery after controller crash.",
                "No automatic Agent/tool-call rewriting, semantic answer cache, or production OpenAI-compatible streaming gateway in this endpoint.",
            ],
        }

    def start(self, plan, key):
        plan = ApplicationPlan.model_validate(plan)
        svc = self.service
        digest = fingerprint(["l10", plan.model_dump()])
        with svc.lock:
            with svc.store.lock:
                old = svc.store.db.execute(
                    "SELECT digest,run_id FROM idempotency WHERE key=?", (key,)
                ).fetchone()
            if old:
                if old[0] != digest:
                    raise ServiceError("IDEMPOTENCY_CONFLICT", "Key already used")
                return svc.store.get(old[1])
            svc._reserve()
            if self.stop.is_set() or self.quarantined:
                raise ServiceError(
                    "L10_RECOVERY_REQUIRED",
                    "服务关闭或上次节点清理未确认；请检查实验子任务与节点租约。",
                )
            if svc.pending or svc.benchmark_owner:
                raise ServiceError(
                    "DEVICE_BUSY", "请在运行记录等待当前任务或实验清理完成。"
                )
            if self.settings.privileged_worker:
                raise ServiceError(
                    "L10_PRIVILEGED_UNSUPPORTED",
                    "L10 需要普通本地 PyTorch worker。",
                    422,
                )
            profiles = {}
            for target in plan.targets:
                if target.machine_id != "local":
                    profiles[target.id] = self.registry.get(target.machine_id)
                    continue
                cfg, d = (
                    svc.configs.get(target.deployment_id),
                    svc.states.get(target.deployment_id),
                )
                if cfg is None or cfg.provider != "pytorch" or not cfg.manifest:
                    raise ServiceError(
                        "L10_REGISTER_LOCAL_MODEL",
                        "请先在模型页登记原始制品与 manifest。",
                        422,
                    )
                if target.lifecycle == "existing" and d["state"] != "ready":
                    raise ServiceError(
                        "L10_LOAD_MODEL_FIRST",
                        "已有副本模式要求先在设备与部署页加载所选模型。",
                    )
                if target.lifecycle == "managed" and d["state"] not in {
                    "unloaded",
                    "failed",
                }:
                    raise ServiceError(
                        "L10_MANAGED_REQUIRES_COLD_TARGET",
                        "托管副本必须从未加载状态开始，不能接管用户驻留模型。",
                    )
            self.settings.data_dir.mkdir(parents=True, exist_ok=True)
            used = sum(
                p.stat().st_size
                for p in (self.settings.data_dir / "captures").rglob("*")
                if p.is_file()
            )
            if (
                shutil.disk_usage(self.settings.data_dir).free < 512 * 1024**2
                or used + 64 * 1024**2 > self.settings.artifact_budget_bytes
            ):
                raise ServiceError(
                    "L10_ARTIFACT_BUDGET", "请在机器与存储页检查磁盘和观测配额。"
                )
            first = plan.targets[0]
            cfg = (
                svc.configs.get(first.deployment_id)
                if first.machine_id == "local"
                else None
            )
            deployment = (
                cfg.public()
                if cfg
                else {
                    "id": first.deployment_id,
                    "name": first.machine_id + " / " + first.deployment_id,
                }
            )
            run, _ = svc.store.create(
                "application",
                {"deployment": deployment, "application_plan": plan.model_dump()},
                key,
                digest,
            )
            owner = "l10:" + run["id"]
            try:
                local = next((t for t in plan.targets if t.machine_id == "local"), None)
                if local:
                    svc.claim_benchmark(owner, local.deployment_id)
                else:
                    svc.benchmark_owner = owner
                self.executor.submit(self.execute, run["id"], plan, profiles)
            except Exception as exc:
                svc.release_benchmark(owner)
                svc.store.update(
                    run["id"],
                    {"state": "failed", "cleanup": "not_started", "error": str(exc)},
                    "error",
                )
                raise
            return run

    def execute(self, identity, plan, profiles):
        svc, owner = self.service, "l10:" + identity
        directory = self.settings.data_dir / "captures" / identity
        endpoints, acquired, variants, engine = {}, [], [], None
        final, cleanup, error = "failed", "pending", None
        started = time.monotonic()
        trace, trace_lock = [], threading.RLock()

        def cancelled():
            return (
                self.stop.is_set()
                or svc.store.get(identity)["state"] == "cancel_requested"
                or time.monotonic() - started > plan.timeout_s
            )

        def record(kind, **event):
            with trace_lock:
                event = dict(event)
                if "offset_ms" in event:
                    event["engine_offset_ms"] = event.pop("offset_ms")
                row = {
                    "kind": kind,
                    "offset_ms": (time.monotonic() - started) * 1000,
                    **event,
                }
                trace.append(row)
                svc.store.update(
                    identity,
                    {"progress": {"stage": kind, **event}},
                    "application.progress",
                    row,
                )

        try:
            directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            with svc.lock:
                if (
                    svc.store.get(identity)["state"] == "cancel_requested"
                    or self.stop.is_set()
                ):
                    final = "cancelled"
                    return
                svc.store.update(identity, {"state": "running"}, "started")
            for target in plan.targets:
                endpoint = (
                    LocalEndpoint(svc, target, owner, record)
                    if target.machine_id == "local"
                    else RemoteEndpoint(
                        self.connections, profiles[target.id], target, owner, record
                    )
                )
                if any(e.node_id == endpoint.node_id for e in acquired):
                    endpoint.close()
                    raise ValueError("L10_ALIASED_TARGETS_SHARE_PHYSICAL_NODE")
                endpoints[target.id] = endpoint
                endpoint.acquire()
                acquired.append(endpoint)
            nodes = [e.node_id for e in endpoints.values()]
            if len(nodes) != len(set(nodes)):
                raise ValueError("L10_ALIASED_TARGETS_SHARE_PHYSICAL_NODE")
            initial = {}
            for t in plan.targets:
                snap = endpoints[t.id].snapshot()
                initial[t.id] = snap
                if t.lifecycle == "managed" and snap["state"] not in {
                    "unloaded",
                    "failed",
                }:
                    raise ValueError("L10_MANAGED_TARGET_ALREADY_RESIDENT")
                if t.lifecycle == "existing" and snap["state"] != "ready":
                    raise ValueError("L10_EXISTING_TARGET_NOT_READY")
            engine = ApplicationEngine(
                plan,
                endpoints,
                cancelled,
                lambda event: record(
                    event["kind"], **{k: v for k, v in event.items() if k != "kind"}
                ),
            )
            variants = engine.documents
            for name in ("baseline", "candidate"):
                if cancelled():
                    break
                record("variant.started", variant=name)
                engine.run(getattr(plan, name), name)
                svc.store.update(
                    identity,
                    {"application_completed_variants": len(variants)},
                    "application.variant",
                    {"name": name, "metrics": variants[-1]["metrics"]},
                )
            final = (
                (
                    "timed_out"
                    if time.monotonic() - started > plan.timeout_s
                    else "cancelled"
                )
                if cancelled()
                else "completed"
            )
            if final == "completed" and (
                len(variants) != 2 or any(v["metrics"]["failed"] for v in variants)
            ):
                final, error = "failed", "L10_REQUESTS_FAILED_OR_INCOMPLETE"
        except Exception as exc:
            error = str(exc)[:2200]
            if cancelled():
                final = (
                    "timed_out"
                    if time.monotonic() - started > plan.timeout_s
                    else "cancelled"
                )
        finally:
            cleanup_errors = []
            for endpoint in endpoints.values():
                if endpoint not in acquired and endpoint.quarantined:
                    cleanup_errors.append(
                        "Uncertain remote lease acquisition: " + endpoint.target.id
                    )
            if engine:
                for target in list(engine.prefix_scopes):
                    try:
                        engine.end_prefix(target)
                    except Exception as exc:
                        cleanup_errors.append(str(exc))
                for target in list(engine.owned):
                    try:
                        engine.unload(target, "job_finally")
                    except Exception as exc:
                        cleanup_errors.append(str(exc))
            for endpoint in reversed(acquired):
                try:
                    # A terminal child can precede its pending decrement.
                    end = time.monotonic() + 4
                    if isinstance(endpoint, LocalEndpoint) and not isinstance(
                        endpoint, RemoteEndpoint
                    ):
                        while svc.pending and time.monotonic() < end:
                            time.sleep(0.05)
                    endpoint.release()
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            for endpoint in endpoints.values():
                endpoint.close()
            cleanup = "failed" if cleanup_errors else "succeeded"
            if cleanup_errors:
                self.quarantined = True
                error = (error or "") + "; cleanup: " + str(cleanup_errors)[:1200]
                final = "failed"
            try:
                from hqsb.observation.identity import environment

                doc = {
                    "schema_version": "l10-v1",
                    "run_id": identity,
                    "status": "applied" if final == "completed" else final,
                    "cleanup": cleanup,
                    "error": error,
                    "plan": plan.model_dump(),
                    "variants": variants,
                    "events": trace,
                    "job_total_ms": (time.monotonic() - started) * 1000,
                    "source_digest": environment(Path.cwd())["source_digest"],
                    "comparison": compare_variants(*variants)
                    if len(variants) == 2
                    else None,
                    "timing_scope": "coordinator monotonic E2E incl admission, preprocessing, network and backend polling; backend runtime metrics separate; served Token/s counts cached delivery, not GPU compute throughput",
                    "limitations": self.catalog()["limits"],
                }
                atomic_json(directory / "application-receipt.json", doc)
                atomic_json(
                    directory / "execution-receipt.json",
                    {
                        "run_id": identity,
                        "status": doc["status"],
                        "cleanup_status": cleanup,
                        "source_digest": doc["source_digest"],
                        "application_receipt_sha256": sha256(
                            directory / "application-receipt.json"
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
                            "application_receipt": doc,
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
                if not cleanup_errors:
                    svc.release_benchmark(owner)

    def read(self, identity):
        run = self.service.store.get(identity)
        if run["kind"] != "application" or run["state"] not in TERMINAL:
            raise ServiceError(
                "L10_TERMINAL_REQUIRED", "请等待服务请求实验结束和节点清理确认。"
            )
        root = self.settings.data_dir / "captures" / identity
        manifest, doc = (
            read_manifest(root),
            read_json(root / "application-receipt.json"),
        )
        if (
            doc["run_id"] != identity
            or doc != run["metrics"].get("application_receipt")
            or "application-receipt.json"
            not in {a["path"] for a in manifest["artifacts"]}
        ):
            raise ServiceError("L10_INTEGRITY", "Application receipt identity mismatch")
        return doc, manifest

    def close(self):
        self.stop.set()
        self.executor.shutdown(wait=True)
