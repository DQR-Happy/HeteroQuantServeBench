"""L10 endpoint adapters: existing console device owner or registered HTTP/SSH.

Only this boundary knows console lifecycles and transport. Engine never owns
actors, credentials or arbitrary URLs. Every accepted operation has one stable ID.
"""

import json
import time
import httpx
from hqsb.console.service import ServiceError
from hqsb.console.store import TERMINAL
from hqsb.core.optimization import fingerprint


class LocalEndpoint:
    def __init__(self, service, target, owner, record=lambda *a, **k: None):
        self.service, self.target, self.owner, self.record = (
            service,
            target,
            owner,
            record,
        )
        self.node_id = "local"
        self.epoch = None
        self.load_id = None
        self.quarantined = False

    def acquire(self):
        svc = self.service
        with svc.lock:
            if svc.benchmark_owner == self.owner:
                return
            svc.claim_benchmark(self.owner, self.target.deployment_id)

    def check(self):
        if self.service.benchmark_owner != self.owner:
            raise ServiceError(
                "L10_LEASE_REQUIRED", "当前服务请求实验不拥有该节点的执行租约。"
            )

    def release(self):
        with self.service.lock:
            self.check()
            if self.service.pending or self.quarantined:
                raise ServiceError(
                    "L10_CLEANUP_UNCONFIRMED",
                    "请先在运行记录确认子任务清理；不能释放未知状态的节点。",
                )
            self.service.release_benchmark(self.owner)

    def worker(self, operation, payload=None):
        svc, identity = self.service, self.target.deployment_id
        with svc.lock:
            self.check()
            d = svc.states[identity]
            if d["state"] != "ready":
                raise ServiceError("L10_TARGET_NOT_READY", "请在设备与部署页加载模型。")
            svc.pending += 1
            future = svc.executors[identity].submit(
                svc.actors[identity].execute,
                operation,
                payload or {},
                lambda _: None,
                lambda: svc.closed,
                20,
            )
        try:
            result = future.result()
            if result.get("worker_terminated"):
                self.quarantined = True
                raise RuntimeError("L10_WORKER_METADATA_OPERATION_TERMINATED")
            return result
        finally:
            with svc.lock:
                svc.pending -= 1

    def state(self):
        with self.service.lock:
            self.check()
            identity = self.target.deployment_id
            if identity not in self.service.configs:
                raise ServiceError(
                    "L10_DEPLOYMENT_NOT_FOUND", "请在对应节点的模型页登记源模型。", 404
                )
            cfg = self.service.configs[identity]
            if cfg.provider != "pytorch" or not cfg.manifest:
                raise ServiceError(
                    "L10_VERIFIED_PYTORCH_REQUIRED",
                    "L10 内部观测需要已登记 manifest 的 PyTorch 模型。",
                    422,
                )
            return json.loads(
                json.dumps(
                    {
                        k: self.service.states[identity].get(k)
                        for k in (
                            "state",
                            "epoch",
                            "load_run_id",
                            "precision",
                            "quantization",
                            "kernel_profile",
                            "structure_artifact_id",
                        )
                    }
                )
            )

    def snapshot(self):
        d = self.state()
        result = {
            "state": d["state"],
            "epoch": d.get("epoch"),
            "load_run_id": d.get("load_run_id"),
            "deployment_id": self.target.deployment_id,
            "node_id": self.node_id,
        }
        if d["state"] == "ready":
            meta = self.worker("application_metadata")
            result.update(
                identity={
                    "weights": meta["model_identity"],
                    "execution": meta.get("execution_identity"),
                    "tokenizer": meta["tokenizer_identity"],
                    "precision": d["precision"],
                    "quantization": d.get("quantization"),
                    "kernel": d.get("kernel_profile"),
                    "structure": d.get("structure_artifact_id"),
                    "worker_source_digest": meta["source_digest"],
                },
                prefix_supported=meta["prefix_capability"]["prefix_supported"],
                prefix_reason=meta["prefix_capability"].get("prefix_reason"),
            )
        self.epoch, self.load_id = result["epoch"], result.get("load_run_id")
        return result

    def prepare(self, item, policy):
        return self.worker("application_prepare", {"input": item, "policy": policy})

    def clear_prefix(self):
        return self.worker("memory_cache_clear")

    def begin_prefix(self):
        return self.worker("application_prefix_begin", {"owner": self.owner})

    def end_prefix(self):
        return self.worker("application_prefix_end", {"owner": self.owner})

    def operation(self, kind, payload, key):
        self.check()
        self.record(
            "backend.submitting", target=self.target.id, operation=kind, key=key
        )
        stable_key = self.owner + ":" + key
        # A lost HTTP response can arrive after load/unload changed state.
        # Delegate existing keys to the service's payload conflict check before
        # fresh-admission guards, preserving the original operation identity.
        with self.service.store.lock:
            previous = self.service.store.db.execute(
                "SELECT run_id FROM idempotency WHERE key=?", (stable_key,)
            ).fetchone()
        if previous:
            return self.service.submit(
                kind,
                self.target.deployment_id,
                payload,
                stable_key,
                benchmark_owner=self.owner,
            )
        if kind == "load":
            if (
                payload
                or self.target.lifecycle != "managed"
                or self.state()["state"] not in {"unloaded", "failed"}
            ):
                raise ServiceError(
                    "L10_MANAGED_LOAD_ONLY",
                    "只能以默认配置加载已声明托管的冷副本。",
                    422,
                )
        if kind == "unload":
            d = self.state()
            load_id = payload.get("load_run_id")
            if (
                set(payload) != {"expected_epoch", "load_run_id"}
                or d["epoch"] != payload.get("expected_epoch")
                or d.get("load_run_id") != load_id
                or not load_id
                or self.poll(load_id)["config"].get("benchmark_id") != self.owner
            ):
                raise ServiceError(
                    "L10_REFUSE_UNLOAD_FOREIGN_MODEL",
                    "只能卸载本实验实际加载且 epoch/load ID 一致的副本。",
                )
        return self.service.submit(
            kind,
            self.target.deployment_id,
            payload,
            stable_key,
            benchmark_owner=self.owner,
        )

    def poll(self, identity):
        return self.service.store.get(identity)

    def status(self, identity):
        return self.service.store.status(identity)

    def cancel(self, identity):
        row = self.poll(identity)
        if row["config"].get("benchmark_id") != self.owner:
            raise ServiceError("L10_CHILD_OWNERSHIP", "不能取消其他服务的任务。")
        return self.service.cancel(identity)

    def await_run(self, run, key, deadline, cancelled):
        identity = run["id"]
        self.record(
            "backend.accepted", target=self.target.id, child_run_id=identity, key=key
        )
        stopping, cleanup_deadline = False, None
        while True:
            run = self.status(identity)
            if run["state"] in TERMINAL:
                # state can become terminal just before the executor's finally
                # decrements pending; lease release waits separately for idle.
                if run["cleanup"] not in {
                    "succeeded",
                    "not_started",
                    "worker_terminated",
                }:
                    self.quarantined = True
                    raise RuntimeError("L10_CHILD_CLEANUP_UNCONFIRMED: " + identity)
                if run["state"] != "completed":
                    raise RuntimeError(f"L10_CHILD_{run['state']}: {run.get('error')}")
                return self.poll(identity)
            if cancelled() or time.monotonic() > deadline:
                if not stopping:
                    self.cancel(identity)
                    stopping, cleanup_deadline = True, time.monotonic() + 25
                elif time.monotonic() > cleanup_deadline:
                    self.quarantined = True
                    raise TimeoutError("L10_CANCEL_ACK_NOT_CONFIRMED: " + identity)
            time.sleep(0.1)

    def generate(self, payload, key, deadline, cancelled):
        run = self.operation(
            "generate",
            {
                **payload,
                "deadline_ms": max(
                    1000, min(120000, int((deadline - time.monotonic()) * 1000))
                ),
            },
            key,
        )
        return self.await_run(run, key, deadline, cancelled)

    def load(self, key):
        run = self.operation("load", {}, key)
        result = self.await_run(run, key, time.monotonic() + 150, lambda: False)
        d = self.snapshot()
        if d["load_run_id"] != result["id"]:
            raise RuntimeError("L10_LOAD_OWNERSHIP_NOT_CONFIRMED")
        return {"id": result["id"], "epoch": d["epoch"]}

    def unload(self, epoch, key, load_run_id):
        d = self.state()
        if d["epoch"] != epoch or d.get("load_run_id") != load_run_id:
            self.quarantined = True
            raise RuntimeError("L10_REFUSE_UNLOAD_FOREIGN_MODEL")
        result = self.await_run(
            self.operation(
                "unload", {"expected_epoch": epoch, "load_run_id": load_run_id}, key
            ),
            key,
            time.monotonic() + 50,
            lambda: False,
        )
        if self.state()["state"] != "unloaded":
            self.quarantined = True
            raise RuntimeError("L10_UNLOAD_NOT_CONFIRMED")
        return result

    def close(self):
        pass


class RemoteEndpoint(LocalEndpoint):
    def __init__(self, connections, profile, target, owner, record):
        self.target, self.owner, self.record = target, owner, record
        self.client = httpx.Client(
            base_url=connections.origin(profile) + "/api/console/v1",
            headers=connections.headers(profile),
            timeout=8,
            trust_env=False,
            follow_redirects=False,
        )
        self.quarantined = False
        try:
            session = self.call("/session")
            if session["node_id"] == connections.registry.node_id:
                raise ValueError("L10_REMOTE_SELF_REFERENCE")
            if self.call("/application/catalog")["schema_version"] != "l10-v1":
                raise ValueError("L10_REMOTE_PROTOCOL_UNSUPPORTED")
            self.node_id = session["node_id"]
        except Exception:
            self.close()
            raise

    def call(self, path, body=None, key=None):
        response = (
            self.client.get(path)
            if body is None
            else self.client.post(
                path,
                json=body,
                headers={
                    "Idempotency-Key": key or fingerprint([self.owner, path, body])
                },
                timeout=35,
            )
        )
        if not response.is_success:
            raise RuntimeError(
                f"L10_ENDPOINT_HTTP_{response.status_code}: {response.text[:900]}"
            )
        return response.json()

    def acquire(self):
        try:
            self.call(
                "/application/leases/acquire",
                {"owner": self.owner, "target": self.target.model_dump()},
            )
        except httpx.TransportError:
            self.quarantined = True
            raise

    def release(self):
        if self.quarantined:
            raise RuntimeError("L10_REMOTE_CLEANUP_UNCONFIRMED")
        self.call(
            "/application/leases/release",
            {"owner": self.owner, "target": self.target.model_dump()},
        )

    def snapshot(self):
        result = self.call(
            "/application/worker",
            {
                "owner": self.owner,
                "target": self.target.model_dump(),
                "operation": "snapshot",
            },
        )
        self.epoch, self.load_id = result["epoch"], result.get("load_run_id")
        return result

    def worker(self, operation, payload=None):
        return self.call(
            "/application/worker",
            {
                "owner": self.owner,
                "target": self.target.model_dump(),
                "operation": operation,
                "payload": payload or {},
            },
        )

    def state(self):
        return self.snapshot()

    def operation(self, kind, payload, key):
        self.record(
            "backend.submitting", target=self.target.id, operation=kind, key=key
        )
        body = {
            "owner": self.owner,
            "target": self.target.model_dump(),
            "kind": kind,
            "payload": payload,
        }
        stable = self.owner + ":" + key
        try:
            return self.call("/application/operations", body, stable)
        except httpx.TransportError:
            # Recover an uncertain POST with the same server-side idempotency
            # key. Never route an uncertain operation to another endpoint.
            try:
                return self.call("/application/operations", body, stable)
            except httpx.TransportError:
                self.quarantined = True
                raise

    def poll(self, identity):
        return self.call("/runs/" + identity)

    def status(self, identity):
        try:
            return self.call("/application/operations/" + identity + "/status")
        except Exception:
            self.quarantined = True
            raise

    def cancel(self, identity):
        row = self.poll(identity)
        if row["config"].get("benchmark_id") != self.owner:
            raise RuntimeError("L10_REMOTE_CHILD_OWNERSHIP")
        return self.call("/requests/" + identity + "/cancel", {})

    def close(self):
        self.client.close()
