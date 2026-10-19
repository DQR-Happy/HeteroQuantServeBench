"""Two-node inference DP over registered HQSB HTTP/SSH transports.

Each participant owns a normal L9 single-device job and its own GPU lease.
The coordinator owns their IDs, cancellation, sealed inputs and output merge.
"""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
import uuid
import httpx
from hqsb.core.parallel_execution import ParallelPlan
from hqsb.core.optimization import fingerprint
from hqsb.console.service import ServiceError
from hqsb.console.store import TERMINAL
from hqsb.observation.storage import atomic_json, sha256
from hqsb.observation.identity import environment
from hqsb.experiments.artifacts import seal
from pathlib import Path


class PeerJobs:
    def __init__(self, settings, service, jobs, registry, connections):
        self.settings, self.service, self.jobs, self.registry, self.connections = (
            settings,
            service,
            jobs,
            registry,
            connections,
        )
        self.stop = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="l9-peer")
        self.active = None

    def start(self, deployment_id, machine_id, remote_deployment, plan, key):
        p = ParallelPlan.model_validate(plan)
        if p.strategy != "single" or len(p.prompts) < 2:
            raise ServiceError(
                "L9_PEER_DP_REQUIRES_SINGLE_DEVICE_PER_NODE",
                "双节点 DP 每节点使用 single 计划，至少两条请求。",
                422,
            )
        profile = self.registry.get(machine_id)
        digest = fingerprint(
            ["l9-peers", deployment_id, machine_id, remote_deployment, p.model_dump()]
        )
        svc = self.service
        with svc.lock:
            with svc.store.lock:
                old = svc.store.db.execute(
                    "SELECT digest,run_id FROM idempotency WHERE key=?", (key,)
                ).fetchone()
            if old:
                if old[0] != digest:
                    raise ServiceError("IDEMPOTENCY_CONFLICT", "Key already used")
                return svc.store.get(old[1])
            if self.active or self.stop.is_set():
                raise ServiceError(
                    "L9_PEER_COORDINATOR_BUSY", "请等待当前跨节点任务完成。"
                )
            if deployment_id not in svc.configs:
                raise ServiceError("NOT_FOUND", "Unknown deployment", 404)
            if svc.pending or svc.benchmark_owner:
                raise ServiceError("DEVICE_BUSY", "请等待节点任务结束。")
            run, _ = svc.store.create(
                "distributed",
                {
                    "deployment": svc.configs[deployment_id].public(),
                    "parallel_plan": {**p.model_dump(), "strategy": "dp_hosts"},
                    "peer": {
                        "machine_id": machine_id,
                        "deployment_id": remote_deployment,
                    },
                },
                key,
                digest,
            )
            self.active = run["id"]
            self.executor.submit(
                self.execute, run["id"], deployment_id, remote_deployment, p, profile
            )
            return run

    def execute(self, identity, deployment_id, remote_deployment, plan, profile):
        svc = self.service
        owned = {}
        client = None
        final = "failed"
        cleanup = "unknown"
        error = None
        started = time.monotonic()
        docs = []
        remote_attempted = False
        directory = self.settings.data_dir / "captures" / identity
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
            origin = self.connections.origin(profile)
            client = httpx.Client(
                base_url=origin + "/api/console/v1",
                headers=self.connections.headers(profile),
                timeout=8,
                trust_env=False,
                follow_redirects=False,
            )

            def call(path, body=None, key=None):
                response = (
                    client.get(path)
                    if body is None
                    else client.post(
                        path,
                        json=body,
                        headers={"Idempotency-Key": key or uuid.uuid4().hex},
                        timeout=40,
                    )
                )
                if not response.is_success:
                    raise ValueError(
                        f"L9_PEER_HTTP_{response.status_code}: " + response.text[:1400]
                    )
                return response.json()

            # New protocol required explicitly; an old console is never guessed compatible.
            if call("/parallel/catalog")["schema_version"] != "l9-v1":
                raise ValueError("L9_PEER_PROTOCOL_MISMATCH")
            if call("/session").get("node_id") == self.registry.node_id:
                raise ValueError("L9_PEER_MUST_BE_A_DISTINCT_NODE")
            remote_key = "l9-peer-" + identity
            remote_plan = {**plan.model_dump(), "prompts": plan.prompts[1::2]}
            local_plan = {**plan.model_dump(), "prompts": plan.prompts[::2]}
            # Save the remote key before POST; uncertain outcomes are recovered by
            # retrying this same key, never by launching a replacement job.
            body = {"deployment_id": remote_deployment, "plan": remote_plan}
            svc.store.update(
                identity, {"remote_submission_key": remote_key}, "remote.submitting"
            )
            remote_attempted = True
            try:
                remote = call("/parallel/jobs", body, remote_key)
            except httpx.TransportError:
                remote = call("/parallel/jobs", body, remote_key)
            owned["remote"] = remote["id"]
            local = self.jobs.start(deployment_id, local_plan, "l9-local-" + identity)
            owned["local"] = local["id"]
            svc.store.update(identity, {"participants": owned}, "participants", owned)
            while True:
                local = svc.store.get(owned["local"])
                remote = call("/runs/" + owned["remote"])
                if (
                    self.stop.is_set()
                    or svc.store.get(identity)["state"] == "cancel_requested"
                ):
                    final = "cancelled"
                    break
                if time.monotonic() - started > plan.timeout_s + 20:
                    final = "timed_out"
                    break
                if local["state"] in TERMINAL and remote["state"] in TERMINAL:
                    if any(
                        r["state"] != "completed" or r["cleanup"] != "succeeded"
                        for r in [local, remote]
                    ):
                        raise ValueError(
                            "L9_PEER_PARTICIPANT_FAILED: "
                            + str(
                                [
                                    (r["id"], r["state"], r.get("error"))
                                    for r in [local, remote]
                                ]
                            )
                        )
                    a, _ = self.jobs.read(owned["local"])
                    b = call("/runs/" + owned["remote"] + "/parallel")
                    if (
                        a["model_identity"] != b["model_identity"]
                        or a["tokenizer_identity"] != b["tokenizer_identity"]
                    ):
                        raise ValueError("L9_PEER_MODEL_OR_TOKENIZER_IDENTITY_MISMATCH")
                    docs = [a, b]
                    final = "completed"
                    break
                self.stop.wait(0.3)
        except Exception as exc:
            error = str(exc)[:1400]
        finally:
            try:
                if owned:
                    if "local" in owned:
                        svc.cancel(owned["local"])
                    if "remote" in owned:
                        call("/requests/" + owned["remote"] + "/cancel", {})
                    end = time.monotonic() + 25
                    while True:
                        states = (
                            [svc.store.get(owned["local"])] if "local" in owned else []
                        ) + (
                            [call("/runs/" + owned["remote"])]
                            if "remote" in owned
                            else []
                        )
                        if all(
                            r["state"] in TERMINAL and r["cleanup"] == "succeeded"
                            for r in states
                        ):
                            cleanup = "succeeded"
                            break
                        if time.monotonic() > end:
                            break
                        self.stop.wait(0.3)
                elif not remote_attempted:
                    cleanup = "not_started"
            except Exception as exc:
                error = (
                    (error or "")
                    + "; participant cleanup unconfirmed: "
                    + str(exc)[:300]
                )
            if client:
                client.close()
            if cleanup not in {"succeeded", "not_started"} and final == "completed":
                final = "failed"
            try:
                requests = []
                ranks = []
                for node, doc in enumerate(docs):
                    for request in doc["requests"]:
                        requests.append(
                            {
                                **request,
                                "index": request["index"] * 2 + node,
                                "node": node,
                            }
                        )
                    for rank in doc["ranks"]:
                        ranks.append({**rank, "rank": len(ranks), "node": node})
                requests.sort(key=lambda r: r["index"])
                elapsed = (time.monotonic() - started) * 1000
                count = sum(len(r["token_ids"]) for r in requests)
                result = {
                    "schema_version": "l9-v1",
                    "run_id": identity,
                    "status": "applied" if final == "completed" else final,
                    "cleanup": cleanup,
                    "plan": {**plan.model_dump(), "strategy": "dp_hosts"},
                    "ranks": ranks,
                    "requests": requests,
                    "participants": owned,
                    "participant_receipts": docs,
                    "source_digest": environment(Path.cwd())["source_digest"],
                    "model_identity": docs[0]["model_identity"] if docs else None,
                    "tokenizer_identity": docs[0]["tokenizer_identity"]
                    if docs
                    else None,
                    "workload_ms": elapsed,
                    "job_total_ms": elapsed,
                    "output_tokens": count,
                    "output_tokens_per_s": count * 1000 / elapsed if count else None,
                    "timing_scope": "coordinator monotonic wall time, includes both node cold loads, HTTP submission/poll, execution and cleanup acknowledgement",
                    "limitations": [
                        "HTTP request/result DP only; no weight or KV transfer between nodes.",
                        "Each participant owns its actual runtime versions; heterogeneous outputs require explicit per-input baseline comparison.",
                        "Participant job receipts are copied and sealed locally after remote integrity checks.",
                    ],
                }
                atomic_json(directory / "parallel-receipt.json", result)
                atomic_json(
                    directory / "execution-receipt.json",
                    {
                        "run_id": identity,
                        "status": result["status"],
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
                            "parallel_receipt": result,
                            "output_tokens_per_s": result["output_tokens_per_s"],
                            "output_tokens": count,
                            "experiment_bundle": manifest,
                        },
                    },
                    final,
                )
            except Exception as exc:
                svc.store.update(
                    identity,
                    {
                        "state": "failed",
                        "cleanup": cleanup,
                        "error": "L9_PEER_PUBLICATION_FAILED: " + str(exc)[:600],
                    },
                    "error",
                )
            finally:
                with svc.lock:
                    self.active = None

    def close(self):
        self.stop.set()
        self.executor.shutdown(wait=True)
