"""Exclusive resident-model experiments, durable summaries and bounded reads."""
import hashlib
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

from hqsb.console.captures import safe_directory
from hqsb.console.service import ServiceError
from hqsb.console.store import TERMINAL
from hqsb.core.contracts.scheduling import SchedulingExperiment, WorkItem
from hqsb.observation.storage import EvidenceReader, publish


class SchedulingCampaigns:
    def __init__(self, settings, service):
        self.settings, self.service = settings, service
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hqsb-l8")
        self.lock = threading.RLock()
        self.active = None
        self.closed = False

    def _capability_result(self, deployment_id, future):
        try:
            result = future.result()
            if result.get("worker_terminated"):
                raise RuntimeError("Capability worker terminated")
            return result
        except Exception as exc:
            self.service.actors[deployment_id].close()
            with self.service.lock:
                self.service.states[deployment_id].update(state="failed", epoch=None, detail={}, last_error=str(exc)[:800])
            raise ServiceError("WORKER_FAILED", "Capability inspection lost the model worker; reload the deployment", 502) from exc

    def capabilities(self, deployment_id):
        svc = self.service
        with svc.lock:
            if deployment_id not in svc.configs:
                raise ServiceError("NOT_FOUND", "Unknown deployment", 404)
            if self.settings.privileged_worker:
                return {"supported": False, "reasons": ["PRIVILEGED_SESSION_PROTOCOL_NOT_IMPLEMENTED"]}
            if svc.configs[deployment_id].provider != "pytorch":
                return {"supported": False, "reasons": ["LOCAL_PYTORCH_REQUIRED"]}
            if svc.states[deployment_id]["state"] != "ready":
                return {"supported": False, "reasons": ["LOAD_MODEL_TO_INSPECT_CACHE_LAYOUT"]}
            if svc.pending or svc.benchmark_owner:
                raise ServiceError("DEVICE_BUSY", "Wait for the running task before inspecting scheduling capabilities")
            svc.pending += 1
            future = svc.executors[deployment_id].submit(
                svc.actors[deployment_id].execute, "scheduling_capabilities", {}, lambda _: None,
                lambda: svc.closed, 15)
        try:
            return self._capability_result(deployment_id, future)
        finally:
            with svc.lock:
                svc.pending -= 1

    def start(self, body, key):
        spec = SchedulingExperiment.model_validate(body)
        svc = self.service
        fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        with self.lock, svc.lock:
            with svc.store.lock:
                old = svc.store.db.execute("SELECT digest,run_id FROM idempotency WHERE key=?", (key,)).fetchone()
            if old:
                if old[0] != fingerprint:
                    raise ServiceError("IDEMPOTENCY_CONFLICT", "Key already used for another payload")
                return self.get(old[1])
            if self.closed:
                raise ServiceError("SHUTTING_DOWN", "Scheduler is closing", 503)
            if self.settings.privileged_worker:
                raise ServiceError("SCHEDULING_UNSUPPORTED", "Privileged worker session protocol is not implemented", 422)
            cfg, state = svc.configs.get(spec.deployment_id), svc.states.get(spec.deployment_id)
            if cfg is None:
                raise ServiceError("NOT_FOUND", "Unknown deployment", 404)
            if cfg.provider != "pytorch":
                raise ServiceError("SCHEDULING_UNSUPPORTED", "Local PyTorch provider required", 422)
            if state["state"] != "ready" or state["epoch"] != spec.expected_epoch:
                raise ServiceError("STALE_EPOCH", "Load the model and refresh the deployment epoch")
            # Resolve capability on the resident worker before accepting a run.
            # Unsupported layouts are ordinary rejections, not worker failures.
            if svc.pending or svc.benchmark_owner:
                raise ServiceError("DEVICE_BUSY", "Wait for the current task")
            future = svc.executors[spec.deployment_id].submit(
                svc.actors[spec.deployment_id].execute, "scheduling_capabilities", {}, lambda _: None,
                lambda: svc.closed, 15)
            capability = self._capability_result(spec.deployment_id, future)
            if not capability.get("supported"):
                raise ServiceError("SCHEDULING_UNSUPPORTED", ",".join(capability.get("reasons", [])), 422)
            svc._preflight("generate", cfg, {"observation_mode": "basic"})
            svc.claim_benchmark("l8-reserving", spec.deployment_id)
            run, pending_added = None, False
            try:
                stored = spec.model_dump()
                if not spec.save_input:
                    stored["requests"] = [{**item.model_dump(exclude={"prompt"}),
                        "prompt_sha256": hashlib.sha256(item.prompt.encode()).hexdigest()} for item in spec.requests]
                run, _ = svc.store.create("scheduling", {**stored, "deployment": cfg.public(),
                    "workload_sha256": hashlib.sha256(json.dumps([x.model_dump() for x in spec.requests], sort_keys=True).encode()).hexdigest(),
                    "load_run_id": state.get("load_run_id")}, key, fingerprint)
                svc.benchmark_owner = self.active = run["id"]
                svc.pending += 1
                pending_added = True
                self.executor.submit(self._execute, run["id"], spec.model_dump())
            except Exception:
                svc.release_benchmark(run["id"] if run else "l8-reserving")
                if pending_added:
                    svc.pending -= 1
                self.active = None
                if run:
                    svc.store.update(run["id"], {"state": "failed", "cleanup": "not_started",
                        "error": "SCHEDULING_SUBMISSION_FAILED"})
                raise
            return self.get(run["id"])

    def _execute(self, identity, experiment):
        svc, deployment = self.service, experiment["deployment_id"]
        actor = svc.actors[deployment]
        directory = safe_directory(self.settings.data_dir / "captures", identity)
        try:
            directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            svc.store.update(identity, {"state": "running"})
            def progress(event):
                svc.store.update(identity, {"progress": event}, "scheduling.progress", event)
            result = actor.execute("scheduling", {"capture_id": identity, "capture_dir": str(directory),
                "experiment": experiment}, progress,
                lambda: self.closed or svc.closed or svc.store.get(identity)["state"] == "cancel_requested",
                experiment["session_deadline_s"] * len(experiment["modes"]) * experiment["repeats"] + 60)
            if result.get("worker_terminated"):
                raise RuntimeError("SCHEDULING_WORKER_TERMINATED")
            evidence = publish(directory, expected_run_id=identity)
            # Seal ingress and reconcile controls under the same lock used by
            # submit_request. No accepted late message can escape this check.
            with self.lock:
                valid = evidence["capture_status"] == "complete" and result["cleanup"] == "succeeded"
                external = svc.store.get(identity)["config"].get("external_requests", [])
                observed = {s["id"] for v in result["variants"] for s in v["requests"]}
                result["unacknowledged_ingress"] = [r["id"] for r in external if r["id"] not in observed]
                valid = valid and not result["unacknowledged_ingress"]
                reason = result["finish_reason"]
                state = reason if reason in {"cancelled", "timed_out"} else "completed" if valid else "failed"
                svc.store.update(identity, {"state": state, "cleanup": result["cleanup"],
                    "result": result, "quality": result["quality"],
                    "metrics": {"scheduling_evidence": {"capture_status": evidence["capture_status"],
                        "publication": evidence["publication"], "counts": evidence["counts"]}},
                    "error": None if valid else "INCOMPLETE_SCHEDULING_EVIDENCE_OR_INGRESS"})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:1000]
            cleanup = "worker_terminated"
            try:
                actor.close()
            except Exception as stop_error:
                cleanup = "unknown"
                error += f"; cleanup: {stop_error}"
            with svc.lock:
                svc.states[deployment].update(state="failed", epoch=None, detail={}, last_error=error)
            svc.store.update(identity, {"state": "failed", "cleanup": cleanup, "error": error})
        finally:
            with self.lock, svc.lock:
                svc.pending -= 1
                svc.release_benchmark(identity)
                self.active = None

    def get(self, identity):
        try:
            row = self.service.store.get(identity)
        except KeyError as exc:
            raise ServiceError("NOT_FOUND", "Unknown scheduling experiment", 404) from exc
        if row["kind"] != "scheduling":
            raise ServiceError("NOT_FOUND", "Not a scheduling experiment", 404)
        return row

    def list(self):
        svc = self.service
        with svc.store.lock:
            rows = svc.store.db.execute("SELECT json_remove(body,'$.result','$.config') FROM runs WHERE json_extract(body,'$.kind')='scheduling' ORDER BY created DESC LIMIT 30").fetchall()
        return {"items": [json.loads(r[0]) for r in rows],
                "active_id": self.active}

    def cancel(self, identity):
        self.get(identity)
        return self.service.cancel(identity)

    def cancel_request(self, identity, request_id):
        with self.lock:
            row = self.get(identity)
            if self.active != identity or row["state"] in TERMINAL:
                raise ServiceError("SESSION_NOT_RUNNING", "The scheduling experiment is no longer running")
            if request_id not in {r["id"] for r in row["config"]["requests"] + row["config"].get("external_requests", [])}:
                raise ServiceError("NOT_FOUND", "Unknown workload request", 404)
            try:
                self.service.actors[row["config"]["deployment_id"]].control(
                    {"kind": "cancel_request", "run_id": identity, "request_id": request_id})
            except queue.Full as exc:
                raise ServiceError("CONTROL_QUEUE_FULL", "Cancellation queue is full", 429) from exc
            return {"state": "queued", "request_id": request_id,
                    "note": "Worker acknowledgment is recorded in request.terminal or control.refused evidence"}

    def submit_request(self, identity, item):
        """Live ingress is explicit and cannot silently mutate a paired workload."""
        item = WorkItem.model_validate(item)
        with self.lock:
            row = self.get(identity)
            if self.active != identity or row["state"] != "running":
                raise ServiceError("SESSION_NOT_RUNNING", "The session is not accepting controls")
            config = row["config"]
            if len(config["modes"]) != 1 or config["repeats"] != 1:
                raise ServiceError("FROZEN_COMPARISON", "Live ingress requires one mode and one repeat; paired experiments use the frozen sender", 422)
            identities = {r["id"] for r in config["requests"]}
            external = config.get("external_requests", [])
            if item.id in identities or item.id in {r["id"] for r in external}:
                raise ServiceError("DUPLICATE_ID", "Request ID already exists")
            if len(identities) + len(external) >= 128:
                raise ServiceError("SESSION_REQUEST_LIMIT", "At most 128 requests per session", 429)
            try:
                self.service.actors[config["deployment_id"]].control(
                    {"kind": "submit_request", "run_id": identity, "item": item.model_dump()})
            except queue.Full as exc:
                raise ServiceError("CONTROL_QUEUE_FULL", "Ingress queue is full", 429) from exc
            stored = item.model_dump()
            if not config["save_input"]:
                stored.pop("prompt")
                stored["prompt_sha256"] = hashlib.sha256(item.prompt.encode()).hexdigest()
            external.append(stored)
            self.service.store.update(identity, {"config": {**config, "external_requests": external}}, "scheduling.ingress_queued", {"request_id": item.id})
            return {"state": "queued", "request_id": item.id,
                    "note": "Acceptance requires a request.accepted worker record; controls arriving after the last step may remain unacknowledged"}

    def replay(self, identity, key):
        old = self.get(identity)["config"]
        if not old.get("save_input") or old.get("external_requests"):
            raise ServiceError("WORKLOAD_NOT_REPLAYABLE", "Replay requires saved frozen inputs and no live mutations", 422)
        body = {key: value for key, value in old.items() if key in SchedulingExperiment.model_fields}
        state = self.service.states[body["deployment_id"]]
        body["expected_epoch"] = state["epoch"]
        return self.start(body, key)

    def records(self, identity, kind, after, limit):
        row = self.get(identity)
        if row["state"] not in TERMINAL:
            raise ServiceError("EVIDENCE_PENDING", "Evidence is published after session cleanup")
        reader = EvidenceReader(safe_directory(self.settings.data_dir / "captures", identity))
        manifest = reader.manifest()
        if sum(a["bytes"] for a in manifest["artifacts"]) > 128 * 1024**2:
            raise ServiceError("EVIDENCE_BUDGET", "Scheduling evidence exceeds read budget", 422)
        out, cursor = [], after
        # Exhaust the stream to verify hashes, sequence and final counts even for
        # a small requested page. Memory use stays bounded by the page limit.
        for record in reader.records():
            if len(out) < limit and record["seq"] > after and (kind is None or record["kind"] == kind):
                out.append(record)
                cursor = record["seq"]
        return {"items": out, "next_after": cursor, "capture_status": manifest["capture_status"]}

    def close(self):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=True)
