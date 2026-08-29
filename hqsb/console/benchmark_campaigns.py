"""Durable quick comparisons over the existing serial model/device owner.

Browser disconnects do not own execution. Every accepted worker task has one
recorded ID; restart interrupts a campaign, and never resubmits its old slots.
"""
import json
import re
import threading
import time
import uuid

from hqsb.console.kernel_inventory import deployment_catalogue
from hqsb.console.service import ServiceError
from hqsb.console.store import TERMINAL
from hqsb.core.optimization import fingerprint
from hqsb.experiments.benchmark_suite import project_run, suite, summarize
from hqsb.observation.storage import atomic_json

IDENTITY = re.compile(r"bench_[0-9a-f]{32}\Z")
FINAL = {"completed", "cancelled", "failed", "interrupted"}


class CampaignCancelled(Exception):
    pass


class BenchmarkCampaigns:
    def __init__(self, settings, service):
        self.settings, self.service = settings, service
        self.root = settings.data_dir / "benchmarks"
        self.lock = threading.RLock()
        self.thread = None
        self.cancelled = threading.Event()
        self.stopping = threading.Event()
        self.active_id = None
        if self.root.is_symlink():
            raise ValueError("BENCHMARK_STORE_SYMLINK")
        for path in self.root.glob("bench_*.json"):
            document = self.get(path.stem)
            if document["state"] not in FINAL:
                document.update(state="interrupted", phase="interrupted", error="服务已重启；保留原任务，不自动重提。")
                self._save(document)

    def _path(self, identity):
        if not IDENTITY.fullmatch(identity):
            raise ServiceError("BENCHMARK_ID", "Invalid benchmark ID", 422)
        path = self.root / (identity + ".json")
        if path.is_symlink() or self.root.is_symlink():
            raise ServiceError("BENCHMARK_PATH", "Invalid benchmark store")
        return path

    def get(self, identity):
        with self.lock:
            try:
                document = json.loads(self._path(identity).read_text())
            except FileNotFoundError as exc:
                raise ServiceError("BENCHMARK_NOT_FOUND", "对照任务不存在。", 404) from exc
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ServiceError("BENCHMARK_INTEGRITY", "对照记录无法解析。") from exc
            try:
                valid = (document["plan_sha256"] == fingerprint(document["plan"]) and document["id"] == identity
                         and document["records_sha256"] == fingerprint(document["records"])
                         and document["report_sha256"] == fingerprint(document["report"]))
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                raise ServiceError("BENCHMARK_INTEGRITY", "对照计划身份校验失败。")
            if identity == self.active_id and self.cancelled.is_set() and document["state"] not in FINAL:
                document["state"] = "cancelling"
            return document

    def list(self, limit=30):
        with self.lock:
            paths = sorted(self.root.glob("bench_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
            rows = [self.get(path.stem) for path in paths[:limit]]
            return {"items": [{key: row[key] for key in ("id", "state", "phase", "created_at", "updated_at", "plan_sha256", "error")} | {
                "scheme_count": len(row["plan"]["schemes"]), "case_count": len(row["plan"]["cases"])} for row in rows], "total": len(paths)}

    def catalogue(self, deployment_id):
        if deployment_id not in self.service.configs:
            raise ServiceError("NOT_FOUND", "部署不存在。", 404)
        config = self.service.configs[deployment_id]
        profiles = deployment_catalogue(config)
        schemes = [{"id": row["profile_id"], "name": row["label"], "kernel_profile": row["profile_id"],
                    "cleanup_policy": "empty_cache", "availability": row["availability"], "operators": row["operators"]} for row in profiles["profiles"]]
        schemes.append({"id": "native-preserve-allocator", "name": "框架原生 · 保留 allocator 缓存", "kernel_profile": "framework-native",
                        "cleanup_policy": "preserve_allocator", "operators": {}, "availability": schemes[0]["availability"]})
        return {**suite(), "schemes": schemes, "native_artifact": profiles["native_artifact"],
                "defaults": {"baseline_id": "framework-native", "candidate_ids": ["qwen3-norm-vector-v2", "qwen3-local-fusions-v1"],
                             "case_ids": [case["id"] for case in suite()["cases"]], "repeats": 1, "output_tokens": 64, "diagnostics": False},
                "active_id": self.active_id}

    def _save(self, document):
        with self.lock:
            document["updated_at"] = time.time()
            document["report"] = summarize(document["plan"], document["records"])
            document["records_sha256"] = fingerprint(document["records"])
            document["report_sha256"] = fingerprint(document["report"])
            path = self._path(document["id"])
            partial = path.with_name(path.name + ".partial")
            if partial.exists() or partial.is_symlink():
                # A crash may leave the exclusive-create temporary. Preserve it
                # for diagnosis before publishing the authoritative next journal.
                partial.replace(path.with_name(path.name + "." + uuid.uuid4().hex + ".interrupted"))
            atomic_json(path, document)

    def start(self, request, key):
        with self.lock:
            key_sha, body_sha = fingerprint(key), fingerprint(request)
            for path in self.root.glob("bench_*.json"):
                old = self.get(path.stem)
                if old["request_key_sha256"] == key_sha:
                    if old["request_sha256"] != body_sha:
                        raise ServiceError("IDEMPOTENCY_CONFLICT", "此提交键已用于另一个对照计划。")
                    return old
            if self.active_id:
                raise ServiceError("BENCHMARK_BUSY", "已有性能对照正在运行。")
            catalog = self.catalogue(request["deployment_id"])
            case_map = {case["id"]: case for case in catalog["cases"]}
            scheme_map = {scheme["id"]: scheme for scheme in catalog["schemes"]}
            selected = [request["baseline_id"], *request["candidate_ids"]]
            if len(set(selected)) != len(selected) or len(set(request["case_ids"])) != len(request["case_ids"]):
                raise ServiceError("BENCHMARK_DUPLICATE", "基线、候选和输入用例不能重复。", 422)
            if any(identity not in scheme_map or not scheme_map[identity]["availability"]["loadable"] for identity in selected):
                raise ServiceError("BENCHMARK_SCHEME", "所选方案不可用，请核对制品配置。", 422)
            if any(identity not in case_map for identity in request["case_ids"]):
                raise ServiceError("BENCHMARK_CASE", "固定用例不存在。", 422)
            config = self.service.configs[request["deployment_id"]]
            if request["output_tokens"] > config.max_output_tokens:
                raise ServiceError("TOKEN_LIMIT", "输出预算超过部署上限。", 422)
            identity = "bench_" + uuid.uuid4().hex
            self.service.claim_benchmark(identity, config.id)
            try:
                initial = next(row for row in self.service.deployments() if row["id"] == config.id)
                if initial["state"] not in {"ready", "unloaded", "failed"}:
                    raise ServiceError("DEPLOYMENT_BUSY", "模型生命周期任务尚未完成。")
                plan = {**request, "suite_version": catalog["version"], "suite_sha256": catalog["suite_sha256"],
                        "cases": [case_map[name] for name in request["case_ids"]], "schemes": [scheme_map[name] for name in selected],
                        "warmups_per_case": 1, "native_artifact": catalog["native_artifact"], "deployment": config.public()}
                document = {"id": identity, "state": "running", "phase": "starting", "created_at": time.time(),
                            "plan": plan, "plan_sha256": fingerprint(plan), "records": [], "lifecycle": [],
                            "current_run_id": None, "error": None, "restore_error": None,
                            "initial": {key: initial.get(key) for key in ("state", "kernel_profile", "epoch")},
                            "request_key_sha256": key_sha, "request_sha256": body_sha}
                self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
                self._save(document)
                self.active_id = identity
                self.cancelled.clear()
                self.thread = threading.Thread(target=self._run, args=(document,), name="hqsb-benchmark", daemon=True)
                self.thread.start()
                return self.get(identity)
            except BaseException:
                self.active_id = None
                self.service.release_benchmark(identity)
                raise

    def cancel(self, identity):
        with self.lock:
            document = self.get(identity)
            if document["state"] in FINAL:
                return document
            if identity != self.active_id:
                raise ServiceError("BENCHMARK_STATE", "该任务当前没有执行所有者。")
            self.cancelled.set()
            document["state"] = "cancelling"
            # The running thread owns the mutable journal; cancellation intent is an event.
            return document

    def _check(self):
        if self.cancelled.is_set() or self.stopping.is_set():
            raise CampaignCancelled()

    def _state(self, document):
        return next(row for row in self.service.deployments() if row["id"] == document["plan"]["deployment_id"])

    def _task(self, document, kind, payload, phase, *, restoring=False):
        if not restoring:
            self._check()
        run = self.service.submit(kind, document["plan"]["deployment_id"], payload, uuid.uuid4().hex, benchmark_owner=document["id"])
        document.update(phase=phase, current_run_id=run["id"])
        if kind != "generate":
            document["lifecycle"].append({"run_id": run["id"], "kind": kind, "payload": payload})
        self._save(document)
        while run["state"] not in TERMINAL:
            if self.stopping.is_set() or (self.cancelled.is_set() and not restoring):
                self.service.cancel(run["id"])
            if self.stopping.is_set():
                raise CampaignCancelled()
            time.sleep(.2)
            run = self.service.store.get(run["id"])
        document["current_run_id"] = None
        return run

    def _load(self, document, profile, *, restoring=False, unload_only=False):
        state = self._state(document)
        if state["state"] == "ready" and not unload_only and state.get("kernel_profile", "framework-native") == profile:
            return
        if state["state"] == "ready":
            run = self._task(document, "unload", {}, "restoring" if restoring else "unloading", restoring=restoring)
            if run["state"] != "completed" or run["cleanup"] != "succeeded":
                raise RuntimeError(run.get("error") or "模型卸载未完成。")
        if not unload_only:
            run = self._task(document, "load", {"kernel_profile": profile}, "restoring" if restoring else "loading", restoring=restoring)
            actual = self._state(document)
            if run["state"] != "completed" or actual["state"] != "ready" or actual.get("kernel_profile") != profile or actual.get("load_run_id") != run["id"]:
                raise RuntimeError(run.get("error") or "加载后的算子方案未确认。")

    def _run(self, document):
        final = "completed"
        try:
            plan = document["plan"]
            for scheme in plan["schemes"]:
                self._check()
                self._load(document, scheme["kernel_profile"])
                for case in plan["cases"]:
                    roles = [("warmup", 0), *[("measurement", i) for i in range(plan["repeats"])]]
                    if plan["diagnostics"]:
                        roles.append(("diagnostic", 0))
                    for role, repeat in roles:
                        self._check()
                        state = self._state(document)
                        payload = {"expected_epoch": state["epoch"], "messages": case["messages"],
                                   "max_output_tokens": plan["output_tokens"], "deadline_ms": self.settings.request_deadline_ms,
                                   "save_input": True, "observation_mode": "memory" if role == "diagnostic" else "off",
                                   "optimization": {"changes": {"kernel_profile": scheme["kernel_profile"], "cleanup_policy": scheme["cleanup_policy"]}},
                                   "benchmark_links": {"benchmark_id": document["id"], "case_id": case["id"], "scheme_id": scheme["id"], "role": role, "repeat": repeat}}
                        run = self._task(document, "generate", payload, role)
                        document["records"].append(project_run(run, scheme_id=scheme["id"], case=case, role=role, repeat=repeat))
                        self._save(document)
                        self._check()
                        if run["state"] != "completed" or run["cleanup"] != "succeeded":
                            raise RuntimeError(run.get("error") or f"用例任务 {run['state']}")
        except CampaignCancelled:
            final = "interrupted" if self.stopping.is_set() else "cancelled"
        except Exception as exc:
            final = "failed"
            document["error"] = str(exc)[:1000]
        finally:
            try:
                if not self.stopping.is_set():
                    initial = document["initial"]
                    self._load(document, initial.get("kernel_profile") or "framework-native", restoring=True, unload_only=initial["state"] != "ready")
            except Exception as exc:
                document["restore_error"] = str(exc)[:1000]
                final = "failed"
            with self.lock:
                document.update(state=final, phase=final, current_run_id=None)
                try:
                    self._save(document)
                finally:
                    self.service.release_benchmark(document["id"])
                    self.active_id = None

    def close(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=15)
