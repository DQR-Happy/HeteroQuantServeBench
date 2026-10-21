"""Node-owned model discovery and resumable acquisition, outside device workers.

Catalog summaries are not execution evidence. A frozen publisher file plan is
hashed before download; only fully verified snapshots may become deployments.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from itertools import chain
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

from hqsb.console.config import DeploymentConfig
from hqsb.console.resources import host_memory
from hqsb.console.service import ServiceError
from hqsb.console.model_hub_storage import artifact_paths, remove_artifacts
from hqsb.console.telemetry import nvidia_sample
from hqsb.core.optimization import fingerprint
from hqsb.models.manifest import verify_or_raise
from hqsb.models.model_admission import admission, estimate, effective_host_available
from hqsb.models.modelscope_source import ModelScopeSource, PUBLISHERS, UNSUPPORTED_FORMAT, validate_repo

ACTIVE = {"queued", "downloading", "verifying", "cancelling"}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checksum(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


class DownloadCancelled(Exception):
    pass


class ModelHub:
    def __init__(self, settings, service, *, source_factory=ModelScopeSource, capacity=None):
        self.settings, self.service = settings, service
        self.root = settings.data_dir / "model-hub"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.models = (settings.model_download_dir or self.root / "weights").expanduser().resolve()
        self.source_factory, self.capacity_source = source_factory, capacity
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.stop = {}
        self.catalogs, self.jobs = {}, {}
        self.catalog_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hqsb-model-catalog")
        self.download_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hqsb-model-download")
        self.inspections = {}
        for path in self.root.glob("catalog-*.json"):
            item = json.loads(path.read_text())
            if item["state"] == "refreshing":
                item.update(state="interrupted", error="目录刷新被服务重启中断，可重新刷新")
            self.catalogs[item["publisher"]] = item
        for path in self.root.glob("download-*.json"):
            item = json.loads(path.read_text())
            if item["state"] in ACTIVE:
                item.update(state="interrupted", error="服务重启，部分文件已保留，请显式继续下载")
                save_json(path, item)
            self.jobs[item["id"]] = item
        registration = self.root / "deployments.json"
        self.registered = json.loads(registration.read_text()) if registration.exists() else {}
        interrupted = set()
        for item in self.jobs.values():
            if item["state"] in {"deleting", "delete_failed"}:
                interrupted.update(item.get("delete_deployment_ids", []))
                if item["state"] == "deleting":
                    item.update(state="delete_failed", error="上次删除被服务重启中断，请重试删除")
                    save_json(self.root / (item["id"] + ".json"), item)
        if interrupted:
            self.registered = {key: raw for key, raw in self.registered.items() if key not in interrupted}
            save_json(registration, self.registered)
        for raw in self.registered.values():
            self.service.register_deployment(DeploymentConfig.model_validate(raw))

    def capacity(self):
        if self.capacity_source:
            return self.capacity_source()
        self.models.mkdir(parents=True, exist_ok=True, mode=0o700)
        memory = host_memory()
        gpu = nvidia_sample()
        shared = Path("/etc/nv_tegra_release").exists()
        return {"gpu_total_bytes": gpu.get("gpu_memory_total_bytes", memory.get("total_bytes") if shared else None),
                "gpu_name": gpu.get("gpu_name", "Tegra 共享内存" if shared else "未知"),
                "host_available_bytes": effective_host_available(memory.get("available_bytes"), Path("/sys/fs/cgroup")),
                "disk_free_bytes": shutil.disk_usage(self.models).free,
                "download_directory": str(self.models), "sampled_at": time.time(),
                "gpu_scope": "shared_system_memory" if shared else "physical_gpu_0"}

    def config(self):
        return {"publishers": [{"id": key, "name": value[0], "provenance": value[1]} for key, value in PUBLISHERS.items()],
                "capacity": self.capacity(), "disk_reserve_bytes": self.settings.model_download_reserve_bytes,
                "runtime": "FP16 / batch 1 / context ≤ 2048 / native Transformers / no offload",
                "scope": "目录收录登记发布方的公开仓库，平台镜像单列；模型可在下载后加载验证。"}

    def refresh(self, owner):
        if owner not in PUBLISHERS:
            raise ServiceError("MODEL_PUBLISHER_NOT_ALLOWED", "发布方不在来源策略中", 422)
        with self.lock:
            if self.closed.is_set():
                raise ServiceError("SHUTTING_DOWN", "服务正在关闭", 503)
            previous = self.catalogs.get(owner)
            if previous and previous["state"] == "refreshing":
                return self.catalog(owner)
            cached = deepcopy(previous["items"]) if previous else []
            self.catalogs[owner] = {"publisher": owner, "state": "refreshing", "items": cached,
                                    "retained_cached": bool(cached), "scanned": 0, "duplicates": 0,
                                    "page_requests": 0, "retrieval": None,
                                    "total_upstream": previous.get("total_upstream") if previous else None,
                                    "fetched_at": time.time(), "error": None}
            self.catalog_pool.submit(self._refresh, owner)
            return self.catalog(owner)

    def _refresh(self, owner):
        source = None
        try:
            source = self.source_factory()
            seen, fresh, scanned, totals = set(), [], 0, set()
            # Equal-ranked rows can cross upstream page boundaries. Prefer a
            # bounded single response; fall back if the server caps that request.
            requests = chain([(1, 10000)], ((page, 50) for page in range(1, 201)))
            for page, page_size in requests:
                if self.closed.is_set():
                    raise DownloadCancelled()
                rows, total = source.page(owner, page, page_size=page_size)
                with self.lock:
                    item = self.catalogs[owner]
                    scanned += len(rows)
                    totals.add(total)
                    item["page_requests"] += 1
                    item["retrieval"] = "single_response" if page_size == 10000 else "paged"
                    for row in rows:
                        if row["model_id"] not in seen:
                            if len(seen) >= 10000:
                                raise ValueError("目录超过 10000 个仓库索引上限，不能视为全部")
                            fresh.append(row)
                            seen.add(row["model_id"])
                    # Keep a usable prior catalogue visible throughout refresh.
                    # Initial discovery can still expose newly received pages.
                    if not item["retained_cached"]:
                        item["items"] = list(fresh)
                    item["scanned"] = scanned
                    item["duplicates"] = scanned - len(seen)
                    item["total_upstream"] = total
                    item["fetched_at"] = time.time()
                    complete = len(seen) == total and len(totals) == 1
                    if complete or (page_size == 50 and not rows):
                        item["state"] = "completed" if complete else "partial"
                        item["items"] = list(fresh)
                        item["retained_cached"] = False
                        if item["state"] == "partial":
                            item["error"] = (f"上游报告 {total} 项，本轮读取 {scanned} 条记录、去重后 {len(seen)} 个仓库"
                                             f"（重复 {scanned - len(seen)} 条）；上游分页或可见性计数不一致，无法确认完整性。已取得的模型仍可检查。"
                                             + (" 刷新期间上游总数发生变化，请稍后重试。" if len(totals) > 1 else ""))
                    save_json(self.root / f"catalog-{owner}.json", item)
                    if item["state"] != "refreshing":
                        return
            raise ValueError("目录超过 10000 条索引上限；已保留部分结果，不能视为全部")
        except Exception as exc:
            with self.lock:
                self.catalogs[owner].update(state="partial" if self.catalogs[owner]["items"] else "failed", error=f"{type(exc).__name__}: {exc}"[:600])
                save_json(self.root / f"catalog-{owner}.json", self.catalogs[owner])
        finally:
            if source:
                source.close()

    def catalog(self, owner, query="", only_fit=False, offset=0, limit=20, sort_by="default", sort_order="asc"):
        if owner not in PUBLISHERS:
            raise ServiceError("MODEL_PUBLISHER_NOT_ALLOWED", "未知发布方", 422)
        capacity = self.capacity()
        with self.lock:
            item = deepcopy(self.catalogs.get(owner, {"publisher": owner, "state": "not_loaded", "items": [], "error": None, "total_upstream": None}))
        rows = []
        indexed = len(item["items"])
        for row in item.pop("items"):
            # Summary estimate only. Exact revision must pass config/file inspection.
            weights = 2 * row["parameters"] if row["parameters"] else row["weight_bytes"]
            budget = estimate(weights)
            possible = bool(weights and capacity.get("gpu_total_bytes") and budget["estimated_runtime_bytes"] <= capacity["gpu_total_bytes"])
            text_model = any(name.endswith("ForCausalLM") for name in row["architectures"])
            supported_format = not UNSUPPORTED_FORMAT.search(row["model_id"].split("/")[1])
            pending = text_model and supported_format and (not weights or capacity.get("gpu_total_bytes") is None)
            status = "needs_inspection" if pending else "estimated_fit" if possible and text_model and supported_format else "excluded"
            row.update(estimate=budget, resource_candidate=status == "estimated_fit", candidate_status=status)
            if query.lower() in row["model_id"].lower() and (not only_fit or status != "excluded"):
                rows.append(row)
        if sort_by not in {"default", "parameters", "runtime_bytes"} or sort_order not in {"asc", "desc"}:
            raise ServiceError("MODEL_CATALOG_SORT", "不支持的目录排序方式", 422)
        if sort_by == "default":
            rows.sort(key=lambda row: (not row["resource_candidate"], row["model_id"]))
        else:
            def order_key(row):
                value = row["parameters"] if sort_by == "parameters" else row["estimate"]["estimated_runtime_bytes"]
                # Unknown values remain last in both directions; equal values
                # use repository identity for deterministic page boundaries.
                ordered = (-value if sort_order == "desc" else value) if value is not None else 0
                return value is None, ordered, row["model_id"]
            rows.sort(key=order_key)
        return {**item, "indexed": indexed, "total": len(rows),
                "sort_by": sort_by, "sort_order": sort_order,
                "items": rows[offset:offset + limit], "offset": offset, "limit": limit,
                "filter_scope": "发布方已索引的全部摘要；原生纯文本模型缺少体积时保留为待检查，不视为资源通过"}

    def revisions(self, repo):
        source = self.source_factory()
        try:
            return source.revisions(validate_repo(repo))
        finally:
            source.close()

    def inspect(self, repo, revision):
        source = self.source_factory()
        try:
            plan = source.inspect(repo, revision)
        finally:
            source.close()
        identity = fingerprint(plan)
        with self.lock:
            if len(self.inspections) >= 128:
                self.inspections.pop(next(iter(self.inspections)))
            self.inspections[identity] = (time.time(), deepcopy(plan))
        return {"plan_sha256": identity, "plan": plan,
                "admission": admission(plan, self.capacity(), self.settings.model_download_reserve_bytes)}

    def list_jobs(self):
        with self.lock:
            return {"items": [self._public(item) for item in sorted(self.jobs.values(), key=lambda row: row["created_at"], reverse=True)]}

    def _public(self, item):
        return {**deepcopy({key: value for key, value in item.items() if key != "plan"}),
                "deletion": self.deletion_status(item)}

    def deletion_status(self, item):
        """Describe deletion prerequisites using the same registry lock as submission."""
        result = {"allowed": False, "reason": "", "deployment_ids": [], "paths": []}
        if item["state"] == "deleted":
            return {**result, "reason": "本地文件已删除，可重新选择版本下载"}
        if item["state"] in ACTIVE or item["state"] == "deleting":
            return {**result, "reason": "请先取消下载并等待任务停止，再清理本地文件"}
        if any(row["plan_sha256"] == item["plan_sha256"] and row["state"] in ACTIVE for row in self.jobs.values()):
            return {**result, "reason": "相同制品仍有下载任务，请先取消并等待停止"}
        if self.closed.is_set():
            return {**result, "reason": "节点服务正在关闭，请等待重新连接"}
        try:
            paths = artifact_paths(self.models, item)
        except (OSError, ValueError) as exc:
            return {**result, "reason": str(exc)}
        result["paths"] = [str(path) for path in paths if path.exists()]
        with self.service.lock:
            for identity, cfg in self.service.configs.items():
                if cfg.provider not in {"pytorch", "vllm", "sglang"}:
                    continue
                model = Path(cfg.model_path).expanduser().resolve()
                if not any(model == path or model.is_relative_to(path) or path.is_relative_to(model) for path in paths):
                    continue
                result["deployment_ids"].append(identity)
                link = {"to": f"/devices#deployment-{identity}", "label": "打开对应部署"}
                if cfg.provider in {"vllm", "sglang"}:
                    return {**result, "reason": "请先卸载并移除引用此制品的原生引擎部署，再删除权重。",
                            "next": {"to": "/engines?deployment=" + identity, "label": "原生引擎调优"}}
                if (identity not in self.registered and identity not in item.get("delete_deployment_ids", [])) or any(cfg.id == identity for cfg in self.settings.deployments):
                    return {**result, "reason": "该制品被节点配置中的部署引用，请先移除对应配置并重启服务", "next": link}
                state = self.service.states[identity]
                if state["state"] not in {"unloaded", "failed"} or state.get("epoch"):
                    return {**result, "reason": f"{cfg.name} 当前为 {state['state']}，请先在设备与部署页卸载", "next": link}
            if self.service.pending or self.service.benchmark_owner or self.service.closed:
                return {**result, "reason": "当前节点有任务正在执行，请等待任务结束后删除", "next": {"to": "/runs", "label": "查看运行任务"}}
        return {**result, "allowed": True}

    def delete(self, identity, plan_sha):
        with self.lock, self.service.lock:
            item = self.jobs[identity]
            if plan_sha != item["plan_sha256"]:
                raise ServiceError("MODEL_DELETE_IDENTITY", "制品身份已变化，请刷新后重新确认", 409)
            if item["state"] == "deleted":
                return self._public(item)
            siblings = [row for row in self.jobs.values() if row["plan_sha256"] == plan_sha and row["state"] != "deleted"]
            if any(row["state"] in ACTIVE for row in siblings):
                raise ServiceError("MODEL_DELETE_BUSY", "相同制品仍有下载任务，请先取消并等待停止", 409)
            status = self.deletion_status(item)
            if not status["allowed"]:
                raise ServiceError("MODEL_DELETE_BLOCKED", status["reason"], 409)
            identities = sorted(set(status["deployment_ids"]) | set(item.get("delete_deployment_ids", [])))
            for row in siblings:
                self._update(row["id"], state="deleting", delete_deployment_ids=identities)
            try:
                registration = {key: raw for key, raw in self.registered.items() if key not in identities}
                save_json(self.root / "deployments.json", registration)
                self.registered = registration
                self.service.retire_deployments(identities)
                for row in siblings:
                    remove_artifacts(artifact_paths(self.models, row))
            except (OSError, ValueError, RuntimeError, ServiceError) as exc:
                for row in siblings:
                    self._update(row["id"], state="delete_failed", error=f"删除未完成：{exc}"[:600])
                raise ServiceError("MODEL_DELETE_FAILED", "删除未完成，请检查原因后重试：" + str(exc)[:400], 409) from exc
            for row in siblings:
                self._update(row["id"], state="deleted", deleted_at=time.time(), deployment_id=None,
                             error=None, current_file=None, bytes_per_second=None, downloaded_bytes=0, verified_files=0)
            return self._public(item)

    def _update(self, identity, **changes):
        with self.lock:
            item = self.jobs[identity]
            item.update(changes, updated_at=time.time())
            save_json(self.root / (identity + ".json"), item)
            return self._public(item)

    def start(self, plan_sha):
        with self.lock:
            if self.closed.is_set():
                raise ServiceError("SHUTTING_DOWN", "服务正在关闭", 503)
            for item in self.jobs.values():
                if item["plan_sha256"] == plan_sha and item["state"] != "deleted":
                    return self._public(item)
            entry = self.inspections.get(plan_sha)
            if not entry or time.time() - entry[0] > 1800:
                raise ServiceError("MODEL_INSPECTION_EXPIRED", "请重新检查所选版本", 409)
            if len(self.jobs) >= 200 or sum(item["state"] in ACTIVE for item in self.jobs.values()) >= 8:
                raise ServiceError("MODEL_DOWNLOAD_LIMIT", "下载记录或队列达到节点上限", 429)
            plan = deepcopy(entry[1])
            result = admission(plan, self.capacity(), self.settings.model_download_reserve_bytes)
            if not result["eligible"]:
                raise ServiceError("MODEL_ADMISSION", "；".join(result["reasons"]), 409)
            identity = "download-" + uuid.uuid4().hex
            self.jobs[identity] = {"id": identity, "model_id": plan["model_id"], "revision": plan["revision"],
                "plan": plan, "plan_sha256": plan_sha, "state": "queued", "created_at": time.time(),
                "updated_at": time.time(), "total_bytes": plan["download_bytes"], "downloaded_bytes": 0,
                "verified_files": 0, "total_files": len(plan["files"]), "current_file": None,
                "bytes_per_second": None, "error": None, "deployment_id": None, "attempt": 1}
            self._update(identity)
            self.stop[identity] = threading.Event()
            self.download_pool.submit(self._download, identity)
            return self._public(self.jobs[identity])

    def cancel(self, identity):
        with self.lock:
            item = self.jobs[identity]
            if item["state"] in ACTIVE:
                self.stop[identity].set()
                return self._update(identity, state="cancelling")
            return self._public(item)

    def resume(self, identity):
        with self.lock:
            item = self.jobs[identity]
            if item["state"] not in {"failed", "cancelled", "interrupted"}:
                return self._public(item)
            if self.closed.is_set() or sum(row["state"] in ACTIVE for row in self.jobs.values()) >= 8:
                raise ServiceError("MODEL_DOWNLOAD_LIMIT", "下载队列不可用", 429)
            self.stop[identity] = threading.Event()
            self._update(identity, state="queued", error=None, attempt=item["attempt"] + 1)
            self.download_pool.submit(self._download, identity)
            return self._public(item)

    def _download(self, identity):
        source = None
        item = self.jobs[identity]
        plan, sha = item["plan"], item["plan_sha256"]
        directory = self.models / ("partial-" + sha)
        destination = self.models / sha
        started, transferred, last_report = time.monotonic(), 0, 0.0
        def cancelled():
            if self.closed.is_set() or self.stop[identity].is_set():
                raise DownloadCancelled()
        try:
            cancelled()
            source = self.source_factory()
            self.models.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Rename and journal publication can be separated by a crash. Recover
            # the exact sealed snapshot instead of downloading another copy.
            manifest = self.root / (sha + ".sha256")
            if destination.exists():
                if destination.is_symlink():
                    raise ValueError("MODEL_DIRECTORY_SYMLINK")
                verify_or_raise(destination, manifest)
                save_json(self.root / (sha + ".provenance.json"), {"source": "modelscope", "plan": plan, "plan_sha256": sha, "sealed_at": time.time()})
                self._update(identity, state="completed", downloaded_bytes=plan["download_bytes"],
                             snapshot_path=str(destination), manifest_path=str(manifest), verified_files=len(plan["files"]), error=None)
                return
            directory.mkdir(exist_ok=True, mode=0o700)
            if directory.is_symlink() or destination.is_symlink():
                raise ValueError("MODEL_DIRECTORY_SYMLINK")
            present = 0
            for entry in plan["files"]:
                existing = directory / entry["Path"]
                if not existing.exists():
                    existing = existing.with_name(existing.name + ".part")
                if existing.is_symlink():
                    raise ValueError("MODEL_FILE_SYMLINK")
                if existing.exists():
                    present += min(existing.stat().st_size, entry["Size"])
            done = 0
            self._update(identity, state="downloading", downloaded_bytes=present, verified_files=0, bytes_per_second=None)
            for count, entry in enumerate(plan["files"]):
                cancelled()
                path = directory / entry["Path"]
                partial = path.with_name(path.name + ".part")
                if path.is_symlink() or partial.is_symlink():
                    raise ValueError("MODEL_FILE_SYMLINK")
                if path.exists():
                    if path.stat().st_size != entry["Size"] or checksum(path) != entry["Sha256"].lower():
                        raise ValueError("MODEL_EXISTING_FILE_CORRUPT:" + entry["Path"])
                else:
                    offset = partial.stat().st_size if partial.exists() else 0
                    if offset > entry["Size"]:
                        raise ValueError("MODEL_PARTIAL_SIZE_INVALID")
                    # Recheck remaining plan at execution time, including queued tasks.
                    if shutil.disk_usage(self.models).free < plan["download_bytes"] - present - transferred + self.settings.model_download_reserve_bytes:
                        raise ValueError("MODEL_DISK_BUDGET:剩余磁盘不足，部分文件已保留")
                    self._update(identity, current_file=entry["Path"], downloaded_bytes=present + transferred)
                    if offset < entry["Size"]:
                        with source.stream(plan["model_id"], entry, offset) as response, partial.open("ab") as stream:
                            for chunk in response.iter_content(1024 * 1024):
                                cancelled()
                                if not chunk:
                                    continue
                                if offset + len(chunk) > entry["Size"]:
                                    raise ValueError("MODEL_FILE_SIZE_CHANGED")
                                stream.write(chunk)
                                offset += len(chunk)
                                transferred += len(chunk)
                                now = time.monotonic()
                                if now - last_report >= 0.5:
                                    if shutil.disk_usage(self.models).free < self.settings.model_download_reserve_bytes:
                                        raise ValueError("MODEL_DISK_RESERVE_EXHAUSTED")
                                    self._update(identity, downloaded_bytes=present + transferred,
                                        bytes_per_second=transferred / max(now - started, 0.001))
                                    last_report = now
                            stream.flush()
                            os.fsync(stream.fileno())
                    if not partial.exists() and entry["Size"] == 0:
                        partial.touch()
                    if partial.stat().st_size != entry["Size"] or checksum(partial) != entry["Sha256"].lower():
                        # Corrupt data cannot be resumed as a valid prefix; quarantine
                        # only this task-owned partial, preserving diagnosis evidence.
                        quarantine = self.models / "quarantine"
                        quarantine.mkdir(exist_ok=True, mode=0o700)
                        shutil.move(str(partial), str(quarantine / (identity + "-" + uuid.uuid4().hex)))
                        raise ValueError("MODEL_FILE_HASH_MISMATCH:" + entry["Path"])
                    partial.replace(path)
                done += entry["Size"]
                self._update(identity, downloaded_bytes=present + transferred, verified_files=count + 1)
            cancelled()
            self._update(identity, state="verifying", current_file=None, bytes_per_second=None)
            manifest = self.root / (sha + ".sha256")
            manifest.write_text("".join(f"{entry['Sha256'].lower()}  {entry['Path']}\n" for entry in plan["files"]))
            verify_or_raise(directory, manifest)
            if destination.exists():
                verify_or_raise(destination, manifest)
                # Never overwrite an existing sealed snapshot.
                raise ValueError("MODEL_SNAPSHOT_ALREADY_EXISTS:已存在封存目录，请检查登记状态")
            with self.lock:
                cancelled()
                directory.rename(destination)
                save_json(self.root / (sha + ".provenance.json"), {"source": "modelscope", "plan": plan, "plan_sha256": sha, "sealed_at": time.time()})
                self._update(identity, state="completed", downloaded_bytes=plan["download_bytes"],
                             snapshot_path=str(destination), manifest_path=str(manifest), verified_files=len(plan["files"]))
        except DownloadCancelled:
            self._update(identity, state="interrupted" if self.closed.is_set() else "cancelled", bytes_per_second=None, error="部分文件保留，可显式继续")
        except Exception as exc:
            self._update(identity, state="failed", bytes_per_second=None, error=f"{type(exc).__name__}: {exc}"[:800])
        finally:
            if source:
                source.close()

    def register(self, identity):
        with self.lock:
            item = self.jobs[identity]
            if item["state"] != "completed":
                raise ServiceError("MODEL_NOT_SEALED", "只有校验封存完成的权重可以登记", 409)
            name = "ms-" + item["plan_sha256"][:20]
            if item.get("deployment_id"):
                return self._public(item)
            if len(self.service.configs) >= 16:
                raise ServiceError("DEPLOYMENT_LIMIT", "节点最多登记 16 个部署；权重已下载，不受此限制", 409)
            config = DeploymentConfig(id=name, name=item["model_id"].split("/")[1] + " · " + item["revision"],
                model=item["model_id"], model_path=item["snapshot_path"], manifest=item["manifest_path"],
                input_mode=item["plan"].get("input_mode", "chat"),
                precision=item["plan"].get("runtime_precision", "float16"),
                platform="CUDA / " + self.capacity().get("gpu_name", "当前节点"), attention="sdpa",
                context_limit=min(2048, item["plan"]["config"].get("max_position_embeddings") or 2048),
                device_lock_path=next((cfg.device_lock_path for cfg in self.service.configs.values() if cfg.provider == "pytorch" and cfg.device_lock_path), str(Path.home() / ".cache/hqsb/device0.lock")))
            registration = {**self.registered, name: config.model_dump()}
            # persist before publishing the new actor; restart can replay registration.
            save_json(self.root / "deployments.json", registration)
            self.registered = registration
            self.service.register_deployment(config)
            return self._update(identity, deployment_id=name)

    def close(self):
        self.closed.set()
        self.catalog_pool.shutdown(wait=True)
        self.download_pool.shutdown(wait=True)
