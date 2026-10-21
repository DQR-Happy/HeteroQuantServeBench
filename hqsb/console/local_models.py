"""Node-local inventory, storage accounting and reference-aware cleanup."""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time

from hqsb.console.local_model_storage import discover, filesystem, measure
from hqsb.console.model_hub import save_json
from hqsb.console.model_hub_storage import artifact_paths, remove_artifacts
from hqsb.console.service import ServiceError
from hqsb.core.optimization import fingerprint


def overlaps(left, right):
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


class LocalModels:
    def __init__(self, hub):
        self.hub, self.service = hub, hub.service
        self.registry = hub.root / "local-model-roots.json"
        self.extra_roots = json.loads(self.registry.read_text()) if self.registry.exists() else []
        self.lock = threading.RLock()
        self.cached, self.cached_at = None, 0

    def roots(self):
        roots = {self.hub.models}
        for cfg in self.service.configs.values():
            if cfg.provider in {"pytorch", "vllm", "sglang"}:
                model = Path(cfg.model_path).expanduser().resolve()
                roots.add(model)
                if len(model.parent.parts) >= 4:
                    roots.add(model.parent)
        roots.update((self.hub.settings.data_dir / name).resolve() for name in ("l4-artifacts", "l5-artifacts"))
        roots.update(Path(value) for value in self.extra_roots)
        return sorted(roots)

    def add_root(self, raw):
        path = Path(raw).expanduser()
        if not path.is_absolute() or not path.is_dir():
            raise ServiceError("MODEL_DIRECTORY", "请填写当前节点上已存在的绝对目录路径", 422)
        path = path.resolve()
        if path == Path(path.anchor):
            raise ServiceError("MODEL_DIRECTORY", "请填写存放模型的具体目录", 422)
        with self.hub.lock, self.service.lock, self.lock:
            if str(path) not in self.extra_roots:
                if len(self.extra_roots) >= 32:
                    raise ServiceError("MODEL_DIRECTORY_LIMIT", "已登记 32 个额外目录，请在节点配置中整理扫描范围", 409)
                self.extra_roots.append(str(path))
                save_json(self.registry, self.extra_roots)
            self.cached_at = 0
        return self.snapshot(refresh=True)

    def _scan(self):
        roots = self.roots()
        found, issues = discover(roots)
        entries = {}
        def add(path, kind, **extra):
            path = path.absolute()
            key = str(path)
            entries.setdefault(key, {"id": "local-" + fingerprint(key)[:24], "path": key,
                                      "kind": kind, "model_id": path.name, "revision_name": "本地目录"})
            entries[key].update(kind=kind, **extra)
        for path in found:
            kind = "l4" if (path / "l4-manifest.json").exists() else "l5" if (path / "l5-manifest.json").exists() else "local"
            add(path, kind)
        for cfg in self.service.configs.values():
            if cfg.provider in {"pytorch", "vllm", "sglang"}:
                add(Path(cfg.model_path).expanduser().resolve(), "configured", model_id=cfg.model)
        for row in sorted(self.hub.jobs.values(), key=lambda row: row["created_at"]):
            if row["state"] == "deleted":
                continue
            try:
                paths = artifact_paths(self.hub.models, row)
                path = paths[0] if paths[0].exists() or row["state"] == "completed" else paths[1]
                add(path, "download", model_id=row["model_id"], revision_name=row["revision"],
                    download_id=row["id"], download_state=row["state"], plan_sha256=row["plan_sha256"],
                    cleanup_paths=[str(p) for p in paths if p.exists()])
            except (OSError, ValueError) as exc:
                issues.append(str(exc))
        for name in ("l4", "l5"):
            root = (self.hub.settings.data_dir / (name + "-artifacts")).resolve()
            if root.is_dir():
                for path in root.iterdir():
                    if path.is_dir() and not path.is_symlink():
                        add(path, name)
        volumes, all_inodes, remaining = {}, {}, 100000
        for item in entries.values():
            path = Path(item["path"])
            paths = [Path(p) for p in item.get("cleanup_paths", [str(path)])]
            measurements = []
            for target in paths or [path]:
                result = measure(target, budget=min(50000, remaining))
                remaining -= result.pop("entries_scanned")
                measurements.append(result)
            inodes = {f[:2]: f for result in measurements for f in result.pop("inodes")}
            all_inodes.update(inodes)
            item.update(exists=path.exists(), logical_bytes=sum(f[2] for f in inodes.values()),
                        allocated_bytes=sum(f[3] for f in inodes.values()), file_count=len(inodes),
                        complete=all(m["complete"] for m in measurements),
                        issues=[issue for m in measurements for issue in m["issues"]],
                        revision=fingerprint([m["revision"] for m in measurements]))
            try:
                volume = filesystem(path)
                item["device"] = volume["device"]
                volumes.setdefault(volume["device"], volume)
                for filename in ("config.json", "l4-manifest.json", "l5-manifest.json"):
                    source = path / filename
                    if source.is_file() and not source.is_symlink() and source.stat().st_size <= 4 * 1024**2:
                        doc = json.loads(source.read_text())
                        if not isinstance(doc, dict):
                            raise ValueError(f"请检查模型元数据格式：{source}")
                        for field in ("model_type", "artifact_id"):
                            if isinstance(doc.get(field), str):
                                item[field] = doc[field]
                        if item["kind"] in {"l4", "l5"} and isinstance(doc.get("source_model"), str):
                            item["model_id"] = doc["source_model"]
                        if isinstance(doc.get("recipe"), dict) and isinstance(doc["recipe"].get("method"), str):
                            item["variant"] = doc["recipe"]["method"]
                        if filename == "l5-manifest.json" and isinstance(doc.get("creation_run_id"), str):
                            try:
                                run = self.service.store.get(doc["creation_run_id"])
                                source = run["config"].get("deployment", {})
                                if isinstance(source, dict) and isinstance(source.get("model"), str):
                                    item["model_id"] = source["model"]
                            except KeyError:
                                item["issues"].append("源模型名称可在原创建运行记录中核对")
            except (OSError, ValueError) as exc:
                item["issues"].append(str(exc))
                item["complete"] = False
        # Inventory-wide inode union keeps hardlinks and nested directories from
        # inflating model storage. Filesystem used includes all other node data.
        return {"items": list(entries.values()), "roots": [str(p) for p in roots],
                "volumes": list(volumes.values()), "sampled_at": time.time(), "issues": issues,
                "logical_bytes": sum(f[2] for f in all_inodes.values()),
                "allocated_bytes": sum(f[3] for f in all_inodes.values()),
                "file_count": len(all_inodes)}

    def _references(self, item):
        path = Path(item["path"])
        refs = []
        for identity, cfg in self.service.configs.items():
            state = self.service.states[identity]
            model_ref = cfg.provider in {"pytorch", "vllm", "sglang"} and overlaps(path, Path(cfg.model_path).expanduser().resolve())
            artifact_ref = (state["state"] not in {"unloaded", "failed"} or state.get("epoch")) and item.get("artifact_id") and item["artifact_id"] in {
                state.get("structure_artifact_id"), (state.get("quantization") or {}).get("artifact_id")}
            if model_ref or artifact_ref:
                refs.append({"id": identity, "name": cfg.name, "state": state["state"], "epoch": state.get("epoch"),
                             "kind": "deployment" if model_ref else "loaded_artifact"})
        return refs

    def _policy(self, item):
        blocked = {"allowed": False, "reason": ""}
        if item.get("download_id"):
            return self.hub.deletion_status(self.hub.jobs[item["download_id"]])
        if not item["exists"]:
            return {**blocked, "reason": "模型目录已移除，可在节点配置中整理对应部署"}
        if not item["complete"]:
            return {**blocked, "reason": "请先处理目录扫描详情中的问题，再刷新空间统计"}
        if item["references"]:
            ref = item["references"][0]
            reason = (f"{ref['name']} 当前已加载此制品，请先在设备页卸载模型"
                      if ref["kind"] == "loaded_artifact" else
                      f"{ref['name']} 引用此目录，请先在设备页卸载，并在节点 YAML 配置中移除对应路径引用后重启服务")
            return {**blocked, "reason": reason,
                    "next": {"to": "/devices#deployment-" + ref["id"], "label": "查看对应部署"}}
        if self.service.pending or self.service.benchmark_owner or self.service.closed:
            return {**blocked, "reason": "当前节点有任务正在执行，请等待结束后清理文件", "next": {"to": "/runs", "label": "查看运行任务"}}
        path = Path(item["path"])
        for row in self.hub.jobs.values():
            if row["state"] != "deleted" and any(overlaps(path, self.hub.models / name) for name in (row["plan_sha256"], "partial-" + row["plan_sha256"])):
                return {**blocked, "reason": "此目录包含下载制品，请通过本地清单中的对应下载条目逐项清理", "next": {"to": "/models/local", "label": "查看下载制品"}}
        return {"allowed": True, "reason": "", "paths": [item["path"]], "deployment_ids": []}

    def snapshot(self, refresh=False):
        with self.hub.lock, self.service.lock, self.lock:
            current_jobs = {(row["id"], row["state"]) for row in self.hub.jobs.values() if row["state"] != "deleted"}
            cached_jobs = {(row["download_id"], row["download_state"]) for row in (self.cached or {}).get("items", []) if row.get("download_id")}
            if refresh or self.cached is None or current_jobs != cached_jobs or time.monotonic() - self.cached_at >= 30:
                self.cached = self._scan()
                self.cached_at = time.monotonic()
            result = deepcopy(self.cached)
            for item in result["items"]:
                item["references"] = self._references(item)
                item["deletion"] = self._policy(item)
                if item.get("download_id"):
                    job = self.hub.jobs[item["download_id"]]
                    item.update(download_state=job["state"], deployment_id=job.get("deployment_id"))
            result["items"] = [row for row in result["items"] if row.get("download_state") != "deleted"]
            return result

    def delete(self, identity, revision):
        with self.hub.lock, self.service.lock, self.lock:
            row = next((row for row in self.snapshot(refresh=True)["items"] if row["id"] == identity), None)
            if row is None:
                raise ServiceError("MODEL_NOT_FOUND", "目录已变化，请刷新本地模型清单", 404)
            if row["revision"] != revision:
                raise ServiceError("MODEL_DIRECTORY_CHANGED", "目录内容已变化，请刷新统计后重新确认", 409)
            if not row["deletion"]["allowed"]:
                raise ServiceError("MODEL_DELETE_BLOCKED", row["deletion"]["reason"], 409)
            if row.get("download_id"):
                try:
                    self.hub.delete(row["download_id"], row["plan_sha256"])
                finally:
                    self.cached_at = 0
            else:
                path = Path(row["path"])
                if path.is_symlink() or path.resolve() != path:
                    raise ServiceError("MODEL_DIRECTORY_CHANGED", "目录指向已变化，请刷新后检查", 409)
                try:
                    journal = self.hub.root / (identity + ".cleanup.json")
                    save_json(journal, {"state": "deleting", "model": row, "started_at": time.time()})
                    remove_artifacts([path])
                except (OSError, ValueError) as exc:
                    self.cached_at = 0
                    raise ServiceError("MODEL_DELETE_FAILED", str(exc), 409) from exc
                save_json(journal, {"state": "deleted", "model": row, "deleted_at": time.time()})
            self.cached_at = 0
            return {"id": identity, "state": "deleted", "deleted_at": time.time()}
