"""L6 control-plane routes; tensor operations stay in the serial device worker."""
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict
from hqsb.console.service import ServiceError
from hqsb.console.captures import safe_directory
from hqsb.console.store import TERMINAL
from hqsb.core.memory_plan import catalogue
from hqsb.experiments.artifacts import read_manifest, read_json


class ClearPrefix(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deployment_id: str
    expected_epoch: str


class MemoryComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline_run_id: str
    candidate_run_id: str


def memory_router(settings, service):
    router = APIRouter(tags=["memory-management"])

    @router.get("/memory/catalog")
    def catalog():
        return {**catalogue(), "max_deadline_ms": settings.request_deadline_ms}

    def worker(deployment_id, operation, epoch=None):
        with service.lock:
            if deployment_id not in service.configs:
                raise ServiceError("NOT_FOUND", "Unknown deployment", 404)
            state = service.states[deployment_id]
            if state["state"] != "ready":
                raise ServiceError("DEPLOYMENT_NOT_READY", "请在设备与部署页加载模型。", 409)
            if service.configs[deployment_id].provider != "pytorch" or settings.privileged_worker:
                raise ServiceError("L6_UNSUPPORTED_WORKER", "请选择本机非特权 PyTorch 部署。", 422)
            if epoch is not None and state["epoch"] != epoch:
                raise ServiceError("STALE_EPOCH", "模型已切换，请刷新模型状态。", 409)
            if service.pending or service.benchmark_owner:
                raise ServiceError("DEVICE_BUSY", "请等待运行记录中的当前任务结束后再检查或清空 KV。", 409)
            service.pending += 1
            future = service.executors[deployment_id].submit(service.actors[deployment_id].execute,
                operation, {}, lambda _: None, lambda: service.closed, 30)
        try:
            return future.result(timeout=35)
        finally:
            with service.lock:
                service.pending -= 1

    @router.get("/memory/capabilities")
    def capabilities(deployment_id: str):
        return worker(deployment_id, "memory_capabilities")

    @router.post("/memory/prefix/clear")
    def clear(body: ClearPrefix):
        return worker(body.deployment_id, "memory_cache_clear", body.expected_epoch)

    @router.get("/memory/runs")
    def runs(deployment_id: str):
        if deployment_id not in service.configs:
            raise ServiceError("NOT_FOUND", "Unknown deployment", 404)
        with service.store.lock:
            rows = service.store.db.execute(
                "SELECT id FROM runs WHERE json_extract(body, '$.config.deployment.id')=? "
                "AND json_extract(body, '$.config.memory_plan') IS NOT NULL ORDER BY created DESC LIMIT 60", (deployment_id,)).fetchall()
        return {"items": [service.store.get(row[0]) for row in rows]}

    def read(run_id):
        try:
            run = service.store.get(run_id)
            if run["state"] not in TERMINAL:
                raise ServiceError("RUN_NOT_TERMINAL", "请等待设备清理和凭据发布。", 409)
            directory = safe_directory(settings.data_dir / "captures", run_id)
            manifest = read_manifest(directory)
            if "memory-cache.json" not in {a["path"] for a in manifest["artifacts"]}:
                raise ValueError("L6_MEMORY_ARTIFACT_MISSING")
            doc = read_json(directory / "memory-cache.json")
            receipt = read_json(directory / "execution-receipt.json")
            if doc.get("run_id") != run_id or doc.get("epoch") != run["config"].get("epoch") or receipt.get("run_id") != run_id:
                raise ValueError("L6_ARTIFACT_IDENTITY")
            if doc != run["metrics"].get("memory_receipt") or doc != receipt.get("memory_receipt"):
                raise ValueError("L6_RECEIPT_IDENTITY")
            return run, doc, receipt, manifest
        except (KeyError, ValueError, OSError) as exc:
            raise ServiceError("L6_ARTIFACT_UNAVAILABLE", str(exc), 409) from exc

    @router.get("/runs/{run_id}/memory-cache")
    def memory(run_id: str):
        _, doc, _, manifest = read(run_id)
        return {**doc, "publication": manifest["publication"]}

    @router.post("/memory/comparisons")
    def comparison(body: MemoryComparison):
        from hqsb.core.memory_comparison import compare
        a, _, ra, ma = read(body.baseline_run_id)
        b, _, rb, mb = read(body.candidate_run_id)
        return compare(a, b, ra, rb, ma, mb)

    return router
