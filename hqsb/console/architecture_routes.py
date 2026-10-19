"""Read-only artifact inspection on the selected node; never loads a GPU model."""
from fastapi import APIRouter
from hqsb.console.service import ServiceError
from hqsb.models.architecture import inspect_architecture
from hqsb.console.captures import safe_directory
from hqsb.console.store import TERMINAL
from hqsb.experiments.artifacts import read_manifest, read_json


def architecture_router(service):
    router = APIRouter(tags=["model-architecture"])

    @router.get("/deployments/{deployment_id}/architecture")
    def architecture(deployment_id: str):
        config = service.configs.get(deployment_id)
        if config is None:
            raise ServiceError("NOT_FOUND", "部署不存在", 404)
        if config.provider != "pytorch":
            raise ServiceError("ARCHITECTURE_UNAVAILABLE", "架构解析需要本地模型制品，请选择 PyTorch 部署", 422)
        try:
            result = inspect_architecture(config.model_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ServiceError("ARCHITECTURE_INSPECTION", f"无法读取模型结构：{type(exc).__name__}: {exc}"[:600], 422) from exc
        with service.lock:
            state = service.states[deployment_id]
            runtime = {"state": state["state"], "epoch": state["epoch"], "load_run_id": state.get("load_run_id"),
                       "kernel_profile": state.get("kernel_profile"), "model_class": state["detail"].get("model_class"),
                       "weight_verification": state["detail"].get("weight_verification"),
                       "modules": state["detail"].get("module_inventory"),
                       "structure_receipt": state["detail"].get("structure_receipt"),
                       "structure_scope": "original_config_with_l5_module_overrides" if state.get("structure_artifact_id") else "original_model",
                       "loaded_config_matches": state["detail"].get("metadata_hashes", {}).get("config.json") == result["config_sha256"] if state["epoch"] else None}
        with service.store.lock:
            row = service.store.db.execute("SELECT id FROM runs WHERE json_extract(body, '$.config.deployment.id')=? "
                "AND json_extract(body, '$.metrics.architecture_capture') IS NOT NULL ORDER BY created DESC LIMIT 1", (deployment_id,)).fetchone()
        latest = service.store.get(row[0]) if row else None
        return {**result, "deployment_id": deployment_id, "model": config.model, "name": config.name, "runtime": runtime,
                "execution": {"run_id": latest["id"], "state": latest["state"],
                              "epoch_matches": latest["config"]["epoch"] == runtime["epoch"],
                              **latest["metrics"]["architecture_capture"]} if latest else None}

    @router.get("/deployments/{deployment_id}/architecture/execution/{run_id}")
    def execution(deployment_id: str, run_id: str):
        run = service.store.get(run_id)
        if run["config"]["deployment"]["id"] != deployment_id or run["state"] not in TERMINAL:
            raise ServiceError("ARCHITECTURE_RUN", "采集任务尚未结束或不属于此部署。", 409)
        try:
            directory = safe_directory(service.settings.data_dir / "captures", run_id)
            manifest = read_manifest(directory)
            if not any(row["path"] == "architecture-capture.json" for row in manifest["artifacts"]):
                raise ValueError("ARCHITECTURE_CAPTURE_MISSING")
            document = read_json(directory / "architecture-capture.json")
            if document.get("run_id") != run_id:
                raise ValueError("ARCHITECTURE_CAPTURE_IDENTITY")
            return {**document, "publication": manifest["publication"], "run_state": run["state"]}
        except (OSError, ValueError) as exc:
            raise ServiceError("ARCHITECTURE_CAPTURE_UNAVAILABLE", str(exc), 409) from exc

    return router
