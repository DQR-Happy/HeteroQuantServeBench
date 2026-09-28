"""Read-only artifact inspection on the selected node; never loads a GPU model."""
from fastapi import APIRouter
from hqsb.console.service import ServiceError
from hqsb.models.architecture import inspect_architecture


def architecture_router(service):
    router = APIRouter(tags=["model-architecture"])

    @router.get("/deployments/{deployment_id}/architecture")
    def architecture(deployment_id: str):
        config = service.configs.get(deployment_id)
        if config is None:
            raise ServiceError("NOT_FOUND", "部署不存在", 404)
        if config.provider != "pytorch":
            raise ServiceError("ARCHITECTURE_UNAVAILABLE", "外部服务未提供本地模型制品，不能推断内部架构", 422)
        try:
            result = inspect_architecture(config.model_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ServiceError("ARCHITECTURE_INSPECTION", f"无法读取模型结构：{type(exc).__name__}: {exc}"[:600], 422) from exc
        with service.lock:
            state = service.states[deployment_id]
            runtime = {"state": state["state"], "epoch": state["epoch"], "load_run_id": state.get("load_run_id"),
                       "kernel_profile": state.get("kernel_profile"), "model_class": state["detail"].get("model_class"),
                       "weight_verification": state["detail"].get("weight_verification"),
                       "loaded_config_matches": state["detail"].get("metadata_hashes", {}).get("config.json") == result["config_sha256"] if state["epoch"] else None}
        return {**result, "deployment_id": deployment_id, "model": config.model, "name": config.name, "runtime": runtime}

    return router
