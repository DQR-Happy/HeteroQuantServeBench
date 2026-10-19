"""Model acquisition APIs follow the same selected-node/auth boundary as runs."""
from typing import Annotated, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict, Field

from hqsb.console.service import ServiceError


class HubSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=3, max_length=200)
    revision: str = Field(default="master", min_length=1, max_length=128)


class DownloadSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LocalDirectory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=1024)


class LocalDeletion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str = Field(pattern=r"^[0-9a-f]{64}$")


def model_hub_router(hub):
    router = APIRouter(prefix="/model-hub", tags=["model-hub"])
    from hqsb.console.local_models import LocalModels
    local = LocalModels(hub)

    @router.get("/local-models")
    def local_models():
        return local.snapshot()

    @router.post("/local-models/refresh")
    def refresh_local_models():
        return local.snapshot(refresh=True)

    @router.post("/local-models/roots")
    def add_local_root(body: LocalDirectory):
        return local.add_root(body.path)

    @router.post("/local-models/{identity}/delete")
    def delete_local_model(identity: str, body: LocalDeletion):
        return local.delete(identity, body.revision)

    @router.get("/config")
    def config():
        return hub.config()

    @router.get("/catalog")
    def catalog(publisher: str, q: str = Query("", max_length=200), only_fit: bool = False,
                offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=50),
                sort_by: Literal["default", "parameters", "runtime_bytes"] = "default",
                sort_order: Literal["asc", "desc"] = "asc"):
        return hub.catalog(publisher, q, only_fit, offset, limit, sort_by, sort_order)

    @router.post("/catalog/refresh")
    def refresh(publisher: Annotated[str, Query(max_length=80)]):
        return hub.refresh(publisher)

    def remote(operation):
        try:
            return operation()
        except (ValueError, ImportError) as exc:
            raise ServiceError("MODEL_HUB_REQUEST", str(exc)[:600], 422) from exc
        except ServiceError:
            raise
        except Exception as exc:
            # SDK/network failures are explicit, never fake an empty successful catalog.
            raise ServiceError("MODELSCOPE_UNAVAILABLE", f"{type(exc).__name__}: {exc}"[:600], 502) from exc

    @router.post("/revisions")
    def revisions(body: HubSelection):
        return remote(lambda: hub.revisions(body.model_id))

    @router.post("/inspect")
    def inspect(body: HubSelection):
        return remote(lambda: hub.inspect(body.model_id, body.revision))

    @router.get("/downloads")
    def downloads():
        return hub.list_jobs()

    @router.post("/downloads")
    def start(body: DownloadSelection):
        return hub.start(body.plan_sha256)

    @router.post("/downloads/{identity}/cancel")
    def cancel(identity: str):
        return hub.cancel(identity)

    @router.post("/downloads/{identity}/resume")
    def resume(identity: str):
        return hub.resume(identity)

    @router.post("/downloads/{identity}/register")
    def register(identity: str):
        return remote(lambda: hub.register(identity))

    @router.post("/downloads/{identity}/delete")
    def delete(identity: str, body: DownloadSelection):
        result = hub.delete(identity, body.plan_sha256)
        local.cached_at = 0
        return result

    return router
