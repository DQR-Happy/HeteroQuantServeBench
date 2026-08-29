"""Fixed-suite comparison API; browser selects IDs and bounded budgets only."""
from fastapi import APIRouter, Header, Query
from pydantic import BaseModel, ConfigDict, Field


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deployment_id: str
    baseline_id: str = "framework-native"
    candidate_ids: list[str] = Field(min_length=1, max_length=3)
    case_ids: list[str] = Field(min_length=1, max_length=4)
    repeats: int = Field(1, ge=1, le=5)
    output_tokens: int = Field(64, ge=8, le=128)
    diagnostics: bool = False


def benchmark_router(manager):
    router = APIRouter(tags=["benchmarks"])

    @router.get("/benchmark-catalog")
    def catalogue(deployment_id: str):
        return manager.catalogue(deployment_id)

    @router.post("/benchmarks", status_code=202)
    def start(body: BenchmarkRequest, idempotency_key: str = Header(min_length=1, max_length=128)):
        return manager.start(body.model_dump(), idempotency_key)

    @router.get("/benchmarks")
    def listing(limit: int = Query(30, ge=1, le=50)):
        return manager.list(limit)

    @router.get("/benchmarks/{identity}")
    def detail(identity: str):
        return manager.get(identity)

    @router.post("/benchmarks/{identity}/cancel")
    def cancel(identity: str):
        return manager.cancel(identity)

    return router
