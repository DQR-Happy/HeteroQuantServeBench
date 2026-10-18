"""L8 experiments are separate from legacy single-request generation."""
from fastapi import APIRouter, Header, Query

from hqsb.core.contracts.scheduling import SchedulingExperiment, WorkItem


def scheduling_router(manager):
    router = APIRouter(tags=["scheduling"])

    @router.get("/deployments/{deployment_id}/scheduling-capabilities")
    def capabilities(deployment_id: str):
        return manager.capabilities(deployment_id)

    @router.post("/scheduling-benchmarks", status_code=202)
    def start(body: SchedulingExperiment, idempotency_key: str = Header(min_length=1, max_length=128)):
        return manager.start(body.model_dump(), idempotency_key)

    @router.get("/scheduling-benchmarks")
    def listing():
        return manager.list()

    @router.get("/scheduling-benchmarks/{identity}")
    def detail(identity: str):
        return manager.get(identity)

    @router.post("/scheduling-benchmarks/{identity}/cancel")
    def cancel(identity: str):
        return manager.cancel(identity)

    @router.post("/scheduling-benchmarks/{identity}/requests/{request_id}/cancel", status_code=202)
    def cancel_request(identity: str, request_id: str):
        return manager.cancel_request(identity, request_id)

    @router.post("/scheduling-benchmarks/{identity}/requests", status_code=202)
    def submit_request(identity: str, body: WorkItem):
        return manager.submit_request(identity, body.model_dump())

    @router.post("/scheduling-benchmarks/{identity}/replay", status_code=202)
    def replay(identity: str, idempotency_key: str = Header(min_length=1, max_length=128)):
        return manager.replay(identity, idempotency_key)

    @router.get("/scheduling-benchmarks/{identity}/records")
    def records(identity: str, kind: str | None = None, after: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200)):
        return manager.records(identity, kind, after, limit)

    return router
