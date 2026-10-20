"""L10 jobs and leased participant protocol; no unvalidated worker dispatch."""

from typing import Literal
from fastapi import APIRouter, Header
from pydantic import BaseModel, ConfigDict, Field
from hqsb.core.application_plan import (
    ApplicationPlan,
    ApplicationTarget,
    ApplicationInput,
    ApplicationPolicy,
)
from hqsb.console.application_transport import LocalEndpoint
from hqsb.console.schemas import RunRecord
from hqsb.console.service import ServiceError


class ApplicationJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan: ApplicationPlan


class ApplicationLease(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner: str = Field(pattern=r"^l10:run_[a-f0-9]{32}$")
    target: ApplicationTarget


class ApplicationWorkerRequest(ApplicationLease):
    operation: Literal[
        "snapshot",
        "application_metadata",
        "application_prepare",
        "application_prefix_begin",
        "application_prefix_end",
    ]
    payload: dict = Field(default_factory=dict)


class ApplicationOperation(ApplicationLease):
    kind: Literal["load", "unload", "generate"]
    payload: dict = Field(default_factory=dict)


def application_router(service, jobs, inference_type):
    router = APIRouter(tags=["service-application"])

    @router.get("/application/catalog")
    def catalog():
        return jobs.catalog()

    @router.post("/application/jobs", response_model=RunRecord, status_code=202)
    def create(
        body: ApplicationJobRequest,
        idempotency_key: str = Header(min_length=1, max_length=128),
    ):
        return jobs.start(body.plan, idempotency_key)

    @router.post("/application/validate", response_model=ApplicationPlan)
    def validate(body: ApplicationJobRequest):
        return body.plan

    @router.get("/application/jobs")
    def listing():
        return service.store.list(60, kind="application")

    @router.get("/application/operations/{identity}/status")
    def operation_status(identity: str):
        return service.store.status(identity)

    @router.get("/application/targets")
    def targets():
        return {
            "nodes": [
                {"id": r["id"], "name": r["name"], "transport": r["transport"]}
                for r in jobs.registry.public("local")["items"]
            ],
            "deployments": [
                {"id": d["id"], "name": d["name"], "state": d["state"]}
                for d in service.states.values()
            ],
        }

    @router.get("/runs/{identity}/application")
    def receipt(identity: str):
        doc, manifest = jobs.read(identity)
        return {**doc, "publication": manifest["publication"]}

    @router.post("/application/leases/acquire")
    def acquire(body: ApplicationLease):
        endpoint = LocalEndpoint(service, body.target, body.owner)
        endpoint.acquire()
        return {"owner": body.owner, "node_id": jobs.registry.node_id}

    @router.post("/application/leases/release")
    def release(body: ApplicationLease):
        LocalEndpoint(service, body.target, body.owner).release()
        return {"released": True}

    @router.post("/application/worker")
    def worker(body: ApplicationWorkerRequest):
        endpoint = LocalEndpoint(service, body.target, body.owner)
        if body.operation == "snapshot":
            return endpoint.snapshot()
        payload = body.payload
        if body.operation == "application_prepare":
            if set(payload) != {"input", "policy"}:
                raise ServiceError("L10_PREPARE_PAYLOAD", "需要 input 和 policy。", 422)
            payload = {
                "input": ApplicationInput.model_validate(payload["input"]).model_dump(),
                "policy": ApplicationPolicy.model_validate(
                    payload["policy"]
                ).model_dump(),
            }
        elif body.operation in {"application_prefix_begin", "application_prefix_end"}:
            if payload != {"owner": body.owner}:
                raise ServiceError(
                    "L10_PREFIX_SCOPE_OWNER",
                    "前缀作用域需要与节点租约相同的 owner。",
                    422,
                )
        elif payload:
            raise ServiceError(
                "L10_EMPTY_PAYLOAD_REQUIRED", "此元数据操作不接受额外参数。", 422
            )
        return endpoint.worker(body.operation, payload)

    @router.post("/application/operations", response_model=RunRecord, status_code=202)
    def operation(
        body: ApplicationOperation,
        idempotency_key: str = Header(min_length=1, max_length=128),
    ):
        payload = body.payload
        if body.kind == "generate":
            parsed = inference_type.model_validate(
                {**payload, "deployment_id": body.target.deployment_id}
            )
            if (
                parsed.decoding_plan
                or parsed.graph_capture
                or parsed.runtime_plan
                or parsed.optimization
            ):
                raise ServiceError(
                    "L10_NATIVE_REQUEST_ONLY",
                    "L10 participant 只接受原生贪心推理与已验证 L6 前缀。",
                    422,
                )
            payload = parsed.model_dump(exclude={"deployment_id"}, exclude_none=True)
        return LocalEndpoint(service, body.target, body.owner).operation(
            body.kind, payload, idempotency_key
        )

    return router
