"""L1 versioned API; mutations are bounded paired jobs, not shell commands."""

from fastapi import APIRouter, Header
from pydantic import BaseModel, ConfigDict
from hqsb.core.system_execution import SystemPlan
from hqsb.console.schemas import RunRecord


class SystemRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deployment_id: str
    plan: SystemPlan


def system_router(service, jobs):
    router = APIRouter(tags=["hardware-system"])

    @router.get("/system/hardware")
    def hardware():
        return jobs.hardware()

    @router.get("/system/catalog")
    def catalog():
        return {
            "schema_version": "l1-v1",
            "default": SystemPlan().model_dump(),
            "controls": [
                "visible CUDA device",
                "CPU affinity",
                "PyTorch intra/inter-op threads",
                "NUMA future allocation binding",
                "NVML GPU power cap",
            ],
            "not_implemented": [
                "portable clock locking/Jetson nvpmodel changes",
                "NPU execution",
                "heterogeneous graph partitioning",
                "physical hardware replacement",
            ],
            "evidence": "warmup excluded; AB/BA fresh-process trials; board energy and GPU counters are device-wide",
        }

    @router.post("/system/jobs", response_model=RunRecord, status_code=202)
    def create(
        body: SystemRequest, idempotency_key: str = Header(min_length=1, max_length=128)
    ):
        return jobs.start(body.deployment_id, body.plan, idempotency_key)

    @router.get("/system/jobs")
    def listing():
        return service.store.list(60, kind="system")

    @router.get("/runs/{identity}/system")
    def receipt(identity: str):
        doc, manifest = jobs.read(identity)
        return {**doc, "publication": manifest["publication"]}

    return router
