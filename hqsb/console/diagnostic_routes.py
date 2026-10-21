"""Nsight endpoints accept bounded plans and serve owned artifacts."""
from fastapi import APIRouter, Header
from fastapi.responses import FileResponse

from hqsb.core.diagnostics import DiagnosticPlan, DiagnosticComparison
from hqsb.console.schemas import RunRecord


def diagnostic_router(jobs, service):
    router = APIRouter(tags=["diagnostics"])

    @router.get("/diagnostics/catalog")
    def catalog():
        return jobs.catalog()

    @router.post("/diagnostics/jobs", response_model=RunRecord, status_code=202)
    def create(body: DiagnosticPlan, idempotency_key: str = Header(min_length=1, max_length=128)):
        return jobs.start(body, idempotency_key)

    @router.get("/diagnostics/jobs")
    def listing():
        return service.store.list(100, kind="diagnostic")

    @router.post("/diagnostics/comparisons")
    def compare(body: DiagnosticComparison):
        from hqsb.console.service import ServiceError
        from hqsb.observation.diagnostic_comparison import compare as compare_reports
        if body.baseline_run_id == body.candidate_run_id:
            raise ServiceError("DIAGNOSTIC_PAIR", "请选择两条不同的诊断记录。", 422)
        try:
            reports = [service.store.get(i)["metrics"]["diagnostic"] for i in (body.baseline_run_id, body.candidate_run_id)]
        except KeyError as exc:
            raise ServiceError("DIAGNOSTIC_PAIR", "请选择已有采集凭据的两条诊断记录。", 422) from exc
        return compare_reports(*reports)

    @router.get("/diagnostics/jobs/{identity}/artifacts/{relative:path}")
    def download(identity: str, relative: str):
        path = jobs.artifact(identity, relative)
        return FileResponse(path, filename=path.name)

    return router
