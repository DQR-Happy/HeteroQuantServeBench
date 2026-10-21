"""Nsight endpoints accept bounded plans and serve owned artifacts."""
from fastapi import APIRouter, Header, Query
from typing import Literal
from fastapi.responses import FileResponse, JSONResponse

from hqsb.core.diagnostics import DiagnosticPlan, DiagnosticComparison
from hqsb.console.schemas import RunRecord


def diagnostic_router(jobs, service):
    router = APIRouter(tags=["diagnostics"])

    @router.get("/diagnostics/catalog")
    def catalog():
        return jobs.catalog()

    @router.get("/diagnostics/sources")
    def sources():
        return jobs.sources()

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
        result = compare_reports(*reports)
        if result["comparable"] and reports[0].get("panel", {}).get("mode") == "memory":
            from hqsb.observation.dmat_compare import compare_memory
            roots = []
            for identity, report in zip((body.baseline_run_id, body.candidate_run_id), reports):
                roots.append(jobs.artifact(identity, "panel.json").parent)
                for launch in report["panel"]["memory_launches"]:
                    jobs.artifact(identity, launch["trace"])
            result["dmat"] = compare_memory(*(r["panel"] for r in reports), roots=roots)
            if not result["dmat"]["comparable"]:
                result.update(comparable=False, mismatches=result["dmat"]["mismatches"])
        return JSONResponse(result)

    @router.get("/diagnostics/jobs/{identity}/memory/{launch_index}")
    def memory(identity: str, launch_index: int, clock: Literal["thread_entry_clock64", "GPU_globaltimer"] = "thread_entry_clock64",
               page_bytes: int = Query(65536, ge=4096, le=65536, json_schema_extra={"enum": [4096, 16384, 65536]}), time_bins: int = Query(160, ge=16, le=512),
               start: int | None = Query(None, ge=0, le=2**53), end: int | None = Query(None, ge=1, le=2**53)):
        return JSONResponse(jobs.memory_view(identity, launch_index, clock=clock, page_bytes=page_bytes,
                                time_bins=time_bins, start=start, end=end))

    @router.get("/diagnostics/jobs/{identity}/artifacts/{relative:path}")
    def download(identity: str, relative: str):
        path = jobs.artifact(identity, relative)
        return FileResponse(path, filename=path.name)

    return router
