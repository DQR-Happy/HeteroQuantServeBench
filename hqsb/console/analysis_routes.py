"""Analysis API composition; device work remains in isolated actors."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hqsb.console.captures import CaptureRepository, safe_directory, sha256_file
from hqsb.console.dataflow import build_dataflow
from hqsb.console.optimization import markdown_report, optimization_report
from hqsb.console.research import ResearchCatalog
from hqsb.console.resources import observation_capabilities, resource_snapshot
from hqsb.console.schemas import RunRecord
from hqsb.console.store import TERMINAL
from hqsb.observation.storage import EvidenceReader
from hqsb.observation.correlation import trace_links
from hqsb.observation.analysis import analyze as analyze_observations, compare as compare_observations


class QuantizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deployment_id: str
    expected_epoch: str
    bits: Literal[4, 8]
    group_size: int | None = Field(None)

    @model_validator(mode="after")
    def supported(self):
        if self.bits == 4:
            if self.group_size is None:
                self.group_size = 128
            if self.group_size not in (32, 64, 128):
                raise ValueError("W4 group_size must be 32, 64 or 128")
        elif self.group_size is not None:
            raise ValueError("W8 is per-channel; group_size must be null")
        return self


class ObservationComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_ids: list[str] = Field(min_length=2, max_length=2)
    intervention: Literal["attention_requested", "cleanup_policy"] = "attention_requested"


def _download(body: bytes | str, filename: str, media_type="application/json"):
    return Response(
        body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def analysis_router(settings, service) -> APIRouter:
    router = APIRouter(tags=["analysis"])
    captures = CaptureRepository(settings.data_dir / "captures")
    research = ResearchCatalog(settings.evidence_root)

    def finished(run_id):
        run = service.store.get(run_id)
        if run["state"] not in TERMINAL:
            raise HTTPException(409, "Capture is still running; await final cleanup")
        return run

    @router.get("/resources")
    def resources():
        return resource_snapshot(service)

    def observation_directory(run_id):
        finished(run_id)
        return safe_directory(settings.data_dir / "captures", run_id)

    @router.get("/runs/{run_id}/observations")
    def observations(run_id: str):
        directory = observation_directory(run_id)
        try:
            result = analyze_observations(directory)
            lifetimes = result.get("allocation_lifetimes")
            if lifetimes:
                lifetimes["total_items"] = len(lifetimes["items"])
                lifetimes["items"] = lifetimes["items"][:500]
                lifetimes["projection_limit"] = 500
            return result
        except FileNotFoundError as exc:
            if exc.filename == str(directory / "observations/manifest.json"):
                return {"availability": "not_collected", "metrics": [], "notice": "No sealed observation bundle; legacy/off/interrupted captures cannot be reconstructed."}
            raise HTTPException(409, "A sealed observation artifact is missing") from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(409, f"Observation integrity check failed: {exc}") from exc

    @router.get("/runs/{run_id}/observations/records")
    def observation_records(run_id: str, kind: str | None = Query(None, max_length=80),
                            after: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=1000)):
        try:
            rows, size, more = [], 0, False
            for row in EvidenceReader(observation_directory(run_id)).records():
                if row["seq"] > after and (kind is None or row["kind"] == kind):
                    if len(json.dumps(row)) > 256 * 1024:
                        row = {**row, "payload": None, "projection": "raw_record_exceeds_response_budget",
                               "raw_available_in_verified_artifact": True}
                    encoded_size = len(json.dumps(row).encode())
                    if len(rows) >= limit or size + encoded_size > 4 * 1024 * 1024:
                        more = True
                        break
                    rows.append(row)
                    size += encoded_size
            return {"items": rows, "next_after": rows[-1]["seq"] if more else None,
                    "response_budget_bytes": 4 * 1024 * 1024}
        except (ValueError, OSError) as exc:
            raise HTTPException(409, f"Observation evidence unavailable: {exc}") from exc

    @router.post("/observation-comparisons")
    def observation_comparison(payload: ObservationComparisonRequest):
        ids = payload.run_ids
        try:
            return compare_observations(*(analyze_observations(observation_directory(key)) for key in ids),
                                        intervention=payload.intervention)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(409, f"Comparison evidence unavailable: {exc}") from exc

    @router.get("/runs/{run_id}/observations/manifest")
    def observation_manifest(run_id: str):
        try:
            return EvidenceReader(observation_directory(run_id)).manifest()
        except (ValueError, OSError) as exc:
            raise HTTPException(409, f"Observation manifest unavailable: {exc}") from exc

    @router.get("/runs/{run_id}/observations/artifacts/{index}")
    def observation_artifact(run_id: str, index: int):
        try:
            directory = observation_directory(run_id)
            reader = EvidenceReader(directory)
            artifacts = reader.manifest()["artifacts"]
            if not 0 <= index < len(artifacts):
                raise HTTPException(404, "Unknown evidence artifact")
            # Validate paths and content before exposing a raw stream.
            for _ in reader.records():
                pass
            artifact = artifacts[index]
            return FileResponse(directory / artifact["path"], media_type="application/x-ndjson",
                                filename=f"{run_id}-{index}.jsonl",
                                headers={"X-Content-SHA256": artifact["sha256"]})
        except (ValueError, OSError) as exc:
            raise HTTPException(409, f"Observation artifact unavailable: {exc}") from exc

    @router.get("/runs/{run_id}/observations/links")
    def observation_links(run_id: str):
        finished(run_id)
        directory = observation_directory(run_id)
        try:
            analyze_observations(directory)
        except FileNotFoundError as exc:
            if exc.filename != str(directory / "observations/manifest.json"):
                raise HTTPException(409, "A sealed observation artifact is missing") from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(409, f"Observation integrity check failed: {exc}") from exc
        return captures.project(run_id, lambda events, summary: {"trace": summary, **trace_links(events)})

    @router.get("/capabilities")
    def capabilities():
        return observation_capabilities(service)

    @router.get("/deployments/{deployment_id}/memory-inventory")
    def memory_inventory(deployment_id: str):
        return service.memory_inventory(deployment_id)

    @router.get("/research")
    def research_overview():
        return research.overview()

    @router.get("/research/export")
    def research_export():
        return _download(
            json.dumps(research.overview(), ensure_ascii=False, indent=2),
            "hqsb-historical-research.json",
        )

    @router.get("/research/artifacts/{artifact_id}/download")
    def research_download(artifact_id: str):
        try:
            path, data, digest = research.artifact(artifact_id)
        except ValueError as exc:
            raise HTTPException(
                409, "Historical artifact changed or exceeds the read budget"
            ) from exc
        return Response(
            data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="research-{artifact_id}{path.suffix}"',
                "X-Content-SHA256": digest,
            },
        )

    @router.get("/runs/{run_id}/analysis")
    def analysis(run_id: str):
        run = service.store.get(run_id)
        return {
            "schema_version": 1,
            "run_id": run_id,
            "observation": run["metrics"].get("observation"),
            "trace": captures.summary(run_id)
            if run["state"] in TERMINAL
            else {
                "status": "pending",
                "events": 0,
                "limitations": ["采集未完成，等待清理与导出。"],
            },
            "resource_before": run["metrics"].get("resource_before"),
            "resource_after": run["metrics"].get("resource_after"),
            "limitations": ["诊断观测，尚未验证零开销或正式基准可比性。"],
        }

    @router.get("/runs/{run_id}/trace")
    def trace(run_id: str, format: Literal["chrome"] = "chrome"):
        finished(run_id)
        path = captures.trace_file(run_id)
        return FileResponse(
            path,
            media_type="application/json",
            filename=run_id + "-trace.json",
            headers={"X-Content-SHA256": sha256_file(path)},
        )

    @router.get("/runs/{run_id}/dataflow")
    def dataflow(run_id: str):
        run = finished(run_id)
        return captures.project(
            run_id, lambda events, summary: build_dataflow(run, events, summary)
        )

    @router.get("/runs/{run_id}/trace/events")
    def trace_events(
        run_id: str,
        category: str | None = None,
        search: str = Query("", max_length=200),
        start_ms: float = Query(0, ge=0, allow_inf_nan=False),
        end_ms: float | None = Query(None, ge=0, allow_inf_nan=False),
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ):
        finished(run_id)
        if end_ms is not None and end_ms < start_ms:
            raise HTTPException(422, "end_ms must not precede start_ms")
        return captures.query(
            run_id,
            category=category,
            search=search,
            start_ms=start_ms,
            end_ms=end_ms,
            offset=offset,
            limit=limit,
        )

    @router.get("/runs/{run_id}/optimization")
    def optimization(run_id: str, format: Literal["json", "markdown"] = "json"):
        run = finished(run_id)
        report = optimization_report(run, captures.summary(run_id))
        if format == "markdown":
            return _download(
                markdown_report(report), run_id + "-optimization.md", "text/markdown"
            )
        return report

    @router.get("/runs/{run_id}/bundle")
    def evidence_bundle(run_id: str):
        run = finished(run_id)
        report = optimization_report(run, captures.summary(run_id))
        contents = {
            "run.json": json.dumps(run, ensure_ascii=False, indent=2).encode(),
            "optimization.json": json.dumps(
                report, ensure_ascii=False, indent=2
            ).encode(),
            "report.md": markdown_report(report).encode(),
        }
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "trace": report["trace"],
            "raw_trace_included": False,
            "raw_trace_download": f"/api/console/v1/runs/{run_id}/trace"
            if report["trace"]["status"] != "not_collected"
            else None,
            "files": [
                {
                    "path": name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                for name, data in contents.items()
            ],
            "limitations": [
                "摘要包不复制大型原始 trace；如存在请单独下载并核对 hash。",
                "仅包含原来选择保存的输入；未保存输入无法仅靠 hash 精确重放。",
            ],
        }
        contents["manifest.json"] = json.dumps(
            manifest, ensure_ascii=False, indent=2
        ).encode()
        target = io.BytesIO()
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in contents.items():
                archive.writestr(name, data)
        return _download(target.getvalue(), run_id + "-evidence.zip", "application/zip")

    @router.post("/quantization/jobs", status_code=202, response_model=RunRecord)
    def quantize(
        body: QuantizationRequest,
        idempotency_key: str = Header(min_length=1, max_length=128),
    ):
        return service.submit(
            "quantize",
            body.deployment_id,
            body.model_dump(exclude={"deployment_id"}),
            idempotency_key,
        )

    @router.get("/quantization/artifacts")
    def artifacts():
        records = service.store.list(100, kind="quantize")["items"]
        return {
            "items": [
                {
                    "id": row["id"],
                    "run_id": row["id"],
                    "bits": row["config"].get("bits"),
                    "group_size": row["config"].get("group_size"),
                    "created_at": row["created_at"],
                    "status": row["state"],
                    "summary": row.get("metrics", {}).get("quantization", {}),
                    "execution_label": "storage_only",
                    "quality": "not_evaluated",
                    "native_deployment_available": False,
                }
                for row in records
            ]
        }

    @router.get("/quantization/artifacts/{artifact_id}/manifest")
    def artifact_manifest(artifact_id: str):
        run = finished(artifact_id)
        if run["kind"] != "quantize" or run["state"] != "completed":
            raise KeyError(artifact_id)
        directory = safe_directory(settings.data_dir / "artifacts", artifact_id)
        path = directory / "manifest.json"
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 8 * 1024**2:
            raise KeyError(artifact_id)
        return FileResponse(
            path,
            media_type="application/json",
            filename=artifact_id + "-manifest.json",
            headers={"X-Content-SHA256": sha256_file(path)},
        )

    return router
