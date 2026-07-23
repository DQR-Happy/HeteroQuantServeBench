"""Authenticated Console API and same-origin static web application."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from contextlib import asynccontextmanager
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from hqsb.console.config import Settings
from hqsb.console.evidence import EvidenceCatalog
from hqsb.console.service import ConsoleService, ServiceError
from hqsb.console.schemas import RunPage, RunRecord
from hqsb.console.store import TERMINAL
from hqsb.console.telemetry import Telemetry

PREFIX = "/api/console/v1"


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32768)


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deployment_id: str
    expected_epoch: str
    messages: list[Message] = Field(min_length=1, max_length=32)
    max_output_tokens: int = Field(128, ge=1, le=4096)
    deadline_ms: int = Field(120000, ge=1000, le=600000)
    save_input: bool = False


class Login(BaseModel):
    token: str = Field(min_length=1, max_length=512)


class ComparisonRequest(BaseModel):
    run_ids: list[str] = Field(min_length=2, max_length=6)


def create_app(settings: Settings, access_token: str, *, service=None, monitor=True):
    svc = service or ConsoleService(settings)
    evidence = EvidenceCatalog(settings.evidence_root)
    telemetry = Telemetry()
    login_attempts: dict[str, list[float]] = {}

    @asynccontextmanager
    async def lifespan(_app):
        if monitor:
            telemetry.start()
        yield
        telemetry.close()
        await asyncio.to_thread(svc.close)

    app = FastAPI(
        title="HQSB Console",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=PREFIX + "/openapi.json",
    )
    app.state.service = svc

    def valid_session(cookie):
        try:
            expiry, signature = cookie.split(".", 1)
            expected = hmac.new(
                access_token.encode(), expiry.encode(), hashlib.sha256
            ).hexdigest()
            return int(expiry) > time.time() and hmac.compare_digest(
                signature, expected
            )
        except (ValueError, AttributeError):
            return False

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/"):
            if request.method == "POST":
                try:
                    size = int(request.headers.get("content-length", "-1"))
                except ValueError:
                    size = -1
                if size < 0:
                    return JSONResponse(
                        {
                            "error": {
                                "code": "LENGTH_REQUIRED",
                                "message": "Content-Length required",
                            }
                        },
                        411,
                    )
                if size > settings.max_body_bytes:
                    return JSONResponse(
                        {
                            "error": {
                                "code": "BODY_TOO_LARGE",
                                "message": "Request exceeds body limit",
                            }
                        },
                        413,
                    )
                origin = request.headers.get("origin")
                if origin and urlsplit(origin).netloc != request.headers.get("host"):
                    return JSONResponse(
                        {
                            "error": {
                                "code": "ORIGIN_REJECTED",
                                "message": "Same-origin requests required",
                            }
                        },
                        403,
                    )
                if (
                    not request.headers.get("authorization")
                    and request.headers.get("x-hqsb-client") != "console"
                ):
                    return JSONResponse(
                        {
                            "error": {
                                "code": "CSRF_REJECTED",
                                "message": "Console client header required",
                            }
                        },
                        403,
                    )
            if path != PREFIX + "/session/login":
                bearer = request.headers.get("authorization", "")
                authorized = (
                    hmac.compare_digest(bearer, "Bearer " + access_token)
                    if bearer
                    else False
                )
                if not authorized and not valid_session(
                    request.cookies.get("hqsb_session", "")
                ):
                    return JSONResponse(
                        {
                            "error": {
                                "code": "UNAUTHENTICATED",
                                "message": "请先输入工作台访问令牌",
                            }
                        },
                        401,
                    )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        )
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ServiceError)
    async def service_error(_request, exc):
        return JSONResponse(
            {"error": {"code": exc.code, "message": str(exc)}}, exc.status
        )

    @app.exception_handler(KeyError)
    async def missing(_request, _exc):
        return JSONResponse(
            {"error": {"code": "NOT_FOUND", "message": "Resource not found"}}, 404
        )

    @app.get("/healthz")
    def health():
        return {"status": "ok", "api_version": "1"}

    @app.post(PREFIX + "/session/login")
    def login(body: Login, request: Request):
        host = request.client.host if request.client else "unknown"
        now = time.time()
        recent = [t for t in login_attempts.get(host, []) if now - t < 60]
        if len(recent) >= 10:
            raise HTTPException(429, "Too many login attempts; wait one minute")
        login_attempts[host] = recent + [now]
        if not hmac.compare_digest(body.token, access_token):
            raise HTTPException(401, "访问令牌不正确")
        expiry = str(int(now) + 8 * 3600)
        signature = hmac.new(
            access_token.encode(), expiry.encode(), hashlib.sha256
        ).hexdigest()
        response = JSONResponse({"role": "operator", "expires_at": int(expiry)})
        response.set_cookie(
            "hqsb_session",
            expiry + "." + signature,
            httponly=True,
            secure=settings.secure_cookie,
            samesite="strict",
            max_age=8 * 3600,
        )
        return response

    @app.get(PREFIX + "/session")
    def session():
        return {
            "api_version": "1",
            "role": "operator",
            "version": "0.1.0",
            "mode": "live",
            "features": [
                "inference",
                "stream",
                "cancel",
                "evidence",
                "telemetry",
                "compare",
                "export",
            ],
            "limits": {
                "max_pending": settings.max_pending,
                "max_deadline_ms": settings.request_deadline_ms,
            },
        }

    @app.post(PREFIX + "/session/logout")
    def logout():
        response = JSONResponse({"ok": True})
        response.delete_cookie("hqsb_session")
        return response

    @app.get(PREFIX + "/overview")
    def overview():
        records = evidence.scan()
        counts = {}
        for record in records:
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        return {
            "deployments": svc.deployments(),
            "runs": svc.store.list(8),
            "evidence_count": len(records),
            "verdict_counts": counts,
            "pending": svc.pending,
            "telemetry": telemetry.snapshot(),
            "observed_at": time.time(),
        }

    @app.get(PREFIX + "/deployments")
    def deployments():
        return {"items": svc.deployments()}

    @app.post(
        PREFIX + "/deployments/{deployment_id}/{action}",
        status_code=202,
        response_model=RunRecord,
    )
    def deployment_action(
        deployment_id: str,
        action: Literal["load", "unload"],
        idempotency_key: str = Header(min_length=1, max_length=128),
    ):
        return svc.submit(action, deployment_id, {}, idempotency_key)

    @app.post(PREFIX + "/requests", status_code=202, response_model=RunRecord)
    def inference(
        body: InferenceRequest,
        idempotency_key: str = Header(min_length=1, max_length=128),
    ):
        if body.deadline_ms > settings.request_deadline_ms:
            raise ServiceError(
                "DEADLINE_LIMIT", "Requested deadline exceeds server policy", 422
            )
        if not any(m.role == "user" for m in body.messages):
            raise ServiceError(
                "MISSING_USER_MESSAGE", "At least one user message is required", 422
            )
        return svc.submit(
            "generate",
            body.deployment_id,
            body.model_dump(exclude={"deployment_id"}),
            idempotency_key,
        )

    @app.get(PREFIX + "/runs", response_model=RunPage)
    def runs(limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0)):
        return svc.store.list(limit, offset)

    @app.get(PREFIX + "/runs/{run_id}", response_model=RunRecord)
    @app.get(PREFIX + "/requests/{run_id}", response_model=RunRecord)
    def run(run_id: str):
        return svc.store.get(run_id)

    @app.post(PREFIX + "/requests/{run_id}/cancel", response_model=RunRecord)
    def cancel(run_id: str):
        return svc.cancel(run_id)

    @app.get(PREFIX + "/requests/{run_id}/events")
    async def events(
        run_id: str,
        request: Request,
        after: int = Query(0, ge=0),
        last_event_id: str | None = Header(None),
    ):
        current = svc.store.get(run_id)
        if last_event_id:
            try:
                resource, seq = last_event_id.rsplit(":", 1)
                if resource != run_id:
                    raise ValueError()
                after = max(after, int(seq))
            except ValueError:
                raise HTTPException(400, "Invalid Last-Event-ID")
        if after > current["seq"]:
            raise HTTPException(409, "Event cursor is ahead of this request")

        async def stream():
            cursor, last_heartbeat = after, time.monotonic()
            while not await request.is_disconnected():
                batch = svc.store.events(run_id, cursor)
                for event in batch:
                    cursor = event["seq"]
                    yield f"id: {run_id}:{cursor}\nevent: {event['kind']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                snapshot = svc.store.get(run_id)
                if snapshot["state"] in TERMINAL and cursor >= snapshot["seq"]:
                    return
                if time.monotonic() - last_heartbeat > 10:
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.monotonic()
                await asyncio.sleep(0.1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    @app.get(PREFIX + "/telemetry")
    def metrics():
        return telemetry.snapshot()

    @app.get(PREFIX + "/evidence")
    def catalog():
        return {"items": evidence.scan()}

    @app.get(PREFIX + "/catalog")
    def source_catalog():
        evidence.scan()
        return {
            "items": evidence.sources,
            "notice": "Source inventory is not proof of model-level activation.",
        }

    @app.get(PREFIX + "/historical/performance")
    def historical_performance():
        experiments = evidence.scan()
        rows = []
        for experiment in experiments:
            if experiment["experiment"] != "E05-02":
                continue
            for ref in experiment["files"]:
                if ref["name"] != "summary.json":
                    continue
                detail = evidence.detail(ref["id"])
                for method, workloads in (
                    detail["content"].get("performance", {}).items()
                ):
                    if not isinstance(workloads, dict):
                        continue
                    for workload, metrics in workloads.items():
                        if not isinstance(metrics, dict):
                            continue
                        values = {
                            k: v.get("median")
                            for k, v in metrics.items()
                            if isinstance(v, dict)
                        }
                        rows.append(
                            {
                                "method": method,
                                "workload": workload,
                                "metrics": values,
                                "evidence_id": ref["id"],
                                "source_sha256": detail["sha256"],
                                "verdict": experiment["status"],
                                "measurement_profile": "historical-E05-02-model-core",
                            }
                        )
        return {
            "items": rows,
            "notice": "历史 E05-02：W8/W4 为全权重 FP16 dequant 控制路径，不能声称原生低比特收益。",
        }

    @app.get(PREFIX + "/evidence/{evidence_id}")
    def evidence_detail(evidence_id: str):
        return evidence.detail(evidence_id)

    @app.get(PREFIX + "/evidence/{evidence_id}/download")
    def evidence_download(evidence_id: str):
        path, data, digest = evidence.file(evidence_id)
        return Response(
            data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=evidence-"
                + evidence_id
                + path.suffix,
                "X-Content-SHA256": digest,
            },
        )

    @app.get(PREFIX + "/runs/{run_id}/export")
    def export(run_id: str, format: Literal["json", "markdown"] = "json"):
        row = svc.store.get(run_id)
        if format == "markdown":
            # JSON fenced text prevents output from being interpreted as executable HTML.
            body = (
                f"# HQSB Console run {run_id}\n\nQuality: not evaluated. Interactive diagnostic, not a formal benchmark.\n\n```json\n"
                + json.dumps(row, ensure_ascii=False, indent=2).replace("```", "` ` `")
                + "\n```\n"
            )
        else:
            body = json.dumps(row, ensure_ascii=False, indent=2)
        return Response(
            body,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename={run_id}.{'md' if format == 'markdown' else 'json'}"
            },
        )

    @app.post(PREFIX + "/comparisons")
    def compare(body: ComparisonRequest):
        if len(set(body.run_ids)) != len(body.run_ids):
            raise ServiceError("DUPLICATE_RUN", "Choose distinct runs", 422)
        rows = [svc.store.get(key) for key in body.run_ids]
        reasons = []
        if any(
            row["kind"] != "generate" or row["state"] != "completed" for row in rows
        ):
            reasons.append("只允许完整完成的推理请求参与诊断对比")
        signatures = []
        for row in rows:
            cfg = row["config"]
            signatures.append(
                (
                    cfg.get("input_sha256"),
                    cfg.get("max_output_tokens"),
                    cfg["deployment"]["model"],
                    cfg["deployment"]["precision"],
                    row["metrics"].get("measurement_profile"),
                    row["metrics"].get("output_tokens"),
                )
            )
        if any(item != signatures[0] for item in signatures[1:]):
            reasons.append(
                "输入、模型、精度、输出长度或测量口径不同；不能计算统一加速比"
            )
        return {
            "items": rows,
            "comparable": not reasons,
            "reasons": reasons,
            "claim_level": "diagnostic_only",
            "quality": "not_evaluated",
            "notice": "交互单次运行仅作诊断；尚无质量基准、重复统计或硬件因果结论。",
        }

    @app.get("/{path:path}")
    def frontend(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "API route not found")
        root = settings.web_dist.resolve()
        file = (root / path).resolve()
        if not file.is_relative_to(root):
            raise HTTPException(404)
        if file.is_file():
            return FileResponse(file)
        if (root / "index.html").is_file():
            return FileResponse(root / "index.html")
        return JSONResponse(
            {"message": "Frontend has not been built. See web/console/README.md"}, 503
        )

    return app
