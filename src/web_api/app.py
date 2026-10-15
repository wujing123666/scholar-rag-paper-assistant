"""Single-process FastAPI boundary around the existing RAG service."""

import asyncio
import hashlib
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from filelock import FileLock, Timeout
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

from src.core.settings import REPO_ROOT
from src.paper_assistant.request_gate import RequestQueueFullError
from src.web_api.jobs import JobCapacityError, JobConflictError, Jobs
from src.web_api.registry import Registry
from src.web_api.schemas import ApiError, Health, Job, Paper, PaperList, QueryRequest
from src.web_api.settings import WebSettings


def create_app(settings: WebSettings | None = None, *, registry_factory: Any = Registry) -> FastAPI:
    settings = settings or WebSettings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.state_path.mkdir(parents=True, exist_ok=True)
        process_lock = FileLock(settings.state_path / "api-process.lock")
        try:
            process_lock.acquire(timeout=0)
        except Timeout:
            raise RuntimeError("Only one API process may own this state directory") from None
        registry = None
        jobs = None
        try:
            registry = await asyncio.to_thread(registry_factory, settings)
            app.state.registry = registry
            jobs = Jobs(
                settings.state_path / registry.library.dataset_id / "jobs.sqlite3", registry,
                max_rows=settings.job_limit, ttl=settings.job_ttl_seconds,
                answer_workers=settings.answer_workers, search_workers=settings.search_workers,
                queue_size=settings.queue_size, queue_wait=settings.queue_wait,
                search_queue_wait=settings.search_queue_wait,
            )
            app.state.jobs = jobs
            yield
        finally:
            try:
                if jobs is not None:
                    await asyncio.to_thread(jobs.close)
                if registry is not None:
                    await asyncio.to_thread(registry.close)
            finally:
                process_lock.release()

    app = FastAPI(title="ScholarRAG API", version="0.1.0", lifespan=lifespan)

    def error_response(code: str, message: str, status: int, request_id: str | None = None):
        return JSONResponse(status_code=status, content=ApiError(
            code=code, message=message, request_id=request_id or uuid.uuid4().hex).model_dump())

    @app.middleware("http")
    async def browser_boundary(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin not in settings.allowed_origins:
                return error_response("ORIGIN_REJECTED", "请求来源未获允许。", 403)
            if "application/json" not in request.headers.get("content-type", ""):
                return error_response("JSON_REQUIRED", "请使用JSON请求。", 415)
        session = request.cookies.get("scholarrag_session", "")
        fresh = re.fullmatch(r"[a-f0-9]{64}", session) is None
        if fresh:
            session = secrets.token_hex(32)
        request.state.owner = hashlib.sha256(session.encode()).hexdigest()
        response = await call_next(request)
        if fresh:
            response.set_cookie("scholarrag_session", session, httponly=True,
                                samesite="lax", max_age=604800, path="/api/v1")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, error: RequestValidationError):
        return error_response("INVALID_REQUEST", "请求格式不正确，请检查问题长度、论文范围和资料库。", 422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException):
        if error.detail == "RECEIPT_CAPACITY":
            return error_response("RECEIPT_CAPACITY", "任务回执数量已达上限，请等待旧回执过期或调整服务端保留设置。", 429)
        return error_response(str(error.detail) if isinstance(error.detail, str) else "HTTP_ERROR",
                              {404: "记录或原文不存在，可能已过期。", 409: "资料库或请求编号已变化，请刷新后重试。",
                               429: "请求容量已满，请稍后重试。", 503: "服务暂时不可用，请检查服务端状态。"}.get(error.status_code, "请求未能完成。"),
                              error.status_code)

    @app.exception_handler(Exception)
    async def service_error(request: Request, error: Exception):
        return error_response("SERVICE_ERROR", "服务暂时不可用，请检查服务端状态。", 503)

    @app.get("/api/v1/health", response_model=Health)
    def health(request: Request):
        library = request.app.state.registry.library
        return Health(ready=True, dataset_id=library.dataset_id, label=settings.label,
                      papers_count=len(library.papers), verification_mode=settings.rag.verify_claims,
                      session_id=request.state.owner[:24])

    @app.get("/api/v1/papers", response_model=PaperList)
    def papers(request: Request):
        library = request.app.state.registry.library
        return PaperList(dataset_id=library.dataset_id, papers=list(library.papers.values()))

    @app.get("/api/v1/papers/{paper_id}", response_model=Paper)
    def paper(paper_id: str, request: Request):
        try:
            return request.app.state.registry.library.papers[paper_id]
        except KeyError:
            raise HTTPException(404, "PAPER_NOT_FOUND") from None

    @app.get("/api/v1/papers/{paper_id}/files/{file_id}")
    def source(paper_id: str, file_id: str, request: Request):
        library = request.app.state.registry.library
        try:
            library.assert_unchanged()
            path = library.files[paper_id, file_id]
            if not path.resolve().is_relative_to(library.inbox):
                raise KeyError(file_id)
            return FileResponse(path, media_type="application/pdf", filename=path.name,
                                content_disposition_type="inline")
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "SOURCE_NOT_FOUND") from None
        except ValueError:
            raise HTTPException(409, "DATASET_CHANGED") from None

    def submit(kind: str, body: QueryRequest, request: Request, idem: str):
        library = request.app.state.registry.library
        if body.dataset_id != library.dataset_id:
            raise HTTPException(409, "DATASET_CHANGED")
        if set(body.paper_ids) - library.papers.keys():
            raise HTTPException(422, "UNKNOWN_PAPER")
        if re.fullmatch(r"[a-zA-Z0-9_-]{16,80}", idem) is None:
            raise HTTPException(422, "INVALID_IDEMPOTENCY_KEY")
        try:
            library.assert_unchanged()
            return request.app.state.jobs.submit(request.state.owner, idem, kind, body)
        except JobConflictError:
            raise HTTPException(409, "IDEMPOTENCY_CONFLICT") from None
        except JobCapacityError:
            raise HTTPException(429, "RECEIPT_CAPACITY") from None
        except RequestQueueFullError:
            raise HTTPException(429, "QUEUE_FULL") from None
        except (ValueError, FileNotFoundError):
            raise HTTPException(409, "DATASET_CHANGED") from None

    @app.post("/api/v1/answers", response_model=Job, status_code=202)
    def answers(body: QueryRequest, request: Request,
                idempotency_key: Annotated[str, Header()]):
        return submit("answer", body, request, idempotency_key)

    @app.post("/api/v1/paper-searches", response_model=Job, status_code=202)
    def searches(body: QueryRequest, request: Request,
                 idempotency_key: Annotated[str, Header()]):
        return submit("search", body, request, idempotency_key)

    @app.get("/api/v1/jobs/{job_id}", response_model=Job)
    def job(job_id: str, request: Request):
        try:
            return request.app.state.jobs.get(job_id, request.state.owner)
        except KeyError:
            raise HTTPException(404, "JOB_NOT_FOUND") from None

    dist = REPO_ROOT / "frontend/dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app
