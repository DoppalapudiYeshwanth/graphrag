"""FastAPI service: a programmatic API (/ask) and the public web app (/ and /api/*).

Run locally:  uvicorn graphrag.api:app --port 8000
Web app:      GET /            (uploads, questions)       cookie-scoped private workspaces
API:          POST /ask        (shared demo corpus)       protected by X-API-Key when set
              GET /health      liveness + Neo4j reachability
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from groq import APIConnectionError, APIStatusError, RateLimitError
from pydantic import BaseModel, Field

from graphrag.config import get_settings, setup_logging
from graphrag.graph import PUBLIC, GraphStore
from graphrag.ingest import ingest_stream
from graphrag.models import GraphExtractor
from graphrag.retrieval import GraphFact, GraphRAG

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
SOURCE_TEXT_LIMIT = 400
COOKIE = "ws"
WORKSPACE_RE = re.compile(r"^[a-f0-9]{32}$")
DOC_ID_RE = re.compile(r"^UP-[a-f0-9]{12}$")
CSP = "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class WebAskRequest(AskRequest):
    scope: Literal["mine", "demo", "both"] = "both"


class SourceOut(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float | None = None
    via: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    facts: list[GraphFact]


class SlidingWindowLimiter:
    """In-memory per-key limiter. State is per process, so it is per Cloud Run instance."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self._limit = limit
        self._window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _is_https(request: Request) -> bool:
    return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"


def create_app(
    rag: Any | None = None,
    health_check: Callable[[], None] | None = None,
    store: Any | None = None,
    extractor: Any | None = None,
) -> FastAPI:
    """Build the app. Passing `rag` (and optionally `store`, `extractor`) skips Neo4j and
    model start-up, which is how the tests run."""
    settings = get_settings()
    api_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)
    ask_ws_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)
    ask_ip_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute * 3)
    upload_ws_limiter = SlidingWindowLimiter(settings.upload_limit_per_hour, window_s=3600)
    upload_ip_limiter = SlidingWindowLimiter(settings.upload_limit_per_hour * 2, window_s=3600)
    ingest_slots = threading.BoundedSemaphore(settings.max_concurrent_ingests)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned: GraphStore | None = None
        if rag is None:
            setup_logging()
            owned = GraphStore()  # loads the embedding model once, at start-up
            owned.__enter__()  # verifies the Neo4j connection
            owned.setup_schema()  # idempotent: constraints, workspace indexes, vector index
            app.state.store = owned
            app.state.extractor = GraphExtractor()
            app.state.rag = GraphRAG(owned)
            app.state.health_check = owned.ping
        else:
            app.state.store = store
            app.state.extractor = extractor
            app.state.rag = rag
            app.state.health_check = health_check or (lambda: None)
        if settings.api_key is None:
            logger.warning("API_KEY is not set: /ask is open to anyone who can reach this service")
        try:
            yield
        finally:
            if owned is not None:
                owned.close()

    docs = None if settings.docs_enabled else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title="Graph RAG", version="2.0.0", lifespan=lifespan, **(docs or {}))

    # ── cross-cutting: security headers and request size ─────────────────────
    @app.middleware("http")
    async def harden(request: Request, call_next):
        if request.url.path == "/api/upload":
            declared = request.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes + 50_000:
                return JSONResponse({"detail": "File too large"}, status_code=413)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers.setdefault("Content-Security-Policy", CSP)
            response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    # ── dependencies ─────────────────────────────────────────────────────────
    def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
        """Authenticate (when API_KEY is set) and rate-limit programmatic clients."""
        expected = settings.api_key
        if expected is not None and (
            x_api_key is None
            or not secrets.compare_digest(x_api_key.encode(), expected.get_secret_value().encode())
        ):
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
        if not api_limiter.allow(x_api_key or _client_ip(request)):
            raise HTTPException(status_code=429, detail="Too many requests; slow down")

    def visitor(request: Request, response: Response) -> str:
        """The visitor's private workspace id, from an unguessable cookie (created on first use)."""
        ws = request.cookies.get(COOKIE)
        if ws is None or not WORKSPACE_RE.match(ws):
            ws = uuid.uuid4().hex
            response.set_cookie(
                COOKIE,
                ws,
                max_age=int(settings.workspace_ttl_days * 86400),
                httponly=True,
                samesite="lax",
                secure=_is_https(request),
                path="/",
            )
        return ws

    def need_store(request: Request) -> GraphStore:
        if request.app.state.store is None:
            raise HTTPException(status_code=503, detail="Uploads are not available")
        return request.app.state.store

    def limit(
        request: Request,
        ws: str,
        ws_limiter: SlidingWindowLimiter,
        ip_limiter: SlidingWindowLimiter,
        what: str,
    ) -> None:
        """Limit per visitor and per address, so rotating cookies does not evade the cap."""
        if not (ws_limiter.allow(f"ws:{ws}") and ip_limiter.allow(f"ip:{_client_ip(request)}")):
            raise HTTPException(status_code=429, detail=f"Too many {what}; please slow down")

    def answer(request: Request, question: str, scopes: list[str]) -> AskResponse:
        try:
            result = request.app.state.rag.ask(question, scopes)
        except RateLimitError as exc:
            logger.warning("Upstream model rate limit: %s", exc)
            raise HTTPException(status_code=429, detail="Model quota reached; try again later") from exc
        except (APIConnectionError, APIStatusError) as exc:
            logger.error("Upstream model error: %s", exc)
            raise HTTPException(status_code=502, detail="Upstream model unavailable") from exc
        except Exception as exc:
            logger.exception("Unhandled error answering question")
            raise HTTPException(status_code=500, detail="Internal error") from exc
        return AskResponse(
            answer=result.answer,
            sources=[
                SourceOut(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    text=c.text[:SOURCE_TEXT_LIMIT],
                    score=c.score,
                    via=c.via,
                )
                for c in result.context.chunks
            ],
            facts=result.context.facts,
        )

    # ── programmatic API (shared demo corpus) ────────────────────────────────
    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        try:
            request.app.state.health_check()
        except Exception as exc:  # any failure means "not ready"
            logger.error("Health check failed: %s", exc)
            raise HTTPException(status_code=503, detail="Neo4j unreachable") from exc
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
    def ask(body: AskRequest, request: Request) -> AskResponse:
        question = body.question.strip()
        if len(question) > settings.max_question_chars:
            raise HTTPException(
                status_code=422, detail=f"Question longer than {settings.max_question_chars} characters"
            )
        return answer(request, question, [PUBLIC])

    # ── web app ──────────────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request, response: Response) -> HTMLResponse:
        visitor(request, response)  # called for its side effect: sets the cookie on `response`
        page = HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))
        page.headers["Cache-Control"] = "no-store"
        for header, value in response.headers.items():
            if header.lower() == "set-cookie":
                page.headers.append("set-cookie", value)
        return page

    @app.get("/api/workspace")
    def workspace(request: Request, response: Response, ws: str = Depends(visitor)) -> dict[str, Any]:
        st = need_store(request)
        return {
            "documents": [
                {"doc_id": d["doc_id"], "title": d["title"], "chunks": d["chunks"]}
                for d in st.list_documents(ws)
            ],
            "limits": {
                "max_documents": settings.max_docs_per_workspace,
                "max_chars": settings.max_doc_chars,
                "max_mb": settings.max_upload_bytes // 1_000_000,
                "ttl_days": settings.workspace_ttl_days,
            },
            "capacity": {
                "chunks_left": st.budget_remaining("chunks", settings.daily_chunk_budget),
                "questions_left": st.budget_remaining("asks", settings.daily_ask_budget),
            },
        }

    @app.post("/api/upload")
    async def upload(request: Request, file: UploadFile = File(...)):
        st = need_store(request)
        ws = request.cookies.get(COOKIE, "")
        if not WORKSPACE_RE.match(ws):
            raise HTTPException(status_code=400, detail="Open the home page first")
        limit(request, ws, upload_ws_limiter, upload_ip_limiter, "uploads")
        data = await file.read(settings.max_upload_bytes + 1)
        filename = file.filename or "document"
        if not ingest_slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="Busy processing other uploads; retry shortly")

        def events() -> Iterator[bytes]:
            try:
                for event in ingest_stream(st, request.app.state.extractor, ws, filename, data):
                    yield (json.dumps(event) + "\n").encode()
            finally:
                ingest_slots.release()

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: str, request: Request, ws: str = Depends(visitor)) -> dict[str, bool]:
        if not DOC_ID_RE.match(doc_id):
            raise HTTPException(status_code=404, detail="Document not found")
        if not need_store(request).delete_document(ws, doc_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": True}

    @app.delete("/api/workspace")
    def delete_workspace(request: Request, ws: str = Depends(visitor)) -> dict[str, bool]:
        need_store(request).delete_workspace(ws)
        return {"deleted": True}

    @app.post("/api/ask", response_model=AskResponse)
    def web_ask(
        body: WebAskRequest, request: Request, response: Response, ws: str = Depends(visitor)
    ) -> AskResponse:
        st = need_store(request)
        question = body.question.strip()
        if len(question) > settings.max_question_chars:
            raise HTTPException(
                status_code=422, detail=f"Question longer than {settings.max_question_chars} characters"
            )
        limit(request, ws, ask_ws_limiter, ask_ip_limiter, "questions")
        if body.scope == "mine" and not st.list_documents(ws):
            raise HTTPException(status_code=400, detail="Upload a document first, or choose the demo corpus")
        scopes = {"mine": [ws], "demo": [PUBLIC], "both": [PUBLIC, ws]}[body.scope]
        if not st.reserve_budget("asks", 1, settings.daily_ask_budget):
            raise HTTPException(status_code=429, detail="Today's shared question capacity is used up")
        try:
            return answer(request, question, scopes)
        except HTTPException:
            st.release_budget("asks", 1)  # a failed answer does not use up shared capacity
            raise

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
