"""FastAPI service exposing the Graph RAG pipeline.

Run locally:  uvicorn graphrag.api:app --port 8080
Endpoints:    GET /health (liveness + Neo4j reachability), POST /ask (protected by X-API-Key).
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from groq import APIConnectionError, APIStatusError, RateLimitError
from pydantic import BaseModel, Field

from graphrag.config import get_settings, setup_logging
from graphrag.graph import GraphStore
from graphrag.retrieval import GraphFact, GraphRAG

logger = logging.getLogger(__name__)

SOURCE_TEXT_LIMIT = 400


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


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
    """In-memory per-client limiter. State is per process, so it is per Cloud Run instance."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self._limit = limit
        self._window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True


def create_app(rag: Any | None = None, health_check: Callable[[], None] | None = None) -> FastAPI:
    """Build the app. Passing `rag` (e.g. a fake in tests) skips Neo4j and model start-up."""
    settings = get_settings()
    limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store: GraphStore | None = None
        if rag is None:
            setup_logging()
            store = GraphStore()  # loads the embedding model once, at start-up
            store.__enter__()  # verifies the Neo4j connection
            app.state.rag = GraphRAG(store)
            app.state.health_check = store.ping
        else:
            app.state.rag = rag
            app.state.health_check = health_check or (lambda: None)
        if settings.api_key is None:
            logger.warning("API_KEY is not set: /ask is open to anyone who can reach this service")
        try:
            yield
        finally:
            if store is not None:
                store.close()

    app = FastAPI(title="Graph RAG", version="1.0.0", lifespan=lifespan)

    def guard(request: Request, x_api_key: str | None = Header(default=None)) -> None:
        """Authenticate (when API_KEY is set) and rate-limit by key, else by client address."""
        expected = settings.api_key
        if expected is not None:
            if x_api_key is None or not secrets.compare_digest(
                x_api_key.encode(), expected.get_secret_value().encode()
            ):
                raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
        client = x_api_key or (request.client.host if request.client else "unknown")
        if not limiter.allow(client):
            raise HTTPException(status_code=429, detail="Too many requests; slow down")

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        try:
            request.app.state.health_check()
        except Exception as exc:  # any failure means "not ready"
            logger.error("Health check failed: %s", exc)
            raise HTTPException(status_code=503, detail="Neo4j unreachable") from exc
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(guard)])
    def ask(body: AskRequest, request: Request) -> AskResponse:
        question = body.question.strip()
        if len(question) > settings.max_question_chars:
            raise HTTPException(
                status_code=422, detail=f"Question longer than {settings.max_question_chars} characters"
            )
        try:
            result = request.app.state.rag.ask(question)
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

    return app


app = create_app()
