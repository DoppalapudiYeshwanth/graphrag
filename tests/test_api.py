"""API behaviour with a fake pipeline: no Neo4j, no model download, no LLM calls."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APIConnectionError, RateLimitError

from graphrag.api import SOURCE_TEXT_LIMIT, SlidingWindowLimiter, create_app
from graphrag.retrieval import Answer, GraphFact, RetrievalContext, RetrievedChunk


class FakeRag:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.questions: list[str] = []

    def ask(self, question: str) -> Answer:
        self.questions.append(question)
        if self.error:
            raise self.error
        ctx = RetrievalContext(
            chunks=[
                RetrievedChunk(chunk_id="D::0", doc_id="D", text="x" * 1000, score=0.9),
                RetrievedChunk(chunk_id="E::0", doc_id="E", text="y", via="graph"),
            ],
            facts=[GraphFact(source="A", relation="USES", target="B")],
        )
        return Answer(question=question, answer="the answer [D::0]", context=ctx)


def client_for(settings_env, rag=None, health_check=None, **env: str) -> TestClient:
    settings_env(**env)
    return TestClient(create_app(rag=rag or FakeRag(), health_check=health_check))


def _groq_error(cls):
    request = httpx.Request("POST", "https://example.test")
    if cls is APIConnectionError:
        return APIConnectionError(request=request)
    return RateLimitError("limit", response=httpx.Response(429, request=request), body=None)


def test_health_ok(settings_env) -> None:
    with client_for(settings_env) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_health_reports_database_down(settings_env) -> None:
    def down() -> None:
        raise RuntimeError("boom")

    with client_for(settings_env, health_check=down) as client:
        assert client.get("/health").status_code == 503


def test_ask_returns_answer_sources_and_facts(settings_env) -> None:
    with client_for(settings_env) as client:
        r = client.post("/ask", json={"question": "What is X?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "the answer [D::0]"
    assert [s["chunk_id"] for s in body["sources"]] == ["D::0", "E::0"]
    assert body["sources"][1]["via"] == "graph"
    assert body["facts"] == [
        {"source": "A", "relation": "USES", "target": "B", "description": "", "source_chunks": []}
    ]


def test_source_text_is_truncated(settings_env) -> None:
    with client_for(settings_env) as client:
        body = client.post("/ask", json={"question": "What is X?"}).json()
    assert len(body["sources"][0]["text"]) == SOURCE_TEXT_LIMIT


def test_question_is_stripped_before_use(settings_env) -> None:
    rag = FakeRag()
    with client_for(settings_env, rag=rag) as client:
        client.post("/ask", json={"question": "   padded question   "})
    assert rag.questions == ["padded question"]


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "hi"}, {"question": 5}])
def test_invalid_requests_are_rejected(settings_env, payload) -> None:
    rag = FakeRag()
    with client_for(settings_env, rag=rag) as client:
        assert client.post("/ask", json=payload).status_code == 422
    assert rag.questions == []


def test_overlong_question_is_rejected(settings_env) -> None:
    with client_for(settings_env, MAX_QUESTION_CHARS="50") as client:
        assert client.post("/ask", json={"question": "q" * 51}).status_code == 422


# ── auth ─────────────────────────────────────────────────────────────────────
def test_api_key_required_when_configured(settings_env) -> None:
    rag = FakeRag()
    with client_for(settings_env, rag=rag, API_KEY="s3cret") as client:
        assert client.post("/ask", json={"question": "What is X?"}).status_code == 401
        bad = client.post("/ask", json={"question": "What is X?"}, headers={"X-API-Key": "nope"})
        assert bad.status_code == 401
        good = client.post("/ask", json={"question": "What is X?"}, headers={"X-API-Key": "s3cret"})
        assert good.status_code == 200
    assert rag.questions == ["What is X?"]  # rejected requests never reached the pipeline


def test_health_needs_no_key(settings_env) -> None:
    with client_for(settings_env, API_KEY="s3cret") as client:
        assert client.get("/health").status_code == 200


# ── rate limiting ────────────────────────────────────────────────────────────
def test_rate_limit_per_client(settings_env) -> None:
    with client_for(settings_env, RATE_LIMIT_PER_MINUTE="2") as client:
        codes = [client.post("/ask", json={"question": "What is X?"}).status_code for _ in range(4)]
    assert codes == [200, 200, 429, 429]


def test_rate_limit_is_per_api_key(settings_env) -> None:
    with client_for(settings_env, RATE_LIMIT_PER_MINUTE="1") as client:
        a = client.post("/ask", json={"question": "What is X?"}, headers={"X-API-Key": "a"})
        b = client.post("/ask", json={"question": "What is X?"}, headers={"X-API-Key": "b"})
        a2 = client.post("/ask", json={"question": "What is X?"}, headers={"X-API-Key": "a"})
    assert (a.status_code, b.status_code, a2.status_code) == (200, 200, 429)


def test_limiter_window_expires() -> None:
    limiter = SlidingWindowLimiter(limit=1, window_s=0.05)
    assert limiter.allow("c") and not limiter.allow("c")
    import time

    time.sleep(0.06)
    assert limiter.allow("c")


# ── upstream failures map to clean status codes ──────────────────────────────
def test_upstream_quota_maps_to_429(settings_env) -> None:
    rag = FakeRag(error=_groq_error(RateLimitError))
    with client_for(settings_env, rag=rag) as client:
        assert client.post("/ask", json={"question": "What is X?"}).status_code == 429


def test_upstream_connection_error_maps_to_502(settings_env) -> None:
    rag = FakeRag(error=_groq_error(APIConnectionError))
    with client_for(settings_env, rag=rag) as client:
        assert client.post("/ask", json={"question": "What is X?"}).status_code == 502


def test_unexpected_error_hides_details(settings_env) -> None:
    rag = FakeRag(error=RuntimeError("secret internal detail"))
    with client_for(settings_env, rag=rag) as client:
        r = client.post("/ask", json={"question": "What is X?"})
    assert r.status_code == 500 and "secret" not in r.text
