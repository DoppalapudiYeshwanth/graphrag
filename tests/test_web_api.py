"""Web app routes with in-memory fakes: cookies, uploads, scoping, limits, isolation, headers."""

from __future__ import annotations

import json
import threading

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import RateLimitError

from graphrag.api import COOKIE, create_app
from graphrag.graph import PUBLIC

from .fakes import FakeExtractor, FakeRag, FakeStore, long_text


def make_client(settings_env, store=None, rag=None, extractor=None, **env: str):
    settings_env(**env)
    store = store or FakeStore()
    rag = rag or FakeRag()
    app = create_app(rag=rag, store=store, extractor=extractor or FakeExtractor())
    return TestClient(app), store, rag


def upload(client: TestClient, name="paper.txt", data=None):
    r = client.post("/api/upload", files={"file": (name, data if data is not None else long_text(40))})
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    return r, events


# ── home page, cookie, headers ───────────────────────────────────────────────
def test_home_page_sets_private_cookie_and_security_headers(settings_env) -> None:
    client, *_ = make_client(settings_env)
    with client:
        r = client.get("/")
    assert r.status_code == 200 and "Graph RAG" in r.text
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{COOKIE}=") and "HttpOnly" in cookie and "SameSite=lax" in cookie
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY" and r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "no-store"


def test_cookie_is_secure_behind_https_proxy(settings_env) -> None:
    client, *_ = make_client(settings_env)
    with client:
        r = client.get("/", headers={"x-forwarded-proto": "https"})
    assert "Secure" in r.headers["set-cookie"]


def test_returning_visitor_keeps_the_same_workspace(settings_env) -> None:
    client, *_ = make_client(settings_env)
    with client:
        first = client.get("/").cookies[COOKIE]
        assert "set-cookie" not in client.get("/").headers
        assert client.cookies[COOKIE] == first


def test_malformed_cookie_is_replaced(settings_env) -> None:
    client, *_ = make_client(settings_env)
    with client:
        r = client.get("/", cookies={COOKIE: "../../etc/passwd"})
    assert r.cookies[COOKIE] != "../../etc/passwd" and len(r.cookies[COOKIE]) == 32


def test_static_assets_are_served_with_csp(settings_env) -> None:
    client, *_ = make_client(settings_env)
    with client:
        js = client.get("/static/app.js")
    assert js.status_code == 200 and "content-security-policy" in js.headers


def test_api_docs_page_is_not_given_the_strict_csp(settings_env) -> None:
    client, *_ = make_client(settings_env)
    with client:
        assert "content-security-policy" not in client.get("/docs").headers  # swagger needs a CDN


def test_docs_can_be_disabled_for_public_runs(settings_env) -> None:
    client, *_ = make_client(settings_env, DOCS_ENABLED="false")
    with client:
        assert [client.get(p).status_code for p in ("/docs", "/redoc", "/openapi.json")] == [404] * 3
        assert client.get("/health").status_code == 200  # the app itself is unaffected


# ── workspace listing ────────────────────────────────────────────────────────
def test_workspace_starts_empty_and_reports_limits_and_capacity(settings_env) -> None:
    client, *_ = make_client(settings_env, MAX_DOCS_PER_WORKSPACE="3")
    with client:
        body = client.get("/api/workspace").json()
    assert body["documents"] == []
    assert body["limits"]["max_documents"] == 3
    assert body["capacity"] == {"chunks_left": 1000, "questions_left": 1000}


# ── upload ───────────────────────────────────────────────────────────────────
def test_upload_streams_progress_and_lists_the_document(settings_env) -> None:
    client, *_ = make_client(settings_env)
    with client:
        client.get("/")
        r, events = upload(client, "my notes.txt")
        docs = client.get("/api/workspace").json()["documents"]
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert events[0]["event"] == "start" and events[-1]["event"] == "done"
    assert [d["title"] for d in docs] == ["my notes.txt"]


def test_upload_requires_the_cookie_from_the_home_page(settings_env) -> None:
    client, *_ = make_client(settings_env)
    with client:
        r = client.post("/api/upload", files={"file": ("a.txt", long_text())})
    assert r.status_code == 400


def test_upload_problems_are_reported_in_the_stream(settings_env) -> None:
    client, *_ = make_client(settings_env)
    with client:
        client.get("/")
        _, events = upload(client, "virus.exe")
    assert events[-1]["event"] == "error" and "Unsupported" in events[-1]["detail"]


def test_oversized_upload_is_rejected_early_by_content_length(settings_env) -> None:
    client, *_ = make_client(settings_env, MAX_UPLOAD_BYTES="1000")
    with client:
        client.get("/")
        r = client.post("/api/upload", files={"file": ("a.txt", b"x" * 100_000)})
    assert r.status_code == 413


def test_upload_rate_limit(settings_env) -> None:
    client, *_ = make_client(settings_env, UPLOAD_LIMIT_PER_HOUR="2")
    with client:
        client.get("/")
        codes = [client.post("/api/upload", files={"file": ("a.txt", long_text())}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


class BlockingExtractor(FakeExtractor):
    """Holds the first extraction until released, so a second upload overlaps with it."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def extract(self, chunk):
        self.started.set()
        assert self.release.wait(timeout=10)
        return super().extract(chunk)


def test_concurrent_ingestion_slots_are_limited(settings_env) -> None:
    extractor = BlockingExtractor()
    client, *_ = make_client(settings_env, extractor=extractor, MAX_CONCURRENT_INGESTS="1")
    result: dict = {}
    with client:
        client.get("/")
        first = threading.Thread(
            target=lambda: result.setdefault(
                "first", client.post("/api/upload", files={"file": ("a.txt", long_text())})
            )
        )
        first.start()
        assert extractor.started.wait(timeout=10)  # the first upload now holds the only slot

        second = client.post("/api/upload", files={"file": ("b.txt", long_text())})
        assert second.status_code == 503 and "Busy" in second.json()["detail"]

        extractor.release.set()
        first.join(timeout=20)
        assert result["first"].status_code == 200
        # The slot is free again once the first upload finishes.
        assert client.post("/api/upload", files={"file": ("c.txt", long_text())}).status_code == 200


# ── isolation between visitors ───────────────────────────────────────────────
def test_visitors_cannot_see_or_delete_each_others_documents(settings_env) -> None:
    store = FakeStore()
    a, _, _ = make_client(settings_env, store=store)
    b, _, _ = make_client(settings_env, store=store)
    with a, b:
        a.get("/"), b.get("/")
        _, events = upload(a)
        doc_id = events[0]["doc_id"]

        assert b.get("/api/workspace").json()["documents"] == []
        assert b.delete(f"/api/documents/{doc_id}").status_code == 404
        assert len(a.get("/api/workspace").json()["documents"]) == 1
        assert a.delete(f"/api/documents/{doc_id}").status_code == 200


@pytest.mark.parametrize("bad", ["x", "UP-zzzz", "UP-123", "../x", "UP-" + "g" * 12])
def test_malformed_document_ids_are_404(settings_env, bad) -> None:
    client, *_ = make_client(settings_env)
    with client:
        client.get("/")
        assert client.delete(f"/api/documents/{bad}").status_code == 404


def test_delete_workspace_removes_everything(settings_env) -> None:
    client, *_ = make_client(settings_env)
    with client:
        client.get("/")
        upload(client)
        assert client.delete("/api/workspace").json() == {"deleted": True}
        assert client.get("/api/workspace").json()["documents"] == []


# ── asking ───────────────────────────────────────────────────────────────────
def ws_of(client: TestClient) -> str:
    return client.cookies[COOKIE]


@pytest.mark.parametrize("scope", ["both", "demo"])
def test_scope_selects_the_corpora_searched(settings_env, scope) -> None:
    client, _, rag = make_client(settings_env)
    with client:
        client.get("/")
        assert client.post("/api/ask", json={"question": "What is X?", "scope": scope}).status_code == 200
        expected = {"both": [PUBLIC, ws_of(client)], "demo": [PUBLIC]}[scope]
    assert rag.calls[-1][1] == expected


def test_mine_scope_searches_only_the_visitors_workspace(settings_env) -> None:
    client, _, rag = make_client(settings_env)
    with client:
        client.get("/")
        upload(client)
        assert client.post("/api/ask", json={"question": "What is X?", "scope": "mine"}).status_code == 200
        assert rag.calls[-1][1] == [ws_of(client)]


def test_mine_scope_without_documents_is_a_clear_400(settings_env) -> None:
    client, _, rag = make_client(settings_env)
    with client:
        client.get("/")
        r = client.post("/api/ask", json={"question": "What is X?", "scope": "mine"})
    assert r.status_code == 400 and "Upload a document first" in r.json()["detail"]
    assert rag.calls == []


def test_client_cannot_choose_another_workspace(settings_env) -> None:
    client, _, rag = make_client(settings_env)
    with client:
        client.get("/")
        client.post("/api/ask", json={"question": "What is X?", "scope": "both", "workspace": "c" * 32})
    assert "c" * 32 not in rag.calls[-1][1]  # scopes come only from the cookie


def test_invalid_scope_is_rejected(settings_env) -> None:
    client, *_ = make_client(settings_env)
    with client:
        client.get("/")
        assert client.post("/api/ask", json={"question": "What is X?", "scope": "everyone"}).status_code == 422


def test_question_budget_is_enforced_and_shared(settings_env) -> None:
    store = FakeStore(ask_budget=1)
    client, *_ = make_client(settings_env, store=store)
    with client:
        client.get("/")
        codes = [client.post("/api/ask", json={"question": "What is X?"}).status_code for _ in range(2)]
    assert codes == [200, 429]


def test_failed_answer_does_not_consume_budget(settings_env) -> None:
    request = httpx.Request("POST", "https://example.test")
    error = RateLimitError("limit", response=httpx.Response(429, request=request), body=None)
    store = FakeStore(ask_budget=5)
    client, *_ = make_client(settings_env, store=store, rag=FakeRag(error=error))
    with client:
        client.get("/")
        assert client.post("/api/ask", json={"question": "What is X?"}).status_code == 429
    assert store.budget["asks"] == 5


def test_web_ask_rate_limit_per_visitor(settings_env) -> None:
    client, *_ = make_client(settings_env, RATE_LIMIT_PER_MINUTE="2")
    with client:
        client.get("/")
        codes = [client.post("/api/ask", json={"question": "What is X?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_rotating_cookies_does_not_evade_the_per_address_limit(settings_env) -> None:
    client, *_ = make_client(settings_env, RATE_LIMIT_PER_MINUTE="1")  # per-address limit = 3
    with client:
        codes = []
        for _ in range(5):
            client.cookies.clear()
            client.get("/")
            codes.append(client.post("/api/ask", json={"question": "What is X?"}).status_code)
    assert codes[:3] == [200, 200, 200] and 429 in codes[3:]


def test_demo_api_still_ignores_visitor_workspaces(settings_env) -> None:
    client, _, rag = make_client(settings_env)
    with client:
        client.get("/")
        client.post("/ask", json={"question": "What is X?"})
    assert rag.calls[-1][1] == [PUBLIC]


def test_web_routes_are_unavailable_without_a_store(settings_env) -> None:
    settings_env()
    client = TestClient(create_app(rag=FakeRag()))
    with client:
        client.get("/")
        assert client.get("/api/workspace").status_code == 503
