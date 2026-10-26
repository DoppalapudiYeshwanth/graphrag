"""The Streamlit page, driven by Streamlit's own test harness with in-memory fakes."""

from __future__ import annotations

import logging
import re

import httpx
import pytest
from groq import AuthenticationError, RateLimitError
from streamlit.testing.v1 import AppTest

from graphrag.graph import PUBLIC

from .fakes import WS_A, WS_B, long_text


def _app() -> None:
    """Runs inside the Streamlit script. State lives in session_state so it survives reruns."""
    import streamlit as st

    from graphrag.streamlit_ui import render
    from tests.fakes import FakeExtractor, FakeRag, FakeStore

    render(
        st.session_state.setdefault("_store", FakeStore()),
        st.session_state.setdefault("_rag", FakeRag()),
        st.session_state.setdefault("_ex", FakeExtractor()),
    )


@pytest.fixture
def at(settings_env) -> AppTest:
    settings_env()
    app = AppTest.from_function(_app, default_timeout=30)
    app.run()
    assert not app.exception
    return app


def ask(at: AppTest, question: str, scope: str | None = None) -> AppTest:
    if scope:
        at.radio[0].set_value(scope)
    at.text_area[0].set_value(question)
    next(b for b in at.button if b.label == "Ask").click()
    return at.run()


def fake(at: AppTest, name: str):
    return at.session_state[name]


# ── workspace id ─────────────────────────────────────────────────────────────
def test_page_renders_and_assigns_a_private_workspace(at: AppTest) -> None:
    assert at.title[0].value == "Graph RAG"
    assert re.fullmatch(r"[a-f0-9]{32}", at.query_params["ws"])


def test_each_new_visitor_gets_a_different_workspace(settings_env) -> None:
    settings_env()
    ids = set()
    for _ in range(3):
        app = AppTest.from_function(_app, default_timeout=30).run()
        ids.add(app.query_params["ws"])
    assert len(ids) == 3


def test_existing_workspace_in_the_url_is_kept(settings_env) -> None:
    settings_env()
    app = AppTest.from_function(_app, default_timeout=30)
    app.query_params["ws"] = WS_A
    app.run()
    assert app.query_params["ws"] == WS_A


@pytest.mark.parametrize("bad", ["x", "../../etc/passwd", "A" * 32, "a" * 31, "a" * 32 + "'"])
def test_malformed_workspace_ids_are_replaced(settings_env, bad: str) -> None:
    settings_env()
    app = AppTest.from_function(_app, default_timeout=30)
    app.query_params["ws"] = bad
    app.run()
    assert re.fullmatch(r"[a-f0-9]{32}", app.query_params["ws"])
    assert app.query_params["ws"] != bad


# ── asking ───────────────────────────────────────────────────────────────────
def test_demo_scope_searches_only_the_public_corpus(at: AppTest) -> None:
    ask(at, "What is X?", "Demo corpus")
    assert fake(at, "_rag").calls[-1] == ("What is X?", [PUBLIC])
    assert any("the answer" in m.value for m in at.markdown)


def test_both_scope_adds_the_visitors_workspace(at: AppTest) -> None:
    ask(at, "What is X?", "Both")
    assert fake(at, "_rag").calls[-1][1] == [PUBLIC, at.query_params["ws"]]


def test_my_documents_scope_searches_only_the_visitors_workspace(at: AppTest) -> None:
    ws = at.query_params["ws"]
    fake(at, "_store").docs["UP-aaaaaaaaaaaa"] = {"workspace": ws, "title": "mine.txt", "chunks": 1}
    at.run()
    ask(at, "What is X?", "My documents")
    assert fake(at, "_rag").calls[-1][1] == [ws]  # never the public corpus, never another workspace


def test_my_documents_without_any_is_a_clear_error_and_no_call(at: AppTest) -> None:
    ask(at, "What is X?", "My documents")
    assert any("Upload a document first" in e.value for e in at.error)
    assert fake(at, "_rag").calls == []


def test_short_questions_are_rejected_without_spending_budget(at: AppTest) -> None:
    ask(at, "hi")
    assert at.error and fake(at, "_rag").calls == []
    assert fake(at, "_store").budget["asks"] == 1000


def test_daily_question_budget_is_enforced(settings_env) -> None:
    settings_env()
    app = AppTest.from_function(_app, default_timeout=30).run()
    app.session_state["_store"].budget["asks"] = 0
    ask(app, "What is X?")
    assert any("capacity is used up" in e.value for e in app.error)
    assert app.session_state["_rag"].calls == []


def test_a_failed_answer_refunds_the_budget_and_hides_details(settings_env) -> None:
    settings_env()
    app = AppTest.from_function(_app, default_timeout=30).run()
    app.session_state["_rag"].error = RuntimeError("secret internal detail")
    ask(app, "What is X?")
    assert app.session_state["_store"].budget["asks"] == 1000
    assert app.error and all("secret" not in e.value for e in app.error)


def test_per_minute_rate_limit(settings_env) -> None:
    settings_env(RATE_LIMIT_PER_MINUTE="2")
    app = AppTest.from_function(_app, default_timeout=30).run()
    for _ in range(3):
        ask(app, "What is X?")
    assert any("slow down" in e.value for e in app.error)
    assert len(app.session_state["_rag"].calls) == 2


def test_answers_accumulate_newest_first(at: AppTest) -> None:
    ask(at, "First question?")
    ask(at, "Second question?")
    history = at.session_state["history"]
    assert [q for q, _ in history] == ["Second question?", "First question?"]


# ── documents ────────────────────────────────────────────────────────────────
def test_uploading_a_document_analyses_and_lists_it(at: AppTest) -> None:
    at.sidebar.file_uploader[0].upload("notes.txt", long_text(40), "text/plain")
    at.run()
    at.sidebar.button[0].click()
    at.run()
    store = fake(at, "_store")
    assert [d["title"] for d in store.list_documents(at.query_params["ws"])] == ["notes.txt"]
    assert any("Ready" in s.value for s in at.sidebar.success)


def test_upload_errors_are_shown_not_raised(at: AppTest) -> None:
    at.sidebar.file_uploader[0].upload("tiny.txt", b"too short", "text/plain")
    at.run()
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception
    assert any("Not enough readable text" in e.value for e in at.sidebar.error)
    assert fake(at, "_store").docs == {}


def test_a_visitor_only_sees_and_deletes_their_own_documents(settings_env) -> None:
    settings_env()
    mine = AppTest.from_function(_app, default_timeout=30)
    mine.query_params["ws"] = WS_A
    mine.run()
    store = mine.session_state["_store"]
    store.docs["UP-aaaaaaaaaaaa"] = {"workspace": WS_A, "title": "mine.txt", "chunks": 2}
    store.docs["UP-bbbbbbbbbbbb"] = {"workspace": WS_B, "title": "theirs.txt", "chunks": 2}
    mine.run()

    shown = " ".join(m.value for m in mine.sidebar.markdown)
    assert "mine.txt" in shown and "theirs.txt" not in shown

    next(b for b in mine.sidebar.button if b.label == "Delete").click()
    mine.run()
    assert "UP-aaaaaaaaaaaa" not in store.docs and "UP-bbbbbbbbbbbb" in store.docs  # theirs untouched


def test_delete_all_removes_only_this_workspace(settings_env) -> None:
    settings_env()
    app = AppTest.from_function(_app, default_timeout=30)
    app.query_params["ws"] = WS_A
    app.run()
    store = app.session_state["_store"]
    store.docs["UP-aaaaaaaaaaaa"] = {"workspace": WS_A, "title": "mine.txt", "chunks": 1}
    store.docs["UP-bbbbbbbbbbbb"] = {"workspace": WS_B, "title": "theirs.txt", "chunks": 1}
    app.run()
    next(b for b in app.sidebar.button if b.label == "Delete all my documents").click()
    app.run()
    assert list(store.docs) == ["UP-bbbbbbbbbbbb"]


def test_privacy_notice_is_always_visible(at: AppTest) -> None:
    text = " ".join(w.value for w in at.sidebar.warning)
    assert "Groq" in text and "sensitive" in text


# ── failures are explained to the visitor and logged for the operator ────────
def _groq_error(cls):
    request = httpx.Request("POST", "https://example.test")
    return cls(
        "boom",
        response=httpx.Response(401 if cls is AuthenticationError else 429, request=request),
        body=None,
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_groq_error(AuthenticationError), "key is not working"),
        (_groq_error(RateLimitError), "free quota is used up"),
        (RuntimeError("secret internal detail"), "unavailable"),
    ],
)
def test_failures_get_an_accurate_message_and_are_logged(settings_env, caplog, error, expected) -> None:
    settings_env()
    app = AppTest.from_function(_app, default_timeout=30).run()
    app.session_state["_rag"].error = error
    with caplog.at_level(logging.ERROR, logger="graphrag.streamlit_ui"):
        ask(app, "What is X?")
    assert any(expected in e.value for e in app.error)
    assert all("secret internal detail" not in e.value for e in app.error)  # never shown to the visitor
    assert any("Answering failed" in r.message for r in caplog.records)  # always logged
    assert app.session_state["_store"].budget["asks"] == 1000  # and refunded
