"""Streamlit user interface for Graph RAG.

All behaviour (retrieval, uploads, private workspaces, daily budgets) lives in the same modules the
FastAPI app uses. This file only draws the page, so `render` takes its services as arguments and
can be tested with in-memory fakes.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

import streamlit as st
from groq import AuthenticationError, PermissionDeniedError, RateLimitError

from graphrag.config import get_settings
from graphrag.graph import PUBLIC
from graphrag.ingest import ingest_stream

logger = logging.getLogger(__name__)

WORKSPACE_RE = re.compile(r"^[a-f0-9]{32}$")
SCOPES = {"Both": "both", "My documents": "mine", "Demo corpus": "demo"}
EXAMPLES = (
    "How is homomorphic encryption used to protect biometric data?",
    "What approaches reduce energy use in wireless networks?",
)


def workspace_id() -> str:
    """The visitor's private workspace id, kept in the page URL (?ws=...).

    Anyone holding the URL can open the workspace, exactly like a cookie-based id, so the page tells
    visitors to keep the link private.
    """
    ws = st.query_params.get("ws", "")
    if not WORKSPACE_RE.match(ws):
        ws = uuid.uuid4().hex
        st.query_params["ws"] = ws
    return ws


def _allow(kind: str, limit: int, window_s: float) -> bool:
    """Per-session sliding-window limit. The shared daily budgets are the real cap."""
    now = time.monotonic()
    hits = [t for t in st.session_state.setdefault(f"hits_{kind}", []) if now - t < window_s]
    if len(hits) >= limit:
        st.session_state[f"hits_{kind}"] = hits
        return False
    st.session_state[f"hits_{kind}"] = [*hits, now]
    return True


def _scopes_for(choice: str, ws: str) -> list[str]:
    return {"both": [PUBLIC, ws], "mine": [ws], "demo": [PUBLIC]}[SCOPES[choice]]


def _sidebar(store: Any, extractor: Any, ws: str) -> list[dict]:
    s = get_settings()
    nonce = st.session_state.setdefault("upload_nonce", 0)
    st.sidebar.header("Your documents")

    notice = st.session_state.pop("notice", None)
    if notice:
        (st.sidebar.success if notice[0] == "ok" else st.sidebar.error)(notice[1])

    uploaded = st.sidebar.file_uploader(
        f"Upload a .txt, .md or .pdf (up to {s.max_doc_chars:,} characters)",
        type=["txt", "md", "pdf"],
        key=f"upload_{nonce}",
    )
    if st.sidebar.button("Analyse document", disabled=uploaded is None, type="primary"):
        if not _allow("upload", s.upload_limit_per_hour, 3600):
            st.sidebar.error("Too many uploads this hour; please wait.")
        else:
            _analyse(store, extractor, ws, uploaded)

    docs = store.list_documents(ws)
    if not docs:
        st.sidebar.caption("No documents yet. You can still ask about the demo corpus.")
    for doc in docs:
        left, right = st.sidebar.columns([4, 1])
        left.markdown(f"**{doc['title']}**  \n{doc['chunks']} part(s)")
        if right.button("Delete", key=f"del_{doc['doc_id']}"):
            store.delete_document(ws, doc["doc_id"])
            st.rerun()
    if docs and st.sidebar.button("Delete all my documents"):
        store.delete_workspace(ws)
        st.rerun()

    st.sidebar.caption(
        f"Capacity left today (shared by everyone): "
        f"{store.budget_remaining('asks', s.daily_ask_budget)} questions, "
        f"{store.budget_remaining('chunks', s.daily_chunk_budget)} text chunks."
    )
    st.sidebar.warning(
        "Your documents are private to this link and deleted after "
        f"{s.workspace_ttl_days:g} idle days. Keep the link private. Their text is processed by a "
        "third-party AI service (Groq), so do not upload anything sensitive."
    )
    return docs


def _analyse(store: Any, extractor: Any, ws: str, uploaded: Any) -> None:
    bar = st.sidebar.progress(0.0, text=f"Reading {uploaded.name}...")
    outcome: tuple[str, str] = ("error", "Something went wrong. Please try again.")
    for event in ingest_stream(store, extractor, ws, uploaded.name, uploaded.getvalue()):
        kind = event["event"]
        if kind == "start":
            bar.progress(0.05, text=f"Analysing {event['title']} ({event['chunks']} parts)...")
        elif kind == "progress":
            bar.progress(
                0.05 + 0.9 * event["done"] / event["total"], text=f"Part {event['done']} of {event['total']}"
            )
        elif kind == "done":
            outcome = (
                "ok",
                f"Ready: {event['entities']} concepts and {event['relationships']} relationships found.",
            )
        elif kind == "error":
            outcome = ("error", event["detail"])
    st.session_state["notice"] = outcome
    st.session_state["upload_nonce"] += 1  # clears the uploader widget
    st.rerun()


def _ask(store: Any, rag: Any, ws: str, docs: list[dict], question: str, choice: str) -> None:
    s = get_settings()
    question = question.strip()
    if len(question) < 3:
        st.error("Please type a question.")
    elif len(question) > s.max_question_chars:
        st.error(f"Questions are limited to {s.max_question_chars} characters.")
    elif SCOPES[choice] == "mine" and not docs:
        st.error("Upload a document first, or choose the demo corpus.")
    elif not _allow("ask", s.rate_limit_per_minute, 60):
        st.error("Too many questions in a minute; please slow down.")
    elif not store.reserve_budget("asks", 1, s.daily_ask_budget):
        st.error("Today's shared question capacity is used up. Please try again tomorrow.")
    else:
        try:
            with st.spinner("Searching and reading the sources..."):
                answer = rag.ask(question, _scopes_for(choice, ws))
        except Exception as exc:
            store.release_budget("asks", 1)  # a failed answer does not use up shared capacity
            logger.exception("Answering failed")  # the operator needs the real cause in the logs
            st.error(_friendly_error(exc))
        else:
            st.session_state.setdefault("history", []).insert(0, (question, answer))


def _friendly_error(exc: Exception) -> str:
    """A visitor-safe message that still points at the right problem. Details go to the log only."""
    if isinstance(exc, AuthenticationError | PermissionDeniedError):
        return "The site's language-model key is not working. Please tell the site owner."
    if isinstance(exc, RateLimitError):
        return "The language model's free quota is used up for now. Please try again later."
    return "The language model is unavailable right now. Please try again later."


def _show(question: str, answer: Any) -> None:
    with st.container(border=True):
        st.caption(question)
        st.markdown(answer.answer)
        chunks = answer.context.chunks
        with st.expander(f"Sources ({len(chunks)})"):
            for i, chunk in enumerate(chunks, start=1):
                how = "found via graph" if chunk.via == "graph" else "similar"
                st.markdown(f"**{i}. {chunk.chunk_id}** ({how})")
                st.text(chunk.text[:400])
        if answer.context.facts:
            with st.expander(f"Knowledge-graph facts ({len(answer.context.facts)})"):
                st.text("\n".join(fact.render() for fact in answer.context.facts))


def render(store: Any, rag: Any, extractor: Any) -> None:
    """Draw the whole page. Must be called from a Streamlit script."""
    st.set_page_config(page_title="Graph RAG: ask your documents", layout="wide")
    ws = workspace_id()
    st.title("Graph RAG")
    st.write(
        "Upload a document and ask questions about it. Every answer cites its sources and shows "
        "the knowledge-graph facts it used."
    )
    docs = _sidebar(store, extractor, ws)

    choice = st.radio("Search in", list(SCOPES), horizontal=True)
    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            height=90,
            max_chars=get_settings().max_question_chars,
            placeholder=EXAMPLES[0],
        )
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted:
        _ask(store, rag, ws, docs, question, choice)

    for past_question, past_answer in st.session_state.get("history", []):
        _show(past_question, past_answer)
    st.caption(
        "Retrieval combines vector search with a knowledge graph. Answers can be wrong; check the cited passages."
    )
