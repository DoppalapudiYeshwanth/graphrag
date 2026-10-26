"""Streamlit Community Cloud entry point:  streamlit run streamlit_app.py

Secrets come from Streamlit's secrets store (or a local .streamlit/secrets.toml) and are copied into
environment variables, because the application settings are read from the environment.
"""

from __future__ import annotations

import contextlib
import os

import streamlit as st


def _secrets_to_environment() -> None:
    """Copy flat string secrets (GROQ_API_KEY, NEO4J_URI, ...) into os.environ.

    Real environment variables win (setdefault), so a local `.env` or shell export still overrides.
    With no secrets file at all this does nothing and the app falls back to `.env`.
    """
    with contextlib.suppress(Exception):
        for key, value in st.secrets.items():
            if isinstance(value, str | int | float | bool):
                os.environ.setdefault(str(key).upper(), str(value))


_secrets_to_environment()

from graphrag.graph import GraphStore  # noqa: E402  (must come after the environment is populated)
from graphrag.models import GraphExtractor  # noqa: E402
from graphrag.retrieval import GraphRAG  # noqa: E402
from graphrag.streamlit_ui import render  # noqa: E402


@st.cache_resource(show_spinner="Starting up (loading the embedding model)...")
def services() -> tuple[GraphStore, GraphRAG, GraphExtractor]:
    """Build the heavy objects once per server process, not once per page view."""
    store = GraphStore()
    store.__enter__()  # verifies the Neo4j connection
    store.setup_schema()  # idempotent: constraints, workspace indexes, vector index
    return store, GraphRAG(store), GraphExtractor()


store, rag, extractor = services()
render(store, rag, extractor)
