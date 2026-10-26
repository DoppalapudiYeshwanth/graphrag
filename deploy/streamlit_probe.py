"""Deployment probe: can THIS host run Graph RAG?

Deploy this single file to Streamlit Community Cloud (or any host) BEFORE committing to it. It checks
the things that decide whether the real app can work, and prints no secret values:

  1. outbound TCP on 7687 to your Aura instance (the port Neo4j's driver uses; some hosts block it)
  2. outbound TCP on 443 (control: should always work)
  3. a real Neo4j driver connection and query
  4. whether the embedding model can load, how long it takes, and how much memory the process uses
  5. reachability of the Groq API host

Needs the secrets NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD (Streamlit: App settings > Secrets).
"""

from __future__ import annotations

import platform
import resource
import socket
import sys
import time
from urllib.parse import urlparse

import streamlit as st

st.set_page_config(page_title="Deployment probe")
st.title("Deployment probe")
st.caption("Each line below is a real check run from this host. Red lines are blockers.")


def run(label: str, check) -> bool:
    start = time.perf_counter()
    try:
        detail = check()
    except Exception as exc:
        st.error(f"{label}: FAILED ({type(exc).__name__}: {str(exc)[:160]})")
        return False
    st.success(f"{label}: {detail} ({time.perf_counter() - start:.1f}s)")
    return True


def tcp(host: str, port: int) -> str:
    with socket.create_connection((host, port), timeout=8):
        return f"connected to {host}:{port}"


def max_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


def neo4j_query(uri: str, user: str, password: str) -> str:
    from neo4j import GraphDatabase

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        value = driver.session().run("RETURN 1 AS ok").single()["ok"]
    return f"driver connected, query returned {value}"


def load_model() -> str:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vector = model.encode(["probe"], normalize_embeddings=True)[0]
    return f"loaded; vector has {len(vector)} dimensions; peak memory so far {max_rss_mb():.0f} MB"


st.write(f"Python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")

secrets = st.secrets
missing = [k for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD") if k not in secrets]
if missing:
    st.error(f"Missing secrets: {', '.join(missing)}. Add them in App settings > Secrets.")
    st.stop()

uri = secrets["NEO4J_URI"]
host = urlparse(uri).hostname or ""
st.write(f"Target host: `{host}` (scheme `{urlparse(uri).scheme}`)")

results = {
    "DNS lookup": run("DNS lookup", lambda: socket.gethostbyname(host)),
    "Port 443 (control)": run("Port 443 (control)", lambda: tcp(host, 443)),
    "Port 7687 (Neo4j driver)": run("Port 7687 (Neo4j driver)", lambda: tcp(host, 7687)),
    "Neo4j driver + query": run(
        "Neo4j driver + query",
        lambda: neo4j_query(uri, secrets["NEO4J_USERNAME"], secrets["NEO4J_PASSWORD"]),
    ),
    "Groq API host": run("Groq API host", lambda: tcp("api.groq.com", 443)),
    "Embedding model": run("Embedding model", load_model),
}

st.divider()
if all(results.values()):
    st.success("All checks passed: this host can run the full app.")
else:
    failed = [name for name, ok in results.items() if not ok]
    st.error("Blockers: " + ", ".join(failed))
    if "Port 7687 (Neo4j driver)" in failed and results["Port 443 (control)"]:
        st.warning("Port 7687 is blocked here. This host cannot talk to Aura with the Bolt driver.")
