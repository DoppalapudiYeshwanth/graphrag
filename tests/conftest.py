"""Shared fixtures. Tests never use real credentials and never touch the real graph."""

from __future__ import annotations

import hashlib
import os

# Dummy secrets set BEFORE graphrag is imported: environment variables beat `.env`,
# so a developer's real keys are never used by the test suite.
os.environ["GROQ_API_KEY"] = "test-key"
os.environ["NEO4J_PASSWORD"] = "test-password"
# Pin every setting the tests rely on, so results never depend on a developer's `.env`.
for _key, _value in {
    "GROQ_MODEL": "model-a",
    "GROQ_FALLBACK_MODELS": "",
    "ANSWER_MODEL": "",
    "CHUNK_SIZE": "800",
    "CHUNK_OVERLAP": "100",
    "TEST_SIZE": "0.2",
    "RANDOM_SEED": "42",
    "TOP_K": "5",
    "EXPANSION_CHUNKS": "2",
    "EMBEDDING_DIM": "384",
}.items():
    os.environ[_key] = _value

import numpy as np
import pytest
from neo4j import GraphDatabase

from graphrag.config import get_settings
from graphrag.graph import GraphStore


class FakeEmbedder:
    """Deterministic 384-d unit vectors derived from the text; no model download."""

    dim = 384

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            vec = np.random.default_rng(seed).normal(size=self.dim)
            out.append((vec / np.linalg.norm(vec)).tolist())
        return out


@pytest.fixture
def settings_env(monkeypatch):
    """Set environment overrides and rebuild the cached Settings; restored afterwards."""

    def apply(**env: str):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()
        return get_settings()

    yield apply
    monkeypatch.undo()
    get_settings.cache_clear()


# ── throwaway Neo4j fixtures (integration tests only) ───────────────────────
TEST_ENV = {
    "TOP_K": "1",  # one vector seed, so graph expansion has room to add chunks
    "EXPANSION_CHUNKS": "1",
}


@pytest.fixture(scope="module")
def target():
    uri = os.environ.get("TEST_NEO4J_URI")
    if not uri:
        pytest.skip("TEST_NEO4J_URI not set (use a throwaway Neo4j, never your real graph)")
    return {
        "NEO4J_URI": uri,
        "NEO4J_USERNAME": os.environ.get("TEST_NEO4J_USERNAME", "neo4j"),
        "NEO4J_PASSWORD": os.environ.get("TEST_NEO4J_PASSWORD", "testpassword"),
        "NEO4J_DATABASE": os.environ.get("TEST_NEO4J_DATABASE", "neo4j"),
    }


@pytest.fixture
def store(target, monkeypatch):
    """A clean GraphStore on the throwaway database, with settings pointed at it."""
    for key, value in {**target, **TEST_ENV}.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()

    driver = GraphDatabase.driver(
        target["NEO4J_URI"], auth=(target["NEO4J_USERNAME"], target["NEO4J_PASSWORD"])
    )
    try:
        with driver.session(database=target["NEO4J_DATABASE"]) as s:
            foreign = s.run(
                "MATCH (n) WHERE NOT n:Document AND NOT n:Chunk AND NOT n:Entity "
                "AND NOT n:TestMarker AND NOT n:Budget RETURN count(n) AS n"
            ).single()["n"]
            total = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
            marked = s.run("MATCH (m:TestMarker) RETURN count(m) AS n").single()["n"]
            if foreign or (total and not marked):
                pytest.fail(
                    "Refusing to run: the target database holds data that is not from this test "
                    "suite. Point TEST_NEO4J_URI at a throwaway instance."
                )
    finally:
        driver.close()

    gs = GraphStore(embedder=FakeEmbedder())
    gs.__enter__()
    gs.clear()
    with gs._driver.session(database=target["NEO4J_DATABASE"]) as s:
        s.run("CREATE (:TestMarker)").consume()
    gs.setup_schema()
    yield gs
    gs.clear()
    gs.close()
    get_settings.cache_clear()
