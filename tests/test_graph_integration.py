"""Neo4j integration tests (LLM and embedding model mocked).

These WIPE the target database, so they only run against a throwaway instance named by
TEST_NEO4J_URI, and refuse to touch a database that holds non-test data.
"""

from __future__ import annotations

import os

import pytest
from neo4j import GraphDatabase

from graphrag.config import get_settings
from graphrag.data import Chunk
from graphrag.graph import GraphStore
from graphrag.models import Entity, ExtractionResult, Relationship
from graphrag.retrieval import GraphRetriever

from .conftest import FakeEmbedder

pytestmark = pytest.mark.integration

TEST_ENV = {
    "TOP_K": "1",  # one vector seed, so graph expansion has room to add chunks
    "EXPANSION_CHUNKS": "1",
}


def _chunk(doc: str, idx: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc}::{idx}", doc_id=doc, chunk_index=idx, text=text,
        category="Security", source="arXiv",
    )


CHUNKS = [
    _chunk("DOC-A", 0, "Scheme one authenticates users with biometrics and a smart card."),
    _chunk("DOC-B", 0, "Scheme two stores keys on a smart card for roaming services."),
    _chunk("DOC-C", 0, "A completely unrelated note about wireless channel allocation."),
]


def _extractions() -> dict[str, ExtractionResult]:
    smart = Entity(name="Smart card", type="Technology", description="tamper-resistant token")
    return {
        "DOC-A::0": ExtractionResult(
            entities=[Entity(name="Scheme one", type="Algorithm"), smart,
                      Entity(name="Biometrics", type="Concept")],
            relationships=[
                Relationship(source="Scheme one", target="Smart card", type="uses", description="a"),
                Relationship(source="Scheme one", target="Biometrics", type="based on"),
            ],
        ),
        "DOC-B::0": ExtractionResult(
            entities=[Entity(name="Scheme two", type="Algorithm"), smart],
            relationships=[Relationship(source="Scheme two", target="smart_card", type="uses")],
        ),
        "DOC-C::0": ExtractionResult(entities=[Entity(name="Channel allocation", type="Task")]),
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
            foreign = s.run("MATCH (n) WHERE NOT n:Document AND NOT n:Chunk AND NOT n:Entity "
                            "AND NOT n:TestMarker RETURN count(n) AS n").single()["n"]
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


def _count(store: GraphStore, cypher: str) -> int:
    with store._driver.session(database=get_settings().neo4j_database) as s:
        return s.run(cypher).single()["n"]


def test_schema_setup_is_idempotent(store: GraphStore) -> None:
    store.setup_schema()
    store.setup_schema()
    assert _count(store, "SHOW CONSTRAINTS YIELD name RETURN count(name) AS n") >= 3
    assert _count(store, "SHOW VECTOR INDEXES YIELD name RETURN count(name) AS n") == 1


def test_upsert_chunks_creates_documents_and_chunks(store: GraphStore) -> None:
    store.upsert_chunks(CHUNKS)
    stats = store.stats()
    assert stats["documents"] == 3 and stats["chunks"] == 3
    assert _count(store, "MATCH (:Document)-[r:HAS_CHUNK]->(:Chunk) RETURN count(r) AS n") == 3


def test_upserts_are_idempotent(store: GraphStore) -> None:
    for _ in range(2):
        store.upsert_chunks(CHUNKS)
        store.upsert_extractions(_extractions())
    stats = store.stats()
    assert stats["chunks"] == 3 and stats["documents"] == 3
    assert stats["mentions"] == 6  # 3 + 2 + 1 distinct (chunk, entity) pairs
    assert stats["entity_relationships"] == 3


def test_shared_entity_merges_across_documents(store: GraphStore) -> None:
    store.upsert_chunks(CHUNKS)
    store.upsert_extractions(_extractions())
    assert _count(store, "MATCH (e:Entity {key: 'smart card'}) RETURN count(e) AS n") == 1
    assert _count(
        store, "MATCH (:Chunk)-[:MENTIONS]->(:Entity {key: 'smart card'}) RETURN count(*) AS n"
    ) == 2


def test_edge_keeps_unique_source_chunks(store: GraphStore) -> None:
    store.upsert_chunks(CHUNKS)
    for _ in range(2):
        store.upsert_extractions(_extractions())
    with store._driver.session(database=get_settings().neo4j_database) as s:
        rec = s.run(
            "MATCH (:Entity {key:'scheme one'})-[r:USES]->(:Entity {key:'smart card'}) "
            "RETURN r.source_chunks AS chunks"
        ).single()
    assert rec["chunks"] == ["DOC-A::0"]


def test_vector_search_returns_exact_match_first(store: GraphStore) -> None:
    store.upsert_chunks(CHUNKS)
    hits = store.vector_search(CHUNKS[1].text, top_k=3)
    assert hits[0]["chunk_id"] == "DOC-B::0"
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-3)
    assert len(hits) == 3


def test_retriever_returns_graph_facts(store: GraphStore) -> None:
    store.upsert_chunks(CHUNKS)
    store.upsert_extractions(_extractions())
    ctx = GraphRetriever(store).retrieve(CHUNKS[0].text)
    rendered = {f.render() for f in ctx.facts}
    assert "(Scheme one) -[USES]-> (Smart card): a" in rendered


def test_co_mention_expansion_reaches_second_document(store: GraphStore) -> None:
    store.upsert_chunks(CHUNKS)
    store.upsert_extractions(_extractions())
    ctx = GraphRetriever(store, expansion_mode="co_mention").retrieve(CHUNKS[0].text)
    assert [c.doc_id for c in ctx.chunks if c.via == "vector"] == ["DOC-A"]
    assert [c.doc_id for c in ctx.chunks if c.via == "graph"] == ["DOC-B"]  # shares 'Smart card'


def test_graph_hop_expansion_never_repeats_seed_chunks(store: GraphStore) -> None:
    store.upsert_chunks(CHUNKS)
    store.upsert_extractions(_extractions())
    ctx = GraphRetriever(store, expansion_mode="graph_hop").retrieve(CHUNKS[0].text)
    ids = [c.chunk_id for c in ctx.chunks]
    assert len(ids) == len(set(ids))


def test_unrelated_chunk_is_not_expanded_into(store: GraphStore) -> None:
    store.upsert_chunks(CHUNKS)
    store.upsert_extractions(_extractions())
    ctx = GraphRetriever(store, expansion_mode="co_mention").retrieve(CHUNKS[0].text)
    assert "DOC-C" not in {c.doc_id for c in ctx.chunks}
