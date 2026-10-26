"""Workspace isolation, budgets and cleanup against a real (throwaway) Neo4j."""

from __future__ import annotations

import time

import pytest

from graphrag.data import Chunk
from graphrag.graph import PUBLIC, GraphStore
from graphrag.models import Entity, ExtractionResult, Relationship
from graphrag.retrieval import GraphRetriever

pytestmark = pytest.mark.integration

WS_A = "a" * 32
WS_B = "b" * 32


def chunk(doc: str, text: str, ws_title: str = "doc.txt") -> Chunk:
    return Chunk(
        chunk_id=f"{doc}::0", doc_id=doc, chunk_index=0, text=text, category="Uploaded", source=ws_title
    )


def extraction(*names: str, rel: tuple[str, str] | None = None) -> ExtractionResult:
    return ExtractionResult(
        entities=[Entity(name=n, type="Concept") for n in names],
        relationships=[Relationship(source=rel[0], target=rel[1], type="uses")] if rel else [],
    )


def put(store: GraphStore, ws: str | None, doc: str, text: str, *entities: str, rel=None) -> Chunk:
    c = chunk(doc, text)
    store.upsert_chunks([c], workspace=ws, title=f"{doc}.txt")
    store.upsert_extractions({c.chunk_id: extraction(*entities, rel=rel)}, workspace=ws)
    return c


def count(store: GraphStore, cypher: str, **params) -> int:
    with store._driver.session(database=store._s.neo4j_database) as s:
        return s.run(cypher, **params).single()["n"]


# ── search isolation ─────────────────────────────────────────────────────────
def test_private_chunks_are_invisible_to_other_workspaces_and_to_public_search(store) -> None:
    a = put(store, WS_A, "UP-a", "Alice private notes about quantum gardening.", "Quantum gardening")
    assert [h["chunk_id"] for h in store.vector_search(a.text, 5, scopes=[WS_A])] == ["UP-a::0"]
    assert store.vector_search(a.text, 5, scopes=[WS_B]) == []
    assert store.vector_search(a.text, 5, scopes=[PUBLIC]) == []
    assert store.vector_search(a.text, 5) == []  # default scope is public only


def test_public_and_private_results_are_merged_by_score(store) -> None:
    pub = put(store, None, "PUB-1", "Public text about graph databases and indexing.", "Graph databases")
    mine = put(
        store, WS_A, "UP-a", "My private text about graph databases and indexing today.", "Graph databases"
    )
    hits = store.vector_search(mine.text, 5, scopes=[PUBLIC, WS_A])
    assert [h["chunk_id"] for h in hits][0] == "UP-a::0"  # exact match ranks first
    assert {h["chunk_id"] for h in hits} == {pub.chunk_id, mine.chunk_id}
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-3)  # exact scan scores like the index


def test_private_search_finds_a_few_chunks_among_many_public_ones(store) -> None:
    for i in range(30):  # a crowded public corpus must not hide the visitor's own chunk
        put(store, None, f"PUB-{i}", f"Public filler passage number {i} about unrelated topics.")
    mine = put(store, WS_A, "UP-a", "The one private passage that matters to this visitor.")
    hits = store.vector_search(mine.text, 3, scopes=[WS_A])
    assert hits and hits[0]["chunk_id"] == "UP-a::0"


def test_invalid_workspace_ids_are_rejected_before_any_query(store) -> None:
    for bad in ["x", "A" * 32, "a" * 31, "a" * 32 + "'", "../" + "a" * 29]:
        with pytest.raises(ValueError):
            store.vector_search("q", 3, scopes=[bad])
        with pytest.raises(ValueError):
            store.list_documents(bad)


# ── entities and graph expansion stay inside a workspace ─────────────────────
def test_same_entity_name_in_two_workspaces_is_two_separate_nodes(store) -> None:
    put(store, WS_A, "UP-a", "Alice writes about encryption schemes.", "Encryption")
    put(store, WS_B, "UP-b", "Bob also writes about encryption schemes.", "Encryption")
    assert count(store, "MATCH (e:Entity) WHERE e.name = 'Encryption' RETURN count(e) AS n") == 2
    keys = {r["k"] for r in store._driver.session().run("MATCH (e:Entity) RETURN e.key AS k").data()}
    assert keys == {f"{WS_A}::encryption", f"{WS_B}::encryption"}


def test_graph_expansion_never_crosses_workspaces(store, monkeypatch) -> None:
    monkeypatch.setenv("TOP_K", "1")
    monkeypatch.setenv("EXPANSION_CHUNKS", "2")
    from graphrag.config import get_settings

    get_settings.cache_clear()
    a1 = put(store, WS_A, "UP-a1", "Alice document one about smart cards.", "Smart card")
    put(store, WS_A, "UP-a2", "Alice document two also about smart cards.", "Smart card")
    put(store, WS_B, "UP-b1", "Bob document about smart cards as well.", "Smart card")
    put(store, None, "PUB-1", "Public document about smart cards too.", "Smart card")

    for mode in ("co_mention", "graph_hop"):
        ctx = GraphRetriever(store, expansion_mode=mode).retrieve(a1.text, scopes=[WS_A])
        assert {c.doc_id for c in ctx.chunks} <= {"UP-a1", "UP-a2"}, mode

    both = GraphRetriever(store, expansion_mode="co_mention").retrieve(a1.text, scopes=[PUBLIC, WS_A])
    assert "UP-b1" not in {c.doc_id for c in both.chunks}


@pytest.mark.parametrize("mode", ["co_mention", "graph_hop"])
def test_scope_filter_holds_even_if_data_links_across_workspaces(store, monkeypatch, mode) -> None:
    """Entity keys are namespaced, so normal data cannot cross workspaces. The query-level scope
    filter is defence in depth: it must still hold if a bug or manual edit creates a bad link."""
    monkeypatch.setenv("TOP_K", "1")
    monkeypatch.setenv("EXPANSION_CHUNKS", "5")
    from graphrag.config import get_settings

    get_settings.cache_clear()
    a = put(
        store,
        WS_A,
        "UP-a",
        "Alice document about a scheme using a token.",
        "Scheme",
        "Token",
        rel=("Scheme", "Token"),
    )
    put(store, WS_B, "UP-b", "Bob unrelated document about weather patterns.", "Weather")
    with store._driver.session(database=store._s.neo4j_database) as s:  # simulate corrupt data
        s.run(
            """
            MATCH (c:Chunk {chunk_id: 'UP-b::0'}), (e:Entity) WHERE e.key STARTS WITH $prefix
            MERGE (c)-[:MENTIONS]->(e)
            """,
            prefix=f"{WS_A}::",
        ).consume()
    ctx = GraphRetriever(store, expansion_mode=mode).retrieve(a.text, scopes=[WS_A])
    assert "UP-b" not in {c.doc_id for c in ctx.chunks}


def test_graph_facts_do_not_leak_another_workspaces_relationships(store) -> None:
    a = put(
        store, WS_A, "UP-a", "Alice: the Scheme uses a Token.", "Scheme", "Token", rel=("Scheme", "Token")
    )
    put(store, WS_B, "UP-b", "Bob: the Scheme uses a Secret.", "Scheme", "Secret", rel=("Scheme", "Secret"))
    ctx = GraphRetriever(store).retrieve(a.text, scopes=[WS_A])
    assert {(f.source, f.target) for f in ctx.facts} == {("Scheme", "Token")}


# ── documents and deletion ───────────────────────────────────────────────────
def test_list_and_delete_a_document(store) -> None:
    put(store, WS_A, "UP-a", "Alice document about topic one.", "Topic one")
    put(store, WS_A, "UP-b", "Alice document about topic two.", "Topic two")
    docs = store.list_documents(WS_A)
    assert {d["doc_id"] for d in docs} == {"UP-a", "UP-b"} and all(d["chunks"] == 1 for d in docs)

    assert store.delete_document(WS_A, "UP-a") is True
    assert [d["doc_id"] for d in store.list_documents(WS_A)] == ["UP-b"]
    assert count(store, "MATCH (e:Entity {name: 'Topic one'}) RETURN count(e) AS n") == 0  # orphan removed
    assert count(store, "MATCH (e:Entity {name: 'Topic two'}) RETURN count(e) AS n") == 1


def test_delete_keeps_entities_still_used_by_another_document(store) -> None:
    put(store, WS_A, "UP-a", "Doc A mentions shared concept.", "Shared concept")
    put(store, WS_A, "UP-b", "Doc B mentions shared concept.", "Shared concept")
    store.delete_document(WS_A, "UP-a")
    assert count(store, "MATCH (e:Entity {name: 'Shared concept'}) RETURN count(e) AS n") == 1


def test_cannot_delete_another_workspaces_document(store) -> None:
    put(store, WS_A, "UP-a", "Alice's private document text here.", "Concept")
    assert store.delete_document(WS_B, "UP-a") is False
    assert count(store, "MATCH (d:Document {doc_id: 'UP-a'}) RETURN count(d) AS n") == 1


def test_delete_workspace_removes_only_that_workspace(store) -> None:
    put(store, WS_A, "UP-a", "Alice document text for deletion.", "A concept", rel=None)
    put(store, WS_B, "UP-b", "Bob document text to keep around.", "B concept")
    put(store, None, "PUB-1", "Public document text to keep around.", "Public concept")
    store.delete_workspace(WS_A)
    assert store.list_documents(WS_A) == []
    assert [d["doc_id"] for d in store.list_documents(WS_B)] == ["UP-b"]
    assert count(store, "MATCH (d:Document {doc_id: 'PUB-1'}) RETURN count(d) AS n") == 1
    assert count(store, f"MATCH (n) WHERE n.workspace = '{WS_A}' RETURN count(n) AS n") == 0


def test_public_chunks_are_unaffected_by_workspace_features(store) -> None:
    pub = put(store, None, "PUB-1", "Public text about retrieval augmented generation.", "RAG")
    assert store.vector_search(pub.text, 3)[0]["chunk_id"] == "PUB-1::0"
    assert count(store, "MATCH (c:Chunk) WHERE c.workspace IS NULL RETURN count(c) AS n") == 1


# ── budgets ──────────────────────────────────────────────────────────────────
def test_budget_reserve_release_and_limit(store) -> None:
    assert store.budget_remaining("chunks", 10) == 10
    assert store.reserve_budget("chunks", 6, 10) is True
    assert store.budget_remaining("chunks", 10) == 4
    assert store.reserve_budget("chunks", 5, 10) is False  # would exceed the limit
    assert store.budget_remaining("chunks", 10) == 4  # a refused reservation changes nothing
    assert store.reserve_budget("chunks", 4, 10) is True  # exactly the limit is allowed
    store.release_budget("chunks", 3)
    assert store.budget_remaining("chunks", 10) == 3


def test_budget_never_goes_negative_and_kinds_are_independent(store) -> None:
    store.reserve_budget("asks", 2, 5)
    store.release_budget("asks", 99)
    assert store.budget_remaining("asks", 5) == 5
    store.reserve_budget("chunks", 1, 5)
    assert store.budget_remaining("asks", 5) == 5


def test_budget_survives_a_new_store_instance(store) -> None:
    store.reserve_budget("chunks", 7, 10)
    other = GraphStore(embedder=store._embedder)
    try:
        assert other.budget_remaining("chunks", 10) == 3  # persisted in Neo4j, not in memory
    finally:
        other.close()


# ── cleanup ──────────────────────────────────────────────────────────────────
def test_stale_workspaces_are_found_by_age(store) -> None:
    put(store, WS_A, "UP-a", "Alice old document text content.", "Old")
    put(store, WS_B, "UP-b", "Bob fresh document text content.", "New")
    with store._driver.session(database=store._s.neo4j_database) as s:  # age Alice's document by 10 days
        s.run("MATCH (d:Document {doc_id: 'UP-a'}) SET d.created = timestamp() - 864000000").consume()
    assert store.stale_workspaces(7) == [WS_A]
    assert store.stale_workspaces(30) == []
    time.sleep(0.01)
