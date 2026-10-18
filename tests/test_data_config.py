"""Data loading, stratified split, chunking, settings and prompt-context helpers."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from graphrag.config import Settings
from graphrag.data import chunk_documents, load_corpus, split_corpus
from graphrag.retrieval import GraphFact, RetrievalContext, RetrievedChunk, build_context_text


def write_csv(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def row(i: int, category: str = "A", text: str = "some text") -> dict:
    return {"doc_id": f"DOC-{i}", "text": text, "category": category, "source": "arXiv"}


# ── load_corpus ──────────────────────────────────────────────────────────────
def test_load_corpus_ok(tmp_path) -> None:
    path = tmp_path / "c.csv"
    write_csv(path, [row(1), row(2)])
    assert len(load_corpus(path)) == 2


def test_load_corpus_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope.csv")


def test_load_corpus_missing_column(tmp_path) -> None:
    path = tmp_path / "c.csv"
    pd.DataFrame([{"doc_id": "1", "text": "t"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_corpus(path)


def test_load_corpus_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "c.csv"
    write_csv(path, [row(1), row(1)])
    with pytest.raises(ValueError, match="unique"):
        load_corpus(path)


def test_load_corpus_drops_empty_text(tmp_path) -> None:
    path = tmp_path / "c.csv"
    write_csv(path, [row(1), row(2, text="   ")])
    assert list(load_corpus(path)["doc_id"]) == ["DOC-1"]


# ── split_corpus ─────────────────────────────────────────────────────────────
def test_split_is_stratified_and_reproducible() -> None:
    df = pd.DataFrame([row(i, "A") for i in range(50)] + [row(i + 50, "B") for i in range(50)])
    train, test = split_corpus(df)
    assert len(train) == 80 and len(test) == 20
    assert test["category"].value_counts().to_dict() == {"A": 10, "B": 10}
    assert set(train["doc_id"]).isdisjoint(test["doc_id"])
    train2, _ = split_corpus(df)
    assert list(train["doc_id"]) == list(train2["doc_id"])


def test_split_keeps_singleton_category_in_train() -> None:
    df = pd.DataFrame([row(i, "A") for i in range(20)] + [row(99, "RARE")])
    train, test = split_corpus(df)
    assert "RARE" in set(train["category"])
    assert "RARE" not in set(test["category"])


# ── chunk_documents ──────────────────────────────────────────────────────────
def test_chunks_carry_metadata_and_deterministic_ids() -> None:
    sentence = "Neural networks learn representations. "
    df = pd.DataFrame([row(1, "AI", text=sentence * 60)])  # ~2,300 chars, so several chunks
    chunks = chunk_documents(df)
    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks] == [f"DOC-1::{i}" for i in range(len(chunks))]
    assert all(c.doc_id == "DOC-1" and c.category == "AI" and c.source == "arXiv" for c in chunks)
    assert all(len(c.text) <= 800 for c in chunks)


def test_short_document_is_a_single_chunk() -> None:
    chunks = chunk_documents(pd.DataFrame([row(1, text="Short abstract.")]))
    assert [c.chunk_id for c in chunks] == ["DOC-1::0"]


# ── settings ─────────────────────────────────────────────────────────────────
def test_model_chain_is_ordered_and_deduplicated() -> None:
    s = Settings(groq_model="a", groq_fallback_models="b, a ,c,", neo4j_password="x", groq_api_key="y")
    assert s.model_chain == ["a", "b", "c"]


def test_model_chain_without_fallbacks() -> None:
    s = Settings(groq_model="a", neo4j_password="x", groq_api_key="y")
    assert s.model_chain == ["a"]


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=100, neo4j_password="x", groq_api_key="y")


def test_secrets_are_masked() -> None:
    s = Settings(neo4j_password="topsecret", groq_api_key="alsosecret")
    assert "topsecret" not in repr(s) and "alsosecret" not in repr(s)


# ── retrieval helpers (no database) ──────────────────────────────────────────
def test_graph_fact_render() -> None:
    fact = GraphFact(source="THRIVE", relation="USES", target="Encryption", description="uses it")
    assert fact.render() == "(THRIVE) -[USES]-> (Encryption): uses it"


def test_build_context_text_contains_ids_and_facts() -> None:
    ctx = RetrievalContext(
        chunks=[RetrievedChunk(chunk_id="D::0", doc_id="D", text="hello", score=0.9)],
        facts=[GraphFact(source="A", relation="USES", target="B")],
    )
    text = build_context_text(ctx)
    assert "[D::0] hello" in text and "(A) -[USES]-> (B)" in text


def test_build_context_text_without_facts() -> None:
    ctx = RetrievalContext(chunks=[RetrievedChunk(chunk_id="D::0", doc_id="D", text="x")], facts=[])
    assert "(none)" in build_context_text(ctx)
