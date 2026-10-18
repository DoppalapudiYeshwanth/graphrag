"""Hybrid Graph RAG: vector search + graph expansion + grounded generation."""

from __future__ import annotations

import logging
import re
from typing import Any

from groq import Groq
from pydantic import BaseModel, Field

from graphrag.config import get_settings
from graphrag.graph import GraphStore

logger = logging.getLogger(__name__)


class GraphFact(BaseModel):
    """A single entity-entity relationship with provenance."""

    source: str
    relation: str
    target: str
    description: str = ""
    source_chunks: list[str] = Field(default_factory=list)

    def render(self) -> str:
        text = f"({self.source}) -[{self.relation}]-> ({self.target})"
        return f"{text}: {self.description}" if self.description else text


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float | None = None
    via: str = "vector"  # "vector" or "graph"


class RetrievalContext(BaseModel):
    chunks: list[RetrievedChunk]
    facts: list[GraphFact]


class Answer(BaseModel):
    question: str
    answer: str
    context: RetrievalContext


SYSTEM_PROMPT = """You answer questions about computer-science research using ONLY the provided context.
The context has two parts: numbered text passages (each with an id) and knowledge-graph facts.
Rules:
- Use only information in the context. If it is insufficient, say so plainly; do not guess.
- Cite supporting passages inline by id, like [DOC-0123::0].
- Use graph facts to connect ideas across passages, but cite the passages that support them.
- Be concise and precise."""


class GraphRetriever:
    """Combines vector similarity search with graph traversal."""

    def __init__(self, store: GraphStore, expansion_mode: str | None = None) -> None:
        self._store = store
        self._s = get_settings()
        self._mode = expansion_mode or self._s.expansion_mode

    def retrieve(self, question: str) -> RetrievalContext:
        """Vector search, then expand through the entity graph."""
        s = self._s
        hits = self._store.vector_search(question, s.top_k)
        chunks = [
            RetrievedChunk(
                chunk_id=h["chunk_id"], doc_id=h["doc_id"], text=h["text"], score=h["score"]
            )
            for h in hits
        ]
        ids = [c.chunk_id for c in chunks]
        if not ids:
            return RetrievalContext(chunks=[], facts=[])

        facts = self._graph_facts(ids)
        if s.expansion_chunks:
            expand = (
                self._co_mention_chunks if self._mode == "co_mention" else self._expansion_chunks
            )
            chunks.extend(expand(ids))
        return RetrievalContext(chunks=chunks, facts=facts)

    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self._store._driver.session(database=self._s.neo4j_database) as session:
            return session.run(cypher, **params).data()

    def _graph_facts(self, chunk_ids: list[str]) -> list[GraphFact]:
        """Relationships touching entities mentioned in the retrieved chunks."""
        rows = self._run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)-[r]-(n:Entity)
            WHERE c.chunk_id IN $ids
            WITH DISTINCT r, startNode(r) AS s, endNode(r) AS t
            RETURN s.name AS source, type(r) AS relation, t.name AS target,
                   r.description AS description, r.source_chunks AS source_chunks
            LIMIT $limit
            """,
            ids=chunk_ids,
            limit=self._s.graph_fact_limit,
        )
        return [
            GraphFact(
                source=r["source"],
                relation=r["relation"],
                target=r["target"],
                description=r["description"] or "",
                source_chunks=r["source_chunks"] or [],
            )
            for r in rows
        ]

    def _expansion_chunks(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        """Chunks reached by hopping chunk -> entity -> related entity -> chunk."""
        rows = self._run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(:Entity)-[r]-(n:Entity)<-[:MENTIONS]-(other:Chunk)
            WHERE c.chunk_id IN $ids AND NOT other.chunk_id IN $ids
            WITH other, count(DISTINCT n) AS strength
            ORDER BY strength DESC
            LIMIT $limit
            RETURN other.chunk_id AS chunk_id, other.doc_id AS doc_id, other.text AS text
            """,
            ids=chunk_ids,
            limit=self._s.expansion_chunks,
        )
        return [RetrievedChunk(via="graph", **r) for r in rows]

    def _co_mention_chunks(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        """Chunks that mention the same entities as the seed chunks (shared-entity hop)."""
        rows = self._run(_co_mention_query(), ids=chunk_ids, limit=self._s.expansion_chunks)
        return [RetrievedChunk(via="graph", **r) for r in rows]


def _co_mention_query() -> str:
    """Chunks sharing entities with the seeds; entities mentioned rarely weigh more."""
    return """
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(other:Chunk)
    WHERE c.chunk_id IN $ids AND NOT other.chunk_id IN $ids
    WITH other, sum(1.0 / COUNT { (e)<-[:MENTIONS]-() }) AS strength
    ORDER BY strength DESC
    LIMIT $limit
    RETURN other.chunk_id AS chunk_id, other.doc_id AS doc_id, other.text AS text
    """


def build_context_text(ctx: RetrievalContext) -> str:
    """Render retrieval results into the prompt context block."""
    passages = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in ctx.chunks)
    facts = "\n".join(f"- {f.render()}" for f in ctx.facts) or "(none)"
    return f"PASSAGES:\n{passages}\n\nGRAPH FACTS:\n{facts}"


class GraphRAG:
    """End-to-end question answering."""

    def __init__(self, store: GraphStore) -> None:
        s = get_settings()
        self._s = s
        self._retriever = GraphRetriever(store)
        self._llm = Groq(api_key=s.groq_api_key.get_secret_value())

    def ask(self, question: str) -> Answer:
        ctx = self._retriever.retrieve(question)
        if not ctx.chunks:
            return Answer(question=question, answer="No relevant context found.", context=ctx)

        resp = self._llm.chat.completions.create(
            model=self._s.answer_model or self._s.groq_model,
            temperature=self._s.llm_temperature,
            max_tokens=self._s.answer_max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{build_context_text(ctx)}\n\nQUESTION: {question}",
                },
            ],
        )
        text = re.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=re.S)
        text = text.strip()
        return Answer(question=question, answer=text, context=ctx)
