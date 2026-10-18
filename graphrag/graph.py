"""Neo4j integration: schema, embeddings, node/edge upserts, vector search."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from types import TracebackType
from typing import Any, Iterator, Sequence

from neo4j import Driver, GraphDatabase, ManagedTransaction
from sentence_transformers import SentenceTransformer

from graphrag.config import get_settings
from graphrag.data import Chunk
from graphrag.models import ExtractionResult, normalize_name

logger = logging.getLogger(__name__)

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REL_TYPE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _batched(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class Embedder:
    """Local SentenceTransformer wrapper (L2-normalized vectors)."""

    def __init__(self) -> None:
        s = get_settings()
        self._model = SentenceTransformer(s.embedding_model, device=s.resolved_device)
        dim = self._model.get_sentence_embedding_dimension()
        if dim != s.embedding_dim:
            raise ValueError(f"Model dim {dim} != configured EMBEDDING_DIM {s.embedding_dim}")

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        vecs = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.tolist()


class GraphStore:
    """Thin, idempotent persistence layer over Neo4j."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        s = get_settings()
        self._s = s
        self._driver: Driver = GraphDatabase.driver(
            s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password.get_secret_value())
        )
        self._embedder = embedder or Embedder()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def __enter__(self) -> "GraphStore":
        self._driver.verify_connectivity()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._driver.close()

    def ping(self) -> None:
        """Raise if Neo4j is unreachable (used by the API health check)."""
        self._driver.verify_connectivity()

    # ── schema ───────────────────────────────────────────────────────────────
    def setup_schema(self) -> None:
        """Create uniqueness constraints and the vector index (idempotent)."""
        s = self._s
        if not _IDENT.match(s.vector_index_name):
            raise ValueError(f"Unsafe index name: {s.vector_index_name!r}")

        statements = [
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
            f"""
            CREATE VECTOR INDEX {s.vector_index_name} IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {s.embedding_dim},
                `vector.similarity_function`: '{s.vector_similarity}'
            }}}}
            """,
        ]
        with self._driver.session(database=s.neo4j_database) as session:
            for stmt in statements:
                session.run(stmt).consume()
            session.run("CALL db.awaitIndexes(60)").consume()
        logger.info("Schema ready (constraints + vector index '%s')", s.vector_index_name)

    # ── writes ───────────────────────────────────────────────────────────────
    def upsert_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> None:
        """Embed chunks and upsert Document -> Chunk nodes."""
        for batch in _batched(chunks, batch_size):
            vectors = self._embedder.embed([c.text for c in batch])
            rows = [{**c.model_dump(), "embedding": v} for c, v in zip(batch, vectors)]
            self._write(self._tx_upsert_chunks, rows)
        logger.info("Upserted %d chunks", len(chunks))

    def upsert_extractions(
        self, extractions: dict[str, ExtractionResult], batch_size: int = 500
    ) -> None:
        """Upsert Entity nodes, MENTIONS edges, and typed entity-entity edges."""
        entity_rows: list[dict[str, Any]] = []
        rels_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for chunk_id, result in extractions.items():
            for e in result.entities:
                entity_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "key": e.key,
                        "name": e.name,
                        "type": e.type,
                        "description": e.description,
                    }
                )
            for r in result.relationships:
                if not _REL_TYPE.match(r.type):  # defense: type is interpolated into Cypher
                    logger.warning("Skipping unsafe relationship type %r", r.type)
                    continue
                rels_by_type[r.type].append(
                    {
                        "chunk_id": chunk_id,
                        "source": normalize_name(r.source),
                        "target": normalize_name(r.target),
                        "description": r.description,
                    }
                )

        for batch in _batched(entity_rows, batch_size):
            self._write(self._tx_upsert_entities, list(batch))
        for rel_type, rows in rels_by_type.items():
            for batch in _batched(rows, batch_size):
                self._write(self._tx_upsert_relationships, rel_type, list(batch))
        logger.info(
            "Upserted %d entity mentions and %d relationships (%d types)",
            len(entity_rows),
            sum(len(v) for v in rels_by_type.values()),
            len(rels_by_type),
        )

    # ── reads ────────────────────────────────────────────────────────────────
    def vector_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return the top-k chunks most similar to `query`."""
        k = top_k or self._s.top_k
        qvec = self._embedder.embed([query])[0]
        cypher = """
        CALL db.index.vector.queryNodes($index, $k, $qvec) YIELD node, score
        RETURN node.chunk_id AS chunk_id, node.doc_id AS doc_id,
               node.category AS category, node.text AS text, score
        ORDER BY score DESC
        """
        with self._driver.session(database=self._s.neo4j_database) as session:
            return session.run(
                cypher, index=self._s.vector_index_name, k=k, qvec=qvec
            ).data()

    def stats(self) -> dict[str, int]:
        """Node/relationship counts for quick sanity checks."""
        queries = {
            "documents": "MATCH (:Document) RETURN count(*) AS n",
            "chunks": "MATCH (:Chunk) RETURN count(*) AS n",
            "entities": "MATCH (:Entity) RETURN count(*) AS n",
            "mentions": "MATCH ()-[:MENTIONS]->() RETURN count(*) AS n",
            "entity_relationships": (
                "MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS n"
            ),
        }
        with self._driver.session(database=self._s.neo4j_database) as session:
            return {k: session.run(q).single()["n"] for k, q in queries.items()}

    def clear(self) -> None:
        """Delete ALL nodes and relationships. Destructive; dev use only."""
        with self._driver.session(database=self._s.neo4j_database) as session:
            session.run("MATCH (n) DETACH DELETE n").consume()
        logger.warning("Graph cleared")

    # ── transaction functions ────────────────────────────────────────────────
    def _write(self, fn: Any, *args: Any) -> None:
        with self._driver.session(database=self._s.neo4j_database) as session:
            session.execute_write(fn, *args)

    @staticmethod
    def _tx_upsert_chunks(tx: ManagedTransaction, rows: list[dict[str, Any]]) -> None:
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (d:Document {doc_id: row.doc_id})
              ON CREATE SET d.category = row.category, d.source = row.source
            MERGE (c:Chunk {chunk_id: row.chunk_id})
              SET c.doc_id = row.doc_id, c.chunk_index = row.chunk_index,
                  c.text = row.text, c.category = row.category
            WITH c, d, row
            CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            rows=rows,
        ).consume()

    @staticmethod
    def _tx_upsert_entities(tx: ManagedTransaction, rows: list[dict[str, Any]]) -> None:
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {key: row.key})
              ON CREATE SET e.name = row.name, e.type = row.type,
                            e.description = row.description
            WITH e, row
            MATCH (c:Chunk {chunk_id: row.chunk_id})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            rows=rows,
        ).consume()

    @staticmethod
    def _tx_upsert_relationships(
        tx: ManagedTransaction, rel_type: str, rows: list[dict[str, Any]]
    ) -> None:
        # rel_type is validated against _REL_TYPE before reaching here.
        tx.run(
            f"""
            UNWIND $rows AS row
            MATCH (s:Entity {{key: row.source}}), (t:Entity {{key: row.target}})
            MERGE (s)-[r:{rel_type}]->(t)
              ON CREATE SET r.description = row.description, r.source_chunks = [row.chunk_id]
              ON MATCH SET r.source_chunks =
                CASE WHEN row.chunk_id IN r.source_chunks
                     THEN r.source_chunks
                     ELSE r.source_chunks + row.chunk_id END
            """,
            rows=rows,
        ).consume()