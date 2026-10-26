"""Chunk -> extract -> embed -> load into Neo4j."""

from __future__ import annotations

import argparse
import logging
import sys

from graphrag.config import setup_logging
from graphrag.data import chunk_documents, load_corpus, split_corpus
from graphrag.graph import GraphStore
from graphrag.models import ExtractionAbortedError, GraphExtractor

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process N chunks.")
    parser.add_argument("--reset", action="store_true", help="Clear the graph first.")
    args = parser.parse_args()

    setup_logging()
    train, _ = split_corpus(load_corpus())
    chunks = chunk_documents(train)
    if args.limit:
        chunks = chunks[: args.limit]

    try:
        extractions = GraphExtractor().extract_many(chunks)  # cached + resumable
    except ExtractionAbortedError as exc:
        logger.error("Extraction aborted, graph NOT modified: %s", exc)
        sys.exit(1)

    covered = [c for c in chunks if c.chunk_id in extractions]
    if not any(extractions[c.chunk_id].entities for c in covered):
        logger.error("No entities extracted for any chunk; graph NOT modified.")
        sys.exit(1)
    logger.info("Extraction coverage: %d of %d chunks", len(covered), len(chunks))

    with GraphStore() as store:
        if args.reset:
            store.clear()
        store.setup_schema()
        store.upsert_chunks(chunks)
        store.upsert_extractions(
            {c.chunk_id: extractions[c.chunk_id] for c in chunks if c.chunk_id in extractions}
        )
        logger.info("Graph stats: %s", store.stats())
