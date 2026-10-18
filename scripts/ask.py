"""Ask the Graph RAG system a question from the command line."""

from __future__ import annotations

import argparse

from graphrag.config import setup_logging
from graphrag.graph import GraphStore
from graphrag.retrieval import GraphRAG

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()

    setup_logging("WARNING")
    with GraphStore() as store:
        result = GraphRAG(store).ask(args.question)

    print(f"\n{result.answer}\n")
    print("Sources:")
    for c in result.context.chunks:
        score = f"{c.score:.3f}" if c.score is not None else "graph"
        print(f"  [{c.chunk_id}] ({c.via}, {score}) {c.text[:80]}...")
    if args.show_context:
        print("\nGraph facts:")
        for f in result.context.facts:
            print(f"  {f.render()}")
