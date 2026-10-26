"""Preflight check: is this machine, and this configuration, ready to run Graph RAG?

Run it before starting the app or deploying:

    python -m scripts.check_setup            # fast checks
    python -m scripts.check_setup --model    # also loads the embedding model (about 15 seconds)

It prints one line per check, never prints a secret, and exits non-zero if anything is wrong.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from groq import APIConnectionError, AuthenticationError, Groq
from neo4j import GraphDatabase

from graphrag.config import Settings, get_settings

Result = tuple[bool, str]


def check_python() -> Result:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < (3, 10):
        return False, f"Python {version} is too old; 3.10 is the minimum"
    if (v.major, v.minor) == (3, 10):
        return (
            True,
            f"Python {version} works, but 3.10 reaches end of life in October 2026; 3.12 is recommended",
        )
    return True, f"Python {version}"


def check_groq(settings: Settings, client: Any | None = None) -> Result:
    """The key must be accepted and every configured model must be available to it."""
    client = client or Groq(api_key=settings.groq_api_key.get_secret_value(), max_retries=1)
    try:
        available = {m.id for m in client.models.list().data}
    except AuthenticationError:
        return (
            False,
            "Groq rejected the API key (expired or wrong). Create a new one at https://console.groq.com/keys",
        )
    except APIConnectionError:
        return False, "Could not reach the Groq API (network problem?)"
    wanted = [*settings.model_chain, settings.answer_model or settings.groq_model]
    missing = sorted({m for m in wanted if m not in available})
    if missing:
        return (
            False,
            f"Groq key works, but these configured models are not available to it: {', '.join(missing)}",
        )
    return True, f"Groq key accepted; all {len(set(wanted))} configured models are available"


def check_neo4j(settings: Settings) -> Result:
    """Reachable, authenticated, vector index present, and the demo corpus loaded."""
    try:
        with GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
        ) as driver:
            driver.verify_connectivity()
            with driver.session(database=settings.neo4j_database) as session:
                chunks = session.run(
                    "MATCH (c:Chunk) WHERE c.workspace IS NULL RETURN count(c) AS n"
                ).single()["n"]
                index = session.run(
                    "SHOW VECTOR INDEXES YIELD name WHERE name = $name RETURN count(*) AS n",
                    name=settings.vector_index_name,
                ).single()["n"]
    except Exception as exc:
        return (
            False,
            f"Neo4j check failed ({type(exc).__name__}). Is the database running and the password right?",
        )
    if not index:
        return (
            False,
            f"Connected, but the vector index '{settings.vector_index_name}' does not exist (start the app or run build_graph once)",
        )
    if not chunks:
        return (
            True,
            "Connected, vector index present, but the demo corpus is empty (uploads still work; run build_graph for the demo)",
        )
    return True, f"Connected, vector index present, {chunks} demo chunks loaded"


def check_embedding_model(settings: Settings) -> Result:
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(settings.embedding_model, device=settings.resolved_device)
    except Exception as exc:
        return False, f"Could not load the embedding model ({type(exc).__name__})"
    dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    if dim != settings.embedding_dim:
        return False, f"Embedding model has {dim} dimensions but EMBEDDING_DIM is {settings.embedding_dim}"
    return True, f"Embedding model loaded on {settings.resolved_device} ({dim} dimensions)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", action="store_true", help="also load the embedding model")
    args = parser.parse_args()

    try:
        settings = get_settings()
    except Exception as exc:  # missing or invalid configuration: say which names, never the values
        names = (
            ", ".join(sorted({str(e["loc"][0]).upper() for e in getattr(exc, "errors", lambda: [])()}))
            or "settings"
        )
        print(
            f"FAIL  Configuration is invalid or incomplete: {names}. Copy .env.example to .env and fill it in."
        )
        return 1

    checks: list[tuple[str, Result]] = [
        ("Python", check_python()),
        ("Groq", check_groq(settings)),
        ("Neo4j", check_neo4j(settings)),
    ]
    if args.model:
        checks.append(("Embeddings", check_embedding_model(settings)))

    failed = 0
    for name, (ok, detail) in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}: {detail}")
        failed += not ok
    print("\nReady." if not failed else f"\n{failed} problem(s) to fix before starting.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
