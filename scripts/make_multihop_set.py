"""Generate multi-hop evaluation questions from document pairs that share a graph entity.

Each question must need BOTH documents to answer, so retrieving only one is a partial miss.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time

from groq import Groq

from graphrag.config import PROCESSED_DATA_DIR, get_settings, setup_logging
from graphrag.data import load_corpus
from graphrag.graph import GraphStore

logger = logging.getLogger(__name__)

OUT_PATH = PROCESSED_DATA_DIR / "multihop_questions.jsonl"
BRIDGE_TYPES = ["Algorithm", "Technology", "Model", "Framework", "Dataset", "Metric"]

# Entities mentioned in exactly 2-3 documents make specific, non-generic bridges.
BRIDGE_QUERY = """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE e.type IN $types
WITH e, collect(DISTINCT c.doc_id) AS docs
WHERE size(docs) >= 2 AND size(docs) <= 3
RETURN e.name AS entity, docs
"""

PROMPT = """Below are two research abstracts (A and B) that both involve "{entity}".
Write ONE question that can only be answered by combining information from BOTH abstracts,
for example a comparison, a connection, or how one builds on or differs from the other.
Rules: do not mention titles, authors, or the words 'abstract' or 'paper'; make the question
understandable on its own and mention "{entity}" only if it is natural. Return only the question.

Abstract A:
{a}

Abstract B:
{b}"""


def pick_pairs(rows: list[dict], n: int, seed: int) -> list[tuple[str, str, str]]:
    """Choose up to n (entity, doc_a, doc_b) triples with every document used once."""
    rng = random.Random(seed)
    rows = [r for r in rows if len(r["entity"]) >= 4]
    rng.shuffle(rows)
    used: set[str] = set()
    triples: list[tuple[str, str, str]] = []
    for r in rows:
        free = [d for d in r["docs"] if d not in used]
        if len(free) < 2:
            continue
        a, b = free[0], free[1]
        used.update((a, b))
        triples.append((r["entity"], a, b))
        if len(triples) == n:
            break
    return triples


def main(n: int) -> None:
    s = get_settings()
    client = Groq(api_key=s.groq_api_key.get_secret_value())
    model = s.answer_model or s.groq_model

    docs = load_corpus().set_index("doc_id")
    with GraphStore() as store:
        with store._driver.session(database=s.neo4j_database) as session:
            rows = session.run(BRIDGE_QUERY, types=BRIDGE_TYPES).data()
    triples = pick_pairs(rows, n, s.random_seed)
    logger.info("Selected %d document pairs from %d candidate bridge entities", len(triples), len(rows))

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for entity, a, b in triples:
            prompt = PROMPT.format(entity=entity, a=docs.loc[a, "text"], b=docs.loc[b, "text"])
            resp = client.chat.completions.create(
                model=model, temperature=0.3, messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.choices[0].message.content or ""
            q = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
            f.write(json.dumps({"question": q, "gold_doc_ids": [a, b], "bridge_entity": entity}) + "\n")
            f.flush()
            logger.info("%s + %s via %r -> %s", a, b, entity, q)
            time.sleep(s.extraction_delay_s)
    logger.info("Wrote %d questions to %s", len(triples), OUT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    args = parser.parse_args()
    setup_logging()
    main(args.n)
