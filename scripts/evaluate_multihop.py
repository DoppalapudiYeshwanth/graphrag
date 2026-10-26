"""Multi-hop evaluation: does graph expansion find the second document?

Systems compared (retrieved chunk budget in brackets):
  vector@5    plain vector search                          [5]
  vector@7    plain vector search, same budget as below    [7]  <- fairness control
  graph_hop   vector@5 + chunks of neighbouring entities   [5+2]
  co_mention  vector@5 + chunks sharing the seeds' entities [5+2]
"""

from __future__ import annotations

import json

from graphrag.config import PROCESSED_DATA_DIR, get_settings, setup_logging
from graphrag.graph import GraphStore
from graphrag.retrieval import GraphRetriever

QUESTIONS_PATH = PROCESSED_DATA_DIR / "multihop_questions.jsonl"
SYSTEMS = ["vector@5", "vector@7", "graph_hop", "co_mention"]


def main() -> None:
    s = get_settings()
    rows = [json.loads(line) for line in QUESTIONS_PATH.open(encoding="utf-8")]
    both = dict.fromkeys(SYSTEMS, 0)
    any_ = dict.fromkeys(SYSTEMS, 0)
    recall = dict.fromkeys(SYSTEMS, 0.0)
    helped: list[dict] = []

    with GraphStore() as store:
        hop = GraphRetriever(store, expansion_mode="graph_hop")
        com = GraphRetriever(store, expansion_mode="co_mention")
        for r in rows:
            q, gold = r["question"], set(r["gold_doc_ids"])
            hop_ctx, com_ctx = hop.retrieve(q), com.retrieve(q)
            found = {
                "vector@5": {c.doc_id for c in hop_ctx.chunks if c.via == "vector"},
                "vector@7": {h["doc_id"] for h in store.vector_search(q, s.top_k + s.expansion_chunks)},
                "graph_hop": {c.doc_id for c in hop_ctx.chunks},
                "co_mention": {c.doc_id for c in com_ctx.chunks},
            }
            for k in SYSTEMS:
                hit = len(gold & found[k])
                both[k] += hit == len(gold)
                any_[k] += hit >= 1
                recall[k] += hit / len(gold)
            if len(gold & found["co_mention"]) > len(gold & found["vector@7"]):
                helped.append({"q": q, "bridge": r["bridge_entity"], "gold": sorted(gold)})

    n = len(rows)
    print(f"\nMulti-hop questions: {n}  (top_k={s.top_k}, expansion={s.expansion_chunks})\n")
    print(f"{'system':12s} {'both docs':>10s} {'>=1 doc':>9s} {'doc recall':>11s}")
    for k in SYSTEMS:
        print(f"{k:12s} {both[k] / n:>10.1%} {any_[k] / n:>9.1%} {recall[k] / n:>11.1%}")
    print(f"\nQuestions where co_mention beat the same-budget vector@7: {len(helped)}")
    for h in helped[:3]:
        print(f"  bridge={h['bridge']!r} gold={h['gold']}\n    Q: {h['q'][:200]}")


if __name__ == "__main__":
    setup_logging("WARNING")
    main()
