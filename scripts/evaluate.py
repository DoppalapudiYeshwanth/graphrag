"""Retrieval evaluation: vector-only vs. vector + graph expansion."""

from __future__ import annotations

import json
from collections import defaultdict

from graphrag.config import PROCESSED_DATA_DIR, get_settings, setup_logging
from graphrag.graph import GraphStore
from graphrag.retrieval import GraphRetriever

QUESTIONS_PATH = PROCESSED_DATA_DIR / "eval_questions.jsonl"


def unique(seq: list[str]) -> list[str]:
    return list(dict.fromkeys(seq))


def main() -> None:
    rows = [json.loads(line) for line in QUESTIONS_PATH.open(encoding="utf-8")]
    k = get_settings().top_k

    vec_hits = hyb_hits = com_hits = 0
    rr_sum = 0.0
    per_cat: dict[str, list[int]] = defaultdict(list)
    misses: list[dict] = []

    with GraphStore() as store:
        retriever = GraphRetriever(store, expansion_mode="graph_hop")
        co_retriever = GraphRetriever(store, expansion_mode="co_mention")
        for r in rows:
            ctx = retriever.retrieve(r["question"])
            vec_docs = unique([c.doc_id for c in ctx.chunks if c.via == "vector"])
            all_docs = unique([c.doc_id for c in ctx.chunks])
            gold = r["gold_doc_id"]
            com_hits += gold in {c.doc_id for c in co_retriever.retrieve(r["question"]).chunks}

            hit = gold in vec_docs
            vec_hits += hit
            hyb_hits += gold in all_docs
            rr_sum += 1 / (vec_docs.index(gold) + 1) if hit else 0.0
            per_cat[r["category"]].append(int(hit))
            if not hit:
                misses.append({"question": r["question"], "gold": gold, "got": vec_docs})

    n = len(rows)
    print(f"\nQuestions: {n}   top_k: {k}")
    print(f"Vector-only   hit@{k}: {vec_hits / n:.1%}   MRR: {rr_sum / n:.3f}")
    print(f"graph_hop     hit@{k}+expansion: {hyb_hits / n:.1%}")
    print(f"co_mention    hit@{k}+expansion: {com_hits / n:.1%}")
    print("\nVector hit rate by category:")
    for cat, hits in sorted(per_cat.items()):
        print(f"  {cat:28s} {sum(hits) / len(hits):.1%}  (n={len(hits)})")
    if misses:
        print(f"\n{len(misses)} misses (first 5):")
        for m in misses[:5]:
            print(f"  gold={m['gold']} got={m['got']}\n    Q: {m['question']}")


if __name__ == "__main__":
    setup_logging("WARNING")
    main()
