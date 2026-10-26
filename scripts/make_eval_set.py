"""Generate one evaluation question per sampled document."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time

from groq import Groq

from graphrag.config import PROCESSED_DATA_DIR, get_settings, setup_logging
from graphrag.data import load_corpus, split_corpus
from graphrag.models import EXTRACTIONS_PATH

logger = logging.getLogger(__name__)

OUT_PATH = PROCESSED_DATA_DIR / "eval_questions.jsonl"

PROMPT = """Read this research abstract and write ONE specific question that the abstract answers.
Rules: do not mention the title, authors, or the words 'abstract' or 'paper'; make the question
understandable on its own. Return only the question."""


def extracted_doc_ids() -> set[str]:
    """Doc ids that already have graph extractions in the cache."""
    ids: set[str] = set()
    with EXTRACTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            ids.add(json.loads(line)["chunk_id"].split("::")[0])
    return ids


def main(n: int) -> None:
    s = get_settings()
    client = Groq(api_key=s.groq_api_key.get_secret_value())
    model = s.answer_model or s.groq_model

    train, _ = split_corpus(load_corpus())
    pool = train[train["doc_id"].isin(extracted_doc_ids())]
    sample = pool.sample(min(n, len(pool)), random_state=s.random_seed)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in sample.itertuples(index=False):
            resp = client.chat.completions.create(
                model=model,
                temperature=0.3,
                messages=[{"role": "user", "content": f"{PROMPT}\n\nAbstract:\n{row.text}"}],
            )
            raw = resp.choices[0].message.content or ""
            q = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
            f.write(json.dumps({"question": q, "gold_doc_id": row.doc_id, "category": row.category}) + "\n")
            f.flush()
            logger.info("%s -> %s", row.doc_id, q)
            time.sleep(s.extraction_delay_s)
    logger.info("Wrote %d questions to %s", len(sample), OUT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()
    setup_logging()
    main(args.n)
