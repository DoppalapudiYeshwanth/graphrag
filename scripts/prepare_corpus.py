"""Build data/raw/corpus.csv from the ArXiv abstracts dataset on Hugging Face.

Output schema: doc_id, text, category, source.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from datasets import load_dataset

from graphrag.config import RAW_DATA_DIR, ensure_directories, setup_logging

logger = logging.getLogger(__name__)

# Primary arXiv category -> readable domain tag.
CATEGORY_MAP: dict[str, str] = {
    "cs.AI": "Artificial Intelligence",
    "cs.NI": "Computer Networks",
    "cs.SE": "Software Engineering",
    "cs.DB": "Databases",
    "cs.CR": "Security and Cryptography",
}


def build_corpus(per_category: int, seed: int) -> pd.DataFrame:
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train")
    df = ds.to_pandas()

    # `categories` may be a list or a space-separated string depending on version.
    def primary(cats) -> str:
        first = cats[0] if not isinstance(cats, str) else cats.split()[0]
        return first.strip()

    df["primary"] = df["categories"].map(primary)
    df = df[df["primary"].isin(CATEGORY_MAP)].copy()
    df["category"] = df["primary"].map(CATEGORY_MAP)
    df["text"] = (df["title"].str.strip() + ". " + df["abstract"].str.strip()).str.replace(
        r"\s+", " ", regex=True
    )

    df = (
        df.groupby("category", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_category), random_state=seed))
        .reset_index(drop=True)
    )
    df["doc_id"] = [f"DOC-{i + 1:04d}" for i in range(len(df))]
    df["source"] = "arXiv"
    return df[["doc_id", "text", "category", "source"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    ensure_directories()
    out = build_corpus(args.per_category, args.seed)
    path = RAW_DATA_DIR / "corpus.csv"
    out.to_csv(path, index=False)
    logger.info("Wrote %d docs to %s\n%s", len(out), path, out["category"].value_counts())