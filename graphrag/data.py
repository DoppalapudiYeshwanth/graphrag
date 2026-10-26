"""Document ingestion, stratified splitting, and recursive chunking."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from graphrag.config import RAW_DATA_DIR, get_settings

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("doc_id", "text", "category", "source")
DEFAULT_CORPUS_PATH: Path = RAW_DATA_DIR / "corpus.csv"


class Chunk(BaseModel):
    """A text chunk with parent-document metadata for Neo4j traceability."""

    chunk_id: str = Field(..., description="Unique id: '<doc_id>::<index>'.")
    doc_id: str
    chunk_index: int = Field(..., ge=0)
    text: str
    category: str
    source: str


def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> pd.DataFrame:
    """Load and validate the corpus CSV.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if required columns are missing or doc_ids are not unique.
    """
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found at {path}. Run scripts/prepare_corpus.py.")

    df = pd.read_csv(path, dtype=str)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Corpus is missing columns: {sorted(missing)}")

    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    if len(df) < before:
        logger.warning("Dropped %d rows with missing/empty fields", before - len(df))
    if df["doc_id"].duplicated().any():
        raise ValueError("doc_id values must be unique")

    logger.info("Loaded %d documents across %d categories", len(df), df["category"].nunique())
    return df


def split_corpus(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on `category`.

    Categories with fewer than 2 documents cannot be stratified and are
    kept in the train split.
    """
    s = get_settings()
    counts = df["category"].value_counts()
    rare = counts[counts < 2].index
    if len(rare):
        logger.warning("Categories too small to stratify (kept in train): %s", list(rare))

    splittable = df[~df["category"].isin(rare)]
    train, test = train_test_split(
        splittable,
        test_size=s.test_size,
        random_state=s.random_seed,
        stratify=splittable["category"],
    )
    train = pd.concat([train, df[df["category"].isin(rare)]])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    logger.info("Split: %d train / %d test", len(train), len(test))
    return train, test


def chunk_documents(df: pd.DataFrame) -> list[Chunk]:
    """Recursively chunk each document, attaching parent metadata to every chunk."""
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for row in df.itertuples(index=False):
        for i, piece in enumerate(splitter.split_text(row.text)):
            chunks.append(
                Chunk(
                    chunk_id=f"{row.doc_id}::{i}",
                    doc_id=row.doc_id,
                    chunk_index=i,
                    text=piece,
                    category=row.category,
                    source=row.source,
                )
            )
    logger.info("Created %d chunks from %d documents", len(chunks), len(df))
    return chunks
