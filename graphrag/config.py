"""Centralized configuration for the Graph RAG application.

All tunable parameters, filesystem paths, and secrets are defined here.
Secrets and environment-specific values come from `.env`; everything else
has a sensible default that can be overridden via environment variables.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Filesystem layout ────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"


class Settings(BaseSettings):
    """Application settings, validated at startup (fail fast on bad config)."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # ── Groq / LLM ───────────────────────────────────────────────────────────
    groq_api_key: SecretStr = Field(..., description="Groq API key.")
    groq_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    llm_max_retries: int = Field(3, ge=0)
    extraction_delay_s: float = Field(2.0, ge=0.0, description="Pause between LLM calls.")
    groq_fallback_models: str = Field(
        "", description="Comma-separated models tried in order when the primary is exhausted."
    )
    max_failure_rate: float = Field(0.1, gt=0.0, le=1.0, description="Abort above this share.")
    min_failure_sample: int = Field(20, gt=0, description="Chunks attempted before the guard.")

    # ── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr = Field(..., description="Neo4j password.")
    neo4j_database: str = "neo4j"

    # ── Embeddings / vector index ────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(384, gt=0)
    embedding_device: Literal["auto", "cpu", "mps"] = "auto"
    vector_index_name: str = "chunk_embeddings"
    vector_similarity: Literal["cosine", "euclidean"] = "cosine"

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(800, gt=0, description="Max characters per chunk.")
    chunk_overlap: int = Field(100, ge=0)

    # ── Data splitting ───────────────────────────────────────────────────────
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    random_seed: int = 42

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = Field(5, gt=0)
    graph_fact_limit: int = Field(25, gt=0, description="Max graph facts in the context.")
    expansion_chunks: int = Field(2, ge=0, description="Extra chunks found via graph hops.")
    expansion_mode: Literal["graph_hop", "co_mention"] = Field(
        "co_mention",
        description="graph_hop: chunks of neighbouring entities; co_mention: chunks sharing entities.",
    )
    answer_max_tokens: int = Field(1024, gt=0)

    # API service
    api_key: SecretStr | None = Field(None, description="If set, /ask requires header X-API-Key.")
    rate_limit_per_minute: int = Field(20, gt=0, description="Per client, per instance.")
    max_question_chars: int = Field(500, gt=0)
    answer_model: str | None = Field(None, description="Model for answers; defaults to groq_model.")

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

    @property
    def model_chain(self) -> list[str]:
        """Primary model followed by fallbacks, de-duplicated, in order."""
        extra = [m.strip() for m in self.groq_fallback_models.split(",") if m.strip()]
        return list(dict.fromkeys([self.groq_model, *extra]))

    @property
    def resolved_device(self) -> str:
        """Pick the best torch device; Apple Silicon uses `mps` when available."""
        if self.embedding_device != "auto":
            return self.embedding_device
        try:
            import torch

            return "mps" if torch.backends.mps.is_available() else "cpu"
        except ImportError:
            return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated Settings instance."""
    return Settings()  # type: ignore[call-arg]


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once for the whole application."""
    logging.basicConfig(
        level=level or get_settings().log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    for noisy in ("httpx", "httpcore", "neo4j", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def ensure_directories() -> None:
    """Create data directories if they do not exist."""
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)