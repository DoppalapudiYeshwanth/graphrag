"""Shared fixtures. Tests never use real credentials and never touch the real graph."""

from __future__ import annotations

import hashlib
import os

# Dummy secrets set BEFORE graphrag is imported: environment variables beat `.env`,
# so a developer's real keys are never used by the test suite.
os.environ["GROQ_API_KEY"] = "test-key"
os.environ["NEO4J_PASSWORD"] = "test-password"
# Pin every setting the tests rely on, so results never depend on a developer's `.env`.
for _key, _value in {
    "GROQ_MODEL": "model-a",
    "GROQ_FALLBACK_MODELS": "",
    "ANSWER_MODEL": "",
    "CHUNK_SIZE": "800",
    "CHUNK_OVERLAP": "100",
    "TEST_SIZE": "0.2",
    "RANDOM_SEED": "42",
    "TOP_K": "5",
    "EXPANSION_CHUNKS": "2",
    "EMBEDDING_DIM": "384",
}.items():
    os.environ[_key] = _value

import numpy as np
import pytest

from graphrag.config import get_settings


class FakeEmbedder:
    """Deterministic 384-d unit vectors derived from the text; no model download."""

    dim = 384

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            vec = np.random.default_rng(seed).normal(size=self.dim)
            out.append((vec / np.linalg.norm(vec)).tolist())
        return out


@pytest.fixture
def settings_env(monkeypatch):
    """Set environment overrides and rebuild the cached Settings; restored afterwards."""

    def apply(**env: str):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()
        return get_settings()

    yield apply
    monkeypatch.undo()
    get_settings.cache_clear()
