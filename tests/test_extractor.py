"""Extractor control flow with a stubbed LLM call: fallback chain, backoff, guard, cache."""

from __future__ import annotations

import json

import httpx
import pytest
from groq import RateLimitError

from graphrag import models as m
from graphrag.data import Chunk

CHAIN = "model-b,model-c"


def _rate_limit(message: str) -> RateLimitError:
    request = httpx.Request("POST", "https://example.test")
    return RateLimitError(message, response=httpx.Response(429, request=request), body=None)


def tpd() -> RateLimitError:
    return _rate_limit("Rate limit reached on tokens per day (TPD): Limit 200000")


def tpm() -> RateLimitError:
    return _rate_limit("Rate limit reached on tokens per minute (TPM)")


def make_chunk(i: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"DOC-{i}::0",
        doc_id=f"DOC-{i}",
        chunk_index=0,
        text="text",
        category="cat",
        source="src",
    )


@pytest.fixture
def extractor(settings_env, monkeypatch):
    settings_env(GROQ_MODEL="model-a", GROQ_FALLBACK_MODELS=CHAIN, EXTRACTION_DELAY_S="0")
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)
    return m.GraphExtractor()


def test_model_chain_order(extractor) -> None:
    assert extractor._chain == ["model-a", "model-b", "model-c"]


def test_falls_through_chain_on_daily_limit(extractor) -> None:
    calls: list[str] = []
    ok = m.ExtractionResult()

    def fake(model: str, chunk: Chunk):
        calls.append(model)
        if model != "model-c":
            raise tpd()
        return ok

    extractor._call = fake
    assert extractor.extract(make_chunk()) is ok
    assert calls == ["model-a", "model-b", "model-c"]
    assert extractor.current_model == "model-c"


def test_skips_unavailable_model(extractor) -> None:
    calls: list[str] = []

    def fake(model: str, chunk: Chunk):
        calls.append(model)
        if model == "model-a":
            raise RuntimeError("Error code: 404 - model_not_found")
        return m.ExtractionResult()

    extractor._call = fake
    extractor.extract(make_chunk())
    assert calls == ["model-a", "model-b"]


def test_raises_when_every_model_is_exhausted(extractor) -> None:
    def fake(model: str, chunk: Chunk):
        raise tpd()

    extractor._call = fake
    with pytest.raises(m.QuotaExhaustedError):
        extractor.extract(make_chunk())


def test_per_minute_limit_backs_off_without_switching_model(extractor) -> None:
    attempts = {"n": 0}
    ok = m.ExtractionResult()

    def fake(model: str, chunk: Chunk):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise tpm()
        return ok

    extractor._call = fake
    assert extractor.extract(make_chunk()) is ok
    assert extractor.current_model == "model-a"
    assert attempts["n"] == 3


def test_rate_limit_inside_wrapper_is_detected_via_cause() -> None:
    wrapper = RuntimeError("wrapped by instructor")
    wrapper.__cause__ = tpd()
    assert m._is_rate_limit(wrapper)
    assert not m._is_rate_limit(ValueError("nope"))


def test_non_rate_limit_errors_propagate(extractor) -> None:
    def fake(model: str, chunk: Chunk):
        raise ValueError("bad output")

    extractor._call = fake
    with pytest.raises(ValueError):
        extractor.extract(make_chunk())


def test_gives_up_after_repeated_per_minute_limits(extractor) -> None:
    def fake(model: str, chunk: Chunk):
        raise tpm()

    extractor._call = fake
    with pytest.raises(RuntimeError, match="Exhausted rate-limit retries"):
        extractor.extract(make_chunk(), max_rate_limit_retries=2)


def test_failure_rate_guard_aborts_and_caches_nothing(extractor, tmp_path) -> None:
    def fake(model: str, chunk: Chunk):
        raise ValueError("bad output")

    extractor._call = fake
    cache = tmp_path / "cache.jsonl"
    with pytest.raises(m.ExtractionAbortedError):
        extractor.extract_many([make_chunk(i) for i in range(30)], cache_path=cache)
    assert cache.read_text() == ""


def test_extract_many_resumes_from_cache(extractor, tmp_path) -> None:
    cache = tmp_path / "cache.jsonl"
    done = make_chunk(0)
    cache.write_text(
        json.dumps({"chunk_id": done.chunk_id, "result": m.ExtractionResult().model_dump()}) + "\n"
    )
    seen: list[str] = []

    def fake(model: str, chunk: Chunk):
        seen.append(chunk.chunk_id)
        return m.ExtractionResult()

    extractor._call = fake
    results = extractor.extract_many([make_chunk(0), make_chunk(1)], cache_path=cache)

    assert seen == ["DOC-1::0"]  # the cached chunk was not sent to the LLM again
    assert set(results) == {"DOC-0::0", "DOC-1::0"}
    assert len(cache.read_text().strip().splitlines()) == 2


def test_extract_many_stops_cleanly_when_all_quotas_are_gone(extractor, tmp_path) -> None:
    def fake(model: str, chunk: Chunk):
        raise tpd()

    extractor._call = fake
    results = extractor.extract_many([make_chunk(i) for i in range(3)], cache_path=tmp_path / "c.jsonl")
    assert results == {}  # a quota stop is a pause, not a crash
