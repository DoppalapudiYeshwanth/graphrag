"""Pydantic schemas and Instructor/Groq wrapper for entity & relationship extraction."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Literal

import instructor
from groq import Groq, RateLimitError
from pydantic import BaseModel, Field, field_validator, model_validator

from graphrag.config import PROCESSED_DATA_DIR, get_settings
from graphrag.data import Chunk

logger = logging.getLogger(__name__)

EntityType = Literal[
    "Algorithm",
    "Model",
    "Framework",
    "Concept",
    "Task",
    "Dataset",
    "Metric",
    "Technology",
    "Organization",
]

EXTRACTIONS_PATH: Path = PROCESSED_DATA_DIR / "extractions.jsonl"


def normalize_name(name: str) -> str:
    """Canonical key for entity dedup: lowercase, collapsed whitespace."""
    return re.sub(r"[\s_]+", " ", name).strip().lower()


class Entity(BaseModel):
    """A named technical concept found in the text."""

    name: str = Field(..., description="Canonical name, e.g. 'Transformer', 'BERT'.")
    type: EntityType
    description: str = Field("", description="One short sentence describing the entity.")

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        v = re.sub(r"[\s_]+", " ", v).strip()
        if not v:
            raise ValueError("entity name must not be empty")
        return v

    @property
    def key(self) -> str:
        return normalize_name(self.name)


ALLOWED_RELATIONS: tuple[str, ...] = (
    "USES",
    "APPLIED_TO",
    "IMPROVES",
    "PART_OF",
    "EVALUATED_ON",
    "BASED_ON",
    "OUTPERFORMS",
    "EXTENDS",
    "RELATED_TO",
)

# Only same-direction synonyms are mapped; reversed verbs (USED_BY, INCLUDES) fall to RELATED_TO
# so an edge is never silently flipped.
_RELATION_SYNONYMS: dict[str, str] = {
    "UTILIZES": "USES",
    "EMPLOYS": "USES",
    "LEVERAGES": "USES",
    "RELIES_ON": "USES",
    "DEPENDS_ON": "USES",
    "REQUIRES": "USES",
    "USED": "USES",
    "USE": "USES",
    "APPLIES_TO": "APPLIED_TO",
    "USED_FOR": "APPLIED_TO",
    "USED_IN": "APPLIED_TO",
    "ENHANCES": "IMPROVES",
    "OPTIMIZES": "IMPROVES",
    "REDUCES": "IMPROVES",
    "BOOSTS": "IMPROVES",
    "INCREASES": "IMPROVES",
    "IMPROVE": "IMPROVES",
    "COMPONENT_OF": "PART_OF",
    "MEMBER_OF": "PART_OF",
    "SUBSET_OF": "PART_OF",
    "TESTED_ON": "EVALUATED_ON",
    "EVALUATED_WITH": "EVALUATED_ON",
    "BENCHMARKED_ON": "EVALUATED_ON",
    "EVALUATED_ON": "EVALUATED_ON",
    "DERIVED_FROM": "BASED_ON",
    "BUILT_ON": "BASED_ON",
    "INSPIRED_BY": "BASED_ON",
    "BEATS": "OUTPERFORMS",
    "SURPASSES": "OUTPERFORMS",
    "OUTPERFORM": "OUTPERFORMS",
    "BUILDS_ON": "EXTENDS",
    "GENERALIZES": "EXTENDS",
    "EXTEND": "EXTENDS",
}
_RELATION_PREFIXES = ("CAN_BE_", "IS_", "ARE_", "WAS_", "CAN_")


def canonical_relation(raw: str) -> str:
    """Map a free-form relationship label onto the closed ALLOWED_RELATIONS set."""
    v = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").upper()
    if not v:
        raise ValueError("relationship type must not be empty")
    for prefix in _RELATION_PREFIXES:
        if v.startswith(prefix) and len(v) > len(prefix):
            v = v[len(prefix) :]
            break
    if v in ALLOWED_RELATIONS:
        return v
    return _RELATION_SYNONYMS.get(v, "RELATED_TO")


class Relationship(BaseModel):
    """A directed, typed relationship between two entities."""

    source: str = Field(..., description="Name of the source entity (must be in entities).")
    target: str = Field(..., description="Name of the target entity (must be in entities).")
    type: str = Field(
        ...,
        description="One of: " + ", ".join(ALLOWED_RELATIONS) + ". Use RELATED_TO only if none fits.",
    )
    description: str = Field("", description="One short sentence of evidence.")

    @field_validator("type")
    @classmethod
    def _to_allowed_rel_type(cls, v: str) -> str:
        return canonical_relation(v)


class ExtractionResult(BaseModel):
    """Structured output for one chunk."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

    @model_validator(mode="after")
    def _prune_invalid(self) -> ExtractionResult:
        """Dedupe entities, drop dangling/self-referential relationships."""
        seen: dict[str, Entity] = {}
        for e in self.entities:
            seen.setdefault(e.key, e)
        self.entities = list(seen.values())

        valid: list[Relationship] = []
        for r in self.relationships:
            s, t = normalize_name(r.source), normalize_name(r.target)
            if s in seen and t in seen and s != t:
                valid.append(r)
        self.relationships = valid
        return self


SYSTEM_PROMPT = """You are an expert knowledge-graph builder for computer science research.
Extract entities and relationships from the given text.

Rules:
- Only extract what is explicitly stated in the text. Do not invent facts.
- Use canonical, singular names (e.g. 'Transformer', not 'the transformers').
- Resolve abbreviations to the most common name used in the text.
- Every relationship's source and target MUST exactly match an entity name you extracted.
- Relationship type MUST be exactly one of: USES, IMPROVES, EXTENDS, EVALUATED_ON,
  OUTPERFORMS, BASED_ON, APPLIED_TO, PART_OF, RELATED_TO. Use RELATED_TO only if none of the
  others fits. Never invent other relationship types.
- EVERY entity MUST include "name", "type" (one of the allowed types) and "description".
- Prefer fewer, high-quality items over exhaustive noisy ones. Return empty lists if nothing fits."""


MODEL_RETRY_AFTER_S = 3600.0


class QuotaExhaustedError(RuntimeError):
    """Raised when every model in the chain has used up its daily token quota."""


class ExtractionAbortedError(RuntimeError):
    """Raised when too many chunks fail, so the run does not look successful while empty."""


def _is_rate_limit(exc: BaseException) -> bool:
    """True if `exc` (or its cause, e.g. inside Instructor's retry wrapper) is a 429."""
    return isinstance(exc, RateLimitError) or isinstance(exc.__cause__, RateLimitError)


def _is_daily_limit(exc: BaseException) -> bool:
    """True for a tokens-per-day 429, where waiting seconds will not help."""
    text = str(exc)
    return "(TPD)" in text or "per day" in text


def llm_extra_params(model: str) -> dict[str, str]:
    """Per-model request options. gpt-oss models are reasoning models; low effort saves most tokens."""
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    return {}


def _is_model_unavailable(exc: BaseException) -> bool:
    """True when the model is retired or this key cannot access it."""
    text = str(exc)
    return "model_not_found" in text or "model_decommissioned" in text


class GraphExtractor:
    """Extracts structured graph data from chunks using Groq via Instructor.

    Models are tried in `Settings.model_chain` order. A model whose daily quota is
    exhausted (or that is unavailable) is skipped for the rest of the run.
    """

    def __init__(self) -> None:
        s = get_settings()
        self._settings = s
        self._chain: list[str] = s.model_chain
        self._idx = 0
        self._advanced_at = 0.0
        self._client = instructor.from_groq(
            Groq(api_key=s.groq_api_key.get_secret_value()),
            mode=instructor.Mode.JSON,
        )

    @property
    def current_model(self) -> str:
        # A long-running server must not stay on a fallback forever: daily quotas reset,
        # so after an hour we try the primary model again.
        if self._idx > 0 and time.monotonic() - self._advanced_at > MODEL_RETRY_AFTER_S:
            logger.info("Retrying primary model %s", self._chain[0])
            self._idx = 0
        return self._chain[self._idx]

    def _advance_model(self, exc: BaseException) -> None:
        """Move to the next model in the chain, or raise when none are left."""
        logger.warning("Model %s unusable (%s); trying next", self.current_model, str(exc)[:120])
        self._idx += 1
        self._advanced_at = time.monotonic()
        if self._idx >= len(self._chain):
            raise QuotaExhaustedError(
                f"All models exhausted or unavailable: {', '.join(self._chain)}"
            ) from exc
        logger.info("Switched to model %s", self.current_model)

    @staticmethod
    def _extra_params(model: str) -> dict[str, str]:
        return llm_extra_params(model)

    def _call(self, model: str, chunk: Chunk) -> ExtractionResult:
        s = self._settings
        return self._client.chat.completions.create(
            model=model,
            temperature=s.llm_temperature,
            max_retries=s.llm_max_retries,
            **self._extra_params(model),
            response_model=ExtractionResult,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Domain: {chunk.category}\n\nText:\n{chunk.text}"},
            ],
        )

    def extract(self, chunk: Chunk, max_rate_limit_retries: int = 5) -> ExtractionResult:
        """Extract entities/relationships from one chunk.

        Instructor retries on schema-validation failures (`llm_max_retries`). Here we
        back off on per-minute 429s and fall through the model chain on daily-quota
        or model-unavailable errors.
        """
        attempt = 0
        while attempt < max_rate_limit_retries:
            try:
                return self._call(self.current_model, chunk)
            except Exception as exc:
                if _is_model_unavailable(exc) or (_is_rate_limit(exc) and _is_daily_limit(exc)):
                    self._advance_model(exc)
                    continue
                if not _is_rate_limit(exc):
                    raise
                wait = 2**attempt * 5
                logger.warning("Rate limited on %s; sleeping %ds", chunk.chunk_id, wait)
                time.sleep(wait)
                attempt += 1
        raise RuntimeError(f"Exhausted rate-limit retries for {chunk.chunk_id}")

    def extract_many(
        self, chunks: list[Chunk], cache_path: Path = EXTRACTIONS_PATH
    ) -> dict[str, ExtractionResult]:
        """Extract for all chunks, caching to JSONL so reruns skip finished work."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        results: dict[str, ExtractionResult] = {}

        if cache_path.exists():
            with cache_path.open(encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    results[rec["chunk_id"]] = ExtractionResult.model_validate(rec["result"])
            logger.info("Loaded %d cached extractions", len(results))

        todo = [c for c in chunks if c.chunk_id not in results]
        failed: list[str] = []
        s = self._settings
        with cache_path.open("a", encoding="utf-8") as f:
            for i, chunk in enumerate(todo, 1):
                try:
                    res = self.extract(chunk)
                except QuotaExhaustedError as exc:
                    logger.error(
                        "Daily token quota exhausted on every model; stopping. Rerun later "
                        "to resume (%d chunks done). %s",
                        len(results),
                        exc,
                    )
                    break
                except Exception as exc:
                    logger.exception("Extraction failed for %s", chunk.chunk_id)
                    failed.append(chunk.chunk_id)
                    if i >= s.min_failure_sample and len(failed) / i > s.max_failure_rate:
                        raise ExtractionAbortedError(
                            f"{len(failed)} of {i} chunks failed "
                            f"(limit {s.max_failure_rate:.0%}); aborting. First: {failed[:5]}"
                        ) from exc
                    continue
                results[chunk.chunk_id] = res
                f.write(json.dumps({"chunk_id": chunk.chunk_id, "result": res.model_dump()}) + "\n")
                f.flush()
                logger.info(
                    "[%d/%d] %s -> %d entities, %d rels",
                    i,
                    len(todo),
                    chunk.chunk_id,
                    len(res.entities),
                    len(res.relationships),
                )
                time.sleep(self._settings.extraction_delay_s)

        if failed:
            logger.warning("%d chunks failed (rerun to retry): %s", len(failed), failed[:10])
        return results
