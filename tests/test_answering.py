"""Answer generation: output caps, the model fallback chain, and fast failure on a bad key."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from groq import AuthenticationError, BadRequestError, NotFoundError, RateLimitError

from graphrag.retrieval import GraphRAG, RetrievalContext, RetrievedChunk

# The exact refusal a free account got from qwen3.8-27b when max_tokens was 1024.
OTPM_MESSAGE = (
    "Error code: 429 - {'error': {'message': \"Request too large for model `qwen/qwen3.8-27b` in "
    "organization `org_x` service tier `on_demand` on output tokens per minute (OTPM): Limit 1000, "
    "Requested 1024. The request's expected output tokens exceed the enforced limit; reduce max_tokens "
    "(or the request's expected output) and try again.\", 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)
TPM_MESSAGE = "Rate limit reached on tokens per minute (TPM): Limit 8000, Used 7900, Requested 500."


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://example.test"))


def err(cls, message: str):
    status = {RateLimitError: 429, AuthenticationError: 401, NotFoundError: 404, BadRequestError: 400}[cls]
    return cls(message, response=_response(status), body=None)


class FakeLLM:
    """Scripted per-model outcomes: an exception to raise, or text to return."""

    def __init__(self, outcomes: dict[str, list] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        queue = self.outcomes.get(kwargs["model"], [])
        outcome = queue.pop(0) if queue else f"answer from {kwargs['model']}"
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))])


def make_rag(settings_env, llm: FakeLLM, with_context: bool = True, **env: str) -> GraphRAG:
    defaults = {"GROQ_MODEL": "model-a", "GROQ_FALLBACK_MODELS": "model-b,model-c", "ANSWER_MODEL": "model-x"}
    settings_env(**{**defaults, **env})
    rag = GraphRAG(store=SimpleNamespace())
    rag._llm = llm
    chunks = [RetrievedChunk(chunk_id="D::0", doc_id="D", text="some passage")] if with_context else []
    rag._retriever = SimpleNamespace(
        retrieve=lambda question, scopes=None: RetrievalContext(chunks=chunks, facts=[])
    )
    return rag


def models_called(llm: FakeLLM) -> list[str]:
    return [c["model"] for c in llm.calls]


def test_default_output_cap_is_below_the_known_free_tier_limit(settings_env) -> None:
    assert settings_env().answer_max_tokens < 1000


def test_normal_answer_uses_the_answer_model_first(settings_env) -> None:
    llm = FakeLLM()
    result = make_rag(settings_env, llm).ask("What is X?")
    assert result.answer == "answer from model-x"
    assert models_called(llm) == ["model-x"]


# ── the production incident: output cap exceeded ─────────────────────────────
def test_output_cap_refusal_is_retried_once_with_a_smaller_limit(settings_env) -> None:
    llm = FakeLLM({"model-x": [err(RateLimitError, OTPM_MESSAGE)]})
    result = make_rag(settings_env, llm, ANSWER_MAX_TOKENS="1024").ask("What is X?")
    assert result.answer == "answer from model-x"
    assert [(c["model"], c["max_tokens"]) for c in llm.calls] == [("model-x", 1024), ("model-x", 900)]


def test_a_learned_cap_is_remembered_so_the_next_question_does_not_fail_again(settings_env) -> None:
    llm = FakeLLM({"model-x": [err(RateLimitError, OTPM_MESSAGE)]})
    rag = make_rag(settings_env, llm, ANSWER_MAX_TOKENS="1024")
    rag.ask("First?")
    llm.calls.clear()
    rag.ask("Second?")
    assert [(c["model"], c["max_tokens"]) for c in llm.calls] == [("model-x", 900)]


# ── fallback chain ───────────────────────────────────────────────────────────
def test_other_rate_limits_fall_through_to_the_next_model(settings_env) -> None:
    llm = FakeLLM({"model-x": [err(RateLimitError, TPM_MESSAGE)]})
    result = make_rag(settings_env, llm).ask("What is X?")
    assert result.answer == "answer from model-a"
    assert models_called(llm) == ["model-x", "model-a"]


def test_a_retired_model_is_skipped(settings_env) -> None:
    llm = FakeLLM({"model-x": [err(NotFoundError, "model_not_found")]})
    assert make_rag(settings_env, llm).ask("What is X?").answer == "answer from model-a"


def test_a_decommissioned_model_reported_as_bad_request_is_skipped(settings_env) -> None:
    llm = FakeLLM({"model-x": [err(BadRequestError, "model_decommissioned")]})
    assert make_rag(settings_env, llm).ask("What is X?").answer == "answer from model-a"


def test_chain_order_is_answer_model_then_the_extraction_chain(settings_env) -> None:
    llm = FakeLLM(
        {
            "model-x": [err(RateLimitError, TPM_MESSAGE)],
            "model-a": [err(RateLimitError, TPM_MESSAGE)],
            "model-b": [err(RateLimitError, TPM_MESSAGE)],
        }
    )
    assert make_rag(settings_env, llm).ask("What is X?").answer == "answer from model-c"
    assert models_called(llm) == ["model-x", "model-a", "model-b", "model-c"]


def test_all_models_exhausted_raises_the_last_error(settings_env) -> None:
    llm = FakeLLM(
        {m: [err(RateLimitError, TPM_MESSAGE)] for m in ("model-x", "model-a", "model-b", "model-c")}
    )
    with pytest.raises(RateLimitError):
        make_rag(settings_env, llm).ask("What is X?")
    assert len(llm.calls) == 4


# ── failures that must NOT be hidden by falling back ─────────────────────────
def test_a_rejected_api_key_fails_immediately(settings_env) -> None:
    llm = FakeLLM({"model-x": [err(AuthenticationError, "expired_api_key")]})
    with pytest.raises(AuthenticationError):
        make_rag(settings_env, llm).ask("What is X?")
    assert models_called(llm) == ["model-x"]  # no pointless retries on other models


def test_a_genuine_bad_request_is_raised_not_swallowed(settings_env) -> None:
    llm = FakeLLM({"model-x": [err(BadRequestError, "messages must not be empty")]})
    with pytest.raises(BadRequestError):
        make_rag(settings_env, llm).ask("What is X?")
    assert models_called(llm) == ["model-x"]


# ── request details ──────────────────────────────────────────────────────────
def test_reasoning_effort_is_low_only_for_gpt_oss_models(settings_env) -> None:
    llm = FakeLLM()
    make_rag(settings_env, llm, ANSWER_MODEL="openai/gpt-oss-20b").ask("What is X?")
    assert llm.calls[0]["reasoning_effort"] == "low"

    llm2 = FakeLLM()
    make_rag(settings_env, llm2).ask("What is X?")
    assert "reasoning_effort" not in llm2.calls[0]


def test_think_blocks_are_removed_from_the_answer(settings_env) -> None:
    llm = FakeLLM({"model-x": ["<think>private reasoning</think>The real answer."]})
    assert make_rag(settings_env, llm).ask("What is X?").answer == "The real answer."


def test_no_context_means_no_model_call(settings_env) -> None:
    llm = FakeLLM()
    result = make_rag(settings_env, llm, with_context=False).ask("What is X?")
    assert result.answer == "No relevant context found." and llm.calls == []
