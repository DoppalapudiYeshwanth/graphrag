"""The preflight checks, with a fake Groq client (no network, no real keys)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
from groq import APIConnectionError, AuthenticationError

from graphrag.config import get_settings
from scripts.check_setup import check_groq, check_python


class FakeGroq:
    def __init__(self, models: list[str] | None = None, error: Exception | None = None) -> None:
        self._models, self._error = models or [], error
        self.models = SimpleNamespace(list=self._list)

    def _list(self):
        if self._error:
            raise self._error
        return SimpleNamespace(data=[SimpleNamespace(id=m) for m in self._models])


def _request() -> httpx.Request:
    return httpx.Request("GET", "https://example.test")


def test_expired_key_is_reported_clearly(settings_env) -> None:
    settings_env()
    error = AuthenticationError("expired", response=httpx.Response(401, request=_request()), body=None)
    ok, detail = check_groq(get_settings(), FakeGroq(error=error))
    assert not ok and "rejected the API key" in detail and "console.groq.com/keys" in detail


def test_network_failure_is_not_blamed_on_the_key(settings_env) -> None:
    settings_env()
    ok, detail = check_groq(get_settings(), FakeGroq(error=APIConnectionError(request=_request())))
    assert not ok and "network" in detail and "key" not in detail.lower().replace("api", "")


def test_missing_configured_models_are_named(settings_env) -> None:
    settings = settings_env(GROQ_MODEL="model-a", GROQ_FALLBACK_MODELS="model-b", ANSWER_MODEL="model-c")
    ok, detail = check_groq(settings, FakeGroq(models=["model-a"]))
    assert not ok and "model-b" in detail and "model-c" in detail and "model-a" not in detail.split(":")[-1]


def test_all_models_available_passes(settings_env) -> None:
    settings = settings_env(GROQ_MODEL="model-a", GROQ_FALLBACK_MODELS="model-b", ANSWER_MODEL="model-c")
    ok, detail = check_groq(settings, FakeGroq(models=["model-a", "model-b", "model-c", "extra"]))
    assert ok and "3 configured models" in detail


def test_python_check_reports_a_version() -> None:
    ok, detail = check_python()
    assert ok and "Python 3." in detail
