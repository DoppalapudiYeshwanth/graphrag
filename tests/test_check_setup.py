"""The preflight checks, with a fake Groq client (no network, no real keys)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
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


# ── the deployment probe refuses to test the wrong target ────────────────────
PROBE = str(Path(__file__).resolve().parent.parent / "deploy" / "streamlit_probe.py")


def _probe(uri: str):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(PROBE, default_timeout=30)
    app.secrets["NEO4J_URI"] = uri
    app.secrets["NEO4J_USERNAME"] = "neo4j"
    app.secrets["NEO4J_PASSWORD"] = "placeholder"
    return app.run()


@pytest.mark.parametrize(
    "uri",
    ["bolt://localhost:7687", "neo4j://127.0.0.1:7687", "bolt://192.168.1.20:7687", "bolt://[::1]:7687"],
)
def test_probe_stops_with_a_clear_message_for_local_addresses(uri: str) -> None:
    app = _probe(uri)
    assert any("your own computer" in e.value for e in app.error)
    assert len(app.success) == 0  # it must not run network checks against a wrong target


def test_probe_missing_secrets_are_named() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(PROBE, default_timeout=30).run()
    assert any("Missing secrets" in e.value and "NEO4J_URI" in e.value for e in app.error)
