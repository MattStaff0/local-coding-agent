"""Model names must be overridable via environment variables (repo plan #3).

The constants are read from the environment at import time, so these tests
reload the modules after changing the env, and re-reload at teardown so the
rest of the suite sees the defaults again.
"""

import importlib

import pytest

import agent
import rag


@pytest.fixture()
def reload_after(monkeypatch: pytest.MonkeyPatch):
    """Restore default constants after the env-override reloads."""
    yield monkeypatch
    monkeypatch.undo()
    importlib.reload(rag)
    importlib.reload(agent)


def test_chat_model_defaults_without_env() -> None:
    assert rag.CHAT_MODEL == "qwen2.5-coder:3b"
    assert rag.EMBED_MODEL == "nomic-embed-text"
    assert agent.AGENT_MODEL == "qwen2.5-coder:3b"


def test_ollama_chat_model_env_overrides_chat_model(reload_after) -> None:
    reload_after.setenv("OLLAMA_CHAT_MODEL", "qwen3:8b")

    importlib.reload(rag)

    assert rag.CHAT_MODEL == "qwen3:8b"


def test_ollama_embed_model_env_overrides_embed_model(reload_after) -> None:
    reload_after.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-code")

    importlib.reload(rag)

    assert rag.EMBED_MODEL == "nomic-embed-code"


def test_agent_model_follows_ollama_chat_model_env(reload_after) -> None:
    reload_after.setenv("OLLAMA_CHAT_MODEL", "qwen3:8b")

    importlib.reload(agent)

    assert agent.AGENT_MODEL == "qwen3:8b"
