"""Model names must be overridable via environment variables (repo plan #3).

The constants are read from the environment at import time. Reloading modules
in-process would replace classes like EmptyIndexError and break identity for
every other test, so each case imports the module in a clean subprocess with
the environment it wants.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def import_and_print(expression: str, env_overrides: dict[str, str]) -> str:
    """Print an expression from a fresh interpreter with the given env."""
    env = os.environ.copy()
    env.pop("OLLAMA_CHAT_MODEL", None)
    env.pop("OLLAMA_EMBED_MODEL", None)
    env.update(env_overrides)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-c", f"import agent, rag; print({expression})"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=True,
    )
    return result.stdout.strip()


def test_models_default_without_env() -> None:
    line = import_and_print(
        "rag.CHAT_MODEL, rag.EMBED_MODEL, agent.AGENT_MODEL", {}
    )

    assert line == "qwen2.5-coder:3b nomic-embed-text qwen2.5-coder:3b"


def test_ollama_chat_model_env_overrides_chat_model() -> None:
    line = import_and_print("rag.CHAT_MODEL", {"OLLAMA_CHAT_MODEL": "qwen3:8b"})

    assert line == "qwen3:8b"


def test_ollama_embed_model_env_overrides_embed_model() -> None:
    line = import_and_print(
        "rag.EMBED_MODEL", {"OLLAMA_EMBED_MODEL": "nomic-embed-code"}
    )

    assert line == "nomic-embed-code"


def test_agent_model_follows_ollama_chat_model_env() -> None:
    line = import_and_print("agent.AGENT_MODEL", {"OLLAMA_CHAT_MODEL": "qwen3:8b"})

    assert line == "qwen3:8b"
