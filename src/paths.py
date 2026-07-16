"""Anchors every data path to the project directory, and loads .env.

The installed `lca` command runs from arbitrary cwds; without this module a
run from ~/school/project would create a second empty chroma_db there.
LCA_HOME overrides the anchor for tests and future multi-index setups.

The .env load lives here because paths is imported before `ollama` in every
entry point — the ollama package builds its default client (capturing
OLLAMA_HOST) the moment it is imported, so loading any later is too late.
"""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    """Set KEY=VALUE lines from a .env file, never overriding the real env."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


# The .env sits next to pyproject.toml, not under LCA_HOME: LCA_HOME itself
# may be defined inside it.
_load_env_file(_REPO_ROOT / ".env")

ENV_FILE = _REPO_ROOT / ".env"

PROJECT_ROOT = Path(os.getenv("LCA_HOME", _REPO_ROOT))

DOCS_DIR = PROJECT_ROOT / "docs"
DB_DIR = str(PROJECT_ROOT / "chroma_db")
MANIFEST_PATH = Path(os.getenv("RAG_MANIFEST_PATH", PROJECT_ROOT / "manifest.jsonl"))
HISTORY_FILE = PROJECT_ROOT / "chat_history.json"
ASK_HISTORY_FILE = PROJECT_ROOT / ".ask_history"
