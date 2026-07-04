"""Anchors every data path to the project directory.

The installed `lca` command runs from arbitrary cwds; without this module a
run from ~/school/project would create a second empty chroma_db there.
LCA_HOME overrides the anchor for tests and future multi-index setups.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("LCA_HOME", Path(__file__).resolve().parent.parent))

DOCS_DIR = PROJECT_ROOT / "docs"
DB_DIR = str(PROJECT_ROOT / "chroma_db")
MANIFEST_PATH = Path(os.getenv("RAG_MANIFEST_PATH", PROJECT_ROOT / "manifest.jsonl"))
HISTORY_FILE = PROJECT_ROOT / "chat_history.json"
ASK_HISTORY_FILE = PROJECT_ROOT / ".ask_history"
