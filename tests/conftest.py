"""Keep every test off the real repo-root data files.

``index_docs`` and ``index_code`` can rebuild manifests through module-global
paths. This autouse fixture redirects those import-time bindings before every
test, so an incomplete per-test database fixture cannot overwrite a real index.
"""
from pathlib import Path

import pytest

import docs_cli
import paths
import rag


@pytest.fixture(autouse=True)
def _isolate_data_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_dir = str(tmp_path / "chroma_db")
    manifest_path = tmp_path / "manifest.jsonl"
    code_manifest_path = tmp_path / "code-manifest.jsonl"

    monkeypatch.setattr(rag, "DB_DIR", db_dir)
    monkeypatch.setattr(rag, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(rag, "CODE_MANIFEST_PATH", code_manifest_path)

    # Independent import-time bindings of the same paths need separate patches.
    monkeypatch.setattr(paths, "DB_DIR", db_dir)
    monkeypatch.setattr(paths, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(docs_cli, "MANIFEST_PATH", manifest_path)
