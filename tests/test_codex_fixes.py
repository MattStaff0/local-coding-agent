"""Regression tests for the 2026-07-03 external (Codex) review findings."""
from pathlib import Path

import rag


def _fake_embed_batch(calls):
    def fake(texts):
        calls.append(list(texts))
        return [[0.0, 0.0, 1.0] for _ in texts]

    return fake


def test_heading_only_doc_never_calls_embed(tmp_path, monkeypatch):
    docs = tmp_path / "docs" / "python"
    docs.mkdir(parents=True)
    (docs / "empty.md").write_text("# Just a heading\n", encoding="utf-8")

    calls: list[list[str]] = []
    monkeypatch.setattr(rag, "embed_batch", _fake_embed_batch(calls))
    monkeypatch.setattr(rag, "DB_DIR", str(tmp_path / "db"))

    added = rag.index_docs(tmp_path / "docs", full=True)

    assert added == 0
    assert calls == []  # embed_batch([]) would be a rejected-by-endpoint call
