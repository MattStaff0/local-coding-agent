"""Regression tests for the 2026-07-03 external (Codex) review findings."""
from pathlib import Path

from agent import TOOL_SCHEMAS
from agent_tools import read_file
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


def _index(tmp_path, monkeypatch, docs_dir, calls):
    monkeypatch.setattr(rag, "embed_batch", _fake_embed_batch(calls))
    monkeypatch.setattr(rag, "DB_DIR", str(tmp_path / "db"))
    return rag.index_docs(docs_dir)


def test_reingest_from_relative_path_is_a_noop(tmp_path, monkeypatch):
    docs = tmp_path / "docs" / "python"
    docs.mkdir(parents=True)
    (docs / "a.md").write_text("# T\n\nBody text here.\n", encoding="utf-8")

    calls: list[list[str]] = []
    assert _index(tmp_path, monkeypatch, tmp_path / "docs", calls) > 0

    # Same docs, addressed relatively: must be recognized as unchanged.
    monkeypatch.chdir(tmp_path)
    added = _index(tmp_path, monkeypatch, Path("docs"), calls)

    assert added == 0


def test_metadata_records_posix_relative_path(tmp_path, monkeypatch):
    docs = tmp_path / "docs" / "python"
    docs.mkdir(parents=True)
    (docs / "a.md").write_text("# T\n\nBody text here.\n", encoding="utf-8")

    calls: list[list[str]] = []
    _index(tmp_path, monkeypatch, tmp_path / "docs", calls)

    collection = rag.get_client().get_collection(rag.COLLECTION_NAME)
    metadata = collection.get(include=["metadatas"])["metadatas"][0]

    assert metadata["relative_path"] == "python/a.md"


def test_legacy_index_without_relative_path_triggers_full_rebuild(
    tmp_path, monkeypatch, capsys
):
    docs = tmp_path / "docs" / "python"
    docs.mkdir(parents=True)
    doc = docs / "a.md"
    doc.write_text("# T\n\nBody text here.\n", encoding="utf-8")

    calls: list[list[str]] = []
    monkeypatch.setattr(rag, "embed_batch", _fake_embed_batch(calls))
    monkeypatch.setattr(rag, "DB_DIR", str(tmp_path / "db"))

    # Build a "legacy" collection by hand: cosine space but no relative_path.
    collection = rag.reset_collection(rag.get_client())
    collection.add(
        ids=["python__a.md-0"],
        documents=["old"],
        embeddings=[[0.0, 0.0, 1.0]],
        metadatas=[
            {
                "source": "python",
                "path": str(doc),
                "heading": "T",
                "chunk_index": 0,
                "file_hash": "stale",
            }
        ],
    )

    added = rag.index_docs(tmp_path / "docs")

    assert "full rebuild" in capsys.readouterr().out.lower()
    assert added > 0  # full=True return counts every chunk


def test_read_file_clamps_nonpositive_start_line(tmp_path):
    (tmp_path / "f.py").write_text("first\nsecond\n", encoding="utf-8")

    assert read_file(tmp_path, "f.py", 0) == read_file(tmp_path, "f.py", 1)
    assert read_file(tmp_path, "f.py", -3) == read_file(tmp_path, "f.py", 1)
    assert "1: first" in read_file(tmp_path, "f.py", 0)


def test_read_file_schema_declares_minimum_start_line():
    read_schema = next(
        s for s in TOOL_SCHEMAS if s["function"]["name"] == "read_file"
    )
    start_line = read_schema["function"]["parameters"]["properties"]["start_line"]

    assert start_line["minimum"] == 1
