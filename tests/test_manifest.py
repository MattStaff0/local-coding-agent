"""The manifest sidecar: retrieval's lexical index, rebuilt on every ingest."""
from pathlib import Path

import manifest
import rag


def test_write_and_load_roundtrip(tmp_path):
    records = [{"id": "a-0", "tokens": ["tensor", "cuda"]}]
    path = tmp_path / "manifest.jsonl"

    manifest.write_manifest(records, path)

    assert manifest.load_manifest(path) == records
    assert not path.with_suffix(".tmp").exists()


def test_load_missing_manifest_returns_empty(tmp_path):
    assert manifest.load_manifest(tmp_path / "nope.jsonl") == []


def test_ingest_rebuilds_manifest(tmp_path, monkeypatch):
    docs = tmp_path / "docs" / "python"
    docs.mkdir(parents=True)
    (docs / "a.md").write_text("# Tensors\n\nUse torch.tensor here.\n", encoding="utf-8")

    monkeypatch.setattr(rag, "embed_batch", lambda texts: [[0.0, 0.0, 1.0]] * len(texts))
    monkeypatch.setattr(rag, "DB_DIR", str(tmp_path / "db"))
    manifest_path = tmp_path / "manifest.jsonl"
    monkeypatch.setattr(rag, "MANIFEST_PATH", manifest_path)

    rag.index_docs(tmp_path / "docs", full=True)

    records = manifest.load_manifest(manifest_path)
    assert len(records) == 1
    record = records[0]
    assert record["relative_path"] == "python/a.md"
    assert record["source"] == "python"
    assert "tensor" in record["tokens"]
    assert record["approx_tokens"] > 0
