"""The manifest sidecar: retrieval's lexical index, rebuilt on every ingest."""
import threading
from pathlib import Path

import manifest
import rag


def test_write_and_load_roundtrip(tmp_path):
    records = [{"id": "a-0", "tokens": ["tensor", "cuda"]}]
    path = tmp_path / "manifest.jsonl"

    manifest.write_manifest(records, path)

    assert manifest.load_manifest(path) == records
    assert not list(tmp_path.glob(".manifest-*.tmp"))


def test_load_missing_manifest_returns_empty(tmp_path):
    assert manifest.load_manifest(tmp_path / "nope.jsonl") == []


def test_write_manifest_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "manifest.jsonl"

    manifest.write_manifest([{"id": "a-0"}], path)

    assert manifest.load_manifest(path) == [{"id": "a-0"}]


def test_concurrent_manifest_writes_do_not_collide(tmp_path, monkeypatch):
    path = tmp_path / "manifest.jsonl"
    original_replace = Path.replace
    barrier = threading.Barrier(2)

    def synchronized_replace(self, target):
        barrier.wait(timeout=5)
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", synchronized_replace)
    errors = []

    def write(records):
        try:
            manifest.write_manifest(records, path)
        except Exception as error:
            errors.append(error)

    first = threading.Thread(target=write, args=([{"id": "first"}],))
    second = threading.Thread(target=write, args=([{"id": "second"}],))
    first.start()
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)

    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert manifest.load_manifest(path) in ([{"id": "first"}], [{"id": "second"}])
    assert not list(tmp_path.glob(".manifest-*.tmp"))


def test_serialization_failure_preserves_target_and_cleans_temp(tmp_path):
    path = tmp_path / "manifest.jsonl"
    original = [{"id": "original"}]
    manifest.write_manifest(original, path)

    try:
        manifest.write_manifest([{"bad": object()}], path)
    except TypeError:
        pass
    else:
        raise AssertionError("expected a JSON serialization failure")

    assert manifest.load_manifest(path) == original
    assert not list(tmp_path.glob(".manifest-*.tmp"))


def test_replace_failure_preserves_target_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "manifest.jsonl"
    original = [{"id": "original"}]
    manifest.write_manifest(original, path)

    def broken_replace(self, target):
        raise OSError("replace unavailable")

    monkeypatch.setattr(Path, "replace", broken_replace)

    try:
        manifest.write_manifest([{"id": "replacement"}], path)
    except OSError as error:
        assert "replace unavailable" in str(error)
    else:
        raise AssertionError("expected replace to fail")

    assert manifest.load_manifest(path) == original
    assert not list(tmp_path.glob(".manifest-*.tmp"))


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


class _SpyCollection:
    """Duck-typed collection that records .get() calls."""

    def __init__(self, ids, documents, metadatas, distances):
        self._data = dict(zip(ids, zip(documents, metadatas)))
        self._query = {"ids": [ids[:2]], "distances": [distances[:2]]}
        self.get_calls = []

    def query(self, **kwargs):
        return self._query

    def get(self, ids=None, where=None, include=None):
        self.get_calls.append({"ids": ids, "where": where})
        picked = ids if ids is not None else list(self._data)
        return {
            "ids": picked,
            "documents": [self._data[i][0] for i in picked],
            "metadatas": [self._data[i][1] for i in picked],
        }


def _fake_client(collection):
    class _Client:
        def get_collection(self, name):
            return collection

    return _Client()


def test_hybrid_retrieve_uses_manifest_not_full_corpus(tmp_path, monkeypatch):
    ids = ["a-0", "b-0", "c-0"]
    documents = ["torch tensor basics", "python lists", "cuda devices"]
    metadatas = [{"source": "pytorch"}, {"source": "python"}, {"source": "pytorch"}]

    manifest_path = tmp_path / "manifest.jsonl"
    manifest.write_manifest(
        [
            {"id": i, "source": m["source"], "tokens": rag._tokenize(d)}
            for i, d, m in zip(ids, documents, metadatas)
        ],
        manifest_path,
    )
    monkeypatch.setattr(rag, "MANIFEST_PATH", manifest_path)
    rag._manifest_cache.clear()

    spy = _SpyCollection(ids, documents, metadatas, [0.2, 0.3])
    monkeypatch.setattr(rag, "get_client", lambda: _fake_client(spy))
    monkeypatch.setattr(rag, "embed", lambda text: [0.0, 0.0, 1.0])

    results = rag.retrieve("torch tensor")

    assert results["ids"][0]  # got fused results
    # The only .get() calls carry explicit ids - never a full-corpus fetch.
    assert spy.get_calls, "expected an ids fetch for the fused results"
    assert all(call["ids"] is not None for call in spy.get_calls)


def test_bm25_cache_invalidates_on_manifest_change(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest.write_manifest(
        [{"id": "a-0", "source": "python", "tokens": ["lists"]}], manifest_path
    )
    monkeypatch.setattr(rag, "MANIFEST_PATH", manifest_path)
    rag._manifest_cache.clear()

    first = rag._bm25_for_source(None)
    again = rag._bm25_for_source(None)
    assert first is again  # cached: same tuple object back

    import os as _os

    manifest.write_manifest(
        [{"id": "b-0", "source": "python", "tokens": ["dicts"]}], manifest_path
    )
    _os.utime(manifest_path, ns=(1, 1))  # force a different mtime_ns

    rebuilt = rag._bm25_for_source(None)
    assert rebuilt is not first
    assert rebuilt[1] == ["b-0"]


def test_bm25_ranking_tolerates_all_tokenless_documents():
    assert rag._bm25_ranking("tensor", ["a-0"], ["and the or"]) == []


def test_manifest_bm25_cache_tolerates_all_tokenless_records(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest.write_manifest(
        [{"id": "a-0", "source": "python", "tokens": []}], manifest_path
    )
    monkeypatch.setattr(rag, "MANIFEST_PATH", manifest_path)
    rag._manifest_cache.clear()

    assert rag._bm25_for_source("python") is None


def test_bm25_cache_entries_are_isolated_by_manifest_path(tmp_path):
    docs_manifest = tmp_path / "manifest.jsonl"
    code_manifest = tmp_path / "code-manifest.jsonl"
    manifest.write_manifest(
        [{"id": "docs-0", "source": "numpy", "tokens": ["broadcasting"]}],
        docs_manifest,
    )
    manifest.write_manifest(
        [{"id": "code-0", "source": "project", "tokens": ["retrieve"]}],
        code_manifest,
    )
    rag._manifest_cache.clear()

    docs = rag._bm25_for_source(None, manifest_path=docs_manifest)
    code = rag._bm25_for_source(None, manifest_path=code_manifest)
    docs_again = rag._bm25_for_source(None, manifest_path=docs_manifest)

    assert docs is docs_again
    assert docs is not None and docs[1] == ["docs-0"]
    assert code is not None and code[1] == ["code-0"]
    assert set(rag._manifest_cache) == {str(docs_manifest), str(code_manifest)}
