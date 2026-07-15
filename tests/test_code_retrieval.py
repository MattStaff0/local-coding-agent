"""Retrieval parametrized by collection + manifest: code and docs stay isolated."""
import manifest
import rag


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


def _fake_client(collections):
    class _Client:
        def get_collection(self, name):
            return collections[name]

    return _Client()


def _write_manifest(path, ids, documents, source):
    manifest.write_manifest(
        [
            {"id": i, "source": source, "tokens": rag._tokenize(d)}
            for i, d in zip(ids, documents)
        ],
        path,
    )


def test_code_retrieval_uses_its_own_collection_and_manifest(tmp_path, monkeypatch):
    ids = ["rag.py-0", "rag.py-1", "ask.py-0"]
    documents = ["def retrieve hybrid", "def rrf fusion scores", "def chat loop"]
    metadatas = [{"source": "lca"}, {"source": "lca"}, {"source": "lca"}]

    code_manifest = tmp_path / "code-manifest.jsonl"
    _write_manifest(code_manifest, ids, documents, "lca")

    # Garbage at the docs manifest path must have zero effect on code retrieval.
    docs_manifest = tmp_path / "manifest.jsonl"
    docs_manifest.write_text("{not json at all", encoding="utf-8")
    monkeypatch.setattr(rag, "MANIFEST_PATH", docs_manifest)

    spy = _SpyCollection(ids, documents, metadatas, [0.2, 0.3])
    monkeypatch.setattr(
        rag, "get_client", lambda: _fake_client({"local_code": spy})
    )
    monkeypatch.setattr(rag, "embed", lambda text: [0.0, 0.0, 1.0])

    results = rag.retrieve(
        "rrf fusion",
        collection_name="local_code",
        manifest_path=code_manifest,
    )

    assert results["ids"][0]
    # Fast path: only explicit-ids fetches, never a full-corpus get.
    assert spy.get_calls and all(c["ids"] is not None for c in spy.get_calls)


def test_manifest_caches_are_isolated_per_path(tmp_path, monkeypatch):
    docs_ids = ["d-0"]
    code_ids = ["c-0"]
    docs_docs = ["torch tensor basics"]
    code_docs = ["def retrieve hybrid fusion"]

    docs_manifest = tmp_path / "manifest.jsonl"
    code_manifest = tmp_path / "code-manifest.jsonl"
    _write_manifest(docs_manifest, docs_ids, docs_docs, "pytorch")
    _write_manifest(code_manifest, code_ids, code_docs, "lca")

    docs_spy = _SpyCollection(docs_ids, docs_docs, [{"source": "pytorch"}], [0.2])
    code_spy = _SpyCollection(code_ids, code_docs, [{"source": "lca"}], [0.2])
    monkeypatch.setattr(
        rag,
        "get_client",
        lambda: _fake_client({"local_docs": docs_spy, "local_code": code_spy}),
    )
    monkeypatch.setattr(rag, "embed", lambda text: [0.0, 0.0, 1.0])
    monkeypatch.setattr(rag, "MANIFEST_PATH", docs_manifest)

    docs_results = rag.retrieve("torch tensor")
    code_results = rag.retrieve(
        "retrieve fusion",
        collection_name="local_code",
        manifest_path=code_manifest,
    )

    assert docs_results["ids"][0] == ["d-0"]
    assert code_results["ids"][0] == ["c-0"]
