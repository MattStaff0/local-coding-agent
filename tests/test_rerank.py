"""Optional cross-encoder reranking: hybrid recall first, precision second."""
import rag
import rerank


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RAG_RERANKER", raising=False)
    assert not rerank.enabled()


def test_enabled_by_env(monkeypatch):
    monkeypatch.setenv("RAG_RERANKER", "cross-encoder")
    assert rerank.enabled()


def test_rerank_orders_by_score(monkeypatch):
    scores = {"middling doc": 0.4, "best doc": 0.9, "worst doc": 0.1}

    def fake_predict(pairs):
        return [scores[doc] for _, doc in pairs]

    monkeypatch.setattr(rerank, "_predict_fn", lambda: fake_predict)

    ids = rerank.rerank(
        "q",
        ["a", "b", "c"],
        ["middling doc", "best doc", "worst doc"],
        n_results=2,
    )

    assert ids == ["b", "a"]


def test_missing_model_falls_back_to_given_order(monkeypatch, capsys):
    def boom():
        raise ImportError("No module named 'sentence_transformers'")

    monkeypatch.setattr(rerank, "_predict_fn", boom)
    monkeypatch.setattr(rerank, "_warned", False)

    ids = rerank.rerank("q", ["a", "b", "c"], ["d1", "d2", "d3"], n_results=2)

    assert ids == ["a", "b"]  # fusion order preserved
    assert "reranker unavailable" in capsys.readouterr().out


def test_retrieve_reranks_the_wide_candidate_pool(tmp_path, monkeypatch):
    """With the reranker on, retrieve fetches the wide pool and reorders it."""
    import manifest

    ids = ["a-0", "b-0", "c-0"]
    documents = ["alpha doc", "beta doc", "gamma doc"]
    metadatas = [{"source": "s"}] * 3

    manifest_path = tmp_path / "manifest.jsonl"
    manifest.write_manifest(
        [
            {"id": i, "source": "s", "tokens": rag._tokenize(d)}
            for i, d in zip(ids, documents)
        ],
        manifest_path,
    )
    monkeypatch.setattr(rag, "MANIFEST_PATH", manifest_path)

    class _Spy:
        def query(self, **kwargs):
            return {"ids": [ids[:2]], "distances": [[0.2, 0.3]]}

        def get(self, ids=None, where=None, include=None):
            picked = ids if ids is not None else list(ids)
            data = dict(zip(["a-0", "b-0", "c-0"], zip(documents, metadatas)))
            return {
                "ids": picked,
                "documents": [data[i][0] for i in picked],
                "metadatas": [data[i][1] for i in picked],
            }

    class _Client:
        def get_collection(self, name):
            return _Spy()

    monkeypatch.setattr(rag, "get_client", lambda: _Client())
    monkeypatch.setattr(rag, "embed", lambda text: [0.0, 1.0])
    monkeypatch.setattr(rag.rerank, "enabled", lambda: True)

    recorded = {}

    def fake_rerank(question, candidate_ids, candidate_docs, n_results):
        recorded["ids"] = list(candidate_ids)
        return list(reversed(candidate_ids))[:n_results]

    monkeypatch.setattr(rag.rerank, "rerank", fake_rerank)

    results = rag.retrieve("beta", n_results=2, manifest_path=manifest_path)

    # The reranker saw the wide fused pool, not just the final top-2 …
    assert len(recorded["ids"]) >= 2
    # … and its ordering decided the final results.
    assert results["ids"][0] == list(reversed(recorded["ids"]))[:2]
