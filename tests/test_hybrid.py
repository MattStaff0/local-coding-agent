"""Hybrid retrieval (repo plan #7): BM25 + vector rankings fused with RRF.

Exact identifiers like `nn.Module` are where embeddings get fuzzy and keyword
search shines; fusion should surface such docs even when the fake embedding
ranks them last.
"""

from pathlib import Path

import pytest

import rag


@pytest.fixture()
def temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(rag, "DB_DIR", str(tmp_path / "db"))


@pytest.fixture()
def keyword_blind_embed(monkeypatch: pytest.MonkeyPatch):
    """Embeddings that cluster on the word 'tensor' and ignore identifiers."""

    def embed(text: str) -> list[float]:
        # Only one deterministic signal: does the text mention "tensor"?
        # Both tensor pages tie exactly with a tensor question, so the
        # identifier page can never sneak into a vector top-2.
        return [float("tensor" in text.lower()), 0.0, 1.0]

    monkeypatch.setattr(rag, "embed", embed)
    monkeypatch.setattr(rag, "embed_batch", lambda texts: [embed(t) for t in texts])
    return embed


@pytest.fixture()
def indexed_docs(keyword_blind_embed, temp_db, tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    (docs / "pytorch").mkdir(parents=True)
    (docs / "pytorch" / "tensors.md").write_text(
        "# Tensors\n\nA tensor is a tensor of tensor data.", encoding="utf-8"
    )
    (docs / "pytorch" / "more-tensors.md").write_text(
        "# More tensors\n\nAnother tensor page about tensor tensor math.",
        encoding="utf-8",
    )
    (docs / "pytorch" / "modules.md").write_text(
        "# Modules\n\nSubclass nn.Module to build networks.", encoding="utf-8"
    )
    rag.index_docs(docs_dir=docs)
    return docs


def test_tokenize_lowercases_and_splits_identifiers() -> None:
    assert rag._tokenize("Subclass nn.Module!") == ["subclass", "nn", "module"]


def test_rrf_prefers_docs_ranked_well_in_both_lists() -> None:
    scores = rag._rrf_scores([["a", "b", "c"], ["b", "c", "a"]])

    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]


def test_hybrid_surfaces_exact_identifier_match_the_vector_ranking_buries(
    indexed_docs,
) -> None:
    vector_only = rag.retrieve("what is nn.Module tensor?", n_results=2, mode="vector")
    hybrid = rag.retrieve("what is nn.Module tensor?", n_results=2, mode="hybrid")

    vector_paths = {m["path"] for m in vector_only["metadatas"][0]}
    hybrid_paths = {m["path"] for m in hybrid["metadatas"][0]}

    assert not any(p.endswith("modules.md") for p in vector_paths)
    assert any(p.endswith("modules.md") for p in hybrid_paths)


def test_hybrid_is_the_default_mode(indexed_docs) -> None:
    default = rag.retrieve("what is nn.Module tensor?", n_results=2)
    hybrid = rag.retrieve("what is nn.Module tensor?", n_results=2, mode="hybrid")

    default_paths = [m["path"] for m in default["metadatas"][0]]
    hybrid_paths = [m["path"] for m in hybrid["metadatas"][0]]

    assert default_paths == hybrid_paths


def test_hybrid_respects_the_source_filter(indexed_docs, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    (docs / "python").mkdir()
    (docs / "python" / "lists.md").write_text(
        "# Lists\n\nnn.Module is mentioned here too.", encoding="utf-8"
    )
    rag.index_docs(docs_dir=docs)

    results = rag.retrieve("nn.Module", n_results=4, source="pytorch")

    assert results["metadatas"][0], "expected results"
    assert all(m["source"] == "pytorch" for m in results["metadatas"][0])


def test_hybrid_results_keep_the_chroma_shape_with_distances(indexed_docs) -> None:
    results = rag.retrieve("tensor", n_results=2)

    assert len(results["documents"][0]) == len(results["metadatas"][0])
    assert len(results["documents"][0]) == len(results["distances"][0])
    assert all(isinstance(d, float) for d in results["distances"][0])
