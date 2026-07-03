"""Incremental ingestion (repo plan #13): only changed files are re-embedded.

Embedding is the slow step on a big corpus; unchanged files must be skipped
by comparing stored content hashes.
"""

from pathlib import Path

import pytest

import rag
from rag import index_docs


@pytest.fixture()
def temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(rag, "DB_DIR", str(tmp_path / "db"))


@pytest.fixture()
def counting_embed(monkeypatch: pytest.MonkeyPatch):
    """Fake embeddings that record every batch call."""
    calls: list[list[str]] = []

    def embed(text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 0.0]

    def embed_batch(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [embed(t) for t in texts]

    monkeypatch.setattr(rag, "embed", embed)
    monkeypatch.setattr(rag, "embed_batch", embed_batch)
    return calls


@pytest.fixture()
def docs_tree(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    (docs / "pytorch").mkdir(parents=True)
    (docs / "pytorch" / "tensors.md").write_text(
        "# Tensors\n\ntorch tensors are arrays.", encoding="utf-8"
    )
    (docs / "python").mkdir()
    (docs / "python" / "lists.md").write_text(
        "# Lists\n\nLists hold items.", encoding="utf-8"
    )
    return docs


def indexed_paths() -> set[str]:
    collection = rag.get_client().get_collection(rag.COLLECTION_NAME)
    records = collection.get(include=["metadatas"])
    return {m["path"] for m in records["metadatas"]}


def test_reingest_of_unchanged_docs_embeds_nothing(
    counting_embed, temp_db, docs_tree: Path
) -> None:
    index_docs(docs_dir=docs_tree)
    first_run_calls = len(counting_embed)

    count = index_docs(docs_dir=docs_tree)

    assert first_run_calls == 2
    assert len(counting_embed) == first_run_calls  # no new embedding calls
    assert count == 0


def test_changed_file_is_reembedded_alone(
    counting_embed, temp_db, docs_tree: Path
) -> None:
    index_docs(docs_dir=docs_tree)
    counting_embed.clear()

    (docs_tree / "pytorch" / "tensors.md").write_text(
        "# Tensors\n\ntorch tensors are UPDATED arrays.", encoding="utf-8"
    )
    index_docs(docs_dir=docs_tree)

    assert len(counting_embed) == 1
    assert any("UPDATED" in text for text in counting_embed[0])

    collection = rag.get_client().get_collection(rag.COLLECTION_NAME)
    docs = collection.get(include=["documents"])["documents"]
    assert any("UPDATED" in d for d in docs)
    assert not any("are arrays" in d for d in docs)


def test_deleted_file_leaves_the_index(
    counting_embed, temp_db, docs_tree: Path
) -> None:
    index_docs(docs_dir=docs_tree)

    (docs_tree / "python" / "lists.md").unlink()
    index_docs(docs_dir=docs_tree)

    assert all("lists.md" not in p for p in indexed_paths())


def test_new_file_is_added_without_touching_the_rest(
    counting_embed, temp_db, docs_tree: Path
) -> None:
    index_docs(docs_dir=docs_tree)
    counting_embed.clear()

    (docs_tree / "python" / "dicts.md").write_text(
        "# Dicts\n\nDicts map keys.", encoding="utf-8"
    )
    index_docs(docs_dir=docs_tree)

    assert len(counting_embed) == 1
    assert any("dicts.md" in p for p in indexed_paths())


def test_full_rebuild_reembeds_everything(
    counting_embed, temp_db, docs_tree: Path
) -> None:
    index_docs(docs_dir=docs_tree)
    counting_embed.clear()

    count = index_docs(docs_dir=docs_tree, full=True)

    assert len(counting_embed) == 2
    assert count == 2
