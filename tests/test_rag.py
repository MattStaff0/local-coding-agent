from pathlib import Path

import pytest

import rag
from rag import index_docs, list_sources, retrieve, source_for


@pytest.fixture()
def fake_embed(monkeypatch: pytest.MonkeyPatch):
    """Replace the Ollama embedding call with a cheap deterministic vector."""

    def embed(text: str) -> list[float]:
        return [float(len(text) % 7), float(text.count("torch")), 1.0]

    monkeypatch.setattr(rag, "embed", embed)
    return embed


@pytest.fixture()
def temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point Chroma at a throwaway directory so tests never touch chroma_db/."""
    monkeypatch.setattr(rag, "DB_DIR", str(tmp_path / "db"))


@pytest.fixture()
def docs_tree(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    (docs / "pytorch").mkdir(parents=True)
    (docs / "python").mkdir()
    (docs / "pytorch" / "tensors.md").write_text(
        "# Tensors\n\ntorch tensors are arrays.", encoding="utf-8"
    )
    (docs / "python" / "lists.md").write_text(
        "# Lists\n\nLists hold items.", encoding="utf-8"
    )
    (docs / "notes.md").write_text("# Notes\n\nLoose top-level doc.", encoding="utf-8")
    return docs


def test_source_for_uses_top_level_folder(docs_tree: Path) -> None:
    assert source_for(docs_tree / "pytorch" / "tensors.md", docs_tree) == "pytorch"


def test_source_for_flat_file_defaults_to_general(docs_tree: Path) -> None:
    assert source_for(docs_tree / "notes.md", docs_tree) == "general"


def test_index_docs_stores_source_metadata(
    fake_embed, temp_db, docs_tree: Path
) -> None:
    count = index_docs(docs_dir=docs_tree)

    assert count == 3
    collection = rag.get_client().get_collection(rag.COLLECTION_NAME)
    records = collection.get(include=["metadatas"])
    sources = {m["source"] for m in records["metadatas"]}
    assert sources == {"pytorch", "python", "general"}


def test_index_docs_stores_heading_path_and_breadcrumbed_text(
    fake_embed, temp_db, docs_tree: Path
) -> None:
    index_docs(docs_dir=docs_tree)

    collection = rag.get_client().get_collection(rag.COLLECTION_NAME)
    records = collection.get(include=["metadatas", "documents"])

    by_heading = dict(zip(
        [m["heading"] for m in records["metadatas"]], records["documents"]
    ))
    assert "Tensors" in by_heading
    assert by_heading["Tensors"].startswith("Tensors")
    assert "torch tensors are arrays." in by_heading["Tensors"]


def test_retrieve_filters_by_source(fake_embed, temp_db, docs_tree: Path) -> None:
    index_docs(docs_dir=docs_tree)

    results = retrieve("anything", n_results=3, source="python")

    metadatas = results["metadatas"][0]
    assert metadatas, "expected at least one result"
    assert all(m["source"] == "python" for m in metadatas)


def test_retrieve_without_source_searches_everything(
    fake_embed, temp_db, docs_tree: Path
) -> None:
    index_docs(docs_dir=docs_tree)

    results = retrieve("anything", n_results=3)

    sources = {m["source"] for m in results["metadatas"][0]}
    assert len(sources) > 1


def test_list_sources_returns_indexed_sources_sorted(
    fake_embed, temp_db, docs_tree: Path
) -> None:
    index_docs(docs_dir=docs_tree)

    assert list_sources() == ["general", "python", "pytorch"]
