from pathlib import Path

import chromadb.errors
import pytest

import rag
from rag import EmptyIndexError, chunk_text, index_docs, reset_collection


@pytest.fixture()
def fake_embed(monkeypatch: pytest.MonkeyPatch):
    vector = lambda text: [float(len(text) % 7), 1.0]
    monkeypatch.setattr(rag, "embed", vector)
    monkeypatch.setattr(rag, "embed_batch", lambda texts: [vector(t) for t in texts])


@pytest.fixture()
def temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(rag, "DB_DIR", str(tmp_path / "db"))


def test_chunk_text_terminates_when_overlap_exceeds_chunk_size() -> None:
    text = "word " * 200

    chunks = chunk_text(text, chunk_size=100)  # default overlap is 150 > 100

    assert chunks, "expected at least one chunk"
    assert all(len(c) <= 100 for c in chunks)
    # A sane forward step: nowhere near one chunk per character.
    assert len(chunks) < 50


def test_reset_collection_propagates_unexpected_delete_errors() -> None:
    class BrokenClient:
        def delete_collection(self, name: str) -> None:
            raise RuntimeError("database is locked")

        def create_collection(self, name: str):
            raise AssertionError("should not get here")

    with pytest.raises(RuntimeError, match="locked"):
        reset_collection(BrokenClient())


def _seed_collection(docs_dir: Path) -> int:
    """Build a real index once so later failures have something to destroy."""
    (docs_dir / "python").mkdir(parents=True, exist_ok=True)
    (docs_dir / "python" / "lists.md").write_text(
        "# Lists\n\nLists hold items.", encoding="utf-8"
    )
    return index_docs(docs_dir=docs_dir)


def _indexed_count() -> int:
    collection = rag.get_client().get_collection(rag.COLLECTION_NAME)
    return collection.count()


def test_failed_ingest_preserves_the_existing_index(
    fake_embed, temp_db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = tmp_path / "docs"
    assert _seed_collection(docs) == 1

    def exploding_embed_batch(texts: list[str]) -> list[list[float]]:
        raise ConnectionError("ollama is down")

    monkeypatch.setattr(rag, "embed_batch", exploding_embed_batch)

    with pytest.raises(ConnectionError):
        index_docs(docs_dir=docs)

    # The old index must survive a failed rebuild.
    assert _indexed_count() == 1


def test_ingest_of_empty_docs_dir_preserves_the_existing_index(
    fake_embed, temp_db, tmp_path: Path
) -> None:
    docs = tmp_path / "docs"
    assert _seed_collection(docs) == 1

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert index_docs(docs_dir=empty_dir) == 0
    assert _indexed_count() == 1


def test_answer_question_raises_clearly_when_nothing_is_retrieved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda question, n_results=4, source=None: {
            "documents": [[]],
            "metadatas": [[]],
        },
    )
    monkeypatch.setattr(
        rag, "ask_model", lambda prompt: pytest.fail("model must not be called")
    )

    with pytest.raises(EmptyIndexError, match="ingest"):
        rag.answer_question("anything")
