"""Regression tests for the 2026-07-02 PR-review findings.

Each test pins a behavior a review agent found missing or broken: the
embed-model fingerprint, the L2→cosine migration, the BM25/refusal boundary,
narrow rewrite error handling, and history-file robustness.
"""

import json
from pathlib import Path

import pytest

import ask
import rag
from rag import NoRelevantDocsError, index_docs


@pytest.fixture()
def temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(rag, "DB_DIR", str(tmp_path / "db"))


@pytest.fixture()
def counting_embed(monkeypatch: pytest.MonkeyPatch):
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
    return docs


# --- Incremental ingest: model changes and legacy collections ---


def test_changing_the_embed_model_forces_a_reembed(
    counting_embed, temp_db, docs_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_docs(docs_dir=docs_tree)
    counting_embed.clear()

    monkeypatch.setattr(rag, "EMBED_MODEL", "some-other-model")
    index_docs(docs_dir=docs_tree)

    assert len(counting_embed) == 1  # same file content, new model → re-embed


def test_incremental_ingest_rebuilds_a_non_cosine_collection(
    counting_embed, temp_db, docs_tree: Path, capsys: pytest.CaptureFixture
) -> None:
    # Simulate a pre-upgrade index: default (L2) space, no file hashes.
    legacy = rag.get_client().create_collection(name=rag.COLLECTION_NAME)
    legacy.add(
        ids=["old-0"],
        documents=["old chunk"],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"source": "pytorch", "path": "old.md", "heading": ""}],
    )

    index_docs(docs_dir=docs_tree)

    collection = rag.get_client().get_collection(rag.COLLECTION_NAME)
    assert collection.metadata.get("hnsw:space") == "cosine"
    assert "rebuild" in capsys.readouterr().out.lower()


def test_deletions_are_reported_not_silent(
    counting_embed, temp_db, docs_tree: Path, capsys: pytest.CaptureFixture
) -> None:
    index_docs(docs_dir=docs_tree)
    (docs_tree / "pytorch" / "tensors.md").unlink()
    (docs_tree / "pytorch" / "other.md").write_text("# O\n\nbody", encoding="utf-8")
    capsys.readouterr()

    index_docs(docs_dir=docs_tree)

    assert "tensors.md" in capsys.readouterr().out  # the removal is announced


# --- Hybrid retrieval vs the refusal cutoff ---


def test_distance_exactly_at_the_cutoff_does_not_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda question, n_results=4, source=None: {
            "documents": [["chunk"]],
            "metadatas": [[{"source": "pytorch", "path": "d.md"}]],
            "distances": [[rag.RELEVANCE_CUTOFF]],
        },
    )
    monkeypatch.setattr(rag, "ask_model", lambda prompt, on_token=None: "answer")

    answer, _ = rag.answer_question("q")

    assert answer == "answer"


def test_tokenize_drops_stopwords_but_keeps_identifiers() -> None:
    tokens = rag._tokenize("How do I use the nn.Module class?")

    assert "nn" in tokens and "module" in tokens and "class" in tokens
    assert "how" not in tokens and "the" not in tokens


@pytest.fixture()
def orthogonal_embed(monkeypatch: pytest.MonkeyPatch):
    """Questions embed far (cosine distance 1.0) from every indexed doc."""

    def embed(text: str) -> list[float]:
        return [0.0, 1.0, 0.0] if text.endswith("?") else [1.0, 0.0, 0.0]

    monkeypatch.setattr(rag, "embed", embed)
    monkeypatch.setattr(rag, "embed_batch", lambda texts: [embed(t) for t in texts])
    return embed


def test_identifier_question_answers_despite_far_vectors(
    orthogonal_embed, temp_db, docs_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (docs_tree / "pytorch" / "modules.md").write_text(
        "# Modules\n\nSubclass nn.Module to build networks.", encoding="utf-8"
    )
    # A third unrelated doc keeps BM25's IDF positive: in a 2-doc corpus a
    # term in exactly half the corpus scores ln(1) = 0 and looks like a miss.
    (docs_tree / "pytorch" / "data.md").write_text(
        "# Data\n\nDatasets feed training loops.", encoding="utf-8"
    )
    index_docs(docs_dir=docs_tree)

    def fake_chat(model, messages, stream=False):
        return iter([{"message": {"content": "grounded"}}])

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)

    answer, _ = rag.answer_question("What does nn.Module do?")

    assert answer == "grounded"


def test_off_topic_question_still_refuses_in_hybrid_mode(
    orthogonal_embed, temp_db, docs_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_docs(docs_dir=docs_tree)
    monkeypatch.setattr(
        rag.ollama, "chat", lambda **kw: pytest.fail("model must not be called")
    )

    with pytest.raises(NoRelevantDocsError):
        rag.answer_question("Explain quaternion camera rigs?")


def test_retrieve_rejects_unknown_modes(temp_db) -> None:
    with pytest.raises(ValueError, match="mode"):
        rag.retrieve("q", mode="hybird")


# --- rewrite_query error handling ---


def test_rewrite_announces_infrastructure_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def broken_chat(**kwargs):
        raise ConnectionError("ollama is down")

    monkeypatch.setattr(rag.ollama, "chat", broken_chat)

    result = rag.rewrite_query("How do I train it?", [{"role": "user", "content": "x"}])

    assert result == "How do I train it?"
    assert "rewrite failed" in capsys.readouterr().out


def test_rewrite_lets_our_own_bugs_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    # A malformed response shape is a bug (ours or the library's), not an
    # infrastructure hiccup — it must not be silently swallowed.
    monkeypatch.setattr(rag.ollama, "chat", lambda **kw: {"unexpected": "shape"})

    with pytest.raises(KeyError):
        rag.rewrite_query("q", [{"role": "user", "content": "x"}])


def test_empty_rewrite_falls_back_to_the_original_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag.ollama, "chat", lambda **kw: {"message": {"content": "   \n"}}
    )

    assert rag.rewrite_query("q", [{"role": "user", "content": "x"}]) == "q"


# --- history robustness ---


def test_corrupt_history_is_backed_up_not_clobbered(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    path = tmp_path / "history.json"
    path.write_text("{not json", encoding="utf-8")

    assert ask.load_history(path) == []

    backup = tmp_path / "history.json.bak"
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == "{not json"
    assert "backed up" in capsys.readouterr().out


def test_non_list_history_is_backed_up_too(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"role": "user"}), encoding="utf-8")

    assert ask.load_history(path) == []
    assert (tmp_path / "history.json.bak").exists()


def test_missing_history_stays_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    assert ask.load_history(tmp_path / "nope.json") == []
    assert capsys.readouterr().out == ""


# --- chat_loop wiring (scripted terminal session) ---


def test_chat_loop_wires_history_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(ask, "HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(ask, "EXPORT_DIR", tmp_path / "notes")
    monkeypatch.setattr(
        ask,
        "answer_question",
        lambda q, history, source, on_token=None: (
            "an answer [1]",
            [{"path": "docs/pytorch/t.md", "heading": "T", "source": "pytorch"}],
        ),
    )

    lines = iter(["/export", "a question", "/export", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))

    ask.chat_loop()

    out = capsys.readouterr().out
    assert "Nothing to export yet" in out          # /export before any answer
    assert "Saved study note:" in out              # /export after the answer

    saved = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert [m["role"] for m in saved] == ["user", "assistant"]

    notes = list((tmp_path / "notes").glob("*.md"))
    assert len(notes) == 1
    assert "an answer [1]" in notes[0].read_text(encoding="utf-8")
