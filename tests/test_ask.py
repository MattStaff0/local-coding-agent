import chromadb.errors
import httpx
import pytest

import rag
from ask import apply_source_command, describe_error


@pytest.fixture()
def available(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    sources = ["python", "pytorch"]
    monkeypatch.setattr("ask.list_sources", lambda: sources)
    return sources


def test_source_command_scopes_to_a_known_source(available: list[str]) -> None:
    handled, source, message = apply_source_command("/source pytorch", None)

    assert handled
    assert source == "pytorch"
    assert "pytorch" in message


def test_source_command_rejects_unknown_source(available: list[str]) -> None:
    handled, source, message = apply_source_command("/source rust", "python")

    assert handled
    assert source == "python"  # unchanged
    assert "rust" in message and "python" in message


def test_source_all_clears_the_scope(available: list[str]) -> None:
    handled, source, message = apply_source_command("/source all", "pytorch")

    assert handled
    assert source is None


def test_bare_source_command_reports_current_scope(available: list[str]) -> None:
    handled, source, message = apply_source_command("/source", "pytorch")
    assert handled and source == "pytorch" and "pytorch" in message

    handled, source, message = apply_source_command("/source", None)
    assert handled and source is None and "all" in message


def test_sources_command_lists_available(available: list[str]) -> None:
    handled, source, message = apply_source_command("/sources", "pytorch")

    assert handled
    assert source == "pytorch"  # listing does not change the scope
    assert "python" in message and "pytorch" in message


def test_sources_command_survives_a_missing_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_index() -> list[str]:
        raise chromadb.errors.NotFoundError("Collection [local_docs] does not exist")

    monkeypatch.setattr("ask.list_sources", no_index)

    handled, source, message = apply_source_command("/sources", None)

    assert handled
    assert source is None
    assert "ingest" in message


def test_source_command_survives_a_missing_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_index() -> list[str]:
        raise chromadb.errors.NotFoundError("Collection [local_docs] does not exist")

    monkeypatch.setattr("ask.list_sources", no_index)

    handled, source, message = apply_source_command("/source pytorch", None)

    assert handled
    assert source is None
    assert "ingest" in message


def test_describe_error_missing_index_says_run_ingest() -> None:
    error = chromadb.errors.NotFoundError("Collection [local_docs] does not exist")

    assert "ingest" in describe_error(error)


def test_describe_error_empty_index_uses_its_own_message() -> None:
    error = rag.EmptyIndexError("Retrieval returned no chunks — run ingest.")

    assert describe_error(error) == str(error)


def test_describe_error_connection_refused_mentions_ollama() -> None:
    assert "Ollama" in describe_error(httpx.ConnectError("connection refused"))
    assert "Ollama" in describe_error(ConnectionError("refused"))


def test_describe_error_unexpected_bug_shows_type_and_traceback() -> None:
    try:
        raise KeyError("embedding")
    except KeyError as error:
        text = describe_error(error)

    assert "KeyError" in text
    assert "Traceback" in text


def test_regular_question_is_not_a_command(available: list[str]) -> None:
    handled, source, message = apply_source_command("how do lists work?", None)

    assert not handled
    assert source is None


def test_answer_question_passes_source_to_retrieve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_retrieve(question: str, n_results: int = 4, source: str | None = None):
        seen["source"] = source
        return {"documents": [["chunk"]], "metadatas": [[{"source": "pytorch"}]]}

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "ask_model", lambda prompt: "an answer")

    answer, metadatas = rag.answer_question("q", history=[], source="pytorch")

    assert answer == "an answer"
    assert seen["source"] == "pytorch"
