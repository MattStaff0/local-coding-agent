import pytest

import rag
from ask import apply_source_command


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


def test_sources_command_lists_available(available: list[str]) -> None:
    handled, source, message = apply_source_command("/sources", "pytorch")

    assert handled
    assert source == "pytorch"  # listing does not change the scope
    assert "python" in message and "pytorch" in message


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
