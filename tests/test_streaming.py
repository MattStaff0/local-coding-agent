"""Streaming answers (repo plan #2): print tokens as the model produces them.

ollama.chat is mocked with a scripted generator — the real model lives on
another machine.
"""

import pytest

import ask
import rag


def scripted_stream(monkeypatch: pytest.MonkeyPatch, tokens: list[str]) -> list[dict]:
    """Replace ollama.chat with a fake that records calls and streams tokens."""
    calls: list[dict] = []

    def fake_chat(model: str, messages: list, stream: bool = False):
        calls.append({"model": model, "messages": messages, "stream": stream})
        assert stream, "ask_model must request a streaming response"
        return ({"message": {"content": token}} for token in tokens)

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)
    return calls


def test_ask_model_joins_streamed_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    scripted_stream(monkeypatch, ["Tensors", " are", " arrays."])

    assert rag.ask_model("prompt") == "Tensors are arrays."


def test_ask_model_reports_tokens_as_they_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_stream(monkeypatch, ["a", "b", "c"])
    seen: list[str] = []

    rag.ask_model("prompt", on_token=seen.append)

    assert seen == ["a", "b", "c"]


def test_answer_question_passes_on_token_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_stream(monkeypatch, ["grounded", " answer"])
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda question, n_results=4, source=None: {
            "documents": [["chunk"]],
            "metadatas": [[{"source": "pytorch", "path": "d.md"}]],
            "distances": [[0.1]],
        },
    )
    seen: list[str] = []

    answer, _ = rag.answer_question("q", on_token=seen.append)

    assert answer == "grounded answer"
    assert seen == ["grounded", " answer"]


def test_print_token_writes_without_newline(capsys: pytest.CaptureFixture) -> None:
    ask.print_token("Tens")
    ask.print_token("ors")

    assert capsys.readouterr().out == "Tensors"
