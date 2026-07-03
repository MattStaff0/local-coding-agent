"""Follow-up query rewriting (repo plan #9).

"How do I train it?" embeds terribly on its own; before retrieval the model
rewrites follow-ups into standalone queries using the chat history.
"""

import pytest

import rag

HISTORY = [
    {"role": "user", "content": "How do I build a neural network in PyTorch?"},
    {"role": "assistant", "content": "Subclass nn.Module [1]."},
]


def test_no_history_returns_the_question_without_a_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag.ollama, "chat", lambda **kwargs: pytest.fail("no call expected")
    )

    assert rag.rewrite_query("How do I train it?", []) == "How do I train it?"


def test_follow_up_is_rewritten_using_the_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}

    def fake_chat(model: str, messages: list, **kwargs) -> dict:
        seen["prompt"] = messages[-1]["content"]
        return {"message": {"content": " How do I train a PyTorch neural network?\n"}}

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)

    rewritten = rag.rewrite_query("How do I train it?", HISTORY)

    assert rewritten == "How do I train a PyTorch neural network?"
    assert "How do I build a neural network in PyTorch?" in seen["prompt"]
    assert "How do I train it?" in seen["prompt"]


def test_rewrite_failure_falls_back_to_the_original_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_chat(**kwargs):
        raise ConnectionError("ollama is down")

    monkeypatch.setattr(rag.ollama, "chat", broken_chat)

    assert rag.rewrite_query("How do I train it?", HISTORY) == "How do I train it?"


def test_answer_question_retrieves_with_the_rewritten_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}

    monkeypatch.setattr(rag, "rewrite_query", lambda q, h: "REWRITTEN QUERY")

    def fake_retrieve(question: str, n_results: int = 4, source: str | None = None):
        seen["query"] = question
        return {
            "documents": [["chunk"]],
            "metadatas": [[{"source": "pytorch", "path": "d.md"}]],
            "distances": [[0.1]],
        }

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "ask_model", lambda prompt, on_token=None: "answer")

    rag.answer_question("How do I train it?", history=HISTORY)

    assert seen["query"] == "REWRITTEN QUERY"


def test_answer_question_skips_the_rewrite_without_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag, "rewrite_query", lambda q, h: pytest.fail("rewrite not expected")
    )
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda question, n_results=4, source=None: {
            "documents": [["chunk"]],
            "metadatas": [[{"source": "pytorch", "path": "d.md"}]],
            "distances": [[0.1]],
        },
    )
    monkeypatch.setattr(rag, "ask_model", lambda prompt, on_token=None: "answer")

    answer, _ = rag.answer_question("What is a tensor?", history=[])

    assert answer == "answer"
