"""Prompt budgeting: retrieval order is quality order, so trim from the tail."""
import rag


def test_budget_keeps_top_ranked_chunks():
    docs = ["a" * 100, "b" * 100, "c" * 100]
    metadatas = [{"i": 1}, {"i": 2}, {"i": 3}]

    kept_docs, kept_metadatas = rag.budget_chunks(docs, metadatas, budget=250)

    assert kept_docs == ["a" * 100, "b" * 100]
    assert kept_metadatas == [{"i": 1}, {"i": 2}]


def test_budget_always_keeps_first_chunk():
    docs = ["a" * 500]
    kept_docs, _ = rag.budget_chunks(docs, [{}], budget=10)
    assert kept_docs == docs


def test_budget_default_is_env_tunable(monkeypatch):
    docs = ["a" * 100, "b" * 100]
    monkeypatch.setattr(rag, "PROMPT_CHAR_BUDGET", 150)
    kept_docs, _ = rag.budget_chunks(docs, [{}, {}])
    assert kept_docs == ["a" * 100]


def test_answer_question_trims_and_notes(monkeypatch, capsys):
    results = {
        "ids": [["a", "b"]],
        "documents": [["x" * 200, "y" * 200]],
        "metadatas": [
            [
                {"source": "s", "path": "p", "heading": "h"},
                {"source": "s", "path": "p2", "heading": "h2"},
            ]
        ],
        "distances": [[0.1, 0.2]],
        "keyword_hits": [[True, False]],
    }
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: results)
    monkeypatch.setattr(rag, "ask_model", lambda prompt, on_token=None: "fine [1]")
    monkeypatch.setattr(rag, "PROMPT_CHAR_BUDGET", 250)

    answer, metadatas = rag.answer_question("q")

    assert answer == "fine [1]"
    assert len(metadatas) == 1  # the trimmed chunk's legend entry is gone too
    assert "trimmed to 1 of 2" in capsys.readouterr().out
