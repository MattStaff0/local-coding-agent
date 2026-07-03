import pytest

import rag
from rag import build_prompt, source_legend

METADATAS = [
    {
        "source": "pytorch",
        "path": "docs/pytorch/tensorqs-tutorial.md",
        "heading": "Tensors > Initializing a Tensor",
        "chunk_index": 2,
    },
    {
        "source": "python",
        "path": "docs/python/datastructures.md",
        "heading": "5. Data Structures > 5.1. More on Lists",
        "chunk_index": 0,
    },
]

DOCS = ["Tensors can be created from data.", "Lists have append and extend."]


def test_build_prompt_numbers_and_labels_context_chunks() -> None:
    prompt = build_prompt("How do I make a tensor?", DOCS, metadatas=METADATAS)

    assert "[1] docs/pytorch/tensorqs-tutorial.md § Tensors > Initializing a Tensor" in prompt
    assert "[2] docs/python/datastructures.md § 5. Data Structures > 5.1. More on Lists" in prompt
    assert "Tensors can be created from data." in prompt


def test_build_prompt_demands_grounded_cited_answers() -> None:
    prompt = build_prompt("q", DOCS, metadatas=METADATAS)
    lowered = prompt.lower()

    assert "only" in lowered and "context" in lowered
    assert "cite" in lowered
    assert "[1]" in prompt
    # The model must admit gaps instead of answering from its own training.
    assert "missing" in lowered or "not covered" in lowered


def test_build_prompt_still_works_without_metadata() -> None:
    prompt = build_prompt("q", DOCS)

    assert "Tensors can be created from data." in prompt


def test_source_legend_maps_numbers_to_path_and_heading() -> None:
    legend = source_legend(METADATAS)

    assert legend == [
        "[1] docs/pytorch/tensorqs-tutorial.md § Tensors > Initializing a Tensor",
        "[2] docs/python/datastructures.md § 5. Data Structures > 5.1. More on Lists",
    ]


def test_source_legend_handles_missing_heading() -> None:
    legend = source_legend([{"source": "general", "path": "docs/notes.md", "heading": ""}])

    assert legend == ["[1] docs/notes.md"]


def test_answer_question_builds_labeled_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda question, n_results=4, source=None: {
            "documents": [DOCS],
            "metadatas": [METADATAS],
        },
    )

    def fake_ask(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Use torch.tensor(data) [1]."

    monkeypatch.setattr(rag, "ask_model", fake_ask)

    answer, metadatas = rag.answer_question("How do I make a tensor?")

    assert "[1] docs/pytorch/tensorqs-tutorial.md" in captured["prompt"]
    assert metadatas == METADATAS
