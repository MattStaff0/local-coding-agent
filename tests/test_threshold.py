"""Similarity threshold: refuse before the model does (repo plan #1).

When even the best retrieved chunk is too far from the question, the pipeline
should say so immediately instead of handing weak context to the chat model.
"""

from pathlib import Path

import pytest

import ask
import rag
from rag import NoRelevantDocsError


@pytest.fixture()
def temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(rag, "DB_DIR", str(tmp_path / "db"))


def fake_results(distances: list[float]) -> dict:
    n = len(distances)
    return {
        "documents": [[f"chunk {i}" for i in range(n)]],
        "metadatas": [[{"source": "pytorch", "path": f"d{i}.md"} for i in range(n)]],
        "distances": [distances],
    }


def test_collection_is_created_with_cosine_space(temp_db) -> None:
    collection = rag.reset_collection(rag.get_client())

    assert collection.metadata.get("hnsw:space") == "cosine"


def test_far_context_refuses_without_calling_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda question, n_results=4, source=None: fake_results([0.9, 0.95]),
    )
    monkeypatch.setattr(
        rag, "ask_model", lambda prompt: pytest.fail("model must not be called")
    )

    with pytest.raises(NoRelevantDocsError, match="/sources"):
        rag.answer_question("what is a quaternion camera rig?")


def test_close_context_still_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda question, n_results=4, source=None: fake_results([0.2, 0.9]),
    )
    monkeypatch.setattr(rag, "ask_model", lambda prompt, on_token=None: "grounded answer")

    answer, metadatas = rag.answer_question("what is a tensor?")

    assert answer == "grounded answer"
    assert len(metadatas) == 2


def test_results_without_distances_are_not_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = fake_results([])
    del results["distances"]
    results["documents"] = [["chunk"]]
    results["metadatas"] = [[{"source": "pytorch", "path": "d.md"}]]

    monkeypatch.setattr(
        rag, "retrieve", lambda question, n_results=4, source=None: results
    )
    monkeypatch.setattr(rag, "ask_model", lambda prompt, on_token=None: "answer")

    answer, _ = rag.answer_question("q")

    assert answer == "answer"


def test_describe_error_gives_the_refusal_message_without_traceback() -> None:
    message = ask.describe_error(NoRelevantDocsError("nothing relevant indexed"))

    assert message == "nothing relevant indexed"
    assert "Traceback" not in message
