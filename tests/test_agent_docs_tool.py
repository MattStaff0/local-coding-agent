"""The agent's search_docs tool: docs retrieval from inside the agent loop."""
from pathlib import Path

import pytest

import agent
import rag


def _retrieval(documents: list[str], metadatas: list[dict]) -> dict:
    """Shape a fake rag.retrieve result the way the real one returns it."""
    n = len(documents)
    return {
        "ids": [[f"chunk-{i}" for i in range(n)]],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [[0.1] * n],
    }


def test_search_docs_is_registered_within_the_tool_cap():
    names = [schema["function"]["name"] for schema in agent.TOOL_SCHEMAS]
    assert "search_docs" in names
    # CLAUDE.md: small local models degrade sharply past 8 tools.
    assert len(names) <= 8


def test_system_prompt_tells_the_model_about_search_docs():
    assert "search_docs" in agent.SYSTEM_PROMPT


def test_dispatch_search_docs_returns_labeled_chunks(monkeypatch, tmp_path):
    seen = {}

    def fake_retrieve(query, n_results=4, source=None, **kwargs):
        seen["query"] = query
        seen["source"] = source
        return _retrieval(
            ["Tensors are multi-dimensional arrays."],
            [{"path": "docs/pytorch/tensors.md", "heading": "Tensors"}],
        )

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)

    result = agent.dispatch_tool(
        "search_docs", {"query": "what is a tensor"}, tmp_path
    )

    assert seen["query"] == "what is a tensor"
    assert seen["source"] is None
    assert "docs/pytorch/tensors.md" in result
    assert "Tensors are multi-dimensional arrays." in result


def test_dispatch_search_docs_passes_source_filter(monkeypatch, tmp_path):
    seen = {}

    def fake_retrieve(query, n_results=4, source=None, **kwargs):
        seen["source"] = source
        return _retrieval(["chunk"], [{"path": "docs/numpy/x.md"}])

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)

    agent.dispatch_tool(
        "search_docs", {"query": "broadcasting", "source": "numpy"}, tmp_path
    )

    assert seen["source"] == "numpy"


def test_dispatch_search_docs_reports_no_matches_as_text(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: _retrieval([], []))

    result = agent.dispatch_tool("search_docs", {"query": "zzz"}, tmp_path)

    assert "No documentation matched" in result


def test_dispatch_search_docs_rejects_irrelevant_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda *a, **k: _retrieval(
            ["Unrelated documentation."], [{"path": "docs/python/lists.md"}]
        ),
    )
    monkeypatch.setattr(rag, "would_refuse", lambda results: True)

    result = agent.dispatch_tool("search_docs", {"query": "zzz"}, tmp_path)

    assert "No relevant documentation matched" in result


def test_dispatch_search_docs_returns_tool_error_when_index_unavailable(
    monkeypatch, tmp_path
):
    def broken_retrieve(*args, **kwargs):
        raise RuntimeError("collection does not exist")

    monkeypatch.setattr(rag, "retrieve", broken_retrieve)

    result = agent.dispatch_tool("search_docs", {"query": "anything"}, tmp_path)

    # Failures must come back as text: a raised exception would abort the
    # whole agent turn, and the "Tool error:" prefix marks the trace ERROR.
    assert result.startswith("Tool error:")
    assert "collection does not exist" in result


def test_dispatch_search_docs_requires_a_query(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rag, "retrieve", lambda *a, **k: _retrieval(["x"], [{"path": "p"}])
    )

    result = agent.dispatch_tool("search_docs", {}, tmp_path)

    assert result.startswith("Tool error:")


def test_run_agent_can_answer_via_search_docs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda *a, **k: _retrieval(
            ["Broadcasting stretches arrays."], [{"path": "docs/numpy/b.md"}]
        ),
    )

    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_docs",
                                "arguments": {"query": "broadcasting"},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Broadcasting stretches arrays. [1]",
                }
            },
        ]
    )
    monkeypatch.setattr(agent.ollama, "chat", lambda **kwargs: next(responses))

    answer, trace = agent.run_agent("what is broadcasting?", root=tmp_path)

    assert "Broadcasting" in answer
    assert any("search_docs" in step and "ERROR" not in step for step in trace)
