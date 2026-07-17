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


def test_system_prompt_requires_grounded_mixed_evidence():
    prompt = agent.SYSTEM_PROMPT

    assert "library/API claims" in prompt
    assert "mixed questions" in prompt
    assert "file evidence as path:line" in prompt
    assert "numbered [n] path § heading label" in prompt


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


def test_run_agent_combines_live_file_and_docs_evidence(monkeypatch, tmp_path):
    (tmp_path / "app.py").write_text(
        "import pandas as pd\nresult = pd.concat([])\n", encoding="utf-8"
    )
    seen = {}

    def fake_retrieve(query, n_results=4, source=None, **kwargs):
        seen["query"] = query
        seen["source"] = source
        return _retrieval(
            ["concat combines pandas objects."],
            [
                {
                    "path": "docs/pandas/merging.md",
                    "heading": "Concatenating objects",
                }
            ],
        )

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "app.py"},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_docs",
                                "arguments": {
                                    "query": "pandas concat empty inputs",
                                    "source": "pandas",
                                },
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "The call is at app.py:2; compare it with "
                        "[1] docs/pandas/merging.md § Concatenating objects."
                    ),
                }
            },
        ]
    )
    model_calls = []

    def fake_chat(**kwargs):
        model_calls.append(kwargs["messages"])
        return next(responses)

    monkeypatch.setattr(agent.ollama, "chat", fake_chat)

    answer, trace = agent.run_agent(
        "Why does the pandas call in app.py fail?", root=tmp_path
    )

    assert seen == {"query": "pandas concat empty inputs", "source": "pandas"}
    assert [entry.split("(", 1)[0] for entry in trace] == [
        "read_file",
        "search_docs",
    ]
    assert all("ERROR" not in entry for entry in trace)
    final_messages = model_calls[-1]
    assert any(
        message.get("role") == "tool"
        and message.get("tool_name") == "read_file"
        and "2: result = pd.concat([])" in message["content"]
        for message in final_messages
    )
    assert any(
        message.get("role") == "tool"
        and message.get("tool_name") == "search_docs"
        and "[1] docs/pandas/merging.md § Concatenating objects"
        in message["content"]
        for message in final_messages
    )
    assert "app.py:2" in answer
    assert "[1] docs/pandas/merging.md § Concatenating objects" in answer


def test_docs_failure_does_not_abort_live_file_reasoning(monkeypatch, tmp_path):
    (tmp_path / "app.py").write_text("answer = 42\n", encoding="utf-8")

    def broken_retrieve(*args, **kwargs):
        raise RuntimeError("collection does not exist")

    monkeypatch.setattr(rag, "retrieve", broken_retrieve)
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
                                "arguments": {"query": "meaning of answer"},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "app.py"},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "Documentation was unavailable, but app.py:1 sets answer to 42."
                    ),
                }
            },
        ]
    )
    model_calls = []

    def fake_chat(**kwargs):
        model_calls.append(kwargs["messages"])
        return next(responses)

    monkeypatch.setattr(agent.ollama, "chat", fake_chat)

    answer, trace = agent.run_agent("Explain answer", root=tmp_path)

    assert trace[0].startswith("search_docs(") and trace[0].endswith("-> ERROR")
    assert trace[1].startswith("read_file(") and "ERROR" not in trace[1]
    final_messages = model_calls[-1]
    assert any(
        message.get("role") == "tool"
        and message.get("tool_name") == "search_docs"
        and message["content"].startswith("Tool error: docs search unavailable")
        for message in final_messages
    )
    assert any(
        message.get("role") == "tool"
        and message.get("tool_name") == "read_file"
        and "1: answer = 42" in message["content"]
        for message in final_messages
    )
    assert "Documentation was unavailable" in answer
    assert "app.py:1" in answer


def test_session_source_scope_overrides_model_search_docs_source(
    monkeypatch, tmp_path
):
    seen = {}

    def fake_retrieve(query, n_results=3, source=None):
        seen["source"] = source
        return _retrieval(["chunk"], [{"path": "docs/pandas/api.md"}])

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
                                "arguments": {"query": "concat", "source": "numpy"},
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "done"}},
        ]
    )
    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(agent.ollama, "chat", lambda **kwargs: next(responses))

    session = agent.AgentSession(root=tmp_path, docs_source="pandas")
    agent.run_agent("q", session=session)

    assert seen["source"] == "pandas"


def test_unscoped_session_preserves_model_selected_source(monkeypatch, tmp_path):
    seen = {}

    def fake_retrieve(query, n_results=3, source=None):
        seen["source"] = source
        return _retrieval(["chunk"], [{"path": "docs/numpy/api.md"}])

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
                                "arguments": {"query": "arrays", "source": "numpy"},
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "done"}},
        ]
    )
    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(agent.ollama, "chat", lambda **kwargs: next(responses))

    agent.run_agent("q", session=agent.AgentSession(root=tmp_path))

    assert seen["source"] == "numpy"
