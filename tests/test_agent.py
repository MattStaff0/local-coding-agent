from pathlib import Path

import pytest

import agent


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text(
        "def retrieve():\n    return 4\n", encoding="utf-8"
    )
    return tmp_path


def test_dispatch_routes_grep(project: Path) -> None:
    result = agent.dispatch_tool("grep", {"pattern": "retrieve"}, project)

    assert "app.py:1" in result


def test_dispatch_routes_read_file(project: Path) -> None:
    result = agent.dispatch_tool("read_file", {"path": "app.py"}, project)

    assert result.startswith("1: def retrieve():")


def test_dispatch_routes_list_files(project: Path) -> None:
    result = agent.dispatch_tool("list_files", {}, project)

    assert "app.py" in result


def test_dispatch_reports_missing_required_arguments(project: Path) -> None:
    result = agent.dispatch_tool("grep", {}, project)

    assert "Tool error" in result


def test_dispatch_reports_sandbox_escapes_as_tool_errors(project: Path) -> None:
    result = agent.dispatch_tool("read_file", {"path": "../secrets.txt"}, project)

    assert "Tool error" in result


def test_dispatch_names_available_tools_for_unknown_names(project: Path) -> None:
    result = agent.dispatch_tool("delete_everything", {}, project)

    assert "Unknown tool" in result
    assert "grep" in result


def test_every_schema_is_a_complete_function_definition() -> None:
    names = set()

    for schema in agent.TOOL_SCHEMAS:
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["description"]
        assert function["parameters"]["type"] == "object"
        names.add(function["name"])

    assert names == {"list_files", "grep", "read_file"}


def scripted_chat(monkeypatch: pytest.MonkeyPatch, responses: list[dict]) -> list[dict]:
    """Replace ollama.chat with a script; returns the recorded calls."""
    calls: list[dict] = []

    def fake_chat(model: str, messages: list, tools: list | None = None) -> dict:
        calls.append({"model": model, "messages": list(messages), "tools": tools})
        return responses.pop(0)

    monkeypatch.setattr(agent.ollama, "chat", fake_chat)
    return calls


def answer(content: str) -> dict:
    return {"message": {"role": "assistant", "content": content}}


def tool_call(name: str, arguments: dict) -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
        }
    }


def tool_messages(call: dict) -> list[dict]:
    return [m for m in call["messages"] if m.get("role") == "tool"]


def test_direct_answer_needs_no_tools(
    monkeypatch: pytest.MonkeyPatch, project: Path
) -> None:
    scripted_chat(monkeypatch, [answer("It is a RAG project.")])

    result, trace = agent.run_agent("what is this?", root=project)

    assert result == "It is a RAG project."
    assert trace == []


def test_tool_results_are_sent_back_to_the_model(
    monkeypatch: pytest.MonkeyPatch, project: Path
) -> None:
    calls = scripted_chat(
        monkeypatch,
        [
            tool_call("grep", {"pattern": "def retrieve"}),
            answer("retrieve() is defined at app.py:1."),
        ],
    )

    result, trace = agent.run_agent("where is retrieve?", root=project)

    assert "app.py:1" in result
    assert "app.py:1" in tool_messages(calls[1])[0]["content"]
    assert trace == ["grep({'pattern': 'def retrieve'})"]


def test_loop_stops_at_max_iterations(
    monkeypatch: pytest.MonkeyPatch, project: Path
) -> None:
    scripted_chat(monkeypatch, [tool_call("list_files", {})] * 3)

    result, trace = agent.run_agent("q", root=project, max_iterations=3)

    assert "Stopped" in result
    assert len(trace) == 3


def test_repeated_identical_call_gets_a_nudge_not_a_rerun(
    monkeypatch: pytest.MonkeyPatch, project: Path
) -> None:
    calls = scripted_chat(
        monkeypatch,
        [
            tool_call("grep", {"pattern": "x"}),
            tool_call("grep", {"pattern": "x"}),
            answer("done"),
        ],
    )

    result, trace = agent.run_agent("q", root=project)

    assert result == "done"
    assert "already" in tool_messages(calls[2])[-1]["content"]


def test_unknown_tool_calls_are_reported_to_the_model(
    monkeypatch: pytest.MonkeyPatch, project: Path
) -> None:
    calls = scripted_chat(
        monkeypatch,
        [tool_call("delete_everything", {}), answer("ok")],
    )

    agent.run_agent("q", root=project)

    assert "Unknown tool" in tool_messages(calls[1])[0]["content"]


def test_parse_agent_command_extracts_the_question() -> None:
    assert agent.parse_agent_command("/agent where is retrieve?") == "where is retrieve?"


def test_parse_agent_command_ignores_other_lines() -> None:
    assert agent.parse_agent_command("where is retrieve?") is None
    assert agent.parse_agent_command("/agent") is None
    assert agent.parse_agent_command("/agent   ") is None
    assert agent.parse_agent_command("/agentfoo") is None
    assert agent.parse_agent_command("/agents list") is None


def test_format_agent_reply_shows_the_tool_trace() -> None:
    reply = agent.format_agent_reply(
        "It is in rag.py.", ["grep({'pattern': 'retrieve'})"]
    )

    assert "grep({'pattern': 'retrieve'})" in reply
    assert reply.endswith("It is in rag.py.")


def test_format_agent_reply_without_tools_is_just_the_answer() -> None:
    assert agent.format_agent_reply("Hi.", []) == "Hi."
