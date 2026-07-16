"""MCP tools ride along in the agent loadout, behind the 8-tool cap."""
from types import SimpleNamespace

import agent


def _scripted_ollama(monkeypatch, replies):
    """Records (messages, tools) for every call; replays scripted replies."""
    calls = []

    def fake_chat(model, messages, tools):
        calls.append((list(messages), list(tools)))
        return {"message": replies.pop(0)}

    monkeypatch.setattr(agent.ollama, "chat", fake_chat)
    return calls


def _schema(name):
    return {
        "type": "function",
        "function": {"name": name, "description": "d", "parameters": {}},
    }


class _FakeManager:
    def __init__(self, names, results, confirm_tools=()):
        self.schemas = [_schema(name) for name in names]
        self._results = results
        self._confirm_tools = set(confirm_tools)
        self.calls = []

    def owns(self, name):
        return any(s["function"]["name"] == name for s in self.schemas)

    def needs_confirm(self, name):
        return name in self._confirm_tools

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        return self._results[name]


def test_mcp_tool_call_routes_through_manager(tmp_path, monkeypatch):
    manager = _FakeManager(["git_status"], {"git_status": "clean tree"})
    calls = _scripted_ollama(monkeypatch, [
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "git_status", "arguments": {}}}
            ],
        },
        {"content": "nothing changed", "tool_calls": []},
    ])

    answer, trace = agent.run_agent("what changed?", root=tmp_path, mcp=manager)

    assert answer == "nothing changed"
    assert manager.calls == [("git_status", {})]
    # The MCP schema was offered to the model alongside the native tools.
    first_tools = [t["function"]["name"] for t in calls[0][1]]
    assert "git_status" in first_tools and "grep" in first_tools
    # The tool result reached the model.
    assert any("clean tree" in str(m) for m in calls[1][0])


def _gated_run(tmp_path, monkeypatch, confirm):
    """One scripted run where the model calls a confirm-gated MCP tool."""
    manager = _FakeManager(
        ["git_commit"], {"git_commit": "committed"}, confirm_tools=["git_commit"]
    )
    calls = _scripted_ollama(monkeypatch, [
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "git_commit", "arguments": {"m": "x"}}}
            ],
        },
        {"content": "done", "tool_calls": []},
    ])
    agent.run_agent("commit it", root=tmp_path, mcp=manager, confirm=confirm)
    return manager, calls


def test_gated_mcp_tool_declined_is_never_called(tmp_path, monkeypatch):
    manager, calls = _gated_run(tmp_path, monkeypatch, confirm=lambda d, p: False)

    assert manager.calls == []
    assert any("User declined the change." in str(m) for m in calls[1][0])


def test_gated_mcp_tool_without_channel_is_declined(tmp_path, monkeypatch):
    manager, _ = _gated_run(tmp_path, monkeypatch, confirm=None)
    assert manager.calls == []


def test_gated_mcp_tool_approved_is_called(tmp_path, monkeypatch):
    manager, _ = _gated_run(tmp_path, monkeypatch, confirm=lambda d, p: True)
    assert manager.calls == [("git_commit", {"m": "x"})]


def test_failed_tool_calls_are_marked_in_the_trace(tmp_path, monkeypatch):
    _scripted_ollama(monkeypatch, [
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "no_such_tool", "arguments": {}}}
            ],
        },
        {"content": "gave up", "tool_calls": []},
    ])

    _, trace = agent.run_agent("q", root=tmp_path)

    assert len(trace) == 1 and trace[0].endswith("-> ERROR")


def test_over_cap_loadout_warns_once(tmp_path, monkeypatch, capsys):
    extra = [f"srv_tool{i}" for i in range(4)]  # 7 native + 4 = 11 > 8
    manager = _FakeManager(extra, {})
    _scripted_ollama(monkeypatch, [{"content": "ok", "tool_calls": []}])

    agent.run_agent("q", root=tmp_path, mcp=manager)

    out = capsys.readouterr().out
    assert out.count("small models degrade past 8") == 1
