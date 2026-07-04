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
    def __init__(self, names, results):
        self.schemas = [_schema(name) for name in names]
        self._results = results
        self.calls = []

    def owns(self, name):
        return any(s["function"]["name"] == name for s in self.schemas)

    def needs_confirm(self, name):
        return False

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


def test_over_cap_loadout_warns_once(tmp_path, monkeypatch, capsys):
    extra = [f"srv_tool{i}" for i in range(4)]  # 6 native + 4 = 10 > 8
    manager = _FakeManager(extra, {})
    _scripted_ollama(monkeypatch, [{"content": "ok", "tool_calls": []}])

    agent.run_agent("q", root=tmp_path, mcp=manager)

    out = capsys.readouterr().out
    assert out.count("small models degrade past 8") == 1
