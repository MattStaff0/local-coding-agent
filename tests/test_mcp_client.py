"""MCP client plumbing that runs offline: config, namespacing, allowlists."""
import json
from types import SimpleNamespace

import mcp_client


def _tool(name, description="d", schema=None):
    return SimpleNamespace(
        name=name, description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


def test_load_config_missing_file_is_empty(tmp_path):
    assert mcp_client.load_config(tmp_path / "mcp.json") == {}


def test_to_ollama_schema_namespaces_by_server():
    schema = mcp_client.to_ollama_schema("fetch", _tool("fetch"))
    assert schema["function"]["name"] == "fetch"          # already prefixed
    schema = mcp_client.to_ollama_schema("git", _tool("git_status"))
    assert schema["function"]["name"] == "git_status"     # already prefixed
    schema = mcp_client.to_ollama_schema("notes", _tool("search"))
    assert schema["function"]["name"] == "notes_search"   # gets the prefix


def test_allowlist_filters_discovered_tools():
    config = {"tools": ["git_status"]}
    tools = [_tool("git_status"), _tool("git_reset_hard")]
    kept = mcp_client.allowed_tools("git", config, tools)
    assert [t.name for t in kept] == ["git_status"]


def test_allowlist_matches_namespaced_names():
    # The config allowlists what the model sees ("notes_search"), which must
    # match a server tool named just "search".
    config = {"tools": ["notes_search"]}
    kept = mcp_client.allowed_tools("notes", config, [_tool("search"), _tool("drop")])
    assert [t.name for t in kept] == ["search"]


def test_load_config_bad_json_raises(tmp_path):
    bad = tmp_path / "mcp.json"
    bad.write_text("{not json", encoding="utf-8")
    try:
        mcp_client.load_config(bad)
    except json.JSONDecodeError:
        pass  # a syntax error must be loud (start_mcp catches and degrades)
    else:
        raise AssertionError("expected JSONDecodeError")


class _FakeSession:
    def __init__(self, tools, results):
        self._tools, self._results = tools, results
        self.calls = []

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._results[name])]
        )


def _manager(config, sessions):
    return mcp_client.MCPManager(
        config, session_factory=lambda name, cfg: sessions[name]
    )


def test_start_returns_allowlisted_namespaced_schemas():
    config = {"servers": {"git": {"tools": ["git_status"]}}}
    session = _FakeSession([_tool("git_status"), _tool("git_reset_hard")], {})
    manager = _manager(config, {"git": session})

    schemas = manager.start()

    assert [s["function"]["name"] for s in schemas] == ["git_status"]
    assert manager.owns("git_status") and not manager.owns("read_file")
    manager.stop()


def test_call_routes_to_owning_session_and_strips_namespace():
    config = {"servers": {"notes": {"tools": ["notes_search"]}}}
    session = _FakeSession([_tool("search")], {"search": "found it"})
    manager = _manager(config, {"notes": session})
    manager.start()

    result = manager.call("notes_search", {"q": "x"})

    assert result == "found it"
    assert session.calls == [("search", {"q": "x"})]
    manager.stop()


def test_start_populates_confirm_gate_from_config():
    config = {
        "servers": {
            "git": {"tools": ["git_status", "git_commit"], "confirm": ["git_commit"]}
        }
    }
    session = _FakeSession([_tool("git_status"), _tool("git_commit")], {})
    manager = _manager(config, {"git": session})
    manager.start()

    assert manager.needs_confirm("git_commit")
    assert not manager.needs_confirm("git_status")
    manager.stop()


def test_call_failure_is_a_string_not_an_exception():
    config = {"servers": {"git": {"tools": ["git_status"]}}}

    class _Exploding(_FakeSession):
        async def call_tool(self, name, arguments):
            raise RuntimeError("server died")

    manager = _manager(config, {"git": _Exploding([_tool("git_status")], {})})
    manager.start()

    assert "MCP tool error" in manager.call("git_status", {})
    manager.stop()
