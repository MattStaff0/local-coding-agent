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
    kept = mcp_client.allowed_tools(config, tools)
    assert [t.name for t in kept] == ["git_status"]
