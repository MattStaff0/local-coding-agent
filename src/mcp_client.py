"""Hand-rolled MCP client plumbing for the agent loop.

Discovered tools are namespaced `server_tool` because two servers may both
export a generic name like `search`; descriptions are capped at 200 chars
because every tool schema competes for the small model's context budget.
"""
import json
from pathlib import Path


def load_config(path: Path) -> dict:
    """Read mcp.json; a missing file just means no MCP servers."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def namespaced(server: str, tool_name: str) -> str:
    """Prefix a tool with its server name unless it already carries it."""
    if tool_name.startswith(server):
        return tool_name
    return f"{server}_{tool_name}"


def to_ollama_schema(server: str, tool) -> dict:
    """Convert one MCP tool (duck-typed) into an ollama tool schema."""
    return {
        "type": "function",
        "function": {
            "name": namespaced(server, tool.name),
            "description": (tool.description or "")[:200],
            "parameters": tool.inputSchema,
        },
    }


def allowed_tools(server_config: dict, tools: list) -> list:
    """Keep only the tools the config explicitly allowlists."""
    allowed = set(server_config.get("tools", []))
    return [tool for tool in tools if tool.name in allowed]
