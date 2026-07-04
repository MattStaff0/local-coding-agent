"""Hand-rolled MCP client plumbing for the agent loop.

Discovered tools are namespaced `server_tool` because two servers may both
export a generic name like `search`; descriptions are capped at 200 chars
because every tool schema competes for the small model's context budget.
"""
import asyncio
import json
import threading
from contextlib import AsyncExitStack
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


class MCPManager:
    """Runs MCP sessions on a background asyncio loop, exposes sync calls.

    The agent loop is synchronous; the MCP SDK is asyncio. A dedicated
    daemon thread owns the event loop so tool calls become plain blocking
    function calls via run_coroutine_threadsafe.
    """

    def __init__(self, config: dict, session_factory=None):
        self._config = config
        self._session_factory = session_factory or self._sdk_session_factory
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stack = AsyncExitStack()
        # namespaced tool name -> (session, original tool name)
        self._routes: dict[str, tuple[object, str]] = {}

    def _run(self, coroutine, timeout: int = 60):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout)

    def _sdk_session_factory(self, server: str, server_config: dict):
        """Connect to a stdio server with the official SDK (live use only)."""

        async def connect():
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=server_config["command"], args=server_config.get("args", [])
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return _SDKSession(session)

        return self._run(connect)

    def start(self) -> list[dict]:
        """Connect every configured server; return merged ollama schemas."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        schemas: list[dict] = []

        for server, server_config in self._config.get("servers", {}).items():
            try:
                session = self._session_factory(server, server_config)
                tools = self._run(session.list_tools())
            except Exception as error:
                print(f"(mcp: server '{server}' unavailable: {error})")
                continue

            allowed = set(server_config.get("tools", []))

            for tool in tools:
                name = namespaced(server, tool.name)

                if allowed and name not in allowed:
                    continue

                self._routes[name] = (session, tool.name)
                schemas.append(to_ollama_schema(server, tool))

        return schemas

    def owns(self, tool_name: str) -> bool:
        return tool_name in self._routes

    def call(self, tool_name: str, arguments: dict) -> str:
        """Call one MCP tool; every failure is a string for the model."""
        try:
            session, original_name = self._routes[tool_name]
            result = self._run(session.call_tool(original_name, arguments))
            texts = [
                item.text
                for item in result.content
                if getattr(item, "type", "") == "text"
            ]
            return "\n".join(texts)
        except Exception as error:
            return f"MCP tool error: {error}"

    def stop(self) -> None:
        if self._loop is None:
            return

        try:
            self._run(self._stack.aclose(), timeout=10)
        except Exception:
            pass

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        self._loop.close()
        self._loop = None


class _SDKSession:
    """Adapts mcp.ClientSession to the duck type MCPManager expects."""

    def __init__(self, session):
        self._session = session

    async def list_tools(self):
        return (await self._session.list_tools()).tools

    async def call_tool(self, name, arguments):
        return await self._session.call_tool(name, arguments)
