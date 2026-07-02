import json
from pathlib import Path
from typing import Any

import ollama

from agent_tools import grep_files, list_files, read_file

AGENT_MODEL = "qwen2.5-coder:3b"
MAX_ITERATIONS = 8

# Terse on purpose: a 3B-12B model follows short, imperative instructions
# far better than long prose. The "method" line is the core steering.
SYSTEM_PROMPT = """\
You answer questions about the code in this project using tools.

Tools:
- list_files(subdir): see what files exist
- grep(pattern, subdir): find where something is defined or used
- read_file(path, start_line): read a file that matched

Method: grep for a specific identifier first, read only the files that
matched, then answer. Cite evidence as path:line. If two different searches
find nothing, say what you could not find instead of guessing. Never invent
file contents.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the project's text files, optionally under one subdirectory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subdir": {
                        "type": "string",
                        "description": "Directory relative to the project root, like 'src'. Omit for the whole project.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents with a Python regular expression. Returns 'path:line: text' matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression, like 'def retrieve'.",
                    },
                    "subdir": {
                        "type": "string",
                        "description": "Limit the search to this directory, like 'src'.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file with numbered lines. Long files truncate; call again with start_line to continue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the project root, like 'src/rag.py'.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read. Defaults to 1.",
                    },
                },
                "required": ["path"],
            },
        },
    },
]


def dispatch_tool(name: str, arguments: dict, root: Path) -> str:
    """Run one tool call; every failure comes back as text for the model."""
    try:
        if name == "list_files":
            return list_files(root, arguments.get("subdir", "."))

        if name == "grep":
            return grep_files(root, arguments["pattern"], arguments.get("subdir", "."))

        if name == "read_file":
            return read_file(root, arguments["path"], int(arguments.get("start_line", 1)))
    except (KeyError, ValueError, TypeError) as error:
        return f"Tool error: {error}"

    return f"Unknown tool '{name}'. Available: list_files, grep, read_file."


def run_agent(
    question: str,
    root: Path,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[str, list[str]]:
    """Answer a codebase question by letting the model drive search tools.

    Returns (answer, trace) where trace lists every tool call made, so the
    user can see how the agent found its answer.
    """
    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace: list[str] = []
    last_signature: str | None = None

    for _ in range(max_iterations):
        response = ollama.chat(
            model=AGENT_MODEL, messages=messages, tools=TOOL_SCHEMAS
        )
        message = response["message"]
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            return message["content"], trace

        messages.append(message)

        for call in tool_calls:
            name = call["function"]["name"]
            arguments = dict(call["function"]["arguments"])
            signature = f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"

            if signature == last_signature:
                # A small model can get stuck re-issuing one call forever;
                # answering it with a nudge breaks the loop cheaply.
                result = (
                    "You already ran exactly this call. Use the previous "
                    "result, or answer with what you have."
                )
            else:
                result = dispatch_tool(name, arguments, root)

            last_signature = signature
            trace.append(f"{name}({arguments})")
            messages.append({"role": "tool", "tool_name": name, "content": result})

    return (
        "Stopped after reaching the tool-call limit without a final answer. "
        "Try asking a more specific question.",
        trace,
    )
