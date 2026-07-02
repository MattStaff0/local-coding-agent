from pathlib import Path

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
