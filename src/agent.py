import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import ollama

from agent_tools import (
    apply_content,
    grep_files,
    list_files,
    preview_edit,
    preview_write,
    read_file,
    run_command,
)

# Same env override as rag.CHAT_MODEL: the agent loop runs on whatever chat
# model the machine has, without a code edit.
AGENT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5-coder:3b")
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

For code changes: use edit_file with the smallest unique old_text. The user
approves or declines each change; a declined change is an answer, not an error.
End answers about code with "Evidence:" followed by path:line citations
from your tool results. Never cite lines you did not read.
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
                        "minimum": 1,
                        "description": "First line to read. Defaults to 1.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace one exact text match in a file; the user approves each change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the project root.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace; must appear exactly once.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite one file; the user approves each write.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the project root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full new file content.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run pytest or python in the project root; the user approves each run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command line, like 'pytest -q' or 'python script.py'.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def _gated_write(
    name: str,
    preview: dict,
    path: str,
    root: Path,
    confirm: Callable[[str, str], bool] | None,
) -> str:
    """Apply a previewed change only if the user approves the diff."""
    if "error" in preview:
        return preview["error"]

    if confirm is None:
        return "No confirmation channel available."

    if not confirm(f"{name} {path}", preview["diff"]):
        return "User declined the change."

    apply_content(root, path, preview["new_content"])
    return f"Applied:\n{preview['diff']}"


def dispatch_tool(
    name: str,
    arguments: dict,
    root: Path,
    confirm: Callable[[str, str], bool] | None = None,
) -> str:
    """Run one tool call; every failure comes back as text for the model.

    Write/run tools go through `confirm`; read tools never prompt.
    """
    try:
        if name == "list_files":
            return list_files(root, arguments.get("subdir", "."))

        if name == "grep":
            return grep_files(root, arguments["pattern"], arguments.get("subdir", "."))

        if name == "read_file":
            return read_file(root, arguments["path"], int(arguments.get("start_line", 1)))

        if name == "edit_file":
            preview = preview_edit(
                root, arguments["path"], arguments["old_text"], arguments["new_text"]
            )
            return _gated_write(name, preview, arguments["path"], root, confirm)

        if name == "write_file":
            preview = preview_write(root, arguments["path"], arguments["content"])
            return _gated_write(name, preview, arguments["path"], root, confirm)

        if name == "run_command":
            command = arguments["command"]

            if confirm is None:
                return "No confirmation channel available."
            if not confirm(f"run: {command}", command):
                return "User declined the change."

            return run_command(root, command)
    except (KeyError, ValueError, TypeError) as error:
        return f"Tool error: {error}"

    return (
        f"Unknown tool '{name}'. Available: list_files, grep, read_file, "
        "edit_file, write_file, run_command."
    )


@dataclass
class AgentSession:
    """One /agent conversation: where it looks (root) and what was said."""

    root: Path
    messages: list[Any] = field(default_factory=list)


def run_agent(
    question: str,
    root: Path | None = None,
    max_iterations: int = MAX_ITERATIONS,
    session: AgentSession | None = None,
    confirm: Callable[[str, str], bool] | None = None,
) -> tuple[str, list[str]]:
    """Answer a codebase question by letting the model drive search tools.

    Returns (answer, trace) where trace lists every tool call made, so the
    user can see how the agent found its answer. A session carries the
    conversation across calls; the system prompt is prepended fresh every
    call (never stored) so prompt upgrades apply to old sessions.
    """
    if session is None:
        if root is None:
            raise ValueError("root is required without a session")
        session = AgentSession(root=root)

    session.messages.append({"role": "user", "content": question})
    trace: list[str] = []
    last_signature: str | None = None

    for _ in range(max_iterations):
        response = ollama.chat(
            model=AGENT_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *session.messages],
            tools=TOOL_SCHEMAS,
        )
        message = response["message"]
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            session.messages.append(
                {"role": "assistant", "content": message["content"]}
            )
            return message["content"], trace

        session.messages.append(message)

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
                result = dispatch_tool(name, arguments, session.root, confirm)

            last_signature = signature
            trace.append(f"{name}({arguments})")
            session.messages.append(
                {"role": "tool", "tool_name": name, "content": result}
            )

    return (
        "Stopped after reaching the tool-call limit without a final answer. "
        "Try asking a more specific question.",
        trace,
    )


def parse_agent_command(line: str) -> tuple[str, str] | None:
    """Parse '/agent ...' into (subcommand, argument).

    Returns ("ask", question), ("reset", ""), ("root", path), ("status", "")
    or None when the line is not an /agent command. Only exact 'reset'/'root'
    first words are subcommands — anything else is part of a question.
    """
    stripped = line.strip()

    if stripped != "/agent" and not stripped.startswith("/agent "):
        return None

    rest = stripped.removeprefix("/agent").strip()

    if not rest:
        return ("status", "")

    first, _, remainder = rest.partition(" ")

    if first == "reset" and not remainder:
        return ("reset", "")

    if first == "root":
        return ("root", remainder.strip())

    return ("ask", rest)


def format_agent_reply(answer: str, trace: list[str]) -> str:
    """Show how the agent searched, then its answer."""
    if not trace:
        return answer

    lines = ["Tool calls:"]
    lines.extend(f"  {entry}" for entry in trace)
    lines.append("")
    lines.append(answer)

    return "\n".join(lines)


def main() -> None:
    """Answer one codebase question from the command line."""
    if len(sys.argv) < 2:
        print('Usage: python src/agent.py "where is retrieval scoring done?"')
        raise SystemExit(1)

    question = " ".join(sys.argv[1:])
    answer, trace = run_agent(question, root=Path.cwd())

    print(format_agent_reply(answer, trace))


if __name__ == "__main__":
    main()
