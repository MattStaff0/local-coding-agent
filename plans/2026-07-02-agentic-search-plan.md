---
date: 2026-07-02
repo: /Users/matt_staff/Desktop/local-coding-agent
status: implemented (Tasks 1-6; Task 7 live smoke pending on Ollama PC)
---

# Agentic Search Implementation Plan — Option A (no code index)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tool-calling agent loop — `grep`, `read_file`, `list_files` — so the local model can answer questions about a codebase by searching it live, with no code index at all (the Claude Code approach).

**Architecture:** Two new modules. `src/agent_tools.py` holds pure, read-only tool functions (path-sandboxed, output-capped) with zero Ollama dependency, so they test instantly. `src/agent.py` holds the Ollama tool schemas, a dispatch function, and the agent loop that lets the model call tools until it answers or hits guardrails (iteration cap, repeat-call nudge). The existing RAG pipeline is untouched; the agent is a parallel entry point (`python src/agent.py "..."` and `/agent ...` in chat).

**Tech Stack:** Python 3.12, `ollama` (tool calling via `ollama.chat(..., tools=...)`), pytest + monkeypatch. No ripgrep dependency — grep is implemented with Python `re` so it behaves identically on macOS and the Windows PC.

## Why this design (context)

This is milestone #16 in [[07-02-2026-repo-improvement-plan]], pulled forward as
its own night. Option A trade-offs, accepted knowingly:

- **Pros:** no embedding cost, never stale, exact identifier matches, fully
  private. This *is* the "build a real agent" learning milestone.
- **Cons:** a 12B-class model is much weaker than Claude at driving multi-step
  tool use, and grep dumps eat a small context window fast. The plan's answer
  to both: hard output caps on every tool, a terse system prompt with an
  explicit method ("grep → read → answer"), an iteration cap, and a
  repeated-call breaker — the guardrails the field guides recommend for local
  function calling.
- Semantic/conceptual questions stay with the existing RAG pipeline; this does
  not replace it, it sits beside it (RAG-as-a-tool can come later).

Key references: [Vadim's blog](https://vadim.blog/claude-code-no-indexing/),
[Aram on Medium](https://zerofilter.medium.com/why-claude-code-is-special-for-not-doing-rag-vector-search-agent-search-tool-calling-versus-41b9a6c0f4d9),
[Milvus counterpoint on token burn](https://milvus.io/blog/why-im-against-claude-codes-grep-only-retrieval-it-just-burns-too-many-tokens.md),
[grep vs semantic nuance](https://www.nuss-and-bolts.com/p/on-the-lost-nuance-of-grep-vs-semantic),
[Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling),
[local function-calling guardrails](https://insiderllm.com/guides/function-calling-local-llms/).

## Global Constraints

- All tests must pass without Ollama running — Ollama lives on the other PC.
  Mock `agent.ollama.chat` with `monkeypatch`, never call the network.
- Tools are strictly read-only and path-sandboxed: any path resolving outside
  the project root is refused. No shell subprocesses anywhere.
- Output caps (small-context protection): `MAX_GREP_MATCHES = 40`,
  `MAX_FILE_CHARS = 8_000`, `MAX_ITERATIONS = 8`. Exact names/values as shown.
- Agent model constant: `AGENT_MODEL = "qwen2.5-coder:3b"` in `src/agent.py`
  (env-var override is repo-plan item #3, out of scope here).
- Tool errors are returned to the model as strings, never raised out of the
  loop — a bad regex from the model must not crash the chat.
- Python 3.12, tests in `tests/`, `pytest` from the repo root. Tests import
  `agent_tools` / `agent` bare (same mechanism as the existing `import rag`).
- Commit after every task.

---

## File Structure

- Create `src/agent_tools.py` — pure tool functions: `list_files`,
  `grep_files`, `read_file`, plus the `_resolve_inside` sandbox helper.
- Create `src/agent.py` — `TOOL_SCHEMAS`, `SYSTEM_PROMPT`, `dispatch_tool`,
  `run_agent` loop, CLI `main`.
- Modify `src/ask.py` — add `/agent <question>` command to the chat loop.
- Create `tests/test_agent_tools.py`, `tests/test_agent.py`.

---

### Task 1: Sandbox helper + `list_files` tool

**Files:**
- Create: `src/agent_tools.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `list_files(root: Path, subdir: str = ".") -> str`,
  `_resolve_inside(root: Path, relative: str) -> Path` (raises `ValueError` on
  escape), module constants `SKIP_DIRS`, `TEXT_SUFFIXES`. Later tasks reuse
  `_resolve_inside` and `_iter_text_files`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_tools.py`:

```python
from pathlib import Path

import pytest

import agent_tools


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A tiny fake repo: text files, a .git dir, and a binary file."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def retrieve():\n    return 4\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# Demo project\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    return tmp_path


def test_list_files_returns_relative_text_files_only(project: Path) -> None:
    listing = agent_tools.list_files(project)

    assert "src/main.py" in listing
    assert "README.md" in listing
    assert ".git" not in listing
    assert "logo.png" not in listing


def test_list_files_scopes_to_a_subdir(project: Path) -> None:
    listing = agent_tools.list_files(project, "src")

    assert "src/main.py" in listing
    assert "README.md" not in listing


def test_list_files_reports_missing_subdir(project: Path) -> None:
    assert "No such directory" in agent_tools.list_files(project, "nope")


def test_paths_outside_root_are_refused(project: Path) -> None:
    with pytest.raises(ValueError):
        agent_tools.list_files(project, "../..")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent_tools'`

- [ ] **Step 3: Write the implementation**

Create `src/agent_tools.py`:

```python
from pathlib import Path

# Output caps protect the small model's context window: a tool result that
# does not fit in context is worse than no result at all.
MAX_GREP_MATCHES = 40
MAX_FILE_CHARS = 8_000

SKIP_DIRS = {".git", "__pycache__", "chroma_db", ".venv", "node_modules"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini"}


def _resolve_inside(root: Path, relative: str) -> Path:
    """Resolve a relative path, refusing anything that escapes the root."""
    target = (root / relative).resolve()

    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path '{relative}' is outside the project root.")

    return target


def _iter_text_files(base: Path):
    """Yield readable text files under base, skipping caches and binaries."""
    for path in sorted(base.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            yield path


def list_files(root: Path, subdir: str = ".") -> str:
    """List the project's text files, relative to the root."""
    base = _resolve_inside(root, subdir)

    if not base.is_dir():
        return f"No such directory: {subdir}"

    lines = [
        path.relative_to(root).as_posix() for path in _iter_text_files(base)
    ]

    if not lines:
        return f"No text files under {subdir}."

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_tools.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_tools.py tests/test_agent_tools.py
git commit -m "feat: add sandboxed list_files agent tool"
```

---

### Task 2: `grep_files` tool

**Files:**
- Modify: `src/agent_tools.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Consumes: `_resolve_inside`, `_iter_text_files`, `MAX_GREP_MATCHES` from Task 1.
- Produces: `grep_files(root: Path, pattern: str, subdir: str = ".") -> str`
  returning `path:line: text` lines, a "No matches" message, or an
  "Invalid regex" message.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_tools.py`:

```python
def test_grep_reports_path_line_and_text(project: Path) -> None:
    result = agent_tools.grep_files(project, r"def retrieve")

    assert "src/main.py:1: def retrieve():" in result


def test_grep_scopes_to_a_subdir(project: Path) -> None:
    result = agent_tools.grep_files(project, "Demo", "src")

    assert "No matches" in result


def test_grep_reports_when_nothing_matches(project: Path) -> None:
    assert "No matches" in agent_tools.grep_files(project, "unicorn")


def test_grep_reports_invalid_regex_instead_of_raising(project: Path) -> None:
    assert "Invalid regex" in agent_tools.grep_files(project, "(")


def test_grep_caps_the_match_count(project: Path) -> None:
    (project / "big.txt").write_text("hit\n" * 200, encoding="utf-8")

    result = agent_tools.grep_files(project, "hit")

    assert len(result.splitlines()) == agent_tools.MAX_GREP_MATCHES + 1
    assert "capped" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_tools.py -v -k grep`
Expected: FAIL with `AttributeError: ... has no attribute 'grep_files'`

- [ ] **Step 3: Write the implementation**

Add to `src/agent_tools.py` (add `import re` at the top, above
`from pathlib import Path`):

```python
def grep_files(root: Path, pattern: str, subdir: str = ".") -> str:
    """Search file contents with a regex; returns 'path:line: text' matches."""
    try:
        compiled = re.compile(pattern)
    except re.error as error:
        return f"Invalid regex '{pattern}': {error}"

    base = _resolve_inside(root, subdir)
    matches: list[str] = []

    for path in _iter_text_files(base):
        text = path.read_text(encoding="utf-8", errors="replace")

        for line_number, line in enumerate(text.splitlines(), start=1):
            if not compiled.search(line):
                continue

            relative = path.relative_to(root).as_posix()
            matches.append(f"{relative}:{line_number}: {line.strip()}")

            if len(matches) >= MAX_GREP_MATCHES:
                matches.append(
                    f"... capped at {MAX_GREP_MATCHES} matches; use a more specific pattern."
                )
                return "\n".join(matches)

    if not matches:
        return f"No matches for '{pattern}'."

    return "\n".join(matches)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_tools.py -v`
Expected: 9 PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_tools.py tests/test_agent_tools.py
git commit -m "feat: add capped regex grep agent tool"
```

---

### Task 3: `read_file` tool with pagination

**Files:**
- Modify: `src/agent_tools.py`
- Test: `tests/test_agent_tools.py`

**Interfaces:**
- Consumes: `_resolve_inside`, `MAX_FILE_CHARS` from Task 1.
- Produces: `read_file(root: Path, path: str, start_line: int = 1) -> str`
  returning `N: line` numbered text; long files truncate with a
  `start_line=` resume hint so the model can page through.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent_tools.py`:

```python
def test_read_file_numbers_every_line(project: Path) -> None:
    result = agent_tools.read_file(project, "src/main.py")

    assert result.startswith("1: def retrieve():")
    assert "2:     return 4" in result


def test_read_file_starts_at_the_requested_line(project: Path) -> None:
    result = agent_tools.read_file(project, "src/main.py", start_line=2)

    assert result.startswith("2:")
    assert "1:" not in result


def test_read_file_reports_missing_files(project: Path) -> None:
    assert "No such file" in agent_tools.read_file(project, "nope.py")


def test_read_file_truncates_with_a_resume_hint(project: Path) -> None:
    (project / "long.md").write_text("some line here\n" * 5000, encoding="utf-8")

    result = agent_tools.read_file(project, "long.md")

    assert len(result) <= agent_tools.MAX_FILE_CHARS + 100
    assert "truncated" in result
    assert "start_line=" in result


def test_read_file_reports_start_line_past_the_end(project: Path) -> None:
    result = agent_tools.read_file(project, "src/main.py", start_line=99)

    assert "has 2 lines" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_tools.py -v -k read_file`
Expected: FAIL with `AttributeError: ... has no attribute 'read_file'`

- [ ] **Step 3: Write the implementation**

Add to `src/agent_tools.py`:

```python
def read_file(root: Path, path: str, start_line: int = 1) -> str:
    """Read one file with numbered lines, truncating with a resume hint."""
    target = _resolve_inside(root, path)

    if not target.is_file():
        return f"No such file: {path}"

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    picked: list[str] = []
    used = 0

    for number, line in enumerate(lines[start_line - 1 :], start=start_line):
        numbered = f"{number}: {line}"

        if used + len(numbered) > MAX_FILE_CHARS:
            picked.append(
                f"... truncated; call read_file again with start_line={number} for the rest."
            )
            break

        picked.append(numbered)
        used += len(numbered) + 1

    if not picked:
        return f"{path} has no line {start_line}; the file has {len(lines)} lines."

    return "\n".join(picked)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_tools.py -v`
Expected: 14 PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_tools.py tests/test_agent_tools.py
git commit -m "feat: add paginated read_file agent tool"
```

---

### Task 4: Tool schemas + dispatch

**Files:**
- Create: `src/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `list_files`, `grep_files`, `read_file` from Tasks 1–3.
- Produces: `TOOL_SCHEMAS: list[dict]` (Ollama function-calling format, tool
  names `list_files` / `grep` / `read_file`),
  `dispatch_tool(name: str, arguments: dict, root: Path) -> str` which never
  raises — bad input comes back as a `"Tool error: ..."` string.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: Write the implementation**

Create `src/agent.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent.py tests/test_agent.py
git commit -m "feat: add agent tool schemas and dispatch"
```

---

### Task 5: Agent loop with guardrails

**Files:**
- Modify: `src/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `dispatch_tool`, `TOOL_SCHEMAS`, `SYSTEM_PROMPT`, `AGENT_MODEL`,
  `MAX_ITERATIONS` from Task 4; `ollama.chat` (mocked in tests).
- Produces: `run_agent(question: str, root: Path, max_iterations: int = MAX_ITERATIONS) -> tuple[str, list[str]]`
  returning `(answer_text, trace)` where trace entries look like
  `"grep({'pattern': 'def retrieve'})"`. Tool results go back to the model as
  `{"role": "tool", "tool_name": name, "content": result}` messages.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent.py` (top of file already imports `agent`):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v -k "run_agent or loop or nudge or unknown_tool_calls or direct"`
Expected: FAIL with `AttributeError: module 'agent' has no attribute 'run_agent'`

- [ ] **Step 3: Write the implementation**

Add to `src/agent.py` (add `import json` and `from typing import Any` at the
top). Note: the real Ollama client returns pydantic objects, but they support
dict-style access (`response["message"]`, `.get()`) exactly like the dicts the
tests use — same pattern `rag.ask_model` already relies on.

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: 12 PASS

- [ ] **Step 5: Run the whole suite**

Run: `pytest`
Expected: all tests PASS (42 existing + 19 new)

- [ ] **Step 6: Commit**

```bash
git add src/agent.py tests/test_agent.py
git commit -m "feat: add tool-calling agent loop with iteration and repeat guardrails"
```

---

### Task 6: CLI entry + `/agent` chat command

**Files:**
- Modify: `src/agent.py` (add `main`)
- Modify: `src/ask.py` (add `/agent` handling in `chat_loop`)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `run_agent` from Task 5.
- Produces: `parse_agent_command(line: str) -> str | None` and
  `format_agent_reply(answer: str, trace: list[str]) -> str` in `src/agent.py`;
  CLI `python src/agent.py "question"`; `/agent <question>` inside the chat.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent.py`:

```python
def test_parse_agent_command_extracts_the_question() -> None:
    assert agent.parse_agent_command("/agent where is retrieve?") == "where is retrieve?"


def test_parse_agent_command_ignores_other_lines() -> None:
    assert agent.parse_agent_command("where is retrieve?") is None
    assert agent.parse_agent_command("/agent") is None
    assert agent.parse_agent_command("/agent   ") is None


def test_format_agent_reply_shows_the_tool_trace() -> None:
    reply = agent.format_agent_reply(
        "It is in rag.py.", ["grep({'pattern': 'retrieve'})"]
    )

    assert "grep({'pattern': 'retrieve'})" in reply
    assert reply.endswith("It is in rag.py.")


def test_format_agent_reply_without_tools_is_just_the_answer() -> None:
    assert agent.format_agent_reply("Hi.", []) == "Hi."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v -k "parse_agent or format_agent"`
Expected: FAIL with `AttributeError: module 'agent' has no attribute 'parse_agent_command'`

- [ ] **Step 3: Write the implementation**

Add to `src/agent.py` (add `import sys` at the top):

```python
def parse_agent_command(line: str) -> str | None:
    """Extract the question from '/agent <question>', if this line is one."""
    stripped = line.strip()

    if not stripped.startswith("/agent"):
        return None

    question = stripped.removeprefix("/agent").strip()

    return question or None


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
```

In `src/ask.py`, add the import near the top:

```python
from agent import format_agent_reply, parse_agent_command, run_agent
```

and inside `chat_loop`, right after the `/exit` check and before
`apply_source_command`:

```python
        agent_question = parse_agent_command(question)
        if agent_question is not None:
            try:
                answer, trace = run_agent(agent_question, root=Path.cwd())
            except Exception as error:
                print(f"\nError: {error}")
                continue

            print("\n" + format_agent_reply(answer, trace))
            continue
```

(`ask.py` needs `from pathlib import Path` added to its imports.) Also update
the greeting line in `chat_loop` to mention the new command:

```python
    print("Scope answers with /sources, /source <name>, /source all.")
    print("Search this codebase live with /agent <question>.")
```

- [ ] **Step 4: Run the whole suite**

Run: `pytest`
Expected: all PASS (23 new tests total across both new files)

- [ ] **Step 5: Commit**

```bash
git add src/agent.py src/ask.py tests/test_agent.py
git commit -m "feat: expose agent via CLI and /agent chat command"
```

---

### Task 7: Live smoke test (needs the Ollama PC — manual)

No code; this is the acceptance run for the next session on the PC.

- [ ] **Step 1:** `ollama pull qwen2.5-coder:3b` (already present) and confirm
  the model supports tools: `ollama show qwen2.5-coder:3b` should list
  `tools` under capabilities.
- [ ] **Step 2:** From the repo root run:
  `python src/agent.py "where is the similarity search done?"`
  Expected shape: 1–3 tool calls in the trace (a grep for something like
  `query` or `retrieve`, maybe a read of `src/rag.py`), then an answer citing
  `src/rag.py` line numbers.
- [ ] **Step 3:** Try a conceptual question ("how does chunking keep code
  fences intact?") and a miss ("where is the websocket server?") — the miss
  must end with "could not find", not an invented file.
- [ ] **Step 4:** If the 3B model flails (ignores tools, loops, malformed
  calls), retry with `qwen2.5-coder:7b` or `qwen3:8b` by editing
  `AGENT_MODEL` — record which model works in the vault note. Model
  configurability is repo-plan item #3 and lands separately.
- [ ] **Step 5:** Note observations in the vault
  (`local-coding-agent/agent-smoke-notes.md`): tool-call count per question,
  wrong-tool choices, context overflows. These feed the next iteration
  (RAG-as-a-tool, repo-plan #16 full milestone).

---

## Self-review notes

- Spec coverage: grep/read/list tools ✔ (Tasks 1–3), model-driven loop ✔
  (Task 5), small-model guardrails (iteration cap, repeat breaker, output
  caps, terse prompt) ✔ (Tasks 1–5), token-hunger mitigation ✔ (caps +
  pagination), CLI/chat integration ✔ (Task 6), live validation ✔ (Task 7).
- Conceptual-question weakness is acknowledged, not solved: existing RAG stays
  the default path; `/agent` is opt-in per question.
- Type consistency: `run_agent` returns `tuple[str, list[str]]` everywhere;
  trace entry format `"name({...})"` matches between Task 5 loop code, its
  tests, and Task 6's `format_agent_reply` test.
