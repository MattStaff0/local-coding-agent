import json
import os
import re
import sys
import traceback
from datetime import date
from pathlib import Path
from typing import Any

import chromadb.errors
import httpx

import mcp_client
import paths
import ui
from agent import AgentSession, format_agent_reply, parse_agent_command, run_agent
from rag import (
    EmptyIndexError,
    NoRelevantDocsError,
    answer_question,
    list_sources,
    source_legend,
)

NO_INDEX_HINT = "No index found. Run 'python src/ingest.py' first."

# Chat history persists across restarts; only the recent tail is saved so the
# file cannot grow without bound (the prompt is capped separately by
# rag.MAX_HISTORY_TURNS).
HISTORY_FILE = paths.HISTORY_FILE
MAX_SAVED_MESSAGES = 100

# Where /export writes study notes. Point this at the Obsidian vault, e.g.
# STUDY_NOTES_DIR="$HOME/Documents/matt-vault/study-notes".
EXPORT_DIR = Path(os.getenv("STUDY_NOTES_DIR", paths.PROJECT_ROOT / "study-notes"))


def load_history(path: Path = HISTORY_FILE) -> list[dict[str, str]]:
    """Load saved chat history; only a missing file is a silent fresh start.

    An unreadable file is preserved as a .bak before the next save can
    overwrite it — chat history should never be destroyed silently.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as error:
        print(f"Could not read {path} ({error}) — starting with empty history.")
        return []

    try:
        history = json.loads(raw)
    except json.JSONDecodeError:
        history = None

    if isinstance(history, list):
        return history

    backup = path.with_suffix(path.suffix + ".bak")
    path.rename(backup)
    print(f"{path} is unreadable — backed up to {backup}, starting fresh.")
    return []


def save_history(
    history: list[dict[str, str]], path: Path = HISTORY_FILE
) -> None:
    """Persist the most recent chat messages to disk."""
    path.write_text(
        json.dumps(history[-MAX_SAVED_MESSAGES:], indent=2), encoding="utf-8"
    )


def export_note(
    question: str,
    answer: str,
    metadatas: list[dict[str, Any]],
    notes_dir: Path = EXPORT_DIR,
) -> Path:
    """Write one answered question as a markdown study note.

    Every good answer can become a vault note — that's the learning flywheel.
    """
    notes_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60] or "note"
    stem = f"{date.today().isoformat()}-{slug}"

    path = notes_dir / f"{stem}.md"
    counter = 2
    while path.exists():
        path = notes_dir / f"{stem}-{counter}.md"
        counter += 1

    lines = [
        "---",
        f"date: {date.today().isoformat()}",
        "kind: study-note",
        "---",
        "",
        f"# {question}",
        "",
        answer.strip(),
        "",
    ]

    if metadatas:
        lines.append("## Sources")
        lines.extend(f"- {line}" for line in source_legend(metadatas))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

    return path


def describe_error(error: Exception) -> str:
    """Turn a pipeline failure into an actionable message.

    Expected failures (no index yet, Ollama unreachable) get one-line hints;
    anything else is a bug, so keep the full traceback visible.
    """
    if isinstance(error, chromadb.errors.NotFoundError):
        return NO_INDEX_HINT

    if isinstance(error, (EmptyIndexError, NoRelevantDocsError)):
        return str(error)

    if isinstance(error, (httpx.TransportError, ConnectionError)):
        return f"Could not reach the local models ({error}). Is Ollama running?"

    return f"Unexpected error ({type(error).__name__}):\n{traceback.format_exc()}"


def print_token(token: str) -> None:
    """Print one streamed answer token immediately, without a newline."""
    print(token, end="", flush=True)


def safe_list_sources() -> list[str]:
    """Indexed source names for tab-completion; never crash the prompt."""
    try:
        return list_sources()
    except Exception:
        return []


def start_mcp():
    """Start MCP servers from mcp.json, once per chat process.

    Returns None when nothing is configured or startup fails — the agent
    then runs with native tools only, which is a degradation, not an error.
    """
    config = mcp_client.load_config(paths.PROJECT_ROOT / "mcp.json")

    if not config.get("servers"):
        return None

    try:
        manager = mcp_client.MCPManager(config)
        manager.start()
    except Exception as error:
        print(f"(mcp unavailable: {error})")
        return None

    return manager


HELP_TEXT = """\
Commands:
  /help             show this help
  /sources          list indexed doc sources
  /source <name>    answer only from one source (/source all to reset)
  /agent <question> search this codebase live with tools
  /export           save the last answer as a study note
  /exit             quit"""


def apply_source_command(
    line: str,
    active_source: str | None,
) -> tuple[bool, str | None, str]:
    """Handle /sources and /source commands.

    Returns (handled, new_active_source, message to print).
    """
    stripped = line.strip()

    if stripped == "/sources":
        try:
            names = ", ".join(list_sources())
        except chromadb.errors.NotFoundError:
            return True, active_source, NO_INDEX_HINT
        return True, active_source, f"Available sources: {names}"

    if stripped == "/source" or stripped.startswith("/source "):
        parts = stripped.split(maxsplit=1)

        if len(parts) == 1:
            current = active_source or "all"
            return True, active_source, f"Current source: {current}"

        name = parts[1].strip()

        if name == "all":
            return True, None, "Searching all sources."

        try:
            available = list_sources()
        except chromadb.errors.NotFoundError:
            return True, active_source, NO_INDEX_HINT
        if name not in available:
            return (
                True,
                active_source,
                f"Unknown source '{name}'. Available: {', '.join(available)}."
                + (f" Still scoped to '{active_source}'." if active_source else ""),
            )

        return True, name, f"Now answering only from '{name}' docs."

    return False, active_source, ""


def chat_loop(renderer=None, read_input=None) -> None:
    """Run an interactive terminal chat with persistent memory."""
    renderer = renderer or ui.make_renderer()
    read_input = read_input or ui.make_input(safe_list_sources)

    # Pass the module globals explicitly: default arguments bind at def time,
    # which would ignore a HISTORY_FILE override (tests, future config).
    history = load_history(HISTORY_FILE)
    active_source: str | None = None
    last_export: tuple[str, str, list[dict[str, Any]]] | None = None
    agent_session: AgentSession | None = None
    mcp_manager = None
    mcp_started = False

    renderer.show_message("Local RAG chat")
    renderer.show_message("Type your question, /help for commands, /exit to quit.")

    if history:
        renderer.show_message(f"(restored {len(history)} messages from {HISTORY_FILE})")

    while True:
        prompt_text = f"[{active_source}] You: " if active_source else "You: "
        question = read_input(prompt_text).strip()

        # Ignore blank lines so accidental Enter presses do not call the model.
        if not question:
            continue

        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            renderer.show_message("Goodbye.")
            return

        if question == "/help":
            renderer.show_message(HELP_TEXT)
            continue

        if question == "/export":
            if last_export is None:
                renderer.show_message("Nothing to export yet — ask a question first.")
                continue

            try:
                path = export_note(*last_export, notes_dir=EXPORT_DIR)
            except OSError as error:
                # A bad STUDY_NOTES_DIR must not kill the whole chat session.
                renderer.show_error(f"Could not write the study note ({error}).")
                continue

            renderer.show_message(f"Saved study note: {path}")
            continue

        agent_command = parse_agent_command(question)
        if agent_command is not None:
            subcommand, argument = agent_command

            if subcommand == "status":
                if agent_session is None:
                    renderer.show_message(
                        "No agent session yet. Ask with /agent <question>."
                    )
                else:
                    renderer.show_message(
                        f"Agent root: {agent_session.root} "
                        f"({len(agent_session.messages)} messages)"
                    )
                continue

            if subcommand == "reset":
                agent_session = None
                renderer.show_message("Agent session cleared.")
                continue

            if subcommand == "root":
                new_root = Path(argument).expanduser()
                if not new_root.is_dir():
                    renderer.show_error(f"No such directory: {argument}")
                    continue
                # Fresh session on purpose: old context describes the old
                # repo, and carrying it over invites cross-repo hallucination.
                agent_session = AgentSession(root=new_root)
                renderer.show_message(f"Agent root set to {new_root} (fresh session).")
                continue

            if agent_session is None:
                agent_session = AgentSession(root=Path.cwd())

            if not mcp_started:
                # One manager per chat process, started lazily so plain RAG
                # chats never pay the server-spawn cost.
                mcp_manager = start_mcp()
                mcp_started = True

            def confirm(description: str, preview: str) -> bool:
                renderer.show_message(f"\n{description}")
                if preview != description:
                    renderer.show_message(preview)
                return read_input("Apply? [y/N]: ").strip().lower() in {"y", "yes"}

            try:
                answer, trace = run_agent(
                    argument,
                    session=agent_session,
                    confirm=confirm,
                    mcp=mcp_manager,
                )
            except Exception as error:
                renderer.show_error(f"\n{describe_error(error)}")
                continue

            renderer.show_message("\n" + format_agent_reply(answer, trace))
            continue

        handled, active_source, message = apply_source_command(
            question, active_source
        )
        if handled:
            renderer.show_message(message)
            continue

        try:
            # The answer streams to the terminal as the model writes it.
            with renderer.status("thinking…"):
                answer, metadatas = answer_question(
                    question, history, active_source, on_token=renderer.on_token
                )
        except Exception as error:
            renderer.show_error(describe_error(error))
            continue

        renderer.finish_answer()
        renderer.show_sources(source_legend(metadatas))

        # Save the turn after the model answers so follow-up questions have
        # enough context to understand words like "that" or "it".
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        last_export = (question, answer, metadatas)

        try:
            save_history(history, HISTORY_FILE)
        except OSError as error:
            # Losing persistence is worth a warning, not a dead session.
            renderer.show_message(f"(could not save chat history: {error})")


def main() -> None:
    """Use chat mode by default, or answer a single command-line question."""
    if len(sys.argv) < 2:
        chat_loop()
        return

    # This keeps the old one-shot usage:
    # python src/ask.py "How do I make a PyTorch model?"
    question = " ".join(sys.argv[1:])

    renderer = ui.make_renderer()
    renderer.show_message("Answer:\n")

    try:
        answer, metadatas = answer_question(
            question, history=[], on_token=renderer.on_token
        )
    except Exception as error:
        renderer.show_error(describe_error(error))
        raise SystemExit(1)

    renderer.finish_answer()
    renderer.show_sources(source_legend(metadatas))


if __name__ == "__main__":
    main()
