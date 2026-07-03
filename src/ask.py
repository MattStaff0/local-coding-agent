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

from agent import format_agent_reply, parse_agent_command, run_agent
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
HISTORY_FILE = Path("chat_history.json")
MAX_SAVED_MESSAGES = 100

# Where /export writes study notes. Point this at the Obsidian vault, e.g.
# STUDY_NOTES_DIR="$HOME/Documents/matt-vault/study-notes".
EXPORT_DIR = Path(os.getenv("STUDY_NOTES_DIR", "study-notes"))


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


def print_sources(metadatas: list[dict[str, Any]]) -> None:
    """Show the citation legend: which chunk each [n] in the answer refers to."""
    print("\nSources:")

    for line in source_legend(metadatas):
        print(f"  {line}")


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


def chat_loop() -> None:
    """Run an interactive terminal chat with persistent memory."""
    # Pass the module globals explicitly: default arguments bind at def time,
    # which would ignore a HISTORY_FILE override (tests, future config).
    history = load_history(HISTORY_FILE)
    active_source: str | None = None
    last_export: tuple[str, str, list[dict[str, Any]]] | None = None

    print("Local RAG chat")
    print("Type your question, or type /exit to quit.")
    print("Scope answers with /sources, /source <name>, /source all.")
    print("Search this codebase live with /agent <question>.")
    print("Save the last answer as a study note with /export.")

    if history:
        print(f"(restored {len(history)} messages from {HISTORY_FILE})")

    while True:
        question = input("\nYou: ").strip()

        # Ignore blank lines so accidental Enter presses do not call the model.
        if not question:
            continue

        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye.")
            return

        if question == "/export":
            if last_export is None:
                print("Nothing to export yet — ask a question first.")
                continue

            try:
                path = export_note(*last_export, notes_dir=EXPORT_DIR)
            except OSError as error:
                # A bad STUDY_NOTES_DIR must not kill the whole chat session.
                print(f"Could not write the study note ({error}).")
                continue

            print(f"Saved study note: {path}")
            continue

        agent_question = parse_agent_command(question)
        if agent_question is not None:
            try:
                answer, trace = run_agent(agent_question, root=Path.cwd())
            except Exception as error:
                print(f"\n{describe_error(error)}")
                continue

            print("\n" + format_agent_reply(answer, trace))
            continue

        handled, active_source, message = apply_source_command(
            question, active_source
        )
        if handled:
            print(message)
            continue

        print("\nAssistant:\n")

        try:
            # The answer streams to the terminal as the model writes it.
            answer, metadatas = answer_question(
                question, history, active_source, on_token=print_token
            )
        except Exception as error:
            print(describe_error(error))
            continue

        print()
        print_sources(metadatas)

        # Save the turn after the model answers so follow-up questions have
        # enough context to understand words like "that" or "it".
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        last_export = (question, answer, metadatas)

        try:
            save_history(history, HISTORY_FILE)
        except OSError as error:
            # Losing persistence is worth a warning, not a dead session.
            print(f"(could not save chat history: {error})")


def main() -> None:
    """Use chat mode by default, or answer a single command-line question."""
    if len(sys.argv) < 2:
        chat_loop()
        return

    # This keeps the old one-shot usage:
    # python src/ask.py "How do I make a PyTorch model?"
    question = " ".join(sys.argv[1:])

    print("Answer:\n")

    try:
        answer, metadatas = answer_question(question, history=[], on_token=print_token)
    except Exception as error:
        print(describe_error(error))
        raise SystemExit(1)

    print()
    print_sources(metadatas)


if __name__ == "__main__":
    main()
