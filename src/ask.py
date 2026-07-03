import sys
import traceback
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
    """Run an interactive terminal chat with temporary session memory."""
    # This list is the chat memory for the current terminal session only.
    # It disappears when you close the program.
    history: list[dict[str, str]] = []
    active_source: str | None = None

    print("Local RAG chat")
    print("Type your question, or type /exit to quit.")
    print("Scope answers with /sources, /source <name>, /source all.")
    print("Search this codebase live with /agent <question>.")

    while True:
        question = input("\nYou: ").strip()

        # Ignore blank lines so accidental Enter presses do not call the model.
        if not question:
            continue

        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye.")
            return

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

        try:
            answer, metadatas = answer_question(question, history, active_source)
        except Exception as error:
            print(f"\n{describe_error(error)}")
            continue

        print_sources(metadatas)
        print("\nAssistant:\n")
        print(answer)

        # Save the turn after the model answers so follow-up questions have
        # enough context to understand words like "that" or "it".
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


def main() -> None:
    """Use chat mode by default, or answer a single command-line question."""
    if len(sys.argv) < 2:
        chat_loop()
        return

    # This keeps the old one-shot usage:
    # python src/ask.py "How do I make a PyTorch model?"
    question = " ".join(sys.argv[1:])

    try:
        answer, metadatas = answer_question(question, history=[])
    except Exception as error:
        print(describe_error(error))
        raise SystemExit(1)

    print_sources(metadatas)
    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()
