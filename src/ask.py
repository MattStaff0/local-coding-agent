import sys
from typing import Any

from rag import answer_question, list_sources


def print_sources(metadatas: list[dict[str, Any]]) -> None:
    """Show which chunks Chroma retrieved for the current question."""
    print("\nRetrieved sources:")

    for metadata in metadatas:
        path = metadata.get("path", metadata["source"])
        print(f"- [{metadata['source']}] {path} chunk {metadata['chunk_index']}")


def apply_source_command(
    line: str,
    active_source: str | None,
) -> tuple[bool, str | None, str]:
    """Handle /sources and /source commands.

    Returns (handled, new_active_source, message to print).
    """
    stripped = line.strip()

    if stripped == "/sources":
        names = ", ".join(list_sources())
        return True, active_source, f"Available sources: {names}"

    if stripped == "/source" or stripped.startswith("/source "):
        parts = stripped.split(maxsplit=1)

        if len(parts) == 1:
            current = active_source or "all"
            return True, active_source, f"Current source: {current}"

        name = parts[1].strip()

        if name == "all":
            return True, None, "Searching all sources."

        available = list_sources()
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

    while True:
        question = input("\nYou: ").strip()

        # Ignore blank lines so accidental Enter presses do not call the model.
        if not question:
            continue

        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye.")
            return

        handled, active_source, message = apply_source_command(
            question, active_source
        )
        if handled:
            print(message)
            continue

        try:
            answer, metadatas = answer_question(question, history, active_source)
        except Exception as error:
            print(f"\nError: {error}")
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
    # python src\ask.py "How do I make a PyTorch model?"
    question = " ".join(sys.argv[1:])
    answer, metadatas = answer_question(question, history=[])

    print_sources(metadatas)
    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()
