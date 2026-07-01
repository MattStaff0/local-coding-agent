import sys
from typing import Any

from rag import answer_question


def print_sources(metadatas: list[dict[str, Any]]) -> None:
    """Show which chunks Chroma retrieved for the current question."""
    print("\nRetrieved sources:")

    for metadata in metadatas:
        print(f"- {metadata['source']} chunk {metadata['chunk_index']}")


def chat_loop() -> None:
    """Run an interactive terminal chat with temporary session memory."""
    # This list is the chat memory for the current terminal session only.
    # It disappears when you close the program.
    history: list[dict[str, str]] = []

    print("Local RAG chat")
    print("Type your question, or type /exit to quit.")

    while True:
        question = input("\nYou: ").strip()

        # Ignore blank lines so accidental Enter presses do not call the model.
        if not question:
            continue

        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye.")
            return

        try:
            answer, metadatas = answer_question(question, history)
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
