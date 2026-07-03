"""Persistent chat history + /export study notes (repo plan #17).

Chat memory survives restarts, and any good answer can be exported as a
markdown study note (pointed at the Obsidian vault via STUDY_NOTES_DIR).
"""

import json
from pathlib import Path

import ask


def test_load_history_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert ask.load_history(tmp_path / "nope.json") == []


def test_history_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]

    ask.save_history(history, path)

    assert ask.load_history(path) == history


def test_corrupt_history_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("{not json", encoding="utf-8")

    assert ask.load_history(path) == []


def test_save_history_keeps_only_the_most_recent_messages(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    history = [{"role": "user", "content": str(i)} for i in range(200)]

    ask.save_history(history, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved) == ask.MAX_SAVED_MESSAGES
    assert saved[-1]["content"] == "199"


def test_export_note_writes_question_answer_and_citations(tmp_path: Path) -> None:
    metadatas = [
        {"path": "docs/pytorch/tensors.md", "heading": "Tensors", "source": "pytorch"}
    ]

    path = ask.export_note(
        question="How do I make a tensor?",
        answer="Use torch.tensor(data) [1].",
        metadatas=metadatas,
        notes_dir=tmp_path,
    )

    content = path.read_text(encoding="utf-8")
    assert path.parent == tmp_path
    assert "how-do-i-make-a-tensor" in path.name
    assert "How do I make a tensor?" in content
    assert "Use torch.tensor(data) [1]." in content
    assert "docs/pytorch/tensors.md § Tensors" in content


def test_export_note_filenames_do_not_collide(tmp_path: Path) -> None:
    first = ask.export_note("same q", "a1", [], notes_dir=tmp_path)
    second = ask.export_note("same q", "a2", [], notes_dir=tmp_path)

    assert first != second
    assert first.exists() and second.exists()
