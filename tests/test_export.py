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


def test_history_store_keeps_sessions_by_canonical_root(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    one = (tmp_path / "one").resolve()
    two = (tmp_path / "two").resolve()

    ask.save_history([{"role": "user", "content": "one"}], path, root=one)
    ask.save_history([{"role": "user", "content": "two"}], path, root=two)

    assert ask.load_history(path, root=one)[0]["content"] == "one"
    assert ask.load_history(path, root=two)[0]["content"] == "two"


def test_history_store_migrates_legacy_list_only_to_project_home(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "history.json"
    project_home = (tmp_path / "lca-home").resolve()
    foreign = (tmp_path / "foreign").resolve()
    legacy = [{"role": "user", "content": "legacy docs question"}]
    path.write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(ask.paths, "PROJECT_ROOT", project_home)

    assert ask.load_history(path, root=foreign) == []
    assert ask.load_history(path, root=project_home) == legacy

    ask.save_history([{"role": "user", "content": "foreign"}], path, root=foreign)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["version"] == 2
    assert stored["sessions"][str(project_home)] == legacy
    assert stored["sessions"][str(foreign)][0]["content"] == "foreign"


def test_history_projection_excludes_tool_messages_and_payloads() -> None:
    messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "read_file"}}],
        },
        {"role": "tool", "tool_name": "read_file", "content": "secret payload"},
        {"role": "assistant", "content": "final answer"},
    ]

    assert ask.clean_history(messages) == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "final answer"},
    ]


def test_history_cap_applies_per_root(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    root = tmp_path.resolve()
    history = [{"role": "user", "content": str(i)} for i in range(200)]

    ask.save_history(history, path, root=root)

    assert len(ask.load_history(path, root=root)) == ask.MAX_SAVED_MESSAGES
    assert ask.load_history(path, root=root)[-1]["content"] == "199"


def test_export_note_accepts_formatted_agent_sources(tmp_path: Path) -> None:
    path = ask.export_note(
        "How does concat work?",
        "Use concat [1].",
        ["[1] docs/pandas/merge.md § Concat"],
        notes_dir=tmp_path,
    )

    content = path.read_text(encoding="utf-8")
    assert "## Sources" in content
    assert "[1] docs/pandas/merge.md § Concat" in content


def test_project_only_export_omits_empty_sources_heading(tmp_path: Path) -> None:
    path = ask.export_note(
        "Where is retrieve?", "Evidence: src/rag.py:1", [], notes_dir=tmp_path
    )

    assert "## Sources" not in path.read_text(encoding="utf-8")


def test_root_scoped_save_backs_up_corrupt_history(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("{not json", encoding="utf-8")

    ask.save_history([], path, root=tmp_path.resolve())

    assert (tmp_path / "history.json.bak").read_text(encoding="utf-8") == "{not json"
    assert ask.load_history(path, root=tmp_path.resolve()) == []
