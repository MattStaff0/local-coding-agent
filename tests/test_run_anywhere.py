"""Run-anywhere workflow: lca doctor, --root, and no stray files elsewhere."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import ask
import paths
import ui

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


class _RecordingRenderer(ui.PlainRenderer):
    def __init__(self):
        self.messages = []

    def show_message(self, text):
        self.messages.append(text)


# --- lca doctor ---


def test_doctor_reports_home_env_and_models(capsys):
    ask.doctor()
    out = capsys.readouterr().out
    assert str(paths.PROJECT_ROOT) in out
    assert ".env" in out
    assert "OLLAMA_HOST" in out
    assert "chat model" in out
    assert "embed model" in out


def test_doctor_counts_manifest_chunks_and_sources(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.jsonl"
    records = [
        {"id": "a-0", "source": "numpy", "tokens": ["x"]},
        {"id": "a-1", "source": "numpy", "tokens": ["y"]},
        {"id": "b-0", "source": "pandas", "tokens": ["z"]},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    monkeypatch.setattr(paths, "MANIFEST_PATH", manifest)

    ask.doctor()
    out = capsys.readouterr().out
    assert "3 chunks" in out
    assert "2 sources" in out


def test_doctor_reports_missing_indexes_without_crashing(tmp_path, monkeypatch, capsys):
    import rag

    monkeypatch.setattr(paths, "MANIFEST_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(rag, "CODE_MANIFEST_PATH", tmp_path / "missing-code.jsonl")

    ask.doctor()
    out = capsys.readouterr().out
    assert "not built" in out


def test_main_doctor_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["lca", "doctor"])
    ask.main()
    assert str(paths.PROJECT_ROOT) in capsys.readouterr().out


# --- lca --root ---


def test_main_root_option_presets_agent_root(monkeypatch, tmp_path):
    seen = {}

    def fake_chat_loop(agent_root=None):
        seen["root"] = agent_root

    monkeypatch.setattr(ask, "chat_loop", fake_chat_loop)
    monkeypatch.setattr(sys, "argv", ["lca", "--root", str(tmp_path)])

    ask.main()

    assert seen["root"] == tmp_path.resolve()


def test_main_root_rejects_missing_directory(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["lca", "--root", str(tmp_path / "nope")])

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 2
    assert "No such directory" in capsys.readouterr().out


def test_main_root_requires_a_path(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["lca", "--root"])

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 2


def test_chat_loop_agent_root_preset_shows_in_status(monkeypatch, tmp_path):
    renderer = _RecordingRenderer()
    inputs = iter(["/agent status", "/exit"])

    monkeypatch.setattr(ask, "load_history", lambda path: [])
    monkeypatch.setattr(ask, "save_history", lambda history, path: None)
    monkeypatch.setattr(ask, "start_mcp", lambda: None)

    ask.chat_loop(
        renderer=renderer,
        read_input=lambda prompt_text: next(inputs),
        agent_root=tmp_path,
    )

    status = "\n".join(renderer.messages)
    assert str(tmp_path) in status


# --- launching from a foreign directory leaves it untouched ---


def test_doctor_from_foreign_cwd_leaves_no_stray_files(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SRC_DIR / "ask.py"), "doctor"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert str(paths.PROJECT_ROOT) in result.stdout
    # The launch directory must stay pristine: no chroma_db, no manifest.
    assert list(tmp_path.iterdir()) == []
