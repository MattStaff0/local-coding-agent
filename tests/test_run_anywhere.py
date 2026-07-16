"""Run-anywhere workflow: lca doctor, --root, and no stray files elsewhere."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import ask
import paths
import rag
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
    # Patch the code manifest too so the test never reads real on-disk state.
    monkeypatch.setattr(rag, "CODE_MANIFEST_PATH", tmp_path / "missing-code.jsonl")

    ask.doctor()
    out = capsys.readouterr().out
    assert "3 chunks" in out
    assert "2 sources" in out


def test_doctor_survives_corrupt_manifest_line(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.jsonl"
    # A torn write: valid line, then a truncated one.
    manifest.write_text('{"id": "a-0", "source": "numpy"}\n{"id": "a-1", "sour\n')
    monkeypatch.setattr(paths, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(rag, "CODE_MANIFEST_PATH", tmp_path / "missing-code.jsonl")

    ask.doctor()

    out = capsys.readouterr().out
    assert "unreadable" in out
    # The crash must not truncate the report: later lines still print.
    assert "code index" in out
    assert "agent root" in out


def test_doctor_survives_non_dict_manifest_records(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.jsonl"
    # Valid JSON lines that are not objects must not crash the summary.
    manifest.write_text('{"id": "a-0", "source": "numpy"}\n[1, 2, 3]\n')
    monkeypatch.setattr(paths, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(rag, "CODE_MANIFEST_PATH", tmp_path / "missing-code.jsonl")

    ask.doctor()

    assert "2 chunks" in capsys.readouterr().out


def test_doctor_distinguishes_empty_manifest_from_missing(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")
    monkeypatch.setattr(paths, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(rag, "CODE_MANIFEST_PATH", tmp_path / "missing-code.jsonl")

    ask.doctor()

    out = capsys.readouterr().out
    # An index that ingested zero chunks is a different problem than one
    # that was never built; doctor must not conflate them.
    assert "empty" in out
    assert "not built" in out


def test_doctor_reports_reranker_on(tmp_path, monkeypatch, capsys):
    import rerank

    monkeypatch.setenv("RAG_RERANKER", "cross-encoder")
    monkeypatch.setattr(paths, "MANIFEST_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(rag, "CODE_MANIFEST_PATH", tmp_path / "missing-code.jsonl")

    ask.doctor()

    out = capsys.readouterr().out
    assert f"on ({rerank.RERANK_MODEL})" in out


def test_doctor_reports_configured_models(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "MANIFEST_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(rag, "CODE_MANIFEST_PATH", tmp_path / "missing-code.jsonl")

    ask.doctor()

    out = capsys.readouterr().out
    assert rag.CHAT_MODEL in out
    assert rag.EMBED_MODEL in out


def test_doctor_reports_missing_indexes_without_crashing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "MANIFEST_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr(rag, "CODE_MANIFEST_PATH", tmp_path / "missing-code.jsonl")

    ask.doctor()
    out = capsys.readouterr().out
    assert "not built" in out


def test_main_doctor_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["lca", "doctor"])
    ask.main()
    assert str(paths.PROJECT_ROOT) in capsys.readouterr().out


def test_main_doctor_rejects_trailing_args(monkeypatch, capsys):
    # An unquoted question starting with "doctor" must error loudly, not
    # run the diagnostic and silently discard the rest of the question.
    monkeypatch.setattr(sys, "argv", ["lca", "doctor", "visits", "in", "pandas"])

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 2
    assert "doctor takes no arguments" in capsys.readouterr().out


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
    assert "no such directory" in capsys.readouterr().out


def test_main_root_rejects_file_path_with_honest_message(monkeypatch, tmp_path, capsys):
    afile = tmp_path / "afile.txt"
    afile.write_text("hi")
    monkeypatch.setattr(sys, "argv", ["lca", "--root", str(afile)])

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 2
    out = capsys.readouterr().out
    # The path exists — the error must say "not a directory", not deny it.
    assert "not a directory" in out


def test_main_root_requires_a_path(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["lca", "--root"])

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 2


def test_main_root_rejects_trailing_args(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys, "argv", ["lca", "--root", str(tmp_path), "stray", "question"]
    )

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 2
    assert "chat mode" in capsys.readouterr().out


def test_main_root_expands_tilde(monkeypatch, tmp_path):
    (tmp_path / "proj").mkdir()
    seen = {}

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(ask, "chat_loop", lambda agent_root=None: seen.update(root=agent_root))
    monkeypatch.setattr(sys, "argv", ["lca", "--root", "~/proj"])

    ask.main()

    assert seen["root"] == (tmp_path / "proj").resolve()


def test_main_rejects_unknown_options(monkeypatch, capsys):
    # Unknown flags must not be silently shipped to the model as a question.
    monkeypatch.setattr(sys, "argv", ["lca", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 2
    assert "Unknown option" in capsys.readouterr().out


def test_chat_loop_agent_root_preset_shows_in_status(monkeypatch, tmp_path):
    renderer = _RecordingRenderer()
    # Bare "/agent" is the status command; "/agent status" would send the
    # literal word "status" to the model as a question.
    inputs = iter(["/agent", "/exit"])

    monkeypatch.setattr(ask, "load_history", lambda path: [])
    monkeypatch.setattr(ask, "save_history", lambda history, path: None)
    monkeypatch.setattr(ask, "start_mcp", lambda: None)

    ask.chat_loop(
        renderer=renderer,
        read_input=lambda prompt_text: next(inputs),
        agent_root=tmp_path,
    )

    # Assert the /agent status reply specifically — the startup banner also
    # prints the root, so a bare substring check would pass with a broken
    # status command.
    assert f"Agent root: {tmp_path} (0 messages)" in renderer.messages


# --- console-script entry points ---


def test_console_script_entry_points_exist():
    # pyproject.toml [project.scripts] targets; a rename in any of these
    # modules would ship a broken installed command with no red test.
    import importlib
    import tomllib

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text())["project"]["scripts"]

    assert set(scripts) == {"lca", "lca-fetch-docs", "lca-ingest", "lca-ingest-code"}
    for target in scripts.values():
        module_name, attr = target.split(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attr))


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
