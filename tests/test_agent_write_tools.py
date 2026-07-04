"""Write tools are pure: they compute diffs; the loop owns confirmation."""
import pytest

from agent_tools import apply_content, preview_edit, preview_write, run_command


def test_preview_edit_produces_unified_diff(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\nb = 2\n", encoding="utf-8")

    result = preview_edit(tmp_path, "f.py", "b = 2", "b = 3")

    assert "error" not in result
    assert "-b = 2" in result["diff"] and "+b = 3" in result["diff"]
    assert result["new_content"] == "a = 1\nb = 3\n"


def test_preview_edit_rejects_ambiguous_match(tmp_path):
    (tmp_path / "f.py").write_text("x\nx\n", encoding="utf-8")
    result = preview_edit(tmp_path, "f.py", "x", "y")
    assert "2 times" in result["error"]


def test_preview_edit_rejects_missing_text(tmp_path):
    (tmp_path / "f.py").write_text("a\n", encoding="utf-8")
    assert "not found" in preview_edit(tmp_path, "f.py", "zzz", "y")["error"]


def test_preview_write_diffs_against_empty_for_new_file(tmp_path):
    result = preview_write(tmp_path, "new.py", "print('hi')\n")
    assert "+print('hi')" in result["diff"]


def test_apply_content_writes_inside_root_only(tmp_path):
    assert "Wrote" in apply_content(tmp_path, "sub/new.py", "x = 1\n")
    assert (tmp_path / "sub" / "new.py").read_text() == "x = 1\n"

    with pytest.raises(ValueError):
        apply_content(tmp_path, "../escape.py", "bad")


def test_disallowed_command_is_refused_without_running(tmp_path):
    result = run_command(tmp_path, "rm -rf /")
    assert "not allowed" in result
    assert list(tmp_path.iterdir()) == []  # nothing happened


def test_python_runs_in_root_with_output(tmp_path):
    (tmp_path / "hello.py").write_text("print('hi from test')", encoding="utf-8")
    result = run_command(tmp_path, "python hello.py")
    # allowlist matches basename, so plain "python hello.py" is the call shape
    assert "exit code 0" in result
    assert "hi from test" in result


def test_empty_and_malformed_commands_are_errors(tmp_path):
    assert "not allowed" in run_command(tmp_path, "")
    # shlex keeps "pytest;" one arg -> basename mismatch, refused without crash
    assert "not allowed" in run_command(tmp_path, "pytest; rm x")
