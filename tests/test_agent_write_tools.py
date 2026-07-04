"""Write tools are pure: they compute diffs; the loop owns confirmation."""
import pytest

from agent_tools import apply_content, preview_edit, preview_write


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
