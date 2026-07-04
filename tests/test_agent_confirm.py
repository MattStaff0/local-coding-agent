"""The loop owns the y/n gate; tools never touch files without approval."""
from pathlib import Path

import agent


def test_declined_edit_leaves_file_untouched(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\n", encoding="utf-8")

    result = agent.dispatch_tool(
        "edit_file",
        {"path": "f.py", "old_text": "a = 1", "new_text": "a = 2"},
        tmp_path,
        confirm=lambda description, preview: False,
    )

    assert result == "User declined the change."
    assert (tmp_path / "f.py").read_text() == "a = 1\n"


def test_approved_edit_applies_and_reports_diff(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\n", encoding="utf-8")
    seen = {}

    def confirm(description, preview):
        seen["description"], seen["preview"] = description, preview
        return True

    result = agent.dispatch_tool(
        "edit_file",
        {"path": "f.py", "old_text": "a = 1", "new_text": "a = 2"},
        tmp_path,
        confirm=confirm,
    )

    assert (tmp_path / "f.py").read_text() == "a = 2\n"
    assert "+a = 2" in result
    assert "edit_file f.py" in seen["description"]
    assert "+a = 2" in seen["preview"]


def test_no_confirm_channel_declines_writes(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\n", encoding="utf-8")
    result = agent.dispatch_tool(
        "edit_file",
        {"path": "f.py", "old_text": "a = 1", "new_text": "a = 2"},
        tmp_path,
        confirm=None,
    )
    assert "No confirmation channel" in result
    assert (tmp_path / "f.py").read_text() == "a = 1\n"


def test_read_tools_never_ask(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\n", encoding="utf-8")

    def exploding_confirm(description, preview):
        raise AssertionError("read tools must not prompt")

    result = agent.dispatch_tool(
        "read_file", {"path": "f.py"}, tmp_path, confirm=exploding_confirm
    )
    assert "a = 1" in result
