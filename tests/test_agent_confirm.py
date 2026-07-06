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


def test_declined_run_command_never_executes(tmp_path):
    result = agent.dispatch_tool(
        "run_command",
        {"command": "python -c \"open('proof.txt','w').write('ran')\""},
        tmp_path,
        confirm=lambda description, preview: False,
    )

    assert result == "User declined the change."
    assert not (tmp_path / "proof.txt").exists()


def test_approved_run_command_executes(tmp_path):
    (tmp_path / "ok.py").write_text("print('ran fine')", encoding="utf-8")
    seen = {}

    def confirm(description, preview):
        seen["description"] = description
        return True

    result = agent.dispatch_tool(
        "run_command", {"command": "python ok.py"}, tmp_path, confirm=confirm
    )

    assert "exit code 0" in result and "ran fine" in result
    assert seen["description"] == "run: python ok.py"


def test_run_command_without_channel_is_declined(tmp_path):
    result = agent.dispatch_tool(
        "run_command", {"command": "python -c pass"}, tmp_path, confirm=None
    )
    assert "No confirmation channel" in result


def test_write_file_routes_through_the_gate(tmp_path):
    declined = agent.dispatch_tool(
        "write_file",
        {"path": "new.py", "content": "x = 1\n"},
        tmp_path,
        confirm=lambda description, preview: False,
    )
    assert declined == "User declined the change."
    assert not (tmp_path / "new.py").exists()

    approved = agent.dispatch_tool(
        "write_file",
        {"path": "new.py", "content": "x = 1\n"},
        tmp_path,
        confirm=lambda description, preview: True,
    )
    assert "Applied:" in approved
    assert (tmp_path / "new.py").read_text() == "x = 1\n"


def test_read_tools_never_ask(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\n", encoding="utf-8")

    def exploding_confirm(description, preview):
        raise AssertionError("read tools must not prompt")

    result = agent.dispatch_tool(
        "read_file", {"path": "f.py"}, tmp_path, confirm=exploding_confirm
    )
    assert "a = 1" in result
