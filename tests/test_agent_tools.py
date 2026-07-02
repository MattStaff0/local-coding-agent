from pathlib import Path

import pytest

import agent_tools


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """A tiny fake repo: text files, a .git dir, and a binary file."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def retrieve():\n    return 4\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# Demo project\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    return tmp_path


def test_list_files_returns_relative_text_files_only(project: Path) -> None:
    listing = agent_tools.list_files(project)

    assert "src/main.py" in listing
    assert "README.md" in listing
    assert ".git" not in listing
    assert "logo.png" not in listing


def test_list_files_scopes_to_a_subdir(project: Path) -> None:
    listing = agent_tools.list_files(project, "src")

    assert "src/main.py" in listing
    assert "README.md" not in listing


def test_list_files_reports_missing_subdir(project: Path) -> None:
    assert "No such directory" in agent_tools.list_files(project, "nope")


def test_paths_outside_root_are_refused(project: Path) -> None:
    with pytest.raises(ValueError):
        agent_tools.list_files(project, "../..")


def test_grep_reports_path_line_and_text(project: Path) -> None:
    result = agent_tools.grep_files(project, r"def retrieve")

    assert "src/main.py:1: def retrieve():" in result


def test_grep_scopes_to_a_subdir(project: Path) -> None:
    result = agent_tools.grep_files(project, "Demo", "src")

    assert "No matches" in result


def test_grep_reports_when_nothing_matches(project: Path) -> None:
    assert "No matches" in agent_tools.grep_files(project, "unicorn")


def test_grep_reports_invalid_regex_instead_of_raising(project: Path) -> None:
    assert "Invalid regex" in agent_tools.grep_files(project, "(")


def test_grep_caps_the_match_count(project: Path) -> None:
    (project / "big.txt").write_text("hit\n" * 200, encoding="utf-8")

    result = agent_tools.grep_files(project, "hit")

    assert len(result.splitlines()) == agent_tools.MAX_GREP_MATCHES + 1
    assert "capped" in result
