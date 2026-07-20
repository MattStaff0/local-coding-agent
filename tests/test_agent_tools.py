import tempfile
from pathlib import Path

import pytest

import agent_tools


def _can_symlink() -> bool:
    try:
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "link"
            link.symlink_to(Path(directory))
        return True
    except (OSError, NotImplementedError):
        return False


requires_symlinks = pytest.mark.skipif(
    not _can_symlink(),
    reason="symlink creation unavailable (Windows: needs Developer Mode/elevation)",
)


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


def test_grep_reports_missing_subdir(project: Path) -> None:
    assert "No such directory" in agent_tools.grep_files(project, "retrieve", "nope")


def test_grep_reports_when_nothing_matches(project: Path) -> None:
    assert "No matches" in agent_tools.grep_files(project, "unicorn")


def test_grep_reports_invalid_regex_instead_of_raising(project: Path) -> None:
    assert "Invalid regex" in agent_tools.grep_files(project, "(")


def test_grep_caps_the_match_count(project: Path) -> None:
    (project / "big.txt").write_text("hit\n" * 200, encoding="utf-8")

    result = agent_tools.grep_files(project, "hit")

    assert len(result.splitlines()) == agent_tools.MAX_GREP_MATCHES + 1
    assert "capped" in result


def test_read_file_numbers_every_line(project: Path) -> None:
    result = agent_tools.read_file(project, "src/main.py")

    assert result.startswith("1: def retrieve():")
    assert "2:     return 4" in result


def test_read_file_starts_at_the_requested_line(project: Path) -> None:
    result = agent_tools.read_file(project, "src/main.py", start_line=2)

    assert result.startswith("2:")
    assert "1:" not in result


def test_read_file_reports_missing_files(project: Path) -> None:
    assert "No such file" in agent_tools.read_file(project, "nope.py")


def test_read_file_truncates_with_a_resume_hint(project: Path) -> None:
    (project / "long.md").write_text("some line here\n" * 5000, encoding="utf-8")

    result = agent_tools.read_file(project, "long.md")

    assert len(result) <= agent_tools.MAX_FILE_CHARS + 100
    assert "truncated" in result
    assert "start_line=" in result


def test_read_file_reports_start_line_past_the_end(project: Path) -> None:
    result = agent_tools.read_file(project, "src/main.py", start_line=99)

    assert "has 2 lines" in result


@requires_symlinks
def test_symlink_files_are_excluded_from_list_and_grep(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Symlinks pointing outside the sandbox root must not be listed or searched."""
    root = tmp_path_factory.mktemp("repo")
    secret_dir = tmp_path_factory.mktemp("outside")
    secret = secret_dir / "secret.md"
    secret.write_text("SECRET content here\n", encoding="utf-8")

    # A symlink INSIDE the root that points OUTSIDE it.
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def main(): pass\n", encoding="utf-8")
    (root / "leak.md").symlink_to(secret)

    listing = agent_tools.list_files(root)
    assert "leak.md" not in listing

    grep_result = agent_tools.grep_files(root, "SECRET")
    assert "No matches" in grep_result
