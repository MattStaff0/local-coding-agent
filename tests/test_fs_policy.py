"""Shared ignore/deny policy: one module, two different teeth.

denied()  — security boundary: applies even to explicit @path attachments.
ignored() — search hygiene: filters live list/grep, explicit attach overrides.
"""
from pathlib import Path

import pytest

import fs_policy


# --- denied(): the absolute list ---


@pytest.mark.parametrize(
    "relative",
    [
        ".env",
        ".env.local",
        "config/secrets.yaml",
        "certs/key.pem",
        ".ssh/id_rsa",
        ".ssh/id_ed25519.pub",
        "aws_credentials",
        "models/model.safetensors",
        "models/weights.pt",
        "cache/db.sqlite",
        "data/data.parquet",
        "img/photo.png",
        "backup/archive.zip",
        ".git/config",
    ],
)
def test_denied_rejects_secrets_and_binary_artifacts(relative):
    assert fs_policy.denied(Path(relative)) is not None


@pytest.mark.parametrize(
    "relative",
    ["src/train.py", "README.md", "notebooks/lesson.ipynb", "pyproject.toml"],
)
def test_denied_allows_ordinary_project_files(relative):
    assert fs_policy.denied(Path(relative)) is None


def test_denied_reason_names_the_rule():
    reason = fs_policy.denied(Path(".env"))
    assert ".env" in reason


def test_is_reparse_or_symlink_is_false_for_regular_paths(tmp_path):
    regular_file = tmp_path / "regular.py"
    regular_file.write_text("x = 1\n")
    regular_directory = tmp_path / "regular-dir"
    regular_directory.mkdir()

    assert not fs_policy.is_reparse_or_symlink(regular_file)
    assert not fs_policy.is_reparse_or_symlink(regular_directory)


# --- ignored(): .gitignore + built-ins ---


def test_ignored_honors_root_gitignore_patterns(tmp_path):
    (tmp_path / ".gitignore").write_text(
        "# build junk\n*.log\nbuild/\nscratch.py\n"
    )

    assert fs_policy.ignored(tmp_path, Path("debug.log"))
    assert fs_policy.ignored(tmp_path, Path("build/gen.py"))
    assert fs_policy.ignored(tmp_path, Path("scratch.py"))
    assert not fs_policy.ignored(tmp_path, Path("src/train.py"))


def test_ignored_negation_lines_are_skipped_not_crashed(tmp_path):
    # '!' negation is documented-unsupported: the line is ignored entirely.
    (tmp_path / ".gitignore").write_text("*.log\n!keep.log\n")

    assert fs_policy.ignored(tmp_path, Path("keep.log"))


def test_ignored_without_gitignore_only_builtins_apply(tmp_path):
    assert not fs_policy.ignored(tmp_path, Path("anything.py"))
    assert fs_policy.ignored(tmp_path, Path(".venv/lib/x.py"))
    assert fs_policy.ignored(tmp_path, Path("chroma_db/data.bin"))
    assert fs_policy.ignored(tmp_path, Path("src/__pycache__/x.pyc"))


# --- live tools consult the policy ---


def test_grep_no_longer_reads_env_files(tmp_path):
    import agent_tools

    (tmp_path / ".env").write_text("OLLAMA_HOST=http://example:11434\n")
    (tmp_path / "app.py").write_text("host = 'OLLAMA_HOST'\n")

    result = agent_tools.grep_files(tmp_path, "OLLAMA_HOST")

    assert ".env" not in result
    assert "app.py" in result


def test_list_files_skips_gitignored_files(tmp_path):
    import agent_tools

    (tmp_path / ".gitignore").write_text("generated.py\n")
    (tmp_path / "generated.py").write_text("x = 1\n")
    (tmp_path / "real.py").write_text("y = 2\n")

    result = agent_tools.list_files(tmp_path)

    assert "generated.py" not in result
    assert "real.py" in result


# --- review findings (codex, 2026-07-16): nested/anchored dir patterns ---


def test_ignored_nested_directory_pattern(tmp_path):
    (tmp_path / ".gitignore").write_text("generated/cache/\n")

    assert fs_policy.ignored(tmp_path, Path("generated/cache/x.py"))
    assert not fs_policy.ignored(tmp_path, Path("generated/other/x.py"))


def test_ignored_root_anchored_directory_pattern(tmp_path):
    (tmp_path / ".gitignore").write_text("/build/\n")

    assert fs_policy.ignored(tmp_path, Path("build/gen.py"))


def test_read_file_refuses_denied_files(tmp_path):
    import agent_tools

    (tmp_path / ".env").write_text("KEY=supersecret\n")

    result = agent_tools.read_file(tmp_path, ".env")

    assert "supersecret" not in result
    assert "deny" in result


def test_edit_preview_refuses_denied_files(tmp_path):
    import agent_tools

    (tmp_path / "key.pem").write_text("PRIVATE KEY MATERIAL\n")

    preview = agent_tools.preview_edit(tmp_path, "key.pem", "PRIVATE", "PUBLIC")

    assert "error" in preview
    assert "PRIVATE KEY MATERIAL" not in str(preview.get("diff", ""))
