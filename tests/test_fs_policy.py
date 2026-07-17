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
