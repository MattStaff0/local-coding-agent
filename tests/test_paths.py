"""Paths must anchor to the project, not to whatever directory lca runs from."""
import os
import subprocess
import sys
import tomllib
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def test_paths_anchor_to_project_root(tmp_path):
    # Run from an unrelated cwd: paths must still point at the repo.
    code = (
        "import paths; "
        "print(paths.DOCS_DIR); print(paths.DB_DIR); print(paths.HISTORY_FILE)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        capture_output=True, text=True, check=True,
    )
    docs, db, history = result.stdout.strip().splitlines()

    project_root = SRC.parent
    assert docs == str(project_root / "docs")
    assert db == str(project_root / "chroma_db")
    assert history == str(project_root / "chat_history.json")


def test_lca_home_overrides_root(tmp_path):
    code = "import paths; print(paths.PROJECT_ROOT)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(SRC), "LCA_HOME": str(tmp_path)},
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == str(tmp_path)


def test_entry_point_declared():
    with open(SRC.parent / "pyproject.toml", "rb") as f:
        config = tomllib.load(f)

    assert config["project"]["scripts"]["lca"] == "ask:main"
    assert any(d.startswith("rich") for d in config["project"]["dependencies"])
    assert any(d.startswith("prompt_toolkit") for d in config["project"]["dependencies"])
