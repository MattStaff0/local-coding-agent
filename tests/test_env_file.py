"""The .env loader: fills missing env vars, never overrides real ones."""
import os

import paths


def test_sets_missing_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("LCA_TEST_HOST", raising=False)
    env = tmp_path / ".env"
    env.write_text("LCA_TEST_HOST=http://pc:11434\n")

    paths._load_env_file(env)

    assert os.environ.pop("LCA_TEST_HOST") == "http://pc:11434"


def test_real_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("LCA_TEST_MODEL", "from-shell")
    env = tmp_path / ".env"
    env.write_text("LCA_TEST_MODEL=from-file\n")

    paths._load_env_file(env)

    assert os.environ["LCA_TEST_MODEL"] == "from-shell"


def test_ignores_comments_blanks_and_junk(tmp_path, monkeypatch):
    monkeypatch.delenv("LCA_TEST_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "not a key value line\n"
        "=no-key\n"
        "LCA_TEST_KEY=kept\n"
    )

    paths._load_env_file(env)

    assert os.environ.pop("LCA_TEST_KEY") == "kept"
    assert "not a key value line" not in os.environ


def test_strips_whitespace_and_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("LCA_TEST_QUOTED", raising=False)
    env = tmp_path / ".env"
    env.write_text('  LCA_TEST_QUOTED = "http://pc:11434"  \n')

    paths._load_env_file(env)

    assert os.environ.pop("LCA_TEST_QUOTED") == "http://pc:11434"


def test_missing_file_is_a_noop(tmp_path):
    paths._load_env_file(tmp_path / "does-not-exist.env")
