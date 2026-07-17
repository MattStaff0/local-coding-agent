"""guide/ integrity: complete, current, and free of machine-specific leaks."""
from pathlib import Path

import pytest

GUIDE = Path(__file__).resolve().parent.parent / "guide"

FILES = ["setup.md", "usage.md", "example-flows.md", "troubleshooting.md"]


@pytest.mark.parametrize("name", FILES)
def test_guide_file_exists(name):
    assert (GUIDE / name).is_file()


def test_setup_covers_install_and_lan_ollama():
    text = (GUIDE / "setup.md").read_text()
    for needed in ["pip install -e .", "OLLAMA_HOST", ".env", ".venv"]:
        assert needed in text


def test_usage_covers_the_command_surface():
    text = (GUIDE / "usage.md").read_text()
    for needed in [
        "@path", "--context", "/root", "/source", "lca doctor",
        "lca docs status", "LCA_TEACHING_STYLE",
    ]:
        assert needed in text


def test_example_flows_show_escalation_and_version_mismatch():
    text = (GUIDE / "example-flows.md").read_text()
    assert "show me" in text
    assert "mismatch" in text.lower() or "these docs are for" in text


def test_troubleshooting_covers_recovery_and_rollback():
    text = (GUIDE / "troubleshooting.md").read_text()
    for needed in ["lca doctor", "--full", "git checkout", "offline"]:
        assert needed in text.lower() or needed in text


@pytest.mark.parametrize("name", FILES)
def test_no_machine_specific_leaks(name):
    text = (GUIDE / name).read_text()
    for leak in ["matt_staff", "192.168.", "MattStaff0", "/Users/"]:
        assert leak not in text, f"{name} leaks '{leak}'"


# --- review findings (codex, 2026-07-17): stronger-than-phrase guards ---


def test_setup_commands_match_pyproject_entry_points():
    import tomllib

    pyproject = GUIDE.parent / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text())["project"]["scripts"]
    setup_text = (GUIDE / "setup.md").read_text()

    # Every installed command the guide names must actually exist, and the
    # core ones must be documented.
    for command in ("lca", "lca-fetch-docs", "lca-ingest"):
        assert command in scripts
        assert command in setup_text


def test_usage_env_vars_exist_in_env_example():
    env_example = (GUIDE.parent / ".env.example").read_text()

    assert "LCA_TEACHING_STYLE" in env_example
    assert "OLLAMA_HOST" in env_example


def test_readme_has_no_machine_specific_leaks():
    readme = (GUIDE.parent / "README.md").read_text()

    for leak in ["matt_staff", "MattStaff0", "/Users/"]:
        assert leak not in readme, f"README leaks '{leak}'"


def test_guide_documents_one_shot_mutation_refusal():
    text = (GUIDE / "usage.md").read_text()

    assert "interactive-only" in text
