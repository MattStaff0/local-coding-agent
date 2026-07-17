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
