"""Non-executing target-project version discovery (workstream 04).

Everything reads files off disk — no interpreter in the target project is
ever executed, per the documentation spec's safety rule.
"""
from pathlib import Path

import pytest

import project_versions
from project_versions import DetectedVersion


def make_dist_info(site_packages: Path, name: str, version: str) -> None:
    info = site_packages / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )


# --- individual formats ---


def test_uv_lock_exact_pin(tmp_path):
    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "numpy"\nversion = "1.26.4"\n'
    )

    found = project_versions.detect_versions(tmp_path, ["numpy"])

    assert found["numpy"] == DetectedVersion("1.26.4", "lock")


def test_poetry_lock_exact_pin(tmp_path):
    (tmp_path / "poetry.lock").write_text(
        '[[package]]\nname = "pandas"\nversion = "2.2.1"\n'
    )

    found = project_versions.detect_versions(tmp_path, ["pandas"])

    assert found["pandas"] == DetectedVersion("2.2.1", "lock")


def test_requirements_double_equals_pin(tmp_path):
    (tmp_path / "requirements.txt").write_text(
        "# deps\npandas==2.2.1\nnumpy>=1.20\n"
    )

    found = project_versions.detect_versions(tmp_path, ["pandas"])

    assert found["pandas"] == DetectedVersion("2.2.1", "lock")


def test_pyproject_constraint_recorded_as_constraint(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["tensorflow>=2.16,<3"]\n'
    )

    found = project_versions.detect_versions(tmp_path, ["tensorflow"])

    assert found["tensorflow"] == DetectedVersion(">=2.16,<3", "constraint")


def test_dist_info_metadata_is_installed_confidence(tmp_path):
    site = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    make_dist_info(site, "numpy", "2.1.0")

    found = project_versions.detect_versions(tmp_path, ["numpy"])

    assert found["numpy"] == DetectedVersion("2.1.0", "installed")


def test_windows_venv_layout(tmp_path):
    site = tmp_path / ".venv" / "Lib" / "site-packages"
    make_dist_info(site, "pandas", "2.0.3")

    found = project_versions.detect_versions(tmp_path, ["pandas"])

    assert found["pandas"] == DetectedVersion("2.0.3", "installed")


# --- priority and misses ---


def test_lockfile_beats_pyproject_and_installed(tmp_path):
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "numpy"\nversion = "1.26.4"\n'
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["numpy>=1.20"]\n'
    )
    site = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    make_dist_info(site, "numpy", "2.1.0")

    found = project_versions.detect_versions(tmp_path, ["numpy"])

    assert found["numpy"] == DetectedVersion("1.26.4", "lock")


def test_nothing_found_is_unknown(tmp_path):
    found = project_versions.detect_versions(tmp_path, ["numpy"])

    assert found["numpy"] == DetectedVersion(None, "unknown")


def test_distribution_names_are_normalized(tmp_path):
    # scikit_learn-1.5.0.dist-info must satisfy a scikit-learn lookup.
    site = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    make_dist_info(site, "scikit_learn", "1.5.0")

    found = project_versions.detect_versions(tmp_path, ["scikit-learn"])

    assert found["scikit-learn"] == DetectedVersion("1.5.0", "installed")


def test_malformed_lockfile_degrades_to_next_source(tmp_path):
    (tmp_path / "uv.lock").write_text("{not toml at all")
    (tmp_path / "requirements.txt").write_text("numpy==1.26.4\n")

    found = project_versions.detect_versions(tmp_path, ["numpy"])

    assert found["numpy"] == DetectedVersion("1.26.4", "lock")


# --- compatibility states ---


@pytest.mark.parametrize(
    "project, docs, policy, expected",
    [
        ("2.2.1", "2.2", "major_minor", "exact"),
        ("2.2.1", "2.2.1", "major_minor", "exact"),
        ("2.2.1", "2.3", "major_minor", "mismatch"),
        ("2.2.1", "2.3", "major", "major-minor"),
        ("2.2.1", "3.0", "major", "mismatch"),
        (None, "2.2", "major_minor", "unknown"),
        ("2.2.1", None, "major_minor", "unknown"),
        ("2.2.1", "stable-at-fetch", "major_minor", "unknown"),
        (">=2.16,<3", "2.16", "major_minor", "unknown"),
    ],
)
def test_compatibility_states(project, docs, policy, expected):
    assert project_versions.compatibility(project, docs, policy) == expected
