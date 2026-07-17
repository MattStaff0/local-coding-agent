"""Discover a target project's library versions without executing its code.

Priority order (documentation spec): lockfile pins beat pyproject constraints
beat installed metadata. Everything is plain file reading — parsing TOML and
.dist-info METADATA off disk — because running a project's interpreter just
to ask its versions is an arbitrary-code-execution hazard.

Confidence travels with every answer ("lock" / "constraint" / "installed" /
"unknown") and the UI always prints it: a confidently-wrong version is worse
than an unknown one.
"""
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectedVersion:
    version: str | None
    confidence: str  # lock | constraint | installed | unknown


def _normalize(name: str) -> str:
    """PEP 503 normalization: scikit_learn == scikit-learn == Scikit.Learn."""
    return re.sub(r"[-_.]+", "-", name.lower())


def _read_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        return {}


def _lock_versions(root: Path) -> tuple[dict[str, str], set[str]]:
    """Exact pins from uv.lock / poetry.lock / requirements*.txt.

    Returns (pins, conflicted). A distribution pinned to two different
    versions across lock-grade files (dev requirements, uv multi-platform
    resolutions) is ambiguous — file ordering must not pick the "winner",
    so it is reported as conflicted and resolved to unknown.
    """
    pins: dict[str, str] = {}
    conflicted: set[str] = set()

    def record(name: str, version: str) -> None:
        key = _normalize(name)
        if key in pins and pins[key] != version:
            conflicted.add(key)
        else:
            pins.setdefault(key, version)

    for lock_name in ("uv.lock", "poetry.lock"):
        for package in _read_toml(root / lock_name).get("package", []):
            if isinstance(package, dict) and "name" in package and "version" in package:
                record(str(package["name"]), str(package["version"]))

    for requirements in sorted(root.glob("requirements*.txt")):
        try:
            text = requirements.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            match = re.match(
                r"^\s*([A-Za-z0-9._-]+)\s*==\s*([^\s;#]+)", line
            )
            if match:
                record(match[1], match[2])

    return pins, conflicted


def _constraint_versions(root: Path) -> dict[str, str]:
    """PEP 508 specifiers from [project].dependencies in pyproject.toml."""
    constraints: dict[str, str] = {}
    dependencies = _read_toml(root / "pyproject.toml").get("project", {}).get(
        "dependencies", []
    )
    if not isinstance(dependencies, list):
        return constraints

    for requirement in dependencies:
        match = re.match(
            r"^\s*([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*(?P<spec>[<>=!~].+)$",
            str(requirement),
        )
        if match:
            constraints.setdefault(_normalize(match[1]), match["spec"].strip())

    return constraints


def _installed_versions(root: Path) -> dict[str, str]:
    """Name/Version headers read straight out of .venv dist-info METADATA."""
    installed: dict[str, str] = {}
    site_dirs = list(root.glob(".venv/lib/python*/site-packages")) + list(
        root.glob(".venv/Lib/site-packages")
    )

    for site in site_dirs:
        for info in site.glob("*.dist-info"):
            metadata = info / "METADATA"
            name = version = None
            try:
                for line in metadata.read_text(encoding="utf-8").splitlines():
                    if line.startswith("Name:"):
                        name = line.split(":", 1)[1].strip()
                    elif line.startswith("Version:"):
                        version = line.split(":", 1)[1].strip()
                    if name and version:
                        break
            except OSError:
                continue
            if name and version:
                installed.setdefault(_normalize(name), version)

    return installed


def detect_versions(
    root: Path, distributions: list[str]
) -> dict[str, DetectedVersion]:
    """Best available version per distribution, with its confidence source."""
    locks, conflicted = _lock_versions(root)
    constraints = _constraint_versions(root)
    installed = _installed_versions(root)

    found: dict[str, DetectedVersion] = {}
    for distribution in distributions:
        key = _normalize(distribution)
        if key in conflicted:
            found[distribution] = DetectedVersion(None, "unknown")
        elif key in locks:
            found[distribution] = DetectedVersion(locks[key], "lock")
        elif key in constraints:
            found[distribution] = DetectedVersion(constraints[key], "constraint")
        elif key in installed:
            found[distribution] = DetectedVersion(installed[key], "installed")
        else:
            found[distribution] = DetectedVersion(None, "unknown")

    return found


def _version_parts(version: str) -> list[int] | None:
    parts = []
    for piece in version.split("."):
        if not piece.isdigit():
            return None
        parts.append(int(piece))
    return parts or None


def compatibility(
    project_version: str | None, docs_version: str | None, policy: str
) -> str:
    """exact | major-minor | mismatch | unknown, comparing at docs precision.

    Docs versions are usually major.minor ("2.2"), so a 2.2.1 project is an
    exact match for 2.2 docs. A constraint string (">=2.16,<3") is not a
    version — it stays unknown rather than pretending precision.
    """
    if not project_version or not docs_version:
        return "unknown"

    project_parts = _version_parts(project_version)
    docs_parts = _version_parts(docs_version)
    if project_parts is None or docs_parts is None:
        return "unknown"

    if docs_parts == project_parts[: len(docs_parts)]:
        return "exact"

    if docs_parts[0] == project_parts[0] and policy == "major":
        return "major-minor"

    return "mismatch"
