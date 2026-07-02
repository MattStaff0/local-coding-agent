import re
from pathlib import Path

# Output caps protect the small model's context window: a tool result that
# does not fit in context is worse than no result at all.
MAX_GREP_MATCHES = 40
MAX_FILE_CHARS = 8_000

SKIP_DIRS = {".git", "__pycache__", "chroma_db", ".venv", "node_modules"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini"}


def _resolve_inside(root: Path, relative: str) -> Path:
    """Resolve a relative path, refusing anything that escapes the root."""
    target = (root / relative).resolve()

    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path '{relative}' is outside the project root.")

    return target


def _iter_text_files(base: Path):
    """Yield readable text files under base, skipping caches and binaries."""
    for path in sorted(base.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            yield path


def list_files(root: Path, subdir: str = ".") -> str:
    """List the project's text files, relative to the root."""
    base = _resolve_inside(root, subdir)

    if not base.is_dir():
        return f"No such directory: {subdir}"

    lines = [
        path.relative_to(root).as_posix() for path in _iter_text_files(base)
    ]

    if not lines:
        return f"No text files under {subdir}."

    return "\n".join(lines)


def grep_files(root: Path, pattern: str, subdir: str = ".") -> str:
    """Search file contents with a regex; returns 'path:line: text' matches."""
    try:
        compiled = re.compile(pattern)
    except re.error as error:
        return f"Invalid regex '{pattern}': {error}"

    base = _resolve_inside(root, subdir)
    matches: list[str] = []

    for path in _iter_text_files(base):
        text = path.read_text(encoding="utf-8", errors="replace")

        for line_number, line in enumerate(text.splitlines(), start=1):
            if not compiled.search(line):
                continue

            relative = path.relative_to(root).as_posix()
            matches.append(f"{relative}:{line_number}: {line.strip()}")

            if len(matches) >= MAX_GREP_MATCHES:
                matches.append(
                    f"... capped at {MAX_GREP_MATCHES} matches; use a more specific pattern."
                )
                return "\n".join(matches)

    if not matches:
        return f"No matches for '{pattern}'."

    return "\n".join(matches)


def read_file(root: Path, path: str, start_line: int = 1) -> str:
    """Read one file with numbered lines, truncating with a resume hint."""
    target = _resolve_inside(root, path)

    if not target.is_file():
        return f"No such file: {path}"

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    picked: list[str] = []
    used = 0

    for number, line in enumerate(lines[start_line - 1 :], start=start_line):
        numbered = f"{number}: {line}"

        if used + len(numbered) > MAX_FILE_CHARS:
            picked.append(
                f"... truncated; call read_file again with start_line={number} for the rest."
            )
            break

        picked.append(numbered)
        used += len(numbered) + 1

    if not picked:
        return f"{path} has no line {start_line}; the file has {len(lines)} lines."

    return "\n".join(picked)
