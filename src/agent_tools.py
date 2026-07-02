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
