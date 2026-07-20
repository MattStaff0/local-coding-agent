import difflib
import os
import re
import shlex
import subprocess
from pathlib import Path

import fs_policy

# Output caps protect the small model's context window: a tool result that
# does not fit in context is worse than no result at all.
MAX_GREP_MATCHES = 40
MAX_FILE_CHARS = 8_000

# Kept as an alias: fs_policy is the single source of truth for skip rules.
SKIP_DIRS = fs_policy.SKIP_DIRS
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini"}


def _resolve_inside(root: Path, relative: str) -> Path:
    """Resolve a relative path, refusing anything that escapes the root."""
    target = (root / relative).resolve()

    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path '{relative}' is outside the project root.")

    return target


def _deny_reason(root: Path, target: Path) -> str | None:
    """The fs_policy deny reason for a resolved in-root path, if any.

    Applied to reads AND writes: the deny list is how secrets stay out of
    the model's context, and writing .env through the agent is equally out.
    """
    return fs_policy.denied(target.relative_to(root.resolve()))


def _iter_text_files(root: Path, base: Path):
    """Yield searchable text files under base, per the shared fs policy."""
    for path in sorted(base.rglob("*")):
        ancestors = (path, *path.parents)
        if any(
            fs_policy.is_reparse_or_symlink(ancestor)
            for ancestor in ancestors
            if ancestor.is_relative_to(base)
        ):
            continue

        if not (path.is_file() and path.suffix in TEXT_SUFFIXES):
            continue

        relative = path.relative_to(root)
        if fs_policy.ignored(root, relative) or fs_policy.denied(relative):
            continue

        yield path


def list_files(root: Path, subdir: str = ".") -> str:
    """List the project's text files, relative to the root."""
    base = _resolve_inside(root, subdir)

    if not base.is_dir():
        return f"No such directory: {subdir}"

    lines = [
        path.relative_to(root).as_posix() for path in _iter_text_files(root, base)
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

    if not base.is_dir():
        return f"No such directory: {subdir}"

    matches: list[str] = []

    for path in _iter_text_files(root, base):
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
    # A 3B model happily sends start_line=0; slicing with -1 would silently
    # return the wrong lines, which is worse than correcting the argument.
    start_line = max(1, start_line)
    target = _resolve_inside(root, path)

    if not target.is_file():
        return f"No such file: {path}"

    reason = _deny_reason(root, target)
    if reason:
        return f"{path} is not readable ({reason})."

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


def _unified_diff(path: str, old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}",
        )
    )


def preview_edit(root: Path, path: str, old_text: str, new_text: str) -> dict:
    """Exact-match replacement preview; ambiguity is an error, not a guess."""
    target = _resolve_inside(root, path)

    if not target.is_file():
        return {"error": f"No such file: {path}"}

    reason = _deny_reason(root, target)
    if reason:
        return {"error": f"{path} is not editable ({reason})."}

    content = target.read_text(encoding="utf-8")
    count = content.count(old_text)

    if count == 0:
        return {"error": f"old_text not found in {path}."}
    if count > 1:
        return {
            "error": (
                f"old_text appears {count} times in {path}; "
                "include more context to make it unique."
            )
        }

    new_content = content.replace(old_text, new_text, 1)
    return {"diff": _unified_diff(path, content, new_content), "new_content": new_content}


def preview_write(root: Path, path: str, content: str) -> dict:
    target = _resolve_inside(root, path)
    reason = _deny_reason(root, target)
    if reason:
        return {"error": f"{path} is not writable ({reason})."}
    old = target.read_text(encoding="utf-8") if target.is_file() else ""
    return {"diff": _unified_diff(path, old, content), "new_content": content}


def apply_content(root: Path, path: str, new_content: str) -> str:
    target = _resolve_inside(root, path)
    reason = _deny_reason(root, target)
    if reason:
        raise ValueError(f"{path} is not writable ({reason}).")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(new_content, encoding="utf-8")
    return f"Wrote {path}"


# Deliberately tiny: the agent can run tests and scripts, nothing else.
# Growing this set is a review decision, not a convenience edit.
ALLOWED_COMMANDS = {"pytest", "python"}


def run_command(root: Path, command: str, timeout: int = 120) -> str:
    """Run one allowlisted command inside root; every failure is a string."""
    try:
        argv = shlex.split(command, posix=(os.name != "nt"))
    except ValueError as error:
        return f"Could not parse command: {error}"

    if not argv or Path(argv[0]).name not in ALLOWED_COMMANDS:
        allowed = ", ".join(sorted(ALLOWED_COMMANDS))
        return f"Command not allowed: '{command}'. Allowed: {allowed}."

    try:
        completed = subprocess.run(
            argv, cwd=root, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        # The partial output usually explains WHY it hung (e.g. which test).
        partial = (error.stdout or "") if isinstance(error.stdout, str) else ""
        tail = f"\n{partial[-1000:]}" if partial else ""
        return f"Command timed out after {timeout}s: {command}{tail}"
    except OSError as error:
        return f"Command failed to start: {error}"

    output = f"exit code {completed.returncode}\n{completed.stdout}\n{completed.stderr}"

    if len(output) > MAX_FILE_CHARS:
        output = output[:MAX_FILE_CHARS] + "\n... output truncated"

    return output
