"""Explicit @path attachments: parse, sandbox, and render saved files.

The grammar's one load-bearing rule: an @token is an attachment only if its
path part names an existing file inside the root. Everything else —
@dataclass, @app.route, an email — stays literal prompt text. That makes
attachment syntax collision-free with ordinary Python conversation without
any escaping, and it is deterministic against the saved filesystem.

Attachments resolve BEFORE the model call and are embedded into that turn's
user message, so the model sees current saved content with no indexing step.
"""
import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path

import fs_policy

MAX_ATTACH_BYTES = 256 * 1024
NOTEBOOK_OUTPUT_CAP = 500

# Punctuation that is prose, not path: "look at @src/train.py, please".
_TRAILING_PUNCT = ",.;?!)]}\"'"

_TOKEN = re.compile(
    r"""@(?:"(?P<dquoted>[^"]+)"|'(?P<squoted>[^']+)'|(?P<bare>\S+))"""
    r"""(?(dquoted)(?::(?P<qrange>\d+(?:-\d+)?))?)"""
)
_RANGE_SUFFIX = re.compile(r"^(?P<path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$")


class AttachmentError(ValueError):
    """Any attachment problem the user must fix; always names the culprit."""


@dataclass(frozen=True)
class AttachmentSpec:
    """One parsed @token: a path and an optional 1-based inclusive range."""

    path: str
    start: int | None
    end: int | None


@dataclass(frozen=True)
class Attachment:
    """One resolved attachment ready for the model."""

    label: str
    content: str


def _split_range(token: str) -> tuple[str, int | None, int | None]:
    match = _RANGE_SUFFIX.match(token)
    if match is None:
        return token, None, None
    start = int(match["start"])
    end = int(match["end"]) if match["end"] else start
    return match["path"], start, end


def _validate_range(token: str, start: int, end: int) -> None:
    if start < 1 or end < start:
        raise AttachmentError(
            f"Invalid range in '@{token}': lines are 1-based and start must "
            "not exceed end."
        )


def _merge_specs(specs: list[AttachmentSpec]) -> list[AttachmentSpec]:
    """Union ranges per file; a whole-file mention absorbs every range."""
    by_path: dict[str, list[AttachmentSpec]] = {}
    for spec in specs:
        by_path.setdefault(spec.path, []).append(spec)

    merged: list[AttachmentSpec] = []
    for path, group in by_path.items():
        if any(spec.start is None for spec in group):
            merged.append(AttachmentSpec(path, None, None))
            continue

        ranges = sorted((spec.start, spec.end) for spec in group)
        combined = [list(ranges[0])]
        for start, end in ranges[1:]:
            if start <= combined[-1][1] + 1:
                combined[-1][1] = max(combined[-1][1], end)
            else:
                combined.append([start, end])
        merged.extend(AttachmentSpec(path, s, e) for s, e in combined)

    return merged


def parse_attachments(text: str, exists) -> tuple[str, list[AttachmentSpec]]:
    """Extract @path tokens that resolve via `exists`; leave the rest verbatim.

    Returns (cleaned question text, merged attachment specs). Raises
    AttachmentError for a resolving token with an invalid range, or when
    extraction leaves no question to answer.
    """
    specs: list[AttachmentSpec] = []
    removals: list[tuple[int, int]] = []

    for match in _TOKEN.finditer(text):
        quoted = match["dquoted"] or match["squoted"]
        if quoted is not None:
            candidate = quoted
            if match.groupdict().get("qrange"):
                start_text = match["qrange"]
                start, _, end_text = start_text.partition("-")
                start, end = int(start), int(end_text) if end_text else int(start)
            else:
                start = end = None
        else:
            token = match["bare"].rstrip(_TRAILING_PUNCT)
            if not token:
                continue
            candidate, start, end = _split_range(token)

        if not exists(candidate):
            # Not a file in this project: @decorator, @email, prose. Even a
            # range suffix doesn't rescue it — the path part decides.
            continue

        if start is not None:
            _validate_range(f"{candidate}:{start}-{end}", start, end)

        specs.append(AttachmentSpec(candidate, start, end))
        if quoted is not None:
            removal_end = match.end()
        else:
            # Trailing prose punctuation was stripped from the token, so it
            # stays in the question text: "look at @a.py, please" keeps ",".
            stripped = match["bare"].rstrip(_TRAILING_PUNCT)
            removal_end = match.start() + 1 + len(stripped)
        removals.append((match.start(), removal_end))

    if not specs:
        return text, []

    cleaned_parts: list[str] = []
    cursor = 0
    for start_index, end_index in removals:
        cleaned_parts.append(text[cursor:start_index])
        cursor = end_index
    cleaned_parts.append(text[cursor:])
    cleaned = " ".join("".join(cleaned_parts).split())

    if not cleaned:
        raise AttachmentError(
            "Attachment without a question — add what you want to know about it."
        )

    return cleaned, _merge_specs(specs)


def _reject_symlinked_parents(root: Path, relative: Path) -> None:
    """Never follow directory symlinks, even ones that stay inside root."""
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise AttachmentError(
                f"{relative.as_posix()}: refusing to follow directory "
                f"symlink '{current.name}' (root: {root})"
            )


def _resolve_target(root: Path, spec: AttachmentSpec) -> tuple[Path, str]:
    root_resolved = root.resolve()
    candidate = Path(spec.path)

    if candidate.is_absolute():
        raw = candidate
    else:
        _reject_symlinked_parents(root_resolved, candidate)
        raw = root_resolved / candidate

    # Containment first: a path outside the root must say "outside" whether
    # or not it exists — "No such file" would leak filesystem structure.
    target = raw.resolve()
    if not target.is_relative_to(root_resolved):
        raise AttachmentError(
            f"{spec.path}: outside the project root {root_resolved}"
        )

    if not target.exists():
        raise AttachmentError(
            f"No such file: {spec.path} (root: {root_resolved})"
        )
    if not target.is_file():
        raise AttachmentError(f"{spec.path}: not a file")

    relative = target.relative_to(root_resolved).as_posix()

    reason = fs_policy.denied(Path(relative))
    if reason:
        raise AttachmentError(f"{spec.path}: not attachable ({reason})")

    size = target.stat().st_size
    if size > MAX_ATTACH_BYTES:
        raise AttachmentError(
            f"{spec.path}: too large ({size // 1024} KB; cap "
            f"{MAX_ATTACH_BYTES // 1024} KB) — attach a line range instead"
        )

    return target, relative


def _read_text_strict(target: Path, display: str) -> str:
    data = target.read_bytes()
    if b"\x00" in data[:1024]:
        raise AttachmentError(f"{display}: binary or non-UTF-8 content")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise AttachmentError(f"{display}: binary or non-UTF-8 content")


def resolve_attachment(root: Path, spec: AttachmentSpec) -> Attachment:
    """Resolve one spec against the live filesystem, enforcing the sandbox."""
    target, relative = _resolve_target(root, spec)

    if target.suffix == ".ipynb":
        return _render_notebook(target, relative, spec)

    text = _read_text_strict(target, spec.path)
    lines = text.splitlines()

    if spec.start is None:
        picked = list(enumerate(lines, start=1))
        label = f"{relative} ({len(lines)} lines)"
    else:
        if spec.start > len(lines):
            raise AttachmentError(
                f"{spec.path}: range starts at line {spec.start} but the "
                f"file has only {len(lines)} lines"
            )
        end = min(spec.end, len(lines))
        picked = [
            (number, lines[number - 1]) for number in range(spec.start, end + 1)
        ]
        label = f"{relative}:{spec.start}-{end} ({len(picked)} lines)"

    content = "\n".join(f"{number}: {line}" for number, line in picked)
    return Attachment(label=label, content=content)


def _cell_output_lines(outputs: list[dict]) -> list[str]:
    lines: list[str] = []
    for output in outputs:
        text = output.get("text")
        if text is None and isinstance(output.get("data"), dict):
            data = output["data"]
            if "text/plain" in data:
                text = data["text/plain"]
            else:
                for key in data:
                    lines.append(f"   output omitted ({key})")
                continue
        if text is None:
            continue
        joined = "".join(text) if isinstance(text, list) else str(text)
        if len(joined) > NOTEBOOK_OUTPUT_CAP:
            joined = joined[:NOTEBOOK_OUTPUT_CAP] + " …(capped)"
        lines.append(f"   output: {joined}")
    return lines


def _render_notebook(target: Path, relative: str, spec: AttachmentSpec) -> Attachment:
    try:
        data = json.loads(_read_text_strict(target, spec.path))
        cells = data["cells"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise AttachmentError(
            f"{spec.path}: not a valid notebook (json error: {error})"
        )

    if spec.start is not None and spec.start > len(cells):
        raise AttachmentError(
            f"{spec.path}: range starts at cell {spec.start} but the "
            f"notebook has only {len(cells)} cells"
        )

    start = spec.start or 1
    end = min(spec.end, len(cells)) if spec.end is not None else len(cells)

    blocks: list[str] = []
    for number in range(start, end + 1):
        cell = cells[number - 1]
        cell_type = cell.get("cell_type", "code")
        source = cell.get("source", [])
        source_text = "".join(source) if isinstance(source, list) else str(source)

        if cell_type == "code":
            body = "\n".join(
                f"{i}: {line}"
                for i, line in enumerate(source_text.splitlines(), start=1)
            )
        else:
            body = source_text.rstrip("\n")

        block = f"cell-{number} [{cell_type}]:\n{body}"
        output_lines = _cell_output_lines(cell.get("outputs", []))
        if output_lines:
            block += "\n" + "\n".join(output_lines)
        blocks.append(block)

    count = end - start + 1
    if spec.start is None:
        label = f"{relative} ({count} cells)"
    elif start == end:
        label = f"{relative}:cell-{start} ({count} cells)"
    else:
        label = f"{relative}:cells-{start}-{end} ({count} cells)"

    return Attachment(label=label, content="\n\n".join(blocks))
