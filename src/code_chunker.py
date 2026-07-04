"""ast-based chunking for Python source files.

One chunk per top-level function/class keeps each retrievable unit a whole,
meaningful definition (decorators and docstrings included) instead of an
arbitrary character window. Module-level statements form one leading chunk.
"""
import ast

from rag import chunk_text

# A class whose source exceeds this splits into per-method chunks; beyond it
# a single chunk stops fitting the prompt budget alongside other context.
MAX_CLASS_CHARS = 4000

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _segment_start(node) -> int:
    """First line of a definition, counting its decorators."""
    candidates = [d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]
    return min(candidates)


def _segment(lines: list[str], start_line: int, end_line: int) -> str:
    return "\n".join(lines[start_line - 1 : end_line])


def _chunk(heading: str, body: str, start_line: int) -> dict:
    return {
        "heading": heading,
        "text": f"{heading}\n\n{body}",
        "start_line": start_line,
    }


def _split_class(node: ast.ClassDef, lines: list[str], prefix: str) -> list[dict]:
    """Per-method chunks for a class too big to retrieve whole."""
    heading = f"{prefix} > {node.name}"
    chunks = []
    header_lines: list[str] = []
    header_start = _segment_start(node)

    for child in node.body:
        if isinstance(child, _DEFS):
            start = _segment_start(child)
            chunks.append(
                _chunk(
                    f"{heading} > {child.name}",
                    _segment(lines, start, child.end_lineno),
                    start,
                )
            )
        else:
            # Class line + docstring + class-level assignments stay together
            # as a header chunk so the class itself remains retrievable.
            header_lines.append(_segment(lines, child.lineno, child.end_lineno))

    header_body = "\n".join([lines[node.lineno - 1]] + header_lines)
    return [_chunk(heading, header_body, header_start)] + chunks


def chunk_python(text: str, relative_path: str) -> list[dict]:
    """Chunk Python source into one dict per top-level definition.

    Never raises: unparsable files fall back to plain character chunks so a
    repo with one broken file still indexes.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [
            {
                "heading": f"{relative_path} > (unparsed)",
                "text": piece,
                "start_line": 1,
            }
            for piece in chunk_text(text, chunk_size=1500)
        ]

    lines = text.splitlines()
    module_parts: list[str] = []
    module_start: int | None = None
    chunks: list[dict] = []

    for node in tree.body:
        if isinstance(node, _DEFS):
            start = _segment_start(node)
            segment = _segment(lines, start, node.end_lineno)

            if isinstance(node, ast.ClassDef) and len(segment) > MAX_CLASS_CHARS:
                chunks.extend(_split_class(node, lines, relative_path))
            else:
                chunks.append(
                    _chunk(f"{relative_path} > {node.name}", segment, start)
                )
        else:
            if module_start is None:
                module_start = node.lineno
            module_parts.append(_segment(lines, node.lineno, node.end_lineno))

    if module_parts:
        module_chunk = _chunk(
            f"{relative_path} > module", "\n".join(module_parts), module_start
        )
        chunks.insert(0, module_chunk)

    return chunks
