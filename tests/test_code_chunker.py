"""ast chunking: one retrievable unit per top-level definition."""
from code_chunker import chunk_python

SAMPLE = '''\
"""Module docstring."""
import os

LIMIT = 10


@decorator
def top(a: int) -> int:
    """Docstring."""
    return a + LIMIT


class Thing:
    """A class."""

    def method(self):
        return os.name
'''


def test_one_chunk_per_top_level_definition():
    chunks = chunk_python(SAMPLE, "src/sample.py")
    headings = [c["heading"] for c in chunks]

    assert headings == [
        "src/sample.py > module",
        "src/sample.py > top",
        "src/sample.py > Thing",
    ]


def test_decorators_and_docstrings_survive():
    chunks = chunk_python(SAMPLE, "src/sample.py")
    top = next(c for c in chunks if c["heading"].endswith("> top"))

    assert "@decorator" in top["text"]
    assert '"""Docstring."""' in top["text"]
    assert top["start_line"] == 7  # the decorator line


def test_module_chunk_holds_imports_and_constants():
    module = chunk_python(SAMPLE, "src/sample.py")[0]
    assert "import os" in module["text"]
    assert "LIMIT = 10" in module["text"]


def test_methods_stay_with_small_class():
    thing = chunk_python(SAMPLE, "src/sample.py")[2]
    assert "def method" in thing["text"]


def test_oversized_class_splits_per_method():
    big = "class Big:\n" + "".join(
        f"    def m{i}(self):\n        return {'x' * 200!r}\n" for i in range(30)
    )
    chunks = chunk_python(big, "big.py")
    headings = [c["heading"] for c in chunks]

    assert "big.py > Big" in headings           # header chunk
    assert "big.py > Big > m0" in headings      # per-method chunks
    assert all(len(c["text"]) < 5000 for c in chunks)


def test_syntax_error_falls_back_instead_of_raising():
    chunks = chunk_python("def broken(:\n    pass", "bad.py")
    assert chunks
    assert chunks[0]["heading"] == "bad.py > (unparsed)"


def test_async_defs_get_their_own_chunk():
    chunks = chunk_python("async def fetch():\n    return 1\n", "aio.py")
    assert [c["heading"] for c in chunks] == ["aio.py > fetch"]


def test_breadcrumb_is_first_line_of_text():
    chunks = chunk_python(SAMPLE, "src/sample.py")
    for chunk in chunks:
        assert chunk["text"].startswith(chunk["heading"])
