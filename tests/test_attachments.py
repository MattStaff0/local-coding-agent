"""@path attachment grammar and resolution (workstream 03).

Parsing is pure: it takes an `exists` predicate instead of touching disk, so
every filesystem shape is injectable. Resolution tests use real tmp_path
roots because sandboxing IS filesystem behavior.
"""
import json
from pathlib import Path

import pytest

import attachments
from attachments import AttachmentError, AttachmentSpec


# --- parsing ---


def exists_none(path: str) -> bool:
    return False


def make_exists(*known: str):
    return lambda path: path in known


class TestParse:
    def test_whole_file_token_extracted_and_removed(self):
        clean, specs = attachments.parse_attachments(
            "why does @src/train.py crash?", make_exists("src/train.py")
        )

        assert specs == [AttachmentSpec("src/train.py", None, None)]
        assert clean == "why does crash?"

    def test_non_resolving_token_stays_literal(self):
        clean, specs = attachments.parse_attachments(
            "what does @dataclass do in python?", exists_none
        )

        assert specs == []
        assert clean == "what does @dataclass do in python?"

    def test_decorator_with_range_lookalike_stays_literal(self):
        clean, specs = attachments.parse_attachments(
            "explain @app.route:80 here", exists_none
        )

        assert specs == []
        assert "@app.route:80" in clean

    def test_single_line_and_range_suffixes(self):
        _, specs = attachments.parse_attachments(
            "@a.py:80 and @b.py:80-130", make_exists("a.py", "b.py")
        )

        assert specs == [
            AttachmentSpec("a.py", 80, 80),
            AttachmentSpec("b.py", 80, 130),
        ]

    def test_quoted_path_with_spaces(self):
        _, specs = attachments.parse_attachments(
            'check @"my dir/my file.py":3-9 please', make_exists("my dir/my file.py")
        )

        assert specs == [AttachmentSpec("my dir/my file.py", 3, 9)]

    def test_trailing_punctuation_stripped_before_resolution(self):
        clean, specs = attachments.parse_attachments(
            "look at @src/train.py, please", make_exists("src/train.py")
        )

        assert specs == [AttachmentSpec("src/train.py", None, None)]
        assert clean == "look at , please"

    def test_same_file_twice_ranges_unioned(self):
        _, specs = attachments.parse_attachments(
            "@a.py:10-20 versus @a.py:15-30", make_exists("a.py")
        )

        assert specs == [AttachmentSpec("a.py", 10, 30)]

    def test_whole_file_wins_over_range_of_same_file(self):
        _, specs = attachments.parse_attachments(
            "@a.py and @a.py:5", make_exists("a.py")
        )

        assert specs == [AttachmentSpec("a.py", None, None)]

    def test_disjoint_ranges_of_same_file_kept_separate(self):
        _, specs = attachments.parse_attachments(
            "@a.py:1-3 and @a.py:100-110", make_exists("a.py")
        )

        assert specs == [
            AttachmentSpec("a.py", 1, 3),
            AttachmentSpec("a.py", 100, 110),
        ]

    def test_attachment_only_prompt_is_an_error(self):
        with pytest.raises(AttachmentError, match="without a question"):
            attachments.parse_attachments("@src/train.py", make_exists("src/train.py"))

    def test_invalid_range_on_resolving_file_is_an_error(self):
        with pytest.raises(AttachmentError, match="9-3"):
            attachments.parse_attachments("@a.py:9-3 why?", make_exists("a.py"))

        with pytest.raises(AttachmentError, match=":0"):
            attachments.parse_attachments("@a.py:0 why?", make_exists("a.py"))


# --- resolution ---


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text(
        "\n".join(f"line {n}" for n in range(1, 21)) + "\n"
    )
    return tmp_path


class TestResolve:
    def test_whole_file_rendered_with_real_line_numbers(self, root):
        attachment = attachments.resolve_attachment(
            root, AttachmentSpec("src/train.py", None, None)
        )

        assert "1: line 1" in attachment.content
        assert "20: line 20" in attachment.content
        assert attachment.label == "src/train.py (20 lines)"

    def test_range_slices_inclusive_with_stable_numbering(self, root):
        attachment = attachments.resolve_attachment(
            root, AttachmentSpec("src/train.py", 5, 7)
        )

        assert attachment.content.splitlines() == [
            "5: line 5", "6: line 6", "7: line 7",
        ]
        assert attachment.label == "src/train.py:5-7 (3 lines)"

    def test_end_past_eof_clamps_start_past_eof_errors(self, root):
        clamped = attachments.resolve_attachment(
            root, AttachmentSpec("src/train.py", 18, 9999)
        )
        assert clamped.content.splitlines()[-1] == "20: line 20"

        with pytest.raises(AttachmentError, match="20 lines"):
            attachments.resolve_attachment(
                root, AttachmentSpec("src/train.py", 21, 30)
            )

    def test_missing_file_error_names_path_and_root(self, root):
        with pytest.raises(AttachmentError) as excinfo:
            attachments.resolve_attachment(root, AttachmentSpec("nope.py", None, None))

        assert "nope.py" in str(excinfo.value)
        assert str(root) in str(excinfo.value)

    def test_absolute_path_inside_root_allowed(self, root):
        attachment = attachments.resolve_attachment(
            root, AttachmentSpec(str(root / "src" / "train.py"), 1, 1)
        )

        assert attachment.label.startswith("src/train.py")

    def test_absolute_path_outside_root_rejected(self, root, tmp_path_factory):
        outside = tmp_path_factory.mktemp("elsewhere") / "x.py"
        outside.write_text("x = 1\n")

        with pytest.raises(AttachmentError, match="outside"):
            attachments.resolve_attachment(
                root, AttachmentSpec(str(outside), None, None)
            )

    def test_traversal_rejected(self, root):
        with pytest.raises(AttachmentError, match="outside"):
            attachments.resolve_attachment(
                root, AttachmentSpec("../secrets.txt", None, None)
            )

    def test_file_symlink_escaping_root_rejected(self, root, tmp_path_factory):
        outside = tmp_path_factory.mktemp("elsewhere") / "real.py"
        outside.write_text("x = 1\n")
        (root / "link.py").symlink_to(outside)

        with pytest.raises(AttachmentError, match="outside"):
            attachments.resolve_attachment(root, AttachmentSpec("link.py", None, None))

    def test_file_symlink_inside_root_allowed(self, root):
        (root / "alias.py").symlink_to(root / "src" / "train.py")

        attachment = attachments.resolve_attachment(
            root, AttachmentSpec("alias.py", 1, 1)
        )

        assert "1: line 1" in attachment.content

    def test_directory_symlink_parent_rejected_even_when_target_in_root(self, root):
        (root / "shortcut").symlink_to(root / "src")

        with pytest.raises(AttachmentError, match="symlink"):
            attachments.resolve_attachment(
                root, AttachmentSpec("shortcut/train.py", None, None)
            )

    def test_denied_file_rejected_even_explicitly(self, root):
        (root / ".env").write_text("KEY=value\n")

        with pytest.raises(AttachmentError, match="deny"):
            attachments.resolve_attachment(root, AttachmentSpec(".env", None, None))

    def test_gitignored_file_allowed_when_explicit(self, root):
        (root / ".gitignore").write_text("generated.py\n")
        (root / "generated.py").write_text("x = 1\n")

        attachment = attachments.resolve_attachment(
            root, AttachmentSpec("generated.py", None, None)
        )

        assert "x = 1" in attachment.content

    def test_oversized_file_rejected_with_size(self, root):
        big = root / "big.py"
        big.write_text("x" * (attachments.MAX_ATTACH_BYTES + 1))

        with pytest.raises(AttachmentError, match="KB"):
            attachments.resolve_attachment(root, AttachmentSpec("big.py", None, None))

    def test_binary_content_rejected(self, root):
        (root / "blob.py").write_bytes(b"\x00\x01\x02 not text")

        with pytest.raises(AttachmentError, match="binary or non-UTF-8"):
            attachments.resolve_attachment(root, AttachmentSpec("blob.py", None, None))


# --- notebooks ---


def notebook_json(cells: list[dict]) -> str:
    return json.dumps({"nbformat": 4, "cells": cells})


class TestNotebook:
    def make_notebook(self, root: Path, cells: list[dict]) -> None:
        (root / "lesson.ipynb").write_text(notebook_json(cells))

    def test_cells_numbered_with_types_and_code_line_numbers(self, tmp_path):
        self.make_notebook(
            tmp_path,
            [
                {"cell_type": "markdown", "source": ["# Lesson 3\n"], "outputs": []},
                {
                    "cell_type": "code",
                    "source": ["import numpy as np\n", "x = np.arange(10)\n"],
                    "outputs": [],
                },
            ],
        )

        attachment = attachments.resolve_attachment(
            tmp_path, AttachmentSpec("lesson.ipynb", None, None)
        )

        assert "cell-1 [markdown]" in attachment.content
        assert "cell-2 [code]" in attachment.content
        assert "1: import numpy as np" in attachment.content
        assert "2: x = np.arange(10)" in attachment.content

    def test_text_output_capped_and_binary_output_omitted(self, tmp_path):
        self.make_notebook(
            tmp_path,
            [
                {
                    "cell_type": "code",
                    "source": ["print('x')\n"],
                    "outputs": [
                        {"output_type": "stream", "text": ["y" * 2000]},
                        {
                            "output_type": "display_data",
                            "data": {"image/png": "aWJi..."},
                        },
                    ],
                },
            ],
        )

        attachment = attachments.resolve_attachment(
            tmp_path, AttachmentSpec("lesson.ipynb", None, None)
        )

        assert "y" * 501 not in attachment.content
        assert "output omitted (image/png)" in attachment.content

    def test_range_selects_cells_not_lines(self, tmp_path):
        self.make_notebook(
            tmp_path,
            [
                {"cell_type": "markdown", "source": ["first\n"], "outputs": []},
                {"cell_type": "markdown", "source": ["second\n"], "outputs": []},
                {"cell_type": "markdown", "source": ["third\n"], "outputs": []},
            ],
        )

        attachment = attachments.resolve_attachment(
            tmp_path, AttachmentSpec("lesson.ipynb", 2, 2)
        )

        assert "second" in attachment.content
        assert "first" not in attachment.content
        assert attachment.label == "lesson.ipynb:cell-2 (1 cells)"

    def test_invalid_notebook_json_is_a_clear_error(self, tmp_path):
        (tmp_path / "lesson.ipynb").write_text("{not json")

        with pytest.raises(AttachmentError, match="not a valid notebook"):
            attachments.resolve_attachment(
                tmp_path, AttachmentSpec("lesson.ipynb", None, None)
            )
