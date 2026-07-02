from rag import chunk_markdown

HEADED_DOC = """\
# PyTorch Basics

Intro paragraph about tensors.

## Building a Model

Models subclass nn.Module.

## Training

Loops call backward().
"""

FENCED_DOC = """\
# Shell Guide

## Comments

```bash
# this is a comment, not a heading
echo hi

# another comment after a blank line
```

Text after the fence.
"""


def test_chunks_follow_heading_sections() -> None:
    chunks = chunk_markdown(HEADED_DOC)

    headings = [c["heading"] for c in chunks]
    assert headings == [
        "PyTorch Basics",
        "PyTorch Basics > Building a Model",
        "PyTorch Basics > Training",
    ]


def test_chunk_text_starts_with_heading_breadcrumb() -> None:
    chunks = chunk_markdown(HEADED_DOC)

    model_chunk = chunks[1]
    assert model_chunk["text"].startswith("PyTorch Basics > Building a Model")
    assert "nn.Module" in model_chunk["text"]
    # Section bodies stay separated: the training text is not in this chunk.
    assert "backward()" not in model_chunk["text"]


def test_doc_without_headings_is_one_chunk() -> None:
    chunks = chunk_markdown("Just some loose notes.\n\nMore notes.")

    assert len(chunks) == 1
    assert chunks[0]["heading"] == ""
    assert chunks[0]["text"].startswith("Just some loose notes.")


def test_hash_inside_code_fence_is_not_a_heading() -> None:
    chunks = chunk_markdown(FENCED_DOC)

    # "Shell Guide" has no body of its own, so it yields no chunk — its title
    # survives in the child's breadcrumb instead.
    headings = [c["heading"] for c in chunks]
    assert headings == ["Shell Guide > Comments"]

    fence_chunk = chunks[0]["text"]
    # The whole fence, including its blank line, stays in one chunk.
    assert "# this is a comment, not a heading" in fence_chunk
    assert "# another comment after a blank line" in fence_chunk
    assert "Text after the fence." in fence_chunk


def test_yaml_frontmatter_is_not_indexed_as_a_chunk() -> None:
    doc = "---\nurl: https://example.com/a.html\nfetched: 2026-07-01\n---\n\n# Tensors\n\nBody text."

    chunks = chunk_markdown(doc)

    assert [c["heading"] for c in chunks] == ["Tensors"]
    assert "fetched" not in chunks[0]["text"]


def test_stacked_frontmatter_blocks_are_all_skipped() -> None:
    # A fetched raw-markdown doc can carry its own frontmatter (e.g. myst)
    # right after the scraper's provenance block.
    doc = (
        "---\nurl: https://example.com/a.md\nfetched: 2026-07-02\n---\n\n"
        "---\nmyst:\n  number_code_blocks: [\"python3\"]\n---\n\n"
        "# Function Calling\n\nBody text."
    )

    chunks = chunk_markdown(doc)

    assert [c["heading"] for c in chunks] == ["Function Calling"]
    assert "myst" not in chunks[0]["text"]


def test_oversized_section_splits_on_paragraph_boundaries() -> None:
    paragraphs = [f"Paragraph {i} " + ("word " * 40).strip() + "." for i in range(12)]
    doc = "# Big Section\n\n" + "\n\n".join(paragraphs)

    chunks = chunk_markdown(doc, chunk_size=600)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["heading"] == "Big Section"
        assert chunk["text"].startswith("Big Section")
        assert len(chunk["text"]) <= 600
        # Paragraphs that fit the budget are kept whole: each chunk ends at a
        # paragraph end. (A single paragraph bigger than the budget falls back
        # to character windows — covered separately below.)
        assert chunk["text"].rstrip().endswith(".")

    # Neighboring chunks overlap by one paragraph so context is not lost.
    first_tail = chunks[0]["text"].strip().split("\n\n")[-1]
    assert first_tail in chunks[1]["text"]


def test_code_fence_bigger_than_chunk_size_is_kept_whole() -> None:
    fence = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(60)) + "\n```"
    doc = f"# Big Example\n\nIntro.\n\n{fence}\n\nOutro."

    chunks = chunk_markdown(doc, chunk_size=300)

    fence_chunks = [c for c in chunks if "```python" in c["text"]]
    assert len(fence_chunks) == 1
    # Both delimiters and every line live in one (oversized) chunk.
    assert fence_chunks[0]["text"].count("```") == 2
    assert "line_0 = 0" in fence_chunks[0]["text"]
    assert "line_59 = 59" in fence_chunks[0]["text"]


def test_single_paragraph_bigger_than_budget_splits_into_character_windows() -> None:
    doc = "# Wall\n\n" + "x" * 2000

    chunks = chunk_markdown(doc, chunk_size=300)

    assert len(chunks) > 1
    assert all(len(c["text"]) <= 300 for c in chunks)
    assert all(c["heading"] == "Wall" for c in chunks)
    # No content is lost across the window splits.
    recombined = "".join(c["text"].removeprefix("Wall\n\n") for c in chunks)
    assert "x" * 100 in recombined
