from pathlib import Path

import pytest

import fetch_docs
from fetch_docs import (
    fetch_page,
    fetch_source,
    html_to_markdown,
    load_sources,
    slug_for_url,
    write_doc,
)


def test_load_sources_reads_source_names_and_urls(tmp_path: Path) -> None:
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        "pytorch:\n"
        "  - https://example.com/pytorch/basics.html\n"
        "python:\n"
        "  - https://example.com/python/datastructures.html\n"
        "  - https://example.com/python/functions.html\n",
        encoding="utf-8",
    )

    sources = load_sources(registry)

    assert sources == {
        "pytorch": ["https://example.com/pytorch/basics.html"],
        "python": [
            "https://example.com/python/datastructures.html",
            "https://example.com/python/functions.html",
        ],
    }


def test_load_sources_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sources(tmp_path / "missing.yaml")


def test_slug_for_url_uses_last_path_segment() -> None:
    url = "https://docs.python.org/3/tutorial/datastructures.html"
    assert slug_for_url(url) == "datastructures"


def test_slug_for_url_handles_trailing_slash_and_odd_characters() -> None:
    url = "https://example.com/guide/My%20Page/"
    assert slug_for_url(url) == "my-20page"


def test_slug_for_url_falls_back_to_index_for_bare_domain() -> None:
    assert slug_for_url("https://example.com/") == "index"


def test_html_to_markdown_converts_main_content() -> None:
    html = """
    <html><body>
      <nav><a href="/">Skip this nav</a></nav>
      <main>
        <h1>PyTorch Basics</h1>
        <p>Tensors are arrays.</p>
        <pre><code>import torch</code></pre>
      </main>
      <script>console.log("junk")</script>
      <footer>Skip this footer</footer>
    </body></html>
    """

    markdown = html_to_markdown(html)

    assert "# PyTorch Basics" in markdown
    assert "Tensors are arrays." in markdown
    assert "import torch" in markdown
    assert "Skip this nav" not in markdown
    assert "Skip this footer" not in markdown
    assert "console.log" not in markdown


def test_html_to_markdown_uses_body_when_no_main_element() -> None:
    html = "<html><body><h2>Lists</h2><p>Lists hold items.</p></body></html>"

    markdown = html_to_markdown(html)

    assert "## Lists" in markdown
    assert "Lists hold items." in markdown


def test_html_to_markdown_prefers_article_over_surrounding_main_chrome() -> None:
    html = """
    <main>
      <div>Rate this Page ★ ★ ★</div>
      <article><h1>Build Model</h1><p>Layers stack up.</p></article>
    </main>
    """

    markdown = html_to_markdown(html)

    assert "# Build Model" in markdown
    assert "Rate this Page" not in markdown


def test_html_to_markdown_strips_navigation_roles_and_headerlinks() -> None:
    html = """
    <body>
      <div role="navigation"><a href="/">index</a> | <a href="/mod">modules</a></div>
      <div role="main">
        <h1>Data Structures<a class="headerlink" href="#ds">¶</a></h1>
        <p>Lists hold items.</p>
      </div>
    </body>
    """

    markdown = html_to_markdown(html)

    assert "# Data Structures" in markdown
    assert "¶" not in markdown
    assert "modules" not in markdown


def test_html_to_markdown_drops_images_and_collapses_blank_lines() -> None:
    html = """
    <main>
      <p><img src="logo.svg" alt="logo"></p>
      <h1>Tensors</h1>
      <p><img src="colab.svg"></p>
      <p>Tensors are arrays.</p>
    </main>
    """

    markdown = html_to_markdown(html)

    assert "logo" not in markdown
    assert ".svg" not in markdown
    assert "\n\n\n" not in markdown
    assert "# Tensors" in markdown


def test_html_to_markdown_strips_invisible_anchor_links_in_headings() -> None:
    # Mintlify-style docs put an anchor link holding only a zero-width space
    # inside every heading; it must not leak into the markdown heading text.
    html = (
        '<main><h2><a href="#calling-a-single-tool">​</a>'
        "Calling a single tool</h2><p>Body.</p></main>"
    )

    markdown = html_to_markdown(html)

    assert "## Calling a single tool" in markdown
    assert "#calling-a-single-tool" not in markdown
    assert "​" not in markdown


def test_fetch_page_defaults_to_utf8_when_no_charset_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeResponse("placeholder")
    response.headers = {"content-type": "text/html"}
    response.encoding = "ISO-8859-1"

    monkeypatch.setattr(fetch_docs.requests, "get", lambda url, **kwargs: response)

    fetch_page("https://example.com/page.html")

    assert response.encoding == "utf-8"


def test_fetch_page_keeps_declared_charset(monkeypatch: pytest.MonkeyPatch) -> None:
    response = FakeResponse("placeholder")
    response.headers = {"content-type": "text/html; charset=latin-1"}
    response.encoding = "latin-1"

    monkeypatch.setattr(fetch_docs.requests, "get", lambda url, **kwargs: response)

    fetch_page("https://example.com/page.html")

    assert response.encoding == "latin-1"


def test_write_doc_creates_source_directory_with_frontmatter(tmp_path: Path) -> None:
    path = write_doc(
        docs_dir=tmp_path,
        source="pytorch",
        slug="basics",
        markdown="# PyTorch Basics\n\nTensors are arrays.",
        url="https://example.com/pytorch/basics.html",
        fetched="2026-07-01",
    )

    assert path == tmp_path / "pytorch" / "basics.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "url: https://example.com/pytorch/basics.html" in text
    assert "fetched: 2026-07-01" in text
    assert "# PyTorch Basics" in text
    assert text.endswith("Tensors are arrays.\n")


class FakeResponse:
    def __init__(self, text: str, status_error: Exception | None = None) -> None:
        self.text = text
        self._status_error = status_error
        self.headers = {"content-type": "text/html; charset=utf-8"}
        self.encoding = "utf-8"

    def raise_for_status(self) -> None:
        if self._status_error:
            raise self._status_error


def test_fetch_page_returns_html(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        seen["url"] = url
        seen["kwargs"] = kwargs
        return FakeResponse("<html><body>hi</body></html>")

    monkeypatch.setattr(fetch_docs.requests, "get", fake_get)

    html = fetch_page("https://example.com/page.html")

    assert html == "<html><body>hi</body></html>"
    assert seen["url"] == "https://example.com/page.html"
    # A timeout keeps an overnight run from hanging on one slow page.
    assert seen["kwargs"].get("timeout")


def test_fetch_page_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        return FakeResponse("nope", status_error=RuntimeError("404"))

    monkeypatch.setattr(fetch_docs.requests, "get", fake_get)

    with pytest.raises(RuntimeError):
        fetch_page("https://example.com/missing.html")


def test_fetch_source_writes_one_file_per_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = {
        "https://example.com/a.html": "<main><h1>A</h1><p>alpha</p></main>",
        "https://example.com/b.html": "<main><h1>B</h1><p>beta</p></main>",
    }
    monkeypatch.setattr(fetch_docs, "fetch_page", lambda url: pages[url])

    written = fetch_source("demo", list(pages), docs_dir=tmp_path)

    assert [p.name for p in written] == ["a.md", "b.md"]
    assert "alpha" in (tmp_path / "demo" / "a.md").read_text(encoding="utf-8")
    assert "beta" in (tmp_path / "demo" / "b.md").read_text(encoding="utf-8")


def test_fetch_source_saves_raw_markdown_urls_without_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = "# Function Calling\n\n```python\ntools = [...]  # not html\n```\n"
    monkeypatch.setattr(fetch_docs, "fetch_page", lambda url: raw)

    written = fetch_source(
        "qwen",
        ["https://raw.githubusercontent.com/QwenLM/Qwen3/main/docs/function_call.md"],
        docs_dir=tmp_path,
    )

    text = written[0].read_text(encoding="utf-8")
    # The markdown body is preserved exactly (after the frontmatter block).
    assert text.endswith(raw.strip() + "\n")
    assert written[0].name == "function-call.md"


def test_fetch_source_skips_failed_pages_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def flaky_fetch(url: str) -> str:
        if "bad" in url:
            raise RuntimeError("boom")
        return "<main><h1>Good</h1></main>"

    monkeypatch.setattr(fetch_docs, "fetch_page", flaky_fetch)

    written = fetch_source(
        "demo",
        ["https://example.com/bad.html", "https://example.com/good.html"],
        docs_dir=tmp_path,
    )

    assert [p.name for p in written] == ["good.md"]
