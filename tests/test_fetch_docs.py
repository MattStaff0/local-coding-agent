import sys
from datetime import date
from pathlib import Path

import pytest
import requests

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


def test_load_sources_rejects_scalar_url(tmp_path: Path) -> None:
    # A common YAML typo: a bare URL instead of a list. Without validation it
    # would explode into per-character "pages".
    registry = tmp_path / "sources.yaml"
    registry.write_text("pytorch: https://example.com/a.html\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pytorch"):
        load_sources(registry)


def test_load_sources_empty_file_means_no_sources(tmp_path: Path) -> None:
    registry = tmp_path / "sources.yaml"
    registry.write_text("", encoding="utf-8")

    assert load_sources(registry) == {}


def test_slug_for_url_uses_last_path_segment() -> None:
    url = "https://docs.python.org/3/tutorial/datastructures.html"
    assert slug_for_url(url) == "datastructures"


def test_slug_for_url_decodes_percent_escapes() -> None:
    url = "https://example.com/guide/My%20Page/"
    assert slug_for_url(url) == "my-page"


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

    def fake_fetch(url: str) -> str:
        if url not in pages:  # the .md probe: this site has no native markdown
            raise requests.RequestException("404")
        return pages[url]

    monkeypatch.setattr(fetch_docs, "fetch_page", fake_fetch)

    written, failed = fetch_source("demo", list(pages), docs_dir=tmp_path)

    assert [p.name for p in written] == ["a.md", "b.md"]
    assert failed == []
    assert "alpha" in (tmp_path / "demo" / "a.md").read_text(encoding="utf-8")
    assert "beta" in (tmp_path / "demo" / "b.md").read_text(encoding="utf-8")


def test_fetch_source_disambiguates_colliding_slugs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls = [
        "https://example.com/tensors/index.html",
        "https://example.com/models/index.html",
    ]
    monkeypatch.setattr(
        fetch_docs, "fetch_page", lambda url: f"<main><h1>{url}</h1></main>"
    )

    written, failed = fetch_source("demo", urls, docs_dir=tmp_path)

    # Both pages survive: same final URL segment must not overwrite a file.
    assert len(written) == 2
    assert len({p.name for p in written}) == 2
    assert failed == []
    texts = [p.read_text(encoding="utf-8") for p in written]
    assert any("tensors" in t for t in texts)
    assert any("models" in t for t in texts)


def test_fetch_source_saves_raw_markdown_urls_without_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = "# Function Calling\n\n```python\ntools = [...]  # not html\n```\n"
    monkeypatch.setattr(fetch_docs, "fetch_page", lambda url: raw)

    written, failed = fetch_source(
        "qwen",
        ["https://raw.githubusercontent.com/QwenLM/Qwen3/main/docs/function_call.md"],
        docs_dir=tmp_path,
    )

    text = written[0].read_text(encoding="utf-8")
    # The markdown body is preserved exactly (after the frontmatter block).
    assert text.endswith(raw.strip() + "\n")
    assert written[0].name == "function-call.md"


def test_fetch_source_skips_failed_pages_and_reports_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def flaky_fetch(url: str) -> str:
        if "bad" in url:
            raise requests.ConnectionError("boom")
        return "<main><h1>Good</h1></main>"

    monkeypatch.setattr(fetch_docs, "fetch_page", flaky_fetch)

    written, failed = fetch_source(
        "demo",
        ["https://example.com/bad.html", "https://example.com/good.html"],
        docs_dir=tmp_path,
    )

    assert [p.name for p in written] == ["good.md"]
    assert failed == ["https://example.com/bad.html"]


def test_fetch_source_lets_programmer_bugs_crash_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only network failures are skip-and-continue; a code bug must not be
    # reported as a run of flaky pages.
    def buggy_fetch(url: str) -> str:
        raise TypeError("bug in fetch_page")

    monkeypatch.setattr(fetch_docs, "fetch_page", buggy_fetch)

    with pytest.raises(TypeError):
        fetch_source("demo", ["https://example.com/a.html"], docs_dir=tmp_path)


def test_main_rejects_unknown_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    registry = tmp_path / "sources.yaml"
    registry.write_text("pytorch:\n  - https://example.com/a.html\n", encoding="utf-8")
    monkeypatch.setattr(fetch_docs, "SOURCES_FILE", registry)
    monkeypatch.setattr(sys, "argv", ["fetch_docs.py", "pytroch"])

    with pytest.raises(SystemExit) as excinfo:
        fetch_docs.main()

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "pytroch" in out and "pytorch" in out


def test_main_exits_nonzero_when_pages_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    registry = tmp_path / "sources.yaml"
    registry.write_text("demo:\n  - https://example.com/a.html\n", encoding="utf-8")
    monkeypatch.setattr(fetch_docs, "SOURCES_FILE", registry)
    monkeypatch.setattr(sys, "argv", ["fetch_docs.py"])
    monkeypatch.setattr(
        fetch_docs,
        "fetch_source",
        lambda name, urls, docs_dir=None, max_age_days=None: ([], list(urls)),
    )

    with pytest.raises(SystemExit) as excinfo:
        fetch_docs.main()

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "https://example.com/a.html" in out
    assert "failed" in out.lower()


# --- Native markdown probing (llms.txt convention) ---


def test_markdown_variant_swaps_html_suffix() -> None:
    assert (
        fetch_docs.markdown_variant_url("https://ex.com/docs/page.html")
        == "https://ex.com/docs/page.md"
    )


def test_markdown_variant_appends_md_to_extensionless_pages() -> None:
    assert (
        fetch_docs.markdown_variant_url("https://ex.com/capabilities/tools")
        == "https://ex.com/capabilities/tools.md"
    )


def test_markdown_variant_probes_llms_txt_for_site_roots() -> None:
    assert (
        fetch_docs.markdown_variant_url("https://ex.com/")
        == "https://ex.com/llms.txt"
    )


def test_markdown_variant_skips_urls_already_markdown() -> None:
    assert fetch_docs.markdown_variant_url("https://ex.com/page.md") is None
    assert fetch_docs.markdown_variant_url("https://ex.com/llms.txt") is None


def test_probe_hit_uses_native_markdown_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith(".md"):
            return FakeResponse("# Native\n\nreal markdown")
        raise AssertionError(f"HTML should not be fetched, got {url}")

    monkeypatch.setattr(fetch_docs.requests, "get", fake_get)

    written, failed = fetch_docs.fetch_source(
        "ex", ["https://ex.com/docs/tools"], docs_dir=tmp_path
    )

    assert not failed
    content = written[0].read_text(encoding="utf-8")
    assert "# Native\n\nreal markdown" in content
    # Provenance records the markdown variant actually fetched.
    assert "url: https://ex.com/docs/tools.md" in content


def test_probe_miss_falls_back_to_html_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith(".md"):
            return FakeResponse("nope", status_error=requests.HTTPError("404"))
        return FakeResponse("<html><main><h1>T</h1><p>body</p></main></html>")

    monkeypatch.setattr(fetch_docs.requests, "get", fake_get)

    written, failed = fetch_docs.fetch_source(
        "ex", ["https://ex.com/docs/tools"], docs_dir=tmp_path
    )

    assert not failed
    content = written[0].read_text(encoding="utf-8")
    assert "# T" in content
    assert "url: https://ex.com/docs/tools\n" in content


def test_probe_that_returns_html_is_treated_as_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        if url.endswith(".md"):
            # Soft-404: the site serves its SPA shell at any path.
            return FakeResponse("<!DOCTYPE html><html><body>app</body></html>")
        return FakeResponse("<html><main><h1>Real</h1></main></html>")

    monkeypatch.setattr(fetch_docs.requests, "get", fake_get)

    written, failed = fetch_docs.fetch_source(
        "ex", ["https://ex.com/docs/tools"], docs_dir=tmp_path
    )

    assert not failed
    assert "# Real" in written[0].read_text(encoding="utf-8")


# --- Staleness refresh (--refresh Nd) ---


def test_parse_refresh_extracts_days_and_remaining_args() -> None:
    assert fetch_docs.parse_refresh(["--refresh", "30d", "ollama"]) == (30, ["ollama"])
    assert fetch_docs.parse_refresh(["pytorch"]) == (None, ["pytorch"])


def test_parse_refresh_rejects_malformed_values() -> None:
    with pytest.raises(ValueError, match="refresh"):
        fetch_docs.parse_refresh(["--refresh", "soon"])
    with pytest.raises(ValueError, match="refresh"):
        fetch_docs.parse_refresh(["--refresh"])


def test_doc_age_days_reads_the_fetched_date(tmp_path: Path) -> None:
    doc = tmp_path / "page.md"
    doc.write_text(
        "---\nurl: https://ex.com/p\nfetched: 2026-06-01\n---\n\n# P\n",
        encoding="utf-8",
    )

    age = fetch_docs.doc_age_days(doc, today=date(2026, 7, 1))

    assert age == 30


def test_doc_age_days_is_none_without_frontmatter(tmp_path: Path) -> None:
    doc = tmp_path / "page.md"
    doc.write_text("# no frontmatter\n", encoding="utf-8")

    assert fetch_docs.doc_age_days(doc, today=date(2026, 7, 1)) is None


def test_fetch_source_skips_docs_fresher_than_max_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh = tmp_path / "demo" / "fresh.md"
    fresh.parent.mkdir(parents=True)
    fresh.write_text(
        f"---\nurl: https://ex.com/fresh\nfetched: {date.today().isoformat()}\n---\n\nok\n",
        encoding="utf-8",
    )

    def fake_fetch(url: str) -> str:
        raise AssertionError(f"fresh page must not be fetched: {url}")

    monkeypatch.setattr(fetch_docs, "fetch_page", fake_fetch)

    written, failed = fetch_source(
        "demo", ["https://ex.com/fresh"], docs_dir=tmp_path, max_age_days=30
    )

    assert written == []
    assert failed == []


def test_fetch_source_refetches_docs_older_than_max_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = tmp_path / "demo" / "stale.md"
    stale.parent.mkdir(parents=True)
    stale.write_text(
        "---\nurl: https://ex.com/stale\nfetched: 2020-01-01\n---\n\nold\n",
        encoding="utf-8",
    )

    def fake_fetch(url: str) -> str:
        if url.endswith(".md"):
            raise requests.RequestException("404")
        return "<main><h1>New</h1></main>"

    monkeypatch.setattr(fetch_docs, "fetch_page", fake_fetch)

    written, failed = fetch_source(
        "demo", ["https://ex.com/stale"], docs_dir=tmp_path, max_age_days=30
    )

    assert len(written) == 1
    assert "New" in written[0].read_text(encoding="utf-8")
