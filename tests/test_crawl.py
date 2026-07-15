"""Crawl mode: discover doc pages under a URL prefix, politely."""
import fetch_docs


INDEX_HTML = """
<html><body>
  <a href="intro.html">Intro</a>
  <a href="/tutorials/beginner/tensors.html">Tensors</a>
  <a href="tensors.html#section">Anchor dupe</a>
  <a href="https://elsewhere.example.com/off-prefix.html">Elsewhere</a>
  <a href="../advanced/off-prefix.html">Parent escape</a>
  <a href="pic.png">Image</a>
</body></html>
"""

PREFIX = "https://docs.example.com/tutorials/beginner/"


def test_crawl_discovers_only_links_under_the_prefix(monkeypatch):
    monkeypatch.setattr(fetch_docs, "fetch_page", lambda url: INDEX_HTML)

    urls = fetch_docs.crawl_urls(PREFIX)

    assert urls == [
        PREFIX,
        f"{PREFIX}intro.html",
        f"{PREFIX}tensors.html",
    ]


def test_crawl_respects_the_page_cap(monkeypatch):
    monkeypatch.setattr(fetch_docs, "fetch_page", lambda url: INDEX_HTML)
    urls = fetch_docs.crawl_urls(PREFIX, max_pages=2)
    assert len(urls) == 2


def test_crawl_index_failure_returns_no_urls(monkeypatch, capsys):
    def boom(url):
        raise fetch_docs.requests.ConnectionError("no route")

    monkeypatch.setattr(fetch_docs, "fetch_page", boom)

    assert fetch_docs.crawl_urls(PREFIX) == []
    assert "crawl failed" in capsys.readouterr().out


def test_load_sources_normalizes_lists_and_crawl_dicts(tmp_path):
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        "python:\n"
        "  - https://docs.python.org/3/tutorial/datastructures.html\n"
        "sklearn:\n"
        "  crawl: https://scikit-learn.org/stable/modules/\n"
        "  max_pages: 5\n"
        "  delay: 0.5\n",
        encoding="utf-8",
    )

    sources = fetch_docs.load_sources(registry)

    assert sources["python"]["pages"] == [
        "https://docs.python.org/3/tutorial/datastructures.html"
    ]
    assert sources["python"]["crawl"] is None
    assert sources["sklearn"]["crawl"] == "https://scikit-learn.org/stable/modules/"
    assert sources["sklearn"]["max_pages"] == 5
    assert sources["sklearn"]["delay"] == 0.5


def test_load_sources_still_rejects_bare_strings(tmp_path):
    registry = tmp_path / "sources.yaml"
    registry.write_text("python: https://docs.python.org\n", encoding="utf-8")

    try:
        fetch_docs.load_sources(registry)
    except ValueError as error:
        assert "python" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_fetch_source_crawls_and_sleeps_between_pages(tmp_path, monkeypatch):
    pages = {
        PREFIX: INDEX_HTML,
        f"{PREFIX}intro.html": "<html><body><p>intro text</p></body></html>",
        f"{PREFIX}tensors.html": "<html><body><p>tensor text</p></body></html>",
    }
    # Markdown-variant probes miss (HTML shell), page fetches hit.
    monkeypatch.setattr(
        fetch_docs, "fetch_page", lambda url: pages.get(url, "<html>miss</html>")
    )
    naps = []
    monkeypatch.setattr(fetch_docs.time, "sleep", lambda s: naps.append(s))

    config = {"pages": [], "crawl": PREFIX, "max_pages": 10, "delay": 0.25}
    written, failed = fetch_docs.fetch_source("example", config, docs_dir=tmp_path)

    assert failed == []
    assert len(written) == 3  # the index page itself + the two linked pages
    assert (tmp_path / "example" / "intro.md").exists()
    assert (tmp_path / "example" / "tensors.md").exists()
    # Politeness: one nap between each consecutive page download.
    assert naps == [0.25, 0.25]


def test_fetch_source_still_accepts_plain_url_lists(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch_docs,
        "fetch_page",
        lambda url: "<html><body><p>page body</p></body></html>",
    )

    written, failed = fetch_docs.fetch_source(
        "python",
        ["https://docs.example.com/one.html"],
        docs_dir=tmp_path,
    )

    assert failed == []
    assert [p.name for p in written] == ["one.md"]
