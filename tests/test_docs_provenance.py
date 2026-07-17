"""Registry v2, fetch provenance, and version-labeled chunks (workstream 04)."""
import json
from pathlib import Path

import pytest

import fetch_docs
import rag


# --- registry v2 ---


def write_registry(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(text)
    return path


def test_v2_source_keys_parse_with_defaults(tmp_path):
    registry = write_registry(
        tmp_path,
        """
numpy:
  pages: [https://numpy.org/doc/stable/user/quickstart.html]
  distribution: numpy
  import_names: [numpy]
  docs_version_pattern: 'numpy\\.org/doc/(?P<version>[\\d.]+|stable)/'
  refresh_ttl_days: 14
  version_policy: major_minor
""",
    )

    source = fetch_docs.load_sources(registry)["numpy"]

    assert source["distribution"] == "numpy"
    assert source["refresh_ttl_days"] == 14
    # official_origins defaults to the origins of the configured pages.
    assert source["official_origins"] == ["https://numpy.org"]


def test_v1_list_form_still_works_with_v2_defaults(tmp_path):
    registry = write_registry(
        tmp_path, "pytorch:\n  - https://docs.pytorch.org/tutorials/a.html\n"
    )

    source = fetch_docs.load_sources(registry)["pytorch"]

    assert source["pages"] == ["https://docs.pytorch.org/tutorials/a.html"]
    assert source["distribution"] == "pytorch"
    assert source["official_origins"] == ["https://docs.pytorch.org"]
    assert source["refresh_ttl_days"] == 7


def test_unknown_registry_keys_are_rejected_naming_the_source(tmp_path):
    registry = write_registry(
        tmp_path, "numpy:\n  pages: []\n  version_polciy: major\n"
    )

    with pytest.raises(ValueError, match="numpy.*version_polciy"):
        fetch_docs.load_sources(registry)


# --- provenance in fetched docs ---


class FakeResponse:
    def __init__(self, text="# Title\n\nBody.", status_code=200, headers=None, url=None):
        self.text = text
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}
        self.url = url
        self.encoding = "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise fetch_docs.requests.HTTPError(str(self.status_code))


def numpy_config(**overrides):
    config = {
        "pages": ["https://numpy.org/doc/1.26/user/quickstart.html"],
        "official_origins": ["https://numpy.org"],
        "docs_version_pattern": r"numpy\.org/doc/(?P<version>[\d.]+|stable)/",
    }
    config.update(overrides)
    return fetch_docs._normalize_source(config, "numpy", Path("sources.yaml"))


def test_fetch_records_validators_and_docs_version(tmp_path, monkeypatch):
    url = "https://numpy.org/doc/1.26/user/quickstart.html"

    def fake_get(request_url, **kwargs):
        return FakeResponse(
            "<main><h1>Quickstart</h1><p>hello</p></main>",
            headers={
                "content-type": "text/html; charset=utf-8",
                "ETag": '"abc123"',
                "Last-Modified": "Wed, 01 Jan 2026 00:00:00 GMT",
            },
            url=request_url,
        )

    monkeypatch.setattr(fetch_docs.requests, "get", fake_get)
    monkeypatch.setattr(fetch_docs, "probe_native_markdown", lambda url: None)

    written, failed = fetch_docs.fetch_source("numpy", numpy_config(), docs_dir=tmp_path)

    assert failed == []
    text = written[0].read_text()
    assert 'etag: "abc123"' in text
    assert "last_modified: Wed, 01 Jan 2026 00:00:00 GMT" in text
    assert "docs_version: 1.26" in text


def test_stable_url_records_stable_at_fetch(tmp_path, monkeypatch):
    config = numpy_config(
        pages=["https://numpy.org/doc/stable/user/quickstart.html"]
    )
    monkeypatch.setattr(
        fetch_docs.requests,
        "get",
        lambda url, **kwargs: FakeResponse("<main><p>hi</p></main>", url=url),
    )
    monkeypatch.setattr(fetch_docs, "probe_native_markdown", lambda url: None)

    written, _ = fetch_docs.fetch_source("numpy", config, docs_dir=tmp_path)

    assert "docs_version: stable-at-fetch" in written[0].read_text()


def test_redirect_off_official_origin_is_rejected_keeping_last_good(
    tmp_path, monkeypatch
):
    config = numpy_config()
    existing = tmp_path / "numpy" / "quickstart.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("---\nurl: x\nfetched: 2026-01-01\n---\n\nold good copy\n")

    monkeypatch.setattr(
        fetch_docs.requests,
        "get",
        lambda url, **kwargs: FakeResponse(
            "<main><p>evil</p></main>", url="https://evil.example.com/page"
        ),
    )
    monkeypatch.setattr(fetch_docs, "probe_native_markdown", lambda url: None)

    written, failed = fetch_docs.fetch_source("numpy", config, docs_dir=tmp_path)

    assert written == []
    assert failed == config["pages"]
    assert "old good copy" in existing.read_text()


def test_conditional_refresh_304_keeps_content_and_updates_fetched(
    tmp_path, monkeypatch
):
    config = numpy_config()
    existing = tmp_path / "numpy" / "quickstart.md"
    existing.parent.mkdir(parents=True)
    existing.write_text(
        '---\nurl: https://numpy.org/doc/1.26/user/quickstart.html\n'
        'fetched: 2020-01-01\netag: "abc123"\n---\n\ncached body\n'
    )

    seen_headers = {}

    def fake_get(url, **kwargs):
        seen_headers.update(kwargs.get("headers", {}))
        return FakeResponse("", status_code=304, url=url)

    monkeypatch.setattr(fetch_docs.requests, "get", fake_get)
    monkeypatch.setattr(fetch_docs, "probe_native_markdown", lambda url: None)

    written, failed = fetch_docs.fetch_source(
        "numpy", config, docs_dir=tmp_path, max_age_days=0
    )

    assert failed == []
    assert seen_headers.get("If-None-Match") == '"abc123"'
    text = existing.read_text()
    assert "cached body" in text
    assert "fetched: 2020-01-01" not in text  # validated today


# --- ingest carries provenance into chunk metadata ---


def test_prepare_file_reads_provenance_frontmatter(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    (docs / "numpy").mkdir(parents=True)
    doc = docs / "numpy" / "quickstart.md"
    doc.write_text(
        "---\nurl: https://numpy.org/doc/1.26/user/quickstart.html\n"
        "fetched: 2026-07-01\ndocs_version: 1.26\n---\n\n# Quickstart\n\nBody text.\n"
    )
    monkeypatch.setattr(rag, "embed_batch", lambda texts: [[0.0]] * len(texts))

    _, _, _, metadatas = rag._prepare_file(doc, doc.read_text(), docs)

    assert metadatas[0]["url"] == "https://numpy.org/doc/1.26/user/quickstart.html"
    assert metadatas[0]["fetched"] == "2026-07-01"
    assert metadatas[0]["docs_version"] == "1.26"


def test_prepare_file_without_provenance_omits_the_keys(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    (docs / "notes").mkdir(parents=True)
    doc = docs / "notes" / "plain.md"
    doc.write_text("# Plain\n\nNo frontmatter at all.\n")
    monkeypatch.setattr(rag, "embed_batch", lambda texts: [[0.0]] * len(texts))

    _, _, _, metadatas = rag._prepare_file(doc, doc.read_text(), docs)

    assert "url" not in metadatas[0]
    assert "docs_version" not in metadatas[0]


# --- review findings (codex, 2026-07-17) ---


def test_probe_result_from_off_origin_redirect_is_rejected(tmp_path, monkeypatch):
    config = numpy_config()
    monkeypatch.setattr(
        fetch_docs,
        "probe_native_markdown",
        lambda url: ("https://evil.example.com/quickstart.md", "# evil markdown"),
    )

    written, failed = fetch_docs.fetch_source("numpy", config, docs_dir=tmp_path)

    assert written == []
    assert failed == config["pages"]


def test_304_from_off_origin_redirect_does_not_validate_cache(
    tmp_path, monkeypatch
):
    config = numpy_config()
    existing = tmp_path / "numpy" / "quickstart.md"
    existing.parent.mkdir(parents=True)
    existing.write_text(
        '---\nurl: x\nfetched: 2020-01-01\netag: "abc"\n---\n\ncached\n'
    )

    monkeypatch.setattr(
        fetch_docs.requests,
        "get",
        lambda url, **kwargs: FakeResponse(
            "", status_code=304, url="https://evil.example.com/q"
        ),
    )
    monkeypatch.setattr(fetch_docs, "probe_native_markdown", lambda url: None)

    _, failed = fetch_docs.fetch_source(
        "numpy", config, docs_dir=tmp_path, max_age_days=0
    )

    assert failed == config["pages"]
    assert "fetched: 2020-01-01" in existing.read_text()  # NOT touched


def test_sources_file_is_anchored_to_project_root_not_cwd():
    import paths

    assert fetch_docs.SOURCES_FILE.is_absolute()
    assert fetch_docs.SOURCES_FILE == paths.PROJECT_ROOT / "sources.yaml"


def test_rebuild_manifest_carries_provenance(tmp_path):
    class FakeCollection:
        def get(self, include):
            return {
                "ids": ["a-0", "b-0"],
                "documents": ["versioned chunk", "plain chunk"],
                "metadatas": [
                    {
                        "relative_path": "pandas/x.md",
                        "source": "pandas",
                        "heading": "H",
                        "file_hash": "h1",
                        "url": "https://pandas.pydata.org/docs/x.html",
                        "fetched": "2026-07-01",
                        "docs_version": "3.0",
                    },
                    {
                        "relative_path": "notes/y.md",
                        "source": "notes",
                        "heading": "H",
                        "file_hash": "h2",
                    },
                ],
            }

    count = rag.rebuild_manifest(FakeCollection(), path=tmp_path / "m.jsonl")

    records = [
        json.loads(line)
        for line in (tmp_path / "m.jsonl").read_text().splitlines()
    ]
    assert count == 2
    assert records[0]["docs_version"] == "3.0"
    assert records[0]["fetched"] == "2026-07-01"
    assert "docs_version" not in records[1]


def test_file_hash_includes_metadata_schema_version():
    # Bumping the schema must re-embed (and re-metadata) every file once, so
    # a pre-WS04 index cannot keep hash-matching while lacking provenance.
    import hashlib

    legacy = hashlib.sha256(f"{rag.EMBED_MODEL}\nsome text".encode()).hexdigest()

    assert rag._file_hash("some text") != legacy
