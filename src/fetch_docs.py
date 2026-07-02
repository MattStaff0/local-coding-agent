import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from markdownify import markdownify

from rag import DOCS_DIR

SOURCES_FILE = Path("sources.yaml")
REQUEST_TIMEOUT = 30
# Some doc sites reject the default python-requests user agent.
USER_AGENT = "local-coding-agent-docs-fetcher (personal RAG study tool)"


def load_sources(registry_path: Path) -> dict[str, list[str]]:
    """Read the sources registry mapping source names to lists of doc URLs."""
    text = registry_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return {name: list(urls) for name, urls in data.items()}


def slug_for_url(url: str) -> str:
    """Turn a doc URL into a safe markdown filename stem."""
    path = urlparse(url).path
    segment = Path(path).stem if path.strip("/") else ""

    if not segment:
        return "index"

    # Lowercase and collapse anything that is not filename-friendly to dashes.
    slug = re.sub(r"[^a-z0-9]+", "-", segment.lower()).strip("-")
    return slug or "index"


# Page chrome that should never end up in the markdown docs.
_STRIP_TAGS = ["nav", "footer", "header", "script", "style", "aside", "img"]
_STRIP_SELECTORS = ["[role=navigation]", "[role=search]", "a.headerlink"]

# Most-specific first: sphinx-style sites wrap the real content in <article>
# or [role=main], while <main> often still contains rating/colab chrome.
_CONTENT_SELECTORS = ["article", "[role=main]", "main", "body"]


def html_to_markdown(html: str) -> str:
    """Extract the main content of an HTML page and convert it to markdown."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    for selector in _STRIP_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    content = soup
    for selector in _CONTENT_SELECTORS:
        found = soup.select_one(selector)
        if found:
            content = found
            break

    markdown = markdownify(str(content), heading_style="ATX", code_language="")

    # Dropping nav/images leaves runs of empty lines; keep at most one blank.
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    return markdown.strip()


def write_doc(
    docs_dir: Path,
    source: str,
    slug: str,
    markdown: str,
    url: str,
    fetched: str,
) -> Path:
    """Save converted markdown under docs/<source>/ with provenance frontmatter."""
    source_dir = docs_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)

    path = source_dir / f"{slug}.md"
    frontmatter = f"---\nurl: {url}\nfetched: {fetched}\n---\n\n"
    path.write_text(frontmatter + markdown.strip() + "\n", encoding="utf-8")

    return path


def fetch_page(url: str) -> str:
    """Download one doc page and return its raw HTML."""
    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    # Servers that omit the charset make requests guess ISO-8859-1, which
    # mangles UTF-8 pages (e.g. docs.python.org). Modern doc sites are UTF-8.
    if "charset" not in response.headers.get("content-type", "").lower():
        response.encoding = "utf-8"

    return response.text


def fetch_source(source: str, urls: list[str], docs_dir: Path = DOCS_DIR) -> list[Path]:
    """Fetch every URL for one source; a failed page is reported, not fatal."""
    written = []

    for url in urls:
        try:
            html = fetch_page(url)
        except Exception as error:
            print(f"  FAILED {url}: {error}")
            continue

        markdown = html_to_markdown(html)
        path = write_doc(
            docs_dir=docs_dir,
            source=source,
            slug=slug_for_url(url),
            markdown=markdown,
            url=url,
            fetched=date.today().isoformat(),
        )
        print(f"  {url} -> {path}")
        written.append(path)

    return written


def main() -> None:
    """Fetch doc pages listed in sources.yaml into docs/<source>/.

    Usage: python src/fetch_docs.py [source ...]
    With no arguments every source in the registry is fetched.
    """
    sources = load_sources(SOURCES_FILE)

    requested = sys.argv[1:] or list(sources)
    unknown = [name for name in requested if name not in sources]
    if unknown:
        print(f"Unknown sources: {', '.join(unknown)}")
        print(f"Available: {', '.join(sources)}")
        raise SystemExit(1)

    total = 0
    for name in requested:
        print(f"Fetching source '{name}' ({len(sources[name])} pages)")
        total += len(fetch_source(name, sources[name]))

    print(f"Done. Wrote {total} docs. Re-run 'python src/ingest.py' to index them.")


if __name__ == "__main__":
    main()
