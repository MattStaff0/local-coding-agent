import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlparse

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
    data = yaml.safe_load(text) or {}

    for name, urls in data.items():
        # A bare URL instead of a list would iterate per character and produce
        # dozens of one-letter "pages" — fail with a pointed message instead.
        if isinstance(urls, str) or not isinstance(urls, list):
            raise ValueError(
                f"Source '{name}' in {registry_path} must be a list of URLs, "
                f"got {type(urls).__name__}"
            )

    return {name: list(urls) for name, urls in data.items()}


def slug_for_url(url: str) -> str:
    """Turn a doc URL into a safe markdown filename stem."""
    path = unquote(urlparse(url).path)
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

    # Self-link anchors with no visible text (e.g. Mintlify's zero-width-space
    # heading links) would otherwise leak "[​](#...)" into heading text.
    for anchor in soup.find_all("a", href=lambda h: h and h.startswith("#")):
        if not anchor.get_text().strip("​ \t\n"):
            anchor.decompose()

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
    """Download one doc page: HTML, or raw markdown for .md/.txt URLs."""
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


def markdown_variant_url(url: str) -> str | None:
    """Guess the native-markdown URL for a doc page (llms.txt convention).

    Doc platforms that adopt llms.txt serve `<page>.md` beside each HTML page
    and an `llms.txt` index at the site root — markdown straight from the
    source beats scraping HTML every time.
    """
    parsed = urlparse(url)
    path = parsed.path

    if path.endswith((".md", ".txt")):
        return None

    if not path.strip("/"):
        return f"{parsed.scheme}://{parsed.netloc}/llms.txt"

    if path.endswith((".html", ".htm")):
        new_path = re.sub(r"\.html?$", ".md", path)
    else:
        new_path = path.rstrip("/") + ".md"

    return parsed._replace(path=new_path).geturl()


def probe_native_markdown(url: str) -> tuple[str, str] | None:
    """Fetch a page's markdown variant if the site serves one.

    Returns (variant_url, markdown) on a hit. Sites without the convention
    404; SPA-ish sites answer every path with their HTML shell — both count
    as misses and the caller falls back to HTML scraping.
    """
    variant = markdown_variant_url(url)

    if variant is None:
        return None

    try:
        text = fetch_page(variant)
    except requests.RequestException:
        return None

    # Markdown never leads with a tag; an SPA shell or error page always does.
    if text.lstrip().startswith("<"):
        return None

    return variant, text


def _unique_slug(url: str, taken: set[str]) -> str:
    """Pick a slug that no other URL in this source already claimed."""
    slug = slug_for_url(url)

    if slug in taken:
        # Same final segment (e.g. two .../index.html pages): pull in the
        # parent path segment so both files survive with meaningful names.
        parent = Path(unquote(urlparse(url).path)).parent.name
        parent_slug = re.sub(r"[^a-z0-9]+", "-", parent.lower()).strip("-")
        if parent_slug:
            slug = f"{parent_slug}-{slug}"

    counter = 2
    while slug in taken:
        slug = f"{slug_for_url(url)}-{counter}"
        counter += 1

    taken.add(slug)
    return slug


def parse_refresh(argv: list[str]) -> tuple[int | None, list[str]]:
    """Pull a '--refresh 30d' option out of the CLI arguments."""
    if "--refresh" not in argv:
        return None, list(argv)

    index = argv.index("--refresh")
    remaining = argv[:index] + argv[index + 2 :]

    try:
        value = argv[index + 1]
        days = int(value.rstrip("d"))
    except (IndexError, ValueError):
        raise ValueError(
            "--refresh needs an age in days, like: --refresh 30d"
        ) from None

    return days, remaining


def doc_age_days(path: Path, today: date | None = None) -> int | None:
    """Age of a fetched doc in days, from its frontmatter; None if unknown."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    match = re.search(r"^fetched: (\d{4}-\d{2}-\d{2})$", text, flags=re.MULTILINE)

    if not match:
        return None

    fetched = date.fromisoformat(match.group(1))
    return ((today or date.today()) - fetched).days


def fetch_source(
    source: str,
    urls: list[str],
    docs_dir: Path = DOCS_DIR,
    max_age_days: int | None = None,
) -> tuple[list[Path], list[str]]:
    """Fetch every URL for one source; a failed page is reported, not fatal.

    Returns the written file paths and the URLs that failed. Only network
    failures are skipped — a bug in our own code still crashes loudly. With
    max_age_days set, pages whose existing doc is younger are not re-fetched.
    """
    written = []
    failed = []
    taken_slugs: set[str] = set()

    for url in urls:
        slug = _unique_slug(url, taken_slugs)

        if max_age_days is not None:
            age = doc_age_days(docs_dir / source / f"{slug}.md")
            if age is not None and age <= max_age_days:
                print(f"  {url} is {age}d old, fresh enough — skipped")
                continue

        # Prefer native markdown when the site publishes it (llms.txt
        # convention) — no conversion loss, and far fewer tokens than HTML.
        probe = probe_native_markdown(url)

        if probe is not None:
            fetched_url, markdown = probe
        else:
            fetched_url = url
            try:
                html = fetch_page(url)
            except requests.RequestException as error:
                print(f"  FAILED {url}: {error}")
                failed.append(url)
                continue

            # Pages already published as markdown need no conversion at all.
            if urlparse(url).path.endswith((".md", ".txt")):
                markdown = html
            else:
                markdown = html_to_markdown(html)

        path = write_doc(
            docs_dir=docs_dir,
            source=source,
            slug=slug,
            markdown=markdown,
            url=fetched_url,
            fetched=date.today().isoformat(),
        )
        print(f"  {url} -> {path}")
        written.append(path)

    return written, failed


def main() -> None:
    """Fetch doc pages listed in sources.yaml into docs/<source>/.

    Usage: python src/fetch_docs.py [--refresh 30d] [source ...]
    With no sources every source in the registry is fetched. --refresh only
    re-downloads pages whose saved copy is older than the given age.
    """
    sources = load_sources(SOURCES_FILE)

    try:
        max_age_days, args = parse_refresh(sys.argv[1:])
    except ValueError as error:
        print(error)
        raise SystemExit(1)

    requested = args or list(sources)
    unknown = [name for name in requested if name not in sources]
    if unknown:
        print(f"Unknown sources: {', '.join(unknown)}")
        print(f"Available: {', '.join(sources)}")
        raise SystemExit(1)

    total = 0
    all_failed: list[str] = []
    for name in requested:
        print(f"Fetching source '{name}' ({len(sources[name])} pages)")
        written, failed = fetch_source(
            name, sources[name], max_age_days=max_age_days
        )
        total += len(written)
        all_failed.extend(failed)

    print(f"Done. Wrote {total} docs. Re-run 'python src/ingest.py' to index them.")

    # Per-page FAILED lines scroll away in a long run; summarize and exit
    # nonzero so an overnight fetch can't fail silently.
    if all_failed:
        print(f"WARNING: {len(all_failed)} pages failed:")
        for url in all_failed:
            print(f"  - {url}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
