import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from markdownify import markdownify

from rag import DOCS_DIR

SOURCES_FILE = Path("sources.yaml")
REQUEST_TIMEOUT = 30
# Some doc sites reject the default python-requests user agent.
USER_AGENT = "local-coding-agent-docs-fetcher (personal RAG study tool)"

# Crawl politeness defaults: enough pages for one doc section, one request
# per second so a personal tool never looks like a scraper.
DEFAULT_CRAWL_MAX_PAGES = 30
DEFAULT_CRAWL_DELAY = 1.0

_SKIP_LINK_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".zip")


def load_sources(registry_path: Path) -> dict[str, dict]:
    """Read the sources registry; every source normalizes to one dict.

    A source is either a plain list of page URLs (the original format) or a
    mapping with per-source options:

        sklearn:
          crawl: https://scikit-learn.org/stable/modules/
          max_pages: 30
          delay: 1.0
          pages: [...]        # optional extra hand-picked URLs
    """
    text = registry_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}

    sources: dict[str, dict] = {}

    for name, value in data.items():
        # A bare URL instead of a list would iterate per character and produce
        # dozens of one-letter "pages" — fail with a pointed message instead.
        if isinstance(value, list):
            sources[name] = _normalize_source({"pages": value}, name, registry_path)
        elif isinstance(value, dict):
            sources[name] = _normalize_source(value, name, registry_path)
        else:
            raise ValueError(
                f"Source '{name}' in {registry_path} must be a list of URLs "
                f"or a mapping with crawl/pages, got {type(value).__name__}"
            )

    return sources


# Registry v2 (workstream 04): version/provenance keys. Unknown keys are
# rejected so a typo cannot silently disable origin checks or TTLs.
_KNOWN_SOURCE_KEYS = {
    "pages", "crawl", "max_pages", "delay",
    "official_origins", "distribution", "import_names",
    "docs_version_pattern", "versioned_url_template",
    "refresh_ttl_days", "version_policy",
}

DEFAULT_REFRESH_TTL_DAYS = 7


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _normalize_source(value: dict, name: str, registry_path: Path) -> dict:
    unknown = set(value) - _KNOWN_SOURCE_KEYS
    if unknown:
        raise ValueError(
            f"Source '{name}' in {registry_path}: unknown keys: "
            f"{', '.join(sorted(unknown))}"
        )

    pages = value.get("pages", [])

    if isinstance(pages, str) or not isinstance(pages, list):
        raise ValueError(
            f"Source '{name}' in {registry_path}: 'pages' must be a list of "
            f"URLs, got {type(pages).__name__}"
        )

    origins = value.get("official_origins")
    if not origins:
        candidates = list(pages) + ([value["crawl"]] if value.get("crawl") else [])
        origins = sorted({_origin(url) for url in candidates})

    distribution = value.get("distribution", name)

    return {
        "pages": list(pages),
        "crawl": value.get("crawl"),
        "max_pages": int(value.get("max_pages", DEFAULT_CRAWL_MAX_PAGES)),
        "delay": float(value.get("delay", DEFAULT_CRAWL_DELAY)),
        "official_origins": list(origins),
        "distribution": distribution,
        "import_names": list(value.get("import_names", [distribution])),
        "docs_version_pattern": value.get("docs_version_pattern"),
        "versioned_url_template": value.get("versioned_url_template"),
        "refresh_ttl_days": int(
            value.get("refresh_ttl_days", DEFAULT_REFRESH_TTL_DAYS)
        ),
        "version_policy": value.get("version_policy", "major_minor"),
    }


def crawl_urls(
    prefix: str, max_pages: int = DEFAULT_CRAWL_MAX_PAGES
) -> list[str]:
    """Discover doc pages: fetch the prefix page, keep its links under prefix.

    Depth-1 on purpose — doc sections almost always have an index page that
    links every page in the section. Point crawl at that index; nested
    sub-indexes need their own crawl entry. Fragments are stripped, binary
    assets skipped, order preserved, capped at max_pages (index included).
    """
    try:
        html = fetch_page(prefix)
    except requests.RequestException as error:
        print(f"  crawl failed for {prefix}: {error}")
        return []

    urls = [prefix]
    seen = {prefix}
    soup = BeautifulSoup(html, "html.parser")

    for anchor in soup.find_all("a", href=True):
        link = urljoin(prefix, anchor["href"]).split("#")[0]

        if (
            link in seen
            or not link.startswith(prefix)
            or link.lower().endswith(_SKIP_LINK_SUFFIXES)
        ):
            continue

        seen.add(link)
        urls.append(link)

        if len(urls) >= max_pages:
            break

    return urls


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
    etag: str | None = None,
    last_modified: str | None = None,
    docs_version: str | None = None,
) -> Path:
    """Save converted markdown under docs/<source>/ with provenance frontmatter."""
    source_dir = docs_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)

    path = source_dir / f"{slug}.md"
    lines = [f"url: {url}", f"fetched: {fetched}"]
    if etag:
        lines.append(f"etag: {etag}")
    if last_modified:
        lines.append(f"last_modified: {last_modified}")
    if docs_version:
        lines.append(f"docs_version: {docs_version}")
    frontmatter = "---\n" + "\n".join(lines) + "\n---\n\n"
    path.write_text(frontmatter + markdown.strip() + "\n", encoding="utf-8")

    return path


def _doc_validators(path: Path) -> dict[str, str]:
    """Stored HTTP validators from a fetched doc's frontmatter, if any."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    validators = {}
    etag = re.search(r"^etag: (.+)$", text, flags=re.MULTILINE)
    if etag:
        validators["etag"] = etag.group(1).strip()
    modified = re.search(r"^last_modified: (.+)$", text, flags=re.MULTILINE)
    if modified:
        validators["last_modified"] = modified.group(1).strip()
    return validators


def _touch_fetched(path: Path, today: str) -> None:
    """A 304 proves currency: refresh the fetched date, keep the content."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    path.write_text(
        re.sub(
            r"^fetched: .+$", f"fetched: {today}", text, count=1, flags=re.MULTILINE
        ),
        encoding="utf-8",
    )


def docs_version_for_url(url: str, pattern: str | None) -> str | None:
    """Docs version read out of a URL, or 'stable-at-fetch' when unpinned.

    'stable-at-fetch' is deliberately not a version: a stable-alias page is
    whatever the project shipped that day, so compatibility stays unknown.
    """
    if not pattern:
        return None
    match = re.search(pattern, url)
    if not match:
        return "stable-at-fetch"
    version = match.group("version")
    return "stable-at-fetch" if version == "stable" else version


def fetch_response(url: str, extra_headers: dict | None = None):
    """GET one doc page; returns the response (304 passes through un-raised)."""
    headers = {"User-Agent": USER_AGENT}
    if extra_headers:
        headers.update(extra_headers)

    response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)

    if response.status_code == 304:
        return response

    response.raise_for_status()

    # Servers that omit the charset make requests guess ISO-8859-1, which
    # mangles UTF-8 pages (e.g. docs.python.org). Modern doc sites are UTF-8.
    if "charset" not in response.headers.get("content-type", "").lower():
        response.encoding = "utf-8"

    return response


def fetch_page(url: str) -> str:
    """Download one doc page: HTML, or raw markdown for .md/.txt URLs."""
    return fetch_response(url).text


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
    except requests.HTTPError:
        # The designed miss: this site simply doesn't serve the convention.
        return None
    except requests.RequestException as error:
        # A timeout/DNS/TLS problem is not a miss — say why the quality
        # downgrade to HTML scraping is happening.
        print(f"  markdown probe failed for {variant}: {error} — scraping HTML")
        return None

    # Heuristic: a leading "<" almost always means an HTML shell or error
    # page. A rare markdown file that opens with raw HTML is misclassified
    # and just falls back to HTML scraping.
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
    config: dict | list[str],
    docs_dir: Path = DOCS_DIR,
    max_age_days: int | None = None,
) -> tuple[list[Path], list[str]]:
    """Fetch every page for one source; a failed page is reported, not fatal.

    Accepts a plain URL list (the original format) or a normalized source
    dict from load_sources. A 'crawl' prefix expands into discovered page
    URLs first, and crawled sources pause 'delay' seconds between downloads.

    Returns the written file paths and the URLs that failed. Only network
    failures are skipped — a bug in our own code still crashes loudly. With
    max_age_days set, pages whose existing doc is not older than that age are
    not re-fetched.
    """
    if isinstance(config, list):
        config = {"pages": config, "crawl": None, "delay": 0.0}

    urls = list(config.get("pages", []))
    delay = 0.0

    if config.get("crawl"):
        delay = config.get("delay", DEFAULT_CRAWL_DELAY)
        known = set(urls)
        urls.extend(
            url
            for url in crawl_urls(
                config["crawl"],
                max_pages=config.get("max_pages", DEFAULT_CRAWL_MAX_PAGES),
            )
            if url not in known
        )

    written = []
    failed = []
    taken_slugs: set[str] = set()

    origins = config.get("official_origins") or []

    for index, url in enumerate(urls):
        if delay and index:
            time.sleep(delay)

        slug = _unique_slug(url, taken_slugs)
        existing = docs_dir / source / f"{slug}.md"

        conditional: dict[str, str] = {}
        if max_age_days is not None:
            age = doc_age_days(existing)
            if age is not None and age <= max_age_days:
                print(f"  {url} is {age}d old, fresh enough — skipped")
                continue
            # Conditional refresh: let the server say "unchanged" cheaply.
            validators = _doc_validators(existing)
            if validators.get("etag"):
                conditional["If-None-Match"] = validators["etag"]
            if validators.get("last_modified"):
                conditional["If-Modified-Since"] = validators["last_modified"]

        # Prefer native markdown when the site publishes it (llms.txt
        # convention) — no conversion loss, and far fewer tokens than HTML.
        # A conditional refresh skips the probe: validators belong to the
        # URL variant they came from.
        probe = None if conditional else probe_native_markdown(url)
        meta: dict[str, str | None] = {}
        # Sources without v2 metadata (raw URL lists) keep the plain-text
        # fetch path; origin enforcement and validators need the response.
        wants_response = bool(origins or config.get("docs_version_pattern"))

        if probe is not None:
            fetched_url, markdown = probe
        else:
            fetched_url = url
            try:
                if conditional or wants_response:
                    response = fetch_response(url, conditional)
                else:
                    response = None
                    html = fetch_page(url)
            except requests.RequestException as error:
                print(f"  FAILED {url}: {error}")
                failed.append(url)
                continue

            if response is not None:
                if response.status_code == 304:
                    _touch_fetched(existing, date.today().isoformat())
                    print(f"  {url} not modified — cache validated")
                    continue

                final_url = response.url or url
                if origins and _origin(final_url) not in origins:
                    # A redirect off the official origin is a rejected page,
                    # never a replacement for the last good copy.
                    print(f"  FAILED {url}: redirected off official origin "
                          f"({final_url})")
                    failed.append(url)
                    continue

                html = response.text
                meta = {
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }

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
            etag=meta.get("etag"),
            last_modified=meta.get("last_modified"),
            docs_version=docs_version_for_url(
                fetched_url, config.get("docs_version_pattern")
            ),
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
        config = sources[name]
        parts = []
        if config["pages"]:
            parts.append(f"{len(config['pages'])} pages")
        if config["crawl"]:
            parts.append(f"crawl {config['crawl']}")
        print(f"Fetching source '{name}' ({', '.join(parts) or 'empty'})")
        written, failed = fetch_source(name, config, max_age_days=max_age_days)
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
