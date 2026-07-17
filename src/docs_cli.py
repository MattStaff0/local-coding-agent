"""`lca docs status` / `lca docs sync` — freshness vs compatibility at a glance.

status is offline by design (like doctor): manifest + registry + project
files only, so it works exactly when the network doesn't. sync is the one
explicit network action; the agent never refreshes docs on its own.
"""
from datetime import date
from pathlib import Path

import manifest as manifest_module
import paths
import project_versions
from fetch_docs import SOURCES_FILE, fetch_source, load_sources

MANIFEST_PATH = paths.MANIFEST_PATH


def _load_registry() -> dict:
    try:
        return load_sources(SOURCES_FILE)
    except (OSError, ValueError) as error:
        print(f"(could not read {SOURCES_FILE}: {error})")
        return {}


def _age_days(fetched: str) -> int | None:
    try:
        return (date.today() - date.fromisoformat(fetched)).days
    except ValueError:
        return None


def status(root: Path) -> None:
    """Per source: corpus size, fetch age vs TTL, and version compatibility."""
    registry = _load_registry()

    if not MANIFEST_PATH.exists():
        print(f"docs index: not built (run lca-ingest); registry has "
              f"{len(registry)} sources configured")
        return

    records = manifest_module.load_manifest(MANIFEST_PATH)
    by_source: dict[str, list[dict]] = {}
    for record in records:
        if isinstance(record, dict):
            by_source.setdefault(record.get("source", "?"), []).append(record)

    distributions = [
        config["distribution"] for config in registry.values()
    ]
    detected = project_versions.detect_versions(root, distributions)

    print(f"lca docs status (root: {root})")
    for name, config in registry.items():
        chunks = by_source.get(name, [])
        pages = {record.get("relative_path") for record in chunks}
        ttl = config["refresh_ttl_days"]

        ages = [
            age
            for record in chunks
            if record.get("fetched")
            and (age := _age_days(record["fetched"])) is not None
        ]
        if ages:
            oldest = max(ages)
            freshness = f"oldest fetch {oldest}d old (ttl {ttl}d) — " + (
                "stale" if oldest > ttl else "fresh"
            )
        else:
            freshness = "fetch age unknown"

        project = detected.get(config["distribution"])
        project_label = (
            f"{project.version} ({project.confidence})"
            if project and project.version
            else "unknown"
        )

        versions = sorted(
            {record.get("docs_version") for record in chunks}
            - {None}
        )
        compat_parts = [
            f"docs v{docs_version} — "
            + project_versions.compatibility(
                project.version if project else None,
                docs_version,
                config["version_policy"],
            )
            for docs_version in versions
        ] or ["docs version unknown"]

        print(f"  {name}: {len(pages)} pages, {len(chunks)} chunks; {freshness}")
        print(
            f"    project {config['distribution']} {project_label}; "
            + "; ".join(compat_parts)
        )

    unregistered = sorted(set(by_source) - set(registry))
    if unregistered:
        print(f"  (indexed but not in registry: {', '.join(unregistered)})")


def _fetch_source_by_name(name: str, max_age_days: int | None):
    registry = _load_registry()
    return fetch_source(name, registry[name], max_age_days=max_age_days)


def _reindex() -> int:
    import rag

    return rag.index_docs()


def sync(source_name: str | None, max_age_days: int | None = 0) -> None:
    """Bounded explicit refresh of one source (or all), then re-index."""
    registry = _load_registry()

    if source_name is not None and source_name not in registry:
        print(f"Unknown source '{source_name}'. "
              f"Available: {', '.join(registry) or 'none'}")
        raise SystemExit(2)

    names = [source_name] if source_name else list(registry)
    all_failed: list[str] = []
    for name in names:
        print(f"Syncing '{name}'…")
        _, failed = _fetch_source_by_name(name, max_age_days)
        all_failed.extend(failed)

    _reindex()

    if all_failed:
        # Failed pages kept their last-good copies; say so and exit nonzero.
        print(f"WARNING: {len(all_failed)} pages failed (last-good copies kept)")
        raise SystemExit(1)
