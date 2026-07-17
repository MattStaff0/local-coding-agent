"""lca docs status/sync: freshness and compatibility, offline (workstream 04)."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

import ask
import docs_cli


def write_registry(path: Path) -> Path:
    path.write_text(
        """
pandas:
  pages: [https://pandas.pydata.org/docs/user_guide/10min.html]
  distribution: pandas
  docs_version_pattern: 'pandas\\.pydata\\.org/pandas-docs/version/(?P<version>[\\d.]+)/'
  refresh_ttl_days: 7
""",
    )
    return path


def write_manifest(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


@pytest.fixture()
def env(tmp_path, monkeypatch):
    registry = write_registry(tmp_path / "sources.yaml")
    manifest = write_manifest(
        tmp_path / "manifest.jsonl",
        [
            {
                "id": "a-0",
                "source": "pandas",
                "relative_path": "pandas/10min.md",
                "tokens": ["x"],
                "docs_version": "3.0",
                "fetched": (date.today() - timedelta(days=30)).isoformat(),
            },
        ],
    )
    project = tmp_path / "project"
    project.mkdir()
    (project / "requirements.txt").write_text("pandas==2.2.1\n")

    monkeypatch.setattr(docs_cli, "SOURCES_FILE", registry)
    monkeypatch.setattr(docs_cli, "MANIFEST_PATH", manifest)
    return project


def test_status_reports_pages_age_ttl_and_compatibility(env, capsys):
    docs_cli.status(root=env)

    out = capsys.readouterr().out
    assert "pandas" in out
    assert "1 pages" in out
    assert "30d old" in out
    assert "stale" in out  # 30d > 7d ttl
    assert "2.2.1 (lock)" in out
    assert "mismatch" in out  # project 2.2 vs docs 3.0


def test_status_is_offline_and_never_touches_network(env, monkeypatch, capsys):
    import requests

    def boom(*args, **kwargs):
        raise AssertionError("docs status must not touch the network")

    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "head", boom)

    docs_cli.status(root=env)

    assert "pandas" in capsys.readouterr().out


def test_status_with_missing_manifest_says_not_built(env, monkeypatch, capsys):
    monkeypatch.setattr(docs_cli, "MANIFEST_PATH", Path("/nonexistent/manifest"))

    docs_cli.status(root=env)

    assert "not built" in capsys.readouterr().out


def test_main_docs_status_subcommand(env, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["lca", "docs", "status"])
    monkeypatch.setattr(ask, "canonical_root", lambda value: env)

    ask.main()

    assert "pandas" in capsys.readouterr().out


def test_main_docs_without_subcommand_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["lca", "docs"])

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 2
    assert "status" in capsys.readouterr().out


def test_main_docs_unknown_subcommand_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["lca", "docs", "destroy"])

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 2


def test_sync_calls_fetch_then_ingest(env, monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(
        docs_cli,
        "_fetch_source_by_name",
        lambda name, max_age_days: calls.append(("fetch", name)) or ([], []),
    )
    monkeypatch.setattr(
        docs_cli, "_reindex", lambda: calls.append(("ingest", None)) or 0
    )

    docs_cli.sync("pandas")

    assert calls == [("fetch", "pandas"), ("ingest", None)]


def test_sync_unknown_source_exits_2(env, capsys):
    with pytest.raises(SystemExit) as excinfo:
        docs_cli.sync("nope")

    assert excinfo.value.code == 2
    assert "nope" in capsys.readouterr().out
