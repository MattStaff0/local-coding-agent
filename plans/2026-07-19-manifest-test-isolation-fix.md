---
date: 2026-07-19
repo: C:\Users\mstaf\Desktop\local-ai-coding-agent
status: proposed (not started)
severity: high — `pytest` from the repo root corrupts the real docs index sidecar
---

# Plan: stop the test suite from clobbering the real `manifest.jsonl`

## Symptom (observed this session)

Running `pytest` from the repo root silently overwrote the real
`manifest.jsonl` from **1107 records (1,623,080 bytes)** down to **1 record
(269 bytes)**. Chroma (`chroma_db/`, 1107 embeddings) was untouched. The
manifest was rebuilt from Chroma via `rag.rebuild_manifest` and is currently
restored to 1107 records — but it will be clobbered again on the next bare
`pytest` run until this is fixed.

> **Why this matters for retrieval (the learning-project "why"):** the manifest
> is the sidecar BM25 reads at query time so retrieval never has to pull the
> whole corpus out of Chroma. A 1-record manifest doesn't crash anything — it
> silently guts the keyword half of hybrid retrieval (BM25 → RRF), so answers
> quietly get worse. Nothing warns you. That's the dangerous kind of bug.

## Root cause

- `rag.rebuild_manifest(collection, path=None)` (`src/rag.py:579-584`) defaults
  `path` to the module global `MANIFEST_PATH`, bound at import time from
  `paths.MANIFEST_PATH` (`src/rag.py:9`, `src/paths.py:44`) = `PROJECT_ROOT/manifest.jsonl`.
- `index_docs()` (`src/rag.py:387-489`) calls `rebuild_manifest(collection)` with
  **no path argument** at both the `full=True` return (line 429) and the
  incremental return (line 488).
- Any fixture that patches `rag.DB_DIR` (redirecting Chroma) but **not**
  `rag.MANIFEST_PATH`, then calls `index_docs`, writes the real repo-root
  manifest. `get_client()` reads `DB_DIR` fresh each call (`rag.py:266-268`),
  which is why Chroma was correctly isolated and the manifest was not.
- Three independent import-time bindings of "the same" path exist and must be
  patched separately: `paths.MANIFEST_PATH`, `rag.MANIFEST_PATH` (the one that
  matters for ingest/retrieve), and `docs_cli.MANIFEST_PATH` (`src/docs_cli.py:15`).
- There is **no `conftest.py`** anywhere in the repo and no suite-wide env
  isolation in `pytest.ini`. Isolation is per-fixture convention only, and six
  files got it wrong independently.

## Blast radius — leaking test files (patch `DB_DIR` only, call `index_docs`)

- `tests/test_hybrid.py:16-17` (`temp_db`) → calls at 49, 93
- `tests/test_rag.py:22-24` (`temp_db`) → calls at 53, 65, 79, 91, 102, 148
- `tests/test_incremental.py:16-17` (`temp_db`) → calls throughout
- `tests/test_failure_paths.py:18-19` (`temp_db`) → calls at 51, 71, 95, 109
- `tests/test_codex_fixes.py:24,34,77` (inline `DB_DIR` patch) → calls at 26, 35, 96
- `tests/test_review_fixes.py:19-20` (`temp_db`) → calls throughout
- `tests/test_threshold.py:18` (`temp_db`) — not currently calling `index_docs`,
  but shares the pattern; one keystroke from becoming a 7th leak.

**Already safe** (patch `rag.MANIFEST_PATH` or pass `path=` explicitly):
`test_manifest.py:107`, `test_docs_provenance.py:296`, `test_ingest_code.py`
(`code_env` patches `CODE_MANIFEST_PATH`), `test_rerank.py`, `test_code_retrieval.py`,
`test_docs_status.py` (patches `docs_cli.MANIFEST_PATH`).

## Recommended fix — autouse `tests/conftest.py` backstop (Option A)

Chosen over "fix each fixture" (Option B — that's the status quo that already
failed 6×, nothing stops a 7th) and "refactor `index_docs`/`rebuild_manifest`
to inject the path" (Option C — a production/retrieval-adjacent change that per
CLAUDE.md would require before/after `eval_retrieval.py` numbers + README, and
is disproportionate to a test-isolation gap). Option A is a narrow, precise
backstop that protects every current and future test with zero author effort,
and cannot launder unrelated path bugs because it only redirects the exact
globals implicated.

### Implementation steps

- [ ] Create `tests/conftest.py` with an `autouse=True`, function-scoped fixture
      that redirects, before every test:
      `rag.DB_DIR`, `rag.MANIFEST_PATH`, `rag.CODE_MANIFEST_PATH`,
      `paths.DB_DIR`, `paths.MANIFEST_PATH`, `docs_cli.MANIFEST_PATH`
      to `tmp_path`-based locations. (Do **not** touch `PROJECT_ROOT`,
      `HISTORY_FILE`, or unrelated globals — keep the blast radius minimal.)
- [ ] Keep `DB_DIR` as `str(tmp_path / "chroma_db")` and the manifests as `Path`
      objects, matching every existing fixture's convention (Windows-safe).
- [ ] Leave existing per-fixture `temp_db` patches in place (they compose safely
      with the autouse fixture via the shared function-scoped `monkeypatch`
      instance; LIFO teardown reverts both). Optional later cleanup: trim the now-
      redundant per-file `DB_DIR` patches — not required, separate change.

### `tests/conftest.py` sketch

```python
"""Keep every test off the real repo-root data files.

index_docs/index_code (src/rag.py:387-489) hand rebuild_manifest a bare
path=None, which falls back to the module-global MANIFEST_PATH /
CODE_MANIFEST_PATH (src/rag.py:579-584) — the real repo-root manifest.jsonl.
Several fixtures (test_hybrid, test_incremental, test_rag, test_review_fixes,
test_codex_fixes, test_failure_paths) redirect rag.DB_DIR for Chroma but never
the manifest path, so a plain `pytest` run overwrites the real manifest.jsonl.
This autouse fixture is the backstop: every test gets tmp_path-based data paths
before it runs, so forgetting a per-file patch can no longer touch real files.
"""
from pathlib import Path

import pytest

import docs_cli
import paths
import rag


@pytest.fixture(autouse=True)
def _isolate_data_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_dir = str(tmp_path / "chroma_db")
    manifest_path = tmp_path / "manifest.jsonl"
    code_manifest_path = tmp_path / "code-manifest.jsonl"

    monkeypatch.setattr(rag, "DB_DIR", db_dir)
    monkeypatch.setattr(rag, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(rag, "CODE_MANIFEST_PATH", code_manifest_path)

    # Independent import-time bindings of "the same" path — patch separately.
    monkeypatch.setattr(paths, "DB_DIR", db_dir)
    monkeypatch.setattr(paths, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(docs_cli, "MANIFEST_PATH", manifest_path)
```

## Verification (must not re-clobber the real manifest)

> **Correction to note:** `manifest.jsonl` is **gitignored** (`git check-ignore`
> confirms), so `git status` will NOT catch an mtime-only touch. Verify by
> byte-comparison against a copy.

- [ ] Record baseline: copy `manifest.jsonl` to the scratchpad and note size/mtime
      (currently 1,623,080 bytes; restored 1107 records).
- [ ] Add `tests/conftest.py`.
- [ ] Run the suite with an out-of-repo basetemp:
      `pytest -p no:cacheprovider --basetemp=<scratchpad>/pt`
- [ ] `Compare-Object` (or `fc`) the live `manifest.jsonl` against the saved copy
      — must be byte-identical. Also confirm no stray `manifest.jsonl` /
      `code-manifest.jsonl` / `chroma_db/` appeared under any `tmp_path`-adjacent
      repo location.
- [ ] Only after zero-diff is confirmed, run `pytest` normally to confirm the
      suite still passes (expect the pre-existing platform failures below to
      remain — they are separate).

## Out of scope / follow-ups

- The `.pytest_cache` and `AppData\...\Temp\pytest-of-mstaf` **PermissionError**
  seen earlier is an environment/ACL issue, not this bug. Workaround: pass
  `--basetemp` to a clean dir. Consider setting a repo-local `--basetemp` default.
- Platform-specific failures unrelated to isolation (POSIX symlink tests, `~`
  expansion, the `.env`-present "default models" test) are tracked with the
  Windows portability work in
  `plans/2026-07-19-attachments-windows-portability-fix.md`.

## Definition of done (per CLAUDE.md)

- [ ] `pytest` (with a clean basetemp) green apart from the separately-tracked
      platform tests; the real `manifest.jsonl` is provably byte-unchanged by a run.
- [ ] No retrieval logic changed → no `eval_retrieval.py` delta required, but note
      in the PR that this is a test-only change.
- [ ] README/CLAUDE.md unchanged (no command/env/flag change). Mention the new
      `tests/conftest.py` backstop in the PR description.
</content>
