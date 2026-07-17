# Troubleshooting, migration, and rollback

## First move, always

```bash
lca doctor
```

Doctor is offline by design — it works exactly when the network doesn't.
It reports the project home, `.env` state, Ollama host, models, prompt
revision, and both index states, and it never crashes on a broken index
(a torn manifest is reported as `unreadable`, not a traceback).

## Common symptoms

| Symptom | Likely cause | Fix |
|---|---|---|
| `Could not reach the local models` | Ollama down / wrong host | check `OLLAMA_HOST` in `.env`; `ollama serve` on the host |
| `No index found` / `not built` | never ingested | `lca-ingest` (docs), `lca-ingest-code <repo>` (code) |
| docs index `empty (0 chunks)` | ingest ran against empty docs/ | `lca-fetch-docs` first, then `lca-ingest` |
| docs answers feel stale | TTL exceeded | `lca docs status`, then `lca docs sync <source>` |
| version-mismatch warnings | your project pins an older lib | that's the feature — read the warning; fetch versioned docs if needed |
| `ModuleNotFoundError: No module named 'ask'` from `lca` | editable-install `.pth` hidden by a file-sync tool | use launcher scripts (see setup.md) or reinstall `pip install -e .` |
| attachment refused | file is denied (secret/binary) or outside root | that's the sandbox; move the content or quote the relevant lines |
| answers cite nothing | model too small / docs not indexed | check `lca docs status`; try a larger `OLLAMA_CHAT_MODEL` |

## Migration from older checkouts

- **Indexes:** the ingest fingerprint includes a metadata schema version;
  after upgrading, the next `lca-ingest` automatically re-embeds everything
  once so old chunks gain docs-version provenance. To force it:
  `python src/ingest.py --full`.
- **Chat history:** `chat_history.json` upgrades in place to the per-root
  v2 format; an unreadable file is backed up to `.bak`, never destroyed.
- **sources.yaml:** old list-form sources keep working; the v2 mapping form
  adds version metadata per source.
- **Generated state is disposable:** `chroma_db/`, `manifest.jsonl`, and
  code manifests can be deleted and rebuilt at any time with the ingest
  commands. Never hand-edit them.

## Offline behavior

Everything except `lca docs sync` and `lca-fetch-docs` works offline
(given a reachable Ollama). Offline, answers come from the last good docs
cache and are labeled with their fetch age; a failed refresh never deletes
the previous copy of a page.

## Rollback

Every feature landed as a single merge commit, so rolling back is:

```bash
git log --oneline --merges     # pick the last-good merge
git checkout <merge-sha>       # or: git revert -m 1 <bad-merge-sha>
python src/ingest.py --full    # rebuild indexes under that version
```

Docs cache files are plain markdown and safe across versions; launchers
don't embed a version at all (they exec the repo's current code).

## Known limitations

- **Saved files only.** The terminal cannot see unsaved editor buffers;
  there is no editor adapter yet.
- **Configured corpus, not the web.** Docs answers come from the official
  pages you fetched — nothing else. If a library isn't in `sources.yaml`,
  the agent should say so rather than guess.
- **Local-model ceiling.** A 3B model misroutes tools and misses nuance a
  frontier model would catch; the eval harnesses
  (`src/eval_retrieval.py`, `src/eval_learning.py`, `src/eval_routing.py`)
  exist to measure exactly how much, per model, before you trust it more.
- **Prompt injection** in fetched docs or project files is bounded by the
  confirmation gates and origin enforcement rather than detected outright.
