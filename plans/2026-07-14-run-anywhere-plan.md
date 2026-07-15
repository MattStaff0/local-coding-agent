# Run-anywhere workflow plan

## Status: DONE 2026-07-14 — all items shipped or resolved

- Items 1, 2, 4, 6, 7 (console commands, --root, doctor, persistent PATH
  command, target-repo workflow): implemented.
- Item 3 (tests): `tests/test_run_anywhere.py` — doctor, --root parsing and
  validation, agent-root preset, and a subprocess test proving a foreign
  launch directory stays pristine.
- Items 5, 8 (index profiles, zsh alternative): documented in README
  ("Run it from anywhere").
- Found during verification: macOS (likely iCloud Desktop sync) sets the
  hidden flag on venv `.pth` files and Python ≥3.12 skips hidden `.pth`
  files, which silently broke the editable install's `lca`. The persistent
  commands are therefore launcher scripts in `~/.local/bin` that run
  `python src/ask.py` directly — no `.pth` involved. Documented in README.

## Original plan follows

## Current status

The core behavior already works:

- `src/paths.py` anchors the documentation index, Chroma database, manifests,
  and chat history to the installed repository instead of the current folder.
- `pip install -e .` creates the `lca` command, which can be launched from any
  working directory.
- `/agent root <path>` lets the live agent inspect a different repository and
  starts a fresh agent session for that root.
- `LCA_HOME` can relocate the complete local index and history when a separate
  index location is needed.

## Remaining improvements

1. Add dedicated console commands for `fetch_docs`, `ingest`, and
   `ingest_code`, so users do not need absolute paths to `src/*.py`.
2. Add an explicit `--root` option or environment variable for the agent,
   making the target repository selectable at startup.
3. Add tests that launch the installed `lca` command from a temporary working
   directory and verify that no local `chroma_db` or manifest is created there.
4. Add a short `lca doctor` command to report the active repository root,
   `.env` location, Ollama host, selected models, and index status.
5. Document separate index profiles if multiple repositories need independent
   code collections rather than one shared code collection.

6. Provide a persistent macOS command that behaves like `codex` or `claude`:
   install the project into a dedicated virtual environment, expose `lca` on
   the user's PATH, and make `lca` work from any current directory without
   changing the repository's anchored documentation index.
7. Support a clear target-repository workflow from any directory. The command
   should either use the current directory for `/agent` or accept an explicit
   repository path, with an equivalent to `/agent root <path>` documented and
   tested.
8. Document both a normal shell command and, if useful, a shell alias/function
   for macOS (`zsh`). Prefer an installed console script over an alias as the
   primary interface, because aliases do not reliably carry arguments or work
   in non-interactive shells.

## Recommended next implementation

Implement items 1, 3, 4, and 6 first. The final workflow should include:

- installation instructions for macOS using the repository's `.venv`;
- a persistent `lca` command available after opening a new terminal;
- an optional `lca --root /path/to/repo` or equivalent documented workflow;
- tests proving that launching from another directory does not create a stray
  Chroma database or manifest there;
- a doctor/status command showing the active project root, target repository,
  Ollama host, selected models, and index locations.
