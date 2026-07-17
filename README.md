# Local AI Coding Agent

This is a small local coding assistant. Every free-form prompt enters one
directory-scoped tool-calling agent, which can read current saved project files,
search locally indexed documentation, propose confirmed edits, and run a narrow
set of confirmed checks through a local Ollama model.

The goal is to start simple and build toward a local coding agent that can read
docs, understand a repo, suggest patches, and eventually run safe commands.

## Current unified flow

```text
plain prompt or compatibility alias
|
deterministic preflight: canonical root, slash commands, docs scope,
history, MCP lifecycle, and confirmation channel
|
persistent AgentSession + local Ollama tool loop
|-- list_files / grep / read_file -> current saved project evidence
|-- search_docs -> hybrid docs retrieval (vector + manifest BM25 + RRF)
|                  -> relevance refusal or numbered file § heading passages
|-- edit_file / write_file / run_command -> preview + y/N confirmation
`-- allowlisted MCP tools -> optional confirmation from mcp.json
|
stream final answer through Rich/plain renderer
|
show compact tool trace + docs labels; persist clean root-scoped history
```

## Project Structure

```text
local-ai-coding-agent/
|-- docs/          # Markdown documentation to index
|-- src/
|   |-- rag.py            # Source of truth for RAG logic
|   |-- ingest.py         # Updates the Chroma docs index (incremental)
|   |-- ask.py            # Terminal chat interface
|   |-- fetch_docs.py     # Fetches official docs into docs/<source>/
|   |-- agent.py          # Tool-calling loop used by every free-form prompt
|   |-- agent_tools.py    # Sandboxed list/grep/read tools for the agent
|   `-- eval_retrieval.py # Scores retrieval against tests/golden.yaml
|-- tests/         # Pytest suite (no Ollama needed to run it)
|-- .github/       # CI: pytest on every push and pull request
|-- sources.yaml   # Registry of doc sources and URLs to fetch
|-- data/          # Local generated/raw data, not committed
|-- chroma_db/     # Local Chroma vector database, not committed
|-- manifest.jsonl # Generated lexical sidecar, not committed
|-- requirements.txt
`-- README.md
```

## Configuration (environment variables)

All of these can live in a `.env` file at the repo root — copy `.env.example`
to `.env` and edit. It is loaded automatically on startup (before the Ollama
client is created, so `OLLAMA_HOST` works from it), it is gitignored, and a
variable already exported in your shell always beats the file.

| Variable | Default | Effect |
|---|---|---|
| `OLLAMA_CHAT_MODEL` | `qwen2.5-coder:3b` | Chat + agent model (switch 3B ↔ 12B without code edits) |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `RAG_RELEVANCE_CUTOFF` | `0.65` | Cosine distance past which retrieval refuses to answer |
| `RAG_PROMPT_BUDGET` | `12000` | Character budget for retrieved documentation context in the prompt |
| `RAG_MANIFEST_PATH` | `manifest.jsonl` | Path to the generated JSONL chunk manifest used for BM25 retrieval |
| `STUDY_NOTES_DIR` | `study-notes` | Where `/export` writes study notes (point it at your vault) |
| `LCA_HOME` | project directory | Overrides where the docs, index, and chat history live (`src/paths.py`) |
| `RAG_RERANKER` | off | Set to `cross-encoder` to rerank the hybrid pool with a local cross-encoder (needs `pip install sentence-transformers`) |
| `RAG_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Which cross-encoder the reranker loads |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Point at a remote Ollama, e.g. the PC from the MacBook: `OLLAMA_HOST=http://192.168.x.x:11434` |

## What Each Python File Does

`src/rag.py` contains the main RAG system:

- configuration for docs, ChromaDB, and Ollama models (env-overridable)
- heading-section chunking with breadcrumbs
- batched embedding (`embed_batch`) via Ollama
- incremental indexing with per-file content hashes (`--full` rebuilds)
- manifest rebuilds after ingest; `manifest.jsonl` is generated and not committed
- hybrid retrieval: vector + manifest-backed BM25 fused with Reciprocal Rank Fusion
- follow-up query rewriting from chat history
- a relevance cutoff that refuses off-topic questions before the model runs
- prompt-context budgeting that trims lower-ranked chunks from the tail
- warn-only grounding checks for missing or out-of-range `[n]` citations
- prompt building and streamed chat model calls

`src/ingest.py` is the indexing command:

- incremental by default: embeds only changed/new files, drops deleted ones
- `python src/ingest.py --full` rebuilds the whole collection

`src/ask.py` is the terminal interface:

- starts one persistent agent session at the canonical current directory if no
  question is passed
- sends both plain and one-shot questions through the same agent loop
- streams agent answers token-by-token, live-rendered as markdown with
  syntax-highlighted code blocks on a real terminal (plain text when piped)
- prompt has up-arrow history and tab-completion for slash commands
  (`/help` lists them all)
- persists clean user/final-answer history separately for each canonical root
  in `chat_history.json` (tool payloads are not persisted)
- prints returned documentation labels such as `[1] file § heading`; project
  evidence stays jumpable as `path:line` in the answer
- supports `/sources` and `/source <name>` as a constraint on agent docs search
- `/root`, `/reset`, and `/status` manage the active rooted session
- `/agent` and `/code` are temporary deprecated aliases for asking directly
- `/export` saves the last answer + citations as a markdown study note
- `lca doctor` prints project home, `.env` state, Ollama host, models, and
  index status without touching the network — run it first when things break

`src/agent.py` + `src/agent_tools.py` are the tool-calling agent:

- the model drives sandboxed `list_files` / `grep` / `read_file` tools in a
  loop with guardrails (iteration cap, repeated-call breaker, output caps)
- run it directly: `python src/agent.py "where is the similarity search?"`
- see "Unified always-agent sessions" below for sessions, edits, and MCP

`src/mcp_client.py` is the hand-rolled MCP client:

- reads `mcp.json`, connects each server over stdio on a background asyncio
  thread, and exposes synchronous `owns()` / `call()` to the agent loop

`src/eval_retrieval.py` measures retrieval quality:

- runs the questions in `tests/golden.yaml` and prints hit-rate@1, hit-rate@k,
  and MRR, overall and per source
- scores negative golden questions with an additional `refusal` line when
  entries use `expect: refusal`
- run it before and after retrieval changes

`src/fetch_docs.py` is the docs downloader:

- reads `sources.yaml` (source name -> list of doc URLs)
- probes for native markdown first (the `llms.txt` convention: `<page>.md`
  next to each HTML page) and uses it verbatim when the site serves it
- otherwise downloads the page and converts HTML to markdown (URLs ending in
  `.md` or `.txt` are saved as-is, no conversion)
- writes `docs/<source>/<page-slug>.md` with the URL and fetch date on top
- `--refresh 30d` re-downloads only pages older than the given age
- skips failed pages, then summarizes failures and exits nonzero if any

## Setup

Create and activate a virtual environment.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Install the `lca` command

An editable install adds `lca`, `lca-fetch-docs`, `lca-ingest`, and
`lca-ingest-code` console commands that work from any directory — data paths
(docs, index, chat history) always resolve to this project via
`src/paths.py`, so running `lca` from another folder never creates a stray
`chroma_db/` there:

```bash
pip install -e .
lca                      # interactive chat from anywhere (venv active)
lca "Where is validation computed?"  # one-shot rooted at the current directory
lca doctor               # where everything lives + model config, no network
lca --root ~/school/proj # interactive agent pointed at another repo
lca --root ~/school/proj "Why does train.py fail?" # rooted one-shot
```

Packaging note: the project installs the flat `src/*.py` modules top-level
(`rag`, `ask`, `ui`, …) rather than as a package — renaming to a package
would churn every import in the repo for zero behavior change.

### Run it from anywhere (macOS, like `claude` or `codex`)

The venv commands above only exist while the venv is activated. For a
persistent command in every new terminal, install thin launchers into
`~/.local/bin` (one-time):

```bash
cd ~/Desktop/local-coding-agent
[ -x .venv/bin/python ] || { echo "run this from the repo root (no .venv here)"; exit 1; }
mkdir -p ~/.local/bin
for pair in "lca:ask" "lca-fetch-docs:fetch_docs" "lca-ingest:ingest" "lca-ingest-code:ingest_code"; do
  cmd="${pair%%:*}"; mod="${pair##*:}"
  printf '#!/bin/sh\nexec "%s/.venv/bin/python" "%s/src/%s.py" "$@"\n' "$PWD" "$PWD" "$mod" > ~/.local/bin/"$cmd"
  chmod +x ~/.local/bin/"$cmd"
done
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
```

Open a new terminal and `lca` works from any directory. `lca doctor` is the
health check: project home, `.env`, Ollama host, models, and index status.

Why launcher scripts instead of symlinking the venv console scripts: on this
Mac something (likely iCloud Desktop sync) sets the macOS *hidden* file flag
on `.pth` files inside the venv, and Python ≥ 3.12 silently skips hidden
`.pth` files — which breaks the editable install's import path and makes
`lca` die with `ModuleNotFoundError: No module named 'ask'`. The launchers
run `python src/ask.py` directly, which needs no `.pth` at all. If the venv
`lca` ever shows that error, that's the cause (`chflags nohidden` fixes it
until the flag comes back).

Heads-up: in a shell with the venv activated, the venv's own `lca` console
script shadows these launchers (`.venv/bin` lands first on PATH). If `lca`
fails with `ModuleNotFoundError` only in venv shells, that's why —
`deactivate` (or `hash -r`) and check `which lca`.

Working on two repos with two independent doc/code indexes? Set `LCA_HOME`
to a second data directory in that shell (or a second launcher) — all
generated state (docs, chroma_db, manifests, history) follows it.

### How run-anywhere actually works

Three pieces make `lca` behave the same from any directory:

1. **The launcher pins the interpreter.** Each script in `~/.local/bin` is
   two lines: `exec <repo>/.venv/bin/python <repo>/src/ask.py "$@"`. The
   repo's venv and code are baked in as absolute paths, so it doesn't matter
   what shell, venv, or directory you're in when you type `lca`.
2. **`src/paths.py` pins the data.** Every module gets its file locations
   from `paths.py`, which anchors them to the *repo's* location (it derives
   the repo root from its own file path, not from `os.getcwd()`). So the
   docs, Chroma DB, manifests, `.env`, and chat history always live in the
   project folder — launching from `~/school/proj` reads and writes the same
   index as launching from the repo. The current directory is never used for
   state, which is why the test suite can assert a foreign launch directory
   stays byte-for-byte empty.
3. **The agent follows you.** The one thing that *should* depend on where you
   are is which code the live tools inspect. That defaults to your canonical
   current directory, and `lca --root <path>` (or `/root <path>` mid-chat)
   points it somewhere else. Changing root starts a fresh session, clears the
   docs scope/export state, and restarts root-sensitive MCP servers so context
   from one repo never leaks into another.

So: documentation knowledge is global-and-shared (one index, wherever you
are), while conversation and live file context are scoped by canonical project
root. `lca doctor` prints both halves so you can see exactly what a terminal
will use.

### When to use this vs the Claude Code harness on a local model

The tempting alternative: keep the local model but borrow the grown-up
harness — run Claude Code pointed at Ollama through an Anthropic-API
translation proxy (e.g. LiteLLM) via `ANTHROPIC_BASE_URL`. It works, but
it's a mismatch, because a harness is *tuned to a model class*:

- **Prompt weight.** Claude Code front-loads a system prompt plus 15+ tool
  schemas — tens of thousands of tokens before your question. Small models
  degrade sharply past ~8 tools (the reason this repo caps its MCP loadout),
  and at local tok/s you pay that prompt tax every single turn.
- **Loop assumptions.** Claude Code's long multi-step turns, subagents, and
  recovery behavior assume the model can plan and self-correct. A 14B model
  inside that loop tends to thrash: wrong tool, malformed call, retry, drift.
- **This harness is shaped for small models on purpose.** Tiny prompt, ≤8
  tools, retrieval doing the heavy lifting so the model mostly has to read
  the right chunk and phrase an answer — that's the regime where a small
  model is actually good.

So the working split: **frontier model → Claude Code harness; local model →
a small-model harness like this one.** Mixing them (big harness, small
model) mostly buys friction. Where `lca` earns its keep regardless: it's
free and private (nothing leaves the LAN), it answers from the exact doc
versions you ingested with citations instead of training-data recall, and
it's a glass box you can take apart — which is the point of this project.

A larger local model (e.g. a 14B coder on the PC) raises this harness's
ceiling and is worth trying — set `OLLAMA_CHAT_MODEL` and check tool-calling
still works — the architecture stays the same, the answers get better.

Install Ollama from:

```text
https://ollama.com/download
```

Pull the models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:3b
```

## Fetch Official Docs

`sources.yaml` maps a source name to the doc pages to download. Each source
becomes a folder under `docs/`:

```yaml
pytorch:
  - https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
python:
  - https://docs.python.org/3/tutorial/datastructures.html
```

A source can also crawl a doc section instead of hand-listing every page:
point `crawl` at the section's index page and its links under that prefix
are fetched too (depth-1, capped, with a polite delay between downloads):

```yaml
sklearn:
  crawl: https://scikit-learn.org/stable/modules/
  max_pages: 30     # cap, index page included (default 30)
  delay: 1.0        # seconds between downloads (default 1.0)
  pages: []         # optional extra hand-picked URLs
```

Fetch everything in the registry (or just one source by name):

```bash
python src/fetch_docs.py
python src/fetch_docs.py pytorch
```

Pages are converted from HTML to markdown (navigation, footers, and scripts are
stripped; the main article content is kept), and URLs that already point at
`.md` or `.txt` files are saved as-is with no conversion. Each page lands at
`docs/<source>/<page-slug>.md` with the original URL and fetch date in a small
frontmatter block. Re-run ingestion afterwards to index the new files.

## Add Docs

The top-level folder under `docs/` is the doc's **source name** — it is what
`/source <name>` filters on in chat, so organize docs into per-source folders:

```text
docs/python/data-structures.md   -> source "python"
docs/pytorch/build-model.md      -> source "pytorch"
docs/system-design/caching.md    -> source "system-design"
```

Markdown files placed directly in `docs/` still work; they are indexed under
the source name `general`.

## Build the Index

Run ingestion after adding or editing docs:

```bash
python src/ingest.py          # incremental: only changed/new/removed files
python src/ingest.py --full   # rebuild the whole collection from scratch
```

Incremental runs compare a stored fingerprint (file content + embedding
model) per file, so unchanged docs are never re-embedded — and switching
`OLLAMA_EMBED_MODEL` automatically re-embeds everything. An index built by an
older version of this project is detected and rebuilt automatically. It is
safe to rerun whenever docs change.

Every successful ingest also rebuilds `manifest.jsonl`, a generated JSONL
sidecar with one record per chunk: id, relative path, source, heading, file
hash, approximate token count, and pre-tokenized lexical terms. Retrieval uses
that manifest for BM25 so each question does not pull the full corpus out of
Chroma. If the manifest is missing, retrieval falls back to the old slow
per-query BM25 path and prints a notice; rerun ingestion to regenerate it.

## Ask Questions

Start interactive chat:

```bash
python src/ask.py
```

Example questions:

```text
How do I create a neural network in PyTorch?
What does the forward method do?
How do Python lists work?
```

On a real terminal the agent answer streams as live-rendered markdown
(headings, tables, syntax-highlighted code), the prompt supports up-arrow
history and tab-completion, and a spinner shows while the agent thinks. Piped
output stays plain text. Every free-form line uses the same live-file/docs agent;
slash commands remain deterministic. `/help` lists all commands:

```text
/help             show this help
/status           show root, session size, docs scope, and MCP state
/root [path]      show or change root (a change starts fresh)
/reset            clear this root's session
/sources          list indexed doc sources
/source <name>    constrain docs search (/source all to reset)
/export           save the last answer as a study note
/exit             quit
/agent <question> deprecated alias for asking directly
/code <question>  deprecated alias for asking directly
```

You can also ask one question directly:

```bash
python src/ask.py "How do I create a neural network in PyTorch?"
```

Documentation tool results retain numbered path/heading labels, and the agent
prompt requires those full labels for library claims. File claims use `path:line`
from current reads. The legacy docs-only RAG answer function still has its
warn-only citation checker, but it is no longer a user-facing free-form route.

## Unified always-agent sessions

Run `lca` in a project and ask normally. Follow-ups use the same `AgentSession`;
you never need to enter an agent mode. Session commands are deterministic:

```text
/status          show canonical root, message count, docs scope, MCP state,
                 and the confirmation policy
/reset           clear messages, docs scope, and export state; keep the root
/root            show the canonical root
/root <path>     change root and start fresh
```

The agent consults ingested library documentation mid-loop with
`search_docs(query, source?)` — the same hybrid docs retriever as before, so
"why does this line of my code misuse pandas" can pull the pandas docs while
reading your file. Failures (index not built, Ollama down) come back to the
model as text, never crash the loop, and show as ERROR in the trace.

`/source pandas` does not switch modes or force retrieval. It constrains any
later `search_docs` call to pandas even if the model omits or requests another
source; `/source all` removes the constraint.

The agent can also propose file edits (`edit_file`, `write_file`) and run
allowlisted commands (`run_command`: only `pytest` and `python`). Every
write or run shows a unified diff (or the command line) and asks
`Apply? [y/N]` first — declining is an answer the model sees, not an error:

```text
edit_file src/rag.py
--- a/src/rag.py
+++ b/src/rag.py
@@ ...
Apply? [y/N]:
```

### MCP servers (`mcp.json`)

External tools come from MCP servers declared in `mcp.json` at the project
root (requires [`uvx`](https://docs.astral.sh/uv/)); servers start lazily on
the first free-form interactive turn, or immediately for a one-shot:

```json
{
  "servers": {
    "git":   {"command": "uvx", "args": ["mcp-server-git", "--repository", "."],
              "tools": ["git_status", "git_diff_unstaged", "git_log"]},
    "fetch": {"command": "uvx", "args": ["mcp-server-fetch"], "tools": ["fetch"]}
  }
}
```

Each stdio server starts with the canonical agent root as its working directory.
Changing `/root` stops the old manager and lazily starts a new one; exit, EOF,
and interrupt stop it as well. Missing/malformed config or unavailable servers
degrade to the seven native tools instead of killing the session.

To add a server: add an entry with its launch command and a `"tools"`
allowlist (tool names are namespaced `server_tool`). An optional
`"confirm": [...]` list routes specific tools through the y/N gate. Keep the
total loadout at 8 tools or fewer — small local models degrade sharply when
choosing between more, and the CLI warns when a config exceeds it. The seven
built-in tools already use most of that budget, so allowlist MCP tools one
at a time.

## Optional code index (ingest/evaluation subsystem)

Python repos index into a second Chroma collection (`local_code`) with an
ast-based chunker: one chunk per top-level function/class (decorators and
docstrings kept, oversized classes split per method), plus a module chunk
for imports and constants. Docs and code share the same hybrid retrieval
machinery but separate collections and manifests, so docs retrieval numbers
are untouched.

```bash
python src/ingest_code.py .                                # index this repo
python src/ingest_code.py ~/school/dl-project --name dl-project
```

Citations point at jumpable locations (`src/rag.py:478 § src/rag.py > retrieve`).
Re-run `ingest_code.py` after code changes — it is incremental (content
hashes), so unchanged files are never re-embedded. Score the code index with
`python src/eval_retrieval.py --code` (golden set: `tests/golden_code.yaml`).

The unified CLI does not automatically use this semantic index: live
`list_files`/`grep`/`read_file` tools are the current project path. `/code` is
kept for one release only as a compatibility alias and now enters that same
live agent path. Workstream 06 will measure routing before any semantic project
search is added to the default loop.

Stretch experiment: compare `OLLAMA_EMBED_MODEL=nomic-embed-text` against a
code-tuned embedder (e.g. CodeRankEmbed) with `eval_retrieval.py --code` on
the PC — numbers decide, not vibes.

## Current memory and export behavior

The chat has persistent memory:

- one `AgentSession` remembers previous user, assistant, and tool turns in the
  current process
- `chat_history.json` stores a versioned map by canonical root; only clean
  user/final-answer messages are restored, capped at 100 per root
- `/reset` clears the active root's saved session; `/root <path>` always starts
  fresh rather than restoring an older target-root conversation
- `/export` turns the last successful project, docs, mixed, or alias answer and
  its returned docs labels into a study note under `STUDY_NOTES_DIR`
- one-shot questions do not modify interactive history

Compatibility note: `/agent` and `/code` print one deprecation hint per process
and otherwise use exactly the same session, streaming, source scope, MCP tools,
history, export, and confirmation gates as a plain question. Both aliases are
scheduled for removal after one documented release.

Explicit `@path` attachments, line ranges, notebooks, and `--context` are not
implemented yet; those interfaces belong to the next workstream. Terminal mode
currently sees saved files through the live root-sandboxed tools.

## When to Reingest

Run this again whenever docs change:

```bash
python src/ingest.py
```

Reingest after:

- adding a new markdown file
- editing a markdown file
- deleting a markdown file
- changing chunk size
- changing embedding model

## Optional: Cross-Encoder Reranking

Hybrid retrieval is tuned for recall; an optional local cross-encoder adds
precision by rescoring the whole fused candidate pool before the final
top-k cut. It is off by default so baseline behavior and eval numbers never
change:

```bash
pip install sentence-transformers   # not in requirements.txt (pulls torch)
RAG_RERANKER=cross-encoder python src/ask.py "How do DataLoaders batch?"
RAG_RERANKER=cross-encoder python src/eval_retrieval.py   # prove it helps
```

If the model is missing the reranker warns once and keeps the fusion order —
it degrades precision, never availability.

## Near-Term Next Steps

Good next improvements:

1. Add more official docs slowly (grow `sources.yaml`).
2. Compare a code-tuned embedder (CodeRankEmbed) vs nomic-embed-text with
   `eval_retrieval.py --code`.
3. Measure the reranker on the PC (`RAG_RERANKER=cross-encoder` before/after).
