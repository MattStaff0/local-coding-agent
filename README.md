# Local AI Coding Agent

This is a small local RAG-based coding assistant. It stores markdown docs locally,
indexes them with embeddings, retrieves relevant chunks for a question, and sends
those chunks to a local Ollama chat model.

The goal is to start simple and build toward a local coding agent that can read
docs, understand a repo, suggest patches, and eventually run safe commands.

## Current RAG Flow

```text
docs/*.md
|
split docs into chunks by markdown heading sections
(each chunk keeps its heading breadcrumb, e.g. "Tensors > Initializing a Tensor")
|
embed each file's chunks in one batched call (nomic-embed-text via Ollama)
|
store chunks and embeddings in ChromaDB (cosine space; per-file content hash)
|
rebuild manifest.jsonl with chunk ids, metadata, token counts, and BM25 tokens
|
rewrite follow-up questions into standalone queries using chat history
|
hybrid retrieval: vector similarity + manifest-backed BM25, fused with RRF
|
refuse before answering if even the best chunk is past the relevance cutoff
|
trim retrieved chunks to the prompt budget, preserving top-ranked chunks whole
|
build a prompt with numbered doc chunks + chat history + question
(the model must answer ONLY from the docs and cite chunks like [1])
|
ask the chat model through Ollama, streaming tokens as they arrive
|
warn if the answer has missing or out-of-range citations
|
print the answer, then a source legend ([n] -> file § heading)
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
|   |-- agent.py          # Tool-calling agent loop (/agent, CLI)
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

- starts an interactive chat if no question is passed
- streams answers token-by-token, live-rendered as markdown with
  syntax-highlighted code blocks on a real terminal (plain text when piped)
- prompt has up-arrow history and tab-completion for slash commands
  (`/help` lists them all)
- persists chat history across sessions (`chat_history.json`)
- prints a citation legend mapping each `[n]` in the answer to `file § heading`
- supports `/sources` and `/source <name>` to scope answers to one source
- `/agent <question>` searches this codebase live with the tool-calling agent
- `/export` saves the last answer + citations as a markdown study note

`src/agent.py` + `src/agent_tools.py` are the tool-calling agent:

- the model drives sandboxed `list_files` / `grep` / `read_file` tools in a
  loop with guardrails (iteration cap, repeated-call breaker, output caps)
- run it directly: `python src/agent.py "where is the similarity search?"`
- see the "Agent v2" section below for sessions, edits, and MCP

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

An editable install adds an `lca` console command that works from any
directory — data paths (docs, index, chat history) always resolve to this
project via `src/paths.py`, so running `lca` from another folder never
creates a stray `chroma_db/` there:

```bash
pip install -e .
lca                      # interactive chat from anywhere
lca "How do RNNs work?"  # one-shot question
```

Packaging note: the project installs the flat `src/*.py` modules top-level
(`rag`, `ask`, `ui`, …) rather than as a package — renaming to a package
would churn every import in the repo for zero behavior change.

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

On a real terminal the answer streams as live-rendered markdown (headings,
tables, syntax-highlighted code), the prompt supports up-arrow history and
tab-completion, and a spinner shows while retrieval runs. Piped output stays
plain text. `/help` lists all commands:

```text
/help             show this help
/sources          list indexed doc sources
/source <name>    answer only from one source (/source all to reset)
/agent <question> search this codebase live with tools
/export           save the last answer as a study note
/exit             quit
```

You can also ask one question directly:

```bash
python src/ask.py "How do I create a neural network in PyTorch?"
```

Answers are checked after generation for citation shape. If the model omits
`[n]` citations or cites a chunk number that was not sent in the prompt, the
CLI prints a grounding warning but does not rerun or reject the answer.

## Agent v2: Sessions, Edits, MCP

`/agent` conversations are now persistent within a chat session — follow-up
`/agent` questions remember earlier ones. Subcommands:

```text
/agent <question>     ask (continues the current session)
/agent                show current root and message count
/agent reset          clear the session
/agent root <path>    point the agent at another repo (starts a fresh session)
```

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
the first `/agent` call:

```json
{
  "servers": {
    "git":   {"command": "uvx", "args": ["mcp-server-git", "--repository", "."],
              "tools": ["git_status", "git_diff_unstaged", "git_log"]},
    "fetch": {"command": "uvx", "args": ["mcp-server-fetch"], "tools": ["fetch"]}
  }
}
```

To add a server: add an entry with its launch command and a `"tools"`
allowlist (tool names are namespaced `server_tool`). An optional
`"confirm": [...]` list routes specific tools through the y/N gate. Keep the
total loadout at 8 tools or fewer — small local models degrade sharply when
choosing between more, and the CLI warns when a config exceeds it.

## Code Indexing (`/code`)

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

Then in chat:

```text
/code where is the relevance cutoff applied?
```

Citations point at jumpable locations (`src/rag.py:478 § src/rag.py > retrieve`).
Re-run `ingest_code.py` after code changes — it is incremental (content
hashes), so unchanged files are never re-embedded. Score the code index with
`python src/eval_retrieval.py --code` (golden set: `tests/golden_code.yaml`).

Stretch experiment: compare `OLLAMA_EMBED_MODEL=nomic-embed-text` against a
code-tuned embedder (e.g. CodeRankEmbed) with `eval_retrieval.py --code` on
the PC — numbers decide, not vibes.

## Current Memory Behavior

The chat has persistent memory:

- it remembers previous turns and uses them to understand follow-up questions
  (follow-ups are rewritten into standalone queries before retrieval)
- history is saved to `chat_history.json` (not committed) and restored on the
  next run; only the most recent 100 messages are kept
- `/export` turns the last answer and its citations into a study note under
  `STUDY_NOTES_DIR`

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
