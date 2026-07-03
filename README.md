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
rewrite follow-up questions into standalone queries using chat history
|
hybrid retrieval: vector similarity + BM25 keyword ranking, fused with RRF
|
refuse before answering if even the best chunk is past the relevance cutoff
|
build a prompt with numbered doc chunks + chat history + question
(the model must answer ONLY from the docs and cite chunks like [1])
|
ask the chat model through Ollama, streaming tokens as they arrive
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
|-- requirements.txt
`-- README.md
```

## Configuration (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `OLLAMA_CHAT_MODEL` | `qwen2.5-coder:3b` | Chat + agent model (switch 3B ↔ 12B without code edits) |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `RAG_RELEVANCE_CUTOFF` | `0.65` | Cosine distance past which retrieval refuses to answer |
| `STUDY_NOTES_DIR` | `study-notes` | Where `/export` writes study notes (point it at your vault) |

## What Each Python File Does

`src/rag.py` contains the main RAG system:

- configuration for docs, ChromaDB, and Ollama models (env-overridable)
- heading-section chunking with breadcrumbs
- batched embedding (`embed_batch`) via Ollama
- incremental indexing with per-file content hashes (`--full` rebuilds)
- hybrid retrieval: vector + BM25 fused with Reciprocal Rank Fusion
- follow-up query rewriting from chat history
- a relevance cutoff that refuses off-topic questions before the model runs
- prompt building and streamed chat model calls

`src/ingest.py` is the indexing command:

- incremental by default: embeds only changed/new files, drops deleted ones
- `python src/ingest.py --full` rebuilds the whole collection

`src/ask.py` is the terminal interface:

- starts an interactive chat if no question is passed
- streams answers token-by-token
- persists chat history across sessions (`chat_history.json`)
- prints a citation legend mapping each `[n]` in the answer to `file § heading`
- supports `/sources` and `/source <name>` to scope answers to one source
- `/agent <question>` searches this codebase live with the tool-calling agent
- `/export` saves the last answer + citations as a markdown study note

`src/agent.py` + `src/agent_tools.py` are the tool-calling agent:

- the model drives sandboxed `list_files` / `grep` / `read_file` tools in a
  loop with guardrails (iteration cap, repeated-call breaker, output caps)
- run it directly: `python src/agent.py "where is the similarity search?"`

`src/eval_retrieval.py` measures retrieval quality:

- runs the questions in `tests/golden.yaml` and prints hit-rate@1, hit-rate@k,
  and MRR, overall and per source — run it before and after retrieval changes

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

Scope answers to one documentation source (a top-level `docs/` folder):

```text
/sources          list indexed sources
/source pytorch   answer only from docs/pytorch/
/source all       search everything again
```

Exit chat:

```text
/exit
```

You can also ask one question directly:

```bash
python src/ask.py "How do I create a neural network in PyTorch?"
```

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

## Near-Term Next Steps

Good next improvements:

1. Add more official docs slowly (grow `sources.yaml`).
2. Add repo indexing for local source code (see the code-indexing options doc).
3. Connect MCP servers to the agent loop (see the MCP options doc in the vault).
4. Add a local cross-encoder reranker on top of hybrid retrieval.
5. Crawl mode for the scraper (fetch everything under a URL prefix, politely).
