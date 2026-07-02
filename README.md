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
split docs into chunks
|
embed each chunk with nomic-embed-text through Ollama
|
store chunks and embeddings in ChromaDB
|
embed the user's question
|
retrieve the most similar chunks
|
build a prompt with docs + chat history + question
|
ask qwen2.5-coder through Ollama
|
print the answer and retrieved sources
```

## Project Structure

```text
local-ai-coding-agent/
|-- docs/          # Markdown documentation to index
|-- src/
|   |-- rag.py        # Source of truth for RAG logic
|   |-- ingest.py     # Rebuilds the Chroma docs index
|   |-- ask.py        # Terminal chat interface
|   `-- fetch_docs.py # Fetches official docs into docs/<source>/
|-- tests/         # Pytest suite (no Ollama needed to run it)
|-- sources.yaml   # Registry of doc sources and URLs to fetch
|-- data/          # Local generated/raw data, not committed
|-- chroma_db/     # Local Chroma vector database, not committed
|-- requirements.txt
`-- README.md
```

## What Each Python File Does

`src/rag.py` contains the main RAG system:

- configuration for docs, ChromaDB, and Ollama models
- document chunking
- embedding text with `nomic-embed-text`
- rebuilding the Chroma collection
- retrieving relevant chunks for a question
- formatting temporary chat history
- building the final prompt
- calling the local chat model

`src/ingest.py` is the indexing command:

- reads markdown files from `docs/`
- calls `rag.index_docs()`
- rebuilds the local ChromaDB collection

`src/ask.py` is the terminal interface:

- starts an interactive chat if no question is passed
- supports one-shot questions from the command line
- keeps temporary memory while the chat process is open
- prints retrieved source chunks before each answer

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Install Ollama from:

```text
https://ollama.com/download
```

Pull the models:

```powershell
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

```powershell
python src\fetch_docs.py
python src\fetch_docs.py pytorch
```

Pages are converted from HTML to markdown (navigation, footers, and scripts are
stripped; the main article content is kept) and saved as
`docs/<source>/<page-slug>.md` with the original URL and fetch date in a small
frontmatter block. Re-run ingestion afterwards to index the new files.

## Add Docs

Put markdown files in `docs/`.

Example:

```text
docs/python-data-structures.md
docs/pytorch-basics.md
```

The current ingestion code uses `docs/**/*.md`, so docs can also be organized
into folders later:

```text
docs/python/data-structures.md
docs/pytorch/build-model.md
docs/system-design/caching.md
```

## Build the Index

Run ingestion after adding or editing docs:

```powershell
python src\ingest.py
```

This deletes and recreates the `local_docs` Chroma collection, so chunks are not
duplicated. It is safe to rerun whenever docs change.

## Ask Questions

Start interactive chat:

```powershell
python src\ask.py
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

```powershell
python src\ask.py "How do I create a neural network in PyTorch?"
```

## Current Memory Behavior

The chat has temporary session memory.

That means:

- it remembers previous turns while `python src\ask.py` is still running
- it uses recent chat history to understand follow-up questions
- it forgets the conversation when you exit
- it does not save chat history to disk yet

## When to Reingest

Run this again whenever docs change:

```powershell
python src\ingest.py
```

Reingest after:

- adding a new markdown file
- editing a markdown file
- deleting a markdown file
- changing chunk size
- changing embedding model

## Near-Term Next Steps

Good next improvements:

1. Add more official docs slowly.
2. Tighten the prompt for stricter doc-grounded answers.
3. Add better source citations in answers.
4. Add a docs scraper/converter for selected official pages.
5. Add repo indexing for local source code.
6. Add a configurable chat model, such as switching between 3B and 7B.
