# local-coding-agent

Local docs-grounded RAG assistant and coding agent on Ollama — and a learning project: the point is understanding retrieval, not just shipping it, so when you change retrieval behavior, explain the why briefly as you go. The README is thorough and actively maintained; trust it, and update it when behavior changes.

## Commands

- Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (Python ≥ 3.12; `pip install -e .` adds the `lca` command)
- Run: `python src/ask.py` (or `lca`)
- Test all: `pytest`
- Test one: `pytest tests/test_rag.py::test_name -x`
- Retrieval eval: `python src/eval_retrieval.py` (docs) / `python src/eval_retrieval.py --code` (code index)

## Definition of done

- `pytest` green — the suite must pass with no Ollama and no network; CI runs it on every push.
- Any retrieval change: run `eval_retrieval.py` before and after, and report both numbers. Numbers decide, not vibes.
- README updated whenever a command, env var, or flag changes.

## Environment

- Ollama runs on the PC, not this MacBook — reach it via `OLLAMA_HOST`. Tests always mock the Ollama client.
- Generated, never hand-edited: `chroma_db/`, `manifest.jsonl`, `chat_history.json`. Rebuild via `src/ingest.py`.
- `docs/<source>/` contents are fetched by `src/fetch_docs.py` from `sources.yaml` — regenerate fetched pages rather than editing them.

## Architecture — what the tree doesn't show

- `src/rag.py` is the source of truth for retrieval logic; `ask.py` is the UI, `ingest.py` is indexing. Start reading in `rag.py`.
- Hybrid retrieval = Chroma vector search + manifest-backed BM25, fused with RRF; the optional cross-encoder (`RAG_RERANKER`) rescores the fused pool.
- The top-level folder name under `docs/` IS the source name that `/source <name>` filters on in chat.

## Looks wrong, isn't

- Flat `src/*.py` modules installed top-level instead of a package — deliberate; packaging would churn every import for zero behavior change.
- The reranker is off by default so baseline eval numbers never move; a missing reranker model degrades to fusion order with a one-time warning, on purpose.
- Citation grounding checks warn but never reject or rerun the answer — warn-only is the design.
- `fetch_docs.py` uses a fixed polite delay between crawled pages, not backoff — courtesy, not a bug.

## Do not touch

- `tests/golden.yaml` and `tests/golden_code.yaml` are eval ground truth. Never edit them to make numbers pass — only to genuinely improve the benchmark, and say so when you do.
- Keep the agent's MCP tool loadout at 8 tools or fewer — small local models degrade sharply past that (the CLI warns).
