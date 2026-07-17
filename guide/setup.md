# Setup

A local, docs-grounded coding agent that runs entirely on your machines: a
small chat model served by [Ollama](https://ollama.com/download), retrieval
over official documentation you fetch yourself, and a sandboxed tool loop
over your project files. Nothing leaves your network.

## 1. Get the code and Python environment

Requires Python ≥ 3.12.

```bash
git clone <this repo> local-coding-agent
cd local-coding-agent
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .                   # installs the `lca` command
```

`pip install -e .` also installs `lca-fetch-docs`, `lca-ingest`, and
`lca-ingest-code`. The project supports both venv layouts
(`.venv/lib/python*/site-packages` and `.venv/Lib/site-packages`).

## 2. Point it at Ollama

Ollama can run on the same machine (default) or elsewhere on your LAN:

```bash
cp .env.example .env
# then edit .env:
#   OLLAMA_HOST=http://<ollama-host>:11434    # omit if Ollama is local
#   OLLAMA_CHAT_MODEL=qwen2.5-coder:3b        # any tool-calling model
#   OLLAMA_EMBED_MODEL=nomic-embed-text
```

Pull the models on the Ollama machine:

```bash
ollama pull qwen2.5-coder:3b
ollama pull nomic-embed-text
```

The chat model must support **tool calling** — check the model's page on
the Ollama library. Bigger models (7B–14B) noticeably improve the agent.

## 3. Fetch and index documentation

```bash
lca-fetch-docs            # downloads the sources.yaml corpus into docs/
lca-ingest                # embeds it into the local Chroma index
```

Add libraries by editing `sources.yaml` (see the registry comments in that
file for the version-aware options), then `lca docs sync <source>`. To
**remove** a source, deleting the registry entry isn't enough — sync never
deletes cached pages — so also delete `docs/<source>/` and re-run
`lca-ingest`, which drops chunks for files that no longer exist.

## 4. Verify

```bash
lca doctor                # offline health check: paths, models, indexes
pytest                    # full suite; needs no Ollama and no network
```

`lca doctor` is the first thing to run whenever anything misbehaves — see
[troubleshooting.md](troubleshooting.md).

## Run it from any directory

Install once, then `cd` into any project and run `lca` — the docs index and
chat history stay anchored to this repository (or to `LCA_HOME` if set),
while the agent reads the project you launched from. If your platform's
editable-install path breaks (some sync tools hide `.pth` files), create
thin launchers instead — a two-line shell script that runs
`<repo>/.venv/bin/python <repo>/src/ask.py "$@"` from anywhere works on any
Unix; on Windows the console scripts in `.venv\Scripts` serve the same role.
