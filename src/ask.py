import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import chromadb.errors
import httpx

import attachments as attachments_module
import docs_cli
import manifest as manifest_module
import mcp_client
import paths
import rag
import rerank
import ui
import agent as agent_module
from agent import AgentSession, parse_agent_command, run_agent
from rag import (
    EmptyIndexError,
    NoRelevantDocsError,
    list_sources,
    source_legend,
)

NO_INDEX_HINT = "No index found. Run 'python src/ingest.py' first."

# Chat history persists across restarts; only the recent tail is saved so the
# file cannot grow without bound (the prompt is capped separately by
# rag.MAX_HISTORY_TURNS).
HISTORY_FILE = paths.HISTORY_FILE
MAX_SAVED_MESSAGES = 100

# Where /export writes study notes. Point this at the Obsidian vault, e.g.
# STUDY_NOTES_DIR="$HOME/Documents/matt-vault/study-notes".
EXPORT_DIR = Path(os.getenv("STUDY_NOTES_DIR", paths.PROJECT_ROOT / "study-notes"))


@dataclass(frozen=True)
class AgentTurn:
    question: str
    answer: str
    trace: list[str]
    doc_sources: list[str]


def run_agent_turn(
    question: str,
    *,
    session: AgentSession,
    renderer,
    confirm,
    mcp,
    cli_contexts: tuple[str, ...] = (),
) -> AgentTurn:
    """Run and render one agent turn for every free-form CLI entry path.

    Attachments (@path in the prompt, --context flags) resolve here, before
    the model call; an AttachmentError propagates to the caller and nothing
    reaches the model.
    """
    prepared = attachments_module.prepare_turn(session.root, question, cli_contexts)
    for label in prepared.labels:
        renderer.show_message(f"Attached: {label}")

    message_start = len(session.messages)
    answer, trace = run_agent(
        prepared.question,
        session=session,
        confirm=confirm,
        mcp=mcp,
        on_token=renderer.on_token,
    )
    renderer.finish_answer()

    if trace:
        renderer.show_message(
            "Tool calls:\n" + "\n".join(f"  {entry}" for entry in trace)
        )

    doc_sources: list[str] = []
    for message in session.messages[message_start:]:
        if message.get("role") != "tool" or message.get("tool_name") != "search_docs":
            continue
        for line in str(message.get("content", "")).splitlines():
            if re.match(r"^\[\d+\] ", line) and line not in doc_sources:
                doc_sources.append(line)

    if doc_sources:
        renderer.show_sources(doc_sources)

    return AgentTurn(question, answer, trace, doc_sources)


def load_history(
    path: Path = HISTORY_FILE, root: Path | None = None
) -> list[dict[str, str]]:
    """Load saved chat history; only a missing file is a silent fresh start.

    An unreadable file is preserved as a .bak before the next save can
    overwrite it — chat history should never be destroyed silently.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as error:
        print(f"Could not read {path} ({error}) — starting with empty history.")
        return []

    try:
        history = json.loads(raw)
    except json.JSONDecodeError:
        history = None

    if isinstance(history, list):
        if root is None or root == paths.PROJECT_ROOT.expanduser().resolve():
            return history
        return []

    if (
        isinstance(history, dict)
        and history.get("version") == 2
        and isinstance(history.get("sessions"), dict)
    ):
        if root is None:
            return []
        messages = history["sessions"].get(str(root), [])
        return messages if isinstance(messages, list) else []

    backup = path.with_suffix(path.suffix + ".bak")
    path.rename(backup)
    print(f"{path} is unreadable — backed up to {backup}, starting fresh.")
    return []


def save_history(
    history: list[dict[str, str]],
    path: Path = HISTORY_FILE,
    root: Path | None = None,
) -> None:
    """Persist the most recent chat messages to disk."""
    if root is not None:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            existing = None
        except json.JSONDecodeError:
            backup = path.with_suffix(path.suffix + ".bak")
            path.rename(backup)
            existing = None
        except OSError:
            existing = None

        if isinstance(existing, list):
            sessions = {
                str(paths.PROJECT_ROOT.expanduser().resolve()): existing[
                    -MAX_SAVED_MESSAGES:
                ]
            }
        elif (
            isinstance(existing, dict)
            and existing.get("version") == 2
            and isinstance(existing.get("sessions"), dict)
        ):
            sessions = dict(existing["sessions"])
        else:
            sessions = {}

        sessions[str(root)] = history[-MAX_SAVED_MESSAGES:]
        path.write_text(
            json.dumps({"version": 2, "sessions": sessions}, indent=2),
            encoding="utf-8",
        )
        return

    path.write_text(
        json.dumps(history[-MAX_SAVED_MESSAGES:], indent=2), encoding="utf-8"
    )


def clean_history(messages: list[Any]) -> list[dict[str, str]]:
    """Project an agent transcript into safe persisted user/final-answer turns."""
    clean: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            clean.append({"role": "user", "content": str(message.get("content", ""))})
        elif role == "assistant" and not message.get("tool_calls"):
            clean.append(
                {"role": "assistant", "content": str(message.get("content", ""))}
            )
    return clean


def export_note(
    question: str,
    answer: str,
    metadatas: list[dict[str, Any] | str],
    notes_dir: Path = EXPORT_DIR,
) -> Path:
    """Write one answered question as a markdown study note.

    Every good answer can become a vault note — that's the learning flywheel.
    """
    notes_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60] or "note"
    stem = f"{date.today().isoformat()}-{slug}"

    path = notes_dir / f"{stem}.md"
    counter = 2
    while path.exists():
        path = notes_dir / f"{stem}-{counter}.md"
        counter += 1

    lines = [
        "---",
        f"date: {date.today().isoformat()}",
        "kind: study-note",
        "---",
        "",
        f"# {question}",
        "",
        answer.strip(),
        "",
    ]

    if metadatas:
        lines.append("## Sources")
        source_lines = (
            list(metadatas)
            if isinstance(metadatas[0], str)
            else source_legend(metadatas)
        )
        lines.extend(f"- {line}" for line in source_lines)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

    return path


def describe_error(error: Exception) -> str:
    """Turn a pipeline failure into an actionable message.

    Expected failures (no index yet, Ollama unreachable) get one-line hints;
    anything else is a bug, so keep the full traceback visible.
    """
    if isinstance(error, chromadb.errors.NotFoundError):
        return NO_INDEX_HINT

    if isinstance(error, (EmptyIndexError, NoRelevantDocsError)):
        return str(error)

    if isinstance(error, (httpx.TransportError, ConnectionError)):
        return f"Could not reach the local models ({error}). Is Ollama running?"

    return f"Unexpected error ({type(error).__name__}):\n{traceback.format_exc()}"


def print_token(token: str) -> None:
    """Print one streamed answer token immediately, without a newline."""
    print(token, end="", flush=True)


def safe_list_sources() -> list[str]:
    """Indexed source names for tab-completion; never crash the prompt."""
    try:
        return list_sources()
    except chromadb.errors.NotFoundError:
        return []  # no index yet — nothing to complete, not an error
    except Exception as error:
        print(f"(source completion disabled: {type(error).__name__}: {error})")
        return []


def start_mcp(root: Path | None = None):
    """Start MCP servers from mcp.json, once per chat process.

    Returns None when nothing is configured or startup fails — the agent
    then runs with native tools only, which is a degradation, not an error.
    """
    manager = None
    try:
        config = mcp_client.load_config(paths.PROJECT_ROOT / "mcp.json")

        if not config.get("servers"):
            return None

        manager = mcp_client.MCPManager(
            config, root=canonical_root(root if root is not None else Path.cwd())
        )
        manager.start()
    except Exception as error:
        if manager is not None:
            manager.stop()
        # A typo in mcp.json must degrade to native tools, not kill the chat.
        print(f"(mcp unavailable: {type(error).__name__}: {error})")
        return None

    return manager


HELP_TEXT = """\
Every free-form prompt uses the agent over current saved files and docs.
Writes and commands always require confirmation.
Attach saved files (not editor buffers): @path, @path:80, @path:80-130
  in any prompt, or lca --context <path[:a-b]> "question" one-shot.
Commands:
  /help             show this help
  /status           show root, session size, and docs scope
  /root [path]      show or change root (a change starts fresh)
  /reset            clear this root's session
  /sources          list indexed doc sources
  /source <name>    constrain docs search (/source all to reset)
  /export           save the last answer as a study note
  /exit             quit
Compatibility aliases (deprecated for removal after one release):
  /agent <question> same as asking directly
  /code <question>  same as asking directly"""


def apply_source_command(
    line: str,
    active_source: str | None,
) -> tuple[bool, str | None, str]:
    """Handle /sources and /source commands.

    Returns (handled, new_active_source, message to print).
    """
    stripped = line.strip()

    if stripped == "/sources":
        try:
            names = ", ".join(list_sources())
        except chromadb.errors.NotFoundError:
            return True, active_source, NO_INDEX_HINT
        return True, active_source, f"Available sources: {names}"

    if stripped == "/source" or stripped.startswith("/source "):
        parts = stripped.split(maxsplit=1)

        if len(parts) == 1:
            current = active_source or "all"
            return True, active_source, f"Current source: {current}"

        name = parts[1].strip()

        if name == "all":
            return True, None, "Searching all sources."

        try:
            available = list_sources()
        except chromadb.errors.NotFoundError:
            return True, active_source, NO_INDEX_HINT
        if name not in available:
            return (
                True,
                active_source,
                f"Unknown source '{name}'. Available: {', '.join(available)}."
                + (f" Still scoped to '{active_source}'." if active_source else ""),
            )

        return True, name, f"Now answering only from '{name}' docs."

    return False, active_source, ""


def canonical_root(value: str | Path) -> Path:
    """Expand, validate, and canonicalize one agent root."""
    supplied = str(value)
    candidate = Path(value).expanduser()
    try:
        if not candidate.is_dir():
            detail = "not a directory" if candidate.exists() else "no such directory"
            raise ValueError(f"{supplied}: {detail}")
        return candidate.resolve()
    except OSError as error:
        raise ValueError(f"{supplied}: cannot access ({error})") from error


def chat_loop(renderer=None, read_input=None, agent_root: Path | None = None) -> None:
    """Run an interactive terminal chat with persistent memory."""
    renderer = renderer or ui.make_renderer()
    read_input = read_input or ui.make_input(safe_list_sources)

    # Pass the module globals explicitly: default arguments bind at def time,
    # which would ignore a HISTORY_FILE override (tests, future config).
    root = canonical_root(agent_root if agent_root is not None else Path.cwd())
    history = load_history(HISTORY_FILE, root=root)
    last_export: tuple[str, str, list[dict[str, Any] | str]] | None = None
    agent_session = AgentSession(root=root, messages=list(history))
    mcp_manager = None
    mcp_started = False
    deprecated_aliases: set[str] = set()

    renderer.show_message("Local coding agent")
    renderer.show_message("Type your question, /help for commands, /exit to quit.")
    renderer.show_message(f"Agent root: {agent_session.root}")

    if history:
        renderer.show_message(f"(restored {len(history)} messages from {HISTORY_FILE})")

    while True:
        prompt_text = (
            f"[{agent_session.docs_source}] You: "
            if agent_session.docs_source
            else "You: "
        )
        try:
            question = read_input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt, StopIteration):
            if mcp_manager is not None:
                mcp_manager.stop()
            renderer.show_message("Goodbye.")
            return

        # Ignore blank lines so accidental Enter presses do not call the model.
        if not question:
            continue

        def confirm(description: str, preview: str) -> bool:
            renderer.show_message(f"\n{description}")
            if preview != description:
                renderer.show_message(preview)
            return read_input("Apply? [y/N]: ").strip().lower() in {"y", "yes"}

        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            if mcp_manager is not None:
                mcp_manager.stop()
            renderer.show_message("Goodbye.")
            return

        compatibility_question = False
        agent_command = parse_agent_command(question)
        if agent_command is not None:
            if "/agent" not in deprecated_aliases:
                renderer.show_message(
                    "(deprecated: /agent is no longer needed; ask directly)"
                )
                deprecated_aliases.add("/agent")
            subcommand, argument = agent_command
            if subcommand == "status":
                question = "/status"
            elif subcommand == "reset":
                question = "/reset"
            elif subcommand == "root":
                if not argument:
                    renderer.show_error("Usage: /agent root <path>")
                    continue
                question = f"/root {argument}"
            else:
                question = argument
                compatibility_question = True

        if not compatibility_question and (
            question == "/code" or question.startswith("/code ")
        ):
            if "/code" not in deprecated_aliases:
                renderer.show_message(
                    "(deprecated: /code now uses the live agent; ask directly)"
                )
                deprecated_aliases.add("/code")
            code_question = question.removeprefix("/code").strip()
            if not code_question:
                renderer.show_message(
                    "Usage: /code <question> (deprecated; ask directly instead)."
                )
                continue
            question = code_question
            compatibility_question = True

        if not compatibility_question and question == "/help":
            renderer.show_message(HELP_TEXT)
            continue

        if not compatibility_question and question == "/status":
            mcp_status = (
                "not started"
                if not mcp_started
                else ("loaded" if mcp_manager is not None else "native only")
            )
            renderer.show_message(
                f"Agent root: {agent_session.root} "
                f"({len(agent_session.messages)} messages); "
                f"source: {agent_session.docs_source or 'all'}; "
                f"MCP: {mcp_status}; prompt: {agent_module.PROMPT_REVISION}; "
                "mutations: confirmation required"
            )
            continue

        if not compatibility_question and question == "/reset":
            agent_session = AgentSession(root=agent_session.root)
            last_export = None
            try:
                save_history([], HISTORY_FILE, root=agent_session.root)
            except OSError as error:
                renderer.show_message(f"(could not save chat history: {error})")
            renderer.show_message("Agent session cleared.")
            continue

        if not compatibility_question and (
            question == "/root" or question.startswith("/root ")
        ):
            argument = question.removeprefix("/root").strip()
            if not argument:
                renderer.show_message(f"Agent root: {agent_session.root}")
                continue
            try:
                new_root = canonical_root(argument)
            except ValueError as error:
                detail = str(error)
                if "no such directory" in detail:
                    renderer.show_error(f"No such directory: {argument}")
                else:
                    renderer.show_error(detail)
                continue
            if mcp_manager is not None:
                mcp_manager.stop()
            mcp_manager = None
            mcp_started = False
            agent_session = AgentSession(root=new_root)
            last_export = None
            try:
                save_history([], HISTORY_FILE, root=new_root)
            except OSError as error:
                renderer.show_message(f"(could not save chat history: {error})")
            renderer.show_message(f"Agent root set to {new_root} (fresh session).")
            continue

        if not compatibility_question and question == "/export":
            if last_export is None:
                renderer.show_message("Nothing to export yet — ask a question first.")
                continue

            try:
                path = export_note(*last_export, notes_dir=EXPORT_DIR)
            except OSError as error:
                # A bad STUDY_NOTES_DIR must not kill the whole chat session.
                renderer.show_error(f"Could not write the study note ({error}).")
                continue

            renderer.show_message(f"Saved study note: {path}")
            continue

        handled, active_source, message = (
            (False, agent_session.docs_source, "")
            if compatibility_question
            else apply_source_command(question, agent_session.docs_source)
        )
        if handled:
            agent_session.docs_source = active_source
            renderer.show_message(message)
            continue

        if not mcp_started:
            mcp_manager = start_mcp(agent_session.root)
            mcp_started = True

        try:
            with renderer.status("thinking…"):
                turn = run_agent_turn(
                    question,
                    session=agent_session,
                    renderer=renderer,
                    confirm=confirm,
                    mcp=mcp_manager,
                )
        except (EOFError, KeyboardInterrupt):
            if mcp_manager is not None:
                mcp_manager.stop()
            renderer.show_message("Goodbye.")
            return
        except attachments_module.AttachmentError as error:
            # User-fixable: bad path, denied file, invalid range. The model
            # was never called; the session is untouched.
            renderer.show_error(str(error))
            continue
        except Exception as error:
            renderer.show_error(describe_error(error))
            continue

        last_export = (question, turn.answer, turn.doc_sources)

        try:
            save_history(
                clean_history(agent_session.messages),
                HISTORY_FILE,
                root=agent_session.root,
            )
        except OSError as error:
            # Losing persistence is worth a warning, not a dead session.
            renderer.show_message(f"(could not save chat history: {error})")


def _index_summary(manifest_path: Path) -> str:
    """One line describing an index from its manifest — no network, no Chroma.

    Never raises: a torn or unreadable manifest is one of the broken states
    doctor exists to diagnose, so it must be reported, not crash the report.
    """
    try:
        manifest_path.stat()
    except FileNotFoundError:
        return "not built"
    except OSError as error:
        # exists() would hide symlink loops and permission errors as False,
        # turning a broken path into a misleading "not built".
        return f"unreadable ({error}) — rebuild with lca-ingest"
    try:
        records = manifest_module.load_manifest(manifest_path)
    except (json.JSONDecodeError, OSError) as error:
        return f"unreadable ({error}) — rebuild with lca-ingest"
    if not records:
        return "empty (0 chunks) — rebuild with lca-ingest"
    sources = {
        record.get("source", "?") if isinstance(record, dict) else "?"
        for record in records
    }
    return f"{len(records)} chunks across {len(sources)} sources"


def doctor() -> None:
    """Report where everything lives and how the models are configured.

    Never touches the network or opens Chroma, so it is safe to run when
    Ollama is down — that is exactly when you need it.
    """
    env_state = "found" if paths.ENV_FILE.exists() else "not found"
    host = os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434 (default)"
    reranker = f"on ({rerank.RERANK_MODEL})" if rerank.enabled() else "off"

    print("lca doctor")
    print(f"  project home:  {paths.PROJECT_ROOT}")
    print(f"  .env file:     {paths.ENV_FILE} ({env_state})")
    print(f"  OLLAMA_HOST:   {host}")
    print(f"  chat model:    {rag.CHAT_MODEL}")
    print(f"  prompt rev:    {agent_module.PROMPT_REVISION} "
          f"(style: {os.getenv('LCA_TEACHING_STYLE', 'coach')})")
    print(f"  embed model:   {rag.EMBED_MODEL}")
    print(f"  reranker:      {reranker}")
    print(f"  docs index:    {_index_summary(paths.MANIFEST_PATH)}")
    print(f"  code index:    {_index_summary(rag.CODE_MANIFEST_PATH)}")
    print(
        "  agent root:    current directory, canonicalized at start"
        " (override: lca --root <path>, or /root <path> in chat)"
    )


def main() -> None:
    """Use chat mode by default, or answer a single command-line question."""
    args = sys.argv[1:]

    if args and args[0] == "doctor":
        if args[1:]:
            # Refusing beats silently discarding the rest of a one-shot
            # question that happened to start with the word "doctor".
            print('doctor takes no arguments; quote one-shot questions: lca "..."')
            raise SystemExit(2)
        doctor()
        return

    if args and args[0] == "docs":
        subcommand = args[1] if len(args) > 1 else None
        if subcommand == "status" and len(args) == 2:
            docs_cli.status(root=canonical_root(Path.cwd()))
            return
        if subcommand == "sync" and len(args) <= 3:
            docs_cli.sync(args[2] if len(args) == 3 else None)
            return
        print("Usage: lca docs status | lca docs sync [source]")
        raise SystemExit(2)

    selected_root: Path | None = None
    contexts: list[str] = []
    while args and args[0] in {"--root", "--context"}:
        flag = args[0]
        if len(args) < 2:
            print(f"Usage: lca {flag} <path>")
            raise SystemExit(2)
        if flag == "--root":
            try:
                selected_root = canonical_root(args[1])
            except ValueError as error:
                print(f"--root {error}")
                raise SystemExit(2)
        else:
            contexts.append(args[1])
        args = args[2:]

    if not args:
        if contexts:
            # --context attaches to a one-shot question; chat turns use @path.
            print('--context needs a question: lca --context <path> "question"')
            raise SystemExit(2)
        chat_loop(agent_root=selected_root)
        return

    if args[0].startswith("-"):
        # Anything else would be silently sent to the model as a "question".
        print(f'Unknown option: {args[0]} — usage: lca | lca "question" | '
              "lca doctor | lca --root <path> | lca --context <path> \"question\"")
        raise SystemExit(2)

    # This keeps the old one-shot usage:
    # python src/ask.py "How do I make a PyTorch model?"
    question = " ".join(args)
    root = selected_root or canonical_root(Path.cwd())

    renderer = ui.make_renderer()
    mcp_manager = start_mcp(root)

    try:
        run_agent_turn(
            question,
            session=AgentSession(root=root),
            renderer=renderer,
            confirm=None,
            mcp=mcp_manager,
            cli_contexts=tuple(contexts),
        )
    except attachments_module.AttachmentError as error:
        print(str(error))
        raise SystemExit(2)
    except Exception as error:
        renderer.show_error(describe_error(error))
        raise SystemExit(1)
    finally:
        if mcp_manager is not None:
            mcp_manager.stop()


if __name__ == "__main__":
    main()
