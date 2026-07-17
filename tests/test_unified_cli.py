"""Every free-form CLI prompt shares one rooted agent-turn adapter."""
from pathlib import Path

import agent
import ask
import ui


class RecordingRenderer(ui.PlainRenderer):
    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.messages: list[str] = []
        self.sources: list[list[str]] = []
        self.finished = 0
        self.errors: list[str] = []

    def on_token(self, token: str) -> None:
        self.tokens.append(token)

    def finish_answer(self) -> None:
        self.finished += 1

    def show_message(self, text: str) -> None:
        self.messages.append(text)

    def show_sources(self, legend_lines: list[str]) -> None:
        self.sources.append(legend_lines)

    def show_error(self, text: str) -> None:
        self.errors.append(text)


def run_chat(monkeypatch, lines, renderer, root: Path):
    inputs = iter([*lines, "/exit"])
    monkeypatch.setattr(ask, "load_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(ask, "save_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(ask, "start_mcp", lambda *args, **kwargs: None)
    ask.chat_loop(
        renderer=renderer,
        read_input=lambda prompt: next(inputs),
        agent_root=root,
    )


def test_run_agent_turn_streams_once_and_renders_compact_trace(
    monkeypatch, tmp_path: Path
) -> None:
    renderer = RecordingRenderer()
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "app.py"},
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "The answer."}},
        ]
    )
    (tmp_path / "app.py").write_text("answer = 42\n", encoding="utf-8")
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        lambda **kwargs: iter([next(responses)]) if kwargs.get("stream") else next(responses),
    )

    turn = ask.run_agent_turn(
        "what is the answer?",
        session=agent.AgentSession(root=tmp_path),
        renderer=renderer,
        confirm=None,
        mcp=None,
    )

    assert turn.answer == "The answer."
    assert renderer.finished == 1
    assert "The answer." not in "\n".join(renderer.messages)
    trace_text = "\n".join(renderer.messages)
    assert "Tool calls:" in trace_text and "read_file" in trace_text
    assert "answer = 42" not in trace_text


def test_run_agent_turn_collects_only_returned_docs_labels(
    monkeypatch, tmp_path: Path
) -> None:
    renderer = RecordingRenderer()
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_docs",
                                "arguments": {"query": "concat"},
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "Use concat [1]."}},
        ]
    )
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        lambda **kwargs: iter([next(responses)]) if kwargs.get("stream") else next(responses),
    )
    monkeypatch.setattr(
        agent,
        "search_docs",
        lambda query, source=None, root=None: (
            "[1] docs/pandas/merge.md § Concat\npassage\n\n"
            "[1] docs/pandas/merge.md § Concat\npassage"
        ),
    )

    turn = ask.run_agent_turn(
        "how does concat work?",
        session=agent.AgentSession(root=tmp_path),
        renderer=renderer,
        confirm=None,
        mcp=None,
    )

    assert turn.doc_sources == ["[1] docs/pandas/merge.md § Concat"]
    assert renderer.sources == [["[1] docs/pandas/merge.md § Concat"]]


def test_chat_creates_agent_session_at_canonical_root(
    monkeypatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    link = tmp_path / "project-link"
    link.symlink_to(project, target_is_directory=True)
    seen = []

    def fake_turn(question, *, session, **kwargs):
        seen.append(session.root)
        session.messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "ok"},
            ]
        )
        return ask.AgentTurn(question, "ok", [], [])

    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    run_chat(monkeypatch, ["question"], RecordingRenderer(), link)

    assert seen == [project.resolve()]


def test_plain_project_question_routes_to_agent(monkeypatch, tmp_path: Path) -> None:
    seen = []

    def fake_turn(question, *, session, **kwargs):
        seen.append((question, session))
        session.messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "project answer"},
            ]
        )
        return ask.AgentTurn(question, "project answer", ["read_file({})"], [])

    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    run_chat(monkeypatch, ["where is retrieve?"], RecordingRenderer(), tmp_path)

    assert seen[0][0] == "where is retrieve?"
    assert seen[0][1].root == tmp_path.resolve()


def test_plain_docs_question_routes_to_same_agent_session(
    monkeypatch, tmp_path: Path
) -> None:
    sessions = []

    def fake_turn(question, *, session, **kwargs):
        sessions.append(session)
        session.messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "answer"},
            ]
        )
        return ask.AgentTurn(
            question, "answer", ["search_docs({})"], ["[1] docs/pandas/a.md"]
        )

    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    run_chat(
        monkeypatch,
        ["What is concat?", "and axis?"],
        RecordingRenderer(),
        tmp_path,
    )

    assert sessions[0] is sessions[1]


def test_reset_keeps_root_and_clears_messages_and_source(
    monkeypatch, tmp_path: Path
) -> None:
    seen = []

    def fake_turn(question, *, session, **kwargs):
        seen.append((session, list(session.messages), session.docs_source))
        session.messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "ok"},
            ]
        )
        return ask.AgentTurn(question, "ok", [], [])

    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    monkeypatch.setattr(ask, "list_sources", lambda: ["pandas"])
    run_chat(
        monkeypatch,
        ["/source pandas", "first", "/reset", "second"],
        RecordingRenderer(),
        tmp_path,
    )

    assert seen[0][2] == "pandas"
    assert seen[1][0].root == tmp_path.resolve()
    assert seen[1][1] == []
    assert seen[1][2] is None


def test_root_command_canonicalizes_and_starts_fresh_session(
    monkeypatch, tmp_path: Path
) -> None:
    original = tmp_path / "one"
    replacement = tmp_path / "two"
    original.mkdir()
    replacement.mkdir()
    link = tmp_path / "two-link"
    link.symlink_to(replacement, target_is_directory=True)
    seen = []

    def fake_turn(question, *, session, **kwargs):
        seen.append((session, list(session.messages)))
        session.messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "ok"},
            ]
        )
        return ask.AgentTurn(question, "ok", [], [])

    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    run_chat(
        monkeypatch,
        ["first", f"/root {link}", "second"],
        RecordingRenderer(),
        original,
    )

    assert seen[0][0] is not seen[1][0]
    assert seen[1][0].root == replacement.resolve()
    assert seen[1][1] == []


def test_invalid_root_keeps_existing_session(monkeypatch, tmp_path: Path) -> None:
    renderer = RecordingRenderer()
    seen = []

    def fake_turn(question, *, session, **kwargs):
        seen.append(session)
        return ask.AgentTurn(question, "ok", [], [])

    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    run_chat(
        monkeypatch,
        [f"/root {tmp_path / 'missing'}", "question"],
        renderer,
        tmp_path,
    )

    assert seen[0].root == tmp_path.resolve()
    assert any("No such directory" in error for error in renderer.errors)


def test_main_root_and_question_select_one_shot_agent(
    monkeypatch, tmp_path: Path
) -> None:
    import sys

    seen = {}
    renderer = RecordingRenderer()

    def fake_turn(question, *, session, **kwargs):
        seen.update(question=question, root=session.root)
        return ask.AgentTurn(question, "ok", [], [])

    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    monkeypatch.setattr(ask, "start_mcp", lambda *args: None)
    monkeypatch.setattr(ask.ui, "make_renderer", lambda: renderer)
    monkeypatch.setattr(
        sys, "argv", ["lca", "--root", str(tmp_path), "one-shot question"]
    )

    ask.main()

    assert seen == {"question": "one-shot question", "root": tmp_path.resolve()}


def test_agent_and_code_aliases_warn_once_and_use_unified_turn(
    monkeypatch, tmp_path: Path
) -> None:
    renderer = RecordingRenderer()
    questions = []

    def fake_turn(question, *, session, **kwargs):
        questions.append(question)
        return ask.AgentTurn(question, "ok", [], [])

    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    run_chat(
        monkeypatch,
        [
            "/agent first",
            "/agent second",
            "/code third",
            "/code fourth",
        ],
        renderer,
        tmp_path,
    )

    assert questions == ["first", "second", "third", "fourth"]
    output = "\n".join(renderer.messages)
    assert output.count("deprecated: /agent") == 1
    assert output.count("deprecated: /code") == 1


def test_alias_turns_share_plain_session(monkeypatch, tmp_path: Path) -> None:
    sessions = []

    def fake_turn(question, *, session, **kwargs):
        sessions.append(session)
        session.messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "ok"},
            ]
        )
        return ask.AgentTurn(question, "ok", [], [])

    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    run_chat(
        monkeypatch,
        ["plain", "/agent alias", "/code code alias"],
        RecordingRenderer(),
        tmp_path,
    )

    assert sessions[0] is sessions[1] is sessions[2]


def test_bare_agent_alias_maps_to_status(monkeypatch, tmp_path: Path) -> None:
    renderer = RecordingRenderer()
    monkeypatch.setattr(
        ask,
        "run_agent_turn",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called")),
    )

    run_chat(monkeypatch, ["/agent"], renderer, tmp_path)

    output = "\n".join(renderer.messages)
    assert "deprecated: /agent" in output
    assert f"Agent root: {tmp_path.resolve()}" in output


def test_agent_reset_and_root_aliases_map_to_operational_commands(
    monkeypatch, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    seen = []

    def fake_turn(question, *, session, **kwargs):
        seen.append((question, session.root, list(session.messages)))
        session.messages.append({"role": "assistant", "content": "old"})
        return ask.AgentTurn(question, "ok", [], [])

    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    run_chat(
        monkeypatch,
        ["first", "/agent reset", "second", f"/agent root {other}", "third"],
        RecordingRenderer(),
        tmp_path,
    )

    assert seen[1][2] == []
    assert seen[2][1] == other.resolve()
    assert seen[2][2] == []


def test_bare_code_alias_shows_deprecated_usage_without_model_call(
    monkeypatch, tmp_path: Path
) -> None:
    renderer = RecordingRenderer()
    monkeypatch.setattr(
        ask,
        "run_agent_turn",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called")),
    )

    run_chat(monkeypatch, ["/code"], renderer, tmp_path)

    output = "\n".join(renderer.messages)
    assert "deprecated: /code" in output
    assert "ask directly" in output


def test_alias_question_that_looks_like_command_reaches_agent(
    monkeypatch, tmp_path: Path
) -> None:
    questions = []

    def fake_turn(question, **kwargs):
        questions.append(question)
        return ask.AgentTurn(question, "ok", [], [])

    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    run_chat(
        monkeypatch,
        ["/agent /status", "/code /reset"],
        RecordingRenderer(),
        tmp_path,
    )

    assert questions == ["/status", "/reset"]


def test_unified_turn_persists_clean_root_scoped_history(
    monkeypatch, tmp_path: Path
) -> None:
    history_path = tmp_path / "history.json"

    def fake_turn(question, *, session, **kwargs):
        session.messages.extend(
            [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "read_file"}}],
                },
                {"role": "tool", "tool_name": "read_file", "content": "payload"},
                {"role": "assistant", "content": "final"},
            ]
        )
        return ask.AgentTurn(question, "final", ["read_file({})"], [])

    monkeypatch.setattr(ask, "HISTORY_FILE", history_path)
    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    monkeypatch.setattr(ask, "start_mcp", lambda *args: None)
    inputs = iter(["question", "/exit"])

    ask.chat_loop(
        renderer=RecordingRenderer(),
        read_input=lambda prompt: next(inputs),
        agent_root=tmp_path,
    )

    assert ask.load_history(history_path, root=tmp_path.resolve()) == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "final"},
    ]
    assert "payload" not in history_path.read_text(encoding="utf-8")


def test_reset_persists_empty_current_root_session(monkeypatch, tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    ask.save_history(
        [{"role": "user", "content": "old"}],
        history_path,
        root=tmp_path.resolve(),
    )
    monkeypatch.setattr(ask, "HISTORY_FILE", history_path)
    monkeypatch.setattr(ask, "start_mcp", lambda *args: None)
    inputs = iter(["/reset", "/exit"])

    ask.chat_loop(
        renderer=RecordingRenderer(),
        read_input=lambda prompt: next(inputs),
        agent_root=tmp_path,
    )

    assert ask.load_history(history_path, root=tmp_path.resolve()) == []


def test_root_change_does_not_restore_prior_target_history(
    monkeypatch, tmp_path: Path
) -> None:
    original = tmp_path / "one"
    target = tmp_path / "two"
    original.mkdir()
    target.mkdir()
    history_path = tmp_path / "history.json"
    ask.save_history(
        [{"role": "user", "content": "target old"}],
        history_path,
        root=target.resolve(),
    )
    seen = []

    def fake_turn(question, *, session, **kwargs):
        seen.append(list(session.messages))
        return ask.AgentTurn(question, "ok", [], [])

    monkeypatch.setattr(ask, "HISTORY_FILE", history_path)
    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    monkeypatch.setattr(ask, "start_mcp", lambda *args: None)
    inputs = iter([f"/root {target}", "new", "/exit"])

    ask.chat_loop(
        renderer=RecordingRenderer(),
        read_input=lambda prompt: next(inputs),
        agent_root=original,
    )

    assert seen == [[]]
    assert ask.load_history(history_path, root=target.resolve()) == []


class FakeMCPManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.schemas = []
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


def test_chat_starts_mcp_on_first_plain_turn_only(monkeypatch, tmp_path: Path) -> None:
    starts = []
    seen_managers = []

    def fake_start(root):
        manager = FakeMCPManager(root)
        starts.append(root)
        return manager

    def fake_turn(question, *, mcp, **kwargs):
        seen_managers.append(mcp)
        return ask.AgentTurn(question, "ok", [], [])

    monkeypatch.setattr(ask, "start_mcp", fake_start)
    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    monkeypatch.setattr(ask, "load_history", lambda *a, **k: [])
    monkeypatch.setattr(ask, "save_history", lambda *a, **k: None)
    inputs = iter(["/status", "first", "second", "/exit"])

    ask.chat_loop(
        renderer=RecordingRenderer(),
        read_input=lambda prompt: next(inputs),
        agent_root=tmp_path,
    )

    assert starts == [tmp_path.resolve()]
    assert seen_managers[0] is seen_managers[1]
    assert seen_managers[0].stops == 1


def test_root_change_stops_manager_and_lazily_restarts_at_new_root(
    monkeypatch, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    managers = []

    def fake_start(root):
        manager = FakeMCPManager(root)
        managers.append(manager)
        return manager

    monkeypatch.setattr(ask, "start_mcp", fake_start)
    monkeypatch.setattr(
        ask,
        "run_agent_turn",
        lambda question, **kwargs: ask.AgentTurn(question, "ok", [], []),
    )
    monkeypatch.setattr(ask, "load_history", lambda *a, **k: [])
    monkeypatch.setattr(ask, "save_history", lambda *a, **k: None)
    inputs = iter(["first", f"/root {other}", "second", "/exit"])

    ask.chat_loop(
        renderer=RecordingRenderer(),
        read_input=lambda prompt: next(inputs),
        agent_root=tmp_path,
    )

    assert [manager.root for manager in managers] == [
        tmp_path.resolve(),
        other.resolve(),
    ]
    assert [manager.stops for manager in managers] == [1, 1]


def test_chat_stops_mcp_on_eof(monkeypatch, tmp_path: Path) -> None:
    manager = FakeMCPManager(tmp_path.resolve())
    inputs = iter(["question"])

    monkeypatch.setattr(ask, "start_mcp", lambda root: manager)
    monkeypatch.setattr(
        ask,
        "run_agent_turn",
        lambda question, **kwargs: ask.AgentTurn(question, "ok", [], []),
    )
    monkeypatch.setattr(ask, "load_history", lambda *a, **k: [])
    monkeypatch.setattr(ask, "save_history", lambda *a, **k: None)

    ask.chat_loop(
        renderer=RecordingRenderer(),
        read_input=lambda prompt: next(inputs),
        agent_root=tmp_path,
    )

    assert manager.stops == 1


def test_chat_stops_mcp_when_agent_turn_is_interrupted(
    monkeypatch, tmp_path: Path
) -> None:
    manager = FakeMCPManager(tmp_path.resolve())

    monkeypatch.setattr(ask, "start_mcp", lambda root: manager)
    monkeypatch.setattr(
        ask,
        "run_agent_turn",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(ask, "load_history", lambda *a, **k: [])
    monkeypatch.setattr(ask, "save_history", lambda *a, **k: None)
    inputs = iter(["question"])

    ask.chat_loop(
        renderer=RecordingRenderer(),
        read_input=lambda prompt: next(inputs),
        agent_root=tmp_path,
    )

    assert manager.stops == 1


def test_one_shot_starts_and_stops_mcp_once(monkeypatch, tmp_path: Path) -> None:
    import sys

    manager = FakeMCPManager(tmp_path.resolve())
    seen = {}
    monkeypatch.setattr(ask, "start_mcp", lambda root: manager)
    monkeypatch.setattr(
        ask,
        "run_agent_turn",
        lambda question, *, mcp, **kwargs: (
            seen.update(mcp=mcp) or ask.AgentTurn(question, "ok", [], [])
        ),
    )
    monkeypatch.setattr(ask.ui, "make_renderer", RecordingRenderer)
    monkeypatch.setattr(sys, "argv", ["lca", "--root", str(tmp_path), "question"])

    ask.main()

    assert seen["mcp"] is manager
    assert manager.stops == 1


def test_partial_mcp_start_failure_is_stopped(monkeypatch, tmp_path: Path) -> None:
    class PartialManager(FakeMCPManager):
        def start(self):
            raise RuntimeError("boom")

    manager = PartialManager(tmp_path.resolve())
    monkeypatch.setattr(
        ask.mcp_client, "load_config", lambda path: {"servers": {"x": {}}}
    )
    monkeypatch.setattr(ask.mcp_client, "MCPManager", lambda config, root: manager)

    assert ask.start_mcp(tmp_path.resolve()) is None
    assert manager.stops == 1


def test_mcp_startup_failure_keeps_native_agent_turn_available(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "app.py").write_text("answer = 42\n", encoding="utf-8")
    renderer = RecordingRenderer()
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "app.py"},
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "app.py:1 is 42."}},
        ]
    )

    monkeypatch.setattr(ask, "start_mcp", lambda root: None)
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        lambda **kwargs: iter([next(responses)]) if kwargs.get("stream") else next(responses),
    )
    monkeypatch.setattr(ask, "load_history", lambda *a, **k: [])
    monkeypatch.setattr(ask, "save_history", lambda *a, **k: None)
    inputs = iter(["what is answer?", "/exit"])

    ask.chat_loop(
        renderer=renderer,
        read_input=lambda prompt: next(inputs),
        agent_root=tmp_path,
    )

    assert renderer.tokens == ["app.py:1 is 42."]
    assert any("read_file" in message for message in renderer.messages)


def test_declined_confirmation_reaches_model_and_leaves_file_unchanged(
    monkeypatch, tmp_path: Path
) -> None:
    target = tmp_path / "app.py"
    target.write_text("answer = 1\n", encoding="utf-8")
    model_calls = []
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "edit_file",
                                "arguments": {
                                    "path": "app.py",
                                    "old_text": "answer = 1",
                                    "new_text": "answer = 2",
                                },
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "No change applied."}},
        ]
    )

    def fake_chat(**kwargs):
        model_calls.append(list(kwargs["messages"]))
        response = next(responses)
        return iter([response]) if kwargs.get("stream") else response

    monkeypatch.setattr(agent.ollama, "chat", fake_chat)
    monkeypatch.setattr(ask, "start_mcp", lambda root: None)
    monkeypatch.setattr(ask, "load_history", lambda *a, **k: [])
    monkeypatch.setattr(ask, "save_history", lambda *a, **k: None)
    inputs = iter(["change answer", "n", "/exit"])

    ask.chat_loop(
        renderer=RecordingRenderer(),
        read_input=lambda prompt: next(inputs),
        agent_root=tmp_path,
    )

    assert target.read_text(encoding="utf-8") == "answer = 1\n"
    assert any("User declined the change." in str(message) for message in model_calls[1])


def test_one_shot_without_confirmation_channel_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    import sys

    target = tmp_path / "app.py"
    target.write_text("answer = 1\n", encoding="utf-8")
    model_calls = []
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "edit_file",
                                "arguments": {
                                    "path": "app.py",
                                    "old_text": "answer = 1",
                                    "new_text": "answer = 2",
                                },
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "Cannot apply."}},
        ]
    )

    def fake_chat(**kwargs):
        model_calls.append(list(kwargs["messages"]))
        response = next(responses)
        return iter([response]) if kwargs.get("stream") else response

    monkeypatch.setattr(agent.ollama, "chat", fake_chat)
    monkeypatch.setattr(ask, "start_mcp", lambda root: None)
    monkeypatch.setattr(ask.ui, "make_renderer", RecordingRenderer)
    monkeypatch.setattr(
        sys, "argv", ["lca", "--root", str(tmp_path), "change answer"]
    )

    ask.main()

    assert target.read_text(encoding="utf-8") == "answer = 1\n"
    assert any(
        "No confirmation channel available." in str(message)
        for message in model_calls[1]
    )


def test_source_scope_applies_to_later_plain_agent_turn(
    monkeypatch, tmp_path: Path
) -> None:
    scopes = []

    def fake_turn(question, *, session, **kwargs):
        scopes.append(session.docs_source)
        return ask.AgentTurn(question, "ok", [], [])

    monkeypatch.setattr(ask, "run_agent_turn", fake_turn)
    monkeypatch.setattr(ask, "list_sources", lambda: ["pandas"])
    run_chat(
        monkeypatch,
        ["/source pandas", "docs question", "/source all", "other question"],
        RecordingRenderer(),
        tmp_path,
    )

    assert scopes == ["pandas", None]


def test_status_reports_mcp_state_and_confirmation_policy(
    monkeypatch, tmp_path: Path
) -> None:
    renderer = RecordingRenderer()
    run_chat(monkeypatch, ["/status"], renderer, tmp_path)

    status = next(message for message in renderer.messages if "source: all" in message)
    assert "MCP: not started" in status
    assert "mutations: confirmation required" in status


def test_plain_mixed_question_preserves_file_and_docs_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "app.py").write_text(
        "import pandas as pd\nresult = pd.concat([])\n", encoding="utf-8"
    )
    renderer = RecordingRenderer()
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "app.py"},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_docs",
                                "arguments": {"query": "pandas concat"},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "See app.py:2 and "
                        "[1] docs/pandas/merge.md § Concatenating objects."
                    ),
                }
            },
        ]
    )

    monkeypatch.setattr(
        agent.rag,
        "retrieve",
        lambda *args, **kwargs: {
            "documents": [["concat combines objects"]],
            "metadatas": [[{"path": "docs/pandas/merge.md", "heading": "Concatenating objects"}]],
            "distances": [[0.1]],
        },
    )
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        lambda **kwargs: iter([next(responses)]) if kwargs.get("stream") else next(responses),
    )
    monkeypatch.setattr(ask, "start_mcp", lambda root: None)
    monkeypatch.setattr(ask, "load_history", lambda *a, **k: [])
    monkeypatch.setattr(ask, "save_history", lambda *a, **k: None)
    inputs = iter(["why does concat fail?", "/exit"])

    ask.chat_loop(
        renderer=renderer,
        read_input=lambda prompt: next(inputs),
        agent_root=tmp_path,
    )

    assert "app.py:2" in "".join(renderer.tokens)
    assert renderer.sources == [
        ["[1] docs/pandas/merge.md § Concatenating objects"]
    ]
    trace = "\n".join(renderer.messages)
    assert "read_file" in trace and "search_docs" in trace


def test_docs_failure_plain_turn_continues_with_file_reasoning(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "app.py").write_text("answer = 42\n", encoding="utf-8")
    renderer = RecordingRenderer()
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_docs",
                                "arguments": {"query": "answer"},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "app.py"},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Docs unavailable; app.py:1 sets 42.",
                }
            },
        ]
    )

    def broken_retrieve(*args, **kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(agent.rag, "retrieve", broken_retrieve)
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        lambda **kwargs: iter([next(responses)]) if kwargs.get("stream") else next(responses),
    )
    monkeypatch.setattr(ask, "start_mcp", lambda root: None)
    monkeypatch.setattr(ask, "load_history", lambda *a, **k: [])
    monkeypatch.setattr(ask, "save_history", lambda *a, **k: None)
    inputs = iter(["explain answer", "/exit"])

    ask.chat_loop(
        renderer=renderer,
        read_input=lambda prompt: next(inputs),
        agent_root=tmp_path,
    )

    trace = "\n".join(renderer.messages)
    assert "search_docs" in trace and "-> ERROR" in trace
    assert "read_file" in trace
    assert "app.py:1" in "".join(renderer.tokens)
