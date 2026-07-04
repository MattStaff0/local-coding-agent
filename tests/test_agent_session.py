"""The agent remembers earlier /agent turns within a chat session."""
from pathlib import Path

import agent


def _scripted_ollama(monkeypatch, replies):
    """Each reply is a message dict; records every messages list passed in."""
    calls = []

    def fake_chat(model, messages, tools):
        calls.append(list(messages))
        return {"message": replies.pop(0)}

    monkeypatch.setattr(agent.ollama, "chat", fake_chat)
    return calls


def test_session_carries_history_between_calls(tmp_path, monkeypatch):
    calls = _scripted_ollama(monkeypatch, [
        {"content": "first answer", "tool_calls": []},
        {"content": "second answer", "tool_calls": []},
    ])
    session = agent.AgentSession(root=tmp_path)

    agent.run_agent("what is rag.py?", session=session)
    agent.run_agent("and its tests?", session=session)

    second_call_messages = calls[1]
    assert any("what is rag.py?" in str(m) for m in second_call_messages)
    assert any("first answer" in str(m) for m in second_call_messages)
    # System prompt present exactly once, at the front.
    assert second_call_messages[0]["role"] == "system"
    assert sum(1 for m in second_call_messages if m.get("role") == "system") == 1


def test_one_shot_still_works_without_session(tmp_path, monkeypatch):
    _scripted_ollama(monkeypatch, [{"content": "ok", "tool_calls": []}])
    answer, trace = agent.run_agent("q", root=tmp_path)
    assert answer == "ok"


def test_parse_agent_subcommands():
    assert agent.parse_agent_command("/agent reset") == ("reset", "")
    assert agent.parse_agent_command("/agent root ~/school/proj") == ("root", "~/school/proj")
    assert agent.parse_agent_command("/agent why?") == ("ask", "why?")
    assert agent.parse_agent_command("/agent") == ("status", "")
    assert agent.parse_agent_command("plain text") is None


def test_root_change_starts_fresh_session(tmp_path, monkeypatch):
    import ask
    import ui

    new_root = tmp_path / "other-repo"
    new_root.mkdir()
    recorded = []

    def fake_run_agent(question, root=None, session=None, **kwargs):
        recorded.append((question, session.root, len(session.messages)))
        return "ok", []

    inputs = iter([
        "/agent first question",
        f"/agent root {new_root}",
        "/agent second question",
        "/exit",
    ])
    monkeypatch.setattr(ask, "run_agent", fake_run_agent)
    monkeypatch.setattr(ask, "load_history", lambda path: [])
    monkeypatch.setattr(ask, "save_history", lambda history, path: None)

    ask.chat_loop(
        renderer=ui.PlainRenderer(),
        read_input=lambda prompt_text: next(inputs),
    )

    # Fresh session after a root change: no carried-over messages, new root.
    # (len is taken before run_agent appends, so both calls see 0 prior.)
    assert recorded[0][1] != new_root and recorded[0][2] == 0
    assert recorded[1] == ("second question", new_root, 0)
