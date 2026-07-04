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
