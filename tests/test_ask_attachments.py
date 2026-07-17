"""@path and --context wiring: attachments resolve before the model is called."""
import sys

import pytest

import ask
import ui


class _RecordingRenderer(ui.PlainRenderer):
    def __init__(self):
        self.messages = []
        self.errors = []

    def show_message(self, text):
        self.messages.append(text)

    def show_error(self, text):
        self.errors.append(text)


def capture_run_agent(monkeypatch):
    """Replace ask.run_agent, recording every question it receives."""
    questions = []

    def fake_run_agent(question, session=None, on_token=None, **kwargs):
        questions.append(question)
        session.messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "ok"},
            ]
        )
        return "ok", []

    monkeypatch.setattr(ask, "run_agent", fake_run_agent)
    return questions


def run_chat(monkeypatch, lines, renderer, root):
    inputs = iter(lines + ["/exit"])
    monkeypatch.setattr(ask, "load_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(ask, "save_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(ask, "start_mcp", lambda *args: None)
    ask.chat_loop(
        renderer=renderer, read_input=lambda _: next(inputs), agent_root=root
    )


@pytest.fixture()
def root(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\ny = 2\n")
    return tmp_path


def test_interactive_at_path_embeds_content_before_the_model_call(
    monkeypatch, root
):
    questions = capture_run_agent(monkeypatch)
    renderer = _RecordingRenderer()

    run_chat(monkeypatch, ["@app.py why does this crash?"], renderer, root)

    assert len(questions) == 1
    assert "1: x = 1" in questions[0]
    assert "why does this crash?" in questions[0]
    assert any("Attached: app.py (2 lines)" in m for m in renderer.messages)


def test_interactive_attachment_error_never_calls_the_model(monkeypatch, root):
    (root / ".env").write_text("KEY=value\n")
    questions = capture_run_agent(monkeypatch)
    renderer = _RecordingRenderer()

    run_chat(monkeypatch, ["@.env what is in here?"], renderer, root)

    assert questions == []
    assert any("deny" in e for e in renderer.errors)
    # The secret's content must never appear anywhere in the transcript.
    assert not any("KEY=value" in m for m in renderer.messages + renderer.errors)


def test_interactive_attachment_only_prompt_is_an_error(monkeypatch, root):
    questions = capture_run_agent(monkeypatch)
    renderer = _RecordingRenderer()

    run_chat(monkeypatch, ["@app.py"], renderer, root)

    assert questions == []
    assert any("without a question" in e for e in renderer.errors)


def test_one_shot_context_flag_and_inline_at_path_are_equivalent(
    monkeypatch, root, capsys
):
    questions = capture_run_agent(monkeypatch)
    monkeypatch.setattr(ask, "start_mcp", lambda *args: None)

    monkeypatch.setattr(
        sys, "argv",
        ["lca", "--root", str(root), "--context", "app.py", "why crash?"],
    )
    ask.main()

    monkeypatch.setattr(
        sys, "argv", ["lca", "--root", str(root), "why crash? @app.py"]
    )
    ask.main()

    assert len(questions) == 2
    assert questions[0] == questions[1]
    assert "1: x = 1" in questions[0]


def test_one_shot_context_with_range_suffix(monkeypatch, root):
    questions = capture_run_agent(monkeypatch)
    monkeypatch.setattr(ask, "start_mcp", lambda *args: None)
    monkeypatch.setattr(
        sys, "argv",
        ["lca", "--root", str(root), "--context", "app.py:2", "what is y?"],
    )

    ask.main()

    assert "2: y = 2" in questions[0]
    assert "1: x = 1" not in questions[0]


def test_one_shot_missing_context_exits_2_naming_path_and_root(
    monkeypatch, root, capsys
):
    capture_run_agent(monkeypatch)
    monkeypatch.setattr(ask, "start_mcp", lambda *args: None)
    monkeypatch.setattr(
        sys, "argv",
        ["lca", "--root", str(root), "--context", "nope.py", "why?"],
    )

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 2
    out = capsys.readouterr().out
    assert "nope.py" in out
    assert str(root) in out


def test_help_documents_attachments_and_saved_file_visibility(monkeypatch, root):
    renderer = _RecordingRenderer()
    run_chat(monkeypatch, ["/help"], renderer, root)

    help_text = "\n".join(renderer.messages)
    assert "@path" in help_text
    assert "--context" in help_text
    assert "saved" in help_text.lower()
    assert len(ask.HELP_TEXT.splitlines()) < 20
