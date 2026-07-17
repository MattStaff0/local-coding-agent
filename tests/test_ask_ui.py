"""Chat loop drives the renderer, never print(), for user-visible output."""
import ask
import ui


class _RecordingRenderer(ui.PlainRenderer):
    def __init__(self):
        self.messages = []

    def show_message(self, text):
        self.messages.append(text)


def _run_chat(monkeypatch, lines, renderer, prompts=None):
    inputs = iter(lines + ["/exit"])

    def read(prompt_text):
        if prompts is not None:
            prompts.append(prompt_text)
        return next(inputs)

    monkeypatch.setattr(ask, "load_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(ask, "save_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(ask, "start_mcp", lambda *args: None)
    ask.chat_loop(renderer=renderer, read_input=read)


def test_help_lists_unified_operational_commands_and_deprecated_aliases(monkeypatch):
    renderer = _RecordingRenderer()
    _run_chat(monkeypatch, ["/help"], renderer)
    help_text = "\n".join(renderer.messages)
    for command in [
        "/status",
        "/root",
        "/reset",
        "/agent",
        "/code",
        "/export",
        "/help",
        "/source",
        "/sources",
        "/exit",
    ]:
        assert command in help_text
    assert "deprecated" in help_text.lower()
    assert "saved files" in help_text.lower()
    assert "confirmation" in help_text.lower()
    assert len(ask.HELP_TEXT.splitlines()) < 20


def test_prompt_shows_active_source(monkeypatch):
    renderer = _RecordingRenderer()
    prompts = []
    monkeypatch.setattr(ask, "list_sources", lambda: ["pytorch"])
    _run_chat(monkeypatch, ["/source pytorch"], renderer, prompts=prompts)
    assert prompts[-1] == "[pytorch] You: "


def test_answer_streams_through_renderer(monkeypatch):
    tokens = []

    class _TokenRenderer(_RecordingRenderer):
        def on_token(self, token):
            tokens.append(token)

    def fake_answer(question, session=None, on_token=None, **kwargs):
        for token in ["hi", "!"]:
            on_token(token)
        session.messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "hi!"},
            ]
        )
        return "hi!", []

    monkeypatch.setattr(ask, "run_agent", fake_answer)
    _run_chat(monkeypatch, ["a question"], _TokenRenderer())
    assert tokens == ["hi", "!"]
