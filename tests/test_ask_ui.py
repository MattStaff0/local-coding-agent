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

    monkeypatch.setattr(ask, "load_history", lambda path: [])
    monkeypatch.setattr(ask, "save_history", lambda history, path: None)
    ask.chat_loop(renderer=renderer, read_input=read)


def test_help_lists_every_command(monkeypatch):
    renderer = _RecordingRenderer()
    _run_chat(monkeypatch, ["/help"], renderer)
    help_text = "\n".join(renderer.messages)
    for command in ["/agent", "/export", "/help", "/source", "/sources", "/exit"]:
        assert command in help_text


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

    def fake_answer(question, history, source, on_token=None):
        for token in ["hi", "!"]:
            on_token(token)
        return "hi!", [{"source": "s", "path": "p", "heading": "h"}]

    monkeypatch.setattr(ask, "answer_question", fake_answer)
    _run_chat(monkeypatch, ["a question"], _TokenRenderer())
    assert tokens == ["hi", "!"]
