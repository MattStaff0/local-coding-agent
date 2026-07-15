"""Renderers: rich for humans, plain for pipes and tests."""
import io
from contextlib import redirect_stdout

from rich.console import Console

import ui


def test_plain_renderer_streams_tokens_verbatim():
    out = io.StringIO()
    renderer = ui.PlainRenderer()
    with redirect_stdout(out):
        renderer.on_token("hel")
        renderer.on_token("lo")
        renderer.finish_answer()
    assert out.getvalue() == "hello\n"


def test_plain_status_is_a_noop_context_manager():
    with ui.PlainRenderer().status("retrieving…"):
        pass  # must not raise, must not print


def test_make_renderer_picks_plain_when_not_a_tty(monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: False, raising=False)
    assert isinstance(ui.make_renderer(), ui.PlainRenderer)


def test_make_renderer_forced_plain_beats_tty(monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    assert isinstance(ui.make_renderer(force_plain=True), ui.PlainRenderer)


def test_rich_renderer_accumulates_and_resets_buffer():
    console = Console(file=io.StringIO(), force_terminal=True)
    renderer = ui.RichRenderer(console=console)
    renderer.on_token("# hi\n")
    renderer.on_token("code")
    assert renderer.buffer == "# hi\ncode"
    renderer.finish_answer()
    assert renderer.buffer == ""  # reset for the next turn


def test_show_error_mid_stream_resets_the_buffer():
    # An Ollama drop mid-answer must not leave the dead answer's text
    # prefixed onto the next one.
    console = Console(file=io.StringIO(), force_terminal=True)
    renderer = ui.RichRenderer(console=console)
    renderer.on_token("partial answer")
    renderer.show_error("connection lost")

    assert renderer.buffer == ""
    assert renderer._live is None

    renderer.on_token("fresh")
    assert renderer.buffer == "fresh"
    renderer.finish_answer()


def test_completion_words_include_commands_and_sources():
    words = ui.completion_words(["python", "pytorch"])
    assert "/help" in words
    assert "/source python" in words
    assert "/source all" in words


def test_make_input_falls_back_to_builtin_off_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    reader = ui.make_input(lambda: [])
    monkeypatch.setattr("builtins.input", lambda prompt: "hello")
    assert reader("You: ") == "hello"
