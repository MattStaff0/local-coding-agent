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
