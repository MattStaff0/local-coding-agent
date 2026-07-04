"""All terminal presentation for the chat loop.

Two renderers behind one duck-typed interface: RichRenderer live-renders
the streamed answer as markdown (syntax-highlighted code blocks), while
PlainRenderer reproduces the pre-rich plain output byte-for-byte for
pipes and tests. Logic modules never import rich directly.
"""
import sys
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel


class PlainRenderer:
    def on_token(self, token: str) -> None:
        print(token, end="", flush=True)

    def finish_answer(self) -> None:
        print()

    def show_message(self, text: str) -> None:
        print(text)

    def show_error(self, text: str) -> None:
        print(text)

    def show_sources(self, legend_lines: list[str]) -> None:
        print("\nSources:")
        for line in legend_lines:
            print(f"  {line}")

    @contextmanager
    def status(self, text: str):
        yield


class RichRenderer:
    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.buffer = ""
        self._live: Live | None = None

    def on_token(self, token: str) -> None:
        if self._live is None:
            self._live = Live(
                Markdown(self.buffer),
                console=self.console,
                refresh_per_second=8,
                vertical_overflow="visible",
            )
            self._live.start()
        self.buffer += token
        self._live.update(Markdown(self.buffer))

    def finish_answer(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self.buffer = ""
        self.console.print()

    def show_message(self, text: str) -> None:
        self.console.print(text)

    def show_error(self, text: str) -> None:
        self.console.print(f"[red]{text}[/red]")

    def show_sources(self, legend_lines: list[str]) -> None:
        self.console.print(Panel("\n".join(legend_lines), title="Sources", style="dim"))

    def status(self, text: str):
        return self.console.status(text)


def make_renderer(force_plain: bool = False):
    if force_plain or not sys.stdout.isatty():
        return PlainRenderer()
    return RichRenderer()


COMMANDS = ["/agent", "/code", "/exit", "/export", "/help", "/source", "/sources"]


def completion_words(sources: list[str]) -> list[str]:
    return COMMANDS + [f"/source {source}" for source in sources] + ["/source all"]


def make_input(sources_fn):
    """Line reader: prompt_toolkit on a real terminal, builtin input elsewhere.

    Building the PromptSession lazily keeps `import ui` safe in tests and
    pipes, where prompt_toolkit would fail probing the terminal.
    """
    if not sys.stdin.isatty():
        return lambda prompt_text: input(prompt_text)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory

    import paths

    session = PromptSession(
        history=FileHistory(str(paths.ASK_HISTORY_FILE)),
        completer=WordCompleter(completion_words(sources_fn()), sentence=True),
    )
    return lambda prompt_text: session.prompt(prompt_text)
