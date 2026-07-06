"""/code answers from the code index, streaming through the renderer."""
import ask
import rag
import ui
from rag import NoRelevantDocsError


class _RecordingRenderer(ui.PlainRenderer):
    def __init__(self):
        self.messages = []
        self.errors = []
        self.tokens = []

    def show_message(self, text):
        self.messages.append(text)

    def show_error(self, text):
        self.errors.append(text)

    def on_token(self, token):
        self.tokens.append(token)


def _run_chat(monkeypatch, lines, renderer):
    inputs = iter(lines + ["/exit"])
    monkeypatch.setattr(ask, "load_history", lambda path: [])
    monkeypatch.setattr(ask, "save_history", lambda history, path: None)
    ask.chat_loop(renderer=renderer, read_input=lambda prompt_text: next(inputs))


def test_code_question_streams_through_renderer(monkeypatch):
    renderer = _RecordingRenderer()
    recorded = {}

    def fake_answer_code(question, history=None, repo=None, on_token=None):
        recorded["question"] = question
        for token in ["ret", "rieve"]:
            on_token(token)
        return "retrieve", [
            {"path": "src/rag.py", "start_line": 478, "heading": "src/rag.py > retrieve"}
        ]

    monkeypatch.setattr(ask, "answer_code_question", fake_answer_code)
    _run_chat(monkeypatch, ["/code how does retrieve work?"], renderer)

    assert recorded["question"] == "how does retrieve work?"
    assert renderer.tokens == ["ret", "rieve"]


def test_code_alone_prints_usage(monkeypatch):
    renderer = _RecordingRenderer()
    _run_chat(monkeypatch, ["/code"], renderer)
    assert any("/code" in m for m in renderer.messages)


def test_code_refusal_surfaces_the_message(monkeypatch):
    renderer = _RecordingRenderer()

    def refusing(question, history=None, repo=None, on_token=None):
        raise NoRelevantDocsError("Nothing relevant is indexed for that question.")

    monkeypatch.setattr(ask, "answer_code_question", refusing)
    _run_chat(monkeypatch, ["/code what is the meaning of life?"], renderer)

    assert any("Nothing relevant" in e for e in renderer.errors)


def test_chunk_label_shows_start_line_for_code():
    label = rag.chunk_label(
        1,
        {"path": "src/rag.py", "start_line": 478, "heading": "src/rag.py > retrieve"},
    )
    assert label == "[1] src/rag.py:478 § src/rag.py > retrieve"


def test_help_and_completion_mention_code():
    assert "/code" in ask.HELP_TEXT
    assert "/code" in ui.COMMANDS


def _code_results(distance=0.1):
    return {
        "documents": [["src/rag.py > retrieve\n\ndef retrieve(): ..."]],
        "metadatas": [[{"path": "src/rag.py", "start_line": 478, "heading": "h"}]],
        "distances": [[distance]],
        "keyword_hits": [[False]],
    }


def test_answer_code_question_targets_the_code_collection(monkeypatch):
    recorded = {}

    def fake_retrieve(question, **kwargs):
        recorded.update(kwargs)
        return _code_results()

    prompts = {}

    def fake_ask_model(prompt, on_token=None):
        prompts["p"] = prompt
        return "answer [1]"

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "ask_model", fake_ask_model)

    answer, metadatas = rag.answer_code_question("how does retrieve work?")

    assert recorded["collection_name"] == rag.CODE_COLLECTION_NAME
    assert recorded["manifest_path"] == rag.CODE_MANIFEST_PATH
    assert rag.CODE_PROMPT_RULES.splitlines()[0] in prompts["p"]
    assert metadatas[0]["start_line"] == 478


def test_answer_code_question_refuses_far_matches(monkeypatch):
    import pytest

    monkeypatch.setattr(rag, "retrieve", lambda question, **kwargs: _code_results(distance=1.9))

    with pytest.raises(NoRelevantDocsError):
        rag.answer_code_question("what is the meaning of life?")
