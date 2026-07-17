"""eval_learning: semi-automated rubric runner with honest human columns."""
from pathlib import Path

import pytest

import eval_learning


@pytest.fixture()
def fake_agent(monkeypatch):
    """Scripted run_agent: turn-1 replies come from `first_reply`."""
    state = {"first_reply": "Concept hint. Next check: print x.shape. [1]"}
    calls = []

    def fake_run_agent(question, session=None, **kwargs):
        calls.append(question)
        reply = (
            state["first_reply"]
            if len(session.messages) == 0
            else "Escalated answer."
        )
        session.messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reply},
            ]
        )
        return reply, []

    monkeypatch.setattr(eval_learning, "run_agent", fake_run_agent)
    return state, calls


def run_and_read(tmp_path, monkeypatch) -> str:
    monkeypatch.setattr(eval_learning, "OUTPUT_DIR", tmp_path)
    report = eval_learning.run_rubric(root=tmp_path)
    return report.read_text()


def test_report_records_model_prompt_revision_and_date(
    tmp_path, monkeypatch, fake_agent
):
    import agent

    text = run_and_read(tmp_path, monkeypatch)

    assert agent.PROMPT_REVISION in text
    assert eval_learning.AGENT_MODEL in text
    assert "digest:" in text
    assert "unknown" in text  # Ollama unreachable in tests → digest unknown


def test_every_rubric_case_appears_with_all_turns(tmp_path, monkeypatch, fake_agent):
    _, calls = fake_agent

    text = run_and_read(tmp_path, monkeypatch)

    assert "tf-shape-mismatch" in text
    assert "proj-fix-and-decline" in text
    assert len(calls) >= 32  # 16 cases x >=2 turns


def test_full_code_at_hint_stage_is_flagged(tmp_path, monkeypatch, fake_agent):
    state, _ = fake_agent
    state["first_reply"] = (
        "Here you go:\n```python\na=1\nb=2\nc=3\nd=4\ne=5\nf=6\n```"
    )

    text = run_and_read(tmp_path, monkeypatch)

    assert "full_code_at_hint: FAIL" in text


def test_hint_without_code_passes_the_lexical_cell(tmp_path, monkeypatch, fake_agent):
    text = run_and_read(tmp_path, monkeypatch)

    assert "full_code_at_hint: pass" in text
    assert "next_check: pass" in text
    assert "citation: pass" in text


def test_human_review_cells_are_left_blank(tmp_path, monkeypatch, fake_agent):
    text = run_and_read(tmp_path, monkeypatch)

    # Teaching quality is not lexically measurable; these stay for a human.
    assert "- [ ] concept_correct" in text
    assert "- [ ] evidence_supports_claim" in text
    assert "- [ ] escalates_correctly" in text
    assert "- [ ] uncertainty_labeled" in text
    assert "human review" in text.lower()


def test_unreachable_model_fails_with_pc_command(monkeypatch, tmp_path, capsys):
    def boom(question, session=None, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(eval_learning, "run_agent", boom)
    monkeypatch.setattr(eval_learning, "OUTPUT_DIR", tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        eval_learning.run_rubric(root=tmp_path)

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "OLLAMA_HOST" in out
