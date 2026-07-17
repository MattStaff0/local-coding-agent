"""Learning-first contract: versioned prompt, coach/direct styles (WS05)."""
import re
from pathlib import Path

import yaml

import agent


def test_prompt_revision_exists_and_is_versioned():
    assert re.fullmatch(r"lc-\d+", agent.PROMPT_REVISION)


def test_coach_prompt_contains_the_learning_ladder(monkeypatch):
    monkeypatch.delenv("LCA_TEACHING_STYLE", raising=False)

    prompt = agent.system_prompt()

    # The ladder: concept, evidence, one next check before full solutions.
    assert "concept" in prompt
    assert "next check" in prompt.lower()
    # Escalation is conversational, not a mode.
    assert "show me" in prompt
    # An explicit direct-answer request always wins the current turn.
    assert "direct" in prompt.lower()
    # Evidence labels keep file facts, doc facts, and inference apart.
    assert "the file says" in prompt
    assert "the docs say" in prompt
    assert "I infer" in prompt
    # Failure-mode guards: no pure Socratic, no unsolicited full code.
    assert "one concrete" in prompt.lower()
    assert "full code" in prompt.lower() or "full solution" in prompt.lower()
    # Declined edits return to coaching.
    assert "declined" in prompt.lower()


def test_direct_style_swaps_the_ladder(monkeypatch):
    monkeypatch.setenv("LCA_TEACHING_STYLE", "direct")

    prompt = agent.system_prompt()

    assert "Answer directly first" in prompt
    # Direct changes depth, never evidence discipline.
    assert "the file says" in prompt
    assert "next check" not in prompt.lower()


def test_invalid_teaching_style_falls_back_to_coach(monkeypatch):
    monkeypatch.setenv("LCA_TEACHING_STYLE", "sensei")

    prompt = agent.system_prompt()

    assert "next check" in prompt.lower()


def test_prompt_stays_within_small_model_budget(monkeypatch):
    monkeypatch.delenv("LCA_TEACHING_STYLE", raising=False)

    assert len(agent.system_prompt()) < 2600


def test_run_agent_sends_the_composed_prompt(monkeypatch, tmp_path):
    seen = {}

    def fake_chat(model, messages, tools):
        seen["system"] = messages[0]["content"]
        return {"message": {"role": "assistant", "content": "ok"}}

    monkeypatch.setattr(agent.ollama, "chat", fake_chat)
    monkeypatch.setenv("LCA_TEACHING_STYLE", "direct")

    agent.run_agent("q", root=tmp_path)

    assert "Answer directly first" in seen["system"]


# --- rubric file schema ---


RUBRIC_PATH = Path(__file__).resolve().parent / "learning_rubric.yaml"


def test_rubric_has_16_balanced_multi_turn_cases():
    cases = yaml.safe_load(RUBRIC_PATH.read_text())

    assert len(cases) >= 16
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))

    areas = {}
    for case in cases:
        areas.setdefault(case["area"], []).append(case)
        assert len(case["turns"]) >= 2, f"{case['id']} needs escalation turns"
        assert case["notes"], f"{case['id']} needs human-reviewer notes"

    for area in ("tensorflow", "numpy", "pandas", "project"):
        assert len(areas.get(area, [])) >= 4


def test_status_and_doctor_show_prompt_revision(monkeypatch, tmp_path, capsys):
    import sys

    import ask

    monkeypatch.setattr(sys, "argv", ["lca", "doctor"])
    ask.main()
    assert agent.PROMPT_REVISION in capsys.readouterr().out
