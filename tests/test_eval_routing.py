"""Routing evaluation harness: frozen cases, scoring, decision bands (WS06)."""
from pathlib import Path

import pytest
import yaml

import agent
import eval_routing

CASES_PATH = Path(__file__).resolve().parent / "routing_cases.yaml"


@pytest.fixture(autouse=True)
def no_network_metadata(monkeypatch):
    """Unit tests never touch the network, whatever is listening locally."""
    monkeypatch.setattr(eval_routing, "_ollama_version", lambda: "unknown")
    monkeypatch.setattr(eval_routing, "_model_digest", lambda: "unknown")


# --- frozen case schema ---


def test_frozen_cases_shape_and_balance():
    cases = yaml.safe_load(CASES_PATH.read_text())

    assert len(cases) >= 30
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))

    counts = {}
    tool_names = {
        schema["function"]["name"] for schema in agent.TOOL_SCHEMAS
    } | {"ANY"}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
        assert case["expected_evidence"] in {"project", "docs", "mixed", "refusal"}
        assert set(case["acceptable_first_tools"]) <= tool_names, case["id"]
        assert isinstance(case["mutation_expected"], bool)

    assert counts["project"] >= 8
    assert counts["docs"] >= 8
    assert counts["mixed"] >= 5
    assert counts["attachment"] >= 3
    assert counts["ambiguous"] >= 2
    assert counts["mutation"] >= 2
    assert counts["negative"] >= 2


# --- evidence classification ---


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("The cap is set at src/agent_tools.py:9.", "project"),
        ("Broadcasting aligns trailing dims. Evidence: [1] numpy § rules", "docs"),
        ("Our retrieve (src/rag.py:812) matches [2] chroma § filtering.", "mixed"),
        ("I searched grep and the docs and could not find that symbol.", "refusal"),
        ("No relevant documentation matched; I cannot answer that.", "refusal"),
    ],
)
def test_classify_evidence(answer, expected):
    assert eval_routing.classify_evidence(answer) == expected


# --- scoring and report ---


def scripted(monkeypatch, reply="See src/rag.py:1.", trace=None, accept=False):
    def fake_run_agent(question, session=None, confirm=None, **kwargs):
        if confirm is not None and "mutate" in question.lower():
            confirm("edit_file src/x.py", "diff")
        session.messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reply},
            ]
        )
        return reply, list(trace or ["grep({'pattern': 'x'})"])

    monkeypatch.setattr(eval_routing, "run_agent", fake_run_agent)


def test_report_scores_thresholds_and_recommends(tmp_path, monkeypatch):
    scripted(monkeypatch)
    monkeypatch.setattr(eval_routing, "OUTPUT_DIR", tmp_path)

    report = eval_routing.run_cases(root=tmp_path)
    text = report.read_text()

    assert "evidence-source accuracy" in text
    assert "malformed calls" in text
    assert "p95 iterations" in text
    assert "safety-gate bypasses: 0" in text
    assert "Recommendation" in text
    assert agent.PROMPT_REVISION in text
    assert "digest:" in text
    assert "ollama version: unknown" in text  # offline in tests


def test_low_accuracy_recommends_deterministic_routing(tmp_path, monkeypatch):
    # Every reply cites project evidence, so all docs/mixed/refusal cases
    # miss → accuracy far below 80% → deterministic-routing recommendation.
    scripted(monkeypatch, reply="See src/rag.py:1.")
    monkeypatch.setattr(eval_routing, "OUTPUT_DIR", tmp_path)

    text = eval_routing.run_cases(root=tmp_path).read_text()

    assert "below 80%" in text
    assert "deterministic routing" in text


def test_malformed_and_iteration_metrics_come_from_trace(tmp_path, monkeypatch):
    scripted(
        monkeypatch,
        trace=[
            "grep({'pattern': 'x'}) -> ERROR",
            "grep({'pattern': 'y'})",
            "read_file({'path': 'a.py'})",
        ],
    )
    monkeypatch.setattr(eval_routing, "OUTPUT_DIR", tmp_path)

    text = eval_routing.run_cases(root=tmp_path).read_text()

    assert "malformed calls: 33.3%" in text  # 1 of 3 per case
    assert "p95 iterations: 3" in text


def test_accepted_mutation_is_a_safety_failure():
    passing = eval_routing.CaseResult(
        case_id="x", category="project", expected="project", actual="project",
        first_tool="grep", first_tool_ok=True, iterations=1, malformed=0,
        proposals=["edit_file a.py"], accepted_mutations=0, latency_s=0.1,
    )
    bypassed = eval_routing.CaseResult(
        case_id="y", category="mutation", expected="project", actual="project",
        first_tool="grep", first_tool_ok=True, iterations=1, malformed=0,
        proposals=["edit_file a.py"], accepted_mutations=1, latency_s=0.1,
    )

    ok_text, ok = eval_routing.summarize([passing], metadata={})
    fail_text, failed_ok = eval_routing.summarize([bypassed], metadata={})

    assert ok and "safety-gate bypasses: 0" in ok_text
    assert not failed_ok and "SAFETY GATE BYPASS" in fail_text


def test_first_tool_mismatch_is_reported(tmp_path, monkeypatch):
    # search_docs first on a project-only question → flagged in clusters.
    scripted(monkeypatch, trace=["search_docs({'query': 'x'})"])
    monkeypatch.setattr(eval_routing, "OUTPUT_DIR", tmp_path)

    text = eval_routing.run_cases(root=tmp_path).read_text()

    assert "unexpected first tool" in text


def test_unreachable_model_names_the_pc_command(tmp_path, monkeypatch, capsys):
    def boom(question, session=None, **kwargs):
        raise ConnectionError("refused")

    monkeypatch.setattr(eval_routing, "run_agent", boom)
    monkeypatch.setattr(eval_routing, "OUTPUT_DIR", tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        eval_routing.run_cases(root=tmp_path)

    assert excinfo.value.code == 1
    assert "eval_routing" in capsys.readouterr().out
