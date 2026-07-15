"""Retrieval eval harness (repo plan #10): hit-rate@k and MRR over golden.yaml.

The metrics are computed against a scripted retrieve function — real numbers
come from running src/eval_retrieval.py on the machine that has Ollama.
"""

from pathlib import Path

import eval_retrieval

REPO_ROOT = Path(__file__).resolve().parent.parent


def results_with_paths(paths: list[str]) -> dict:
    return {
        "documents": [[f"chunk from {p}" for p in paths]],
        "metadatas": [[{"path": p, "source": p.split("/")[1]} for p in paths]],
        "distances": [[0.1 * (i + 1) for i in range(len(paths))]],
    }


def test_rank_of_expected_matches_paths_by_suffix() -> None:
    metadatas = [
        {"path": "docs/python/datastructures.md"},
        {"path": "docs/pytorch/tensorqs-tutorial.md"},
    ]

    rank = eval_retrieval.rank_of_expected(
        "docs/pytorch/tensorqs-tutorial.md", metadatas
    )

    assert rank == 2


def test_rank_of_expected_returns_none_for_a_miss() -> None:
    metadatas = [{"path": "docs/python/datastructures.md"}]

    assert eval_retrieval.rank_of_expected("docs/pytorch/intro.md", metadatas) is None


def test_evaluate_computes_hit_rates_and_mrr() -> None:
    golden = [
        {"question": "q1", "path": "docs/pytorch/a.md"},
        {"question": "q2", "path": "docs/python/b.md"},
    ]

    def retrieve_fn(question: str, n_results: int = 4, source: str | None = None):
        if question == "q1":  # hit at rank 1
            return results_with_paths(["docs/pytorch/a.md", "docs/python/b.md"])
        # hit at rank 3
        return results_with_paths(
            ["docs/pytorch/a.md", "docs/chroma/c.md", "docs/python/b.md"]
        )

    report = eval_retrieval.evaluate(golden, k=4, retrieve_fn=retrieve_fn)

    overall = report["overall"]
    assert overall["n"] == 2
    assert overall["hit@1"] == 0.5
    assert overall["hit@k"] == 1.0
    assert abs(overall["mrr"] - (1.0 + 1.0 / 3.0) / 2.0) < 1e-9


def test_evaluate_counts_misses_as_zero() -> None:
    golden = [{"question": "q", "path": "docs/pytorch/a.md"}]

    def retrieve_fn(question: str, n_results: int = 4, source: str | None = None):
        return results_with_paths(["docs/python/b.md"])

    overall = eval_retrieval.evaluate(golden, k=4, retrieve_fn=retrieve_fn)["overall"]

    assert overall["hit@1"] == 0.0
    assert overall["hit@k"] == 0.0
    assert overall["mrr"] == 0.0


def test_evaluate_breaks_results_down_per_source() -> None:
    golden = [
        {"question": "q1", "path": "docs/pytorch/a.md"},
        {"question": "q2", "path": "docs/python/b.md"},
    ]

    def retrieve_fn(question: str, n_results: int = 4, source: str | None = None):
        return results_with_paths(["docs/pytorch/a.md"])

    report = eval_retrieval.evaluate(golden, k=4, retrieve_fn=retrieve_fn)

    assert report["per_source"]["pytorch"]["hit@k"] == 1.0
    assert report["per_source"]["python"]["hit@k"] == 0.0


def test_negative_entries_score_refusal():
    golden = [
        {"question": "covered", "path": "docs/python/a.md"},
        {"question": "off topic", "expect": "refusal"},
        {"question": "also off topic", "expect": "refusal"},
    ]

    def fake_retrieve(question, n_results=4):
        if question == "covered":
            return {
                "metadatas": [[{"path": "docs/python/a.md"}]],
                "distances": [[0.1]],
                "keyword_hits": [[True]],
            }
        # Far from everything, no keyword rescue -> should refuse.
        refused = question == "off topic"
        return {
            "metadatas": [[{"path": "docs/python/other.md"}]],
            "distances": [[0.9 if refused else 0.1]],
            "keyword_hits": [[False]],
        }

    report = eval_retrieval.evaluate(golden, k=4, retrieve_fn=fake_retrieve)

    assert report["refusal"] == {"n": 2, "correct": 0.5}
    assert report["overall"]["n"] == 1  # negatives never dilute hit metrics
    assert "refusal" in eval_retrieval.format_report(report, k=4)


def test_negative_only_golden_scores_refusal_without_positive_metrics_crash():
    golden = [{"question": "off topic", "expect": "refusal"}]

    def fake_retrieve(question, n_results=4):
        return {"metadatas": [[]], "distances": [[0.9]], "keyword_hits": [[False]]}

    report = eval_retrieval.evaluate(golden, k=4, retrieve_fn=fake_retrieve)

    assert report["overall"] == {"n": 0, "hit@1": 0.0, "hit@k": 0.0, "mrr": 0.0}
    assert report["refusal"] == {"n": 1, "correct": 1.0}
    assert "refusal" in eval_retrieval.format_report(report, k=4)


def test_format_report_is_readable() -> None:
    golden = [{"question": "q1", "path": "docs/pytorch/a.md"}]

    def retrieve_fn(question: str, n_results: int = 4, source: str | None = None):
        return results_with_paths(["docs/pytorch/a.md"])

    report = eval_retrieval.evaluate(golden, k=4, retrieve_fn=retrieve_fn)
    text = eval_retrieval.format_report(report, k=4)

    assert "hit@1" in text
    assert "hit@4" in text
    assert "mrr" in text
    assert "pytorch" in text


def test_code_flag_wires_the_code_retrieve_fn(monkeypatch) -> None:
    recorded = {}

    def fake_retrieve(question, n_results=4, source=None, **kwargs):
        recorded.update(kwargs)
        return results_with_paths(["src/rag.py"])

    monkeypatch.setattr(eval_retrieval.rag, "retrieve", fake_retrieve)

    golden = [{"question": "where is rrf?", "path": "src/rag.py"}]
    eval_retrieval.evaluate(
        golden, k=4, retrieve_fn=eval_retrieval.code_retrieve_fn()
    )

    assert recorded["collection_name"] == "local_code"
    assert recorded["manifest_path"] == eval_retrieval.rag.CODE_MANIFEST_PATH


def test_code_golden_file_loads_and_every_path_exists() -> None:
    golden = eval_retrieval.load_golden(REPO_ROOT / "tests" / "golden_code.yaml")

    assert len(golden) >= 6
    for entry in golden:
        assert entry["question"].strip()
        assert (REPO_ROOT / entry["path"]).is_file(), entry["path"]


def test_golden_file_loads_and_every_path_exists() -> None:
    golden = eval_retrieval.load_golden(REPO_ROOT / "tests" / "golden.yaml")

    assert len(golden) >= 15
    for entry in golden:
        assert entry["question"].strip()
        if entry.get("expect") == "refusal":
            assert "path" not in entry
            continue
        assert (REPO_ROOT / entry["path"]).is_file(), entry["path"]
