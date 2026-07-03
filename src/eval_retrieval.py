"""Measure retrieval quality against the golden question set.

Usage (on the machine with Ollama, after ingesting):

    python src/eval_retrieval.py [--k 4]

Prints hit-rate@1, hit-rate@k, and MRR overall and per source. Chunking or
retrieval changes should move these numbers, not vibes.
"""

import argparse
from pathlib import Path
from typing import Any, Callable

import yaml

import rag

GOLDEN_PATH = Path("tests/golden.yaml")


def load_golden(path: Path = GOLDEN_PATH) -> list[dict[str, str]]:
    """Load the golden set: a list of {question, path} mappings."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def rank_of_expected(
    expected_path: str, metadatas: list[dict[str, Any]]
) -> int | None:
    """1-based rank of the first retrieved chunk from the expected document.

    Paths are compared by suffix so absolute indexed paths still match the
    repo-relative paths golden.yaml uses.
    """
    for rank, metadata in enumerate(metadatas, start=1):
        if str(metadata.get("path", "")).endswith(expected_path):
            return rank

    return None


def _scores(ranks: list[int | None], k: int) -> dict[str, float | int]:
    n = len(ranks)
    if n == 0:
        return {"n": 0, "hit@1": 0.0, "hit@k": 0.0, "mrr": 0.0}

    return {
        "n": n,
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        "hit@k": sum(1 for r in ranks if r is not None and r <= k) / n,
        "mrr": sum(1.0 / r for r in ranks if r is not None) / n,
    }


def evaluate(
    golden: list[dict[str, str]],
    k: int = 4,
    retrieve_fn: Callable[..., dict[str, Any]] = rag.retrieve,
) -> dict[str, Any]:
    """Run every golden question through retrieval and score the ranks."""
    ranks: list[int | None] = []
    ranks_by_source: dict[str, list[int | None]] = {}
    positives = [entry for entry in golden if entry.get("expect") != "refusal"]
    negatives = [entry for entry in golden if entry.get("expect") == "refusal"]

    refusals = 0
    for entry in negatives:
        results = retrieve_fn(entry["question"], n_results=k)
        if rag.would_refuse(results):
            refusals += 1

    for entry in positives:
        results = retrieve_fn(entry["question"], n_results=k)
        rank = rank_of_expected(entry["path"], results["metadatas"][0])

        ranks.append(rank)
        # The expected doc's top-level folder is its source (docs/<source>/...).
        source = Path(entry["path"]).parts[1]
        ranks_by_source.setdefault(source, []).append(rank)

    report = {
        "overall": _scores(ranks, k),
        "per_source": {
            source: _scores(source_ranks, k)
            for source, source_ranks in sorted(ranks_by_source.items())
        },
    }
    if negatives:
        report["refusal"] = {"n": len(negatives), "correct": refusals / len(negatives)}

    return report


def format_report(report: dict[str, Any], k: int) -> str:
    """Render the metric table the way it should be read: overall first."""

    def line(name: str, scores: dict[str, float | int]) -> str:
        return (
            f"{name:<12} n={scores['n']:<3} "
            f"hit@1={scores['hit@1']:.2f} "
            f"hit@{k}={scores['hit@k']:.2f} "
            f"mrr={scores['mrr']:.2f}"
        )

    lines = [line("overall", report["overall"])]
    lines.extend(
        line(source, scores) for source, scores in report["per_source"].items()
    )
    if "refusal" in report:
        refusal = report["refusal"]
        lines.append(
            f"{'refusal':<12} n={refusal['n']:<3} correct={refusal['correct']:.2f}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrieval on golden.yaml")
    parser.add_argument("--k", type=int, default=4, help="results per query")
    args = parser.parse_args()

    golden = load_golden()
    report = evaluate(golden, k=args.k)

    print(format_report(report, k=args.k))


if __name__ == "__main__":
    main()
