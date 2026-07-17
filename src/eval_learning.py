"""Run the learning-contract rubric against the live model (workstream 05).

Semi-automated on purpose: three rubric cells are lexically checkable
(full code at hint stage, presence of a next check, presence of a citation);
the four that measure actual teaching quality are written as blank
checkboxes for a human reviewer. Pretending pedagogy is grep-able would be
the eval-harness version of citation theater.

Usage (on the Ollama machine):
    python src/eval_learning.py
Writes eval/learning/<date>-<model>-<rev>.md — commit it with the review.
"""
import os
import re
import sys
from datetime import date
from pathlib import Path

import httpx
import yaml

import paths
from agent import AGENT_MODEL, PROMPT_REVISION, AgentSession, run_agent

RUBRIC_PATH = Path(__file__).resolve().parent.parent / "tests" / "learning_rubric.yaml"
OUTPUT_DIR = paths.PROJECT_ROOT / "eval" / "learning"

# Auto-scored cells.
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_NEXT_CHECK_WORDS = re.compile(
    r"\b(check|print|run|compare|inspect|try|look at|next)\b", re.IGNORECASE
)
_CITATION = re.compile(r"\[\d+\]|\b[\w./-]+\.\w+:\d+")

# Human-review cells, straight from the learning-contract spec's rubric.
HUMAN_CELLS = (
    "concept_correct",
    "evidence_supports_claim",
    "escalates_correctly",
    "uncertainty_labeled",
)


def _full_code_at_hint(reply: str) -> bool:
    """A fenced block of 5+ lines in a turn-1 reply is a ladder violation."""
    return any(
        block.count("\n") >= 6  # fence + 5 content lines
        for block in _CODE_BLOCK.findall(reply)
    )


def _model_digest() -> str:
    """The exact model blob the results belong to; 'unknown' offline."""
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        response = httpx.post(
            f"{host}/api/show", json={"model": AGENT_MODEL}, timeout=10
        )
        response.raise_for_status()
        return response.json().get("modelinfo", {}).get(
            "general.basename", ""
        ) or response.json().get("digest", "unknown")
    except Exception:
        return "unknown"


def _score_first_reply(reply: str) -> list[str]:
    lines = []
    lines.append(
        "  full_code_at_hint: "
        + ("FAIL" if _full_code_at_hint(reply) else "pass")
    )
    lines.append(
        "  next_check: "
        + ("pass" if _NEXT_CHECK_WORDS.search(reply) else "FAIL")
    )
    lines.append(
        "  citation: " + ("pass" if _CITATION.search(reply) else "none")
    )
    return lines


def run_rubric(root: Path) -> Path:
    """Run every rubric case in a fresh session; write the review report."""
    cases = yaml.safe_load(RUBRIC_PATH.read_text(encoding="utf-8"))
    digest = _model_digest()

    report_lines = [
        f"# Learning-contract evaluation — {date.today().isoformat()}",
        "",
        f"- model: {AGENT_MODEL}",
        f"- digest: {digest}",
        f"- prompt revision: {PROMPT_REVISION}",
        f"- cases: {len(cases)}",
        "",
        "Three cells per case are auto-scored (lexical); four require human",
        "review. Release threshold: >=90% of ALL cells pass, zero unsupported",
        "citations, zero unconfirmed mutations.",
        "",
    ]

    for case in cases:
        session = AgentSession(root=root)
        report_lines.append(f"## {case['id']} ({case['area']})")
        report_lines.append(f"reviewer notes: {case['notes'].strip()}")

        first_reply = None
        for turn_number, question in enumerate(case["turns"], start=1):
            try:
                reply, _ = run_agent(question, session=session)
            except (ConnectionError, httpx.HTTPError, OSError) as error:
                print(
                    f"Model unreachable ({error}). Run this on the Ollama "
                    "machine with OLLAMA_HOST set (see README): "
                    "python src/eval_learning.py"
                )
                raise SystemExit(1)

            if first_reply is None:
                first_reply = reply
            report_lines.append(f"**turn {turn_number} — you:** {question}")
            report_lines.append(f"**lca:** {reply}")

        report_lines.append("auto-scored:")
        report_lines.extend(_score_first_reply(first_reply or ""))
        report_lines.append("human review:")
        report_lines.extend(f"- [ ] {cell}" for cell in HUMAN_CELLS)
        report_lines.append("")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^a-zA-Z0-9.-]+", "-", AGENT_MODEL)
    report = OUTPUT_DIR / (
        f"{date.today().isoformat()}-{safe_model}-{PROMPT_REVISION}.md"
    )
    report.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote {report}")
    return report


def main() -> None:
    run_rubric(root=paths.PROJECT_ROOT)


if __name__ == "__main__":
    main()
