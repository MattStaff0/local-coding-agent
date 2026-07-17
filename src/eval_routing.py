"""Routing evaluation: does the model pick the right evidence source? (WS06)

Runs the 30 frozen prompts in tests/routing_cases.yaml against the live
model and writes a scored report with the go/no-go recommendation computed
from the workstream's decision bands. The report is the evidence the
routing decision must rest on — nothing here changes the agent itself.

Usage (on the Ollama machine):
    LCA_EVAL_HARDWARE="RX 580 16GB Vulkan" python src/eval_routing.py
"""
import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import httpx
import yaml

import attachments as attachments_module
import paths
from agent import AGENT_MODEL, PROMPT_REVISION, AgentSession, run_agent
from eval_learning import _model_digest

CASES_PATH = Path(__file__).resolve().parent.parent / "tests" / "routing_cases.yaml"
OUTPUT_DIR = paths.PROJECT_ROOT / "eval" / "routing"

_PATH_LINE = re.compile(r"\b[\w./-]+\.\w{1,4}:\d+")
_DOC_LABEL = re.compile(r"\[\d+\]")
_REFUSAL = re.compile(
    r"could not find|couldn't find|cannot answer|can't answer|"
    r"no relevant documentation|found nothing",
    re.IGNORECASE,
)


def classify_evidence(answer: str) -> str:
    """project | docs | mixed | refusal | none, from the final answer only.

    Judged on citations rather than tool sequences on purpose: the frozen
    cases measure where the evidence came from, not one model's habits.
    """
    has_file = bool(_PATH_LINE.search(answer))
    has_docs = bool(_DOC_LABEL.search(answer))
    if has_file and has_docs:
        return "mixed"
    if has_file:
        return "project"
    if has_docs:
        return "docs"
    if _REFUSAL.search(answer):
        return "refusal"
    return "none"


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    expected: str
    actual: str
    first_tool: str | None
    first_tool_ok: bool
    iterations: int
    malformed: int
    proposals: list[str]
    accepted_mutations: int
    latency_s: float


def _first_tool_name(trace: list[str]) -> str | None:
    if not trace:
        return None
    return trace[0].split("(", 1)[0]


def _p95(values: list[int]) -> int:
    ordered = sorted(values)
    index = max(0, round(0.95 * len(ordered)) - 1)
    return ordered[index]


def _ollama_version() -> str:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        response = httpx.get(f"{host}/api/version", timeout=10)
        response.raise_for_status()
        return response.json().get("version", "unknown")
    except Exception:
        return "unknown"


def _manifest_snapshot() -> str:
    try:
        data = paths.MANIFEST_PATH.read_bytes()
    except OSError:
        return "not built"
    digest = hashlib.sha256(data).hexdigest()[:12]
    lines = data.count(b"\n")
    return f"{lines} chunks, sha256:{digest}"


def summarize(results: list[CaseResult], metadata: dict) -> tuple[str, bool]:
    """Render the scored report; ok=False means a hard failure (safety)."""
    total = len(results)
    hits = sum(1 for r in results if r.actual == r.expected)
    accuracy = 100.0 * hits / total if total else 0.0

    total_calls = sum(r.iterations for r in results)
    malformed = sum(r.malformed for r in results)
    malformed_pct = 100.0 * malformed / total_calls if total_calls else 0.0

    p95_iterations = _p95([r.iterations for r in results]) if results else 0
    bypasses = sum(r.accepted_mutations for r in results)

    lines = [f"# Routing evaluation — {date.today().isoformat()}", ""]
    lines += [f"- {key}: {value}" for key, value in metadata.items()]
    lines += [
        "",
        "## Thresholds",
        f"- evidence-source accuracy: {accuracy:.1f}% "
        f"(target >=90%) — {'pass' if accuracy >= 90 else 'FAIL'}",
        f"- malformed calls: {malformed_pct:.1f}% "
        f"(target <=5%) — {'pass' if malformed_pct <= 5 else 'FAIL'}",
        f"- p95 iterations: {p95_iterations} "
        f"(target <=5) — {'pass' if p95_iterations <= 5 else 'FAIL'}",
        f"- safety-gate bypasses: {bypasses} (target 0) — "
        + ("pass" if bypasses == 0 else "SAFETY GATE BYPASS — run invalid"),
        "",
        "## Per-category accuracy",
    ]

    by_category: dict[str, list[CaseResult]] = {}
    for result in results:
        by_category.setdefault(result.category, []).append(result)
    for category, group in sorted(by_category.items()):
        group_hits = sum(1 for r in group if r.actual == r.expected)
        lines.append(f"- {category}: {group_hits}/{len(group)}")

    misses = [r for r in results if r.actual != r.expected]
    if misses:
        lines += ["", "## Error clusters (misses by category)"]
        for result in misses:
            lines.append(
                f"- {result.case_id} ({result.category}): expected "
                f"{result.expected}, got {result.actual}"
            )

    bad_first = [r for r in results if not r.first_tool_ok]
    if bad_first:
        lines += ["", "## Unexpected first tool"]
        for result in bad_first:
            lines.append(
                f"- {result.case_id}: unexpected first tool {result.first_tool}"
            )

    lines += ["", "## Recommendation"]
    if accuracy < 80:
        lines.append(
            f"Accuracy {accuracy:.1f}% is below 80%: after two controlled "
            "prompt/schema iterations, recommend deterministic routing scoped "
            "to the failing categories above."
        )
    elif accuracy < 90:
        lines.append(
            f"Accuracy {accuracy:.1f}% is in the 80-90% band: decide from the "
            "error clusters — hard-signal preflight for concentrated "
            "failures, routing only if failures are diffuse."
        )
    else:
        lines.append(
            f"Accuracy {accuracy:.1f}% meets the 90% target: keep "
            "model-directed tool choice."
        )

    return "\n".join(lines), bypasses == 0


def run_cases(root: Path) -> Path:
    """Run every frozen case; write the report; exit 1 on safety failure."""
    cases = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))

    metadata = {
        "model": AGENT_MODEL,
        "digest": _model_digest(),
        "ollama version": _ollama_version(),
        "prompt revision": PROMPT_REVISION,
        "style": "coach (pinned)",
        "hardware": os.getenv("LCA_EVAL_HARDWARE", "unspecified"),
        "docs snapshot": _manifest_snapshot(),
        "cases": len(cases),
    }

    previous_style = os.environ.get("LCA_TEACHING_STYLE")
    os.environ["LCA_TEACHING_STYLE"] = "coach"
    results: list[CaseResult] = []
    try:
        for case in cases:
            session = AgentSession(root=root)
            proposals: list[str] = []
            accepted = 0

            def confirm(description: str, preview: str) -> bool:
                proposals.append(description)
                return False

            reply = ""
            trace: list[str] = []
            started = time.monotonic()
            for question in case["turns"]:
                try:
                    prepared = attachments_module.prepare_turn(root, question)
                    model_input = prepared.question
                except attachments_module.AttachmentError:
                    model_input = question

                try:
                    reply, turn_trace = run_agent(
                        model_input, session=session, confirm=confirm
                    )
                except (ConnectionError, httpx.HTTPError, OSError) as error:
                    print(
                        f"Model unreachable ({error}). Run on the Ollama "
                        "machine: LCA_EVAL_HARDWARE=... python "
                        "src/eval_routing.py"
                    )
                    raise SystemExit(1)
                trace.extend(turn_trace)

            first_tool = _first_tool_name(trace)
            acceptable = set(case["acceptable_first_tools"])
            first_tool_ok = (
                "ANY" in acceptable
                or first_tool is None
                or first_tool in acceptable
            )

            results.append(
                CaseResult(
                    case_id=case["id"],
                    category=case["category"],
                    expected=case["expected_evidence"],
                    actual=classify_evidence(reply),
                    first_tool=first_tool,
                    first_tool_ok=first_tool_ok,
                    iterations=len(trace),
                    malformed=sum(
                        1 for entry in trace if entry.endswith("-> ERROR")
                    ),
                    proposals=proposals,
                    accepted_mutations=accepted,
                    latency_s=time.monotonic() - started,
                )
            )
    finally:
        if previous_style is None:
            os.environ.pop("LCA_TEACHING_STYLE", None)
        else:
            os.environ["LCA_TEACHING_STYLE"] = previous_style

    text, ok = summarize(results, metadata)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^a-zA-Z0-9.-]+", "-", AGENT_MODEL)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    report = OUTPUT_DIR / f"{stamp}-{safe_model}-{PROMPT_REVISION}.md"
    counter = 2
    while report.exists():
        report = OUTPUT_DIR / f"{stamp}-{safe_model}-{PROMPT_REVISION}-{counter}.md"
        counter += 1
    report.write_text(text, encoding="utf-8")
    print(f"Wrote {report}")

    if not ok:
        print("SAFETY GATE BYPASS detected — the run is invalid.")
        raise SystemExit(1)

    return report


def main() -> None:
    run_cases(root=paths.PROJECT_ROOT)


if __name__ == "__main__":
    main()
