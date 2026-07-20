---
date: 2026-07-19
repo: C:\Users\mstaf\Desktop\local-ai-coding-agent
status: proposed (not started)
severity: medium — Windows-only correctness/security gaps + test-infra portability
depends_on: commit d3e8985 (cross-platform attachment reads, manifest retry, test isolation)
---

# Plan: Windows hardening follow-ups

## Provenance of this plan

The primary Windows bugs (the `O_NOFOLLOW` attachment crash and the `pytest`
manifest clobber) are **fixed and committed** in `d3e8985`. A headless Codex
review of that commit came back **clean — no findings**:

> "The changes correctly provide a Windows-safe attachment read fallback, retry
> transient manifest replacement failures, and isolate test data paths. The full
> suite's remaining Windows failures are unrelated pre-existing symlink/
> environment portability issues."

This plan therefore does **not** cover the committed code. It collects the
follow-up items that the earlier portability audit explicitly scoped **out** of
that commit — real, independent Windows gaps that remain. Line numbers verified
against the working tree at `d3e8985`.

Do the items in order; each is independent and separately committable. Work on a
branch. Show diffs before committing.

---

## Item 1 — NTFS junction blindness in `is_symlink()` checks (MEDIUM, security)

**Problem.** Three sandbox/traversal guards use `Path.is_symlink()`:
- `src/attachments.py:245` (`_reject_symlinked_parents` — walks parent dirs)
- `src/agent_tools.py:41` (`_iter_text_files` — list/grep file walk)
- `src/rag.py:510` (`index_code` — skips symlinks during code ingest)

On Windows, `Path.is_symlink()` recognizes only reparse tag
`IO_REPARSE_TAG_SYMLINK`. It does **not** recognize **NTFS junctions**
(`IO_REPARSE_TAG_MOUNT_POINT`, created with `mklink /J`, no admin required). A
junction planted inside the sandbox root can therefore redirect traversal to an
out-of-root physical location and walk straight past all three guards on Windows.
This is a Windows-only bypass of a defense the POSIX code believes it has.

**Fix.** Add one shared helper (suggest `src/fs_policy.py`, since it already owns
the shared deny/ignore policy and both `attachments.py` and `agent_tools.py`
import it) that treats reparse points as symlinks on Windows:

```python
# fs_policy.py
import os
import stat as _stat

def is_reparse_or_symlink(path) -> bool:
    """True for POSIX symlinks AND Windows reparse points (symlinks + junctions).

    Path.is_symlink() misses NTFS junctions (mklink /J), which can redirect a
    seemingly-in-root path outside the sandbox. Junctions set the
    FILE_ATTRIBUTE_REPARSE_POINT bit, surfaced by os.lstat via st_reparse_tag /
    the stat.FILE_ATTRIBUTE_REPARSE_POINT flag.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if _stat.S_ISLNK(st.st_mode):
        return True
    attrs = getattr(st, "st_file_attributes", 0)
    return bool(attrs & getattr(_stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
```

Then replace the three `path.is_symlink()` call sites with
`fs_policy.is_reparse_or_symlink(path)`. On POSIX the reparse branch is `0`, so
behavior is unchanged there.

**Tests.**
- [ ] Unit-test `is_reparse_or_symlink` returns `False` for a regular file/dir.
- [ ] Windows-only test (guarded, see Item 4): create a junction with
      `subprocess.run(["cmd", "/c", "mklink", "/J", link, target])`, assert the
      helper returns `True` and that `_reject_symlinked_parents` /
      `_iter_text_files` reject/skip it. Guard with a capability probe so it
      skips cleanly on POSIX and on Windows where junction creation is blocked.

**Verify.** `pytest -p no:cacheprovider --basetemp=<clean external dir>
tests/test_attachments.py tests/test_agent_tools.py` green; POSIX behavior
unchanged.

---

## Item 2 — `shlex.split` mangles Windows backslash paths in `run_command` (MEDIUM)

**Problem.** `src/agent_tools.py:207` does `argv = shlex.split(command)` with the
default `posix=True`. In POSIX mode backslash is an escape character, so a normal
Windows path argument is silently corrupted before `subprocess.run` sees it:
`pytest tests\test_rag.py::test_x` → `pytest teststest_rag.py::test_x`. The agent
(or the user) then gets a confusing "file not found" instead of a run. `pytest`
and `python` are the only allowlisted commands (`agent_tools.py:201`), and both
routinely take path args, so this is reachable in normal use on Windows.

**Fix — pick one, prefer (a):**
- (a) Split in non-POSIX mode on Windows: `shlex.split(command, posix=(os.name
  != "nt"))`. Non-POSIX mode preserves backslashes and still handles quotes,
  which is the correct tokenizer for Windows-style command strings.
- (b) Normalize separators before/after splitting. More surprising; avoid.

Keep the `ValueError` guard (line 208) and the allowlist check (line 211)
exactly as-is.

**Interaction with Item 3.** After this fix, the token for the program may be
`python` or `pytest` as typed — the allowlist check in Item 3 still needs the
`.exe` tolerance below.

**Tests.**
- [ ] `run_command(root, r"pytest tests\test_x.py")` tokenizes to
      `["pytest", r"tests\test_x.py"]` on Windows (assert via a monkeypatched
      `subprocess.run` capturing `argv`, so no real process spawns).
- [ ] POSIX unchanged: `run_command(root, "pytest tests/test_x.py")` →
      `["pytest", "tests/test_x.py"]`.
- [ ] Existing `run_command` allow/deny tests still pass.

---

## Item 3 — `ALLOWED_COMMANDS` misses `python.exe` / `pytest.exe` (LOW)

**Problem.** `src/agent_tools.py:211` checks `Path(argv[0]).name not in
ALLOWED_COMMANDS` where `ALLOWED_COMMANDS = {"pytest", "python"}`. On Windows a
resolved launcher is often `python.exe` / `pytest.exe`, whose `.name` is
`"python.exe"` — not in the set — so a legitimate command is rejected.

**Fix.** Compare against the name with any `.exe` suffix stripped, without
widening the allowlist's intent:

```python
program = Path(argv[0]).name.lower()
if program.endswith(".exe"):
    program = program[:-4]
if not argv or program not in ALLOWED_COMMANDS:
    ...
```

Keep the allowlist set itself unchanged (`{"pytest", "python"}`) — the comment at
`agent_tools.py:199-200` correctly says growing that set is a review decision.
This is only about matching the Windows executable name, not adding commands.

**Tests.**
- [ ] `python.exe ...` and `pytest.exe ...` are allowed; `pip`, `powershell`,
      `cmd`, `python3.11.exe`-with-a-path-traversal remain rejected.
- [ ] Case-insensitivity: `Python.EXE` allowed (Windows is case-insensitive).

---

## Item 4 — Symlink tests error instead of skip without Developer Mode (test-infra)

**Problem.** `tests/test_attachments.py` creates symlinks unguarded:
`symlink_to` at lines 188, 194, 203. On stock Windows without Developer Mode or
elevation, `Path.symlink_to` raises `OSError: [WinError 1314]` (privilege not
held), so these three tests **error** rather than pass or skip. They are the
"pre-existing Windows failures" both reviews referenced — orthogonal to the
attachment fix, but they make the suite look broken on a fresh Windows checkout.

**Fix.** Add a capability probe and `skipif` so they skip cleanly when symlink
creation isn't available:

```python
# tests/test_attachments.py (near the top)
import pytest, tempfile
from pathlib import Path

def _can_symlink() -> bool:
    try:
        with tempfile.TemporaryDirectory() as d:
            link = Path(d) / "l"
            link.symlink_to(Path(d))
        return True
    except (OSError, NotImplementedError):
        return False

requires_symlinks = pytest.mark.skipif(
    not _can_symlink(), reason="symlink creation unavailable (Windows: needs Developer Mode/elevation)"
)
```

Apply `@requires_symlinks` to `test_file_symlink_escaping_root_rejected`,
`test_file_symlink_inside_root_allowed`, and
`test_directory_symlink_parent_rejected_even_when_target_in_root`. Do **not**
weaken what they assert — a POSIX/Dev-Mode machine must still run them fully. The
containment logic they cover is real; we're only fixing how they degrade where
the OS can't create the fixture.

**Note.** If Item 1's junction test lands, it exercises the junction bypass on
plain Windows (no privilege needed), giving coverage on the exact machines where
these symlink tests skip — a nice complement.

**Verify.** On this Windows box (no Dev Mode), these three now **skip**, not
error; the earlier "6 symlink privilege failures" drop out of the failure count.

---

## Out of scope (do not touch)

- The committed `d3e8985` changes — reviewed clean, leave them.
- `tests/golden.yaml`, `tests/golden_code.yaml`, `tests/routing_cases.yaml`,
  `tests/learning_rubric.yaml` — eval ground truth, never edited to pass numbers.
- The `.env`-present "default models" test and `~`-expansion test — those are
  environment/config assumptions, not portability defects; a separate decision.

## Constraints (all items)

- **Never run a bare `pytest` from the repo root** — the isolation backstop
  (`tests/conftest.py`) now exists, but still pass
  `-p no:cacheprovider --basetemp=<clean dir outside the repo>` because
  `AppData\Local\Temp\pytest-of-mstaf` throws PermissionError here. After any
  run, confirm `manifest.jsonl` is byte-unchanged (it's gitignored — byte-compare
  against a copy, don't rely on `git status`).
- No retrieval logic changes in any item → no `eval_retrieval.py` delta required.
  If that ever stops being true, run it before/after and report both numbers.
- Keep the codebase dependency-light: stdlib only (`os`, `stat`, `shlex`,
  `subprocess`, `tempfile`) — no `pywin32`/`ctypes`.

## Definition of done (per CLAUDE.md)

- [ ] `pytest` (clean basetemp) green on Windows apart from genuinely
      environment-gated tests, which now **skip** rather than error.
- [ ] POSIX behavior byte-identical for every changed function.
- [ ] `manifest.jsonl` provably byte-unchanged by a test run.
- [ ] README/`how-it-works.md` note the junction-aware guard if it changes any
      documented security claim; otherwise no doc change needed.
- [ ] Each item its own commit with a clear message; do not force-add the
      gitignored `how-it-works.md` / `RUNBOOK.md`.
</content>
