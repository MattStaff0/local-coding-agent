---
date: 2026-07-19
repo: C:\Users\mstaf\Desktop\local-ai-coding-agent
status: proposed (not started)
severity: critical — every @path / --context attachment crashes on Windows
---

# Plan: make attachment reads cross-platform (the `os.O_NOFOLLOW` crash)

## Symptom (confirmed live this session)

```
You: @src\ask.py can you read this
Unexpected error (AttributeError):
  ...
  File "src\attachments.py", line 303, in _read_text_strict
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
AttributeError: module 'os' has no attribute 'O_NOFOLLOW'
```

`os.O_NOFOLLOW` is POSIX-only; CPython on Windows never defines the name, so the
attribute lookup raises **before** `os.open` runs. `_read_text_strict`
(`src/attachments.py:299-314`) is the single chokepoint every attachment read
funnels through — plain files (line 324) and notebooks (line 375) alike — so
**all** `@path`/`--context` attachments are dead on Windows, and 18+ tests in
`tests/test_attachments.py` (`TestResolve`, `TestNotebook`) error out here.

## What `O_NOFOLLOW` was protecting (do not just delete it)

`_resolve_target` (`attachments.py:251-296`) does `target = raw.resolve()`
(line 263, fully dereferences symlinks), `is_relative_to(root)` containment
(line 264), and `fs_policy.denied` (line 278). `_reject_symlinked_parents`
(line 239) walks parent components. So by the time `_read_text_strict` runs,
`target` is a canonical, in-root, symlink-free path **at check time**.
`O_NOFOLLOW`'s narrow job: refuse the open if, in the non-atomic gap between that
`.resolve()` and `os.open()`, the file at that exact path was swapped for a
symlink (a TOCTOU / check-then-act race that could redirect the read outside the
sandbox). The pre-open checks are the "check" half; `O_NOFOLLOW` guarded the
"act" half. **Dropping the flag removes a real security control on every
platform, not just Windows** — so the fix must preserve the guarantee, not
delete it.

## Recommended fix — Option B + C (stat-identity guard + `O_BINARY`)

Rejected **Option A** (`getattr(os, "O_NOFOLLOW", 0)` alone): unblocks Windows
but silently drops the TOCTOU guard there — a real regression, and without
`O_BINARY` it also risks silent truncation (below).

Chosen: on POSIX keep `O_NOFOLLOW` (behavior byte-identical to today); on Windows,
since `target` is already resolved, `lstat` it immediately before open (reject if
it's now a symlink — reproduces most of what `O_NOFOLLOW` refuses), then
`fstat` the opened fd and `os.path.samestat` it against the pre-open `lstat` to
catch a mid-race swap. Stdlib only (`os`, `stat`) — no pywin32/ctypes, matching
this repo's dependency-light style. Add `O_BINARY` on all platforms so raw
`os.read` doesn't hit the Windows CRT text-mode CRLF/Ctrl-Z(`0x1A`) translation
that would corrupt the null-byte/UTF-8 binary detection.

> **Correctness detail (verified by the fix-design pass):** the comparison must be
> `lstat(target)` vs `fstat(fd)`, and it is only correct *because* `_resolve_target`
> already dereferenced everything, so `target` denotes the real file. Comparing a
> still-symlinked path's `lstat` against the followed `fstat` would spuriously fail
> `test_file_symlink_inside_root_allowed` (a symlink's inode never equals its
> target's). Legitimate case: they match; real race: they diverge.

### Implementation steps

- [ ] Add `import stat` to `src/attachments.py` top-level imports.
- [ ] Replace `_read_text_strict` body (lines 299-314) with the sketch below:
      compute `flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)`;
      on the no-`O_NOFOLLOW` branch do the pre-open `lstat`/symlink reject and the
      post-open `samestat` check; read as `"rb"` and keep the existing null-byte +
      UTF-8 decode checks unchanged.
- [ ] Keep all `AttachmentError` messages/shape so existing error-path tests and
      the CLI handler in `ask.py` (lines 580-584, 728-730) still catch cleanly.

### Code sketch (`_read_text_strict`, replacing lines 299-314)

```python
import stat  # top-level import

def _read_text_strict(target: Path, display: str) -> str:
    """target is already fully-resolved, symlink-free and in-root (see
    _resolve_target). O_NOFOLLOW closes the TOCTOU window between that
    resolution and this open on POSIX. Windows has no O_NOFOLLOW, so there we
    lstat immediately before open (reject a symlink that just appeared at this
    exact path) and pair it with a post-open fstat/samestat identity check for
    the remaining non-atomic gap. O_BINARY avoids the CRT text-mode CRLF/Ctrl-Z
    translation so the raw-byte checks below see exactly what's on disk.
    """
    have_nofollow = hasattr(os, "O_NOFOLLOW")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)

    try:
        pre = None
        if not have_nofollow:
            pre = os.lstat(target)
            if stat.S_ISLNK(pre.st_mode):
                raise OSError(f"{target} is a symlink")
        fd = os.open(target, flags)
    except OSError as error:
        raise AttachmentError(f"{display}: cannot read ({error})")

    if not have_nofollow:
        try:
            same = os.path.samestat(pre, os.fstat(fd))
        except OSError:
            same = False
        if not same:
            os.close(fd)
            raise AttachmentError(f"{display}: changed while opening (refusing)")

    with os.fdopen(fd, "rb") as handle:
        data = handle.read()

    if b"\x00" in data:
        raise AttachmentError(f"{display}: binary or non-UTF-8 content")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise AttachmentError(f"{display}: binary or non-UTF-8 content")
```

On POSIX, `getattr(os, "O_BINARY", 0) == 0` and `have_nofollow is True`, so the
path is byte-identical to today — no behavior drift, no eval impact.

## Verification

- [ ] Manually reproduce the fix: `lca` then `@src\ask.py summarize this` — must
      attach and answer, no `AttributeError`.
- [ ] Run `pytest tests/test_attachments.py --basetemp=<clean dir>`:
      - `test_binary_content_rejected`, `test_nul_byte_late_in_file_is_still_binary`
        exercise the read/`O_BINARY` path on any platform.
      - `test_file_symlink_inside_root_allowed` exercises the new Windows branch's
        *legitimate* path — confirms the lstat/fstat pairing doesn't false-positive.
      - `TestNotebook` + whole-file/range tests should go from erroring to passing.
- [ ] Confirm POSIX behavior is unchanged (ideally run the same tests on the Mac,
      or reason from the `have_nofollow is True` short-circuit).

## Companion Windows gaps found during the audit (scope decisions)

These are real but distinct from the crash. Recommend fixing #1 in this same PR
(cheap, same file family) and filing #2/#3 as follow-ups:

- [ ] **(1) `O_BINARY`** — included above; required, not optional. Without it a
      naive `O_NOFOLLOW`-drop trades a loud crash for silent truncation at `0x1A`.
- [ ] **(2) NTFS junction blindness — MEDIUM, follow-up.** `Path.is_symlink()` at
      `attachments.py:244`, `agent_tools.py:41`, `rag.py:510` does not recognize
      NTFS junctions (`mklink /J`, no admin needed), so a junction planted in the
      sandbox walks past these checks on Windows. Needs a reparse-point-aware check
      to fully close; separate hardening ticket.
- [ ] **(3) `shlex.split` mangles backslash paths — MEDIUM, follow-up.**
      `agent_tools.py:207` uses posix-mode `shlex.split`, so `run_command` corrupts
      Windows-style path args (`pytest tests\test_rag.py` → `teststest_rag.py`).
      Consider `shlex.split(command, posix=False)` or normalizing separators. Also
      `agent_tools.py:211` (`ALLOWED_COMMANDS` misses `python.exe`) — LOW.

- [ ] **(4) Test-infra, orthogonal:** `test_file_symlink_*` /
      `test_directory_symlink_parent_*` call `Path.symlink_to()` with no guard;
      on stock Windows (no Developer Mode) symlink creation raises `WinError 1314`,
      so they **error** rather than skip. Add a `pytest.mark.skipif` privilege probe.
      Independent of the `_read_text_strict` fix.

## Residual tradeoff (stated plainly)

Option B is not atomic at the syscall level the way `O_NOFOLLOW` is: a theoretical
sliver exists between the pre-open `lstat` and the post-open `samestat` where a
symlink could be planted and reverted to evade detection. Exploiting it needs a
local attacker with concurrent write access to the exact target path in the user's
own project tree — a threat model with far cheaper avenues already. B+C is strictly
better than A (zero Windows protection) and keeps the security contract intentional
and documented rather than silently dropped.

## Definition of done (per CLAUDE.md)

- [ ] `pytest tests/test_attachments.py` green on Windows (minus the privilege-gated
      symlink-creation tests, item 4).
- [ ] POSIX behavior unchanged; no retrieval logic touched → no `eval_retrieval.py`
      delta.
- [ ] README/`guide/usage.md` mention that attachments now work on Windows if any
      user-facing caveat text currently says otherwise; `how-it-works.md` already
      carries the "known Windows gotcha" note — update it to "fixed" when this lands.
</content>
