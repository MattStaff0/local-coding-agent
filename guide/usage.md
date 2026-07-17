# Usage

## The one interaction model

`cd` into the project you're working on and run `lca`. Every free-form line
goes to one directory-scoped agent that can read your **saved** files,
search the **official docs you indexed**, and (with your confirmation)
propose edits or run tests. There are no modes to switch.

```text
$ cd ~/projects/ml-assignment
$ lca
You: @train.py why is my validation loss shape wrong?
lca: What to notice: your final layer is Dense(1) (train.py:55) but y_val
     has two columns (train.py:67) — the docs say Dense's output must match
     the target's last dimension [1].
     Next check: print model.output_shape and y_val.shape.
     [1] tensorflow/keras-sequential § Output shape (docs v2.16, fetched 3d ago)
```

One-shot works too: `lca "what does groupby return before aggregation?"`

## Attaching files

Any prompt can attach current saved files with `@path` (the terminal cannot
see unsaved editor buffers):

```text
@src/train.py                whole file
@src/train.py:80             one line
@src/train.py:80-130         range (1-based, inclusive)
@"my dir/notes file.md"      quoted paths for spaces
lca --context notebooks/lesson.ipynb "why does cell 4 fail?"
```

An `@token` only attaches when it names a real file inside the project —
`@dataclass` or `@app.route` in a normal question stays plain text.
Notebooks render as numbered cells (`lesson.ipynb:cell-2`); ranges on a
notebook select cells. Secrets and binary artifacts (`.env`, keys, weights,
databases) are refused even when named explicitly.

## Teaching style

By default the agent **coaches**: concept → cited evidence → one next check.
Escalate naturally — "show me" gets a sketch, "give me the code" or "apply
it" gets the solution or a confirmation-gated edit, and "just give me the
answer" is honored immediately. Prefer full answers all the time? Set
`LCA_TEACHING_STYLE=direct` in `.env`.

Every claim is labeled: *the file says (path:line)*, *the docs say [n]*, or
*I infer*. Docs citations carry their version and fetch age, and you get a
warning when the docs version contradicts your project's pinned version.

## Session commands

```text
/help             all commands
/status           root, session size, docs scope, prompt revision
/root [path]      show or change the project root (change = fresh session)
/reset            clear this root's session
/sources          list indexed doc sources
/source <name>    constrain docs search (/source all to reset)
/export           save the last answer as a markdown study note
/exit             quit
```

## Operational commands

```text
lca doctor            offline health check (run it first when things break)
lca docs status       per-source corpus size, fetch age vs TTL, version
                      compatibility with YOUR project — fully offline
lca docs sync [src]   explicit bounded refresh + re-index (the only command
                      that fetches from the network)
lca --root <path>     use a different project root for this run
```

## Safety model

Reads are sandboxed to the project root (symlinks that escape are
rejected). Writes and commands always show a preview and ask `Apply? [y/N]`
— declining is a normal answer, not an error. Doc fetches only ever touch
the official origins declared in `sources.yaml`, and your code never gets
sent to a documentation site.
