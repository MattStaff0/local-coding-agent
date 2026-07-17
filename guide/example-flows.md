# Example flows

Transcript-style walkthroughs of the behaviors that define this tool. Your
exact wording will vary with the model you run; the *shape* should not.

## 1. Debug your file, coached

```text
You: @train.py why is my validation loss shape wrong?
lca: What to notice: the file says train.py:55 builds Dense(1), but
     train.py:67 stacks y_val with two columns. The docs say the output
     layer's units must match the target's last dimension [1].
     Next check: print model.output_shape and y_val.shape.
     [1] tensorflow/keras-sequential § Output shape (docs v2.16, fetched 3d ago)
```

## 2. Escalate when you want more

```text
You: show me
lca: Sketch: either y_val = y_val[:, 0] (if column 2 is a duplicate) or
     Dense(2) (if it's a real second label). Which is true in your data?

You: it's a duplicate — apply the reshape
lca: edit_file train.py
     --- a/train.py
     +++ b/train.py
     -y_val = np.stack([labels, labels], axis=1)
     +y_val = labels
     Apply? [y/N]:
```

Declining (`n`) is fine — the agent returns to explaining instead of
re-proposing the same diff.

## 3. Skip the coaching when you want to

```text
You: just give me the code, no hints
lca: (full snippet immediately — still citing train.py lines and docs)
```

## 4. Docs question, version-aware

```text
You: why did df.append stop working after I moved machines?
lca: warning: these docs are for pandas 3.0, but this project has
     pandas 2.2.1 (lock) — behavior may differ.
     The docs say DataFrame.append was removed in 3.0 [1]; in 2.x it is
     deprecated but present. On YOUR pinned version the call still runs
     with a FutureWarning; use pd.concat either way.
     [1] pandas/whatsnew § Removals (docs v3.0, fetched 2d ago)
```

That warning is the point of version-aware docs: **fresh** documentation is
not automatically **compatible** documentation.

## 5. Honest misses

```text
You: where is normalize_embeddings_v2 defined here?
lca: I grepped for normalize_embeddings_v2 and normalize_embeddings and
     found neither in this project. The docs index has nothing under that
     name either. I infer you may mean embed_batch (src/rag.py:252) — but
     that part is inference, not evidence.
```

No invented file paths, no fabricated line numbers — a miss is reported as
a miss, with the searches that produced it.

## 6. Working across projects

```text
$ cd ~/projects/scraper && lca        # agent reads THIS project
You: /root ~/projects/other-thing    # fresh session, other project
You: /source numpy                    # docs answers constrained to numpy
$ lca docs status                     # is my corpus fresh AND compatible?
```

Sessions and history are kept per project root; changing root never leaks
context from the previous project.
