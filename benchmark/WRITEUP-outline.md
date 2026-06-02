# Writeup outline: "The bugs ruff and mypy can't see"

A short post (600-900 words) that doubles as the credibility anchor for
neurarch-lint and neurarch-mcp. Goal: a serious ML engineer reads it and thinks
"this team actually understands the failure mode," then stars the repo. This is
the lint equivalent of graphiti's arXiv paper.

Do not lead with "visual canvas" or "visualization." Lead with the bug and the
GPU bill.

## 1. The hook (1 short paragraph)
A model that imports clean, passes `ruff` and `mypy`, and crashes 40 minutes into
a GPU run because `embed_dim=400` is not divisible by `num_heads=6`. Concrete,
real, costs money. (Reuse the `buggy_transformer.py` example.)

## 2. Why text linters structurally cannot catch this (1 paragraph)
`ruff` / `mypy` reason about syntax and types. The bug lives in the *tensor
structure*, the relationship between two integer args, which is type-valid and
syntactically fine. Frame the thesis: a model is a typed graph, not a wall of
text; the bug is only visible to something that reads it as a graph.

## 3. The class of bug, with a taxonomy (the meat)
Group the 20 rules into 3-4 buckets a reader recognizes:
- **Divisibility** (head-dim, GQA heads, GroupNorm channels)
- **Double-applied ops** (Softmax+CrossEntropy, Sigmoid+BCEWithLogits)
- **Invalid constructions** (zero features, stride 0, p>=1 dropout)
- **Ordering / numerics** (norm-after-activation, log-then-softmax, view-after-transpose)
One crisp example per bucket. This is what makes it read as "they get it."

## 4. The benchmark (the credibility number)
"We ran it across N pinned public PyTorch repos." Report:
- Total findings, blocking count, rule-types hit (from `report.md`).
- The verified true-positive rate on a sampled check (state the sample size).
- Pinned commits, so anyone can reproduce: link `repos.json` + `run-benchmark.mjs`.
Honesty is the point. A reproducible 47 beats an unverifiable 500.

## 5. Where it runs (adoption ask, low-key)
CI on every PR / pre-commit / SARIF in the Security tab. One `npx` line.
Then the honest scope line: v1 is 20 regex-detectable rules; the whole-graph
shape propagator (cross-layer mismatch, param explosion) needs the typed graph
and runs in the app + via [neurarch-mcp]. v2 will bundle it.

## 6. Close + two soft CTAs
- Star neurarch-lint if it would have caught a bug for you.
- If you want the same structural awareness inside Cursor / Claude Code, that is
  neurarch-mcp (one `npx` line). This cross-link is the funnel to the repo you
  most want stars on.

## Distribution (where to post)
- Hacker News ("Show HN: structural lint for PyTorch models, finds bugs ruff/mypy miss")
- r/MachineLearning, r/pytorch
- The PyTorch forums / discuss
- X/LinkedIn with the PR-comment screenshot as the visual
Time it with the demo GIF landing in both READMEs.
