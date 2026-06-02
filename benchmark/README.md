# neurarch-lint benchmark

A reproducible "what does neurarch-lint actually catch in the wild" number, the
credibility anchor that makes the repo read as serious (graphiti has its arXiv
paper; this is the lint equivalent).

## Run it

```bash
# from the neurarch-lint repo root, with the linter at ./lint.mjs
node benchmark/run-benchmark.mjs --lint ./lint.mjs
# writes benchmark/report.md
```

Each repo in `repos.json` is shallow-cloned at its pinned ref, linted with
`lint.mjs --json`, and aggregated by rule / severity / repo. Repos that fail to
clone or lint are reported as **skipped** and excluded from totals (never
silently dropped).

## Make it reproducible BEFORE quoting it

`repos.json` currently pins **branch names** (`main` / `master`) so it runs out
of the box. A branch moves, so the number is not yet reproducible. Before you
publish a number:

1. Replace each `"ref"` with the **commit SHA** the run actually used
   (`git -C .cache/<name> rev-parse HEAD` after a `--keep` run).
2. Re-run with `--keep` so the clones are cached and the number is stable.
3. Commit `repos.json` (pinned) + `report.md` together.

## Honesty checklist (do not skip)

The v1 linter is regex-based and can over-match (e.g. dynamic construction).
Before any number goes into the README, a deck, or a tweet:

- [ ] Spot-check a random sample of findings (say 30) and record the
      true-positive rate.
- [ ] Quote the **verified** count, or quote raw + the sampled precision
      ("X findings, 90% true-positive on a 30-finding sample").
- [ ] State N (repos) and that commits are pinned.
- [ ] Never round up. See the project's no-signal-inflation rule.

A small, honest, reproducible number ("47 real structural bugs across 10 pinned
repos, ruff/mypy clean") beats a big unverifiable one every time, especially
with the kind of engineer who stars a linter.

## Curating `repos.json`

Big, heavily-maintained libraries (torchvision, detectron2) tend to be clean, so
they make the floor honest but add few findings. Paper-implementation and
tutorial repos surface more. Aim for a mix so the number is credible, not
cherry-picked, and say in the writeup how you picked them.
