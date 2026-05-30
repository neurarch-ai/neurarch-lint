# Changelog

All notable changes to neurarch-lint are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

- `--github` output: GitHub Actions workflow-command annotations (`::error` / `::warning`) so findings render inline on the PR diff and the Checks summary with no PR-comment permission. The Action now emits these alongside the human / markdown / JSON runs.
- `--sarif` output: SARIF 2.1.0 document for upload to GitHub Code Scanning. The rule list is derived from the `RULES` array so it never drifts.
- `docs/RULES.md`: full catalog of all 10 rules (rationale, triggering snippet, fix, reference) plus a pointer to the 12 propagator-only rules in the web app.
- `examples/`: a `buggy_transformer.py` that trips 4 rules, a clean `fixed_transformer.py`, and a README with the real captured lint output.
- pre-commit hook support (`.pre-commit-hooks.yaml`).
- `CONTRIBUTING.md` with a 4-step "add a rule" guide.
- 4 new structural rules (10 total): `groupnorm-channel-divisibility` (block, `num_channels` not divisible by `num_groups`), `sigmoid-bce-with-logits` (warn, explicit Sigmoid double-applied with `BCEWithLogitsLoss`), `dropout-p-range` (block, Dropout `p` outside `[0, 1)`), `softmax-no-dim` (warn, Softmax / `F.softmax` with no explicit `dim`).
- README: concrete PR-comment example block under the new "Example" heading.

## [v1] - 2026-05-28

Initial public release. Extracted from the Neurarch monorepo.

- 6 regex-detectable structural rules: `head-dim-divisibility`, `gqa-head-divisibility`, `softmax-cross-entropy`, `zero-features`, `bn-after-activation`, `deep-no-residual`.
- CLI: `node lint.mjs FILES...` with `--dir=`, `--json`, `--markdown`.
- GitHub Action: `uses: neurarch-ai/neurarch-lint@v1`, lints PR-changed `.py` files, optional PR comment.
- Exit codes: `0` clean, `1` blocking issue, `2` usage error.
