# Changelog

All notable changes to neurarch-lint are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

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
