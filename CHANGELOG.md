# Changelog

All notable changes to neurarch-lint are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

- 6 new structural rules (20 total): `bceloss-without-sigmoid` (warn, `nn.BCELoss` with no `Sigmoid` in the file, raw logits fed to a probability loss), `log-then-softmax` (warn, numerically unstable `torch.log(F.softmax(...))` / `F.softmax(...).log()`, use `F.log_softmax`), `view-after-transpose` (warn, `.view()` chained on a non-contiguous `.transpose(...)` / `.permute(...)` result, use `.reshape()`), `scheduler-step-before-optimizer` (warn, `scheduler.step()` on an earlier line than the first `optimizer.step()`), `relu-then-softmax` (warn, `ReLU` directly before `Softmax` / `LogSoftmax` in an `nn.Sequential`, clamps logits non-negative), `conv-padding-negative` (block, Conv/Pool with negative `padding`).
- 4 new structural rules (14 total): `conv-stride-zero` (block, Conv/Pool `stride=0`), `negative-or-zero-kernel` (block, Conv/Pool `kernel_size` zero or negative), `linear-bias-before-norm` (warn, `bias=True` on a Conv/Linear directly before a `BatchNorm` in an `nn.Sequential`), `embedding-zero-size` (block, `nn.Embedding` with `num_embeddings=0` or `embedding_dim=0`).
- `--version` / `-v`: prints `neurarch-lint X.Y.Z` and exits 0.
- `--help` / `-h`: now prints a full usage block (flags, exit codes, rule-reference link) to stdout and exits 0, instead of a one-line usage error. The no-args case stays a usage error (exit 2).
- Performance / robustness: `collectPyFiles` also skips `.git`, `dist`, `build`, `.venv`, `site-packages`, `.mypy_cache`, `.pytest_cache`, and tolerates unreadable directories instead of crashing the run. Files larger than 2 MB are skipped with an `info` finding rather than read.
- npm publish readiness: `prepublishOnly` runs the test suite, `pack:check` previews the tarball, and a `.npmignore` keeps fixtures / examples / docs / tests out of the published package (lean 4-file tarball driven by the `files` allowlist).
- CI: test matrix across Node 18 / 20 / 22, a `dogfood` job that runs the Action's CLI against the examples end-to-end, and a `sarif` job that uploads a SARIF report to Code Scanning (guarded to the canonical repo).
- `--github` output: GitHub Actions workflow-command annotations (`::error` / `::warning`) so findings render inline on the PR diff and the Checks summary with no PR-comment permission. The Action now emits these alongside the human / markdown / JSON runs.
- `--sarif` output: SARIF 2.1.0 document for upload to GitHub Code Scanning. The rule list is derived from the `RULES` array so it never drifts.
- `docs/RULES.md`: full catalog of all 14 rules (rationale, triggering snippet, fix, reference) plus a pointer to the 8 propagator-only rules in the web app.
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
