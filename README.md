# neurarch-lint

[![CI](https://github.com/neurarch-ai/neurarch-lint/actions/workflows/ci.yml/badge.svg)](https://github.com/neurarch-ai/neurarch-lint/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/neurarch-ai/neurarch-lint.svg?style=social)](https://github.com/neurarch-ai/neurarch-lint/stargazers)
[![Try Neurarch](https://img.shields.io/badge/Neurarch-try_it-7c3aed)](https://neurarch.com)

Pre-flight structural lint for PyTorch models. Catches the bugs Cursor / Copilot can't see at the text layer, before they cost you a GPU hour.

> **Built by [Neurarch](https://neurarch.com).** Describe a task, an AI agent designs the model, the full lint engine checks it before you train, and you export clean `nn.Module` code you own. This repo is the CI and CLI slice of that engine (20 regex-detectable rules). **[See what Neurarch does →](https://neurarch.com)**

It runs in CI on every pull request, reads the `.py` files that changed, and blocks (or comments) when it finds a structural defect like an attention layer whose `embed_dim` is not divisible by `num_heads`. No Python install, no model loading, pure pattern analysis.

<!-- Demo (highest-converting element for stars): a screenshot of the PR comment neurarch-lint posts
     when it catches a head_dim bug. Save it to docs/pr-comment.png and uncomment:
![neurarch-lint commenting on a pull request](docs/pr-comment.png)
-->

### What ruff / flake8 / mypy won't catch

Style and type linters check syntax and types. They happily pass a model whose attention `embed_dim` isn't divisible by `num_heads`, or that applies `Softmax` right before `CrossEntropyLoss`. neurarch-lint checks the **tensor structure**, the class of bug that only surfaces once you actually run the model (or worse, halfway through training).

### Example

When the action finds something, it posts a comment like this on the pull request (this is the `--markdown` output, the same text the CLI prints locally):

> ### neurarch-lint
>
> Found **3** structural issues in this PR:
>
> :no_entry: **head-dim-divisibility** (block)
>   `models/encoder.py:18`
>   MultiheadAttention has embed_dim=384, num_heads=5. head_dim would be 76.80 (must be an integer).
>
> :no_entry: **groupnorm-channel-divisibility** (block)
>   `models/encoder.py:24`
>   GroupNorm has num_channels=16, num_groups=3. num_channels must be divisible by num_groups (16 / 3 is not an integer).
>
> :warning: **softmax-no-dim** (warn)
>   `models/head.py:9`
>   Softmax called without an explicit dim. The implicit dimension is ambiguous and deprecated; pass dim= (usually dim=-1).
>
> ---
> Full rule reference: <https://neurarch.com/rules.html>

The two `block` findings fail the check (exit code `1`); the `warn` is informational unless you set `fail-on-warn: true`.

And here is the kind of bug it blocks. The two snippets below are from [`examples/buggy_transformer.py`](examples/buggy_transformer.py) (left) and [`examples/fixed_transformer.py`](examples/fixed_transformer.py) (right). `ruff`, `flake8`, and `mypy` all pass the left; it imports cleanly and only crashes once you build the module on a GPU. The fix is a single character (`6` to `8`).

```python
# BLOCKED  (examples/buggy_transformer.py)        # PASSES  (examples/fixed_transformer.py)
class EncoderBlock(nn.Module):                     class EncoderBlock(nn.Module):
    def __init__(self, dim=400, num_heads=6):          def __init__(self, dim=400, num_heads=8):
        super().__init__()                                 super().__init__()
        # head_dim 400 / 6 = 66.67 (not int)               # head_dim 400 / 8 = 50 (int)
        self.attn = nn.MultiheadAttention(                 self.attn = nn.MultiheadAttention(
            embed_dim=400, num_heads=6)                        embed_dim=400, num_heads=8)
        # 400 not divisible by 6 groups                    # 400 divisible by 8 groups
        self.norm = nn.GroupNorm(                          self.norm = nn.GroupNorm(
            num_groups=6, num_channels=400)                    num_groups=8, num_channels=400)
```

The linter blocks the left at `head-dim-divisibility` and `groupnorm-channel-divisibility`; the right lints clean (0 findings, exit `0`).

## What it catches (v1)

| Rule | Severity | Trigger |
|------|----------|---------|
| `head-dim-divisibility` | block | `MultiheadAttention(embed_dim=384, num_heads=5)`. `head_dim` must be an integer. |
| `gqa-head-divisibility` | block | `GroupedQueryAttention(num_heads=32, num_kv_heads=7)`. `num_heads % num_kv_heads` must be 0. |
| `softmax-cross-entropy` | warn | `nn.Softmax` + `nn.CrossEntropyLoss` in the same file. CrossEntropy applies LogSoftmax internally; explicit Softmax double-applies. |
| `zero-features` | block | `nn.Linear(0, 10)` or `Conv2d(in_channels=0, ...)`. Fails at construction. |
| `bn-after-activation` | warn | A `BatchNormXd` / `LayerNorm` / `InstanceNorm` / `GroupNorm` / `RMSNorm` call wired right after an activation in `forward()`. Norm should be before the activation (Ioffe & Szegedy 2015). |
| `deep-no-residual` | warn | >= 8 weight-carrying layers (Linear / ConvXd / MultiheadAttention) and zero residual signals. Gradient signal degrades without skips (He et al. 2015). |
| `groupnorm-channel-divisibility` | block | `nn.GroupNorm(3, 16)`. `num_channels` must be divisible by `num_groups` or construction crashes. |
| `sigmoid-bce-with-logits` | warn | `nn.Sigmoid` + `nn.BCEWithLogitsLoss` in the same file. BCEWithLogitsLoss applies sigmoid internally; explicit Sigmoid double-applies. |
| `dropout-p-range` | block | `nn.Dropout(p=1.0)` (or any `p >= 1` / `p < 0`). `p >= 1` zeros the whole signal; `p` must be in `[0, 1)`. |
| `softmax-no-dim` | warn | `nn.Softmax()` / `F.softmax(x)` with no explicit `dim`. The implicit dimension is ambiguous and deprecated. |
| `conv-stride-zero` | block | `nn.Conv2d(..., stride=0)` (or any Conv/Pool). The output-size formula divides by stride, so `stride=0` is a guaranteed runtime error. |
| `negative-or-zero-kernel` | block | `nn.Conv2d(..., kernel_size=0)` or a negative `kernel_size` (Conv / Pool). `kernel_size` must be a positive integer. |
| `linear-bias-before-norm` | warn | `nn.Conv2d(..., bias=True)` / `nn.Linear(..., bias=True)` immediately before a `BatchNorm` in an `nn.Sequential`. BatchNorm has its own bias (beta); the layer bias is redundant. |
| `embedding-zero-size` | block | `nn.Embedding(0, 128)` or `embedding_dim=0`. A zero-row table or zero-width vectors crash or are useless. |
| `bceloss-without-sigmoid` | warn | `nn.BCELoss` with no `Sigmoid` anywhere in the file. BCELoss expects probabilities in `[0, 1]`; raw logits make the loss wrong (it can go negative). Add a Sigmoid or switch to `BCEWithLogitsLoss`. |
| `log-then-softmax` | warn | `torch.log(F.softmax(x))` or `F.softmax(x).log()`. Computing `log(softmax(x))` directly is numerically unstable; use `F.log_softmax`. |
| `view-after-transpose` | warn | `x.transpose(1, 2).view(...)` / `x.permute(...).view(...)`. The tensor is non-contiguous after transpose/permute, so `.view()` raises at runtime; use `.reshape()`. |
| `scheduler-step-before-optimizer` | warn | `scheduler.step()` on an earlier line than the first `optimizer.step()`. Stepping the scheduler first skips the initial learning rate (PyTorch warns about this). |
| `relu-then-softmax` | warn | `nn.ReLU()` directly before `nn.Softmax` / `nn.LogSoftmax` in an `nn.Sequential`. ReLU clamps logits to non-negative and distorts the output distribution. |
| `conv-padding-negative` | block | `nn.Conv2d(..., padding=-1)` (or any Conv/Pool). Negative padding is invalid and raises at construction. |

This CLI / Action ships **20 regex-detectable structural rules**. The full [Neurarch](https://neurarch.com) web app adds propagator-based checks that need the typed architecture graph (whole-graph shape-mismatch detection, parameter-explosion estimates, cycle and orphan detection, layer-level GQA introspection). Those cannot be expressed as a text regex and run inside the app; a v2 action that bundles the typed-graph parser is on the roadmap.

Full rationale and fixes for each rule: [docs/RULES.md](docs/RULES.md). Online catalog: <https://neurarch.com/rules.html>

## Use in a GitHub workflow

```yaml
# .github/workflows/structural-lint.yml
name: Structural lint
on:
  pull_request:
    paths: ['**/*.py']

permissions:
  contents: read
  pull-requests: write   # only needed if comment-on-pr: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0          # need full history for the PR diff
      - uses: neurarch-ai/neurarch-lint@v1
        with:
          comment-on-pr: true
          fail-on-warn:  false
```

By default it lints **only the files changed in the PR**. To lint a fixed path instead:

```yaml
      - uses: neurarch-ai/neurarch-lint@v1
        with:
          paths: 'models/ src/networks/'
```

### Inputs

| Input | Default | Description |
|---|---|---|
| `paths` | `''` (changed files) | Space-separated `.py` files or directories to lint. |
| `fail-on-warn` | `false` | Exit non-zero on warnings, not just blockers. |
| `comment-on-pr` | `true` | Post findings as a PR comment (needs `pull-requests: write`). |

### Outputs

| Output | Description |
|---|---|
| `findings-count` | Total issues found. |
| `blocking-count` | Issues with `severity=block`. |

### GitHub Code Scanning (SARIF)

The CLI can emit a [SARIF 2.1.0](https://json.schemastore.org/sarif-2.1.0.json) report and upload it to GitHub Code Scanning, so findings show up in the **Security** tab and inline on the PR diff. This path needs no PR-comment permission, only `security-events: write`:

```yaml
# .github/workflows/structural-scan.yml
name: Structural scan
on:
  pull_request:
    paths: ['**/*.py']

permissions:
  contents: read
  security-events: write   # required to upload SARIF

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npx neurarch-lint --sarif --dir=. > results.sarif || true
      - uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
```

`--dir=.` scans every `.py` file in the repo; pass explicit files (`--sarif a.py b.py`) to scope it. The `|| true` lets the upload run even when a blocking finding sets a non-zero exit code; the findings still surface as Code Scanning alerts. If you want the build to fail on blockers, add the [Action](#use-in-a-github-workflow) (or a plain `npx neurarch-lint --dir=.` step) as a separate gating step.

## Local use

No install needed, the linter is a single self-contained Node script:

```bash
node lint.mjs path/to/model.py
node lint.mjs --dir=models
node lint.mjs --json file.py        # machine-readable
node lint.mjs --markdown file.py    # PR-comment style
```

Exit codes: `0` clean, `1` blocking issue found, `2` usage error.

See [examples/](examples/) for a buggy model and its lint output.

The CLI also has two CI-oriented formats: `--github` emits [GitHub Actions annotations](https://docs.github.com/actions/using-workflows/workflow-commands-for-github-actions) (inline PR comments, no permissions needed), and `--sarif` emits a [SARIF 2.1.0](#github-code-scanning-sarif) report for GitHub Code Scanning.

(Once published to npm you will also be able to run `npx neurarch-lint file.py`.)

## Use with pre-commit

Catch the bugs at commit time, before they ever reach CI. Add to your `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/neurarch-ai/neurarch-lint
    rev: v1
    hooks:
      - id: neurarch-lint
```

Every `git commit` that touches a `.py` file now runs the structural lint locally. Blocking issues stop the commit; warnings don't.

## What's not in v1

- **No real Python AST.** The regex catches the canonical class-instantiation form (`nn.MultiheadAttention(embed_dim=..., num_heads=...)`). It misses dynamic construction (`AttentionType(**config)`).
- **No shape propagation.** The web app and [neurarch-mcp](https://github.com/neurarch-ai/neurarch-mcp) have the full propagator; this action will gain it in v2 once the TypeScript rule code is bundled to ESM.
- **No fix suggestions in the comment.** Just the location and the rule.

## About Neurarch

[Neurarch](https://neurarch.com) is where the rules in this repo come from. It is a model-design environment built on one idea: your architecture is a **typed graph**, so an AI agent (and a linter) can actually reason about it instead of guessing at text.

- **Describe, then design.** Type the problem ("classify support tickets into 5 categories") or upload a CSV; the agent picks the layers, wires them, and propagates tensor shapes.
- **Lint before you train.** The full engine (this repo is the 20-rule regex-detectable slice) catches shape mismatches, head-dim bugs, and missing residuals before a GPU bill.
- **Export and own it.** Clean PyTorch / TensorFlow / ONNX, no runtime dependency on Neurarch.
- **[neurarch-mcp](https://github.com/neurarch-ai/neurarch-mcp):** the same graph awareness inside Claude Code, Cursor, and other MCP agents.

**[Try Neurarch → neurarch.com](https://neurarch.com)**

## Contributing

A new rule is a small, self-contained PR. See [CONTRIBUTING.md](./CONTRIBUTING.md) for the 4-step "add a rule" guide, and [good first issues](https://github.com/neurarch-ai/neurarch-lint/labels/good%20first%20issue) to start.

## Star this repo

If neurarch-lint caught a bug before it cost you a GPU hour, a ⭐ helps other ML engineers find it.

## License

MIT. See [LICENSE](./LICENSE).
