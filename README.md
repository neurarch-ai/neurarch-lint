# neurarch-lint

[![CI](https://github.com/neurarch-ai/neurarch-lint/actions/workflows/ci.yml/badge.svg)](https://github.com/neurarch-ai/neurarch-lint/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/neurarch-ai/neurarch-lint.svg?style=social)](https://github.com/neurarch-ai/neurarch-lint/stargazers)

Pre-flight structural lint for PyTorch models. Catches the bugs Cursor / Copilot can't see at the text layer, before they cost you a GPU hour.

It runs in CI on every pull request, reads the `.py` files that changed, and blocks (or comments) when it finds a structural defect like an attention layer whose `embed_dim` is not divisible by `num_heads`. No Python install, no model loading, pure pattern analysis.

<!-- Demo (highest-converting element for stars): a screenshot of the PR comment neurarch-lint posts
     when it catches a head_dim bug. Save it to docs/pr-comment.png and uncomment:
![neurarch-lint commenting on a pull request](docs/pr-comment.png)
-->

## What it catches (v1)

| Rule | Severity | Trigger |
|------|----------|---------|
| `head-dim-divisibility` | block | `MultiheadAttention(embed_dim=384, num_heads=5)`. `head_dim` must be an integer. |
| `gqa-head-divisibility` | block | `GroupedQueryAttention(num_heads=32, num_kv_heads=7)`. `num_heads % num_kv_heads` must be 0. |
| `softmax-cross-entropy` | warn | `nn.Softmax` + `nn.CrossEntropyLoss` in the same file. CrossEntropy applies LogSoftmax internally; explicit Softmax double-applies. |
| `zero-features` | block | `nn.Linear(0, 10)` or `Conv2d(in_channels=0, ...)`. Fails at construction. |
| `bn-after-activation` | warn | A `BatchNormXd` / `LayerNorm` / `InstanceNorm` / `GroupNorm` / `RMSNorm` call wired right after an activation in `forward()`. Norm should be before the activation (Ioffe & Szegedy 2015). |
| `deep-no-residual` | warn | >= 8 weight-carrying layers (Linear / ConvXd / MultiheadAttention) and zero residual signals. Gradient signal degrades without skips (He et al. 2015). |

The full Neurarch rule set is 22 checks (5 guardrail gates + 17 advisor rules). v1 of this action covers the 6 most regex-detectable ones. The propagator-based checks (full shape mismatch, layer-level GQA introspection) live in the [Neurarch](https://neurarch.com) web app and are on the roadmap for a v2 action that bundles the typed-graph parser.

Full rule catalog: <https://neurarch.com/rules.html>

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

## Local use

No install needed, the linter is a single self-contained Node script:

```bash
node lint.mjs path/to/model.py
node lint.mjs --dir=models
node lint.mjs --json file.py        # machine-readable
node lint.mjs --markdown file.py    # PR-comment style
```

Exit codes: `0` clean, `1` blocking issue found, `2` usage error.

(Once published to npm you will also be able to run `npx neurarch-lint file.py`.)

## What's not in v1

- **No real Python AST.** The regex catches the canonical class-instantiation form (`nn.MultiheadAttention(embed_dim=..., num_heads=...)`). It misses dynamic construction (`AttentionType(**config)`).
- **No shape propagation.** The web app and [neurarch-mcp](https://github.com/neurarch-ai/neurarch-mcp) have the full propagator; this action will gain it in v2 once the TypeScript rule code is bundled to ESM.
- **No fix suggestions in the comment.** Just the location and the rule.

## Related

- **[neurarch-mcp](https://github.com/neurarch-ai/neurarch-mcp)** gives your AI coding agent (Claude Code, Cursor, ...) structural awareness of a model graph.
- **[Neurarch](https://neurarch.com)** is the visual editor where you design, lint, train, and export the model. Same rule engine, full propagator.

## Star this repo

If neurarch-lint caught a bug before it cost you a GPU hour, a ⭐ helps other ML engineers find it.

## License

MIT. See [LICENSE](./LICENSE).
