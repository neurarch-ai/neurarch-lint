# Contributing to neurarch-lint

Thanks for helping catch PyTorch bugs before they cost a GPU hour. The most valuable contribution is a **new rule**, and the codebase is set up so adding one is a small, self-contained PR.

## Setup

```bash
git clone https://github.com/neurarch-ai/neurarch-lint
cd neurarch-lint
npm install
npm test            # vitest, runs every fixture against its expected rule set
node lint.mjs fixtures/rule_head_dim.py   # see the CLI in action
```

No build step. `lint.mjs` is a single self-contained ESM script with zero runtime dependencies.

## Add a rule in 4 steps

Every rule is one object in the `RULES` array in `lint.mjs`:

```js
{ id, title, severity, check(content, file) -> Finding[] }
```

A `Finding` is `{ file, line, rule, severity, message }`. Helpers `extractKwarg(args, names)` and `lineOf(content, index)` are already there.

1. **Add the rule** to `RULES` in `lint.mjs`. Keep `severity` honest: `block` only for things that crash or are provably wrong (a non-integer head_dim), `warn` for smells (Softmax before CrossEntropy).
2. **Add a fixture** `fixtures/rule_<your_rule>.py` containing the smallest code that should trigger it. Add a clean counterpart if the rule is prone to false positives.
3. **Add an expectation** in `lint.test.ts`: map your fixture to the rule ids that MUST fire (`expectedRules`) and, where useful, ids that MUST NOT (`forbiddenRules`).
4. **Document it** in the README "What it catches" table, and run `npm test`.

A good rule is **regex-detectable on the canonical class-instantiation form** and has a credible reason (a paper, the PyTorch docs, or a reproducible runtime error). If it needs real Python AST or shape propagation, open an issue instead, that is the v2 track.

## Rule quality bar

- **No false positives on the clean fixtures.** A linter that cries wolf gets disabled. When in doubt, downgrade `block` to `warn`, or narrow the regex.
- **Message names the fix, not just the problem.** "head_dim would be 76.80 (must be an integer)" beats "invalid attention config".
- **One rule, one concern.** Don't bundle.

## Pull requests

- Branch, commit, open a PR. Keep it to one rule or one fix.
- CI runs `npm test` and self-lints a clean fixture; both must stay green.
- Looking for a starter task? See issues labeled [`good first issue`](https://github.com/neurarch-ai/neurarch-lint/labels/good%20first%20issue).

## Scope

This repo is the regex-based v1 lint (CLI + GitHub Action). The full rule engine, shape propagator, and graph tooling live in the [Neurarch](https://neurarch.com) app and the [neurarch-mcp](https://github.com/neurarch-ai/neurarch-mcp) server. Rules that need the typed graph belong there; file an issue and we will route it.
