# Rule backlog

Candidate rules we would accept as contributions. Each is a small,
self-contained PR (see [CONTRIBUTING.md](../CONTRIBUTING.md) for the 4-step
add-a-rule guide). These are deliberately the cases that need a judgment call
on precision, the high-certainty ones are already shipped. Pick one, open an
issue or a draft PR, and bring a fixture for both the buggy case and a clean
counterpart.

The theme: these move past per-layer config into training-loop and
shape-adjacency correctness, the bugs that survive ruff / mypy and only
surface at runtime.

## 1. missing-zero-grad (warn)

Forgetting `optimizer.zero_grad()` makes gradients accumulate across steps,
silently corrupting training. In a file that contains both `loss.backward()`
and `optimizer.step()`, fire when there is no `*.zero_grad()` call anywhere.

```python
for x, y in loader:
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()   # no zero_grad() anywhere -> grads accumulate
```

Open question: count `model.zero_grad()` and `set_to_none=True` forms as
present, so they do not false-positive.

## 2. missing-eval-mode (warn)

Running inference without `model.eval()` leaves Dropout and BatchNorm in
training mode, producing wrong or non-deterministic outputs. Conservative
trigger: a function literally named `evaluate` / `test` / `validate` /
`inference` that calls `model(` but never `*.eval()`.

Open question: this is the highest false-positive idea in the backlog. Is the
`def evaluate(...)` heuristic precise enough, or should it also require a
`with torch.no_grad():` anchor? Propose the narrowest trigger you can defend.

## 3. tensor-device-mismatch (warn)

`torch.zeros(...)` / `torch.randn(...)` default to CPU; if the model is
`.cuda()` / `.to(device)`, the forward pass crashes with a device mismatch.
Conservative trigger: file contains `.cuda()` or `.to(device)` on a model AND
a bare `torch.zeros(` / `torch.ones(` / `torch.randn(` / `torch.tensor(` with
no `device=` argument.

Open question: lots of code creates CPU tensors on purpose. Should this only
fire inside a `forward()` method?

## 4. sequential-conv-channel-mismatch (block)

Within a single `nn.Sequential(...)` literal, consecutive ConvXd calls where
`conv[n].in_channels != conv[n-1].out_channels` are a guaranteed shape error.

```python
nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.Conv2d(32, 128, 3),  # in_channels=32 != upstream out_channels=64
)
```

Open question: pooling and activations preserve channel count, norms and
flatten may not. Define which layer types are channel-preserving so the
adjacency check stays correct.

## 5. layernorm-shape-suspect (warn)

`nn.LayerNorm(normalized_shape)` must match the last dim of its input. A
regex approximation: in an `nn.Sequential`, fire when `nn.Linear(in, out)` is
followed by `nn.LayerNorm(N)` where `N != out`.

Open question: this is a regex approximation of a shape-propagation check (the
full version lives in the Neurarch app). What is the smallest adjacency that
stays high precision? A good first taste of the regex versus AST trade-off
this project is built around.

## 6. dropout-on-logits (warn)

Dropout applied to the final logits (after the last Linear, just before the
loss or softmax) randomly zeroes class scores, which is almost never intended.
In an `nn.Sequential`, fire when `nn.Dropout(...)` is the last or
second-to-last entry following the final `nn.Linear`.

Open question: the "final classifier" is approximated by the last Linear in
the Sequential. Bring a fixture for both the buggy and the legitimate
(dropout-before-Linear) case.

## 7. split-size-mismatch (warn)

`torch.split(x, [a, b, c], dim=...)` raises if `a + b + c` does not equal the
size of that dim. When both the source dim and the split sizes are literals,
an obvious mismatch is detectable.

Open question: the source dim has to be a visible literal, which is rare. Is
the precision worth the limited recall, or does this belong in the AST-based
v2? Good discussion for the roadmap.

## 8. publish-on-tag workflow (ci, not a rule)

The package is publish-ready (`prepublishOnly` runs the tests, lean tarball)
but there is no automation. Add `.github/workflows/publish.yml` that, on a
pushed `v*` tag, runs `npm ci`, `npm test`, then `npm publish --provenance`
with an `NPM_TOKEN` secret, guarded by
`if: github.repository == 'neurarch-ai/neurarch-lint'`.

Open question: require a GitHub Release rather than a bare tag? And do we want
npm provenance (needs `id-token: write`)? A first PR can stub the workflow;
wiring the secret is a maintainer step.

---

These are the regex-detectable slice. Checks that genuinely need tensor-shape
propagation across the whole graph (full shape-mismatch, parameter-explosion
estimates, cycle and orphan detection, layer-level GQA introspection) run in
the [Neurarch](https://neurarch.com) web app, and a v2 action that bundles the
typed-graph parser is on the roadmap.
