# neurarch-lint examples

Two files you can lint right after cloning to see the value in ten seconds:

- `buggy_transformer.py`: a realistic transformer-ish classifier with four deliberate structural bugs (a non-divisible attention head dim, a GroupNorm whose channels are not divisible by its groups, a `Softmax` with no `dim`, and an explicit `Sigmoid` that double-applies with `BCEWithLogitsLoss`).
- `fixed_transformer.py`: the same model with all four bugs fixed. Lints clean (0 findings, exit 0).

## Run it

```bash
node lint.mjs examples/buggy_transformer.py
node lint.mjs --markdown examples/buggy_transformer.py   # PR-comment style
```

## Actual output

Running `node lint.mjs examples/buggy_transformer.py` prints (exit code `1`, because two findings are blocking):

```
neurarch-lint: 4 issues found.

  [BLOCK] examples/buggy_transformer.py:27  head-dim-divisibility
         MultiheadAttention has embed_dim=400, num_heads=6. head_dim would be 66.67 (must be an integer).
  [BLOCK] examples/buggy_transformer.py:30  groupnorm-channel-divisibility
         GroupNorm has num_channels=400, num_groups=6. num_channels must be divisible by num_groups (400 / 6 is not an integer).
  [WARN] examples/buggy_transformer.py:51  sigmoid-bce-with-logits
         nn.Sigmoid combined with nn.BCEWithLogitsLoss double-applies the sigmoid. BCEWithLogitsLoss expects raw logits; drop the explicit Sigmoid (or use BCELoss).
  [WARN] examples/buggy_transformer.py:49  softmax-no-dim
         Softmax called without an explicit dim. The implicit dimension is ambiguous and deprecated; pass dim= (usually dim=-1).

Rule reference: https://neurarch.com/rules.html
```

The corrected file is clean:

```
$ node lint.mjs examples/fixed_transformer.py
neurarch-lint: no structural issues found.
```

Full rationale and the fix for each rule: [../docs/RULES.md](../docs/RULES.md). The full 22-rule engine (including shape propagation) runs in the [Neurarch](https://neurarch.com) web app.
