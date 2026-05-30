# neurarch-lint rule catalog

The twenty rules this CLI / Action ships, with the failure mode each one prevents,
a minimal triggering snippet, and the fix. The rule ids, severities, and
messages here are pulled straight from the `RULES` array in `lint.mjs`, so this
catalog matches the code.

Severity legend: `block` exits the CLI non-zero (fails the check); `warn` is
informational unless you set `fail-on-warn: true` on the Action.

| Rule | Severity |
|------|----------|
| [head-dim-divisibility](#head-dim-divisibility) | block |
| [gqa-head-divisibility](#gqa-head-divisibility) | block |
| [groupnorm-channel-divisibility](#groupnorm-channel-divisibility) | block |
| [zero-features](#zero-features) | block |
| [dropout-p-range](#dropout-p-range) | block |
| [conv-stride-zero](#conv-stride-zero) | block |
| [negative-or-zero-kernel](#negative-or-zero-kernel) | block |
| [embedding-zero-size](#embedding-zero-size) | block |
| [conv-padding-negative](#conv-padding-negative) | block |
| [softmax-cross-entropy](#softmax-cross-entropy) | warn |
| [sigmoid-bce-with-logits](#sigmoid-bce-with-logits) | warn |
| [bceloss-without-sigmoid](#bceloss-without-sigmoid) | warn |
| [softmax-no-dim](#softmax-no-dim) | warn |
| [log-then-softmax](#log-then-softmax) | warn |
| [linear-bias-before-norm](#linear-bias-before-norm) | warn |
| [bn-after-activation](#bn-after-activation) | warn |
| [relu-then-softmax](#relu-then-softmax) | warn |
| [view-after-transpose](#view-after-transpose) | warn |
| [scheduler-step-before-optimizer](#scheduler-step-before-optimizer) | warn |
| [deep-no-residual](#deep-no-residual) | warn |

---

## head-dim-divisibility

**Severity:** block

`nn.MultiheadAttention` splits `embed_dim` evenly across `num_heads`, so each
head gets `head_dim = embed_dim / num_heads` channels. When `embed_dim` is not
divisible by `num_heads` the split is not an integer and PyTorch raises
`embed_dim must be divisible by num_heads` at construction time. It never
trains; it crashes when the module is built. The cost is a failed run, often
discovered only after you have provisioned the GPU.

```python
import torch.nn as nn

# 384 / 5 = 76.8, not an integer.
attn = nn.MultiheadAttention(embed_dim=384, num_heads=5)
```

**Fix:** pick a `num_heads` that divides `embed_dim` (for 384: 6 gives head_dim
64, 8 gives 48), or round `embed_dim` to a multiple of `num_heads`.

```python
attn = nn.MultiheadAttention(embed_dim=384, num_heads=6)  # head_dim = 64
```

**Reference:** PyTorch [`nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) requires `embed_dim` divisible by `num_heads`.

---

## gqa-head-divisibility

**Severity:** block

Grouped-query and multi-query attention share a smaller set of key/value heads
across the query heads. Each KV head serves `num_heads / num_kv_heads` query
heads, so `num_heads` must be divisible by `num_kv_heads`. A non-integer ratio
means the query heads cannot be partitioned into equal groups, and the layer
fails to build (or silently misroutes the grouping in a hand-rolled
implementation).

```python
# 32 % 7 = 4, not 0: the 32 query heads do not split evenly into 7 KV groups.
attn = GroupedQueryAttention(num_heads=32, num_kv_heads=7)
```

**Fix:** choose a `num_kv_heads` that divides `num_heads` (for 32: 8 gives
groups of 4, 4 gives groups of 8).

```python
attn = GroupedQueryAttention(num_heads=32, num_kv_heads=8)  # 4 queries per KV head
```

**Reference:** Ainslie et al. 2023, [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245).

---

## groupnorm-channel-divisibility

**Severity:** block

`nn.GroupNorm` partitions `num_channels` into `num_groups` equal groups and
normalizes within each. If `num_channels` is not divisible by `num_groups`,
PyTorch raises `num_channels must be divisible by num_groups` at construction.
Like the attention head dim, this is a guaranteed crash, not a slow degradation.

```python
import torch.nn as nn

# 16 / 3 is not an integer.
norm = nn.GroupNorm(num_groups=3, num_channels=16)
```

**Fix:** use a `num_groups` that divides `num_channels` (for 16: 2, 4, 8, or 16).

```python
norm = nn.GroupNorm(num_groups=4, num_channels=16)
```

**Reference:** PyTorch [`nn.GroupNorm`](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html) requires `num_channels` divisible by `num_groups`.

---

## zero-features

**Severity:** block

A `Linear` or `ConvXd` layer with a zero in- or out-feature count has no weights
to learn and fails at construction. This usually slips in when a dimension is
computed (`hidden = a - b`) and the arithmetic collapses to zero, or a config
default is left unset.

```python
import torch.nn as nn

# A computed dim that collapsed to 0.
proj = nn.Linear(0, 10)
```

**Fix:** ensure both dimensions are positive; trace back the expression that
produced the zero.

```python
proj = nn.Linear(256, 10)
```

**Reference:** PyTorch [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) / [`nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) require positive feature / channel counts.

---

## dropout-p-range

**Severity:** block

`nn.Dropout` takes a probability `p` in `[0, 1)`. At `p >= 1` it zeros the entire
signal during training, so nothing propagates and the network cannot learn; at
`p < 0` the value is invalid. PyTorch rejects out-of-range `p` with
`dropout probability has to be between 0 and 1`.

```python
import torch.nn as nn

# Zeros 100% of activations: the layer outputs all zeros in train mode.
drop = nn.Dropout(p=1.0)
```

**Fix:** use a probability strictly below 1 (typical values are 0.1 to 0.5).

```python
drop = nn.Dropout(p=0.5)
```

**Reference:** PyTorch [`nn.Dropout`](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) requires `p` in `[0, 1)`.

---

## conv-stride-zero

**Severity:** block

Convolution and pooling layers compute their output size with a formula that
divides by `stride`. A `stride=0` therefore divides by zero and is a guaranteed
runtime error, not a slow degradation. It usually slips in when a stride is
computed from a config value that collapsed to zero.

```python
import torch.nn as nn

# stride=0 divides by zero in the output-size formula.
conv = nn.Conv2d(3, 16, kernel_size=3, stride=0)
```

**Fix:** use a stride of at least 1 (the default is usually 1 for convs, the
kernel size for pools).

```python
conv = nn.Conv2d(3, 16, kernel_size=3, stride=1)
```

**Reference:** PyTorch [`nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) output-size formula divides by `stride`.

---

## negative-or-zero-kernel

**Severity:** block

The `kernel_size` of a convolution or pooling layer is the size of the sliding
window and must be a positive integer. A `kernel_size=0` (or a negative value)
describes no window at all and fails at construction.

```python
import torch.nn as nn

# kernel_size=0 is not a valid window.
conv = nn.Conv2d(3, 16, kernel_size=0)
```

**Fix:** use a positive `kernel_size` (1, 3, 5, ... are typical).

```python
conv = nn.Conv2d(3, 16, kernel_size=3)
```

**Reference:** PyTorch [`nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) / pooling layers require a positive `kernel_size`.

---

## embedding-zero-size

**Severity:** block

`nn.Embedding` builds a lookup table of shape `(num_embeddings, embedding_dim)`.
A `num_embeddings=0` makes a zero-row table (no token can be looked up) and an
`embedding_dim=0` makes zero-width vectors. Both are useless and crash or
produce empty tensors. This mirrors `zero-features` for the embedding table.

```python
import torch.nn as nn

# Zero-row table: there is no vocabulary to index into.
embed = nn.Embedding(num_embeddings=0, embedding_dim=128)
```

**Fix:** set `num_embeddings` to the real vocabulary size and `embedding_dim` to
a positive width.

```python
embed = nn.Embedding(num_embeddings=30000, embedding_dim=128)
```

**Reference:** PyTorch [`nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) requires a positive table size and vector width.

---

## softmax-cross-entropy

**Severity:** warn

`nn.CrossEntropyLoss` applies `LogSoftmax` to its input internally and expects
raw logits. If you apply an explicit `Softmax` (or `F.softmax`) to the model's
output before passing it to `CrossEntropyLoss`, the softmax is applied twice.
The doubly-squashed distribution has tiny gradients, so training is far slower
and the model often plateaus at poor accuracy. It does not crash, which is why
it is a `warn`: it silently wastes the run.

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=-1)   # double-applied below

    def forward(self, x):
        return self.softmax(self.head(x))   # already a probability distribution

loss_fn = nn.CrossEntropyLoss()             # applies LogSoftmax again
```

**Fix:** return raw logits from the model and let `CrossEntropyLoss` do the
softmax. Drop the explicit `Softmax`.

```python
    def forward(self, x):
        return self.head(x)                 # raw logits
```

**Reference:** PyTorch [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) documents that the input is expected to contain unnormalized logits.

---

## sigmoid-bce-with-logits

**Severity:** warn

`nn.BCEWithLogitsLoss` combines a sigmoid and the binary cross-entropy loss in
one numerically stable step and expects raw logits. An explicit `nn.Sigmoid`
(or `torch.sigmoid` / `F.sigmoid`) on the output before this loss applies the
sigmoid twice, which flattens gradients and breaks training. It is the binary
analog of `softmax-cross-entropy`.

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()         # double-applied below

    def forward(self, x):
        return self.sigmoid(self.fc(x))

loss_fn = nn.BCEWithLogitsLoss()            # applies sigmoid again
```

**Fix:** return raw logits and keep `BCEWithLogitsLoss` (preferred, more stable),
or drop `BCEWithLogitsLoss` for plain `nn.BCELoss` if you must keep the explicit
sigmoid.

```python
    def forward(self, x):
        return self.fc(x)                   # raw logits, paired with BCEWithLogitsLoss
```

**Reference:** PyTorch [`nn.BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) documents that it expects logits, not probabilities.

---

## softmax-no-dim

**Severity:** warn

Calling `nn.Softmax()` or `F.softmax(x)` without an explicit `dim` leaves the
normalization axis implicit. PyTorch warns that the implicit dim is ambiguous
and deprecated, and the inferred axis is frequently not the one you intended
(normalizing over the batch instead of the feature axis, for example), which
silently corrupts the output.

```python
import torch.nn as nn

# No dim=: PyTorch warns and guesses the axis.
softmax = nn.Softmax()
```

**Fix:** pass `dim` explicitly, usually `dim=-1` for the last (feature) axis.

```python
softmax = nn.Softmax(dim=-1)
```

**Reference:** PyTorch [`nn.Softmax`](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) deprecates the implicit dimension.

---

## linear-bias-before-norm

**Severity:** warn

A `BatchNorm` layer learns its own shift parameter (beta), so any bias on the
`Conv` or `Linear` immediately before it is redundant: the normalization
subtracts the running mean and then re-adds beta, which cancels the upstream
bias. The extra bias just wastes parameters and a little compute. This rule is
deliberately conservative and only fires on the explicit `nn.Sequential(...)`
adjacency, where a `Conv` / `Linear` with `bias=True` is directly followed by a
`BatchNormXd`, to avoid false positives.

```python
import torch.nn as nn

block = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, bias=True),  # redundant bias
    nn.BatchNorm2d(16),
    nn.ReLU(),
)
```

**Fix:** set `bias=False` on the layer that feeds the BatchNorm.

```python
block = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(),
)
```

**Reference:** the BatchNorm beta absorbs an upstream bias; see PyTorch [`nn.BatchNorm2d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) and common ResNet implementations that set `bias=False` before norm.

---

## bn-after-activation

**Severity:** warn

Normalization layers (`BatchNorm`, `LayerNorm`, `InstanceNorm`, `GroupNorm`,
`RMSNorm`) are designed to stabilize the pre-activation distribution. When a
norm is wired immediately after an activation in `forward()`, it normalizes the
already-rectified signal instead, which loses the intended effect on the
internal covariate shift and tends to train worse than the conventional
norm-then-activate order.

```python
import torch.nn as nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)   # activation
        x = self.bn(x)      # normalization AFTER the activation
        return x
```

**Fix:** normalize before the activation.

```python
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)      # normalize the pre-activation distribution
        x = self.relu(x)
        return x
```

**Reference:** Ioffe & Szegedy 2015, [Batch Normalization](https://arxiv.org/abs/1502.03167), which places normalization before the nonlinearity.

---

## deep-no-residual

**Severity:** warn

A network with eight or more weight-carrying layers (`Linear`, `ConvXd`,
`MultiheadAttention`, `GroupedQueryAttention`) and no residual / skip / additive
merge is prone to vanishing or exploding gradients: the signal degrades as it
propagates through depth, and accuracy often gets worse, not better, as you add
layers. Residual connections give the gradient a short path back and are the
standard fix above this depth.

```python
import torch.nn as nn
import torch.nn.functional as F

class DeepMLP(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        # ... eight more nn.Linear layers, no skip connection ...
        self.classifier = nn.Linear(dim, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # ... no `x = x + ...` anywhere ...
        return self.classifier(x)
```

**Fix:** add residual connections (`x = x + block(x)`) around blocks, or adopt a
ResNet / pre-norm Transformer style that carries skips by construction.

```python
    def forward(self, x):
        x = x + F.relu(self.fc1(x))   # residual: gradient gets a short path back
        return self.classifier(x)
```

**Reference:** He et al. 2015, [Deep Residual Learning](https://arxiv.org/abs/1512.03385).

---

## conv-padding-negative

**Severity:** block

Convolution and pooling layers take a non-negative `padding` (the number of
implicit border elements added on each side). A negative `padding` describes
removing border elements, which is not a valid construction argument, and
PyTorch raises at construction. It is the padding sibling of `conv-stride-zero`
and `negative-or-zero-kernel`, and usually slips in when the padding is computed
(`pad = (k - 1) // 2 - drop`) and the arithmetic goes below zero.

```python
import torch.nn as nn

# padding=-1 is rejected at construction.
conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=-1)
```

**Fix:** use a padding of zero or more (for a kernel of 3 with `stride=1`,
`padding=1` keeps the spatial size).

```python
conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
```

**Reference:** PyTorch [`nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) / pooling layers require `padding >= 0`.

---

## bceloss-without-sigmoid

**Severity:** warn

`nn.BCELoss` is the binary cross-entropy of a **probability** against a target,
so its input must already be in `[0, 1]` (the output of a sigmoid). When a file
uses `nn.BCELoss` but has no `Sigmoid` (`nn.Sigmoid` / `torch.sigmoid` /
`F.sigmoid`) anywhere, the model is almost certainly feeding raw logits into the
loss. The cross-entropy of a value outside `[0, 1]` is mathematically wrong and
the loss can even go negative, so training silently optimizes the wrong
objective. This is the complement of `sigmoid-bce-with-logits` (sigmoid + the
fused `BCEWithLogitsLoss`); here a sigmoid is missing rather than doubled.

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc(x)          # raw logits, not probabilities

loss_fn = nn.BCELoss()             # expects probabilities in [0, 1]
```

**Fix:** either add a `Sigmoid` before the loss so the input is a probability,
or (preferred, more numerically stable) keep the raw logits and switch to
`nn.BCEWithLogitsLoss`, which fuses the sigmoid in.

```python
loss_fn = nn.BCEWithLogitsLoss()   # consumes raw logits directly
```

**Reference:** PyTorch [`nn.BCELoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) documents that the input must be a probability in `[0, 1]`.

---

## log-then-softmax

**Severity:** warn

Computing `log(softmax(x))` in two separate steps is numerically unstable: the
softmax can underflow to zero for large negative logits, and the following `log`
then returns `-inf`, which poisons the gradient. `F.log_softmax` computes the
same quantity in one fused, numerically stable pass (it never materializes the
tiny intermediate probabilities). This rule flags the explicit composed forms
`torch.log(F.softmax(...))` and `F.softmax(...).log()`.

```python
import torch
import torch.nn.functional as F

# Underflows for confident logits, then log() returns -inf.
log_probs = torch.log(F.softmax(logits, dim=-1))
```

**Fix:** use `F.log_softmax` (or `nn.LogSoftmax`) directly.

```python
log_probs = F.log_softmax(logits, dim=-1)
```

**Reference:** PyTorch [`F.log_softmax`](https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html) is documented as faster and more numerically stable than `log(softmax(x))`.

---

## relu-then-softmax

**Severity:** warn

A `ReLU` placed directly before a `Softmax` (or `LogSoftmax`) clamps every logit
to be non-negative. Softmax is shift-invariant but **not** clamp-invariant: once
all logits are `>= 0`, the model can no longer push any class below the others
by making its logit negative, so low-probability classes are systematically
inflated and the output distribution is distorted. Softmax is meant to run on
raw, unbounded logits. This rule fires on the conservative `nn.Sequential(...)`
adjacency where a `ReLU` is directly followed by a `Softmax` / `LogSoftmax`.

```python
import torch.nn as nn

classifier = nn.Sequential(
    nn.Linear(128, 10),
    nn.ReLU(),          # clamps logits to >= 0
    nn.Softmax(dim=-1),
)
```

**Fix:** drop the `ReLU` and apply the softmax to the raw logits.

```python
classifier = nn.Sequential(
    nn.Linear(128, 10),
    nn.Softmax(dim=-1),
)
```

**Reference:** PyTorch [`nn.Softmax`](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) operates on unnormalized logits; clamping them first changes the distribution.

---

## view-after-transpose

**Severity:** warn

`.transpose(...)` and `.permute(...)` return a view with non-contiguous memory
strides. `.view()` requires a contiguous tensor and raises a runtime error
(`view size is not compatible with input tensor's size and stride ...`) when
called on that result. The fix is `.reshape()`, which falls back to a copy when
the tensor is non-contiguous, or an explicit `.contiguous()` before `.view()`.
This rule flags the directly-chained `x.transpose(...).view(...)` /
`x.permute(...).view(...)` form.

```python
# transpose makes x non-contiguous; .view() then raises at runtime.
y = x.transpose(1, 2).view(x.size(0), -1)
```

**Fix:** use `.reshape()` (or insert `.contiguous()`).

```python
y = x.transpose(1, 2).reshape(x.size(0), -1)
```

**Reference:** PyTorch [`Tensor.view`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) requires compatible contiguous strides; [`Tensor.reshape`](https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html) handles the non-contiguous case.

---

## scheduler-step-before-optimizer

**Severity:** warn

In a PyTorch training loop the optimizer must take its step before the learning
-rate scheduler. When `scheduler.step()` runs before the first
`optimizer.step()`, PyTorch emits "Detected call of `lr_scheduler.step()` before
`optimizer.step()`" and the very first learning-rate value in the schedule is
skipped (warmup, in particular, then starts one step late). This rule fires when
the first `scheduler.step()` line precedes the first `optimizer.step()` line in
the same file.

```python
for batch, target in loader:
    optimizer.zero_grad()
    loss = loss_fn(model(batch), target)
    loss.backward()
    scheduler.step()     # WRONG: runs before optimizer.step()
    optimizer.step()
```

**Fix:** step the optimizer first, then the scheduler.

```python
    optimizer.step()
    scheduler.step()
```

**Reference:** PyTorch [`torch.optim.lr_scheduler`](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) warns that the scheduler must be stepped after the optimizer.

---

## The other rules

The twenty rules above are the ones reliably detectable from source text with a
regex. The full [Neurarch](https://neurarch.com) web app adds propagator-based
checks that need the typed architecture graph rather than raw text:

- full shape-mismatch detection across the whole graph,
- layer-level GQA introspection (not just the constructor call),
- parameter-explosion estimates,
- cycle detection in the architecture graph,
- orphan / disconnected-layer detection,

among others. Those run on a real architecture graph, so they cannot be
expressed as a regex here.

See the complete catalog at <https://neurarch.com/rules.html>, and try the full
engine at <https://neurarch.com>.
