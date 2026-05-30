#!/usr/bin/env node
/**
 * neurarch-lint v1: regex-based pre-flight structural lint for PyTorch code.
 *
 * Scope: catches the highest-value structural bugs (attention head_dim
 * divisibility, GQA head ratio, Softmax + CrossEntropyLoss co-use) without
 * needing a real Python AST. The full propagator with shape inference,
 * residual checks, BN ordering, etc. lives in the web app. Wiring that
 * into CI requires a TS bundle and is on the roadmap.
 *
 * Usage:
 *   node lint.mjs file1.py file2.py ...           # lint specific files
 *   node lint.mjs --dir=models                    # lint all .py in a dir
 *   node lint.mjs --json file.py                  # JSON output (for CI)
 *   node lint.mjs --github file.py                # GitHub Actions annotations
 *   node lint.mjs --sarif file.py                 # SARIF 2.1.0 (Code Scanning)
 *   node lint.mjs --version                       # print version and exit
 *   node lint.mjs --help                          # print full usage and exit
 *
 * Exit codes:
 *   0 = no blocking issues
 *   1 = at least one blocking issue found
 *   2 = usage error
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';

// ─── Rule catalogue (mirrors public/rules.html) ──────────────────────────────

/**
 * Each rule: { id, title, severity, check(content, file) -> Finding[] }
 * Findings carry { file, line, rule, severity, message }.
 */
const RULES = [
  // R-MHA: attention head_dim divisibility
  {
    id: 'head-dim-divisibility',
    title: 'Attention head_dim divisibility',
    severity: 'block',
    check: (content, file) => {
      const findings = [];
      // Match nn.MultiheadAttention(embed_dim=..., num_heads=...) across lines.
      const re = /(?:nn\.)?(?:MultiheadAttention|MultiHeadAttention|SelfAttention|CausalAttention)\s*\(([^)]*)\)/g;
      let m;
      while ((m = re.exec(content)) !== null) {
        const args = m[1];
        const embed = extractKwarg(args, ['embed_dim', 'embedDim', 'hidden_dim', 'd_model', 'dim']);
        const heads = extractKwarg(args, ['num_heads', 'numHeads', 'n_heads', 'nhead']);
        if (embed !== null && heads !== null && heads > 0 && embed % heads !== 0) {
          findings.push({
            file,
            line: lineOf(content, m.index),
            rule: 'head-dim-divisibility',
            severity: 'block',
            message: `MultiheadAttention has embed_dim=${embed}, num_heads=${heads}. head_dim would be ${(embed / heads).toFixed(2)} (must be an integer).`,
          });
        }
      }
      return findings;
    },
  },

  // R-GQA: GroupedQueryAttention head ratio
  {
    id: 'gqa-head-divisibility',
    title: 'GQA num_heads / num_kv_heads ratio',
    severity: 'block',
    check: (content, file) => {
      const findings = [];
      const re = /(?:nn\.)?(?:GroupedQueryAttention|GQA|MultiQueryAttention|MQA)\s*\(([^)]*)\)/g;
      let m;
      while ((m = re.exec(content)) !== null) {
        const args = m[1];
        const heads = extractKwarg(args, ['num_heads', 'numHeads', 'n_heads', 'nhead']);
        const kvHeads = extractKwarg(args, ['num_kv_heads', 'numKVHeads', 'num_key_value_heads', 'n_kv_heads']);
        if (heads !== null && kvHeads !== null && kvHeads > 0 && heads % kvHeads !== 0) {
          findings.push({
            file,
            line: lineOf(content, m.index),
            rule: 'gqa-head-divisibility',
            severity: 'block',
            message: `GroupedQueryAttention has num_heads=${heads}, num_kv_heads=${kvHeads}. ${heads} % ${kvHeads} = ${heads % kvHeads}, must be 0.`,
          });
        }
      }
      return findings;
    },
  },

  // R-Softmax-CE: Softmax + CrossEntropyLoss co-use in same file
  {
    id: 'softmax-cross-entropy',
    title: 'Softmax + CrossEntropyLoss double-applied',
    severity: 'warn',
    check: (content, file) => {
      const findings = [];
      const hasExplicitSoftmax = /\bnn\.Softmax\s*\(/.test(content) || /F\.softmax\s*\(/.test(content);
      const hasCE = /\bnn\.CrossEntropyLoss\s*\(/.test(content);
      if (hasExplicitSoftmax && hasCE) {
        // Approximate the line by finding the first Softmax mention.
        const m = /\b(?:nn\.Softmax|F\.softmax)\s*\(/.exec(content);
        findings.push({
          file,
          line: m ? lineOf(content, m.index) : 1,
          rule: 'softmax-cross-entropy',
          severity: 'warn',
          message: 'nn.CrossEntropyLoss applies LogSoftmax internally; an explicit Softmax before it double-applies and slows training.',
        });
      }
      return findings;
    },
  },

  // R-zero-features: nn.Linear or Conv with zero in/out
  {
    id: 'zero-features',
    title: 'Linear / Conv with zero in or out features',
    severity: 'block',
    check: (content, file) => {
      const findings = [];
      const re = /(?:nn\.)?(?:Linear|Conv1d|Conv2d|Conv3d)\s*\(([^)]*)\)/g;
      let m;
      while ((m = re.exec(content)) !== null) {
        const args = m[1];
        // First two positional ints, or in_features / out_features kwargs.
        const positionalNums = args
          .split(',')
          .map(s => s.trim())
          .filter(s => /^\d+$/.test(s))
          .map(Number);
        const inF = extractKwarg(args, ['in_features', 'in_channels']) ?? positionalNums[0];
        const outF = extractKwarg(args, ['out_features', 'out_channels']) ?? positionalNums[1];
        if ((inF !== undefined && inF !== null && inF === 0) || (outF !== undefined && outF !== null && outF === 0)) {
          findings.push({
            file,
            line: lineOf(content, m.index),
            rule: 'zero-features',
            severity: 'block',
            message: `Linear / Conv layer has a zero dim: in=${inF}, out=${outF}. Will fail at construction.`,
          });
        }
      }
      return findings;
    },
  },

  // R-BN-after-act: BatchNorm / LayerNorm wired right after an activation
  // in the forward() body. The advisor rule cites Ioffe & Szegedy 2015:
  // normalization belongs BEFORE the activation.
  {
    id: 'bn-after-activation',
    title: 'BatchNorm / LayerNorm placed after an activation',
    severity: 'warn',
    check: (content, file) => {
      const findings = [];
      // Build attribute -> class table from the __init__ block.
      // Patterns like `self.bn1 = nn.BatchNorm2d(...)` or
      // `self.relu = nn.ReLU(...)`.
      const attrClass = new Map(); // attr -> 'norm' | 'act'
      const ACT_CLASSES = /(ReLU|GELU|SiLU|Swish|Mish|LeakyReLU|ELU|SELU|PReLU)/;
      const NORM_CLASSES = /(BatchNorm|LayerNorm|InstanceNorm|GroupNorm|RMSNorm)/;
      const attrDefRe = /self\.(\w+)\s*=\s*(?:nn\.)?(\w+)\s*\(/g;
      let dm;
      while ((dm = attrDefRe.exec(content)) !== null) {
        const [, attr, cls] = dm;
        if (ACT_CLASSES.test(cls)) attrClass.set(attr, 'act');
        else if (NORM_CLASSES.test(cls)) attrClass.set(attr, 'norm');
      }

      // Walk forward() lines and look for an activation call directly
      // followed by a norm call (within 2 lines).
      const lines = content.split('\n');
      const forwardIdx = lines.findIndex(l => /\bdef\s+forward\s*\(/.test(l));
      if (forwardIdx < 0) return findings;
      let lastKind = null;
      let lastLine = -1;
      for (let i = forwardIdx + 1; i < lines.length; i++) {
        const line = lines[i];
        if (/^\s*def\s+\w+/.test(line) || /^class\s+/.test(line)) break;
        // Pick the last self.X call on this line (forward chain).
        const matches = [...line.matchAll(/self\.(\w+)\s*\(/g)];
        if (matches.length === 0) continue;
        // Functional calls F.relu, F.gelu count as activations too.
        const lastSelfAttr = matches[matches.length - 1][1];
        const kind = attrClass.get(lastSelfAttr);
        if (!kind) continue;
        if (lastKind === 'act' && kind === 'norm' && i - lastLine <= 2) {
          findings.push({
            file,
            line: i + 1,
            rule: 'bn-after-activation',
            severity: 'warn',
            message: `'${lastSelfAttr}' (normalization) follows an activation in forward(). Normalization is meant to stabilize the pre-activation distribution; place it before the activation.`,
          });
        }
        lastKind = kind;
        lastLine = i;
      }

      // Also catch F.relu(...) immediately before a self.bn(...) on the
      // same line: `x = self.bn(F.relu(self.conv(x)))`.
      const innerRe = /self\.(\w+)\s*\(\s*F\.(?:relu|gelu|silu|swish|tanh)\s*\(/g;
      let im;
      while ((im = innerRe.exec(content)) !== null) {
        const attr = im[1];
        if (attrClass.get(attr) === 'norm') {
          findings.push({
            file,
            line: lineOf(content, im.index),
            rule: 'bn-after-activation',
            severity: 'warn',
            message: `'${attr}' (normalization) wraps an activation inline. Apply normalization before the activation.`,
          });
        }
      }
      return findings;
    },
  },

  // R-deep-no-residual: >= 8 weight-carrying layers and zero residual /
  // skip / add merge points. The advisor rule cites He et al. 2015 (ResNet).
  {
    id: 'deep-no-residual',
    title: 'Deep network with no residual connections',
    severity: 'warn',
    check: (content, file) => {
      const findings = [];
      // Count weight-carrying layer instantiations.
      const weightRe = /(?:nn\.)?(?:Linear|Conv1d|Conv2d|Conv3d|MultiheadAttention|GroupedQueryAttention)\s*\(/g;
      const weightCount = (content.match(weightRe) ?? []).length;
      if (weightCount < 8) return findings;

      // Residual indicators. Conservative: only count when the user is
      // clearly summing into a previously-named tensor.
      const RESIDUAL_PATTERNS = [
        /\bx\s*=\s*x\s*\+\s*\w/,                  // x = x + something
        /\b\w+\s*\+\s*self\.\w+\s*\(/,            // residual + self.layer(
        /\btorch\.add\s*\(/,                       // torch.add(x, y)
        /class\s+\w*(?:Residual|Skip)\w*\s*\(/,    // class FooResidual(nn.Module)
        /=\s*\w+\s*\+\s*\w+\s*,/,                  // ff_out = x + h ,
        /\breturn\s+\w+\s*\+\s*\w/,                // return x + h
      ];
      const hasResidual = RESIDUAL_PATTERNS.some(p => p.test(content));
      if (hasResidual) return findings;

      // No residual found. Anchor finding to the first weight-carrying
      // layer for the PR comment to land somewhere useful.
      const firstWeight = weightRe.exec(content);
      findings.push({
        file,
        line: firstWeight ? lineOf(content, firstWeight.index) : 1,
        rule: 'deep-no-residual',
        severity: 'warn',
        message: `${weightCount} weight-carrying layers and no residual / skip connection. Gradient signal degrades through depth; consider ResNet-style skips above 8 layers.`,
      });
      return findings;
    },
  },

  // R-GroupNorm: num_channels must be divisible by num_groups.
  // Guaranteed runtime crash at construction, analogous to head_dim.
  {
    id: 'groupnorm-channel-divisibility',
    title: 'GroupNorm num_channels divisibility',
    severity: 'block',
    check: (content, file) => {
      const findings = [];
      const re = /(?:nn\.)?GroupNorm\s*\(([^)]*)\)/g;
      let m;
      while ((m = re.exec(content)) !== null) {
        const args = m[1];
        // Kwarg form first, then fall back to positional (num_groups, num_channels).
        const positionalNums = args
          .split(',')
          .map(s => s.trim())
          .filter(s => /^\d+$/.test(s))
          .map(Number);
        const groups = extractKwarg(args, ['num_groups', 'numGroups']) ?? positionalNums[0];
        const channels = extractKwarg(args, ['num_channels', 'numChannels']) ?? positionalNums[1];
        if (
          groups !== undefined && groups !== null && groups > 0 &&
          channels !== undefined && channels !== null &&
          channels % groups !== 0
        ) {
          findings.push({
            file,
            line: lineOf(content, m.index),
            rule: 'groupnorm-channel-divisibility',
            severity: 'block',
            message: `GroupNorm has num_channels=${channels}, num_groups=${groups}. num_channels must be divisible by num_groups (${channels} / ${groups} is not an integer).`,
          });
        }
      }
      return findings;
    },
  },

  // R-Sigmoid-BCE: explicit Sigmoid + BCEWithLogitsLoss in same file.
  // BCEWithLogitsLoss applies sigmoid internally; an explicit sigmoid
  // double-applies and breaks training. Mirrors softmax-cross-entropy.
  {
    id: 'sigmoid-bce-with-logits',
    title: 'Sigmoid + BCEWithLogitsLoss double-applied',
    severity: 'warn',
    check: (content, file) => {
      const findings = [];
      const hasExplicitSigmoid =
        /\bnn\.Sigmoid\s*\(/.test(content) ||
        /\b(?:torch|F)\.sigmoid\s*\(/.test(content);
      const hasBCEWithLogits = /\bnn\.BCEWithLogitsLoss\s*\(/.test(content);
      if (hasExplicitSigmoid && hasBCEWithLogits) {
        // Approximate the line by finding the first Sigmoid mention.
        const m = /\b(?:nn\.Sigmoid|(?:torch|F)\.sigmoid)\s*\(/.exec(content);
        findings.push({
          file,
          line: m ? lineOf(content, m.index) : 1,
          rule: 'sigmoid-bce-with-logits',
          severity: 'warn',
          message: 'nn.Sigmoid combined with nn.BCEWithLogitsLoss double-applies the sigmoid. BCEWithLogitsLoss expects raw logits; drop the explicit Sigmoid (or use BCELoss).',
        });
      }
      return findings;
    },
  },

  // R-Dropout-range: Dropout probability must be in [0, 1).
  // p >= 1 zeros the entire signal; p < 0 is invalid.
  {
    id: 'dropout-p-range',
    title: 'Dropout probability out of range',
    severity: 'block',
    check: (content, file) => {
      const findings = [];
      const re = /(?:nn\.)?(?:Dropout|Dropout1d|Dropout2d|Dropout3d|AlphaDropout|FeatureAlphaDropout)\s*\(([^)]*)\)/g;
      let m;
      while ((m = re.exec(content)) !== null) {
        const args = m[1];
        // Kwarg `p=` first, then the first positional float/int.
        const p = extractFloatKwarg(args, ['p']) ?? firstPositionalFloat(args);
        if (p !== null && (p >= 1.0 || p < 0)) {
          findings.push({
            file,
            line: lineOf(content, m.index),
            rule: 'dropout-p-range',
            severity: 'block',
            message: `Dropout has p=${p}, which ${p < 0 ? 'is negative and invalid' : 'zeros the entire signal during training'} (p must be in [0, 1)).`,
          });
        }
      }
      return findings;
    },
  },

  // R-Softmax-no-dim: Softmax / LogSoftmax without an explicit dim.
  // PyTorch warns the implicit dim is ambiguous and deprecated.
  {
    id: 'softmax-no-dim',
    title: 'Softmax without explicit dim',
    severity: 'warn',
    check: (content, file) => {
      const findings = [];
      // Module form: nn.Softmax( ... ) / nn.LogSoftmax( ... ) with no dim= arg.
      const moduleRe = /(?:nn\.)?(?:Softmax|LogSoftmax)\s*\(([^)]*)\)/g;
      let m;
      while ((m = moduleRe.exec(content)) !== null) {
        const args = m[1];
        if (!/\bdim\s*=/.test(args)) {
          findings.push({
            file,
            line: lineOf(content, m.index),
            rule: 'softmax-no-dim',
            severity: 'warn',
            message: 'Softmax called without an explicit dim. The implicit dimension is ambiguous and deprecated; pass dim= (usually dim=-1).',
          });
        }
      }
      // Functional form: F.softmax(x) / F.log_softmax(x) with a single arg, no dim=.
      const funcRe = /F\.(?:softmax|log_softmax)\s*\(([^)]*)\)/g;
      while ((m = funcRe.exec(content)) !== null) {
        const args = m[1];
        if (/\bdim\s*=/.test(args)) continue;
        // Only flag the single-argument form; multi-arg calls may pass dim positionally.
        const argCount = args.split(',').map(s => s.trim()).filter(s => s.length > 0).length;
        if (argCount === 1) {
          findings.push({
            file,
            line: lineOf(content, m.index),
            rule: 'softmax-no-dim',
            severity: 'warn',
            message: 'Softmax called without an explicit dim. The implicit dimension is ambiguous and deprecated; pass dim= (usually dim=-1).',
          });
        }
      }
      return findings;
    },
  },

  // R-Conv-stride-zero: ConvXd / PoolXd with stride=0. The output-size
  // formula divides by stride, so stride=0 is a guaranteed runtime error.
  {
    id: 'conv-stride-zero',
    title: 'Conv/Pool stride of zero',
    severity: 'block',
    check: (content, file) => {
      const findings = [];
      const re = /(?:nn\.)?(Conv1d|Conv2d|Conv3d|MaxPool1d|MaxPool2d|MaxPool3d|AvgPool1d|AvgPool2d|AvgPool3d)\s*\(([^)]*)\)/g;
      let m;
      while ((m = re.exec(content)) !== null) {
        const cls = m[1];
        const args = m[2];
        const stride = extractKwarg(args, ['stride']);
        if (stride === 0) {
          findings.push({
            file,
            line: lineOf(content, m.index),
            rule: 'conv-stride-zero',
            severity: 'block',
            message: `${cls} has stride=0, which is invalid (stride must be >= 1; output size divides by stride).`,
          });
        }
      }
      return findings;
    },
  },

  // R-zero-or-negative-kernel: ConvXd / PoolXd with kernel_size=0 or a
  // negative kernel_size. kernel_size must be a positive integer.
  {
    id: 'negative-or-zero-kernel',
    title: 'Conv/Pool kernel_size of zero or negative',
    severity: 'block',
    check: (content, file) => {
      const findings = [];
      const re = /(?:nn\.)?(Conv1d|Conv2d|Conv3d|MaxPool1d|MaxPool2d|MaxPool3d|AvgPool1d|AvgPool2d|AvgPool3d)\s*\(([^)]*)\)/g;
      let m;
      while ((m = re.exec(content)) !== null) {
        const cls = m[1];
        const args = m[2];
        const kernel = extractSignedKwarg(args, ['kernel_size']);
        if (kernel !== null && kernel <= 0) {
          findings.push({
            file,
            line: lineOf(content, m.index),
            rule: 'negative-or-zero-kernel',
            severity: 'block',
            message: `${cls} has kernel_size=${kernel} (must be a positive integer).`,
          });
        }
      }
      return findings;
    },
  },

  // R-linear-bias-before-norm: a ConvXd / Linear with explicit bias=True
  // immediately followed by a BatchNorm in an nn.Sequential(...) literal.
  // BatchNorm has its own bias (beta), so the preceding bias is redundant.
  // Kept deliberately conservative (Sequential adjacency only) to avoid
  // false positives.
  {
    id: 'linear-bias-before-norm',
    title: 'Bias=True on a layer immediately before BatchNorm',
    severity: 'warn',
    check: (content, file) => {
      const findings = [];
      // A ConvXd / Linear with explicit bias=True, then (optionally an
      // activation) directly followed by a BatchNormXd, inside a Sequential.
      const re = /(?:nn\.)?(Conv1d|Conv2d|Conv3d|Linear)\s*\([^)]*\bbias\s*=\s*True[^)]*\)\s*,\s*(?:nn\.)?BatchNorm(?:1d|2d|3d)\s*\(/g;
      let m;
      while ((m = re.exec(content)) !== null) {
        findings.push({
          file,
          line: lineOf(content, m.index),
          rule: 'linear-bias-before-norm',
          severity: 'warn',
          message: 'Linear/Conv with bias=True immediately before BatchNorm. BatchNorm has its own bias (beta); set bias=False to save parameters.',
        });
      }
      return findings;
    },
  },

  // R-embedding-zero-size: nn.Embedding with a zero vocabulary
  // (num_embeddings) or zero-width vectors (embedding_dim). Mirrors
  // zero-features but for the Embedding table. Both forms crash or are useless.
  {
    id: 'embedding-zero-size',
    title: 'Embedding with zero num_embeddings or embedding_dim',
    severity: 'block',
    check: (content, file) => {
      const findings = [];
      const re = /(?:nn\.)?Embedding\s*\(([^)]*)\)/g;
      let m;
      while ((m = re.exec(content)) !== null) {
        const args = m[1];
        // First two positional ints, or num_embeddings / embedding_dim kwargs.
        const positionalNums = args
          .split(',')
          .map(s => s.trim())
          .filter(s => /^\d+$/.test(s))
          .map(Number);
        const num = extractKwarg(args, ['num_embeddings']) ?? positionalNums[0];
        const dim = extractKwarg(args, ['embedding_dim']) ?? positionalNums[1];
        if (num === 0) {
          findings.push({
            file,
            line: lineOf(content, m.index),
            rule: 'embedding-zero-size',
            severity: 'block',
            message: 'Embedding has num_embeddings=0 (the vocabulary/table size must be >= 1).',
          });
        } else if (dim === 0) {
          findings.push({
            file,
            line: lineOf(content, m.index),
            rule: 'embedding-zero-size',
            severity: 'block',
            message: 'Embedding has embedding_dim=0 (the vector width must be >= 1).',
          });
        }
      }
      return findings;
    },
  },
];

// ─── Helpers ────────────────────────────────────────────────────────────────

function extractKwarg(args, keys) {
  for (const k of keys) {
    const re = new RegExp(`\\b${k}\\s*=\\s*(\\d+)`);
    const m = re.exec(args);
    if (m) return parseInt(m[1], 10);
  }
  return null;
}

// Sign-aware sibling of extractKwarg: returns a (possibly negative) int, or
// null, for `key=-3`. Plain extractKwarg only matches unsigned ints.
function extractSignedKwarg(args, keys) {
  for (const k of keys) {
    const re = new RegExp(`\\b${k}\\s*=\\s*(-?\\d+)`);
    const m = re.exec(args);
    if (m) return parseInt(m[1], 10);
  }
  return null;
}

// Float-aware sibling of extractKwarg: returns a float (or null) for `key=1.0`.
function extractFloatKwarg(args, keys) {
  for (const k of keys) {
    const re = new RegExp(`\\b${k}\\s*=\\s*(-?\\d*\\.?\\d+)`);
    const m = re.exec(args);
    if (m) return parseFloat(m[1]);
  }
  return null;
}

// First bare positional numeric argument as a float (skips kwargs), or null.
function firstPositionalFloat(args) {
  const first = args.split(',')[0]?.trim() ?? '';
  if (first.includes('=')) return null; // leading arg is a kwarg, not positional
  const m = /^(-?\d*\.?\d+)$/.exec(first);
  return m ? parseFloat(m[1]) : null;
}

function lineOf(content, offset) {
  let line = 1;
  for (let i = 0; i < offset && i < content.length; i++) {
    if (content[i] === '\n') line++;
  }
  return line;
}

// Non-dot directories we never descend into: virtualenvs, build output, and
// vendored site-packages (which can hold huge generated trees). Dot-prefixed
// dirs (.git, .venv, .mypy_cache, .pytest_cache, ...) are already skipped by
// the startsWith('.') guard below.
const SKIP_DIRS = new Set([
  'node_modules', 'venv', '__pycache__', 'dist', 'build', 'site-packages',
]);

// Skip files larger than ~2 MB: they are almost always generated (vendored
// weights-as-code, autograd dumps) and reading them stalls the run.
const MAX_FILE_BYTES = 2 * 1024 * 1024;

function collectPyFiles(dir) {
  const out = [];
  let entries;
  try {
    entries = readdirSync(dir);
  } catch {
    // Unreadable dir (permissions, race): emit nothing, keep scanning the rest.
    return out;
  }
  for (const entry of entries) {
    if (entry.startsWith('.') || SKIP_DIRS.has(entry)) continue;
    const full = join(dir, entry);
    let s;
    try {
      s = statSync(full);
    } catch {
      continue; // dangling symlink / vanished entry: skip it.
    }
    if (s.isDirectory()) out.push(...collectPyFiles(full));
    else if (entry.endsWith('.py')) out.push(full);
  }
  return out;
}

export function lintContent(content, filePath = '<inline>') {
  const findings = [];
  for (const rule of RULES) {
    try {
      findings.push(...rule.check(content, filePath));
    } catch (e) {
      findings.push({
        file: filePath,
        line: 0,
        rule: rule.id,
        severity: 'warn',
        message: `Rule threw: ${e.message}`,
      });
    }
  }
  return findings;
}

export { RULES, formatGithub, formatSarif };

export function lintFile(filePath) {
  // Size guard: skip generated / vendored files larger than MAX_FILE_BYTES
  // rather than reading the whole thing into memory for a regex pass.
  try {
    const s = statSync(filePath);
    if (s.size > MAX_FILE_BYTES) {
      return [{
        file: filePath,
        line: 0,
        rule: 'file-too-large',
        severity: 'info',
        message: `file too large to scan (>2MB), skipped (${s.size} bytes).`,
      }];
    }
  } catch {
    // statSync failed: fall through to readFileSync, which reports the error.
  }
  let content;
  try {
    content = readFileSync(filePath, 'utf8');
  } catch (e) {
    return [{
      file: filePath,
      line: 0,
      rule: 'read-error',
      severity: 'warn',
      message: `Could not read file: ${e.message}`,
    }];
  }
  const findings = lintContent(content, filePath);
  return findings;
}

// ─── Output formatters ──────────────────────────────────────────────────────

function formatHuman(findings) {
  if (findings.length === 0) {
    return 'neurarch-lint: no structural issues found.';
  }
  const lines = [`neurarch-lint: ${findings.length} issue${findings.length === 1 ? '' : 's'} found.`, ''];
  const bySeverity = { block: [], warn: [], info: [] };
  for (const f of findings) (bySeverity[f.severity] ?? bySeverity.warn).push(f);
  for (const sev of ['block', 'warn', 'info']) {
    if (bySeverity[sev].length === 0) continue;
    const tag = sev === 'block' ? 'BLOCK' : sev === 'warn' ? 'WARN' : 'INFO';
    for (const f of bySeverity[sev]) {
      lines.push(`  [${tag}] ${f.file}:${f.line}  ${f.rule}`);
      lines.push(`         ${f.message}`);
    }
  }
  lines.push('');
  lines.push('Rule reference: https://neurarch.com/rules.html');
  return lines.join('\n');
}

function formatMarkdown(findings) {
  if (findings.length === 0) {
    return '### neurarch-lint\n\nNo structural issues found in this PR.';
  }
  const lines = [
    `### neurarch-lint`,
    '',
    `Found **${findings.length}** structural issue${findings.length === 1 ? '' : 's'} in this PR:`,
    '',
  ];
  for (const f of findings) {
    const icon = f.severity === 'block' ? ':no_entry:' : f.severity === 'warn' ? ':warning:' : ':information_source:';
    lines.push(`${icon} **${f.rule}** (${f.severity})`);
    lines.push(`  \`${f.file}:${f.line}\``);
    lines.push(`  ${f.message}`);
    lines.push('');
  }
  lines.push('---');
  lines.push('Full rule reference: <https://neurarch.com/rules.html>');
  return lines.join('\n');
}

// GitHub Actions workflow commands. Each finding becomes an ::error or
// ::warning annotation so it renders inline on the PR diff and in the Checks
// summary, with no pull-requests:write permission needed.
// See: https://docs.github.com/actions/using-workflows/workflow-commands-for-github-actions
function formatGithub(findings) {
  const lines = [];
  for (const f of findings) {
    const cmd = f.severity === 'block' ? 'error' : 'warning';
    const file = ghProp(f.file);
    const line = ghProp(String(f.line));
    const title = ghProp(`neurarch-lint: ${f.rule}`);
    const message = ghData(f.message);
    lines.push(`::${cmd} file=${file},line=${line},title=${title}::${message}`);
  }
  return lines.join('\n');
}

// Escape a message body for a workflow command (the part after `::`).
function ghData(s) {
  return String(s)
    .replace(/%/g, '%25')
    .replace(/\r/g, '%0D')
    .replace(/\n/g, '%0A');
}

// Escape a workflow command property (file / line / title). Properties need
// the data escapes plus `,` and `:`.
function ghProp(s) {
  return ghData(s)
    .replace(/,/g, '%2C')
    .replace(/:/g, '%3A');
}

// SARIF 2.1.0 document for upload to GitHub Code Scanning (or any SARIF
// viewer). The rule list is derived from RULES so it never drifts from the
// code. See: https://json.schemastore.org/sarif-2.1.0.json
function formatSarif(findings) {
  const levelOf = sev => (sev === 'block' ? 'error' : 'warning');
  const sarif = {
    $schema: 'https://json.schemastore.org/sarif-2.1.0.json',
    version: '2.1.0',
    runs: [
      {
        tool: {
          driver: {
            name: 'neurarch-lint',
            informationUri: 'https://neurarch.com',
            version: packageVersion(),
            rules: RULES.map(r => ({
              id: r.id,
              name: r.title,
              shortDescription: { text: r.title },
              helpUri: 'https://neurarch.com/rules.html',
            })),
          },
        },
        results: findings.map(f => ({
          ruleId: f.rule,
          level: levelOf(f.severity),
          message: { text: f.message },
          locations: [
            {
              physicalLocation: {
                artifactLocation: { uri: f.file },
                region: { startLine: f.line > 0 ? f.line : 1 },
              },
            },
          ],
        })),
      },
    ],
  };
  return JSON.stringify(sarif, null, 2);
}

// Read the version from package.json next to this script. Falls back to
// '0.0.0' if it cannot be read so SARIF output never crashes the run.
function packageVersion() {
  try {
    const pkgPath = join(dirname(fileURLToPath(import.meta.url)), 'package.json');
    return JSON.parse(readFileSync(pkgPath, 'utf8')).version ?? '0.0.0';
  } catch {
    return '0.0.0';
  }
}

// ─── Entry point ────────────────────────────────────────────────────────────

const HELP_TEXT = `neurarch-lint: pre-flight structural lint for PyTorch models.

Usage:
  neurarch-lint [--json|--markdown|--github|--sarif] [--dir=PATH] FILES...

Flags:
  --json          Machine-readable JSON output (for CI).
  --markdown      PR-comment style markdown output.
  --github        GitHub Actions annotations (::error / ::warning, inline on the PR).
  --sarif         SARIF 2.1.0 document for GitHub Code Scanning.
  --dir=PATH      Lint every .py file under PATH (recursively).
  --version, -v   Print the version and exit.
  --help, -h      Print this help and exit.

Exit codes:
  0   no blocking issues (clean)
  1   at least one blocking issue found
  2   usage error

Rule reference: https://neurarch.com/rules.html`;

function main() {
  const argv = process.argv.slice(2);
  if (argv.includes('--version') || argv.includes('-v')) {
    console.log(`neurarch-lint ${packageVersion()}`);
    process.exit(0);
  }
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log(HELP_TEXT);
    process.exit(0);
  }
  if (argv.length === 0) {
    console.error('Usage: node lint.mjs [--json|--markdown|--github|--sarif] [--dir=PATH] FILES...');
    process.exit(2);
  }

  const wantJson = argv.includes('--json');
  const wantMarkdown = argv.includes('--markdown');
  const wantGithub = argv.includes('--github');
  const wantSarif = argv.includes('--sarif');
  let files = argv.filter(a => !a.startsWith('-') && a.endsWith('.py'));

  for (const a of argv) {
    if (a.startsWith('--dir=')) {
      const dir = a.slice('--dir='.length);
      files.push(...collectPyFiles(dir));
    }
  }

  if (files.length === 0) {
    console.error('neurarch-lint: no .py files to scan.');
    process.exit(0);
  }

  const findings = [];
  for (const f of files) findings.push(...lintFile(f));

  if (wantJson) {
    console.log(JSON.stringify({ files: files.length, findings }, null, 2));
  } else if (wantMarkdown) {
    console.log(formatMarkdown(findings));
  } else if (wantGithub) {
    console.log(formatGithub(findings));
  } else if (wantSarif) {
    console.log(formatSarif(findings));
  } else {
    console.log(formatHuman(findings));
  }

  const hasBlocker = findings.some(f => f.severity === 'block');
  process.exit(hasBlocker ? 1 : 0);
}

// Only run main() when invoked as a script, not when imported as a module.
import { fileURLToPath } from 'node:url';
const invokedDirectly =
  process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (invokedDirectly) {
  main();
}
