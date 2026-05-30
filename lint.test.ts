/** @vitest-environment node */
/**
 * Regression suite for the neurarch-lint GitHub Action.
 *
 * Each fixture is paired with an expectation: the set of rule ids that must
 * fire and (where useful) a finding count. Touching a rule's regex without
 * also updating its dedicated fixture will flip this red.
 *
 * Source-of-truth fixtures live in .github/actions/neurarch-lint/fixtures.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { lintContent, formatGithub, formatSarif, RULES } from './lint.mjs';

const HERE         = dirname(fileURLToPath(import.meta.url));
const FIXTURE_DIR  = join(HERE, 'fixtures');

interface Expect {
  /** Rule ids that MUST appear (in any order; duplicates are okay). */
  expectedRules: string[];
  /** Rule ids that MUST NOT appear. Catches false-positives. */
  forbiddenRules?: string[];
  /** Exact finding count when known; omit when only checking rule set. */
  exactCount?: number;
}

const FIXTURES: Array<{ file: string; expect: Expect }> = [
  {
    file: 'rule_head_dim.py',
    expect: { expectedRules: ['head-dim-divisibility'], exactCount: 1 },
  },
  {
    file: 'rule_gqa.py',
    expect: { expectedRules: ['gqa-head-divisibility'], exactCount: 1 },
  },
  {
    file: 'rule_softmax_ce.py',
    expect: { expectedRules: ['softmax-cross-entropy'], exactCount: 1 },
  },
  {
    file: 'rule_zero_features.py',
    expect: { expectedRules: ['zero-features'], exactCount: 1 },
  },
  {
    file: 'rule_bn_after_act.py',
    expect: { expectedRules: ['bn-after-activation'], exactCount: 1 },
  },
  {
    file: 'deep_no_residual.py',
    expect: { expectedRules: ['deep-no-residual'], exactCount: 1 },
  },
  {
    file: 'rule_groupnorm.py',
    expect: { expectedRules: ['groupnorm-channel-divisibility'], exactCount: 1 },
  },
  {
    file: 'rule_sigmoid_bce.py',
    expect: {
      expectedRules: ['sigmoid-bce-with-logits'],
      // The sigmoid head must not be mistaken for a softmax-no-dim hit.
      forbiddenRules: ['softmax-no-dim'],
      exactCount: 1,
    },
  },
  {
    file: 'rule_dropout_range.py',
    expect: { expectedRules: ['dropout-p-range'], exactCount: 1 },
  },
  {
    file: 'rule_softmax_no_dim.py',
    expect: {
      expectedRules: ['softmax-no-dim'],
      // No CrossEntropyLoss here, so softmax-cross-entropy must stay quiet.
      forbiddenRules: ['softmax-cross-entropy'],
      exactCount: 1,
    },
  },
  {
    file: 'rule_conv_stride_zero.py',
    expect: { expectedRules: ['conv-stride-zero'], exactCount: 1 },
  },
  {
    file: 'rule_negative_kernel.py',
    expect: { expectedRules: ['negative-or-zero-kernel'], exactCount: 1 },
  },
  {
    file: 'rule_linear_bias_before_norm.py',
    expect: {
      expectedRules: ['linear-bias-before-norm'],
      // The Sequential has a ReLU, but the norm precedes it, so the
      // forward()-based bn-after-activation rule must stay quiet.
      forbiddenRules: ['bn-after-activation'],
      exactCount: 1,
    },
  },
  {
    file: 'rule_embedding_zero.py',
    expect: { expectedRules: ['embedding-zero-size'], exactCount: 1 },
  },
  {
    file: 'rule_bceloss_without_sigmoid.py',
    expect: {
      expectedRules: ['bceloss-without-sigmoid'],
      // A sigmoid + BCELoss would be correct usage; this fixture has neither
      // that nor BCEWithLogitsLoss, so sigmoid-bce-with-logits must stay quiet.
      forbiddenRules: ['sigmoid-bce-with-logits'],
      exactCount: 1,
    },
  },
  {
    file: 'rule_log_then_softmax.py',
    expect: {
      expectedRules: ['log-then-softmax'],
      // The softmax here passes dim=-1, so softmax-no-dim must not fire.
      forbiddenRules: ['softmax-no-dim'],
      exactCount: 1,
    },
  },
  {
    file: 'rule_view_after_transpose.py',
    expect: { expectedRules: ['view-after-transpose'], exactCount: 1 },
  },
  {
    file: 'rule_scheduler_before_optimizer.py',
    expect: { expectedRules: ['scheduler-step-before-optimizer'], exactCount: 1 },
  },
  {
    file: 'rule_relu_then_softmax.py',
    expect: {
      expectedRules: ['relu-then-softmax'],
      // Softmax has an explicit dim and there is no CrossEntropyLoss, so
      // neither softmax-no-dim nor softmax-cross-entropy should fire.
      forbiddenRules: ['softmax-no-dim', 'softmax-cross-entropy'],
      exactCount: 1,
    },
  },
  {
    file: 'rule_conv_padding_negative.py',
    expect: {
      expectedRules: ['conv-padding-negative'],
      // kernel_size and stride are valid here, so the sibling Conv rules
      // must not fire on the negative padding.
      forbiddenRules: ['conv-stride-zero', 'negative-or-zero-kernel'],
      exactCount: 1,
    },
  },
  {
    file: 'bad_model.py',
    expect: {
      // Aggregate fixture: 4 rules fire together.
      expectedRules: [
        'head-dim-divisibility',
        'gqa-head-divisibility',
        'softmax-cross-entropy',
        'bn-after-activation',
      ],
      // deep-no-residual must NOT fire here (only 3 weight layers in bad_model).
      forbiddenRules: ['deep-no-residual'],
    },
  },
  {
    file: 'clean_model.py',
    expect: { expectedRules: [], exactCount: 0 },
  },
];

describe('neurarch-lint action: per-rule regression', () => {
  for (const { file, expect: exp } of FIXTURES) {
    it(file, () => {
      const content = readFileSync(join(FIXTURE_DIR, file), 'utf8');
      const findings = lintContent(content, file);
      const firedRules = findings.map(f => f.rule);

      if (exp.exactCount !== undefined) {
        expect(findings.length, `findings on ${file}: ${JSON.stringify(firedRules)}`)
          .toBe(exp.exactCount);
      }
      for (const ruleId of exp.expectedRules) {
        expect(firedRules, `${file} should fire ${ruleId}`).toContain(ruleId);
      }
      for (const ruleId of exp.forbiddenRules ?? []) {
        expect(firedRules, `${file} must NOT fire ${ruleId}`).not.toContain(ruleId);
      }
    });
  }
});

describe('neurarch-lint action: output formatters', () => {
  // A snippet that fires one block (head-dim) and one warn (softmax-no-dim),
  // so both annotation levels and the escaping path are exercised.
  const BUGGY = [
    'import torch.nn as nn',
    'attn = nn.MultiheadAttention(embed_dim=384, num_heads=5)',
    'softmax = nn.Softmax()',
  ].join('\n');

  it('formatGithub emits ::error / ::warning with escaped properties', () => {
    const findings = lintContent(BUGGY, 'models/encoder.py');
    const out = formatGithub(findings);
    expect(out).toContain('::error ');
    expect(out).toContain('::warning ');
    // The `:` in `file:line` titles and the path are escaped to %3A.
    expect(out).toContain('%3A');
    // Annotations point at the buggy file.
    expect(out).toContain('file=models/encoder.py');
  });

  it('formatSarif parses as a 2.1.0 doc with matching rule / result counts', () => {
    const findings = lintContent(BUGGY, 'models/encoder.py');
    const doc = JSON.parse(formatSarif(findings));
    expect(doc.version).toBe('2.1.0');
    const run = doc.runs[0];
    expect(run.tool.driver.rules.length).toBe(RULES.length);
    expect(run.results.length).toBe(findings.length);
  });
});

describe('neurarch-lint action: fixture inventory', () => {
  it('every .py fixture is registered in this harness', () => {
    const onDisk = readdirSync(FIXTURE_DIR).filter(f => f.endsWith('.py'));
    const registered = new Set(FIXTURES.map(f => f.file));
    const orphans = onDisk.filter(f => !registered.has(f));
    expect(orphans, 'unregistered fixtures').toEqual([]);
  });
});
