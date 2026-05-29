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
import { lintContent } from './lint.mjs';

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

describe('neurarch-lint action: fixture inventory', () => {
  it('every .py fixture is registered in this harness', () => {
    const onDisk = readdirSync(FIXTURE_DIR).filter(f => f.endsWith('.py'));
    const registered = new Set(FIXTURES.map(f => f.file));
    const orphans = onDisk.filter(f => !registered.has(f));
    expect(orphans, 'unregistered fixtures').toEqual([]);
  });
});
