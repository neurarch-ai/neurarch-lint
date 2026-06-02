#!/usr/bin/env node
/**
 * neurarch-lint benchmark runner.
 *
 * Goal: produce ONE honest, reproducible number for the README / a writeup, the
 * "arXiv equivalent" that makes the repo read as serious:
 *
 *   "Across N pinned public PyTorch repos, neurarch-lint flagged X structural
 *    issues that ruff + mypy pass clean, spanning R rule types."
 *
 * It clones each repo at a PINNED commit (so the number is reproducible), runs
 * `lint.mjs --json` over it, and aggregates findings by rule, severity, and repo.
 *
 * It does NOT invent or upgrade numbers. Repos that fail to clone or lint are
 * reported as skipped and excluded from totals, never silently dropped.
 *
 * Usage:
 *   node run-benchmark.mjs                         # uses ./repos.json, ../lint.mjs
 *   node run-benchmark.mjs --lint ../../lint.mjs   # point at the linter
 *   node run-benchmark.mjs --manifest repos.json --out report.md
 *   node run-benchmark.mjs --keep                  # keep clones in ./.cache (faster reruns)
 *
 * Requires: git on PATH, Node 20+.
 */
import { execFileSync, execSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, existsSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1) return fallback;
  const next = process.argv[i + 1];
  return next && !next.startsWith('--') ? next : true;
}

const LINT_PATH  = resolve(__dirname, arg('lint', '../lint.mjs'));
const MANIFEST   = resolve(__dirname, arg('manifest', 'repos.json'));
const OUT        = resolve(__dirname, arg('out', 'report.md'));
const KEEP       = Boolean(arg('keep', false));
const CACHE_DIR  = resolve(__dirname, '.cache');

if (!existsSync(LINT_PATH)) {
  console.error(`lint script not found at ${LINT_PATH}. Pass --lint <path-to-lint.mjs>.`);
  process.exit(2);
}
if (!existsSync(MANIFEST)) {
  console.error(`manifest not found at ${MANIFEST}.`);
  process.exit(2);
}

const repos = JSON.parse(readFileSync(MANIFEST, 'utf8'));
if (!Array.isArray(repos) || repos.length === 0) {
  console.error('manifest must be a non-empty JSON array of { name, url, ref, paths? }.');
  process.exit(2);
}

/** Shallow-clone `url` at `ref` into `dest`. Returns true on success. */
function cloneAt(url, ref, dest) {
  if (existsSync(dest)) return true; // cached
  mkdirSync(dest, { recursive: true });
  try {
    // Clone the single pinned commit only — fast and exactly reproducible.
    execSync(`git init -q "${dest}"`, { stdio: 'ignore' });
    execSync(`git -C "${dest}" remote add origin "${url}"`, { stdio: 'ignore' });
    execSync(`git -C "${dest}" fetch -q --depth 1 origin "${ref}"`, { stdio: 'ignore' });
    execSync(`git -C "${dest}" checkout -q FETCH_HEAD`, { stdio: 'ignore' });
    return true;
  } catch (e) {
    return false;
  }
}

/** Run the linter over `dir` and return a flat array of findings. */
function runLint(dir, paths) {
  // Directories go through `--dir=`; the README's CLI takes a dir that way, not
  // as a positional arg. A manifest `paths` entry is treated as a sub-dir scope.
  const target = paths && paths.length ? paths.map(p => `--dir=${join(dir, p)}`) : [`--dir=${dir}`];
  let stdout = '';
  try {
    stdout = execFileSync('node', [LINT_PATH, '--json', ...target], {
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
      // exit code 1 = blocking finding (expected); execFileSync throws on non-zero,
      // so we read stdout from the error object below.
    });
  } catch (e) {
    if (e.stdout) stdout = e.stdout.toString();
    else throw e;
  }
  const parsed = JSON.parse(stdout);
  // Tolerate both `[...]` and `{ findings: [...] }` shapes.
  const findings = Array.isArray(parsed) ? parsed : (parsed.findings ?? []);
  return findings;
}

const byRule = new Map();        // rule -> count
const bySeverity = new Map();    // severity -> count
const perRepo = [];              // { name, findings, blocking, warn, rules:Set }
const skipped = [];              // { name, reason }

const workRoot = KEEP ? CACHE_DIR : mkdtempSync(join(tmpdir(), 'nlint-bench-'));
mkdirSync(workRoot, { recursive: true });

for (const repo of repos) {
  const { name, url, ref, paths } = repo;
  if (!name || !url || !ref) { skipped.push({ name: name || url || '?', reason: 'missing name/url/ref' }); continue; }
  process.stderr.write(`• ${name} @ ${ref.slice(0, 12)} ... `);
  const dest = join(workRoot, name.replace(/[^a-zA-Z0-9._-]/g, '_'));
  if (!cloneAt(url, ref, dest)) { skipped.push({ name, reason: 'clone failed (check url/ref)' }); process.stderr.write('clone failed\n'); continue; }
  let findings;
  try {
    findings = runLint(dest, paths);
  } catch (e) {
    skipped.push({ name, reason: `lint failed: ${String(e.message || e).slice(0, 120)}` });
    process.stderr.write('lint failed\n');
    continue;
  }
  const rules = new Set();
  let blocking = 0, warn = 0;
  for (const f of findings) {
    const rule = f.rule ?? f.id ?? 'unknown';
    const sev  = f.severity ?? f.level ?? 'unknown';
    byRule.set(rule, (byRule.get(rule) ?? 0) + 1);
    bySeverity.set(sev, (bySeverity.get(sev) ?? 0) + 1);
    rules.add(rule);
    if (sev === 'block') blocking++; else if (sev === 'warn') warn++;
  }
  perRepo.push({ name, url, ref, total: findings.length, blocking, warn, rules });
  process.stderr.write(`${findings.length} findings (${blocking} block, ${warn} warn)\n`);
}

if (!KEEP) { try { rmSync(workRoot, { recursive: true, force: true }); } catch { /* ignore */ } }

// ── Aggregate ──────────────────────────────────────────────────────────────
const totalFindings = perRepo.reduce((a, r) => a + r.total, 0);
const totalBlocking = perRepo.reduce((a, r) => a + r.blocking, 0);
const ruleTypesHit = byRule.size;
const reposScanned = perRepo.length;

const sortedRules = [...byRule.entries()].sort((a, b) => b[1] - a[1]);

// ── Emit markdown report ─────────────────────────────────────────────────────
const lines = [];
lines.push('# neurarch-lint benchmark');
lines.push('');
lines.push('> Reproducible: each repo is cloned at a pinned commit (see `repos.json`).');
lines.push('> Re-run with `node run-benchmark.mjs`. Numbers below are raw linter output, not curated.');
lines.push('');
lines.push('## Headline');
lines.push('');
lines.push(`Across **${reposScanned}** pinned public PyTorch repos, neurarch-lint flagged **${totalFindings}** structural issues `);
lines.push(`(**${totalBlocking}** blocking) spanning **${ruleTypesHit}** rule types, the class of bug \`ruff\` and \`mypy\` pass clean.`);
lines.push('');
lines.push('## By rule');
lines.push('');
lines.push('| Rule | Findings |');
lines.push('|------|---------:|');
for (const [rule, n] of sortedRules) lines.push(`| \`${rule}\` | ${n} |`);
lines.push('');
lines.push('## By repo');
lines.push('');
lines.push('| Repo | Findings | Blocking | Warn | Rule types |');
lines.push('|------|---------:|---------:|-----:|-----------:|');
for (const r of perRepo.sort((a, b) => b.total - a.total)) {
  lines.push(`| [${r.name}](${r.url}) | ${r.total} | ${r.blocking} | ${r.warn} | ${r.rules.size} |`);
}
lines.push('');
if (skipped.length) {
  lines.push('## Skipped (excluded from totals)');
  lines.push('');
  for (const s of skipped) lines.push(`- **${s.name}** — ${s.reason}`);
  lines.push('');
}
lines.push('---');
lines.push('');
lines.push('Honesty note: this counts raw linter findings. Before quoting it publicly, spot-check a');
lines.push('sample to confirm they are true positives (the v1 regex can over-match dynamic construction).');
lines.push('Report the verified number, and state the sample size you checked.');

writeFileSync(OUT, lines.join('\n'));
console.error(`\nWrote ${OUT}`);
console.error(`Scanned ${reposScanned} repos, ${totalFindings} findings (${totalBlocking} blocking), ${ruleTypesHit} rule types. ${skipped.length} skipped.`);
