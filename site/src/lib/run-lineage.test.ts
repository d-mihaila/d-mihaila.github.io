import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildRunIndex, type IndexRun } from './run-index.ts';
import { lineageGroups, runTree } from './run-lineage.ts';
function row(id: string, depends_on: string[] = [], stage = 'compute', experiment = 'exp001'): IndexRun {
  return { storage: 'Local', record: { id, experiment, stage, depends_on, created_at: '2026-11-12T10:00:00Z' } };
}
test('display forests separate unused branches without requiring a stage triplet', () => {
  const index = buildRunIndex({ source: row('source'), present: row('present', ['source'], 'present'), old: row('old') });
  const { groups, used } = lineageGroups(index, ['exp001']);
  assert.deepEqual([...used], ['present', 'source']);
  assert.deepEqual(groups[0].inUse.map(node => node.id), ['present']);
  assert.deepEqual(groups[0].unused.map(node => node.id), ['old']);
  assert.equal(groups[0].clean, false);
  delete index.runs.old;
  assert.equal(lineageGroups(index, ['exp001']).groups[0].clean, true);
});
test('shared dependencies expand once and cycles remain visible', () => {
  const index = buildRunIndex({ source: row('source'), analysis: row('analysis', ['source'], 'analyse'),
    present: row('present', ['analysis', 'source'], 'present') });
  const tree = runTree(index, 'present');
  assert.equal(tree.children[0].children[0].repeated, false);
  assert.equal(tree.children[1].repeated, true);
  index.runs.source.record.depends_on = ['present'];
  assert.equal(runTree(index, 'present').children[0].children[0].children[0].issue, 'Dependency cycle');
  assert.equal(lineageGroups(index, ['exp001']).groups[0].clean, false);
});
test('missing and external dependencies do not disappear from the tree', () => {
  const index = buildRunIndex({ external: row('external', [], 'compute', 'exp002'),
    present: row('present', ['external', 'missing'], 'present') });
  const tree = runTree(index, 'present');
  assert.equal(tree.children[0].row?.record.experiment, 'exp002');
  assert.equal(tree.children[1].issue, 'Run unavailable');
});

test('extra source branches used by other reports are clean, while unused extras are not', () => {
  const index = buildRunIndex({ source: row('source'), shared: row('shared'),
    present: row('present', ['source'], 'present'), consumer: row('consumer', ['shared'], 'present', 'exp002') });
  const group = () => lineageGroups(index, ['exp001', 'exp002']).groups.find(group => group.experiment === 'exp001')!;
  assert.equal(group().clean, true);
  assert.deepEqual(group().unused, []);
  index.runs.leftover = row('leftover');
  assert.equal(group().clean, false);
  assert.deepEqual(group().unused.map(node => node.id), ['leftover']);
});
