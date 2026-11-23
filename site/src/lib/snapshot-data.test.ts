import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RunStore } from './run-store.ts';
import { readSnapshotEvidence } from './snapshot-data.ts';

function fixture() {
  const rows: Record<string, any> = {};
  const add = (id: string, stage: string, inputs: Record<string, string>) => {
    rows[id] = { experiment: 'exp001', stage, depends_on: Object.values(inputs) };
    rows[`${id}/details.json`] = { inputs: Object.fromEntries(Object.entries(inputs).map(([role, run_id]) => [role, { run_id }])), execution: { configuration: {} } };
  };
  add('compute', 'compute', {});
  add('analysis', 'analyse', { compute: 'compute' });
  add('other-analysis', 'analyse', { compute: 'compute' });
  add('original', 'present', { analyse: 'analysis', comparison: 'other-analysis' });
  add('edited', 'present', { previous_present: 'original', compute: 'compute' });
  add('latest', 'present', { restored_present: 'edited' });
  const store = { json: async (id: string, file?: string) => {
    const key = file ? `${id}/${file}` : id;
    if (!(key in rows)) throw Object.assign(new Error('Missing evidence'), { code: 'ENOENT' });
    return rows[key];
  } } as RunStore;
  const read = (id = 'latest', extra = {}) => readSnapshotEvidence(id, {
    experiment: 'exp001', snapshots: {}, analysisInput: 'analyse', computeInput: 'compute', store, ...extra,
  });
  return { rows, add, read };
}

test('presentation edits inherit explicitly named scientific evidence through multiple revisions', async () => {
  const { read } = fixture();
  assert.equal((await read())?.compute.run_id, 'compute');
  assert.equal((await read('original'))?.compute.run_id, 'compute');
});

test('ambiguous presentation ancestry and cycles fail explicitly', async () => {
  const { add, read } = fixture();
  add('latest', 'present', { first: 'edited', second: 'original' });
  await assert.rejects(read(), /Expected one present source/);
  add('latest', 'present', { previous: 'latest' });
  await assert.rejects(read(), /Cyclic snapshot dependency/);
});

test('invalid named analysis and missing parents never fall back to older evidence', async () => {
  const { rows, add, read } = fixture();
  rows['latest/details.json'].inputs.analyse = { run_id: 'analysis' };
  await assert.rejects(read(), /not a dependency/);
  add('latest', 'present', { previous: 'absent' });
  await assert.rejects(read(), /Missing evidence/);
});
