import { ancestry, type RunIndex, type IndexRun } from './run-index.ts';
export interface RunTreeNode {
  id: string;
  row?: IndexRun;
  issue?: string;
  repeated: boolean;
  children: RunTreeNode[];
}
export function runTree(index: RunIndex, root: string, path = new Set<string>(), seen = new Set<string>()): RunTreeNode {
  const row = index.runs[root];
  const issue = path.has(root) ? 'Dependency cycle' : !row ? 'Run unavailable' : undefined;
  const repeated = seen.has(root) && !issue;
  seen.add(root);
  return { id: root, row, issue, repeated, children: issue || repeated ? [] : row.record.depends_on
    .map(id => runTree(index, id, new Set([...path, root]), seen)) };
}
export function reportRuns(index: RunIndex, experiments: string[]) {
  return new Set(experiments.flatMap(experiment => index.selections[experiment]
    ? ancestry(index.runs, index.selections[experiment]) : []));
}

/** Display grouping only: no workflow constraints or storage-retention policy. */
export function lineageGroups(index: RunIndex, declared: string[]) {
  const used = reportRuns(index, declared);
  const experiments = [...new Set(Object.values(index.runs).map(row => row.record.experiment))]
    .sort((a, b) => b.localeCompare(a, 'en', { numeric: true }));
  const groups = experiments.map(experiment => {
    const ids = Object.keys(index.runs).filter(id => index.runs[id].record.experiment === experiment)
      .sort((a, b) => b.localeCompare(a, 'en', { numeric: true }));
    const forest = (members: string[]) => {
      const covered = new Set<string>();
      const trees: RunTreeNode[] = [];
      const mark = (node: RunTreeNode) => {
        if (node.row?.record.experiment !== experiment) return;
        covered.add(node.id);
        node.children.forEach(mark);
      };
      const add = (id: string) => {
        const node = runTree(index, id);
        trees.push(node);
        mark(node);
      };
      for (const id of members) if (index.runs[id].record.stage === 'present') add(id);
      const referenced = new Set(members.flatMap(id => index.runs[id].record.depends_on));
      for (const id of members) if (!covered.has(id) && !referenced.has(id)) add(id);
      for (const id of members) if (!covered.has(id)) add(id);
      return trees;
    };
    const all = forest(ids);
    const hasIssue = (node: RunTreeNode): boolean => !!node.issue || node.children.some(hasIssue);
    return { experiment, clean: (all.length === 1 || all.every(node => used.has(node.id))) && !all.some(hasIssue),
      inUse: forest(ids.filter(id => used.has(id))), unused: forest(ids.filter(id => !used.has(id))) };
  });
  return { groups, used };
}
