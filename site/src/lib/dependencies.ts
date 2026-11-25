// Scientific dependencies between reports hosted on this site, for example
// `exp501: ['exp176']`. Pingstore run dependencies (such as exp022) are found automatically.
export const dependencies: Record<string, string[]> = {};
export function downstream(experiment: string): string[] {
  return Object.keys(dependencies).filter(id => dependencies[id].includes(experiment)).sort();
}
