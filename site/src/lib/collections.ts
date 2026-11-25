import type { Post } from "./posts";

const collections = [
  {
    id: "experiments",
    label: "Experiments",
    description: "All experiment reports on this site.",
  },
  {
    id: "literature",
    label: "Literature",
    description: "Research papers, reading notes, and discussions of published findings.",
  },
  {
    id: "notes",
    label: "Notes",
    description: "Practical project notes, infrastructure guidance, and operational workflows.",
  },
  {
    id: "slides",
    label: "Slides",
    description: "Talks and journal-club presentations on research and related ideas.",
  },
  {
    id: "demo",
    label: "Uncategorized",
    description: "Articles and experiments without a subject collection.",
  },
];

export function collectionMemberships(value: string | string[]): string[] {
  return typeof value === "string" ? [value] : value;
}

export function collectionCounts(entries: readonly { collections: readonly string[] }[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const entry of entries) {
    for (const collection of entry.collections) {
      counts.set(collection, (counts.get(collection) ?? 0) + 1);
    }
  }
  return counts;
}

export function collectionLabel(id: string): string {
  return collections.find((collection) => collection.id === id)?.label ?? id;
}

export function compareCollections(a: string, b: string): number {
  const rank = (id: string) => {
    const index = collections.findIndex((collection) => collection.id === id);
    return index < 0 ? collections.length : index;
  };
  return rank(a) - rank(b) || a.localeCompare(b);
}

export function collectionId(collection: string): string {
  return `collection-${collection}`;
}
export function collectionUrl(collection: string): string {
  return `/collections/${encodeURIComponent(collection)}`;
}

export function collectionDescription(id: string): string {
  return (
    collections.find((collection) => collection.id === id)?.description ??
    `Articles in ${collectionLabel(id)}.`
  );
}

export function collectionSummary(posts: readonly Post[]): { count: number; latestDate?: string } {
  const members = [...new Map(posts.map((post) => [post.href, post])).values()];
  return {
    count: members.length,
    latestDate: members
      .map((post) => post.data.updated_at)
      .sort()
      .at(-1),
  };
}
