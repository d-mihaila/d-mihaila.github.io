#!/usr/bin/env bash
# Merge upstream Pinglab's tooling and site into this fork. This fork owns experiments/,
# notes/ and papers/ (its experiment numbers overlap upstream's), so those are kept
# exactly as they were. Leaves the merge uncommitted for review; content conflicts in
# shared files this fork changed are listed at the end.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain)" ]; then echo "Commit or stash your changes first." >&2; exit 1; fi
git fetch upstream
git merge --no-commit --no-ff upstream/main || true

owned=(experiments notes papers site/src/lib/dependencies.ts)
git rm -r -q --force --ignore-unmatch -- "${owned[@]}"
git checkout HEAD -- "${owned[@]}"
for path in CNAME site/public/recovered; do
  [ -e "$path" ] && git rm -r -q --force --ignore-unmatch "$path"
done
# Upstream may have edited files this fork deleted; keep them deleted.
git diff --name-only --diff-filter=U | while read -r path; do
  [ -e "$path" ] || git rm -q --ignore-unmatch "$path"
done

conflicts=$(git diff --name-only --diff-filter=U)
if [ -n "$conflicts" ]; then
  printf 'Resolve these conflicts, then run git commit:\n%s\n' "$conflicts"
else
  echo "Merge prepared. Review with git status and git diff --cached, then run git commit."
fi
