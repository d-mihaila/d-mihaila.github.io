#!/usr/bin/env bash
# Merge upstream Pinglab into this fork, then remove upstream content this fork does not host.
# Leaves the merge uncommitted so you can review it. Content conflicts in files this fork
# changed (site identity, rules) are listed at the end for manual resolution.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain)" ]; then echo "Commit or stash your changes first." >&2; exit 1; fi
git fetch upstream
git merge --no-commit --no-ff upstream/main || true

hosted() { case "$1" in exp17[6-9]|exp180|exp[5-9][0-9][0-9]) return 0 ;; *) return 1 ;; esac; }
for dir in experiments/exp[0-9][0-9][0-9]; do
  [ -e "$dir" ] || continue
  hosted "$(basename "$dir")" || git rm -r -q --force --ignore-unmatch "$dir"
done
for path in notes/*; do
  [ -e "$path" ] && [ "$path" != notes/rules ] && git rm -r -q --force --ignore-unmatch "$path"
done
for path in CNAME site/public/recovered; do
  [ -e "$path" ] && git rm -r -q --force --ignore-unmatch "$path"
done
# This fork owns its paper index; keep it instead of upstream's.
git checkout HEAD -- papers/index.yaml
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
