#!/usr/bin/env bash
# Publish the built site to the `gh-pages` branch.
#
#   scripts/publish_site.sh            # commit _site/ onto gh-pages, locally; print the push command
#   scripts/publish_site.sh --push     # ... and push it to origin
#
# The branch is rewritten from scratch each time (one commit, no history): the site is a build
# artefact, and keeping every encoded clip of every build in the repository's history would add
# tens of megabytes a publish. The main branch never carries any of it -- `_site/` is gitignored
# and the media are re-encoded from `outputs/`, which is gitignored too.
#
# GitHub Pages then has to be pointed at the branch once, by hand: Settings -> Pages -> Build and
# deployment -> Deploy from a branch -> gh-pages / (root). The site appears at
# https://<user>.github.io/<repo>/.
set -euo pipefail

BRANCH="${BRANCH:-gh-pages}"
REMOTE="${REMOTE:-origin}"
PUSH=0
[ "${1:-}" = "--push" ] && PUSH=1

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
site="$repo_root/_site"

if [ ! -f "$site/index.html" ]; then
  echo "no build in $site -- run 'make site' first" >&2
  exit 2
fi

source_commit=$(git rev-parse --short HEAD)
worktree=$(mktemp -d "${TMPDIR:-/tmp}/irsim-gh-pages.XXXXXX")
cleanup() { git worktree remove --force "$worktree" >/dev/null 2>&1 || rm -rf "$worktree"; }
trap cleanup EXIT

git worktree add --force --detach "$worktree" >/dev/null

(
  cd "$worktree"
  if git show-ref --quiet "refs/remotes/$REMOTE/$BRANCH"; then
    git checkout -B "$BRANCH" "$REMOTE/$BRANCH" >/dev/null 2>&1
  elif git show-ref --quiet "refs/heads/$BRANCH"; then
    git checkout -B "$BRANCH" "$BRANCH" >/dev/null 2>&1
  else
    git checkout --orphan "$BRANCH" >/dev/null 2>&1
  fi
  git rm -rq --cached . 2>/dev/null || true
  find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
  cp -a "$site/." .
  git add -A
  if git diff --cached --quiet; then
    echo "the built site is identical to what $BRANCH already holds; nothing to publish"
    exit 0
  fi
  git commit -q -m "site: publish from $source_commit" \
    -m "Built by scripts/build_site.py. The branch is rewritten each publish; the site is an
artefact, not history."
  echo "committed $(git rev-parse --short HEAD) on $BRANCH ($(du -sh . | cut -f1))"
)

if [ "$PUSH" = 1 ]; then
  git push --force "$REMOTE" "$BRANCH"
  url_path=$(git remote get-url "$REMOTE" | sed -E 's#\.git$##; s#.*[:/]([^/]+)/([^/]+)$#https://\1.github.io/\2/#')
  echo "pushed. Once Pages is pointed at $BRANCH the site is at $url_path"
else
  echo "not pushed. To publish it:  git push --force $REMOTE $BRANCH"
fi
