#!/usr/bin/env bash
# Stage only your own changes to a file several people are editing at once.
#
# Why this exists
# ---------------
# This repo is worked on by several sessions in ONE working tree, not in separate worktrees, so
# TECHNICAL_REPORT.md, CHANGELOG.md and docs/roadmap.md usually hold somebody else's in-flight prose as well
# as yours. `git add README.md` stages the file, which means it stages their paragraph too, and it
# lands in your commit under your message. That has already happened here more than once.
#
# The fix is not to avoid touching shared files; it is to stage a version of the file built from
# HEAD plus your edit alone. This script does that with `git reset` + `git apply --cached`:
#
#     export STAGE_OWN_HUNK_ID=my-session-name                   # once, see below
#     scripts/stage_own_hunk.sh snapshot README.md CHANGELOG.md   # BEFORE you edit them
#     ... edit the files normally; other sessions may edit and even commit them meanwhile ...
#     scripts/stage_own_hunk.sh stage README.md CHANGELOG.md      # stages your diff only
#     git commit -m ...
#
# Set STAGE_OWN_HUNK_ID to something unique per session. Snapshots live under .git, keyed by that
# id, and two sessions sharing the default slot would overwrite each other's baseline -- which is
# the same class of bug this script exists to prevent. `status` prints the id in use.
#
# `check` is read-only and is what the pre-commit hook (.pre-commit-config.yaml) and `make stage`
# run before a commit: it refuses when a shared file's staged content is not what `stage` would
# have produced, which is the signature of a plain `git add` clobbering a hunk this session did
# not author.
#
# The worktree is never modified: everyone's edits stay in it, and the other session's changes are
# simply left unstaged for them to commit. Because the patch is re-applied to whatever HEAD is at
# `stage` time, it still works if someone commits in between -- which is the normal case.
#
# Snapshot LATE and stage PROMPTLY. The snapshot is the only record of what the file looked like
# before you touched it, so anything that appears between `snapshot` and `stage` is attributed to
# you. A change another session *committed* in that window is handled correctly (it is on both
# sides of the merge and becomes a no-op); an *uncommitted* one from a session that edited after
# your snapshot is the one case this cannot tell from your own work, so `stage` prints the hunk
# headers it staged for you to glance at.
#
# If the merge conflicts, someone changed the same lines you did. Nothing is staged; read their
# version and redo your edit on top rather than forcing it.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
SESSION_ID=${STAGE_OWN_HUNK_ID:-default}
SNAPDIR="$(git rev-parse --git-dir)/stage-own-hunk/$SESSION_ID"
cd "$ROOT"

usage() {
    sed -n '2,41p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

snap_path() { printf '%s/%s.before' "$SNAPDIR" "${1//\//%2F}"; }

cmd_snapshot() {
    mkdir -p "$SNAPDIR"
    for f in "$@"; do
        [ -f "$f" ] || { echo "stage_own_hunk: no such file: $f" >&2; exit 2; }
        cp -- "$f" "$(snap_path "$f")"
        echo "snapshot $f"
    done
}

cmd_stage() {
    local rc=0
    for f in "$@"; do
        local before base mine head merged sha conflicts
        before=$(snap_path "$f")
        if [ ! -f "$before" ]; then
            echo "stage_own_hunk: no snapshot for $f -- run 'snapshot $f' before editing it" >&2
            rc=2
            continue
        fi
        if ! git ls-tree --name-only HEAD -- "$f" | grep -qx -- "$f"; then
            echo "stage_own_hunk: $f is not in HEAD, so all of it is yours -- use: git add $f" >&2
            rc=2
            continue
        fi
        if cmp -s -- "$before" "$f"; then
            echo "unchanged since the snapshot, nothing staged: $f"
            continue
        fi
        base=$(mktemp); mine=$(mktemp); head=$(mktemp); merged=$(mktemp)
        cp -- "$before" "$base"
        cp -- "$f" "$mine"
        git show "HEAD:$f" > "$head"
        # Three-way merge: base = your snapshot, "mine" = the worktree, "other" = HEAD. The result
        # is HEAD plus your edit. Anything another session COMMITTED since your snapshot is on both
        # sides and merges to a no-op, so it does not reappear in your commit -- which is why this
        # is a merge and not `git apply`, whose patch would simply fail against the newer HEAD.
        conflicts=0
        git merge-file -q -p -- "$mine" "$base" "$head" > "$merged" || conflicts=$?
        if [ "$conflicts" -ne 0 ]; then
            echo "stage_own_hunk: $f -- $conflicts conflict(s): another session changed the same" >&2
            echo "                lines you did. Read their version and redo your edit on top." >&2
            rm -f "$base" "$mine" "$head" "$merged"
            rc=1
            continue
        fi
        sha=$(git hash-object -w -- "$merged")
        git update-index --cacheinfo "100644,$sha,$f"
        echo "staged your hunks only: $f"
        git diff --cached --stat -- "$f" | sed 's/^/    /'
        # Every hunk below is attributed to you. If one of them is somebody else's uncommitted
        # edit, your snapshot was taken too early -- see the note at the top of this file.
        git diff --cached -U0 -- "$f" | { grep '^@@' || true; } | sed 's/^/    /'
        rm -f "$base" "$mine" "$head" "$merged"
    done
    return $rc
}

cmd_check() {
    # Pre-commit guard: refuse a commit whose staged content for a shared file could not have come
    # from `stage` -- the signature of a plain `git add` clobbering a hunk this session did not
    # author. Read-only: never touches the index or the worktree, so it is safe to run on every
    # commit, not just ones that used this script.
    local rc=0
    for f in "$@"; do
        git diff --cached --quiet -- "$f" && continue  # nothing staged for this file, nothing to guard
        git ls-tree --name-only HEAD -- "$f" | grep -qx -- "$f" || continue  # new file, no HEAD to clobber

        local before mine head merged conflicts
        before=$(snap_path "$f")
        if [ ! -f "$before" ]; then
            echo "stage_own_hunk check: $f is staged with no snapshot for session '$SESSION_ID'." >&2
            echo "  Run 'scripts/stage_own_hunk.sh snapshot $f' before editing shared files, then" >&2
            echo "  'scripts/stage_own_hunk.sh stage $f' before committing -- see TECHNICAL_REPORT.md Contributing." >&2
            rc=1
            continue
        fi

        mine=$(mktemp); head=$(mktemp); merged=$(mktemp)
        git show ":$f" > "$mine"
        git show "HEAD:$f" > "$head"
        conflicts=0
        git merge-file -q -p -- "$mine" "$before" "$head" > "$merged" || conflicts=$?
        if [ "$conflicts" -ne 0 ] || ! cmp -s -- "$mine" "$merged"; then
            echo "stage_own_hunk check: $f's staged content is not the sanctioned merge of your" >&2
            echo "  snapshot onto current HEAD -- a plain 'git add $f' would clobber a change" >&2
            echo "  another session committed since your snapshot. Re-run:" >&2
            echo "  'scripts/stage_own_hunk.sh stage $f' and commit what it stages instead." >&2
            rc=1
        fi
        rm -f "$mine" "$head" "$merged"
    done
    return $rc
}

cmd_status() {
    echo "session id: $SESSION_ID (set STAGE_OWN_HUNK_ID to change it)"
    [ -d "$SNAPDIR" ] || { echo "no snapshots"; return 0; }
    local any=0
    for s in "$SNAPDIR"/*.before; do
        [ -e "$s" ] || continue
        any=1
        local name=${s##*/}
        name=${name%.before}
        printf '%s\n' "${name//%2F//}"
    done
    [ "$any" = 1 ] || echo "no snapshots"
}

case "${1:-}" in
    snapshot) shift; [ $# -gt 0 ] || usage 2; cmd_snapshot "$@" ;;
    stage)    shift; [ $# -gt 0 ] || usage 2; cmd_stage "$@" ;;
    check)    shift; [ $# -gt 0 ] || usage 2; cmd_check "$@" ;;
    status)   cmd_status ;;
    -h|--help|"") usage ;;
    *) echo "stage_own_hunk: unknown command '$1'" >&2; usage 2 ;;
esac
