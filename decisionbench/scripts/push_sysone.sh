#!/usr/bin/env bash
# Publish this repo's commits into saikrishnarallabandi/sys-one-tool-router under decisionbench/ (git subtree).
#
#   scripts/push_sysone.sh [branch]      # default branch: add-decisionbench (open PR #1); use "main" once merged
#
# Commit locally first; only committed history on the current local branch is pushed.
set -euo pipefail

BRANCH="${1:-add-decisionbench}"
REMOTE_URL="https://github.com/saikrishnarallabandi/sys-one-tool-router.git"
PREFIX="decisionbench"
ROOT="$(git rev-parse --show-toplevel)"
CLONE="$ROOT/.cache/sysone"
LOCAL_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"

if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
  echo "warning: uncommitted changes in $ROOT are not pushed" >&2
fi

[ -d "$CLONE/.git" ] || git clone -q "$REMOTE_URL" "$CLONE"
git -C "$CLONE" fetch -q origin
if git -C "$CLONE" show-ref -q --verify "refs/remotes/origin/$BRANCH"; then
  git -C "$CLONE" checkout -q -B "$BRANCH" "origin/$BRANCH"
else
  git -C "$CLONE" checkout -q -B "$BRANCH" origin/main
fi

if [ -d "$CLONE/$PREFIX" ]; then
  git -C "$CLONE" subtree pull -q --prefix="$PREFIX" "$ROOT" "$LOCAL_BRANCH" -m "Update $PREFIX/ from project_decisionmodels"
else
  git -C "$CLONE" subtree add -q --prefix="$PREFIX" "$ROOT" "$LOCAL_BRANCH"
fi
git -C "$CLONE" push -q origin "$BRANCH"
echo "pushed $LOCAL_BRANCH -> sys-one-tool-router:$BRANCH ($PREFIX/)"
