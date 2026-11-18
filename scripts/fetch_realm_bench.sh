#!/usr/bin/env bash
# Fetch the parts of REALM-Bench (Geng & Chang, arXiv 2502.18836) this repo
# reuses: the P3 urban ride-sharing instances and the evaluation metrics.
#
# REALM-Bench ships no license file, so its code and data are fetched into
# third_party/ (gitignored) rather than copied into this repository. The
# commit is pinned so the instances cannot change under a result.
set -euo pipefail

REPO=https://github.com/genglongling/REALM-Bench.git
COMMIT=9c3aa2ae97d65198f6ee29fe942d99f9b3a9c6eb
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/REALM-Bench"

if [ -d "$DEST/.git" ] && [ "$(git -C "$DEST" rev-parse HEAD)" = "$COMMIT" ]; then
    echo "REALM-Bench already at $COMMIT in $DEST"
    exit 0
fi

rm -rf "$DEST"
mkdir -p "$DEST"
git -C "$DEST" init -q
git -C "$DEST" remote add origin "$REPO"
# the repository commits a full virtualenv (50k files); take only what we use
git -C "$DEST" sparse-checkout set --no-cone /datasets/P3/ /evaluation/
git -C "$DEST" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git -C "$DEST" checkout -q FETCH_HEAD
echo "REALM-Bench $COMMIT -> $DEST"
