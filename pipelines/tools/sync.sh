#!/usr/bin/env bash
# sync.sh <ssh-host> <remote-root> [model-dir-name ...]
# Copies code + data packages (not raw data, caches, runs, secrets) to the remote root, then the
# named model directories (resumable). Example:
#   pipelines/tools/sync.sh popos-6000b /home/isaac/workdir/CV/vlm_prep Qwen3-VL-32B-Instruct
set -euo pipefail
HOST="$1"; DEST="$2"; shift 2
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "$DEST" in *Dropbox*) echo "refusing to sync into Dropbox: $DEST" >&2; exit 1;; esac
ssh "$HOST" "mkdir -p '$DEST/models'"
rsync -a --partial \
  --exclude '.DS_Store' --exclude '__pycache__' --exclude '.env' --exclude '.cache/' \
  --exclude 'data/raw/' --exclude 'data/preview/' --exclude 'data/formats/' \
  --exclude 'runs/' --exclude 'models/' \
  "$SRC/" "$HOST:$DEST/"
for model in "$@"; do
  # --partial-dir: an interrupted transfer never leaves a truncated shard under its final name
  rsync -a --partial-dir=.rsync-partial --exclude '.cache/' --exclude '.DS_Store' "$SRC/models/$model" "$HOST:$DEST/models/"
done
echo "synced -> $HOST:$DEST"
