#!/usr/bin/env bash
# Sync tinygrad to remote host and optionally run a command there.
#
# Usage:
#   ./sync.sh                       # sync only
#   ./sync.sh <cmd> [args...]       # sync, then run `<cmd> [args...]` on remote in /root/tinygrad
#
# Env overrides:
#   REMOTE=root@192.168.1.107
#   REMOTE_DIR=/root/tinygrad

set -euo pipefail

REMOTE="${REMOTE:-root@192.168.1.107}"
REMOTE_DIR="${REMOTE_DIR:-/root/tinygrad}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'"

rsync -azP --delete \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.venv/' \
  --exclude='.venv-*/' \
  --exclude='*.pyc' \
  --exclude='*.so' \
  --exclude='*.egg-info/' \
  --exclude='build/' \
  --exclude='dist/' \
  --exclude='.idea/' \
  --exclude='.vscode/' \
  --exclude='.mypy_cache/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='*.prof' \
  --exclude='extra/datasets/cifar-10-python.tar.gz' \
  --exclude='extra/datasets/librispeech/' \
  --exclude='extra/datasets/imagenet/' \
  --exclude='extra/datasets/wiki/' \
  --exclude='extra/datasets/kits19/' \
  "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"

if [ "$#" -gt 0 ]; then
  echo "--- running on $REMOTE: $* ---"
  ssh -t "$REMOTE" "cd '$REMOTE_DIR' && PYTHONPATH='$REMOTE_DIR' $*"
fi
