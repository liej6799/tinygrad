#!/usr/bin/env bash
# Sync then run examples/matmul.py on the remote with DEBUG=6 DEV=CPU
# inside the remote's .venv.
#
# Env overrides:
#   REMOTE=root@192.168.1.107
#   REMOTE_DIR=/root/tinygrad
#   VENV=.venv

set -euo pipefail

REMOTE="${REMOTE:-root@192.168.1.107}"
REMOTE_DIR="${REMOTE_DIR:-/root/tinygrad}"
VENV="${VENV:-.venv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/sync.sh"

ssh -t "$REMOTE" "cd '$REMOTE_DIR' && source '$VENV/bin/activate' && DEBUG=6 DEV=CPU python3 examples/matmul.py"
