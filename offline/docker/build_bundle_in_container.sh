#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace"
TARGET="${1:?Usage: $0 <linux-x86_64|linux-arm64> [builder-args...]}"
shift || true

PYTHON_BIN="${PYTHON_BIN:-/opt/python/cp312-cp312/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.12 interpreter not found in container: $PYTHON_BIN" >&2
  exit 1
fi

export HOME="${HOME:-/tmp/datus-offline-builder}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$HOME"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/build_offline_bundle.py" --target "$TARGET" "$@"
