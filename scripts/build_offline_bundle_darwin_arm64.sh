#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# macOS bundles must be built natively — Docker on macOS runs Linux containers,
# so it cannot produce macosx_* wheels or a Darwin PBS runtime.
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "darwin-arm64 bundle must be built on macOS." >&2
  echo "Found: $(uname -s)" >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "darwin-arm64 bundle must be built on Apple Silicon (arm64)." >&2
  echo "Found: $(uname -m). Use a native arm64 shell, not an x86_64 Rosetta shell." >&2
  exit 1
fi

# Resolve a Python 3.12 with the build dependencies installed
# (packaging + huggingface_hub + pip). Priority:
#   1. OFFLINE_BUNDLE_PIP_PYTHON env override
#   2. Repo's own .venv (created by `uv sync`)
#   3. python3.12 on PATH (Homebrew etc.)
PYTHON_BIN="${OFFLINE_BUNDLE_PIP_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  else
    echo "Need Python 3.12 with pip + packaging + huggingface_hub to build the bundle." >&2
    echo "Run 'uv sync' in the repo root first, or set OFFLINE_BUNDLE_PIP_PYTHON to a suitable interpreter." >&2
    exit 1
  fi
fi

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/build_offline_bundle.py" --target darwin-arm64 "$@"
