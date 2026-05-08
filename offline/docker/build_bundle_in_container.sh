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

# `huggingface_hub` is needed at build time to pre-download the fastembed
# embedding model snapshot into `assets/fastembed-cache/`. The upper bound
# avoids 0.32+ which makes `hf_xet` a hard runtime dep — `hf_xet` ships only
# `manylinux_2_28_aarch64` wheels, but this container is `manylinux2014`
# (glibc 2.17), so pip would fall back to a from-source Rust build that has
# no toolchain available. We do not need Xet protocol support here.
# The bundle builder itself does not redistribute this dependency: it only
# uses `snapshot_download` to populate the cache directory.
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel "huggingface_hub>=0.20,<0.32"

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/build_offline_bundle.py" --target "$TARGET" "$@"
