#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:?Usage: $0 <linux-x86_64|linux-arm64> [builder-args...]}"
shift || true

case "$TARGET" in
  linux-x86_64)
    DOCKER_PLATFORM="linux/amd64"
    MANYLINUX_IMAGE="quay.io/pypa/manylinux2014_x86_64"
    ;;
  linux-arm64)
    DOCKER_PLATFORM="linux/arm64"
    MANYLINUX_IMAGE="quay.io/pypa/manylinux2014_aarch64"
    ;;
  *)
    echo "Unsupported target: $TARGET" >&2
    echo "Supported targets: linux-x86_64, linux-arm64" >&2
    exit 1
    ;;
esac

DOCKER_ENV_ARGS=()
for env_name in \
  PIP_INDEX_URL \
  PIP_EXTRA_INDEX_URL \
  PIP_TRUSTED_HOST \
  HTTP_PROXY \
  HTTPS_PROXY \
  NO_PROXY \
  ALL_PROXY; do
  if [[ -n "${!env_name:-}" ]]; then
    DOCKER_ENV_ARGS+=(-e "$env_name=${!env_name}")
  fi
done

USER_ARGS=()
if command -v id >/dev/null 2>&1; then
  USER_ARGS=(--user "$(id -u):$(id -g)")
fi

mkdir -p "$ROOT_DIR/offline/dist"

DOCKER_CMD=(docker run --rm --platform "$DOCKER_PLATFORM")

if ((${#USER_ARGS[@]})); then
  DOCKER_CMD+=("${USER_ARGS[@]}")
fi

DOCKER_CMD+=(-e HOME=/tmp/datus-offline-builder)

if ((${#DOCKER_ENV_ARGS[@]})); then
  DOCKER_CMD+=("${DOCKER_ENV_ARGS[@]}")
fi

DOCKER_CMD+=(
  -v "$ROOT_DIR:/workspace"
  -w /workspace
  "$MANYLINUX_IMAGE"
  bash /workspace/offline/docker/build_bundle_in_container.sh "$TARGET" "$@"
)

exec "${DOCKER_CMD[@]}"
