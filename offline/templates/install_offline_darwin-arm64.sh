#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_OS="Darwin"
EXPECTED_ARCHES="arm64"

BUNDLED_RUNTIME_DIR="$BUNDLE_DIR/python-runtime"
BUNDLED_PYTHON_HOME="$BUNDLE_DIR/python"
BUNDLED_PYTHON_BIN="$BUNDLED_PYTHON_HOME/bin/python3.12"

usage() {
  cat <<'USAGE'
Usage:
  install_offline.sh                 Install into $BUNDLE_DIR/.venv (default)
  install_offline.sh /path/to/venv   Install into the given virtualenv
  install_offline.sh --user          Install into the current user's site-packages (pip --user)
  install_offline.sh --system        Install into the active Python (pip --break-system-packages)
  install_offline.sh --help          Show this help

Modes:
  venv     Create (if missing) and populate an isolated Python 3.12 virtualenv. Safest.
  --user   Install to ~/Library/Python/3.12/lib/python/site-packages (or equivalent).
           No sudo. Add the user-base bin dir to PATH.
           Requires a host python3.12 on PATH (the bundled runtime is venv-only).
  --system Install into the active Python (e.g. Homebrew's python3.12). On
           PEP 668 environments pip needs --break-system-packages. Requires a
           host python3.12 on PATH.

Flags:
  --skip-runtime-assets  Do not copy bundled fastembed/HF snapshots into the
                         user's cache. Datus will then try to download the
                         embedding model from huggingface.co on first use.

Environment:
  PYTHON_BIN    Override Python selection (e.g. PYTHON_BIN=/opt/homebrew/bin/python3.12).
                Falls back to 'python3.12' on PATH, then to the bundle's own runtime.
USAGE
}

MODE=venv
VENV_DIR="$BUNDLE_DIR/.venv"
SKIP_RUNTIME_ASSETS=0
POSITIONAL_VENV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)                MODE=user ;;
    --system)              MODE=system ;;
    --skip-runtime-assets) SKIP_RUNTIME_ASSETS=1 ;;
    --help|-h)             usage; exit 0 ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$POSITIONAL_VENV" ]]; then
        echo "Too many positional arguments." >&2
        usage >&2
        exit 2
      fi
      POSITIONAL_VENV="$1"
      ;;
  esac
  shift
done

if [[ -n "$POSITIONAL_VENV" ]]; then
  VENV_DIR="$POSITIONAL_VENV"
fi

if [[ "$(uname -s)" != "$EXPECTED_OS" ]]; then
  echo "This bundle only supports $EXPECTED_OS (macOS)." >&2
  exit 1
fi

CURRENT_ARCH="$(uname -m)"
case "$CURRENT_ARCH" in
  arm64) ;;
  *)
    echo "This bundle does not support architecture: $CURRENT_ARCH" >&2
    echo "Expected: $EXPECTED_ARCHES (Apple Silicon)" >&2
    exit 1
    ;;
esac

# macOS version gate: the wheels and PBS runtime in this bundle target
# macOS 11 (Big Sur) or newer.
MACOS_PRODUCT_VERSION="$(sw_vers -productVersion 2>/dev/null || echo 0)"
MACOS_MAJOR="${MACOS_PRODUCT_VERSION%%.*}"
if [[ -z "$MACOS_MAJOR" || "$MACOS_MAJOR" -lt 11 ]]; then
  echo "macOS 11 (Big Sur) or newer is required; found: $MACOS_PRODUCT_VERSION" >&2
  exit 1
fi

# Resolve Python: user override -> system python3.12 -> bundled PBS runtime.
USING_BUNDLED=0
RESOLVED_PYTHON=""

if [[ -n "${PYTHON_BIN:-}" ]]; then
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "PYTHON_BIN is set to '$PYTHON_BIN' but that command was not found." >&2
    exit 1
  fi
  RESOLVED_PYTHON="$PYTHON_BIN"
elif command -v python3.12 >/dev/null 2>&1; then
  RESOLVED_PYTHON="python3.12"
elif [[ -x "$BUNDLED_PYTHON_BIN" ]]; then
  RESOLVED_PYTHON="$BUNDLED_PYTHON_BIN"
  USING_BUNDLED=1
elif [[ -d "$BUNDLED_RUNTIME_DIR" ]]; then
  BUNDLED_TARBALL="$(ls "$BUNDLED_RUNTIME_DIR"/cpython-3.12*.tar.gz 2>/dev/null | head -1 || true)"
  if [[ -z "$BUNDLED_TARBALL" ]]; then
    echo "Bundle's python-runtime/ directory is present but contains no tarball." >&2
    exit 1
  fi
  echo "Extracting bundled Python runtime from $(basename "$BUNDLED_TARBALL") ..."
  tar -xzf "$BUNDLED_TARBALL" -C "$BUNDLE_DIR"
  if [[ ! -x "$BUNDLED_PYTHON_BIN" ]]; then
    echo "Extraction failed: $BUNDLED_PYTHON_BIN not found after untar." >&2
    exit 1
  fi
  RESOLVED_PYTHON="$BUNDLED_PYTHON_BIN"
  USING_BUNDLED=1
else
  echo "Python 3.12 is required but was not found, and this bundle does not ship one." >&2
  echo "Install python3.12 (e.g. 'brew install python@3.12') or set PYTHON_BIN=/path/to/python3.12." >&2
  exit 1
fi

"$RESOLVED_PYTHON" - <<'PY'
import sys

required = tuple(int(part) for part in "3.12".split("."))
current = sys.version_info[:2]
if current != required:
    raise SystemExit(
        f"Python {required[0]}.{required[1]} is required, but found {current[0]}.{current[1]}."
    )
PY

if [[ "$USING_BUNDLED" -eq 1 && "$MODE" != "venv" ]]; then
  echo "The bundled Python runtime can only host 'venv' mode installs." >&2
  echo "For --user or --system, install Python 3.12 on the host first" >&2
  echo "(e.g. 'brew install python@3.12'), then rerun." >&2
  exit 2
fi

PIP_CMD=(
  -m pip install
  --no-index
  --no-warn-script-location
  --find-links "$BUNDLE_DIR/wheelhouse"
  -r "$BUNDLE_DIR/requirements.lock"
)

case "$MODE" in
  venv)
    if [[ ! -d "$VENV_DIR" ]]; then
      "$RESOLVED_PYTHON" -m venv "$VENV_DIR"
    fi
    INSTALL_PYTHON="$VENV_DIR/bin/python"
    ;;
  user)
    INSTALL_PYTHON="$RESOLVED_PYTHON"
    PIP_CMD+=(--user)
    ;;
  system)
    INSTALL_PYTHON="$RESOLVED_PYTHON"
    # Homebrew's python3.12 marks site-packages as externally-managed (PEP 668).
    # --break-system-packages is required; --ignore-installed mirrors the Linux
    # template so we overlay our pinned wheels without trying to uninstall
    # whatever the host package manager (brew/conda) put in place.
    PIP_CMD+=(--break-system-packages --ignore-installed)
    ;;
esac

"$INSTALL_PYTHON" "${PIP_CMD[@]}"
"$INSTALL_PYTHON" -m pip check

# Mirror bundled runtime assets (fastembed/HF snapshots) into the user's
# fastembed cache so the first KB embedding op does not reach huggingface.co.
# Datus's `_resolve_cache_dir()` looks at $FASTEMBED_CACHE_PATH first, then
# $HF_HOME/fastembed, then ~/.cache/huggingface/fastembed; we install to the
# last (default) location and let users override via env var if they want.
RUNTIME_ASSETS_DIR="$BUNDLE_DIR/assets/fastembed-cache"
RUNTIME_ASSETS_INSTALLED=""
if [[ "$SKIP_RUNTIME_ASSETS" -eq 0 && -d "$RUNTIME_ASSETS_DIR" ]]; then
  # When invoked with sudo, prefer the invoking user's HOME so the cache is
  # actually visible to the user who runs Datus afterwards. macOS lacks
  # `getent`; use `dscl` to resolve the home directory.
  TARGET_HOME="$HOME"
  if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    SUDO_HOME="$(dscl . -read "/Users/$SUDO_USER" NFSHomeDirectory 2>/dev/null | awk '{print $2}' || true)"
    if [[ -n "$SUDO_HOME" && -d "$SUDO_HOME" ]]; then
      TARGET_HOME="$SUDO_HOME"
    fi
  fi
  FASTEMBED_DEST="${FASTEMBED_CACHE_PATH:-${HF_HOME:-$TARGET_HOME/.cache/huggingface}/fastembed}"
  mkdir -p "$FASTEMBED_DEST"
  # `cp -Rn` is "no clobber" on BSD cp (macOS): existing snapshots are kept,
  # we only fill gaps. The `/.` suffix copies the directory's contents into the
  # destination rather than nesting the directory itself — same trick as the
  # Linux template.
  cp -Rn "$RUNTIME_ASSETS_DIR"/. "$FASTEMBED_DEST"/
  if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" && "$TARGET_HOME" != "$HOME" ]]; then
    chown -R "$SUDO_USER" "$FASTEMBED_DEST" 2>/dev/null || true
  fi
  RUNTIME_ASSETS_INSTALLED="$FASTEMBED_DEST"
fi

case "$MODE" in
  venv)
    cat <<EOF
Offline installation completed.
Python used: $RESOLVED_PYTHON$( [[ $USING_BUNDLED -eq 1 ]] && echo "  (bundled)" )
Virtualenv:  $VENV_DIR
Activate with:
  source "$VENV_DIR/bin/activate"
EOF
    ;;
  user)
    USER_BIN="$("$INSTALL_PYTHON" -c 'import site, os; print(os.path.join(site.getuserbase(), "bin"))')"
    cat <<EOF
Offline installation completed (user-level install).
Python:  $INSTALL_PYTHON
Scripts: $USER_BIN
If that directory is not on your PATH, add it now:
  export PATH="$USER_BIN:\$PATH"
And persist it by appending the same line to ~/.zshrc (or your shell's rc file).
EOF
    ;;
  system)
    cat <<EOF
Offline installation completed (system-wide install).
Python: $INSTALL_PYTHON
EOF
    ;;
esac

if [[ -n "$RUNTIME_ASSETS_INSTALLED" ]]; then
  cat <<EOF
Runtime assets:
  fastembed snapshot mirrored to $RUNTIME_ASSETS_INSTALLED
  Override with FASTEMBED_CACHE_PATH=$RUNTIME_ASSETS_DIR if you want Datus
  to read directly from the bundle instead of the user cache.
EOF
elif [[ "$SKIP_RUNTIME_ASSETS" -eq 1 ]]; then
  echo "Runtime assets: skipped (--skip-runtime-assets); first KB embedding op will reach huggingface.co."
elif [[ ! -d "$RUNTIME_ASSETS_DIR" ]]; then
  echo "Runtime assets: bundle has no assets/fastembed-cache; first KB embedding op will reach huggingface.co."
fi
