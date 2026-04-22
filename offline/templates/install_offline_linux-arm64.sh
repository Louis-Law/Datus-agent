#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_OS="Linux"
EXPECTED_ARCHES="aarch64 arm64"

usage() {
  cat <<'USAGE'
Usage:
  install_offline.sh                 Install into $BUNDLE_DIR/.venv (default)
  install_offline.sh /path/to/venv   Install into the given virtualenv
  install_offline.sh --user          Install into the current user's site-packages (pip --user)
  install_offline.sh --system        Install into the system Python (pip --break-system-packages)
  install_offline.sh --help          Show this help

Modes:
  venv     Create (if missing) and populate an isolated Python 3.12 virtualenv. Safest.
  --user   Install to ~/.local/lib/python3.12/site-packages. No sudo. Add ~/.local/bin to PATH.
  --system Install into the global Python. Conflicts with distro packages on PEP 668 systems;
           may require sudo. Hardest to uninstall cleanly.
USAGE
}

MODE=venv
VENV_DIR="$BUNDLE_DIR/.venv"

if [[ $# -gt 1 ]]; then
  echo "Too many arguments." >&2
  usage >&2
  exit 2
elif [[ $# -eq 1 ]]; then
  case "$1" in
    --user)    MODE=user ;;
    --system)  MODE=system ;;
    --help|-h) usage; exit 0 ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      VENV_DIR="$1"
      ;;
  esac
fi

if [[ "$(uname -s)" != "$EXPECTED_OS" ]]; then
  echo "This bundle only supports $EXPECTED_OS." >&2
  exit 1
fi

CURRENT_ARCH="$(uname -m)"
case "$CURRENT_ARCH" in
  aarch64|arm64) ;;
  *)
    echo "This bundle does not support architecture: $CURRENT_ARCH" >&2
    echo "Expected one of: $EXPECTED_ARCHES" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Python 3.12 is required but was not found." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" - <<'PY'
import sys

required = tuple(int(part) for part in "3.12".split("."))
current = sys.version_info[:2]
if current != required:
    raise SystemExit(
        f"Python {required[0]}.{required[1]} is required, but found {current[0]}.{current[1]}."
    )
PY

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
      "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    INSTALL_PYTHON="$VENV_DIR/bin/python"
    ;;
  user)
    INSTALL_PYTHON="$PYTHON_BIN"
    PIP_CMD+=(--user)
    ;;
  system)
    INSTALL_PYTHON="$PYTHON_BIN"
    PIP_CMD+=(--break-system-packages)
    ;;
esac

"$INSTALL_PYTHON" "${PIP_CMD[@]}"
"$INSTALL_PYTHON" -m pip check

case "$MODE" in
  venv)
    cat <<EOF
Offline installation completed.
Virtualenv: $VENV_DIR
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
And persist it by appending the same line to ~/.bashrc (or your shell's rc file).
EOF
    ;;
  system)
    cat <<EOF
Offline installation completed (system-wide install).
Python: $INSTALL_PYTHON
EOF
    ;;
esac
