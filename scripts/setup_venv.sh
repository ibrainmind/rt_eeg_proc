#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN not found. Set PYTHON_BIN to a valid interpreter." >&2
  exit 1
fi

echo "Using Python: $($PYTHON_BIN --version 2>&1)"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists at: $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
  echo "Installing dependencies from requirements.txt"
  python -m pip install -r "$ROOT_DIR/requirements.txt"
else
  echo "No requirements.txt found; skipping dependency install."
fi

cat <<EOF

Virtual environment is ready.

Activate in new shells with:
  source .venv/bin/activate

Deactivate with:
  deactivate

EOF
