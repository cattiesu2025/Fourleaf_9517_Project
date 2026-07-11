#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and rerun." >&2
  exit 1
fi

uv python install 3.11
uv venv --python 3.11 "$ROOT_DIR/.venv"
uv pip install --python "$ROOT_DIR/.venv/bin/python" -r "$ROOT_DIR/requirements-dev.txt"
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/check_environment.py"
