#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "[run_app] .venv not found. Create it first:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -r backend/requirements.txt"
  exit 1
fi

exec "$PY" "$ROOT/scripts/run_app.py" "$@"
