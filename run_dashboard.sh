#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "Could not find executable Python at: $PYTHON_BIN" >&2
	echo "Activate or create your venv first." >&2
	exit 1
fi

echo "Starting dashboard..."
exec "$PYTHON_BIN" -m streamlit run src/dashboard/dashboard.py
