#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="5678"
WAIT_FOR_CLIENT="true"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--port)
			PORT="$2"
			shift 2
			;;
		--no-wait)
			WAIT_FOR_CLIENT="false"
			shift
			;;
		-h|--help)
			echo "Usage: $0 [--port <port>] [--no-wait]"
			echo ""
			echo "Starts Streamlit dashboard under debugpy."
			echo "Default: waits for debugger attach on port 5678."
			exit 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			exit 1
			;;
	esac
done

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "Could not find executable Python at: $PYTHON_BIN" >&2
	echo "Activate or create your venv first." >&2
	exit 1
fi

if "$WAIT_FOR_CLIENT"; then
	echo "Starting dashboard with debugpy on port $PORT (waiting for debugger attach)..."
	exec "$PYTHON_BIN" -m debugpy --listen "$PORT" --wait-for-client -m streamlit run src/dashboard_2/dashboard_2.py
else
	echo "Starting dashboard with debugpy on port $PORT (not waiting for debugger attach)..."
	exec "$PYTHON_BIN" -m debugpy --listen "$PORT" -m streamlit run src/dashboard_2/dashboard_2.py
fi
