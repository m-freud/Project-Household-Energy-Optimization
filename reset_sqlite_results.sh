#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DB_PATH="$SCRIPT_DIR/sqlite/energy.db"
FULL_RESET="false"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--full)
			FULL_RESET="true"
			shift
			;;
		-h|--help)
			echo "Usage: $0 [--full]"
			echo ""
			echo "Clears simulation output tables."
			echo "--full: also run VACUUM to reclaim DB file space."
			exit 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			exit 1
			;;
	esac
done

if [[ ! -f "$DB_PATH" ]]; then
	echo "Database not found at: $DB_PATH" >&2
	exit 1
fi

TABLES=(
	results
	net_load
	net_cost
	total_consumption
	total_cost
	bess_soc
	bess_power
	ev1_soc
	ev1_power
	ev2_soc
	ev2_power
)

echo "Resetting simulation output tables in $DB_PATH"

if command -v sqlite3 >/dev/null 2>&1; then
	for table in "${TABLES[@]}"; do
		sqlite3 "$DB_PATH" "DELETE FROM $table WHERE 1;" 2>/dev/null || true
	done
	if [[ "$FULL_RESET" == "true" ]]; then
		sqlite3 "$DB_PATH" "VACUUM;" 2>/dev/null || true
	fi
else
	PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
	if [[ ! -x "$PYTHON_BIN" ]]; then
		echo "sqlite3 CLI not found and Python venv missing at $PYTHON_BIN" >&2
		echo "Install sqlite3 or create the venv, then retry." >&2
		exit 1
	fi

	export FULL_RESET

	"$PYTHON_BIN" - <<'PY'
import os
import sqlite3
from pathlib import Path

db_path = Path("sqlite/energy.db")
full_reset = os.environ.get("FULL_RESET", "false") == "true"

tables = [
    "results",
    "net_load",
    "net_cost",
    "total_consumption",
    "total_cost",
    "bess_soc",
    "bess_power",
    "ev1_soc",
    "ev1_power",
    "ev2_soc",
    "ev2_power",
]

with sqlite3.connect(db_path) as conn:
    for table in tables:
        try:
            conn.execute(f"DELETE FROM {table} WHERE 1")
        except sqlite3.OperationalError:
            pass
    conn.commit()

if full_reset:
	with sqlite3.connect(db_path, isolation_level=None) as conn:
		conn.execute("VACUUM")

print(f"Reset done: {db_path}")
PY
fi

if [[ "$FULL_RESET" == "true" ]]; then
	echo "Done. Simulation outputs cleared and DB vacuumed."
else
	echo "Done. Simulation outputs cleared."
fi
