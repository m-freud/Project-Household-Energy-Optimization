from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
import sys

repo_root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").exists()), None)
if repo_root is None:
    raise RuntimeError("Could not find repository root containing 'src'")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import scenarios as scenario_catalog
from src.simulation.simulation import Simulation, make_mpc_controller


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_db = root / "sqlite" / "energy.db"
    backup_dir = root / "sqlite" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_db = backup_dir / f"mpc_oracle_baseline_{timestamp}.db"
    shutil.copy2(source_db, new_db)

    conn = sqlite3.connect(new_db)
    cur = conn.cursor()

    # Keep structural/input tables. Clear old simulation outputs.
    tables = [row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    for table in tables:
        cols = [row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()]
        if "policy" in cols:
            cur.execute(f"DELETE FROM {table}")
    conn.commit()

    # Force this run to write/read from the new baseline DB.
    Config.SQLITE_PATH = new_db

    sim = Simulation(conn, ensure_schema=False)
    households = list(range(1, 51))
    controller_factory = make_mpc_controller("mpc_oracle", horizon=96, duration_hours=sim.duration_hours)
    run_contexts = [
        RunContext(
            controller_factory=controller_factory,
            controller_name="mpc_oracle",
            scenario=scenario,
            run_id="1",
            start_time=1,
        )
        for scenario in scenario_catalog
    ]

    print(f"baseline_db={new_db}")
    print(f"scenarios={len(run_contexts)} households={len(households)} horizon=96")
    print("Starting oracle baseline run...")

    for idx, run_context in enumerate(run_contexts, start=1):
        print(f"[{idx}/{len(run_contexts)}] scenario={run_context.scenario.name} ...", flush=True)
        sim.run_all_households(
            run_context,
            household_ids=households,
            parallel_households=True,
            parallel_workers=6,
        )
        scenario_rows = cur.execute(
            "SELECT COUNT(*) FROM results WHERE policy='mpc_oracle' AND scenario=?",
            (run_context.scenario.name,),
        ).fetchone()[0]
        print(f"    done scenario={run_context.scenario.name}, rows={scenario_rows}", flush=True)

    summary = cur.execute(
        "SELECT COUNT(*), COUNT(DISTINCT scenario), COUNT(DISTINCT player_id), COUNT(DISTINCT run_id) FROM results WHERE policy='mpc_oracle'"
    ).fetchone()

    print("Finished.")
    print("oracle_rows=", summary[0], "scenarios=", summary[1], "households=", summary[2], "run_ids=", summary[3])
    print(f"output_db={new_db}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
